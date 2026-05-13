"""Event-driven backtest engine.

Pipeline per event:
  1. drain pending strategy actions whose ready_ts <= event.ts
     (place / cancel are applied to the order manager only after latency)
  2. dispatch the event:
       - SnapEvent: lob.apply_snapshot, then check fills against the new
         opposite-side best (snapshot-driven matching), then ask the strategy
         for new quotes and schedule cancel(old)+place(new) at ts + latency.
       - TradeEvent: lob.apply_trade returns the consumed levels; check fills
         against them.
  3. record metrics (mid + inventory) on every snapshot.

PnL is mark-to-market: cash + inventory * last_mid.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Optional

from .data_loader import SnapEvent, TradeEvent
from .lob import LimitOrderBook
from .metrics import Metrics
from .orders import OrderManager
from strategies.base import Quotes, Strategy


@dataclass(order=True)
class _Action:
    ready_ts: int
    seq: int
    kind: str = ""              # 'place_bid','place_ask','cancel_bid','cancel_ask'
    price: float = 0.0
    qty: float = 0.0


class Engine:
    def __init__(self, *, strategy: Strategy, latency_us: int,
                 metrics: Metrics, progress_every: int = 500_000) -> None:
        self.strategy = strategy
        self.latency_us = int(latency_us)
        self.metrics = metrics
        self.book = LimitOrderBook()
        self.om = OrderManager()
        self._action_queue: list[_Action] = []
        self._action_seq = 0
        self._desired: Optional[Quotes] = None
        self._progress_every = int(progress_every)

    # ----- helpers -----
    def _schedule(self, ready_ts: int, kind: str, price: float = 0.0,
                  qty: float = 0.0) -> None:
        self._action_seq += 1
        heapq.heappush(self._action_queue,
                       _Action(ready_ts=ready_ts, seq=self._action_seq,
                               kind=kind, price=price, qty=qty))

    def _drain_actions(self, up_to_ts: int) -> None:
        while self._action_queue and self._action_queue[0].ready_ts <= up_to_ts:
            a = heapq.heappop(self._action_queue)
            if a.kind == "cancel_bid":
                self.om.cancel("bid")
            elif a.kind == "cancel_ask":
                self.om.cancel("ask")
            elif a.kind == "place_bid":
                self.om.place("bid", a.price, a.qty, a.ready_ts)
            elif a.kind == "place_ask":
                self.om.place("ask", a.price, a.qty, a.ready_ts)

    def _record_fills(self, fills) -> None:
        for f in fills:
            self.metrics.on_fill(ts=f.ts, side=f.side, price=f.price,
                                 qty=f.qty, liquidity=f.liquidity)

    def _update_desired(self, ts: int, new_q: Optional[Quotes]) -> None:
        """Schedule cancel+place commands so the OM mirrors new_q after latency."""
        if new_q is None:
            return
        ready_ts = ts + self.latency_us
        cur = self._desired
        # Bid side:
        cur_bp = cur.bid_price if cur else None
        cur_size = cur.size if cur else 0.0
        if new_q.bid_price != cur_bp or new_q.size != cur_size:
            if cur_bp is not None:
                self._schedule(ready_ts, "cancel_bid")
            if new_q.bid_price is not None:
                self._schedule(ready_ts, "place_bid",
                               price=new_q.bid_price, qty=new_q.size)
        # Ask side:
        cur_ap = cur.ask_price if cur else None
        if new_q.ask_price != cur_ap or new_q.size != cur_size:
            if cur_ap is not None:
                self._schedule(ready_ts, "cancel_ask")
            if new_q.ask_price is not None:
                self._schedule(ready_ts, "place_ask",
                               price=new_q.ask_price, qty=new_q.size)
        self._desired = new_q

    # ----- main loop -----
    def run(self, events: Iterable[object]) -> None:
        n_events = 0
        last_log_pnl = 0.0
        for ev in events:
            self._drain_actions(ev.ts)
            if isinstance(ev, SnapEvent):
                self.book.apply_snapshot(ev.ask_p, ev.ask_a, ev.bid_p, ev.bid_a)
                fills = self.om.match_on_snap(self.book, ev.ts)
                self._record_fills(fills)
                mid = self.book.mid()
                if mid is not None:
                    self.metrics.on_mid(ev.ts, mid)
                    new_q = self.strategy.on_book(self.book, ev.ts,
                                                  inventory=self.metrics.inv)
                    self._update_desired(ev.ts, new_q)
            elif isinstance(ev, TradeEvent):
                delta = self.book.apply_trade(ev.side, ev.price, ev.amount)
                fills = self.om.match_on_trade(delta, ev.ts)
                self._record_fills(fills)
            n_events += 1
            if self._progress_every and n_events % self._progress_every == 0:
                pnl = self.metrics.total_pnl()
                d_pnl = pnl - last_log_pnl
                last_log_pnl = pnl
                print(f"  [progress] events={n_events:>10}  "
                      f"ts={ev.ts}  pnl={pnl:+.4f}  d={d_pnl:+.4f}  "
                      f"inv={self.metrics.inv:+.0f}  "
                      f"fills={self.metrics.n_fills_bid + self.metrics.n_fills_ask}")
        # Drain any remaining actions (no further matching can happen).
        self._drain_actions(10**18)
