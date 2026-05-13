"""Order manager + fill logic.

Per the spec:
  - Up to two resting orders at any time: one bid, one ask.
  - No partial fills, our queue volume is ignored.
  - A resting order fills in two situations:
      (1) opposite-side best of the LOB reaches our price (or better);
      (2) a trade fully eats the level our order sits on (after subtracting
          trade.amount from the level volume the result is <= 0). The lob
          module reports such consumed levels via TradeDelta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .lob import LimitOrderBook, TradeDelta


@dataclass
class Order:
    side: str            # 'bid' or 'ask'
    price: float
    qty: float
    placed_ts: int


@dataclass
class Fill:
    ts: int
    side: str            # 'bid' (we bought) or 'ask' (we sold)
    price: float
    qty: float
    liquidity: str       # 'maker' or 'taker'


class OrderManager:
    def __init__(self) -> None:
        self.bid: Optional[Order] = None
        self.ask: Optional[Order] = None

    # ----- placement -----
    def place(self, side: str, price: float, qty: float, ts: int) -> None:
        order = Order(side=side, price=price, qty=qty, placed_ts=ts)
        if side == "bid":
            self.bid = order
        elif side == "ask":
            self.ask = order
        else:
            raise ValueError(f"bad side: {side}")

    def cancel(self, side: str) -> None:
        if side == "bid":
            self.bid = None
        elif side == "ask":
            self.ask = None
        else:
            raise ValueError(f"bad side: {side}")

    # ----- matching -----
    def match_on_snap(self, book: LimitOrderBook, ts: int) -> list[Fill]:
        """Check fills triggered by an LOB snapshot (opposite-side best reached us)."""
        fills: list[Fill] = []
        if self.bid is not None and book.best_ask <= self.bid.price:
            fills.append(Fill(ts=ts, side="bid", price=self.bid.price,
                              qty=self.bid.qty, liquidity="maker"))
            self.bid = None
        if self.ask is not None and book.best_bid >= self.ask.price:
            fills.append(Fill(ts=ts, side="ask", price=self.ask.price,
                              qty=self.ask.qty, liquidity="maker"))
            self.ask = None
        return fills

    def match_on_trade(self, delta: TradeDelta, ts: int) -> list[Fill]:
        """Check fills triggered by a trade that consumed levels."""
        fills: list[Fill] = []
        if self.bid is not None:
            for lvl in delta.consumed_bid_levels:
                if lvl == self.bid.price:
                    fills.append(Fill(ts=ts, side="bid", price=self.bid.price,
                                      qty=self.bid.qty, liquidity="maker"))
                    self.bid = None
                    break
        if self.ask is not None:
            for lvl in delta.consumed_ask_levels:
                if lvl == self.ask.price:
                    fills.append(Fill(ts=ts, side="ask", price=self.ask.price,
                                      qty=self.ask.qty, liquidity="maker"))
                    self.ask = None
                    break
        return fills
