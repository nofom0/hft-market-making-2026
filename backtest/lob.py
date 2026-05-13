"""Limit order book maintained from snapshots and patched by trades.

Asks are stored sorted ascending (asks[0] = best ask, lowest price).
Bids are stored sorted descending (bids[0] = best bid, highest price).

apply_trade implements the patching rules agreed in the spec:
  - aggressive buy at price > best_ask: sweep all ask levels with price <= trade_price,
    then subtract trade.amount from the (new) best_ask. If after subtraction
    the level is <= 0, drop it.
  - aggressive sell at price < best_bid: symmetric for bids (sweep >= trade_price,
    subtract from new best_bid).
  - regular trades (buy at <= best_ask / sell at >= best_bid): only subtract from
    the touched level (best on the corresponding side).

The method returns "consumed levels" — prices of levels that were exhausted
by the trade. The order manager uses this to decide whether our resting order
sitting on that price gets filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TradeDelta:
    consumed_ask_levels: list[float] = field(default_factory=list)
    consumed_bid_levels: list[float] = field(default_factory=list)


class LimitOrderBook:
    __slots__ = ("ask_p", "ask_a", "bid_p", "bid_a", "_n_ask", "_n_bid", "_cap")

    def __init__(self, capacity: int = 25) -> None:
        self._cap = capacity
        self.ask_p = np.zeros(capacity, dtype=np.float64)
        self.ask_a = np.zeros(capacity, dtype=np.float64)
        self.bid_p = np.zeros(capacity, dtype=np.float64)
        self.bid_a = np.zeros(capacity, dtype=np.float64)
        self._n_ask = 0
        self._n_bid = 0

    # ---------- properties ----------
    @property
    def best_ask(self) -> float:
        return float(self.ask_p[0]) if self._n_ask > 0 else float("inf")

    @property
    def best_bid(self) -> float:
        return float(self.bid_p[0]) if self._n_bid > 0 else float("-inf")

    @property
    def best_ask_amount(self) -> float:
        return float(self.ask_a[0]) if self._n_ask > 0 else 0.0

    @property
    def best_bid_amount(self) -> float:
        return float(self.bid_a[0]) if self._n_bid > 0 else 0.0

    def mid(self) -> Optional[float]:
        if self._n_ask == 0 or self._n_bid == 0:
            return None
        return 0.5 * (self.best_ask + self.best_bid)

    def microprice(self) -> Optional[float]:
        if self._n_ask == 0 or self._n_bid == 0:
            return None
        qa = self.best_ask_amount
        qb = self.best_bid_amount
        denom = qa + qb
        if denom <= 0:
            return self.mid()
        return (self.best_ask * qb + self.best_bid * qa) / denom

    def imbalance(self) -> Optional[float]:
        if self._n_ask == 0 or self._n_bid == 0:
            return None
        qa = self.best_ask_amount
        qb = self.best_bid_amount
        denom = qa + qb
        if denom <= 0:
            return 0.5
        return qb / denom

    # ---------- apply snapshot ----------
    def apply_snapshot(self, ask_p: np.ndarray, ask_a: np.ndarray,
                       bid_p: np.ndarray, bid_a: np.ndarray) -> None:
        # Snapshots in the dataset come pre-sorted (asks ascending, bids descending),
        # but we re-sort defensively and skip NaN/zero rows.
        self._n_ask = self._load_side(self.ask_p, self.ask_a, ask_p, ask_a, ascending=True)
        self._n_bid = self._load_side(self.bid_p, self.bid_a, bid_p, bid_a, ascending=False)

    def _load_side(self, dst_p: np.ndarray, dst_a: np.ndarray,
                   src_p: np.ndarray, src_a: np.ndarray, ascending: bool) -> int:
        mask = (~np.isnan(src_p)) & (~np.isnan(src_a)) & (src_a > 0)
        p = src_p[mask]
        a = src_a[mask]
        if ascending:
            order = np.argsort(p, kind="stable")
        else:
            order = np.argsort(-p, kind="stable")
        n = min(len(p), self._cap)
        order = order[:n]
        dst_p[:n] = p[order]
        dst_a[:n] = a[order]
        if n < self._cap:
            dst_p[n:] = 0.0
            dst_a[n:] = 0.0
        return n

    # ---------- apply trade ----------
    def apply_trade(self, side: str, price: float, amount: float) -> TradeDelta:
        delta = TradeDelta()
        if side == "buy":
            self._apply_buy(price, amount, delta)
        elif side == "sell":
            self._apply_sell(price, amount, delta)
        return delta

    def _apply_buy(self, price: float, amount: float, delta: TradeDelta) -> None:
        # Sweep ask levels with price <= trade_price (when trade ate through book).
        if self._n_ask > 0 and price > self.ask_p[0]:
            sweep_to = 0
            while sweep_to < self._n_ask and self.ask_p[sweep_to] <= price:
                delta.consumed_ask_levels.append(float(self.ask_p[sweep_to]))
                sweep_to += 1
            if sweep_to > 0:
                self._shift_left(self.ask_p, self.ask_a, sweep_to, self._n_ask)
                self._n_ask -= sweep_to
        # Subtract from new best ask.
        if self._n_ask > 0:
            self.ask_a[0] -= amount
            if self.ask_a[0] <= 0:
                delta.consumed_ask_levels.append(float(self.ask_p[0]))
                self._shift_left(self.ask_p, self.ask_a, 1, self._n_ask)
                self._n_ask -= 1

    def _apply_sell(self, price: float, amount: float, delta: TradeDelta) -> None:
        if self._n_bid > 0 and price < self.bid_p[0]:
            sweep_to = 0
            while sweep_to < self._n_bid and self.bid_p[sweep_to] >= price:
                delta.consumed_bid_levels.append(float(self.bid_p[sweep_to]))
                sweep_to += 1
            if sweep_to > 0:
                self._shift_left(self.bid_p, self.bid_a, sweep_to, self._n_bid)
                self._n_bid -= sweep_to
        if self._n_bid > 0:
            self.bid_a[0] -= amount
            if self.bid_a[0] <= 0:
                delta.consumed_bid_levels.append(float(self.bid_p[0]))
                self._shift_left(self.bid_p, self.bid_a, 1, self._n_bid)
                self._n_bid -= 1

    @staticmethod
    def _shift_left(p: np.ndarray, a: np.ndarray, by: int, n: int) -> None:
        if by <= 0 or n - by <= 0:
            p[: n - by] = 0.0 if n - by < 0 else p[by:n]
            a[: n - by] = 0.0 if n - by < 0 else a[by:n]
            return
        p[: n - by] = p[by:n]
        a[: n - by] = a[by:n]
        p[n - by:n] = 0.0
        a[n - by:n] = 0.0
