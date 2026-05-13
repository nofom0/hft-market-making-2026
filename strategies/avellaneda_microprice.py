"""AS-2018 style extension: replace mid with microprice and add an
imbalance-based skew on the reservation price.

Microprice (Stoikov 2018, simple form):
    P_micro = (P_ask * Q_bid + P_bid * Q_ask) / (Q_ask + Q_bid)

Imbalance:
    I = Q_bid / (Q_ask + Q_bid)

The reservation price gets a skew term:
    r = P_micro - q * gamma * sigma^2 * (T - t) + beta * (I - 0.5)

A positive imbalance (more bids than asks at the top) lifts both quotes
upward — it predicts upward short-term drift.
"""

from __future__ import annotations

from typing import Optional

from backtest.lob import LimitOrderBook
from .avellaneda_stoikov import AvellanedaStoikov


class AvellanedaMicroprice(AvellanedaStoikov):
    def __init__(self, *, beta: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = float(beta)
        self._last_imbalance: float = 0.5

    def reference_price(self, book: LimitOrderBook) -> Optional[float]:
        mp = book.microprice()
        if mp is None:
            return None
        imb = book.imbalance()
        self._last_imbalance = 0.5 if imb is None else imb
        return mp + self.beta * (self._last_imbalance - 0.5)
