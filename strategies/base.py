"""Strategy interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backtest.lob import LimitOrderBook


@dataclass
class Quotes:
    bid_price: Optional[float]   # None = do not quote bid side
    ask_price: Optional[float]   # None = do not quote ask side
    size: float


class Strategy:
    def on_book(self, book: LimitOrderBook, ts: int, inventory: float) -> Optional[Quotes]:
        """Return new desired quotes or None if no change is needed.

        ts is in microseconds. inventory is current signed position (base units).
        """
        raise NotImplementedError
