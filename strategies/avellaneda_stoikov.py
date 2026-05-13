"""Avellaneda-Stoikov 2008 baseline market-making strategy.

Reservation price:
    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

Optimal half-spread:
    delta = 0.5 * gamma * sigma^2 * (T - t) + (1 / gamma) * ln(1 + gamma / k)

Quotes:
    bid = round_to_tick_floor(r - delta)
    ask = round_to_tick_ceil (r + delta)

Volatility: EWMA of squared log-returns of mid-price, scaled to per-second.
The EWMA half-life is configurable (sigma_window_sec).
"""

from __future__ import annotations

import math
from typing import Optional

from backtest.lob import LimitOrderBook
from .base import Quotes, Strategy


class AvellanedaStoikov(Strategy):
    def __init__(self, *, gamma: float, k: float, sigma_window_sec: float,
                 quote_size: float, q_max: float, tick_size: float,
                 t_end_us: int, t_start_us: int,
                 risk_horizon_sec: float = 300.0,
                 max_inv_skew_ticks: float = 0.0) -> None:
        self.gamma = float(gamma)
        self.k = float(k)
        self.sigma_window_sec = float(sigma_window_sec)
        self.quote_size = float(quote_size)
        self.q_max = float(q_max)
        self.tick_size = float(tick_size)
        self.t_end_us = int(t_end_us)
        self.t_start_us = int(t_start_us)
        # Rolling/capped horizon: prevents tau from being huge at the start of the run.
        # tau = min(seconds_to_end, risk_horizon_sec). Default 300 s.
        self.risk_horizon_sec = float(risk_horizon_sec)
        # Optional hard clamp on the inventory skew (in ticks). 0 = disabled.
        self.max_inv_skew_ticks = float(max_inv_skew_ticks)

        # EWMA of squared log-returns per second (variance per second).
        self._sigma2: float = 0.0
        self._prev_mid: Optional[float] = None
        self._prev_ts: Optional[int] = None
        self._initialized: bool = False

    # ----- volatility -----
    def _update_sigma(self, mid: float, ts: int) -> None:
        if self._prev_mid is None or self._prev_ts is None:
            self._prev_mid, self._prev_ts = mid, ts
            return
        dt_us = ts - self._prev_ts
        if dt_us <= 0 or self._prev_mid <= 0 or mid <= 0:
            self._prev_mid, self._prev_ts = mid, ts
            return
        dt_sec = dt_us / 1_000_000.0
        log_ret = math.log(mid / self._prev_mid)
        instant_var = (log_ret * log_ret) / dt_sec    # variance per second
        # EWMA with half-life = sigma_window_sec
        half_life = max(self.sigma_window_sec, 1e-6)
        alpha = 1.0 - math.exp(-dt_sec * math.log(2.0) / half_life)
        if not self._initialized:
            self._sigma2 = instant_var
            self._initialized = True
        else:
            self._sigma2 = (1 - alpha) * self._sigma2 + alpha * instant_var
        self._prev_mid, self._prev_ts = mid, ts

    # ----- reference price -----
    def reference_price(self, book: LimitOrderBook) -> Optional[float]:
        return book.mid()

    # ----- main entry -----
    def on_book(self, book: LimitOrderBook, ts: int, inventory: float) -> Optional[Quotes]:
        ref = self.reference_price(book)
        if ref is None:
            return None
        # Update vol tracker on raw mid (kept consistent across variants).
        mid = book.mid()
        if mid is not None:
            self._update_sigma(mid, ts)

        sigma2 = self._sigma2
        # Cap tau at risk_horizon_sec so the inventory penalty doesn't explode at
        # the start of a long run (e.g. tau=86400s * q=1000 * gamma * sigma2 >> spread).
        time_remaining = max(0.0, (self.t_end_us - ts) / 1_000_000.0)
        tau = min(time_remaining, self.risk_horizon_sec)

        # Normalize inventory to lots (quote_size units) so gamma is independent
        # of position sizing. q_norm = 1 means "one full order held".
        q_norm = inventory / self.quote_size if self.quote_size > 0 else inventory

        # Inventory skew with optional clamp.
        inv_skew = -q_norm * self.gamma * sigma2 * tau
        if self.max_inv_skew_ticks > 0:
            max_s = self.max_inv_skew_ticks * self.tick_size
            inv_skew = max(-max_s, min(max_s, inv_skew))

        # Reservation price + half-spread (AS 2008).
        r = ref + inv_skew
        delta = 0.5 * self.gamma * sigma2 * tau + (1.0 / self.gamma) * math.log1p(self.gamma / self.k)

        # Floor/ceil to tick grid.
        bid = math.floor((r - delta) / self.tick_size) * self.tick_size
        ask = math.ceil((r + delta) / self.tick_size) * self.tick_size

        # Defensive: never cross the book — if our quote is at or beyond the
        # opposite best, pull it back inside by one tick (becomes a maker).
        best_ask = book.best_ask
        best_bid = book.best_bid
        if math.isfinite(best_ask) and bid >= best_ask:
            bid = best_ask - self.tick_size
        if math.isfinite(best_bid) and ask <= best_bid:
            ask = best_bid + self.tick_size

        # Inventory hard cap.
        bid_q: Optional[float] = bid
        ask_q: Optional[float] = ask
        if inventory >= self.q_max:
            bid_q = None
        if inventory <= -self.q_max:
            ask_q = None

        return Quotes(bid_price=bid_q, ask_price=ask_q, size=self.quote_size)
