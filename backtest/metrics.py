"""Online metrics accumulator.

State invariants:
  cash      — running cash balance (net of fees)
  inv       — running inventory (base units, signed)
  total_pnl = cash + inv * mid_last     (mark-to-market)

Equity samples are recorded with a configurable wall-time period (default 1 s)
and are used to plot PnL, derive 1-second PnL deltas, Sharpe and max drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math


@dataclass
class FillRecord:
    ts: int
    side: str       # 'bid' or 'ask'
    price: float
    qty: float
    liquidity: str  # 'maker'/'taker'
    fee: float


@dataclass
class EquityPoint:
    ts: int
    pnl: float
    inv: float
    mid: float


@dataclass
class Metrics:
    fee_maker: float = 0.0
    fee_taker: float = 0.0
    sample_period_us: int = 1_000_000   # 1 second

    cash: float = 0.0
    inv: float = 0.0
    last_mid: float = 0.0
    turnover: float = 0.0

    n_fills_bid: int = 0
    n_fills_ask: int = 0

    max_abs_inv: float = 0.0
    sum_abs_inv: float = 0.0
    inv_samples: int = 0

    fills: list[FillRecord] = field(default_factory=list)
    equity: list[EquityPoint] = field(default_factory=list)
    _next_sample_ts: int = 0

    def on_mid(self, ts: int, mid: float) -> None:
        self.last_mid = mid
        if self._next_sample_ts == 0:
            self._next_sample_ts = ts + self.sample_period_us
            self.equity.append(EquityPoint(ts, self.total_pnl(), self.inv, mid))
        elif ts >= self._next_sample_ts:
            self.equity.append(EquityPoint(ts, self.total_pnl(), self.inv, mid))
            # advance to next aligned sample
            while self._next_sample_ts <= ts:
                self._next_sample_ts += self.sample_period_us
        self.sum_abs_inv += abs(self.inv)
        self.inv_samples += 1
        if abs(self.inv) > self.max_abs_inv:
            self.max_abs_inv = abs(self.inv)

    def on_fill(self, ts: int, side: str, price: float, qty: float,
                liquidity: str) -> None:
        rate = self.fee_maker if liquidity == "maker" else self.fee_taker
        notional = price * qty
        fee = abs(rate) * notional * (1 if rate >= 0 else -1)
        if side == "bid":
            # we bought
            self.cash -= notional
            self.inv += qty
            self.n_fills_bid += 1
        elif side == "ask":
            # we sold
            self.cash += notional
            self.inv -= qty
            self.n_fills_ask += 1
        else:
            raise ValueError(side)
        # Apply fee: positive rate = fee paid (cash decreases),
        # negative rate = rebate (cash increases).
        self.cash -= fee
        self.turnover += notional
        self.fills.append(FillRecord(ts=ts, side=side, price=price, qty=qty,
                                     liquidity=liquidity, fee=fee))

    def total_pnl(self) -> float:
        return self.cash + self.inv * self.last_mid

    # ---------- final aggregates ----------
    def summarize(self) -> dict:
        eq = self.equity
        sharpe = 0.0
        max_dd = 0.0
        ret_std = 0.0
        ret_mean = 0.0
        if len(eq) >= 2:
            diffs = [eq[i].pnl - eq[i - 1].pnl for i in range(1, len(eq))]
            n = len(diffs)
            ret_mean = sum(diffs) / n
            ret_std = math.sqrt(max(0.0, sum((d - ret_mean) ** 2 for d in diffs) / n))
            if ret_std > 0:
                # Annualized w.r.t. 1-second samples: sqrt(seconds_in_year)
                sec_per_year = 365 * 24 * 3600
                period_sec = self.sample_period_us / 1_000_000
                sharpe = (ret_mean / ret_std) * math.sqrt(sec_per_year / period_sec)
            peak = eq[0].pnl
            for p in eq:
                if p.pnl > peak:
                    peak = p.pnl
                dd = peak - p.pnl
                if dd > max_dd:
                    max_dd = dd
        avg_abs_inv = (self.sum_abs_inv / self.inv_samples) if self.inv_samples else 0.0
        return {
            "total_pnl": self.total_pnl(),
            "cash": self.cash,
            "final_inventory": self.inv,
            "last_mid": self.last_mid,
            "turnover": self.turnover,
            "n_fills": self.n_fills_bid + self.n_fills_ask,
            "n_fills_bid": self.n_fills_bid,
            "n_fills_ask": self.n_fills_ask,
            "max_abs_inventory": self.max_abs_inv,
            "avg_abs_inventory": avg_abs_inv,
            "sharpe_annualized_1s": sharpe,
            "max_drawdown": max_dd,
            "pnl_diff_mean": ret_mean,
            "pnl_diff_std": ret_std,
            "n_equity_samples": len(eq),
        }
