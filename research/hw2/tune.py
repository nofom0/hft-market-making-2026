# -*- coding: utf-8 -*-
"""Grid-search (window, threshold) per horizon, pooled over the train split.

Single-week argmax overfits to noise (it pins thresholds at grid extremes).
Instead we walk the train period one week at a time (memory-bounded), and for
every (tau, window, threshold) accumulate the pooled Score numerators/denominators
across all weeks.  Final Score(tau) = PnL_kept - PnL_all on the pooled sums.
Constraint: pooled kept-turnover/day >= 500k.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

from baseline import US, TAUS, _liq_cumulative, compute_pnl
from data_io import load_window, n_days

WINDOWS = [3.0, 5.0, 10.0, 20.0]                       # seconds
THRESHOLDS = [50e3, 100e3, 200e3, 400e3, 800e3]        # USD


def _weeks(start: str, end: str) -> list[tuple[str, str]]:
    lo = pl.Series([start]).str.to_datetime(time_zone="UTC")[0]
    hi = pl.Series([end]).str.to_datetime(time_zone="UTC")[0]
    out, cur = [], lo
    import datetime as dt
    while cur < hi:
        nxt = min(cur + dt.timedelta(days=7), hi)
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return out


def signed_liq_window_np(trades: pl.DataFrame, liq: pl.DataFrame, window_s: float) -> np.ndarray:
    t = trades.select(
        idx=pl.int_range(pl.len(), dtype=pl.UInt32), timestamp=pl.col("timestamp"),
    ).sort("timestamp")
    t = t.join_asof(
        liq.rename({"ts": "timestamp", "cum": "cum_hi"}), on="timestamp", strategy="backward",
    ).with_columns(pl.col("cum_hi").fill_null(0.0))
    t = (
        t.with_columns(t_lo=pl.col("timestamp") - int(window_s * US))
        .join_asof(liq.rename({"ts": "t_lo_key", "cum": "cum_lo"}),
                   left_on="t_lo", right_on="t_lo_key", strategy="backward")
        .with_columns(pl.col("cum_lo").fill_null(0.0))
    )
    return t.with_columns(lw=pl.col("cum_hi") - pl.col("cum_lo")).sort("idx")["lw"].to_numpy()


def tune(sym: str, start: str, end: str) -> dict[int, tuple[float, float]]:
    weeks = _weeks(start, end)
    total_days = n_days(start, end)
    # pooled accumulators
    all_wp = {tau: 0.0 for tau in TAUS}
    all_w = {tau: 0.0 for tau in TAUS}
    keep_wp = {(tau, wi, th): 0.0 for tau in TAUS for wi in WINDOWS for th in THRESHOLDS}
    keep_w = {(tau, wi, th): 0.0 for tau in TAUS for wi in WINDOWS for th in THRESHOLDS}

    print(f"\n=== {sym.upper()}  {start}..{end}  ({len(weeks)} weeks, {total_days:.0f} days) ===", flush=True)
    for (ws, we) in weeks:
        trades, bbo, lb, ly = load_window(sym, ws, we)
        pnl = compute_pnl(trades, bbo)
        s = pnl["s"].to_numpy()
        w = pnl["w"].to_numpy()
        liq = _liq_cumulative(lb, ly)
        lw = {wi: signed_liq_window_np(trades, liq, wi) for wi in WINDOWS}
        metric = {wi: s * lw[wi] for wi in WINDOWS}      # s_i * signed_liq_window
        for tau in TAUS:
            p = pnl[f"pnl_{tau}"].to_numpy()
            ok = np.isfinite(p)
            wo, po = w[ok], p[ok]
            all_wp[tau] += float((wo * po).sum())
            all_w[tau] += float(wo.sum())
            for wi in WINDOWS:
                mo = metric[wi][ok]
                for th in THRESHOLDS:
                    keep = mo > -th                       # not filtered
                    keep_wp[(tau, wi, th)] += float((wo[keep] * po[keep]).sum())
                    keep_w[(tau, wi, th)] += float(wo[keep].sum())
        print(f"   {ws}..{we}  trades={trades.height:,}", flush=True)
        del trades, bbo, lb, ly, pnl, lw, metric

    best: dict[int, tuple[float, float]] = {}
    for tau in TAUS:
        pnl_all = all_wp[tau] / all_w[tau]
        rows = []
        for wi in WINDOWS:
            for th in THRESHOLDS:
                kw = keep_w[(tau, wi, th)]
                if kw <= 0 or kw / total_days < 500_000.0:
                    continue
                pnl_kept = keep_wp[(tau, wi, th)] / kw
                rows.append((pnl_kept - pnl_all, wi, th, pnl_kept, kw / total_days))
        rows.sort(reverse=True)
        score, wi, th, pk, kt = rows[0]
        best[tau] = (wi, th)
        print(f"  tau={tau:3d}  win={wi:>4}s thr=${th:>10,.0f}  Score={score:+.4f}  "
              f"PnL_all={pnl_all:+.4f} PnL_kept={pk:+.4f}  keptTO/day=${kt:,.0f}")
    print(f"  -> DEFAULT_PARAMS['{sym}'] = {{" +
          ", ".join(f"{t}: ({best[t][0]}, {best[t][1]:.0f})" for t in TAUS) + "}", flush=True)
    return best


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2025-12-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-02-01"
    sym = sys.argv[3] if len(sys.argv) > 3 else "btc"
    tune(sym, start, end)
