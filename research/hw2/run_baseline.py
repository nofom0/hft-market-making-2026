# -*- coding: utf-8 -*-
"""Evaluate the baseline classifier and report the required metrics.

Walks a date range one week at a time (memory-bounded) and pools the Score
numerators/denominators, so the printed numbers are exact pooled metrics over
the whole period -- per symbol and per horizon:

    PnL_all, PnL_kept, PnL_filtered, Score = PnL_kept - PnL_all,
    KeptTurnover/day  (constraint: >= 500_000 USD/day).

Usage:
    python run_baseline.py                      # validation month, both symbols
    python run_baseline.py 2025-12-01 2026-02-01 btc   # custom range/symbol
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

import datetime as dt

from baseline import DEFAULT_PARAMS, TAUS, classify_trades, compute_pnl
from data_io import load_window, n_days, to_us


def weeks(start: str, end: str) -> list[tuple[str, str]]:
    lo = dt.datetime.fromtimestamp(to_us(start) / 1e6, dt.timezone.utc)
    hi = dt.datetime.fromtimestamp(to_us(end) / 1e6, dt.timezone.utc)
    out, cur = [], lo
    while cur < hi:
        nxt = min(cur + dt.timedelta(days=7), hi)
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return out


def evaluate_pooled(sym: str, start: str, end: str) -> pl.DataFrame:
    total_days = n_days(start, end)
    acc = {tau: dict(all_wp=0.0, all_w=0.0, k_wp=0.0, k_w=0.0, f_wp=0.0, f_w=0.0, n=0)
           for tau in TAUS}

    print(f"\n### {sym.upper()}  {start}..{end}  ({total_days:.0f} days)", flush=True)
    for ws, we in weeks(start, end):
        trades, bbo, lb, ly = load_window(sym, ws, we)
        pnl = compute_pnl(trades, bbo)
        # data_io drops the ticker column to save memory, so pass per-symbol
        # params explicitly here.  The submission frames carry `ticker`, so
        # classify_trades auto-detects the symbol when params is None.
        filt = classify_trades(trades, bbo, lb, ly, params=DEFAULT_PARAMS[sym])
        w = pnl["w"].to_numpy()
        for tau in TAUS:
            p = pnl[f"pnl_{tau}"].to_numpy()
            f = filt[tau].to_numpy().astype(bool)
            ok = np.isfinite(p)
            wo, po, fo = w[ok], p[ok], f[ok]
            a = acc[tau]
            a["all_wp"] += float((wo * po).sum()); a["all_w"] += float(wo.sum())
            keep = ~fo
            a["k_wp"] += float((wo[keep] * po[keep]).sum()); a["k_w"] += float(wo[keep].sum())
            a["f_wp"] += float((wo[fo] * po[fo]).sum()); a["f_w"] += float(wo[fo].sum())
            a["n"] += int(ok.sum())
        print(f"   {ws}..{we}  trades={trades.height:,}", flush=True)
        del trades, bbo, lb, ly, pnl, filt

    rows = []
    for tau in TAUS:
        a = acc[tau]
        pnl_all = a["all_wp"] / a["all_w"] if a["all_w"] else float("nan")
        pnl_kept = a["k_wp"] / a["k_w"] if a["k_w"] else float("nan")
        pnl_filt = a["f_wp"] / a["f_w"] if a["f_w"] else float("nan")
        kept_to = a["k_w"] / total_days
        rows.append({
            "sym": sym, "tau": tau, "n": a["n"],
            "frac_filtered": a["f_w"] / a["all_w"] if a["all_w"] else float("nan"),
            "PnL_all": round(pnl_all, 4), "PnL_kept": round(pnl_kept, 4),
            "PnL_filtered": round(pnl_filt, 4), "Score": round(pnl_kept - pnl_all, 4),
            "KeptTO/day": round(kept_to, 0), "constraint_ok": kept_to >= 500_000.0,
        })
    return pl.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        start, end = sys.argv[1], sys.argv[2]
        syms = [sys.argv[3]] if len(sys.argv) > 3 else ["btc", "eth"]
    else:
        start, end = "2026-02-01", "2026-03-01"   # validation month
        syms = ["btc", "eth"]

    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(300)
    tables = [evaluate_pooled(s, start, end) for s in syms]
    print("\n==================== SUMMARY ====================")
    print(pl.concat(tables))
