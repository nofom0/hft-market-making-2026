"""Backtest report: summary.json, fills.csv, equity.csv, and PNG plots."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import Metrics


def _ts_to_dt(us: int) -> datetime:
    return datetime.utcfromtimestamp(us / 1_000_000)


def write_report(metrics: Metrics, out_dir: str, run_name: str,
                 config_snapshot: dict) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(out_dir, f"{run_name}_{stamp}")
    os.makedirs(folder, exist_ok=True)

    # summary.json
    summary = metrics.summarize()
    summary["run_name"] = run_name
    summary["config"] = config_snapshot
    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    # fills.csv
    with open(os.path.join(folder, "fills.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts_us", "side", "price", "qty", "liquidity", "fee"])
        for r in metrics.fills:
            w.writerow([r.ts, r.side, r.price, r.qty, r.liquidity, r.fee])

    # equity.csv
    with open(os.path.join(folder, "equity.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts_us", "pnl", "inventory", "mid"])
        for p in metrics.equity:
            w.writerow([p.ts, p.pnl, p.inv, p.mid])

    # Plots
    _plot_pnl(metrics, os.path.join(folder, "pnl.png"))
    _plot_inventory(metrics, os.path.join(folder, "inventory.png"))
    _plot_mid_with_fills(metrics, os.path.join(folder, "mid_with_fills.png"))

    return folder


def _plot_pnl(metrics: Metrics, path: str) -> None:
    if not metrics.equity:
        return
    xs = [_ts_to_dt(p.ts) for p in metrics.equity]
    ys = [p.pnl for p in metrics.equity]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, color="tab:blue", lw=1)
    ax.set_title("Total PnL (mark-to-market)")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("PnL")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_inventory(metrics: Metrics, path: str) -> None:
    if not metrics.equity:
        return
    xs = [_ts_to_dt(p.ts) for p in metrics.equity]
    ys = [p.inv for p in metrics.equity]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, color="tab:orange", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Inventory")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Inventory (base units)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_mid_with_fills(metrics: Metrics, path: str) -> None:
    if not metrics.equity:
        return
    xs = [_ts_to_dt(p.ts) for p in metrics.equity]
    mid = [p.mid for p in metrics.equity]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, mid, color="black", lw=0.6, label="mid")
    bid_x = [_ts_to_dt(r.ts) for r in metrics.fills if r.side == "bid"]
    bid_y = [r.price for r in metrics.fills if r.side == "bid"]
    ask_x = [_ts_to_dt(r.ts) for r in metrics.fills if r.side == "ask"]
    ask_y = [r.price for r in metrics.fills if r.side == "ask"]
    if bid_x:
        ax.scatter(bid_x, bid_y, color="tab:green", s=8, label="bid fills", alpha=0.6)
    if ask_x:
        ax.scatter(ask_x, ask_y, color="tab:red", s=8, label="ask fills", alpha=0.6)
    ax.set_title("Mid + fills")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Price")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
