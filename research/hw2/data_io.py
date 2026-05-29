# -*- coding: utf-8 -*-
"""Lazy, time-windowed loading of the six-month parquet files (polars)."""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

US = 1_000_000
# Where the parquet files live (not committed to git). Override with the
# LIQ_DATA_DIR env var; otherwise expect a `data/` folder next to this file.
DATA = Path(os.environ.get("LIQ_DATA_DIR", Path(__file__).resolve().parent / "data"))

SYMS = {"btc": "btcusdt", "eth": "ethusdt"}


def to_us(date: str) -> int:
    """UTC date/datetime string -> int64 microseconds since epoch."""
    return int(pl.Series([date]).str.to_datetime(time_zone="UTC").dt.timestamp("us")[0])


def _scan(path: Path, t0: int, t1: int, columns: list[str]) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select(columns)
        .filter((pl.col("timestamp") >= t0) & (pl.col("timestamp") < t1))
        .collect(engine="streaming")
    )


def load_window(sym: str, start: str, end: str, bbo_pad_s: int = 300):
    """Load the four frames for ``sym`` over [start, end).

    BBO is padded ``bbo_pad_s`` seconds past ``end`` so the longest mark-out
    horizon still has forward quotes for trades near the window edge.
    Returns (trades, bbo, liq_binance, liq_bybit).
    """
    s = SYMS[sym]
    t0, t1 = to_us(start), to_us(end)

    trades = _scan(
        DATA / "binance_trades" / f"perp_{s}.parquet", t0, t1,
        ["timestamp", "side", "price", "amount"],
    )
    bbo = _scan(
        DATA / "binance_booktickers" / f"perp_{s}.parquet", t0, t1 + bbo_pad_s * US,
        ["timestamp", "bid_price", "ask_price"],
    )
    liq_b = _scan(
        DATA / "binance_liquidations" / f"perp_{s}.parquet", t0, t1,
        ["timestamp", "side", "price", "amount"],
    )
    liq_y = _scan(
        DATA / "bybit_liquidations" / f"{s}.parquet", t0, t1,
        ["timestamp", "side", "price", "amount"],
    )
    return trades, bbo, liq_b, liq_y


def n_days(start: str, end: str) -> float:
    return (to_us(end) - to_us(start)) / (24 * 3600 * US)
