"""Streaming sort-merge of LOB snapshots and trades.

The two CSV files are read in chunks (no full load into RAM) and merged on
``local_timestamp`` into a single iterator of events. Each event is one of:

    SnapEvent(ts, ask_p, ask_a, bid_p, bid_a)
    TradeEvent(ts, side, price, amount)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd


LOB_LEVELS = 25


@dataclass
class SnapEvent:
    ts: int
    ask_p: np.ndarray
    ask_a: np.ndarray
    bid_p: np.ndarray
    bid_a: np.ndarray


@dataclass
class TradeEvent:
    ts: int
    side: str  # 'buy' or 'sell'
    price: float
    amount: float


def _lob_columns() -> list[str]:
    cols = ["local_timestamp"]
    for i in range(LOB_LEVELS):
        cols.extend([f"asks[{i}].price", f"asks[{i}].amount",
                     f"bids[{i}].price", f"bids[{i}].amount"])
    return cols


def _iter_lob(path: str, start_ts: Optional[int], end_ts: Optional[int],
              chunksize: int = 50_000) -> Iterator[SnapEvent]:
    cols = _lob_columns()
    ask_p_cols = [f"asks[{i}].price" for i in range(LOB_LEVELS)]
    ask_a_cols = [f"asks[{i}].amount" for i in range(LOB_LEVELS)]
    bid_p_cols = [f"bids[{i}].price" for i in range(LOB_LEVELS)]
    bid_a_cols = [f"bids[{i}].amount" for i in range(LOB_LEVELS)]

    reader = pd.read_csv(path, usecols=cols, chunksize=chunksize)
    for chunk in reader:
        if start_ts is not None:
            chunk = chunk[chunk["local_timestamp"] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk["local_timestamp"] < end_ts]
        if chunk.empty:
            continue
        ts_arr = chunk["local_timestamp"].to_numpy(dtype=np.int64)
        ap = chunk[ask_p_cols].to_numpy(dtype=np.float64)
        aa = chunk[ask_a_cols].to_numpy(dtype=np.float64)
        bp = chunk[bid_p_cols].to_numpy(dtype=np.float64)
        ba = chunk[bid_a_cols].to_numpy(dtype=np.float64)
        for i in range(len(ts_arr)):
            yield SnapEvent(int(ts_arr[i]), ap[i], aa[i], bp[i], ba[i])


def _iter_trades(path: str, start_ts: Optional[int], end_ts: Optional[int],
                 chunksize: int = 200_000) -> Iterator[TradeEvent]:
    cols = ["local_timestamp", "side", "price", "amount"]
    reader = pd.read_csv(path, usecols=cols, chunksize=chunksize)
    for chunk in reader:
        if start_ts is not None:
            chunk = chunk[chunk["local_timestamp"] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk["local_timestamp"] < end_ts]
        if chunk.empty:
            continue
        ts_arr = chunk["local_timestamp"].to_numpy(dtype=np.int64)
        sides = chunk["side"].to_numpy()
        prices = chunk["price"].to_numpy(dtype=np.float64)
        amounts = chunk["amount"].to_numpy(dtype=np.float64)
        for i in range(len(ts_arr)):
            yield TradeEvent(int(ts_arr[i]), str(sides[i]),
                             float(prices[i]), float(amounts[i]))


def event_stream(lob_path: str, trades_path: str,
                 start_ts: Optional[int] = None,
                 end_ts: Optional[int] = None) -> Iterator[object]:
    """Yields SnapEvent / TradeEvent in non-decreasing ts order.

    On ts ties, snapshot comes before the trade — the snapshot is the state
    "as of" that ts and a trade with the same ts logically happens after.
    """
    lob_it = _iter_lob(lob_path, start_ts, end_ts)
    tr_it = _iter_trades(trades_path, start_ts, end_ts)

    s = next(lob_it, None)
    t = next(tr_it, None)

    while s is not None and t is not None:
        if s.ts <= t.ts:
            yield s
            s = next(lob_it, None)
        else:
            yield t
            t = next(tr_it, None)

    while s is not None:
        yield s
        s = next(lob_it, None)
    while t is not None:
        yield t
        t = next(tr_it, None)
