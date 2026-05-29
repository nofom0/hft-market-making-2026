# -*- coding: utf-8 -*-
"""Baseline liquidation-signal solution (polars).

Pieces
------
* ``classify_trades`` -- the submission function.  Takes the four frames
  (trades, bbo, liq_binance, liq_bybit) with the public schema and returns, for
  every horizon tau in {30, 120, 300}, a 0/1 array the same length as ``trades``
  (0 = keep, 1 = filter).
* ``compute_pnl`` -- maker mark-out pnl_i(tau) per trade (description section
  "Markout").
* ``evaluate`` -- Score, PnL_all, PnL_kept, PnL_filtered and kept turnover/day
  per horizon for a given filter.

Heuristic
---------
Liquidations are *lagging* indicators: they mark the end of a move, after which
the price tends to mean-revert (see EDA event studies).  So a maker fill is
toxic when it leans *against* that imminent reversion -- i.e. the maker trades
in the same direction the price is about to revert away from.  Concretely, after
recent sell-liquidation pressure (price flushed down, about to bounce up) a maker
who just *sold* (taker buy) is adversely selected; symmetrically for buy
liquidations.

We sum signed liquidation notional (buy=+, sell=-) from both venues over the
last ``window_s`` seconds before each trade (Bybit timestamps shifted +200 ms
per the cross-exchange delay rule) and filter the trade when the maker side
opposes that pressure beyond a notional threshold:

    f_i = 1  iff  s_i * signed_liq_notional_window <= -threshold

with s_i = +1 for taker buy (maker sell) and -1 for taker sell (maker buy).
The (window_s, threshold) pair is tuned per horizon.
"""
from __future__ import annotations

import polars as pl

US = 1_000_000               # microseconds per second
TAUS = (30, 120, 300)        # mark-out horizons, seconds
BYBIT_LAG_US = 200_000       # +200 ms cross-exchange delay for Bybit events
REBATE_BPS = 0.5             # maker rebate added to every fill
NOTIONAL_CAP = 100_000.0     # w_i = min(notional_i, cap)

# Per-symbol, per-horizon (window_seconds, signed-notional threshold in USD).
# Tuned on the train split (see tune.py).  Larger threshold => filter fewer,
# safer trades.  BTC and ETH liquidation notionals live on different scales, so
# thresholds are kept per symbol.
DEFAULT_PARAMS: dict[str, dict[int, tuple[float, float]]] = {
    # Pooled grid-search over the train split 2025-12-01..2026-02-01 (see tune.py).
    "btc": {30: (5.0, 50_000.0), 120: (20.0, 200_000.0), 300: (20.0, 800_000.0)},
    "eth": {30: (20.0, 50_000.0), 120: (20.0, 50_000.0), 300: (20.0, 800_000.0)},
}


def _detect_symbol(frame: pl.DataFrame) -> str:
    """'btc' or 'eth' from a frame's ticker column (defaults to 'btc')."""
    if "ticker" in frame.columns and frame.height:
        return "eth" if "eth" in str(frame["ticker"][0]).lower() else "btc"
    return "btc"


def _signed_notional(side_col: str = "side") -> pl.Expr:
    """price*amount with sign +1 for buy, -1 for sell."""
    sign = pl.when(pl.col(side_col) == "buy").then(1.0).otherwise(-1.0)
    return sign * pl.col("price") * pl.col("amount")


def _liq_cumulative(liq_binance: pl.DataFrame, liq_bybit: pl.DataFrame) -> pl.DataFrame:
    """Merged, time-sorted liquidation stream with a running signed notional.

    Bybit timestamps are shifted +200 ms (events only become observable then).
    Returns columns: ``ts`` (int64 us), ``cum`` (cumulative signed notional).
    """
    lb = liq_binance.select(ts=pl.col("timestamp"), sn=_signed_notional())
    ly = liq_bybit.select(ts=pl.col("timestamp") + BYBIT_LAG_US, sn=_signed_notional())
    liq = pl.concat([lb, ly]).sort("ts")
    return liq.with_columns(cum=pl.col("sn").cum_sum()).select("ts", "cum")


def classify_trades(
    trades: pl.DataFrame,
    bbo: pl.DataFrame,          # noqa: ARG001 - kept for signature compatibility
    liq_binance: pl.DataFrame,
    liq_bybit: pl.DataFrame,
    params: dict[int, tuple[float, float]] | None = None,
) -> dict[int, "pl.Series"]:
    """Return {tau: 0/1 Series aligned with ``trades``} (0 keep, 1 filter)."""
    params = params or DEFAULT_PARAMS[_detect_symbol(trades)]

    if liq_binance.height == 0 and liq_bybit.height == 0:
        zero = pl.zeros(trades.height, dtype=pl.Int8, eager=True)
        return {tau: zero.clone() for tau in TAUS}

    liq = _liq_cumulative(liq_binance, liq_bybit)

    # Keep the original row order so the output arrays line up with `trades`.
    t = (
        trades.select(
            idx=pl.int_range(pl.len(), dtype=pl.UInt32),
            timestamp=pl.col("timestamp"),
            s=pl.when(pl.col("side") == "buy").then(pl.lit(1.0)).otherwise(pl.lit(-1.0)),
        )
        .sort("timestamp")
    )

    # Cumulative signed liq notional at t_i (last event at-or-before the trade).
    t = t.join_asof(
        liq.rename({"ts": "timestamp", "cum": "cum_hi"}),
        on="timestamp", strategy="backward",
    ).with_columns(pl.col("cum_hi").fill_null(0.0))

    out: dict[int, pl.Series] = {}
    for tau in TAUS:
        window_s, thr = params[tau]
        # Cumulative at the window's lower edge t_i - window.  t_lo is monotone
        # in timestamp, so the frame is still sorted on the join key.
        a = (
            t.with_columns(t_lo=pl.col("timestamp") - int(window_s * US))
            .join_asof(
                liq.rename({"ts": "t_lo_key", "cum": "cum_lo"}),
                left_on="t_lo", right_on="t_lo_key", strategy="backward",
            )
            .with_columns(pl.col("cum_lo").fill_null(0.0))
        )
        # Signed liquidation notional inside (t_i - window, t_i].
        liq_win = pl.col("cum_hi") - pl.col("cum_lo")
        f = (pl.col("s") * liq_win <= -thr).cast(pl.Int8)
        res = a.select("idx", f=f).sort("idx")
        out[tau] = res["f"]
    return out


def compute_pnl(trades: pl.DataFrame, bbo: pl.DataFrame) -> pl.DataFrame:
    """Per-trade maker mark-out.

    Returns a frame in original trade order with columns ``s``, ``w`` and one
    ``pnl_<tau>`` per horizon (null when t_i + tau falls outside the BBO range,
    which excludes the trade from that horizon's metrics).
    """
    b = (
        bbo.sort("timestamp")
        .with_columns(mid=0.5 * (pl.col("bid_price") + pl.col("ask_price")))
        .select("timestamp", "mid")
    )
    t_max = b["timestamp"][-1]

    t = (
        trades.select(
            idx=pl.int_range(pl.len(), dtype=pl.UInt32),
            timestamp=pl.col("timestamp"),
            price=pl.col("price"),
            s=pl.when(pl.col("side") == "buy").then(pl.lit(1.0)).otherwise(pl.lit(-1.0)),
            notional=pl.col("price") * pl.col("amount"),
        )
        .with_columns(w=pl.min_horizontal(pl.col("notional"), pl.lit(NOTIONAL_CAP)))
        .sort("timestamp")
    )

    pnl_cols = []
    for tau in TAUS:
        target = pl.col("timestamp") + tau * US
        a = (
            t.with_columns(target=target)
            .join_asof(
                b.rename({"timestamp": "bbo_ts", "mid": f"mid_{tau}"}),
                left_on="target", right_on="bbo_ts", strategy="backward",
            )
        )
        # pnl = -s * (mid(tau) - price)/price * 1e4 + rebate; null past BBO range.
        pnl = (
            pl.when(pl.col("target") <= t_max)
            .then(
                -pl.col("s") * (pl.col(f"mid_{tau}") - pl.col("price")) / pl.col("price")
                * 1e4 + REBATE_BPS
            )
            .otherwise(None)
        ).alias(f"pnl_{tau}")
        t = a.with_columns(pnl).drop("target", "bbo_ts", f"mid_{tau}")
        pnl_cols.append(f"pnl_{tau}")

    return t.sort("idx").select("idx", "s", "w", *pnl_cols)


def evaluate(
    pnl: pl.DataFrame,
    filters: dict[int, pl.Series],
    n_days: float,
) -> pl.DataFrame:
    """Score / PnL_all / PnL_kept / PnL_filtered / kept-turnover-per-day per tau."""
    rows = []
    for tau in TAUS:
        f = filters[tau].cast(pl.Float64)
        df = pnl.select(
            "w",
            pnl=pl.col(f"pnl_{tau}"),
            f=f,
        ).filter(pl.col("pnl").is_not_null())

        w = df["w"]
        p = df["pnl"]
        keep = 1.0 - df["f"]
        filt = df["f"]

        def wavg(mask: pl.Series) -> float:
            num = (mask * w * p).sum()
            den = (mask * w).sum()
            return float(num / den) if den > 0 else float("nan")

        ones = pl.Series([1.0] * df.height)
        pnl_all = wavg(ones)
        pnl_kept = wavg(keep)
        pnl_filt = wavg(filt)
        kept_turnover_day = float((keep * w).sum() / n_days)

        rows.append({
            "tau": tau,
            "n": df.height,
            "frac_filtered": float(filt.mean()) if df.height else float("nan"),
            "PnL_all": pnl_all,
            "PnL_kept": pnl_kept,
            "PnL_filtered": pnl_filt,
            "Score": pnl_kept - pnl_all,
            "KeptTurnover/day": kept_turnover_day,
            "constraint_ok": kept_turnover_day >= 500_000.0,
        })
    return pl.DataFrame(rows)
