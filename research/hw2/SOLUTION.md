# HW2 — Liquidation signal: baseline solution

**Goal (week 2).** Build a simple baseline that filters Binance maker fills using
a liquidation signal: load 6 months of data with **polars**, write a function
that classifies each trade as *keep* (0) or *filter* (1), compute the performance
metrics (`Score`, `PnL_kept`, `PnL_filtered`, turnover/day), and make sure the
kept-turnover constraint (≥ 500 k\$/day) holds. Full task statement in
[description.md](description.md).

> Data (multi-GB parquet) is **not** in this repo. Point the loader at it with
> `LIQ_DATA_DIR=/path/to/data` or drop a `data/` folder next to the code.

## TL;DR — what was done and the answer

- **Loader** ([data_io.py](data_io.py)): lazy, time-windowed polars scans with
  column + timestamp pushdown. The raw files are ~6 months / billions of rows
  (ETH trades alone ≈ 1.37 B), so nothing is ever fully materialised.
- **Classifier** ([baseline.py](baseline.py) → `classify_trades`): the submission
  function. Returns, for each horizon τ ∈ {30, 120, 300} s, a 0/1 array aligned
  with `trades`.
- **Metrics** ([baseline.py](baseline.py) → `compute_pnl`, `evaluate`): exact
  mark-out PnL and the `Score = PnL_kept − PnL_all` metric.
- **Tuning** ([tune.py](tune.py)) + **reporting** ([run_baseline.py](run_baseline.py)).

**Answer:** the baseline gives a **positive Score for every symbol and every
horizon, both in-sample and out-of-sample**, while filtering only ~1–7 % of
clipped volume and meeting the turnover constraint by ~4 orders of magnitude.

## The signal (heuristic)

Liquidations are **lagging** indicators — they mark the *end* of a move, after
which the price tends to mean-revert (confirmed by the week-1 EDA event studies).
So a maker fill is *toxic* when it leans **against** that imminent reversion.

For every trade, sum the **signed liquidation notional** (buy = +, sell = −) from
both venues over the last `window` seconds, with **Bybit timestamps shifted
+200 ms** (the cross-exchange delay rule). Filter the trade when the maker side
opposes that pressure beyond a threshold:

```
f_i = 1   iff   s_i · signed_liq_notional_window  <=  −threshold
```

where `s_i = +1` for a taker buy (maker sell) and `−1` for a taker sell (maker
buy). The window sum is computed cheaply: one cumulative sum over the merged,
time-sorted liquidation stream, then two as-of joins (`cum(t_i) − cum(t_i −
window)`). `(window, threshold)` is tuned **per symbol** (BTC and ETH live on
different notional scales) and **per horizon**.

Tuned parameters (pooled grid-search over the train split `2025-12-01 →
2026-02-01`), stored in `DEFAULT_PARAMS`:

| symbol | τ = 30 s | τ = 120 s | τ = 300 s |
|---|---|---|---|
| BTC | (5 s, \$50 k)  | (20 s, \$200 k) | (20 s, \$800 k) |
| ETH | (20 s, \$50 k) | (20 s, \$50 k)  | (20 s, \$800 k) |

## Metrics (exactly as specified)

- mid forward-filled to `t_i + τ` (as-of backward join on the BBO); trades whose
  `t_i + τ` is past the available BBO are dropped from that horizon;
- `pnl_i(τ) = −s_i · (m_i(τ) − p_i) / p_i · 10⁴ + 0.5` (the `+0.5` bps maker rebate);
- `w_i = min(notional_i, 100 000)`;
- `Score = PnL_kept − PnL_all`, plus `PnL_filtered` and `KeptTurnover/day`.

## Results

All numbers are **pooled** over the whole period (exact, not week-averaged).

### Validation — out-of-sample (2026-02-01 → 2026-03-01)

| sym | τ | **Score** | PnL_all | PnL_kept | PnL_filtered | frac filtered | KeptTO/day | constraint |
|---|---|---|---|---|---|---|---|---|
| BTC | 30  | **+0.056** | −0.188 | −0.131 | −1.86  | 3.3 % | \$14.2 B | ✅ |
| BTC | 120 | **+0.103** | −0.174 | −0.071 | −2.75  | 3.8 % | \$14.1 B | ✅ |
| BTC | 300 | **+0.098** | −0.156 | −0.058 | −6.68  | 1.5 % | \$14.4 B | ✅ |
| ETH | 30  | **+0.094** | +0.119 | +0.213 | −1.19  | 6.7 % | \$11.7 B | ✅ |
| ETH | 120 | **+0.270** | +0.133 | +0.404 | −3.65  | 6.7 % | \$11.7 B | ✅ |
| ETH | 300 | **+0.124** | +0.235 | +0.360 | −12.62 | 1.0 % | \$12.4 B | ✅ |

(≈ 190 M BTC / 317 M ETH evaluated maker fills.)

### Train — in-sample (2025-12-01 → 2026-02-01)

| sym | τ | Score | PnL_all | PnL_kept |
|---|---|---|---|---|
| BTC | 30 / 120 / 300 | +0.046 / +0.073 / +0.061 | −0.105 / +0.004 / +0.034 | −0.059 / +0.077 / +0.095 |
| ETH | 30 / 120 / 300 | +0.067 / +0.234 / +0.246 | −0.034 / +0.108 / +0.181 | +0.033 / +0.342 / +0.427 |

**Reading the results**

- **Score > 0 for every symbol/horizon, train and validation** → the filter
  generalises out-of-sample.
- **PnL_filtered is strongly negative everywhere** (−1 to −13 bps): the dropped
  trades really are the toxic ones; we keep ~93–99 % of clipped volume.
- **The 500 k\$/day kept-turnover constraint never binds** — kept turnover is in
  the \$10 B/day range.

## How to run

```bash
set PYTHONIOENCODING=utf-8          # Windows: UTF-8 for the polars table glyphs
set LIQ_DATA_DIR=C:\path\to\data    # folder containing binance_trades/, ... 

# tune one symbol over the train split:
python tune.py 2025-12-01 2026-02-01 btc
python tune.py 2025-12-01 2026-02-01 eth

# evaluate on the validation month (one symbol per process):
python run_baseline.py 2026-02-01 2026-03-01 btc
python run_baseline.py 2026-02-01 2026-03-01 eth
```

> Note: on Windows, loading two symbols (or a 2-month slice) in a single process
> can exhaust memory and segfault, so evaluation walks one week at a time and
> runs one symbol per process; the printed metrics are exact pooled values.

## Files

| file | role |
|---|---|
| [baseline.py](baseline.py) | `classify_trades` (submission fn), `compute_pnl`, `evaluate`, tuned `DEFAULT_PARAMS` |
| [data_io.py](data_io.py)   | lazy, time-windowed parquet loading |
| [tune.py](tune.py)         | pooled `(window, threshold)` grid-search per symbol/horizon |
| [run_baseline.py](run_baseline.py) | pooled metric report for any date range/symbol |

## Next steps (not in this baseline)

Add book-imbalance and sweep-size features, separate cascade *middle* from *end*,
and replace the threshold rule with a notional-weighted classifier.
