# Backtester + Avellaneda–Stoikov

Event-driven backtester for one-instrument market-making, with two reference strategies:

1. **Avellaneda–Stoikov 2008** baseline (`avellaneda_stoikov`).
2. **AS + microprice + imbalance-skew** extension (`avellaneda_microprice`, 2018-style).

The engine replays historical L25 LOB snapshots and trades, simulates resting maker orders with configurable latency, and produces a report (PnL, inventory, turnover, fills, plots).

## Install

```powershell
python -m pip install -r requirements.txt
```

Tested on Python 3.12.

## Data

Place the source CSVs in `MD/`:
- `MD/lob.csv` — L25 snapshots, columns `local_timestamp, asks[i].price, asks[i].amount, bids[i].price, bids[i].amount` for `i=0..24`. Microsecond timestamps.
- `MD/trades.csv` — `local_timestamp, side, price, amount`. `side ∈ {buy, sell}` is the **aggressor** side.

Both files are streamed in chunks; the full ~1.9 GB dataset never sits in RAM.

## Run

```powershell
python run.py --config configs/baseline.yaml
python run.py --config configs/microprice.yaml
```

Each run prints progress every 500k events and writes a fresh report folder under `reports/<run_name>_<timestamp>/`:

```
summary.json       all aggregate metrics + the config snapshot
fills.csv          one row per executed maker fill
equity.csv         1-second-sampled (ts, total_pnl, inventory, mid)
pnl.png            PnL curve
inventory.png      inventory curve
mid_with_fills.png mid + bid/ask fill markers
```

To smoke-test on a small window (5 min) without burning a full day:

```powershell
python run.py --config configs/smoke.yaml
```

## Config schema

```yaml
run_name: <string used as the report-folder prefix>

data:
  lob_path: MD/lob.csv
  trades_path: MD/trades.csv
  start_ts: <int microseconds, null = from beginning>
  end_ts:   <int microseconds, null = to end>

sim:
  latency_us: 1000          # place/cancel latency in microseconds
  tick_size: 0.0000001      # price grid

fees:
  maker: 0.0                # positive = fee, negative = rebate (fraction of notional)
  taker: 0.0

strategy:
  kind: avellaneda_stoikov | avellaneda_microprice
  gamma: 0.1
  k: 15000000
  sigma_window_sec: 60      # EWMA half-life of the volatility estimate
  quote_size: 1000          # per-side order size (base units, like `amount`)
  q_max: 10000              # inventory hard cap (one-side stop quoting at the cap)
  beta: 0.0000005           # microprice variant only — imbalance-skew strength

output:
  report_dir: reports
```

## Comparing the two strategies

After running both configs, populate this table from each `summary.json`:

| Metric             | AS-2008 baseline | AS + microprice |
|--------------------|------------------|-----------------|
| total_pnl          |                  |                 |
| n_fills            |                  |                 |
| max_abs_inventory  |                  |                 |
| avg_abs_inventory  |                  |                 |
| turnover           |                  |                 |
| sharpe_annualized_1s |                |                 |
| max_drawdown       |                  |                 |

A short summary of what changed and why goes in `docs/strategy.md`.

## Repository layout

See [the plan file](../) for the agreed structure. In short:

```
backtest/   data_loader, lob, orders, engine, metrics, report
strategies/ base, avellaneda_stoikov, avellaneda_microprice
configs/    baseline.yaml, microprice.yaml, smoke.yaml
docs/       README.md, engine.md, strategy.md
run.py      CLI
```

## Further reading

- Engine internals — see [engine.md](engine.md).
- Strategy math, parameters, and improvement roadmap — see [strategy.md](strategy.md).
