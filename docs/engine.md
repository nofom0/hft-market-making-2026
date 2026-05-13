# Engine internals

## Event stream

`backtest/data_loader.py` reads `MD/lob.csv` and `MD/trades.csv` in pandas chunks and sort-merges them on `local_timestamp` (microseconds). The result is a single iterator of typed events:

- `SnapEvent(ts, ask_p, ask_a, bid_p, bid_a)` — four numpy arrays with the 25-deep snapshot.
- `TradeEvent(ts, side, price, amount)` — `side ∈ {buy, sell}` is the aggressor.

On a tie (`snap.ts == trade.ts`) the snapshot is yielded first: the snapshot represents state "as of" that ts, and a trade with the same ts logically happens just after.

## Order book maintenance

`backtest/lob.py` keeps two fixed-length numpy arrays (asks ascending, bids descending). Every `SnapEvent` overwrites the book completely (`apply_snapshot`).

`apply_trade(side, price, amount)` patches the book between snapshots, per the agreed rules:

- **Aggressive buy at `price > best_ask`** — sweep all ask levels with `level_price ≤ trade_price`, then subtract `amount` from the new best ask. If the new best ask amount goes ≤ 0, drop that level too.
- **Aggressive sell at `price < best_bid`** — symmetric for the bid side.
- **Trade inside the book** (no sweep) — only the touched best level loses `amount`; if the result is ≤ 0, the level is dropped.

The method returns a `TradeDelta(consumed_ask_levels, consumed_bid_levels)`. Both lists are price values of levels that were exhausted and removed in this trade. The order manager uses them to decide fills.

The book also exposes:

- `mid()` — `(best_bid + best_ask) / 2`.
- `microprice()` — `(P_a · Q_b + P_b · Q_a) / (Q_a + Q_b)`.
- `imbalance()` — `Q_b / (Q_a + Q_b)`.

## Resting orders & fill logic

`backtest/orders.py`:

- We hold up to two resting orders at once: one bid, one ask. No partial fills, no own-volume tracking on a level.
- A fill is triggered in two situations:
  1. **On a snapshot** — the new opposite-side best reaches our price or better:
     - bid fills if `new_best_ask ≤ our_bid_price`;
     - ask fills if `new_best_bid ≥ our_ask_price`.
  2. **On a trade** — the trade fully consumed the level our order sits on (the level price appears in `consumed_*_levels`).
- Fill price is **our limit price** (we are always a maker in this simulator). Fees: `cash -= maker_rate · price · qty` (negative rate is a rebate).

## Latency

`backtest/engine.py` keeps a min-heap `_action_queue` of pending strategy actions. When the strategy decides at ts `t` to place or cancel something, the action is scheduled with `ready_ts = t + latency_us`. Before each event the engine drains all actions with `ready_ts ≤ event.ts` into the order manager.

The practical consequence: between the strategy decision and order arrival, the LOB can change and trades can fly past — none of those interact with a quote that hasn't arrived yet. With `latency_us = 1000` (1 ms) this matches the spec.

## Main loop

```
for event in stream:
    drain actions with ready_ts <= event.ts
    if SnapEvent:
        lob.apply_snapshot(event)
        fills = order_manager.match_on_snap(lob, event.ts)
        record fills, record (mid, inv) sample
        new_quotes = strategy.on_book(lob, event.ts, inv)
        schedule cancel(old)+place(new) at event.ts + latency
    elif TradeEvent:
        delta = lob.apply_trade(event)
        fills = order_manager.match_on_trade(delta, event.ts)
        record fills
```

## Metrics & report

`backtest/metrics.py` accumulates online:

- `cash`, `inv`, `last_mid`, `total_pnl = cash + inv * last_mid`;
- per-fill record (ts, side, price, qty, liquidity, fee);
- 1-second-sampled equity curve `(ts, pnl, inv, mid)`;
- aggregates: `n_fills`, `n_fills_bid/ask`, `max|inv|`, `avg|inv|`, `turnover`, `max_drawdown`, annualised Sharpe over 1-second PnL diffs.

`backtest/report.py` writes `summary.json`, `fills.csv`, `equity.csv` and three PNGs (PnL, inventory, mid+fills) into `reports/<run_name>_<timestamp>/`.

## Edge cases handled

- LOB rows with `NaN` price/amount or zero amount are skipped on snapshot load.
- `mid()` returns `None` until both sides have been seen — the strategy is not asked for quotes before that.
- The strategy refuses to cross the book: a `bid` ≥ `best_ask` is pulled to `best_ask − tick`, symmetric for `ask`.
- The inventory cap stops quoting on one side rather than producing a market order.

## Limitations

- No partial fills; a single trade fully fills our entire order if the level is consumed.
- We are always treated as a maker — fills are at our limit price (no price improvement when the opposite best blows past us between snapshots).
- Constant 1 ms latency, no jitter or distribution.
- Volatility / `k` are estimated/fixed offline; no online recalibration of `k`.
- End of run leaves the position open, marked-to-market by the last seen mid (no force-flatten).
