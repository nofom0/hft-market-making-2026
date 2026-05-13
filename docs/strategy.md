# Strategies: Avellaneda–Stoikov 2008 + microprice extension

## 1. AS-2008 baseline

Reference: Avellaneda & Stoikov, *High-Frequency Trading in a Limit Order Book*, 2008.

Setup:
- The mid-price is modelled as Brownian motion `dS = σ dW`.
- The market-maker holds inventory `q`, has horizon `T` and risk-aversion `γ`, and faces an exponential order-arrival intensity `λ(δ) = A · exp(−k · δ)` at half-spread `δ`.

Solving the HJB equation gives a closed form:

**Reservation price** (the maker's indifference price given inventory):
```
r(s, q, t) = s − q · γ · σ² · (T − t)
```

**Optimal half-spread**:
```
δ* = ½ · γ · σ² · (T − t) + (1 / γ) · ln(1 + γ / k)
```

**Quotes**:
```
bid = round_floor( r − δ*, tick )
ask = round_ceil ( r + δ*, tick )
```

The quotes are symmetric around `r`, so the inventory term shifts both quotes down when long and up when short — in expectation this brings the position back toward zero.

### Parameters

| Symbol | Source | Role |
|---|---|---|
| γ | hyper, configured | Risk aversion. Larger γ ⇒ wider spread + stronger inventory pull. |
| σ | online (EWMA of squared log-returns of mid, half-life `sigma_window_sec`) | Mid-price volatility per second. |
| k | hyper, configured (default `1.5 / tick_size`) | Decay of arrival intensity with depth. Larger k ⇒ tighter quotes. |
| T − t | from config window (`end_ts − ts`) | Time to horizon, in seconds. |
| q | runtime state | Signed inventory in base units. |
| `quote_size` | configured | Per-side order size. |
| `q_max` | configured | Hard cap; one side stops quoting at the cap. |
| `tick_size` | configured | Price grid for rounding. |

### Implementation notes

- Volatility tracker: EWMA of `(log(mid_t) − log(mid_{t−1}))² / Δt_sec`, with half-life `sigma_window_sec`. Initialised from the first observation.
- We always round outward (floor for bid, ceil for ask) so the resulting `δ` ≥ the formula minimum after rounding.
- A defensive clamp prevents the quote from crossing the book when the formula puts it through the spread.
- The inventory cap is implemented as "stop quoting the side that would worsen `|q|`" (i.e. no bid when `q ≥ q_max`).

## 2. AS + microprice + imbalance-skew (2018-style)

Two changes vs. the baseline.

### 2.1 Microprice replaces mid

Following Stoikov 2018 (*The Microprice: A High-Frequency Estimator of Future Prices*), we replace `s` with the simple weighted microprice:

```
P_micro = (P_ask · Q_bid + P_bid · Q_ask) / (Q_ask + Q_bid)
```

`P_micro` lies between best bid and best ask, leaning toward the side with more queued volume on the **other** side — i.e. the price more likely to clear next. Replacing the mid with microprice nudges the reservation price (and therefore both quotes) in the predicted-direction-of-drift, which reduces adverse selection on the side that's about to be hit.

### 2.2 Imbalance skew on the reservation price

```
I = Q_bid / (Q_ask + Q_bid)            # ∈ [0, 1], 0.5 = balanced
r = P_micro − q · γ · σ² · (T − t) + β · (I − 0.5)
```

- `I > 0.5` (more bid liquidity at the top) lifts both quotes — short-term direction is up.
- `I < 0.5` lowers both quotes.
- `β` controls the strength of the skew. Default `5 · tick_size`.

This is mathematically equivalent to a small horizontal shift of the whole AS quote pair, so it preserves the symmetric `[r − δ, r + δ]` structure.

### Why this should help

- **Microprice** captures the "next mid" better than the simple mid, so quotes become more anticipatory and less reactive.
- **Imbalance skew** explicitly bakes a short-term predictor into the reservation price, on top of the inventory term that AS-2008 already has.

The price for both is one extra hyper-parameter (`β`), kept in the config.

## 3. Improvement roadmap

Pieces deliberately out of scope for this assignment, ordered by expected payoff:

1. **Partial fills + queue-position tracking** — match against actual queued volume on the level instead of always-fill. Allows realistic `k` calibration.
2. **Online `k` estimator** — fit `λ(δ) = A · e^{−k δ}` from a rolling window of trade-arrival distances to the best, and feed live `k` into the spread formula.
3. **Adaptive `γ`** — make risk aversion respond to recent volatility regime (e.g. `γ_t = γ_0 · σ̄/σ_t`).
4. **Better volatility model** — replace the EWMA with HAR or GARCH on microprice returns.
5. **Full Stoikov-2018 microprice** — fit the G-matrix on (imbalance, spread) bins offline, use as `S_t` (out-of-sample only).
6. **Short-term predictors** — Order-Flow Imbalance (OFI), trade-flow imbalance — added as a small drift term to `r`.
7. **Soft inventory penalty** — replace the hard `q_max` cut with a continuous penalty (e.g. quadratic in `q/q_max`) that smoothly skews quotes near the cap.
8. **Latency model upgrades** — sample latency from a distribution (lognormal), distinguish place vs. cancel.
9. **Multi-size laddering** — quote multiple price levels (e.g. inner and outer) sized differently, instead of one quote per side.
10. **Force-flatten at end of horizon** — close residual inventory via a taker leg and report realised PnL only.
