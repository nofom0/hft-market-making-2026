"""Hyperparameter optimizer for AS strategies using Optuna.

Usage:
    python optimize.py --config configs/baseline.yaml --n-trials 50
    python optimize.py --config configs/microprice.yaml --n-trials 50 --study-name mp_study

The objective is total_pnl (mark-to-market at end of window).
Each trial runs a full backtest in-process — no report files are written.

After optimization the best config is saved to:
    configs/<run_name>_best.yaml

A study plot (optimization history + param importances) is saved to:
    reports/optuna_<run_name>/
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time

import optuna
import yaml

from backtest.data_loader import event_stream
from backtest.engine import Engine
from backtest.metrics import Metrics
from strategies.avellaneda_stoikov import AvellanedaStoikov
from strategies.avellaneda_microprice import AvellanedaMicroprice


# ──────────────────────────────────────────────
# Core: run one backtest without writing a report
# ──────────────────────────────────────────────

def _build_strategy(s: dict, sim: dict, t_start_us: int, t_end_us: int):
    common = dict(
        gamma=s["gamma"],
        k=s["k"],
        sigma_window_sec=s["sigma_window_sec"],
        quote_size=s["quote_size"],
        q_max=s["q_max"],
        risk_horizon_sec=s.get("risk_horizon_sec", 300.0),
        max_inv_skew_ticks=s.get("max_inv_skew_ticks", 0.0),
        tick_size=sim["tick_size"],
        t_start_us=t_start_us,
        t_end_us=t_end_us,
    )
    kind = s["kind"]
    if kind == "avellaneda_stoikov":
        return AvellanedaStoikov(**common)
    if kind == "avellaneda_microprice":
        return AvellanedaMicroprice(beta=s["beta"], **common)
    raise ValueError(f"unknown strategy: {kind}")


def run_backtest(cfg: dict) -> dict:
    """Run backtest from cfg dict; return metrics summary. No disk I/O."""
    data = cfg["data"]
    sim = cfg["sim"]
    fees = cfg.get("fees", {})
    s = cfg["strategy"]

    start_ts = data.get("start_ts")
    end_ts = data.get("end_ts")

    metrics = Metrics(
        fee_maker=float(fees.get("maker", 0.0)),
        fee_taker=float(fees.get("taker", 0.0)),
    )
    strategy = _build_strategy(s, sim,
                                t_start_us=start_ts or 0,
                                t_end_us=end_ts or 10**18)
    engine = Engine(strategy=strategy,
                    latency_us=sim["latency_us"],
                    metrics=metrics,
                    progress_every=0)   # silence progress output

    stream = event_stream(data["lob_path"], data["trades_path"],
                          start_ts=start_ts, end_ts=end_ts)
    engine.run(stream)
    return metrics.summarize()


# ──────────────────────────────────────────────
# Search space
# ──────────────────────────────────────────────

def _suggest_params(trial: optuna.Trial, kind: str, fixed: dict) -> dict:
    """Return a strategy sub-dict with Optuna-suggested values."""
    p: dict = dict(fixed)   # copy fixed fields (kind, quote_size, q_max, tick)

    p["gamma"] = trial.suggest_float("gamma", 0.5, 20.0, log=True)
    p["k"] = trial.suggest_float("k", 1e5, 1e8, log=True)
    p["sigma_window_sec"] = trial.suggest_int("sigma_window_sec", 10, 300)
    p["risk_horizon_sec"] = trial.suggest_int("risk_horizon_sec", 30, 1800)
    p["max_inv_skew_ticks"] = trial.suggest_int("max_inv_skew_ticks", 0, 20)

    if kind == "avellaneda_microprice":
        p["beta"] = trial.suggest_float("beta", 1e-8, 1e-5, log=True)

    return p


# ──────────────────────────────────────────────
# Objective
# ──────────────────────────────────────────────

def make_objective(base_cfg: dict):
    kind = base_cfg["strategy"]["kind"]
    # Fields kept fixed across trials (sizing, paths etc.)
    fixed = {
        "kind": kind,
        "quote_size": base_cfg["strategy"]["quote_size"],
        "q_max": base_cfg["strategy"]["q_max"],
    }
    if kind == "avellaneda_microprice" and "beta" not in fixed:
        pass  # beta will be suggested

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        cfg["strategy"] = _suggest_params(trial, kind, fixed)

        try:
            summary = run_backtest(cfg)
        except Exception as e:
            # Mark failed trial; pruned trials don't affect the study
            trial.set_user_attr("error", str(e))
            raise optuna.exceptions.TrialPruned()

        pnl = summary["total_pnl"]

        # Guard against NaN/Inf (e.g. sigma2 == 0 at start)
        if not math.isfinite(pnl):
            raise optuna.exceptions.TrialPruned()

        # Log extra attrs for post-analysis
        trial.set_user_attr("sharpe", summary["sharpe_annualized_1s"])
        trial.set_user_attr("max_drawdown", summary["max_drawdown"])
        trial.set_user_attr("n_fills", summary["n_fills"])
        trial.set_user_attr("max_abs_inv", summary["max_abs_inventory"])
        trial.set_user_attr("turnover", summary["turnover"])
        trial.set_user_attr("cash", summary["cash"])

        return pnl

    return objective


# ──────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────

def _save_best_config(study: optuna.Study, base_cfg: dict, out_path: str) -> None:
    best = study.best_trial
    cfg = copy.deepcopy(base_cfg)
    kind = cfg["strategy"]["kind"]
    fixed = {
        "kind": kind,
        "quote_size": cfg["strategy"]["quote_size"],
        "q_max": cfg["strategy"]["q_max"],
    }
    cfg["strategy"] = dict(fixed)
    cfg["strategy"].update(best.params)
    # round floats for readability
    for k, v in cfg["strategy"].items():
        if isinstance(v, float) and abs(v) > 1e-4:
            cfg["strategy"][k] = round(v, 6)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Best config saved -> {out_path}")


def _save_plots(study: optuna.Study, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Optimization history
        fig, ax = plt.subplots(figsize=(10, 4))
        values = [t.value for t in study.trials if t.value is not None]
        ax.plot(values, marker=".", ms=4, lw=0.8)
        ax.set_xlabel("Trial")
        ax.set_ylabel("total_pnl")
        ax.set_title("Optimization history")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "history.png"), dpi=120)
        plt.close(fig)

        # Param scatter vs PnL (top 4 params)
        completed = [t for t in study.trials if t.value is not None]
        if len(completed) > 5:
            params = list(completed[0].params.keys())[:4]
            fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 4))
            if len(params) == 1:
                axes = [axes]
            for ax, p in zip(axes, params):
                xs = [t.params[p] for t in completed]
                ys = [t.value for t in completed]
                ax.scatter(xs, ys, s=15, alpha=0.6)
                ax.set_xlabel(p)
                ax.set_ylabel("pnl")
                ax.set_title(p)
                ax.grid(True, alpha=0.3)
                if max(xs) / (min(xs) + 1e-30) > 100:
                    ax.set_xscale("log")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "param_scatter.png"), dpi=120)
            plt.close(fig)

    except Exception as e:
        print(f"  [plot error] {e}")


def _print_results(study: optuna.Study) -> None:
    trials = [t for t in study.trials if t.value is not None]
    trials_sorted = sorted(trials, key=lambda t: t.value, reverse=True)

    print("\n" + "=" * 70)
    print(f"Optimization done.  Completed trials: {len(trials)}")
    print(f"Best PnL: {study.best_value:.6f}  (trial #{study.best_trial.number})")
    print("\nBest params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k:25s} = {v}")
    print("\nExtra attrs of best trial:")
    for k, v in study.best_trial.user_attrs.items():
        if isinstance(v, float):
            print(f"  {k:25s} = {v:.4f}")
        else:
            print(f"  {k:25s} = {v}")

    print("\nTop-5 trials:")
    print(f"  {'#':>4}  {'pnl':>10}  {'sharpe':>8}  {'n_fills':>8}  {'max_inv':>8}")
    for t in trials_sorted[:5]:
        print(f"  {t.number:>4}  {t.value:>10.4f}"
              f"  {t.user_attrs.get('sharpe', 0):>8.3f}"
              f"  {t.user_attrs.get('n_fills', 0):>8}"
              f"  {t.user_attrs.get('max_abs_inv', 0):>8.0f}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Base config YAML (data window + fixed params)")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--study-name", default="",
                   help="Optuna study name; defaults to run_name from config")
    p.add_argument("--storage", default="",
                   help="Optuna storage URL for persistence, e.g. sqlite:///study.db")
    p.add_argument("--direction", default="maximize",
                   choices=["maximize", "minimize"])
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    run_name = base_cfg.get("run_name", "run")
    study_name = args.study_name or f"opt_{run_name}"

    out_dir = os.path.join(base_cfg.get("output", {}).get("report_dir", "reports"),
                           f"optuna_{run_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Optuna verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = args.storage or None
    study = optuna.create_study(
        study_name=study_name,
        direction=args.direction,
        storage=storage,
        load_if_exists=bool(storage),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective_fn = make_objective(base_cfg)

    print(f"Optimizing '{run_name}' ({base_cfg['strategy']['kind']}), "
          f"{args.n_trials} trials …")
    t0 = time.time()

    def _trial_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        if trial.value is not None:
            n_done = len([t for t in study.trials if t.value is not None])
            print(f"  trial {trial.number:>4}  pnl={trial.value:+.4f}"
                  f"  fills={trial.user_attrs.get('n_fills', '?'):>4}"
                  f"  max_inv={trial.user_attrs.get('max_abs_inv', '?'):>6.0f}"
                  f"  best={study.best_value:+.4f}"
                  f"  [{n_done}/{args.n_trials}]")

    study.optimize(objective_fn,
                   n_trials=args.n_trials,
                   callbacks=[_trial_callback],
                   show_progress_bar=False)

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.1f} s  ({elapsed / args.n_trials:.1f} s/trial avg)")

    _print_results(study)
    _save_plots(study, out_dir)

    best_cfg_path = os.path.join("configs", f"{run_name}_best.yaml")
    _save_best_config(study, base_cfg, best_cfg_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
