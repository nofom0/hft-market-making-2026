"""CLI entrypoint:
    python run.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
import time

import yaml

from backtest.data_loader import event_stream
from backtest.engine import Engine
from backtest.metrics import Metrics
from backtest.report import write_report
from strategies.avellaneda_stoikov import AvellanedaStoikov
from strategies.avellaneda_microprice import AvellanedaMicroprice


def build_strategy(cfg: dict, t_start_us: int, t_end_us: int):
    s = cfg["strategy"]
    sim = cfg["sim"]
    kind = s["kind"]
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
    if kind == "avellaneda_stoikov":
        return AvellanedaStoikov(**common)
    if kind == "avellaneda_microprice":
        return AvellanedaMicroprice(beta=s["beta"], **common)
    raise ValueError(f"unknown strategy kind: {kind}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-events", type=int, default=0,
                   help="if >0, stop after this many events (smoke testing)")
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = cfg["data"]
    sim = cfg["sim"]
    fees = cfg.get("fees", {})
    out = cfg.get("output", {})

    start_ts = data.get("start_ts")
    end_ts = data.get("end_ts")

    metrics = Metrics(
        fee_maker=float(fees.get("maker", 0.0)),
        fee_taker=float(fees.get("taker", 0.0)),
    )
    strategy = build_strategy(cfg, t_start_us=start_ts or 0,
                              t_end_us=end_ts or 10**18)
    engine = Engine(strategy=strategy, latency_us=sim["latency_us"],
                    metrics=metrics)

    events_iter = event_stream(data["lob_path"], data["trades_path"],
                               start_ts=start_ts, end_ts=end_ts)

    if args.max_events > 0:
        def limited():
            for i, ev in enumerate(events_iter):
                if i >= args.max_events:
                    break
                yield ev
        stream = limited()
    else:
        stream = events_iter

    print(f"Running '{cfg.get('run_name', 'run')}' "
          f"strategy={cfg['strategy']['kind']} "
          f"window=[{start_ts}, {end_ts})")
    t0 = time.time()
    engine.run(stream)
    elapsed = time.time() - t0

    summary = metrics.summarize()
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f} s. Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    folder = write_report(metrics,
                          out_dir=out.get("report_dir", "reports"),
                          run_name=cfg.get("run_name", "run"),
                          config_snapshot=cfg)
    print(f"Report written to: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
