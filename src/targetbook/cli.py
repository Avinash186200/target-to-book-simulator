"""targetbook bench | report | demo"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import bench as B


def _demo(seed: int, signal: str) -> None:
    from .pipeline import LADDER, prepare, run_signal

    res_dir = Path(__file__).resolve().parents[2] / "results"
    dev_file = res_dir / "dev_choices.json"
    if dev_file.exists():
        desk = B.frozen_desk(json.loads(dev_file.read_text(encoding="utf-8")))
    else:  # pragma: no cover - installed without the results folder
        from .pipeline import DeskConfig
        desk = DeskConfig()
    prep = prepare(seed, desk=desk)
    out = run_signal(prep, signal, desk, sweeps=False)
    c = out["configs"]
    print(f"seed {seed}, {signal} signal, desk schedule {desk.schedule.label}, band {desk.band:g} ({desk.band_mode})")
    print(f"{'stage':<10} {'return %':>9} {'Sharpe':>7} {'vol %':>6} {'gross':>6} {'turn/yr':>8} {'delta %':>8}")
    prev = None
    for st in LADDER:
        s = c[f"ladder/{st}"]
        delta = "" if prev is None else f"{100 * (s['ann_return'] - prev):+8.2f}"
        print(f"{st:<10} {100 * s['ann_return']:9.2f} {s['sharpe']:7.2f} {100 * s['ann_vol']:6.2f} "
              f"{s['avg_gross']:6.2f} {s['turnover']:8.1f} {delta:>8}")
        prev = s["ann_return"]
    phases = [c[f"luck/single_phase{p}"]["sharpe"] for p in range(4)]
    print("single book every 4 weeks, Sharpe by phase: " + ", ".join(f"{x:.2f}" for x in phases)
          + f"; staggered: {c['luck/staggered']['sharpe']:.2f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="targetbook")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="development choices, then held-out seeds; writes results/")
    b.add_argument("--out", default="results")
    b.add_argument("--workers", type=int, default=1, help="processes (results do not depend on it)")
    r = sub.add_parser("report", help="recompute summaries and RESULTS.md from results/runs.json")
    r.add_argument("--out", default="results")
    d = sub.add_parser("demo", help="one seed, print the waterfall")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--signal", choices=["model", "control"], default="model")
    a = ap.parse_args(argv)
    if a.cmd == "bench":
        B.run_bench(Path(a.out), workers=a.workers)
    elif a.cmd == "report":
        out = Path(a.out)
        B.write_all(json.loads((out / "runs.json").read_text(encoding="utf-8")), out)
        print(f"rewrote summaries in {out}")
    else:
        _demo(a.seed, a.signal)


if __name__ == "__main__":
    main()
