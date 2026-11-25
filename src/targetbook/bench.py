"""Benchmark: choices on development seeds, then held-out seeds with those choices frozen.

``run_bench`` writes, under ``results/``:

* ``runs.json``: every per-seed record (development and held-out);
* ``dev_choices.json``, ``waterfall.json``, ``timing.json``, ``repair.json``,
  ``bands.json``: summaries recomputed from ``runs.json`` alone;
* ``RESULTS.md``: rendered from the summaries;
* ``manifest.json``: the frozen configuration, seeds, versions and the SHA-256
  of every file above, so a run can be checked or re-created exactly.
"""
from __future__ import annotations

import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import __version__
from .data import MarketConfig
from .mandate import SOLVER, Mandate
from .pipeline import (
    ALONE, BAND_GRID, COST_MULTS, LADDER, LUCK_PERIOD, SCHEDULES, DeskConfig, band_configs,
    dev_task, heldout_task, with_choices,
)

DEV_SEEDS = (1, 2, 3, 4, 5)
HELDOUT_SEEDS = tuple(range(101, 111))
SUMMARY_FILES = ("dev_choices", "waterfall", "timing", "repair", "bands")
SIGNALS = ("model", "control")


# ---------------------------------------------------------------- helpers

def _round(x):
    if isinstance(x, float):
        return float(f"{x:.12g}")
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(f"{float(x):.12g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _dist(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "n": int(v.size)}


def _dumps(obj) -> str:
    return json.dumps(_round(obj), indent=1, sort_keys=True) + "\n"


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))  # LF only on every platform


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- running

def _map(fn, args, workers: int):
    if workers <= 1:
        return [fn(*a) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, *zip(*args)))


def run_dev(workers: int = 1) -> list[dict]:
    return _map(dev_task, [(s,) for s in DEV_SEEDS], workers)


def run_heldout(desk: DeskConfig, cost_bands: dict, workers: int = 1) -> list[dict]:
    return _map(heldout_task, [(s, desk, cost_bands) for s in HELDOUT_SEEDS], workers)


# ---------------------------------------------------------------- summaries

def summarize_dev(runs: dict) -> dict:
    """Pick schedule, band and band mode by mean development Sharpe at the scenario's
    cost level; then, with that schedule, the best band at each cost multiplier."""
    dev = runs["dev"]

    def mean_se(rows):
        v = np.array([r["sharpe"] for r in rows])
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))

    grid = []
    for n, per in SCHEDULES:
        for b, m in band_configs():
            rows = [r for d in dev for r in d["rows"] if r["n_tranches"] == n and r["period_weeks"] == per
                    and r["band"] == b and r["band_mode"] == m and r["cost_mult"] == 1.0]
            mu, se = mean_se(rows)
            grid.append({"n_tranches": n, "period_weeks": per, "band": b, "band_mode": m,
                         "sharpe_mean": mu, "sharpe_se": se,
                         "turnover_mean": float(np.mean([r["turnover"] for r in rows]))})
    best = max(grid, key=lambda g: g["sharpe_mean"])
    chosen = {k: best[k] for k in ("n_tranches", "period_weeks", "band", "band_mode")}

    cost_grid, cost_bands = [], {}
    for mult in COST_MULTS:
        cells = []
        for b, m in band_configs():
            rows = [r for d in dev for r in d["rows"] if r["n_tranches"] == chosen["n_tranches"]
                    and r["period_weeks"] == chosen["period_weeks"] and r["band"] == b
                    and r["band_mode"] == m and r["cost_mult"] == mult]
            mu, se = mean_se(rows)
            cells.append({"cost_mult": mult, "band": b, "band_mode": m, "sharpe_mean": mu, "sharpe_se": se})
        cost_grid += cells
        top = max(cells, key=lambda c: c["sharpe_mean"])
        cost_bands[f"{mult:g}"] = [top["band"], top["band_mode"]]
    sched_best = []
    for n, per in SCHEDULES:
        cells = [g for g in grid if g["n_tranches"] == n and g["period_weeks"] == per]
        sched_best.append(max(cells, key=lambda g: g["sharpe_mean"]))
    return {
        "seeds": [d["seed"] for d in dev],
        "paper_sharpe": [d["paper_sharpe"] for d in dev],
        "grid": grid,
        "best_per_schedule": sched_best,
        "chosen": chosen,
        "cost_grid": cost_grid,
        "cost_bands": cost_bands,
    }


def _cfg(rec, sig, name, field):
    return rec[sig]["configs"][name][field]


def summarize_waterfall(runs: dict) -> dict:
    held = runs["heldout"]
    out = {}
    for sig in SIGNALS:
        stages = []
        for i, st in enumerate(LADDER):
            name = f"ladder/{st}"
            row = {"stage": st}
            for f in ("ann_return", "sharpe", "ann_vol", "avg_gross", "turnover", "cost_drag", "carry_drag"):
                row[f] = _dist([_cfg(r, sig, name, f) for r in held])
            if i:
                prev = f"ladder/{LADDER[i - 1]}"
                for f in ("ann_return", "sharpe"):
                    d = [_cfg(r, sig, name, f) - _cfg(r, sig, prev, f) for r in held]
                    row[f"delta_{f}"] = _dist(d)
                    row[f"delta_{f}_n_negative"] = int(sum(x < 0 for x in d))
            stages.append(row)
        alone = []
        for st in ALONE:
            name = "ladder/mandate" if st == "mandate" else f"alone/{st}"
            row = {"step": st}
            for f in ("ann_return", "sharpe"):
                d = [_cfg(r, sig, name, f) - _cfg(r, sig, "ladder/paper", f) for r in held]
                row[f"delta_{f}"] = _dist(d)
                row[f"delta_{f}_n_negative"] = int(sum(x < 0 for x in d))
            alone.append(row)
        paper = [_cfg(r, sig, "ladder/paper", "ann_return") for r in held]
        real = [_cfg(r, sig, "ladder/bands", "ann_return") for r in held]
        total = [b - a for a, b in zip(paper, real)]
        out[sig] = {
            "stages": stages,
            "alone": alone,
            "total_delta_ann_return": _dist(total),
            "pooled_share_of_paper_return_kept": float(np.mean(real) / np.mean(paper)),
            "real_tracking_error": _dist([_cfg(r, sig, "ladder/bands", "tracking_error") for r in held]),
            "real_corr_to_paper": _dist([_cfg(r, sig, "ladder/bands", "corr_to_paper") for r in held]),
            "real_n_repairs": _dist([_cfg(r, sig, "ladder/bands", "n_repairs") for r in held]),
            "real_max_gross": _dist([_cfg(r, sig, "ladder/bands", "max_gross") for r in held]),
            "real_share_days_over_gross_cap": _dist([_cfg(r, sig, "ladder/bands", "share_days_over_gross_cap")
                                                     for r in held]),
            "real_tranche_rebalances": _dist([_cfg(r, sig, "ladder/bands", "n_tranche_rebalances") for r in held]),
        }
    return {"seeds": [r["seed"] for r in held], **out}


def summarize_timing(runs: dict) -> dict:
    held = runs["heldout"]
    out = {"period_weeks": LUCK_PERIOD}
    for sig in SIGNALS:
        phases = [f"luck/single_phase{p}" for p in range(LUCK_PERIOD)]
        res = {}
        for f in ("ann_return", "sharpe"):
            per_seed = [[_cfg(r, sig, ph, f) for ph in phases] for r in held]
            spread = [max(v) - min(v) for v in per_seed]
            stag = [_cfg(r, sig, "luck/staggered", f) for r in held]
            mean_ph = [float(np.mean(v)) for v in per_seed]
            res[f] = {
                "phase_spread": _dist(spread),
                "single_phase": _dist([x for v in per_seed for x in v]),
                "staggered": _dist(stag),
                "staggered_minus_mean_phase": _dist([a - b for a, b in zip(stag, mean_ph)]),
                "weekly": _dist([_cfg(r, sig, "luck/weekly", f) for r in held]),
            }
        res["ann_vol_single"] = _dist([_cfg(r, sig, ph, "ann_vol") for r in held for ph in phases])
        res["ann_vol_staggered"] = _dist([_cfg(r, sig, "luck/staggered", "ann_vol") for r in held])
        res["gross_single"] = _dist([_cfg(r, sig, ph, "avg_gross") for r in held for ph in phases])
        res["gross_staggered"] = _dist([_cfg(r, sig, "luck/staggered", "avg_gross") for r in held])
        res["turnover_single"] = _dist([_cfg(r, sig, ph, "turnover") for r in held for ph in phases])
        res["turnover_staggered"] = _dist([_cfg(r, sig, "luck/staggered", "turnover") for r in held])
        res["turnover_weekly"] = _dist([_cfg(r, sig, "luck/weekly", "turnover") for r in held])
        out[sig] = res
    return out


def summarize_repair(runs: dict) -> dict:
    held = runs["heldout"]
    out = {}
    for sig in SIGNALS:
        diag = {k: _dist([r[sig]["diag"][k] for r in held]) for k in held[0][sig]["diag"]}
        pairs = {}
        for f in ("ann_return", "sharpe", "avg_gross", "tracking_error"):
            proj = [_cfg(r, sig, "ladder/mandate", f) for r in held]
            scl = [_cfg(r, sig, "repair/scale_only", f) for r in held]
            pairs[f"mandate_only_{f}"] = {"projection": _dist(proj), "scaling": _dist(scl),
                                          "projection_minus_scaling": _dist([a - b for a, b in zip(proj, scl)])}
            proj = [_cfg(r, sig, "ladder/bands", f) for r in held]
            scl = [_cfg(r, sig, "repair/scale_real", f) for r in held]
            pairs[f"real_{f}"] = {"projection": _dist(proj), "scaling": _dist(scl),
                                  "projection_minus_scaling": _dist([a - b for a, b in zip(proj, scl)])}
        out[sig] = {"diag": diag, "pairs": pairs}
    out["solver"] = {k: int(sum(r["solver"][k] for r in held)) for k in ("solves", "almost_solved", "fallbacks")}
    out["solver"]["name"] = SOLVER
    return out


def summarize_bands(runs: dict) -> dict:
    held = runs["heldout"]
    dev = summarize_dev(runs)
    rows = []
    for b, m in band_configs():
        name = f"bands/{b:g}/{m}"
        row = {"band": b, "band_mode": m}
        for f in ("ann_return", "sharpe", "turnover", "cost_drag", "tracking_error", "n_repairs"):
            row[f] = _dist([_cfg(r, "model", name, f) for r in held])
        d = [_cfg(r, "model", name, "sharpe") - _cfg(r, "model", "bands/0/edge", "sharpe") for r in held]
        row["delta_sharpe_vs_no_band"] = _dist(d)
        row["n_better_than_no_band"] = int(sum(x > 0 for x in d))
        rows.append(row)
    cost = []
    for mult in COST_MULTS:
        ch = [_cfg(r, "model", f"cost/{mult:g}/chosen", "sharpe") for r in held]
        nb = [_cfg(r, "model", f"cost/{mult:g}/no_band", "sharpe") for r in held]
        d = [a - b for a, b in zip(ch, nb)]
        cost.append({"cost_mult": mult, "band": dev["cost_bands"][f"{mult:g}"][0],
                     "band_mode": dev["cost_bands"][f"{mult:g}"][1],
                     "sharpe_chosen": _dist(ch), "sharpe_no_band": _dist(nb),
                     "delta": _dist(d), "n_better": int(sum(x > 0 for x in d)),
                     "turnover_chosen": _dist([_cfg(r, "model", f"cost/{mult:g}/chosen", "turnover") for r in held]),
                     "turnover_no_band": _dist([_cfg(r, "model", f"cost/{mult:g}/no_band", "turnover") for r in held])})
    # Without bands every rebalance trades to the target, so the two measures differ only
    # by drift: |target_k - drifted holdings| versus |target_k - target_(k-1)|.
    turn = [_cfg(r, "model", "ladder/carry", "turnover") for r in held]
    naive = [_cfg(r, "model", "ladder/carry", "turnover_vs_targets") for r in held]
    cdrift = [_cfg(r, "model", "ladder/carry", "cost_drag") for r in held]
    bturn = [_cfg(r, "model", "ladder/bands", "turnover") for r in held]
    bnaive = [_cfg(r, "model", "ladder/bands", "turnover_vs_targets") for r in held]
    return {"grid": rows, "cost_sweep": cost, "band_grid": list(BAND_GRID),
            "turnover_measure": {
                "no_band_drifted": _dist(turn), "no_band_vs_targets": _dist(naive),
                "drifted_minus_vs_targets": _dist([a - b for a, b in zip(turn, naive)]),
                "drifted_over_targets": _dist([a / b for a, b in zip(turn, naive)]),
                "cost_drag_drifted": _dist(cdrift),
                "real_drifted": _dist(bturn), "real_vs_targets": _dist(bnaive),
                "real_drifted_over_targets": _dist([a / b for a, b in zip(bturn, bnaive)])}}


SUMMARIZERS = {
    "dev_choices": summarize_dev,
    "waterfall": summarize_waterfall,
    "timing": summarize_timing,
    "repair": summarize_repair,
    "bands": summarize_bands,
}


# ---------------------------------------------------------------- rendering

def _pct(d, k="median", nd=2):
    return f"{100 * d[k]:.{nd}f}"


def _rng_pct(d, nd=2):
    return f"{100 * d['median']:.{nd}f} [{100 * d['min']:.{nd}f}, {100 * d['max']:.{nd}f}]"


def _rng(d, nd=2):
    return f"{d['median']:.{nd}f} [{d['min']:.{nd}f}, {d['max']:.{nd}f}]"


STAGE_LABEL = {
    "paper": "paper target (weekly, frictionless)",
    "mandate": "+ mandate repair (projection)",
    "vol_cap": "+ vol cap",
    "schedule": "+ desk rebalance schedule",
    "costs": "+ trading costs (drifted weights)",
    "carry": "+ carry on long and short exposure",
    "bands": "+ turnover bands = real book",
}


def render(res_dir: Path) -> str:
    load = lambda n: json.loads((res_dir / f"{n}.json").read_text(encoding="utf-8"))  # noqa: E731
    dev, wf, tm, rp, bd = (load(n) for n in SUMMARY_FILES)
    ch = dev["chosen"]
    sched = "weekly" if ch["n_tranches"] == 1 and ch["period_weeks"] == 1 else \
        f"{ch['n_tranches']} tranches x {ch['period_weeks']} weeks"
    L = []
    a = L.append
    a("# Results\n")
    a("Synthetic data only, generated by `make bench` (`python -m targetbook.cli bench --out results`).")
    a(f"Development seeds {', '.join(map(str, dev['seeds']))} choose the desk's schedule and bands; "
      f"held-out seeds {wf['seeds'][0]}-{wf['seeds'][-1]} (n = {len(wf['seeds'])}) are reported with those choices frozen; "
      "held-out results were never used for any choice (held-out sections first read after the third full run; a fourth run only added diagnostic fields; "
      "see docs/methodology.md).")
    a("Each seed is an independent 11-year panel (1 warm-up year, 10 evaluated years). "
      "Cells are `median [min, max]` across held-out seeds. Returns are annualised arithmetic excess returns, in %.\n")

    a("## Development choices\n")
    a("Mean net Sharpe over the 5 development seeds with every friction on (scenario cost level), best band per schedule:\n")
    a("| schedule | best band | mode | dev Sharpe (mean ± s.e.) | turnover / yr |")
    a("|---|---|---|---|---|")
    for g in dev["best_per_schedule"]:
        lab = "weekly" if g["n_tranches"] == 1 else f"{g['n_tranches']} tranches x {g['period_weeks']} weeks"
        a(f"| {lab} | {g['band']:g} | {g['band_mode'] if g['band'] else '-'} | {g['sharpe_mean']:.3f} ± {g['sharpe_se']:.3f} | {g['turnover_mean']:.1f} |")
    a(f"\nChosen: **{sched}, band {ch['band']:g}"
      + (f" ({ch['band_mode']} mode)" if ch["band"] else "") + "**. "
      f"Development paper Sharpe per seed: {', '.join(f'{x:.2f}' for x in dev['paper_sharpe'])}.\n")
    a("Development Sharpe by band and cost level with the chosen schedule (mean over 5 seeds; edge / target mode):\n")
    bands = sorted({c["band"] for c in dev["cost_grid"]})
    mults = sorted({c["cost_mult"] for c in dev["cost_grid"]})
    cg = {(c["cost_mult"], c["band"], c["band_mode"]): c["sharpe_mean"] for c in dev["cost_grid"]}
    a("| cost x | " + " | ".join(f"band {b:g}" for b in bands) + " | chosen |")
    a("|---|" + "---|" * (len(bands) + 1))
    for mult in mults:
        cells = [f"{cg[(mult, b, 'edge')]:.3f}" if b == 0 else
                 f"{cg[(mult, b, 'edge')]:.3f} / {cg[(mult, b, 'target')]:.3f}" for b in bands]
        cb = dev["cost_bands"][f"{mult:g}"]
        a(f"| {mult:g} | " + " | ".join(cells) + f" | {cb[0]:g} {cb[1] if cb[0] else ''} |")
    a("")
    a("Full grid at the scenario cost level:\n")
    a("| schedule | band | mode | dev Sharpe | s.e. | turnover / yr |")
    a("|---|---|---|---|---|---|")
    for g in dev["grid"]:
        lab = "weekly" if g["n_tranches"] == 1 else f"{g['n_tranches']}x{g['period_weeks']}w"
        a(f"| {lab} | {g['band']:g} | {g['band_mode'] if g['band'] else '-'} | {g['sharpe_mean']:.3f} | {g['sharpe_se']:.3f} | {g['turnover_mean']:.1f} |")
    a("")

    for sig in SIGNALS:
        w = wf[sig]
        a(f"## Waterfall: {'model signal (planted edge)' if sig == 'model' else 'no-edge control'}\n")
        a("Each row adds one step to the row above. Delta columns are paired per seed (stage minus previous stage).\n")
        a("| stage | return % | Sharpe | vol % | gross | turnover / yr | delta return % | delta Sharpe | seeds with a loss |")
        a("|---|---|---|---|---|---|---|---|---|")
        for s in w["stages"]:
            dr = _rng_pct(s["delta_ann_return"]) if "delta_ann_return" in s else "-"
            dsh = _rng(s["delta_sharpe"]) if "delta_sharpe" in s else "-"
            neg = f"{s['delta_ann_return_n_negative']}/{s['ann_return']['n']}" if "delta_ann_return" in s else "-"
            a(f"| {STAGE_LABEL[s['stage']]} | {_rng_pct(s['ann_return'])} | {_rng(s['sharpe'])} | {_pct(s['ann_vol'])} | "
              f"{s['avg_gross']['median']:.2f} | {s['turnover']['median']:.1f} | {dr} | {dsh} | {neg} |")
        a(f"\nPaper to real, paired: {_rng_pct(w['total_delta_ann_return'])} % per year. "
          + (f"Pooled share of paper return kept (mean real / mean paper): {100 * w['pooled_share_of_paper_return_kept']:.0f}%. "
             if sig == "model" else "")
          + f"Real book tracking error to paper {_rng_pct(w['real_tracking_error'])} %, correlation {_rng(w['real_corr_to_paper'])}; "
          f"mandate repairs after banding {w['real_n_repairs']['median']:.0f} of {w['real_tranche_rebalances']['median']:.0f} tranche rebalances (median). "
          f"The mandate is enforced at rebalances only: between them the real book's gross exceeded the cap on "
          f"{_rng_pct(w['real_share_days_over_gross_cap'])}% of days, with a maximum of {w['real_max_gross']['max']:.3f}.\n")
        a("Each step alone on top of the paper book (paired, minus paper):\n")
        a("| step alone | delta return % | delta Sharpe | seeds with a loss (return) |")
        a("|---|---|---|---|")
        for s in w["alone"]:
            a(f"| {s['step']} | {_rng_pct(s['delta_ann_return'])} | {_rng(s['delta_sharpe'])} | "
              f"{s['delta_ann_return_n_negative']}/{s['delta_ann_return']['n']} |")
        a("")

    a(f"## Rebalance-timing luck (one book every {tm['period_weeks']} weeks vs {tm['period_weeks']} staggered tranches)\n")
    a("Real-book frictions, bands and mandate repair as chosen. `phase spread` is, per seed, the best minus the worst of the "
      f"{tm['period_weeks']} possible rebalance weeks.\n")
    a("| signal | phase spread, return % | phase spread, Sharpe | single book Sharpe (all phases) | staggered Sharpe | staggered minus mean phase, Sharpe | weekly Sharpe |")
    a("|---|---|---|---|---|---|---|")
    for sig in SIGNALS:
        t = tm[sig]
        a(f"| {sig} | {_rng_pct(t['ann_return']['phase_spread'])} | {_rng(t['sharpe']['phase_spread'])} | "
          f"{_rng(t['sharpe']['single_phase'])} | {_rng(t['sharpe']['staggered'])} | "
          f"{_rng(t['sharpe']['staggered_minus_mean_phase'])} | {_rng(t['sharpe']['weekly'])} |")
    a("")
    a("| signal | gross: single / staggered | vol %: single / staggered | turnover / yr: single / staggered / weekly |")
    a("|---|---|---|---|")
    for sig in SIGNALS:
        t = tm[sig]
        a(f"| {sig} | {t['gross_single']['median']:.2f} / {t['gross_staggered']['median']:.2f} | "
          f"{_pct(t['ann_vol_single'])} / {_pct(t['ann_vol_staggered'])} | "
          f"{t['turnover_single']['median']:.1f} / {t['turnover_staggered']['median']:.1f} / {t['turnover_weekly']['median']:.1f} |")
    a("")

    a("## Mandate repair: projection vs uniform scaling (model signal)\n")
    d = rp["model"]["diag"]
    a(f"Targets outside the mandate: {_pct(d['share_targets_outside_mandate'], nd=0)}% of weeks. "
      f"Mean gross: target {d['gross_target']['median']:.2f}, projected {_rng(d['gross_projected'])}, scaled {_rng(d['gross_scaled'])}. "
      f"Mean L2 distance to the target: projected {_rng(d['distance_projected'], 3)}, scaled {_rng(d['distance_scaled'], 3)}. "
      f"Vol cap binding on the projected book: {_rng_pct(d['share_vol_cap_binding_projected'], 0)}% of weeks. "
      f"Mean Ledoit-Wolf shrinkage intensity: {d['mean_shrinkage']['median']:.3f}.\n")
    a("| book | measure | projection | scaling | projection minus scaling |")
    a("|---|---|---|---|---|")
    for pre, lab in (("mandate_only", "mandate only"), ("real", "real book")):
        for f, nm, pct in (("ann_return", "return %", True), ("sharpe", "Sharpe", False),
                           ("avg_gross", "gross", False), ("tracking_error", "tracking error to paper %", True)):
            p = rp["model"]["pairs"][f"{pre}_{f}"]
            fmt = _rng_pct if pct else _rng
            a(f"| {lab} | {nm} | {fmt(p['projection'])} | {fmt(p['scaling'])} | {fmt(p['projection_minus_scaling'])} |")
    s = rp["solver"]
    a(f"\nSolver: {s['name']}, {s['solves']} solves over all held-out seeds; {s['almost_solved']} returned "
      f"reduced-accuracy solutions (polished to feasibility); {s['fallbacks']} fell back to scaling.\n")

    a("## Turnover bands (model signal, chosen schedule, every friction on)\n")
    a("Held-out results for every band in the grid; the band itself was chosen on development seeds.\n")
    a("| band | mode | return % | Sharpe | turnover / yr | cost % / yr | delta Sharpe vs no band | seeds better | repairs |")
    a("|---|---|---|---|---|---|---|---|---|")
    for r in bd["grid"]:
        a(f"| {r['band']:g} | {r['band_mode'] if r['band'] else '-'} | {_pct(r['ann_return'])} | {_rng(r['sharpe'])} | "
          f"{r['turnover']['median']:.1f} | {_pct(r['cost_drag'])} | {_rng(r['delta_sharpe_vs_no_band'], 3)} | "
          f"{r['n_better_than_no_band']}/{r['sharpe']['n']} | {r['n_repairs']['median']:.0f} |")
    a("\nBand chosen on development seeds at each cost level, then scored on held-out seeds:\n")
    a("| cost x | band | mode | Sharpe with band | Sharpe, no band | delta | seeds better | turnover: band / none |")
    a("|---|---|---|---|---|---|---|---|")
    for c in bd["cost_sweep"]:
        a(f"| {c['cost_mult']:g} | {c['band']:g} | {c['band_mode'] if c['band'] else '-'} | {_rng(c['sharpe_chosen'])} | "
          f"{_rng(c['sharpe_no_band'])} | {_rng(c['delta'], 3)} | {c['n_better']}/{c['delta']['n']} | "
          f"{c['turnover_chosen']['median']:.1f} / {c['turnover_no_band']['median']:.1f} |")
    tmz = bd["turnover_measure"]
    a(f"\nTurnover measured on drifted weights vs between consecutive targets (desk schedule, no bands, "
      f"so every rebalance trades fully to the target): {_rng(tmz['no_band_drifted'], 1)} vs "
      f"{_rng(tmz['no_band_vs_targets'], 1)} per year; paired difference {_rng(tmz['drifted_minus_vs_targets'], 2)}, "
      f"ratio {_rng(tmz['drifted_over_targets'], 3)}. "
      f"With the chosen bands the holdings are no longer the last target: the real book trades "
      f"{_rng(tmz['real_drifted'], 1)} per year on drifted weights, while consecutive targets differ by "
      f"{_rng(tmz['real_vs_targets'], 1)} (ratio {_rng(tmz['real_drifted_over_targets'], 3)}).\n")
    return "\n".join(L)


# ---------------------------------------------------------------- top level

def default_mandate() -> Mandate:
    mc = MarketConfig()
    return Mandate(groups=tuple(int(g) for g in np.repeat(np.arange(len(mc.group_names)), mc.per_group)))


def frozen_desk(dev_choices: dict) -> DeskConfig:
    return with_choices(DeskConfig(), dev_choices["chosen"])


def write_all(runs: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "runs.json", _dumps(runs))
    runs = json.loads((out / "runs.json").read_text(encoding="utf-8"))  # summarise the rounded records
    for name in SUMMARY_FILES:
        _write(out / f"{name}.json", _dumps(SUMMARIZERS[name](runs)))
    _write(out / "RESULTS.md", render(out))
    write_manifest(out)


def manifest(out: Path) -> dict:
    dev = json.loads((out / "dev_choices.json").read_text(encoding="utf-8"))
    import clarabel
    import scipy

    return {
        "version": __version__,
        "market": MarketConfig().to_dict(),
        "desk": frozen_desk(dev).to_dict(),
        "mandate": default_mandate().to_dict(),
        "dev_seeds": list(DEV_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "solver": SOLVER,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "clarabel": clarabel.__version__},
        "platform": platform.platform(),
        "sha256": {f: _sha(out / f) for f in ["runs.json", *(f"{n}.json" for n in SUMMARY_FILES), "RESULTS.md"]},
    }


def write_manifest(out: Path) -> None:
    _write(out / "manifest.json", _dumps(manifest(out)))


def run_bench(out: Path, workers: int = 1, log=print) -> None:
    log(f"development seeds {DEV_SEEDS} ...")
    dev = json.loads(_dumps(run_dev(workers)))  # choose from the same rounded records that are saved
    choices = summarize_dev({"dev": dev})
    desk = frozen_desk(choices)
    log(f"chosen: {choices['chosen']}; per cost level: {choices['cost_bands']}")
    log(f"held-out seeds {HELDOUT_SEEDS} ...")
    held = run_heldout(desk, choices["cost_bands"], workers)
    write_all({"dev": dev, "heldout": held}, out)
    log(f"wrote {out}")
