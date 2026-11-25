"""The committed results are reproducible.

* Every summary and RESULTS.md is recomputed from the per-seed records in
  results/runs.json and must match the committed files.
* manifest.json holds the SHA-256 of each results file and the frozen
  configuration; both must match.
* A fixed subset of seeds is rerun from scratch and must match its record:
  the first held-out seed (every configuration, both signals) and the first
  development seed (the whole choice grid). Set FULL=1 to rerun every seed
  (about a minute per development seed on a laptop; ``make verify``).
* Every row of the README results tables appears in RESULTS.md.
"""
import json
import math
import os
import re
from pathlib import Path

import pytest

from targetbook import bench as B
from targetbook.data import MarketConfig
from targetbook.pipeline import DeskConfig, dev_task, heldout_task

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FULL = os.environ.get("FULL") == "1"


def _diff(a, b, path="$"):
    """Differences between two JSON trees; floats match to 1e-6 absolute, integers to 1 or 1%."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(a.keys() | b.keys()):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: missing")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) and not isinstance(b, bool):
        # counts (repairs, solves) may shift by one on another platform if a band decision flips
        return [] if abs(a - b) <= max(1, 0.01 * max(abs(a), abs(b))) else [f"{path}: {a} != {b}"]
    if isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
        return [] if ok else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs():
    return _load("runs.json")


@pytest.mark.parametrize("name", B.SUMMARY_FILES)
def test_summaries_are_computed_from_the_records(runs, name):
    fresh = json.loads(B._dumps(B.SUMMARIZERS[name](runs)))
    diffs = _diff(fresh, _load(f"{name}.json"))
    assert not diffs, "\n".join(diffs[:20])


def test_results_md_is_rendered_from_the_summaries():
    assert B.render(RES) + "" == (RES / "RESULTS.md").read_text(encoding="utf-8")


def test_results_written_with_lf_only():
    for f in RES.iterdir():
        if f.suffix in (".json", ".md"):
            assert b"\r\n" not in f.read_bytes(), f.name


def test_manifest_checksums_and_frozen_configuration():
    man = _load("manifest.json")
    for name, digest in man["sha256"].items():
        assert B._sha(RES / name) == digest, name
    assert man["market"] == json.loads(json.dumps(MarketConfig().to_dict()))
    assert man["desk"] == json.loads(json.dumps(DeskConfig().to_dict()))
    assert man["mandate"] == json.loads(json.dumps(B.default_mandate().to_dict()))
    assert man["dev_seeds"] == list(B.DEV_SEEDS) and man["heldout_seeds"] == list(B.HELDOUT_SEEDS)


def test_library_defaults_are_the_frozen_development_choices():
    dev = _load("dev_choices.json")
    assert DeskConfig().choices() == dev["chosen"]


def test_heldout_and_development_seeds_are_disjoint(runs):
    dev = {r["seed"] for r in runs["dev"]}
    held = {r["seed"] for r in runs["heldout"]}
    assert dev == set(B.DEV_SEEDS) and held == set(B.HELDOUT_SEEDS)
    assert not dev & held


def test_development_choice_is_not_at_the_edge_of_the_band_grid():
    dev = _load("dev_choices.json")
    grid = _load("bands.json")["band_grid"]
    assert dev["chosen"]["band"] < max(grid)
    assert all(b < max(grid) for b, _ in dev["cost_bands"].values())


def test_solver_never_fell_back():
    assert _load("repair.json")["solver"]["fallbacks"] == 0


def _norm(x):
    return json.loads(B._dumps(x))


def test_heldout_seeds_rerun_from_scratch(runs):
    dev = _load("dev_choices.json")
    desk = B.frozen_desk(dev)
    diffs = []
    for i, rec in enumerate(runs["heldout"]):
        if FULL or i == 0:
            fresh = _norm(heldout_task(rec["seed"], desk, dev["cost_bands"]))
            diffs += _diff(fresh, rec, f"heldout[{i}]")
    assert not diffs, "\n".join(diffs[:20])


def test_development_seeds_rerun_from_scratch(runs):
    diffs = []
    for i, rec in enumerate(runs["dev"]):
        if FULL or i == 0:
            diffs += _diff(_norm(dev_task(rec["seed"])), rec, f"dev[{i}]")
    assert not diffs, "\n".join(diffs[:20])


def _table_rows(text):
    return [ln.strip() for ln in text.splitlines()
            if ln.strip().startswith("|") and not re.fullmatch(r"\|[-| ]+\|", ln.strip())]


def test_readme_results_tables_come_from_results_md():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- results:start -->")[1].split("<!-- results:end -->")[0]
    rows = set(_table_rows((RES / "RESULTS.md").read_text(encoding="utf-8")))
    missing = [r for r in _table_rows(block) if r not in rows]
    assert _table_rows(block), "no results rows found in README"
    assert not missing, "\n".join(missing)
