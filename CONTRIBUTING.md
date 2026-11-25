# Contributing

Thanks for taking a look. Changes are judged on whether they keep the repo offline, honest about its numbers, and free of look-ahead.

## Ground rules

1. **Everything runs offline and on CPU.** Tests and `make bench` need no network and no GPU.
2. **Numbers come from results.** Any number in a doc must be a field in a `results/*.json` file written by `make bench`, or output of a documented command. If you change code that affects results, rerun `make bench`, commit the new JSON, `RESULTS.md` and `manifest.json` with the change, and run `make verify`. README results tables must stay copies of rows in `RESULTS.md` (a test checks this).
3. **Never tune on held-out seeds.** Schedule, bands and any new design choice are made on development seeds (1-5). If a change needs a new choice after held-out results exist, make it on development seeds and record the history in `docs/methodology.md`.
4. **No look-ahead.** Anything decided at a hand-over day may use data up to and including that day only. Extend `test_no_look_ahead_in_targets_covariance_and_book` when you add a new input.
5. **No timing assertions in tests.** CI runners are shared and slow; tests check correctness and reproducibility, never speed.
6. **Determinism.** Every generator takes a seed; BLAS threads are pinned to 1 on import; Clarabel runs single-threaded with fixed tolerances. Results must not depend on `--workers`.
7. **No secrets.** Nothing here needs credentials; `.env` is git-ignored and CI runs a secret scan.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[dev]"
pytest -q
```

## Adding an implementation step

1. Add the step to `pipeline.run_signal`, both as a new ladder stage (one setting changed from the stage before) and as an "alone" run on the paper book.
2. Add it to `LADDER` / `ALONE` and to `STAGE_LABEL` in `bench.py`.
3. Add unit tests for the step, rerun `make bench`, and update the docs from the new JSON.
