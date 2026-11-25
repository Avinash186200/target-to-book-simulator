.PHONY: install test verify bench report demo

install:
	pip install -c constraints.txt -e ".[dev]"

# Recomputes every summary from results/runs.json, checks the manifest and reruns
# held-out seed 101 and development seed 1 from scratch.
test:
	pytest -q

# Reruns every seed and checks it against results/runs.json (a few minutes).
verify:
	FULL=1 pytest -q tests/test_results.py

# Offline, deterministic: development choices, then held-out seeds once.
# Writes results/{runs,dev_choices,waterfall,timing,repair,bands,manifest}.json and RESULTS.md.
bench:
	python -m targetbook.cli bench --out results --workers 5

report:
	python -m targetbook.cli report --out results

demo:
	python -m targetbook.cli demo --seed 1
