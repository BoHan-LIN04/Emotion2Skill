#!/usr/bin/env bash
set -euo pipefail
BENCHMARK="${1:-webshop}"
CONFIG="${2:-configs/default.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
for METHOD in ZeroShot ReAct Reflexion TextSimilarity; do
  "${PYTHON_BIN}" evaluate.py --config "${CONFIG}" --env "${BENCHMARK}" --method "${METHOD}" \
    --n_runs 3 --out_dir "results/baselines/${BENCHMARK}/${METHOD}"
done
# ExpeL needs --baseline_artifact; upstream skill methods also need --external_adapter.
