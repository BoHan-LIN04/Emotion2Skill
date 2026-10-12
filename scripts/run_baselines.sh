#!/usr/bin/env bash
set -euo pipefail
BENCHMARK="${1:-webshop}"
CONFIG="${2:-configs/default.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
for METHOD in ZeroShot ReAct Reflexion TextSimilarity; do
  "${PYTHON_BIN}" evaluate.py --config "${CONFIG}" --env "${BENCHMARK}" --method "${METHOD}" \
    --n_runs 3 --out_dir "results/baselines/${BENCHMARK}/${METHOD}"
done
# ExpeL requires --baseline_artifact from actual training episodes.
# MASA/EvoSkill/Trace2Skill require both a real artifact and --external_adapter module:factory.
