#!/usr/bin/env bash
set -euo pipefail

# Simple dataset generator for one coupling scenario:
# fixed beta_c and diffc scenario, configurable run parameter lists.
#
# Usage:
#   ./generate_one_coupling_scenario.sh [OUTDIR] [BETA_C] [DIFFC] [LAG]
#
# Example:
#   ./generate_one_coupling_scenario.sh ./data_beta07_diff057 0.7 0.57024 1

OUTDIR="${1:-./data_one_scenario}"
BETA_C="${2:-0.7}"
DIFFC="${3:-0.57024}"
LAG="${4:-1}"

# Optional run-variation dimensions (comma-separated lists).
# Keep these as single values for the smallest dataset.
HK_VALUES="${HK_VALUES:-864.0}"
POR_VALUES="${POR_VALUES:-0.35}"
INFLOW_VALUES="${INFLOW_VALUES:-2.851}"
GHB_HEAD_VALUES="${GHB_HEAD_VALUES:-1.0}"
AL_VALUES="${AL_VALUES:-0.0}"
AT_VALUES="${AT_VALUES:-0.0}"
CINLET="${CINLET:-35.0}"

# Runtime controls
MF6_EXE="${MF6_EXE:-./.venv/bin/mf6}"
MAX_RUNS_PER_SCENARIO="${MAX_RUNS_PER_SCENARIO:-}"
SAVE_TIMESERIES="${SAVE_TIMESERIES:-0}"
OVERWRITE="${OVERWRITE:-0}"
SCENARIO_PAIRS="${BETA_C}:${DIFFC}"

CMD=(
  uv run python run_henry.py
  --outdir "$OUTDIR"
  --mf6-exe "$MF6_EXE"
  --scenario-pairs "$SCENARIO_PAIRS"
  --lag "$LAG"
  --hk-values "$HK_VALUES"
  --por-values "$POR_VALUES"
  --al-values "$AL_VALUES"
  --at-values "$AT_VALUES"
  --inflow-values "$INFLOW_VALUES"
  --ghb-head-values "$GHB_HEAD_VALUES"
  --cinlet "$CINLET"
)

if [[ -n "$MAX_RUNS_PER_SCENARIO" ]]; then
  CMD+=(--max-runs-per-scenario "$MAX_RUNS_PER_SCENARIO")
fi

if [[ "$SAVE_TIMESERIES" == "1" ]]; then
  CMD+=(--save-timeseries)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Running one coupling scenario dataset generation"
echo "  outdir:    $OUTDIR"
echo "  beta_c:    $BETA_C"
echo "  diffc:     $DIFFC"
echo "  lag:       $LAG"
echo "  mf6 exe:   $MF6_EXE"
echo "  command:   ${CMD[*]}"

"${CMD[@]}"

echo
echo "Done. See manifest: $OUTDIR/manifest.json"
