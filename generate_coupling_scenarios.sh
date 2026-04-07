#!/usr/bin/env bash
set -euo pipefail

# Generate 20 Henry coupling/diffusion scenarios and reorganize outputs into
# clean folders with numeric run names and JSON parameter metadata.
#
# Scenario design:
#   - First 10 scenarios: beta_c in [0.01, 1.0], diffc fixed at 0.57024
#   - Next 10 scenarios: diffc in [0.001, 1.0], beta_c fixed at 0.70
#
# Usage:
#   ./generate_coupling_scenarios.sh [OUTDIR] [LAG]

OUTDIR="${1:-/Users/akap5486/Projects/groundwater/data/henry_data/coupling_scenarios}"
LAG="${2:-1}"

# Scenario-grid controls
BETA_MIN="${BETA_MIN:-0.01}"
BETA_MAX="${BETA_MAX:-1.0}"
BETA_COUNT="${BETA_COUNT:-10}"
DIFFC_MIN="${DIFFC_MIN:-0.001}"
DIFFC_MAX="${DIFFC_MAX:-1.0}"
DIFFC_COUNT="${DIFFC_COUNT:-10}"
FIXED_BETA="${FIXED_BETA:-0.70}"
FIXED_DIFFC="${FIXED_DIFFC:-0.57024}"

# Run-variation dimensions (CSV lists)
HK_VALUES="${HK_VALUES:-664.0,700.0,864.0,1000.0}"
POR_VALUES="${POR_VALUES:-0.25,0.30,0.35,0.40,0.45}"
INFLOW_VALUES="${INFLOW_VALUES:-1.426,2.851,4.2765,5.7024}"
GHB_HEAD_VALUES="${GHB_HEAD_VALUES:-0.90,0.98,1.00,1.02,1.10}"
AL_VALUES="${AL_VALUES:-0.0,0.01}"
AT_VALUES="${AT_VALUES:-0.0,0.005}"
CINLET="${CINLET:-35.0}"

# Grid/time controls
NCOL="${NCOL:-80}"
NLAY="${NLAY:-40}"
TOTAL_TIME="${TOTAL_TIME:-0.5}"
NSTP="${NSTP:-100}"

# Dataset split controls
SEED="${SEED:-42}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
VAL_FRAC="${VAL_FRAC:-0.15}"

# Runtime controls
MF6_EXE="${MF6_EXE:-./.venv/bin/mf6}"
MAX_RUNS_PER_SCENARIO="${MAX_RUNS_PER_SCENARIO:-}"
SAVE_TIMESERIES="${SAVE_TIMESERIES:-0}"
OVERWRITE="${OVERWRITE:-1}"
WARM_START="${WARM_START:-0}"
KEEP_RAW="${KEEP_RAW:-0}"

RAW_OUTDIR="$OUTDIR/_raw_generation"

CMD=(
  uv run python run_henry.py
  --outdir "$RAW_OUTDIR"
  --ncol "$NCOL"
  --nlay "$NLAY"
  --total-time "$TOTAL_TIME"
  --nstp "$NSTP"
  --mf6-exe "$MF6_EXE"
    --coupling-diffusion-grid
    --beta-min "$BETA_MIN"
    --beta-max "$BETA_MAX"
    --beta-count "$BETA_COUNT"
    --diffc-min "$DIFFC_MIN"
    --diffc-max "$DIFFC_MAX"
    --diffc-count "$DIFFC_COUNT"
    --fixed-beta "$FIXED_BETA"
    --fixed-diffc "$FIXED_DIFFC"
  --lag "$LAG"
  --hk-values "$HK_VALUES"
  --por-values "$POR_VALUES"
  --al-values "$AL_VALUES"
  --at-values "$AT_VALUES"
  --inflow-values "$INFLOW_VALUES"
  --ghb-head-values "$GHB_HEAD_VALUES"
  --cinlet "$CINLET"
  --seed "$SEED"
  --train-frac "$TRAIN_FRAC"
  --val-frac "$VAL_FRAC"
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

if [[ "$WARM_START" == "1" ]]; then
  CMD+=(--warm-start)
else
  CMD+=(--no-warm-start)
fi

echo "Generating coupling/diffusion scenario grid"
echo "  outdir:      $OUTDIR"
echo "  raw outdir:  $RAW_OUTDIR"
echo "  lag:         $LAG"
echo "  beta grid:   [$BETA_MIN, $BETA_MAX] count=$BETA_COUNT (diffc=$FIXED_DIFFC)"
echo "  diffc grid:  [$DIFFC_MIN, $DIFFC_MAX] count=$DIFFC_COUNT (beta_c=$FIXED_BETA)"
echo "  command:     ${CMD[*]}"

"${CMD[@]}"

REORG_CMD=(
  uv run python -m henry_data.reorganize
  --raw-outdir "$RAW_OUTDIR"
  --outdir "$OUTDIR"
  --beta-count "$BETA_COUNT"
  --diffc-count "$DIFFC_COUNT"
  --lag "$LAG"
)

if [[ "$OVERWRITE" == "1" ]]; then
  REORG_CMD+=(--overwrite)
fi

echo "Reorganizing raw outputs into clean scenario layout"
echo "  command:     ${REORG_CMD[*]}"
"${REORG_CMD[@]}"

if [[ "$KEEP_RAW" == "0" ]]; then
  rm -rf "$RAW_OUTDIR"
fi

echo
echo "Done. See: $OUTDIR/scenarios_manifest.json"
