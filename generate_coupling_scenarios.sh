#!/usr/bin/env bash
set -euo pipefail

# Generate Henry coupling/diffusion scenarios on a full beta_c x diffc grid and reorganize outputs into
# clean folders with numeric run names and JSON parameter metadata.
#
# Scenario design:
#   - Full Cartesian grid of beta_c in [BETA_MIN, BETA_MAX] and
#     diffc in [DIFFC_MIN, DIFFC_MAX]
#   - With BETA_COUNT=5 and DIFFC_COUNT=5, this yields 25 scenarios
#
# Usage:
#   ./generate_coupling_scenarios.sh [OUTDIR] [LAG]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTDIR="${1:-/Users/$USER/Projects/groundwater/data/henry_data/grid_scenarios_realistic_20x40}"
LAG="${2:-1}"

# Scenario-grid controls
BETA_MIN="${BETA_MIN:-0.1}"
BETA_MAX="${BETA_MAX:-1.0}"
BETA_COUNT="${BETA_COUNT:-5}"
DIFFC_MIN="${DIFFC_MIN:-0.0008}"
DIFFC_MAX="${DIFFC_MAX:-0.128}"
DIFFC_COUNT="${DIFFC_COUNT:-5}"
FIXED_BETA="${FIXED_BETA:-0.70}"
FIXED_DIFFC="${FIXED_DIFFC:-0.57024}"

# Run-variation dimensions (CSV lists)
HK_VALUES="${HK_VALUES:-864.0}"
POR_VALUES="${POR_VALUES:-0.35}"
# Inflow scaled for the 8×4 m domain to preserve Q/(K·Lz) ratio
# (original 2×1 m values of 1.426,2.1385,2.851,4.2767,5.7024 multiplied by 5 for the 4 m domain):
INFLOW_VALUES="${INFLOW_VALUES:-7.1305,10.6958,14.261,21.3915,28.522}"
# ghb_head = 75% of domain height (0.75×Lz = 0.75×4.0 = 3.0 m):
GHB_HEAD_VALUES="${GHB_HEAD_VALUES:-3.0}"
AL_VALUES="${AL_VALUES:-0.0}"
AT_VALUES="${AT_VALUES:-0.0}"
CINLET="${CINLET:-35.0}"

# Grid/time controls
NCOL="${NCOL:-40}"
NLAY="${NLAY:-20}"
# Physical domain dimensions (matching the one_coupling scenario script)
LX="${LX:-8.0}"  # horizontal extent [m]
LZ="${LZ:-4.0}"  # vertical extent [m]
TOTAL_TIME="${TOTAL_TIME:-60}"
NSTP="${NSTP:-480}"

# Spin-up controls (warm-start pre-run before the main simulation)
SPINUP_TIME="${SPINUP_TIME:-0.5}"
SPINUP_NSTP="${SPINUP_NSTP:-4}"

# Tidal forcing parameters
TIDAL_AMPLITUDE="${TIDAL_AMPLITUDE:-0.50}"
SPRING_NEAP_AMP="${SPRING_NEAP_AMP:-0.20}"
SPRING_NEAP_PHASE="${SPRING_NEAP_PHASE:-3.14159}"
SLR_RATE="${SLR_RATE:-0.003}"

# Prediction lag (in wall-clock days)
LAG_DAYS="${LAG_DAYS:-1}"

# Freshwater inflow — stochastic shot-noise model parameters
STORM_RATE="${STORM_RATE:-0.2}"
STORM_AMP_MEAN="${STORM_AMP_MEAN:-1.0}"
STORM_AMP_STD="${STORM_AMP_STD:-0.25}"
RECESSION_K="${RECESSION_K:-1.0}"
AR1_PHI="${AR1_PHI:-0.85}"
AR1_SIGMA="${AR1_SIGMA:-0.05}"
INFLOW_TREND_AMP="${INFLOW_TREND_AMP:--0.4}"

# Dataset split controls
SEED="${SEED:-42}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
VAL_FRAC="${VAL_FRAC:-0.15}"

# Runtime controls
MF6_EXE="${MF6_EXE:-$SCRIPT_DIR/.venv/bin/mf6}"
MAX_RUNS_PER_SCENARIO="${MAX_RUNS_PER_SCENARIO:-}"
SAVE_TIMESERIES="${SAVE_TIMESERIES:-0}"
SAVE_MODFLOW_FILES="${SAVE_MODFLOW_FILES:-0}"
OVERWRITE="${OVERWRITE:-1}"
WARM_START="${WARM_START:-0}"
KEEP_RAW="${KEEP_RAW:-0}"
DYNAMIC_INFLOW="${DYNAMIC_INFLOW:-1}"
DYNAMIC_TIDES="${DYNAMIC_TIDES:-1}"
ADD_STORAGE="${ADD_STORAGE:-1}"

RAW_OUTDIR="$OUTDIR/_raw_generation"

if [[ "$MF6_EXE" == */* && ! -x "$MF6_EXE" ]]; then
  echo "ERROR: mf6 executable not found or not executable: $MF6_EXE" >&2
  echo "Hint: set MF6_EXE to an absolute executable path, or install mf6 in PATH and set MF6_EXE=mf6." >&2
  exit 1
fi

CMD=(
  uv run python run_henry.py
  --outdir "$RAW_OUTDIR"
  --ncol "$NCOL"
  --nlay "$NLAY"
  --lx "$LX"
  --lz "$LZ"
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
  --lag-days "$LAG_DAYS"
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
  --spinup-time "$SPINUP_TIME"
  --spinup-nstp "$SPINUP_NSTP"
  --tidal-amplitude "$TIDAL_AMPLITUDE"
  --spring-neap-amp "$SPRING_NEAP_AMP"
  --spring-neap-phase "$SPRING_NEAP_PHASE"
  --slr-rate "$SLR_RATE"
  --storm-rate "$STORM_RATE"
  --storm-amp-mean "$STORM_AMP_MEAN"
  --storm-amp-std "$STORM_AMP_STD"
  --recession-k "$RECESSION_K"
  --ar1-phi "$AR1_PHI"
  --ar1-sigma "$AR1_SIGMA"
  --inflow-trend-amp "$INFLOW_TREND_AMP"
)

if [[ -n "$MAX_RUNS_PER_SCENARIO" ]]; then
  CMD+=(--max-runs-per-scenario "$MAX_RUNS_PER_SCENARIO")
fi

if [[ "$SAVE_TIMESERIES" == "1" ]]; then
  CMD+=(--save-timeseries)
fi

if [[ "$SAVE_MODFLOW_FILES" == "1" ]]; then
  CMD+=(--save-modflow-files)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ "$WARM_START" == "1" ]]; then
  CMD+=(--warm-start)
else
  CMD+=(--no-warm-start)
fi

if [[ "$DYNAMIC_INFLOW" == "1" ]]; then
  CMD+=(--dynamic-inflow)
fi

if [[ "$DYNAMIC_TIDES" == "1" ]]; then
  CMD+=(--dynamic-tides)
fi

if [[ "$ADD_STORAGE" == "1" ]]; then
  CMD+=(--add-storage)
fi


echo "Generating coupling/diffusion scenario grid"
echo "  outdir:      $OUTDIR"
echo "  raw outdir:  $RAW_OUTDIR"
echo "  lag:         $LAG"
echo "  lag_days:    $LAG_DAYS"
echo "  domain:      Lx=$LX Lz=$LZ"
echo "  grid:        nlay=$NLAY ncol=$NCOL"
echo "  time:        total_time=$TOTAL_TIME nstp=$NSTP"
echo "  spinup:      time=$SPINUP_TIME nstp=$SPINUP_NSTP"
echo "  tidal:       amp=$TIDAL_AMPLITUDE spring_neap=$SPRING_NEAP_AMP slr_rate=$SLR_RATE"
echo "  inflow:      trend_amp=$INFLOW_TREND_AMP"
echo "  save mf6:    $SAVE_MODFLOW_FILES"
echo "  beta grid:   [$BETA_MIN, $BETA_MAX] count=$BETA_COUNT"
echo "  diffc grid:  [$DIFFC_MIN, $DIFFC_MAX] count=$DIFFC_COUNT"
echo "  scenarios:   $((BETA_COUNT * DIFFC_COUNT)) (full Cartesian grid)"
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
