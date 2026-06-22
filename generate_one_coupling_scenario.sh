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

OUTDIR="${1:-/Users/$USER/Projects/groundwater/data/henry_data/one_coupling_scenario}"
LAG="${4:-1}"

# Classic Henry benchmark values for beta_c and diffc are 0.7 and 0.57024, respectively.
BETA_C="${2:-0.7}"
DIFFC="${3:-0.57024}"


# Optional run-variation dimensions (comma-separated lists).
# Defaults are centered on the MODFLOW 6 Henry benchmark definition.
# HK_VALUES="${HK_VALUES:-664.0,700.0,864.0,1000.0}"
# POR_VALUES="${POR_VALUES:-0.25,0.30,0.35,0.40,0.45}"
# INFLOW_VALUES="${INFLOW_VALUES:-1.426,2.851,4.2765,5.7024}"
# GHB_HEAD_VALUES="${GHB_HEAD_VALUES:-0.90,0.98,1.00,1.02,1.10}"
# AL_VALUES="${AL_VALUES:-0.0,0.01}"
# AT_VALUES="${AT_VALUES:-0.0,0.005}"
# CINLET="${CINLET:-35.0}"


# Classic Example values (single value per parameter)
HK_VALUES="${HK_VALUES:-864.0}"
POR_VALUES="${POR_VALUES:-0.35}"
INFLOW_VALUES="${INFLOW_VALUES:-5.7024,2.851}"
GHB_HEAD_VALUES="${GHB_HEAD_VALUES:-0.4}"
AL_VALUES="${AL_VALUES:-0.0}"
AT_VALUES="${AT_VALUES:-0.0}"
CINLET="${CINLET:-35.0}"



# Grid/time controls
NCOL="${NCOL:-40}"
NLAY="${NLAY:-20}"
TOTAL_TIME="${TOTAL_TIME:-30}"
NSTP="${NSTP:-240}"

# Spin-up controls (warm-start pre-run before the main simulation)
SPINUP_TIME="${SPINUP_TIME:-10}"
SPINUP_NSTP="${SPINUP_NSTP:-80}"

# Tidal forcing parameters
# Reduced from 0.5/0.3 to match real island data: neap ±0.05m, spring ±0.25m around MSL.
TIDAL_AMPLITUDE="${TIDAL_AMPLITUDE:-0.15}"
SPRING_NEAP_AMP="${SPRING_NEAP_AMP:-0.10}"
# Phase offset in radians: pi (3.14159) = start at neap so amplitude grows to spring around day 7.
# Use 0 to start at spring (old behaviour). Neap-start matches real coastal tidal records better.
SPRING_NEAP_PHASE="${SPRING_NEAP_PHASE:-3.14159}"
# Sea-level rise [m/day]: 0.003 gives +0.09 m over 30 days (exaggerated but detectable)
SLR_RATE="${SLR_RATE:-0.003}"

# Freshwater inflow trend: -0.4 cuts inflow by 40% of mean by end of run (sustained drying)
INFLOW_TREND_AMP="${INFLOW_TREND_AMP:--0.4}"

# Prediction lag (in wall-clock days; overrides --lag when set)
LAG_DAYS="${LAG_DAYS:-1}"

# Dataset split controls
SEED="${SEED:-42}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
VAL_FRAC="${VAL_FRAC:-0.15}"

# Runtime controls
MF6_EXE="${MF6_EXE:-./.venv/bin/mf6}"
MAX_RUNS_PER_SCENARIO="${MAX_RUNS_PER_SCENARIO:-}"
SAVE_TIMESERIES="${SAVE_TIMESERIES:-0}"
SAVE_MODFLOW_FILES="${SAVE_MODFLOW_FILES:-1}"
OVERWRITE="${OVERWRITE:-1}"
WARM_START="${WARM_START:-0}"
SCENARIO_PAIRS="${BETA_C}:${DIFFC}"
DYNAMIC_INFLOW="${DYNAMIC_INFLOW:-1}"
DYNAMIC_TIDES="${DYNAMIC_TIDES:-1}"
ADD_STORAGE="${ADD_STORAGE:-1}"

# Animation controls (post-generation)
GENERATE_ANIMATION="${GENERATE_ANIMATION:-1}"
ANIMATE_FPS="${ANIMATE_FPS:-20}"
ANIMATE_DPI="${ANIMATE_DPI:-150}"
ANIMATE_SKIP="${ANIMATE_SKIP:-1}"

CMD=(
  uv run python run_henry.py
  --outdir "$OUTDIR"
  --ncol "$NCOL"
  --nlay "$NLAY"
  --total-time "$TOTAL_TIME"
  --nstp "$NSTP"
  --mf6-exe "$MF6_EXE"
  --scenario-pairs "$SCENARIO_PAIRS"
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

echo "Running one coupling scenario dataset generation"
echo "  outdir:      $OUTDIR"
echo "  beta_c:      $BETA_C"
echo "  diffc:       $DIFFC"
echo "  lag:         $LAG"
echo "  lag_days:    $LAG_DAYS"
echo "  grid:        nlay=$NLAY ncol=$NCOL"
echo "  time:        total_time=$TOTAL_TIME nstp=$NSTP"
echo "  spinup:      time=$SPINUP_TIME nstp=$SPINUP_NSTP"
echo "  tidal:       amp=$TIDAL_AMPLITUDE spring_neap=$SPRING_NEAP_AMP slr_rate=$SLR_RATE"
echo "  inflow:      trend_amp=$INFLOW_TREND_AMP"
echo "  split:       seed=$SEED train=$TRAIN_FRAC val=$VAL_FRAC"
echo "  warm_start:  $WARM_START"
echo "  dyn_inflow:  $DYNAMIC_INFLOW"
echo "  dyn_tides:   $DYNAMIC_TIDES"
echo "  storage:     $ADD_STORAGE"
echo "  save mf6:    $SAVE_MODFLOW_FILES"
echo "  mf6 exe:     $MF6_EXE"
echo "  animate:     $GENERATE_ANIMATION"
echo "  command:     ${CMD[*]}"

"${CMD[@]}"

if [[ "$GENERATE_ANIMATION" == "1" && "$SAVE_MODFLOW_FILES" != "1" ]]; then
  echo
  echo "Skipping animation because SAVE_MODFLOW_FILES=0 removed gwf.hds/gwt.ucn."
  echo "Set SAVE_MODFLOW_FILES=1 to enable animation output."
fi

if [[ "$GENERATE_ANIMATION" == "1" && "$SAVE_MODFLOW_FILES" == "1" ]]; then
  RUN_WORKSPACE="$(uv run python - "$OUTDIR" "$BETA_C" "$DIFFC" <<'PY'
import json
import pathlib as pl
import sys

outdir = pl.Path(sys.argv[1])
beta = float(sys.argv[2])
diffc = float(sys.argv[3])
manifest_path = outdir / "manifest.json"

if not manifest_path.exists():
    raise SystemExit(f"manifest not found: {manifest_path}")

with manifest_path.open("r", encoding="utf-8") as fp:
    manifest = json.load(fp)

def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol

candidates = [
    r for r in manifest.get("runs", [])
    if close(float(r.get("beta_c", float("nan"))), beta)
    and close(float(r.get("diffc", float("nan"))), diffc)
    and r.get("status") in {"ok", "skipped"}
]

if not candidates:
    raise SystemExit(
        f"No run workspace found for beta_c={beta}, diffc={diffc} in {manifest_path}"
    )

print(candidates[-1]["workspace"])
PY
)"

  ANIMATE_CMD=(
    uv run python animate_henry.py
    --dataset-path "$OUTDIR"
    --run-path "$RUN_WORKSPACE"
    --fps "$ANIMATE_FPS"
    --dpi "$ANIMATE_DPI"
    --skip "$ANIMATE_SKIP"
  )

  echo
  echo "Generating animation from saved outputs"
  echo "  run path:  $RUN_WORKSPACE"
  echo "  command:   ${ANIMATE_CMD[*]}"
  "${ANIMATE_CMD[@]}"
fi

echo
echo "Done. See manifest: $OUTDIR/manifest.json"
