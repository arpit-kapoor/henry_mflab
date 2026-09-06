#!/usr/bin/env bash
# =============================================================================
# generate_simple_coupling_scenarios.sh
#
# Structured dataset generator for the SIMPLIFIED Henry problem:
#   - Full Cartesian grid of beta_c x diffc scenarios
#   - Zero specific storage (Ss = 0) — elliptic groundwater flow equation
#   - Zero influx — no freshwater inflow (WEL) or tidal forcing (GHB)
#   - Homogeneous Dirichlet BCs: p = 0 and C = 0 on all four sides
#   - Buoyancy-driven flow only via ρ(C) = ρ₀(1 + β_C C)
#   - Reorganizes outputs into clean numeric scenario and run directories:
#       <OUTDIR>/
#       ├── scenarios_manifest.json
#       └── scenarios/
#           ├── scenario_01/
#           │   ├── scenario_config.json
#           │   ├── runs_config.json
#           │   └── run_000001/
#           │       └── windows.npz
#
# Usage:
#   ./generate_simple_coupling_scenarios.sh [OUTDIR] [LAG]
#
# Example:
#   ./generate_simple_coupling_scenarios.sh
#   ./generate_simple_coupling_scenarios.sh ./simple_scenarios_out 2
#
# All parameters can be overridden via environment variables.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Output directory and prediction lag
# ---------------------------------------------------------------------------
OUTDIR="${1:-${OUTDIR:-/Users/$USER/Projects/groundwater/data/simple_henry_data/grid_scenarios_20x40}}"
LAG="${2:-${LAG:-1}}"

# ---------------------------------------------------------------------------
# Physical parameters (finalized simple Henry values)
# ---------------------------------------------------------------------------
BETA_C_VALUES="${BETA_C_VALUES:-"0.01, 0.05, 0.1, 0.2"}"
# diffc: effective diffusion coefficient [m²/d].
# For a 1-day run, diffc=0.01 gives a diffusion timescale τ_diff = Lz²/diffc = 100 days >> 1 day.
# This prevents premature decay while keeping the system in a convective regime.
DIFFC_VALUES="${DIFFC_VALUES:-"0.00001, 0.0001, 0.01, 0.1"}"
# hk: hydraulic conductivity [m/d].
# For a 1-day run, hk=50.0 keeps the Rayleigh number convective (Ra=350)
# and limits the maximum Courant number: Co = v_max * dt / dz = (50.0 * 0.07) * 0.01 / 0.05 = 0.7 < 1.0.
# This guarantees advective numerical stability.
HK_VALUES="${HK_VALUES:-50.0}"
POR_VALUES="${POR_VALUES:-0.35}"
RHO0="${RHO0:-1000.0}"

# Dispersivity (zero = pure molecular diffusion, consistent with theory)
AL="${AL:-0.0}"
AT="${AT:-0.0}"

# ---------------------------------------------------------------------------
# Initial Concentration Configuration
# ---------------------------------------------------------------------------
C_X_TOE_VALUES="${C_X_TOE_VALUES:-"-0.5, 0.0, 0.5"}"
C_X_TOP_VALUES="${C_X_TOP_VALUES:-"1.0, 1.5"}"
C_TRANS_WIDTH_VALUES="${C_TRANS_WIDTH_VALUES:-"0.01"}"

# ---------------------------------------------------------------------------
# Grid / time controls
# 40×20 grid, 2 m × 1 m domain, total time = 2.0 days, 50 steps (dt = 0.04 d)
# ---------------------------------------------------------------------------
NCOL="${NCOL:-40}"
NLAY="${NLAY:-20}"
LX="${LX:-2.0}"
LZ="${LZ:-1.0}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
NSTP="${NSTP:-50}"

# ---------------------------------------------------------------------------
# Dataset / split controls
# ---------------------------------------------------------------------------
SEED="${SEED:-42}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
VAL_FRAC="${VAL_FRAC:-0.15}"
MAX_RUNS_PER_SCENARIO="${MAX_RUNS_PER_SCENARIO:-}"

# ---------------------------------------------------------------------------
# Runtime controls
# ---------------------------------------------------------------------------
MF6_EXE="${MF6_EXE:-$SCRIPT_DIR/.venv/bin/mf6}"
SAVE_TIMESERIES="${SAVE_TIMESERIES:-0}"
SAVE_MODFLOW_FILES="${SAVE_MODFLOW_FILES:-1}"
OVERWRITE="${OVERWRITE:-1}"
KEEP_RAW="${KEEP_RAW:-0}"
KAPPA_FILE="${KAPPA_FILE:-}"

# ---------------------------------------------------------------------------
# Resolve scenario grid counts
# ---------------------------------------------------------------------------
if [[ -z "${BETA_COUNT:-}" ]]; then
  BETA_COUNT=$(uv run python -c "print(len([x for x in '''$BETA_C_VALUES'''.split(',') if x.strip()]))")
fi
if [[ -z "${DIFFC_COUNT:-}" ]]; then
  DIFFC_COUNT=$(uv run python -c "print(len([x for x in '''$DIFFC_VALUES'''.split(',') if x.strip()]))")
fi

RAW_OUTDIR="$OUTDIR/_raw_generation"

if [[ "$MF6_EXE" == */* && ! -x "$MF6_EXE" ]]; then
  echo "ERROR: mf6 executable not found or not executable: $MF6_EXE" >&2
  echo "Hint: set MF6_EXE to an absolute executable path, or install mf6 in PATH and set MF6_EXE=mf6." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build generation command
# ---------------------------------------------------------------------------
CMD=(
  uv run python run_simple_henry.py
  --outdir        "$RAW_OUTDIR"
  --ncol          "$NCOL"
  --nlay          "$NLAY"
  --lx            "$LX"
  --lz            "$LZ"
  --total-time    "$TOTAL_TIME"
  --nstp          "$NSTP"
  --c0-x-toe-values "$C_X_TOE_VALUES"
  --c0-x-top-values "$C_X_TOP_VALUES"
  --c0-trans-width-values "$C_TRANS_WIDTH_VALUES"
  --beta-c-values "$BETA_C_VALUES"
  --diffc-values  "$DIFFC_VALUES"
  --hk-values     "$HK_VALUES"
  --por-values    "$POR_VALUES"
  --al            "$AL"
  --at            "$AT"
  --rho0          "$RHO0"
  --lag           "$LAG"
  --seed          "$SEED"
  --train-frac    "$TRAIN_FRAC"
  --val-frac      "$VAL_FRAC"
  --mf6-exe       "$MF6_EXE"
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

if [[ -n "$KAPPA_FILE" ]]; then
  CMD+=(--kappa-file "$KAPPA_FILE")
fi

# ---------------------------------------------------------------------------
# Print configuration and run
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Simplified Henry coupling/diffusion scenario generator"
echo "  PDE:        elliptic flow + parabolic transport"
echo "  Storage:    Ss = 0  (no STO package)"
echo "  Influx:     zero    (no WEL / GHB)"
echo "  BCs:        homogeneous Dirichlet p=0, C=0 on all sides"
echo "============================================================"
echo "  outdir:         $OUTDIR"
echo "  raw outdir:     $RAW_OUTDIR"
echo "  grid:           nlay=$NLAY  ncol=$NCOL  Lx=$LX  Lz=$LZ"
echo "  time:           total=$TOTAL_TIME d  nstp=$NSTP  dt=$(uv run python -c "print($TOTAL_TIME/$NSTP)") d"
echo "  initial C:      x_toe=$C_X_TOE_VALUES  x_top=$C_X_TOP_VALUES  trans_width=$C_TRANS_WIDTH_VALUES"
echo "  beta_c values:  $BETA_C_VALUES (count=$BETA_COUNT)"
echo "  diffc values:   $DIFFC_VALUES (count=$DIFFC_COUNT)"
echo "  scenarios:      $((BETA_COUNT * DIFFC_COUNT))"
echo "  hk values:      $HK_VALUES"
echo "  por values:     $POR_VALUES"
echo "  al / at:        $AL / $AT"
echo "  rho0:           $RHO0 kg/m³"
echo "  lag:            $LAG step(s)"
echo "  split seed:     $SEED  train=$TRAIN_FRAC  val=$VAL_FRAC"
echo "  save mf6 files: $SAVE_MODFLOW_FILES"
echo "  mf6 exe:        $MF6_EXE"
echo "  command:        ${CMD[*]}"
echo "============================================================"

"${CMD[@]}"

# ---------------------------------------------------------------------------
# Reorganize into clean numeric scenario and run layout
# ---------------------------------------------------------------------------
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

echo
echo "Reorganizing raw outputs into clean scenario layout"
echo "  command:     ${REORG_CMD[*]}"
"${REORG_CMD[@]}"

# ---------------------------------------------------------------------------
# Clean up raw outputs if KEEP_RAW=0
# ---------------------------------------------------------------------------
if [[ "$KEEP_RAW" == "0" ]]; then
  rm -rf "$RAW_OUTDIR"
fi

echo
echo "Done. See: $OUTDIR/scenarios_manifest.json"

