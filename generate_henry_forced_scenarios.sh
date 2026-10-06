#!/usr/bin/env bash
# =============================================================================
# generate_henry_forced_scenarios.sh
#
# Initial-condition + inflow → full-trajectory 3-D dataset generator for the
# FORCED Henry problem: classic MF6 Henry BCs (constant freshwater WEL inflow
# on the left, sea-level GHB with seawater on the right, no-flow top/bottom,
# zero storage). Intermediate between simple_henry and henry_data.
#
# Scenarios are the beta_c x diffc grid. Runs differ by initial condition
# (random field) and a constant inflow Q ~ U[INFLOW_MIN, INFLOW_MAX]. Each
# scenario draws its own run set from seed [SEED, scenario_index]
# (SHARED_RUNS=1: one run set from SEED reused by every scenario).
#
# Output layout:
#   <OUTDIR>/
#   ├── manifest.json
#   ├── scenario_001/
#   │   ├── scenario_manifest.json
#   │   └── scenario.npz        ← all runs batched: (n_runs, 4, T-1, nlay, ncol)
#   └── ...
#
# The raw MODFLOW series (t = 0 .. TOTAL_TIME) is downsampled by SKIP to T
# frames, then
#   input  → [C0, inflow, beta_c, diffc, coord_t, coord_z, coord_x] over T-1 times
#            (COORD_CHANNELS=0 drops the three coordinate channels)
#            (inflow = Q over the whole domain;
#             INFLOW_ENCODING=left_column: Q in column 0 only) shape (7, T-1, nlay, ncol)
#   target → C_t, H_t for t_1..t_{T-1}                       shape (2, T-1, nlay, ncol)
#
# Defaults (TOTAL_TIME=0.25, NSTP=100, SKIP=4) give T = 26 frames, i.e. 25
# target steps at Δt_eff = 0.01 d.
#
# Usage:
#   ./generate_henry_forced_scenarios.sh [OUTDIR]
#
# All parameters can be overridden via environment variables.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Scenario grid (centred on classic Henry: β_C = 7e-4, D = 0.57024 m²/d)
# ---------------------------------------------------------------------------
# 0.0028 is left out: with Q = 2–6 its Henry a = Q/(K·β·c_sea·Lz) is 0.02–0.07 and the
# wedge sits on the inland boundary (classic Henry: a ≈ 0.26).
BETA_C_VALUES="${BETA_C_VALUES:-"0.00035, 0.0007, 0.0014"}"
DIFFC_VALUES="${DIFFC_VALUES:-"0.05, 0.1, 0.3, 0.57024"}"

# ---------------------------------------------------------------------------
# Physical parameters (classic Henry)
# ---------------------------------------------------------------------------
HK="${HK:-864.0}"
POR="${POR:-0.35}"
AL="${AL:-0.0}"
AT="${AT:-0.0}"
RHO0="${RHO0:-1000.0}"
C_SEA="${C_SEA:-35.0}"

# ---------------------------------------------------------------------------
# Inflow forcing: total Q [m³/d] per run ~ U[INFLOW_MIN, INFLOW_MAX]
# (classic Henry 5.7024, low-inflow variant 2.851)
# ---------------------------------------------------------------------------
INFLOW_MIN="${INFLOW_MIN:-2.0}"
INFLOW_MAX="${INFLOW_MAX:-6.0}"
# broadcast: Q over the whole domain (like beta_c/diffc); left_column: inflow column only
INFLOW_ENCODING="${INFLOW_ENCODING:-broadcast}"
# 1: append (t, z, x) coordinate channels to the input (7 channels); 0: 4 channels
COORD_CHANNELS="${COORD_CHANNELS:-1}"

# ---------------------------------------------------------------------------
# Initial Concentration Configuration
# ---------------------------------------------------------------------------
INIT_METHOD="${INIT_METHOD:-random}"  # Options: "random", "seawater"

RANDOM_FIELD_TYPE="${RANDOM_FIELD_TYPE:-grf}"
RANDOM_FIELD_SMOOTHNESS="${RANDOM_FIELD_SMOOTHNESS:-0.8}"
RANDOM_FIELD_LEN_SCALE="${RANDOM_FIELD_LEN_SCALE:-0.1, 0.3, 0.5, 0.7, 0.9}"
RANDOM_FIELD_VAR="${RANDOM_FIELD_VAR:-0.1, 0.3, 0.5, 0.7}"
RANDOM_FIELD_COUNT="${RANDOM_FIELD_COUNT:-20}"
# No zero-concentration BCs here: no zero-edge taper (1 = simple_henry-style
# taper), and period 2 so opposite edges of C0 are not correlated (1 = periodic).
RANDOM_FIELD_TAPER="${RANDOM_FIELD_TAPER:-0}"
RANDOM_FIELD_PERIOD="${RANDOM_FIELD_PERIOD:-2.0}"

# ---------------------------------------------------------------------------
# Grid and time controls
#
#   SKIP — temporal stride applied as time_series[::skip].
#          Choose NSTP divisible by SKIP so the last frame is t = TOTAL_TIME.
# ---------------------------------------------------------------------------
NCOL="${NCOL:-40}"
NLAY="${NLAY:-20}"
LX="${LX:-2.0}"
LZ="${LZ:-1.0}"
TOTAL_TIME="${TOTAL_TIME:-0.25}"
NSTP="${NSTP:-50}"
SKIP="${SKIP:-2}"

# ---------------------------------------------------------------------------
# Parallelism (blank → auto: min(n_scenarios, cpu_count); 1 → sequential)
# ---------------------------------------------------------------------------
N_WORKERS="${N_WORKERS:-}"

# ---------------------------------------------------------------------------
# Dataset / runtime controls
# ---------------------------------------------------------------------------
OUTBASEDIR="${OUTBASEDIR:-/Users/$USER/Projects/groundwater/data/henry_forced_data}"
OUTDIR="${1:-${OUTDIR:-${OUTBASEDIR}/grid_scenarios_forced_${INIT_METHOD}_skip${SKIP}_${NLAY}x${NCOL}}}"

SEED="${SEED:-42}"
# 0: each scenario draws its own C0/Q run set (seed [SEED, i]); 1: one run set shared by all
SHARED_RUNS="${SHARED_RUNS:-0}"
MAX_RUNS_PER_SCENARIO="${MAX_RUNS_PER_SCENARIO:-}"
MF6_EXE="${MF6_EXE:-$SCRIPT_DIR/.venv/bin/mf6}"
SAVE_TIMESERIES="${SAVE_TIMESERIES:-0}"
SAVE_MODFLOW_FILES="${SAVE_MODFLOW_FILES:-0}"
OVERWRITE="${OVERWRITE:-1}"
KAPPA_FILE="${KAPPA_FILE:-}"

if [[ "$MF6_EXE" == */* && ! -x "$MF6_EXE" ]]; then
  echo "ERROR: mf6 executable not found or not executable: $MF6_EXE" >&2
  echo "Hint: set MF6_EXE to an absolute executable path, or install mf6 in PATH and set MF6_EXE=mf6." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build generation command
# ---------------------------------------------------------------------------
CMD=(
  uv run python "$SCRIPT_DIR/run_henry_forced.py"
  --outdir        "$OUTDIR"
  --ncol          "$NCOL"
  --nlay          "$NLAY"
  --lx            "$LX"
  --lz            "$LZ"
  --total-time    "$TOTAL_TIME"
  --nstp          "$NSTP"
  --skip          "$SKIP"
  --init-method   "$INIT_METHOD"
  --random-field-type       "$RANDOM_FIELD_TYPE"
  --random-field-smoothness "$RANDOM_FIELD_SMOOTHNESS"
  --random-field-len-scale  "$RANDOM_FIELD_LEN_SCALE"
  --random-field-var        "$RANDOM_FIELD_VAR"
  --random-field-count      "$RANDOM_FIELD_COUNT"
  --random-field-period     "$RANDOM_FIELD_PERIOD"
  --inflow-min    "$INFLOW_MIN"
  --inflow-max    "$INFLOW_MAX"
  --inflow-encoding "$INFLOW_ENCODING"
  --beta-c-values "$BETA_C_VALUES"
  --diffc-values  "$DIFFC_VALUES"
  --hk            "$HK"
  --por           "$POR"
  --al            "$AL"
  --at            "$AT"
  --rho0          "$RHO0"
  --c-sea         "$C_SEA"
  --seed          "$SEED"
  --mf6-exe       "$MF6_EXE"
)

if [[ "$COORD_CHANNELS" == "1" ]]; then
  CMD+=(--coord-channels)
else
  CMD+=(--no-coord-channels)
fi
if [[ "$SHARED_RUNS" == "1" ]]; then
  CMD+=(--shared-runs)
fi
if [[ "$RANDOM_FIELD_TAPER" == "1" ]]; then
  CMD+=(--random-field-taper)
fi
if [[ -n "$N_WORKERS" ]]; then
  CMD+=(--n-workers "$N_WORKERS")
fi
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
echo "  Forced Henry 3-D (space × time) IC + inflow → trajectory generator"
echo "  PDE:        elliptic flow + parabolic transport"
echo "  Storage:    Ss = 0  (no STO package)"
echo "  Left BC:    WEL constant inflow Q, C = 0"
echo "  Right BC:   GHB head = Lz, inflow C = $C_SEA"
echo "============================================================"
echo "  outdir:         $OUTDIR"
echo "  grid:           nlay=$NLAY  ncol=$NCOL  Lx=$LX  Lz=$LZ"
echo "  time:           total=$TOTAL_TIME d  nstp=$NSTP  skip=$SKIP"
echo "  n_workers:      ${N_WORKERS:-auto (min(n_scenarios, cpu_count))}"
echo "  beta_c values:  $BETA_C_VALUES"
echo "  diffc values:   $DIFFC_VALUES"
echo "  inflow range:   [$INFLOW_MIN, $INFLOW_MAX] m³/d  (encoding=$INFLOW_ENCODING)"
echo "  hk / por:       $HK / $POR"
echo "  al / at:        $AL / $AT"
echo "  rho0 / c_sea:   $RHO0 / $C_SEA"
echo "  init method:    $INIT_METHOD  (taper=$RANDOM_FIELD_TAPER  period=$RANDOM_FIELD_PERIOD)"
echo "  seed:           $SEED  (shared_runs=$SHARED_RUNS)"
echo "  mf6 exe:        $MF6_EXE"
if [[ "$COORD_CHANNELS" == "1" ]]; then
  echo "  layout:         input [C0, inflow, beta_c, diffc, t, z, x] (7, T-1, ...) → target C,H (2, T-1, ...)"
else
  echo "  layout:         input [C0, inflow, beta_c, diffc] (4, T-1, ...) → target C,H (2, T-1, ...)"
fi
echo "============================================================"

"${CMD[@]}"

# ---------------------------------------------------------------------------
# Compress the output directory into a tar.gz archive
# ---------------------------------------------------------------------------
ARCHIVE_PATH="${OUTDIR%/}.tar.gz"
tar -C "$(dirname "$OUTDIR")" -czf "$ARCHIVE_PATH" "$(basename "$OUTDIR")"

echo
echo "Done. Archive: ${ARCHIVE_PATH}"
echo "See:  ${OUTDIR}/manifest.json"
