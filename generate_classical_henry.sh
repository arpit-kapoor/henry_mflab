#!/usr/bin/env bash
# =============================================================================
# generate_classical_henry.sh — dataset of paper App. I (classical Henry problem)
#
#   - Elliptic variable-density flow (Ss = 0) + parabolic transport
#   - Left: constant freshwater inflow Q ~ U[INFLOW_MIN, INFLOW_MAX] (WEL, C = 0)
#   - Right: sea-level GHB (h = Lz), inflowing water carries C = C_SEA
#   - Top/bottom: no-flow
#   - 3 x 4 scenario grid of beta_c x D_C, 400 GRF initial fields per scenario
#   - T = 0.25 d simulated with NSTP = 50 steps, kept every SKIP = 2 → 25 frames
#     at Δt = 0.01 d
#
# Output: <OUTDIR>/manifest.json and <OUTDIR>/scenario_NNN/scenario.npz with
#   input_tensor  (n_runs, 7, 25, 20, 40)  [C0, inflow, beta_c, diffc, t, z, x]
#   output_tensor (n_runs, 2, 25, 20, 40)  [concentration, head]
#
# Usage:
#   ./generate_classical_henry.sh [OUTDIR]
#   MAX_RUNS_PER_SCENARIO=2 ./generate_classical_henry.sh ./data/test   # quick test
#
# Every variable below can be overridden from the environment.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${1:-${OUTDIR:-$SCRIPT_DIR/data/classical_henry}}"

# Scenario grid (centred on classical Henry: beta_c = 7e-4, D_C = 0.57024).
# Larger beta_c (e.g. 0.0028) pushes the wedge onto the inland boundary for Q in [2, 6].
BETA_C_VALUES="${BETA_C_VALUES:-0.00035, 0.0007, 0.0014}"   # [m³/kg]
DIFFC_VALUES="${DIFFC_VALUES:-0.05, 0.1, 0.3, 0.57024}"     # [m²/d]

# Physics
HK="${HK:-864.0}"         # hydraulic conductivity [m/d]
POR="${POR:-0.35}"        # porosity
RHO0="${RHO0:-1000.0}"    # reference density [kg/m³]
C_SEA="${C_SEA:-35.0}"    # seawater concentration [kg/m³]

# Inflow: total Q [m²/d per unit width] per run ~ U[INFLOW_MIN, INFLOW_MAX]
INFLOW_MIN="${INFLOW_MIN:-2.0}"
INFLOW_MAX="${INFLOW_MAX:-6.0}"

# Initial concentration (GRF, not tapered, period 2 so opposite edges are uncorrelated)
RANDOM_FIELD_LEN_SCALE="${RANDOM_FIELD_LEN_SCALE:-0.1, 0.3, 0.5, 0.7, 0.9}"
RANDOM_FIELD_VAR="${RANDOM_FIELD_VAR:-0.1, 0.3, 0.5, 0.7}"
RANDOM_FIELD_COUNT="${RANDOM_FIELD_COUNT:-20}"

# Grid and time
NCOL="${NCOL:-40}"; NLAY="${NLAY:-20}"
LX="${LX:-2.0}";    LZ="${LZ:-1.0}"
TOTAL_TIME="${TOTAL_TIME:-0.25}"
NSTP="${NSTP:-50}"
SKIP="${SKIP:-2}"

# Runtime
SEED="${SEED:-42}"
N_WORKERS="${N_WORKERS:-}"                 # blank → min(n_scenarios, cpu_count)
MAX_RUNS_PER_SCENARIO="${MAX_RUNS_PER_SCENARIO:-}"
MF6_EXE="${MF6_EXE:-$SCRIPT_DIR/.venv/bin/mf6}"

if [[ "$MF6_EXE" == */* && ! -x "$MF6_EXE" ]]; then
  echo "ERROR: mf6 executable not found: $MF6_EXE (see README for installation, or set MF6_EXE)" >&2
  exit 1
fi

CMD=(
  uv run python "$SCRIPT_DIR/run_classical_henry.py"
  --outdir        "$OUTDIR"
  --beta-c-values "$BETA_C_VALUES"
  --diffc-values  "$DIFFC_VALUES"
  --hk            "$HK"
  --por           "$POR"
  --rho0          "$RHO0"
  --c-sea         "$C_SEA"
  --inflow-min    "$INFLOW_MIN"
  --inflow-max    "$INFLOW_MAX"
  --random-field-len-scale "$RANDOM_FIELD_LEN_SCALE"
  --random-field-var       "$RANDOM_FIELD_VAR"
  --random-field-count     "$RANDOM_FIELD_COUNT"
  --ncol "$NCOL" --nlay "$NLAY" --lx "$LX" --lz "$LZ"
  --total-time "$TOTAL_TIME" --nstp "$NSTP" --skip "$SKIP"
  --seed "$SEED"
  --mf6-exe "$MF6_EXE"
  --overwrite
)
[[ -n "$N_WORKERS" ]] && CMD+=(--n-workers "$N_WORKERS")
[[ -n "$MAX_RUNS_PER_SCENARIO" ]] && CMD+=(--max-runs-per-scenario "$MAX_RUNS_PER_SCENARIO")

echo "Classical Henry dataset → $OUTDIR"
echo "  beta_c: $BETA_C_VALUES | D_C: $DIFFC_VALUES | Q ~ U[$INFLOW_MIN, $INFLOW_MAX]"
"${CMD[@]}"
