#!/usr/bin/env bash
# =============================================================================
# generate_simple_henry.sh
#
# Dataset generator for the SIMPLIFIED Henry problem:
#   - Zero specific storage (Ss = 0) — elliptic groundwater flow equation
#   - Zero influx — no freshwater inflow (WEL) or tidal forcing (GHB)
#   - Homogeneous Dirichlet BCs: p = 0 and C = 0 on all four sides
#   - Buoyancy-driven flow only via ρ(C) = ρ₀(1 + β_C C)
#
# Usage:
#   ./generate_simple_henry.sh [OUTDIR]
#
# Example:
#   ./generate_simple_henry.sh ./simple_henry_data
#   BETA_C_VALUES="0.5,0.7,1.0" DIFFC_VALUES="0.28512,0.57024" \
#     ./generate_simple_henry.sh ./simple_henry_sweep
#
# All parameters can be overridden via environment variables.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTDIR="${1:-/Users/$USER/Projects/groundwater/data/simple_henry_data}"

# ---------------------------------------------------------------------------
# Physical parameters
# Classic Henry benchmark values:
#   β_C    = 0.7   [m³/kg]
#   diffc  = 0.57024 [m²/d]
#   hk     = 864.0 [m/d]
#   por    = 0.35
# ---------------------------------------------------------------------------
BETA_C_VALUES="${BETA_C_VALUES:-"0.1"}"
# diffc: effective diffusion coefficient [m²/d].
# For a 1-day run, diffc=0.01 gives a diffusion timescale τ_diff = Lz²/diffc = 100 days >> 1 day.
# This prevents premature decay while keeping the system in a convective regime.
DIFFC_VALUES="${DIFFC_VALUES:-"0.01"}"
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
INIT_METHOD="${INIT_METHOD:-random}"  # Options: "wedge", "random"

RANDOM_FIELD_TYPE="${RANDOM_FIELD_TYPE:-"perlin", "mattern", "grf"}"  # Options: "fourier_1d", "fourier", "polynomial", "grbf", "white_noise", "perlin", "grf", "matern", "exp".
RANDOM_FIELD_SMOOTHNESS="${RANDOM_FIELD_SMOOTHNESS:-1.0, 0.5, 0.2, 0.05}"
RANDOM_FIELD_COUNT="${RANDOM_FIELD_COUNT:-2}"  # In case of random fields, this determines the number of runs per scenario (i.e., how many random fields to generate for each parameter combination).

C_X_TOE_VALUES="${C_X_TOE_VALUES:-"-0.5, 0.0, 0.5"}"
C_X_TOP_VALUES="${C_X_TOP_VALUES:-"1.0, 1.5"}"
C_TRANS_WIDTH_VALUES="${C_TRANS_WIDTH_VALUES:-"0.01"}"

# ---------------------------------------------------------------------------
# Grid / time controls
# Default: 40×20 grid, 2 m × 1 m domain, 1-day run, 100 steps (Δt = 0.01 d)
# ---------------------------------------------------------------------------
NCOL="${NCOL:-40}"
NLAY="${NLAY:-20}"
LX="${LX:-2.0}"
LZ="${LZ:-1.0}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
NSTP="${NSTP:-50}"

# ---------------------------------------------------------------------------
# Prediction lag (steps)
# ---------------------------------------------------------------------------
LAG="${LAG:-1}"

# ---------------------------------------------------------------------------
# Dataset / split controls
# ---------------------------------------------------------------------------
SEED="${SEED:-42}"
MAX_RUNS_PER_SCENARIO="${MAX_RUNS_PER_SCENARIO:-}"

# ---------------------------------------------------------------------------
# Runtime controls
# ---------------------------------------------------------------------------
MF6_EXE="${MF6_EXE:-./.venv/bin/mf6}"
SAVE_TIMESERIES="${SAVE_TIMESERIES:-0}"
SAVE_MODFLOW_FILES="${SAVE_MODFLOW_FILES:-1}"
OVERWRITE="${OVERWRITE:-0}"
KAPPA_FILE="${KAPPA_FILE:-}"

# ---------------------------------------------------------------------------
# Animation controls (post-generation)
# ---------------------------------------------------------------------------
GENERATE_ANIMATION="${GENERATE_ANIMATION:-1}"
ANIMATE_FPS="${ANIMATE_FPS:-20}"
ANIMATE_DPI="${ANIMATE_DPI:-150}"
ANIMATE_SKIP="${ANIMATE_SKIP:-1}"

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD=(
  uv run python run_simple_henry.py
  --outdir        "$OUTDIR"
  --ncol          "$NCOL"
  --nlay          "$NLAY"
  --lx            "$LX"
  --lz            "$LZ"
  --total-time    "$TOTAL_TIME"
  --nstp          "$NSTP"
  --init-method   "$INIT_METHOD"
  --beta-c-values "$BETA_C_VALUES"
  --diffc-values  "$DIFFC_VALUES"
  --hk-values     "$HK_VALUES"
  --por-values    "$POR_VALUES"
  --al            "$AL"
  --at            "$AT"
  --rho0          "$RHO0"
  --lag           "$LAG"
  --seed          "$SEED"
  --mf6-exe       "$MF6_EXE"
)

if [[ "$INIT_METHOD" == "random" ]]; then
  CMD+=(
    --random-field-type       "$RANDOM_FIELD_TYPE"
    --random-field-smoothness "$RANDOM_FIELD_SMOOTHNESS"
    --random-field-count     "$RANDOM_FIELD_COUNT"
  )
elif [[ "$INIT_METHOD" == "wedge" ]]; then
  CMD+=(
    --c0-x-toe-values   "$C_X_TOE_VALUES"
    --c0-x-top-values   "$C_X_TOP_VALUES"
    --c0-trans-width-values "$C_TRANS_WIDTH_VALUES"
  )
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
echo "  Simplified Henry dataset generator"
echo "  PDE:      elliptic flow + parabolic transport"
echo "  Storage:  Ss = 0  (no STO package)"
echo "  Influx:   zero    (no WEL / GHB)"
echo "  BCs:      homogeneous Dirichlet p=0, C=0 on all sides"
echo "============================================================"
echo "  outdir:         $OUTDIR"
echo "  grid:           nlay=$NLAY  ncol=$NCOL  Lx=$LX  Lz=$LZ"
echo "  time:           total=$TOTAL_TIME d  nstp=$NSTP  dt=$(echo "scale=4; $TOTAL_TIME/$NSTP" | bc) d"
if [[ "$INIT_METHOD" == "random" ]]; then
  echo "  initial C:      random (type=$RANDOM_FIELD_TYPE  smoothness=$RANDOM_FIELD_SMOOTHNESS  count=$RANDOM_FIELD_COUNT)"
else
  echo "  initial C:      wedge (x_toe=$C_X_TOE_VALUES  x_top=$C_X_TOP_VALUES  trans_width=$C_TRANS_WIDTH_VALUES)"
fi
echo "  beta_c values:  $BETA_C_VALUES"
echo "  diffc values:   $DIFFC_VALUES"
echo "  hk values:      $HK_VALUES"
echo "  por values:     $POR_VALUES"
echo "  al / at:        $AL / $AT"
echo "  rho0:           $RHO0 kg/m³"
echo "  lag:            $LAG step(s)"
echo "  save mf6 files: $SAVE_MODFLOW_FILES"
echo "  mf6 exe:        $MF6_EXE"
echo "  animate:        $GENERATE_ANIMATION  (fps=$ANIMATE_FPS  dpi=$ANIMATE_DPI  skip=$ANIMATE_SKIP)"
echo "  command:        ${CMD[*]}"
echo "============================================================"

"${CMD[@]}"

if [[ "$GENERATE_ANIMATION" == "1" && "$SAVE_MODFLOW_FILES" != "1" ]]; then
  echo
  echo "Skipping animation because SAVE_MODFLOW_FILES=0 removed gwf.hds/gwt.ucn."
  echo "Set SAVE_MODFLOW_FILES=1 to enable animation output."
fi

if [[ "$GENERATE_ANIMATION" == "1" && "$SAVE_MODFLOW_FILES" == "1" ]]; then
  # Resolve the run workspace from manifest.json (last successful run).
  RUN_WORKSPACE="$(uv run python - "$OUTDIR" <<'PY'
import json
import pathlib as pl
import sys

outdir = pl.Path(sys.argv[1])
manifest_path = outdir / "manifest.json"

if not manifest_path.exists():
    raise SystemExit(f"manifest not found: {manifest_path}")

with manifest_path.open("r", encoding="utf-8") as fp:
    manifest = json.load(fp)

candidates = [
    r for r in manifest.get("runs", [])
    if r.get("status") in {"ok", "skipped"}
]

if not candidates:
    raise SystemExit(
        f"No successful run workspace found in {manifest_path}"
    )

print(candidates[-1]["workspace"])
PY
)"

  ANIMATE_CMD=(
    uv run python animate_simple_henry.py
    --dataset-path "$OUTDIR"
    --run-path     "$RUN_WORKSPACE"
    --all
    --fps          "$ANIMATE_FPS"
    --dpi          "$ANIMATE_DPI"
    --skip         "$ANIMATE_SKIP"
  )

  echo
  echo "Generating animation from saved outputs"
  echo "  run path:  $RUN_WORKSPACE"
  echo "Generating animations for all runs from saved outputs"
  echo "  command:   ${ANIMATE_CMD[*]}"
  "${ANIMATE_CMD[@]}"
fi

echo
echo "Done. See manifest: $OUTDIR/manifest.json"
