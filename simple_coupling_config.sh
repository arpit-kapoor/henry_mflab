# shellcheck shell=bash
# =============================================================================
# simple_coupling_config.sh
#
# Shared configuration for the simplified Henry beta_c x diffc scenario
# generators. Sourced (not executed) by:
#   - generate_simple_coupling_scenarios.sh  (split: [0, t_split) → [t_split, T))
#   - generate_simple_ic_scenarios.sh        (ic_repeat: C0 → full trajectory)
#
# Each generator defines its own time / downsampling window (TOTAL_TIME, NSTP,
# SKIP) and output directory name; everything else lives here.
#
# All parameters can be overridden via environment variables.
# Requires SCRIPT_DIR to be set by the sourcing script.
# =============================================================================

# ---------------------------------------------------------------------------
# Physical parameters (finalized simple Henry values)
# ---------------------------------------------------------------------------
BETA_C_VALUES="${BETA_C_VALUES:-"0.000005, 0.00002, 0.00007, 0.00025, 0.001"}"
DIFFC_VALUES="${DIFFC_VALUES:-"0.001, 0.003, 0.01, 0.03"}"
# hk: hydraulic conductivity [m/d].
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

RANDOM_FIELD_TYPE="${RANDOM_FIELD_TYPE:-grf}"
RANDOM_FIELD_SMOOTHNESS="${RANDOM_FIELD_SMOOTHNESS:-0.8}"
RANDOM_FIELD_LEN_SCALE="${RANDOM_FIELD_LEN_SCALE:-0.1, 0.3, 0.5, 0.7, 0.9}"
RANDOM_FIELD_VAR="${RANDOM_FIELD_VAR:-0.1, 0.3, 0.5, 0.7}"
RANDOM_FIELD_COUNT="${RANDOM_FIELD_COUNT:-20}"

C_X_TOE_VALUES="${C_X_TOE_VALUES:-"-0.5, 0.0, 0.5"}"
C_X_TOP_VALUES="${C_X_TOP_VALUES:-"1.0, 1.5, 2.0"}"
C_TRANS_WIDTH_VALUES="${C_TRANS_WIDTH_VALUES:-"0.001, 0.01, 0.05, 0.1"}"

# ---------------------------------------------------------------------------
# Grid controls
# ---------------------------------------------------------------------------
NCOL="${NCOL:-40}"
NLAY="${NLAY:-20}"
LX="${LX:-2.0}"
LZ="${LZ:-1.0}"

# ---------------------------------------------------------------------------
# Parallelism
#
#   N_WORKERS — number of parallel worker processes (one per scenario).
#             Leave blank to use min(n_scenarios, cpu_count).
#             Set to 1 to run sequentially (useful for debugging).
# ---------------------------------------------------------------------------
N_WORKERS="${N_WORKERS:-}"  # blank → auto (min(n_scenarios, cpu_count))

# ---------------------------------------------------------------------------
# Output base directory (each generator appends its own dataset name)
# ---------------------------------------------------------------------------
OUTBASEDIR="${OUTBASEDIR:-/Users/$USER/Projects/groundwater/data/simple_henry_data}"

# ---------------------------------------------------------------------------
# Dataset controls
# ---------------------------------------------------------------------------
SEED="${SEED:-42}"
MAX_RUNS_PER_SCENARIO="${MAX_RUNS_PER_SCENARIO:-}"

# ---------------------------------------------------------------------------
# Runtime controls
# ---------------------------------------------------------------------------
MF6_EXE="${MF6_EXE:-$SCRIPT_DIR/.venv/bin/mf6}"
SAVE_TIMESERIES="${SAVE_TIMESERIES:-0}"
SAVE_MODFLOW_FILES="${SAVE_MODFLOW_FILES:-0}"
OVERWRITE="${OVERWRITE:-1}"
KAPPA_FILE="${KAPPA_FILE:-}"

# ---------------------------------------------------------------------------
# Resolve scenario grid counts (for display only)
# ---------------------------------------------------------------------------
if [[ -z "${BETA_COUNT:-}" ]]; then
  BETA_COUNT=$(uv run python -c "print(len([x for x in '''$BETA_C_VALUES'''.split(',') if x.strip()]))")
fi
if [[ -z "${DIFFC_COUNT:-}" ]]; then
  DIFFC_COUNT=$(uv run python -c "print(len([x for x in '''$DIFFC_VALUES'''.split(',') if x.strip()]))")
fi

if [[ "$MF6_EXE" == */* && ! -x "$MF6_EXE" ]]; then
  echo "ERROR: mf6 executable not found or not executable: $MF6_EXE" >&2
  echo "Hint: set MF6_EXE to an absolute executable path, or install mf6 in PATH and set MF6_EXE=mf6." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Append the arguments shared by every generator to the CMD array.
# Expects OUTDIR, TOTAL_TIME, NSTP and SKIP to be set by the caller.
append_common_args() {
  CMD+=(
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
    --skip          "$SKIP"
    --seed          "$SEED"
    --mf6-exe       "$MF6_EXE"
  )

  # Optional n_workers (omit to use Python default of min(n_scenarios, cpu_count))
  if [[ -n "$N_WORKERS" ]]; then
    CMD+=(--n-workers "$N_WORKERS")
  fi

  if [[ "$INIT_METHOD" == "random" ]]; then
    CMD+=(
      --random-field-type       "$RANDOM_FIELD_TYPE"
      --random-field-smoothness "$RANDOM_FIELD_SMOOTHNESS"
      --random-field-len-scale  "$RANDOM_FIELD_LEN_SCALE"
      --random-field-var        "$RANDOM_FIELD_VAR"
      --random-field-count      "$RANDOM_FIELD_COUNT"
    )
  elif [[ "$INIT_METHOD" == "wedge" ]]; then
    CMD+=(
      --c0-x-toe-values       "$C_X_TOE_VALUES"
      --c0-x-top-values       "$C_X_TOP_VALUES"
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
}

# Print the shared part of the configuration banner.
print_common_config() {
  echo "  outdir:         $OUTDIR"
  echo "  grid:           nlay=$NLAY  ncol=$NCOL  Lx=$LX  Lz=$LZ"
  echo "  time:           total=$TOTAL_TIME d  nstp=$NSTP  dt=$(uv run python -c "print($TOTAL_TIME/$NSTP)") d"
  echo "  skip:           $SKIP  (dt_eff=$(uv run python -c "print($TOTAL_TIME/$NSTP*$SKIP)") d)"
  echo "  n_workers:      ${N_WORKERS:-auto (min(n_scenarios, cpu_count))}"
  echo "  beta_c values:  $BETA_C_VALUES (count=$BETA_COUNT)"
  echo "  diffc values:   $DIFFC_VALUES (count=$DIFFC_COUNT)"
  echo "  scenarios:      $((BETA_COUNT * DIFFC_COUNT))"
  echo "  hk values:      $HK_VALUES"
  echo "  por values:     $POR_VALUES"
  echo "  al / at:        $AL / $AT"
  echo "  rho0:           $RHO0 kg/m³"
  echo "  seed:           $SEED"
  echo "  save mf6 files: $SAVE_MODFLOW_FILES"
  echo "  mf6 exe:        $MF6_EXE"
}

# Compress OUTDIR into OUTDIR.tar.gz (next to it) and set ARCHIVE_PATH.
archive_outdir() {
  ARCHIVE_PATH="${OUTDIR%/}.tar.gz"
  tar -C "$(dirname "$OUTDIR")" -czf "$ARCHIVE_PATH" "$(basename "$OUTDIR")"

  echo
  echo "Done. Archive: ${ARCHIVE_PATH}"
  echo "See:  ${OUTDIR}/manifest.json"
}
