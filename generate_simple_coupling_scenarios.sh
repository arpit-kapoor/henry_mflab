#!/usr/bin/env bash
# =============================================================================
# generate_simple_coupling_scenarios.sh
#
# Structured 3-D dataset generator for the SIMPLIFIED Henry problem:
#   - Full Cartesian grid of beta_c x diffc scenarios
#   - Zero specific storage (Ss = 0) — elliptic groundwater flow equation
#   - Zero influx — no freshwater inflow (WEL) or tidal forcing (GHB)
#   - Homogeneous Dirichlet BCs: p = 0 and C = 0 on all four sides
#   - Buoyancy-driven flow only via ρ(C) = ρ₀(1 + β_C C)
#
# Output layout:
#   <OUTDIR>/
#   ├── manifest.json
#   ├── scenario_001/
#   │   ├── scenario_manifest.json
#   │   └── scenario.npz        ← all runs batched: (n_runs, 4, T_in, nlay, ncol)
#   ├── scenario_002/
#   │   └── ...
#   └── ...
#
# Each run is a single 3-D data point. The raw MODFLOW time-series is first
# downsampled by SKIP (applied as [::skip]), then split at T_SPLIT:
#   input  → time steps [0,       T_SPLIT)   shape (4, T_in,  nlay, ncol)
#   target → time steps [T_SPLIT, T)         shape (2, T_out, nlay, ncol)
#
# Shared settings (physics, scenario grid, initial condition, runtime) live in
# simple_coupling_config.sh.
#
# Usage:
#   ./generate_simple_coupling_scenarios.sh [OUTDIR]
#
# All parameters can be overridden via environment variables.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/simple_coupling_config.sh"

# ---------------------------------------------------------------------------
# Time, 3-D split and downsampling controls
#
#   SKIP    — temporal stride applied as time_series[::skip] before splitting.
#             skip=1 uses every step; skip=2 uses every other step, etc.
#   T_SPLIT — number of downsampled steps assigned to the input window.
#             Leave blank to use the Python default (T // 2).
# ---------------------------------------------------------------------------
TOTAL_TIME="${TOTAL_TIME:-1.0}"
NSTP="${NSTP:-99}"
SKIP="${SKIP:-2}"
T_SPLIT="${T_SPLIT:-}"   # blank → Python default

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTDIR="${1:-${OUTDIR:-${OUTBASEDIR}/grid_scenarios_${INIT_METHOD}_skip${SKIP}_${NLAY}x${NCOL}_updated_coupling}}"

# ---------------------------------------------------------------------------
# Build generation command
# ---------------------------------------------------------------------------
CMD=(uv run python run_simple_henry.py --mode split)
append_common_args

# Optional t_split (omit to use Python default of T // 2)
if [[ -n "$T_SPLIT" ]]; then
  CMD+=(--t-split "$T_SPLIT")
fi

# ---------------------------------------------------------------------------
# Print configuration and run
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Simplified Henry 3-D (space × time) scenario generator"
echo "  PDE:        elliptic flow + parabolic transport"
echo "  Storage:    Ss = 0  (no STO package)"
echo "  Influx:     zero    (no WEL / GHB)"
echo "  BCs:        homogeneous Dirichlet p=0, C=0 on all sides"
echo "============================================================"
print_common_config
echo "  t_split:        ${T_SPLIT:-auto (T // 2)}"
echo "  command:        ${CMD[*]}"
echo "============================================================"

"${CMD[@]}"

# ---------------------------------------------------------------------------
# Compress the output directory into a tar.gz archive
# ---------------------------------------------------------------------------
archive_outdir

# ---------------------------------------------------------------------------
# Copy the archive to the RDS
# ---------------------------------------------------------------------------

# # RDS Location
# remote_user=${RDSUSER}
# remote_host=research-data-ext.sydney.edu.au
# remote_path=/rds/${RDSPROJECT}/data/simple_henry/

# local_file="${ARCHIVE_PATH}"
# filename="$(basename "$local_file")"

# echo "Local file: ${local_file}"
# echo "Remote path: ${remote_user}@${remote_host}:${remote_path}${filename}"

# # Transfer results to RDS
# sftp "${remote_user}@${remote_host}:${remote_path}" <<< "put -r ${local_file}"
