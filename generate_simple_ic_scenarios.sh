#!/usr/bin/env bash
# =============================================================================
# generate_simple_ic_scenarios.sh
#
# Initial-condition → full-trajectory 3-D dataset generator for the
# SIMPLIFIED Henry problem (same beta_c x diffc scenario grid and physics as
# generate_simple_coupling_scenarios.sh; only the tensor layout differs).
#
# Output layout:
#   <OUTDIR>/
#   ├── manifest.json
#   ├── scenario_001/
#   │   ├── scenario_manifest.json
#   │   └── scenario.npz        ← all runs batched: (n_runs, 3, T-1, nlay, ncol)
#   └── ...
#
# The raw MODFLOW series (t = 0 .. TOTAL_TIME) is downsampled by SKIP to T
# frames, then
#   input  → C0 repeated T-1 times      shape (3, T-1, nlay, ncol)
#            channels: concentration_0, beta_c, diffc
#   target → C_t, H_t for t_1..t_{T-1}   shape (2, T-1, nlay, ncol)
#
# Head is not an input: H(t=0) is only a zero placeholder because MODFLOW
# reports head from t = Δt onwards.
#
# Defaults (TOTAL_TIME=1.0, NSTP=100, SKIP=2) give T = 51 frames, i.e. 50
# target steps at Δt_eff = 0.02 ending exactly at t = 1.0.
#
# Shared settings (physics, scenario grid, initial condition, runtime) live in
# simple_coupling_config.sh.
#
# Usage:
#   ./generate_simple_ic_scenarios.sh [OUTDIR]
#
# All parameters can be overridden via environment variables.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/simple_coupling_config.sh"

# ---------------------------------------------------------------------------
# Time and downsampling controls
#
#   SKIP — temporal stride applied as time_series[::skip].
#          Choose NSTP divisible by SKIP so the last frame is t = TOTAL_TIME.
# ---------------------------------------------------------------------------
TOTAL_TIME="${TOTAL_TIME:-1.0}"
NSTP="${NSTP:-50}"
SKIP="${SKIP:-2}"

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTDIR="${1:-${OUTDIR:-${OUTBASEDIR}/grid_scenarios_ic_${INIT_METHOD}_skip${SKIP}_${NLAY}x${NCOL}}}"

# ---------------------------------------------------------------------------
# Build generation command
# ---------------------------------------------------------------------------
CMD=(uv run python run_simple_henry.py --mode ic_repeat)
append_common_args

# ---------------------------------------------------------------------------
# Print configuration and run
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Simplified Henry 3-D (space × time) IC → trajectory generator"
echo "  PDE:        elliptic flow + parabolic transport"
echo "  Storage:    Ss = 0  (no STO package)"
echo "  Influx:     zero    (no WEL / GHB)"
echo "  BCs:        homogeneous Dirichlet p=0, C=0 on all sides"
echo "============================================================"
print_common_config
echo "  layout:         input C0 repeated (3, T-1, ...) → target C,H (2, T-1, ...)"
echo "  command:        ${CMD[*]}"
echo "============================================================"

"${CMD[@]}"

# ---------------------------------------------------------------------------
# Compress the output directory into a tar.gz archive
# ---------------------------------------------------------------------------
archive_outdir
