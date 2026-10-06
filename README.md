# Data generation for "Fourier Neural Operator Emulation for Coupled Parametric PDEs"

This code generates the two datasets used in the paper. Both are produced with
[MODFLOW 6](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model) through
[FloPy](https://github.com/modflowpy/flopy). Each dataset couples variable-density groundwater flow
(an elliptic equation, with no storage) to solute transport (a parabolic equation) through
ρ(C) = ρ₀(1 + β_C C).

| Dataset | Paper | Package | Script |
|---|---|---|---|
| Simplified Henry-type problem | Sec. 5 | [`simplified_henry/`](simplified_henry) | [`generate_simplified_henry.sh`](generate_simplified_henry.sh) |
| Classical Henry-type problem | App. I | [`classical_henry/`](classical_henry) | [`generate_classical_henry.sh`](generate_classical_henry.sh) |

The full datasets (about 0.8 GB and 0.6 GB) are not included because of their size. The scripts
regenerate them. [`notebooks/visualise_datasets.ipynb`](notebooks/visualise_datasets.ipynb) shows
example trajectories from both datasets, and its outputs are saved so you can view them without
running anything.

---

## Setup

You need Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # flopy, numpy, gstools
uv sync --group notebook                  # optional: matplotlib + ipykernel for the notebook

# MODFLOW 6 binary (version 6.6.3 was used for the paper) into .venv/bin/mf6
uv run python -c "import flopy; flopy.utils.get_modflow(bindir='.venv/bin', repo='modflow6', release_id='6.6.3')"
```

The scripts look for `.venv/bin/mf6`. To use another binary, set `MF6_EXE=/path/to/mf6`.

## Generating the datasets

```bash
# quick test: 2 runs per scenario (seconds)
MAX_RUNS_PER_SCENARIO=2 ./generate_simplified_henry.sh ./data/test_simplified
MAX_RUNS_PER_SCENARIO=2 ./generate_classical_henry.sh  ./data/test_classical

# full datasets as used in the paper
./generate_simplified_henry.sh    # → ./data/simplified_henry   (16 scenarios × 400 runs)
./generate_classical_henry.sh     # → ./data/classical_henry    (12 scenarios × 400 runs)
```

Each scenario runs in its own process (set `N_WORKERS` to change this). A single run takes
about 0.3 s, so a full dataset needs about 40 CPU-minutes (5–10 minutes on a 14-core laptop). Every setting in the scripts can be overridden from the
environment, for example `BETA_C_VALUES="0.0007" DIFFC_VALUES="0.57024" ./generate_classical_henry.sh`.
The Python entry points `run_simplified_henry.py` and `run_classical_henry.py` take the same
options as command-line flags (`--help`).

Scenario *i* draws its initial fields (and, for the classical problem, its inflows) from the seed
`[SEED, i]` (`SEED=42` by default), so the datasets are deterministic.

## Problem settings

Both problems use Ω = (0, 2 m) × (0, 1 m) on a 40 × 20 grid (Δx = Δz = 0.05 m), with porosity
η = 0.35, ρ₀ = 1000 kg m⁻³, and pure molecular diffusion D = D_C I (zero dispersivity). Each run
is simulated with 50 MODFLOW time steps, and every second step is kept, giving 25 target frames.

| | Simplified (Sec. 5) | Classical (App. I) |
|---|---|---|
| Hydraulic conductivity K | 50 m d⁻¹ | 864 m d⁻¹ |
| Time horizon / frame spacing | 1 d / 0.04 d | 0.25 d / 0.01 d |
| β_C [m³ kg⁻¹] | 2×10⁻⁵, 7×10⁻⁵, 2.5×10⁻⁴, 10⁻³ | 3.5×10⁻⁴, 7×10⁻⁴, 1.4×10⁻³ |
| D_C [m² d⁻¹] | 10⁻³, 3×10⁻³, 10⁻², 3×10⁻² | 0.05, 0.1, 0.3, 0.57024 |
| Concentration BCs | C = 0 on all sides (CNC) | inflow C = 0 left (WEL), C = 35 where sea water enters right (GHB), no-flux top/bottom |
| Head BCs | h = 0 left/right, no-flow top/bottom (see below) | inflow Q left, h = L_z right, no-flow top/bottom |
| Forcing | none (buoyancy only) | Q ~ U[2, 6] m² d⁻¹ per run |
| Initial condition C₀ | GRF, shifted to ≥ 0, raised-cosine taper (b = 0.1), boundary cells set to 0, scaled to [0, 35] | GRF (period 2, no taper), scaled to [0, 35] |

The GRF has a Gaussian covariance with length scale ℓ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} and variance
σ² ∈ {0.1, 0.3, 0.5, 0.7}, both in normalised coordinates. There are 20 realisations per (ℓ, σ²)
pair, giving 400 runs per scenario.

**Head boundary condition (simplified problem).** The dataset used in the paper fixes h = 0 on the
left and right walls only, so the top and bottom are no-flow (`--head-bc lr`, the default).
`HEAD_BC=all ./generate_simplified_henry.sh` sets h = 0 on all four sides instead.

**Initial field on the boundary (simplified problem).** The taper w_b(x̃) w_b(z̃) of Eq. (22) is
evaluated at the cell centres. These lie half a cell inside ∂Ω, so the outer ring of cells (the
cells that carry the C = 0 condition) is then set to exactly 0, and C₀ satisfies the boundary
condition on the grid. Note: the simplified dataset used for the paper's experiments was generated
with an earlier version, which read the taper from the nearest lower node of a
(ncol + 1) × (nlay + 1) grid. There, C₀ was not exactly zero on the top row and right column.
In both versions the CNC package holds all boundary cells at C = 0 from the first time step.

## Output format

```
<OUTDIR>/
├── manifest.json                 # settings, channel names, shapes, per-scenario summary
└── scenario_NNN/
    ├── scenario_manifest.json
    └── scenario.npz              # all runs of one (β_C, D_C) scenario
```

Keys in `scenario.npz`:

| Key | Simplified | Classical |
|---|---|---|
| `input_tensor` | `(n_runs, 3, 25, 20, 40)`: C₀, β_C, D_C | `(n_runs, 7, 25, 20, 40)`: C₀, Q, β_C, D_C, t, z, x |
| `output_tensor` | `(n_runs, 2, 25, 20, 40)`: C, h | same |
| `times_out` | target times t₁ … t₂₅ [d] | same |
| `run_params` | per-run JSON (GRF ℓ, σ², sample index, field seed) | adds the inflow Q |
| `inflow` | – | `(n_runs,)` Q per run |

The input channels are repeated along the time axis, so input and output share the
(t, z, x) = (25, 20, 40) grid of a 3-D FNO. Layer index 0 is the top of the aquifer and column 0
is the left (inland) boundary. Concentration is in kg m⁻³ and head is equivalent freshwater head
in m.

## Repository layout

```
simplified_henry/      simulation.py (MODFLOW 6 model), generators.py (dataset), cli.py, init_functions.py (GRF)
classical_henry/       simulation.py, generators.py, cli.py (reuses the GRF and helpers above)
generate_*.sh          dataset scripts with the paper settings
run_*.py               Python entry points
notebooks/             visualise_datasets.ipynb
```

## Third-party software

MODFLOW 6 (U.S. Geological Survey, public domain), FloPy (CC0) and GSTools (LGPL-3.0).
This code is released under the MIT licence (see `LICENSE`).
