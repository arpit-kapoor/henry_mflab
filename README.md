# Henry Problem – Dynamic Saltwater Intrusion Simulation & Dataset Generator

This repository simulates transient, coupled variable-density groundwater flow and solute transport based on the Henry saltwater intrusion problem using [MODFLOW 6](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model) via [FloPy](https://github.com/modflowpy/flopy). It provides an automated pipeline to generate windowed spatio-temporal datasets for surrogate and machine learning model training.

---

## 🚀 Environment Setup

### Prerequisites
- Python $\ge 3.12$
- [uv](https://docs.astral.sh/uv/) for Python package and environment management

### 1. Install Python Dependencies
```bash
uv sync
```

### 2. Download MODFLOW 6 Executable
Download the USGS MODFLOW 6 binary into the local virtual environment:
```bash
uv run python -c "import flopy; flopy.utils.get_modflow(bindir='.venv/bin')"
```
Verify that `.venv/bin/mf6` exists and is executable.

---

## 🌊 What is Simulated

The model solves coupled transient groundwater flow (GWF) and solute transport (GWT) in a 2D vertical cross-section (default: $L_x = 8.0\text{ m}, L_z = 4.0\text{ m}$, discretized into $40 \text{ columns} \times 20 \text{ layers}$).

```
Inland (Freshwater Inflow)                   Coastal Boundary (Dynamic Tides)
┌──────────────────────────────────────────────┐  z = Lz (4.0 m)
│  Stochastic Shot-Noise Inflow (WEL)          │
│  - Poisson storm arrivals                    │  M2 semi-diurnal tides +
│  - Log-normal peak amplitudes                │  spring-neap envelope (GHB)
│  - Exponential baseflow recession            │  + dynamic per-layer salinity:
│  - AR(1) background noise                    │    - Submerged: 35.0 g/L
│  - Optional wetting / drying drift           │    - Exposed:    0.0 g/L
│                        Saltwater Wedge       │
│                             ↗                │
└──────────────────────────────────────────────┘  z = 0.0 m
├─────────────────── Lx (8.0 m) ───────────────┤
```

### Physical & Numerical Dynamics (`henry_data/simulation.py`)
- **Transient Storage & Flow (`GWF`, `STO`)**: Specific storage ($S_s$) and specific yield ($S_y$) enable realistic transient water-table and pressure response.
- **Variable-Density Coupling (`BUY`)**: Fluid density depends linearly on concentration via buoyancy coefficient $\beta_c$.
- **Solute Transport (`GWT`)**: Solves advection (upstream weighting) and dispersion/diffusion with molecular diffusion $D_m$ (`diffc`) and dispersivities ($\alpha_L, \alpha_T$).
- **Dynamic Coastal Boundary (Right, `GHB`)**:
  - Head varies according to an M2 semi-diurnal tidal carrier ($T \approx 12.42\text{ h}$) modulated by a fortnightly spring-neap envelope ($T_{\text{sn}} \approx 14.77\text{ days}$) with optional sea-level rise drift and Gaussian noise.
  - Saltwater inlet concentration at each vertical layer is dynamic: cells submerged by the instantaneous tide receive seawater concentration ($35\text{ g/L}$), while exposed cells receive freshwater ($0\text{ g/L}$).
- **Dynamic Inland Inflow (Left, `WEL`)**:
  - Multi-scale freshwater injection driven by a stochastic shot-noise process: storm arrivals sampled from a Poisson process, storm pulse amplitudes sampled from a log-normal distribution, exponential recession decay, and autocorrelated AR(1) noise.
- **Spin-up Period**: Each run optionally performs a warm-start pre-run (`--spinup-time`) so the salinity wedge reaches a realistic dynamic state before dataset collection begins.

---

## 📦 Data Generation

### 1. Automated Coupling & Diffusion Grid Generation
To generate a complete Cartesian grid of coupling scenarios ($\beta_c \times D_m$) across hydrological parameter variations:

```bash
./generate_coupling_scenarios.sh [OUTDIR] [LAG]
```
- **`OUTDIR`** (default: `~/Projects/groundwater/data/henry_data/grid_scenarios_realistic_20x40`): Destination directory.
- **`LAG`** (default: `1` step): Prediction time-lag between input and target states.

#### Pipeline Flow:
1. **Simulation Phase (`run_henry.py` / `henry_data.cli`)**: Runs simulation batches across combinations of $\beta_c$, $D_m$, hydraulic conductivity ($K$), porosity ($\theta$), inflow ($Q$), and tidal heads into a temporary `_raw_generation/` directory.
2. **Reorganization Phase (`henry_data.reorganize`)**: Restructures raw runs into a standardized numeric format (`scenario_01/run_000001/`), writes scenario and run metadata JSON files, and removes temporary raw files.

#### Key Environment Variable Overrides:
```bash
# Example: 3x3 scenario grid over 60 simulated days with custom lag in days
BETA_MIN=0.1 BETA_MAX=1.0 BETA_COUNT=3 \
DIFFC_MIN=0.01 DIFFC_MAX=0.1 DIFFC_COUNT=3 \
TOTAL_TIME=60 NSTP=480 LAG_DAYS=1 \
./generate_coupling_scenarios.sh ./data/my_scenarios 1
```

### 2. Single Scenario Generation
To generate runs for a single fixed $(\beta_c, D_m)$ scenario:
```bash
./generate_one_coupling_scenario.sh [OUTDIR] [BETA_C] [DIFFC] [LAG]
# Example:
./generate_one_coupling_scenario.sh ./data/single_scenario 0.7 0.57024 1
```

### 3. Direct Python CLI
You can invoke the simulation generator directly via Python:
```bash
uv run python run_henry.py \
  --outdir ./out_custom \
  --scenario-pairs 0.7:0.57024 \
  --dynamic-inflow \
  --dynamic-tides \
  --add-storage \
  --total-time 30 \
  --nstp 240 \
  --lag 1
```

---

## 📂 Output Dataset Structure

Generated datasets are organized as follows:

```
<OUTDIR>/
├── scenarios_manifest.json           # Top-level index of all scenarios and run counts
└── scenarios/
    ├── scenario_01/
    │   ├── scenario_config.json      # Scenario parameters (beta_c, diffc, lag)
    │   ├── runs_config.json          # Per-run parameter details and execution logs
    │   ├── run_000001/
    │   │   └── windows.npz           # Windowed input/output tensors for ML
    │   └── run_000002/
    │       └── windows.npz
    └── scenario_02/
        └── ...
```

### `windows.npz` Arrays:
- **`input_tensor`** `[N_windows, N_channels, N_lay, N_col]`:
  - Channels:
    1. `concentration_t`: Salt concentration at time $t$
    2. `head_t`: Hydraulic head at time $t$
    3. `flux_left_boundary`: Inland freshwater inflow flux
    4. `ghb_flux_right_boundary`: Coastal tidal boundary head / flux
    5. `cinlet_right_boundary`: Per-layer coastal salinity inlet condition
    6. `beta_c`: Fluid density coupling coefficient
    7. `diffc`: Molecular diffusion coefficient
    8. *(Optional)* `tidal_phase`: Current M2 tidal phase $[0, 2\pi]$ (if enabled)
- **`output_tensor`** `[N_windows, 2, N_lay, N_col]`:
  - Channel 0: Salt concentration at future time $t + \Delta t_{\text{lag}}$
  - Channel 1: Hydraulic head at future time $t + \Delta t_{\text{lag}}$
- **`t_index`** & **`t_lag_index`**: Time-step indices matching input and prediction horizons.

---

## 🧂 Forced Henry (intermediate setup, `henry_forced/`)

Sits between `simple_henry/` (closed box, buoyancy only) and `henry_data/` (tides, storms, storage). It uses the classic [MF6 Henry](https://modflow6-examples.readthedocs.io/en/latest/_examples/ex-gwt-henry.html) boundary conditions so a saltwater wedge forms:

- **Left (WEL)**: constant freshwater inflow $Q$ split evenly over the column, $C = 0$
- **Right (GHB)**: sea-level head $h = L_z$, inflowing water carries $C = 35$ kg/m³
- **Top / bottom**: no-flow; no storage package (quasi-steady flow)
- Defaults: $K = 864$ m/d, $\theta = 0.35$, $\rho = \rho_0(1 + \beta_C C)$

Scenarios are the $\beta_C \times D_m$ grid ($\beta_C \le 0.0014$ by default; larger values push the wedge onto the inland boundary). Runs differ by a random initial concentration field and a constant inflow $Q \sim U[2, 6]$ m³/d (classic Henry: 5.7024, low-inflow: 2.851). Each scenario draws its own run set from seed `[SEED, scenario_index]`, so the dataset has a distinct initial field for every run. `SHARED_RUNS=1` reuses one run set in every scenario (paired design).

Since no boundary fixes the concentration, the random initial field is **not** tapered to zero at the edges (unlike `simple_henry`). It is also sampled with period 2 relative to the domain, so opposite edges are not correlated. `RANDOM_FIELD_TAPER=1 RANDOM_FIELD_PERIOD=1.0` restores the simple_henry-style field.

```bash
./generate_henry_forced_scenarios.sh [OUTDIR]
# quick test
MAX_RUNS_PER_SCENARIO=3 BETA_C_VALUES=0.0007 DIFFC_VALUES=0.57024 ./generate_henry_forced_scenarios.sh ./forced_test
```

Each `scenario_NNN/scenario.npz` holds:
- **`input_tensor`** `[n_runs, 7, T-1, nlay, ncol]`: `concentration_0`, `inflow` (Q broadcast over the domain; `INFLOW_ENCODING=left_column` puts it in the inflow column only), `beta_c`, `diffc`, all repeated over time, then `coord_t`, `coord_z`, `coord_x` (frame time [d], cell-centre elevation and distance [m]). The coordinates are the same for every run. They tell the model where the boundaries are, since every other input except C0 is constant in space (`COORD_CHANNELS=0` drops them)
- **`output_tensor`** `[n_runs, 2, T-1, nlay, ncol]`: concentration and head at $t_1 \dots t_{T-1}$
- **`inflow`** `[n_runs]` and **`run_params`** (per-run JSON with IC parameters, field seed and Q)

`--init-method seawater` starts from a domain full of seawater (the classic Henry IC), which is useful for checking the setup against the MF6 example.

---

## 📊 Exploration & Visualization

- **Notebooks**:
  - `notebooks/test_scenarios.ipynb` – Inspect and validate generated scenario datasets and manifests.
  - `notebooks/test_data.ipynb` – Visualize windowed tensor channels, concentration fields, and boundary hydrographs.
  - `notebooks/test_boundaries.ipynb` – Explore and tune stochastic inflow and tidal boundary parameters.
- **Animation**:
  - `animate_henry.py` – Render 2D spatial video / GIF animations of head and salinity fields across time steps.
