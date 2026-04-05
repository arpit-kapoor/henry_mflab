# Henry Problem - Saltwater Intrusion Simulation

This project simulates the classic **Henry saltwater intrusion problem** using MODFLOW 6 with FloPy.

## 📁 Files

- **`run_henry.py`** - Main simulation script that runs the coupled flow-transport model
- **`animate_henry.py`** - Visualization script to create animations from simulation results
- **`requirements.txt`** - Python dependencies
- **`out/`** - Output directory containing simulation results and animations

## 🚀 Quick Start

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for fast Python environment management. Please [install uv](https://docs.astral.sh/uv/getting-started/installation/) before proceeding.

### 1. Setup Environment & Install MODFLOW 6

```bash
# Create the virtual environment and install Python dependencies (defined in pyproject.toml)
uv sync

# Download MODFLOW 6 executables into the virtual environment's bin folder
./.venv/bin/python -c "import flopy; flopy.utils.get_modflow(bindir='.venv/bin')"
```

### 2. Run the Simulation

```bash
# Run the script using the uv virtual environment
uv run python run_henry.py
```

Useful options for experiments:

```bash
# Configurable dataset generation (single unified workflow)
uv run python run_henry.py \
   --outdir ./data_configurable \
   --mf6-exe ./.venv/bin/mf6 \
   --beta-c-values 0.0,0.3,0.7,1.2 \
   --diffc-values 0.57024,0.28512 \
   --inflow-values 2.4,2.851,3.2 \
   --ghb-head-values 0.98,1.0,1.02 \
   --cinlet-values 34.5,35.0 \
   --strt-head-values 34.0,35.0 \
   --strt-conc-values 34.0,35.0 \
   --right-bc-kind ghb_head \
   --max-runs 200 \
   --save-timeseries
```

This generates:
- `data_configurable/<sample_id>/sample.npz` - FNO-ready sample with input/output tensors and metadata
- `data_configurable/manifest.json` - run metadata, failure logs, and train/val/test split IDs

### 2. Create Animation

```bash
uv run python animate_henry.py --skip 5 --fps 30
```

Options:
- `--workspace` - Input directory (default: `./out`)
- `--output` - Output filename (default: `henry_animation.mp4`)
- `--fps` - Frames per second (default: 20)
- `--dpi` - Resolution (default: 150)
- `--skip` - Plot every Nth time step (default: 1)

**Example outputs:**
- `out/henry_animation.mp4` - Video (227 KB, 3.3 seconds @ 30 fps)
- `out/henry_animation.gif` - GIF alternative (2.5 MB)

## 📐 Problem Description

### Physical Setup

**Classic Henry saltwater intrusion** in a 2D vertical cross-section:

```
Freshwater Inflow →             Ocean/Seawater (Hydrostatic)
┌─────────────────────────────────────┐  ← Top (z = 1.0 m)
│  Q = 5.7 m³/d              GHB Pkg  │
│  CONC = 0.0      ←flow→   CONC = 35 │
│         Saltwater wedge             │
│              ↗ (from ocean)         │
└─────────────────────────────────────┘  ← Bottom (z = 0.0 m)
├────────── 2.0 m ────────────────────┤
```

### Domain
- **Dimensions**: 2.0 m (horizontal) × 1.0 m (vertical)
- **Grid**: 80 columns × 40 layers × 1 row
- **Cell size**: Δx = 0.025 m, Δz = 0.025 m

### Simulation Time
- **Duration**: 0.5 days
- **Time steps**: 500 (Δt = 0.001 days)

### Physical Parameters

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Hydraulic conductivity (horizontal) | K_h | 864.0 | m/day |
| Hydraulic conductivity (vertical) | K_v | 864.0 | m/day |
| Porosity | θ | 0.35 | - |
| Molecular diffusion coefficient | D_m | 0.57024 | m²/day |
| Longitudinal dispersivity | α_L | 0.0 | m |
| Transverse dispersivity | α_T | 0.0 | m |
| Freshwater inflow rate | Q_in | 5.7024 | m³/day |
| Seawater concentration | C_ocean | 35.0 | g/L |

### Boundary Conditions

**Flow (GWF):**
- Left boundary: Specified flux (`WEL` package) total inflow = 5.7024 m³/d
- Right boundary: Hydrostatic mixed boundary (`GHB` package)
- Initial: Head = 35.0 everywhere

**Transport (GWT):**
- Left boundary (`WEL`): Concentration = 0.0 g/L (freshwater injection)
- Right boundary (`GHB`): Concentration = 35.0 g/L for water entering the domain
- Initial: Concentration = 35.0 g/L everywhere (domain is initially filled with seawater)

## 📊 Equations Solved

### 1. Groundwater Flow Equation

$$S_s \frac{\partial h}{\partial t} = \nabla \cdot (K \nabla h) + Q$$

Solves for hydraulic head **h** using Newton-Raphson with BiCGSTAB solver.

### 2. Advection-Dispersion Equation

$$\theta \frac{\partial C}{\partial t} = \nabla \cdot (\theta D \nabla C) - \nabla \cdot (\mathbf{q} C) + Q_s$$

Where:
- **Storage term**: $\theta \frac{\partial C}{\partial t}$
- **Dispersion term**: $\nabla \cdot (\theta D \nabla C)$
- **Advection term**: $\nabla \cdot (\mathbf{q} C)$

Solves for salt concentration **C** using UPSTREAM scheme for advection.

## 🔧 Model Components

### GWF (Groundwater Flow) Model
- **DIS**: Discretization
- **IC**: Initial conditions
- **NPF**: Node Property Flow (hydraulic conductivity)
- **GHB**: General-Head Boundary (hydrostatic right boundary)
- **WEL**: Well package (specified freshwater flux on left boundary)
- **BUY**: Buoyancy package (couples density/concentration to flow equations)
- **OC**: Output control (saves all time steps)

### GWT (Groundwater Transport) Model
- **DIS**: Discretization
- **IC**: Initial conditions
- **ADV**: Advection package (UPSTREAM scheme)
- **DSP**: Dispersion package (pure molecular diffusion, $D_m = 0.57024$, off-diagonal disabled)
- **SSM**: Source/Sink Mixing (transfers flow boundaries `WEL` and `GHB` values)
- **MST**: Mobile Storage Transfer (porosity)
- **OC**: Output control (saves all time steps)

## ⚠️ Important Notes

1. **Two-way density coupling**: Flow $\leftrightarrow$ Transport
   - Active variable-density coupling is enabled natively in MODFLOW 6 via the `BUY` (Buoyancy) package. The transport model maps seawater concentration back to hydraulic density driving the wedge.

2. **Coupling control for experiments**
   - Use `--beta-c-values` to dial coupling strength across the dataset.
   - Include `0.0` in `--beta-c-values` for an uncoupled baseline.

   - If MODFLOW 6 is not on PATH, pass `--mf6-exe ./.venv/bin/mf6`.

3. **Sharp front challenge controls**
   - Use `--diffc-values`, `--al-values`, and `--at-values` to change mixing-front sharpness.
   - Lower `diffc` and smaller dispersivities generally produce sharper fronts.

4. **Nonlinear density law caveat**
   - Current setup uses MODFLOW 6 `BUY`, which is linear in concentration.
   - Exponential density coupling is not natively available in this workflow and should be treated as a separate phase.

5. **File sizes**: Saving all 500 time steps creates ~13 MB files
   - To save only final state, change `saverecord=[("HEAD", "LAST")]` in `run_henry.py`

6. **Animation performance**: Use `--skip` option to reduce rendering time
   - `--skip 5`: renders every 5th time step (100 frames instead of 500)
   - `--skip 10`: renders every 10th time step (50 frames)

## 🎯 Use Cases

- Benchmark problem validation
- ML model training data
- Educational demonstrations
- Understanding coastal aquifer dynamics

## 📦 Output Files

```
out/
├── gwf.hds              # Hydraulic head (binary, all time steps)
├── gwt.ucn              # Concentration (binary, all time steps)
├── henry_final.npz      # Final state (NumPy arrays)
├── henry_animation.mp4  # Animation (MP4 video)
└── henry_animation.gif  # Animation (GIF, larger file)
```

Dataset outputs:

```
data_configurable/
├── sample_000001_beta0.000_diffc0.57024_in2.8510/
│   ├── gwf.hds
│   ├── gwt.ucn
│   └── sample.npz
├── ...
└── manifest.json
```

## 🛠️ Dependencies

All dependencies are defined in `pyproject.toml` and are managed automatically using `uv`. Core dependencies include:
- `flopy`
- `numpy`
- `matplotlib`
- `h5py` and `zarr` (optional output formats)

You will also need `ffmpeg` installed on your system if you want to save MP4 animations.

