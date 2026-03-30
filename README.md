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

This generates:
- `out/gwf.hds` - Hydraulic head data (13 MB, 500 time steps)
- `out/gwt.ucn` - Concentration data (13 MB, 500 time steps)
- `out/henry_final.npz` - Final time step data (head and conc arrays)

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
Freshwater →                    Ocean/Seawater
┌─────────────────────────────────────┐  ← Top (z = 0)
│  HEAD = 1.0                HEAD = 0 │
│  CONC = 0        ←flow→    CONC = 0 │
│         Saltwater wedge             │
│              ↗ (from inlet)         │
└─────────────────────────────────────┘  ← Bottom (z = -1.0)
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
| Hydraulic conductivity (horizontal) | K_h | 1.0 | m/day |
| Hydraulic conductivity (vertical) | K_v | 1.0 | m/day |
| Porosity | θ | 0.35 | - |
| Longitudinal dispersivity | α_L | 0.1 | m |
| Transverse dispersivity | α_T | 0.01 | m |
| Inlet concentration | C_inlet | 35.0 | g/L |

### Boundary Conditions

**Flow (GWF):**
- Left boundary: Constant head = 1.0 m (freshwater inflow)
- Right boundary: Constant head = 0.0 m (ocean level)
- Initial: Head = 0.0 everywhere

**Transport (GWT):**
- Left boundary: Constant concentration = 35.0 g/L (saltwater)
- Right boundary: Concentration = 0.0 g/L
- Initial: Concentration = 0.0 everywhere

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
- **CHD**: Constant head boundaries
- **OC**: Output control (saves all time steps)

### GWT (Groundwater Transport) Model
- **DIS**: Discretization
- **IC**: Initial conditions
- **ADV**: Advection package (UPSTREAM scheme)
- **DSP**: Dispersion package
- **CNC**: Constant concentration boundary
- **SSM**: Source/Sink Mixing
- **MST**: Mobile Storage Transfer (porosity)
- **OC**: Output control (saves all time steps)

## ⚠️ Important Notes

1. **One-way coupling**: Flow → Transport (not density-dependent)
   - For true variable-density flow, use SEAWAT or MODFLOW 6 Buy package
   - This simplified setup is common for ML training datasets

2. **File sizes**: Saving all 500 time steps creates ~13 MB files
   - To save only final state, change `saverecord=[("HEAD", "LAST")]` in `run_henry.py`

3. **Animation performance**: Use `--skip` option to reduce rendering time
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

## 🛠️ Dependencies

All dependencies are defined in `pyproject.toml` and are managed automatically using `uv`. Core dependencies include:
- `flopy`
- `numpy`
- `matplotlib`
- `h5py` and `zarr` (optional output formats)

You will also need `ffmpeg` installed on your system if you want to save MP4 animations.

