#!/usr/bin/env python
"""
Animate simplified Henry problem results: head and concentration fields over time.

Reads domain dimensions (Lx, Lz) from the dataset manifest.json so the animation
is always spatially correct regardless of the grid used.

Usage:
    uv run python animate_simple_henry.py --dataset-path /path/to/simple_henry_data
    uv run python animate_simple_henry.py --dataset-path /path/to/data --run-path /path/to/run
"""
import argparse
import json
import pathlib as pl
import re

import flopy.utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _find_run_dirs(path: pl.Path) -> list[pl.Path]:
    """Return run directories that contain both gwf.hds and gwt.ucn."""
    if (path / "gwf.hds").exists() and (path / "gwt.ucn").exists():
        return [path]

    run_dirs = []
    for run_dir in sorted(path.glob("scenario_*/run_*")):
        if (run_dir / "gwf.hds").exists() and (run_dir / "gwt.ucn").exists():
            run_dirs.append(run_dir)
    for run_dir in sorted(path.glob("run_*")):
        if (run_dir / "gwf.hds").exists() and (run_dir / "gwt.ucn").exists():
            run_dirs.append(run_dir)
    return run_dirs


def _resolve_run_workspace(
    dataset_path: pl.Path, run_path: str | None
) -> tuple[pl.Path, list[pl.Path]]:
    if run_path is not None:
        ws = pl.Path(run_path)
        if not ((ws / "gwf.hds").exists() and (ws / "gwt.ucn").exists()):
            raise FileNotFoundError(
                f"run_path must contain gwf.hds and gwt.ucn: {ws}"
            )
        return ws, _find_run_dirs(dataset_path)

    run_dirs = _find_run_dirs(dataset_path)
    if not run_dirs:
        raise FileNotFoundError(
            "No run workspace with gwf.hds + gwt.ucn found under "
            f"{dataset_path}"
        )
    if len(run_dirs) > 1:
        raise ValueError(
            "Multiple run workspaces found. Pass --run-path to select one.\n"
            f"Found {len(run_dirs)} runs under {dataset_path}"
        )
    return run_dirs[0], run_dirs


def _load_manifest_grid(dataset_path: pl.Path) -> dict:
    """Load grid dimensions from manifest.json, falling back to defaults."""
    manifest_path = dataset_path / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fp:
            manifest = json.load(fp)
        grid = manifest.get("grid", {})
        return {
            "lx": float(grid.get("lx", 2.0)),
            "lz": float(grid.get("lz", 1.0)),
        }
    # Fallback to simple Henry defaults
    return {"lx": 2.0, "lz": 1.0}


def _default_output_path(dataset_path: pl.Path, run_workspace: pl.Path) -> pl.Path:
    if dataset_path.name.startswith("run_"):
        base_dir = dataset_path
    elif dataset_path.name.startswith("scenario_"):
        base_dir = dataset_path.parent
    else:
        base_dir = dataset_path

    scenario_name = run_workspace.parent.name
    run_name = run_workspace.name
    out_name = (
        f"simple_henry_animation"
        f"_{_slugify(scenario_name)}"
        f"_{_slugify(run_name)}.mp4"
    )
    return base_dir / out_name


def _fixed_levels(vmin: float, vmax: float, n_levels: int = 21) -> np.ndarray:
    if np.isclose(vmin, vmax):
        eps = max(1e-6, abs(vmin) * 1e-6)
        return np.linspace(vmin - eps, vmax + eps, n_levels)
    return np.linspace(vmin, vmax, n_levels)


# ---------------------------------------------------------------------------
# Main animation function
# ---------------------------------------------------------------------------

def animate_simple_henry(
    dataset_path,
    output_video=None,
    fps: int = 20,
    dpi: int = 150,
    skip_frames: int = 1,
    run_path=None,
    dynamic_scales: bool = False,
    lx: float | None = None,
    lz: float | None = None,
):
    """Create an animation of the simplified Henry problem results.

    Parameters
    ----------
    dataset_path : str or Path
        Run workspace or dataset root containing scenario/run workspaces.
    output_video : str or None
        Output video path.  If relative, resolved under *dataset_path*.
        If None, auto-generated in the dataset root directory.
    fps : int
        Frames per second.
    dpi : int
        Resolution (dots per inch).
    skip_frames : int
        Only render every N-th time step (useful for large runs).
    run_path : str or None
        Explicit run workspace path containing ``gwf.hds`` and ``gwt.ucn``.
        Recommended when *dataset_path* contains multiple runs.
    dynamic_scales : bool
        If True, the colourmap range is recomputed per frame.
    lx, lz : float or None
        Domain extents [m].  When None (default), read from ``manifest.json``.
    """
    ds = pl.Path(dataset_path)
    ws, run_dirs = _resolve_run_workspace(ds, run_path)

    print(f"Dataset path:          {ds}")
    print(f"Selected run workspace: {ws}")
    if len(run_dirs) > 1 and run_path is not None:
        print(f"Available run workspaces: {len(run_dirs)}")

    # Resolve output path
    if output_video is None:
        output_path = _default_output_path(ds, ws)
    else:
        out = pl.Path(output_video)
        output_path = out if out.is_absolute() else ds / out
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read domain dimensions from manifest when not explicitly overridden
    if lx is None or lz is None:
        grid_info = _load_manifest_grid(ds)
        lx = lx if lx is not None else grid_info["lx"]
        lz = lz if lz is not None else grid_info["lz"]
    print(f"Domain: Lx={lx} m, Lz={lz} m")

    # -----------------------------------------------------------------------
    # Load binary outputs
    # -----------------------------------------------------------------------
    print(f"Reading data from: {ws}")
    print("Loading head data...")
    hobj = flopy.utils.HeadFile(ws / "gwf.hds")
    head_data = hobj.get_alldata().squeeze()   # (ntimes, nlay, ncol)
    times = np.asarray(hobj.get_times())

    print("Loading concentration data...")
    cobj = flopy.utils.HeadFile(ws / "gwt.ucn", text="CONCENTRATION")
    conc_data = cobj.get_alldata().squeeze()   # (ntimes, nlay, ncol)

    ntimes, nlay, ncol = head_data.shape
    print(f"Data shape: {ntimes} time steps, {nlay} layers, {ncol} columns")

    # -----------------------------------------------------------------------
    # Spatial grid for plotting
    # -----------------------------------------------------------------------
    x = np.linspace(0, lx, ncol)
    z = np.linspace(lz, 0, nlay)   # top → bottom
    X, Z = np.meshgrid(x, z)

    frame_indices = list(range(0, ntimes, skip_frames))
    nframes = len(frame_indices)
    print(f"Creating animation: {nframes} frames (skip_frames={skip_frames})")

    # -----------------------------------------------------------------------
    # Figure setup
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Simplified Henry Problem: Density-Driven Convection\n"
        "(zero storage · zero influx · homogeneous Dirichlet BCs)",
        fontsize=13, fontweight="bold",
    )

    # Global colour limits (filter MF6 dummy values ≥ 1e20)
    head_valid = head_data[head_data < 1e20]
    head_vmin, head_vmax = float(head_valid.min()), float(head_valid.max())
    head_levels = _fixed_levels(head_vmin, head_vmax)

    conc_valid = conc_data[conc_data < 1e20]
    conc_vmin, conc_vmax = float(conc_valid.min()), float(conc_valid.max())
    conc_levels = _fixed_levels(conc_vmin, conc_vmax)

    ax_head, ax_conc = axes

    # --- Initial frame ---
    head_plot0 = np.where(head_data[0] < 1e20, head_data[0], np.nan)
    conc_plot0 = np.where(conc_data[0] < 1e20, conc_data[0], np.nan)

    im_head = ax_head.contourf(X, Z, head_plot0, levels=head_levels, cmap="Blues")
    ax_head.set_xlabel("x  (m)", fontsize=11)
    ax_head.set_ylabel("z  (m)", fontsize=11)
    ax_head.set_title("Hydraulic Head (m)", fontsize=12)
    ax_head.set_aspect("equal")
    cbar_head = fig.colorbar(im_head, ax=ax_head, label="Head (m)")

    im_conc = ax_conc.contourf(X, Z, conc_plot0, levels=conc_levels, cmap="Reds")
    ax_conc.set_xlabel("x  (m)", fontsize=11)
    ax_conc.set_ylabel("z  (m)", fontsize=11)
    ax_conc.set_title("Concentration (kg/m³)", fontsize=12)
    ax_conc.set_aspect("equal")
    cbar_conc = fig.colorbar(im_conc, ax=ax_conc, label="Concentration (kg/m³)")

    time_text = fig.text(
        0.5, 0.02, "", ha="center", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])

    # -----------------------------------------------------------------------
    # Frame update
    # -----------------------------------------------------------------------
    def update_frame(frame_num):
        nonlocal cbar_head, cbar_conc
        idx = frame_indices[frame_num]

        for coll in ax_head.collections:
            coll.remove()
        for coll in ax_conc.collections:
            coll.remove()

        head_plot = np.where(head_data[idx] < 1e20, head_data[idx], np.nan)
        conc_plot = np.where(conc_data[idx] < 1e20, conc_data[idx], np.nan)

        if dynamic_scales:
            hv = head_plot[~np.isnan(head_plot)]
            cv = conc_plot[~np.isnan(conc_plot)]
            h_levs = _fixed_levels(hv.min(), hv.max()) if len(hv) else head_levels
            c_levs = _fixed_levels(cv.min(), cv.max()) if len(cv) else conc_levels
            im_h = ax_head.contourf(X, Z, head_plot, levels=h_levs, cmap="Blues")
            im_c = ax_conc.contourf(X, Z, conc_plot, levels=c_levs, cmap="Reds")
            cbar_head.ax.clear()
            cbar_conc.ax.clear()
            cbar_head = fig.colorbar(im_h, cax=cbar_head.ax, label="Head (m)")
            cbar_conc = fig.colorbar(im_c, cax=cbar_conc.ax, label="Concentration (kg/m³)")
        else:
            ax_head.contourf(X, Z, head_plot, levels=head_levels, cmap="Blues")
            ax_conc.contourf(X, Z, conc_plot, levels=conc_levels, cmap="Reds")

        time_text.set_text(f"Time: {times[idx]:.4f} days  (step {idx + 1}/{ntimes})")

        if frame_num % 10 == 0:
            print(f"  Rendering frame {frame_num + 1}/{nframes} (t={times[idx]:.3f} d)")

        return ax_head.collections + ax_conc.collections + [time_text]

    # -----------------------------------------------------------------------
    # Render and save
    # -----------------------------------------------------------------------
    print("\nCreating animation...")
    anim = animation.FuncAnimation(
        fig, update_frame, frames=nframes,
        interval=1000 / fps, blit=False, repeat=True,
    )

    print(f"Saving animation to: {output_path}")
    print(f"  Resolution: {dpi} DPI  |  {fps} FPS  |  ~{nframes / fps:.1f} s")

    available_writers = animation.writers.list()
    print(f"  Available writers: {available_writers}")

    writer = None
    if "ffmpeg" in available_writers:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata=dict(artist="FloPy"), bitrate=3000)
        print("  Using ffmpeg writer")
    elif "pillow" in available_writers:
        output_path = output_path.with_suffix(".gif")
        Writer = animation.writers["pillow"]
        writer = Writer(fps=fps)
        print("  Using pillow writer (saving as GIF)")
    elif "imagemagick" in available_writers:
        Writer = animation.writers["imagemagick"]
        writer = Writer(fps=fps)
        print("  Using imagemagick writer")
    else:
        raise RuntimeError(
            f"No suitable video writer found. Available: {available_writers}\n"
            "Install ffmpeg: conda install -c conda-forge ffmpeg"
        )

    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"\n✓ Animation saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    return str(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate simplified Henry problem head and concentration fields."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Run workspace or dataset root (containing scenario_*/run_* subdirs).",
    )
    parser.add_argument(
        "--run-path",
        type=str,
        default=None,
        help=(
            "Explicit run workspace path containing gwf.hds and gwt.ucn. "
            "Required when --dataset-path contains multiple runs."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output video path.  If relative, resolved under --dataset-path. "
            "If omitted, auto-generated in the dataset root directory."
        ),
    )
    parser.add_argument("--fps", type=int, default=20, help="Frames per second. Default: 20.")
    parser.add_argument("--dpi", type=int, default=150, help="Resolution in DPI. Default: 150.")
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Render every N-th frame (default: 1, no skipping).",
    )
    parser.add_argument(
        "--dynamic-scales",
        action="store_true",
        help="Recompute colormap range per frame instead of using global limits.",
    )
    parser.add_argument(
        "--lx",
        type=float,
        default=None,
        help="Domain horizontal extent [m]. Overrides manifest.json value.",
    )
    parser.add_argument(
        "--lz",
        type=float,
        default=None,
        help="Domain vertical extent [m]. Overrides manifest.json value.",
    )

    args = parser.parse_args()
    animate_simple_henry(
        dataset_path=args.dataset_path,
        output_video=args.output,
        fps=args.fps,
        dpi=args.dpi,
        skip_frames=args.skip,
        run_path=args.run_path,
        dynamic_scales=args.dynamic_scales,
        lx=args.lx,
        lz=args.lz,
    )


if __name__ == "__main__":
    main()
