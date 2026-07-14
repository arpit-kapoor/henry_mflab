#!/usr/bin/env python
"""
Animate Henry problem results: head and concentration fields over time.
Creates a video showing the evolution of hydraulic head and salt concentration.
"""
import argparse
import pathlib as pl
import re

import flopy.utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


DEFAULT_DATASET_ROOT = pl.Path(
    "/Users/akap5486/Projects/groundwater/data/henry_data/one_coupling_scenario"
)


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _find_run_dirs(path: pl.Path):
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


def _resolve_run_workspace(dataset_path: pl.Path, run_path: str | None):
    if run_path is not None:
        run_workspace = pl.Path(run_path)
        if not ((run_workspace / "gwf.hds").exists() and (run_workspace / "gwt.ucn").exists()):
            raise FileNotFoundError(
                "run_path must point to a run workspace containing gwf.hds and gwt.ucn: "
                f"{run_workspace}"
            )
        return run_workspace, _find_run_dirs(dataset_path)

    run_dirs = _find_run_dirs(dataset_path)
    if not run_dirs:
        raise FileNotFoundError(
            "No run workspace found with gwf.hds and gwt.ucn under "
            f"{dataset_path}"
        )
    if len(run_dirs) > 1:
        raise ValueError(
            "Multiple run workspaces found. Pass --run-path to select one explicitly. "
            f"Found {len(run_dirs)} runs under {dataset_path}"
        )
    return run_dirs[0], run_dirs


def _default_output_path(dataset_path: pl.Path, run_workspace: pl.Path):
    # Save output in the dataset root path, not inside the run workspace.
    if dataset_path.name.startswith("run_"):
        base_dir = dataset_path
    elif dataset_path.name.startswith("scenario_"):
        base_dir = dataset_path.parent
    else:
        base_dir = dataset_path

    scenario_name = run_workspace.parent.name
    run_name = run_workspace.name
    out_name = (
        f"henry_animation_{_slugify(scenario_name)}_{_slugify(run_name)}.mp4"
    )
    return base_dir / out_name


def _fixed_levels(vmin: float, vmax: float, n_levels: int = 21):
    """Build stable contour levels for the full animation."""
    if np.isclose(vmin, vmax):
        eps = max(1e-6, abs(vmin) * 1e-6)
        return np.linspace(vmin - eps, vmax + eps, n_levels)
    return np.linspace(vmin, vmax, n_levels)


def animate_henry(
    dataset_path,
    output_video=None,
    fps=20,
    dpi=150,
    skip_frames=1,
    run_path=None,
    dynamic_scales=False,
):
    """
    Create animation of Henry problem results.
    
    Parameters
    ----------
    dataset_path : str or Path
        Run workspace or dataset root containing scenario/run workspaces
    output_video : str or None
        Output video filename/path. If None, auto-generated in dataset root path.
    fps : int
        Frames per second in video
    dpi : int
        Resolution (dots per inch)
    skip_frames : int
        Only plot every Nth time step (for faster rendering)
    run_path : str or None
        Explicit path to the run workspace containing gwf.hds and gwt.ucn
    dynamic_scales : bool
        If True, scale the colormap independently for each frame
    """
    ds = pl.Path(dataset_path)
    ws, run_dirs = _resolve_run_workspace(ds, run_path)

    print(f"Dataset path: {ds}")
    print(f"Selected run workspace: {ws}")
    if len(run_dirs) > 1 and run_path is not None:
        print(f"Available run workspaces under dataset path: {len(run_dirs)}")

    if output_video is None:
        output_path = _default_output_path(ds, ws)
    else:
        out = pl.Path(output_video)
        output_path = out if out.is_absolute() else ds / out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading data from: {ws}")
    
    # Read head and concentration files
    print("Loading head data...")
    hobj = flopy.utils.HeadFile(ws / "gwf.hds")
    head_data = hobj.get_alldata()  # Shape: (ntimes, nlay, nrow, ncol)
    times = hobj.get_times()
    
    print("Loading concentration data...")
    cobj = flopy.utils.HeadFile(ws / "gwt.ucn", text="CONCENTRATION")
    conc_data = cobj.get_alldata()  # Shape: (ntimes, nlay, nrow, ncol)
    
    # Squeeze out the row dimension (nrow=1 for 2D x-z slice)
    head_data = head_data.squeeze()  # Shape: (ntimes, nlay, ncol)
    conc_data = conc_data.squeeze()  # Shape: (ntimes, nlay, ncol)
    
    ntimes, nlay, ncol = head_data.shape
    print(f"Data shape: {ntimes} time steps, {nlay} layers, {ncol} columns")
    
    # Create spatial coordinates (assuming domain from run_henry.py defaults)
    Lx = 8.0  # horizontal extent (m)
    Lz = 4.0  # vertical extent (m)
    x = np.linspace(0, Lx, ncol)
    z = np.linspace(Lz, 0, nlay)
    X, Z = np.meshgrid(x, z)
    
    # Determine frames to plot
    frame_indices = range(0, ntimes, skip_frames)
    nframes = len(frame_indices)
    print(f"Creating animation with {nframes} frames (skip_frames={skip_frames})")
    
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Henry Problem: Saltwater Intrusion", fontsize=14, fontweight='bold')
    
    # Pre-compute color limits for consistent scaling, ignoring dummy values (e.g. 1e30)
    head_valid = head_data[head_data < 1e20]
    head_vmin, head_vmax = head_valid.min(), head_valid.max()
    head_levels = _fixed_levels(head_vmin, head_vmax)
    
    conc_valid = conc_data[conc_data < 1e20]
    conc_vmin, conc_vmax = conc_valid.min(), conc_valid.max()
    conc_levels = _fixed_levels(conc_vmin, conc_vmax)
    
    # Initialize plots
    ax_head, ax_conc = axes
    
    # Head subplot - masking dummy values
    head_plot0 = np.where(head_data[0] < 1e20, head_data[0], np.nan)
    im_head = ax_head.contourf(X, Z, head_plot0, levels=head_levels, cmap='Blues')
    ax_head.set_xlabel('Distance (m)', fontsize=11)
    ax_head.set_ylabel('Elevation (m)', fontsize=11)
    ax_head.set_title('Hydraulic Head (m)', fontsize=12)
    ax_head.set_aspect('equal')
    cbar_head = fig.colorbar(im_head, ax=ax_head, label='Head (m)')
    
    # Concentration subplot - masking dummy values
    conc_plot0 = np.where(conc_data[0] < 1e20, conc_data[0], np.nan)
    im_conc = ax_conc.contourf(X, Z, conc_plot0, levels=conc_levels, cmap='Reds')
    ax_conc.set_xlabel('Distance (m)', fontsize=11)
    ax_conc.set_ylabel('Elevation (m)', fontsize=11)
    ax_conc.set_title('Concentration (kg/m³)', fontsize=12)
    ax_conc.set_aspect('equal')
    cbar_conc = fig.colorbar(im_conc, ax=ax_conc, label='Concentration (kg/m³)')
    
    # Time annotation
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=11, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    def update_frame(frame_num):
        nonlocal cbar_head, cbar_conc
        """Update function for animation."""
        idx = frame_indices[frame_num]
        
        # Clear previous contours
        for coll in ax_head.collections:
            coll.remove()
        for coll in ax_conc.collections:
            coll.remove()
        
        # Plot new data - ignoring dummy values
        head_plot = np.where(head_data[idx] < 1e20, head_data[idx], np.nan)
        conc_plot = np.where(conc_data[idx] < 1e20, conc_data[idx], np.nan)
        
        if dynamic_scales:
            head_valid_f = head_plot[~np.isnan(head_plot)]
            conc_valid_f = conc_plot[~np.isnan(conc_plot)]
            
            h_levs = _fixed_levels(head_valid_f.min(), head_valid_f.max()) if len(head_valid_f) else head_levels
            c_levs = _fixed_levels(conc_valid_f.min(), conc_valid_f.max()) if len(conc_valid_f) else conc_levels
            
            im_h = ax_head.contourf(X, Z, head_plot, levels=h_levs, cmap='Blues')
            im_c = ax_conc.contourf(X, Z, conc_plot, levels=c_levs, cmap='Reds')
            
            cbar_head.ax.clear()
            cbar_conc.ax.clear()
            cbar_head = fig.colorbar(im_h, cax=cbar_head.ax, label='Head (m)')
            cbar_conc = fig.colorbar(im_c, cax=cbar_conc.ax, label='Concentration (kg/m³)')
        else:
            ax_head.contourf(X, Z, head_plot, levels=head_levels, cmap='Blues')
            ax_conc.contourf(X, Z, conc_plot, levels=conc_levels, cmap='Reds')
        
        # Update time text
        time_text.set_text(f'Time: {times[idx]:.4f} days (Step {idx+1}/{ntimes})')
        
        if frame_num % 10 == 0:
            print(f"  Rendering frame {frame_num+1}/{nframes} (time step {idx+1})")
        
        return ax_head.collections + ax_conc.collections + [time_text]
    
    # Create animation
    print("\nCreating animation...")
    anim = animation.FuncAnimation(
        fig, update_frame, frames=nframes,
        interval=1000/fps, blit=False, repeat=True
    )
    
    # Save animation - try different writers in order of preference
    print(f"Saving animation to: {output_path}")
    print(f"  Resolution: {dpi} DPI")
    print(f"  Frame rate: {fps} FPS")
    print(f"  Duration: {nframes/fps:.2f} seconds")
    
    # Try to find available writer
    available_writers = animation.writers.list()
    print(f"\nAvailable writers: {available_writers}")
    
    writer = None
    if 'ffmpeg' in available_writers:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='FloPy'), bitrate=3000)
        print("Using ffmpeg writer")
    elif 'pillow' in available_writers:
        # Save as GIF if ffmpeg not available
        output_path = output_path.with_suffix('.gif')
        Writer = animation.writers['pillow']
        writer = Writer(fps=fps)
        print("Using pillow writer (saving as GIF)")
    elif 'imagemagick' in available_writers:
        Writer = animation.writers['imagemagick']
        writer = Writer(fps=fps)
        print("Using imagemagick writer")
    else:
        raise RuntimeError(
            f"No suitable video writer found. Available: {available_writers}\n"
            "Install ffmpeg with: conda install -c conda-forge ffmpeg"
        )
    
    anim.save(str(output_path), writer=writer, dpi=dpi)
    
    print(f"\n✓ Animation saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    plt.close(fig)
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Animate Henry problem head and concentration fields"
    )
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        default=str(DEFAULT_DATASET_ROOT),
        help=(
            "Run workspace path or dataset root path containing scenario/run folders "
            f"(default: {DEFAULT_DATASET_ROOT})"
        )
    )
    parser.add_argument(
        "--run-path",
        type=str,
        default=None,
        help=(
            "Explicit run workspace path containing gwf.hds and gwt.ucn. "
            "Recommended when --dataset-path contains multiple runs."
        )
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help=(
            "Output video path. If relative, resolved under --dataset-path. "
            "If omitted, auto-generated in the dataset root path."
        )
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=20,
        help="Frames per second (default: 20)"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=150,
        help="Resolution in DPI (default: 150)"
    )
    parser.add_argument(
        "--skip", 
        type=int, 
        default=1,
        help="Skip every N frames for faster rendering (default: 1, no skipping)"
    )
    parser.add_argument(
        "--dynamic-scales",
        action="store_true",
        help="Scale colormaps dynamically per frame instead of globally"
    )
    
    args = parser.parse_args()
    
    animate_henry(
        dataset_path=args.dataset_path,
        output_video=args.output,
        fps=args.fps,
        dpi=args.dpi,
        skip_frames=args.skip,
        run_path=args.run_path,
        dynamic_scales=args.dynamic_scales,
    )


if __name__ == "__main__":
    main()

