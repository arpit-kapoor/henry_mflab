#!/usr/bin/env python
"""
Animate Henry problem results: head and concentration fields over time.
Creates a video showing the evolution of hydraulic head and salt concentration.
"""
import argparse
import pathlib as pl

import flopy.utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def animate_henry(
    workspace,
    output_video="henry_animation.mp4",
    fps=20,
    dpi=150,
    skip_frames=1,
):
    """
    Create animation of Henry problem results.
    
    Parameters
    ----------
    workspace : str or Path
        Directory containing MODFLOW output files
    output_video : str
        Output video filename
    fps : int
        Frames per second in video
    dpi : int
        Resolution (dots per inch)
    skip_frames : int
        Only plot every Nth time step (for faster rendering)
    """
    ws = pl.Path(workspace)
    
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
    Lx = 2.0  # horizontal extent (m)
    Lz = 1.0  # vertical extent (m)
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
    
    conc_valid = conc_data[conc_data < 1e20]
    conc_vmin, conc_vmax = conc_valid.min(), conc_valid.max()
    
    # Initialize plots
    ax_head, ax_conc = axes
    
    # Head subplot - masking dummy values
    head_plot0 = np.where(head_data[0] < 1e20, head_data[0], np.nan)
    im_head = ax_head.contourf(X, Z, head_plot0, levels=20, cmap='Blues', 
                               vmin=head_vmin, vmax=head_vmax)
    ax_head.set_xlabel('Distance (m)', fontsize=11)
    ax_head.set_ylabel('Elevation (m)', fontsize=11)
    ax_head.set_title('Hydraulic Head (m)', fontsize=12)
    ax_head.set_aspect('equal')
    cbar_head = fig.colorbar(im_head, ax=ax_head, label='Head (m)')
    
    # Concentration subplot - masking dummy values
    conc_plot0 = np.where(conc_data[0] < 1e20, conc_data[0], np.nan)
    im_conc = ax_conc.contourf(X, Z, conc_plot0, levels=20, cmap='Reds',
                               vmin=conc_vmin, vmax=conc_vmax)
    ax_conc.set_xlabel('Distance (m)', fontsize=11)
    ax_conc.set_ylabel('Elevation (m)', fontsize=11)
    ax_conc.set_title('Concentration (g/L)', fontsize=12)
    ax_conc.set_aspect('equal')
    cbar_conc = fig.colorbar(im_conc, ax=ax_conc, label='Concentration (g/L)')
    
    # Time annotation
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=11, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    def update_frame(frame_num):
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
        
        ax_head.contourf(X, Z, head_plot, levels=20, cmap='Blues',
                        vmin=head_vmin, vmax=head_vmax)
        ax_conc.contourf(X, Z, conc_plot, levels=20, cmap='Reds',
                        vmin=conc_vmin, vmax=conc_vmax)
        
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
    output_path = ws / output_video
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
        "--workspace", 
        type=str, 
        default="./out",
        help="Directory containing MODFLOW output files (default: ./out)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="henry_animation.mp4",
        help="Output video filename (default: henry_animation.mp4)"
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
    
    args = parser.parse_args()
    
    animate_henry(
        workspace=args.workspace,
        output_video=args.output,
        fps=args.fps,
        dpi=args.dpi,
        skip_frames=args.skip,
    )


if __name__ == "__main__":
    main()

