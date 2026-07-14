"""Command-line interface for the simplified Henry dataset generator."""
import argparse
import pathlib as pl

import numpy as np

from .generators import generate_simple_henry_dataset


def _parse_float_csv(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate windowed Henry datasets for the simplified density-driven "
            "convection problem (zero storage, zero influx, homogeneous Dirichlet BCs)."
        )
    )

    # Output
    ap.add_argument("--outdir", type=str, default="./simple_henry_out",
                    help="Root output directory. Default: ./simple_henry_out")

    # Grid
    ap.add_argument("--ncol", type=int, default=80,
                    help="Number of columns. Default: 80.")
    ap.add_argument("--nlay", type=int, default=40,
                    help="Number of layers. Default: 40.")
    ap.add_argument("--lx", type=float, default=2.0,
                    help="Domain horizontal extent [m]. Default: 2.0.")
    ap.add_argument("--lz", type=float, default=1.0,
                    help="Domain vertical extent [m]. Default: 1.0.")

    # Time
    ap.add_argument("--total-time", type=float, default=30.0,
                    help="Simulation duration [days]. Default: 30.")
    ap.add_argument("--nstp", type=int, default=240,
                    help="Number of uniform time steps. Default: 240.")

    # Initial concentration
    ap.add_argument("--c0", type=float, default=35.0,
                    help=(
                        "Uniform initial concentration C₀ [kg/m³]. "
                        "Broadcast to a spatially constant field over the whole domain. "
                        "Default: 35.0 (salt-saturated)."
                    ))

    # Parameter sweep
    ap.add_argument("--beta-c-values", type=str, default="0.7",
                    help="Comma-separated β_C values. Default: 0.7.")
    ap.add_argument("--diffc-values", type=str, default="0.57024",
                    help="Comma-separated diffusion coefficients [m²/d]. Default: 0.57024.")
    ap.add_argument("--hk-values", type=str, default="864.0",
                    help="Comma-separated hydraulic conductivity values [m/d]. Default: 864.0.")
    ap.add_argument("--por-values", type=str, default="0.35",
                    help="Comma-separated porosity values. Default: 0.35.")

    # Dispersion
    ap.add_argument("--al", type=float, default=0.0,
                    help="Longitudinal dispersivity [m]. Default: 0.0.")
    ap.add_argument("--at", type=float, default=0.0,
                    help="Transverse dispersivity [m]. Default: 0.0.")

    # Density
    ap.add_argument("--rho0", type=float, default=1000.0,
                    help="Reference fluid density ρ₀ [kg/m³]. Default: 1000.0.")

    # Dataset controls
    ap.add_argument("--lag", type=int, default=1,
                    help="Prediction lag [time steps]. Default: 1.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing windows.npz files.")
    ap.add_argument("--max-runs-per-scenario", type=int, default=None,
                    help="Cap on run combinations per scenario (for quick tests).")
    ap.add_argument("--save-timeseries", action="store_true",
                    help="Include full head/conc time-series in windows.npz.")
    ap.add_argument("--save-modflow-files", action="store_true",
                    help="Keep all MODFLOW 6 workspace files (default: prune to windows.npz).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for train/val/test split. Default: 42.")
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="Fraction of windows in the training split. Default: 0.7.")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraction of windows in the validation split. Default: 0.15.")

    # Executable
    ap.add_argument("--mf6-exe", type=str, default="mf6",
                    help="MODFLOW 6 executable name or path. Default: mf6.")

    # Optional spatially varying K
    ap.add_argument("--kappa-file", type=str, default=None,
                    help=(
                        "Path to an .npz file with 'hk' (and optionally 'vk') arrays "
                        "of shape (nlay, ncol). Overrides --hk-values."
                    ))

    return ap


def run(args: argparse.Namespace):
    outdir = pl.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hk_field = vk_field = None
    if args.kappa_file:
        data = np.load(args.kappa_file)
        if "hk" not in data:
            raise ValueError(f"kappa file {args.kappa_file} must contain 'hk'")
        hk_field = np.asarray(data["hk"], dtype=float)
        vk_field = np.asarray(data["vk"], dtype=float) if "vk" in data else hk_field.copy()

    generate_simple_henry_dataset(
        outdir=outdir,
        beta_c_values=_parse_float_csv(args.beta_c_values),
        diffc_values=_parse_float_csv(args.diffc_values),
        hk_values=_parse_float_csv(args.hk_values),
        por_values=_parse_float_csv(args.por_values),
        C0=args.c0,
        ncol=args.ncol,
        nlay=args.nlay,
        lx=args.lx,
        lz=args.lz,
        total_time=args.total_time,
        nstp=args.nstp,
        al=args.al,
        at=args.at,
        rho0=args.rho0,
        hk_field=hk_field,
        vk_field=vk_field,
        lag=args.lag,
        overwrite=args.overwrite,
        max_runs_per_scenario=args.max_runs_per_scenario,
        save_timeseries=args.save_timeseries,
        save_modflow_files=args.save_modflow_files,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        exe_name=args.mf6_exe,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)
