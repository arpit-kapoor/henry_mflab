"""Command-line interface for the simplified Henry 3-D dataset generator."""
import argparse

from .generators import generate_simplified_henry_dataset
from .simulation import HEAD_BCS


def _parse_float_csv(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate 3-D (space × time) datasets for the simplified Henry problem "
            "(zero storage, zero influx, C = 0 on all boundaries). Each run maps "
            "input (3, T-1, nlay, ncol) = [C0, beta_c, diffc] to "
            "target (2, T-1, nlay, ncol) = [concentration, head]. "
            "All runs within a scenario are batched into scenario.npz."
        )
    )

    # Output
    ap.add_argument("--outdir", type=str, default="./data/simplified_henry",
                    help="Root output directory. Default: ./data/simplified_henry")

    # Grid
    ap.add_argument("--ncol", type=int, default=40, help="Number of columns. Default: 40.")
    ap.add_argument("--nlay", type=int, default=20, help="Number of layers. Default: 20.")
    ap.add_argument("--lx", type=float, default=2.0,
                    help="Domain horizontal extent [m]. Default: 2.0.")
    ap.add_argument("--lz", type=float, default=1.0,
                    help="Domain vertical extent [m]. Default: 1.0.")

    # Time
    ap.add_argument("--total-time", type=float, default=1.0,
                    help="Simulation duration [days]. Default: 1.0.")
    ap.add_argument("--nstp", type=int, default=50,
                    help="Number of uniform MODFLOW time steps. Default: 50.")
    ap.add_argument("--skip", type=int, default=2,
                    help="Temporal stride applied as time_series[::skip]. Default: 2.")

    # Initial concentration (GRF)
    ap.add_argument("--random-field-len-scale", type=str, default="0.1, 0.3, 0.5, 0.7, 0.9",
                    help="Comma-separated GRF length scales (normalised coordinates). "
                         "Default: 0.1, 0.3, 0.5, 0.7, 0.9.")
    ap.add_argument("--random-field-var", type=str, default="0.1, 0.3, 0.5, 0.7",
                    help="Comma-separated GRF variances. Default: 0.1, 0.3, 0.5, 0.7.")
    ap.add_argument("--random-field-count", type=int, default=20,
                    help="GRF samples per (len_scale, var) pair. Default: 20.")

    # Scenario sweep
    ap.add_argument("--beta-c-values", type=str, default="0.00002, 0.00007, 0.00025, 0.001",
                    help="Comma-separated β_C values [m³/kg]. "
                         "Default: 0.00002, 0.00007, 0.00025, 0.001.")
    ap.add_argument("--diffc-values", type=str, default="0.001, 0.003, 0.01, 0.03",
                    help="Comma-separated diffusion coefficients D_C [m²/d]. "
                         "Default: 0.001, 0.003, 0.01, 0.03.")

    # Physics
    ap.add_argument("--hk", type=float, default=50.0,
                    help="Hydraulic conductivity [m/d]. Default: 50.0.")
    ap.add_argument("--por", type=float, default=0.35, help="Porosity. Default: 0.35.")
    ap.add_argument("--al", type=float, default=0.0,
                    help="Longitudinal dispersivity [m]. Default: 0.0.")
    ap.add_argument("--at", type=float, default=0.0,
                    help="Transverse dispersivity [m]. Default: 0.0.")
    ap.add_argument("--rho0", type=float, default=1000.0,
                    help="Reference fluid density ρ₀ [kg/m³]. Default: 1000.0.")
    ap.add_argument("--c0-max", type=float, default=35.0,
                    help="Maximum of the min-max scaled initial field [kg/m³]. Default: 35.0.")
    ap.add_argument("--head-bc", type=str, choices=HEAD_BCS, default="lr",
                    help="'lr': h = 0 on the left/right walls, no-flow top/bottom (the "
                         "paper's dataset). 'all': h = 0 on all four boundaries. Default: lr.")

    # Dataset controls
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing scenario.npz files.")
    ap.add_argument("--max-runs-per-scenario", type=int, default=None,
                    help="Cap on runs per scenario (for quick tests).")
    ap.add_argument("--save-timeseries", action="store_true",
                    help="Include full downsampled head/conc time-series in scenario.npz.")
    ap.add_argument("--save-modflow-files", action="store_true",
                    help="Keep all MODFLOW 6 workspace files (default: prune).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed for the C0 fields; scenario i uses [seed, i]. Default: 42.")
    ap.add_argument("--n-workers", type=int, default=None,
                    help="Parallel worker processes (one scenario each). Defaults to "
                         "min(n_scenarios, os.cpu_count()); 1 runs sequentially.")

    # Executable
    ap.add_argument("--mf6-exe", type=str, default="mf6",
                    help="MODFLOW 6 executable name or path. Default: mf6.")

    return ap


def run(args: argparse.Namespace):
    init_field_args = {
        "random_field_len_scale": _parse_float_csv(args.random_field_len_scale),
        "random_field_var":       _parse_float_csv(args.random_field_var),
        "random_field_count":     int(args.random_field_count),
    }

    generate_simplified_henry_dataset(
        outdir=args.outdir,
        beta_c_values=_parse_float_csv(args.beta_c_values),
        diffc_values=_parse_float_csv(args.diffc_values),
        init_field_args=init_field_args,
        ncol=args.ncol,
        nlay=args.nlay,
        lx=args.lx,
        lz=args.lz,
        total_time=args.total_time,
        nstp=args.nstp,
        hk=args.hk,
        por=args.por,
        al=args.al,
        at=args.at,
        rho0=args.rho0,
        c_sea=args.c0_max,
        head_bc=args.head_bc,
        skip=args.skip,
        overwrite=args.overwrite,
        max_runs_per_scenario=args.max_runs_per_scenario,
        save_timeseries=args.save_timeseries,
        save_modflow_files=args.save_modflow_files,
        seed=args.seed,
        exe_name=args.mf6_exe,
        n_workers=args.n_workers,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)
