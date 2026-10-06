"""Command-line interface for the forced Henry 3-D dataset generator."""
import argparse
import pathlib as pl

import numpy as np

from .generators import INFLOW_ENCODINGS, INIT_METHODS, generate_henry_forced_dataset


def _parse_float_csv(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate 3-D (space × time) datasets for the forced Henry problem "
            "(classic Henry BCs: constant freshwater inflow on the left, sea-level "
            "GHB with seawater on the right, zero storage). Each run maps "
            "input (4, T-1, nlay, ncol) = [C0, inflow, beta_c, diffc] to "
            "target (2, T-1, nlay, ncol) = [concentration, head]. "
            "All runs within a scenario are batched into scenario.npz."
        )
    )

    # Output
    ap.add_argument("--outdir", type=str, default="./henry_forced_out",
                    help="Root output directory. Default: ./henry_forced_out")

    # Grid
    ap.add_argument("--ncol", type=int, default=40, help="Number of columns. Default: 40.")
    ap.add_argument("--nlay", type=int, default=20, help="Number of layers. Default: 20.")
    ap.add_argument("--lx", type=float, default=2.0,
                    help="Domain horizontal extent [m]. Default: 2.0.")
    ap.add_argument("--lz", type=float, default=1.0,
                    help="Domain vertical extent [m]. Default: 1.0.")

    # Time
    ap.add_argument("--total-time", type=float, default=0.5,
                    help="Simulation duration [days]. Default: 0.5.")
    ap.add_argument("--nstp", type=int, default=500,
                    help="Number of uniform MODFLOW time steps. Default: 500.")
    ap.add_argument("--skip", type=int, default=1,
                    help="Temporal stride applied as time_series[::skip]. Default: 1.")

    # Initial concentration
    ap.add_argument("--init-method", type=str, choices=INIT_METHODS, default="random",
                    help="'random' (random field) or 'seawater' (uniform c_sea, classic "
                         "Henry IC). Default: random.")
    ap.add_argument("--random-field-type", type=str, default="grf",
                    help="Comma-separated random field types (see "
                         "simple_henry.init_functions.sample_field_2d). Default: grf.")
    ap.add_argument("--random-field-smoothness", type=str, default="1.0",
                    help="Comma-separated random field smoothness values. Default: 1.0.")
    ap.add_argument("--random-field-len-scale", type=str, default="0.1",
                    help="Comma-separated random field length scales. Default: 0.1.")
    ap.add_argument("--random-field-var", type=str, default="0.1",
                    help="Comma-separated random field variances. Default: 0.1.")
    ap.add_argument("--random-field-count", type=int, default=1,
                    help="Samples per random-field parameter combination "
                         "(number of runs for 'seawater'). Default: 1.")
    ap.add_argument("--random-field-taper", action=argparse.BooleanOptionalAction, default=False,
                    help="Taper the 'grf' field to zero on every edge (simple_henry's "
                         "zero-Dirichlet ICs). Off by default: the forced Henry BCs do not "
                         "fix concentration.")
    ap.add_argument("--random-field-period", type=float, default=2.0,
                    help="Periodicity of the gstools random field relative to the domain. "
                         "1.0 makes opposite edges match; 2.0 (default) removes that correlation.")

    # Inflow forcing
    ap.add_argument("--inflow-min", type=float, default=2.0,
                    help="Lower bound of the uniformly sampled total inflow Q [m³/d]. "
                         "Default: 2.0.")
    ap.add_argument("--inflow-max", type=float, default=6.0,
                    help="Upper bound of the uniformly sampled total inflow Q [m³/d]. "
                         "Default: 6.0.")
    ap.add_argument("--inflow-encoding", type=str, choices=INFLOW_ENCODINGS, default="broadcast",
                    help="How Q enters the input tensor: 'broadcast' over the domain like "
                         "beta_c/diffc, or 'left_column' (inflow column only). Default: broadcast.")
    ap.add_argument("--coord-channels", action=argparse.BooleanOptionalAction, default=True,
                    help="Append (t, z, x) coordinate channels to the input so the model knows "
                         "where the boundaries are. Default: on (--no-coord-channels drops them).")

    # Scenario sweep
    ap.add_argument("--beta-c-values", type=str, default="0.0007",
                    help="Comma-separated β_C values [m³/kg]. Default: 0.0007.")
    ap.add_argument("--diffc-values", type=str, default="0.57024",
                    help="Comma-separated diffusion coefficients [m²/d]. Default: 0.57024.")

    # Physics
    ap.add_argument("--hk", type=float, default=864.0,
                    help="Hydraulic conductivity [m/d]. Default: 864.0.")
    ap.add_argument("--por", type=float, default=0.35, help="Porosity. Default: 0.35.")
    ap.add_argument("--al", type=float, default=0.0,
                    help="Longitudinal dispersivity [m]. Default: 0.0.")
    ap.add_argument("--at", type=float, default=0.0,
                    help="Transverse dispersivity [m]. Default: 0.0.")
    ap.add_argument("--rho0", type=float, default=1000.0,
                    help="Reference fluid density ρ₀ [kg/m³]. Default: 1000.0.")
    ap.add_argument("--c-sea", type=float, default=35.0,
                    help="Seawater concentration [kg/m³]. Default: 35.0.")

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
                    help="Base seed for the run sets (C0 fields, inflow); scenario i uses "
                         "[seed, i]. Default: 42.")
    ap.add_argument("--shared-runs", action="store_true",
                    help="Reuse one run set (same C0 and Q per run index) in every scenario "
                         "(paired design). Default: independent run set per scenario.")
    ap.add_argument("--n-workers", type=int, default=None,
                    help="Parallel worker processes (one scenario each). Defaults to "
                         "min(n_scenarios, os.cpu_count()); 1 runs sequentially.")

    # Executable
    ap.add_argument("--mf6-exe", type=str, default="mf6",
                    help="MODFLOW 6 executable name or path. Default: mf6.")

    # Optional spatially varying K
    ap.add_argument("--kappa-file", type=str, default=None,
                    help="Path to an .npz file with 'hk' (and optionally 'vk') arrays "
                         "of shape (nlay, ncol). Overrides --hk.")

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

    init_field_args = {
        "random_field_type":       [s.strip() for s in args.random_field_type.split(",") if s.strip()],
        "random_field_smoothness": _parse_float_csv(args.random_field_smoothness),
        "random_field_len_scale":  _parse_float_csv(args.random_field_len_scale),
        "random_field_var":        _parse_float_csv(args.random_field_var),
        "random_field_count":      int(args.random_field_count),
        "random_field_taper":      bool(args.random_field_taper),
        "random_field_period":     float(args.random_field_period),
    }

    generate_henry_forced_dataset(
        outdir=outdir,
        beta_c_values=_parse_float_csv(args.beta_c_values),
        diffc_values=_parse_float_csv(args.diffc_values),
        inflow_min=args.inflow_min,
        inflow_max=args.inflow_max,
        inflow_encoding=args.inflow_encoding,
        coord_channels=args.coord_channels,
        init_method=args.init_method,
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
        c_sea=args.c_sea,
        hk_field=hk_field,
        vk_field=vk_field,
        skip=args.skip,
        overwrite=args.overwrite,
        max_runs_per_scenario=args.max_runs_per_scenario,
        save_timeseries=args.save_timeseries,
        save_modflow_files=args.save_modflow_files,
        seed=args.seed,
        shared_runs=args.shared_runs,
        exe_name=args.mf6_exe,
        n_workers=args.n_workers,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)
