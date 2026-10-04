"""Command-line interface for the simplified Henry 3-D dataset generator."""
import argparse
import pathlib as pl

import numpy as np

from .generators import generate_simple_henry_dataset


def _parse_float_csv(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate 3-D (space × time) Henry datasets for the simplified "
            "density-driven convection problem (zero storage, zero influx, "
            "homogeneous Dirichlet BCs). Each run produces one data point: "
            "input (4, T_in, nlay, ncol) and target (2, T_out, nlay, ncol). "
            "All runs within a scenario are batched into scenario.npz."
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
                    help="Number of uniform MODFLOW time steps. Default: 240.")

    # Initial concentration
    ap.add_argument("--init-method", type=str, default="random",
                    help="Method for initializing the concentration field. "
                         "Options: 'wedge', 'random'. Default: 'random'.")
    ap.add_argument("--random-field-type", type=str, default="grf",
                    help="Type of random field. Options: 'fourier_1d', 'fourier', "
                         "'polynomial', 'grbf', 'white_noise', 'perlin', 'grf', "
                         "'matern', 'exp'. Default: 'grf'.")
    ap.add_argument("--random-field-smoothness", type=str, default="1.0",
                    help="Smoothness of the random field (comma-separated floats). "
                         "Default: 1.0.")
    ap.add_argument("--random-field-len-scale", type=str, default="0.1",
                    help="Length scale of the random field (comma-separated floats). "
                         "Default: 0.1.")
    ap.add_argument("--random-field-var", type=str, default="0.1",
                    help="Variance of the random field (comma-separated floats). "
                         "Default: 0.1.")
    ap.add_argument("--random-field-count", type=int, default=1,
                    help="Number of random fields per parameter combination. "
                         "Default: 1.")

    ap.add_argument("--c0-x-toe-values", type=str, default="0.5, 0.0, 0.5",
                    help="Comma-separated wedge toe positions [m]. "
                         "Default: 0.5, 0.0, 0.5.")
    ap.add_argument("--c0-x-top-values", type=str, default="0.5, 1.0, 1.5",
                    help="Comma-separated wedge top positions [m]. "
                         "Default: 0.5, 1.0, 1.5.")
    ap.add_argument("--c0-trans-width-values", type=str, default="0.001, 0.01, 0.1",
                    help="Comma-separated transition widths [m]. "
                         "Default: 0.001, 0.01, 0.1.")

    # Parameter sweep
    ap.add_argument("--beta-c-values", type=str, default="0.7",
                    help="Comma-separated β_C values. Default: 0.7.")
    ap.add_argument("--diffc-values", type=str, default="0.57024",
                    help="Comma-separated diffusion coefficients [m²/d]. "
                         "Default: 0.57024.")
    ap.add_argument("--hk-values", type=str, default="864.0",
                    help="Comma-separated hydraulic conductivity values [m/d]. "
                         "Default: 864.0.")
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
    ap.add_argument(
        "--t-split", type=int, default=None,
        help=(
            "Number of downsampled input time steps [0, t_split). "
            "Remaining steps [t_split, T) become the target. "
            "Defaults to T // 2 where T = len(range(0, nstp+1, skip))."
        ),
    )
    ap.add_argument(
        "--mode", type=str, choices=["split", "ic_repeat"], default="split",
        help=(
            "Tensor layout. 'split': input [0, t_split) -> target [t_split, T). "
            "'ic_repeat': input C0 repeated T-1 times (channels C0, beta_c, "
            "diffc) -> target full trajectory [1, T). Default: split."
        ),
    )
    ap.add_argument(
        "--skip", type=int, default=1,
        help=(
            "Temporal stride for downsampling the MODFLOW time-series before "
            "splitting: time_series[::skip]. skip=1 uses every step. Default: 1."
        ),
    )
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing scenario.npz files.")
    ap.add_argument("--max-runs-per-scenario", type=int, default=None,
                    help="Cap on run combinations per scenario (for quick tests).")
    ap.add_argument("--save-timeseries", action="store_true",
                    help="Include full downsampled head/conc time-series in "
                         "scenario.npz.")
    ap.add_argument("--save-modflow-files", action="store_true",
                    help="Keep all MODFLOW 6 workspace files (default: prune).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed. Default: 42.")
    ap.add_argument(
        "--n-workers", type=int, default=None,
        help=(
            "Number of parallel worker processes for scenario-level parallelism. "
            "Each worker runs all simulations for one scenario independently. "
            "Defaults to min(n_scenarios, os.cpu_count()). "
            "Set to 1 to run sequentially (useful for debugging)."
        ),
    )

    # Executable
    ap.add_argument("--mf6-exe", type=str, default="mf6",
                    help="MODFLOW 6 executable name or path. Default: mf6.")

    # Optional spatially varying K
    ap.add_argument("--kappa-file", type=str, default=None,
                    help=(
                        "Path to an .npz file with 'hk' (and optionally 'vk') "
                        "arrays of shape (nlay, ncol). Overrides --hk-values."
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

    if args.init_method == "wedge":
        init_field_args = {
            "c0_x_toe":       _parse_float_csv(args.c0_x_toe_values),
            "c0_x_top":       _parse_float_csv(args.c0_x_top_values),
            "c0_trans_width": _parse_float_csv(args.c0_trans_width_values),
        }
    elif args.init_method == "random":
        rf_types  = [s.strip() for s in args.random_field_type.split(",") if s.strip()]
        rf_smooth = _parse_float_csv(str(args.random_field_smoothness))
        rf_len    = _parse_float_csv(str(args.random_field_len_scale))
        rf_var    = _parse_float_csv(str(args.random_field_var))
        init_field_args = {
            "random_field_type":       rf_types,
            "random_field_smoothness": rf_smooth,
            "random_field_len_scale":  rf_len,
            "random_field_var":        rf_var,
            "random_field_count":      int(args.random_field_count),
        }
    else:
        raise ValueError(
            f"Unknown init_method '{args.init_method}'. Expected 'wedge' or 'random'."
        )

    generate_simple_henry_dataset(
        outdir=outdir,
        beta_c_values=_parse_float_csv(args.beta_c_values),
        diffc_values=_parse_float_csv(args.diffc_values),
        hk_values=_parse_float_csv(args.hk_values),
        por_values=_parse_float_csv(args.por_values),
        init_method=args.init_method,
        init_field_args=init_field_args,
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
        t_split=args.t_split,
        skip=args.skip,
        overwrite=args.overwrite,
        max_runs_per_scenario=args.max_runs_per_scenario,
        save_timeseries=args.save_timeseries,
        save_modflow_files=args.save_modflow_files,
        seed=args.seed,
        exe_name=args.mf6_exe,
        n_workers=args.n_workers,
        mode=args.mode,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)
