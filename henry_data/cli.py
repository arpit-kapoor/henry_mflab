import argparse
import pathlib as pl

from .generators import generate_windowed_scenario_dataset
from .utils import (
    build_coupling_diffusion_scenario_pairs,
    load_kappa_fields,
    parse_float_csv,
    parse_scenario_pairs,
)


def build_parser():
    ap = argparse.ArgumentParser(
        description="Generate scenario-windowed Henry datasets for model training."
    )
    ap.add_argument("--outdir", type=str, default="./out")
    ap.add_argument("--ncol", type=int, default=80)
    ap.add_argument("--nlay", type=int, default=40)
    ap.add_argument("--total-time", type=float, default=0.5)
    ap.add_argument("--nstp", type=int, default=500)
    ap.add_argument("--lag", type=int, default=1)

    ap.add_argument(
        "--scenario-pairs",
        type=str,
        default="0.7:0.57024",
        help="Comma-separated beta_c:diffc pairs, e.g. '0.0:0.57024,0.7:0.28512'.",
    )
    ap.add_argument(
        "--coupling-diffusion-grid",
        action="store_true",
        help=(
            "Generate scenario pairs from linear-spaced beta_c and diffc ranges "
            "as a full Cartesian grid."
        ),
    )
    ap.add_argument("--beta-min", type=float, default=0.01)
    ap.add_argument("--beta-max", type=float, default=1.0)
    ap.add_argument("--beta-count", type=int, default=10)
    ap.add_argument("--diffc-min", type=float, default=0.001)
    ap.add_argument("--diffc-max", type=float, default=1.0)
    ap.add_argument("--diffc-count", type=int, default=10)
    ap.add_argument("--fixed-beta", type=float, default=0.70)
    ap.add_argument("--fixed-diffc", type=float, default=0.57024)

    ap.add_argument("--hk-values", type=str, default="864.0")
    ap.add_argument("--por-values", type=str, default="0.35")
    ap.add_argument("--inflow-values", type=str, default="2.851")
    ap.add_argument("--ghb-head-values", type=str, default="1.0")

    ap.add_argument("--al-values", type=str, default="0.0")
    ap.add_argument("--at-values", type=str, default="0.0")
    ap.add_argument("--cinlet", type=float, default=35.0)
    ap.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start each run from the previous successful run within a scenario.",
    )

    ap.add_argument("--kappa-file", type=str, default=None)

    ap.add_argument("--save-timeseries", action="store_true")
    ap.add_argument(
        "--save-modflow-files",
        action="store_true",
        help="Keep full MODFLOW6 workspace files for each run (default keeps only windows.npz).",
    )
    ap.add_argument("--mf6-exe", type=str, default="mf6")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-runs-per-scenario", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    return ap


def run(args):
    outdir = pl.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hk_field, vk_field = (None, None)
    if args.kappa_file:
        hk_field, vk_field = load_kappa_fields(args.kappa_file)

    if args.coupling_diffusion_grid:
        scenario_pairs = build_coupling_diffusion_scenario_pairs(
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            beta_count=args.beta_count,
            diffc_min=args.diffc_min,
            diffc_max=args.diffc_max,
            diffc_count=args.diffc_count,
            fixed_beta=args.fixed_beta,
            fixed_diffc=args.fixed_diffc,
        )
    else:
        scenario_pairs = parse_scenario_pairs(args.scenario_pairs)

    generate_windowed_scenario_dataset(
        outdir=outdir,
        scenario_pairs=scenario_pairs,
        al_values=parse_float_csv(args.al_values),
        at_values=parse_float_csv(args.at_values),
        hk_values=parse_float_csv(args.hk_values),
        por_values=parse_float_csv(args.por_values),
        inflow_values=parse_float_csv(args.inflow_values),
        ghb_head_values=parse_float_csv(args.ghb_head_values),
        cinlet=args.cinlet,
        ncol=args.ncol,
        nlay=args.nlay,
        total_time=args.total_time,
        nstp=args.nstp,
        hk_field=hk_field,
        vk_field=vk_field,
        exe_name=args.mf6_exe,
        overwrite=args.overwrite,
        max_runs_per_scenario=args.max_runs_per_scenario,
        lag=args.lag,
        save_timeseries=args.save_timeseries,
        save_modflow_files=args.save_modflow_files,
        warm_start=args.warm_start,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)
