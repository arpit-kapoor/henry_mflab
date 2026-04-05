import argparse
import pathlib as pl

from .generators import generate_configurable_dataset
from .utils import load_kappa_fields, parse_float_csv


def build_parser():
    ap = argparse.ArgumentParser(
        description="Generate Henry problem datasets with configurable IC/BC/parameter sweeps."
    )
    ap.add_argument("--outdir", type=str, default="./out")
    ap.add_argument("--ncol", type=int, default=80)
    ap.add_argument("--nlay", type=int, default=40)
    ap.add_argument("--total-time", type=float, default=0.5)
    ap.add_argument("--nstp", type=int, default=500)

    ap.add_argument("--beta-c-values", type=str, default="0.0,0.2,0.4,0.7,1.0")
    ap.add_argument("--diffc-values", type=str, default="0.57024,0.28512,0.14256,0.07128")
    ap.add_argument("--al-values", type=str, default="0.0")
    ap.add_argument("--at-values", type=str, default="0.0")

    ap.add_argument("--inflow-values", type=str, default="2.851")
    ap.add_argument("--ghb-head-values", type=str, default="1.0")
    ap.add_argument("--cinlet-values", type=str, default="35.0")

    ap.add_argument("--strt-head-values", type=str, default="35.0")
    ap.add_argument("--strt-conc-values", type=str, default="35.0")

    ap.add_argument("--por", type=float, default=0.35)
    ap.add_argument("--hk", type=float, default=864.0)
    ap.add_argument("--vk", type=float, default=864.0)
    ap.add_argument("--kappa-file", type=str, default=None)

    ap.add_argument(
        "--right-bc-kind",
        choices=["ghb_head", "cinlet"],
        default="ghb_head",
        help="Scalar used for right boundary channel in input_tensor.",
    )

    ap.add_argument("--save-timeseries", action="store_true")
    ap.add_argument("--mf6-exe", type=str, default="mf6")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
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
    generate_configurable_dataset(
        outdir=outdir,
        beta_c_values=parse_float_csv(args.beta_c_values),
        diffc_values=parse_float_csv(args.diffc_values),
        al_values=parse_float_csv(args.al_values),
        at_values=parse_float_csv(args.at_values),
        inflow_values=parse_float_csv(args.inflow_values),
        ghb_head_values=parse_float_csv(args.ghb_head_values),
        cinlet_values=parse_float_csv(args.cinlet_values),
        strt_head_values=parse_float_csv(args.strt_head_values),
        strt_conc_values=parse_float_csv(args.strt_conc_values),
        ncol=args.ncol,
        nlay=args.nlay,
        total_time=args.total_time,
        nstp=args.nstp,
        por=args.por,
        hk=args.hk,
        vk=args.vk,
        hk_field=hk_field,
        vk_field=vk_field,
        exe_name=args.mf6_exe,
        overwrite=args.overwrite,
        max_runs=args.max_runs,
        save_timeseries=args.save_timeseries,
        right_bc_kind=args.right_bc_kind,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)
