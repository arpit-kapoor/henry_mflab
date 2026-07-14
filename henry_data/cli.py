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
    ap.add_argument("--ncol", type=int, default=40)
    ap.add_argument("--nlay", type=int, default=20)
    ap.add_argument("--lx", type=float, default=2.0, help="Domain horizontal extent [m]. Default: 2.0.")
    ap.add_argument("--lz", type=float, default=1.0, help="Domain vertical extent [m]. Default: 1.0.")
    ap.add_argument("--total-time", type=float, default=30.0)
    ap.add_argument("--nstp", type=int, default=240)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument(
        "--lag-days",
        type=float,
        default=None,
        help=(
            "Prediction lag expressed in wall-clock days. "
            "Overrides --lag when provided; resolved to steps as round(lag_days / dt)."
        ),
    )

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

    ap.add_argument("--dynamic-inflow", action="store_true", help="Use a dynamic inflow boundary condition and save its time series data.")
    ap.add_argument("--dynamic-tides", action="store_true", help="Use dynamic tidal boundary conditions and save their time series data.")
    ap.add_argument("--add-storage", action="store_true", help="Add storage to the model and include it as a feature in the dataset.")

    # --- Tidal forcing parameters ---
    ap.add_argument(
        "--tidal-amplitude",
        type=float,
        default=0.5,
        help="Mean M2 tidal amplitude [m] around the mean sea level (ghb_head). Default: 0.5.",
    )
    ap.add_argument(
        "--spring-neap-amp",
        type=float,
        default=0.3,
        help="Spring-neap modulation amplitude [m]. Spring-tide amp = tidal-amplitude + spring-neap-amp. Default: 0.3.",
    )
    ap.add_argument(
        "--tidal-period",
        type=float,
        default=0.517,
        help="Semi-diurnal (M2) tidal period [days]. Default: 0.517 (~12.42 h).",
    )
    ap.add_argument(
        "--spring-neap-period",
        type=float,
        default=14.77,
        help="Spring-neap beat period [days]. Default: 14.77.",
    )
    ap.add_argument(
        "--spring-neap-phase",
        type=float,
        default=3.14159,
        help=(
            "Phase offset of the spring-neap envelope [radians]. "
            "Default pi (≈3.14159) starts the simulation at neap tide (minimum "
            "amplitude), so the first spring-tide peak occurs at T_sn/2 ≈7.4 days. "
            "Use 0 to start at spring (maximum amplitude)."
        ),
    )
    ap.add_argument(
        "--tidal-noise-std",
        type=float,
        default=0.02,
        help="Std of Gaussian noise added to tidal head nodes [m]. Default: 0.02.",
    )
    ap.add_argument(
        "--slr-rate",
        type=float,
        default=0.0,
        help=(
            "Sea-level rise rate [m/day] added as a linear drift to the mean tidal "
            "baseline.  0 = no rise (default).  E.g. 0.003 gives +0.09 m over 30 days."
        ),
    )

    # --- Freshwater inflow parameters (stochastic shot-noise model) ---
    ap.add_argument(
        "--storm-rate",
        type=float,
        default=1.0,
        help="Poisson storm arrival rate [storms/day]. Default: 1.0 (one storm/day on average).",
    )
    ap.add_argument(
        "--storm-amp-mean",
        type=float,
        default=1.0,
        help="Mean storm peak amplitude as fraction of q_mean (log-normal mu). Default: 1.0.",
    )
    ap.add_argument(
        "--storm-amp-std",
        type=float,
        default=0.5,
        help="Std of storm peak amplitude fraction (log-normal sigma). Default: 0.5.",
    )
    ap.add_argument(
        "--recession-k",
        type=float,
        default=3.0,
        help="Aquifer recession constant [days]. Larger = slower baseflow decay. Default: 3.0.",
    )
    ap.add_argument(
        "--ar1-phi",
        type=float,
        default=0.85,
        help="AR(1) autocorrelation coefficient in (0,1). Default: 0.85.",
    )
    ap.add_argument(
        "--ar1-sigma",
        type=float,
        default=0.05,
        help="AR(1) white-noise std as fraction of q_mean. Default: 0.05.",
    )
    ap.add_argument(
        "--inflow-trend-amp",
        type=float,
        default=0.0,
        help=(
            "Monotone inflow trend as a fraction of mean per-layer inflow. "
            "Negative = drying (e.g. -0.4 cuts inflow by 40%% by end of run). "
            "0 = no trend (default)."
        ),
    )

    # --- Spin-up parameters ---
    ap.add_argument(
        "--spinup-time",
        type=float,
        default=10.0,
        help="Duration of the warm-start spin-up pre-run [days]. Default: 10.0.",
    )
    ap.add_argument(
        "--spinup-nstp",
        type=int,
        default=80,
        help="Number of time steps for the spin-up pre-run. Default: 80 (8 steps/day × 10 days).",
    )

    # --- Optional tidal-phase channel ---
    ap.add_argument(
        "--add-tidal-phase",
        action="store_true",
        help=(
            "Append a 'tidal_phase' input channel (value in [0, 2π]) to the "
            "input tensor. Increases channel count by 1."
        ),
    )

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
        lx=args.lx,
        lz=args.lz,
        total_time=args.total_time,
        nstp=args.nstp,
        hk_field=hk_field,
        vk_field=vk_field,
        exe_name=args.mf6_exe,
        overwrite=args.overwrite,
        max_runs_per_scenario=args.max_runs_per_scenario,
        lag=args.lag,
        lag_days=args.lag_days,
        save_timeseries=args.save_timeseries,
        save_modflow_files=args.save_modflow_files,
        warm_start=args.warm_start,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        dynamic_inflow=args.dynamic_inflow,
        dynamic_tides=args.dynamic_tides,
        add_storage=args.add_storage,
        spinup_time=args.spinup_time,
        spinup_nstp=args.spinup_nstp,
        tidal_amplitude=args.tidal_amplitude,
        spring_neap_amp=args.spring_neap_amp,
        tidal_period=args.tidal_period,
        spring_neap_period=args.spring_neap_period,
        spring_neap_phase=args.spring_neap_phase,
        tidal_noise_std=args.tidal_noise_std,
        slr_rate=args.slr_rate,
        storm_rate=args.storm_rate,
        storm_amp_mean=args.storm_amp_mean,
        storm_amp_std=args.storm_amp_std,
        recession_k=args.recession_k,
        ar1_phi=args.ar1_phi,
        ar1_sigma=args.ar1_sigma,
        inflow_trend_amp=args.inflow_trend_amp,
        add_tidal_phase=args.add_tidal_phase,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)
