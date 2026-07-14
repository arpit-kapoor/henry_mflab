import itertools
import json
import shutil

import numpy as np

from .simulation import build_and_run_henry
from .utils import build_splits, valid_window_indices


def resolve_lag(lag_steps: int, lag_days: float | None, dt: float) -> int:
    """Resolve the prediction lag to a number of time steps.

    If *lag_days* is provided it takes priority and is converted to the nearest
    positive integer number of steps using ``dt`` (days per step).
    Otherwise, *lag_steps* is returned unchanged.

    Parameters
    ----------
    lag_steps:
        Fixed lag expressed as a count of time steps (fallback).
    lag_days:
        Desired lag in wall-clock days.  ``None`` defers to *lag_steps*.
    dt:
        Size of a single time step in days (``total_time / nstp``).
    """
    if lag_days is not None:
        return max(1, round(lag_days / dt))
    return lag_steps


STANDARD_INIT_HEAD = 1.0
STANDARD_INIT_CONCENTRATION = 35.0
INPUT_CHANNEL_NAMES = (
    "concentration_t",
    "head_t",
    # "hk",
    # "porosity",
    "flux_left_boundary",
    "ghb_flux_right_boundary",
    "cinlet_right_boundary",
    "beta_c",
    "diffc",
)
INPUT_CHANNEL_INDEX = {name: idx for idx, name in enumerate(INPUT_CHANNEL_NAMES)}
REQUIRED_RUN_FILES = {"windows.npz"} #, "gwf.cbc", "gwt.cbc", "gwf.hds", "gwt.ucn"}


def _validate_input_channel_config():
    if len(INPUT_CHANNEL_INDEX) != len(INPUT_CHANNEL_NAMES):
        raise ValueError("INPUT_CHANNEL_NAMES contains duplicate channel names")

    required = {
        "concentration_t",
        "head_t",
        # "hk",
        # "porosity",
        "flux_left_boundary",
        # "ghb_flux_right_boundary",
        "beta_c",
        "diffc",
    }
    missing = sorted(required - set(INPUT_CHANNEL_NAMES))
    if missing:
        raise ValueError(f"Missing required input channels: {missing}")


_validate_input_channel_config()


def _scenario_tag(beta_c, diffc):
    return f"scenario_beta{beta_c:.3f}_diffc{diffc:.5f}"


def _run_tag(run_index, params):
    return (
        f"run_{run_index:06d}_"
        # f"hk{params['hk']:.2f}_"
        # f"por{params['por']:.3f}_"
        f"in{params['inflow']:.4f}_"
        # f"ghb{params['ghb_head']:.4f}"
    )


def _broadcast_channel(value, nlay, ncol):
    return np.full((nlay, ncol), float(value), dtype=float)


def _left_boundary_channel(value, nlay, ncol):
    field = np.zeros((nlay, ncol), dtype=float)
    field[:, 0] = float(value)
    return field


def _right_boundary_channel(value, nlay, ncol):
    field = np.zeros((nlay, ncol), dtype=float)
    field[:, -1] = float(value)
    return field


def _right_boundary_channel_per_layer(values, nlay, ncol):
    """Right-boundary channel where each layer has its own value."""
    field = np.zeros((nlay, ncol), dtype=float)
    field[:, -1] = np.asarray(values, dtype=float)
    return field


def _initial_head_field(nlay, ncol):
    # Keep head and concentration ICs explicit and separate for clarity.
    return np.full((nlay, ncol), float(STANDARD_INIT_HEAD), dtype=float)


def _initial_concentration_field(nlay, ncol):
    return np.full((nlay, ncol), float(STANDARD_INIT_CONCENTRATION), dtype=float)


def _prune_run_workspace(run_dir, keep_files):
    """Delete files not needed for plotting or model training outputs."""
    for child in run_dir.iterdir():
        if child.name in keep_files:
            continue
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child)


def _build_window_tensors(
    head_ts,
    conc_ts,
    q_in_ts,
    q_ghb_ts,
    cinlet_ts,
    lag,
    nlay,
    ncol,
    params,
    times=None,
    tidal_period=None,
):
    """Build (input, output) window tensors from full time-series arrays.

    Parameters
    ----------
    times:
        Array of MF6 output times [days].  Required when ``tidal_period`` is
        provided to compute the optional tidal-phase channel.
    tidal_period:
        M2 semi-diurnal period [days].  When not ``None`` a ``tidal_phase``
        channel (value in ``[0, 2π]``) is appended to the input tensor.
    """
    t_indices = valid_window_indices(head_ts.shape[0], lag)
    n_windows = int(t_indices.size)
    if n_windows == 0:
        return None

    add_tidal_phase = (times is not None and tidal_period is not None)
    channel_names = list(INPUT_CHANNEL_NAMES)
    if add_tidal_phase:
        channel_names = channel_names + ["tidal_phase"]
    channel_index = {name: idx for idx, name in enumerate(channel_names)}

    cin = len(channel_names)
    input_tensor = np.empty((n_windows, cin, nlay, ncol), dtype=np.float32)
    output_tensor = np.empty((n_windows, 2, nlay, ncol), dtype=np.float32)

    beta_field = _broadcast_channel(params["beta_c"], nlay, ncol)
    diffc_field = _broadcast_channel(params["diffc"], nlay, ncol)

    static_channel_fields = {
        "beta_c": beta_field,
        "diffc": diffc_field,
    }

    # Fill channels that are constant across all windows once to avoid index-order mistakes.
    for channel_name, field in static_channel_fields.items():
        input_tensor[:, channel_index[channel_name], :, :] = field

    for i, t in enumerate(t_indices):
        t_lag = t + lag
        input_tensor[i, channel_index["concentration_t"]] = conc_ts[t]
        input_tensor[i, channel_index["head_t"]] = head_ts[t]
        input_tensor[i, channel_index["flux_left_boundary"]] = _left_boundary_channel(q_in_ts[t], nlay, ncol)
        input_tensor[i, channel_index["ghb_flux_right_boundary"]] = _right_boundary_channel(q_ghb_ts[t], nlay, ncol)
        input_tensor[i, channel_index["cinlet_right_boundary"]] = _right_boundary_channel_per_layer(cinlet_ts[t], nlay, ncol)

        if add_tidal_phase:
            phase_val = float((2.0 * np.pi * times[t] / tidal_period) % (2.0 * np.pi))
            input_tensor[i, channel_index["tidal_phase"], :, :] = phase_val

        output_tensor[i, 0] = conc_ts[t_lag]
        output_tensor[i, 1] = head_ts[t_lag]

    return {
        "input_tensor": input_tensor,
        "output_tensor": output_tensor,
        "input_channel_names_used": channel_names,
        "t_index": t_indices,
        "t_lag_index": t_indices + lag,
    }


def generate_windowed_scenario_dataset(
    outdir,
    scenario_pairs,
    al_values,
    at_values,
    hk_values,
    por_values,
    inflow_values,
    ghb_head_values,
    cinlet,
    ncol,
    nlay,
    total_time,
    nstp,
    hk_field,
    vk_field,
    exe_name,
    overwrite,
    max_runs_per_scenario,
    lag,
    save_timeseries,
    save_modflow_files,
    warm_start,
    seed,
    train_frac,
    val_frac,
    dynamic_inflow,
    dynamic_tides,
    add_storage,
    # Domain dimensions (default = original Henry benchmark)
    lx=2.0,
    lz=1.0,
    # Warm-start spin-up parameters
    spinup_time=10.0,
    spinup_nstp=80,
    # Lag in wall-clock days (overrides `lag` when not None)
    lag_days=None,
    # Tidal forcing parameters
    tidal_amplitude=0.5,
    spring_neap_amp=0.3,
    tidal_period=0.517,
    spring_neap_period=14.77,
    spring_neap_phase=3.14159,
    tidal_noise_std=0.02,
    slr_rate=0.0,
    # Freshwater inflow parameters (stochastic shot-noise model)
    storm_rate=1.0,
    storm_amp_mean=1.0,
    storm_amp_std=0.5,
    recession_k=3.0,
    ar1_phi=0.85,
    ar1_sigma=0.05,
    inflow_trend_amp=0.0,
    # Optional tidal-phase input channel
    add_tidal_phase=False,
):
    outdir.mkdir(parents=True, exist_ok=True)

    if hk_field is not None and len(hk_values) != 1:
        raise ValueError("hk sweep is incompatible with --kappa-file; provide one hk value")
    if vk_field is not None and len(hk_values) != 1:
        raise ValueError("vk from kappa file is incompatible with hk sweep")

    # Resolve lag: lag_days takes priority over lag (in steps) when provided.
    dt = total_time / nstp  # time step size in days
    lag = resolve_lag(lag, lag_days, dt)

    global_window_ids = []
    scenarios_summary = []
    run_records = []
    run_failures = []

    for scenario_index, (beta_c, diffc) in enumerate(scenario_pairs, start=1):
        scenario_tag = _scenario_tag(beta_c, diffc)
        scenario_dir = outdir / scenario_tag
        scenario_dir.mkdir(parents=True, exist_ok=True)

        run_combinations = list(
            itertools.product(hk_values, por_values, inflow_values, ghb_head_values, al_values, at_values)
        )
        if max_runs_per_scenario is not None:
            run_combinations = run_combinations[:max_runs_per_scenario]

        scenario_runs = []
        scenario_failures = []
        prev_head_final = None
        prev_conc_final = None

        print(
            f"Scenario [{scenario_index:03d}/{len(scenario_pairs):03d}] {scenario_tag} "
            f"with {len(run_combinations)} runs"
        )

        for run_index, combo in enumerate(run_combinations, start=1):
            hk, por, inflow, ghb_head, al, at = combo

            if warm_start and prev_head_final is not None and prev_conc_final is not None:
                strt_head = prev_head_final
                strt_conc = prev_conc_final
                init_mode = "warm_start_previous_run"
            else:
                strt_head = _initial_head_field(nlay=nlay, ncol=ncol)
                strt_conc = _initial_concentration_field(nlay=nlay, ncol=ncol)
                init_mode = "separate_constant_ic"

            params = {
                "beta_c": float(beta_c),
                "diffc": float(diffc),
                "hk": float(hk),
                "por": float(por),
                "inflow": float(inflow),
                "ghb_head": float(ghb_head),
                "al": float(al),
                "at": float(at),
                "cinlet": float(cinlet),
                "init_mode": init_mode,
                "initial_head": float(STANDARD_INIT_HEAD),
                "initial_concentration": float(STANDARD_INIT_CONCENTRATION),
            }

            run_tag = _run_tag(run_index, params)
            run_dir = scenario_dir / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)
            sample_file = run_dir / "windows.npz"

            if sample_file.exists() and not overwrite:
                if not save_modflow_files:
                    _prune_run_workspace(run_dir, REQUIRED_RUN_FILES)
                record = {
                    "id": f"{scenario_tag}/{run_tag}",
                    "scenario": scenario_tag,
                    "run": run_tag,
                    "workspace": str(run_dir),
                    "status": "skipped",
                    **params,
                }
                run_records.append(record)
                scenario_runs.append(record)
                continue

            print(f"  [{run_index:04d}/{len(run_combinations):04d}] RUN  {run_tag}")
            try:
                # ----------------------------------------------------------
                # Two-pass warm-start: run a short spin-up first so that the
                # main simulation begins from a quasi-periodic tidal state
                # rather than an arbitrary uniform initial condition.
                # ----------------------------------------------------------
                common_sim_kwargs = dict(
                    ncol=ncol,
                    nlay=nlay,
                    Lx=lx,
                    Lz=lz,
                    cinlet=params["cinlet"],
                    por=params["por"],
                    hk=params["hk"],
                    vk=params["hk"],
                    al=params["al"],
                    at=params["at"],
                    diffc=params["diffc"],
                    inflow=params["inflow"],
                    ghb_head=params["ghb_head"],
                    beta_c=params["beta_c"],
                    hk_field=hk_field,
                    vk_field=vk_field,
                    exe_name=exe_name,
                    dynamic_inflow=dynamic_inflow,
                    dynamic_tides=dynamic_tides,
                    add_storage=add_storage,
                    tidal_amplitude=tidal_amplitude,
                    spring_neap_amp=spring_neap_amp,
                    tidal_period=tidal_period,
                    spring_neap_period=spring_neap_period,
                    spring_neap_phase=spring_neap_phase,
                    tidal_noise_std=tidal_noise_std,
                    slr_rate=slr_rate,
                    storm_rate=storm_rate,
                    storm_amp_mean=storm_amp_mean,
                    storm_amp_std=storm_amp_std,
                    recession_k=recession_k,
                    ar1_phi=ar1_phi,
                    ar1_sigma=ar1_sigma,
                    inflow_trend_amp=inflow_trend_amp,
                )

                spinup_dir = run_dir / "_spinup"
                print(f"    spin-up ({spinup_time} days, {spinup_nstp} steps) ...")
                spinup_head, spinup_conc, *_ = build_and_run_henry(
                    workspace=spinup_dir,
                    total_time=spinup_time,
                    nstp=spinup_nstp,
                    strt_head=strt_head,
                    strt_conc=strt_conc,
                    return_timeseries=False,
                    **common_sim_kwargs,
                )
                shutil.rmtree(spinup_dir, ignore_errors=True)

                print(f"    main run ({total_time} days, {nstp} steps) ...")
                head_ts, conc_ts, q_in_ts, q_ghb_ts, cinlet_ts, times = build_and_run_henry(
                    workspace=run_dir,
                    total_time=total_time,
                    nstp=nstp,
                    strt_head=spinup_head,
                    strt_conc=spinup_conc,
                    return_timeseries=True,
                    **common_sim_kwargs,
                )

                prev_head_final = np.asarray(head_ts[-1], dtype=float).copy()
                prev_conc_final = np.asarray(conc_ts[-1], dtype=float).copy()

                windowed = _build_window_tensors(
                    head_ts=head_ts,
                    conc_ts=conc_ts,
                    q_in_ts=q_in_ts,
                    q_ghb_ts=q_ghb_ts,
                    cinlet_ts=cinlet_ts,
                    lag=lag,
                    nlay=nlay,
                    ncol=ncol,
                    params=params,
                    times=times if add_tidal_phase else None,
                    tidal_period=tidal_period if add_tidal_phase else None,
                )
                if windowed is None:
                    raise ValueError(
                        f"no valid windows for lag={lag}; available times={head_ts.shape[0]}"
                    )

                window_ids = [
                    f"{scenario_tag}/{run_tag}/w{int(t):05d}"
                    for t in windowed["t_index"].tolist()
                ]
                global_window_ids.extend(window_ids)

                payload = {
                    "input_tensor": windowed["input_tensor"],
                    "output_tensor": windowed["output_tensor"],
                    "input_channel_names": np.asarray(windowed["input_channel_names_used"]),
                    "t_index": windowed["t_index"],
                    "t_lag_index": windowed["t_lag_index"],
                    "time_t": times[windowed["t_index"]],
                    "time_t_lag": times[windowed["t_lag_index"]],
                    "window_ids": np.asarray(window_ids),
                    "lag": int(lag),
                    "lag_days": float(lag * dt),
                    "dt": float(dt),
                    "grid": {
                        "ncol": ncol,
                        "nlay": nlay,
                        "lx": lx,
                        "lz": lz,
                    },
                    "total_time": total_time,
                    "nstp": nstp,
                    "spinup_time": float(spinup_time),
                    "spinup_nstp": int(spinup_nstp),
                    **params,
                }
                if save_timeseries:
                    payload["head_timeseries"] = head_ts
                    payload["conc_timeseries"] = conc_ts
                    payload["times"] = times

                np.savez_compressed(sample_file, **payload)
                if not save_modflow_files:
                    _prune_run_workspace(run_dir, REQUIRED_RUN_FILES)

                record = {
                    "id": f"{scenario_tag}/{run_tag}",
                    "scenario": scenario_tag,
                    "run": run_tag,
                    "workspace": str(run_dir),
                    "status": "ok",
                    "n_windows": int(windowed["input_tensor"].shape[0]),
                    "input_shape": [
                        int(windowed["input_tensor"].shape[1]),
                        int(windowed["input_tensor"].shape[2]),
                        int(windowed["input_tensor"].shape[3]),
                    ],
                    "output_shape": [
                        int(windowed["output_tensor"].shape[1]),
                        int(windowed["output_tensor"].shape[2]),
                        int(windowed["output_tensor"].shape[3]),
                    ],
                    **params,
                }
                run_records.append(record)
                scenario_runs.append(record)
            except Exception as exc:
                failure = {
                    "id": f"{scenario_tag}/{run_tag}",
                    "scenario": scenario_tag,
                    "run": run_tag,
                    "workspace": str(run_dir),
                    "status": "failed",
                    "error": str(exc),
                    **params,
                }
                run_failures.append(failure)
                scenario_failures.append(failure)
                print(f"    FAILED: {exc}")

        scenario_manifest = {
            "scenario": scenario_tag,
            "beta_c": float(beta_c),
            "diffc": float(diffc),
            "lag": int(lag),
            "lag_days": float(lag * dt),
            "dt": float(dt),
            "n_total_runs": len(run_combinations),
            "n_ok_runs": sum(r["status"] == "ok" for r in scenario_runs),
            "n_skipped_runs": sum(r["status"] == "skipped" for r in scenario_runs),
            "n_failed_runs": len(scenario_failures),
            "runs": scenario_runs,
            "failures": scenario_failures,
        }
        with (scenario_dir / "scenario_manifest.json").open("w", encoding="utf-8") as fp:
            json.dump(scenario_manifest, fp, indent=2)

        scenarios_summary.append(
            {
                "scenario": scenario_tag,
                "beta_c": float(beta_c),
                "diffc": float(diffc),
                "n_total_runs": len(run_combinations),
                "n_ok_runs": scenario_manifest["n_ok_runs"],
                "n_skipped_runs": scenario_manifest["n_skipped_runs"],
                "n_failed_runs": scenario_manifest["n_failed_runs"],
            }
        )

    splits = (
        build_splits(global_window_ids, train_frac, val_frac, seed)
        if global_window_ids
        else {"train": [], "val": [], "test": []}
    )

    manifest = {
        "workflow": "windowed_scenario_dataset",
        "initialization": {
            "warm_start": bool(warm_start),
            "spinup_time": float(spinup_time),
            "spinup_nstp": int(spinup_nstp),
            "fallback_head": float(STANDARD_INIT_HEAD),
            "fallback_concentration": float(STANDARD_INIT_CONCENTRATION),
            "notes": "Each run is preceded by a spin-up pass whose final state is used as IC.",
        },
        "tidal_forcing": {
            "tidal_amplitude": float(tidal_amplitude),
            "spring_neap_amp": float(spring_neap_amp),
            "tidal_period": float(tidal_period),
            "spring_neap_period": float(spring_neap_period),
            "spring_neap_phase": float(spring_neap_phase),
            "tidal_noise_std": float(tidal_noise_std),
            "slr_rate": float(slr_rate),
        },
        "inflow_forcing": {
            "storm_rate": float(storm_rate),
            "storm_amp_mean": float(storm_amp_mean),
            "storm_amp_std": float(storm_amp_std),
            "recession_k": float(recession_k),
            "ar1_phi": float(ar1_phi),
            "ar1_sigma": float(ar1_sigma),
            "inflow_trend_amp": float(inflow_trend_amp),
        },
        "artifacts": {
            "save_modflow_files": bool(save_modflow_files),
            "required_run_files": sorted(REQUIRED_RUN_FILES),
        },
        "lag": int(lag),
        "lag_days": float(lag * dt),
        "dt": float(dt),
        "add_tidal_phase": bool(add_tidal_phase),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "n_scenarios": len(scenario_pairs),
        "n_total_runs": sum(s["n_total_runs"] for s in scenarios_summary),
        "n_ok_runs": sum(s["n_ok_runs"] for s in scenarios_summary),
        "n_skipped_runs": sum(s["n_skipped_runs"] for s in scenarios_summary),
        "n_failed_runs": sum(s["n_failed_runs"] for s in scenarios_summary),
        "n_total_windows": len(global_window_ids),
        "scenarios": scenarios_summary,
        "splits": splits,
        "runs": run_records,
        "failures": run_failures,
    }

    with (outdir / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(
        "Generation done: "
        f"scenarios={manifest['n_scenarios']} "
        f"runs_ok={manifest['n_ok_runs']} "
        f"runs_failed={manifest['n_failed_runs']} "
        f"windows={manifest['n_total_windows']}"
    )
    print(f"Manifest: {outdir / 'manifest.json'}")
