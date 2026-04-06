import itertools
import json

import numpy as np

from .simulation import build_and_run_henry
from .utils import build_splits, valid_window_indices


STANDARD_INIT_HEAD = 1.0
STANDARD_INIT_CONCENTRATION = 35.0


def _scenario_tag(beta_c, diffc):
    return f"scenario_beta{beta_c:.3f}_diffc{diffc:.5f}"


def _run_tag(run_index, params):
    return (
        f"run_{run_index:06d}_"
        f"hk{params['hk']:.2f}_"
        f"por{params['por']:.3f}_"
        f"in{params['inflow']:.4f}_"
        f"ghb{params['ghb_head']:.4f}"
    )


def _broadcast_channel(value, nlay, ncol):
    return np.full((nlay, ncol), float(value), dtype=float)


def _initial_head_field(nlay, ncol):
    # Keep head and concentration ICs explicit and separate for clarity.
    return np.full((nlay, ncol), float(STANDARD_INIT_HEAD), dtype=float)


def _initial_concentration_field(nlay, ncol):
    return np.full((nlay, ncol), float(STANDARD_INIT_CONCENTRATION), dtype=float)


def _build_window_tensors(head_ts, conc_ts, lag, nlay, ncol, params):
    t_indices = valid_window_indices(head_ts.shape[0], lag)
    n_windows = int(t_indices.size)
    if n_windows == 0:
        return None

    cin = 8
    input_tensor = np.empty((n_windows, cin, nlay, ncol), dtype=np.float32)
    output_tensor = np.empty((n_windows, 2, nlay, ncol), dtype=np.float32)

    hk_field = _broadcast_channel(params["hk"], nlay, ncol)
    por_field = _broadcast_channel(params["por"], nlay, ncol)
    inflow_field = _broadcast_channel(params["inflow"], nlay, ncol)
    ghb_field = _broadcast_channel(params["ghb_head"], nlay, ncol)
    beta_field = _broadcast_channel(params["beta_c"], nlay, ncol)
    diffc_field = _broadcast_channel(params["diffc"], nlay, ncol)

    for i, t in enumerate(t_indices):
        t_lag = t + lag
        input_tensor[i, 0] = conc_ts[t]
        input_tensor[i, 1] = head_ts[t]
        input_tensor[i, 2] = hk_field
        input_tensor[i, 3] = por_field
        input_tensor[i, 4] = inflow_field
        input_tensor[i, 5] = ghb_field
        input_tensor[i, 6] = beta_field
        input_tensor[i, 7] = diffc_field

        output_tensor[i, 0] = conc_ts[t_lag]
        output_tensor[i, 1] = head_ts[t_lag]

    return {
        "input_tensor": input_tensor,
        "output_tensor": output_tensor,
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
    warm_start,
    seed,
    train_frac,
    val_frac,
):
    outdir.mkdir(parents=True, exist_ok=True)

    if hk_field is not None and len(hk_values) != 1:
        raise ValueError("hk sweep is incompatible with --kappa-file; provide one hk value")
    if vk_field is not None and len(hk_values) != 1:
        raise ValueError("vk from kappa file is incompatible with hk sweep")

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
                head_ts, conc_ts, times = build_and_run_henry(
                    workspace=run_dir,
                    ncol=ncol,
                    nlay=nlay,
                    total_time=total_time,
                    nstp=nstp,
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
                    strt_head=strt_head,
                    strt_conc=strt_conc,
                    hk_field=hk_field,
                    vk_field=vk_field,
                    return_timeseries=True,
                    exe_name=exe_name,
                )

                prev_head_final = np.asarray(head_ts[-1], dtype=float).copy()
                prev_conc_final = np.asarray(conc_ts[-1], dtype=float).copy()

                windowed = _build_window_tensors(
                    head_ts=head_ts,
                    conc_ts=conc_ts,
                    lag=lag,
                    nlay=nlay,
                    ncol=ncol,
                    params=params,
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
                    "t_index": windowed["t_index"],
                    "t_lag_index": windowed["t_lag_index"],
                    "time_t": times[windowed["t_index"]],
                    "time_t_lag": times[windowed["t_lag_index"]],
                    "window_ids": np.asarray(window_ids),
                    "lag": int(lag),
                    "ncol": ncol,
                    "nlay": nlay,
                    "total_time": total_time,
                    "nstp": nstp,
                    **params,
                }
                if save_timeseries:
                    payload["head_timeseries"] = head_ts
                    payload["conc_timeseries"] = conc_ts
                    payload["times"] = times

                np.savez_compressed(sample_file, **payload)

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
            "fallback_head": float(STANDARD_INIT_HEAD),
            "fallback_concentration": float(STANDARD_INIT_CONCENTRATION),
            "notes": "GWF and GWT initial conditions use separate explicit constants.",
        },
        "lag": int(lag),
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
