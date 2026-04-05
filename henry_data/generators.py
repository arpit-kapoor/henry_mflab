import itertools
import json

import numpy as np

from .simulation import build_and_run_henry
from .utils import build_fno_io_tensors, build_splits


def _sample_tag(i, params):
    return (
        f"sample_{i:06d}_"
        f"beta{params['beta_c']:.3f}_"
        f"diffc{params['diffc']:.5f}_"
        f"in{params['inflow']:.4f}"
    )


def generate_configurable_dataset(
    outdir,
    beta_c_values,
    diffc_values,
    al_values,
    at_values,
    inflow_values,
    ghb_head_values,
    cinlet_values,
    strt_head_values,
    strt_conc_values,
    ncol,
    nlay,
    total_time,
    nstp,
    por,
    hk,
    vk,
    hk_field,
    vk_field,
    exe_name,
    overwrite,
    max_runs,
    save_timeseries,
    right_bc_kind,
    seed,
    train_frac,
    val_frac,
):
    outdir.mkdir(parents=True, exist_ok=True)

    combinations = list(
        itertools.product(
            beta_c_values,
            diffc_values,
            al_values,
            at_values,
            inflow_values,
            ghb_head_values,
            cinlet_values,
            strt_head_values,
            strt_conc_values,
        )
    )
    if max_runs is not None:
        combinations = combinations[:max_runs]

    runs = []
    failures = []

    print(f"Dataset size: {len(combinations)} runs")
    for idx, combo in enumerate(combinations, start=1):
        (
            beta_c,
            diffc,
            al,
            at,
            inflow,
            ghb_head,
            cinlet,
            strt_head,
            strt_conc,
        ) = combo

        params = {
            "beta_c": float(beta_c),
            "diffc": float(diffc),
            "al": float(al),
            "at": float(at),
            "inflow": float(inflow),
            "ghb_head": float(ghb_head),
            "cinlet": float(cinlet),
            "strt_head": float(strt_head),
            "strt_conc": float(strt_conc),
        }

        tag = _sample_tag(idx, params)
        run_ws = outdir / tag
        run_ws.mkdir(parents=True, exist_ok=True)
        sample_file = run_ws / "sample.npz"

        if sample_file.exists() and not overwrite:
            runs.append(
                {
                    "id": tag,
                    "workspace": str(run_ws),
                    "status": "skipped",
                    **params,
                }
            )
            continue

        print(f"[{idx:04d}/{len(combinations):04d}] RUN  {tag}")
        try:
            head_ts, conc_ts, times = build_and_run_henry(
                workspace=run_ws,
                ncol=ncol,
                nlay=nlay,
                total_time=total_time,
                nstp=nstp,
                cinlet=params["cinlet"],
                por=por,
                hk=hk,
                vk=vk,
                al=params["al"],
                at=params["at"],
                diffc=params["diffc"],
                inflow=params["inflow"],
                ghb_head=params["ghb_head"],
                beta_c=params["beta_c"],
                strt_head=params["strt_head"],
                strt_conc=params["strt_conc"],
                hk_field=hk_field,
                vk_field=vk_field,
                return_timeseries=True,
                exe_name=exe_name,
            )

            right_bc_scalar = params["ghb_head"] if right_bc_kind == "ghb_head" else params["cinlet"]
            input_tensor, output_tensor = build_fno_io_tensors(
                nlay=nlay,
                ncol=ncol,
                strt_head=params["strt_head"],
                strt_conc=params["strt_conc"],
                inflow=params["inflow"],
                right_bc_scalar=right_bc_scalar,
                conc_final=conc_ts[-1],
                head_final=head_ts[-1],
            )

            payload = {
                "input_tensor": input_tensor,
                "output_tensor": output_tensor,
                "head_final": head_ts[-1],
                "conc_final": conc_ts[-1],
                "times": times,
                "right_bc_kind": right_bc_kind,
                "right_bc_scalar": right_bc_scalar,
                "ncol": ncol,
                "nlay": nlay,
                "total_time": total_time,
                "nstp": nstp,
                "por": por,
                "hk": hk,
                "vk": vk,
                **params,
            }
            if save_timeseries:
                payload["head_timeseries"] = head_ts
                payload["conc_timeseries"] = conc_ts

            np.savez_compressed(sample_file, **payload)
            runs.append(
                {
                    "id": tag,
                    "workspace": str(run_ws),
                    "status": "ok",
                    "shape": [int(head_ts.shape[1]), int(head_ts.shape[2])],
                    "ntimes": int(head_ts.shape[0]),
                    **params,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "id": tag,
                    "workspace": str(run_ws),
                    "status": "failed",
                    "error": str(exc),
                    **params,
                }
            )
            print(f"  FAILED: {exc}")

    ok_ids = [r["id"] for r in runs if r["status"] == "ok"]
    splits = build_splits(ok_ids, train_frac, val_frac, seed) if ok_ids else {"train": [], "val": [], "test": []}

    manifest = {
        "workflow": "configurable_dataset",
        "n_total": len(combinations),
        "n_ok": sum(r["status"] == "ok" for r in runs),
        "n_skipped": sum(r["status"] == "skipped" for r in runs),
        "n_failed": len(failures),
        "right_bc_kind": right_bc_kind,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "splits": splits,
        "runs": runs,
        "failures": failures,
    }

    with (outdir / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(
        "Generation done: "
        f"ok={manifest['n_ok']} skipped={manifest['n_skipped']} failed={manifest['n_failed']}"
    )
    print(f"Manifest: {outdir / 'manifest.json'}")
