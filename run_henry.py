#!/usr/bin/env python
import argparse
import itertools
import json
import pathlib as pl

import flopy
import numpy as np


def build_and_run_henry(
    workspace,
    ncol=80,
    nlay=40,
    Lx=2.0,
    Lz=1.0,
    total_time=0.5,
    nstp=500,
    cinlet=35.0,  # seawater concentration
    por=0.35,
    hk=864.0,
    vk=864.0,
    al=0.0,
    at=0.0,
    diffc=0.57024,
    inflow=2.851,
    beta_c=0.7,
    return_timeseries=False,
    exe_name="mf6",
):
    ws = pl.Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    # Discretisation
    nrow = 1
    delr = Lx / ncol
    delc = 1.0
    delv = Lz / nlay
    top = Lz
    botm = [Lz - delv * (k + 1) for k in range(nlay)]
    perlen = [total_time]
    nper = 1
    nstp = [nstp]
    tsmult = [1.0]

    sim = flopy.mf6.MFSimulation(sim_name="henry", sim_ws=str(ws), exe_name=exe_name)

    flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=nper, perioddata=list(zip(perlen, nstp, tsmult))
    )

    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-10, 1e-6, 0.97
    
    ims_gwf = flopy.mf6.ModflowIms(
        sim, print_option="SUMMARY", outer_dvclose=hclose, outer_maximum=nouter,
        under_relaxation="NONE", inner_maximum=ninner, inner_dvclose=hclose,
        rcloserecord=rclose, linear_acceleration="BICGSTAB", scaling_method="NONE",
        reordering_method="NONE", relaxation_factor=relax, filename="gwf.ims"
    )
    ims_gwt = flopy.mf6.ModflowIms(
        sim, print_option="SUMMARY", outer_dvclose=hclose, outer_maximum=nouter,
        under_relaxation="NONE", inner_maximum=ninner, inner_dvclose=hclose,
        rcloserecord=rclose, linear_acceleration="BICGSTAB", scaling_method="NONE",
        reordering_method="NONE", relaxation_factor=relax, filename="gwt.ims"
    )

    gwf = flopy.mf6.ModflowGwf(
        sim, modelname="gwf", save_flows=True
    )

    flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm
    )

    flopy.mf6.ModflowGwfic(gwf, strt=35.0)

    flopy.mf6.ModflowGwfnpf(
        gwf, icelltype=0, k=hk, k33=vk, save_specific_discharge=True
    )

    flopy.mf6.ModflowGwfbuy(
        gwf, packagedata=[(0, beta_c, 0.0, "gwt", "concentration")]
    )

    ghbcond = hk * delv * delc / (0.5 * delr)
    ghb_spd = [[(k, 0, ncol - 1), top, ghbcond, cinlet] for k in range(nlay)]
    flopy.mf6.ModflowGwfghb(
        gwf, stress_period_data=ghb_spd, pname="GHB-1",
        auxiliary="CONCENTRATION"
    )

    wel_spd = [[(k, 0, 0), inflow / nlay, 0.0] for k in range(nlay)]
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=wel_spd, pname="WEL-1",
        auxiliary="CONCENTRATION"
    )

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="gwf.hds",
        budget_filerecord="gwf.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    gwt = flopy.mf6.ModflowGwt(sim, modelname="gwt", save_flows=True)

    flopy.mf6.ModflowGwtdis(
        gwt, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm
    )
    flopy.mf6.ModflowGwtic(gwt, strt=35.0)

    flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")
    flopy.mf6.ModflowGwtdsp(gwt, alh=al, ath1=at, xt3d_off=True, diffc=diffc)

    sourcerecarray = [
        ("GHB-1", "AUX", "CONCENTRATION"),
        ("WEL-1", "AUX", "CONCENTRATION"),
    ]
    flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray)

    flopy.mf6.ModflowGwtmst(gwt, porosity=por)

    flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord="gwt.ucn",
        budget_filerecord="gwt.cbc",
        saverecord=[("CONCENTRATION", "ALL")],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    )

    sim.register_ims_package(ims_gwf, [gwf.name])
    sim.register_ims_package(ims_gwt, [gwt.name])

    flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea="gwf", exgmnameb="gwt")


    sim.write_simulation()
    
    success, buff = sim.run_simulation(silent=False)

    if not success:
        raise RuntimeError("MODFLOW 6 failed")

    hobj = flopy.utils.HeadFile(ws / "gwf.hds")
    cobj = flopy.utils.HeadFile(ws / "gwt.ucn", text="CONCENTRATION")

    head_ts = hobj.get_alldata().squeeze()  # (ntimes, nlay, ncol)
    conc_ts = cobj.get_alldata().squeeze()  # (ntimes, nlay, ncol)
    times = np.asarray(hobj.get_times(), dtype=float)

    head = head_ts[-1]  # (nlay, ncol)
    conc = conc_ts[-1]  # (nlay, ncol)

    if return_timeseries:
        return head_ts, conc_ts, times

    return head, conc


def _parse_float_csv(values):
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def run_sweep(
    outdir,
    beta_c_values,
    diffc_values,
    al_values,
    at_values,
    ncol,
    nlay,
    total_time,
    nstp,
    cinlet,
    por,
    hk,
    vk,
    inflow,
    exe_name,
    overwrite=False,
    max_runs=None,
):
    outdir.mkdir(parents=True, exist_ok=True)

    combinations = list(
        itertools.product(beta_c_values, diffc_values, al_values, at_values)
    )
    if max_runs is not None:
        combinations = combinations[:max_runs]

    records = []
    failures = []

    print(f"Sweep size: {len(combinations)} runs")
    for idx, (beta_c, diffc, al, at) in enumerate(combinations, start=1):
        tag = f"beta{beta_c:.3f}_diffc{diffc:.5f}_al{al:.4f}_at{at:.4f}"
        run_ws = outdir / tag
        run_ws.mkdir(parents=True, exist_ok=True)
        timeseries_file = run_ws / "henry_timeseries.npz"

        if timeseries_file.exists() and not overwrite:
            print(f"[{idx:03d}/{len(combinations):03d}] SKIP {tag}")
            records.append(
                {
                    "id": tag,
                    "workspace": str(run_ws),
                    "beta_c": beta_c,
                    "diffc": diffc,
                    "al": al,
                    "at": at,
                    "status": "skipped",
                }
            )
            continue

        print(f"[{idx:03d}/{len(combinations):03d}] RUN  {tag}")
        try:
            head_ts, conc_ts, times = build_and_run_henry(
                workspace=run_ws,
                ncol=ncol,
                nlay=nlay,
                total_time=total_time,
                nstp=nstp,
                cinlet=cinlet,
                por=por,
                hk=hk,
                vk=vk,
                al=al,
                at=at,
                diffc=diffc,
                inflow=inflow,
                beta_c=beta_c,
                return_timeseries=True,
                exe_name=exe_name,
            )
            np.savez_compressed(
                timeseries_file,
                head=head_ts,
                conc=conc_ts,
                times=times,
                beta_c=beta_c,
                diffc=diffc,
                al=al,
                at=at,
                ncol=ncol,
                nlay=nlay,
                total_time=total_time,
                nstp=nstp,
                cinlet=cinlet,
                por=por,
                hk=hk,
                vk=vk,
                inflow=inflow,
            )
            np.savez(run_ws / "henry_final.npz", head=head_ts[-1], conc=conc_ts[-1])
            records.append(
                {
                    "id": tag,
                    "workspace": str(run_ws),
                    "beta_c": beta_c,
                    "diffc": diffc,
                    "al": al,
                    "at": at,
                    "status": "ok",
                    "ntimes": int(head_ts.shape[0]),
                    "shape": [int(head_ts.shape[1]), int(head_ts.shape[2])],
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "id": tag,
                    "workspace": str(run_ws),
                    "beta_c": beta_c,
                    "diffc": diffc,
                    "al": al,
                    "at": at,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"  FAILED: {exc}")

    manifest = {
        "n_total": len(combinations),
        "n_ok": sum(r["status"] == "ok" for r in records),
        "n_skipped": sum(r["status"] == "skipped" for r in records),
        "n_failed": len(failures),
        "runs": records,
        "failures": failures,
    }
    with (outdir / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(
        "Sweep done: "
        f"ok={manifest['n_ok']} skipped={manifest['n_skipped']} failed={manifest['n_failed']}"
    )
    print(f"Manifest: {outdir / 'manifest.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "sweep"], default="single")
    ap.add_argument("--outdir", type=str, default="./out")
    ap.add_argument("--ncol", type=int, default=80)
    ap.add_argument("--nlay", type=int, default=40)
    ap.add_argument("--total-time", type=float, default=0.5)
    ap.add_argument("--nstp", type=int, default=500)
    ap.add_argument("--cinlet", type=float, default=35.0)
    ap.add_argument("--por", type=float, default=0.35)
    ap.add_argument("--hk", type=float, default=864.0)
    ap.add_argument("--vk", type=float, default=864.0)
    ap.add_argument("--inflow", type=float, default=2.851)
    ap.add_argument("--beta-c", type=float, default=0.7)
    ap.add_argument("--al", type=float, default=0.0)
    ap.add_argument("--at", type=float, default=0.0)
    ap.add_argument("--diffc", type=float, default=0.57024)
    ap.add_argument("--save-timeseries", action="store_true")
    ap.add_argument("--mf6-exe", type=str, default="mf6")

    ap.add_argument("--beta-c-values", type=str, default="0.0,0.2,0.4,0.7,1.0")
    ap.add_argument("--diffc-values", type=str, default="0.57024,0.28512,0.14256,0.07128")
    ap.add_argument("--al-values", type=str, default="0.0,0.005,0.01")
    ap.add_argument("--at-values", type=str, default="0.0,0.001")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()
    outdir = pl.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        if args.save_timeseries:
            head_ts, conc_ts, times = build_and_run_henry(
                workspace=outdir,
                ncol=args.ncol,
                nlay=args.nlay,
                total_time=args.total_time,
                nstp=args.nstp,
                cinlet=args.cinlet,
                por=args.por,
                hk=args.hk,
                vk=args.vk,
                al=args.al,
                at=args.at,
                diffc=args.diffc,
                inflow=args.inflow,
                beta_c=args.beta_c,
                return_timeseries=True,
                exe_name=args.mf6_exe,
            )
            np.savez_compressed(
                outdir / "henry_timeseries.npz",
                head=head_ts,
                conc=conc_ts,
                times=times,
                beta_c=args.beta_c,
                diffc=args.diffc,
                al=args.al,
                at=args.at,
                ncol=args.ncol,
                nlay=args.nlay,
                total_time=args.total_time,
                nstp=args.nstp,
                cinlet=args.cinlet,
                por=args.por,
                hk=args.hk,
                vk=args.vk,
                inflow=args.inflow,
            )
            np.savez(outdir / "henry_final.npz", head=head_ts[-1], conc=conc_ts[-1])
            print(
                f"Saved: {outdir / 'henry_timeseries.npz'} "
                f"head{head_ts.shape} conc{conc_ts.shape}"
            )
        else:
            head, conc = build_and_run_henry(
                workspace=outdir,
                ncol=args.ncol,
                nlay=args.nlay,
                total_time=args.total_time,
                nstp=args.nstp,
                cinlet=args.cinlet,
                por=args.por,
                hk=args.hk,
                vk=args.vk,
                al=args.al,
                at=args.at,
                diffc=args.diffc,
                inflow=args.inflow,
                beta_c=args.beta_c,
                exe_name=args.mf6_exe,
            )
            np.savez(outdir / "henry_final.npz", head=head, conc=conc)
            print(f"Saved: {outdir / 'henry_final.npz'}  head{head.shape} conc{conc.shape}")
        return

    run_sweep(
        outdir=outdir,
        beta_c_values=_parse_float_csv(args.beta_c_values),
        diffc_values=_parse_float_csv(args.diffc_values),
        al_values=_parse_float_csv(args.al_values),
        at_values=_parse_float_csv(args.at_values),
        ncol=args.ncol,
        nlay=args.nlay,
        total_time=args.total_time,
        nstp=args.nstp,
        cinlet=args.cinlet,
        por=args.por,
        hk=args.hk,
        vk=args.vk,
        inflow=args.inflow,
        exe_name=args.mf6_exe,
        overwrite=args.overwrite,
        max_runs=args.max_runs,
    )


if __name__ == "__main__":
    main()
