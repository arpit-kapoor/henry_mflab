#!/usr/bin/env python
import argparse
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
    inflow=5.7024,
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

    sim = flopy.mf6.MFSimulation(sim_name="henry", sim_ws=str(ws), exe_name="mf6")

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
        gwf, packagedata=[(0, 0.7, 0.0, "gwt", "concentration")]
    )

    ghbcond = hk * delv * delc / (0.5 * delr)
    ghb_spd = [[(k, 0, ncol - 1), top, ghbcond, 35.0] for k in range(nlay)]
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

    head = hobj.get_alldata()[-1].squeeze()  # (nlay, ncol)
    conc = cobj.get_alldata()[-1].squeeze()  # (nlay, ncol)

    return head, conc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./out")
    args = ap.parse_args()
    outdir = pl.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    head, conc = build_and_run_henry(workspace=outdir)

    np.savez(outdir / "henry_final.npz", head=head, conc=conc)
    print(f"Saved: {outdir/'henry_final.npz'}  head{head.shape} conc{conc.shape}")


if __name__ == "__main__":
    main()
