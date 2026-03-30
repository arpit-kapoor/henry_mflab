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
    cinlet=35.0,  # g/L saline left (arbitrary units)
    por=0.35,
    hk=1.0,
    vk=1.0,
    al=0.1,
    at=0.01,
):
    """
    Minimal Henry: 2D column (x-z), 1 row, variable-density-like transport
    using MODFLOW 6 GWT (note: MF6 GWT is species transport; density coupling
    is not automatic—this demo treats classic Henry transport without buoyancy
    feedback; for buoyancy-coupled cases you'd use SEAWAT or MF6 with advanced
    coupling; for ML dataset of the steady wedge, this setup is commonly used).
    """

    ws = pl.Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    # Discretisation
    nrow = 1
    delr = Lx / ncol
    delc = 1.0
    delv = Lz / nlay
    top = 0.0
    botm = [-delv * (k + 1) for k in range(nlay)]
    perlen = [total_time]
    nper = 1
    nstp = [nstp]
    tsmult = [1.0]

    sim = flopy.mf6.MFSimulation(sim_name="henry", sim_ws=str(ws), exe_name="mf6")

    # TDIS
    flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=nper, perioddata=list(zip(perlen, nstp, tsmult))
    )

    # --- IMS: create TWO solvers and register in order (GWF first, then GWT) ---
    ims_gwf = flopy.mf6.ModflowIms(
        sim, print_option="SUMMARY", complexity="SIMPLE", filename="gwf.ims",
        linear_acceleration="BICGSTAB"  # Required for asymmetric matrix
    )
    ims_gwt = flopy.mf6.ModflowIms(
        sim, print_option="SUMMARY", complexity="SIMPLE", filename="gwt.ims",
        linear_acceleration="BICGSTAB"  # Use same solver for consistency
    )

    # ---------------- GWF (flow) ----------------
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname="gwf", newtonoptions="NEWTON", save_flows=True
    )

    # DIS
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm
    )

    # IC (initial head)
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)

    # NPF (conductivity)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=hk, k33=vk)

    # Constant head: left & right boundaries (classic Henry mixed boundary at right)
    chd_spd = []
    for k in range(nlay):
        chd_spd.append(((k, 0, 0), 1.0, 0.0))  # left column head = 1.0, conc = 0.0
        chd_spd.append(((k, 0, ncol - 1), 0.0, 0.0))  # right column head = 0.0, conc = 0.0
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd, pname="CHD",
                           auxiliary=["CONCENTRATION"])

    # Output control - save all time steps for animation
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="gwf.hds",
        budget_filerecord="gwf.cbc",
        saverecord=[("HEAD", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    # ---------------- GWT (transport) ----------------
    gwt = flopy.mf6.ModflowGwt(sim, modelname="gwt", save_flows=True)

    flopy.mf6.ModflowGwtdis(
        gwt, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm
    )
    flopy.mf6.ModflowGwtic(gwt, strt=0.0)

    # Advection + dispersion
    flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")  # change to "TVD" later if desired
    flopy.mf6.ModflowGwtdsp(gwt, alh=al, ath1=at)

    # Sources: left boundary at fixed concentration (Dirichlet via CNC)
    cnc_spd = [((k, 0, 0), cinlet) for k in range(nlay)]
    flopy.mf6.ModflowGwtcnc(gwt, stress_period_data=cnc_spd, pname="CNC")
    
    # Add SSM package for GWT model
    sourcerecarray = [("CHD", "AUX", "CONCENTRATION")]
    flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray)

    # Mass storage params
    flopy.mf6.ModflowGwtmst(gwt, porosity=por)

    # Output control for concentration
    flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord="gwt.ucn",
        budget_filerecord="gwt.cbc",
        saverecord=[("CONCENTRATION", "ALL")],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    )

    # Register IMS packages to models (GWF FIRST, then GWT)
    sim.register_ims_package(ims_gwf, [gwf.name])
    sim.register_ims_package(ims_gwt, [gwt.name])

    # Register GWF-GWT exchange
    flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea="gwf", exgmnameb="gwt")

    # --- Write & run with stronger diagnostics ---
    # If you want each run isolated automatically, use a fresh subfolder like:
    # ws_run = ws / f"run_{np.random.randint(1e9):09d}"
    # ws_run.mkdir(parents=True, exist_ok=True)
    # sim.simulation_data.mfpath.set_sim_path(str(ws_run))

    sim.write_simulation()
    print(f"[INFO] Simulation path: {gwf.simulation_data.mfpath.get_sim_path()}")

    # Show mf6 stdout/stderr; collect messages if it fails
    success, buff = sim.run_simulation(silent=False)

    if not success:
        from pathlib import Path

        simdir = Path(gwf.simulation_data.mfpath.get_sim_path())
        lst = simdir / "mfsim.lst"

        msg = []
        msg.append("[ERROR] MODFLOW 6 failed to run.")
        msg.append("---- mf6 stdout/stderr (buff) ----")
        try:
            msg.append("\n".join(buff))
        except Exception:
            msg.append("(no buff)")

        msg.append("---- directory listing ----")
        try:
            msg.append("\n".join(sorted(p.name for p in simdir.iterdir())))
        except Exception:
            msg.append("(cannot list dir)")

        msg.append("---- mfsim.lst (tail) ----")
        try:
            msg.append("\n".join(lst.read_text(errors="ignore").splitlines()[-200:]))
        except Exception:
            msg.append("(no mfsim.lst)")

        raise RuntimeError("\n".join(msg))

    # Read outputs
    hobj = flopy.utils.HeadFile(ws / "gwf.hds")
    # Use HeadFile for concentration as well (works better with MF6)
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
