import pathlib as pl

import flopy
import numpy as np

from .utils import to_layer_col_field


def build_and_run_henry(
    workspace,
    ncol=80,
    nlay=40,
    Lx=2.0,
    Lz=1.0,
    total_time=0.5,
    nstp=500,
    cinlet=35.0,
    por=0.35,
    hk=864.0,
    vk=864.0,
    al=0.0,
    at=0.0,
    diffc=0.57024,
    inflow=2.851,
    ghb_head=None,
    beta_c=0.7,
    strt_head=1.0,
    strt_conc=35.0,
    hk_field=None,
    vk_field=None,
    return_timeseries=False,
    exe_name="mf6",
):
    ws = pl.Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    nrow = 1
    delr = Lx / ncol
    delc = 1.0
    delv = Lz / nlay
    top = Lz
    if ghb_head is None:
        ghb_head = top
    botm = [Lz - delv * (k + 1) for k in range(nlay)]
    perlen = [total_time]
    nper = 1
    nstp = [nstp]
    tsmult = [1.0]

    # Create a MODFLOW 6 simulation container (workspace + executable).
    sim = flopy.mf6.MFSimulation(sim_name="henry", sim_ws=str(ws), exe_name=exe_name)

    # Define time discretization (single stress period split into nstp time steps).
    flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=nper, perioddata=list(zip(perlen, nstp, tsmult))
    )

    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-10, 1e-6, 0.97

    # Iterative solver settings for the groundwater flow model (GWF).
    ims_gwf = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename="gwf.ims",
    )
    # Separate iterative solver settings for the transport model (GWT).
    ims_gwt = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename="gwt.ims",
    )

    # Build the groundwater flow model.
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)

    # Structured grid discretization for flow.
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm
    )

    head0 = to_layer_col_field(strt_head, nlay, ncol, "strt_head")
    conc0 = to_layer_col_field(strt_conc, nlay, ncol, "strt_conc")
    hk_arr = to_layer_col_field(hk if hk_field is None else hk_field, nlay, ncol, "hk_field")
    vk_arr = to_layer_col_field(vk if vk_field is None else vk_field, nlay, ncol, "vk_field")

    # Initial hydraulic head distribution.
    flopy.mf6.ModflowGwfic(gwf, strt=head0.reshape(nlay, 1, ncol))

    # Hydraulic properties (horizontal K and vertical K33).
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=hk_arr.reshape(nlay, 1, ncol),
        k33=vk_arr.reshape(nlay, 1, ncol),
        save_specific_discharge=True,
    )

    # Buoyancy coupling: maps concentration from GWT to density effects in GWF.
    flopy.mf6.ModflowGwfbuy(gwf, packagedata=[(0, beta_c, 0.0, "gwt", "concentration")])

    # Right boundary (GHB): fixed head with conductance and seawater concentration.
    ghbcond = hk_arr[:, -1] * delv * delc / (0.5 * delr)
    ghb_spd = [[(k, 0, ncol - 1), ghb_head, float(ghbcond[k]), cinlet] for k in range(nlay)]
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb_spd, pname="GHB-1", auxiliary="CONCENTRATION")

    # Left boundary (WEL): distributed freshwater inflow with zero salinity.
    wel_spd = [[(k, 0, 0), inflow / nlay, 0.0] for k in range(nlay)]
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd, pname="WEL-1", auxiliary="CONCENTRATION")

    # Save and print flow outputs.
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="gwf.hds",
        budget_filerecord="gwf.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    # Build the groundwater transport model.
    gwt = flopy.mf6.ModflowGwt(sim, modelname="gwt", save_flows=True)

    # Structured grid discretization for transport.
    flopy.mf6.ModflowGwtdis(
        gwt, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm
    )
    # Initial salt concentration distribution.
    flopy.mf6.ModflowGwtic(gwt, strt=conc0.reshape(nlay, 1, ncol))

    # Advection scheme for solute transport.
    flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")
    # Dispersion/diffusion settings for solute transport.
    flopy.mf6.ModflowGwtdsp(gwt, alh=al, ath1=at, xt3d_off=True, diffc=diffc)

    # Source/sink mixing: transfer auxiliary concentrations from boundary packages.
    sourcerecarray = [("GHB-1", "AUX", "CONCENTRATION"), ("WEL-1", "AUX", "CONCENTRATION")]
    flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray)

    # Mobile storage term for concentration (uses porosity).
    flopy.mf6.ModflowGwtmst(gwt, porosity=por)

    # Save and print transport outputs.
    flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord="gwt.ucn",
        budget_filerecord="gwt.cbc",
        saverecord=[("CONCENTRATION", "ALL")],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    )

    # Register separate linear solvers and explicitly couple GWF <-> GWT.
    sim.register_ims_package(ims_gwf, [gwf.name])
    sim.register_ims_package(ims_gwt, [gwt.name])
    flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea="gwf", exgmnameb="gwt")

    # Write input files and run the MODFLOW 6 simulation.
    sim.write_simulation()
    success, _ = sim.run_simulation(silent=False)
    if not success:
        raise RuntimeError("MODFLOW 6 failed")

    # Read binary outputs as full time series arrays.
    hobj = flopy.utils.HeadFile(ws / "gwf.hds")
    cobj = flopy.utils.HeadFile(ws / "gwt.ucn", text="CONCENTRATION")

    head_ts = hobj.get_alldata().squeeze()
    conc_ts = cobj.get_alldata().squeeze()
    times = np.asarray(hobj.get_times(), dtype=float)

    if return_timeseries:
        return head_ts, conc_ts, times

    return head_ts[-1], conc_ts[-1]
