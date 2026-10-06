"""MODFLOW 6 simulation for the classical Henry-type problem (paper App. I).

Uses the classical Henry boundary conditions of the MODFLOW 6 example
``ex-gwt-henry`` so that a saltwater wedge forms:

    - Left boundary  (x = 0)  : WEL, constant freshwater inflow Q/nlay per cell, C = 0
    - Right boundary (x = Lx) : GHB, head = Lz, inflowing water carries C = c_sea
    - Top / bottom            : no-flow
    - Zero specific storage (no STO) — flow equation is quasi-static/elliptic
    - Linear equation of state: ρ(C) = ρ₀(1 + β_C · C)
    - Initial condition: C(x, 0) = C₀(x)  (spatially varying matrix)

Unlike ``simplified_henry`` there is no CNC package: concentration at the sea
boundary is set by SSM, i.e. c_sea where water enters and the computed
concentration where it leaves.
"""
import pathlib as pl

import flopy
import numpy as np

from simplified_henry.simulation import _to_layer_col_field


def build_and_run_classical_henry(
    workspace,
    conc0,
    inflow: float = 5.7024,
    # Grid parameters
    ncol: int = 80,
    nlay: int = 40,
    Lx: float = 2.0,
    Lz: float = 1.0,
    # Time discretisation
    total_time: float = 0.5,
    nstp: int = 500,
    # Hydraulic parameters
    por: float = 0.35,
    hk: float = 864.0,   # isotropic hydraulic conductivity [m/d]
    # Dispersion parameters
    al: float = 0.0,     # longitudinal dispersivity [m]
    at: float = 0.0,     # transverse dispersivity [m]
    diffc: float = 0.57024,  # effective molecular diffusion coefficient [m²/d]
    # Density coupling
    beta_c: float = 0.0007,  # solutal expansion coefficient β_C [m³/kg]
    rho0: float = 1000.0,    # reference fluid density ρ₀ [kg/m³]
    # Sea boundary
    c_sea: float = 35.0,     # seawater concentration [kg/m³]
    ghb_head: float | None = None,  # sea-level head [m]; defaults to Lz
    exe_name: str = "mf6",
):
    """Build and run the classical Henry-type saltwater-intrusion problem.

    Parameters
    ----------
    workspace : str or Path
        Directory where MODFLOW 6 input/output files are written.
    conc0 : array_like of shape (nlay, ncol) or scalar
        Initial concentration field C₀ [kg/m³].
    inflow : float
        Total freshwater inflow through the left boundary [m³/d], split evenly
        over the nlay left-column cells. Classic Henry: 5.7024; low-inflow
        variant: 2.851.
    ncol, nlay : int
        Number of columns and layers (rows is fixed at 1).
    Lx, Lz : float
        Horizontal and vertical domain extents [m].
    total_time : float
        Simulation duration [days].
    nstp : int
        Number of uniform time steps.
    por : float
        Porosity η ∈ (0, 1).
    hk : float
        Isotropic hydraulic conductivity [m/d].
    al, at : float
        Longitudinal and transverse dispersivity [m].
    diffc : float
        Effective molecular diffusion coefficient [m²/d].
    beta_c : float
        Solutal expansion coefficient β_C [m³/kg] in ρ(C) = ρ₀(1 + β_C C).
    rho0 : float
        Reference fluid density ρ₀ [kg/m³].
    c_sea : float
        Concentration of water entering through the sea boundary [kg/m³].
    ghb_head : float or None
        Sea-level (GHB) head [m]. Defaults to the model top ``Lz``.
    exe_name : str or Path
        Name or path of the ``mf6`` executable.

    Returns
    -------
    head_ts : ndarray of shape (nstp + 1, nlay, ncol)  — includes t=0
    conc_ts : ndarray of shape (nstp + 1, nlay, ncol)  — includes t=0
    times   : ndarray of shape (nstp + 1,)  — times [days] starting at 0.0
    """
    ws = pl.Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    # Resolve executable path once (avoids workspace-relative lookup failures).
    exe = str(exe_name)
    exe_path = pl.Path(exe).expanduser()
    if exe_path.parent != pl.Path("."):
        exe = str(exe_path.resolve())

    # -----------------------------------------------------------------------
    # Grid geometry
    # -----------------------------------------------------------------------
    nrow = 1
    delr = Lx / ncol   # column width  [m]
    delc = 1.0          # row width (unit depth in 2-D) [m]
    delv = Lz / nlay   # layer thickness [m]
    top = Lz
    botm = [Lz - delv * (k + 1) for k in range(nlay)]
    if ghb_head is None:
        ghb_head = top

    conc0_arr = _to_layer_col_field(conc0, nlay, ncol, "conc0")
    hk_arr    = _to_layer_col_field(hk, nlay, ncol, "hk")

    # -----------------------------------------------------------------------
    # MODFLOW 6 simulation container — single stress period, nstp uniform steps
    # -----------------------------------------------------------------------
    sim = flopy.mf6.MFSimulation(
        sim_name="classical_henry", sim_ws=str(ws), exe_name=exe
    )

    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=1,
        perioddata=[(total_time, int(nstp), 1.0)],
    )

    # -----------------------------------------------------------------------
    # Iterative solvers — shared tight tolerances for accuracy
    # -----------------------------------------------------------------------
    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-10, 1e-6, 0.97

    def _ims(filename):
        return flopy.mf6.ModflowIms(
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
            filename=filename,
        )

    ims_gwf = _ims("gwf.ims")
    ims_gwt = _ims("gwt.ims")

    # -----------------------------------------------------------------------
    # Groundwater Flow Model (GWF)
    # -----------------------------------------------------------------------
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)

    flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=nrow, ncol=ncol,
        delr=delr, delc=delc, top=top, botm=botm,
    )

    # Initial head = sea level (only an initial guess: no storage)
    flopy.mf6.ModflowGwfic(gwf, strt=np.full((nlay, nrow, ncol), ghb_head))

    # Hydraulic conductivity — no storage package (Ss = 0 ↔ elliptic flow eq.)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,                        # confined
        k=hk_arr.reshape(nlay, nrow, ncol),
        k33=hk_arr.reshape(nlay, nrow, ncol),
        save_specific_discharge=True,
    )

    # Buoyancy coupling: ρ(C) = ρ₀(1 + β_C C)  ⇒  dρ/dC = ρ₀ β_C
    # (classic Henry: ρ₀ β_C = 0.7, seawater density 1024.5 kg/m³).
    flopy.mf6.ModflowGwfbuy(
        gwf,
        denseref=rho0,
        packagedata=[(0, rho0 * beta_c, 0.0, "gwt", "concentration")],
    )

    # Sea boundary (right column): hydrostatic sea level, seawater on inflow.
    ghbcond = hk_arr[:, -1] * delv * delc / (0.5 * delr)
    ghb_spd = [
        [(k, 0, ncol - 1), ghb_head, float(ghbcond[k]), c_sea]
        for k in range(nlay)
    ]
    flopy.mf6.ModflowGwfghb(
        gwf, stress_period_data=ghb_spd, pname="GHB-1", auxiliary="CONCENTRATION"
    )

    # Inland boundary (left column): constant freshwater inflow, split evenly.
    wel_spd = [[(k, 0, 0), inflow / nlay, 0.0] for k in range(nlay)]
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=wel_spd, pname="WEL-1", auxiliary="CONCENTRATION"
    )

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="gwf.hds",
        budget_filerecord="gwf.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    # -----------------------------------------------------------------------
    # Groundwater Transport Model (GWT)
    # -----------------------------------------------------------------------
    gwt = flopy.mf6.ModflowGwt(sim, modelname="gwt", save_flows=True)

    flopy.mf6.ModflowGwtdis(
        gwt, nlay=nlay, nrow=nrow, ncol=ncol,
        delr=delr, delc=delc, top=top, botm=botm,
    )

    flopy.mf6.ModflowGwtic(gwt, strt=conc0_arr.reshape(nlay, nrow, ncol))
    flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")
    flopy.mf6.ModflowGwtdsp(gwt, alh=al, ath1=at, xt3d_off=True, diffc=diffc)

    # Boundary concentrations come from the GHB / WEL auxiliary variables.
    flopy.mf6.ModflowGwtssm(
        gwt,
        sources=[("GHB-1", "AUX", "CONCENTRATION"), ("WEL-1", "AUX", "CONCENTRATION")],
    )
    flopy.mf6.ModflowGwtmst(gwt, porosity=por)

    flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord="gwt.ucn",
        budget_filerecord="gwt.cbc",
        saverecord=[("CONCENTRATION", "ALL")],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    )

    # -----------------------------------------------------------------------
    # Register solvers and couple GWF ↔ GWT
    # -----------------------------------------------------------------------
    sim.register_ims_package(ims_gwf, [gwf.name])
    sim.register_ims_package(ims_gwt, [gwt.name])
    flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea="gwf", exgmnameb="gwt")

    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True, report=False)
    if not success:
        raise RuntimeError("MODFLOW 6 failed — check the listing file in: " + str(ws))

    # -----------------------------------------------------------------------
    # Read binary outputs and prepend the t = 0 state
    # -----------------------------------------------------------------------
    hobj = flopy.utils.HeadFile(ws / "gwf.hds")
    cobj = flopy.utils.HeadFile(ws / "gwt.ucn", text="CONCENTRATION")

    head_ts = hobj.get_alldata()[:, :, 0, :]   # (nstp, nlay, ncol)
    conc_ts = cobj.get_alldata()[:, :, 0, :]   # (nstp, nlay, ncol)
    times   = np.asarray(hobj.get_times(), dtype=float)

    head_ts = np.concatenate([np.full((1, nlay, ncol), ghb_head), head_ts], axis=0)
    conc_ts = np.concatenate([conc0_arr[np.newaxis].astype(float), conc_ts], axis=0)
    times   = np.concatenate([[0.0], times], axis=0)

    return head_ts, conc_ts, times
