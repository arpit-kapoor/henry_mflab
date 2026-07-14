"""MODFLOW 6 simulation for the simplified Henry problem.

Implements the coupled elliptic-parabolic system:

    ∇·(κ/μ ∇p) = ∇·(κ/μ ρ g)                  (groundwater flow, elliptic)
    η ∂C/∂t = ∇·(κ/μ (∇p − ρg) C) + ∇·(η D ∇C)  (solute transport, parabolic)

with:
    - Zero specific storage (Ss = 0) — flow equation is quasi-static/elliptic
    - No boundary influx (no WEL, no GHB)
    - Homogeneous Dirichlet BCs: p = 0 and C = 0 on all of ∂Ω
    - Linear equation of state: ρ(C) = ρ₀(1 + β_C · C)
    - Initial condition: C(x, 0) = C₀(x)  (spatially varying matrix)

The advective flux q := κ/μ (∇p − ρg) is divergence-free by construction
(∇·q = 0 is precisely the flow equation restated).
"""
import pathlib as pl

import flopy
import numpy as np


def _to_layer_col_field(value, nlay, ncol, name):
    """Broadcast a scalar or validate an existing (nlay, ncol) array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((nlay, ncol), float(arr), dtype=float)
    if arr.shape != (nlay, ncol):
        raise ValueError(f"{name} must have shape ({nlay}, {ncol}), got {arr.shape}")
    return arr


def build_and_run_simple_henry(
    workspace,
    # Grid parameters
    ncol: int = 80,
    nlay: int = 40,
    Lx: float = 2.0,
    Lz: float = 1.0,
    # Time discretisation
    total_time: float = 30.0,
    nstp: int = 240,
    # Initial condition — scalar or (nlay, ncol) array; uniform 35 kg/m³ by default
    C0=35.0,
    # Hydraulic parameters
    por: float = 0.35,
    hk: float = 864.0,   # horizontal hydraulic conductivity [m/d]  (= κ/μ proxy)
    vk: float = 864.0,   # vertical hydraulic conductivity [m/d]
    # Dispersion parameters (same structure as henry_data.simulation)
    al: float = 0.0,     # longitudinal dispersivity [m]
    at: float = 0.0,     # transverse dispersivity [m]
    diffc: float = 0.57024,  # effective molecular diffusion coefficient [m²/d]
    # Density coupling
    beta_c: float = 0.7,     # solutal expansion coefficient β_C [m³/kg]
    rho0: float = 1000.0,    # reference fluid density ρ₀ [kg/m³]  (MF6 BUY default)
    # Optional spatially varying K fields (override scalar hk/vk)
    hk_field=None,
    vk_field=None,
    # Output control
    return_timeseries: bool = False,
    exe_name: str = "mf6",
):
    """Build and run the simplified Henry density-driven convection problem.

    The domain Ω = [0, Lx] × [0, Lz] is discretised on an nlay × ncol
    structured grid (1 row, so effectively 2-D).  All four boundaries carry
    constant-head (CHD) cells with head = 0 and auxiliary concentration = 0,
    encoding the homogeneous Dirichlet conditions p|∂Ω = 0, C|∂Ω = 0.
    There is no storage package (Ss = 0), no well inflow, and no GHB tidal
    forcing — motion arises purely from the buoyancy term in the BUY package.

    Parameters
    ----------
    workspace : str or Path
        Directory where MODFLOW 6 input/output files are written.
    ncol, nlay : int
        Number of columns and layers (rows is fixed at 1).
    Lx, Lz : float
        Horizontal and vertical domain extents [m].
    total_time : float
        Simulation duration [days].
    nstp : int
        Number of uniform time steps.
    C0 : float or array_like of shape (nlay, ncol)
        Initial solute concentration field [kg/m³].  A scalar value is
        broadcast to fill the whole domain; pass a 2-D array for spatially
        varying initial conditions.
    por : float
        Porosity η ∈ (0, 1).
    hk, vk : float
        Horizontal and vertical hydraulic conductivity [m/d].
    al, at : float
        Longitudinal and transverse dispersivity [m].
    diffc : float
        Effective molecular diffusion coefficient [m²/d].
    beta_c : float
        Solutal expansion coefficient β_C [m³/kg] in ρ(C) = ρ₀(1 + β_C C).
    rho0 : float
        Reference fluid density ρ₀ [kg/m³].  Passed to the BUY package as
        the reference density.
    hk_field, vk_field : array_like of shape (nlay, ncol) or None
        Spatially varying K fields.  When provided they override the scalar
        ``hk`` / ``vk`` values.
    return_timeseries : bool
        If True, return full time-series arrays (head_ts, conc_ts, times).
        If False, return only the final-step arrays (head_final, conc_final).
    exe_name : str or Path
        Name or path of the ``mf6`` executable.

    Returns
    -------
    If ``return_timeseries=False`` (default):
        head_final : ndarray of shape (nlay, ncol)
        conc_final : ndarray of shape (nlay, ncol)
    If ``return_timeseries=True``:
        head_ts  : ndarray of shape (nstp, nlay, ncol)
        conc_ts  : ndarray of shape (nstp, nlay, ncol)
        times    : ndarray of shape (nstp,)  — end-of-step times [days]
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

    # -----------------------------------------------------------------------
    # Time discretisation: single stress period, nstp uniform steps.
    # -----------------------------------------------------------------------
    perlen = [total_time]
    nper = 1
    nstp_list = [int(nstp)]
    tsmult = [1.0]

    # -----------------------------------------------------------------------
    # Validate / broadcast spatially varying fields
    # -----------------------------------------------------------------------
    conc0_arr = _to_layer_col_field(C0, nlay, ncol, "C0")
    hk_arr    = _to_layer_col_field(hk if hk_field is None else hk_field, nlay, ncol, "hk_field")
    vk_arr    = _to_layer_col_field(vk if vk_field is None else vk_field, nlay, ncol, "vk_field")

    # -----------------------------------------------------------------------
    # MODFLOW 6 simulation container
    # -----------------------------------------------------------------------
    sim = flopy.mf6.MFSimulation(
        sim_name="simple_henry", sim_ws=str(ws), exe_name=exe
    )

    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=nper,
        perioddata=list(zip(perlen, nstp_list, tsmult)),
    )

    # -----------------------------------------------------------------------
    # Iterative solvers — shared tight tolerances for accuracy
    # -----------------------------------------------------------------------
    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-10, 1e-6, 0.97

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

    # -----------------------------------------------------------------------
    # Groundwater Flow Model (GWF)
    # -----------------------------------------------------------------------
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)

    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    # Initial head = 0  (consistent with Dirichlet p|∂Ω = 0)
    flopy.mf6.ModflowGwfic(gwf, strt=np.zeros((nlay, nrow, ncol)))

    # Hydraulic conductivity — no storage package (Ss = 0 ↔ elliptic flow eq.)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,                        # confined
        k=hk_arr.reshape(nlay, nrow, ncol),
        k33=vk_arr.reshape(nlay, nrow, ncol),
        save_specific_discharge=True,
    )

    # Buoyancy coupling: maps GWT concentration to density effects in GWF.
    # DRHODC = dρ/dC = beta_c  (same convention as the original Henry simulation;
    # for C in kg/m³ this gives ρ(35) ≈ 1000 + 0.7×35 = 1024.5 kg/m³ ✓).
    # Do NOT multiply by rho0 — the BUY DRHODC column is already dρ/dC directly.
    flopy.mf6.ModflowGwfbuy(
        gwf,
        packagedata=[(0, beta_c, 0.0, "gwt", "concentration")],
    )

    # -------------------------------------------------------------------
    # Constant-Head (CHD) package — encodes p = 0 on all of ∂Ω.
    #
    # Boundary layout (structured, 1-row grid):
    #   Left column  : j = 0
    #   Right column : j = ncol-1
    #   Top layer    : k = 0     (layer index 0 is the topmost in MF6)
    #   Bottom layer : k = nlay-1
    #
    # We use a set to avoid double-counting corner cells.
    # Note: NO auxiliary concentration on CHD. The transport Dirichlet BC
    # C = 0 is imposed separately via ModflowGwtcnc (see GWT section below).
    # SSM+AUX only works when there is inflow at the boundary; with uniform
    # initial density and zero head BCs there is no inflow, so SSM never fires.
    # -------------------------------------------------------------------
    chd_cells = set()

    # Left and right columns (all layers)
    for k in range(nlay):
        chd_cells.add((k, 0, 0))          # left
        chd_cells.add((k, 0, ncol - 1))   # right

    # Top and bottom layers (all columns, corners already covered above)
    for j in range(ncol):
        chd_cells.add((0, 0, j))           # top
        chd_cells.add((nlay - 1, 0, j))    # bottom

    # CHD stress period data: (cellid, head)  — no auxiliary concentration
    chd_spd = [(*cell, 0.0) for cell in sorted(chd_cells)]

    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chd_spd,
        pname="CHD-1",
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
        gwt,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    # Initial concentration field C₀ (scalar or spatially varying)
    flopy.mf6.ModflowGwtic(gwt, strt=conc0_arr.reshape(nlay, nrow, ncol))

    # Advection scheme
    flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")

    # Dispersion tensor D: isotropic molecular diffusion + mechanical dispersivity.
    # alh = longitudinal, ath1 = transverse horizontal dispersivity.
    # diffc = effective diffusion coefficient [m²/d].
    flopy.mf6.ModflowGwtdsp(gwt, alh=al, ath1=at, xt3d_off=True, diffc=diffc)

    # Source/sink mixing: MF6 requires SSM when the flow model has boundary
    # packages (CHD), even though SSM has no active sources here.  The actual
    # transport Dirichlet BC (C = 0 on ∂Ω) is enforced by CNC below.
    flopy.mf6.ModflowGwtssm(gwt, sources=None)

    # -------------------------------------------------------------------
    # Constant-Concentration (CNC) package — encodes C = 0 on all of ∂Ω.
    #
    # This is the GWT analogue of CHD for head.  Unlike SSM (which only
    # activates when there is inflow at a stress boundary), CNC directly
    # fixes the concentration at the specified cells at every time step,
    # correctly imposing the homogeneous Dirichlet condition C|∂Ω = 0
    # regardless of the local Darcy velocity.
    # -------------------------------------------------------------------
    cnc_spd = [(*cell, 0.0) for cell in sorted(chd_cells)]
    flopy.mf6.ModflowGwtcnc(gwt, stress_period_data=cnc_spd, pname="CNC-1")

    # Mobile storage term for concentration (uses porosity η).
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

    # -----------------------------------------------------------------------
    # Write inputs and run MODFLOW 6
    # -----------------------------------------------------------------------
    sim.write_simulation()
    success, _ = sim.run_simulation(silent=False)
    if not success:
        raise RuntimeError("MODFLOW 6 failed — check the listing file in: " + str(ws))

    # -----------------------------------------------------------------------
    # Read binary outputs
    # -----------------------------------------------------------------------
    hobj = flopy.utils.HeadFile(ws / "gwf.hds")
    cobj = flopy.utils.HeadFile(ws / "gwt.ucn", text="CONCENTRATION")

    head_ts = hobj.get_alldata().squeeze()   # shape: (nstp, nlay, ncol)
    conc_ts = cobj.get_alldata().squeeze()   # shape: (nstp, nlay, ncol)
    times   = np.asarray(hobj.get_times(), dtype=float)  # shape: (nstp,)

    if return_timeseries:
        return head_ts, conc_ts, times

    return head_ts[-1], conc_ts[-1]
