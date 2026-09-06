"""MODFLOW 6 simulation for the simplified Henry problem.

Implements the coupled elliptic-parabolic system:

    ∇·(κ/μ ∇p) = ∇·(κ/μ ρ g)                  (groundwater flow, elliptic)
    η ∂C/∂t = ∇·(κ/μ (∇p − ρg) C) + ∇·(η D ∇C)  (solute transport, parabolic)

with:
    - Zero specific storage (Ss = 0) — flow equation is quasi-static/elliptic
    - No boundary influx (no WEL, no GHB)
    - Dirichlet BCs: p = 0 on all of ∂Ω; C = 0 on top/left (groundwater) and C = 35 kg/m³ on right/bottom (seawater)
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


def create_s_shaped_wedge(
    nlay: int = 40,
    ncol: int = 80,
    Lx: float = 2.0,
    Lz: float = 1.0,
    c_fresh: float = 0.0,
    c_sea: float = 35.0,
    x_toe: float | None = None,
    x_top: float | None = None,
    trans_width: float | None = None,
) -> np.ndarray:
    """Create an S-shaped saltwater wedge initial concentration field.

    Simulates the intrusion of seawater (high concentration) from the right
    boundary (x = Lx) into freshwater (low concentration) on the left (x = 0),
    forming a wedge that penetrates inland along the aquifer base (z = 0) and
    recedes toward the coast near the aquifer surface (z = Lz).

    The interface position x_int(z) follows a smooth S-curve (cubic smoothstep)
    from x_toe at the bottom (z = 0) to x_top at the top (z = Lz).
    The concentration across the interface transitions smoothly between
    c_fresh and c_sea using a sigmoidal profile (logistic S-curve) with
    characteristic transition width `trans_width`.

    Parameters
    ----------
    nlay : int, default=40
        Number of layers (vertical discretization).
    ncol : int, default=80
        Number of columns (horizontal discretization).
    Lx : float, default=2.0
        Horizontal domain length [m].
    Lz : float, default=1.0
        Vertical domain length [m].
    c_fresh : float, default=0.0
        Freshwater solute concentration [kg/m³].
    c_sea : float, default=35.0
        Seawater solute concentration [kg/m³].
    x_toe : float or None, default=None
        Horizontal position of the wedge toe at the base (z = 0) [m].
        Defaults to 0.4 * Lx.
    x_top : float or None, default=None
        Horizontal position of the wedge interface at the top (z = Lz) [m].
        Defaults to 0.8 * Lx.
    trans_width : float or None, default=None
        Characteristic transition width (mixing zone thickness) [m].
        Defaults to 0.05 * Lx.

    Returns
    -------
    conc : np.ndarray of shape (nlay, ncol)
        2-D concentration field [kg/m³], where layer 0 corresponds to the top
        (z ≈ Lz) and layer nlay-1 corresponds to the bottom (z ≈ 0).
    """
    if x_toe is None:
        x_toe = 0.4 * Lx
    if x_top is None:
        x_top = 1.0 * Lx
    if trans_width is None:
        trans_width = 0.05 * Lx

    dx = Lx / ncol
    dz = Lz / nlay
    # Cell center coordinates
    x = (np.arange(ncol, dtype=float) + 0.5) * dx
    # Layer 0 is top (z ≈ Lz), layer nlay-1 is bottom (z ≈ 0)
    z = Lz - (np.arange(nlay, dtype=float) + 0.5) * dz

    # Normalized vertical coordinate: 0 at base, 1 at surface
    z_norm = np.clip(z / Lz, 0.0, 1.0)

    # Smooth S-curve (smoothstep) for interface position as a function of depth
    s_z = 3.0 * z_norm**2 - 2.0 * z_norm**3
    x_int = x_toe + (x_top - x_toe) * s_z

    # Sigmoidal S-shaped transition across the interface in the horizontal direction
    # Freshwater (c_fresh) on left (x << x_int), seawater (c_sea) on right (x >> x_int)
    diff = (x[None, :] - x_int[:, None]) / trans_width
    diff = np.clip(diff, -50.0, 50.0)
    sigmoid = 1.0 / (1.0 + np.exp(-diff))

    return c_fresh + (c_sea - c_fresh) * sigmoid


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
    # Initial concentration profile
    c0_x_toe: float = None,
    c0_x_top: float = None,
    c0_trans_width: float = None,
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
    structured grid (1 row, so effectively 2-D). All four boundaries carry
    constant-head (CHD) cells with head = 0 (encoding p|∂Ω = 0).
    The transport model imposes Dirichlet boundary conditions via CNC cells:
    C = 0 kg/m³ on the top and left boundaries (freshwater/groundwater), and
    C = 35 kg/m³ on the right and bottom boundaries (seawater interface).
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
    c0_x_toe, c0_x_top, c0_trans_width : float
        Parameters controlling the initial concentration profile C₀(x, z):
        - c0_x_toe : horizontal position of the wedge toe at the base (z = 0) [m]
        - c0_x_top : horizontal position of the wedge interface at the top (z = Lz) [m]
        - c0_trans_width : characteristic transition width (mixing zone thickness) [m]
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
    conc0_arr = create_s_shaped_wedge(nlay=nlay, ncol=ncol, Lx=Lx, Lz=Lz,
                                      x_toe=c0_x_toe, x_top=c0_x_top,
                                      trans_width=c0_trans_width)
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
    # Constant-Concentration (CNC) package:
    #   - Left (j = 0) and top (k = 0) boundaries: C = 0 kg/m³ (fresh groundwater)
    #   - Right (j = ncol - 1) and bottom (k = nlay - 1) boundaries: C = 35 kg/m³ (seawater interface)
    #
    # This is the GWT analogue of CHD for head. Unlike SSM (which only
    # activates when there is inflow at a stress boundary), CNC directly
    # fixes the concentration at the specified cells at every time step,
    # correctly imposing the Dirichlet boundary conditions.
    # -------------------------------------------------------------------
    c_sea_bc = float(np.max(conc0_arr)) if conc0_arr is not None else 35.0
    c_fresh_bc = 0.0

    cnc_dict = {}
    # Top and left boundaries: freshwater / inland (C = 0)
    for j in range(ncol):
        cnc_dict[(0, 0, j)] = c_fresh_bc
    for k in range(nlay):
        cnc_dict[(k, 0, 0)] = c_fresh_bc

    # Right-hand and bottom boundaries: seawater interface (C = 35 kg/m³)
    for k in range(nlay):
        cnc_dict[(k, 0, ncol - 1)] = c_sea_bc
    for j in range(ncol):
        cnc_dict[(nlay - 1, 0, j)] = c_sea_bc

    cnc_spd = [(*cell, conc) for cell, conc in sorted(cnc_dict.items())]
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
