"""MODFLOW 6 simulation for the simplified Henry problem (paper Sec. 5).

Implements the coupled elliptic-parabolic system:

    ∇·(κ/μ ∇p) = ∇·(κ/μ ρ g)                  (groundwater flow, elliptic)
    η ∂C/∂t = ∇·(κ/μ (∇p − ρg) C) + ∇·(η D ∇C)  (solute transport, parabolic)

with:
    - Zero specific storage (Ss = 0) — flow equation is quasi-static/elliptic
    - No boundary influx (no WEL, no GHB)
    - Concentration: homogeneous Dirichlet C = 0 on all of ∂Ω (CNC package)
    - Head (see ``head_bc``): h = 0 on the left and right walls with no-flow
      top/bottom (``'lr'``, used for the paper's dataset), or h = 0 on all of
      ∂Ω (``'all'``)
    - Linear equation of state: ρ(C) = ρ₀(1 + β_C · C)
    - Initial condition: C(x, 0) = C₀(x)  (spatially varying matrix)

The advective flux q := κ/μ (∇p − ρg) is divergence-free by construction
(∇·q = 0 is precisely the flow equation restated).
"""
import pathlib as pl

import flopy
import numpy as np

from .init_functions import sample_field_2d

HEAD_BCS = ("lr", "all")


def _to_layer_col_field(value, nlay, ncol, name):
    """Broadcast a scalar or validate an existing (nlay, ncol) array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((nlay, ncol), float(arr), dtype=float)
    if arr.shape != (nlay, ncol):
        raise ValueError(f"{name} must have shape ({nlay}, {ncol}), got {arr.shape}")
    return arr


def create_random_field(
    nlay: int = 20,
    ncol: int = 40,
    Lx: float = 2.0,
    Lz: float = 1.0,
    c_fresh: float = 0.0,
    c_sea: float = 35.0,
    len_scale: float = 0.1,
    var: float = 0.1,
    seed: int | None = None,
    taper: bool = True,
    period: float = 1.0,
) -> np.ndarray:
    """Sample a GRF initial concentration field on the cell centres.

    The field from ``sample_field_2d`` (non-negative, optionally tapered
    towards zero at the edges) is evaluated at the normalised cell centres.
    With ``taper=True`` the outer ring of cells — the cells that carry the
    C = 0 Dirichlet condition (CNC) in the simplified problem — is then set to
    exactly zero, so C₀ satisfies the boundary condition on the grid. Finally
    the field is min–max scaled to [c_fresh, c_sea].

    Returns
    -------
    field : np.ndarray of shape (nlay, ncol)
        Layer 0 is the top (z ≈ Lz), column 0 the left wall (x ≈ 0).
    """
    f = sample_field_2d(
        res=(ncol, nlay),
        len_scale=len_scale,
        var=var,
        seed=seed,
        taper=taper,
        period=period,
    )

    dx = Lx / ncol
    dz = Lz / nlay
    # Normalized cell center coordinates in [0, 1]
    x_norm = (np.arange(ncol, dtype=float) + 0.5) * (dx / Lx)
    # Layer 0 is top (z_norm ≈ 1), layer nlay-1 is bottom (z_norm ≈ 0)
    z_norm = 1.0 - (np.arange(nlay, dtype=float) + 0.5) * (dz / Lz)
    X_norm, Z_norm = np.meshgrid(x_norm, z_norm)

    field = f((X_norm.ravel(), Z_norm.ravel())).reshape((nlay, ncol))

    # Cell centres lie half a cell inside ∂Ω, so the taper alone leaves the
    # boundary cells small but nonzero; zero them to match the Dirichlet BC.
    if taper:
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0

    # Min-max normalize the field to [c_fresh, c_sea]
    f_min = float(np.min(field))
    f_max = float(np.max(field))
    if f_max > f_min:
        field_norm = (field - f_min) / (f_max - f_min)
    else:
        field_norm = np.zeros_like(field)

    return c_fresh + (c_sea - c_fresh) * field_norm


def build_and_run_simplified_henry(
    workspace,
    conc0,
    # Grid parameters
    ncol: int = 40,
    nlay: int = 20,
    Lx: float = 2.0,
    Lz: float = 1.0,
    # Time discretisation
    total_time: float = 1.0,
    nstp: int = 50,
    # Hydraulic parameters
    por: float = 0.35,
    hk: float = 50.0,    # hydraulic conductivity K = κ ρ₀ g / μ [m/d] (isotropic)
    # Dispersion parameters
    al: float = 0.0,     # longitudinal dispersivity [m]
    at: float = 0.0,     # transverse dispersivity [m]
    diffc: float = 0.003,    # effective molecular diffusion coefficient D_C [m²/d]
    # Density coupling
    beta_c: float = 7e-5,    # solutal expansion coefficient β_C [m³/kg]
    rho0: float = 1000.0,    # reference fluid density ρ₀ [kg/m³]
    # Boundary conditions
    head_bc: str = "lr",
    exe_name: str = "mf6",
):
    """Build and run the simplified Henry density-driven convection problem.

    The domain Ω = [0, Lx] × [0, Lz] is discretised on an nlay × ncol
    structured grid (1 row, so effectively 2-D). Constant-head (CHD) cells
    with h = 0 are placed on the boundary selected by ``head_bc``; constant-
    concentration (CNC) cells fix C = 0 on all four boundaries. There is no
    storage package (Ss = 0) and no inflow — motion arises purely from the
    buoyancy term in the BUY package.

    Parameters
    ----------
    workspace : str or Path
        Directory where MODFLOW 6 input/output files are written.
    conc0 : array_like of shape (nlay, ncol) or scalar
        Initial concentration field C₀ [kg/m³].
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
        Reference fluid density ρ₀ [kg/m³] (BUY reference density).
    head_bc : str
        ``'lr'``: h = 0 on the left and right walls, top and bottom no-flow
        (the setting used to generate the paper's dataset).
        ``'all'``: h = 0 on all four boundaries (Eq. (19) of the paper).
    exe_name : str or Path
        Name or path of the ``mf6`` executable.

    Returns
    -------
    head_ts : ndarray of shape (nstp + 1, nlay, ncol)  — includes t=0
        (head at t=0 is a zero placeholder; MODFLOW reports head from t=Δt)
    conc_ts : ndarray of shape (nstp + 1, nlay, ncol)  — includes t=0
    times   : ndarray of shape (nstp + 1,)  — times [days] starting at 0.0
    """
    if head_bc not in HEAD_BCS:
        raise ValueError(f"Unknown head_bc '{head_bc}'. Expected one of {HEAD_BCS}.")

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

    conc0_arr = _to_layer_col_field(conc0, nlay, ncol, "conc0")

    # -----------------------------------------------------------------------
    # MODFLOW 6 simulation container — single stress period, nstp uniform steps
    # -----------------------------------------------------------------------
    sim = flopy.mf6.MFSimulation(
        sim_name="simplified_henry", sim_ws=str(ws), exe_name=exe
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

    # Initial head = 0  (consistent with the Dirichlet head BC)
    flopy.mf6.ModflowGwfic(gwf, strt=np.zeros((nlay, nrow, ncol)))

    # Hydraulic conductivity — no storage package (Ss = 0 ↔ elliptic flow eq.)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,                        # confined
        k=hk,
        k33=hk,
        save_specific_discharge=True,
    )

    # Buoyancy coupling: ρ(C) = ρ₀(1 + β_C C) = ρ₀ + ρ₀ β_C C, so the
    # density slope required by the BUY package is dρ/dC = ρ₀ β_C.
    flopy.mf6.ModflowGwfbuy(
        gwf,
        denseref=rho0,
        packagedata=[(0, rho0 * beta_c, 0.0, "gwt", "concentration")],
    )

    # -------------------------------------------------------------------
    # Constant-Head (CHD) package, h = 0 on the boundary cells.
    #   Left column  : j = 0           (always)
    #   Right column : j = ncol-1      (always)
    #   Top layer    : k = 0           (head_bc='all' only)
    #   Bottom layer : k = nlay-1      (head_bc='all' only)
    # Boundaries without CHD cells are no-flow. A set avoids double-counting
    # corner cells. The transport BC C = 0 is imposed separately via CNC.
    # -------------------------------------------------------------------
    chd_cells = set()
    for k in range(nlay):
        chd_cells.add((k, 0, 0))          # left
        chd_cells.add((k, 0, ncol - 1))   # right
    if head_bc == "all":
        for j in range(ncol):
            chd_cells.add((0, 0, j))          # top
            chd_cells.add((nlay - 1, 0, j))   # bottom

    chd_spd = [(*cell, 0.0) for cell in sorted(chd_cells)]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd, pname="CHD-1")

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

    # Dispersion tensor D: isotropic molecular diffusion + mechanical dispersivity
    # (al = at = 0 gives D = D_C I, as in the paper).
    flopy.mf6.ModflowGwtdsp(gwt, alh=al, ath1=at, xt3d_off=True, diffc=diffc)

    # MF6 requires SSM when the flow model has boundary packages (CHD), even
    # though there are no active sources here.
    flopy.mf6.ModflowGwtssm(gwt, sources=None)

    # Constant-Concentration (CNC) package: C = 0 on all four boundaries.
    # Unlike SSM (which only acts where water enters), CNC fixes the boundary
    # concentration at every time step.
    cnc_cells = set()
    for j in range(ncol):
        cnc_cells.add((0, 0, j))          # top
        cnc_cells.add((nlay - 1, 0, j))   # bottom
    for k in range(nlay):
        cnc_cells.add((k, 0, 0))          # left
        cnc_cells.add((k, 0, ncol - 1))   # right
    cnc_spd = [(*cell, 0.0) for cell in sorted(cnc_cells)]
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

    head_ts = np.concatenate([np.zeros((1, nlay, ncol)), head_ts], axis=0)
    conc_ts = np.concatenate([conc0_arr[np.newaxis].astype(float), conc_ts], axis=0)
    times   = np.concatenate([[0.0], times], axis=0)

    return head_ts, conc_ts, times
