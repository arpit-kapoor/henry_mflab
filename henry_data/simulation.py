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
    total_time=30.0,
    nstp=240,
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
    ss=1e-3,
    sy=0.15,
    return_timeseries=False,
    exe_name="mf6",
    dynamic_inflow=True,
    dynamic_tides=True,
    add_storage=False,
    # Tidal forcing parameters (right / sea boundary)
    tidal_amplitude=0.15,      # Mean M2 tidal amplitude [m]  (was 0.5; real data ~0.05-0.35m)
    spring_neap_amp=0.10,      # Spring-neap modulation amplitude [m]  (was 0.3)
    tidal_period=0.517,        # M2 semi-diurnal period [days] (~12.42 h)
    spring_neap_period=14.77,  # Spring-neap beat period [days]
    spring_neap_phase=3.14159, # Phase offset [rad]; π → starts at neap (min amplitude)
    tidal_noise_std=0.01,      # Std of Gaussian noise added to tidal head [m]  (was 0.02)
    slr_rate=0.0,              # Sea-level rise rate [m/day]; 0 = no trend
    # Freshwater inflow parameters (left boundary)
    inflow_seasonal_amp=0.5,   # Seasonal half-sine amplitude factor (fraction of mean)
    inflow_event_amp=0.3,      # Short-term event amplitude factor (fraction of mean)
    inflow_event_period=7.0,   # Event recurrence period [days]
    inflow_trend_amp=0.0,      # Monotone trend amplitude (fraction of mean); negative = drying
):
    """Build and run a MODFLOW 6 Henry saltwater-intrusion problem.

    The right (sea) boundary uses a General-Head Boundary (GHB) whose head
    follows a realistic M2 semi-diurnal tidal signal modulated by a spring-neap
    envelope.  The left boundary uses a Well (WEL) package with a multi-scale
    freshwater inflow composed of a seasonal trend and episodic events.

    Parameters
    ----------
    total_time : float
        Total simulation duration in days (default 30).
    nstp : int
        Number of uniform time steps (default 240 → Δt = 0.125 day ≈ 3 h,
        resolving ~4 steps per semi-diurnal tidal cycle).
    tidal_amplitude : float
        Mean M2 tidal amplitude around ``ghb_head`` [m].
    spring_neap_amp : float
        Additional amplitude superimposed by the fortnightly spring-neap cycle [m].
    tidal_period : float
        Semi-diurnal tidal period [days]. Default 0.517 (≈ 12.42 h, M2 tide).
    spring_neap_period : float
        Spring-neap modulation period [days]. Default 14.77.
    spring_neap_phase : float
        Phase offset of the spring-neap envelope [radians].  Default π starts the
        simulation at neap tide (minimum amplitude), so the envelope grows to spring
        around t = T_sn/2 ≈ 7.4 days then returns to neap — matching real tidal
        records that start quiet and build to a spring-tide peak.  Use 0 to start
        at spring (maximum amplitude) as in the classic formulation.
    tidal_noise_std : float
        Standard deviation of Gaussian noise added to each tidal head node [m].
    slr_rate : float
        Sea-level rise rate [m/day] added as a linear drift to the mean tidal
        baseline.  0 (default) = no rise.  A value of 0.003 gives +0.09 m over
        30 days, clearly visible as a slow salt-wedge advance.
    inflow_seasonal_amp : float
        Amplitude of seasonal (half-sine) freshwater inflow variation as a
        fraction of the mean per-layer inflow.
    inflow_event_amp : float
        Amplitude of episodic rainfall events as a fraction of mean per-layer inflow.
    inflow_event_period : float
        Recurrence period of episodic rainfall events [days].
    inflow_trend_amp : float
        Monotone inflow trend expressed as a fraction of mean per-layer inflow.
        Negative = drying (inflow decreases linearly to ``inflow_trend_amp * q_mean``
        by end of run).  Positive = wetting.  0 (default) = no trend.
        Example: ``-0.4`` reduces inflow by 40 % of mean over the simulation.
    """
    ws = pl.Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    # flopy runs MF6 from the simulation workspace, so resolve path-like
    # executable values once to avoid workspace-relative lookup failures.
    exe = str(exe_name)
    exe_path = pl.Path(exe).expanduser()
    if exe_path.parent != pl.Path("."):
        exe = str(exe_path.resolve())

    nrow = 1
    delr = Lx / ncol
    delc = 1.0
    delv = Lz / nlay
    top = Lz
    if ghb_head is None:
        ghb_head = top
    botm = [Lz - delv * (k + 1) for k in range(nlay)]

    # Capture the scalar nstp before it is wrapped into a list for flopy.
    _nstp = int(nstp)
    perlen = [total_time]
    nper = 1
    nstp_list = [_nstp]
    tsmult = [1.0]

    # Create a MODFLOW 6 simulation container (workspace + executable).
    sim = flopy.mf6.MFSimulation(sim_name="henry", sim_ws=str(ws), exe_name=exe)

    # Define time discretization (single stress period split into nstp time steps).
    flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=nper, perioddata=list(zip(perlen, nstp_list, tsmult))
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

    # Storage package — makes the flow model truly transient.
    # iconvert=0 matches icelltype=0 (strictly confined); sy is defined but inactive.
    if add_storage:
        flopy.mf6.ModflowGwfsto(
            gwf,
            save_flows=True,
            iconvert=0,
            ss=ss,
            sy=sy,
            transient={0: True},
        )

    # Buoyancy coupling: maps concentration from GWT to density effects in GWF.
    flopy.mf6.ModflowGwfbuy(gwf, packagedata=[(0, beta_c, 0.0, "gwt", "concentration")])

    # Calculate conductance (remains constant regardless of tides)
    ghbcond = hk_arr[:, -1] * delv * delc / (0.5 * delr)

    if not dynamic_tides:
        # Original steady-state GHB: constant head and concentration.
        ghb_spd = [[(k, 0, ncol - 1), ghb_head, float(ghbcond[k]), cinlet] for k in range(nlay)]
        flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb_spd, pname="GHB-1", auxiliary="CONCENTRATION")
    else:
        # ---------------------------------------------------------------
        # Realistic tidal forcing on the right (sea) boundary.
        #
        # Time-series nodes: _nstp+1 points spanning [0, total_time].
        # Using one extra node ensures the MF6 time-series file covers the
        # final output time (total_time) without extrapolation gaps.
        # Node index 0  → t = 0       (start of stress period)
        # Node index k  → t = k*dt    (end of time step k)
        # ---------------------------------------------------------------
        t_forcing = np.linspace(0.0, total_time, _nstp + 1)

        # Composite tidal head signal:
        #   M2 semi-diurnal carrier modulated by a fortnightly spring-neap envelope,
        #   superimposed on a linearly rising mean sea level (sea-level rise).
        #   Amplitude ranges from (tidal_amplitude - spring_neap_amp) at neap tide
        #   to (tidal_amplitude + spring_neap_amp) at spring tide.
        mean_sl   = ghb_head + slr_rate * t_forcing
        raw_tidal = (
            mean_sl
            + (tidal_amplitude + spring_neap_amp * np.cos(2.0 * np.pi * t_forcing / spring_neap_period + spring_neap_phase))
            * np.cos(2.0 * np.pi * t_forcing / tidal_period)
            + np.random.normal(0.0, tidal_noise_std, len(t_forcing))
        )
        tidal_heads = np.clip(raw_tidal, 0.0, top)

        # Per-layer concentration time series:
        #   35 kg/m³ when tidal head reaches or exceeds the layer top (submerged),
        #   0 otherwise (layer exposed to fresh water).
        c_names = [f"c_lay{k}" for k in range(nlay)]

        ts_data = []
        for i, t in enumerate(t_forcing):
            h_t = tidal_heads[i]
            row = [t, h_t]
            for k in range(nlay):
                lay_top = top if k == 0 else botm[k - 1]
                row.append(35.0 if h_t >= lay_top else 0.0)
            ts_data.append(tuple(row))

        # cinlet_raw: shape (_nstp+1, nlay), one row per forcing node.
        cinlet_raw = np.array([row[2:] for row in ts_data])

        # Each layer references its own concentration name; head is shared.
        ghb_spd = {
            0: [
                [(k, 0, ncol - 1), "tide_head", float(ghbcond[k]), f"c_lay{k}"]
                for k in range(nlay)
            ]
        }

        ghb = flopy.mf6.ModflowGwfghb(
            gwf,
            stress_period_data=ghb_spd,
            pname="GHB-1",
            auxiliary="CONCENTRATION",
        )

        # Multi-column time series: first column is tide_head, remaining are
        # per-layer concentrations.  linearend interpolation preserves the
        # smooth sinusoidal shape of the tidal signal.
        ghb.ts.initialize(
            filename="ghb_ts.ts",
            timeseries=ts_data,
            time_series_namerecord=["tide_head"] + c_names,
            interpolation_methodrecord=["linearend"] * (nlay + 1),
        )

    if not dynamic_inflow:
        # Left boundary (WEL): constant distributed freshwater inflow, zero salinity.
        wel_spd = [[(k, 0, 0), inflow / nlay, 0.0] for k in range(nlay)]
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd, pname="WEL-1", auxiliary="CONCENTRATION")
    else:
        # ---------------------------------------------------------------
        # Multi-scale freshwater inflow on the left boundary.
        #
        # Two superimposed components:
        #   1. Seasonal: half-sine ramp over the full simulation window,
        #      mimicking a wet → dry → wet seasonal cycle.
        #   2. Episodic events: absolute-sine pulses at a weekly recurrence,
        #      representing individual rainfall/runoff events.
        # ---------------------------------------------------------------
        t_forcing_q = np.linspace(0.0, total_time, _nstp + 1)
        q_mean = inflow / nlay

        # Three superimposed components:
        #   1. Seasonal: symmetric half-sine ramp (wet→dry→wet), zero net drift.
        #   2. Episodic events: absolute-sine pulses at weekly recurrence.
        #   3. Monotone trend: linear drift from 0 to inflow_trend_amp × q_mean.
        #      Negative value = sustained drying (salt wedge advances long-term).
        q_seasonal = q_mean * (1.0 + inflow_seasonal_amp * np.sin(np.pi * t_forcing_q / total_time))
        q_event    = q_mean * inflow_event_amp * np.abs(np.sin(np.pi * t_forcing_q / inflow_event_period))
        q_drift    = q_mean * inflow_trend_amp * (t_forcing_q / total_time)   # 0 at t=0, full at t=T
        q_in_series = (
            q_seasonal
            + q_event
            + q_drift
            + np.random.normal(0.0, 0.005 * q_mean, len(t_forcing_q))
        )
        # Enforce a positive minimum inflow so the well never becomes an abstraction.
        q_in_series = np.clip(q_in_series, 0.01 * q_mean, None)

        ts_data_q = list(zip(t_forcing_q.tolist(), q_in_series.tolist()))

        # Stress period data references the "Q_in" time-series name.
        wel_spd = {0: [[(k, 0, 0), "Q_in", 0.0] for k in range(nlay)]}
        wel = flopy.mf6.ModflowGwfwel(
            gwf,
            stress_period_data=wel_spd,
            pname="WEL-1",
            auxiliary="CONCENTRATION",
        )

        # linearend gives a smooth, realistic inflow hydrograph.
        wel.ts.initialize(
            filename="wel_ts.ts",
            timeseries=ts_data_q,
            time_series_namerecord="Q_in",
            interpolation_methodrecord="linearend",
        )

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
    n_out = len(times)  # number of MF6 output records (= _nstp)

    # Align forcing arrays to MF6 output times.
    # Forcing nodes 0..n_out cover t=0 to t=total_time; MF6 outputs correspond
    # to the END of each time step (nodes 1..n_out), so we skip node 0.
    if not dynamic_tides:
        # Broadcast scalar cinlet to every simulated time step and layer.
        cinlet_ts = np.full((n_out, nlay), float(cinlet))  # shape (n_out, nlay)
    else:
        # Sub-sample from the (_nstp+1)-node forcing arrays to n_out MF6 records.
        cinlet_ts = cinlet_raw[1 : n_out + 1]   # shape (n_out, nlay)

    # Read the cell-by-cell budget file to extract boundary flow rates.
    bobj = flopy.utils.CellBudgetFile(ws / "gwf.cbc")

    # Total well flux summed across all cells (positive = inflow).
    q_in_ts = np.array([np.sum(stepdata["q"]) for stepdata in bobj.get_data(text="WEL")])

    if dynamic_tides:
        # Return the tidal head at MF6 output times as the right-boundary feature.
        q_ghb_ts = tidal_heads[1 : n_out + 1]   # shape (n_out,)
    else:
        q_ghb_ts = np.array([np.sum(stepdata["q"]) for stepdata in bobj.get_data(text="GHB")])

    if return_timeseries:
        return head_ts, conc_ts, q_in_ts, q_ghb_ts, cinlet_ts, times

    return head_ts[-1], conc_ts[-1], q_in_ts[-1], q_ghb_ts[-1], cinlet_ts[-1]
