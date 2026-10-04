"""Dataset generator for the simplified Henry problem.

Generates 3-D (space × time) (input, output) training tensors from MODFLOW 6
time-series output of the simplified density-driven convection problem.

Each simulation **run** produces one data point:
    input_tensor  : float32, shape (4, T_in,  nlay, ncol)
    output_tensor : float32, shape (2, T_out, nlay, ncol)

where the MODFLOW time-series is first downsampled by ``skip`` (applied as
``[::skip]`` on the raw T_raw = nstp + 1 series including t = 0), then split at
``t_split``:
    T     = len(range(0, T_raw, skip))   (downsampled length)
    T_in  = t_split
    T_out = T - t_split

Input channels (cin = 4):
    ch 0 — concentration_t : solute concentration field over [0, t_split)
    ch 1 — head_t          : hydraulic head field over [0, t_split)
    ch 2 — beta_c          : solutal expansion coefficient (broadcast over time)
    ch 3 — diffc           : effective diffusion coefficient (broadcast over time)

Output channels (cout = 2):
    ch 0 — concentration at [t_split, T)
    ch 1 — head           at [t_split, T)

All runs within a scenario (fixed beta_c, diffc) share the same split and are
stacked into a single ``scenario.npz`` file:
    input_tensor  : (n_runs, 4, T_in,  nlay, ncol)
    output_tensor : (n_runs, 2, T_out, nlay, ncol)

Parallelism
-----------
Scenario-level parallelism is available via the ``n_workers`` parameter of
``generate_simple_henry_dataset``.  Each scenario (one ``scenario.npz``) is
executed in a separate worker process using
``concurrent.futures.ProcessPoolExecutor``.  Workers share no state: each
operates on its own sub-directory and writes its own files.  The main process
collects results and writes the global ``manifest.json`` once all scenarios
finish.  Set ``n_workers=1`` to disable parallelism (default when unset is
``min(n_scenarios, os.cpu_count())``).

Initial-condition mode (``mode="ic_repeat"``)
--------------------------------------------
Instead of splitting the trajectory, the target is the full trajectory after
t = 0 and the input is the initial concentration C0 repeated along time so
that input and output share the same temporal resolution (required by a 3-D
FNO). Head is omitted from the input because H(t=0) is only a zero
placeholder (MODFLOW reports head from t = Δt onwards):
    input_tensor  : (3, T - 1, nlay, ncol)  channels C0 (repeated), beta_c, diffc
    output_tensor : (2, T - 1, nlay, ncol)  concentration, head at t_1 .. t_{T-1}
"""
import dataclasses
import itertools
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .simulation import build_and_run_simple_henry

# ---------------------------------------------------------------------------
# Input / output channel specification
# ---------------------------------------------------------------------------
INPUT_CHANNEL_NAMES = (
    "concentration_t",
    "head_t",
    "beta_c",
    "diffc",
)
OUTPUT_CHANNEL_NAMES = (
    "concentration",
    "head",
)
# Input channels for mode="ic_repeat" (C0 repeated along time; no head)
IC_INPUT_CHANNEL_NAMES = (
    "concentration_0",
    "beta_c",
    "diffc",
)

MODES = ("split", "ic_repeat")

REQUIRED_SCENARIO_FILES = {"scenario.npz"}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scenario_tag(scenario_index: int) -> str:
    return f"scenario_{scenario_index:03d}"


def _run_tag(run_index: int) -> str:
    return f"run_{run_index:03d}"


# ---------------------------------------------------------------------------
# Time-series downsampler
# ---------------------------------------------------------------------------

def _downsample_timeseries(
    head_ts: np.ndarray,
    conc_ts: np.ndarray,
    times: np.ndarray,
    skip: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (head, conc, times) subsampled at [::skip].

    Parameters
    ----------
    head_ts, conc_ts : ndarray of shape (T_raw, nlay, ncol)
    times            : ndarray of shape (T_raw,)
    skip             : int >= 1; stride applied as [::skip]

    Returns
    -------
    Tuple of three arrays with first axis length T = ceil(T_raw / skip).
    """
    if skip <= 0:
        raise ValueError(f"skip must be >= 1, got {skip}")
    return head_ts[::skip], conc_ts[::skip], times[::skip]


# ---------------------------------------------------------------------------
# 3-D tensor builder
# ---------------------------------------------------------------------------

def _build_3d_tensors(
    head_ts: np.ndarray,
    conc_ts: np.ndarray,
    t_split: int,
    nlay: int,
    ncol: int,
    params: dict,
) -> dict | None:
    """Build a single (input, output) 3-D tensor pair from a full downsampled
    time-series.

    Parameters
    ----------
    head_ts, conc_ts : ndarray of shape (T, nlay, ncol)
        Downsampled time-series (already subsampled by ``skip``).
    t_split : int
        Number of time steps assigned to the **input** window.
        Input  → indices [0,       t_split)  → shape (t_split,     nlay, ncol)
        Target → indices [t_split, T)        → shape (T - t_split, nlay, ncol)
    nlay, ncol : int
        Spatial grid dimensions.
    params : dict
        Must contain ``beta_c`` and ``diffc`` (scalar floats).

    Returns
    -------
    dict with keys:
        ``input_tensor``  : float32 array of shape (4, T_in,  nlay, ncol)
        ``output_tensor`` : float32 array of shape (2, T_out, nlay, ncol)
    or ``None`` if t_split is out of range.
    """
    T = head_ts.shape[0]
    T_in  = t_split
    T_out = T - T_in
    if T_in <= 0 or T_out <= 0:
        return None

    # Dynamic channels: concentration and head over the input window
    in_conc = conc_ts[:T_in].astype(np.float32)   # (T_in, nlay, ncol)
    in_head = head_ts[:T_in].astype(np.float32)   # (T_in, nlay, ncol)

    # Static channels: scalar parameters broadcast over (T_in, nlay, ncol)
    beta_field  = np.full((T_in, nlay, ncol), params["beta_c"], dtype=np.float32)
    diffc_field = np.full((T_in, nlay, ncol), params["diffc"],  dtype=np.float32)

    # Stack along channel axis: (4, T_in, nlay, ncol)
    input_tensor = np.stack([in_conc, in_head, beta_field, diffc_field], axis=0)

    # Target channels over the output window: (2, T_out, nlay, ncol)
    output_tensor = np.stack(
        [conc_ts[T_in:].astype(np.float32),
         head_ts[T_in:].astype(np.float32)],
        axis=0,
    )

    return {"input_tensor": input_tensor, "output_tensor": output_tensor}


def _build_ic_repeat_tensors(
    head_ts: np.ndarray,
    conc_ts: np.ndarray,
    nlay: int,
    ncol: int,
    params: dict,
) -> dict | None:
    """Build an (initial condition → full trajectory) 3-D tensor pair.

    Parameters
    ----------
    head_ts, conc_ts : ndarray of shape (T, nlay, ncol)
        Downsampled time-series; index 0 is the initial condition (t = 0).
    nlay, ncol : int
        Spatial grid dimensions.
    params : dict
        Must contain ``beta_c`` and ``diffc`` (scalar floats).

    Returns
    -------
    dict with keys:
        ``input_tensor``  : float32 array of shape (3, T - 1, nlay, ncol)
            C0 repeated T - 1 times, beta_c and diffc broadcast.
        ``output_tensor`` : float32 array of shape (2, T - 1, nlay, ncol)
            concentration and head at indices [1, T).
    or ``None`` if the series has fewer than two time steps.
    """
    T_out = head_ts.shape[0] - 1
    if T_out <= 0:
        return None

    in_conc     = np.repeat(conc_ts[:1].astype(np.float32), T_out, axis=0)
    beta_field  = np.full((T_out, nlay, ncol), params["beta_c"], dtype=np.float32)
    diffc_field = np.full((T_out, nlay, ncol), params["diffc"],  dtype=np.float32)

    # Stack along channel axis: (3, T_out, nlay, ncol)
    input_tensor = np.stack([in_conc, beta_field, diffc_field], axis=0)

    # Full trajectory after t = 0: (2, T_out, nlay, ncol)
    output_tensor = np.stack(
        [conc_ts[1:].astype(np.float32),
         head_ts[1:].astype(np.float32)],
        axis=0,
    )

    return {"input_tensor": input_tensor, "output_tensor": output_tensor}


# ---------------------------------------------------------------------------
# Scenario worker — must be a module-level function to be picklable
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _ScenarioConfig:
    """All data needed by a single scenario worker process."""
    scenario_index:   int
    n_scenarios:      int        # total scenario count (for progress messages)
    beta_c:           float
    diffc:            float
    run_combinations: list       # list of (hk, por, c0_params) tuples
    outdir:           Path
    ncol:             int
    nlay:             int
    lx:               float
    lz:               float
    total_time:       float
    nstp:             int
    al:               float
    at:               float
    rho0:             float
    hk_field:         object     # np.ndarray | None  (avoid annotation for pickle compat)
    vk_field:         object     # np.ndarray | None
    init_method:      str
    t_split:          int
    T_in:             int
    T_out:            int
    T:                int
    skip:             int
    dt:               float
    dt_eff:           float
    overwrite:        bool
    save_timeseries:  bool
    save_modflow_files: bool
    exe_name:         str
    mode:             str = "split"


def _run_scenario(cfg: _ScenarioConfig) -> dict:
    """Run all simulations for one scenario and write ``scenario.npz``.

    Executed inside a worker process (or the main process when n_workers=1).

    Parameters
    ----------
    cfg : _ScenarioConfig
        All parameters for this scenario.

    Returns
    -------
    dict with keys:
        ``scenario_summary``  : dict  (for the global manifest)
        ``run_records``       : list[dict]
        ``run_failures``      : list[dict]
    """
    scenario_tag = _scenario_tag(cfg.scenario_index)
    scenario_dir = cfg.outdir / scenario_tag
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_file = scenario_dir / "scenario.npz"

    prefix = f"[{cfg.scenario_index:03d}/{cfg.n_scenarios:03d}] {scenario_tag}"

    # --- Skip if already done ---
    if scenario_file.exists() and not cfg.overwrite:
        print(f"{prefix}  SKIP (scenario.npz exists)")
        summary = {
            "scenario":        scenario_tag,
            "scenario_index":  cfg.scenario_index,
            "beta_c":          cfg.beta_c,
            "diffc":           cfg.diffc,
            "n_total_runs":    len(cfg.run_combinations),
            "n_ok_runs":       None,
            "n_skipped_runs":  len(cfg.run_combinations),
            "n_failed_runs":   0,
        }
        return {"scenario_summary": summary, "run_records": [], "run_failures": []}

    n_runs = len(cfg.run_combinations)
    print(
        f"{prefix}  beta_c={cfg.beta_c}  diffc={cfg.diffc}  runs={n_runs}"
    )

    # Accumulators
    scenario_inputs:   list[np.ndarray] = []
    scenario_outputs:  list[np.ndarray] = []
    scenario_ts_head:  list[np.ndarray] = []  # full downsampled head (T, nlay, ncol)
    scenario_ts_conc:  list[np.ndarray] = []  # full downsampled conc (T, nlay, ncol)
    scenario_run_meta: list[dict]       = []
    run_records:       list[dict]       = []
    run_failures:      list[dict]       = []
    times_ds_ref:      np.ndarray | None = None

    for run_index, (hk, por, c0_params) in enumerate(cfg.run_combinations, start=1):
        params = {
            "beta_c":      cfg.beta_c,
            "diffc":       cfg.diffc,
            "hk":          float(hk),
            "por":         float(por),
            "al":          cfg.al,
            "at":          cfg.at,
            "rho0":        cfg.rho0,
            "init_method": cfg.init_method,
            **c0_params,
        }

        run_tag = _run_tag(run_index)
        run_dir = scenario_dir / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"  {prefix}  [{run_index:04d}/{n_runs:04d}] {run_tag} "
            f"hk={hk} por={por}"
        )

        c0_kwargs = {k: v for k, v in c0_params.items() if k != "sample_idx"}
        try:
            head_ts_raw, conc_ts_raw, times_raw = build_and_run_simple_henry(
                workspace=run_dir,
                ncol=cfg.ncol,
                nlay=cfg.nlay,
                Lx=cfg.lx,
                Lz=cfg.lz,
                total_time=cfg.total_time,
                nstp=cfg.nstp,
                por=por,
                hk=hk,
                vk=hk,   # isotropic; vk_field override available
                al=cfg.al,
                at=cfg.at,
                diffc=cfg.diffc,
                beta_c=cfg.beta_c,
                rho0=cfg.rho0,
                hk_field=cfg.hk_field,
                vk_field=cfg.vk_field,
                return_timeseries=True,
                exe_name=cfg.exe_name,
                init_method=cfg.init_method,
                **c0_kwargs,
            )

            # 1. Downsample
            head_ds, conc_ds, times_ds = _downsample_timeseries(
                head_ts_raw, conc_ts_raw, times_raw, cfg.skip
            )
            if times_ds_ref is None:
                times_ds_ref = times_ds  # identical for all runs in this scenario

            # 2. Build 3-D tensors
            if cfg.mode == "ic_repeat":
                tensors = _build_ic_repeat_tensors(
                    head_ts=head_ds,
                    conc_ts=conc_ds,
                    nlay=cfg.nlay,
                    ncol=cfg.ncol,
                    params=params,
                )
            else:
                tensors = _build_3d_tensors(
                    head_ts=head_ds,
                    conc_ts=conc_ds,
                    t_split=cfg.t_split,
                    nlay=cfg.nlay,
                    ncol=cfg.ncol,
                    params=params,
                )
            if tensors is None:
                raise ValueError(
                    f"No valid {cfg.mode} tensors for t_split={cfg.t_split}, "
                    f"T={head_ds.shape[0]}"
                )

            scenario_inputs.append(tensors["input_tensor"])
            scenario_outputs.append(tensors["output_tensor"])
            if cfg.save_timeseries:
                scenario_ts_head.append(head_ds.astype(np.float32))
                scenario_ts_conc.append(conc_ds.astype(np.float32))
            scenario_run_meta.append(params)

            # Remove MODFLOW workspace files unless explicitly kept
            if not cfg.save_modflow_files:
                shutil.rmtree(run_dir, ignore_errors=True)

            run_records.append({
                "id":       f"{scenario_tag}/{run_tag}",
                "scenario": scenario_tag,
                "run":      run_tag,
                "status":   "ok",
                **params,
            })

        except Exception as exc:
            failure = {
                "id":       f"{scenario_tag}/{run_tag}",
                "scenario": scenario_tag,
                "run":      run_tag,
                "status":   "failed",
                "error":    str(exc),
                **params,
            }
            run_failures.append(failure)
            run_records.append(failure)
            print(f"    {prefix}  {run_tag} FAILED: {exc}")
            if not cfg.save_modflow_files and run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    # --- Batch and save scenario.npz ---
    n_ok = len(scenario_inputs)
    if n_ok > 0:
        inputs_batch  = np.stack(scenario_inputs,  axis=0)   # (n_ok, cin, T_in,  nlay, ncol)
        outputs_batch = np.stack(scenario_outputs, axis=0)   # (n_ok, 2,   T_out, nlay, ncol)

        input_channel_names = (
            IC_INPUT_CHANNEL_NAMES if cfg.mode == "ic_repeat" else INPUT_CHANNEL_NAMES
        )
        payload: dict = {
            "input_tensor":         inputs_batch,
            "output_tensor":        outputs_batch,
            "input_channel_names":  np.asarray(list(input_channel_names)),
            "output_channel_names": np.asarray(list(OUTPUT_CHANNEL_NAMES)),
            "mode":                 cfg.mode,
            "t_split":              int(cfg.t_split),
            "T_in":                 int(cfg.T_in),
            "T_out":                int(cfg.T_out),
            "times_in":             times_ds_ref[:cfg.t_split].astype(np.float64),
            "times_out":            times_ds_ref[cfg.t_split:].astype(np.float64),
            "skip":                 int(cfg.skip),
            "dt_raw":               float(cfg.dt),
            "dt_eff":               float(cfg.dt_eff),
            "beta_c":               float(cfg.beta_c),
            "diffc":                float(cfg.diffc),
            "al":                   float(cfg.al),
            "at":                   float(cfg.at),
            "rho0":                 float(cfg.rho0),
            "init_method":          cfg.init_method,
            "run_params":           np.asarray([json.dumps(m) for m in scenario_run_meta]),
            "ncol":                 int(cfg.ncol),
            "nlay":                 int(cfg.nlay),
            "lx":                   float(cfg.lx),
            "lz":                   float(cfg.lz),
            "total_time":           float(cfg.total_time),
            "nstp":                 int(cfg.nstp),
            "scenario_index":       int(cfg.scenario_index),
        }

        if cfg.save_timeseries:
            # Full downsampled series stacked over runs: (n_ok, T, nlay, ncol)
            payload["head_timeseries"] = np.stack(scenario_ts_head, axis=0)
            payload["conc_timeseries"] = np.stack(scenario_ts_conc, axis=0)
            payload["times_ds"]        = times_ds_ref.astype(np.float64)

        np.savez_compressed(scenario_file, **payload)
        print(
            f"{prefix}  DONE  n_ok={n_ok}/{n_runs}  "
            f"input={inputs_batch.shape}  output={outputs_batch.shape}"
        )
    else:
        print(f"{prefix}  WARNING: all runs failed — scenario.npz not written")

    # Scenario-level manifest (written per worker, not dependent on other scenarios)
    scenario_manifest = {
        "scenario":       scenario_tag,
        "scenario_index": cfg.scenario_index,
        "beta_c":         float(cfg.beta_c),
        "diffc":          float(cfg.diffc),
        "mode":           cfg.mode,
        "t_split":        int(cfg.t_split),
        "T_in":           int(cfg.T_in),
        "T_out":          int(cfg.T_out),
        "skip":           int(cfg.skip),
        "dt_raw":         float(cfg.dt),
        "dt_eff":         float(cfg.dt_eff),
        "n_total_runs":   len(cfg.run_combinations),
        "n_ok_runs":      n_ok,
        "n_failed_runs":  len(run_failures),
        "failures":       run_failures,
    }
    with (scenario_dir / "scenario_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(scenario_manifest, fp, indent=2)

    summary = {
        "scenario":       scenario_tag,
        "scenario_index": cfg.scenario_index,
        "beta_c":         float(cfg.beta_c),
        "diffc":          float(cfg.diffc),
        "n_total_runs":   len(cfg.run_combinations),
        "n_ok_runs":      n_ok,
        "n_skipped_runs": 0,
        "n_failed_runs":  len(run_failures),
    }
    return {
        "scenario_summary": summary,
        "run_records":      run_records,
        "run_failures":     run_failures,
    }


# ---------------------------------------------------------------------------
# Main dataset generator
# ---------------------------------------------------------------------------

def generate_simple_henry_dataset(
    outdir,
    # Parameter sweep (each list entry is a distinct value; Cartesian product)
    beta_c_values,
    diffc_values,
    hk_values,
    por_values,
    # Initial concentration profile parameters
    init_method: str = "wedge",
    init_field_args: dict | None = None,
    # Grid / time
    ncol: int = 80,
    nlay: int = 40,
    lx: float = 2.0,
    lz: float = 1.0,
    total_time: float = 30.0,
    nstp: int = 240,
    # Dispersion
    al: float = 0.0,
    at: float = 0.0,
    # Spatially varying K fields (override scalar hk; disables hk sweep)
    hk_field=None,
    vk_field=None,
    # Reference density
    rho0: float = 1000.0,
    # Dataset controls
    t_split: int | None = None,
    skip: int = 1,
    overwrite: bool = False,
    max_runs_per_scenario: int | None = None,
    save_timeseries: bool = False,
    save_modflow_files: bool = False,
    seed: int = 42,
    exe_name: str = "mf6",
    # Parallelism
    n_workers: int | None = None,
    # Tensor layout
    mode: str = "split",
):
    """Generate a 3-D (space × time) dataset for the simplified Henry problem.

    For each (beta_c, diffc) *scenario* and each (hk, por, c0) *run*
    combination within that scenario, one MODFLOW 6 simulation is run. The
    resulting time-series (shape ``(T_raw, nlay, ncol)``) is:

    1. Downsampled by ``skip``:  ``[::skip]``  →  shape ``(T, nlay, ncol)``
    2. Split at ``t_split``:
         input  → ``[:t_split]``  → channels (4, T_in,  nlay, ncol)
         target → ``[t_split:]``  → channels (2, T_out, nlay, ncol)

    All runs in a scenario are stacked and saved as a single
    ``scenario_NNN/scenario.npz``.

    Parameters
    ----------
    outdir : str or Path
        Root output directory. Created if it does not exist.
    beta_c_values, diffc_values, hk_values, por_values : list of float
        Parameter values to sweep. Full Cartesian product.
    init_method : str
        Initial concentration method: ``'wedge'`` or ``'random'``.
    init_field_args : dict or None
        Arguments for the initial field generator, depending on *init_method*.
    ncol, nlay : int
        Grid dimensions (columns and layers).
    lx, lz : float
        Domain horizontal and vertical extents [m].
    total_time : float
        Simulation duration [days].
    nstp : int
        Number of MODFLOW time steps.
    al, at : float
        Longitudinal / transverse dispersivity [m].
    hk_field, vk_field : array_like of shape (nlay, ncol) or None
        Spatially varying K fields overriding scalar *hk_values*.
    rho0 : float
        Reference fluid density [kg/m³].
    t_split : int or None
        Number of downsampled steps assigned to the input window.
        Defaults to ``T // 2`` where ``T = len(range(0, nstp + 1, skip))``.
    skip : int
        Temporal stride applied to the raw MODFLOW time-series before
        splitting: ``time_series[::skip]``. ``skip=1`` uses every step.
    overwrite : bool
        If False (default) skip scenarios whose ``scenario.npz`` already exists.
    max_runs_per_scenario : int or None
        Cap on run combinations per scenario (useful for quick tests).
    save_timeseries : bool
        If True, include full downsampled head/conc time-series in
        ``scenario.npz`` as ``head_timeseries`` and ``conc_timeseries``
        arrays of shape ``(n_ok, T, nlay, ncol)``.
    save_modflow_files : bool
        If True, keep all MODFLOW 6 workspace files after each run.
    seed : int
        Random seed saved to the manifest.
    exe_name : str or Path
        MODFLOW 6 executable name or path.
    n_workers : int or None
        Number of parallel worker processes for scenario-level parallelism.
        Each worker handles one scenario (one ``scenario.npz``) at a time.
        Defaults to ``min(n_scenarios, os.cpu_count())``.
        Set to ``1`` to run sequentially without spawning subprocesses.
    mode : str
        ``'split'`` (default): input is [0, t_split), target is [t_split, T).
        ``'ic_repeat'``: input is C0 repeated T - 1 times (channels C0,
        beta_c, diffc), target is the full trajectory [1, T). ``t_split``
        must not be given in this mode.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if hk_field is not None and len(hk_values) != 1:
        raise ValueError("hk sweep is incompatible with --kappa-file; provide one hk value")
    if vk_field is not None and len(hk_values) != 1:
        raise ValueError("vk from kappa file is incompatible with hk sweep")

    if init_field_args is None:
        init_field_args = {}

    if skip <= 0:
        raise ValueError(f"skip must be >= 1, got {skip}")

    dt     = total_time / nstp        # raw time step size [days]
    dt_eff = dt * skip                # effective time step after downsampling [days]
    T_raw  = nstp + 1                 # includes t = 0
    T      = len(range(0, T_raw, skip))
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {MODES}.")
    if mode == "ic_repeat":
        if t_split is not None:
            raise ValueError("t_split is not used with mode='ic_repeat'")
        if T < 2:
            raise ValueError(
                f"mode='ic_repeat' needs T >= 2 (T = len(range(0, {T_raw}, {skip})))"
            )
        # Input window is the initial condition [0, 1), repeated to T - 1 steps
        t_split = 1
        T_in  = T - 1
        T_out = T - 1
        input_channel_names = IC_INPUT_CHANNEL_NAMES
    else:
        if t_split is None:
            t_split = T // 2
        if not (0 < t_split < T):
            raise ValueError(
                f"t_split={t_split} must be in (0, {T}) "
                f"(T = len(range(0, {T_raw}, {skip})))"
            )
        T_in  = t_split
        T_out = T - t_split
        input_channel_names = INPUT_CHANNEL_NAMES

    scenario_pairs = list(itertools.product(beta_c_values, diffc_values))
    n_scenarios = len(scenario_pairs)

    # -----------------------------------------------------------------------
    # Build run combinations per scenario
    # -----------------------------------------------------------------------
    def _build_run_combinations(beta_c, diffc):
        if init_method == "wedge":
            x_toe_list = (
                init_field_args.get("c0_x_toe")
                or init_field_args.get("init0_x_toe")
                or [0.4 * lx]
            )
            x_top_list = (
                init_field_args.get("c0_x_top")
                or init_field_args.get("init0_x_top")
                or [1.0 * lx]
            )
            trans_width_list = (
                init_field_args.get("c0_trans_width")
                or init_field_args.get("init0_trans_width")
                or [0.05 * lx]
            )
            combos = list(itertools.product(
                hk_values, por_values, x_toe_list, x_top_list, trans_width_list
            ))
            if max_runs_per_scenario is not None:
                combos = combos[:max_runs_per_scenario]
            return [
                (hk, por, {
                    "c0_x_toe":       float(xtoe),
                    "c0_x_top":       float(xtop),
                    "c0_trans_width": float(tw),
                })
                for (hk, por, xtoe, xtop, tw) in combos
            ]
        elif init_method == "random":
            rf_types = init_field_args.get("random_field_type") or ["grf"]
            if isinstance(rf_types, str):
                rf_types = [s.strip() for s in rf_types.split(",") if s.strip()]
            rf_smooth = init_field_args.get("random_field_smoothness") or [1.0]
            if isinstance(rf_smooth, (int, float)):
                rf_smooth = [float(rf_smooth)]
            rf_len_scale = init_field_args.get("random_field_len_scale") or [0.1]
            if isinstance(rf_len_scale, (int, float)):
                rf_len_scale = [float(rf_len_scale)]
            rf_var = init_field_args.get("random_field_var") or [0.1]
            if isinstance(rf_var, (int, float)):
                rf_var = [float(rf_var)]
            rf_count = int(init_field_args.get("random_field_count", 1))
            combos = list(itertools.product(
                hk_values, por_values, rf_types, rf_smooth, rf_len_scale, rf_var, range(rf_count)
            ))
            if max_runs_per_scenario is not None:
                combos = combos[:max_runs_per_scenario]
            return [
                (hk, por, {
                    "random_field_type":       str(ft),
                    "random_field_smoothness": float(sm),
                    "random_field_len_scale":  float(ls),
                    "random_field_var":        float(var),
                    "sample_idx":              int(idx),
                })
                for (hk, por, ft, sm, ls, var, idx) in combos
            ]
        else:
            raise ValueError(f"Unknown init_method '{init_method}'")

    # -----------------------------------------------------------------------
    # Build per-scenario configs
    # -----------------------------------------------------------------------
    configs = [
        _ScenarioConfig(
            scenario_index=scenario_index,
            n_scenarios=n_scenarios,
            beta_c=float(beta_c),
            diffc=float(diffc),
            run_combinations=_build_run_combinations(beta_c, diffc),
            outdir=outdir,
            ncol=ncol,
            nlay=nlay,
            lx=lx,
            lz=lz,
            total_time=total_time,
            nstp=nstp,
            al=al,
            at=at,
            rho0=rho0,
            hk_field=hk_field,
            vk_field=vk_field,
            init_method=init_method,
            t_split=t_split,
            T_in=T_in,
            T_out=T_out,
            T=T,
            skip=skip,
            dt=dt,
            dt_eff=dt_eff,
            overwrite=overwrite,
            save_timeseries=save_timeseries,
            save_modflow_files=save_modflow_files,
            exe_name=exe_name,
            mode=mode,
        )
        for scenario_index, (beta_c, diffc) in enumerate(scenario_pairs, start=1)
    ]

    # -----------------------------------------------------------------------
    # Resolve n_workers
    # -----------------------------------------------------------------------
    cpu_count = os.cpu_count() or 1
    effective_workers = min(n_scenarios, cpu_count) if n_workers is None else n_workers
    effective_workers = max(1, effective_workers)

    print(
        f"Starting generation: mode={mode}  scenarios={n_scenarios}  workers={effective_workers}  "
        f"T_raw={T_raw}  skip={skip}  T={T}  t_split={t_split}  "
        f"T_in={T_in}  T_out={T_out}"
    )

    # -----------------------------------------------------------------------
    # Run scenarios — parallel or sequential
    # -----------------------------------------------------------------------
    all_summaries:   list[dict] = []
    all_run_records: list[dict] = []
    all_failures:    list[dict] = []

    if effective_workers == 1:
        # Sequential — avoids subprocess overhead for small jobs or debugging
        for cfg in configs:
            result = _run_scenario(cfg)
            all_summaries.append(result["scenario_summary"])
            all_run_records.extend(result["run_records"])
            all_failures.extend(result["run_failures"])
    else:
        # Parallel — one scenario per worker process
        futures = {}
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            for cfg in configs:
                fut = pool.submit(_run_scenario, cfg)
                futures[fut] = cfg.scenario_index

            completed = 0
            for fut in as_completed(futures):
                completed += 1
                idx = futures[fut]
                try:
                    result = fut.result()
                    all_summaries.append(result["scenario_summary"])
                    all_run_records.extend(result["run_records"])
                    all_failures.extend(result["run_failures"])
                    print(
                        f"[{completed:03d}/{n_scenarios:03d}] "
                        f"scenario_{idx:03d} finished"
                    )
                except Exception as exc:
                    print(
                        f"[{completed:03d}/{n_scenarios:03d}] "
                        f"scenario_{idx:03d} raised an unexpected error: {exc}"
                    )
                    all_failures.append({
                        "scenario_index": idx,
                        "error": str(exc),
                        "status": "scenario_error",
                    })

    # Sort summaries by scenario_index for a deterministic manifest
    all_summaries.sort(key=lambda s: s["scenario_index"])

    # -----------------------------------------------------------------------
    # Write global manifest
    # -----------------------------------------------------------------------
    manifest = {
        "workflow": "simple_henry_3d_dataset",
        "pde": {
            "description": (
                "Simplified Henry density-driven convection: elliptic pressure + "
                "parabolic solute transport, zero storage, zero influx, "
                "homogeneous Dirichlet BCs on all sides."
            ),
            "Ss": 0.0,
            "influx": 0.0,
            "bc_type": "homogeneous_dirichlet_all_sides",
        },
        "mode": mode,
        "input_channel_names":  list(input_channel_names),
        "output_channel_names": list(OUTPUT_CHANNEL_NAMES),
        "grid":  {"ncol": ncol, "nlay": nlay, "lx": lx, "lz": lz},
        "time":  {"total_time": total_time, "nstp": nstp, "dt": dt},
        "dispersion": {"al": al, "at": at},
        "t_split":  int(t_split),
        "T_in":     int(T_in),
        "T_out":    int(T_out),
        "T":        int(T),
        "skip":     int(skip),
        "dt_raw":   float(dt),
        "dt_eff":   float(dt_eff),
        "seed":     int(seed),
        "n_workers": effective_workers,
        "input_shape":  [len(input_channel_names),  T_in,  nlay, ncol],
        "output_shape": [len(OUTPUT_CHANNEL_NAMES), T_out, nlay, ncol],
        "n_scenarios":    n_scenarios,
        "n_total_runs":   sum(s.get("n_total_runs",   0) for s in all_summaries),
        "n_ok_runs":      sum(s.get("n_ok_runs",      0) for s in all_summaries if s.get("n_ok_runs") is not None),
        "n_skipped_runs": sum(s.get("n_skipped_runs", 0) for s in all_summaries),
        "n_failed_runs":  sum(s.get("n_failed_runs",  0) for s in all_summaries),
        "scenarios":      all_summaries,
        "failures":       all_failures,
    }

    with (outdir / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(
        "Generation done: "
        f"scenarios={manifest['n_scenarios']}  "
        f"workers={effective_workers}  "
        f"runs_ok={manifest['n_ok_runs']}  "
        f"runs_failed={manifest['n_failed_runs']}  "
        f"input_shape={manifest['input_shape']}  "
        f"output_shape={manifest['output_shape']}"
    )
    print(f"Manifest: {outdir / 'manifest.json'}")
