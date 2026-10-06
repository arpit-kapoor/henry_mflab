"""Dataset generator for the simplified Henry problem (paper Sec. 5).

Generates 3-D (space × time) (input, output) tensors mapping an initial
concentration field + scenario parameters to the full simulated trajectory.

Each simulation **run** produces one data point. The raw MODFLOW series
(T_raw = nstp + 1 frames including t = 0) is downsampled by ``skip``
(``[::skip]``) to T frames, then:
    input_tensor  : float32, shape (3, T - 1, nlay, ncol)
    output_tensor : float32, shape (2, T - 1, nlay, ncol)

Input channels (cin = 3), each repeated over the T - 1 target times:
    ch 0 — concentration_0 : initial concentration field C₀
    ch 1 — beta_c          : solutal expansion coefficient (broadcast)
    ch 2 — diffc           : effective diffusion coefficient (broadcast)

Output channels (cout = 2), at t_1 .. t_{T-1}:
    ch 0 — concentration
    ch 1 — head

Head is not an input: H(t=0) is only a zero placeholder (MODFLOW reports head
from t = Δt onwards).

Scenarios are the (beta_c, diffc) Cartesian product. Runs differ by the
initial condition: ``random_field_count`` GRF samples for every
(len_scale, var) pair. Scenario i draws its run set from the seed
``[seed, i]``, so the dataset is reproducible and every run has a distinct C₀.

All runs within a scenario are stacked into a single ``scenario.npz``:
    input_tensor  : (n_runs, 3, T - 1, nlay, ncol)
    output_tensor : (n_runs, 2, T - 1, nlay, ncol)

Parallelism: each scenario runs in its own worker process
(``n_workers``, default ``min(n_scenarios, os.cpu_count())``; 1 = sequential).
"""
import dataclasses
import itertools
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .simulation import HEAD_BCS, build_and_run_simplified_henry, create_random_field

# ---------------------------------------------------------------------------
# Input / output channel specification
# ---------------------------------------------------------------------------
INPUT_CHANNEL_NAMES = (
    "concentration_0",
    "beta_c",
    "diffc",
)
OUTPUT_CHANNEL_NAMES = (
    "concentration",
    "head",
)

# ---------------------------------------------------------------------------
# Helpers shared with classical_henry
# ---------------------------------------------------------------------------

def _scenario_tag(scenario_index: int) -> str:
    return f"scenario_{scenario_index:03d}"


def _run_tag(run_index: int) -> str:
    return f"run_{run_index:03d}"


def _downsample_timeseries(
    head_ts: np.ndarray,
    conc_ts: np.ndarray,
    times: np.ndarray,
    skip: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (head, conc, times) subsampled at [::skip]."""
    if skip <= 0:
        raise ValueError(f"skip must be >= 1, got {skip}")
    return head_ts[::skip], conc_ts[::skip], times[::skip]


# ---------------------------------------------------------------------------
# 3-D tensor builder
# ---------------------------------------------------------------------------

def _build_tensors(
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
    dict with ``input_tensor`` (3, T - 1, nlay, ncol) and ``output_tensor``
    (2, T - 1, nlay, ncol), or ``None`` if the series has fewer than two steps.
    """
    T_out = head_ts.shape[0] - 1
    if T_out <= 0:
        return None

    in_conc     = np.repeat(conc_ts[:1].astype(np.float32), T_out, axis=0)
    beta_field  = np.full((T_out, nlay, ncol), params["beta_c"], dtype=np.float32)
    diffc_field = np.full((T_out, nlay, ncol), params["diffc"],  dtype=np.float32)

    input_tensor = np.stack([in_conc, beta_field, diffc_field], axis=0)
    output_tensor = np.stack(
        [conc_ts[1:].astype(np.float32),
         head_ts[1:].astype(np.float32)],
        axis=0,
    )

    return {"input_tensor": input_tensor, "output_tensor": output_tensor}


# ---------------------------------------------------------------------------
# Run-set sampling
# ---------------------------------------------------------------------------

def _sample_runs(
    init_field_args: dict,
    nlay: int,
    ncol: int,
    lx: float,
    lz: float,
    c_sea: float,
    seed: list[int],
    max_runs: int | None,
) -> list[tuple[np.ndarray, dict]]:
    """Sample the (C₀, ic_params) pairs of one run set from ``seed``."""
    rng = np.random.default_rng(seed)
    combos = list(itertools.product(
        init_field_args.get("random_field_len_scale") or [0.1],
        init_field_args.get("random_field_var") or [0.1],
        range(int(init_field_args.get("random_field_count", 1))),
    ))
    if max_runs is not None:
        combos = combos[:max_runs]

    runs = []
    for ls, var, idx in combos:
        field_seed = int(rng.integers(0, 2**31 - 1))
        conc0 = create_random_field(
            nlay=nlay, ncol=ncol, Lx=lx, Lz=lz, c_fresh=0.0, c_sea=c_sea,
            len_scale=float(ls), var=float(var), seed=field_seed,
            taper=True, period=1.0,
        )
        ic_params = {
            "random_field_len_scale": float(ls),
            "random_field_var":       float(var),
            "sample_idx":             int(idx),
            "field_seed":             field_seed,
        }
        runs.append((conc0, ic_params))
    return runs


# ---------------------------------------------------------------------------
# Scenario worker — must be a module-level function to be picklable
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _ScenarioConfig:
    """All data needed by a single scenario worker process."""
    scenario_index:   int
    n_scenarios:      int
    beta_c:           float
    diffc:            float
    run_seed:         list       # [seed, scenario_index]
    init_field_args:  dict
    max_runs:         object     # int | None
    outdir:           Path
    ncol:             int
    nlay:             int
    lx:               float
    lz:               float
    total_time:       float
    nstp:             int
    hk:               float
    por:              float
    al:               float
    at:               float
    rho0:             float
    c_sea:            float
    head_bc:          str
    T:                int
    skip:             int
    dt:               float
    dt_eff:           float
    overwrite:        bool
    save_timeseries:  bool
    save_modflow_files: bool
    exe_name:         str


def _run_scenario(cfg: _ScenarioConfig) -> dict:
    """Run all simulations for one scenario and write ``scenario.npz``."""
    scenario_tag = _scenario_tag(cfg.scenario_index)
    scenario_dir = cfg.outdir / scenario_tag
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_file = scenario_dir / "scenario.npz"

    prefix = f"[{cfg.scenario_index:03d}/{cfg.n_scenarios:03d}] {scenario_tag}"
    runs = _sample_runs(
        init_field_args=cfg.init_field_args,
        nlay=cfg.nlay,
        ncol=cfg.ncol,
        lx=cfg.lx,
        lz=cfg.lz,
        c_sea=cfg.c_sea,
        seed=cfg.run_seed,
        max_runs=cfg.max_runs,
    )
    n_runs = len(runs)

    if scenario_file.exists() and not cfg.overwrite:
        print(f"{prefix}  SKIP (scenario.npz exists)")
        summary = {
            "scenario":        scenario_tag,
            "scenario_index":  cfg.scenario_index,
            "beta_c":          cfg.beta_c,
            "diffc":           cfg.diffc,
            "n_total_runs":    n_runs,
            "n_ok_runs":       None,
            "n_skipped_runs":  n_runs,
            "n_failed_runs":   0,
        }
        return {"scenario_summary": summary, "run_failures": []}

    print(f"{prefix}  beta_c={cfg.beta_c}  diffc={cfg.diffc}  runs={n_runs}")

    scenario_inputs:   list[np.ndarray] = []
    scenario_outputs:  list[np.ndarray] = []
    scenario_ts_head:  list[np.ndarray] = []
    scenario_ts_conc:  list[np.ndarray] = []
    scenario_run_meta: list[dict]       = []
    run_failures:      list[dict]       = []
    times_ds_ref:      np.ndarray | None = None

    for run_index, (conc0, ic_params) in enumerate(runs, start=1):
        params = {
            "beta_c":  cfg.beta_c,
            "diffc":   cfg.diffc,
            "hk":      cfg.hk,
            "por":     cfg.por,
            "al":      cfg.al,
            "at":      cfg.at,
            "rho0":    cfg.rho0,
            "head_bc": cfg.head_bc,
            **ic_params,
        }

        run_tag = _run_tag(run_index)
        run_dir = scenario_dir / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"  {prefix}  [{run_index:04d}/{n_runs:04d}] {run_tag}")

        try:
            head_ts_raw, conc_ts_raw, times_raw = build_and_run_simplified_henry(
                workspace=run_dir,
                conc0=conc0,
                ncol=cfg.ncol,
                nlay=cfg.nlay,
                Lx=cfg.lx,
                Lz=cfg.lz,
                total_time=cfg.total_time,
                nstp=cfg.nstp,
                por=cfg.por,
                hk=cfg.hk,
                al=cfg.al,
                at=cfg.at,
                diffc=cfg.diffc,
                beta_c=cfg.beta_c,
                rho0=cfg.rho0,
                head_bc=cfg.head_bc,
                exe_name=cfg.exe_name,
            )

            head_ds, conc_ds, times_ds = _downsample_timeseries(
                head_ts_raw, conc_ts_raw, times_raw, cfg.skip
            )
            if times_ds_ref is None:
                times_ds_ref = times_ds  # identical for all runs in this scenario

            tensors = _build_tensors(
                head_ts=head_ds,
                conc_ts=conc_ds,
                nlay=cfg.nlay,
                ncol=cfg.ncol,
                params=params,
            )
            if tensors is None:
                raise ValueError(f"Need T >= 2 frames, got T={head_ds.shape[0]}")

            scenario_inputs.append(tensors["input_tensor"])
            scenario_outputs.append(tensors["output_tensor"])
            if cfg.save_timeseries:
                scenario_ts_head.append(head_ds.astype(np.float32))
                scenario_ts_conc.append(conc_ds.astype(np.float32))
            scenario_run_meta.append(params)

            if not cfg.save_modflow_files:
                shutil.rmtree(run_dir, ignore_errors=True)

        except Exception as exc:
            run_failures.append({
                "id":     f"{scenario_tag}/{run_tag}",
                "status": "failed",
                "error":  str(exc),
                **params,
            })
            print(f"    {prefix}  {run_tag} FAILED: {exc}")
            if not cfg.save_modflow_files and run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    # --- Batch and save scenario.npz ---
    n_ok = len(scenario_inputs)
    if n_ok > 0:
        inputs_batch  = np.stack(scenario_inputs,  axis=0)   # (n_ok, 3, T-1, nlay, ncol)
        outputs_batch = np.stack(scenario_outputs, axis=0)   # (n_ok, 2, T-1, nlay, ncol)

        payload: dict = {
            "input_tensor":         inputs_batch,
            "output_tensor":        outputs_batch,
            "input_channel_names":  np.asarray(list(INPUT_CHANNEL_NAMES)),
            "output_channel_names": np.asarray(list(OUTPUT_CHANNEL_NAMES)),
            "run_seed":             np.asarray(cfg.run_seed),
            "T_out":                int(cfg.T - 1),
            "times_out":            times_ds_ref[1:].astype(np.float64),
            "skip":                 int(cfg.skip),
            "dt_raw":               float(cfg.dt),
            "dt_eff":               float(cfg.dt_eff),
            "beta_c":               float(cfg.beta_c),
            "diffc":                float(cfg.diffc),
            "hk":                   float(cfg.hk),
            "por":                  float(cfg.por),
            "al":                   float(cfg.al),
            "at":                   float(cfg.at),
            "rho0":                 float(cfg.rho0),
            "head_bc":              cfg.head_bc,
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

    scenario_manifest = {
        "scenario":       scenario_tag,
        "scenario_index": cfg.scenario_index,
        "beta_c":         float(cfg.beta_c),
        "diffc":          float(cfg.diffc),
        "T_out":          int(cfg.T - 1),
        "skip":           int(cfg.skip),
        "dt_raw":         float(cfg.dt),
        "dt_eff":         float(cfg.dt_eff),
        "n_total_runs":   n_runs,
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
        "n_total_runs":   n_runs,
        "n_ok_runs":      n_ok,
        "n_skipped_runs": 0,
        "n_failed_runs":  len(run_failures),
    }
    return {"scenario_summary": summary, "run_failures": run_failures}


# ---------------------------------------------------------------------------
# Main dataset generator
# ---------------------------------------------------------------------------

def generate_simplified_henry_dataset(
    outdir,
    # Scenario sweep (Cartesian product)
    beta_c_values,
    diffc_values,
    # Initial concentration
    init_field_args: dict | None = None,
    # Grid / time
    ncol: int = 40,
    nlay: int = 20,
    lx: float = 2.0,
    lz: float = 1.0,
    total_time: float = 1.0,
    nstp: int = 50,
    # Physics
    hk: float = 50.0,
    por: float = 0.35,
    al: float = 0.0,
    at: float = 0.0,
    rho0: float = 1000.0,
    c_sea: float = 35.0,
    head_bc: str = "lr",
    # Dataset controls
    skip: int = 2,
    overwrite: bool = False,
    max_runs_per_scenario: int | None = None,
    save_timeseries: bool = False,
    save_modflow_files: bool = False,
    seed: int = 42,
    exe_name: str = "mf6",
    n_workers: int | None = None,
):
    """Generate a 3-D (space × time) dataset for the simplified Henry problem.

    For each (beta_c, diffc) *scenario* every run of its run set (one GRF C₀
    each) is simulated with MODFLOW 6. The time-series is downsampled by
    ``skip`` and converted to an (IC → trajectory) tensor pair; all runs in a
    scenario are saved to ``scenario_NNN/scenario.npz``.

    Parameters
    ----------
    outdir : str or Path
        Root output directory. Created if it does not exist.
    beta_c_values, diffc_values : list of float
        Scenario parameter values (full Cartesian product).
    init_field_args : dict or None
        Lists ``random_field_len_scale`` and ``random_field_var`` (Cartesian
        product) and ``random_field_count`` GRF samples per pair.
    ncol, nlay, lx, lz : grid dimensions and extents [m].
    total_time, nstp : simulation duration [days] and number of time steps.
    hk, por, al, at, rho0 : physical parameters (see simulation module).
    c_sea : float
        Maximum of the min–max scaled initial field [kg/m³].
    head_bc : str
        ``'lr'`` (paper dataset) or ``'all'``; see
        ``build_and_run_simplified_henry``.
    skip : int
        Temporal stride applied as ``time_series[::skip]``.
    overwrite : bool
        If False skip scenarios whose ``scenario.npz`` already exists.
    max_runs_per_scenario : int or None
        Cap on runs per scenario (for quick tests).
    save_timeseries : bool
        Also store full downsampled head/conc series in ``scenario.npz``.
    save_modflow_files : bool
        Keep MODFLOW 6 workspace files after each run.
    seed : int
        Base seed for the run sets; scenario i uses ``[seed, i]``.
    exe_name : str or Path
        MODFLOW 6 executable name or path.
    n_workers : int or None
        Parallel worker processes (one scenario each). Defaults to
        ``min(n_scenarios, os.cpu_count())``; ``1`` runs sequentially.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if init_field_args is None:
        init_field_args = {}
    if skip <= 0:
        raise ValueError(f"skip must be >= 1, got {skip}")
    if head_bc not in HEAD_BCS:
        raise ValueError(f"Unknown head_bc '{head_bc}'. Expected one of {HEAD_BCS}.")

    dt     = total_time / nstp        # raw time step size [days]
    dt_eff = dt * skip                # effective time step after downsampling [days]
    T_raw  = nstp + 1                 # includes t = 0
    T      = len(range(0, T_raw, skip))
    if T < 2:
        raise ValueError(f"Need T >= 2 frames (T = len(range(0, {T_raw}, {skip})))")

    scenario_pairs = list(itertools.product(beta_c_values, diffc_values))
    n_scenarios = len(scenario_pairs)

    configs = [
        _ScenarioConfig(
            scenario_index=scenario_index,
            n_scenarios=n_scenarios,
            beta_c=float(beta_c),
            diffc=float(diffc),
            run_seed=[int(seed), scenario_index],
            init_field_args=init_field_args,
            max_runs=max_runs_per_scenario,
            outdir=outdir,
            ncol=ncol,
            nlay=nlay,
            lx=lx,
            lz=lz,
            total_time=total_time,
            nstp=nstp,
            hk=float(hk),
            por=float(por),
            al=al,
            at=at,
            rho0=rho0,
            c_sea=c_sea,
            head_bc=head_bc,
            T=T,
            skip=skip,
            dt=dt,
            dt_eff=dt_eff,
            overwrite=overwrite,
            save_timeseries=save_timeseries,
            save_modflow_files=save_modflow_files,
            exe_name=exe_name,
        )
        for scenario_index, (beta_c, diffc) in enumerate(scenario_pairs, start=1)
    ]

    cpu_count = os.cpu_count() or 1
    effective_workers = min(n_scenarios, cpu_count) if n_workers is None else n_workers
    effective_workers = max(1, effective_workers)

    print(
        f"Starting generation: scenarios={n_scenarios}  workers={effective_workers}  "
        f"T_raw={T_raw}  skip={skip}  T={T}  T_out={T - 1}  head_bc={head_bc}"
    )

    all_summaries: list[dict] = []
    all_failures:  list[dict] = []

    if effective_workers == 1:
        for cfg in configs:
            result = _run_scenario(cfg)
            all_summaries.append(result["scenario_summary"])
            all_failures.extend(result["run_failures"])
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            for cfg in configs:
                futures[pool.submit(_run_scenario, cfg)] = cfg.scenario_index

            completed = 0
            for fut in as_completed(futures):
                completed += 1
                idx = futures[fut]
                try:
                    result = fut.result()
                    all_summaries.append(result["scenario_summary"])
                    all_failures.extend(result["run_failures"])
                    print(f"[{completed:03d}/{n_scenarios:03d}] scenario_{idx:03d} finished")
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

    all_summaries.sort(key=lambda s: s["scenario_index"])

    manifest = {
        "workflow": "simplified_henry_3d_dataset",
        "pde": {
            "description": (
                "Simplified Henry density-driven convection: elliptic pressure + "
                "parabolic solute transport, zero storage, zero influx, "
                "C = 0 on all boundaries."
            ),
            "Ss": 0.0,
            "influx": 0.0,
            "concentration_bc": "C=0 on all sides",
            "head_bc": (
                "h=0 on left/right, no-flow top/bottom" if head_bc == "lr"
                else "h=0 on all sides"
            ),
            "hk": float(hk),
            "por": float(por),
            "rho0": float(rho0),
        },
        "init_field_args": init_field_args,
        "c0_max": float(c_sea),
        "input_channel_names":  list(INPUT_CHANNEL_NAMES),
        "output_channel_names": list(OUTPUT_CHANNEL_NAMES),
        "grid":  {"ncol": ncol, "nlay": nlay, "lx": lx, "lz": lz},
        "time":  {"total_time": total_time, "nstp": nstp, "dt": dt},
        "dispersion": {"al": al, "at": at},
        "T":        int(T),
        "T_out":    int(T - 1),
        "skip":     int(skip),
        "dt_raw":   float(dt),
        "dt_eff":   float(dt_eff),
        "seed":     int(seed),
        "n_workers": effective_workers,
        "input_shape":  [len(INPUT_CHANNEL_NAMES),  T - 1, nlay, ncol],
        "output_shape": [len(OUTPUT_CHANNEL_NAMES), T - 1, nlay, ncol],
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
