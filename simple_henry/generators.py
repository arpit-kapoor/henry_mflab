"""Dataset generator for the simplified Henry problem.

Generates windowed (input, output) training tensors from MODFLOW 6
time-series output of the simplified density-driven convection problem.

Input channels (no boundary flux channels — all BCs are homogeneous Dirichlet):
    - concentration_t  : solute concentration field at time t
    - head_t           : hydraulic head field at time t
    - beta_c           : solutal expansion coefficient (constant field)
    - diffc            : effective diffusion coefficient (constant field)

Output channels:
    - concentration at t + lag
    - head at t + lag
"""
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

from .simulation import build_and_run_simple_henry

# ---------------------------------------------------------------------------
# Input channel specification
# ---------------------------------------------------------------------------
INPUT_CHANNEL_NAMES = (
    "concentration_t",
    "head_t",
    "beta_c",
    "diffc",
)
INPUT_CHANNEL_INDEX = {name: idx for idx, name in enumerate(INPUT_CHANNEL_NAMES)}

REQUIRED_RUN_FILES = {"windows.npz"}

# Default initial head for all runs (consistent with Dirichlet p|∂Ω = 0)
STANDARD_INIT_HEAD = 0.0
# Default initial concentration — uniform, salt-saturated [kg/m³]
STANDARD_INIT_CONCENTRATION = 35.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _broadcast_channel(value, nlay, ncol):
    """Return a (nlay, ncol) array filled with ``value``."""
    return np.full((nlay, ncol), float(value), dtype=float)


def _valid_window_indices(n_times: int, lag: int) -> np.ndarray:
    if lag <= 0:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if n_times <= lag:
        return np.empty((0,), dtype=int)
    return np.arange(0, n_times - lag, dtype=int)


def _prune_run_workspace(run_dir: Path, keep_files: set):
    """Delete files in *run_dir* that are not in *keep_files*."""
    for child in run_dir.iterdir():
        if child.name in keep_files:
            continue
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child)


def _build_splits(ids, train_frac: float, val_frac: float, seed: int) -> dict:
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and train_frac + val_frac < 1.0):
        raise ValueError("train/val fractions must be in (0,1) and sum to < 1")
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(np.floor(train_frac * n))
    n_val   = int(np.floor(val_frac * n))
    return {
        "train": ids[:n_train],
        "val":   ids[n_train : n_train + n_val],
        "test":  ids[n_train + n_val :],
    }


def _scenario_tag(beta_c: float, diffc: float) -> str:
    return f"scenario_beta{beta_c:.3f}_diffc{diffc:.5f}"


def _run_tag(run_index: int, params: dict) -> str:
    return (
        f"run_{run_index:06d}_"
        f"hk{params['hk']:.2f}_"
        f"por{params['por']:.3f}"
    )


# ---------------------------------------------------------------------------
# Window tensor builder
# ---------------------------------------------------------------------------

def _build_window_tensors(
    head_ts: np.ndarray,
    conc_ts: np.ndarray,
    lag: int,
    nlay: int,
    ncol: int,
    params: dict,
):
    """Build (input, output) window tensors from full time-series arrays.

    Parameters
    ----------
    head_ts, conc_ts : ndarray of shape (n_times, nlay, ncol)
    lag : int
        Number of time steps ahead to predict.
    params : dict
        Must contain ``beta_c`` and ``diffc``.

    Returns
    -------
    dict with keys:
        input_tensor   : float32 array of shape (n_windows, 4, nlay, ncol)
        output_tensor  : float32 array of shape (n_windows, 2, nlay, ncol)
        t_index        : int array of shape (n_windows,)
        t_lag_index    : int array of shape (n_windows,)
    or None if there are no valid windows.
    """
    t_indices = _valid_window_indices(head_ts.shape[0], lag)
    n_windows = int(t_indices.size)
    if n_windows == 0:
        return None

    n_channels = len(INPUT_CHANNEL_NAMES)
    input_tensor  = np.empty((n_windows, n_channels, nlay, ncol), dtype=np.float32)
    output_tensor = np.empty((n_windows, 2, nlay, ncol), dtype=np.float32)

    # Static channels: broadcast scalar parameters over the spatial domain
    beta_field  = _broadcast_channel(params["beta_c"], nlay, ncol)
    diffc_field = _broadcast_channel(params["diffc"],  nlay, ncol)

    input_tensor[:, INPUT_CHANNEL_INDEX["beta_c"], :, :] = beta_field
    input_tensor[:, INPUT_CHANNEL_INDEX["diffc"],  :, :] = diffc_field

    for i, t in enumerate(t_indices):
        t_lag = t + lag
        input_tensor[i, INPUT_CHANNEL_INDEX["concentration_t"]] = conc_ts[t]
        input_tensor[i, INPUT_CHANNEL_INDEX["head_t"]]          = head_ts[t]

        output_tensor[i, 0] = conc_ts[t_lag]
        output_tensor[i, 1] = head_ts[t_lag]

    return {
        "input_tensor":  input_tensor,
        "output_tensor": output_tensor,
        "t_index":       t_indices,
        "t_lag_index":   t_indices + lag,
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
    # Initial concentration — scalar or (nlay, ncol) array
    C0=STANDARD_INIT_CONCENTRATION,
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
    # Spatially varying K fields (override scalar hk/vk; disables hk sweep)
    hk_field=None,
    vk_field=None,
    # Reference density
    rho0: float = 1000.0,
    # Dataset controls
    lag: int = 1,
    overwrite: bool = False,
    max_runs_per_scenario: int | None = None,
    save_timeseries: bool = False,
    save_modflow_files: bool = False,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    exe_name: str = "mf6",
):
    """Generate a windowed dataset for the simplified Henry problem.

    For each (beta_c, diffc) *scenario* and each (hk, por) *run* combination
    within that scenario, one MODFLOW 6 simulation is run, time-series outputs
    are sliced into overlapping (input_t, output_{t+lag}) windows, and the
    result is saved as ``windows.npz`` alongside a ``manifest.json``.

    Parameters
    ----------
    outdir : str or Path
        Root output directory.  Will be created if it does not exist.
    beta_c_values, diffc_values, hk_values, por_values : list of float
        Parameter values to sweep.  All four lists are combined as a full
        Cartesian product.  Pass singleton lists to fix a parameter.
    C0 : float or array_like of shape (nlay, ncol)
        Initial concentration field.  A scalar is broadcast uniformly.
    lag : int
        Prediction lag in time steps.
    overwrite : bool
        If False (default) skip runs whose ``windows.npz`` already exists.
    max_runs_per_scenario : int or None
        Cap on total run combinations per scenario (useful for quick tests).
    save_timeseries : bool
        If True, include full head/conc time-series arrays in ``windows.npz``.
    save_modflow_files : bool
        If True, keep all MODFLOW 6 workspace files.  If False, prune to
        ``windows.npz`` only.
    seed : int
        Random seed for train/val/test split.
    train_frac, val_frac : float
        Fractions of windows assigned to train and val splits.
    exe_name : str or Path
        MODFLOW 6 executable name or path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if hk_field is not None and len(hk_values) != 1:
        raise ValueError("hk sweep is incompatible with --kappa-file; provide one hk value")
    if vk_field is not None and len(hk_values) != 1:
        raise ValueError("vk from kappa file is incompatible with hk sweep")

    dt = total_time / nstp  # time step size [days]

    global_window_ids: list[str] = []
    scenarios_summary: list[dict] = []
    run_records:       list[dict] = []
    run_failures:      list[dict] = []

    scenario_pairs = list(itertools.product(beta_c_values, diffc_values))

    for scenario_index, (beta_c, diffc) in enumerate(scenario_pairs, start=1):
        scenario_tag = _scenario_tag(beta_c, diffc)
        scenario_dir = outdir / scenario_tag
        scenario_dir.mkdir(parents=True, exist_ok=True)

        run_combinations = list(itertools.product(hk_values, por_values))
        if max_runs_per_scenario is not None:
            run_combinations = run_combinations[:max_runs_per_scenario]

        print(
            f"Scenario [{scenario_index:03d}/{len(scenario_pairs):03d}] "
            f"{scenario_tag} with {len(run_combinations)} run(s)"
        )

        scenario_runs:     list[dict] = []
        scenario_failures: list[dict] = []

        for run_index, (hk, por) in enumerate(run_combinations, start=1):
            params = {
                "beta_c":  float(beta_c),
                "diffc":   float(diffc),
                "hk":      float(hk),
                "por":     float(por),
                "al":      float(al),
                "at":      float(at),
                "rho0":    float(rho0),
                "C0_scalar": float(C0) if np.ndim(C0) == 0 else None,
            }

            run_tag = _run_tag(run_index, params)
            run_dir = scenario_dir / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)
            sample_file = run_dir / "windows.npz"

            if sample_file.exists() and not overwrite:
                if not save_modflow_files:
                    _prune_run_workspace(run_dir, REQUIRED_RUN_FILES)
                record = {
                    "id":        f"{scenario_tag}/{run_tag}",
                    "scenario":  scenario_tag,
                    "run":       run_tag,
                    "workspace": str(run_dir),
                    "status":    "skipped",
                    **params,
                }
                run_records.append(record)
                scenario_runs.append(record)
                continue

            print(f"  [{run_index:04d}/{len(run_combinations):04d}] RUN  {run_tag}")
            try:
                head_ts, conc_ts, times = build_and_run_simple_henry(
                    workspace=run_dir,
                    ncol=ncol,
                    nlay=nlay,
                    Lx=lx,
                    Lz=lz,
                    total_time=total_time,
                    nstp=nstp,
                    C0=C0,
                    por=por,
                    hk=hk,
                    vk=hk,   # isotropic by default; vk_field override available
                    al=al,
                    at=at,
                    diffc=diffc,
                    beta_c=beta_c,
                    rho0=rho0,
                    hk_field=hk_field,
                    vk_field=vk_field,
                    return_timeseries=True,
                    exe_name=exe_name,
                )

                windowed = _build_window_tensors(
                    head_ts=head_ts,
                    conc_ts=conc_ts,
                    lag=lag,
                    nlay=nlay,
                    ncol=ncol,
                    params=params,
                )
                if windowed is None:
                    raise ValueError(
                        f"No valid windows for lag={lag}; "
                        f"available time steps={head_ts.shape[0]}"
                    )

                window_ids = [
                    f"{scenario_tag}/{run_tag}/w{int(t):05d}"
                    for t in windowed["t_index"].tolist()
                ]
                global_window_ids.extend(window_ids)

                payload = {
                    "input_tensor":         windowed["input_tensor"],
                    "output_tensor":        windowed["output_tensor"],
                    "input_channel_names":  np.asarray(list(INPUT_CHANNEL_NAMES)),
                    "t_index":              windowed["t_index"],
                    "t_lag_index":          windowed["t_lag_index"],
                    "time_t":               times[windowed["t_index"]],
                    "time_t_lag":           times[windowed["t_lag_index"]],
                    "window_ids":           np.asarray(window_ids),
                    "lag":                  int(lag),
                    "lag_days":             float(lag * dt),
                    "dt":                   float(dt),
                    "grid": {
                        "ncol": ncol,
                        "nlay": nlay,
                        "lx":   lx,
                        "lz":   lz,
                    },
                    "total_time": total_time,
                    "nstp":       nstp,
                    **params,
                }
                if save_timeseries:
                    payload["head_timeseries"] = head_ts
                    payload["conc_timeseries"] = conc_ts
                    payload["times"]           = times

                np.savez_compressed(sample_file, **payload)
                if not save_modflow_files:
                    _prune_run_workspace(run_dir, REQUIRED_RUN_FILES)

                record = {
                    "id":        f"{scenario_tag}/{run_tag}",
                    "scenario":  scenario_tag,
                    "run":       run_tag,
                    "workspace": str(run_dir),
                    "status":    "ok",
                    "n_windows": int(windowed["input_tensor"].shape[0]),
                    "input_shape": list(windowed["input_tensor"].shape[1:]),
                    "output_shape": list(windowed["output_tensor"].shape[1:]),
                    **params,
                }
                run_records.append(record)
                scenario_runs.append(record)

            except Exception as exc:
                failure = {
                    "id":        f"{scenario_tag}/{run_tag}",
                    "scenario":  scenario_tag,
                    "run":       run_tag,
                    "workspace": str(run_dir),
                    "status":    "failed",
                    "error":     str(exc),
                    **params,
                }
                run_failures.append(failure)
                scenario_failures.append(failure)
                print(f"    FAILED: {exc}")

        # Scenario-level manifest
        scenario_manifest = {
            "scenario":        scenario_tag,
            "beta_c":          float(beta_c),
            "diffc":           float(diffc),
            "lag":             int(lag),
            "lag_days":        float(lag * dt),
            "dt":              float(dt),
            "n_total_runs":    len(run_combinations),
            "n_ok_runs":       sum(r["status"] == "ok"      for r in scenario_runs),
            "n_skipped_runs":  sum(r["status"] == "skipped" for r in scenario_runs),
            "n_failed_runs":   len(scenario_failures),
            "runs":            scenario_runs,
            "failures":        scenario_failures,
        }
        with (scenario_dir / "scenario_manifest.json").open("w", encoding="utf-8") as fp:
            json.dump(scenario_manifest, fp, indent=2)

        scenarios_summary.append({
            "scenario":        scenario_tag,
            "beta_c":          float(beta_c),
            "diffc":           float(diffc),
            "n_total_runs":    len(run_combinations),
            "n_ok_runs":       scenario_manifest["n_ok_runs"],
            "n_skipped_runs":  scenario_manifest["n_skipped_runs"],
            "n_failed_runs":   scenario_manifest["n_failed_runs"],
        })

    splits = (
        _build_splits(global_window_ids, train_frac, val_frac, seed)
        if global_window_ids
        else {"train": [], "val": [], "test": []}
    )

    manifest = {
        "workflow": "simple_henry_windowed_dataset",
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
        "input_channel_names": list(INPUT_CHANNEL_NAMES),
        "grid": {"ncol": ncol, "nlay": nlay, "lx": lx, "lz": lz},
        "time": {"total_time": total_time, "nstp": nstp, "dt": dt},
        "dispersion": {"al": al, "at": at},
        "lag":        int(lag),
        "lag_days":   float(lag * dt),
        "train_frac": train_frac,
        "val_frac":   val_frac,
        "n_scenarios":      len(scenario_pairs),
        "n_total_runs":     sum(s["n_total_runs"]   for s in scenarios_summary),
        "n_ok_runs":        sum(s["n_ok_runs"]       for s in scenarios_summary),
        "n_skipped_runs":   sum(s["n_skipped_runs"]  for s in scenarios_summary),
        "n_failed_runs":    sum(s["n_failed_runs"]   for s in scenarios_summary),
        "n_total_windows":  len(global_window_ids),
        "scenarios":        scenarios_summary,
        "splits":           splits,
        "runs":             run_records,
        "failures":         run_failures,
    }

    with (outdir / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(
        "Generation done: "
        f"scenarios={manifest['n_scenarios']} "
        f"runs_ok={manifest['n_ok_runs']} "
        f"runs_failed={manifest['n_failed_runs']} "
        f"windows={manifest['n_total_windows']}"
    )
    print(f"Manifest: {outdir / 'manifest.json'}")
