import numpy as np


def to_layer_col_field(value, nlay, ncol, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((nlay, ncol), float(arr), dtype=float)
    if arr.shape != (nlay, ncol):
        raise ValueError(f"{name} must have shape ({nlay}, {ncol}), got {arr.shape}")
    return arr


def load_kappa_fields(kappa_file):
    data = np.load(kappa_file)
    if "hk" not in data:
        raise ValueError(f"kappa file {kappa_file} must contain 'hk'")
    hk_field = np.asarray(data["hk"], dtype=float)
    vk_field = np.asarray(data["vk"], dtype=float) if "vk" in data else hk_field.copy()
    return hk_field, vk_field


def latin_hypercube_1d(low, high, n, rng):
    if n <= 0:
        return np.empty((0,), dtype=float)
    if high < low:
        raise ValueError(f"invalid range [{low}, {high}]")
    if high == low:
        return np.full((n,), float(low), dtype=float)
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.uniform(cut[:-1], cut[1:])
    rng.shuffle(u)
    return low + (high - low) * u


def boundary_masks(nlay, ncol, left_value, right_value):
    left = np.zeros((nlay, ncol), dtype=float)
    right = np.zeros((nlay, ncol), dtype=float)
    left[:, 0] = float(left_value)
    right[:, -1] = float(right_value)
    return left, right


def build_fno_io_tensors(
    nlay,
    ncol,
    strt_head,
    strt_conc,
    inflow,
    right_bc_scalar,
    conc_final,
    head_final,
):
    c0 = to_layer_col_field(strt_conc, nlay, ncol, "strt_conc")
    h0 = to_layer_col_field(strt_head, nlay, ncol, "strt_head")
    left_mask, right_mask = boundary_masks(nlay, ncol, inflow, right_bc_scalar)
    input_tensor = np.stack([c0, h0, left_mask, right_mask], axis=0)
    output_tensor = np.stack([conc_final, head_final], axis=0)
    return input_tensor, output_tensor


def build_splits(ids, train_frac, val_frac, seed):
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and train_frac + val_frac < 1.0):
        raise ValueError("train/val fractions must be in (0,1) and sum to < 1")
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def parse_float_csv(values):
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def parse_scenario_pairs(values):
    pairs = []
    for raw_pair in values.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                "scenario pairs must be comma-separated 'beta:diffc' values, "
                f"got '{pair}'"
            )
        beta_str, diffc_str = pair.split(":", 1)
        pairs.append((float(beta_str.strip()), float(diffc_str.strip())))
    if not pairs:
        raise ValueError("at least one scenario pair is required")
    return pairs


def build_coupling_diffusion_scenario_pairs(
    beta_min,
    beta_max,
    beta_count,
    diffc_min,
    diffc_max,
    diffc_count,
    fixed_beta,
    fixed_diffc,
):
    beta_count = int(beta_count)
    diffc_count = int(diffc_count)
    if beta_count <= 0 or diffc_count <= 0:
        raise ValueError("beta_count and diffc_count must be >= 1")

    beta_vals = np.linspace(float(beta_min), float(beta_max), beta_count)
    diffc_vals = np.linspace(float(diffc_min), float(diffc_max), diffc_count)

    # Build the full Cartesian grid of (beta_c, diffc) pairs.
    pairs = [(float(beta), float(diffc)) for beta in beta_vals for diffc in diffc_vals]
    return pairs


def valid_window_indices(n_times, lag):
    if lag <= 0:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if n_times <= lag:
        return np.empty((0,), dtype=int)
    return np.arange(0, n_times - lag, dtype=int)
