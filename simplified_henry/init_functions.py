"""Gaussian random field (GRF) sampler for the initial concentration C₀."""
import numpy as np
import gstools as gs


def sample_field_2d(
    res: int | tuple[int, int],
    len_scale: float = 0.5,
    var: float = 1.0,
    seed: int | None = None,
    taper: bool = True,
    period: float = 1.0,
    boundary_width: float = 0.1,
):
    """Sample a non-negative GRF on the unit square.

    Parameters
    ----------
    res : int or (int, int)
        Grid resolution (res_x, res_y); the field is sampled on a
        (res_x + 1) × (res_y + 1) grid of nodes spanning [0, 1]².
    len_scale, var : float
        Length scale (in normalised coordinates) and variance of the Gaussian
        covariance model.
    seed : int or None
        Seed for the gstools random generator. None draws a fresh field.
    taper : bool
        Multiply by a raised-cosine (Hann) envelope that is zero on every edge
        (zero-Dirichlet compatible ICs, simplified problem). Set False for
        problems without zero concentration BCs (classical problem).
    period : float
        Periodicity of the gstools Fourier field relative to the unit domain.
        1.0 makes opposite edges match; > 1 (e.g. 2.0) samples a window of a
        larger periodic field, removing that artificial edge correlation.
    boundary_width : float
        Width b of the taper zone as a fraction of the domain.

    Returns
    -------
    f : callable
        ``f((x, y))`` with flat arrays of normalised coordinates in [0, 1]
        returns the shifted GRF at the nearest lower grid node, multiplied by
        the taper evaluated at the query coordinates themselves.

    Notes
    -----
    The raw GRF F is shifted to be non-negative, then tapered:
    ``(F - min F) · w_b(x) · w_b(y)``, with
    ``w_b(t) = ½[1 - cos(π t / b)]`` for ``t < b``, 1 in the interior and
    ``½[1 - cos(π (1 - t) / b)]`` for ``t > 1 - b``.
    The taper is evaluated at the query points rather than at the grid nodes,
    so it is symmetric between opposite edges.
    """
    if isinstance(res, (tuple, list, np.ndarray)):
        rx, ry = int(res[0]), int(res[1])
    else:
        rx = ry = int(res)

    # Fourier-mode spacing is 2π/period, so mode_no scales with the period to keep
    # the spectral cutoff (π·mode_no/period) fixed; gstools needs an even mode_no.
    modes = [r if period == 1.0 else 2 * int(np.ceil(r * period / 2)) for r in (rx, ry)]
    cov = gs.Gaussian(dim=2, var=var, len_scale=len_scale)
    srf = gs.SRF(cov, generator="Fourier", period=[float(period)] * 2, mode_no=modes, seed=seed)

    nx, ny = rx + 1, ry + 1
    grid_x = np.linspace(0.0, 1.0, nx)
    grid_y = np.linspace(0.0, 1.0, ny)
    field = srf.structured([grid_x, grid_y])

    # Shift to be non-negative (covariance structure is unchanged)
    field = field - field.min()

    bw = float(np.clip(boundary_width, 1e-6, 0.5))

    def _hann_taper_1d(t: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        w = np.ones_like(t)
        left = t < bw
        w[left] = 0.5 * (1.0 - np.cos(np.pi * t[left] / bw))
        right = t > (1.0 - bw)
        w[right] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - t[right]) / bw))
        return w

    def f(x):
        xi = np.clip((x[0] * (nx - 1)).astype(int), 0, nx - 1)
        yi = np.clip((x[1] * (ny - 1)).astype(int), 0, ny - 1)
        values = field[xi, yi]
        if taper:
            values = values * _hann_taper_1d(x[0]) * _hann_taper_1d(x[1])
        return values

    return f
