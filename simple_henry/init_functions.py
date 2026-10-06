import numpy as np
import gstools as gs
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def sample_field_2d(
    res: int | tuple[int, int] | list[int] | None = None,
    res_x: int | None = None,
    res_y: int | None = None,
    num_modes_x: int = 10,
    num_modes_y: int = 10,
    degree: int = 5,
    len_scale: float = 0.5,
    var: float = 1.0,
    smoothness: float = 0.5,
    amplitude: float = 1.0,
    type: str = "grf",
    seed: int | None = None,
    taper: bool = True,
    period: float = 1.0,
):
    """
    Parameters
    ----------
    res : int or tuple/list of (int, int), optional
        Grid resolution. If single int, used for both axes (res_x = res_y = res).
        If tuple/list (res_x, res_y), specifies resolution independently per axis.
        Default is 512 if neither res nor (res_x, res_y) is specified.
    res_x, res_y : int, optional
        Independent grid resolution along x and y axes. Overrides `res` for the respective axis.
    num_modes_x, num_modes_y : int
        Number of spectral modes (for Fourier) or degrees/density controls for other samplers.
    smoothness : float
        Single smoothness control:
        • Fourier    : spectral decay exponent (higher ⇒ smoother).
        • Polynomial : degree = int(smoothness) if sampler_type='polynomial'.
        • GRBF       : Gaussian bump width σ = 1/smoothness.
        • GRF types  : correlation length ℓ = smoothness.
        • Perlin     : base noise frequency = smoothness.
    amplitude : float
        Global amplitude or standard deviation for all sampler types.
    type : str
        One of: "fourier_1d", "fourier", "polynomial", "grbf", "white_noise", "perlin", "grf", "matern", "exp".
    seed : int or None
        Seed for the gstools random generator ("grf", "matern", "exp"). None draws a fresh field.
    taper : bool
        "grf" only: multiply by a Hann envelope that is zero on every edge (zero-Dirichlet
        compatible ICs, as in simple_henry). Set False for problems without zero BCs.
    period : float
        Periodicity of the gstools Fourier fields ("grf", "matern", "exp") relative to the
        unit domain. 1.0 makes opposite edges match; > 1 (e.g. 2.0) samples a window of a
        larger periodic field, removing that artificial edge correlation.

    Returns
    -------
    forcing_func : callable
        A function f(x) that accepts a tuple / array‑dict of coordinates with x[0] = x‑coords, x[1] = y‑coords  (each
        flat), and returns f values at those points. For 'fourier_1d', returns a function of x[0] (1D).
    """
    if res_x is not None or res_y is not None:
        default_res = 512 if res is None else (res[0] if isinstance(res, (tuple, list, np.ndarray)) else int(res))
        rx = default_res if res_x is None else int(res_x)
        ry = default_res if res_y is None else int(res_y)
    elif res is not None:
        if isinstance(res, (tuple, list, np.ndarray)):
            rx, ry = int(res[0]), int(res[1])
        else:
            rx = ry = int(res)
    else:
        rx = ry = 512

    def _srf(cov):
        # Fourier-mode spacing is 2π/period, so mode_no scales with the period to keep
        # the spectral cutoff (π·mode_no/period) fixed; gstools needs an even mode_no.
        modes = [r if period == 1.0 else 2 * int(np.ceil(r * period / 2)) for r in (rx, ry)]
        return gs.SRF(cov, generator="Fourier", period=[float(period)] * 2, mode_no=modes, seed=seed)

    # Fourier-series generator
    def _fourier():
        n = np.arange(1, num_modes_x + 1)[:, None]
        m = np.arange(1, num_modes_y + 1)[:, None]
        sigma = 1.0 / (1.0 + n**2 + m.T**2) ** (smoothness / 2)
        coeffs = np.random.randn(num_modes_x, num_modes_y) * sigma
        def f(x):
            xx = x[0].reshape(1, -1)
            yy = x[1].reshape(1, -1)
            sinx = np.sin(np.pi * n * xx)
            siny = np.sin(np.pi * m * yy)
            out = np.einsum('ij,ik,jk->k', coeffs, sinx, siny)
            return amplitude * out
        return f

    # Random polynomial generator
    def _polynomial():
        coeffs = np.random.randn(degree+1, degree+1) * amplitude
        def f(x):
            xx, yy = x[0], x[1]
            out = np.zeros_like(xx)
            for i in range(degree+1):
                for j in range(degree+1):
                    out += coeffs[i,j] * (xx**i) * (yy**j)
            return out
        return f

    # Gaussian radial basis function generator
    def _grbf():
        centers = np.random.rand(degree, 2)
        amps = np.random.randn(degree) * amplitude
        def f(x):
            xx, yy = x[0], x[1]
            out = np.zeros_like(xx)
            for (cx,cy), amp in zip(centers, amps):
                out += amp * np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*smoothness**2))
            return out
        return f

    # White noise generator
    def _white_noise():
        def f(x):
            return amplitude * np.random.randn(*x[0].shape)
        return f

    # Perlin noise generator
    def _perlin():
        try:
            from noise import pnoise2
        except ImportError:
            raise ImportError("Perlin noise requires the 'noise' package")
        def f(x):
            xx, yy = x[0], x[1]
            vec = np.vectorize(lambda xi, yi: pnoise2(xi/smoothness, yi/smoothness))
            return amplitude * vec(xx, yy)
        return f

    # Matérn GRF
    def _matern():
        cov = gs.Matern(dim=2, var=var, len_scale=len_scale, nu=smoothness)
        srf = _srf(cov)
        nx = rx + 1
        ny = ry + 1
        grid_x = np.linspace(0.0, 1.0, nx)
        grid_y = np.linspace(0.0, 1.0, ny)
        field = srf.structured([grid_x, grid_y]).astype(np.float64)

        field -= field.mean()

        # Bilinear Interpolation
        def _bilinear_eval(F, xg, yg):
            x = np.clip(xg, 0.0, nx - 1.0)
            y = np.clip(yg, 0.0, ny - 1.0)
            x0 = np.floor(x).astype(int)
            y0 = np.floor(y).astype(int)
            x1 = np.minimum(x0 + 1, nx - 1)
            y1 = np.minimum(y0 + 1, ny - 1)
            tx = x - x0
            ty = y - y0

            f00 = F[x0, y0]
            f10 = F[x1, y0]
            f01 = F[x0, y1]
            f11 = F[x1, y1]
            return ((1 - tx) * (1 - ty) * f00 +
                    tx * (1 - ty) * f10 +
                    (1 - tx) * ty * f01 +
                    tx * ty * f11)

        def f(x):
            xi = x[0] * (nx - 1)
            yi = x[1] * (ny - 1)
            return _bilinear_eval(field, xi, yi)

        return f

    # Exponential GRF
    def _exp():
        cov = gs.Exponential(
            dim=2,
            var=var,
            len_scale=len_scale,
        )
        srf = _srf(cov)
        nx = rx + 1
        ny = ry + 1
        grid_x = np.linspace(0.0, 1.0, nx)
        grid_y = np.linspace(0.0, 1.0, ny)
        field = srf.structured([grid_x, grid_y])
        def f(x):
            xi = np.clip((x[0]*(nx-1)).astype(int), 0, nx-1)
            yi = np.clip((x[1]*(ny-1)).astype(int), 0, ny-1)
            return field[xi, yi]
        return f

    # GRF
    def _grf(boundary_width: float = 0.1):
        """Gaussian Random Field with smooth zero-boundary envelope.

        The raw GRF is multiplied by a 2-D raised-cosine (Hann) taper that
        is exactly zero at every boundary edge and rises smoothly to 1.0 in
        the interior.  This guarantees zero concentration at the boundaries
        while preserving the spatial correlation structure of the GRF in the
        interior.

        Parameters
        ----------
        boundary_width : float
            Width of the taper zone as a fraction of the domain [0, 1].
            ``0.1`` means the field fades to zero over the outer 10% on each
            side.  Smaller values → sharper transition; larger values → wider
            fade.

        The envelope is skipped when the outer ``taper`` is False.
        """
        cov = gs.Gaussian(
            dim=2,
            var=var,
            len_scale=len_scale,
        )
        srf = _srf(cov)
        nx = rx + 1
        ny = ry + 1
        grid_x = np.linspace(0.0, 1.0, nx)
        grid_y = np.linspace(0.0, 1.0, ny)
        field = srf.structured([grid_x, grid_y])

        # --- Build 2-D raised-cosine (Hann) boundary envelope ---------------
        # For each axis, a 1-D taper is:
        #   w(t) = 0.5 * (1 - cos(pi * t / bw))   for t in [0, bw]
        #   w(t) = 1.0                              for t in [bw, 1-bw]
        #   w(t) = 0.5 * (1 - cos(pi*(1-t) / bw)) for t in [1-bw, 1]
        # The 2-D envelope is the outer product: W(x, y) = w_x(x) * w_y(y).
        bw = float(np.clip(boundary_width, 1e-6, 0.5))

        def _hann_taper_1d(coords: np.ndarray) -> np.ndarray:
            t = coords  # coords in [0, 1]
            w = np.ones_like(t)
            # Left taper: [0, bw]
            left = t < bw
            w[left] = 0.5 * (1.0 - np.cos(np.pi * t[left] / bw))
            # Right taper: [1-bw, 1]
            right = t > (1.0 - bw)
            w[right] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - t[right]) / bw))
            return w

        wx = _hann_taper_1d(grid_x)   # (nx,)
        wy = _hann_taper_1d(grid_y)   # (ny,)
        envelope = np.outer(wx, wy)    # (nx, ny) — outer product gives 2-D taper

        # --- Shift raw GRF to be non-negative before tapering ----------------
        # A GRF is a zero-mean process, so roughly half its values are negative.
        # Subtracting the global minimum translates the whole field upward so
        # that min(field) = 0.  Spatial covariance structure is preserved exactly
        # (a global shift changes nothing about relative differences).
        # After this shift every value is >= 0, and the taper (∈ [0, 1]) then
        # keeps them >= 0: boundary edges → 0, interior → positive.
        field = field - field.min()

        # Apply envelope to the non-negative GRF grid
        if taper:
            field = field * envelope


        def f(x):
            xi = np.clip((x[0] * (nx - 1)).astype(int), 0, nx - 1)
            yi = np.clip((x[1] * (ny - 1)).astype(int), 0, ny - 1)
            return field[xi, yi]

        return f


    samplers = {
        'fourier': _fourier,
        'polynomial': _polynomial,
        'grbf': _grbf,
        'white_noise': _white_noise,
        'perlin': _perlin,
        'grf': _grf,
        'matern': _matern,
        'exp': _exp,
    }

    if type.lower() not in samplers:
        raise ValueError(f"Unknown sampler type '{type}'")

    return samplers[type.lower()]()


def display_field(f, plot_res: int | tuple[int, int] = 256, title: str = "Sampled field"):
    """Show f on [0,1]^2 as an interactive 3D surface and a heatmap."""
    if isinstance(plot_res, (tuple, list)):
        rx, ry = int(plot_res[0]), int(plot_res[1])
    else:
        rx = ry = int(plot_res)

    grid_x = np.linspace(0.0, 1.0, rx)
    grid_y = np.linspace(0.0, 1.0, ry)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = f((X.ravel(), Y.ravel())).reshape(ry, rx)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "heatmap"}]],
        subplot_titles=("3D surface", "Heatmap"),
    )
    fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=Z, colorscale="Viridis", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(x=grid_x, y=grid_y, z=Z, colorscale="Viridis"), row=1, col=2)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"),
    )
    fig.update_yaxes(scaleanchor="x", row=1, col=2)
    fig.show(renderer="browser")


if __name__ == "__main__":
    field_type = "grf"
    var = 0.6
    # len_scale controls the spatial correlation length relative to the [0,1]^2 domain.
    # len_scale=1.0 means the correlation length equals the entire domain, producing a
    # nearly flat field where the boundary taper dominates the visual appearance.
    # Use a value in [0.1, 0.5] to see meaningful GRF spatial variation alongside the taper.
    len_scale = 0.8
    res_x = 40
    res_y = 80
    f = sample_field_2d(res=(res_x, res_y), type=field_type, var=var, len_scale=len_scale)
    display_field(f, title=f"Sampled field ({field_type}, len_scale={len_scale})")