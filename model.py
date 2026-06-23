"""
Cogenetic Minkowski Functional (CMF).

Cogenesis is a residual-aware framework; CMF is its Minkowski-stage functional.
    (e.g. satellite → target line of sight through atmosphere).

Process time t indexes resolution state R(t) = (ℓ, τ, Λ, ε, Σ).
Core R is resolution, not a rename of t — t ↦ R(t) as the observation runs.

Contrast with standard calculus:
    fixed functional F[Φ]  +  bolt-on environment C, time as external input
vs this frame:
    process functional F[Φ; C, R(t)]  where C and R(t) are inside the map.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class ObserverThroughline:
    """
    Point detector on a throughline — e.g. satellite looking at surface targets.

    The throughline is the ray along which environmental density gradients define C.
    """

    position: tuple[float, float]  # (t, x) observer location
    target: tuple[float, float] = (0.0, 0.0)  # look direction terminates here


@dataclass(frozen=True)
class Context:
    """
    C: environmental density gradient along the observer throughline.

    Not declared metadata — the physical gradient between observer and targets
    that shapes how the apparatus can distinguish process from environment.
    """

    throughline: ObserverThroughline


@dataclass(frozen=True)
class ProcessTime:
    """
    Process time t — index of the running observation.

    Does not replace core resolution R. Use resolution_at_process_time() for R(t).
    """

    t: float
    tolerance: float = 1e-6
    noise_cov: float = 0.0


@dataclass(frozen=True)
class ResolutionState:
    """R(t) = (ℓ, τ, Λ, ε, Σ) at process time t."""

    ell: float
    tau: float
    bandwidth: float
    tolerance: float
    noise_cov: float


@dataclass(frozen=True)
class MetricCoeffs:
    c: float = 1.0
    a: float = 0.08
    b: float = 0.04
    c_xi: float = 0.05


@dataclass(frozen=True)
class CMFConfig:
    context: Context
    process_time: ProcessTime
    coeffs: MetricCoeffs = field(default_factory=MetricCoeffs)


@dataclass
class SpacetimeGrid:
    t: np.ndarray
    spatial: tuple[np.ndarray, ...]
    phi: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.spatial)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.phi.shape

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def dx(self) -> tuple[float, ...]:
        return tuple(float(s[1] - s[0]) for s in self.spatial)


@dataclass
class ProcessSnapshot:
    """Process functional evaluated at process time t with resolution R(t)."""

    process_time: float
    distinction_capacity: float
    context_along_line: float
    residual_integral: float
    h_residual_mean: float
    observer_value: float
    g_xx_target: float
    g_xx_vacuum: float


@dataclass
class PredictionReport:
    """
    Stipulations vs determination vs prediction on the satellite demo.

    Minkowski stipulates: |r|=0, g=η, F[Φ] with no R-map.
    CMF determines at each R: residual, g_xx correction, O(C,R).
    CMF predicts over R: falling residual, sharpening O, refinement to O_late.
    """

    process_times: list[float]
    residuals: list[float]
    cmf_observer: list[float]
    g_xx_target: list[float]
    interval_errors: list[float]
    refinement_errors: list[float]
    bolted_on_fixed: float
    late_observer: float
    late_g_xx_target: float
    late_residual: float
    minkowski_residual_claim: float = 0.0
    minkowski_g_xx_claim: float = 1.0


def _coord_grids(grid: SpacetimeGrid) -> list[np.ndarray]:
    coords = [grid.t] + list(grid.spatial)
    idx_shape = grid.shape

    def coord_grid(axis: int) -> np.ndarray:
        shape = [1] * len(idx_shape)
        shape[axis] = -1
        return np.broadcast_to(coords[axis].reshape(shape), idx_shape)

    return [coord_grid(i) for i in range(len(coords))]


def environmental_density(grid: SpacetimeGrid) -> np.ndarray:
    """
    rho_env(x): environmental density field.

    Example: atmosphere — dense near surface (x=0), thin at satellite altitude.
    """
    _tt, xx = _coord_grids(grid)[0], _coord_grids(grid)[1]
    _ = _tt
    # exponential atmosphere: density falls off with altitude |x|
    scale_height = 2.5
    return np.exp(-np.abs(xx) / scale_height)


def throughline_parameter(grid: SpacetimeGrid, throughline: ObserverThroughline) -> np.ndarray:
    """Normalised parameter s in [0,1] along observer → target ray."""
    t0, x0 = throughline.position
    t1, x1 = throughline.target
    tt, xx = _coord_grids(grid)[0], _coord_grids(grid)[1]
    dx, dy = x1 - x0, t1 - t0
    seg_len_sq = dx * dx + dy * dy + 1e-12
    s = ((xx - x0) * dx + (tt - t0) * dy) / seg_len_sq
    return np.clip(s, 0.0, 1.0)


def context_gradient_field(grid: SpacetimeGrid, context: Context) -> np.ndarray:
    """
    C(x): environmental density gradient projected onto the observer throughline.

    This is the gradient the detector sees between itself and its targets —
    not an external label bolted onto a fixed functional.
    """
    rho_env = environmental_density(grid)
    drho = _gradient(rho_env, grid.dt, grid.dx)
    t0, x0 = context.throughline.position
    t1, x1 = context.throughline.target
    dt_dir = t1 - t0
    dx_dir = x1 - x0
    norm = np.sqrt(dt_dir**2 + dx_dir**2) + 1e-12
    # C = n · ∇rho_env  along throughline direction n
    return (dt_dir / norm) * drho[0] + (dx_dir / norm) * drho[1]


def context_along_throughline(grid: SpacetimeGrid, context: Context) -> float:
    """Integrated |C| along the throughline — scalar summary of the environment."""
    c_field = context_gradient_field(grid, context)
    s = throughline_parameter(grid, context.throughline)
    on_line = s * (1.0 - s) * 4.0  # peaks on segment, zero off ends
    cell = grid.dt * np.prod(grid.dx)
    return float(np.sum(np.abs(c_field) * on_line) * cell)


def resolution_at_process_time(
    process_time: ProcessTime,
    t_min: float,
    t_max: float,
) -> ResolutionState:
    """
    t ↦ R(t) = (ℓ(t), τ(t), Λ(t), ε(t), Σ(t)).

    Process time indexes the evolving resolution state; core R is resolution, not t.
    """
    progress = np.clip((process_time.t - t_min) / (t_max - t_min + 1e-12), 0.0, 1.0)
    sigma_target = 1.0
    ell = max(4.0 * (1.0 - progress) + sigma_target, sigma_target)
    tau = max(4.0 * (1.0 - progress) + sigma_target, sigma_target)
    bandwidth = 1.0 / max(ell, 1e-12)
    return ResolutionState(
        ell=ell,
        tau=tau,
        bandwidth=bandwidth,
        tolerance=process_time.tolerance,
        noise_cov=process_time.noise_cov,
    )


def distinction_capacity(process_time: ProcessTime, t_min: float, t_max: float) -> tuple[float, float]:
    """Spatial and temporal resolution (ℓ(t), τ(t)) from R(t)."""
    r = resolution_at_process_time(process_time, t_min, t_max)
    return r.ell, r.tau


def minkowski_eta(config: CMFConfig, dim: int) -> np.ndarray:
    eta = np.eye(dim + 1, dtype=float)
    eta[0, 0] = -(config.coeffs.c**2)
    return eta


def minkowski_eta_inv(config: CMFConfig, dim: int) -> np.ndarray:
    eta_inv = np.eye(dim + 1, dtype=float)
    eta_inv[0, 0] = -1.0 / (config.coeffs.c**2)
    return eta_inv


def _gradient(field: np.ndarray, dt: float, dx: tuple[float, ...]) -> list[np.ndarray]:
    grads = [np.gradient(field, dt, axis=0)]
    for axis, h in enumerate(dx, start=1):
        grads.append(np.gradient(field, h, axis=axis))
    return grads


def _smooth(field: np.ndarray, ell: float, tau: float, dim: int) -> np.ndarray:
    sigma = [max(tau, 1e-12)] + [max(ell, 1e-12)] * dim
    real = gaussian_filter(np.real(field), sigma=sigma, mode="nearest")
    imag = gaussian_filter(np.imag(field), sigma=sigma, mode="nearest")
    return real + 1j * imag


def embed_projection(projected: np.ndarray, original: np.ndarray) -> np.ndarray:
    _ = original
    return projected


def project_phi(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid) -> np.ndarray:
    """
    Π_{C,R(t)}Φ: projection shaped by environmental gradient C and process time R.

    C weights the kernel along the throughline; R(t) sets how sharp the kernel is.
    """
    ell, tau = distinction_capacity(config.process_time, float(grid.t[0]), float(grid.t[-1]))
    c_field = np.abs(context_gradient_field(grid, config.context))
    c_norm = c_field / (np.max(c_field) + 1e-12)

    # environmental weighting: modest coupling where density gradient is steep
    weighted = phi * (1.0 + 0.25 * c_norm)
    smoothed = _smooth(weighted, ell, tau, grid.dim)

    # restrict to throughline neighbourhood — footprint sharpens on target as R grows
    s = throughline_parameter(grid, config.context.throughline)
    footprint = np.exp(-((s - 1.0) ** 2) / (2 * max(ell, 1e-6) ** 2))
    return smoothed * footprint


def residual_field(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid) -> np.ndarray:
    return phi - embed_projection(project_phi(phi, config, grid), phi)


def residual_norm_squared(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid) -> np.ndarray:
    return np.abs(residual_field(phi, config, grid)) ** 2


def residual_norm_integral(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid) -> float:
    r2 = residual_norm_squared(phi, config, grid)
    s = throughline_parameter(grid, config.context.throughline)
    on_line = s > 0.01
    cell = grid.dt * np.prod(grid.dx)
    return float(np.sum(r2[on_line]) * cell)


def rho_projected(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid) -> np.ndarray:
    return np.abs(project_phi(phi, config, grid)) ** 2


def residual_current(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid) -> list[np.ndarray]:
    return _gradient(residual_norm_squared(phi, config, grid), grid.dt, grid.dx)


def decoherence_tensor(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid) -> np.ndarray:
    dim = grid.dim
    eta = minkowski_eta(config, dim)
    eta_inv = minkowski_eta_inv(config, dim)
    pi_phi = project_phi(phi, config, grid)
    grads = _gradient(pi_phi, grid.dt, grid.dx)
    xi = np.zeros(phi.shape + (dim + 1, dim + 1), dtype=float)
    kinetic = np.zeros(phi.shape, dtype=float)
    for mu in range(dim + 1):
        for nu in range(dim + 1):
            term = grads[mu] * np.conjugate(grads[nu])
            xi[..., mu, nu] = np.real(term)
            if mu == nu:
                kinetic += eta_inv[mu, nu] * np.real(term)
    trace_part = np.einsum("ij,...->...", eta, kinetic) / (dim + 1)
    for mu in range(dim + 1):
        for nu in range(dim + 1):
            xi[..., mu, nu] -= trace_part * eta[mu, nu]
    return xi


def residual_driven_correction(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid) -> np.ndarray:
    dim = grid.dim
    jr = residual_current(phi, config, grid)
    c = config.coeffs
    h = np.zeros(phi.shape + (dim + 1, dim + 1), dtype=float)
    for mu in range(dim + 1):
        for nu in range(dim + 1):
            h[..., mu, nu] = c.b * jr[mu] * jr[nu]
    return h


def metric_correction(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid) -> np.ndarray:
    dim = grid.dim
    rho = rho_projected(phi, config, grid)
    drho = _gradient(rho, grid.dt, grid.dx)
    jr = residual_current(phi, config, grid)
    xi = decoherence_tensor(phi, config, grid)
    c = config.coeffs
    h = np.zeros(phi.shape + (dim + 1, dim + 1), dtype=float)
    for mu in range(dim + 1):
        for nu in range(dim + 1):
            h[..., mu, nu] = c.a * drho[mu] * drho[nu] + c.b * jr[mu] * jr[nu] + c.c_xi * xi[..., mu, nu]
    return h


def effective_metric(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid) -> np.ndarray:
    eta = minkowski_eta(config, grid.dim)
    h = metric_correction(phi, config, grid)
    g = np.zeros_like(h)
    for mu in range(grid.dim + 1):
        for nu in range(grid.dim + 1):
            g[..., mu, nu] = eta[mu, nu] + h[..., mu, nu]
    return g


def metric_measure_factor(g: np.ndarray) -> np.ndarray:
    det = g[..., 0, 0] * g[..., 1, 1] - g[..., 0, 1] * g[..., 1, 0]
    return np.sqrt(np.abs(det))


def config_at_process_time(
    context: Context,
    t: float,
    coeffs: MetricCoeffs | None = None,
) -> CMFConfig:
    return CMFConfig(context=context, process_time=ProcessTime(t=t), coeffs=coeffs or MetricCoeffs())


def bolted_on_functional(phi: np.ndarray, grid: SpacetimeGrid) -> float:
    """
    Standard approach: fixed functional, environment ignored.

    F[Φ] = ∫|Φ|² dμ — same at all process times, C plays no role inside the map.
    """
    cell = grid.dt * np.prod(grid.dx)
    return float(np.sum(np.abs(phi) ** 2) * cell)


def process_functional(
    phi: np.ndarray,
    config: CMFConfig,
    grid: SpacetimeGrid,
) -> float:
    """
    Process functional F[Φ; C, R(t)] — C and R are inside the map.

    At early process time R the projection is coarse; at late R it is sharp.
    C shapes weighting along the throughline throughout.
    """
    obs, _ = observer_reading(phi, config, grid)
    return obs


def _target_indices(grid: SpacetimeGrid, centre: tuple[float, float]) -> tuple[int, int]:
    ti = int(np.argmin(np.abs(grid.t - centre[0])))
    xi = int(np.argmin(np.abs(grid.spatial[0] - centre[1])))
    return ti, xi


def g_xx_at_point(phi: np.ndarray, config: CMFConfig, grid: SpacetimeGrid, centre: tuple[float, float]) -> float:
    ti, xi = _target_indices(grid, centre)
    g = effective_metric(phi, config, grid)
    return float(g[ti, xi, 1, 1])


def minkowski_interval_error(g_xx: float, eta_xx: float = 1.0) -> float:
    """Interval correction |g_xx - η_xx| — CMF determines; Minkowski stipulates zero."""
    return abs(g_xx - eta_xx)


def observer_reading(
    phi: np.ndarray,
    config: CMFConfig,
    grid: SpacetimeGrid,
    functional: Literal["rho_projected", "residual"] = "rho_projected",
    centre: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """O(C, R(t)): point detector reading at this moment of the process."""
    ell, _tau = distinction_capacity(config.process_time, float(grid.t[0]), float(grid.t[-1]))
    if centre is None:
        centre = config.context.throughline.target

    coords = _coord_grids(grid)
    dist_sq = sum((coords[i] - centre[i]) ** 2 for i in range(len(centre)))
    kernel = np.exp(-0.5 * dist_sq / max(ell, 1e-12) ** 2)
    kernel /= np.sum(kernel)

    g = effective_metric(phi, config, grid)
    weights = kernel * metric_measure_factor(g)
    weights /= np.sum(weights)

    if functional == "rho_projected":
        field = rho_projected(phi, config, grid)
    else:
        field = residual_norm_squared(phi, config, grid)

    mean = float(np.sum(weights * field))
    epsilon = float(np.std(field) / np.sqrt(np.sum(weights**2)))
    return mean, epsilon


def evolve_process(
    phi: np.ndarray,
    grid: SpacetimeGrid,
    context: Context,
    *,
    coeffs: MetricCoeffs | None = None,
    n_times: int = 20,
) -> list[ProcessSnapshot]:
    """
    Evaluate the process functional at successive process times t (each with R(t)).

    Not a tuning loop — this is the observation running, getting more accurate.
    """
    coeffs = coeffs or MetricCoeffs()
    t_vals = np.linspace(grid.t[0], grid.t[-1], n_times)
    history: list[ProcessSnapshot] = []

    for t in t_vals:
        config = config_at_process_time(context, float(t), coeffs)
        ell, _ = distinction_capacity(config.process_time, float(grid.t[0]), float(grid.t[-1]))
        residual = residual_norm_integral(phi, config, grid)
        h_res = residual_driven_correction(phi, config, grid)
        h_res_mean = float(np.sqrt(np.mean(np.sum(h_res**2, axis=(-2, -1)))))
        obs, _ = observer_reading(phi, config, grid, centre=context.throughline.target)
        g_xx_tgt = g_xx_at_point(phi, config, grid, context.throughline.target)

        ti = len(grid.t) // 2
        xi_vac = int(0.92 * len(grid.spatial[0]))
        g = effective_metric(phi, config, grid)
        g_xx_vac = float(g[ti, xi_vac, 1, 1])

        history.append(
            ProcessSnapshot(
                process_time=float(t),
                distinction_capacity=ell,
                context_along_line=context_along_throughline(grid, context),
                residual_integral=residual,
                h_residual_mean=h_res_mean,
                observer_value=obs,
                g_xx_target=g_xx_tgt,
                g_xx_vacuum=g_xx_vac,
            )
        )
    return history


def prediction_report(
    phi: np.ndarray,
    grid: SpacetimeGrid,
    context: Context,
    history: list[ProcessSnapshot],
) -> PredictionReport:
    """
    Stipulations vs determination vs prediction.

    Minkowski stipulates |r|=0 and g=η (constants in report).
    CMF determines residual, interval correction, O at each R.
    CMF predicts refinement: |O(R) - O_late| falls with process time.
    """
    late = history[-1]
    late_obs = late.observer_value
    bolted = bolted_on_functional(phi, grid)

    process_times = [s.process_time for s in history]
    residuals = [s.residual_integral for s in history]
    cmf_observer = [s.observer_value for s in history]
    g_xx_target = [s.g_xx_target for s in history]
    interval_errors = [minkowski_interval_error(g) for g in g_xx_target]
    refinement_errors = [abs(obs - late_obs) for obs in cmf_observer]

    return PredictionReport(
        process_times=process_times,
        residuals=residuals,
        cmf_observer=cmf_observer,
        g_xx_target=g_xx_target,
        interval_errors=interval_errors,
        refinement_errors=refinement_errors,
        bolted_on_fixed=bolted,
        late_observer=late_obs,
        late_g_xx_target=late.g_xx_target,
        late_residual=late.residual_integral,
    )
