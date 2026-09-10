"""1D radial (cylindrical) compressible Euler solver with a Joule-heating
source term.

Models hydrodynamic expansion of the spark/arc discharge channel into the
surrounding ambient gas -- the most significant documented limitation of the
0D lumped reactor (see MODELING_NOTES.md, "No channel expansion"): a fixed,
non-expanding parcel has no way to cool by doing PdV work on its
surroundings, which the 0D model worked around with an artificial
temperature ceiling. This solver lets the channel actually push outward.

Governing equations, cylindrical (r) symmetry:

    dU/dt + dF/dr = S_geom(U) + S_heat(r, t)

    U = [rho, rho*u, E]                 conserved variables
    F = [rho*u, rho*u^2 + p, u*(E + p)] flux
    S_geom = -F / r                     cylindrical divergence source
    S_heat = [0, 0, q(r, t)]            volumetric Joule heating (energy eq. only)

Numerics: a first-order Godunov finite-volume scheme with the HLLC
approximate Riemann solver (Toro, "Riemann Solvers and Numerical Methods for
Fluid Dynamics", 3rd ed., ch. 10), explicit forward-Euler time stepping
under a CFL condition, a reflective boundary at r=0 (axis of symmetry), and
a transmissive (zero-gradient outflow) boundary at r=R_max.

Two deliberate simplifications, both natural next steps rather than
correctness bugs:
  * The gas is a calorically-perfect ideal gas (constant gamma, R) --
    decoupled from the 0D model's Cantera real-gas thermochemistry.
  * Joule heating reuses the 0D model's own computed electron->gas elastic
    energy-transfer rate q_eM(t) (see physics.electron_heavy_energy_exchange
    and postprocess.to_dataframe's "q_eM" column) as a *tabulated* source
    term, rather than re-deriving an independent estimate. This matters:
    the discharge's raw electrical power rho_res*j^2 is absorbed by the
    electron population first (which has a tiny heat capacity, so it can
    reach extreme Te almost instantly without implying much heat reaches
    the gas); q_eM is what the 0D model computed *actually* flows on to
    the heavy gas, at the rate set by electron-neutral collisions. Using
    the raw electrical power directly as a gas heat source (an earlier
    version of this module did exactly that) is not just quantitatively
    off but qualitatively wrong -- it skips the very buffering mechanism
    that keeps real gas from heating as fast as the electrons do, and (not
    coincidentally) blows up numerically. This is still one-way coupling
    (the hydro solver borrows the 0D solution's heating profile; it does
    not feed channel expansion back into the 0D electron kinetics).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate as sp_interpolate

GAMMA = 1.4         # ratio of specific heats (diatomic ideal gas, ~ air)
R_SPECIFIC = 287.0  # specific gas constant [J/(kg K)] (~ air)


# ---------------------------------------------------------------------------
# State conversions
# ---------------------------------------------------------------------------

def primitive_to_conserved(rho, u, p, gamma: float = GAMMA):
    """[rho, u, p] -> [rho, rho*u, E]."""
    E = p / (gamma - 1.0) + 0.5 * rho * u ** 2
    return np.array([rho, rho * u, E])


def conserved_to_primitive(U, gamma: float = GAMMA):
    """[rho, rho*u, E] -> (rho, u, p), all broadcasting over arrays."""
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u ** 2)
    return rho, u, p


def sound_speed(rho, p, gamma: float = GAMMA):
    return np.sqrt(gamma * p / rho)


def euler_flux(rho, u, p, gamma: float = GAMMA):
    E = p / (gamma - 1.0) + 0.5 * rho * u ** 2
    return np.array([rho * u, rho * u ** 2 + p, u * (E + p)])


# ---------------------------------------------------------------------------
# HLLC approximate Riemann solver (Toro ch. 10), vectorized over interfaces
# ---------------------------------------------------------------------------

def hllc_flux(rhoL, uL, pL, rhoR, uR, pR, gamma: float = GAMMA):
    """HLLC numerical flux at an array of interfaces, given left/right
    primitive states on each side. Returns an array shaped (3, n_interfaces)."""
    cL = sound_speed(rhoL, pL, gamma)
    cR = sound_speed(rhoR, pR, gamma)

    # Davis wave-speed estimates for the outer (fastest) waves.
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)

    # HLLC contact-wave speed estimate (Toro eq. 10.37).
    num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)
    den = rhoL * (SL - uL) - rhoR * (SR - uR)
    S_star = num / den

    FL = euler_flux(rhoL, uL, pL, gamma)
    FR = euler_flux(rhoR, uR, pR, gamma)
    UL = primitive_to_conserved(rhoL, uL, pL, gamma)
    UR = primitive_to_conserved(rhoR, uR, pR, gamma)

    def star_state(rho, u, p, E, S, S_star):
        coef = rho * (S - u) / (S - S_star)
        rho_star = coef
        mom_star = coef * S_star
        E_star = coef * (E / rho + (S_star - u) * (S_star + p / (rho * (S - u))))
        return np.array([rho_star, mom_star, E_star])

    U_starL = star_state(rhoL, uL, pL, UL[2], SL, S_star)
    U_starR = star_state(rhoR, uR, pR, UR[2], SR, S_star)

    F_starL = FL + SL * (U_starL - UL)
    F_starR = FR + SR * (U_starR - UR)

    flux = np.empty_like(FL)
    n = rhoL.shape[0]
    for k in range(3):
        flux[k] = np.select(
            [SL >= 0.0, (SL < 0.0) & (S_star >= 0.0), (S_star < 0.0) & (SR >= 0.0), SR < 0.0],
            [FL[k], F_starL[k], F_starR[k], FR[k]],
        )
    return flux


# ---------------------------------------------------------------------------
# Grid, ambient state, and heating configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridParams:
    n_cells: int = 400
    r_max: float = 5.0e-3          # outer domain radius [m]
    geometry: str = "cylindrical"  # "cylindrical" (radial channel) or "planar" (validation)

    @property
    def dr(self) -> float:
        return self.r_max / self.n_cells

    def cell_centers(self) -> np.ndarray:
        return (np.arange(self.n_cells) + 0.5) * self.dr


@dataclass(frozen=True)
class AmbientState:
    T0: float = 300.0
    p0: float = 101325.0

    @property
    def rho0(self) -> float:
        return self.p0 / (R_SPECIFIC * self.T0)


@dataclass
class HeatingParams:
    """Volumetric Joule heating confined to r <= r_channel, following a
    *tabulated* q_eM(t) profile imported from a 0D reactor solution (see
    module docstring for why this -- not the raw electrical power -- is the
    physically appropriate source term for gas heating)."""

    # Channel radius matching the 0D model's arc cross-section A_arc=1 mm^2
    # (r = sqrt(A_arc/pi)).
    r_channel: float
    t_profile: np.ndarray  # times [s] of the tabulated q_eM(t), ascending
    q_profile: np.ndarray  # q_eM(t) [W/m^3] at those times

    @classmethod
    def from_0d_dataframe(cls, df, r_channel: float = (1.0e-6 / np.pi) ** 0.5) -> "HeatingParams":
        """Build from a `postprocess.to_dataframe(...)` result (needs its
        "t" and "q_eM" columns)."""
        return cls(r_channel=r_channel, t_profile=df["t"].to_numpy(), q_profile=df["q_eM"].to_numpy())

    def power_density(self, r: np.ndarray, t: float) -> np.ndarray:
        """Volumetric Joule heating [W/m^3] at radius r, time t."""
        q_t = np.interp(t, self.t_profile, self.q_profile,
                         left=self.q_profile[0], right=self.q_profile[-1])
        return np.where(r <= self.r_channel, q_t, 0.0)


# ---------------------------------------------------------------------------
# Boundary conditions and RHS
# ---------------------------------------------------------------------------

def apply_boundary_conditions(U: np.ndarray, geometry: str) -> np.ndarray:
    """Returns U padded with one ghost cell on each side."""
    Ug = np.empty((3, U.shape[1] + 2))
    Ug[:, 1:-1] = U
    if geometry == "cylindrical":
        # Reflective axis at r=0: mirror density/energy, negate velocity.
        Ug[0, 0] = U[0, 0]
        Ug[1, 0] = -U[1, 0]
        Ug[2, 0] = U[2, 0]
    else:
        # Planar validation case (Sod tube): reflective walls at both ends.
        Ug[0, 0] = U[0, 0]
        Ug[1, 0] = -U[1, 0]
        Ug[2, 0] = U[2, 0]
    # Outer boundary: transmissive / zero-gradient outflow.
    Ug[:, -1] = U[:, -1]
    return Ug


def compute_rhs(U: np.ndarray, r: np.ndarray, dr: float, grid: GridParams,
                 heating: HeatingParams | None, t: float, gamma: float = GAMMA) -> np.ndarray:
    Ug = apply_boundary_conditions(U, grid.geometry)
    rho, u, p = conserved_to_primitive(Ug, gamma)

    flux = hllc_flux(rho[:-1], u[:-1], p[:-1], rho[1:], u[1:], p[1:], gamma)

    if grid.geometry == "cylindrical":
        # Integrate (1/r) d(r F)/dr exactly over each cell using the face
        # radii, rather than subtracting a point-evaluated F(r_i)/r_i source
        # term. The two are equivalent in the continuum limit, but the
        # point-source form amplifies any small residual at the innermost
        # cell by ~1/r_0 (very large, since r_0 = dr/2), which is numerically
        # fragile there; the face-integrated form instead multiplies the
        # axis face's flux by r=0 exactly, so it can never blow up at r=0
        # regardless of what that face's flux value is.
        n = grid.n_cells
        r_faces = np.arange(n + 1) * dr  # r_faces[0] = 0 (the axis), r_faces[-1] = r_max
        dUdt = -(r_faces[1:] * flux[:, 1:] - r_faces[:-1] * flux[:, :-1]) / (dr * r)
        # The momentum flux F_mom = rho*u^2 + p is what the mass/energy-style
        # (1/r)d(rF)/dr treatment above implicitly assumes for *all* three
        # equations -- but the true cylindrical radial-momentum equation is
        # d(rho*u)/dt + (1/r)d(r*rho*u^2)/dr + dp/dr = 0, i.e. only the
        # convective part (rho*u^2) picks up the (1/r)d(r*)/dr divergence;
        # the pressure term is a plain gradient dp/dr, not (1/r)d(rp)/dr.
        # Applying the divergence treatment to the pressure part too (as the
        # flux-differencing above does, since it can't tell momentum's flux
        # apart from mass/energy's) silently adds a spurious p/r term --
        # (1/r)d(rp)/dr = dp/dr + p/r, so it overshoots by exactly p/r. Add
        # it back here. (This is the standard cylindrical/spherical
        # "geometric source term" split, e.g. Toro ch. 17: H(U) in the -H/r
        # source uses rho*u^2 for momentum, not rho*u^2+p.) Without this
        # correction, even perfectly uniform, at-rest gas experiences a
        # fictitious inward force -p/r from the discretization alone.
        rho_c, u_c, p_c = conserved_to_primitive(U, gamma)
        dUdt[1] = dUdt[1] + p_c / r
    else:
        dUdt = -(flux[:, 1:] - flux[:, :-1]) / dr

    if heating is not None:
        q = heating.power_density(r, t)
        dUdt[2] = dUdt[2] + q

    return dUdt


def max_wave_speed(U: np.ndarray, gamma: float = GAMMA) -> float:
    rho, u, p = conserved_to_primitive(U, gamma)
    c = sound_speed(rho, p, gamma)
    return float(np.max(np.abs(u) + c))


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

@dataclass
class HydroResult:
    t: list
    r: np.ndarray
    snapshots: dict  # t_saved -> (rho, u, p) arrays


def _heating_time_scale(U: np.ndarray, r: np.ndarray, heating: HeatingParams | None,
                         t: float, safety: float = 0.2) -> float:
    """Explicit-source stability constraint: cap dt so the volumetric Joule
    heating cannot add more than `safety` times a heated cell's current
    energy density in a single step. The CFL condition alone only bounds
    advective transport; it says nothing about a strong local energy
    source, and a current step (e.g. the 50 A -> 2.5 kA spark->arc jump,
    a 2500x jump in power density since heating ~ I^2) can otherwise
    overshoot into an unphysical state before the (much slower to adapt)
    hydrodynamic CFL dt has a chance to react."""
    if heating is None:
        return np.inf
    q = heating.power_density(r, t)
    heated = q > 0.0
    if not np.any(heated):
        return np.inf
    E = U[2, heated]
    return safety * float(np.min(E / q[heated]))


def _clamp_temperature(U: np.ndarray, T_max: float, gamma: float = GAMMA) -> np.ndarray:
    """Cap internal energy so no cell's translational temperature exceeds
    T_max, leaving density and momentum untouched.

    This ideal-gas model (fixed gamma=1.4, no dissociation/ionization) has
    no mechanism to represent real air's large high-temperature heat
    capacity: above ~a few 1e4 K, a real gas dissociates and ionizes,
    absorbing enormous additional energy per degree with comparatively
    little further temperature rise. Without that sink, dumping the raw
    electrical power density straight into translational energy (with no
    intermediary electron heat capacity, unlike the 0D model) drives T
    without bound and the sound speed with it, which is both unphysical
    for this gas model and numerically catastrophic (the CFL time step
    collapses toward zero while temperature keeps climbing). Capping T is
    the same documented stand-in the 0D reactor uses for its own thermo
    ceiling -- a placeholder for missing high-temperature physics, not a
    claim that real gas stops heating at T_max.
    """
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u ** 2)
    T = p / (rho * R_SPECIFIC)
    over = T > T_max
    if np.any(over):
        p_capped = np.where(over, rho * R_SPECIFIC * T_max, p)
        U = U.copy()
        U[2] = p_capped / (gamma - 1.0) + 0.5 * rho * u ** 2
    return U


def run(
    grid: GridParams,
    U0: np.ndarray,
    t_end: float,
    heating: HeatingParams | None = None,
    cfl: float = 0.4,
    heating_safety: float = 0.2,
    T_max: float = 2.0e4,
    save_times: np.ndarray | None = None,
    gamma: float = GAMMA,
) -> HydroResult:
    """Explicit forward-Euler / HLLC finite-volume integration from t=0 to
    t_end. Returns density/velocity/pressure snapshots at `save_times`
    (defaults to a handful of log-ish spaced times). `T_max` caps
    translational temperature (see `_clamp_temperature`)."""
    r = grid.cell_centers()
    dr = grid.dr
    U = U0.copy()
    t = 0.0

    if save_times is None:
        save_times = np.linspace(0.0, t_end, 6)[1:]
    save_times = np.sort(np.asarray(save_times))
    snapshots = {}
    save_idx = 0

    n_steps = 0
    max_steps = 5_000_000
    while t < t_end and n_steps < max_steps:
        wave_speed = max_wave_speed(U, gamma)
        dt = cfl * dr / max(wave_speed, 1e-6)
        dt = min(dt, _heating_time_scale(U, r, heating, t, heating_safety))
        if save_idx < len(save_times):
            dt = min(dt, save_times[save_idx] - t)
        dt = min(dt, t_end - t)
        if dt <= 0.0:
            dt = min(cfl * dr / max(wave_speed, 1e-6), t_end - t)

        k1 = compute_rhs(U, r, dr, grid, heating, t, gamma)
        U_mid = U + dt * k1
        rho_mid, _, p_mid = conserved_to_primitive(U_mid, gamma)
        if not np.all(np.isfinite(U_mid)) or np.any(rho_mid <= 0.0) or np.any(p_mid <= 0.0):
            raise RuntimeError(
                f"Non-physical state (NaN/Inf, rho<=0, or p<=0) at t={t:.3e}s, dt={dt:.3e}s, "
                f"step {n_steps} -- reduce CFL/heating_safety or refine the grid."
            )
        U = _clamp_temperature(U_mid, T_max, gamma)
        t += dt
        n_steps += 1

        if save_idx < len(save_times) and t >= save_times[save_idx] - 1e-15:
            rho_s, u_s, p_s = conserved_to_primitive(U, gamma)
            snapshots[float(save_times[save_idx])] = (rho_s.copy(), u_s.copy(), p_s.copy())
            save_idx += 1

    if n_steps >= max_steps:
        raise RuntimeError(f"Exceeded max_steps={max_steps} without reaching t_end={t_end:.3e}s")

    return HydroResult(t=list(snapshots.keys()), r=r, snapshots=snapshots)


# ---------------------------------------------------------------------------
# Core density trace: the first step toward feeding channel expansion back
# into the 0D reactor (currently one-way -- the 0D model still runs as a
# fixed-volume parcel; see MODELING_NOTES.md's "no self-consistent feedback"
# limitation). Reduces a HydroResult down to eta(t) = rho_core(t)/rho_ambient,
# the density-reduction factor an expanding-volume 0D reactor variant could
# use to relax its own fixed-volume assumption.
# ---------------------------------------------------------------------------

@dataclass
class CoreDensityTrace:
    t: np.ndarray            # [s], ascending
    rho_core: np.ndarray     # area-weighted average density within r<=r_channel [kg/m^3]
    eta: np.ndarray          # rho_core(t) / rho_ambient
    rho_ambient: float
    r_channel: float


def core_density_trace(result: HydroResult, r_channel: float, rho_ambient: float) -> CoreDensityTrace:
    """Area-weighted average density within r <= r_channel at each of
    `result`'s saved snapshots, and the resulting density-reduction factor
    eta(t) = rho_core(t) / rho_ambient.

    `r_channel` and `rho_ambient` must be passed explicitly (matching the
    values the hydro run actually used -- e.g. `HeatingParams.r_channel` and
    `AmbientState.rho0`) rather than inferred from the data: assuming
    r_channel equals some default, or that the first saved snapshot already
    equals the true ambient density, are both silent sources of error if
    save times or channel geometry change.
    """
    t = np.array(sorted(result.t))
    core_mask = result.r <= r_channel
    if not np.any(core_mask):
        raise ValueError(f"No grid cells fall within r_channel={r_channel:.3e} m (dr={result.r[1]-result.r[0]:.3e} m)")
    r_core = result.r[core_mask]
    dr = result.r[1] - result.r[0]
    area_weights = 2.0 * np.pi * r_core * dr
    total_area = np.sum(area_weights)

    rho_core = np.array([
        np.sum(result.snapshots[ti][0][core_mask] * area_weights) / total_area
        for ti in t
    ])
    eta = rho_core / rho_ambient
    return CoreDensityTrace(t=t, rho_core=rho_core, eta=eta, rho_ambient=rho_ambient, r_channel=r_channel)


def _eta_pchip(trace: CoreDensityTrace) -> sp_interpolate.PchipInterpolator:
    """Shared PCHIP fit used by both eta_interpolator and eta_rate_interpolator.

    Plain not-a-knot cubic splines (`interp1d(kind="cubic")`, an earlier
    version of this function) can overshoot between sparse, sharply-curved
    points -- exactly this trace's shape (eta can fall >50% between
    consecutive saved snapshots during the fast early expansion). Verified
    on real CH4 trace data: the plain cubic spline is *not* monotonic
    (produces a spurious local increase in eta despite every sample
    decreasing), while PCHIP -- built specifically to preserve the
    monotonicity of the data it's fit to -- is monotonic everywhere and
    gives a physically-sensible (always non-positive, since heating only
    ever rarefies the core in this model) derivative. `extrapolate=False`
    because we handle the flat-hold region ourselves, explicitly, below.
    """
    return sp_interpolate.PchipInterpolator(trace.t, trace.eta, extrapolate=False)


def eta_interpolator(trace: CoreDensityTrace):
    """eta(t), flat-extrapolated outside the trace's time range (holds the
    first/last computed value) -- e.g. eta(t) = 1 for any t before the
    first saved snapshot, since nothing has expanded yet at that point."""
    pchip = _eta_pchip(trace)
    t_min, t_max = float(trace.t[0]), float(trace.t[-1])

    def f(t):
        t_arr = np.clip(np.asarray(t, dtype=float), t_min, t_max)
        return pchip(t_arr)

    return f


def eta_rate_interpolator(trace: CoreDensityTrace):
    """d(eta)/dt, as the *exact* analytic derivative of the same monotonic
    PCHIP interpolant `eta_interpolator` evaluates -- not a finite
    difference. A finite-difference derivative needs a step size, and
    there is no single good choice here: too large (e.g. 1e-6 s tried
    initially) and it can straddle across the t_min flat-extrapolation
    boundary and fabricate a nonzero rate where the true one is exactly
    zero; too small relative to the actual sample spacing (as fine as
    ~4e-7 s between the first two snapshots here) and it just measures
    roundoff. Using the interpolant's own derivative sidesteps the
    step-size question entirely. Returns exactly 0 outside the trace's
    time range, consistent with eta_interpolator holding eta constant
    there (a constant has zero rate of change, not an undefined one)."""
    pchip = _eta_pchip(trace)
    deriv = pchip.derivative()
    t_min, t_max = float(trace.t[0]), float(trace.t[-1])

    def f(t):
        t_arr = np.asarray(t, dtype=float)
        inside = (t_arr >= t_min) & (t_arr <= t_max)
        t_clamped = np.clip(t_arr, t_min, t_max)
        return np.where(inside, deriv(t_clamped), 0.0)

    return f


def expansion_cooling_terms(t: float, trace: CoreDensityTrace, eta_fn=None, eta_rate_fn=None):
    """eta(t), d(eta)/dt, and the dilatation rate -(1/rho)(drho/dt) =
    -(1/eta)(d(eta)/dt) at time t, for coupling channel expansion into a
    0D gas energy balance (see `gas_energy_rate_with_expansion`).

    Pass `eta_fn`/`eta_rate_fn` (from `eta_interpolator`/`eta_rate_interpolator`)
    if calling this many times, to avoid rebuilding the PCHIP fit on every call.
    """
    eta_fn = eta_fn or eta_interpolator(trace)
    eta_rate_fn = eta_rate_fn or eta_rate_interpolator(trace)
    eta = float(eta_fn(t))
    deta_dt = float(eta_rate_fn(t))
    dilatation_rate = -(1.0 / eta) * deta_dt if eta > 0.0 else 0.0
    return eta, deta_dt, dilatation_rate


def gas_energy_rate_with_expansion(
    t: float, T_g: float, q_eM: float, c_v: float, gamma: float,
    trace: CoreDensityTrace, eta_fn=None, eta_rate_fn=None,
):
    """dT_g/dt for a gas parcel of fixed mass but time-varying volume
    (rho(t) = rho_ambient*eta(t)), heated at volumetric rate q_eM, derived
    from the first law dU/dt = Qdot - p dV/dt with U=m*c_v*T:

        c_v dT/dt = q_eM/rho - (gamma-1)*T*[-(1/rho)(drho/dt)]

    i.e. Joule heating minus adiabatic expansion (PdV) cooling. Returns
    (dT_g_dt, rho_current). This *replaces* the raw electrical heating
    channel used elsewhere with the same rate-limited q_eM(t) the 0D model
    itself computes (see HeatingParams docstring) -- q_eM here should be
    that same quantity, not rho_res*j**2.
    """
    eta, _, dilatation_rate = expansion_cooling_terms(t, trace, eta_fn, eta_rate_fn)
    rho_current = trace.rho_ambient * eta
    dT_dt_joule = q_eM / (rho_current * c_v)
    dT_dt_expansion = -(gamma - 1.0) * T_g * dilatation_rate
    return dT_dt_joule + dT_dt_expansion, rho_current
