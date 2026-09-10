import numpy as np
import pytest

from src import radial_hydro as rh
from src import exact_riemann as er


# ---------------------------------------------------------------------------
# Sod shock tube: validates the HLLC Riemann solver / finite-volume update
# against the exact Euler solution, independent of the cylindrical geometric
# source term or heating (geometry="planar", heating=None).
# ---------------------------------------------------------------------------

def _run_sod(n_cells=200, cfl=0.4):
    grid = rh.GridParams(n_cells=n_cells, r_max=1.0, geometry="planar")
    x = grid.cell_centers()
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    rho0 = np.where(x < 0.5, rhoL, rhoR)
    u0 = np.where(x < 0.5, uL, uR)
    p0 = np.where(x < 0.5, pL, pR)
    U0 = rh.primitive_to_conserved(rho0, u0, p0)

    t_end = 0.2
    result = rh.run(grid, U0, t_end, heating=None, cfl=cfl, T_max=1e12, save_times=np.array([t_end]))
    rho_num, u_num, p_num = result.snapshots[t_end]
    rho_ex, u_ex, p_ex = er.sod_exact_solution(x, t_end, rhoL, uL, pL, rhoR, uR, pR, x0=0.5, gamma=1.4)
    return rho_num, u_num, p_num, rho_ex, u_ex, p_ex


def test_sod_shock_tube_matches_exact_solution():
    rho_num, u_num, p_num, rho_ex, u_ex, p_ex = _run_sod()
    assert np.all(np.isfinite(rho_num)) and np.all(np.isfinite(p_num))
    # First-order HLLC at 200 cells: expect O(1e-2) L1 error (numerical
    # diffusion smears the contact discontinuity and shock front over a
    # handful of cells), comfortably above machine/roundoff scale but far
    # below a "the solver is wrong" scale.
    assert np.mean(np.abs(rho_num - rho_ex)) < 0.03
    assert np.mean(np.abs(u_num - u_ex)) < 0.05
    assert np.mean(np.abs(p_num - p_ex)) < 0.03


def test_sod_density_and_pressure_stay_positive_and_ordered():
    rho_num, u_num, p_num, *_ = _run_sod()
    assert np.all(rho_num > 0.0)
    assert np.all(p_num > 0.0)
    # Density must stay within the initial left/right bounds (no
    # under/overshoot beyond the physical range for this test problem).
    assert rho_num.max() <= 1.05
    assert rho_num.min() >= 0.125 * 0.95


def test_geometric_source_is_null_for_uniform_at_rest_state():
    """Regression test for a real derivation bug: naively wrapping the full
    momentum flux (rho*u^2 + p) inside the (1/r) d(r*F)/dr divergence adds a
    spurious p/r inward force even to perfectly uniform, at-rest gas. The
    correct cylindrical radial-momentum source excludes pressure from the
    divergence-treated part (see radial_hydro.compute_rhs)."""
    grid = rh.GridParams(n_cells=100, r_max=1.0e-3, geometry="cylindrical")
    r = grid.cell_centers()
    rho0 = np.full_like(r, 1.2)
    u0 = np.zeros_like(r)
    p0 = np.full_like(r, 101325.0)
    U0 = rh.primitive_to_conserved(rho0, u0, p0)
    dUdt = rh.compute_rhs(U0, r, grid.dr, grid, heating=None, t=0.0)
    # The cancellation (flux terms ~p/dr ~ 1e10 scale) is exact analytically;
    # atol here just needs to clear double-precision roundoff at that scale,
    # not assert bit-exact zero.
    assert np.allclose(dUdt, 0.0, atol=1e-2)


def test_axis_cell_stable_with_strong_localized_heating():
    """Regression test for the axis-cell instability caused by evaluating
    the geometric source as a point value F(r_0)/r_0 (which amplifies any
    residual by ~1/r_0, very large at the innermost cell). Uses a strong,
    step-function heating pulse concentrated well within the channel to
    stress the innermost cells specifically."""
    grid = rh.GridParams(n_cells=200, r_max=2.0e-3, geometry="cylindrical")
    r = grid.cell_centers()
    ambient = rh.AmbientState(T0=300.0, p0=101325.0)
    rho0 = np.full_like(r, ambient.rho0)
    u0 = np.zeros_like(r)
    p0 = np.full_like(r, ambient.p0)
    U0 = rh.primitive_to_conserved(rho0, u0, p0)

    r_channel = 2.0e-4
    heating = rh.HeatingParams(
        r_channel=r_channel,
        t_profile=np.array([0.0, 1e-9, 1e-6]),
        q_profile=np.array([1e11, 1e11, 1e11]),
    )
    result = rh.run(grid, U0, 2.0e-7, heating=heating, cfl=0.4, T_max=2.0e4,
                     save_times=np.array([5e-8, 1e-7, 2e-7]))
    for rho_s, u_s, p_s in result.snapshots.values():
        assert np.all(np.isfinite(rho_s)) and np.all(np.isfinite(u_s)) and np.all(np.isfinite(p_s))
        assert np.all(rho_s > 0.0)
        assert np.all(p_s > 0.0)


def test_temperature_clamp_caps_at_T_max():
    grid = rh.GridParams(n_cells=50, r_max=1.0e-3, geometry="cylindrical")
    r = grid.cell_centers()
    rho0 = np.full_like(r, 1.2)
    u0 = np.zeros_like(r)
    p0 = np.full_like(r, 101325.0)
    U0 = rh.primitive_to_conserved(rho0, u0, p0)
    heating = rh.HeatingParams(
        r_channel=grid.r_max,
        t_profile=np.array([0.0, 1e-6]),
        q_profile=np.array([1e18, 1e18]),  # absurdly large, to force the clamp
    )
    result = rh.run(grid, U0, 1e-9, heating=heating, cfl=0.4, T_max=2.0e4,
                     save_times=np.array([1e-9]))
    rho_s, u_s, p_s = result.snapshots[1e-9]
    T_s = p_s / (rho_s * rh.R_SPECIFIC)
    assert np.all(T_s <= 2.0e4 * (1.0 + 1e-8))


def test_ambient_gas_stays_at_rest_far_from_channel():
    """Cells well outside the heated channel should be essentially
    undisturbed on a short timescale (the blast hasn't reached them yet)."""
    grid = rh.GridParams(n_cells=400, r_max=5.0e-3, geometry="cylindrical")
    r = grid.cell_centers()
    ambient = rh.AmbientState(T0=300.0, p0=101325.0)
    rho0 = np.full_like(r, ambient.rho0)
    u0 = np.zeros_like(r)
    p0 = np.full_like(r, ambient.p0)
    U0 = rh.primitive_to_conserved(rho0, u0, p0)

    heating = rh.HeatingParams(
        r_channel=5.64e-4,
        t_profile=np.array([0.0, 1e-7, 1e-6]),
        q_profile=np.array([1e10, 1e10, 1e10]),
    )
    result = rh.run(grid, U0, 2e-7, heating=heating, cfl=0.4, save_times=np.array([2e-7]))
    rho_s, u_s, p_s = result.snapshots[2e-7]
    far_field = r > 4.0e-3  # far from the r_channel=0.564mm heated region
    assert np.allclose(rho_s[far_field], ambient.rho0, rtol=1e-3)
    assert np.allclose(u_s[far_field], 0.0, atol=1.0)


# ---------------------------------------------------------------------------
# core_density_trace / eta_interpolator
# ---------------------------------------------------------------------------

def _heated_channel_result(save_times):
    grid = rh.GridParams(n_cells=400, r_max=5.0e-3, geometry="cylindrical")
    r = grid.cell_centers()
    ambient = rh.AmbientState(T0=300.0, p0=101325.0)
    rho0 = np.full_like(r, ambient.rho0)
    u0 = np.zeros_like(r)
    p0 = np.full_like(r, ambient.p0)
    U0 = rh.primitive_to_conserved(rho0, u0, p0)
    r_channel = 5.641895835477562e-4
    heating = rh.HeatingParams(
        r_channel=r_channel,
        t_profile=np.array([0.0, 1e-6, 5e-5]),
        q_profile=np.array([2e10, 2e10, 2e10]),
    )
    result = rh.run(grid, U0, save_times[-1], heating=heating, cfl=0.4, save_times=save_times)
    return result, ambient, r_channel


def test_core_density_trace_starts_at_eta_one_and_decreases_under_heating():
    save_times = np.array([1e-9, 1e-6, 1e-5, 3e-5])
    result, ambient, r_channel = _heated_channel_result(save_times)
    trace = rh.core_density_trace(result, r_channel=r_channel, rho_ambient=ambient.rho0)

    assert trace.eta[0] == pytest.approx(1.0, rel=1e-3)  # heating has barely acted by t=1ns
    # sustained heating should monotonically rarefy the core over this window
    assert np.all(np.diff(trace.eta) < 0.0)
    assert np.all(trace.eta > 0.0)


def test_core_density_trace_rejects_r_channel_smaller_than_grid_spacing():
    save_times = np.array([1e-6])
    result, ambient, _ = _heated_channel_result(save_times)
    with pytest.raises(ValueError):
        rh.core_density_trace(result, r_channel=1e-9, rho_ambient=ambient.rho0)


def test_eta_interpolator_matches_samples_and_clamps_outside_range():
    save_times = np.array([1e-9, 1e-6, 1e-5, 3e-5])
    result, ambient, r_channel = _heated_channel_result(save_times)
    trace = rh.core_density_trace(result, r_channel=r_channel, rho_ambient=ambient.rho0)
    f = rh.eta_interpolator(trace)

    for ti, eta_i in zip(trace.t, trace.eta):
        assert f(ti) == pytest.approx(eta_i, rel=1e-8)
    assert f(-1.0) == pytest.approx(trace.eta[0])
    assert f(1.0) == pytest.approx(trace.eta[-1])


def test_core_density_trace_is_sensitive_to_r_channel_choice():
    """Regression check for the original bug report: using a wrong/mismatched
    r_channel (as opposed to the one the heating actually used) silently
    changes which cells are averaged and thus the resulting eta(t)."""
    save_times = np.array([1e-6, 3e-5])
    result, ambient, r_channel = _heated_channel_result(save_times)
    trace_correct = rh.core_density_trace(result, r_channel=r_channel, rho_ambient=ambient.rho0)
    trace_too_small = rh.core_density_trace(result, r_channel=r_channel * 0.7, rho_ambient=ambient.rho0)
    assert not np.allclose(trace_correct.eta, trace_too_small.eta)


# ---------------------------------------------------------------------------
# eta_rate_interpolator / expansion_cooling_terms / gas_energy_rate_with_expansion
#
# Regression tests for two real bugs found in a hand-rolled finite-difference
# version of this coupling: (1) a fixed eps=1e-6 s straddled the interpolator's
# t_min flat-extrapolation boundary and fabricated a large nonzero derivative
# at t=0, where the true rate is exactly zero; (2) even a properly-small eps
# didn't fix it, because the underlying eta(t) fit (a plain not-a-knot cubic
# spline through only 8 sparse, sharply-curved points) is not itself
# monotonic -- it overshoots between samples, so *any* derivative of it,
# finite-difference or exact, can have the wrong sign. PCHIP fixes both: it
# preserves the monotonicity of the data, and its derivative is exact (no
# step-size choice at all).
# ---------------------------------------------------------------------------

def test_eta_rate_is_zero_before_first_snapshot():
    save_times = np.array([1e-7, 1e-6, 1e-5, 5e-5])
    result, ambient, r_channel = _heated_channel_result(save_times)
    trace = rh.core_density_trace(result, r_channel=r_channel, rho_ambient=ambient.rho0)
    rate_fn = rh.eta_rate_interpolator(trace)
    assert rate_fn(0.0) == 0.0
    assert rate_fn(save_times[0] * 0.5) == 0.0


def test_eta_is_monotonically_non_increasing_under_sustained_heating():
    """PCHIP must not overshoot: eta(t) should never tick back up between
    samples given monotonically-decreasing sample values (sustained
    heating only ever rarefies the core further in this model)."""
    save_times = np.array([1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 2e-5, 3.5e-5, 5e-5])
    result, ambient, r_channel = _heated_channel_result(save_times)
    trace = rh.core_density_trace(result, r_channel=r_channel, rho_ambient=ambient.rho0)
    assert np.all(np.diff(trace.eta) <= 0.0)  # the samples themselves are monotonic here

    eta_fn = rh.eta_interpolator(trace)
    t_dense = np.linspace(trace.t[0], trace.t[-1], 500)
    eta_dense = eta_fn(t_dense)
    assert np.all(np.diff(eta_dense) <= 1e-12)


def test_eta_rate_matches_analytic_derivative_sign_of_monotonic_decrease():
    save_times = np.array([1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 2e-5, 3.5e-5, 5e-5])
    result, ambient, r_channel = _heated_channel_result(save_times)
    trace = rh.core_density_trace(result, r_channel=r_channel, rho_ambient=ambient.rho0)
    rate_fn = rh.eta_rate_interpolator(trace)
    t_dense = np.linspace(trace.t[0], trace.t[-1], 500)
    # eta only ever decreases in this scenario, so its rate must never be positive.
    assert np.all(rate_fn(t_dense) <= 1e-8)


def test_gas_energy_rate_with_expansion_matches_hand_derivation():
    """dT/dt = q/(rho*cv) - (gamma-1)*T*dilatation_rate, from the first law
    dU/dt = Qdot - p dV/dt for a fixed-mass, time-varying-volume parcel.
    Check against a hand-picked case with q_eM=0 (pure adiabatic expansion,
    no heating): dT/dt should reduce to exactly -(gamma-1)*T*dilatation_rate."""
    save_times = np.array([1e-7, 1e-6, 1e-5, 5e-5])
    result, ambient, r_channel = _heated_channel_result(save_times)
    trace = rh.core_density_trace(result, r_channel=r_channel, rho_ambient=ambient.rho0)

    t_probe = 1e-6
    gamma = 1.4
    c_v = 287.0 / (gamma - 1.0)
    T_g = 1000.0
    dT_dt, rho_current = rh.gas_energy_rate_with_expansion(t_probe, T_g, 0.0, c_v, gamma, trace)

    _, _, dilatation_rate = rh.expansion_cooling_terms(t_probe, trace)
    expected = -(gamma - 1.0) * T_g * dilatation_rate
    assert dT_dt == pytest.approx(expected, rel=1e-10)
    assert dilatation_rate > 0.0  # core is expanding (rarefying) here
    assert dT_dt < 0.0            # so pure expansion cooling must be negative
    assert rho_current == pytest.approx(trace.rho_ambient * float(rh.eta_interpolator(trace)(t_probe)))
