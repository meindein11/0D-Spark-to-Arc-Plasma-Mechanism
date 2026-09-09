import os

import numpy as np
import cantera as ct
import pytest

from src.circuit import CircuitParams
from src.reactor import PlasmaReactor, ReactorParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRI30 = os.path.join(ROOT, "mechanisms", "gri30.yaml")
AMMONIA = os.path.join(ROOT, "mechanisms", "ammonia-CO-H2-Alzueta-2023.yaml")

# Short-duration circuit so the convergence tests run quickly; the physics
# (spark handoff at t_spark, arc decay tau_arc) is unchanged from the default.
FAST_CIRCUIT = CircuitParams(t_end=2e-6)


def _build_methane_reactor(circuit=FAST_CIRCUIT):
    gas = ct.Solution(GRI30)
    gas.set_equivalence_ratio(0.5, "CH4", "O2:1.0, N2:3.76")
    gas.TP = 300.0, ct.one_atm
    params = ReactorParams(mechanism=GRI30, circuit=circuit)
    return PlasmaReactor(gas, params)


def _build_ammonia_reactor(circuit=FAST_CIRCUIT):
    gas = ct.Solution(AMMONIA)
    gas.set_equivalence_ratio(0.7, "NH3", "O2:1.0, N2:3.76")
    gas.TP = 300.0, ct.one_atm
    params = ReactorParams(mechanism=AMMONIA, circuit=circuit)
    return PlasmaReactor(gas, params)


@pytest.mark.parametrize("build", [_build_methane_reactor, _build_ammonia_reactor])
def test_integration_succeeds_and_stays_finite(build):
    reactor = build()
    result = reactor.integrate()

    assert np.all(np.isfinite(result.Tg))
    assert np.all(np.isfinite(result.Te))
    assert np.all(np.isfinite(result.ne))
    assert np.all(np.isfinite(result.Y))


@pytest.mark.parametrize("build", [_build_methane_reactor, _build_ammonia_reactor])
def test_mass_fractions_stay_normalized(build):
    reactor = build()
    result = reactor.integrate()
    totals = result.Y.sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-6)


@pytest.mark.parametrize("build", [_build_methane_reactor, _build_ammonia_reactor])
def test_temperature_and_density_stay_physical(build):
    reactor = build()
    result = reactor.integrate()
    assert np.all(result.Tg >= reactor.Tg_min - 1e-6)
    assert np.all(result.Tg <= reactor.Tg_max + 1e-6)
    assert np.all(result.Te > 0.0)
    assert np.all(result.ne > 0.0)


def test_refining_tolerance_does_not_change_ignition_delay_much():
    """Convergence check: tightening rtol/atol should not materially change
    the predicted ignition delay for the (fast, reduced-duration) methane
    case -- a basic verification that the stiff BDF integration has
    converged rather than being an artifact of loose tolerances."""
    from src import postprocess

    circuit = CircuitParams(t_end=3e-5)
    reactor_loose = _build_methane_reactor(circuit)
    result_loose = reactor_loose.integrate(rtol=1e-6)
    df_loose = postprocess.to_dataframe(result_loose)

    reactor_tight = _build_methane_reactor(circuit)
    result_tight = reactor_tight.integrate(rtol=1e-10)
    df_tight = postprocess.to_dataframe(result_tight)

    tau_loose = postprocess.ignition_delay(df_loose)
    tau_tight = postprocess.ignition_delay(df_tight)
    assert tau_loose == pytest.approx(tau_tight, rel=0.05)
