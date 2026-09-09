import numpy as np
import pytest

from src.circuit import CircuitParams, current, current_density


def test_current_at_t0_matches_spark_current():
    p = CircuitParams()
    assert current(0.0, p) == p.i_spark


def test_current_is_exactly_constant_through_spark_stage():
    """Regression test: an earlier version smoothed the spark->arc handoff
    with a logistic blend that leaked near-kA current into the supposedly
    constant, low-current spark stage well before t_spark."""
    p = CircuitParams()
    t = np.linspace(0.0, p.t_spark, 50)
    i = np.array([current(ti, p) for ti in t])
    assert np.all(i == p.i_spark)


def test_current_steps_up_at_t_spark():
    p = CircuitParams()
    just_before = current(p.t_spark, p)
    just_after = current(p.t_spark * 1.0000001, p)
    assert just_before == p.i_spark
    assert just_after == pytest.approx(p.i_peak, rel=1e-3)


def test_current_decays_exponentially_in_arc_stage():
    p = CircuitParams()
    t1 = p.t_spark + 2 * p.tau_arc
    t2 = p.t_spark + 4 * p.tau_arc
    i1 = current(t1, p)
    i2 = current(t2, p)
    expected_ratio = np.exp(-(t2 - t1) / p.tau_arc)
    assert i2 / i1 == pytest.approx(expected_ratio, rel=1e-6)


def test_current_density_is_current_over_area():
    p = CircuitParams()
    t = 5e-6
    assert current_density(t, p) == current(t, p) / p.area_arc


def test_current_monotonically_decreasing_in_arc_stage():
    p = CircuitParams()
    # start strictly after t_spark: current(t_spark, p) is the last spark-stage
    # sample (50 A) by the "<=" boundary convention, and the very next instant
    # legitimately jumps up to i_peak -- that jump is the point of
    # test_current_steps_up_at_t_spark, not a violation of arc-stage decay.
    t = np.linspace(p.t_spark * 1.001, p.t_end, 200)
    i = np.array([current(ti, p) for ti in t])
    assert np.all(np.diff(i) <= 0.0)
