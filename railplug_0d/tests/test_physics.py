import numpy as np
import cantera as ct
import pytest

from src import physics


@pytest.fixture(scope="module")
def gas():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    g = ct.Solution(os.path.join(root, "mechanisms", "gri30.yaml"))
    g.set_equivalence_ratio(0.5, "CH4", "O2:1.0, N2:3.76")
    g.TP = 300.0, ct.one_atm
    return g


def test_cross_sections_are_positive_and_finite(gas):
    for species in ["N2", "O2", "CH4", "NH3", "H2O", "CO2", "OH"]:
        for Te in [300.0, 5000.0, 30000.0]:
            sigma = physics.electron_neutral_cross_section(species, Te)
            assert np.isfinite(sigma)
            assert sigma > 0.0


def test_tabulated_cross_section_clamps_outside_table_range(gas):
    # far below the table's lowest energy: interpolation clamps to the
    # first tabulated (lowest-energy) cross section
    sigma_lo = physics.electron_neutral_cross_section("N2", 1.0)  # ~1e-4 eV
    assert sigma_lo == pytest.approx(1.1e-24, rel=0.05)  # 1.1e-20 cm^2 -> m^2

    # far above the table's highest energy (10000 eV): clamps to the last
    # tabulated (highest-energy) cross section
    sigma_hi = physics.electron_neutral_cross_section("N2", 1e12)
    assert sigma_hi == pytest.approx(2.1e-25, rel=1e-6)  # 2.1e-21 cm^2 -> m^2


def test_collision_frequency_positive(gas):
    nu = physics.collision_frequency(gas, 1000.0)
    assert nu > 0.0
    assert np.isfinite(nu)


def test_resistivity_decreases_with_electron_density(gas):
    r_low_ne = physics.plasma_resistivity(gas, 5000.0, 1e18)
    r_high_ne = physics.plasma_resistivity(gas, 5000.0, 1e22)
    assert r_high_ne < r_low_ne


def test_energy_exchange_sign_follows_temperature_difference(gas):
    q_hot_electrons = physics.electron_heavy_energy_exchange(gas, 20000.0, 300.0, 1e20)
    q_equal = physics.electron_heavy_energy_exchange(gas, 300.0, 300.0, 1e20)
    q_cold_electrons = physics.electron_heavy_energy_exchange(gas, 300.0, 2000.0, 1e20)
    assert q_hot_electrons > 0.0   # Te > Tg: energy flows electrons -> gas
    assert q_equal == pytest.approx(0.0, abs=1e-30)
    assert q_cold_electrons < 0.0  # Te < Tg: energy flows gas -> electrons


def test_ionization_rate_increases_with_electron_temperature():
    k_cold = physics.ionization_rate_coefficient(5000.0, "N2")
    k_hot = physics.ionization_rate_coefficient(50000.0, "N2")
    assert k_hot > k_cold > 0.0


def test_recombination_rate_decreases_with_electron_temperature():
    k_cold = physics.recombination_rate_coefficient(300.0)
    k_hot = physics.recombination_rate_coefficient(30000.0)
    assert 0.0 < k_hot < k_cold


def test_electron_density_terms_are_nonnegative(gas):
    s_ion, s_rec = physics.electron_density_terms(gas, 20000.0, 1e19)
    assert s_ion >= 0.0
    assert s_rec >= 0.0
