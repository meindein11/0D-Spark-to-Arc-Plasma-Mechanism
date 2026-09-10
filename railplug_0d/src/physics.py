"""Electron-neutral collision physics for the 0D spark-to-arc discharge model.

Provides: electron-neutral momentum-transfer cross sections, collision
frequencies, effective plasma resistivity, electron<->heavy-particle elastic
energy exchange, and a simplified electron-density source/sink closure.

Units are SI throughout (m, s, kg, J, K, A, C) unless noted otherwise.
"""
from __future__ import annotations

import numpy as np
import cantera as ct
from scipy import constants as sc

QE = sc.e
ME = sc.m_e
KB = sc.k
AVOGADRO_PER_KMOL = sc.N_A * 1000.0  # cantera's molar unit is kmol, not mol

# ---------------------------------------------------------------------------
# Electron-neutral momentum-transfer ("effective") cross sections sigma_en.
#
# N2 and O2: full energy-resolved curves taken verbatim from the "effective"
# (momentum-transfer) electron-collision data in `air-plasma-Phelps.yaml`,
# bundled by the Cantera project at Cantera/cantera-example-data (retrieved
# 2026-09-09). That file's own provenance note: "A compilation of atomic and
# molecular cross-section data assembled by A. V. Phelps ... ", distributed
# via LXCat (https://www.lxcat.net, contributor id d19). Energies are in eV;
# cross sections are converted here from cm^2 to m^2.
#
# CH4, NH3, H2O, CO2: no energy-resolved data is bundled for these species, so
# single representative "effective" cross sections are used instead (roughly
# valid for the ~1-3 eV mean electron energies expected in this discharge),
# following order-of-magnitude values reported in electron-swarm literature
# (e.g. Itikawa (2006) for H2O; Itikawa & Mason-class reviews for CO2;
# Morgan/Yousfi-database-class values for CH4 and NH3, both of which have
# large low-energy momentum-transfer cross sections due to strong dipole /
# vibrational coupling). These are explicitly engineering approximations, not
# fitted swarm data.
# ---------------------------------------------------------------------------

_LXCAT_ENERGY_EV = np.array([
    0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.0085, 0.01, 0.015, 0.02,
    0.03, 0.04, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4,
    0.5, 0.7, 1.0, 1.2, 1.3, 1.5, 1.7, 1.9, 2.1, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6,
    4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0,
    50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 1500.0,
    2000.0, 3000.0, 5000.0, 7000.0, 10000.0,
])

_N2_SIGMA_CM2 = np.array([
    1.1e-20, 1.36e-20, 1.49e-20, 1.62e-20, 1.81e-20, 2e-20, 2.1e-20, 2.19e-20,
    2.55e-20, 2.85e-20, 3.4e-20, 3.85e-20, 4.33e-20, 5.1e-20, 5.95e-20,
    6.45e-20, 7.1e-20, 7.4e-20, 7.9e-20, 8.5e-20, 9e-20, 9.4e-20, 9.7e-20,
    9.9e-20, 1e-19, 1e-19, 1.04e-19, 1.1e-19, 1.2e-19, 1.38e-19, 1.96e-19,
    2.7e-19, 2.85e-19, 3e-19, 2.8e-19, 2.17e-19, 1.72e-19, 1.47e-19, 1.26e-19,
    1.13e-19, 1.09e-19, 1.04e-19, 1.01e-19, 1e-19, 1.04e-19, 1.09e-19, 1.1e-19,
    1.07e-19, 1.02e-19, 9.5e-20, 9e-20, 8.6e-20, 6.6e-20, 5.8e-20, 4.9e-20,
    4.2e-20, 3.3e-20, 2.44e-20, 1.96e-20, 1.55e-20, 1.12e-20, 8.1e-21, 6.3e-21,
    4e-21, 2.9e-21, 2.1e-21,
])

_O2_SIGMA_CM2 = np.array([
    3.5e-21, 3.5e-21, 3.6e-21, 4e-21, 5e-21, 5.8e-21, 6.4e-21, 7e-21, 8.7e-21,
    9.9e-21, 1.24e-20, 1.44e-20, 1.6e-20, 2.1e-20, 2.5e-20, 2.8e-20, 3.1e-20,
    3.3e-20, 3.6e-20, 4.1e-20, 4.5e-20, 4.7e-20, 5.2e-20, 5.7e-20, 6.1e-20,
    7.2e-20, 7.9e-20, 7.9e-20, 7.6e-20, 7.3e-20, 6.9e-20, 6.6e-20, 6.5e-20,
    6.1e-20, 5.8e-20, 5.7e-20, 5.5e-20, 5.45e-20, 5.5e-20, 5.55e-20, 5.6e-20,
    6e-20, 6.6e-20, 7.1e-20, 8e-20, 8.5e-20, 8.8e-20, 8.7e-20, 8.6e-20, 8.2e-20,
    8e-20, 7.7e-20, 6.8e-20, 6.5e-20, 6.7e-20, 6e-20, 4.9e-20, 3.6e-20, 2.9e-20,
    2.12e-20, 1.48e-20, 1.14e-20, 7.9e-21, 5.1e-21, 3.8e-21, 2.8e-21,
])

_TABULATED_SIGMA_EN_M2 = {
    "N2": (_LXCAT_ENERGY_EV, _N2_SIGMA_CM2 * 1e-4),
    "O2": (_LXCAT_ENERGY_EV, _O2_SIGMA_CM2 * 1e-4),
}

_CONSTANT_SIGMA_EN_M2 = {
    "CH4": 7.0e-20,
    "NH3": 1.0e-19,
    "H2O": 1.5e-19,
    "CO2": 3.0e-20,
    "CO": 2.0e-19,
    "H2": 1.0e-19,
}
_DEFAULT_SIGMA_EN_M2 = 3.0e-20  # generic fallback for trace/radical species

# Electron-impact ionization thresholds [eV], used only by the simplified
# ionization closure below. N2/O2 values match the reaction thresholds in
# air-plasma-Phelps.yaml.
ION_THRESHOLD_EV = {"N2": 15.6, "O2": 12.06}
_DEFAULT_ION_THRESHOLD_EV = 12.0

C_ION = 1.0e-13   # m^3/s/sqrt(eV); simplified ionization-rate prefactor (see notes)
K_REC0 = 2.0e-13  # m^3/s at 300 K; dissociative-recombination-scale coefficient
T_REC0 = 300.0    # K


def electron_temperature_eV(Te: float) -> float:
    """Te expressed in eV (kB*Te/e), the convention used for Arrhenius-style
    ionization/recombination rate scalings below."""
    return KB * Te / QE


def mean_electron_energy_eV(Te: float) -> float:
    """Mean kinetic energy of a Maxwellian electron population at
    temperature Te, in eV (1.5*kB*Te/e). Used only to look up sigma_en(eps)
    at a single representative energy, per the eps_en formula in the model
    spec -- not a full EEDF average."""
    return 1.5 * KB * Te / QE


def electron_neutral_cross_section(species: str, Te: float) -> float:
    """Effective (momentum-transfer) electron-neutral cross section [m^2]
    for `species`, evaluated at the mean electron energy for temperature
    Te [K]. Tabulated LXCat/Phelps data is interpolated (and clamped to the
    tabulated energy range); other species use a constant approximation."""
    eps_mean_eV = mean_electron_energy_eV(Te)
    table = _TABULATED_SIGMA_EN_M2.get(species)
    if table is not None:
        energies, sigmas = table
        return float(np.interp(eps_mean_eV, energies, sigmas))
    return _CONSTANT_SIGMA_EN_M2.get(species, _DEFAULT_SIGMA_EN_M2)


def electron_thermal_speed(Te: float) -> float:
    """Mean electron thermal speed sqrt(8 kB Te / (pi me)) [m/s]."""
    return np.sqrt(8.0 * KB * Te / (np.pi * ME))


def species_number_densities(gas: ct.Solution) -> dict[str, float]:
    """Number density [1/m^3] of every species in `gas`'s current state."""
    conc = gas.concentrations  # kmol/m^3
    return {
        name: float(c) * AVOGADRO_PER_KMOL
        for name, c in zip(gas.species_names, conc)
    }


def species_collision_frequencies(gas: ct.Solution, Te: float) -> dict[str, float]:
    """Per-species electron-neutral collision frequency nu_ek [1/s]."""
    v_th = electron_thermal_speed(Te)
    n_k = species_number_densities(gas)
    return {
        name: n * electron_neutral_cross_section(name, Te) * v_th
        for name, n in n_k.items()
    }


def collision_frequency(gas: ct.Solution, Te: float) -> float:
    """Total electron-neutral momentum-transfer collision frequency nu_en
    [1/s]: nu_en = sum_k n_k * sigma_en,k(Te) * v_th(Te)."""
    return sum(species_collision_frequencies(gas, Te).values())


def plasma_resistivity(gas: ct.Solution, Te: float, ne: float, ne_floor: float = 1e6) -> float:
    """Effective (Lorentz-gas / Spitzer-form) plasma resistivity rho_res
    [Ohm*m] = me * nu_en / (ne * e^2)."""
    nu_en = collision_frequency(gas, Te)
    ne_eff = max(ne, ne_floor)
    return ME * nu_en / (ne_eff * QE ** 2)


def electron_heavy_energy_exchange(gas: ct.Solution, Te: float, Tg: float, ne: float) -> float:
    """Elastic electron -> heavy-particle energy-relaxation power density
    q_e->M [W/m^3] = (2*me/M_mean) * nu_en(Te) * 1.5*kB*(Te - Tg) * ne.

    Regularized to use the mixture *mean* molecular weight M_mean in the
    (2*me/Mk) elastic energy-transfer-fraction factor, rather than summing
    that factor per-species (sum_k (2*me/Mk)*nu_ek, an earlier version of
    this function). The per-species form has a genuine physical
    justification in isolation (a light species really does absorb more
    energy per elastic collision than a heavy one), but summed across
    species it makes q_eM's *sensitivity to gas composition* scale with
    1/Mk, so trace amounts of very light dissociation fragments (atomic H,
    Mk ~1 g/mol, appearing e.g. as CH4/air heats past ~3000-4000 K) can
    dominate the sum out of proportion to their actual heat capacity or
    energy content. This was traced directly (see
    simulations/debug_ch4_bifurcation.py, MODELING_NOTES.md's phi-sweep
    follow-up) to an intrinsic positive-feedback runaway in
    ExpansionCoupledReactor's uncapped CH4 phi-sweep: hotter -> more H/H2
    dissociation -> larger sum_k(2*me/Mk)*nu_ek -> larger q_eM -> more
    heating -- a ~20x q_eM disparity between two adjacent phi points that
    no thermal-inertia or chemistry-side fix could address, because the
    runaway lived entirely in this term. Using the bulk mean molecular
    weight removes that per-species leverage while keeping the same
    overall nu_en(Te) (total electron-neutral collision frequency, via
    `collision_frequency`) and (Te-Tg) driving-temperature dependence.

    Note: this only changes the mass-ratio weighting, not the sign
    convention -- q_eM remains negative when Te < Tg (energy flowing
    gas -> electrons), matching `species_collision_frequencies`'s and
    `collision_frequency`'s existing behavior and this project's own
    `tests/test_physics.py::test_energy_exchange_sign_follows_temperature_
    difference`, which a naive `if Te <= Tg: return 0.0` early-out
    (as an initial draft of this fix considered) would have silently
    broken -- that guard was not applied.
    """
    if ne <= 0.0:
        return 0.0
    nu_en = collision_frequency(gas, Te)
    M_mean = gas.mean_molecular_weight / AVOGADRO_PER_KMOL  # kg per mean molecule
    mass_ratio_term = 2.0 * ME / M_mean
    return mass_ratio_term * nu_en * 1.5 * KB * (Te - Tg) * ne


# ---------------------------------------------------------------------------
# Simplified electron-density source/sink closure.
#
# The governing equations in the model spec describe electron ENERGY balance
# but do not specify how the electron POPULATION itself is produced or lost;
# standard combustion mechanisms (GRI-Mech 3.0, the Alzueta NH3/CO/H2
# mechanism) carry no charged species or electron-impact reactions. We close
# this gap with a standard two-process avalanche/recombination model:
#
#   dne/dt = k_ion(Te) * n_neutral * ne  -  k_rec(Te) * ne**2
#
# k_ion follows a Lotz/Drawin-shaped electron-impact ionization rate driven
# by the dominant bath-gas species' ionization threshold (N2: 15.6 eV, O2:
# 12.06 eV, taken from the electron-collision thresholds in
# air-plasma-Phelps.yaml). k_rec follows the ~Te^-1/2 scaling widely reported
# for dissociative recombination of diatomic molecular ions (order
# 1e-13 m^3/s near 300 K; see e.g. Florescu-Mitchell & Mitchell, Phys. Rep.
# 430 (2006), for a review of representative magnitudes).
#
# C_ION, K_REC0 and the reactor's seed electron density (ReactorParams.ne0)
# are simplified, tunable engineering parameters calibrated to give a
# qualitatively realistic spark->arc resistivity collapse; they are not
# fitted to a specific experiment.
# ---------------------------------------------------------------------------


def dominant_neutral_species(gas: ct.Solution) -> str:
    """Name of the most abundant (by mass fraction) species in `gas`."""
    idx = int(np.argmax(gas.Y))
    return gas.species_names[idx]


def ionization_rate_coefficient(Te: float, species: str) -> float:
    """Simplified electron-impact ionization rate coefficient k_ion(Te)
    [m^3/s] for collisions with `species`."""
    eps_eV = max(electron_temperature_eV(Te), 1e-6)
    Ei = ION_THRESHOLD_EV.get(species, _DEFAULT_ION_THRESHOLD_EV)
    return C_ION * np.sqrt(eps_eV) * np.exp(-Ei / eps_eV)


def recombination_rate_coefficient(Te: float) -> float:
    """Simplified dissociative-recombination rate coefficient k_rec(Te)
    [m^3/s], scaling as (300 K / Te)^0.5."""
    return K_REC0 * np.sqrt(T_REC0 / max(Te, 1.0))


def electron_density_terms(gas: ct.Solution, Te: float, ne: float, ne_floor: float = 1e6):
    """Ionization source S_ion and recombination sink S_rec [1/(m^3 s)],
    each >= 0, such that dne/dt = S_ion - S_rec. Returned separately (rather
    than only their difference) because the electron energy equation in
    reactor.py needs them individually: ionization is an energy sink for the
    electron population (see `ionization_energy_cost_eV`), while
    recombination removes electrons at the population's average energy and
    so does not, by itself, change Te."""
    species = dominant_neutral_species(gas)
    n_k = species_number_densities(gas)
    n_neutral = n_k.get(species, sum(n_k.values()))
    ne_eff = max(ne, ne_floor)
    k_ion = ionization_rate_coefficient(Te, species)
    k_rec = recombination_rate_coefficient(Te)
    s_ion = k_ion * n_neutral * ne_eff
    s_rec = k_rec * ne_eff ** 2
    return s_ion, s_rec


def ionization_energy_cost_eV(gas: ct.Solution) -> float:
    """Ionization threshold [eV] of the dominant bath-gas species, i.e. the
    energy the electron population loses per ionization event."""
    species = dominant_neutral_species(gas)
    return ION_THRESHOLD_EV.get(species, _DEFAULT_ION_THRESHOLD_EV)


def electron_density_source(gas: ct.Solution, Te: float, ne: float, ne_floor: float = 1e6) -> float:
    """Net dne/dt [1/(m^3 s)] from ionization of the dominant bath-gas
    species minus dissociative recombination."""
    s_ion, s_rec = electron_density_terms(gas, Te, ne, ne_floor)
    return s_ion - s_rec
