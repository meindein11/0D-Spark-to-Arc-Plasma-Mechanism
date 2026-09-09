"""Coupled 0D spark-to-arc plasma reactor.

Integrates, at constant pressure, the joint state

    y = [Tg, Te, ne, Y_1, ..., Y_nsp]

where Tg is the gas temperature, Te the electron temperature, ne the electron
number density, and Y_k the species mass fractions of a Cantera `Solution`.
Gas-phase thermochemistry and kinetics come from Cantera; the electron
energy/density source terms come from `physics.py`; the discharge forcing
I(t)/j(t) comes from `circuit.py`. Time integration uses SciPy's stiff
implicit solvers (BDF/Radau), run in two phases (spark, then arc) so the
~100 ns spark transient and the ~10-50 us arc decay are each resolved with
step sizes appropriate to their own time scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
from scipy import constants as sc

from . import physics
from .circuit import CircuitParams, current_density

KB = sc.k
QE = sc.e


@dataclass
class ReactorParams:
    mechanism: str                          # path to the Cantera mechanism yaml
    T0: float = 300.0                        # initial gas temperature [K]
    P0: float = ct.one_atm                   # (constant) pressure [Pa]
    Te0: float | None = None                 # initial electron temperature [K]; defaults to T0
    # Seed electron density [1/m^3] at breakdown onset (t=0). Calibrated so
    # that the self-consistent E/N = rho_res(T0, ne0)*j(0)/n_neutral at t=0
    # lands near the ~250 Td reduced field the model spec cites for the
    # spark stage (rather than being an arbitrary free parameter).
    ne0: float = 2.7e18
    ne_floor: float = 1.0e6                  # numerical floor for ne [1/m^3]
    Te_floor: float = 1.0                    # numerical floor for Te [K], guards sqrt(Te) against a
                                              # transient non-positive value during stiff-solver trial steps
    q_loss_coefficient: float = 0.0          # Newtonian wall-loss coeff [W/(m^3 K)]; 0 = adiabatic
    circuit: CircuitParams = field(default_factory=CircuitParams)

    def __post_init__(self) -> None:
        if self.Te0 is None:
            self.Te0 = self.T0
        if self.ne0 <= self.ne_floor:
            raise ValueError(
                f"ne0 ({self.ne0:.3e}) must be above ne_floor ({self.ne_floor:.3e}); "
                "otherwise the seed electron density is silently overridden by the floor "
                "in every physics evaluation."
            )


@dataclass
class ReactorResult:
    t: np.ndarray             # [s]
    Tg: np.ndarray            # [K]
    Te: np.ndarray            # [K]
    ne: np.ndarray            # [1/m^3]
    Y: np.ndarray             # (n_t, n_species) mass fractions
    species_names: list[str]
    mechanism: str
    P0: float
    circuit: CircuitParams
    ne_floor: float


class PlasmaReactor:
    """Owns a Cantera `Solution` and integrates the coupled
    [Tg, Te, ne, Y] state at constant pressure."""

    def __init__(self, gas: ct.Solution, params: ReactorParams):
        self.gas = gas
        self.params = params
        self.nsp = gas.n_species
        # Standard NASA-7 combustion thermo data (as used by GRI-Mech 3.0 and
        # the Alzueta NH3/CO/H2 mechanism) is only valid up to ~3000 K, far
        # below the raw electron-Joule-heating power density implied by a
        # kA-scale current confined to a 1 mm^2 channel. Real spark/arc
        # channels do reach such temperatures locally, but the energy is
        # rapidly spread by hydrodynamic channel expansion, thermal
        # dissociation/ionization, and radiation -- none of which this
        # lumped, constant-pressure, no-expansion 0D model represents. We cap
        # Tg at the mechanism's validity ceiling as a explicit, documented
        # stand-in for those un-modeled sinks (see MODELING_NOTES.md), rather
        # than silently extrapolating NASA polynomials/Arrhenius rates into a
        # regime they were never fit for (which is also what was driving the
        # stiff solver to fail outright).
        self.Tg_min = gas.min_temp
        self.Tg_max = gas.max_temp

    def initial_state(self) -> np.ndarray:
        p = self.params
        y0 = np.empty(3 + self.nsp)
        y0[0] = p.T0
        y0[1] = p.Te0
        y0[2] = p.ne0
        y0[3:] = self.gas.Y
        return y0

    def _unpack(self, y: np.ndarray):
        Tg_raw = y[0]
        Tg = min(max(Tg_raw, self.Tg_min), self.Tg_max)
        Te = max(y[1], self.params.Te_floor)
        ne = max(y[2], self.params.ne_floor)
        Y = np.clip(y[3:], 0.0, None)
        total = Y.sum()
        if total > 0.0:
            Y = Y / total
        return Tg_raw, Tg, Te, ne, Y

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        p = self.params
        gas = self.gas
        Tg_raw, Tg, Te, ne, Y = self._unpack(y)
        gas.TPY = Tg, p.P0, Y

        j = current_density(t, p.circuit)
        rho = gas.density
        cp = gas.cp_mass
        wdot = gas.net_production_rates      # kmol/(m^3 s)
        hk = gas.partial_molar_enthalpies    # J/kmol
        Wk = gas.molecular_weights           # kg/kmol

        rho_res = physics.plasma_resistivity(gas, Te, ne, p.ne_floor)
        joule_heating = rho_res * j ** 2                                  # W/m^3, stays with electrons
        q_eM = physics.electron_heavy_energy_exchange(gas, Te, Tg, ne)    # W/m^3, electrons -> gas

        # Cantera convention: net production rates (wdot) and partial molar
        # enthalpies (hk) combine as -sum(wdot*hk) to give a *positive*
        # chemical heat-release rate for net-exothermic chemistry.
        q_chem = -np.dot(wdot, hk)                # W/m^3
        q_loss = p.q_loss_coefficient * (Tg - p.T0)

        s_ion, s_rec = physics.electron_density_terms(gas, Te, ne, p.ne_floor)
        dne_dt = s_ion - s_rec

        dTg_dt = (q_eM + q_chem - q_loss) / (rho * cp)
        if Tg_raw >= self.Tg_max and dTg_dt > 0.0:
            dTg_dt = 0.0  # saturate at the mechanism's thermo validity ceiling
        elif Tg_raw <= self.Tg_min and dTg_dt < 0.0:
            dTg_dt = 0.0  # symmetric floor: never push the raw state below the mechanism's valid range either

        # Electron energy balance: d(1.5 kB ne Te)/dt = joule_heating - q_eM
        # - (ionization energy cost + dilution by newly-born, ~zero-energy
        # electrons) * s_ion - (energy carried away by recombining
        # electrons, at the population's own average energy) * s_rec.
        # Expanding d(ne*Te)/dt = Te*dne/dt + ne*dTe/dt with dne/dt =
        # s_ion - s_rec, the "average energy" recombination loss term
        # exactly cancels the Te*(-s_rec) piece of the product rule (an
        # electron leaving with the mean energy does not, by itself, change
        # the mean of those left behind) -- so only the ionization terms
        # survive in dTe/dt. This also keeps dTe/dt well-scaled even when
        # s_rec is numerically huge (deep in the arc stage, ne is large and
        # recombination is fast), since s_rec no longer appears there.
        ei_cost_J = QE * physics.ionization_energy_cost_eV(gas)
        dTe_dt = (joule_heating - q_eM) / (1.5 * KB * ne) - s_ion * (ei_cost_J + 1.5 * KB * Te) / (1.5 * KB * ne)

        dY_dt = wdot * Wk / rho

        dy = np.empty_like(y)
        dy[0] = dTg_dt
        dy[1] = dTe_dt
        dy[2] = dne_dt
        dy[3:] = dY_dt
        return dy

    def _default_atol(self) -> np.ndarray:
        return np.concatenate(([1e-6, 1e-6, 1.0], np.full(self.nsp, 1e-16)))

    def integrate(
        self,
        method: str = "BDF",
        rtol: float = 1e-8,
        atol: np.ndarray | None = None,
    ) -> ReactorResult:
        p = self.params
        y0 = self.initial_state()
        if atol is None:
            atol = self._default_atol()

        t_spark = p.circuit.t_spark
        t_end = p.circuit.t_end
        if not (0.0 < t_spark < t_end):
            raise ValueError("CircuitParams requires 0 < t_spark < t_end")

        sol_spark = solve_ivp(
            self.rhs, (0.0, t_spark), y0, method=method,
            rtol=rtol, atol=atol, max_step=t_spark / 20.0,
        )
        if not sol_spark.success:
            raise RuntimeError(f"Spark-stage integration failed: {sol_spark.message}")

        sol_arc = solve_ivp(
            self.rhs, (t_spark, t_end), sol_spark.y[:, -1], method=method,
            rtol=rtol, atol=atol, max_step=(t_end - t_spark) / 200.0,
        )
        if not sol_arc.success:
            raise RuntimeError(f"Arc-stage integration failed: {sol_arc.message}")

        t = np.concatenate([sol_spark.t, sol_arc.t[1:]])
        y = np.concatenate([sol_spark.y, sol_arc.y[:, 1:]], axis=1)

        # Clip/renormalize Y the same way _unpack() does for every rhs() call,
        # so the *returned* trajectory matches what the physics actually saw
        # rather than exposing raw solver roundoff (small negatives, sums
        # slightly off 1) that rhs() silently cleaned up on its own.
        Y = np.clip(y[3:, :], 0.0, None)
        sums = Y.sum(axis=0)
        sums[sums == 0.0] = 1.0
        Y = Y / sums

        return ReactorResult(
            t=t,
            Tg=y[0, :],
            Te=y[1, :],
            ne=y[2, :],
            Y=Y.T,
            species_names=list(self.gas.species_names),
            mechanism=p.mechanism,
            P0=p.P0,
            circuit=p.circuit,
            ne_floor=p.ne_floor,
        )
