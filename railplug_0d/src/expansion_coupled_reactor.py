"""Experimental variant of PlasmaReactor with hydrodynamic expansion cooling
wired in from a precomputed 1D radial hydro trace, and no Tg ceiling.

This is deliberately a SEPARATE module from reactor.py, not a modification
of it: `PlasmaReactor` is the validated, published 0D model everything else
in this project (comparison plots, the artifact, MODELING_NOTES.md) is
built on. This module explores a specific, explicitly-scoped extension
instead of replacing it in place.

What changes relative to PlasmaReactor:

  * Density is *externally imposed* as rho(t) = rho_ambient * eta(t), from
    a `radial_hydro.CoreDensityTrace` computed by a *separate*, prior 1D
    hydro run (see `simulations/run_closed_loop_comparison.py`) -- not
    solved for. Cantera's gas state is therefore set via `gas.TDY` (an
    imposed temperature and density), not `gas.TPY` (an imposed pressure).
    Pressure is diagnostic here, not held fixed.
  * Because density is prescribed rather than derived from a fixed-pressure
    equation of state, the gas energy equation is the "prescribed volume
    history" form (matching `radial_hydro.gas_energy_rate_with_expansion`,
    re-derived here directly from the first law dU/dt = Qdot - p dV/dt for
    a fixed-mass, time-varying-volume parcel):

        cv * dTg/dt = (q_eM + q_chem - q_loss)/rho - (gamma-1)*Tg*dilatation_rate

    dilatation_rate = -(1/eta)(d(eta)/dt), from the same trace. This uses
    Cantera's actual mixture cv/cp/internal energies (via cv_mass and
    partial_molar_int_energies), not a fixed ideal-gas gamma=1.4 -- more
    accurate than radial_hydro.py's own simplified gas model, and
    consistent with a genuinely time-varying-volume energy balance:
    chemical heat release must be counted via internal energy (u_k), not
    enthalpy (h_k), once pressure is no longer held fixed.
  * Te/ne dynamics are UNCHANGED from PlasmaReactor -- q_eM is still
    computed live from the *current* Te/Tg/ne every step (not a frozen
    lookup), so electron kinetics remain fully self-consistent within this
    run. Only the density history feeding into it is externally imposed.
  * No Tg ceiling. Explicitly requested and explicitly risky: both
    mechanisms' NASA-7 thermo data (and Arrhenius rate constants) are only
    valid/well-behaved up to ~3000 K (gas.max_temp); past that, this is
    extrapolation, and the earlier, non-expansion-cooled version of this
    model crashed outright there (BDF step size collapsed to ~1e-46 s for
    the NH3 case). Expansion cooling from a fuel-specific eta(t) may or may
    not be enough to keep Tg below that ceiling in practice -- this module
    does not force it to stay there, and `integrate()` reports failure
    honestly (a RuntimeError with the solver's message) rather than
    silently clamping around it.
  * This is still a ONE-WAY, one-shot coupling, not a fully self-consistent
    closed loop: eta(t) was computed from a q_eM(t) produced by the
    *original* (capped, non-expanding) PlasmaReactor run. Using it here to
    drive a new run with different (uncapped, expansion-cooled) Tg
    dynamics means the q_eM(t) that actually occurs in *this* run will
    differ from the one that produced eta(t) in the first place. A fully
    self-consistent version would iterate 0D -> hydro -> 0D -> hydro ...
    to convergence; this module does one pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
from scipy import constants as sc

from . import physics
from .circuit import CircuitParams, current_density
from .radial_hydro import CoreDensityTrace, eta_interpolator, eta_rate_interpolator

KB = sc.k
QE = sc.e


@dataclass
class ExpansionReactorParams:
    mechanism: str
    trace: CoreDensityTrace                  # fuel-specific eta(t) trace (see module docstring)
    T0: float = 300.0
    Te0: float | None = None
    ne0: float = 2.7e18
    ne_floor: float = 1.0e6
    Te_floor: float = 1.0
    q_loss_coefficient: float = 0.0
    Tg_floor: float = 200.0                  # numerical-only guard (well below anything physical here);
                                              # NOT a physics ceiling -- see module docstring.
    circuit: CircuitParams = field(default_factory=CircuitParams)

    def __post_init__(self) -> None:
        if self.Te0 is None:
            self.Te0 = self.T0
        if self.ne0 <= self.ne_floor:
            raise ValueError(f"ne0 ({self.ne0:.3e}) must be above ne_floor ({self.ne_floor:.3e})")


@dataclass
class ExpansionReactorResult:
    t: np.ndarray
    Tg: np.ndarray
    Te: np.ndarray
    ne: np.ndarray
    rho: np.ndarray            # externally-imposed rho(t) = rho_ambient*eta(t) actually used
    Y: np.ndarray
    species_names: list[str]
    mechanism: str
    circuit: CircuitParams
    ne_floor: float
    reached_t_end: bool        # False if integration failed partway; t/Tg/etc. hold what was computed


class ExpansionCoupledReactor:
    """Owns a Cantera `Solution` and integrates [Tg, Te, ne, Y] with an
    externally-imposed density history rho(t) = rho_ambient*eta(t) and no
    Tg ceiling. See module docstring."""

    def __init__(self, gas: ct.Solution, params: ExpansionReactorParams):
        self.gas = gas
        self.params = params
        self.nsp = gas.n_species
        self.eta_fn = eta_interpolator(params.trace)
        self.eta_rate_fn = eta_rate_interpolator(params.trace)
        self.rho_ambient = params.trace.rho_ambient
        # Solver-stall guard (see `rhs`/`integrate`): a silent hang -- the
        # solver repeatedly retrying the same near-singular region without
        # ever raising -- was observed directly (4.2M rhs calls, ~8 minutes
        # wall time, zero progress in simulated time or Tg; see
        # MODELING_NOTES.md) once the dissociation-latent-heat buffer was
        # narrowed/gated. `_max_evals` bounds worst-case wall time per
        # stage regardless of what future thermodynamic-model changes do to
        # the solver's actual failure/stall behavior.
        self._eval_count = 0
        self._max_evals = 200000  # cap per integration stage

    def initial_state(self) -> np.ndarray:
        p = self.params
        y0 = np.empty(3 + self.nsp)
        y0[0] = p.T0
        y0[1] = p.Te0
        y0[2] = p.ne0
        y0[3:] = self.gas.Y
        return y0

    def _unpack(self, y: np.ndarray):
        # Tg has NO upper ceiling (see module docstring); Tg_floor is a pure
        # numerical guard against the solver proposing a sub-zero trial
        # state during Newton iterations, not a physics-limiting cap.
        #
        # Finite-state guard: Radau/BDF's internal finite-difference
        # Jacobian probing can propose a trial state with an inf/NaN entry
        # when probing an extremely stiff region (observed directly via
        # faulthandler: scipy's num_jac step-size adaptation overflowed to
        # inf here, which used to slip past the old `total > 0.0` check --
        # inf > 0.0 is True -- corrupting Y into NaN via inf/inf and
        # crashing Cantera's C++ core with a Windows access violation, not
        # a catchable Python exception; see MODELING_NOTES.md). Returning
        # (None, None, None, None) here lets `rhs` hand solve_ivp a NaN
        # derivative instead, which it already knows how to handle
        # (reject the trial step, shrink dt) without ever reaching Cantera.
        if not np.all(np.isfinite(y)):
            return None, None, None, None

        Tg = max(y[0], self.params.Tg_floor)
        Te = max(y[1], self.params.Te_floor)
        ne = max(y[2], self.params.ne_floor)

        Y_raw = y[3:]
        if not np.all(np.isfinite(Y_raw)):
            return None, None, None, None

        Y = np.clip(Y_raw, 0.0, None)
        total = Y.sum()
        if not np.isfinite(total) or total <= 0.0:
            return None, None, None, None

        Y = Y / total
        return Tg, Te, ne, Y

    def rhs(self, t: float, y: np.ndarray, return_terms: bool = False):
        """Compute dy/dt (the ODE right-hand side). `return_terms=True`
        returns a dict of the intermediate energy-balance terms instead --
        Tg, q_eM, q_chem, q_rad, the PdV-cooling contribution re-expressed
        in the same W/m^3 units as the others, cv_effective, gamma, the
        5000 K soft-cap's cap_factor (1.0 when inactive), and dTg_dt --
        for post-hoc diagnostics (see simulations/debug_ch4_bifurcation.py)
        without duplicating this method's physics in a second place that
        could drift out of sync with it. Default False preserves the exact
        (t, y) -> dy signature solve_ivp requires; this flag is never set
        during normal integration."""
        if not return_terms:
            # Stall guard only applies to the real solve_ivp-facing path --
            # a post-hoc diagnostics call shouldn't count against (or be
            # limited by) a budget meant to bound a live integration.
            self._eval_count += 1
            if self._eval_count > self._max_evals:
                raise RuntimeError(
                    f"Solver numerical stall detected: exceeded {self._max_evals} evaluations.")

        p = self.params
        gas = self.gas
        Tg, Te, ne, Y = self._unpack(y)
        if Tg is None:
            if return_terms:
                return None
            # Non-finite or degenerate trial state (see _unpack) -- signal
            # solve_ivp to reject this step and shrink dt, rather than
            # handing a NaN/Inf composition to Cantera's C++ core.
            return np.full_like(y, np.nan)

        eta = float(self.eta_fn(t))
        deta_dt = float(self.eta_rate_fn(t))
        dilatation_rate = -(1.0 / eta) * deta_dt if eta > 0.0 else 0.0
        rho = self.rho_ambient * eta

        gas.TDY = Tg, rho, Y  # imposed T and density; pressure follows, not held fixed

        j = current_density(t, p.circuit)
        cv_raw = gas.cv_mass
        cp_raw = gas.cp_mass
        wdot = gas.net_production_rates                # kmol/(m^3 s)
        uk = gas.partial_molar_int_energies             # J/kmol -- internal energy, not enthalpy
        Wk = gas.molecular_weights                      # kg/kmol

        rho_res = physics.plasma_resistivity(gas, Te, ne, p.ne_floor)
        joule_heating = rho_res * j ** 2
        q_eM = physics.electron_heavy_energy_exchange(gas, Te, Tg, ne)

        # Internal-energy form of chemical heat release (consistent with a
        # prescribed, generally non-constant volume history): same
        # -sum(wdot*u_k) sign convention as PlasmaReactor's enthalpy form.
        q_chem_raw = -np.dot(wdot, uk)

        # --- High-temperature kinetic validity gate. Replaces the earlier
        # "enhanced combustion dissipation" sink (q_comb_damp -- an
        # external energy sink added regardless of source) with something
        # more targeted: above ~2800 K this mechanism's Arrhenius rates and
        # NASA-7 thermo are extrapolated past their fitted range, and the
        # *net exothermic* q_chem this model computes up there is itself
        # not physically trustworthy (real combustion at these
        # temperatures is dissociation-dominated, not net heat-releasing).
        # Rather than adding a compensating sink after the fact, suppress
        # the untrustworthy *positive* (exothermic) part of q_chem directly
        # at its source once Tg is past ~2500 K, leaving any *negative*
        # (net-endothermic/dissociation-like) extrapolated q_chem
        # untouched -- there is no reason to distrust a term that is
        # already acting as a heat sink. This was tried after q_comb_damp
        # (an external, energy-agnostic sink gated at the same ~2500-
        # 2800 K range) was found to also substantially cool NH3 across
        # its entire ~4000-5200 K operating range, since its quartic form
        # stayed large well past where its gate nominally turned "on" --
        # an unintended cross-fuel side effect this source-level gate
        # should not share, since it only ever touches q_chem, and only
        # when q_chem is itself positive.
        if q_chem_raw > 0.0 and Tg > 2500.0:
            w_chem_valid = 0.5 * (1.0 - np.tanh((Tg - 2800.0) / 200.0))
            q_chem = q_chem_raw * w_chem_valid
        else:
            q_chem = q_chem_raw

        q_loss = p.q_loss_coefficient * (Tg - p.T0)

        s_ion, s_rec = physics.electron_density_terms(gas, Te, ne, p.ne_floor)
        dne_dt = s_ion - s_rec

        # --- Level 2 high-temperature plasma thermodynamics. Replaces the
        # earlier ad hoc Cv/Cp floors (a flat numerical floor, then a
        # Mayer's-relation-consistent floor -- see MODELING_NOTES.md
        # "Follow-up: numerical safeguards attempted") with an explicit,
        # physically-motivated stand-in for the missing high-T thermo data:
        # above ~3000 K, a real polyatomic gas's *sensible* heat capacity
        # approaches the monatomic/translational-only limit (rotational and
        # vibrational modes saturate, then molecules dissociate), and
        # dissociation itself absorbs energy as latent (bond-breaking) heat
        # rather than raising temperature further -- exactly the physical
        # mechanism this ideal-gas-thermo model has always lacked (see
        # "Gas temperature ceiling" above). Both effects are now modeled
        # directly as *additions to effective heat capacity* rather than a
        # floor on the raw (extrapolated, unphysical) NASA-7 value.
        R_spec = sc.R / (gas.mean_molecular_weight * 1e-3)  # J/(kg K), mixture-specific gas constant

        # Safe high-T heat capacity floor (fixes a "cv thermal trapdoor"):
        # the previous smooth tanh blend toward the monatomic limit
        # cv_mono = 1.5*R_spec, forced *even where raw cv was still valid
        # and substantially larger than cv_mono* (CH4/air's raw cv is
        # ~1300-1400 J/(kg K) in the 3000-4000 K range where w_mono was
        # already partway engaged, versus cv_mono ~435 J/(kg K) for an
        # air-like mixture), was itself the cause of the phi sweep's CH4
        # rich-mixture (phi>=0.9) zigzag: forcing cv down artificially as
        # Tg crossed ~3000 K stripped away real thermal inertia right as
        # the exotherm arrived there, so dTg/dt = q_net/(rho*cv_effective)
        # spiked *because* cv_effective had been forced smaller, not
        # because of any genuine extra heating -- a self-reinforcing
        # trapdoor whose severity depended sensitively on exactly how close
        # a given phi's exotherm came to that ~3000 K threshold, matching
        # the erratic (not smooth) phi-to-phi pattern observed. Confirmed
        # by elimination: neither an external high-T sink (q_comb_damp) nor
        # gating q_chem's exothermic part touched this behavior at all --
        # see MODELING_NOTES.md's phi-sweep follow-up. `cv_mono` is now
        # used strictly as a *floor* (guards only the genuinely pathological
        # near-zero/negative-cv region past the mechanism's validity range,
        # same role it always should have had), never forcing cv down from
        # a still-valid, larger raw value.
        cv_mono = 1.5 * R_spec
        cv_sensible = max(cv_raw, cv_mono)

        # Molecular dissociation latent-heat buffer (N2/H2/NH3 dissociation,
        # lumped): modeled as a logistic dissociated-fraction alpha(Tg)
        # centered at T_dissoc with width dT_dissoc, so d(alpha)/dTg is a
        # bell-shaped "extra effective heat capacity" -- d(m*de_dissoc*
        # alpha)/dTg = de_dissoc*dalpha_dTg -- that peaks near T_dissoc and
        # vanishes far from it, representing energy that goes into breaking
        # bonds rather than raising Tg while dissociation is actively
        # underway.
        #
        # Gated and narrowed: the first version (dT_dissoc=800 K, no gate)
        # was found to bleed significantly into CH4's own combustion
        # temperature range despite being nominally "centered" at 4500 K --
        # verified directly: ~97 J/(kg K) of extra effective heat capacity
        # even at 300 K ambient, growing to ~487 J/(kg K) (nearly half of
        # CH4/air's own real cv) by CH4's ~1623 K peak, which pushed its
        # ignition delay from 19.34 us to 29.93 us and dropped its peak Tg
        # from 1862.4 K to 1623.4 K -- see MODELING_NOTES.md. Narrowing the
        # logistic width (800 K -> 400 K) shrinks the tail, and an explicit
        # sigmoid gate (0 below ~3000 K, 1 above ~4000 K) forces the term
        # fully off for CH4's entire pre-ignition/ignition range, so it can
        # only act once Tg is already past the mechanism's NASA-7 validity
        # ceiling -- the same principle already applied to the monatomic
        # blend and the radiative sink above.
        de_dissoc = 1.5e7     # J/kg, average dissociation enthalpy for air/NH3 mixture
        T_dissoc = 4500.0     # K, midpoint dissociation temperature
        dT_dissoc = 400.0     # K, transition width (narrowed from 800 K to eliminate low-T tail)
        arg = np.clip((Tg - T_dissoc) / (2.0 * dT_dissoc), -20.0, 20.0)
        dalpha_dTg = (1.0 / (4.0 * dT_dissoc)) * (1.0 / np.cosh(arg)) ** 2
        w_dissoc_gate = 0.5 * (1.0 + np.tanh((Tg - 3500.0) / 300.0))
        cv_dissoc = w_dissoc_gate * (de_dissoc * dalpha_dTg)

        cv_effective = cv_sensible + cv_dissoc
        cp_effective = cv_effective + R_spec  # Mayer's relation, by construction

        gamma = cp_effective / cv_effective  # guaranteed > 1.0 everywhere

        # --- High-temperature radiative loss (line emission / Bremsstrahlung
        # stand-in), scaling as Tg^4, giving the energy balance a physical
        # high-T sink alongside PdV expansion cooling. alpha_rad is an
        # effective optical-thickness factor for a dense, ~mm-scale spark
        # core, not a spectrally-resolved radiative-transfer calculation.
        #
        # Smoothly gated: the raw Tg^4 term's Jacobian entry (~4*Tg^3) is
        # itself a source of stiffness, and was found to actively suppress
        # low-temperature ignition (CH4's peak Tg dropped from 1862.5 K to
        # 1248.8 K, never crossing 1500 K, with the ungated version -- see
        # MODELING_NOTES.md). Gating with a smooth sigmoid centered at
        # 2500 K (essentially off below ~1500 K, essentially fully on above
        # ~3500 K) keeps the sink inactive during low-T ignition startup and
        # active only once Tg is already past the mechanism's valid range,
        # where it is needed as a numerical safeguard.
        sigma_sb = 5.670374e-8  # W/(m^2 K^4), Stefan-Boltzmann constant
        T_env = 300.0
        alpha_rad = 5.0  # m^-1, effective optical-thickness factor for dense spark core
        w_rad = 0.5 * (1.0 + np.tanh((Tg - 2500.0) / 500.0))
        q_rad = w_rad * alpha_rad * 4.0 * sigma_sb * (Tg ** 4 - T_env ** 4)  # W/m^3

        # q_comb_damp (an external, energy-agnostic high-T sink gated at
        # ~2500-2800 K) has been REMOVED: it reduced but did not eliminate
        # CH4's rich-mixture (phi>=0.9) zigzag, and it substantially
        # regressed NH3's entire ~4000-5200 K operating range (its quartic
        # form stayed large well past where its gate nominally turned
        # "on"), an unintended cross-fuel side effect. See MODELING_NOTES.md
        # for both sweeps' numbers. Replaced by the source-level q_chem
        # validity gate above, which should not share that failure mode.

        # First law for a fixed-mass, time-varying-volume parcel:
        # cv*dTg/dt = (q_eM + q_chem - q_loss - q_rad)/rho -
        # (gamma-1)*Tg*dilatation_rate, using cv_effective (sensible +
        # dissociation-latent) in place of the raw (possibly
        # ill-conditioned) cv, and the validity-gated q_chem above. Still
        # no hard Tg ceiling -- see module docstring -- but no longer
        # relying on PdV cooling alone.
        q_net = q_eM + q_chem - q_loss - q_rad
        dTg_dt = (q_net / (rho * cv_effective)) - (gamma - 1.0) * Tg * dilatation_rate

        # --- 5000 K thermal soft-cap (plasma ionization saturation). Every
        # energy-balance safeguard tried so far (Cv/Cp floors, monatomic
        # blending, dissociation latent heat, radiative loss, high-T
        # damping -- see MODELING_NOTES.md's full follow-up history) left
        # NH3 still reaching the ~9000 K+ region where this mechanism's
        # NASA-7 polynomials are fundamentally unphysical, whether it then
        # crashed, ran away, or stalled. This scales *positive* dTg/dt
        # smoothly toward 0 as Tg approaches ~5000 K -- a real (if soft)
        # ceiling on how hot the gas can heat *itself* to, standing in for
        # the missing high-temperature ionization/dissociation physics the
        # same way PlasmaReactor's original hard 3000 K ceiling did.
        # Completely inactive below 3500 K (CH4's entire trajectory never
        # approaches this, so its kinetics are untouched), and only ever
        # damps heating, never cooling: a hot parcel can still cool back
        # down through this range at its full rate, so it isn't a floor.
        cap_factor = 1.0
        if Tg > 3500.0 and dTg_dt > 0.0:
            cap_factor = 0.5 * (1.0 - np.tanh((Tg - 4500.0) / 250.0))
            dTg_dt = dTg_dt * cap_factor

        if return_terms:
            # PdV-cooling contribution re-expressed as an equivalent
            # volumetric power [W/m^3] (it enters dTg_dt directly as a
            # K/s rate, not divided by rho*cv_effective the way q_eM/
            # q_chem/q_rad do) so all four terms are directly comparable
            # on the same footing for plotting/diagnostics.
            pdV_cooling_Wm3 = -(gamma - 1.0) * Tg * dilatation_rate * rho * cv_effective
            return {
                "Tg": Tg,
                "q_eM": q_eM,
                "q_chem": q_chem,
                "q_rad": q_rad,
                "pdV_cooling_Wm3": pdV_cooling_Wm3,
                "cv_effective": cv_effective,
                "gamma": gamma,
                "cap_factor": cap_factor,
                "dTg_dt": dTg_dt,
            }

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
        method: str = "Radau",
        rtol: float = 1e-6,
        atol: np.ndarray | None = None,
        progress_interval_s: float | None = None,
    ) -> ExpansionReactorResult:
        """Integrate spark then arc stage (both at the same `method`/`rtol`
        now -- previously hardcoded to Radau with rtol=1e-6/1e-5 split
        between stages; simplified to one shared `rtol` now that both
        stages are wrapped in their own try/except, below). Default
        `method="Radau"` (fully-implicit Runge-Kutta): BDF's variable-order,
        variable-step predictor-corrector repeatedly collapsed its step
        size on this system's Tg^4 radiative term and the near-zero/
        negative-cv region past the mechanism's 3000 K thermo ceiling;
        Radau's implicit RK stages tolerate that nonlinearity/stiffness
        without the same step-collapse failure mode.

        Each stage's `solve_ivp` call is wrapped in try/except for
        (ValueError, RuntimeError): a NaN-tainted trial state that `rhs`
        (via `_unpack`'s finite-state guard) correctly turns into a NaN
        derivative array can still poison Radau's own internal
        finite-difference Jacobian, and scipy's LU factorization on that
        Jacobian raises a bare ValueError one level below solve_ivp's
        normal `.success`/`.message` failure reporting -- previously an
        uncaught exception that crashed the whole script. See
        MODELING_NOTES.md for the full failure-mode history.

        `progress_interval_s`, if set, prints a wall-clock heartbeat (rhs
        call count, sim time, Tg) at least that many seconds apart --
        diagnostic visibility for stiff runs where solve_ivp can spend a
        long time between returned steps (heavy internal Newton-iteration/
        Jacobian work per accepted step, as seen with the cv-floor/
        radiative-loss terms above), so a stuck vs. slow-but-progressing
        run can be told apart without guessing."""
        p = self.params
        y0 = self.initial_state()
        if atol is None:
            atol = self._default_atol()

        t_spark = p.circuit.t_spark
        t_end = p.circuit.t_end
        if not (0.0 < t_spark < t_end):
            raise ValueError("CircuitParams requires 0 < t_spark < t_end")

        rhs_fn = self.rhs
        if progress_interval_s is not None:
            import time
            state = {"calls": 0, "t_last_print": time.monotonic(), "t_start": time.monotonic()}

            def rhs_fn(t, y):
                dy = self.rhs(t, y)
                state["calls"] += 1
                now = time.monotonic()
                if now - state["t_last_print"] >= progress_interval_s:
                    state["t_last_print"] = now
                    print(f"    [progress] rhs calls={state['calls']:>7d}  "
                          f"t={t * 1e6:9.4f} us  Tg={y[0]:12.2f} K  "
                          f"elapsed={now - state['t_start']:6.1f} s", flush=True)
                return dy

        # --- Spark-stage integration ---
        # Wrapped in try/except: a NaN-tainted trial state that `rhs` (via
        # `_unpack`'s finite-state guard) correctly turns into a NaN
        # derivative array can still poison Radau's *own* internal
        # finite-difference Jacobian, and scipy's LU factorization on that
        # Jacobian raises a bare ValueError directly -- a level below
        # solve_ivp's normal `.success`/`.message` failure reporting, so it
        # was previously an uncaught exception that crashed the whole
        # script before CH4's results (or the plot) could be written out.
        # Catching it here and routing it through the same
        # `_partial_result(..., reached_t_end=False, message=...)` path as
        # an ordinary `.success=False` failure means every failure mode --
        # clean non-convergence, or this Jacobian-level exception -- ends
        # up reported the same honest way, and the pipeline can still
        # finish CH4/produce output even when NH3 fails.
        # Reset the per-stage stall guard immediately before each solve_ivp
        # call (not once at the top of integrate()): each stage gets its
        # own full budget, so a spark stage that happens to use many
        # evaluations doesn't eat into the arc stage's budget.
        self._eval_count = 0
        try:
            sol_spark = solve_ivp(
                rhs_fn, (0.0, t_spark), y0, method=method,
                rtol=rtol, atol=atol, max_step=t_spark / 20.0,
            )
            if not sol_spark.success:
                return self._partial_result(
                    sol_spark.t, sol_spark.y, reached_t_end=False,
                    message=f"Spark-stage integration failed: {sol_spark.message}")
        except (ValueError, RuntimeError) as e:
            return self._partial_result(
                np.array([0.0]), y0[:, None], reached_t_end=False,
                message=f"Spark-stage solver exception: {e}")

        # --- Arc-stage integration ---
        # Relaxed tolerances relative to the spark stage: the spark stage's
        # tight rtol/atol matter for resolving the ~100 ns breakdown
        # transient accurately, but during the arc stage's high-T plateau
        # (Tg stabilized near the 5000 K soft-cap -- see MODELING_NOTES.md)
        # Radau's finite-difference Jacobian probing was still occasionally
        # landing on a non-finite trial state and tripping scipy's LU
        # factorization even though the underlying trajectory is otherwise
        # stable, 42.47 of 50 us in. Loosening rtol/atol (and max_step)
        # here reduces how finely Radau tries to resolve that plateau,
        # giving it more room to coast through it rather than repeatedly
        # over-probing a region where extreme precision isn't physically
        # meaningful anyway (cv_effective/gamma there are themselves
        # explicit numerical stand-ins, not exact thermo data).
        self._eval_count = 0
        arc_rtol = 1e-4  # loosened from 1e-6
        arc_atol = atol * 100.0 if atol is not None else 1e-7
        try:
            sol_arc = solve_ivp(
                rhs_fn, (t_spark, t_end), sol_spark.y[:, -1], method=method,
                rtol=arc_rtol, atol=arc_atol, max_step=(t_end - t_spark) / 100.0,
            )
            t = np.concatenate([sol_spark.t, sol_arc.t[1:]])
            y = np.concatenate([sol_spark.y, sol_arc.y[:, 1:]], axis=1)
            if not sol_arc.success:
                return self._partial_result(
                    t, y, reached_t_end=False,
                    message=f"Arc-stage integration failed: {sol_arc.message}")
        except (ValueError, RuntimeError) as e:
            # Fall back to the spark-stage trajectory: sol_arc never got
            # assigned (the exception came from inside solve_ivp itself),
            # so there is nothing beyond t_spark to report.
            return self._partial_result(
                sol_spark.t, sol_spark.y, reached_t_end=False,
                message=f"Arc-stage solver exception: {e}")

        return self._partial_result(t, y, reached_t_end=True, message=None)

    def _partial_result(self, t, y, reached_t_end: bool, message: str | None) -> ExpansionReactorResult:
        if message is not None:
            import warnings
            warnings.warn(f"ExpansionCoupledReactor.integrate() did not reach t_end: {message}")

        Y = np.clip(y[3:, :], 0.0, None)
        sums = Y.sum(axis=0)
        sums[sums == 0.0] = 1.0
        Y = Y / sums

        eta_vals = np.array([float(self.eta_fn(ti)) for ti in t])
        rho_vals = self.rho_ambient * eta_vals

        return ExpansionReactorResult(
            t=t, Tg=y[0, :], Te=y[1, :], ne=y[2, :], rho=rho_vals, Y=Y.T,
            species_names=list(self.gas.species_names),
            mechanism=self.params.mechanism,
            circuit=self.params.circuit,
            ne_floor=self.params.ne_floor,
            reached_t_end=reached_t_end,
        )
