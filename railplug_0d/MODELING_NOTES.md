# Modeling notes

This document records the modeling choices, data provenance, and known
limitations behind the code in `src/`. The governing equations in the
project brief specify electron energy balance, gas energy/species balance,
and the two-stage discharge current, but leave several closures
unspecified (notably electron-density kinetics and the constant-pressure
sign convention); this file explains how those gaps were filled and why.

## Usage

```
pip install -r requirements.txt                       # or requirements-dev.txt to also get pytest
python simulations/run_methane.py                      # CH4/Air, phi=0.5, capped model
python simulations/run_ammonia.py                       # NH3/Air, phi=0.7, capped model
python simulations/compare.py                           # both cases + comparison plots/summary in data/
python simulations/run_channel_expansion.py             # 1D radial hydro channel-expansion demo
python simulations/run_closed_loop_comparison.py         # uncapped, expansion-coupled model, both fuels
python simulations/run_parametric_phi_sweep.py           # equivalence-ratio sweep (phi=0.4-1.2), both fuels
python simulations/debug_ch4_bifurcation.py              # term-by-term CH4 phi=0.9 vs phi=1.0 diagnostic
pytest tests/                                            # unit tests + a tolerance-convergence check
```

The first four are the original, validated (hard 3000 K ceiling) model. The
last three are the experimental, uncapped, expansion-coupled reactor and
its diagnostics -- see "Uncapped, expansion-coupled reactor" below for why
that model needed six additional numerical fixes and one real bug fix (a
dissociation-driven feedback loop in `physics.py`'s electron-heating term)
before it would run reliably.

## Mechanisms (`mechanisms/`)

- `gri30.yaml` -- GRI-Mech 3.0, copied from the installed Cantera package's
  own data directory.
- `ammonia-CO-H2-Alzueta-2023.yaml` -- Alzueta et al. (2023) NH3/CO/H2
  mechanism, retrieved from the official
  `Cantera/cantera-example-data` GitHub repository. Contains all radicals
  the brief asks to track for the ammonia case (OH, H, O, NH2, NH, NO).
- `air-plasma-Phelps.yaml` -- bundled from the same repository purely as the
  *source* of the N2/O2 electron-neutral cross-section data used in
  `physics.py` (see below); it is not loaded as a Cantera phase by any of
  the simulation code.

Both combustion mechanisms are only thermodynamically valid up to
`gas.max_temp = 3000 K` (NASA-7 polynomials). See "Gas temperature ceiling"
below for why this matters.

## Electron-neutral cross sections (`physics.py`)

`sigma_en(species, Te)` is evaluated at the *mean* electron energy
`eps_mean = 1.5*kB*Te/e` [eV], per the `nu_en = n*sigma_en*sqrt(8kBTe/(pi
me))` formula in the brief (i.e. a single representative-energy lookup, not
a full Maxwellian-averaged rate coefficient from a Boltzmann solver).

- **N2, O2**: real energy-resolved "effective" (momentum-transfer) cross
  sections, copied verbatim (cm^2 -> m^2) from the `electron-collisions`
  blocks of `air-plasma-Phelps.yaml`. That file's own provenance note credits
  A. V. Phelps' compilation, distributed via LXCat (lxcat.net, contributor
  id d19); cite accordingly for any scientific use beyond this project.
- **CH4, NH3, H2O, CO2**: no bundled energy-resolved data exists for these,
  so single representative (energy-independent) values are used, chosen to
  be roughly right for the ~1-3 eV mean electron energies relevant here.
  These are order-of-magnitude engineering approximations, not fitted swarm
  data -- treat any quantitative resistivity/E-N comparison that hinges on
  them accordingly. NH3's value (1e-19 m^2, larger than CH4's 7e-20 m^2)
  reflects its polar/dipole character; this directly drives the ammonia
  case's higher self-consistent E/N (see below).
- All other species (minor radicals, CO, H2, etc.) fall back to a generic
  `3e-20 m^2` default.

## Electron-density closure (ionization/recombination)

The brief's electron energy equation says nothing about how `ne` itself
evolves, and standard combustion mechanisms carry no charged species. We
close this with a standard two-process avalanche model,
`dne/dt = k_ion(Te)*n_neutral*ne - k_rec(Te)*ne**2`:

- `k_ion(Te) = C_ION*sqrt(eps_eV)*exp(-Ei/eps_eV)`, a Lotz/Drawin-shaped
  electron-impact ionization rate against the *dominant* bath-gas species
  (by mass fraction -- N2 for both fuel/air mixtures here), using that
  species' ionization threshold (N2: 15.6 eV, O2: 12.06 eV, taken from the
  same `air-plasma-Phelps.yaml` reaction thresholds).
- `k_rec(Te) = K_REC0*sqrt(300/Te)`, the standard ~Te^-1/2 scaling reported
  for dissociative recombination of diatomic molecular ions (order
  1e-13 m^3/s near 300 K; see e.g. Florescu-Mitchell & Mitchell, *Phys.
  Rep.* 430 (2006), for representative magnitudes).
- `C_ION`, `K_REC0`, and the seed electron density `ReactorParams.ne0` are
  simplified, tunable parameters, **calibrated (not first-principles
  derived)** so the self-consistent reduced field `E/N = rho_res(T0,ne0)*
  j(0)/n_neutral` at breakdown onset lands near the ~250 Td the brief cites
  for the spark stage, for the *methane* case specifically. The ammonia
  case, using the same `ne0`, self-consistently comes out around ~1100 Td
  at t=0 because NH3's larger cross section raises `rho_res` at fixed `ne`
  -- this is a real, physically-explainable model output (not re-tuned
  away), not a bug, but it means the model's E/N should be read
  qualitatively, not as a validated per-fuel prediction.

**Electron energy equation and the recombination term.** Naively applying
the product rule to `eps_e = 1.5*kB*ne*Te` with `dne/dt` from the closure
above makes recombination (which can be extremely fast once `ne` is large,
deep in the arc stage) appear directly in `dTe/dt` with large-magnitude,
opposite-signed contributions that made the stiff solver fail outright
("step size less than spacing between numbers"). The fix is physical, not
just numerical: a recombining electron leaves with (on average) the
population's own mean energy, so removing it does not change the mean left
behind -- there is no reason for recombination to appear in `dTe/dt` at
all. Splitting `dne/dt = s_ion - s_rec` and writing the energy balance as

```
d(eps_e)/dt = joule_heating - q_eM - e*Ei*s_ion - 1.5*kB*Te*s_rec
```

(ionization cast as an energy sink of `Ei` per event, plus dilution by
near-zero-energy newborn electrons; recombination removing exactly the
mean energy per event) and expanding the product rule shows the `s_rec`
terms cancel exactly, leaving

```
dTe/dt = (joule_heating - q_eM)/(1.5*kB*ne) - s_ion*(e*Ei + 1.5*kB*Te)/(1.5*kB*ne)
```

with no `s_rec` dependence at all. This is what `reactor.py` implements,
and it is what made the arc-stage integration converge.

## Gas temperature ceiling

Both mechanisms' NASA-7 thermo data is valid only up to 3000 K. The raw
electron-Joule-heating power density implied by a kA-scale current confined
to a 1 mm^2 channel area easily drives the *bulk-gas* energy balance well
past that (real spark/arc channels do reach many thousands of K locally,
but the energy is rapidly spread by channel hydrodynamic expansion, thermal
dissociation/ionization, and radiation -- none of which this lumped,
constant-pressure, no-expansion 0D model represents). Extrapolating NASA
polynomials and Arrhenius rate constants far outside their fitted range is
both physically meaningless and was the direct cause of solver failure.
`PlasmaReactor` therefore clips `Tg` to `[gas.min_temp, gas.max_temp]` for
all thermo/kinetics evaluation and saturates `dTg/dt` at 0 once the ceiling
is reached, as an explicit, documented stand-in for those un-modeled energy
sinks. Both mechanisms here happen to share the same 3000 K ceiling, which
keeps the two-fuel comparison on equal footing.

## Reactor formulation

- Constant pressure (matches the brief's `rho*cp*dTg/dt = ... + sum(wdot*hk)`
  pairing of `cp` with enthalpy `hk`, rather than `cv`/internal energy for a
  constant-volume reactor).
- Chemical heat release is `q_chem = -sum(wdot_k * hk_k)` (Cantera's own
  sign convention for `net_production_rates` x `partial_molar_enthalpies`);
  the brief's `+ sum(wdot*hk)` should be read as "the heat-release term",
  not a literal sign instruction, since Cantera's convention makes that sum
  *negative* for net-exothermic chemistry.
- Wall/radiative heat loss (`q_loss`) defaults to zero (adiabatic, aside
  from the electron-coupling and chemistry terms) since the discharge
  timescales here (<=50 us) are short compared to any plausible wall heat
  transfer time; `ReactorParams.q_loss_coefficient` is provided for
  extension but inactive by default.
- Integration is split into a spark-stage `solve_ivp` call (tight `max_step`
  to resolve the ~100 ns transient) followed by an arc-stage call seeded
  from its endpoint, both using `BDF` by default. `tests/test_reactor.py`
  includes a basic convergence check (tightening `rtol` by 4 orders of
  magnitude changes the predicted ignition delay by <5%).

## Numerical robustness fixes (post-review)

An adversarial multi-reviewer code review surfaced several real issues, fixed as follows:

- **Discharge current leak (high severity, changed reported numbers).** `circuit.py`'s original
  spark->arc handoff smoothed the transition with a logistic blend, reasoning that a differentiable
  I(t) helped the stiff integrator. That reasoning didn't hold: `reactor.py` already integrates the
  spark and arc stages as two separate `solve_ivp` calls split exactly at `t_spark`, so I(t) never
  needed to be smooth *across* that boundary, only within each stage. Worse, because the blend's
  arc-side term was pinned at the full `i_peak` (not a value that itself ramps up from ~0) for all
  `t <= t_spark`, the blend leaked near-kA current into the last several `transition_width`s of the
  supposedly ~50 A, constant spark stage -- overstating Joule heating there by up to two orders of
  magnitude. Fixed by replacing the smoothed blend with a clean, unsmoothed step at `t_spark`
  (`circuit.py`). This changes the simulated results (spark-stage electron heating is now
  correctly small), so the CH4/NH3 comparison figures were regenerated after this fix.
- **Unclamped `Te`.** `Tg` and `ne` were floored/clamped before every physics evaluation, but `Te`
  was not, so a transient non-positive `Te` proposed mid-step by the stiff solver could send
  `electron_thermal_speed`'s `sqrt(Te)` to NaN, silently propagating through `rho_res`, `q_eM`, and
  both temperature derivatives. Fixed with a small `ReactorParams.Te_floor` (1 K), applied in
  `_unpack()` alongside the existing `Tg`/`ne` handling.
- **Asymmetric `Tg` ceiling.** `dTg/dt` was saturated at 0 once `Tg` reached the mechanism's upper
  thermo limit, but there was no symmetric floor guard, so a large `q_loss_coefficient` (an
  extension knob, off by default) could in principle push the raw state below the mechanism's valid
  range with nothing to stop it. Added the mirror-image floor saturation.
- **`ne0` silently overridden by `ne_floor`.** Nothing prevented `ReactorParams(ne0=...)` below
  `ne_floor`, in which case every physics evaluation would silently use `ne_floor` instead of the
  requested seed density. `ReactorParams.__post_init__` now raises `ValueError` if `ne0 <= ne_floor`.
- **Returned `Y` bypassed `_unpack`'s cleanup.** `rhs()` always evaluates chemistry on a
  clip-negatives/renormalize-to-1 copy of `Y`, but `integrate()` previously returned the raw solver
  state instead, so the exported trajectory could carry small negative mass fractions or sums off
  1 that the physics itself never actually saw. `integrate()` now applies the same cleanup to the
  returned array.
- **`postprocess.py` recomputed `rho_res` with the wrong `ne_floor`.** It called
  `physics.plasma_resistivity` without passing a floor, silently using `physics.py`'s own default
  (1e6) instead of the run's actual `ReactorParams.ne_floor`. Harmless while both defaults matched,
  but wrong for any run configured with a different floor. `ReactorResult` now carries `ne_floor`
  through so `postprocess.to_dataframe` uses the value that actually governed the simulation.

Not changed: the electron-density floor (`ne_floor=1e6`) still creates a hard (non-smoothed) kink in
`dTe/dt` exactly at `ne = ne_floor`, structurally like the `Tg` ceiling. Given `ne0` is now validated
to sit far above the floor and `ne` only approaches it through slow recombination, the floor is not
binding for either fuel case here; a smooth (e.g. softplus) floor would remove the kink in principle
but wasn't judged worth the added complexity for a boundary the trajectories never actually reach.

## 1D radial channel-expansion hydro model (`radial_hydro.py`)

Addresses the 0D model's "no channel expansion" limitation directly: a 1D
cylindrical-symmetry compressible Euler solver (mass/momentum/energy) with
a Joule-heating source term, letting the discharge channel actually push
outward and cool via PdV work instead of relying on an artificial
temperature ceiling.

**Governing equations and numerics.** `dU/dt + dF/dr = S_geom(U) + S_heat`,
`U=[rho, rho*u, E]`, first-order Godunov finite-volume update with the HLLC
approximate Riemann solver (Toro ch. 10), explicit forward-Euler time
stepping under a CFL condition, a reflective axis boundary at r=0 and a
transmissive outflow boundary at r=R_max. Validated against the exact
solution of the classic Sod shock tube (`exact_riemann.py`,
`tests/test_radial_hydro.py`) with no geometric source or heating active,
isolating the core Riemann-solver/finite-volume machinery from the
cylindrical- and heating-specific additions; L1 errors (~0.01-0.02 in
density/velocity/pressure at 200 cells) match the expected magnitude for a
first-order scheme at this resolution.

**Heating source: reused q_eM(t), not raw electrical power.** An earlier
version drove this solver with `rho_res(t)*j(t)^2` (the same raw power the
0D electron energy equation absorbs) times an assumed representative
resistivity. That is wrong, not just approximate: `rho_res*j^2` is the
power the *electron population* absorbs, which can reach extreme values
almost instantly because electrons have a tiny heat capacity -- exactly why
the 0D model's Te spikes to millions of K without Tg following. Dumping
that same raw power directly into the *gas* (with no electron intermediary
to buffer/rate-limit it) makes the gas heat as fast as the electrons did,
which is neither physically correct nor numerically tractable (translational
temperatures reached >1e11 K and velocities >1e6 m/s -- see below). The fix
was to reuse the 0D model's own `q_eM(t)` -- `physics.electron_heavy_energy_
exchange`, now exposed as a column in `postprocess.to_dataframe` -- which
*is* the actual, rate-limited electron->gas energy-transfer rate the 0D
model computed (typically 1e9-1e11 W/m^3, several orders of magnitude below
the raw `rho_res*j^2`). This is a genuine (if still one-way: hydro borrows
the 0D solution, does not feed expansion back into the electron kinetics)
coupling between the two models, not an independent guess.

**Two numerical bugs found and fixed while building this:**

1. *Axis-cell instability.* The naive cylindrical geometric source term,
   evaluated as a point value `S_geom = -F(r_i)/r_i` at each cell center,
   amplifies any small residual at the innermost cell by ~1/r_0 (very
   large, since r_0 = dr/2) -- this blew up in testing (a runaway that
   grew every iteration even as the adaptive time step collapsed toward
   1e-46 s). Fixed by integrating the geometric term exactly over each
   cell using face radii (`r_face * F_face`) rather than a point source:
   the axis face's radius is exactly 0, so its flux contributes exactly
   nothing there regardless of its value, which cannot blow up.
2. *Missing momentum correction (a real derivation error, not a numerical
   tuning issue).* The true cylindrical radial-momentum equation is
   `d(rho*u)/dt + (1/r)d(r*rho*u^2)/dr + dp/dr = 0` -- only the convective
   part picks up the `(1/r)d(r*)/dr` divergence; the pressure term is a
   plain gradient. Applying the same face-integrated divergence treatment
   used for mass/energy to the *full* momentum flux `rho*u^2+p` (as an
   initial version did, for implementation simplicity) silently adds a
   spurious `+p/r` term -- `(1/r)d(rp)/dr = dp/dr + p/r` overshoots the
   correct equation by exactly `p/r`. Fixed by adding `p_center/r` back to
   the momentum RHS after the face-integrated flux difference. The symptom
   was dramatic: perfectly uniform, at-rest gas showed a persistent, growing
   *inward* velocity
   from the very first time step, with density piling up at the axis by
   many orders of magnitude over the run. `tests/test_radial_hydro.py`'s
   `test_geometric_source_is_null_for_uniform_at_rest_state` is a
   regression test for this specifically (uniform gas at rest must have
   exactly zero net force).

**Temperature ceiling, again.** Like the 0D model, this ideal-gas solver
(fixed gamma=1.4, no dissociation/ionization) has no high-temperature heat
sink, so sustained heating can still drive temperature (and the resulting
sound speed / CFL constraint) without physical bound. `radial_hydro.run`
caps translational temperature at `T_max` (default 2e4 K) for the same
documented reason as the 0D model's `Tg` ceiling -- a stand-in for
un-modeled dissociation/ionization, not a claim that real gas stops heating
there. With the corrected q_eM(t)-based heating (much more modest than the
raw electrical power), this ceiling is rarely if ever reached in practice;
it remains as a safety net.

**A first step toward closing the feedback loop.** `core_density_trace()`
reduces a `HydroResult` to eta(t) = (area-weighted average density within
r<=r_channel) / (ambient density) -- the factor a fixed-volume 0D reactor
would need to relax its own volume assumption by. `eta_interpolator()`
gives a callable over it. Both take `r_channel` and `rho_ambient` as
required, explicit arguments rather than inferring them (e.g. from the
first saved snapshot, or a hardcoded default): a channel radius that
doesn't match what the heating source actually used silently changes which
cells get averaged, and the first *saved* snapshot is not necessarily t=0
-- both are easy, silent ways to get eta(t) subtly wrong. This is not yet
wired back into the 0D reactor itself (still one-way, per the limitation
below); `simulations/run_channel_expansion.py` saves the resulting trace as
`channel_expansion_eta.csv`/`.png`.

**`eta_interpolator` uses PCHIP, not a plain cubic spline -- found via a
real bug in a downstream consumer.** A first attempt at using eta(t) to
drive expansion cooling in a 0D-style energy equation
(`dT/dt = q/(rho*c_v) - (gamma-1)*T*dilatation_rate`, now
`gas_energy_rate_with_expansion`) took eta(t)'s derivative by finite
difference with a fixed `eps=1e-6` s. That produced a spurious ~8.7e4 /s
"expansion rate" at t=0, where the true rate is exactly zero (nothing has
happened yet -- q_eM(0)=0 too): `eps` was 10x the gap to the first real
sample (t=1e-7 s), so the difference silently straddled the flat
"nothing-has-happened-yet" extrapolation region and the real data,
fabricating a nonzero slope from a region that should be constant.
Shrinking `eps` did not fix it -- worse, it revealed the deeper problem:
`eta_interpolator`'s original plain not-a-knot cubic spline
(`interp1d(kind="cubic")`) is not itself monotonic given how sparse (8
points) and sharply-curved (eta can fall >50% between consecutive
snapshots during the fast early expansion) this trace is, so *any*
derivative of it -- finite-difference or exact -- can have the wrong sign
near a steep region (verified: near t=1e-7 s, the fitted cubic spline's
own slope is briefly *positive*, implying the core is momentarily
re-compressing, despite every sample decreasing). Switched to
`scipy.interpolate.PchipInterpolator`, which is built specifically to
preserve the monotonicity of the data it interpolates (verified
monotonic on a dense grid, where the plain cubic spline was not) and has
an exact analytic derivative (`eta_rate_interpolator`, via
`PchipInterpolator.derivative()`), sidestepping the finite-difference
step-size question entirely. `expansion_cooling_terms` and
`gas_energy_rate_with_expansion` use these exact functions rather than
approximating either.

**Known limitation of this module specifically:** the reflective axis
boundary, combined with later-time convergent flow (gas rebounding back
toward the center after the initial outward pulse, once the arc current --
and with it q_eM -- has mostly decayed), can produce numerically
unphysical axis-focusing behavior if run well past the ~10s-of-microseconds
window explored here; this is a well-known hard problem in cylindrical/
spherical gas dynamics (related to converging-shock focusing) that would
need artificial viscosity, a higher-resolution/adaptive grid near the
axis, or a higher-order scheme to handle robustly -- out of scope for this
skeleton.

## Uncapped, expansion-coupled reactor (`src/expansion_coupled_reactor.py`,
`simulations/run_closed_loop_comparison.py`)

An experimental, explicitly-scoped extension of `PlasmaReactor` (see that
module's docstring) that imposes a fuel-specific density history
`rho(t) = rho_ambient*eta(t)` -- from a `radial_hydro.CoreDensityTrace`
computed per-fuel from that fuel's *own* `q_eM(t)` -- and removes the
`Tg` ceiling entirely, to see whether hydrodynamic PdV expansion cooling
alone (with no artificial cap) can keep `Tg` inside the mechanisms'
NASA-7-valid range.

**Pipeline (`build_eta_trace` in `run_closed_loop_comparison.py`):** for
each fuel, run the validated, *capped* `PlasmaReactor` to `t_end` = 50 us
to get its `q_eM(t)` (this always reaches `t_end`, since the 3000 K
ceiling keeps its own thermo/kinetics evaluations inside the mechanism's
valid range); feed that full-length `q_eM(t)` into `radial_hydro.run` to
get a fuel-specific `CoreDensityTrace`; hand that trace to
`ExpansionCoupledReactor`, which re-integrates `Tg`/`Te`/`ne`/`Y` from
scratch with that externally-imposed density history and no ceiling.

**Results (both fuels, 0 to 50 us, `python simulations/run_closed_loop_comparison.py`):**

| Fuel/Phi | eta(t_end) | status | peak Tg | ignition delay |
|---|---|---|---|---|
| CH4/Air, phi=0.5 | 0.131 | reached t_end | 1862.5 K at t=50.0 us | 19.34 us |
| NH3/Air, phi=0.7 | 0.0138 | **FAILED** at t=7.47 us | 10,573 K | 1.11 us |

CH4 succeeds: expansion cooling alone (no ceiling) stabilizes peak core
`Tg` at 1862.5 K, comfortably inside the mechanism's valid range. NH3
fails the same way it did before this per-fuel hydro-trace pipeline
existed: the BDF arc-stage integration collapses ("step size less than
spacing between numbers") at t=7.47 us, Tg=10,573 K.

**Root-cause diagnosis: this is not an insufficient-cooling problem.**
It would be natural to assume NH3's failure means its `eta(t)` trace
just isn't aggressive enough yet, and that feeding NH3's own strong
early heating into the hydro solver (rather than reusing a weaker,
non-fuel-specific trace) would fix it. That assumption does not hold up:

- NH3's `q_eM(t)` is *already* fuel-specific and *already* spans the
  full 0-50 us range with no truncation needing extrapolation -- it
  comes from the capped `PlasmaReactor`'s own successful, full-length
  run (`gas.max_temp` clipping keeps that run numerically well-behaved
  regardless of what the raw electrical heating implies), not from the
  uncapped run that fails. There is nothing to extrapolate.
- The resulting NH3 core-density trace is already far more aggressive
  than CH4's: `eta` collapses to 0.0146 by t=10 us and 0.0138 by t=50 us
  -- roughly **10x deeper core evacuation than CH4's 0.131**, precisely
  because NH3's larger electron-neutral cross section (see "Electron-
  neutral cross sections" above) drives stronger early Joule heating.
  So the "iterate the hydro trace with NH3's actual heating" step this
  diagnosis called for was already what the existing pipeline does, and
  it still fails at the same t=7.47 us / ~10,533-10,573 K point a
  weaker/generic trace would.
- Instrumenting `ExpansionCoupledReactor.rhs` at the failure point shows
  why: at t=7.4724 us (`eta`=0.0494, `rho`=0.0582 kg/m^3), `q_eM` and
  `q_chem` are both finite and essentially frozen across the solver's
  repeated Newton-iteration retries at that same t (~7.9e11 and
  -5.3e11 W/m^3), yet the reported `dTg/dt` swings from -3.1e13 K/s to
  **-1.3e16 K/s** and back across those retries with the state barely
  moving. That is the signature of a near-*singular* `1/cv` term, not a
  heating/cooling imbalance -- expansion cooling (`dilatation_rate` sits
  at a healthy 5.25e5 /s throughout) is not the thing failing.
- Directly sweeping `gas.cv_mass` for this mechanism at fixed density
  confirms it: `cv_mass` decreases smoothly from ~1341 J/(kg K) at
  3000 K to a maximum near 5000 K, then **falls through zero between
  8500 K and 9000 K and goes negative beyond that** (-388 J/(kg K) at
  9000 K, -3632 at 10,500 K) -- an unambiguous artifact of extrapolating
  the Alzueta et al. (2023) mechanism's NASA-7 polynomials roughly 3x
  past their fitted 3000 K ceiling, not a real thermodynamic property.
  `dTg/dt = (q_eM+q_chem-q_loss)/(rho*cv) - (gamma-1)*Tg*dilatation_rate`
  is singular exactly at `cv=0` and sign-flipped past it, which is what
  drives the BDF step size to collapse: the solver is correctly refusing
  to step through a discontinuity the model itself creates.
- NH3's very short ignition delay (1.11 us, vs CH4's 19.34 us at the
  same heating pipeline) is what puts this mechanism in reach at all:
  its chemical heat release happens early, while `eta` is still close to
  1 (0.72 at t=1 us) and the core has barely expanded, so exothermic
  chemistry pushes `Tg` past 3000 K and on toward the mechanism's
  `cv`-zero-crossing before the (already-aggressive) rarefaction has had
  time to remove much energy per unit mass. By the time `eta` has
  collapsed to its asymptotic ~0.014 (t>~10 us), the damage from that
  early exotherm is already done.

**Conclusion:** intensifying or otherwise iterating NH3's `eta(t)` trace
cannot fix this failure mode, because the failure is not a shortfall in
PdV cooling capacity -- it is the mechanism's own equation of state
becoming unphysical (negative heat capacity) once extrapolated far
enough past its 3000 K validity ceiling, something no density history,
however aggressive, changes. A genuine fix would need either
high-temperature-valid thermodynamic data for this mechanism (e.g.
extended/NASA-9 polynomials with dissociation chemistry, which the
Alzueta et al. mechanism does not provide), or some explicit, documented
stand-in for missing high-T physics -- exactly the role the original
`PlasmaReactor` ceiling already plays.

### Follow-up: numerical safeguards attempted, and where NH3 stands now

The conclusion above didn't end the investigation -- several numerical
safeguards were tried, in sequence, to see whether the singular-`cv`
failure could be made survivable without reintroducing a hard `Tg`
ceiling. Each attempt is kept here because each one taught something
about *why* the next was needed; none of them, individually or combined,
gets NH3 to `t_end`, but the end state is a much better-characterized
failure than the original BDF step-collapse.

1. **A flat `cv` floor (1500 J/(kg K), unconditional) plus a `Tg^4`
   radiative-loss sink.** Fixed NH3's original BDF collapse but silently
   broke CH4: peak `Tg` dropped from 1862.5 K to 1248.8 K and it stopped
   igniting at all. Root cause turned out to be the `cv` floor itself
   (not the radiative term): CH4/air's real `cv` is ~750-1100 J/(kg K)
   across its *entire* pre-ignition range (verified directly), so an
   unconditional 1500 J/(kg K) floor was active for the whole run, not
   just the intended high-`Tg` extrapolated regime, roughly doubling
   apparent thermal inertia throughout and suppressing the exotherm that
   drives ignition.
2. **Temperature-gated `cv` floor** (a smooth sigmoid, ~0 below 2000 K,
   ~1 above 3000 K) restored CH4 to its original 1862.4 K / 19.34 us
   ignition delay. NH3, however, now diverged to unbounded runaway
   heating (`Tg` climbing past 12,700 K with no sign of leveling off,
   5+ minutes of wall time, never converging or erroring) instead of
   cleanly failing. Cause: `gamma = cp/cv_effective` still used the raw,
   *unguarded* `cp`, which also goes negative in the same extrapolated
   region (about -3323 J/(kg K) at 10,500 K, confirmed by direct sweep)
   -- flipping the sign of the PdV expansion-cooling term
   `(gamma-1)*Tg*dilatation_rate` into spurious additional heating.
3. **`cp` floored via Mayer's relation** (`cp_effective = max(cp,
   cv_effective + R_specific)`, so `gamma = cp_effective/cv_effective` is
   guaranteed `> 1` wherever the gate is active) removed the runaway.
   Instead, NH3 crashed the whole Python process outright with a Windows
   access violation (confirmed via `faulthandler`) inside Cantera's C++
   core. Cause, traced through `_unpack`: Radau's internal
   finite-difference Jacobian probing, deep in this same stiff region,
   proposed a trial state with an `inf` mass-fraction entry (from scipy's
   own step-size-adaptation overflow); the old `if total > 0.0: Y =
   Y/total` guard let it through (`inf > 0.0` is `True`), producing a NaN
   composition via `inf/inf` that was then handed straight to Cantera
   with no validation.
4. **A finite-state guard in `_unpack`/`rhs`** (reject any non-finite
   trial state and return a NaN derivative array instead of touching
   Cantera) stopped the native crash, but exposed a *different* uncaught
   failure: enough NaN-tainted `rhs` evaluations built into Radau's own
   numerical Jacobian that scipy's LU factorization raised a bare
   `ValueError: array must not contain infs or NaNs` -- one level below
   `solve_ivp`'s normal `.success`/`.message` reporting, so nothing in
   `ExpansionCoupledReactor.integrate()` caught it and the whole script
   aborted before CH4's results (already computed successfully) or the
   comparison plot could be written out.
5. **`try/except (ValueError, RuntimeError)` around both `solve_ivp`
   calls** in `integrate()`, routing a caught exception through the same
   `_partial_result(..., reached_t_end=False, message=...)` path as an
   ordinary solver failure, finally lets the script complete end-to-end.

**Current final state** (`python simulations/run_closed_loop_comparison.py`):

| Fuel/Phi | status | peak Tg (reported) | ignition delay |
|---|---|---|---|
| CH4/Air, phi=0.5 | reached t_end | 1862.4 K at t=50.0 us | 19.34 us |
| NH3/Air, phi=0.7 | **FAILED**, caught cleanly | 307.5 K at t=0.1 us (see caveat) | never reached 1500 K |

CH4 is fully restored to its original behavior. NH3 no longer crashes
the process; `run_uncapped()` reports `status: FAILED:
ExpansionCoupledReactor.integrate() did not reach t_end: Arc-stage
solver exception (Jacobian LU failure): array must not contain infs or
NaNs` -- an honest, catchable description of exactly what happened,
traceable back through this whole chain to the same root cause identified
at the very top of this section: the mechanism's NASA-7 polynomials
producing an unphysical (negative, then NaN-adjacent) heat capacity once
extrapolated past ~9000 K.

**Reporting caveat -- read the peak-Tg column with the heartbeat log, not
alone.** When the arc-stage `solve_ivp` call raises instead of returning
normally, `sol_arc` is never assigned, so `integrate()`'s except-branch
has nothing to fall back to but the spark-stage trajectory (the first
0.1 us) -- there is no scipy-level mechanism to recover partial progress
from a call that raised mid-integration. That makes the table's "peak Tg
reached: 307.5 K" technically accurate for what `ExpansionReactorResult`
actually contains, but badly misleading read on its own: the
`progress_interval_s` heartbeat log from this same run shows NH3 actually
reached **Tg > 10,100 K by t=8.5 us** before the arc-stage integrator's
Jacobian factorization failed. Recovering that intermediate trajectory
for reporting (rather than just a console heartbeat) would need a
different integration strategy -- e.g. stepping `solve_ivp` manually in a
loop with `dense_output`/small fixed spans and checkpointing `y` after
each successful span, rather than one long call per stage -- which has
not been implemented. `data/temperature_trajectories_uncapped.png` and
the console summary should be read with this in mind: NH3's curve is
real up to where it's plotted, but the run went further (and hotter)
than either currently show before failing.

**Bottom line, as of the safeguards above:** no combination of them got
NH3/Air (phi=0.7) to `t_end` -- every one either crashed, ran away, or
stalled, always somewhere past ~9000-11,000 K. What changed across them
was *how* it failed -- from an opaque BDF step-size collapse, through a
silent CH4 regression, an unbounded-runaway stall, and a native process
crash, to a clean, catchable, accurately-described `ValueError` --
real, meaningful progress for debugging, even though none of it actually
closed the gap to `t_end` on its own.

### Resolution: a soft ceiling, plus solver tolerances that can actually traverse it

Two more changes finally get both fuels to `t_end`:

1. **A 5000 K thermal soft-cap** on `rhs`'s computed `dTg_dt`: once
   `Tg > 3500 K` *and* the computed rate is positive (heating, never
   cooling), it's scaled by `0.5*(1 - tanh((Tg-4500)/250))` -- ~1 well
   below 4000 K, ~0 by ~5000 K. This is, honestly, the same kind of
   device as `PlasmaReactor`'s original hard 3000 K ceiling: an explicit,
   documented stand-in for the missing high-temperature
   dissociation/ionization physics, just smoothed and moved higher (5000 K
   vs 3000 K) rather than hard-clipped. Unlike a hard ceiling, a hot
   parcel can still cool back down through this range at its full,
   uncapped rate -- it only throttles further self-heating, not cooling.
   Applied alone, this got NH3 to t=42.47 us (up from the prior best of
   24.4 us) with Tg genuinely stabilizing near 5067 K, rather than
   crashing/stalling/running away -- but the arc-stage integration still
   occasionally hit a non-finite trial state while finely probing that
   now-stable plateau and raised the same `ValueError` as before, just
   much later.
2. **Relaxed arc-stage solver tolerances**: `rtol` 1e-6 -> 1e-4, `atol`
   scaled 100x, `max_step` loosened (200 sub-steps -> 100 across the arc
   stage). The spark stage (resolving the ~100 ns breakdown transient)
   keeps its original tight tolerances -- only the arc stage, where Tg has
   already settled near the soft-cap plateau, is relaxed. This matters
   because that residual failure was Radau's finite-difference Jacobian
   over-probing a region where extreme numerical precision isn't
   physically meaningful anyway (the effective `cv`/`gamma` there are
   themselves explicit stand-ins, not exact data) -- asking for less
   precision there gives it room to coast through instead of repeatedly
   landing on a non-finite trial state.

**Final result** (`python simulations/run_closed_loop_comparison.py`, both
fuels reach `t_end` for the first time in this whole investigation):

| Fuel/Phi | status | peak Tg | ignition delay |
|---|---|---|---|
| CH4/Air, phi=0.5 | reached t_end | 1862.7 K at t=50.0 us | 19.343 us |
| NH3/Air, phi=0.7 | reached t_end | 5075.1 K at t=50.0 us | 1.111 us |

CH4 is essentially unchanged from its original validated baseline (the
spark stage, and hence CH4's ~1862 K trajectory which never approaches
the soft-cap's 3500 K floor, is untouched by either change). NH3 now
completes the full 50 us window with its core temperature held near
5075 K by the soft-cap, rather than crashing or stalling somewhere past
9000 K.

**How to read this honestly:** NH3's 5075 K "peak temperature" is not a
first-principles prediction -- it is the location this investigation's
explicit numerical stand-in (the soft-cap) was set to hold the gas at,
chosen because it was the point past which every other combination of
safeguards eventually failed regardless of how the energy balance was
modeled. That is the same epistemic status as `PlasmaReactor`'s original
3000 K hard ceiling, just a different, smoother, and empirically-tuned
value -- not evidence that real NH3/air plasma actually equilibrates
near 5000 K. Getting a genuinely first-principles answer for NH3's peak
core temperature still requires either high-temperature-valid
thermodynamic data (extended/NASA-9 polynomials with real
dissociation/ionization chemistry) for the Alzueta et al. mechanism, or
an explicitly physically-derived (not tuned-to-not-crash) high-T energy
sink. What this investigation does establish reliably: CH4/Air at
phi=0.5 stabilizes on its own, via PdV expansion cooling alone, at a
peak core temperature of 1862 K -- comfortably inside the mechanism's
valid thermodynamic range, needing none of these safeguards.

### Parametric equivalence-ratio sweep, and a real bug found in `physics.py`

`simulations/run_parametric_phi_sweep.py` runs the stabilized
`ExpansionCoupledReactor` (the soft-cap/relaxed-tolerance configuration
above) across phi = 0.4-1.2 for both fuels, building a fresh fuel/phi-
specific `eta(t)` trace for every point (q_eM(t) depends on mixture
composition, hence phi, so a trace built at one phi cannot be reused at
another). Ignition delay is defined as the time Tg first crosses
T0+400 K post-spark (not a raw dTg/dt peak: an early version of that
definition was contaminated by the spark->arc current step itself, which
produces a large, purely-numerical dTg/dt spike right at t_spark that
swamped any real, later chemical-ignition signal).

**The first sweep surfaced a real, unexplained irregularity.** NH3 came
out smooth and monotonic across the whole range. CH4 did not: peak Tg
rose smoothly through phi=0.8 (1554-2309 K), then zigzagged erratically
for phi>=0.9 -- 4664 K, then a *dip* to 2504 K at phi=1.0 (exactly
stoichiometric, where peak temperature should typically be highest, not
lowest), then back up to 4231 K and 4890 K. Not physically plausible for
an adiabatic-flame-temperature-like curve.

**Four fix attempts targeting the thermal/chemistry side of the energy
balance, in sequence, each failed to resolve it:**

1. An external high-T energy sink (`q_comb_damp`, gated ~2500-2800 K)
   reduced the zigzag's amplitude but did not eliminate it, and
   substantially regressed NH3's entire ~4000-5200 K operating range
   (4751-5226 K -> 3991-5101 K) as an unintended side effect -- its
   quartic form stayed large well past where its gate nominally turned
   "on".
2. Gating `q_chem`'s exothermic part above ~2500-2800 K (removing
   `q_comb_damp`, suppressing untrustworthy extrapolated combustion
   exothermicity at its source instead) fixed the NH3 regression but had
   *zero* measurable effect on CH4's zigzag -- it came back to
   essentially the original amplitude (4638/2503/4158/4885 K).
3. A "cv thermal trapdoor" fix (the smooth monatomic-limit blend for
   `cv_sensible`, which forced cv down toward `cv_mono` even where the
   raw NASA-7 value was still valid and several times larger -- a
   self-reinforcing loss of thermal inertia right as Tg crossed ~3000 K
   -- replaced with a strict `max(cv_raw, cv_mono)` floor) was a
   real, independently-justified fix to a genuine problem, but also had
   essentially no effect on the zigzag (4573/2492/3990/4848 K).
4. `simulations/debug_ch4_bifurcation.py` (added specifically to stop
   guessing at more energy-balance terms and instead trace the *actual*
   term-by-term RHS solve_ivp produced for phi=0.9 vs phi=1.0, via a new
   `ExpansionCoupledReactor.rhs(t, y, return_terms=True)` diagnostics
   hook) found the task brief's assumed divergence window (the first
   15 us) was itself wrong -- both trajectories stay under 750 K there;
   the real separation starts around t~17 us and grows through the rest
   of the run, with each case's peak Tg landing at t=50 us (t_end)
   itself, not an interior peak. At the point of largest divergence
   (t~33 us, phi=0.9 at 4217 K vs phi=1.0 at 2219 K), `q_eM` was ~20x
   larger for phi=0.9 (1.17e10 vs 5.73e8 W/m^3) while `q_chem` and PdV
   cooling differed by comparable, much smaller factors -- identifying
   `q_eM`, not chemistry or heat capacity, as the actual driver, which is
   exactly why none of the first three attempts (all on the
   thermal/chemistry side) could have worked.

**Root cause, found by reading `physics.electron_heavy_energy_exchange`'s
source directly:** `q_eM = sum_k (2*me/Mk)*nu_ek * 1.5*kB*(Te-Tg)*ne`.
Each species' contribution is weighted by `1/Mk`, so light dissociation
fragments (atomic H, Mk~1 g/mol) contribute far out of proportion to
their actual number or heat content once a mixture starts dissociating.
This creates a genuine positive-feedback loop intrinsic to the physics
formula itself: hotter -> more H/H2 dissociation -> larger
`sum_k(2*me/Mk)*nu_ek` -> larger q_eM -> more heating -- with no
numerical safeguard on the thermal-inertia or chemistry side able to
touch it, because the runaway lived entirely in this term. (Checking
Te/ne directly at the divergence point showed Te nearly identical between
the two phi cases and ne only ~1.8x higher for phi=0.9 -- nowhere near
enough to explain a 20x q_eM gap on their own, pointing at the
composition-dependent `1/Mk` weighting as the remaining, dominant factor.)

**Fix:** `electron_heavy_energy_exchange` (`src/physics.py`) now uses the
mixture's bulk *mean* molecular weight in the elastic energy-transfer
factor (`2*me/M_mean`, applied to the total collision frequency
`nu_en(Te)` via the existing `collision_frequency` helper) instead of
summing `2*me/Mk` per species. This is a change to a function used by
`PlasmaReactor` (the validated, published model) as well as
`ExpansionCoupledReactor`, not just the experimental uncapped path --
every q_eM(t) profile in this project, including the original CH4/NH3
comparison, is computed by this function. One deliberate deviation from
an initial draft of this fix: it did *not* add an `if Te <= Tg: return
0.0` early-out, because that would have silently broken
`tests/test_physics.py::test_energy_exchange_sign_follows_temperature_
difference`, which deliberately checks that q_eM goes *negative* when
Te < Tg (energy flowing gas -> electrons) -- a real, intentional,
already-tested physical behavior unrelated to the dissociation-feedback
bug being fixed here.

**Result: this is the fix that actually worked.** Re-running the phi
sweep afterward:

| Fuel/Phi | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 |
|---|---|---|---|---|---|---|---|---|---|
| CH4 peak Tg [K] | 1105 | 1314 | 1516 | 1711 | 1898 | 2063 | 2137 | 2170 | 2161 |
| NH3 peak Tg [K] | 3611 | 3853 | 4114 | 4293 | 4428 | 4536 | 4620 | 4681 | 4726 |

Both fuels are now fully smooth and monotonic (CH4 peaks slightly rich of
stoichiometric, at phi=1.1, then eases very slightly at phi=1.2 -- the
qualitatively correct shape for an adiabatic-flame-temperature-like
curve, not an artifact). Ignition delay is likewise monotonic for both
fuels across the whole range. This resolves an issue none of the four
prior, more targeted attempts could -- because it was the first one that
actually addressed the mechanism doing the damage, rather than adding a
new safeguard to a part of the model the runaway didn't live in.

Both fuels' absolute peak-Tg magnitudes shifted down from their pre-fix
values (CH4's phi=0.5 point: 1862 K -> 1314 K; NH3's: ~4900 K -> ~3853 K)
because the mean-molecular-weight regularization reduces q_eM overall,
not just its pathological high-dissociation spikes -- this is expected
(the per-species form was always somewhat too generous whenever *any*
light species was present, not only in the runaway regime) but means
every previously-reported peak-temperature/ignition-delay number in this
document, including the original validated CH4/NH3 comparison, reflects
the old, now-superseded q_eM formula and would shift somewhat if
regenerated.

## Known limitations

- The 0D lumped reactor still has no *self-consistent* hydrodynamic
  expansion feedback -- `radial_hydro.py` demonstrates the channel
  expanding and cooling given the 0D model's own heating profile, one-way,
  but the 0D reactor itself still runs as a fixed-volume parcel with an
  artificial temperature ceiling standing in for the energy that expansion
  would otherwise carry away. Peak "gas temperature" in the 0D results
  should still be read as an upper bound on a highly localized channel.
- The ionization/recombination closure and non-N2/O2 cross sections are
  simplified engineering approximations calibrated for plausible order of
  magnitude and qualitative behavior, not fitted to experimental spark data.
- `E/N` is reported as a diagnostic derived from `rho_res(t)*j(t)`, not
  imposed as an independent input.
