# Modeling notes

This document records the modeling choices, data provenance, and known
limitations behind the code in `src/`. The governing equations in the
project brief specify electron energy balance, gas energy/species balance,
and the two-stage discharge current, but leave several closures
unspecified (notably electron-density kinetics and the constant-pressure
sign convention); this file explains how those gaps were filled and why.

## Usage

```
pip install -r requirements.txt        # or requirements-dev.txt to also get pytest
python simulations/run_methane.py      # CH4/Air, phi=0.5
python simulations/run_ammonia.py      # NH3/Air, phi=0.7
python simulations/compare.py          # both cases + comparison plots/summary in data/
pytest tests/                          # unit tests + a tolerance-convergence check
```

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

## Known limitations

- No spatial/hydrodynamic channel expansion -- this is a lumped 0D model,
  so peak "gas temperature" during the discharge should be read as an upper
  bound on a highly localized channel, not a flame-kernel-average
  temperature.
- The ionization/recombination closure and non-N2/O2 cross sections are
  simplified engineering approximations calibrated for plausible order of
  magnitude and qualitative behavior, not fitted to experimental spark data.
- `E/N` is reported as a diagnostic derived from `rho_res(t)*j(t)`, not
  imposed as an independent input.
