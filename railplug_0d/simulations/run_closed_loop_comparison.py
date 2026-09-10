"""Uncapped CH4 vs NH3 comparison, with hydrodynamic expansion cooling
wired in from each fuel's own 1D radial hydro trace, and NO Tg ceiling.

This is an explicitly experimental, higher-risk companion to
simulations/compare.py (which uses the validated, capped PlasmaReactor and
remains the primary published comparison). See
src/expansion_coupled_reactor.py's module docstring for exactly what is and
is not different here, and why removing the ceiling is expected to be
risky rather than a safe cleanup.

Builds a SEPARATE eta(t) trace per fuel (q_eM(t) is fuel-specific -- CH4's
trace must not be reused for NH3), runs ExpansionCoupledReactor for both,
and reports whichever of {reached t_end with some peak Tg, or failed at
some t with the solver's own error} actually happened -- not a forced
"success".
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cantera as ct
import matplotlib.pyplot as plt

from src.circuit import CircuitParams
from src import radial_hydro as rh
from src.expansion_coupled_reactor import ExpansionCoupledReactor, ExpansionReactorParams
import run_channel_expansion as chan
import run_methane
import run_ammonia

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

FUELS = {
    "CH4/Air (phi=0.5)": {
        "mechanism": run_methane.MECHANISM,
        "build_gas": lambda mech: run_methane.build_gas(),
        "color": "tab:blue",
    },
    "NH3/Air (phi=0.7)": {
        "mechanism": run_ammonia.MECHANISM,
        "build_gas": lambda mech: run_ammonia.build_gas(),
        "color": "tab:orange",
    },
}


def build_eta_trace(mechanism: str, build_gas) -> rh.CoreDensityTrace:
    """Fuel-specific: 0D reactor -> q_eM(t) -> 1D hydro -> core density trace."""
    df_0d = chan.run_0d_heating_profile(mechanism, build_gas=lambda mech: build_gas(mech))
    hydro_result = chan.run_hydro(df_0d)
    ambient = rh.AmbientState(T0=300.0, p0=101325.0)
    heating = rh.HeatingParams.from_0d_dataframe(df_0d)
    return rh.core_density_trace(hydro_result, r_channel=heating.r_channel, rho_ambient=ambient.rho0)


def run_uncapped(mechanism: str, build_gas, trace: rh.CoreDensityTrace, progress_interval_s: float | None = 5.0):
    gas = build_gas(mechanism)
    params = ExpansionReactorParams(mechanism=mechanism, trace=trace, circuit=CircuitParams())
    reactor = ExpansionCoupledReactor(gas, params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reactor.integrate(progress_interval_s=progress_interval_s)
        # `integrate()` can emit many incidental RuntimeWarnings from scipy
        # (e.g. "overflow encountered in multiply" in num_jac's step-size
        # adaptation) well before it actually terminates. Grabbing
        # `caught[0]` (the first warning chronologically, an earlier
        # version of this function did exactly that) silently reports one
        # of those instead of the real reason: `_partial_result`'s own
        # "ExpansionCoupledReactor.integrate() did not reach t_end: ..."
        # warning, which is always the *last* one raised (it's emitted
        # right when integrate() returns its failure result). Search for
        # that specific message rather than assuming position.
        failure_message = None
        for w in caught:
            msg = str(w.message)
            if "did not reach t_end" in msg:
                failure_message = msg
                break
        if failure_message is None and caught:
            failure_message = str(caught[-1].message)
    return result, failure_message


def ignition_delay_from_arrays(t, Tg, T_threshold=1500.0):
    above = np.flatnonzero(Tg >= T_threshold)
    if above.size == 0:
        return None
    i = above[0]
    if i == 0:
        return float(t[0])
    frac = (T_threshold - Tg[i - 1]) / (Tg[i] - Tg[i - 1])
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def main() -> None:
    results = {}
    for name, cfg in FUELS.items():
        print(f"--- {name}: building fuel-specific eta(t) trace ---", flush=True)
        trace = build_eta_trace(cfg["mechanism"], cfg["build_gas"])
        print(f"    eta(t_end) = {trace.eta[-1]:.4f} (r_channel={trace.r_channel*1e3:.3f} mm)", flush=True)

        print(f"--- {name}: running uncapped, expansion-coupled reactor ---", flush=True)
        result, failure_message = run_uncapped(cfg["mechanism"], cfg["build_gas"], trace)
        results[name] = (result, failure_message)

        status = "reached t_end" if result.reached_t_end else f"FAILED: {failure_message}"
        tau = ignition_delay_from_arrays(result.t, result.Tg)
        tau_str = f"{tau*1e6:.3f} us" if tau is not None else "never reached 1500 K"
        print(f"    status: {status}", flush=True)
        print(f"    peak Tg reached: {result.Tg.max():.1f} K at t={result.t[np.argmax(result.Tg)]*1e6:.3f} us "
              f"(last computed t={result.t[-1]*1e6:.3f} us)", flush=True)
        print(f"    ignition delay (Tg>=1500K): {tau_str}", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, cfg in FUELS.items():
        result, failure_message = results[name]
        t_us = result.t * 1e6
        marker = "" if result.reached_t_end else " (solver stopped early)"
        ax.plot(t_us, result.Tg, color=cfg["color"], linestyle="-", label=f"{name} Tg{marker}")
        ax.plot(t_us, result.Te, color=cfg["color"], linestyle="--", label=f"{name} Te{marker}")
        if not result.reached_t_end:
            ax.axvline(t_us[-1], color=cfg["color"], linestyle=":", alpha=0.5)

    ax.axhline(3000.0, color="k", linestyle=":", alpha=0.4, label="old 3000 K ceiling (removed here)")
    ax.set_xlabel("time [us]")
    ax.set_ylabel("temperature [K]")
    ax.set_yscale("log")
    ax.set_title("Uncapped Tg/Te with expansion cooling (experimental -- see script docstring)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(DATA_DIR, "temperature_trajectories_uncapped.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"\nPlot written to {outpath}")


if __name__ == "__main__":
    main()
