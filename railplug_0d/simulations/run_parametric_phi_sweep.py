"""Parametric equivalence-ratio sweep, CH4 vs NH3, using the stabilized
`ExpansionCoupledReactor` (Radau integration, 5000 K thermal soft-cap on
positive heating, gated/narrowed dissociation latent-heat buffer, Mayer's-
relation-consistent Cp/gamma, finite-state guard, and a bounded per-stage
evaluation cap -- see `src/expansion_coupled_reactor.py` and
MODELING_NOTES.md's "Uncapped, expansion-coupled reactor" section for the
full history of why each of those exists).

For each (fuel, phi) pair: build a fuel/phi-specific core-density trace
eta(t) -- capped `PlasmaReactor` -> q_eM(t) -> 1D radial hydro
(`radial_hydro.py`) -> `CoreDensityTrace` -- then run the uncapped,
expansion-coupled reactor across that trace to phi's ignition/thermal
behavior. Every (fuel, phi) point gets its own trace: q_eM(t) (and hence
eta(t)) depends on the mixture's own electron-neutral collision physics,
which is composition- (and therefore phi-) dependent, so a single fixed
trace reused across phi would silently mix each point's actual heating
history with another point's expansion profile.

Writes `data/phi_sweep_results.csv` (one row per (fuel, phi)) and
`data/phi_sweep_ignition_delay.png` (ignition delay and peak core
temperature vs phi, both fuels).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt

from src.reactor import PlasmaReactor, ReactorParams
from src.circuit import CircuitParams
from src import postprocess
from src import radial_hydro as rh
from src.expansion_coupled_reactor import ExpansionCoupledReactor, ExpansionReactorParams

import run_methane
import run_ammonia
import run_channel_expansion as chan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# --- Sweep configuration ---
PHI_VALUES = [round(0.4 + 0.1 * i, 1) for i in range(9)]  # 0.4, 0.5, ..., 1.2 (9 points)
T0 = 300.0
P0 = ct.one_atm
CIRCUIT = CircuitParams(t_spark=0.5e-6, t_end=50.0e-6)  # 0.5 us spark, 50 us total

FUELS = {
    "CH4": {
        "mechanism": run_methane.MECHANISM,
        "build_gas": lambda phi: run_methane.build_gas(phi=phi, T0=T0, P0=P0),
    },
    "NH3": {
        "mechanism": run_ammonia.MECHANISM,
        "build_gas": lambda phi: run_ammonia.build_gas(phi=phi, T0=T0, P0=P0),
    },
}


def build_eta_trace(mechanism: str, build_gas, circuit: CircuitParams) -> rh.CoreDensityTrace:
    """Fuel/phi-specific: capped 0D reactor -> q_eM(t) -> 1D hydro -> core
    density trace. `build_gas` is a no-arg callable (already bound to this
    point's phi) returning a fresh `ct.Solution`."""
    gas = build_gas()
    reactor = PlasmaReactor(gas, ReactorParams(mechanism=mechanism, T0=T0, P0=P0, circuit=circuit))
    result = reactor.integrate()
    df_0d = postprocess.to_dataframe(result)

    hydro_result = chan.run_hydro(df_0d)
    ambient = rh.AmbientState(T0=T0, p0=P0)
    heating = rh.HeatingParams.from_0d_dataframe(df_0d)
    return rh.core_density_trace(hydro_result, r_channel=heating.r_channel, rho_ambient=ambient.rho0)


def ignition_delay_us(t: np.ndarray, Tg: np.ndarray, t_spark: float, T0: float,
                       delta_T: float = 400.0) -> float | None:
    """Ignition delay: time Tg first crosses T0 + delta_T post-spark,
    linearly interpolated between the bracketing samples for sub-step
    precision (same style as `ignition_delay_from_arrays` in
    run_closed_loop_comparison.py, just at a 400 K rise instead of a fixed
    1500 K).

    This is a *threshold-crossing* definition, not the "argmax(dTg/dt)"
    approach the task brief's phrasing initially suggested -- deliberately
    changed after finding that a direct dTg/dt-peak search reported ~0.5 us
    "ignition delays" for cases that visibly don't heat up until tens of
    us later. Cause: the spark->arc current step at t=t_spark is itself a
    genuine discontinuity in the forcing (see circuit.py), which produces
    a large, purely-numerical spike in dTg/dt right at that boundary --
    completely swamping the much smaller, smoother rate of any real later
    chemical ignition. A threshold crossing on Tg itself doesn't have that
    problem. Returns None if Tg never crosses the threshold post-spark
    (no ignition).
    """
    post = t > t_spark
    if not np.any(post):
        return None
    t_post, Tg_post = t[post], Tg[post]
    threshold = T0 + delta_T
    above = Tg_post >= threshold
    if not np.any(above):
        return None
    i = int(np.argmax(above))  # index of first True
    if i == 0:
        return float(t_post[0] * 1e6)
    t_lo, t_hi = t_post[i - 1], t_post[i]
    Tg_lo, Tg_hi = Tg_post[i - 1], Tg_post[i]
    frac = (threshold - Tg_lo) / (Tg_hi - Tg_lo) if Tg_hi != Tg_lo else 0.0
    return float((t_lo + frac * (t_hi - t_lo)) * 1e6)


def run_one(fuel: str, phi: float, mechanism: str, build_gas) -> dict:
    t_wall_start = time.perf_counter()
    trace = build_eta_trace(mechanism, lambda: build_gas(phi), CIRCUIT)

    gas = build_gas(phi)
    params = ExpansionReactorParams(mechanism=mechanism, trace=trace, T0=T0, circuit=CIRCUIT)
    reactor = ExpansionCoupledReactor(gas, params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reactor.integrate()
        failure_message = None
        for w in caught:
            msg = str(w.message)
            if "did not reach t_end" in msg:
                failure_message = msg
                break
    wall_time_s = time.perf_counter() - t_wall_start

    tau = ignition_delay_us(result.t, result.Tg, CIRCUIT.t_spark, T0)
    return {
        "fuel": fuel,
        "phi": phi,
        "reached_t_end": bool(result.reached_t_end),
        "peak_Tg_K": float(result.Tg.max()),
        "ignition_delay_us": tau,
        "simulation_wall_time_s": wall_time_s,
        "failure_message": failure_message,
    }


def main() -> None:
    rows = []
    for fuel, cfg in FUELS.items():
        for phi in PHI_VALUES:
            print(f"--- {fuel} phi={phi:.1f} ---", flush=True)
            row = run_one(fuel, float(phi), cfg["mechanism"], cfg["build_gas"])
            rows.append(row)
            tau_str = f"{row['ignition_delay_us']:.3f} us" if row["ignition_delay_us"] is not None else "no ignition"
            print(f"    reached_t_end={row['reached_t_end']}  peak_Tg={row['peak_Tg_K']:.1f} K  "
                  f"tau_ign={tau_str}  wall_time={row['simulation_wall_time_s']:.1f} s", flush=True)
            if row["failure_message"]:
                print(f"    ({row['failure_message']})", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "phi_sweep_results.csv")
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    colors = {"CH4": "tab:blue", "NH3": "tab:orange"}
    markers = {"CH4": "o", "NH3": "s"}
    for fuel in FUELS:
        sub = df[df["fuel"] == fuel].sort_values("phi")
        color, marker = colors[fuel], markers[fuel]
        axes[0].plot(sub["phi"], sub["ignition_delay_us"], marker=marker, linestyle="-",
                     color=color, label=fuel)
        failed = sub[~sub["reached_t_end"]]
        if not failed.empty:
            axes[1].scatter(failed["phi"], failed["peak_Tg_K"], marker="x", color=color,
                             s=80, zorder=5, label=f"{fuel} (did not reach t_end)")
        axes[1].plot(sub["phi"], sub["peak_Tg_K"], marker=marker, linestyle="-",
                     color=color, label=fuel)

    axes[0].set_ylabel(r"ignition delay $\tau_{ign}$ [us]")
    axes[0].set_title("Ignition delay vs equivalence ratio")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_ylabel(r"peak core $T_g$ [K]")
    axes[1].set_xlabel(r"equivalence ratio $\phi$")
    axes[1].set_title("Peak core temperature vs equivalence ratio")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("Uncapped, expansion-coupled reactor: phi sweep (CH4 vs NH3)")
    fig.tight_layout()
    png_path = os.path.join(DATA_DIR, "phi_sweep_ignition_delay.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print("\n=== Summary table ===")
    display_cols = ["fuel", "phi", "reached_t_end", "peak_Tg_K", "ignition_delay_us", "simulation_wall_time_s"]
    print(df[display_cols].to_string(index=False))
    print(f"\nCSV written to {csv_path}")
    print(f"Plot written to {png_path}")


if __name__ == "__main__":
    main()
