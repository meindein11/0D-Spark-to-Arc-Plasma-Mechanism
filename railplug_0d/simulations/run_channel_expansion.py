"""1D radial channel-expansion simulation, coupled to the 0D CH4/Air case.

Runs the existing 0D PlasmaReactor (methane, phi=0.5) to get its computed
electron->gas heating rate q_eM(t), then uses that as the volumetric Joule
heating source for the 1D radial compressible Euler solver (radial_hydro.py)
-- demonstrating hydrodynamic channel expansion, the 0D model's main
documented limitation (see MODELING_NOTES.md).

Saves radial density/velocity/pressure/temperature profiles at several times
to data/channel_expansion_profiles.png, plus a CSV of the raw snapshot data.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt

from src.reactor import PlasmaReactor, ReactorParams
from src import postprocess
from src import radial_hydro as rh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MECHANISM = os.path.join(ROOT, "mechanisms", "gri30.yaml")
DATA_DIR = os.path.join(ROOT, "data")

SAVE_TIMES_S = np.array([1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 2e-5, 3.5e-5, 5e-5])


def run_0d_heating_profile(mechanism: str = MECHANISM, build_gas=None):
    """Run the 0D PlasmaReactor and return its q_eM(t) (and other
    diagnostics) as a DataFrame. `build_gas(mechanism) -> ct.Solution`
    defaults to the CH4/Air phi=0.5 case; pass a different one (e.g.
    matching run_ammonia.build_gas) to get another fuel's own profile --
    q_eM(t) (and hence the resulting eta(t)) is fuel-specific and must not
    be reused across fuels."""
    if build_gas is None:
        def build_gas(mech):
            gas = ct.Solution(mech)
            gas.set_equivalence_ratio(0.5, "CH4", "O2:1.0, N2:3.76")
            gas.TP = 300.0, ct.one_atm
            return gas
    gas = build_gas(mechanism)
    reactor = PlasmaReactor(gas, ReactorParams(mechanism=mechanism))
    result = reactor.integrate()
    return postprocess.to_dataframe(result)


def run_hydro(df_0d) -> rh.HydroResult:
    heating = rh.HeatingParams.from_0d_dataframe(df_0d)
    grid = rh.GridParams(n_cells=400, r_max=5.0e-3, geometry="cylindrical")
    ambient = rh.AmbientState(T0=300.0, p0=101325.0)

    r = grid.cell_centers()
    rho0 = np.full_like(r, ambient.rho0)
    u0 = np.zeros_like(r)
    p0 = np.full_like(r, ambient.p0)
    U0 = rh.primitive_to_conserved(rho0, u0, p0)

    return rh.run(grid, U0, SAVE_TIMES_S[-1], heating=heating, cfl=0.4, save_times=SAVE_TIMES_S)


def save_csv(result: rh.HydroResult) -> None:
    rows = []
    for t in result.t:
        rho, u, p = result.snapshots[t]
        T = p / (rho * rh.R_SPECIFIC)
        for ri, rhoi, ui, pi, Ti in zip(result.r, rho, u, p, T):
            rows.append({"t": t, "r": ri, "rho": rhoi, "u": ui, "p": pi, "T": Ti})
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "channel_expansion_profiles.csv"), index=False)


def plot_profiles(result: rh.HydroResult, r_channel: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    r_mm = result.r * 1e3
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(result.t)))

    for t, color in zip(result.t, colors):
        rho, u, p = result.snapshots[t]
        label = f"{t * 1e6:.2f} us"
        axes[0].plot(r_mm, rho, color=color, label=label)
        axes[1].plot(r_mm, u, color=color, label=label)
        axes[2].plot(r_mm, p / 1e5, color=color, label=label)

    for ax in axes:
        ax.axvline(r_channel * 1e3, color="k", linestyle=":", linewidth=1, label="r_channel" if ax is axes[0] else None)
        ax.set_xlabel("radius [mm]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("density [kg/m^3]")
    axes[0].set_title("Gas density")
    axes[1].set_ylabel("radial velocity [m/s]")
    axes[1].set_title("Radial velocity")
    axes[2].set_ylabel("pressure [bar]")
    axes[2].set_title("Pressure")
    axes[0].legend(fontsize=7, title="time")

    fig.suptitle("1D radial channel expansion (CH4/Air 0D model's q_eM(t) as heating source)")
    fig.tight_layout()
    os.makedirs(DATA_DIR, exist_ok=True)
    fig.savefig(os.path.join(DATA_DIR, "channel_expansion_profiles.png"), dpi=200)
    plt.close(fig)


def save_eta_trace(trace: rh.CoreDensityTrace) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame({"t": trace.t, "rho_core": trace.rho_core, "eta": trace.eta}).to_csv(
        os.path.join(DATA_DIR, "channel_expansion_eta.csv"), index=False
    )


def plot_eta_trace(trace: rh.CoreDensityTrace) -> None:
    f = rh.eta_interpolator(trace)
    t_dense = np.linspace(trace.t[0], trace.t[-1], 400)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_dense * 1e6, f(t_dense), "b-", linewidth=2, label=r"$\eta(t)$ (cubic-spline interpolant)")
    ax.plot(trace.t * 1e6, trace.eta, "o", color="tab:blue", markersize=5, label="hydro snapshots")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="unperturbed gas")
    ax.set_xlabel("time [us]")
    ax.set_ylabel(r"core density-reduction factor $\eta(t) = \bar{\rho}_{core}(t)/\rho_0$")
    ax.set_title(f"Core expansion trace (r_channel={trace.r_channel * 1e3:.3f} mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(DATA_DIR, exist_ok=True)
    fig.savefig(os.path.join(DATA_DIR, "channel_expansion_eta.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    df_0d = run_0d_heating_profile()
    result = run_hydro(df_0d)
    save_csv(result)

    ambient = rh.AmbientState(T0=300.0, p0=101325.0)
    heating_r_channel = rh.HeatingParams.from_0d_dataframe(df_0d).r_channel
    plot_profiles(result, heating_r_channel)

    trace = rh.core_density_trace(result, r_channel=heating_r_channel, rho_ambient=ambient.rho0)
    save_eta_trace(trace)
    plot_eta_trace(trace)

    last_t = result.t[-1]
    rho, u, p = result.snapshots[last_t]
    T = p / (rho * rh.R_SPECIFIC)
    print(f"At t={last_t * 1e6:.2f} us: center density {rho[0]:.4f} kg/m^3 "
          f"(ambient {ambient.rho0:.4f}), center T {T[0]:.1f} K, "
          f"peak outward velocity {u.max():.2f} m/s")
    print(f"eta(t) at t_end: {trace.eta[-1]:.4f} (core density down to "
          f"{trace.eta[-1] * 100:.1f}% of ambient)")
    print(f"Plots and CSVs written to {DATA_DIR}")


if __name__ == "__main__":
    main()
