"""Post-processing, diagnostics, and comparison plots for PlasmaReactor runs.

Given one or more `ReactorResult` objects, this module recomputes derived
quantities (plasma resistivity, reduced electric field, mole fractions) along
the saved trajectory, detects ignition delay, and produces the comparison
plots / markdown summary table used by `simulations/compare.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
from scipy import constants as sc

from . import physics
from .reactor import ReactorResult
from .circuit import current_density

TOWNSEND = 1e-21  # 1 Td = 1e-21 V*m^2


def to_dataframe(result: ReactorResult) -> pd.DataFrame:
    """Recompute rho_res(t), E/N(t), and species mole fractions along the
    saved [Tg, Te, ne, Y] trajectory, returned as a tidy DataFrame."""
    gas = ct.Solution(result.mechanism)
    n_t = len(result.t)
    n_sp = len(result.species_names)

    rho_res = np.empty(n_t)
    e_over_n_td = np.empty(n_t)
    X = np.empty((n_t, n_sp))

    for i in range(n_t):
        gas.TPY = result.Tg[i], result.P0, result.Y[i]
        rho_res[i] = physics.plasma_resistivity(gas, result.Te[i], result.ne[i], result.ne_floor)
        j = current_density(result.t[i], result.circuit)
        E = rho_res[i] * j
        n_tot = gas.density / gas.mean_molecular_weight * physics.AVOGADRO_PER_KMOL
        e_over_n_td[i] = (E / n_tot) / TOWNSEND
        X[i, :] = gas.X

    df = pd.DataFrame({
        "t": result.t,
        "Tg": result.Tg,
        "Te": result.Te,
        "ne": result.ne,
        "rho_res": rho_res,
        "E_over_N_Td": e_over_n_td,
    })
    for k, name in enumerate(result.species_names):
        df[f"X_{name}"] = X[:, k]
    return df


def ignition_delay(df: pd.DataFrame, T_threshold: float = 1500.0) -> float:
    """Ignition delay tau_ign [s]: first time Tg crosses `T_threshold`
    (linearly interpolated between saved samples), or, if Tg never reaches
    the threshold, the time of peak dTg/dt."""
    t = df["t"].to_numpy()
    Tg = df["Tg"].to_numpy()
    above = np.flatnonzero(Tg >= T_threshold)
    if above.size > 0:
        i = above[0]
        if i == 0:
            return float(t[0])
        frac = (T_threshold - Tg[i - 1]) / (Tg[i] - Tg[i - 1])
        return float(t[i - 1] + frac * (t[i] - t[i - 1]))
    dTdt = np.gradient(Tg, t)
    return float(t[np.argmax(dTdt)])


def radical_persistence(df: pd.DataFrame, species: str, frac: float = 0.1) -> float:
    """Duration [s] that species X_`species` stays at or above `frac` times
    its own peak mole fraction -- a simple measure of radical-pool
    persistence."""
    col = f"X_{species}"
    if col not in df.columns:
        return 0.0
    t = df["t"].to_numpy()
    X = df[col].to_numpy()
    peak = X.max()
    if peak <= 0.0:
        return 0.0
    above = np.flatnonzero(X >= frac * peak)
    if above.size == 0:
        return 0.0
    return float(t[above[-1]] - t[above[0]])


def summary_table(
    cases: dict[str, pd.DataFrame],
    fragment_species: dict[str, str],
    T_threshold: float = 1500.0,
) -> str:
    """Markdown table contrasting ignition delay, peak Tg, minimum rho_res,
    and radical persistence (OH and the fuel-specific fragment) across
    `cases`."""
    header = (
        "| Case | tau_ign [us] | Peak Tg [K] | Min rho_res [Ohm*m] "
        "| OH persistence [us] | Fragment persistence [us] |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, df in cases.items():
        tau = ignition_delay(df, T_threshold) * 1e6
        peak_tg = df["Tg"].max()
        rho_min = df["rho_res"][df["rho_res"] > 0].min()
        oh_persist = radical_persistence(df, "OH") * 1e6
        frag = fragment_species.get(name)
        frag_persist = radical_persistence(df, frag) * 1e6 if frag else float("nan")
        rows.append(
            f"| {name} | {tau:.3f} | {peak_tg:.1f} | {rho_min:.3e} "
            f"| {oh_persist:.3f} | {frag_persist:.3f} ({frag}) |"
        )
    return header + "\n".join(rows) + "\n"


def _t_us(df: pd.DataFrame) -> np.ndarray:
    return df["t"].to_numpy() * 1e6


def plot_temperatures(cases: dict[str, pd.DataFrame], outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (name, df), color in zip(cases.items(), colors):
        ax.plot(_t_us(df), df["Tg"], color=color, linestyle="-", label=f"{name} Tg")
        ax.plot(_t_us(df), df["Te"], color=color, linestyle="--", label=f"{name} Te")
    ax.set_xlabel("time [us]")
    ax.set_ylabel("temperature [K]")
    ax.set_yscale("log")
    ax.set_title("Gas and electron temperature trajectories")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_resistivity(cases: dict[str, pd.DataFrame], outpath: str, t_max_us: float = 50.0) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, df in cases.items():
        mask = _t_us(df) <= t_max_us
        ax.plot(_t_us(df)[mask], df["rho_res"].to_numpy()[mask], label=name)
    ax.set_xlabel("time [us]")
    ax.set_ylabel("effective plasma resistivity rho_res [Ohm*m]")
    ax.set_yscale("log")
    ax.set_title("Effective plasma resistivity")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_radicals(
    cases: dict[str, pd.DataFrame],
    fragment_species: dict[str, str],
    outpath: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False)

    # Mole fractions fall to sub-1e-30 floating-point noise at very early
    # times (before any real chemistry has occurred); a bare log scale would
    # let that noise floor set the axis range and squash the physically
    # meaningful part of the curves, so we clip the visible range instead.
    floor = 1e-10

    ax = axes[0]
    for name, df in cases.items():
        for species, style in (("OH", "-"), ("H", "--"), ("O", ":")):
            col = f"X_{species}"
            if col in df.columns:
                ax.plot(_t_us(df), df[col].clip(lower=floor), linestyle=style, label=f"{name} {species}")
    ax.set_xlabel("time [us]")
    ax.set_ylabel("mole fraction")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.set_title("Common radicals: OH, H, O")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    for name, df in cases.items():
        frag = fragment_species.get(name)
        col = f"X_{frag}" if frag else None
        if col and col in df.columns:
            ax.plot(_t_us(df), df[col].clip(lower=floor), label=f"{name} {frag}")
    ax.set_xlabel("time [us]")
    ax.set_ylabel("mole fraction")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.set_title("Fuel-specific fragments")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
