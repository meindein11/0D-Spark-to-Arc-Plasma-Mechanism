"""Diagnostic: trace CH4's energy balance term-by-term at phi=0.9 (a
"spike" point in the phi sweep, peak_Tg ~4573-4664 K) vs phi=1.0 (a "dip"
point, peak_Tg ~2492-2504 K), to find exactly where and why their
trajectories diverge.

Context: `run_parametric_phi_sweep.py`'s CH4 results show a non-monotonic,
physically implausible zigzag in peak core temperature for phi>=0.9 --
phi=1.0 (exactly stoichiometric, where peak temperature should typically
be *highest*, not a dip) lands far below its phi=0.9 and phi=1.1
neighbors. Four separate fix attempts targeting different energy-balance
terms (an external high-T sink, gating the chemistry term's exothermic
part, and fixing a "cv thermal trapdoor" in the heat-capacity blending)
each left this pattern essentially unchanged -- see MODELING_NOTES.md's
phi-sweep follow-up for all four sweeps' numbers. Rather than guess at a
fifth term, this script traces the *actual* term-by-term energy balance
solve_ivp produced for both cases, post-hoc, using the same `rhs()` this
model actually integrates (via its `return_terms=True` diagnostics hook --
see `ExpansionCoupledReactor.rhs`) so there is no risk of the diagnostic
duplicating -- and silently drifting out of sync with -- the real physics.

Writes data/ch4_bifurcation_trace.png (Tg vs t, and the four
heating/cooling rate terms vs t, both phi, 0-15 us) and prints the first
timestep where the two trajectories' Tg differ by more than 50 K.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.expansion_coupled_reactor import ExpansionCoupledReactor, ExpansionReactorParams
import run_parametric_phi_sweep as sweep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

PHI_CASES = [0.9, 1.0]
# The task brief assumed the phi=0.9/1.0 trajectories diverge within the
# first 15 us. Checked directly and that assumption doesn't hold: both
# stay below ~750 K through 15 us (ignition/early heating only); the real
# separation starts around t~17 us and grows through the rest of the run,
# with each case's peak Tg landing at t=50 us (t_end) itself, not an
# interior peak. T_WINDOW_S is widened to 50 us (the full run) accordingly
# -- narrowing back to 15 us here would have silently hidden the actual
# bifurcation entirely.
T_WINDOW_S = 50.0e-6
MECHANISM = sweep.FUELS["CH4"]["mechanism"]
BUILD_GAS = sweep.FUELS["CH4"]["build_gas"]


def run_and_trace(phi: float) -> tuple[pd.DataFrame, "ExpansionCoupledReactor"]:
    """Run CH4 at this phi through the full stabilized pipeline (same as
    run_parametric_phi_sweep.py: fuel/phi-specific eta(t) trace -> uncapped
    ExpansionCoupledReactor), then reconstruct rhs()'s exact term-by-term
    output at every one of solve_ivp's own saved timesteps within the
    first T_WINDOW_S seconds."""
    trace = sweep.build_eta_trace(MECHANISM, lambda: BUILD_GAS(phi), sweep.CIRCUIT)

    gas = BUILD_GAS(phi)
    params = ExpansionReactorParams(mechanism=MECHANISM, trace=trace, T0=sweep.T0, circuit=sweep.CIRCUIT)
    reactor = ExpansionCoupledReactor(gas, params)
    result = reactor.integrate()

    mask = result.t <= T_WINDOW_S
    t_win = result.t[mask]
    Tg_win = result.Tg[mask]
    Te_win = result.Te[mask]
    ne_win = result.ne[mask]
    Y_win = result.Y[mask]

    rows = []
    for i in range(len(t_win)):
        y_i = np.concatenate(([Tg_win[i], Te_win[i], ne_win[i]], Y_win[i]))
        terms = reactor.rhs(float(t_win[i]), y_i, return_terms=True)
        if terms is None:
            continue
        rows.append({"t": float(t_win[i]), **terms})

    df = pd.DataFrame(rows)
    df["reached_t_end"] = result.reached_t_end
    df["peak_Tg_full_run"] = float(result.Tg.max())
    return df, reactor


def find_divergence(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str,
                     threshold_K: float = 50.0) -> None:
    """Interpolate both traces onto a common fine time grid and report the
    first time their Tg differs by more than `threshold_K` (the task's
    literal ask), plus a few larger thresholds for context: the first
    50 K crossing turns out to be a trivial, pre-ignition difference (both
    trajectories still under ~750 K -- see console output), not the real
    bifurcation, so reporting only it would be misleading on its own."""
    t_common = np.linspace(0.0, T_WINDOW_S, 3000)
    Tg_a = np.interp(t_common, df_a["t"], df_a["Tg"])
    Tg_b = np.interp(t_common, df_b["t"], df_b["Tg"])
    diff = np.abs(Tg_a - Tg_b)

    print(f"\n=== Divergence check ({label_a} vs {label_b}) ===")
    last_found_t = None
    for thresh in (threshold_K, 200.0, 1000.0, 2000.0):
        above = np.flatnonzero(diff > thresh)
        if above.size == 0:
            print(f"    Never differ by more than {thresh:.0f} K within "
                  f"[0, {T_WINDOW_S * 1e6:.1f}] us (max diff = {diff.max():.2f} K).")
            continue
        i = above[0]
        t_div = t_common[i]
        last_found_t = t_div
        tag = " <- literal task threshold" if thresh == threshold_K else ""
        print(f"    >{thresh:.0f} K at t = {t_div * 1e6:8.4f} us: "
              f"{label_a}={Tg_a[i]:8.2f} K   {label_b}={Tg_b[i]:8.2f} K   "
              f"|delta|={diff[i]:7.2f} K{tag}")

    if last_found_t is None:
        return

    # Report the nearest actual (non-interpolated) rows from each trace,
    # at the *largest* threshold actually reached (the real bifurcation,
    # not the trivial pre-ignition 50 K difference), for a real,
    # physically-evaluated term-by-term snapshot at that moment.
    print(f"\n    Term-by-term snapshot at the largest divergence point reached "
          f"(t~{last_found_t * 1e6:.4f} us):")
    for label, df in [(label_a, df_a), (label_b, df_b)]:
        j = int(np.argmin(np.abs(df["t"].to_numpy() - last_found_t)))
        row = df.iloc[j]
        print(f"      [{label} nearest saved step, t={row['t'] * 1e6:.4f} us]  "
              f"Tg={row['Tg']:.2f} K  q_eM={row['q_eM']:.3e}  q_chem={row['q_chem']:.3e}  "
              f"q_rad={row['q_rad']:.3e}  pdV_cooling={row['pdV_cooling_Wm3']:.3e}  "
              f"cv_eff={row['cv_effective']:.2f}  cap_factor={row['cap_factor']:.4f}  "
              f"dTg_dt={row['dTg_dt']:.3e}")


def plot_traces(traces: dict[float, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    colors = {0.9: "tab:red", 1.0: "tab:blue"}
    term_styles = {"q_eM": "-", "q_chem": "--", "q_rad": ":", "pdV_cooling_Wm3": "-."}

    for phi, df in traces.items():
        t_us = df["t"] * 1e6
        axes[0].plot(t_us, df["Tg"], color=colors[phi], linewidth=2, label=f"phi={phi:.1f}")

    axes[0].set_ylabel(r"$T_g$ [K]")
    axes[0].set_title(f"CH4 core gas temperature, 0-{T_WINDOW_S * 1e6:.0f} us (phi=0.9 vs phi=1.0)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for phi, df in traces.items():
        t_us = df["t"] * 1e6
        for term, style in term_styles.items():
            label = f"phi={phi:.1f}: {term}"
            axes[1].plot(t_us, df[term], color=colors[phi], linestyle=style, linewidth=1.3, label=label)

    axes[1].axhline(0.0, color="k", linewidth=0.7, alpha=0.5)
    axes[1].set_yscale("symlog", linthresh=1e8)
    axes[1].set_ylabel(r"rate [W/m$^3$] (symlog)")
    axes[1].set_xlabel(r"time [us]")
    axes[1].set_title("Dominant heating/cooling terms: $q_{eM}$, $q_{chem}$, $q_{rad}$, PdV cooling")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    os.makedirs(DATA_DIR, exist_ok=True)
    outpath = os.path.join(DATA_DIR, "ch4_bifurcation_trace.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"\nPlot written to {outpath}")


def main() -> None:
    traces = {}
    for phi in PHI_CASES:
        print(f"--- Running and tracing CH4 phi={phi:.1f} ---", flush=True)
        df, reactor = run_and_trace(phi)
        traces[phi] = df
        print(f"    {len(df)} traced points in [0, {T_WINDOW_S * 1e6:.1f}] us; "
              f"full-run peak Tg = {df['peak_Tg_full_run'].iloc[0]:.1f} K "
              f"(reached_t_end={bool(df['reached_t_end'].iloc[0])})", flush=True)

    find_divergence(traces[0.9], traces[1.0], "phi=0.9", "phi=1.0")
    plot_traces(traces)


if __name__ == "__main__":
    main()
