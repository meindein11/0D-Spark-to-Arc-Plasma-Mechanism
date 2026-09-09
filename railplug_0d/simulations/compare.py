"""Side-by-side benchmark: ultra-lean CH4/Air (phi=0.5) vs pure NH3/Air
(phi=0.7) spark-to-arc discharge and ignition.

Runs both cases, writes per-case CSVs (via run_methane.py / run_ammonia.py),
and produces comparison plots plus a markdown summary table under data/.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import postprocess
import run_methane
import run_ammonia

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

FRAGMENT_SPECIES = {
    "CH4/Air (phi=0.5)": "CH3",
    "NH3/Air (phi=0.7)": "NH2",
}


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    df_ch4 = run_methane.run()
    df_nh3 = run_ammonia.run()
    cases = {
        "CH4/Air (phi=0.5)": df_ch4,
        "NH3/Air (phi=0.7)": df_nh3,
    }

    postprocess.plot_temperatures(cases, os.path.join(DATA_DIR, "compare_temperatures.png"))
    postprocess.plot_resistivity(cases, os.path.join(DATA_DIR, "compare_resistivity.png"))
    postprocess.plot_radicals(cases, FRAGMENT_SPECIES, os.path.join(DATA_DIR, "compare_radicals.png"))

    table = postprocess.summary_table(cases, FRAGMENT_SPECIES)
    summary_path = os.path.join(DATA_DIR, "summary_table.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# CH4 vs NH3 spark-to-arc ignition comparison\n\n")
        f.write(table)

    print(table)
    print(f"Plots and summary written to {DATA_DIR}")


if __name__ == "__main__":
    main()
