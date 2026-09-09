"""Ultra-lean CH4/Air (phi=0.5) spark-to-arc ignition case."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cantera as ct
import pandas as pd

from src.circuit import CircuitParams
from src.reactor import PlasmaReactor, ReactorParams
from src import postprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MECHANISM = os.path.join(ROOT, "mechanisms", "gri30.yaml")
DATA_DIR = os.path.join(ROOT, "data")

PHI = 0.5
T0 = 300.0
P0 = ct.one_atm
RADICALS = ["OH", "H", "O", "CH3"]


def build_gas(phi: float = PHI, T0: float = T0, P0: float = P0) -> ct.Solution:
    gas = ct.Solution(MECHANISM)
    gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")
    gas.TP = T0, P0
    return gas


def run(circuit: CircuitParams | None = None) -> pd.DataFrame:
    gas = build_gas()
    params = ReactorParams(
        mechanism=MECHANISM, T0=T0, P0=P0,
        circuit=circuit if circuit is not None else CircuitParams(),
    )
    reactor = PlasmaReactor(gas, params)
    result = reactor.integrate()
    df = postprocess.to_dataframe(result)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "methane_phi0.5.csv"), index=False)
    return df


if __name__ == "__main__":
    df = run()
    tau_ign = postprocess.ignition_delay(df)
    print(f"CH4/Air phi={PHI}: tau_ign = {tau_ign * 1e6:.3f} us, peak Tg = {df['Tg'].max():.1f} K")
