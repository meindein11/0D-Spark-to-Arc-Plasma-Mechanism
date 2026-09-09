"""Spark-to-arc discharge current waveform for the 0D plasma reactor.

Two-stage profile:
  * Spark stage  (0 <= t <= t_spark):   low, constant breakdown current.
  * Arc stage    (t > t_spark):         exponentially decaying follow-on arc,
                                         I(t) = I_peak * exp(-(t-t_spark)/tau_arc).

The handoff at t_spark is a genuine step change (a real spark-to-arc
transition, not a modeling artifact to be smoothed away): reactor.py
integrates the spark and arc stages as two separate `solve_ivp` calls split
exactly at t_spark, so I(t) only ever needs to be smooth *within* each
stage, not across the boundary between them. An earlier version smoothed
the handoff with a logistic blend, but since `i_arc` is defined as
`i_peak * exp(-max(t-t_spark, 0)/tau_arc)` it is pinned at the full
`i_peak` for all t <= t_spark (not a value that itself ramps up from
~0) -- so blending towards it starts leaking near-kA current into the
supposedly ~50 A spark stage many transition-widths before t_spark. Since
Joule heating scales as I(t)^2, that leak overstated spark-stage electron
heating by two orders of magnitude. A clean, unsmoothed step is both
simpler and correct here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitParams:
    t_spark: float = 100e-9        # spark-stage duration [s]
    i_spark: float = 50.0          # spark-stage current [A]
    i_peak: float = 2500.0         # peak follow-on arc current [A]
    tau_arc: float = 15e-6         # arc current decay time constant [s]
    area_arc: float = 1.0e-6       # arc cross-sectional area [m^2] (1.0 mm^2)
    t_end: float = 50e-6           # total simulated duration [s]


def current(t: float, params: CircuitParams = CircuitParams()) -> float:
    """Instantaneous discharge current I(t) [A]."""
    if t <= params.t_spark:
        return params.i_spark
    return params.i_peak * math.exp(-(t - params.t_spark) / params.tau_arc)


def current_density(t: float, params: CircuitParams = CircuitParams()) -> float:
    """Current density j(t) = I(t) / A_arc [A/m^2]."""
    return current(t, params) / params.area_arc
