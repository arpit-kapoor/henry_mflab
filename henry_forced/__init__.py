"""henry_forced: Henry saltwater intrusion with classic Henry BCs and constant inflow forcing.

Intermediate between ``simple_henry`` (closed box, buoyancy only) and
``henry_data`` (tides, storms, storage). Runs differ by a random initial
concentration field and a constant left-boundary inflow; the dataset maps
(C₀, inflow, beta_c, diffc) to the full concentration/head trajectory.
"""
from .simulation import build_and_run_henry_forced
from .generators import generate_henry_forced_dataset

__all__ = [
    "build_and_run_henry_forced",
    "generate_henry_forced_dataset",
]
