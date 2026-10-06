"""classical_henry: classical Henry-type saltwater intrusion (paper App. I).

Classical Henry boundary conditions (constant freshwater inflow on the left,
sea-level GHB with seawater inflow on the right, no-flow top/bottom). Runs
differ by a random initial concentration field and a constant inflow Q; the
dataset maps (C₀, Q, beta_c, diffc, coordinates) to the full concentration and
head trajectory.
"""
from .simulation import build_and_run_classical_henry
from .generators import generate_classical_henry_dataset

__all__ = [
    "build_and_run_classical_henry",
    "generate_classical_henry_dataset",
]
