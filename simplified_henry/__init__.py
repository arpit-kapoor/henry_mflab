"""simplified_henry: simplified Henry-type problem (paper Sec. 5).

Solves the coupled elliptic groundwater flow + parabolic advection-diffusion system
with zero storage, zero influx and C = 0 on all boundaries, driven purely by
buoyancy via a linear equation of state ρ(C) = ρ₀(1 + β_C C).
"""
from .simulation import build_and_run_simplified_henry, create_random_field
from .generators import generate_simplified_henry_dataset

__all__ = [
    "build_and_run_simplified_henry",
    "create_random_field",
    "generate_simplified_henry_dataset",
]
