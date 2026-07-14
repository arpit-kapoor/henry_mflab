"""simple_henry: simplified density-driven convection (Henry/Elder-type) problem.

Solves the coupled elliptic groundwater flow + parabolic advection-diffusion system
with zero storage, zero influx, and homogeneous Dirichlet boundary conditions on all
sides, driven purely by buoyancy via a linear equation of state ρ(C) = ρ₀(1 + β_C C).
"""
from .simulation import build_and_run_simple_henry
from .generators import generate_simple_henry_dataset

__all__ = ["build_and_run_simple_henry", "generate_simple_henry_dataset"]
