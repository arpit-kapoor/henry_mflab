"""Entry-point for the forced Henry dataset generator.

Usage:
    uv run python run_henry_forced.py --help
    uv run python run_henry_forced.py --outdir ./forced_out --ncol 40 --nlay 20 --nstp 500 --skip 10
"""
from henry_forced.cli import main

if __name__ == "__main__":
    main()
