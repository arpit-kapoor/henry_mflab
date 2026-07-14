"""Entry-point for the simplified Henry dataset generator.

Usage:
    uv run python run_simple_henry.py --help
    uv run python run_simple_henry.py --outdir ./simple_out --ncol 40 --nlay 20 --nstp 100
"""
from simple_henry.cli import main

if __name__ == "__main__":
    main()
