"""Entry-point for the classical Henry dataset generator (paper App. I).

Usage:
    uv run python run_classical_henry.py --help
    uv run python run_classical_henry.py --outdir ./data/test --max-runs-per-scenario 2
"""
from classical_henry.cli import main

if __name__ == "__main__":
    main()
