"""Entry-point for the simplified Henry dataset generator (paper Sec. 5).

Usage:
    uv run python run_simplified_henry.py --help
    uv run python run_simplified_henry.py --outdir ./data/test --max-runs-per-scenario 2
"""
from simplified_henry.cli import main

if __name__ == "__main__":
    main()
