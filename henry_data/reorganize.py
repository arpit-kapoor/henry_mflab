import argparse
import json
import shutil
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _move_runs_numeric(source_dir: Path, target_dir: Path, scenario_data: dict, overwrite: bool) -> list[dict]:
    runs = scenario_data.get("runs", [])
    run_records = []

    for idx, run in enumerate(runs, start=1):
        old_name = run["run"]
        old_path = source_dir / old_name
        new_name = f"run_{idx:06d}"
        new_path = target_dir / new_name

        if new_path.exists():
            if overwrite:
                if new_path.is_dir():
                    shutil.rmtree(new_path)
                else:
                    new_path.unlink()
            else:
                raise FileExistsError(f"Target run path exists: {new_path}")

        if old_path.exists() and run.get("status") in {"ok", "skipped"}:
            shutil.move(str(old_path), str(new_path))

        run_meta = dict(run)
        run_meta["source_run_name"] = old_name
        run_meta["run"] = new_name
        run_meta["workspace"] = str(new_path)
        run_records.append(run_meta)

    return run_records


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def reorganize_coupling_diffusion_outputs(
    raw_outdir: Path,
    outdir: Path,
    beta_count: int,
    diffc_count: int,
    lag: int,
    overwrite: bool,
) -> dict:
    scenarios_root = outdir / "scenarios"
    scenarios_root.mkdir(parents=True, exist_ok=True)

    scenarios_manifest = {
        "layout": "coupling_diffusion_grid",
        "lag": int(lag),
        "scenarios": [],
    }

    raw_manifest = _load_json(raw_outdir / "manifest.json")
    scenario_entries = raw_manifest.get("scenarios", [])
    expected = int(beta_count) * int(diffc_count)
    if len(scenario_entries) != expected:
        raise ValueError(
            f"Expected {expected} scenarios in raw manifest, found {len(scenario_entries)}"
        )

    for idx, scen_summary in enumerate(scenario_entries, start=1):
        raw_name = scen_summary["scenario"]
        raw_dir = raw_outdir / raw_name
        if not raw_dir.exists():
            raise FileNotFoundError(f"Expected raw scenario directory not found: {raw_dir}")

        target_dir = scenarios_root / f"scenario_{idx:02d}"
        if target_dir.exists() and overwrite:
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        scen = _load_json(raw_dir / "scenario_manifest.json")
        runs_cfg = _move_runs_numeric(raw_dir, target_dir, scen, overwrite=overwrite)

        scenario_config = {
            "scenario_index": idx,
            "beta_c": float(scen["beta_c"]),
            "diffc": float(scen["diffc"]),
            "lag": int(lag),
            "n_total_runs": scen.get("n_total_runs", len(runs_cfg)),
            "n_ok_runs": scen.get("n_ok_runs", 0),
            "n_skipped_runs": scen.get("n_skipped_runs", 0),
            "n_failed_runs": scen.get("n_failed_runs", 0),
        }

        _write_json(target_dir / "scenario_config.json", scenario_config)
        _write_json(
            target_dir / "runs_config.json",
            {"runs": runs_cfg, "failures": scen.get("failures", [])},
        )

        scenarios_manifest["scenarios"].append(
            {
                "scenario_dir": str(target_dir),
                **scenario_config,
            }
        )

    _write_json(outdir / "scenarios_manifest.json", scenarios_manifest)
    return scenarios_manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reorganize raw coupling/diffusion grid outputs into clean scenario/run naming."
    )
    ap.add_argument("--raw-outdir", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--beta-count", type=int, required=True)
    ap.add_argument("--diffc-count", type=int, required=True)
    ap.add_argument("--lag", type=int, required=True)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()
    manifest = reorganize_coupling_diffusion_outputs(
        raw_outdir=Path(args.raw_outdir),
        outdir=Path(args.outdir),
        beta_count=args.beta_count,
        diffc_count=args.diffc_count,
        lag=args.lag,
        overwrite=bool(args.overwrite),
    )
    print(f"Reorganized scenarios in: {args.outdir}")
    print(f"  Grid scenarios:      {len(manifest['scenarios'])}")


if __name__ == "__main__":
    main()
