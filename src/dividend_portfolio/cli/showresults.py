from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_portfolio_config
from ..models import StrategyConfig
from ..reporting.dynamic_results import generate_dynamic_showresults
from .run_dynamic_strategy import _normalize_experiment_group, _upsert_experiment_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate graphs and summary JSON for a dynamic strategy run from SQLite."
    )
    parser.add_argument("--config", default="config/portfolio.yaml", help="Path to portfolio config")
    parser.add_argument("--db-path", default=None, help="SQLite path override")
    parser.add_argument("--run-id", default=None, help="Run id to analyze; default is latest run in DB")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots + summary JSON (default: data/runs_dynamic/<run_id>/showresults)",
    )
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmark fetch/comparison")
    parser.add_argument("--benchmark-ric", default=".SPX", help="Benchmark RIC for comparison")
    parser.add_argument(
        "--benchmark-label",
        default="S&P 500 (.SPX)",
        help="Display label used in chart/summary benchmark section",
    )
    parser.add_argument(
        "--experiment-group",
        default=None,
        help="Optional experiment group name for writing cross-run comparison tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_portfolio_config(args.config)
    strategy = cfg.strategy or StrategyConfig(mode="dynamic_100_25")
    db_path = args.db_path or strategy.sqlite_path
    experiment_group = _normalize_experiment_group(args.experiment_group) or _normalize_experiment_group(
        strategy.experiment_group
    )

    run_id, out_dir = generate_dynamic_showresults(
        db_path=db_path,
        run_id=args.run_id,
        output_dir=args.output_dir,
        include_benchmark=not args.no_benchmark,
        benchmark_ric=args.benchmark_ric,
        benchmark_label=args.benchmark_label,
    )

    print(f"[OK] showresults generated for run_id={run_id}")
    print(f"[INFO] Output directory: {Path(out_dir)}")
    summary_path = Path(out_dir) / "summary.json"
    print(f"[INFO] Summary JSON: {summary_path}")
    if experiment_group:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        csv_path, json_path = _upsert_experiment_comparison(
            summary=summary,
            experiment_group=experiment_group,
        )
        print(f"[OK] Experiment comparison updated: {csv_path}")
        print(f"[INFO] Experiment comparison JSON: {json_path}")


if __name__ == "__main__":
    main()
