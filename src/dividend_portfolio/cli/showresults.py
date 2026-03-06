from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_portfolio_config
from ..models import StrategyConfig
from ..reporting.dynamic_results import generate_dynamic_showresults


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_portfolio_config(args.config)
    strategy = cfg.strategy or StrategyConfig(mode="dynamic_100_25")
    db_path = args.db_path or strategy.sqlite_path

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
    print(f"[INFO] Summary JSON: {Path(out_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()

