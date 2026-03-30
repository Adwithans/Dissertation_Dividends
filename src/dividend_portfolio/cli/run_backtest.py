from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-stock dividend portfolio backtest.")
    parser.add_argument("--config", default="config/portfolio.yaml", help="Path to portfolio config")
    parser.add_argument(
        "--history-dir",
        default="data/raw/history",
        help="Directory containing cached history CSV files",
    )
    parser.add_argument(
        "--fallback-history-dir",
        default="stock_data",
        help="Fallback directory for history CSV files",
    )
    parser.add_argument("--run-base-dir", default="data/runs", help="Base output directory for run artifacts")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable PNG chart outputs",
    )
    return parser.parse_args()


def run_backtest(
    *,
    config_path: str = "config/portfolio.yaml",
    history_dir: str = "data/raw/history",
    fallback_history_dir: str | None = "stock_data",
    run_base_dir: str = "data/runs",
    generate_plots: bool = True,
) -> Path:
    raise RuntimeError(
        "The static backtest workflow has been retired. "
        "Use dynamic strategy runner instead: "
        "`python -m src.dividend_portfolio.cli.run_dynamic_strategy --config config/portfolio.yaml`."
    )


def main() -> None:
    args = parse_args()
    run_backtest(
        config_path=args.config,
        history_dir=args.history_dir,
        fallback_history_dir=args.fallback_history_dir,
        run_base_dir=args.run_base_dir,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
