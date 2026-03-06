from __future__ import annotations

import argparse
from pathlib import Path

from .fetch_histories import run_fetch_histories
from .run_backtest import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch histories and run backtest in one command.")
    parser.add_argument("--config", default="config/portfolio.yaml", help="Path to portfolio config")
    parser.add_argument(
        "--history-dir",
        default="data/raw/history",
        help="Directory to write/read history CSVs",
    )
    parser.add_argument("--run-base-dir", default="data/runs", help="Base output directory for run artifacts")
    parser.add_argument("--start-date", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--no-plots", action="store_true", help="Disable PNG chart outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_fetch_histories(
        config_path=args.config,
        output_dir=args.history_dir,
        start_date_override=args.start_date,
        end_date_override=args.end_date,
    )

    run_dir = run_backtest(
        config_path=args.config,
        history_dir=args.history_dir,
        fallback_history_dir=None,
        run_base_dir=args.run_base_dir,
        generate_plots=not args.no_plots,
    )

    print(f"[OK] Pipeline complete. Outputs at: {Path(run_dir)}")


if __name__ == "__main__":
    main()
