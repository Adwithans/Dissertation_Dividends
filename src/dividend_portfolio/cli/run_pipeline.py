from __future__ import annotations

import argparse



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
    _ = parse_args()
    raise RuntimeError(
        "Static pipeline has been retired. "
        "Run dynamic strategy directly: "
        "`python -m src.dividend_portfolio.cli.run_dynamic_strategy --config config/portfolio.yaml`."
    )


if __name__ == "__main__":
    main()
