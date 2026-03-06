from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ..config import load_portfolio_config
from ..data.history_builder import build_history_for_ticker
from ..data.refinitiv_client import RefinitivClient
from ..io.history_io import ric_to_filename, save_history_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and cache per-stock histories.")
    parser.add_argument("--config", default="config/portfolio.yaml", help="Path to portfolio config YAML")
    parser.add_argument(
        "--output-dir",
        default="data/raw/history",
        help="Directory to write history CSV files",
    )
    parser.add_argument("--start-date", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Override end date (YYYY-MM-DD)")
    return parser.parse_args()


def run_fetch_histories(
    *,
    config_path: str = "config/portfolio.yaml",
    output_dir: str = "data/raw/history",
    start_date_override: str | None = None,
    end_date_override: str | None = None,
) -> dict[str, Path]:
    cfg = load_portfolio_config(config_path)

    start_dt = start_date_override or cfg.start_date.isoformat()
    end_dt = end_date_override or (cfg.end_date.isoformat() if cfg.end_date else date.today().isoformat())

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    with RefinitivClient() as client:
        for asset in cfg.assets:
            ric = asset.ric
            df = build_history_for_ticker(client, ric, start_dt, end_dt)
            if df is None or df.empty:
                print(f"[WARN] No history for {ric}; skipped")
                continue

            path = output / ric_to_filename(ric)
            save_history_csv(df, path)
            written[ric] = path
            print(f"[OK] Wrote {ric} -> {path}")

    if not written:
        raise RuntimeError("No history files were written.")

    return written


def main() -> None:
    args = parse_args()
    run_fetch_histories(
        config_path=args.config,
        output_dir=args.output_dir,
        start_date_override=args.start_date,
        end_date_override=args.end_date,
    )


if __name__ == "__main__":
    main()
