from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from ..config import load_portfolio_config
from ..data.refinitiv_client import RefinitivClient
from ..models import StrategyConfig
from ..reporting.dynamic_results import generate_dynamic_showresults
from ..strategy.engine import run_dynamic_rotation
from ..strategy.provider import RefinitivStrategyDataProvider, utc_now_id
from ..strategy.storage import StrategyStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamic 100->25 S&P dividend rotation strategy and store outputs in SQLite."
    )
    parser.add_argument("--config", default="config/portfolio.yaml", help="Path to portfolio config")
    parser.add_argument("--start-date", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Override sqlite output path (default from strategy.sqlite_path)",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id. Default is UTC timestamp.")
    parser.add_argument(
        "--reuse-run-id",
        default=None,
        help="Reuse an existing run id from SQLite and skip refetch/recompute.",
    )
    parser.add_argument(
        "--stable-run-id",
        action="store_true",
        help="Generate deterministic run id from config+date range.",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip refetch/recompute if run id already exists in SQLite.",
    )
    parser.add_argument(
        "--csv-output-dir",
        default=None,
        help="Optional output directory for CSV exports from SQLite",
    )
    parser.add_argument(
        "--no-csv-export",
        action="store_true",
        help="Disable CSV export even if strategy.csv_export_enabled is true.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Refinitiv request batch size for multi-RIC calls (higher can reduce API round-trips).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable in-memory API response caching for this run.",
    )
    parser.add_argument(
        "--no-showresults",
        action="store_true",
        help="Disable automatic post-run showresults generation.",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Disable S&P benchmark fetch in showresults generation.",
    )
    return parser.parse_args()


def _stable_run_id(config_obj, start_date: str, end_date: str) -> str:
    payload = {
        "start_date": start_date,
        "end_date": end_date,
        "config": asdict(config_obj),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    start_s = start_date.replace("-", "")
    end_s = end_date.replace("-", "")
    return f"dyn_{start_s}_{end_s}_{digest}"


def _export_if_needed(store: StrategyStore, run_id: str, csv_dir: str | None) -> None:
    if csv_dir:
        exported = store.export_run_csv(run_id, csv_dir)
        print(f"[OK] Exported CSV outputs to: {exported}")


def _showresults_if_enabled(
    *,
    enabled: bool,
    db_path: str,
    run_id: str,
    include_benchmark: bool,
) -> None:
    if not enabled:
        return
    out_dir = Path("data/runs_dynamic") / run_id / "showresults"
    try:
        _, generated_dir = generate_dynamic_showresults(
            db_path=db_path,
            run_id=run_id,
            output_dir=out_dir,
            include_benchmark=include_benchmark,
            benchmark_ric=".SPX",
            benchmark_label="S&P 500 (.SPX)",
        )
        print(f"[OK] showresults generated: {generated_dir}")
        print(f"[INFO] showresults summary: {Path(generated_dir) / 'summary.json'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] showresults generation failed for run_id={run_id}: {exc}")


def main() -> None:
    args = parse_args()
    cfg = load_portfolio_config(args.config)
    strategy = cfg.strategy or StrategyConfig(mode="dynamic_100_25")
    if strategy.mode not in {"dynamic_100_25", "static"}:
        raise ValueError(f"Unsupported strategy.mode for dynamic run: {strategy.mode}")
    if args.reuse_run_id and (args.run_id or args.stable_run_id):
        raise ValueError("--reuse-run-id cannot be combined with --run-id or --stable-run-id")

    start_date = args.start_date or cfg.start_date.isoformat()
    end_date = args.end_date or (cfg.end_date.isoformat() if cfg.end_date else date.today().isoformat())
    db_path = args.db_path or strategy.sqlite_path
    csv_dir = args.csv_output_dir
    if args.no_csv_export:
        csv_dir = None
    elif csv_dir is None and strategy.csv_export_enabled:
        csv_dir = str(Path("data/runs_dynamic") / (args.reuse_run_id or "latest"))

    store = StrategyStore(db_path)

    if args.reuse_run_id:
        run_id = args.reuse_run_id
        if not store.run_exists(run_id):
            store.close()
            raise ValueError(f"Run id '{run_id}' does not exist in {Path(db_path)}")
        if csv_dir and csv_dir.endswith("/latest"):
            csv_dir = str(Path("data/runs_dynamic") / run_id)
        _export_if_needed(store, run_id, csv_dir)
        print(f"[OK] Reused existing run: run_id={run_id}")
        print(f"[INFO] SQLite DB: {Path(db_path)}")
        print(f"[INFO] Portfolio daily rows: {store.row_count('portfolio_daily', run_id)}")
        print(f"[INFO] Holdings rows: {store.row_count('holdings_daily', run_id)}")
        print(f"[INFO] Trades rows: {store.row_count('trades', run_id)}")
        store.close()
        _showresults_if_enabled(
            enabled=not args.no_showresults,
            db_path=db_path,
            run_id=run_id,
            include_benchmark=not args.no_benchmark,
        )
        return

    if args.run_id:
        run_id = args.run_id
    elif args.stable_run_id or args.skip_if_exists:
        run_id = _stable_run_id(cfg, start_date, end_date)
    else:
        run_id = utc_now_id()

    if csv_dir and csv_dir.endswith("/latest"):
        csv_dir = str(Path("data/runs_dynamic") / run_id)

    if args.skip_if_exists and store.run_exists(run_id):
        _export_if_needed(store, run_id, csv_dir)
        print(f"[OK] Run exists; skipped recompute: run_id={run_id}")
        print(f"[INFO] SQLite DB: {Path(db_path)}")
        print(f"[INFO] Portfolio daily rows: {store.row_count('portfolio_daily', run_id)}")
        print(f"[INFO] Holdings rows: {store.row_count('holdings_daily', run_id)}")
        print(f"[INFO] Trades rows: {store.row_count('trades', run_id)}")
        store.close()
        _showresults_if_enabled(
            enabled=not args.no_showresults,
            db_path=db_path,
            run_id=run_id,
            include_benchmark=not args.no_benchmark,
        )
        return

    with RefinitivClient() as client:
        provider = RefinitivStrategyDataProvider(
            client,
            batch_size=max(args.batch_size, 1),
            enable_cache=not args.no_cache,
        )
        result = run_dynamic_rotation(
            config=cfg,
            provider=provider,
            store=store,
            start_date=start_date,
            end_date=end_date,
            run_id=run_id,
        )
        provider_stats = provider.stats()

    _export_if_needed(store, run_id, csv_dir)

    store.close()

    _showresults_if_enabled(
        enabled=not args.no_showresults,
        db_path=db_path,
        run_id=run_id,
        include_benchmark=not args.no_benchmark,
    )

    print(f"[OK] Dynamic strategy run complete: run_id={result.run_id}")
    print(f"[INFO] SQLite DB: {Path(db_path)}")
    print(f"[INFO] Portfolio daily rows: {len(result.portfolio_daily)}")
    print(f"[INFO] Holdings rows: {len(result.holdings_daily)}")
    print(f"[INFO] Trades rows: {len(result.trades)}")
    print(f"[INFO] Provider stats: {provider_stats}")


if __name__ == "__main__":
    main()
