from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_portfolio_config
from ..models import PortfolioConfig, StrategyConfig
from ..reporting.dynamic_results import generate_dynamic_showresults
from ..strategy.evaluation import evaluate_strategy
from ..strategy.provider import utc_now_id
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
    parser.add_argument(
        "--experiment-group",
        default=None,
        help="Optional experiment group name for writing cross-run comparison tables.",
    )
    parser.add_argument(
        "--portfolio-size",
        type=int,
        default=None,
        help="Override strategy.portfolio_size for this run.",
    )
    parser.add_argument(
        "--rebalance-interval-quarters",
        type=int,
        default=None,
        help="Override strategy.rebalance_interval_quarters for this run.",
    )
    parser.add_argument(
        "--allocation-strategy",
        default=None,
        help="Override strategy.allocation_strategy for this run.",
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


def _normalized_allocation_strategy(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized == "normalized_yield_score":
        return "yield_proportional"
    return normalized


def _apply_cli_hyperparameter_overrides(config_obj: PortfolioConfig, args: argparse.Namespace) -> PortfolioConfig:
    strategy = config_obj.strategy or StrategyConfig(mode="dynamic_100_25")
    next_strategy = replace(
        strategy,
        portfolio_size=(
            int(args.portfolio_size)
            if args.portfolio_size is not None
            else int(strategy.portfolio_size)
        ),
        rebalance_interval_quarters=(
            int(args.rebalance_interval_quarters)
            if args.rebalance_interval_quarters is not None
            else int(strategy.rebalance_interval_quarters)
        ),
        allocation_strategy=(
            _normalized_allocation_strategy(args.allocation_strategy)
            if _normalized_allocation_strategy(args.allocation_strategy) is not None
            else str(strategy.allocation_strategy).strip().lower()
        ),
    )
    return replace(config_obj, strategy=next_strategy)


def _export_if_needed(store: StrategyStore, run_id: str, csv_dir: str | None) -> None:
    if csv_dir:
        exported = store.export_run_csv(run_id, csv_dir)
        print(f"[OK] Exported CSV outputs to: {exported}")


def _normalize_experiment_group(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    safe = safe.strip("._")
    return safe or "default"


def _build_experiment_comparison_row(summary: dict[str, Any]) -> dict[str, Any]:
    portfolio_metrics = summary.get("portfolio_metrics", {}) if isinstance(summary, dict) else {}
    trading_activity = summary.get("trading_activity", {}) if isinstance(summary, dict) else {}
    transaction_costs = summary.get("transaction_costs", {}) if isinstance(summary, dict) else {}
    dividends = summary.get("dividends", {}) if isinstance(summary, dict) else {}
    objective_metrics = summary.get("objective_metrics", {}) if isinstance(summary, dict) else {}
    constraint_metrics = summary.get("constraint_metrics", {}) if isinstance(summary, dict) else {}
    strategy_section = summary.get("strategy", {}) if isinstance(summary, dict) else {}
    hyperparameters = summary.get("hyperparameters", {}) if isinstance(summary, dict) else {}
    run_id = str(summary.get("run_id", "")).strip()

    return {
        "run_id": run_id,
        "created_at_utc": summary.get("created_at_utc"),
        "start_date": summary.get("start_date"),
        "end_date": summary.get("end_date"),
        "portfolio_size": hyperparameters.get("portfolio_size", strategy_section.get("portfolio_size")),
        "rebalance_interval_quarters": hyperparameters.get(
            "rebalance_interval_quarters",
            strategy_section.get("rebalance_interval_quarters", 1),
        ),
        "allocation_strategy": hyperparameters.get(
            "allocation_strategy",
            strategy_section.get("allocation_strategy"),
        ),
        "policy_name": strategy_section.get("selection_policy_name", "full_refresh"),
        "max_replacements_per_quarter": strategy_section.get("max_replacements_per_quarter"),
        "total_return": portfolio_metrics.get("total_return"),
        "cagr": objective_metrics.get("cagr", portfolio_metrics.get("cagr")),
        "sortino_ratio": objective_metrics.get("sortino_ratio", portfolio_metrics.get("sortino_ratio")),
        "calmar_ratio": objective_metrics.get("calmar_ratio", portfolio_metrics.get("calmar_ratio")),
        "information_ratio": objective_metrics.get("information_ratio"),
        "annualized_excess_return": objective_metrics.get("annualized_excess_return"),
        "sharpe_ratio": portfolio_metrics.get("sharpe_ratio"),
        "max_drawdown": constraint_metrics.get("max_drawdown", portfolio_metrics.get("max_drawdown")),
        "end_value": portfolio_metrics.get("end_value"),
        "tracking_error_annualized": constraint_metrics.get("tracking_error_annualized"),
        "gross_turnover": constraint_metrics.get("gross_turnover", trading_activity.get("gross_turnover")),
        "trade_count": trading_activity.get("number_of_trades"),
        "total_transaction_cost": constraint_metrics.get(
            "total_transaction_cost",
            transaction_costs.get("total_transaction_cost"),
        ),
        "cost_drag_pct_of_start_value": constraint_metrics.get(
            "cost_drag_pct_of_start_value",
            transaction_costs.get("cost_drag_pct_of_start_value"),
        ),
        "total_dividend_cash": dividends.get("total_dividend_cash"),
        "dividend_share_of_total_gain": dividends.get("dividend_share_of_total_gain"),
    }


def _upsert_experiment_comparison(
    *,
    summary: dict[str, Any],
    experiment_group: str,
    base_dir: str | Path = "data/runs_dynamic/experiments",
) -> tuple[Path, Path]:
    row = _build_experiment_comparison_row(summary)
    run_id = str(row.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("summary.run_id is required for experiment comparison upsert")

    group_dir = Path(base_dir) / experiment_group
    group_dir.mkdir(parents=True, exist_ok=True)
    csv_path = group_dir / "comparison.csv"
    json_path = group_dir / "comparison.json"

    columns = list(row.keys())
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
    else:
        existing = pd.DataFrame(columns=columns)

    for col in columns:
        if col not in existing.columns:
            existing[col] = None
    existing = existing[columns]

    new_row = pd.DataFrame([row], columns=columns)
    out = new_row.copy() if existing.empty else pd.concat([existing, new_row], ignore_index=True)
    out = out.drop_duplicates(subset=["run_id"], keep="last")
    if "created_at_utc" in out.columns:
        out = out.sort_values(["created_at_utc", "run_id"], ascending=[True, True], na_position="last")
    else:
        out = out.sort_values(["run_id"], ascending=[True])
    out = out.reset_index(drop=True)

    out.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out.to_dict(orient="records"), f, indent=2, sort_keys=False)
    return csv_path, json_path


def _showresults_if_enabled(
    *,
    enabled: bool,
    db_path: str,
    run_id: str,
    include_benchmark: bool,
    experiment_group: str | None,
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
        summary_path = Path(generated_dir) / "summary.json"
        print(f"[INFO] showresults summary: {summary_path}")
        if experiment_group:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            csv_path, json_path = _upsert_experiment_comparison(
                summary=summary,
                experiment_group=experiment_group,
            )
            print(f"[OK] Experiment comparison updated: {csv_path}")
            print(f"[INFO] Experiment comparison JSON: {json_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] showresults generation failed for run_id={run_id}: {exc}")


def main() -> None:
    args = parse_args()
    cfg = _apply_cli_hyperparameter_overrides(load_portfolio_config(args.config), args)
    strategy = cfg.strategy or StrategyConfig(mode="dynamic_100_25")
    if strategy.mode not in {"dynamic_100_25", "static"}:
        raise ValueError(f"Unsupported strategy.mode for dynamic run: {strategy.mode}")
    if args.reuse_run_id and (args.run_id or args.stable_run_id):
        raise ValueError("--reuse-run-id cannot be combined with --run-id or --stable-run-id")

    start_date = args.start_date or cfg.start_date.isoformat()
    end_date = args.end_date or (cfg.end_date.isoformat() if cfg.end_date else date.today().isoformat())
    db_path = args.db_path or strategy.sqlite_path
    experiment_group = _normalize_experiment_group(args.experiment_group) or _normalize_experiment_group(
        strategy.experiment_group
    )
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
            experiment_group=experiment_group,
        )
        return

    store.close()

    if args.run_id:
        run_id = args.run_id
    elif args.stable_run_id or args.skip_if_exists:
        run_id = _stable_run_id(cfg, start_date, end_date)
    else:
        run_id = utc_now_id()

    if csv_dir and csv_dir.endswith("/latest"):
        csv_dir = str(Path("data/runs_dynamic") / run_id)

    if args.skip_if_exists:
        check_store = StrategyStore(db_path)
        run_exists = check_store.run_exists(run_id)
        if run_exists:
            _export_if_needed(check_store, run_id, csv_dir)
            print(f"[OK] Run exists; skipped recompute: run_id={run_id}")
            print(f"[INFO] SQLite DB: {Path(db_path)}")
            print(f"[INFO] Portfolio daily rows: {check_store.row_count('portfolio_daily', run_id)}")
            print(f"[INFO] Holdings rows: {check_store.row_count('holdings_daily', run_id)}")
            print(f"[INFO] Trades rows: {check_store.row_count('trades', run_id)}")
            check_store.close()
            _showresults_if_enabled(
                enabled=not args.no_showresults,
                db_path=db_path,
                run_id=run_id,
                include_benchmark=not args.no_benchmark,
                experiment_group=experiment_group,
            )
            return
        check_store.close()

    evaluation = evaluate_strategy(
        base_config=cfg,
        hyperparameters=None,
        persist="full",
        benchmark="none",
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        db_path=db_path,
        batch_size=max(args.batch_size, 1),
        enable_cache=not args.no_cache,
        persistent_cache_enabled=True,
    )
    result = evaluation.dynamic_run

    export_store = StrategyStore(db_path)
    _export_if_needed(export_store, run_id, csv_dir)
    export_store.close()

    _showresults_if_enabled(
        enabled=not args.no_showresults,
        db_path=db_path,
        run_id=run_id,
        include_benchmark=not args.no_benchmark,
        experiment_group=experiment_group,
    )

    print(f"[OK] Dynamic strategy run complete: run_id={evaluation.run_id}")
    print(f"[INFO] SQLite DB: {Path(db_path)}")
    print(f"[INFO] Portfolio daily rows: {len(result.portfolio_daily) if result is not None else 0}")
    print(f"[INFO] Holdings rows: {len(result.holdings_daily) if result is not None else 0}")
    print(f"[INFO] Trades rows: {len(result.trades) if result is not None else 0}")
    return


if __name__ == "__main__":
    main()
