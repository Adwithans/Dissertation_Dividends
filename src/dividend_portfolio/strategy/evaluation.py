from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.refinitiv_client import RefinitivClient
from ..models import EvaluationContext, EvaluationResult, PortfolioConfig, StrategyConfig
from ..reporting.dynamic_results import (
    SP500_LABEL_DEFAULT,
    SP500_RIC_DEFAULT,
    RUSSELL_1000_LABEL_DEFAULT,
    RUSSELL_1000_RIC_DEFAULT,
    _fetch_benchmark_close,
    _to_jsonable,
    compute_summary_from_data,
)
from .engine import run_dynamic_rotation
from .provider import RefinitivStrategyDataProvider, utc_now_id
from .storage import StrategyStore


def _normalize_evaluation_context(
    evaluation_context: EvaluationContext | dict[str, Any] | None,
) -> dict[str, Any]:
    if evaluation_context is None:
        return {}
    if isinstance(evaluation_context, EvaluationContext):
        return {k: v for k, v in asdict(evaluation_context).items() if v is not None}
    return {str(k): v for k, v in dict(evaluation_context).items() if v is not None}


def _strategy_or_default(config: PortfolioConfig) -> StrategyConfig:
    if config.strategy is None:
        return StrategyConfig(mode="dynamic_100_25")
    if config.strategy.mode == "static":
        return StrategyConfig(mode="dynamic_100_25")
    return config.strategy


def _apply_hyperparameters(
    base_config: PortfolioConfig,
    hyperparameters: dict[str, Any] | None,
    *,
    persist: str,
) -> PortfolioConfig:
    strategy = _strategy_or_default(base_config)
    values = dict(hyperparameters or {})
    next_strategy = replace(
        strategy,
        portfolio_size=int(values.get("portfolio_size", strategy.portfolio_size)),
        rebalance_interval_quarters=int(
            values.get("rebalance_interval_quarters", strategy.rebalance_interval_quarters)
        ),
        allocation_strategy=str(values.get("allocation_strategy", strategy.allocation_strategy)).strip().lower(),
        parquet_enabled=(strategy.parquet_enabled if persist == "full" else False),
        csv_export_enabled=(strategy.csv_export_enabled if persist == "full" else False),
    )
    return replace(base_config, strategy=next_strategy)


def _extract_hyperparameters(config: PortfolioConfig) -> dict[str, Any]:
    strategy = _strategy_or_default(config)
    return {
        "portfolio_size": int(strategy.portfolio_size),
        "rebalance_interval_quarters": int(strategy.rebalance_interval_quarters),
        "allocation_strategy": str(strategy.allocation_strategy).strip().lower(),
    }


def _fetch_benchmarks(
    *,
    benchmark: str,
    start_date: str,
    end_date: str,
    benchmark_close: pd.DataFrame | None,
    sp500_close: pd.DataFrame | None,
    russell_1000_close: pd.DataFrame | None,
    benchmark_cache_db_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
    if benchmark == "none":
        return (
            benchmark_close.copy() if benchmark_close is not None else empty.copy(),
            sp500_close.copy() if sp500_close is not None else empty.copy(),
            russell_1000_close.copy() if russell_1000_close is not None else empty.copy(),
        )

    if benchmark_close is not None or sp500_close is not None or russell_1000_close is not None:
        return (
            benchmark_close.copy() if benchmark_close is not None else empty.copy(),
            sp500_close.copy() if sp500_close is not None else empty.copy(),
            russell_1000_close.copy() if russell_1000_close is not None else empty.copy(),
        )

    benchmark_df = _fetch_benchmark_close(
        ric=SP500_RIC_DEFAULT,
        start_date=start_date,
        end_date=end_date,
        cache_db_path=benchmark_cache_db_path,
    )
    russell_df = _fetch_benchmark_close(
        ric=RUSSELL_1000_RIC_DEFAULT,
        start_date=start_date,
        end_date=end_date,
        cache_db_path=benchmark_cache_db_path,
    )
    return benchmark_df.copy(), benchmark_df.copy(), russell_df


def prepare_benchmark_data(
    *,
    benchmark: str,
    start_date: str,
    end_date: str,
    benchmark_close: pd.DataFrame | None = None,
    sp500_close: pd.DataFrame | None = None,
    russell_1000_close: pd.DataFrame | None = None,
    benchmark_cache_db_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _fetch_benchmarks(
        benchmark=benchmark,
        start_date=start_date,
        end_date=end_date,
        benchmark_close=benchmark_close,
        sp500_close=sp500_close,
        russell_1000_close=russell_1000_close,
        benchmark_cache_db_path=benchmark_cache_db_path,
    )


def _persist_summary_only(
    *,
    run_id: str,
    summary: dict[str, Any],
    output_dir: str | Path | None,
) -> dict[str, str]:
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path("data/runs_dynamic") / run_id / "summary_only"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, indent=2, sort_keys=False)
    return {"summary_json": str(summary_path)}


def evaluate_strategy(
    *,
    base_config: PortfolioConfig,
    hyperparameters: dict[str, Any] | None = None,
    persist: str = "summary",
    benchmark: str = "sp500",
    evaluation_context: EvaluationContext | dict[str, Any] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
    db_path: str | None = None,
    batch_size: int = 250,
    enable_cache: bool = True,
    persistent_cache_enabled: bool | None = None,
    provider=None,
    benchmark_close: pd.DataFrame | None = None,
    sp500_close: pd.DataFrame | None = None,
    russell_1000_close: pd.DataFrame | None = None,
    benchmark_cache_db_path: str | Path | None = None,
) -> EvaluationResult:
    persist_mode = str(persist).strip().lower()
    if persist_mode not in {"summary", "full", "none"}:
        raise ValueError("persist must be one of: summary, full, none")

    effective_config = _apply_hyperparameters(base_config, hyperparameters, persist=persist_mode)
    strategy = _strategy_or_default(effective_config)
    if db_path is not None:
        strategy = replace(strategy, sqlite_path=str(db_path))
        effective_config = replace(effective_config, strategy=strategy)
    hyperparameter_values = _extract_hyperparameters(effective_config)
    evaluation_context_dict = _normalize_evaluation_context(evaluation_context)

    start = start_date or effective_config.start_date.isoformat()
    end = end_date or (
        effective_config.end_date.isoformat()
        if effective_config.end_date is not None
        else pd.Timestamp.today().date().isoformat()
    )
    resolved_run_id = run_id or utc_now_id()

    store: StrategyStore | None = None
    owns_provider = provider is None
    if persist_mode == "full":
        store = StrategyStore(strategy.sqlite_path)

    if provider is None:
        client = RefinitivClient()
        client.__enter__()
        provider = RefinitivStrategyDataProvider(
            client,
            batch_size=max(int(batch_size), 1),
            enable_cache=bool(enable_cache),
            persistent_cache_db_path=strategy.sqlite_path,
            persistent_cache_enabled=(
                bool(persistent_cache_enabled)
                if persistent_cache_enabled is not None
                else True
            ),
        )
    else:
        client = None

    try:
        result = run_dynamic_rotation(
            config=effective_config,
            provider=provider,
            store=store,
            start_date=start,
            end_date=end,
            run_id=resolved_run_id,
            evaluation_context=evaluation_context_dict,
        )
    finally:
        if owns_provider and provider is not None:
            provider.close()
        if client is not None:
            client.__exit__(None, None, None)

    benchmark_df, sp500_df, russell_df = prepare_benchmark_data(
        benchmark=benchmark,
        start_date=start,
        end_date=end,
        benchmark_close=benchmark_close,
        sp500_close=sp500_close,
        russell_1000_close=russell_1000_close,
        benchmark_cache_db_path=benchmark_cache_db_path,
    )
    metadata = {
        "run_id": resolved_run_id,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "start_date": start,
        "end_date": end,
        "config_json": {
            "portfolio": asdict(effective_config),
            "strategy": asdict(strategy),
            "evaluation_context": evaluation_context_dict,
        },
    }
    summary = compute_summary_from_data(
        run_id=resolved_run_id,
        run_metadata=metadata,
        portfolio_daily=result.portfolio_daily,
        holdings_daily=result.holdings_daily,
        trades=result.trades,
        target_weights=result.target_weights,
        quarter_scores=result.quarter_scores,
        candidate_universe=result.candidate_universe,
        benchmark_close=benchmark_df,
        benchmark_label=SP500_LABEL_DEFAULT,
        sp500_close=sp500_df,
        sp500_label=SP500_LABEL_DEFAULT,
        russell_1000_close=russell_df,
        russell_1000_label=RUSSELL_1000_LABEL_DEFAULT,
    )

    persisted_artifacts: dict[str, str] = {}
    if persist_mode == "summary":
        persisted_artifacts = _persist_summary_only(
            run_id=resolved_run_id,
            summary=summary,
            output_dir=output_dir,
        )
    elif persist_mode == "full":
        persisted_artifacts["sqlite_db"] = str(Path(strategy.sqlite_path))
    if store is not None:
        store.close()

    return EvaluationResult(
        run_id=resolved_run_id,
        hyperparameters=hyperparameter_values,
        objective_metrics=dict(summary.get("objective_metrics", {})),
        constraint_metrics=dict(summary.get("constraint_metrics", {})),
        diagnostic_metrics=dict(summary.get("diagnostic_metrics", {})),
        summary=summary,
        persisted_artifacts=persisted_artifacts,
        dynamic_run=result,
    )
