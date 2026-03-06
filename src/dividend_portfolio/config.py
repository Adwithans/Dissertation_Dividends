from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .models import (
    AssetConfig,
    PortfolioConfig,
    QuarterlyMetricsConfig,
    RebalanceConfig,
    StrategyConfig,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return obj


def _require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required config key '{key}'")
    return data[key]


def _parse_date(value: Any, field_name: str) -> date | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid ISO date for '{field_name}': {value}") from exc


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    cfg_path = Path(path)
    data = _read_yaml(cfg_path)

    base_currency = str(_require(data, "base_currency")).upper()
    if base_currency != "USD":
        raise ValueError("This configuration is USD-only. Set base_currency to 'USD'.")

    initial_capital = float(_require(data, "initial_capital"))
    if initial_capital <= 0:
        raise ValueError("initial_capital must be > 0")

    start_date = _parse_date(_require(data, "start_date"), "start_date")
    if start_date is None:
        raise ValueError("start_date is required")

    end_date = _parse_date(data.get("end_date"), "end_date")
    if end_date is not None and end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    reinvest_dividends = bool(data.get("reinvest_dividends", False))
    auto_align_splits = bool(data.get("auto_align_splits", True))
    use_cum_factor = bool(data.get("use_cum_factor", True))
    risk_free_rate = float(data.get("risk_free_rate", 0.0))

    rebalancing_raw = data.get("rebalancing", {})
    if not isinstance(rebalancing_raw, dict):
        raise ValueError("rebalancing must be an object")
    rebalance_enabled = bool(rebalancing_raw.get("enabled", False))
    rebalance_frequency = str(rebalancing_raw.get("frequency", "quarterly")).strip().lower()
    rebalance_trigger = str(
        rebalancing_raw.get("trigger", "first_trading_day_after_quarter_end")
    ).strip()
    rebalance_drift_tolerance = float(rebalancing_raw.get("drift_tolerance", 0.0))

    if rebalance_frequency not in {"quarterly", "monthly", "yearly"}:
        raise ValueError(
            "rebalancing.frequency must be one of: quarterly, monthly, yearly"
        )
    if rebalance_drift_tolerance < 0:
        raise ValueError("rebalancing.drift_tolerance must be >= 0")

    quarterly_metrics_raw = data.get("quarterly_metrics", {})
    if not isinstance(quarterly_metrics_raw, dict):
        raise ValueError("quarterly_metrics must be an object")
    quarterly_metrics_enabled = bool(quarterly_metrics_raw.get("enabled", True))
    dividend_return_basis = str(
        quarterly_metrics_raw.get("dividend_return_basis", "quarter_start_market_value")
    ).strip()
    if dividend_return_basis != "quarter_start_market_value":
        raise ValueError(
            "quarterly_metrics.dividend_return_basis must be 'quarter_start_market_value'"
        )

    assets_raw = _require(data, "assets")
    if not isinstance(assets_raw, list) or not assets_raw:
        raise ValueError("assets must be a non-empty list")

    assets: list[AssetConfig] = []
    seen_rics: set[str] = set()
    weight_sum = 0.0

    for idx, item in enumerate(assets_raw):
        if not isinstance(item, dict):
            raise ValueError(f"assets[{idx}] must be an object")

        ric = str(_require(item, "ric")).strip()
        if not ric:
            raise ValueError(f"assets[{idx}].ric cannot be empty")
        if ric in seen_rics:
            raise ValueError(f"Duplicate ric in assets: {ric}")

        weight = float(_require(item, "weight"))
        if weight <= 0:
            raise ValueError(f"assets[{idx}].weight must be > 0")

        assets.append(AssetConfig(ric=ric, weight=weight))
        seen_rics.add(ric)
        weight_sum += weight

    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Asset weights must sum to 1.0, got {weight_sum:.12f}")

    strategy_raw = data.get("strategy")
    strategy: StrategyConfig | None = None
    if strategy_raw is not None:
        if not isinstance(strategy_raw, dict):
            raise ValueError("strategy must be an object")

        mode = str(strategy_raw.get("mode", "static")).strip().lower()
        universe_scope = str(strategy_raw.get("universe_scope", "sp500")).strip().lower()
        candidate_count = int(strategy_raw.get("candidate_count", 100))
        portfolio_size = int(strategy_raw.get("portfolio_size", 25))
        lookback_months = int(strategy_raw.get("dividend_payer_lookback", 12))
        selection_metric = str(
            strategy_raw.get("selection_metric", "quarter_dividend_yield")
        ).strip().lower()
        yield_denominator = str(
            strategy_raw.get("yield_denominator", "quarter_average_close")
        ).strip().lower()
        rebalance_timing = str(
            strategy_raw.get("rebalance_timing", "first_trading_day_after_quarter_end")
        ).strip()
        initial_selection = str(strategy_raw.get("initial_selection", "market_cap")).strip().lower()
        initial_weighting = str(strategy_raw.get("initial_weighting", "market_cap")).strip().lower()
        quarterly_weighting = str(
            strategy_raw.get("quarterly_weighting", "normalized_yield_score")
        ).strip().lower()
        missing_data_policy = str(
            strategy_raw.get("missing_data_policy", "backfill_next_ranked")
        ).strip().lower()
        sqlite_path = str(strategy_raw.get("sqlite_path", "data/store/portfolio_100.sqlite")).strip()
        parquet_dir = str(strategy_raw.get("parquet_dir", "data/store/parquet")).strip()
        parquet_enabled = bool(strategy_raw.get("parquet_enabled", True))
        csv_export_enabled = bool(strategy_raw.get("csv_export_enabled", False))

        allowed_modes = {"static", "dynamic_100_25"}
        if mode not in allowed_modes:
            raise ValueError(f"strategy.mode must be one of {sorted(allowed_modes)}")
        if universe_scope != "sp500":
            raise ValueError("strategy.universe_scope must be 'sp500'")
        if candidate_count <= 0:
            raise ValueError("strategy.candidate_count must be > 0")
        if portfolio_size <= 0:
            raise ValueError("strategy.portfolio_size must be > 0")
        if portfolio_size > candidate_count:
            raise ValueError("strategy.portfolio_size must be <= strategy.candidate_count")
        if lookback_months <= 0:
            raise ValueError("strategy.dividend_payer_lookback must be > 0")
        if selection_metric != "quarter_dividend_yield":
            raise ValueError("strategy.selection_metric must be 'quarter_dividend_yield'")
        allowed_denoms = {"quarter_average_close", "quarter_start_close", "quarter_end_close"}
        if yield_denominator not in allowed_denoms:
            raise ValueError(
                "strategy.yield_denominator must be one of quarter_average_close, "
                "quarter_start_close, quarter_end_close"
            )
        if rebalance_timing != "first_trading_day_after_quarter_end":
            raise ValueError(
                "strategy.rebalance_timing must be 'first_trading_day_after_quarter_end'"
            )
        if initial_selection != "market_cap":
            raise ValueError("strategy.initial_selection must be 'market_cap'")
        if initial_weighting != "market_cap":
            raise ValueError("strategy.initial_weighting must be 'market_cap'")
        if quarterly_weighting != "normalized_yield_score":
            raise ValueError("strategy.quarterly_weighting must be 'normalized_yield_score'")
        if missing_data_policy != "backfill_next_ranked":
            raise ValueError("strategy.missing_data_policy must be 'backfill_next_ranked'")

        strategy = StrategyConfig(
            mode=mode,
            universe_scope=universe_scope,
            candidate_count=candidate_count,
            portfolio_size=portfolio_size,
            dividend_payer_lookback_months=lookback_months,
            selection_metric=selection_metric,
            yield_denominator=yield_denominator,
            rebalance_timing=rebalance_timing,
            initial_selection=initial_selection,
            initial_weighting=initial_weighting,
            quarterly_weighting=quarterly_weighting,
            missing_data_policy=missing_data_policy,
            sqlite_path=sqlite_path,
            parquet_dir=parquet_dir,
            parquet_enabled=parquet_enabled,
            csv_export_enabled=csv_export_enabled,
        )

    return PortfolioConfig(
        base_currency=base_currency,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        reinvest_dividends=reinvest_dividends,
        auto_align_splits=auto_align_splits,
        use_cum_factor=use_cum_factor,
        risk_free_rate=risk_free_rate,
        rebalancing=RebalanceConfig(
            enabled=rebalance_enabled,
            frequency=rebalance_frequency,
            trigger=rebalance_trigger,
            drift_tolerance=rebalance_drift_tolerance,
        ),
        quarterly_metrics=QuarterlyMetricsConfig(
            enabled=quarterly_metrics_enabled,
            dividend_return_basis=dividend_return_basis,
        ),
        assets=assets,
        strategy=strategy,
    )
