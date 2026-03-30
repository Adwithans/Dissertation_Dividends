from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AssetConfig:
    ric: str
    weight: float


@dataclass(frozen=True)
class RebalanceConfig:
    enabled: bool
    frequency: str
    trigger: str
    drift_tolerance: float


@dataclass(frozen=True)
class QuarterlyMetricsConfig:
    enabled: bool
    dividend_return_basis: str


@dataclass(frozen=True)
class TransactionCostsConfig:
    enabled: bool = False
    commission_bps: float = 1.0
    commission_min_usd: float = 1.0
    slippage_bps_per_side: float = 2.0
    fallback_full_spread_bps: float = 5.0
    use_bid_ask_when_available: bool = True
    sizing_rule: str = "cost_aware_scaling"


@dataclass(frozen=True)
class SelectionPolicyConfig:
    name: str = "full_refresh"
    max_replacements_per_quarter: int = 25
    rank_metric: str = "quarter_dividend_yield_score"


@dataclass(frozen=True)
class StrategyConfig:
    mode: str = "static"
    universe_scope: str = "sp500"
    candidate_count: int = 100
    portfolio_size: int = 25
    rebalance_interval_quarters: int = 1
    dividend_payer_lookback_months: int = 12
    selection_metric: str = "quarter_dividend_yield"
    yield_denominator: str = "quarter_average_close"
    rebalance_timing: str = "first_trading_day_after_quarter_end"
    initial_selection: str = "market_cap"
    initial_weighting: str = "market_cap"
    allocation_strategy: str = "yield_proportional"
    quarterly_weighting: str | None = None
    missing_data_policy: str = "backfill_next_ranked"
    sqlite_path: str = "data/store/portfolio_100.sqlite"
    parquet_dir: str = "data/store/parquet"
    parquet_enabled: bool = True
    csv_export_enabled: bool = False
    selection_policy: SelectionPolicyConfig = field(default_factory=SelectionPolicyConfig)
    experiment_group: str | None = None


@dataclass(frozen=True)
class PortfolioConfig:
    base_currency: str
    initial_capital: float
    start_date: date
    end_date: date | None
    reinvest_dividends: bool
    auto_align_splits: bool
    use_cum_factor: bool
    risk_free_rate: float
    rebalancing: RebalanceConfig
    quarterly_metrics: QuarterlyMetricsConfig
    assets: list[AssetConfig]
    transaction_costs: TransactionCostsConfig = field(default_factory=TransactionCostsConfig)
    strategy: StrategyConfig | None = None


@dataclass
class SimulationResult:
    portfolio_df: pd.DataFrame
    asset_results: dict[str, pd.DataFrame]
    effective_start: pd.Timestamp
    rebalance_log: pd.DataFrame | None = None


@dataclass(frozen=True)
class EvaluationContext:
    study_id: str | None = None
    trial_id: str | None = None
    search_space_id: str | None = None
    trial_index: int | None = None
    total_trials_attempted: int | None = None
    effective_independent_trial_count_estimate: float | None = None


@dataclass
class EvaluationResult:
    run_id: str
    hyperparameters: dict[str, Any]
    objective_metrics: dict[str, Any]
    constraint_metrics: dict[str, Any]
    diagnostic_metrics: dict[str, Any]
    summary: dict[str, Any]
    persisted_artifacts: dict[str, str] = field(default_factory=dict)
    dynamic_run: Any | None = None
