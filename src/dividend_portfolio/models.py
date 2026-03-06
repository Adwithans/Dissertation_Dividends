from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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
class StrategyConfig:
    mode: str = "static"
    universe_scope: str = "sp500"
    candidate_count: int = 100
    portfolio_size: int = 25
    dividend_payer_lookback_months: int = 12
    selection_metric: str = "quarter_dividend_yield"
    yield_denominator: str = "quarter_average_close"
    rebalance_timing: str = "first_trading_day_after_quarter_end"
    initial_selection: str = "market_cap"
    initial_weighting: str = "market_cap"
    quarterly_weighting: str = "normalized_yield_score"
    missing_data_policy: str = "backfill_next_ranked"
    sqlite_path: str = "data/store/portfolio_100.sqlite"
    parquet_dir: str = "data/store/parquet"
    parquet_enabled: bool = True
    csv_export_enabled: bool = False


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
    strategy: StrategyConfig | None = None


@dataclass
class SimulationResult:
    portfolio_df: pd.DataFrame
    asset_results: dict[str, pd.DataFrame]
    effective_start: pd.Timestamp
    rebalance_log: pd.DataFrame | None = None
