from __future__ import annotations

from pathlib import Path

import pytest

from src.dividend_portfolio.config import load_portfolio_config


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_selection_policy_defaults_to_full_refresh(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "portfolio.yaml",
        """
base_currency: USD
initial_capital: 1000000
start_date: 2024-01-01
end_date: 2024-12-31
reinvest_dividends: false
auto_align_splits: true
use_cum_factor: true
risk_free_rate: 0.0
rebalancing:
  enabled: true
  frequency: quarterly
  trigger: first_trading_day_after_quarter_end
  drift_tolerance: 0.0
quarterly_metrics:
  enabled: true
  dividend_return_basis: quarter_start_market_value
assets:
  - ric: AAPL.O
    weight: 1.0
strategy:
  mode: dynamic_100_25
  universe_scope: sp500
  candidate_count: 100
  portfolio_size: 25
  dividend_payer_lookback: 12
  selection_metric: quarter_dividend_yield
  yield_denominator: quarter_average_close
  rebalance_timing: first_trading_day_after_quarter_end
  initial_selection: market_cap
  initial_weighting: market_cap
  quarterly_weighting: normalized_yield_score
  missing_data_policy: backfill_next_ranked
  sqlite_path: data/store/portfolio_100.sqlite
  parquet_dir: data/store/parquet
  parquet_enabled: true
  csv_export_enabled: false
""".strip(),
    )
    cfg = load_portfolio_config(cfg_path)
    assert cfg.strategy is not None
    assert cfg.strategy.selection_policy.name == "full_refresh"
    assert cfg.strategy.selection_policy.max_replacements_per_quarter == 25
    assert cfg.strategy.rebalance_interval_quarters == 1


def test_load_config_dynamic_mode_allows_missing_assets(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "portfolio.yaml",
        """
base_currency: USD
initial_capital: 1000000
start_date: 2024-01-01
end_date: 2024-12-31
reinvest_dividends: false
auto_align_splits: true
use_cum_factor: true
risk_free_rate: 0.0
rebalancing:
  enabled: true
  frequency: quarterly
  trigger: first_trading_day_after_quarter_end
  drift_tolerance: 0.0
quarterly_metrics:
  enabled: true
  dividend_return_basis: quarter_start_market_value
strategy:
  mode: dynamic_100_25
  universe_scope: sp500
  candidate_count: 100
  portfolio_size: 25
  dividend_payer_lookback: 12
  selection_metric: quarter_dividend_yield
  yield_denominator: quarter_average_close
  rebalance_timing: first_trading_day_after_quarter_end
  initial_selection: market_cap
  initial_weighting: market_cap
  quarterly_weighting: normalized_yield_score
  missing_data_policy: backfill_next_ranked
  sqlite_path: data/store/portfolio_100.sqlite
  parquet_dir: data/store/parquet
  parquet_enabled: true
  csv_export_enabled: false
""".strip(),
    )
    cfg = load_portfolio_config(cfg_path)
    assert cfg.strategy is not None
    assert cfg.strategy.mode == "dynamic_100_25"
    assert cfg.assets == []


def test_load_config_selection_policy_replace_bottom_n_and_experiment_group(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "portfolio.yaml",
        """
base_currency: USD
initial_capital: 1000000
start_date: 2024-01-01
end_date: 2024-12-31
reinvest_dividends: false
auto_align_splits: true
use_cum_factor: true
risk_free_rate: 0.0
rebalancing:
  enabled: true
  frequency: quarterly
  trigger: first_trading_day_after_quarter_end
  drift_tolerance: 0.0
quarterly_metrics:
  enabled: true
  dividend_return_basis: quarter_start_market_value
assets:
  - ric: AAPL.O
    weight: 1.0
strategy:
  mode: dynamic_100_25
  universe_scope: sp500
  candidate_count: 100
  portfolio_size: 25
  rebalance_interval_quarters: 2
  dividend_payer_lookback: 12
  selection_metric: quarter_dividend_yield
  yield_denominator: quarter_average_close
  rebalance_timing: first_trading_day_after_quarter_end
  initial_selection: market_cap
  initial_weighting: market_cap
  quarterly_weighting: normalized_yield_score
  missing_data_policy: backfill_next_ranked
  sqlite_path: data/store/portfolio_100.sqlite
  parquet_dir: data/store/parquet
  parquet_enabled: true
  csv_export_enabled: false
  experiment_group: alt_reallocations
  selection_policy:
    name: replace_bottom_n
    max_replacements_per_quarter: 10
    rank_metric: quarter_dividend_yield_score
""".strip(),
    )
    cfg = load_portfolio_config(cfg_path)
    assert cfg.strategy is not None
    assert cfg.strategy.selection_policy.name == "replace_bottom_n"
    assert cfg.strategy.selection_policy.max_replacements_per_quarter == 10
    assert cfg.strategy.rebalance_interval_quarters == 2
    assert cfg.strategy.experiment_group == "alt_reallocations"


def test_load_config_selection_policy_rejects_invalid_name(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "portfolio.yaml",
        """
base_currency: USD
initial_capital: 1000000
start_date: 2024-01-01
end_date: 2024-12-31
reinvest_dividends: false
auto_align_splits: true
use_cum_factor: true
risk_free_rate: 0.0
rebalancing:
  enabled: true
  frequency: quarterly
  trigger: first_trading_day_after_quarter_end
  drift_tolerance: 0.0
quarterly_metrics:
  enabled: true
  dividend_return_basis: quarter_start_market_value
assets:
  - ric: AAPL.O
    weight: 1.0
strategy:
  mode: dynamic_100_25
  universe_scope: sp500
  candidate_count: 100
  portfolio_size: 25
  dividend_payer_lookback: 12
  selection_metric: quarter_dividend_yield
  yield_denominator: quarter_average_close
  rebalance_timing: first_trading_day_after_quarter_end
  initial_selection: market_cap
  initial_weighting: market_cap
  quarterly_weighting: normalized_yield_score
  missing_data_policy: backfill_next_ranked
  sqlite_path: data/store/portfolio_100.sqlite
  parquet_dir: data/store/parquet
  parquet_enabled: true
  csv_export_enabled: false
  selection_policy:
    name: not_supported
""".strip(),
    )
    with pytest.raises(ValueError, match="strategy.selection_policy.name must be one of"):
        _ = load_portfolio_config(cfg_path)


def test_load_config_rejects_non_positive_rebalance_interval_quarters(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "portfolio.yaml",
        """
base_currency: USD
initial_capital: 1000000
start_date: 2024-01-01
end_date: 2024-12-31
reinvest_dividends: false
auto_align_splits: true
use_cum_factor: true
risk_free_rate: 0.0
rebalancing:
  enabled: true
  frequency: quarterly
  trigger: first_trading_day_after_quarter_end
  drift_tolerance: 0.0
quarterly_metrics:
  enabled: true
  dividend_return_basis: quarter_start_market_value
assets:
  - ric: AAPL.O
    weight: 1.0
strategy:
  mode: dynamic_100_25
  universe_scope: sp500
  candidate_count: 100
  portfolio_size: 25
  rebalance_interval_quarters: 0
  dividend_payer_lookback: 12
  selection_metric: quarter_dividend_yield
  yield_denominator: quarter_average_close
  rebalance_timing: first_trading_day_after_quarter_end
  initial_selection: market_cap
  initial_weighting: market_cap
  quarterly_weighting: normalized_yield_score
  missing_data_policy: backfill_next_ranked
  sqlite_path: data/store/portfolio_100.sqlite
  parquet_dir: data/store/parquet
  parquet_enabled: true
  csv_export_enabled: false
""".strip(),
    )
    with pytest.raises(ValueError, match="strategy.rebalance_interval_quarters must be > 0"):
        _ = load_portfolio_config(cfg_path)


def test_load_config_accepts_allocation_strategy(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "portfolio.yaml",
        """
base_currency: USD
initial_capital: 1000000
start_date: 2024-01-01
end_date: 2024-12-31
reinvest_dividends: false
auto_align_splits: true
use_cum_factor: true
risk_free_rate: 0.0
rebalancing:
  enabled: true
  frequency: quarterly
  trigger: first_trading_day_after_quarter_end
  drift_tolerance: 0.0
quarterly_metrics:
  enabled: true
  dividend_return_basis: quarter_start_market_value
strategy:
  mode: dynamic_100_25
  universe_scope: sp500
  candidate_count: 100
  portfolio_size: 25
  dividend_payer_lookback: 12
  selection_metric: quarter_dividend_yield
  yield_denominator: quarter_average_close
  rebalance_timing: first_trading_day_after_quarter_end
  initial_selection: market_cap
  initial_weighting: market_cap
  allocation_strategy: inverse_market_cap
  missing_data_policy: backfill_next_ranked
  sqlite_path: data/store/portfolio_100.sqlite
  parquet_dir: data/store/parquet
  parquet_enabled: true
  csv_export_enabled: false
""".strip(),
    )
    cfg = load_portfolio_config(cfg_path)
    assert cfg.strategy is not None
    assert cfg.strategy.allocation_strategy == "inverse_market_cap"


def test_load_config_legacy_quarterly_weighting_maps_to_allocation_strategy(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "portfolio.yaml",
        """
base_currency: USD
initial_capital: 1000000
start_date: 2024-01-01
end_date: 2024-12-31
reinvest_dividends: false
auto_align_splits: true
use_cum_factor: true
risk_free_rate: 0.0
rebalancing:
  enabled: true
  frequency: quarterly
  trigger: first_trading_day_after_quarter_end
  drift_tolerance: 0.0
quarterly_metrics:
  enabled: true
  dividend_return_basis: quarter_start_market_value
strategy:
  mode: dynamic_100_25
  universe_scope: sp500
  candidate_count: 100
  portfolio_size: 25
  dividend_payer_lookback: 12
  selection_metric: quarter_dividend_yield
  yield_denominator: quarter_average_close
  rebalance_timing: first_trading_day_after_quarter_end
  initial_selection: market_cap
  initial_weighting: market_cap
  quarterly_weighting: normalized_yield_score
  missing_data_policy: backfill_next_ranked
  sqlite_path: data/store/portfolio_100.sqlite
  parquet_dir: data/store/parquet
  parquet_enabled: true
  csv_export_enabled: false
""".strip(),
    )
    cfg = load_portfolio_config(cfg_path)
    assert cfg.strategy is not None
    assert cfg.strategy.allocation_strategy == "yield_proportional"
    assert cfg.strategy.quarterly_weighting == "normalized_yield_score"


def test_load_config_rejects_conflicting_allocation_strategy_and_legacy_alias(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "portfolio.yaml",
        """
base_currency: USD
initial_capital: 1000000
start_date: 2024-01-01
end_date: 2024-12-31
reinvest_dividends: false
auto_align_splits: true
use_cum_factor: true
risk_free_rate: 0.0
rebalancing:
  enabled: true
  frequency: quarterly
  trigger: first_trading_day_after_quarter_end
  drift_tolerance: 0.0
quarterly_metrics:
  enabled: true
  dividend_return_basis: quarter_start_market_value
strategy:
  mode: dynamic_100_25
  universe_scope: sp500
  candidate_count: 100
  portfolio_size: 25
  dividend_payer_lookback: 12
  selection_metric: quarter_dividend_yield
  yield_denominator: quarter_average_close
  rebalance_timing: first_trading_day_after_quarter_end
  initial_selection: market_cap
  initial_weighting: market_cap
  allocation_strategy: market_cap
  quarterly_weighting: normalized_yield_score
  missing_data_policy: backfill_next_ranked
  sqlite_path: data/store/portfolio_100.sqlite
  parquet_dir: data/store/parquet
  parquet_enabled: true
  csv_export_enabled: false
""".strip(),
    )
    with pytest.raises(ValueError, match="conflicts with legacy"):
        _ = load_portfolio_config(cfg_path)


def test_load_config_bond_universe_and_baseline_sell_default_to_disabled(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "portfolio.yaml",
        """
base_currency: USD
initial_capital: 1000000
start_date: 2024-01-01
end_date: 2024-12-31
reinvest_dividends: false
auto_align_splits: true
use_cum_factor: true
risk_free_rate: 0.0
rebalancing:
  enabled: true
  frequency: quarterly
  trigger: first_trading_day_after_quarter_end
  drift_tolerance: 0.0
quarterly_metrics:
  enabled: true
  dividend_return_basis: quarter_start_market_value
strategy:
  mode: dynamic_100_25
  universe_scope: sp500
  candidate_count: 100
  portfolio_size: 25
  sqlite_path: data/store/portfolio_100.sqlite
  parquet_dir: data/store/parquet
  parquet_enabled: false
  csv_export_enabled: false
""".strip(),
    )
    cfg = load_portfolio_config(cfg_path)
    assert cfg.strategy is not None
    assert cfg.strategy.bond_universe.enabled is False
    assert cfg.strategy.baseline_sell_enabled is False
    assert cfg.strategy.baseline_sell_threshold == pytest.approx(0.10)
    assert cfg.strategy.bond_universe.rics == ("IEF.OQ", "TLT.OQ")


def test_load_config_rejects_baseline_sell_without_bond_universe(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "portfolio.yaml",
        """
base_currency: USD
initial_capital: 1000000
start_date: 2024-01-01
end_date: 2024-12-31
reinvest_dividends: false
auto_align_splits: true
use_cum_factor: true
risk_free_rate: 0.0
rebalancing:
  enabled: true
  frequency: quarterly
  trigger: first_trading_day_after_quarter_end
  drift_tolerance: 0.0
quarterly_metrics:
  enabled: true
  dividend_return_basis: quarter_start_market_value
strategy:
  mode: dynamic_100_25
  universe_scope: sp500
  candidate_count: 100
  portfolio_size: 25
  baseline_sell_enabled: true
  sqlite_path: data/store/portfolio_100.sqlite
  parquet_dir: data/store/parquet
  parquet_enabled: false
  csv_export_enabled: false
""".strip(),
    )
    with pytest.raises(ValueError, match="baseline_sell_enabled requires"):
        _ = load_portfolio_config(cfg_path)
