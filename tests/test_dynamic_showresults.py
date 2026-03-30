from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.dividend_portfolio.analytics.volatility_models import fit_arch_garch_models
import src.dividend_portfolio.reporting.dynamic_results as dynamic_results_module
from src.dividend_portfolio.reporting.dynamic_results import (
    _fetch_benchmark_close,
    _quadrant_period_stats,
    compute_holding_presence_stats,
    compute_summary_from_data,
)


def test_compute_holding_presence_stats_longest_streak() -> None:
    portfolio_dates = pd.bdate_range("2024-01-02", periods=6)
    holdings = pd.DataFrame(
        [
            {"date": portfolio_dates[0], "ric": "AAA", "shares": 10},
            {"date": portfolio_dates[1], "ric": "AAA", "shares": 10},
            {"date": portfolio_dates[2], "ric": "AAA", "shares": 10},
            {"date": portfolio_dates[4], "ric": "AAA", "shares": 10},
            {"date": portfolio_dates[5], "ric": "AAA", "shares": 10},
            {"date": portfolio_dates[0], "ric": "BBB", "shares": 10},
            {"date": portfolio_dates[1], "ric": "BBB", "shares": 10},
            {"date": portfolio_dates[2], "ric": "BBB", "shares": 10},
            {"date": portfolio_dates[3], "ric": "BBB", "shares": 10},
            {"date": portfolio_dates[4], "ric": "BBB", "shares": 10},
            {"date": portfolio_dates[5], "ric": "BBB", "shares": 10},
        ]
    )

    stats = compute_holding_presence_stats(holdings, portfolio_dates)
    assert list(stats["ric"]) == ["BBB", "AAA"]
    assert int(stats.iloc[0]["longest_consecutive_days"]) == 6
    assert int(stats.iloc[1]["longest_consecutive_days"]) == 3


def test_compute_summary_from_data_key_outputs() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    portfolio_daily = pd.DataFrame(
        [
            {
                "date": dates[0].date().isoformat(),
                "quarter": "2024Q1",
                "portfolio_market_value": 1000.0,
                "portfolio_cash_balance": 0.0,
                "portfolio_total_value": 1000.0,
                "portfolio_dividend_cash_daily": 0.0,
                "rebalance_flag": 1,
            },
            {
                "date": dates[1].date().isoformat(),
                "quarter": "2024Q1",
                "portfolio_market_value": 1010.0,
                "portfolio_cash_balance": 5.0,
                "portfolio_total_value": 1015.0,
                "portfolio_dividend_cash_daily": 5.0,
                "rebalance_flag": 0,
            },
            {
                "date": dates[2].date().isoformat(),
                "quarter": "2024Q1",
                "portfolio_market_value": 1020.0,
                "portfolio_cash_balance": 5.0,
                "portfolio_total_value": 1025.0,
                "portfolio_dividend_cash_daily": 0.0,
                "rebalance_flag": 0,
            },
            {
                "date": dates[3].date().isoformat(),
                "quarter": "2024Q1",
                "portfolio_market_value": 1030.0,
                "portfolio_cash_balance": 10.0,
                "portfolio_total_value": 1040.0,
                "portfolio_dividend_cash_daily": 5.0,
                "rebalance_flag": 0,
            },
        ]
    )

    holdings_daily = pd.DataFrame(
        [
            {"date": dates[0], "ric": "AAA", "shares": 5.0, "dividend_cash_daily": 0.0, "market_value": 500.0, "close": 100.0},
            {"date": dates[1], "ric": "AAA", "shares": 5.0, "dividend_cash_daily": 1.0, "market_value": 505.0, "close": 101.0},
            {"date": dates[2], "ric": "AAA", "shares": 5.0, "dividend_cash_daily": 0.0, "market_value": 510.0, "close": 102.0},
            {"date": dates[3], "ric": "AAA", "shares": 5.0, "dividend_cash_daily": 1.0, "market_value": 515.0, "close": 103.0},
            {"date": dates[0], "ric": "BBB", "shares": 5.0, "dividend_cash_daily": 0.0, "market_value": 500.0, "close": 100.0},
            {"date": dates[1], "ric": "BBB", "shares": 5.0, "dividend_cash_daily": 4.0, "market_value": 505.0, "close": 101.0},
            {"date": dates[2], "ric": "BBB", "shares": 0.0, "dividend_cash_daily": 0.0, "market_value": 0.0, "close": 102.0},
            {"date": dates[3], "ric": "BBB", "shares": 0.0, "dividend_cash_daily": 0.0, "market_value": 0.0, "close": 103.0},
        ]
    )

    trades = pd.DataFrame(
        [
            {
                "date": dates[0],
                "quarter": "2024Q1",
                "ric": "AAA",
                "trade_shares": 5.0,
                "trade_value": 500.0,
                "commission_cost": 1.0,
                "slippage_cost": 0.5,
                "spread_cost": 0.5,
                "total_transaction_cost": 2.0,
            },
            {
                "date": dates[0],
                "quarter": "2024Q1",
                "ric": "BBB",
                "trade_shares": 5.0,
                "trade_value": 500.0,
                "commission_cost": 1.0,
                "slippage_cost": 0.5,
                "spread_cost": 0.5,
                "total_transaction_cost": 2.0,
            },
            {
                "date": dates[2],
                "quarter": "2024Q1",
                "ric": "BBB",
                "trade_shares": -5.0,
                "trade_value": -510.0,
                "commission_cost": 1.0,
                "slippage_cost": 0.5,
                "spread_cost": 0.5,
                "total_transaction_cost": 2.0,
            },
        ]
    )

    target_weights = pd.DataFrame(
        [
            {"quarter": "2024Q1", "ric": "AAA", "weight": 0.6, "rank_in_portfolio": 1},
            {"quarter": "2024Q1", "ric": "BBB", "weight": 0.4, "rank_in_portfolio": 2},
        ]
    )
    quarter_scores = pd.DataFrame(
        [
            {"quarter": "2024Q1", "ric": "AAA", "score": 0.02, "rank_score": 2},
            {"quarter": "2024Q1", "ric": "BBB", "score": 0.03, "rank_score": 1},
        ]
    )

    benchmark_close = pd.DataFrame(
        {
            "Date": dates,
            "RIC": [".SPX"] * len(dates),
            "CLOSE": [100.0, 101.0, 102.0, 103.0],
        }
    )

    metadata = {
        "created_at_utc": "2026-03-06T00:00:00+00:00",
        "start_date": "2024-01-02",
        "end_date": "2024-01-05",
        "config_json": {
            "portfolio": {
                "initial_capital": 1000.0,
                "risk_free_rate": 0.0,
                "reinvest_dividends": False,
                "transaction_costs": {"enabled": True},
            },
            "strategy": {
                "mode": "dynamic_100_25",
                "candidate_count": 100,
                "portfolio_size": 25,
                "rebalance_interval_quarters": 2,
                "selection_policy": {
                    "name": "replace_bottom_n",
                    "max_replacements_per_quarter": 10,
                    "rank_metric": "quarter_dividend_yield_score",
                },
                "experiment_group": "alt_reallocations",
            },
        },
    }

    summary = compute_summary_from_data(
        run_id="run_test",
        run_metadata=metadata,
        portfolio_daily=portfolio_daily,
        holdings_daily=holdings_daily,
        trades=trades,
        target_weights=target_weights,
        quarter_scores=quarter_scores,
        candidate_universe=pd.DataFrame(),
        benchmark_close=benchmark_close,
        benchmark_label="S&P 500 (.SPX)",
        sp500_close=benchmark_close.copy(),
        sp500_label="S&P 500 (.SPX)",
        russell_1000_close=benchmark_close.rename(columns={"RIC": "RIC"}).copy(),
        russell_1000_label="Russell 1000 (.RUI)",
    )

    assert summary["run_id"] == "run_test"
    assert math.isclose(summary["portfolio_metrics"]["total_return"], 0.04, rel_tol=1e-12)
    assert math.isclose(summary["dividends"]["total_dividend_cash"], 10.0, rel_tol=1e-12)
    assert summary["dividends"]["highest_dividend_paying_stock"]["ric"] == "BBB"
    assert summary["holdings"]["longest_staying_stock"]["ric"] == "AAA"
    assert summary["trading_activity"]["number_of_trades"] == 3
    assert summary["benchmark_comparison"]["enabled"] is True
    assert summary["sp500_comparison"]["enabled"] is True
    assert summary["russell_1000_comparison"]["enabled"] is True
    assert "correlation_with_portfolio_daily_returns" in summary["russell_1000_comparison"]
    assert "annualized_excess_return" in summary["sp500_comparison"]
    assert "monthly_up_down_counts" in summary["comparative_period_stats"]
    assert "quarterly_vs_sp500" in summary["comparative_period_stats"]
    assert "upside_capture_ratio" in summary["comparative_period_stats"]["quarterly_vs_sp500"]
    assert "downside_capture_ratio" in summary["comparative_period_stats"]["quarterly_vs_sp500"]
    assert "quarterly_total_return_comparison" in summary["detailed_breakdowns"]
    assert "portfolio_quarterly_performance" in summary["detailed_breakdowns"]
    assert "enabled" in summary["volatility_models"]
    assert math.isclose(summary["transaction_costs"]["total_transaction_cost"], 6.0, rel_tol=1e-12)
    assert math.isclose(summary["transaction_costs"]["avg_transaction_cost_per_trade"], 2.0, rel_tol=1e-12)
    assert math.isclose(
        summary["transaction_costs"]["avg_transaction_cost_bps_of_notional"],
        10000.0 * 6.0 / (500.0 + 500.0 + 510.0),
        rel_tol=1e-12,
    )
    assert summary["transaction_costs"]["enabled_in_config"] is True
    assert "trades_per_quarter" in summary["detailed_breakdowns"]
    assert summary["hyperparameters"]["portfolio_size"] == 25
    assert summary["hyperparameters"]["rebalance_interval_quarters"] == 2
    assert summary["hyperparameters"]["allocation_strategy"] == "yield_proportional"
    assert "objective_metrics" in summary
    assert "constraint_metrics" in summary
    assert "diagnostic_metrics" in summary
    assert summary["objective_metrics"]["cagr"] == summary["portfolio_metrics"]["cagr"]
    assert summary["constraint_metrics"]["max_drawdown"] == summary["portfolio_metrics"]["max_drawdown"]
    assert summary["diagnostic_metrics"]["positive_days"] == summary["portfolio_metrics"]["positive_days"]
    assert summary["dsr_readiness"]["ready_for_deflated_sharpe"] is False
    assert summary["strategy"]["selection_policy_name"] == "replace_bottom_n"
    assert summary["strategy"]["max_replacements_per_quarter"] == 10
    assert summary["strategy"]["rebalance_interval_quarters"] == 2
    assert summary["strategy"]["allocation_strategy"] == "yield_proportional"
    assert summary["strategy"]["experiment_group"] == "alt_reallocations"


def test_quadrant_period_stats_counts() -> None:
    paired_returns = pd.DataFrame(
        {
            "strategy_return": [0.10, 0.05, -0.03, -0.02, 0.01],
            "benchmark_return": [0.08, -0.04, 0.02, -0.01, 0.0],
        },
        index=["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"],
    )

    stats = _quadrant_period_stats(paired_returns)

    assert stats["enabled"] is True
    assert stats["total_periods"] == 5
    assert stats["both_up_periods"] == 1
    assert stats["both_down_periods"] == 1
    assert stats["strategy_up_benchmark_down_periods"] == 1
    assert stats["strategy_down_benchmark_up_periods"] == 1
    assert stats["on_axis_periods"] == 1
    assert stats["same_direction_periods"] == 2
    assert stats["opposite_direction_periods"] == 2
    assert stats["beat_benchmark_periods"] == 3
    assert math.isclose(stats["strategy_up_when_benchmark_up_rate"], 0.5, rel_tol=1e-12)
    assert math.isclose(stats["strategy_down_when_benchmark_down_rate"], 0.5, rel_tol=1e-12)
    assert math.isclose(stats["beat_benchmark_rate"], 0.6, rel_tol=1e-12)
    assert stats["correlation"] is not None
    assert math.isclose(stats["upside_capture_ratio"], 0.7, rel_tol=1e-12)
    assert math.isclose(stats["downside_capture_ratio"], -0.6, rel_tol=1e-12)


def test_fetch_benchmark_close_uses_persistent_cache(monkeypatch, tmp_path) -> None:
    calls = {"count": 0}

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def get_history(self, **kwargs):
            calls["count"] += 1
            return pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "TRDPRC_1": [100.0, 101.0],
                }
            )

    monkeypatch.setattr(dynamic_results_module, "RefinitivClient", DummyClient)
    cache_db = tmp_path / "benchmark_cache.sqlite"

    first = _fetch_benchmark_close(
        ric=".SPX",
        start_date="2024-01-02",
        end_date="2024-01-03",
        cache_db_path=cache_db,
    )
    second = _fetch_benchmark_close(
        ric=".SPX",
        start_date="2024-01-02",
        end_date="2024-01-03",
        cache_db_path=cache_db,
    )

    assert calls["count"] == 1
    assert len(first) == 2
    assert first.equals(second)


def test_fit_arch_garch_models_optional_dependency() -> None:
    idx = pd.bdate_range("2024-01-02", periods=260)
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(loc=0.0002, scale=0.01, size=len(idx)), index=idx)

    summary, series = fit_arch_garch_models(returns)
    assert "enabled" in summary
    assert "warnings" in summary
    if summary["enabled"]:
        assert not series.empty
        assert ("arch_1_0_cond_vol" in series.columns) or ("garch_1_1_cond_vol" in series.columns)
