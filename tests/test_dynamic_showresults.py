from __future__ import annotations

import math

import pandas as pd

from src.dividend_portfolio.reporting.dynamic_results import (
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
            {"date": dates[0], "quarter": "2024Q1", "ric": "AAA", "trade_shares": 5.0, "trade_value": 500.0},
            {"date": dates[0], "quarter": "2024Q1", "ric": "BBB", "trade_shares": 5.0, "trade_value": 500.0},
            {"date": dates[2], "quarter": "2024Q1", "ric": "BBB", "trade_shares": -5.0, "trade_value": -510.0},
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
            }
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
    )

    assert summary["run_id"] == "run_test"
    assert math.isclose(summary["portfolio_metrics"]["total_return"], 0.04, rel_tol=1e-12)
    assert math.isclose(summary["dividends"]["total_dividend_cash"], 10.0, rel_tol=1e-12)
    assert summary["dividends"]["highest_dividend_paying_stock"]["ric"] == "BBB"
    assert summary["holdings"]["longest_staying_stock"]["ric"] == "AAA"
    assert summary["trading_activity"]["number_of_trades"] == 3
    assert summary["benchmark_comparison"]["enabled"] is True
    assert "trades_per_quarter" in summary["trading_activity"]
