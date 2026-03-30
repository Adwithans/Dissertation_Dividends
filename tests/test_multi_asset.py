from __future__ import annotations

from datetime import date

import pandas as pd

from src.dividend_portfolio.models import (
    AssetConfig,
    PortfolioConfig,
    QuarterlyMetricsConfig,
    RebalanceConfig,
    TransactionCostsConfig,
)
from src.dividend_portfolio.sim.multi_asset import simulate_portfolio


def _history(prices: list[float], dates: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "CLOSE": prices,
            "Dividend": [0.0] * len(prices),
            "SplitFactor": [1.0] * len(prices),
            "cum_factor": [1.0] * len(prices),
        },
        index=idx,
    )


def test_multi_asset_aggregation_identity() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    histories = {
        "AAPL.O": _history([100.0, 102.0, 104.0], dates),
        "MSFT.O": _history([50.0, 51.0, 52.0], dates),
    }

    cfg = PortfolioConfig(
        base_currency="USD",
        initial_capital=1000.0,
        start_date=date(2024, 1, 2),
        end_date=None,
        reinvest_dividends=False,
        auto_align_splits=True,
        use_cum_factor=True,
        risk_free_rate=0.0,
        rebalancing=RebalanceConfig(
            enabled=False,
            frequency="quarterly",
            trigger="first_trading_day_after_quarter_end",
            drift_tolerance=0.02,
        ),
        quarterly_metrics=QuarterlyMetricsConfig(
            enabled=True,
            dividend_return_basis="quarter_start_market_value",
        ),
        assets=[AssetConfig("AAPL.O", 0.5), AssetConfig("MSFT.O", 0.5)],
    )

    sim = simulate_portfolio(histories, cfg)
    p = sim.portfolio_df

    summed = sim.asset_results["AAPL.O"]["Market_Value"] + sim.asset_results["MSFT.O"]["Market_Value"]
    assert (summed.round(8) == p["Portfolio_Market_Value"].round(8)).all()


def test_multi_asset_transaction_costs_with_fallback_spread() -> None:
    dates = ["2024-03-29", "2024-04-01", "2024-04-02"]
    histories = {
        "AAPL.O": _history([100.0, 110.0, 111.0], dates),
        "MSFT.O": _history([100.0, 90.0, 89.0], dates),
    }

    cfg = PortfolioConfig(
        base_currency="USD",
        initial_capital=1000.0,
        start_date=date(2024, 3, 29),
        end_date=None,
        reinvest_dividends=False,
        auto_align_splits=True,
        use_cum_factor=True,
        risk_free_rate=0.0,
        rebalancing=RebalanceConfig(
            enabled=True,
            frequency="quarterly",
            trigger="first_trading_day_after_quarter_end",
            drift_tolerance=0.0,
        ),
        quarterly_metrics=QuarterlyMetricsConfig(
            enabled=True,
            dividend_return_basis="quarter_start_market_value",
        ),
        assets=[AssetConfig("AAPL.O", 0.5), AssetConfig("MSFT.O", 0.5)],
        transaction_costs=TransactionCostsConfig(
            enabled=True,
            commission_bps=0.0,
            commission_min_usd=0.0,
            slippage_bps_per_side=0.0,
            fallback_full_spread_bps=10.0,
            use_bid_ask_when_available=True,
            sizing_rule="cost_aware_scaling",
        ),
    )

    sim = simulate_portfolio(histories, cfg)
    p = sim.portfolio_df
    assert "Portfolio_Transaction_Cost_Daily" in p.columns
    assert "Portfolio_Total_Value_Gross" in p.columns
    assert float(p["Portfolio_Transaction_Cost_Daily"].sum()) > 0.0
    assert (p["Portfolio_Total_Value_Gross"] >= p["Portfolio_Total_Value"]).all()
