from __future__ import annotations

import pandas as pd

from src.dividend_portfolio.analytics.attribution import compute_asset_attribution
from src.dividend_portfolio.analytics.metrics import compute_portfolio_metrics


def test_compute_portfolio_metrics_exposes_dividend_totals() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    portfolio_df = pd.DataFrame(
        {
            "Portfolio_Total_Value": [1000.0, 1100.0],
            "Portfolio_Daily_Return": [0.0, 0.1],
            "Portfolio_Dividend_Income": [0.0, 25.0],
        },
        index=idx,
    )

    metrics = compute_portfolio_metrics(portfolio_df, initial_capital=1000.0, risk_free_rate=0.0)

    assert metrics["total_dividend_value_gained_usd"] == 25.0
    assert metrics["total_dividend_value_gained_pct_of_initial"] == 0.025
    assert metrics["net_total_gain_usd"] == 100.0
    assert abs(metrics["dividend_share_of_total_gain"] - 0.25) < 1e-12


def test_compute_asset_attribution_exposes_per_stock_dividend_totals() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    asset_results = {
        "AAPL.O": pd.DataFrame(
            {
                "Market_Value": [500.0, 530.0],
                "Dividend_Cash_Daily": [0.0, 10.0],
                "Rebalance_Trade_Value": [0.0, 0.0],
            },
            index=idx,
        ),
        "MSFT.O": pd.DataFrame(
            {
                "Market_Value": [500.0, 570.0],
                "Dividend_Cash_Daily": [0.0, 30.0],
                "Rebalance_Trade_Value": [0.0, 0.0],
            },
            index=idx,
        ),
    }

    out = compute_asset_attribution(asset_results, initial_capital=1000.0)
    by_ric = out.set_index("RIC")

    assert by_ric.loc["AAPL.O", "Total_Dividend_Payments"] == 10.0
    assert by_ric.loc["MSFT.O", "Total_Dividend_Payments"] == 30.0
    assert abs(by_ric.loc["AAPL.O", "Dividend_Share_of_Portfolio_Dividends"] - 0.25) < 1e-12
    assert abs(by_ric.loc["MSFT.O", "Dividend_Share_of_Portfolio_Dividends"] - 0.75) < 1e-12
