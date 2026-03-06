from __future__ import annotations

import pandas as pd

from src.dividend_portfolio.analytics.quarterly import compute_quarterly_stock_metrics


def test_quarterly_dividend_return_metric() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-02-01", "2024-03-28"])
    asset_df = pd.DataFrame(
        {
            "Market_Value": [1000.0, 1020.0, 1010.0],
            "Dividend_Cash_Daily": [0.0, 10.0, 0.0],
        },
        index=idx,
    )
    portfolio_df = pd.DataFrame({"Portfolio_Total_Value": [2000.0, 2050.0, 2040.0]}, index=idx)

    out = compute_quarterly_stock_metrics({"AAPL.O": asset_df}, portfolio_df)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["RIC"] == "AAPL.O"
    assert abs(row["Quarter_Dividend_Return"] - 0.01) < 1e-12
