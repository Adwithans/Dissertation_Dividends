from __future__ import annotations

import pandas as pd

from src.dividend_portfolio.sim.single_asset import simulate_asset


def test_single_asset_dividend_cash_path() -> None:
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    df = pd.DataFrame(
        {
            "CLOSE": [100.0, 100.0, 100.0],
            "Dividend": [0.0, 1.0, 0.0],
            "SplitFactor": [1.0, 1.0, 1.0],
            "cum_factor": [1.0, 1.0, 1.0],
        },
        index=idx,
    )

    out = simulate_asset(df, initial_investment=1000.0, reinvest_dividends=False)

    assert out["Shares_Held"].iloc[0] == 10.0
    assert out["Dividend_Cash_Daily"].iloc[1] == 10.0
    assert out["Cash_Balance"].iloc[-1] == 10.0
    assert out["Total_Value"].iloc[-1] == 1010.0
