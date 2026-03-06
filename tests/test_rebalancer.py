from __future__ import annotations

import pandas as pd

from src.dividend_portfolio.sim.rebalancer import (
    apply_rebalance,
    build_rebalance_dates,
    compute_drifts,
    should_rebalance,
)


def test_quarterly_rebalance_dates() -> None:
    dates = pd.to_datetime(["2024-03-29", "2024-04-01", "2024-06-28", "2024-07-01"])
    rebal_dates = build_rebalance_dates(
        pd.DatetimeIndex(dates),
        frequency="quarterly",
        trigger="first_trading_day_after_quarter_end",
    )
    assert pd.Timestamp("2024-04-01") in rebal_dates
    assert pd.Timestamp("2024-07-01") in rebal_dates


def test_apply_rebalance_moves_to_target_weights() -> None:
    shares = {"A": 1.0, "B": 3.0}
    prices = {"A": 100.0, "B": 100.0}
    target = {"A": 0.5, "B": 0.5}

    market = {k: shares[k] * prices[k] for k in shares}
    drifts = compute_drifts(market, sum(market.values()), target)

    assert should_rebalance(
        date=pd.Timestamp("2024-04-01"),
        rebalance_dates={pd.Timestamp("2024-04-01")},
        drift_by_ric=drifts,
        drift_tolerance=0.01,
    )

    new_shares, cash, _, _, _ = apply_rebalance(
        prices_by_ric=prices,
        shares_by_ric=shares,
        portfolio_cash=0.0,
        target_weights=target,
    )

    assert cash == 0.0
    assert abs(new_shares["A"] - 2.0) < 1e-12
    assert abs(new_shares["B"] - 2.0) < 1e-12
