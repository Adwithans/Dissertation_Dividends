from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _period_label(ts: pd.Timestamp, frequency: str) -> str:
    if frequency == "quarterly":
        return str(ts.to_period("Q"))
    if frequency == "monthly":
        return str(ts.to_period("M"))
    if frequency == "yearly":
        return str(ts.to_period("Y"))
    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def build_rebalance_dates(
    dates: pd.DatetimeIndex,
    *,
    frequency: str,
    trigger: str,
) -> set[pd.Timestamp]:
    if len(dates) <= 1:
        return set()

    if trigger != "first_trading_day_after_quarter_end" and frequency == "quarterly":
        raise ValueError(
            "For quarterly mode, trigger must be 'first_trading_day_after_quarter_end'."
        )

    out: set[pd.Timestamp] = set()
    prev = dates[0]
    prev_label = _period_label(prev, frequency)

    for i in range(1, len(dates)):
        current = dates[i]
        current_label = _period_label(current, frequency)
        if current_label != prev_label:
            out.add(current)
        prev_label = current_label

    return out


def should_rebalance(
    *,
    date: pd.Timestamp,
    rebalance_dates: set[pd.Timestamp],
    drift_by_ric: dict[str, float],
    drift_tolerance: float,
) -> bool:
    if date not in rebalance_dates:
        return False
    if not drift_by_ric:
        return False
    return max(drift_by_ric.values()) >= drift_tolerance


def apply_rebalance(
    *,
    prices_by_ric: dict[str, float],
    shares_by_ric: dict[str, float],
    portfolio_cash: float,
    target_weights: dict[str, float],
) -> tuple[dict[str, float], float, dict[str, float], dict[str, float], float]:
    rics = list(shares_by_ric.keys())

    market_before = {ric: shares_by_ric[ric] * prices_by_ric[ric] for ric in rics}
    total_before = sum(market_before.values()) + portfolio_cash

    trade_value_by_ric: dict[str, float] = {}
    trade_shares_by_ric: dict[str, float] = {}

    for ric in rics:
        target_market = total_before * target_weights[ric]
        trade_value = target_market - market_before[ric]
        price = prices_by_ric[ric]
        trade_shares = 0.0 if price == 0 else trade_value / price

        shares_by_ric[ric] = shares_by_ric[ric] + trade_shares
        trade_value_by_ric[ric] = trade_value
        trade_shares_by_ric[ric] = trade_shares

    portfolio_cash = portfolio_cash - sum(trade_value_by_ric.values())
    if abs(portfolio_cash) < 1e-10:
        portfolio_cash = 0.0

    market_after = {ric: shares_by_ric[ric] * prices_by_ric[ric] for ric in rics}
    total_after = sum(market_after.values()) + portfolio_cash

    return shares_by_ric, portfolio_cash, trade_shares_by_ric, trade_value_by_ric, total_after


def compute_drifts(
    market_by_ric: dict[str, float],
    portfolio_total: float,
    target_weights: dict[str, float],
) -> dict[str, float]:
    drifts: dict[str, float] = {}
    if portfolio_total <= 0:
        for ric in market_by_ric:
            drifts[ric] = 0.0
        return drifts

    for ric, mv in market_by_ric.items():
        current_weight = mv / portfolio_total
        drifts[ric] = abs(current_weight - target_weights[ric])
    return drifts


def sorted_dates(values: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
    return sorted(values)
