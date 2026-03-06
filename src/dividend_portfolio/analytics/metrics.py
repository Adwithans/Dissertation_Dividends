from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_portfolio_metrics(
    portfolio_df: pd.DataFrame,
    *,
    initial_capital: float,
    risk_free_rate: float,
) -> dict[str, float]:
    if portfolio_df.empty:
        raise ValueError("portfolio_df is empty")

    total = pd.to_numeric(portfolio_df["Portfolio_Total_Value"], errors="coerce")
    returns = pd.to_numeric(portfolio_df["Portfolio_Daily_Return"], errors="coerce").fillna(0.0)

    start_value = float(total.iloc[0])
    end_value = float(total.iloc[-1])
    total_return = end_value / initial_capital - 1.0

    day_count = max((portfolio_df.index[-1] - portfolio_df.index[0]).days, 1)
    cagr = (end_value / initial_capital) ** (365.25 / day_count) - 1.0

    daily_std = float(returns.std(ddof=0))
    annualized_volatility = daily_std * math.sqrt(252)

    mean_daily = float(returns.mean())
    rf_daily = risk_free_rate / 252.0
    if daily_std > 0:
        sharpe = ((mean_daily - rf_daily) / daily_std) * math.sqrt(252)
    else:
        sharpe = float("nan")

    running_max = total.cummax()
    drawdown = total / running_max - 1.0
    max_drawdown = float(drawdown.min())

    cumulative_div = float(portfolio_df["Portfolio_Dividend_Income"].iloc[-1])
    dividend_yield_on_cost = cumulative_div / initial_capital
    total_gain = end_value - initial_capital
    dividend_share_of_total_gain = cumulative_div / total_gain if total_gain != 0 else float("nan")

    return {
        "start_value": start_value,
        "end_value": end_value,
        "net_total_gain_usd": float(total_gain),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "cumulative_dividend_income": cumulative_div,
        "total_dividend_value_gained_usd": cumulative_div,
        "total_dividend_value_gained_pct_of_initial": float(dividend_yield_on_cost),
        "dividend_share_of_total_gain": float(dividend_share_of_total_gain),
        "dividend_yield_on_cost": float(dividend_yield_on_cost),
    }
