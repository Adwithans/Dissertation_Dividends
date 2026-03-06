from __future__ import annotations

import pandas as pd

from .split_math import build_split_multiplier


def simulate_asset(
    df: pd.DataFrame,
    *,
    initial_investment: float,
    price_col: str = "CLOSE",
    dividend_col: str = "Dividend",
    split_col: str = "SplitFactor",
    cum_factor_col: str = "cum_factor",
    use_cum_factor: bool = True,
    auto_align_splits: bool = True,
    reinvest_dividends: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    out = df.copy().sort_index()

    if price_col not in out.columns:
        raise ValueError(f"Missing required price column '{price_col}'.")

    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.loc[out[price_col].notna() & (out[price_col] > 0)].copy()
    if out.empty:
        raise ValueError("No valid rows after price cleaning.")

    out[dividend_col] = pd.to_numeric(out.get(dividend_col, 0.0), errors="coerce").fillna(0.0)

    out["Split_Multiplier"] = build_split_multiplier(
        out,
        price_col=price_col,
        split_col=split_col,
        cum_factor_col=cum_factor_col,
        use_cum_factor=use_cum_factor,
        auto_align=auto_align_splits,
    )

    first_price = float(out[price_col].iloc[0])
    shares = initial_investment / first_price
    cash = 0.0
    cum_div = 0.0

    shares_list: list[float] = []
    cash_list: list[float] = []
    div_daily_list: list[float] = []
    cum_div_list: list[float] = []
    market_value_list: list[float] = []
    total_list: list[float] = []

    for _, row in out.iterrows():
        split_mult = float(row["Split_Multiplier"])
        price = float(row[price_col])
        div = float(row[dividend_col])

        if split_mult != 1.0:
            shares *= split_mult

        div_cash = 0.0
        if div > 0:
            div_cash = shares * div
            cum_div += div_cash
            if reinvest_dividends:
                shares += div_cash / price
            else:
                cash += div_cash

        market = shares * price
        total = market + cash

        shares_list.append(shares)
        cash_list.append(cash)
        div_daily_list.append(div_cash)
        cum_div_list.append(cum_div)
        market_value_list.append(market)
        total_list.append(total)

    out["Shares_Held"] = shares_list
    out["Cash_Balance"] = cash_list
    out["Dividend_Cash_Daily"] = div_daily_list
    out["Dividend_Income"] = cum_div_list
    out["Market_Value"] = market_value_list
    out["Total_Value"] = total_list
    out["Target_Weight"] = 1.0
    out["Weight_EOD"] = 1.0
    out["Rebalance_Trade_Shares"] = 0.0
    out["Rebalance_Trade_Value"] = 0.0
    return out
