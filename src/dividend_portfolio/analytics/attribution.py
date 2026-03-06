from __future__ import annotations

import pandas as pd


def compute_asset_attribution(
    asset_results: dict[str, pd.DataFrame],
    *,
    initial_capital: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    portfolio_dividend_total = 0.0
    for df in asset_results.values():
        if df.empty:
            continue
        portfolio_dividend_total += float(df["Dividend_Cash_Daily"].sum())

    for ric, df in asset_results.items():
        if df.empty:
            continue

        start_market = float(df["Market_Value"].iloc[0])
        end_market = float(df["Market_Value"].iloc[-1])
        dividend_cash = float(df["Dividend_Cash_Daily"].sum())
        net_rebalance_flow = float(df["Rebalance_Trade_Value"].sum())

        price_pnl_ex_flows = end_market - start_market - net_rebalance_flow
        total_pnl_ex_flows = price_pnl_ex_flows + dividend_cash

        rows.append(
            {
                "RIC": ric,
                "Start_Market_Value": start_market,
                "End_Market_Value": end_market,
                "Dividend_Cash": dividend_cash,
                "Total_Dividend_Payments": dividend_cash,
                "Dividend_Share_of_Portfolio_Dividends": (
                    dividend_cash / portfolio_dividend_total if portfolio_dividend_total > 0 else float("nan")
                ),
                "Net_Rebalance_Flow": net_rebalance_flow,
                "Price_PnL_ExFlows": price_pnl_ex_flows,
                "Total_PnL_ExFlows": total_pnl_ex_flows,
                "Contribution_to_Portfolio_Return": total_pnl_ex_flows / initial_capital,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("Contribution_to_Portfolio_Return", ascending=False).reset_index(drop=True)
