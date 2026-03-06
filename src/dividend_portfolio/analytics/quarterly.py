from __future__ import annotations

import pandas as pd


def compute_quarterly_stock_metrics(
    asset_results: dict[str, pd.DataFrame],
    portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for ric, df in asset_results.items():
        if df.empty:
            continue

        grouped = df.groupby(df.index.to_period("Q"))
        for quarter, qdf in grouped:
            qdf = qdf.sort_index()
            start_date = qdf.index[0]
            end_date = qdf.index[-1]

            quarter_start_value = float(qdf["Market_Value"].iloc[0])
            quarter_end_value = float(qdf["Market_Value"].iloc[-1])
            quarter_dividend_cash = float(qdf["Dividend_Cash_Daily"].sum())

            if quarter_start_value > 0:
                quarter_total_return = (quarter_end_value - quarter_start_value) / quarter_start_value
                quarter_dividend_return = quarter_dividend_cash / quarter_start_value
            else:
                quarter_total_return = float("nan")
                quarter_dividend_return = float("nan")

            quarter_price_pnl = (quarter_end_value - quarter_start_value) - quarter_dividend_cash

            portfolio_start_value = float(portfolio_df.loc[start_date, "Portfolio_Total_Value"])
            if portfolio_start_value > 0:
                div_contribution = quarter_dividend_cash / portfolio_start_value
            else:
                div_contribution = float("nan")

            rows.append(
                {
                    "RIC": ric,
                    "Quarter": str(quarter),
                    "Quarter_Start_Date": start_date.date().isoformat(),
                    "Quarter_End_Date": end_date.date().isoformat(),
                    "Quarter_Start_Value": quarter_start_value,
                    "Quarter_End_Value": quarter_end_value,
                    "Quarter_Dividend_Cash": quarter_dividend_cash,
                    "Quarter_Dividend_Return": quarter_dividend_return,
                    "Quarter_Total_Return": quarter_total_return,
                    "Quarter_Price_PnL": quarter_price_pnl,
                    "Quarter_Dividend_Contribution_to_Portfolio": div_contribution,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["Quarter", "RIC"]).reset_index(drop=True)
