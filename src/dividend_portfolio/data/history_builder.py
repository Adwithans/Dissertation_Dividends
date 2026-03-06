from __future__ import annotations

import pandas as pd

from .fetch_events import fetch_dividend_events, fetch_split_events
from .fetch_prices import fetch_prices
from .refinitiv_client import RefinitivClient


def build_history_for_ticker(
    client: RefinitivClient,
    ticker: str,
    start_dt: str,
    end_dt: str,
) -> pd.DataFrame | None:
    prices = fetch_prices(client, ticker, start_dt, end_dt)
    if prices is None or prices.empty:
        return None

    div_series, _ = fetch_dividend_events(client, ticker, start_dt, end_dt)
    split_series, _ = fetch_split_events(client, ticker, start_dt, end_dt)

    data = prices.join(div_series, how="left")
    data = data.join(split_series, how="left")

    if "CLOSE" not in data.columns:
        close_col = next((c for c in data.columns if c.upper() == "CLOSE"), None)
        if close_col is None:
            numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
            if not numeric_cols:
                return None
            close_col = numeric_cols[0]
        data = data.rename(columns={close_col: "CLOSE"})

    data["CLOSE"] = pd.to_numeric(data["CLOSE"], errors="coerce")
    data = data.loc[data["CLOSE"].notna() & (data["CLOSE"] > 0)].copy()
    if data.empty:
        return None

    data["Dividend"] = pd.to_numeric(data.get("Dividend", 0.0), errors="coerce").fillna(0.0)
    data["SplitFactor"] = pd.to_numeric(data.get("SplitFactor", 1.0), errors="coerce").fillna(1.0)

    data = data.sort_index()
    data["CumulativeDividend"] = data["Dividend"].cumsum()

    sf = data["SplitFactor"].fillna(1.0)
    sf = sf.where(sf > 0, 1.0)
    split_mult = sf.where(sf >= 1.0, 1.0 / sf)
    data["cum_factor"] = split_mult.cumprod()

    return data[["CLOSE", "Dividend", "SplitFactor", "CumulativeDividend", "cum_factor"]]


def build_histories_for_tickers(
    client: RefinitivClient,
    tickers: list[str],
    start_dt: str,
    end_dt: str,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = build_history_for_ticker(client, ticker, start_dt, end_dt)
        if df is not None and not df.empty:
            out[ticker] = df
    return out
