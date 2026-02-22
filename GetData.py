from datetime import date, timedelta
from pathlib import Path

import eikon as ek
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
import refinitiv.data as rd

load_dotenv()
ek.set_app_key(os.environ["EIKON_API_KEY"])

# Largest dividend-paying stocks by market cap (example set).
ric = [
    "NVDA.O",
    "AAPL.O",
    "MSFT.O",
    "JNJ.N",
    "PG.N",
    "SHEL.L",
    "V.N",
    "JPM.N",
    "XOM.N",
]

end_date = date.today().isoformat()
start_date = (date.today() - timedelta(days=365 * 15)).isoformat()

price_fields = ["OPEN", "HIGH", "LOW", "CLOSE"]
use_lseg_unadjusted = True


def _rd_df(data):
    # rd.get_data sometimes returns (df, metadata)
    if isinstance(data, tuple):
        return data[0]
    return data

def fetch_dividend_events(ticker, start_dt, end_dt):
    fields = ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivCurr"]
    params = {"SDate": start_dt, "EDate": end_dt, "DateType": "ED"}
    try:
        df, _ = ek.get_data(ticker, fields, params)
    except Exception:
        return pd.Series(dtype="float64", name="Dividend"), None

    if df is None or df.empty:
        return pd.Series(dtype="float64", name="Dividend"), None

    date_col = next((c for c in df.columns if "ex" in c.lower() and "date" in c.lower()), None)
    amount_col = next(
        (c for c in df.columns if "gross" in c.lower() or ("div" in c.lower() and "amount" in c.lower())),
        None,
    )
    if date_col is None or amount_col is None:
        return pd.Series(dtype="float64", name="Dividend"), "no_exdate_or_amount_columns"

    events = df[[date_col, amount_col]].dropna()
    events[date_col] = pd.to_datetime(events[date_col], errors="coerce")
    events[amount_col] = pd.to_numeric(events[amount_col], errors="coerce")
    events = events.dropna(subset=[date_col, amount_col])
    if events.empty:
        return pd.Series(dtype="float64", name="Dividend"), "no_events"

    series = events.groupby(date_col)[amount_col].sum()
    series.name = "Dividend"
    return series, f"{date_col}/{amount_col}"

def fetch_split_events(ticker, start_dt, end_dt):
    params = {"CAEventType": "SSP", "SDate": start_dt, "EDate": end_dt}
    fields = [
        "TR.CAExDate",
        "TR.CAEffectiveDate",
        "TR.CACorpActDate",
        "TR.CAAdjustmentFactor",
    ]
    try:
        df = _rd_df(rd.get_data(ticker, fields, params))
    except Exception:
        return pd.Series(dtype="float64", name="SplitFactor"), None

    if df is None or df.empty:
        return pd.Series(dtype="float64", name="SplitFactor"), "no_events"

    # Prefer ExDate if available, then EffectiveDate, then CorpActDate.
    date_col = next((c for c in df.columns if "exdate" in c.lower()), None)
    if date_col is None:
        date_col = next((c for c in df.columns if "effect" in c.lower() and "date" in c.lower()), None)
    if date_col is None:
        date_col = next((c for c in df.columns if "corpact" in c.lower() and "date" in c.lower()), None)
    if date_col is None:
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
    adj_col = next((c for c in df.columns if "adjustment" in c.lower() and "factor" in c.lower()), None)
    if date_col is None:
        return pd.Series(dtype="float64", name="SplitFactor"), "no_date_column"
    if adj_col is None:
        return pd.Series(dtype="float64", name="SplitFactor"), "no_adjustment_factor"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[adj_col] = pd.to_numeric(df[adj_col], errors="coerce")
    df["SplitFactor"] = df[adj_col]

    df = df.dropna(subset=[date_col, "SplitFactor"])
    series = df.groupby(date_col)["SplitFactor"].prod()
    series.name = "SplitFactor"
    return series, "Success"

def fetch_prices(ticker, start_dt, end_dt):
    # Prefer Refinitiv Data Platform unadjusted prices (same method as test.py).
    def _normalize(df):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

        if isinstance(df.columns, pd.MultiIndex):
            for level in range(df.columns.nlevels):
                level_vals = set(df.columns.get_level_values(level))
                if any(field in level_vals for field in price_fields):
                    df.columns = df.columns.get_level_values(level)
                    break
        return df

    # 1) Try OHLC unadjusted.
    try:
        prices = rd.get_history(
            universe=[ticker],
            fields=price_fields,
            interval="daily",
            start=start_dt,
            end=end_dt,
            adjustments="unadjusted",
        )
    except Exception:
        prices = None

    if prices is not None and not prices.empty:
        prices = _normalize(prices)
        available = [c for c in price_fields if c in prices.columns]
        if available:
            return prices[available]
        return prices

    # 2) Fallback to TRDPRC_1 (unadjusted) and map to CLOSE if OHLC is unavailable.
    try:
        prices = rd.get_history(
            universe=[ticker],
            fields=["TRDPRC_1"],
            interval="daily",
            start=start_dt,
            end=end_dt,
            adjustments="unadjusted",
        )
    except Exception:
        return None

    if prices is None or prices.empty:
        return None

    prices = _normalize(prices)
    # Rename the single series to CLOSE for downstream compatibility.
    if "TRDPRC_1" in prices.columns:
        prices = prices.rename(columns={"TRDPRC_1": "CLOSE"})
    return prices[["CLOSE"]] if "CLOSE" in prices.columns else prices

def build_history(ticker):
    prices = fetch_prices(ticker, start_date, end_date)
    if prices is None:
        print(f"Price fetch returned no data for {ticker}; skipping.")
        return None

    div_series, _ = fetch_dividend_events(ticker, start_date, end_date)
    split_series, _ = fetch_split_events(ticker, start_date, end_date)

    data = prices.join(div_series, how="left")
    data = data.join(split_series, how="left")

    data["Dividend"] = data["Dividend"].fillna(0)
    data["CumulativeDividend"] = data["Dividend"].cumsum()
    data["SplitFactor"] = data["SplitFactor"].fillna(1)
    
    data = data.sort_index()
    # Build a running *cumulative* split multiplier that starts at 1 and increases as splits occur.
    #
    # Refinitiv's "adjustment factor" for splits is often < 1 (e.g. 0.25 for a 4-for-1, 0.1 for a 10-for-1).
    # For a "running cumulative split" we want the multiplier (> 1), so invert factors in (0, 1).
    sf = pd.to_numeric(data["SplitFactor"], errors="coerce").fillna(1.0)
    sf = sf.where(sf > 0, 1.0)  # guard against zeros/negatives
    split_mult = sf.where(sf >= 1.0, 1.0 / sf)
    data["cum_factor"] = split_mult.cumprod()
    return data

def main():
    data_dir = Path("stock_data")
    data_dir.mkdir(exist_ok=True)

    rd.open_session()
    try:
        for ticker in ric:
            data = build_history(ticker)
            if data is None:
                continue
            filename = f"{ticker.replace('.', '_')}_history.csv"
            data_to_save = data.round(4)
            data_to_save.to_csv(data_dir / filename, float_format="%.4f")

    finally:
        rd.close_session()

if __name__ == "__main__":
    main()
