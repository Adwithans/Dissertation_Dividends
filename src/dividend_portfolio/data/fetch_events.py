from __future__ import annotations

import pandas as pd

from .refinitiv_client import RefinitivClient


def fetch_dividend_events(
    client: RefinitivClient, ticker: str, start_dt: str, end_dt: str
) -> tuple[pd.Series, str | None]:
    fields = ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivCurr"]
    params = {"SDate": start_dt, "EDate": end_dt, "DateType": "ED"}

    try:
        df, _ = client.get_eikon_data(ticker, fields, params)
    except Exception:  # noqa: BLE001
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

    events = df[[date_col, amount_col]].copy().dropna()
    events[date_col] = pd.to_datetime(events[date_col], errors="coerce")
    events[amount_col] = pd.to_numeric(events[amount_col], errors="coerce")
    events = events.dropna(subset=[date_col, amount_col])

    if events.empty:
        return pd.Series(dtype="float64", name="Dividend"), "no_events"

    series = events.groupby(date_col)[amount_col].sum()
    series.name = "Dividend"
    return series, f"{date_col}/{amount_col}"


def fetch_split_events(
    client: RefinitivClient, ticker: str, start_dt: str, end_dt: str
) -> tuple[pd.Series, str | None]:
    params = {"CAEventType": "SSP", "SDate": start_dt, "EDate": end_dt}
    fields = [
        "TR.CAExDate",
        "TR.CAEffectiveDate",
        "TR.CACorpActDate",
        "TR.CAAdjustmentFactor",
    ]

    try:
        df = client.get_data(ticker, fields, params)
    except Exception:  # noqa: BLE001
        return pd.Series(dtype="float64", name="SplitFactor"), None

    if df is None or df.empty:
        return pd.Series(dtype="float64", name="SplitFactor"), "no_events"

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

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[adj_col] = pd.to_numeric(out[adj_col], errors="coerce")
    out = out.dropna(subset=[date_col, adj_col])

    if out.empty:
        return pd.Series(dtype="float64", name="SplitFactor"), "no_events"

    out["SplitFactor"] = out[adj_col]
    series = out.groupby(date_col)["SplitFactor"].prod()
    series.name = "SplitFactor"
    return series, "success"
