from __future__ import annotations

import pandas as pd

from .refinitiv_client import RefinitivClient

PRICE_FIELDS = ["OPEN", "HIGH", "LOW", "CLOSE"]
QUOTE_FIELDS = ["BID", "ASK"]


def _normalize_price_df(df: pd.DataFrame, expected_fields: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    expected = expected_fields or PRICE_FIELDS

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.set_index("Date")
    else:
        out.index = pd.to_datetime(out.index, errors="coerce")

    if isinstance(out.columns, pd.MultiIndex):
        for level in range(out.columns.nlevels):
            level_vals = set(out.columns.get_level_values(level))
            if any(field in level_vals for field in expected):
                out.columns = out.columns.get_level_values(level)
                break

    out = out.loc[out.index.notna()].sort_index()
    return out


def fetch_prices(client: RefinitivClient, ticker: str, start_dt: str, end_dt: str) -> pd.DataFrame | None:
    """Fetch unadjusted daily prices using already-tested API patterns only."""

    try:
        prices = client.get_history(
            universe=[ticker],
            fields=PRICE_FIELDS,
            interval="daily",
            start=start_dt,
            end=end_dt,
            adjustments="unadjusted",
        )
    except Exception:  # noqa: BLE001
        prices = None

    if prices is not None and not prices.empty:
        prices = _normalize_price_df(prices, expected_fields=PRICE_FIELDS)
        available = [c for c in PRICE_FIELDS if c in prices.columns]
        if available:
            return prices[available]
        return prices

    try:
        fallback = client.get_history(
            universe=[ticker],
            fields=["TRDPRC_1"],
            interval="daily",
            start=start_dt,
            end=end_dt,
            adjustments="unadjusted",
        )
    except Exception:  # noqa: BLE001
        return None

    if fallback is None or fallback.empty:
        return None

    fallback = _normalize_price_df(fallback, expected_fields=["TRDPRC_1", "CLOSE"])
    if "TRDPRC_1" in fallback.columns:
        fallback = fallback.rename(columns={"TRDPRC_1": "CLOSE"})

    return fallback[["CLOSE"]] if "CLOSE" in fallback.columns else fallback


def fetch_bid_ask(client: RefinitivClient, ticker: str, start_dt: str, end_dt: str) -> pd.DataFrame:
    try:
        quotes = client.get_history(
            universe=[ticker],
            fields=QUOTE_FIELDS,
            interval="daily",
            start=start_dt,
            end=end_dt,
            adjustments="unadjusted",
        )
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=QUOTE_FIELDS)

    if quotes is None or quotes.empty:
        return pd.DataFrame(columns=QUOTE_FIELDS)

    quotes = _normalize_price_df(quotes, expected_fields=QUOTE_FIELDS)
    available = [c for c in QUOTE_FIELDS if c in quotes.columns]
    if not available:
        return pd.DataFrame(columns=QUOTE_FIELDS)
    out = quotes[available].copy()
    if "BID" not in out.columns:
        out["BID"] = pd.NA
    if "ASK" not in out.columns:
        out["ASK"] = pd.NA
    return out[["BID", "ASK"]]
