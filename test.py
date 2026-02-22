from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import refinitiv.data as rd


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # rd.get_history often returns MultiIndex columns like (RIC, Field).
    if isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.columns = ["_".join(str(x) for x in col if x not in (None, "")) for col in out.columns]
        return out
    return df


def _to_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index, errors="coerce")
    return df.loc[df.index.notna()].sort_index()


def _first_numeric_col(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("No numeric column to plot.")
    return cols[0]


def fetch_rd_close(ric: str, start: str, end: str, *, adjustments: Optional[str]) -> pd.Series:
    # Use a *pricing* field so Refinitiv can apply `adjustments=...`.
    # TR.PriceClose is a TR field and will ignore `adjustments` (you'll see a warning).
    kwargs = dict(universe=[ric], fields=["TRDPRC_1"], interval="daily", start=start, end=end)
    if adjustments is not None:
        kwargs["adjustments"] = adjustments
    df = rd.get_history(**kwargs)
    df = _to_dt_index(_flatten_columns(df))
    col = _first_numeric_col(df)
    name = f"RD_TRDPRC_1_{adjustments or 'default'}"
    return pd.to_numeric(df[col], errors="coerce").rename(name)


def _rd_df(data):
    # rd.get_data sometimes returns (df, metadata)
    if isinstance(data, tuple):
        return data[0]
    return data


def _extract_dates(df: pd.DataFrame, field: str) -> list[date]:
    if df is None or df.empty:
        return []
    if field in df.columns:
        s = pd.to_datetime(df[field], errors="coerce")
    else:
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col is None:
            return []
        s = pd.to_datetime(df[date_col], errors="coerce")
    s = s.dropna().dt.date.unique()
    return sorted(s.tolist())


def print_split_dates_api(ric: str, start: str, end: str) -> None:
    params = {"CAEventType": "SSP", "SDate": start, "EDate": end}
    fields = [
        ("CorpActDate", "TR.CACorpActDate"),
        ("EffectiveDate", "TR.CAEffectiveDate"),
        ("ExDate", "TR.CAExDate"),
        ("RecordDate", "TR.CARecordDate"),
        ("ActualDate", "TR.CAActualDate"),
        ("AnnouncementDate", "TR.CAAnnouncementDate"),
        ("PayDate", "TR.CAPayDate"),
    ]

    # Baseline call to confirm the endpoint is returning rows.
    base = _rd_df(rd.get_data(ric, ["TR.CACorpActDate", "TR.CAAdjustmentFactor"], params))
    if base is None or base.empty:
        print("AAPL.O split dates (API): none found (baseline query returned empty)")
        return
    base_dates = _extract_dates(base, "TR.CACorpActDate")
    print("AAPL.O split dates (API) - CorpActDate:", base_dates)

    # Query each date field individually so a bad field doesn't blank the whole response.
    for label, field in fields[1:]:
        try:
            df = _rd_df(rd.get_data(ric, [field], params))
        except Exception as exc:
            print(f"AAPL.O split dates (API) - {label}: error {exc}")
            continue
        dates = _extract_dates(df, field)
        print(f"AAPL.O split dates (API) - {label}:", dates)


def main() -> None:
    ric = "AAPL.O"
    start = "2010-01-01"
    end = date.today().isoformat()

    rd.open_session()
    try:
        close_unadj = fetch_rd_close(ric, start, end, adjustments="unadjusted")
        close_cch = fetch_rd_close(ric, start, end, adjustments="CCH")
        print_split_dates_api(ric, start, end)
    finally:
        rd.close_session()

    df = pd.concat([close_unadj, close_cch], axis=1).sort_index()
    print(df.tail())

    # Plot explicitly so overlapping lines are still distinguishable.
    ax = df[close_unadj.name].plot(figsize=(12, 5), lw=1.4, color="black", label=close_unadj.name)
    df[close_cch.name].plot(ax=ax, lw=1.2, ls="--", color="tab:orange", label=close_cch.name)
    ax.set_title(f"{ric} (rd): unadjusted vs CCH ({start}..{end})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
