from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import eikon as ek
import pandas as pd
import refinitiv.data as rd
from dotenv import load_dotenv

from ..runtime_warnings import configure_runtime_warning_filters

configure_runtime_warning_filters()


def _rd_df(data):
    if isinstance(data, tuple):
        return data[0]
    return data


def _norm_cols(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _pick_col(df: pd.DataFrame, include_all: tuple[str, ...]) -> str | None:
    for col in df.columns:
        name = col.lower()
        if all(token in name for token in include_all):
            return col
    return None


def _call_retry(fn, *args, **kwargs):
    delays = (1.0, 2.0, 4.0)
    last_exc: Exception | None = None
    for i, delay in enumerate((0.0, *delays)):
        if delay > 0:
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if i == len(delays):
                break
    assert last_exc is not None
    raise last_exc


def _get_constituents_as_of(index_ric: str, as_of_date: str) -> list[str]:
    date_code = as_of_date.replace("-", "")
    chain_universe = f"0#{index_ric}({date_code})"
    params = {"SDate": as_of_date, "EDate": as_of_date}

    chain_df = _norm_cols(_rd_df(_call_retry(rd.get_data, chain_universe, ["TR.PriceClose"], params)))
    if chain_df.empty:
        return []

    ric_col = _pick_col(chain_df, ("instrument",))
    if ric_col is None:
        ric_col = _pick_col(chain_df, ("ric",))
    if ric_col is None:
        return []

    rics = chain_df[ric_col].dropna().astype(str).str.strip()
    rics = [r for r in rics if r]
    # preserve order, remove duplicates
    return list(dict.fromkeys(rics))


def _fetch_market_cap_snapshot(rics: list[str], as_of_date: str) -> pd.DataFrame:
    if not rics:
        return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])
    fields = ["TR.CompanyMarketCap", "TR.CompanyMarketCap.date"]
    params = {"SDate": as_of_date, "EDate": as_of_date}
    df, _ = _call_retry(ek.get_data, rics, fields, params)
    df = _norm_cols(df)

    ric_col = _pick_col(df, ("instrument",))
    mcap_col = _pick_col(df, ("market", "cap"))
    date_col = _pick_col(df, ("date",))

    if ric_col is None or mcap_col is None:
        return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])

    out = pd.DataFrame()
    out["RIC"] = df[ric_col].astype(str).str.strip()
    out["MarketCap"] = pd.to_numeric(df[mcap_col], errors="coerce")
    if date_col:
        out["MarketCapDate"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        out["MarketCapDate"] = pd.NaT
    out = out.dropna(subset=["RIC", "MarketCap"])
    out = out.loc[out["MarketCap"] > 0]
    out = out.drop_duplicates(subset=["RIC"], keep="last")
    return out


def _fetch_dividend_payers(rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    if not rics:
        return pd.DataFrame(columns=["RIC", "DividendCashInQuarter"])
    fields = ["TR.DivExDate", "TR.DivUnadjustedGross"]
    params = {"SDate": start_date, "EDate": end_date, "DateType": "ED"}
    df, _ = _call_retry(ek.get_data, rics, fields, params)
    df = _norm_cols(df)

    ric_col = _pick_col(df, ("instrument",))
    amt_col = _pick_col(df, ("gross",)) or _pick_col(df, ("div", "amount"))
    if ric_col is None or amt_col is None:
        return pd.DataFrame(columns=["RIC", "DividendCashInQuarter"])

    out = pd.DataFrame()
    out["RIC"] = df[ric_col].astype(str).str.strip()
    out["DividendCashInQuarter"] = pd.to_numeric(df[amt_col], errors="coerce")
    out = out.dropna(subset=["RIC", "DividendCashInQuarter"])
    out = out.groupby("RIC", as_index=False)["DividendCashInQuarter"].sum()
    out = out.loc[out["DividendCashInQuarter"] > 0]
    return out


def _quarter_windows(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    periods = pd.period_range(start=start.to_period("Q"), end=end.to_period("Q"), freq="Q")

    windows: list[tuple[str, str, str]] = []
    for p in periods:
        q_start = max(p.start_time.normalize(), start).date().isoformat()
        q_end = min(p.end_time.normalize(), end).date().isoformat()
        windows.append((str(p), q_start, q_end))
    return windows


def _infer_range_from_portfolio_timeseries(path: Path) -> tuple[str, str]:
    df = pd.read_csv(path, usecols=["Date"])
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna().sort_values()
    if dates.empty:
        raise RuntimeError(f"No valid Date values found in {path}")
    return dates.iloc[0].date().isoformat(), dates.iloc[-1].date().isoformat()


def build_rankings(
    *,
    index_ric: str,
    start_date: str,
    end_date: str,
    top_n: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    windows = _quarter_windows(start_date, end_date)

    for quarter_label, q_start, q_end in windows:
        print(f"[INFO] {quarter_label}: as_of={q_end}")
        rics = _get_constituents_as_of(index_ric, q_end)
        if not rics:
            print(f"[WARN] {quarter_label}: no constituents returned")
            continue

        market_cap = _fetch_market_cap_snapshot(rics, q_end)
        div_payers = _fetch_dividend_payers(rics, q_start, q_end)

        if market_cap.empty or div_payers.empty:
            print(f"[WARN] {quarter_label}: empty market-cap or dividend set")
            continue

        merged = market_cap.merge(div_payers, how="inner", on="RIC")
        if merged.empty:
            print(f"[WARN] {quarter_label}: no overlapping dividend-paying names with market cap")
            continue

        merged = merged.sort_values("MarketCap", ascending=False).reset_index(drop=True)
        merged["RankByMarketCap"] = merged.index + 1
        merged = merged.loc[merged["RankByMarketCap"] <= top_n].copy()
        merged.insert(0, "Quarter", quarter_label)
        merged.insert(1, "QuarterStart", q_start)
        merged.insert(2, "QuarterEnd", q_end)
        rows.append(merged)

    if not rows:
        return pd.DataFrame(
            columns=[
                "Quarter",
                "QuarterStart",
                "QuarterEnd",
                "RankByMarketCap",
                "RIC",
                "MarketCap",
                "MarketCapDate",
                "DividendCashInQuarter",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    out["MarketCapDate"] = pd.to_datetime(out["MarketCapDate"], errors="coerce")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank S&P 500 dividend-paying constituents by market cap every quarter."
    )
    parser.add_argument("--index-ric", default=".SPX", help="Index RIC, default .SPX")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=20, help="Top N market-cap names per quarter")
    parser.add_argument(
        "--portfolio-timeseries",
        default="data/runs/latest/portfolio_timeseries.csv",
        help="Used to infer date range when start/end are omitted",
    )
    parser.add_argument(
        "--output",
        default="data/runs/latest/quarterly_top_dividend_marketcap.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_date = args.start_date
    end_date = args.end_date
    if start_date is None or end_date is None:
        inferred_start, inferred_end = _infer_range_from_portfolio_timeseries(Path(args.portfolio_timeseries))
        start_date = start_date or inferred_start
        end_date = end_date or inferred_end

    load_dotenv(".env")
    app_key = os.getenv("EIKON_API_KEY")
    if app_key:
        ek.set_app_key(app_key)

    print(
        "[INFO] Building quarterly rankings for "
        f"{args.index_ric} from {start_date} to {end_date}; top_n={args.top_n}"
    )

    rd.open_session()
    try:
        rankings = build_rankings(
            index_ric=args.index_ric,
            start_date=start_date,
            end_date=end_date,
            top_n=max(args.top_n, 1),
        )
    finally:
        rd.close_session()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output_path, index=False)

    q_count = rankings["Quarter"].nunique() if not rankings.empty else 0
    print(f"[OK] Wrote {len(rankings)} rows across {q_count} quarters to {output_path}")

    if not rankings.empty:
        leaders = rankings.loc[rankings["RankByMarketCap"] == 1, ["Quarter", "RIC", "MarketCap"]]
        leaders_path = output_path.with_name("quarterly_top1_dividend_marketcap.csv")
        leaders.to_csv(leaders_path, index=False)
        print(f"[OK] Wrote quarter leaders to {leaders_path}")


if __name__ == "__main__":
    main()
