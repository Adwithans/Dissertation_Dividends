from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import eikon as ek
import pandas as pd
import refinitiv.data as rd
from dotenv import load_dotenv

from src.dividend_portfolio.runtime_warnings import configure_runtime_warning_filters

configure_runtime_warning_filters()


@dataclass
class FeasibilityRow:
    ric: str
    dividend_event_count: int
    market_cap_points: int
    first_market_cap_date: str | None
    last_market_cap_date: str | None
    market_cap_source: str
    dividend_source: str
    notes: str


def _rd_df(data):
    if isinstance(data, tuple):
        return data[0]
    return data


def _normalize_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _pick_col(df: pd.DataFrame, include_all: tuple[str, ...]) -> str | None:
    for col in df.columns:
        lower = col.lower()
        if all(token in lower for token in include_all):
            return col
    return None


def _extract_ric_list(df: pd.DataFrame) -> list[str]:
    ric_col = _pick_col(df, ("instrument",))
    if ric_col is None:
        ric_col = _pick_col(df, ("ric",))
    if ric_col is None:
        return []
    values = df[ric_col].dropna().astype(str).str.strip()
    seen: set[str] = set()
    ordered: list[str] = []
    for ric in values:
        if ric and ric not in seen:
            ordered.append(ric)
            seen.add(ric)
    return ordered


def _get_data_with_fallback(
    universe: str | list[str], fields: list[str], params: dict[str, str]
) -> tuple[pd.DataFrame, str, str | None]:
    try:
        df, err = ek.get_data(universe, fields, params)
        out = _normalize_df(df)
        if out is not None:
            err_msg = None
            if err:
                err_msg = str(err)
            return out, "eikon", err_msg
    except Exception as exc:  # noqa: BLE001
        eikon_error = str(exc)
    else:
        eikon_error = None

    try:
        data = rd.get_data(universe, fields, params)
        out = _normalize_df(_rd_df(data))
        return out, "rd", eikon_error
    except Exception as exc:  # noqa: BLE001
        merged = str(exc) if eikon_error is None else f"eikon={eikon_error}; rd={exc}"
        return pd.DataFrame(), "none", merged


def get_constituents_as_of(index_ric: str, as_of_date: str) -> tuple[list[str], str, str | None]:
    date_code = as_of_date.replace("-", "")
    params = {"SDate": as_of_date, "EDate": as_of_date}

    try:
        chain_universe = f"0#{index_ric}({date_code})"
        chain_df = _normalize_df(_rd_df(rd.get_data(chain_universe, ["TR.PriceClose"], params)))
        rics = _extract_ric_list(chain_df)
        if rics:
            return rics, "0# chain-as-of", None
    except Exception as exc:  # noqa: BLE001
        chain_error = str(exc)
    else:
        chain_error = "0# chain returned empty"

    try:
        fields = ["TR.IndexConstituentRIC", "TR.IndexConstituentName"]
        fallback_df = _normalize_df(_rd_df(rd.get_data(index_ric, fields, {"SDate": as_of_date})))
        rics = _extract_ric_list(fallback_df)
        if rics:
            return rics, "TR.IndexConstituentRIC", chain_error
        return [], "TR.IndexConstituentRIC", f"{chain_error}; fallback returned empty"
    except Exception as exc:  # noqa: BLE001
        return [], "none", f"{chain_error}; fallback={exc}"


def get_constituent_changes(index_ric: str, start: str, end: str) -> tuple[pd.DataFrame, str | None]:
    fields = [
        "TR.IndexJLConstituentChangeDate",
        "TR.IndexJLConstituentRIC",
        "TR.IndexJLConstituentName",
        "TR.IndexJLConstituentChange",
    ]
    params = {"SDATE": start, "EDATE": end, "IC": "B"}
    try:
        df = _normalize_df(_rd_df(rd.get_data(index_ric, fields, params)))
        if df.empty:
            return df, "empty response"
        date_col = _pick_col(df, ("change", "date")) or _pick_col(df, ("date",))
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df, None
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), str(exc)


def assess_ric(ric: str, start: str, end: str) -> FeasibilityRow:
    div_fields = ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivCurr"]
    div_params = {"SDate": start, "EDate": end, "DateType": "ED"}

    div_df, div_source, div_err = _get_data_with_fallback(ric, div_fields, div_params)
    div_date_col = _pick_col(div_df, ("ex", "date"))
    div_amt_col = _pick_col(div_df, ("gross",)) or _pick_col(div_df, ("div", "amount"))

    div_event_count = 0
    if div_date_col and div_amt_col and not div_df.empty:
        div_df = div_df[[div_date_col, div_amt_col]].copy()
        div_df[div_date_col] = pd.to_datetime(div_df[div_date_col], errors="coerce")
        div_df[div_amt_col] = pd.to_numeric(div_df[div_amt_col], errors="coerce")
        div_df = div_df.dropna(subset=[div_date_col, div_amt_col])
        div_df = div_df.loc[div_df[div_amt_col] > 0]
        div_event_count = int(div_df.shape[0])

    mc_fields = ["TR.CompanyMarketCap", "TR.CompanyMarketCap.date"]
    mc_params = {"SDate": start, "EDate": end, "Frq": "M"}
    mc_df, mc_source, mc_err = _get_data_with_fallback(ric, mc_fields, mc_params)
    mc_date_col = _pick_col(mc_df, ("date",))
    mc_val_col = _pick_col(mc_df, ("market", "cap"))

    market_cap_points = 0
    first_date: str | None = None
    last_date: str | None = None
    if mc_date_col and mc_val_col and not mc_df.empty:
        mc_df = mc_df[[mc_date_col, mc_val_col]].copy()
        mc_df[mc_date_col] = pd.to_datetime(mc_df[mc_date_col], errors="coerce")
        mc_df[mc_val_col] = pd.to_numeric(mc_df[mc_val_col], errors="coerce")
        mc_df = mc_df.dropna(subset=[mc_date_col, mc_val_col]).sort_values(by=mc_date_col)
        market_cap_points = int(mc_df.shape[0])
        if market_cap_points > 0:
            first_date = mc_df[mc_date_col].iloc[0].date().isoformat()
            last_date = mc_df[mc_date_col].iloc[-1].date().isoformat()

    notes = []
    if div_err:
        notes.append(f"div_err={div_err}")
    if mc_err:
        notes.append(f"mcap_err={mc_err}")
    if not div_date_col or not div_amt_col:
        notes.append("div_columns_missing")
    if not mc_date_col or not mc_val_col:
        notes.append("mcap_columns_missing")

    return FeasibilityRow(
        ric=ric,
        dividend_event_count=div_event_count,
        market_cap_points=market_cap_points,
        first_market_cap_date=first_date,
        last_market_cap_date=last_date,
        market_cap_source=mc_source,
        dividend_source=div_source,
        notes="; ".join(notes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feasibility test for S&P dividend payers + historical market cap with Refinitiv/LSEG APIs."
    )
    parser.add_argument("--index-ric", default=".SPX", help="Index RIC, default is .SPX")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="End date in YYYY-MM-DD")
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Date used for initial constituent snapshot; defaults to --start-date.",
    )
    parser.add_argument("--sample-size", type=int, default=60, help="How many constituents to test for feasibility")
    parser.add_argument(
        "--output-dir",
        default="data/raw/research",
        help="Directory for CSV outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    as_of_date = args.as_of_date or args.start_date
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    app_key = os.getenv("EIKON_API_KEY")
    if app_key:
        ek.set_app_key(app_key)

    print(f"Opening Refinitiv session for index={args.index_ric}, range={args.start_date}..{args.end_date}")
    rd.open_session()
    try:
        constituents, method, const_err = get_constituents_as_of(args.index_ric, as_of_date)
        print(f"Constituent snapshot method: {method}")
        if const_err:
            print(f"Constituent snapshot note: {const_err}")
        print(f"Constituents found: {len(constituents)}")

        if not constituents:
            raise RuntimeError("No constituents found for as-of date. Cannot continue feasibility test.")

        changes_df, changes_err = get_constituent_changes(args.index_ric, args.start_date, args.end_date)
        if changes_err:
            print(f"Constituent changes note: {changes_err}")
        else:
            print(f"Constituent changes rows: {len(changes_df)}")

        sample_rics = constituents[: max(args.sample_size, 1)]
        print(f"Testing {len(sample_rics)} constituents for dividends + market cap history")

        rows = [assess_ric(ric, args.start_date, args.end_date) for ric in sample_rics]
        feasibility_df = pd.DataFrame([row.__dict__ for row in rows])

        snapshot_path = output_dir / f"sp500_constituents_asof_{as_of_date}.csv"
        changes_path = output_dir / f"sp500_constituent_changes_{args.start_date}_{args.end_date}.csv"
        feasibility_path = output_dir / f"sp500_dividend_mcap_feasibility_{args.start_date}_{args.end_date}.csv"

        pd.DataFrame({"RIC": constituents}).to_csv(snapshot_path, index=False)
        changes_df.to_csv(changes_path, index=False)
        feasibility_df.to_csv(feasibility_path, index=False)

        with_dividend = int((feasibility_df["dividend_event_count"] > 0).sum())
        with_mcap = int((feasibility_df["market_cap_points"] > 0).sum())

        print(f"Saved: {snapshot_path}")
        print(f"Saved: {changes_path}")
        print(f"Saved: {feasibility_path}")
        print(f"Dividend coverage in sample: {with_dividend}/{len(feasibility_df)}")
        print(f"Market-cap coverage in sample: {with_mcap}/{len(feasibility_df)}")
    finally:
        rd.close_session()
        print("Refinitiv session closed.")


if __name__ == "__main__":
    main()
