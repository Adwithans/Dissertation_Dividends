from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import pandas as pd

from ..data.refinitiv_client import RefinitivClient


class StrategyDataProvider(Protocol):
    def get_sp500_constituents_as_of(self, as_of_date: str) -> list[str]:
        ...

    def get_market_cap_snapshot(self, rics: list[str], as_of_date: str) -> pd.DataFrame:
        ...

    def get_dividend_events(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        ...

    def get_close_history(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        ...


def _norm_cols(df: pd.DataFrame | None) -> pd.DataFrame:
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


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


@dataclass
class RefinitivStrategyDataProvider:
    client: RefinitivClient
    batch_size: int = 200
    enable_cache: bool = True
    _constituents_cache: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _market_cap_cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False
    )
    _dividend_cache: dict[tuple[str, str, tuple[str, ...]], pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False
    )
    _close_cache: dict[tuple[str, str, tuple[str, ...]], pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False
    )
    _stats: dict[str, int] = field(
        default_factory=lambda: {
            "constituents_calls": 0,
            "market_cap_calls": 0,
            "dividend_calls": 0,
            "close_calls": 0,
            "constituents_cache_hits": 0,
            "market_cap_cache_hits": 0,
            "dividend_cache_hits": 0,
            "close_cache_hits": 0,
        },
        init=False,
        repr=False,
    )

    @staticmethod
    def _rd_df(data):
        if isinstance(data, tuple):
            return data[0]
        return data

    @staticmethod
    def _ric_key(rics: list[str]) -> tuple[str, ...]:
        return tuple(sorted(set(str(r).strip() for r in rics if str(r).strip())))

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def get_sp500_constituents_as_of(self, as_of_date: str) -> list[str]:
        if self.enable_cache and as_of_date in self._constituents_cache:
            self._stats["constituents_cache_hits"] += 1
            return list(self._constituents_cache[as_of_date])

        self._stats["constituents_calls"] += 1
        date_code = as_of_date.replace("-", "")
        params = {"SDate": as_of_date, "EDate": as_of_date}

        chain_universe = f"0#.SPX({date_code})"
        raw = self.client.get_data(chain_universe, ["TR.PriceClose"], params)
        df = _norm_cols(self._rd_df(raw))
        if df.empty:
            return []

        ric_col = _pick_col(df, ("instrument",)) or _pick_col(df, ("ric",))
        if ric_col is None:
            return []

        values = df[ric_col].dropna().astype(str).str.strip()
        out = list(dict.fromkeys([v for v in values if v]))
        if self.enable_cache:
            self._constituents_cache[as_of_date] = out
        return out

    def get_market_cap_snapshot(self, rics: list[str], as_of_date: str) -> pd.DataFrame:
        key = (as_of_date, self._ric_key(rics))
        if self.enable_cache and key in self._market_cap_cache:
            self._stats["market_cap_cache_hits"] += 1
            return self._market_cap_cache[key].copy()

        self._stats["market_cap_calls"] += 1
        fields = ["TR.CompanyMarketCap", "TR.CompanyMarketCap.date"]
        params = {"SDate": as_of_date, "EDate": as_of_date}

        parts: list[pd.DataFrame] = []
        for chunk in _chunked(rics, self.batch_size):
            df, _ = self.client.get_eikon_data(chunk, fields, params)
            df = _norm_cols(df)
            if df.empty:
                continue

            ric_col = _pick_col(df, ("instrument",))
            mcap_col = _pick_col(df, ("market", "cap"))
            date_col = _pick_col(df, ("date",))
            if ric_col is None or mcap_col is None:
                continue

            out = pd.DataFrame()
            out["RIC"] = df[ric_col].astype(str).str.strip()
            out["MarketCap"] = pd.to_numeric(df[mcap_col], errors="coerce")
            out["MarketCapDate"] = (
                pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.Timestamp(as_of_date)
            )
            out = out.dropna(subset=["RIC", "MarketCap"])
            out = out.loc[out["MarketCap"] > 0]
            parts.append(out)

        if not parts:
            out = pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])
            if self.enable_cache:
                self._market_cap_cache[key] = out
            return out

        out = pd.concat(parts, ignore_index=True)
        out = out.sort_values(["RIC", "MarketCapDate"]).drop_duplicates(subset=["RIC"], keep="last")
        out = out.reset_index(drop=True)
        if self.enable_cache:
            self._market_cap_cache[key] = out
        return out.copy()

    def get_dividend_events(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        key = (start_date, end_date, self._ric_key(rics))
        if self.enable_cache and key in self._dividend_cache:
            self._stats["dividend_cache_hits"] += 1
            return self._dividend_cache[key].copy()

        self._stats["dividend_calls"] += 1
        fields = ["TR.DivExDate", "TR.DivUnadjustedGross"]
        params = {"SDate": start_date, "EDate": end_date, "DateType": "ED"}

        parts: list[pd.DataFrame] = []
        for chunk in _chunked(rics, self.batch_size):
            df, _ = self.client.get_eikon_data(chunk, fields, params)
            df = _norm_cols(df)
            if df.empty:
                continue

            ric_col = _pick_col(df, ("instrument",))
            date_col = _pick_col(df, ("ex", "date")) or _pick_col(df, ("date",))
            amt_col = _pick_col(df, ("gross",)) or _pick_col(df, ("div", "amount"))
            if ric_col is None or date_col is None or amt_col is None:
                continue

            out = pd.DataFrame()
            out["RIC"] = df[ric_col].astype(str).str.strip()
            out["Date"] = pd.to_datetime(df[date_col], errors="coerce")
            out["Dividend"] = pd.to_numeric(df[amt_col], errors="coerce")
            out = out.dropna(subset=["RIC", "Date", "Dividend"])
            out = out.loc[out["Dividend"] > 0]
            parts.append(out)

        if not parts:
            out = pd.DataFrame(columns=["RIC", "Date", "Dividend"])
            if self.enable_cache:
                self._dividend_cache[key] = out
            return out
        out = pd.concat(parts, ignore_index=True)
        if self.enable_cache:
            self._dividend_cache[key] = out
        return out.copy()

    @staticmethod
    def _close_history_to_long(raw: pd.DataFrame, rics: list[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["Date", "RIC", "CLOSE"])

        df = raw.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

        df = df.loc[df.index.notna()].sort_index()

        if isinstance(df.columns, pd.MultiIndex):
            long_rows: list[pd.DataFrame] = []
            for ric in rics:
                close_col = None
                for col in df.columns:
                    if len(col) < 2:
                        continue
                    # Common shape: (RIC, FIELD)
                    if str(col[0]) == ric and str(col[1]).upper() in {"TRDPRC_1", "CLOSE"}:
                        close_col = col
                        break
                    # Alternate shape: (FIELD, RIC)
                    if str(col[1]) == ric and str(col[0]).upper() in {"TRDPRC_1", "CLOSE"}:
                        close_col = col
                        break
                if close_col is None:
                    continue
                part = pd.DataFrame(
                    {
                        "Date": df.index,
                        "RIC": ric,
                        "CLOSE": pd.to_numeric(df[close_col], errors="coerce"),
                    }
                )
                long_rows.append(part)
            if not long_rows:
                return pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
            out = pd.concat(long_rows, ignore_index=True)
            out = out.dropna(subset=["Date", "CLOSE"])
            out = out.loc[out["CLOSE"] > 0]
            return out

        # Common multi-universe response shape is single-level columns with RIC names.
        ric_set = set(rics)
        ric_cols = [c for c in df.columns if str(c) in ric_set]
        if ric_cols:
            wide = df[ric_cols].copy()
            wide = wide.reset_index().rename(columns={wide.index.name or "index": "Date"})
            if "Date" not in wide.columns:
                wide = wide.rename(columns={wide.columns[0]: "Date"})
            long = wide.melt(id_vars=["Date"], var_name="RIC", value_name="CLOSE")
            long["CLOSE"] = pd.to_numeric(long["CLOSE"], errors="coerce")
            long = long.dropna(subset=["Date", "RIC", "CLOSE"])
            long = long.loc[long["CLOSE"] > 0]
            return long[["Date", "RIC", "CLOSE"]]

        out = pd.DataFrame({"Date": df.index})
        if "TRDPRC_1" in df.columns:
            out["CLOSE"] = pd.to_numeric(df["TRDPRC_1"], errors="coerce")
            out["RIC"] = rics[0] if rics else ""
        elif "CLOSE" in df.columns:
            out["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
            out["RIC"] = rics[0] if rics else ""
        else:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                return pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
            out["CLOSE"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce")
            out["RIC"] = str(numeric_cols[0])

        out = out.dropna(subset=["Date", "RIC", "CLOSE"])
        out = out.loc[out["CLOSE"] > 0]
        return out[["Date", "RIC", "CLOSE"]]

    def get_close_history(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if not rics:
            return pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
        key = (start_date, end_date, self._ric_key(rics))
        if self.enable_cache and key in self._close_cache:
            self._stats["close_cache_hits"] += 1
            return self._close_cache[key].copy()

        self._stats["close_calls"] += 1
        parts: list[pd.DataFrame] = []
        for chunk in _chunked(rics, self.batch_size):
            try:
                raw = self.client.get_history(
                    universe=chunk,
                    fields=["TRDPRC_1"],
                    interval="daily",
                    start=start_date,
                    end=end_date,
                    adjustments="unadjusted",
                )
                chunk_out = self._close_history_to_long(raw, chunk)
            except Exception:  # noqa: BLE001
                chunk_out = pd.DataFrame(columns=["Date", "RIC", "CLOSE"])

            if chunk_out.empty and len(chunk) > 1:
                # Some deprecated/renamed RICs can null the whole batch response.
                for ric in chunk:
                    try:
                        raw_one = self.client.get_history(
                            universe=[ric],
                            fields=["TRDPRC_1"],
                            interval="daily",
                            start=start_date,
                            end=end_date,
                            adjustments="unadjusted",
                        )
                        one_out = self._close_history_to_long(raw_one, [ric])
                    except Exception:  # noqa: BLE001
                        continue
                    if not one_out.empty:
                        parts.append(one_out)
                continue

            if not chunk_out.empty:
                parts.append(chunk_out)

        if not parts:
            out = pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
            if self.enable_cache:
                self._close_cache[key] = out
            return out

        out = pd.concat(parts, ignore_index=True)
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).sort_values(["Date", "RIC"]).reset_index(drop=True)
        out = out.drop_duplicates(subset=["Date", "RIC"], keep="last")
        if self.enable_cache:
            self._close_cache[key] = out
        return out.copy()


def trailing_lookback_start(as_of_date: str, months: int) -> str:
    as_of = pd.Timestamp(as_of_date)
    start = as_of - pd.DateOffset(months=months)
    return start.date().isoformat()


def utc_now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
