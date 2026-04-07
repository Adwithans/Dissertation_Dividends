from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import sqlite3
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

    def get_bid_ask_history(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
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


def _day_before(iso_date: str) -> str:
    return (pd.Timestamp(iso_date) - pd.Timedelta(days=1)).date().isoformat()


def _day_after(iso_date: str) -> str:
    return (pd.Timestamp(iso_date) + pd.Timedelta(days=1)).date().isoformat()


@dataclass
class RefinitivStrategyDataProvider:
    client: RefinitivClient
    batch_size: int = 200
    enable_cache: bool = True
    persistent_cache_db_path: str | Path | None = None
    persistent_cache_enabled: bool = True
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
    _bid_ask_cache: dict[tuple[str, str, tuple[str, ...]], pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False
    )
    _stats: dict[str, int] = field(
        default_factory=lambda: {
            "constituents_calls": 0,
            "market_cap_calls": 0,
            "dividend_calls": 0,
            "close_calls": 0,
            "bid_ask_calls": 0,
            "constituents_cache_hits": 0,
            "market_cap_cache_hits": 0,
            "dividend_cache_hits": 0,
            "close_cache_hits": 0,
            "bid_ask_cache_hits": 0,
            "persistent_cache_hits": 0,
            "persistent_cache_misses": 0,
            "persistent_cache_writes": 0,
        },
        init=False,
        repr=False,
    )
    _persistent_conn: sqlite3.Connection | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.persistent_cache_enabled:
            return
        if self.persistent_cache_db_path is None:
            return
        db_path = Path(self.persistent_cache_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._persistent_conn = sqlite3.connect(db_path, timeout=30)
        self._persistent_conn.execute("PRAGMA journal_mode=WAL;")
        self._persistent_conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_persistent_cache_schema()

    def close(self) -> None:
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None

    def _ensure_persistent_cache_schema(self) -> None:
        if self._persistent_conn is None:
            return
        cur = self._persistent_conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS provider_cache_constituents_coverage (
                as_of_date TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS provider_cache_constituents (
                as_of_date TEXT NOT NULL,
                ric TEXT NOT NULL,
                PRIMARY KEY (as_of_date, ric)
            );

            CREATE TABLE IF NOT EXISTS provider_cache_market_caps (
                as_of_date TEXT NOT NULL,
                ric TEXT NOT NULL,
                market_cap REAL,
                market_cap_date TEXT,
                PRIMARY KEY (as_of_date, ric)
            );

            CREATE TABLE IF NOT EXISTS provider_cache_dividends (
                ric TEXT NOT NULL,
                date TEXT NOT NULL,
                dividend REAL,
                PRIMARY KEY (ric, date)
            );

            CREATE TABLE IF NOT EXISTS provider_cache_prices (
                ric TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL,
                PRIMARY KEY (ric, date)
            );

            CREATE TABLE IF NOT EXISTS provider_cache_bid_ask (
                ric TEXT NOT NULL,
                date TEXT NOT NULL,
                bid REAL,
                ask REAL,
                PRIMARY KEY (ric, date)
            );

            CREATE TABLE IF NOT EXISTS provider_cache_coverage (
                dataset TEXT NOT NULL,
                ric TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                PRIMARY KEY (dataset, ric, start_date, end_date)
            );
            """
        )
        self._persistent_conn.commit()

    def _persistent_enabled(self) -> bool:
        return self._persistent_conn is not None

    def _mark_coverage(self, dataset: str, rics: list[str], start_date: str, end_date: str) -> None:
        if not self._persistent_enabled() or not rics:
            return
        rows = [(dataset, ric, start_date, end_date) for ric in sorted(set(rics))]
        self._persistent_conn.executemany(
            """
            INSERT OR IGNORE INTO provider_cache_coverage (dataset, ric, start_date, end_date)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self._persistent_conn.commit()
        self._stats["persistent_cache_writes"] += len(rows)

    def _covered_rics(self, dataset: str, rics: list[str], start_date: str, end_date: str) -> set[str]:
        if not self._persistent_enabled() or not rics:
            return set()
        unique_rics = sorted(set(rics))
        placeholders = ",".join(["?"] * len(unique_rics))
        rows = self._persistent_conn.execute(
            f"""
            SELECT ric
            FROM provider_cache_coverage
            WHERE dataset = ?
              AND ric IN ({placeholders})
              AND start_date <= ?
              AND end_date >= ?
            GROUP BY ric
            """,
            [dataset, *unique_rics, start_date, end_date],
        ).fetchall()
        return {str(row[0]) for row in rows}

    def _clear_coverage(self, dataset: str, rics: list[str], start_date: str, end_date: str) -> int:
        if not self._persistent_enabled() or not rics:
            return 0
        unique_rics = sorted(set(rics))
        placeholders = ",".join(["?"] * len(unique_rics))
        cur = self._persistent_conn.execute(
            f"""
            DELETE FROM provider_cache_coverage
            WHERE dataset = ?
              AND ric IN ({placeholders})
              AND start_date <= ?
              AND end_date >= ?
            """,
            [dataset, *unique_rics, start_date, end_date],
        )
        self._persistent_conn.commit()
        return int(cur.rowcount or 0)

    def _coverage_bounds(
        self,
        dataset: str,
        rics: list[str],
    ) -> dict[str, tuple[str, str]]:
        if not self._persistent_enabled() or not rics:
            return {}
        unique_rics = sorted(set(rics))
        placeholders = ",".join(["?"] * len(unique_rics))
        rows = self._persistent_conn.execute(
            f"""
            SELECT ric, MIN(start_date) AS min_start, MAX(end_date) AS max_end
            FROM provider_cache_coverage
            WHERE dataset = ?
              AND ric IN ({placeholders})
            GROUP BY ric
            """,
            [dataset, *unique_rics],
        ).fetchall()
        return {str(r[0]): (str(r[1]), str(r[2])) for r in rows if r[1] is not None and r[2] is not None}

    def _missing_ranges_by_window(
        self,
        *,
        dataset: str,
        rics: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[tuple[str, str], list[str]]:
        unique_rics = sorted(set(str(r).strip() for r in rics if str(r).strip()))
        if not unique_rics:
            return {}
        if not self._persistent_enabled():
            return {(start_date, end_date): unique_rics}

        bounds = self._coverage_bounds(dataset, unique_rics)
        req_start = pd.Timestamp(start_date).normalize()
        req_end = pd.Timestamp(end_date).normalize()
        windows: dict[tuple[str, str], list[str]] = {}

        for ric in unique_rics:
            if ric not in bounds:
                windows.setdefault((start_date, end_date), []).append(ric)
                continue

            cov_start, cov_end = bounds[ric]
            cov_start_ts = pd.Timestamp(cov_start).normalize()
            cov_end_ts = pd.Timestamp(cov_end).normalize()

            if req_start < cov_start_ts:
                left_end = _day_before(cov_start)
                left_end_ts = pd.Timestamp(left_end).normalize()
                if left_end_ts >= req_start:
                    windows.setdefault((start_date, left_end), []).append(ric)

            if req_end > cov_end_ts:
                right_start = _day_after(cov_end)
                right_start_ts = pd.Timestamp(right_start).normalize()
                if right_start_ts <= req_end:
                    windows.setdefault((right_start, end_date), []).append(ric)

        return windows

    @staticmethod
    def _rd_df(data):
        if isinstance(data, tuple):
            return data[0]
        return data

    @staticmethod
    def _ric_key(rics: list[str]) -> tuple[str, ...]:
        return tuple(sorted(set(str(r).strip() for r in rics if str(r).strip())))

    @staticmethod
    def _is_rate_limited_error(exc: Exception) -> bool:
        msg = str(exc)
        return "429" in msg or "Too many requests" in msg

    @staticmethod
    def _should_fallback_to_chunking(exc: Exception) -> bool:
        msg = str(exc).lower()
        # Fallback to chunking only for request-size style failures, not throttling.
        if "429" in msg or "too many requests" in msg:
            return False
        markers = (
            "payload",
            "request entity too large",
            "413",
            "uri too long",
            "too many instruments",
            "exceeds",
            "max",
        )
        return any(marker in msg for marker in markers)

    def _prefer_single_eikon_request(self) -> bool:
        # Default to chunked mode for better provider-side throttling behavior.
        raw = os.getenv("EIKON_DISABLE_BATCHING", "0").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _fetch_eikon_frames(
        self,
        *,
        rics: list[str],
        fields: list[str],
        params: dict[str, object],
    ) -> list[pd.DataFrame]:
        if not rics:
            return []
        unique_rics = sorted(set(str(r).strip() for r in rics if str(r).strip()))
        if not unique_rics:
            return []

        if self._prefer_single_eikon_request():
            try:
                df = self.client.get_data(unique_rics, fields, params)
                return [_norm_cols(df)]
            except Exception as rd_exc:  # noqa: BLE001
                if not self._should_fallback_to_chunking(rd_exc):
                    try:
                        df, _ = self.client.get_eikon_data(unique_rics, fields, params)
                        return [_norm_cols(df)]
                    except Exception as ek_exc:  # noqa: BLE001
                        if not self._should_fallback_to_chunking(ek_exc):
                            raise

        out: list[pd.DataFrame] = []
        for chunk in _chunked(unique_rics, max(self.batch_size, 1)):
            try:
                df = self.client.get_data(chunk, fields, params)
                out.append(_norm_cols(df))
                continue
            except Exception:  # noqa: BLE001
                df, _ = self.client.get_eikon_data(chunk, fields, params)
                out.append(_norm_cols(df))
        return out

    def _fetch_eikon_only_frames(
        self,
        *,
        rics: list[str],
        fields: list[str],
        params: dict[str, object],
    ) -> list[pd.DataFrame]:
        if not rics:
            return []
        unique_rics = sorted(set(str(r).strip() for r in rics if str(r).strip()))
        if not unique_rics:
            return []

        out: list[pd.DataFrame] = []
        for chunk in _chunked(unique_rics, max(self.batch_size, 1)):
            df, _ = self.client.get_eikon_data(chunk, fields, params)
            out.append(_norm_cols(df))
        return out

    @staticmethod
    def _collect_market_cap_parts(frames: list[pd.DataFrame], *, as_of_date: str) -> list[pd.DataFrame]:
        parts_local: list[pd.DataFrame] = []
        for df in frames:
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
            if not out.empty:
                parts_local.append(out)
        return parts_local

    def _fetch_shares_outstanding_snapshot(self, rics: list[str], as_of_date: str) -> pd.DataFrame:
        if not rics:
            return pd.DataFrame(columns=["RIC", "SharesOutstanding", "SharesDate"])
        fields = ["TR.CompanySharesOutstanding", "TR.CompanySharesOutstanding.date"]
        params = {"SDate": as_of_date, "EDate": as_of_date}

        def collect_parts(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
            parts_local: list[pd.DataFrame] = []
            for df in frames:
                if df.empty:
                    continue
                ric_col = _pick_col(df, ("instrument",))
                shares_col = _pick_col(df, ("shares", "outstanding"))
                date_col = _pick_col(df, ("date",))
                if ric_col is None or shares_col is None:
                    continue

                out = pd.DataFrame()
                out["RIC"] = df[ric_col].astype(str).str.strip()
                out["SharesOutstanding"] = pd.to_numeric(df[shares_col], errors="coerce")
                out["SharesDate"] = (
                    pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.Timestamp(as_of_date)
                )
                out = out.dropna(subset=["RIC", "SharesOutstanding"])
                out = out.loc[out["SharesOutstanding"] > 0]
                if not out.empty:
                    parts_local.append(out)
            return parts_local

        parts = collect_parts(self._fetch_eikon_frames(rics=rics, fields=fields, params=params))
        if not parts:
            try:
                parts = collect_parts(self._fetch_eikon_only_frames(rics=rics, fields=fields, params=params))
            except Exception:  # noqa: BLE001
                parts = []
        if not parts:
            return pd.DataFrame(columns=["RIC", "SharesOutstanding", "SharesDate"])
        out = pd.concat(parts, ignore_index=True)
        out = out.sort_values(["RIC", "SharesDate"]).drop_duplicates(subset=["RIC"], keep="last")
        return out.reset_index(drop=True)

    def _build_market_cap_from_close_and_shares(self, rics: list[str], as_of_date: str) -> pd.DataFrame:
        if not rics:
            return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])

        shares = self._fetch_shares_outstanding_snapshot(rics, as_of_date)
        if shares.empty:
            return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])

        close_history = self.get_close_history(rics, as_of_date, as_of_date)
        if close_history.empty:
            return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])

        close_latest = close_history.copy()
        close_latest["Date"] = pd.to_datetime(close_latest["Date"], errors="coerce")
        close_latest["CLOSE"] = pd.to_numeric(close_latest["CLOSE"], errors="coerce")
        close_latest = close_latest.dropna(subset=["RIC", "Date", "CLOSE"])
        close_latest = close_latest.sort_values(["RIC", "Date"]).drop_duplicates(subset=["RIC"], keep="last")
        if close_latest.empty:
            return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])

        merged = shares.merge(close_latest[["RIC", "Date", "CLOSE"]], on="RIC", how="inner")
        if merged.empty:
            return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])

        out = pd.DataFrame()
        out["RIC"] = merged["RIC"].astype(str)
        out["MarketCap"] = (
            pd.to_numeric(merged["SharesOutstanding"], errors="coerce")
            * pd.to_numeric(merged["CLOSE"], errors="coerce")
        )
        out["MarketCapDate"] = pd.to_datetime(merged["Date"], errors="coerce").fillna(
            pd.to_datetime(merged["SharesDate"], errors="coerce")
        )
        out = out.dropna(subset=["RIC", "MarketCap"])
        out = out.loc[out["MarketCap"] > 0]
        return out.reset_index(drop=True)

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _constituents_cached(self, as_of_date: str) -> tuple[bool, list[str]]:
        if not self._persistent_enabled():
            return False, []
        has_cov = self._persistent_conn.execute(
            "SELECT 1 FROM provider_cache_constituents_coverage WHERE as_of_date = ? LIMIT 1",
            (as_of_date,),
        ).fetchone()
        if has_cov is None:
            return False, []
        rows = self._persistent_conn.execute(
            "SELECT ric FROM provider_cache_constituents WHERE as_of_date = ? ORDER BY ric",
            (as_of_date,),
        ).fetchall()
        return True, [str(r[0]) for r in rows]

    def _store_constituents(self, as_of_date: str, rics: list[str]) -> None:
        if not self._persistent_enabled():
            return
        unique = sorted(set(rics))
        self._persistent_conn.execute(
            "INSERT OR REPLACE INTO provider_cache_constituents_coverage (as_of_date) VALUES (?)",
            (as_of_date,),
        )
        if unique:
            rows = [(as_of_date, ric) for ric in unique]
            self._persistent_conn.executemany(
                "INSERT OR REPLACE INTO provider_cache_constituents (as_of_date, ric) VALUES (?, ?)",
                rows,
            )
            self._stats["persistent_cache_writes"] += len(rows)
        self._persistent_conn.commit()

    def _load_market_caps(self, rics: list[str], as_of_date: str) -> pd.DataFrame:
        if not self._persistent_enabled() or not rics:
            return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])
        unique = sorted(set(rics))
        placeholders = ",".join(["?"] * len(unique))
        df = pd.read_sql_query(
            f"""
            SELECT ric AS RIC, market_cap AS MarketCap, market_cap_date AS MarketCapDate
            FROM provider_cache_market_caps
            WHERE as_of_date = ?
              AND ric IN ({placeholders})
            """,
            self._persistent_conn,
            params=[as_of_date, *unique],
        )
        if df.empty:
            return df
        df["MarketCap"] = pd.to_numeric(df["MarketCap"], errors="coerce")
        df["MarketCapDate"] = pd.to_datetime(df["MarketCapDate"], errors="coerce")
        return df.dropna(subset=["RIC", "MarketCap"]).reset_index(drop=True)

    def _store_market_caps(self, as_of_date: str, df: pd.DataFrame, queried_rics: list[str]) -> None:
        if not self._persistent_enabled():
            return
        out = df.copy()
        if not out.empty:
            out["RIC"] = out["RIC"].astype(str).str.strip()
            out["MarketCap"] = pd.to_numeric(out["MarketCap"], errors="coerce")
            out["MarketCapDate"] = pd.to_datetime(out["MarketCapDate"], errors="coerce")
            out = out.dropna(subset=["RIC", "MarketCap"])
            rows = [
                (
                    as_of_date,
                    str(r.RIC),
                    float(r.MarketCap),
                    pd.Timestamp(r.MarketCapDate).date().isoformat()
                    if pd.notna(r.MarketCapDate)
                    else as_of_date,
                )
                for r in out.itertuples(index=False)
            ]
            if rows:
                self._persistent_conn.executemany(
                    """
                    INSERT OR REPLACE INTO provider_cache_market_caps
                    (as_of_date, ric, market_cap, market_cap_date)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )
                self._stats["persistent_cache_writes"] += len(rows)
        self._persistent_conn.commit()
        self._mark_coverage("market_caps", queried_rics, as_of_date, as_of_date)

    def _load_series_from_cache(
        self,
        *,
        table: str,
        value_cols: list[str],
        rics: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not self._persistent_enabled() or not rics:
            cols = ["Date", "RIC", *value_cols]
            return pd.DataFrame(columns=cols)
        unique = sorted(set(rics))
        placeholders = ",".join(["?"] * len(unique))
        select_cols = ", ".join([f"{c} AS {c.upper()}" for c in value_cols])
        df = pd.read_sql_query(
            f"""
            SELECT date AS Date, ric AS RIC, {select_cols}
            FROM {table}
            WHERE ric IN ({placeholders})
              AND date >= ?
              AND date <= ?
            ORDER BY date, ric
            """,
            self._persistent_conn,
            params=[*unique, start_date, end_date],
        )
        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for c in value_cols:
            df[c.upper()] = pd.to_numeric(df[c.upper()], errors="coerce")
        return df.dropna(subset=["Date", "RIC"]).reset_index(drop=True)

    def _store_series_cache(
        self,
        *,
        table: str,
        value_cols: list[str],
        df: pd.DataFrame,
        dataset_key: str,
        queried_rics: list[str],
        start_date: str,
        end_date: str,
        covered_rics: list[str] | None = None,
    ) -> None:
        if not self._persistent_enabled():
            return
        out = df.copy()
        if not out.empty:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
            out["RIC"] = out["RIC"].astype(str).str.strip()
            out = out.dropna(subset=["Date", "RIC"])
            for c in value_cols:
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
                elif c.upper() in out.columns:
                    out[c.upper()] = pd.to_numeric(out[c.upper()], errors="coerce")
                elif c.capitalize() in out.columns:
                    out[c.capitalize()] = pd.to_numeric(out[c.capitalize()], errors="coerce")

            if table == "provider_cache_prices":
                rows = [
                    (str(r.RIC), pd.Timestamp(r.Date).date().isoformat(), float(r.CLOSE))
                    for r in out.itertuples(index=False)
                    if pd.notna(r.CLOSE)
                ]
                if rows:
                    self._persistent_conn.executemany(
                        "INSERT OR REPLACE INTO provider_cache_prices (ric, date, close) VALUES (?, ?, ?)",
                        rows,
                    )
                    self._stats["persistent_cache_writes"] += len(rows)
            elif table == "provider_cache_dividends":
                rows = [
                    (str(r.RIC), pd.Timestamp(r.Date).date().isoformat(), float(r.Dividend))
                    for r in out.itertuples(index=False)
                    if pd.notna(r.Dividend)
                ]
                if rows:
                    self._persistent_conn.executemany(
                        "INSERT OR REPLACE INTO provider_cache_dividends (ric, date, dividend) VALUES (?, ?, ?)",
                        rows,
                    )
                    self._stats["persistent_cache_writes"] += len(rows)
            elif table == "provider_cache_bid_ask":
                rows = [
                    (
                        str(r.RIC),
                        pd.Timestamp(r.Date).date().isoformat(),
                        float(r.BID) if pd.notna(r.BID) else None,
                        float(r.ASK) if pd.notna(r.ASK) else None,
                    )
                    for r in out.itertuples(index=False)
                ]
                if rows:
                    self._persistent_conn.executemany(
                        "INSERT OR REPLACE INTO provider_cache_bid_ask (ric, date, bid, ask) VALUES (?, ?, ?, ?)",
                        rows,
                    )
                    self._stats["persistent_cache_writes"] += len(rows)
        self._persistent_conn.commit()
        rics_to_cover = queried_rics if covered_rics is None else covered_rics
        self._mark_coverage(dataset_key, rics_to_cover, start_date, end_date)

    def get_sp500_constituents_as_of(self, as_of_date: str) -> list[str]:
        if self.enable_cache and as_of_date in self._constituents_cache:
            self._stats["constituents_cache_hits"] += 1
            return list(self._constituents_cache[as_of_date])

        cached_known, cached_values = self._constituents_cached(as_of_date)
        if cached_known:
            self._stats["persistent_cache_hits"] += 1
            if self.enable_cache:
                self._constituents_cache[as_of_date] = list(cached_values)
            return list(cached_values)
        self._stats["persistent_cache_misses"] += 1

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
        self._store_constituents(as_of_date, out)
        if self.enable_cache:
            self._constituents_cache[as_of_date] = out
        return out

    def get_market_cap_snapshot(self, rics: list[str], as_of_date: str) -> pd.DataFrame:
        key = (as_of_date, self._ric_key(rics))
        if self.enable_cache and key in self._market_cap_cache:
            self._stats["market_cap_cache_hits"] += 1
            return self._market_cap_cache[key].copy()

        unique_rics = sorted(set(str(r).strip() for r in rics if str(r).strip()))
        if not unique_rics:
            return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])

        covered = self._covered_rics("market_caps", unique_rics, as_of_date, as_of_date)
        missing = [ric for ric in unique_rics if ric not in covered]
        if not missing:
            cached = self._load_market_caps(unique_rics, as_of_date)
            if not cached.empty:
                self._stats["persistent_cache_hits"] += 1
                if self.enable_cache:
                    self._market_cap_cache[key] = cached
                return cached.copy()
            self._clear_coverage("market_caps", unique_rics, as_of_date, as_of_date)
            missing = unique_rics
        self._stats["persistent_cache_misses"] += 1

        self._stats["market_cap_calls"] += 1
        fields = ["TR.CompanyMarketCap", "TR.CompanyMarketCap.date"]
        params = {"SDate": as_of_date, "EDate": as_of_date}

        parts = self._collect_market_cap_parts(
            self._fetch_eikon_frames(rics=missing, fields=fields, params=params),
            as_of_date=as_of_date,
        )
        if not parts:
            try:
                parts = self._collect_market_cap_parts(
                    self._fetch_eikon_only_frames(rics=missing, fields=fields, params=params),
                    as_of_date=as_of_date,
                )
            except Exception:  # noqa: BLE001
                parts = []
        direct = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"])
        )
        present_rics = set(direct["RIC"].dropna().astype(str)) if not direct.empty else set()
        residual = [ric for ric in missing if ric not in present_rics]
        if residual:
            fallback = self._build_market_cap_from_close_and_shares(residual, as_of_date)
            if not fallback.empty:
                direct = pd.concat([direct, fallback], ignore_index=True)

        if direct.empty:
            self._store_market_caps(as_of_date, pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"]), missing)
            out = self._load_market_caps(unique_rics, as_of_date)
            if self.enable_cache:
                self._market_cap_cache[key] = out
            return out.copy()

        out = direct.copy()
        out = out.sort_values(["RIC", "MarketCapDate"]).drop_duplicates(subset=["RIC"], keep="last")
        out = out.reset_index(drop=True)
        self._store_market_caps(as_of_date, out, missing)
        out = self._load_market_caps(unique_rics, as_of_date)
        if self.enable_cache:
            self._market_cap_cache[key] = out
        return out.copy()

    def get_dividend_events(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        key = (start_date, end_date, self._ric_key(rics))
        if self.enable_cache and key in self._dividend_cache:
            self._stats["dividend_cache_hits"] += 1
            return self._dividend_cache[key].copy()

        unique_rics = sorted(set(str(r).strip() for r in rics if str(r).strip()))
        if not unique_rics:
            return pd.DataFrame(columns=["RIC", "Date", "Dividend"])

        covered = self._covered_rics("dividends", unique_rics, start_date, end_date)
        missing = [ric for ric in unique_rics if ric not in covered]
        if not missing:
            cached = self._load_series_from_cache(
                table="provider_cache_dividends",
                value_cols=["dividend"],
                rics=unique_rics,
                start_date=start_date,
                end_date=end_date,
            )
            out = (
                cached.rename(columns={"DIVIDEND": "Dividend"})[["RIC", "Date", "Dividend"]]
                if not cached.empty
                else pd.DataFrame(columns=["RIC", "Date", "Dividend"])
            )
            out = out.dropna(subset=["RIC", "Date", "Dividend"])
            out = out.loc[pd.to_numeric(out["Dividend"], errors="coerce") > 0].reset_index(drop=True)
            self._stats["persistent_cache_hits"] += 1
            if self.enable_cache:
                self._dividend_cache[key] = out
            return out.copy()
        self._stats["persistent_cache_misses"] += 1

        fetch_plan = self._missing_ranges_by_window(
            dataset="dividends",
            rics=missing,
            start_date=start_date,
            end_date=end_date,
        )
        fields = [
            "TR.DivExDate",
            "TR.DivUnadjustedGross",
            "TR.FundExDate",
            "TR.FundDiv",
        ]

        def collect_dividend_parts(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
            parts_local: list[pd.DataFrame] = []
            for df in frames:
                if df.empty:
                    continue

                ric_col = _pick_col(df, ("instrument",))
                date_col = _pick_col(df, ("ex", "date")) or _pick_col(df, ("date",))
                amt_col = (
                    _pick_col(df, ("gross",))
                    or _pick_col(df, ("fund", "div"))
                    or _pick_col(df, ("dividend",))
                    or _pick_col(df, ("div", "amount"))
                )
                if ric_col is None or date_col is None or amt_col is None:
                    continue

                out = pd.DataFrame()
                out["RIC"] = df[ric_col].astype(str).str.strip()
                out["Date"] = pd.to_datetime(df[date_col], errors="coerce")
                out["Dividend"] = pd.to_numeric(df[amt_col], errors="coerce")
                out = out.dropna(subset=["RIC", "Date", "Dividend"])
                out = out.loc[out["Dividend"] > 0]
                if not out.empty:
                    parts_local.append(out)
            return parts_local

        for (req_start, req_end), req_rics in fetch_plan.items():
            if not req_rics:
                continue
            self._stats["dividend_calls"] += 1
            params = {"SDate": req_start, "EDate": req_end, "DateType": "ED"}
            parts = collect_dividend_parts(self._fetch_eikon_frames(rics=req_rics, fields=fields, params=params))
            if not parts:
                try:
                    parts = collect_dividend_parts(
                        self._fetch_eikon_only_frames(rics=req_rics, fields=fields, params=params)
                    )
                except Exception:  # noqa: BLE001
                    parts = []
            fetched = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["RIC", "Date", "Dividend"])
            self._store_series_cache(
                table="provider_cache_dividends",
                value_cols=["dividend"],
                df=fetched,
                dataset_key="dividends",
                queried_rics=req_rics,
                start_date=req_start,
                end_date=req_end,
            )
        cached = self._load_series_from_cache(
            table="provider_cache_dividends",
            value_cols=["dividend"],
            rics=unique_rics,
            start_date=start_date,
            end_date=end_date,
        )
        out = (
            cached.rename(columns={"DIVIDEND": "Dividend"})[["RIC", "Date", "Dividend"]]
            if not cached.empty
            else pd.DataFrame(columns=["RIC", "Date", "Dividend"])
        )
        out = out.dropna(subset=["RIC", "Date", "Dividend"])
        out = out.loc[pd.to_numeric(out["Dividend"], errors="coerce") > 0].reset_index(drop=True)
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

        out = pd.DataFrame(index=df.index)
        out["Date"] = df.index
        if "TRDPRC_1" in df.columns:
            out["CLOSE"] = pd.to_numeric(df["TRDPRC_1"], errors="coerce").to_numpy()
            out["RIC"] = rics[0] if rics else ""
        elif "CLOSE" in df.columns:
            out["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce").to_numpy()
            out["RIC"] = rics[0] if rics else ""
        else:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                return pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
            out["CLOSE"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce").to_numpy()
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

        unique_rics = sorted(set(str(r).strip() for r in rics if str(r).strip()))
        covered = self._covered_rics("prices", unique_rics, start_date, end_date)
        missing = [ric for ric in unique_rics if ric not in covered]
        if not missing:
            cached = self._load_series_from_cache(
                table="provider_cache_prices",
                value_cols=["close"],
                rics=unique_rics,
                start_date=start_date,
                end_date=end_date,
            )
            out = (
                cached.rename(columns={"CLOSE": "CLOSE"})[["Date", "RIC", "CLOSE"]]
                if not cached.empty
                else pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
            )
            out = out.dropna(subset=["Date", "RIC"]).sort_values(["Date", "RIC"]).reset_index(drop=True)
            if out.empty and self._persistent_enabled():
                # Coverage with zero cached rows indicates stale/poisoned metadata; force refetch.
                self._clear_coverage("prices", unique_rics, start_date, end_date)
                missing = list(unique_rics)
            else:
                self._stats["persistent_cache_hits"] += 1
                if self.enable_cache:
                    self._close_cache[key] = out
                return out.copy()
        if missing:
            self._stats["persistent_cache_misses"] += 1

        fetch_plan = self._missing_ranges_by_window(
            dataset="prices",
            rics=missing,
            start_date=start_date,
            end_date=end_date,
        )

        for (req_start, req_end), req_rics in fetch_plan.items():
            if not req_rics:
                continue
            self._stats["close_calls"] += 1
            parts: list[pd.DataFrame] = []
            covered_this_window: set[str] = set()
            for chunk in _chunked(req_rics, self.batch_size):
                chunk_failed = False
                try:
                    raw = self.client.get_history(
                        universe=chunk,
                        fields=["TRDPRC_1"],
                        interval="daily",
                        start=req_start,
                        end=req_end,
                        adjustments="unadjusted",
                    )
                    chunk_out = self._close_history_to_long(raw, chunk)
                except Exception:  # noqa: BLE001
                    chunk_out = pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
                    chunk_failed = True

                if chunk_out.empty and len(chunk) > 1:
                    # Some deprecated/renamed RICs can null the whole batch response.
                    for ric in chunk:
                        try:
                            raw_one = self.client.get_history(
                                universe=[ric],
                                fields=["TRDPRC_1"],
                                interval="daily",
                                start=req_start,
                                end=req_end,
                                adjustments="unadjusted",
                            )
                            one_out = self._close_history_to_long(raw_one, [ric])
                        except Exception:  # noqa: BLE001
                            continue
                        covered_this_window.add(str(ric))
                        if not one_out.empty:
                            parts.append(one_out)
                    continue

                if not chunk_out.empty:
                    parts.append(chunk_out)
                    covered_this_window.update(chunk_out["RIC"].dropna().astype(str))
                elif len(chunk) == 1 and not chunk_failed:
                    # A successful single-name query with no rows is still a covered empty range.
                    covered_this_window.add(str(chunk[0]))

            fetched = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
            self._store_series_cache(
                table="provider_cache_prices",
                value_cols=["close"],
                df=fetched,
                dataset_key="prices",
                queried_rics=req_rics,
                start_date=req_start,
                end_date=req_end,
                covered_rics=sorted(covered_this_window),
            )
        out = self._load_series_from_cache(
            table="provider_cache_prices",
            value_cols=["close"],
            rics=unique_rics,
            start_date=start_date,
            end_date=end_date,
        )
        if out.empty:
            out = pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
        else:
            out = out.rename(columns={"CLOSE": "CLOSE"})[["Date", "RIC", "CLOSE"]]
            out = out.dropna(subset=["Date"]).sort_values(["Date", "RIC"]).reset_index(drop=True)
            out = out.drop_duplicates(subset=["Date", "RIC"], keep="last")
        if self.enable_cache:
            self._close_cache[key] = out
        return out.copy()

    @staticmethod
    def _bid_ask_history_to_long(raw: pd.DataFrame, rics: list[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["Date", "RIC", "BID", "ASK"])

        df = raw.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

        df = df.loc[df.index.notna()].sort_index()
        if df.empty:
            return pd.DataFrame(columns=["Date", "RIC", "BID", "ASK"])

        if isinstance(df.columns, pd.MultiIndex):
            rows: list[pd.DataFrame] = []
            for ric in rics:
                bid_col = None
                ask_col = None
                for col in df.columns:
                    if len(col) < 2:
                        continue
                    a, b = str(col[0]), str(col[1])
                    if a == ric and b.upper() == "BID":
                        bid_col = col
                    elif a == ric and b.upper() == "ASK":
                        ask_col = col
                    elif b == ric and a.upper() == "BID":
                        bid_col = col
                    elif b == ric and a.upper() == "ASK":
                        ask_col = col
                if bid_col is None and ask_col is None:
                    continue
                part = pd.DataFrame({"Date": df.index, "RIC": ric})
                part["BID"] = pd.to_numeric(df[bid_col], errors="coerce") if bid_col is not None else pd.NA
                part["ASK"] = pd.to_numeric(df[ask_col], errors="coerce") if ask_col is not None else pd.NA
                rows.append(part)

            if not rows:
                return pd.DataFrame(columns=["Date", "RIC", "BID", "ASK"])
            out = pd.concat(rows, ignore_index=True)
            out = out.dropna(subset=["Date", "RIC"], how="any")
            return out[["Date", "RIC", "BID", "ASK"]]

        out = pd.DataFrame(index=df.index)
        out["Date"] = df.index
        if "BID" in df.columns:
            out["BID"] = pd.to_numeric(df["BID"], errors="coerce").to_numpy()
        else:
            out["BID"] = pd.NA
        if "ASK" in df.columns:
            out["ASK"] = pd.to_numeric(df["ASK"], errors="coerce").to_numpy()
        else:
            out["ASK"] = pd.NA
        out["RIC"] = rics[0] if rics else ""
        out = out.dropna(subset=["Date", "RIC"], how="any")
        return out[["Date", "RIC", "BID", "ASK"]]

    def get_bid_ask_history(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if not rics:
            return pd.DataFrame(columns=["Date", "RIC", "BID", "ASK"])
        key = (start_date, end_date, self._ric_key(rics))
        if self.enable_cache and key in self._bid_ask_cache:
            self._stats["bid_ask_cache_hits"] += 1
            return self._bid_ask_cache[key].copy()

        unique_rics = sorted(set(str(r).strip() for r in rics if str(r).strip()))
        covered = self._covered_rics("bid_ask", unique_rics, start_date, end_date)
        missing = [ric for ric in unique_rics if ric not in covered]
        if not missing:
            cached = self._load_series_from_cache(
                table="provider_cache_bid_ask",
                value_cols=["bid", "ask"],
                rics=unique_rics,
                start_date=start_date,
                end_date=end_date,
            )
            out = (
                cached.rename(columns={"BID": "BID", "ASK": "ASK"})[["Date", "RIC", "BID", "ASK"]]
                if not cached.empty
                else pd.DataFrame(columns=["Date", "RIC", "BID", "ASK"])
            )
            out = out.dropna(subset=["Date", "RIC"]).sort_values(["Date", "RIC"]).reset_index(drop=True)
            self._stats["persistent_cache_hits"] += 1
            if self.enable_cache:
                self._bid_ask_cache[key] = out
            return out.copy()
        self._stats["persistent_cache_misses"] += 1

        fetch_plan = self._missing_ranges_by_window(
            dataset="bid_ask",
            rics=missing,
            start_date=start_date,
            end_date=end_date,
        )

        for (req_start, req_end), req_rics in fetch_plan.items():
            if not req_rics:
                continue
            self._stats["bid_ask_calls"] += 1
            parts: list[pd.DataFrame] = []
            covered_this_window: set[str] = set()
            for chunk in _chunked(req_rics, self.batch_size):
                chunk_failed = False
                try:
                    raw = self.client.get_history(
                        universe=chunk,
                        fields=["BID", "ASK"],
                        interval="daily",
                        start=req_start,
                        end=req_end,
                        adjustments="unadjusted",
                    )
                    chunk_out = self._bid_ask_history_to_long(raw, chunk)
                except Exception:  # noqa: BLE001
                    chunk_out = pd.DataFrame(columns=["Date", "RIC", "BID", "ASK"])
                    chunk_failed = True

                if chunk_out.empty and len(chunk) > 1:
                    for ric in chunk:
                        try:
                            raw_one = self.client.get_history(
                                universe=[ric],
                                fields=["BID", "ASK"],
                                interval="daily",
                                start=req_start,
                                end=req_end,
                                adjustments="unadjusted",
                            )
                            one_out = self._bid_ask_history_to_long(raw_one, [ric])
                        except Exception:  # noqa: BLE001
                            continue
                        covered_this_window.add(str(ric))
                        if not one_out.empty:
                            parts.append(one_out)
                    continue

                if not chunk_out.empty:
                    parts.append(chunk_out)
                    covered_this_window.update(chunk_out["RIC"].dropna().astype(str))
                elif len(chunk) == 1 and not chunk_failed:
                    covered_this_window.add(str(chunk[0]))

            fetched = (
                pd.concat(parts, ignore_index=True)
                if parts
                else pd.DataFrame(columns=["Date", "RIC", "BID", "ASK"])
            )
            self._store_series_cache(
                table="provider_cache_bid_ask",
                value_cols=["bid", "ask"],
                df=fetched,
                dataset_key="bid_ask",
                queried_rics=req_rics,
                start_date=req_start,
                end_date=req_end,
                covered_rics=sorted(covered_this_window),
            )
        out = self._load_series_from_cache(
            table="provider_cache_bid_ask",
            value_cols=["bid", "ask"],
            rics=unique_rics,
            start_date=start_date,
            end_date=end_date,
        )
        if out.empty:
            out = pd.DataFrame(columns=["Date", "RIC", "BID", "ASK"])
        else:
            out = out.rename(columns={"BID": "BID", "ASK": "ASK"})[["Date", "RIC", "BID", "ASK"]]
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
            out["BID"] = pd.to_numeric(out["BID"], errors="coerce")
            out["ASK"] = pd.to_numeric(out["ASK"], errors="coerce")
            out = out.dropna(subset=["Date", "RIC"]).sort_values(["Date", "RIC"]).reset_index(drop=True)
            out = out.drop_duplicates(subset=["Date", "RIC"], keep="last")
        if self.enable_cache:
            self._bid_ask_cache[key] = out
        return out.copy()


def trailing_lookback_start(as_of_date: str, months: int) -> str:
    as_of = pd.Timestamp(as_of_date)
    start = as_of - pd.DateOffset(months=months)
    return start.date().isoformat()


def utc_now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
