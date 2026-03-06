from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from datetime import date, datetime

import pandas as pd


class StrategyStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self.ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_metadata (
                run_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                config_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidate_universe (
                run_id TEXT NOT NULL,
                quarter TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                ric TEXT NOT NULL,
                market_cap REAL,
                market_cap_date TEXT,
                is_dividend_payer_12m INTEGER NOT NULL DEFAULT 1,
                rank_market_cap INTEGER NOT NULL,
                PRIMARY KEY (run_id, quarter, ric)
            );

            CREATE TABLE IF NOT EXISTS quarter_scores (
                run_id TEXT NOT NULL,
                quarter TEXT NOT NULL,
                quarter_start TEXT NOT NULL,
                quarter_end TEXT NOT NULL,
                ric TEXT NOT NULL,
                avg_close REAL,
                dividend_sum_ps REAL,
                score REAL,
                rank_score INTEGER,
                PRIMARY KEY (run_id, quarter, ric)
            );

            CREATE TABLE IF NOT EXISTS target_weights (
                run_id TEXT NOT NULL,
                quarter TEXT NOT NULL,
                rebalance_date TEXT NOT NULL,
                source TEXT NOT NULL,
                ric TEXT NOT NULL,
                weight REAL NOT NULL,
                rank_in_portfolio INTEGER NOT NULL,
                PRIMARY KEY (run_id, quarter, ric)
            );

            CREATE TABLE IF NOT EXISTS trades (
                run_id TEXT NOT NULL,
                date TEXT NOT NULL,
                quarter TEXT NOT NULL,
                ric TEXT NOT NULL,
                price REAL,
                trade_shares REAL,
                trade_value REAL,
                reason TEXT NOT NULL,
                PRIMARY KEY (run_id, date, quarter, ric)
            );

            CREATE TABLE IF NOT EXISTS holdings_daily (
                run_id TEXT NOT NULL,
                date TEXT NOT NULL,
                quarter TEXT NOT NULL,
                ric TEXT NOT NULL,
                shares REAL,
                close REAL,
                market_value REAL,
                cash_balance REAL,
                dividend_cash_daily REAL,
                total_value REAL,
                weight_eod REAL,
                PRIMARY KEY (run_id, date, ric)
            );

            CREATE TABLE IF NOT EXISTS portfolio_daily (
                run_id TEXT NOT NULL,
                date TEXT NOT NULL,
                quarter TEXT NOT NULL,
                portfolio_market_value REAL,
                portfolio_cash_balance REAL,
                portfolio_total_value REAL,
                portfolio_dividend_cash_daily REAL,
                rebalance_flag INTEGER NOT NULL,
                PRIMARY KEY (run_id, date)
            );
            """
        )
        self.conn.commit()

    def run_exists(self, run_id: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM run_metadata WHERE run_id = ? LIMIT 1",
            (run_id,),
        )
        return cur.fetchone() is not None

    def row_count(self, table: str, run_id: str) -> int:
        allowed = {
            "candidate_universe",
            "quarter_scores",
            "target_weights",
            "trades",
            "holdings_daily",
            "portfolio_daily",
        }
        if table not in allowed:
            raise ValueError(f"Unsupported table for row_count: {table}")
        cur = self.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def write_run_metadata(
        self,
        *,
        run_id: str,
        created_at_utc: str,
        start_date: str,
        end_date: str,
        config: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO run_metadata (
                run_id, created_at_utc, start_date, end_date, config_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, created_at_utc, start_date, end_date, json.dumps(config, sort_keys=True, default=str)),
        )
        self.conn.commit()

    def _upsert_df(self, table: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df.to_sql(table, self.conn, if_exists="append", index=False, method="multi")

    @staticmethod
    def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        def _norm_value(v: Any) -> Any:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            if isinstance(v, pd.Timestamp):
                return v.isoformat()
            if isinstance(v, datetime):
                return v.isoformat()
            if isinstance(v, date):
                return v.isoformat()
            return v

        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].apply(_norm_value)
            elif out[col].dtype == "object":
                out[col] = out[col].apply(_norm_value)
        return out.to_dict(orient="records")

    def upsert_candidate_universe(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = self._to_records(df)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO candidate_universe (
                run_id, quarter, as_of_date, ric, market_cap, market_cap_date,
                is_dividend_payer_12m, rank_market_cap
            ) VALUES (
                :run_id, :quarter, :as_of_date, :ric, :market_cap, :market_cap_date,
                :is_dividend_payer_12m, :rank_market_cap
            )
            """,
            rows,
        )
        self.conn.commit()

    def upsert_quarter_scores(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = self._to_records(df)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO quarter_scores (
                run_id, quarter, quarter_start, quarter_end, ric, avg_close, dividend_sum_ps, score, rank_score
            ) VALUES (
                :run_id, :quarter, :quarter_start, :quarter_end, :ric, :avg_close, :dividend_sum_ps, :score, :rank_score
            )
            """,
            rows,
        )
        self.conn.commit()

    def upsert_target_weights(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = self._to_records(df)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO target_weights (
                run_id, quarter, rebalance_date, source, ric, weight, rank_in_portfolio
            ) VALUES (
                :run_id, :quarter, :rebalance_date, :source, :ric, :weight, :rank_in_portfolio
            )
            """,
            rows,
        )
        self.conn.commit()

    def upsert_trades(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = self._to_records(df)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO trades (
                run_id, date, quarter, ric, price, trade_shares, trade_value, reason
            ) VALUES (
                :run_id, :date, :quarter, :ric, :price, :trade_shares, :trade_value, :reason
            )
            """,
            rows,
        )
        self.conn.commit()

    def upsert_holdings_daily(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = self._to_records(df)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO holdings_daily (
                run_id, date, quarter, ric, shares, close, market_value,
                cash_balance, dividend_cash_daily, total_value, weight_eod
            ) VALUES (
                :run_id, :date, :quarter, :ric, :shares, :close, :market_value,
                :cash_balance, :dividend_cash_daily, :total_value, :weight_eod
            )
            """,
            rows,
        )
        self.conn.commit()

    def upsert_portfolio_daily(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = self._to_records(df)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO portfolio_daily (
                run_id, date, quarter, portfolio_market_value, portfolio_cash_balance,
                portfolio_total_value, portfolio_dividend_cash_daily, rebalance_flag
            ) VALUES (
                :run_id, :date, :quarter, :portfolio_market_value, :portfolio_cash_balance,
                :portfolio_total_value, :portfolio_dividend_cash_daily, :rebalance_flag
            )
            """,
            rows,
        )
        self.conn.commit()

    def export_run_csv(self, run_id: str, output_dir: str | Path) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tables = [
            "candidate_universe",
            "quarter_scores",
            "target_weights",
            "trades",
            "holdings_daily",
            "portfolio_daily",
        ]
        for table in tables:
            df = pd.read_sql_query(
                f"SELECT * FROM {table} WHERE run_id = ? ORDER BY 1, 2, 3",
                self.conn,
                params=[run_id],
            )
            df.to_csv(out_dir / f"{table}.csv", index=False)

        meta = pd.read_sql_query(
            "SELECT * FROM run_metadata WHERE run_id = ?",
            self.conn,
            params=[run_id],
        )
        meta.to_csv(out_dir / "run_metadata.csv", index=False)
        return out_dir
