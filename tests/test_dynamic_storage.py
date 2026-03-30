from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dividend_portfolio.strategy.storage import StrategyStore


def test_sqlite_schema_idempotent_insert_and_csv_export_parity(tmp_path: Path) -> None:
    db_path = tmp_path / "strategy.sqlite"
    store = StrategyStore(db_path)

    run_id = "run_test_1"
    store.write_run_metadata(
        run_id=run_id,
        created_at_utc="2026-03-06T00:00:00",
        start_date="2024-01-01",
        end_date="2024-03-31",
        config={"strategy": "dynamic"},
    )
    assert store.run_exists(run_id)
    assert not store.run_exists("missing_run")

    candidate = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "quarter": "2024Q1",
                "as_of_date": "2024-01-02",
                "ric": "A",
                "market_cap": 100.0,
                "market_cap_date": "2024-01-02",
                "is_dividend_payer_12m": 1,
                "rank_market_cap": 1,
            }
        ]
    )
    store.upsert_candidate_universe(candidate)
    store.upsert_candidate_universe(candidate)

    count_df = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM candidate_universe WHERE run_id = ?",
        store.conn,
        params=[run_id],
    )
    assert int(count_df.iloc[0]["n"]) == 1

    trade_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "date": "2024-01-02",
                "quarter": "2024Q1",
                "ric": "A",
                "price": 100.0,
                "trade_shares": 1.0,
                "trade_value": 100.0,
                "reason": "test_trade",
            }
        ]
    )
    store.upsert_trades(trade_df)

    portfolio_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "date": "2024-01-02",
                "quarter": "2024Q1",
                "portfolio_market_value": 100.0,
                "portfolio_cash_balance": 0.0,
                "portfolio_total_value": 100.0,
                "portfolio_dividend_cash_daily": 0.0,
                "rebalance_flag": 1,
            }
        ]
    )
    store.upsert_portfolio_daily(portfolio_df)

    trade_cols = {
        row[1] for row in store.conn.execute("PRAGMA table_info(trades)").fetchall()
    }
    assert "total_transaction_cost" in trade_cols
    assert "execution_price" in trade_cols
    pcols = {
        row[1] for row in store.conn.execute("PRAGMA table_info(portfolio_daily)").fetchall()
    }
    assert "portfolio_transaction_cost_daily" in pcols
    assert "portfolio_total_value_gross" in pcols

    out_dir = tmp_path / "exports"
    store.export_run_csv(run_id, out_dir)
    exported = pd.read_csv(out_dir / "candidate_universe.csv")
    assert len(exported) == 1
    assert exported.iloc[0]["ric"] == "A"
    exported_trades = pd.read_csv(out_dir / "trades.csv")
    assert "total_transaction_cost" in exported_trades.columns
    assert float(exported_trades["total_transaction_cost"].iloc[0]) == 0.0
    assert store.row_count("candidate_universe", run_id) == 1

    store.close()
