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

    out_dir = tmp_path / "exports"
    store.export_run_csv(run_id, out_dir)
    exported = pd.read_csv(out_dir / "candidate_universe.csv")
    assert len(exported) == 1
    assert exported.iloc[0]["ric"] == "A"
    assert store.row_count("candidate_universe", run_id) == 1

    store.close()
