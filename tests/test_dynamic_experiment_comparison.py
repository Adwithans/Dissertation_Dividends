from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dividend_portfolio.cli.run_dynamic_strategy import (
    _normalize_experiment_group,
    _upsert_experiment_comparison,
)


def _build_summary(*, run_id: str, total_return: float, trade_count: int) -> dict:
    return {
        "run_id": run_id,
        "created_at_utc": "2026-03-12T12:00:00+00:00",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "portfolio_metrics": {
            "total_return": total_return,
            "cagr": 0.12,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.2,
            "end_value": 1100000.0,
        },
        "trading_activity": {
            "gross_turnover": 12345.0,
            "number_of_trades": trade_count,
        },
        "transaction_costs": {
            "total_transaction_cost": 456.0,
        },
        "dividends": {
            "total_dividend_cash": 123.0,
            "dividend_share_of_total_gain": 0.25,
        },
        "objective_metrics": {
            "cagr": 0.12,
            "sortino_ratio": 0.9,
            "calmar_ratio": 0.6,
            "information_ratio": 0.4,
            "annualized_excess_return": 0.03,
        },
        "constraint_metrics": {
            "max_drawdown": -0.2,
            "tracking_error_annualized": 0.1,
            "gross_turnover": 12345.0,
            "total_transaction_cost": 456.0,
            "cost_drag_pct_of_start_value": 0.004,
        },
        "hyperparameters": {
            "portfolio_size": 25,
            "rebalance_interval_quarters": 2,
            "allocation_strategy": "yield_proportional",
        },
        "strategy": {
            "portfolio_size": 25,
            "rebalance_interval_quarters": 2,
            "allocation_strategy": "yield_proportional",
            "selection_policy_name": "replace_bottom_n",
            "max_replacements_per_quarter": 10,
        },
    }


def test_upsert_experiment_comparison_writes_csv_and_json(tmp_path: Path) -> None:
    summary = _build_summary(run_id="run_1", total_return=0.15, trade_count=100)
    csv_path, json_path = _upsert_experiment_comparison(
        summary=summary,
        experiment_group="bottom10_test",
        base_dir=tmp_path / "experiments",
    )

    assert csv_path.exists()
    assert json_path.exists()
    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) == 1
    assert csv_df.iloc[0]["run_id"] == "run_1"
    assert float(csv_df.iloc[0]["total_return"]) == 0.15

    records = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["run_id"] == "run_1"
    assert records[0]["policy_name"] == "replace_bottom_n"
    assert int(records[0]["rebalance_interval_quarters"]) == 2
    assert int(records[0]["portfolio_size"]) == 25
    assert records[0]["allocation_strategy"] == "yield_proportional"


def test_upsert_experiment_comparison_is_idempotent_by_run_id(tmp_path: Path) -> None:
    base_dir = tmp_path / "experiments"
    group = "bottom10_test"
    _upsert_experiment_comparison(
        summary=_build_summary(run_id="run_1", total_return=0.15, trade_count=100),
        experiment_group=group,
        base_dir=base_dir,
    )
    csv_path, json_path = _upsert_experiment_comparison(
        summary=_build_summary(run_id="run_1", total_return=0.22, trade_count=130),
        experiment_group=group,
        base_dir=base_dir,
    )

    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) == 1
    assert float(csv_df.iloc[0]["total_return"]) == 0.22
    assert int(csv_df.iloc[0]["trade_count"]) == 130

    records = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert float(records[0]["total_return"]) == 0.22


def test_normalize_experiment_group_sanitizes_invalid_chars() -> None:
    assert _normalize_experiment_group("  my group / 2026  ") == "my_group___2026"
