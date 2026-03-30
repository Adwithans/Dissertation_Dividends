from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.dividend_portfolio.models import (
    AssetConfig,
    EvaluationContext,
    PortfolioConfig,
    QuarterlyMetricsConfig,
    RebalanceConfig,
    StrategyConfig,
)
from src.dividend_portfolio.strategy.evaluation import evaluate_strategy
import src.dividend_portfolio.strategy.evaluation as evaluation_module


@dataclass
class FakeProvider:
    rics: list[str]
    market_caps: dict[str, float]
    prices: pd.DataFrame
    dividends: pd.DataFrame

    def get_sp500_constituents_as_of(self, as_of_date: str) -> list[str]:
        return list(self.rics)

    def get_market_cap_snapshot(self, rics: list[str], as_of_date: str) -> pd.DataFrame:
        rows = []
        ts = pd.Timestamp(as_of_date)
        for ric in rics:
            if ric in self.market_caps:
                rows.append({"RIC": ric, "MarketCap": self.market_caps[ric], "MarketCapDate": ts})
        return pd.DataFrame(rows)

    def get_dividend_events(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        out = self.dividends.copy()
        out = out.loc[out["RIC"].isin(rics)]
        out = out.loc[(out["Date"] >= pd.Timestamp(start_date)) & (out["Date"] <= pd.Timestamp(end_date))]
        return out.reset_index(drop=True)

    def get_close_history(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        out = self.prices.copy()
        out = out.loc[out["RIC"].isin(rics)]
        out = out.loc[(out["Date"] >= pd.Timestamp(start_date)) & (out["Date"] <= pd.Timestamp(end_date))]
        return out.reset_index(drop=True)

    def get_bid_ask_history(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        out = self.prices.copy()
        out = out.loc[out["RIC"].isin(rics)]
        out = out.loc[(out["Date"] >= pd.Timestamp(start_date)) & (out["Date"] <= pd.Timestamp(end_date))]
        out = out[["Date", "RIC", "CLOSE"]].copy()
        out["BID"] = out["CLOSE"] * 0.999
        out["ASK"] = out["CLOSE"] * 1.001
        return out[["Date", "RIC", "BID", "ASK"]].reset_index(drop=True)

    def close(self) -> None:
        return


def _build_fake_provider() -> FakeProvider:
    rics = ["A", "B", "C", "D"]
    market_caps = {"A": 400.0, "B": 300.0, "C": 200.0, "D": 100.0}

    dates = pd.bdate_range("2024-01-02", "2024-09-30")
    price_rows = []
    for ric in rics:
        for dt in dates:
            price_rows.append({"Date": dt, "RIC": ric, "CLOSE": 100.0})
    prices = pd.DataFrame(price_rows)

    dividends = pd.DataFrame(
        [
            {"Date": pd.Timestamp("2023-12-15"), "RIC": "A", "Dividend": 0.5},
            {"Date": pd.Timestamp("2023-12-15"), "RIC": "B", "Dividend": 0.5},
            {"Date": pd.Timestamp("2023-12-15"), "RIC": "C", "Dividend": 0.5},
            {"Date": pd.Timestamp("2023-12-15"), "RIC": "D", "Dividend": 0.5},
            {"Date": pd.Timestamp("2024-02-15"), "RIC": "A", "Dividend": 2.0},
            {"Date": pd.Timestamp("2024-02-15"), "RIC": "B", "Dividend": 1.0},
            {"Date": pd.Timestamp("2024-02-15"), "RIC": "C", "Dividend": 3.0},
            {"Date": pd.Timestamp("2024-05-15"), "RIC": "A", "Dividend": 1.0},
            {"Date": pd.Timestamp("2024-05-15"), "RIC": "B", "Dividend": 4.0},
            {"Date": pd.Timestamp("2024-05-15"), "RIC": "C", "Dividend": 2.0},
            {"Date": pd.Timestamp("2024-05-15"), "RIC": "D", "Dividend": 3.0},
        ]
    )
    return FakeProvider(rics=rics, market_caps=market_caps, prices=prices, dividends=dividends)


def test_evaluate_strategy_summary_only_returns_compact_metrics(tmp_path: Path) -> None:
    cfg = PortfolioConfig(
        base_currency="USD",
        initial_capital=1000.0,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 9, 30),
        reinvest_dividends=False,
        auto_align_splits=True,
        use_cum_factor=True,
        risk_free_rate=0.0,
        rebalancing=RebalanceConfig(
            enabled=True,
            frequency="quarterly",
            trigger="first_trading_day_after_quarter_end",
            drift_tolerance=0.0,
        ),
        quarterly_metrics=QuarterlyMetricsConfig(
            enabled=True,
            dividend_return_basis="quarter_start_market_value",
        ),
        assets=[AssetConfig("A", 1.0)],
        strategy=StrategyConfig(
            mode="dynamic_100_25",
            candidate_count=4,
            portfolio_size=2,
            rebalance_interval_quarters=1,
            allocation_strategy="yield_rank_linear",
            sqlite_path=str(tmp_path / "dyn.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
        ),
    )
    provider = _build_fake_provider()
    benchmark_dates = pd.bdate_range("2024-01-02", "2024-09-30")
    benchmark_close = pd.DataFrame(
        {
            "Date": benchmark_dates,
            "RIC": [".SPX"] * len(benchmark_dates),
            "CLOSE": [100.0 + idx for idx in range(len(benchmark_dates))],
        }
    )

    result = evaluate_strategy(
        base_config=cfg,
        persist="summary",
        benchmark="sp500",
        evaluation_context=EvaluationContext(
            study_id="study_1",
            trial_id="trial_1",
            search_space_id="space_a",
            trial_index=3,
        ),
        provider=provider,
        benchmark_close=benchmark_close,
        sp500_close=benchmark_close.copy(),
        russell_1000_close=benchmark_close.copy(),
        output_dir=tmp_path / "summary_only",
    )

    assert result.hyperparameters["portfolio_size"] == 2
    assert result.hyperparameters["rebalance_interval_quarters"] == 1
    assert result.hyperparameters["allocation_strategy"] == "yield_rank_linear"
    assert "cagr" in result.objective_metrics
    assert "max_drawdown" in result.constraint_metrics
    assert "total_dividend_cash" in result.diagnostic_metrics
    assert Path(result.persisted_artifacts["summary_json"]).exists()
    assert result.summary["evaluation_context"]["study_id"] == "study_1"
    assert result.summary["dsr_readiness"]["trial_id"] == "trial_1"
    assert result.summary["dsr_readiness"]["ready_for_deflated_sharpe"] is True


def test_evaluate_strategy_enables_persistent_provider_cache_by_default(monkeypatch, tmp_path: Path) -> None:
    cfg = PortfolioConfig(
        base_currency="USD",
        initial_capital=1000.0,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 3, 31),
        reinvest_dividends=False,
        auto_align_splits=True,
        use_cum_factor=True,
        risk_free_rate=0.0,
        rebalancing=RebalanceConfig(
            enabled=True,
            frequency="quarterly",
            trigger="first_trading_day_after_quarter_end",
            drift_tolerance=0.0,
        ),
        quarterly_metrics=QuarterlyMetricsConfig(
            enabled=True,
            dividend_return_basis="quarter_start_market_value",
        ),
        assets=[AssetConfig("A", 1.0)],
        strategy=StrategyConfig(
            mode="dynamic_100_25",
            candidate_count=4,
            portfolio_size=2,
            sqlite_path=str(tmp_path / "dyn.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
        ),
    )

    captured: dict[str, object] = {}

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    class DummyProvider:
        def __init__(self, client, **kwargs):
            captured.update(kwargs)

        def close(self) -> None:
            return

    class DummyRunResult:
        portfolio_daily = pd.DataFrame()
        holdings_daily = pd.DataFrame()
        trades = pd.DataFrame(columns=["date", "ric"])
        target_weights = pd.DataFrame(columns=["quarter", "ric"])
        quarter_scores = pd.DataFrame(columns=["quarter", "ric", "score"])
        candidate_universe = pd.DataFrame(columns=["quarter", "ric"])

    def fake_run_dynamic_rotation(**kwargs):
        return DummyRunResult()

    monkeypatch.setattr(evaluation_module, "RefinitivClient", DummyClient)
    monkeypatch.setattr(evaluation_module, "RefinitivStrategyDataProvider", DummyProvider)
    monkeypatch.setattr(evaluation_module, "run_dynamic_rotation", fake_run_dynamic_rotation)
    monkeypatch.setattr(
        evaluation_module,
        "_fetch_benchmarks",
        lambda **kwargs: (
            pd.DataFrame(columns=["Date", "RIC", "CLOSE"]),
            pd.DataFrame(columns=["Date", "RIC", "CLOSE"]),
            pd.DataFrame(columns=["Date", "RIC", "CLOSE"]),
        ),
    )
    monkeypatch.setattr(
        evaluation_module,
        "compute_summary_from_data",
        lambda **kwargs: {
            "objective_metrics": {},
            "constraint_metrics": {},
            "diagnostic_metrics": {},
        },
    )

    evaluate_strategy(
        base_config=cfg,
        persist="none",
        benchmark="none",
    )

    assert captured["persistent_cache_enabled"] is True
