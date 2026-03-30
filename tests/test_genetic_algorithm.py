from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

import src.dividend_portfolio.optimization.genetic_algorithm as ga_module
from src.dividend_portfolio.models import (
    AssetConfig,
    PortfolioConfig,
    QuarterlyMetricsConfig,
    RebalanceConfig,
    StrategyConfig,
)
from src.dividend_portfolio.optimization import (
    GeneticAlgorithmConfig,
    GeneticSearchSpace,
    run_genetic_algorithm,
)


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
        out = self.get_close_history(rics, start_date, end_date)[["Date", "RIC", "CLOSE"]].copy()
        out["BID"] = out["CLOSE"] * 0.999
        out["ASK"] = out["CLOSE"] * 1.001
        return out[["Date", "RIC", "BID", "ASK"]].reset_index(drop=True)

    def close(self) -> None:
        return


def _build_fake_provider() -> FakeProvider:
    rics = ["A", "B", "C", "D"]
    market_caps = {"A": 400.0, "B": 300.0, "C": 200.0, "D": 100.0}
    dates = pd.bdate_range("2024-01-02", "2024-09-30")
    prices = pd.DataFrame(
        [{"Date": dt, "RIC": ric, "CLOSE": 100.0} for ric in rics for dt in dates]
    )
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


def test_run_genetic_algorithm_returns_pareto_front(tmp_path: Path) -> None:
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
            allocation_strategy="yield_proportional",
            sqlite_path=str(tmp_path / "dyn.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
        ),
    )

    result = run_genetic_algorithm(
        base_config=cfg,
        search_space=GeneticSearchSpace(
            portfolio_sizes=(1, 2),
            rebalance_interval_quarters=(1, 2),
            allocation_strategies=("equal_weight", "yield_proportional"),
        ),
        config=GeneticAlgorithmConfig(
            population_size=4,
            generations=2,
            elite_count=1,
            random_seed=7,
            persist_trials="none",
            benchmark="none",
            trial_output_dir=tmp_path / "ga_trials",
        ),
        provider=_build_fake_provider(),
    )

    assert result.study_id
    assert result.total_trials_evaluated >= 1
    assert result.pareto_front
    assert result.best_by_return is not None
    assert result.best_by_drawdown is not None
    assert len(result.final_population) == 4
    assert len(result.history) == 3
    assert len(result.trial_results) == result.total_trials_evaluated
    assert Path(result.study_artifacts["study_summary_json"]).exists()
    assert Path(result.study_artifacts["trial_results_csv"]).exists()
    assert Path(result.study_artifacts["population_history_csv"]).exists()
    assert Path(result.study_artifacts["best_by_return_summary_json"]).exists()
    assert Path(result.study_artifacts["best_by_drawdown_summary_json"]).exists()
    for individual in result.final_population:
        assert individual.hyperparameters["portfolio_size"] in {1, 2}
        assert individual.hyperparameters["rebalance_interval_quarters"] in {1, 2}
        assert individual.hyperparameters["allocation_strategy"] in {"equal_weight", "yield_proportional"}


def test_run_genetic_algorithm_skips_invalid_trials_when_some_are_feasible(tmp_path: Path) -> None:
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
            allocation_strategy="yield_proportional",
            sqlite_path=str(tmp_path / "dyn.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
        ),
    )

    provider = _build_fake_provider()
    provider.dividends = provider.dividends.loc[provider.dividends["RIC"].isin(["A"])].reset_index(drop=True)

    result = run_genetic_algorithm(
        base_config=cfg,
        search_space=GeneticSearchSpace(
            portfolio_sizes=(1, 2),
            rebalance_interval_quarters=(1,),
            allocation_strategies=("equal_weight",),
        ),
        config=GeneticAlgorithmConfig(
            population_size=2,
            generations=1,
            elite_count=0,
            random_seed=11,
            persist_trials="none",
            benchmark="none",
            trial_output_dir=tmp_path / "ga_trials",
        ),
        provider=provider,
    )

    assert result.best_by_return is not None
    assert result.best_by_return.error is None
    assert result.best_by_return.hyperparameters["portfolio_size"] == 1
    assert result.total_trials_evaluated == 2
    assert any(individual.error is not None for individual in result.failed_trials)


def test_run_genetic_algorithm_raises_clean_error_when_all_trials_fail(tmp_path: Path) -> None:
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
            allocation_strategy="yield_proportional",
            sqlite_path=str(tmp_path / "dyn.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
        ),
    )

    provider = _build_fake_provider()
    provider.dividends = provider.dividends.iloc[0:0].copy()

    try:
        run_genetic_algorithm(
            base_config=cfg,
            search_space=GeneticSearchSpace(
                portfolio_sizes=(1, 2),
                rebalance_interval_quarters=(1,),
                allocation_strategies=("equal_weight",),
            ),
            config=GeneticAlgorithmConfig(
                population_size=2,
                generations=1,
                elite_count=0,
                random_seed=3,
                persist_trials="none",
                benchmark="none",
                trial_output_dir=tmp_path / "ga_trials",
            ),
            provider=provider,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ValueError when all GA trials are infeasible")

    assert "produced no valid trials" in message
    assert "Initial candidate set too small" in message


def test_run_genetic_algorithm_prefetches_benchmarks_once(monkeypatch, tmp_path: Path) -> None:
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
            allocation_strategy="yield_proportional",
            sqlite_path=str(tmp_path / "dyn.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
        ),
    )

    benchmark_dates = pd.bdate_range("2024-01-02", "2024-09-30")
    benchmark_close = pd.DataFrame(
        {
            "Date": benchmark_dates,
            "RIC": [".SPX"] * len(benchmark_dates),
            "CLOSE": [100.0 + idx for idx in range(len(benchmark_dates))],
        }
    )
    calls = {"count": 0}

    def fake_prepare_benchmark_data(**kwargs):
        calls["count"] += 1
        return benchmark_close.copy(), benchmark_close.copy(), benchmark_close.copy()

    monkeypatch.setattr(ga_module, "prepare_benchmark_data", fake_prepare_benchmark_data)

    result = run_genetic_algorithm(
        base_config=cfg,
        search_space=GeneticSearchSpace(
            portfolio_sizes=(1, 2),
            rebalance_interval_quarters=(1,),
            allocation_strategies=("equal_weight",),
        ),
        config=GeneticAlgorithmConfig(
            population_size=2,
            generations=1,
            elite_count=0,
            random_seed=5,
            persist_trials="none",
            benchmark="sp500",
            trial_output_dir=tmp_path / "ga_trials",
        ),
        provider=_build_fake_provider(),
    )

    assert calls["count"] == 1
    assert result.total_trials_evaluated == 2


def test_process_pool_context_uses_spawn_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(ga_module.sys, "platform", "darwin")
    assert ga_module._process_pool_context_name() == "spawn"
