from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.dividend_portfolio.models import (
    AssetConfig,
    PortfolioConfig,
    QuarterlyMetricsConfig,
    RebalanceConfig,
    StrategyConfig,
)
from src.dividend_portfolio.strategy.engine import run_dynamic_rotation
from src.dividend_portfolio.strategy.storage import StrategyStore


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


def _build_fake_provider() -> FakeProvider:
    rics = ["A", "B", "C", "D"]
    market_caps = {"A": 400.0, "B": 300.0, "C": 200.0, "D": 100.0}

    dates = pd.bdate_range("2024-01-02", "2024-09-30")
    price_rows = []
    for ric in rics:
        for dt in dates:
            price_rows.append({"Date": dt, "RIC": ric, "CLOSE": 100.0})
    prices = pd.DataFrame(price_rows)

    div_rows = [
        # lookback events so all names qualify as dividend-payers at t0
        {"Date": pd.Timestamp("2023-12-15"), "RIC": "A", "Dividend": 0.5},
        {"Date": pd.Timestamp("2023-12-15"), "RIC": "B", "Dividend": 0.5},
        {"Date": pd.Timestamp("2023-12-15"), "RIC": "C", "Dividend": 0.5},
        {"Date": pd.Timestamp("2023-12-15"), "RIC": "D", "Dividend": 0.5},
        # Q1 scores: C > A > B > D
        {"Date": pd.Timestamp("2024-02-15"), "RIC": "A", "Dividend": 2.0},
        {"Date": pd.Timestamp("2024-02-15"), "RIC": "B", "Dividend": 1.0},
        {"Date": pd.Timestamp("2024-02-15"), "RIC": "C", "Dividend": 3.0},
        {"Date": pd.Timestamp("2024-02-15"), "RIC": "D", "Dividend": 0.0},
        # Q2 scores: B > D > C > A
        {"Date": pd.Timestamp("2024-05-15"), "RIC": "A", "Dividend": 1.0},
        {"Date": pd.Timestamp("2024-05-15"), "RIC": "B", "Dividend": 4.0},
        {"Date": pd.Timestamp("2024-05-15"), "RIC": "C", "Dividend": 2.0},
        {"Date": pd.Timestamp("2024-05-15"), "RIC": "D", "Dividend": 3.0},
        # Q3 scores: D > A > C > B
        {"Date": pd.Timestamp("2024-08-15"), "RIC": "A", "Dividend": 3.0},
        {"Date": pd.Timestamp("2024-08-15"), "RIC": "B", "Dividend": 1.0},
        {"Date": pd.Timestamp("2024-08-15"), "RIC": "C", "Dividend": 2.0},
        {"Date": pd.Timestamp("2024-08-15"), "RIC": "D", "Dividend": 4.0},
    ]
    dividends = pd.DataFrame(div_rows)

    return FakeProvider(rics=rics, market_caps=market_caps, prices=prices, dividends=dividends)


def test_dynamic_engine_three_quarter_rotation(tmp_path: Path) -> None:
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
            dividend_payer_lookback_months=12,
            selection_metric="quarter_dividend_yield",
            yield_denominator="quarter_average_close",
            rebalance_timing="first_trading_day_after_quarter_end",
            initial_selection="market_cap",
            initial_weighting="market_cap",
            quarterly_weighting="normalized_yield_score",
            missing_data_policy="backfill_next_ranked",
            sqlite_path=str(tmp_path / "dyn.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
        ),
    )

    provider = _build_fake_provider()
    store = StrategyStore(tmp_path / "dyn.sqlite")
    result = run_dynamic_rotation(
        config=cfg,
        provider=provider,
        store=store,
        start_date="2024-01-02",
        end_date="2024-09-30",
        run_id="run_three_q",
    )

    assert result.candidate_universe["quarter"].nunique() == 3
    assert len(result.candidate_universe) == 12
    assert len(result.portfolio_daily) > 0
    assert len(result.holdings_daily) > 0

    tw = result.target_weights
    q1 = tw.loc[tw["quarter"] == "2024Q1"].sort_values("rank_in_portfolio")
    q2 = tw.loc[tw["quarter"] == "2024Q2"].sort_values("rank_in_portfolio")
    q3 = tw.loc[tw["quarter"] == "2024Q3"].sort_values("rank_in_portfolio")

    assert list(q1["ric"]) == ["A", "B"]
    assert list(q2["ric"]) == ["C", "A"]
    assert list(q3["ric"]) == ["B", "D"]

    store.close()
