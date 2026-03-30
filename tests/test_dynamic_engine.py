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
    SelectionPolicyConfig,
    StrategyConfig,
    TransactionCostsConfig,
)
from src.dividend_portfolio.strategy.engine import (
    _build_weighted_selection,
    _select_next_portfolio_by_policy,
    run_dynamic_rotation,
)
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

    def get_bid_ask_history(self, rics: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        out = self.prices.copy()
        out = out.loc[out["RIC"].isin(rics)]
        out = out.loc[(out["Date"] >= pd.Timestamp(start_date)) & (out["Date"] <= pd.Timestamp(end_date))]
        out = out[["Date", "RIC", "CLOSE"]].copy()
        out["BID"] = out["CLOSE"] * 0.999
        out["ASK"] = out["CLOSE"] * 1.001
        return out[["Date", "RIC", "BID", "ASK"]].reset_index(drop=True)


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


def test_dynamic_engine_applies_transaction_costs(tmp_path: Path) -> None:
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
        transaction_costs=TransactionCostsConfig(
            enabled=True,
            commission_bps=0.0,
            commission_min_usd=0.0,
            slippage_bps_per_side=0.0,
            fallback_full_spread_bps=5.0,
            use_bid_ask_when_available=True,
            sizing_rule="cost_aware_scaling",
        ),
        strategy=StrategyConfig(
            mode="dynamic_100_25",
            candidate_count=4,
            portfolio_size=1,
            dividend_payer_lookback_months=12,
            selection_metric="quarter_dividend_yield",
            yield_denominator="quarter_average_close",
            rebalance_timing="first_trading_day_after_quarter_end",
            initial_selection="market_cap",
            initial_weighting="market_cap",
            quarterly_weighting="normalized_yield_score",
            missing_data_policy="backfill_next_ranked",
            sqlite_path=str(tmp_path / "dyn_cost.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
        ),
    )

    provider = _build_fake_provider()
    store = StrategyStore(tmp_path / "dyn_cost.sqlite")
    result = run_dynamic_rotation(
        config=cfg,
        provider=provider,
        store=store,
        start_date="2024-01-02",
        end_date="2024-03-31",
        run_id="run_tx_cost",
    )

    assert len(result.trades) > 0
    assert "total_transaction_cost" in result.trades.columns
    assert "portfolio_transaction_cost_daily" in result.portfolio_daily.columns
    assert float(result.trades["total_transaction_cost"].sum()) > 0.0
    assert float(result.trades["commission_cost"].sum()) == 0.0
    assert float(result.trades["slippage_cost"].sum()) == 0.0
    assert float(result.trades["spread_cost"].sum()) > 0.0
    assert float(result.portfolio_daily["portfolio_transaction_cost_daily"].sum()) > 0.0

    store.close()


def test_dynamic_engine_replace_bottom_n_rotates_at_most_one_incumbent(tmp_path: Path) -> None:
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
            sqlite_path=str(tmp_path / "dyn_replace.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
            selection_policy=SelectionPolicyConfig(
                name="replace_bottom_n",
                max_replacements_per_quarter=1,
                rank_metric="quarter_dividend_yield_score",
            ),
        ),
    )

    provider = _build_fake_provider()
    store = StrategyStore(tmp_path / "dyn_replace.sqlite")
    result = run_dynamic_rotation(
        config=cfg,
        provider=provider,
        store=store,
        start_date="2024-01-02",
        end_date="2024-09-30",
        run_id="run_replace_bottom_n",
    )

    tw = result.target_weights
    q1 = tw.loc[tw["quarter"] == "2024Q1"].sort_values("rank_in_portfolio")
    q2 = tw.loc[tw["quarter"] == "2024Q2"].sort_values("rank_in_portfolio")
    q3 = tw.loc[tw["quarter"] == "2024Q3"].sort_values("rank_in_portfolio")

    assert list(q1["ric"]) == ["A", "B"]
    assert list(q2["ric"]) == ["C", "A"]
    assert list(q3["ric"]) == ["B", "C"]
    assert set(tw["source"].unique()) == {"initial_market_cap", "quarter_dividend_yield_score_replace_bottom_n"}

    store.close()


def test_select_next_portfolio_replace_bottom_n_respects_cap_with_no_outsiders() -> None:
    strategy = StrategyConfig(
        mode="dynamic_100_25",
        candidate_count=2,
        portfolio_size=2,
        selection_policy=SelectionPolicyConfig(
            name="replace_bottom_n",
            max_replacements_per_quarter=10,
            rank_metric="quarter_dividend_yield_score",
        ),
    )
    current_selection = pd.DataFrame(
        {
            "RIC": ["A", "B"],
            "Weight": [0.6, 0.4],
            "RankInPortfolio": [1, 2],
        }
    )
    score_df = pd.DataFrame(
        {
            "RIC": ["A", "B"],
            "Score": [0.01, 0.02],
        }
    )
    candidates = pd.DataFrame(
        {
            "RIC": ["A", "B"],
            "MarketCap": [200.0, 100.0],
            "RankByMarketCap": [1, 2],
        }
    )

    selected, source = _select_next_portfolio_by_policy(
        strategy=strategy,
        score_df=score_df,
        candidates=candidates,
        current_selection=current_selection,
    )

    assert source == "quarter_dividend_yield_score_replace_bottom_n"
    assert set(selected["RIC"]) == {"A", "B"}


def test_select_next_portfolio_replace_bottom_n_tie_breaks_deterministically() -> None:
    strategy = StrategyConfig(
        mode="dynamic_100_25",
        candidate_count=4,
        portfolio_size=2,
        selection_policy=SelectionPolicyConfig(
            name="replace_bottom_n",
            max_replacements_per_quarter=1,
            rank_metric="quarter_dividend_yield_score",
        ),
    )
    current_selection = pd.DataFrame(
        {
            "RIC": ["A", "B"],
            "Weight": [0.5, 0.5],
            "RankInPortfolio": [1, 2],
        }
    )
    score_df = pd.DataFrame(
        {
            "RIC": ["A", "B", "C", "D"],
            "Score": [0.01, 0.01, 0.05, 0.05],
        }
    )
    candidates = pd.DataFrame(
        {
            "RIC": ["A", "B", "C", "D"],
            "MarketCap": [400.0, 300.0, 200.0, 100.0],
            "RankByMarketCap": [1, 2, 3, 4],
        }
    )

    selected, _source = _select_next_portfolio_by_policy(
        strategy=strategy,
        score_df=score_df,
        candidates=candidates,
        current_selection=current_selection,
    )

    assert list(selected["RIC"]) == ["C", "B"]


def test_select_next_portfolio_full_refresh_matches_top_score_selection() -> None:
    strategy = StrategyConfig(
        mode="dynamic_100_25",
        candidate_count=4,
        portfolio_size=2,
        selection_policy=SelectionPolicyConfig(
            name="full_refresh",
            max_replacements_per_quarter=2,
            rank_metric="quarter_dividend_yield_score",
        ),
    )
    current_selection = pd.DataFrame(
        {
            "RIC": ["A", "B"],
            "Weight": [0.5, 0.5],
            "RankInPortfolio": [1, 2],
        }
    )
    score_df = pd.DataFrame(
        {
            "RIC": ["A", "B", "C", "D"],
            "Score": [0.01, 0.06, 0.05, 0.04],
        }
    )
    candidates = pd.DataFrame(
        {
            "RIC": ["A", "B", "C", "D"],
            "MarketCap": [400.0, 300.0, 200.0, 100.0],
            "RankByMarketCap": [1, 2, 3, 4],
        }
    )

    selected, source = _select_next_portfolio_by_policy(
        strategy=strategy,
        score_df=score_df,
        candidates=candidates,
        current_selection=current_selection,
    )

    assert source == "quarter_dividend_yield_score"
    assert list(selected["RIC"]) == ["B", "C"]


def test_dynamic_engine_rebalance_interval_quarters_two(tmp_path: Path) -> None:
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
            rebalance_interval_quarters=2,
            dividend_payer_lookback_months=12,
            selection_metric="quarter_dividend_yield",
            yield_denominator="quarter_average_close",
            rebalance_timing="first_trading_day_after_quarter_end",
            initial_selection="market_cap",
            initial_weighting="market_cap",
            quarterly_weighting="normalized_yield_score",
            missing_data_policy="backfill_next_ranked",
            sqlite_path=str(tmp_path / "dyn_interval.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
        ),
    )

    provider = _build_fake_provider()
    store = StrategyStore(tmp_path / "dyn_interval.sqlite")
    result = run_dynamic_rotation(
        config=cfg,
        provider=provider,
        store=store,
        start_date="2024-01-02",
        end_date="2024-09-30",
        run_id="run_interval_two",
    )

    tw = result.target_weights
    q1 = tw.loc[tw["quarter"] == "2024Q1"].sort_values("rank_in_portfolio")
    q2 = tw.loc[tw["quarter"] == "2024Q2"].sort_values("rank_in_portfolio")
    q3 = tw.loc[tw["quarter"] == "2024Q3"].sort_values("rank_in_portfolio")
    assert list(q1["ric"]) == ["A", "B"]
    assert list(q2["ric"]) == ["A", "B"]
    assert list(q3["ric"]) == ["B", "D"]
    assert math.isclose(float(q1.iloc[0]["weight"]), 4.0 / 7.0, rel_tol=1e-12)
    assert math.isclose(float(q1.iloc[1]["weight"]), 3.0 / 7.0, rel_tol=1e-12)
    assert math.isclose(float(q2.iloc[0]["weight"]), float(q1.iloc[0]["weight"]), rel_tol=1e-12)
    assert math.isclose(float(q2.iloc[1]["weight"]), float(q1.iloc[1]["weight"]), rel_tol=1e-12)

    q2_source = set(q2["source"].astype(str).tolist())
    assert q2_source == {"carry_forward_no_rebalance"}

    rebalance_days = int(result.portfolio_daily["rebalance_flag"].sum())
    assert rebalance_days == 2
    trades_quarters = set(result.trades["quarter"].astype(str).tolist())
    assert trades_quarters == {"2024Q1", "2024Q3"}

    store.close()


def test_build_weighted_selection_supports_allocation_strategies() -> None:
    selection = pd.DataFrame(
        {
            "RIC": ["A", "B", "C"],
            "Score": [0.03, 0.02, 0.01],
            "RankInPortfolio": [1, 2, 3],
        }
    )
    market_caps = pd.DataFrame(
        {
            "RIC": ["A", "B", "C"],
            "MarketCap": [300.0, 200.0, 100.0],
            "MarketCapDate": [pd.Timestamp("2024-01-02")] * 3,
        }
    )

    equal_weight = _build_weighted_selection(selection, weight_strategy="equal_weight", market_caps=market_caps)
    market_cap = _build_weighted_selection(selection, weight_strategy="market_cap", market_caps=market_caps)
    inverse_market_cap = _build_weighted_selection(
        selection,
        weight_strategy="inverse_market_cap",
        market_caps=market_caps,
    )
    yield_prop = _build_weighted_selection(selection, weight_strategy="yield_proportional", market_caps=market_caps)
    yield_rank = _build_weighted_selection(selection, weight_strategy="yield_rank_linear", market_caps=market_caps)

    for weighted in [equal_weight, market_cap, inverse_market_cap, yield_prop, yield_rank]:
        assert math.isclose(float(weighted["Weight"].sum()), 1.0, rel_tol=1e-12)
        assert (weighted["Weight"] > 0).all()

    assert all(math.isclose(float(w), 1.0 / 3.0, rel_tol=1e-12) for w in equal_weight["Weight"])
    assert list(market_cap["Weight"]) == [0.5, 1.0 / 3.0, 1.0 / 6.0]
    assert list(inverse_market_cap["Weight"]) == [1.0 / 6.0, 1.0 / 4.0, 7.0 / 12.0]
    assert list(yield_prop["Weight"]) == [0.5, 1.0 / 3.0, 1.0 / 6.0]
    assert list(yield_rank["Weight"]) == [0.5, 1.0 / 3.0, 1.0 / 6.0]


def test_dynamic_engine_market_cap_allocation_strategy_reweights_selected_names(tmp_path: Path) -> None:
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
            allocation_strategy="market_cap",
            sqlite_path=str(tmp_path / "dyn_mcap.sqlite"),
            parquet_dir=str(tmp_path / "parquet"),
            parquet_enabled=False,
            csv_export_enabled=False,
        ),
    )

    provider = _build_fake_provider()
    store = StrategyStore(tmp_path / "dyn_mcap.sqlite")
    result = run_dynamic_rotation(
        config=cfg,
        provider=provider,
        store=store,
        start_date="2024-01-02",
        end_date="2024-09-30",
        run_id="run_mcap_alloc",
    )

    q2 = result.target_weights.loc[result.target_weights["quarter"] == "2024Q2"].sort_values("rank_in_portfolio")
    assert list(q2["ric"]) == ["C", "A"]
    weights = {str(row["ric"]): float(row["weight"]) for row in q2.to_dict(orient="records")}
    assert math.isclose(weights["A"], 2.0 / 3.0, rel_tol=1e-12)
    assert math.isclose(weights["C"], 1.0 / 3.0, rel_tol=1e-12)

    store.close()
