from __future__ import annotations

import pandas as pd

from src.dividend_portfolio.strategy.rules import (
    compute_quarter_dividend_yield_scores,
    select_initial_portfolio_by_market_cap,
    select_top_candidates_by_market_cap,
    select_top_portfolio_by_score,
)


def test_dividend_yield_score_and_tie_break_are_deterministic() -> None:
    prices = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
            "RIC": ["A", "A", "B", "B"],
            "CLOSE": [100.0, 100.0, 100.0, 100.0],
        }
    )
    dividends = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "RIC": ["A", "B"],
            "Dividend": [1.0, 1.0],
        }
    )

    out = compute_quarter_dividend_yield_scores(prices, dividends, denominator="quarter_average_close")
    assert list(out["RIC"]) == ["A", "B"]
    assert out.iloc[0]["Score"] == out.iloc[1]["Score"]
    assert list(out["RankByScore"]) == [1, 2]


def test_initial_selection_top100_to_top25_by_market_cap_weights() -> None:
    mcap = pd.DataFrame(
        {
            "RIC": ["A", "B", "C"],
            "MarketCap": [300.0, 200.0, 100.0],
            "MarketCapDate": pd.to_datetime(["2024-01-02"] * 3),
        }
    )
    candidates = select_top_candidates_by_market_cap(mcap, {"A", "B", "C"}, candidate_count=3)
    portfolio = select_initial_portfolio_by_market_cap(candidates, portfolio_size=2)

    assert list(portfolio["RIC"]) == ["A", "B"]
    assert abs(float(portfolio["Weight"].sum()) - 1.0) < 1e-12
    assert abs(float(portfolio.loc[portfolio["RIC"] == "A", "Weight"].iloc[0]) - 0.6) < 1e-12
    assert abs(float(portfolio.loc[portfolio["RIC"] == "B", "Weight"].iloc[0]) - 0.4) < 1e-12


def test_select_top_portfolio_by_score_replaces_constituents() -> None:
    scores = pd.DataFrame(
        {
            "RIC": ["A", "B", "C", "D"],
            "Score": [0.01, 0.06, 0.05, 0.04],
        }
    )
    selected = select_top_portfolio_by_score(scores, portfolio_size=2)
    assert list(selected["RIC"]) == ["B", "C"]
    assert abs(float(selected["Weight"].sum()) - 1.0) < 1e-12


def test_fallback_promotion_when_scores_zero_assigns_equal_weights() -> None:
    scores = pd.DataFrame(
        {
            "RIC": ["A", "B", "C"],
            "Score": [0.0, 0.0, 0.0],
        }
    )
    selected = select_top_portfolio_by_score(scores, portfolio_size=2)
    assert len(selected) == 2
    assert abs(float(selected["Weight"].sum()) - 1.0) < 1e-12
    assert abs(float(selected.iloc[0]["Weight"]) - 0.5) < 1e-12
    assert abs(float(selected.iloc[1]["Weight"]) - 0.5) < 1e-12
