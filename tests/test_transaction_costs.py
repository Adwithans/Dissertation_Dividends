from __future__ import annotations

import math

from src.dividend_portfolio.models import TransactionCostsConfig
from src.dividend_portfolio.sim.transaction_costs import (
    estimate_trade_costs,
    rebalance_to_target_with_costs,
)


def test_trade_cost_buy_sell_symmetry() -> None:
    tx = TransactionCostsConfig(
        enabled=True,
        commission_bps=1.0,
        commission_min_usd=0.0,
        slippage_bps_per_side=2.0,
        fallback_full_spread_bps=5.0,
        use_bid_ask_when_available=True,
        sizing_rule="cost_aware_scaling",
    )
    buy = estimate_trade_costs(
        trade_shares=10.0,
        reference_price=100.0,
        tx=tx,
        bid=99.95,
        ask=100.05,
    )
    sell = estimate_trade_costs(
        trade_shares=-10.0,
        reference_price=100.0,
        tx=tx,
        bid=99.95,
        ask=100.05,
    )

    assert math.isclose(buy["total_transaction_cost"], sell["total_transaction_cost"], rel_tol=1e-12)
    assert buy["execution_price"] > 100.0
    assert sell["execution_price"] < 100.0
    assert buy["net_cash_flow"] < 0
    assert sell["net_cash_flow"] > 0


def test_trade_cost_enforces_min_ticket_fee() -> None:
    tx = TransactionCostsConfig(
        enabled=True,
        commission_bps=0.1,
        commission_min_usd=1.0,
        slippage_bps_per_side=0.0,
        fallback_full_spread_bps=0.0,
        use_bid_ask_when_available=False,
        sizing_rule="cost_aware_scaling",
    )
    out = estimate_trade_costs(
        trade_shares=1.0,
        reference_price=10.0,
        tx=tx,
        bid=None,
        ask=None,
    )
    assert math.isclose(out["commission_cost"], 1.0, rel_tol=1e-12)


def test_trade_cost_uses_fallback_spread_when_bid_ask_invalid() -> None:
    tx = TransactionCostsConfig(
        enabled=True,
        commission_bps=0.0,
        commission_min_usd=0.0,
        slippage_bps_per_side=0.0,
        fallback_full_spread_bps=7.5,
        use_bid_ask_when_available=True,
        sizing_rule="cost_aware_scaling",
    )
    out = estimate_trade_costs(
        trade_shares=10.0,
        reference_price=100.0,
        tx=tx,
        bid=0.0,
        ask=0.0,
    )
    assert math.isclose(out["spread_bps_used"], 7.5, rel_tol=1e-12)


def test_cost_aware_scaling_avoids_negative_cash() -> None:
    tx = TransactionCostsConfig(
        enabled=True,
        commission_bps=20.0,
        commission_min_usd=5.0,
        slippage_bps_per_side=50.0,
        fallback_full_spread_bps=100.0,
        use_bid_ask_when_available=False,
        sizing_rule="cost_aware_scaling",
    )
    result = rebalance_to_target_with_costs(
        prices_by_ric={"A": 100.0, "B": 100.0},
        shares_by_ric={"A": 0.0, "B": 0.0},
        portfolio_cash=100.0,
        target_weights={"A": 0.5, "B": 0.5},
        tx=tx,
        bid_by_ric={},
        ask_by_ric={},
    )

    assert result.cash_after >= -1e-8
    assert 0.0 <= result.scale_used <= 1.0
    assert result.transaction_cost_total > 0.0
