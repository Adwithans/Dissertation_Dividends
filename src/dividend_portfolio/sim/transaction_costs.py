from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import TransactionCostsConfig

EPS = 1e-12


@dataclass
class RebalanceCostResult:
    shares_after: dict[str, float]
    cash_after: float
    trade_shares_by_ric: dict[str, float]
    trade_value_by_ric: dict[str, float]
    total_value_after: float
    trade_rows: list[dict[str, Any]]
    commission_cost_total: float
    slippage_cost_total: float
    spread_cost_total: float
    transaction_cost_total: float
    scale_used: float
    buy_scale_used: float


def _valid_bid_ask(bid: float | None, ask: float | None) -> bool:
    if bid is None or ask is None:
        return False
    if bid <= 0 or ask <= 0:
        return False
    return ask >= bid


def spread_bps_used(
    *,
    reference_price: float,
    bid: float | None,
    ask: float | None,
    tx: TransactionCostsConfig,
) -> float:
    if tx.use_bid_ask_when_available and _valid_bid_ask(bid, ask):
        mid = ((bid or 0.0) + (ask or 0.0)) / 2.0
        if mid > 0:
            return float(((ask or 0.0) - (bid or 0.0)) / mid * 10000.0)
    return float(tx.fallback_full_spread_bps)


def estimate_trade_costs(
    *,
    trade_shares: float,
    reference_price: float,
    tx: TransactionCostsConfig,
    bid: float | None = None,
    ask: float | None = None,
) -> dict[str, float]:
    q = float(trade_shares)
    ref = float(reference_price)
    if abs(q) <= EPS or ref <= 0:
        return {
            "reference_price": ref,
            "execution_price": ref,
            "gross_notional": 0.0,
            "spread_bps_used": 0.0,
            "slippage_bps_used": 0.0,
            "commission_cost": 0.0,
            "slippage_cost": 0.0,
            "spread_cost": 0.0,
            "total_transaction_cost": 0.0,
            "trade_value": 0.0,
            "net_cash_flow": 0.0,
            "side": 0.0,
            "bid_price": float(bid) if bid is not None else float("nan"),
            "ask_price": float(ask) if ask is not None else float("nan"),
        }

    gross_notional = abs(q) * ref
    spread_bps = spread_bps_used(reference_price=ref, bid=bid, ask=ask, tx=tx)
    half_spread_bps = spread_bps / 2.0
    slippage_bps = float(tx.slippage_bps_per_side)
    commission_cost = max(float(tx.commission_min_usd), gross_notional * float(tx.commission_bps) / 10000.0)
    spread_cost = gross_notional * half_spread_bps / 10000.0
    slippage_cost = gross_notional * slippage_bps / 10000.0
    total_cost = commission_cost + spread_cost + slippage_cost

    side = 1.0 if q > 0 else -1.0
    execution_price = ref * (1.0 + side * (half_spread_bps + slippage_bps) / 10000.0)
    trade_value = q * ref
    net_cash_flow = -trade_value - total_cost

    return {
        "reference_price": ref,
        "execution_price": float(execution_price),
        "gross_notional": float(gross_notional),
        "spread_bps_used": float(spread_bps),
        "slippage_bps_used": float(slippage_bps),
        "commission_cost": float(commission_cost),
        "slippage_cost": float(slippage_cost),
        "spread_cost": float(spread_cost),
        "total_transaction_cost": float(total_cost),
        "trade_value": float(trade_value),
        "net_cash_flow": float(net_cash_flow),
        "side": float(side),
        "bid_price": float(bid) if bid is not None else float("nan"),
        "ask_price": float(ask) if ask is not None else float("nan"),
    }


def rebalance_to_target_with_costs(
    *,
    prices_by_ric: dict[str, float],
    shares_by_ric: dict[str, float],
    portfolio_cash: float,
    target_weights: dict[str, float],
    tx: TransactionCostsConfig,
    bid_by_ric: dict[str, float] | None = None,
    ask_by_ric: dict[str, float] | None = None,
    max_iter: int = 40,
) -> RebalanceCostResult:
    bid_by_ric = bid_by_ric or {}
    ask_by_ric = ask_by_ric or {}
    all_rics = sorted(set(shares_by_ric).union(target_weights))
    tradable_rics = [ric for ric in all_rics if float(prices_by_ric.get(ric, 0.0)) > 0]

    pre_trade_market = sum(float(shares_by_ric.get(ric, 0.0)) * float(prices_by_ric[ric]) for ric in tradable_rics)
    total_before = pre_trade_market + float(portfolio_cash)

    def _evaluate(scale: float, buy_scale: float) -> RebalanceCostResult:
        trade_shares: dict[str, float] = {}
        trade_values: dict[str, float] = {}
        shares_after = dict(shares_by_ric)
        trade_rows: list[dict[str, Any]] = []
        commission_total = 0.0
        slippage_total = 0.0
        spread_total = 0.0
        transaction_total = 0.0
        cash_after = float(portfolio_cash)

        for ric in tradable_rics:
            px = float(prices_by_ric[ric])
            old_shares = float(shares_by_ric.get(ric, 0.0))
            target_weight = float(target_weights.get(ric, 0.0))
            target_shares = 0.0 if target_weight <= 0 else (scale * total_before * target_weight) / px
            q = target_shares - old_shares
            if q > 0:
                q *= buy_scale
            if abs(q) <= EPS:
                q = 0.0

            cost_info = (
                estimate_trade_costs(
                    trade_shares=q,
                    reference_price=px,
                    tx=tx,
                    bid=float(bid_by_ric.get(ric)) if ric in bid_by_ric else None,
                    ask=float(ask_by_ric.get(ric)) if ric in ask_by_ric else None,
                )
                if tx.enabled
                else {
                    "reference_price": px,
                    "execution_price": px,
                    "gross_notional": abs(q) * px,
                    "spread_bps_used": 0.0,
                    "slippage_bps_used": 0.0,
                    "commission_cost": 0.0,
                    "slippage_cost": 0.0,
                    "spread_cost": 0.0,
                    "total_transaction_cost": 0.0,
                    "trade_value": q * px,
                    "net_cash_flow": -(q * px),
                    "side": 1.0 if q > 0 else (-1.0 if q < 0 else 0.0),
                    "bid_price": float(bid_by_ric.get(ric, float("nan"))),
                    "ask_price": float(ask_by_ric.get(ric, float("nan"))),
                }
            )

            shares_after[ric] = old_shares + q
            trade_shares[ric] = q
            trade_values[ric] = float(cost_info["trade_value"])

            commission_total += float(cost_info["commission_cost"])
            slippage_total += float(cost_info["slippage_cost"])
            spread_total += float(cost_info["spread_cost"])
            transaction_total += float(cost_info["total_transaction_cost"])
            cash_after += float(cost_info["net_cash_flow"])

            trade_rows.append(
                {
                    "ric": ric,
                    "trade_shares": float(q),
                    "trade_value": float(cost_info["trade_value"]),
                    "reference_price": float(cost_info["reference_price"]),
                    "execution_price": float(cost_info["execution_price"]),
                    "bid_price": float(cost_info["bid_price"]),
                    "ask_price": float(cost_info["ask_price"]),
                    "gross_notional": float(cost_info["gross_notional"]),
                    "spread_bps_used": float(cost_info["spread_bps_used"]),
                    "slippage_bps_used": float(cost_info["slippage_bps_used"]),
                    "commission_cost": float(cost_info["commission_cost"]),
                    "slippage_cost": float(cost_info["slippage_cost"]),
                    "spread_cost": float(cost_info["spread_cost"]),
                    "total_transaction_cost": float(cost_info["total_transaction_cost"]),
                    "net_cash_flow": float(cost_info["net_cash_flow"]),
                }
            )

        total_after = cash_after + sum(float(shares_after.get(ric, 0.0)) * float(prices_by_ric[ric]) for ric in tradable_rics)
        return RebalanceCostResult(
            shares_after=shares_after,
            cash_after=float(cash_after),
            trade_shares_by_ric=trade_shares,
            trade_value_by_ric=trade_values,
            total_value_after=float(total_after),
            trade_rows=trade_rows,
            commission_cost_total=float(commission_total),
            slippage_cost_total=float(slippage_total),
            spread_cost_total=float(spread_total),
            transaction_cost_total=float(transaction_total),
            scale_used=float(scale),
            buy_scale_used=float(buy_scale),
        )

    if not tx.enabled or not tradable_rics:
        return _evaluate(1.0, 1.0)

    result = _evaluate(1.0, 1.0)
    if tx.sizing_rule != "cost_aware_scaling" or result.cash_after >= -1e-8:
        return result

    lo, hi = 0.0, 1.0
    best = _evaluate(0.0, 1.0)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        candidate = _evaluate(mid, 1.0)
        if candidate.cash_after >= 0:
            lo = mid
            best = candidate
        else:
            hi = mid

    if best.cash_after >= -1e-8:
        return best

    lo_b, hi_b = 0.0, 1.0
    best_b = _evaluate(best.scale_used, 0.0)
    for _ in range(max_iter):
        mid_b = (lo_b + hi_b) / 2.0
        candidate = _evaluate(best.scale_used, mid_b)
        if candidate.cash_after >= 0:
            lo_b = mid_b
            best_b = candidate
        else:
            hi_b = mid_b

    if -1e-8 < best_b.cash_after < 0:
        best_b.cash_after = 0.0
    return best_b

