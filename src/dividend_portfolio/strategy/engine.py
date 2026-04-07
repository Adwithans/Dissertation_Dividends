from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from typing import Any

import pandas as pd

from ..logging_utils import get_logger
from ..models import PortfolioConfig, StrategyConfig, TransactionCostsConfig
from ..sim.transaction_costs import rebalance_to_target_with_costs
from .parquet_sidecar import ParquetSidecarWriter
from .provider import StrategyDataProvider, trailing_lookback_start, utc_now_id
from .rules import (
    compute_quarter_dividend_yield_scores,
    select_initial_portfolio_by_market_cap,
    select_top_candidates_by_market_cap,
    select_top_portfolio_by_score,
)
from .storage import StrategyStore


@dataclass
class DynamicRunResult:
    run_id: str
    candidate_universe: pd.DataFrame
    quarter_scores: pd.DataFrame
    target_weights: pd.DataFrame
    trades: pd.DataFrame
    holdings_daily: pd.DataFrame
    portfolio_daily: pd.DataFrame


@dataclass
class PendingSelection:
    selection: pd.DataFrame
    source: str
    weight_strategy: str


def _quarter_windows(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    periods = pd.period_range(start=start.to_period("Q"), end=end.to_period("Q"), freq="Q")

    out: list[tuple[str, str, str]] = []
    for p in periods:
        q_start = max(start, p.start_time.normalize()).date().isoformat()
        q_end = min(end, p.end_time.normalize()).date().isoformat()
        out.append((str(p), q_start, q_end))
    return out


def _build_top100_candidates(
    provider: StrategyDataProvider,
    *,
    as_of_date: str,
    lookback_months: int,
    candidate_count: int,
    supplemental_rics: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    supplemental = [
        str(ric).strip().upper()
        for ric in (supplemental_rics or [])
        if str(ric).strip()
    ]
    constituents = [
        str(ric).strip().upper()
        for ric in provider.get_sp500_constituents_as_of(as_of_date)
        if str(ric).strip()
    ]
    universe_rics = list(dict.fromkeys(constituents + supplemental))
    payer_start = trailing_lookback_start(as_of_date, lookback_months)
    scan_chunk = int(os.getenv("DIVIDEND_PAYER_SCAN_CHUNK", "75"))
    scan_chunk = max(scan_chunk, 1)
    stats: dict[str, Any] = {
        "as_of_date": as_of_date,
        "payer_lookback_start": payer_start,
        "lookback_months": int(lookback_months),
        "candidate_count_target": int(candidate_count),
        "scan_chunk": int(scan_chunk),
        "constituent_count": int(len(constituents)),
        "supplemental_universe_count": int(len(supplemental)),
        "market_cap_count": 0,
        "ranked_market_cap_count": 0,
        "dividend_event_count": 0,
        "dividend_payer_count": 0,
        "supplemental_candidate_count": 0,
        "candidate_count_found": 0,
    }
    if not universe_rics:
        return (
            pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate", "RankByMarketCap"]),
            pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"]),
            pd.DataFrame(columns=["RIC", "Date", "Dividend"]),
            stats,
        )

    market_caps = provider.get_market_cap_snapshot(universe_rics, as_of_date)
    stats["market_cap_count"] = int(len(market_caps))
    if market_caps.empty:
        return (
            pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate", "RankByMarketCap"]),
            market_caps,
            pd.DataFrame(columns=["RIC", "Date", "Dividend"]),
            stats,
        )

    ranked_caps = market_caps.copy()
    ranked_caps["RIC"] = ranked_caps["RIC"].astype(str).str.strip()
    ranked_caps["MarketCap"] = pd.to_numeric(ranked_caps["MarketCap"], errors="coerce")
    ranked_caps = ranked_caps.dropna(subset=["RIC", "MarketCap"])
    ranked_caps = ranked_caps.loc[ranked_caps["MarketCap"] > 0]
    ranked_caps = ranked_caps.sort_values(["MarketCap", "RIC"], ascending=[False, True]).reset_index(drop=True)
    stats["ranked_market_cap_count"] = int(len(ranked_caps))
    ranked_rics = ranked_caps["RIC"].tolist()
    if not ranked_rics:
        return (
            pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate", "RankByMarketCap"]),
            market_caps,
            pd.DataFrame(columns=["RIC", "Date", "Dividend"]),
            stats,
        )

    dividend_payers: set[str] = set()
    payer_parts: list[pd.DataFrame] = []
    candidates = pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate", "RankByMarketCap"])
    for offset in range(0, len(ranked_rics), scan_chunk):
        probe = ranked_rics[offset : offset + scan_chunk]
        payer_slice = provider.get_dividend_events(probe, payer_start, as_of_date)
        if not payer_slice.empty:
            payer_parts.append(payer_slice)
            dividend_payers.update(payer_slice["RIC"].dropna().astype(str))

        candidates = select_top_candidates_by_market_cap(
            market_caps=market_caps,
            dividend_payers=dividend_payers,
            candidate_count=candidate_count,
        )
        if not candidates.empty:
            stats["supplemental_candidate_count"] = int(
                candidates["RIC"].astype(str).isin(set(supplemental)).sum()
            )
        if len(candidates) >= candidate_count:
            break

    payer_events = (
        pd.concat(payer_parts, ignore_index=True)
        if payer_parts
        else pd.DataFrame(columns=["RIC", "Date", "Dividend"])
    )
    if not payer_events.empty:
        payer_events = payer_events.drop_duplicates(subset=["RIC", "Date", "Dividend"]).reset_index(drop=True)
    stats["dividend_event_count"] = int(len(payer_events))
    stats["dividend_payer_count"] = int(len(dividend_payers))
    stats["candidate_count_found"] = int(len(candidates))
    return candidates, market_caps, payer_events, stats


def _normalize_weights(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    denom = float(pd.to_numeric(out[score_col], errors="coerce").fillna(0.0).sum())
    if denom > 0:
        out["Weight"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0) / denom
    else:
        out["Weight"] = 1.0 / len(out)
    return out


def _strategy_allocation_strategy(strategy: StrategyConfig) -> str:
    raw = strategy.allocation_strategy or strategy.quarterly_weighting or "yield_proportional"
    normalized = str(raw).strip().lower()
    if normalized == "normalized_yield_score":
        return "yield_proportional"
    return normalized


def _strategy_bond_rics(strategy: StrategyConfig) -> list[str]:
    if not strategy.bond_universe.enabled:
        return []
    return [
        str(ric).strip().upper()
        for ric in strategy.bond_universe.rics
        if str(ric).strip()
    ]


def _selection_from_holdings(holdings_df: pd.DataFrame) -> pd.DataFrame:
    if holdings_df.empty:
        return pd.DataFrame(columns=["RIC", "Weight", "RankInPortfolio"])
    out = holdings_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["market_value"] = pd.to_numeric(out["market_value"], errors="coerce")
    out = out.dropna(subset=["date", "ric", "market_value"])
    if out.empty:
        return pd.DataFrame(columns=["RIC", "Weight", "RankInPortfolio"])
    last_date = out["date"].max()
    last_holdings = out.loc[out["date"] == last_date].copy()
    last_holdings = last_holdings.loc[last_holdings["market_value"] > 0].copy()
    if last_holdings.empty:
        return pd.DataFrame(columns=["RIC", "Weight", "RankInPortfolio"])
    total_market = float(last_holdings["market_value"].sum())
    last_holdings["RIC"] = last_holdings["ric"].astype(str)
    last_holdings = last_holdings.sort_values(["market_value", "RIC"], ascending=[False, True]).reset_index(drop=True)
    last_holdings["Weight"] = (
        last_holdings["market_value"] / total_market if total_market > 0 else 1.0 / len(last_holdings)
    )
    last_holdings["RankInPortfolio"] = last_holdings.index + 1
    return last_holdings[["RIC", "Weight", "RankInPortfolio"]]


def _pick_refuge_bond(
    *,
    bond_rics: list[str],
    refuge_score_df: pd.DataFrame | None,
    prices_at_date: pd.Series,
) -> str | None:
    available = [
        ric for ric in bond_rics if ric in prices_at_date.index and pd.notna(prices_at_date.get(ric)) and float(prices_at_date.get(ric)) > 0
    ]
    if not available:
        return None
    if refuge_score_df is not None and not refuge_score_df.empty:
        ranked = refuge_score_df.copy()
        ranked["RIC"] = ranked["RIC"].astype(str)
        ranked["Score"] = pd.to_numeric(ranked["Score"], errors="coerce")
        ranked = ranked.dropna(subset=["RIC", "Score"])
        ranked = ranked.loc[ranked["RIC"].isin(available)]
        ranked = ranked.sort_values(["Score", "RIC"], ascending=[False, True]).reset_index(drop=True)
        if not ranked.empty:
            return str(ranked.iloc[0]["RIC"])
    return sorted(available)[0]


def _selection_template(selection: pd.DataFrame) -> pd.DataFrame:
    out = selection.copy()
    keep = [col for col in ["RIC", "Score", "RankInPortfolio", "Weight"] if col in out.columns]
    out = out[keep].copy()
    out["RIC"] = out["RIC"].astype(str)
    if "RankInPortfolio" not in out.columns:
        out["RankInPortfolio"] = range(1, len(out) + 1)
    out["RankInPortfolio"] = pd.to_numeric(out["RankInPortfolio"], errors="coerce").fillna(0).astype(int)
    return out.reset_index(drop=True)


def _equal_weight_selection(selection: pd.DataFrame) -> pd.DataFrame:
    out = _selection_template(selection)
    if out.empty:
        out["Weight"] = pd.Series(dtype=float)
        return out
    out["Weight"] = 1.0 / len(out)
    return out


def _market_cap_lookup(market_caps: pd.DataFrame) -> dict[str, float]:
    if market_caps.empty:
        return {}
    out = market_caps.copy()
    out["RIC"] = out["RIC"].astype(str).str.strip()
    out["MarketCap"] = pd.to_numeric(out["MarketCap"], errors="coerce")
    out = out.dropna(subset=["RIC", "MarketCap"])
    out = out.loc[out["MarketCap"] > 0]
    return {str(r.RIC): float(r.MarketCap) for r in out.itertuples(index=False)}


def _build_weighted_selection(
    selection: pd.DataFrame,
    *,
    weight_strategy: str,
    market_caps: pd.DataFrame,
) -> pd.DataFrame:
    template = _selection_template(selection)
    if template.empty:
        template["Weight"] = pd.Series(dtype=float)
        return template

    if weight_strategy == "fixed":
        if "Weight" not in template.columns:
            return _equal_weight_selection(template)
        fixed = pd.to_numeric(template["Weight"], errors="coerce")
        if fixed.isna().any() or (fixed <= 0).any():
            return _equal_weight_selection(template)
        denom = float(fixed.sum())
        if denom <= 0:
            return _equal_weight_selection(template)
        template["Weight"] = fixed / denom
        return template

    if weight_strategy == "equal_weight":
        return _equal_weight_selection(template)

    if weight_strategy in {"market_cap", "inverse_market_cap"}:
        lookup = _market_cap_lookup(market_caps)
        raw = template["RIC"].map(lookup)
        if raw.isna().any() or (raw <= 0).any():
            return _equal_weight_selection(template)
        if weight_strategy == "inverse_market_cap":
            raw = 1.0 / raw
    elif weight_strategy == "yield_proportional":
        raw = pd.to_numeric(template.get("Score"), errors="coerce")
        if raw.isna().any() or (raw <= 0).any():
            return _equal_weight_selection(template)
    elif weight_strategy == "yield_rank_linear":
        ranked = template.copy()
        ranked["Score"] = pd.to_numeric(ranked.get("Score"), errors="coerce")
        if ranked["Score"].isna().any():
            return _equal_weight_selection(template)
        ranked = ranked.sort_values(["Score", "RIC"], ascending=[False, True]).reset_index(drop=True)
        ranked["RawWeight"] = list(range(len(ranked), 0, -1))
        raw_lookup = {
            str(r.RIC): float(r.RawWeight)
            for r in ranked[["RIC", "RawWeight"]].itertuples(index=False)
        }
        raw = template["RIC"].map(raw_lookup)
    else:
        raise ValueError(f"Unsupported weight strategy: {weight_strategy!r}")

    denom = float(pd.to_numeric(raw, errors="coerce").sum())
    if pd.isna(denom) or denom <= 0:
        return _equal_weight_selection(template)
    template["Weight"] = pd.to_numeric(raw, errors="coerce") / denom
    return template


def _materialize_pending_selection(
    pending: PendingSelection,
    *,
    market_caps: pd.DataFrame,
) -> pd.DataFrame:
    return _build_weighted_selection(
        pending.selection,
        weight_strategy=pending.weight_strategy,
        market_caps=market_caps,
    )


def _select_next_portfolio_with_backfill(
    score_df: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    portfolio_size: int,
) -> pd.DataFrame:
    selected = select_top_portfolio_by_score(score_df, portfolio_size=portfolio_size)
    if len(selected) >= portfolio_size:
        return selected

    selected_rics = set(selected["RIC"].astype(str))
    needed = portfolio_size - len(selected)
    if needed <= 0:
        return selected

    fallback = candidates.sort_values(["RankByMarketCap", "RIC"], ascending=[True, True]).copy()
    fallback = fallback.loc[~fallback["RIC"].astype(str).isin(selected_rics)].head(needed)
    if fallback.empty:
        return selected

    fill = pd.DataFrame(
        {
            "RIC": fallback["RIC"].astype(str).tolist(),
            "Score": [0.0] * len(fallback),
            "RankInPortfolio": [0] * len(fallback),
            "Weight": [0.0] * len(fallback),
        }
    )
    out = pd.concat([selected, fill], ignore_index=True)
    out = _normalize_weights(out, "Score").reset_index(drop=True)
    out["RankInPortfolio"] = out.index + 1
    return out


def _to_score_lookup(score_df: pd.DataFrame) -> dict[str, float]:
    if score_df.empty:
        return {}
    out = score_df.copy()
    out["RIC"] = out["RIC"].astype(str)
    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    out = out.dropna(subset=["RIC", "Score"])
    return {str(r.RIC): float(r.Score) for r in out.itertuples(index=False)}


def _select_next_portfolio_replace_bottom_n(
    *,
    score_df: pd.DataFrame,
    candidates: pd.DataFrame,
    current_selection: pd.DataFrame,
    portfolio_size: int,
    max_replacements_per_quarter: int,
) -> pd.DataFrame:
    current_rics = (
        current_selection.get("RIC", pd.Series(dtype=object)).astype(str).dropna().drop_duplicates().tolist()
    )
    if not current_rics:
        return _select_next_portfolio_with_backfill(score_df, candidates, portfolio_size=portfolio_size)

    score_lookup = _to_score_lookup(score_df)
    current_set = set(current_rics)

    incumbent_rank = pd.DataFrame({"RIC": current_rics})
    incumbent_rank["Score"] = incumbent_rank["RIC"].map(score_lookup).fillna(0.0)
    incumbent_rank = incumbent_rank.sort_values(["Score", "RIC"], ascending=[True, True]).reset_index(drop=True)

    outsiders = candidates.get("RIC", pd.Series(dtype=object)).astype(str).dropna()
    outsider_rank = pd.DataFrame({"RIC": outsiders.loc[~outsiders.isin(current_set)].drop_duplicates().tolist()})
    outsider_rank["Score"] = outsider_rank["RIC"].map(score_lookup).fillna(0.0)
    outsider_rank = outsider_rank.sort_values(["Score", "RIC"], ascending=[False, True]).reset_index(drop=True)

    replacement_cap = max(int(max_replacements_per_quarter), 0)
    k = min(replacement_cap, len(incumbent_rank), len(outsider_rank))

    if k <= 0:
        final_rics = list(current_rics)
    else:
        to_replace = set(incumbent_rank.head(k)["RIC"].astype(str))
        entrants = outsider_rank.head(k)["RIC"].astype(str).tolist()
        final_rics = [ric for ric in current_rics if ric not in to_replace] + entrants

    seen: set[str] = set()
    unique_final: list[str] = []
    for ric in final_rics:
        if ric not in seen:
            unique_final.append(ric)
            seen.add(ric)
    final_rics = unique_final[:portfolio_size]

    if len(final_rics) < portfolio_size:
        fallback = candidates.copy()
        if "RankByMarketCap" not in fallback.columns:
            fallback["RankByMarketCap"] = range(1, len(fallback) + 1)
        fallback = fallback.sort_values(["RankByMarketCap", "RIC"], ascending=[True, True]).copy()
        for ric in fallback["RIC"].astype(str).tolist():
            if ric in seen:
                continue
            final_rics.append(ric)
            seen.add(ric)
            if len(final_rics) >= portfolio_size:
                break

    final_score_df = pd.DataFrame(
        {
            "RIC": final_rics,
            "Score": [score_lookup.get(ric, float("nan")) for ric in final_rics],
        }
    )
    return _select_next_portfolio_with_backfill(final_score_df, candidates, portfolio_size=portfolio_size)


def _select_next_portfolio_by_policy(
    *,
    strategy: StrategyConfig,
    score_df: pd.DataFrame,
    candidates: pd.DataFrame,
    current_selection: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    policy = strategy.selection_policy
    if policy.rank_metric != "quarter_dividend_yield_score":
        raise ValueError(
            "Unsupported strategy.selection_policy.rank_metric: "
            f"{policy.rank_metric!r}"
        )
    if policy.name == "replace_bottom_n":
        selected = _select_next_portfolio_replace_bottom_n(
            score_df=score_df,
            candidates=candidates,
            current_selection=current_selection,
            portfolio_size=strategy.portfolio_size,
            max_replacements_per_quarter=policy.max_replacements_per_quarter,
        )
        return selected, "quarter_dividend_yield_score_replace_bottom_n"

    selected = _select_next_portfolio_with_backfill(
        score_df,
        candidates,
        portfolio_size=strategy.portfolio_size,
    )
    return selected, "quarter_dividend_yield_score"


def _pivot_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    out = prices.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["CLOSE"] = pd.to_numeric(out["CLOSE"], errors="coerce")
    out = out.dropna(subset=["Date", "RIC", "CLOSE"])
    pivot = out.pivot_table(index="Date", columns="RIC", values="CLOSE", aggfunc="last")
    return pivot.sort_index()


def _pivot_bid_ask(bid_ask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bid_ask.empty:
        return pd.DataFrame(), pd.DataFrame()
    out = bid_ask.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["BID"] = pd.to_numeric(out.get("BID"), errors="coerce")
    out["ASK"] = pd.to_numeric(out.get("ASK"), errors="coerce")
    out = out.dropna(subset=["Date", "RIC"], how="any")
    if out.empty:
        return pd.DataFrame(), pd.DataFrame()
    bid = out.pivot_table(index="Date", columns="RIC", values="BID", aggfunc="last").sort_index()
    ask = out.pivot_table(index="Date", columns="RIC", values="ASK", aggfunc="last").sort_index()
    return bid, ask


def _find_rebalance_date(price_pivot: pd.DataFrame, selection_rics: list[str], floor_date: str) -> pd.Timestamp:
    floor = pd.Timestamp(floor_date)
    eligible = price_pivot.loc[price_pivot.index >= floor]
    if eligible.empty:
        raise ValueError(f"No price rows on/after rebalance floor date {floor_date}")
    mask = eligible[selection_rics].notna().all(axis=1)
    dates = eligible.index[mask]
    if len(dates) == 0:
        raise ValueError(f"No common rebalance date with complete prices for selected assets at {floor_date}")
    return pd.Timestamp(dates[0])


def _simulate_quarter(
    *,
    run_id: str,
    quarter_label: str,
    quarter_start: str,
    quarter_end: str,
    selection: pd.DataFrame,
    prev_shares: dict[str, float],
    prev_cash: float,
    transaction_costs: TransactionCostsConfig,
    prev_transaction_cost_cumulative: float,
    do_rebalance: bool = True,
    provider: StrategyDataProvider | None = None,
    price_long: pd.DataFrame | None = None,
    div_long: pd.DataFrame | None = None,
    bid_ask_long: pd.DataFrame | None = None,
    prev_entry_price_reference: dict[str, float] | None = None,
    baseline_sell_enabled: bool = False,
    baseline_sell_threshold: float = 0.10,
    bond_rics: list[str] | None = None,
    refuge_score_df: pd.DataFrame | None = None,
) -> tuple[
    dict[str, float],
    float,
    float,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
]:
    eps = 1e-12
    selection_rics = selection["RIC"].astype(str).tolist()
    target_weights = dict(zip(selection_rics, selection["Weight"].astype(float), strict=False))
    bond_rics = [str(ric).strip().upper() for ric in (bond_rics or []) if str(ric).strip()]
    bond_set = set(bond_rics)
    prior_rics = [str(ric) for ric, shares in prev_shares.items() if abs(float(shares)) > eps]
    universe_for_prices = sorted(set(selection_rics).union(prior_rics).union(bond_set))

    if price_long is None:
        if provider is None:
            raise ValueError("Either provider or price_long must be supplied to _simulate_quarter")
        price_long = provider.get_close_history(universe_for_prices, quarter_start, quarter_end)
    else:
        price_long = price_long.loc[price_long["RIC"].astype(str).isin(universe_for_prices)].copy()

    price_pivot = _pivot_prices(price_long).ffill()
    if price_pivot.empty:
        raise ValueError(f"No price data returned for quarter {quarter_label}")

    first_valuation_date = _find_rebalance_date(price_pivot, selection_rics, quarter_start)
    quarter_dates = price_pivot.index[price_pivot.index >= first_valuation_date]
    quarter_dates = quarter_dates[quarter_dates <= pd.Timestamp(quarter_end)]
    if len(quarter_dates) == 0:
        raise ValueError(f"No tradable dates in quarter window for {quarter_label}")

    if bid_ask_long is None:
        if provider is None:
            bid_ask_long = pd.DataFrame(columns=["Date", "RIC", "BID", "ASK"])
        else:
            bid_ask_long = provider.get_bid_ask_history(universe_for_prices, quarter_start, quarter_end)
    else:
        bid_ask_long = bid_ask_long.loc[bid_ask_long["RIC"].astype(str).isin(universe_for_prices)].copy()
    bid_pivot, ask_pivot = _pivot_bid_ask(bid_ask_long)
    if not bid_pivot.empty:
        bid_pivot = bid_pivot.ffill()
    if not ask_pivot.empty:
        ask_pivot = ask_pivot.ffill()

    if div_long is None:
        if provider is None:
            raise ValueError("Either provider or div_long must be supplied to _simulate_quarter")
        div_long = provider.get_dividend_events(universe_for_prices, quarter_start, quarter_end)
    else:
        div_long = div_long.loc[div_long["RIC"].astype(str).isin(universe_for_prices)].copy()
    if div_long.empty:
        div_map: dict[tuple[pd.Timestamp, str], float] = {}
    else:
        div = div_long.copy()
        div["Date"] = pd.to_datetime(div["Date"], errors="coerce")
        div["Dividend"] = pd.to_numeric(div["Dividend"], errors="coerce")
        div = div.dropna(subset=["Date", "RIC", "Dividend"])
        div = div.groupby(["Date", "RIC"], as_index=False)["Dividend"].sum()
        div_map = {(r.Date, str(r.RIC)): float(r.Dividend) for r in div.itertuples(index=False)}

    rebalance_date: pd.Timestamp | None = first_valuation_date if do_rebalance else None
    trades: list[dict[str, Any]] = []
    transaction_cost_cumulative = float(prev_transaction_cost_cumulative)
    cash = float(prev_cash)
    if abs(cash) < 1e-10:
        cash = 0.0
    new_shares = {
        str(ric): float(shares)
        for ric, shares in prev_shares.items()
        if abs(float(shares)) > eps
    }
    entry_price_reference = {
        str(ric): float(price_ref)
        for ric, price_ref in (prev_entry_price_reference or {}).items()
        if str(ric) not in bond_set and abs(float(new_shares.get(str(ric), 0.0))) > eps
    }

    if do_rebalance:
        prices_at_rebalance = price_pivot.loc[first_valuation_date]
        all_trade_rics = sorted(set(prior_rics).union(selection_rics).union(bond_set))
        prices_by_ric: dict[str, float] = {}
        bids_by_ric: dict[str, float] = {}
        asks_by_ric: dict[str, float] = {}
        for ric in all_trade_rics:
            px = float(prices_at_rebalance.get(ric, float("nan")))
            if pd.isna(px) or px <= 0:
                continue
            prices_by_ric[ric] = px
            bids_by_ric[ric] = (
                float(bid_pivot.loc[first_valuation_date, ric])
                if (not bid_pivot.empty and first_valuation_date in bid_pivot.index and ric in bid_pivot.columns)
                else float("nan")
            )
            asks_by_ric[ric] = (
                float(ask_pivot.loc[first_valuation_date, ric])
                if (not ask_pivot.empty and first_valuation_date in ask_pivot.index and ric in ask_pivot.columns)
                else float("nan")
            )

        rebalance_result = rebalance_to_target_with_costs(
            prices_by_ric=prices_by_ric,
            shares_by_ric={ric: float(prev_shares.get(ric, 0.0)) for ric in all_trade_rics},
            portfolio_cash=float(prev_cash),
            target_weights=target_weights,
            tx=transaction_costs,
            bid_by_ric=bids_by_ric,
            ask_by_ric=asks_by_ric,
        )
        cash = float(rebalance_result.cash_after)
        if abs(cash) < 1e-10:
            cash = 0.0
        new_shares = {
            str(ric): float(shares)
            for ric, shares in rebalance_result.shares_after.items()
            if abs(float(shares)) > eps
        }

        for row in rebalance_result.trade_rows:
            if abs(float(row["trade_shares"])) <= eps:
                continue
            trades.append(
                {
                    "run_id": run_id,
                    "date": first_valuation_date.date().isoformat(),
                    "quarter": quarter_label,
                    "ric": str(row["ric"]),
                    "price": float(row["reference_price"]),
                    "trade_shares": float(row["trade_shares"]),
                    "trade_value": float(row["trade_value"]),
                    "reference_price": float(row["reference_price"]),
                    "execution_price": float(row["execution_price"]),
                    "bid_price": float(row["bid_price"]),
                    "ask_price": float(row["ask_price"]),
                    "gross_notional": float(row["gross_notional"]),
                    "spread_bps_used": float(row["spread_bps_used"]),
                    "slippage_bps_used": float(row["slippage_bps_used"]),
                    "commission_cost": float(row["commission_cost"]),
                    "slippage_cost": float(row["slippage_cost"]),
                    "spread_cost": float(row["spread_cost"]),
                    "total_transaction_cost": float(row["total_transaction_cost"]),
                    "net_cash_flow": float(row["net_cash_flow"]),
                    "reason": "quarterly_rotation",
                }
            )

        refreshed_entry_refs: dict[str, float] = {}
        for ric, shares_after in new_shares.items():
            if ric in bond_set or shares_after <= eps:
                continue
            prev_qty = float(prev_shares.get(ric, 0.0))
            if prev_qty > eps and ric in entry_price_reference:
                refreshed_entry_refs[ric] = float(entry_price_reference[ric])
            else:
                refreshed_entry_refs[ric] = float(prices_at_rebalance.get(ric))
        entry_price_reference = refreshed_entry_refs

    holdings_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    scheduled_stop_sales: set[str] = set()

    for idx, dt in enumerate(quarter_dates):
        dividend_cash_daily = 0.0
        transaction_cost_daily = 0.0
        commission_cost_daily = 0.0
        slippage_cost_daily = 0.0
        spread_cost_daily = 0.0

        if rebalance_date is not None and dt == rebalance_date:
            rebalance_tx = [
                trade
                for trade in trades
                if trade["date"] == dt.date().isoformat() and trade["reason"] == "quarterly_rotation"
            ]
            transaction_cost_daily += float(sum(float(row["total_transaction_cost"]) for row in rebalance_tx))
            commission_cost_daily += float(sum(float(row["commission_cost"]) for row in rebalance_tx))
            slippage_cost_daily += float(sum(float(row["slippage_cost"]) for row in rebalance_tx))
            spread_cost_daily += float(sum(float(row["spread_cost"]) for row in rebalance_tx))
            transaction_cost_cumulative += transaction_cost_daily

        if baseline_sell_enabled and scheduled_stop_sales:
            prices_at_stop = price_pivot.loc[dt]
            refuge_bond = _pick_refuge_bond(
                bond_rics=bond_rics,
                refuge_score_df=refuge_score_df,
                prices_at_date=prices_at_stop,
            )
            if refuge_bond is None and scheduled_stop_sales:
                raise ValueError(
                    "Baseline sell triggered but no Treasury ETF refuge asset had usable pricing on "
                    f"{dt.date().isoformat()}."
                )
            if refuge_bond is not None:
                active_rics = sorted(
                    {
                        ric
                        for ric, shares in new_shares.items()
                        if abs(float(shares)) > eps and ric in price_pivot.columns
                    }.union({refuge_bond})
                )
                current_market = {
                    ric: float(new_shares.get(ric, 0.0)) * float(prices_at_stop.get(ric, 0.0))
                    for ric in active_rics
                    if not pd.isna(prices_at_stop.get(ric)) and float(prices_at_stop.get(ric, 0.0)) > 0
                }
                stop_rics = [
                    ric
                    for ric in sorted(scheduled_stop_sales)
                    if abs(float(new_shares.get(ric, 0.0))) > eps and ric in current_market
                ]
                if stop_rics:
                    total_before = float(cash + sum(current_market.values()))
                    target_values = dict(current_market)
                    stop_value = 0.0
                    for ric in stop_rics:
                        stop_value += float(target_values.get(ric, 0.0))
                        target_values[ric] = 0.0
                    target_values[refuge_bond] = float(target_values.get(refuge_bond, 0.0)) + stop_value
                    stop_target_weights = (
                        {
                            ric: float(value) / total_before
                            for ric, value in target_values.items()
                            if total_before > 0 and float(value) > 0
                        }
                        if total_before > 0
                        else {}
                    )
                    prices_by_ric = {ric: float(prices_at_stop.get(ric)) for ric in current_market}
                    bids_by_ric = {
                        ric: (
                            float(bid_pivot.loc[dt, ric])
                            if (not bid_pivot.empty and dt in bid_pivot.index and ric in bid_pivot.columns)
                            else float("nan")
                        )
                        for ric in prices_by_ric
                    }
                    asks_by_ric = {
                        ric: (
                            float(ask_pivot.loc[dt, ric])
                            if (not ask_pivot.empty and dt in ask_pivot.index and ric in ask_pivot.columns)
                            else float("nan")
                        )
                        for ric in prices_by_ric
                    }
                    stop_result = rebalance_to_target_with_costs(
                        prices_by_ric=prices_by_ric,
                        shares_by_ric={ric: float(new_shares.get(ric, 0.0)) for ric in prices_by_ric},
                        portfolio_cash=float(cash),
                        target_weights=stop_target_weights,
                        tx=transaction_costs,
                        bid_by_ric=bids_by_ric,
                        ask_by_ric=asks_by_ric,
                    )
                    cash = float(stop_result.cash_after)
                    if abs(cash) < 1e-10:
                        cash = 0.0
                    new_shares = {
                        str(ric): float(shares)
                        for ric, shares in stop_result.shares_after.items()
                        if abs(float(shares)) > eps
                    }
                    transaction_cost_daily += float(stop_result.transaction_cost_total)
                    commission_cost_daily += float(stop_result.commission_cost_total)
                    slippage_cost_daily += float(stop_result.slippage_cost_total)
                    spread_cost_daily += float(stop_result.spread_cost_total)
                    transaction_cost_cumulative += float(stop_result.transaction_cost_total)
                    for ric in stop_rics:
                        entry_price_reference.pop(ric, None)
                    for row in stop_result.trade_rows:
                        trade_shares = float(row["trade_shares"])
                        if abs(trade_shares) <= eps:
                            continue
                        ric = str(row["ric"])
                        if ric in stop_rics and trade_shares < 0:
                            reason = "baseline_sell_exit"
                        elif ric == refuge_bond and trade_shares > 0:
                            reason = "baseline_sell_refuge_buy"
                        else:
                            reason = "baseline_sell_rebalance"
                        trades.append(
                            {
                                "run_id": run_id,
                                "date": dt.date().isoformat(),
                                "quarter": quarter_label,
                                "ric": ric,
                                "price": float(row["reference_price"]),
                                "trade_shares": trade_shares,
                                "trade_value": float(row["trade_value"]),
                                "reference_price": float(row["reference_price"]),
                                "execution_price": float(row["execution_price"]),
                                "bid_price": float(row["bid_price"]),
                                "ask_price": float(row["ask_price"]),
                                "gross_notional": float(row["gross_notional"]),
                                "spread_bps_used": float(row["spread_bps_used"]),
                                "slippage_bps_used": float(row["slippage_bps_used"]),
                                "commission_cost": float(row["commission_cost"]),
                                "slippage_cost": float(row["slippage_cost"]),
                                "spread_cost": float(row["spread_cost"]),
                                "total_transaction_cost": float(row["total_transaction_cost"]),
                                "net_cash_flow": float(row["net_cash_flow"]),
                                "reason": reason,
                            }
                        )
                    scheduled_stop_sales = set()

        active_rics = [
            ric
            for ric, shares in sorted(new_shares.items())
            if abs(float(shares)) > eps and ric in price_pivot.columns and not pd.isna(price_pivot.loc[dt, ric])
        ]
        market_by_ric: dict[str, float] = {}
        for ric in active_rics:
            px = float(price_pivot.loc[dt, ric])
            shares = float(new_shares.get(ric, 0.0))
            div_ps = float(div_map.get((dt, ric), 0.0))
            div_cash = shares * div_ps
            if div_cash > 0:
                cash += div_cash
            dividend_cash_daily += div_cash
            market_by_ric[ric] = shares * px

        portfolio_market = float(sum(market_by_ric.values()))
        portfolio_total = portfolio_market + cash
        portfolio_total_gross = portfolio_total + transaction_cost_cumulative

        for ric in active_rics:
            mv = float(market_by_ric.get(ric, 0.0))
            w = mv / portfolio_total if portfolio_total > 0 else 0.0
            holdings_rows.append(
                {
                    "run_id": run_id,
                    "date": dt.date().isoformat(),
                    "quarter": quarter_label,
                    "ric": ric,
                    "shares": float(new_shares.get(ric, 0.0)),
                    "close": float(price_pivot.loc[dt, ric]),
                    "market_value": mv,
                    "cash_balance": cash,
                    "dividend_cash_daily": float(div_map.get((dt, ric), 0.0)) * float(new_shares.get(ric, 0.0)),
                    "total_value": mv,
                    "weight_eod": w,
                }
            )

        portfolio_rows.append(
            {
                "run_id": run_id,
                "date": dt.date().isoformat(),
                "quarter": quarter_label,
                "portfolio_market_value": portfolio_market,
                "portfolio_cash_balance": cash,
                "portfolio_total_value": portfolio_total,
                "portfolio_total_value_gross": portfolio_total_gross,
                "portfolio_dividend_cash_daily": dividend_cash_daily,
                "portfolio_transaction_cost_daily": transaction_cost_daily,
                "portfolio_commission_cost_daily": commission_cost_daily,
                "portfolio_slippage_cost_daily": slippage_cost_daily,
                "portfolio_spread_cost_daily": spread_cost_daily,
                "portfolio_transaction_cost_cumulative": transaction_cost_cumulative,
                "rebalance_flag": int(rebalance_date is not None and dt == rebalance_date),
            }
        )

        if baseline_sell_enabled and idx < len(quarter_dates) - 1:
            next_scheduled = set()
            for ric in active_rics:
                if ric in bond_set:
                    continue
                reference_price = float(entry_price_reference.get(ric, 0.0))
                if reference_price <= 0:
                    continue
                px = float(price_pivot.loc[dt, ric])
                if px <= reference_price * (1.0 - float(baseline_sell_threshold)):
                    next_scheduled.add(ric)
            scheduled_stop_sales.update(next_scheduled)

    holdings_df = pd.DataFrame(holdings_rows)
    portfolio_df = pd.DataFrame(portfolio_rows)
    trades_df = pd.DataFrame(trades)
    entry_price_reference = {
        ric: float(price_ref)
        for ric, price_ref in entry_price_reference.items()
        if ric not in bond_set and abs(float(new_shares.get(ric, 0.0))) > eps
    }
    return (
        new_shares,
        cash,
        transaction_cost_cumulative,
        trades_df,
        holdings_df,
        portfolio_df,
        price_long,
        div_long,
        bid_ask_long,
        entry_price_reference,
    )


def _strategy_or_default(config: PortfolioConfig) -> StrategyConfig:
    if config.strategy is None:
        return StrategyConfig(mode="dynamic_100_25")
    if config.strategy.mode == "static":
        return StrategyConfig(mode="dynamic_100_25")
    return config.strategy


def run_dynamic_rotation(
    *,
    config: PortfolioConfig,
    provider: StrategyDataProvider,
    store: StrategyStore | None,
    start_date: str,
    end_date: str,
    run_id: str | None = None,
    evaluation_context: dict[str, Any] | None = None,
) -> DynamicRunResult:
    strategy = _strategy_or_default(config)
    allocation_strategy = _strategy_allocation_strategy(strategy)
    bond_rics = _strategy_bond_rics(strategy)
    logger = get_logger("dividend_portfolio.strategy.engine")

    run_id = run_id or utc_now_id()
    created_at = datetime.now(timezone.utc).isoformat()
    if store is not None:
        store.write_run_metadata(
            run_id=run_id,
            created_at_utc=created_at,
            start_date=start_date,
            end_date=end_date,
            config={
                "portfolio": asdict(config),
                "strategy": asdict(strategy),
                "evaluation_context": evaluation_context or {},
            },
        )

    parquet = ParquetSidecarWriter(
        strategy.parquet_dir,
        enabled=(strategy.parquet_enabled and store is not None),
    )

    quarters = _quarter_windows(start_date, end_date)
    if not quarters:
        raise ValueError("No quarter windows in requested range")

    candidate_rows: list[pd.DataFrame] = []
    score_rows: list[pd.DataFrame] = []
    weight_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    holdings_rows: list[pd.DataFrame] = []
    portfolio_rows: list[pd.DataFrame] = []

    _, first_start, _ = quarters[0]
    candidates, market_caps, payer_events, candidate_stats = _build_top100_candidates(
        provider,
        as_of_date=first_start,
        lookback_months=strategy.dividend_payer_lookback_months,
        candidate_count=strategy.candidate_count,
        supplemental_rics=bond_rics,
    )
    initial_candidates = candidates.loc[~candidates["RIC"].astype(str).isin(set(bond_rics))].copy()
    if len(initial_candidates) < strategy.portfolio_size:
        raise ValueError(
            "Initial candidate set too small "
            f"({len(initial_candidates)}) for portfolio_size={strategy.portfolio_size}. "
            f"Diagnostics: {candidate_stats}"
        )
    pending_selection = PendingSelection(
        selection=_selection_template(
            select_initial_portfolio_by_market_cap(initial_candidates, strategy.portfolio_size)
        ),
        source="initial_market_cap",
        weight_strategy="fixed",
    )

    shares: dict[str, float] = {}
    entry_price_reference: dict[str, float] = {}
    cash = float(config.initial_capital)
    transaction_cost_cumulative = 0.0
    rebalance_interval_quarters = max(int(strategy.rebalance_interval_quarters), 1)
    latest_completed_scores = pd.DataFrame(columns=["RIC", "Score"])

    for i, (quarter_label, q_start, q_end) in enumerate(quarters):
        logger.info("Processing %s (%s..%s)", quarter_label, q_start, q_end)
        if i > 0:
            candidates, market_caps, payer_events, candidate_stats = _build_top100_candidates(
                provider,
                as_of_date=q_start,
                lookback_months=strategy.dividend_payer_lookback_months,
                candidate_count=strategy.candidate_count,
                supplemental_rics=bond_rics,
            )
            if candidates.empty:
                raise ValueError(
                    f"No candidates returned for {quarter_label} as-of {q_start}. "
                    f"Diagnostics: {candidate_stats}"
                )

        cand = candidates.copy()
        cand = cand.rename(columns={"RIC": "ric", "MarketCap": "market_cap", "MarketCapDate": "market_cap_date"})
        cand["run_id"] = run_id
        cand["quarter"] = quarter_label
        cand["as_of_date"] = q_start
        cand["is_dividend_payer_12m"] = 1
        cand["rank_market_cap"] = cand["RankByMarketCap"].astype(int)
        cand = cand[
            [
                "run_id",
                "quarter",
                "as_of_date",
                "ric",
                "market_cap",
                "market_cap_date",
                "is_dividend_payer_12m",
                "rank_market_cap",
            ]
        ]
        if store is not None:
            store.upsert_candidate_universe(cand)
        candidate_rows.append(cand)
        parquet.write_constituents(cand, quarter_label)
        parquet.write_market_caps(market_caps, quarter_label)

        current_selection = _materialize_pending_selection(
            pending_selection,
            market_caps=market_caps,
        )
        current_selection = current_selection[["RIC", "Weight", "RankInPortfolio"]].copy()
        current_rics = current_selection["RIC"].astype(str).tolist()
        candidate_rics = candidates["RIC"].astype(str).tolist()
        quarter_fetch_rics = sorted(
            set(candidate_rics).union(current_rics).union(set(shares.keys())).union(set(bond_rics))
        )
        should_rebalance_this_quarter = (i % rebalance_interval_quarters) == 0

        # Fetch once per quarter for all required names, then reuse for simulation and scoring.
        quarter_prices_all = provider.get_close_history(quarter_fetch_rics, q_start, q_end)
        quarter_divs_all = provider.get_dividend_events(quarter_fetch_rics, q_start, q_end)
        quarter_bid_ask_all = provider.get_bid_ask_history(quarter_fetch_rics, q_start, q_end)
        if strategy.baseline_sell_enabled:
            available_bond_prices = (
                set(quarter_prices_all["RIC"].dropna().astype(str).tolist())
                if not quarter_prices_all.empty and "RIC" in quarter_prices_all.columns
                else set()
            )
            if bond_rics and not any(ric in available_bond_prices for ric in bond_rics):
                raise ValueError(
                    "Baseline sell is enabled but no Treasury ETF close history was available for "
                    f"{quarter_label}. Required one of {bond_rics!r}."
                )
        parquet.write_prices(quarter_prices_all, quarter_label)
        parquet.write_dividends(quarter_divs_all, quarter_label)
        parquet.write_bid_ask(quarter_bid_ask_all, quarter_label)

        (
            shares,
            cash,
            transaction_cost_cumulative,
            trades_df,
            holdings_df,
            portfolio_df,
            _prices_selected,
            _dividends_selected,
            _bid_ask_selected,
            entry_price_reference,
        ) = _simulate_quarter(
            run_id=run_id,
            quarter_label=quarter_label,
            quarter_start=q_start,
            quarter_end=q_end,
            selection=current_selection,
            prev_shares=shares,
            prev_entry_price_reference=entry_price_reference,
            prev_cash=cash,
            transaction_costs=config.transaction_costs,
            prev_transaction_cost_cumulative=transaction_cost_cumulative,
            do_rebalance=should_rebalance_this_quarter,
            provider=None,
            price_long=quarter_prices_all,
            div_long=quarter_divs_all,
            bid_ask_long=quarter_bid_ask_all,
            baseline_sell_enabled=bool(strategy.baseline_sell_enabled),
            baseline_sell_threshold=float(strategy.baseline_sell_threshold),
            bond_rics=bond_rics,
            refuge_score_df=latest_completed_scores,
        )
        # Selected slices are included in quarter-wide sidecar snapshots above.

        if not trades_df.empty:
            if store is not None:
                store.upsert_trades(trades_df)
            trade_rows.append(trades_df)
        if not holdings_df.empty:
            if store is not None:
                store.upsert_holdings_daily(holdings_df)
            holdings_rows.append(holdings_df)
        if not portfolio_df.empty:
            if store is not None:
                store.upsert_portfolio_daily(portfolio_df)
            portfolio_rows.append(portfolio_df)

        rebalance_date = portfolio_df.loc[portfolio_df["rebalance_flag"] == 1, "date"].min()
        if pd.isna(rebalance_date):
            rebalance_date = str(portfolio_df["date"].min()) if not portfolio_df.empty else q_start
        weights_out = current_selection.copy()
        weights_out["run_id"] = run_id
        weights_out["quarter"] = quarter_label
        weights_out["rebalance_date"] = rebalance_date
        weights_out["source"] = pending_selection.source
        weights_out = weights_out.rename(
            columns={
                "RIC": "ric",
                "Weight": "weight",
                "RankInPortfolio": "rank_in_portfolio",
            }
        )
        weights_out = weights_out[
            ["run_id", "quarter", "rebalance_date", "source", "ric", "weight", "rank_in_portfolio"]
        ]
        if store is not None:
            store.upsert_target_weights(weights_out)
        weight_rows.append(weights_out)

        end_of_quarter_selection = _selection_from_holdings(holdings_df)

        score_prices = quarter_prices_all.loc[quarter_prices_all["RIC"].astype(str).isin(candidate_rics)].copy()
        score_divs = quarter_divs_all.loc[quarter_divs_all["RIC"].astype(str).isin(candidate_rics)].copy()
        candidate_score_df = compute_quarter_dividend_yield_scores(
            score_prices,
            score_divs,
            denominator=strategy.yield_denominator,
        )
        score_df_out = candidate_score_df.copy()
        score_df_out["run_id"] = run_id
        score_df_out["quarter"] = quarter_label
        score_df_out["quarter_start"] = q_start
        score_df_out["quarter_end"] = q_end
        score_df_out = score_df_out.rename(
            columns={
                "RIC": "ric",
                "AvgClose": "avg_close",
                "DividendSumPS": "dividend_sum_ps",
                "Score": "score",
                "RankByScore": "rank_score",
            }
        )
        score_df_out = score_df_out[
            [
                "run_id",
                "quarter",
                "quarter_start",
                "quarter_end",
                "ric",
                "avg_close",
                "dividend_sum_ps",
                "score",
                "rank_score",
            ]
        ]
        if store is not None:
            store.upsert_quarter_scores(score_df_out)
        score_rows.append(score_df_out)
        latest_completed_scores = candidate_score_df[["RIC", "Score"]].copy()

        should_update_next_selection = ((i + 1) % rebalance_interval_quarters) == 0
        if should_update_next_selection:
            selection_scores = candidate_score_df[["RIC", "Score"]].copy()
            if strategy.selection_policy.name == "replace_bottom_n":
                policy_rics = sorted(set(candidate_rics).union(set(end_of_quarter_selection["RIC"].astype(str).tolist())))
                policy_prices = quarter_prices_all.loc[quarter_prices_all["RIC"].astype(str).isin(policy_rics)].copy()
                policy_divs = quarter_divs_all.loc[quarter_divs_all["RIC"].astype(str).isin(policy_rics)].copy()
                policy_scores = compute_quarter_dividend_yield_scores(
                    policy_prices,
                    policy_divs,
                    denominator=strategy.yield_denominator,
                )
                selection_scores = policy_scores[["RIC", "Score"]].copy()

            next_selection, next_weight_source = _select_next_portfolio_by_policy(
                strategy=strategy,
                score_df=selection_scores,
                candidates=candidates,
                current_selection=end_of_quarter_selection,
            )
            pending_selection = PendingSelection(
                selection=_selection_template(next_selection),
                source=next_weight_source,
                weight_strategy=allocation_strategy,
            )
        else:
            pending_selection = PendingSelection(
                selection=_selection_template(end_of_quarter_selection),
                source="carry_forward_no_rebalance",
                weight_strategy="fixed",
            )

    return DynamicRunResult(
        run_id=run_id,
        candidate_universe=pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame(),
        quarter_scores=pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame(),
        target_weights=pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame(),
        trades=pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(),
        holdings_daily=pd.concat(holdings_rows, ignore_index=True) if holdings_rows else pd.DataFrame(),
        portfolio_daily=pd.concat(portfolio_rows, ignore_index=True) if portfolio_rows else pd.DataFrame(),
    )
