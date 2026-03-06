from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..logging_utils import get_logger
from ..models import PortfolioConfig, StrategyConfig
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    constituents = provider.get_sp500_constituents_as_of(as_of_date)
    if not constituents:
        return (
            pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate", "RankByMarketCap"]),
            pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate"]),
            pd.DataFrame(columns=["RIC", "Date", "Dividend"]),
        )

    market_caps = provider.get_market_cap_snapshot(constituents, as_of_date)
    payer_start = trailing_lookback_start(as_of_date, lookback_months)
    payer_events = provider.get_dividend_events(constituents, payer_start, as_of_date)
    dividend_payers = set(payer_events["RIC"].dropna().astype(str)) if not payer_events.empty else set()

    candidates = select_top_candidates_by_market_cap(
        market_caps=market_caps,
        dividend_payers=dividend_payers,
        candidate_count=candidate_count,
    )
    return candidates, market_caps, payer_events


def _normalize_weights(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    denom = float(pd.to_numeric(out[score_col], errors="coerce").fillna(0.0).sum())
    if denom > 0:
        out["Weight"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0) / denom
    else:
        out["Weight"] = 1.0 / len(out)
    return out


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


def _pivot_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    out = prices.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["CLOSE"] = pd.to_numeric(out["CLOSE"], errors="coerce")
    out = out.dropna(subset=["Date", "RIC", "CLOSE"])
    pivot = out.pivot_table(index="Date", columns="RIC", values="CLOSE", aggfunc="last")
    return pivot.sort_index()


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
    provider: StrategyDataProvider | None = None,
    price_long: pd.DataFrame | None = None,
    div_long: pd.DataFrame | None = None,
) -> tuple[dict[str, float], float, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_rics = selection["RIC"].astype(str).tolist()
    target_weights = dict(zip(selection_rics, selection["Weight"].astype(float), strict=False))
    prior_rics = list(prev_shares.keys())
    universe_for_prices = sorted(set(selection_rics).union(prior_rics))

    if price_long is None:
        if provider is None:
            raise ValueError("Either provider or price_long must be supplied to _simulate_quarter")
        price_long = provider.get_close_history(universe_for_prices, quarter_start, quarter_end)
    else:
        price_long = price_long.loc[price_long["RIC"].astype(str).isin(universe_for_prices)].copy()

    price_pivot = _pivot_prices(price_long).ffill()
    if price_pivot.empty:
        raise ValueError(f"No price data returned for quarter {quarter_label}")

    rebalance_date = _find_rebalance_date(price_pivot, selection_rics, quarter_start)
    quarter_dates = price_pivot.index[price_pivot.index >= rebalance_date]
    quarter_dates = quarter_dates[quarter_dates <= pd.Timestamp(quarter_end)]
    if len(quarter_dates) == 0:
        raise ValueError(f"No tradable dates in quarter window for {quarter_label}")

    prices_at_rebalance = price_pivot.loc[rebalance_date]
    pre_trade_market = 0.0
    for ric, shares in prev_shares.items():
        px = float(prices_at_rebalance.get(ric, float("nan")))
        if pd.notna(px) and px > 0:
            pre_trade_market += shares * px

    total_before = pre_trade_market + prev_cash
    trades: list[dict[str, Any]] = []

    new_shares: dict[str, float] = {}
    trade_value_sum = 0.0
    all_trade_rics = sorted(set(prior_rics).union(selection_rics))
    for ric in all_trade_rics:
        px = float(prices_at_rebalance.get(ric, float("nan")))
        if pd.isna(px) or px <= 0:
            continue
        old_shares = float(prev_shares.get(ric, 0.0))
        target_shares = 0.0
        if ric in target_weights:
            target_value = total_before * target_weights[ric]
            target_shares = target_value / px
            new_shares[ric] = target_shares

        trade_shares = target_shares - old_shares
        trade_value = trade_shares * px
        trade_value_sum += trade_value
        if abs(trade_shares) > 1e-12:
            trades.append(
                {
                    "run_id": run_id,
                    "date": rebalance_date.date().isoformat(),
                    "quarter": quarter_label,
                    "ric": ric,
                    "price": px,
                    "trade_shares": trade_shares,
                    "trade_value": trade_value,
                    "reason": "quarterly_rotation",
                }
            )

    cash = float(prev_cash - trade_value_sum)
    if abs(cash) < 1e-10:
        cash = 0.0

    if div_long is None:
        if provider is None:
            raise ValueError("Either provider or div_long must be supplied to _simulate_quarter")
        div_long = provider.get_dividend_events(selection_rics, quarter_start, quarter_end)
    else:
        div_long = div_long.loc[div_long["RIC"].astype(str).isin(selection_rics)].copy()
    if div_long.empty:
        div_map = {}
    else:
        div = div_long.copy()
        div["Date"] = pd.to_datetime(div["Date"], errors="coerce")
        div["Dividend"] = pd.to_numeric(div["Dividend"], errors="coerce")
        div = div.dropna(subset=["Date", "RIC", "Dividend"])
        div = div.groupby(["Date", "RIC"], as_index=False)["Dividend"].sum()
        div_map = {(r.Date, r.RIC): float(r.Dividend) for r in div.itertuples(index=False)}

    holdings_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []

    for dt in quarter_dates:
        dividend_cash_daily = 0.0
        market_by_ric: dict[str, float] = {}
        for ric in selection_rics:
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
        for ric in selection_rics:
            mv = market_by_ric[ric]
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
                "portfolio_dividend_cash_daily": dividend_cash_daily,
                "rebalance_flag": int(dt == rebalance_date),
            }
        )

    holdings_df = pd.DataFrame(holdings_rows)
    portfolio_df = pd.DataFrame(portfolio_rows)
    trades_df = pd.DataFrame(trades)
    return (
        new_shares,
        cash,
        trades_df,
        holdings_df,
        portfolio_df,
        price_long,
        div_long,
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
    store: StrategyStore,
    start_date: str,
    end_date: str,
    run_id: str | None = None,
) -> DynamicRunResult:
    strategy = _strategy_or_default(config)
    logger = get_logger("dividend_portfolio.strategy.engine")

    run_id = run_id or utc_now_id()
    created_at = datetime.now(timezone.utc).isoformat()
    store.write_run_metadata(
        run_id=run_id,
        created_at_utc=created_at,
        start_date=start_date,
        end_date=end_date,
        config={
            "portfolio": asdict(config),
            "strategy": asdict(strategy),
        },
    )

    parquet = ParquetSidecarWriter(strategy.parquet_dir, enabled=strategy.parquet_enabled)

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
    candidates, market_caps, payer_events = _build_top100_candidates(
        provider,
        as_of_date=first_start,
        lookback_months=strategy.dividend_payer_lookback_months,
        candidate_count=strategy.candidate_count,
    )
    if len(candidates) < strategy.portfolio_size:
        raise ValueError(
            f"Initial candidate set too small ({len(candidates)}) for portfolio_size={strategy.portfolio_size}"
        )
    next_selection = select_initial_portfolio_by_market_cap(candidates, strategy.portfolio_size)
    next_weight_source = "initial_market_cap"

    shares: dict[str, float] = {}
    cash = float(config.initial_capital)

    for i, (quarter_label, q_start, q_end) in enumerate(quarters):
        logger.info("Processing %s (%s..%s)", quarter_label, q_start, q_end)
        if i > 0:
            candidates, market_caps, payer_events = _build_top100_candidates(
                provider,
                as_of_date=q_start,
                lookback_months=strategy.dividend_payer_lookback_months,
                candidate_count=strategy.candidate_count,
            )
            if candidates.empty:
                raise ValueError(f"No candidates returned for {quarter_label} as-of {q_start}")

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
        store.upsert_candidate_universe(cand)
        candidate_rows.append(cand)
        parquet.write_constituents(cand, quarter_label)
        parquet.write_market_caps(market_caps, quarter_label)

        current_selection = next_selection.copy()
        current_selection = _normalize_weights(current_selection, "Weight")
        current_selection = current_selection[["RIC", "Weight", "RankInPortfolio"]]
        current_rics = current_selection["RIC"].astype(str).tolist()
        candidate_rics = candidates["RIC"].astype(str).tolist()
        quarter_fetch_rics = sorted(set(candidate_rics).union(current_rics).union(set(shares.keys())))

        # Fetch once per quarter for all required names, then reuse for simulation and scoring.
        quarter_prices_all = provider.get_close_history(quarter_fetch_rics, q_start, q_end)
        quarter_divs_all = provider.get_dividend_events(quarter_fetch_rics, q_start, q_end)
        parquet.write_prices(quarter_prices_all, quarter_label)
        parquet.write_dividends(quarter_divs_all, quarter_label)

        (
            shares,
            cash,
            trades_df,
            holdings_df,
            portfolio_df,
            _prices_selected,
            _dividends_selected,
        ) = _simulate_quarter(
            run_id=run_id,
            quarter_label=quarter_label,
            quarter_start=q_start,
            quarter_end=q_end,
            selection=current_selection,
            prev_shares=shares,
            prev_cash=cash,
            provider=None,
            price_long=quarter_prices_all,
            div_long=quarter_divs_all,
        )
        # Selected slices are included in quarter-wide sidecar snapshots above.

        if not trades_df.empty:
            store.upsert_trades(trades_df)
            trade_rows.append(trades_df)
        if not holdings_df.empty:
            store.upsert_holdings_daily(holdings_df)
            holdings_rows.append(holdings_df)
        if not portfolio_df.empty:
            store.upsert_portfolio_daily(portfolio_df)
            portfolio_rows.append(portfolio_df)

        rebalance_date = portfolio_df.loc[portfolio_df["rebalance_flag"] == 1, "date"].min()
        weights_out = current_selection.copy()
        weights_out["run_id"] = run_id
        weights_out["quarter"] = quarter_label
        weights_out["rebalance_date"] = rebalance_date
        weights_out["source"] = next_weight_source
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
        store.upsert_target_weights(weights_out)
        weight_rows.append(weights_out)

        score_prices = quarter_prices_all.loc[quarter_prices_all["RIC"].astype(str).isin(candidate_rics)].copy()
        score_divs = quarter_divs_all.loc[quarter_divs_all["RIC"].astype(str).isin(candidate_rics)].copy()
        score_df = compute_quarter_dividend_yield_scores(
            score_prices,
            score_divs,
            denominator=strategy.yield_denominator,
        )
        score_df["run_id"] = run_id
        score_df["quarter"] = quarter_label
        score_df["quarter_start"] = q_start
        score_df["quarter_end"] = q_end
        score_df = score_df.rename(
            columns={
                "RIC": "ric",
                "AvgClose": "avg_close",
                "DividendSumPS": "dividend_sum_ps",
                "Score": "score",
                "RankByScore": "rank_score",
            }
        )
        score_df = score_df[
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
        store.upsert_quarter_scores(score_df)
        score_rows.append(score_df)

        selection_input = score_df.rename(columns={"ric": "RIC", "score": "Score"})
        next_selection = _select_next_portfolio_with_backfill(
            selection_input[["RIC", "Score"]],
            candidates,
            portfolio_size=strategy.portfolio_size,
        )
        next_weight_source = "quarter_dividend_yield_score"

    return DynamicRunResult(
        run_id=run_id,
        candidate_universe=pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame(),
        quarter_scores=pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame(),
        target_weights=pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame(),
        trades=pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(),
        holdings_daily=pd.concat(holdings_rows, ignore_index=True) if holdings_rows else pd.DataFrame(),
        portfolio_daily=pd.concat(portfolio_rows, ignore_index=True) if portfolio_rows else pd.DataFrame(),
    )
