from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ..models import PortfolioConfig, SimulationResult
from .rebalancer import (
    apply_rebalance,
    apply_rebalance_with_costs,
    build_rebalance_dates,
    compute_drifts,
    should_rebalance,
)
from .split_math import build_split_multiplier


def _sanitize_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
            out = out.set_index("Date")
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")

    out = out.loc[out.index.notna()].sort_index()
    if "CLOSE" not in out.columns:
        raise ValueError("History is missing CLOSE column")

    out["CLOSE"] = pd.to_numeric(out["CLOSE"], errors="coerce")
    out = out.loc[out["CLOSE"].notna() & (out["CLOSE"] > 0)].copy()

    out["Dividend"] = pd.to_numeric(out.get("Dividend", 0.0), errors="coerce").fillna(0.0)
    out["SplitFactor"] = pd.to_numeric(out.get("SplitFactor", 1.0), errors="coerce").fillna(1.0)
    if "BID" in out.columns:
        out["BID"] = pd.to_numeric(out["BID"], errors="coerce")
    if "ASK" in out.columns:
        out["ASK"] = pd.to_numeric(out["ASK"], errors="coerce")

    if "cum_factor" in out.columns:
        out["cum_factor"] = pd.to_numeric(out["cum_factor"], errors="coerce").ffill().fillna(1.0)
    else:
        sf = out["SplitFactor"].where(out["SplitFactor"] > 0, 1.0)
        split_mult = sf.where(sf >= 1.0, 1.0 / sf)
        out["cum_factor"] = split_mult.cumprod()

    return out


def _clip_dates(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    out = df.loc[df.index >= start]
    if end is not None:
        out = out.loc[out.index <= end]
    return out


def _first_common_date(histories: dict[str, pd.DataFrame]) -> pd.Timestamp:
    common_index: pd.DatetimeIndex | None = None
    for df in histories.values():
        idx = df.index
        common_index = idx if common_index is None else common_index.intersection(idx)

    if common_index is None or common_index.empty:
        raise ValueError("No common trading date exists across selected assets.")

    return common_index[0]


def _master_union_index(histories: dict[str, pd.DataFrame], start: pd.Timestamp) -> pd.DatetimeIndex:
    union: pd.DatetimeIndex | None = None
    for df in histories.values():
        idx = df.index[df.index >= start]
        union = idx if union is None else union.union(idx)

    if union is None or union.empty:
        raise ValueError("No portfolio dates found after effective start.")

    return union.sort_values()


def _prepare_asset_frame(
    df: pd.DataFrame,
    master_dates: pd.DatetimeIndex,
    *,
    use_cum_factor: bool,
    auto_align_splits: bool,
) -> pd.DataFrame:
    out = df.reindex(master_dates).copy()

    out["CLOSE"] = pd.to_numeric(out["CLOSE"], errors="coerce").ffill()
    if out["CLOSE"].isna().any():
        raise ValueError("Missing CLOSE values after forward-fill. Check asset start dates.")

    out["Dividend"] = pd.to_numeric(out.get("Dividend", 0.0), errors="coerce").fillna(0.0)
    out["SplitFactor"] = pd.to_numeric(out.get("SplitFactor", 1.0), errors="coerce").fillna(1.0)
    out["cum_factor"] = pd.to_numeric(out.get("cum_factor", 1.0), errors="coerce").ffill().fillna(1.0)
    if "BID" in out.columns:
        out["BID"] = pd.to_numeric(out["BID"], errors="coerce").ffill()
    else:
        out["BID"] = float("nan")
    if "ASK" in out.columns:
        out["ASK"] = pd.to_numeric(out["ASK"], errors="coerce").ffill()
    else:
        out["ASK"] = float("nan")

    out["Split_Multiplier"] = build_split_multiplier(
        out,
        price_col="CLOSE",
        split_col="SplitFactor",
        cum_factor_col="cum_factor",
        use_cum_factor=use_cum_factor,
        auto_align=auto_align_splits,
    )

    return out


def simulate_portfolio(
    histories_by_ric: dict[str, pd.DataFrame],
    config: PortfolioConfig,
) -> SimulationResult:
    if not histories_by_ric:
        raise ValueError("No histories provided.")

    asset_order = [asset.ric for asset in config.assets]

    sanitized: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(config.start_date)
    end_ts = pd.Timestamp(config.end_date) if config.end_date is not None else None

    for ric in asset_order:
        if ric not in histories_by_ric:
            raise ValueError(f"Missing history for configured asset {ric}")
        cleaned = _sanitize_history(histories_by_ric[ric])
        clipped = _clip_dates(cleaned, start_ts, end_ts)
        if clipped.empty:
            raise ValueError(f"No usable rows for {ric} in requested date range.")
        sanitized[ric] = clipped

    effective_start = _first_common_date(sanitized)
    master_dates = _master_union_index(sanitized, effective_start)

    prepared: dict[str, pd.DataFrame] = {}
    for ric in asset_order:
        prepared[ric] = _prepare_asset_frame(
            sanitized[ric].loc[sanitized[ric].index >= effective_start],
            master_dates,
            use_cum_factor=config.use_cum_factor,
            auto_align_splits=config.auto_align_splits,
        )

    target_weights = {asset.ric: float(asset.weight) for asset in config.assets}

    shares: dict[str, float] = {ric: 0.0 for ric in asset_order}
    cumulative_dividends: dict[str, float] = {ric: 0.0 for ric in asset_order}
    initial_trade_rows: list[dict[str, float | str]] = []
    initial_trade_shares_by_ric = {ric: 0.0 for ric in asset_order}
    initial_trade_value_by_ric = {ric: 0.0 for ric in asset_order}
    initial_trade_cost_by_ric = {ric: 0.0 for ric in asset_order}
    initial_trade_exec_px_by_ric = {ric: float("nan") for ric in asset_order}
    initial_commission_cost = 0.0
    initial_slippage_cost = 0.0
    initial_spread_cost = 0.0
    initial_total_cost = 0.0
    initial_established = False

    first_date = master_dates[0]
    prices_first = {ric: float(prepared[ric].loc[first_date, "CLOSE"]) for ric in asset_order}
    bids_first = {ric: float(prepared[ric].loc[first_date, "BID"]) for ric in asset_order}
    asks_first = {ric: float(prepared[ric].loc[first_date, "ASK"]) for ric in asset_order}

    portfolio_cash = float(config.initial_capital)
    if config.transaction_costs.enabled:
        (
            shares,
            portfolio_cash,
            initial_trade_shares_by_ric,
            initial_trade_value_by_ric,
            _,
            initial_trade_rows,
            initial_commission_cost,
            initial_slippage_cost,
            initial_spread_cost,
            initial_total_cost,
        ) = apply_rebalance_with_costs(
            prices_by_ric=prices_first,
            shares_by_ric=shares,
            portfolio_cash=portfolio_cash,
            target_weights=target_weights,
            transaction_costs=config.transaction_costs,
            bid_by_ric=bids_first,
            ask_by_ric=asks_first,
        )
        for row in initial_trade_rows:
            ric = str(row["ric"])
            initial_trade_cost_by_ric[ric] = float(row["total_transaction_cost"])
            initial_trade_exec_px_by_ric[ric] = float(row["execution_price"])
        initial_established = True
    else:
        for ric in asset_order:
            first_price = float(prepared[ric].loc[first_date, "CLOSE"])
            allocation = config.initial_capital * target_weights[ric]
            shares[ric] = allocation / first_price
        portfolio_cash = 0.0

    portfolio_cum_dividends = 0.0
    portfolio_cum_transaction_costs = 0.0

    rebalance_dates = (
        build_rebalance_dates(
            master_dates,
            frequency=config.rebalancing.frequency,
            trigger=config.rebalancing.trigger,
        )
        if config.rebalancing.enabled
        else set()
    )

    asset_records: dict[str, dict[str, list[float | pd.Timestamp]]] = {
        ric: defaultdict(list) for ric in asset_order
    }

    portfolio_records: dict[str, list[float | pd.Timestamp | bool]] = defaultdict(list)
    rebalance_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for date in master_dates:
        prices_by_ric: dict[str, float] = {}
        bids_by_ric: dict[str, float] = {}
        asks_by_ric: dict[str, float] = {}
        split_by_ric: dict[str, float] = {}
        div_ps_by_ric: dict[str, float] = {}
        div_cash_by_ric: dict[str, float] = {}

        for ric in asset_order:
            row = prepared[ric].loc[date]
            price = float(row["CLOSE"])
            split_mult = float(row["Split_Multiplier"])
            div_ps = float(row["Dividend"])
            bid = float(row["BID"]) if pd.notna(row.get("BID")) else float("nan")
            ask = float(row["ASK"]) if pd.notna(row.get("ASK")) else float("nan")

            if split_mult != 1.0:
                shares[ric] *= split_mult

            div_cash = 0.0
            if div_ps > 0:
                div_cash = shares[ric] * div_ps
                cumulative_dividends[ric] += div_cash
                if config.reinvest_dividends:
                    shares[ric] += div_cash / price
                else:
                    portfolio_cash += div_cash

            prices_by_ric[ric] = price
            bids_by_ric[ric] = bid
            asks_by_ric[ric] = ask
            split_by_ric[ric] = split_mult
            div_ps_by_ric[ric] = div_ps
            div_cash_by_ric[ric] = div_cash

        market_by_ric = {ric: shares[ric] * prices_by_ric[ric] for ric in asset_order}
        portfolio_total_before = sum(market_by_ric.values()) + portfolio_cash
        portfolio_cash_before = portfolio_cash

        weights_before = {
            ric: (market_by_ric[ric] / portfolio_total_before if portfolio_total_before > 0 else 0.0)
            for ric in asset_order
        }
        drifts = compute_drifts(market_by_ric, portfolio_total_before, target_weights)

        rebalance_flag = False
        trade_shares_by_ric = {ric: 0.0 for ric in asset_order}
        trade_value_by_ric = {ric: 0.0 for ric in asset_order}
        trade_total_cost_by_ric = {ric: 0.0 for ric in asset_order}
        trade_exec_price_by_ric = {ric: float("nan") for ric in asset_order}
        trade_commission_cost_by_ric = {ric: 0.0 for ric in asset_order}
        trade_slippage_cost_by_ric = {ric: 0.0 for ric in asset_order}
        trade_spread_cost_by_ric = {ric: 0.0 for ric in asset_order}
        trade_spread_bps_by_ric = {ric: 0.0 for ric in asset_order}
        trade_slippage_bps_by_ric = {ric: 0.0 for ric in asset_order}
        trade_reference_price_by_ric = {ric: float("nan") for ric in asset_order}
        trade_gross_notional_by_ric = {ric: 0.0 for ric in asset_order}
        trade_net_cash_flow_by_ric = {ric: 0.0 for ric in asset_order}
        transaction_cost_daily = 0.0
        commission_cost_daily = 0.0
        slippage_cost_daily = 0.0
        spread_cost_daily = 0.0

        if initial_established and date == first_date:
            rebalance_flag = True
            trade_shares_by_ric = dict(initial_trade_shares_by_ric)
            trade_value_by_ric = dict(initial_trade_value_by_ric)
            trade_total_cost_by_ric = dict(initial_trade_cost_by_ric)
            trade_exec_price_by_ric = dict(initial_trade_exec_px_by_ric)
            for ric in asset_order:
                trade_reference_price_by_ric[ric] = prices_by_ric[ric]
                trade_gross_notional_by_ric[ric] = abs(trade_shares_by_ric[ric]) * prices_by_ric[ric]
                trade_net_cash_flow_by_ric[ric] = -trade_value_by_ric[ric] - trade_total_cost_by_ric[ric]
            for row in initial_trade_rows:
                ric = str(row["ric"])
                trade_commission_cost_by_ric[ric] = float(row["commission_cost"])
                trade_slippage_cost_by_ric[ric] = float(row["slippage_cost"])
                trade_spread_cost_by_ric[ric] = float(row["spread_cost"])
                trade_spread_bps_by_ric[ric] = float(row["spread_bps_used"])
                trade_slippage_bps_by_ric[ric] = float(row["slippage_bps_used"])
                trade_reference_price_by_ric[ric] = float(row["reference_price"])
                trade_gross_notional_by_ric[ric] = float(row["gross_notional"])
                trade_net_cash_flow_by_ric[ric] = float(row["net_cash_flow"])
            transaction_cost_daily += float(initial_total_cost)
            commission_cost_daily += float(initial_commission_cost)
            slippage_cost_daily += float(initial_slippage_cost)
            spread_cost_daily += float(initial_spread_cost)

        if config.rebalancing.enabled and should_rebalance(
            date=date,
            rebalance_dates=rebalance_dates,
            drift_by_ric=drifts,
            drift_tolerance=config.rebalancing.drift_tolerance,
        ):
            if config.transaction_costs.enabled:
                (
                    shares,
                    portfolio_cash,
                    trade_shares_by_ric,
                    trade_value_by_ric,
                    _,
                    trade_rows,
                    commission_cost_daily,
                    slippage_cost_daily,
                    spread_cost_daily,
                    transaction_cost_daily,
                ) = apply_rebalance_with_costs(
                    prices_by_ric=prices_by_ric,
                    shares_by_ric=shares,
                    portfolio_cash=portfolio_cash,
                    target_weights=target_weights,
                    transaction_costs=config.transaction_costs,
                    bid_by_ric=bids_by_ric,
                    ask_by_ric=asks_by_ric,
                )
                for row in trade_rows:
                    ric = str(row["ric"])
                    trade_total_cost_by_ric[ric] = float(row["total_transaction_cost"])
                    trade_exec_price_by_ric[ric] = float(row["execution_price"])
                    trade_commission_cost_by_ric[ric] = float(row["commission_cost"])
                    trade_slippage_cost_by_ric[ric] = float(row["slippage_cost"])
                    trade_spread_cost_by_ric[ric] = float(row["spread_cost"])
                    trade_spread_bps_by_ric[ric] = float(row["spread_bps_used"])
                    trade_slippage_bps_by_ric[ric] = float(row["slippage_bps_used"])
                    trade_reference_price_by_ric[ric] = float(row["reference_price"])
                    trade_gross_notional_by_ric[ric] = float(row["gross_notional"])
                    trade_net_cash_flow_by_ric[ric] = float(row["net_cash_flow"])
            else:
                (
                    shares,
                    portfolio_cash,
                    trade_shares_by_ric,
                    trade_value_by_ric,
                    _,
                ) = apply_rebalance(
                    prices_by_ric=prices_by_ric,
                    shares_by_ric=shares,
                    portfolio_cash=portfolio_cash,
                    target_weights=target_weights,
                )
            rebalance_flag = True

        market_by_ric = {ric: shares[ric] * prices_by_ric[ric] for ric in asset_order}
        portfolio_total = sum(market_by_ric.values()) + portfolio_cash
        portfolio_cum_transaction_costs += float(transaction_cost_daily)
        portfolio_total_gross = portfolio_total + portfolio_cum_transaction_costs
        weights_after = {
            ric: (market_by_ric[ric] / portfolio_total if portfolio_total > 0 else 0.0)
            for ric in asset_order
        }

        portfolio_div_daily = float(sum(div_cash_by_ric.values()))
        portfolio_cum_dividends += portfolio_div_daily

        portfolio_records["Date"].append(date)
        portfolio_records["Portfolio_Market_Value"].append(sum(market_by_ric.values()))
        portfolio_records["Portfolio_Cash_Balance"].append(portfolio_cash)
        portfolio_records["Portfolio_Total_Value"].append(portfolio_total)
        portfolio_records["Portfolio_Dividend_Cash_Daily"].append(portfolio_div_daily)
        portfolio_records["Portfolio_Dividend_Income"].append(portfolio_cum_dividends)
        portfolio_records["Portfolio_Transaction_Cost_Daily"].append(float(transaction_cost_daily))
        portfolio_records["Portfolio_Commission_Cost_Daily"].append(float(commission_cost_daily))
        portfolio_records["Portfolio_Slippage_Cost_Daily"].append(float(slippage_cost_daily))
        portfolio_records["Portfolio_Spread_Cost_Daily"].append(float(spread_cost_daily))
        portfolio_records["Portfolio_Transaction_Cost_Cumulative"].append(float(portfolio_cum_transaction_costs))
        portfolio_records["Portfolio_Total_Value_Gross"].append(float(portfolio_total_gross))
        portfolio_records["Rebalance_Flag"].append(rebalance_flag)

        for ric in asset_order:
            rec = asset_records[ric]
            rec["Date"].append(date)
            rec["CLOSE"].append(prices_by_ric[ric])
            rec["Split_Multiplier"].append(split_by_ric[ric])
            rec["Dividend"].append(div_ps_by_ric[ric])
            rec["Shares_Held"].append(shares[ric])
            rec["Dividend_Cash_Daily"].append(div_cash_by_ric[ric])
            rec["Dividend_Income"].append(cumulative_dividends[ric])
            rec["Cash_Balance"].append(0.0)
            rec["Market_Value"].append(market_by_ric[ric])
            rec["Total_Value"].append(market_by_ric[ric])
            rec["Weight_EOD"].append(weights_after[ric])
            rec["Target_Weight"].append(target_weights[ric])
            rec["Rebalance_Trade_Shares"].append(trade_shares_by_ric[ric])
            rec["Rebalance_Trade_Value"].append(trade_value_by_ric[ric])
            rec["Rebalance_Trade_Execution_Price"].append(trade_exec_price_by_ric[ric])
            rec["Rebalance_Trade_Total_Cost"].append(trade_total_cost_by_ric[ric])

            if rebalance_flag and abs(trade_value_by_ric[ric]) > 0:
                is_initial_trade = bool(initial_established and date == first_date)
                rebalance_rows.append(
                    {
                        "Date": date,
                        "RIC": ric,
                        "Price": prices_by_ric[ric],
                        "Trade_Shares": trade_shares_by_ric[ric],
                        "Trade_Value": trade_value_by_ric[ric],
                        "Reference_Price": trade_reference_price_by_ric[ric],
                        "Execution_Price": trade_exec_price_by_ric[ric],
                        "Gross_Notional": trade_gross_notional_by_ric[ric],
                        "Spread_Bps_Used": trade_spread_bps_by_ric[ric],
                        "Slippage_Bps_Used": trade_slippage_bps_by_ric[ric],
                        "Commission_Cost": trade_commission_cost_by_ric[ric],
                        "Slippage_Cost": trade_slippage_cost_by_ric[ric],
                        "Spread_Cost": trade_spread_cost_by_ric[ric],
                        "Total_Transaction_Cost": trade_total_cost_by_ric[ric],
                        "Net_Cash_Flow": trade_net_cash_flow_by_ric[ric],
                        "Weight_Before": (0.0 if is_initial_trade else weights_before[ric]),
                        "Weight_After": weights_after[ric],
                        "Target_Weight": target_weights[ric],
                        "Drift_Before": (target_weights[ric] if is_initial_trade else drifts[ric]),
                        "Portfolio_Total_Before": (
                            float(config.initial_capital) if is_initial_trade else portfolio_total_before
                        ),
                        "Portfolio_Total_After": portfolio_total,
                        "Cash_Before": (float(config.initial_capital) if is_initial_trade else portfolio_cash_before),
                        "Cash_After": portfolio_cash,
                        "Reason": ("initial_establishment" if is_initial_trade else "scheduled_rebalance"),
                    }
                )

    portfolio_df = pd.DataFrame(portfolio_records).set_index("Date")
    portfolio_df["Portfolio_Daily_Return"] = (
        portfolio_df["Portfolio_Total_Value"].pct_change().fillna(0.0)
    )
    portfolio_df["Portfolio_Cumulative_Return"] = (
        portfolio_df["Portfolio_Total_Value"] / float(config.initial_capital) - 1.0
    )

    asset_results: dict[str, pd.DataFrame] = {}
    for ric in asset_order:
        df = pd.DataFrame(asset_records[ric]).set_index("Date")
        asset_results[ric] = df

    rebalance_log = pd.DataFrame(rebalance_rows)

    return SimulationResult(
        portfolio_df=portfolio_df,
        asset_results=asset_results,
        effective_start=effective_start,
        rebalance_log=rebalance_log,
    )
