from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ..models import PortfolioConfig, SimulationResult
from .rebalancer import apply_rebalance, build_rebalance_dates, compute_drifts, should_rebalance
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

    shares: dict[str, float] = {}
    cumulative_dividends: dict[str, float] = {ric: 0.0 for ric in asset_order}

    for ric in asset_order:
        first_price = float(prepared[ric].iloc[0]["CLOSE"])
        allocation = config.initial_capital * target_weights[ric]
        shares[ric] = allocation / first_price

    portfolio_cash = 0.0
    portfolio_cum_dividends = 0.0

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
        split_by_ric: dict[str, float] = {}
        div_ps_by_ric: dict[str, float] = {}
        div_cash_by_ric: dict[str, float] = {}

        for ric in asset_order:
            row = prepared[ric].loc[date]
            price = float(row["CLOSE"])
            split_mult = float(row["Split_Multiplier"])
            div_ps = float(row["Dividend"])

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

        if config.rebalancing.enabled and should_rebalance(
            date=date,
            rebalance_dates=rebalance_dates,
            drift_by_ric=drifts,
            drift_tolerance=config.rebalancing.drift_tolerance,
        ):
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

            if rebalance_flag and abs(trade_value_by_ric[ric]) > 0:
                rebalance_rows.append(
                    {
                        "Date": date,
                        "RIC": ric,
                        "Price": prices_by_ric[ric],
                        "Trade_Shares": trade_shares_by_ric[ric],
                        "Trade_Value": trade_value_by_ric[ric],
                        "Weight_Before": weights_before[ric],
                        "Weight_After": weights_after[ric],
                        "Target_Weight": target_weights[ric],
                        "Drift_Before": drifts[ric],
                        "Portfolio_Total_Before": portfolio_total_before,
                        "Portfolio_Total_After": portfolio_total,
                        "Cash_Before": portfolio_cash_before,
                        "Cash_After": portfolio_cash,
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
