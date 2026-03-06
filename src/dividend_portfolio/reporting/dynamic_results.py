from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Matplotlib needs a writable config/cache dir in some environments (e.g. locked HOME).
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = (Path.cwd() / "data" / ".mplconfig").resolve()
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from ..data.refinitiv_client import RefinitivClient
from ..strategy.provider import RefinitivStrategyDataProvider


def _safe_float(value: Any) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(out):
        return float("nan")
    return float(out)


def _date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
        out = out.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
    return out


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    return obj


def resolve_run_id(db_path: str | Path, run_id: str | None = None) -> str:
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"SQLite file not found: {db}")

    conn = sqlite3.connect(db, timeout=30)
    try:
        if run_id:
            row = conn.execute(
                "SELECT run_id FROM run_metadata WHERE run_id = ? LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"run_id '{run_id}' not found in {db}")
            return str(row[0])

        row = conn.execute(
            """
            SELECT run_id
            FROM run_metadata
            ORDER BY datetime(created_at_utc) DESC, run_id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise ValueError(f"No runs found in {db}")
        return str(row[0])
    finally:
        conn.close()


def _load_run_data(db_path: str | Path, run_id: str) -> dict[str, Any]:
    db = Path(db_path)
    conn = sqlite3.connect(db, timeout=30)
    try:
        meta = pd.read_sql_query(
            "SELECT * FROM run_metadata WHERE run_id = ?",
            conn,
            params=[run_id],
        )
        if meta.empty:
            raise ValueError(f"run_id '{run_id}' not found in {db}")

        out = {
            "metadata": meta.iloc[0].to_dict(),
            "portfolio_daily": pd.read_sql_query(
                "SELECT * FROM portfolio_daily WHERE run_id = ? ORDER BY date",
                conn,
                params=[run_id],
            ),
            "holdings_daily": pd.read_sql_query(
                "SELECT * FROM holdings_daily WHERE run_id = ? ORDER BY date, ric",
                conn,
                params=[run_id],
            ),
            "trades": pd.read_sql_query(
                "SELECT * FROM trades WHERE run_id = ? ORDER BY date, ric",
                conn,
                params=[run_id],
            ),
            "target_weights": pd.read_sql_query(
                "SELECT * FROM target_weights WHERE run_id = ? ORDER BY quarter, rank_in_portfolio",
                conn,
                params=[run_id],
            ),
            "quarter_scores": pd.read_sql_query(
                "SELECT * FROM quarter_scores WHERE run_id = ? ORDER BY quarter, rank_score",
                conn,
                params=[run_id],
            ),
            "candidate_universe": pd.read_sql_query(
                "SELECT * FROM candidate_universe WHERE run_id = ? ORDER BY quarter, rank_market_cap",
                conn,
                params=[run_id],
            ),
        }
        return out
    finally:
        conn.close()


def _parse_config_blob(metadata_row: dict[str, Any]) -> dict[str, Any]:
    blob = metadata_row.get("config_json")
    if isinstance(blob, dict):
        return blob
    if not isinstance(blob, str):
        return {}
    try:
        parsed = json.loads(blob)
    except Exception:  # noqa: BLE001
        return {}
    return parsed if isinstance(parsed, dict) else {}


def compute_holding_presence_stats(
    holdings_daily: pd.DataFrame,
    portfolio_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    if holdings_daily.empty or len(portfolio_dates) == 0:
        return pd.DataFrame(
            columns=[
                "ric",
                "days_held",
                "longest_consecutive_days",
                "first_held_date",
                "last_held_date",
            ]
        )

    held = _date_col(holdings_daily, "date")
    held["shares"] = pd.to_numeric(held.get("shares"), errors="coerce").fillna(0.0)
    held = held.loc[held["shares"] > 1e-12]
    if held.empty:
        return pd.DataFrame(
            columns=[
                "ric",
                "days_held",
                "longest_consecutive_days",
                "first_held_date",
                "last_held_date",
            ]
        )

    date_pos = {d: i for i, d in enumerate(portfolio_dates)}
    rows: list[dict[str, Any]] = []

    for ric, grp in held.groupby("ric"):
        idxs = sorted({date_pos[d] for d in grp["date"] if d in date_pos})
        if not idxs:
            continue
        longest = 0
        cur = 0
        prev = -10
        for idx in idxs:
            if idx == prev + 1:
                cur += 1
            else:
                cur = 1
            longest = max(longest, cur)
            prev = idx

        rows.append(
            {
                "ric": str(ric),
                "days_held": int(len(idxs)),
                "longest_consecutive_days": int(longest),
                "first_held_date": portfolio_dates[min(idxs)].date().isoformat(),
                "last_held_date": portfolio_dates[max(idxs)].date().isoformat(),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["longest_consecutive_days", "days_held", "ric"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _prepare_benchmark_series(benchmark_close: pd.DataFrame, portfolio_index: pd.DatetimeIndex) -> pd.Series | None:
    if benchmark_close is None or benchmark_close.empty or len(portfolio_index) == 0:
        return None

    bench = benchmark_close.copy()
    if "Date" not in bench.columns or "CLOSE" not in bench.columns:
        return None
    bench["Date"] = pd.to_datetime(bench["Date"], errors="coerce")
    bench["CLOSE"] = pd.to_numeric(bench["CLOSE"], errors="coerce")
    bench = bench.dropna(subset=["Date", "CLOSE"])
    bench = bench.loc[bench["CLOSE"] > 0]
    if bench.empty:
        return None

    close = bench.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    series = close.set_index("Date")["CLOSE"]
    aligned = series.reindex(portfolio_index).ffill().bfill()
    if aligned.isna().all():
        return None
    return aligned


def compute_summary_from_data(
    *,
    run_id: str,
    run_metadata: dict[str, Any],
    portfolio_daily: pd.DataFrame,
    holdings_daily: pd.DataFrame,
    trades: pd.DataFrame,
    target_weights: pd.DataFrame,
    quarter_scores: pd.DataFrame,
    candidate_universe: pd.DataFrame | None = None,
    benchmark_close: pd.DataFrame | None = None,
    benchmark_label: str = "S&P 500 (.SPX)",
    warning_messages: list[str] | None = None,
) -> dict[str, Any]:
    warnings_out = warning_messages or []
    if portfolio_daily.empty:
        raise ValueError(f"portfolio_daily is empty for run {run_id}")

    portfolio = _date_col(portfolio_daily, "date")
    total = pd.to_numeric(portfolio["portfolio_total_value"], errors="coerce")
    market = pd.to_numeric(portfolio["portfolio_market_value"], errors="coerce")
    cash = pd.to_numeric(portfolio["portfolio_cash_balance"], errors="coerce")
    div_daily = pd.to_numeric(portfolio["portfolio_dividend_cash_daily"], errors="coerce").fillna(0.0)
    rebalance = pd.to_numeric(portfolio["rebalance_flag"], errors="coerce").fillna(0).astype(int)
    portfolio = portfolio.assign(
        portfolio_total_value=total,
        portfolio_market_value=market,
        portfolio_cash_balance=cash,
        portfolio_dividend_cash_daily=div_daily,
        rebalance_flag=rebalance,
    )
    portfolio = portfolio.dropna(subset=["portfolio_total_value", "date"])
    portfolio = portfolio.set_index("date").sort_index()
    if portfolio.empty:
        raise ValueError(f"No valid portfolio_total_value data for run {run_id}")

    run_config = _parse_config_blob(run_metadata)
    portfolio_cfg = run_config.get("portfolio", {}) if isinstance(run_config, dict) else {}
    risk_free_rate = _safe_float(portfolio_cfg.get("risk_free_rate", 0.0))
    if pd.isna(risk_free_rate):
        risk_free_rate = 0.0
    initial_capital_cfg = _safe_float(portfolio_cfg.get("initial_capital", float("nan")))
    config_reinvest = bool(portfolio_cfg.get("reinvest_dividends", False))

    total_value = portfolio["portfolio_total_value"].astype(float)
    returns = total_value.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    running_max = total_value.cummax()
    drawdown = total_value / running_max - 1.0

    start_value = float(total_value.iloc[0])
    end_value = float(total_value.iloc[-1])
    initial_capital = float(initial_capital_cfg) if not pd.isna(initial_capital_cfg) else start_value
    total_return = end_value / start_value - 1.0 if start_value > 0 else float("nan")

    period_days = max((portfolio.index[-1] - portfolio.index[0]).days, 1)
    if start_value > 0:
        cagr = (end_value / start_value) ** (365.25 / period_days) - 1.0
    else:
        cagr = float("nan")

    mean_daily = float(returns.mean())
    std_daily = float(returns.std(ddof=0))
    rf_daily = risk_free_rate / 252.0
    annualized_vol = std_daily * math.sqrt(252.0)
    sharpe = ((mean_daily - rf_daily) / std_daily) * math.sqrt(252.0) if std_daily > 0 else float("nan")

    downside = returns.loc[returns < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 0 else float("nan")
    sortino = (
        ((mean_daily - rf_daily) / downside_std) * math.sqrt(252.0)
        if downside_std and not math.isnan(downside_std) and downside_std > 0
        else float("nan")
    )

    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else float("nan")

    max_dd_end = drawdown.idxmin()
    max_dd_start = total_value.loc[:max_dd_end].idxmax()
    peak_val = float(total_value.loc[max_dd_start])
    recovery_dates = total_value.loc[max_dd_end:]
    recovered = recovery_dates.loc[recovery_dates >= peak_val]
    max_dd_recovery_date = recovered.index[0] if not recovered.empty else pd.NaT
    max_dd_duration_days = (
        int((max_dd_recovery_date - max_dd_start).days)
        if pd.notna(max_dd_recovery_date)
        else int((portfolio.index[-1] - max_dd_start).days)
    )

    best_day = float(returns.max())
    worst_day = float(returns.min())
    positive_days = int((returns > 0).sum())
    negative_days = int((returns < 0).sum())
    flat_days = int((returns == 0).sum())
    trading_days = int(len(portfolio))

    div_total = float(portfolio["portfolio_dividend_cash_daily"].sum())
    div_days = int((portfolio["portfolio_dividend_cash_daily"] > 0).sum())
    total_gain = end_value - start_value
    div_share_gain = div_total / total_gain if total_gain != 0 else float("nan")
    div_yield_on_start = div_total / start_value if start_value > 0 else float("nan")
    div_yield_on_initial_capital = div_total / initial_capital if initial_capital > 0 else float("nan")

    holdings = _date_col(holdings_daily, "date")
    for col in ["shares", "dividend_cash_daily", "market_value", "close"]:
        if col in holdings.columns:
            holdings[col] = pd.to_numeric(holdings[col], errors="coerce")
    held_only = holdings.loc[holdings.get("shares", pd.Series(dtype=float)).fillna(0.0) > 1e-12].copy()

    hold_stats = compute_holding_presence_stats(held_only, portfolio.index)
    unique_stocks_held = int(held_only["ric"].nunique()) if not held_only.empty else 0
    positions_daily = (
        held_only.groupby("date")["ric"].nunique().reindex(portfolio.index, fill_value=0).astype(int)
        if not held_only.empty
        else pd.Series(0, index=portfolio.index, dtype=int)
    )

    div_by_stock = (
        held_only.groupby("ric", as_index=False)["dividend_cash_daily"].sum().sort_values("dividend_cash_daily", ascending=False)
        if not held_only.empty
        else pd.DataFrame(columns=["ric", "dividend_cash_daily"])
    )
    highest_dividend_stock = (
        {
            "ric": str(div_by_stock.iloc[0]["ric"]),
            "total_dividend_cash": float(div_by_stock.iloc[0]["dividend_cash_daily"]),
        }
        if not div_by_stock.empty
        else None
    )

    trades_df = _date_col(trades, "date")
    for col in ["trade_shares", "trade_value"]:
        if col in trades_df.columns:
            trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce").fillna(0.0)
    trade_count = int(len(trades_df))
    buy_count = int((trades_df.get("trade_shares", pd.Series(dtype=float)) > 0).sum())
    sell_count = int((trades_df.get("trade_shares", pd.Series(dtype=float)) < 0).sum())
    gross_turnover = (
        float(trades_df["trade_value"].abs().sum()) if "trade_value" in trades_df.columns else 0.0
    )
    net_trade_value = float(trades_df["trade_value"].sum()) if "trade_value" in trades_df.columns else 0.0
    trade_days = int(trades_df["date"].nunique()) if "date" in trades_df.columns and not trades_df.empty else 0

    trades_per_quarter = (
        trades_df.groupby("quarter", as_index=False).size().rename(columns={"size": "trade_count"})
        if ("quarter" in trades_df.columns and not trades_df.empty)
        else pd.DataFrame(columns=["quarter", "trade_count"])
    )
    turnover_per_quarter = (
        trades_df.groupby("quarter", as_index=False)["trade_value"].apply(lambda s: s.abs().sum()).rename(columns={"trade_value": "gross_turnover"})
        if ("quarter" in trades_df.columns and "trade_value" in trades_df.columns and not trades_df.empty)
        else pd.DataFrame(columns=["quarter", "gross_turnover"])
    )

    quarterly_dividends = portfolio.groupby("quarter", as_index=False)["portfolio_dividend_cash_daily"].sum()
    quarterly_dividends = quarterly_dividends.rename(columns={"portfolio_dividend_cash_daily": "quarter_dividend_cash"})
    quarterly_performance: list[dict[str, Any]] = []
    for quarter, qdf in portfolio.groupby("quarter"):
        q_start = float(qdf["portfolio_total_value"].iloc[0])
        q_end = float(qdf["portfolio_total_value"].iloc[-1])
        q_ret = q_end / q_start - 1.0 if q_start > 0 else float("nan")
        q_div = float(qdf["portfolio_dividend_cash_daily"].sum())
        quarterly_performance.append(
            {
                "quarter": str(quarter),
                "quarter_start_value": q_start,
                "quarter_end_value": q_end,
                "quarter_total_return": q_ret,
                "quarter_dividend_cash": q_div,
            }
        )

    benchmark_section: dict[str, Any] = {"enabled": False, "label": benchmark_label}
    benchmark_aligned: pd.Series | None = None
    if benchmark_close is not None and not benchmark_close.empty:
        benchmark_aligned = _prepare_benchmark_series(benchmark_close, portfolio.index)

    if benchmark_aligned is None:
        benchmark_section["enabled"] = False
        if benchmark_close is not None and not benchmark_close.empty:
            warnings_out.append("Benchmark data available but could not be aligned to portfolio dates.")
    else:
        bench_returns = benchmark_aligned.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bench_total_return = float(benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0] - 1.0)
        excess = returns - bench_returns
        tracking_error_daily = float(excess.std(ddof=0))
        tracking_error = tracking_error_daily * math.sqrt(252.0)
        information_ratio = (
            (float(excess.mean()) / tracking_error_daily) * math.sqrt(252.0)
            if tracking_error_daily > 0
            else float("nan")
        )
        correlation = float(returns.corr(bench_returns))
        beta = (
            float(np.cov(returns, bench_returns, ddof=0)[0, 1] / np.var(bench_returns))
            if float(np.var(bench_returns)) > 0
            else float("nan")
        )

        benchmark_section = {
            "enabled": True,
            "label": benchmark_label,
            "total_return": bench_total_return,
            "alpha_total_return": float(total_return - bench_total_return),
            "tracking_error_annualized": float(tracking_error),
            "information_ratio": float(information_ratio),
            "correlation_with_portfolio_daily_returns": correlation,
            "beta_to_benchmark": beta,
        }

    top_longest = hold_stats.head(10).to_dict(orient="records") if not hold_stats.empty else []
    top_dividend = (
        div_by_stock.head(10).rename(columns={"dividend_cash_daily": "total_dividend_cash"}).to_dict(orient="records")
        if not div_by_stock.empty
        else []
    )

    score_df = quarter_scores.copy()
    if not score_df.empty and "score" in score_df.columns:
        score_df["score"] = pd.to_numeric(score_df["score"], errors="coerce")
        score_df = score_df.sort_values("score", ascending=False)
    top_score_stock = (
        {
            "ric": str(score_df.iloc[0]["ric"]),
            "score": float(score_df.iloc[0]["score"]),
            "quarter": str(score_df.iloc[0]["quarter"]),
        }
        if not score_df.empty
        else None
    )

    result = {
        "run_id": run_id,
        "created_at_utc": run_metadata.get("created_at_utc"),
        "start_date": run_metadata.get("start_date"),
        "end_date": run_metadata.get("end_date"),
        "risk_free_rate": float(risk_free_rate),
        "portfolio_metrics": {
            "start_value": start_value,
            "end_value": end_value,
            "initial_capital_from_config": initial_capital,
            "net_total_gain": float(end_value - start_value),
            "total_return": float(total_return),
            "cagr": float(cagr),
            "annualized_volatility": float(annualized_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "max_drawdown_start_date": max_dd_start.date().isoformat(),
            "max_drawdown_end_date": max_dd_end.date().isoformat(),
            "max_drawdown_recovery_date": (
                max_dd_recovery_date.date().isoformat() if pd.notna(max_dd_recovery_date) else None
            ),
            "max_drawdown_duration_days": max_dd_duration_days,
            "best_day_return": float(best_day),
            "worst_day_return": float(worst_day),
            "trading_days": trading_days,
            "positive_days": positive_days,
            "negative_days": negative_days,
            "flat_days": flat_days,
            "quarterly_performance": quarterly_performance,
        },
        "dividends": {
            "total_dividend_cash": float(div_total),
            "dividend_days": div_days,
            "dividend_share_of_total_gain": float(div_share_gain),
            "dividend_yield_on_start_value": float(div_yield_on_start),
            "dividend_yield_on_initial_capital": float(div_yield_on_initial_capital),
            "highest_dividend_paying_stock": highest_dividend_stock,
            "highest_quarterly_yield_score_stock": top_score_stock,
            "top_dividend_contributors": top_dividend,
            "quarterly_dividend_cash": quarterly_dividends.to_dict(orient="records"),
        },
        "trading_activity": {
            "number_of_trades": trade_count,
            "buy_trades": buy_count,
            "sell_trades": sell_count,
            "trade_days": trade_days,
            "rebalance_days": int(portfolio["rebalance_flag"].sum()),
            "gross_turnover": float(gross_turnover),
            "net_trade_value": float(net_trade_value),
            "trades_per_quarter": trades_per_quarter.to_dict(orient="records"),
            "turnover_per_quarter": turnover_per_quarter.to_dict(orient="records"),
        },
        "holdings": {
            "unique_stocks_held": unique_stocks_held,
            "average_positions_per_day": float(positions_daily.mean()) if len(positions_daily) else 0.0,
            "min_positions_per_day": int(positions_daily.min()) if len(positions_daily) else 0,
            "max_positions_per_day": int(positions_daily.max()) if len(positions_daily) else 0,
            "longest_staying_stock": (top_longest[0] if top_longest else None),
            "top_longest_held_stocks": top_longest,
        },
        "universe": {
            "candidate_universe_rows": int(len(candidate_universe)) if candidate_universe is not None else None,
            "target_weight_rows": int(len(target_weights)),
            "quarter_score_rows": int(len(quarter_scores)),
        },
        "benchmark_comparison": benchmark_section,
        "dividend_policy": {
            "config_reinvest_dividends": config_reinvest,
            "effective_behavior": (
                "Dividends are credited to cash on pay dates and deployed into holdings at quarterly rebalances."
            ),
        },
        "warnings": warnings_out,
    }
    return _to_jsonable(result)


def _fetch_benchmark_close(
    *,
    ric: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    with RefinitivClient() as client:
        provider = RefinitivStrategyDataProvider(client, batch_size=1, enable_cache=True)
        return provider.get_close_history([ric], start_date, end_date)


def _plot_and_save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _create_plots(
    *,
    out_dir: Path,
    portfolio_daily: pd.DataFrame,
    holdings_daily: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark_close: pd.DataFrame | None,
    benchmark_label: str,
) -> list[str]:
    created: list[str] = []

    portfolio = _date_col(portfolio_daily, "date")
    portfolio["portfolio_total_value"] = pd.to_numeric(portfolio["portfolio_total_value"], errors="coerce")
    portfolio["portfolio_market_value"] = pd.to_numeric(portfolio["portfolio_market_value"], errors="coerce")
    portfolio["portfolio_cash_balance"] = pd.to_numeric(portfolio["portfolio_cash_balance"], errors="coerce")
    portfolio["portfolio_dividend_cash_daily"] = pd.to_numeric(portfolio["portfolio_dividend_cash_daily"], errors="coerce").fillna(0.0)
    portfolio = portfolio.dropna(subset=["date", "portfolio_total_value"]).set_index("date").sort_index()
    if portfolio.empty:
        return created

    returns = portfolio["portfolio_total_value"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    running_max = portfolio["portfolio_total_value"].cummax()
    drawdown = portfolio["portfolio_total_value"] / running_max - 1.0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(portfolio.index, portfolio["portfolio_total_value"], label="Portfolio Total Value", linewidth=1.8)
    ax.plot(portfolio.index, portfolio["portfolio_market_value"], label="Invested Market Value", alpha=0.85)
    ax.plot(portfolio.index, portfolio["portfolio_cash_balance"], label="Cash Balance", alpha=0.85)
    ax.set_title("Portfolio Value Decomposition")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.3)
    ax.legend()
    file_name = "portfolio_value_cash_market.png"
    _plot_and_save(out_dir / file_name)
    created.append(file_name)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(drawdown.index, drawdown, color="tab:red", label="Drawdown")
    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    file_name = "drawdown.png"
    _plot_and_save(out_dir / file_name)
    created.append(file_name)

    rolling_window = 63
    rolling_mean = returns.rolling(rolling_window).mean()
    rolling_std = returns.rolling(rolling_window).std(ddof=0)
    rolling_sharpe = (rolling_mean / rolling_std) * math.sqrt(252.0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe, color="tab:blue", label=f"Rolling Sharpe ({rolling_window}d)")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title("Rolling Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    ax.legend()
    file_name = "rolling_sharpe_63d.png"
    _plot_and_save(out_dir / file_name)
    created.append(file_name)

    rolling_vol = returns.rolling(rolling_window).std(ddof=0) * math.sqrt(252.0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_vol.index, rolling_vol, color="tab:orange", label=f"Rolling Volatility ({rolling_window}d)")
    ax.set_title("Rolling Annualized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True, alpha=0.3)
    ax.legend()
    file_name = "rolling_volatility_63d.png"
    _plot_and_save(out_dir / file_name)
    created.append(file_name)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(returns, bins=60, color="tab:purple", alpha=0.75)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.2)
    file_name = "daily_return_distribution.png"
    _plot_and_save(out_dir / file_name)
    created.append(file_name)

    cum_port = portfolio["portfolio_total_value"] / float(portfolio["portfolio_total_value"].iloc[0]) - 1.0
    benchmark = _prepare_benchmark_series(benchmark_close if benchmark_close is not None else pd.DataFrame(), portfolio.index)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cum_port.index, cum_port, label="Portfolio", linewidth=1.8)
    if benchmark is not None:
        cum_bench = benchmark / float(benchmark.iloc[0]) - 1.0
        ax.plot(cum_bench.index, cum_bench, label=benchmark_label, linewidth=1.6)
    ax.set_title("Cumulative Return Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    file_name = "cumulative_return_vs_sp500.png"
    _plot_and_save(out_dir / file_name)
    created.append(file_name)

    q_div = portfolio.groupby("quarter")["portfolio_dividend_cash_daily"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    q_div.plot(kind="bar", ax=ax, color="tab:green", alpha=0.8)
    ax.set_title("Quarterly Dividend Cash")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Dividend Cash")
    ax.grid(True, axis="y", alpha=0.25)
    file_name = "quarterly_dividend_cash.png"
    _plot_and_save(out_dir / file_name)
    created.append(file_name)

    holdings = _date_col(holdings_daily, "date")
    if not holdings.empty:
        holdings["shares"] = pd.to_numeric(holdings.get("shares"), errors="coerce").fillna(0.0)
        holdings["dividend_cash_daily"] = pd.to_numeric(holdings.get("dividend_cash_daily"), errors="coerce").fillna(0.0)
        held = holdings.loc[holdings["shares"] > 1e-12]

        if not held.empty:
            top_div = (
                held.groupby("ric", as_index=False)["dividend_cash_daily"]
                .sum()
                .sort_values("dividend_cash_daily", ascending=False)
                .head(10)
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(top_div["ric"], top_div["dividend_cash_daily"], color="tab:green", alpha=0.85)
            ax.set_title("Top 10 Dividend Cash Contributors")
            ax.set_xlabel("RIC")
            ax.set_ylabel("Total Dividend Cash")
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=45)
            file_name = "top_dividend_stocks.png"
            _plot_and_save(out_dir / file_name)
            created.append(file_name)

            hold_stats = compute_holding_presence_stats(held, portfolio.index).head(10)
            if not hold_stats.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(hold_stats["ric"], hold_stats["longest_consecutive_days"], color="tab:cyan", alpha=0.85)
                ax.set_title("Top 10 Longest Consecutive Holdings")
                ax.set_xlabel("RIC")
                ax.set_ylabel("Longest Consecutive Days")
                ax.grid(True, axis="y", alpha=0.25)
                ax.tick_params(axis="x", rotation=45)
                file_name = "longest_held_stocks.png"
                _plot_and_save(out_dir / file_name)
                created.append(file_name)

            positions_daily = held.groupby("date")["ric"].nunique().reindex(portfolio.index, fill_value=0)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(positions_daily.index, positions_daily.values, color="tab:brown")
            ax.set_title("Number of Active Holdings Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            file_name = "positions_count_over_time.png"
            _plot_and_save(out_dir / file_name)
            created.append(file_name)

    trades_df = _date_col(trades, "date")
    if not trades_df.empty and "quarter" in trades_df.columns:
        trades_df["trade_value"] = pd.to_numeric(trades_df.get("trade_value"), errors="coerce").fillna(0.0)
        trades_per_q = trades_df.groupby("quarter").size()
        fig, ax = plt.subplots(figsize=(12, 4))
        trades_per_q.plot(kind="bar", ax=ax, color="tab:blue", alpha=0.85)
        ax.set_title("Trade Count per Quarter")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Trades")
        ax.grid(True, axis="y", alpha=0.25)
        file_name = "trades_per_quarter.png"
        _plot_and_save(out_dir / file_name)
        created.append(file_name)

        turnover_q = trades_df.groupby("quarter")["trade_value"].apply(lambda s: s.abs().sum())
        fig, ax = plt.subplots(figsize=(12, 4))
        turnover_q.plot(kind="bar", ax=ax, color="tab:orange", alpha=0.85)
        ax.set_title("Gross Turnover per Quarter")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Gross Turnover")
        ax.grid(True, axis="y", alpha=0.25)
        file_name = "turnover_per_quarter.png"
        _plot_and_save(out_dir / file_name)
        created.append(file_name)

    return created


def generate_dynamic_showresults(
    *,
    db_path: str | Path,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
    include_benchmark: bool = True,
    benchmark_ric: str = ".SPX",
    benchmark_label: str = "S&P 500 (.SPX)",
) -> tuple[str, Path]:
    resolved_run_id = resolve_run_id(db_path, run_id=run_id)
    data = _load_run_data(db_path, resolved_run_id)
    metadata = data["metadata"]

    out_dir = Path(output_dir) if output_dir is not None else Path("data/runs_dynamic") / resolved_run_id / "showresults"
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df = pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
    warnings_out: list[str] = []
    if include_benchmark:
        try:
            benchmark_df = _fetch_benchmark_close(
                ric=benchmark_ric,
                start_date=str(metadata.get("start_date")),
                end_date=str(metadata.get("end_date")),
            )
        except Exception as exc:  # noqa: BLE001
            warnings_out.append(f"Benchmark fetch failed for {benchmark_ric}: {exc}")

    summary = compute_summary_from_data(
        run_id=resolved_run_id,
        run_metadata=metadata,
        portfolio_daily=data["portfolio_daily"],
        holdings_daily=data["holdings_daily"],
        trades=data["trades"],
        target_weights=data["target_weights"],
        quarter_scores=data["quarter_scores"],
        candidate_universe=data["candidate_universe"],
        benchmark_close=benchmark_df,
        benchmark_label=benchmark_label,
        warning_messages=warnings_out,
    )

    plots = _create_plots(
        out_dir=out_dir,
        portfolio_daily=data["portfolio_daily"],
        holdings_daily=data["holdings_daily"],
        trades=data["trades"],
        benchmark_close=benchmark_df if include_benchmark else None,
        benchmark_label=benchmark_label,
    )
    summary["generated_plots"] = plots

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, indent=2, sort_keys=True)

    return resolved_run_id, out_dir
