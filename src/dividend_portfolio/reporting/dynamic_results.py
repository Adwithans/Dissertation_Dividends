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
from matplotlib import ticker as mticker

from ..analytics.volatility_models import fit_arch_garch_models
from ..data.refinitiv_client import RefinitivClient


SP500_RIC_DEFAULT = ".SPX"
SP500_LABEL_DEFAULT = "S&P 500 (.SPX)"
RUSSELL_1000_RIC_DEFAULT = ".RUI"
RUSSELL_1000_LABEL_DEFAULT = "Russell 1000 (.RUI)"
BENCHMARK_CACHE_DB_DEFAULT = Path("data") / "store" / "benchmark_cache.sqlite"


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


def _normalized_allocation_strategy(strategy_cfg: dict[str, Any]) -> str:
    raw = (
        strategy_cfg.get("allocation_strategy")
        or strategy_cfg.get("quarterly_weighting")
        or "yield_proportional"
    )
    normalized = str(raw).strip().lower()
    if normalized == "normalized_yield_score":
        return "yield_proportional"
    return normalized


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


def _compute_benchmark_section(
    *,
    benchmark_close: pd.DataFrame | None,
    benchmark_label: str,
    portfolio_returns: pd.Series,
    portfolio_total_return: float,
    portfolio_index: pd.DatetimeIndex,
    warning_messages: list[str],
    warning_prefix: str = "Benchmark",
) -> dict[str, Any]:
    section: dict[str, Any] = {"enabled": False, "label": benchmark_label}
    if benchmark_close is None or benchmark_close.empty:
        return section

    benchmark_aligned = _prepare_benchmark_series(benchmark_close, portfolio_index)
    if benchmark_aligned is None:
        warning_messages.append(f"{warning_prefix} data available but could not be aligned to portfolio dates.")
        return section

    bench_returns = benchmark_aligned.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bench_total_return = float(benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0] - 1.0)
    benchmark_days = max((portfolio_index[-1] - portfolio_index[0]).days, 1) if len(portfolio_index) else 1
    benchmark_cagr = (
        (float(benchmark_aligned.iloc[-1]) / float(benchmark_aligned.iloc[0])) ** (365.25 / benchmark_days) - 1.0
        if len(benchmark_aligned) > 0 and float(benchmark_aligned.iloc[0]) > 0
        else float("nan")
    )
    aligned_returns = pd.concat(
        [
            pd.to_numeric(portfolio_returns, errors="coerce").rename("portfolio_return"),
            pd.to_numeric(bench_returns, errors="coerce").rename("benchmark_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned_returns.empty:
        warning_messages.append(f"{warning_prefix} returns could not be aligned to portfolio returns.")
        return section

    port_arr = aligned_returns["portfolio_return"].to_numpy(dtype=float)
    bench_arr = aligned_returns["benchmark_return"].to_numpy(dtype=float)
    excess = port_arr - bench_arr
    tracking_error_daily = float(np.std(excess, ddof=0))
    tracking_error = tracking_error_daily * math.sqrt(252.0)
    annualized_excess_return = float(np.mean(excess) * 252.0) if len(excess) > 0 else float("nan")
    information_ratio = (
        (float(np.mean(excess)) / tracking_error_daily) * math.sqrt(252.0)
        if tracking_error_daily > 0
        else float("nan")
    )
    correlation = (
        float(np.corrcoef(port_arr, bench_arr)[0, 1])
        if len(port_arr) >= 2 and float(np.std(port_arr, ddof=0)) > 0 and float(np.std(bench_arr, ddof=0)) > 0
        else float("nan")
    )
    beta = (
        float(np.cov(port_arr, bench_arr, ddof=0)[0, 1] / np.var(bench_arr))
        if float(np.var(bench_arr)) > 0
        else float("nan")
    )

    return {
        "enabled": True,
        "label": benchmark_label,
        "total_return": bench_total_return,
        "benchmark_cagr": float(benchmark_cagr),
        "alpha_total_return": float(portfolio_total_return - bench_total_return),
        "annualized_excess_return": float(annualized_excess_return),
        "tracking_error_annualized": float(tracking_error),
        "information_ratio": float(information_ratio),
        "correlation_with_portfolio_daily_returns": correlation,
        "beta_to_benchmark": beta,
    }


def _period_total_return(series: pd.Series, freq: str) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    s = series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[s.index.notna()]
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float)
    out = s.groupby(s.index.to_period(freq)).apply(
        lambda x: float(x.iloc[-1] / x.iloc[0] - 1.0) if len(x) > 0 and float(x.iloc[0]) > 0 else float("nan")
    )
    out.index = out.index.astype(str)
    return out


def _pair_period_return_series(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    paired = pd.concat(
        [
            pd.to_numeric(strategy_returns, errors="coerce").rename("strategy_return"),
            pd.to_numeric(benchmark_returns, errors="coerce").rename("benchmark_return"),
        ],
        axis=1,
    )
    paired = paired.dropna()
    paired.index = pd.Index(paired.index.astype(str), name="period")
    return paired


def _quadrant_period_stats(paired_returns: pd.DataFrame) -> dict[str, Any]:
    if paired_returns is None or paired_returns.empty:
        return {
            "enabled": False,
            "frequency": "quarterly",
            "total_periods": 0,
            "strategy_up_periods": 0,
            "strategy_down_periods": 0,
            "benchmark_up_periods": 0,
            "benchmark_down_periods": 0,
            "both_up_periods": 0,
            "both_down_periods": 0,
            "strategy_up_benchmark_down_periods": 0,
            "strategy_down_benchmark_up_periods": 0,
            "on_axis_periods": 0,
            "same_direction_periods": 0,
            "opposite_direction_periods": 0,
            "strategy_up_when_benchmark_up_rate": None,
            "strategy_down_when_benchmark_down_rate": None,
            "beat_benchmark_periods": 0,
            "beat_benchmark_rate": None,
            "beat_benchmark_when_benchmark_up_rate": None,
            "beat_benchmark_when_benchmark_down_rate": None,
            "correlation": None,
            "average_strategy_return": None,
            "average_benchmark_return": None,
            "average_excess_return": None,
            "upside_capture_ratio": None,
            "downside_capture_ratio": None,
        }

    strategy = pd.to_numeric(paired_returns["strategy_return"], errors="coerce")
    benchmark = pd.to_numeric(paired_returns["benchmark_return"], errors="coerce")

    both_up = (strategy > 0.0) & (benchmark > 0.0)
    both_down = (strategy < 0.0) & (benchmark < 0.0)
    strategy_up_benchmark_down = (strategy > 0.0) & (benchmark < 0.0)
    strategy_down_benchmark_up = (strategy < 0.0) & (benchmark > 0.0)
    on_axis = (strategy == 0.0) | (benchmark == 0.0)
    same_direction = both_up | both_down
    opposite_direction = strategy_up_benchmark_down | strategy_down_benchmark_up
    benchmark_up = benchmark > 0.0
    benchmark_down = benchmark < 0.0
    beat_benchmark = strategy > benchmark

    corr = float(strategy.corr(benchmark)) if len(paired_returns) >= 2 else float("nan")
    benchmark_up_count = int(benchmark_up.sum())
    benchmark_down_count = int(benchmark_down.sum())
    total_periods = int(len(paired_returns))
    beat_count = int(beat_benchmark.sum())
    benchmark_up_mean = float(benchmark.loc[benchmark_up].mean()) if benchmark_up_count > 0 else float("nan")
    benchmark_down_mean = float(benchmark.loc[benchmark_down].mean()) if benchmark_down_count > 0 else float("nan")
    upside_capture = (
        float(strategy.loc[benchmark_up].mean() / benchmark_up_mean)
        if benchmark_up_count > 0 and benchmark_up_mean != 0.0
        else None
    )
    downside_capture = (
        float(strategy.loc[benchmark_down].mean() / benchmark_down_mean)
        if benchmark_down_count > 0 and benchmark_down_mean != 0.0
        else None
    )

    return {
        "enabled": True,
        "frequency": "quarterly",
        "total_periods": total_periods,
        "strategy_up_periods": int((strategy > 0.0).sum()),
        "strategy_down_periods": int((strategy < 0.0).sum()),
        "benchmark_up_periods": benchmark_up_count,
        "benchmark_down_periods": benchmark_down_count,
        "both_up_periods": int(both_up.sum()),
        "both_down_periods": int(both_down.sum()),
        "strategy_up_benchmark_down_periods": int(strategy_up_benchmark_down.sum()),
        "strategy_down_benchmark_up_periods": int(strategy_down_benchmark_up.sum()),
        "on_axis_periods": int(on_axis.sum()),
        "same_direction_periods": int(same_direction.sum()),
        "opposite_direction_periods": int(opposite_direction.sum()),
        "strategy_up_when_benchmark_up_rate": (
            float(both_up.sum() / benchmark_up_count) if benchmark_up_count > 0 else None
        ),
        "strategy_down_when_benchmark_down_rate": (
            float(both_down.sum() / benchmark_down_count) if benchmark_down_count > 0 else None
        ),
        "beat_benchmark_periods": beat_count,
        "beat_benchmark_rate": float(beat_count / total_periods) if total_periods > 0 else None,
        "beat_benchmark_when_benchmark_up_rate": (
            float((beat_benchmark & benchmark_up).sum() / benchmark_up_count) if benchmark_up_count > 0 else None
        ),
        "beat_benchmark_when_benchmark_down_rate": (
            float((beat_benchmark & benchmark_down).sum() / benchmark_down_count) if benchmark_down_count > 0 else None
        ),
        "correlation": corr,
        "average_strategy_return": float(strategy.mean()) if total_periods > 0 else None,
        "average_benchmark_return": float(benchmark.mean()) if total_periods > 0 else None,
        "average_excess_return": float((strategy - benchmark).mean()) if total_periods > 0 else None,
        "upside_capture_ratio": upside_capture,
        "downside_capture_ratio": downside_capture,
    }


def _monthly_up_down_counts(series: pd.Series) -> dict[str, int]:
    s = series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[s.index.notna()]
    monthly_close = (
        pd.to_numeric(s, errors="coerce")
        .dropna()
        .groupby(s.index.to_period("M"))
        .last()
    )
    monthly_ret = monthly_close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    up = int((monthly_ret > 0).sum())
    down = int((monthly_ret < 0).sum())
    flat = int((monthly_ret == 0).sum())
    return {
        "up_months": up,
        "down_months": down,
        "flat_months": flat,
        "total_months": int(len(monthly_ret)),
    }


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
    benchmark_label: str = SP500_LABEL_DEFAULT,
    sp500_close: pd.DataFrame | None = None,
    sp500_label: str = SP500_LABEL_DEFAULT,
    russell_1000_close: pd.DataFrame | None = None,
    russell_1000_label: str = RUSSELL_1000_LABEL_DEFAULT,
    warning_messages: list[str] | None = None,
    volatility_models_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings_out = list(warning_messages or [])
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
    strategy_cfg = run_config.get("strategy", {}) if isinstance(run_config, dict) else {}
    evaluation_context_cfg = (
        run_config.get("evaluation_context", {})
        if isinstance(run_config, dict)
        else {}
    )
    selection_policy_cfg = (
        strategy_cfg.get("selection_policy", {})
        if isinstance(strategy_cfg, dict)
        else {}
    )
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
    returns_skewness = float(returns.skew()) if len(returns) > 0 else float("nan")
    returns_kurtosis = float(returns.kurt()) if len(returns) > 0 else float("nan")

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
    for col in [
        "trade_shares",
        "trade_value",
        "commission_cost",
        "slippage_cost",
        "spread_cost",
        "total_transaction_cost",
        "gross_notional",
    ]:
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
    tx_by_quarter = (
        trades_df.groupby("quarter", as_index=False)[["commission_cost", "slippage_cost", "spread_cost", "total_transaction_cost"]]
        .sum()
        if ("quarter" in trades_df.columns and "total_transaction_cost" in trades_df.columns and not trades_df.empty)
        else pd.DataFrame(columns=["quarter", "commission_cost", "slippage_cost", "spread_cost", "total_transaction_cost"])
    )
    tx_by_stock = (
        trades_df.groupby("ric", as_index=False)[["commission_cost", "slippage_cost", "spread_cost", "total_transaction_cost"]]
        .sum()
        .sort_values("total_transaction_cost", ascending=False)
        if ("ric" in trades_df.columns and "total_transaction_cost" in trades_df.columns and not trades_df.empty)
        else pd.DataFrame(columns=["ric", "commission_cost", "slippage_cost", "spread_cost", "total_transaction_cost"])
    )

    total_tx_cost = float(trades_df["total_transaction_cost"].sum()) if "total_transaction_cost" in trades_df.columns else 0.0
    total_commission_cost = float(trades_df["commission_cost"].sum()) if "commission_cost" in trades_df.columns else 0.0
    total_slippage_cost = float(trades_df["slippage_cost"].sum()) if "slippage_cost" in trades_df.columns else 0.0
    total_spread_cost = float(trades_df["spread_cost"].sum()) if "spread_cost" in trades_df.columns else 0.0
    avg_tx_cost_per_trade = total_tx_cost / trade_count if trade_count > 0 else 0.0
    avg_commission_per_trade = total_commission_cost / trade_count if trade_count > 0 else 0.0
    avg_slippage_per_trade = total_slippage_cost / trade_count if trade_count > 0 else 0.0
    avg_spread_per_trade = total_spread_cost / trade_count if trade_count > 0 else 0.0
    avg_tx_cost_bps = 10000.0 * total_tx_cost / gross_turnover if gross_turnover > 0 else 0.0
    avg_commission_bps = 10000.0 * total_commission_cost / gross_turnover if gross_turnover > 0 else 0.0
    avg_slippage_bps = 10000.0 * total_slippage_cost / gross_turnover if gross_turnover > 0 else 0.0
    avg_spread_bps = 10000.0 * total_spread_cost / gross_turnover if gross_turnover > 0 else 0.0
    tx_cost_drag_initial = total_tx_cost / initial_capital if initial_capital > 0 else float("nan")
    tx_cost_drag_start = total_tx_cost / start_value if start_value > 0 else float("nan")

    if "portfolio_total_value_gross" in portfolio.columns:
        gross_total_series = pd.to_numeric(portfolio["portfolio_total_value_gross"], errors="coerce").fillna(total_value)
    elif "portfolio_transaction_cost_cumulative" in portfolio.columns:
        gross_total_series = total_value + pd.to_numeric(portfolio["portfolio_transaction_cost_cumulative"], errors="coerce").fillna(0.0)
    else:
        gross_total_series = total_value + total_tx_cost
    gross_end_value = float(gross_total_series.iloc[-1])
    gross_total_return = gross_end_value / start_value - 1.0 if start_value > 0 else float("nan")

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

    benchmark_section = _compute_benchmark_section(
        benchmark_close=benchmark_close,
        benchmark_label=benchmark_label,
        portfolio_returns=returns,
        portfolio_total_return=total_return,
        portfolio_index=portfolio.index,
        warning_messages=warnings_out,
        warning_prefix="Benchmark",
    )
    sp500_section = _compute_benchmark_section(
        benchmark_close=sp500_close,
        benchmark_label=sp500_label,
        portfolio_returns=returns,
        portfolio_total_return=total_return,
        portfolio_index=portfolio.index,
        warning_messages=warnings_out,
        warning_prefix="S&P 500 benchmark",
    )
    russell_1000_section = _compute_benchmark_section(
        benchmark_close=russell_1000_close,
        benchmark_label=russell_1000_label,
        portfolio_returns=returns,
        portfolio_total_return=total_return,
        portfolio_index=portfolio.index,
        warning_messages=warnings_out,
        warning_prefix="Russell 1000 benchmark",
    )

    sp500_aligned = (
        _prepare_benchmark_series(sp500_close, portfolio.index)
        if sp500_close is not None and not sp500_close.empty
        else None
    )
    russell_aligned = (
        _prepare_benchmark_series(russell_1000_close, portfolio.index)
        if russell_1000_close is not None and not russell_1000_close.empty
        else None
    )

    strategy_quarterly = _period_total_return(total_value, "Q")
    sp500_quarterly = _period_total_return(sp500_aligned, "Q") if sp500_aligned is not None else pd.Series(dtype=float)
    russell_quarterly = (
        _period_total_return(russell_aligned, "Q") if russell_aligned is not None else pd.Series(dtype=float)
    )
    quarterly_vs_sp500 = _pair_period_return_series(strategy_quarterly, sp500_quarterly)
    quarterly_vs_sp500_stats = _quadrant_period_stats(quarterly_vs_sp500)
    quarterly_compare_df = pd.concat(
        [
            strategy_quarterly.rename("strategy_total_return"),
            russell_quarterly.rename("russell_1000_total_return"),
            sp500_quarterly.rename("sp500_total_return"),
        ],
        axis=1,
    )
    quarterly_compare_df.index.name = "quarter"

    strategy_monthly_counts = _monthly_up_down_counts(total_value)
    sp500_monthly_counts = (
        _monthly_up_down_counts(sp500_aligned)
        if sp500_aligned is not None
        else {"up_months": 0, "down_months": 0, "flat_months": 0, "total_months": 0}
    )
    russell_monthly_counts = (
        _monthly_up_down_counts(russell_aligned)
        if russell_aligned is not None
        else {"up_months": 0, "down_months": 0, "flat_months": 0, "total_months": 0}
    )

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

    volatility_section = volatility_models_summary or {
        "enabled": False,
        "library": "arch",
        "input_observations": int(len(returns)),
        "models": {},
        "warnings": ["ARCH/GARCH models were not run for this report."],
    }
    for warn in volatility_section.get("warnings", []):
        if isinstance(warn, str) and warn:
            warnings_out.append(warn)

    allocation_strategy = _normalized_allocation_strategy(strategy_cfg)
    hyperparameters = {
        "portfolio_size": strategy_cfg.get("portfolio_size"),
        "rebalance_interval_quarters": strategy_cfg.get("rebalance_interval_quarters", 1),
        "allocation_strategy": allocation_strategy,
    }
    objective_metrics = {
        "cagr": float(cagr),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "information_ratio": _safe_float(sp500_section.get("information_ratio")),
        "annualized_excess_return": _safe_float(sp500_section.get("annualized_excess_return")),
    }
    constraint_metrics = {
        "max_drawdown": float(max_drawdown),
        "tracking_error_annualized": _safe_float(sp500_section.get("tracking_error_annualized")),
        "total_transaction_cost": float(total_tx_cost),
        "cost_drag_pct_of_start_value": float(tx_cost_drag_start),
        "gross_turnover": float(gross_turnover),
    }
    diagnostic_metrics = {
        "total_dividend_cash": float(div_total),
        "dividend_share_of_total_gain": float(div_share_gain),
        "positive_days": positive_days,
        "negative_days": negative_days,
        "quarterly_vs_sp500": _to_jsonable(quarterly_vs_sp500_stats),
    }
    dsr_readiness = {
        "study_id": evaluation_context_cfg.get("study_id"),
        "trial_id": evaluation_context_cfg.get("trial_id"),
        "search_space_id": evaluation_context_cfg.get("search_space_id"),
        "trial_index": evaluation_context_cfg.get("trial_index"),
        "total_trials_attempted": evaluation_context_cfg.get("total_trials_attempted"),
        "effective_independent_trial_count_estimate": evaluation_context_cfg.get(
            "effective_independent_trial_count_estimate"
        ),
        "sharpe_ratio": float(sharpe),
        "sample_length": int(len(returns)),
        "skewness": float(returns_skewness),
        "kurtosis_excess": float(returns_kurtosis),
        "ready_for_deflated_sharpe": bool(evaluation_context_cfg),
    }

    result = {
        "run_id": run_id,
        "created_at_utc": run_metadata.get("created_at_utc"),
        "start_date": run_metadata.get("start_date"),
        "end_date": run_metadata.get("end_date"),
        "risk_free_rate": float(risk_free_rate),
        "evaluation_context": evaluation_context_cfg,
        "hyperparameters": hyperparameters,
        "portfolio_metrics": {
            "start_value": start_value,
            "end_value": end_value,
            "gross_end_value_no_transaction_costs": gross_end_value,
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
        },
        "transaction_costs": {
            "enabled_in_config": bool(run_config.get("portfolio", {}).get("transaction_costs", {}).get("enabled", False)),
            "total_transaction_cost": float(total_tx_cost),
            "total_commission_cost": float(total_commission_cost),
            "total_slippage_cost": float(total_slippage_cost),
            "total_spread_cost": float(total_spread_cost),
            "avg_transaction_cost_per_trade": float(avg_tx_cost_per_trade),
            "avg_commission_per_trade": float(avg_commission_per_trade),
            "avg_slippage_per_trade": float(avg_slippage_per_trade),
            "avg_spread_per_trade": float(avg_spread_per_trade),
            "avg_transaction_cost_bps_of_notional": float(avg_tx_cost_bps),
            "avg_commission_bps_of_notional": float(avg_commission_bps),
            "avg_slippage_bps_of_notional": float(avg_slippage_bps),
            "avg_spread_bps_of_notional": float(avg_spread_bps),
            "cost_drag_pct_of_initial_capital": float(tx_cost_drag_initial),
            "cost_drag_pct_of_start_value": float(tx_cost_drag_start),
            "gross_total_return_no_transaction_costs": float(gross_total_return),
            "net_vs_gross_total_return_delta": float(gross_total_return - total_return),
        },
        "dividends": {
            "total_dividend_cash": float(div_total),
            "dividend_days": div_days,
            "dividend_share_of_total_gain": float(div_share_gain),
            "dividend_yield_on_start_value": float(div_yield_on_start),
            "dividend_yield_on_initial_capital": float(div_yield_on_initial_capital),
            "highest_dividend_paying_stock": highest_dividend_stock,
            "highest_quarterly_yield_score_stock": top_score_stock,
        },
        "trading_activity": {
            "number_of_trades": trade_count,
            "buy_trades": buy_count,
            "sell_trades": sell_count,
            "trade_days": trade_days,
            "rebalance_days": int(portfolio["rebalance_flag"].sum()),
            "gross_turnover": float(gross_turnover),
            "net_trade_value": float(net_trade_value),
        },
        "holdings": {
            "unique_stocks_held": unique_stocks_held,
            "average_positions_per_day": float(positions_daily.mean()) if len(positions_daily) else 0.0,
            "min_positions_per_day": int(positions_daily.min()) if len(positions_daily) else 0,
            "max_positions_per_day": int(positions_daily.max()) if len(positions_daily) else 0,
            "longest_staying_stock": (top_longest[0] if top_longest else None),
        },
        "universe": {
            "candidate_universe_rows": int(len(candidate_universe)) if candidate_universe is not None else None,
            "target_weight_rows": int(len(target_weights)),
            "quarter_score_rows": int(len(quarter_scores)),
        },
        "benchmark_comparison": benchmark_section,
        "sp500_comparison": sp500_section,
        "russell_1000_comparison": russell_1000_section,
        "objective_metrics": objective_metrics,
        "constraint_metrics": constraint_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "dsr_readiness": dsr_readiness,
        "comparative_period_stats": {
            "monthly_up_down_counts": {
                "strategy": strategy_monthly_counts,
                "sp500": sp500_monthly_counts,
                "russell_1000": russell_monthly_counts,
            },
            "quarterly_vs_sp500": quarterly_vs_sp500_stats,
        },
        "volatility_models": volatility_section,
        "dividend_policy": {
            "config_reinvest_dividends": config_reinvest,
            "effective_behavior": (
                "Dividends are credited to cash on pay dates and deployed into holdings at quarterly rebalances."
            ),
        },
        "strategy": {
            "mode": strategy_cfg.get("mode"),
            "universe_scope": strategy_cfg.get("universe_scope"),
            "candidate_count": strategy_cfg.get("candidate_count"),
            "portfolio_size": strategy_cfg.get("portfolio_size"),
            "rebalance_interval_quarters": strategy_cfg.get("rebalance_interval_quarters", 1),
            "allocation_strategy": allocation_strategy,
            "selection_policy_name": selection_policy_cfg.get("name", "full_refresh"),
            "max_replacements_per_quarter": selection_policy_cfg.get("max_replacements_per_quarter"),
            "selection_policy_rank_metric": selection_policy_cfg.get(
                "rank_metric",
                "quarter_dividend_yield_score",
            ),
            "experiment_group": strategy_cfg.get("experiment_group"),
        },
        "detailed_breakdowns": {
            "portfolio_quarterly_performance": quarterly_performance,
            "transaction_costs_per_quarter": tx_by_quarter.to_dict(orient="records"),
            "transaction_costs_per_stock_top10": tx_by_stock.head(10).to_dict(orient="records"),
            "quarterly_dividend_cash": quarterly_dividends.to_dict(orient="records"),
            "top_dividend_contributors": top_dividend,
            "trades_per_quarter": trades_per_quarter.to_dict(orient="records"),
            "turnover_per_quarter": turnover_per_quarter.to_dict(orient="records"),
            "top_longest_held_stocks": top_longest,
            "quarterly_total_return_comparison": quarterly_compare_df.reset_index().to_dict(orient="records"),
        },
        "warnings": warnings_out,
    }
    return _to_jsonable(result)


def _fetch_benchmark_close(
    *,
    ric: str,
    start_date: str,
    end_date: str,
    cache_db_path: str | Path | None = None,
) -> pd.DataFrame:
    def load_from_cache(db_path: Path) -> pd.DataFrame | None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS benchmark_close_cache_requests (
                    ric TEXT NOT NULL,
                    request_start_date TEXT NOT NULL,
                    request_end_date TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    fetched_at_utc TEXT NOT NULL,
                    PRIMARY KEY (ric, request_start_date, request_end_date)
                );

                CREATE TABLE IF NOT EXISTS benchmark_close_cache_prices (
                    ric TEXT NOT NULL,
                    request_start_date TEXT NOT NULL,
                    request_end_date TEXT NOT NULL,
                    date TEXT NOT NULL,
                    close REAL NOT NULL,
                    PRIMARY KEY (ric, request_start_date, request_end_date, date)
                );
                """
            )
            row = conn.execute(
                """
                SELECT row_count
                FROM benchmark_close_cache_requests
                WHERE ric = ?
                  AND request_start_date = ?
                  AND request_end_date = ?
                LIMIT 1
                """,
                (ric, start_date, end_date),
            ).fetchone()
            if row is None:
                return None
            expected_rows = int(row[0])
            cached = pd.read_sql_query(
                """
                SELECT date AS Date, ric AS RIC, close AS CLOSE
                FROM benchmark_close_cache_prices
                WHERE ric = ?
                  AND request_start_date = ?
                  AND request_end_date = ?
                ORDER BY date
                """,
                conn,
                params=[ric, start_date, end_date],
            )
            if len(cached) != expected_rows:
                conn.execute(
                    """
                    DELETE FROM benchmark_close_cache_requests
                    WHERE ric = ?
                      AND request_start_date = ?
                      AND request_end_date = ?
                    """,
                    (ric, start_date, end_date),
                )
                conn.execute(
                    """
                    DELETE FROM benchmark_close_cache_prices
                    WHERE ric = ?
                      AND request_start_date = ?
                      AND request_end_date = ?
                    """,
                    (ric, start_date, end_date),
                )
                conn.commit()
                return None
            if cached.empty:
                return pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
            cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce")
            cached["CLOSE"] = pd.to_numeric(cached["CLOSE"], errors="coerce")
            cached = cached.dropna(subset=["Date", "CLOSE"])
            cached = cached.loc[cached["CLOSE"] > 0]
            return cached.reset_index(drop=True)
        finally:
            conn.close()

    def store_in_cache(db_path: Path, out: pd.DataFrame) -> None:
        if out.empty:
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS benchmark_close_cache_requests (
                    ric TEXT NOT NULL,
                    request_start_date TEXT NOT NULL,
                    request_end_date TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    fetched_at_utc TEXT NOT NULL,
                    PRIMARY KEY (ric, request_start_date, request_end_date)
                );

                CREATE TABLE IF NOT EXISTS benchmark_close_cache_prices (
                    ric TEXT NOT NULL,
                    request_start_date TEXT NOT NULL,
                    request_end_date TEXT NOT NULL,
                    date TEXT NOT NULL,
                    close REAL NOT NULL,
                    PRIMARY KEY (ric, request_start_date, request_end_date, date)
                );
                """
            )
            conn.execute(
                """
                DELETE FROM benchmark_close_cache_prices
                WHERE ric = ?
                  AND request_start_date = ?
                  AND request_end_date = ?
                """,
                (ric, start_date, end_date),
            )
            rows = [
                (
                    str(ric),
                    str(start_date),
                    str(end_date),
                    pd.Timestamp(row.Date).date().isoformat(),
                    float(row.CLOSE),
                )
                for row in out.itertuples(index=False)
                if pd.notna(row.Date) and pd.notna(row.CLOSE)
            ]
            if rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO benchmark_close_cache_prices
                    (ric, request_start_date, request_end_date, date, close)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_close_cache_requests
                (ric, request_start_date, request_end_date, row_count, fetched_at_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(ric),
                    str(start_date),
                    str(end_date),
                    len(rows),
                    pd.Timestamp.utcnow().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    resolved_cache_db = Path(cache_db_path) if cache_db_path is not None else BENCHMARK_CACHE_DB_DEFAULT
    cached = load_from_cache(resolved_cache_db)
    if cached is not None:
        return cached

    with RefinitivClient() as client:
        frames: list[pd.DataFrame] = []
        for field in ("TRDPRC_1", "TR.PriceClose"):
            try:
                history_kwargs = {
                    "universe": [ric],
                    "fields": [field],
                    "interval": "daily",
                    "start": start_date,
                    "end": end_date,
                }
                if field == "TRDPRC_1":
                    history_kwargs["adjustments"] = "unadjusted"
                raw = client.get_history(**history_kwargs)
            except Exception:  # noqa: BLE001
                continue

            if raw is None or raw.empty:
                continue
            df = raw.copy()
            if "Date" in df.columns:
                dates = pd.to_datetime(df["Date"], errors="coerce")
            else:
                dates = pd.to_datetime(df.index, errors="coerce")
            df = df.loc[dates.notna()].copy()
            if df.empty:
                continue
            df_dates = dates[dates.notna()]

            if "TRDPRC_1" in df.columns:
                close = pd.to_numeric(df["TRDPRC_1"], errors="coerce")
            elif "Price Close" in df.columns:
                close = pd.to_numeric(df["Price Close"], errors="coerce")
            else:
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if not numeric_cols:
                    continue
                close = pd.to_numeric(df[numeric_cols[0]], errors="coerce")

            out = pd.DataFrame({"Date": df_dates.values, "RIC": ric, "CLOSE": close.values})
            out = out.dropna(subset=["Date", "CLOSE"])
            out = out.loc[out["CLOSE"] > 0]
            if not out.empty:
                frames.append(out[["Date", "RIC", "CLOSE"]])
                break

        if not frames:
            return pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        store_in_cache(resolved_cache_db, out)
        return out


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
    sp500_close: pd.DataFrame | None = None,
    sp500_label: str = SP500_LABEL_DEFAULT,
    russell_1000_close: pd.DataFrame | None = None,
    russell_1000_label: str = RUSSELL_1000_LABEL_DEFAULT,
    volatility_model_series: pd.DataFrame | None = None,
) -> list[str]:
    created: list[str] = []

    portfolio = _date_col(portfolio_daily, "date")
    portfolio["portfolio_total_value"] = pd.to_numeric(portfolio["portfolio_total_value"], errors="coerce")
    portfolio["portfolio_market_value"] = pd.to_numeric(portfolio["portfolio_market_value"], errors="coerce")
    portfolio["portfolio_cash_balance"] = pd.to_numeric(portfolio["portfolio_cash_balance"], errors="coerce")
    portfolio["portfolio_dividend_cash_daily"] = pd.to_numeric(portfolio["portfolio_dividend_cash_daily"], errors="coerce").fillna(0.0)
    if "portfolio_transaction_cost_daily" in portfolio.columns:
        portfolio["portfolio_transaction_cost_daily"] = pd.to_numeric(
            portfolio["portfolio_transaction_cost_daily"], errors="coerce"
        ).fillna(0.0)
    else:
        portfolio["portfolio_transaction_cost_daily"] = 0.0
    if "portfolio_commission_cost_daily" in portfolio.columns:
        portfolio["portfolio_commission_cost_daily"] = pd.to_numeric(
            portfolio["portfolio_commission_cost_daily"], errors="coerce"
        ).fillna(0.0)
    else:
        portfolio["portfolio_commission_cost_daily"] = 0.0
    if "portfolio_slippage_cost_daily" in portfolio.columns:
        portfolio["portfolio_slippage_cost_daily"] = pd.to_numeric(
            portfolio["portfolio_slippage_cost_daily"], errors="coerce"
        ).fillna(0.0)
    else:
        portfolio["portfolio_slippage_cost_daily"] = 0.0
    if "portfolio_spread_cost_daily" in portfolio.columns:
        portfolio["portfolio_spread_cost_daily"] = pd.to_numeric(
            portfolio["portfolio_spread_cost_daily"], errors="coerce"
        ).fillna(0.0)
    else:
        portfolio["portfolio_spread_cost_daily"] = 0.0
    if "portfolio_transaction_cost_cumulative" in portfolio.columns:
        portfolio["portfolio_transaction_cost_cumulative"] = pd.to_numeric(
            portfolio["portfolio_transaction_cost_cumulative"], errors="coerce"
        ).fillna(0.0)
    else:
        portfolio["portfolio_transaction_cost_cumulative"] = 0.0
    if "portfolio_total_value_gross" in portfolio.columns:
        portfolio["portfolio_total_value_gross"] = pd.to_numeric(
            portfolio["portfolio_total_value_gross"], errors="coerce"
        )
    else:
        portfolio["portfolio_total_value_gross"] = pd.NA
    if portfolio["portfolio_total_value_gross"].isna().all():
        portfolio["portfolio_total_value_gross"] = (
            portfolio["portfolio_total_value"] + portfolio["portfolio_transaction_cost_cumulative"]
        )
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
    ax.plot(
        portfolio.index,
        portfolio["portfolio_transaction_cost_cumulative"],
        color="tab:red",
        label="Cumulative Transaction Costs",
    )
    ax.set_title("Cumulative Transaction Costs")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.3)
    ax.legend()
    file_name = "cumulative_transaction_costs.png"
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

    if volatility_model_series is not None and not volatility_model_series.empty:
        model_vol = volatility_model_series.copy()
        model_vol.index = pd.to_datetime(model_vol.index, errors="coerce")
        model_vol = model_vol.loc[model_vol.index.notna()].sort_index()
        model_vol = model_vol.reindex(portfolio.index).ffill()
        realized_21d = returns.rolling(21).std(ddof=0) * math.sqrt(252.0)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(realized_21d.index, realized_21d, color="black", alpha=0.6, label="Realized Vol (21d)")
        if "arch_1_0_cond_vol" in model_vol.columns:
            ax.plot(
                model_vol.index,
                pd.to_numeric(model_vol["arch_1_0_cond_vol"], errors="coerce") * math.sqrt(252.0),
                color="tab:blue",
                label="ARCH(1) Cond. Vol",
            )
        if "garch_1_1_cond_vol" in model_vol.columns:
            ax.plot(
                model_vol.index,
                pd.to_numeric(model_vol["garch_1_1_cond_vol"], errors="coerce") * math.sqrt(252.0),
                color="tab:red",
                label="GARCH(1,1) Cond. Vol",
            )
        ax.set_title("ARCH/GARCH Conditional Volatility vs Realized Volatility")
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualized Volatility")
        ax.grid(True, alpha=0.3)
        ax.legend()
        file_name = "arch_garch_conditional_volatility.png"
        _plot_and_save(out_dir / file_name)
        created.append(file_name)

    quarterly_returns_for_hist = _period_total_return(portfolio["portfolio_total_value"], "Q").dropna()
    if not quarterly_returns_for_hist.empty:
        bin_width = 0.02  # 2 percentage points
        x_min = float(quarterly_returns_for_hist.min())
        x_max = float(quarterly_returns_for_hist.max())
        left = math.floor(x_min / bin_width) * bin_width
        right = math.ceil(x_max / bin_width) * bin_width
        if right <= left:
            right = left + bin_width
        hist_bins = np.arange(left, right + (bin_width * 1.01), bin_width)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(quarterly_returns_for_hist, bins=hist_bins, color="tab:purple", alpha=0.75)
        ax.set_title("Quarterly Return Distribution")
        ax.set_xlabel("Quarterly Return (%)")
        ax.set_ylabel("Frequency")
        ax.set_xlim(left, right)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(bin_width))
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, alpha=0.2)
        file_name = "quarterly_return_distribution.png"
        _plot_and_save(out_dir / file_name)
        created.append(file_name)

    cum_port = portfolio["portfolio_total_value"] / float(portfolio["portfolio_total_value"].iloc[0]) - 1.0
    benchmark = _prepare_benchmark_series(benchmark_close if benchmark_close is not None else pd.DataFrame(), portfolio.index)
    sp500 = _prepare_benchmark_series(sp500_close if sp500_close is not None else pd.DataFrame(), portfolio.index)
    russell_1000 = _prepare_benchmark_series(
        russell_1000_close if russell_1000_close is not None else pd.DataFrame(),
        portfolio.index,
    )
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

    wealth_port = portfolio["portfolio_total_value"] / float(portfolio["portfolio_total_value"].iloc[0])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wealth_port.index, wealth_port, label="Portfolio", linewidth=1.8)
    if benchmark is not None:
        wealth_bench = benchmark / float(benchmark.iloc[0])
        ax.plot(wealth_bench.index, wealth_bench, label=benchmark_label, linewidth=1.6)
    ax.set_yscale("log")
    ax.set_title("Normalized Value Comparison (Log Scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Value (log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    file_name = "cumulative_return_vs_sp500_log.png"
    _plot_and_save(out_dir / file_name)
    created.append(file_name)

    quarter_returns = pd.DataFrame(
        {
            "Strategy": _period_total_return(portfolio["portfolio_total_value"], "Q"),
            russell_1000_label: _period_total_return(russell_1000, "Q")
            if russell_1000 is not None
            else pd.Series(dtype=float),
            sp500_label: _period_total_return(sp500, "Q") if sp500 is not None else pd.Series(dtype=float),
        }
    )
    quarter_returns = quarter_returns.dropna(how="all")
    if not quarter_returns.empty:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(quarter_returns.index, quarter_returns["Strategy"], label="Strategy", linewidth=1.9, marker="o", markersize=3)
        if russell_1000_label in quarter_returns.columns and quarter_returns[russell_1000_label].notna().any():
            ax.plot(
                quarter_returns.index,
                quarter_returns[russell_1000_label],
                label=russell_1000_label,
                linewidth=1.5,
                marker="o",
                markersize=3,
            )
        if sp500_label in quarter_returns.columns and quarter_returns[sp500_label].notna().any():
            ax.plot(
                quarter_returns.index,
                quarter_returns[sp500_label],
                label=sp500_label,
                linewidth=1.5,
                marker="o",
                markersize=3,
            )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
        ax.set_title("Quarterly Total Return Comparison")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Quarterly Return")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=75)
        ax.legend()
        file_name = "quarterly_return_comparison_strategy_russell_sp500.png"
        _plot_and_save(out_dir / file_name)
        created.append(file_name)

    if sp500 is not None:
        quarterly_vs_sp500 = _pair_period_return_series(
            _period_total_return(portfolio["portfolio_total_value"], "Q"),
            _period_total_return(sp500, "Q"),
        )
        quarterly_vs_sp500_stats = _quadrant_period_stats(quarterly_vs_sp500)
        if not quarterly_vs_sp500.empty:
            strategy_ret = quarterly_vs_sp500["strategy_return"]
            sp500_ret = quarterly_vs_sp500["benchmark_return"]
            max_abs = max(
                float(strategy_ret.abs().max()),
                float(sp500_ret.abs().max()),
                0.05,
            )
            step = 0.05 if max_abs >= 0.25 else 0.02
            axis_limit = math.ceil(max_abs / step) * step

            fig, ax = plt.subplots(figsize=(8, 8))
            quadrant_groups = [
                ("Both Up", (strategy_ret > 0.0) & (sp500_ret > 0.0), "tab:green"),
                ("Strategy Up, S&P Down", (strategy_ret > 0.0) & (sp500_ret < 0.0), "tab:blue"),
                ("Strategy Down, S&P Up", (strategy_ret < 0.0) & (sp500_ret > 0.0), "tab:orange"),
                ("Both Down", (strategy_ret < 0.0) & (sp500_ret < 0.0), "tab:red"),
            ]
            for label, mask, color in quadrant_groups:
                if bool(mask.any()):
                    ax.scatter(
                        strategy_ret.loc[mask],
                        sp500_ret.loc[mask],
                        s=54,
                        alpha=0.85,
                        color=color,
                        edgecolors="white",
                        linewidths=0.5,
                        label=f"{label} (n={int(mask.sum())})",
                    )

            on_axis = (strategy_ret == 0.0) | (sp500_ret == 0.0)
            if bool(on_axis.any()):
                ax.scatter(
                    strategy_ret.loc[on_axis],
                    sp500_ret.loc[on_axis],
                    s=54,
                    alpha=0.85,
                    color="tab:gray",
                    edgecolors="white",
                    linewidths=0.5,
                    label=f"On Axis (n={int(on_axis.sum())})",
                )

            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
            ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
            ax.plot(
                [-axis_limit, axis_limit],
                [-axis_limit, axis_limit],
                linestyle="--",
                color="black",
                linewidth=1.0,
                alpha=0.35,
                label="Equal Return",
            )
            ax.set_xlim(-axis_limit, axis_limit)
            ax.set_ylim(-axis_limit, axis_limit)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
            ax.set_title("Quarterly Return Quadrant: Strategy vs S&P 500")
            ax.set_xlabel("Strategy Quarterly Return")
            ax.set_ylabel("S&P 500 Quarterly Return")
            ax.grid(True, alpha=0.25)

            corr_text = quarterly_vs_sp500_stats.get("correlation")
            corr_label = f"{corr_text:.2f}" if corr_text is not None and not math.isnan(corr_text) else "n/a"
            stats_text = "\n".join(
                [
                    f"Corr: {corr_label}",
                    (
                        f"Up when S&P up: {quarterly_vs_sp500_stats['both_up_periods']}"
                        f"/{quarterly_vs_sp500_stats['benchmark_up_periods']}"
                    ),
                    (
                        f"Down when S&P down: {quarterly_vs_sp500_stats['both_down_periods']}"
                        f"/{quarterly_vs_sp500_stats['benchmark_down_periods']}"
                    ),
                    (
                        f"Same direction: {quarterly_vs_sp500_stats['same_direction_periods']}"
                        f"/{quarterly_vs_sp500_stats['total_periods']}"
                    ),
                    (
                        f"Beat S&P: {quarterly_vs_sp500_stats['beat_benchmark_periods']}"
                        f"/{quarterly_vs_sp500_stats['total_periods']}"
                    ),
                ]
            )
            ax.text(
                0.03,
                0.97,
                stats_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
            )
            ax.legend(loc="lower right")
            file_name = "quarterly_return_quadrant_strategy_vs_sp500.png"
            _plot_and_save(out_dir / file_name)
            created.append(file_name)

    month_counts: list[tuple[str, dict[str, int]]] = [("Strategy", _monthly_up_down_counts(portfolio["portfolio_total_value"]))]
    if russell_1000 is not None:
        month_counts.append((russell_1000_label, _monthly_up_down_counts(russell_1000)))
    if sp500 is not None:
        month_counts.append((sp500_label, _monthly_up_down_counts(sp500)))

    if month_counts:
        labels = [item[0] for item in month_counts]
        up_vals = [item[1]["up_months"] for item in month_counts]
        down_vals = [item[1]["down_months"] for item in month_counts]
        x = np.arange(len(labels))
        width = 0.36

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - width / 2, up_vals, width=width, label="Up Months", color="tab:green", alpha=0.85)
        ax.bar(x + width / 2, down_vals, width=width, label="Down Months", color="tab:red", alpha=0.85)
        ax.set_title("Monthly Up/Down Count Comparison")
        ax.set_xlabel("Series")
        ax.set_ylabel("Months")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        file_name = "monthly_up_down_counts_strategy_russell_sp500.png"
        _plot_and_save(out_dir / file_name)
        created.append(file_name)

    gross_cum = portfolio["portfolio_total_value_gross"] / float(portfolio["portfolio_total_value_gross"].iloc[0]) - 1.0
    net_cum = portfolio["portfolio_total_value"] / float(portfolio["portfolio_total_value"].iloc[0]) - 1.0
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(net_cum.index, net_cum, label="Net (after transaction costs)", linewidth=1.8)
    ax.plot(gross_cum.index, gross_cum, label="Gross (without transaction costs)", linewidth=1.6)
    ax.set_title("Gross vs Net Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    file_name = "gross_vs_net_cumulative_return.png"
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
        if "commission_cost" in trades_df.columns:
            trades_df["commission_cost"] = pd.to_numeric(trades_df["commission_cost"], errors="coerce").fillna(0.0)
        else:
            trades_df["commission_cost"] = 0.0
        if "slippage_cost" in trades_df.columns:
            trades_df["slippage_cost"] = pd.to_numeric(trades_df["slippage_cost"], errors="coerce").fillna(0.0)
        else:
            trades_df["slippage_cost"] = 0.0
        if "spread_cost" in trades_df.columns:
            trades_df["spread_cost"] = pd.to_numeric(trades_df["spread_cost"], errors="coerce").fillna(0.0)
        else:
            trades_df["spread_cost"] = 0.0
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

        tx_quarter = trades_df.groupby("quarter")[["commission_cost", "slippage_cost", "spread_cost"]].sum()
        if not tx_quarter.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            tx_quarter.plot(kind="bar", stacked=True, ax=ax, alpha=0.9)
            ax.set_title("Quarterly Transaction Cost Components")
            ax.set_xlabel("Quarter")
            ax.set_ylabel("USD")
            ax.grid(True, axis="y", alpha=0.25)
            file_name = "quarterly_transaction_cost_components.png"
            _plot_and_save(out_dir / file_name)
            created.append(file_name)

    return created


def generate_dynamic_showresults(
    *,
    db_path: str | Path,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
    include_benchmark: bool = True,
    benchmark_ric: str = SP500_RIC_DEFAULT,
    benchmark_label: str = SP500_LABEL_DEFAULT,
    sp500_ric: str = SP500_RIC_DEFAULT,
    sp500_label: str = SP500_LABEL_DEFAULT,
    russell_1000_ric: str = RUSSELL_1000_RIC_DEFAULT,
    russell_1000_label: str = RUSSELL_1000_LABEL_DEFAULT,
) -> tuple[str, Path]:
    resolved_run_id = resolve_run_id(db_path, run_id=run_id)
    data = _load_run_data(db_path, resolved_run_id)
    metadata = data["metadata"]

    out_dir = Path(output_dir) if output_dir is not None else Path("data/runs_dynamic") / resolved_run_id / "showresults"
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df = pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
    sp500_df = pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
    russell_1000_df = pd.DataFrame(columns=["Date", "RIC", "CLOSE"])
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
        try:
            if sp500_ric == benchmark_ric:
                sp500_df = benchmark_df.copy()
            else:
                sp500_df = _fetch_benchmark_close(
                    ric=sp500_ric,
                    start_date=str(metadata.get("start_date")),
                    end_date=str(metadata.get("end_date")),
                )
        except Exception as exc:  # noqa: BLE001
            warnings_out.append(f"S&P 500 benchmark fetch failed for {sp500_ric}: {exc}")
        try:
            if russell_1000_ric == benchmark_ric:
                russell_1000_df = benchmark_df.copy()
            else:
                russell_1000_df = _fetch_benchmark_close(
                    ric=russell_1000_ric,
                    start_date=str(metadata.get("start_date")),
                    end_date=str(metadata.get("end_date")),
                )
        except Exception as exc:  # noqa: BLE001
            warnings_out.append(f"Russell 1000 benchmark fetch failed for {russell_1000_ric}: {exc}")

    portfolio_for_models = _date_col(data["portfolio_daily"], "date")
    portfolio_for_models["portfolio_total_value"] = pd.to_numeric(
        portfolio_for_models.get("portfolio_total_value"),
        errors="coerce",
    )
    portfolio_for_models = portfolio_for_models.dropna(subset=["date", "portfolio_total_value"]).set_index("date").sort_index()
    model_returns = (
        portfolio_for_models["portfolio_total_value"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if not portfolio_for_models.empty
        else pd.Series(dtype=float)
    )
    volatility_summary, volatility_series = fit_arch_garch_models(model_returns)

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
        sp500_close=sp500_df,
        sp500_label=sp500_label,
        russell_1000_close=russell_1000_df,
        russell_1000_label=russell_1000_label,
        warning_messages=warnings_out,
        volatility_models_summary=volatility_summary,
    )

    plots = _create_plots(
        out_dir=out_dir,
        portfolio_daily=data["portfolio_daily"],
        holdings_daily=data["holdings_daily"],
        trades=data["trades"],
        benchmark_close=benchmark_df if include_benchmark else None,
        benchmark_label=benchmark_label,
        sp500_close=sp500_df if include_benchmark else None,
        sp500_label=sp500_label,
        russell_1000_close=russell_1000_df if include_benchmark else None,
        russell_1000_label=russell_1000_label,
        volatility_model_series=volatility_series,
    )
    summary["generated_plots"] = plots

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, indent=2, sort_keys=False)

    return resolved_run_id, out_dir
