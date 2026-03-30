from __future__ import annotations

import pandas as pd


def _stable_desc_rank(df: pd.DataFrame, value_col: str, rank_col: str) -> pd.DataFrame:
    out = df.copy()
    out["RIC"] = out["RIC"].astype(str)
    out = out.sort_values(by=[value_col, "RIC"], ascending=[False, True]).reset_index(drop=True)
    out[rank_col] = out.index + 1
    return out


def select_top_candidates_by_market_cap(
    market_caps: pd.DataFrame,
    dividend_payers: set[str],
    candidate_count: int,
) -> pd.DataFrame:
    if market_caps.empty:
        return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate", "RankByMarketCap"])

    out = market_caps.copy()
    out["RIC"] = out["RIC"].astype(str).str.strip()
    out["MarketCap"] = pd.to_numeric(out["MarketCap"], errors="coerce")
    out["MarketCapDate"] = pd.to_datetime(out["MarketCapDate"], errors="coerce")
    out = out.dropna(subset=["RIC", "MarketCap"]).loc[out["MarketCap"] > 0]
    out = out.loc[out["RIC"].isin(dividend_payers)]
    if out.empty:
        return pd.DataFrame(columns=["RIC", "MarketCap", "MarketCapDate", "RankByMarketCap"])

    out = _stable_desc_rank(out, "MarketCap", "RankByMarketCap")
    out = out.loc[out["RankByMarketCap"] <= max(candidate_count, 1)]
    return out.reset_index(drop=True)


def select_initial_portfolio_by_market_cap(candidates: pd.DataFrame, portfolio_size: int) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["RIC", "MarketCap", "RankInPortfolio", "Weight"])

    out = candidates.copy()
    if "RankByMarketCap" not in out.columns:
        out = _stable_desc_rank(out, "MarketCap", "RankByMarketCap")

    out = out.sort_values(["RankByMarketCap", "RIC"], ascending=[True, True]).head(max(portfolio_size, 1))
    total = float(out["MarketCap"].sum())
    out["Weight"] = out["MarketCap"] / total if total > 0 else 1.0 / len(out)
    out = out.reset_index(drop=True)
    out["RankInPortfolio"] = out.index + 1
    return out[["RIC", "MarketCap", "RankInPortfolio", "Weight"]]


def _quarter_close_stat(prices: pd.DataFrame, denominator: str) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["RIC", "CloseDenominator"])

    out = prices.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["CLOSE"] = pd.to_numeric(out["CLOSE"], errors="coerce")
    out = out.dropna(subset=["RIC", "Date", "CLOSE"]).sort_values(["RIC", "Date"])

    if denominator == "quarter_start_close":
        stat = out.groupby("RIC", as_index=False).first()[["RIC", "CLOSE"]]
    elif denominator == "quarter_end_close":
        stat = out.groupby("RIC", as_index=False).last()[["RIC", "CLOSE"]]
    else:
        stat = out.groupby("RIC", as_index=False)["CLOSE"].mean()

    stat = stat.rename(columns={"CLOSE": "CloseDenominator"})
    return stat


def compute_quarter_dividend_yield_scores(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    denominator: str = "quarter_average_close",
) -> pd.DataFrame:
    price_stat = _quarter_close_stat(prices, denominator)
    if price_stat.empty:
        return pd.DataFrame(columns=["RIC", "AvgClose", "DividendSumPS", "Score"])

    div_sum = pd.DataFrame(columns=["RIC", "DividendSumPS"])
    if not dividends.empty:
        div = dividends.copy()
        div["RIC"] = div["RIC"].astype(str).str.strip()
        div["Dividend"] = pd.to_numeric(div["Dividend"], errors="coerce")
        div = div.dropna(subset=["RIC", "Dividend"])
        div_sum = div.groupby("RIC", as_index=False)["Dividend"].sum().rename(columns={"Dividend": "DividendSumPS"})

    out = price_stat.merge(div_sum, on="RIC", how="left")
    out["DividendSumPS"] = out["DividendSumPS"].fillna(0.0)
    out["AvgClose"] = pd.to_numeric(out["CloseDenominator"], errors="coerce")
    out = out.drop(columns=["CloseDenominator"])
    out["Score"] = pd.Series(0.0, index=out.index, dtype=float)
    mask = out["AvgClose"] > 0
    out.loc[mask, "Score"] = (
        pd.to_numeric(out.loc[mask, "DividendSumPS"], errors="coerce").astype(float)
        / pd.to_numeric(out.loc[mask, "AvgClose"], errors="coerce").astype(float)
    ).astype(float)
    out = _stable_desc_rank(out, "Score", "RankByScore")
    return out[["RIC", "AvgClose", "DividendSumPS", "Score", "RankByScore"]]


def select_top_portfolio_by_score(
    scores: pd.DataFrame,
    portfolio_size: int,
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["RIC", "Score", "RankInPortfolio", "Weight"])

    out = scores.copy()
    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    out = out.dropna(subset=["RIC", "Score"])
    out = out.sort_values(["Score", "RIC"], ascending=[False, True]).reset_index(drop=True)

    positive = out.loc[out["Score"] > 0].copy()
    if len(positive) >= portfolio_size:
        selected = positive.head(portfolio_size).copy()
    else:
        selected = out.head(max(portfolio_size, 1)).copy()

    weight_den = float(selected["Score"].sum())
    if weight_den > 0:
        selected["Weight"] = selected["Score"] / weight_den
    else:
        selected["Weight"] = 1.0 / len(selected)

    selected = selected.reset_index(drop=True)
    selected["RankInPortfolio"] = selected.index + 1
    return selected[["RIC", "Score", "RankInPortfolio", "Weight"]]
