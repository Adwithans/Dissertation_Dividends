from __future__ import annotations

import os
from pathlib import Path

# Matplotlib needs a writable config/cache dir in some environments (e.g. locked HOME).
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = (Path.cwd() / "data" / ".mplconfig").resolve()
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


def save_portfolio_plots(
    portfolio_df: pd.DataFrame,
    asset_results: dict[str, pd.DataFrame],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(portfolio_df.index, portfolio_df["Portfolio_Total_Value"], label="Portfolio Total Value")
    ax.set_title("Portfolio Total Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "portfolio_total_value.png", dpi=140)
    plt.close(fig)

    running_max = portfolio_df["Portfolio_Total_Value"].cummax()
    drawdown = portfolio_df["Portfolio_Total_Value"] / running_max - 1.0
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(portfolio_df.index, drawdown, color="tab:red", label="Drawdown")
    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "portfolio_drawdown.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for ric, df in asset_results.items():
        ax.plot(df.index, df["Market_Value"], label=ric)
    ax.set_title("Per-Asset Market Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "asset_market_values.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        portfolio_df.index,
        portfolio_df["Portfolio_Dividend_Income"],
        label="Cumulative Dividend Income",
        color="tab:green",
    )
    ax.set_title("Portfolio Cumulative Dividend Income")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "portfolio_dividend_income.png", dpi=140)
    plt.close(fig)
