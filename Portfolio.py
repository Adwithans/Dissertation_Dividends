from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# IO
# -------------------------
def load_history_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.loc[df.index.notna()].sort_index()
    return df


# -------------------------
# Split utilities
# -------------------------
def _effective_share_multiplier(raw_factor: float, snap_tol: float = 0.02) -> float:
    """
    Convert various vendor split factor formats into a SHARE multiplier.

    - If raw_factor is already a split ratio (e.g., 4), returns 4.
    - If raw_factor looks like a price adjustment factor (<1), e.g. 0.25 for 4-for-1,
      invert it -> 4 (and snap close-to-integer values).
    """
    if pd.isna(raw_factor) or raw_factor <= 0:
        return 1.0

    if raw_factor < 1.0:
        inv = 1.0 / raw_factor
        inv_round = round(inv)
        if abs(inv - inv_round) <= snap_tol:
            return float(inv_round)
        return float(inv)

    # raw_factor >= 1.0: might already be a split ratio
    return float(raw_factor)


def build_split_multiplier(
    df: pd.DataFrame,
    *,
    price_col: str,
    split_col: str = "SplitFactor",
    cum_factor_col: str = "cum_factor",
    use_cum_factor: bool = True,
    auto_align: bool = True,
) -> pd.Series:
    """
    Returns a Series 'Split_Multiplier' which is the SHARE multiplier applied on each date.
    """
    df = df.copy()

    # Choose raw split signal:
    # - If cum_factor exists: use its day-to-day ratio (often a price-adjustment factor)
    # - Else: use split_col directly
    raw = None
    if use_cum_factor and (cum_factor_col in df.columns):
        cf = pd.to_numeric(df[cum_factor_col], errors="coerce").ffill().fillna(1.0)
        raw = (cf / cf.shift(1)).fillna(1.0)
    else:
        if split_col in df.columns:
            raw = pd.to_numeric(df[split_col], errors="coerce").fillna(1.0)
        else:
            raw = pd.Series(1.0, index=df.index)

    # Convert whatever raw is into a SHARE multiplier
    split_mult = raw.apply(_effective_share_multiplier).fillna(1.0)

    # Optional: auto-align split to match the actual price jump day (prevents spike-then-crash)
    if auto_align and price_col in df.columns:
        price = pd.to_numeric(df[price_col], errors="coerce")
        expected = (price.shift(1) / price).replace([np.inf, -np.inf], np.nan)

        mask = split_mult.ne(1.0) & expected.notna()

        if mask.any():
            # compare "same day" vs "shifted by 1 day"
            err_same = np.nanmedian(np.abs(np.log(split_mult[mask] / expected[mask])))
            shifted = split_mult.shift(1)
            err_shift1 = np.nanmedian(np.abs(np.log(shifted[mask] / expected[mask])))

            if np.isfinite(err_shift1) and err_shift1 < err_same:
                split_mult = shifted.fillna(1.0)

    return split_mult


# -------------------------
# Simulation
# -------------------------
def simulate_single_stock_portfolio(
    df: pd.DataFrame,
    *,
    initial_investment: float = 1_000_000,
    price_col: str = "UNADJ_CLOSE",
    dividend_col: str = "Dividend",
    split_col: str = "SplitFactor",
    cum_factor_col: str = "cum_factor",
    use_cum_factor: bool = True,
    auto_align_splits: bool = True,
    reinvest_dividends: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    df = df.sort_index().copy()

    # price column fallback
    if price_col not in df.columns:
        if "CLOSE" in df.columns:
            price_col = "CLOSE"
        else:
            raise ValueError(f"Missing price column '{price_col}' (and no 'CLOSE' fallback).")

    # clean numeric columns
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.loc[df[price_col].notna() & (df[price_col] > 0)].copy()
    if df.empty:
        raise ValueError("No valid price rows after cleaning.")

    if dividend_col not in df.columns:
        df[dividend_col] = 0.0
    df[dividend_col] = pd.to_numeric(df[dividend_col], errors="coerce").fillna(0.0)

    # build split multiplier (share multiplier)
    df["Split_Multiplier"] = build_split_multiplier(
        df,
        price_col=price_col,
        split_col=split_col,
        cum_factor_col=cum_factor_col,
        use_cum_factor=use_cum_factor,
        auto_align=auto_align_splits,
    )

    # initial shares
    first_price = df[price_col].iloc[0]
    shares = initial_investment / first_price
    cash = 0.0
    cum_div = 0.0

    total_values = []
    shares_list = []
    cash_list = []
    cumdiv_list = []

    if verbose:
        print("--- Simulation Started ---")
        print(f"Initial Investment: ${initial_investment:,.2f}")
        print(f"Start date: {df.index[0].date()}  Price: ${first_price:,.4f}")
        print(f"Initial shares: {shares:,.6f}")

    for date, row in df.iterrows():
        split_mult = float(row["Split_Multiplier"])
        div = float(row[dividend_col])
        price = float(row[price_col])

        # apply split to shares
        if split_mult != 1.0:
            shares *= split_mult

        # dividends
        if div > 0:
            div_cash = shares * div
            cum_div += div_cash
            if reinvest_dividends:
                shares += div_cash / price
            else:
                cash += div_cash

        stock_value = shares * price
        total = stock_value + cash

        total_values.append(total)
        shares_list.append(shares)
        cash_list.append(cash)
        cumdiv_list.append(cum_div)

    out = df.copy()
    out["Shares_Held"] = shares_list
    out["Cash_Balance"] = cash_list
    out["Dividend_Income"] = cumdiv_list
    out["Total_Value"] = total_values
    return out


# -------------------------
# Plot
# -------------------------
def plot_portfolio(df: pd.DataFrame, *, initial_investment: float, title: str):
    if df is None or df.empty or "Total_Value" not in df.columns:
        raise ValueError("Dataframe must include 'Total_Value'.")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Total_Value"], label="Total Portfolio Value (Price + Cash Dividends)")
    ax.axhline(initial_investment, linestyle="--", label=f"Initial ${initial_investment:,.0f}")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (USD)")
    ax.grid(True, alpha=0.3)

    # dividends on secondary axis
    if "Dividend_Income" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df.index, df["Dividend_Income"], alpha=0.8, label="Cumulative Dividends")
        ax2.set_ylabel("Dividend Cash (USD)")

        # combined legend
        lines = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left")
    else:
        ax.legend(loc="upper left")

    fig.tight_layout()
    plt.show()


# -------------------------
# Main
# -------------------------
def main():
    data_path = Path("stock_data/AAPL_O_history.csv")
    df = load_history_csv(data_path)

    results = simulate_single_stock_portfolio(
        df,
        initial_investment=1_000_000,
        price_col="UNADJ_CLOSE",
        dividend_col="Dividend",
        split_col="SplitFactor",
        cum_factor_col="cum_factor",
        use_cum_factor=True,       # will use cum_factor if present
        auto_align_splits=True,    # fixes common “one-day off” stamping
        reinvest_dividends=False,
        verbose=True,
    )

    plot_portfolio(results, initial_investment=1_000_000, title="AAPL.O Portfolio (Unadjusted)")


if __name__ == "__main__":
    main()

