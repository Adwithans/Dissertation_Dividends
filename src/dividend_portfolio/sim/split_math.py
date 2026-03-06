from __future__ import annotations

import numpy as np
import pandas as pd


def effective_share_multiplier(raw_factor: float, snap_tol: float = 0.02) -> float:
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
    """Return a daily share multiplier series for split processing."""
    out = df.copy()

    if use_cum_factor and (cum_factor_col in out.columns):
        cf = pd.to_numeric(out[cum_factor_col], errors="coerce").ffill().fillna(1.0)
        raw = (cf / cf.shift(1)).fillna(1.0)
    elif split_col in out.columns:
        raw = pd.to_numeric(out[split_col], errors="coerce").fillna(1.0)
    else:
        raw = pd.Series(1.0, index=out.index)

    split_mult = raw.apply(effective_share_multiplier).fillna(1.0)

    if auto_align and price_col in out.columns:
        price = pd.to_numeric(out[price_col], errors="coerce")
        expected = (price.shift(1) / price).replace([np.inf, -np.inf], np.nan)

        mask = split_mult.ne(1.0) & expected.notna()
        if mask.any():
            err_same = np.nanmedian(np.abs(np.log(split_mult[mask] / expected[mask])))
            shifted = split_mult.shift(1)
            err_shift1 = np.nanmedian(np.abs(np.log(shifted[mask] / expected[mask])))
            if np.isfinite(err_shift1) and err_shift1 < err_same:
                split_mult = shifted.fillna(1.0)

    return split_mult
