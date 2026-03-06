from __future__ import annotations

import pandas as pd

from src.dividend_portfolio.sim.split_math import build_split_multiplier, effective_share_multiplier


def test_effective_share_multiplier_inverts_adjustment_factor() -> None:
    assert effective_share_multiplier(0.25) == 4.0
    assert effective_share_multiplier(4.0) == 4.0
    assert effective_share_multiplier(-1.0) == 1.0


def test_build_split_multiplier_auto_aligns_one_day_shift() -> None:
    idx = pd.to_datetime(["2020-08-27", "2020-08-28", "2020-08-31", "2020-09-01"])
    df = pd.DataFrame(
        {
            "CLOSE": [500.0, 499.0, 125.0, 126.0],
            "SplitFactor": [1.0, 0.25, 1.0, 1.0],
        },
        index=idx,
    )

    split = build_split_multiplier(
        df,
        price_col="CLOSE",
        split_col="SplitFactor",
        use_cum_factor=False,
        auto_align=True,
    )

    assert split.loc[pd.Timestamp("2020-08-31")] == 4.0
