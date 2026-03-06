from __future__ import annotations

import pandas as pd

from src.dividend_portfolio.io.history_io import load_histories, save_history_csv


def _make_history(start: str, periods: int, close_start: float) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "CLOSE": [close_start + i for i in range(periods)],
            "Dividend": [0.0] * periods,
            "SplitFactor": [1.0] * periods,
            "CumulativeDividend": [0.0] * periods,
            "cum_factor": [1.0] * periods,
        },
        index=idx,
    )


def test_load_histories_uses_fallback_when_primary_too_short(tmp_path) -> None:
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    ric = "AAPL.O"
    filename = "AAPL_O_history.csv"

    save_history_csv(_make_history("2026-02-19", periods=2, close_start=100.0), primary / filename)
    save_history_csv(_make_history("2011-02-16", periods=20, close_start=10.0), fallback / filename)

    out = load_histories([ric], primary_dir=primary, fallback_dir=fallback, min_primary_rows=5)
    df = out[ric]

    assert len(df) == 20
    assert df.index.min() == pd.Timestamp("2011-02-16")


def test_load_histories_prefers_primary_when_it_is_sufficient(tmp_path) -> None:
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    ric = "AAPL.O"
    filename = "AAPL_O_history.csv"

    save_history_csv(_make_history("2011-02-16", periods=20, close_start=10.0), primary / filename)
    save_history_csv(_make_history("2011-02-16", periods=10, close_start=100.0), fallback / filename)

    out = load_histories([ric], primary_dir=primary, fallback_dir=fallback, min_primary_rows=5)
    df = out[ric]

    assert len(df) == 20
    assert df.iloc[0]["CLOSE"] == 10.0
