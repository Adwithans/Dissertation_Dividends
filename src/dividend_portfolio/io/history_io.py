from __future__ import annotations

from pathlib import Path

import pandas as pd


def ric_to_filename(ric: str) -> str:
    return f"{ric.replace('.', '_')}_history.csv"


def load_history_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.loc[df.index.notna()].sort_index()
    return df


def save_history_csv(df: pd.DataFrame, path: str | Path) -> None:
    out = df.copy().sort_index()
    out.index.name = "Date"
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(path_obj, float_format="%.6f")


def _prefer_fallback(
    primary_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    *,
    min_primary_rows: int,
) -> bool:
    if primary_df.empty and not fallback_df.empty:
        return True

    if len(primary_df) < min_primary_rows and len(fallback_df) > len(primary_df):
        return True

    p_start = primary_df.index.min() if not primary_df.empty else pd.NaT
    f_start = fallback_df.index.min() if not fallback_df.empty else pd.NaT

    if pd.notna(p_start) and pd.notna(f_start) and p_start > f_start:
        return True

    return False


def load_histories(
    rics: list[str],
    primary_dir: str | Path,
    fallback_dir: str | Path | None = "stock_data",
    min_primary_rows: int = 30,
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    primary = Path(primary_dir)
    fallback = Path(fallback_dir) if fallback_dir is not None else None

    for ric in rics:
        filename = ric_to_filename(ric)
        p1 = primary / filename
        p2 = fallback / filename if fallback is not None else None

        p1_exists = p1.exists()
        p2_exists = p2 is not None and p2.exists()

        if not p1_exists and not p2_exists:
            raise FileNotFoundError(
                f"History CSV missing for {ric}. Tried: {p1}"
                + (f" and {p2}" if p2 is not None else "")
            )

        df_primary = load_history_csv(p1) if p1_exists else None
        df_fallback = load_history_csv(p2) if p2_exists and p2 is not None else None

        if df_primary is not None and df_fallback is not None:
            if _prefer_fallback(df_primary, df_fallback, min_primary_rows=min_primary_rows):
                print(
                    f"[WARN] Using fallback history for {ric}: {p2} "
                    f"(primary {p1} had {len(df_primary)} rows)"
                )
                histories[ric] = df_fallback
            else:
                histories[ric] = df_primary
        elif df_primary is not None:
            histories[ric] = df_primary
        elif df_fallback is not None:
            histories[ric] = df_fallback
        else:
            raise FileNotFoundError(f"Unable to load history for {ric}.")

    return histories
