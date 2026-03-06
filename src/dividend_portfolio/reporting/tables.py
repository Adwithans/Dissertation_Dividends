from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_metrics_json(metrics: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def save_dataframe_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index)
