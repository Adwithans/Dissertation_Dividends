from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _refresh_latest_alias(run_dir: Path) -> None:
    latest = run_dir.parent / "latest"
    if latest.is_symlink() or latest.is_file():
        latest.unlink()
    elif latest.exists():
        shutil.rmtree(latest)

    try:
        latest.symlink_to(run_dir.name, target_is_directory=True)
    except OSError:
        latest.mkdir(parents=True, exist_ok=True)
        (latest / "LATEST_RUN.txt").write_text(f"{run_dir}\n", encoding="utf-8")


def create_run_dir(base_dir: str | Path = "data/runs") -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / ts
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        _refresh_latest_alias(run_dir)
        return run_dir

    for i in range(1, 1000):
        candidate = base / f"{ts}_{i:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            _refresh_latest_alias(candidate)
            return candidate

    raise RuntimeError("Unable to create unique run directory after 1000 attempts.")


def write_run_outputs(
    run_dir: str | Path,
    portfolio_df: pd.DataFrame,
    asset_results: dict[str, pd.DataFrame],
    metrics: dict[str, Any],
    attribution_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    rebalance_log: pd.DataFrame,
) -> Path:
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    portfolio_df.to_csv(out_dir / "portfolio_timeseries.csv", index=True)

    for ric, df in asset_results.items():
        safe = ric.replace(".", "_")
        df.to_csv(out_dir / f"asset_{safe}_timeseries.csv", index=True)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    attribution_df.to_csv(out_dir / "asset_attribution.csv", index=False)
    quarterly_df.to_csv(out_dir / "quarterly_stock_metrics.csv", index=False)
    rebalance_log.to_csv(out_dir / "rebalance_log.csv", index=False)

    return out_dir
