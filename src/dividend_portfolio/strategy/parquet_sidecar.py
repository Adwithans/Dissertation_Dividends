from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..logging_utils import get_logger


class ParquetSidecarWriter:
    def __init__(self, base_dir: str | Path, enabled: bool = True):
        self.base_dir = Path(base_dir)
        self.enabled = enabled
        self.logger = get_logger("dividend_portfolio.strategy.parquet")
        self._disabled_by_runtime = False
        if enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, df: pd.DataFrame, dataset: str, quarter: str, date_col: str | None = None) -> None:
        if not self.enabled or self._disabled_by_runtime or df.empty:
            return

        if date_col and date_col in df.columns:
            date_vals = pd.to_datetime(df[date_col], errors="coerce")
            year = str(int(date_vals.dropna().dt.year.mode().iloc[0])) if not date_vals.dropna().empty else "unknown"
        else:
            year = "unknown"

        out_dir = self.base_dir / dataset / f"year={year}" / f"quarter={quarter}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "data.parquet"

        try:
            df.to_parquet(out_path, index=False)
        except Exception as exc:  # noqa: BLE001
            # Runtime environments without pyarrow/fastparquet should not stop the pipeline.
            self._disabled_by_runtime = True
            self.logger.warning(
                "Parquet sidecar disabled because write failed (%s). "
                "Install pyarrow to enable parquet outputs.",
                exc,
            )

    def write_constituents(self, df: pd.DataFrame, quarter: str) -> None:
        self._write(df, "constituents", quarter, date_col="as_of_date")

    def write_market_caps(self, df: pd.DataFrame, quarter: str) -> None:
        self._write(df, "market_caps", quarter, date_col="MarketCapDate")

    def write_prices(self, df: pd.DataFrame, quarter: str) -> None:
        self._write(df, "prices", quarter, date_col="Date")

    def write_dividends(self, df: pd.DataFrame, quarter: str) -> None:
        self._write(df, "dividends", quarter, date_col="Date")
