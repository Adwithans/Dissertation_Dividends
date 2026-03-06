from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import eikon as ek
import pandas as pd
import refinitiv.data as rd
from dotenv import load_dotenv

from ..logging_utils import get_logger


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0)


class RefinitivClient:
    """Thin wrapper over tested rd/eikon calls with session lifecycle and retries."""

    def __init__(self, retry_policy: RetryPolicy | None = None):
        self.retry_policy = retry_policy or RetryPolicy()
        self._session_open = False
        self.logger = get_logger("dividend_portfolio.refinitiv")

    @staticmethod
    def _rd_df(data: Any) -> pd.DataFrame:
        if isinstance(data, tuple):
            return data[0]
        return data

    def _call_with_retry(self, fn, description: str, *args, **kwargs):
        last_exc: Exception | None = None
        attempts = self.retry_policy.max_attempts
        backoffs = self.retry_policy.backoff_seconds

        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not self._is_retryable(exc):
                    raise
                if i >= attempts - 1:
                    break
                wait = backoffs[min(i, len(backoffs) - 1)]
                self.logger.warning(
                    "%s failed on attempt %s/%s: %s. Retrying in %.1fs",
                    description,
                    i + 1,
                    attempts,
                    exc,
                    wait,
                )
                time.sleep(wait)

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Only retry transient transport/session errors, not semantic request errors."""
        msg = str(exc)
        non_retry_markers = (
            "UserRequestError",
            "No data to return",
            "The universe does not support",
            "invalid argument",
            "invalid field",
        )
        return not any(marker in msg for marker in non_retry_markers)

    def open(self) -> None:
        if self._session_open:
            return

        load_dotenv()
        app_key = os.getenv("EIKON_API_KEY")
        if app_key:
            ek.set_app_key(app_key)

        self._call_with_retry(rd.open_session, "rd.open_session")
        self._session_open = True

    def close(self) -> None:
        if not self._session_open:
            return
        try:
            rd.close_session()
        finally:
            self._session_open = False

    def __enter__(self) -> "RefinitivClient":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_history(
        self,
        *,
        universe: Iterable[str],
        fields: list[str],
        interval: str,
        start: str,
        end: str,
        adjustments: str | None = None,
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = dict(
            universe=list(universe),
            fields=fields,
            interval=interval,
            start=start,
            end=end,
        )
        if adjustments is not None:
            kwargs["adjustments"] = adjustments

        return self._call_with_retry(rd.get_history, "rd.get_history", **kwargs)

    def get_data(self, universe: str, fields: list[str], parameters: dict[str, Any]) -> pd.DataFrame:
        data = self._call_with_retry(rd.get_data, "rd.get_data", universe, fields, parameters)
        return self._rd_df(data)

    def get_eikon_data(
        self, universe: str, fields: list[str], parameters: dict[str, Any]
    ) -> tuple[pd.DataFrame, Any]:
        return self._call_with_retry(ek.get_data, "ek.get_data", universe, fields, parameters)
