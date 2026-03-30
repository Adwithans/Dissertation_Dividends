from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv

from ..logging_utils import get_logger

try:
    import eikon as ek
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    ek = None

try:
    import refinitiv.data as rd
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    rd = None


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    backoff_seconds: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 30.0, 45.0)


class RefinitivClient:
    """Thin wrapper over tested rd/eikon calls with session lifecycle and retries."""

    def __init__(self, retry_policy: RetryPolicy | None = None):
        if retry_policy is None:
            max_attempts = int(os.getenv("EIKON_MAX_RETRIES", "8"))
            retry_policy = RetryPolicy(max_attempts=max_attempts)
        self.retry_policy = retry_policy
        self._session_open = False
        self.logger = get_logger("dividend_portfolio.refinitiv")
        self._last_data_call_ts = 0.0
        self._min_interval_seconds = float(os.getenv("EIKON_MIN_INTERVAL_SECONDS", "0.5"))

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
                wait = self._retry_wait_seconds(exc, i, backoffs)
                if self._is_rate_limited(exc):
                    self._min_interval_seconds = max(
                        self._min_interval_seconds,
                        min(3.0, wait / 3.0),
                    )
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
    def _retry_wait_seconds(exc: Exception, attempt_idx: int, backoffs: tuple[float, ...]) -> float:
        msg = str(exc)
        if "429" in msg or "Too many requests" in msg:
            rate_backoff = (5.0, 10.0, 20.0, 40.0, 60.0, 90.0)
            return rate_backoff[min(attempt_idx, len(rate_backoff) - 1)]
        return backoffs[min(attempt_idx, len(backoffs) - 1)]

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        msg = str(exc)
        return "429" in msg or "Too many requests" in msg

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

    def _pace_data_calls(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_data_call_ts
        wait = self._min_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_data_call_ts = time.monotonic()

    def open(self) -> None:
        if self._session_open:
            return
        if rd is None:
            raise ModuleNotFoundError(
                "refinitiv.data is not installed. Install project dependencies before using RefinitivClient."
            )

        load_dotenv()
        app_key = os.getenv("EIKON_API_KEY")
        if app_key and ek is not None:
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

        if rd is None:
            raise ModuleNotFoundError(
                "refinitiv.data is not installed. Install project dependencies before using RefinitivClient.get_history."
            )
        return self._call_with_retry(rd.get_history, "rd.get_history", **kwargs)

    def get_data(self, universe: str | Iterable[str], fields: list[str], parameters: dict[str, Any]) -> pd.DataFrame:
        if rd is None:
            raise ModuleNotFoundError(
                "refinitiv.data is not installed. Install project dependencies before using RefinitivClient.get_data."
            )
        self._pace_data_calls()
        data = self._call_with_retry(rd.get_data, "rd.get_data", universe, fields, parameters)
        return self._rd_df(data)

    def get_eikon_data(
        self, universe: str, fields: list[str], parameters: dict[str, Any]
    ) -> tuple[pd.DataFrame, Any]:
        if ek is None:
            raise ModuleNotFoundError(
                "eikon is not installed. Install project dependencies before using RefinitivClient.get_eikon_data."
            )
        self._pace_data_calls()
        return self._call_with_retry(ek.get_data, "ek.get_data", universe, fields, parameters)
