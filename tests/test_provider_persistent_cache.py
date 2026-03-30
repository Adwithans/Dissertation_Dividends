from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.dividend_portfolio.strategy.provider import RefinitivStrategyDataProvider


@dataclass
class FakeClient:
    history_calls: int = 0
    _prices_by_ric: dict[str, list[float]] = field(
        default_factory=lambda: {"A": [100.0, 101.0], "B": [50.0, 51.0]}
    )

    def get_data(self, universe, fields, parameters):  # noqa: ANN001
        _ = (universe, fields, parameters)
        return pd.DataFrame()

    def get_eikon_data(self, universe, fields, parameters):  # noqa: ANN001
        _ = (universe, fields, parameters)
        return pd.DataFrame(), None

    def get_history(self, *, universe, fields, interval, start, end, adjustments=None):  # noqa: ANN001
        _ = (interval, start, end, adjustments)
        self.history_calls += 1
        ric = str(list(universe)[0])
        dates = pd.bdate_range("2024-01-02", periods=2)
        if fields == ["TRDPRC_1"]:
            return pd.DataFrame({"Date": dates, "TRDPRC_1": self._prices_by_ric.get(ric, [1.0, 1.0])})
        if fields == ["BID", "ASK"]:
            px = self._prices_by_ric.get(ric, [1.0, 1.0])
            return pd.DataFrame({"Date": dates, "BID": [p - 0.05 for p in px], "ASK": [p + 0.05 for p in px]})
        return pd.DataFrame({"Date": dates})


@dataclass
class FakeDividendClient:
    data_calls: list[tuple[tuple[str, ...], str, str]] = field(default_factory=list)
    eikon_calls: list[tuple[tuple[str, ...], str, str]] = field(default_factory=list)

    def get_data(self, universe, fields, parameters):  # noqa: ANN001
        _ = fields
        rics = tuple(sorted(str(r) for r in universe))
        start = str(parameters["SDate"])
        end = str(parameters["EDate"])
        self.data_calls.append((rics, start, end))
        rows = [
            {
                "Instrument": ric,
                "Dividend Ex Date": end,
                "Gross Dividend Amount": 0.5,
            }
            for ric in rics
        ]
        return pd.DataFrame(rows)

    def get_history(self, *, universe, fields, interval, start, end, adjustments=None):  # noqa: ANN001
        _ = (universe, fields, interval, start, end, adjustments)
        return pd.DataFrame()

    def get_eikon_data(self, universe, fields, parameters):  # noqa: ANN001
        _ = fields
        rics = tuple(sorted(str(r) for r in universe))
        start = str(parameters["SDate"])
        end = str(parameters["EDate"])
        self.eikon_calls.append((rics, start, end))
        rows = [
            {
                "Instrument": ric,
                "Dividend Ex Date": end,
                "Gross Dividend Amount": 0.5,
            }
            for ric in rics
        ]
        return pd.DataFrame(rows), None


def test_provider_uses_persistent_sqlite_cache_for_price_history(tmp_path) -> None:
    db_path = tmp_path / "provider_cache.sqlite"
    client = FakeClient()
    provider = RefinitivStrategyDataProvider(
        client=client,
        batch_size=1,
        enable_cache=False,
        persistent_cache_db_path=db_path,
        persistent_cache_enabled=True,
    )
    first = provider.get_close_history(["A", "B"], "2024-01-02", "2024-01-05")
    first_calls = client.history_calls
    provider.close()

    provider2 = RefinitivStrategyDataProvider(
        client=client,
        batch_size=1,
        enable_cache=False,
        persistent_cache_db_path=db_path,
        persistent_cache_enabled=True,
    )
    second = provider2.get_close_history(["A", "B"], "2024-01-02", "2024-01-05")
    second_calls = client.history_calls
    provider2.close()

    assert first_calls > 0
    assert second_calls == first_calls
    assert len(first) == len(second)
    assert set(first["RIC"]) == {"A", "B"}


def test_dividend_fetch_uses_incremental_right_tail_window(tmp_path) -> None:
    db_path = tmp_path / "provider_cache.sqlite"
    client = FakeDividendClient()
    provider = RefinitivStrategyDataProvider(
        client=client,
        batch_size=10,
        enable_cache=False,
        persistent_cache_db_path=db_path,
        persistent_cache_enabled=True,
    )

    provider.get_dividend_events(["A", "B"], "2024-01-01", "2024-03-31")
    provider.get_dividend_events(["A", "B"], "2024-02-01", "2024-06-30")
    provider.close()

    calls = client.data_calls or client.eikon_calls
    assert len(calls) == 2
    assert calls[0][1:] == ("2024-01-01", "2024-03-31")
    assert calls[1][1:] == ("2024-04-01", "2024-06-30")
