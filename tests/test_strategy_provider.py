from __future__ import annotations

import pandas as pd

from src.dividend_portfolio.strategy.provider import RefinitivStrategyDataProvider


class FakeRefinitivClient:
    def get_data(self, universe, fields, parameters):
        return pd.DataFrame()

    def get_eikon_data(self, universe, fields, parameters):
        rics = [str(ric) for ric in universe]
        field0 = str(fields[0])
        if "CompanyMarketCap" in field0:
            return pd.DataFrame(), None
        if "CompanySharesOutstanding" in field0:
            df = pd.DataFrame(
                {
                    "Instrument": rics,
                    "Company Shares Outstanding": [10.0, 5.0][: len(rics)],
                    "Company Shares Outstanding Date": [parameters["SDate"]] * len(rics),
                }
            )
            return df, None
        return pd.DataFrame(), None

    def get_history(self, *, universe, fields, interval, start, end, adjustments=None):
        idx = pd.to_datetime([start])
        return pd.DataFrame({str(ric): [price] for ric, price in zip(universe, [20.0, 8.0], strict=False)}, index=idx)


def test_market_cap_snapshot_falls_back_to_close_times_shares() -> None:
    provider = RefinitivStrategyDataProvider(
        client=FakeRefinitivClient(),
        batch_size=50,
        enable_cache=False,
        persistent_cache_enabled=False,
    )

    out = provider.get_market_cap_snapshot(["A", "B"], "2011-02-16")

    assert list(out["RIC"]) == ["A", "B"]
    assert list(out["MarketCap"]) == [200.0, 40.0]
    assert out["MarketCapDate"].notna().all()
