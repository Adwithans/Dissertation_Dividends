# Multi-Stock USD Portfolio Architecture Plan (with Rebalancing + Quarterly Dividend Metrics)

## 1. Scope and Non-Negotiables

1. Portfolio is USD-only.
2. Use only API calls already proven in this repo.
3. Use unadjusted prices for simulation to avoid double-adjustment errors.
4. Support multi-stock backtests, scheduled rebalancing, and per-stock quarterly dividend return metrics.
5. Preserve current split/dividend mechanics from `Portfolio.py`.

## 2. Existing Baseline to Preserve

### 2.1 Proven data APIs

- Prices (primary):
  - `rd.get_history(universe=[ric], fields=["OPEN","HIGH","LOW","CLOSE"], interval="daily", start=..., end=..., adjustments="unadjusted")`
- Prices (fallback):
  - `rd.get_history(universe=[ric], fields=["TRDPRC_1"], interval="daily", start=..., end=..., adjustments="unadjusted")`
  - rename `TRDPRC_1 -> CLOSE`
- Dividends:
  - `ek.get_data(ric, ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivCurr"], {"SDate":..., "EDate":..., "DateType":"ED"})`
- Splits:
  - `rd.get_data(ric, ["TR.CAExDate", "TR.CAEffectiveDate", "TR.CACorpActDate", "TR.CAAdjustmentFactor"], {"CAEventType":"SSP", "SDate":..., "EDate":...})`

### 2.2 Proven simulation logic

- Split conversion and alignment logic from `Portfolio.py`:
  - `_effective_share_multiplier(...)`
  - `build_split_multiplier(..., auto_align=True)`
- Daily event order (must remain exact):
  1. apply split multiplier to shares
  2. process dividend cash from current shares
  3. optionally reinvest dividend at same-day close
  4. compute market value and total value

## 3. Target Folder Structure

```text
Dissertation_Dividends/
  config/
    portfolio.yaml
  data/
    raw/
      history/
    runs/
  src/
    dividend_portfolio/
      __init__.py
      config.py
      models.py
      logging_utils.py
      io/
        history_io.py
        run_io.py
      data/
        refinitiv_client.py
        fetch_prices.py
        fetch_events.py
        history_builder.py
      sim/
        split_math.py
        single_asset.py
        rebalancer.py
        multi_asset.py
      analytics/
        metrics.py
        attribution.py
        quarterly.py
      reporting/
        plots.py
        tables.py
      cli/
        fetch_histories.py
        run_backtest.py
        run_pipeline.py
  tests/
    fixtures/
    test_split_math.py
    test_single_asset.py
    test_rebalancer.py
    test_multi_asset.py
    test_quarterly_metrics.py
  Portfolio.py
  GetData.py
```

## 4. Configuration Contract

## 4.1 `config/portfolio.yaml` schema

Required top-level keys:

- `base_currency` (must be `USD`)
- `initial_capital` (float > 0)
- `start_date` (ISO date)
- `end_date` (ISO date or null)
- `reinvest_dividends` (bool)
- `auto_align_splits` (bool)
- `use_cum_factor` (bool)
- `risk_free_rate` (annual decimal)
- `rebalancing` (object)
- `quarterly_metrics` (object)
- `assets` (list of `{ric, weight}`)

### 4.2 Rebalancing config

- `rebalancing.enabled`: bool
- `rebalancing.frequency`: `quarterly | monthly | yearly`
- `rebalancing.trigger`: `first_trading_day_after_quarter_end`
- `rebalancing.drift_tolerance`: absolute weight deviation threshold (e.g., `0.02`)

Default production mode for your use case:

- `enabled: true`
- `frequency: quarterly`
- `trigger: first_trading_day_after_quarter_end`
- `drift_tolerance: 0.02`

### 4.3 Quarterly metrics config

- `quarterly_metrics.enabled`: bool
- `quarterly_metrics.dividend_return_basis`: `quarter_start_market_value`

Definition is fixed for now (no ambiguity):

`stock_quarterly_dividend_return = dividends_received_in_quarter / market_value_at_quarter_start`

## 5. Data Model and Output Schema

## 5.1 Raw per-asset history CSV

Canonical columns:

- `Date`
- `CLOSE`
- `Dividend`
- `SplitFactor`
- `CumulativeDividend`
- `cum_factor`

## 5.2 Per-asset simulation timeseries columns

- `Shares_Held`
- `Cash_Balance`
- `Dividend_Cash_Daily`
- `Dividend_Income` (cumulative)
- `Market_Value`
- `Total_Value`
- `Weight_EOD` (relative to total portfolio)
- `Target_Weight` (from config)
- `Rebalance_Trade_Shares` (0 on non-rebalance days)
- `Rebalance_Trade_Value`

## 5.3 Portfolio-level timeseries columns

- `Portfolio_Market_Value`
- `Portfolio_Cash_Balance`
- `Portfolio_Total_Value`
- `Portfolio_Dividend_Cash_Daily`
- `Portfolio_Dividend_Income`
- `Portfolio_Daily_Return`
- `Portfolio_Cumulative_Return`
- `Rebalance_Flag`

## 6. Rebalancing Engine (Exact Behavior)

## 6.1 Rebalance calendar

For `frequency=quarterly` and `trigger=first_trading_day_after_quarter_end`:

1. Compute quarter-end dates (`Mar 31`, `Jun 30`, `Sep 30`, `Dec 31`) within backtest period.
2. For each quarter-end date, find first trading date present in portfolio master calendar on/after that date.
3. That date is a rebalance date.

## 6.2 Rebalance mechanics on a rebalance date

Given EOD pre-trade state (after split/dividend handling):

1. Compute pre-trade portfolio total value:
   - `V = sum_i(Market_Value_i + Cash_i_allocated_component)`
2. Target value per stock:
   - `TargetValue_i = V * target_weight_i`
3. Current value per stock:
   - `CurrentValue_i = Market_Value_i` (cash kept at portfolio level)
4. Drift check:
   - `drift_i = abs(CurrentWeight_i - target_weight_i)`
   - If all `drift_i < drift_tolerance`, skip rebalance.
5. Trades (fractional-share mode to match your existing math):
   - `TradeValue_i = TargetValue_i - CurrentValue_i`
   - `TradeShares_i = TradeValue_i / CLOSE_i`
6. Apply trades instantly at that day close.
7. Ensure post-trade:
   - `sum_i(Market_Value_i_post) + PortfolioCash_post == V`
   - weights match targets within numerical tolerance.

## 6.3 Cost model

Initial version: zero transaction costs and zero slippage (explicitly documented).

Extension point:

- add `commission_bps` and `slippage_bps` later in `rebalancer.py`.

## 7. Quarterly Stock Metrics (Exact Definitions)

For each stock and quarter:

1. `Quarter_Start_Date`: first trading day in that quarter for that stock.
2. `Quarter_End_Date`: last trading day in that quarter for that stock.
3. `Quarter_Start_Value`: `Shares_Held * CLOSE + per-asset cash allocation` at start date.
4. `Quarter_End_Value`: same at end date.
5. `Quarter_Dividend_Cash`: sum of `Dividend_Cash_Daily` in that quarter.
6. `Quarter_Price_PnL`: `(Quarter_End_Value - Quarter_Start_Value) - Quarter_Dividend_Cash`.
7. `Quarter_Total_Return`:
   - `(Quarter_End_Value - Quarter_Start_Value) / Quarter_Start_Value`.
8. `Quarter_Dividend_Return`:
   - `Quarter_Dividend_Cash / Quarter_Start_Value`.
9. `Quarter_Dividend_Contribution_to_Portfolio`:
   - `Quarter_Dividend_Cash / Portfolio_Quarter_Start_Value`.

Output table file:

- `data/runs/<timestamp>/quarterly_stock_metrics.csv`

Mandatory columns:

- `RIC`
- `Quarter` (e.g., `2024Q3`)
- `Quarter_Start_Date`
- `Quarter_End_Date`
- `Quarter_Start_Value`
- `Quarter_End_Value`
- `Quarter_Dividend_Cash`
- `Quarter_Dividend_Return`
- `Quarter_Total_Return`
- `Quarter_Price_PnL`
- `Quarter_Dividend_Contribution_to_Portfolio`

## 8. Module-by-Module Build Plan

1. `data/refinitiv_client.py`
- session lifecycle and retry wrappers for existing API calls only.

2. `data/fetch_prices.py`
- port proven unadjusted price logic exactly.

3. `data/fetch_events.py`
- port proven dividend/split event extraction exactly.

4. `data/history_builder.py`
- build canonical history schema and write to `data/raw/history`.

5. `sim/split_math.py`
- move split conversion/alignment logic from current `Portfolio.py` with no math changes.

6. `sim/single_asset.py`
- implement per-asset daily simulation using canonical columns.

7. `sim/rebalancer.py`
- deterministic schedule builder + rebalance trade calculator + application.

8. `sim/multi_asset.py`
- orchestrate all assets on master calendar, apply rebalances, aggregate portfolio.

9. `analytics/quarterly.py`
- compute per-stock quarterly metrics and emit CSV.

10. `analytics/metrics.py` and `analytics/attribution.py`
- portfolio summary metrics and stock-level attribution.

11. `cli/fetch_histories.py`, `cli/run_backtest.py`, `cli/run_pipeline.py`
- operational entrypoints.

12. `Portfolio.py` and `GetData.py`
- compatibility wrappers to new CLI functions.

## 9. Testing Plan

`test_split_math.py`
- validate factor inversion and auto-align behavior against known split dates.

`test_single_asset.py`
- verify event order and reinvest/non-reinvest paths.

`test_rebalancer.py`
- verify quarterly schedule generation and trade math.
- verify no-rebalance when drift below tolerance.

`test_multi_asset.py`
- verify aggregation consistency:
  - `sum(asset_total) == portfolio_total`
  - post-rebalance weights match targets.

`test_quarterly_metrics.py`
- verify quarterly dividend return formulas and output columns.

## 10. Runtime Artifacts

Each run writes to `data/runs/<timestamp>/`:

- `portfolio_timeseries.csv`
- `asset_<RIC>_timeseries.csv` for each asset
- `metrics.json`
- `asset_attribution.csv`
- `quarterly_stock_metrics.csv`
- `rebalance_log.csv`

## 11. Acceptance Criteria

1. Single-stock mode reproduces current `Portfolio.py` result profile.
2. Multi-stock mode outputs stable portfolio totals and per-asset states.
3. Quarterly rebalancing runs on deterministic trading dates and logs trades.
4. Quarterly per-stock dividend return metric is present for every quarter-stock pair.
5. No FX logic exists in config, code, or outputs.
6. All prices used in simulation come from unadjusted series.
