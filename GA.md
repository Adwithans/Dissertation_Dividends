Run this from the repo root with your activated virtualenv.

This version is the full intensive sweep:
- portfolio sizes `7..35`
- rebalance intervals `1, 2, 3, 4` quarters
- all allocation strategies
- every combination is evaluated exactly once
- S&P-aware search objective
  - maximize `cagr`
  - maximize `annualized_excess_return`
  - maximize `strategy_up_benchmark_down_periods`
  - minimize `strategy_down_benchmark_up_periods`
  - minimize drawdown

```bash
python -m src.dividend_portfolio.cli.run_genetic_search \
  --config config/portfolio.yaml \
  --benchmark sp500 \
  --persist-trials none \
  --max-workers 2
```

What gets written:
- `study_summary.json`
  - best by return
  - best by drawdown
  - Pareto front
  - study history
- full winner summaries
  - after the search finishes, the workflow automatically reruns:
    - `best_by_return`
    - `best_by_drawdown`
  - and generates full `showresults` outputs for each under:
    - `data/runs_dynamic/genetic_search/<study_id>/winner_summaries/`
- `trial_results.csv`
  - one row per unique evaluated hyperparameter set
  - includes portfolio size, rebalance interval, allocation strategy, status, error, the optimizer-facing metrics, and the S&P quadrant metrics
- `trial_results.json`
  - same data as JSON
- `population_history.csv`
  - one row per population member per generation
- `generation_history.csv`
  - headline stats per generation

Default study output folder:
- `data/runs_dynamic/genetic_search/<study_id>/`
- benchmark close cache:
  - `data/store/benchmark_cache.sqlite`

Notes:
- This uses whatever `python` is active in your shell.
- On macOS, use the module command above rather than a heredoc (`python - <<'PY'`) when `max_workers > 1`. The multiprocessing `spawn` context requires a real module/script entrypoint.
- Because `population_size=space.combination_count()` and `generations=1`, this is effectively a full grid search over the current discrete hyperparameter space while still using the same GA entrypoint.
- The search objective is benchmark-aware in this run, so `trial_results.csv` will include `search_cagr`, `search_annualized_excess_return`, `search_strategy_up_benchmark_down_periods`, `search_inverse_strategy_down_benchmark_up_periods`, and `search_max_drawdown`.
- Benchmark `.SPX` and `.RUI` close history is prefetched once per study and persisted in `data/store/benchmark_cache.sqlite`, so reruns do not keep consuming extra Refinitiv calls for those series.
- `max_workers=2` enables multiprocessing across independent evaluations. This path is ready for process-based parallelism. Threads are still not recommended because Refinitiv client/session state and SQLite-backed caches are safer when isolated per process. If Refinitiv is stable and the cache is warm, you can try `4`.
- If you want a search-only run without the post-search full winner summaries, add:
  - `--no-winner-full-summaries`
