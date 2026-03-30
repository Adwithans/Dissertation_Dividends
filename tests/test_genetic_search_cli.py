from __future__ import annotations

from src.dividend_portfolio.cli.run_genetic_search import parse_args


def test_run_genetic_search_parse_args_accepts_parallel_options(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_genetic_search.py",
            "--config",
            "config/portfolio.yaml",
            "--benchmark",
            "sp500",
            "--max-workers",
            "2",
            "--persist-trials",
            "none",
        ],
    )
    args = parse_args()
    assert args.config == "config/portfolio.yaml"
    assert args.benchmark == "sp500"
    assert args.max_workers == 2
    assert args.persist_trials == "none"
    assert args.no_winner_full_summaries is False
