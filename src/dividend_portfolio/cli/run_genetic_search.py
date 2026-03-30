from __future__ import annotations

import argparse
from datetime import datetime, timezone

from ..config import load_portfolio_config
from ..optimization import GeneticAlgorithmConfig, GeneticSearchSpace, run_genetic_algorithm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dividend portfolio hyperparameter search with the built-in optimizer."
    )
    parser.add_argument("--config", default="config/portfolio.yaml", help="Path to portfolio config")
    parser.add_argument(
        "--benchmark",
        default="sp500",
        choices=("sp500", "none"),
        help="Benchmark mode used for evaluation and search objectives.",
    )
    parser.add_argument(
        "--persist-trials",
        default="none",
        choices=("none", "summary"),
        help="Per-trial persistence mode.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Process worker count for parallel trial evaluation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for candidate ordering and GA operations.",
    )
    parser.add_argument(
        "--study-id",
        default=None,
        help="Optional study id override. Default is a UTC timestamp-based id.",
    )
    parser.add_argument(
        "--benchmark-cache-db-path",
        default=None,
        help="Optional SQLite path for cached benchmark close history.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Population size override. Default is exhaustive evaluation of all combinations.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=1,
        help="Number of optimizer generations to run.",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=0,
        help="Elite individuals to carry forward each generation.",
    )
    parser.add_argument(
        "--no-winner-full-summaries",
        action="store_true",
        help="Skip rerunning best-by-return and best-by-drawdown as full showresults workflows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_portfolio_config(args.config)
    space = GeneticSearchSpace.intensive_defaults()
    study_id = args.study_id or f"ga_exhaustive_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    population_size = args.population_size if args.population_size is not None else space.combination_count()

    result = run_genetic_algorithm(
        base_config=cfg,
        search_space=space,
        config=GeneticAlgorithmConfig(
            population_size=population_size,
            generations=int(args.generations),
            elite_count=int(args.elite_count),
            random_seed=int(args.random_seed),
            persist_trials=args.persist_trials,
            benchmark=args.benchmark,
            max_workers=int(args.max_workers),
            study_id=study_id,
            generate_winner_full_summaries=not args.no_winner_full_summaries,
        ),
        benchmark_cache_db_path=args.benchmark_cache_db_path,
    )

    print("study_id:", result.study_id)
    print("combinations_tested:", result.total_trials_evaluated)
    print("failed_trials:", len(result.failed_trials))
    print("best_by_return:", result.best_by_return.hyperparameters if result.best_by_return else None)
    print("best_by_drawdown:", result.best_by_drawdown.hyperparameters if result.best_by_drawdown else None)
    print("pareto_front_size:", len(result.pareto_front))
    print("study_summary_json:", result.study_artifacts.get("study_summary_json"))
    print("trial_results_csv:", result.study_artifacts.get("trial_results_csv"))
    print("population_history_csv:", result.study_artifacts.get("population_history_csv"))
    print("best_by_return_summary_json:", result.study_artifacts.get("best_by_return_summary_json"))
    print("best_by_drawdown_summary_json:", result.study_artifacts.get("best_by_drawdown_summary_json"))


if __name__ == "__main__":
    main()
