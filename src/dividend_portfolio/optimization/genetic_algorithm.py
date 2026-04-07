from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Any

import pandas as pd

try:
    from ..config import load_portfolio_config
    from ..models import EvaluationContext, EvaluationResult, PortfolioConfig
    from ..reporting.dynamic_results import generate_dynamic_showresults
    from ..strategy.evaluation import evaluate_strategy, prepare_benchmark_data
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.dividend_portfolio.config import load_portfolio_config
    from src.dividend_portfolio.models import EvaluationContext, EvaluationResult, PortfolioConfig
    from src.dividend_portfolio.reporting.dynamic_results import generate_dynamic_showresults
    from src.dividend_portfolio.strategy.evaluation import evaluate_strategy, prepare_benchmark_data


ALLOCATION_STRATEGIES = (
    "equal_weight",
    "market_cap",
    "inverse_market_cap",
    "yield_proportional",
    "yield_rank_linear",
)
BASELINE_SELL_THRESHOLDS = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
DEFAULT_PORTFOLIO_SIZES = tuple(range(7, 36))
DEFAULT_REBALANCE_INTERVALS = (1, 2, 3, 4)


@dataclass(frozen=True)
class GeneticSearchSpace:
    portfolio_sizes: tuple[int, ...] = DEFAULT_PORTFOLIO_SIZES
    rebalance_interval_quarters: tuple[int, ...] = DEFAULT_REBALANCE_INTERVALS
    allocation_strategies: tuple[str, ...] = ALLOCATION_STRATEGIES
    baseline_sell_thresholds: tuple[float, ...] = BASELINE_SELL_THRESHOLDS

    @classmethod
    def intensive_defaults(cls) -> "GeneticSearchSpace":
        return cls()

    def validated(
        self,
        *,
        candidate_count: int | None = None,
        baseline_sell_enabled: bool = False,
    ) -> "GeneticSearchSpace":
        portfolio_sizes = tuple(
            sorted(
                {
                    int(v)
                    for v in self.portfolio_sizes
                    if int(v) > 0 and (candidate_count is None or int(v) <= int(candidate_count))
                }
            )
        )
        rebalance_interval_quarters = tuple(
            sorted({int(v) for v in self.rebalance_interval_quarters if int(v) > 0})
        )
        allocation_strategies = tuple(
            str(v).strip().lower()
            for v in self.allocation_strategies
            if str(v).strip().lower()
        )
        baseline_sell_thresholds = tuple(
            sorted(
                {
                    float(v)
                    for v in self.baseline_sell_thresholds
                    if float(v) > 0.0 and float(v) < 1.0
                }
            )
        )
        if not portfolio_sizes:
            raise ValueError("GeneticSearchSpace.portfolio_sizes must contain at least one positive value")
        if not rebalance_interval_quarters:
            raise ValueError(
                "GeneticSearchSpace.rebalance_interval_quarters must contain at least one positive value"
            )
        if not allocation_strategies:
            raise ValueError(
                "GeneticSearchSpace.allocation_strategies must contain at least one non-empty value"
            )
        if baseline_sell_enabled and not baseline_sell_thresholds:
            raise ValueError(
                "GeneticSearchSpace.baseline_sell_thresholds must contain at least one value in (0, 1)"
            )
        return GeneticSearchSpace(
            portfolio_sizes=portfolio_sizes,
            rebalance_interval_quarters=rebalance_interval_quarters,
            allocation_strategies=allocation_strategies,
            baseline_sell_thresholds=(
                baseline_sell_thresholds if baseline_sell_enabled else tuple()
            ),
        )

    def all_combinations(self) -> list[dict[str, Any]]:
        thresholds: tuple[float | None, ...] = (
            tuple(float(v) for v in self.baseline_sell_thresholds)
            if self.baseline_sell_thresholds
            else (None,)
        )
        combinations: list[dict[str, Any]] = []
        for portfolio_size, rebalance_interval, allocation_strategy, baseline_sell_threshold in itertools.product(
            self.portfolio_sizes,
            self.rebalance_interval_quarters,
            self.allocation_strategies,
            thresholds,
        ):
            row = {
                "portfolio_size": int(portfolio_size),
                "rebalance_interval_quarters": int(rebalance_interval),
                "allocation_strategy": str(allocation_strategy).strip().lower(),
            }
            if baseline_sell_threshold is not None:
                row["baseline_sell_threshold"] = float(baseline_sell_threshold)
            combinations.append(row)
        return combinations

    def combination_count(self) -> int:
        return len(self.all_combinations())


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    population_size: int = 12
    generations: int = 8
    elite_count: int = 2
    crossover_rate: float = 0.75
    mutation_rate: float = 0.20
    tournament_size: int = 3
    random_seed: int | None = None
    persist_trials: str = "none"
    benchmark: str = "none"
    study_id: str | None = None
    trial_output_dir: str | Path | None = None
    persist_study_artifacts: bool = True
    study_output_dir: str | Path | None = None
    max_workers: int = 5
    generate_winner_full_summaries: bool = True


@dataclass
class GeneticIndividual:
    hyperparameters: dict[str, Any]
    evaluation: EvaluationResult
    generation: int
    trial_index: int
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    objectives: tuple[float, ...] = field(default_factory=tuple)
    error: str | None = None
    search_objectives: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneticSearchResult:
    study_id: str
    search_space: GeneticSearchSpace
    config: GeneticAlgorithmConfig
    generations_completed: int
    total_trials_evaluated: int
    failed_trials: list[GeneticIndividual]
    pareto_front: list[GeneticIndividual]
    best_by_return: GeneticIndividual | None
    best_by_drawdown: GeneticIndividual | None
    final_population: list[GeneticIndividual]
    history: list[dict[str, Any]]
    trial_results: list[dict[str, Any]] = field(default_factory=list)
    population_history: list[dict[str, Any]] = field(default_factory=list)
    study_artifacts: dict[str, str] = field(default_factory=dict)
    objective_names: tuple[str, ...] = field(default_factory=tuple)


def _safe_objective(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:  # noqa: BLE001
        return float("-inf")
    if math.isnan(numeric) or math.isinf(numeric):
        return float("-inf")
    return numeric


def _search_objective_names(config: GeneticAlgorithmConfig) -> tuple[str, ...]:
    if str(config.benchmark).strip().lower() == "none":
        return ("cagr", "max_drawdown")
    return (
        "cagr",
        "annualized_excess_return",
        "strategy_up_benchmark_down_periods",
        "inverse_strategy_down_benchmark_up_periods",
        "max_drawdown",
    )


def _search_objective_values(
    evaluation: EvaluationResult,
    *,
    config: GeneticAlgorithmConfig,
) -> dict[str, Any]:
    quarterly_vs_sp500 = evaluation.diagnostic_metrics.get("quarterly_vs_sp500", {})
    if not isinstance(quarterly_vs_sp500, dict):
        quarterly_vs_sp500 = {}
    down_up_periods = quarterly_vs_sp500.get("strategy_down_benchmark_up_periods")
    inverse_down_up_periods = (
        -float(down_up_periods) if down_up_periods is not None else None
    )
    values = {
        "cagr": evaluation.objective_metrics.get("cagr"),
        "annualized_excess_return": evaluation.objective_metrics.get("annualized_excess_return"),
        "strategy_up_benchmark_down_periods": quarterly_vs_sp500.get("strategy_up_benchmark_down_periods"),
        "inverse_strategy_down_benchmark_up_periods": inverse_down_up_periods,
        "max_drawdown": evaluation.constraint_metrics.get("max_drawdown"),
    }
    return {name: values.get(name) for name in _search_objective_names(config)}


def _hyperparameter_key(hyperparameters: dict[str, Any]) -> tuple[Any, ...]:
    key: list[Any] = [
        int(hyperparameters["portfolio_size"]),
        int(hyperparameters["rebalance_interval_quarters"]),
        str(hyperparameters["allocation_strategy"]).strip().lower(),
    ]
    if "baseline_sell_threshold" in hyperparameters:
        key.append(round(float(hyperparameters["baseline_sell_threshold"]), 10))
    return tuple(key)


def _study_id(search_space: GeneticSearchSpace, config: GeneticAlgorithmConfig) -> str:
    if config.study_id:
        return str(config.study_id).strip()
    payload = {
        "portfolio_sizes": list(search_space.portfolio_sizes),
        "rebalance_interval_quarters": list(search_space.rebalance_interval_quarters),
        "allocation_strategies": list(search_space.allocation_strategies),
        "baseline_sell_thresholds": list(search_space.baseline_sell_thresholds),
        "population_size": int(config.population_size),
        "generations": int(config.generations),
        "random_seed": config.random_seed,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"ga_{digest}"


def _trial_output_dir(
    *,
    config: GeneticAlgorithmConfig,
    study_id: str,
    trial_id: str,
) -> Path | None:
    if str(config.persist_trials).strip().lower() != "summary":
        return None
    base = (
        Path(config.trial_output_dir)
        if config.trial_output_dir is not None
        else Path("data") / "runs_dynamic" / "genetic_search" / study_id
    )
    return base / trial_id


def _study_output_dir(*, config: GeneticAlgorithmConfig, study_id: str) -> Path:
    base = (
        Path(config.study_output_dir)
        if config.study_output_dir is not None
        else (
            Path(config.trial_output_dir)
            if config.trial_output_dir is not None
            else Path("data") / "runs_dynamic" / "genetic_search" / study_id
        )
    )
    return base


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return value


def _flatten_dict(data: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}_{key}"
        if isinstance(value, dict):
            out.update(_flatten_dict(value, name))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(_jsonable(value), sort_keys=True)
        else:
            out[name] = _jsonable(value)
    return out


def _compact_evaluation_result(evaluation: EvaluationResult) -> EvaluationResult:
    return EvaluationResult(
        run_id=evaluation.run_id,
        hyperparameters=dict(evaluation.hyperparameters),
        objective_metrics=dict(evaluation.objective_metrics),
        constraint_metrics=dict(evaluation.constraint_metrics),
        diagnostic_metrics=dict(evaluation.diagnostic_metrics),
        summary=dict(evaluation.summary),
        persisted_artifacts=dict(evaluation.persisted_artifacts),
        dynamic_run=None,
    )


def _effective_hyperparameters(
    *,
    base_config: PortfolioConfig,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    strategy = base_config.strategy
    effective = dict(hyperparameters)
    if strategy is None:
        return effective
    effective.setdefault("bond_universe_enabled", bool(strategy.bond_universe.enabled))
    effective.setdefault("baseline_sell_enabled", bool(strategy.baseline_sell_enabled))
    effective.setdefault("baseline_sell_threshold", float(strategy.baseline_sell_threshold))
    return effective


def _worker_payload(
    *,
    base_config: PortfolioConfig,
    hyperparameters: dict[str, Any],
    generation: int,
    trial_index: int,
    study_id: str,
    search_space_id: str,
    config: GeneticAlgorithmConfig,
    provider,
    benchmark_close,
    sp500_close,
    russell_1000_close,
    benchmark_cache_db_path,
) -> dict[str, Any]:
    return {
        "base_config": base_config,
        "hyperparameters": dict(hyperparameters),
        "generation": int(generation),
        "trial_index": int(trial_index),
        "study_id": study_id,
        "search_space_id": search_space_id,
        "config": config,
        "provider": provider,
        "benchmark_close": benchmark_close,
        "sp500_close": sp500_close,
        "russell_1000_close": russell_1000_close,
        "benchmark_cache_db_path": benchmark_cache_db_path,
    }


def _evaluate_individual_worker(payload: dict[str, Any]) -> GeneticIndividual:
    individual = _evaluate_individual(
        base_config=payload["base_config"],
        hyperparameters=payload["hyperparameters"],
        generation=payload["generation"],
        trial_index=payload["trial_index"],
        study_id=payload["study_id"],
        search_space_id=payload["search_space_id"],
        config=payload["config"],
        provider=payload["provider"],
        benchmark_close=payload["benchmark_close"],
        sp500_close=payload["sp500_close"],
        russell_1000_close=payload["russell_1000_close"],
        benchmark_cache_db_path=payload["benchmark_cache_db_path"],
    )
    individual.evaluation = _compact_evaluation_result(individual.evaluation)
    return individual


def _can_parallelize(provider, *, config: GeneticAlgorithmConfig) -> bool:
    if int(config.max_workers) <= 1:
        return False
    if os.name == "nt":
        return False
    if _process_pool_context_name() == "spawn" and not _supports_spawn_process_pool():
        return False
    try:
        pickle.dumps(provider)
    except Exception:  # noqa: BLE001
        return provider is None
    return True


def _parallel_diagnostics(provider, *, config: GeneticAlgorithmConfig) -> dict[str, Any]:
    requested = max(int(config.max_workers), 1)
    context_name = _process_pool_context_name()
    supports_spawn = _supports_spawn_process_pool()
    effective = _process_pool_max_workers(config)
    reason: str | None = None
    if requested <= 1:
        reason = "max_workers<=1"
    elif os.name == "nt":
        reason = "windows_process_pool_disabled"
    elif context_name == "spawn" and not supports_spawn:
        reason = "spawn_requires_module_or_script_entrypoint"
    else:
        try:
            pickle.dumps(provider)
        except Exception:  # noqa: BLE001
            if provider is not None:
                reason = "provider_not_picklable"
    parallel_enabled = reason is None
    return {
        "parallel_enabled": bool(parallel_enabled),
        "requested_max_workers": int(requested),
        "effective_max_workers": int(effective if parallel_enabled else 1),
        "process_pool_context": context_name,
        "supports_spawn_process_pool": bool(supports_spawn),
        "parallel_disable_reason": reason,
    }


def _process_pool_max_workers(config: GeneticAlgorithmConfig) -> int:
    requested = max(int(config.max_workers), 1)
    cpu_count = os.cpu_count() or 1
    return max(1, min(requested, cpu_count))


def _process_pool_context_name() -> str:
    if sys.platform == "darwin":
        return "spawn"
    return "fork"


def _supports_spawn_process_pool() -> bool:
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return False
    main_spec = getattr(main_module, "__spec__", None)
    if main_spec is not None:
        return True
    main_file = getattr(main_module, "__file__", None)
    return isinstance(main_file, str) and bool(main_file) and main_file != "<stdin>"


def _individual_row(
    *,
    study_id: str,
    individual: GeneticIndividual,
    population_generation: int | None = None,
    front_keys: set[tuple[Any, ...]] | None = None,
    best_by_return_key: tuple[Any, ...] | None = None,
    best_by_drawdown_key: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    key = _hyperparameter_key(individual.hyperparameters)
    effective_hyperparameters = dict(individual.evaluation.hyperparameters or {})
    for hp_key, hp_value in individual.hyperparameters.items():
        effective_hyperparameters.setdefault(hp_key, hp_value)
    row: dict[str, Any] = {
        "study_id": study_id,
        "run_id": individual.evaluation.run_id,
        "trial_index": int(individual.trial_index),
        "first_generation_evaluated": int(individual.generation),
        "population_generation": (
            int(population_generation) if population_generation is not None else int(individual.generation)
        ),
        "status": "failed" if individual.error else "ok",
        "error": individual.error,
        "portfolio_size": int(effective_hyperparameters["portfolio_size"]),
        "rebalance_interval_quarters": int(effective_hyperparameters["rebalance_interval_quarters"]),
        "allocation_strategy": str(effective_hyperparameters["allocation_strategy"]),
        "bond_universe_enabled": bool(effective_hyperparameters.get("bond_universe_enabled", False)),
        "baseline_sell_enabled": bool(effective_hyperparameters.get("baseline_sell_enabled", False)),
        "baseline_sell_threshold": effective_hyperparameters.get("baseline_sell_threshold"),
        "pareto_rank": int(individual.pareto_rank),
        "crowding_distance": (
            None if math.isinf(individual.crowding_distance) else float(individual.crowding_distance)
        ),
        "search_objective_vector": json.dumps([_jsonable(v) for v in individual.objectives]),
        "is_pareto_front": bool(front_keys is not None and key in front_keys),
        "is_best_by_return": bool(best_by_return_key is not None and key == best_by_return_key),
        "is_best_by_drawdown": bool(best_by_drawdown_key is not None and key == best_by_drawdown_key),
        "summary_json_path": individual.evaluation.persisted_artifacts.get("summary_json"),
    }
    row.update(_flatten_dict(individual.search_objectives, "search"))
    row.update(_flatten_dict(individual.evaluation.objective_metrics, "objective"))
    row.update(_flatten_dict(individual.evaluation.constraint_metrics, "constraint"))
    row.update(_flatten_dict(individual.evaluation.diagnostic_metrics, "diagnostic"))
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=False)


def _random_hyperparameters(
    search_space: GeneticSearchSpace,
    *,
    rng: random.Random,
) -> dict[str, Any]:
    hyperparameters = {
        "portfolio_size": rng.choice(search_space.portfolio_sizes),
        "rebalance_interval_quarters": rng.choice(search_space.rebalance_interval_quarters),
        "allocation_strategy": rng.choice(search_space.allocation_strategies),
    }
    if search_space.baseline_sell_thresholds:
        hyperparameters["baseline_sell_threshold"] = rng.choice(search_space.baseline_sell_thresholds)
    return hyperparameters


def _crossover(
    parent_a: dict[str, Any],
    parent_b: dict[str, Any],
    *,
    rng: random.Random,
    crossover_rate: float,
) -> dict[str, Any]:
    if rng.random() >= float(crossover_rate):
        return dict(parent_a)
    child: dict[str, Any] = {}
    keys = ["portfolio_size", "rebalance_interval_quarters", "allocation_strategy"]
    if "baseline_sell_threshold" in parent_a or "baseline_sell_threshold" in parent_b:
        keys.append("baseline_sell_threshold")
    for key in keys:
        child[key] = parent_a[key] if rng.random() < 0.5 else parent_b[key]
    return child


def _mutate(
    hyperparameters: dict[str, Any],
    *,
    search_space: GeneticSearchSpace,
    rng: random.Random,
    mutation_rate: float,
) -> dict[str, Any]:
    child = dict(hyperparameters)
    if rng.random() < float(mutation_rate):
        child["portfolio_size"] = rng.choice(search_space.portfolio_sizes)
    if rng.random() < float(mutation_rate):
        child["rebalance_interval_quarters"] = rng.choice(search_space.rebalance_interval_quarters)
    if rng.random() < float(mutation_rate):
        child["allocation_strategy"] = rng.choice(search_space.allocation_strategies)
    if search_space.baseline_sell_thresholds and rng.random() < float(mutation_rate):
        child["baseline_sell_threshold"] = rng.choice(search_space.baseline_sell_thresholds)
    return child


def _dominates(left: GeneticIndividual, right: GeneticIndividual) -> bool:
    return (
        all(left_value >= right_value for left_value, right_value in zip(left.objectives, right.objectives, strict=False))
        and any(left_value > right_value for left_value, right_value in zip(left.objectives, right.objectives, strict=False))
    )


def _assign_pareto_rank_and_crowding(population: list[GeneticIndividual]) -> list[list[GeneticIndividual]]:
    if not population:
        return []

    domination_count = {id(individual): 0 for individual in population}
    dominates_map: dict[int, list[GeneticIndividual]] = {id(individual): [] for individual in population}
    fronts: list[list[GeneticIndividual]] = [[]]

    for individual in population:
        for other in population:
            if individual is other:
                continue
            if _dominates(individual, other):
                dominates_map[id(individual)].append(other)
            elif _dominates(other, individual):
                domination_count[id(individual)] += 1
        if domination_count[id(individual)] == 0:
            individual.pareto_rank = 0
            fronts[0].append(individual)

    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[GeneticIndividual] = []
        for individual in fronts[front_index]:
            for dominated in dominates_map[id(individual)]:
                domination_count[id(dominated)] -= 1
                if domination_count[id(dominated)] == 0:
                    dominated.pareto_rank = front_index + 1
                    next_front.append(dominated)
        if next_front:
            fronts.append(next_front)
        front_index += 1

    for front in fronts:
        if not front:
            continue
        for individual in front:
            individual.crowding_distance = 0.0
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float("inf")
            continue

        objective_count = len(front[0].objectives)
        for objective_idx in range(objective_count):
            front.sort(key=lambda item: item.objectives[objective_idx])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")
            min_value = front[0].objectives[objective_idx]
            max_value = front[-1].objectives[objective_idx]
            if max_value == min_value:
                continue
            scale = max_value - min_value
            for idx in range(1, len(front) - 1):
                if math.isinf(front[idx].crowding_distance):
                    continue
                front[idx].crowding_distance += (
                    front[idx + 1].objectives[objective_idx]
                    - front[idx - 1].objectives[objective_idx]
                ) / scale

    return [front for front in fronts if front]


def _select_parent(
    population: list[GeneticIndividual],
    *,
    rng: random.Random,
    tournament_size: int,
) -> GeneticIndividual:
    contenders = rng.sample(population, k=min(max(int(tournament_size), 1), len(population)))
    contenders.sort(
        key=lambda item: (
            item.pareto_rank,
            -item.crowding_distance,
            *(-objective for objective in item.objectives),
        )
    )
    return contenders[0]


def _sort_population(population: list[GeneticIndividual]) -> list[GeneticIndividual]:
    return sorted(
        population,
        key=lambda item: (
            item.pareto_rank,
            -item.crowding_distance,
            *(-objective for objective in item.objectives),
        ),
    )


def _evaluate_individual(
    *,
    base_config: PortfolioConfig,
    hyperparameters: dict[str, Any],
    generation: int,
    trial_index: int,
    study_id: str,
    search_space_id: str,
    config: GeneticAlgorithmConfig,
    provider,
    benchmark_close,
    sp500_close,
    russell_1000_close,
    benchmark_cache_db_path,
) -> GeneticIndividual:
    trial_id = f"{study_id}_g{generation:02d}_t{trial_index:04d}"
    objective_names = _search_objective_names(config)
    error: str | None = None
    try:
        evaluation = evaluate_strategy(
            base_config=base_config,
            hyperparameters=hyperparameters,
            persist=config.persist_trials,
            benchmark=config.benchmark,
            evaluation_context=EvaluationContext(
                study_id=study_id,
                trial_id=trial_id,
                search_space_id=search_space_id,
                trial_index=trial_index,
            ),
            provider=provider,
            benchmark_close=benchmark_close,
            sp500_close=sp500_close,
            russell_1000_close=russell_1000_close,
            benchmark_cache_db_path=benchmark_cache_db_path,
            run_id=trial_id,
            output_dir=_trial_output_dir(config=config, study_id=study_id, trial_id=trial_id),
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        effective_hyperparameters = _effective_hyperparameters(
            base_config=base_config,
            hyperparameters=hyperparameters,
        )
        evaluation = EvaluationResult(
            run_id=trial_id,
            hyperparameters=effective_hyperparameters,
            objective_metrics={"cagr": None},
            constraint_metrics={"max_drawdown": None},
            diagnostic_metrics={},
            summary={
                "run_id": trial_id,
                "status": "failed",
                "error": error,
                "hyperparameters": effective_hyperparameters,
            },
            persisted_artifacts={},
            dynamic_run=None,
        )
    search_objectives = _search_objective_values(evaluation, config=config)
    objectives = tuple(_safe_objective(search_objectives.get(name)) for name in objective_names)
    return GeneticIndividual(
        hyperparameters=dict(hyperparameters),
        evaluation=evaluation,
        generation=int(generation),
        trial_index=int(trial_index),
        objectives=objectives,
        error=error,
        search_objectives=search_objectives,
    )


def _raise_if_all_failed(population: list[GeneticIndividual], *, study_id: str, stage: str) -> None:
    if not population or any(individual.error is None for individual in population):
        return
    first_failed = population[0]
    raise ValueError(
        f"Genetic search {study_id} produced no valid trials during {stage}. "
        f"First failure for hyperparameters={first_failed.hyperparameters}: {first_failed.error}"
    )


def _generation_summary(
    population: list[GeneticIndividual],
    generation: int,
    *,
    objective_names: tuple[str, ...],
) -> dict[str, Any]:
    if not population:
        summary = {
            "generation": int(generation),
            "population_size": 0,
            "pareto_front_size": 0,
        }
        for name in objective_names:
            summary[f"best_{name}"] = None
        return summary
    ordered = _sort_population(population)
    front_size = sum(1 for individual in ordered if individual.pareto_rank == 0)
    summary = {
        "generation": int(generation),
        "population_size": int(len(population)),
        "pareto_front_size": int(front_size),
    }
    for idx, name in enumerate(objective_names):
        best_value = max(individual.objectives[idx] for individual in ordered)
        summary[f"best_{name}"] = float(best_value)
    return summary


def _study_summary_payload(
    *,
    study_id: str,
    search_space: GeneticSearchSpace,
    config: GeneticAlgorithmConfig,
    execution_plan: dict[str, Any],
    generations_completed: int,
    total_trials_evaluated: int,
    failed_trials: list[GeneticIndividual],
    pareto_front: list[GeneticIndividual],
    best_by_return: GeneticIndividual | None,
    best_by_drawdown: GeneticIndividual | None,
    history: list[dict[str, Any]],
    study_artifacts: dict[str, str],
    objective_names: tuple[str, ...],
    winner_summaries: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def compact(individual: GeneticIndividual | None) -> dict[str, Any] | None:
        if individual is None:
            return None
        return {
            "run_id": individual.evaluation.run_id,
            "trial_index": int(individual.trial_index),
            "generation": int(individual.generation),
            "hyperparameters": dict(individual.evaluation.hyperparameters or individual.hyperparameters),
            "search_objectives": dict(individual.search_objectives),
            "objective_metrics": dict(individual.evaluation.objective_metrics),
            "constraint_metrics": dict(individual.evaluation.constraint_metrics),
            "diagnostic_metrics": dict(individual.evaluation.diagnostic_metrics),
            "error": individual.error,
            "summary_json_path": individual.evaluation.persisted_artifacts.get("summary_json"),
        }

    return {
        "study_id": study_id,
        "search_space": {
            "portfolio_sizes": list(search_space.portfolio_sizes),
            "rebalance_interval_quarters": list(search_space.rebalance_interval_quarters),
            "allocation_strategies": list(search_space.allocation_strategies),
            "baseline_sell_thresholds": list(search_space.baseline_sell_thresholds),
            "combination_count": int(search_space.combination_count()),
        },
        "config": asdict(config),
        "execution_plan": execution_plan,
        "objective_names": list(objective_names),
        "generations_completed": int(generations_completed),
        "total_trials_evaluated": int(total_trials_evaluated),
        "failed_trial_count": int(len(failed_trials)),
        "history": history,
        "best_by_return": compact(best_by_return),
        "best_by_drawdown": compact(best_by_drawdown),
        "pareto_front": [compact(individual) for individual in pareto_front],
        "artifacts": study_artifacts,
        "winner_summaries": winner_summaries or {},
    }


def _winner_summary_targets(
    *,
    best_by_return: GeneticIndividual | None,
    best_by_drawdown: GeneticIndividual | None,
) -> list[tuple[str, GeneticIndividual]]:
    targets: list[tuple[str, GeneticIndividual]] = []
    if best_by_return is not None and best_by_return.error is None:
        targets.append(("best_by_return", best_by_return))
    if best_by_drawdown is not None and best_by_drawdown.error is None:
        targets.append(("best_by_drawdown", best_by_drawdown))
    return targets


def _generate_winner_full_summaries(
    *,
    base_config: PortfolioConfig,
    ga_config: GeneticAlgorithmConfig,
    study_id: str,
    search_space_id: str,
    study_dir: Path,
    provider,
    benchmark_close,
    sp500_close,
    russell_1000_close,
    benchmark_cache_db_path: str | Path | None,
    best_by_return: GeneticIndividual | None,
    best_by_drawdown: GeneticIndividual | None,
) -> dict[str, Any]:
    if not ga_config.generate_winner_full_summaries:
        return {}

    targets = _winner_summary_targets(
        best_by_return=best_by_return,
        best_by_drawdown=best_by_drawdown,
    )
    if not targets:
        return {}

    include_benchmark = str(ga_config.benchmark).strip().lower() != "none"
    winner_root = study_dir / "winner_summaries"
    winner_root.mkdir(parents=True, exist_ok=True)

    generated_by_key: dict[tuple[int, int, str], tuple[str, dict[str, Any]]] = {}
    winner_summaries: dict[str, Any] = {}

    for label, individual in targets:
        key = _hyperparameter_key(individual.hyperparameters)
        if key in generated_by_key:
            primary_label, primary_payload = generated_by_key[key]
            aliased = dict(primary_payload)
            aliased["label"] = label
            aliased["aliased_to"] = primary_label
            winner_summaries[label] = aliased
            continue

        winner_dir = winner_root / label
        winner_dir.mkdir(parents=True, exist_ok=True)
        winner_run_id = f"{study_id}_{label}"

        try:
            evaluation = evaluate_strategy(
                base_config=base_config,
                hyperparameters=individual.hyperparameters,
                persist="full",
                benchmark=ga_config.benchmark,
                evaluation_context=EvaluationContext(
                    study_id=study_id,
                    trial_id=winner_run_id,
                    search_space_id=search_space_id,
                    trial_index=individual.trial_index,
                ),
                run_id=winner_run_id,
                provider=provider,
                benchmark_close=benchmark_close,
                sp500_close=sp500_close,
                russell_1000_close=russell_1000_close,
                benchmark_cache_db_path=benchmark_cache_db_path,
            )
            _, showresults_dir = generate_dynamic_showresults(
                db_path=evaluation.persisted_artifacts["sqlite_db"],
                run_id=evaluation.run_id,
                output_dir=winner_dir / "showresults",
                include_benchmark=include_benchmark,
            )
            payload = {
                "label": label,
                "run_id": evaluation.run_id,
                "db_path": evaluation.persisted_artifacts["sqlite_db"],
                "showresults_dir": str(showresults_dir),
                "summary_json": str(Path(showresults_dir) / "summary.json"),
                "hyperparameters": dict(evaluation.hyperparameters),
                "source_trial_run_id": individual.evaluation.run_id,
            }
        except Exception as exc:  # noqa: BLE001
            payload = {
                "label": label,
                "hyperparameters": dict(individual.evaluation.hyperparameters or individual.hyperparameters),
                "source_trial_run_id": individual.evaluation.run_id,
                "error": f"{type(exc).__name__}: {exc}",
            }

        generated_by_key[key] = (label, dict(payload))
        winner_summaries[label] = payload

    return winner_summaries


def _clone_individual(individual: GeneticIndividual, *, generation: int) -> GeneticIndividual:
    return GeneticIndividual(
        hyperparameters=dict(individual.hyperparameters),
        evaluation=_compact_evaluation_result(individual.evaluation),
        generation=int(generation),
        trial_index=int(individual.trial_index),
        pareto_rank=int(individual.pareto_rank),
        crowding_distance=float(individual.crowding_distance),
        objectives=tuple(individual.objectives),
        error=individual.error,
        search_objectives=dict(individual.search_objectives),
    )


def run_genetic_algorithm(
    *,
    base_config: PortfolioConfig,
    search_space: GeneticSearchSpace | None = None,
    config: GeneticAlgorithmConfig | None = None,
    provider=None,
    benchmark_close=None,
    sp500_close=None,
    russell_1000_close=None,
    benchmark_cache_db_path: str | Path | None = None,
) -> GeneticSearchResult:
    ga_config = config or GeneticAlgorithmConfig()
    objective_names = _search_objective_names(ga_config)
    strategy = base_config.strategy
    candidate_count = int(strategy.candidate_count) if strategy is not None else None
    baseline_sell_enabled = bool(strategy.baseline_sell_enabled) if strategy is not None else False
    space = (search_space or GeneticSearchSpace()).validated(
        candidate_count=candidate_count,
        baseline_sell_enabled=baseline_sell_enabled,
    )
    study_id = _study_id(space, ga_config)
    search_space_id = hashlib.sha1(
        json.dumps(
            {
                "portfolio_sizes": list(space.portfolio_sizes),
                "rebalance_interval_quarters": list(space.rebalance_interval_quarters),
                "allocation_strategies": list(space.allocation_strategies),
                "baseline_sell_thresholds": list(space.baseline_sell_thresholds),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    rng = random.Random(ga_config.random_seed)
    all_candidates = space.all_combinations()
    if not all_candidates:
        raise ValueError("Search space must contain at least one candidate")

    start = base_config.start_date.isoformat()
    end = (
        base_config.end_date.isoformat()
        if base_config.end_date is not None
        else pd.Timestamp.today().date().isoformat()
    )

    benchmark_close, sp500_close, russell_1000_close = prepare_benchmark_data(
        benchmark=ga_config.benchmark,
        start_date=start,
        end_date=end,
        benchmark_close=benchmark_close,
        sp500_close=sp500_close,
        russell_1000_close=russell_1000_close,
        benchmark_cache_db_path=benchmark_cache_db_path,
    )

    population_size = min(max(int(ga_config.population_size), 1), len(all_candidates))
    elite_count = min(max(int(ga_config.elite_count), 0), population_size)

    evaluation_cache: dict[tuple[int, int, str], GeneticIndividual] = {}
    total_trials = 0
    population_history: list[dict[str, Any]] = []
    execution_plan = _parallel_diagnostics(provider, config=ga_config)
    parallel_enabled = bool(execution_plan["parallel_enabled"])
    executor: ProcessPoolExecutor | None = None
    if parallel_enabled:
        executor = ProcessPoolExecutor(
            max_workers=int(execution_plan["effective_max_workers"]),
            mp_context=multiprocessing.get_context(_process_pool_context_name()),
        )

    def evaluate_candidates(
        hyperparameter_candidates: list[dict[str, Any]],
        *,
        generation: int,
    ) -> list[GeneticIndividual]:  
        nonlocal total_trials
        results: list[GeneticIndividual | None] = [None] * len(hyperparameter_candidates)
        payloads: list[dict[str, Any]] = []
        pending_keys: list[tuple[int, int, str]] = []

        for idx, hyperparameters in enumerate(hyperparameter_candidates):
            key = _hyperparameter_key(hyperparameters)
            cached = evaluation_cache.get(key)
            if cached is not None:
                results[idx] = _clone_individual(cached, generation=generation)
                continue
            if key in pending_keys:
                continue
            total_trials += 1
            payloads.append(
                _worker_payload(
                    base_config=base_config,
                    hyperparameters=hyperparameters,
                    generation=generation,
                    trial_index=total_trials,
                    study_id=study_id,
                    search_space_id=search_space_id,
                    config=ga_config,
                    provider=provider,
                    benchmark_close=benchmark_close,
                    sp500_close=sp500_close,
                    russell_1000_close=russell_1000_close,
                    benchmark_cache_db_path=benchmark_cache_db_path,
                )
            )
            pending_keys.append(key)

        if payloads:
            if executor is not None and len(payloads) > 1:
                evaluated_batch = list(executor.map(_evaluate_individual_worker, payloads))
            else:
                evaluated_batch = [_evaluate_individual_worker(payload) for payload in payloads]
            for key, evaluated in zip(pending_keys, evaluated_batch, strict=False):
                evaluation_cache[key] = evaluated

        for idx, hyperparameters in enumerate(hyperparameter_candidates):
            if results[idx] is not None:
                continue
            key = _hyperparameter_key(hyperparameters)
            results[idx] = _clone_individual(evaluation_cache[key], generation=generation)

        return [individual for individual in results if individual is not None]

    try:
        initial_candidates = rng.sample(all_candidates, k=population_size)
        population = evaluate_candidates(initial_candidates, generation=0)
        _raise_if_all_failed(population, study_id=study_id, stage="the initial population")
        fronts = _assign_pareto_rank_and_crowding(population)
        history = [_generation_summary(population, generation=0, objective_names=objective_names)]
        generation_front_keys = (
            {_hyperparameter_key(individual.hyperparameters) for individual in fronts[0]} if fronts else set()
        )
        population_history.extend(
            _individual_row(
                study_id=study_id,
                individual=individual,
                population_generation=0,
                front_keys=generation_front_keys,
            )
            for individual in _sort_population(population)
        )

        for generation in range(1, max(int(ga_config.generations), 1) + 1):
            ordered = _sort_population(population)
            next_population: list[GeneticIndividual] = [
                _clone_individual(individual, generation=generation)
                for individual in ordered[:elite_count]
            ]

            child_candidates: list[dict[str, Any]] = []
            while len(next_population) + len(child_candidates) < population_size:
                parent_a = _select_parent(
                    population,
                    rng=rng,
                    tournament_size=ga_config.tournament_size,
                )
                parent_b = _select_parent(
                    population,
                    rng=rng,
                    tournament_size=ga_config.tournament_size,
                )
                child_params = _crossover(
                    parent_a.hyperparameters,
                    parent_b.hyperparameters,
                    rng=rng,
                    crossover_rate=ga_config.crossover_rate,
                )
                child_params = _mutate(
                    child_params,
                    search_space=space,
                    rng=rng,
                    mutation_rate=ga_config.mutation_rate,
                )
                child_candidates.append(child_params)

            next_population.extend(evaluate_candidates(child_candidates, generation=generation))

            population = next_population[:population_size]
            _raise_if_all_failed(population, study_id=study_id, stage=f"generation {generation}")
            fronts = _assign_pareto_rank_and_crowding(population)
            history.append(_generation_summary(population, generation=generation, objective_names=objective_names))
            generation_front_keys = (
                {_hyperparameter_key(individual.hyperparameters) for individual in fronts[0]} if fronts else set()
            )
            population_history.extend(
                _individual_row(
                    study_id=study_id,
                    individual=individual,
                    population_generation=generation,
                    front_keys=generation_front_keys,
                )
                for individual in _sort_population(population)
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    ordered_final = _sort_population(population)
    evaluated_population = [
        GeneticIndividual(
            hyperparameters=dict(individual.hyperparameters),
            evaluation=individual.evaluation,
            generation=int(individual.generation),
            trial_index=int(individual.trial_index),
            objectives=tuple(individual.objectives),
            error=individual.error,
            search_objectives=dict(individual.search_objectives),
        )
        for individual in evaluation_cache.values()
    ]
    evaluated_fronts = _assign_pareto_rank_and_crowding(evaluated_population)
    ordered_evaluated = _sort_population(evaluated_population)
    pareto_front = list(evaluated_fronts[0]) if evaluated_fronts else []
    failed_trials = [individual for individual in ordered_evaluated if individual.error is not None]
    cagr_idx = objective_names.index("cagr")
    drawdown_idx = objective_names.index("max_drawdown")
    best_by_return = (
        max(
            ordered_evaluated,
            key=lambda item: (
                item.objectives[cagr_idx],
                item.objectives[drawdown_idx],
                *item.objectives,
            ),
        )
        if ordered_evaluated
        else None
    )
    best_by_drawdown = (
        max(
            ordered_evaluated,
            key=lambda item: (
                item.objectives[drawdown_idx],
                item.objectives[cagr_idx],
                *item.objectives,
            ),
        )
        if ordered_evaluated
        else None
    )
    front_keys = {_hyperparameter_key(individual.hyperparameters) for individual in pareto_front}
    best_by_return_key = (
        _hyperparameter_key(best_by_return.hyperparameters) if best_by_return is not None else None
    )
    best_by_drawdown_key = (
        _hyperparameter_key(best_by_drawdown.hyperparameters) if best_by_drawdown is not None else None
    )
    trial_results = [
        _individual_row(
            study_id=study_id,
            individual=individual,
            front_keys=front_keys,
            best_by_return_key=best_by_return_key,
            best_by_drawdown_key=best_by_drawdown_key,
        )
        for individual in sorted(ordered_evaluated, key=lambda item: item.trial_index)
    ]
    study_artifacts: dict[str, str] = {}
    winner_summaries: dict[str, Any] = {}
    if ga_config.persist_study_artifacts:
        study_dir = _study_output_dir(config=ga_config, study_id=study_id)
        study_dir.mkdir(parents=True, exist_ok=True)
        trial_results_csv = study_dir / "trial_results.csv"
        trial_results_json = study_dir / "trial_results.json"
        population_history_csv = study_dir / "population_history.csv"
        generation_history_csv = study_dir / "generation_history.csv"
        generation_history_json = study_dir / "generation_history.json"
        _write_csv(trial_results_csv, trial_results)
        _write_json(trial_results_json, trial_results)
        _write_csv(population_history_csv, population_history)
        _write_csv(generation_history_csv, history)
        _write_json(generation_history_json, history)
        study_artifacts = {
            "study_dir": str(study_dir),
            "trial_results_csv": str(trial_results_csv),
            "trial_results_json": str(trial_results_json),
            "population_history_csv": str(population_history_csv),
            "generation_history_csv": str(generation_history_csv),
            "generation_history_json": str(generation_history_json),
        }
        winner_summaries = _generate_winner_full_summaries(
            base_config=base_config,
            ga_config=ga_config,
            study_id=study_id,
            search_space_id=search_space_id,
            study_dir=study_dir,
            provider=provider,
            benchmark_close=benchmark_close,
            sp500_close=sp500_close,
            russell_1000_close=russell_1000_close,
            benchmark_cache_db_path=benchmark_cache_db_path,
            best_by_return=best_by_return,
            best_by_drawdown=best_by_drawdown,
        )
        for label, payload in winner_summaries.items():
            if not isinstance(payload, dict):
                continue
            summary_json = payload.get("summary_json")
            if summary_json:
                study_artifacts[f"{label}_summary_json"] = str(summary_json)
            showresults_dir = payload.get("showresults_dir")
            if showresults_dir:
                study_artifacts[f"{label}_showresults_dir"] = str(showresults_dir)
            db_path = payload.get("db_path")
            if db_path:
                study_artifacts[f"{label}_db_path"] = str(db_path)
        study_summary_json = study_dir / "study_summary.json"
        _write_json(
            study_summary_json,
            _study_summary_payload(
                study_id=study_id,
                search_space=space,
                config=ga_config,
                execution_plan=execution_plan,
                generations_completed=max(int(ga_config.generations), 1),
                total_trials_evaluated=int(total_trials),
                failed_trials=failed_trials,
                pareto_front=pareto_front,
                best_by_return=best_by_return,
                best_by_drawdown=best_by_drawdown,
                history=history,
                study_artifacts=study_artifacts,
                objective_names=objective_names,
                winner_summaries=winner_summaries,
            ),
        )
        study_artifacts["study_summary_json"] = str(study_summary_json)

    return GeneticSearchResult(
        study_id=study_id,
        search_space=space,
        config=ga_config,
        generations_completed=max(int(ga_config.generations), 1),
        total_trials_evaluated=int(total_trials),
        failed_trials=failed_trials,
        pareto_front=pareto_front,
        best_by_return=best_by_return,
        best_by_drawdown=best_by_drawdown,
        final_population=ordered_final,
        history=history,
        trial_results=trial_results,
        population_history=population_history,
        study_artifacts=study_artifacts,
        objective_names=objective_names,
    )


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
    strategy = cfg.strategy
    validated_space = GeneticSearchSpace().validated(
        candidate_count=(int(strategy.candidate_count) if strategy is not None else None),
        baseline_sell_enabled=(bool(strategy.baseline_sell_enabled) if strategy is not None else False),
    )
    study_id = args.study_id or f"ga_exhaustive_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    population_size = (
        args.population_size if args.population_size is not None else validated_space.combination_count()
    )
    execution_plan = _parallel_diagnostics(None, config=GeneticAlgorithmConfig(max_workers=int(args.max_workers)))

    print("search_space_combinations:", validated_space.combination_count())
    print("population_size:", int(population_size))
    print("parallel_enabled:", execution_plan["parallel_enabled"])
    print("requested_max_workers:", execution_plan["requested_max_workers"])
    print("effective_max_workers:", execution_plan["effective_max_workers"])
    print("process_pool_context:", execution_plan["process_pool_context"])
    if execution_plan["parallel_disable_reason"]:
        print("parallel_disable_reason:", execution_plan["parallel_disable_reason"])

    result = run_genetic_algorithm(
        base_config=cfg,
        search_space=validated_space,
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
