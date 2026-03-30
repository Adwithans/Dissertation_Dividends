"""Dividend portfolio backtesting package."""

from .runtime_warnings import configure_runtime_warning_filters
from .config import load_portfolio_config
from .models import (
    AssetConfig,
    EvaluationContext,
    EvaluationResult,
    PortfolioConfig,
    QuarterlyMetricsConfig,
    RebalanceConfig,
    SelectionPolicyConfig,
    StrategyConfig,
)
from .optimization import (
    GeneticAlgorithmConfig,
    GeneticIndividual,
    GeneticSearchResult,
    GeneticSearchSpace,
    run_genetic_algorithm,
)
from .strategy import evaluate_strategy

configure_runtime_warning_filters()

__all__ = [
    "AssetConfig",
    "EvaluationContext",
    "EvaluationResult",
    "GeneticAlgorithmConfig",
    "GeneticIndividual",
    "GeneticSearchResult",
    "GeneticSearchSpace",
    "PortfolioConfig",
    "QuarterlyMetricsConfig",
    "RebalanceConfig",
    "SelectionPolicyConfig",
    "StrategyConfig",
    "evaluate_strategy",
    "load_portfolio_config",
    "run_genetic_algorithm",
]
