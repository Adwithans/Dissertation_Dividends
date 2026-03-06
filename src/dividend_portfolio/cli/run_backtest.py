from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..analytics.attribution import compute_asset_attribution
from ..analytics.metrics import compute_portfolio_metrics
from ..analytics.quarterly import compute_quarterly_stock_metrics
from ..config import load_portfolio_config
from ..io.history_io import load_histories
from ..io.run_io import create_run_dir, write_run_outputs
from ..sim.multi_asset import simulate_portfolio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-stock dividend portfolio backtest.")
    parser.add_argument("--config", default="config/portfolio.yaml", help="Path to portfolio config")
    parser.add_argument(
        "--history-dir",
        default="data/raw/history",
        help="Directory containing cached history CSV files",
    )
    parser.add_argument(
        "--fallback-history-dir",
        default="stock_data",
        help="Fallback directory for history CSV files",
    )
    parser.add_argument("--run-base-dir", default="data/runs", help="Base output directory for run artifacts")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable PNG chart outputs",
    )
    return parser.parse_args()


def run_backtest(
    *,
    config_path: str = "config/portfolio.yaml",
    history_dir: str = "data/raw/history",
    fallback_history_dir: str | None = "stock_data",
    run_base_dir: str = "data/runs",
    generate_plots: bool = True,
) -> Path:
    cfg = load_portfolio_config(config_path)
    rics = [asset.ric for asset in cfg.assets]

    histories = load_histories(
        rics,
        primary_dir=history_dir,
        fallback_dir=fallback_history_dir,
    )

    sim = simulate_portfolio(histories, cfg)

    portfolio_df = sim.portfolio_df
    asset_results = sim.asset_results
    rebalance_log = sim.rebalance_log if sim.rebalance_log is not None else pd.DataFrame()

    metrics = compute_portfolio_metrics(
        portfolio_df,
        initial_capital=cfg.initial_capital,
        risk_free_rate=cfg.risk_free_rate,
    )
    attribution_df = compute_asset_attribution(asset_results, initial_capital=cfg.initial_capital)

    if cfg.quarterly_metrics.enabled:
        quarterly_df = compute_quarterly_stock_metrics(asset_results, portfolio_df)
    else:
        quarterly_df = pd.DataFrame()

    run_dir = create_run_dir(run_base_dir)
    write_run_outputs(
        run_dir=run_dir,
        portfolio_df=portfolio_df,
        asset_results=asset_results,
        metrics=metrics,
        attribution_df=attribution_df,
        quarterly_df=quarterly_df,
        rebalance_log=rebalance_log,
    )

    if generate_plots:
        try:
            from ..reporting.plots import save_portfolio_plots
        except ModuleNotFoundError as exc:
            print(f"[WARN] Plot generation skipped: {exc}")
        else:
            try:
                save_portfolio_plots(portfolio_df, asset_results, run_dir)
                print(f"[INFO] Plot files written under: {run_dir}")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Plot generation failed: {exc}")

    print(f"[OK] Backtest complete. Run directory: {run_dir}")
    print(f"[INFO] Latest run alias: {Path(run_dir).parent / 'latest'}")
    print(f"[INFO] Effective start: {sim.effective_start.date().isoformat()}")
    print(f"[INFO] End value: {metrics['end_value']:.2f}  Total return: {metrics['total_return']:.4%}")
    print(
        "[INFO] Total dividend value gained: "
        f"{metrics['total_dividend_value_gained_usd']:.2f} "
        f"({metrics['total_dividend_value_gained_pct_of_initial']:.4%} of initial capital)"
    )
    print("[INFO] Per-stock total dividend payments: see asset_attribution.csv (Total_Dividend_Payments)")

    return run_dir


def main() -> None:
    args = parse_args()
    run_backtest(
        config_path=args.config,
        history_dir=args.history_dir,
        fallback_history_dir=args.fallback_history_dir,
        run_base_dir=args.run_base_dir,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
