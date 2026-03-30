from __future__ import annotations

from src.dividend_portfolio.cli.showresults import parse_args


def test_showresults_parse_args_accepts_experiment_group(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "showresults.py",
            "--config",
            "config/portfolio.yaml",
            "--run-id",
            "run_1",
            "--experiment-group",
            "realloc_frequency",
        ],
    )
    args = parse_args()
    assert args.config == "config/portfolio.yaml"
    assert args.run_id == "run_1"
    assert args.experiment_group == "realloc_frequency"
