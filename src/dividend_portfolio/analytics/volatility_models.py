from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _empty_summary(reason: str, *, input_observations: int = 0) -> dict[str, Any]:
    return {
        "enabled": False,
        "library": "arch",
        "input_observations": int(input_observations),
        "models": {},
        "warnings": [reason],
    }


def _fit_single_model(
    scaled_returns: pd.Series,
    *,
    arch_model_fn,
    vol: str,
    p: int,
    q: int,
    name: str,
) -> tuple[dict[str, Any] | None, pd.Series | None, str | None]:
    try:
        model = arch_model_fn(
            scaled_returns,
            mean="Constant",
            vol=vol,
            p=p,
            q=q,
            dist="normal",
            rescale=False,
        )
        fitted = model.fit(disp="off")
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{name} fit failed: {exc}"

    params = {str(k): float(v) for k, v in fitted.params.items()}
    llf = float(fitted.loglikelihood)
    aic = float(fitted.aic)
    bic = float(fitted.bic)

    forecast_vol_1d = float("nan")
    try:
        variance_fc = fitted.forecast(horizon=1, reindex=False).variance
        forecast_var_1d = float(variance_fc.iloc[-1, 0])
        if forecast_var_1d >= 0:
            forecast_vol_1d = math.sqrt(forecast_var_1d) / 100.0
    except Exception:  # noqa: BLE001
        pass

    persistence = float("nan")
    if vol.upper() == "ARCH":
        persistence = float(params.get("alpha[1]", float("nan")))
    elif vol.upper() == "GARCH":
        alpha = float(params.get("alpha[1]", float("nan")))
        beta = float(params.get("beta[1]", float("nan")))
        if not math.isnan(alpha) and not math.isnan(beta):
            persistence = alpha + beta

    unconditional_vol = float("nan")
    omega = float(params.get("omega", float("nan")))
    if not math.isnan(omega) and not math.isnan(persistence) and persistence < 1 and omega > 0:
        unconditional_var = omega / (1.0 - persistence)
        if unconditional_var > 0:
            unconditional_vol = math.sqrt(unconditional_var) / 100.0

    cond_vol = pd.Series(fitted.conditional_volatility, index=scaled_returns.index, dtype=float) / 100.0
    summary = {
        "name": name,
        "log_likelihood": llf,
        "aic": aic,
        "bic": bic,
        "persistence": persistence,
        "forecast_1d_volatility": forecast_vol_1d,
        "unconditional_volatility": unconditional_vol,
        "parameters": params,
    }
    return summary, cond_vol, None


def fit_arch_garch_models(returns: pd.Series) -> tuple[dict[str, Any], pd.DataFrame]:
    cleaned = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return _empty_summary("No valid return observations available for ARCH/GARCH fitting."), pd.DataFrame()
    if len(cleaned) < 80:
        return _empty_summary(
            "Need at least 80 return observations for stable ARCH/GARCH estimation.",
            input_observations=len(cleaned),
        ), pd.DataFrame()

    try:
        from arch import arch_model
    except ModuleNotFoundError:
        return _empty_summary("Python package 'arch' is not installed.", input_observations=len(cleaned)), pd.DataFrame()

    scaled = cleaned.astype(float) * 100.0
    warnings_out: list[str] = []
    model_summaries: dict[str, Any] = {}
    cond_vol_df = pd.DataFrame(index=scaled.index)

    arch_summary, arch_cond, arch_err = _fit_single_model(
        scaled,
        arch_model_fn=arch_model,
        vol="ARCH",
        p=1,
        q=0,
        name="ARCH(1)",
    )
    if arch_err:
        warnings_out.append(arch_err)
    if arch_summary is not None and arch_cond is not None:
        model_summaries["arch_1_0"] = arch_summary
        cond_vol_df["arch_1_0_cond_vol"] = arch_cond

    garch_summary, garch_cond, garch_err = _fit_single_model(
        scaled,
        arch_model_fn=arch_model,
        vol="GARCH",
        p=1,
        q=1,
        name="GARCH(1,1)",
    )
    if garch_err:
        warnings_out.append(garch_err)
    if garch_summary is not None and garch_cond is not None:
        model_summaries["garch_1_1"] = garch_summary
        cond_vol_df["garch_1_1_cond_vol"] = garch_cond

    enabled = len(model_summaries) > 0
    summary = {
        "enabled": enabled,
        "library": "arch",
        "input_observations": int(len(cleaned)),
        "models": model_summaries,
        "warnings": warnings_out,
    }
    return summary, cond_vol_df
