from __future__ import annotations

import warnings


def configure_runtime_warning_filters() -> None:
    # Eikon emits this internally; we can't change third-party code in site-packages.
    warnings.filterwarnings(
        "ignore",
        message=r"errors='ignore' is deprecated and will raise in a future version\..*",
        category=FutureWarning,
        module=r"eikon\.data_grid",
    )
    # Refinitiv data lib also emits a noisy pandas downcasting FutureWarning.
    warnings.filterwarnings(
        "ignore",
        message=r"Downcasting behavior in `replace` is deprecated and will be removed in a future version\..*",
        category=FutureWarning,
        module=r"refinitiv\.data\._tools\._dataframe",
    )
