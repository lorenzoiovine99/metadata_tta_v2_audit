from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from metadata_tta.config import (
    ExperimentConfig,
)

from .config import (
    apply_overrides_to_dict,
)
from .results import (
    load_best_result,
)


def _best_overrides(
    path: Path,
) -> dict[str, Any]:

    result = load_best_result(
        path
    )

    overrides = result.get(
        "overrides",
        {},
    )

    if not isinstance(
        overrides,
        Mapping,
    ):
        raise ValueError(
            f"Invalid tuning result: "
            f"{path}"
        )

    return dict(
        overrides
    )


def materialize_tuned_config(
    *,
    base_config: ExperimentConfig,
    output_root: Path,
    enabled_methods: list[str],
    destination: Path,
) -> Path:

    data = (
        base_config.as_dict()
    )

    baseline_paths = [
        output_root
        / "baselines"
        / "single_head"
        / "best.yaml",

        output_root
        / "baselines"
        / "double_head"
        / "best.yaml",
    ]

    for path in baseline_paths:

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing baseline tuning "
                f"result: {path}"
            )

        data = (
            apply_overrides_to_dict(
                data=data,
                overrides=(
                    _best_overrides(
                        path
                    )
                ),
            )
        )

    for method_name in (
        enabled_methods
    ):

        path = (
            output_root
            / "tta"
            / method_name
            / "best.yaml"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing TTA tuning "
                f"result: {path}"
            )

        data = (
            apply_overrides_to_dict(
                data=data,
                overrides=(
                    _best_overrides(
                        path
                    )
                ),
            )
        )

    destination = Path(
        destination
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )

    return destination