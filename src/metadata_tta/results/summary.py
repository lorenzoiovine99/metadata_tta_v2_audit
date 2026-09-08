from __future__ import annotations

from collections import defaultdict

import numpy as np

from metadata_tta.evaluation import (
    EvaluationRecord,
)


def build_ood_summary(
    records: list[EvaluationRecord]
    | tuple[EvaluationRecord, ...],
    reference_method: str,
) -> list[dict[str, object]]:

    grouped: dict[
        str,
        list[float],
    ] = defaultdict(list)

    for record in records:
        grouped[
            record.method
        ].append(
            float(record.accuracy)
        )

    if reference_method not in grouped:
        raise ValueError(
            f"Reference method "
            f"{reference_method!r} "
            "is missing from OOD records."
        )

    reference_mean = float(
        np.mean(
            grouped[
                reference_method
            ]
        )
    )

    rows: list[
        dict[str, object]
    ] = []

    for method in sorted(
        grouped
    ):

        values = np.asarray(
            grouped[method],
            dtype=np.float64,
        )

        mean_ood = float(
            np.mean(values)
        )

        rows.append(
            {
                "method": method,
                "mean_ood": mean_ood,
                "std_ood": float(
                    np.std(values)
                ),
                "worst_ood": float(
                    np.min(values)
                ),
                "delta_vs_reference":
                    mean_ood
                    - reference_mean,
            }
        )

    return rows