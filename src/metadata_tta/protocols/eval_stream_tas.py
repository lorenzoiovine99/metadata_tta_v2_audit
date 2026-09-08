from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metadata_tta.config import ExperimentConfig
from metadata_tta.data import (
    DataBundle,
    YearData,
    build_split_indices,
)


@dataclass(frozen=True)
class SourceStreamYear:
    year: int

    X_supervised: np.ndarray
    y_main_supervised: np.ndarray
    y_aux_supervised: np.ndarray

    X_id: np.ndarray
    y_main_id: np.ndarray
    y_aux_id: np.ndarray


@dataclass(frozen=True)
class OODStreamYear:
    year: int

    X_reference_train: np.ndarray
    y_main_reference_train: np.ndarray
    y_aux_reference_train: np.ndarray

    X_test: np.ndarray
    y_main_test: np.ndarray
    y_aux_test: np.ndarray


@dataclass(frozen=True)
class EvalStreamTASData:
    source_years: tuple[SourceStreamYear, ...]
    ood_years: tuple[OODStreamYear, ...]


def _get_year(
    bundle: DataBundle,
    year: int,
) -> YearData:
    return bundle.year(
        int(year)
    )


def _supervised_indices_from_split(
    train: np.ndarray,
    validation: np.ndarray,
) -> np.ndarray:
    return np.sort(
        np.concatenate(
            [
                train,
                validation,
            ]
        )
    )


def build_eval_stream_tas(
    bundle: DataBundle,
    config: ExperimentConfig,
) -> EvalStreamTASData:
    """
    Eval-Stream-TAS protocol.

    Source years:
        For every source year, split 80/20.
        Train/fine-tune on the 80%.
        Evaluate ID immediately on that year's 20%.

    OOD years:
        For every OOD year, split 80/20.
        20% is the chronological OOD test stream.
        80% is used only for supervised reference models
        used by TAS.

    All test streams preserve original CSV row order by sorting
    the selected test indices.
    """

    protocol = config.section(
        "protocol"
    )
    dataset = config.section(
        "dataset"
    )

    split_config = dataset[
        "split"
    ]

    source_start = int(
        protocol[
            "source_start_year"
        ]
    )
    source_end = int(
        protocol[
            "source_end_year"
        ]
    )

    ood_start = int(
        protocol[
            "ood_start_year"
        ]
    )
    ood_end = int(
        protocol[
            "ood_end_year"
        ]
    )

    test_size = float(
        split_config[
            "test_size"
        ]
    )
    validation_size = float(
        split_config[
            "validation_size_within_train"
        ]
    )
    split_seed = int(
        split_config[
            "seed"
        ]
    )

    source_years: list[
        SourceStreamYear
    ] = []

    for year in range(
        source_start,
        source_end + 1,
    ):

        data = _get_year(
            bundle=bundle,
            year=year,
        )

        split = build_split_indices(
            y_main=data.y_main,
            test_size=test_size,
            validation_size_within_train=validation_size,
            seed=split_seed,
        )

        supervised_indices = (
            _supervised_indices_from_split(
                train=split.train,
                validation=split.validation,
            )
        )
        id_indices = np.sort(
            split.test
        )

        source_years.append(
            SourceStreamYear(
                year=year,

                X_supervised=data.X[
                    supervised_indices
                ],
                y_main_supervised=data.y_main[
                    supervised_indices
                ],
                y_aux_supervised=data.y_aux[
                    supervised_indices
                ],

                X_id=data.X[
                    id_indices
                ],
                y_main_id=data.y_main[
                    id_indices
                ],
                y_aux_id=data.y_aux[
                    id_indices
                ],
            )
        )

    ood_years: list[
        OODStreamYear
    ] = []

    for year in range(
        ood_start,
        ood_end + 1,
    ):

        data = _get_year(
            bundle=bundle,
            year=year,
        )

        split = build_split_indices(
            y_main=data.y_main,
            test_size=test_size,
            validation_size_within_train=validation_size,
            seed=split_seed,
        )

        reference_indices = (
            _supervised_indices_from_split(
                train=split.train,
                validation=split.validation,
            )
        )
        test_indices = np.sort(
            split.test
        )

        ood_years.append(
            OODStreamYear(
                year=year,

                X_reference_train=data.X[
                    reference_indices
                ],
                y_main_reference_train=data.y_main[
                    reference_indices
                ],
                y_aux_reference_train=data.y_aux[
                    reference_indices
                ],

                X_test=data.X[
                    test_indices
                ],
                y_main_test=data.y_main[
                    test_indices
                ],
                y_aux_test=data.y_aux[
                    test_indices
                ],
            )
        )

    return EvalStreamTASData(
        source_years=tuple(
            source_years
        ),
        ood_years=tuple(
            ood_years
        ),
    )