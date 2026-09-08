from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class SplitIndices:
    """
    Indices defining one deterministic yearly split.

    Keeping indices rather than only copied arrays lets us
    preserve the canonical full yearly stream.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


# ============================================================
# STRATIFICATION
# ============================================================

def _safe_stratify_labels(
    labels: np.ndarray,
) -> np.ndarray | None:
    """
    Return labels for stratification when every represented
    class contains enough samples.

    Otherwise return None so sklearn can perform a regular
    shuffled split instead of failing.
    """

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    if len(labels) == 0:
        return None

    _, counts = np.unique(
        labels,
        return_counts=True,
    )

    if len(counts) == 0:
        return None

    if int(counts.min()) < 2:
        return None

    return labels


# ============================================================
# SPLIT INDICES
# ============================================================

def build_split_indices(
    y_main: np.ndarray,
    test_size: float,
    validation_size_within_train: float,
    seed: int,
) -> SplitIndices:
    """
    Build deterministic train / validation / test indices.

    Split procedure
    ---------------
    1. Split the complete year into:

           development
           test

    2. Split development into:

           train
           validation

    Stratification uses the main-task label whenever possible.
    """

    y_main = np.asarray(
        y_main,
        dtype=np.int64,
    )

    n_samples = len(
        y_main
    )

    if n_samples < 2:
        raise ValueError(
            "At least two samples are required "
            "to create a train/test split."
        )

    indices = np.arange(
        n_samples,
        dtype=np.int64,
    )

    stratify_all = (
        _safe_stratify_labels(
            y_main
        )
    )

    development_indices, test_indices = (
        train_test_split(
            indices,
            test_size=float(
                test_size
            ),
            random_state=int(
                seed
            ),
            shuffle=True,
            stratify=stratify_all,
        )
    )

    development_indices = np.asarray(
        development_indices,
        dtype=np.int64,
    )

    test_indices = np.asarray(
        test_indices,
        dtype=np.int64,
    )

    validation_fraction = float(
        validation_size_within_train
    )

    if validation_fraction == 0.0:
        train_indices = (
            development_indices
        )

        validation_indices = np.empty(
            0,
            dtype=np.int64,
        )

    else:
        development_labels = y_main[
            development_indices
        ]

        stratify_development = (
            _safe_stratify_labels(
                development_labels
            )
        )

        (
            train_indices,
            validation_indices,
        ) = train_test_split(
            development_indices,
            test_size=validation_fraction,
            random_state=int(
                seed
            ),
            shuffle=True,
            stratify=stratify_development,
        )

        train_indices = np.asarray(
            train_indices,
            dtype=np.int64,
        )

        validation_indices = np.asarray(
            validation_indices,
            dtype=np.int64,
        )

    return SplitIndices(
        train=train_indices,
        validation=validation_indices,
        test=test_indices,
    )


# ============================================================
# MATERIALIZE SPLIT
# ============================================================

def materialize_split(
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    split: SplitIndices,
) -> dict[str, np.ndarray]:
    """
    Materialize arrays corresponding to SplitIndices.
    """

    return {
        "X_train":
            X[
                split.train
            ],

        "y_main_train":
            y_main[
                split.train
            ],

        "y_aux_train":
            y_aux[
                split.train
            ],

        "X_validation":
            X[
                split.validation
            ],

        "y_main_validation":
            y_main[
                split.validation
            ],

        "y_aux_validation":
            y_aux[
                split.validation
            ],

        "X_test":
            X[
                split.test
            ],

        "y_main_test":
            y_main[
                split.test
            ],

        "y_aux_test":
            y_aux[
                split.test
            ],
    }


# ============================================================
# CONVENIENCE WRAPPER
# ============================================================

def split_year_data(
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    test_size: float,
    validation_size_within_train: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Convenience wrapper equivalent to:

        build_split_indices(...)
        materialize_split(...)
    """

    split = build_split_indices(
        y_main=y_main,
        test_size=test_size,
        validation_size_within_train=(
            validation_size_within_train
        ),
        seed=seed,
    )

    return materialize_split(
        X=X,
        y_main=y_main,
        y_aux=y_aux,
        split=split,
    )


# ============================================================
# SOURCE SUPERVISION
# ============================================================

def concatenate_train_and_validation(
    split_data: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return the complete supervised source subset.

    With the standard configuration:

        test_size = 0.20
        validation_size_within_train = 0.125

    this corresponds to 80% of the complete year.
    """

    X = np.concatenate(
        [
            split_data[
                "X_train"
            ],
            split_data[
                "X_validation"
            ],
        ],
        axis=0,
    )

    y_main = np.concatenate(
        [
            split_data[
                "y_main_train"
            ],
            split_data[
                "y_main_validation"
            ],
        ],
        axis=0,
    )

    y_aux = np.concatenate(
        [
            split_data[
                "y_aux_train"
            ],
            split_data[
                "y_aux_validation"
            ],
        ],
        axis=0,
    )

    return (
        X,
        y_main,
        y_aux,
    )