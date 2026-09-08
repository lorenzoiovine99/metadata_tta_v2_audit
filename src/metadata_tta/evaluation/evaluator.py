from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from metadata_tta.tta import TTAMethod

from .metrics import accuracy
from .records import (
    EvaluationRecord,
    EvaluationResult,
)


def _as_numpy(
    value: np.ndarray | torch.Tensor,
) -> np.ndarray:

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _predict_frozen(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:

    model.eval()

    x_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        logits = model(x_tensor)

    return (
        logits.argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )


def evaluate_frozen_year(
    model: nn.Module,
    X: np.ndarray,
    y_main: np.ndarray,
    year: int,
    method_name: str,
    device: torch.device,
) -> EvaluationRecord:

    predictions = _predict_frozen(
        model=model,
        X=X,
        device=device,
    )

    return EvaluationRecord(
        year=int(year),
        method=str(method_name),
        accuracy=accuracy(
            y_main,
            predictions,
        ),
        n_samples=int(len(y_main)),
    )


def _samplewise_tta_predictions(
    method: TTAMethod,
    X: np.ndarray,
    y_aux: np.ndarray | None,
) -> np.ndarray:
    """
    Strict prequential evaluation:

        predict x_t
        record prediction
        observe metadata_t
        update

    The update can never influence the prediction for t.
    """

    predictions: list[int] = []

    for index in range(len(X)):

        x = X[index]

        # --------------------------------------------
        # 1. PREDICT
        # --------------------------------------------

        logits = method.predict_logits(
            x
        )

        prediction = int(
            logits.argmax(
                dim=1
            )[0].item()
        )

        # --------------------------------------------
        # 2. RECORD
        # --------------------------------------------

        predictions.append(
            prediction
        )

        # --------------------------------------------
        # 3. OBSERVE / UPDATE
        # --------------------------------------------

        auxiliary_label = (
            None
            if y_aux is None
            else int(
                y_aux[index]
            )
        )

        method.observe(
            x=x,
            y_aux=auxiliary_label,
        )

    return np.asarray(
        predictions,
        dtype=np.int64,
    )


def _tent_batch_ranges(
    n_samples: int,
    batch_size: int,
) -> list[tuple[int, int]]:
    """
    Build contiguous temporal batches for TENT.

    BatchNorm cannot process a final singleton batch. If the
    final remainder is one sample, that sample is merged into
    the preceding batch.

    No reordering occurs.
    """

    if n_samples < 2:
        raise ValueError(
            "TENT requires at least two samples "
            "in an evaluated stream."
        )

    if batch_size < 2:
        raise ValueError(
            "TENT batch_size must be >= 2."
        )

    ranges: list[
        tuple[int, int]
    ] = []

    start = 0

    while start < n_samples:

        end = min(
            start + batch_size,
            n_samples,
        )

        remainder = (
            n_samples
            - end
        )

        if remainder == 1:
            end = n_samples

        ranges.append(
            (
                start,
                end,
            )
        )

        start = end

    return ranges


def _batchwise_tta_predictions(
    method: TTAMethod,
    X: np.ndarray,
) -> np.ndarray:
    """
    Strict batch-prequential TENT:

        predict batch B_t
        record predictions B_t
        update on B_t

    Temporal order inside the dataset is preserved.
    """

    batch_size = int(
        getattr(
            method,
            "batch_size",
            64,
        )
    )

    predictions: list[
        np.ndarray
    ] = []

    for start, end in _tent_batch_ranges(
        n_samples=len(X),
        batch_size=batch_size,
    ):

        x_batch = X[
            start:end
        ]

        # --------------------------------------------
        # 1. PREDICT
        # --------------------------------------------

        logits = method.predict_logits(
            x_batch
        )

        batch_predictions = (
            logits.argmax(
                dim=1
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        # --------------------------------------------
        # 2. RECORD
        # --------------------------------------------

        predictions.append(
            batch_predictions
        )

        # --------------------------------------------
        # 3. OBSERVE / UPDATE
        # --------------------------------------------

        method.observe(
            x=x_batch,
            y_aux=None,
        )

    return np.concatenate(
        predictions,
        axis=0,
    )


def evaluate_tta_year(
    method: TTAMethod,
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray | None,
    year: int,
) -> EvaluationRecord:
    """
    Evaluate one chronological year.

    y_main is used ONLY after predictions have been produced,
    for metric computation. It is never passed to the TTA
    method.
    """

    X = _as_numpy(X).astype(
        np.float32,
        copy=False,
    )

    y_main = _as_numpy(
        y_main
    ).astype(
        np.int64,
        copy=False,
    )

    if y_aux is not None:
        y_aux = _as_numpy(
            y_aux
        ).astype(
            np.int64,
            copy=False,
        )

    if len(X) != len(y_main):
        raise ValueError(
            "X and y_main must have equal length."
        )

    if (
        y_aux is not None
        and len(X) != len(y_aux)
    ):
        raise ValueError(
            "X and y_aux must have equal length."
        )

    if len(X) == 0:
        raise ValueError(
            "Cannot evaluate an empty year."
        )

    if (
        method.requires_aux_labels
        and y_aux is None
    ):
        raise ValueError(
            f"{method.method_name} requires auxiliary labels."
        )

    # TENT is the stable batch-wise method.
    if method.method_name == "tent":

        predictions = (
            _batchwise_tta_predictions(
                method=method,
                X=X,
            )
        )

    else:

        predictions = (
            _samplewise_tta_predictions(
                method=method,
                X=X,
                y_aux=y_aux,
            )
        )

    if len(predictions) != len(y_main):
        raise RuntimeError(
            "Evaluator produced an incorrect "
            "number of predictions."
        )

    return EvaluationRecord(
        year=int(year),
        method=method.method_name,
        accuracy=accuracy(
            y_main,
            predictions,
        ),
        n_samples=int(
            len(y_main)
        ),
    )


def evaluate_tta_stream(
    method: TTAMethod,
    years: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
        ]
    ],
    reset_each_year: bool,
) -> EvaluationResult:
    """
    Evaluate a chronological multi-year stream.

    State carries across years unless reset_each_year=True.
    """

    records: list[
        EvaluationRecord
    ] = []

    for year_index, (
        year,
        X,
        y_main,
        y_aux,
    ) in enumerate(years):

        if (
            reset_each_year
            and year_index > 0
        ):
            method.reset()

        record = evaluate_tta_year(
            method=method,
            X=X,
            y_main=y_main,
            y_aux=y_aux,
            year=year,
        )

        records.append(
            record
        )

    return EvaluationResult(
        records=tuple(
            records
        ),
        diagnostics=method.diagnostics(),
    )