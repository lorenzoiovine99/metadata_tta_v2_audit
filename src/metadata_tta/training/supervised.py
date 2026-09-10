from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from metadata_tta.config import (
    ExperimentConfig,
)

from metadata_tta.models import (
    DoubleHeadClassifier,
    SingleHeadClassifier,
)


# ============================================================
# TRAINING RESULT
# ============================================================

@dataclass
class TrainingResult:
    """
    Result of one supervised training stage.
    """

    model: nn.Module

    optimizer: Optimizer

    epochs: int

    learning_rate: float

    mean_epoch_losses: list[float]


# ============================================================
# TEMPORAL TRAINING SCHEDULE
# ============================================================

@dataclass(frozen=True)
class TrainingSchedule:
    """
    Hyperparameters used for one temporal supervised stage.
    """

    learning_rate: float

    epochs: int

    batch_size: int

    weight_decay: float

    reset_optimizer: bool

    initialized_from_previous: bool

def _merged_training_section(
    config: ExperimentConfig,
    model_kind: str | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:

    training = config.section(
        "training"
    )

    base = dict(
        training[
            "base"
        ]
    )

    temporal = dict(
        training[
            "temporal"
        ]
    )

    if model_kind is None:
        return (
            base,
            temporal,
        )

    head_section = training.get(
        model_kind,
        {}
    )

    if not isinstance(
        head_section,
        dict,
    ):
        return (
            base,
            temporal,
        )

    # Optional per-head base-training overrides.
    for key in (
        "learning_rate",
        "weight_decay",
        "epochs",
        "batch_size",
    ):
        if key in head_section:
            base[
                key
            ] = head_section[
                key
            ]

    head_temporal = (
        head_section.get(
            "temporal",
            {},
        )
    )

    if isinstance(
        head_temporal,
        dict,
    ):
        temporal.update(
            head_temporal
        )

    return (
        base,
        temporal,
    )

def build_training_schedule(
    config: ExperimentConfig,
    initialized_from_previous: bool,
    model_kind: str | None = None,
) -> TrainingSchedule:
    """
    Determine supervised training hyperparameters.

    model_kind can be:
        None
        "single_head"
        "double_head"

    Per-head overrides are optional and backward compatible.

    Examples:
        training.single_head.learning_rate
        training.double_head.batch_size

        training.single_head.temporal.learning_rate
        training.double_head.temporal.learning_rate
    """

    (
        base,
        temporal,
    ) = _merged_training_section(
        config=config,
        model_kind=model_kind,
    )

    if (
        initialized_from_previous
        and bool(
            temporal[
                "enabled"
            ]
        )
    ):

        learning_rate = float(
            temporal[
                "learning_rate"
            ]
        )

        epochs = int(
            temporal[
                "epochs"
            ]
        )

        reset_optimizer = bool(
            temporal[
                "reset_optimizer_each_year"
            ]
        )

    else:

        learning_rate = float(
            base[
                "learning_rate"
            ]
        )

        epochs = int(
            base[
                "epochs"
            ]
        )

        reset_optimizer = True

    return TrainingSchedule(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=int(
            base[
                "batch_size"
            ]
        ),
        weight_decay=float(
            base[
                "weight_decay"
            ]
        ),
        reset_optimizer=(
            reset_optimizer
        ),
        initialized_from_previous=bool(
            initialized_from_previous
        ),
    )

def _merged_model_config(
    config: ExperimentConfig,
    model_kind: str,
) -> dict[str, Any]:

    model_config = (
        config.section(
            "model"
        )
    )

    head_overrides = (
        model_config.get(
            model_kind,
            {},
        )
    )

    if isinstance(
        head_overrides,
        dict,
    ):
        model_config.update(
            head_overrides
        )

    return model_config

# ============================================================
# MODEL CREATION
# ============================================================

def create_single_head_model(
    input_dim: int,
    config: ExperimentConfig,
) -> SingleHeadClassifier:
    """
    Create a new SingleHeadClassifier from configuration.
    """

    dataset = config.section(
        "dataset"
    )

    model_config = _merged_model_config(
        config=config,
        model_kind="single_head",
    )

    return SingleHeadClassifier(
        input_dim=int(
            input_dim
        ),
        n_classes_main=int(
            dataset[
                "n_classes_main"
            ]
        ),
        shared_hidden_dim=int(
            model_config[
                "shared_hidden_dim"
            ]
        ),
        bottleneck_dim=int(
            model_config[
                "adapter_bottleneck_dim"
            ]
        ),
        dropout=float(
            model_config[
                "dropout"
            ]
        ),
        use_shared_trunk=bool(
            model_config[
                "use_shared_trunk"
            ]
        ),
    )


def create_double_head_model(
    input_dim: int,
    config: ExperimentConfig,
) -> DoubleHeadClassifier:
    """
    Create a new DoubleHeadClassifier from configuration.
    """

    dataset = config.section(
        "dataset"
    )

    model_config = _merged_model_config(
        config=config,
        model_kind="double_head",
    )

    return DoubleHeadClassifier(
        input_dim=int(
            input_dim
        ),
        n_classes_main=int(
            dataset[
                "n_classes_main"
            ]
        ),
        n_classes_aux=int(
            dataset[
                "n_classes_aux"
            ]
        ),
        shared_hidden_dim=int(
            model_config[
                "shared_hidden_dim"
            ]
        ),
        bottleneck_dim=int(
            model_config[
                "adapter_bottleneck_dim"
            ]
        ),
        dropout=float(
            model_config[
                "dropout"
            ]
        ),
        use_shared_trunk=bool(
            model_config[
                "use_shared_trunk"
            ]
        ),
    )


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    """
    Create the supervised optimizer.

    Adam is used for the supervised models.
    """

    return Adam(
        model.parameters(),
        lr=float(
            learning_rate
        ),
        weight_decay=float(
            weight_decay
        ),
    )


def set_optimizer_learning_rate(
    optimizer: Optimizer,
    learning_rate: float,
) -> None:
    """
    Change the learning rate of an existing optimizer.
    """

    learning_rate = float(
        learning_rate
    )

    for parameter_group in (
        optimizer.param_groups
    ):
        parameter_group[
            "lr"
        ] = learning_rate


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_training_arrays(
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray | None = None,
) -> None:

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D, got {X.shape}."
        )

    if y_main.ndim != 1:
        raise ValueError(
            "y_main must be 1-D."
        )

    if len(X) != len(y_main):
        raise ValueError(
            "X and y_main must contain "
            "the same number of samples."
        )

    if y_aux is not None:

        if y_aux.ndim != 1:
            raise ValueError(
                "y_aux must be 1-D."
            )

        if len(X) != len(y_aux):
            raise ValueError(
                "X and y_aux must contain "
                "the same number of samples."
            )

    if len(X) == 0:
        raise ValueError(
            "Cannot train on an empty dataset."
        )

def _should_drop_last(
    n_samples: int,
    batch_size: int,
) -> bool:
    """
    Drop the final batch only when it would contain
    exactly one sample.

    This is required by BatchNorm1d during supervised
    training, while avoiding unnecessary sample loss.
    """

    n_samples = int(n_samples)
    batch_size = int(batch_size)

    return (
        n_samples > 1
        and batch_size > 1
        and n_samples % batch_size == 1
    )

# ============================================================
# DATA LOADERS
# ============================================================

def _single_head_loader(
    X: np.ndarray,
    y_main: np.ndarray,
    batch_size: int,
) -> DataLoader:

    dataset = TensorDataset(
        torch.as_tensor(
            X,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            y_main,
            dtype=torch.long,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=True,
        drop_last=_should_drop_last(
            n_samples=len(dataset),
            batch_size=batch_size,
        ),
    )


def _double_head_loader(
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    batch_size: int,
) -> DataLoader:

    dataset = TensorDataset(
        torch.as_tensor(
            X,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            y_main,
            dtype=torch.long,
        ),
        torch.as_tensor(
            y_aux,
            dtype=torch.long,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=True,
        drop_last=_should_drop_last(
            n_samples=len(dataset),
            batch_size=batch_size,
        ),
    )


# ============================================================
# SINGLE HEAD TRAINING
# ============================================================

def train_single_head(
    model: SingleHeadClassifier,
    X: np.ndarray,
    y_main: np.ndarray,
    schedule: TrainingSchedule,
    device: torch.device,
    optimizer: Optimizer | None = None,
) -> TrainingResult:
    """
    Train or temporally fine-tune a Single Head model.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y_main = np.asarray(
        y_main,
        dtype=np.int64,
    )

    _validate_training_arrays(
        X=X,
        y_main=y_main,
    )

    model = model.to(
        device
    )

    if (
        optimizer is None
        or schedule.reset_optimizer
    ):
        optimizer = create_optimizer(
            model=model,
            learning_rate=(
                schedule.learning_rate
            ),
            weight_decay=(
                schedule.weight_decay
            ),
        )

    else:
        set_optimizer_learning_rate(
            optimizer=optimizer,
            learning_rate=(
                schedule.learning_rate
            ),
        )

    criterion = nn.CrossEntropyLoss()

    loader = _single_head_loader(
        X=X,
        y_main=y_main,
        batch_size=(
            schedule.batch_size
        ),
    )

    mean_epoch_losses: list[
        float
    ] = []

    model.train()

    for _ in range(
        schedule.epochs
    ):

        total_loss = 0.0
        total_samples = 0

        for (
            batch_X,
            batch_y,
        ) in loader:

            batch_X = batch_X.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_X
            )

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            batch_n = int(
                len(
                    batch_X
                )
            )

            total_loss += (
                float(
                    loss.detach().item()
                )
                * batch_n
            )

            total_samples += batch_n

        mean_epoch_losses.append(
            total_loss
            / max(
                total_samples,
                1,
            )
        )

    return TrainingResult(
        model=model,
        optimizer=optimizer,
        epochs=int(
            schedule.epochs
        ),
        learning_rate=float(
            schedule.learning_rate
        ),
        mean_epoch_losses=(
            mean_epoch_losses
        ),
    )


# ============================================================
# DOUBLE HEAD TRAINING
# ============================================================

def train_double_head(
    model: DoubleHeadClassifier,
    X: np.ndarray,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    schedule: TrainingSchedule,
    aux_loss_weight: float,
    device: torch.device,
    optimizer: Optimizer | None = None,
) -> TrainingResult:
    """
    Train or temporally fine-tune a Double Head model.

    Loss:

        L = L_main + lambda_aux * L_aux
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y_main = np.asarray(
        y_main,
        dtype=np.int64,
    )

    y_aux = np.asarray(
        y_aux,
        dtype=np.int64,
    )

    _validate_training_arrays(
        X=X,
        y_main=y_main,
        y_aux=y_aux,
    )

    aux_loss_weight = float(
        aux_loss_weight
    )

    if aux_loss_weight < 0.0:
        raise ValueError(
            "aux_loss_weight must be >= 0."
        )

    model = model.to(
        device
    )

    if (
        optimizer is None
        or schedule.reset_optimizer
    ):
        optimizer = create_optimizer(
            model=model,
            learning_rate=(
                schedule.learning_rate
            ),
            weight_decay=(
                schedule.weight_decay
            ),
        )

    else:
        set_optimizer_learning_rate(
            optimizer=optimizer,
            learning_rate=(
                schedule.learning_rate
            ),
        )

    main_criterion = (
        nn.CrossEntropyLoss()
    )

    aux_criterion = (
        nn.CrossEntropyLoss()
    )

    loader = _double_head_loader(
        X=X,
        y_main=y_main,
        y_aux=y_aux,
        batch_size=(
            schedule.batch_size
        ),
    )

    mean_epoch_losses: list[
        float
    ] = []

    model.train()

    for _ in range(
        schedule.epochs
    ):

        total_loss = 0.0
        total_samples = 0

        for (
            batch_X,
            batch_main,
            batch_aux,
        ) in loader:

            batch_X = batch_X.to(
                device
            )

            batch_main = batch_main.to(
                device
            )

            batch_aux = batch_aux.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                main_logits,
                aux_logits,
            ) = model.forward_both(
                batch_X
            )

            main_loss = (
                main_criterion(
                    main_logits,
                    batch_main,
                )
            )

            aux_loss = (
                aux_criterion(
                    aux_logits,
                    batch_aux,
                )
            )

            loss = (
                main_loss
                + aux_loss_weight
                * aux_loss
            )

            loss.backward()

            optimizer.step()

            batch_n = int(
                len(
                    batch_X
                )
            )

            total_loss += (
                float(
                    loss.detach().item()
                )
                * batch_n
            )

            total_samples += (
                batch_n
            )

        mean_epoch_losses.append(
            total_loss
            / max(
                total_samples,
                1,
            )
        )

    return TrainingResult(
        model=model,
        optimizer=optimizer,
        epochs=int(
            schedule.epochs
        ),
        learning_rate=float(
            schedule.learning_rate
        ),
        mean_epoch_losses=(
            mean_epoch_losses
        ),
    )


# ============================================================
# CONFIG CONVENIENCE
# ============================================================

def get_aux_loss_weight(
    config: ExperimentConfig,
) -> float:
    """
    Read the Double Head auxiliary-loss weight.
    """

    return float(
        config.get(
            "training",
            "double_head",
            "aux_loss_weight",
        )
    )