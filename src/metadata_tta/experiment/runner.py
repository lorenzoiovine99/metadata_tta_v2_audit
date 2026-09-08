from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from metadata_tta.config import (
    ExperimentConfig,
)
from metadata_tta.data import (
    build_data_bundle,
)
from metadata_tta.evaluation import (
    EvaluationRecord,
    evaluate_frozen_year,
    evaluate_tta_stream,
)
from metadata_tta.models import (
    DoubleHeadClassifier,
    SingleHeadClassifier,
)
from metadata_tta.protocols import (
    build_eval_fix,
)
from metadata_tta.reproducibility import (
    set_seed,
)
from metadata_tta.results import (
    ResultsWriter,
    build_ood_summary,
)
from metadata_tta.training import (
    build_training_schedule,
    create_optimizer,
    train_double_head,
    train_single_head,
)
from metadata_tta.tta import (
    get_method_class,
)


@dataclass(frozen=True)
class ExperimentResult:
    run_directory: Path
    id_records: tuple[
        EvaluationRecord,
        ...
    ]
    ood_records: tuple[
        EvaluationRecord,
        ...
    ]


def _device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def _build_single_model(
    input_dim: int,
    config: ExperimentConfig,
) -> SingleHeadClassifier:

    model_config = config.section(
        "model"
    )

    dataset_config = config.section(
        "dataset"
    )

    return SingleHeadClassifier(
        input_dim=input_dim,
        n_classes_main=int(
            dataset_config[
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


def _build_double_model(
    input_dim: int,
    config: ExperimentConfig,
) -> DoubleHeadClassifier:

    model_config = config.section(
        "model"
    )

    dataset_config = config.section(
        "dataset"
    )

    return DoubleHeadClassifier(
        input_dim=input_dim,
        n_classes_main=int(
            dataset_config[
                "n_classes_main"
            ]
        ),
        n_classes_aux=int(
            dataset_config[
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

def _train_source_models(
    protocol_data: Any,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[
    nn.Module | None,
    nn.Module | None,
]:

    first_year = (
        protocol_data.source_years[0]
    )

    input_dim = int(
        first_year.X_supervised.shape[1]
    )

    # ========================================================
    # MODEL CREATION
    # ========================================================

    single_needed = (
        config.baseline_enabled(
            "single_head"
        )
        or config.method_enabled(
            "tent"
        )
    )

    single_model = (
        _build_single_model(
            input_dim=input_dim,
            config=config,
        ).to(device)
        if single_needed
        else None
    )

    double_needed = (
        config.baseline_enabled(
            "double_head"
        )
        or any(
            method_name != "tent"
            for method_name
            in config.enabled_methods
        )
    )

    double_model = (
        _build_double_model(
            input_dim=input_dim,
            config=config,
        ).to(device)
        if double_needed
        else None
    )

    single_optimizer = None
    double_optimizer = None

    aux_loss_weight = float(
        config.section(
            "training"
        )[
            "double_head"
        ][
            "aux_loss_weight"
        ]
    )

    # ========================================================
    # TEMPORAL SOURCE TRAINING
    # ========================================================

    for year_index, year_data in enumerate(
        protocol_data.source_years
    ):

        initialized_from_previous = (
            year_index > 0
        )

        schedule = build_training_schedule(
            config=config,
            initialized_from_previous=(
                initialized_from_previous
            ),
        )

        print()
        print(
            f"Source year {year_data.year} | "
            f"n={len(year_data.y_main_supervised)} | "
            f"epochs={schedule.epochs} | "
            f"lr={schedule.learning_rate:g}"
        )

        # ----------------------------------------------------
        # SINGLE HEAD
        # ----------------------------------------------------

        if single_model is not None:

            result = train_single_head(
                model=single_model,
                X=year_data.X_supervised,
                y_main=(
                    year_data.y_main_supervised
                ),
                schedule=schedule,
                device=device,
                optimizer=single_optimizer,
            )

            single_model = result.model
            single_optimizer = (
                result.optimizer
            )

            print(
                "  Single Head loss:",
                f"{result.mean_epoch_losses[-1]:.6f}",
            )

        # ----------------------------------------------------
        # DOUBLE HEAD
        # ----------------------------------------------------

        if double_model is not None:

            result = train_double_head(
                model=double_model,
                X=year_data.X_supervised,
                y_main=(
                    year_data.y_main_supervised
                ),
                y_aux=(
                    year_data.y_aux_supervised
                ),
                schedule=schedule,
                aux_loss_weight=(
                    aux_loss_weight
                ),
                device=device,
                optimizer=double_optimizer,
            )

            double_model = result.model
            double_optimizer = (
                result.optimizer
            )

            print(
                "  Double Head loss:",
                f"{result.mean_epoch_losses[-1]:.6f}",
            )

    # ========================================================
    # FREEZE SOURCE TRAINING PHASE
    # ========================================================

    if single_model is not None:
        single_model.eval()

    if double_model is not None:
        double_model.eval()

    return (
        single_model,
        double_model,
    )

def run_experiment(
    config: ExperimentConfig,
) -> ExperimentResult:

    if config.protocol_name == "eval_stream_tas":
        from metadata_tta.experiment.eval_stream_tas_runner import (
            run_eval_stream_tas,
        )

        return run_eval_stream_tas(
            config
        )

    if config.protocol_name != "eval_fix":
        raise NotImplementedError(
            "The V2 runner currently executes "
            "Eval-Fix and Eval-Stream-TAS only."
        )

    set_seed(
        config.seed
    )

    device = _device()

    print(
        f"Device: {device}"
    )

    # ========================================================
    # DATA + PROTOCOL
    # ========================================================

    bundle = build_data_bundle(
        config
    )

    protocol_data = build_eval_fix(
        bundle=bundle,
        config=config,
    )

    # ========================================================
    # SOURCE TRAINING
    # ========================================================

    single_model, double_model = (
        _train_source_models(
            protocol_data=
                protocol_data,
            config=config,
            device=device,
        )
    )

    double_required = (
        config.baseline_enabled(
            "double_head"
        )
        or any(
            method_name != "tent"
            for method_name
            in config.enabled_methods
        )
    )

    if (
        double_required
        and double_model is None
    ):
        raise RuntimeError(
            "Double-head source model is required "
            "for double-head baselines and metadata TTA methods."
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_directory = (
        config.results_root
        / config.dataset_name
        / config.protocol_name
        / (
            f"{config.experiment_name}_"
            f"{timestamp}"
        )
    )

    writer = ResultsWriter(
        run_directory
    )

    writer.write_config(
        config.as_dict()
    )

    writer.write_manifest(
        {
            "experiment":
                config.experiment_name,
            "dataset":
                config.dataset_name,
            "protocol":
                config.protocol_name,
            "device":
                str(device),
            "seed":
                config.seed,
            "tta_order":
                "predict_then_update",
            "pca_active":
                False,
        }
    )

    id_records: list[
        EvaluationRecord
    ] = []

    ood_records: list[
        EvaluationRecord
    ] = []

    # ========================================================
    # ID
    # ========================================================

    id_data = (
        protocol_data.id_year
    )

    if single_model is not None:

        id_records.append(
            evaluate_frozen_year(
                model=single_model,
                X=id_data.X,
                y_main=id_data.y_main,
                year=id_data.year,
                method_name=
                    "single_head_frozen",
                device=device,
            )
        )

    if config.baseline_enabled(
        "double_head"
    ):

        id_records.append(
            evaluate_frozen_year(
                model=double_model,
                X=id_data.X,
                y_main=id_data.y_main,
                year=id_data.year,
                method_name=
                    "double_head_frozen",
                device=device,
            )
        )

    # ========================================================
    # OOD FROZEN BASELINES
    # ========================================================

    for year_data in (
        protocol_data.ood_years
    ):

        if single_model is not None:

            ood_records.append(
                evaluate_frozen_year(
                    model=single_model,
                    X=year_data.X,
                    y_main=
                        year_data.y_main,
                    year=year_data.year,
                    method_name=
                        "single_head_frozen",
                    device=device,
                )
            )

        if config.baseline_enabled(
            "double_head"
        ):

            ood_records.append(
                evaluate_frozen_year(
                    model=double_model,
                    X=year_data.X,
                    y_main=
                        year_data.y_main,
                    year=year_data.year,
                    method_name=
                        "double_head_frozen",
                    device=device,
                )
            )

    # ========================================================
    # TTA METHODS
    # ========================================================

    stream = [
        (
            year_data.year,
            year_data.X,
            year_data.y_main,
            year_data.y_aux,
        )
        for year_data
        in protocol_data.ood_years
    ]

    methods_config = config.section(
        "methods"
    )

    for method_name in (
        config.enabled_methods
    ):

        method_class = (
            get_method_class(
                method_name
            )
        )

        method_config = copy.deepcopy(
            config.section(
                method_name
            )
        )

        if method_name == "tent":

            if single_model is None:
                raise RuntimeError(
                    "TENT requires a SingleHead "
                    "source model."
                )

            source_model = single_model

        else:

            if double_model is None:
                raise RuntimeError(
                    f"{method_name} requires a "
                    "DoubleHead source model."
                )

            source_model = double_model

        method = method_class(
            source_model=source_model,
            config=method_config,
            device=device,
        )

        reset_each_year = bool(
            method_config.get(
                "reset_each_year",
                False,
            )
        )

        result = evaluate_tta_stream(
            method=method,
            years=stream,
            reset_each_year=
                reset_each_year,
        )

        ood_records.extend(
            result.records
        )

        if config.section(
            "output"
        ).get(
            "save_diagnostics",
            True,
        ):
            writer.write_diagnostics(
                method_name=
                    method_name,
                diagnostics=
                    result.diagnostics,
            )

    # ========================================================
    # WRITE RESULTS
    # ========================================================

    writer.write_records(
        "id.csv",
        id_records,
    )

    writer.write_records(
        "ood.csv",
        ood_records,
    )

    reference_method = (
        "double_head_frozen"
        if config.baseline_enabled(
            "double_head"
        )
        else "single_head_frozen"
    )

    summary = build_ood_summary(
        records=ood_records,
        reference_method=
            reference_method,
    )

    writer.write_summary(
        summary
    )

    print(
        f"Results: {run_directory}"
    )

    return ExperimentResult(
        run_directory=
            run_directory,
        id_records=tuple(
            id_records
        ),
        ood_records=tuple(
            ood_records
        ),
    )