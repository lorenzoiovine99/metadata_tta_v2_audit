from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch import nn

from metadata_tta.config import ExperimentConfig
from metadata_tta.data import build_data_bundle
from metadata_tta.evaluation import (
    EvaluationRecord,
    evaluate_frozen_year,
    evaluate_tta_stream,
)
from metadata_tta.experiment.runner import (
    ExperimentResult,
    _build_double_model,
    _build_single_model,
    _device,
)
from metadata_tta.protocols import (
    EvalStreamTASData,
    OODStreamYear,
    SourceStreamYear,
    build_eval_stream_tas,
)
from metadata_tta.reproducibility import set_seed
from metadata_tta.results import (
    ResultsWriter,
    build_ood_summary,
)
from metadata_tta.training import (
    TrainingSchedule,
    build_training_schedule,
    train_double_head,
    train_single_head,
)
from metadata_tta.tta import get_method_class


def _method_family(
    method_name: str,
) -> str:
    if method_name in {
        "single_head_frozen",
        "tent",
    }:
        return "single"

    return "double"


def _concat_year_arrays(
    years: list[OODStreamYear],
    field_name: str,
) -> np.ndarray:
    arrays = [
        getattr(
            year,
            field_name,
        )
        for year in years
    ]

    if not arrays:
        raise ValueError(
            "Cannot concatenate an empty year list."
        )

    return np.concatenate(
        arrays,
        axis=0,
    )


def _train_source_models_with_yearly_id(
    protocol_data: EvalStreamTASData,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[
    nn.Module | None,
    nn.Module | None,
    list[EvaluationRecord],
]:

    first_year = (
        protocol_data.source_years[0]
    )

    input_dim = int(
        first_year.X_supervised.shape[1]
    )

    single_needed = (
        config.baseline_enabled(
            "single_head"
        )
        or config.method_enabled(
            "tent"
        )
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

    single_model = (
        _build_single_model(
            input_dim=input_dim,
            config=config,
        ).to(device)
        if single_needed
        else None
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
        config.get(
            "training",
            "double_head",
            "aux_loss_weight",
        )
    )

    id_records: list[
        EvaluationRecord
    ] = []

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
            f"train_n={len(year_data.y_main_supervised)} | "
            f"id_n={len(year_data.y_main_id)} | "
            f"epochs={schedule.epochs} | "
            f"lr={schedule.learning_rate:g}"
        )

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

            if config.baseline_enabled(
                "single_head"
            ):
                id_records.append(
                    evaluate_frozen_year(
                        model=single_model,
                        X=year_data.X_id,
                        y_main=year_data.y_main_id,
                        year=year_data.year,
                        method_name=(
                            "single_head_frozen"
                        ),
                        device=device,
                    )
                )

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

            if config.baseline_enabled(
                "double_head"
            ):
                id_records.append(
                    evaluate_frozen_year(
                        model=double_model,
                        X=year_data.X_id,
                        y_main=year_data.y_main_id,
                        year=year_data.year,
                        method_name=(
                            "double_head_frozen"
                        ),
                        device=device,
                    )
                )

    if single_model is not None:
        single_model.eval()

    if double_model is not None:
        double_model.eval()

    return (
        single_model,
        double_model,
        id_records,
    )


def _fine_tune_single_reference(
    source_model: nn.Module,
    ood_years: list[OODStreamYear],
    config: ExperimentConfig,
    device: torch.device,
) -> nn.Module:

    model = copy.deepcopy(
        source_model
    ).to(device)

    optimizer = None

    for year_data in ood_years:

        schedule = build_training_schedule(
            config=config,
            initialized_from_previous=True,
        )

        result = train_single_head(
            model=model,
            X=year_data.X_reference_train,
            y_main=(
                year_data.y_main_reference_train
            ),
            schedule=schedule,
            device=device,
            optimizer=optimizer,
        )

        model = result.model
        optimizer = result.optimizer

    model.eval()

    return model


def _fine_tune_double_reference(
    source_model: nn.Module,
    ood_years: list[OODStreamYear],
    config: ExperimentConfig,
    device: torch.device,
) -> nn.Module:

    model = copy.deepcopy(
        source_model
    ).to(device)

    optimizer = None

    aux_loss_weight = float(
        config.get(
            "training",
            "double_head",
            "aux_loss_weight",
        )
    )

    for year_data in ood_years:

        schedule = build_training_schedule(
            config=config,
            initialized_from_previous=True,
        )

        result = train_double_head(
            model=model,
            X=year_data.X_reference_train,
            y_main=(
                year_data.y_main_reference_train
            ),
            y_aux=(
                year_data.y_aux_reference_train
            ),
            schedule=schedule,
            aux_loss_weight=(
                aux_loss_weight
            ),
            device=device,
            optimizer=optimizer,
        )

        model = result.model
        optimizer = result.optimizer

    model.eval()

    return model


def _evaluate_supervised_references(
    source_single_model: nn.Module | None,
    source_double_model: nn.Module | None,
    protocol_data: EvalStreamTASData,
    config: ExperimentConfig,
    device: torch.device,
) -> list[EvaluationRecord]:

    records: list[
        EvaluationRecord
    ] = []

    cumulative_years: list[
        OODStreamYear
    ] = []

    for year_data in protocol_data.ood_years:

        cumulative_years.append(
            year_data
        )

        print()
        print(
            f"Supervised TAS reference up to "
            f"{year_data.year}"
        )

        if source_single_model is not None:

            single_reference = (
                _fine_tune_single_reference(
                    source_model=source_single_model,
                    ood_years=cumulative_years,
                    config=config,
                    device=device,
                )
            )

            records.append(
                evaluate_frozen_year(
                    model=single_reference,
                    X=year_data.X_test,
                    y_main=year_data.y_main_test,
                    year=year_data.year,
                    method_name=(
                        "single_head_supervised_reference"
                    ),
                    device=device,
                )
            )

        if source_double_model is not None:

            double_reference = (
                _fine_tune_double_reference(
                    source_model=source_double_model,
                    ood_years=cumulative_years,
                    config=config,
                    device=device,
                )
            )

            records.append(
                evaluate_frozen_year(
                    model=double_reference,
                    X=year_data.X_test,
                    y_main=year_data.y_main_test,
                    year=year_data.year,
                    method_name=(
                        "double_head_supervised_reference"
                    ),
                    device=device,
                )
            )

    return records


def _build_tas_rows(
    ood_records: list[EvaluationRecord],
    supervised_reference_records: list[EvaluationRecord],
) -> list[dict[str, Any]]:

    accuracy_by_key = {
        (
            record.method,
            record.year,
        ): float(
            record.accuracy
        )
        for record in (
            list(ood_records)
            + list(
                supervised_reference_records
            )
        )
    }

    rows: list[
        dict[str, Any]
    ] = []

    for record in ood_records:

        family = _method_family(
            record.method
        )

        if family == "single":
            frozen_method = (
                "single_head_frozen"
            )
            supervised_method = (
                "single_head_supervised_reference"
            )
        else:
            frozen_method = (
                "double_head_frozen"
            )
            supervised_method = (
                "double_head_supervised_reference"
            )

        frozen_accuracy = accuracy_by_key.get(
            (
                frozen_method,
                record.year,
            )
        )
        supervised_accuracy = accuracy_by_key.get(
            (
                supervised_method,
                record.year,
            )
        )

        tas_defined = False
        tas_value: float | str = ""

        if (
            frozen_accuracy is not None
            and supervised_accuracy is not None
        ):
            denominator = (
                supervised_accuracy
                - frozen_accuracy
            )

            if abs(denominator) > 1.0e-12:
                tas_value = (
                    float(record.accuracy)
                    - frozen_accuracy
                ) / denominator
                tas_defined = True

        rows.append(
            {
                "year":
                    int(record.year),

                "method":
                    record.method,

                "family":
                    family,

                "accuracy":
                    float(record.accuracy),

                "frozen_reference_method":
                    frozen_method,

                "frozen_reference_accuracy":
                    (
                        ""
                        if frozen_accuracy is None
                        else float(frozen_accuracy)
                    ),

                "supervised_reference_method":
                    supervised_method,

                "supervised_reference_accuracy":
                    (
                        ""
                        if supervised_accuracy is None
                        else float(supervised_accuracy)
                    ),

                "tas":
                    tas_value,

                "tas_defined":
                    bool(tas_defined),
            }
        )

    return rows


def run_eval_stream_tas(
    config: ExperimentConfig,
) -> ExperimentResult:

    set_seed(
        config.seed
    )

    device = _device()

    print(
        f"Device: {device}"
    )
    print(
        "Protocol: eval_stream_tas"
    )

    bundle = build_data_bundle(
        config
    )

    protocol_data = build_eval_stream_tas(
        bundle=bundle,
        config=config,
    )

    (
        single_model,
        double_model,
        id_records,
    ) = _train_source_models_with_yearly_id(
        protocol_data=protocol_data,
        config=config,
        device=device,
    )

    if (
        config.method_enabled(
            "tent"
        )
        and single_model is None
    ):
        raise RuntimeError(
            "TENT requires a SingleHead source model."
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
            "DoubleHead source model is required "
            "for double-head baselines and metadata TTA."
        )

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

            "source_id_policy":
                (
                    "evaluate each source-year held-out "
                    "split immediately after training that year"
                ),

            "ood_policy":
                (
                    "per-year 80/20 split; 20% test stream; "
                    "80% supervised TAS reference pool"
                ),

            "tta_order":
                "predict_then_update",

            "tta_state":
                "carries_across_ood_years",

            "tent_source_model":
                "single_head",

            "metadata_tta_source_model":
                "double_head",

            "pca_active":
                False,
        }
    )

    ood_records: list[
        EvaluationRecord
    ] = []

    # ========================================================
    # OOD FROZEN BASELINES
    # ========================================================

    for year_data in protocol_data.ood_years:

        print()
        print(
            f"OOD frozen year {year_data.year} | "
            f"test_n={len(year_data.y_main_test)}"
        )

        if (
            single_model is not None
            and config.baseline_enabled(
                "single_head"
            )
        ):
            ood_records.append(
                evaluate_frozen_year(
                    model=single_model,
                    X=year_data.X_test,
                    y_main=(
                        year_data.y_main_test
                    ),
                    year=year_data.year,
                    method_name=(
                        "single_head_frozen"
                    ),
                    device=device,
                )
            )

        if (
            double_model is not None
            and config.baseline_enabled(
                "double_head"
            )
        ):
            ood_records.append(
                evaluate_frozen_year(
                    model=double_model,
                    X=year_data.X_test,
                    y_main=(
                        year_data.y_main_test
                    ),
                    year=year_data.year,
                    method_name=(
                        "double_head_frozen"
                    ),
                    device=device,
                )
            )

    # ========================================================
    # TTA METHODS
    # ========================================================

    stream = [
        (
            year_data.year,
            year_data.X_test,
            year_data.y_main_test,
            year_data.y_aux_test,
        )
        for year_data
        in protocol_data.ood_years
    ]

    for method_name in config.enabled_methods:

        method_class = get_method_class(
            method_name
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

        print()
        print(
            f"TTA method {method_name} | "
            "state carries across OOD years"
        )

        method = method_class(
            source_model=source_model,
            config=method_config,
            device=device,
        )

        result = evaluate_tta_stream(
            method=method,
            years=stream,
            reset_each_year=False,
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
                method_name=method_name,
                diagnostics=result.diagnostics,
            )

    # ========================================================
    # SUPERVISED TAS REFERENCES
    # ========================================================

    supervised_reference_records = (
        _evaluate_supervised_references(
            source_single_model=single_model,
            source_double_model=double_model,
            protocol_data=protocol_data,
            config=config,
            device=device,
        )
    )

    tas_rows = _build_tas_rows(
        ood_records=ood_records,
        supervised_reference_records=(
            supervised_reference_records
        ),
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

    writer.write_records(
        "supervised_reference.csv",
        supervised_reference_records,
    )

    writer.write_tas_rows(
        tas_rows
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
        reference_method=reference_method,
    )

    writer.write_summary(
        summary
    )

    print(
        f"Results: {run_directory}"
    )

    return ExperimentResult(
        run_directory=run_directory,
        id_records=tuple(
            id_records
        ),
        ood_records=tuple(
            ood_records
        ),
    )