from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from metadata_tta.config import (
    ExperimentConfig,
)
from metadata_tta.data import (
    DataBundle,
    build_data_bundle,
    build_split_indices,
)
from metadata_tta.evaluation import (
    evaluate_frozen_year,
    evaluate_tta_stream,
)
from metadata_tta.experiment.runner import (
    _device,
)
from metadata_tta.reproducibility import (
    set_seed,
)
from metadata_tta.training import (
    build_training_schedule,
    create_double_head_model,
    create_single_head_model,
    get_aux_loss_weight,
    train_double_head,
    train_single_head,
)
from metadata_tta.tta import (
    get_method_class,
)

from .config import (
    apply_overrides,
)
from .inheritance import (
    inherited_overrides_from_best,
    merge_overrides,
)
from .results import (
    ensure_directory,
    load_best_result,
    save_best_result,
    save_trial_rows,
)
from .search import (
    build_search_strategy,
)


@dataclass(frozen=True)
class TTATuningResult:
    method_name: str
    best_score: float
    best_parameters: dict[str, Any]
    best_overrides: dict[str, Any]
    output_directory: Path


# ============================================================
# MODEL FAMILY
# ============================================================


def _method_family(
    method_name: str,
) -> str:

    if method_name == "tent":
        return "single_head"

    if method_name in {
        "metadata",
        "temporal_gradient",
        "consensus",
    }:
        return "double_head"

    raise ValueError(
        f"Unsupported tuning method: "
        f"{method_name}"
    )


# ============================================================
# YEAR SPLIT
# ============================================================


def _split_for_year(
    bundle: DataBundle,
    config: ExperimentConfig,
    year: int,
):

    data = bundle.year(
        year
    )

    split_config = config.get(
        "dataset",
        "split",
    )

    split = build_split_indices(
        y_main=data.y_main,
        test_size=float(
            split_config[
                "test_size"
            ]
        ),
        validation_size_within_train=float(
            split_config[
                "validation_size_within_train"
            ]
        ),
        seed=int(
            split_config[
                "seed"
            ]
        ),
    )

    return (
        data,
        split,
    )


# ============================================================
# SOURCE TRAINING DATA
# ============================================================


def _source_supervised_arrays(
    bundle: DataBundle,
    config: ExperimentConfig,
    year: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    data, split = (
        _split_for_year(
            bundle=bundle,
            config=config,
            year=year,
        )
    )

    # During TTA hyperparameter tuning the years BEFORE the
    # pseudo-OOD boundary are source years.
    #
    # Their train + validation portions can therefore be used
    # to train the pseudo-source model.
    supervised_indices = np.sort(
        np.concatenate(
            [
                split.train,
                split.validation,
            ]
        )
    )

    return (
        data.X[
            supervised_indices
        ],
        data.y_main[
            supervised_indices
        ],
        data.y_aux[
            supervised_indices
        ],
    )


# ============================================================
# SOURCE MODEL
# ============================================================


def _train_source_model(
    *,
    family: str,
    config: ExperimentConfig,
    bundle: DataBundle,
    source_years: list[int],
    device: torch.device,
) -> nn.Module:

    if not source_years:
        raise ValueError(
            "TTA tuning requires at least "
            "one supervised source year."
        )

    if family == "single_head":

        model = (
            create_single_head_model(
                input_dim=bundle.feature_dim,
                config=config,
            )
            .to(device)
        )

    elif family == "double_head":

        model = (
            create_double_head_model(
                input_dim=bundle.feature_dim,
                config=config,
            )
            .to(device)
        )

    else:
        raise ValueError(
            f"Unknown family: {family}"
        )

    optimizer = None

    aux_loss_weight = (
        get_aux_loss_weight(
            config
        )
        if family == "double_head"
        else None
    )

    for index, year in enumerate(
        source_years
    ):

        (
            X,
            y_main,
            y_aux,
        ) = _source_supervised_arrays(
            bundle=bundle,
            config=config,
            year=year,
        )

        schedule = (
            build_training_schedule(
                config=config,
                initialized_from_previous=(
                    index > 0
                ),
                model_kind=family,
            )
        )

        if family == "single_head":

            result = train_single_head(
                model=model,
                X=X,
                y_main=y_main,
                schedule=schedule,
                device=device,
                optimizer=optimizer,
            )

        else:

            result = train_double_head(
                model=model,
                X=X,
                y_main=y_main,
                y_aux=y_aux,
                schedule=schedule,
                aux_loss_weight=float(
                    aux_loss_weight
                ),
                device=device,
                optimizer=optimizer,
            )

        model = result.model
        optimizer = result.optimizer

    model.eval()

    return model


# ============================================================
# CONTINUOUS PSEUDO-OOD STREAM
# ============================================================


def _build_pseudo_stream(
    *,
    bundle: DataBundle,
    config: ExperimentConfig,
    years: list[int],
) -> list[
    tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
]:

    stream = []

    for year in years:

        data, split = (
            _split_for_year(
                bundle=bundle,
                config=config,
                year=year,
            )
        )

        # The split itself is stratified, but once selected the
        # samples are restored to their original relative
        # temporal order.
        indices = np.sort(
            split.test
        )

        stream.append(
            (
                int(year),
                data.X[
                    indices
                ],
                data.y_main[
                    indices
                ],
                data.y_aux[
                    indices
                ],
            )
        )

    return stream


# ============================================================
# FROZEN REFERENCES
# ============================================================


def _frozen_accuracies(
    *,
    source_model: nn.Module,
    stream: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    device: torch.device,
    family: str,
) -> dict[int, float]:

    values = {}

    for (
        year,
        X,
        y_main,
        _,
    ) in stream:

        record = (
            evaluate_frozen_year(
                model=source_model,
                X=X,
                y_main=y_main,
                year=year,
                method_name=(
                    f"{family}_frozen"
                ),
                device=device,
            )
        )

        values[
            int(year)
        ] = float(
            record.accuracy
        )

    return values


# ============================================================
# LOAD BEST BASELINE
# ============================================================


def _load_baseline_config(
    *,
    base_config: ExperimentConfig,
    baseline_best_path: Path,
) -> ExperimentConfig:

    result = load_best_result(
        baseline_best_path
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
            f"Invalid overrides in "
            f"{baseline_best_path}"
        )

    return apply_overrides(
        config=base_config,
        overrides=overrides,
    )


# ============================================================
# TTA PARAMETER INHERITANCE
# ============================================================


def _tta_parent_overrides(
    *,
    method_name: str,
    method_section: Mapping[str, Any],
    output_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:

    parent_name = method_section.get(
        "inherits_from"
    )

    if parent_name is None:
        return (
            {},
            {},
        )

    parent_name = str(
        parent_name
    )

    mapping = method_section.get(
        "inherit_parameters",
        {},
    )

    if not isinstance(
        mapping,
        Mapping,
    ):
        raise ValueError(
            f"tta.methods.{method_name}."
            "inherit_parameters must be "
            "a mapping."
        )

    parent_best_path = (
        output_root
        / "tta"
        / parent_name
        / "best.yaml"
    )

    if not parent_best_path.is_file():
        raise FileNotFoundError(
            f"TTA method {method_name} "
            f"inherits from {parent_name}, "
            "but its result does not exist: "
            f"{parent_best_path}"
        )

    inherited = (
        inherited_overrides_from_best(
            best_path=parent_best_path,
            parameter_mapping=mapping,
        )
    )

    return (
        inherited,
        {
            "inherits_from":
                parent_name,
            "parent_best_path":
                str(
                    parent_best_path
                ),
        },
    )


# ============================================================
# ONE TTA METHOD
# ============================================================


def tune_tta_method(
    *,
    method_name: str,
    method_section: Mapping[str, Any],
    source_model: nn.Module,
    source_config: ExperimentConfig,
    stream: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    frozen: Mapping[int, float],
    output_root: Path,
    device: torch.device,
) -> TTATuningResult:

    strategy = (
        build_search_strategy(
            str(
                method_section.get(
                    "search_strategy",
                    "grid",
                )
            )
        )
    )

    space = method_section.get(
        "space",
        {},
    )

    if not isinstance(
        space,
        Mapping,
    ):
        raise ValueError(
            f"{method_name}.space "
            "must be a mapping."
        )

    output_directory = (
        ensure_directory(
            output_root
            / "tta"
            / method_name
        )
    )

    (
        inherited_overrides,
        inheritance_info,
    ) = _tta_parent_overrides(
        method_name=method_name,
        method_section=method_section,
        output_root=output_root,
    )

    if inherited_overrides:

        print()
        print(
            f"{method_name} inherited "
            "TTA parameters:"
        )

        for path, value in (
            inherited_overrides.items()
        ):
            print(
                f"  {path} = {value}"
            )

    rows: list[
        dict[str, Any]
    ] = []

    best_score = float(
        "-inf"
    )

    best_parameters: dict[
        str,
        Any,
    ] = {}

    best_overrides: dict[
        str,
        Any,
    ] = {}

    method_class = (
        get_method_class(
            method_name
        )
    )

    for trial in strategy.generate(
        space
    ):

        print()
        print(
            "=" * 80
        )
        print(
            f"TTA TUNING | "
            f"{method_name} | "
            f"trial={trial.trial_id}"
        )
        print(
            trial.parameters
        )
        print(
            "=" * 80
        )

        complete_overrides = (
            merge_overrides(
                inherited_overrides,
                trial.overrides,
            )
        )

        trial_config = (
            apply_overrides(
                config=source_config,
                overrides=(
                    complete_overrides
                ),
            )
        )

        set_seed(
            trial_config.seed
        )

        method_config = (
            trial_config.section(
                method_name
            )
        )

        # IMPORTANT:
        #
        # Every TTA trial starts from exactly the SAME source
        # checkpoint.
        #
        # Temporal Gradient / Consensus inherit only TTA
        # hyperparameters from Metadata, NOT Metadata-adapted
        # weights.
        method = method_class(
            source_model=copy.deepcopy(
                source_model
            ),
            config=method_config,
            device=device,
        )

        # Continuous temporal stream.
        #
        # State is never reset between pseudo-OOD years.
        evaluation = (
            evaluate_tta_stream(
                method=method,
                years=stream,
                reset_each_year=False,
            )
        )

        tta_by_year = {
            int(record.year):
                float(
                    record.accuracy
                )
            for record
            in evaluation.records
        }

        deltas = {
            year:
                tta_by_year[year]
                - float(
                    frozen[year]
                )
            for year
            in tta_by_year
        }

        delta_values = list(
            deltas.values()
        )

        score = float(
            mean(
                delta_values
            )
        )

        std = float(
            pstdev(
                delta_values
            )
            if len(
                delta_values
            ) > 1
            else 0.0
        )

        worst = float(
            min(
                delta_values
            )
        )

        n_improved = int(
            sum(
                value > 0.0
                for value
                in delta_values
            )
        )

        row: dict[
            str,
            Any,
        ] = {
            "trial_id":
                trial.trial_id,
            "mean_delta_accuracy":
                score,
            "std_delta_accuracy":
                std,
            "worst_delta_accuracy":
                worst,
            "n_improved_years":
                n_improved,
        }

        for key, value in (
            trial.parameters.items()
        ):
            row[
                f"param_{key}"
            ] = value

        for path, value in (
            inherited_overrides.items()
        ):
            row[
                f"inherited_{path}"
            ] = value

        for year in sorted(
            tta_by_year
        ):

            row[
                f"frozen_accuracy_{year}"
            ] = frozen[
                year
            ]

            row[
                f"tta_accuracy_{year}"
            ] = tta_by_year[
                year
            ]

            row[
                f"delta_accuracy_{year}"
            ] = deltas[
                year
            ]

        rows.append(
            row
        )

        if score > best_score:

            best_score = score

            # Only parameters searched in THIS stage.
            best_parameters = dict(
                trial.parameters
            )

            # Complete config for THIS method, including the
            # inherited common values.
            best_overrides = dict(
                complete_overrides
            )

        save_trial_rows(
            output_directory
            / "trials.csv",
            rows,
        )

    save_best_result(
        output_directory
        / "best.yaml",
        kind="tta",
        name=method_name,
        score_name=(
            "mean_delta_accuracy"
        ),
        score=best_score,
        parameters=best_parameters,
        overrides=best_overrides,
        extra={
            "stream_years": [
                year
                for (
                    year,
                    _,
                    _,
                    _,
                )
                in stream
            ],
            "reset_each_year":
                False,
            **inheritance_info,
        },
    )

    return TTATuningResult(
        method_name=method_name,
        best_score=best_score,
        best_parameters=(
            best_parameters
        ),
        best_overrides=(
            best_overrides
        ),
        output_directory=(
            output_directory
        ),
    )


# ============================================================
# FULL STAGED TTA PIPELINE
# ============================================================


def tune_tta(
    *,
    base_config: ExperimentConfig,
    tuning_config: Mapping[str, Any],
    output_root: Path,
) -> dict[
    str,
    TTATuningResult,
]:

    tta_config = tuning_config.get(
        "tta",
        {},
    )

    if not isinstance(
        tta_config,
        Mapping,
    ):
        raise ValueError(
            "tta tuning section must "
            "be a mapping."
        )

    protocol = base_config.section(
        "protocol"
    )

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

    stream_start = int(
        tta_config[
            "stream_start_year"
        ]
    )

    if not (
        source_start
        < stream_start
        <= source_end
    ):
        raise ValueError(
            "tta.stream_start_year must "
            "be after source_start_year and "
            "<= source_end_year."
        )

    supervised_years = list(
        range(
            source_start,
            stream_start,
        )
    )

    stream_years = list(
        range(
            stream_start,
            source_end + 1,
        )
    )

    print()
    print(
        "=" * 80
    )
    print(
        "TTA TUNING TEMPORAL SPLIT"
    )
    print(
        f"Pseudo-source years: "
        f"{supervised_years}"
    )
    print(
        f"Pseudo-OOD stream years: "
        f"{stream_years}"
    )
    print(
        "=" * 80
    )

    bundle = build_data_bundle(
        base_config
    )

    device = _device()

    baseline_root = (
        output_root
        / "baselines"
    )

    single_best = (
        baseline_root
        / "single_head"
        / "best.yaml"
    )

    double_best = (
        baseline_root
        / "double_head"
        / "best.yaml"
    )

    if not single_best.is_file():
        raise FileNotFoundError(
            "TTA tuning requires the tuned "
            "single-head baseline first: "
            f"{single_best}"
        )

    if not double_best.is_file():
        raise FileNotFoundError(
            "TTA tuning requires the tuned "
            "double-head baseline first: "
            f"{double_best}"
        )

    single_config = (
        _load_baseline_config(
            base_config=base_config,
            baseline_best_path=(
                single_best
            ),
        )
    )

    double_config = (
        _load_baseline_config(
            base_config=base_config,
            baseline_best_path=(
                double_best
            ),
        )
    )

    # ========================================================
    # TRAIN PSEUDO-SOURCE SINGLE
    # ========================================================

    set_seed(
        base_config.seed
    )

    source_single = (
        _train_source_model(
            family="single_head",
            config=single_config,
            bundle=bundle,
            source_years=(
                supervised_years
            ),
            device=device,
        )
    )

    # ========================================================
    # TRAIN PSEUDO-SOURCE DOUBLE
    # ========================================================

    set_seed(
        base_config.seed
    )

    source_double = (
        _train_source_model(
            family="double_head",
            config=double_config,
            bundle=bundle,
            source_years=(
                supervised_years
            ),
            device=device,
        )
    )

    # ========================================================
    # SAME TEMPORAL STREAM
    # ========================================================

    single_stream = (
        _build_pseudo_stream(
            bundle=bundle,
            config=single_config,
            years=stream_years,
        )
    )

    double_stream = (
        _build_pseudo_stream(
            bundle=bundle,
            config=double_config,
            years=stream_years,
        )
    )

    # ========================================================
    # FROZEN REFERENCES
    # ========================================================

    frozen_single = (
        _frozen_accuracies(
            source_model=source_single,
            stream=single_stream,
            device=device,
            family="single_head",
        )
    )

    frozen_double = (
        _frozen_accuracies(
            source_model=source_double,
            stream=double_stream,
            device=device,
            family="double_head",
        )
    )

    methods_config = (
        tta_config.get(
            "methods",
            {},
        )
    )

    if not isinstance(
        methods_config,
        Mapping,
    ):
        raise ValueError(
            "tta.methods must be "
            "a mapping."
        )

    results: dict[
        str,
        TTATuningResult,
    ] = {}

    # --------------------------------------------------------
    # Dependency order is EXPLICIT.
    #
    # Metadata must run before Temporal Gradient / Consensus.
    # TENT is independent but uses the tuned single baseline.
    # --------------------------------------------------------

    ordered_methods = (
        "metadata",
        "temporal_gradient",
        "consensus",
        "tent",
    )

    for method_name in (
        ordered_methods
    ):

        method_section = (
            methods_config.get(
                method_name
            )
        )

        if method_section is None:
            continue

        if not isinstance(
            method_section,
            Mapping,
        ):
            raise ValueError(
                f"tta.methods.{method_name} "
                "must be a mapping."
            )

        if not bool(
            method_section.get(
                "enabled",
                True,
            )
        ):
            continue

        family = _method_family(
            method_name
        )

        if family == "single_head":

            source_model = (
                source_single
            )

            source_config = (
                single_config
            )

            stream = (
                single_stream
            )

            frozen = (
                frozen_single
            )

        else:

            source_model = (
                source_double
            )

            source_config = (
                double_config
            )

            stream = (
                double_stream
            )

            frozen = (
                frozen_double
            )

        results[
            method_name
        ] = tune_tta_method(
            method_name=(
                method_name
            ),
            method_section=(
                method_section
            ),
            source_model=(
                source_model
            ),
            source_config=(
                source_config
            ),
            stream=stream,
            frozen=frozen,
            output_root=(
                output_root
            ),
            device=device,
        )

    return results