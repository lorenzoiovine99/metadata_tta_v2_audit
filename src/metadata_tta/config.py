from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


# ============================================================
# CONSTANTS
# ============================================================

VALID_PROTOCOLS = {
    "eval_fix",
    "eval_stream",
    "eval_stream_tas",
}


REQUIRED_TOP_LEVEL_SECTIONS = {
    "experiment",
    "paths",
    "dataset",
    "protocol",
    "pca",
    "model",
    "training",
    "baselines",
    "methods",
    "cache",
    "output",
}


# ============================================================
# EXCEPTIONS
# ============================================================

class ConfigError(ValueError):
    """
    Raised when an experiment configuration is invalid.
    """


# ============================================================
# CONFIGURATION
# ============================================================

class ExperimentConfig:
    """
    Immutable-style wrapper around an experiment configuration.

    The internal representation remains a standard dictionary,
    but all public accessors return deep copies.

    This makes configurations safe to reuse across experiments
    and easy to modify programmatically for future
    hyperparameter tuning.

    Example
    -------
    candidate_config = config.with_overrides(
        {
            "metadata.learning_rate": 1e-5,
            "metadata.regularization": 1e-3,
        }
    )
    """

    def __init__(
        self,
        data: Mapping[str, Any],
    ) -> None:

        if not isinstance(data, Mapping):
            raise ConfigError(
                "Configuration must be a mapping."
            )

        self._data: dict[str, Any] = copy.deepcopy(
            dict(data)
        )

        self._validate()

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate(
        self,
    ) -> None:

        self._validate_required_sections()
        self._validate_experiment()
        self._validate_paths()
        self._validate_dataset()
        self._validate_protocol()
        self._validate_pca()
        self._validate_model()
        self._validate_training()
        self._validate_baselines()
        self._validate_methods()
        self._validate_cache()
        self._validate_output()

    # --------------------------------------------------------
    # Required sections
    # --------------------------------------------------------

    def _validate_required_sections(
        self,
    ) -> None:

        missing = (
            REQUIRED_TOP_LEVEL_SECTIONS
            - set(self._data.keys())
        )

        if missing:
            raise ConfigError(
                "Missing required configuration sections: "
                + ", ".join(sorted(missing))
            )

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    def _validate_experiment(
        self,
    ) -> None:

        section = self._require_mapping(
            "experiment"
        )

        self._require_non_empty_string(
            section,
            "name",
            "experiment",
        )

        self._require_integer(
            section,
            "seed",
            "experiment",
            minimum=0,
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    def _validate_paths(
        self,
    ) -> None:

        section = self._require_mapping(
            "paths"
        )

        required = (
            "data_root",
            "results_root",
            "artifacts_root",
            "cache_root",
        )

        for key in required:
            self._require_non_empty_string(
                section,
                key,
                "paths",
            )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    def _validate_dataset(
        self,
    ) -> None:

        section = self._require_mapping(
            "dataset"
        )

        self._require_non_empty_string(
            section,
            "name",
            "dataset",
        )

        self._require_integer(
            section,
            "n_classes_main",
            "dataset",
            minimum=1,
        )

        self._require_integer(
            section,
            "n_classes_aux",
            "dataset",
            minimum=1,
        )

        split = section.get(
            "split"
        )

        if not isinstance(
            split,
            Mapping,
        ):
            raise ConfigError(
                "dataset.split must be a mapping."
            )

        test_size = self._require_number(
            split,
            "test_size",
            "dataset.split",
        )

        validation_size = self._require_number(
            split,
            "validation_size_within_train",
            "dataset.split",
        )

        if not 0.0 < test_size < 1.0:
            raise ConfigError(
                "dataset.split.test_size must be "
                "strictly between 0 and 1."
            )

        if not 0.0 <= validation_size < 1.0:
            raise ConfigError(
                "dataset.split.validation_size_within_train "
                "must satisfy 0 <= value < 1."
            )

        self._require_integer(
            split,
            "seed",
            "dataset.split",
            minimum=0,
        )

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    def _validate_protocol(
        self,
    ) -> None:

        section = self._require_mapping(
            "protocol"
        )

        name = self._require_non_empty_string(
            section,
            "name",
            "protocol",
        ).lower()

        if name not in VALID_PROTOCOLS:
            raise ConfigError(
                f"Invalid protocol.name: {name!r}. "
                "Expected one of: "
                f"{sorted(VALID_PROTOCOLS)}"
            )

        if name == "eval_fix":
            self._validate_eval_fix_protocol(
                section
            )

        elif name == "eval_stream":
            self._validate_eval_stream_protocol(
                section
            )

        elif name == "eval_stream_tas":
            self._validate_eval_stream_tas_protocol(
                section
            )

    def _validate_eval_fix_protocol(
        self,
        section: Mapping[str, Any],
    ) -> None:

        source_start = self._require_integer(
            section,
            "source_start_year",
            "protocol",
        )

        source_end = self._require_integer(
            section,
            "source_end_year",
            "protocol",
        )

        ood_start = self._require_integer(
            section,
            "ood_start_year",
            "protocol",
        )

        ood_end = self._require_integer(
            section,
            "ood_end_year",
            "protocol",
        )

        if source_start > source_end:
            raise ConfigError(
                "protocol.source_start_year must be "
                "<= protocol.source_end_year."
            )

        if ood_start > ood_end:
            raise ConfigError(
                "protocol.ood_start_year must be "
                "<= protocol.ood_end_year."
            )

        if source_end >= ood_start:
            raise ConfigError(
                "Eval-Fix requires source_end_year "
                "< ood_start_year."
            )

    def _validate_eval_stream_tas_protocol(
        self,
        section: Mapping[str, Any],
    ) -> None:

        source_start = self._require_integer(
            section,
            "source_start_year",
            "protocol",
        )

        source_end = self._require_integer(
            section,
            "source_end_year",
            "protocol",
        )

        ood_start = self._require_integer(
            section,
            "ood_start_year",
            "protocol",
        )

        ood_end = self._require_integer(
            section,
            "ood_end_year",
            "protocol",
        )

        if source_start > source_end:
            raise ConfigError(
                "protocol.source_start_year must be "
                "<= protocol.source_end_year."
            )

        if ood_start > ood_end:
            raise ConfigError(
                "protocol.ood_start_year must be "
                "<= protocol.ood_end_year."
            )

        if source_end >= ood_start:
            raise ConfigError(
                "Eval-Stream-TAS requires source_end_year "
                "< ood_start_year."
            )

    def _validate_eval_stream_protocol(
        self,
        section: Mapping[str, Any],
    ) -> None:

        evaluation_start = self._require_integer(
            section,
            "evaluation_start_year",
            "protocol",
        )

        evaluation_end = self._require_integer(
            section,
            "evaluation_end_year",
            "protocol",
        )

        horizon = self._require_integer(
            section,
            "ood_horizon",
            "protocol",
            minimum=1,
        )

        if evaluation_start > evaluation_end:
            raise ConfigError(
                "protocol.evaluation_start_year must be "
                "<= protocol.evaluation_end_year."
            )

        if horizon < 1:
            raise ConfigError(
                "protocol.ood_horizon must be positive."
            )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    def _validate_pca(
        self,
    ) -> None:

        section = self._require_mapping(
            "pca"
        )

        enabled = section.get(
            "enabled"
        )

        if not isinstance(
            enabled,
            bool,
        ):
            raise ConfigError(
                "pca.enabled must be boolean."
            )

        if not enabled:
            return

        self._require_integer(
            section,
            "n_components",
            "pca",
            minimum=1,
        )

        warmup_start = self._require_integer(
            section,
            "warmup_start_year",
            "pca",
        )

        warmup_end = self._require_integer(
            section,
            "warmup_end_year",
            "pca",
        )

        if warmup_start > warmup_end:
            raise ConfigError(
                "pca.warmup_start_year must be "
                "<= pca.warmup_end_year."
            )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    def _validate_model(
        self,
    ) -> None:

        section = self._require_mapping(
            "model"
        )

        self._require_integer(
            section,
            "shared_hidden_dim",
            "model",
            minimum=1,
        )

        self._require_integer(
            section,
            "adapter_bottleneck_dim",
            "model",
            minimum=1,
        )

        dropout = self._require_number(
            section,
            "dropout",
            "model",
        )

        if not 0.0 <= dropout < 1.0:
            raise ConfigError(
                "model.dropout must satisfy "
                "0 <= dropout < 1."
            )

        use_shared_trunk = section.get(
            "use_shared_trunk"
        )

        if not isinstance(
            use_shared_trunk,
            bool,
        ):
            raise ConfigError(
                "model.use_shared_trunk must be boolean."
            )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def _validate_training(
        self,
    ) -> None:

        section = self._require_mapping(
            "training"
        )

        base = section.get(
            "base"
        )

        if not isinstance(
            base,
            Mapping,
        ):
            raise ConfigError(
                "training.base must be a mapping."
            )

        self._require_positive_number(
            base,
            "learning_rate",
            "training.base",
        )

        self._require_non_negative_number(
            base,
            "weight_decay",
            "training.base",
        )

        self._require_integer(
            base,
            "epochs",
            "training.base",
            minimum=1,
        )

        self._require_integer(
            base,
            "batch_size",
            "training.base",
            minimum=1,
        )

        double_head = section.get(
            "double_head"
        )

        if not isinstance(
            double_head,
            Mapping,
        ):
            raise ConfigError(
                "training.double_head must be a mapping."
            )

        self._require_non_negative_number(
            double_head,
            "aux_loss_weight",
            "training.double_head",
        )

        temporal = section.get(
            "temporal"
        )

        if not isinstance(
            temporal,
            Mapping,
        ):
            raise ConfigError(
                "training.temporal must be a mapping."
            )

        enabled = temporal.get(
            "enabled"
        )

        if not isinstance(
            enabled,
            bool,
        ):
            raise ConfigError(
                "training.temporal.enabled must be boolean."
            )

        if enabled:
            self._require_positive_number(
                temporal,
                "learning_rate",
                "training.temporal",
            )

            self._require_integer(
                temporal,
                "epochs",
                "training.temporal",
                minimum=1,
            )

            reset_optimizer = temporal.get(
                "reset_optimizer_each_year"
            )

            if not isinstance(
                reset_optimizer,
                bool,
            ):
                raise ConfigError(
                    "training.temporal."
                    "reset_optimizer_each_year "
                    "must be boolean."
                )

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    def _validate_baselines(
        self,
    ) -> None:

        section = self._require_mapping(
            "baselines"
        )

        required = {
            "single_head",
            "double_head",
        }

        missing = required - set(
            section.keys()
        )

        if missing:
            raise ConfigError(
                "Missing baseline flags: "
                + ", ".join(sorted(missing))
            )

        for name, enabled in section.items():

            if not isinstance(
                enabled,
                bool,
            ):
                raise ConfigError(
                    f"baselines.{name} must be boolean."
                )

        if not section[
            "single_head"
        ] and not section[
            "double_head"
        ]:
            raise ConfigError(
                "At least one frozen baseline "
                "must be enabled."
            )

    # --------------------------------------------------------
    # Methods
    # --------------------------------------------------------

    def _validate_methods(
        self,
    ) -> None:

        section = self._require_mapping(
            "methods"
        )

        if not section:
            raise ConfigError(
                "methods cannot be empty."
            )

        for name, enabled in section.items():

            if not isinstance(
                name,
                str,
            ) or not name.strip():
                raise ConfigError(
                    "Every methods key must be "
                    "a non-empty string."
                )

            if not isinstance(
                enabled,
                bool,
            ):
                raise ConfigError(
                    f"methods.{name} must be boolean."
                )

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    def _validate_cache(
        self,
    ) -> None:

        section = self._require_mapping(
            "cache"
        )

        enabled = section.get(
            "enabled"
        )

        if not isinstance(
            enabled,
            bool,
        ):
            raise ConfigError(
                "cache.enabled must be boolean."
            )

        version = section.get(
            "version"
        )

        if (
            not isinstance(version, str)
            or not version.strip()
        ):
            raise ConfigError(
                "cache.version must be a "
                "non-empty string."
            )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    def _validate_output(
        self,
    ) -> None:

        section = self._require_mapping(
            "output"
        )

        for key in (
            "save_diagnostics",
            "save_plots",
        ):

            value = section.get(
                key
            )

            if not isinstance(
                value,
                bool,
            ):
                raise ConfigError(
                    f"output.{key} must be boolean."
                )

    # ========================================================
    # VALIDATION HELPERS
    # ========================================================

    def _require_mapping(
        self,
        section: str,
    ) -> Mapping[str, Any]:

        value = self._data.get(
            section
        )

        if not isinstance(
            value,
            Mapping,
        ):
            raise ConfigError(
                f"{section} must be a mapping."
            )

        return value

    @staticmethod
    def _require_non_empty_string(
        mapping: Mapping[str, Any],
        key: str,
        section: str,
    ) -> str:

        value = mapping.get(
            key
        )

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ConfigError(
                f"{section}.{key} must be a "
                "non-empty string."
            )

        return value.strip()

    @staticmethod
    def _require_integer(
        mapping: Mapping[str, Any],
        key: str,
        section: str,
        minimum: int | None = None,
    ) -> int:

        value = mapping.get(
            key
        )

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise ConfigError(
                f"{section}.{key} must be an integer."
            )

        if (
            minimum is not None
            and value < minimum
        ):
            raise ConfigError(
                f"{section}.{key} must be >= {minimum}."
            )

        return value

    @staticmethod
    def _require_number(
        mapping: Mapping[str, Any],
        key: str,
        section: str,
    ) -> float:

        value = mapping.get(
            key
        )

        if (
            not isinstance(
                value,
                (int, float),
            )
            or isinstance(value, bool)
        ):
            raise ConfigError(
                f"{section}.{key} must be numeric."
            )

        return float(value)

    @classmethod
    def _require_positive_number(
        cls,
        mapping: Mapping[str, Any],
        key: str,
        section: str,
    ) -> float:

        value = cls._require_number(
            mapping,
            key,
            section,
        )

        if value <= 0:
            raise ConfigError(
                f"{section}.{key} must be > 0."
            )

        return value

    @classmethod
    def _require_non_negative_number(
        cls,
        mapping: Mapping[str, Any],
        key: str,
        section: str,
    ) -> float:

        value = cls._require_number(
            mapping,
            key,
            section,
        )

        if value < 0:
            raise ConfigError(
                f"{section}.{key} must be >= 0."
            )

        return value

    # ========================================================
    # GENERAL ACCESS
    # ========================================================

    def as_dict(
        self,
    ) -> dict[str, Any]:
        """
        Return an independent deep copy of the configuration.
        """

        return copy.deepcopy(
            self._data
        )

    def clone(
        self,
    ) -> "ExperimentConfig":
        """
        Return an independent configuration clone.
        """

        return ExperimentConfig(
            self.as_dict()
        )

    def section(
        self,
        name: str,
    ) -> dict[str, Any]:
        """
        Return a deep copy of one top-level section.
        """

        if name not in self._data:
            raise KeyError(
                f"Unknown configuration section: {name}"
            )

        value = self._data[
            name
        ]

        if not isinstance(
            value,
            Mapping,
        ):
            raise ConfigError(
                f"{name} is not a mapping."
            )

        return copy.deepcopy(
            dict(value)
        )

    def get(
        self,
        *keys: str,
    ) -> Any:
        """
        Retrieve a nested configuration value.

        Example
        -------
        config.get(
            "training",
            "base",
            "learning_rate",
        )
        """

        if not keys:
            raise ValueError(
                "At least one configuration key "
                "is required."
            )

        value: Any = self._data

        for key in keys:

            if not isinstance(
                value,
                Mapping,
            ):
                raise KeyError(
                    ".".join(keys)
                )

            if key not in value:
                raise KeyError(
                    ".".join(keys)
                )

            value = value[
                key
            ]

        return copy.deepcopy(
            value
        )

    def get_optional(
        self,
        *keys: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a nested value without raising when absent.
        """

        if not keys:
            raise ValueError(
                "At least one configuration key "
                "is required."
            )

        value: Any = self._data

        for key in keys:

            if (
                not isinstance(value, Mapping)
                or key not in value
            ):
                return copy.deepcopy(
                    default
                )

            value = value[
                key
            ]

        return copy.deepcopy(
            value
        )

    # ========================================================
    # FUTURE TUNING SUPPORT
    # ========================================================

    def with_overrides(
        self,
        overrides: Mapping[str, Any],
    ) -> "ExperimentConfig":
        """
        Return an independent configuration with dot-separated
        overrides applied.

        This is intentionally generic so that future tuning
        infrastructure does not need method-specific config
        mutation functions.

        Example
        -------
        candidate = config.with_overrides(
            {
                "metadata.learning_rate": 1e-5,
                "metadata.regularization": 1e-3,
            }
        )
        """

        updated = self.as_dict()

        for path, new_value in overrides.items():

            if (
                not isinstance(path, str)
                or not path.strip()
            ):
                raise ConfigError(
                    "Override keys must be "
                    "non-empty strings."
                )

            keys = path.split(".")

            current: dict[str, Any] = updated

            for key in keys[:-1]:

                if key not in current:
                    raise ConfigError(
                        "Cannot override unknown "
                        f"configuration path: {path}"
                    )

                next_value = current[
                    key
                ]

                if not isinstance(
                    next_value,
                    dict,
                ):
                    raise ConfigError(
                        "Cannot override nested path "
                        f"{path}: {key} is not "
                        "a mapping."
                    )

                current = next_value

            final_key = keys[-1]

            if final_key not in current:
                raise ConfigError(
                    "Cannot override unknown "
                    f"configuration path: {path}"
                )

            current[
                final_key
            ] = copy.deepcopy(
                new_value
            )

        return ExperimentConfig(
            updated
        )

    # ========================================================
    # FREQUENTLY USED PROPERTIES
    # ========================================================

    @property
    def experiment_name(
        self,
    ) -> str:

        return str(
            self._data[
                "experiment"
            ]["name"]
        )

    @property
    def seed(
        self,
    ) -> int:

        return int(
            self._data[
                "experiment"
            ]["seed"]
        )

    @property
    def dataset_name(
        self,
    ) -> str:

        return str(
            self._data[
                "dataset"
            ]["name"]
        )

    @property
    def protocol_name(
        self,
    ) -> str:

        return str(
            self._data[
                "protocol"
            ]["name"]
        ).lower()

    @property
    def data_root(
        self,
    ) -> Path:

        return Path(
            self._data[
                "paths"
            ]["data_root"]
        ).expanduser()

    @property
    def results_root(
        self,
    ) -> Path:

        return Path(
            self._data[
                "paths"
            ]["results_root"]
        ).expanduser()

    @property
    def artifacts_root(
        self,
    ) -> Path:

        return Path(
            self._data[
                "paths"
            ]["artifacts_root"]
        ).expanduser()

    @property
    def cache_root(
        self,
    ) -> Path:

        return Path(
            self._data[
                "paths"
            ]["cache_root"]
        ).expanduser()

    @property
    def enabled_methods(
        self,
    ) -> list[str]:
        """
        Return TTA methods enabled in the configuration.
        """

        methods = self._data[
            "methods"
        ]

        return [
            str(name)
            for name, enabled
            in methods.items()
            if enabled
        ]

    def method_enabled(
        self,
        name: str,
    ) -> bool:
        """
        Return whether a TTA method is enabled.
        """

        methods = self._data[
            "methods"
        ]

        return bool(
            methods.get(
                name,
                False,
            )
        )

    def baseline_enabled(
        self,
        name: str,
    ) -> bool:
        """
        Return whether a frozen baseline is enabled.
        """

        baselines = self._data[
            "baselines"
        ]

        return bool(
            baselines.get(
                name,
                False,
            )
        )

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ) -> str:

        return (
            "ExperimentConfig("
            f"dataset={self.dataset_name!r}, "
            f"protocol={self.protocol_name!r}, "
            f"seed={self.seed}, "
            f"enabled_methods={self.enabled_methods!r}"
            ")"
        )


# ============================================================
# YAML LOADER
# ============================================================

def load_config(
    path: str | Path,
) -> ExperimentConfig:
    """
    Load and validate an experiment YAML file.
    """

    config_path = Path(
        path
    ).expanduser()

    if not config_path.is_file():
        raise FileNotFoundError(
            "Configuration file not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        raw_config = yaml.safe_load(
            file
        )

    if raw_config is None:
        raise ConfigError(
            "Configuration file is empty: "
            f"{config_path}"
        )

    if not isinstance(
        raw_config,
        dict,
    ):
        raise ConfigError(
            "The root of the configuration "
            "file must be a mapping."
        )

    return ExperimentConfig(
        raw_config
    )