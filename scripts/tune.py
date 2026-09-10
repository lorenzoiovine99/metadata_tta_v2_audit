from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Make src/ importable when running:
#
#     python scripts/tune.py ...
#
# This keeps the script usable even if the project has not been
# installed as a Python package.
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )


from metadata_tta.tuning.baseline_tuner import (
    tune_baselines,
)
from metadata_tta.tuning.config import (
    load_tuning_yaml,
    resolve_base_config,
)
from metadata_tta.tuning.materialize import (
    materialize_tuned_config,
)
from metadata_tta.tuning.tta_tuner import (
    tune_tta,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter tuning for "
            "metadata_tta."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to tuning YAML.",
    )

    parser.add_argument(
        "--stage",
        choices=[
            "baseline",
            "tta",
            "all",
        ],
        default="all",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    tuning_path = Path(
        args.config
    ).expanduser().resolve()

    tuning_config = (
        load_tuning_yaml(
            tuning_path
        )
    )

    (
        base_config_path,
        base_config,
    ) = resolve_base_config(
        tuning_yaml_path=(
            tuning_path
        ),
        tuning_config=(
            tuning_config
        ),
    )

    print(
        f"Base experiment config: "
        f"{base_config_path}"
    )

    output_root = Path(
        tuning_config.get(
            "output_root",
            (
                base_config.results_root
                / "tuning"
                / base_config.dataset_name
            ),
        )
    ).expanduser()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.stage in {
        "baseline",
        "all",
    }:
        tune_baselines(
            base_config=base_config,
            tuning_config=(
                tuning_config
            ),
            output_root=output_root,
        )

    if args.stage in {
        "tta",
        "all",
    }:
        tune_tta(
            base_config=base_config,
            tuning_config=(
                tuning_config
            ),
            output_root=output_root,
        )

    if args.stage == "all":

        tta_section = (
            tuning_config.get(
                "tta",
                {},
            )
        )

        method_sections = (
            tta_section.get(
                "methods",
                {},
            )
        )

        enabled_methods = [
            str(name)
            for name, section
            in method_sections.items()
            if isinstance(
                section,
                dict,
            )
            and bool(
                section.get(
                    "enabled",
                    True,
                )
            )
        ]

        destination = (
            output_root
            / "tuned_experiment.yaml"
        )

        materialize_tuned_config(
            base_config=base_config,
            output_root=output_root,
            enabled_methods=(
                enabled_methods
            ),
            destination=destination,
        )

        print()
        print(
            "=" * 80
        )
        print(
            "TUNING COMPLETE"
        )
        print(
            f"Tuned experiment config: "
            f"{destination}"
        )
        print(
            "=" * 80
        )


if __name__ == "__main__":
    main()