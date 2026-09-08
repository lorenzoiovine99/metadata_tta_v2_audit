from __future__ import annotations

import argparse

from metadata_tta.config import (
    load_config,
)
from metadata_tta.experiment import (
    run_experiment,
)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    result = run_experiment(
        config
    )

    print(
        "\nRun completed."
    )

    print(
        result.run_directory
    )


if __name__ == "__main__":
    main()