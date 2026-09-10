from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

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