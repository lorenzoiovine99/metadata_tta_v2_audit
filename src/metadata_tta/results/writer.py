from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from metadata_tta.evaluation import (
    EvaluationRecord,
)


class ResultsWriter:

    def __init__(
        self,
        run_directory: str | Path,
    ) -> None:

        self.run_directory = Path(
            run_directory
        )

        self.metrics_directory = (
            self.run_directory
            / "metrics"
        )

        self.diagnostics_directory = (
            self.run_directory
            / "diagnostics"
        )

        self.plots_directory = (
            self.run_directory
            / "plots"
        )

        for directory in (
            self.run_directory,
            self.metrics_directory,
            self.diagnostics_directory,
            self.plots_directory,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def write_config(
        self,
        config: dict[str, Any],
    ) -> None:

        path = (
            self.run_directory
            / "config.yaml"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(
                config,
                handle,
                sort_keys=False,
            )

    def write_manifest(
        self,
        manifest: dict[str, Any],
    ) -> None:

        path = (
            self.run_directory
            / "manifest.json"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
                default=str,
            )

    def write_records(
        self,
        filename: str,
        records: list[EvaluationRecord]
        | tuple[EvaluationRecord, ...],
    ) -> None:

        path = (
            self.metrics_directory
            / filename
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "year",
                    "method",
                    "accuracy",
                    "n_samples",
                ],
            )

            writer.writeheader()

            for record in records:
                writer.writerow(
                    {
                        "year":
                            record.year,
                        "method":
                            record.method,
                        "accuracy":
                            record.accuracy,
                        "n_samples":
                            record.n_samples,
                    }
                )

    def write_summary(
        self,
        rows: list[
            dict[str, object]
        ],
    ) -> None:

        path = (
            self.metrics_directory
            / "summary.csv"
        )

        fieldnames = [
            "method",
            "mean_ood",
            "std_ood",
            "worst_ood",
            "delta_vs_reference",
        ]

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    def write_diagnostics(
        self,
        method_name: str,
        diagnostics: dict[str, Any],
    ) -> None:

        path = (
            self.diagnostics_directory
            / f"{method_name}.json"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                diagnostics,
                handle,
                indent=2,
                sort_keys=True,
                default=str,
            )