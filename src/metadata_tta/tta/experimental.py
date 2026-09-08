from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .base import (
    AdaptationResult,
    TTAMethod,
)


class ExperimentalTTA(
    TTAMethod
):
    """
    Temporary research slot for new TTA ideas.

    This class intentionally cannot be instantiated while
    empty.

    When testing a new method, replace this implementation
    temporarily.

    If the method becomes stable, move it to its own module
    and register it with a permanent method name.
    """

    method_name = "experimental"

    requires_aux_labels = False

    def __init__(
        self,
        source_model: nn.Module,
        config: Mapping[str, Any],
        device: torch.device,
    ) -> None:

        raise NotImplementedError(
            "Experimental TTA is enabled but no "
            "experimental method is implemented."
        )

    def predict_logits(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:

        raise NotImplementedError

    def observe(
        self,
        x: np.ndarray | torch.Tensor,
        y_aux: (
            np.ndarray
            | torch.Tensor
            | int
            | None
        ) = None,
    ) -> AdaptationResult:

        raise NotImplementedError