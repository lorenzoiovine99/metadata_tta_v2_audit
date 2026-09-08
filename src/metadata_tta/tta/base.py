from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


# ============================================================
# ADAPTATION RESULT
# ============================================================

@dataclass(frozen=True)
class AdaptationResult:
    """
    Result of one test-time observation/update step.

    Parameters
    ----------
    applied:
        True when the method actually applied an adaptation
        update to its model.

    diagnostics:
        Optional scalar diagnostics associated with the step.

        Examples:
            auxiliary loss
            entropy
            gradient norm
            consensus cosine
            update probability
    """

    applied: bool

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# BASE TTA METHOD
# ============================================================

class TTAMethod(ABC):
    """
    Common interface for every deployable test-time
    adaptation method.

    Scientific contract
    -------------------
    The evaluator is responsible for enforcing:

        PREDICT
        RECORD PREDICTION
        OBSERVE
        UPDATE

    A TTA method therefore exposes prediction and observation
    as two separate operations.

    Subclasses must never require OOD main-task labels.
    """

    method_name: str = "base"

    requires_aux_labels: bool = False

    def __init__(
        self,
        source_model: nn.Module,
        config: Mapping[str, Any],
        device: torch.device,
    ) -> None:

        if not isinstance(
            source_model,
            nn.Module,
        ):
            raise TypeError(
                "source_model must be a torch.nn.Module."
            )

        if not isinstance(
            config,
            Mapping,
        ):
            raise TypeError(
                "config must be a mapping."
            )

        self.device = device

        self.config: dict[str, Any] = (
            copy.deepcopy(
                dict(config)
            )
        )

        # Important:
        #
        # The source model supplied by the training pipeline
        # must never be modified by a TTA method.
        #
        # Every TTA method therefore receives its own model
        # copy.
        self.model = copy.deepcopy(
            source_model
        ).to(
            self.device
        )

        # Store an immutable numerical reference to the source
        # parameters/buffers.
        #
        # This is useful both for:
        #
        # - episodic reset;
        # - source-anchor regularization.
        #
        # Keeping tensors on CPU avoids holding another complete
        # model on GPU/MPS memory.
        self._source_state = {
            name: tensor.detach()
            .cpu()
            .clone()
            for name, tensor
            in self.model.state_dict().items()
        }

        self._number_of_observations = 0
        self._number_of_updates = 0

    # ========================================================
    # REQUIRED INTERFACE
    # ========================================================

    @abstractmethod
    def predict_logits(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict main-task logits using the current adapted state.

        This method MUST NOT perform an adaptation update.
        """

        raise NotImplementedError

    @abstractmethod
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
        """
        Observe test-time information and optionally update.

        Main-task OOD labels are deliberately absent from this
        interface.

        Methods using metadata receive y_aux.

        Methods such as TENT may ignore y_aux.
        """

        raise NotImplementedError

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ) -> None:
        """
        Restore the method to the frozen source state.

        Subclasses may override _reset_adaptation_state() to
        rebuild optimizers, gradient memories, EMA state, etc.
        """

        self._restore_source_model()

        self._number_of_observations = 0
        self._number_of_updates = 0

        self._reset_adaptation_state()

    def _restore_source_model(
        self,
    ) -> None:
        """
        Restore parameters and buffers from the original
        source checkpoint.
        """

        state = {
            name: tensor.clone()
            for name, tensor
            in self._source_state.items()
        }

        self.model.load_state_dict(
            state,
            strict=True,
        )

        self.model.to(
            self.device
        )

    def _reset_adaptation_state(
        self,
    ) -> None:
        """
        Optional subclass hook executed after restoring the
        source model.

        Examples
        --------
        Metadata TTA:
            rebuild optimizer

        Temporal Gradient:
            clear gradient EMA

        Consensus:
            clear per-class memories

        TENT:
            rebuild optimizer / adaptation statistics
        """

        return None

    # ========================================================
    # INPUT CONVERSION
    # ========================================================

    def _as_feature_tensor(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert input features to float32 on the method device.

        A one-dimensional feature vector becomes a batch
        containing one sample.
        """

        if isinstance(
            x,
            torch.Tensor,
        ):
            tensor = x.to(
                device=self.device,
                dtype=torch.float32,
            )

        else:
            tensor = torch.as_tensor(
                x,
                dtype=torch.float32,
                device=self.device,
            )

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(
                0
            )

        if tensor.ndim != 2:
            raise ValueError(
                "TTA feature input must be 1-D or 2-D. "
                f"Received shape {tuple(tensor.shape)}."
            )

        return tensor

    def _as_label_tensor(
        self,
        y: (
            np.ndarray
            | torch.Tensor
            | int
        ),
    ) -> torch.Tensor:
        """
        Convert labels to a one-dimensional LongTensor.
        """

        if isinstance(
            y,
            torch.Tensor,
        ):
            tensor = y.to(
                device=self.device,
                dtype=torch.long,
            )

        else:
            tensor = torch.as_tensor(
                y,
                dtype=torch.long,
                device=self.device,
            )

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(
                0
            )

        tensor = tensor.reshape(
            -1
        )

        return tensor

    # ========================================================
    # SOURCE REFERENCE
    # ========================================================

    def source_tensor(
        self,
        name: str,
    ) -> torch.Tensor:
        """
        Retrieve one source-state tensor on the active device.

        Useful for source-anchor regularization.
        """

        if name not in self._source_state:
            raise KeyError(
                f"Unknown source-state tensor: {name}"
            )

        return self._source_state[
            name
        ].to(
            self.device
        )

    # ========================================================
    # COUNTERS
    # ========================================================

    def _record_observation(
        self,
    ) -> None:

        self._number_of_observations += 1

    def _record_update(
        self,
    ) -> None:

        self._number_of_updates += 1

    @property
    def number_of_observations(
        self,
    ) -> int:

        return int(
            self._number_of_observations
        )

    @property
    def number_of_updates(
        self,
    ) -> int:

        return int(
            self._number_of_updates
        )

    @property
    def update_rate(
        self,
    ) -> float:

        if self._number_of_observations == 0:
            return 0.0

        return float(
            self._number_of_updates
            / self._number_of_observations
        )

    # ========================================================
    # CONVENIENCE PREDICTION
    # ========================================================

    def predict_classes(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Return main-task predicted class indices.

        No adaptation is performed here.
        """

        logits = self.predict_logits(
            x
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        return (
            predictions.detach()
            .cpu()
            .numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )

    # ========================================================
    # GLOBAL DIAGNOSTICS
    # ========================================================

    def diagnostics(
        self,
    ) -> dict[str, Any]:
        """
        Return method-level cumulative diagnostics.

        Subclasses may extend this method but should preserve
        these common fields.
        """

        return {
            "method":
                self.method_name,

            "n_observations":
                self.number_of_observations,

            "n_updates":
                self.number_of_updates,

            "update_rate":
                self.update_rate,
        }