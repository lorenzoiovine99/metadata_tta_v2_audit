from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import (
    AdaptationResult,
)

from .metadata import (
    MetadataTTA,
)


class ConsensusTTA(
    MetadataTTA
):
    """
    Metadata Gradient Consensus TTA-B.

    For each auxiliary metadata class m, maintain a separate
    exponential moving average of metadata gradients:

        memory_m <- beta * memory_m
                    + (1 - beta) * g_t

    where g_t is the raw auxiliary metadata gradient.

    Warmup
    ------
    During the first class_warmup_updates accepted gradients
    for metadata class m, apply the raw metadata gradient.

    Consensus
    ---------
    After warmup, compare the current raw gradient against the
    previous EMA memory for the same metadata class:

        cosine(g_t, memory_m)

    If the cosine is below min_consensus_cosine:

        do not update model parameters.

    The raw gradient still updates the class-conditioned
    memory so the memory can continue tracking distribution
    drift.

    If consensus passes:

        g_mix =
            current_gradient_weight * g_t
            +
            memory_gradient_weight * memory_m

    If preserve_gradient_norm is enabled, g_mix is rescaled
    to match the norm of g_t.

    Finally, the source-anchor gradient is added and the
    online adapter is updated.

    Only online_adapter is trainable.

    No OOD main-task labels are used.
    """

    method_name = "consensus"

    requires_aux_labels = True

    def __init__(
        self,
        source_model: nn.Module,
        config: Mapping[str, Any],
        device: torch.device,
    ) -> None:

        # MetadataTTA supplies:
        #
        # - independent source-model copy
        # - adapter-only trainability
        # - source adapter reference
        # - Adam optimizer
        # - auxiliary-loss gate
        # - common metadata diagnostics
        super().__init__(
            source_model=source_model,
            config=config,
            device=device,
        )

        # ====================================================
        # CONSENSUS CONFIG
        # ====================================================

        self.consensus_beta = float(
            self.config[
                "consensus_beta"
            ]
        )

        self.class_warmup_updates = int(
            self.config[
                "class_warmup_updates"
            ]
        )

        self.current_gradient_weight = float(
            self.config[
                "current_gradient_weight"
            ]
        )

        self.memory_gradient_weight = float(
            self.config[
                "memory_gradient_weight"
            ]
        )

        self.preserve_gradient_norm = bool(
            self.config.get(
                "preserve_gradient_norm",
                True,
            )
        )

        self.min_consensus_cosine = float(
            self.config.get(
                "min_consensus_cosine",
                0.0,
            )
        )

        self.epsilon = float(
            self.config.get(
                "epsilon",
                1.0e-12,
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        if not (
            0.0
            <= self.consensus_beta
            < 1.0
        ):
            raise ValueError(
                "consensus.consensus_beta must satisfy "
                "0 <= consensus_beta < 1."
            )

        if self.class_warmup_updates < 1:
            raise ValueError(
                "consensus.class_warmup_updates "
                "must be >= 1."
            )

        if self.current_gradient_weight < 0.0:
            raise ValueError(
                "consensus.current_gradient_weight "
                "must be >= 0."
            )

        if self.memory_gradient_weight < 0.0:
            raise ValueError(
                "consensus.memory_gradient_weight "
                "must be >= 0."
            )

        if (
            self.current_gradient_weight
            + self.memory_gradient_weight
            <= 0.0
        ):
            raise ValueError(
                "At least one consensus gradient weight "
                "must be positive."
            )

        if not (
            -1.0
            <= self.min_consensus_cosine
            <= 1.0
        ):
            raise ValueError(
                "consensus.min_consensus_cosine must "
                "be between -1 and 1."
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "consensus.epsilon must be > 0."
            )

        # The stable V1 Consensus implementation performs
        # exactly one gradient/update attempt per observation.
        #
        # Its selected configuration has steps = 1.
        #
        # Refuse other values rather than silently introducing
        # a new algorithmic interpretation during the V2 port.
        if self.steps != 1:
            raise ValueError(
                "ConsensusTTA currently requires steps == 1 "
                "to preserve the stable V1 algorithm."
            )

        # Consensus is defined sample-by-sample.
        if self.batch_size != 1:
            raise ValueError(
                "ConsensusTTA requires batch_size == 1."
            )

        # ====================================================
        # CLASS-CONDITIONED MEMORY
        # ====================================================

        self._reset_consensus_state()

    # ========================================================
    # RESET
    # ========================================================

    def _reset_adaptation_state(
        self,
    ) -> None:
        """
        Restore optimizer, common metadata stats and all
        class-conditioned consensus memories.
        """

        self._configure_trainable_parameters()

        self.optimizer = (
            self._build_optimizer()
        )

        self._reset_method_stats()

        self._reset_consensus_state()

    def _reset_consensus_state(
        self,
    ) -> None:

        self.gradient_memory: dict[
            int,
            list[torch.Tensor],
        ] = {}

        # Important:
        #
        # This counts gradients that passed the auxiliary-loss
        # gate for each class, matching the V1 semantics.
        self.class_observation_count: dict[
            int,
            int,
        ] = {
            metadata_class: 0
            for metadata_class in range(
                self.n_classes_aux
            )
        }

        self.n_warmup_updates = 0

        self.n_consensus_checked = 0
        self.n_consensus_pass = 0
        self.n_consensus_reject = 0

        self.sum_consensus_cosine = 0.0
        self.sum_positive_consensus_cosine = 0.0

        self.sum_raw_gradient_norm = 0.0
        self.sum_memory_gradient_norm = 0.0
        self.sum_consensus_gradient_norm = 0.0

        self.sum_parameter_gradient_norm = 0.0
        self.sum_consensus_parameter_delta = 0.0

    # ========================================================
    # TENSOR-LIST UTILITIES
    # ========================================================

    @staticmethod
    def _clone_tensor_list(
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:

        return [
            tensor
            .detach()
            .clone()

            for tensor in tensors
        ]

    @staticmethod
    def _tensor_list_dot(
        first: list[torch.Tensor],
        second: list[torch.Tensor],
    ) -> float:

        if len(first) != len(second):
            raise ValueError(
                "Tensor lists must have equal length."
            )

        result = 0.0

        for (
            first_tensor,
            second_tensor,
        ) in zip(
            first,
            second,
        ):

            result += float(
                torch.sum(
                    first_tensor
                    .detach()
                    .float()
                    *
                    second_tensor
                    .detach()
                    .float()
                ).item()
            )

        return float(
            result
        )

    @classmethod
    def _tensor_list_norm(
        cls,
        tensors: list[torch.Tensor],
    ) -> float:

        squared_norm = (
            cls._tensor_list_dot(
                tensors,
                tensors,
            )
        )

        return float(
            math.sqrt(
                max(
                    squared_norm,
                    0.0,
                )
            )
        )

    def _tensor_list_cosine(
        self,
        first: list[torch.Tensor],
        second: list[torch.Tensor],
    ) -> float:

        first_norm = (
            self._tensor_list_norm(
                first
            )
        )

        second_norm = (
            self._tensor_list_norm(
                second
            )
        )

        if (
            first_norm <= self.epsilon
            or second_norm <= self.epsilon
        ):
            return 0.0

        cosine = (
            self._tensor_list_dot(
                first,
                second,
            )
            /
            (
                first_norm
                * second_norm
                + self.epsilon
            )
        )

        return float(
            max(
                -1.0,
                min(
                    1.0,
                    cosine,
                ),
            )
        )

    # ========================================================
    # RAW METADATA GRADIENT
    # ========================================================

    def _compute_raw_metadata_gradient(
        self,
        x_tensor: torch.Tensor,
        y_aux_tensor: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
    ]:
        """
        Compute only the auxiliary metadata gradient.

        The source-anchor gradient is deliberately excluded
        because consensus is defined on the metadata signal.
        """

        shared_features = (
            self.model
            .extract_shared_features(
                x_tensor
            )
        )

        aux_logits = (
            self.model
            .aux_head(
                shared_features
            )
        )

        aux_loss = F.cross_entropy(
            aux_logits,
            y_aux_tensor,
        )

        gradients = torch.autograd.grad(
            aux_loss,
            self.trainable_parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        clean_gradients: list[
            torch.Tensor
        ] = []

        for (
            parameter,
            gradient,
        ) in zip(
            self.trainable_parameters,
            gradients,
        ):

            if gradient is None:

                clean_gradients.append(
                    torch.zeros_like(
                        parameter
                    )
                )

            else:

                clean_gradients.append(
                    gradient
                    .detach()
                    .clone()
                )

        return (
            aux_loss.detach(),
            clean_gradients,
        )

    # ========================================================
    # SOURCE ANCHOR GRADIENT
    # ========================================================

    def _source_anchor_gradients(
        self,
    ) -> list[torch.Tensor]:

        if self.regularization <= 0.0:

            return [
                torch.zeros_like(
                    parameter
                )

                for parameter
                in self.trainable_parameters
            ]

        return [
            (
                2.0
                * self.regularization
                * (
                    parameter
                    - source_parameter
                )
            )
            .detach()
            .clone()

            for (
                parameter,
                source_parameter,
            ) in zip(
                self.trainable_parameters,
                self.source_parameter_values,
            )
        ]

    # ========================================================
    # CLASS-CONDITIONED MEMORY
    # ========================================================

    def _update_gradient_memory(
        self,
        metadata_class: int,
        raw_gradient: list[torch.Tensor],
    ) -> None:
        """
        Update the EMA associated only with metadata_class.
        """

        metadata_class = int(
            metadata_class
        )

        if (
            metadata_class
            not in self.gradient_memory
        ):

            self.gradient_memory[
                metadata_class
            ] = (
                self._clone_tensor_list(
                    raw_gradient
                )
            )

            return

        old_memory = (
            self.gradient_memory[
                metadata_class
            ]
        )

        new_memory = [
            (
                self.consensus_beta
                * old
                +
                (
                    1.0
                    - self.consensus_beta
                )
                * current
            )

            for (
                old,
                current,
            ) in zip(
                old_memory,
                raw_gradient,
            )
        ]

        self.gradient_memory[
            metadata_class
        ] = (
            self._clone_tensor_list(
                new_memory
            )
        )

    # ========================================================
    # CONSENSUS GRADIENT
    # ========================================================

    def _build_consensus_gradient(
        self,
        raw_gradient: list[torch.Tensor],
        memory_gradient: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Construct Version-B consensus gradient.

        Stable configuration:

            0.50 * current
            +
            0.50 * memory

        Optionally preserve the raw gradient norm.
        """

        mixed_gradient = [
            (
                self.current_gradient_weight
                * current
                +
                self.memory_gradient_weight
                * memory
            )

            for (
                current,
                memory,
            ) in zip(
                raw_gradient,
                memory_gradient,
            )
        ]

        if not self.preserve_gradient_norm:

            return (
                mixed_gradient
            )

        raw_norm = (
            self._tensor_list_norm(
                raw_gradient
            )
        )

        mixed_norm = (
            self._tensor_list_norm(
                mixed_gradient
            )
        )

        if (
            raw_norm <= self.epsilon
            or mixed_norm <= self.epsilon
        ):

            return (
                mixed_gradient
            )

        scale = (
            raw_norm
            / mixed_norm
        )

        return [
            scale
            * gradient

            for gradient
            in mixed_gradient
        ]

    # ========================================================
    # APPLY GRADIENT
    # ========================================================

    def _apply_gradient(
        self,
        metadata_gradient: list[torch.Tensor],
    ) -> tuple[
        float,
        float,
    ]:
        """
        Add source-anchor gradient, clip and apply Adam update.
        """

        anchor_gradient = (
            self._source_anchor_gradients()
        )

        final_gradient = [
            metadata_component
            + anchor_component

            for (
                metadata_component,
                anchor_component,
            ) in zip(
                metadata_gradient,
                anchor_gradient,
            )
        ]

        parameters_before = [
            parameter
            .detach()
            .clone()
            .float()

            for parameter
            in self.trainable_parameters
        ]

        self.optimizer.zero_grad(
            set_to_none=True
        )

        for (
            parameter,
            gradient,
        ) in zip(
            self.trainable_parameters,
            final_gradient,
        ):

            parameter.grad = (
                gradient
                .detach()
                .clone()
            )

        parameter_gradient_norm = (
            self._gradient_norm()
        )

        if self.gradient_clip > 0.0:

            torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters,
                max_norm=(
                    self.gradient_clip
                ),
            )

        self.optimizer.step()

        parameter_delta = (
            self._parameter_delta_norm(
                parameters=(
                    self.trainable_parameters
                ),
                before=(
                    parameters_before
                ),
            )
        )

        return (
            float(
                parameter_gradient_norm
            ),
            float(
                parameter_delta
            ),
        )

    # ========================================================
    # OBSERVE / UPDATE
    # ========================================================

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

        self._record_observation()

        if y_aux is None:

            raise ValueError(
                "ConsensusTTA requires auxiliary labels."
            )

        x_tensor = (
            self._as_feature_tensor(
                x
            )
        )

        y_aux_tensor = (
            self._as_label_tensor(
                y_aux
            )
        )

        if len(x_tensor) != 1:

            raise ValueError(
                "ConsensusTTA is defined sample-by-sample "
                "and requires exactly one feature vector."
            )

        if len(y_aux_tensor) != 1:

            raise ValueError(
                "ConsensusTTA requires exactly one "
                "auxiliary label."
            )

        aux_label = int(
            y_aux_tensor[
                0
            ].item()
        )

        if (
            aux_label < 0
            or aux_label >= self.n_classes_aux
        ):

            return AdaptationResult(
                applied=False,
                diagnostics={
                    "reason":
                        "invalid_aux_label",
                },
            )

        # ====================================================
        # AUXILIARY LOSS GATE
        # ====================================================

        self.model.eval()

        with torch.no_grad():

            gate_loss = (
                self._auxiliary_loss(
                    x_tensor=x_tensor,
                    y_aux_tensor=(
                        y_aux_tensor
                    ),
                )
            )

        aux_loss_value = float(
            gate_loss.item()
        )

        normalized_aux_loss = float(
            aux_loss_value
            / max(
                self.aux_loss_normalizer,
                self.epsilon,
            )
        )

        self.sum_aux_loss += (
            aux_loss_value
        )

        self.sum_normalized_aux_loss += (
            normalized_aux_loss
        )

        # IMPORTANT:
        #
        # Below-threshold observations do NOT enter the
        # class-conditioned memory and do NOT count toward
        # warmup. This matches the stable V1 implementation.
        if (
            normalized_aux_loss
            < self.normalized_aux_loss_threshold
        ):

            return AdaptationResult(
                applied=False,
                diagnostics={
                    "reason":
                        "below_aux_threshold",

                    "aux_loss":
                        aux_loss_value,

                    "normalized_aux_loss":
                        normalized_aux_loss,

                    "metadata_class":
                        aux_label,
                },
            )

        self.n_aux_threshold_pass += 1

        # ====================================================
        # RAW METADATA GRADIENT
        # ====================================================

        (
            aux_loss,
            raw_gradient,
        ) = (
            self._compute_raw_metadata_gradient(
                x_tensor=x_tensor,
                y_aux_tensor=(
                    y_aux_tensor
                ),
            )
        )

        raw_gradient_norm = (
            self._tensor_list_norm(
                raw_gradient
            )
        )

        class_count_before = int(
            self.class_observation_count[
                aux_label
            ]
        )

        memory_exists = (
            aux_label
            in self.gradient_memory
        )

        memory_before = (
            self._clone_tensor_list(
                self.gradient_memory[
                    aux_label
                ]
            )
            if memory_exists
            else None
        )

        # ====================================================
        # CLASS WARMUP
        # ====================================================

        if (
            class_count_before
            < self.class_warmup_updates
            or memory_before is None
        ):

            (
                parameter_gradient_norm,
                parameter_delta,
            ) = (
                self._apply_gradient(
                    raw_gradient
                )
            )

            self._record_update()

            self.n_warmup_updates += 1

            self.sum_raw_gradient_norm += (
                raw_gradient_norm
            )

            # During warmup the consensus gradient is simply
            # the raw gradient.
            self.sum_consensus_gradient_norm += (
                raw_gradient_norm
            )

            self.sum_parameter_gradient_norm += (
                parameter_gradient_norm
            )

            self.sum_consensus_parameter_delta += (
                parameter_delta
            )

            # V1 ordering:
            #
            # parameter update
            # -> memory update
            # -> class count increment
            self._update_gradient_memory(
                metadata_class=(
                    aux_label
                ),
                raw_gradient=(
                    raw_gradient
                ),
            )

            self.class_observation_count[
                aux_label
            ] = (
                class_count_before
                + 1
            )

            return AdaptationResult(
                applied=True,
                diagnostics={
                    "reason":
                        "warmup_update",

                    "metadata_class":
                        aux_label,

                    "class_count_before":
                        class_count_before,

                    "class_count_after":
                        (
                            class_count_before
                            + 1
                        ),

                    "warmup":
                        True,

                    "consensus_checked":
                        False,

                    "consensus_pass":
                        False,

                    "consensus_cosine":
                        0.0,

                    "aux_loss":
                        float(
                            aux_loss.item()
                        ),

                    "normalized_aux_loss":
                        normalized_aux_loss,

                    "raw_gradient_norm":
                        raw_gradient_norm,

                    "memory_gradient_norm":
                        0.0,

                    "consensus_gradient_norm":
                        raw_gradient_norm,

                    "parameter_gradient_norm":
                        parameter_gradient_norm,

                    "parameter_delta":
                        parameter_delta,
                },
            )

        # ====================================================
        # CONSENSUS CHECK
        # ====================================================

        assert memory_before is not None

        memory_gradient_norm = (
            self._tensor_list_norm(
                memory_before
            )
        )

        consensus_cosine = (
            self._tensor_list_cosine(
                raw_gradient,
                memory_before,
            )
        )

        self.n_consensus_checked += 1

        self.sum_consensus_cosine += (
            consensus_cosine
        )

        self.sum_raw_gradient_norm += (
            raw_gradient_norm
        )

        self.sum_memory_gradient_norm += (
            memory_gradient_norm
        )

        consensus_pass = bool(
            consensus_cosine
            >= self.min_consensus_cosine
        )

        # ====================================================
        # UPDATE MEMORY REGARDLESS OF PARAMETER UPDATE
        # ====================================================
        #
        # This is a key part of the V1 algorithm:
        #
        # a rejected gradient still updates the metadata-class
        # memory, allowing the memory itself to track drift.
        #
        # The consensus test above used memory_before, so the
        # current sample cannot influence its own acceptance.
        # ====================================================

        self._update_gradient_memory(
            metadata_class=(
                aux_label
            ),
            raw_gradient=(
                raw_gradient
            ),
        )

        self.class_observation_count[
            aux_label
        ] = (
            class_count_before
            + 1
        )

        # ====================================================
        # CONSENSUS REJECT
        # ====================================================

        if not consensus_pass:

            self.n_consensus_reject += 1

            return AdaptationResult(
                applied=False,
                diagnostics={
                    "reason":
                        "consensus_reject",

                    "metadata_class":
                        aux_label,

                    "class_count_before":
                        class_count_before,

                    "class_count_after":
                        (
                            class_count_before
                            + 1
                        ),

                    "warmup":
                        False,

                    "consensus_checked":
                        True,

                    "consensus_pass":
                        False,

                    "consensus_cosine":
                        consensus_cosine,

                    "aux_loss":
                        float(
                            aux_loss.item()
                        ),

                    "normalized_aux_loss":
                        normalized_aux_loss,

                    "raw_gradient_norm":
                        raw_gradient_norm,

                    "memory_gradient_norm":
                        memory_gradient_norm,

                    "consensus_gradient_norm":
                        0.0,

                    "parameter_gradient_norm":
                        0.0,

                    "parameter_delta":
                        0.0,
                },
            )

        # ====================================================
        # CONSENSUS PASS
        # ====================================================

        self.n_consensus_pass += 1

        self.sum_positive_consensus_cosine += (
            consensus_cosine
        )

        consensus_gradient = (
            self._build_consensus_gradient(
                raw_gradient=(
                    raw_gradient
                ),
                memory_gradient=(
                    memory_before
                ),
            )
        )

        consensus_gradient_norm = (
            self._tensor_list_norm(
                consensus_gradient
            )
        )

        (
            parameter_gradient_norm,
            parameter_delta,
        ) = (
            self._apply_gradient(
                consensus_gradient
            )
        )

        self._record_update()

        self.sum_consensus_gradient_norm += (
            consensus_gradient_norm
        )

        self.sum_parameter_gradient_norm += (
            parameter_gradient_norm
        )

        self.sum_consensus_parameter_delta += (
            parameter_delta
        )

        return AdaptationResult(
            applied=True,
            diagnostics={
                "reason":
                    "consensus_update",

                "metadata_class":
                    aux_label,

                "class_count_before":
                    class_count_before,

                "class_count_after":
                    (
                        class_count_before
                        + 1
                    ),

                "warmup":
                    False,

                "consensus_checked":
                    True,

                "consensus_pass":
                    True,

                "consensus_cosine":
                    consensus_cosine,

                "aux_loss":
                    float(
                        aux_loss.item()
                    ),

                "normalized_aux_loss":
                    normalized_aux_loss,

                "raw_gradient_norm":
                    raw_gradient_norm,

                "memory_gradient_norm":
                    memory_gradient_norm,

                "consensus_gradient_norm":
                    consensus_gradient_norm,

                "parameter_gradient_norm":
                    parameter_gradient_norm,

                "parameter_delta":
                    parameter_delta,
            },
        )

    # ========================================================
    # GLOBAL DIAGNOSTICS
    # ========================================================

    def diagnostics(
        self,
    ) -> dict[str, Any]:

        # Do not call MetadataTTA.diagnostics() here because
        # Consensus maintains its parameter-gradient/delta
        # aggregates with V1-compatible denominators.
        diagnostics = {
            "method":
                self.method_name,

            "n_observations":
                self.number_of_observations,

            "n_updates":
                self.number_of_updates,

            "update_rate":
                self.update_rate,
        }

        n_observations = max(
            self.number_of_observations,
            1,
        )

        threshold_pass = max(
            self.n_aux_threshold_pass,
            1,
        )

        consensus_checked = max(
            self.n_consensus_checked,
            1,
        )

        consensus_pass = max(
            self.n_consensus_pass,
            1,
        )

        updates = max(
            self.number_of_updates,
            1,
        )

        diagnostics.update(
            {
                "aux_threshold_pass_count":
                    int(
                        self.n_aux_threshold_pass
                    ),

                "aux_threshold_pass_rate":
                    float(
                        self.n_aux_threshold_pass
                        / n_observations
                    ),

                "warmup_update_count":
                    int(
                        self.n_warmup_updates
                    ),

                "warmup_update_rate":
                    float(
                        self.n_warmup_updates
                        / threshold_pass
                    ),

                "consensus_checked_count":
                    int(
                        self.n_consensus_checked
                    ),

                "consensus_pass_count":
                    int(
                        self.n_consensus_pass
                    ),

                "consensus_reject_count":
                    int(
                        self.n_consensus_reject
                    ),

                "consensus_pass_rate":
                    float(
                        self.n_consensus_pass
                        / consensus_checked
                    ),

                "consensus_reject_rate":
                    float(
                        self.n_consensus_reject
                        / consensus_checked
                    ),

                "mean_consensus_cosine":
                    float(
                        self.sum_consensus_cosine
                        / consensus_checked
                    ),

                "mean_positive_consensus_cosine":
                    float(
                        self.sum_positive_consensus_cosine
                        / consensus_pass
                    ),

                "mean_aux_loss":
                    float(
                        self.sum_aux_loss
                        / n_observations
                    ),

                "mean_normalized_aux_loss":
                    float(
                        self.sum_normalized_aux_loss
                        / n_observations
                    ),

                "mean_raw_gradient_norm":
                    float(
                        self.sum_raw_gradient_norm
                        / threshold_pass
                    ),

                "mean_memory_gradient_norm":
                    float(
                        self.sum_memory_gradient_norm
                        / consensus_checked
                    ),

                "mean_consensus_gradient_norm":
                    float(
                        self.sum_consensus_gradient_norm
                        / updates
                    ),

                "mean_parameter_gradient_norm":
                    float(
                        self.sum_parameter_gradient_norm
                        / updates
                    ),

                "mean_parameter_delta":
                    float(
                        self.sum_consensus_parameter_delta
                        / updates
                    ),

                "active_metadata_memories":
                    int(
                        len(
                            self.gradient_memory
                        )
                    ),
            }
        )

        return diagnostics