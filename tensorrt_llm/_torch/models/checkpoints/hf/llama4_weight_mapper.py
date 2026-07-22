# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    WeightGroup
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Llama4ForConditionalGeneration")
class Llama4HfWeightMapper(HfWeightMapper):
    """
    Weight mapper for Llama4ForConditionalGeneration that handles the
    'language_model.' prefix removal from weight keys.
    """

    _VISION_PREFIXES = ("vision_model.", "multi_modal_projector.")

    @property
    def borrowed_source_tensors_safe(self) -> bool:
        """Whether this Llama4 profile can release source views per group."""
        return self._borrowed_source_tensor_config_is_safe()

    @staticmethod
    def _canonical_source_weight_name(key: str) -> str:
        if key.startswith("language_model."):
            return key[len("language_model."):]
        return key

    def _source_weight_group_id(self, key: str) -> str:
        if key.startswith(self._VISION_PREFIXES):
            # Llama4VisionEncoder.load_state_dict requires both the vision
            # tower and projector parameter sets at once.
            return "llama4.vision_and_projector"

        # Llama 4 Scout and Maverick are routed-MoE models. Use the
        # config-aware dependency root so router, shared/expert projections,
        # and scales for one MLP are presented in one partial-load call.
        return f"llama4.{self._incremental_dependency_root(key)}"

    def get_weight_groups(self,
                          keys: Iterable[str]) -> list[WeightGroup] | None:
        """Return atomic Llama4 vision and text dependency groups."""
        grouped_keys: dict[str, list[str]] = {}
        seen_keys = set()
        for key in keys:
            if key in seen_keys:
                raise ValueError(f"Duplicate checkpoint tensor name {key!r}")
            seen_keys.add(key)
            group_id = self._source_weight_group_id(key)
            grouped_keys.setdefault(group_id, []).append(key)

        return [
            WeightGroup(group_id=group_id, keys=tuple(group_keys))
            for group_id, group_keys in grouped_keys.items()
        ]

    def get_incremental_load_roots(
            self, keys: Iterable[str]) -> tuple[str, ...] | None:
        """Dispatch text dependencies without traversing all decoder layers."""
        return self._get_hf_incremental_load_roots(keys)

    def filter_weights(self, prefix: str, weights: dict) -> dict:
        transformed_weights = {}
        for key, value in weights.items():
            if key.startswith("language_model."):
                new_key = key[len("language_model."):]
                transformed_weights[new_key] = value
            else:
                transformed_weights[key] = value

        return super().filter_weights(prefix, transformed_weights)
