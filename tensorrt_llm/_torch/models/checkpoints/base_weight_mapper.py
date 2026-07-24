# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    WeightGroup
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM


class BaseWeightMapper(ABC):

    # A mapper must opt in explicitly when every source tensor can be loaded
    # independently. Most model families fuse or jointly transform checkpoint
    # tensors, so the conservative default requires an atomic group manifest.
    single_tensor_groups_safe = False

    @property
    def borrowed_source_tensors_safe(self) -> bool:
        """Whether incremental loading may borrow transport-owned tensors.

        The conservative default requires rank-local staging. Opting in means
        mapper callbacks and model loading neither mutate source tensors nor
        retain their storage after ``load_weights`` returns. The orchestrator
        must still wait for asynchronous H2D reads before releasing the source
        batch.
        """
        return False

    def __init__(self):
        self._callbacks: list[Callable] = []
        self._mapping: dict = {}
        self._skip_modules = []
        self._model: Union[nn.Module, DecoderModelForCausalLM] | None = None
        self._config: ModelConfig | None = None
        self._incremental_expected_group_ids: tuple[str, ...] | None = None
        self._incremental_loaded_group_ids: set[str] = set()

    def init_model_and_config(self, model: Union[nn.Module,
                                                 DecoderModelForCausalLM],
                              config: ModelConfig):
        self._model = model
        self._config = config

        if not hasattr(model, 'model_config') or not isinstance(
                model.model_config, ModelConfig):
            raise ValueError("model must have a model_config attribute")
        if not hasattr(model, 'config'):
            raise ValueError("model must have a config attribute")

        self._tp_size = 1 if model.model_config.mapping.enable_attention_dp else model.model_config.mapping.tp_size

        self.map_weights()

    def cleanup(self) -> None:
        self._model = None
        self._config = None
        self._incremental_expected_group_ids = None
        self._incremental_loaded_group_ids.clear()

    def get_weight_groups(self,
                          keys: Iterable[str]) -> list[WeightGroup] | None:
        """Return source tensors that must be materialized atomically.

        The returned groups must partition ``keys`` exactly once. The default
        is intentionally unsupported because a generic mapper cannot prove
        that tensors are independent across model-specific fusion,
        quantization, or alias transformations.

        This hook is evaluated after :meth:`init_model_and_config`, allowing a
        mapper to account for the model configuration and rank mapping.
        """
        del keys
        return None

    def get_incremental_load_roots(
            self, keys: Iterable[str]) -> tuple[str, ...] | None:
        """Return model subtrees that can consume an incremental source group.

        Returning ``None`` selects the conservative full-model traversal. A
        mapper should override this only when it can map every source tensor
        name to a destination subtree without omitting model-specific fusion,
        quantization, or callback handling. The returned roots may overlap;
        the loader de-duplicates their module traversal.

        Args:
            keys: Tensor names after mapper preprocessing and optional
                ``params_map`` renaming.

        Returns:
            Destination module roots, or ``None`` when a full traversal is
            required.
        """
        del keys
        return None

    def begin_incremental_load(self, groups: Sequence[WeightGroup]) -> None:
        """Start source-manifest validation for an incremental load.

        This lifecycle validates that every declared source group is presented
        exactly once. It deliberately does not prove destination-parameter
        completeness; each model-family opt-in remains responsible for
        choosing groups that preserve its loading invariants.
        """
        if self._incremental_expected_group_ids is not None:
            raise RuntimeError("An incremental weight load is already active")

        group_ids = tuple(group.group_id for group in groups)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("Incremental weight group IDs must be unique")
        self._incremental_expected_group_ids = group_ids
        self._incremental_loaded_group_ids.clear()

    def record_incremental_group_loaded(self, group_id: str) -> None:
        """Record one successfully materialized source dependency group."""
        if self._incremental_expected_group_ids is None:
            raise RuntimeError("No incremental weight load is active")
        if group_id not in self._incremental_expected_group_ids:
            raise ValueError(f"Unknown incremental weight group {group_id!r}")
        if group_id in self._incremental_loaded_group_ids:
            raise ValueError(
                f"Incremental weight group {group_id!r} was loaded twice")
        self._incremental_loaded_group_ids.add(group_id)

    def finalize_incremental_load(self) -> None:
        """Validate complete source-group coverage and close the lifecycle."""
        if self._incremental_expected_group_ids is None:
            raise RuntimeError("No incremental weight load is active")

        missing = [
            group_id for group_id in self._incremental_expected_group_ids
            if group_id not in self._incremental_loaded_group_ids
        ]
        if missing:
            missing_sample = ", ".join(missing[:8])
            if len(missing) > 8:
                missing_sample += ", ..."
            raise RuntimeError(
                "Incremental weight load did not materialize "
                f"{len(missing)} source groups: {missing_sample}")

        self._incremental_expected_group_ids = None
        self._incremental_loaded_group_ids.clear()

    def abort_incremental_load(self) -> None:
        """Reset incremental validation without masking the primary error."""
        self._incremental_expected_group_ids = None
        self._incremental_loaded_group_ids.clear()

    @abstractmethod
    def map_weights(self) -> None:
        """
        Maps weights from TRT-LLM to a source state dictionary (e.g., Hugging Face)
        """

    @abstractmethod
    def apply_callbacks(self, module: nn.Module, module_name: str,
                        module_names_breakdown: list[str],
                        weights: dict) -> list[dict]:
        """
        Applies a series of transformation functions to an internal representation
        of weights or to guide the mapping process. The exact behavior might depend
        on the implementation (e.g., storing callbacks to be applied later).

        Args:
            module: The module to apply the callbacks to
            module_name: The specific module name (e.g., 'qkv_proj', 'gate_up_proj')
            module_names_breakdown: List of module path components for building full paths
            weights: The weights dictionary to process
        """

    def rename_by_params_map(self, params_map: dict[str, str],
                             weights: dict) -> dict:
        """
        Rename weight keys according to regex pattern matching.

        Args:
            pattern_mapping: A dictionary mapping regex patterns to replacement strings. The key is HF name pattern, and the value is corresponding TRT-LLM name pattern.
                The patterns will be used to match keys in the weights dict and replace
                them according to the replacement string, which can use regex backreferences.
                Example:
                HF name: vision_model.encoder.layers.1.self_attn.out_proj.{weight,bias}
                TRT-LLM name: vision_model.encoder.layers.1.self_attn.o_proj.{weight,bias}
                Then the pattern_mapping could be:
                pattern_mapping = {
                    r'(.*?)out_proj(.*)': r'\1o_proj\2'
                }
            weights: A dictionary of weights (or ConsumableWeightsDict)

        Returns:
            A dictionary of weights with renamed keys (preserves ConsumableWeightsDict if input was one)
        """
        import re

        from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
            ConsumableWeightsDict

        # Check if input is a ConsumableWeightsDict to preserve the type
        is_consumable = isinstance(weights, ConsumableWeightsDict)

        # Create a new dictionary to store the renamed weights
        renamed_weights = {}

        # Keep track of keys that have been matched by a pattern
        matched_keys = set()

        # Process each key in the weights dictionary
        for key in list(weights.keys()):
            # Check each pattern for a match
            for pattern, replacement in params_map.items():
                if re.match(pattern, key):
                    # Create the new key by applying the regex replacement
                    new_key = re.sub(pattern, replacement, key)
                    # Store the weight with the new key
                    renamed_weights[new_key] = weights[key]
                    matched_keys.add(key)
                    break

            # If the key wasn't matched by any pattern, keep it as is
            if key not in matched_keys:
                renamed_weights[key] = weights[key]

        # Preserve ConsumableWeightsDict type if that's what was passed in
        if is_consumable:
            return ConsumableWeightsDict(renamed_weights)
        return renamed_weights

    def preprocess_weights(self, weights: dict) -> dict:
        """
        Preprocess weights before starting the loading process.
        """
        ...

    def handle_manual_copy(self,
                           module_name: str,
                           module_weights: dict,
                           n: str,
                           p: nn.Parameter,
                           allow_partial_loading: bool = False) -> None:
        if not allow_partial_loading:
            assert n in module_weights
        if n in module_weights:
            p.data.copy_(module_weights[n][:])

    def does_require_special_handling(self, module_name: str) -> bool:
        return module_name in self.mapping

    def is_special_instance_module(self, module: nn.Module) -> bool:
        return False

    def handle_special_instance_module(
            self,
            module: nn.Module,
            module_name: str,
            module_weights: dict,
            allow_partial_loading: bool = False) -> None:
        raise NotImplementedError()

    @property
    def skip_modules(self) -> List[str]:
        return self._skip_modules

    def add_skip_modules(self, value: List[str]) -> None:
        self._skip_modules.extend(value)

    def should_skip_module(self, module_name: str) -> bool:
        return any(skip_module in module_name
                   for skip_module in self._skip_modules)

    def filter_weights(self, prefix: str, weights: dict) -> dict:
        result = {}
        for k, v in weights.items():
            if k.startswith(prefix):
                new_k = k[len(prefix) + 1:]
                result[new_k] = v
        return result

    @property
    def mapping(self) -> dict:
        return self._mapping

    @property
    def config(self) -> ModelConfig:
        if self._config is None:
            raise RuntimeError("Weight mapper is not initialized")
        return self._config

    @property
    def model(self) -> Union[nn.Module, DecoderModelForCausalLM]:
        if self._model is None:
            raise RuntimeError("Weight mapper is not initialized")
        return self._model

    @property
    def _head_dim(self) -> int:
        model = self.model
        head_dim = model.config.head_dim if hasattr(
            model.config, 'head_dim'
        ) and model.config.head_dim is not None else model.config.hidden_size // model.config.num_attention_heads
        return head_dim
