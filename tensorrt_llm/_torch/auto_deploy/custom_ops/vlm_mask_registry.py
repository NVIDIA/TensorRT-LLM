# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VLM Mask Generator Registry for model-specific attention mask generation.

This module provides a registry for VLM (Vision-Language Model) mask generators,
allowing different models (Gemma3, Qwen2-VL, etc.) to register their own
custom mask generation logic.

Usage:
    # Register a mask generator for a model type
    @VlmMaskGeneratorRegistry.register("gemma3")
    def generate_gemma3_mask(token_info, qo_indptr, seq_len, sliding_window):
        ...
        return custom_mask

    # Look up and use a mask generator
    mask_gen = VlmMaskGeneratorRegistry.get("gemma3")
    if mask_gen:
        mask = mask_gen(token_info, qo_indptr, seq_len, sliding_window)
"""

from typing import Callable, Dict, Optional

from torch import Tensor


class VlmMetadataKeys:
    """Attribute names for VLM metadata on GraphModules and Module classes.

    These constants define the contract between:
    - Model patches (set MODULE_INPUT_NAMES on Module class)
    - Export/transforms (read MODULE_INPUT_NAMES, set GRAPH_* on GraphModule)
    - Runtime/ADExecutor (read GRAPH_* from GraphModule)

    Using constants prevents typos and provides a single source of truth.
    """

    # Set on GraphModule by export/transforms, read by ADExecutor and kvcache
    GRAPH_INPUTS = "_vlm_inputs"
    """List of VLM input names that need to be injected at runtime (e.g., ["token_type_ids"])."""

    GRAPH_MODEL_TYPE = "_vlm_model_type"
    """Model type string for mask generator dispatch (e.g., "gemma3")."""

    # Set on Module class by patches, read by export/transforms
    MODULE_INPUT_NAMES = "_vlm_input_names"
    """List of VLM input names that the patch injects during export."""


# Type alias for mask generator functions
# Args: (token_info, qo_indptr, seq_len, sliding_window)
# Returns: custom_mask
# Note: sliding_window may be ignored by backends with native sliding window support.
VlmMaskGeneratorFn = Callable[
    [Tensor, Tensor, Tensor, int],
    Tensor,
]


class VlmMaskGeneratorRegistry:
    """Registry for VLM-specific attention mask generators.

    Different VLM models may require different attention masking strategies
    (e.g., Gemma3 uses bidirectional attention for image tokens). This registry
    allows each model to register its own mask generation function, which is
    then looked up at runtime based on model_type.

    The sliding_window parameter is provided for backends without native sliding
    window support. Backends like FlashInfer may ignore it and use window_left.
    """

    _registry: Dict[str, VlmMaskGeneratorFn] = {}

    @classmethod
    def register(cls, model_type: str) -> Callable[[VlmMaskGeneratorFn], VlmMaskGeneratorFn]:
        """Decorator to register a mask generator for a specific model type.

        Args:
            model_type: The model type identifier (e.g., "gemma3", "qwen2_vl").
                Should match the model_type from HuggingFace config.

        Returns:
            Decorator function that registers the mask generator.

        Example:
            @VlmMaskGeneratorRegistry.register("gemma3")
            def generate_gemma3_mask(token_info, qo_indptr, seq_len, sliding_window):
                ...
        """

        def decorator(fn: VlmMaskGeneratorFn) -> VlmMaskGeneratorFn:
            cls._registry[model_type] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, model_type: Optional[str]) -> Optional[VlmMaskGeneratorFn]:
        """Get the mask generator for a model type.

        Args:
            model_type: The model type identifier.

        Returns:
            The registered mask generator function, or None if not found.
        """
        if model_type is None:
            return None
        return cls._registry.get(model_type)

    @classmethod
    def has(cls, model_type: str) -> bool:
        """Check if a mask generator is registered for a model type.

        Args:
            model_type: The model type identifier.

        Returns:
            True if a generator is registered, False otherwise.
        """
        return model_type in cls._registry

    @classmethod
    def registered_model_types(cls) -> list:
        """Return a list of all registered model types."""
        return list(cls._registry.keys())
