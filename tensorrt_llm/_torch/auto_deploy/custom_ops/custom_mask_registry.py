# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Mask Generator Registry for model and backend-specific attention mask generation.

This module provides a registry for custom attention mask generators that are
keyed by (model_type, backend). This allows different model/backend combinations
to have their own mask generation logic.

The registry is used during KV cache transformation to replace custom_mask_marker
ops with actual mask computation.

Flow:
1. Export time: custom_mask_marker inserted with model_type, layer_idx, metadata
2. KV cache transform:
   - Extract model_type from marker
   - Determine backend from attention descriptor
   - Look up generator: Registry[(model_type, backend)]
   - Generator activates required args in SequenceInfo
   - Generator adds graph inputs at transform time
   - Generator creates mask subgraph
   - Replace marker with mask node
3. Runtime:
   - SequenceInfo provides activated args
   - Mask op executes with those inputs

Usage:
    # Register a mask generator for (model_type, backend)
    @CustomMaskGeneratorRegistry.register("gemma3_text", "flashinfer")
    def gemma3_flashinfer_generator(gm, cm, layer_idx, metadata, attn_descriptor):
        # Add inputs at transform time (skip activate for _extra_args inputs)
        # Note: _ad_ prefix bypasses outer VLM model kwargs consumption
        _ad_token_type_ids_node = add_graph_input(gm, "_ad_token_type_ids")
        # Build mask computation, return mask node
        ...

    # Look up generator at transform time
    generator = CustomMaskGeneratorRegistry.get("gemma3_text", "flashinfer")
    if generator:
        mask_node = generator(gm, cm, layer_idx, metadata, attn_descriptor)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from torch.fx import GraphModule, Node

# Type alias for mask generator functions
# Args: (gm, cm, layer_idx, metadata, attn_descriptor, meta_nodes_std)
# Returns: mask Node in the graph
CustomMaskGeneratorFn = Callable[
    [GraphModule, Any, int, Dict[str, Any], Any, List[Node]],
    Node,
]


class CustomMaskGeneratorRegistry:
    """Registry for (model_type, backend) → mask generator functions.

    This registry enables model-specific and backend-specific custom attention
    mask generation. The generator is called during KV cache transformation
    to replace custom_mask_marker ops with actual mask computation.

    Key design principles:
    - Generator is self-contained: activates args, adds inputs, creates mask
    - No special metadata tracking in infrastructure
    - SequenceInfo is single source of truth for runtime inputs

    Example:
        @CustomMaskGeneratorRegistry.register("gemma3_text", "flashinfer")
        def gemma3_flashinfer_generator(gm, cm, layer_idx, metadata, attn_descriptor, meta_nodes_std):
            # Generator handles everything:
            # 1. Activate required args in SequenceInfo
            # 2. Add graph inputs at transform time
            # 3. Create mask computation
            # 4. Cache to avoid duplicate masks
            return mask_node
    """

    _registry: Dict[Tuple[str, str], CustomMaskGeneratorFn] = {}

    @classmethod
    def register(
        cls, model_type: str, backend: str
    ) -> Callable[[CustomMaskGeneratorFn], CustomMaskGeneratorFn]:
        """Decorator to register a mask generator for (model_type, backend).

        Args:
            model_type: Model type identifier (e.g., "gemma3_text", "qwen2_vl").
            backend: Backend identifier (e.g., "flashinfer", "triton", "torch").

        Returns:
            Decorator function that registers the mask generator.

        Example:
            @CustomMaskGeneratorRegistry.register("gemma3_text", "flashinfer")
            def my_generator(gm, cm, layer_idx, metadata, attn_descriptor, meta_nodes_std):
                ...
        """

        def decorator(fn: CustomMaskGeneratorFn) -> CustomMaskGeneratorFn:
            cls._registry[(model_type, backend)] = fn
            return fn

        return decorator

    @classmethod
    def get(
        cls, model_type: Optional[str], backend: Optional[str]
    ) -> Optional[CustomMaskGeneratorFn]:
        """Get the mask generator for (model_type, backend).

        Args:
            model_type: Model type identifier.
            backend: Backend identifier.

        Returns:
            The registered generator function, or None if not found.
        """
        if model_type is None or backend is None:
            return None
        return cls._registry.get((model_type, backend))

    @classmethod
    def has(cls, model_type: str, backend: str) -> bool:
        """Check if a generator is registered for (model_type, backend).

        Args:
            model_type: Model type identifier.
            backend: Backend identifier.

        Returns:
            True if a generator is registered, False otherwise.
        """
        return (model_type, backend) in cls._registry

    @classmethod
    def registered_keys(cls) -> List[Tuple[str, str]]:
        """Return a list of all registered (model_type, backend) pairs."""
        return list(cls._registry.keys())

    @classmethod
    def registered_model_types(cls) -> List[str]:
        """Return a list of unique registered model types."""
        return list(set(k[0] for k in cls._registry.keys()))

    @classmethod
    def registered_backends(cls) -> List[str]:
        """Return a list of unique registered backends."""
        return list(set(k[1] for k in cls._registry.keys()))
