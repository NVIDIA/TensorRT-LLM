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

"""Custom mask marker op for export-time metadata capture.

This module provides a marker op that is inserted during export to capture
layer-specific metadata for custom attention mask generation. The marker
is replaced with actual mask computation during KV cache transformation.

Key features:
- Captures arbitrary metadata at export time when layer info is available
- Replaced by backend-specific mask generator during KV cache transform
- No heuristics needed at transform time - all info captured in marker

Flow:
1. Export time: Marker inserted per attention layer with model_type, layer_idx, metadata
2. KV cache transform: Marker replaced with actual mask from registered generator

Note: metadata is passed as a JSON string because torch.library.custom_op
only supports primitive types (str, int, float, bool) and tensors.
"""

import json

import torch
from torch import Tensor


@torch.library.custom_op("auto_deploy::custom_mask_marker", mutates_args=())
def custom_mask_marker(
    model_type: str,
    layer_idx: int,
    metadata_json: str,
) -> Tensor:
    """Marker op for custom attention mask generation.

    This op is inserted during export to capture layer-specific metadata.
    It is replaced by actual mask computation during KV cache transformation.

    Args:
        model_type: Model type identifier for registry lookup (e.g., "gemma3_text").
        layer_idx: Layer index for this attention layer.
        metadata_json: JSON-serialized metadata dict - contents defined by the export patch.
            Common keys: sliding_window, attention_type, etc.

    Returns:
        Placeholder tensor - replaced with actual mask at transform time.

    Raises:
        RuntimeError: If called at runtime without being replaced.
    """
    metadata = json.loads(metadata_json)
    raise RuntimeError(
        "custom_mask_marker should be replaced during KV cache transformation. "
        f"model_type={model_type}, layer_idx={layer_idx}, metadata={metadata}"
    )


@custom_mask_marker.register_fake
def custom_mask_marker_fake(
    model_type: str,
    layer_idx: int,
    metadata_json: str,
) -> Tensor:
    """Fake implementation for tracing.

    Returns an empty bool tensor as placeholder. The actual shape
    is determined at transform time when the marker is replaced.
    """
    # Return empty tensor on meta device for tracing
    # Actual shape depends on runtime batch structure
    return torch.empty(0, dtype=torch.bool, device="meta")
