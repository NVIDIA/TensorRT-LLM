# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sage attention shared by the PrimTS decode and context kernels.

* ``config``: the public recipe (``SageAttentionConfig``) and per-run scale
  tensors (``SageAttentionParams``), their flat layout, shapes and validation,
  and the INT8 Q/K device gate.
* ``layout``: the word layout of the scale buffer a plan prepares before
  attention (``_SageScaleLayout``).
* ``prepare``: the kernel that writes that buffer from the flat scales.
* ``index``: device-side index math and loads over the layout.
* ``modes``: the names of how a softmax obtains and applies the K scales.
* ``int32_scores``: INT8 Q/K accumulated as INT32 scores on a float bias.
* ``plan``: the host-side state of a plan's prepared buffer and its compiled
  prepare launch, for plans that run the prepare as its own launch.

Each attention kernel keeps its own resources, pipelines and softmax for
the scales; only these components are shared. This package imports the
host-side modules only; kernels import ``prepare``, ``index`` and
``int32_scores`` directly.
"""

from .config import (
    SAGE_K_BLOCK_SIZES,
    SageAttentionConfig,
    SageAttentionParams,
    is_power_of_two,
    sage_flat_scale_numel,
    sage_flat_scale_slot,
    sage_log2_block_size,
    sage_scale_shapes,
    validate_sage_params,
    validate_sage_recipe,
)

__all__ = [
    "SAGE_K_BLOCK_SIZES",
    "SageAttentionConfig",
    "SageAttentionParams",
    "is_power_of_two",
    "sage_flat_scale_numel",
    "sage_flat_scale_slot",
    "sage_log2_block_size",
    "sage_scale_shapes",
    "validate_sage_params",
    "validate_sage_recipe",
]
