# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Engine-path support for the Qwen3.5/Qwen3.6 hybrid MoE family.

This package targets the *classic TensorRT-engine* flow (convert checkpoint ->
``trtllm-build`` -> serialized ``.engine``) for checkpoints whose HF
``architectures[0]`` is ``Qwen3_5MoeForConditionalGeneration`` (``model_type``
``qwen3_5_moe``), e.g. ``nvidia/Qwen3.6-35B-A3B-NVFP4``.

The architecture is a Qwen3-Next-style hybrid: a 3:1 interleave of Gated
DeltaNet "linear attention" layers and full-attention layers, with a 256-expert
MoE (plus shared expert) per layer. This initial engine-path drop covers the
text-generation path; the MTP head and the Qwen3-VL vision tower are not built
here.
"""

from .config import Qwen3NextConfig
from .model import Qwen3NextForCausalLM

__all__ = [
    "Qwen3NextConfig",
    "Qwen3NextForCausalLM",
]
