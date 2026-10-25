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
"""FlashInfer JIT module of the rotary embedding layers.

``RotaryEmbedding`` applies RoPE to a query and key pair with FlashInfer's
``apply_rope_with_cos_sin_cache_inplace`` whenever FlashInfer is available,
whose module (``rope``) FlashInfer builds at the first call: about 27 s of
nvcc per rank on a cold cache, in warmup's first forward. JIT compile ahead
names it here once the model is constructed, so a FlashInfer helper builds it
while the weights load (``jit_prefetch_flashinfer_helper``).
"""

from __future__ import annotations

from typing import List


def flashinfer_modules(model) -> List[str]:
    """``["rope"]`` if ``model`` holds a ``RotaryEmbedding`` and FlashInfer
    is available, else none."""
    from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
    from .rotary_embedding import RotaryEmbedding

    if IS_FLASHINFER_AVAILABLE and any(isinstance(m, RotaryEmbedding) for m in model.modules()):
        return ["rope"]
    return []
