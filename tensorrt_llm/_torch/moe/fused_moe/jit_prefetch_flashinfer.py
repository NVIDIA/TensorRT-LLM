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
"""FlashInfer JIT module of the TRTLLM-Gen MoE layers that run on FlashInfer.

A TRTLLM-Gen MoE layer whose op backend is FlashInfer's
(``moe_op_backend.FlashinferOpBackend``) runs FlashInfer's TRTLLM-Gen fused
MoE ops, whose launcher module FlashInfer builds at the first call
(``flashinfer.fused_moe.core.get_trtllm_moe_sm100_module``): an nvcc build of
about 15 minutes per rank on a cold cache, in warmup's first MoE forward.
JIT compile ahead names it here once the model is constructed, so a
FlashInfer helper builds it while the weights load
(``jit_prefetch_flashinfer_helper``); warmup joins it before its first forward.
"""

from __future__ import annotations

from typing import List


def flashinfer_modules(model) -> List[str]:
    """``["fused_moe_trtllm_sm100"]`` (``..._sm107`` on SM 10.7, where
    FlashInfer builds its Rubin variant) if a layer of ``model`` runs its MoE
    on FlashInfer's op backend, else none."""
    from ...._utils import get_sm_version
    from .moe_op_backend import FlashinferOpBackend

    for mod in model.modules():
        if isinstance(getattr(mod, "op_backend", None), FlashinferOpBackend):
            sm = get_sm_version()
            return ["fused_moe_trtllm_sm107" if sm == 107 else "fused_moe_trtllm_sm100"]
    return []
