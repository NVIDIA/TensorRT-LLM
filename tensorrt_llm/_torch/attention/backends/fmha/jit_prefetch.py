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
"""JIT compile-ahead plan of the TRTLLM backend's FMHA libraries (see
``tensorrt_llm/_torch/jit_prefetch.py``)."""

from typing import List

# Attributes under which attention modules hold their attention backend.
_BACKEND_ATTRS = ("attn", "mha", "mqa", "attn_backend", "backend")


def flashinfer_modules(model) -> List[str]:
    """FlashInfer JIT modules ``model``'s attention layers build in their
    first forward: ``["fmha_gen"]`` if a non-MLA layer can run on FlashInfer's
    TRTLLM-GEN FMHA library (``FlashInferTrtllmGenFmha``), else none.

    ``fmha_gen`` holds the launchers of FlashInfer's TRTLLM-GEN context and
    decode attention kernels, an nvcc build of about a minute on a cold cache,
    which the first forward that selects the library pays. A layer's
    ``FmhaManager`` lists the libraries that can serve it when the layer is
    constructed, from static checks that later configuration does not
    change, and among the default libraries only TRTLLM-GEN serves a regular
    (non-MLA, non-sparse) layer's paged attention before the fallback. MLA
    layers are left out: their decode goes to the CuTe DSL MLA library
    first, and glm_kpool's own plan covers its TRTLLM-GEN MLA decode.
    """
    from .flashinfer_trtllm_gen import FlashInferTrtllmGenFmha

    for mod in model.modules():
        for name in _BACKEND_ATTRS:
            backend = mod.__dict__.get(name)
            if backend is None or getattr(backend, "is_mla_enable", False):
                continue
            libs = getattr(getattr(backend, "_fmha_manager", None), "fmha_libs", None) or ()
            if any(isinstance(lib, FlashInferTrtllmGenFmha) for lib in libs):
                return ["fmha_gen"]
    return []
