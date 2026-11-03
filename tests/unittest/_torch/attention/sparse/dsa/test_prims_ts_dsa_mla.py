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
"""DSA generation on the PrimTS D576 sparse MLA FMHA library (Kimi K3.1 nvfp4_ds_mla)."""

from tensorrt_llm._torch.attention.backends.fmha import registry
from tensorrt_llm._torch.attention.backends.fmha.prims_ts_dsa_mla import PrimsTSDsaMlaFmha

LIB_NAME = "prims_ts_dsa_mla"


def test_registry_entry_precedes_trtllm_gen() -> None:
    """The library is in the default set ahead of the TRTLLM-Gen libraries."""
    assert registry.FMHA_LIBS[LIB_NAME] is PrimsTSDsaMlaFmha
    assert LIB_NAME in registry.DEFAULT_FMHA_LIBS
    names = tuple(registry.FMHA_LIBS)
    assert names.index(LIB_NAME) < names.index("flashinfer_trtllm_gen")
    assert names.index(LIB_NAME) < names.index("fallback")


def test_unavailable_without_nvfp4_ds_mla() -> None:
    """Without an nvfp4_ds_mla cache the library never reports itself available."""

    class _Attn:
        is_mla_enable = True
        skip_correction_threshold = 0.0

    assert not PrimsTSDsaMlaFmha.is_available(_Attn())
