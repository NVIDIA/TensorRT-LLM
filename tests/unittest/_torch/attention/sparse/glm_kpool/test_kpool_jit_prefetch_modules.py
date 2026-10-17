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
"""The FlashInfer module JIT prefetch builds while GLM-5.3-Flash's weights load.

``glm_kpool.jit_prefetch.flashinfer_modules`` names ``fmha_gen`` exactly when a
layer's small generation batches run FlashInfer's TRTLLM-GEN MLA decode: a
BF16 latent cache, 16 TP-local query heads of width 512, SM100 / SM103, the
same test ``GlmKpoolSparseAttention.forward`` makes. ``start_model_builds``
starts a helper on it only with prefetch on.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tensorrt_llm._torch import jit_prefetch as jp
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import GlmKpoolSparseAttention
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import backend as kpool_backend
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.jit_prefetch import flashinfer_modules
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


class _Layer(nn.Module):
    def __init__(self, num_heads: int, head_dim: int = 512):
        super().__init__()
        # The attention backend is a plain attribute, as in Glm5NextSparseAttention.
        self.attn_backend = object.__new__(GlmKpoolSparseAttention)
        self.attn_backend.num_heads = num_heads
        self.attn_backend.head_dim = head_dim


class _Model(nn.Module):
    def __init__(self, num_heads=16, head_dim=512, kv_quant=None, dtype=torch.bfloat16):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2), _Layer(num_heads, head_dim)])
        self.model_config = SimpleNamespace(
            quant_config=QuantConfig(kv_cache_quant_algo=kv_quant), torch_dtype=dtype
        )


@pytest.fixture
def sm():
    """Patch the SM version the backend's native-decode test reads."""

    def set_sm(version):
        patcher = patch.object(kpool_backend, "get_sm_version", return_value=version)
        patcher.start()
        return patcher

    patchers = []
    yield lambda version: patchers.append(set_sm(version))
    for p in patchers:
        p.stop()


@pytest.mark.parametrize("version", [100, 103])
def test_native_decode_geometry_selects_fmha_gen(sm, version):
    sm(version)
    assert flashinfer_modules(_Model()) == ["fmha_gen"]


@pytest.mark.parametrize(
    "model,version",
    [
        (_Model(num_heads=32), 100),  # TP-local heads the kernel does not take
        (_Model(num_heads=64), 103),  # attention DP: every head on one rank
        (_Model(head_dim=256), 100),
        (_Model(kv_quant=QuantAlgo.FP8), 100),  # FP8 latent cache: FlashMLA path
        (_Model(kv_quant=QuantAlgo.NVFP4), 100),
        (_Model(dtype=torch.float16), 100),
        (_Model(), 90),
        (_Model(), 120),
    ],
    ids=["tp1-heads", "adp-heads", "head-dim", "fp8-kv", "nvfp4-kv", "fp16", "sm90", "sm120"],
)
def test_other_geometries_select_nothing(sm, model, version):
    sm(version)
    assert flashinfer_modules(model) == []


def test_model_without_kpool_layers_selects_nothing(sm):
    sm(100)
    model = nn.Sequential(nn.Linear(2, 2))
    model.model_config = SimpleNamespace(quant_config=QuantConfig(), torch_dtype=torch.bfloat16)
    assert flashinfer_modules(model) == []


@pytest.mark.parametrize("prefetch", [False, True])
def test_start_model_builds_starts_a_helper_only_with_prefetch(sm, monkeypatch, prefetch):
    sm(100)
    monkeypatch.setenv("TLLM_JIT_PREFETCH", "1" if prefetch else "0")
    with patch.object(jp, "start_flashinfer_build", return_value=True) as start:
        started = jp.start_model_builds(_Model())
    assert started is prefetch
    if prefetch:
        start.assert_called_once_with(("fmha_gen",))
    else:
        start.assert_not_called()
