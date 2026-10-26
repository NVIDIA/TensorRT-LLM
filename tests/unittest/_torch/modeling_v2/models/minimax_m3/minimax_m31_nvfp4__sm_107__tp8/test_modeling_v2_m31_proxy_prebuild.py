# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PROXYPB on the sm_107 TP8 target: every fmha_sm100 variant the MSA indexer's proxy pass can pick (OnlyScore over
the FP8 paged index-K, page 128, no split-KV) is built once before warmup. One index head per rank packs nothing, so the
longest row alone picks the tile: <= 64, 65..128, > 128.

The variant builds are recorded rather than compiled (an nvcc build each); the core is a stand-in with the attributes
the prebuild reads.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    require_msa_module,
)

INDEX_HEADS_PER_RANK = 1
WANT = {"1_0_0_2_1_0_0", "1_0_1_2_1_0_0", "1_1_1_2_1_0_0"}


@pytest.fixture
def proxy_builds(monkeypatch):
    require_msa_module()  # its cutlass compatibility shim must precede the fmha_sm100 import, as in the target
    from fmha_sm100 import jit as fmha_jit

    built = []

    def record(*key):
        built.append(fmha_jit._variant_key_from_runtime(*key)[0])
        return SimpleNamespace(run=None)

    monkeypatch.setattr(fmha_jit, "get_fmha_variant", record)
    return built


def _core():
    attn = SimpleNamespace(
        indexer_kv_dtype="fp8", m3_config=SimpleNamespace(num_index_heads=INDEX_HEADS_PER_RANK)
    )
    return SimpleNamespace(sparse_layers=frozenset({3, 4, 5}), attn={3: attn, 4: attn, 5: attn})


def test_builds_every_proxy_variant_once(proxy_builds) -> None:
    names = modeling.MiniMaxM3Tp8ModelingV2._prebuild_msa_proxy(_core())
    assert set(names) == WANT
    assert sorted(proxy_builds) == sorted(WANT), (
        "each variant is built once, across every sparse layer"
    )


def test_prebuild_is_on_by_default() -> None:
    assert modeling.MSA_PROXY_PREBUILD is True
