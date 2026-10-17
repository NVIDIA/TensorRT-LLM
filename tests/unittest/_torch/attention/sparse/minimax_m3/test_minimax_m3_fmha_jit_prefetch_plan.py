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
"""CPU checks of the MiniMax-M3 MSA fmha_sm100 JIT-prefetch planner.

The planner's variant per longest context row, its build order, and the
kernels it lists for MiniMax-M3's standard context server (TP2, NVFP4 KV,
FP8 indexer). ``test_minimax_m3_fmha_jit_prefetch.py`` checks the variants
against the real ``fmha_sm100`` selection on a GPU.
"""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import jit_prefetch_fmha as jpf

pytestmark = pytest.mark.cpu_only

FP8, BF16 = jpf._DTYPE_CODES["float8_e4m3fn"], jpf._DTYPE_CODES["bfloat16"]


def _name(key):
    """fmha_sm100's variant directory name for a get_fmha_variant key."""
    code, tile, single_wg, mode, page, split, pack = key
    return "_".join(
        str(i)
        for i in (
            {BF16: 0, FP8: 1}[code],
            {128: 0, 256: 1}[tile],
            0 if single_wg else 1,
            mode,
            {-1: 0, 128: 1}[page],
            int(split),
            (1, 2, 4, 6, 8, 16).index(pack),
        )
    )


@pytest.mark.parametrize(
    "qo_len, heads, kv_heads, mode, expected",
    [
        # Proxy max-score pass at TP2 (2 index heads over one MQA head).
        (1, 2, 1, jpf._ONLY_SCORE, "1_0_0_2_1_0_1"),
        (32, 2, 1, jpf._ONLY_SCORE, "1_0_0_2_1_0_1"),
        (33, 2, 1, jpf._ONLY_SCORE, "1_0_0_2_1_0_0"),
        (65, 2, 1, jpf._ONLY_SCORE, "1_0_1_2_1_0_0"),
        (129, 2, 1, jpf._ONLY_SCORE, "1_1_1_2_1_0_0"),
        # Dense layers at TP2 (32 q heads over 2 KV heads): pack factor 16 at 1-8
        # rows, 8 to 16, 4 to 32; then the 128- and 256-row tiles.
        (1, 32, 2, jpf._OFF, "1_0_0_3_1_0_5"),
        (8, 32, 2, jpf._OFF, "1_0_1_3_1_0_5"),
        (9, 32, 2, jpf._OFF, "1_0_1_3_1_0_4"),
        (17, 32, 2, jpf._OFF, "1_0_1_3_1_0_2"),
        (33, 32, 2, jpf._OFF, "1_0_0_3_1_0_0"),
        (65, 32, 2, jpf._OFF, "1_0_1_3_1_0_0"),
        (16384, 32, 2, jpf._OFF, "1_1_1_3_1_0_0"),
    ],
)
def test_row_variant_matches_fmha_sm100_names(qo_len, heads, kv_heads, mode, expected):
    assert _name(jpf.row_variant("float8_e4m3fn", heads, kv_heads, mode, 128, qo_len)) == expected


def test_block_sparse_long_rows_run_no_variant():
    """Block-sparse rows past 32 tokens are planned for the CuTe DSL sparse
    prefill, which builds no fmha_sm100 variant."""
    assert jpf.row_variant("float8_e4m3fn", 32, 2, jpf._SPARSE, 128, 32) is not None
    assert jpf.row_variant("float8_e4m3fn", 32, 2, jpf._SPARSE, 128, 33) is None


def test_rows_past_129_select_nothing_new():
    for heads, kv_heads, mode in ((2, 1, jpf._ONLY_SCORE), (32, 2, jpf._OFF), (16, 1, jpf._OFF)):
        ref = jpf.row_variant("float8_e4m3fn", heads, kv_heads, mode, 128, 129)
        for qo_len in (130, 255, 256, 257, 4096, 16384):
            assert jpf.row_variant("float8_e4m3fn", heads, kv_heads, mode, 128, qo_len) == ref


def _provider(monkeypatch, sites, nvfp4):
    """A provider over fake MSA layers: (q heads, KV heads, index heads,
    sparse layer) per layer."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import kernels, msa_backend
    from tensorrt_llm.bindings import DataType

    class FakeMsa(msa_backend.MiniMaxM3MsaSparseAttention):
        def __init__(self, q, kv, idx):  # noqa: D107 - no backend state needed
            self.m3_config = SimpleNamespace(
                num_q_heads=q, num_kv_heads=kv, num_index_heads=idx, block_size=128
            )

    class Layer:
        def __init__(self, q, kv, idx, sparse):
            self.attn = FakeMsa(q, kv, idx)
            self.is_sparse_attention_layer = sparse

    layers = [Layer(*site) for site in sites]
    model = SimpleNamespace(modules=lambda: iter(layers))
    kv_cache_manager = SimpleNamespace(
        tokens_per_block=128,
        dtype=DataType.NVFP4 if nvfp4 else DataType.FP8,
        indexer_kv_dtype="fp8",
    )
    monkeypatch.setattr("tensorrt_llm._utils.is_sm_100f", lambda *a: True)
    monkeypatch.setattr(kernels.msa_utils, "msa_package_available", lambda: True)
    return jpf.MiniMaxM3FmhaProvider(model, kv_cache_manager)


def test_provider_lists_the_standard_ctx_server_kernels(monkeypatch):
    """TP2 MiniMax-M3, NVFP4 KV (FP8 dense layers), FP8 indexer: the planner,
    the k2q CSR builder, and the 11 variants a cold server builds (the 8 of
    its warmup and the 3 of its first serving steps)."""
    provider = _provider(monkeypatch, [(32, 2, 2, False)] * 3 + [(32, 2, 2, True)] * 57, True)
    ops = [json.loads(s) for s in provider.specs]
    assert ops[0] == {"op": "plan"} and ops[1] == {"op": "k2q_csr"}
    names = [_name(op["args"]) for op in ops[2:]]
    assert len(names) == len(set(names))
    assert set(names) == {
        "1_0_0_3_1_0_5", "1_0_0_2_1_0_1", "1_1_1_3_1_0_0", "1_1_1_2_1_0_0",
        "1_0_0_3_1_0_0", "1_0_0_2_1_0_0", "1_0_1_3_1_0_2", "1_0_1_3_1_0_4",
        "1_0_1_3_1_0_0", "1_0_1_2_1_0_0", "1_0_1_3_1_0_5",
    }  # fmt: skip
    # The single-row and longest-row variants of every site come first, as
    # warmup needs them first (dense layers lead: the model's first layers).
    assert set(names[:4]) == {"1_0_0_2_1_0_1", "1_0_0_3_1_0_5", "1_1_1_2_1_0_0", "1_1_1_3_1_0_0"}


def test_provider_fp8_cache_plans_block_sparse_short_rows(monkeypatch):
    provider = _provider(monkeypatch, [(32, 2, 2, False), (32, 2, 2, True)], False)
    ops = [json.loads(s) for s in provider.specs]
    assert {"op": "k2q_csr"} not in ops
    assert any(op.get("args", [None] * 4)[3] == jpf._SPARSE for op in ops)


def test_provider_without_msa_layers_lists_nothing(monkeypatch):
    assert not _provider(monkeypatch, [], True)
