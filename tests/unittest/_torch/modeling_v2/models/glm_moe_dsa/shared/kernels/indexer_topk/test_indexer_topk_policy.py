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
"""Eligibility table of the indexer_topk indexer decode package (no GPU needed).

``policy.py`` is loaded by file path so this test needs neither torch nor the
compiled extensions.
"""

import importlib.util
from pathlib import Path

import pytest

_POLICY = (
    Path(__file__).resolve().parents[9]
    / "tensorrt_llm"
    / "_torch"
    / "_experimental"
    / "modeling_v2"
    / "models"
    / "glm_moe_dsa"
    / "shared"
    / "kernels"
    / "indexer_topk"
    / "policy.py"
)


def _load_policy():
    spec = importlib.util.spec_from_file_location("indexer_topk_policy", _POLICY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Served GEN: max_seq_len 2**20 with MTP K = 5 and the overlap
# scheduler gives an indexer max_seq_len of 2**20 + 14, and KV cache manager V2
# sizes the block table at round_up(ceil((2**20 + 14 + 10) / 64), 4) columns.
SERVED_MAX_SEQ_LEN = (1 << 20) + 14
V2_BLOCK_TABLE_COLS = 16388
# KV cache manager V1 at the same max_seq_len: ceil((2**20 + 14) / 64) columns.
V1_BLOCK_TABLE_COLS = 16385

# Served GEN: sm103, CuTe-DSL logits, FP8 indexer keys, self-sampling GVR,
# one request with 1 + 5 verify rows, 32 heads of dim 128, 64-token pages,
# TP8 without attention DP, the served indexer max_seq_len.
_SERVED_TP8 = dict(
    sm_version=103,
    cute_dsl_available=True,
    native_uses_cute_dsl_logits=True,
    use_fp4_indexer=False,
    use_gvr_emission=False,
    needs_gvr_prior=False,
    is_ragged_verify=False,
    expanded_mtp_buffers=False,
    in_mtp_draft_loop=False,
    attention_dp=False,
    topk_decode_impl="cute_dsl_gvr",
    topk_self_sampling=True,
    index_topk=2048,
    compress_ratio=1,
    num_heads=32,
    head_dim=128,
    query_dtype="torch.float8_e4m3fn",
    query_shape=(6, 32, 128),
    query_contiguous=True,
    weights_dtype="torch.float32",
    weights_shape=(6, 32),
    weights_contiguous=True,
    k_cache_dtype="torch.uint8",
    k_cache_shape=(16384, 64, 1, 132),
    block_table_cols=V2_BLOCK_TABLE_COLS,
    max_seq_len=SERVED_MAX_SEQ_LEN,
    num_contexts=0,
    token_offset=0,
    num_gen_tokens=6,
    num_generations=1,
    uniform_verify_width=True,
)


def _with_shape(heads=32, head_dim=128, page=64, next_n=6, batch=1, **extra):
    """Served decision inputs with a different indexer/verify shape."""
    rows = next_n * batch
    return {
        **_SERVED_TP8,
        "num_heads": heads,
        "head_dim": head_dim,
        "query_shape": (rows, heads, head_dim),
        "weights_shape": (rows, heads),
        "k_cache_shape": (16384, page, 1, head_dim + 4),
        "num_gen_tokens": rows,
        "num_generations": batch,
        **extra,
    }


def test_the_served_tp8_shape_is_eligible():
    decision = _load_policy().decide(**_SERVED_TP8)
    assert decision.eligible
    assert decision.reason == ""


@pytest.mark.parametrize(
    "override, reason_part",
    [
        ({"sm_version": 90}, "sm90"),
        # SM100 has the same kernels but has not been validated on a GPU.
        ({"sm_version": 100}, "sm100"),
        ({"sm_version": 120}, "sm120"),
        ({"cute_dsl_available": False}, "DSL"),
        ({"native_uses_cute_dsl_logits": False}, "CuTe-DSL"),
        ({"use_fp4_indexer": True}, "FP4"),
        ({"use_gvr_emission": True}, "emission"),
        ({"needs_gvr_prior": True}, "prior"),
        ({"is_ragged_verify": True}, "ragged"),
        ({"expanded_mtp_buffers": True}, "expanded"),
        ({"in_mtp_draft_loop": True}, "draft"),
        ({"attention_dp": True}, "data parallelism"),
        ({"topk_decode_impl": "cuda_radix"}, "top-k engine"),
        ({"topk_decode_impl": "cute_dsl_radix"}, "top-k engine"),
        ({"topk_self_sampling": False}, "top-k engine"),
        ({"index_topk": 512}, "index_topk"),
        ({"compress_ratio": 4}, "compress_ratio"),
        ({"query_dtype": "torch.bfloat16"}, "query dtype"),
        ({"query_contiguous": False}, "layout"),
        ({"query_shape": (6, 32, 64)}, "layout"),
        ({"weights_dtype": "torch.bfloat16"}, "weights dtype"),
        ({"weights_contiguous": False}, "layout"),
        ({"weights_shape": (6, 16)}, "layout"),
        ({"k_cache_dtype": "torch.float8_e4m3fn"}, "K cache layout"),
        ({"k_cache_shape": (16384, 64, 1, 68)}, "K cache layout"),
        ({"k_cache_shape": (16384, 64, 132)}, "K cache view"),
        ({"max_seq_len": SERVED_MAX_SEQ_LEN + 1}, "max_seq_len"),
        ({"max_seq_len": 1 << 21}, "max_seq_len"),
        ({"max_seq_len": 0}, "max_seq_len"),
        # V1 table at the served max_seq_len: the last 128-token tile of a
        # 2**20 + 1 .. 2**20 + 14 token request would read one entry past the row.
        ({"block_table_cols": V1_BLOCK_TABLE_COLS}, "block table width"),
        ({"block_table_cols": 0}, "block table width"),
        ({"num_contexts": 1}, "context requests"),
        ({"token_offset": 1}, "exceed"),
        ({"num_generations": 0}, "no generation"),
        (
            {
                "num_gen_tokens": 7,
                "num_generations": 2,
                "query_shape": (7, 32, 128),
                "weights_shape": (7, 32),
            },
            "non-uniform",
        ),
        ({"uniform_verify_width": False}, "non-uniform"),
    ],
)
def test_native_fallback_cases(override, reason_part):
    decision = _load_policy().decide(**{**_SERVED_TP8, **override})
    assert not decision.eligible
    assert reason_part in decision.reason


@pytest.mark.parametrize(
    "shape, reason_part",
    [
        # Constructor-invalid for the rows kernel (8 cached weights > 4 heads):
        # must fall back before any kernel is built.
        (dict(heads=4, next_n=4), "num_heads"),
        (dict(heads=16), "num_heads"),
        (dict(heads=64), "num_heads"),
        (dict(head_dim=64), "head_dim"),
        (dict(page=16), "tokens_per_block"),
        (dict(page=32), "tokens_per_block"),
        (dict(page=128), "tokens_per_block"),
        (dict(next_n=1), "next_n"),
        (dict(next_n=4), "next_n"),
        (dict(next_n=8), "next_n"),
        (dict(batch=5), "generation requests"),
    ],
    ids=lambda v: "-".join(f"{k}{w}" for k, w in v.items()) if isinstance(v, dict) else v,
)
def test_unverified_shapes_fall_back(shape, reason_part):
    decision = _load_policy().decide(**_with_shape(**shape))
    assert not decision.eligible
    assert reason_part in decision.reason


def test_envelope_is_the_served_bound():
    policy = _load_policy()
    assert policy.MAX_INDEXER_SEQ_LEN == SERVED_MAX_SEQ_LEN
    # net max_seq_len + (K) overlap lead + (K - 1) extra KV tokens + (K) tokens
    # per generation step beyond the golden token, for K = 5.
    assert policy.MAX_INDEXER_SEQ_LEN == (1 << 20) + 5 + 4 + 5


@pytest.mark.parametrize(
    "max_seq_len, block_table_cols",
    [
        (SERVED_MAX_SEQ_LEN, V2_BLOCK_TABLE_COLS),
        # V1 and V2 at the 2**20 net envelope without MTP slack.
        (1 << 20, 16384),
        (1 << 20, 16388),
        (131072, 2048),
    ],
)
def test_served_lengths_and_table_widths_are_eligible(max_seq_len, block_table_cols):
    decision = _load_policy().decide(
        **{**_SERVED_TP8, "max_seq_len": max_seq_len, "block_table_cols": block_table_cols}
    )
    assert decision.eligible, decision.reason


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_verified_batch_sizes(batch):
    assert _load_policy().decide(**_with_shape(batch=batch)).eligible


def test_eligible_set_is_the_verified_envelope():
    """No shape outside the GPU-verified one is eligible."""
    policy = _load_policy()
    eligible = set()
    for sm in (90, 100, 103, 120):
        for heads in (4, 8, 16, 32, 64):
            for head_dim in (64, 128, 256):
                for page in (16, 32, 64, 128):
                    for next_n in range(1, 10):
                        for batch in range(1, 9):
                            inputs = _with_shape(heads, head_dim, page, next_n, batch)
                            inputs["sm_version"] = sm
                            if policy.decide(**inputs).eligible:
                                eligible.add((sm, heads, head_dim, page, next_n, batch))
    assert eligible == {(103, 32, 128, 64, 6, b) for b in range(1, 5)}
