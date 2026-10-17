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
"""rope_kv_append composed with its neighbour packages (needs a tree that contains them).

Producers of rope_kv_append's inputs: the native projection + absorption BMM, absorbed_query (target
layers), absorbed_query + shared_indexer_absorbed_query (shared-indexer target layers) and
draft_absorbed_query (the MTP layer); consumers of its outputs: the native trtllm-gen sparse MLA or
sparse_mla_decode, with the native or indexer_topk indexer logits in between. Every case runs
``forward_sparse_attn`` (projections + attention) from hidden states with the neighbour set alone,
twice, and with rope_kv_append added, from identical state. Compared byte for byte
against the rope_kv_append-off runs: every KV and indexer-K pool byte, each layer's FP8
query buffer, ``cu_q``/``cu_kv`` scheduler offsets, scheduler counter, BMM1/BMM2
scales and global KV indices (a buffer that differs between the two rope_kv_append-off
runs is reported, not compared). rope_kv_append and every selected neighbour must engage,
and rope_kv_append must not change the neighbours' engagement.
"""

import os
import sys

import pytest
import torch
from rope_append_stack import MTP_LAYER, PATTERN, Stack, load_dev_library

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plan_testlib  # noqa: E402


def _have(pkg: str, name: str) -> bool:
    try:
        dispatch.get_impl(pkg, name)
    except (ImportError, KeyError):
        return False
    return True


NEIGHBOURS_PRESENT = all(
    _have(p, n)
    for p, n in (
        ("absorbed_query", "dsa_proj"),
        ("absorbed_query", "shared_dsa_proj"),
        ("absorbed_query", "mtp_dsa_proj"),
        ("sparse_mla_decode", "fmha_lib"),
        ("indexer_topk", "indexer_decode"),
    )
)


pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not NEIGHBOURS_PRESENT,
        reason="absorbed_query/draft_absorbed_query/shared_indexer_absorbed_query/"
        "sparse_mla_decode/indexer_topk are not in this tree",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="rope_kv_append targets Blackwell decode",
    ),
]

CTX = 32768
TARGET_CASES = {
    "native": "",
    "absorbed_query": "absorbed_query",
    "absorbed_query_shared_query": "absorbed_query,shared_indexer_absorbed_query",
    "indexer_topk": "indexer_topk",
    "sparse_mla_decode": "sparse_mla_decode",
    "all": "absorbed_query,shared_indexer_absorbed_query,draft_absorbed_query,indexer_topk,sparse_mla_decode",
}
DRAFT_CASES = {
    "native": "",
    "draft_absorbed_query": "draft_absorbed_query",
    "sparse_mla_decode": "sparse_mla_decode",
    "all": "absorbed_query,shared_indexer_absorbed_query,draft_absorbed_query,indexer_topk,sparse_mla_decode",
}
NEIGHBOUR_KEYS = (
    "aq_packet",
    "shared_query_packet",
    "draft_query_packet",
    "indexer_topk_decode",
    "sparse_mla_fmha",
)


@pytest.fixture(scope="module")
def stack():
    load_dev_library()
    # The adopted layers declare the sparse-MLA decode library only under a plan that runs
    # sparse_mla_decode: the stack is built under the plan of ``plan_testlib.FAST_PATHS_ENV``.
    plan_testlib.select_from_env(dispatch)
    s = Stack(CTX, real_proj=True)
    yield s
    s.shutdown()


def _check_neighbours(features: str, base_counts, rope_append_counts, layers: int, draft: bool):
    feats = {f for f in features.split(",") if f}
    for key in NEIGHBOUR_KEYS:
        assert base_counts.get(key, 0) == rope_append_counts.get(key, 0), (
            key,
            base_counts,
            rope_append_counts,
        )
    if "absorbed_query" in feats and not draft:
        shared = sum(1 for p in PATTERN[:layers] if p == "S")
        assert (
            rope_append_counts.get("aq_packet", 0)
            + rope_append_counts.get("shared_query_packet", 0)
            == layers
        )
        if "shared_indexer_absorbed_query" in feats:
            assert rope_append_counts.get("shared_query_packet", 0) == shared
    if "draft_absorbed_query" in feats and draft:
        assert rope_append_counts.get("draft_query_packet", 0) == 1
    if "sparse_mla_decode" in feats:
        assert rope_append_counts.get("sparse_mla_fmha", 0) == layers


def _sparse_mla_declared(stack) -> bool:
    """Whether the stack's layers put the sparse-MLA decode library ahead of the registry's.

    Once declared, sparse_mla_decode serves every eligible layer, so sparse_mla_decode cases run
    in a process whose stack was built with ``sparse_mla_decode`` in
    ``plan_testlib.FAST_PATHS_ENV``, the other cases in one without.
    """
    return any(getattr(mla.mqa, "prepended_fmha_lib_classes", ()) for mla in stack.layers)


def _skip_unless_sparse_mla_matches(stack, features: str):
    wants_sparse_mla = "sparse_mla_decode" in features.split(",")
    if wants_sparse_mla != _sparse_mla_declared(stack):
        pytest.skip(
            "sparse_mla_decode cases need the stack built with sparse_mla_decode running; others one without"
            if wants_sparse_mla
            else "the stack declares sparse_mla_decode, which would serve this case"
        )


@pytest.mark.parametrize("case", list(TARGET_CASES))
def test_target_layers_composed(stack, case):
    """Verify step (M = 6) through full + 3 shared target layers."""
    features = TARGET_CASES[case]
    _skip_unless_sparse_mla_matches(stack, features)
    md = stack.metadata(6, stack.ctx_len)
    stack.compare(
        md,
        lambda: stack.run_proj(md, range(len(PATTERN)), [(6, stack.ctx_len)], seed=51),
        base=features,
        capture=True,
    )
    base_counts, _, rope_append_counts = stack.engagement
    print(f"{case}: base {base_counts}\n{case}: rope_kv_append {rope_append_counts}")
    assert (
        rope_append_counts.get("rope_append_mla_rope_generation") == 4
        and rope_append_counts.get("rope_append_fused_scatter") == 1
    )
    assert (
        rope_append_counts.get("rope_append_remap_topk_to_global") == 4
        and rope_append_counts.get("rope_append_bridge_kernel") == 1
    )
    assert rope_append_counts.get("rope_append_check_indexer_k_scatter_consumed") == 1
    assert not any(k.startswith("native") for k in rope_append_counts), rope_append_counts
    _check_neighbours(features, base_counts, rope_append_counts, len(PATTERN), draft=False)
    report = stack.capture_report
    # quant_q may differ between runs without rope_kv_append; the rest may not.
    assert report["rope_exact"] >= 4 * 5
    assert report["idx_exact"] + report["idx_set_exact"] + report["idx_native_nondet"] == 4


@pytest.mark.parametrize("case", list(DRAFT_CASES))
def test_mtp_draft_steps_composed(stack, case):
    """MTP layer: draft step 0 (6 rows, last-accepted-row stash) and a reuse step (1 row)."""
    features = DRAFT_CASES[case]
    _skip_unless_sparse_mla_matches(stack, features)
    md0 = stack.metadata(6, stack.ctx_len)
    md0.set_in_mtp_draft_loop(True)
    md0.set_mtp_num_accepted(torch.tensor([4], dtype=torch.int32, device=stack.dev))
    md0.set_skip_topk(False)
    stack.compare(
        md0,
        lambda: stack.run_proj(md0, [MTP_LAYER], [(6, stack.ctx_len)], seed=52),
        base=features,
        capture=True,
    )
    base_counts, _, rope_append_counts = stack.engagement
    print(f"{case} step0: base {base_counts}\n{case} step0: rope_kv_append {rope_append_counts}")
    assert (
        rope_append_counts.get("rope_append_fused_scatter") == 1
        and rope_append_counts.get("rope_append_bridge_kernel") == 1
    )
    assert not any(k.startswith("native") for k in rope_append_counts), rope_append_counts
    _check_neighbours(features, base_counts, rope_append_counts, 1, draft=True)

    md1 = stack.metadata(1, stack.ctx_len + 5)
    md1.set_in_mtp_draft_loop(True)
    md1.set_skip_topk(True)
    md1.shared_topk_indices.copy_(md0.shared_topk_indices)
    stack.compare(
        md1,
        lambda: stack.run_proj(md1, [MTP_LAYER], [(1, stack.ctx_len + 5)], seed=53),
        base=features,
        capture=True,
    )
    base_counts, _, rope_append_counts = stack.engagement
    print(f"{case} step1: base {base_counts}\n{case} step1: rope_kv_append {rope_append_counts}")
    assert (
        rope_append_counts.get("rope_append_fused_scatter") == 1
        and rope_append_counts.get("rope_append_bridge_kernel") == 1
    )
    assert not any(k.startswith("native") for k in rope_append_counts), rope_append_counts
    _check_neighbours(features, base_counts, rope_append_counts, 1, draft=True)
