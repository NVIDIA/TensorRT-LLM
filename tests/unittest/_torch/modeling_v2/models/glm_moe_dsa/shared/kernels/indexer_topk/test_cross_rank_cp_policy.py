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
"""Eligibility of the cross_rank_topk context-partitioned indexer region (no GPU needed).

``cross_rank_policy.py`` and ``policy.py`` are loaded from the indexer_topk package directory
without executing the package ``__init__`` (which imports torch).
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

_INDEXER_TOPK_DIR = (
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
)


def _load_cp_policy():
    name = "_cross_rank_policy_pkg"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(_INDEXER_TOPK_DIR)]
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.cross_rank_policy")


cp_policy = _load_cp_policy()

SERVED_MAX_SEQ_LEN = (1 << 20) + 14
# Largest per-rank share of the served envelope at world 8, and the kernel limit on sm_103.
LOCAL_CAPACITY_DRAFT_LOOP = 131136
MAX_LOCAL_CAPACITY = 179712

_SERVED_TP8 = dict(
    world_size=8,
    op_available=True,
    local_capacity=LOCAL_CAPACITY_DRAFT_LOOP,
    max_local_capacity=MAX_LOCAL_CAPACITY,
    allow_draft_step=False,
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
    block_table_cols=16388,
    max_seq_len=SERVED_MAX_SEQ_LEN,
    num_contexts=0,
    token_offset=0,
    num_gen_tokens=6,
    num_generations=1,
    uniform_verify_width=True,
)


def test_tp8_decode_is_eligible():
    d = cp_policy.decide_cp(**_SERVED_TP8)
    assert d.eligible, d.reason


@pytest.mark.parametrize("world", [4, 8])
def test_supported_world_sizes(world):
    assert cp_policy.decide_cp(**dict(_SERVED_TP8, world_size=world)).eligible


FALLBACKS = {
    "two_requests": dict(
        num_generations=2, num_gen_tokens=12, query_shape=(12, 32, 128), weights_shape=(12, 32)
    ),
    "world_2": dict(world_size=2),
    "world_16": dict(world_size=16),
    "op_missing": dict(op_available=False),
    "local_width_above_kernel_limit": dict(local_capacity=MAX_LOCAL_CAPACITY + 64),
    "mtp_draft_step": dict(in_mtp_draft_loop=True),
    "attention_dp": dict(attention_dp=True),
    "indexer_topk_envelope_page32": dict(k_cache_shape=(32768, 32, 1, 132)),
    "indexer_topk_envelope_next_n4": dict(
        num_gen_tokens=4, query_shape=(4, 32, 128), weights_shape=(4, 32)
    ),
    "indexer_topk_envelope_max_seq_len": dict(max_seq_len=SERVED_MAX_SEQ_LEN + 1),
    "indexer_topk_envelope_sm100": dict(sm_version=100),
    "indexer_topk_envelope_fp4": dict(use_fp4_indexer=True),
    "indexer_topk_envelope_ragged": dict(is_ragged_verify=True),
    "indexer_topk_envelope_mixed_batch": dict(num_contexts=1),
}


@pytest.mark.parametrize("name", sorted(FALLBACKS))
def test_native_fallback_cases(name):
    d = cp_policy.decide_cp(**dict(_SERVED_TP8, **FALLBACKS[name]))
    assert not d.eligible
    assert d.reason


def test_draft_step_only_when_allowed():
    inputs = dict(_SERVED_TP8, in_mtp_draft_loop=True)
    assert not cp_policy.decide_cp(**inputs).eligible
    assert cp_policy.decide_cp(**dict(inputs, allow_draft_step=True)).eligible


# The rank-local facts of a decision: a rank's build and device.
_LOCAL_KEYS = ("op_available", "max_local_capacity", "sm_version", "cute_dsl_available")


def _uniform(inputs):
    return {k: v for k, v in inputs.items() if k not in _LOCAL_KEYS}


def _local(inputs):
    return dict(
        op_available=inputs["op_available"],
        local_capacity=inputs["local_capacity"],
        max_local_capacity=inputs["max_local_capacity"],
        sm_version=inputs["sm_version"],
        cute_dsl_available=inputs["cute_dsl_available"],
    )


@pytest.mark.parametrize("name", sorted(FALLBACKS))
def test_uniform_and_local_parts_reproduce_decide_cp(name):
    """``decide_cp`` = rank-uniform decision AND no rank-local reason.

    The adapter decides a forward with ``decide_cp_uniform`` (identical on
    every rank) and hands ``rank_local_reason`` to the collective workspace
    creation, so a fact of one rank's build or device keeps every rank native
    instead of moving that rank alone out of the exchange.
    """
    inputs = dict(_SERVED_TP8, **FALLBACKS[name])
    full = cp_policy.decide_cp(**inputs)
    uniform = cp_policy.decide_cp_uniform(**_uniform(inputs))
    local = cp_policy.rank_local_reason(**_local(inputs))
    assert full.eligible == (uniform.eligible and not local), (full, uniform, local)


@pytest.mark.parametrize(
    "fact, value",
    [
        ("op_available", False),
        ("sm_version", 100),
        ("cute_dsl_available", False),
        ("max_local_capacity", LOCAL_CAPACITY_DRAFT_LOOP - 64),
    ],
)
def test_rank_local_facts_never_decide_a_forward(fact, value):
    """A rank-local fact never changes the per-forward decision.

    A rank whose own build or device differs makes the same decision as its
    peers; only its local reason differs.
    """
    inputs = dict(_SERVED_TP8, **{fact: value})
    assert cp_policy.decide_cp_uniform(**_uniform(inputs)).eligible
    assert cp_policy.rank_local_reason(**_local(inputs))
    assert not cp_policy.rank_local_reason(**_local(_SERVED_TP8))
