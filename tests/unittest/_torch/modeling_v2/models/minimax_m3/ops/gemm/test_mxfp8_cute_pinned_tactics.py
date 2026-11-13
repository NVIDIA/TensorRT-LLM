# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ops/gemm/mxfp8_cute_pinned_tactics: the family's pinned FlashInfer CuTe DSL MXFP8 GEMM tactics. Every entry is
non-split-K, each pinned call equals FlashInfer's default (heuristic) CuTe DSL path bit for bit, the table serves only
calls made inside ``engaged_for`` above 32 rows, and the kill switch restores the default path."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as pin,
)
from tensorrt_llm._utils import get_sm_version

sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)


def test_table_is_non_split_k_and_pinned():
    assert pin.TACTICS, "empty table"
    for (n, k, m), (tile, cluster, swap, prefetch, split_k) in pin.TACTICS.items():
        assert split_k == 1, ((n, k, m), "a split-K tactic changes the K order (not bitwise)")
        assert m >= 64 and m & (m - 1) == 0 and m <= pin.MAX_BUCKET, (n, k, m)
        assert (
            len(tile) == 2
            and len(cluster) == 2
            and isinstance(swap, bool)
            and isinstance(prefetch, bool)
        )


DEP_ENTRIES = {
    (9856, 6144, 64): ((128, 64), (1, 4), False, True, 1),
    (9856, 6144, 128): ((128, 64), (1, 4), False, True, 1),
    (6144, 8192, 64): ((256, 64), (2, 2), True, True, 1),
    (6144, 8192, 128): ((256, 64), (2, 2), True, True, 1),
}


def test_the_dep_projection_entries():
    """The DEP qkv_proj [9856, 6144] and o_proj [6144, 8192] tactics at buckets 64 / 128, and no other bucket."""
    assert {key: pin.TACTICS[key] for key in DEP_ENTRIES} == DEP_ENTRIES
    assert {key for key in pin.TACTICS if key[:2] in {(9856, 6144), (6144, 8192)}} == set(
        DEP_ENTRIES
    )


DEP_SHARED_ENTRIES = {
    (6144, 6144, 64): ((256, 64), (2, 1), True, False, 1),
    (6144, 6144, 128): ((256, 64), (2, 1), True, False, 1),
    (6144, 3072, 64): ((256, 64), (2, 1), True, False, 1),
    (6144, 3072, 128): ((256, 64), (2, 1), True, False, 1),
}


def test_the_dep_shared_expert_entries():
    """The DEP shared expert's gate_up_proj [6144, 6144] and down_proj [6144, 3072] tactics at buckets 64 / 128;
    down_proj has no other bucket."""
    assert {key: pin.TACTICS[key] for key in DEP_SHARED_ENTRIES} == DEP_SHARED_ENTRIES
    assert {key for key in pin.TACTICS if key[:2] == (6144, 3072)} == {
        (6144, 3072, 64),
        (6144, 3072, 128),
    }


def test_a_caller_engages_the_table_for_its_own_calls_only(monkeypatch):
    monkeypatch.setattr(pin, "get_sm_version", lambda: 107)
    assert pin.tactic_for(88, 9856, 6144) is None
    with pin.engaged_for("caller"):
        assert pin.tactic_for(88, 9856, 6144) == DEP_ENTRIES[(9856, 6144, 128)]
        assert pin.tactic_for(32, 9856, 6144) is None
        assert pin.tactic_for(129, 9856, 6144) is None
    assert pin.tactic_for(88, 9856, 6144) is None
    with pytest.raises(RuntimeError), pin.engaged_for("caller"):
        raise RuntimeError
    assert pin.tactic_for(88, 9856, 6144) is None
    monkeypatch.setattr(pin, "ENABLED", False)
    with pin.engaged_for("caller"):
        assert pin.tactic_for(88, 9856, 6144) is None


def test_bucket_and_row_gate():
    assert (
        pin._bucket(33) == 64
        and pin._bucket(64) == 64
        and pin._bucket(65) == 128
        and pin._bucket(184) == 256
    )
    assert pin.tactic_for(32, 4992, 6144) is None  # 1..32 rows keep FlashInfer's own low-M path


def _operands(m, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, k, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    q, s = torch.ops.trtllm.mxfp8_quantize(x, True)
    wq, ws = torch.ops.trtllm.mxfp8_quantize(w, True)
    return q, s, wq, ws


def _default(q, s, wq, ws):
    from flashinfer import autotune, mm_mxfp8

    with autotune(tune_mode=False, skip_ops="mxfp8_gemm"):
        return mm_mxfp8(
            q, wq.t(), s, ws, out_dtype=torch.bfloat16, use_8x4_sf_layout=False, backend="cute-dsl"
        )


@sm107
@pytest.mark.parametrize("key", sorted(pin.TACTICS), ids=lambda k: f"N{k[0]}_K{k[1]}_M{k[2]}")
def test_pinned_equals_default_bitwise(key):
    n, k, bucket = key
    for m in sorted({bucket // 2 + 1 if bucket > 64 else 33, bucket}):  # both ends of the bucket
        q, s, wq, ws = _operands(m, n, k, seed=hash(key) % 1000 + m)
        with pin.engaged_for("test"):
            assert pin.tactic_for(m, n, k) == pin.TACTICS[key]
            out = pin.run(q, s, wq, ws, torch.bfloat16)
        assert out is not None, "the pinned path declined a table entry"
        assert torch.equal(out, _default(q, s, wq, ws)), (key, m)


@sm107
def test_the_kill_switch_restores_the_default_path(monkeypatch):
    monkeypatch.setattr(pin, "ENABLED", False)
    q, s, wq, ws = _operands(80, 4992, 6144, seed=80)
    with pin.engaged_for("test"):
        assert pin.tactic_for(80, 4992, 6144) is None
        assert pin.run(q, s, wq, ws, torch.bfloat16) is None


@sm107
@pytest.mark.parametrize("key", sorted(DEP_ENTRIES), ids=lambda k: f"N{k[0]}_K{k[1]}_M{k[2]}")
def test_a_caller_runs_the_dep_entries_bitwise_in_a_cuda_graph(key):
    """Inside engaged_for() a dep table entry runs captured: the replay on new operands writes the eager pinned run's
    bits on them. (Pinned against the default path is test_pinned_equals_default_bitwise.)"""
    n, k, bucket = key
    for m in sorted({bucket // 2 + 1 if bucket > 64 else 33, bucket}):
        q, s, wq, ws = _operands(m, n, k, seed=m)
        with pin.engaged_for("test"):
            assert pin.run(q, s, wq, ws, torch.bfloat16) is not None, (
                "the pinned path declined a table entry"
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = pin.run(q, s, wq, ws, torch.bfloat16)
        fresh = _operands(m, n, k, seed=m + 1)
        for static, value in zip((q, s, wq, ws), fresh):
            static.copy_(value)
        graph.replay()
        with pin.engaged_for("test"):
            eager = pin.run(*fresh, torch.bfloat16)
        torch.cuda.synchronize()
        assert torch.equal(captured, eager), (key, m)
