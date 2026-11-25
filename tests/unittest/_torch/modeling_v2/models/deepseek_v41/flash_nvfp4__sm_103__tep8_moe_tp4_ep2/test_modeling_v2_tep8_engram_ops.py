# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's own decode-path Engram entry `engram_ll_allgather` (peer-memory all-gather of the head / WKV
shards). The gather moves bits: exact. The fused gate and the look-back hash are the family's entries
(`ops/engram/engram_gate_fused`, `ops/engram/engram_lookback_hash`, with tep8's layer-major ids), tested there.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # _v41_util
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # _validating
from _v41_util import DEV, gen_for  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.ops.engram.engram_ll_allgather import (  # noqa: E402,E501
    engram_ll_allgather,
    ll_allgather_takes,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.ops.engram.engram_ll_allgather_pair import (  # noqa: E402,E501
    engram_ll_allgather_pair,
    ll_allgather_pair_takes,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)


def _one_rank_gather():
    """A one-rank group of the engine's gather: its own receive buffer is the only peer (same kernel and protocol)."""
    from tensorrt_llm._torch.distributed.ll_allgather import _DEFAULT_MAX_BYTES, LowLatencyAllGather

    buf = torch.zeros(
        LowLatencyAllGather.buffer_bytes(1, _DEFAULT_MAX_BYTES), dtype=torch.uint8, device=DEV
    )
    return LowLatencyAllGather(0, 1, [buf.data_ptr()], _DEFAULT_MAX_BYTES, keepalive=buf)


@pytest.fixture(scope="module")
def gather():
    return _one_rank_gather()


@pytest.mark.parametrize(
    "cell", engram_ll_allgather.CELLS, ids=[c.why[:48] for c in engram_ll_allgather.CELLS]
)
def test_ll_allgather_cells(cell, gather, certifying):  # noqa: F811
    certifying(engram_ll_allgather)
    s = cell.spec
    gen = gen_for(cell)
    for _ in range(3):  # the epoch advances every launch: repeated gathers must keep landing
        x = torch.randn(s["rows"], s["width"], generator=gen, device=DEV).bfloat16()
        got = engram_ll_allgather(gather, x)
        assert torch.equal(got, engram_ll_allgather.reference(gather, x))


def test_ll_allgather_cuda_graph(gather):
    x = torch.zeros(6, 3200, dtype=torch.bfloat16, device=DEV)
    engram_ll_allgather(gather, x)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = engram_ll_allgather(gather, x)
    for seed in range(3):
        x.copy_(
            torch.randn(
                6, 3200, generator=torch.Generator(device=DEV).manual_seed(seed), device=DEV
            )
        )
        g.replay()
        assert torch.equal(out, x)


def test_ll_allgather_rejections(gather):
    assert not ll_allgather_takes(
        gather, 64, 3200, torch.bfloat16
    )  # 400 KiB > 256 KiB: the target keeps NCCL
    assert not ll_allgather_takes(None, 6, 3200, torch.bfloat16)
    with pytest.raises(AssertionError, match="larger than the peer buffers"):
        engram_ll_allgather.is_valid(
            gather, torch.zeros(64, 3200, dtype=torch.bfloat16, device=DEV)
        )
    with pytest.raises(AssertionError, match="contiguous"):
        engram_ll_allgather.is_valid(
            gather, torch.zeros(3200, 6, dtype=torch.bfloat16, device=DEV).t()
        )


@pytest.mark.parametrize(
    "cell", engram_ll_allgather_pair.CELLS, ids=[c.why[:48] for c in engram_ll_allgather_pair.CELLS]
)
def test_ll_allgather_pair_cells(cell, gather, certifying):  # noqa: F811
    certifying(engram_ll_allgather_pair)
    s = cell.spec
    cap = s.get("max_programs")
    gen = gen_for(cell)
    assert ll_allgather_pair_takes(gather, s["rows"], s["width"], torch.bfloat16)
    for _ in range(
        3
    ):  # the epoch advances every launch (single and pair share it): repeated gathers keep landing
        x0, x1 = (
            torch.randn(s["rows"], s["width"], generator=gen, device=DEV).bfloat16()
            for _ in range(2)
        )
        engram_ll_allgather_pair.is_valid(gather, x0, x1, cap)
        got = engram_ll_allgather_pair(gather, x0, x1, cap)
        want = engram_ll_allgather_pair.reference(gather, x0, x1, cap)
        assert all(torch.equal(g, w) for g, w in zip(got, want))
        single = engram_ll_allgather(
            gather, x0
        )  # interleaved with the single gather on the same epoch
        assert torch.equal(single, want[0])
    if cap is not None:  # the launch takes the cap (fewer programs, each moving more blocks)
        words = 2 * (s["rows"] * s["width"] * 2 // 4)
        assert gather.pair_programs(x0, x1, max_programs=cap) == min(cap, -(-words // 128))


def test_ll_allgather_pair_refuses_beyond_one_slot(gather):
    """24 rows of both WKV shards exceed one shard slot: the target gathers them one by one (as the built-in)."""
    assert not ll_allgather_pair_takes(gather, 24, 3200, torch.bfloat16)
    assert ll_allgather_takes(gather, 24, 3200, torch.bfloat16)
