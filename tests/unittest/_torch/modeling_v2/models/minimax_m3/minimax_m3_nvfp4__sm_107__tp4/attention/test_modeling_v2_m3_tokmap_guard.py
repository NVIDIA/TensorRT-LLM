# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (SM 10.7) for ops/m3_tokmap_guard: a one-request multi-token call of
thop.trtllm_gen_generation_preprocess (rows padded to the bucket, the request's generation length 1..bucket) with a
stale token map in the workspace gives, through the guard, the bytes the op gives with a freshly built map: the
appended FP8 K/V pages and the processed query's live rows, bit for bit, eager and from one captured graph replayed for
every length. Without the guard the stale map moves rows (the negative control), and a several-request call passes
through with its own map untouched.

The fresh-map reference forces the op's own build (a scaled RoPE type with no inv-freq cache, at rotary dim 0, so no
row is rotated either way).
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
    m3_tokmap_guard as guard,
)
from tensorrt_llm.bindings.internal import thop

assert torch.cuda.is_available(), "m3_tokmap_guard requires a CUDA device"
if torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the tp4 target is certified on SM 10.7 only", allow_module_level=True)

H, KVH, D, PAGE = 16, 1, 128, 128
BUCKET = 4
_PEER_EXTRA = 777  # the second request's extra length behind the first
FP8_KV_QUANT_MODE = 128
_DEV = torch.device("cuda")  # unindexed on purpose: install() and the lookup normalize it
_TAKEN = []


@pytest.fixture(scope="module", autouse=True)
def _installed():
    guard.install(_DEV, 48, _TAKEN.append)
    yield


class _Step:
    """One request (or ``nreq``) of BUCKET padded rows over a fresh FP8 paged cache, persistent across calls like a
    real step's buffers."""

    def __init__(self, kv_len: int, nreq: int = 1, seed: int = 0):
        g = torch.Generator(device=_DEV).manual_seed(seed + kv_len)
        self.nreq = nreq
        # Per-request inputs (page tables, generation lengths, position offsets) for at least two requests, the page
        # tables at the peer's longer length (_seq_with_peer): the stale map sends rows to request 1, which must have
        # real entries behind it, also on the unguarded calls.
        tables = max(nreq, 2)
        pages = math.ceil((kv_len + _PEER_EXTRA + BUCKET) / PAGE)
        self.nblocks = tables * pages + 2
        self.pool0 = (
            torch.randn(self.nblocks, 2, KVH, PAGE, D, device=_DEV, generator=g)
            .clamp_(-4, 4)
            .to(torch.float8_e4m3fn)
        )
        self.pool = self.pool0.clone()
        perm = torch.randperm(self.nblocks, device=_DEV, generator=g)[: tables * pages].view(
            tables, pages
        )
        self.offsets = torch.zeros(1, tables, 2, pages, dtype=torch.int32, device=_DEV)
        self.offsets[0, :, 0] = 2 * perm
        self.offsets[0, :, 1] = 2 * perm + 1
        self.pointers = torch.tensor([[self.pool.data_ptr(), 0]], dtype=torch.int64)
        self.mapping = torch.zeros(1, 2, dtype=torch.int32)
        self.qkv = torch.randn(
            nreq * BUCKET, (H + 2 * KVH) * D, dtype=torch.bfloat16, device=_DEV, generator=g
        )
        # Request lengths after this step's append; generation lengths (accepted rows) set per case.
        self.seq = torch.full((nreq,), kv_len, dtype=torch.int32, device=_DEV)
        self.gen = torch.full((tables,), BUCKET, dtype=torch.int32, device=_DEV)
        self.pos = (
            torch.arange(BUCKET, dtype=torch.int32, device=_DEV).repeat(tables, 1).contiguous()
        )
        self.one = torch.ones(1, dtype=torch.float32, device=_DEV)
        self.ws = torch.empty(64 << 20, dtype=torch.uint8, device=_DEV)
        self.kv_len = kv_len

    def call(self, op, fresh_map: bool = False):
        """One generation preprocess; returns a copy of the processed query."""
        out = op(
            self.qkv,
            self.ws,
            self.seq,
            self.gen,
            self.pos,
            self.offsets,
            self.pointers,
            self.mapping,
            self.one,
            self.one,
            None,
            None,
            None,
            None,
            0,
            0,
            H,
            KVH,
            D,
            PAGE,
            FP8_KV_QUANT_MODE,
            1 << 20,
            1 << 20,
            self.nreq * BUCKET,
            self.nreq,
            BUCKET,
            self.kv_len,
            0,
            10000.0,
            1 if fresh_map else 0,
            1.0,
            1 << 20,
            0,
            1.0,
            1.0,
            False,
            BUCKET,
            0,
            torch.cuda.get_device_properties(0).multi_processor_count,
            self.nblocks,
            2,
            True,
            False,
        )
        return out[0].clone()

    def poison_map(self) -> None:
        """A stale multi-request map in the op's one-request map slice: rows 1.. on request 1."""
        layout = thop.get_trtllm_gen_generation_workspace_layout(
            self.qkv.dtype, 1, self.nreq * BUCKET, H, D, 0, KVH, 0, False, False
        )
        off = int(layout["tokens_info_offset"])
        stale = torch.tensor(
            [[0, 0]] + [[1, t] for t in range(BUCKET - 1)], dtype=torch.int32, device=_DEV
        )
        self.ws[off : off + 8 * BUCKET].view(torch.int32).view(BUCKET, 2).copy_(stale)

    def reset(self) -> None:
        self.pool.copy_(self.pool0)


def _seq_with_peer(step: _Step) -> None:
    # A second request's length behind the first, so a row the stale map moves to request 1 reads a real length.
    step.seq = torch.tensor(
        [step.kv_len, step.kv_len + _PEER_EXTRA], dtype=torch.int32, device=_DEV
    )


@pytest.mark.parametrize("kv_len", [130, 8191, 34117])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_guard_equals_a_fresh_map(kv_len, n) -> None:
    step = _Step(kv_len)
    _seq_with_peer(step)
    step.gen.fill_(n)
    step.poison_map()
    step.reset()
    q_ref = step.call(guard._original, fresh_map=True)
    pool_ref = step.pool.clone()

    step.poison_map()
    step.reset()
    q = step.call(thop.trtllm_gen_generation_preprocess)
    # The processed query's n live rows (rows past the request's length are padding the op leaves unwritten).
    assert torch.equal(q[:n].view(torch.uint8), q_ref[:n].view(torch.uint8))
    assert torch.equal(step.pool.view(torch.uint8), pool_ref.view(torch.uint8))
    assert BUCKET in _TAKEN


def test_stale_map_moves_rows_without_the_guard() -> None:
    step = _Step(34117)
    _seq_with_peer(step)
    step.gen.fill_(BUCKET)
    step.poison_map()
    step.reset()
    step.call(guard._original, fresh_map=True)
    pool_ref = step.pool.clone()
    step.poison_map()
    step.reset()
    step.call(guard._original)
    assert not torch.equal(step.pool.view(torch.uint8), pool_ref.view(torch.uint8))


def test_one_captured_graph_serves_every_length() -> None:
    step = _Step(34000)
    _seq_with_peer(step)
    refs = {}
    for n in range(1, BUCKET + 1):
        step.gen.fill_(n)
        step.poison_map()
        step.reset()
        q = step.call(guard._original, fresh_map=True)
        refs[n] = (q, step.pool.clone())
    step.gen.fill_(BUCKET)
    step.call(thop.trtllm_gen_generation_preprocess)  # eager first, as the warmup runs it
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = thop.trtllm_gen_generation_preprocess
        q_static = step.call(out)
    for n in range(1, BUCKET + 1):
        step.gen.fill_(n)
        step.poison_map()
        step.reset()
        g.replay()
        torch.cuda.synchronize()
        q_ref, pool_ref = refs[n]
        assert torch.equal(q_static[:n].view(torch.uint8), q_ref[:n].view(torch.uint8)), n
        assert torch.equal(step.pool.view(torch.uint8), pool_ref.view(torch.uint8)), n


def test_several_requests_pass_through() -> None:
    step = _Step(8191, nreq=2)
    layout = thop.get_trtllm_gen_generation_workspace_layout(
        step.qkv.dtype, 2, 2 * BUCKET, H, D, 0, KVH, 0, False, False
    )
    step.reset()
    step.call(thop.trtllm_gen_generation_preprocess)
    pool_guarded = step.pool.clone()
    step.reset()
    step.call(guard._original)
    assert torch.equal(step.pool.view(torch.uint8), pool_guarded.view(torch.uint8))
    assert int(layout["tokens_info_size"]) >= 8 * 2 * BUCKET
