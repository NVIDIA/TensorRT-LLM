# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IDX-PDL (MiniMax-M3.1 DEP, sm_107): the NVFP4 horizontal producer's early PDL trigger and the Q8KV4 decode
reader's pre-wait L2 prefetch leave every output bitwise unchanged.

* the producer with early_trigger on equals it off: q, index-Q, the main K/V data and scales, the NVFP4 index-K pages;
* the reader with 6 or 32 pre-wait prefetch pages equals the reader without them (native and hpr layouts), on requests
  whose new tokens straddle a page edge (a history page the producer writes in the same step), a request without
  history pages and more than 32 requests (the prefetch reads the scheduler in chunks of 32);
* the served chain producer -> reader -> select (DEP_VLLM_TOPK) captured as one CUDA graph with PDL, each arm against
  the arm with both off, replayed on new rows.
"""

import os

# The producer's PDL launch attribute follows TRTLLM_ENABLE_PDL (read once per process).
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa import (
    msa_indexer,  # noqa: E402
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels import (  # noqa: E402
    nvfp4_index_k,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import (
    family_ops,  # noqa: E402
)

PAGE = 128
DIM = 128
Q_HEADS, KV_HEADS = 64, 4  # a DEP rank: every head; index heads = KV heads
QUERY_LEN = 8
# Requests whose eight new tokens straddle a page edge: (ctx - 1) % 128 < 7.
STRADDLE = (1029, 8197)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _require_sm107():
    if torch.cuda.get_device_capability() != (10, 7) or not hasattr(
        family_ops(), "m31idxk_nvfp4_idxk_write"
    ):
        pytest.skip("the NVFP4 index-K pages and the Q8KV4 decode reader's SM107 path need sm_107")
    producer = torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert.default
    if "early_trigger" not in str(producer._schema):
        pytest.skip("this build's NVFP4 producer has no early PDL trigger")


def _rope_cache(max_positions, rotary_dim=64, base=5_000_000.0):
    positions = torch.arange(max_positions, dtype=torch.float32, device="cuda")
    inverse_frequency = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim)
    )
    frequency = torch.outer(positions, inverse_frequency)
    return torch.stack((frequency.cos(), frequency.sin()), dim=1).contiguous()


class _Step:
    """One decode step of a DEP rank: every request's eight new tokens, its pages, and the producer's inputs."""

    def __init__(self, ctx_lens, *, seed, cols=None):
        g = torch.Generator(device="cuda").manual_seed(seed)
        gc = torch.Generator().manual_seed(seed)
        self.ctx = list(ctx_lens)
        batch = len(self.ctx)
        npages = [(c + PAGE - 1) // PAGE for c in self.ctx]
        total = sum(npages)
        self.cols = cols or max(npages) + 3
        # A guard page on each side of the pool; requests own a random permutation of the rest.
        self.num_pages = total + 2
        perm = (torch.randperm(total, generator=gc) + 1).to(torch.int32)
        bt = torch.zeros(batch, self.cols, dtype=torch.int32)
        off = 0
        for b, n in enumerate(npages):
            bt[b, :n] = perm[off : off + n]
            off += n
        self.block_table = bt.cuda()
        self.seq_lens = torch.tensor(self.ctx, dtype=torch.int32, device="cuda")
        pos = torch.cat([torch.arange(c - QUERY_LEN, c) for c in self.ctx]).to(torch.int32)
        page_of = torch.stack(
            [bt[b, int(p) // PAGE] for b, c in enumerate(self.ctx) for p in range(c - QUERY_LEN, c)]
        )
        self.positions = pos.cuda()
        self.slots = (page_of * PAGE + pos % PAGE).to(torch.int32).cuda()
        rows = batch * QUERY_LEN
        heads = Q_HEADS + 3 * KV_HEADS + 1
        self.packed = torch.randn(rows, heads * DIM, generator=g, device="cuda").to(torch.bfloat16)
        self.weights = [
            (torch.randn(DIM, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
            for _ in range(4)
        ]
        self.rope = _rope_cache(max(self.ctx) + 1)
        self.kv_scales = torch.tensor([1.0, 1.75, 0.625], dtype=torch.float32, device="cuda")
        # Consistent fill: every slot of every page is the writer's NVFP4 of a random index-K row, as served pages are.
        self.pages0 = torch.zeros(
            self.num_pages, 1, PAGE, nvfp4_index_k.BYTES_PER_TOKEN, dtype=torch.uint8, device="cuda"
        )
        all_slots = torch.arange(self.num_pages * PAGE, dtype=torch.int32, device="cuda")
        for chunk in all_slots.split(1 << 16):
            k_rows = (torch.randn(chunk.numel(), DIM, generator=g, device="cuda") * 2.0).to(
                torch.bfloat16
            )
            nvfp4_index_k.write(k_rows, chunk, self.pages0)
        self.data0 = torch.randint(
            0, 256, (self.num_pages, 2, KV_HEADS, PAGE, DIM // 2), generator=g, device="cuda"
        ).to(torch.uint8)
        self.scale0 = torch.randint(
            0x30, 0x40, (self.num_pages, 2, KV_HEADS, PAGE, DIM // 16), generator=g, device="cuda"
        ).to(torch.uint8)
        self.workspace = nvfp4_index_k.decode_history_workspace(self.seq_lens, self.cols)
        # Candidate blocks of each query row: its own page and every page before it.
        self.n_valid_blocks = (self.positions // PAGE + 1).to(torch.int32)

    def fresh_caches(self):
        return self.pages0.clone(), self.data0.clone(), self.scale0.clone()

    def produce(self, caches, early_trigger):
        pages, data, scale = caches
        return torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(
            self.packed,
            data,
            scale,
            pages,
            self.slots,
            self.kv_scales,
            Q_HEADS,
            KV_HEADS,
            KV_HEADS,
            DIM,
            64,
            1e-5,
            *self.weights,
            self.rope,
            self.positions,
            early_trigger,
        )

    def read(self, index_q, pages, prefetch, query_len=QUERY_LEN):
        nvfp4_index_k.set_decode_prewait_prefetch(prefetch)
        try:
            return nvfp4_index_k.score_decode_rows(
                index_q,
                pages,
                self.block_table,
                self.seq_lens,
                query_len,
                KV_HEADS,
                self.workspace,
            )
        finally:
            nvfp4_index_k.set_decode_prewait_prefetch(0)

    def select(self, scores):
        return msa_indexer._dep_vllm_topk_select(scores, self.n_valid_blocks, 0, 1)

    def chain(self, caches, early_trigger, prefetch):
        q, index_q = self.produce(caches, early_trigger)
        scores = self.read(index_q, caches[0], prefetch)
        return q, index_q, scores, self.select(scores)


def _bytes(t):
    # A fresh dense copy: an empty or strided slice keeps its strides through contiguous().
    return t.clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)


def _assert_same(got, want):
    assert len(got) == len(want)
    for a, b in zip(got, want):
        if b is None:
            assert a is None
            continue
        # Raw bits (scores: only the history pages the reader writes, see _history_scores).
        assert torch.equal(_bytes(a), _bytes(b))


def _history_scores(step, scores, query_len=QUERY_LEN):
    """The scores the reader writes ([heads, pages, rows] view): per query row, the pages before its local page.
    A request's query_len rows are its last query_len of the eight new tokens."""
    out = []
    for b, c in enumerate(step.ctx):
        for j in range(query_len):
            local = (c - query_len + j) // PAGE
            out.append(scores[:, :local, b * query_len + j].contiguous())
    return out


@torch.inference_mode()
@pytest.mark.parametrize("ctx_lens", [STRADDLE + (9, 260), (133, 4100, 1029)])
def test_producer_early_trigger_is_bitwise(ctx_lens):
    _require_sm107()
    step = _Step(ctx_lens, seed=7)
    off_caches, on_caches = step.fresh_caches(), step.fresh_caches()
    off = step.produce(off_caches, False)
    on = step.produce(on_caches, True)
    torch.cuda.synchronize()
    _assert_same(on, off)
    _assert_same(on_caches, off_caches)
    # The new tokens landed (the caches differ from the initial fill exactly at the step's slots).
    assert not torch.equal(on_caches[0], step.pages0)


@torch.inference_mode()
@pytest.mark.parametrize("prefetch", [6, 32])
@pytest.mark.parametrize("query_len", [QUERY_LEN, 3])
@pytest.mark.parametrize("requests", ["straddle", "many"])
def test_reader_prewait_prefetch_is_bitwise(prefetch, query_len, requests):
    _require_sm107()
    if requests == "straddle":
        ctx = STRADDLE + (9, 130, 2000)
    else:
        g = torch.Generator().manual_seed(11)
        ctx = tuple(int(x) for x in torch.randint(9, 3000, (40,), generator=g)) + (9,)
    step = _Step(ctx, seed=3)
    caches = step.fresh_caches()
    _, index_q = step.produce(caches, False)
    if query_len != QUERY_LEN:
        index_q = index_q.view(len(ctx), QUERY_LEN, KV_HEADS, DIM)[:, -query_len:].reshape(
            -1, KV_HEADS, DIM
        )
    off = step.read(index_q, caches[0], 0, query_len).clone()
    on = step.read(index_q, caches[0], prefetch, query_len).clone()
    torch.cuda.synchronize()
    _assert_same(_history_scores(step, on, query_len), _history_scores(step, off, query_len))


@torch.inference_mode()
@pytest.mark.parametrize("early_trigger,prefetch", [(True, 0), (True, 6), (False, 6), (True, 32)])
def test_chain_cuda_graph_replay_is_bitwise(early_trigger, prefetch):
    """producer -> reader -> select captured with PDL; replays on new rows equal the off / off chain run eagerly."""
    _require_sm107()
    if msa_indexer._dep_vllm_topk_op(torch.cuda.current_device()) is None:
        pytest.skip("minimax_m3_sm107_dep_vllm_topk is not in this build")
    step = _Step(STRADDLE + (9, 700), seed=5)
    caches = step.fresh_caches()
    step.chain(caches, early_trigger, prefetch)  # warm-up (compiles the reader)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step.chain(caches, early_trigger, prefetch)
    for rep in range(3):
        step.packed.copy_(
            torch.randn(
                step.packed.shape,
                generator=torch.Generator(device="cuda").manual_seed(rep),
                device="cuda",
            )
        )
        for dst, src in zip(caches, (step.pages0, step.data0, step.scale0)):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        ref_caches = step.fresh_caches()
        ref = step.chain(ref_caches, False, 0)
        torch.cuda.synchronize()
        q, index_q, scores, table = graph_out
        _assert_same((q, index_q, table), (ref[0], ref[1], ref[3]))
        _assert_same(_history_scores(step, scores), _history_scores(step, ref[2]))
        _assert_same(caches, ref_caches)
