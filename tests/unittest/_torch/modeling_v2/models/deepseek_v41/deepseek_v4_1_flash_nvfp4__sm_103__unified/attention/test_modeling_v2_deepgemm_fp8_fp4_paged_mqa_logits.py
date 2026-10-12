# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the deepgemm_fp8_fp4_paged_mqa_logits and deepgemm_paged_mqa_logits_metadata entries.

The entries own what is certified: `CELLS` is the list, `reference` and `compare` are the gate.
The schedule has no plain-torch reference; it is certified at the logits cells, as their input,
and its values are gated by the logits they drive. This file turns a cell's spec into tensors and
keeps the checks a cell cannot carry: the exact head-reduction order, input immutability, CUDA-graph
capture, the runtime widths, pools and mixed-step views, every batch and width as a prefix, the
loud rejections, the wrong-layout controls, the pool-size bounds and the long-context envelope.
``_Step.reference`` is the float64 twin of the entry's reference that reads the keys from their
logical storage (so it also serves pools held as views of a larger allocation) and takes the
wrong variants the controls need; ``_check`` applies the entry's gate wherever the step holds its
pool.

The DeepSeek-V4.1 decode / DSpark-verify indexer logits over the native MXFP4 index-K pool:
``score[b*next_n + j, s] = sum_h relu(q[b, j, h] . k[s]) * w[b*next_n + j, h]`` for the compressed
positions ``s < (p_b + j + 1) // ratio`` of request ``b`` whose first query token sits at absolute
position ``p_b``. Every input is packed here from native torch ops (no DeepGEMM helper, no other
catalog entry), and the reference decodes those same bytes and computes the score in fp64.
"""

import re
import time
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.deepgemm_fp8_fp4_paged_mqa_logits import (  # noqa: E501
    deepgemm_fp8_fp4_paged_mqa_logits as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.deepgemm_fp8_fp4_paged_mqa_logits import (  # noqa: E501
    deepgemm_paged_mqa_logits_metadata as metadata_op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "deepgemm_fp8_fp4_paged_mqa_logits requires a CUDA device"

DEV = "cuda"
HEADS = 32  # index_n_heads, replicated on every rank
DIM = 128  # index_head_dim
KEY_BYTES = 68  # 64 E2M1 code bytes + 4 UE8M0 scale bytes per key
# The pool's page count is the cache manager's number, never the batch's: each case allocates
# POOL_FACTOR x the pages its batch needs (+ POOL_EXTRA) and scatters the batch's pages over it
# with a random permutation, so "the page the block table names" and "the page this request
# happens to own" are different numbers. Chosen here, not taken from a runtime dump.
POOL_FACTOR = 3
POOL_EXTRA = 8
# Logits rows are padded to a 1024-byte multiple of a 256-column split (fp32: 256 columns).
SPLIT_KV = 256

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
    device=DEV,
)
_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=DEV)


# ─── packing (native torch; the reference recipe of the checkpoint's fp4_act_quant) ───────────


def _quant_mxfp4(x: torch.Tensor):
    """x [..., 128] -> (int8 [..., 64], int32 [...]).

    Per 32-block: scale = 2^ceil(log2(max(amax, 6*2^-126) / 6)) stored as a biased UE8M0 byte
    (byte b of the int32 = block b); codes = E2M1(x / scale) rounded to nearest-even, element 2i
    in the low nibble of byte i.
    """
    xf = x.float().unflatten(-1, (4, 32))
    amax = xf.abs().amax(-1).clamp_min(6 * 2.0**-126)
    bits = (amax * (1.0 / 6.0)).contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = ((e + 127) << 23).view(torch.float32)
    y = (xf / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    a = y.abs()
    code = (a.unsqueeze(-1) > _MIDPOINTS).sum(-1)
    code = code + ((a == 0.75) | (a == 1.75) | (a == 3.5)).to(code.dtype)  # ties to even code
    code = (code + 8 * ((y < 0) & (code > 0)).to(code.dtype)).flatten(-2)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8).view(torch.int8)
    sf = (e + 127).to(torch.uint8).contiguous().view(torch.int32).squeeze(-1)
    return packed.contiguous(), sf.contiguous()


def _decode(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """int8 [..., 64] + int32 [...] -> fp64 [..., 128], exact."""
    p = packed.view(torch.uint8).long()
    v = torch.stack([_E2M1[p & 0xF], _E2M1[p >> 4]], dim=-1).flatten(-2)
    sb = sf.contiguous().view(torch.uint8).view(*sf.shape, 4).long()
    # 2^(byte - 127) built from its fp64 bit pattern: exact (torch.pow(2.0, e) on CUDA is not,
    # it was measured 1 ulp low for some e)
    scale = ((sb + (1023 - 127)) << 52).view(torch.float64)
    return (v.unflatten(-1, (4, 32)) * scale.unsqueeze(-1)).flatten(-2)


def _write_pool(codes: torch.Tensor, sf: torch.Tensor, layout: str = "paged") -> torch.Tensor:
    """codes int8 [P, bk, 64] + sf int32 [P, bk] -> uint8 [P, bk, 1, 68].

    "paged" is the kernel's layout: a page holds its bk keys' 64 code bytes back to back, then
    their bk 4-byte scale words. "interleaved" (a control) stores each key's 68 bytes together.
    """
    P, bk, _ = codes.shape
    if layout == "paged":
        flat = torch.cat(
            [
                codes.view(torch.uint8).reshape(P, bk * 64),
                sf.view(torch.uint8).reshape(P, bk * 4),
            ],
            dim=1,
        )
    else:
        flat = torch.cat([codes.view(torch.uint8), sf.view(torch.uint8).view(P, bk, 4)], dim=2)
    return flat.contiguous().view(P, bk, 1, KEY_BYTES)


# ─── one decode step ──────────────────────────────────────────────────────────────────────────


class _Step:
    """A decode / DSpark-verify step of one indexer layer.

    ``positions[b]`` is the absolute position of request b's first query token, or None for a
    CUDA-graph padding request (no pages, context length 0).
    """

    def __init__(
        self,
        positions,
        next_n,
        ratio,
        block_kv,
        heads=HEADS,
        seed=0,
        extra_ctx=37,
        max_ctx=None,
        pool_pages=None,
    ):
        """``max_ctx`` fixes the static width (default: largest length + ``extra_ctx``).
        ``pool_pages`` sizes the pool to a runtime page count: the batch's pages are then drawn at
        random from the whole pool (always including its last page), and every other page holds
        0xFF bytes, whose E8M0 scale is NaN, so reading a page the table does not name turns a
        visible element into NaN."""
        g = torch.Generator(device=DEV)
        g.manual_seed(seed)
        self.B, self.n, self.H, self.bk, self.ratio = len(positions), next_n, heads, block_kv, ratio
        lens = [
            [0] * next_n if p is None else [(p + j + 1) // ratio for j in range(next_n)]
            for p in positions
        ]
        self.lens = torch.tensor(lens, dtype=torch.int32, device=DEV)
        # the static width a captured decode step would bake in: wider than this batch needs
        self.max_ctx = int(self.lens.max()) + extra_ctx if max_ctx is None else max_ctx
        assert self.max_ctx >= int(self.lens.max())
        pages = [(max(row) + block_kv - 1) // block_kv for row in lens]
        width = (self.max_ctx + block_kv - 1) // block_kv + 2
        if pool_pages is None:
            self.used = None
            self.num_pages = POOL_FACTOR * sum(pages) + POOL_EXTRA
            self.num_pages += self.num_pages % 2  # even, so a control can re-view it as 2x pages
            self.k_codes = torch.empty(
                self.num_pages, block_kv, DIM // 2, dtype=torch.int8, device=DEV
            )
            self.k_sf = torch.empty(self.num_pages, block_kv, dtype=torch.int32, device=DEV)
            self.refill_keys(g)
            perm = torch.randperm(self.num_pages, generator=g, device=DEV).to(torch.int32)
        else:
            self.num_pages = pool_pages
            perm = torch.randperm(pool_pages, generator=g, device=DEV)[: sum(pages)].to(torch.int32)
            if sum(pages) and not bool((perm == pool_pages - 1).any()):
                perm[-1] = pool_pages - 1
            self.used = perm.long()
            self.k_codes = torch.full(
                (pool_pages, block_kv, DIM // 2), -1, dtype=torch.int8, device=DEV
            )
            self.k_sf = torch.full((pool_pages, block_kv), -1, dtype=torch.int32, device=DEV)
            self.refill_keys(g)
        self.block_table = torch.full((self.B, width), -1, dtype=torch.int32, device=DEV)
        off = 0
        for b, npg in enumerate(pages):
            self.block_table[b, :npg] = perm[off : off + npg]
            off += npg
        # Keys past a request's length inside its last page are never visible; poison their
        # scale bytes with 0xFF (the E8M0 NaN encoding), as stale or unwritten slots may be.
        sf_bytes = self.k_sf.view(torch.uint8).view(self.num_pages, block_kv, 4)
        for b, row in enumerate(lens):
            L = max(row)
            if L % block_kv:
                sf_bytes[self.block_table[b, L // block_kv].long(), L % block_kv :] = 0xFF
        self.pool = _write_pool(self.k_codes, self.k_sf)
        q = torch.randn(self.B, next_n, heads, DIM, generator=g, device=DEV) * 2.0
        self.q, self.q_sf = _quant_mxfp4(q)
        # weights_proj(x) * 128^-0.5 * 32^-0.5 is a bf16 value; the kernel takes it as fp32
        w = torch.randn(self.B * next_n, heads, generator=g, device=DEV) * 2.0**-4
        self.weights = w.to(torch.bfloat16).float()

    def refill_keys(self, g):
        if self.used is not None:
            k = torch.randn(self.used.numel(), self.bk, DIM, generator=g, device=DEV)
            self.k_codes[self.used], self.k_sf[self.used] = _quant_mxfp4(k)
            return
        for c in range(0, self.num_pages, 4096):
            k = torch.randn(min(4096, self.num_pages - c), self.bk, DIM, generator=g, device=DEV)
            self.k_codes[c : c + k.shape[0]], self.k_sf[c : c + k.shape[0]] = _quant_mxfp4(k)

    def run(self, **over):
        a = dict(
            q=self.q,
            q_sf=self.q_sf,
            kv_cache=self.pool,
            weights=self.weights,
            context_lens=self.lens,
            block_table=self.block_table,
            max_context_len=self.max_ctx,
        )
        block_kv = over.pop("meta_block_kv", self.bk)
        a.update(over)
        meta = metadata_op(a["context_lens"], block_kv)
        return op(schedule_meta=meta, **a)

    def _keys(self, b):
        L = int(self.lens[b].max())
        npg = (L + self.bk - 1) // self.bk
        idx = self.block_table[b, :npg].long()
        return _decode(self.k_codes[idx], self.k_sf[idx]).reshape(-1, DIM)[:L]

    def _dots(self, b):
        """exact per-head q.k in fp64: [n, H, L]"""
        return torch.einsum("jhd,sd->jhs", _decode(self.q[b], self.q_sf[b]), self._keys(b))

    def reference(self, relu=True, weights=None):
        """fp64 [B*n, max_ctx]; -inf at every column row (b, j) must not see."""
        w = (self.weights if weights is None else weights).double()
        out = torch.full(
            (self.B * self.n, self.max_ctx), float("-inf"), dtype=torch.float64, device=DEV
        )
        for b in range(self.B):
            if int(self.lens[b].max()) == 0:
                continue
            d = self._dots(b)
            d = d.clamp_min(0) if relu else d
            score = (d * w[b * self.n : (b + 1) * self.n].unsqueeze(-1)).sum(1)
            for j in range(self.n):
                L = int(self.lens[b, j])
                out[b * self.n + j, :L] = score[j, :L]
        return out

    def emulate(self):
        """fp32 emulation of the kernel's head reduction: per head a = q.k (exact in fp32);
        four FMA chains c_l = fma(a_h + |a_h|, w_h, c_l) over heads h = l, l+4, ...; then
        ((c_0 + c_2) + (c_1 + c_3)) / 2. fma is formed exactly in fp64 and rounded once."""
        out = torch.full((self.B * self.n, self.max_ctx), float("-inf"), device=DEV)
        for b in range(self.B):
            if int(self.lens[b].max()) == 0:
                continue
            a = self._dots(b).float()
            two_relu = (a + a.abs()).double()
            w = self.weights[b * self.n : (b + 1) * self.n].double()
            chains = []
            for lane in range(4):
                acc = torch.zeros(self.n, a.shape[-1], device=DEV)
                for h in range(lane, self.H, 4):
                    acc = (two_relu[:, h] * w[:, h : h + 1] + acc.double()).float()
                chains.append(acc)
            red = ((chains[0] + chains[2]) + (chains[1] + chains[3])) / 2
            for j in range(self.n):
                L = int(self.lens[b, j])
                out[b * self.n + j, :L] = red[j, :L]
        return out


def _aligned_width(max_ctx: int, dtype: torch.dtype) -> int:
    split = -(-max_ctx // SPLIT_KV) * SPLIT_KV
    per_1k = 1024 // torch.tensor([], dtype=dtype).element_size()
    return -(-split // per_1k) * per_1k


def _check(step: _Step, out: torch.Tensor):
    """Shape/stride contract, the entry's gate where the step holds its pool, default fp32
    assert_close vs the fp64 twin, and the exact reduction order."""
    rows = step.B * step.n
    assert out.shape == (rows, step.max_ctx) and out.dtype == torch.float32
    assert out.stride() == (_aligned_width(step.max_ctx, torch.float32), 1)
    if step.pool is not None:
        op.compare(
            out,
            op.reference(
                step.q,
                step.q_sf,
                step.pool,
                step.weights,
                step.lens,
                step.block_table,
                None,
                step.max_ctx,
            ),
        )
    ref = step.reference()
    vis = torch.isfinite(ref)
    assert int(vis.sum()) == int(step.lens.sum())
    got = out[vis]
    torch.testing.assert_close(got, ref.float()[vis])
    # Stronger than the tolerance: the kernel's fp32 result is exactly the documented reduction
    # (measured bit-identical on every visible element on sm_103).
    assert torch.equal(got, step.emulate()[vis])
    return ref, vis


def _outside_fraction(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Fraction of elements outside the default fp32 assert_close gate."""
    ref32 = ref.float()
    tol = 1e-5 + 1.3e-6 * ref32.abs()
    return ((got - ref32).abs() > tol).float().mean().item()


# ─── cases ────────────────────────────────────────────────────────────────────────────────────

# (ratio, page): the INDEXER_COMPRESS page holds tokens_per_block / ratio entries;
# tokens_per_block 128 gives (2, 64) and (1, 128); tokens_per_block 256 gives (2, 128).
PAGES = [(2, 64), (1, 128), (2, 128)]


def _positions(regime, B, ratio, block_kv, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    if regime == "boundary":
        # first-row compressed lengths on page and 256-column split edges
        lens = [1, 2, block_kv - 1, block_kv, block_kv + 1, 255, 256, 257, 511, 513, 700, 1000]
        pos = [lens[i % len(lens)] * ratio - 1 + (i // len(lens)) for i in range(B)]
    else:
        lo, hi = {
            "gate": (1024, 16384),  # the accuracy gates' contexts
            "long": (20480, 40960),  # the long-context run (>= 20K tokens)
            "64k": (65536 * ratio - 8192, 65536 * ratio - 8),  # ~64K compressed positions
        }[regime]
        pos = torch.randint(lo, hi, (B,), generator=g).tolist()
    if B >= 8:
        pos[B // 2] = None  # a CUDA-graph padding request in the middle of the batch
    return pos


def _cell_step(spec, seed):
    """A cell's decode step: first-row positions from a regime, or the cell's own."""
    ratio, block_kv = spec["ratio"], spec["block_kv"]
    if "regime" in spec:
        positions = _positions(spec["regime"], spec["batch"], ratio, block_kv, seed)
    else:
        positions = spec["positions"]
    return _Step(positions, spec["next_n"], ratio, block_kv, heads=spec["heads"], seed=seed)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    # The schedule is certified at the logits cells, as their input: both entries are armed, and
    # the schedule's values are gated by the logits they drive.
    assert [c.spec for c in metadata_op.CELLS] == [c.spec for c in op.CELLS]
    certifying(op, metadata_op)
    spec = cell.spec
    dtype = spec.get("logits_dtype", torch.float32)
    step = _cell_step(spec, zlib.crc32(cell.why.encode()))
    before = [t.clone() for t in (step.q, step.q_sf, step.weights, step.lens, step.block_table)]
    pool_before = step.pool.clone() if step.pool.numel() < (1 << 26) else None
    meta = metadata_op(step.lens, step.bk)
    metadata_op.compare(meta, metadata_op.reference(step.lens, step.bk))
    args = (step.q, step.q_sf, step.pool, step.weights, step.lens, step.block_table, meta)
    out = op(*args, step.max_ctx, dtype)
    op.compare(out, op.reference(*args, step.max_ctx, dtype))
    assert out.shape == (step.B * step.n, step.max_ctx) and out.dtype == dtype
    assert out.stride() == (_aligned_width(step.max_ctx, dtype), 1)
    # Stronger than the gate: the kernel's fp32 result is exactly the documented head reduction
    # (bf16 logits: that value rounded once), on every visible element.
    vis = torch.arange(step.max_ctx, device=DEV)[None, :] < step.lens.view(-1, 1)
    assert torch.equal(out[vis], step.emulate()[vis].to(dtype))
    # inputs are not modified
    for t, t0 in zip((step.q, step.q_sf, step.weights, step.lens, step.block_table), before):
        assert torch.equal(t, t0)
    if pool_before is not None:
        assert torch.equal(step.pool, pool_before)


def test_cuda_graph_replay():
    """Metadata + logits captured together. Each replay reads that step's lengths, block-table
    rows, pool contents, queries and weights from the captured buffers; the captured static
    width (max_context_len) stays."""
    capacity = [5000, 12000, 700, 3000, None, 200, 64, 9000]
    step = _Step(capacity, 6, 2, 64, seed=11, extra_ctx=64)
    eager = step.run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            out = step.run()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    vis = torch.isfinite(step.reference())
    assert torch.equal(out[vis], eager[vis])
    g = torch.Generator(device=DEV)
    g.manual_seed(12)
    rows = step.block_table.clone()
    for perm in ([1, 0, 2, 3, 4, 5, 6, 7], [7, 1, 4, 3, 0, 6, 5, 2]):
        # request b now owns table row perm[b]; its new position fits that row's pages
        positions = []
        for b, r in enumerate(perm):
            cap = capacity[r]
            positions.append(None if cap is None or b == 4 else [cap - 3, 1, cap // 2][b % 3])
        new = _Step(positions, 6, 2, 64, seed=int(perm[0]) + 100)
        step.lens.copy_(new.lens)
        step.q.copy_(new.q)
        step.q_sf.copy_(new.q_sf)
        step.weights.copy_(new.weights)
        step.block_table.copy_(rows[perm])
        step.refill_keys(g)
        step.pool.copy_(_write_pool(step.k_codes, step.k_sf))
        graph.replay()
        torch.cuda.synchronize()
        _check(step, out)


# ─── the DeepSeek-V4.1 core target's runtime values ──────────────────────────────────────────
# From the core target's runtime-value dump (boot/eval/GSM8K runs, tep4/dep4/tp4):
# - the engine's index-K pools: (453120, 64, 1, 68) at ratio 2 and (151040, 128, 1, 68) at ratio 1;
# - max_context_len is the metadata's static max_compressed_kv_lens = max_positions // ratio, with
#   max_positions = max_seq_len + max_draft_tokens: 8192 / 16384 at max_seq_len 16384 and
#   16384 / 32768 at max_seq_len 32768 (ratio 2 / ratio 1), far above most rows' lengths;
# - decode batches (generation requests per step, next_n = 1) 1-10, 16-19, 24 and 27-32; the
#   ones below are those the boundary/gate/long/64k sweep does not enumerate.
RUNTIME_POOL_PAGES = {64: 453120, 128: 151040}
STATIC = [(2, 64, 8192), (1, 128, 16384), (2, 64, 16384), (1, 128, 32768)]
DUMP_BATCHES = [2, 3, 4, 5, 6, 7, 9, 10, 16, 17, 18, 19, 24, 27, 28, 29, 30, 31]


def _static_positions(B, next_n, ratio, max_ctx, seed):
    """First-row positions under a static max_context_len: request 0 sits at the cap (its last row
    sees exactly max_ctx entries), request 1 at position 1, the rest drawn from GSM8K-to-passkey
    contexts, so most rows are far shorter than the width."""
    g = torch.Generator()
    g.manual_seed(seed)
    cap = max_ctx * ratio - next_n
    pos = torch.randint(2, min(cap, 24576), (B,), generator=g).tolist()
    pos[0] = cap
    if B > 1:
        pos[1] = 1
    return pos


@pytest.mark.parametrize("batch", DUMP_BATCHES)
@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx", STATIC, ids=[f"r{r}_page{p}_w{w}" for r, p, w in STATIC]
)
def test_static_width_decode(ratio, block_kv, max_ctx, batch):
    """Decode at the target's static width over a runtime-sized pool: every visible element is
    exact (_check), and no row reads a page its table row does not name (those pages are NaN)."""
    seed = zlib.crc32(f"static/{ratio}/{block_kv}/{max_ctx}/{batch}".encode())
    pos = _static_positions(batch, 1, ratio, max_ctx, seed)
    step = _Step(
        pos, 1, ratio, block_kv, seed=seed, max_ctx=max_ctx, pool_pages=RUNTIME_POOL_PAGES[block_kv]
    )
    out = step.run()
    _check(step, out)
    assert int(step.lens.max()) == max_ctx
    assert out.is_contiguous()  # the static widths are multiples of 256 columns


@pytest.mark.parametrize("batch", [3, 10, 19, 31])
@pytest.mark.parametrize("ratio,block_kv,max_seq_len", [(2, 64, 16384), (1, 128, 32768)])
def test_static_width_dspark_verify(ratio, block_kv, max_seq_len, batch):
    """next_n = 6 under the DSpark static width (max_seq_len + 5 draft tokens) // ratio, which is
    not a multiple of 256: the rows are padded (stride 8448 / 33024)."""
    next_n, max_ctx = 6, (max_seq_len + 5) // ratio
    seed = zlib.crc32(f"dspark/{ratio}/{block_kv}/{max_ctx}/{batch}".encode())
    pos = _static_positions(batch, next_n, ratio, max_ctx, seed)
    step = _Step(
        pos,
        next_n,
        ratio,
        block_kv,
        seed=seed,
        max_ctx=max_ctx,
        pool_pages=RUNTIME_POOL_PAGES[block_kv],
    )
    _check(step, step.run())
    assert int(step.lens.max()) == max_ctx


@pytest.mark.parametrize("ratio,block_kv,pool_pages", [(2, 64, 600000), (1, 128, 300000)])
def test_pool_past_2gib(ratio, block_kv, pool_pages):
    """A pool of 2.61 GB, about 1.3x the runtime's 1.97 GB (453120 x 64 pages): the pages the batch
    owns are drawn from the whole pool, so about a sixth of them start past byte 2^31."""
    max_ctx = 16384 * 2 // ratio
    seed = zlib.crc32(f"pool2g/{ratio}/{block_kv}".encode())
    pos = _static_positions(8, 1, ratio, max_ctx, seed)
    step = _Step(pos, 1, ratio, block_kv, seed=seed, max_ctx=max_ctx, pool_pages=pool_pages)
    page_bytes = block_kv * KEY_BYTES
    assert step.pool.numel() > 2**31 and int((step.used * page_bytes >= 2**31).sum()) > 0
    _check(step, step.run())


def _embed(x, offset, fill):
    """x placed at row ``offset`` of a larger buffer; returns that row-offset view."""
    buf = torch.full((offset + x.shape[0] + 3, *x.shape[1:]), fill, dtype=x.dtype, device=DEV)
    buf[offset : offset + x.shape[0]] = x
    return buf[offset : offset + x.shape[0]]


def _mixed_views(step, tc, nc, pad):
    """The views a mixed (context + generation) step hands the op: q / q_sf / weights /
    context_lens start at token row tc of the step's buffers, the table at request row nc of
    the metadata's [max_num_sequences, max_blocks_per_seq] table (here ``pad`` columns wider)."""
    rows = step.B * step.n
    width = step.block_table.shape[1]
    table = torch.zeros(nc + step.B + 2, width + pad, dtype=torch.int32, device=DEV)
    table[nc : nc + step.B, :width] = step.block_table
    return dict(
        q=_embed(step.q.view(rows, step.H, DIM // 2), tc, 0).view(step.B, step.n, step.H, DIM // 2),
        q_sf=_embed(step.q_sf.view(rows, step.H), tc, 0).view(step.B, step.n, step.H),
        weights=_embed(step.weights, tc, 0.0),
        context_lens=_embed(step.lens.view(-1), tc, 0).view(step.B, step.n),
        block_table=table[nc : nc + step.B],
    )


@pytest.mark.parametrize("tc,nc,pad", [(1, 1, 1), (573, 3, 0), (7674, 2, 1)])
def test_mixed_step_row_offset_views(tc, nc, pad):
    """Row-offset views, including context_lens and table bases only 4-byte aligned (tc 1 / 573;
    nc 1 at a 131-column table) and a table row stride wider than the rows need."""
    pos = [100, 3000, 7000, 16383, 555, 1, 2, 9000, 4000, 12000]
    step = _Step(pos, 1, 2, 64, seed=tc + nc, max_ctx=8192, pool_pages=RUNTIME_POOL_PAGES[64])
    views = _mixed_views(step, tc, nc, pad)
    assert views["block_table"].stride(0) == step.block_table.shape[1] + pad
    _check(step, step.run(**views))


# The union of the gates' capture sets, per rank: every decode batch 1..8 (boot engine, padding
# off) and 1/2/4/8/16/24/32 (accuracy engine, padding on), at next_n 1.
CORE_GRAPH_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32]


def _poison_tails(step):
    """0xFF (E8M0 NaN) scale bytes on every slot past each request's last-row length in its last
    page: such slots are never visible, whatever the pool holds there."""
    sf_bytes = step.k_sf.view(torch.uint8).view(step.num_pages, step.bk, 4)
    for b in range(step.B):
        L = int(step.lens[b].max())
        if L % step.bk:
            sf_bytes[step.block_table[b, L // step.bk].long(), L % step.bk :] = 0xFF


def _capture(step):
    """Metadata + logits captured together on a side stream, as the target's step captures them."""
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            out = step.run()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, out


def _replay_and_check(step, graph, out, tag, prev):
    """One replay into an output poisoned with NaN: exact vs the fp64 reference and bit-exact vs
    the documented reduction (_check), bitwise equal to an eager call on the same inputs, and
    rejected by the previous step's reference (a replay that re-read the captured step is seen)."""
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    ref, vis = _check(step, out)
    again = step.run()
    assert torch.equal(out[vis], again[vis])
    prev_ref, prev_vis = prev
    both = vis & prev_vis
    stale = _outside_fraction(out[both], prev_ref[both])
    print(
        f"[{tag}] visible {int(vis.sum())}, previous step's reference: {stale:.4f} of "
        f"{int(both.sum())} outside the gate"
    )
    assert stale > 0.5
    return ref, vis


@pytest.mark.parametrize("batch", CORE_GRAPH_BATCHES)
@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx", STATIC, ids=[f"r{r}_page{p}_w{w}" for r, p, w in STATIC]
)
def test_cuda_graph_static_width(ratio, block_kv, max_ctx, batch):
    """The target captures metadata + logits per decode graph batch with the static width baked
    in. Every request's table row covers the full width. The first replay equals the eager
    warm-up call bitwise; each of 3 more replays reads new lengths (incl. 0, 1 and the cap),
    queries, weights and keys (tails poisoned NaN) and is checked by _replay_and_check."""
    seed = zlib.crc32(f"graph/{ratio}/{block_kv}/{max_ctx}/{batch}".encode())
    step = _Step(
        [max_ctx * ratio - 1] * batch,
        1,
        ratio,
        block_kv,
        seed=seed,
        max_ctx=max_ctx,
        pool_pages=RUNTIME_POOL_PAGES[block_kv],
    )
    eager = step.run()
    torch.cuda.synchronize()
    graph, out = _capture(step)
    graph.replay()
    torch.cuda.synchronize()
    prev = _check(step, out)
    assert torch.equal(out[prev[1]], eager[prev[1]])
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    for rep in range(3):
        lens = torch.randint(0, max_ctx + 1, (batch, 1), generator=g, device=DEV, dtype=torch.int32)
        lens[0] = max_ctx
        if batch > 1:
            lens[-1] = rep % 2  # a padding-like row: 0 or 1 entries
        step.lens.copy_(lens)
        new = _Step([1] * batch, 1, ratio, block_kv, seed=seed + rep + 1)
        step.q.copy_(new.q)
        step.q_sf.copy_(new.q_sf)
        step.weights.copy_(new.weights)
        step.refill_keys(g)
        _poison_tails(step)
        step.pool.copy_(_write_pool(step.k_codes, step.k_sf))
        prev = _replay_and_check(
            step, graph, out, f"graph r{ratio} w{max_ctx} B{batch} rep{rep}", prev
        )


# ─── DSpark verify under CUDA graph at the target's graph batches ─────────────────────────────
# With DSpark on, the target's generation-only steps are captured with next_n = 6 rows per
# request at the union of the gates' capture sets: every G in 1..8 (the DSpark boot
# engine: max_batch_size 8, padding off; its runtime-value dump records G 3, 5, 6 and 7 at the
# graph-copy widths 8201 / 16403) and G 1, 2, 4, 8, 16, 24, 32 (the accuracy engine, padding on).
# max_context_len is the metadata's max_compressed_kv_lens[r] = (max_seq_len + max_draft_tokens)
# // r: max_draft_tokens is 0 in the eager metadata and can be 5 in its CUDA-graph copy (which
# re-runs __post_init__ after the spec-dec update), so both widths are covered, at max_seq_len
# 16384 (gates) and 32768 (the long-context run).
DSPARK_GRAPH_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32]
DSPARK_STATIC = [
    (2, 64, 8192),
    (2, 64, 8194),
    (1, 128, 16384),
    (1, 128, 16389),
    (2, 64, 16384),
    (2, 64, 16386),
    (1, 128, 32768),
    (1, 128, 32773),
    # The DSpark engine's own widths: the runtime raises the engine's max_seq_len
    # to 16398, so the eager / graph-copy widths are 16398 // r and 16403 // r.
    (2, 64, 8199),
    (2, 64, 8201),
    (1, 128, 16398),
    (1, 128, 16403),
]


def _dspark_lens(B, ratio, max_ctx, rep, g):
    """Per-row lengths (p + j + 1) // ratio of a verify step: request 0 at the cap (its last row
    sees exactly max_ctx entries), request 1 at position rep % 2 (first rows see 0 or 1 entries),
    on even reps a graph padding request (all 0) last, the rest random below the cap."""
    cap = max_ctx * ratio - 6
    pos = torch.randint(0, cap + 1, (B,), generator=g).tolist()
    pos[0] = cap
    if B > 1:
        pos[1] = rep % 2
    if B > 2 and rep % 2 == 0:
        pos[-1] = None
    lens = [[0] * 6 if p is None else [(p + j + 1) // ratio for j in range(6)] for p in pos]
    return torch.tensor(lens, dtype=torch.int32, device=DEV)


@pytest.mark.parametrize("batch", DSPARK_GRAPH_BATCHES)
@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx", DSPARK_STATIC, ids=[f"r{r}_page{p}_w{w}" for r, p, w in DSPARK_STATIC]
)
def test_cuda_graph_dspark_verify(ratio, block_kv, max_ctx, batch):
    """next_n = 6 captured at every DSpark graph batch with each static width: metadata + logits
    in one graph, as the target captures them. The first replay equals the eager warm-up call
    bitwise; 3 more replays read new lengths (the cap, 0/1, a padding request), queries, weights
    and keys (tails poisoned NaN), each checked by _replay_and_check."""
    seed = zlib.crc32(f"graph-dspark/{ratio}/{block_kv}/{max_ctx}/{batch}".encode())
    step = _Step(
        [max_ctx * ratio - 6] * batch,
        6,
        ratio,
        block_kv,
        seed=seed,
        max_ctx=max_ctx,
        pool_pages=RUNTIME_POOL_PAGES[block_kv],
    )
    assert int(step.lens.max()) == max_ctx
    eager = step.run()
    torch.cuda.synchronize()
    graph, out = _capture(step)
    graph.replay()
    torch.cuda.synchronize()
    prev = _check(step, out)
    assert torch.equal(out[prev[1]], eager[prev[1]])
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    gh = torch.Generator()
    gh.manual_seed(seed)
    for rep in range(3):
        step.lens.copy_(_dspark_lens(batch, ratio, max_ctx, rep, gh))
        new = _Step([1] * batch, 6, ratio, block_kv, seed=seed + rep + 1)
        step.q.copy_(new.q)
        step.q_sf.copy_(new.q_sf)
        step.weights.copy_(new.weights)
        step.refill_keys(g)
        _poison_tails(step)
        step.pool.copy_(_write_pool(step.k_codes, step.k_sf))
        prev = _replay_and_check(
            step, graph, out, f"graph dspark r{ratio} w{max_ctx} G{batch} rep{rep}", prev
        )


# ─── every generation batch 1..32 at the static widths ───────────────────────────────────────
# Rows are independent: each (b, j) row reads only its own request's q, weights, length and
# table row, so a batch's result is the first B requests of a 32-request call. One 32-request
# step per (width, next_n) is checked against the reference, and every B in 1..31 over the first
# B requests (prefix views of the same buffers, schedule rebuilt from those lengths) must be
# bitwise equal to the first B * next_n rows of that call on every visible element.
PREFIX_CASES = [(r, p, w, 1) for r, p, w in STATIC] + [(r, p, w, 6) for r, p, w in DSPARK_STATIC]


@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx,next_n",
    PREFIX_CASES,
    ids=[f"r{r}_page{p}_w{w}_nextn{n}" for r, p, w, n in PREFIX_CASES],
)
def test_every_batch_is_a_prefix(ratio, block_kv, max_ctx, next_n):
    seed = zlib.crc32(f"prefix/{ratio}/{block_kv}/{max_ctx}/{next_n}".encode())
    pos = _static_positions(32, next_n, ratio, max_ctx, seed)
    pos[5] = None  # a graph padding request: length 0, all -1 table row
    step = _Step(
        pos,
        next_n,
        ratio,
        block_kv,
        seed=seed,
        max_ctx=max_ctx,
        pool_pages=RUNTIME_POOL_PAGES[block_kv],
    )
    full = step.run()
    _, vis = _check(step, full)
    for B in range(1, 32):
        rows = B * next_n
        out = step.run(
            q=step.q[:B],
            q_sf=step.q_sf[:B],
            weights=step.weights[:rows],
            context_lens=step.lens[:B],
            block_table=step.block_table[:B],
        )
        assert out.shape == (rows, max_ctx)
        v = vis[:rows]
        assert torch.equal(out[v], full[:rows][v]), f"batch {B} differs from the prefix"


# ─── deliberately wrong controls: the same gate must reject each ─────────────────────────────


def _control_step():
    return _Step([130, 999, 4100, 20001, 777, 3, 12000, 64], 6, 2, 64, seed=21)


def test_control_wrong_block_table():
    """Each request reads its neighbour's pages (block-table rows rolled by one)."""
    step = _control_step()
    ref = step.reference()
    vis = torch.isfinite(ref)
    rolled = step.block_table.roll(1, dims=0).contiguous()
    got = step.run(block_table=rolled)[vis]
    frac = _outside_fraction(got, ref[vis])
    print(f"[control wrong_block_table] outside the gate: {frac:.4f}")
    assert frac > 0.5
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, ref.float()[vis])


def test_control_wrong_page_size():
    """Pages written 64 entries wide, read as 128-entry pages (the table halved accordingly)."""
    step = _control_step()
    ref = step.reference()
    vis = torch.isfinite(ref)
    pool128 = step.pool.view(-1, 128, 1, KEY_BYTES)
    bt128 = (step.block_table // 2).contiguous()
    got = step.run(kv_cache=pool128, block_table=bt128, meta_block_kv=128)[vis]
    frac = _outside_fraction(got, ref[vis])
    print(f"[control wrong_page_size] outside the gate: {frac:.4f}")
    assert frac > 0.5
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, ref.float()[vis])


def test_control_key_interleaved_layout():
    """The pool written key-interleaved ([64 codes | 4 scales] per key) instead of paged."""
    step = _control_step()
    ref = step.reference()
    vis = torch.isfinite(ref)
    pool = _write_pool(step.k_codes, step.k_sf, layout="interleaved")
    got = step.run(kv_cache=pool)[vis]
    frac = _outside_fraction(got, ref[vis])
    print(f"[control key_interleaved_layout] outside the gate: {frac:.4f}")
    assert frac > 0.5
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, ref.float()[vis])


def test_control_reference_without_relu():
    """The gate sees the relu: the op's output against a relu-free score."""
    step = _control_step()
    ref = step.reference(relu=False)
    vis = torch.isfinite(ref)
    got = step.run()[vis]
    frac = _outside_fraction(got, ref[vis])
    print(f"[control reference_without_relu] outside the gate: {frac:.4f}")
    assert frac > 0.5
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, ref.float()[vis])


def test_control_heads_reversed_weights():
    """Head weights applied in reverse head order."""
    step = _control_step()
    ref = step.reference()
    vis = torch.isfinite(ref)
    got = step.run(weights=step.weights.flip(-1).contiguous())[vis]
    frac = _outside_fraction(got, ref[vis])
    print(f"[control heads_reversed_weights] outside the gate: {frac:.4f}")
    assert frac > 0.5
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, ref.float()[vis])


# ─── loud rejections (host-side asserts; the CUDA context survives) ───────────────────────────


def _raises(fn, text):
    with pytest.raises(RuntimeError, match=re.escape(text)):
        fn()


def test_rejections():
    step = _Step([130, 999], 1, 2, 64, seed=31)
    meta = metadata_op(step.lens, 64)
    base = dict(
        q=step.q,
        q_sf=step.q_sf,
        kv_cache=step.pool,
        weights=step.weights,
        context_lens=step.lens,
        block_table=step.block_table,
        schedule_meta=meta,
        max_context_len=step.max_ctx,
    )

    def call(**over):
        return op(**{**base, **over})

    page_check = "block_kv == 32 or block_kv == 64 or block_kv == 128"
    # tokens_per_block 256 at ratio 1 would give 256-entry index-K pages
    _raises(lambda: metadata_op(step.lens, 256), page_check)
    pool256 = torch.zeros(4, 256, 1, KEY_BYTES, dtype=torch.uint8, device=DEV)
    _raises(lambda: call(kv_cache=pool256), f"(arch_major == 10 and ({page_check}))")
    # Any multiple of 4 up to 128 heads is taken (24 is measured against the reference in
    # test_head_count_off_the_cells_matches_the_reference); 6 is refused.
    _raises(
        lambda: call(
            q=step.q[:, :, :6].contiguous(),
            q_sf=step.q_sf[:, :, :6].contiguous(),
            weights=step.weights[:, :6].contiguous(),
        ),
        "num_heads > 0 and num_heads <= 128 and num_heads % 4 == 0",
    )
    _raises(lambda: call(context_lens=step.lens.view(-1)), "context_lens.dim() == 2")
    _raises(
        lambda: metadata_op(step.lens.view(-1), 64),
        "context_lens.dim() == 2",
    )
    other_sms = metadata_op(step.lens, 64, num_sms=8)
    _raises(
        lambda: call(schedule_meta=other_sms),
        "_schedule_meta_size == num_sms / num_ctas_per_cluster + 1",
    )
    _raises(
        lambda: call(weights=step.weights.bfloat16()),
        "weights.scalar_type() != torch::kBFloat16 or logits_dtype == torch::kBFloat16",
    )
    _raises(
        lambda: call(logits_dtype=torch.float16),
        "logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16",
    )
    wt = torch.empty(HEADS, step.B, device=DEV).t()
    wt.copy_(step.weights)
    _raises(lambda: call(weights=wt), "weights.stride(1) == 1")
    _raises(
        lambda: call(block_table=step.block_table.long()),
        "block_table.scalar_type() == torch::kInt",
    )
    _raises(
        lambda: call(context_lens=step.lens.long()), "context_lens.scalar_type() == torch::kInt"
    )
    _raises(
        lambda: call(q=step.q.view(torch.uint8)),
        "q_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn)",
    )
    _raises(
        lambda: call(kv_cache=step.pool.view(torch.int8)),
        "fused_kv_cache.scalar_type() == torch::kByte",
    )
    # the context survived every rejection
    _check(step, step.run())


def test_head_count_off_the_cells_matches_the_reference():
    """24 heads, which the cells do not cover and which does not divide 128: the kernel takes it,
    and its logits are held to the entry's gate and the fp64 twin. A measurement that the
    acceptance is not a silent wrong answer, not a certification of 24 heads."""
    step = _Step([130, 999], 1, 2, 64, heads=24, seed=32)
    out = step.run()
    rows = step.B * step.n
    assert out.shape == (rows, step.max_ctx) and out.dtype == torch.float32
    op.compare(
        out,
        op.reference(
            step.q,
            step.q_sf,
            step.pool,
            step.weights,
            step.lens,
            step.block_table,
            None,
            step.max_ctx,
        ),
    )
    ref = step.reference()
    vis = torch.isfinite(ref)
    torch.testing.assert_close(out[vis], ref.float()[vis])


# ─── controls for the runtime-value cases above ───────────────────────────────────────────────


def test_control_table_names_unowned_page():
    """The runtime-sized pool makes a stray page read visible: one table entry per request moved
    to a page the batch does not own (0xFF bytes, NaN scale) fails the gate with NaN."""
    step = _Step(
        [100, 3000, 7000, 16383, 555, 9000, 4000, 12000],
        1,
        2,
        64,
        seed=41,
        max_ctx=8192,
        pool_pages=RUNTIME_POOL_PAGES[64],
    )
    ref = step.reference()
    vis = torch.isfinite(ref)
    owned = torch.zeros(step.num_pages, dtype=torch.bool, device=DEV)
    owned[step.used] = True
    stray = int(torch.nonzero(~owned)[0])
    table = step.block_table.clone()
    table[:, 0] = stray
    got = step.run(block_table=table)[vis]
    ref32 = ref.float()[vis]
    n_out = int(
        (~((got - ref32).abs() <= 1e-5 + 1.3e-6 * ref32.abs())).sum()
    )  # NaN counts as outside
    n_nan = int(torch.isnan(got).sum())
    print(
        f"[control table_names_unowned_page] outside the gate: {n_out} of {got.numel()} visible "
        f"(the entries of each request's first page), NaN: {n_nan}"
    )
    assert n_nan > 0
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, ref.float()[vis])


def test_control_mixed_step_offset_lost():
    """A mixed step whose query-side views lost their token offset (each generation row reads
    the row tc - 1 before it, i.e. the last context token for row 0 and its neighbour for the
    rest) fails the gate."""
    tc = 573
    step = _Step(
        [100, 3000, 7000, 16383, 555, 1, 2, 9000, 4000, 12000],
        1,
        2,
        64,
        seed=42,
        max_ctx=8192,
        pool_pages=RUNTIME_POOL_PAGES[64],
    )
    views = _mixed_views(step, tc, 1, 1)
    rows = step.B * step.n
    for k in ("q", "q_sf", "weights"):
        base = views[k].reshape(rows, *views[k].shape[2:]) if k != "weights" else views[k]
        shifted = base.as_strided(base.shape, base.stride(), base.storage_offset() - base.stride(0))
        views[k] = shifted.reshape(views[k].shape)
    ref = step.reference()
    vis = torch.isfinite(ref)
    got = step.run(**views)[vis]
    frac = _outside_fraction(got, ref[vis])
    print(f"[control mixed_step_offset_lost] outside the gate: {frac:.4f}")
    assert frac > 0.5
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, ref.float()[vis])


# ─── pool size: any B <= B_BOUND[ratio] blocks ─────────────────────────────────────────────────
# The cache manager's INDEXER_COMPRESS storage at B blocks (tokens_per_block 128): ratio 1 is one
# uint8 [B, 128, 1, 68] pool; ratio 2 is ONE allocation of 3B 64-entry pages shared by the three
# ratio-2 owners, owner k (k = 0, 1, 2) seeing the view alloc[k:] of 3B - k pages, its logical
# block b at view page 3b (its table holds 3 x block). B depends on free memory. For a manager that
# holds every layer, 298,013,687,808 B of device memory over the 113,920 persistent bytes per block
# of native KV storage (890 B/token x 128) gives B_MAX = 2,615,990; the bf16-storage engines
# (349,440 B/block) stay below 852,832. A pipeline stage's manager holds fewer bytes per block, so
# that derivation does not bound it; its measured maxima follow.
B_MAX = 2_615_990
B_CORE_BF16 = 852_832
# The engine envelope's managers (max_num_tokens up to 16384, max_batch_size up to 256, max_seq_len up
# to 1,048,576), measured at 0.5 x the GB300's memory as KV quota over a sampled (max_num_tokens,
# max_batch_size, DSpark, max_seq_len) grid: the unsplit manager's index-K pools reach 1,295,549
# blocks (ratio 1; ratio 2: 3,886,647 pages). A PP2 stage manager holds half the layers:
# stage 1 (layers 20-39, the ratio-1 index source) reaches 3,245,959 ratio-1 blocks built alone and
# 2,159,957 in a PP2 engine (whose ranks agree on the smaller stage's token capacity); stage 0 (the
# ratio-2 sources) reaches 6,480,780 ratio-2 pages = 3 x 2,160,260 blocks, below 3 x B_MAX.
B_ENV_UNSPLIT = 1_295_549
B_PP2_R1_ENGINE = 2_159_957
B_PP2_R2_STAGE = 2_160_260
B_PP2_R1_STAGE = 3_245_959
B_BOUND = {1: B_PP2_R1_STAGE, 2: B_MAX}  # the certified ceiling per ratio, in blocks
INV_BLOCKS = 160  # the invariance batches use blocks 0..159 only: B_min = 160
POOL_SIZES = {
    1: [
        INV_BLOCKS,
        481,
        1493,
        151040,
        B_CORE_BF16,
        B_ENV_UNSPLIT,
        B_PP2_R1_ENGINE,
        B_MAX,
        B_PP2_R1_STAGE,
    ],
    2: [INV_BLOCKS, 481, 1493, 151040, B_CORE_BF16, B_ENV_UNSPLIT, B_PP2_R2_STAGE, B_MAX],
}
GATE_WIDTHS = [(2, 64, 8192), (1, 128, 16384)]


def _alloc_index_k(B, ratio):
    """The manager's storage at B blocks, every byte 0xFF (NaN scale): an exact-size allocation."""
    pages, bk = (B, 128) if ratio == 1 else (3 * B, 64)
    return torch.full((pages, bk, 1, KEY_BYTES), 0xFF, dtype=torch.uint8, device=DEV)


def _owners(ratio):
    return [0] if ratio == 1 else [0, 1, 2]


class _PoolStep(_Step):
    """A step whose keys live in a caller-owned pool: ``block_table`` holds view page ids and
    ``slots`` [B, width] the row of ``codes`` / ``sf`` each named page holds (-1 past a request's
    pages). The reference and the emulation decode the keys from those rows."""

    def __init__(self, positions, next_n, ratio, block_kv, pages_of, seed, max_ctx):
        g = torch.Generator(device=DEV)
        g.manual_seed(seed)
        self.B, self.n, self.H, self.bk, self.ratio = len(positions), next_n, HEADS, block_kv, ratio
        lens = [
            [0] * next_n if p is None else [(p + j + 1) // ratio for j in range(next_n)]
            for p in positions
        ]
        self.lens = torch.tensor(lens, dtype=torch.int32, device=DEV)
        self.max_ctx = max_ctx
        self.pool = None  # every call passes its owner view
        width = (max_ctx + block_kv - 1) // block_kv + 2
        self.block_table = torch.full((self.B, width), -1, dtype=torch.int32, device=DEV)
        for b, row in enumerate(lens):
            npg = (max(row) + block_kv - 1) // block_kv
            if npg:
                self.block_table[b, :npg] = torch.tensor(pages_of(b, npg), dtype=torch.int32)
        named = self.block_table[self.block_table >= 0]
        self.pages = torch.unique(named).long()  # sorted view page ids the batch names
        self.slots = torch.full_like(self.block_table, -1)
        self.slots[self.block_table >= 0] = torch.searchsorted(self.pages, named.long()).to(
            torch.int32
        )
        k = torch.randn(self.pages.numel(), block_kv, DIM, generator=g, device=DEV)
        self.k_codes, self.k_sf = _quant_mxfp4(k)
        self.page_bytes = _write_pool(self.k_codes, self.k_sf)
        q = torch.randn(self.B, next_n, HEADS, DIM, generator=g, device=DEV) * 2.0
        self.q, self.q_sf = _quant_mxfp4(q)
        w = torch.randn(self.B * next_n, HEADS, generator=g, device=DEV) * 2.0**-4
        self.weights = w.to(torch.bfloat16).float()

    def _keys(self, b):
        L = int(self.lens[b].max())
        npg = (L + self.bk - 1) // self.bk
        idx = self.slots[b, :npg].long()
        return _decode(self.k_codes[idx], self.k_sf[idx]).reshape(-1, DIM)[:L]

    def physical(self, k):
        """Allocation pages of the named view pages for the owner whose view starts k pages in."""
        return self.pages + k

    def run_on(self, view, **over):
        return self.run(kv_cache=view, **over)


def _written_span(step, out):
    """Every element the kernel writes: request b's rows up to ceil256 of its last row's length
    (at most the width). Returned as int32 bit patterns so NaN compares bitwise."""
    bits = out.view(torch.int32)
    parts = []
    for b in range(step.B):
        span = min(-(-int(step.lens[b].max()) // SPLIT_KV) * SPLIT_KV, step.max_ctx)
        parts.append(bits[b * step.n : (b + 1) * step.n, :span].reshape(-1))
    return torch.cat(parts)


def _invariance_steps(ratio, block_kv, max_ctx):
    """next_n 1 and 6 at batches 1, 7 and 32 (request 0 at the width, request 1 at position 1, a
    padding request at 32), every request's pages a random choice of blocks 0..INV_BLOCKS-1 (pages
    alias across requests, as prefix reuse does), block INV_BLOCKS-1 always named."""
    mult = 1 if ratio == 1 else 3
    steps = {}
    for next_n in (1, 6):
        for batch in (1, 7, 32):
            seed = zlib.crc32(f"pool-inv/{ratio}/{max_ctx}/{next_n}/{batch}".encode())
            pos = _static_positions(batch, next_n, ratio, max_ctx, seed)
            if batch == 32:
                pos[5] = None
            gh = torch.Generator()
            gh.manual_seed(seed)

            def pages_of(b, npg, gh=gh):
                blocks = torch.randperm(INV_BLOCKS, generator=gh)[:npg]
                if b == 0 and not bool((blocks == INV_BLOCKS - 1).any()):
                    blocks[-1] = (
                        INV_BLOCKS - 1
                    )  # the smallest pool is exactly the one that holds it
                return (blocks * mult).tolist()

            steps[(next_n, batch)] = _PoolStep(
                pos, next_n, ratio, block_kv, pages_of, seed, max_ctx
            )
    return steps


@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx", GATE_WIDTHS, ids=[f"r{r}_page{p}_w{w}" for r, p, w in GATE_WIDTHS]
)
def test_pool_size_invariance(ratio, block_kv, max_ctx):
    """The same queries, weights, lengths, block tables and page bytes at every pool size B in
    POOL_SIZES[ratio] (an exact-size allocation each, one at a time; every page the batch does not
    own is 0xFF): the result is bitwise identical across B on every element the kernel writes (each
    request's rows up to ceil256 of its last row's length), and passes the reference at B_min.
    Ratio 2 runs each owner's view (0, 1 and 2 pages into the shared allocation)."""
    steps = _invariance_steps(ratio, block_kv, max_ctx)
    first = {}
    sizes = POOL_SIZES[ratio]
    assert max(sizes) == B_BOUND[ratio]
    for B in sizes:
        alloc = _alloc_index_k(B, ratio)
        for k in _owners(ratio):
            view = alloc[k:]
            for key, step in steps.items():
                phys = step.physical(k)
                alloc[phys] = step.page_bytes
                out = step.run_on(view)
                alloc[phys] = 0xFF
                span = _written_span(step, out)
                if B == sizes[0]:
                    assert int(step.pages.max()) // (3 if ratio == 2 else 1) == INV_BLOCKS - 1
                    _check(step, out)
                    first[(k, key)] = span
                else:
                    assert torch.equal(span, first[(k, key)]), (
                        f"B {B}, owner view +{k}, next_n/batch {key}: differs from B {sizes[0]}"
                    )
        print(
            f"[pool invariance r{ratio} w{max_ctx}] B {B}: {alloc.numel() / 1e9:.2f} GB, "
            f"{len(steps) * len(_owners(ratio))} calls bitwise equal to B {sizes[0]}"
        )
        del alloc, view
        torch.cuda.empty_cache()


def _far_pages(ratio, B, k):
    """Owner k's view pages at the far end of the B-block storage: its last page, and the pages
    around byte offsets 2^31, 2^32, 2^33 and 2^34 of the allocation (the page holding each offset
    straddles it; ratio 2 takes the owner's pages among the five around it)."""
    bk = 128 if ratio == 1 else 64
    pb = bk * KEY_BYTES
    alloc_pages = B if ratio == 1 else 3 * B
    mult = 1 if ratio == 1 else 3
    phys = [alloc_pages - 3 + k if ratio == 2 else alloc_pages - 1]
    straddling = []
    for n in (31, 32, 33, 34):
        p = 2**n // pb
        assert 2**n % pb and p < alloc_pages
        straddling.append(p)
        phys += [q for q in range(p - 2, p + 3) if q % mult == k]
    view = [q - k for q in phys]
    assert all(v % mult == 0 for v in view)
    return view, [p - k for p in straddling if p % mult == k]


# (ratio, page, width, blocks): both ratios at B_MAX at the gate widths, and ratio 1 at the PP2
# stage-1 ceiling at the gate width and at the largest DSpark graph width (8193 pages per request).
FAR_CASES = [
    (2, 64, 8192, B_MAX),
    (1, 128, 16384, B_MAX),
    (1, 128, 16384, B_PP2_R1_STAGE),
    (1, 128, 1048595, B_PP2_R1_STAGE),
]


@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx,B",
    FAR_CASES,
    ids=[f"r{r}_page{p}_w{w}_B{b}" for r, p, w, b in FAR_CASES],
)
def test_pool_far_end_addressing(ratio, block_kv, max_ctx, B):
    """At B blocks (ratio 1 at B_MAX: 22.77 GB, at B_PP2_R1_STAGE: 28.25 GB; ratio 2 at B_MAX:
    7,847,970 pages, 34.15 GB), each owner view: a batch of 8 whose request 0 (at the width) names
    the owner's last page and its pages around byte offsets 2^31..2^34 first, and whose other pages
    are drawn from the whole view, passes the reference at next_n 1 and 6. Two wrong tables must
    fail the same check: every entry moved to its neighbouring page, and page ids whose byte offset
    wraps at 32 bits."""
    alloc = _alloc_index_k(B, ratio)
    pb = block_kv * KEY_BYTES
    mult = 1 if ratio == 1 else 3
    for k in _owners(ratio):
        view = alloc[k:]
        far, straddle = _far_pages(ratio, B, k)
        for next_n in (1, 6):
            key = f"pool-far/{ratio}/{k}/{next_n}" + ("" if B == B_MAX else f"/{B}/{max_ctx}")
            seed = zlib.crc32(key.encode())
            pos = _static_positions(8, next_n, ratio, max_ctx, seed)
            gh = torch.Generator()
            gh.manual_seed(seed)

            def pages_of(b, npg, gh=gh, far=far):
                rnd = (torch.randint(0, B, (npg,), generator=gh) * mult).tolist()
                return (far + rnd)[:npg] if b == 0 else rnd

            step = _PoolStep(pos, next_n, ratio, block_kv, pages_of, seed, max_ctx)
            assert set(far) <= set(step.block_table[0].tolist())
            named = step.pages * pb
            assert far[0] == mult * (B - 1)  # the owner's last page (block B - 1)
            print(
                f"[pool far r{ratio} B{B} w{max_ctx} view+{k} next_n {next_n}] {step.pages.numel()} pages, "
                f"last view page {far[0]}, straddling 2^31..2^34: {straddle}, past 2^32: "
                f"{int((named + k * pb >= 2**32).sum())}, past 2^34: {int((named + k * pb >= 2**34).sum())}"
            )
            phys = step.physical(k)
            alloc[phys] = step.page_bytes
            ref, vis = _check(step, step.run_on(view))
            ref32 = ref.float()[vis]
            # wrong control 1: each entry names its neighbouring page (the last page its predecessor)
            bt = step.block_table
            last = view.shape[0] - 1
            shifted = torch.where(bt < 0, bt, torch.where(bt >= last, bt - 1, bt + 1))
            # wrong control 2: a 32-bit byte offset (page * page_bytes mod 2^32)
            wrapped = torch.where(bt < 0, bt, ((bt.long() * pb) % 2**32 // pb).to(torch.int32))
            assert int((wrapped != bt).sum()) > 0
            for tag, table in (("neighbour page", shifted), ("32-bit wrap", wrapped)):
                got = step.run_on(view, block_table=table.contiguous())[vis]
                n_out = int((~((got - ref32).abs() <= 1e-5 + 1.3e-6 * ref32.abs())).sum())
                print(
                    f"[control pool far r{ratio} B{B} w{max_ctx} view+{k} next_n {next_n} {tag}] outside the "
                    f"gate: {n_out} of {got.numel()} visible"
                )
                assert n_out > 0
                with pytest.raises(AssertionError):
                    torch.testing.assert_close(got, ref32)
            alloc[phys] = 0xFF
    del alloc, view
    torch.cuda.empty_cache()


# The PP2 stage-1 pool under CUDA graph: DSpark off at L 1,048,576 (next_n 1, width 1,048,576) is the
# configuration that attains B_PP2_R1_STAGE; DSpark on captures next_n 6 at 1,048,595.
FAR_GRAPH = [(1, 1048576), (6, 1048595)]


@pytest.mark.parametrize("next_n,max_ctx", FAR_GRAPH, ids=[f"n{n}_w{w}" for n, w in FAR_GRAPH])
def test_pool_far_end_capture(next_n, max_ctx):
    """At B_PP2_R1_STAGE ratio-1 blocks (28.25 GB): metadata + logits captured over a batch of 8
    whose request 0 (at the width) names the pool's last page and the pages around byte offsets
    2^31..2^34 first, the rest of its 8193 pages and the other requests' pages drawn from the whole
    pool. The first replay equals the eager warm-up bitwise; 3 more replays each run a new step
    written into the captured buffers (new lengths, queries, weights, table rows and page bytes, the
    previous step's pages back to 0xFF, the output poisoned NaN): exact against the fp64 reference
    and bitwise the documented reduction (_check), bitwise an eager call, and far from the previous
    step's reference."""
    ratio, block_kv, B = 1, 128, B_PP2_R1_STAGE
    alloc = _alloc_index_k(B, ratio)
    far, _ = _far_pages(ratio, B, 0)

    def make(rep):
        seed = zlib.crc32(f"pool-far-graph/{next_n}/{max_ctx}/{rep}".encode())
        pos = _static_positions(8, next_n, ratio, max_ctx, seed)
        gh = torch.Generator()
        gh.manual_seed(seed)

        def pages_of(b, npg):
            rnd = torch.randint(0, B, (npg,), generator=gh).tolist()
            return (far + rnd)[:npg] if b == 0 else rnd

        st = _PoolStep(pos, next_n, ratio, block_kv, pages_of, seed, max_ctx)
        assert set(far) <= set(st.block_table[0].tolist())
        return st

    step = make(0)
    alloc[step.physical(0)] = step.page_bytes
    eager = step.run_on(alloc)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            out = step.run_on(alloc)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    prev = _check(step, out)
    assert torch.equal(out[prev[1]], eager[prev[1]])
    del eager
    cur = step
    for rep in range(1, 4):
        new = make(rep)
        alloc[cur.physical(0)] = 0xFF
        alloc[new.physical(0)] = new.page_bytes
        for dst, src in (
            (step.lens, new.lens),
            (step.q, new.q),
            (step.q_sf, new.q_sf),
            (step.weights, new.weights),
            (step.block_table, new.block_table),
        ):
            dst.copy_(src)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        ref, vis = _check(new, out)
        again = new.run_on(alloc)
        assert torch.equal(out[vis], again[vis])
        del again
        both = vis & prev[1]
        stale = _outside_fraction(out[both], prev[0][both])
        print(
            f"[pool far graph B{B} w{max_ctx} n{next_n} rep{rep}] {new.pages.numel()} pages, visible "
            f"{int(vis.sum())}, previous step's reference: {stale:.4f} of {int(both.sum())} outside the gate"
        )
        assert stale > 0.5
        prev, cur = (ref, vis), new
    del alloc
    torch.cuda.empty_cache()


# ─── mixed steps at both gate widths, the target's view form ─────────────────────────────────
# In a step with tc context tokens (nc context requests) ahead of G generation requests of next_n
# rows, the target hands the op: q / q_sf / weights = split(buf, [tc, gen])[1] of the step's
# token-major [T, ...] buffers (the rows before tc hold the context tokens' own values);
# context_lens = compressed_visible_lens_cuda[r][tc:T] of a max_num_tokens (8192) buffer, viewed
# [G, next_n]; block_table = indexer_compress_block_tables[r][nc:nc+G] of the metadata's
# [max_num_sequences, max_blocks_per_seq] table, 132 columns at max_seq_len 16384. It does this at
# ratio 1 (128-entry pages, static width 16384; layers 20-36) and ratio 2 (64-entry pages, 8192;
# layers 2-18). A CUDA graph is captured only for steps with no context request (the scheduler's
# can_run_cuda_graph is num_context_requests == 0), so the captured form is tc = nc = 0: prefix
# views of the same buffers.
VIEW_WIDTHS = [(1, 128, 16384), (2, 64, 8192)]
VIEW_MAX_TOKENS = 8192
VIEW_TABLE_COLS = 132
VIEW_TABLE_ROWS = 40


def _view_cells(next_n):
    """(tc, nc, G): odd and even tc and nc from 1 up to the largest steps (T 8192 with 32 requests;
    T 8191 with 31 at next_n 6)."""
    return [
        (1, 1, 10),
        (2, 2, 10),
        (573, 3, 10),
        (574, 4, 10),
        (7673, 5, 10),
        (7674, 2, 10),
        (8191 - 2 * next_n, 29, 3 if next_n == 1 else 2),
        (8192 - 2 * next_n, 30, 2),
    ]


VIEW_MIXED = [
    (r, p, w, n, tc, nc, G)
    for r, p, w in VIEW_WIDTHS
    for n in (1, 6)
    for tc, nc, G in _view_cells(n)
]


def _view_full_table(step, g, rows, cols, first):
    """A [rows, cols] metadata table with step.block_table at rows first..first+B: every entry is a
    page of the step's pool (the -1s past each request's pages and the rows of other requests get
    random pool pages, which the kernel must not read)."""
    table = torch.randint(
        0, step.num_pages, (rows, cols), generator=g, device=DEV, dtype=torch.int64
    )
    table = table.to(torch.int32)
    own = table[first : first + step.B, : step.block_table.shape[1]]
    table[first : first + step.B, : step.block_table.shape[1]] = torch.where(
        step.block_table >= 0, step.block_table, own
    )
    return table


def _view_token_buffers(step, tc, g):
    """The step's token-major buffers with tc context rows ahead of the generation rows."""
    rows = step.B * step.n
    ctx_q, ctx_sf = _quant_mxfp4(torch.randn(tc, step.H, DIM, generator=g, device=DEV) * 2.0)
    q = torch.cat([ctx_q, step.q.view(rows, step.H, DIM // 2)])
    q_sf = torch.cat([ctx_sf, step.q_sf.view(rows, step.H)])
    w = torch.cat([(torch.randn(tc, step.H, generator=g, device=DEV) * 2.0**-4), step.weights])
    lens = torch.randint(
        0, step.max_ctx + 1, (VIEW_MAX_TOKENS,), generator=g, device=DEV, dtype=torch.int32
    )
    lens[tc : tc + rows] = step.lens.view(-1)
    return q, q_sf, w, lens


def _view_outside(got, ref):
    """Fraction of elements outside the default fp32 gate, NaN counted as outside."""
    ref32 = ref.float()
    return (~((got - ref32).abs() <= 1e-5 + 1.3e-6 * ref32.abs())).float().mean().item()


def _target_views(step, bufs, table, tc, nc):
    q, q_sf, w, lens = bufs
    rows = step.B * step.n
    return dict(
        q=q.split([tc, rows], 0)[1].reshape(step.B, step.n, step.H, DIM // 2),
        q_sf=q_sf.split([tc, rows], 0)[1].reshape(step.B, step.n, step.H),
        weights=w.split([tc, rows], 0)[1],
        context_lens=lens[tc : tc + rows].reshape(step.B, step.n),
        block_table=table[nc : nc + step.B],
    )


@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx,next_n,tc,nc,G",
    VIEW_MIXED,
    ids=[f"r{r}_w{w}_n{n}_tc{tc}_nc{nc}_G{G}" for r, p, w, n, tc, nc, G in VIEW_MIXED],
)
def test_mixed_step_views(ratio, block_kv, max_ctx, next_n, tc, nc, G):
    """The target's mixed-step call at the ratio's static width over the runtime-sized pool: views
    at token row tc and table row nc, table 132 columns wide (row bases 16-byte aligned) and 133
    (4-byte aligned). Generation request 0 is at the width, request 1 at position 1. Every
    visible element passes the fp64 reference and equals the documented reduction bitwise, the
    inputs are unchanged, and the result is bitwise the contiguous call's. Controls: the query-side
    views one row short of tc, and the table view one row short of nc (row 0 reads the last context
    request's pages), each fail the gate."""
    assert tc + G * next_n <= VIEW_MAX_TOKENS and nc + G <= 32
    seed = zlib.crc32(f"c10-mixed/{ratio}/{max_ctx}/{next_n}/{tc}/{nc}/{G}".encode())
    pos = _static_positions(G, next_n, ratio, max_ctx, seed)
    step = _Step(
        pos,
        next_n,
        ratio,
        block_kv,
        seed=seed,
        max_ctx=max_ctx,
        pool_pages=RUNTIME_POOL_PAGES[block_kv],
    )
    assert int(step.lens.max()) == max_ctx
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    contiguous = step.run()
    ref, vis = _check(step, contiguous)
    bufs = _view_token_buffers(step, tc, g)
    for cols in (VIEW_TABLE_COLS, VIEW_TABLE_COLS + 1):
        table = _view_full_table(step, g, VIEW_TABLE_ROWS, cols, nc)
        views = _target_views(step, bufs, table, tc, nc)
        assert (
            views["block_table"].stride(0) == cols
            and views["q"].storage_offset() == tc * step.H * DIM // 2
        )
        assert views["context_lens"].storage_offset() == tc
        before = [t.clone() for t in (*bufs, table)]
        out = step.run(**views)
        _check(step, out)
        assert torch.equal(out[vis], contiguous[vis])
        assert all(torch.equal(a, b) for a, b in zip(before, (*bufs, table)))
    # Control 1: q / q_sf / weights views one row short (row 0 reads the last context token).
    short = _target_views(step, bufs, table, tc, nc)
    for k, buf in (("q", bufs[0]), ("q_sf", bufs[1]), ("weights", bufs[2])):
        v = buf[tc - 1 : tc - 1 + step.B * step.n]
        short[k] = v.reshape(short[k].shape)
    frac_q = _view_outside(step.run(**short)[vis], ref[vis])
    # Control 2: the table view one row short of nc (each request reads the previous one's pages).
    shifted = _target_views(step, bufs, table, tc, nc)
    shifted["block_table"] = table[nc - 1 : nc - 1 + step.B]
    frac_t = _view_outside(step.run(**shifted)[vis], ref[vis])
    print(
        f"[view mixed r{ratio} w{max_ctx} n{next_n} tc{tc} nc{nc} G{G}] visible {int(vis.sum())}; "
        f"controls outside the gate: q offset lost {frac_q:.4f}, table offset lost {frac_t:.4f}"
    )
    assert frac_q > 0.5 and frac_t > 0.5


VIEW_GRAPH = [(r, p, w, 1, b) for r, p, w in VIEW_WIDTHS for b in CORE_GRAPH_BATCHES] + [
    (r, p, w, 6, b) for r, p, w in VIEW_WIDTHS for b in DSPARK_GRAPH_BATCHES
]


@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx,next_n,batch",
    VIEW_GRAPH,
    ids=[f"r{r}_w{w}_n{n}_G{b}" for r, p, w, n, b in VIEW_GRAPH],
)
def test_cuda_graph_view_form(ratio, block_kv, max_ctx, next_n, batch):
    """The captured form of the same call (tc = nc = 0) at every graph batch of CORE_GRAPH_BATCHES
    (the gates' capture sets): context_lens a prefix view of the 8192-entry lengths buffer, the table rows 0..G-1 of a
    [40, 132] table. Metadata + logits captured after an eager warm-up; the first replay equals
    it bitwise; 3 more replays read new lengths (the cap, 0/1, a padding request), queries,
    weights, keys and re-assigned table rows (rolled by one request), each checked by
    _replay_and_check."""
    seed = zlib.crc32(f"c10-graph/{ratio}/{max_ctx}/{next_n}/{batch}".encode())
    step = _Step(
        [max_ctx * ratio - next_n] * batch,
        next_n,
        ratio,
        block_kv,
        seed=seed,
        max_ctx=max_ctx,
        pool_pages=RUNTIME_POOL_PAGES[block_kv],
    )
    assert int(step.lens.max()) == max_ctx
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    lens_buf = torch.randint(
        0, max_ctx + 1, (VIEW_MAX_TOKENS,), generator=g, device=DEV, dtype=torch.int32
    )
    rows = batch * next_n
    lens_buf[:rows] = step.lens.view(-1)
    table = _view_full_table(step, g, VIEW_TABLE_ROWS, VIEW_TABLE_COLS, 0)
    step.lens = lens_buf[:rows].view(batch, next_n)
    step.block_table = table[:batch]
    eager = step.run()
    torch.cuda.synchronize()
    graph, out = _capture(step)
    graph.replay()
    torch.cuda.synchronize()
    prev = _check(step, out)
    assert torch.equal(out[prev[1]], eager[prev[1]])
    gh = torch.Generator()
    gh.manual_seed(seed)
    for rep in range(3):
        if next_n == 1:
            lens = torch.randint(
                0, max_ctx + 1, (batch, 1), generator=g, device=DEV, dtype=torch.int32
            )
            lens[0] = max_ctx
            if batch > 1:
                lens[-1] = rep % 2
        else:
            lens = _dspark_lens(batch, ratio, max_ctx, rep, gh)
        step.lens.copy_(lens)
        step.block_table.copy_(step.block_table.roll(1, dims=0))
        new = _Step([1] * batch, next_n, ratio, block_kv, seed=seed + rep + 1)
        step.q.copy_(new.q)
        step.q_sf.copy_(new.q_sf)
        step.weights.copy_(new.weights)
        step.refill_keys(g)
        _poison_tails(step)
        step.pool.copy_(_write_pool(step.k_codes, step.k_sf))
        prev = _replay_and_check(
            step, graph, out, f"view graph r{ratio} w{max_ctx} n{next_n} G{batch} rep{rep}", prev
        )


# ─── the DSpark engine's widths and the static-width axis ────────────────────────────────────
# With DSpark on, the runtime raises the engine's max_seq_len M from the user's 16384 to
# 16384 + 5 (overlap scheduler, tokens_per_gen_step - 1) + 4 (get_num_extra_kv_tokens,
# max_draft_len - 1) + 5 (tokens_per_gen_step - 1) = 16398 (py_executor_creator.py). The static
# width max_compressed_kv_lens[r] = (M + max_draft_tokens) // r is then 16398 // r eagerly and
# 16403 // r in the metadata's CUDA-graph copy: 8199 / 8201 at ratio 2, 16398 / 16403 at ratio 1.
# Positions reach M - 1 = 16397, so lengths reach 16398 (r1) / 8199 (r2) and a request's pages
# reach table column 128 of the 132-column tables.
DSPARK_ENGINE_WIDTHS = [(2, 64, 8199), (2, 64, 8201), (1, 128, 16398), (1, 128, 16403)]
DSPARK_EAGER_WIDTHS = [(1, 128, 16398), (2, 64, 8199)]
DSPARK_GRAPH_COPY_WIDTHS = [(1, 128, 16403), (2, 64, 8201)]


@pytest.mark.parametrize("batch", [3, 10, 19, 31])
@pytest.mark.parametrize("next_n", [1, 6], ids=["decode", "dspark_verify"])
@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx",
    DSPARK_ENGINE_WIDTHS,
    ids=[f"r{r}_page{p}_w{w}" for r, p, w in DSPARK_ENGINE_WIDTHS],
)
def test_dspark_engine_width(ratio, block_kv, max_ctx, next_n, batch):
    """Each DSpark engine width over the runtime-sized pool: request 0 exactly at the width (its
    pages reach table column 128), request 1 at position 1, the rest random. Every visible element
    passes the fp64 reference and equals the documented reduction bitwise (_check); the row stride
    is ceil256(width)."""
    seed = zlib.crc32(f"c12-engine/{ratio}/{block_kv}/{max_ctx}/{next_n}/{batch}".encode())
    pos = _static_positions(batch, next_n, ratio, max_ctx, seed)
    step = _Step(
        pos,
        next_n,
        ratio,
        block_kv,
        seed=seed,
        max_ctx=max_ctx,
        pool_pages=RUNTIME_POOL_PAGES[block_kv],
    )
    out = step.run()
    _check(step, out)
    assert int(step.lens.max()) == max_ctx
    assert (int(step.lens[0].max()) - 1) // block_kv == 128  # the last page is table column 128
    assert out.stride(0) == -(-max_ctx // 256) * 256


DSPARK_DECODE_GRAPH = [(r, p, w, b) for r, p, w in DSPARK_ENGINE_WIDTHS for b in CORE_GRAPH_BATCHES]


@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx,batch",
    DSPARK_DECODE_GRAPH,
    ids=[f"r{r}_page{p}_w{w}_B{b}" for r, p, w, b in DSPARK_DECODE_GRAPH],
)
def test_dspark_cuda_graph_decode_width(ratio, block_kv, max_ctx, batch):
    """next_n 1 captured at every decode batch of CORE_GRAPH_BATCHES at each DSpark engine width (the body of
    test_cuda_graph_static_width: first replay = eager warm-up bitwise, 3 replays on new data)."""
    test_cuda_graph_static_width(ratio, block_kv, max_ctx, batch)


DSPARK_MIXED = [
    (r, p, w, n, tc, nc, G)
    for r, p, w in DSPARK_EAGER_WIDTHS
    for n in (1, 6)
    for tc, nc, G in _view_cells(n)
]


@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx,next_n,tc,nc,G",
    DSPARK_MIXED,
    ids=[f"r{r}_w{w}_n{n}_tc{tc}_nc{nc}_G{G}" for r, p, w, n, tc, nc, G in DSPARK_MIXED],
)
def test_dspark_mixed_step_views(ratio, block_kv, max_ctx, next_n, tc, nc, G):
    """Mixed steps are eager, so they carry the eager DSpark widths 16398 (r1) / 8199 (r2): the
    body of test_mixed_step_views (views at token row tc and table row nc of 132- and
    133-column tables, both controls) at those widths."""
    test_mixed_step_views(ratio, block_kv, max_ctx, next_n, tc, nc, G)


DSPARK_VIEW_GRAPH = [
    (r, p, w, 1, b) for r, p, w in DSPARK_GRAPH_COPY_WIDTHS for b in CORE_GRAPH_BATCHES
] + [(r, p, w, 6, b) for r, p, w in DSPARK_GRAPH_COPY_WIDTHS for b in DSPARK_GRAPH_BATCHES]


@pytest.mark.parametrize(
    "ratio,block_kv,max_ctx,next_n,batch",
    DSPARK_VIEW_GRAPH,
    ids=[f"r{r}_w{w}_n{n}_G{b}" for r, p, w, n, b in DSPARK_VIEW_GRAPH],
)
def test_dspark_cuda_graph_view_form(ratio, block_kv, max_ctx, next_n, batch):
    """The captured view form (lengths a prefix of the 8192-entry buffer, table rows of a [40, 132]
    table) at the graph-copy widths 16403 (r1) / 8201 (r2): the body of
    test_cuda_graph_view_form."""
    test_cuda_graph_view_form(ratio, block_kv, max_ctx, next_n, batch)


# The static-width axis. The width W reaches DeepGEMM only as the output allocation
# [rows, ceil256(W)] (csrc/apis/attention.hpp: align(align(W, 256), 1024 B / 4 B)), its slice to W
# and the kernel's runtime logits_stride argument; the JIT template of sm100_paged_mqa_logits
# (csrc/jit_kernels/impls/sm100_mqa_logits.hpp, generate_impl) does not contain it, and the work
# schedule is built from the lengths alone. So a call at width W with lengths <= W must give, on
# every visible element, the bits a call at any larger width gives. The test runs EVERY W from 1
# to the 132-column table's reach (r1: 16640 = ceil256(16403), 130 pages; r2: 8448 = 132 pages of
# 64, = ceil256(8201)) against one checked call at that maximum, and again on the wider tables of
# a max_seq_len 32768 engine up to r1 33024 / r2 16640 (258 / 260 pages; that engine's widths and their
# DSpark forms, (32768 + 14 + {0, 5}) // r, all lie below).
WIDTH_AXIS = [(1, 128, 16640, 132), (2, 64, 8448, 132), (1, 128, 33024, 260), (2, 64, 16640, 260)]


@pytest.mark.parametrize("next_n", [1, 6], ids=["decode", "dspark_verify"])
@pytest.mark.parametrize(
    "ratio,block_kv,w_max,cols",
    WIDTH_AXIS,
    ids=[f"r{r}_page{p}_wmax{w}_cols{c}" for r, p, w, c in WIDTH_AXIS],
)
def test_every_static_width_is_a_prefix(ratio, block_kv, w_max, cols, next_n):
    """32 requests (request 0 at w_max, request 1 at position 1, request 5 a padding request, the
    rest random) on the runtime-sized pool (unowned pages 0xFF) and a 132- (or 260-) column table whose
    unused entries name random pool pages. The w_max call is checked (_check). Then for every W in
    1..w_max, lengths min(len, W) (request 0 always exactly at W), schedule rebuilt: shape
    (rows, W), row stride ceil256(W), and every visible element bitwise equal to the w_max call's.
    Control: the same output read at row stride W (a consumer ignoring the padding) differs."""
    seed = zlib.crc32(f"c12-axis/{ratio}/{block_kv}/{w_max}/{next_n}".encode())
    pos = _static_positions(32, next_n, ratio, w_max, seed)
    pos[5] = None
    step = _Step(
        pos,
        next_n,
        ratio,
        block_kv,
        seed=seed,
        max_ctx=w_max,
        pool_pages=RUNTIME_POOL_PAGES[block_kv],
    )
    assert int(step.lens.max()) == w_max
    assert bool((step.block_table[:, cols:] == -1).all())
    step.block_table = step.block_table[:, :cols].contiguous()
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    step.block_table = _view_full_table(step, g, step.B, cols, 0)
    assert step.block_table.shape == (32, cols)
    full = step.run()
    _check(step, full)
    full_bits = full.view(torch.int32)
    rows = step.B * step.n
    lens_max = step.lens.clone()
    col = torch.arange(w_max, device=DEV, dtype=torch.int32)
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    seen = torch.zeros((), dtype=torch.int64, device=DEV)
    ctl_rows = {}
    slowest = 0.0
    t0 = time.time()
    for W in range(1, w_max + 1):
        lens = lens_max.clamp(max=W)
        out, call_s = _timed(step.run, context_lens=lens, max_context_len=W)
        slowest = max(slowest, call_s)
        assert out.shape == (rows, W) and out.stride() == (_aligned_width(W, torch.float32), 1)
        vis = col[:W] < lens.view(-1, 1)
        bad += ((out.view(torch.int32) != full_bits[:, :W]) & vis).sum()
        seen += vis.sum()
        if W in (8199, 16398, 32787) or (W == w_max - 1 and W % 256):
            # control: the buffer read at row stride W, as if it were contiguous
            wrong = torch.as_strided(out, (rows, W), (W, 1))
            ctl_rows[W] = (((wrong.view(torch.int32) != full_bits[:, :W]) & vis).sum(1) > 0)[
                step.n :
            ]
    torch.cuda.synchronize()
    dt = time.time() - t0
    L = lens_max.view(-1).long()
    expect = int((L * (L + 1) // 2 + L * (w_max - L)).sum())
    print(
        f"[static-width axis r{ratio} wmax{w_max} cols{cols} n{next_n}] {w_max} widths in {dt:.1f}s, visible elements "
        f"compared {int(seen)}, mismatches {int(bad)}; slowest call {slowest * 1e3:.1f} ms"
    )
    assert int(seen) == expect and int(bad) == 0
    assert slowest < JIT_BOUND_S
    for W, r in ctl_rows.items():
        live = int((lens_max.view(-1)[step.n :] > 0).sum())
        print(
            f"[static-width axis control W {W}] rows read at stride W that differ: {int(r.sum())} of {live} live rows"
        )
        assert int(r.sum()) >= live // 2


# ─── the engine envelope: G <= 256, max_seq_len up to 1,048,576 ──────────────────────────────
# Graph batches: every G in 1..32 plus TRT-LLM's padded list at cuda_graph_config
# max_batch_size 256 (llm_args.py, _generate_cuda_graph_batch_sizes: 1, 2, 4, 8, 16..128 step 8,
# 192, 256); eager steps hold any G in 1..256 generation requests. Per-request length: the user's
# max_seq_len L in ENV_L. With DSpark on, the engine's max_seq_len is Lm = L + 14 and the
# metadata's CUDA-graph copy adds max_draft_tokens 5, so the static widths
# max_compressed_kv_lens[r] are Lm // r (eager) and (Lm + 5) // r (graph); with DSpark off both are
# L // r. The index-K block tables have ceil(Lm / 128) + 3 columns (132 at Lm 16398, 8196 at
# 1048590) and the lengths buffer has max_num_tokens = 16384 entries.
ENV_GSET = list(range(1, 33)) + [40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256]
ENV_L = [16384, 32768, 65536, 131072, 262144, 1048576]
ENV_MAX_NUM_TOKENS = 16384
ENV_TABLE_ROWS = 264  # max_num_sequences >= max_batch_size 256
ENV_PAGE = {1: 128, 2: 64}  # index-K entries per page at tokens_per_block 128
# The reference works on chunks of at most this many (row, column) elements (1 GiB of fp64 dots
# per 4M); the bitwise-only comparisons use the larger budget.
ENV_REF_BUDGET = 1 << 22
ENV_CMP_BUDGET = 1 << 26


def _env_widths():
    """(kind, L, ratio, W, table columns) of every static width the envelope's engines pass."""
    out = []
    for L in ENV_L:
        lm = L + 14
        for r in (1, 2):
            out.append(("on_eager", L, r, lm // r, -(-lm // 128) + 3))
            out.append(("on_graph", L, r, (lm + 5) // r, -(-lm // 128) + 3))
            out.append(("off", L, r, L // r, -(-L // 128) + 3))
    return out


ENV_WIDTHS = _env_widths()

_E2M1_X2 = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float32, device=DEV
)


def _codes_x2(packed: torch.Tensor) -> torch.Tensor:
    """packed E2M1 [..., 64] -> fp32 [..., 128]: twice each element's value, an integer in [-12, 12]."""
    p = packed.view(torch.uint8).int()
    return torch.stack([_E2M1_X2[p & 0xF], _E2M1_X2[p >> 4]], dim=-1).flatten(-2)


def _exp2(e: torch.Tensor) -> torch.Tensor:
    """2^e in fp64 from its bit pattern (exact)."""
    return ((e.long() + 1023) << 52).view(torch.float64)


def _exact_dots(q, q_sf, codes, sf_bytes):
    """Exact per-head q.k in fp64: q int8 [Ba, n, H, 64] + q_sf int32 [Ba, n, H]; keys as their pool
    bytes, codes uint8 [Ba, C, 64] + scale bytes uint8 [Ba, C, 4] -> [Ba, n, H, C].

    Per 32-element block the products of the doubled E2M1 codes sum to an integer of magnitude at
    most 32 * 144 = 4608, which an fp32 GEMM forms exactly in any order; each block's sum times its
    two power-of-two scales (2^(bq - 127) * 2^(bk - 127) / 4) is exact in fp64, and so is the sum of
    the four blocks. test_env_exact_dots_equal_the_decode_reference pins this against _decode."""
    Ba, n, H, _ = q.shape
    C = codes.shape[1]
    qx = _codes_x2(q).view(Ba, n * H, 4, 32)
    qe = q_sf.contiguous().view(torch.uint8).view(Ba, n * H, 4).int()
    kx = _codes_x2(codes).view(Ba, C, 4, 32)
    ke = sf_bytes.int()
    out = None
    for blk in range(4):
        s = torch.bmm(qx[:, :, blk], kx[:, :, blk].transpose(1, 2))
        term = s.double() * _exp2(qe[:, :, blk, None] + ke[:, None, :, blk] - 256)
        out = term if out is None else out + term
    return out.view(Ba, n, H, C)


def _emulate_heads(dots, w):
    """The kernel's documented fp32 head reduction over exact dots [Ba, n, H, C] and fp64 weights
    [Ba, n, H] (the same recipe as _Step.emulate)."""
    Ba, n, H, C = dots.shape
    a = dots.float()
    two = (a + a.abs()).double()
    chains = []
    for lane in range(4):
        acc = torch.zeros(Ba, n, C, device=DEV)
        for h in range(lane, H, 4):
            acc = (two[:, :, h] * w[:, :, h, None] + acc.double()).float()
        chains.append(acc)
    return ((chains[0] + chains[2]) + (chains[1] + chains[3])) / 2


def _env_chunks(lens, budget):
    """(requests, c0, c1) tiles covering every visible element of a [G, next_n] step exactly once:
    the requests whose last row reaches column c0, over a column range sized to the budget."""
    G, n = lens.shape
    req_len = lens.max(1).values.cpu()
    maxlen = int(req_len.max()) if G else 0
    c0 = 0
    while c0 < maxlen:
        bs = torch.nonzero(req_len > c0).flatten()
        C = max(SPLIT_KV, budget // (bs.numel() * n) // SPLIT_KV * SPLIT_KV)
        c1 = min(c0 + C, maxlen)
        yield (None if bs.numel() == G else bs.to(DEV)), c0, c1
        c0 = c1


def _tile(x3, bs, c0, c1):
    t = x3[:, :, c0:c1]
    return t if bs is None else t[bs]


def _outside(got, ref):
    """Elements outside the default fp32 assert_close gate; NaN counts as outside."""
    return int((~((got - ref).abs() <= 1e-5 + 1.3e-6 * ref.abs())).sum())


def _env_compare(out, q, q_sf, weights, lens, table, pool, W, same=(), prev=None):
    """Every visible element (row (b, j), column c < lens[b, j]) of ``out`` against the exact
    per-head dots of the keys ``table`` names in ``pool``: outside the default fp32 gate of the fp64
    score, and not bitwise equal to the documented fp32 reduction. ``same``: outputs that must equal
    ``out`` bitwise on every visible element. ``prev``: (output, lens) of the previous step, compared
    through the same gate on the elements both steps see. Works on request x column tiles, so a
    [1536, 1048595] output never needs a dense fp64 copy. Returns the counts."""
    G, n = lens.shape
    H = q.shape[2]
    P, bk = pool.shape[0], pool.shape[1]
    flat = pool.view(P, bk * KEY_BYTES)
    assert out.shape == (G * n, W) and out.dtype == torch.float32
    assert out.stride() == (_aligned_width(W, torch.float32), 1)
    out3 = out.view(G, n, W)
    w3 = weights.view(G, n, H).double()
    lens64 = lens.long()
    cnt = dict(seen=0, outside=0, not_emu=0, not_same=0, stale_shared=0, stale_outside=0)
    for bs, c0, c1 in _env_chunks(lens, ENV_REF_BUDGET):
        idx = slice(None) if bs is None else bs
        p0 = c0 // bk
        pages = table[idx, p0 : -(-c1 // bk)].long().clamp(0, P - 1)
        Ba = pages.shape[0]
        raw = flat[pages]
        o = c0 - p0 * bk
        codes = raw[..., : bk * 64].reshape(Ba, -1, 64)[:, o : o + c1 - c0]
        sfb = raw[..., bk * 64 :].reshape(Ba, -1, 4)[:, o : o + c1 - c0]
        dots = _exact_dots(q[idx], q_sf[idx], codes, sfb)
        wb = w3[idx]
        ref = (dots.clamp_min(0) * wb[..., None]).sum(2).float()
        emu = _emulate_heads(dots, wb)
        del dots
        vis = torch.arange(c0, c1, device=DEV) < lens64[idx][..., None]
        got = _tile(out3, bs, c0, c1)[vis]
        cnt["seen"] += got.numel()
        cnt["outside"] += _outside(got, ref[vis])
        cnt["not_emu"] += int((got.view(torch.int32) != emu[vis].view(torch.int32)).sum())
        for s in same:
            sv = _tile(s.view(G, n, W), bs, c0, c1)[vis]
            cnt["not_same"] += int((sv.view(torch.int32) != got.view(torch.int32)).sum())
        if prev is not None:
            pout, plens = prev
            both = (
                torch.arange(c0, c1, device=DEV)
                < torch.minimum(lens64, plens.long())[idx][..., None]
            )
            cnt["stale_shared"] += int(both.sum())
            cnt["stale_outside"] += _outside(
                _tile(out3, bs, c0, c1)[both], _tile(pout.view(G, n, W), bs, c0, c1)[both]
            )
    return cnt


def _env_assert_exact(cnt, lens, tag):
    assert cnt["seen"] == int(lens.long().sum()), f"{tag}: visible elements {cnt}"
    assert cnt["outside"] == 0 and cnt["not_emu"] == 0 and cnt["not_same"] == 0, f"{tag}: {cnt}"


def _env_visible_mismatch(a, b, lens):
    """(mismatches, compared): ``a`` against ``b`` bitwise on every element visible under ``lens``
    (the two may have different widths; every visible column lies below both)."""
    G, n = lens.shape
    a3, b3 = a.view(G, n, a.shape[1]), b.view(G, n, b.shape[1])
    lens64 = lens.long()
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    seen = torch.zeros((), dtype=torch.int64, device=DEV)
    for bs, c0, c1 in _env_chunks(lens, ENV_CMP_BUDGET):
        idx = slice(None) if bs is None else bs
        vis = torch.arange(c0, c1, device=DEV) < lens64[idx][..., None]
        diff = _tile(a3, bs, c0, c1).view(torch.int32) != _tile(b3, bs, c0, c1).view(torch.int32)
        bad += (diff & vis).sum()
        seen += vis.sum()
    return int(bad), int(seen)


# The one-compiled-variant probe. DeepGEMM builds a kernel the first time a process launches a new
# template instantiation, and a build of either kernel took 1.18 s (metadata) and 2.1 s (logits)
# here; a launch of a built kernel takes microseconds. So a call at a new batch or width that
# returns within JIT_BOUND_S (synchronized wall time, output allocation included) built nothing. The
# on-disk cache cannot serve as the probe: it is shared with every process on the node, and its
# listing was seen missing an existing entry while other processes renamed new ones in.
JIT_BOUND_S = 0.5
_WARM_FAMILIES = set()


def _reserve(rows, W):
    """Leave a free block of exactly a [rows, W] fp32 output's size in the caching allocator, so the
    timed call that follows reuses it instead of timing a device allocation."""
    torch.empty(rows * _aligned_width(W, torch.float32), device=DEV)


def _timed(fn, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(**kw)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def _warm_family(next_n, block_kv):
    """Build (or load) the (next_n, page) variants of both kernels once per process, at a one-request
    256-column call, so a later timed call measures only what its own shape adds."""
    if (next_n, block_kv) in _WARM_FAMILIES:
        return
    lens = torch.ones(1, next_n, dtype=torch.int32, device=DEV)
    q = torch.zeros(1, next_n, HEADS, DIM // 2, dtype=torch.int8, device=DEV)
    q_sf = torch.full((1, next_n, HEADS), 0x7F7F7F7F, dtype=torch.int32, device=DEV)
    pool = torch.zeros(2, block_kv, 1, KEY_BYTES, dtype=torch.uint8, device=DEV)
    w = torch.zeros(next_n, HEADS, device=DEV)
    table = torch.zeros(1, 4, dtype=torch.int32, device=DEV)
    meta = metadata_op(lens, block_kv)
    op(q, q_sf, pool, w, lens, table, meta, SPLIT_KV)
    torch.cuda.synchronize()
    _WARM_FAMILIES.add((next_n, block_kv))


class _EnvStep:
    """A generation step in the target's own buffer layout at an envelope shape.

    q / q_sf / weights are the rows after ``tc`` context tokens of token-major buffers,
    context_lens is ``lens_buf[tc : tc + G * next_n]`` of a max_num_tokens lengths buffer, and the
    block table is rows ``nc .. nc + G`` of an [ENV_TABLE_ROWS, cols] table (tc = nc = 0 is the
    captured form). ``set_step`` gives every request fresh pages drawn at random from the whole
    ``pool_pages`` pool (its last page always among them) with fresh keys, and fresh queries and
    weights; every page no request owns holds 0xFF bytes (NaN scale), the slots past each request's
    length in its last page are NaN-poisoned, and every table entry past a request's pages names a
    random pool page. The buffers are static, so a captured graph reads each step's values."""

    def __init__(self, G, next_n, ratio, W, cols, pool_pages, seed, tc=0, nc=0):
        self.G, self.n, self.ratio, self.bk, self.W, self.cols = (
            G,
            next_n,
            ratio,
            ENV_PAGE[ratio],
            W,
            cols,
        )
        self.g = torch.Generator(device=DEV)
        self.g.manual_seed(seed)
        self.gh = torch.Generator()
        self.gh.manual_seed(seed)
        self.pool = torch.full(
            (pool_pages, self.bk, 1, KEY_BYTES), 0xFF, dtype=torch.uint8, device=DEV
        )
        self.flat = self.pool.view(pool_pages, self.bk * KEY_BYTES)
        self.table_buf = torch.randint(
            0, pool_pages, (ENV_TABLE_ROWS, cols), generator=self.g, device=DEV, dtype=torch.int32
        )
        self.lens_buf = torch.randint(
            0, W + 1, (ENV_MAX_NUM_TOKENS,), generator=self.g, device=DEV, dtype=torch.int32
        )
        rows = G * next_n
        assert tc + rows <= ENV_MAX_NUM_TOKENS and nc + G <= ENV_TABLE_ROWS
        cq, csf = _quant_mxfp4(
            torch.randn(tc + rows, HEADS, DIM, generator=self.g, device=DEV) * 2.0
        )
        self.q_buf, self.q_sf_buf = cq, csf
        self.w_buf = (
            (torch.randn(tc + rows, HEADS, generator=self.g, device=DEV) * 2.0**-4)
            .to(torch.bfloat16)
            .float()
        )
        self.tc, self.nc = tc, nc
        self.q = self.q_buf[tc:].view(G, next_n, HEADS, DIM // 2)
        self.q_sf = self.q_sf_buf[tc:].view(G, next_n, HEADS)
        self.weights = self.w_buf[tc:]
        self.lens = self.lens_buf[tc : tc + rows].view(G, next_n)
        self.block_table = self.table_buf[nc : nc + G]
        self.owned = None

    def positions(self, rep, far=True):
        """First-query positions: request 0 at the cap (its last row sees exactly W entries),
        request 1 at position rep % 2, on even reps a padding request last, one request in the upper
        half of the range, the rest at gate-like contexts below min(cap, 24576)."""
        G, cap = self.G, self.W * self.ratio - self.n
        pos = torch.randint(0, min(cap, 24576) + 1, (G,), generator=self.gh).tolist()
        pos[0] = cap
        if G > 1:
            pos[1] = rep % 2
        if G > 2 and rep % 2 == 0:
            pos[-1] = None
        if far and G > 3:
            pos[G // 2] = int(torch.randint(cap // 2, cap + 1, (1,), generator=self.gh))
        return pos

    def lens_of(self, positions):
        n, r = self.n, self.ratio
        return [
            [0] * n if p is None else [max(p + j + 1, 0) // r for j in range(n)] for p in positions
        ]

    def set_step(self, positions):
        n, bk, P = self.n, self.bk, self.pool.shape[0]
        if self.owned is not None:
            self.flat[self.owned] = 0xFF
        lens = self.lens_of(positions)
        assert max(max(row) for row in lens) <= self.W
        self.lens.copy_(torch.tensor(lens, dtype=torch.int32))
        npg = [-(-max(row) // bk) for row in lens]
        total = sum(npg)
        assert total <= P
        perm = torch.randperm(P, generator=self.g, device=DEV)[:total]
        if total and not bool((perm == P - 1).any()):
            perm[-1] = P - 1
        self.block_table.copy_(
            torch.randint(
                0, P, (self.G, self.cols), generator=self.g, device=DEV, dtype=torch.int32
            )
        )
        off = 0
        for b, k in enumerate(npg):
            self.block_table[b, :k] = perm[off : off + k].to(torch.int32)
            off += k
        for c in range(0, total, 4096):
            pg = perm[c : c + 4096]
            k, ks = _quant_mxfp4(torch.randn(pg.numel(), bk, DIM, generator=self.g, device=DEV))
            self.flat[pg] = _write_pool(k, ks).view(pg.numel(), bk * KEY_BYTES)
        for b, row in enumerate(lens):
            L = max(row)
            if L % bk:
                self.flat[self.block_table[b, L // bk].long(), bk * 64 + (L % bk) * 4 :] = 0xFF
        self.owned = perm
        q, qs = _quant_mxfp4(torch.randn(self.G, n, HEADS, DIM, generator=self.g, device=DEV) * 2.0)
        self.q.copy_(q)
        self.q_sf.copy_(qs)
        w = torch.randn(self.G * n, HEADS, generator=self.g, device=DEV) * 2.0**-4
        self.weights.copy_(w.to(torch.bfloat16).float())

    def run(self, **over):
        a = dict(
            q=self.q,
            q_sf=self.q_sf,
            kv_cache=self.pool,
            weights=self.weights,
            context_lens=self.lens,
            block_table=self.block_table,
            max_context_len=self.W,
        )
        a.update(over)
        meta = metadata_op(a["context_lens"], self.bk)
        return op(schedule_meta=meta, **a)

    def compare(self, out, **kw):
        return _env_compare(
            out,
            self.q,
            self.q_sf,
            self.weights,
            self.lens,
            self.block_table,
            self.pool,
            self.W,
            **kw,
        )


def test_env_exact_dots_equal_the_decode_reference():
    """_exact_dots (integer-code GEMMs per 32-block, fp64 scaling) is bitwise the fp64 einsum of the
    decoded values the rest of this file uses, with keys over a 2^-8..2^8 magnitude range."""
    g = torch.Generator(device=DEV)
    g.manual_seed(7)
    q, qs = _quant_mxfp4(torch.randn(3, 6, HEADS, DIM, generator=g, device=DEV) * 2.0)
    k, ks = _quant_mxfp4(
        torch.randn(3, 5000, DIM, generator=g, device=DEV)
        * torch.logspace(-8, 8, 5000, base=2.0, device=DEV)[:, None]
    )
    ref = torch.einsum("bjhd,bsd->bjhs", _decode(q, qs), _decode(k, ks))
    got = _exact_dots(
        q, qs, k.view(torch.uint8), ks.contiguous().view(torch.uint8).view(3, 5000, 4)
    )
    assert torch.equal(got, ref)


def test_env_checker_agrees_with_check_and_sees_wrong_tables():
    """_env_compare on a _Step case that _check passes: every visible element counted, none outside,
    all bitwise equal to the emulation. With the block-table rows rolled by one request (a wrong
    output) it reports almost every visible element outside the gate."""
    step = _Step(
        [130, 999, 4100, 20001, 777, 3, 12000, 64],
        6,
        2,
        64,
        seed=21,
        max_ctx=16393,
        pool_pages=RUNTIME_POOL_PAGES[64],
    )
    out = step.run()
    _check(step, out)
    args = (step.q, step.q_sf, step.weights, step.lens, step.block_table, step.pool, step.max_ctx)
    cnt = _env_compare(out, *args)
    _env_assert_exact(cnt, step.lens, "env checker")
    wrong = step.run(block_table=step.block_table.roll(1, dims=0).contiguous())
    bad = _env_compare(wrong, *args)
    print(
        f"[env checker control] rolled table rows: {bad['outside']} of {bad['seen']} outside the gate"
    )
    assert bad["outside"] > bad["seen"] // 2


# ─── capture over the envelope: every graph batch at every width ─────────────────────────────
# Captured widths are the graph copies: DSpark on, (Lm + 5) // r at next_n 6 (verify) and next_n 1;
# DSpark off, L // r at next_n 1. Each at every G in ENV_GSET, in the captured view form (tc = nc = 0:
# lengths a prefix of the 16384-entry buffer, table rows 0 .. G of the [264, cols] table), over the
# runtime-sized pool.
ENV_GRAPH = [
    (L, r, W, c, n)
    for kind, L, r, W, c in ENV_WIDTHS
    for n in ((6, 1) if kind == "on_graph" else (1,) if kind == "off" else ())
]
ENV_GRAPH_CASES = [(L, r, W, c, n, G) for L, r, W, c, n in ENV_GRAPH for G in ENV_GSET]


@pytest.mark.parametrize(
    "L,ratio,W,cols,next_n,G",
    ENV_GRAPH_CASES,
    ids=[f"L{L}_r{r}_w{W}_c{c}_n{n}_G{G}" for L, r, W, c, n, G in ENV_GRAPH_CASES],
)
def test_env_cuda_graph(L, ratio, W, cols, next_n, G):
    """Metadata + logits captured after an eager warm-up; the first replay equals it bitwise; then 3
    replays, each on a new step written into the captured buffers (lengths with request 0 at the
    width, 0/1-long rows, a padding request on even steps; fresh pages, keys, table rows, queries
    and weights; the output poisoned NaN): exact against the fp64 score and bitwise the documented
    reduction on every visible element, bitwise equal to an eager call on the same inputs, and far
    from the previous step's result (a replay that re-read the captured step is seen). The
    eager warm-up at the new (G, W) returns within JIT_BOUND_S (no kernel build)."""
    seed = zlib.crc32(f"env-graph/{L}/{ratio}/{W}/{next_n}/{G}".encode())
    step = _EnvStep(G, next_n, ratio, W, cols, RUNTIME_POOL_PAGES[ENV_PAGE[ratio]], seed)
    step.set_step(step.positions(0))
    _warm_family(next_n, step.bk)
    eager, dt = _timed(step.run)
    assert dt < JIT_BOUND_S, f"eager call at a new (G, W) took {dt:.3f}s"
    graph, out = _capture(step)
    graph.replay()
    torch.cuda.synchronize()
    _env_assert_exact(step.compare(out, same=(eager,)), step.lens, "first replay")
    del eager
    prev = (out.clone(), step.lens.clone())
    for rep in range(1, 4):
        step.set_step(step.positions(rep))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        again = step.run()
        cnt = step.compare(out, same=(again,), prev=prev)
        _env_assert_exact(cnt, step.lens, f"replay {rep}")
        stale = cnt["stale_outside"] / max(cnt["stale_shared"], 1)
        print(
            f"[env graph L{L} r{ratio} w{W} n{next_n} G{G} rep{rep}] visible {cnt['seen']}, previous "
            f"step's result: {stale:.4f} of {cnt['stale_shared']} outside the gate; eager warm-up {dt * 1e3:.1f} ms"
        )
        assert stale > 0.5
        prev = (out.clone(), step.lens.clone())
        del again


# ─── eager: every batch 1..256 at every width ────────────────────────────────────────────────
# Rows are independent, so a batch is a prefix of a larger one: one 256-request step per (width,
# next_n) is checked exactly, and every B in 1..255 (prefix views of the same buffers, schedule
# rebuilt) must equal its first B * next_n rows bitwise on every visible element. next_n 1 at all
# 36 widths; next_n 6 at the 24 DSpark ones (eager and graph copy).
ENV_EAGER = [
    (L, r, W, c, n)
    for kind, L, r, W, c in ENV_WIDTHS
    for n in ((1, 6) if kind.startswith("on") else (1,))
]


@pytest.mark.parametrize(
    "L,ratio,W,cols,next_n",
    ENV_EAGER,
    ids=[f"L{L}_r{r}_w{W}_c{c}_n{n}" for L, r, W, c, n in ENV_EAGER],
)
def test_env_every_batch_is_a_prefix(L, ratio, W, cols, next_n):
    seed = zlib.crc32(f"env-prefix/{L}/{ratio}/{W}/{next_n}".encode())
    step = _EnvStep(256, next_n, ratio, W, cols, RUNTIME_POOL_PAGES[ENV_PAGE[ratio]], seed)
    pos = step.positions(1)
    pos[5] = None  # a padding request inside the batch
    step.set_step(pos)
    full = step.run()
    _env_assert_exact(step.compare(full), step.lens, "256-request step")
    assert int(step.lens.max()) == W
    bad = seen = 0
    slowest, slowest_b = 0.0, 0
    for B in range(1, 256):
        rows = B * next_n
        _reserve(rows, W)
        out, dt = _timed(
            step.run,
            q=step.q[:B],
            q_sf=step.q_sf[:B],
            weights=step.weights[:rows],
            context_lens=step.lens[:B],
            block_table=step.block_table[:B],
        )
        if dt > slowest:
            slowest, slowest_b = dt, B
        assert out.shape == (rows, W) and out.stride() == (_aligned_width(W, torch.float32), 1)
        m, s = _env_visible_mismatch(out, full[:rows], step.lens[:B])
        bad, seen = bad + m, seen + s
    print(
        f"[env prefix L{L} r{ratio} w{W} c{cols} n{next_n}] B 1..255: {seen} visible elements compared, "
        f"{bad} mismatches; slowest call {slowest * 1e3:.1f} ms (B {slowest_b})"
    )
    assert bad == 0
    assert slowest < JIT_BOUND_S


# ─── eager mixed steps at the envelope (tokens x batch x width) ──────────────────────────────
# The target's mixed-step views at T <= 16384 and nc + G <= 256, at the eager widths of L 65536 and
# 1048576 (8196-column tables at Lm 1048590): (tc, nc, G) from 1 context token and 255 generation
# requests up to T 16384 and 255 context requests.
def _env_mixed_cells(n):
    return [
        (1, 1, 255),
        (ENV_MAX_NUM_TOKENS - 255 * n, 1, 255),
        (8192, 128, 128),
        (ENV_MAX_NUM_TOKENS - n, 255, 1),
    ]


ENV_MIXED = [
    (L, r, W, c, n, tc, nc, G)
    for kind, L, r, W, c in ENV_WIDTHS
    if L in (65536, 1048576) and kind in ("on_eager", "off")
    for n in ((1, 6) if kind == "on_eager" else (1,))
    for tc, nc, G in _env_mixed_cells(n)
]


@pytest.mark.parametrize(
    "L,ratio,W,cols,next_n,tc,nc,G",
    ENV_MIXED,
    ids=[f"L{L}_r{r}_w{W}_c{c}_n{n}_tc{tc}_nc{nc}_G{G}" for L, r, W, c, n, tc, nc, G in ENV_MIXED],
)
def test_env_mixed_step_views(L, ratio, W, cols, next_n, tc, nc, G):
    """The views at token row tc and table row nc (tables cols and cols + 1 wide): exact on every
    visible element, bitwise the contiguous call, inputs unchanged. Controls: the query-side views
    one row short of tc and the table view one row short of nc each fail the gate."""
    seed = zlib.crc32(f"env-mixed/{L}/{ratio}/{W}/{next_n}/{tc}/{nc}/{G}".encode())
    step = _EnvStep(
        G, next_n, ratio, W, cols, RUNTIME_POOL_PAGES[ENV_PAGE[ratio]], seed, tc=tc, nc=nc
    )
    step.set_step(step.positions(1))
    assert int(step.lens.max()) == W
    rows = G * next_n
    contiguous = step.run(
        q=step.q.clone(),
        q_sf=step.q_sf.clone(),
        weights=step.weights.clone(),
        context_lens=step.lens.clone(),
        block_table=step.block_table.clone(),
    )
    _env_assert_exact(step.compare(contiguous), step.lens, "contiguous")
    wide = torch.randint(
        0,
        step.pool.shape[0],
        (ENV_TABLE_ROWS, cols + 1),
        generator=step.g,
        device=DEV,
        dtype=torch.int32,
    )
    wide[:, :cols] = step.table_buf
    for table in (step.table_buf, wide):
        view = table[nc : nc + G]
        assert view.stride(0) == table.shape[1] and step.q.storage_offset() == tc * HEADS * DIM // 2
        before = [t.clone() for t in (step.q_buf, step.q_sf_buf, step.w_buf, step.lens_buf, table)]
        out = step.run(block_table=view)
        _env_assert_exact(
            step.compare(out, same=(contiguous,)), step.lens, f"views, {table.shape[1]} columns"
        )
        assert all(
            torch.equal(a, b)
            for a, b in zip(before, (step.q_buf, step.q_sf_buf, step.w_buf, step.lens_buf, table))
        )
    short = step.run(
        q=step.q_buf[tc - 1 : tc - 1 + rows].view(G, next_n, HEADS, DIM // 2),
        q_sf=step.q_sf_buf[tc - 1 : tc - 1 + rows].view(G, next_n, HEADS),
        weights=step.w_buf[tc - 1 : tc - 1 + rows],
    )
    shifted = step.run(block_table=step.table_buf[nc - 1 : nc - 1 + G])
    c_q, c_t = step.compare(short), step.compare(shifted)
    frac_q, frac_t = c_q["outside"] / c_q["seen"], c_t["outside"] / c_t["seen"]
    print(
        f"[env mixed L{L} r{ratio} w{W} n{next_n} tc{tc} nc{nc} G{G}] visible {c_q['seen']}; controls outside "
        f"the gate: q offset lost {frac_q:.4f}, table offset lost {frac_t:.4f}"
    )
    assert frac_q > 0.5 and frac_t > 0.5


# ─── the memory edge: every request of G 256 at the largest widths ───────────────────────────
# All 256 requests' rows at the far end: at next_n 6 and W 1,048,595 the output is [1536, 1048832]
# fp32 (6.44 GB, element offsets up to 1.61e9 and byte offsets past 2^32), over a pool of
# 256 x 8193 pages (ratio 1: 2,097,416 blocks, 18.3 GB; below the ratio-1 bound B_BOUND[1]).
ENV_EDGE = [
    (1, 1048595, 8196, 6),
    (2, 524297, 8196, 6),
    (1, 1048576, 8195, 1),
    (2, 524288, 8195, 1),
]


@pytest.mark.parametrize(
    "ratio,W,cols,next_n", ENV_EDGE, ids=[f"r{r}_w{W}_c{c}_n{n}" for r, W, c, n in ENV_EDGE]
)
def test_env_edge_every_request_at_the_width(ratio, W, cols, next_n):
    """Eager step checked exactly; captured after it (first replay bitwise the eager call); then 2
    replays on new steps (every request within 2048 positions of the cap, fresh pages, keys, table
    rows, queries, weights), each exact, bitwise an eager call, and far from the previous result."""
    G, bk = 256, ENV_PAGE[ratio]
    cap = W * ratio - next_n
    pool_pages = G * -(-W // bk) + 8
    seed = zlib.crc32(f"env-edge/{ratio}/{W}/{next_n}".encode())
    step = _EnvStep(G, next_n, ratio, W, cols, pool_pages, seed)
    step.set_step([cap] * G)
    eager = step.run()
    cnt = step.compare(eager)
    _env_assert_exact(cnt, step.lens, "eager")
    assert int(step.lens[:, -1].min()) == W
    print(
        f"[env edge r{ratio} w{W} n{next_n} eager] {cnt['seen']} visible elements exact; output "
        f"{tuple(eager.shape)} row stride {eager.stride(0)}; pool {pool_pages} pages"
    )
    graph, out = _capture(step)
    graph.replay()
    torch.cuda.synchronize()
    m, s = _env_visible_mismatch(out, eager, step.lens)
    assert m == 0 and s == int(step.lens.long().sum())
    del eager
    prev = (out.clone(), step.lens.clone())
    for rep in range(1, 3):
        step.set_step(torch.randint(cap - 2048, cap + 1, (G,), generator=step.gh).tolist())
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        again = step.run()
        cnt = step.compare(out, same=(again,), prev=prev)
        _env_assert_exact(cnt, step.lens, f"edge replay {rep}")
        stale = cnt["stale_outside"] / max(cnt["stale_shared"], 1)
        print(
            f"[env edge r{ratio} w{W} n{next_n} rep{rep}] visible {cnt['seen']}, previous step's result: "
            f"{stale:.4f} of {cnt['stale_shared']} outside the gate; peak "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
        )
        assert stale > 0.5
        prev = (out.clone(), step.lens.clone())
        del again


# ─── the length axis up to the largest widths ────────────────────────────────────────────────
# A row's value at column c depends on its query, weights and the key at c, not on its length, so a
# row of length t is the first t columns of the same row at full length. One 256-request step with
# every row at the width is checked exactly; then the lengths alone change: at next_n 1 EVERY length
# 1..W is some request's length in exactly one call; at next_n 6 every last-row length 1..256 and
# every one within 1 of a multiple of 64 (the 64- and 128-entry pages and the 256-column splits),
# rows (p + j + 1) // r. Each call's visible elements must equal the full step's bitwise.
ENV_LENGTH_AXIS = [(1, 1048595, 8196), (2, 524297, 8196)]


@pytest.mark.parametrize("next_n", [1, 6], ids=["decode", "dspark_verify"])
@pytest.mark.parametrize(
    "ratio,W,cols", ENV_LENGTH_AXIS, ids=[f"r{r}_w{W}_c{c}" for r, W, c in ENV_LENGTH_AXIS]
)
def test_env_every_length(ratio, W, cols, next_n):
    G, bk = 256, ENV_PAGE[ratio]
    cap = W * ratio - next_n
    seed = zlib.crc32(f"env-length/{ratio}/{W}/{next_n}".encode())
    step = _EnvStep(G, next_n, ratio, W, cols, G * -(-W // bk) + 8, seed)
    step.set_step([cap] * G)
    full = step.run()
    _env_assert_exact(step.compare(full), step.lens, "full-length step")
    if next_n == 1:
        targets = list(range(1, W + 1))
    else:
        targets = sorted(
            set(range(1, 257))
            | {m + d for m in range(64, W + 2, 64) for d in (-1, 0, 1) if 1 <= m + d <= W}
        )
    order = torch.randperm(len(targets), generator=step.gh).tolist()
    targets = [targets[i] for i in order]
    targets += targets[: -len(targets) % G]
    bad = seen = 0
    slowest = 0.0
    t0 = time.time()
    for c in range(0, len(targets), G):
        lens = torch.tensor(
            step.lens_of([t * ratio - next_n for t in targets[c : c + G]]),
            dtype=torch.int32,
            device=DEV,
        )
        assert [int(x) for x in lens[:, -1].tolist()] == targets[c : c + G]
        out, dt = _timed(step.run, context_lens=lens)
        slowest = max(slowest, dt)
        m, s = _env_visible_mismatch(out, full, lens)
        bad, seen = bad + m, seen + s
    torch.cuda.synchronize()
    print(
        f"[env length axis r{ratio} w{W} n{next_n}] {len(set(targets))} last-row lengths in "
        f"{len(targets) // G} calls, {time.time() - t0:.1f}s: {seen} visible elements compared, {bad} mismatches; "
        f"slowest call {slowest * 1e3:.1f} ms"
    )
    assert bad == 0
    assert slowest < JIT_BOUND_S


# ─── every output row stride up to the largest widths ────────────────────────────────────────
# The width W reaches DeepGEMM only as the output allocation [rows, ceil256(W)], its slice to W and
# the kernel's runtime logits_stride = ceil256(W); the JIT template does not contain it and the
# schedule is built from the lengths alone. So any W runs the launch of the other widths with the
# same ceil256(W). This runs every stride class from 256 to ceil256 of the ratio's largest width,
# each at its lowest (W = 256k + 1), second-highest and highest width, plus every W in 1..256 and
# every envelope width of the ratio: 32 requests (request 0 at the maximum, request 1 at position 1,
# request 5 a padding request), lengths min(len, W), schedule rebuilt, every visible element
# bitwise equal to the checked call at the maximum. Control: an output read at row stride W (a
# consumer ignoring the padding) differs on the live rows.
ENV_STRIDE_AXIS = [(1, 1048595, 8196), (2, 524297, 8196)]


@pytest.mark.parametrize("next_n", [1, 6], ids=["decode", "dspark_verify"])
@pytest.mark.parametrize(
    "ratio,W,cols", ENV_STRIDE_AXIS, ids=[f"r{r}_w{W}_c{c}" for r, W, c in ENV_STRIDE_AXIS]
)
def test_env_every_row_stride(ratio, W, cols, next_n):
    seed = zlib.crc32(f"env-stride/{ratio}/{W}/{next_n}".encode())
    step = _EnvStep(32, next_n, ratio, W, cols, RUNTIME_POOL_PAGES[ENV_PAGE[ratio]], seed)
    pos = step.positions(1)
    pos[5] = None
    step.set_step(pos)
    full = step.run()
    _env_assert_exact(step.compare(full), step.lens, "maximal width")
    lens_max = step.lens.clone()
    top = _aligned_width(W, torch.float32)
    engine = {w for _, _, r, w, _ in ENV_WIDTHS if r == ratio and w <= W}
    widths = sorted(
        set(range(1, 257))
        | engine
        | {w for m in range(256, top + 1, 256) for w in (m - 255, m - 1, m)}
    )
    controls = sorted(w for w in engine if w % SPLIT_KV)
    rows = step.G * next_n
    bad = seen = expect = 0
    ctl = {}
    slowest = 0.0
    t0 = time.time()
    for Wp in widths:
        lens = lens_max.clamp(max=Wp)
        out, dt = _timed(step.run, context_lens=lens, max_context_len=Wp)
        slowest = max(slowest, dt)
        assert out.shape == (rows, Wp) and out.stride() == (_aligned_width(Wp, torch.float32), 1)
        m, s = _env_visible_mismatch(out, full, lens)
        bad, seen, expect = bad + m, seen + s, expect + int(lens.long().sum())
        if Wp in controls:
            wrong = torch.as_strided(out, (rows, Wp), (Wp, 1))
            live = lens.view(-1)[next_n:] > 0
            vis = torch.arange(Wp, device=DEV) < lens.view(-1, 1)
            differs = ((wrong.view(torch.int32) != full[:, :Wp].view(torch.int32)) & vis).any(1)
            ctl[Wp] = (int((differs[next_n:] & live).sum()), int(live.sum()))
    torch.cuda.synchronize()
    print(
        f"[env stride axis r{ratio} w{W} n{next_n}] {len(widths)} widths (strides 256..{top}) in "
        f"{time.time() - t0:.1f}s: {seen} visible elements compared, {bad} mismatches; slowest call "
        f"{slowest * 1e3:.1f} ms"
    )
    assert bad == 0 and seen == expect
    assert slowest < JIT_BOUND_S
    for Wp, (m, live) in ctl.items():
        print(
            f"[env stride axis control W {Wp}] rows read at stride W that differ: {m} of {live} live rows"
        )
        assert m >= live // 2
