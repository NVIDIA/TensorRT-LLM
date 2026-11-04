# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp4 target's context rows at the 1M config (max_seq_len 1048576, chunked prefill in steps
of max_num_tokens 32768, max_batch_size 16): the per-step choice between the held workspace and
the pool, at tp4 (four ranks) and at tp8 (eight ranks over two nodes of four).

`PrefillTarget._ctx_workspace` picks the step's path once, on host ints, from N, its context KV
(every context request's cached and new rows; the engine's `num_ctx_kv_tokens` counts the new
ones only), against B, the held workspace's rows: `_ctx_workspace_rows` of the engine's
quantities, 2**19 (`_CTX_WORKSPACE_ROWS`) at this config, where a step can attend cached prefixes.
Up to B rows: `ds_mla_dequant_gather` writes the rows into the first N rows of one bf16 [B, 576]
workspace, allocated on the first context step and held -- layer 35's, which the NoPE layers
39, 43, ..., 71 attend over, into the N rows after them while 2N fits B, and into the first N
again for each of those layers past that --, then `flash_mla_sparse_prefill` attends over it.
Past B: `convert_req_index_to_global` turns the selections into pool slots and
`flash_mla_sparse_decode` attends with every context row a one-row sequence, in sub-batches of
at most `_CTX_DECODE_ROWS` (8192) rows with one FlashMLA scheduler per sub-batch size.
`PrefillTarget._attn_ctx` runs the path. Both kernels take the rank's heads first in a 64-head
query whose other heads are zero: 16 heads at tp4, 8 at tp8. The choice and its paths are the
same at both worlds; the heads are not, so every GPU test below runs at each world's heads (ids
`tp4` and `tp8`), on one GPU either way.

On one GPU, over one fp8_ds_mla pool at the 63-owner layout (52000 blocks, 68.8 GB; every byte
0xFF, a NaN in every field, but the owner rows of layers 1, 35 and 36), with the engine's
[16, 32772] block table and selections shaped like the indexer's (request-local positions, every
earlier position below index_topk, else 2048 distinct ones with the row's own):

* the KV-capacity estimate's step (one fresh 32768-token prompt) allocates the workspace; no
  later step reallocates it, no step past B reaches the gather, and every gather writes the
  held buffer: a step's first N rows, or layer 35's N rows after them;
* one packed step of four chunks at N = B (workspace path) and, with its last chunk one row
  longer, at N = B + 1 (decode path, 4 x 8192): the shared rows' outputs are bitwise the same;
* the envelope gate's 1048559-token prompt's last chunk, 32751 rows behind 1015808 cached keys:
  sub-batches 8192 x 3 + 8175 on two schedulers, the short one's split rows within the survey's
  band;
* a q-split step (32768 context rows, the indexer running, rank 0's share the first 8192 rows at
  tp4, 4096 at tp8) behind 1015808 cached keys takes the decode path and attends every row the
  gather filled; its per-layer peak memory against the estimate's step on the workspace path;
* mixed steps on both paths (generation rows behind the context rows, which the context
  attention leaves alone), and layer 35's sharer (39) reading row 35: where the step kept it
  beside its own rows (2N within B), and gathered again after another owner (36) overwrote the
  workspace (2N past B, N within it);
* the packed four-chunk step of the 1M config (cached 1032192 / 524288 / 0 / 2048), on the path
  the step picks: per-layer time and peak memory, printed.

Every decode-path output is checked against the same kernel over an unbounded workspace (the
workspace path's computation without the bound): bitwise in each full 8192-row sub-batch, and
within relative L2 1e-3 overall. Sampled rows of every output are checked against an fp32 softmax
over keys read straight from the pool through the block table. The CPU test checks
the choice on host metadata alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.convert_req_index_to_global import (  # noqa: E501
    convert_req_index_to_global,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flash_mla_sparse_prefill import (  # noqa: E501
    flash_mla_sparse_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm.flash_mla import get_mla_metadata

_LAYERS = 72
_TPB = 32
_TOPK = 2048
#: The checkpoint's MLA heads, split over the ranks: a rank's share goes first in the kernels'
#: 64-head query, the other heads zero.
_MODEL_HEADS = 64
_KERNEL_HEADS = 64
#: The target's worlds: tp4 (four ranks, 16 heads a rank) and tp8 (eight ranks over two nodes of
#: four, 8 heads a rank). One GPU runs a rank's kernels at either.
WORLDS = [pytest.param(4, id="tp4"), pytest.param(8, id="tp8")]
_LAT_DIM, _NOPE, _TILE, _ROW_BYTES = 576, 512, 128, 656
#: Every fourth layer from 39 on attends over layer 35's latent (the checkpoint's
#: kv_source_layers); the pool holds the 63 owners' rows.
_SOURCES = [35 if i >= 39 and i % 4 == 3 else i for i in range(_LAYERS)]
_OWNERS = [i for i in range(_LAYERS) if _SOURCES[i] == i]
#: The owner layers whose rows the pool holds: a RoPE owner, the NoPE source of the sharers,
#: and the owner after it.
_WRITTEN = (1, 35, 36)
#: Pool blocks: the packed four-chunk step of the 1M config holds 49728 of them.
_BLOCKS = 52000
#: md.block_table at max_seq_len 1048576 on the V2 manager: [max_batch_size, 32772].
_TABLE = (16, 32772)
_SCALE = 128**-0.5
_SEED = 1051
#: What the target reads of the 1M config's attention metadata to size its held workspace
#: (chunked prefill on and block reuse off in its runtime features).
_ENGINE_1M = SimpleNamespace(
    max_num_tokens=32768,
    max_num_requests=16,
    max_seq_len=1048576,
    runtime_features=SimpleNamespace(chunked_prefill=True, cache_reuse=False),
)
#: The held workspace's rows there: 2**19, the ceiling.
_B = M._ctx_workspace_rows(_ENGINE_1M)
_SUB = M._CTX_DECODE_ROWS
#: The last chunk of a 1048576-token prompt reads a cached prefix of 1015808 tokens.
_PREFIX_1M = 1048576 - 32768
#: Four chunks (cached, new) packed to a context KV of exactly B in 32767 rows, and the same
#: step with its last chunk one row longer: B + 1 in 32768 rows.
_AT_B = [(358401, 16384), (131072, 8192), (2048, 4096), (0, 4095)]
_PAST_B = _AT_B[:-1] + [(0, 4096)]
#: The agreement between the paths: bitwise in a full sub-batch of 8192 rows; a shorter one
#: (a chunk's tail) has FlashMLA split some rows' keys over SMs, which moves those rows within the
#: survey's measured band for split sub-batches (relative L2 1.4e-04 to 8.7e-04).
_AGREE = 1e-3
#: An fp32 softmax over the selected keys against the bf16 kernels.
_FP32_BAND = 1e-2
_SAMPLED = 192
_GIB = 2**30


def _cumsum(lens: list[int], dev=None) -> torch.Tensor:
    return torch.tensor([0] + list(lens), dtype=torch.int64, device=dev).cumsum(0)


#: cute_dsl_ds_mla_sparse_decode's handles by head count, built once.
_NATIVE = {}


def _native_decode(heads: int):
    """The generation rows' kernel handle at the rank's `heads`, as `build_kernels` builds it."""
    if heads not in _NATIVE:
        _NATIVE[heads] = M.build_ds_mla_decode(torch.cuda.current_device(), num_heads=heads)
    return _NATIVE[heads]


def _target(heads: int | None = None) -> M.PrefillTarget:
    """The prefill target with only what its context and generation attention read off the
    core: the held workspace's rows at the 1M config, the shared source as the core builds it
    (layer 35 alone), the 64-head kernels' softmax scale, index_topk, the 63-owner pool's
    addressing, and the generation rows' native decode at the rank's `heads` (none on the host:
    no generation row is attended there)."""
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = SimpleNamespace(
        ctx_workspace_rows=_B,
        shared_source=[i == 35 for i in range(_LAYERS)],
        # The generation rows on cute_dsl_ds_mla_sparse_decode at the rank's heads, as shipped.
        gen_attn_flash_mla=False,
        _mla_decode=None if heads is None else _native_decode(heads),
        lat_dim=_LAT_DIM,
        softmax_scale=_SCALE,
        idx_topk=_TOPK,
        _stride_factor=len(_OWNERS) * _TPB,
        _pool_row=[_OWNERS.index(_SOURCES[i]) for i in range(_LAYERS)],
    )
    return target


def _host_step(contexts, gens=(), **extra) -> dict:
    """The host fields `_ctx_workspace` reads, as `PrefillTarget.step_args` builds them under
    chunked prefill (`_ctx_kv_offsets` and `_ctx_host_offsets` over the engine's indptrs) for
    context requests of (cached, new) tokens with generation rows behind them."""
    cached = [c for c, _ in contexts]
    new = [n for _, n in contexts]
    md = SimpleNamespace(
        enable_context_mla_with_cached_kv=True,
        ctx_cached_token_indptr=_cumsum(cached),
        ctx_kv_indptr=_cumsum([c + n for c, n in contexts]),
        max_ctx_seq_len=max(new),
        num_ctx_cached_tokens=sum(cached),
        host_ctx_uncached_token_indptr=_cumsum(new),
        host_ctx_kv_indptr=_cumsum([c + n for c, n in contexts]),
    )
    step = dict(
        num_contexts=len(contexts),
        num_ctx_tokens=sum(new),
        # The engine's count: the step's new context keys only.
        num_ctx_kv_tokens=sum(new),
        num_tokens=sum(new) + len(gens),
        skip_ctx=False,
        indexer_prefill_q_split=None,
        **M._ctx_kv_offsets(md, len(contexts)),
        **M._ctx_host_offsets(md, len(contexts)),
    )
    step.update(extra)
    return step


# ---- host only: the per-step choice ----


@pytest.mark.cpu_only
def test_switch_reads_cached_plus_new_on_host_ints():
    """The held workspace is 2**19 rows at the 1M config. The path follows cached + new at B
    and B + 1, whatever the engine's new-token count, the query split or the generation rows
    say; the decode path's sub-batches tile the context rows at 8192 with one scheduler per
    size; layer 35's rows are kept beside a step's while 2N fits B; the workspace is allocated
    once."""
    assert _B == M._CTX_WORKSPACE_ROWS == 2**19
    target = _target()
    cpu = torch.device("cpu")

    at_b = _host_step(_AT_B)
    assert at_b["ctx_cached_tokens"] + at_b["num_ctx_tokens"] == _B
    ctx = target._ctx_workspace(at_b, at_b["num_ctx_tokens"], cpu)
    buf = target._ws_buf
    assert buf.shape == (_B, _LAT_DIM) and buf.dtype == torch.bfloat16
    assert ctx.ws_buf.shape == (_B, _LAT_DIM) and ctx.ws_buf.data_ptr() == buf.data_ptr()
    assert ctx.kv_req.shape == (_B,) and int(ctx.kv_pos[-1]) == 4094
    assert ctx.kept is None

    past_b = _host_step(_PAST_B)
    assert past_b["ctx_cached_tokens"] + past_b["num_ctx_tokens"] == _B + 1
    ctx = target._ctx_workspace(past_b, past_b["num_ctx_tokens"], cpu)
    assert ctx.ws_buf is None and not hasattr(ctx, "kv_req")
    assert ctx.sub_batches == [(0, 8192), (8192, 16384), (16384, 24576), (24576, 32768)]
    assert sorted(ctx.sched) == [8192] and not ctx.sched[8192].have_initialized

    # The envelope gate's prompt's last chunk (1048559 = 1015808 + 32751): its new tokens alone
    # are far below B.
    ragged = _host_step([(_PREFIX_1M, 32751)])
    assert ragged["num_ctx_kv_tokens"] == 32751 < _B
    ctx = target._ctx_workspace(ragged, 32751, cpu)
    assert ctx.ws_buf is None
    assert ctx.sub_batches == [(0, 8192), (8192, 16384), (16384, 24576), (24576, 32751)]
    assert sorted(ctx.sched) == [8175, 8192]

    # A q-split step (32768 context rows, the indexer running, this rank's share 0..8191) over a
    # cached prefix, with generation rows behind it.
    q_split = _host_step(
        [(_PREFIX_1M, 32768 - 3)], gens=(524288, 3001, 17),
        indexer_prefill_q_split=(0, 8192, [8192, 8192, 8192, 8189]),
    )  # fmt: skip
    ctx = target._ctx_workspace(q_split, q_split["num_ctx_tokens"], cpu)
    assert ctx.ws_buf is None and ctx.sub_batches[-1] == (24576, 32765)

    # A dense step (every request within index_topk) and the estimate's fresh 32768-token prompt
    # keep layer 35's rows beside their own; a step of B / 2 + 1 rows fits B only once.
    for contexts in ([(0, 2048)] * 16, [(0, 32768)], [(2047, 1)] * 16):
        step = _host_step(contexts)
        ctx = target._ctx_workspace(step, step["num_ctx_tokens"], cpu)
        assert ctx.ws_buf is not None and ctx.ws_buf.data_ptr() == buf.data_ptr()
        assert ctx.kept.shape == (32768, _LAT_DIM) and ctx.kept.data_ptr() == buf[32768].data_ptr()
    half = _host_step([(_B // 2 - 32767, 32768)])
    ctx = target._ctx_workspace(half, 32768, cpu)
    assert ctx.ws_buf.shape == (_B // 2 + 1, _LAT_DIM) and ctx.kept is None
    assert target._ws_buf is buf


# ---- one GPU ----


def _dev_rows(n: int, g: torch.Generator, nope: bool) -> torch.Tensor:
    """`n` uint8 656-byte rows as the fp8_ds_mla writer stores them: e4m3 codes, power-of-two
    fp32 tile scales 2^-10..2^-4, bf16 rope (zero on a NoPE owner)."""
    dev = g.device
    codes = ((torch.rand(n, _NOPE, generator=g, device=dev) * 2 - 1) * 448).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-10, -3, (n, _NOPE // _TILE), generator=g, device=dev).float())
    if nope:
        rope = torch.zeros(n, _LAT_DIM - _NOPE, device=dev)
    else:
        rope = torch.randn(n, _LAT_DIM - _NOPE, generator=g, device=dev) * 0.5
    return torch.cat(
        [
            codes.view(torch.uint8),
            scale.view(torch.uint8),
            rope.to(torch.bfloat16).view(torch.uint8),
        ],
        dim=1,
    )


def _dequant(rows: torch.Tensor) -> torch.Tensor:
    """fp8_ds_mla rows `[n, 656]` to bf16 `[n, 576]`: e4m3 times its tile's scale (exact in
    bf16 for power-of-two scales), then the rope."""
    codes = rows[:, :_NOPE].contiguous().view(torch.float8_e4m3fn).float()
    scale = rows[:, _NOPE : _NOPE + 16].contiguous().view(torch.float32)
    nope = (codes.view(-1, _NOPE // _TILE, _TILE) * scale[..., None]).view(-1, _NOPE)
    rope = rows[:, _NOPE + 16 :].contiguous().view(torch.bfloat16)
    return torch.cat([nope.to(torch.bfloat16), rope], dim=1)


def _selections(pos: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """int32 `[rows, 2048]` request-local selections for rows at positions `pos`: below
    index_topk every earlier position in order then -1, else one draw from each of 2048 equal
    strata of [0, p] with p itself in the last, so distinct."""
    p = pos.long()[:, None]
    col = torch.arange(_TOPK, device=pos.device)[None]
    lo = col * (p + 1) // _TOPK
    hi = (col + 1) * (p + 1) // _TOPK
    draw = lo + (torch.rand(pos.numel(), _TOPK, generator=g, device=pos.device) * (hi - lo)).long()
    draw = torch.minimum(draw, hi - 1)
    draw[:, -1] = pos
    return torch.where(p < _TOPK, torch.where(col <= p, col, -1), draw).to(torch.int32)


@pytest.fixture(scope="module", params=WORLDS)
def env(request):
    """The pool, the target, and the KV-capacity estimate's step already run through the
    per-step choice, at the world's heads; every gather the module's tests make is recorded
    (observe only)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() != 103:
        pytest.skip("the mistral_large4 modeling_v2 targets are certified on sm_103 only")
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(_SEED)
    pool = torch.empty((_BLOCKS, len(_OWNERS), _TPB, _ROW_BYTES), dtype=torch.uint8, device=dev)
    pool.fill_(0xFF)
    rows = {layer: _OWNERS.index(layer) for layer in _WRITTEN}
    for layer, row in rows.items():
        for lo in range(0, _BLOCKS, 1 << 15):
            hi = min(lo + (1 << 15), _BLOCKS)
            drawn = _dev_rows((hi - lo) * _TPB, g, nope=layer == 35)
            pool[lo:hi, row] = drawn.view(hi - lo, _TPB, _ROW_BYTES)
            del drawn
    gathers = []
    real_gather = M.ds_mla_dequant_gather

    def recording_gather(pool_, row, kv_req, kv_pos, block_table, out):
        gathers.append(
            SimpleNamespace(rows=int(kv_req.shape[0]), out_ptr=out.data_ptr(), pool_row=row)
        )
        return real_gather(pool_, row, kv_req, kv_pos, block_table, out)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, "ds_mla_dequant_gather", recording_gather)
        e = SimpleNamespace(
            dev=dev, pool=pool, kv=pool.view(-1, _TPB, 1, _ROW_BYTES), rows=rows,
            target=_target(_MODEL_HEADS // request.param), gathers=gathers, world=request.param,
            heads=_MODEL_HEADS // request.param,
        )  # fmt: skip
        assert e.target._ws_buf is None
        # The KV-capacity estimate's step at 1M: one fresh 32768-token prompt.
        e.estimate = _step(e, [(0, 32768)], seed=1)
        e.estimate_ctx = e.target._ctx_workspace(e.estimate.step, e.estimate.tc, dev)
        e.buf = e.target._ws_buf
        assert e.buf.shape == (_B, _LAT_DIM) and e.estimate_ctx.ws_buf is not None
        yield e
    assert e.target._ws_buf is e.buf


def _step(env, contexts, gens=(), seed: int = 0, **extra) -> SimpleNamespace:
    """A step of context requests of (cached, new) tokens with generation rows of the given KV
    lengths behind them: each request's pages distinct blocks of the pool, its table row
    padded to the engine's width, a query per row (this rank's heads drawn -- 16 at tp4, 8 at
    tp8 -- the rest zero) and its selection."""
    dev, heads = env.dev, env.heads
    g = torch.Generator(device=dev).manual_seed(_SEED + seed)
    kv_lens = [c + n for c, n in contexts] + list(gens)
    pages = [-(-k // _TPB) for k in kv_lens]
    assert sum(pages) <= _BLOCKS and max(pages) <= _TABLE[1]
    order = torch.randperm(_BLOCKS, generator=g, device=dev).to(torch.int32)
    table = torch.zeros(_TABLE, dtype=torch.int32, device=dev)
    used = 0
    for r, n in enumerate(pages):
        table[r, :n] = order[used : used + n]
        used += n
    tc = sum(n for _, n in contexts)
    req_idx = torch.cat(
        [torch.full((n,), r, dtype=torch.int32, device=dev) for r, (_, n) in enumerate(contexts)]
        + [torch.arange(len(contexts), len(kv_lens), dtype=torch.int32, device=dev)]
    )
    pos = torch.cat(
        [torch.arange(c, c + n, device=dev) for c, n in contexts]
        + [torch.tensor(gens, device=dev, dtype=torch.int64) - 1]
    )
    q = torch.zeros(pos.numel(), _KERNEL_HEADS, _LAT_DIM, dtype=torch.bfloat16, device=dev)
    q[:, :heads] = torch.randn(pos.numel(), heads, _LAT_DIM, generator=g, device=dev) * 0.1
    step = _host_step(contexts, gens, **extra)
    step = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in step.items()}
    step.update(block_table=table, tokens_per_block=_TPB, latent_pool=env.pool, latent_kv=env.kv)
    return SimpleNamespace(
        step=step, contexts=list(contexts), gens=list(gens), kv_lens=kv_lens, tc=tc,
        rows=pos.numel(), req_idx=req_idx, pos=pos, topk=_selections(pos, g), q=q, table=table,
    )  # fmt: skip


def _request_kv(env, s, layer: int, r: int) -> torch.Tensor:
    """Request r's KV rows 0..kv_len-1 of `layer`, bf16, read through its block table."""
    n = s.kv_lens[r]
    out = torch.empty(n, _LAT_DIM, dtype=torch.bfloat16, device=env.dev)
    for lo in range(0, n, 1 << 19):
        p = torch.arange(lo, min(lo + (1 << 19), n), device=env.dev)
        out[lo : lo + p.numel()] = _dequant(
            env.pool[s.table[r, p // _TPB].long(), env.rows[layer], p % _TPB]
        )
    return out


def _unbounded(env, s, layer: int) -> torch.Tensor:
    """The workspace path's computation without its bound: every context request's rows in one
    bf16 buffer of N rows, the selections offset into it, the same prefill kernel."""
    ws = torch.cat([_request_kv(env, s, layer, r) for r in range(len(s.contexts))])
    off = _cumsum([c + n for c, n in s.contexts], env.dev)[:-1]
    sel = s.topk[: s.tc]
    idx = torch.where(sel >= 0, sel + off[s.req_idx[: s.tc].long()][:, None], sel).to(torch.int32)
    out = torch.empty(s.tc, env.heads, _NOPE, dtype=torch.bfloat16, device=env.dev)
    flash_mla_sparse_prefill(s.q[: s.tc], ws, idx, _SCALE, out)
    return out


def _fp32_check(env, s, layer: int, out: torch.Tensor, rows: torch.Tensor) -> float:
    """`out`'s rows `rows` against an fp32 softmax over each row's selected keys read straight
    from the pool through its request's table row; returns the relative L2 error."""
    sel = s.topk[rows].long()
    req = s.req_idx[rows].long()
    valid = sel >= 0
    key = sel.clamp_min(0)
    blocks = s.table[req[:, None], key // _TPB].long()
    k = _dequant(env.pool[blocks, env.rows[layer], key % _TPB].view(-1, _ROW_BYTES)).float()
    k = k.view(rows.numel(), _TOPK, _LAT_DIM)
    with true_fp32_matmul():
        logits = torch.einsum("rhd,rkd->rhk", s.q[rows, : env.heads].float(), k) * _SCALE
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        exact = torch.einsum("rhk,rkd->rhd", torch.softmax(logits, -1), k[..., :_NOPE])
    got = out[rows].float()
    assert bool(torch.isfinite(got).all()), "a sampled row read an unwritten pool byte"
    return float((got - exact).norm() / exact.norm())


def _sample(rows: int, g_seed: int, must=()) -> torch.Tensor:
    g = torch.Generator().manual_seed(g_seed)
    picked = torch.randperm(rows, generator=g)[:_SAMPLED].tolist()
    keep = sorted(set(picked) | {r for r in must if 0 <= r < rows})
    return torch.tensor(keep, device="cuda")


def _attend(env, s, layer: int, ctx=None, out=None):
    """`_attn_ctx` for `layer` over the step's context rows: the step's path (made here unless
    given) and the output (NaN-filled first unless given)."""
    if ctx is None:
        ctx = env.target._ctx_workspace(s.step, s.tc, env.dev)
    if out is None:
        out = torch.full((s.rows, env.heads, _NOPE), float("nan"), dtype=torch.bfloat16,
                         device=env.dev)  # fmt: skip
    L = SimpleNamespace(kv_source=_SOURCES[layer])
    env.target._attn_ctx(s.step, L, s.q[: s.tc], out[: s.tc], s.req_idx, s.topk, ctx)
    return ctx, out


def _agreement(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, int]:
    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
    return rel, int((got.view(torch.int16) != ref.view(torch.int16)).sum())


def _check_decode_path(env, s, ctx, out, layer: int, label: str) -> None:
    """A decode-path output: finite, within the agreement of the unbounded workspace
    computation (bitwise in every full sub-batch), and the sampled rows within the fp32 band."""
    lat = out[: s.tc]
    assert bool(torch.isfinite(lat).all()), f"{label}: a context row read an unwritten pool byte"
    ref = _unbounded(env, s, layer)
    rel, differ = _agreement(lat, ref)
    rows_differ = [
        int((lat[lo:hi].view(torch.int16) != ref[lo:hi].view(torch.int16)).flatten(1).any(1).sum())
        for lo, hi in ctx.sub_batches
    ]
    edges = [lo for lo, _ in ctx.sub_batches] + [hi - 1 for _, hi in ctx.sub_batches]
    err = _fp32_check(env, s, layer, out, _sample(s.tc, layer, edges))
    print(f"CTXBOUND {label} layer {layer}: decode path, sub-batches "
          f"{[hi - lo for lo, hi in ctx.sub_batches]}, against the unbounded workspace rel L2 "
          f"{rel:.3e} ({differ} of {lat.numel()} elements differ; rows differing per sub-batch "
          f"{rows_differ}); fp32 rel L2 {err:.3e}")  # fmt: skip
    assert rel <= _AGREE, (label, rel)
    for (lo, hi), n in zip(ctx.sub_batches, rows_differ):
        if hi - lo == _SUB:
            assert n == 0, (label, lo, hi, n)
    assert err < _FP32_BAND, (label, err)


def _measure(fn, iters: int = 5) -> tuple[float, int]:
    """(ms per call, peak bytes above the allocation before the call) of a warm `fn`."""
    fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, peak


def test_same_packed_step_at_b_and_b_plus_one(env):
    """Four chunks packed to N = B take the workspace path; the same step with its last chunk
    one row longer (N = B + 1, 32768 rows) takes the decode path in four sub-batches of 8192.
    The 32767 shared rows' outputs are the same."""
    past = _step(env, _PAST_B, seed=2)
    assert sum(past.kv_lens) == _B + 1
    ctx_past, out_past = _attend(env, past, 1)
    assert ctx_past.ws_buf is None
    assert ctx_past.sub_batches == [(lo, lo + _SUB) for lo in range(0, 32768, _SUB)]
    _check_decode_path(env, past, ctx_past, out_past, 1, "N = B + 1")

    # The same step, its last row dropped: the same pool, table, queries and selections.
    at = SimpleNamespace(**vars(past))
    at.contexts = list(_AT_B)
    at.kv_lens = [c + n for c, n in at.contexts]
    at.tc = at.rows = past.tc - 1
    at.step = dict(past.step, **_host_step(at.contexts))
    at.step = {k: v.to(env.dev) if isinstance(v, torch.Tensor) else v for k, v in at.step.items()}
    assert sum(at.kv_lens) == _B
    gathered = len(env.gathers)
    ctx_at, out_at = _attend(env, at, 1)
    assert ctx_at.ws_buf is not None and ctx_at.ws_buf.data_ptr() == env.buf.data_ptr()
    assert [x.rows for x in env.gathers[gathered:]] == [_B]
    rel, differ = _agreement(out_at[: at.tc], out_past[: at.tc])
    print(f"CTXBOUND N = B (workspace) against N = B + 1 (decode) on the {at.tc} shared rows: "
          f"rel L2 {rel:.3e}, {differ} elements differ")  # fmt: skip
    assert rel <= _AGREE and differ == 0, (rel, differ)
    err = _fp32_check(env, at, 1, out_at, _sample(at.tc, 7, (0, 16383, 16384, at.tc - 1)))
    assert err < _FP32_BAND, err
    # The workspace holds every request's rows, request after request.
    off = 0
    for r, n in enumerate(at.kv_lens):
        assert torch.equal(env.buf[off : off + n], _request_kv(env, at, 1, r)), r
        off += n


def test_ragged_last_chunk_of_a_1048559_token_prompt(env):
    """The envelope gate's 1048559-token prompt's last chunk, 32751 rows behind 1015808 cached
    keys: sub-batches 8192 x 3 + 8175, on two schedulers the 72 layers would share."""
    s = _step(env, [(_PREFIX_1M, 32751)], seed=3)
    gathered = len(env.gathers)
    ctx, out = _attend(env, s, 1)
    assert [hi - lo for lo, hi in ctx.sub_batches] == [8192, 8192, 8192, 8175]
    assert sorted(ctx.sched) == [8175, 8192]
    assert all(m.have_initialized for m in ctx.sched.values())
    assert (ctx.sched[8192].config.b, ctx.sched[8175].config.b) == (8192, 8175)
    _check_decode_path(env, s, ctx, out, 1, "ragged 32751 behind 1015808")
    # The next layer reuses the filled schedulers.
    again = _attend(env, s, 1, ctx=ctx)[1]
    assert torch.equal(again[: s.tc], out[: s.tc])
    assert len(env.gathers) == gathered


def test_q_split_step_behind_a_1m_prefix_takes_the_decode_path(env):
    """The q-split step of a 1M prompt's last chunk: 32768 context rows with the indexer running
    and rank 0's share its first 32768 / world rows (8192 at tp4, 4096 at tp8), after the gather
    every row's selection in the buffer. It takes the decode path and attends every row. Its
    per-layer peak memory stays under the KV-capacity estimate's step on the workspace path,
    whose peak sizes the pool."""
    share = 32768 // env.world
    s = _step(
        env, [(_PREFIX_1M, 32768)], seed=4,
        indexer_prefill_q_split=(0, share, [share] * env.world),
    )  # fmt: skip
    assert s.step["skip_ctx"] is False and s.tc >= 8192
    gathered = len(env.gathers)
    ctx, out = _attend(env, s, 1)
    assert ctx.ws_buf is None and len(ctx.sub_batches) == 4 and len(env.gathers) == gathered
    _check_decode_path(env, s, ctx, out, 1, "q-split 32768 behind 1015808")
    assert bool(torch.isfinite(out[share : s.tc]).all())

    est = env.estimate
    lat_est = torch.empty(est.tc, env.heads, _NOPE, dtype=torch.bfloat16, device=env.dev)
    lat_dec = torch.empty(s.tc, env.heads, _NOPE, dtype=torch.bfloat16, device=env.dev)
    L = SimpleNamespace(kv_source=1)
    ws_ms, ws_peak = _measure(
        lambda: env.target._attn_ctx(
            est.step, L, est.q, lat_est, est.req_idx, est.topk, env.estimate_ctx
        )
    )
    dec_ms, dec_peak = _measure(
        lambda: env.target._attn_ctx(s.step, L, s.q, lat_dec, s.req_idx, s.topk, ctx)
    )
    print(f"CTXBOUND per layer: the estimate's step (32768 fresh rows, workspace path) "
          f"{ws_ms:.3f} ms, peak {ws_peak / _GIB:.3f} GiB above the held workspace; "
          f"32768 rows behind 1015808 (decode path) {dec_ms:.3f} ms, peak "
          f"{dec_peak / _GIB:.3f} GiB; margin {(ws_peak - dec_peak) / _GIB:.3f} GiB")  # fmt: skip
    assert dec_peak <= ws_peak, (dec_peak, ws_peak)


def test_mixed_steps_and_layer_35s_sharers(env):
    """Generation rows behind the context rows on each path: the context attention writes the
    context rows only, and the generation rows' own call (`_attn_gen`, the native decode at the
    rank's heads) reads the
    pool. Layer 39 attends over layer 35's rows: on a step whose 2N fits B, where layer 35
    gathered them, the N rows kept after the step's first N, with no gather of its own, layer 36
    having gathered into the first N; on a step of N past B / 2, from its own gather of row 35
    into the first N rows, which layer 36 overwrote; on the decode path, from the pool. Each is
    bitwise layer 35's output."""
    target = env.target
    for label, contexts, gens in (
        ("decode", [(_PREFIX_1M, 16384)], (524288, 3001, 17)),
        ("workspace, kept", [(4096, 640), (0, 300)], (3001, 17)),
        ("workspace, gathered again", [(_B // 2, 640), (0, 300)], (3001, 17)),
    ):
        s = _step(env, contexts, gens, seed=5)
        n_kv = sum(c + n for c, n in contexts)
        ctx = target._ctx_workspace(s.step, s.tc, env.dev)
        assert (ctx.ws_buf is None) == (label == "decode")
        if ctx.ws_buf is not None:
            assert ctx.ws_buf.data_ptr() == env.buf.data_ptr()
            assert (ctx.kept is not None) == (label == "workspace, kept") == (2 * n_kv <= _B)
        outs = {}
        for layer in (35, 36, 39):
            gathered = len(env.gathers)
            ctx, out = _attend(env, s, layer, ctx=ctx)
            assert bool(torch.isnan(out[s.tc :].float()).all()), (label, layer)
            wrote = [(x.rows, x.out_ptr) for x in env.gathers[gathered:]]
            if ctx.ws_buf is None:
                assert wrote == [], (label, layer)
            else:
                # Where the source's rows are now, request after request: layer 35's in the
                # kept rows when the step has them, every other gather's in the first N.
                src = _SOURCES[layer]
                kept = src == 35 and ctx.kept is not None
                ws = ctx.kept if kept else ctx.ws_buf
                if kept:
                    assert ws.data_ptr() == env.buf[n_kv].data_ptr()
                expect = [] if kept and layer == 39 else [(n_kv, ws.data_ptr())]
                assert wrote == expect, (label, layer)
                off = 0
                for r in range(len(contexts)):
                    n = s.kv_lens[r]
                    assert torch.equal(ws[off : off + n], _request_kv(env, s, src, r))
                    off += n
            slots = convert_req_index_to_global(
                s.req_idx[s.tc :], s.table, s.topk[s.tc :], _TPB, _TOPK,
                len(_OWNERS) * _TPB, target.core._pool_row[_SOURCES[layer]],
            )  # fmt: skip
            L = SimpleNamespace(kv_source=_SOURCES[layer])
            target._attn_gen(s.step, L, s.q[s.tc :], out[s.tc :], slots, get_mla_metadata()[0])
            outs[layer] = out
        assert torch.equal(outs[39], outs[35]), label
        assert not torch.equal(outs[36], outs[35]), label
        for layer in (35, 36):
            rows = _sample(s.rows, layer, range(s.tc - 2, s.rows))
            err = _fp32_check(env, s, layer, outs[layer], rows)
            print(f"CTXBOUND mixed step, {label} path, layer {layer}: {s.tc} context rows and "
                  f"{len(gens)} generation rows, fp32 rel L2 {err:.3e}")  # fmt: skip
            assert err < _FP32_BAND, (label, layer, err)
        if label == "decode":
            _check_decode_path(env, s, ctx, outs[35], 35, "mixed, layer 35")
        else:
            # Each context row is the same kernel over its own request's gather.
            start = 0
            for r, (cached, new) in enumerate(contexts):
                kv = _request_kv(env, s, 35, r)
                own = torch.empty(new, env.heads, _NOPE, dtype=torch.bfloat16, device=env.dev)
                flash_mla_sparse_prefill(
                    s.q[start : start + new], kv, s.topk[start : start + new], _SCALE, own
                )
                assert torch.equal(outs[35][start : start + new], own), r
                start += new


def test_packed_four_chunks_at_1m(env):
    """Four chunks of 16384 / 8192 / 4096 / 4096 rows packed behind 1032192 / 524288 / 0 / 2048
    cached keys (N = 1591296): the path the step picks, its per-layer time and peak memory."""
    contexts = [(1032192, 16384), (524288, 8192), (0, 4096), (2048, 4096)]
    s = _step(env, contexts, seed=6)
    ctx, out = _attend(env, s, 1)
    path = "workspace" if ctx.ws_buf is not None else "decode"
    assert path == "decode" and sum(s.kv_lens) == 1591296
    _check_decode_path(env, s, ctx, out, 1, "packed four chunks")
    lat = torch.empty(s.tc, env.heads, _NOPE, dtype=torch.bfloat16, device=env.dev)
    L = SimpleNamespace(kv_source=1)
    ms, peak = _measure(lambda: env.target._attn_ctx(s.step, L, s.q, lat, s.req_idx, s.topk, ctx))
    print(f"CTXBOUND packed four-chunk step (cached 1032192/524288/0/2048, new "
          f"16384/8192/4096/4096, N {sum(s.kv_lens)}): {path} path, {ms:.3f} ms per layer, "
          f"peak {peak / _GIB:.3f} GiB above the held workspace")  # fmt: skip


def test_workspace_held_and_no_gather_past_b(env):
    """After every step above and a sequence of its own (sixteen requests filling B exactly,
    then B + 1, then a 1M chunk), the workspace is the estimate's step's allocation, every
    gather wrote into it -- a step's first N rows, or layer 35's N rows kept after them on a
    step whose 2N fits B --, and no step past B reached the gather."""
    for contexts, path in (
        ([(30720, 2048)] * 16, "workspace"),
        ([(30720, 2048)] * 15 + [(30721, 2048)], "decode"),
        ([(_PREFIX_1M, 32768)], "decode"),
        ([(5, 300)], "workspace"),
    ):
        s = _step(env, contexts, seed=7)
        gathered = len(env.gathers)
        ctx, out = _attend(env, s, 36)
        assert (ctx.ws_buf is None) == (path == "decode"), (contexts[0], path)
        assert len(env.gathers) - gathered == (path == "workspace")
        assert bool(torch.isfinite(out[: s.tc]).all())
    assert env.target._ws_buf is env.buf
    assert env.gathers and all(x.rows <= _B for x in env.gathers)
    base, row_bytes = env.buf.data_ptr(), _LAT_DIM * env.buf.element_size()
    kept = [x for x in env.gathers if x.out_ptr != base]
    for x in kept:
        assert x.out_ptr == base + x.rows * row_bytes and 2 * x.rows <= _B, vars(x)
        assert x.pool_row == _OWNERS.index(35), vars(x)
    print(f"CTXBOUND {len(env.gathers)} gathers, at most {max(x.rows for x in env.gathers)} "
          f"rows, all into the workspace allocated on the estimate's step ({len(kept)} of "
          f"layer 35 into the rows kept after a step's)")  # fmt: skip
