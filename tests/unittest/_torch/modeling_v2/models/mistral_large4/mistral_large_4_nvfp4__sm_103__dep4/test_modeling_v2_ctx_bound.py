# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep4 target's context rows at the 1M config (max_seq_len 1048576, chunked prefill, max
num_tokens 32768 and 16 requests per rank): the per-step choice between the held workspace and
the pool.

`PrefillTarget._ctx_workspace` picks the step's path once, on this rank's host ints, from N, its
context KV (every context request's cached and new rows; the engine's `num_ctx_kv_tokens` counts
the new ones only), against the held workspace's rows, which `_derive_first_step` sizes from the
engine (`_ctx_workspace_rows`): B = 2**19 at this config, the ceiling `_CTX_WORKSPACE_ROWS`. Up to
B rows: `ds_mla_dequant_gather` writes the rows into the first N rows of one bf16 [B, 576]
workspace, allocated on the first context step and held, then `flash_mla_sparse_prefill` attends
over it; layer 35, the latent the NoPE layers 39..71 attend over, is gathered into the N rows
after the step's own when 2N fits B, which its sharers read without a gather, and otherwise
again by every sharer into the first N. Past B: `convert_req_index_to_global` turns the
selections into pool slots and `flash_mla_sparse_decode` attends with every context row a
one-row sequence, in sub-batches of at most `_CTX_DECODE_ROWS` (8192) rows with one FlashMLA
scheduler per sub-batch size. `PrefillTarget._attn_ctx` runs the path. Under attention DP a
rank's step holds its own requests only, all 64 heads of each. (The held workspace at other
configs is test_modeling_v2_ctx_workspace.py's.)

On one GPU, over one fp8_ds_mla pool at the 63-owner layout (52000 blocks, 68.8 GB; every byte
0xFF, a NaN in every field, but the owner rows of layers 1, 35 and 36), with the engine's
[16, 32772] block table, queries at all 64 heads, and selections shaped like the indexer's
(request-local positions, every earlier position below index_topk, which is also the engine's
dense causal index on a step it skipped the indexer for, else 2048 distinct ones with the row's
own):

* the KV-capacity estimate's step (one fresh 32768-token prompt) allocates the workspace; no
  later step reallocates it, no step past B reaches the gather, and every gather writes the
  held buffer, the step's first N rows or, for layer 35 on a step that fits twice, the N after;
* one packed step of four chunks at N = B (workspace path) and, with its last chunk one row
  longer, at N = B + 1 (decode path, 4 x 8192): the shared rows' outputs are bitwise the same;
* a 1M prompt's last full chunk (32768 rows behind 1015808 cached keys, 4 x 8192) and the
  1048559-token prompt's last chunk (32751 rows, 8192 x 3 + 8175 on two schedulers); the
  first's per-layer peak memory against the estimate's step on the workspace path;
* several requests of one rank, one of them over a 1M cached prefix, and the packed four-chunk
  step at max_seq_len (cached 1032192 / 524288 / 0 / 2048): the decode path, with its per-layer
  time and peak memory;
* mixed steps on the decode path and on the workspace path at N fitting B twice and once
  (generation rows behind the context rows, which the context attention leaves alone), with
  layer 35's sharer (39) reading row 35 after another owner (36) gathered: from the kept rows,
  or gathered again after 36 overwrote the step's rows;
* a skipped-indexer (dense) step of sixteen requests within index_topk, on the workspace path
  with layer 35's rows kept;
* a control: the same harness over a workspace one row short of N, and over the new tokens
  alone (the engine's count), fails, though the kernel raises nothing and writes no NaN.

Every decode-path output is checked against the same prefill kernel over an unbounded
workspace (the workspace path's computation without the bound): rows of each full 8192-row
sub-batch bitwise, and the whole within relative L2 1e-3. Sampled rows of every output are
checked against an fp32 softmax over keys read straight from the pool through the block table.
The CPU test checks the path choice on host metadata alone.
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
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm.flash_mla import get_mla_metadata

_LAYERS = 72
_TPB = 32
_TOPK = 2048
#: Attention DP: a rank runs all 64 heads, FlashMLA's own head count.
_HEADS = 64
_LAT_DIM, _NOPE, _TILE, _ROW_BYTES = 576, 512, 128, 656
#: Every fourth layer from 39 on attends over layer 35's latent (the checkpoint's
#: kv_source_layers); the pool holds the 63 owners' rows.
_SOURCES = [35 if i >= 39 and i % 4 == 3 else i for i in range(_LAYERS)]
_OWNERS = [i for i in range(_LAYERS) if _SOURCES[i] == i]
#: The owner layers whose rows the pool holds: a RoPE owner, the NoPE source of the sharers,
#: and the owner after it.
_WRITTEN = (1, 35, 36)
#: Pool blocks: the packed four-chunk step at max_seq_len holds 49728 of them.
_BLOCKS = 52000
#: md.block_table at max_seq_len 1048576 on the V2 manager: [max_batch_size, 32772].
_TABLE = (16, 32772)
_SCALE = 128**-0.5
_SEED = 1051
#: The 1M config's quantities on the attention metadata, which the held workspace is sized from.
_ENGINE_1M = SimpleNamespace(
    max_num_tokens=32768,
    max_num_requests=16,
    max_seq_len=1048576,
    runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=False),
)
_B = M._CTX_WORKSPACE_ROWS
_SUB = M._CTX_DECODE_ROWS
#: The last chunk of a 1048576-token prompt reads a cached prefix of 1015808 tokens.
_PREFIX_1M = 1048576 - 32768
#: Four chunks (cached, new) packed to a context KV of exactly B in 32767 rows, and the same
#: step with its last chunk one row longer: B + 1 in 32768 rows.
_AT_B = [(358401, 16384), (131072, 8192), (2048, 4096), (0, 4095)]
_PAST_B = _AT_B[:-1] + [(0, 4096)]
#: The agreement between the paths, relative L2 over the step: FlashMLA's decode kernel splits
#: some rows' keys over SMs in a sub-batch shorter than 8192 (a chunk's tail), which moves those
#: rows within the survey's measured band for split sub-batches (relative L2 1.4e-04 to
#: 8.7e-04).
_AGREE = 1e-3
#: An fp32 softmax over the selected keys against the bf16 kernels.
_FP32_BAND = 1e-2
_SAMPLED = 128
_GIB = 2**30


def _cumsum(lens: list[int], dev=None) -> torch.Tensor:
    return torch.tensor([0] + list(lens), dtype=torch.int64, device=dev).cumsum(0)


def _target() -> M.PrefillTarget:
    """The 1M config's prefill target with only what its context and generation attention read
    off the core: the held workspace's rows (`_ctx_workspace_rows` of the config, B), layer 35
    the one source kept for its sharers (`shared_source`, as every core holds it), the latent
    width, the kernels' softmax scale, index_topk and the 63-owner pool's addressing."""
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = SimpleNamespace(
        ctx_workspace_rows=M._ctx_workspace_rows(_ENGINE_1M),
        shared_source=[i == 35 for i in range(_LAYERS)],
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
        host_ctx_uncached_token_indptr=_cumsum(new),
        host_ctx_kv_indptr=_cumsum([c + n for c, n in contexts]),
        max_ctx_seq_len=max(new),
        num_ctx_cached_tokens=sum(cached),
    )
    step = dict(
        num_contexts=len(contexts),
        num_ctx_tokens=sum(new),
        # The engine's count: the step's new context keys only.
        num_ctx_kv_tokens=sum(new),
        num_tokens=sum(new) + len(gens),
        skip_ctx=False,
        **M._ctx_kv_offsets(md, len(contexts)),
        **M._ctx_host_offsets(md, len(contexts)),
    )
    step.update(extra)
    return step


# ---- host only: the path choice ----


@pytest.mark.cpu_only
def test_path_reads_cached_plus_new_on_host_ints():
    """The 1M config holds B = 2**19 rows. The path follows cached + new at B and B + 1, whatever
    the engine's new-token count or the generation rows say, and layer 35's kept rows follow
    2N against B; the decode path's sub-batches tile the context rows at 8192 with one scheduler
    per size; the workspace is allocated once, on the first context step."""
    assert M._ctx_workspace_rows(_ENGINE_1M) == _B == 1 << 19
    target = _target()
    cpu = torch.device("cpu")
    assert target._ws_buf is None

    # The KV-capacity estimate's step: one fresh 32768-token prompt per rank, kept rows beside.
    estimate = _host_step([(0, 32768)])
    ctx = target._ctx_workspace(estimate, 32768, cpu)
    buf = target._ws_buf
    assert buf.shape == (_B, _LAT_DIM) and buf.dtype == torch.bfloat16
    assert ctx.ws_buf.shape == (32768, _LAT_DIM) and ctx.ws_buf.data_ptr() == buf.data_ptr()
    assert ctx.kept.shape == (32768, _LAT_DIM)
    assert ctx.kept.data_ptr() == buf.data_ptr() + 32768 * _LAT_DIM * buf.element_size()

    at_b = _host_step(_AT_B)
    assert at_b["ctx_cached_tokens"] + at_b["num_ctx_tokens"] == _B
    ctx = target._ctx_workspace(at_b, at_b["num_ctx_tokens"], cpu)
    assert ctx.ws_buf.shape == (_B, _LAT_DIM) and ctx.ws_buf.data_ptr() == buf.data_ptr()
    assert ctx.kv_req.shape == (_B,) and int(ctx.kv_pos[-1]) == 4094 and ctx.kept is None

    past_b = _host_step(_PAST_B)
    assert past_b["ctx_cached_tokens"] + past_b["num_ctx_tokens"] == _B + 1
    ctx = target._ctx_workspace(past_b, past_b["num_ctx_tokens"], cpu)
    assert ctx.ws_buf is None and not hasattr(ctx, "kv_req")
    assert ctx.sub_batches == [(0, 8192), (8192, 16384), (16384, 24576), (24576, 32768)]
    assert sorted(ctx.sched) == [8192] and not ctx.sched[8192].have_initialized

    # A 1M prompt's last full chunk and the 1048559-token prompt's last chunk (1015808 +
    # 32751): their new tokens alone are far below B.
    for new, subs in ((32768, [8192] * 4), (32751, [8192, 8192, 8192, 8175])):
        step = _host_step([(_PREFIX_1M, new)])
        assert step["num_ctx_kv_tokens"] == new < _B
        ctx = target._ctx_workspace(step, new, cpu)
        assert ctx.ws_buf is None and [hi - lo for lo, hi in ctx.sub_batches] == subs
        assert sorted(ctx.sched) == sorted(set(subs))

    # Several requests of one rank, one over a 1M prefix, with generation rows behind them.
    step = _host_step(
        [(_PREFIX_1M, 16384), (4096, 8192), (0, 4096), (100, 4000)], gens=(524288, 3001, 17)
    )
    ctx = target._ctx_workspace(step, step["num_ctx_tokens"], cpu)
    assert ctx.ws_buf is None and ctx.sub_batches[-1] == (24576, 32672)

    # Dense steps (every request within index_topk) and short ones stay on the workspace, with
    # layer 35's rows kept; past B / 2 a step keeps none.
    for contexts in ([(0, 2048)] * 16, [(2047, 1)] * 16, [(1024, 1024)] * 16, [(5, 300)]):
        step = _host_step(contexts)
        ctx = target._ctx_workspace(step, step["num_ctx_tokens"], cpu)
        assert ctx.ws_buf is not None and ctx.ws_buf.data_ptr() == buf.data_ptr()
        assert ctx.kept is not None
    for n_kv, kept in ((_B // 2, True), (_B // 2 + 1, False)):
        step = _host_step([(n_kv - 640, 640)])
        ctx = target._ctx_workspace(step, 640, cpu)
        assert ctx.ws_buf.shape == (n_kv, _LAT_DIM) and (ctx.kept is not None) == kept
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
    index_topk every earlier position in order then -1 (the engine's dense causal index), else
    one draw from each of 2048 equal strata of [0, p] with p itself in the last, so distinct."""
    p = pos.long()[:, None]
    col = torch.arange(_TOPK, device=pos.device)[None]
    lo = col * (p + 1) // _TOPK
    hi = (col + 1) * (p + 1) // _TOPK
    draw = lo + (torch.rand(pos.numel(), _TOPK, generator=g, device=pos.device) * (hi - lo)).long()
    draw = torch.minimum(draw, hi - 1)
    draw[:, -1] = pos
    return torch.where(p < _TOPK, torch.where(col <= p, col, -1), draw).to(torch.int32)


@pytest.fixture(scope="module")
def env():
    """The pool, the target, and the KV-capacity estimate's step already run through the
    path choice; every gather the module's tests make is recorded (observe only)."""
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
            target=_target(), gathers=gathers,
        )  # fmt: skip
        assert e.target._ws_buf is None
        # The KV-capacity estimate's step at the 1M config: one fresh 32768-token prompt per
        # rank.
        e.estimate = _step(e, [(0, 32768)], seed=1)
        e.estimate_ctx = e.target._ctx_workspace(e.estimate.step, e.estimate.tc, dev)
        e.buf = e.target._ws_buf
        assert e.buf.shape == (_B, _LAT_DIM) and e.estimate_ctx.ws_buf is not None
        yield e
    assert e.target._ws_buf is e.buf


def _step(env, contexts, gens=(), seed: int = 0, **extra) -> SimpleNamespace:
    """A step of context requests of (cached, new) tokens with generation rows of the given KV
    lengths behind them: each request's pages distinct blocks of the pool, its table row
    padded to the engine's width, a query per row at all 64 heads, and its selection."""
    dev = env.dev
    g = torch.Generator(device=dev).manual_seed(_SEED + seed)
    kv_lens = [c + n for c, n in contexts] + list(gens)
    pages = [-(-k // _TPB) for k in kv_lens]
    assert sum(pages) <= _BLOCKS and max(pages) <= _TABLE[1] and len(kv_lens) <= _TABLE[0]
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
    q = torch.randn(pos.numel(), _HEADS, _LAT_DIM, generator=g, device=dev).mul_(0.1)
    q = q.to(torch.bfloat16)
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
    out = torch.empty(s.tc, _HEADS, _NOPE, dtype=torch.bfloat16, device=env.dev)
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
        logits = torch.matmul(s.q[rows].float(), k.transpose(1, 2)) * _SCALE
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        exact = torch.matmul(torch.softmax(logits, -1), k[..., :_NOPE])
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
        out = torch.full((s.rows, _HEADS, _NOPE), float("nan"), dtype=torch.bfloat16,
                         device=env.dev)  # fmt: skip
    L = SimpleNamespace(kv_source=_SOURCES[layer])
    env.target._attn_ctx(s.step, L, s.q[: s.tc], out[: s.tc], s.req_idx, s.topk, ctx)
    return ctx, out


def _agreement(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, int]:
    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
    return rel, int((got.view(torch.int16) != ref.view(torch.int16)).sum())


def _rows_differing(got: torch.Tensor, ref: torch.Tensor, bounds) -> list[int]:
    return [
        int((got[lo:hi].view(torch.int16) != ref[lo:hi].view(torch.int16)).flatten(1).any(1).sum())
        for lo, hi in bounds
    ]


def _check_decode_path(env, s, ctx, out, layer: int, label: str) -> None:
    """A decode-path output: finite, within the agreement of the unbounded workspace
    computation (bitwise in every full sub-batch), and the sampled rows within the fp32 band."""
    lat = out[: s.tc]
    assert bool(torch.isfinite(lat).all()), f"{label}: a context row read an unwritten pool byte"
    ref = _unbounded(env, s, layer)
    rel, differ = _agreement(lat, ref)
    rows_differ = _rows_differing(lat, ref, ctx.sub_batches)
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


def _check_workspace_path(env, s, out, layer: int, label: str) -> None:
    """A workspace-path output: bitwise each context request's rows through the same kernel over
    that request's own gather (its own offsets, no packing), and the sampled rows within the
    fp32 band."""
    lat = out[: s.tc]
    assert bool(torch.isfinite(lat).all()), f"{label}: a context row read an unwritten pool byte"
    start, differ = 0, 0
    for r, (_, new) in enumerate(s.contexts):
        kv = _request_kv(env, s, _SOURCES[layer], r)
        own = torch.empty(new, _HEADS, _NOPE, dtype=torch.bfloat16, device=env.dev)
        flash_mla_sparse_prefill(
            s.q[start : start + new], kv, s.topk[start : start + new], _SCALE, own
        )
        differ += int((lat[start : start + new].view(torch.int16) != own.view(torch.int16)).sum())
        start += new
    err = _fp32_check(env, s, _SOURCES[layer], out, _sample(s.tc, layer, (0, s.tc - 1)))
    print(f"CTXBOUND {label} layer {layer}: workspace path, N {sum(s.kv_lens[: len(s.contexts)])}"
          f", {differ} elements differ from each request alone; fp32 rel L2 {err:.3e}")  # fmt: skip
    assert differ == 0, (label, differ)
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
    The 32767 shared rows' outputs are bitwise the same, as every full sub-batch's rows are
    (`_check_decode_path`)."""
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
    rows_differ = _rows_differing(out_at[: at.tc], out_past[: at.tc], ctx_past.sub_batches)
    print(f"CTXBOUND N = B (workspace) against N = B + 1 (decode) on the {at.tc} shared rows: "
          f"rel L2 {rel:.3e}, {differ} elements differ, rows differing per sub-batch "
          f"{rows_differ}")  # fmt: skip
    assert rel <= _AGREE and differ == 0, (rel, differ)
    err = _fp32_check(env, at, 1, out_at, _sample(at.tc, 7, (0, 16383, 16384, at.tc - 1)))
    assert err < _FP32_BAND, err
    # The workspace holds every request's rows, request after request.
    off = 0
    for r, n in enumerate(at.kv_lens):
        assert torch.equal(env.buf[off : off + n], _request_kv(env, at, 1, r)), r
        off += n


def test_last_chunks_of_a_1m_prompt(env):
    """A 1M prompt's last full chunk, 32768 rows behind 1015808 cached keys: four sub-batches of
    8192 on one scheduler the 72 layers share, its per-layer peak memory under the KV-capacity
    estimate's step on the workspace path, whose peak sizes the pool. The 1048559-token prompt's
    last chunk, 32751 rows behind 1015808 (1048559 KV rows): sub-batches 8192 x 3 + 8175, on two
    schedulers."""
    s = _step(env, [(_PREFIX_1M, 32768)], seed=3)
    gathered = len(env.gathers)
    ctx, out = _attend(env, s, 1)
    assert ctx.sub_batches == [(lo, lo + _SUB) for lo in range(0, 32768, _SUB)]
    assert sorted(ctx.sched) == [8192] and ctx.sched[8192].config.b == 8192
    assert ctx.sched[8192].config.h_q == _HEADS
    _check_decode_path(env, s, ctx, out, 1, "32768 behind 1015808")
    assert len(env.gathers) == gathered

    est = env.estimate
    lat_est = torch.empty(est.tc, _HEADS, _NOPE, dtype=torch.bfloat16, device=env.dev)
    lat_dec = torch.empty(s.tc, _HEADS, _NOPE, dtype=torch.bfloat16, device=env.dev)
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

    s = _step(env, [(_PREFIX_1M, 32751)], seed=4)
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


def test_several_requests_one_over_a_1m_prefix(env):
    """One rank's step of four requests (attention DP gives a rank up to 16 of its own): a chunk
    over a 1015808-token prefix beside three short ones, with three generation rows behind them.
    N is past B, so every context row takes the decode path, the short requests' rows too."""
    contexts = [(_PREFIX_1M, 16384), (4096, 8192), (0, 4096), (100, 4000)]
    s = _step(env, contexts, gens=(524288, 3001, 17), seed=5)
    gathered = len(env.gathers)
    ctx, out = _attend(env, s, 36)
    assert ctx.ws_buf is None and [hi - lo for lo, hi in ctx.sub_batches] == [8192] * 3 + [8096]
    _check_decode_path(env, s, ctx, out, 36, "four requests, one over 1015808")
    assert bool(torch.isnan(out[s.tc :].float()).all())
    assert len(env.gathers) == gathered


def test_packed_four_chunks_at_max_seq_len(env):
    """Four chunks of 16384 / 8192 / 4096 / 4096 rows packed behind 1032192 / 524288 / 0 / 2048
    cached keys (N = 1591296): the decode path, its per-layer time and peak memory."""
    contexts = [(1032192, 16384), (524288, 8192), (0, 4096), (2048, 4096)]
    s = _step(env, contexts, seed=6)
    ctx, out = _attend(env, s, 1)
    assert ctx.ws_buf is None and sum(s.kv_lens) == 1591296
    _check_decode_path(env, s, ctx, out, 1, "packed four chunks")
    lat = torch.empty(s.tc, _HEADS, _NOPE, dtype=torch.bfloat16, device=env.dev)
    L = SimpleNamespace(kv_source=1)
    ms, peak = _measure(lambda: env.target._attn_ctx(s.step, L, s.q, lat, s.req_idx, s.topk, ctx))
    print(f"CTXBOUND packed four-chunk step (cached 1032192/524288/0/2048, new "
          f"16384/8192/4096/4096, N {sum(s.kv_lens)}): decode path, {ms:.3f} ms per layer, "
          f"peak {peak / _GIB:.3f} GiB above the held workspace")  # fmt: skip


def test_mixed_steps_and_layer_35s_sharers(env):
    """Generation rows behind the context rows on every path: the context attention writes the
    context rows only, and the generation rows' own call (`_attn_gen`, one scheduler) reads the
    pool. Layer 39 attends over layer 35's rows after layer 36 gathered its own: from the kept
    rows where N fits B twice (5036 rows: 35 and 36 gathered once each), gathered again into the
    step's rows where it fits once (300940 rows: 35, 36 and 39 each gather), and from the pool on
    the decode path; each time bitwise layer 35's output."""
    target = env.target
    for label, contexts, gens in (
        ("decode", [(_PREFIX_1M, 16384)], (524288, 3001, 17)),
        ("kept", [(4096, 640), (0, 300)], (3001, 17)),
        ("gathered again", [(300000, 640), (0, 300)], (3001, 17)),
    ):
        s = _step(env, contexts, gens, seed=7)
        ctx = target._ctx_workspace(s.step, s.tc, env.dev)
        assert (ctx.ws_buf is None) == (label == "decode")
        if label != "decode":
            n_kv = sum(s.kv_lens[: len(contexts)])
            assert n_kv <= _B and (ctx.kept is not None) == (label == "kept") == (2 * n_kv <= _B)
        gathered = len(env.gathers)
        outs = {}
        for layer in (35, 36, 39):
            ctx, out = _attend(env, s, layer, ctx=ctx)
            assert bool(torch.isnan(out[s.tc :].float()).all()), (label, layer)
            if label != "decode":
                # The source's rows, request after request: in the kept rows for layer 35 on a
                # step that keeps them, else in the step's own rows.
                src = _SOURCES[layer]
                rows = ctx.ws.get(src, ctx.ws_buf)
                assert (rows is ctx.kept) == (label == "kept" and src == 35), (label, layer)
                off = 0
                for r in range(len(contexts)):
                    n = s.kv_lens[r]
                    assert torch.equal(rows[off : off + n], _request_kv(env, s, src, r))
                    off += n
            slots = convert_req_index_to_global(
                s.req_idx[s.tc :], s.table, s.topk[s.tc :], _TPB, _TOPK,
                len(_OWNERS) * _TPB, target.core._pool_row[_SOURCES[layer]],
            )  # fmt: skip
            L = SimpleNamespace(kv_source=_SOURCES[layer])
            target._attn_gen(s.step, L, s.q[s.tc :], out[s.tc :], slots, get_mla_metadata()[0])
            outs[layer] = out
        # On the workspace path one gather per owner, and the sharer's own where nothing is
        # kept; none past B.
        assert len(env.gathers) - gathered == {"decode": 0, "kept": 2, "gathered again": 3}[label]
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
            _check_workspace_path(env, s, outs[35], 35, "mixed, layer 35")
            _check_workspace_path(env, s, outs[39], 39, "mixed, sharer 39")


def test_dense_step_takes_the_workspace(env):
    """A step the engine skipped the indexer for: sixteen requests each within index_topk (fresh,
    behind a cached prefix, and one-token tails), every row selecting its whole causal window.
    N is at most 16 x 2048, so the workspace path, layer 35's rows gathered into the kept rows
    after the step's own."""
    contexts = [(0, 2048)] * 4 + [(1024, 1024)] * 4 + [(2047, 1)] * 4 + [(0, 700), (300, 5)] * 2
    s = _step(env, contexts, seed=8, skip_ctx=True)
    assert all(c + n <= _TOPK for c, n in contexts) and len(contexts) == 16
    ctx, out = _attend(env, s, 35)
    assert ctx.ws_buf is not None and ctx.ws_buf.data_ptr() == env.buf.data_ptr()
    n_kv = sum(s.kv_lens)
    assert set(ctx.ws) == {35} and ctx.ws[35] is ctx.kept
    assert ctx.kept.data_ptr() == env.buf.data_ptr() + n_kv * _LAT_DIM * env.buf.element_size()
    _check_workspace_path(env, s, out, 35, "dense step")


def _short_ctx(env, s, n: int) -> SimpleNamespace:
    """The workspace path's context over only the first `n` of the step's N KV rows: what an N
    computed short would hand the gather and the prefill kernel."""
    cu = s.step["cu_ctx_kv"]
    lens = cu[1:] - cu[:-1]
    kv_req = torch.repeat_interleave(torch.arange(len(s.contexts), device=env.dev), lens)
    kv_pos = torch.arange(kv_req.numel(), device=env.dev) - cu[:-1][kv_req]
    return SimpleNamespace(
        kv_req=kv_req[:n].to(torch.int32), kv_pos=kv_pos[:n].to(torch.int32), ws_off=cu[:-1],
        ws_buf=env.buf[:n], kept=None, ws={},
    )  # fmt: skip


def test_short_n_fails_the_harness(env):
    """The control: a workspace one row short of N, and one of the new tokens alone (the
    engine's `num_ctx_kv_tokens`), drop the selections at or past its end with no error and no
    NaN; the harness's checks refuse both."""
    s = _step(env, [(4096, 640), (0, 300)], seed=9)
    n = sum(s.kv_lens)
    for short in (n - 1, s.step["num_ctx_kv_tokens"]):
        ctx = _short_ctx(env, s, short)
        out = _attend(env, s, 1, ctx=ctx)[1]
        lat = out[: s.tc]
        sel = s.topk[: s.tc]
        idx = torch.where(sel >= 0, sel + ctx.ws_off[s.req_idx[: s.tc].long()][:, None], sel)
        dropped = int((idx >= short).sum())
        print(f"CTXBOUND control: N {n}, workspace of {short} rows: {dropped} selections dropped, "
              f"output finite {bool(torch.isfinite(lat).all())}")  # fmt: skip
        assert dropped > 0 and not bool(torch.isnan(lat.float()).any())
        with pytest.raises(AssertionError):
            _check_workspace_path(env, s, out, 1, f"short {short}")


def test_workspace_held_and_no_gather_past_b(env):
    """After every step above and a sequence of its own (sixteen requests filling B exactly,
    then B + 1, then a 1M chunk), the workspace is the estimate's step's allocation, every
    gather wrote into it -- the step's first N rows, or layer 35's into the N after them on a
    step that fits twice -- and no step past B reached the gather."""
    for contexts, path in (
        ([(30720, 2048)] * 16, "workspace"),
        ([(30720, 2048)] * 15 + [(30721, 2048)], "decode"),
        ([(_PREFIX_1M, 32768)], "decode"),
        ([(5, 300)], "workspace"),
    ):
        s = _step(env, contexts, seed=10)
        gathered = len(env.gathers)
        ctx, out = _attend(env, s, 36)
        assert (ctx.ws_buf is None) == (path == "decode"), (contexts[0], path)
        assert len(env.gathers) - gathered == (path == "workspace")
        assert bool(torch.isfinite(out[: s.tc]).all())
    assert env.target._ws_buf is env.buf
    assert env.gathers and all(x.rows <= _B for x in env.gathers)
    base, row_bytes = env.buf.data_ptr(), _LAT_DIM * env.buf.element_size()
    kept = [x for x in env.gathers if x.out_ptr != base]
    assert all(x.out_ptr == base + x.rows * row_bytes for x in kept)
    assert all(2 * x.rows <= _B and x.pool_row == _OWNERS.index(35) for x in kept)
    print(f"CTXBOUND {len(env.gathers)} gathers, at most {max(x.rows for x in env.gathers)} "
          f"rows, all into the workspace allocated on the estimate's step")  # fmt: skip
