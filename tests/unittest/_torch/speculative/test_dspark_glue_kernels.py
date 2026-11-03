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
"""The DSpark glue-fusion kernels must match the eager chains they replace bit for bit."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import input_commit
from tensorrt_llm._torch.speculative import dspark_glue_kernels as glue
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEV = "cuda"


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(view), b.contiguous().view(view))
    return torch.equal(a, b)


class _Worker(SpecWorkerBase):
    def __init__(self, draft_len: int):
        super().__init__(False)
        self._draft_len = draft_len

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def max_draft_len(self) -> int:
        return self._draft_len


def _spec_metadata(is_cuda_graph: bool = False):
    return SimpleNamespace(
        is_all_greedy_sample=True,
        is_ragged_verify=False,
        is_cuda_graph=is_cuda_graph,
        enable_penalty=False,
        use_rejection_sampling=False,
    )


def _set_glue(monkeypatch, on: bool):
    # The fusion runs on drafting Kimi K3.1 engines without attention DP.
    monkeypatch.setattr(input_commit, "_K31_DRAFTS", on)
    monkeypatch.setattr(input_commit, "_K31_ATTENTION_DP", False)


def test_glue_fusion_pieces(monkeypatch):
    _set_glue(monkeypatch, False)
    assert not glue.glue_fusion_enabled() and not glue.glue_fusion_enabled("logits")
    _set_glue(monkeypatch, True)
    monkeypatch.setenv(glue.GLUE_FUSION_DISABLE_ENV_VAR, "kda, next")
    assert glue.glue_fusion_enabled() and glue.glue_fusion_enabled("logits")
    assert not glue.glue_fusion_enabled("kda") and not glue.glue_fusion_enabled("next")
    # The DSA metadata pieces follow the same switch and the same bisection list.
    assert glue.glue_fusion_enabled("spec_meta") and glue.glue_fusion_enabled("pre_meta")
    monkeypatch.setenv(glue.GLUE_FUSION_DISABLE_ENV_VAR, "spec_meta,pre_meta")
    assert not glue.glue_fusion_enabled("spec_meta") and not glue.glue_fusion_enabled("pre_meta")
    assert {"spec_meta", "pre_meta", "window", "embed"} <= set(glue.GLUE_FUSION_PIECES)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "tp,rows,shard", [(8, 112, 20480), (8, 28, 20480), (16, 24, 10240), (2, 3, 1000)]
)
def test_vocab_shards_to_fp32(tp, rows, shard, dtype):
    """The dim-0 all-gather of per-rank shards -> the dim -1 gather's ``.float()`` rows."""
    shards = [(torch.randn(rows, shard, device=DEV) * 8).to(dtype) for _ in range(tp)]
    shards[0][0, :4] = torch.tensor([float("inf"), float("nan"), -0.0, 1e-40], device=DEV).to(dtype)
    expected = torch.cat(shards, dim=-1).float()
    assert _bit_equal(glue.vocab_shards_to_fp32(torch.cat(shards, dim=0), tp), expected)


@pytest.mark.parametrize("batch,num_ctx", [(16, 0), (4, 0), (5, 2), (33, 0), (128, 0)])
@pytest.mark.parametrize("force", [2.75, 2.62, 3.0, 9.0])
@pytest.mark.parametrize("col_stride", [1, 2])
def test_strict_accept_forced(batch, num_ctx, force, col_stride, monkeypatch):
    """Strict acceptance + the forced override, through SpecWorkerBase, fused vs eager:
    accepted rows, counts and the forced draw's device counter after three steps."""
    draft_len, vocab = 6, 512
    num_gens = batch - num_ctx
    torch.manual_seed(batch)
    logits = torch.randn(num_ctx + num_gens * (draft_len + 1), vocab, device=DEV)
    target = torch.argmax(logits, dim=-1)
    draft = target[num_ctx:].view(num_gens, draft_len + 1)[:, :draft_len].to(torch.int32)
    cut = torch.randint(0, draft_len + 1, (num_gens,), device=DEV)
    cols = torch.arange(draft_len, device=DEV)
    draft = torch.where(cols[None, :] >= cut[:, None], (draft + 1) % vocab, draft)
    if col_stride > 1:  # a strided view, as a sliced draft buffer is
        wide = torch.zeros(num_gens, draft_len * col_stride, dtype=torch.int32, device=DEV)
        wide[:, ::col_stride] = draft
        draft = wide[:, ::col_stride]
    else:
        draft = draft.contiguous()
    results = []
    for fused in (False, True):
        _set_glue(monkeypatch, fused)
        worker = _Worker(draft_len)
        worker.force_num_accepted_tokens = force
        steps = [
            worker._sample_and_accept_draft_tokens_base(
                logits, draft, num_ctx, batch, _spec_metadata()
            )
            for _ in range(3)  # the forced draw's device counter advances in place
        ]
        results.append((steps, worker._force_accept_rng_counter))
    for (acc0, n0), (acc1, n1) in zip(results[0][0], results[1][0]):
        assert _bit_equal(n1, n0)
        assert _bit_equal(acc1[num_ctx:], acc0[num_ctx:])
        assert _bit_equal(acc1[:num_ctx, 0], acc0[:num_ctx, 0])
    if results[0][1] is not None:
        assert _bit_equal(results[1][1], results[0][1])


def test_strict_accept_natural_keeps_her_kernel(monkeypatch):
    """Without forced acceptance the glue fusion leaves the acceptance as it is."""
    draft_len, batch = 6, 8
    logits = torch.randn(batch * (draft_len + 1), 256, device=DEV)
    draft = torch.argmax(logits, -1).view(batch, -1)[:, :draft_len].to(torch.int32).contiguous()
    outs = []
    for fused in (False, True):
        _set_glue(monkeypatch, fused)
        worker = _Worker(draft_len)
        worker.force_num_accepted_tokens = 0.0
        outs.append(
            worker._sample_and_accept_draft_tokens_base(logits, draft, 0, batch, _spec_metadata())
        )
    assert _bit_equal(outs[0][0], outs[1][0]) and _bit_equal(outs[0][1], outs[1][1])


@pytest.mark.parametrize(
    "batch,num_ctx,num_dummy", [(16, 0, 0), (16, 0, 3), (8, 2, 2), (128, 0, 5)]
)
def test_kda_record_acceptance(batch, num_ctx, num_dummy):
    """KDAReplayState.record_acceptance via the V2 manager's update_mamba_states, on a strided
    per-slot count view (one int32 of a state page per slot) and a batch-row dummy mask."""
    num_gens = batch - num_ctx
    pages = torch.randint(0, 7, (700, 16), dtype=torch.int32, device=DEV)
    state_indices = torch.randperm(600, device=DEV)[: 2 * batch].to(torch.int32)[::2]
    num_accepted = torch.randint(1, 8, (batch,), dtype=torch.int32, device=DEV)
    dummy = torch.zeros(batch + 7, dtype=torch.bool, device=DEV)
    if num_dummy:
        dummy[batch - num_dummy : batch] = True
    expected_pages, actual_pages = pages.clone(), pages.clone()
    expected, actual = expected_pages[:, 0], actual_pages[:, 0]
    # MambaHybridCacheManagerV2.update_mamba_states -> KDAReplayState.update/record_acceptance
    slots = state_indices[num_ctx:batch].to(torch.int32).to(torch.long)
    accepted = (num_accepted[num_ctx:batch] - 1).to(torch.int32).clamp(min=0)
    accepted = torch.where(dummy[num_ctx:batch], expected[slots], accepted)
    expected[slots] = accepted
    glue.kda_record_acceptance(actual, state_indices, num_accepted, num_ctx, num_gens, dummy)
    assert _bit_equal(actual_pages, expected_pages)


@pytest.mark.parametrize("batch,num_ctx", [(16, 0), (6, 2), (128, 0)])
def test_kv_lens_prepare_rewind_and_next_tokens(batch, num_ctx):
    """DFlashWorker._prepare_kv_for_draft_forward's KV-length update, then
    _apply_kv_rewind_after_draft + _prepare_next_new_tokens, on strided kv_lens."""
    draft_len = 6
    kv_pages = torch.randint(0, 200_000, (160, 3), dtype=torch.int32, device=DEV)
    num_accepted = torch.randint(1, draft_len + 2, (batch,), dtype=torch.int32, device=DEV)
    kv_ref_pages, kv_fused_pages = kv_pages.clone(), kv_pages.clone()
    kv_ref, kv_fused = kv_ref_pages[:, 1], kv_fused_pages[:, 1]
    rewind_ref = 1 - num_accepted[num_ctx:batch]
    kv_ref[num_ctx:batch] += 1
    rewind = glue.kv_prepare_for_draft(kv_fused, num_accepted, num_ctx, batch)
    assert _bit_equal(rewind, rewind_ref) and _bit_equal(kv_fused, kv_ref)
    # Rewinds past zero clamp.
    kv_ref[num_ctx] = 0
    kv_fused[num_ctx] = 0

    accepted = torch.randint(0, 160_000, (batch, draft_len + 1), dtype=torch.int32, device=DEV)
    next_draft = torch.randint(0, 160_000, (batch, draft_len), dtype=torch.int32, device=DEV)
    batch_indices = torch.arange(2 * batch, dtype=torch.int32, device=DEV)[::2] // 2
    kv_ref[num_ctx:batch] -= rewind_ref
    kv_ref[num_ctx:batch].clamp_(min=0)
    last = accepted[batch_indices[:batch], num_accepted - 1].unsqueeze(1)
    expected = torch.concat([last, next_draft], dim=1)
    actual = glue.next_new_tokens_and_rewind(
        accepted,
        next_draft,
        batch_indices,
        num_accepted,
        batch,
        kv_lens=kv_fused,
        rewind=rewind,
        num_contexts=num_ctx,
    )
    assert _bit_equal(actual, expected) and _bit_equal(kv_fused_pages, kv_ref_pages)
    # Without the rewind (warmup), only the tokens.
    before = kv_fused_pages.clone()
    actual = glue.next_new_tokens_and_rewind(
        accepted, next_draft, batch_indices, num_accepted, batch
    )
    assert _bit_equal(actual, expected) and _bit_equal(kv_fused_pages, before)


def test_forced_accept_replays_in_cuda_graph(monkeypatch):
    """The fused forced acceptance replays inside a CUDA graph with the counter advancing per
    replay, like the eager override."""
    draft_len, batch = 6, 8
    logits = torch.randn(batch * (draft_len + 1), 256, device=DEV)
    draft = torch.argmax(logits, -1).view(batch, -1)[:, :draft_len].to(torch.int32).contiguous()
    outs = []
    for fused in (False, True):
        _set_glue(monkeypatch, fused)
        worker = _Worker(draft_len)
        worker.force_num_accepted_tokens = 2.75
        worker._ensure_force_accept_rng_state(torch.device(DEV))
        md = _spec_metadata(is_cuda_graph=True)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            _, num_accepted = worker._sample_and_accept_draft_tokens_base(
                logits, draft, 0, batch, md
            )
        replays = []
        for _ in range(4):
            graph.replay()
            replays.append(num_accepted.clone())
        outs.append(torch.stack(replays))
    assert _bit_equal(outs[1], outs[0])
    assert not bool((outs[0] == outs[0][0]).all())  # the draws differ between replays


def _window_fixup_operands(batch, block, page, cols, ctx_lens, table_dtype, row_pad, seed):
    """A block-decode fixup of ``batch`` requests over ``cols``-page rows (a column slice of a
    wider table when ``row_pad``), the lengths and key bounds of ``ctx_lens`` + the block."""
    from tensorrt_llm._torch.models import modeling_dspark

    gen = torch.Generator(device=DEV).manual_seed(seed)
    wide = torch.randint(0, 1 << 20, (batch, cols + row_pad), generator=gen, device=DEV)
    table = wide.to(table_dtype)[:, :cols]
    ctx = torch.tensor(ctx_lens, dtype=torch.int64, device=DEV)
    seq_lens = (ctx + block).to(torch.int32)
    kv_bounds = (
        seq_lens.repeat_interleave(block)
        - torch.randint(0, block, (batch * block,), generator=gen, device=DEV)
    ).to(torch.int32)
    fixup = modeling_dspark._MLABlockFixup(
        pages=torch.zeros(batch, block, dtype=torch.int64, device=DEV),
        offsets=torch.zeros(batch, block, dtype=torch.int64, device=DEV),
        valid=None,
        page_tables_i32=table,
        seq_lens_i32=seq_lens,
        kv_bounds_i32=kv_bounds,
    )
    return fixup, ctx


@pytest.mark.parametrize("table_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("row_pad", [0, 3])
@pytest.mark.parametrize(
    "window,ctx_lens,cols",
    [
        (65536, [150000], 2400),  # B1 AgentX: skip ~1320 pages
        (65536, [150000, 40000, 65536, 65600, 0], 2400),  # inside / at / just past the window
        (32768, [140123, 131072, 99, 200000, 64, 7, 120000, 32769], 3200),
        (4096, [300000, 5000], 1100),  # start past the row: every column clamps to the last page
    ],
)
@pytest.mark.parametrize(
    "ctx_given,cp_size,cp_rank",
    [(False, 1, 0), (True, 1, 0), (True, 4, 1), (True, 4, 3), (True, 8, 0)],
)
def test_window_block_fixup_matches_eager(
    table_dtype, row_pad, window, ctx_lens, cols, ctx_given, cp_size, cp_rank, monkeypatch
):
    from tensorrt_llm._torch.models import modeling_dspark

    block, page = 8, 64
    fixup, ctx = _window_fixup_operands(
        len(ctx_lens),
        block,
        page,
        cols,
        ctx_lens,
        table_dtype,
        row_pad,
        seed=len(ctx_lens) * 7 + cp_rank,
    )
    ctx_len = ctx.to(torch.int32) if ctx_given else None
    args = dict(ctx_len=ctx_len, cp_size=cp_size, cp_rank=cp_rank) if ctx_given else {}
    _set_glue(monkeypatch, False)
    ref = modeling_dspark._window_mla_block_fixup(fixup, window, block, page, **args)
    _set_glue(monkeypatch, True)
    got = modeling_dspark._window_mla_block_fixup(fixup, window, block, page, **args)
    for name in ("page_tables_i32", "seq_lens_i32", "kv_bounds_i32"):
        assert _bit_equal(getattr(got, name), getattr(ref, name)), name
    # The block's own positions are untouched.
    assert got.pages is fixup.pages and got.offsets is fixup.offsets
    monkeypatch.setenv(glue.GLUE_FUSION_DISABLE_ENV_VAR, "window")
    eager = modeling_dspark._window_mla_block_fixup(fixup, window, block, page, **args)
    assert _bit_equal(eager.page_tables_i32, ref.page_tables_i32)


def test_window_block_fixup_falls_back_on_other_layouts(monkeypatch):
    """Operands the kernel does not take (a transposed table, int64 lengths) keep the eager
    chain, with the same result."""
    from tensorrt_llm._torch.models import modeling_dspark

    block, page = 8, 64
    fixup, _ = _window_fixup_operands(3, block, page, 900, [70000, 1000, 150000], torch.int32, 0, 5)
    assert (
        glue.window_block_fixup(
            fixup.page_tables_i32.t().contiguous().t(),
            fixup.seq_lens_i32,
            fixup.kv_bounds_i32,
            32768,
            block,
            page,
        )
        is None
    )
    assert (
        glue.window_block_fixup(
            fixup.page_tables_i32,
            fixup.seq_lens_i32.long(),
            fixup.kv_bounds_i32,
            32768,
            block,
            page,
        )
        is None
    )
    _set_glue(monkeypatch, False)
    ref = modeling_dspark._window_mla_block_fixup(fixup, 32768, block, page)
    _set_glue(monkeypatch, True)
    odd = fixup._replace(seq_lens_i32=fixup.seq_lens_i32.long())
    got = modeling_dspark._window_mla_block_fixup(odd, 32768, block, page)
    assert _bit_equal(got.page_tables_i32, ref.page_tables_i32)
    assert _bit_equal(got.seq_lens_i32, ref.seq_lens_i32)
    assert _bit_equal(got.kv_bounds_i32, ref.kv_bounds_i32)


def test_window_block_fixup_in_cuda_graph(monkeypatch):
    """Captured once, the fused fixup follows new lengths and tables on every replay."""
    from tensorrt_llm._torch.models import modeling_dspark

    block, page, window, cols = 8, 64, 65536, 2400
    fixup, _ = _window_fixup_operands(
        4, block, page, cols, [150000, 66000, 100, 140000], torch.int32, 0, seed=11
    )
    _set_glue(monkeypatch, True)
    modeling_dspark._window_mla_block_fixup(fixup, window, block, page)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = modeling_dspark._window_mla_block_fixup(fixup, window, block, page)
    for step, lens in enumerate(
        ([150008, 66008, 108, 140008], [9, 65544, 70000, 153000], [130000, 1, 2, 3])
    ):
        new, _ = _window_fixup_operands(4, block, page, cols, lens, torch.int32, 0, seed=20 + step)
        fixup.page_tables_i32.copy_(new.page_tables_i32)
        fixup.seq_lens_i32.copy_(new.seq_lens_i32)
        fixup.kv_bounds_i32.copy_(new.kv_bounds_i32)
        graph.replay()
        _set_glue(monkeypatch, False)
        ref = modeling_dspark._window_mla_block_fixup(new, window, block, page)
        _set_glue(monkeypatch, True)
        torch.cuda.synchronize()
        for name in ("page_tables_i32", "seq_lens_i32", "kv_bounds_i32"):
            assert _bit_equal(getattr(got, name), getattr(ref, name)), (step, name)


def test_noise_embedding_takes_the_embed_site(monkeypatch):
    """Glue piece "embed": the noise block's rows come from the owner-multicast embedding site
    (bitwise the vocab-sharded lookup + all-reduce: multi_gpu/test_dspark_draft_ar.py); where the
    site does not apply, or the piece is disabled, the lookup path gives the same block."""
    from tensorrt_llm._torch.speculative import dflash

    hidden, block = 64, 8
    table = torch.randn(100, hidden, device=DEV).to(torch.bfloat16)
    embed = torch.nn.Embedding.from_pretrained(table)
    calls = []

    def site(embedding, ids):
        calls.append(ids.clone())
        return None if site.decline else embedding(ids)

    monkeypatch.setattr(dflash, "draft_embed_site", site)
    bonus = torch.tensor([3, 41, 7], device=DEV)
    _set_glue(monkeypatch, False)
    site.decline = False
    ref = dflash.dflash_noise_block_embedding(embed, bonus, 99, block)
    assert not calls
    _set_glue(monkeypatch, True)
    got = dflash.dflash_noise_block_embedding(embed, bonus, 99, block)
    assert len(calls) == 1 and calls[0].tolist() == [3, 41, 7, 99]
    assert _bit_equal(got, ref)
    site.decline = True
    assert _bit_equal(dflash.dflash_noise_block_embedding(embed, bonus, 99, block), ref)
    monkeypatch.setenv(glue.GLUE_FUSION_DISABLE_ENV_VAR, "embed")
    calls.clear()
    assert _bit_equal(dflash.dflash_noise_block_embedding(embed, bonus, 99, block), ref)
    assert not calls
