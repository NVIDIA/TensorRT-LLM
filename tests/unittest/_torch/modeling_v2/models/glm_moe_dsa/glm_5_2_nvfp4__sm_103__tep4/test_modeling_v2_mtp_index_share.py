# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MTP draft loop's index share (`mtp_index_share`) on the GLM-5.2 target, against the engine.

With the engine's flag on (the LLM arg `index_share_for_mtp_iteration`, else the checkpoint's
field; GLM-5.2's config.json sets it true), the engine's DSA indexer stashes draft step 0's
selection, one row per request (a context request's last row, a generation request's last
accepted row), and draft steps 1.. copy it instead of scoring and selecting. What this pins:

* the flag resolves as the engine's attention module resolves the MTP layer's sparse params;
* the stash is the engine's own rows (`Indexer._mtp_last_accepted_rows` plus the context last
  rows), with and without accepted counts, decode and mixed steps;
* the target's MTP layer, over a whole draft loop on a fake core: with the flag on, step 0 runs
  the indexer and stashes, steps 1.. run no q side / scorer / top-k, still project and append
  their keys, and attend over the stashed rows; with it off, every step runs the indexer as
  before (the same calls, in the same order); the two give the same outputs when the indexer
  would select step 0's rows again, and a dense (skipped) step is untouched either way.

Dispatch and index arithmetic only: no kernel runs.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the GLM-5.2 target imports sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer  # noqa: E402
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig  # noqa: E402

TOPK = 2048
#: The MTP layer's index: one past the trunk's 78 layers.
NUM_LAYERS = 78


def _glm_config(share) -> SimpleNamespace:
    """The checkpoint fields `to_sparse_params` reads (GLM-5.2 NVFP4 config.json)."""
    cfg = SimpleNamespace(
        num_hidden_layers=NUM_LAYERS,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=TOPK,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_topk_pattern=None,
    )
    if share is not None:
        cfg.index_share_for_mtp_iteration = share
    return cfg


@pytest.mark.parametrize(
    "arg, ckpt, want",
    [(None, True, True), (None, False, False), (None, None, False), (False, True, False),
     (True, None, True)],
)  # fmt: skip
def test_flag_resolves_as_the_engine(arg, ckpt, want) -> None:
    sparse = DeepSeekSparseAttentionConfig(index_share_for_mtp_iteration=arg)
    model_config = SimpleNamespace(
        sparse_attention_config=sparse, pretrained_config=_glm_config(ckpt)
    )
    assert M._engine_mtp_index_share(model_config) is want
    # The engine's own MTP-layer params on the same inputs.
    engine = sparse.to_sparse_params(pretrained_config=_glm_config(ckpt), layer_idx=NUM_LAYERS)
    assert engine.mtp_index_share is want
    assert M._engine_mtp_index_share(SimpleNamespace(sparse_attention_config=None)) is False


def _engine_stash(md, topk, ctx, tc, seqs, gen_p) -> torch.Tensor:
    """The rows the engine's indexer writes to `shared_topk_indices` (sparse_attn_indexer's
    tail): context last rows, then `Indexer._mtp_last_accepted_rows` (self unused)."""
    rows = []
    if ctx:
        last = torch.cumsum(md.seq_lens_cuda[:ctx].to(torch.long), dim=0) - 1
        rows.append(topk[last])
    if seqs:
        meta = SimpleNamespace(mtp_num_accepted=md.mtp_num_accepted)
        rows.append(Indexer._mtp_last_accepted_rows(None, topk[tc:], meta, ctx, seqs, gen_p))
    return torch.cat(rows)


@pytest.mark.parametrize(
    "ctx_lens, seqs, gen_p, accepted",
    [
        ([], 1, 6, [4]),
        ([], 3, 6, [1, 6, 3]),
        ([], 2, 4, None),
        ([], 32, 6, [1 + (7 * i) % 6 for i in range(32)]),
        ([5, 1, 9], 2, 6, [9, 9, 9, 2, 6]),  # mixed: context counts are not read
        ([7], 0, 1, [7]),
    ],
)
def test_stash_is_the_engine_rows(ctx_lens, seqs, gen_p, accepted) -> None:
    dev = torch.device("cuda")
    ctx, tc = len(ctx_lens), sum(ctx_lens)
    rows = tc + seqs * gen_p
    g = torch.Generator(device=dev).manual_seed(rows)
    topk = torch.randint(-1, 1 << 20, (rows, TOPK), dtype=torch.int32, device=dev, generator=g)
    md = SimpleNamespace(
        seq_lens_cuda=torch.tensor(ctx_lens + [gen_p] * seqs, dtype=torch.int32, device=dev),
        mtp_num_accepted=None
        if accepted is None
        else torch.tensor(accepted, dtype=torch.int32, device=dev),
        shared_topk_indices=torch.full((64, TOPK), -7, dtype=torch.int32, device=dev),
    )
    M._mtp_share_store(md, topk, ctx, tc, seqs, gen_p)
    want = _engine_stash(md, topk, ctx, tc, seqs, gen_p)
    n = ctx + seqs
    assert torch.equal(md.shared_topk_indices[:n], want)
    assert bool((md.shared_topk_indices[n:] == -7).all())
    # A later step's selection is the stash's first `n` rows.
    out = torch.empty((n, TOPK), dtype=torch.int32, device=dev)
    M._mtp_share_load(md, out, n)
    assert torch.equal(out, want)


# ---- the MTP layer over a draft loop, on a fake core ----------------------------------------


class _Loop:
    """A fake core and a draft loop's metadata. `fresh(step)` is the selection the indexer
    would make at `step` (one row per query row); `calls` records the indexer front."""

    def __init__(self, share: bool, max_draft_len: int, seqs: int, fresh, skip_gen=False):
        self.calls: list[tuple] = []
        self.slots: list[torch.Tensor] = []
        self.fresh = fresh
        self.step = 0
        core = SimpleNamespace(
            num_layers=NUM_LAYERS,
            max_draft_len=max_draft_len,
            mtp_index_share=share,
            _mtp_share_stash=share and max_draft_len > 1,
            _attn={NUM_LAYERS: "attn"},
            _idx={NUM_LAYERS: "ix"},
            route_hits={},
            route_misses={},
        )
        core._count = lambda name, hit: core.route_hits.__setitem__(
            name, core.route_hits.get(name, 0) + 1
        )
        core._qkva_ix_fold = lambda x, i, rows: None
        core._qb_ixqb_runner = lambda i, rows: self.calls.append(("qb_ixqb", rows)) or None
        core._qkv = lambda x, a, q_b=True, qkva=None: ("q", "q_a", "latent", None)

        def project(x, q_a, pos, ix, rows, need_q, kw=None, ixq=None, kc=None):
            self.calls.append(("project", rows, need_q))
            k_pre = ("k", rows)
            return ("q_fp8", "w", k_pre) if need_q else (None, None, k_pre)

        core._indexer_project = project
        core._indexer_write_k = lambda md, k_pool, k_pre, rows: self.calls.append(("write_k", rows))

        def decode(md, k_pool, q_fp8, weights, topk_buf, ctx, seqs_, gen_p, memo=None):
            self.calls.append(("select", seqs_, gen_p))
            topk_buf.copy_(self.fresh(self.step, seqs_, gen_p))

        core._indexer_decode = decode
        core._dec8_takes = lambda *a: False
        core._dec8w_takes = lambda *a: False
        core._attn_gen = lambda *a: None
        self.core = core
        self.target = M.DecodeTarget(core)
        self.md = SimpleNamespace(
            in_mtp_draft_loop=True,
            indexer_skip_topk=False,
            mtp_num_accepted=None,
            shared_topk_indices=torch.full((64, TOPK), -7, dtype=torch.int32),
            kv_cache_manager=SimpleNamespace(get_indexer_k_cache_buffers=lambda i: "k_pool"),
        )
        self.seqs = seqs
        self.skip_gen = skip_gen

    def run(self, accepted: list[int]) -> None:
        """The engine's loop: step 0 at `max_draft_len + 1` rows per request, then 1 row."""
        md, d = self.md, self.core.max_draft_len
        md.mtp_num_accepted = torch.tensor(accepted, dtype=torch.int32)
        for step in range(d):
            self.step = step
            md.indexer_skip_topk = step > 0
            gen_p = d + 1 if step == 0 else 1
            rows = self.seqs * gen_p
            topk_buf = (
                torch.full((rows, TOPK), -3, dtype=torch.int32)
                if self.skip_gen
                else torch.empty((rows, TOPK), dtype=torch.int32)
            )

            def slots_of(buf, fresh):
                self.slots.append(buf.clone())
                return buf, "global", None

            self.target._layer_attention(
                md, "x", "pos", "out", NUM_LAYERS, 0, slots_of, topk_buf, rows, gen_p,
                self.skip_gen, False,
            )  # fmt: skip
        md.in_mtp_draft_loop = False


def _fresh_by_step(step: int, seqs: int, gen_p: int) -> torch.Tensor:
    """A selection that changes with the step and the row."""
    rows = seqs * gen_p
    base = torch.arange(rows, dtype=torch.int32)[:, None] * 10_000 + step * 1_000_000
    return base + torch.arange(TOPK, dtype=torch.int32)[None, :]


def _fresh_stable(step: int, seqs: int, gen_p: int) -> torch.Tensor:
    """A selection that depends on the request only: what step 0 selected for a request's
    rows, a later step selects again."""
    req = torch.arange(seqs, dtype=torch.int32).repeat_interleave(gen_p)
    return req[:, None] * 10_000 + torch.arange(TOPK, dtype=torch.int32)[None, :]


@pytest.mark.parametrize("max_draft_len, seqs", [(5, 1), (5, 3), (3, 2)])
def test_off_path_runs_the_indexer_every_step(max_draft_len, seqs) -> None:
    loop = _Loop(False, max_draft_len, seqs, _fresh_by_step)
    loop.run([2] * seqs)
    want = []
    for step in range(max_draft_len):
        gen_p = max_draft_len + 1 if step == 0 else 1
        rows = seqs * gen_p
        want += [("qb_ixqb", rows), ("project", rows, True), ("write_k", rows),
                 ("select", seqs, gen_p)]  # fmt: skip
        assert torch.equal(loop.slots[step], _fresh_by_step(step, seqs, gen_p))
    assert loop.calls == want
    assert bool((loop.md.shared_topk_indices == -7).all())
    assert "mtp_ix_share" not in loop.core.route_hits


@pytest.mark.parametrize("max_draft_len, seqs", [(5, 1), (5, 3), (3, 2)])
def test_on_path_reuses_draft_step_0(max_draft_len, seqs) -> None:
    accepted = [1 + (2 * s + 1) % (max_draft_len + 1) for s in range(seqs)]
    loop = _Loop(True, max_draft_len, seqs, _fresh_by_step)
    loop.run(accepted)
    p0 = max_draft_len + 1
    step0 = _fresh_by_step(0, seqs, p0)
    last = torch.stack([step0[s * p0 + accepted[s] - 1] for s in range(seqs)])
    want = [("qb_ixqb", seqs * p0), ("project", seqs * p0, True), ("write_k", seqs * p0),
            ("select", seqs, p0)]  # fmt: skip
    # Later steps: keys only (no q_b | wq_b fusion, no q side, scorer or top-k).
    want += [("project", seqs, False), ("write_k", seqs)] * (max_draft_len - 1)
    assert loop.calls == want
    assert torch.equal(loop.slots[0], step0)
    for step in range(1, max_draft_len):
        assert torch.equal(loop.slots[step], last), step
    assert torch.equal(loop.md.shared_topk_indices[:seqs], last)
    assert loop.core.route_hits["mtp_ix_share"] == max_draft_len - 1


@pytest.mark.parametrize("max_draft_len, seqs", [(5, 2), (3, 4)])
def test_on_and_off_agree_when_the_selection_is_stable(max_draft_len, seqs) -> None:
    accepted = [1 + s % (max_draft_len + 1) for s in range(seqs)]
    on, off = (_Loop(f, max_draft_len, seqs, _fresh_stable) for f in (True, False))
    on.run(accepted)
    off.run(accepted)
    assert len(on.slots) == len(off.slots) == max_draft_len
    for a, b in zip(on.slots, off.slots):
        assert torch.equal(a, b)


def test_mtp1_and_dense_steps() -> None:
    # MTP1: one draft step, nothing reads a stash, so none is written.
    loop = _Loop(True, 1, 2, _fresh_by_step)
    loop.run([1, 2])
    assert bool((loop.md.shared_topk_indices == -7).all())
    # A dense (skipped) generation phase: no selection anywhere; step 0 still stashes its
    # dense rows (the engine stashes after the dense copy too) and the later steps keep their
    # own dense rows, as the engine's reuse needs a generation phase that is not skipped.
    loop = _Loop(True, 3, 2, _fresh_by_step, skip_gen=True)
    loop.run([1, 4])
    assert [c for c in loop.calls if c[0] == "select"] == []
    assert bool((loop.md.shared_topk_indices[:2] == -3).all())
    assert all(bool((s == -3).all()) for s in loop.slots)
    assert "mtp_ix_share" not in loop.core.route_hits
