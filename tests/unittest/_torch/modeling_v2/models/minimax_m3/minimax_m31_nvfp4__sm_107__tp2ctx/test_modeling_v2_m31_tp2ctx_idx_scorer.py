# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_IDX_SCORER wiring (CPU): the tp2ctx target registers the scorer on every MSA indexer with an unset hook, keeps
an already-set one, honours the kill switch; the indexer keeps the proxy when no hook is set or the hook declines. The
plan: kept at the same lengths (rebuilt on new lengths) while every call re-gathers its block table from the page table
it is handed, so an in-place rewrite is seen, for an inference-mode page table too (no version counter is read)."""

from pathlib import Path
from types import SimpleNamespace

import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    idx_scorer,
)

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"


def _model(indexers):
    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(indexer=ix)))
        for ix in indexers
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def test_register_sets_unset_hooks_only(monkeypatch) -> None:
    monkeypatch.setattr(idx_scorer, "IDX_SCORER", True)
    mine = object()
    cfg = SimpleNamespace(num_kv_heads=2)
    a = SimpleNamespace(prefill_scorer_hook=None, config=cfg)
    b = SimpleNamespace(prefill_scorer_hook=mine, config=cfg)
    assert idx_scorer.register(_model([a, b, None])) == 1
    assert a.prefill_scorer_hook.func is idx_scorer.score
    assert a.prefill_scorer_hook.keywords == {"num_kv_heads": 2} and b.prefill_scorer_hook is mine


def test_kill_switch(monkeypatch) -> None:
    monkeypatch.setattr(idx_scorer, "IDX_SCORER", False)
    a = SimpleNamespace(prefill_scorer_hook=None)
    assert idx_scorer.register(_model([a])) == 0 and a.prefill_scorer_hook is None


def test_target_registers_with_a_kill_switch() -> None:
    src = (_TARGET / "modeling.py").read_text()
    assert "_idx_scorer.register(self)" in src
    assert (
        'IDX_SCORER = m31_enabled("TP2CTX_IDX_SCORER") and not m31_disabled("TP2CTX_IDX_SCORER")'
        in (_TARGET / "idx_scorer.py").read_text()
    )


def test_engagement_lines_have_distinct_prefixes() -> None:
    text = (_TARGET / "idx_scorer.py").read_text()
    assert "IDX_SCORER_ON:" in text and "IDX_SCORER_OFF:" in text


class _FakeWrapper:
    plans = 0

    def __init__(self, num_heads):
        self.num_heads = num_heads

    def plan(self, cu_q, seq_lens, block_table, *, total_q, max_seqlen_q, max_seqlen_k):
        _FakeWrapper.plans += 1
        self.total_q, self.pages, self.block_table = total_q, block_table.shape[1], block_table

    def run_scores(self, q, k):
        # [T * H, pages], T-major: row t * H + h scores (t, h); value 100 h + 10 t + page.
        t = torch.arange(self.total_q, dtype=torch.float32)[:, None, None]
        h = torch.arange(self.num_heads, dtype=torch.float32)[None, :, None]
        p = torch.arange(self.pages, dtype=torch.float32)[None, None, :]
        return (100 * h + 10 * t + p).reshape(self.total_q * self.num_heads, self.pages)


def _score_inputs(qo=256, prefix=4096, heads=2):
    kv = qo + prefix
    q = torch.zeros((qo, heads, 128), dtype=torch.float8_e4m3fn)
    k = torch.zeros((-(-kv // 128), 1, 128, 128), dtype=torch.float8_e4m3fn)
    lens = torch.tensor([qo], dtype=torch.int32), torch.tensor([kv], dtype=torch.int32)
    return q, k, lens[0], lens[1], lens[1] - lens[0], torch.arange(k.shape[0], dtype=torch.int32)


def _fake_module(monkeypatch):
    monkeypatch.setattr(
        idx_scorer, "_module", lambda: SimpleNamespace(BatchPrefillIndexerQ8KV8Wrapper=_FakeWrapper)
    )
    monkeypatch.setattr(idx_scorer, "_STATE", {})
    _FakeWrapper.plans = 0


def test_plan_is_kept_and_the_block_table_follows_an_in_place_rewrite(monkeypatch) -> None:
    _fake_module(monkeypatch)
    q, k, qo, kv, off, kv_indices = _score_inputs()
    assert idx_scorer.score(q, k, qo, kv, off, kv_indices) is not None
    assert idx_scorer.score(q, k, qo, kv, off, kv_indices) is not None
    assert _FakeWrapper.plans == 1
    # In place, same lengths and buffer: the next call must read the new pages.
    kv_indices.copy_(kv_indices.flip(0))
    idx_scorer.score(q, k, qo, kv, off, kv_indices)
    assert _FakeWrapper.plans == 1
    assert torch.equal(idx_scorer._STATE["wrapper"].block_table[0], kv_indices)
    q2, k2, qo2, kv2, off2, kv_indices2 = _score_inputs(prefix=8192)  # new lengths: a new plan
    idx_scorer.score(q2, k2, qo2, kv2, off2, kv_indices2)
    assert _FakeWrapper.plans == 2
    assert torch.equal(idx_scorer._STATE["wrapper"].block_table[0], kv_indices2)


def test_inference_mode_page_table(monkeypatch) -> None:
    """b20 gate 840811: the serving page table is an inference tensor; reading its version raised in warmup."""
    _fake_module(monkeypatch)
    with torch.inference_mode():
        q, k, qo, kv, off, kv_indices = _score_inputs()
        assert kv_indices.is_inference()
        for _ in range(3):
            out = idx_scorer.score(q, k, qo, kv, off, kv_indices)
            assert out is not None and out.shape == (1, k.shape[0], q.shape[0])
        kv_indices.copy_(kv_indices.flip(0))
        idx_scorer.score(q, k, qo, kv, off, kv_indices)
        assert torch.equal(idx_scorer._STATE["wrapper"].block_table[0], kv_indices)
    assert _FakeWrapper.plans == 1
    # The same lengths outside inference mode re-gather into the same plan.
    normal = kv_indices.clone()
    idx_scorer.score(q, k, qo, kv, off, normal.flip(0))
    assert _FakeWrapper.plans == 1
    assert torch.equal(idx_scorer._STATE["wrapper"].block_table[0], normal.flip(0))


def test_scores_are_per_kv_head(monkeypatch) -> None:
    """Gate 841031: tp2ctx has 2 index heads and 2 KV heads per rank; the scores must keep one row per KV head (an
    amax over the rank's heads gave [1, ...] and the fwd rejected the k2q CSR) and match _group_max_reduce."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
        _group_max_reduce,
    )

    for heads, kv_heads in ((2, 2), (2, 1), (4, 2)):
        _fake_module(monkeypatch)
        q, k, qo, kv, off, kv_indices = _score_inputs(qo=8, prefix=256, heads=heads)
        out = idx_scorer.score(q, k, qo, kv, off, kv_indices, num_kv_heads=kv_heads)
        per_head = _FakeWrapper(heads)
        per_head.plan(
            None, None, torch.zeros((1, k.shape[0])), total_q=8, max_seqlen_q=0, max_seqlen_k=0
        )
        raw = per_head.run_scores(q, k).view(8, heads, -1).permute(1, 2, 0)  # [H, pages, T]
        ref = _group_max_reduce(raw, SimpleNamespace(num_index_heads=heads, num_kv_heads=kv_heads))
        assert out.shape == (kv_heads, k.shape[0], 8) and torch.equal(out, ref)
    _fake_module(monkeypatch)
    q, k, qo, kv, off, kv_indices = _score_inputs(heads=2)
    # 2 heads do not split into 3 KV groups: declines, the proxy runs.
    assert idx_scorer.score(q, k, qo, kv, off, kv_indices, num_kv_heads=3) is None


def test_per_kv_head_scores_follow_an_in_place_rewrite_with_one_plan(monkeypatch) -> None:
    """b22: the per-KV-head scores [num_kv_heads, pages, T] (tp2ctx: 2 index heads, 2 KV heads per rank) with the kept
    plan; an in-place rewrite of an inference page table is re-gathered into the planned block table."""
    _fake_module(monkeypatch)
    with torch.inference_mode():
        q, k, qo, kv, off, kv_indices = _score_inputs(heads=2)
        first = idx_scorer.score(q, k, qo, kv, off, kv_indices, num_kv_heads=2)
        kv_indices.copy_(kv_indices.flip(0))
        second = idx_scorer.score(q, k, qo, kv, off, kv_indices, num_kv_heads=2)
        assert torch.equal(idx_scorer._STATE["wrapper"].block_table[0], kv_indices)
    assert first.shape == second.shape == (2, k.shape[0], q.shape[0])
    assert _FakeWrapper.plans == 1
