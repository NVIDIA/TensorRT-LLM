# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp4 target's worker-side routing of the one-launch Markov chain (``dspark_s4.S4WorkerMixin``), without GPUs
or a model: which steps it serves, that the kill switch sets nothing up, that a declined step is the in-tree path, that
the step's logits processor only takes the block rows, and that the chain's tokens replace the in-tree sampling for
that one call. The chain itself is the catalog's ``spec_decode/dspark_markov_chain``, certified by
``spec_decode/_dspark_markov_chain_op_matrix.py``; the in-model equality by the call-site gate."""

import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4 import (  # noqa: E501
    dspark_s4 as w1,
)


class _Base:
    def __init__(self):
        self.calls = []

    def _forward_impl(self, *args):
        self.calls.append("forward")
        return "in-tree"

    def _refine_block_logits(self, draft_model, gen_logits, inputs, spec_metadata):
        self.calls.append("refine")
        return gen_logits

    def sample_draft_tokens(self, logits, spec_metadata, batch_size, **kwargs):
        self.calls.append("sample")
        return torch.full(logits.shape[:2], -1, dtype=torch.int32)

    def _apply_kv_rewind_after_draft(self, attn_metadata, spec_metadata):
        self.calls.append("stock_rewind")
        self._kv_rewind_pending = False


class _Worker(w1.S4WorkerMixin, _Base):
    pass


def _state(block=3, shard=8, rows=8):
    st = w1._S4State(None, rows, shard, 0, None, None, block, {})
    for b in range(1, w1.MAX_FAST_BATCH + 1):
        st.bufs[(b, block)] = torch.full((b, block, rows), float("-inf"))
    return st


def _meta(advanced=False):
    return SimpleNamespace(wants_advanced_draft_sampling=advanced)


def test_serves_only_eligible_steps(monkeypatch):
    wk = _Worker()
    draft = SimpleNamespace(lilicorr=None, has_candidate_selector=False)
    assert not wk._s4_serves(1, _meta(), draft)  # not set up
    wk._s4 = _state()
    assert wk._s4_serves(1, _meta(), draft)
    assert wk._s4_serves(w1.MAX_FAST_BATCH, _meta(), draft)
    assert not wk._s4_serves(0, _meta(), draft)
    assert not wk._s4_serves(w1.MAX_FAST_BATCH + 1, _meta(), draft)
    assert not wk._s4_serves(1, _meta(advanced=True), draft)
    wk._d2t = torch.zeros(1)
    assert not wk._s4_serves(1, _meta(), draft)


def test_kill_switch_sets_nothing_up(monkeypatch):
    wk = _Worker()
    wk._s4 = _state()
    monkeypatch.setattr(w1, "_DISABLED", True)
    assert not wk.prebuild_s4(None, SimpleNamespace(has_markov_head=True))
    assert wk._s4 is None
    assert not wk._s4_serves(
        1, _meta(), SimpleNamespace(lilicorr=None, has_candidate_selector=False)
    )


def test_declined_step_is_the_in_tree_path():
    wk = _Worker()
    attn = SimpleNamespace(num_seqs=12, num_contexts=0)  # above MAX_FAST_BATCH
    draft = SimpleNamespace(lilicorr=None, has_candidate_selector=False, logits_processor="proc")
    wk._s4 = _state()
    assert wk._forward_impl(None, None, None, None, attn, _meta(), draft) == "in-tree"
    assert draft.logits_processor == "proc"
    logits = torch.zeros(2, 3, 16)  # full-vocab width: not this step's shard
    assert wk._refine_block_logits(draft, logits, {}, None) is logits
    assert wk.sample_draft_tokens(logits, None, 2).eq(-1).all()
    assert wk.calls == ["forward", "refine", "sample"]


def test_sharded_logits_processor_takes_only_block_rows():
    st = _state(block=3, shard=8, rows=12)
    inner_calls = []

    def inner(h, head, meta, ctx):
        inner_calls.append(h.shape)
        return "inner"

    head = SimpleNamespace(apply_linear=lambda h, b: h[:, :8] * 2)
    proc = w1._ShardedDraftLogits(inner, st, batch=2)
    h = torch.arange(6 * 10, dtype=torch.float32).view(6, 10)
    out = proc(h, head, None, True)
    assert torch.equal(out, h[:, :8] * 2)
    buf = st.buffer(2)
    assert torch.equal(buf[:, :, :8], (h[:, :8] * 2).view(2, 3, 8))
    assert torch.isinf(buf[:, :, 8:]).all()
    assert proc(torch.zeros(5, 10), head, None, True) == "inner" and inner_calls == [
        torch.Size([5, 10])
    ]


def test_chain_tokens_replace_sampling_once(monkeypatch):
    wk = _Worker()
    st = _state(block=3, shard=8, rows=8)
    wk._s4_step = st
    toks = torch.tensor([[1, 2, 3], [4, 5, 6]])
    monkeypatch.setattr(w1, "chain", SimpleNamespace(supports=lambda *a: True,
                                                     dspark_markov_chain=lambda *a, **k: toks.int(),
                                                     ll_exchange_engaged=lambda batch: False))  # fmt: skip
    logits = torch.zeros(2, 3, 8)
    assert wk._refine_block_logits(None, logits, {"first_prev_tokens": torch.zeros(2, dtype=torch.long)}, None) \
        is logits  # fmt: skip
    out = wk.sample_draft_tokens(logits, None, 2)
    assert out.dtype == torch.int32 and torch.equal(out, toks.int())
    assert wk.sample_draft_tokens(logits, None, 2).eq(-1).all()  # used once
    assert "refine" not in wk.calls


def test_unsupported_shapes_fall_back_on_the_shards(monkeypatch):
    wk = _Worker()
    wk._s4_step = _state(block=3, shard=8, rows=8)

    def _never(*a, **k):
        raise AssertionError("an unsupported step must not reach the op")

    monkeypatch.setattr(
        w1, "chain", SimpleNamespace(supports=lambda *a: False, dspark_markov_chain=_never)
    )
    logits = torch.zeros(2, 3, 8)
    wk._refine_block_logits(
        None, logits, {"first_prev_tokens": torch.zeros(2, dtype=torch.long)}, None
    )
    assert wk.calls == ["refine"]
    assert wk.sample_draft_tokens(logits, None, 2).eq(-1).all()


def test_switch_reads_any_value_but_empty_or_zero(monkeypatch):
    """Read at import, like the targets' m31_switches: "true" turns S4 off, not only "1"."""
    try:
        for value, off in (
            ("1", True),
            ("true", True),
            ("yes", True),
            ("0", False),
            ("", False),
            (" 0 ", False),
        ):
            monkeypatch.setenv(w1.DISABLE_ENV, value)
            importlib.reload(w1)
            assert w1.disabled() is off, value
    finally:
        monkeypatch.delenv(w1.DISABLE_ENV, raising=False)
        importlib.reload(w1)
    assert not w1.disabled()


def test_markov_ll_engagement_logged_once_per_rank(monkeypatch):
    """MARKOV_LL's "[M3 dispatch]" line: once per rank, on the first served step that took the LL kernel."""
    wk = _Worker()
    wk.mapping = SimpleNamespace(tp_rank=3)
    wk._s4_step = _state(block=3, shard=8, rows=8)
    lines = []
    monkeypatch.setattr(w1.logger, "info", lambda msg, *a, **k: lines.append(msg))
    monkeypatch.setattr(w1, "chain", SimpleNamespace(supports=lambda *a: True,
                                                     dspark_markov_chain=lambda *a, **k: torch.zeros(
                                                         a[0].shape[0], 3, dtype=torch.int32),
                                                     ll_exchange_engaged=lambda batch: batch == 1))  # fmt: skip
    inputs2 = {"first_prev_tokens": torch.zeros(2, dtype=torch.long)}
    wk._refine_block_logits(None, torch.zeros(2, 3, 8), inputs2, None)  # batch 2: not the LL kernel
    assert not [m for m in lines if "MARKOV_LL" in m]
    inputs1 = {"first_prev_tokens": torch.zeros(1, dtype=torch.long)}
    for _ in range(3):
        wk._refine_block_logits(None, torch.zeros(1, 3, 8), inputs1, None)
    ll = [m for m in lines if "MARKOV_LL" in m]
    assert ll == [
        "[M3 dispatch] MARKOV_LL: k3_markov LL flag-in-payload exchange (batch 1) on rank 3"
    ], ll


def _markov_ll_in_subprocess(value):
    env = {k: v for k, v in os.environ.items() if k != "TRTLLM_M31_DISABLE_MARKOV_LL"}
    if value is not None:
        env["TRTLLM_M31_DISABLE_MARKOV_LL"] = value
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m31_nvfp4__sm_103__tp4 import dspark_s4 as m; print('MARKOV_LL=' + str(m.MARKOV_LL))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("MARKOV_LL=")][-1]


def test_markov_ll_kill_switch_subprocess():
    """MARKOV_LL is on by default; TRTLLM_M31_DISABLE_MARKOV_LL=1 (read at import) turns it off."""
    assert _markov_ll_in_subprocess(None) == "MARKOV_LL=True"
    assert _markov_ll_in_subprocess("0") == "MARKOV_LL=True"
    assert _markov_ll_in_subprocess("1") == "MARKOV_LL=False"


def _rewind_worker(monkeypatch, batch=2, nc=1, on=True):
    """A worker in a served step with a pending KV-length rewind of ``batch`` rows from ``nc``, and a chain fake that
    records which launch ran."""
    wk = _Worker()
    wk.mapping = SimpleNamespace(tp_rank=5)
    wk._s4_step = _state(block=3, shard=8, rows=8)
    wk._s4_step_attn = SimpleNamespace(kv_lens_cuda=torch.zeros(nc + batch + 1, dtype=torch.int32))
    wk._kv_rewind_pending = True
    wk._kv_rewind_amount = torch.zeros(batch, dtype=torch.int32)
    wk._kv_rewind_nc, wk._kv_rewind_bs = nc, nc + batch
    launches, lines = [], []
    toks = torch.zeros(batch, 3, dtype=torch.int32)

    def plain(*a, **k):
        launches.append("chain")
        return toks

    def with_rewind(base, first_prev, w1_, w2s, offset, kv_lens, amount, first, **k):
        launches.append(
            (
                "rewind",
                kv_lens is wk._s4_step_attn.kv_lens_cuda,
                amount is wk._kv_rewind_amount,
                first,
            )
        )
        return toks

    monkeypatch.setattr(w1, "TAILBK_SPEC", on)
    monkeypatch.setattr(w1.logger, "info", lambda msg, *a, **k: lines.append(msg))
    monkeypatch.setattr(w1, "chain", SimpleNamespace(supports=lambda *a: True, supports_rewind=lambda *a: True,
                                                     dspark_markov_chain=plain,
                                                     dspark_markov_chain_rewind=with_rewind,
                                                     ll_exchange_engaged=lambda b: False))  # fmt: skip
    return wk, launches, lines, {"first_prev_tokens": torch.zeros(batch, dtype=torch.long)}


def test_tailbk_h_rewind_in_chain_launch(monkeypatch):
    """TAILBK_SPEC H: a real step's chain launch applies the pending rewind (this step's kv_lens, amounts, first
    row) and the stock rewind lines are skipped for that step; the engagement line is logged once per rank."""
    wk, launches, lines, inputs = _rewind_worker(monkeypatch)
    meta = SimpleNamespace(is_cuda_graph=False)
    for _ in range(2):
        wk._kv_rewind_pending = True
        wk._refine_block_logits(None, torch.zeros(2, 3, 8), inputs, meta)
        wk._apply_kv_rewind_after_draft(None, meta)
        assert not wk._kv_rewind_pending and not wk._tailbk_rewound
    assert launches == [("rewind", True, True, 1)] * 2, launches
    assert "stock_rewind" not in wk.calls
    assert [m for m in lines if "TAILBK_SPEC H" in m] == [
        "[M3 dispatch] TAILBK_SPEC H: KV-length rewind in the Markov chain launch on rank 5"
    ]


def test_tailbk_h_warmup_keeps_stock_path(monkeypatch):
    """The eager warmup of a graph batch (is_cuda_graph, not capturing): the stock method skips the rewind there, so
    the chain must not apply it either."""
    wk, launches, _, inputs = _rewind_worker(monkeypatch)
    meta = SimpleNamespace(is_cuda_graph=True)
    wk._refine_block_logits(None, torch.zeros(2, 3, 8), inputs, meta)
    wk._apply_kv_rewind_after_draft(None, meta)
    assert launches == ["chain"] and wk.calls == ["stock_rewind"]


def test_tailbk_h_off_or_mismatch_keeps_stock_path(monkeypatch):
    """Kill switch off, no pending rewind, or amounts that are not this chain call's rows: the stock rewind runs."""
    meta = SimpleNamespace(is_cuda_graph=False)
    wk, launches, _, inputs = _rewind_worker(monkeypatch, on=False)
    wk._refine_block_logits(None, torch.zeros(2, 3, 8), inputs, meta)
    wk._apply_kv_rewind_after_draft(None, meta)
    assert launches == ["chain"] and wk.calls == ["stock_rewind"]
    wk, launches, _, inputs = _rewind_worker(monkeypatch)
    wk._kv_rewind_pending = False
    wk._refine_block_logits(None, torch.zeros(2, 3, 8), inputs, meta)
    assert launches == ["chain"]
    wk, launches, _, inputs = _rewind_worker(monkeypatch)
    wk._kv_rewind_amount = torch.zeros(3, dtype=torch.int32)  # not the 2 rows of this call
    wk._refine_block_logits(None, torch.zeros(2, 3, 8), inputs, meta)
    wk._apply_kv_rewind_after_draft(None, meta)
    assert launches == ["chain"] and wk.calls == ["stock_rewind"]


def test_tailbk_h_unsupported_operands_keep_stock_path(monkeypatch):
    """supports_rewind False (e.g. int64 amounts): the plain chain launch and the stock rewind."""
    wk, launches, _, inputs = _rewind_worker(monkeypatch)
    monkeypatch.setattr(w1.chain, "supports_rewind", lambda *a: False)
    meta = SimpleNamespace(is_cuda_graph=False)
    wk._refine_block_logits(None, torch.zeros(2, 3, 8), inputs, meta)
    wk._apply_kv_rewind_after_draft(None, meta)
    assert launches == ["chain"] and wk.calls == ["stock_rewind"]


def _tailbk_in_subprocess(value):
    env = {k: v for k, v in os.environ.items() if k != "TRTLLM_M31_DISABLE_TAILBK_SPEC"}
    if value is not None:
        env["TRTLLM_M31_DISABLE_TAILBK_SPEC"] = value
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m31_nvfp4__sm_103__tp4 import dspark_s4 as m; print('TAILBK_SPEC=' + str(m.TAILBK_SPEC))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("TAILBK_SPEC=")][-1]


def test_tailbk_spec_kill_switch_subprocess():
    """TAILBK_SPEC is on by default; TRTLLM_M31_DISABLE_TAILBK_SPEC=1 (read at import) turns it off."""
    assert _tailbk_in_subprocess(None) == "TAILBK_SPEC=True"
    assert _tailbk_in_subprocess("0") == "TAILBK_SPEC=True"
    assert _tailbk_in_subprocess("1") == "TAILBK_SPEC=False"


def test_tailbk_c_engagement_logged_once_per_rank(monkeypatch):
    """TAILBK_SPEC C's "[M3 dispatch]" line: once per rank, only after a chain call took the cached inputs."""
    wk, launches, lines, inputs = _rewind_worker(monkeypatch)
    wk._kv_rewind_pending = False  # the plain chain launch
    taken = [False]
    monkeypatch.setattr(w1.chain, "cached_inputs_taken", lambda: taken[0], raising=False)
    meta = SimpleNamespace(is_cuda_graph=False)
    wk._refine_block_logits(None, torch.zeros(2, 3, 8), inputs, meta)
    assert not [m for m in lines if "TAILBK_SPEC C" in m]
    taken[0] = True
    for _ in range(3):
        wk._refine_block_logits(None, torch.zeros(2, 3, 8), inputs, meta)
    wk._kv_rewind_pending = True  # and through the rewind launch
    wk._refine_block_logits(None, torch.zeros(2, 3, 8), inputs, meta)
    assert [m for m in lines if "TAILBK_SPEC C" in m] == [
        "[M3 dispatch] TAILBK_SPEC C: chain accepted inputs from precompile cache (taken) on rank 5"
    ]
