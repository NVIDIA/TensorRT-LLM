# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N3 A2 in this target's DSpark worker: the fused strict acceptance and next-step tokens equal the in-tree path
(N3 P and A1, the verify logits, at the end).

The worker's two overrides are driven on a bare worker (no model) against the in-tree methods of its base class on
the same inputs:
- eligible steps (K3 / K7, padded drafts, graph-padding dummy rows) are bitwise;
- forced acceptance (fractional, so the RNG pool path runs) over a run that alternates eligible and ineligible
  steps keeps the same accepted counts and the same RNG counter as an all-in-tree run;
- the sampler runs exactly once per step;
- the switch and an ineligible step take the in-tree path.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import strict_accept
from tensorrt_llm._torch.speculative.dspark import DSparkWorker

TARGET = "minimax_m31_nvfp4__sm_103__tp4"
dspark = importlib.import_module(
    f"tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.{TARGET}.dspark"
)

VOCAB = 200064

assert torch.cuda.is_available(), "the N3 A2 worker test requires a CUDA device"


def _worker(force: float = 0.0):
    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    nn.Module.__init__(w)
    w.force_num_accepted_tokens = force
    w._force_accept_rng_counter = None
    w._force_accept_rng_pool = None
    w.calls = 0
    return w


def _metadata(ragged: bool = False, graph: bool = False):
    return SimpleNamespace(is_ragged_verify=ragged, is_cuda_graph=graph)


def _fixed_sampler(tokens: torch.Tensor):
    """The in-tree sampler stands in as a fixed function of the logits: the overrides must call it once."""

    def sample(self, logits, spec_metadata, num_contexts, batch_size):
        self.calls += 1
        return tokens

    return sample


def _step(n_req: int, draft_len: int, gen: torch.Generator, pad: bool = False, int64: bool = False):
    k1 = draft_len + 1
    tokens = torch.randint(0, VOCAB, (n_req * k1,), generator=gen, device="cuda")
    tokens = tokens if int64 else tokens.to(torch.int32)
    draft = tokens.view(n_req, k1)[:, :draft_len].clone().to(torch.int32)
    cut = torch.randint(0, draft_len + 1, (n_req,), generator=gen, device="cuda")
    pos = torch.arange(draft_len, device="cuda").unsqueeze(0)
    draft = torch.where(pos == cut.unsqueeze(1), (draft + 1) % VOCAB, draft)
    if pad:
        draft[::2, draft_len // 2 :] = -1
    logits = torch.zeros(
        n_req * k1, 8, device="cuda"
    )  # shape only: the sampler stand-in ignores the values
    return logits, draft, tokens


def _in_tree(w, logits, draft, num_contexts, batch_size, md):
    return DSparkWorker._sample_and_accept_draft_tokens_base(
        w, logits, draft, num_contexts, batch_size, md
    )


@pytest.fixture(autouse=True)
def _compiled():
    strict_accept.precompile((3, 7), torch.device("cuda"))


@pytest.mark.parametrize(
    "n_req,draft_len,pad,int64",
    [
        (1, 3, False, False),
        (8, 3, False, False),
        (12, 3, True, False),
        (8, 7, False, False),
        (8, 3, False, True),
    ],
)
def test_eligible_steps_are_bitwise(monkeypatch, n_req, draft_len, pad, int64):
    gen = torch.Generator(device="cuda").manual_seed(3)
    logits, draft, tokens = _step(n_req, draft_len, gen, pad=pad, int64=int64)
    monkeypatch.setattr(dspark.M3DSparkWorker, "_sample_tokens_for_batch", _fixed_sampler(tokens))
    w, ref = _worker(), _worker()
    md = _metadata()
    assert w._n3a_eligible(logits, draft, 0, n_req, md)
    acc, num = w._sample_and_accept_draft_tokens_base(logits, draft, 0, n_req, md)
    racc, rnum = _in_tree(ref, logits, draft, 0, n_req, md)
    assert w.calls == 1 and ref.calls == 1
    assert acc.dtype == racc.dtype and num.dtype == rnum.dtype
    assert torch.equal(acc, racc) and torch.equal(num, rnum)


def test_forced_acceptance_alternating_steps_matches_in_tree(monkeypatch):
    """Fractional forced AL (the RNG pool path) over eligible, ragged (ineligible) and graph-warmup steps: the
    counts and the RNG counter follow the all-in-tree run step for step."""
    gen = torch.Generator(device="cuda").manual_seed(5)
    w, ref = _worker(force=1.6), _worker(force=1.6)
    served = 0
    for step in range(12):
        logits, draft, tokens = _step(8, 3, gen)
        num_contexts, batch_size = 0, 8
        if step % 3 == 2:
            # An ineligible step: one context row ahead of the 8 generation requests (the in-tree path runs).
            ctx_token = torch.randint(0, VOCAB, (1,), generator=gen, device="cuda").to(tokens.dtype)
            tokens = torch.cat([ctx_token, tokens])
            logits = torch.zeros(tokens.numel(), 8, device="cuda")
            num_contexts, batch_size = 1, 9
        monkeypatch.setattr(
            dspark.M3DSparkWorker, "_sample_tokens_for_batch", _fixed_sampler(tokens)
        )
        md = _metadata(graph=(step == 4))
        eligible = w._n3a_eligible(logits, draft, num_contexts, batch_size, md)
        assert eligible == (num_contexts == 0), step
        served += int(eligible)
        acc, num = w._sample_and_accept_draft_tokens_base(
            logits, draft, num_contexts, batch_size, md
        )
        racc, rnum = _in_tree(ref, logits, draft, num_contexts, batch_size, md)
        # A context row defines only column 0 (the in-tree path leaves the rest uninitialized).
        assert torch.equal(acc[:num_contexts, 0], racc[:num_contexts, 0]), step
        assert torch.equal(acc[num_contexts:], racc[num_contexts:]) and torch.equal(num, rnum), step
        if ref._force_accept_rng_counter is not None:
            assert torch.equal(w._force_accept_rng_counter, ref._force_accept_rng_counter), step
    assert served == 8
    # A ragged verify step is declined (the in-tree ragged path keeps it).
    logits, draft, _ = _step(8, 3, gen)
    assert not w._n3a_eligible(logits, draft, 0, 8, _metadata(ragged=True))


def test_ineligible_and_switch_take_the_in_tree_path(monkeypatch):
    gen = torch.Generator(device="cuda").manual_seed(9)
    logits, draft, tokens = _step(4, 3, gen)
    w = _worker()
    # A context row, a row-count mismatch, too many requests: declined.
    assert not w._n3a_eligible(logits, draft, 1, 4, _metadata())
    assert not w._n3a_eligible(logits[:-1], draft, 0, 4, _metadata())
    big_logits, big_draft, _ = _step(strict_accept.max_requests() + 1, 3, gen)
    assert not w._n3a_eligible(big_logits, big_draft, 0, big_draft.shape[0], _metadata())
    monkeypatch.setattr(dspark, "N3A_DISABLED", True)
    assert not w._n3a_eligible(logits, draft, 0, 4, _metadata())


@pytest.mark.parametrize(
    "draft_len,wide,shuffle", [(3, False, False), (3, True, True), (7, False, True)]
)
def test_next_new_tokens_matches_in_tree(draft_len, wide, shuffle):
    gen = torch.Generator(device="cuda").manual_seed(13)
    n_req = 8
    width = draft_len + 1 + (4 if wide else 0)
    accepted = torch.randint(
        0, VOCAB, (n_req, width), generator=gen, device="cuda", dtype=torch.int32
    )
    num = torch.randint(1, draft_len + 2, (n_req,), generator=gen, device="cuda", dtype=torch.int32)
    batch_idx = torch.arange(32, device="cuda", dtype=torch.int32)
    if shuffle:
        batch_idx[:n_req] = torch.randperm(n_req, generator=gen, device="cuda").to(torch.int32)
    next_draft = torch.randint(
        0, VOCAB, (n_req, draft_len), generator=gen, device="cuda", dtype=torch.int32
    )
    w = _worker()
    out = w._prepare_next_new_tokens(accepted, next_draft, batch_idx, n_req, num)
    ref = DSparkWorker._prepare_next_new_tokens(w, accepted, next_draft, batch_idx, n_req, num)
    assert out.dtype == ref.dtype and torch.equal(out, ref)


# N3 P-gid + P-cat: the worker's target logits equal the logits processor's.


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_rank_major_strided_copy_equals_vocab_concat_and_cast(tp):
    """The all-gather along dim 0 + one strided copy writes what the vocab concat + .float() writes."""
    gen = torch.Generator(device="cuda").manual_seed(tp)
    rows, shard = 32, 25008
    shards = [
        torch.randn(rows, shard, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(tp)
    ]
    gathered = torch.cat(shards, dim=0)  # what the dim-0 all-gather returns, rank-major
    out = torch.empty((rows, tp * shard), dtype=torch.float32, device="cuda")
    out.view(rows, tp, shard).copy_(gathered.view(tp, rows, shard).transpose(0, 1))
    assert torch.equal(out, torch.cat(shards, dim=-1).float())


def _lm_head(vocab=50016, hidden=6144):
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm.mapping import Mapping

    head = LMHead(
        vocab,
        hidden,
        dtype=torch.bfloat16,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
    ).cuda()
    with torch.no_grad():
        head.weight.copy_(
            torch.randn_like(head.weight, dtype=torch.float32).to(torch.bfloat16) * 0.02
        )
    return head


def test_target_logits_equal_the_processor_at_tp1():
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor

    head = _lm_head()
    rows = 32
    hidden = torch.randn(rows, 6144, device="cuda").to(torch.bfloat16)
    md = SimpleNamespace(
        is_ragged_verify=False, gather_ids=torch.arange(rows, device="cuda", dtype=torch.int32)
    )
    attn = SimpleNamespace(num_contexts=0)
    out = _worker().target_logits(hidden, attn, md, head)
    ref = LogitsProcessor().forward(hidden[md.gather_ids], head, attn, True)
    assert out is not None and out.dtype == ref.dtype and torch.equal(out, ref)


def test_target_logits_decline_to_the_processor(monkeypatch):
    head = _lm_head(vocab=1024, hidden=256)
    hidden = torch.randn(8, 256, device="cuda").to(torch.bfloat16)
    ids = torch.arange(8, device="cuda", dtype=torch.int32)
    w = _worker()
    # A context row, a gather that is not every row, a ragged verify, a padded head: the processor runs.
    assert (
        w.target_logits(
            hidden,
            SimpleNamespace(num_contexts=1),
            SimpleNamespace(is_ragged_verify=False, gather_ids=ids),
            head,
        )
        is None
    )
    assert (
        w.target_logits(
            hidden,
            SimpleNamespace(num_contexts=0),
            SimpleNamespace(is_ragged_verify=False, gather_ids=ids[:4]),
            head,
        )
        is None
    )
    assert (
        w.target_logits(
            hidden,
            SimpleNamespace(num_contexts=0),
            SimpleNamespace(is_ragged_verify=True, gather_ids=ids),
            head,
        )
        is None
    )
    monkeypatch.setattr(head, "padding_size", 3)
    assert (
        w.target_logits(
            hidden,
            SimpleNamespace(num_contexts=0),
            SimpleNamespace(is_ragged_verify=False, gather_ids=ids),
            head,
        )
        is None
    )
    monkeypatch.setattr(head, "padding_size", 0)
    monkeypatch.setattr(dspark, "N3P_DISABLED", True)
    assert (
        w.target_logits(
            hidden,
            SimpleNamespace(num_contexts=0),
            SimpleNamespace(is_ragged_verify=False, gather_ids=ids),
            head,
        )
        is None
    )


# N3 A1: the verify logits' all-gather + concat + cast in one launch (catalog/comm/allgather_cast) when set up.


def _a1_step(monkeypatch, rows: int, compiled: bool = True):
    """A TP4-shaped call: the head reports tp 4, A1 is set up, the op and the NCCL all-gather are recorded."""
    head = _lm_head(vocab=1024, hidden=256)
    monkeypatch.setattr(head, "tp_size", 4)
    calls = {"a1": 0, "nccl": 0}
    sentinel = torch.full((rows, 4 * 1024), 7.0, device="cuda")

    def fake_a1(shard, **ws):
        assert ws == {"ws": "a1"} and shard.shape == (rows, 1024)
        calls["a1"] += 1
        return sentinel

    def fake_allgather(shard, mapping, dim=0):
        calls["nccl"] += 1
        return torch.cat([shard] * 4, dim=0)

    monkeypatch.setattr(dspark._allgather_cast, "allgather_cast", fake_a1)
    monkeypatch.setattr(dspark._allgather_cast, "is_compiled", lambda *a, **k: compiled)
    monkeypatch.setattr(dspark, "allgather", fake_allgather)
    w = _worker()
    w._a1 = {"ws": "a1"}
    hidden = torch.randn(rows, 256, device="cuda").to(torch.bfloat16)
    md = SimpleNamespace(
        is_ragged_verify=False, gather_ids=torch.arange(rows, device="cuda", dtype=torch.int32)
    )
    out = w.target_logits(hidden, SimpleNamespace(num_contexts=0), md, head)
    return out, sentinel, calls


@pytest.mark.parametrize("rows", [4, 32, 64])
def test_target_logits_take_a1_when_set_up(monkeypatch, rows):
    out, sentinel, calls = _a1_step(monkeypatch, rows)
    assert out is sentinel and calls == {"a1": 1, "nccl": 0}


@pytest.mark.parametrize("rows,compiled", [(68, True), (4, False)])
def test_target_logits_fall_back_to_nccl_past_a1(monkeypatch, rows, compiled):
    """More rows than the workspace holds, or a shape never compiled: the NCCL all-gather + strided copy."""
    out, _, calls = _a1_step(monkeypatch, rows, compiled=compiled)
    assert (
        calls == {"a1": 0, "nccl": 1}
        and out.shape == (rows, 4 * 1024)
        and out.dtype == torch.float32
    )


def _a1_prebuild_worker(monkeypatch, tp: int = 4, draft_len: int = 3):
    w = _worker()
    w.mapping = SimpleNamespace(tp_size=tp, tp_rank=0, enable_attention_dp=False)
    w.spec_config = SimpleNamespace(max_draft_len=draft_len)  # max_draft_len reads it
    # Eligibility needs every SM up to the workspace's grid; any CUDA device passes with a 1-CTA workspace grid.
    monkeypatch.setattr(dspark._allgather_cast, "MAX_GRID", 1)
    monkeypatch.setattr(dspark.M3DSparkWorker, "_a1_all_ok", lambda self, local: local)
    return w


def test_prebuild_a1_sets_up_k3_rows(monkeypatch):
    head = _lm_head(vocab=1024, hidden=256)
    seen = {}

    def fake_precompile(mapping, cols, rows, device, group=None, agree=None):
        seen.update(cols=cols, rows=rows, agreed=agree(True))
        return True

    ws = {"ws_uc": torch.zeros(4, dtype=torch.int32), "rank": 0}
    monkeypatch.setattr(dspark._allgather_cast, "precompile", fake_precompile)
    monkeypatch.setattr(
        dspark._allgather_cast, "workspace_args", lambda mapping, cols, group=None: ws
    )
    assert _a1_prebuild_worker(monkeypatch)._prebuild_a1(head) is ws
    # K3 verify rows of batches 1..16, all within the workspace's 64 rows.
    assert seen == {"cols": 1024, "rows": tuple(range(4, 65, 4)), "agreed": True}


@pytest.mark.parametrize("draft_len", [3, 7])
def test_prebuild_a1_sets_up_the_verify_rows_of_k3_and_k7(monkeypatch, draft_len):
    """K3: rows 4..64 (batches 1..16); K7, the headline: rows 8..64 (batches 1..8)."""
    head = _lm_head(vocab=1024, hidden=256)
    seen = {}

    def fake_precompile(mapping, cols, rows, device, group=None, agree=None):
        seen.update(rows=rows, agreed=agree(True))
        return True

    ws = {"ws_uc": torch.zeros(4, dtype=torch.int32), "rank": 0}
    monkeypatch.setattr(dspark._allgather_cast, "precompile", fake_precompile)
    monkeypatch.setattr(
        dspark._allgather_cast, "workspace_args", lambda mapping, cols, group=None: ws
    )
    assert _a1_prebuild_worker(monkeypatch, draft_len=draft_len)._prebuild_a1(head) is ws
    k1 = draft_len + 1
    assert seen == {"rows": tuple(range(k1, 65, k1)), "agreed": True}


def test_prebuild_a1_off(monkeypatch):
    """The switch, another draft length, one rank, a rank that is not eligible, or a compile that failed on any rank:
    no A1, no workspace."""
    head = _lm_head(vocab=1024, hidden=256)
    built = {"n": 0}

    def fake_precompile(*a, **k):
        built["n"] += 1
        return False  # a failed compile somewhere

    monkeypatch.setattr(dspark._allgather_cast, "precompile", fake_precompile)
    monkeypatch.setattr(dspark, "A1_K7_DISABLED", True)
    assert _a1_prebuild_worker(monkeypatch, draft_len=7)._prebuild_a1(head) is None
    monkeypatch.setattr(dspark, "A1_K7_DISABLED", False)
    assert (
        _a1_prebuild_worker(monkeypatch, draft_len=64)._prebuild_a1(head) is None
    )  # 65 rows at batch 1
    assert _a1_prebuild_worker(monkeypatch, tp=1)._prebuild_a1(head) is None
    assert built["n"] == 0
    assert _a1_prebuild_worker(monkeypatch)._prebuild_a1(head) is None
    assert built["n"] == 1
    # Another rank not eligible: off here too, nothing compiled.
    w = _a1_prebuild_worker(monkeypatch)
    monkeypatch.setattr(dspark.M3DSparkWorker, "_a1_all_ok", lambda self, local: False)
    assert w._prebuild_a1(head) is None and built["n"] == 1
    monkeypatch.setattr(dspark, "A1_DISABLED", True)
    assert _a1_prebuild_worker(monkeypatch)._prebuild_a1(head) is None
    assert built["n"] == 1


# N3A-KV: the draft-forward KV bookkeeping in one launch, through the DFlash hook (bitwise to the stock lines).


def _kv_step(nc: int, bs: int, gen: torch.Generator):
    num = torch.randint(1, 5, (bs + 2,), generator=gen, device="cuda", dtype=torch.int32)
    kv = torch.randint(0, 1 << 20, (bs + 2,), generator=gen, device="cuda", dtype=torch.int32)
    return num, kv


def _kv_worker(fused: bool):
    w = _worker()
    w._fused_kv_rewind = None
    if fused:
        strict_accept.precompile_kv_rewind(torch.device("cuda"))
        w._n3a_kv_buf = torch.empty(strict_accept.max_requests(), dtype=torch.int32, device="cuda")
        w._fused_kv_rewind = w._n3a_kv_rewind
    return w


@pytest.mark.parametrize("nc,bs", [(0, 1), (0, 8), (2, 6), (3, 3)])
def test_kv_rewind_hook_equals_the_stock_lines(nc, bs):
    gen = torch.Generator(device="cuda").manual_seed(5)
    num, kv = _kv_step(nc, bs, gen)
    outs = []
    for fused in (False, True):
        w = _kv_worker(fused)
        attn = SimpleNamespace(kv_lens_cuda=kv.clone(), update_for_spec_dec=lambda: None)
        w._prepare_kv_for_draft_forward(attn, num, nc, bs)
        outs.append(
            (
                w._kv_rewind_amount,
                attn.kv_lens_cuda,
                w._kv_rewind_nc,
                w._kv_rewind_bs,
                w._kv_rewind_pending,
            )
        )
    (r0, k0, *rest0), (r1, k1, *rest1) = outs
    assert torch.equal(k0, k1) and rest0 == rest1
    assert r0.dtype == r1.dtype and r0.shape == r1.shape and torch.equal(r0, r1)


def test_kv_rewind_hook_declines_unsupported_operands():
    w = _kv_worker(True)
    num64 = torch.ones(4, dtype=torch.int64, device="cuda")  # not the buffer's dtype
    kv = torch.zeros(4, dtype=torch.int32, device="cuda")
    assert w._n3a_kv_rewind(num64, kv, 0, 4) is None and torch.equal(kv, torch.zeros_like(kv))
    w._n3a_kv_buf = None
    assert w._n3a_kv_rewind(torch.ones(4, dtype=torch.int32, device="cuda"), kv, 0, 4) is None
