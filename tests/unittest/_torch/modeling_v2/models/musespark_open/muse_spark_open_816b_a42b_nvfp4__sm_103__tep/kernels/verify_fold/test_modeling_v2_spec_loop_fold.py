# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the DFlash worker with ``verify_fold`` / ``path_walk_tail`` against the worker without them.

Two workers are built the way the shell builds them -- re-classed by
``draft_head`` and ``fused_accept`` (``spec_worker``), the second also by
``spec_loop_fold.install`` -- and step side by side over thousands of decode
steps from identical state: the same target logits, drafts, captured target
features, slot map, context lengths, KV lengths and forced-acceptance pool.
Every step, every value the step hands on is compared bit for bit: the
accepted tokens and counts, the drafter inputs (noise embedding, query
positions, context lengths per request, the bonus tokens, the arena row
index), the draft tokens, the next input tokens, and the state the next step
reads -- ``kv_lens_cuda``, the copy of ``_seq_lens_cuda`` the drafter sees,
the restored ``_seq_lens_cuda``, ``_ctx_len``, the drafter's K/V arena, the
forced-acceptance counter. Eagerly (each method in ``DFlashWorker._forward_impl``'s
order), from captured CUDA graphs replayed with the inputs updated in place,
and through the eager warmup forward the CUDA-graph runner makes before a
capture (where the folds run the built-in methods). Forced acceptance on (3.05,
a fractional rate) and off (natural acceptance, where the count depends on the
drafts): the fold must not care which.

The step body is a stand-in for ``DFlashWorker._forward_impl`` on a
generation-only step: the same worker methods in the same order, with a
drafter whose forward is a
fixed function of its inputs. The draft head's sharded selector is replaced by
a fixed function that ends in the walk the worker hands it, so the draft-head path
(and the tail walk) run as installed.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    spec_worker as drafter_worker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
    head as draft_head,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_embedding import (  # noqa: E501
    embedding as replicated_embedding,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.verify_fold import (  # noqa: E501
    fold as spec_loop_fold,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

assert torch.cuda.is_available(), "the DFlash step fold worker test requires a CUDA device"

K = 7
BLOCK = K + 1
TARGET_VOCAB = 384
EMBED_VOCAB = 1024
MASK_ID = 1001
FULL_VOCAB = 4096
TP = 2
HIDDEN = 64
LAYERS, KV_HEADS, HEAD_DIM = 2, 2, 16
SLOTS = 5
CAPACITY = 4096
MAX_CTX = CAPACITY - BLOCK
MAX_BATCH = 4

#: Where the step body raises (``test_a_failed_step_leaves_the_builtin_state``): None,
#: "drafter" (before the draft head's walk) or "after_walk".
_FAIL = {"at": None}

#: How the step body departs from ``DFlashWorker._forward_impl``
#: (``test_a_step_that_does_not_hand_on_the_fold_is_refused``): None,
#: "other-metadata" (a copy of the attention metadata to
#: ``_prepare_attn_metadata_for_dflash``), "skipped-kv-prepare" (no
#: ``_prepare_kv_for_draft_forward``) or "other-metadata-at-rewind" (a copy to
#: ``_apply_kv_rewind_after_draft``).
_MISUSE = {"at": None}


class _InjectedFailure(RuntimeError):
    pass


class _Metadata:
    """The attention metadata fields the DFlash step reads, with the built-in spec-dec save/restore."""

    prepare_for_spec_dec = AttentionMetadata.prepare_for_spec_dec
    restore_from_spec_dec = AttentionMetadata.restore_from_spec_dec
    has_spec_dec_saved_state = AttentionMetadata.has_spec_dec_saved_state

    def __init__(self, batch, g):
        self._saved_tensors = {}
        self.num_contexts, self.num_seqs, self.num_ctx_tokens = 0, batch, 0
        self._seq_lens = torch.full((batch,), BLOCK, dtype=torch.int32)
        self._seq_lens_cuda = torch.randint(1, 9, (MAX_BATCH,), generator=g, device="cuda").int()
        self.kv_lens_cuda = torch.randint(
            1000, 4000, (MAX_BATCH,), generator=g, device="cuda"
        ).int()
        self.spec_dec_updates = 0

    def update_for_spec_dec(self):
        self.spec_dec_updates += 1

    def on_update(self):
        pass


class _Embedding(nn.Module):
    """A one-shard embedding: the drafter's ``embed_tokens`` stand-in."""

    def __init__(self, table):
        super().__init__()
        self.table = table

    def forward(self, ids):
        return self.table[ids]


class _Drafter:
    """The drafter surface the worker's prepare and draft head touch, as fixed functions."""

    def __init__(self, g):
        table = (torch.randn(EMBED_VOCAB, HIDDEN, generator=g, device="cuda") * 0.1).to(
            torch.bfloat16
        )
        table = torch.where(table == 0, torch.full_like(table, 0.5), table)  # no signed zeros
        self.embed = _Embedding(table)
        self.draft_model_full = SimpleNamespace(model=SimpleNamespace(embed_tokens=self.embed))
        self.model = SimpleNamespace()
        self.fc = object()
        self.hidden_norm = object()
        self.candidate_selector = SimpleNamespace(
            predecessor_codebook=torch.empty(FULL_VOCAB, 4, device="cuda"), top_k=16
        )
        self.has_candidate_selector = True
        self.w = (torch.randn(5 * HIDDEN, HIDDEN, generator=g, device="cuda") * 0.05).to(
            torch.bfloat16
        )
        self.head = (torch.randn(HIDDEN, FULL_VOCAB // TP, generator=g, device="cuda")).to(
            torch.bfloat16
        )

    def project_target_hidden(self, x):
        return (x.float() @ self.w.float()).to(torch.bfloat16)

    def precompute_context_kv(self, projected, positions):
        n = projected.shape[0]
        base = projected.float().view(n, LAYERS, KV_HEADS, HEAD_DIM)
        k = (base + positions.view(n, 1, 1, 1).float() * 1e-3).to(torch.bfloat16)
        v = (base * 2).to(torch.bfloat16)
        return k.contiguous(), v.contiguous()

    def block_logits(self, noise, query_positions):
        """The drafter forward + LM-head shard stand-in: ``[g, K, shard]`` bf16 logits."""
        if _FAIL["at"] == "drafter":
            raise _InjectedFailure("drafter")
        h = noise.float() + query_positions.unsqueeze(-1).float() * 1e-4
        return (h[:, 1:, :] @ self.head.float()).to(torch.bfloat16).contiguous()


def _fake_selector(
    local_logits, block_hidden_states, anchor_tokens, selector, mapping, allreduce=None, walk=None
):
    """``draft_head.selector_draft_tokens``'s signature: a fixed lattice, then the walk it is handed."""
    values, ids = local_logits.float().topk(16, dim=-1)
    unary = (values + anchor_tokens.view(-1, 1, 1).float() * 1e-3).contiguous()
    edges = (unary[:, :, :, None] * 0.5 - unary[:, :, None, :] * 0.25).contiguous()
    walk = draft_head.draft_path_walk if walk is None else walk
    return walk(edges, unary, ids.long().contiguous())


def _forward_body(
    self,
    input_ids,
    position_ids,
    hidden_states,
    logits,
    attn_metadata,
    spec_metadata,
    draft_model,
    resource_manager=None,
):
    """``DFlashWorker._forward_impl`` on a generation-only step, method for method."""
    batch_size = attn_metadata.num_seqs
    num_contexts = attn_metadata.num_contexts
    misuse = _MISUSE["at"]
    accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
        logits, attn_metadata, spec_metadata
    )
    self._prepare_attn_metadata_for_dflash(
        copy.copy(attn_metadata) if misuse == "other-metadata" else attn_metadata, spec_metadata
    )
    if misuse != "skipped-kv-prepare":
        self._prepare_kv_for_draft_forward(
            attn_metadata, num_accepted_tokens, num_contexts, batch_size
        )
    inputs = self.prepare_1st_drafter_inputs(
        input_ids=input_ids,
        position_ids=position_ids,
        hidden_states=hidden_states,
        accepted_tokens=accepted_tokens,
        num_accepted_tokens=num_accepted_tokens,
        attn_metadata=attn_metadata,
        spec_metadata=spec_metadata,
        draft_model=draft_model,
        total_target_tokens=batch_size * BLOCK,
    )
    seen_seq_lens = attn_metadata._seq_lens_cuda.clone()
    gen_logits = draft_model.block_logits(inputs["noise_embedding"], inputs["query_positions"])
    selected = self._apply_dflash2_selector(
        draft_model, gen_logits, None, inputs["first_prev_tokens"], spec_metadata
    )
    if _FAIL["at"] == "after_walk":
        raise _InjectedFailure("after the walk")
    gen_draft_tokens = self.sample_draft_tokens(selected, spec_metadata, batch_size, num_contexts=0)
    next_draft_tokens = gen_draft_tokens
    self._restore_attn_metadata_from_spec_dec(attn_metadata)
    self._apply_kv_rewind_after_draft(
        copy.copy(attn_metadata) if misuse == "other-metadata-at-rewind" else attn_metadata,
        spec_metadata,
    )
    next_new_tokens = self._prepare_next_new_tokens(
        accepted_tokens,
        next_draft_tokens,
        spec_metadata.batch_indices_cuda,
        batch_size,
        num_accepted_tokens,
    )
    return {
        "new_tokens": accepted_tokens,
        "new_tokens_lens": num_accepted_tokens,
        "next_draft_tokens": next_draft_tokens,
        "next_new_tokens": next_new_tokens,
        "seen_seq_lens": seen_seq_lens,
        **{f"inputs.{k}": v for k, v in inputs.items() if isinstance(v, torch.Tensor)},
    }


class _Side:
    """One worker with its own copy of every buffer a step writes."""

    def __init__(
        self,
        verify_fold_on: bool,
        rate: float,
        batch: int,
        seed: int,
        replicated_embedding_on: bool = False,
    ):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.drafter = _Drafter(g)
        worker = DFlashWorker.__new__(DFlashWorker)
        # The DFlash context arena's windowing state, as DFlashWorker.__init__ sets it.
        worker._ctx_base = None
        worker._ctx_base_host = None
        worker._ctx_keep = 0
        worker.windowed_ctx_arena = False
        worker._saved_ctx_base = None
        worker._saved_ctx_base_host = None
        nn.Module.__init__(worker)
        worker.force_num_accepted_tokens = rate
        worker._force_accept_rng_pool = None
        worker._force_accept_rng_counter = None
        worker.spec_config = SimpleNamespace(max_draft_len=K, mask_token_id=MASK_ID)
        worker._resolved_mask_token_id = None
        worker._compute_block_size = BLOCK
        worker._max_ctx = MAX_CTX
        worker._ctx_len = torch.randint(100, 3000, (SLOTS,), generator=g, device="cuda").long()
        worker._batch_to_slot = torch.randperm(SLOTS, generator=g, device="cuda")[:MAX_BATCH].long()
        worker._ctx_paged = False
        worker._ctx_block_tables = None
        worker._ctx_kv_buf = None
        worker._ctx_page_table = None
        shape = (SLOTS, LAYERS, CAPACITY, KV_HEADS, HEAD_DIM)
        worker._ctx_k_buf = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        worker._ctx_v_buf = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        worker._kv_rewind_amount = None
        worker._kv_rewind_pending = False
        worker._d2t = None
        worker.mapping = SimpleNamespace(tp_size=TP, tp_rank=0, enable_attention_dp=False)
        worker.__class__ = draft_head.ShardedDraftHeadDFlashWorker
        assert drafter_worker.install_fused_accept(worker) is None
        if verify_fold_on:
            assert spec_loop_fold.install(worker, True, True)
        if replicated_embedding_on:
            # The replicated embedding table on the drafter's embedding (its rows have no -0.0,
            # so the stand-in embedding's rows are the same bits).
            self.drafter.embed.__dict__[replicated_embedding.TABLE_ATTR] = self.drafter.embed.table
        self.worker = worker
        self.metadata = _Metadata(batch, g)
        self.batch = batch

    def spec_metadata(self, draft_tokens, captured, cuda_graph):
        return SimpleNamespace(
            runtime_draft_len=K,
            hidden_size=HIDDEN,
            draft_tokens=draft_tokens,
            is_ragged_verify=False,
            is_cuda_graph=cuda_graph,
            is_all_greedy_sample=True,
            use_rejection_sampling=False,
            enable_penalty=False,
            wants_advanced_draft_sampling=False,
            batch_indices_cuda=self.batch_indices,
            get_hidden_states=lambda n: captured[:n],
        )

    @property
    def batch_indices(self):
        if not hasattr(self, "_batch_indices"):
            self._batch_indices = torch.arange(MAX_BATCH, dtype=torch.int32, device="cuda")
        return self._batch_indices

    def forward(self, logits, draft_tokens, captured, cuda_graph=False, public=False):
        """One step through ``_forward_impl``, or through ``SpecWorkerBase.forward``
        (``public``), which restores the spec-dec state when the step fails."""
        return (self.worker.forward if public else self.worker._forward_impl)(
            input_ids=None,
            position_ids=None,
            hidden_states=None,
            logits=logits,
            attn_metadata=self.metadata,
            spec_metadata=self.spec_metadata(draft_tokens, captured, cuda_graph),
            draft_model=self.drafter,
        )

    def state(self):
        w, m = self.worker, self.metadata
        return {
            "kv_lens_cuda": m.kv_lens_cuda,
            "_seq_lens_cuda": m._seq_lens_cuda,
            "_seq_lens": m._seq_lens,
            "_ctx_len": w._ctx_len,
            "_ctx_k_buf": w._ctx_k_buf,
            "_ctx_v_buf": w._ctx_v_buf,
            "counter": w._force_accept_rng_counter,
            "saved": torch.tensor(len(m._saved_tensors)),
        }


def _inputs(batch, g, previous_drafts=None):
    """Target logits whose argmax agrees with a random prefix of the drafts, the drafts, the captured rows."""
    rows = batch * BLOCK
    drafts = (
        previous_drafts
        if previous_drafts is not None
        else torch.randint(0, TARGET_VOCAB, (batch, K), generator=g, device="cuda").int()
    )
    target = torch.randint(0, TARGET_VOCAB, (batch, BLOCK), generator=g, device="cuda")
    lead = torch.randint(0, BLOCK, (batch,), generator=g, device="cuda")
    keep = torch.arange(K, device="cuda").unsqueeze(0) < lead.unsqueeze(1)
    target[:, :K] = torch.where(keep, drafts.long() % TARGET_VOCAB, target[:, :K])
    logits = torch.randn(rows, TARGET_VOCAB, generator=g, device="cuda")
    logits.scatter_(1, target.reshape(-1, 1), 50.0)
    captured = (torch.randn(rows, 5 * HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    return logits, drafts.reshape(-1).contiguous(), captured


def _compare(step, out_a, out_b, side_a, side_b):
    assert out_a.keys() == out_b.keys(), (out_a.keys(), out_b.keys())
    for name in out_a:
        a, b = out_a[name], out_b[name]
        assert a.dtype == b.dtype and a.shape == b.shape, (
            step,
            name,
            a.dtype,
            b.dtype,
            a.shape,
            b.shape,
        )
        assert torch.equal(a, b), (step, name)
    state_a, state_b = side_a.state(), side_b.state()
    for name in state_a:
        a, b = state_a[name], state_b[name]
        if a is None or b is None:
            assert a is None and b is None, (step, name)
            continue
        assert torch.equal(a, b), (step, name)


@pytest.fixture
def fake_selector(monkeypatch):
    monkeypatch.setattr(draft_head, "selector_draft_tokens", _fake_selector)


@pytest.fixture
def replicated_embedding_table(monkeypatch):
    """The drafter stand-in's table as the replicated embedding table (``replicated_embedding.table_of``)."""
    monkeypatch.setattr(
        replicated_embedding, "table_of", lambda m: m.__dict__.get(replicated_embedding.TABLE_ATTR)
    )


@pytest.fixture
def body(monkeypatch):
    """Routes ``DFlashWorker._forward_impl`` to the step body (the fold's wrapper still runs around it).

    Called after the workers are built: ``spec_loop_fold.install`` checks the
    built-in ``_forward_impl``'s source, which the step body is not.
    """
    return lambda: monkeypatch.setattr(DFlashWorker, "_forward_impl", _forward_body)


def _counted(monkeypatch):
    calls = {"fold": 0, "tail": 0}
    fold, tail = spec_loop_fold.dflash_verify_fold, spec_loop_fold.draft_path_walk_tail

    def counted_fold(*a, **k):
        calls["fold"] += 1
        return fold(*a, **k)

    def counted_tail(*a, **k):
        calls["tail"] += 1
        return tail(*a, **k)

    monkeypatch.setattr(spec_loop_fold, "dflash_verify_fold", counted_fold)
    monkeypatch.setattr(spec_loop_fold, "draft_path_walk_tail", counted_tail)
    return calls


@pytest.mark.parametrize(
    "rate, batch, replicated_embedding_on",
    [(3.05, 1, False), (3.05, 1, True), (0.0, 1, False), (3.0, 2, False), (2.7, 3, True)],
    ids=[
        "forced-3.05",
        "forced-3.05-replicated-embedding",
        "natural",
        "forced-int-2req",
        "forced-2.7-3req-replicated-embedding",
    ],
)
def test_eager_steps_match(
    monkeypatch,
    fake_selector,
    body,
    replicated_embedding_table,
    rate,
    batch,
    replicated_embedding_on,
):
    calls = _counted(monkeypatch)
    builtin = _Side(False, rate, batch, seed=7)
    folded = _Side(True, rate, batch, seed=7, replicated_embedding_on=replicated_embedding_on)
    body()
    g = torch.Generator(device="cuda").manual_seed(int(rate * 100) + batch)
    steps = 2000 if (rate, batch, replicated_embedding_on) == (3.05, 1, False) else 300
    drafts = None
    for step in range(steps):
        logits, flat_drafts, captured = _inputs(batch, g, drafts)
        out_b = builtin.forward(logits, flat_drafts, captured)
        out_f = folded.forward(logits, flat_drafts, captured)
        _compare(step, out_b, out_f, builtin, folded)
        drafts = out_b["next_draft_tokens"]
        # The arena columns past the context length are rewritten, never read: keep the
        # comparison meaningful by letting the lengths wrap before the arena fills.
        if int(builtin.worker._ctx_len.max()) > MAX_CTX - 64:
            for side in (builtin, folded):
                side.worker._ctx_len.sub_(2000).clamp_(min=0)
    assert calls == {"fold": steps, "tail": steps}, calls


def test_warmup_forward_runs_the_builtin_methods(monkeypatch, fake_selector, body):
    """The eager forward before a capture (is_cuda_graph, not capturing): no fold, same values."""
    calls = _counted(monkeypatch)
    builtin = _Side(False, 3.05, 1, seed=3)
    folded = _Side(True, 3.05, 1, seed=3)
    body()
    g = torch.Generator(device="cuda").manual_seed(3)
    for step in range(20):
        logits, drafts, captured = _inputs(1, g)
        _compare(
            step,
            builtin.forward(logits, drafts, captured, cuda_graph=True),
            folded.forward(logits, drafts, captured, cuda_graph=True),
            builtin,
            folded,
        )
    assert calls == {"fold": 0, "tail": 0}


@pytest.mark.parametrize("rate", [3.05, 0.0], ids=["forced-3.05", "natural"])
def test_cuda_graph_replays_match(monkeypatch, fake_selector, body, rate):
    """Each worker's step captured once; replays with the inputs updated in place match."""
    calls = _counted(monkeypatch)
    sides = [_Side(False, rate, 1, seed=11), _Side(True, rate, 1, seed=11)]
    body()
    g = torch.Generator(device="cuda").manual_seed(11)
    logits, drafts, captured = _inputs(1, g)
    # Eager steps first (as serving does), so the forced-acceptance pool exists before capture.
    for side in sides:
        side.forward(logits, drafts, captured)
    assert calls == {"fold": 1, "tail": 1}, "the folded worker's eager step folds"
    calls.update(fold=0, tail=0)
    static = [(logits.clone(), drafts.clone(), captured.clone()) for _ in sides]
    graphs, outs = [], []
    for side, (lg, dr, cp) in zip(sides, static):
        side.forward(lg, dr, cp, cuda_graph=True)  # warmup forward, as the runner does
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            outs.append(side.forward(lg, dr, cp, cuda_graph=True))
        torch.cuda.current_stream().wait_stream(stream)
        graphs.append(graph)
    assert calls == {"fold": 1, "tail": 1}, (
        "the warmup forward does not fold; the capture folds once"
    )
    drafts_next = None
    for step in range(500):
        lg, dr, cp = _inputs(1, g, drafts_next)
        for (s_lg, s_dr, s_cp), graph in zip(static, graphs):
            s_lg.copy_(lg), s_dr.copy_(dr), s_cp.copy_(cp)
            graph.replay()
        torch.cuda.synchronize()
        _compare(step, outs[0], outs[1], sides[0], sides[1])
        drafts_next = outs[0]["next_draft_tokens"].clone()
        if int(sides[0].worker._ctx_len.max()) > MAX_CTX - 64:
            for side in sides:
                side.worker._ctx_len.sub_(2000).clamp_(min=0)


@pytest.mark.parametrize("fail_at", ["drafter", "after_walk"])
def test_a_failed_step_leaves_the_builtin_state(monkeypatch, fake_selector, body, fail_at):
    """A step that raises before or after the draft head's walk (the worker's
    ``forward`` then restores the spec-dec state): ``kv_lens_cuda`` (bumped by the
    fold, then rewound by the walk or still owed) and the restored metadata equal
    the built-in's, and the steps after it still match."""
    calls = _counted(monkeypatch)
    builtin = _Side(False, 3.05, 1, seed=5)
    folded = _Side(True, 3.05, 1, seed=5)
    body()
    g = torch.Generator(device="cuda").manual_seed(5)
    drafts = None
    for step in range(12):
        logits, flat_drafts, captured = _inputs(1, g, drafts)
        if step == 4:
            monkeypatch.setitem(_FAIL, "at", fail_at)
            for side in (builtin, folded):
                with pytest.raises(_InjectedFailure):
                    side.forward(logits, flat_drafts, captured, public=True)
            monkeypatch.setitem(_FAIL, "at", None)
            torch.cuda.synchronize()
            for name in ("kv_lens_cuda", "_seq_lens_cuda", "_seq_lens"):
                a, b = getattr(builtin.metadata, name), getattr(folded.metadata, name)
                assert torch.equal(a, b), (fail_at, name)
            assert not builtin.metadata._saved_tensors and not folded.metadata._saved_tensors
            assert not builtin.worker._kv_rewind_pending and not folded.worker._kv_rewind_pending
            continue
        out_b = builtin.forward(logits, flat_drafts, captured, public=True)
        out_f = folded.forward(logits, flat_drafts, captured, public=True)
        _compare(step, out_b, out_f, builtin, folded)
        drafts = out_b["next_draft_tokens"]
    assert calls == {"fold": 12, "tail": 11 if fail_at == "drafter" else 12}, calls


def _hand_on_copies(worker, tokens_only):
    """Re-class ``worker`` under a class whose ``sample_and_accept_draft_tokens`` hands on
    copies of the accepted tokens (and counts, unless ``tokens_only``) -- what another
    fast path's worker class composed on top of the fold's could do."""
    base = type(worker)

    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        accepted, num_accepted = base.sample_and_accept_draft_tokens(
            self, logits, attn_metadata, spec_metadata
        )
        return accepted.clone(), num_accepted if tokens_only else num_accepted.clone()

    worker.__class__ = type(
        f"Copying{base.__name__}",
        (base,),
        {"sample_and_accept_draft_tokens": sample_and_accept_draft_tokens},
    )


@pytest.mark.parametrize(
    "misuse, method",
    [
        ("copied-outputs", "_prepare_kv_for_draft_forward"),
        ("copied-tokens", "prepare_1st_drafter_inputs"),
        ("other-metadata", "_prepare_attn_metadata_for_dflash"),
        ("skipped-kv-prepare", "not taken by _prepare_kv_for_draft_forward"),
        ("other-metadata-at-rewind", "_apply_kv_rewind_after_draft"),
    ],
)
def test_a_step_that_does_not_hand_on_the_fold_is_refused(
    monkeypatch, fake_selector, body, misuse, method
):
    """Once the fold has applied ``kv_lens_cuda += 1`` and owes the rewind (or the
    walk has applied it), a worker method handed other tensors or another metadata
    than the step's raises instead of running its built-in, which would bump or
    rewind a second time; so does a step that skips one of the fold's consumers.
    Through the worker's ``forward`` the owed rewind is then settled once:
    ``kv_lens_cuda`` and the restored metadata end where the built-in step's do."""
    calls = _counted(monkeypatch)
    builtin = _Side(False, 3.05, 1, seed=13)
    folded = _Side(True, 3.05, 1, seed=13)
    body()
    g = torch.Generator(device="cuda").manual_seed(13)
    drafts = None
    for step in range(3):
        logits, flat_drafts, captured = _inputs(1, g, drafts)
        out_b = builtin.forward(logits, flat_drafts, captured, public=True)
        out_f = folded.forward(logits, flat_drafts, captured, public=True)
        _compare(step, out_b, out_f, builtin, folded)
        drafts = out_b["next_draft_tokens"]
    if misuse.startswith("copied"):
        _hand_on_copies(folded.worker, tokens_only=misuse == "copied-tokens")
    logits, flat_drafts, captured = _inputs(1, g, drafts)
    kv_before = folded.metadata.kv_lens_cuda.clone()
    builtin.forward(logits, flat_drafts, captured, public=True)
    monkeypatch.setitem(_MISUSE, "at", None if misuse.startswith("copied") else misuse)
    with pytest.raises(spec_loop_fold.FoldStateError, match=method):
        folded.forward(logits, flat_drafts, captured, public=True)
    monkeypatch.setitem(_MISUSE, "at", None)
    torch.cuda.synchronize()
    assert calls["fold"] == 4, calls  # the folded worker's three steps and the refused one
    for name in ("kv_lens_cuda", "_seq_lens_cuda", "_seq_lens"):
        a, b = getattr(builtin.metadata, name), getattr(folded.metadata, name)
        assert torch.equal(a, b), (misuse, name, a, b)
    assert not torch.equal(folded.metadata.kv_lens_cuda, kv_before), "the step moved kv_lens"
    assert not folded.metadata._saved_tensors
    assert not folded.worker._kv_rewind_pending


def test_install_refuses_without_the_kernels_it_extends():
    worker = DFlashWorker.__new__(DFlashWorker)
    # The DFlash context arena's windowing state, as DFlashWorker.__init__ sets it.
    worker._ctx_base = None
    worker._ctx_base_host = None
    worker._ctx_keep = 0
    worker.windowed_ctx_arena = False
    worker._saved_ctx_base = None
    worker._saved_ctx_base_host = None
    nn.Module.__init__(worker)
    with pytest.raises(ValueError, match="verify_fold extends the fused acceptance"):
        spec_loop_fold.install(worker, True, False)
    with pytest.raises(ValueError, match="path_walk_tail extends the draft-head path walk"):
        spec_loop_fold.install(worker, False, True)
    assert not spec_loop_fold.install(worker, False, False)
    assert type(worker) is DFlashWorker
