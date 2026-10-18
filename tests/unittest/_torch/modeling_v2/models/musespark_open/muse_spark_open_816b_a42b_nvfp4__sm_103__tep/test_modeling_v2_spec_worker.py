# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the DFlash worker with the context-K/V fork under the verify fold, against the built-in.

Two workers are built the way the shell builds them: both re-classed by ``draft_head``
and the fused acceptance (``drafter_worker``); the second then by
``context_kv_fork`` (the step's context-K/V store on a side stream,
forked at the LM head) and on top of that by ``spec_loop_fold.install``
(``verify_fold`` / ``path_walk_tail``). On a step the fold takes, its prepare takes the fork's
side chain (``SpecLoopFoldDFlashWorker._fold_forked_inputs``: the commit right
after the fold kernel, the noise block, the join) instead of storing the
context itself; on a step it does not fold (the eager warmup forward before a
capture) the fork's own prepare runs. The two workers step side by side from
identical state -- eagerly, with unrelated work on the main stream while the
side chain runs, and from captured CUDA graphs (fork, side chain, commit and
join inside the graph) replayed with the inputs updated in place -- and every
value a step hands on and every buffer the next step reads must be the
built-in worker's, bit for bit (BF16 compared as bits, so a zero's sign
counts): the accepted tokens and counts, the drafter inputs, the draft and
next input tokens, ``kv_lens_cuda``, the sequence lengths, ``_ctx_len`` and
the whole drafter K/V arena. The fork must have launched on every step and
none may be left pending.

The step body, the drafter and the draft head's selector are the stand-ins of
``kernels/verify_fold/test_modeling_v2_spec_loop_fold.py``, with the fork armed
and fired where ``DecodeTarget`` and the LM head arm and fire it (before the
worker's forward).

Also checked: the install order the shell uses is the only one either module
accepts (the fold composed on top of the fork; a worker the fold re-classed first
is refused by the fork), and repeated installs of both accept the composed
class.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    spec_worker as drafter_worker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as context_kv_fork,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
    head as draft_head,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.verify_fold import (  # noqa: E501
    fold as spec_loop_fold,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

assert torch.cuda.is_available(), "the composed worker test requires a CUDA device"

K = 7
BLOCK = K + 1
TARGET_VOCAB = 384
EMBED_VOCAB = 1024
MASK_ID = 1001
FULL_VOCAB = 4096
TP = 2
HIDDEN = 64
TAPS = 5
LAYERS, KV_HEADS, HEAD_DIM = 2, 2, 16
SLOTS = 5
CAPACITY = 4096
MAX_CTX = CAPACITY - BLOCK
MAX_BATCH = 4

#: Unrelated main-stream work between the fork and the verify, or None.
_BUSY = {"buf": None}


class _Metadata:
    """The attention metadata fields the DFlash step reads, with the built-in spec-dec save/restore."""

    prepare_for_spec_dec = AttentionMetadata.prepare_for_spec_dec
    restore_from_spec_dec = AttentionMetadata.restore_from_spec_dec
    has_spec_dec_saved_state = AttentionMetadata.has_spec_dec_saved_state

    def __init__(self, batch, g):
        self._saved_tensors = {}
        self.num_contexts, self.num_seqs, self.num_ctx_tokens = 0, batch, 0
        self.num_tokens = batch * BLOCK
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
    """The drafter surface the prepare, the fork's side chain and the draft head touch."""

    dflash_attention_backend = "VANILLA"

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
        self.w = (torch.randn(TAPS * HIDDEN, HIDDEN, generator=g, device="cuda") * 0.05).to(
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
    """``DFlashWorker._forward_impl`` on a generation-only step, after the target's fork point."""
    fork = self.__dict__.get("_kv_fork")
    if fork is not None:
        # DecodeTarget.forward arms the fork; the LM head's GEMM fires it.
        fork.arm(spec_metadata, attn_metadata)
        fork.fire("head")
        busy = _BUSY["buf"]
        if busy is not None:
            for _ in range(4):
                busy = torch.nn.functional.linear(busy, _BUSY["buf"])
    batch_size = attn_metadata.num_seqs
    num_contexts = attn_metadata.num_contexts
    accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
        logits, attn_metadata, spec_metadata
    )
    self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
    self._prepare_kv_for_draft_forward(attn_metadata, num_accepted_tokens, num_contexts, batch_size)
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
    if fork is not None:
        assert not fork.launched, "the prepare returned without joining the side chain"
    seen_seq_lens = attn_metadata._seq_lens_cuda.clone()
    gen_logits = draft_model.block_logits(inputs["noise_embedding"], inputs["query_positions"])
    selected = self._apply_dflash2_selector(
        draft_model, gen_logits, None, inputs["first_prev_tokens"], spec_metadata
    )
    gen_draft_tokens = self.sample_draft_tokens(selected, spec_metadata, batch_size, num_contexts=0)
    next_draft_tokens = gen_draft_tokens
    self._restore_attn_metadata_from_spec_dec(attn_metadata)
    self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)
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


def _base_worker(g, rate):
    """A DFlash worker re-classed by the draft head and the fused acceptance, with a dense arena."""
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
    worker._ctx_buf_inited = True
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
    return worker


def _fork_worker(worker, drafter):
    """``context_kv_fork.install``'s worker part for the head point: the re-class and the fork."""
    context_kv_fork._reclass(worker)
    worker._kv_fork = context_kv_fork.ContextKvFork(worker, drafter, "head")
    return worker._kv_fork


class _Side:
    """One worker with its own copy of every buffer a step writes."""

    def __init__(self, composed: bool, rate: float, batch: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.drafter = _Drafter(g)
        worker = _base_worker(g, rate)
        if composed:
            # The shell's order: the fork's install, then the fold's on top of the class it left.
            _fork_worker(worker, self.drafter)
            assert spec_loop_fold.install(worker, True, True)
            mro = type(worker).__mro__
            assert mro.index(spec_loop_fold.SpecLoopFoldDFlashWorker) < mro.index(
                context_kv_fork.ContextKvForkDFlashWorker
            )
        self.worker = worker
        self.metadata = _Metadata(batch, g)
        self.batch = batch

    def spec_metadata(self, draft_tokens, captured, cuda_graph):
        return SimpleNamespace(
            runtime_draft_len=K,
            hidden_size=HIDDEN,
            num_capture_layers=TAPS,
            layers_to_capture=[1, 16, 30, 44, 59],
            captured_hidden_states=captured,
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

    def forward(self, logits, draft_tokens, captured, cuda_graph=False):
        return self.worker._forward_impl(
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
    captured = (torch.randn(rows, TAPS * HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    return logits, drafts.reshape(-1).contiguous(), captured


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def _compare(step, out_a, out_b, side_a, side_b):
    assert out_a.keys() == out_b.keys(), (out_a.keys(), out_b.keys())
    for name in out_a:
        a, b = out_a[name], out_b[name]
        assert a.dtype == b.dtype and a.shape == b.shape, (step, name, a.dtype, b.dtype)
        assert torch.equal(_bits(a), _bits(b)), (step, name)
    state_a, state_b = side_a.state(), side_b.state()
    for name in state_a:
        a, b = state_a[name], state_b[name]
        if a is None or b is None:
            assert a is None and b is None, (step, name)
            continue
        assert torch.equal(_bits(a), _bits(b)), (step, name)


def _wrap_lengths(sides):
    """Let the lengths wrap before the arena fills (its columns past a length are never read)."""
    if int(sides[0].worker._ctx_len.max()) > MAX_CTX - 64:
        for side in sides:
            side.worker._ctx_len.sub_(2000).clamp_(min=0)


@pytest.fixture
def harness(monkeypatch):
    """The fake selector, counters on the fold and the forked prepare, and the step body.

    The returned ``body()`` routes ``DFlashWorker._forward_impl`` to the step
    body; it is called after the workers are built (``spec_loop_fold.install``
    checks the built-in's source).
    """
    monkeypatch.setattr(draft_head, "selector_draft_tokens", _fake_selector)
    calls = {"fold": 0, "forked": 0}
    fold = spec_loop_fold.dflash_verify_fold
    forked = spec_loop_fold.SpecLoopFoldDFlashWorker._fold_forked_inputs

    def counted_fold(*a, **k):
        calls["fold"] += 1
        return fold(*a, **k)

    def counted_forked(self, *a, **k):
        calls["forked"] += 1
        return forked(self, *a, **k)

    monkeypatch.setattr(spec_loop_fold, "dflash_verify_fold", counted_fold)
    monkeypatch.setattr(
        spec_loop_fold.SpecLoopFoldDFlashWorker, "_fold_forked_inputs", counted_forked
    )
    monkeypatch.setitem(_BUSY, "buf", None)
    return calls, lambda: monkeypatch.setattr(DFlashWorker, "_forward_impl", _forward_body)


@pytest.mark.parametrize(
    "rate, batch, busy",
    [(3.05, 1, True), (3.05, 1, False), (0.0, 1, True), (3.0, 2, True), (2.7, 3, True)],
    ids=["forced-3.05", "forced-3.05-idle-main", "natural", "forced-int-2req", "forced-2.7-3req"],
)
def test_eager_steps_match_the_builtin_worker(harness, rate, batch, busy):
    calls, body = harness
    builtin = _Side(False, rate, batch, seed=17)
    composed = _Side(True, rate, batch, seed=17)
    body()
    if busy:
        _BUSY["buf"] = torch.full((512, 512), 1.0 / 512, dtype=torch.bfloat16, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(int(rate * 100) + batch)
    steps = 1000 if (rate, batch, busy) == (3.05, 1, True) else 300
    drafts = None
    for step in range(steps):
        logits, flat_drafts, captured = _inputs(batch, g, drafts)
        out_b = builtin.forward(logits, flat_drafts, captured)
        out_c = composed.forward(logits, flat_drafts, captured)
        _compare(step, out_b, out_c, builtin, composed)
        drafts = out_b["next_draft_tokens"]
        _wrap_lengths((builtin, composed))
    fork = composed.worker._kv_fork
    assert calls == {"fold": steps, "forked": steps}, calls
    assert fork.launches == {"head": steps, "verify": 0}, fork.launches
    assert not fork.launched


def test_the_warmup_forward_runs_the_forks_own_prepare(harness):
    """The eager forward before a capture: no fold; the fold hands the step to the fork's prepare."""
    calls, body = harness
    builtin = _Side(False, 3.05, 1, seed=29)
    composed = _Side(True, 3.05, 1, seed=29)
    body()
    g = torch.Generator(device="cuda").manual_seed(29)
    for step in range(20):
        logits, drafts, captured = _inputs(1, g)
        _compare(
            step,
            builtin.forward(logits, drafts, captured, cuda_graph=True),
            composed.forward(logits, drafts, captured, cuda_graph=True),
            builtin,
            composed,
        )
    assert calls == {"fold": 0, "forked": 0}, calls
    assert composed.worker._kv_fork.launches["head"] == 20
    assert not composed.worker._kv_fork.launched


@pytest.mark.parametrize("rate", [3.05, 0.0], ids=["forced-3.05", "natural"])
def test_cuda_graph_replays_match_the_builtin_worker(harness, rate):
    """Each worker's step captured once (fork, side chain, commit and join in the graph), replayed."""
    calls, body = harness
    sides = [_Side(False, rate, 1, seed=23), _Side(True, rate, 1, seed=23)]
    body()
    _BUSY["buf"] = torch.full((512, 512), 1.0 / 512, dtype=torch.bfloat16, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(23)
    logits, drafts, captured = _inputs(1, g)
    # Eager steps first (as serving does), so the forced-acceptance pool exists before capture.
    for side in sides:
        side.forward(logits, drafts, captured)
    assert calls == {"fold": 1, "forked": 1}, calls
    static = [(logits.clone(), drafts.clone(), captured.clone()) for _ in sides]
    graphs, outs = [], []
    for side, (lg, dr, cp) in zip(sides, static):
        side.forward(lg, dr, cp, cuda_graph=True)  # the runner's warmup forward
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            outs.append(side.forward(lg, dr, cp, cuda_graph=True))
        torch.cuda.current_stream().wait_stream(stream)
        graphs.append(graph)
    # The warmup does not fold (the fork's own prepare joins it); the capture folds and forks.
    assert calls == {"fold": 2, "forked": 2}, calls
    fork = sides[1].worker._kv_fork
    assert fork.launches["head"] == 3 and not fork.launched
    drafts_next = None
    for step in range(500):
        lg, dr, cp = _inputs(1, g, drafts_next)
        for (s_lg, s_dr, s_cp), graph in zip(static, graphs):
            s_lg.copy_(lg), s_dr.copy_(dr), s_cp.copy_(cp)
            graph.replay()
        torch.cuda.synchronize()
        _compare(step, outs[0], outs[1], sides[0], sides[1])
        drafts_next = outs[0]["next_draft_tokens"].clone()
        _wrap_lengths(sides)


def test_a_side_chain_for_another_step_shape_is_refused(harness):
    """A side chain launched for another request count than the folded step's raises."""
    calls, body = harness
    composed = _Side(True, 3.05, 1, seed=37)
    body()
    g = torch.Generator(device="cuda").manual_seed(37)
    composed.forward(*_inputs(1, g))
    fork = composed.worker._kv_fork
    fire = fork.fire

    def fire_for_two_requests(point):
        fire(point)
        if fork._launched is not None:
            fork._launched.num_gens = 2

    fork.fire = fire_for_two_requests
    try:
        with pytest.raises(spec_loop_fold.FoldStateError, match="side chain was launched for 2"):
            composed.forward(*_inputs(1, g))
    finally:
        del fork.fire
        fork.settle()
        torch.cuda.synchronize()
    assert calls == {"fold": 2, "forked": 2}, calls


def test_the_install_order_is_context_kv_fork_then_verify_fold(monkeypatch):
    """Both modules' checks accept the fold composed on top of the fork, also on repeated
    installs; the fork refuses a worker the fold re-classed first (its prepare would then run
    in front of the fold's)."""
    monkeypatch.setattr(context_kv_fork, "DFlashForCausalLM", _Drafter)
    g = torch.Generator(device="cuda").manual_seed(31)
    drafter = _Drafter(g)

    worker = _base_worker(g, 3.05)
    assert context_kv_fork.worker_unsupported_reason(worker, drafter) is None
    fork = _fork_worker(worker, drafter)
    assert spec_loop_fold.unsupported_reason(worker) is None
    assert spec_loop_fold.install(worker, True, True)
    composed = type(worker)
    # The shell's next load hook: the fork's checks and re-class, then the fold's install, again.
    assert context_kv_fork.worker_unsupported_reason(worker, drafter) is None
    context_kv_fork._reclass(worker)
    assert not spec_loop_fold.install(worker, True, True)
    assert type(worker) is composed and worker._kv_fork is fork
    for name in context_kv_fork.FORK_METHODS:
        found = getattr(composed, name)
        if found is getattr(spec_loop_fold.SpecLoopFoldDFlashWorker, name, None):
            assert context_kv_fork._spec_loop_fold_on_top(worker, name, found), name

    other = _base_worker(torch.Generator(device="cuda").manual_seed(32), 3.05)
    assert spec_loop_fold.install(other, True, True)
    reason = context_kv_fork.worker_unsupported_reason(other, drafter)
    assert reason is not None and "overrides" in reason, reason
