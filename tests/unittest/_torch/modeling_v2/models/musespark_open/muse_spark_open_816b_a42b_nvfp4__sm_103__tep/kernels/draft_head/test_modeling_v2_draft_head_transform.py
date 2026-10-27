# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the vocab-sharded draft head with the DFlash drafter's logit transform on.

The claim is that with a drafter that declares a logit transform (the Muse
Spark Open drafter's multiplier and softcap) the draft head's fast path still
runs -- the deferring transform hands the bf16 shard through, no full-vocab
all-gather and no full-vocab ``torch.topk`` -- and drafts the built-in tokens
bit for bit: the built-in path transforms each shard with the drafter's own
method, all-gathers, runs ``torch.topk`` and the selector. Then every path
that keeps the built-in computation (sampled drafts, d2t, another layout,
another tensor, a stale record) gets the built-in transform bit for bit, the
fold's tail walk takes the same inputs, CUDA-graph replay follows new logits,
and the install is atomic, idempotent and checked.

TP ranks are emulated in one process: each rank's drafter, worker and head
shard are built here, and the key exchange returns what every rank's kernels
give for the same step.
"""

import types

import pytest
import torch

__extra_import_path__ = ["."]
from _draft_head_cases import (  # noqa: E402
    FAMILIES,
    HIDDEN,
    MUSESPARK,
    TOP_K,
    TP,
    VOCAB,
    DraftStub,
    assert_same_floats,
    builtin_candidates,
    builtin_draft_tokens,
    builtin_full_logits,
    builtin_transform,
    draft_head,
    draft_step,
    logit_transform,
    packed_keys,
    rank_keys,
    seed_of,
    selector,
    shard_keys,
)

from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM  # noqa: E402
from tensorrt_llm._torch.modules.linear import TensorParallelMode  # noqa: E402
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor  # noqa: E402

assert torch.cuda.is_available(), "the vocab-sharded draft head requires a CUDA device"

K = 7


# --------------------------------------------------------------------------- the emulated group


class StubHead:
    """A gathered bf16 column-parallel head whose shard GEMM returns ``source`` (a copy)."""

    tp_mode = TensorParallelMode.COLUMN
    gather_output = True
    padding_size = 0
    gather_output_sizes = None
    lora = None
    enable_lm_head_tp_in_adp = False
    bias = None

    def __init__(self, source: torch.Tensor, ranks: int):
        self.source = source
        self.tp_size = ranks
        self.out_features = source.shape[-1]
        self.weight = torch.zeros(1, dtype=torch.bfloat16, device="cuda")

    def apply_linear(self, hidden_states, bias):
        return self.source.clone()


class Exchange:
    """The key all-reduce, emulated: every rank's keys of ``full`` (read at call time) summed."""

    def __init__(self, full, ranks, t):
        self.full, self.ranks, self.t = full, ranks, t
        self.calls = 0

    def __call__(self, packed):
        self.calls += 1
        chunk = packed_keys(self.full, self.ranks, self.t)
        if packed.shape == chunk.shape:
            return chunk
        return rank_keys(self.full, self.ranks, self.t)


def make_drafter(config, lm_head):
    model = DFlashForCausalLM.__new__(DFlashForCausalLM)
    torch.nn.Module.__init__(model)
    model.logit_multiplier, model.logit_softcap = config
    model.candidate_selector = selector()
    model.logits_processor = LogitsProcessor()
    model.lm_head = lm_head
    return model


def make_worker(rank, ranks=TP):
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm.mapping import Mapping

    config = types.SimpleNamespace(max_draft_len=K, attention_backend="VANILLA")
    return DFlashWorker(config, Mapping(world_size=ranks, tp_size=ranks, rank=rank))


def spec_metadata(requests=1, advanced=False):
    return types.SimpleNamespace(
        wants_advanced_draft_sampling=advanced,
        is_all_greedy_sample=not advanced,
        is_cuda_graph=False,
        draft_vocab_size=VOCAB,
        vocab_size=VOCAB,
        batch_indices_cuda=torch.arange(requests, dtype=torch.int32, device="cuda"),
    )


@pytest.fixture
def single_process(monkeypatch):
    """The install's TP agreement in one process: each emulated rank agrees with itself."""
    monkeypatch.setattr(draft_head, "agreed_reason", lambda mapping, reason, transform: reason)


def install_rank(rank, local, full, ranks=TP, config=MUSESPARK, layout="halves"):
    """One emulated rank: its worker and drafter with the draft head installed."""
    worker = make_worker(rank, ranks)
    drafter = make_drafter(config, StubHead(local, ranks))
    llm = types.SimpleNamespace(spec_worker=worker, draft_model=drafter)
    exchange = Exchange(full, ranks, None)
    assert draft_head.install(llm, make_allreduce=lambda mapping: exchange)
    exchange.t = worker.__dict__.get(draft_head.TRANSFORM_ATTR)
    if layout == "int32":
        worker.__dict__["_draft_head_allreduce"] = None
    elif layout == "rank":
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.wide_draft_topk import (  # noqa: E501
            exchange as wide,
        )

        assert wide.install(llm)
    return llm


def patch_allgather(monkeypatch, full, ranks, t, gathered=None):
    """The head's NCCL all-gather: keys along dim 0; ``gathered`` (or a raise) along dim -1."""

    def allgather(tensor, mapping, dim=-1, sizes=None):
        if dim == 0:
            return shard_keys(full, ranks, t)
        if gathered is None:
            raise AssertionError("the fast path must not all-gather the vocabulary")
        return gathered.clone()

    monkeypatch.setattr(draft_head, "allgather", allgather)


def drafter_step(worker, drafter, hidden, anchor, md):
    """``DFlashWorker._forward_impl`` from the drafter's logits processor to the draft tokens.

    The statements the pin test (``test_modeling_v2_draft_transform_pin.py``)
    holds the built-in forward to.
    """
    num_gens = hidden.shape[0] // K
    gen_logits = drafter.logits_processor(hidden, drafter.lm_head, None, True)
    apply_logit_transform = getattr(drafter, "apply_draft_logit_transform", None)
    if apply_logit_transform is not None:
        gen_logits = apply_logit_transform(gen_logits)
    vocab_size = gen_logits.shape[-1]
    gen_logits = gen_logits.reshape(num_gens, K, vocab_size)
    gen_logits = worker._refine_block_logits(drafter, gen_logits, {}, md)
    gen_logits = worker._apply_dflash2_selector(
        drafter, gen_logits, hidden.reshape(num_gens, K, -1), anchor, md
    )
    return worker.sample_draft_tokens(gen_logits, md, num_gens, num_contexts=0)


# --------------------------------------------------------------------------- end to end


@pytest.mark.parametrize("kind", FAMILIES)
@pytest.mark.parametrize("layout", ["int32", "halves", "rank"])
@pytest.mark.parametrize("ranks", [4, 8])
@pytest.mark.parametrize("requests", [1, 2, 8, 24])
def test_selector_draft_tokens_equal_builtin(requests, ranks, layout, kind, monkeypatch):
    """``selector_draft_tokens`` on every rank's bf16 shard with the transform = the built-in drafts."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.wide_draft_topk import (  # noqa: E501
        exchange as wide,
    )

    keep = make_drafter(MUSESPARK, None)
    t = logit_transform.build(keep, torch.device("cuda"))
    shards, block_hidden, anchor = draft_step(
        requests, K, kind, seed_of(f"e2e{requests}{ranks}{kind}"), ranks=ranks
    )
    full = torch.cat(shards, dim=-1)
    expected = builtin_draft_tokens(shards, block_hidden, anchor, MUSESPARK)
    ref_unary, ref_ids = builtin_candidates(shards, requests, K, MUSESPARK)
    patch_allgather(monkeypatch, full, ranks, t)
    allreduce = {
        "int32": None,
        "halves": Exchange(full, ranks, t),
        "rank": wide.RankTopKExchange(Exchange(full, ranks, t)),
    }[layout]
    for rank in range(ranks):
        seen = {}

        def walk(edges, unary, ids, seen=seen):
            seen["unary"], seen["ids"] = unary.clone(), ids.clone()
            return draft_head.draft_path_walk(edges, unary, ids)

        got = draft_head.selector_draft_tokens(
            shards[rank].view(requests, K, -1),
            block_hidden,
            anchor,
            selector(),
            types.SimpleNamespace(tp_size=ranks, tp_rank=rank),
            allreduce,
            walk=walk,
            transform=t,
        )
        assert torch.equal(got, expected), f"rank {rank}: {got.tolist()} != {expected.tolist()}"
        assert torch.equal(seen["ids"], ref_ids)
        assert_same_floats(seen["unary"], ref_unary)


def test_transform_is_a_required_keyword():
    with pytest.raises(TypeError):
        draft_head.selector_draft_tokens(None, None, None, None, None)


@pytest.mark.parametrize("kind", ["gemm100", "straddle_up", "saturated"])
@pytest.mark.parametrize("requests", [1, 8])
def test_path_walk_tail_takes_the_builtin_inputs(requests, kind, monkeypatch):
    """The fold's tail walk on the fast path's unary values = the tail on the built-in candidates."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.draft_path_walk_tail import (  # noqa: E501
        draft_path_walk_tail,
    )

    keep = make_drafter(MUSESPARK, None)
    t = logit_transform.build(keep, torch.device("cuda"))
    shards, block_hidden, anchor = draft_step(requests, K, kind, seed_of(f"tail{requests}{kind}"))
    full = torch.cat(shards, dim=-1)
    g = torch.Generator(device="cuda").manual_seed(requests)
    accepted = torch.randint(0, VOCAB, (requests, K + 1), generator=g, device="cuda").int()
    num_accepted = torch.randint(1, K + 2, (requests,), generator=g, device="cuda").int()
    batch_indices = torch.arange(requests, dtype=torch.int32, device="cuda")
    kv_lens = torch.randint(10, 100, (requests,), generator=g, device="cuda").int()
    rewind = (1 - num_accepted).int()

    # The built-in candidates and einsum operands, then the tail's reference.
    sel = selector()
    ref_unary, ref_ids = builtin_candidates(shards, requests, K, MUSESPARK)
    gate = sel.hidden_projection(block_hidden.to(sel.hidden_projection.weight.dtype))
    successors = sel.successor_codebook[ref_ids].float()
    predecessor_ids = torch.cat(
        (anchor.long()[:, None, None].expand(-1, 1, TOP_K), ref_ids[:, :-1]), 1
    )
    predecessors = sel.predecessor_codebook[predecessor_ids].float() * gate.float()[:, :, None]
    edges = torch.einsum("blpr,blcr->blpc", predecessors, successors)
    ref_kv = kv_lens.clone()
    ref_tokens, ref_next = draft_path_walk_tail.reference(
        edges, ref_unary, ref_ids, accepted, num_accepted, batch_indices, ref_kv, rewind
    )
    patch_allgather(monkeypatch, full, TP, t)
    ours_kv = kv_lens.clone()

    def walk(edges, unary, ids):
        return draft_path_walk_tail(
            edges, unary, ids, accepted, num_accepted, batch_indices, ours_kv, rewind
        )

    tokens, next_tokens = draft_head.selector_draft_tokens(
        shards[5].view(requests, K, -1),
        block_hidden,
        anchor,
        sel,
        types.SimpleNamespace(tp_size=TP, tp_rank=5),
        Exchange(full, TP, t),
        walk=walk,
        transform=t,
    )
    assert torch.equal(tokens, ref_tokens) and torch.equal(next_tokens, ref_next)
    assert torch.equal(ours_kv, ref_kv)
    assert torch.equal(tokens, builtin_draft_tokens(shards, block_hidden, anchor, MUSESPARK))


# --------------------------------------------------------------------------- the gate


def _forbid_full_vocab_topk(monkeypatch):
    real_topk = torch.topk

    def topk(x, *args, **kwargs):
        if x.shape[-1] == VOCAB:
            raise AssertionError("the fast path must not run torch.topk over the full vocabulary")
        return real_topk(x, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", topk)


@pytest.mark.parametrize("layout", ["int32", "halves", "rank"])
@pytest.mark.parametrize("requests", [1, 5])
def test_the_fast_path_runs_with_the_transform_on(requests, layout, monkeypatch, single_process):
    shards, block_hidden, anchor = draft_step(
        requests, K, "gemm100", seed_of(f"gate{layout}{requests}")
    )
    full = torch.cat(shards, dim=-1)
    expected = builtin_draft_tokens(shards, block_hidden, anchor, MUSESPARK)
    hidden = block_hidden.view(requests * K, HIDDEN)
    llms = [install_rank(rank, shards[rank], full, layout=layout) for rank in range(TP)]
    t = llms[0].spec_worker.__dict__[draft_head.TRANSFORM_ATTR]
    for llm in llms:
        # The drafter instance carries the deferring transform; the class is unchanged.
        assert isinstance(
            vars(llm.draft_model)[draft_head.TRANSFORM_METHOD], draft_head._DeferredTransform
        )
        assert DFlashForCausalLM.apply_draft_logit_transform is not None
    patch_allgather(monkeypatch, full, TP, t)
    _forbid_full_vocab_topk(monkeypatch)
    calls = []
    real = draft_head.selector_draft_tokens

    def spy(*args, **kwargs):
        calls.append(kwargs["transform"])
        return real(*args, **kwargs)

    monkeypatch.setattr(draft_head, "selector_draft_tokens", spy)
    md = spec_metadata(requests)
    for rank, llm in enumerate(llms):
        worker, drafter = llm.spec_worker, llm.draft_model
        out = drafter.logits_processor(hidden, drafter.lm_head, None, True)
        assert out.dtype == torch.bfloat16
        handed = drafter.apply_draft_logit_transform(out)
        assert handed is out, "the deferring transform must hand the shard through"
        block = handed.reshape(requests, K, -1)
        refined = worker._refine_block_logits(drafter, block, {}, md)
        assert refined.dtype == torch.bfloat16 and refined.data_ptr() == out.data_ptr()
        assert worker._shard_selector_applies(refined, md, VOCAB)
        stand_in = worker._apply_dflash2_selector(drafter, refined, block_hidden, anchor, md)
        tokens = worker.sample_draft_tokens(stand_in, md, requests, num_contexts=0)
        assert torch.equal(tokens, expected), f"rank {rank}"
        assert calls[-1] is worker.__dict__[draft_head.TRANSFORM_ATTR]
        # The whole sequence once more, as the built-in forward runs it.
        assert torch.equal(drafter_step(worker, drafter, hidden, anchor, md), expected)


def test_the_fast_path_runs_under_the_fold_tail(monkeypatch, single_process):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.verify_fold import (  # noqa: E501
        fold,
    )

    requests = 2
    shards, block_hidden, anchor = draft_step(requests, K, "straddle_up", seed_of("fold"))
    full = torch.cat(shards, dim=-1)
    expected = builtin_draft_tokens(shards, block_hidden, anchor, MUSESPARK)
    hidden = block_hidden.view(requests * K, HIDDEN)
    g = torch.Generator(device="cuda").manual_seed(3)
    accepted = torch.randint(0, VOCAB, (requests, K + 1), generator=g, device="cuda").int()
    num_accepted = torch.randint(1, K + 2, (requests,), generator=g, device="cuda").int()
    llms = {rank: install_rank(rank, shards[rank], full) for rank in (0, 6)}
    for llm in llms.values():
        assert fold.install(llm.spec_worker, verify_fold=False, path_walk_tail=True)
        assert draft_head._defers_transform(type(llm.spec_worker))
    t = llms[0].spec_worker.__dict__[draft_head.TRANSFORM_ATTR]
    patch_allgather(monkeypatch, full, TP, t)
    _forbid_full_vocab_topk(monkeypatch)
    for rank, llm in llms.items():
        worker, drafter = llm.spec_worker, llm.draft_model
        md = spec_metadata(requests)
        kv_lens = torch.full((requests,), 50, dtype=torch.int32, device="cuda")
        attn = types.SimpleNamespace(num_contexts=0, num_seqs=requests, kv_lens_cuda=kv_lens)
        worker._kv_rewind_amount = (1 - num_accepted).int()
        worker._kv_rewind_pending, worker._kv_rewind_nc, worker._kv_rewind_bs = True, 0, requests
        step = fold._Step(attn_metadata=attn, spec_metadata=md, draft_model=drafter, warmup=False)
        step.accepted, step.num_accepted = accepted, num_accepted
        worker.__dict__["_fold_step"] = step
        try:
            tokens = drafter_step(worker, drafter, hidden, anchor, md)
        finally:
            worker.__dict__["_fold_step"] = None
        assert step.tail is not None and torch.equal(tokens, expected), f"rank {rank}"
        assert torch.equal(kv_lens, (50 - (1 - num_accepted)).clamp(min=0).int())


def test_the_real_lm_head_hands_its_shard_through(single_process):
    """With the gathered column-parallel ``LMHead`` the processor's shard reaches the selector untouched."""
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm.mapping import Mapping

    rank = 3
    head = LMHead(
        VOCAB,
        HIDDEN,
        dtype=torch.bfloat16,
        mapping=Mapping(world_size=TP, tp_size=TP, rank=rank),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        reduce_output=False,
    ).cuda()
    with torch.no_grad():
        head.weight.copy_((torch.randn_like(head.weight, dtype=torch.float32) * 0.05).bfloat16())
    worker = make_worker(rank)
    drafter = make_drafter(MUSESPARK, head)
    llm = types.SimpleNamespace(spec_worker=worker, draft_model=drafter)
    assert draft_head.install(llm, make_allreduce=lambda mapping: object())
    hidden = torch.randn(K, HIDDEN, device="cuda").bfloat16()
    out = drafter.logits_processor(hidden, head, None, True)
    assert out.dtype == torch.bfloat16 and out.shape == (K, VOCAB // TP)
    assert drafter.apply_draft_logit_transform(out) is out
    refined = worker._refine_block_logits(drafter, out.reshape(1, K, -1), {}, spec_metadata())
    assert refined.dtype == torch.bfloat16 and worker.__dict__[draft_head.OWED_ATTR]


# --------------------------------------------------------------------------- built-in paths


def _one_rank(monkeypatch, kind="gemm400", requests=1, rank=2):
    shards, block_hidden, anchor = draft_step(requests, K, kind, seed_of(f"fallback{kind}{rank}"))
    full = torch.cat(shards, dim=-1)
    llm = install_rank(rank, shards[rank], full)
    worker, drafter = llm.spec_worker, llm.draft_model
    hidden = block_hidden.view(requests * K, HIDDEN)
    return shards, block_hidden, anchor, full, worker, drafter, hidden


def _builtin_shard(shard, requests):
    return builtin_transform(MUSESPARK, shard).view(requests, K, -1)


def test_sampled_drafts_get_the_builtin_transform(monkeypatch, single_process):
    shards, _, _, _, worker, drafter, hidden = _one_rank(monkeypatch)
    out = drafter.logits_processor(hidden, drafter.lm_head, None, True)
    handed = drafter.apply_draft_logit_transform(out)
    refined = worker._refine_block_logits(
        drafter, handed.reshape(1, K, -1), {}, spec_metadata(advanced=True)
    )
    # What the inherited _sample_dflash2_block receives: the built-in fp32 shard, bit for bit.
    assert refined.dtype == torch.float32
    assert_same_floats(refined, _builtin_shard(shards[2], 1))
    assert not worker.__dict__[draft_head.OWED_ATTR]


@pytest.mark.parametrize("why", ["d2t", "stride"])
def test_layouts_the_selector_cannot_take_get_the_builtin_path(why, monkeypatch, single_process):
    shards, block_hidden, anchor, full, worker, drafter, hidden = _one_rank(monkeypatch)
    md = spec_metadata()
    out = drafter.logits_processor(hidden, drafter.lm_head, None, True)
    handed = drafter.apply_draft_logit_transform(out)
    block = handed.reshape(1, K, -1)
    if why == "d2t":
        worker._d2t = torch.zeros(VOCAB, dtype=torch.int64, device="cuda")
    else:
        block = block.as_strided(block.shape, (block.shape[-1] * K, 1, K), 0)
        assert block.stride(-1) != 1 and block.data_ptr() == out.data_ptr()
    refined = worker._refine_block_logits(drafter, block, {}, md)
    assert refined.dtype == torch.float32
    assert_same_floats(refined, builtin_transform(MUSESPARK, block))
    if why == "d2t":
        gathered = builtin_full_logits(shards, 1, K, MUSESPARK)
        patch_allgather(monkeypatch, full, TP, None, gathered=gathered)
        got = worker._apply_dflash2_selector(drafter, refined, block_hidden, anchor, md)
        ref = DFlashForCausalLM.select_candidate_path(
            DraftStub(selector()),
            gathered.clone(),
            *reversed(torch.topk(gathered, TOP_K, dim=-1)),
            block_hidden,
            anchor.long(),
        )
        assert torch.equal(got, ref)


def test_any_other_tensor_gets_the_builtin_transform(monkeypatch, single_process):
    shards, _, _, _, worker, drafter, hidden = _one_rank(monkeypatch)
    out = drafter.logits_processor(hidden, drafter.lm_head, None, True)
    other = out.clone()
    assert_same_floats(
        drafter.apply_draft_logit_transform(other), builtin_transform(MUSESPARK, other)
    )
    # The shard itself once the record is spent.
    assert_same_floats(drafter.apply_draft_logit_transform(out), builtin_transform(MUSESPARK, out))
    # A full-vocab bf16 output of the wrapped processor is never deferred.
    full_vocab = torch.randn(K, VOCAB, device="cuda").bfloat16()

    class Inner(torch.nn.Module):
        def forward(self, *args):
            return full_vocab

    drafter.logits_processor.inner = Inner()
    assert drafter.logits_processor(hidden, None, None, True) is full_vocab
    assert draft_head.SHARD_ATTR not in vars(drafter.logits_processor)
    assert_same_floats(
        drafter.apply_draft_logit_transform(full_vocab), builtin_transform(MUSESPARK, full_vocab)
    )
    # A drafter whose attributes moved away from the tables' transform is not deferred.
    out = drafter.logits_processor(hidden, drafter.lm_head, None, True)
    drafter.logit_softcap = 30.0
    handed = drafter.apply_draft_logit_transform(out)
    assert handed is not out and handed.dtype == torch.float32
    assert_same_floats(handed, builtin_transform((MUSESPARK[0], 30.0), out))


def test_a_stale_record_changes_nothing(monkeypatch, single_process):
    shards, _, _, _, worker, drafter, hidden = _one_rank(monkeypatch)
    out = drafter.logits_processor(hidden, drafter.lm_head, None, True)
    drafter.__dict__[draft_head.DEFERRED_ATTR] = (out.data_ptr() + 2, out.numel())
    refined = worker._refine_block_logits(drafter, out.reshape(1, K, -1), {}, spec_metadata())
    assert refined.dtype == torch.float32
    assert_same_floats(refined, _builtin_shard(shards[2], 1))


def test_the_selector_applies_the_transform_without_refine(monkeypatch, single_process):
    """Bypassing refine: bf16 logits at the selector are untransformed, so it still drafts exactly."""
    shards, block_hidden, anchor, full, worker, drafter, hidden = _one_rank(
        monkeypatch, "saturated"
    )
    t = worker.__dict__[draft_head.TRANSFORM_ATTR]
    patch_allgather(monkeypatch, full, TP, t)
    out = drafter.logits_processor(hidden, drafter.lm_head, None, True)
    handed = drafter.apply_draft_logit_transform(out)
    md = spec_metadata()
    stand_in = worker._apply_dflash2_selector(
        drafter, handed.reshape(1, K, -1), block_hidden, anchor, md
    )
    tokens = worker.sample_draft_tokens(stand_in, md, 1, num_contexts=0)
    assert torch.equal(tokens, builtin_draft_tokens(shards, block_hidden, anchor, MUSESPARK))


def test_a_step_that_owes_the_transform_refuses_to_sample(monkeypatch, single_process):
    shards, _, _, _, worker, drafter, hidden = _one_rank(monkeypatch)
    out = drafter.logits_processor(hidden, drafter.lm_head, None, True)
    handed = drafter.apply_draft_logit_transform(out)
    refined = worker._refine_block_logits(drafter, handed.reshape(1, K, -1), {}, spec_metadata())
    assert refined.dtype == torch.bfloat16
    with pytest.raises(draft_head.DraftTransformStateError):
        worker.sample_draft_tokens(refined, spec_metadata(), 1, num_contexts=0)


def test_a_class_composed_above_with_its_own_refine_is_not_deferred(monkeypatch, single_process):
    shards, _, _, _, worker, drafter, hidden = _one_rank(monkeypatch)

    class Above(type(worker)):
        def _refine_block_logits(self, draft_model, gen_logits, inputs, spec_metadata):
            return super()._refine_block_logits(draft_model, gen_logits, inputs, spec_metadata)

    worker.__class__ = Above
    assert not draft_head._defers_transform(Above)
    out = drafter.logits_processor(hidden, drafter.lm_head, None, True)
    handed = drafter.apply_draft_logit_transform(out)
    assert handed.dtype == torch.float32
    assert_same_floats(handed, builtin_transform(MUSESPARK, out))


# --------------------------------------------------------------------------- CUDA graph


@pytest.mark.parametrize("tail", [False, True])
def test_cuda_graph_replay_follows_new_logits(tail, monkeypatch, single_process):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.verify_fold import (  # noqa: E501
        fold,
    )

    requests, rank = 1, 2
    shards, block_hidden, anchor = draft_step(requests, K, "gemm100", seed_of(f"graph{tail}"))
    full = torch.cat(shards, dim=-1)
    shard = VOCAB // TP
    local = full[:, rank * shard : (rank + 1) * shard]
    llm = install_rank(rank, local, full)
    worker, drafter = llm.spec_worker, llm.draft_model
    t = worker.__dict__[draft_head.TRANSFORM_ATTR]
    tables = (t.canon.data_ptr(), t.values.data_ptr())
    patch_allgather(monkeypatch, full, TP, t)
    md = spec_metadata(requests)
    hidden = block_hidden.view(requests * K, HIDDEN)
    kv_lens = torch.full((requests,), 50, dtype=torch.int32, device="cuda")
    accepted = torch.zeros(requests, K + 1, dtype=torch.int32, device="cuda")
    num_accepted = torch.ones(requests, dtype=torch.int32, device="cuda")
    if tail:
        assert fold.install(worker, verify_fold=False, path_walk_tail=True)

    def step():
        if tail:
            attn = types.SimpleNamespace(num_contexts=0, num_seqs=requests, kv_lens_cuda=kv_lens)
            worker._kv_rewind_amount = (1 - num_accepted).int()
            worker._kv_rewind_pending, worker._kv_rewind_nc, worker._kv_rewind_bs = (
                True,
                0,
                requests,
            )
            s = fold._Step(attn_metadata=attn, spec_metadata=md, draft_model=drafter, warmup=False)
            s.accepted, s.num_accepted = accepted, num_accepted
            worker.__dict__["_fold_step"] = s
        try:
            return drafter_step(worker, drafter, hidden, anchor, md)
        finally:
            worker.__dict__["_fold_step"] = None

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    assert worker.__dict__[draft_head.TRANSFORM_ATTR] is t
    assert (t.canon.data_ptr(), t.values.data_ptr()) == tables
    for replay, kind in enumerate(("gemm400", "straddle_up", "saturated", "ladder", "special_nan")):
        new_shards, new_hidden, new_anchor = draft_step(requests, K, kind, 60 + replay)
        full.copy_(torch.cat(new_shards, dim=-1))
        block_hidden.copy_(new_hidden)
        anchor.copy_(new_anchor)
        graph.replay()
        expected = builtin_draft_tokens(new_shards, new_hidden, new_anchor, MUSESPARK)
        assert torch.equal(out, expected), f"replay {replay} ({kind})"


# --------------------------------------------------------------------------- install


def test_install_is_idempotent(single_process):
    shards, _, _ = draft_step(1, K, "gemm3", 5)
    full = torch.cat(shards, dim=-1)
    llm = install_rank(0, shards[0], full)
    t = llm.spec_worker.__dict__[draft_head.TRANSFORM_ATTR]
    shadow = vars(llm.draft_model)[draft_head.TRANSFORM_METHOD]
    assert draft_head.install(llm, make_allreduce=lambda mapping: 1 / 0)
    assert llm.spec_worker.__dict__[draft_head.TRANSFORM_ATTR] is t
    assert vars(llm.draft_model)[draft_head.TRANSFORM_METHOD] is shadow


def test_an_identity_transform_installs_nothing_new(monkeypatch, single_process):
    shards, block_hidden, anchor = draft_step(1, K, "gemm3", 6)
    full = torch.cat(shards, dim=-1)
    llm = install_rank(1, shards[1], full, config=(1.0, None))
    worker, drafter = llm.spec_worker, llm.draft_model
    assert draft_head.TRANSFORM_ATTR not in worker.__dict__
    assert draft_head.TRANSFORM_METHOD not in vars(drafter)
    seen = []
    real = draft_head.draft_shard_topk

    class Spy:
        def __call__(self, *args, **kwargs):
            seen.append(sorted(kwargs))
            return real(*args, **kwargs)

    monkeypatch.setattr(draft_head, "draft_shard_topk", Spy())
    patch_allgather(monkeypatch, full, TP, None)
    md = spec_metadata()
    tokens = drafter_step(worker, drafter, block_hidden.view(K, HIDDEN), anchor, md)
    assert torch.equal(tokens, builtin_draft_tokens(shards, block_hidden, anchor))
    assert seen and all("canon" not in kw for kw in seen)


def test_tie_order_probe_agrees_with_this_torch_under_the_transform():
    keep = make_drafter(MUSESPARK, None)
    t = logit_transform.build(keep, torch.device("cuda"))
    assert draft_head.tie_order_reason(selector(), TP, transform=t) is None
    assert draft_head.tie_order_reason(selector(), 4, transform=t) is None


def _stable_topk(x, k, dim=-1):
    """A top-k that breaks ties by ascending index: torch's selection, another order."""
    values, ids = torch.sort(x, dim=dim, descending=True, stable=True)
    return values.narrow(dim, 0, k), ids.narrow(dim, 0, k)


def test_tie_order_probe_has_teeth_under_the_transform(monkeypatch):
    keep = make_drafter(MUSESPARK, None)
    t = logit_transform.build(keep, torch.device("cuda"))
    reason = draft_head.tie_order_reason(selector(), TP, topk=_stable_topk, transform=t)
    assert reason is not None and "orders top-k ties differently" in reason
    # Without canonical keys the transform probe's saturated ties come out wrong.
    monkeypatch.setattr(draft_head, "canon_kwargs", lambda transform: {})
    reason = draft_head.tie_order_reason(selector(), TP, transform=t)
    assert reason is not None and "probe 1" in reason


def test_transform_probe_logits_hold_the_tie_families():
    keep = make_drafter(MUSESPARK, None)
    t = logit_transform.build(keep, torch.device("cuda"))
    probe = draft_head.transform_probe_logits(VOCAB, VOCAB // TP, t)
    assert probe.dtype == torch.bfloat16 and probe.shape == (draft_head.PROBE_ROWS, VOCAB)
    again = draft_head.transform_probe_logits(VOCAB, VOCAB // TP, t)
    assert torch.equal(probe.view(torch.int16), again.view(torch.int16)), "not deterministic"
    transformed = t.real(probe.cuda())
    # Rows 0, 1, 2, 3, 6: more than top_k candidates share the row's largest transformed value.
    for row in (0, 1, 2, 3, 6):
        top = transformed[row].max()
        assert (transformed[row] == top).sum() > TOP_K, row
        tied = probe[row][transformed[row].cpu() == top.cpu()]
        assert tied.view(torch.int16).unique().numel() > 1, row


def test_a_refused_install_changes_nothing(monkeypatch, single_process):
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    monkeypatch.setattr(
        draft_head, "tie_order_reason", lambda *args, **kwargs: "probe rows [3] differ"
    )
    worker = make_worker(0)
    shards, _, _ = draft_step(1, K, "gemm3", 8)
    drafter = make_drafter(MUSESPARK, StubHead(shards[0], TP))
    inner = drafter.logits_processor
    llm = types.SimpleNamespace(spec_worker=worker, draft_model=drafter)
    built = []
    with pytest.raises(ValueError, match="probe rows"):
        draft_head.install(llm, make_allreduce=lambda mapping: built.append(mapping))
    assert type(worker) is DFlashWorker and drafter.logits_processor is inner and not built
    assert draft_head.TRANSFORM_ATTR not in worker.__dict__
    assert draft_head.TRANSFORM_METHOD not in vars(drafter)


def test_ranks_whose_tables_differ_refuse(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        dispatch,
    )

    keep = make_drafter(MUSESPARK, None)
    t = logit_transform.build(keep, torch.device("cuda"))
    mine = logit_transform.digest(t)
    monkeypatch.setattr(dispatch, "agreed_union", lambda mapping, local: [mine, "another"])
    monkeypatch.setattr(dispatch, "agreed_refusal", lambda mapping, local: local)
    reason = draft_head.agreed_reason(object(), None, t)
    assert reason is not None and "differ in their drafter logit transform" in reason
    monkeypatch.setattr(dispatch, "agreed_union", lambda mapping, local: list(local))
    assert draft_head.agreed_reason(object(), None, t) is None
