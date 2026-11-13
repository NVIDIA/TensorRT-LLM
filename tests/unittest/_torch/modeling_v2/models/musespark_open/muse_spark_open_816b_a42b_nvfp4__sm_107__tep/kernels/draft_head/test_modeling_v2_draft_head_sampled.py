# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: sampled drafts on the vocab-sharded draft head (fast path ``draft_head_sampled``).

With the fast path installed, a draft step that samples its drafts (rejection
sampling of a non-greedy batch) keeps the drafter's bf16 shard untransformed,
builds the selector's lattice on the shards and walks it with
``draft_path_sample``: no full-vocab all-gather, and the draft tokens and the
``draft_probs`` rows are bit for bit what the built-in
``DFlashWorker._sample_dflash2_block`` computes on the gathered, transformed
full-vocab logits (``torch.topk`` over the full vocabulary, as on one rank).
Without the fast path, a step in another sampling mode and a full-vocab step
take the built-in method on the drafter's transformed logits.

TP ranks are emulated in one process as in ``test_modeling_v2_draft_head_transform.py``.
"""

import types

import pytest
import torch

__extra_import_path__ = ["."]
from _draft_head_cases import (  # noqa: E402
    HIDDEN,
    MUSESPARK,
    TOP_K,
    VOCAB,
    assert_same_floats,
    builtin_full_logits,
    draft_head,
    draft_step,
    seed_of,
)
from test_modeling_v2_draft_head_transform import K, install_rank, patch_allgather  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec import (  # noqa: E402,E501
    head_sampler,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker  # noqa: E402
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode  # noqa: E402

pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 7),
    reason="the sm107 head sampler library is built for sm_107a",
)

#: The window slot of draft position 0 (after a request's K + 1 target rows).
SLOT_BASE = K + 1


@pytest.fixture
def single_process(monkeypatch):
    """The installs' TP agreements in one process: each emulated rank agrees with itself."""
    monkeypatch.setattr(draft_head, "agreed_reason", lambda mapping, reason, transform: reason)
    monkeypatch.setattr(draft_head, "agreed_refusal", lambda mapping, reason: reason)


def sampled_metadata(requests, seed, mode=AdvancedSamplingMode.FULL):
    """The spec metadata ``_sample_dflash2_block`` reads, for ``requests`` sampled requests."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = requests + 2
    slots = requests + 3
    return types.SimpleNamespace(
        wants_advanced_draft_sampling=True,
        is_all_greedy_sample=False,
        is_cuda_graph=False,
        draft_vocab_size=VOCAB,
        vocab_size=VOCAB,
        batch_indices_cuda=torch.arange(requests, dtype=torch.int32, device="cuda"),
        advanced_sampling_mode=mode,
        # The checkpoint's generation config for one request in three, filters on the rest.
        request_temperatures=torch.where(
            torch.arange(rows, device="cuda") % 3 == 0,
            1.0,
            torch.rand(rows, generator=g, device="cuda") + 0.5,
        ),
        request_top_ks=torch.where(
            torch.arange(rows, device="cuda") % 3 == 0,
            64,
            torch.randint(1, TOP_K + 1, (rows,), generator=g, device="cuda"),
        ).to(torch.int32),
        request_top_ps=torch.where(
            torch.arange(rows, device="cuda") % 3 == 0,
            1.0,
            torch.rand(rows, generator=g, device="cuda") * 0.4 + 0.6,
        ),
        request_min_ps=torch.zeros(rows, device="cuda"),
        request_seeds=torch.randint(0, 2**62, (rows,), generator=g, device="cuda"),
        request_offsets=torch.randint(0, 2**40, (rows,), generator=g, device="cuda") * 1024,
        rng_draft_slot=lambda step: SLOT_BASE + step,
        draft_probs=torch.full((slots, K, VOCAB), 0.25, device="cuda"),
        batch_slot_ids=torch.randperm(slots, generator=g, device="cuda")[:rows],
        draft_probs_last_dim=None,
    )


def _copy(md):
    out = types.SimpleNamespace(**vars(md))
    out.draft_probs = md.draft_probs.clone()
    return out


def _sampled_rank(monkeypatch, kind, requests, rank=2, sampled=True):
    shards, block_hidden, anchor = draft_step(requests, K, kind, seed_of(f"sampled{kind}{rank}"))
    full = torch.cat(shards, dim=-1)
    llm = install_rank(rank, shards[rank], full)
    if sampled:
        assert draft_head.install_sampled(llm, load=lambda mapping: head_sampler.load_library())
    patch_allgather(monkeypatch, full, 8, llm.spec_worker.__dict__.get(draft_head.TRANSFORM_ATTR))
    return shards, block_hidden, anchor, llm


def _drafts(worker, drafter, block_hidden, anchor, md):
    """``DFlashWorker._forward_impl`` from the drafter's logits processor to the sampled drafts."""
    requests = block_hidden.shape[0]
    hidden = block_hidden.view(requests * K, HIDDEN)
    gen_logits = drafter.logits_processor(hidden, drafter.lm_head, None, True)
    gen_logits = drafter.apply_draft_logit_transform(gen_logits)
    gen_logits = gen_logits.reshape(requests, K, gen_logits.shape[-1])
    gen_logits = worker._refine_block_logits(drafter, gen_logits, {}, md)
    tokens, vocab = worker._sample_dflash2_block(
        drafter, gen_logits, block_hidden, anchor, md, 0, requests
    )
    return gen_logits, tokens, vocab


def _builtin(worker, drafter, shards, block_hidden, anchor, md):
    """The built-in method on the gathered, transformed full-vocab logits."""
    requests = block_hidden.shape[0]
    full = builtin_full_logits(shards, requests, K, MUSESPARK)
    return DFlashWorker._sample_dflash2_block(
        worker, drafter, full, block_hidden, anchor, md, 0, requests
    )


def _spy(calls):
    """A stand-in for the built-in method (whose sharded top-k all-gathers over NCCL) recording its logits."""

    def spy(
        self,
        draft_model,
        gen_logits,
        block_hidden_states,
        anchor_tokens,
        md,
        num_contexts,
        batch_size,
    ):
        calls.append(gen_logits)
        return torch.zeros((batch_size - num_contexts, K), dtype=torch.int32, device="cuda"), VOCAB

    return spy


@pytest.mark.parametrize("kind", ["gemm10", "gemm400", "grid", "ladder", "edges"])
@pytest.mark.parametrize("requests", [1, 3, 8])
def test_sampled_drafts_are_the_builtin_on_full_vocab_candidates(
    requests, kind, monkeypatch, single_process
):
    shards, block_hidden, anchor, llm = _sampled_rank(monkeypatch, kind, requests)
    worker, drafter = llm.spec_worker, llm.draft_model
    md = sampled_metadata(requests, seed_of(kind) + requests)
    want_md = _copy(md)
    gen_logits, tokens, vocab = _drafts(worker, drafter, block_hidden, anchor, md)
    # The fast path took the bf16 shard (the patched all-gather raises on the vocabulary).
    assert gen_logits.dtype == torch.bfloat16 and gen_logits.shape[-1] == VOCAB // 8
    assert not worker.__dict__[draft_head.OWED_ATTR]
    want_tokens, want_vocab = _builtin(worker, drafter, shards, block_hidden, anchor, want_md)
    assert vocab == want_vocab == VOCAB
    assert torch.equal(tokens, want_tokens)
    assert_same_floats(md.draft_probs, want_md.draft_probs)
    assert md.draft_probs_last_dim == want_md.draft_probs_last_dim == VOCAB


def test_without_the_fast_path_sampled_drafts_take_the_builtin_method(monkeypatch, single_process):
    shards, block_hidden, anchor, llm = _sampled_rank(monkeypatch, "gemm400", 2, sampled=False)
    worker, drafter = llm.spec_worker, llm.draft_model
    calls = []
    monkeypatch.setattr(DFlashWorker, "_sample_dflash2_block", _spy(calls))
    _drafts(worker, drafter, block_hidden, anchor, sampled_metadata(2, 5))
    # The built-in method on the drafter's transformed fp32 shard.
    assert len(calls) == 1 and calls[0].dtype == torch.float32
    assert calls[0].shape[-1] == VOCAB // 8


def test_another_sampling_mode_takes_the_builtin_method(monkeypatch, single_process):
    shards, block_hidden, anchor, llm = _sampled_rank(monkeypatch, "gemm400", 2)
    worker, drafter = llm.spec_worker, llm.draft_model
    calls = []
    monkeypatch.setattr(DFlashWorker, "_sample_dflash2_block", _spy(calls))
    md = sampled_metadata(2, 6, mode=AdvancedSamplingMode.NO_TOPK)
    _drafts(worker, drafter, block_hidden, anchor, md)
    assert len(calls) == 1 and calls[0].dtype == torch.float32


def test_install_sampled_needs_the_draft_head():
    llm = types.SimpleNamespace(spec_worker=None, draft_model=None)
    with pytest.raises(ValueError, match="the draft head is not installed"):
        draft_head.install_sampled(llm, load=lambda mapping: "")


def test_install_sampled_is_idempotent_and_checks_the_classes(monkeypatch, single_process):
    _, _, _, llm = _sampled_rank(monkeypatch, "gemm400", 1)
    assert draft_head.install_sampled(llm, load=lambda mapping: pytest.fail("loaded twice"))
    _, _, _, other = _sampled_rank(monkeypatch, "gemm400", 1, sampled=False)

    class OwnRng(type(other.spec_worker)):
        def _rng_state_per_request(self, *args, **kwargs):
            return super()._rng_state_per_request(*args, **kwargs)

    other.spec_worker.__class__ = OwnRng
    with pytest.raises(ValueError, match="overrides _rng_state_per_request"):
        draft_head.install_sampled(other, load=lambda mapping: "")
    assert not other.spec_worker.__dict__.get(draft_head.SAMPLED_ATTR)
