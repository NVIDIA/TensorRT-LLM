# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guided decoding under DFlash2 speculation on the Kimi K3.1 target's DFlash worker.

The guided decoder masks the gathered target logits in place (``logits_bitmask``, rows of guided requests
only) before the worker's acceptance reads them. These tests pin, with a mock bitmask that forbids tokens:

* the worker hands the guided decoder the gathered full-vocab logits even when the target head returned
  its vocab shard, and its acceptance reads the masked tensor (the step's forward, context batch);
* greedy (strict) acceptance and DFlash2 block verification never emit a forbidden token at a guided
  position: a forbidden draft is rejected and the bonus / first token comes from the allowed set;
* the rows of unguided requests in the same batch are bit-identical to the unmasked run.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default import (
    modeling as k31_modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.modeling import (  # noqa: E501
    DFlashWorker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
    dflash2_proposal_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_verify import (
    dflash2_sample_and_accept,
)
from tensorrt_llm._torch.speculative.vocab_parallel import VocabShardedLogits

_CUDA = torch.cuda.is_available()
_SM10X = _CUDA and torch.cuda.get_device_capability() in ((10, 3), (10, 7))


class _MaskingGuidedDecoder:
    """The capturable guided decoder's device side: ``execute`` masks the first rows of ``logits`` in place
    with ``trtllm::logits_bitmask`` (bit set = allowed, rows with ``token_mask`` 0 untouched)."""

    def __init__(self, allowed: torch.Tensor, token_mask: torch.Tensor):
        import xgrammar

        self.vocab = allowed.shape[-1]
        self.bitmask = xgrammar.testing.bool_mask_to_bitmask(allowed)
        self.token_mask = token_mask.to(torch.int32)
        self.calls = []

    def execute(self, logits, d2t=None):
        self.calls.append(("execute", logits))
        rows = self.bitmask.shape[0]
        # The engine masks full-vocab rows; a vocab shard here would mask the wrong columns.
        assert isinstance(logits, torch.Tensor) and logits.shape[-1] == self.vocab
        torch.ops.trtllm.logits_bitmask(
            logits[:rows], self.bitmask, token_mask=self.token_mask, d2t=None
        )

    def rollback_rejected_batch(self, num_accepted_tokens):
        self.calls.append(("rollback_rejected_batch", num_accepted_tokens.clone()))


def _bare_worker(max_draft_len: int = 4) -> DFlashWorker:
    worker = object.__new__(DFlashWorker)
    worker.mapping = None
    worker.guided_decoder = None
    worker._vocab_parallel = None
    worker._draft_model_ref = None
    worker._ctx_kv_manager = None
    worker._accept_stats = None
    worker.spec_config = SimpleNamespace(max_draft_len=max_draft_len)
    worker._apply_force_accepted_tokens = lambda num, num_contexts, k, spec_metadata=None: num
    return worker


def test_shard_selection_is_off_with_a_guided_decoder():
    worker = _bare_worker()
    worker._vocab_parallel = SimpleNamespace(samples=True)
    greedy = SimpleNamespace(is_all_greedy_sample=True)
    sampled = SimpleNamespace(is_all_greedy_sample=False)
    assert worker._target_tokens_from_shard(greedy)
    assert worker._target_tokens_from_shard(sampled)
    worker._vocab_parallel = SimpleNamespace(samples=False)
    assert worker._target_tokens_from_shard(greedy)
    assert not worker._target_tokens_from_shard(sampled)
    worker.guided_decoder = object()
    worker._vocab_parallel = SimpleNamespace(samples=True)
    assert not worker._target_tokens_from_shard(greedy)
    assert not worker._target_tokens_from_shard(sampled)


class _ShardHead:
    """One rank's vocab-parallel head (the whole vocab); the gather goes through the family op."""

    def __init__(self):
        self.lm_head = SimpleNamespace(mapping=None)


@pytest.mark.skipif(not _CUDA, reason="the guided bitmask op runs on CUDA")
def test_forward_masks_gathered_logits_before_acceptance(monkeypatch):
    """A context step whose target head returned its vocab shard: the guided decoder masks the gathered
    logits, acceptance reads the masked tensor, and the matchers are rolled back after drafting."""
    vocab, batch, K = 4096, 3, 4
    worker = _bare_worker(K)
    worker._vocab_parallel = SimpleNamespace(samples=True)
    gen = torch.Generator(device="cuda").manual_seed(3)
    local = torch.randn(batch, vocab, generator=gen, device="cuda").to(torch.bfloat16)
    best = local.float().argmax(-1)
    allowed = torch.ones(batch, vocab, dtype=torch.bool, device="cuda")
    allowed[0, best[0]] = False  # request 0 is guided and its argmax is forbidden
    guided = _MaskingGuidedDecoder(allowed, torch.tensor([1, 0, 0], device="cuda"))
    worker.guided_decoder = guided
    gathered = []
    monkeypatch.setattr(
        k31_modeling,
        "allgather_module",
        lambda x, mapping, dim, sizes: gathered.append(x) or x.clone(),
    )
    seen = {}

    def accept(logits, attn_metadata, spec_metadata):
        seen["logits"] = logits
        values = logits.local.float() if isinstance(logits, VocabShardedLogits) else logits
        tokens = values.argmax(-1).to(torch.int32)
        accepted = torch.zeros(batch, K + 1, dtype=torch.int32, device="cuda")
        accepted[:, 0] = tokens
        return accepted, torch.ones(batch, dtype=torch.int32, device="cuda")

    monkeypatch.setattr(worker, "sample_and_accept_draft_tokens", accept)
    worker.get_draft_kv_cache_manager = lambda resource_manager: None
    worker._ctx_len = torch.zeros(batch, dtype=torch.long, device="cuda")
    worker._ctx_len_host = [0] * batch
    worker._req_ctx_pos = {}
    worker._ctx_buf_inited = False
    for stub in (
        "_lazy_init_ctx_buffers",
        "_refresh_ctx_block_tables",
        "_prepare_attn_metadata_for_dflash",
        "_prepare_kv_for_draft_forward",
        "_store_prefill_context",
        "write_context_onehot_draft_probs",
        "_restore_attn_metadata_from_spec_dec",
        "_apply_kv_rewind_after_draft",
    ):
        setattr(worker, stub, lambda *args, **kwargs: None)
    worker.prepare_1st_drafter_inputs = lambda **kwargs: {}
    worker._prepare_next_new_tokens = lambda accepted, *args: accepted
    spec_metadata = SimpleNamespace(
        runtime_draft_len=K,
        is_all_greedy_sample=True,
        is_cuda_graph=False,
        request_ids=None,
        batch_indices_cuda=None,
    )
    attn_metadata = SimpleNamespace(num_seqs=batch, num_contexts=batch, kv_cache_manager=None)
    out = worker._forward_impl(
        torch.zeros(batch, dtype=torch.long, device="cuda"),
        torch.zeros(1, batch, dtype=torch.long, device="cuda"),
        torch.zeros(batch, 8, device="cuda"),
        VocabShardedLogits(_ShardHead(), local),
        attn_metadata,
        spec_metadata,
        SimpleNamespace(context_kv_only=False),
    )
    assert len(gathered) == 1 and gathered[0] is local
    (name, masked), rollback = guided.calls[0], guided.calls[1]
    assert name == "execute" and isinstance(masked, torch.Tensor)
    assert masked.dtype == torch.float32 and masked.shape == (batch, vocab)
    assert seen["logits"] is masked
    assert rollback[0] == "rollback_rejected_batch"
    assert not out["logits_vocab_shard"] and out["logits"] is masked
    reference = local.float()
    reference[0, best[0]] = float("-inf")
    assert torch.equal(out["new_tokens"][:, 0].long(), reference.argmax(-1))
    assert int(out["new_tokens"][0, 0]) != int(best[0])
    assert torch.equal(out["new_tokens"][1:, 0].long(), best[1:])

    # Without a guided decoder the same all-greedy step selects from the shard.
    worker.guided_decoder = None
    gathered.clear()
    out = worker._forward_impl(
        torch.zeros(batch, dtype=torch.long, device="cuda"),
        torch.zeros(1, batch, dtype=torch.long, device="cuda"),
        torch.zeros(batch, 8, device="cuda"),
        VocabShardedLogits(_ShardHead(), local),
        attn_metadata,
        spec_metadata,
        SimpleNamespace(context_kv_only=False),
    )
    assert not gathered and out["logits_vocab_shard"] and out["logits"] is local


def _strict_metadata(drafts: torch.Tensor, K: int):
    return SimpleNamespace(
        is_ragged_verify=False,
        runtime_draft_len=K,
        draft_tokens=drafts.flatten().to(torch.int32),
        use_rejection_sampling=False,
        is_all_greedy_sample=True,
        enable_penalty=False,
    )


@pytest.mark.skipif(not _CUDA, reason="the guided bitmask op runs on CUDA")
def test_greedy_acceptance_respects_the_bitmask():
    """Strict acceptance of an all-greedy batch (one context, three generations, K = 4): request 0
    (context) and generation 1 are guided."""
    vocab, K, num_ctx, num_gens = 2048, 4, 1, 3
    rows = num_ctx + num_gens * (K + 1)
    gen = torch.Generator(device="cuda").manual_seed(11)
    logits = 4.0 * torch.randn(rows, vocab, generator=gen, device="cuda")
    best = logits.argmax(-1)
    # Every draft is the unmasked target argmax, so unguided all K drafts are accepted.
    drafts = best[num_ctx:].view(num_gens, K + 1)[:, :K].clone()
    allowed = torch.ones(rows, vocab, dtype=torch.bool, device="cuda")
    token_mask = torch.zeros(rows, dtype=torch.int32, device="cuda")
    allowed[0, best[0]] = False  # the context's first token
    token_mask[0] = 1
    g1 = num_ctx + (K + 1)  # generation 1: its second draft is forbidden after the first
    allowed[g1 + 1, drafts[1, 1]] = False
    token_mask[g1 : g1 + 2] = 1
    guided = _MaskingGuidedDecoder(allowed, token_mask)
    attn_metadata = SimpleNamespace(num_seqs=num_ctx + num_gens, num_contexts=num_ctx)

    worker = _bare_worker(K)
    plain_tokens, plain_num = worker.sample_and_accept_draft_tokens(
        logits.clone(), attn_metadata, _strict_metadata(drafts, K)
    )
    worker.guided_decoder = guided
    masked = logits.clone()
    guided.execute(masked)
    tokens, num = worker.sample_and_accept_draft_tokens(
        masked, attn_metadata, _strict_metadata(drafts, K)
    )
    assert plain_num.tolist() == [1, K + 1, K + 1, K + 1]
    # The context row: the best allowed token.
    assert int(tokens[0, 0]) != int(best[0])
    assert int(tokens[0, 0]) == int(masked[0].argmax())
    # Generation 1: the first draft is accepted, the forbidden second one rejected, and the bonus is
    # the best allowed token at that position.
    assert int(num[2]) == 2
    assert int(tokens[2, 0]) == int(drafts[1, 0])
    assert int(tokens[2, 1]) == int(masked[g1 + 1].argmax())
    assert int(tokens[2, 1]) != int(drafts[1, 1])
    # Unguided generations 0 and 2: bit-identical to the unmasked run.
    for row in (1, 3):
        n = int(plain_num[row])
        assert int(num[row]) == n
        assert torch.equal(tokens[row, :n], plain_tokens[row, :n])


class _StubWorker:
    """The engine hooks ``dflash2_sample_and_accept`` calls, as plain torch."""

    max_draft_len = 4

    def __init__(self, top_k):
        self._draft_model_ref = lambda: SimpleNamespace(
            has_candidate_selector=True, candidate_selector=SimpleNamespace(top_k=top_k)
        )

    def _reshape_draft_tokens_for_accept(self, spec_metadata, num_gens, device):
        return spec_metadata.draft_tokens.reshape(num_gens, spec_metadata.runtime_draft_len)

    def _reshape_logits_for_accept(self, logits, num_contexts, num_gens, spec_metadata):
        return logits

    def _apply_occurrence_penalties(self, logits, *args):
        return logits

    def _commit_occurrence_counts(self, accepted, *args):
        return accepted

    def _apply_force_accepted_tokens(self, num_accepted, *args, **kwargs):
        return num_accepted

    def _sample_tokens_for_batch(self, logits, *args):
        return logits.argmax(-1)

    @staticmethod
    def _zero_padding_rows(logits, *args, **kwargs):
        return logits

    @staticmethod
    def _compute_probs_from_logits(mode, logits, temperatures, top_ks, top_ps, min_ps):
        return torch.softmax(logits / temperatures.unsqueeze(-1), dim=-1)

    def _rng_state_per_request(self, spec_metadata, start, end, slot):
        return spec_metadata.request_seeds[start:end], spec_metadata.request_offsets[
            start:end
        ] + slot


@pytest.mark.skipif(
    not _SM10X, reason="the DFlash2 verify kernels are certified on sm_103 / sm_107"
)
def test_block_verification_respects_the_bitmask():
    """A DFlash2 sampling batch (one context, four generations, K = 4, the last generation greedy)
    verified by block verification: the context, generation 0 (sampled) and generation 3 (greedy) are
    guided; generations 1 and 2 are not."""
    K, k, vocab, num_ctx, num_gens = 4, 8, 64, 1, 4
    bs = num_ctx + num_gens
    rows = num_ctx + num_gens * (K + 1)
    gen = torch.Generator().manual_seed(7)
    req_t = torch.tensor([1.0, 1.0, 0.8, 1.2, 1e-5], device="cuda")  # 1e-5: greedy
    tok_t = torch.cat((req_t[:num_ctx], req_t[num_ctx:].repeat_interleave(K + 1)))
    slots = torch.tensor([5, 2, 9, 0, 3], device="cuda")
    cand = torch.stack(
        [torch.randperm(vocab, generator=gen)[:k] for _ in range(num_gens * K)]
    ).view(num_gens, K, k)
    q = torch.softmax(torch.randn(num_gens, K, k, generator=gen), dim=-1)
    pick = torch.multinomial(q.view(-1, k), 1, generator=gen).view(num_gens, K)
    drafts = cand.gather(-1, pick.unsqueeze(-1)).squeeze(-1).cuda()
    logits = torch.randn(rows, vocab, generator=gen).cuda()
    # Every generation's drafts are likely under the target, so unguided runs accept several.
    for g in range(num_gens):
        start = num_ctx + g * (K + 1)
        logits[start + torch.arange(K, device="cuda"), drafts[g]] += 6.0

    def metadata():
        return SimpleNamespace(
            is_ragged_verify=False,
            runtime_draft_len=K,
            use_rejection_sampling=True,
            is_all_greedy_sample=False,
            draft_probs=None,
            num_seq_slots=16,
            max_num_requests=8,
            batch_slot_ids=slots,
            request_temperatures=req_t,
            temperatures=tok_t,
            top_ks=torch.zeros_like(tok_t, dtype=torch.int32),
            top_ps=torch.ones_like(tok_t),
            min_ps=torch.zeros_like(tok_t),
            request_seeds=torch.full((bs,), 42, dtype=torch.int64, device="cuda"),
            request_offsets=torch.arange(bs, dtype=torch.int64, device="cuda") * 1000,
            rng_accept_slot=10,
            advanced_sampling_mode=None,
            draft_tokens=drafts.flatten().to(torch.int32),
        )

    def run(step_logits):
        worker = _StubWorker(top_k=k)
        md = metadata()
        dflash2_proposal_store(worker, md).store(slots[num_ctx:], cand.cuda(), q.cuda(), drafts)
        return dflash2_sample_and_accept(
            worker, step_logits, SimpleNamespace(num_seqs=bs, num_contexts=num_ctx), md
        )

    plain_tokens, plain_num = run(logits.clone())

    allowed = torch.ones(rows, vocab, dtype=torch.bool, device="cuda")
    token_mask = torch.zeros(rows, dtype=torch.int32, device="cuda")
    # The context: three allowed tokens, none of them its unmasked argmax.
    ctx_allowed = [
        t for t in torch.randperm(vocab, generator=gen).tolist() if t != int(logits[0].argmax())
    ][:3]
    allowed[0] = False
    allowed[0, ctx_allowed] = True
    token_mask[0] = 1
    # Generation 0 (sampled): its first draft is forbidden; five tokens are allowed at that position.
    g0 = num_ctx
    g0_allowed = [
        t for t in torch.randperm(vocab, generator=gen).tolist() if t != int(drafts[0, 0])
    ][:5]
    allowed[g0] = False
    allowed[g0, g0_allowed] = True
    token_mask[g0] = 1
    # Generation 3 (greedy): its first two drafts are allowed, the third is forbidden.
    g3 = num_ctx + 3 * (K + 1)
    allowed[g3 + 2, drafts[3, 2]] = False
    token_mask[g3 : g3 + 3] = 1
    masked = logits.clone()
    _MaskingGuidedDecoder(allowed, token_mask).execute(masked)
    tokens, num = run(masked)

    assert int(num[0]) == 1 and int(tokens[0, 0]) in ctx_allowed
    assert int(num[1]) == 1 and int(tokens[1, 0]) in g0_allowed
    # The greedy generation: drafts 1-2 match the (allowed) argmax, draft 3 is forbidden.
    assert int(plain_num[4]) == K + 1
    assert int(num[4]) == 3
    assert tokens[4, :2].tolist() == drafts[3, :2].tolist()
    assert int(tokens[4, 2]) == int(masked[g3 + 2].argmax()) != int(drafts[3, 2])
    # Every emitted token of a guided row is allowed at its position.
    for row, start in ((1, g0), (4, g3)):
        for j in range(int(num[row])):
            assert bool(allowed[start + j, tokens[row, j]])
    # Unguided generations 1 and 2: bit-identical to the unmasked run.
    for row in (2, 3):
        n = int(plain_num[row])
        assert int(num[row]) == n
        assert torch.equal(tokens[row, :n], plain_tokens[row, :n])
    assert int(plain_num[2]) > 1 or int(plain_num[3]) > 1
