# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampled decoding with the DFlash2 candidate selector: the drafting half.

At temperature ``T > 0`` the selector proposes a block by walking its scored lattice once,
sampling each slot from the candidates' distribution under the predecessor actually drawn
for the slot before:

* ``S[b, e, p, c]`` scores candidate ``c`` of slot ``e`` after candidate ``p`` of slot ``e - 1``
  (the anchor for slot 0, so every ``p`` row of slot 0 is the same).
* The walk starts at ``prev = 0``. Slot ``e`` reads the row ``S[b, e, prev]``; a sampling row
  takes ``q = softmax(row / max(T, 1e-5))`` over the ``top_k`` candidates (no top-k / top-p
  on ``q``) and picks by inverse CDF from one uniform ``u``: the index is the number of
  cumulative probabilities ``<= u``, clamped to ``top_k - 1``. A greedy row (the request
  decodes greedily) takes the first argmax and a one-hot ``q``.
* The draft token is ``C[b, e, index]`` and ``prev`` becomes the sampled index, so the next
  slot is scored under the token really proposed.

``q`` is the proposal distribution the verification (``dflash2_verify``) needs: it is sparse,
``top_k`` probabilities over the slot's candidate ids. :class:`DFlash2ProposalStore` keeps it,
the candidate ids and the drafted tokens in rows indexed by the request's sequence slot, from
the step that drafts a block to the step that verifies it.

The uniforms come from :func:`dflash2_uniforms`, a counter-based hash of the request's Philox
``(seed, offset)`` pair: every tensor-parallel rank draws the same values without a collective,
a seeded request is reproducible, and the draws are plain device ops (CUDA-graph capturable).

For the checkpoint's selector (32 candidates, a rank of whole 256 blocks) the scoring, the
uniforms, the walk and the store run as the CuTe DSL op ``ops/spec/dflash2_selector_draft``,
bitwise the torch functions here, which stay as its reference and serve other selector shapes.
"""

from typing import Callable, List, Sequence, Tuple

import torch

from tensorrt_llm._utils import async_tensor_h2d

# Temperature floor of the walk: softmax(S / max(T, 1e-5)).
MIN_WALK_TEMPERATURE = 1e-5

# SplitMix64 constants, as signed 64-bit integers (torch int64 arithmetic wraps).
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15 - (1 << 64)
_MIX_1 = 0xBF58476D1CE4E5B9 - (1 << 64)
_MIX_2 = 0x94D049BB133111EB - (1 << 64)


def _shift_right_logical(x: torch.Tensor, bits: int) -> torch.Tensor:
    return (x >> bits) & ((1 << (64 - bits)) - 1)


def _mix64(z: torch.Tensor) -> torch.Tensor:
    """SplitMix64's output function on int64 values (two's-complement wraparound)."""
    z = (z ^ _shift_right_logical(z, 30)) * _MIX_1
    z = (z ^ _shift_right_logical(z, 27)) * _MIX_2
    return z ^ _shift_right_logical(z, 31)


def dflash2_uniforms(seed: torch.Tensor, offset: torch.Tensor, count: int) -> torch.Tensor:
    """``count`` uniforms in ``[0, 1)`` per row, a pure function of the row's ``(seed, offset)``.

    A SplitMix64 stream keyed by ``mix(mix(seed) ^ offset)``: draw ``j`` is
    ``mix(key + (j + 1) * gamma)``, of which the top 24 bits become a float32 multiple of
    ``2**-24``. Rows sharing a seed are separated by their offsets.

    Args:
        seed, offset: ``[n]`` int64.
        count: draws per row.
    Returns:
        ``[n, count]`` float32.
    """
    key = _mix64(_mix64(seed.long()) ^ offset.long())
    steps = torch.arange(1, count + 1, dtype=torch.int64, device=seed.device) * _GOLDEN_GAMMA
    bits = _shift_right_logical(_mix64(key[:, None] + steps), 40)
    return bits.to(torch.float32) * (1.0 / (1 << 24))


def dflash2_score_edges(
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    gate: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Score every adjacent candidate pair of a drafted block at once, in FP32.

    ``S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>``: the drafter's logit for ``b`` plus a low-rank bilinear
    match between predecessor ``a`` and successor ``b``. Slot 0's predecessor is the anchor token.

    Args:
        predecessor_codebook, successor_codebook: ``[vocab_size, rank]``.
        candidate_ids: ``[B, K, top_k]`` candidate token ids per slot.
        unary_logits: ``[B, K, top_k]`` the drafter's logits of those candidates.
        gate: ``[B, K, rank]`` the selector's projection of each slot's hidden state.
        anchor_token_ids: ``[B]``.
    Returns:
        ``[B, K, top_k, top_k]`` FP32 scores indexed ``[batch, slot, predecessor, candidate]``.
    """
    top_k = candidate_ids.shape[-1]
    successors = successor_codebook[candidate_ids].float()
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]), dim=1
    )
    predecessors = predecessor_codebook[predecessor_ids].float()
    return unary_logits.float()[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * gate.float()[:, :, None], successors
    )


def dflash2_walk_candidate_paths(
    candidate_ids: torch.Tensor,
    edge_scores: torch.Tensor,
) -> torch.Tensor:
    """The greedy walk of the scored lattice: each slot's score row under the predecessor the walk took.

    From the anchor, each step keeps the highest-scoring candidate and conditions the next slot on it.

    Args:
        candidate_ids: ``[B, K, top_k]``.
        edge_scores: ``[B, K, top_k, top_k]`` from :func:`dflash2_score_edges`.
    Returns:
        ``[B, K, top_k]``.
    """
    batch, num_steps, top_k, _ = edge_scores.shape
    realized = edge_scores.new_empty(batch, num_steps, top_k)
    # Step 0 replicates the anchor across the predecessor axis, so any row works.
    predecessor = torch.zeros(batch, 1, 1, dtype=torch.long, device=edge_scores.device)
    for step in range(num_steps):
        scores = edge_scores[:, step].gather(1, predecessor.expand(-1, 1, top_k)).squeeze(1)
        realized[:, step] = scores
        predecessor = scores.argmax(-1).view(batch, 1, 1)
    return realized


def dflash2_selector_walk(
    candidate_ids: torch.Tensor,
    edge_scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walk the selector lattice once: one sampled (or argmax) candidate per slot.

    Args:
        candidate_ids: ``[n, slots, top_k]`` candidate token ids.
        edge_scores: ``[n, slots, top_k, top_k]`` scores ``[request, slot, predecessor,
            candidate]``.
        uniforms: ``[n, slots]`` one uniform in ``[0, 1)`` per slot.
        temperatures: ``[n]`` sampling temperatures (floored at ``MIN_WALK_TEMPERATURE``).
        greedy_mask: ``[n]`` bool, rows that take the argmax.
    Returns:
        ``(tokens [n, slots] int64, q [n, slots, top_k] float32, path [n, slots] int64)``:
        the drafted tokens, each slot's proposal distribution over its candidates, and the
        candidate index the walk took at each slot.
    """
    n, slots, top_k = candidate_ids.shape
    temperature = temperatures.float().clamp(min=MIN_WALK_TEMPERATURE)[:, None]
    greedy = greedy_mask.bool()
    previous = torch.zeros(n, dtype=torch.int64, device=edge_scores.device)
    paths, q_rows = [], []
    for slot in range(slots):
        row = (
            edge_scores[:, slot]
            .gather(1, previous.view(n, 1, 1).expand(n, 1, top_k))
            .squeeze(1)
            .float()
        )
        probs = torch.softmax(row / temperature, dim=-1)
        drawn = (
            (uniforms[:, slot : slot + 1] >= probs.cumsum(dim=-1)).sum(dim=-1).clamp_max(top_k - 1)
        )
        # torch.argmax: the first maximum, a NaN counting as the maximum.
        best = row.argmax(dim=-1)
        index = torch.where(greedy, best, drawn)
        one_hot = torch.nn.functional.one_hot(best, top_k).to(torch.float32)
        q_rows.append(torch.where(greedy[:, None], one_hot, probs))
        paths.append(index)
        previous = index
    path = torch.stack(paths, dim=1)
    tokens = candidate_ids.gather(-1, path.unsqueeze(-1)).squeeze(-1).long()
    return tokens, torch.stack(q_rows, dim=1), path


def greedy_request_mask(request_temperatures: torch.Tensor) -> torch.Tensor:
    """Requests that decode greedily: the engine gives them a temperature below the greedy
    threshold (temperature 0, ``top_k`` 1 or ``top_p`` 0 all resolve to it)."""
    from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

    return request_temperatures <= GREEDY_TEMPERATURE_THRESHOLD


class DFlash2ProposalStore:
    """The selector's proposals of the last drafted block, by request sequence slot.

    ``candidate_ids [rows, K, top_k]``, ``q [rows, K, top_k]`` and the drafted tokens
    ``[rows, K]`` are written by the drafting step and read by the next step's verification.
    A row whose stored tokens differ from the tokens being verified was not drafted by the
    walk (a context request's placeholder drafts, a slot that changed hands, a step that
    drafted nothing): :meth:`load` gives it the point mass on each verified token instead,
    which is that deterministic proposal's exact distribution. :meth:`invalidate` marks
    the rows of context requests, and of generation requests that arrive on a disaggregated
    server (:func:`dflash2_invalidate_arrived_rows`), so a recycled slot never matches.
    Candidate ids are only unique within a walked row; the verification sums over lanes, so
    that is all it needs.

    All shapes are fixed at construction (CUDA-graph capturable reads and writes).
    """

    def __init__(self, num_rows: int, max_draft_len: int, top_k: int, device):
        self.num_rows = num_rows
        self.max_draft_len = max_draft_len
        self.top_k = top_k
        self.candidate_ids = torch.zeros(
            (num_rows, max_draft_len, top_k), dtype=torch.int64, device=device
        )
        self.q = torch.zeros((num_rows, max_draft_len, top_k), dtype=torch.float32, device=device)
        self.tokens = torch.full((num_rows, max_draft_len), -1, dtype=torch.int64, device=device)

    def store(
        self,
        slot_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        q: torch.Tensor,
        tokens: torch.Tensor,
    ) -> None:
        k = candidate_ids.shape[1]
        self.candidate_ids[slot_ids, :k] = candidate_ids.long()
        self.q[slot_ids, :k] = q.float()
        self.tokens[slot_ids, :k] = tokens.long()

    def invalidate(self, slot_ids: torch.Tensor) -> None:
        self.tokens[slot_ids] = -1

    def load(
        self, slot_ids: torch.Tensor, draft_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(candidate_ids [n, K, top_k], q [n, K, top_k])`` for verifying ``draft_tokens``."""
        k = draft_tokens.shape[1]
        drafts = draft_tokens.long()
        candidate_ids = self.candidate_ids[slot_ids, :k]
        q = self.q[slot_ids, :k]
        walked = (self.tokens[slot_ids, :k] == drafts).all(dim=-1)
        # The point mass on the verified token: every lane holds the token, lane 0 carries the
        # mass. The verification only sums over lanes (q at a token, the min(r p, q) terms,
        # the residual's scatter-add), so zero-mass repeats change nothing.
        lane = torch.arange(self.top_k, dtype=torch.int64, device=drafts.device)
        point_ids = drafts[..., None].expand_as(candidate_ids)
        point_q = (lane == 0).to(torch.float32).expand_as(q)
        candidate_ids = torch.where(walked[:, None, None], candidate_ids, point_ids)
        q = torch.where(walked[:, None, None], q, point_q)
        return candidate_ids, q


def _selector_draft_model(worker):
    ref = getattr(worker, "_draft_model_ref", None)
    draft_model = ref() if ref is not None else None
    if draft_model is None or not getattr(draft_model, "has_candidate_selector", False):
        return None
    return draft_model


def dflash2_sampled_step(worker, spec_metadata) -> bool:
    """Whether this step drafts with the sampled selector walk and verifies by
    :func:`~.dflash2_verify.dflash2_block_verify`: a DFlash2 drafter, rejection sampling on
    (``use_rejection_sampling``; off, a sampling batch keeps exact-match verification) and a
    batch that is not all greedy (an all-greedy batch keeps the greedy walk and argmax
    acceptance). Host-side flags only: the same for a step and its CUDA graph."""
    return (
        spec_metadata is not None
        and bool(getattr(spec_metadata, "use_rejection_sampling", False))
        and not spec_metadata.is_all_greedy_sample
        and spec_metadata.runtime_draft_len > 0
        and _selector_draft_model(worker) is not None
    )


def dflash2_proposal_store(worker, spec_metadata) -> DFlash2ProposalStore:
    """The worker's :class:`DFlash2ProposalStore`, created on first use.

    Rows follow the engine's slot-indexed rejection buffers (the sequence-slot pool plus the
    padding row ``batch_slot_ids`` sends CUDA-graph padding requests to). First use is an
    eager forward (graph warmups run before every capture).
    """
    store = getattr(worker, "_dflash2_proposals", None)
    if store is not None:
        return store
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DFlash2 proposal state must be allocated before CUDA graph capture")
    draft_probs = getattr(spec_metadata, "draft_probs", None)
    if draft_probs is not None:
        rows = int(draft_probs.shape[0])
    else:
        rows = int(spec_metadata.num_seq_slots or spec_metadata.max_num_requests) + 1
    draft_model = _selector_draft_model(worker)
    store = DFlash2ProposalStore(
        rows,
        worker.max_draft_len,
        draft_model.candidate_selector.top_k,
        spec_metadata.batch_slot_ids.device,
    )
    worker._dflash2_proposals = store
    from tensorrt_llm.logger import logger

    logger.info(
        f"DFlash2 sampled decoding: selector walk over top-{store.top_k} candidates, "
        "block verification, "
        f"{rows} proposal slots"
    )
    return store


def dflash2_arrived_generation_rows(
    known: set,
    request_ids: Sequence[int],
    num_contexts: int,
    is_real: Callable[[int], bool],
) -> List[int]:
    """Batch rows of the generation requests not in ``known`` (the rows ``num_contexts`` on),
    then every real request of the batch (``is_real``: not a padding or dummy request) is
    added to ``known``. Host only."""
    rows = []
    for row, request_id in enumerate(request_ids):
        if not is_real(request_id):
            continue
        if row >= num_contexts and request_id not in known:
            rows.append(row)
        known.add(request_id)
    return rows


def dflash2_invalidate_arrived_rows(worker, spec_metadata, attn_metadata) -> None:
    """Invalidate the proposal rows of the generation requests that arrive on a disaggregated
    server, before their first step there.

    The store keeps, by sequence slot, the proposals of the block this server's selector walk
    drafted, and a context request's row is invalidated when it samples its first token
    (:func:`~.dflash2_verify.dflash2_sample_and_accept`). A generation request that arrives with
    its KV from a context server had its context step there: the slot it takes may still hold a
    finished request's proposals, which its first verification would read if their tokens equal
    the drafts it arrives with. Its row is invalidated here, so those drafts verify as point
    masses; from then on its rows are this server's walks.

    Runs before the forward (eager, capture or graph replay) on the execution stream, after the
    step's ``batch_slot_ids`` upload, and only on a disaggregated server
    (``kv_cache_manager.is_disagg``) with rejection sampling: elsewhere every generation request
    had its context step on this server and nothing runs. The worker keeps the ids of the
    requests it saw (``_dflash2_proposal_requests``) while the KV cache manager holds them.
    """
    manager = getattr(attn_metadata, "kv_cache_manager", None)
    request_ids = getattr(spec_metadata, "request_ids", None)
    if (
        not getattr(manager, "is_disagg", False)
        or not getattr(spec_metadata, "use_rejection_sampling", False)
        or not request_ids
    ):
        return
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID

    known = getattr(worker, "_dflash2_proposal_requests", None)
    if known is None:
        known = worker._dflash2_proposal_requests = set()
    known.intersection_update(manager.kv_cache_map)
    # CUDA-graph padding ids count down from CUDA_GRAPH_DUMMY_REQUEST_ID (a draft block apart).
    dummy_floor = CUDA_GRAPH_DUMMY_REQUEST_ID - spec_metadata.max_draft_len
    dummy_ids = getattr(manager, "_draft_dummy_request_ids", ())
    rows = dflash2_arrived_generation_rows(
        known,
        request_ids,
        attn_metadata.num_contexts,
        lambda request_id: (
            request_id != ATTENTION_DP_DUMMY_REQUEST_ID
            and request_id < dummy_floor
            and request_id not in dummy_ids
        ),
    )
    store = getattr(worker, "_dflash2_proposals", None)
    if rows and store is not None:
        slot_ids = spec_metadata.batch_slot_ids
        store.invalidate(
            slot_ids.index_select(0, async_tensor_h2d(rows, torch.long, slot_ids.device))
        )


def dflash2_sample_draft(
    worker,
    draft_model,
    gen_logits: torch.Tensor,
    block_hidden_states: torch.Tensor,
    anchor_tokens: torch.Tensor,
    spec_metadata,
    num_contexts: int,
    batch_size: int,
) -> torch.Tensor:
    """Draft tokens of a sampling batch: the selector's lattice over each slot's global top-k
    candidates, one sampled walk (:func:`dflash2_selector_walk`, greedy rows the argmax), and
    the proposals saved by sequence slot for the next step's verification.

    Args:
        gen_logits: ``[num_gens, K, vocab or shard]`` drafter logits of the block slots.
        block_hidden_states: ``[num_gens, K, hidden]`` drafter hidden states of those slots.
        anchor_tokens: ``[num_gens]`` the last accepted token of each request.
    Returns:
        ``[num_gens, K]`` int32 draft tokens.
    """
    selector = draft_model.candidate_selector
    candidate_ids, unary_logits, _ = worker._dflash2_global_top_k(
        gen_logits,
        spec_metadata,
        top_k=selector.top_k,
        full_vocab=selector.predecessor_codebook.shape[0],
    )
    return dflash2_sample_draft_from_candidates(
        worker,
        draft_model,
        candidate_ids,
        unary_logits,
        block_hidden_states,
        anchor_tokens,
        spec_metadata,
        num_contexts,
        batch_size,
    )


def dflash2_sample_draft_from_candidates(
    worker,
    draft_model,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    block_hidden_states: torch.Tensor,
    anchor_tokens: torch.Tensor,
    spec_metadata,
    num_contexts: int,
    batch_size: int,
) -> torch.Tensor:
    """:func:`dflash2_sample_draft` from the slots' global top-k candidates on: the lattice, the
    sampled walk and the proposal store.

    Args:
        candidate_ids: ``[num_gens, K, top_k]`` int64 candidate ids, in ``torch.topk``'s order.
        unary_logits: ``[num_gens, K, top_k]`` their drafter logits.
        block_hidden_states, anchor_tokens: as :func:`dflash2_sample_draft`'s.
    Returns:
        ``[num_gens, K]`` int32 draft tokens.
    """
    selector = draft_model.candidate_selector
    seed, offset = worker._rng_state_per_request(
        spec_metadata, num_contexts, batch_size, slot=spec_metadata.rng_draft_slot(0)
    )
    temperatures = spec_metadata.request_temperatures[num_contexts:batch_size]
    store = dflash2_proposal_store(worker, spec_metadata)
    slot_ids = spec_metadata.batch_slot_ids[num_contexts:batch_size]
    # The selector op takes this module's torch walk as its reference: imported on use.
    from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

    from .dflash2_selector_draft import k31_dflash2_selector_draft

    # Edge scores, the sampled walk (greedy rows the argmax) and the proposal store in one CuTe
    # DSL op, bitwise the torch selector, dflash2_selector_walk and the proposal store. The targets
    # load only selectors of 32 candidates and a rank of whole 256 blocks, the shapes it takes.
    projection = selector.hidden_projection
    return k31_dflash2_selector_draft(
        selector.predecessor_codebook,
        selector.successor_codebook,
        candidate_ids,
        unary_logits,
        projection(block_hidden_states.to(projection.weight.dtype)),
        anchor_tokens,
        seed,
        offset,
        temperatures,
        GREEDY_TEMPERATURE_THRESHOLD,
        store.candidate_ids,
        store.q,
        store.tokens,
        slot_ids,
    )
