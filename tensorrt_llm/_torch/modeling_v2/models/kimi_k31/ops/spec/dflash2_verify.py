# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampled decoding with the DFlash2 candidate selector: the verification half.

A drafted block ``d_1 .. d_g`` (``g = K`` drafts) is verified with block verification
(Sun et al. 2024, "Block Verification Accelerates Speculative Decoding", Algorithm 2) against
the target distributions ``p_0 .. p_g`` (target row ``i`` scores the token after ``i``
accepted drafts) at the request's temperature, top-k and top-p. ``q_i`` is the selector's
proposal for slot ``i`` (``d_{i+1}``): sparse, nonzero only on that slot's candidate ids.

* ``r_0 = 1``, ``r_i = min(1, r_{i-1} p_{i-1}(d_i) / q_{i-1}(d_i))`` (0 where ``q_{i-1}(d_i)``
  is not in ``(0, 1]``).
* For ``i = g`` down to ``1`` with coin ``eta_i``, accept ``tau = i`` at the first ``eta_i <
  h_i``: ``h_g = r_g``; ``h_i = R_i / (R_i + 1 - r_i)`` for ``0 < r_i < 1`` and ``h_i = r_i``
  otherwise, where ``R_i = sum_v max(r_i p_i(v) - q_i(v), 0)``. With ``q_i`` sparse on the
  candidate ids ``C_i``: ``R_i = r_i sum_v p_i(v) - sum_{c in C_i} min(r_i p_i(c), q_i(c))``.
  No acceptance gives ``tau = 0``.
* The emitted block is ``d_1 .. d_tau`` and a bonus token drawn from ``p_g`` when ``tau = g``,
  else from ``max(r_tau p_tau - q_tau, 0)`` renormalized (``r_0 = 1``), by inverse CDF from
  one uniform.

Greedy requests in a sampling batch take the strict rule: the leading drafts equal to the
target argmax, then the argmax.

The worker's step runs two CuTe DSL launches over ``[requests, K + 1, vocab]`` target
probabilities: ``ops/spec/dflash2_block_verify`` (the verification and the bonus draw, its
uniforms drawn in the kernel from each request's seed and offset as ``dflash2_uniforms`` draws
them) and ``ops/spec/dflash2_greedy_verify`` (the strict rule of the greedy rows). The torch
functions here (:func:`dflash2_block_verify`, :func:`dflash2_greedy_verify`) are their
references.
"""

from typing import Tuple

import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.common.multi_stream import do_multi_stream
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_block_verify import (
    dflash2_block_verify_seeded as block_verify_op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_greedy_verify import (
    dflash2_greedy_verify as greedy_verify_op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_proposal_load import (
    TOP_K as PROPOSAL_TOP_K,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_proposal_load import (
    k31_dflash2_proposal_load,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
    dflash2_proposal_store,
    greedy_request_mask,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_step_bookkeeping import (
    FORCE_CONSTANT,
    FORCE_COUNTER_STRIDE,
    FORCE_OFF,
    FORCE_POOL,
    FORCE_POOL_SIZE,
    FORCE_SLOT_STRIDE,
    MAX_REQUESTS,
    k31_dflash2_accept_commit,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.zero_padding_rows import (
    k31_zero_padding_rows,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm._torch.speculative import interface as _spec_interface
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

from .rubin_dflash2_worker import agree_verified_tokens


def _sample_rows(
    weights: torch.Tensor, uniforms: torch.Tensor, fallback: torch.Tensor
) -> torch.Tensor:
    """Inverse-CDF draw from unnormalized nonnegative ``weights [n, V]``: the first index whose
    cumulative weight exceeds ``u * total``. Rows without mass take ``fallback``."""
    cdf = weights.cumsum(dim=-1)
    total = cdf[:, -1]
    target = (uniforms * total).unsqueeze(-1)
    index = torch.searchsorted(cdf, target.contiguous(), right=True).squeeze(-1)
    index = index.clamp_max(weights.shape[-1] - 1)
    has_mass = (total > 0) & (weights.gather(-1, index.unsqueeze(-1)).squeeze(-1) > 0)
    return torch.where(has_mass, index, fallback)


def dflash2_block_verify(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    candidate_ids: torch.Tensor,
    q: torch.Tensor,
    coins: torch.Tensor,
    final_coins: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accepted draft count and bonus token of each request.

    Args:
        target_probs: ``[n, g + 1, V]`` float32 target distributions ``p_0 .. p_g``.
        draft_tokens: ``[n, g]`` drafted tokens ``d_1 .. d_g``.
        candidate_ids: ``[n, g, k]`` candidate ids of each slot.
        q: ``[n, g, k]`` the proposal probability of each candidate.
        coins: ``[n, g]`` uniforms; ``coins[:, i - 1]`` is ``eta_i``.
        final_coins: ``[n]`` uniforms for the bonus draw.
    Returns:
        ``(tau [n] int64, bonus [n] int64)``.
    """
    n, rows, vocab = target_probs.shape
    g = rows - 1
    if g < 1:
        raise ValueError("block verification needs at least one drafted token")
    probs = target_probs.float()
    drafts = draft_tokens.long().clamp(0, vocab - 1)
    cand = candidate_ids.long().clamp(0, vocab - 1)
    q = q.float()
    q = torch.where((q >= 0) & (q <= 1), q, 0.0)
    steps = torch.arange(1, g + 1, dtype=torch.int64, device=probs.device)

    p_draft = probs[:, :g].gather(-1, drafts.unsqueeze(-1)).squeeze(-1)
    q_draft = torch.where(cand == drafts.unsqueeze(-1), q, 0.0).sum(dim=-1)
    q_ok = (q_draft > 0) & (q_draft <= 1)

    prefix = torch.ones(n, dtype=torch.float32, device=probs.device)
    r_cols = []
    for i in range(g):
        prefix = torch.where(
            q_ok[:, i], torch.clamp(prefix * p_draft[:, i] / q_draft[:, i], max=1.0), 0.0
        )
        r_cols.append(prefix)
    r = torch.stack(r_cols, dim=1)  # r[:, i - 1] = r_i
    if g > 1:
        r_mid = r[:, : g - 1]
        p_rows = probs[:, 1:g]
        p_cand = p_rows.gather(-1, cand[:, 1:g])
        overlap = torch.minimum(r_mid.unsqueeze(-1) * p_cand, q[:, 1:g]).sum(dim=-1)
        residual = (r_mid * p_rows.sum(dim=-1) - overlap).clamp(min=0.0)
        interior = (r_mid > 0) & (r_mid < 1)
        h_mid = torch.where(interior, residual / (residual + 1.0 - r_mid), r_mid)
        h = torch.cat((h_mid, r[:, g - 1 :]), dim=1)
    else:
        h = r
    tau = ((coins < h).long() * steps).amax(dim=-1)
    r_tau = torch.cat((torch.ones_like(r[:, :1]), r), dim=1).gather(1, tau.unsqueeze(1))
    scale = r_tau.squeeze(1)

    all_accepted = tau == g
    scale = torch.where(all_accepted, torch.ones_like(scale), scale)
    p_row = probs.gather(1, tau.view(n, 1, 1).expand(n, 1, vocab)).squeeze(1)
    weights = scale.unsqueeze(-1) * p_row
    slot = tau.clamp_max(g - 1).view(n, 1, 1).expand(n, 1, cand.shape[-1])
    cand_row = cand.gather(1, slot).squeeze(1)
    q_row = q.gather(1, slot).squeeze(1)
    # max(r p - q, 0) on the candidates is r p - min(r p, q); the rest of the row keeps r p.
    overlap_row = torch.minimum(weights.gather(-1, cand_row), q_row)
    overlap_row = torch.where(all_accepted.unsqueeze(-1), 0.0, overlap_row)
    weights = weights.scatter_add(-1, cand_row, -overlap_row).clamp(min=0.0)
    bonus = _sample_rows(weights, final_coins.float(), p_row.argmax(dim=-1))
    return tau, bonus


def dflash2_greedy_verify(
    target_logits: torch.Tensor, draft_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Strict acceptance: ``tau`` = leading drafts equal to the target argmax, bonus = the
    argmax after them. ``target_logits [n, g + 1, V]``, ``draft_tokens [n, g]``."""
    best = target_logits.argmax(dim=-1)
    g = draft_tokens.shape[1]
    tau = (draft_tokens.long() == best[:, :g]).long().cumprod(dim=-1).sum(dim=-1)
    return tau, best.gather(1, tau.unsqueeze(1)).squeeze(1)


def _zero_padding_rows(worker, logits, spec_metadata, num_contexts, batch_size, rows_per_request):
    """``worker._zero_padding_rows`` of the verify rows, the padding requests' rows zeroed in place
    (``k31_zero_padding_rows``: the same values for every reader of the result, one launch writing the padding
    rows only) where the engine method's mask applies; the engine method otherwise."""
    slot_ids = getattr(spec_metadata, "batch_slot_ids", None)
    dummy_slot_row = getattr(spec_metadata, "dummy_slot_row", 0)
    num_gens = batch_size - num_contexts
    if (
        type(worker)._zero_padding_rows is not SpecWorkerBase._zero_padding_rows
        or slot_ids is None
        or dummy_slot_row <= 0
        or num_gens <= 0
        or spec_metadata.is_ragged_verify
        or logits.shape[0] != num_gens * rows_per_request
        or logits.dtype != torch.float32
        or not logits.is_contiguous()
        or logits.shape[1] % 4
        or logits.data_ptr() % 16
        or slot_ids.dtype not in (torch.int32, torch.int64)
        or torch.cuda.get_device_capability(logits.device) != (10, 3)
    ):
        return worker._zero_padding_rows(
            logits, spec_metadata, num_contexts, batch_size, rows_per_request, packed_verify=True
        )
    return k31_zero_padding_rows(
        logits, slot_ids[num_contexts:batch_size], dummy_slot_row, rows_per_request
    )


def _fused_commit_applies(worker, num_contexts, draft_tokens, spec_metadata) -> bool:
    """Whether a step's verify tail runs as ``k31_dflash2_accept_commit``: generation requests only, the engine's
    forced-count and pool constants, the drafts' and picks' layout."""
    return (
        num_contexts == 0
        and type(worker)._apply_force_accepted_tokens is SpecWorkerBase._apply_force_accepted_tokens
        and type(worker)._ensure_force_accept_rng_state
        is SpecWorkerBase._ensure_force_accept_rng_state
        and _spec_interface._FORCE_ACCEPT_RNG_POOL_SIZE == FORCE_POOL_SIZE
        and _spec_interface._FORCE_ACCEPT_RNG_COUNTER_STRIDE == FORCE_COUNTER_STRIDE
        and _spec_interface._FORCE_ACCEPT_RNG_SLOT_STRIDE == FORCE_SLOT_STRIDE
        and draft_tokens.dtype in (torch.int32, torch.int64)
        and draft_tokens.is_contiguous()
        and spec_metadata.request_temperatures.dtype == torch.float32
        and 0 < draft_tokens.shape[0] <= MAX_REQUESTS
        and torch.cuda.get_device_capability(draft_tokens.device) == (10, 3)
    )


class _VerifyOverlap:
    """The side stream of a fused verify and its fork / join events (one per worker)."""

    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.fork = torch.cuda.Event()
        self.join = torch.cuda.Event()


def _verify_overlap(worker):
    """The worker's :class:`_VerifyOverlap`, or None where the fused verify stays on the current stream
    (outside multi-stream execution, under torch.compile)."""
    if not do_multi_stream() or torch.compiler.is_compiling():
        return None
    overlap = getattr(worker, "_k31_verify_overlap", None)
    if overlap is None:
        overlap = worker._k31_verify_overlap = _VerifyOverlap()
    return overlap


def _force_args(worker, K: int, spec_metadata, device):
    """``SpecWorkerBase._apply_force_accepted_tokens``' decision for ``K`` drafts, as the arguments of
    ``k31_dflash2_accept_commit``: ``(mode, base_total, max_total, frac, counter, pool)``."""
    forced = worker.force_num_accepted_tokens
    is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
    if forced == 0.0 or is_warmup:
        return FORCE_OFF, 0, 0, 0.0, None, None
    int_part = int(forced)
    frac_part = forced - int_part
    max_total = K + 1
    base_total = min(int_part + 1, max_total)
    if frac_part > 0.0 and base_total < max_total:
        worker._ensure_force_accept_rng_state(device)
        return (
            FORCE_POOL,
            base_total,
            max_total,
            frac_part,
            worker._force_accept_rng_counter,
            worker._force_accept_rng_pool,
        )
    return FORCE_CONSTANT, base_total, max_total, frac_part, None, None


def dflash2_sample_and_accept(worker, logits, attn_metadata, spec_metadata):
    """``SpecWorkerBase.sample_and_accept_draft_tokens`` of a sampling batch on a DFlash2
    drafter: the context rows sample their first token, the generation rows verify the
    selector's block (:func:`dflash2_block_verify`, greedy rows :func:`dflash2_greedy_verify`).

    Returns ``(accepted_tokens [batch, K + 1] int32, num_accepted_tokens [batch] int32)``
    with the engine's contract (tokens emitted, the bonus included).
    """
    if spec_metadata.is_ragged_verify:
        raise NotImplementedError("DFlash2 sampled verification needs a uniform verify width")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("DFlash2 sampled verification needs the gathered target logits")
    batch_size = attn_metadata.num_seqs
    num_contexts = attn_metadata.num_contexts
    num_gens = batch_size - num_contexts
    device = logits.device
    draft_tokens = worker._reshape_draft_tokens_for_accept(spec_metadata, num_gens, device)
    logits = worker._reshape_logits_for_accept(logits, num_contexts, num_gens, spec_metadata)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    logits = worker._apply_occurrence_penalties(
        logits, draft_tokens, num_contexts, batch_size, spec_metadata
    )
    K = draft_tokens.shape[1]
    store = dflash2_proposal_store(worker, spec_metadata)
    slot_ids = spec_metadata.batch_slot_ids

    fused = _fused_commit_applies(worker, num_contexts, draft_tokens, spec_metadata)
    if not fused:
        accepted_tokens = torch.zeros((batch_size, K + 1), dtype=torch.int32, device=device)
        num_accepted_tokens = torch.ones(batch_size, dtype=torch.int32, device=device)
    if num_contexts > 0:
        accepted_tokens[:num_contexts, 0] = worker._sample_tokens_for_batch(
            logits[:num_contexts], spec_metadata, num_contexts, num_contexts
        ).to(torch.int32)
        store.invalidate(slot_ids[:num_contexts])
    if num_gens > 0:
        start, end = num_contexts, num_contexts + num_gens * (K + 1)
        gen_logits = _zero_padding_rows(
            worker, logits[start:end], spec_metadata, num_contexts, batch_size, K + 1
        )
        vocab = gen_logits.shape[-1]

        def target_probs_of_rows():
            return worker._compute_probs_from_logits(
                spec_metadata.advanced_sampling_mode,
                gen_logits,
                spec_metadata.temperatures[start:end],
                spec_metadata.top_ks[start:end],
                spec_metadata.top_ps[start:end],
                spec_metadata.min_ps[start:end],
            ).view(num_gens, K + 1, vocab)

        def verify_inputs():
            """The stored proposals of the drafts, the acceptance seeds and (fused) the greedy verification: inputs
            of the block verification and the commit that do not read the target probabilities."""
            if store.candidate_ids.shape[-1] == PROPOSAL_TOP_K:
                candidate_ids, q = k31_dflash2_proposal_load(
                    store.candidate_ids,
                    store.q,
                    store.tokens,
                    slot_ids[num_contexts:batch_size],
                    draft_tokens,
                )
            else:
                candidate_ids, q = store.load(slot_ids[num_contexts:batch_size], draft_tokens)
            seed, offset = worker._rng_state_per_request(
                spec_metadata, num_contexts, batch_size, slot=spec_metadata.rng_accept_slot
            )
            greedy = (
                greedy_verify_op(gen_logits.view(num_gens, K + 1, vocab), draft_tokens)
                if fused
                else (None, None)
            )
            return candidate_ids, q, seed, offset, *greedy

        overlap = _verify_overlap(worker) if fused else None
        if overlap is None:
            target_probs = target_probs_of_rows()
            candidate_ids, q, seed, offset, greedy_tau, greedy_bonus = verify_inputs()
        else:
            # The verify inputs on the side stream beside the target probabilities (both read the logits only);
            # the block verification waits for both. The store rows are rewritten only after the join (the
            # selector walk of this step's draft).
            current = torch.cuda.current_stream()
            overlap.fork.record(current)
            with torch.cuda.stream(overlap.stream):
                overlap.fork.wait(overlap.stream)
                side = verify_inputs()
                overlap.join.record(overlap.stream)
            target_probs = target_probs_of_rows()
            overlap.join.wait(current)
            if not torch.cuda.is_current_stream_capturing():
                # Allocated in side-stream order and read on this stream.
                for t in side:
                    t.record_stream(current)
            candidate_ids, q, seed, offset, greedy_tau, greedy_bonus = side
        # Block verification and the bonus draw in one CuTe DSL launch, its acceptance uniforms
        # drawn in the kernel from (seed, offset) as dflash2_uniforms draws them.
        tau, bonus = block_verify_op(
            target_probs,
            draft_tokens,
            candidate_ids,
            q,
            seed,
            offset,
        )
        temperatures = spec_metadata.request_temperatures[num_contexts:batch_size]
        if fused:
            # The picks by temperature, the emitted tokens, the counts and the forced count in one launch.
            mode, base_total, max_total, frac, counter, pool = _force_args(
                worker, K, spec_metadata, device
            )
            if counter is None:
                # Not read by the kernel unless it draws from the pool.
                counter = torch.empty(1, dtype=torch.int64, device=device)
                pool = torch.empty(1, dtype=torch.float32, device=device)
            accepted_tokens, num_accepted_tokens = k31_dflash2_accept_commit(
                tau,
                bonus,
                greedy_tau,
                greedy_bonus,
                temperatures,
                GREEDY_TEMPERATURE_THRESHOLD,
                draft_tokens,
                mode,
                base_total,
                max_total,
                frac,
                counter,
                pool,
                FORCE_COUNTER_STRIDE,
                FORCE_SLOT_STRIDE,
            )
            return worker._commit_occurrence_counts(
                agree_verified_tokens(worker, accepted_tokens, num_accepted_tokens),
                batch_size,
                spec_metadata,
            )
        greedy = greedy_request_mask(temperatures)
        greedy_tau, greedy_bonus = greedy_verify_op(
            gen_logits.view(num_gens, K + 1, vocab), draft_tokens
        )
        tau = torch.where(greedy, greedy_tau, tau)
        bonus = torch.where(greedy, greedy_bonus, bonus)
        # Drafts before tau, the bonus at tau; positions past it keep drafts (never emitted
        # unless a forced acceptance count reads them).
        tokens = torch.cat((draft_tokens.long(), draft_tokens[:, -1:].long()), dim=1)
        tokens = tokens.scatter(1, tau.unsqueeze(1), bonus.unsqueeze(1))
        accepted_tokens[num_contexts:] = tokens.to(torch.int32)
        num_accepted_tokens[num_contexts:] = (tau + 1).to(torch.int32)
    num_accepted_tokens = worker._apply_force_accepted_tokens(
        num_accepted_tokens, num_contexts, K, spec_metadata=spec_metadata
    )
    return worker._commit_occurrence_counts(
        agree_verified_tokens(worker, accepted_tokens, num_accepted_tokens),
        batch_size,
        spec_metadata,
    )
