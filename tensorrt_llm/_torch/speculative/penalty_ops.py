# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Occurrence penalties (repetition / presence / frequency) for the one-model
speculative decoding path.

The two entry points mirror the two moments the penalty state moves:

* :func:`apply_penalties` rewrites the target logits before sampling, penalizing
  every speculative position against its own prefix.
* :func:`update_penalty_counts` commits the tokens a step actually accepted.

The penalty formulas match ``TorchSampler``'s
``Fusions._apply_occurrence_penalties_impl`` (the two-model path) and, behind it,
the C++ ``batchApplyPenalty`` kernel, so a request decodes the same way whichever
speculation mode serves it.

Everything here stays CUDA-graph friendly: no host syncs, no data-dependent
shapes, and all state lives in buffers ``SpecMetadata.prepare_penalty_buffers``
allocated up front.
"""

import torch


def _unpack_prompt_mask(
    prompt_mask: torch.Tensor, row_slots: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """Expand the packed prompt bitmask to a ``bool[T, vocab_size]`` view.

    ``prompt_mask`` stores one bit per token id (32 ids per int32 word), which is
    what makes it 32x smaller than a count tensor. Only presence is representable,
    which is all repetition needs.
    """
    words = prompt_mask[row_slots]  # [T, ceil(vocab/32)]
    bits = torch.arange(32, device=prompt_mask.device, dtype=torch.int32)
    unpacked = (words.unsqueeze(-1) >> bits) & 1  # [T, words, 32]
    return unpacked.reshape(words.size(0), -1)[:, :vocab_size].to(torch.bool)


@torch.compile(dynamic=None, fullgraph=True)
def _apply_penalties_impl(
    logits: torch.Tensor,
    counts: torch.Tensor,
    prompt_mask: torch.Tensor,
    row_slots: torch.Tensor,
    intra_step_tokens: torch.Tensor,
    intra_step_valid: torch.Tensor,
    repetition: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
    active: torch.Tensor,
) -> None:
    # Fold this step's earlier speculative positions into the counts each row sees.
    # Position k must be penalized against positions 0..k-1 of the SAME step, which
    # are not in ``counts`` yet (they are only committed once acceptance is known).
    # ``intra_step_tokens`` is [T, T] lower-triangular-masked: entry [r, j] is the
    # token position j contributed to row r, or -1 when it must not count. Only
    # strictly-earlier positions contribute; a position never penalizes itself.
    #
    # One-hot accumulate rather than scatter_add_, because rows sharing a slot would
    # otherwise race on the same counts entry -- and a captured graph cannot rely on
    # scatter ordering.
    safe_tokens = torch.where(
        intra_step_tokens >= 0, intra_step_tokens, intra_step_tokens.new_zeros(())
    )
    intra_counts = torch.zeros_like(logits, dtype=torch.int32)
    intra_counts.scatter_add_(1, safe_tokens.to(torch.int64), intra_step_valid.to(torch.int32))

    # Output-side occurrences: drive all three penalties.
    count = counts[row_slots] + intra_counts

    rep = repetition[row_slots].unsqueeze(1)
    pre = presence[row_slots].unsqueeze(1)
    freq = frequency[row_slots].unsqueeze(1)

    output_seen = count > 0
    # Prompt-side occurrences drive repetition ONLY -- a token the user wrote should
    # not be charged presence/frequency, which describe what the model itself emitted.
    # Storing them as a bitmask costs 1 bit instead of the 32 an int32 count would.
    seen_for_repetition = output_seen | _unpack_prompt_mask(prompt_mask, row_slots, logits.size(-1))

    penalized = logits.float()
    # Repetition is multiplicative and splits on sign: dividing a negative logit
    # would raise it. Presence/frequency are plain subtractions.
    repeated = torch.where(penalized < 0, penalized * rep, penalized / rep)
    penalized = torch.where(seen_for_repetition, repeated, penalized)
    penalized = penalized - torch.where(
        output_seen, pre + freq * count.to(torch.float32), penalized.new_zeros(())
    )

    limit = torch.finfo(logits.dtype).max
    penalized = penalized.clamp(-limit, limit).to(logits.dtype)
    # Cast before the select so untouched rows stay bit-identical.
    row_active = active[row_slots].unsqueeze(1)
    logits.copy_(torch.where(row_active, penalized, logits))


def apply_penalties(
    logits: torch.Tensor,
    spec_metadata,
    row_slots: torch.Tensor,
    intra_step_tokens: torch.Tensor,
    intra_step_valid: torch.Tensor,
    any_active: bool = True,
) -> None:
    """Apply the occurrence penalties to ``logits`` in place, before sampling.

    Args:
        logits: ``[T, vocab_size]`` target logits. Modified in place.
        spec_metadata: carries the buffers from ``prepare_penalty_buffers``.
        row_slots: ``int64[T]`` slot row owning each logits row. Rows belonging to
            no live request must point at ``penalty_dummy_row``, whose parameters
            keep their no-op defaults.
        intra_step_tokens: ``int64[T, T]`` token each earlier position of the same
            step contributes to this row, ``-1`` where it does not apply.
        intra_step_valid: ``bool[T, T]`` mask matching ``intra_step_tokens``.
        any_active: whether any request in this batch actually has a penalty. False
            skips the launch entirely -- the whole pass is a vocab-sized read/write
            that would otherwise be paid to compute a no-op. The caller knows this
            from host state (``SpecMetadata.batch_uses_penalty``), so testing it
            here costs no device sync.

    No-op unless the penalty buffers exist (i.e. ``enable_penalty``).
    """
    if not getattr(spec_metadata, "enable_penalty", False) or not any_active:
        return
    counts = spec_metadata.penalty_counts
    if counts is None or logits.numel() == 0:
        return
    # A vocab-sharded logits view cannot be penalized against full-vocab counts.
    if logits.size(-1) != counts.size(-1):
        return

    _apply_penalties_impl(
        logits,
        counts,
        spec_metadata.penalty_prompt_mask,
        row_slots,
        intra_step_tokens,
        intra_step_valid,
        spec_metadata.penalty_repetition,
        spec_metadata.penalty_presence,
        spec_metadata.penalty_frequency,
        spec_metadata.penalty_active,
    )


def build_row_mapping(
    spec_metadata,
    num_contexts: int,
    batch_size: int,
    draft_len: int,
    draft_tokens: torch.Tensor,
    device: torch.device,
):
    """Derive the per-row slot map and intra-step prefix for a packed logits batch.

    Assumes the layout every linear one-model mode produces once
    ``_reshape_logits_for_accept`` has run: ``[ctx (1 row each), gen (draft_len + 1
    rows each)]``, request-major. PARD emits ``2 * draft_len`` rows per gen request
    but that hook already narrows them to ``draft_len + 1``, so both share this code.

    Tree modes do NOT fit: their rows are tree nodes, so a row's prefix is its
    root path rather than the rows before it. They are rejected at admission.

    Returns ``(row_slots, intra_tokens, intra_valid)`` shaped for
    :func:`apply_penalties`, or ``None`` when there is nothing to penalize.
    """
    slot_ids = spec_metadata.batch_slot_ids
    if slot_ids is None:
        return None

    num_gens = batch_size - num_contexts
    rows_per_gen = draft_len + 1
    total_rows = num_contexts + num_gens * rows_per_gen

    batch_slots = slot_ids[:batch_size].to(torch.int64)
    # Context requests own one row; gen requests own rows_per_gen consecutive rows.
    row_slots = (
        torch.cat(
            [
                batch_slots[:num_contexts],
                batch_slots[num_contexts:batch_size].repeat_interleave(rows_per_gen),
            ]
        )
        if num_gens > 0
        else batch_slots[:num_contexts]
    )

    # Intra-step prefix: within one gen request, row k must see the draft tokens at
    # positions 0..k-1 -- they are this step's earlier speculative positions, not yet
    # committed to the counts. Strictly earlier: a position never penalizes itself.
    intra_tokens = torch.full((total_rows, total_rows), -1, dtype=torch.int64, device=device)
    intra_valid = torch.zeros((total_rows, total_rows), dtype=torch.bool, device=device)
    if num_gens > 0 and draft_len > 0 and draft_tokens.numel() > 0:
        drafts = draft_tokens.reshape(num_gens, -1)[:, :draft_len].to(torch.int64)
        # Build the block-diagonal placement without a Python loop, so the shapes stay
        # static and the whole thing is capturable.
        gen_rows = torch.arange(num_gens * rows_per_gen, device=device)
        g_of_row = gen_rows // rows_per_gen  # which request owns the row
        k_of_row = gen_rows % rows_per_gen  # the row's speculative position
        # Each gen row writes into its own request's column block, so requests can
        # never read each other's tokens.
        col_base = num_contexts + g_of_row * rows_per_gen
        cols = col_base.unsqueeze(1) + torch.arange(draft_len, device=device)
        rows = (num_contexts + gen_rows).unsqueeze(1).expand(-1, draft_len)
        # Column j counts for row k iff j < k: strictly earlier positions only.
        valid = torch.arange(draft_len, device=device).unsqueeze(0) < k_of_row.unsqueeze(1)
        vals = drafts[g_of_row]
        intra_tokens[rows.reshape(-1), cols.reshape(-1)] = vals.reshape(-1)
        intra_valid[rows.reshape(-1), cols.reshape(-1)] = valid.reshape(-1)
    return row_slots, intra_tokens, intra_valid


def seed_prompt_mask(
    spec_metadata,
    slot: int,
    prompt_tokens: torch.Tensor,
) -> None:
    """Record a request's prompt tokens in its packed bitmask row.

    Called once per request, when its sequence starts. Only the fact that a token
    occurred is stored, which is all repetition consumes; presence/frequency
    deliberately ignore the prompt (see ``_apply_penalties_impl``).

    ``prompt_tokens`` may legitimately contain ids outside ``[0, vocab_size)`` --
    multimodal models place placeholders above the vocab -- so they are dropped here
    rather than allowed to index the mask.
    """
    if not getattr(spec_metadata, "enable_penalty", False):
        return
    prompt_mask = spec_metadata.penalty_prompt_mask
    if prompt_mask is None or prompt_tokens.numel() == 0:
        return
    if slot >= prompt_mask.size(0):
        return

    vocab_size = spec_metadata.vocab_size
    tokens = prompt_tokens.to(device=prompt_mask.device, dtype=torch.int64)
    tokens = tokens[(tokens >= 0) & (tokens < vocab_size)]
    if tokens.numel() == 0:
        return
    # Deduplicate first: two ids in the same 32-id word must contribute both bits,
    # so the reduction has to be a bitwise OR. scatter_reduce_ has no OR mode, and
    # 'amax' would keep only the larger bit. Unique ids let a plain OR-accumulate
    # over one-hot bits work instead.
    tokens = torch.unique(tokens)
    words = tokens // 32
    bits = torch.ones_like(tokens, dtype=torch.int32) << (tokens % 32).to(torch.int32)
    row = torch.zeros_like(prompt_mask[slot])
    # Distinct ids in the same word contribute disjoint bits, so summing them is
    # exactly their bitwise OR.
    row.scatter_add_(0, words, bits)
    prompt_mask[slot] = prompt_mask[slot].bitwise_or(row)


@torch.compile(dynamic=None, fullgraph=True)
def _update_penalty_counts_impl(
    counts: torch.Tensor,
    slot_rows: torch.Tensor,
    tokens: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    vocab = counts.size(-1)
    safe_tokens = torch.where(valid, tokens, tokens.new_zeros(()))
    flat = slot_rows.unsqueeze(1) * vocab + safe_tokens
    counts.view(-1).scatter_add_(0, flat.reshape(-1), valid.to(counts.dtype).reshape(-1))


def update_penalty_counts(
    spec_metadata,
    slot_rows: torch.Tensor,
    accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Commit the tokens this step accepted into the persistent counts.

    Only accepted tokens are counted: a rejected speculative token was never part
    of the sequence and must not penalize anything.

    Args:
        spec_metadata: carries the buffers from ``prepare_penalty_buffers``.
        slot_rows: ``int64[B]`` slot row per request; dummy/padding requests must
            point at ``penalty_dummy_row``.
        accepted_tokens: ``int32[B, max_accepted]`` accepted token ids.
        num_accepted_tokens: ``int32[B]`` how many of each row's entries are real.

    No-op unless the penalty buffers exist (i.e. ``enable_penalty``).
    """
    if not getattr(spec_metadata, "enable_penalty", False):
        return
    counts = spec_metadata.penalty_counts
    if counts is None or accepted_tokens.numel() == 0:
        return

    vocab = counts.size(-1)
    positions = torch.arange(accepted_tokens.size(1), device=accepted_tokens.device)
    tokens = accepted_tokens.to(torch.int64)
    # Drop padding columns past num_accepted, and ids outside the vocab
    # (multimodal placeholders and the untouched tail of the buffer).
    valid = (
        (positions.unsqueeze(0) < num_accepted_tokens.unsqueeze(1).to(positions.dtype))
        & (tokens >= 0)
        & (tokens < vocab)
    )
    # Dummy rows keep active=False, so their counts never reach a real request,
    # but they still must not corrupt a shared row: they are masked out here too.
    active = spec_metadata.penalty_active
    if active is not None:
        valid = valid & active[slot_rows].unsqueeze(1)

    _update_penalty_counts_impl(counts, slot_rows, tokens, valid)
