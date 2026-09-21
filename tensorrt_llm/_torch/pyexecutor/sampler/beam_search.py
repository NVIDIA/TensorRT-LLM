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

"""Beam search for ``TorchSampler``.

The candidate-selection, beam-expansion and candidate-beams-array (CBA) logic:
pure tensor functions plus the metadata dataclasses they consume
(:class:`BeamSearchStore`, :class:`BeamSearchMetadata`, :class:`CBAState`,
:class:`BeamHistory`).

:class:`BeamSearchHandler` sits on top of those: it owns the store and the
host-side state the CBA path needs, builds the per-step
:class:`BeamSearchMetadata`, and is driven by ``TorchSampler``, which holds one
instance.

Unlike the per-step log-prob path, a beam's tokens and log-probs are only
final once the winning path is known, so they are accumulated in
:class:`BeamHistory` and emitted in one go by :func:`finalize_beam`.
"""

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, NamedTuple, Optional, TypeAlias, cast

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.executor.result import Logprob

from ..llm_request import LlmRequest, LlmRequestState
from .logprobs import LogProbsStore
from .ops.flashinfer import radix_topk_op
from .ops.vanilla import StrategyMetadata
from .sampler_common import _get_beam_width_in, int_tensor
from .sampler_features import _SideStreamCopier

BEAM_SEARCH_PAD_TOKEN = -1


class BeamSearchEarlyStop(IntEnum):
    """Beam-search stopping mode, mirroring HuggingFace's tri-state
    ``early_stopping`` (``True`` / ``False`` / ``"never"``).

    An ``IntEnum`` so it stays interchangeable with the raw integers coming from
    ``sampling_config.early_stopping`` and used in the strategy grouping key.
    Member names follow HF's ``early_stopping`` values (``True`` / ``False`` /
    ``"never"``) directly.

    The stopping decision differs only in the upper bound used for the best
    score an unfinished beam could still attain (score is
    ``cum_log_prob / gen_length ** length_penalty``, and ``cum_log_prob <= 0``):
    """

    TRUE = 1
    """HF ``True`` (default): stop once ``beam_width`` finished candidates exist."""

    FALSE = 0
    """HF ``False``: CBA path bounding attainability by the beam's current
    score."""

    NEVER = 2
    """HF ``"never"``: CBA path bounding attainability by ``max_seq_len`` when
    ``length_penalty > 0``. The CBA path treats any value other than ``0`` /
    ``1`` as this mode."""

    @classmethod
    def from_raw(cls, value: Optional[int]) -> "BeamSearchEarlyStop":
        """Map a raw ``sampling_config.early_stopping`` value to a mode.

        ``None`` -> ``TRUE`` (the default); ``0`` -> ``FALSE``; every other
        integer -> ``NEVER`` (HF's "never"), matching the CBA path which
        special-cases only ``FALSE``.

        This is deliberately more permissive than the OpenAI-compatible
        server, whose ``bool | Literal["never"]`` field rejects anything else
        outright. ``sampling_config.early_stopping`` is a plain int that
        predates the tri-state and reaches here from the C++ runtime as well,
        so values outside {0, 1, 2} are folded into the nearest defined mode
        rather than raising on a path that used to accept them. Tightening it
        would be an API break for existing callers; the HTTP layer is free to
        be strict because it is a newer surface."""
        if value is None:
            return cls.TRUE
        if value == cls.FALSE:
            return cls.FALSE
        if value == cls.TRUE:
            return cls.TRUE
        return cls.NEVER


def _beam_topk(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorted top-k over the last dim of a 2D tensor, dispatched on row width.

    flashinfer's radix-select kernel is O(n) and much faster than torch.topk on
    large rows (e.g. vocab-sized logits), while torch.topk wins on small rows
    where the radix kernel's fixed per-call cost dominates. The 10k crossover
    follows flashinfer's own guidance (``flashinfer.top_k`` docstring).

    Beam search is not a flashinfer-gated feature, so fall back to torch.topk
    when flashinfer is missing: real vocabularies are always past the crossover,
    which would otherwise make the whole default beam path require it. The two
    kernels agree on values and on descending order; only the index order among
    equal values may differ, which torch.topk leaves unspecified anyway.
    """
    if IS_FLASHINFER_AVAILABLE and values.size(-1) > 10000:
        return radix_topk_op(values, k)
    return torch.topk(values, k=k, dim=-1, sorted=True)


@dataclass(kw_only=True)
class _CBAFields:
    """The candidate-beams-array (CBA) tensors, shared by the persistent
    ``BeamSearchStore`` and the per-step ``CBAState`` view (which reference the
    same tensors). Field shapes and semantics are documented here once; both
    subclasses inherit these fields.

    Maintained for every beam-search request; allocated lazily by
    ``BeamSearchStore.ensure_cba`` on the first such request.

    Fields written on every beam-search step regardless of stopping mode
    (``original_tokens``, ``prompt_lens``, ``batch_dones``) live on
    ``BeamSearchStore`` instead, since they must exist before any CBA
    request arrives.
    """

    cba_tokens: torch.Tensor
    """[max_num_sequences, max_beam_width, max_seq_len] int32, finished-beam
    path snapshots (generated tokens, BEAM_SEARCH_PAD_TOKEN padded)."""
    cba_cum_log_probs: torch.Tensor
    """[max_num_sequences, max_beam_width] float32, raw cumulative log-probs."""
    cba_normed_scores: torch.Tensor
    """[max_num_sequences, max_beam_width] float32, length-normalized scores
    (-inf: empty entry)."""
    cba_lengths: torch.Tensor
    """[max_num_sequences, max_beam_width] int32, generated lengths."""
    cba_caps: torch.Tensor
    """[max_num_sequences] int32, per-slot CBA capacity (the request's
    maximum beam width). Differs from the step's beam_width_out for
    variable-beam-width requests."""
    original_log_probs: torch.Tensor
    """[max_num_sequences, max_beam_width, max_seq_len] float32, uncorrected
    per-slot sampled log-prob per step (log-prob analog of original_tokens).
    Written and read by the CBA path for logprobs."""
    cba_log_probs: torch.Tensor
    """[max_num_sequences, max_beam_width, max_seq_len] float32, per-token
    log-probs of the CBA path snapshots."""


@dataclass(kw_only=True)
class CBAState(_CBAFields):
    """Candidate-Beams-Array (CBA) state, present for every beam-search
    request whatever its early_stopping mode; see
    beam_search_sampling_batch_cba.

    A per-step view over the persistent ``BeamSearchStore`` (shared CBA tensors
    inherited from ``_CBAFields``) plus the step-local fields below. Bundling
    them lets callers gate on a single ``cba is not None`` check (the enclosing
    ``BeamSearchMetadata.cba`` is None outside CBA mode) instead of testing each
    field.
    """

    end_ids: torch.Tensor
    """[max_num_sequences] int32, per-slot end token id (< 0: no end token)."""
    prompt_lens: torch.Tensor
    """[max_num_sequences] int32, per-slot prompt length (from BeamSearchStore)."""
    original_tokens: torch.Tensor
    """[max_num_sequences, max_beam_width, max_seq_len] int32, uncorrected
    per-slot tokens (from BeamSearchStore)."""
    batch_dones: torch.Tensor
    """[max_num_sequences] bool, per-slot termination verdict (from
    BeamSearchStore)."""
    max_seq_len: int = 0
    """Maximum sequence length (prompt + generated), used by the
    best-attainable-score bound of the "never" early-stopping modes."""
    max_gen_len: int = 0
    """Host-known upper bound on the generated length (including this
    step's token) across the group's requests. Bounds the width of the CBA
    path-snapshot and pool-merge operations, which would otherwise run at
    the full max_seq_len width every step. 0 means unknown (full width)."""


@dataclass(kw_only=True)
class BeamSearchStore:
    """Persistent per-sampler beam-search storage.

    The candidate-beams-array tensors live in the optional ``cba`` member,
    allocated by :meth:`ensure_cba` on the first beam-search request. Every
    early_stopping mode runs on that path, so the only sampler that never
    allocates them is one that never sees a beam-search request.
    """

    cache_indirection: torch.Tensor
    """[max_num_sequences, max_beam_width, attention_size] int32, cache
    indirection for beam search sampling."""
    cache_indirection_buffer: torch.Tensor
    """[max_num_sequences, max_beam_width, attention_size] int32, second buffer
    used to update the cache indirection during sampling."""
    cum_log_probs: torch.Tensor
    """[max_num_sequences, max_beam_width] float32, current cumulative logprob
    of each active beam."""
    first_finish_reasons: torch.Tensor
    """[max_num_sequences, max_beam_width] int32, first finish reason per beam."""
    pending_harvest: torch.Tensor
    """[max_num_sequences, max_beam_width] bool, beams latched finished by the
    finish handler and not yet harvested into the CBA.

    Kept apart from ``first_finish_reasons`` on purpose. That tensor is the
    request's reported finish reason and must survive for the whole request,
    while this one is a one-shot signal: the harvest consumes it, and the slot
    it frees goes on to hold an unrelated, unfinished continuation. Reading the
    persistent reason as the latch would harvest that continuation again on the
    next step."""
    predecessor_beams: torch.Tensor
    """[max_num_sequences, max_beam_width] int32, predecessor beam per beam, used
    for stop word detection."""
    beam_idx_arange: torch.Tensor
    """[max_beam_width] int32, cached ``arange(max_beam_width)`` used as the
    scatter source in the per-step ``cache_indirection.scatter_``."""
    original_tokens: torch.Tensor
    """[max_num_sequences, max_beam_width, max_seq_len] int32, uncorrected
    per-slot tokens, written every beam-search step; read with
    ``cache_indirection`` to snapshot finished paths into the CBA, and seeded
    by the disaggregated first-generation handoff."""
    prompt_lens: torch.Tensor
    """[max_num_sequences] int32, per-slot prompt length; used to derive
    generated lengths and snapshot positions."""
    batch_dones: torch.Tensor
    """[max_num_sequences] bool, per-slot beam-search termination verdict."""
    cba: Optional[_CBAFields] = None
    """CBA tensors; None until the first beam-search request."""

    @classmethod
    def create(
        cls,
        *,
        cache_indirection_shape: tuple[int, ...],
        max_num_sequences: int,
        max_beam_width: int,
    ) -> "BeamSearchStore":
        """Allocate the per-sampler beam-search buffers.

        ``cache_indirection_shape`` is [max_num_sequences, max_beam_width,
        attention_size]; the per-beam scalar buffers use its leading two dims.
        """
        per_beam = cache_indirection_shape[:-1]
        return cls(
            cache_indirection=torch.empty(cache_indirection_shape, device="cuda", dtype=torch.int),
            cache_indirection_buffer=int_tensor(cache_indirection_shape),
            cum_log_probs=torch.empty(per_beam, device="cuda", dtype=torch.float32),
            predecessor_beams=int_tensor(per_beam),
            original_tokens=int_tensor(cache_indirection_shape),
            first_finish_reasons=int_tensor(per_beam),
            pending_harvest=torch.zeros(per_beam, device="cuda", dtype=torch.bool),
            beam_idx_arange=torch.arange(max_beam_width, device="cuda", dtype=torch.int32),
            prompt_lens=int_tensor((max_num_sequences,)),
            batch_dones=torch.zeros((max_num_sequences,), device="cuda", dtype=torch.bool),
        )

    def ensure_cba(self) -> _CBAFields:
        """Allocate the CBA tensors on first use and return them.

        Called when the first beam-search request is admitted, whatever its
        early_stopping mode; idempotent afterwards.
        """
        if self.cba is None:
            shape = self.original_tokens.shape
            per_beam = shape[:-1]
            num_sequences = shape[0]
            self.cba = _CBAFields(
                cba_tokens=int_tensor(tuple(shape)),
                cba_cum_log_probs=torch.zeros(per_beam, device="cuda", dtype=torch.float32),
                cba_normed_scores=torch.full(
                    per_beam, float("-inf"), device="cuda", dtype=torch.float32
                ),
                cba_lengths=int_tensor(per_beam),
                cba_caps=int_tensor((num_sequences,)),
                original_log_probs=torch.zeros(tuple(shape), device="cuda", dtype=torch.float32),
                cba_log_probs=torch.zeros(tuple(shape), device="cuda", dtype=torch.float32),
            )
        return self.cba


@dataclass(kw_only=True)
class BeamHistory:
    """Per-beam corrected tokens and log-probs. The three log-prob fields are
    None unless log-probs are requested."""

    tokens: torch.Tensor
    """[num_beams, seq_len] int, corrected token ids per beam."""
    logprobs: torch.Tensor | None = None
    """[num_beams, seq_len] float, per-token sampled log-prob."""
    logprobs_indices: torch.Tensor | None = None
    """[num_beams, seq_len] int, vocab indices of the sampled log-probs."""
    cum_logprobs: torch.Tensor | None = None
    """[num_beams] float, cumulative log-prob per beam."""


def _gather_beam_path(
    *, current_path: torch.Tensor, cache_indirection: torch.Tensor
) -> torch.Tensor:
    """Gather the correct tokens for each beam from current_path."""
    new_path = torch.zeros_like(current_path)
    torch.gather(input=current_path, dim=0, index=cache_indirection, out=new_path)
    return new_path


@dataclass(kw_only=True)
class BeamSearchMetadata(StrategyMetadata):
    """Stateful tensors required by beam_search_sampling_batch_cba."""

    cache_indirection: torch.Tensor
    cache_indirection_buffer: torch.Tensor
    cum_log_probs: torch.Tensor
    new_log_probs: torch.Tensor
    seq_slots: torch.Tensor
    seq_lens: torch.Tensor
    finished_beams: torch.Tensor
    pending_harvest: torch.Tensor
    predecessor_beams: torch.Tensor
    beam_idx_arange: torch.Tensor
    stop_past_tokens: Optional[torch.Tensor] = None
    """[max_stop_word_length, max_num_sequences, max_beam_width] int32, the
    finish handler's rolling stop-word window (FinishReasonsHandler store).
    The beam axis is reordered by the
    step's predecessor beams so the handler's stop-word matching stays correct
    across beam swaps. When None (tests without stop words), the reorder is
    skipped — multi-token stop-word matching would then be unreliable across
    beam swaps."""
    cba: Optional[CBAState] = None
    """Candidate-Beams-Array state, present for every beam-search request
    regardless of its early_stopping mode; None until the first one."""


def _update_cache_indirection_buffer(
    cache_indirection_input: torch.Tensor,
    cache_indirection_output: torch.Tensor,
    seq_slots: torch.Tensor,
) -> None:
    assert cache_indirection_input.device == cache_indirection_output.device
    cache_indirection_input.index_copy_(0, seq_slots, cache_indirection_output[seq_slots])


def _beam_step_preprocess(
    logits: torch.Tensor,
    *,
    beam_width_in: int,
    row_stride: int | None = None,
    temperature: float | None,
    return_probs: bool,
    args: "BeamSearchMetadata",
) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """Shared front-end of both beam-search step ops.

    Applies temperature, snapshots the cache indirection into its buffer, and
    returns ``(logprobs, softmax, batch_size)``. ``softmax`` is None when
    ``return_probs`` is False.

    Eager only: it mutates ``args.cache_indirection_buffer`` in place.

    ``row_stride`` is how many rows the forward path allocated per request,
    which is the static admission width. It differs from ``beam_width_in``
    under Variable-Beam-Width-Search, where only the first ``beam_width_in``
    rows of each request hold live beams and the rest are padding. Reshaping
    by ``beam_width_in`` alone would silently mix rows across requests, since
    ``view`` accepts any shape whose element count divides. Defaults to
    ``beam_width_in`` for callers whose layout already matches.
    """
    assert logits.dim() == 2, "logits should be 2D: [batch_size * beam_width, vocab_size]"
    num_rows, vocab_size = logits.size()
    if row_stride is None:
        row_stride = beam_width_in
    assert row_stride >= beam_width_in, (
        f"row_stride ({row_stride}) must cover beam_width_in ({beam_width_in})"
    )
    batch_size = num_rows // row_stride

    logits = logits.view(batch_size, row_stride, vocab_size)
    if row_stride != beam_width_in:
        # Drop the padding rows; the live beams are the leading ones.
        logits = logits[:, :beam_width_in, :]
    if temperature is not None and temperature != 0:
        logits = logits / max(temperature, 1e-5)
    softmax: Optional[torch.Tensor] = None
    if return_probs:
        softmax = torch.softmax(logits, dim=-1)
    _update_cache_indirection_buffer(
        args.cache_indirection_buffer, args.cache_indirection, args.seq_slots
    )
    assert batch_size == args.seq_slots.size(0)

    return torch.log_softmax(logits, dim=-1), softmax, batch_size


def _pad_next_tokens(next_tokens: torch.Tensor, store_width: int) -> torch.Tensor:
    """Pad a [batch, beam_width_out] token tensor to the store's beam width.

    The batched sampling buffers are allocated at the maximum beam width; on
    variable-beam-width steps the op produces fewer columns and the rest are
    filled with BEAM_SEARCH_PAD_TOKEN.

    Consumers must read only the leading ``beam_width_out`` columns. Appending
    the padded ones puts the sentinel into the request's token history, which
    finalization does later overwrite from the corrected paths -- but the
    padded history is visible in the meantime to streaming consumers and to
    anything reading ``get_tokens()`` mid-flight. See the
    ``_get_beam_width_out`` bound in ``TorchSampler.update_requests``.
    """
    if next_tokens.size(1) >= store_width:
        return next_tokens
    return torch.nn.functional.pad(
        next_tokens, (0, store_width - next_tokens.size(1)), value=BEAM_SEARCH_PAD_TOKEN
    )


def beam_candidate_topk(
    logprobs: torch.Tensor,
    *,
    beam_width_out: int,
    length_penalty: "torch.Tensor | float | None" = None,
    cand_gen_lengths: Optional[torch.Tensor] = None,
    diversity_rate: "torch.Tensor | float | None" = None,
    source_beam_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-stage top-k over beam-expansion candidates with per-source-beam
    ranking adjustments, ranked by::

        (cum_log_prob + diversity_rate * source_beam_index)
            / gen_length**length_penalty

    The diversity term ``diversity_rate * source_beam_index`` spreads selection
    across source beams (candidates from lower-ranked beams get a boost), and
    the length term normalizes by ``gen_length**length_penalty``. Finished
    beams are harvested into the candidate pool, but until the harvest lands
    a finished beam's (single) candidate receives the same ``rate * slot_index``
    boost as any other — slot position thus slightly affects how hard a
    finished beam is to evict.

    Mathematically equivalent to adjusting the full [batch, bw_in, vocab]
    candidate matrix and taking a flat top-k, but avoids touching the vocab
    axis: both adjustments are constant per source beam, so they cannot
    change the ordering *within* a beam. Any global winner is therefore
    among its own beam's top-``beam_width_out`` raw candidates. Stage 1
    takes a per-beam top-k on raw scores; stage 2 adjusts only the
    ``bw_in * bw_out`` survivors and selects the global top-k.

    Args:
        logprobs: [batch, beam_width_in, vocab] raw cumulative log-probs.
        length_penalty: scalar, or per-request tensor of shape [batch].
            None/0 disables length normalization.
        cand_gen_lengths: [batch, beam_width_in] candidate generated lengths.
            Required iff ``length_penalty`` is active.
        diversity_rate: scalar, or per-request tensor of shape [batch].
            None/0 disables the diversity adjustment.
        source_beam_indices: optional cached ``arange(beam_width_in)`` on the
            logprobs device (e.g. ``BeamSearchStore.beam_idx_arange``), used
            by the diversity adjustment; computed on the fly if omitted.

    Returns:
        (sorted_logprobs, predecessor_beams, tokens): the raw (unadjusted)
        cumulative log-probs of the selected candidates, the source-beam index
        each candidate expands from, and its token id — all of shape
        [batch, beam_width_out] (indices int32), ordered by descending
        adjusted score.
    """
    batch_size, beam_width_in, vocab_size = logprobs.shape
    # Clamps are only relevant for tiny test vocabularies: stage 1 cannot
    # exceed the vocab, the global top-k cannot exceed the pooled candidates.
    stage1_k = min(beam_width_out, vocab_size)
    beam_width_out = min(beam_width_out, beam_width_in * stage1_k)
    # Stage 1: raw per-beam top-k (the per-beam adjustments are constant
    # along the vocab axis, so raw ordering == adjusted ordering).
    per_beam_vals, per_beam_tokens = _beam_topk(
        logprobs.view(batch_size * beam_width_in, vocab_size), stage1_k
    )
    per_beam_vals = per_beam_vals.view(batch_size, beam_width_in, stage1_k)
    per_beam_tokens = per_beam_tokens.view(batch_size, beam_width_in, stage1_k)
    # Stage 2: adjust only the survivors and pick the global top-k.
    keys = per_beam_vals
    if diversity_rate is not None:
        rate = (
            diversity_rate.view(-1, 1, 1)
            if isinstance(diversity_rate, torch.Tensor)
            else diversity_rate
        )
        if source_beam_indices is None:
            source_beam_indices = torch.arange(
                beam_width_in, device=logprobs.device, dtype=torch.int32
            )
        keys = keys + rate * source_beam_indices[:beam_width_in].view(1, -1, 1)
    if length_penalty is not None:
        assert cand_gen_lengths is not None, (
            "cand_gen_lengths is required when length_penalty is active"
        )
        exponent = (
            length_penalty.view(-1, 1)
            if isinstance(length_penalty, torch.Tensor)
            else length_penalty
        )
        # Candidate lengths are >= 1 by construction (active beams:
        # generated + 1; finished beams froze after generating at least one
        # token), so the power is always well-defined.
        penalty_factor = cand_gen_lengths.to(logprobs.dtype).pow(exponent)
        keys = keys / penalty_factor.unsqueeze(-1)
    _, selected = _beam_topk(keys.view(batch_size, -1), beam_width_out)
    sorted_logprobs = per_beam_vals.view(batch_size, -1).gather(1, selected)
    predecessor_beams = (selected // stage1_k).to(torch.int32)
    tokens = per_beam_tokens.view(batch_size, -1).gather(1, selected).to(torch.int32)
    return sorted_logprobs, predecessor_beams, tokens


class CBAStepResult(NamedTuple):
    """Return of ``_cba_step_math``. A NamedTuple (not a dataclass) so it is a
    valid output of the torch.compile'd fullgraph function while still naming
    the eleven tensors the caller writes back."""

    slot_pred: torch.Tensor
    """[bs, num_beams] int32, predecessor beam of each continuing slot."""
    slot_tok: torch.Tensor
    """[bs, num_beams] int32, token of each continuing slot."""
    slot_cum: torch.Tensor
    """[bs, num_beams] float32, cumulative log-prob of each continuing slot."""
    step_log_probs: torch.Tensor
    """[bs, num_beams] float32, this step's per-slot log-prob."""
    top_normed: torch.Tensor
    """[bs, pool_width] float32, merged CBA pool normalized scores."""
    merged_cum: torch.Tensor
    """[bs, pool_width] float32, merged CBA pool cumulative log-probs."""
    merged_len: torch.Tensor
    """[bs, pool_width] int32, merged CBA pool lengths."""
    merged_tokens: torch.Tensor
    """[bs, pool_width, snap_len] int32, merged CBA pool path snapshots."""
    merged_lps: torch.Tensor
    """[bs, pool_width, snap_len] float32, merged CBA pool per-token log-probs."""
    done: torch.Tensor
    """[bs] bool, per-slot beam-search termination verdict."""
    reordered_window: Optional[torch.Tensor]
    """Stop-word window reordered to follow the beam swap, or None when no
    stop-word window is present."""


def _cba_step_math(
    # candidates (from beam_candidate_topk)
    cand_cum: torch.Tensor,  # [bs, C] float32
    cand_pred: torch.Tensor,  # [bs, C] int32
    cand_tok: torch.Tensor,  # [bs, C] int32
    # per-request state (store tensors + the group's slots)
    slots: torch.Tensor,  # [bs] int64
    seq_lens: torch.Tensor,  # [bs] int32
    snap_arange: torch.Tensor,  # [S] int64, S = bounded snapshot width
    exponent: torch.Tensor,  # [bs, 1] float32, 0 == no length penalty
    cache_indirection: torch.Tensor,
    original_tokens: torch.Tensor,
    original_log_probs: torch.Tensor,
    cum_log_probs: torch.Tensor,
    pending_harvest: torch.Tensor,
    end_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    cba_caps: torch.Tensor,
    cba_normed_scores: torch.Tensor,
    cba_cum_log_probs: torch.Tensor,
    cba_lengths: torch.Tensor,
    cba_tokens: torch.Tensor,
    cba_log_probs: torch.Tensor,
    stop_past_tokens: Optional[torch.Tensor],
    # static (graph-specializing) parameters
    beam_width_in: int,
    num_beams: int,
    early_stopping: int,  # BeamSearchEarlyStop; only ``FALSE`` (0) is special-cased
    max_seq_len: int,
) -> CBAStepResult:
    """Small-tensor math of the CBA beam-search step (see
    beam_search_sampling_batch_cba), written without data-dependent shapes so
    it can be fused by torch.compile: everything between candidate selection
    and the store writebacks. Pure — all mutations happen in the caller.

    NB: this function is torch.compile'd with ``fullgraph=True`` (see
    ``_cba_step_compiled``), which makes the following a contract rather than
    a style preference:

    - Keep it free of data-dependent shapes and of in-place ops on the input
      tensors (use out-of-place ``masked_fill`` etc.), so tracing stays
      fullgraph-clean.
    - The caller marks dim 0 of ``cand_cum``, ``cand_pred``, ``cand_tok``,
      ``slots``, ``seq_lens`` and ``exponent`` dynamic. Never read that
      dimension as a concrete size or branch on it in Python: recent dynamo
      raises if a ``mark_dynamic`` dim is specialized to a constant.
    - ``snap_arange`` is only ``maybe_mark_dynamic``, so specializing on its
      length is allowed. Its length is the snapshot width, which grows by one
      per step until it saturates at ``max_gen_len``; the resulting
      recompiles are expected to be absorbed by automatic-dynamic.
    - ``beam_width_in``, ``num_beams``, ``early_stopping`` and ``max_seq_len``
      are plain ints and therefore static. Each distinct combination costs one
      compilation, which is why the dispatch groups requests by them.
    """
    batch_size, num_candidates = cand_cum.shape
    neg_inf = float("-inf")

    harvest_mask = pending_harvest[slots, :beam_width_in]
    end_ids_b = end_ids[slots].view(-1, 1)
    caps = cba_caps[slots].view(-1, 1)
    prompts = prompt_lens[slots].view(-1, 1)
    gen_lens = (seq_lens - prompts.view(-1)).view(-1, 1)
    cand_len = gen_lens + 1

    # -inf candidates (from harvested rows) must count as slot fillers, not
    # end candidates, to preserve the >= K actives invariant even when every
    # beam was harvested at once.
    is_end = (cand_tok == end_ids_b) & torch.isfinite(cand_cum)
    cand_rank = torch.arange(num_candidates, device=cand_cum.device).view(1, -1)

    # --- Beam slots continue with the first K non-end candidates. Scatter
    # through a K+1-wide buffer instead of masked_select (data-dependent
    # shapes cannot be compiled); non-selected entries all collide on the
    # spare column and are discarded.
    active_mask = ~is_end
    active_pos = torch.cumsum(active_mask.to(torch.int32), dim=1) - 1
    scatter_idx = torch.where(active_mask & (active_pos < num_beams), active_pos, num_beams).long()

    def _take(src: torch.Tensor) -> torch.Tensor:
        buf = src.new_zeros(batch_size, num_beams + 1)
        buf.scatter_(1, scatter_idx, src)
        return buf[:, :num_beams]

    slot_pred = _take(cand_pred)
    slot_tok = _take(cand_tok)
    slot_cum = _take(cand_cum)
    step_log_probs = slot_cum - cum_log_probs[slots, :beam_width_in].gather(1, slot_pred.long())

    # --- CBA insertion: normalized scores of the new end-token candidates.
    # beam_candidate_topk applies the diversity term for *ranking* only and
    # returns raw cumulative log-probs, so the CBA insertion score below and
    # best_attainable further down are both diversity-free. This matches HF,
    # where diversity is a logits processor (HammingDiversityLogitsProcessor)
    # applied before log_softmax and so never reaches the accumulated score,
    # and vLLM, which ranks purely by cum_logprob / seq_len**length_penalty.
    #
    # The former C++ beam-search kernels diverged from all three: they folded
    # `diversityRate * source_beam_index` into the local log-probs, which then
    # flowed into the normalized CBA scores and the best-attainable score, so
    # with diversity_rate set they ordered the pool and reached the done verdict
    # differently. Do not
    # "fix" this by matching them: a cumulative_logprob carrying
    # `diversity_rate * beam_index` is no longer a log-probability, and the
    # offset depends on which slot the beam happened to occupy (TRTLLM-14792).
    new_normed = cand_cum / cand_len.to(cand_cum.dtype).pow(exponent)
    eligible = is_end & (cand_rank < caps)
    new_normed = new_normed.masked_fill(~eligible, neg_inf)

    # Snapshot candidate paths through the cache indirection (the work tree
    # is rewritten by later steps). Indirection entries beyond the current
    # length are uninitialized: lanes are masked below, but gather indices
    # must be clamped in-bounds first.
    step_idx = (prompts + snap_arange.view(1, -1)).clamp(max=cache_indirection.size(-1) - 1)
    step_idx_e = step_idx.unsqueeze(1).expand(-1, beam_width_in, -1)
    ind_at = torch.gather(cache_indirection[slots, :beam_width_in].long(), 2, step_idx_e)
    ind_at = ind_at.clamp_(0, beam_width_in - 1)
    tok_at = torch.gather(original_tokens[slots, :beam_width_in], 2, step_idx_e)
    lp_at = torch.gather(original_log_probs[slots, :beam_width_in], 2, step_idx_e)
    snap_len = snap_arange.size(0)
    parent_exp = cand_pred.long().unsqueeze(-1).expand(-1, -1, snap_len)
    src_beam = torch.gather(ind_at, 1, parent_exp)
    t_valid = snap_arange.view(1, 1, -1) < gen_lens.view(-1, 1, 1)
    new_paths = torch.gather(tok_at, 1, src_beam).masked_fill(~t_valid, BEAM_SEARCH_PAD_TOKEN)
    new_lp_paths = torch.gather(lp_at, 1, src_beam).masked_fill(~t_valid, 0.0)
    end_pos = gen_lens.view(-1, 1, 1).expand(-1, num_candidates, 1).clamp(max=snap_len - 1)
    new_paths = new_paths.scatter(2, end_pos, cand_tok.unsqueeze(-1).to(new_paths.dtype))
    # the terminating token's own log-prob = candidate cum - parent cum
    parent_cum = cum_log_probs[slots, :beam_width_in].gather(1, cand_pred.long())
    new_lp_paths = new_lp_paths.scatter(2, end_pos, (cand_cum - parent_cum).unsqueeze(-1))

    # Harvested beams (stop-word finishes latched after the previous step):
    # their recorded tokens already include the terminating stop word, so the
    # snapshot is the beam's own path (parent = itself) at the current length.
    harvest_paths = torch.gather(tok_at, 1, ind_at).masked_fill(~t_valid, BEAM_SEARCH_PAD_TOKEN)
    harvest_lp_paths = torch.gather(lp_at, 1, ind_at).masked_fill(~t_valid, 0.0)
    harvest_cum = cum_log_probs[slots, :beam_width_in]
    harvest_normed = harvest_cum / gen_lens.to(harvest_cum.dtype).pow(exponent)
    harvest_normed = harvest_normed.masked_fill(~harvest_mask, neg_inf)

    # --- Pool merge: keep the best pool_width by normalized score, then
    # enforce the per-request capacity (replace-min against the pool).
    pool_width = cba_normed_scores.size(1)
    all_normed = torch.cat([cba_normed_scores[slots], new_normed, harvest_normed], dim=1)
    top_normed, top_i = torch.topk(all_normed, k=pool_width, sorted=True, dim=-1)
    top_normed = top_normed.masked_fill(
        torch.arange(pool_width, device=cand_cum.device).view(1, -1) >= caps, neg_inf
    )
    all_cum = torch.cat([cba_cum_log_probs[slots], cand_cum, harvest_cum], dim=1)
    all_len = torch.cat(
        [
            cba_lengths[slots],
            cand_len.expand(-1, num_candidates),
            gen_lens.expand(-1, beam_width_in),
        ],
        dim=1,
    )
    all_tokens = torch.cat([cba_tokens[slots, :, :snap_len], new_paths, harvest_paths], dim=1)
    all_lps = torch.cat([cba_log_probs[slots, :, :snap_len], new_lp_paths, harvest_lp_paths], dim=1)
    merged_cum = all_cum.gather(1, top_i)
    merged_len = all_len.gather(1, top_i)
    top_i_wide = top_i.unsqueeze(-1).expand(-1, -1, snap_len)
    merged_tokens = all_tokens.gather(1, top_i_wide)
    merged_lps = all_lps.gather(1, top_i_wide)

    # --- Done verdict: CBA full, and the best candidate's
    # attainable normalized score cannot beat the worst kept entry.
    # early_stopping != FALSE (i.e. HF "never") assumes scores can still
    # increase with length for positive penalties, so unfinished beams have no
    # upper bound on attainability (they are bounded by max length); otherwise
    # (FALSE) the beam's current score is the correct bound (assume scores
    # decrease monotonically with sequence length, so longer sequences only get
    # less attractive).
    min_kept = top_normed.gather(1, (caps - 1).long()).view(-1)
    # `min_kept > neg_inf` means the pool holds `caps` finished hypotheses,
    # i.e. the C++ `numBeamsCBA[slot] >= nBM` test.
    pool_full = min_kept > neg_inf
    if early_stopping != BeamSearchEarlyStop.FALSE:
        max_gen = (max_seq_len - prompts.view(-1)).to(cand_len.dtype)
        bound_len = torch.where(exponent.view(-1) > 0, max_gen, cand_len.view(-1))
    else:
        bound_len = cand_len.view(-1)
    best_attainable = cand_cum[:, 0] / bound_len.to(cand_cum.dtype).pow(exponent.view(-1))
    # HF `True` stops as soon as `beam_width` finished candidates exist, without
    # weighing what is still attainable -- C++ beamStage3Kernel short-circuits
    # to done there. Fold that into the same expression rather than branching on
    # it: `early_stopping` is a plain int here, so an extra Python branch is an
    # extra Dynamo guard and recompilation of this fullgraph function.
    ignore_attainable = early_stopping == BeamSearchEarlyStop.TRUE
    done = pool_full & (ignore_attainable | (min_kept >= best_attainable))

    # Reorder the finish handler's rolling stop-word window to follow the
    # beam swap (matching stays correct across swaps).
    reordered_window = None
    if stop_past_tokens is not None:
        window = stop_past_tokens[:, slots, :beam_width_in]
        reordered_window = torch.gather(
            window, 2, slot_pred.long().unsqueeze(0).expand(window.size(0), -1, -1)
        )

    return CBAStepResult(
        slot_pred=slot_pred,
        slot_tok=slot_tok,
        slot_cum=slot_cum,
        step_log_probs=step_log_probs,
        top_normed=top_normed,
        merged_cum=merged_cum,
        merged_len=merged_len,
        merged_tokens=merged_tokens,
        merged_lps=merged_lps,
        done=done,
        reordered_window=reordered_window,
    )


# Compiled lazily on first use; the first CBA-mode request of a process pays
# the inductor compile once (subsequent shapes are covered by the dynamic
# batch/snapshot-length dims marked at the call site).
#
# Dynamo counts recompiles per code object for the lifetime of the process, and
# fullgraph turns exhaustion into a hard failure rather than a fallback to
# eager. A served engine sees one shape family and never approaches the default
# cap of 8, but a process that builds many engines -- a test session, or a
# worker reused across configurations -- does. Give this one function its own
# headroom so a later engine in the same process still compiles; the cap exists
# to catch runaway recompilation, not as a correctness property.
_CBA_RECOMPILE_LIMIT = 256


def _cba_step_compiled(*args: Any, **kwargs: Any) -> CBAStepResult:
    with torch._dynamo.config.patch(recompile_limit=_CBA_RECOMPILE_LIMIT):
        return _cba_step_compiled_inner(*args, **kwargs)


_cba_step_compiled_inner = torch.compile(_cba_step_math, dynamic=None, fullgraph=True)


def beam_search_sampling_batch_cba(
    logits: torch.Tensor,
    *,
    beam_width_in: int,
    beam_width_out: int,
    row_stride: int | None = None,
    beam_search_args: BeamSearchMetadata,
    temperature: float | None,
    early_stopping: int,  # BeamSearchEarlyStop
    length_penalty: "torch.Tensor | float | None" = None,
    diversity_rate: "torch.Tensor | float | None" = None,
    return_probs: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Beam-search step with a candidate-beams array (CBA). Every
    early_stopping mode runs here; the mode only selects the done verdict:

    - The top ``2 * beam_width`` expansion candidates (ranked by raw
      cumulative log-prob, plus the optional diversity adjustment — the
      length penalty does NOT enter candidate ranking) are split by end-token:
      end-token candidates ranked within the top ``beam_width`` are inserted
      into the CBA, which keeps the best ``beam_width`` finished paths seen so
      far by length-normalized score (path snapshots are taken eagerly since
      the work tree is rewritten by later steps). All beam slots then continue
      with the best non-end candidates, so exploration never narrows.
    - Stop words (any length) are detected by the finish handler after the
      step, which latches a per-beam finish reason; at the START of the next
      step this op harvests the latched beams into the CBA (their paths are
      complete, including the stop word) and masks their rows so the freed
      slots refill with active candidates — equivalent to coercing a finished
      beam into a top-ranked end-token candidate.
    - A request is done when the CBA is full and the best active candidate's
      attainable normalized score cannot beat the worst CBA entry.
      ``early_stopping == FALSE`` bounds attainability by the beam's current score
      (assume scores decrease monotonically with sequence length); any other
      value places no upper bound on attainability for unfinished beams (assume
      scores can increase with length, e.g. when ``length_penalty > 0``), so it
      bounds by ``max_seq_len`` (HF's "never"). The verdict is published by
      marking every beam slot finished, which drives the regular stop
      machinery.

    Execution contract:

    - **Mutates ``beam_search_args`` in place** and returns only the sampled
      tokens (and probabilities). The updated beam state -- ``cum_log_probs``,
      ``new_log_probs``, ``finished_beams``, ``predecessor_beams``,
      ``cache_indirection``, ``original_tokens``/``original_log_probs``, and
      the CBA fields (``cba_tokens``, ``cba_cum_log_probs``,
      ``cba_normed_scores``, ``cba_lengths``, ``batch_dones``) -- is read back
      from the caller's tensors, not from the return value. Only the rows
      selected by ``seq_slots`` are written.
    - Requires the CBA fields of ``BeamSearchMetadata`` to be set.
    - Calls ``mark_dynamic`` on the caller's tensors so a single compiled
      graph serves every batch size. The caller must not rely on those
      tensors keeping static shapes afterwards.
    - Dispatches between a ``torch.compile``d ``_cba_step_math`` and an eager
      fallback. Both compute the same values; the split exists because the
      scatter-style write-back does not survive fullgraph tracing, so it runs
      outside the compiled region (see the write-back comments below).
    """
    args = beam_search_args
    cba = args.cba
    assert cba is not None, "CBA metadata is required for beam search"
    num_beams = beam_width_out
    device = logits.device
    slots = args.seq_slots

    logprobs, softmax, batch_size = _beam_step_preprocess(
        logits,
        beam_width_in=beam_width_in,
        row_stride=row_stride,
        temperature=temperature,
        return_probs=return_probs,
        args=args,
    )
    # Beams latched finished by the finish handler after the previous step
    # are harvested into the CBA below; mask their rows so the freed slots
    # refill with active candidates from the other beams. Only STOP_WORDS
    # latches can reach this point: LENGTH fires on all beams at once (their
    # lengths are uniform here) and the END_ID flood below is all-beams too —
    # either way the request stops that same step, so there is no next step
    # to harvest them (finalize handles those paths instead).
    harvest_mask = args.pending_harvest[slots, :beam_width_in]
    logprobs = logprobs.masked_fill(harvest_mask.unsqueeze(-1), float("-inf"))
    logprobs += args.cum_log_probs.unsqueeze(-1)[slots, :beam_width_in]

    # --- Top 2K candidates by raw score (+ diversity), two-stage. The
    # "at least K non-end candidates" invariant that keeps the K beam slots
    # fillable relies on each source beam contributing at most one *end-token*
    # candidate, which holds because end-token detection matches the single
    # per-request end id (``end_ids``) below. Other ways a beam can finish do
    # NOT add end-token candidates and so cannot break the invariant:
    #   * Stop words (any length, including a second EOS supplied as a stop
    #     word) are latched by the finish handler at the previous step and
    #     harvested here (see the docstring). Their rows are masked to -inf
    #     above, so the isfinite guard on ``is_end`` counts them as active
    #     fillers, not end candidates.
    #   * LENGTH / all-beam END_ID floods stop the whole request that step, so
    #     there is no next step whose slots would need filling.
    # A model with multiple EOS tokens is therefore supported by registering
    # the extra EOS ids as stop words; matching several ids directly in
    # ``is_end`` would instead require widening the candidate pool and is out
    # of scope here.
    if not isinstance(diversity_rate, torch.Tensor) and not diversity_rate:
        diversity_rate = None
    # Each source beam contributes at most one finite end-token candidate, so
    # beam_width_in extra candidates on top of the slots to fill always leave
    # >= num_beams active ones — also on narrowing variable-beam-width steps
    # (beam_width_in > num_beams), where 2 * num_beams would not.
    cand_cum, cand_pred, cand_tok = beam_candidate_topk(
        logprobs,
        beam_width_out=num_beams + max(num_beams, beam_width_in),
        diversity_rate=diversity_rate,
        source_beam_indices=args.beam_idx_arange,
    )

    # Length penalty normalized to a per-request exponent tensor (0 == off),
    # so the compiled step math is branch-free.
    if isinstance(length_penalty, torch.Tensor):
        exponent = length_penalty.view(-1, 1).to(torch.float32)
    else:
        exponent = torch.full(
            (batch_size, 1), float(length_penalty or 0.0), dtype=torch.float32, device=device
        )

    snap_len = cba.cba_tokens.size(-1)
    if cba.max_gen_len > 0:
        # Snapshots and pool merges only ever touch generated positions, and
        # pool entry lengths are bounded by the running maximum generation
        # length; columns beyond it keep their (unread) previous content.
        snap_len = min(snap_len, cba.max_gen_len)
    snap_arange = torch.arange(snap_len, device=device)

    # CPU callers (unit tests) take the eager function: inductor's CPU
    # compile latency would dominate, and the fusion only pays off on CUDA.
    step_fn = _cba_step_compiled if logits.is_cuda else _cba_step_math
    if logits.is_cuda:
        for t in (cand_cum, cand_pred, cand_tok, slots, args.seq_lens, exponent):
            torch._dynamo.mark_dynamic(t, 0)
        # snap_arange's length is the (small, bounded) snapshot width, which the
        # compiled step reads as a size; newer dynamo raises if a mark_dynamic
        # dim gets specialized to a constant, so allow specialization here.
        torch._dynamo.maybe_mark_dynamic(snap_arange, 0)
    (
        slot_pred,
        slot_tok,
        slot_cum,
        step_log_probs,
        top_normed,
        merged_cum,
        merged_len,
        merged_tokens,
        merged_lps,
        done,
        reordered_window,
    ) = step_fn(
        cand_cum=cand_cum,
        cand_pred=cand_pred,
        cand_tok=cand_tok,
        slots=slots,
        seq_lens=args.seq_lens,
        snap_arange=snap_arange,
        exponent=exponent,
        cache_indirection=args.cache_indirection,
        original_tokens=cba.original_tokens,
        original_log_probs=cba.original_log_probs,
        cum_log_probs=args.cum_log_probs,
        pending_harvest=args.pending_harvest,
        end_ids=cba.end_ids,
        prompt_lens=cba.prompt_lens,
        cba_caps=cba.cba_caps,
        cba_normed_scores=cba.cba_normed_scores,
        cba_cum_log_probs=cba.cba_cum_log_probs,
        cba_lengths=cba.cba_lengths,
        cba_tokens=cba.cba_tokens,
        cba_log_probs=cba.cba_log_probs,
        stop_past_tokens=args.stop_past_tokens,
        beam_width_in=beam_width_in,
        num_beams=num_beams,
        early_stopping=early_stopping,
        max_seq_len=cba.max_seq_len,
    )
    # --- Writebacks (kept eager: mixed advanced/basic indexing on the store
    # tensors, and the compiled math stays pure).
    cba.cba_normed_scores[slots] = top_normed
    cba.cba_cum_log_probs[slots] = merged_cum
    cba.cba_lengths[slots] = merged_len
    cba.cba_tokens[slots, :, :snap_len] = merged_tokens
    cba.cba_log_probs[slots, :, :snap_len] = merged_lps
    cba.batch_dones[slots] = done
    # The latch is one-shot: the beams it named have just been pooled and their
    # slots refilled with unrelated continuations, so leaving it set would
    # harvest those on the next step (see BeamSearchStore.pending_harvest).
    args.pending_harvest[slots] = torch.zeros_like(args.pending_harvest[slots])
    # Publish the done verdict across the full row: the stop criterion reads the
    # first py_beam_width (== capacity) entries, which can exceed this step's
    # beam_width_out for variable-beam-width requests.
    #
    # This tensor is the finish handler's first_finish_reasons, which records
    # the reason each beam *first* finished by, so only fill entries still at
    # NOT_FINISHED. A beam that ended on a stop word does not freeze here -- it
    # vacates its slot and the request keeps generating -- so its STOP_WORDS
    # entry is the only record of why it ended; overwriting it would make the
    # request report whatever stopped it later instead.
    #
    # Beams with no reason of their own are ending because the pool can no
    # longer be beaten, not because they hit a token: report that as LENGTH,
    # matching what the pool-free path produced for the same situation.
    prev_reasons = args.finished_beams[slots]
    args.finished_beams[slots] = torch.where(
        done.view(-1, 1) & (prev_reasons == FinishReason.NOT_FINISHED.value),
        torch.full_like(prev_reasons, FinishReason.LENGTH.value),
        prev_reasons,
    )
    stop_window = args.stop_past_tokens
    if reordered_window is not None and stop_window is not None:
        stop_window[:, slots, :num_beams] = reordered_window

    # --- Beam-slot state updates (same contract as beam_search_sampling_batch_cba).
    args.predecessor_beams[slots, :num_beams] = slot_pred
    cache_indirection = args.cache_indirection[slots, :num_beams]
    cache_indirection_buffer = args.cache_indirection_buffer[slots, :beam_width_in]
    torch.gather(
        cache_indirection_buffer,
        dim=1,
        index=slot_pred.long().unsqueeze(2).expand(-1, -1, cache_indirection.size(2)),
        out=cache_indirection,
    )
    index = args.seq_lens.view(-1, 1, 1).expand(-1, num_beams, 1)
    src = args.beam_idx_arange[:num_beams].view(1, num_beams, 1).expand(batch_size, num_beams, 1)
    cache_indirection.scatter_(2, index, src)
    args.cache_indirection[slots, :num_beams] = cache_indirection

    args.new_log_probs[slots, :num_beams] = step_log_probs
    # Record this step's per-slot log-prob at the emission position so path
    # snapshots can recover per-token log-probs (analog of original_tokens).
    # Advanced indexing copies, so scatter into the copy and write it back.
    beam_log_probs = cba.original_log_probs[slots, :num_beams]
    beam_log_probs.scatter_(
        2,
        args.seq_lens.view(-1, 1, 1).expand(-1, num_beams, 1).long(),
        step_log_probs.unsqueeze(-1),
    )
    cba.original_log_probs[slots, :num_beams] = beam_log_probs
    args.cum_log_probs[slots, :num_beams] = slot_cum
    return _pad_next_tokens(slot_tok, args.finished_beams.size(1)), softmax


BeamHistoryBuilder: TypeAlias = Callable[[], BeamHistory | None]
"""Builder for BeamHistory.

Used to defer possibly unnecessary host-tensor construction until update_requests().
"""


@dataclass(kw_only=True)
class CBAGroupHost:
    """Host-side snapshot of the CBA state for a group of beam-search requests,
    produced by ``TorchSampler._prepare_cba_group_host`` and consumed by
    ``_prepare_beam_history_cba``. One batched D2H copy per tensor covers the
    whole group; the consumer slices per-request rows via ``pos[slot]``.
    """

    pos: dict[int, int]
    """Maps a request's seq slot to its row index in the batched tensors below."""
    should_stop: torch.Tensor
    cache_indirection: torch.Tensor
    original_tokens: torch.Tensor
    cum: torch.Tensor
    cba_tokens: torch.Tensor
    cba_cum: torch.Tensor
    cba_normed: torch.Tensor
    cba_lengths: torch.Tensor
    original_log_probs: Optional[torch.Tensor]
    """None when no request in the group requests log-probs."""
    cba_log_probs: Optional[torch.Tensor]
    """None when no request in the group requests log-probs."""


def prepare_beam_search(
    beam_search_store: BeamSearchStore,
    log_probs_store: LogProbsStore,
    seq_slots_long: torch.Tensor,
    max_prompt_len: int,
    prompt_lens_cuda: torch.Tensor,
    beam_caps_cuda: torch.Tensor,
) -> None:
    """Prepare the beam search buffers for the requests

    If the last context chunk is being processed,
    initialize/reset the buffers for the request.

    ``seq_slots_long`` must be int64 (required by ``index_fill_``).
    """
    beam_search_store.cache_indirection.narrow(2, 0, max_prompt_len).index_fill_(
        0, seq_slots_long, 0
    )
    beam_search_store.cum_log_probs.index_fill_(0, seq_slots_long, 0)
    log_probs_store.sampled_log_probs.index_fill_(0, seq_slots_long, 0)
    log_probs_store.sampled_log_prob_ranks.index_fill_(0, seq_slots_long, 0)
    beam_search_store.predecessor_beams.index_fill_(0, seq_slots_long, 0)
    beam_search_store.first_finish_reasons.index_fill_(
        0, seq_slots_long, FinishReason.NOT_FINISHED.value
    )
    beam_search_store.pending_harvest.index_fill_(0, seq_slots_long, False)
    beam_search_store.original_tokens.index_fill_(0, seq_slots_long, 0)
    beam_search_store.prompt_lens.index_copy_(0, seq_slots_long, prompt_lens_cuda)
    beam_search_store.batch_dones.index_fill_(0, seq_slots_long, False)
    # The CBA tensors only exist once a beam-search request has
    # been admitted; nothing reads them before that, so skip the reset.
    cba = beam_search_store.cba
    if cba is not None:
        cba.cba_tokens.index_fill_(0, seq_slots_long, BEAM_SEARCH_PAD_TOKEN)
        cba.cba_cum_log_probs.index_fill_(0, seq_slots_long, 0)
        cba.cba_normed_scores.index_fill_(0, seq_slots_long, float("-inf"))
        cba.cba_lengths.index_fill_(0, seq_slots_long, 0)
        cba.original_log_probs.index_fill_(0, seq_slots_long, 0)
        cba.cba_log_probs.index_fill_(0, seq_slots_long, 0)
        cba.cba_caps.index_copy_(0, seq_slots_long, beam_caps_cuda)


def _prepare_beam_history_cba(
    request: LlmRequest,
    *,
    cba_group: CBAGroupHost,
) -> BeamHistoryBuilder | None:
    """CBA-mode variant of ``_prepare_beam_history``.

    The final beams are the top ``beam_width`` of (CBA finished paths |
    current active slot paths) ranked by length-normalized score: the
    unfinished active paths are inserted into the CBA and everything is
    ranked together by normed score. All device state arrives through the
    group-level host snapshot (``cba_group``, see _prepare_cba_group_host);
    this function only slices host rows.
    """
    num_tokens = request.max_beam_num_tokens + 1  # last token is not yet added
    prompt_length = request.py_prompt_len
    num_generated_tokens = num_tokens - prompt_length
    num_beams = request.py_beam_width

    if num_generated_tokens == 0 or request.state == LlmRequestState.GENERATION_COMPLETE:
        return None

    slot = request.py_seq_slot
    assert slot is not None
    row = cba_group.pos[slot]
    # Active beams currently in the slots: the input width of the current
    # step (== num_beams except for variable-beam-width requests).
    active_width = _get_beam_width_in(request)
    return_log_probs = request.py_return_log_probs

    length_penalty = request.sampling_config.length_penalty or 0.0

    def _builder() -> BeamHistory | None:
        if not cba_group.should_stop[row].item():
            return None

        cache_indirection = cba_group.cache_indirection[
            row, :active_width, prompt_length:num_tokens
        ]
        current_path = cba_group.original_tokens[row, :active_width, prompt_length:num_tokens]
        active_cum = cba_group.cum[row, :active_width]
        cba_tokens = cba_group.cba_tokens[row]
        cba_cum = cba_group.cba_cum[row]
        cba_normed = cba_group.cba_normed[row]
        cba_lengths = cba_group.cba_lengths[row]

        active_path = _gather_beam_path(
            current_path=current_path, cache_indirection=cache_indirection
        )
        active_normed = active_cum
        if length_penalty != 0.0:
            active_normed = active_cum / float(num_generated_tokens) ** length_penalty
        active_lp_path: torch.Tensor | None = None
        cba_log_probs: torch.Tensor | None = None
        if return_log_probs:
            assert cba_group.cba_log_probs is not None
            assert cba_group.original_log_probs is not None
            cba_log_probs = cba_group.cba_log_probs[row]
            current_lp_path = cba_group.original_log_probs[
                row, :active_width, prompt_length:num_tokens
            ]
            active_lp_path = _gather_beam_path(
                current_path=current_lp_path, cache_indirection=cache_indirection
            )

        pool_width = cba_normed.size(0)
        all_normed = torch.cat([cba_normed, active_normed])
        order = torch.argsort(all_normed, descending=True)[:num_beams]

        width = max(num_generated_tokens, cast(int, cba_lengths.max().item()))
        tokens = torch.full((num_beams, width), BEAM_SEARCH_PAD_TOKEN, dtype=torch.int32)
        cum_logprobs = torch.zeros((num_beams,), dtype=torch.float32)
        log_probs: torch.Tensor | None = None
        if return_log_probs:
            log_probs = torch.zeros((num_beams, width), dtype=torch.float32)
        for out_idx, merged_idx in enumerate(order.tolist()):
            if not torch.isfinite(all_normed[merged_idx]):
                continue  # unreachable unless fewer finite candidates than
                # output beams (early termination edge); leaves a padded row
            if merged_idx < pool_width:  # CBA entry
                entry_len = int(cba_lengths[merged_idx].item())
                tokens[out_idx, :entry_len] = cba_tokens[merged_idx, :entry_len]
                cum_logprobs[out_idx] = cba_cum[merged_idx]
                if log_probs is not None:
                    assert cba_log_probs is not None
                    log_probs[out_idx, :entry_len] = cba_log_probs[merged_idx, :entry_len]
            else:
                active_idx = merged_idx - pool_width
                tokens[out_idx, :num_generated_tokens] = active_path[active_idx]
                cum_logprobs[out_idx] = active_cum[active_idx]
                if log_probs is not None:
                    assert active_lp_path is not None
                    log_probs[out_idx, :num_generated_tokens] = active_lp_path[active_idx]
        return BeamHistory(
            tokens=tokens,
            # [beam, tokens, 1]: the sampled token's logprob per position,
            # matching the shape contract of convert_logprobs_tensor_to_list.
            logprobs=log_probs.unsqueeze(-1) if log_probs is not None else None,
            logprobs_indices=tokens.unsqueeze(-1) if return_log_probs else None,
            cum_logprobs=cum_logprobs,
        )

    return _builder


def convert_logprobs_tensor_to_list(
    token_tensor: torch.Tensor,
    logprobs_tensor: torch.Tensor,
) -> list[list[dict[int, Logprob]]]:
    """Convert the logprobs tensor to a list of lists of dictionaries of Logprob objects

    Logprobs storage expects logprobs as a list[list[dict[int, Logprob]]] object

    args:
        token_tensor: torch.Tensor. Shape: beam_width, num_tokens, num_logprobs
        logprobs_tensor: torch.Tensor. Shape: beam_width, num_tokens, num_logprobs
    output:
        list[list[dict[int, Logprob]]]. Shape: (beam_width, num_tokens)
    """
    assert token_tensor.dim() == 3 and logprobs_tensor.dim() == 3, (
        f"Token and logprobs tensors must have 3 dimensions (beam_width, num_tokens, num_logprobs). \
        Got shapes (token_tensor) {token_tensor.shape} and (logprobs_tensor) {logprobs_tensor.shape} instead"
    )

    token_log_probs: list[list[dict[int, Logprob]]] = []
    token_list = token_tensor.tolist()
    logprobs_list = logprobs_tensor.tolist()
    for beam_idx in range(token_tensor.shape[0]):
        beam_token_log_probs: list[dict[int, Logprob]] = []
        for topk_token, topk_logprob in zip(token_list[beam_idx], logprobs_list[beam_idx]):
            logprobs = {
                token: Logprob(logprob=logprob, rank=rank + 1)
                for rank, (token, logprob) in enumerate(zip(topk_token, topk_logprob))
            }
            beam_token_log_probs.append(logprobs)
        token_log_probs.append(beam_token_log_probs)

    return token_log_probs


def finalize_beam(
    request: LlmRequest,
    beam_history: BeamHistory,
) -> None:
    """Update the request with the corrected tokens and logprobs for each beam.

    Args:
        request: The request to update
        beam_history: The beam history used to update the request
    """

    beam_width = request.py_beam_width
    assert beam_history.tokens.shape[0] == beam_width, (
        f"Beam_history.tokens.shape[0] should equal beam width: \
            {beam_history.tokens.shape[0]} != {beam_width}"
    )
    if request.py_return_log_probs:
        assert beam_history.logprobs is not None
        assert beam_history.logprobs_indices is not None
        assert beam_history.cum_logprobs is not None
        assert beam_history.logprobs.shape[0] == beam_width, (
            f"Beam_history.logprobs.shape[0] should equal beam width: \
                {beam_history.logprobs.shape[0]} != {beam_width}"
        )
        assert beam_history.logprobs_indices.shape[0] == beam_width, (
            f"Beam_history.logprobs_indices.shape[0] should equal beam width: \
                {beam_history.logprobs_indices.shape[0]} != {beam_width}"
        )
        assert beam_history.cum_logprobs.shape[0] == beam_width, (
            f"Beam_history.cum_logprobs.shape[0] should equal beam width: \
                {beam_history.cum_logprobs.shape[0]} != {beam_width}"
        )
    valid_tokens = (beam_history.tokens != BEAM_SEARCH_PAD_TOKEN).sum(dim=-1).tolist()
    gen_token_list = []
    gen_log_probs_list = []
    for beam_idx in range(beam_width):
        beam_valid_tokens = valid_tokens[beam_idx]
        gen_token_list.append(beam_history.tokens[beam_idx, :beam_valid_tokens].tolist())
        if request.py_return_log_probs:
            assert beam_history.logprobs_indices is not None
            assert beam_history.logprobs is not None
            gen_log_probs_list.append(
                convert_logprobs_tensor_to_list(
                    beam_history.logprobs_indices[beam_idx : beam_idx + 1, :beam_valid_tokens],
                    beam_history.logprobs[beam_idx : beam_idx + 1, :beam_valid_tokens],
                )[0]
            )
    request.set_generated_tokens(gen_token_list)
    if request.py_return_log_probs:
        # cum_log_probs will not change when padding with end tokens.
        # Therefore, we do not need to correct it
        assert beam_history.cum_logprobs is not None
        request.py_result.set_log_probs(
            gen_log_probs_list, cum_log_probs=beam_history.cum_logprobs.tolist()
        )


class BeamSearchHandler:
    """Owns the beam-search store and the host-side state the CBA path needs.

    ``TorchSampler`` holds one instance and drives it per step. The handler
    keeps the pieces that several beam-search helpers share -- the
    :class:`BeamSearchStore`, ``max_seq_len``, and the lagged
    ``first_finish_reasons`` snapshots used by the speculative-D2H predictor --
    so they need not be threaded through every call.
    """

    def __init__(
        self,
        *,
        store: Optional[BeamSearchStore],
        max_seq_len: int,
        max_num_sequences: int,
        use_speculative_d2h: bool,
        has_multi_token_stop_words: Callable[[LlmRequest], bool],
        copy_to_host: Callable[[torch.Tensor], torch.Tensor],
        make_side_stream_copier: Callable[[], AbstractContextManager[_SideStreamCopier]],
    ):
        self._store = store
        self._max_seq_len = max_seq_len
        self._use_speculative_d2h = use_speculative_d2h
        # Bound methods of the owning sampler: the D2H copies must share its
        # worker thread and side stream, so the SamplerEvent it records covers
        # them. A handler-owned stream would not be awaited by that event.
        self._copy_to_host = copy_to_host
        self._make_side_stream_copier = make_side_stream_copier
        # Lagged per-slot first_finish_reasons for the speculative predictor,
        # indexed by py_seq_slot. None for unoccupied slots or before the first
        # step; all-None in default mode.
        self._prev_first_finish_reasons: list[torch.Tensor | None] = [None] * max_num_sequences
        # Stop-word length check lives with finish-reason handling, which also
        # uses it outside beam search; injected rather than duplicated here.
        self._has_multi_token_stop_words = has_multi_token_stop_words

    def build_metadata(
        self,
        *,
        requests: list[LlmRequest],
        group_req_indices: torch.Tensor,
        seq_slots: torch.Tensor,
        seq_lens: torch.Tensor | None,
        seq_slots_cuda: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        num_requests: int,
        new_log_probs: torch.Tensor,
        end_ids_cuda: torch.Tensor,
        past_tokens_cuda: torch.Tensor,
    ) -> BeamSearchMetadata:
        """Build the beam-search metadata for a beam-search group.

        Assembles the per-step view the CBA op consumes: the persistent store
        tensors plus the group's slot/length tensors and the host-derived
        ``max_gen_len`` bound. Mirrors ``TopPDecayHandler.build_metadata``.

        ``new_log_probs``, ``end_ids_cuda`` and ``past_tokens_cuda`` are owned
        by the log-probs and finish-reason handlers respectively and are passed
        in rather than reached for, keeping this module independent of them.
        """
        store = self._store
        assert store is not None
        assert seq_lens is not None, "seq_lens is required for beam search"
        # Reuse the precomputed CUDA tensors when the strategy group
        # covers the full batch (typical single-strategy case);
        # otherwise fall back to a per-group H2D for the subset.
        if group_req_indices.size(0) == num_requests:
            group_seq_slots_cuda = seq_slots_cuda
            group_seq_lens_cuda = seq_lens_cuda
        else:
            group_seq_slots_cuda = seq_slots[group_req_indices].to(
                device="cuda", dtype=torch.int64, non_blocking=True
            )  # Should be on device for beam search, need long for index_copy_
            group_seq_lens_cuda = seq_lens[group_req_indices].to(
                device="cuda", non_blocking=True
            )  # Should be on device for beam search
        return BeamSearchMetadata(
            cache_indirection=store.cache_indirection,
            cache_indirection_buffer=store.cache_indirection_buffer,
            cum_log_probs=store.cum_log_probs,
            new_log_probs=new_log_probs,
            seq_slots=group_seq_slots_cuda,
            seq_lens=group_seq_lens_cuda,
            finished_beams=store.first_finish_reasons,
            pending_harvest=store.pending_harvest,
            predecessor_beams=store.predecessor_beams,
            beam_idx_arange=store.beam_idx_arange,
            stop_past_tokens=past_tokens_cuda,
            # None unless a beam-search request has been
            # admitted; the CBA tensors are not allocated before that.
            cba=None
            if store.cba is None
            else CBAState(
                # Context-only requests already carry the masked end id in the
                # store -- FinishReasonsHandler.prepare_for_new_request writes
                # the sentinel for them once, at setup -- so the CBA op reads
                # it from here with no per-step work.
                end_ids=end_ids_cuda,
                prompt_lens=store.prompt_lens,
                original_tokens=store.original_tokens,
                batch_dones=store.batch_dones,
                cba_tokens=store.cba.cba_tokens,
                cba_cum_log_probs=store.cba.cba_cum_log_probs,
                cba_normed_scores=store.cba.cba_normed_scores,
                cba_lengths=store.cba.cba_lengths,
                original_log_probs=store.cba.original_log_probs,
                cba_log_probs=store.cba.cba_log_probs,
                cba_caps=store.cba.cba_caps,
                max_seq_len=self._max_seq_len,
                max_gen_len=max(
                    (
                        requests[i].max_beam_num_tokens + 2 - requests[i].py_prompt_len
                        for i in group_req_indices.tolist()
                    ),
                    default=0,
                ),
            ),
        )

    def clear_slot(self, slot: int) -> None:
        """Drop stale predictor state from a prior occupant of ``slot``."""
        self._prev_first_finish_reasons[slot] = None

    def record_first_finish_reasons(self, slot: int, reasons: torch.Tensor | None) -> None:
        """Snapshot this step's first_finish_reasons for the next step's predictor."""
        self._prev_first_finish_reasons[slot] = reasons

    def predict_is_likely_finishing(
        self,
        request: LlmRequest,
        *,
        num_generated_tokens: int,
        num_tokens: int,
    ) -> bool:
        """Predict whether this step is likely to trigger beam history finalization.

        Returns True if any of:
          1. Length budget reached (max_new_tokens or max_seq_len).
          2. Multi-token stop_words configured (forces finalization).
          3. Lagged first_finish_reasons shows any beam finished previously.

        Known miss: all beams hit end_id on the same step from a clean state.
        """
        if num_generated_tokens >= request.py_max_new_tokens or num_tokens >= self._max_seq_len:
            return True
        if self._has_multi_token_stop_words(request):
            return True
        assert request.py_seq_slot is not None
        prev = self._prev_first_finish_reasons[request.py_seq_slot]
        # FinishReason.NOT_FINISHED == 0, so a nonzero entry implies that
        # some beam has already finished.
        if prev is not None and prev.any().item():
            return True
        return False

    @nvtx_range("_prepare_cba_group_host")
    def prepare_cba_group_host(
        self,
        requests: list[LlmRequest],
        finish_reasons: torch.Tensor,
        d2h_copier: Callable[[torch.Tensor], torch.Tensor],
    ) -> Optional[CBAGroupHost]:
        """Batch the per-step D2H state needed to finalize CBA-mode requests.

        The per-request variant issued ~8 small copies per request per step,
        which is host-call-count bound; one batched copy per tensor for the
        whole group replaces them (the builders slice the host rows).
        """
        # Every beam-search request runs on this path.
        cba_requests = list(requests)
        if not cba_requests:
            return None
        store = self._store
        assert store is not None
        # cba_requests is non-empty here, so the tensors were allocated at
        # admission (see BeamSearchStore.ensure_cba, called from
        # TorchSampler._sample_batched_by_strategy).
        cba = store.cba
        assert cba is not None, "CBA tensors must be allocated before a CBA step"
        slots = [request.py_seq_slot for request in cba_requests]
        assert all(slot is not None for slot in slots), "CBA requests must have seq slots"
        widths = [request.py_beam_width for request in cba_requests]
        both_host = torch.tensor([slots, widths], dtype=torch.int64, pin_memory=prefer_pinned())
        both_cuda = both_host.to(device="cuda", non_blocking=True)
        slots_cuda, widths_cuda = both_cuda[0], both_cuda[1]
        num_tokens_max = max(request.max_beam_num_tokens + 1 for request in cba_requests)
        attn_width = min(store.cache_indirection.size(-1), num_tokens_max)
        snap_width = min(
            cba.cba_tokens.size(-1),
            max(
                request.max_beam_num_tokens + 2 - request.py_prompt_len for request in cba_requests
            ),
        )
        group_reasons = finish_reasons[slots_cuda]
        should_stop = (
            (group_reasons > 0)
            | (
                torch.arange(group_reasons.size(1), device=group_reasons.device).view(1, -1)
                >= widths_cuda.view(-1, 1)
            )
        ).all(dim=1)
        return_log_probs = any(request.py_return_log_probs for request in cba_requests)
        return CBAGroupHost(
            pos={cast(int, slot): i for i, slot in enumerate(slots)},
            should_stop=d2h_copier(should_stop),
            cache_indirection=d2h_copier(store.cache_indirection[slots_cuda, :, :attn_width]),
            original_tokens=d2h_copier(store.original_tokens[slots_cuda, :, :attn_width]),
            cum=d2h_copier(store.cum_log_probs[slots_cuda]),
            cba_tokens=d2h_copier(cba.cba_tokens[slots_cuda, :, :snap_width]),
            cba_cum=d2h_copier(cba.cba_cum_log_probs[slots_cuda]),
            cba_normed=d2h_copier(cba.cba_normed_scores[slots_cuda]),
            cba_lengths=d2h_copier(cba.cba_lengths[slots_cuda]),
            original_log_probs=(
                d2h_copier(cba.original_log_probs[slots_cuda, :, :attn_width])
                if return_log_probs
                else None
            ),
            cba_log_probs=(
                d2h_copier(cba.cba_log_probs[slots_cuda, :, :snap_width])
                if return_log_probs
                else None
            ),
        )

    def prepare_beam_histories(
        self,
        requests: list[LlmRequest],
        finish_reasons: torch.Tensor,
    ) -> tuple[list[BeamHistoryBuilder | None], torch.cuda.Event | None]:
        """Create the corrected tokens and logprobs for each beam of a request.

        The builders returned by this function create a beam history object containing
        the corrected tokens and logprobs for each beam of a request.

        Returns (builders, side_stream_event). side_stream_event is set
        only when the speculative path queued copies; the caller must
        forward it to _record_sampler_event so SamplerEvent.synchronize
        awaits the side stream before any builder is invoked.
        """
        # The snapshot is only ever read when a request finishes on this step,
        # so in speculative mode skip it entirely on steps where no request can
        # finish. The predictor is per request but the copy is per group, so
        # the group is skipped only when every request is predicted
        # non-terminal; one possible finisher pays for the whole group, which
        # is the same copy it would have cost anyway.
        issue_copies = not self._use_speculative_d2h or any(
            self.predict_is_likely_finishing(
                req,
                num_generated_tokens=req.max_beam_num_tokens + 1 - req.py_prompt_len,
                num_tokens=req.max_beam_num_tokens + 1,
            )
            for req in requests
        )
        if not issue_copies:
            # No snapshot. A builder falls back to a synchronous read if the
            # step turns out to have finished a request after all -- see
            # _deferred_cba_group.
            return [self._speculative_builder(req) for req in requests], None

        # Single `with` for both modes; nullcontext yields None.
        copier_ctx: AbstractContextManager[_SideStreamCopier | None] = (
            self._make_side_stream_copier() if self._use_speculative_d2h else nullcontext()
        )
        with copier_ctx as copier:
            d2h_copier: Callable[[torch.Tensor], torch.Tensor] = (
                copier.stage_copy_to_host if copier is not None else self._copy_to_host
            )
            cba_group = self.prepare_cba_group_host(requests, finish_reasons, d2h_copier)
            builders = [_prepare_beam_history_cba(req, cba_group=cba_group) for req in requests]
        side_stream_event = copier.event if copier is not None else None
        return builders, side_stream_event

    def _speculative_builder(self, request: LlmRequest) -> BeamHistoryBuilder:
        """Builder for a step whose snapshot was skipped.

        Takes the snapshot synchronously when invoked. `should_stop` inside it
        still decides whether a history is produced, so a prediction that held
        costs one blocking copy of this request's rows and returns None; a miss
        costs the same copy and produces the history. Both stall the step,
        which is what the predictor trades against -- it is only worth enabling
        when most steps finalize nothing, in which case no builder for a
        skipped step is ever invoked in the first place.

        NB: deliberately not gated on a host-side finish check. The mirror the
        finish handler keeps (`_prev_first_finish_reasons`) lags by one step,
        so a request finishing on this step -- a stop word, say -- would read
        as unfinished and its history would be dropped.
        """

        def _builder() -> BeamHistory | None:
            store = self._store
            assert store is not None
            cba_group = self.prepare_cba_group_host(
                [request], store.first_finish_reasons, self._copy_to_host
            )
            if cba_group is None:
                return None
            inner = _prepare_beam_history_cba(request, cba_group=cba_group)
            return inner() if inner is not None else None

        return _builder
