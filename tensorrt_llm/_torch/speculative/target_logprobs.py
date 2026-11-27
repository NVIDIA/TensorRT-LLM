# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-model logprobs of the tokens a one-model speculative step emits.

For request ``b`` and emitted position ``j`` the score is

    logit[row(b, j), token(b, j)] / T  -  logsumexp(logit[row(b, j), :] / T)

with ``T = 1`` for ``LogprobMode.RAW`` and the request temperature for ``PROCESSED``. Row ``j`` is the target
distribution given the first ``j`` emitted tokens, so the same expression scores an accepted draft, the token
that replaces a rejected draft, and the bonus token: each is marginally a draw from that target row.

The scores come from one fused launch (two when a few rows must be split over the vocabulary to fill the GPU):

* one online-softmax pass reads every needed logits row exactly once, so nothing is materialised;
* a request that does not ask (``temperature == 0``) or a position past ``new_tokens_lens`` exits before it
  touches the logits, so a batch without logprob requests moves no bytes;
* shapes are static (``B * width`` programs), there is no host synchronisation, and the launch is CUDA-graph safe.

Layout of the logits rows: context requests first, one row each; then ``width = draft_len + 1`` rows per
generation request. A ragged verify layout passes ``row_start`` (the first row of each request).
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

OUTPUT_KEY = "target_logprobs"
"""str: key of the worker outputs under which the model publishes the scores, ``[B, width]`` fp32."""

_BLOCK = 4096
_NUM_WARPS = 8
# A row is split over the vocabulary only while each part still has this many entries to read.
_MIN_PART = 16384
_MAX_SPLITS = 32


@triton.jit
def _row_of(b, j, num_contexts, width, row_start_ptr, HAS_ROW_START: tl.constexpr):
    if HAS_ROW_START:
        return tl.load(row_start_ptr + b).to(tl.int64) + j
    # Context requests own one row each, generation requests ``width`` rows after them.
    return tl.where(
        b < num_contexts,
        b,
        num_contexts + (b - num_contexts) * width + j,
    ).to(tl.int64)


@triton.jit
def _online_softmax_part(
    logits_ptr,
    row,
    stride,
    start,
    stop,
    temperature,
    BLOCK: tl.constexpr,
):
    """Running (max, sum of exp) of ``logits[row, start:stop] / temperature``."""
    row_ptr = logits_ptr + row * stride
    offsets = tl.arange(0, BLOCK)
    run_max = tl.full((BLOCK,), float("-inf"), tl.float32)
    run_sum = tl.zeros((BLOCK,), tl.float32)
    for base in range(start, stop, BLOCK):
        idx = base + offsets
        x = tl.load(row_ptr + idx, mask=idx < stop, other=float("-inf")).to(tl.float32)
        x = x / temperature
        new_max = tl.maximum(run_max, x)
        # An all -inf lane (guided-decoding mask) must not turn exp(-inf - -inf) into NaN.
        safe = tl.where(new_max == float("-inf"), 0.0, new_max)
        run_sum = run_sum * tl.exp(run_max - safe) + tl.exp(x - safe)
        run_max = new_max
    part_max = tl.max(run_max, axis=0)
    safe_max = tl.where(part_max == float("-inf"), 0.0, part_max)
    part_sum = tl.sum(run_sum * tl.exp(run_max - safe_max), axis=0)
    return part_max, part_sum


@triton.jit
def _target_logprob_kernel(
    out_ptr,  # [B, width] fp32
    part_max_ptr,  # [B * width, SPLITS] fp32 (unused when SPLITS == 1)
    part_sum_ptr,
    logits_ptr,  # [rows, V]
    stride,
    tokens_ptr,  # [B, tokens_stride] emitted tokens
    tokens_stride,
    lens_ptr,  # [B] number of emitted tokens
    temperature_ptr,  # [B] 0 = no logprob requested
    row_start_ptr,
    num_contexts,
    width,
    vocab,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ROW_START: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    b = pid // width
    j = pid % width
    temperature = tl.load(temperature_ptr + b)
    active = (temperature != 0.0) & (j < tl.load(lens_ptr + b))
    if not active:
        if SPLITS == 1:
            tl.store(out_ptr + pid, 0.0)
        return
    row = _row_of(b, j, num_contexts, width, row_start_ptr, HAS_ROW_START)
    part = (vocab + SPLITS - 1) // SPLITS
    start = split * part
    stop = tl.minimum(start + part, vocab)
    part_max, part_sum = _online_softmax_part(
        logits_ptr, row, stride, start, stop, temperature, BLOCK
    )
    if SPLITS == 1:
        token = tl.load(tokens_ptr + b * tokens_stride + j).to(tl.int64)
        logit = tl.load(logits_ptr + row * stride + token).to(tl.float32) / temperature
        safe_max = tl.where(part_max == float("-inf"), 0.0, part_max)
        tl.store(out_ptr + pid, logit - (safe_max + tl.log(part_sum)))
    else:
        tl.store(part_max_ptr + pid * SPLITS + split, part_max)
        tl.store(part_sum_ptr + pid * SPLITS + split, part_sum)


@triton.jit
def _combine_kernel(
    out_ptr,
    part_max_ptr,
    part_sum_ptr,
    logits_ptr,
    stride,
    tokens_ptr,
    tokens_stride,
    lens_ptr,
    temperature_ptr,
    row_start_ptr,
    num_contexts,
    width,
    SPLITS: tl.constexpr,
    HAS_ROW_START: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // width
    j = pid % width
    temperature = tl.load(temperature_ptr + b)
    active = (temperature != 0.0) & (j < tl.load(lens_ptr + b))
    if not active:
        tl.store(out_ptr + pid, 0.0)
        return
    parts = tl.arange(0, SPLITS)
    part_max = tl.load(part_max_ptr + pid * SPLITS + parts)
    part_sum = tl.load(part_sum_ptr + pid * SPLITS + parts)
    max_all = tl.max(part_max, axis=0)
    safe = tl.where(max_all == float("-inf"), 0.0, max_all)
    total = tl.sum(part_sum * tl.exp(part_max - safe), axis=0)
    row = _row_of(b, j, num_contexts, width, row_start_ptr, HAS_ROW_START)
    token = tl.load(tokens_ptr + b * tokens_stride + j).to(tl.int64)
    logit = tl.load(logits_ptr + row * stride + token).to(tl.float32) / temperature
    tl.store(out_ptr + pid, logit - (safe + tl.log(total)))


@triton.jit
def _shard_stats_kernel(
    stats_ptr,  # [B * width, 3] fp32: running max, sum of exp, logit of the emitted token if this shard owns it
    logits_ptr,  # [rows, shard_width] this rank's vocabulary columns
    stride,
    tokens_ptr,
    tokens_stride,
    lens_ptr,
    temperature_ptr,
    num_contexts,
    width,
    valid_cols,  # columns of this shard that are real vocabulary entries
    col_base,  # global vocabulary index of the shard's first column
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // width
    j = pid % width
    temperature = tl.load(temperature_ptr + b)
    active = (temperature != 0.0) & (j < tl.load(lens_ptr + b))
    if not active:
        tl.store(stats_ptr + pid * 3, float("-inf"))
        tl.store(stats_ptr + pid * 3 + 1, 0.0)
        tl.store(stats_ptr + pid * 3 + 2, 0.0)
        return
    row = _row_of(b, j, num_contexts, width, lens_ptr, False)
    part_max, part_sum = _online_softmax_part(
        logits_ptr, row, stride, 0, valid_cols, temperature, BLOCK
    )
    token = tl.load(tokens_ptr + b * tokens_stride + j).to(tl.int64)
    local = token - col_base
    owned = (local >= 0) & (local < valid_cols)
    logit = (
        tl.load(logits_ptr + row * stride + tl.where(owned, local, 0)).to(tl.float32) / temperature
    )
    tl.store(stats_ptr + pid * 3, part_max)
    tl.store(stats_ptr + pid * 3 + 1, part_sum)
    tl.store(stats_ptr + pid * 3 + 2, tl.where(owned, logit, 0.0))


@lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _num_splits(programs: int, vocab: int, device: torch.device) -> int:
    """Vocabulary parts per row: enough programs to fill the GPU, each part still long enough to stream."""
    want = triton.cdiv(4 * _num_sms(device.index or 0), max(programs, 1))
    cap = max(1, vocab // _MIN_PART)
    return min(
        triton.next_power_of_2(want), triton.next_power_of_2(cap) if cap > 1 else 1, _MAX_SPLITS
    )


def target_logprobs(
    logits: torch.Tensor,
    new_tokens: torch.Tensor,
    new_tokens_lens: torch.Tensor,
    temperature: torch.Tensor,
    *,
    num_contexts: int,
    width: int,
    row_start: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Logprobs of the emitted tokens under the target logits.

    Args:
        logits: ``[rows, vocab]`` target logits (bf16 or fp32, row stride arbitrary, last dim contiguous).
        new_tokens: ``[B, >= width]`` emitted tokens; only the first ``new_tokens_lens[b]`` are read.
        new_tokens_lens: ``[B]`` number of emitted tokens per request.
        temperature: ``[B]`` fp32. ``0`` skips the request; ``1`` is RAW, the request temperature PROCESSED.
        num_contexts: leading requests that own one logits row (context requests).
        width: ``draft_len + 1``, the output columns and the rows per generation request.
        row_start: optional ``[B]`` first logits row of each request (ragged verify layouts).
        out: optional preallocated ``[B, width]`` fp32 result (a CUDA-graph output buffer).

    Returns:
        ``[B, width]`` fp32. Entries of skipped requests and positions past ``new_tokens_lens`` are 0.
    """
    if logits.dim() != 2 or logits.stride(-1) != 1:
        raise ValueError("logits must be [rows, vocab] with a contiguous last dimension")
    batch = new_tokens_lens.shape[0]
    vocab = logits.shape[1]
    if new_tokens.dim() != 2 or new_tokens.shape[0] != batch or new_tokens.shape[1] < width:
        raise ValueError("new_tokens must be [B, >= width]")
    if temperature.shape[0] < batch:
        raise ValueError("one temperature per request is required")
    if out is None:
        out = torch.empty((batch, width), dtype=torch.float32, device=logits.device)
    programs = batch * width
    if programs == 0:
        return out
    splits = _num_splits(programs, vocab, logits.device)
    part_max = part_sum = out
    if splits > 1:
        part_max = torch.empty((programs, splits), dtype=torch.float32, device=logits.device)
        part_sum = torch.empty_like(part_max)
    has_row_start = row_start is not None
    common = (
        logits,
        logits.stride(0),
        new_tokens,
        new_tokens.stride(0),
        new_tokens_lens,
        temperature,
        row_start if has_row_start else new_tokens_lens,
        num_contexts,
        width,
    )
    _target_logprob_kernel[(programs, splits)](
        out,
        part_max,
        part_sum,
        common[0],
        common[1],
        common[2],
        common[3],
        common[4],
        common[5],
        common[6],
        common[7],
        common[8],
        vocab,
        SPLITS=splits,
        BLOCK=_BLOCK,
        HAS_ROW_START=has_row_start,
        num_warps=_NUM_WARPS,
    )
    if splits > 1:
        _combine_kernel[(programs,)](
            out,
            part_max,
            part_sum,
            *common,
            SPLITS=splits,
            HAS_ROW_START=has_row_start,
            num_warps=1,
        )
    return out


def shard_stats(
    local_logits: torch.Tensor,
    new_tokens: torch.Tensor,
    new_tokens_lens: torch.Tensor,
    temperature: torch.Tensor,
    *,
    num_contexts: int,
    width: int,
    col_base: int,
    vocab_size: int,
) -> torch.Tensor:
    """One rank's partial statistics for vocabulary-sharded logits: ``[B * width, 3]`` fp32 holding, per scored row,
    the running max, the sum of exponentials, and the emitted token's logit when this shard owns its column.

    Args:
        local_logits: ``[rows, shard_width]`` this rank's columns (bf16 or fp32).
        new_tokens: ``[B, >= width]`` emitted tokens.
        new_tokens_lens: ``[B]`` number of emitted tokens per request.
        temperature: ``[B]`` fp32 divisors; ``0`` skips the request.
        num_contexts: leading requests that own one logits row.
        width: ``draft_len + 1``.
        col_base: global vocabulary index of this shard's first column.
        vocab_size: number of real vocabulary entries (columns past it are padding).
    """
    programs = new_tokens_lens.shape[0] * width
    stats = torch.empty((programs, 3), dtype=torch.float32, device=local_logits.device)
    valid_cols = max(0, min(local_logits.shape[1], vocab_size - col_base))
    _shard_stats_kernel[(programs,)](
        stats,
        local_logits,
        local_logits.stride(0),
        new_tokens,
        new_tokens.stride(0),
        new_tokens_lens,
        temperature,
        num_contexts,
        width,
        valid_cols,
        col_base,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return stats


def combine_shard_stats(gathered: torch.Tensor, batch: int, width: int) -> torch.Tensor:
    """Exact logprobs from the statistics of every rank (``[tp * B * width, 3]``, rank blocks in order).

    Returns ``[B, width]`` fp32, zero for skipped requests and positions past ``new_tokens_lens``.
    """
    ranks = gathered.view(-1, batch * width, 3)
    row_max, row_sum, logit = ranks[..., 0], ranks[..., 1], ranks[..., 2]
    top = row_max.max(dim=0).values
    safe_top = torch.where(torch.isinf(top), torch.zeros_like(top), top)
    total = (row_sum * torch.exp(row_max - safe_top)).sum(dim=0)
    scores = logit.sum(dim=0) - (safe_top + torch.log(total))
    return torch.where(total > 0, scores, torch.zeros_like(scores)).view(batch, width)


def target_logprobs_sharded(
    local_logits: torch.Tensor,
    new_tokens: torch.Tensor,
    new_tokens_lens: torch.Tensor,
    temperature: torch.Tensor,
    *,
    num_contexts: int,
    width: int,
    col_base: int,
    vocab_size: int,
    gather,
) -> torch.Tensor:
    """Logprobs of the emitted tokens when each tensor-parallel rank holds only its vocabulary columns.

    Every rank reduces its columns to three numbers per scored row (``shard_stats``); one all-gather of those numbers
    and a small combine (``combine_shard_stats``) give the exact log-softmax value on every rank, without gathering
    the logits. Arguments are those of ``shard_stats`` plus ``gather``, a callable that concatenates a ``[N, 3]``
    tensor of every rank along dim 0.
    """
    stats = shard_stats(
        local_logits,
        new_tokens,
        new_tokens_lens,
        temperature,
        num_contexts=num_contexts,
        width=width,
        col_base=col_base,
        vocab_size=vocab_size,
    )
    return combine_shard_stats(gather(stats), new_tokens_lens.shape[0], width)


_LOGGED_REASONS: set[str] = set()


def _unavailable(reason: str) -> None:
    """Say once why a step's emitted tokens are not scored; a request that asks then fails with a generic error."""
    if reason not in _LOGGED_REASONS:
        _LOGGED_REASONS.add(reason)
        logger.warning(f"target logprobs are not available for this speculative step: {reason}")


def _shard_view(logits, flagged: bool, target_lm_head):
    """``(local logits, global column of the first one, vocabulary size, tensor-parallel mapping)`` when ``logits``
    are this rank's vocabulary columns, else None. The worker hands them over either as a ``VocabShardedLogits``
    that carries its own head, or as a bare shard tensor with the step flagged as sharded."""
    from .vocab_parallel import VocabShardedLogits

    if isinstance(logits, VocabShardedLogits):
        head = logits.head
        return logits.local, head.col_base, head.vocab_size, head.lm_head.mapping
    if flagged and isinstance(logits, torch.Tensor) and target_lm_head is not None:
        mapping = target_lm_head.mapping
        return (
            logits,
            mapping.tp_rank * target_lm_head.out_features,
            target_lm_head.num_embeddings,
            mapping,
        )
    return None


def publish_target_logprobs(
    outputs: dict,
    logits,
    vocab_sharded: bool,
    spec_metadata,
    target_lm_head=None,
) -> None:
    """Publish the target logprob of every emitted token as ``outputs[OUTPUT_KEY]``.

    Called by ``SpecWorkerBase.forward`` after the worker has accepted tokens, so it covers every worker and every
    model through one launch. The scores read the target logits the worker verified: dense ``[rows, vocab]`` logits
    through one kernel, or this rank's vocabulary columns through a per-rank kernel and one tiny all-gather
    (``target_logprobs_sharded``). Nothing is published for a layout this cannot read (ragged windows, tree rows,
    sharded logits under attention-DP); the sampler then rejects a request that asks, and a batch without one is
    unaffected because the launch skips rows without a divisor. Every condition here is static for a captured graph.

    Args:
        outputs: The worker outputs; ``new_tokens`` ``[B, width]`` and ``new_tokens_lens`` ``[B]`` are read.
        logits: The target logits the worker was given: a ``[rows, vocab]`` tensor, a vocabulary shard, or a
            ``VocabShardedLogits``.
        vocab_sharded: Whether the worker was told that ``logits`` are this rank's vocabulary columns.
        spec_metadata: The step's speculative metadata, which owns the per-request divisors.
        target_lm_head: The target model's LM head, needed only when ``logits`` is a bare shard tensor.
    """
    divisors = None if spec_metadata is None else spec_metadata.request_logprob_temperatures
    if divisors is None:
        return _unavailable("the speculative metadata has no per-request logprob divisors")
    if spec_metadata.is_ragged_verify:
        return _unavailable("ragged verification windows are not supported")
    flagged = bool(
        vocab_sharded or outputs.get("logits_vocab_shard") or spec_metadata.sharded_target_logits
    )
    view = _shard_view(logits, flagged, target_lm_head)
    if view is not None:
        local, col_base, vocab_size, mapping = view
        if mapping.tp_size < 2 or mapping.enable_attention_dp:
            return _unavailable(
                "vocabulary-sharded logits are only scored for tensor parallelism without attention-DP"
            )
    elif flagged or not isinstance(logits, torch.Tensor):
        return _unavailable(
            f"the target logits ({type(logits).__name__}) are sharded but no LM head is bound"
        )
    else:
        local = logits
    if local.dim() != 2:
        return _unavailable(f"the target logits have {local.dim()} dimensions, not 2")
    new_tokens = outputs["new_tokens"]
    new_tokens_lens = outputs["new_tokens_lens"]
    batch, width = new_tokens_lens.shape[0], new_tokens.shape[1]
    # The layout is one row per context request and ``width`` rows per generation request, so the row count
    # fixes the number of context requests. Reading it from the attention metadata is not safe: a worker may
    # leave that field changed after its draft passes.
    missing_rows = batch * width - local.shape[0]
    if batch > divisors.numel() or width < 2 or missing_rows < 0 or missing_rows % (width - 1):
        return _unavailable(
            f"unexpected logits layout: {local.shape[0]} rows for {batch} requests of width {width}"
        )
    num_contexts = missing_rows // (width - 1)
    if view is not None:
        from ..distributed import allgather

        outputs[OUTPUT_KEY] = target_logprobs_sharded(
            local,
            new_tokens,
            new_tokens_lens,
            divisors[:batch],
            num_contexts=num_contexts,
            width=width,
            col_base=col_base,
            vocab_size=vocab_size,
            gather=lambda stats: allgather(stats, mapping, dim=0),
        )
        return
    outputs[OUTPUT_KEY] = target_logprobs(
        local,
        new_tokens,
        new_tokens_lens,
        divisors[:batch],
        num_contexts=num_contexts,
        width=width,
    )
