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


def publish_target_logprobs(
    outputs: dict,
    logits: torch.Tensor | None,
    vocab_sharded: bool,
    spec_metadata,
) -> None:
    """Publish the target logprob of every emitted token as ``outputs[OUTPUT_KEY]``.

    Called by ``SpecWorkerBase.forward`` after the worker has accepted tokens, so it covers every worker and every
    model through one launch. The scores read the dense target logits the worker verified. Nothing is published for a
    layout this cannot read (vocabulary-sharded logits, ragged windows, tree rows); the sampler then rejects a
    request that asks, and a batch without one is unaffected because the launch skips rows without a divisor. Every
    condition here is static for a captured graph.

    Args:
        outputs: The worker outputs; ``new_tokens`` ``[B, width]`` and ``new_tokens_lens`` ``[B]`` are read.
        logits: The target logits ``[rows, vocab]`` the worker was given, or None.
        vocab_sharded: Whether ``logits`` are this rank's vocabulary columns.
        spec_metadata: The step's speculative metadata, which owns the per-request divisors.
    """
    divisors = None if spec_metadata is None else spec_metadata.request_logprob_temperatures
    if (
        divisors is None
        or vocab_sharded
        or outputs.get("logits_vocab_shard")
        or spec_metadata.sharded_target_logits
        or spec_metadata.is_ragged_verify
        or not isinstance(logits, torch.Tensor)
        or logits.dim() != 2
    ):
        return
    new_tokens = outputs["new_tokens"]
    new_tokens_lens = outputs["new_tokens_lens"]
    batch, width = new_tokens_lens.shape[0], new_tokens.shape[1]
    # The layout is one row per context request and ``width`` rows per generation request, so the row count
    # fixes the number of context requests. Reading it from the attention metadata is not safe: a worker may
    # leave that field changed after its draft passes.
    missing_rows = batch * width - logits.shape[0]
    if batch > divisors.numel() or width < 2 or missing_rows < 0 or missing_rows % (width - 1):
        return
    num_contexts = missing_rows // (width - 1)
    outputs[OUTPUT_KEY] = target_logprobs(
        logits,
        new_tokens,
        new_tokens_lens,
        divisors[:batch],
        num_contexts=num_contexts,
        width=width,
    )
