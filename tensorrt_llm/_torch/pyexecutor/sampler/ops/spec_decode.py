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

"""Speculative-decoding sampling ops.

Thin, per-request-tensor sampling helpers over the vanilla and flashinfer ops,
used by the speculative-decoding paths (one-model draft sampling, rejection
sampling). Pure tensor operations — no dependency on Strategy or LlmRequest.
"""

from typing import Optional

import torch

from . import flashinfer, vanilla


def sanitize_top_k(top_k: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map ``top_k`` into a backend-safe range before top-k filtering.

    Per ``SamplingParams``, ``top_k == 0`` means "all logits" (top-k disabled),
    but the flashinfer top-k kernels (``top_k_mask_logits``) break on a literal
    0 — they mask the entire row (all-zero probs). Map any non-positive value
    (and any oversized disable sentinel such as ``INT32_MAX``) to
    ``vocab_size`` (== keep all tokens), leaving genuine top_k values
    untouched.
    """
    return top_k.clamp(max=vocab_size).masked_fill_(top_k <= 0, vocab_size)


@torch.compile(options={"max-autotune": True})
def compute_probs_from_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: Optional[torch.Tensor],
    top_p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute filtered+normalized probs via flashinfer (hard dependency).

    ``temperatures``, ``top_k``, ``top_p`` are per-request tensors matching the
    spec-decoding call site in interface.py.
    """
    if top_k is not None:
        top_k = sanitize_top_k(top_k, logits.shape[-1])

    return flashinfer.compute_probs_from_logits_op(logits, temperatures, top_k, top_p)


@torch.compile(options={"max-autotune": True})
def sampling_batch_spec_dec_one_model(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUDA-graph compatible sampling; supports mixed sampling params. Returns sampled tokens."""
    top_k = sanitize_top_k(top_k, logits.shape[-1])
    # Greedy rows (temperature <= threshold) reduce to top_k=1 sampling: with the
    # divisor clamped to 1.0 by safely_apply_temperature_inplace (order-preserving
    # for those rows), flashinfer deterministically returns the max-probability
    # token, i.e. the argmax of the original logits. All ops remain branch-free
    # (no data-dependent control flow), so this stays CUDA-graph safe.
    is_greedy = temperatures <= vanilla.GREEDY_TEMPERATURE_THRESHOLD
    top_k = torch.where(is_greedy, torch.ones_like(top_k), top_k)
    top_p = torch.where(is_greedy, torch.ones_like(top_p), top_p)
    logits = vanilla.safely_apply_temperature_inplace(logits, temperatures)
    return flashinfer.top_k_top_p_sampling_from_logits_op(
        logits, top_k, top_p, seed=seed, offset=offset
    )


@torch.compile(options={"max-autotune": True})
def sampling_batch_spec_dec_one_model_for_rejection(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draft sampler returning tokens AND probs for the downstream rejection-sampling path."""
    # Rejection sampling relies on flashinfer's seed/offset support for
    # determinism and cross-rank consistency.
    probs = compute_probs_from_logits(logits, temperatures, top_k, top_p)
    tokens = flashinfer.sampling_from_probs_op(probs, seed=seed, offset=offset)
    return tokens, probs
