# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a step is classified for the decode kernels: the kernels' bounds and `decode_step`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tensorrt_llm._torch.attention.backends import AttentionMetadata

# The K3 decode kernels' bounds: the token-count kernels take one tile of DECODE_MAX_TOKENS rows; the request-aware
# kernels take MAX_REQUESTS generation requests of at most MAX_TOKENS_PER_REQUEST tokens (1 + 7 drafts with DSpark).
DECODE_MAX_TOKENS = 8


MAX_REQUESTS = 8


MAX_TOKENS_PER_REQUEST = 8


@dataclass(frozen=True)
class DecodeStep:
    """A step the Kimi K3 decode kernels take: ``num_tokens`` rows and, on a pure decode step, ``num_requests``
    generation requests of ``tokens_per_request`` tokens each (None on a step with context requests)."""

    num_tokens: int
    num_requests: Optional[int] = None
    tokens_per_request: Optional[int] = None
    #: Whether the MoE layers' latent all-reduce is the routed experts' push form plus ``comm/k3_latent_reduce``
    #: (``decode_moe.py``): set by ``KimiLinearModel.forward`` from ``latent_push``.
    latent_push: bool = False

    @property
    def small(self) -> bool:
        """Whether the step fits one token tile of the token-count kernels."""
        return self.num_tokens <= DECODE_MAX_TOKENS

    @property
    def decode(self) -> bool:
        """Whether the step is a pure decode step the request-aware kernels take."""
        return self.num_requests is not None

    @property
    def wide(self) -> bool:
        """Whether the step is a pure decode step of more than one token tile: its token-count work keeps the decode
        layout's MoE head and tail, on M-general ops."""
        return self.decode and not self.small


def latent_push(
    step: DecodeStep,
    *,
    capturing: bool,
    breakable: bool,
    kda_draft_records: bool,
    kda_decode_kernels: bool,
    mla_decode_branch: bool,
) -> bool:
    """Whether a step's MoE layers push their routed partials (``DecodeStep.latent_push``): a pure decode step of at
    most 8 tokens whose attention layers all run the decode kernels, captured into a CUDA graph and not inside a
    breakable one. The KDA layers run them on one token per request with every KDA layer taking the K3 kernels
    (``kda_decode_kernels``), and on every verify width with the per-token states (``kda_draft_records``); the MLA
    layers where their decode branch takes the step (``mla_decode_branch``, the same for every MLA layer of a step).
    Every other step keeps the routed experts' all-reduce: the exchange's call-order invariant needs every kernel
    between a reduce and the next push to wait for its predecessor, which only those kernels were checked for, and a
    graph launch orders every pushing replay behind whatever ran before it."""
    if not (step.decode and step.small) or not capturing or breakable:
        return False
    if not (kda_draft_records or (step.tokens_per_request == 1 and kda_decode_kernels)):
        return False
    return mla_decode_branch


def decode_step(attn_metadata: AttentionMetadata, num_tokens: int) -> Optional[DecodeStep]:
    """The step's shape if any Kimi K3 decode kernel takes it, else None (the generic path runs).

    ``num_tokens`` is the step's token count (the rows of the model input). Read on the host from per-step integers
    and the host copy of the sequence lengths only. A CUDA graph is captured per decode batch shape, and every input
    here is fixed by that shape, so a captured step and its replays are classified alike.
    """
    if num_tokens <= 0:
        return None
    requests = _decode_requests(attn_metadata, num_tokens)
    if requests is not None:
        return DecodeStep(num_tokens, requests, num_tokens // requests)
    if num_tokens <= DECODE_MAX_TOKENS:
        return DecodeStep(num_tokens)
    return None


def _decode_requests(attn_metadata: AttentionMetadata, num_tokens: int) -> Optional[int]:
    """R when the step is R <= 8 generation requests of the same T <= 8 tokens and no context request, else None."""
    if attn_metadata.num_contexts != 0:
        return None
    requests = attn_metadata.num_generations
    if not 0 < requests <= MAX_REQUESTS or num_tokens % requests != 0:
        return None
    tokens = num_tokens // requests
    if not 0 < tokens <= MAX_TOKENS_PER_REQUEST:
        return None
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is not None and seq_lens.device.type == "cpu":
        lens = seq_lens[:requests]
        if lens.numel() != requests or bool((lens != tokens).any()):
            return None
    return requests
