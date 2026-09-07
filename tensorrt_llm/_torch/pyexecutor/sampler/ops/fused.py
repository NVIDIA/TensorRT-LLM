# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python interface to the fused sampling op.

A fused CUDA implementation applies temperature, min-p, top-k and top-p together and
skips, per row on device, whichever of them that row leaves neutral. Its internal
dispatcher selects a single-CTA or multi-CTA execution family. The three entry points
below differ only in which outputs they ask for, so each family is instantiated without
the half it does not need:

=========================================  =========================================
``fused_sample_from_logits``               tokens -- the target/draft samplers
``fused_compute_probs_from_logits``        probs  -- target probs for rejection
``fused_sample_from_logits_with_probs``    both   -- draft sampler + its draft probs
=========================================  =========================================

Unlike the flashinfer chain in :mod:`.flashinfer`, no filter is resolved away on the host:
every parameter is passed as a per-row tensor and neutrality is decided on device. That is
what lets one deploy serve a batch mixing filtered and unfiltered requests without a mode
per combination -- and what lets ``min_p`` be added without doubling the mode space.

Disable sentinels, matching ``SpecMetadata._scan_one_model_sampling``: ``top_k`` outside
``(0, vocab_size)`` (including ``INT32_MAX``), ``top_p >= 1``, ``min_p <= 0``.
"""

from typing import Optional

import torch


def fused_sample_from_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    *,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample one token per row. Returns ``[num_rows]`` int32."""
    return torch.ops.trtllm.fused_sample_from_logits(
        logits, temperatures, top_ks, top_ps, min_ps, seed, offset
    )


def fused_sample_from_logits_with_probs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    *,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a token AND return the filtered, renormalized distribution.

    The rejection path needs both, and needs them to come from the *same* filtering, or
    acceptance is computed against a distribution neither side sampled from.
    """
    return torch.ops.trtllm.fused_sample_from_logits_with_probs(
        logits, temperatures, top_ks, top_ps, min_ps, seed, offset
    )


def fused_compute_probs_from_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
) -> torch.Tensor:
    """The filtered, renormalized distribution. Returns ``[num_rows, vocab]`` float32."""
    return torch.ops.trtllm.fused_compute_probs_from_logits(
        logits, temperatures, top_ks, top_ps, min_ps
    )
