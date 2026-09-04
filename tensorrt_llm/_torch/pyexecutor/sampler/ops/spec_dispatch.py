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
"""Backend dispatch for one-model speculative sampling."""

from typing import TYPE_CHECKING, Optional

import torch

from . import flashinfer as fi
from . import fused

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode


def spec_sample_from_logits(
    advanced_sampling_mode: "AdvancedSamplingMode",
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: Optional[torch.Tensor],
    *,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample one token per logits row. Returns ``[num_rows]``."""
    if advanced_sampling_mode.is_fused:
        return fused.fused_sample_from_logits(
            logits, temperatures, top_ks, top_ps, min_ps, seed=seed, offset=offset
        )
    eff_top_k, eff_top_p = fi.resolve_advanced_sampling_filters(
        advanced_sampling_mode, top_ks, top_ps
    )
    return fi.sample_from_logits_op(
        logits, temperatures, eff_top_k, eff_top_p, seed=seed, offset=offset
    )


def spec_sample_from_logits_with_probs(
    advanced_sampling_mode: "AdvancedSamplingMode",
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: Optional[torch.Tensor],
    *,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample tokens AND return the filtered, renormalized distribution.

    The draft sampler needs both, and needs them to come from one filtering pass: the
    rejection path divides the target's probability by the draft's, so a token sampled
    under one distribution and scored under another silently corrupts the acceptance test.
    """
    if advanced_sampling_mode.is_fused:
        return fused.fused_sample_from_logits_with_probs(
            logits, temperatures, top_ks, top_ps, min_ps, seed=seed, offset=offset
        )
    eff_top_k, eff_top_p = fi.resolve_advanced_sampling_filters(
        advanced_sampling_mode, top_ks, top_ps
    )
    return fi.sampling_batch_spec_dec_one_model_for_rejection(
        logits, temperatures, eff_top_k, eff_top_p, seed=seed, offset=offset
    )


def spec_compute_probs_from_logits(
    advanced_sampling_mode: "AdvancedSamplingMode",
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: Optional[torch.Tensor],
) -> torch.Tensor:
    """The filtered, renormalized distribution. Returns ``[num_rows, vocab]`` float32."""
    if advanced_sampling_mode.is_fused:
        return fused.fused_compute_probs_from_logits(logits, temperatures, top_ks, top_ps, min_ps)
    eff_top_k, eff_top_p = fi.resolve_advanced_sampling_filters(
        advanced_sampling_mode, top_ks, top_ps
    )
    return fi.compute_probs_from_logits(logits, temperatures, eff_top_k, eff_top_p)
