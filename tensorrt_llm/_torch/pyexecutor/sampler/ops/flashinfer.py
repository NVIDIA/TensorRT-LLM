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

"""FlashInfer-accelerated sampling kernels.

Pure kernel functions with no dependency on the sampling_utils interface
or other backend implementation modules. All flashinfer imports are guarded by
IS_FLASHINFER_AVAILABLE. Beam search is excluded (torch-only per design).
"""

from typing import Optional

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

if IS_FLASHINFER_AVAILABLE:
    import flashinfer.sampling


def top_k_top_p_sampling_from_logits_op(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    tokens: torch.Tensor = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, seed=seed, offset=offset
    )
    return tokens


def sampling_from_probs_op(
    probs: torch.Tensor,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    tokens: torch.Tensor = flashinfer.sampling.sampling_from_probs(
        probs, deterministic=True, seed=seed, offset=offset
    )
    return tokens


def softmax_op(
    logits: torch.Tensor,
    temperature: Optional[torch.Tensor],
) -> torch.Tensor:
    probs: torch.Tensor = flashinfer.sampling.softmax(
        logits, temperature, enable_pdl=get_env_enable_pdl()
    )
    return probs


def top_k_mask_logits_op(
    logits: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    masked: torch.Tensor = flashinfer.sampling.top_k_mask_logits(logits, top_k)
    return masked


def top_p_renorm_probs_op(
    probs: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    renormed: torch.Tensor = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    return renormed


def sampling_from_probs_generator_op(
    probs: torch.Tensor,
    generator: Optional[torch.Generator],
    check_nan: bool = False,
) -> torch.Tensor:
    tokens: torch.Tensor = flashinfer.sampling.sampling_from_probs(
        probs, deterministic=True, generator=generator, check_nan=check_nan
    )
    return tokens


def top_k_top_p_sampling_from_logits_with_generator_op(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    generator: Optional[torch.Generator],
    check_nan: bool = False,
) -> torch.Tensor:
    tokens: torch.Tensor = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits,
        top_k=top_k,
        top_p=top_p,
        filter_apply_order="top_k_first",
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
    )
    return tokens


def top_k_sampling_from_probs_generator_op(
    probs: torch.Tensor,
    top_k: torch.Tensor,
    generator: Optional[torch.Generator],
    check_nan: bool = False,
) -> torch.Tensor:
    tokens: torch.Tensor = flashinfer.sampling.top_k_sampling_from_probs(
        probs,
        top_k=top_k,
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
    )
    return tokens


def top_p_sampling_from_probs_generator_op(
    probs: torch.Tensor,
    top_p: torch.Tensor,
    generator: Optional[torch.Generator],
    check_nan: bool = False,
) -> torch.Tensor:
    tokens: torch.Tensor = flashinfer.sampling.top_p_sampling_from_probs(
        probs,
        top_p=top_p,
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
    )
    return tokens


def compute_probs_from_logits_op(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: Optional[torch.Tensor] = None,
    top_p: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FlashInfer fast path for compute_probs_from_logits with per-request tensors.

    Used by the spec-decoding path where each request may have different
    temperature / top-k / top-p values.  Note: temperature is applied AFTER
    optional top-k masking (via fused flashinfer softmax+temp).
    """
    if top_k is not None:
        logits = flashinfer.sampling.top_k_mask_logits(logits, top_k)
    probs: torch.Tensor = flashinfer.sampling.softmax(
        logits, temperatures, enable_pdl=get_env_enable_pdl()
    )
    if top_p is not None:
        probs = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    return probs
