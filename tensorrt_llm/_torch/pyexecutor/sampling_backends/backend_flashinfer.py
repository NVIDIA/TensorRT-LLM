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


_GREEDY_TEMPERATURE_THRESHOLD = 1e-4


def compute_probs(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    skip_temperature: bool = False,
) -> torch.Tensor:
    """Compute filtered normalized probabilities using the FlashInfer fast path.

    When skip_temperature=True the fused softmax+temp step is bypassed (plain
    softmax only). When temperature <= _GREEDY_TEMPERATURE_THRESHOLD the result
    is a one-hot regardless of skip_temperature.
    """
    if not IS_FLASHINFER_AVAILABLE:
        raise RuntimeError("FlashInfer is required for backend_flashinfer.compute_probs")

    assert logits.dim() == 2, "logits must be 2D: [batch_size, vocab_size]"
    batch_size, vocab_size = logits.size()

    if temperature <= _GREEDY_TEMPERATURE_THRESHOLD:
        argmax_ids = logits.argmax(dim=-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(1, argmax_ids, 1.0)

    if top_k is not None and top_k < vocab_size:
        top_k_tensor = torch.full((batch_size,), top_k, dtype=torch.int32, device=logits.device)
        logits = flashinfer.sampling.top_k_mask_logits(logits, top_k_tensor)
        if skip_temperature:
            probs = flashinfer.sampling.softmax(logits, None, enable_pdl=get_env_enable_pdl())
        else:
            temp_tensor = torch.full(
                (batch_size,), temperature, dtype=torch.float32, device=logits.device
            )
            probs = flashinfer.sampling.softmax(
                logits, temp_tensor, enable_pdl=get_env_enable_pdl()
            )
    else:
        if skip_temperature:
            probs = flashinfer.sampling.softmax(logits, None, enable_pdl=get_env_enable_pdl())
        else:
            temp_tensor = torch.full(
                (batch_size,), temperature, dtype=torch.float32, device=logits.device
            )
            probs = flashinfer.sampling.softmax(
                logits, temp_tensor, enable_pdl=get_env_enable_pdl()
            )

    if top_p is not None and top_p < 1.0:
        top_p_tensor = torch.full((batch_size,), top_p, dtype=torch.float32, device=logits.device)
        probs = flashinfer.sampling.top_p_renorm_probs(probs, top_p_tensor)

    return probs


def sample_from_probs(
    probs: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    """Sample tokens from a pre-computed probability distribution."""
    if not IS_FLASHINFER_AVAILABLE:
        raise RuntimeError("FlashInfer is required for backend_flashinfer.sample_from_probs")
    return flashinfer.sampling.sampling_from_probs(
        probs,
        deterministic=True,
        generator=generator,
        seed=seed,
        offset=offset,
    )


def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    skip_temperature: bool = False,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute probs then sample; returns (tokens, probs)."""
    probs = compute_probs(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        skip_temperature=skip_temperature,
    )
    tokens = sample_from_probs(probs, generator=generator, seed=seed, offset=offset)
    return tokens, probs


def greedy(
    logits: torch.Tensor,
    *,
    return_probs: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Greedy decoding; returns exact one-hot when return_probs=True."""
    tokens = torch.argmax(logits, dim=-1)
    if return_probs:
        probs = torch.zeros_like(logits)
        probs.scatter_(1, tokens.unsqueeze(-1), 1.0)
        return tokens, probs
    return tokens, None


def top_k_top_p_sampling_from_logits_op(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    return flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, seed=seed, offset=offset
    )


def sampling_from_probs_op(
    probs: torch.Tensor,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return flashinfer.sampling.sampling_from_probs(
        probs, deterministic=True, seed=seed, offset=offset
    )


def softmax_op(
    logits: torch.Tensor,
    temperature: Optional[torch.Tensor],
) -> torch.Tensor:
    return flashinfer.sampling.softmax(logits, temperature, enable_pdl=get_env_enable_pdl())


def top_k_mask_logits_op(
    logits: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    return flashinfer.sampling.top_k_mask_logits(logits, top_k)


def top_p_renorm_probs_op(
    probs: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    return flashinfer.sampling.top_p_renorm_probs(probs, top_p)


def sampling_from_probs_generator_op(
    probs: torch.Tensor,
    generator: Optional[torch.Generator],
    check_nan: bool = False,
) -> torch.Tensor:
    return flashinfer.sampling.sampling_from_probs(
        probs, deterministic=True, generator=generator, check_nan=check_nan
    )


def top_k_top_p_sampling_from_logits_with_generator_op(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    generator: Optional[torch.Generator],
    check_nan: bool = False,
) -> torch.Tensor:
    return flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits,
        top_k=top_k,
        top_p=top_p,
        filter_apply_order="top_k_first",
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
    )


def top_k_sampling_from_probs_op(
    probs: torch.Tensor,
    top_k: torch.Tensor,
    generator: Optional[torch.Generator],
    check_nan: bool = False,
) -> torch.Tensor:
    return flashinfer.sampling.top_k_sampling_from_probs(
        probs,
        top_k=top_k,
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
    )


def top_p_sampling_from_probs_op(
    probs: torch.Tensor,
    top_p: torch.Tensor,
    generator: Optional[torch.Generator],
    check_nan: bool = False,
) -> torch.Tensor:
    return flashinfer.sampling.top_p_sampling_from_probs(
        probs,
        top_p=top_p,
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
    )


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
    probs = flashinfer.sampling.softmax(logits, temperatures, enable_pdl=get_env_enable_pdl())
    if top_p is not None:
        probs = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    return probs
