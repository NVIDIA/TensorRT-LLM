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

The flashinfer import is guarded so this module stays importable without it;
each op then raises an ImportError when called. ``TorchSampler`` checks
availability in its constructor.

Randomness is supplied either as a ``generator`` (host-side ``torch.Generator``,
for eager paths) or as stateless ``seed``/``offset`` device tensors, which are
required under CUDA graph capture. Explicit ``seed``/``offset`` take precedence.

Every op is ``@_compiler_disable``d to keep one clean Dynamo graph break per op
instead of a per-call break from tracing flashinfer's lazy JIT bootstrap.
"""

from typing import Any, Callable, Optional, TypeVar, Union, cast

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

from .vanilla import GREEDY_TEMPERATURE_THRESHOLD

_OpT = TypeVar("_OpT", bound=Callable[..., Any])


def _compiler_disable(fn: _OpT) -> _OpT:
    """``torch.compiler.disable``, typed: the torch stub is untyped and would
    fail mypy strict (untyped-decorator) if applied directly."""
    return cast(_OpT, torch.compiler.disable(fn))


if IS_FLASHINFER_AVAILABLE:
    import flashinfer.sampling
else:

    class _FlashInferUnavailable:
        """Placeholder that raises on first use instead of a bare NameError."""

        def __getattr__(self, name: str) -> Any:
            raise ImportError(
                "flashinfer is required for the FlashInfer sampling ops but is "
                "not installed; please install the version pinned in "
                "requirements.txt."
            )

    flashinfer = _FlashInferUnavailable()  # type: ignore[assignment]

SeedOrTensor = Union[int, torch.Tensor]


@_compiler_disable
def top_k_top_p_sampling_from_logits_op(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    seed: Optional[SeedOrTensor] = None,
    offset: Optional[SeedOrTensor] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    """Fused top-k + top-p sampling from pre-softmax logits.

    Randomness: pass ``generator`` (eager) or ``seed``/``offset`` (CUDA graph);
    see module docstring for the full contract.
    """
    tokens: torch.Tensor = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits,
        top_k=top_k,
        top_p=top_p,
        filter_apply_order="top_k_first",
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
        seed=seed,
        offset=offset,
    )
    return tokens


@_compiler_disable
def top_k_top_p_sampling_from_probs_op(
    probs: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    seed: Optional[SeedOrTensor] = None,
    offset: Optional[SeedOrTensor] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    """Fused top-k + top-p filtering and sampling from a probability distribution.

    ``filter_apply_order="top_k_first"`` matches the renorm pipeline (top-k, then
    top-p over the renormalized survivors). Sampling straight from probs skips
    the filtered full-vocab tensor that separate renorm + sample would
    materialize, so this is the cheapest way to terminate a renorm chain --
    at the cost that no filter may run after it.
    Randomness: pass ``generator`` (eager) or ``seed``/``offset`` (CUDA graph);
    see module docstring for the full contract.
    """
    tokens: torch.Tensor = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs,
        top_k=top_k,
        top_p=top_p,
        filter_apply_order="top_k_first",
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
        seed=seed,
        offset=offset,
    )
    return tokens


@_compiler_disable
def sampling_from_probs_op(
    probs: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    seed: Optional[SeedOrTensor] = None,
    offset: Optional[SeedOrTensor] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    """Categorical sampling from probabilities.

    Randomness: pass ``generator`` (eager) or ``seed``/``offset`` (CUDA graph);
    see module docstring for the full contract.
    """
    tokens: torch.Tensor = flashinfer.sampling.sampling_from_probs(
        probs,
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
        seed=seed,
        offset=offset,
    )
    return tokens


@_compiler_disable
def top_k_sampling_from_probs_op(
    probs: torch.Tensor,
    top_k: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    seed: Optional[SeedOrTensor] = None,
    offset: Optional[SeedOrTensor] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    """Top-k filtered sampling from probabilities.

    Randomness: pass ``generator`` (eager) or ``seed``/``offset`` (CUDA graph);
    see module docstring for the full contract.
    """
    tokens: torch.Tensor = flashinfer.sampling.top_k_sampling_from_probs(
        probs,
        top_k=top_k,
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
        seed=seed,
        offset=offset,
    )
    return tokens


@_compiler_disable
def top_p_sampling_from_probs_op(
    probs: torch.Tensor,
    top_p: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    seed: Optional[SeedOrTensor] = None,
    offset: Optional[SeedOrTensor] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    """Top-p filtered sampling from probabilities.

    Randomness: pass ``generator`` (eager) or ``seed``/``offset`` (CUDA graph);
    see module docstring for the full contract.
    """
    tokens: torch.Tensor = flashinfer.sampling.top_p_sampling_from_probs(
        probs,
        top_p=top_p,
        deterministic=True,
        check_nan=check_nan,
        generator=generator,
        seed=seed,
        offset=offset,
    )
    return tokens


# The four ops below wrap the mask -> softmax -> renorm pipeline stages 1:1.
# The wrappers exist so callers stay importable without flashinfer installed
# (the flashinfer import above is guarded); softmax_op additionally centralizes
# the PDL env decision.


@_compiler_disable
def softmax_op(
    logits: torch.Tensor,
    temperature: Optional[torch.Tensor],
) -> torch.Tensor:
    probs: torch.Tensor = flashinfer.sampling.softmax(
        logits, temperature, enable_pdl=get_env_enable_pdl()
    )
    return probs


@_compiler_disable
def top_k_mask_logits_op(
    logits: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    masked: torch.Tensor = flashinfer.sampling.top_k_mask_logits(logits, top_k)
    return masked


@_compiler_disable
def top_k_renorm_probs_op(
    probs: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    renormed: torch.Tensor = flashinfer.sampling.top_k_renorm_probs(probs, top_k)
    return renormed


@_compiler_disable
def top_p_renorm_probs_op(
    probs: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    renormed: torch.Tensor = flashinfer.sampling.top_p_renorm_probs(probs, top_p)
    return renormed


@_compiler_disable
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


# ---------------------------------------------------------------------------
# Speculative-decoding samplers (per-request tensor params). These build on the
# flashinfer ops above, sharing only vanilla's greedy-temperature threshold;
# used by the speculative-decoding paths (one-model draft sampling, rejection
# sampling).
# ---------------------------------------------------------------------------


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
    spec-decoding call sites in interface.py. min_p is not part of this
    interface: it is unsupported on the one-model speculative path and rejected
    at request admission (``SpecSamplerBase.validate_request``).
    """
    if top_k is not None:
        top_k = sanitize_top_k(top_k, logits.shape[-1])

    return compute_probs_from_logits_op(logits, temperatures, top_k, top_p)


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
    # Greedy rows (temperature <= threshold) reduce to top_k=1 sampling: their
    # divisor is clamped to 1.0 below (order-preserving for those rows), so
    # flashinfer deterministically returns the max-probability token, i.e. the
    # argmax of the original logits. All ops remain branch-free (no
    # data-dependent control flow), so this stays CUDA-graph safe.
    is_greedy = temperatures <= GREEDY_TEMPERATURE_THRESHOLD
    top_k = torch.where(is_greedy, torch.ones_like(top_k), top_k)
    top_p = torch.where(is_greedy, torch.ones_like(top_p), top_p)
    # Dividing by a greedy row's temperature (0 / <= threshold) would blow the
    # logits up to inf/nan; clamp those rows to 1.0 to keep the division safe.
    safe_temp = torch.where(is_greedy, torch.ones_like(temperatures), temperatures)
    logits = logits.div_(safe_temp.unsqueeze(dim=1))
    return top_k_top_p_sampling_from_logits_op(logits, top_k, top_p, seed=seed, offset=offset)


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
    tokens = sampling_from_probs_op(probs, seed=seed, offset=offset)
    return tokens, probs
