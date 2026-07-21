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

These ops depend on flashinfer; the import is guarded so the module stays
importable without it (sampling_utils imports it unconditionally, and the
vanilla/TRTLLM sampler paths must keep working without flashinfer). Without
flashinfer, calling any op raises an ImportError with installation guidance.
Components that will invoke these ops are expected to fail fast at startup
instead of relying on that call-time error: TorchSampler enforces flashinfer
availability in its constructor.

Randomness can be supplied either way (flashinfer accepts both in one
signature; explicit ``seed``/``offset`` take precedence over ``generator``):

- ``generator``: stateful host-side ``torch.Generator``, for eager paths.
- ``seed``/``offset``: stateless device tensors, required under CUDA graph
  capture (a ``torch.Generator`` advances host-side at launch time, so its
  state would be frozen into the graph and every replay would reuse the same
  random values).

Every op is ``@_compiler_disable``d: nothing inside flashinfer is opaque to
Dynamo (its kernels sit behind a ``functools.cache``-d lazy JIT bootstrap and
its own custom-op registration is a no-op), so tracing in turns each
untraceable builtin of the bootstrap into a warn-once plus a permanent
per-call graph break. Disabling keeps one clean graph break per op and
preserves the bootstrap's cache fast path.
"""

from typing import Any, Callable, Optional, TypeVar, Union, cast

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

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
def radix_topk_op(
    values: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorted top-k via flashinfer's radix-select kernel.

    Drop-in for ``torch.topk(values, k, dim=-1, sorted=True)`` on 2D inputs:
    returns ``(values, indices)`` with values descending and int64 indices;
    like torch.topk, the index order among equal values is unspecified.
    O(n) radix select — much faster than torch.topk on large rows, but with a
    fixed per-call cost that torch.topk undercuts on small rows (see
    ``sampling_utils.topk_op`` for the size-dispatching entry point).
    """
    return flashinfer.top_k(values, k, sorted=True)


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


# The three ops below wrap the mask -> softmax -> renorm pipeline stages 1:1.
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
