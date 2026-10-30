# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-level batching for the diffusion worker's rank-zero receive loop."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import torch
from diffusers.utils.torch_utils import randn_tensor

if TYPE_CHECKING:
    from .executor import DiffusionRequest
    from .output import PipelineOutput


@dataclass(frozen=True)
class BatchInfo:
    """Original request boundaries and RNG streams in a merged pipeline call."""

    request_ids: tuple[int, ...]
    sample_counts: tuple[int, ...]
    seeds: tuple[int, ...]


def has_unsupported_batch_quantization(module: torch.nn.Module) -> bool:
    """Detect activation quantization that requires independent requests."""
    from ..modules.linear import FP8QDQLinearMethod

    for layer in module.modules():
        method = getattr(layer, "quant_method", None)
        if getattr(method, "quantizes_nvfp4_activations", False):
            return True
        quantizes_fp8 = isinstance(method, FP8QDQLinearMethod) and not getattr(
            method, "requires_unquantized_activation", False
        )
        if quantizes_fp8 and (
            getattr(layer, "input_scale", None) is None
            or getattr(layer, "force_dynamic_quantization", False)
        ):
            return True
    return False


def request_sample_count(req: DiffusionRequest) -> int:
    """Count generated samples, including prompt lists and per-prompt multiplicity."""
    return len(req.prompt) * req.params.num_images_per_prompt


def request_batch_key(req: DiffusionRequest) -> str | None:
    """Compare all sampling parameters except the independently sampled seed.

    Reference-conditioned requests and prepared tensors require pipeline-specific
    collation and are kept as singleton calls. Unsupported extra-parameter types
    also opt out instead of being dropped from the compatibility key.
    """
    params = req.params
    if (
        params is None
        or params.seed is None
        or not -(1 << 63) <= params.seed < (1 << 64)
        or not req.prompt
        or params.num_images_per_prompt < 1
        or params.image_reference
        or params.video_reference
        or params.audio_reference
        or req.prepared_inputs
        or req.ref_handles
        or req.batch_info is not None
    ):
        return None
    try:
        return json.dumps(
            (
                params.model_dump(exclude={"seed"}),
                sorted(params.model_fields_set - {"seed"}),
            ),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None


class DynamicBatchScheduler:
    """Collect compatible FIFO requests within a bounded waiting window.

    Only rank zero owns this scheduler. A mismatching request is retained for
    the next call, preserving arrival order even for mixed workloads. A shutdown
    sentinel flushes the current batch before stopping the worker.
    """

    def __init__(
        self,
        max_batch_size: int = 1,
        max_queue_delay_ms: float = 0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_queue_delay = max_queue_delay_ms / 1000
        self._clock = clock
        self._pending: tuple[DiffusionRequest, float] | None = None
        self._stopped = False

    def next_request(
        self,
        receive: Callable[[], DiffusionRequest | None],
        poll: Callable[[float], bool],
        *,
        enabled: bool,
    ) -> DiffusionRequest | None:
        """Return one original or merged request; poll timeouts are in seconds."""
        if self._stopped:
            return None
        if self._pending is None:
            first = receive()
            received_at = self._clock()
        else:
            first, received_at = self._pending
            self._pending = None
        if first is None:
            self._stopped = True
            return None
        if not enabled or self.max_batch_size == 1:
            return first
        key = request_batch_key(first)
        if key is None:
            return first

        requests = [first]
        samples = request_sample_count(first)
        deadline = received_at + self.max_queue_delay
        while samples < self.max_batch_size:
            # A zero delay still drains requests already waiting in the socket.
            if not poll(max(0.0, deadline - self._clock())):
                break
            candidate = receive()
            if candidate is None:
                self._stopped = True
                break
            candidate_at = self._clock()
            candidate_key = request_batch_key(candidate)
            if candidate_key != key:
                self._pending = (candidate, candidate_at)
                break
            candidate_samples = request_sample_count(candidate)
            if samples + candidate_samples > self.max_batch_size:
                self._pending = (candidate, candidate_at)
                break
            requests.append(candidate)
            samples += candidate_samples

        if len(requests) == 1:
            return first
        return replace(
            first,
            prompt=[prompt for req in requests for prompt in req.prompt],
            params=first.params.model_copy(deep=True),
            prepared_inputs={},
            batch_info=BatchInfo(
                request_ids=tuple(req.request_id for req in requests),
                sample_counts=tuple(request_sample_count(req) for req in requests),
                seeds=tuple(req.params.seed for req in requests),
            ),
        )


def randn_tensor_for_batch(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None,
    device: torch.device | str,
    dtype: torch.dtype,
    batch_info: BatchInfo | None = None,
) -> torch.Tensor:
    """Sample noise with each request's original shape and RNG stream.

    Sampling one tensor per original request preserves the stream for prompt
    lists and ``num_images_per_prompt > 1``, including CUDA's shape-dependent
    random-number generation. The generator's device also preserves CPU-RNG mode.
    """
    if batch_info is None:
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    if sum(batch_info.sample_counts) != shape[0]:
        raise ValueError("Dynamic batch sample counts do not match the latent batch dimension")
    if generator is None:
        raise ValueError("Dynamic batching requires an explicitly seeded noise generator")
    return torch.cat(
        [
            randn_tensor(
                (count, *shape[1:]),
                generator=torch.Generator(device=generator.device).manual_seed(seed),
                device=device,
                dtype=dtype,
            )
            for count, seed in zip(batch_info.sample_counts, batch_info.seeds, strict=True)
        ],
        dim=0,
    )


def split_batch_output(output: PipelineOutput, batch_info: BatchInfo) -> list[PipelineOutput]:
    """Split every media tensor before IPC, retaining each request's batch axis."""
    tensors = {
        f.name: getattr(output, f.name)
        for f in fields(output)
        if isinstance(getattr(output, f.name), torch.Tensor)
    }
    total = sum(batch_info.sample_counts)
    for name, tensor in tensors.items():
        if tensor.ndim == 0 or tensor.shape[0] != total:
            raise ValueError(f"{name} batch dimension does not match dynamic batch size {total}")
    outputs = []
    offset = 0
    for count in batch_info.sample_counts:
        outputs.append(
            replace(
                output,
                **{
                    name: tensor[offset : offset + count].clone()
                    for name, tensor in tensors.items()
                },
            )
        )
        offset += count
    return outputs
