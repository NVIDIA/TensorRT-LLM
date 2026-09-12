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

"""Request-owned device bias for the single-row-per-request sampling path."""

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned

if TYPE_CHECKING:
    from ..llm_request import LlmRequest


class CachedEmbeddingBias:
    """Own an immutable request bias and its device copy until request termination."""

    def __init__(
        self, source: torch.Tensor, device: torch.device, stream: torch.cuda.Stream
    ) -> None:
        self.source = source
        self.tensor = source.to(device=device, non_blocking=True)
        self.upload_stream = stream.cuda_stream
        self.ready = torch.cuda.Event()
        self.ready.record(stream)

    def wait_for_upload(self, stream: torch.cuda.Stream) -> None:
        if stream.cuda_stream != self.upload_stream:
            stream.wait_event(self.ready)


@triton.jit
def _add_cached_bias_kernel(
    logits,
    bias_pointers,
    vocab_size: tl.constexpr,
    row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(1)
    address = tl.load(bias_pointers + row)
    if address != 0:
        columns = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = columns < vocab_size
        bias = tl.load(address.to(tl.pointer_type(tl.float32)) + columns, mask, other=0)
        values = tl.load(logits + row * row_stride + columns, mask, other=0)
        tl.store(logits + row * row_stride + columns, values + bias, mask)


@torch.library.custom_op("trtllm::add_cached_embedding_bias_", mutates_args=("logits",))
def add_cached_embedding_bias_(
    logits: torch.Tensor, bias_pointers: torch.Tensor, biases: list[torch.Tensor]
) -> None:
    """Add FP32 biases to contiguous FP32 logits [rows, vocab] in place.

    Args:
        logits: Contiguous CUDA FP32 tensor [rows, vocab].
        bias_pointers: CUDA int64 tensor [rows]; zero skips an unbiased row.
        biases: Device tensors referenced by bias_pointers. They are explicit
            inputs so their lifetime and stream use are visible to PyTorch.
    """
    with torch.cuda.device(logits.device):
        stream = torch.cuda.current_stream(logits.device)
        # The kernel dereferences an address table: the allocator cannot infer
        # these uses from bias_pointers alone, especially across CUDA streams.
        for bias in biases:
            bias.record_stream(stream)
        _add_cached_bias_kernel[(triton.cdiv(logits.shape[1], 1024), logits.shape[0])](
            logits, bias_pointers, logits.shape[1], logits.stride(0), 1024, num_warps=4
        )


@add_cached_embedding_bias_.register_fake
def _add_cached_embedding_bias_fake(
    logits: torch.Tensor, bias_pointers: torch.Tensor, biases: list[torch.Tensor]
) -> None:
    return None


def try_apply_cached_embedding_bias(
    logits: torch.Tensor,
    requests: list["LlmRequest"],
    request_steps: torch.Tensor,
    request_beams: torch.Tensor | None = None,
) -> bool:
    """Apply the cached path, or return False without changing logits.

    Only contiguous FP32 CUDA logits with one row per request and contiguous
    FP32 CPU biases are supported. Other layouts and multi-step/beam batches
    keep the existing sampler implementation.
    """
    sources = [request.py_embedding_bias for request in requests]
    if not any(source is not None for source in sources):
        return True
    if (
        logits.device.type != "cuda"
        or logits.dtype != torch.float32
        or logits.ndim != 2
        or not logits.is_contiguous()
        or logits.shape[0] < len(requests)
        or logits.shape[1] == 0
        or request_steps.device.type != "cpu"
        or request_steps.numel() != len(requests)
        or any(step != 1 for step in request_steps.tolist())
    ):
        return False
    # CUDA Graph replay can return padded logits (e.g. 128 rows for 110
    # requests). Beam metadata disambiguates padding from multiple real rows.
    if request_beams is None:
        if logits.shape[0] != len(requests):
            return False
    elif (
        request_beams.device.type != "cpu"
        or request_beams.numel() != len(requests)
        or any(beam != 1 for beam in request_beams.tolist())
    ):
        return False
    if any(
        source is not None
        and (
            source.device.type != "cpu"
            or source.dtype != torch.float32
            or source.shape != (logits.shape[1],)
            or not source.is_contiguous()
        )
        for source in sources
    ):
        return False

    with torch.cuda.device(logits.device):
        # Request preparation and metadata copies stay outside graph capture.
        # Model CUDA Graph replay is independent of this host sampling path.
        if torch.cuda.is_current_stream_capturing():
            return False
        stream = torch.cuda.current_stream(logits.device)
        addresses: list[int] = []
        biases: list[torch.Tensor] = []
        for request, source in zip(requests, sources):
            if source is None:
                addresses.append(0)
                continue
            cache = request._py_embedding_bias_cache
            if cache is None or cache.source is not source or cache.tensor.device != logits.device:
                cache = CachedEmbeddingBias(source, logits.device, stream)
                request._py_embedding_bias_cache = cache
            cache.wait_for_upload(stream)
            addresses.append(cache.tensor.data_ptr())
            biases.append(cache.tensor)
        addresses.extend([0] * (logits.shape[0] - len(requests)))
        pointers_host = torch.tensor(
            addresses, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
        )
        pointers_cuda = pointers_host.to(device=logits.device, non_blocking=True)
        add_cached_embedding_bias_(logits, pointers_cuda, biases)
    return True
