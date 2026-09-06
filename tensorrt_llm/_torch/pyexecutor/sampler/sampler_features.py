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

"""Small sampling features that do not warrant a module of their own.

Four groups, each self-contained and free of ``TorchSampler`` state:

* **Step indexing** -- ``_PackedStepIndexer`` / ``_UnpackedStepIndexer`` and
  their bases, translating between the packed layout the model emits (one row
  per generated token, variable per request) and the rectangular per-slot
  buffers the sampler scatters results into.
* **Stop criteria** -- the ``meet_*`` / ``handle_stop_criteria`` predicates
  deciding when a request is finished.
* **Logit adjustments** -- embedding bias, d2t remapping and the fused greedy
  sampling kernel.
* **Async D2H** -- ``AsyncWorkerMixin`` (used by ``TorchSampler``), its
  private side-stream copier, and the ``SamplerEvent`` that bundles the
  resulting worker futures / CUDA events for callers to await.

Anything here that outgrows a few dozen lines, or acquires per-slot state of
its own, should move to its own module -- as beam search, penalties, token bans
and top-p decay already have.
"""

import enum
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent import futures
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Iterator, Optional, cast

import torch

from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.bindings.executor import FinishReason

from ..llm_request import LlmRequest
from .sampler_common import DEFAULT_BEAM_IDX

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

__all__ = [
    "AsyncWorkerMixin",
    "SamplerEvent",
    "_PackedStepIndexer",
    "_StepIndexTranslator",
    "_StridedStepIndexTranslator",
    "_UnpackedStepIndexer",
    "apply_d2t",
    "apply_embedding_bias",
    "check_stop_words_length",
    "fast_greedy_sample_kernel",
    "handle_stop_criteria",
    "meet_max_token_stop_criteria",
    "meet_stop_token_criteria",
]


# --------------------------------------------------------------------------
# Step indexing
# --------------------------------------------------------------------------


# Helper class for _PackedStepIndexer and _UnpackedStepIndexer, facilitating the
# selection of memory locations of tokens associated with given sets of requests.
class _StepIndexTranslator(ABC):
    def __init__(
        self,
        *,
        num_steps: torch.Tensor,
        req_offsets: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        index_dtype: Optional[torch.dtype] = None,
    ):
        """Build the index.

        Arguments:
            index_dtype: torch.dtype to use for indices (defaults to torch.int32).
            num_steps (index_dtype): Number of steps/tokens for each request
            req_offsets (index_dtype): Index offset at which the data for each request starts.
                                       If not provided, it is computed using calculate_request_offsets(),
                                       which assumes dense packing.
            max_steps (int): The largest value allowed to occur in num_steps.
                             If not provided, it is computed from num_steps.
        """
        if req_offsets is None:
            req_offsets, _ = self.calculate_request_offsets(num_steps)
        if max_steps is None:
            max_steps = cast(int, num_steps.max().item())
        self._index_map, self._index_mask = self._build_index(
            req_offsets=req_offsets,
            num_steps=num_steps,
            max_steps=max_steps,
            index_dtype=(index_dtype or torch.int32),
        )

    @staticmethod
    def calculate_request_offsets(
        req_num_steps: torch.Tensor,
        pin_memory: bool = False,
    ) -> tuple[torch.Tensor, int]:
        if req_num_steps.numel():
            req_offsets = torch.cumsum(req_num_steps, 0)
            sum_steps = int(req_offsets[-1].item())
            req_offsets_rolled = torch.empty_like(req_offsets, pin_memory=pin_memory)
            req_offsets_rolled[1:] = req_offsets[:-1]
            req_offsets_rolled[0] = 0
            req_offsets = req_offsets_rolled
        else:
            req_offsets = torch.empty_like(req_num_steps, pin_memory=pin_memory)
            sum_steps = 0
        return req_offsets, sum_steps

    def _build_index(
        self,
        req_offsets: torch.Tensor,
        num_steps: torch.Tensor,
        max_steps: int,
        index_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps_dim = torch.arange(max_steps, device=num_steps.device, dtype=index_dtype)
        valid_mask = steps_dim.unsqueeze(0) < num_steps.unsqueeze(-1)
        indices = self._compute_index_map(
            index_dtype=index_dtype,
            steps_dim=steps_dim,
            req_offsets=req_offsets,
        )
        # NB: steps_dim and req_offsets may have been overwritten by this point.
        return indices, valid_mask

    @abstractmethod
    def _compute_index_map(
        self,
        index_dtype: torch.dtype,
        steps_dim: torch.Tensor,
        req_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute full tensor index map.

        Should return a tensor of shape (len(num_steps), max_steps) containing the linear
        token index (index_dtype) corresponding to a given request and decoding step.
        Each row corresponds to a request (same ordering as 'req_offsets' and 'num_steps'),
        and the columns correspond to decoding steps 0, ..., num_steps[i]. Entries corresponding
        to decoding steps which are invalid for the given request are masked elsewhere within
        _StepIndexTranslator.

        This method is allowed to repurpose/overwrite 'steps_dim' and 'req_offsets'.

        Arguments:
            num_steps (index_dtype): Number of steps/tokens for each request
            req_offsets (index_dtype): Index offset at which the data for each request starts.
            steps_dim (index_dtype): arange(max_steps)
            index_dtype: torch.dtype to use for indices
        """

    def __getitem__(self, req_indices: Any) -> torch.Tensor:
        """Gather indices for a given set of requests.

        Arguments:
            req_indices: Any 1d torch-compatible indexing expression to select requests, corresponds
                         to the linear indices of the entries in 'num_steps' and 'req_offsets' (cf. __init__).
        Returns:
            Array of linear indices (index_dtype) selecting the tokens/steps associated
            with the requests identified by req_indices, in the same order as
            req_indices.
        """
        indices = self._index_map[req_indices].view(-1)
        mask = self._index_mask[req_indices].view(-1)
        # NB: Return value has dynamic shape (depends on mask nnz), which
        #     implies stream sync if CUDA is used.
        return indices[mask]


# Helper class for _PackedStepIndexer and _UnpackedStepIndexer, facilitating the
# selection of memory locations of tokens associated with given sets of requests,
# for memory layouts that can be parametrized via request offsets and step stride.
class _StridedStepIndexTranslator(_StepIndexTranslator):
    def __init__(
        self,
        *,
        num_steps: torch.Tensor,
        req_offsets: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        index_dtype: Optional[torch.dtype] = None,
        step_stride: Optional[int] = None,
    ):
        """Build the index.

        Allows to specify a custom stride for steps dimension.

        Arguments:
            index_dtype: torch.dtype to use for indices (defaults to torch.int32).
            num_steps (index_dtype): Number of steps/tokens for each request
            req_offsets (index_dtype): Index offset at which the data for each request starts.
                                       If not provided, it is computed using calculate_request_offsets(),
                                       assuming dense packing of tokens (grouped by request). Overriding
                                       this also allows for "request major" indexing into rectangular
                                       tensors.
            max_steps (int): The largest value allowed to occur in num_steps.
                             If not provided, it is computed from 'num_steps'.
            step_stride: Additional stride to multiply 'steps_dim' with (defaults to 1). Allows,
                         e.g., "step major" indexing into rectangular tensors.
        """
        self._step_stride = step_stride
        super().__init__(
            num_steps=num_steps,
            req_offsets=req_offsets,
            max_steps=max_steps,
            index_dtype=index_dtype,
        )

    @override
    def _compute_index_map(
        self,
        index_dtype: torch.dtype,
        steps_dim: torch.Tensor,
        req_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if self._step_stride is not None:
            steps_dim *= self._step_stride  # in-place OK
        return req_offsets.unsqueeze(-1) + steps_dim.unsqueeze(0)


# In sample_async(), each request contains a different number of output positions
# (a.k.a. 'steps') and 'logits_cuda' (and other tensors derived from it) packs those
# tokens into a single contiguous array, with the 'step' axis being the rapidly
# changing one.
#
# The class below builds an index to simplify selecting the linear indices of the
# tokens associated with a given set of requests.
#
# NB: Consider switching to torch.nested (cf. https://github.com/pytorch/pytorch/issues/80577)
class _PackedStepIndexer(_StridedStepIndexTranslator):
    def __init__(
        self,
        *,
        num_steps: torch.Tensor,
        req_offsets: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        index_dtype: Optional[torch.dtype] = None,
    ):
        """Build the index.

        Arguments:
            index_dtype: torch.dtype to use for indices (defaults to torch.int32).
            num_steps (index_dtype): Number of steps/tokens for each request
            req_offsets (index_dtype): Index offset at which the data for each request starts.
                                       If not provided, it is computed using calculate_request_offsets().
            max_steps (int): The largest value allowed to occur in num_steps.
                             If not provided, it is computed from 'num_steps'.
        """
        super().__init__(
            num_steps=num_steps,
            req_offsets=req_offsets,
            max_steps=max_steps,
            index_dtype=index_dtype,
        )


# After gathering results with _PackedStepIndexer in TorchSampler._sample_batched_by_strategy,
# they need to be scattered into result buffers in TorchSampler._unbatch_sampling_results.
# This helper class provides the translation from linear packed request + step/token indices
# to unpacked / rectangular-tensor (but still linearized) request + step/token indices.
#
# NB: Consider switching to torch.nested (cf. https://github.com/pytorch/pytorch/issues/80577)
class _UnpackedStepIndexer(_StridedStepIndexTranslator):
    class DimOrder(enum.Enum):
        SLOT_MAJOR = enum.auto()
        STEP_MAJOR = enum.auto()

    def __init__(
        self,
        *,
        seq_slots: torch.Tensor,
        num_steps: torch.Tensor,
        dim_order: DimOrder = DimOrder.SLOT_MAJOR,
        steps_dim_size: int,
        slots_dim_size: Optional[int] = None,
        index_dtype: Optional[torch.dtype] = None,
    ):
        """Build the index.

        Arguments:
            index_dtype: torch.dtype to use for indices (defaults to torch.int32).
            seq_slots (index_dtype): Request indices in unpacked tensor, enumerated in packed tensor
                                     request order.
            num_steps (index_dtype): Number of steps/tokens for each request
            dim_order: Memory layout of indexed tensor.
            steps_dim_size (int): The extent of the step dimension in the unpacked tensor.
            slots_dim_size (int): The extent of the slot dimension in the unpacked tensor.
                                  Required if dim_order is DimOrder.STEP_MAJOR.
        """
        if dim_order is self.DimOrder.SLOT_MAJOR:
            super().__init__(
                num_steps=num_steps,
                req_offsets=(steps_dim_size * seq_slots),
                max_steps=steps_dim_size,
                index_dtype=index_dtype,
            )
        elif dim_order is self.DimOrder.STEP_MAJOR:
            if slots_dim_size is None:
                raise ValueError("slots_dim_size required for step-major order")
            super().__init__(
                num_steps=num_steps,
                req_offsets=seq_slots,  # no need for stride here
                max_steps=steps_dim_size,
                index_dtype=index_dtype,
                step_stride=slots_dim_size,
            )
        else:
            raise ValueError(f"Invalid dim_order: {dim_order}")


# --------------------------------------------------------------------------
# Stop criteria
# --------------------------------------------------------------------------


def meet_max_token_stop_criteria(
    request: LlmRequest, max_seq_len: int, beam_idx: int = DEFAULT_BEAM_IDX
) -> bool:
    num_tokens = request.get_num_tokens(beam_idx)
    # Wrap in bool(): the operands come from C++ bindings (get_num_tokens,
    # py_orig_prompt_len, py_max_new_tokens) and are Any-typed, so the
    # comparison expression is inferred as Any rather than bool.
    return bool(
        (num_tokens - request.py_orig_prompt_len >= request.py_max_new_tokens)
        or (num_tokens >= max_seq_len)
    )


def meet_stop_token_criteria(
    request: LlmRequest, new_token: int, beam_idx: int = DEFAULT_BEAM_IDX
) -> bool:
    if request.py_stop_words_list:
        assert isinstance(request.py_stop_words_list, list), (
            "request.py_stop_words_list should be a list"
        )
        stop_words_list = request.py_stop_words_list

        # Fast path: all stop words are single tokens
        if all(len(word) == 1 for word in stop_words_list):
            return any(word[0] == new_token for word in stop_words_list)

        # Slow path: at least one multi-token stop word exists
        tokens = request.get_tokens(beam_idx)
        for stop_word in stop_words_list:
            if len(stop_word) > len(tokens):
                continue
            if tokens[-len(stop_word) :] == stop_word:
                return True
    return False


def handle_stop_criteria(
    request: LlmRequest, new_token: int, *, max_seq_len: int, beam_idx: int
) -> bool:
    """Handle stop criteria and set appropriate finish reasons and state.
    Returns True if generation should stop."""
    if new_token == request.py_end_id:
        request.finish_by(FinishReason.END_ID, beam_idx)
        return True

    if meet_max_token_stop_criteria(request, max_seq_len, beam_idx):
        request.finish_by(FinishReason.LENGTH, beam_idx)
        return True

    if meet_stop_token_criteria(request, new_token, beam_idx):
        request.finish_by(FinishReason.STOP_WORDS, beam_idx)
        return True

    return False


def check_stop_words_length(request: LlmRequest) -> bool:
    """Check if the stop words length is greater than 1"""
    # TODO: cache this on the request (e.g. as `request._py_has_multi_token_stop_words`)
    # so we don't recompute it per step from `py_stop_words_list`.
    if request.py_stop_words_list is not None:
        return any(len(word) > 1 for word in request.py_stop_words_list)
    return False


# --------------------------------------------------------------------------
# Logit adjustments
# --------------------------------------------------------------------------


def apply_d2t(tokens: torch.Tensor, model_outputs: dict[str, Any]) -> None:
    """Applies draft-to-target token translation table.

    Modifies tokens in-place.
    """
    if "d2t" in model_outputs:
        d2t = model_outputs["d2t"][tokens]
        tokens += d2t


@nvtx_range("fast_greedy_sample_kernel")
def fast_greedy_sample_kernel(
    logits_cuda: torch.Tensor,
    new_tokens_cuda: torch.Tensor,
    batch_dest_indices: torch.Tensor,
    max_beam_width: int,
    d2t: torch.Tensor | None,
) -> torch.Tensor:
    """Applies fast greedy sampling to the logits.

    Performs argmax, applies d2t translation if present, and scatters
    tokens into the output buffer. All operations are in-place.
    """
    # Simple argmax for greedy sampling
    next_tokens = torch.argmax(logits_cuda, dim=-1).to(dtype=new_tokens_cuda.dtype)

    # Apply draft-to-target token translation if present (for Eagle3)
    if d2t is not None:
        next_tokens += d2t[next_tokens]

    # Scatter tokens into output buffer
    batch_dest_indices_expanded = batch_dest_indices.unsqueeze(1).expand(-1, max_beam_width)
    next_tokens_expanded = next_tokens.unsqueeze(1).expand(-1, max_beam_width)
    new_tokens_cuda.view(-1, *new_tokens_cuda.shape[2:]).scatter_(
        0, batch_dest_indices_expanded, next_tokens_expanded
    )
    return next_tokens


def apply_embedding_bias(
    logits: torch.Tensor,
    requests: list[LlmRequest],
    request_steps: torch.Tensor,
) -> None:
    """Apply embedding bias (aka logit bias) to logits.

    Arguments:
      request_steps: Number of steps/tokens for each request.

    Modifies logits in-place.
    """
    # NB: Unfortunately, Torch provides no combination of torch.index_select (similar to
    #     torch.Tensor.gather -- allows one-to-many mapping) and addition, analogous to how
    #     torch.Tensor.scatter_add_ (and its variant torch.Tensor.index_add_ -- allows
    #     many-to-one mapping) combine addition with torch.Tensor.scatter_.
    #
    #     Notwithstanding the previous point, there are two options:
    #         (i)  materialize a permuted bias tensor with repeated consecutive rows via
    #              torch.repeat_interleave and then use torch.Tensor.index_add_ (poor write
    #              locality / risk of false sharing)
    #        (ii)  materialize the correctly ordered bias tensor via torch.index_select and then
    #              perform a masked addition (poor read locality for request batches randomly
    #              mixing uniform and heterogeneous bias tensors, i.e., mixing slices with high
    #              and low reuse).
    #     Since read-caching is expected to help in typical cases, option (ii) is implemented here.

    # Track which logits require logit bias application
    request_steps_list = request_steps.tolist()
    logits_bias_masks = [False] * logits.size(0)
    _next_bias_index = 0

    def provision_bias_index() -> int:
        nonlocal _next_bias_index
        bias_index = _next_bias_index
        _next_bias_index += 1
        return bias_index

    # Indices of unique bias tensors
    #
    # NB: hash(torch.Tensor) is equivalent to id(torch.Tensor), and does not
    #     depend on tensor contents, cf. https://github.com/pytorch/pytorch/issues/2569
    bias_to_index: dict[torch.Tensor, int] = defaultdict(provision_bias_index)

    # Source indices for bias application
    bias_gather_indices: list[int] = []

    # Collect bias information
    #
    # NB: the mask indexes rows of 'logits', not requests: a request contributes
    #     'steps' consecutive rows (> 1 under speculative decoding), so the offset
    #     has to accumulate 'steps' rather than track the request index. It must
    #     stay in step with 'bias_gather_indices', which is built in row order.
    row_offset = 0
    for req, steps in zip(requests, request_steps_list):
        req_bias = req.py_embedding_bias
        if req_bias is not None:
            for j in range(row_offset, row_offset + steps):
                logits_bias_masks[j] = True
            req_bias_index = bias_to_index[req_bias]
            bias_gather_indices.extend(repeat(req_bias_index, steps))
        row_offset += steps

    if not bias_to_index:
        return
    # NB: take the reference shape from the collected biases rather than from the
    #     loop variable: that holds the *last* request's bias, which is None
    #     whenever a biased request is followed by an unbiased one.
    bias_tensors = tuple(bias_to_index)

    bias_gather_indices_cuda = torch.tensor(
        bias_gather_indices, pin_memory=prefer_pinned(), dtype=torch.int32
    ).to(logits.device, non_blocking=True)
    logits_bias_mask_cuda = torch.tensor(
        logits_bias_masks, pin_memory=prefer_pinned(), dtype=torch.bool
    ).to(logits.device, non_blocking=True)
    biases_tensor = torch.empty(
        (len(bias_tensors), *bias_tensors[0].shape), pin_memory=prefer_pinned()
    )
    biases_tensor = torch.stack(
        bias_tensors,
        out=biases_tensor,
    )
    biases_tensor_cuda = biases_tensor.to(logits.device, non_blocking=True)

    biases_tensor_cuda = torch.index_select(biases_tensor_cuda, 0, bias_gather_indices_cuda)
    # NB: Avoiding logits[bias_scatter_indices] += biases_tensor (and torch.Tensor.scatter_add_), because it
    #     is unclear if this allows for repeated indices, cf.
    #         https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.index_put_.html#torch-tensor-index-put
    #     and thus introduces read-after-write dependencies (including possible false
    #     sharing).
    logits[logits_bias_mask_cuda] += biases_tensor_cuda


# --------------------------------------------------------------------------
# Async D2H
# --------------------------------------------------------------------------


@dataclass(kw_only=True)
class SamplerEvent:
    cuda_event: torch.cuda.Event
    # Side-stream D2H completion, synced host-side without gating the main stream.
    side_stream_event: Optional[torch.cuda.Event] = None
    worker_futures: Optional[list[futures.Future[Any]]] = None

    def synchronize(self) -> None:
        if self.worker_futures:
            futures.wait(self.worker_futures)
        self.cuda_event.synchronize()
        if self.side_stream_event is not None:
            self.side_stream_event.synchronize()


class _SideStreamCopier:
    """Batch non-blocking D2H copies onto a private side stream.

    Inside the `with` block, stage_copy_to_host(src) stages a copy and
    returns a pinned-CPU destination. commit() then issues all staged
    copies on the side stream in a single stream-context and records
    (and returns) an event after them, or None when nothing was staged.

    Caller contract: src must not be mutated on the main stream, and
    the returned host tensor must not be read, until the event has
    been synced host-side. Each copier is single-use.
    """

    def __init__(
        self,
        side_stream: torch.cuda.Stream,
        side_stream_ctx: torch.cuda.StreamContext,
    ) -> None:
        self._side_stream = side_stream
        self._side_stream_ctx = side_stream_ctx
        self._tasks: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.event: torch.cuda.Event | None = None

    def stage_copy_to_host(self, src: torch.Tensor) -> torch.Tensor:
        """Stage a non-blocking D2H copy of src and return its pinned-CPU dst.

        The copy is not issued until `commit()` runs; the returned host
        tensor is only valid after the resulting event has been synced.
        """
        dst = torch.empty_like(src, device="cpu", pin_memory=prefer_pinned())
        self._tasks.append((dst, src))
        return dst

    def commit(self) -> torch.cuda.Event | None:
        """Issue all staged copies and record (and return) an event after them, or None if none were staged."""
        if not self._tasks:
            self.event = None
            return None
        self._side_stream.wait_stream(torch.cuda.current_stream())
        with self._side_stream_ctx:
            for dst, src in self._tasks:
                dst.copy_(src, non_blocking=True)
        self._tasks.clear()
        event = torch.cuda.Event()
        event.record(self._side_stream)
        self.event = event
        return event


class AsyncWorkerMixin:
    """
    Mixin that adds the ability to fork off operations to run on a worker
    thread (particularly D2H copies). If the async worker isn't active,
    operations will seamlessly run on the main thread.

    Also owns a lazily-allocated private D2H side stream, handed out via
    _make_side_stream_copier for batched non-blocking D2H copies.
    """

    MAX_WORKERS = 1

    def _async_worker_active(self) -> bool:
        return getattr(self, "_async_worker", None) is not None

    def _async_worker_init(self, enable_async_worker: bool) -> None:
        self._enable_async_worker = enable_async_worker
        self._async_worker: futures.ThreadPoolExecutor | None = None
        self._async_worker_futures: list[futures.Future[Any]] = []
        # Private D2H side stream + cached stream context shared by all
        # speculative beam-history copiers.
        self._d2h_side_stream: torch.cuda.Stream = torch.cuda.Stream()
        self._d2h_side_stream_ctx: torch.cuda.StreamContext = torch.cuda.stream(
            self._d2h_side_stream
        )

    def async_worker_enabled(self) -> bool:
        return getattr(self, "_enable_async_worker", False)

    def async_worker_start(self) -> None:
        assert self.async_worker_enabled()
        if not self._async_worker_active():

            def _async_worker_initializer(device_id: int) -> None:
                # The current device is set per thread, so we need to set it
                # again here
                torch.cuda.set_device(device_id)
                # Submit the host copies in a separate stream to prevent the
                # blocking copies from gating subsequent async work
                torch.cuda.set_stream(torch.cuda.Stream())

            self._async_worker = futures.ThreadPoolExecutor(
                max_workers=self.MAX_WORKERS,
                initializer=_async_worker_initializer,
                initargs=(torch.cuda.current_device(),),
            )

    def async_worker_stop(self) -> None:
        assert self.async_worker_enabled()
        if self._async_worker_active():
            assert self._async_worker is not None
            self._async_worker.shutdown(wait=True)
            self._async_worker = None

    @torch.inference_mode()
    def _async_copy_to_host(
        self, copy_ready: torch.cuda.Event, dest: torch.Tensor, src: torch.Tensor
    ) -> None:
        # Make sure the async work takes place after all prior operations on
        # the primary stream. synchronize() is intentionally chosen instead of
        # wait() here; otherwise, blocking copies will stall subsequent CUDA
        # API calls on the main stream/thread
        copy_ready.synchronize()

        # Note that the omission of non_blocking=True here is intentional; Work
        # submitted to the async worker is expected to block at the end,
        # consistent with the semantics of futures
        dest.copy_(src)

    def _copy_to_host(self, src: torch.Tensor) -> torch.Tensor:
        dest = torch.empty_like(src, device="cpu", pin_memory=prefer_pinned())
        if self._async_worker_active():
            # Create a snapshot of the source on the main stream, so as to
            # guarantee that the tensor data hasn't been modified before the
            # copy. This precaution is only needed because the copy will
            # execute on a side stream and thus there is no guarantee that
            # future operations on the main stream won't race to modify the
            # tensor data before we copy it.
            src_snapshot = src.clone()

            # Record an event on the main thread/stream that we will
            # synchronize with on the worker thread/stream
            copy_ready = torch.cuda.Event()
            copy_ready.record()

            # Submit the copy to the async worker thread
            assert self._async_worker is not None
            result = self._async_worker.submit(
                self._async_copy_to_host, copy_ready, dest, src_snapshot
            )

            # Save the future, so that we can await it later
            self._async_worker_futures.append(result)
        else:
            # If the async worker is not in use, just copy as usual
            dest.copy_(src, non_blocking=True)
        return dest

    @contextmanager
    def _make_side_stream_copier(self) -> Iterator[_SideStreamCopier]:
        """Yield a fresh copier bound to the shared D2H side stream.

        Staged copies are committed on normal exit; the resulting event
        is exposed as `copier.event` (`None` if nothing was staged or if
        the `with` body raised).
        """
        copier = _SideStreamCopier(self._d2h_side_stream, self._d2h_side_stream_ctx)
        yield copier
        copier.commit()

    def _record_sampler_event(
        self, side_stream_event: torch.cuda.Event | None = None
    ) -> SamplerEvent:
        """Record a SamplerEvent on the main stream.

        side_stream_event, if given, is forwarded so SamplerEvent.synchronize
        also awaits the side-stream copies host-side.
        """
        cuda_event = torch.cuda.Event()
        cuda_event.record()

        # Transfer ownership to worker_futures and re-initialize
        if self._async_worker_active():
            worker_futures = self._async_worker_futures
            self._async_worker_futures = []
        else:
            worker_futures = None

        return SamplerEvent(
            cuda_event=cuda_event,
            side_stream_event=side_stream_event,
            worker_futures=worker_futures,
        )
