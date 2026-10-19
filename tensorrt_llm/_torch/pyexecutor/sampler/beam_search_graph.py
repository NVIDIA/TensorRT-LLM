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
"""Startup-prepared CUDA graphs owned by a single beam-search handler."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace

import torch

from tensorrt_llm.logger import logger

from .beam_search import (
    BEAM_SEARCH_PAD_TOKEN,
    BeamSearchMetadata,
    BeamSearchStore,
    CBAState,
    beam_search_sampling_batch_cba,
)

__all__ = ["BeamSearchGraphCaptureOwner", "BeamSearchGraphProfile", "BeamSearchGraphRunner"]

_BeamOutput = tuple[torch.Tensor, torch.Tensor | None]
_CAPTURE_LOCK = threading.RLock()
_EXECUTOR_LOCK = threading.RLock()
_EXECUTOR_OWNERS: weakref.WeakSet[BeamSearchGraphCaptureOwner] = weakref.WeakSet()
_WARMUP_RUNS = 3


class BeamSearchGraphCaptureOwner:
    """Register an executor before startup and retire it after worker teardown.

    Capture is admitted only for the sole live executor. Registration and
    capture share a lock, including executors that do not enable beam graphs.
    Application CUDA threads and model construction outside the executor must
    still remain quiescent during startup.
    """

    def __init__(self) -> None:
        with _EXECUTOR_LOCK:
            _EXECUTOR_OWNERS.add(self)

    @contextmanager
    def allow_capture(self) -> Iterator[bool]:
        with _EXECUTOR_LOCK:
            yield self in _EXECUTOR_OWNERS and len(_EXECUTOR_OWNERS) == 1

    def close(self) -> None:
        with _EXECUTOR_LOCK:
            _EXECUTOR_OWNERS.discard(self)


@dataclass(frozen=True, kw_only=True)
class BeamSearchGraphProfile:
    """Internal startup shapes; unknown request shapes execute eagerly.

    Widths describe successive output beam widths, starting with one input
    beam. Batch sizes count requests, not beam rows. Temperature is applied by
    BeamSearchStep before this region; tensor penalties use the eager path.
    """

    beam_widths: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    early_stopping: int = 1
    return_probs: bool = False

    def __post_init__(self) -> None:
        if not self.beam_widths or min(self.beam_widths) < 1:
            raise ValueError("Beam graph widths must be positive and nonempty")
        if not self.batch_sizes or min(self.batch_sizes) < 1:
            raise ValueError("Beam graph batch sizes must be positive and nonempty")


@dataclass(frozen=True, kw_only=True)
class _Step:
    beam_width_in: int
    beam_width_out: int
    row_stride: int | None
    temperature: float | None
    early_stopping: int
    length_penalty: torch.Tensor | float | None
    diversity_rate: torch.Tensor | float | None
    return_probs: bool


@dataclass
class _Entry:
    runs: int = 0
    failed: bool = False
    graph: torch.cuda.CUDAGraph | None = None
    logits: torch.Tensor | None = None
    slots: torch.Tensor | None = None
    lens: torch.Tensor | None = None
    metadata: BeamSearchMetadata | None = None
    output_tokens: torch.Tensor | None = None
    output_probs: torch.Tensor | None = None


def _signature(
    value: torch.Tensor | BeamSearchMetadata | CBAState | int | float | None,
) -> Hashable:
    if isinstance(value, torch.Tensor):
        return (
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
        )
    if isinstance(value, (BeamSearchMetadata, CBAState)):
        return tuple(
            (field.name, _signature(getattr(value, field.name)))
            for field in fields(value)
            if field.name not in {"seq_slots", "seq_lens"}
        )
    return value


def _state_tensors(metadata: BeamSearchMetadata) -> Iterator[torch.Tensor]:
    """Enumerate persistent state, excluding the per-step slot/length inputs."""
    for field in fields(metadata):
        if field.name in {"seq_slots", "seq_lens"}:
            continue
        value = getattr(metadata, field.name)
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, CBAState):
            for cba_field in fields(value):
                tensor = getattr(value, cba_field.name)
                if isinstance(tensor, torch.Tensor):
                    yield tensor


class BeamSearchGraphRunner:
    """Serialize a handler's sampling state and replay only startup-prepared keys.

    Capture requires an explicit process-quiescent ``allow_capture`` phase.
    Private radix workspaces isolate this runner from other graph and eager
    top-k callers. CUDA events order state updates and output clones across
    streams. A replay failure poisons the runner, since its state may already
    have been modified. The input budget excludes graph scratch allocations.
    """

    def __init__(
        self, *, max_graphs: int = 192, max_static_input_bytes: int = 512 * 1024 * 1024
    ) -> None:
        self._max_graphs = max_graphs
        self._max_static_input_bytes = max_static_input_bytes
        self._lock = threading.RLock()
        self._capture_thread: int | None = None
        self._entries: dict[Hashable, _Entry] = {}
        self._workspaces: dict[torch.device, torch.Tensor] = {}
        self._inputs: dict[tuple[torch.device, torch.dtype, int], list[torch.Tensor]] = {}
        self._pools: dict[torch.device, tuple[int, int]] = {}
        self._capture_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._last_events: dict[torch.device, tuple[int, torch.cuda.Event]] = {}
        self._replay_error: str | None = None
        self.capture_count = 0
        self.capture_failures = 0
        self.replay_count = 0

    @contextmanager
    def allow_capture(self) -> Iterator[None]:
        """Permit preparation while all unrelated CUDA producers are stopped."""
        with _CAPTURE_LOCK, self._lock:
            if self._capture_thread is not None:
                raise RuntimeError("Beam capture phases cannot be nested")
            torch.cuda.synchronize()
            self._capture_thread = threading.get_ident()
            try:
                yield
            finally:
                self._capture_thread = None
                torch.cuda.synchronize()

    def _workspace(self, device: torch.device) -> torch.Tensor:
        if device not in self._workspaces:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Beam top-k workspace must be allocated before capture")
            with torch.inference_mode(False):
                self._workspaces[device] = torch.zeros(
                    1024 * 1024, device=device, dtype=torch.uint8
                )
        return self._workspaces[device]

    def _call(
        self,
        logits: torch.Tensor,
        args: BeamSearchMetadata,
        step: _Step,
        *,
        use_workspace: bool = True,
    ) -> _BeamOutput:
        workspace = self._workspace(logits.device) if logits.is_cuda and use_workspace else None
        return beam_search_sampling_batch_cba(
            logits,
            beam_search_args=args,
            beam_width_in=step.beam_width_in,
            beam_width_out=step.beam_width_out,
            row_stride=step.row_stride,
            temperature=step.temperature,
            early_stopping=step.early_stopping,
            length_penalty=step.length_penalty,
            diversity_rate=step.diversity_rate,
            return_probs=step.return_probs,
            topk_workspace=workspace,
        )

    def _input(
        self, rows: int, vocab: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        buffers = self._inputs.setdefault((device, dtype, vocab), [])
        for buffer in buffers:
            if buffer.size(0) >= rows:
                with torch.inference_mode(False):
                    return buffer[:rows]
        allocated = sum(
            t.numel() * t.element_size() for group in self._inputs.values() for t in group
        )
        required = rows * vocab * torch.empty((), dtype=dtype).element_size()
        if allocated + required > self._max_static_input_bytes:
            raise RuntimeError("Beam graph static input budget exhausted")
        # Persistent staging can be reused from either inference or normal mode.
        with torch.inference_mode(False):
            buffer = torch.empty((rows, vocab), dtype=dtype, device=device)
        buffers.append(buffer)
        return buffer

    @staticmethod
    def _copy(entry: _Entry, logits: torch.Tensor, args: BeamSearchMetadata, step: _Step) -> None:
        assert entry.logits is not None and entry.slots is not None and entry.lens is not None
        batch, vocab = args.seq_slots.numel(), logits.size(-1)
        stride = step.row_stride or step.beam_width_in
        live = logits.view(batch, stride, vocab)[:, : step.beam_width_in, :]
        entry.logits.view(batch, step.beam_width_in, vocab).copy_(live)
        entry.slots.copy_(args.seq_slots)
        entry.lens.copy_(args.seq_lens)

    def _stage(
        self, entry: _Entry, logits: torch.Tensor, args: BeamSearchMetadata, step: _Step
    ) -> None:
        if entry.logits is None:
            entry.logits = self._input(
                args.seq_slots.numel() * step.beam_width_in,
                logits.size(-1),
                logits.dtype,
                logits.device,
            )
            with torch.inference_mode(False):
                entry.slots = torch.empty_like(args.seq_slots)
                entry.lens = torch.empty_like(args.seq_lens)
            entry.metadata = replace(args, seq_slots=entry.slots, seq_lens=entry.lens)
        self._copy(entry, logits, args, step)

    def _capture(
        self, entry: _Entry, logits: torch.Tensor, args: BeamSearchMetadata, step: _Step
    ) -> None:
        assert entry.logits is not None and entry.metadata is not None
        if logits.device not in self._pools:
            self._pools[logits.device] = torch.cuda.graph_pool_handle()
            self._capture_streams[logits.device] = torch.cuda.Stream(device=logits.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self._pools[logits.device], stream=self._capture_streams[logits.device]
        ):
            outputs = self._call(
                entry.logits, entry.metadata, replace(step, row_stride=step.beam_width_in)
            )
        entry.graph = graph
        entry.output_tokens, entry.output_probs = outputs

    @staticmethod
    def _clone_output(entry: _Entry) -> _BeamOutput:
        assert entry.output_tokens is not None
        return entry.output_tokens.clone(), (
            entry.output_probs.clone() if entry.output_probs is not None else None
        )

    def sample(
        self,
        logits: torch.Tensor,
        *,
        beam_width_in: int,
        beam_width_out: int,
        beam_search_args: BeamSearchMetadata,
        temperature: float | None,
        early_stopping: int,
        row_stride: int | None = None,
        length_penalty: torch.Tensor | float | None = None,
        diversity_rate: torch.Tensor | float | None = None,
        return_probs: bool = True,
    ) -> _BeamOutput:
        """Sample [batch * row_stride, vocabulary] logits; never capture on a miss."""
        step = _Step(
            beam_width_in=beam_width_in,
            beam_width_out=beam_width_out,
            row_stride=row_stride,
            temperature=temperature,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            diversity_rate=diversity_rate,
            return_probs=return_probs,
        )
        with _CAPTURE_LOCK, self._lock:
            if self._replay_error is not None:
                raise RuntimeError(
                    "Beam graph state is invalid after replay failure: " + self._replay_error
                )
            if not logits.is_cuda:
                return self._call(logits, beam_search_args, step)
            with torch.cuda.device(logits.device):
                if torch.cuda.is_current_stream_capturing():
                    return self._call(logits, beam_search_args, step, use_workspace=False)
                stream = torch.cuda.current_stream(logits.device)
                previous = self._last_events.get(logits.device)
                if previous is not None and previous[0] != stream.cuda_stream:
                    stream.wait_event(previous[1])
                try:
                    return self._sample(logits, beam_search_args, step)
                finally:
                    event = previous[1] if previous is not None else torch.cuda.Event()
                    try:
                        event.record(stream)
                    except RuntimeError as exc:
                        self._replay_error = str(exc)
                        raise
                    self._last_events[logits.device] = (stream.cuda_stream, event)

    def _sample(self, logits: torch.Tensor, args: BeamSearchMetadata, step: _Step) -> _BeamOutput:
        if (
            args.cba is None
            or logits.requires_grad
            or not args.seq_slots.is_cuda
            or not args.seq_lens.is_cuda
            or isinstance(step.temperature, torch.Tensor)
            or isinstance(step.length_penalty, torch.Tensor)
            or isinstance(step.diversity_rate, torch.Tensor)
        ):
            return self._call(logits, args, step)
        key = (
            tuple(logits.shape),
            logits.dtype,
            logits.device,
            args.seq_slots.size(0),
            args.seq_slots.dtype,
            args.seq_lens.dtype,
            step,
            _signature(args),
        )
        entry = self._entries.get(key)
        preparing = self._capture_thread == threading.get_ident()
        if entry is None:
            if not preparing or len(self._entries) >= self._max_graphs:
                return self._call(logits, args, step)
            entry = _Entry()
            self._entries[key] = entry
        if entry.failed:
            return self._call(logits, args, step)
        if entry.graph is None:
            if not preparing:
                return self._call(logits, args, step)
            entry.runs += 1
            try:
                self._stage(entry, logits, args, step)
            except RuntimeError as exc:
                self.capture_failures += 1
                self._entries[key] = _Entry(failed=True)
                logger.warning(f"Beam graph staging unavailable; using eager sampling: {exc}")
                return self._call(logits, args, step)
            if entry.runs <= _WARMUP_RUNS:
                # Warm exactly the buffers/layout that capture will use. Dynamo
                # also specializes on inference-tensor metadata, not just shape.
                assert entry.logits is not None and entry.metadata is not None
                return self._call(
                    entry.logits, entry.metadata, replace(step, row_stride=step.beam_width_in)
                )
            try:
                self._capture(entry, logits, args, step)
            except RuntimeError as exc:
                self.capture_failures += 1
                self._entries[key] = _Entry(failed=True)
                logger.warning(f"Beam graph capture failed; using eager sampling: {exc}")
                return self._call(logits, args, step)
            self.capture_count += 1
        else:
            self._copy(entry, logits, args, step)
        assert entry.graph is not None
        try:
            # Capture records but does not execute the state update.
            entry.graph.replay()
            result = self._clone_output(entry)
            self.replay_count += 1
            return result
        except RuntimeError as exc:
            self._replay_error = str(exc)
            raise

    def clear(self) -> None:
        """Drain submitted work before releasing graph, state and workspace references."""
        with _CAPTURE_LOCK, self._lock:
            for device in self._workspaces:
                torch.cuda.synchronize(device)
            for _, event in self._last_events.values():
                event.synchronize()
            self._entries.clear()
            self._pools.clear()
            self._capture_streams.clear()
            self._inputs.clear()
            self._workspaces.clear()
            self._last_events.clear()

    def prepare(
        self,
        store: BeamSearchStore,
        profiles: tuple[BeamSearchGraphProfile, ...],
        *,
        vocab_size: int,
        new_log_probs: torch.Tensor,
        end_ids: torch.Tensor,
        past_tokens: torch.Tensor,
    ) -> None:
        """Warm synthetic steps while preserving all persistent sampler state.

        Caller must own a process-quiescent startup phase. Profiles cover zero
        penalties; temperature is applied before this region. Uncovered shapes
        and tensor penalties execute eagerly at runtime.
        """
        capacity, max_seq_len = store.cache_indirection.shape[1:]
        max_batch = store.cache_indirection.size(0)
        for profile in profiles:
            if max(profile.beam_widths) > capacity or max(profile.batch_sizes) > max_batch:
                raise ValueError("Beam graph profile exceeds sampler capacity")
        if not profiles:
            return
        if vocab_size < 1:
            raise ValueError("Beam graph vocabulary size must be positive")
        device = store.cum_log_probs.device
        cba_fields = store.ensure_cba()
        cba = CBAState(
            **{field.name: getattr(cba_fields, field.name) for field in fields(cba_fields)},
            end_ids=end_ids,
            prompt_lens=store.prompt_lens,
            original_tokens=store.original_tokens,
            batch_dones=store.batch_dones,
            max_seq_len=max_seq_len,
            max_gen_len=2,
        )
        args = BeamSearchMetadata(
            cache_indirection=store.cache_indirection,
            cache_indirection_buffer=store.cache_indirection_buffer,
            cum_log_probs=store.cum_log_probs,
            new_log_probs=new_log_probs,
            seq_slots=torch.zeros(1, dtype=torch.int64, device=device),
            seq_lens=torch.ones(1, dtype=torch.int32, device=device),
            finished_beams=store.first_finish_reasons,
            pending_harvest=store.pending_harvest,
            predecessor_beams=store.predecessor_beams,
            beam_idx_arange=store.beam_idx_arange,
            stop_past_tokens=past_tokens,
            cba=cba,
        )
        with torch.inference_mode(), torch.cuda.device(device), self.allow_capture():
            tensors = {t.data_ptr(): t for t in _state_tensors(args)}
            saved = [(t, t.clone()) for t in tensors.values()]
            try:
                for tensor, _ in saved:
                    tensor.zero_()
                store.beam_idx_arange.copy_(
                    torch.arange(capacity, dtype=torch.int32, device=device)
                )
                store.prompt_lens.fill_(1)
                cba_fields.cba_normed_scores.fill_(-float("inf"))
                cba_fields.cba_tokens.fill_(BEAM_SEARCH_PAD_TOKEN)
                cba_fields.cba_caps.fill_(capacity)
                end_ids.fill_(-1)
                largest = max(
                    batch * width
                    for p in profiles
                    for batch in p.batch_sizes
                    for width in (1, *p.beam_widths[:-1])
                )
                if largest * vocab_size * 4 <= self._max_static_input_bytes:
                    self._input(largest, vocab_size, torch.float32, device)
                for profile in profiles:
                    for batch in profile.batch_sizes:
                        args.seq_slots = torch.arange(batch, dtype=torch.int64, device=device)
                        args.seq_lens = torch.ones(batch, dtype=torch.int32, device=device)
                        for index, (width_in, width_out) in enumerate(
                            zip((1, *profile.beam_widths[:-1]), profile.beam_widths)
                        ):
                            if len(self._entries) >= self._max_graphs or index + 1 >= max_seq_len:
                                break
                            if batch * width_in * vocab_size * 4 > self._max_static_input_bytes:
                                continue
                            args.cba = replace(cba, max_gen_len=index + 2)
                            args.seq_lens.fill_(index + 1)
                            stride = 1 if index == 0 else capacity
                            logits = torch.zeros(
                                (batch * stride, vocab_size), dtype=torch.float32, device=device
                            )
                            for _ in range(_WARMUP_RUNS + 1):
                                self.sample(
                                    logits,
                                    beam_width_in=width_in,
                                    beam_width_out=width_out,
                                    row_stride=stride,
                                    beam_search_args=args,
                                    temperature=None,
                                    early_stopping=profile.early_stopping,
                                    return_probs=profile.return_probs,
                                )
            finally:
                for tensor, saved_tensor in saved:
                    tensor.copy_(saved_tensor)
