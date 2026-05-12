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
"""Microbenchmarks for CUPTI Activity flush behavior.

This is intentionally independent of the Mamba replay benchmark.  It answers
questions about CUPTI overheads with a one-kernel CUDA graph:

* Does flush cost scale with completed kernel records?
* Can a background flush overlap Python execution and CUDA graph enqueue?
* If work B is already enqueued after an event following work A, can a
  non-forced flush deliver A's completed records while B is still running?
* Does CUPTI periodic flushing reduce the explicit end-of-cell flush cost?
* How much of flush time is Python callback work?

TODO: Once the raw parser path is validated in the full benchmark, test
capturing 2/4/8 benchmark iterations per CUDA graph replay.  Larger graphs
should reduce Python graph.replay() calls while keeping graph instantiation
cheap enough for sweeps.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import queue
import sys
import threading
import time
from collections.abc import Iterable
from multiprocessing import shared_memory

import torch


_LIBCUPTI_PATH = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13"
_CUPTI_SUCCESS = 0
_CUPTI_ATTR_DEVICE_BUFFER_SIZE = 0
_CUPTI_ATTR_DEVICE_BUFFER_POOL_LIMIT = 2
_CUPTI_ATTR_ZEROED_OUT_ACTIVITY_BUFFER = 5
_CUPTI_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE = 6
_CUPTI_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED = 8
_CUPTI_ERROR_MAX_LIMIT_REACHED = 12
_CUPTI_ERROR_INVALID_KIND = 21
_CUPTI_ACTIVITY_KIND_KERNEL = 3
_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10


def set_cupti_size_t_attribute(attr: int, value: int | None) -> None:
    if value is None:
        return
    libcupti = ctypes.CDLL(_LIBCUPTI_PATH)
    set_attribute = libcupti.cuptiActivitySetAttribute
    set_attribute.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    set_attribute.restype = ctypes.c_int
    value_obj = ctypes.c_size_t(value)
    size_obj = ctypes.c_size_t(ctypes.sizeof(value_obj))
    result = set_attribute(attr, ctypes.byref(size_obj), ctypes.byref(value_obj))
    if result != _CUPTI_SUCCESS:
        raise RuntimeError(f"cuptiActivitySetAttribute({attr}, {value}) failed with CUptiResult={result}")


def set_cupti_uint8_attribute(attr: int, value: bool | None) -> None:
    if value is None:
        return
    libcupti = ctypes.CDLL(_LIBCUPTI_PATH)
    set_attribute = libcupti.cuptiActivitySetAttribute
    set_attribute.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    set_attribute.restype = ctypes.c_int
    value_obj = ctypes.c_uint8(1 if value else 0)
    size_obj = ctypes.c_size_t(ctypes.sizeof(value_obj))
    result = set_attribute(attr, ctypes.byref(size_obj), ctypes.byref(value_obj))
    if result != _CUPTI_SUCCESS:
        raise RuntimeError(f"cuptiActivitySetAttribute({attr}, {int(value)}) failed with CUptiResult={result}")


class _CuptiActivityKernel11Prefix(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("kind", ctypes.c_int),
        ("cache_config", ctypes.c_uint8),
        ("shared_memory_config", ctypes.c_uint8),
        ("registers_per_thread", ctypes.c_uint16),
        ("partitioned_global_cache_requested", ctypes.c_int),
        ("partitioned_global_cache_executed", ctypes.c_int),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("completed", ctypes.c_uint64),
        ("device_id", ctypes.c_uint32),
        ("context_id", ctypes.c_uint32),
        ("stream_id", ctypes.c_uint32),
        ("grid_x", ctypes.c_int32),
        ("grid_y", ctypes.c_int32),
        ("grid_z", ctypes.c_int32),
        ("block_x", ctypes.c_int32),
        ("block_y", ctypes.c_int32),
        ("block_z", ctypes.c_int32),
        ("static_shared_memory", ctypes.c_int32),
        ("dynamic_shared_memory", ctypes.c_int32),
        ("local_memory_per_thread", ctypes.c_uint32),
        ("local_memory_total", ctypes.c_uint32),
        ("correlation_id", ctypes.c_uint32),
        ("grid_id", ctypes.c_int64),
        ("name", ctypes.c_void_p),
        ("reserved0", ctypes.c_void_p),
        ("queued", ctypes.c_uint64),
        ("submitted", ctypes.c_uint64),
        ("launch_type", ctypes.c_uint8),
        ("is_shared_memory_carveout_requested", ctypes.c_uint8),
        ("shared_memory_carveout_requested", ctypes.c_uint8),
        ("padding", ctypes.c_uint8),
        ("shared_memory_executed", ctypes.c_uint32),
        ("graph_node_id", ctypes.c_uint64),
    ]


def parse_cupti_buffer_ptr(buffer_ptr: int, valid_size: int) -> dict[str, float | int]:
    libcupti = ctypes.CDLL(_LIBCUPTI_PATH)
    get_next_record = libcupti.cuptiActivityGetNextRecord
    get_next_record.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    get_next_record.restype = ctypes.c_int
    record_ptr = ctypes.c_void_p(None)
    records = 0
    kernel_records = 0
    zero_ts = 0
    invalid_kind = 0
    min_start = None
    max_end = 0
    start_s = time.perf_counter()
    while True:
        result = get_next_record(ctypes.c_void_p(buffer_ptr), valid_size, ctypes.byref(record_ptr))
        if result == _CUPTI_SUCCESS:
            records += 1
            kind = ctypes.cast(record_ptr, ctypes.POINTER(ctypes.c_int)).contents.value
            if kind in (_CUPTI_ACTIVITY_KIND_KERNEL, _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL):
                kernel = ctypes.cast(record_ptr, ctypes.POINTER(_CuptiActivityKernel11Prefix)).contents
                if kernel.start == 0 or kernel.end == 0:
                    zero_ts += 1
                else:
                    kernel_records += 1
                    min_start = kernel.start if min_start is None else min(min_start, kernel.start)
                    max_end = max(max_end, kernel.end)
        elif result == _CUPTI_ERROR_MAX_LIMIT_REACHED:
            break
        elif result == _CUPTI_ERROR_INVALID_KIND:
            invalid_kind += 1
            break
        else:
            raise RuntimeError(f"cuptiActivityGetNextRecord failed with CUptiResult={result}")
    parse_s = time.perf_counter() - start_s
    span_us = 0.0 if min_start is None else (max_end - min_start) / 1000.0
    return {
        "records": records,
        "kernel_records": kernel_records,
        "zero_ts": zero_ts,
        "invalid_kind": invalid_kind,
        "parse_ms": 1000.0 * parse_s,
        "span_us": span_us,
    }


def parse_cupti_payload(payload: bytes) -> dict[str, float | int]:
    raw_buffer = ctypes.create_string_buffer(payload)
    return parse_cupti_buffer_ptr(ctypes.addressof(raw_buffer), len(payload))


def cupti_parser_worker(input_queue, output_queue, ready_event, zero_shm_before_ack: bool) -> None:
    ctypes.CDLL(_LIBCUPTI_PATH)
    ready_event.set()
    shared_blocks: dict[str, shared_memory.SharedMemory] = {}
    while True:
        item = input_queue.get()
        if item is None:
            break
        generation = item[0]
        try:
            transport = item[1]
            if transport == "bytes":
                result = parse_cupti_payload(item[2])
            elif transport == "shm":
                _, _, name, buffer_ptr, valid_size = item
                shm = shared_blocks.get(name)
                if shm is None:
                    shm = shared_memory.SharedMemory(name=name)
                    shared_blocks[name] = shm
                shared_char = ctypes.c_char.from_buffer(shm.buf)
                try:
                    parser_ptr = ctypes.addressof(shared_char)
                    result = parse_cupti_buffer_ptr(parser_ptr, valid_size)
                    if zero_shm_before_ack:
                        ctypes.memset(parser_ptr, 0, len(shm.buf))
                finally:
                    del shared_char
                result["buffer_ptr"] = buffer_ptr
            else:
                raise ValueError(f"unknown parser transport {transport!r}")
            result["generation"] = generation
            output_queue.put(result)
        except Exception as exc:  # pragma: no cover - diagnostic worker path
            output_queue.put({"generation": generation, "error": repr(exc)})
    for shm in shared_blocks.values():
        shm.close()


class CuptiActivityProbe:
    """Tiny CUPTI Activity wrapper with selectable callback cost."""

    def __init__(self, callback_mode: str, host_buffer_bytes: int, max_records_per_buffer: int) -> None:
        from cupti import cupti as cupti

        self._cupti = cupti
        self._callback_mode = callback_mode
        self._host_buffer_bytes = host_buffer_bytes
        self._max_records_per_buffer = max_records_per_buffer
        self._lock = threading.Lock()
        self._record_count = 0
        self._zero_ts_count = 0
        self._callback_count = 0
        self._name_count: dict[str, int] = {}
        self._records: list[tuple[int, int, int]] = []
        self._kernel_kinds = (
            cupti.ActivityKind.CONCURRENT_KERNEL,
            cupti.ActivityKind.KERNEL,
        )

        def buffer_requested() -> tuple[int, int]:
            return (self._host_buffer_bytes, self._max_records_per_buffer)

        def buffer_completed(activities) -> None:
            local_count = 0
            local_zero_ts = 0
            local_names: dict[str, int] = {}
            local_records: list[tuple[int, int, int]] = []
            for activity in activities:
                if activity.kind not in self._kernel_kinds:
                    continue
                if activity.start == 0 or activity.end == 0:
                    local_zero_ts += 1
                    continue
                local_count += 1
                if self._callback_mode in ("names", "records"):
                    name = getattr(activity, "name", "?")
                    local_names[name] = local_names.get(name, 0) + 1
                if self._callback_mode in ("numeric", "records"):
                    local_records.append((
                        int(activity.start),
                        int(activity.end),
                        int(activity.graph_node_id),
                    ))
            with self._lock:
                self._record_count += local_count
                self._zero_ts_count += local_zero_ts
                self._callback_count += 1
                for name, count in local_names.items():
                    self._name_count[name] = self._name_count.get(name, 0) + count
                self._records.extend(local_records)

        self._buffer_requested = buffer_requested
        self._buffer_completed = buffer_completed
        cupti.activity_register_callbacks(buffer_requested, buffer_completed)
        cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)

    def clear(self) -> None:
        """Force-drain stale records, then clear local counters."""
        self.flush(flag=1)
        with self._lock:
            self._record_count = 0
            self._zero_ts_count = 0
            self._callback_count = 0
            self._name_count = {}
            self._records = []

    def flush(self, flag: int) -> float:
        start_s = time.perf_counter()
        self._cupti.activity_flush_all(flag)
        return time.perf_counter() - start_s

    def set_flush_period(self, period_ms: int) -> None:
        self._cupti.activity_flush_period(period_ms)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "records": self._record_count,
                "zero_ts": self._zero_ts_count,
                "callbacks": self._callback_count,
                "unique_names": len(self._name_count),
            }


class RawCuptiActivityProbe:
    """CUPTI Activity wrapper using raw ctypes callbacks.

    Unlike cupti-python's high-level callback, this callback receives the raw
    CUPTI activity buffer pointer.  It only records the pointer and valid byte
    count, so flush does not materialize one Python object per activity record.
    """

    _request_callback_type = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    )
    _complete_callback_type = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    )

    def __init__(
        self,
        host_buffer_bytes: int,
        max_records_per_buffer: int,
        parse_process: bool,
        parse_transport: str,
        host_buffer_count: int,
        parser_zero_shm: bool,
    ) -> None:
        self._libcupti = ctypes.CDLL(_LIBCUPTI_PATH)
        self._host_buffer_bytes = host_buffer_bytes
        self._max_records_per_buffer = max_records_per_buffer
        self._parse_transport = parse_transport
        self._lock = threading.Lock()
        self._buffers: dict[int, ctypes.Array] = {}
        self._shared_buffers: dict[int, shared_memory.SharedMemory] = {}
        self._free_ptrs: list[int] = []
        self._completed: list[tuple[int, int, int]] = []
        self._request_count = 0
        self._complete_count = 0
        self._parse_process_enabled = parse_process
        self._parser_records = 0
        self._parser_kernel_records = 0
        self._parser_zero_ts = 0
        self._parser_buffers = 0
        self._parser_errors = 0
        self._parser_parse_ms = 0.0
        self._parser_span_us = 0.0
        self._mp_ctx = None
        self._parse_input_queue = None
        self._parse_output_queue = None
        self._parse_process = None
        self._generation = 0

        self._configure_functions()
        if parse_process:
            self._mp_ctx = mp.get_context("spawn")
            self._parse_input_queue = self._mp_ctx.Queue()
            self._parse_output_queue = self._mp_ctx.Queue()
            ready_event = self._mp_ctx.Event()
            self._parse_process = self._mp_ctx.Process(
                target=cupti_parser_worker,
                args=(self._parse_input_queue, self._parse_output_queue, ready_event, parser_zero_shm),
            )
            self._parse_process.start()
            if not ready_event.wait(timeout=5.0):
                raise RuntimeError("CUPTI parser process did not initialize")
            if parse_transport == "shm":
                for _ in range(host_buffer_count):
                    self._free_ptrs.append(self._allocate_shared_buffer())
        self._request_callback = self._request_callback_type(self._request_buffer)
        self._complete_callback = self._complete_callback_type(self._complete_buffer)
        self._check(self._libcupti.cuptiActivityRegisterCallbacks(
            self._request_callback,
            self._complete_callback,
        ))
        self._check(self._libcupti.cuptiActivityEnable(10))

    def _configure_functions(self) -> None:
        self._libcupti.cuptiActivityRegisterCallbacks.argtypes = [
            self._request_callback_type,
            self._complete_callback_type,
        ]
        self._libcupti.cuptiActivityRegisterCallbacks.restype = ctypes.c_int
        self._libcupti.cuptiActivityEnable.argtypes = [ctypes.c_int]
        self._libcupti.cuptiActivityEnable.restype = ctypes.c_int
        self._libcupti.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
        self._libcupti.cuptiActivityFlushAll.restype = ctypes.c_int
        self._libcupti.cuptiActivityFlushPeriod.argtypes = [ctypes.c_uint32]
        self._libcupti.cuptiActivityFlushPeriod.restype = ctypes.c_int

    def _check(self, result: int) -> None:
        if result != _CUPTI_SUCCESS:
            raise RuntimeError(f"CUPTI call failed with CUptiResult={result}")

    def _allocate_buffer(self) -> int:
        raw_buffer = ctypes.create_string_buffer(self._host_buffer_bytes + 8)
        raw_ptr = ctypes.addressof(raw_buffer)
        aligned_ptr = (raw_ptr + 7) & ~7
        self._buffers[aligned_ptr] = raw_buffer
        return aligned_ptr

    def _allocate_shared_buffer(self) -> int:
        shm = shared_memory.SharedMemory(create=True, size=self._host_buffer_bytes + 8)
        shared_char = ctypes.c_char.from_buffer(shm.buf)
        try:
            raw_ptr = ctypes.addressof(shared_char)
        finally:
            del shared_char
        aligned_ptr = (raw_ptr + 7) & ~7
        if aligned_ptr != raw_ptr:
            raise RuntimeError("shared memory buffer was not 8-byte aligned")
        self._shared_buffers[aligned_ptr] = shm
        return aligned_ptr

    def _request_buffer(self, buffer, size, max_num_records) -> None:
        with self._lock:
            if self._free_ptrs:
                ptr = self._free_ptrs.pop()
            else:
                if self._parse_process_enabled and self._parse_transport == "shm":
                    ptr = self._allocate_shared_buffer()
                else:
                    ptr = self._allocate_buffer()
            self._request_count += 1
        buffer[0] = ptr
        size[0] = self._host_buffer_bytes
        max_num_records[0] = self._max_records_per_buffer

    def _complete_buffer(self, context, stream_id, buffer, size, valid_size) -> None:
        del context, stream_id
        buffer_ptr = int(buffer)
        valid_size_int = int(valid_size)
        size_int = int(size)
        with self._lock:
            generation = self._generation
        if self._parse_process_enabled:
            if self._parse_transport == "shm":
                shm = self._shared_buffers[buffer_ptr]
                self._parse_input_queue.put((
                    generation,
                    "shm",
                    shm.name,
                    buffer_ptr,
                    valid_size_int,
                ))
            else:
                payload = ctypes.string_at(buffer, valid_size_int)
                self._parse_input_queue.put((generation, "bytes", payload))
        with self._lock:
            if self._parse_process_enabled and self._parse_transport == "bytes":
                self._free_ptrs.append(buffer_ptr)
            elif self._parse_process_enabled:
                pass
            else:
                self._completed.append((buffer_ptr, valid_size_int, size_int))
            self._complete_count += 1

    def clear(self) -> None:
        self.flush(flag=1)
        self.wait_for_parser(timeout_s=1.0)
        with self._lock:
            self._free_ptrs.extend(ptr for ptr, _, _ in self._completed)
            self._completed = []
            self._request_count = 0
            self._complete_count = 0
            self._generation += 1
            self._parser_records = 0
            self._parser_kernel_records = 0
            self._parser_zero_ts = 0
            self._parser_buffers = 0
            self._parser_errors = 0
            self._parser_parse_ms = 0.0
            self._parser_span_us = 0.0

    def flush(self, flag: int) -> float:
        start_s = time.perf_counter()
        self._check(self._libcupti.cuptiActivityFlushAll(flag))
        return time.perf_counter() - start_s

    def set_flush_period(self, period_ms: int) -> None:
        self._check(self._libcupti.cuptiActivityFlushPeriod(period_ms))

    def _drain_parser_results(self) -> None:
        if not self._parse_process_enabled:
            return
        while True:
            try:
                result = self._parse_output_queue.get_nowait()
            except queue.Empty:
                break
            if result.get("generation") != self._generation:
                continue
            if "error" in result:
                self._parser_errors += 1
                continue
            buffer_ptr = result.get("buffer_ptr")
            if buffer_ptr is not None:
                with self._lock:
                    self._free_ptrs.append(int(buffer_ptr))
            self._parser_buffers += 1
            self._parser_records += int(result["records"])
            self._parser_kernel_records += int(result["kernel_records"])
            self._parser_zero_ts += int(result["zero_ts"])
            self._parser_parse_ms += float(result["parse_ms"])
            self._parser_span_us = max(self._parser_span_us, float(result["span_us"]))

    def wait_for_parser(self, timeout_s: float = 0.5) -> None:
        if not self._parse_process_enabled:
            return
        deadline_s = time.perf_counter() + timeout_s
        while True:
            self._drain_parser_results()
            with self._lock:
                complete_count = self._complete_count
                parser_buffers = self._parser_buffers + self._parser_errors
            if parser_buffers >= complete_count or time.perf_counter() >= deadline_s:
                break
            time.sleep(0.001)

    def snapshot(self) -> dict[str, object]:
        self._drain_parser_results()
        with self._lock:
            snapshot = {
                "records": None,
                "zero_ts": None,
                "callbacks": self._complete_count,
                "requests": self._request_count,
                "valid_bytes": sum(valid_size for _, valid_size, _ in self._completed),
                "completed_buffers": len(self._completed),
                "unique_names": None,
            }
            if self._parse_process_enabled:
                snapshot.update({
                    "parser_buffers": self._parser_buffers,
                    "parser_records": self._parser_records,
                    "parser_kernel_records": self._parser_kernel_records,
                    "parser_zero_ts": self._parser_zero_ts,
                    "parser_errors": self._parser_errors,
                    "parser_parse_ms": self._parser_parse_ms,
                    "parser_span_us": self._parser_span_us,
                })
            return snapshot

    def close(self) -> None:
        if not self._parse_process_enabled:
            return
        self._parse_input_queue.put(None)
        self._parse_process.join(timeout=5.0)
        if self._parse_process.is_alive():
            self._parse_process.terminate()
            self._parse_process.join(timeout=1.0)
        for shm in self._shared_buffers.values():
            shm.close()
            shm.unlink()


def emit(case: str, **fields: object) -> None:
    print(json.dumps({"case": case, **fields}, sort_keys=True), flush=True)


def parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def make_one_kernel_graph(elements: int, kernels_per_graph: int) -> torch.cuda.CUDAGraph:
    tensor = torch.ones((elements,), device="cuda")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(kernels_per_graph):
            tensor.add_(1.0)
    torch.cuda.synchronize()
    return graph


def replay(graph: torch.cuda.CUDAGraph, count: int) -> float:
    start_s = time.perf_counter()
    for _ in range(count):
        graph.replay()
    return time.perf_counter() - start_s


def graph_setup_case(elements: int, kernels_per_graph_values: Iterable[int], replays: int, trials: int) -> None:
    warm_tensor = torch.ones((elements,), device="cuda")
    for _ in range(3):
        warm_tensor.add_(1.0)
    torch.cuda.synchronize()
    warm_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(warm_graph):
        warm_tensor.add_(1.0)
    torch.cuda.synchronize()
    warm_graph.replay()
    torch.cuda.synchronize()

    for kernels_per_graph in kernels_per_graph_values:
        capture_times_ms = []
        replay_times_ms = []
        warmup_times_ms = []
        for trial in range(trials):
            tensor = torch.ones((elements,), device="cuda")
            torch.cuda.synchronize()

            warm_start_s = time.perf_counter()
            for _ in range(3):
                for _ in range(kernels_per_graph):
                    tensor.add_(1.0)
            torch.cuda.synchronize()
            warmup_s = time.perf_counter() - warm_start_s

            capture_start_s = time.perf_counter()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(kernels_per_graph):
                    tensor.add_(1.0)
            torch.cuda.synchronize()
            capture_s = time.perf_counter() - capture_start_s

            replay_s = replay(graph, replays)
            torch.cuda.synchronize()
            capture_times_ms.append(1000.0 * capture_s)
            replay_times_ms.append(1000.0 * replay_s)
            warmup_times_ms.append(1000.0 * warmup_s)
            emit(
                "graph_setup_trial",
                kernels_per_graph=kernels_per_graph,
                trial=trial,
                warmup_ms=1000.0 * warmup_s,
                capture_instantiate_ms=1000.0 * capture_s,
                replay_count=replays,
                replay_enqueue_ms=1000.0 * replay_s,
                replay_enqueue_us_per_replay=1_000_000.0 * replay_s / replays,
                replay_enqueue_us_per_kernel=1_000_000.0 * replay_s / (replays * kernels_per_graph),
            )
        capture_sorted = sorted(capture_times_ms)
        replay_sorted = sorted(replay_times_ms)
        warmup_sorted = sorted(warmup_times_ms)
        mid = len(capture_sorted) // 2
        emit(
            "graph_setup",
            kernels_per_graph=kernels_per_graph,
            trials=trials,
            warmup_ms_median=warmup_sorted[mid],
            capture_instantiate_ms_median=capture_sorted[mid],
            capture_instantiate_ms_min=min(capture_times_ms),
            capture_instantiate_ms_max=max(capture_times_ms),
            replay_count=replays,
            replay_enqueue_ms_median=replay_sorted[mid],
            replay_enqueue_us_per_replay=1000.0 * replay_sorted[mid] / replays,
            replay_enqueue_us_per_kernel=1000.0 * replay_sorted[mid] / (replays * kernels_per_graph),
        )


def wait_for_parser_if_present(probe, timeout_s: float = 0.5) -> None:
    if hasattr(probe, "wait_for_parser"):
        probe.wait_for_parser(timeout_s=timeout_s)


def completed_scaling(
    probe: CuptiActivityProbe,
    graph: torch.cuda.CUDAGraph,
    counts: Iterable[int],
    flags: Iterable[int],
    kernels_per_graph: int,
) -> None:
    for flag in flags:
        for count in counts:
            probe.clear()
            enqueue_s = replay(graph, count)
            sync_start_s = time.perf_counter()
            torch.cuda.synchronize()
            sync_s = time.perf_counter() - sync_start_s
            flush_s = probe.flush(flag)
            wait_for_parser_if_present(probe)
            emit(
                "completed_scaling",
                count=count,
                graph_replays=count,
                kernels_per_graph=kernels_per_graph,
                expected_kernel_records=count * kernels_per_graph,
                flag=flag,
                enqueue_ms=1000.0 * enqueue_s,
                sync_ms=1000.0 * sync_s,
                flush_ms=1000.0 * flush_s,
                **probe.snapshot(),
            )


def overlap_enqueue(
    probe: CuptiActivityProbe,
    graph: torch.cuda.CUDAGraph,
    count: int,
    flag: int,
    yield_before_overlap: bool,
    kernels_per_graph: int,
) -> None:
    probe.clear()
    replay(graph, 20)
    torch.cuda.synchronize()

    baseline_enqueue_s = replay(graph, count)
    torch.cuda.synchronize()

    probe.clear()
    replay(graph, count)
    torch.cuda.synchronize()

    ready = threading.Event()
    flush_result: dict[str, float] = {}

    def flush_worker() -> None:
        ready.set()
        flush_result["flush_s"] = probe.flush(flag)

    thread = threading.Thread(target=flush_worker)
    thread.start()
    ready.wait()
    if yield_before_overlap:
        time.sleep(0)
    overlap_enqueue_s = replay(graph, count)
    thread.join()
    torch.cuda.synchronize()
    tail_flush_s = probe.flush(flag)
    wait_for_parser_if_present(probe)
    emit(
        "overlap_enqueue",
        count=count,
        graph_replays=count,
        kernels_per_graph=kernels_per_graph,
        expected_kernel_records=count * kernels_per_graph,
        flag=flag,
        baseline_enqueue_ms=1000.0 * baseline_enqueue_s,
        overlap_enqueue_ms=1000.0 * overlap_enqueue_s,
        background_flush_ms=1000.0 * flush_result["flush_s"],
        tail_flush_ms=1000.0 * tail_flush_s,
        switch_interval_ms=1000.0 * sys.getswitchinterval(),
        yield_before_overlap=yield_before_overlap,
        **probe.snapshot(),
    )


def gil_gap(probe: CuptiActivityProbe, graph: torch.cuda.CUDAGraph, count: int, flag: int) -> None:
    probe.clear()
    replay(graph, count)
    torch.cuda.synchronize()

    ready = threading.Event()
    flush_result: dict[str, float] = {}

    def flush_worker() -> None:
        ready.set()
        flush_result["flush_s"] = probe.flush(flag)

    thread = threading.Thread(target=flush_worker)
    thread.start()
    ready.wait()
    loop_start_s = time.perf_counter()
    previous_s = loop_start_s
    max_gap_s = 0.0
    iterations = 0
    while thread.is_alive():
        now_s = time.perf_counter()
        max_gap_s = max(max_gap_s, now_s - previous_s)
        previous_s = now_s
        iterations += 1
    thread.join()
    loop_s = time.perf_counter() - loop_start_s
    wait_for_parser_if_present(probe)
    emit(
        "gil_gap",
        count=count,
        flag=flag,
        background_flush_ms=1000.0 * flush_result["flush_s"],
        main_loop_ms=1000.0 * loop_s,
        max_main_thread_gap_ms=1000.0 * max_gap_s,
        main_loop_iterations=iterations,
        **probe.snapshot(),
    )


def prefix_event_flush(
    probe: CuptiActivityProbe,
    graph: torch.cuda.CUDAGraph,
    prefix_count: int,
    tail_count: int,
    flag: int,
    kernels_per_graph: int,
) -> None:
    probe.clear()
    replay(graph, 20)
    torch.cuda.synchronize()
    probe.clear()

    enqueue_start_s = time.perf_counter()
    replay(graph, prefix_count)
    event = torch.cuda.Event()
    event.record()
    replay(graph, tail_count)
    enqueue_s = time.perf_counter() - enqueue_start_s

    event_sync_start_s = time.perf_counter()
    event.synchronize()
    event_sync_s = time.perf_counter() - event_sync_start_s

    flush1_s = probe.flush(flag)
    snapshot_after_flush1 = probe.snapshot()

    tail_sync_start_s = time.perf_counter()
    torch.cuda.synchronize()
    tail_sync_s = time.perf_counter() - tail_sync_start_s

    flush2_s = probe.flush(flag)
    wait_for_parser_if_present(probe)
    emit(
        "prefix_event_flush",
        prefix_count=prefix_count,
        tail_count=tail_count,
        kernels_per_graph=kernels_per_graph,
        expected_kernel_records=(prefix_count + tail_count) * kernels_per_graph,
        flag=flag,
        enqueue_ms=1000.0 * enqueue_s,
        event_sync_ms=1000.0 * event_sync_s,
        flush1_ms=1000.0 * flush1_s,
        records_after_flush1=snapshot_after_flush1["records"],
        callbacks_after_flush1=snapshot_after_flush1["callbacks"],
        tail_sync_ms=1000.0 * tail_sync_s,
        flush2_ms=1000.0 * flush2_s,
        **probe.snapshot(),
    )


def periodic_flush(
    probe: CuptiActivityProbe,
    graph: torch.cuda.CUDAGraph,
    count: int,
    period_ms: int,
    flag: int,
    kernels_per_graph: int,
) -> None:
    probe.clear()
    probe.set_flush_period(period_ms)
    try:
        enqueue_s = replay(graph, count)
        sync_start_s = time.perf_counter()
        torch.cuda.synchronize()
        sync_s = time.perf_counter() - sync_start_s
        snapshot_before_explicit = probe.snapshot()
        flush_s = probe.flush(flag)
        wait_for_parser_if_present(probe)
        emit(
            "periodic_flush",
            count=count,
            graph_replays=count,
            kernels_per_graph=kernels_per_graph,
            expected_kernel_records=count * kernels_per_graph,
            period_ms=period_ms,
            flag=flag,
            enqueue_ms=1000.0 * enqueue_s,
            sync_ms=1000.0 * sync_s,
            records_before_explicit_flush=snapshot_before_explicit["records"],
            callbacks_before_explicit_flush=snapshot_before_explicit["callbacks"],
            explicit_flush_ms=1000.0 * flush_s,
            **probe.snapshot(),
        )
    finally:
        probe.set_flush_period(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("all", "scaling", "overlap", "gil", "prefix", "periodic", "graph-setup"),
        default="all",
    )
    parser.add_argument("--collector", choices=("python", "raw"), default="python")
    parser.add_argument(
        "--parse-process",
        action="store_true",
        help="With --collector raw, copy completed buffers to a parser process.",
    )
    parser.add_argument(
        "--parse-transport",
        choices=("bytes", "shm"),
        default="bytes",
        help="How the raw collector sends completed buffers to the parser process.",
    )
    parser.add_argument(
        "--parser-zero-shm",
        action="store_true",
        help="With --parse-transport shm, zero the shared buffer in the parser process before acking reuse.",
    )
    parser.add_argument("--callback-mode", choices=("count", "numeric", "names", "records"), default="count")
    parser.add_argument("--host-buffer-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--host-buffer-count", type=int, default=8)
    parser.add_argument("--max-records-per-buffer", type=int, default=0)
    parser.add_argument(
        "--device-buffer-size",
        type=int,
        default=None,
        help="CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, set before CUDA init.",
    )
    parser.add_argument(
        "--device-buffer-pool-limit",
        type=int,
        default=None,
        help="CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, set before CUDA init.",
    )
    parser.add_argument(
        "--device-buffer-prealloc",
        type=int,
        default=None,
        help="CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE, set before CUDA init.",
    )
    parser.add_argument(
        "--device-buffer-host-pinned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED, set before CUDA init.",
    )
    parser.add_argument(
        "--zeroed-out-host-buffer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER. Experimental with cupti-python buffers.",
    )
    parser.add_argument("--elements", type=int, default=1024 * 1024)
    parser.add_argument("--kernels-per-graph", type=int, default=1)
    parser.add_argument("--kernels-per-graph-list", default="1,2,4,8,16")
    parser.add_argument("--graph-setup-replays", type=int, default=300)
    parser.add_argument("--graph-setup-trials", type=int, default=5)
    parser.add_argument("--counts", default="10,50,100,300,1000")
    parser.add_argument("--flags", default="0,1")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--switch-interval-ms", type=float, default=None)
    parser.add_argument("--yield-before-overlap", action="store_true")
    parser.add_argument("--prefix-count", type=int, default=100)
    parser.add_argument("--tail-count", type=int, default=4000)
    parser.add_argument("--period-ms", type=int, default=1)
    args = parser.parse_args()

    if args.switch_interval_ms is not None:
        sys.setswitchinterval(args.switch_interval_ms / 1000.0)

    set_cupti_size_t_attribute(_CUPTI_ATTR_DEVICE_BUFFER_SIZE, args.device_buffer_size)
    set_cupti_size_t_attribute(_CUPTI_ATTR_DEVICE_BUFFER_POOL_LIMIT, args.device_buffer_pool_limit)
    set_cupti_size_t_attribute(_CUPTI_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE, args.device_buffer_prealloc)
    set_cupti_uint8_attribute(_CUPTI_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED, args.device_buffer_host_pinned)
    set_cupti_uint8_attribute(_CUPTI_ATTR_ZEROED_OUT_ACTIVITY_BUFFER, args.zeroed_out_host_buffer)

    torch.cuda.init()
    if args.case == "graph-setup":
        graph_setup_case(
            args.elements,
            parse_ints(args.kernels_per_graph_list),
            args.graph_setup_replays,
            args.graph_setup_trials,
        )
        return

    graph = make_one_kernel_graph(args.elements, args.kernels_per_graph)
    if args.collector == "raw":
        probe = RawCuptiActivityProbe(
            args.host_buffer_bytes,
            args.max_records_per_buffer,
            args.parse_process,
            args.parse_transport,
            args.host_buffer_count,
            args.parser_zero_shm,
        )
    else:
        probe = CuptiActivityProbe(args.callback_mode, args.host_buffer_bytes, args.max_records_per_buffer)
    counts = parse_ints(args.counts)
    flags = parse_ints(args.flags)

    try:
        if args.case in ("all", "scaling"):
            completed_scaling(probe, graph, counts, flags, args.kernels_per_graph)
        if args.case in ("all", "overlap"):
            for flag in flags:
                overlap_enqueue(
                    probe,
                    graph,
                    args.count,
                    flag,
                    args.yield_before_overlap,
                    args.kernels_per_graph,
                )
        if args.case in ("all", "gil"):
            for flag in flags:
                gil_gap(probe, graph, args.count, flag)
        if args.case in ("all", "prefix"):
            for flag in flags:
                prefix_event_flush(
                    probe,
                    graph,
                    args.prefix_count,
                    args.tail_count,
                    flag,
                    args.kernels_per_graph,
                )
        if args.case in ("all", "periodic"):
            for flag in flags:
                periodic_flush(probe, graph, args.count, args.period_ms, flag, args.kernels_per_graph)
    finally:
        if hasattr(probe, "close"):
            probe.close()


if __name__ == "__main__":
    main()
