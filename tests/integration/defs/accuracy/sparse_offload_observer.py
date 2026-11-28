# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only native-cache observer and deterministic decode admission.

The lazy import hook leaves library files untouched. It wraps the real fetch and
release methods, reading counters only at request boundaries; it never changes
indices, KV data, cache capacity, or the return values of either operation.
An optional scheduling gate starts concurrent parity waves with every history
ready and fixes their decode order, leaving transfers and retirement intact.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.attention.backends.sparse.offload import (
        SparseOffloadCacheMixin,
        SparseOffloadState,
    )
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

_MODULE = "tensorrt_llm._torch.attention.backends.sparse.offload"
_INTEGRITY_MODULE = "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.backend"
_EXECUTOR_MODULE = "tensorrt_llm._torch.pyexecutor.py_executor"


def _observe_scheduler(module: ModuleType) -> None:
    control_file = os.environ.get("TLLM_TEST_SPARSE_BATCH_CONTROL")
    if control_file is None:
        return
    from sparse_offload_accuracy import DecodeBatchGate

    control_path = Path(control_file)
    schedule = module.PyExecutor._schedule
    gates = {}

    @functools.wraps(schedule)
    def scheduled_wave(self: PyExecutor) -> tuple:
        result = schedule(self)
        batch = result[0]
        control = json.loads(control_path.read_text())
        if self not in gates:
            gates[self] = DecodeBatchGate()
        requests = batch.generation_requests
        selected = gates[self].select(
            wave=control["wave"],
            expected_lengths=control["prompt_lengths"],
            request_lengths=[request.py_orig_prompt_len for request in requests],
            decoding_iters=[request.py_decoding_iter for request in requests],
        )
        selected_indices = set(selected)
        held = type(batch)()
        held.generation_requests = [
            request for index, request in enumerate(requests) if index not in selected_indices
        ]
        if held.generation_requests:
            self._revert_unexecuted_alloc(held)
        batch.generation_requests = [requests[index] for index in selected]
        return result

    module.PyExecutor._schedule = scheduled_wave


def _observe(module: ModuleType, *, integrity_enabled: bool) -> None:
    import torch

    from tensorrt_llm._utils import mpi_rank

    directory = Path(os.environ["TLLM_TEST_SPARSE_OFFLOAD_EVENTS"])
    directory.mkdir(parents=True, exist_ok=True)
    manager_cls = module.SparseOffloadCacheMixin
    fetch = manager_cls.fetch_sparse_indices
    release = manager_cls._release_sparse_batch
    starts = {}

    @functools.wraps(fetch)
    def observed_fetch(
        self: SparseOffloadCacheMixin,
        state: SparseOffloadState,
        layer_idx: int,
        topk_indices: torch.Tensor,
        request_indices: torch.Tensor,
        unit_lengths: torch.Tensor,
        unit_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.batch is not None and layer_idx == min(state.layers):
            members = self._sparse_batch_members
            new_members = [
                (request_id, member)
                for request_id, member in members.items()
                if (id(self), request_id, id(member.cache)) not in starts
            ]
            if new_members:
                torch.cuda.synchronize()
                hits, misses = state.batch.peek_hit_miss_counter()
                units = unit_lengths.detach().cpu().flatten().tolist()
                for request_id, member in new_members:
                    record = {
                        "rank": mpi_rank(),
                        "request_id": request_id,
                        "first_fetch_ns": time.monotonic_ns(),
                        "history_tokens": int(member.cache.history_length),
                        "selectable_units": [int(value) for value in units],
                        "unit_tokens": int(state.layers[layer_idx].unit_tokens),
                        "index_topk": int(self._sparse_offload_topk),
                        "hits": int(hits),
                        "misses": int(misses),
                    }
                    if integrity_enabled:
                        record["offload_layers"] = sorted(state.layers)
                    starts[id(self), request_id, id(member.cache)] = record
        return fetch(
            self, state, layer_idx, topk_indices, request_indices, unit_lengths, unit_masks
        )

    @functools.wraps(release)
    def observed_release(
        self: SparseOffloadCacheMixin,
        request_ids: Iterable[int] | None = None,
    ) -> None:
        ids = list(self._sparse_batch_members if request_ids is None else request_ids)
        if self._sparse_batch is not None:
            for request_id in ids:
                member = self._sparse_batch_members.get(request_id)
                if member is None:
                    continue
                record = starts.pop((id(self), request_id, id(member.cache)), None)
                if record is not None:
                    torch.cuda.synchronize()
                    hits, misses = self._sparse_batch.peek_hit_miss_counter()
                    record["hits"] = int(hits) - record["hits"]
                    record["misses"] = int(misses) - record["misses"]
                    if integrity_enabled:
                        if TYPE_CHECKING:
                            from .sparse_offload_integrity import finish_request
                        else:
                            from sparse_offload_integrity import finish_request

                        record["integrity"] = finish_request(self, request_id, member.cache)
                    record["release_ns"] = time.monotonic_ns()
                    with (directory / f"rank-{os.getpid()}.jsonl").open("a") as output:
                        output.write(json.dumps(record) + "\n")
        return release(self, ids)

    manager_cls.fetch_sparse_indices = observed_fetch
    manager_cls._release_sparse_batch = observed_release


class _ObserverLoader(importlib.abc.Loader):
    def __init__(
        self, loader: importlib.abc.Loader, callback: Callable[[ModuleType], None]
    ) -> None:
        self._loader = loader
        self._callback = callback

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        self._callback(module)


class _ObserverFinder(importlib.abc.MetaPathFinder):
    def __init__(self, callbacks: dict[str, Callable[[ModuleType], None]]) -> None:
        self._callbacks = callbacks

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        callback = self._callbacks.get(fullname)
        if callback is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        assert spec is not None and spec.loader is not None, (
            f"Observer module not found: {fullname}"
        )
        spec.loader = _ObserverLoader(spec.loader, callback)
        return spec


def install() -> None:
    """Observe the real library when imported, without initializing MPI or CUDA early."""
    integrity_enabled = os.environ.get("TLLM_TEST_SPARSE_OFFLOAD_INTEGRITY") == "1"
    callbacks: dict[str, Callable[[ModuleType], None]] = {
        _MODULE: functools.partial(_observe, integrity_enabled=integrity_enabled),
        _EXECUTOR_MODULE: _observe_scheduler,
    }
    if integrity_enabled:
        if TYPE_CHECKING:
            from .sparse_offload_integrity import observe
        else:
            from sparse_offload_integrity import observe

        callbacks[_INTEGRITY_MODULE] = observe
    for module_name in callbacks:
        assert module_name not in sys.modules, (
            f"Observer must be installed before importing {module_name}"
        )
    sys.meta_path.insert(0, _ObserverFinder(callbacks))
