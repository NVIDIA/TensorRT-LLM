# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only native-cache observer, installed in serving and MPI worker interpreters.

The lazy import hook leaves library files untouched. It wraps the real fetch and
release methods, reading counters only at request boundaries; it never changes
indices, KV data, cache capacity, or the return values of either operation.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.attention.backends.sparse.offload import (
        SparseOffloadCacheMixin,
        SparseOffloadState,
    )

_MODULE = "tensorrt_llm._torch.attention.backends.sparse.offload"


def _observe(module: ModuleType) -> None:
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
                    starts[id(self), request_id, id(member.cache)] = {
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
        return fetch(self, state, layer_idx, topk_indices, request_indices, unit_lengths)

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
                    record["release_ns"] = time.monotonic_ns()
                    with (directory / f"rank-{os.getpid()}.jsonl").open("a") as output:
                        output.write(json.dumps(record) + "\n")
        return release(self, ids)

    manager_cls.fetch_sparse_indices = observed_fetch
    manager_cls._release_sparse_batch = observed_release


class _ObserverLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self._loader = loader

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        _observe(module)


class _ObserverFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != _MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        assert spec is not None and spec.loader is not None, "Sparse offload module not found"
        spec.loader = _ObserverLoader(spec.loader)
        return spec


def install() -> None:
    """Observe the real library when imported, without initializing MPI or CUDA early."""
    assert _MODULE not in sys.modules, "Observer must be installed before the sparse backend"
    sys.meta_path.insert(0, _ObserverFinder())
