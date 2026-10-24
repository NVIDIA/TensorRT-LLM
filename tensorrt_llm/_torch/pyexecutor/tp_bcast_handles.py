# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-storage transport for eligible local tensor-parallel requests."""

import socket
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils._pytree import tree_map

from tensorrt_llm.logger import logger

from ..distributed.communicator import MPIDist
from ..shared_tensor import SharedTensorContainer
from ..shared_tensor.shared_tensor import DTYPE_MAPPING


@dataclass
class _TensorHandles:
    handles: tuple


def _map_tensors(value: Any, operation: Callable[[Any], Any]) -> Any:
    return tree_map(
        operation, value, is_leaf=lambda item: isinstance(item, (_TensorHandles, torch.Size))
    )


def _retire(container: SharedTensorContainer) -> None:
    """Retire one exported reference that no receiver reconstructed."""
    handle = container.tensor_handle
    if container.method_key == 2:
        handle[1]._shared_decref()
    else:
        torch.UntypedStorage._release_ipc_counter(handle[11], handle[12], device=handle[6])


class TpBroadcastHandles:
    def __init__(self, dist: MPIDist) -> None:
        self.dist = dist
        self.same_host: bool | None = None

    def broadcast(self, requests: list, py_objects: tuple | None) -> tuple:
        dist = self.dist
        if self.same_host is None:
            self.same_host = len(set(dist.allgather(socket.gethostname()))) == 1
            if dist.rank == 0:
                logger.info(
                    f"Multimodal TP handle broadcast: same_host={self.same_host}, "
                    f"ranks={dist.world_size}"
                )
        if not self.same_host:
            return dist.broadcast((requests, py_objects), root=0)

        pending, owners = [], []
        wire, error = py_objects, None

        def export(value: Any) -> Any:
            if not isinstance(value, torch.Tensor):
                return value
            # An imported device allocation cannot itself be re-exported.
            owner = (
                torch.empty_strided(
                    value.size(), value.stride(), dtype=value.dtype, device=value.device
                ).copy_(value)
                if value.is_cuda
                else value
            )
            owners.append(owner)
            handles = []
            for _ in range(1, dist.world_size):
                container = SharedTensorContainer.from_tensor(owner)
                if owner.is_cuda:
                    pending.append(container)
                handle = container.dump_to_dict()
                if not owner.is_cuda:
                    pending.append(container)
                handles.append(handle)
            return _TensorHandles(tuple(handles))

        if dist.rank == 0:
            try:
                supported = True

                def check(value: Any) -> Any:
                    nonlocal supported
                    if isinstance(value, torch.Tensor):
                        supported &= (
                            value.layout == torch.strided
                            and value.device.type in ("cpu", "cuda")
                            and value.dtype in DTYPE_MAPPING.values()
                            and value.numel() > 0
                            and not value.requires_grad
                            and not value.is_neg()
                            and (not value.is_cuda or torch._debug_has_internal_overlap(value) != 1)
                        )
                    return value

                for name, value in py_objects:
                    if name == "py_multimodal_data":
                        _map_tensors(value, check)
                if supported:
                    wire = tuple(
                        (
                            name,
                            _map_tensors(value, export) if name == "py_multimodal_data" else value,
                        )
                        for name, value in py_objects
                    )
            except Exception as exc:
                # Every rank must receive export errors before attempting to map.
                error = repr(exc)
                for container in pending:
                    _retire(container)
                pending.clear()
                wire = None

        try:
            result, wire, has_handles, error = dist.broadcast(
                (requests, wire, bool(pending), error), root=0
            )
        except RuntimeError as exc:
            # A serialization-failure header precedes all payload bytes and mappings.
            if dist.rank == 0 and str(exc).startswith("Serialization failed:"):
                for container in pending:
                    _retire(container)
            raise
        if error:
            raise RuntimeError("Multimodal handle export failed: " + error)
        if not has_handles:
            return result, wire

        consumed, errors = [], []

        def restore(value: Any) -> Any:
            if not isinstance(value, _TensorHandles):
                return value
            consumed.append(False)
            try:
                container = SharedTensorContainer.from_dict(value.handles[dist.rank - 1])
                # CPU from_dict opens storage and consumes its exported reference.
                # CUDA consumes its reference later, when get_local_view opens IPC.
                consumed[-1] = container.method_key == 2
                tensor = container.get_local_view()
                consumed[-1] = True
                return tensor
            except Exception as exc:
                # Visit every handle, even when an earlier mapping failed.
                errors.append(repr(exc))
                return None

        if dist.rank != 0:
            wire = tuple(
                (name, _map_tensors(value, restore) if name == "py_multimodal_data" else value)
                for name, value in wire
            )
        acknowledgements = dist.allgather((consumed, errors))
        if dist.rank == 0:
            for index, container in enumerate(pending):
                peer = index % (dist.world_size - 1) + 1
                slot = index // (dist.world_size - 1)
                if not acknowledgements[peer][0][slot]:
                    _retire(container)
        failures = [message for _, messages in acknowledgements for message in messages]
        if failures:
            raise RuntimeError("Multimodal handle mapping failed: " + repr(failures))
        # Storage references now belong to each request. Existing completion,
        # cancellation and error teardown drop them after the final use.
        return (requests, py_objects) if dist.rank == 0 else (result, wire)
