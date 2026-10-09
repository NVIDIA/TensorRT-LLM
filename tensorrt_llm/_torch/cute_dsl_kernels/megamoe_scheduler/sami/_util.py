# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver-call wrapper and setup collectives production SAMI needs.

Two small helpers rather than two small modules.  They are unrelated to each
other, but both are leaf utilities with a single consumer set, and neither is
part of the package's public API.

``check_cuda`` deliberately does not touch torch.  That separation is not
load-bearing for import cost -- ``sami/__init__.py`` already imports ``Comm``
and ``hierarchical``, so torch is on the path for any ``sami.*`` import -- but
it does keep the driver wrapper usable from a torch-free context if the
package layout ever changes.
"""

from __future__ import annotations

from cuda.bindings import driver as cuda
import torch.distributed as dist

__all__ = ["Comm", "check_cuda"]


def check_cuda(result: object, operation: str = "CUDA") -> object:
    """Unpack a cuda-python driver result and raise on failure."""
    if isinstance(result, tuple):
        error, *values = result
    else:
        error, values = result, []
    if int(error) != int(cuda.CUresult.CUDA_SUCCESS):
        _, name = cuda.cuGetErrorName(error)
        _, description = cuda.cuGetErrorString(error)
        if isinstance(name, bytes):
            name = name.decode()
        if isinstance(description, bytes):
            description = description.decode()
        raise RuntimeError(
            f"{operation} failed: {name}: {description} (code {int(error)})"
        )
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


class Comm:
    """The rank identity and setup collectives required by SAMI."""

    def __init__(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before SAMI")
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()

    def barrier(self) -> None:
        dist.barrier()

    def bcast(self, payload: object, root: int) -> object:
        values = [payload]
        dist.broadcast_object_list(values, src=root)
        return values[0]

    def allgather(self, value: object) -> list[object]:
        values: list[object] = [None] * self.world
        dist.all_gather_object(values, value)
        return values
