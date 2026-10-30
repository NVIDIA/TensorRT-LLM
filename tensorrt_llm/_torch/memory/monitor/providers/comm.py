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
"""Communication buffers allocated outside the PyTorch allocator.

Every kind here is invisible to ``memory_reserved()`` and so, without a
provider, lands in the unexplained residual. Most report what was
requested: the allocator rounds each request up, which the Python side
cannot see, so those figures are lower bounds.

Covered elsewhere:

* MoE all-to-all workspaces are ``MnnvlMemory``: :class:`~.mnnvl.MnnvlProvider`.
* NCCL communicators and the fused GEMM+allreduce NVLS workspace are recorded
  by the C++ code that creates them: :class:`~.native.NativeMemoryProvider`.
* Custom-allreduce IPC workspaces live in ``threading.local``, out of reach
  of a report built on another thread, so they are measured as a named
  region around their creation instead.
"""

from ..schema import Basis, Category, Entry, Placement, Source

__all__ = ["McastAllReduceProvider", "NcclWindowProvider", "SymmMemProvider"]


class SymmMemProvider:
    """``SymmetricMemoryAllReduce`` buffers, found by walking the model.

    Each instance belongs to an ``AllReduce`` submodule and there is no
    registry, so this reuses the model the weights were claimed from. torch
    symmetric memory bypasses the caching allocator: measured on torch 2.14,
    a 256 MiB ``symm_mem.empty`` raised ``memory_reserved()`` by 0 and device
    usage by 258 MiB.
    """

    name = "comm.symm_mem"
    category = Category.COMM_SYMM_MEM

    def available(self) -> bool:
        from ..tagging import claimed_modules

        return bool(claimed_modules())

    def collect(self) -> list[Entry]:
        from ..tagging import claimed_modules

        buffers: dict[int, int] = {}
        for model in claimed_modules():
            for module in model.modules():
                inst = getattr(module, "symm_mem_allreduce", None)
                buffer = getattr(inst, "buffer", None) if inst is not None else None
                if buffer is None:
                    continue
                # Deduplicated by address: modules may share an instance.
                buffers[buffer.data_ptr()] = buffer.numel() * buffer.element_size()
        if not buffers:
            return []
        return [
            Entry(
                category=Category.COMM_SYMM_MEM,
                size_bytes=sum(buffers.values()),
                source=Source.COMPONENT,
                placement=Placement.NON_TORCH,
                basis=Basis.REQUESTED,
                owner="SymmetricMemoryAllReduce",
                note=f"{len(buffers)} buffer(s); requested size, allocator rounds up",
            )
        ]


class McastAllReduceProvider:
    """MNNVL/NVLS allreduce multicast workspaces (``McastGPUBuffer``).

    Kept in class-level dicts on ``MNNVLAllReduce``: the active workspace per
    mapping, and workspaces retired by a resize but kept alive for graphs
    that still point at them. Sized from each workspace's unicast view, which
    spans the whole allocation, rather than from a buffer count restated
    here.
    """

    name = "comm.mcast_allreduce"
    category = Category.COMM_NVLS

    def _cls(self):
        try:
            from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
        except ImportError:
            return None
        return MNNVLAllReduce

    def available(self) -> bool:
        return self._cls() is not None

    def collect(self) -> list[Entry]:
        cls = self._cls()
        if cls is None:
            return []
        workspaces = list(getattr(cls, "allreduce_mnnvl_workspaces", {}).values())
        for retired in getattr(cls, "allreduce_mnnvl_retired_workspaces", {}).values():
            workspaces.extend(retired)

        total = 0
        counted = 0
        for workspace in workspaces:
            view = workspace.get("uc_buffer") if isinstance(workspace, dict) else None
            if view is None:
                continue
            total += view.numel() * view.element_size()
            counted += 1
        if not counted:
            return []
        return [
            Entry(
                category=Category.COMM_NVLS,
                size_bytes=total,
                source=Source.COMPONENT,
                placement=Placement.NON_TORCH,
                basis=Basis.REQUESTED,
                owner="MNNVLAllReduce",
                note=f"{counted} multicast workspace(s); excludes signal pad and rounding",
            )
        ]


class NcclWindowProvider:
    """NCCL window buffers, from ``NCCLWindowAllocator``.

    ``ncclMemAlloc`` maps memory through the CUDA driver, outside the
    PyTorch allocator. The C++ allocator pools these buffers per
    communicator for the life of the communicator, and reports what it holds
    on a device through ``trtllm::get_nccl_window_usage``.
    """

    name = "comm.nccl_window"
    category = Category.COMM_NCCL_WINDOW

    @staticmethod
    def _op():
        import torch

        try:
            return torch.ops.trtllm.get_nccl_window_usage
        except (AttributeError, RuntimeError):
            return None

    def available(self) -> bool:
        return self._op() is not None

    def collect(self) -> list[Entry]:
        import torch

        op = self._op()
        if op is None:
            return []
        count, total, in_use = op(torch.cuda.current_device())
        if not count:
            return []
        return [
            Entry(
                category=Category.COMM_NCCL_WINDOW,
                size_bytes=total,
                source=Source.COMPONENT,
                placement=Placement.NON_TORCH,
                basis=Basis.REQUESTED,
                owner="NCCLWindowAllocator",
                note=(
                    f"{count} buffer(s), {in_use / (1 << 20):.1f} MiB checked out; "
                    "requested size, ncclMemAlloc rounds up"
                ),
            )
        ]
