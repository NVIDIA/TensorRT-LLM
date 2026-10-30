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
"""Memory TensorRT-LLM's C++ code allocates outside PyTorch.

Allocation sites in C++ record into a ledger by label
(``tensorrt_llm/common/nativeMemory.h``), and ``trtllm::get_native_memory_usage``
returns the totals for a device. This module says what each label is in the
report's terms. A label it does not know is still reported, under its own
name, rather than dropped into the unexplained remainder.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema import Basis, Category, Entry, Placement, Source

__all__ = ["LABELS", "NativeMemoryProvider"]

_MEASURED = "measured as the drop in free device memory across creation, device-wide"


@dataclass(frozen=True)
class _Label:
    category: Category
    owner: str
    basis: Basis
    note: str
    #: False for memory that may not show in this process's device usage;
    #: it is listed but not summed.
    counts: bool = True


LABELS: dict[str, _Label] = {
    # NCCL_RUNTIME_CONNECT=0 makes NCCL allocate connection buffers inside
    # ncclCommInitRank; NCCL reports no size.
    "nccl_communicators": _Label(
        Category.COMM_NCCL, "NCCL communicators", Basis.PHYSICAL_BACKING, _MEASURED
    ),
    "gemm_allreduce_nvls": _Label(
        Category.COMM_NVLS,
        "fused GEMM+allreduce NVLS workspace",
        Basis.PHYSICAL_BACKING,
        _MEASURED,
    ),
    "ulysses_a2a_slots": _Label(
        Category.COMM_ULYSSES,
        "Ulysses all-to-all slots",
        Basis.REQUESTED,
        "send and symmetric receive buffers; requested size",
    ),
    "moe_gemm_profiler_workspace": _Label(
        Category.NATIVE_WORKSPACE,
        "MoE GEMM profiler workspace",
        Basis.REQUESTED,
        "held while autotuning profiles MoE GEMMs; the next MoE run frees it",
    ),
    "eplb_device": _Label(
        Category.EPLB,
        "MoE load balancer, host-accessible",
        Basis.REQUESTED,
        "expert weight slots and placement buffers (GDRCopy device memory)",
    ),
    "eplb_gpu_numa": _Label(
        Category.EPLB,
        "MoE load balancer, GPU NUMA memory",
        Basis.REQUESTED,
        "GPU memory mapped as a host NUMA node; may not be in per-process device usage",
        counts=False,
    ),
    "eplb_managed": _Label(
        Category.EPLB,
        "MoE load balancer, managed memory",
        Basis.REQUESTED,
        "CUDA managed memory; resident where last touched",
        counts=False,
    ),
}


class NativeMemoryProvider:
    """One entry per label with memory held on the current device."""

    name = "native"
    category = Category.TAGGED_OTHER

    @staticmethod
    def _op():
        import torch

        try:
            return torch.ops.trtllm.get_native_memory_usage
        except (AttributeError, RuntimeError):
            return None

    def available(self) -> bool:
        return self._op() is not None

    def collect(self) -> list[Entry]:
        import torch

        op = self._op()
        if op is None:
            return []
        labels, counts, sizes = op(torch.cuda.current_device())
        entries = []
        for label, count, size in zip(labels, counts, sizes):
            if not count:
                continue
            known = LABELS.get(label)
            if known is None:
                known = _Label(Category.TAGGED_OTHER, label, Basis.REQUESTED, "native allocation")
            entries.append(
                Entry(
                    category=known.category,
                    # A measured creation that coincided with memory being
                    # freed elsewhere on the device reads as negative: no size.
                    size_bytes=size if size >= 0 else None,
                    source=Source.COMPONENT,
                    placement=Placement.NON_TORCH,
                    basis=known.basis,
                    owner=known.owner,
                    note=f"{count} allocation(s); {known.note}",
                    counts_toward_total=known.counts,
                )
            )
        return entries
