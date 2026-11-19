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
"""DeCo-Attention on the tcgen05 targets (B200 sm_100a, GB300 sm_103a, Rubin sm_107a). The family has one kernel, of
PV format "fp8" (fp8pv/: NVFP4 QK and FP8 PV with V smoothing).

    workspace_size(b, s, h, device, pv, rank)             # bytes of the caller's workspace for one call
    prepare(q, k, v, workspace, pv, rank, rows)           # BF16 q, k, v [B, S, H, 128] -> the buffers of the call
    attention(buffers, o, pv, rank)                       # the buffers -> BF16 o [B, S, H, 128]

What does not depend on the PV format is at this level: ptx.py, the inline PTX that the family alone issues,
tmem.py, the tensor-memory loads and stores of register values in DSL operations, qk.py, the NVFP4 QK with both
operands in shared memory and its block-scaled layouts, and prep/, the preparation passes of Q and K (a
sign-randomized Hadamard rotation, NVFP4 codes with one UE4M3 scale per 16 channels and 2 x rank compensation
channels: the projections of the quantization residuals and of the quantized bodies on a shared basis of `rank`
directions estimated from sampled tokens, so that Q' K'^T over the 128 + 2 rank channels adds the first-order error
terms of the product back within that basis). The kernel package holds its attention, its V pass and the chain of
its preparation.

The compensation rank is 32 on every target; targets with the K = 128 block-scaled MMA also offer rank 64 (qk.ranks).

The buffers of a call (the operands and the scratch of its launches) are views into the caller's workspace, laid
out by the kernel package (its layout); a call writes every byte it reads, so the workspace may hold anything. Every
scale stays on the device, nothing reads device data back to the host and a call allocates nothing, so prepare and
attention capture into one CUDA graph. The callers' arguments are checked by deco_attention.api, not here."""

from .. import target
from ..runtime import carve, num_sms, workspace_bytes
from . import fp8pv
from .qk import ranks

__all__ = ["DEFAULT_PV", "attention", "prepare", "pv_formats", "ranks", "workspace_size"]

# PV format -> the package of its kernel
_KERNELS = {"fp8": fp8pv}
DEFAULT_PV = "fp8"


def _guard():
    if target.kernel_family() != "tcgen05":
        raise target.NoKernelError(
            "the tcgen05 kernels run on B200, GB300 and Rubin (sm_100, sm_103, sm_107) only"
        )


def pv_formats():
    """The PV formats of the kernels of this family."""
    return tuple(_KERNELS)


def workspace_size(b, s, h, device, pv, rank):
    """Bytes of the workspace of one call of this shape with the kernel of PV format pv (one of pv_formats()) at
    compensation rank `rank` (one of ranks()) on `device`; deco_attention.api checks both."""
    _guard()
    return workspace_bytes(_KERNELS[pv].layout(b, s, h, rank, num_sms(device)))


def prepare(q, k, v, workspace, pv, rank, rows):
    """The buffers of the call, views into the workspace (name -> tensor, the layout of the kernel package), with the
    operands filled from raw BF16 q, k, v [B, S, H, 128] on the current stream. rows: the int32 indices of the
    sampled tokens that estimate the compensation basis of Q' and K' (inputs.sample_rows)."""
    _guard()
    b, s, h, _ = q.shape
    kernel = _KERNELS[pv]
    buffers = carve(workspace, kernel.layout(b, s, h, rank, num_sms(q.device)))
    kernel.prepare(q, k, v, rows, buffers, rank)
    return buffers


def attention(buffers, o, pv, rank):
    """BF16 o [B, S, H, 128] from the buffers of prepare, on the current stream."""
    _guard()
    _KERNELS[pv].attention(buffers, o, rank)
