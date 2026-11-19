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
"""DeCo-Attention for sm_120a (RTX PRO 6000), the kernel family built on the warp-level MMAs (mma.sync), one kernel
per PV format: "nvfp4" (nvfp4pv/: NVFP4 QK and NVFP4 PV, the default) and "fp8" (fp8pv/: NVFP4 QK and FP8 PV with V
smoothing).

    workspace_size(b, s, h, device, pv, rank)             # bytes of the caller's workspace for one call
    prepare(q, k, v, workspace, pv, rank, rows)           # BF16 q, k, v [B, S, H, 128] -> the buffers of the call
    attention(buffers, o, pv, rank)                       # the buffers -> BF16 o [B, S, H, 128]

What the kernels of the family share is at this level: ptx/, the inline PTX that the family alone issues (one module
for what both kernels or the preparation use, one per kernel), and prep/, the preparation passes of Q and K (Q' and
K' of 192 channels in NVFP4: the 128 Hadamard-rotated channels and the 2 x 32 compensation channels of the shared
rank-32 basis, with their UE4M3 block scales and the score scale) with the basis launch that the V passes join. A
kernel package holds its attention, its V pass and the chain of its preparation.

The buffers of a call (the operands and the scratch of its launches) are views into the caller's workspace, laid
out by the kernel package (its layout); a call writes every byte it reads, so the workspace may hold anything. Both
calls enqueue their launches on the current stream, read no device data back and allocate nothing, so they capture
into a CUDA graph. The callers' arguments are checked by deco_attention.api, not here."""

from .. import target
from ..constants import DEFAULT_RANK
from ..runtime import carve, num_sms, workspace_bytes
from . import fp8pv, nvfp4pv

__all__ = ["DEFAULT_PV", "attention", "prepare", "pv_formats", "ranks", "workspace_size"]

# PV format -> the package of its kernel
_KERNELS = {"fp8": fp8pv, "nvfp4": nvfp4pv}
DEFAULT_PV = "nvfp4"


def _guard():
    if target.kernel_family() != "sm120":
        raise target.NoKernelError("the sm120 kernels run on sm_120a only")


def pv_formats():
    """The PV formats of the kernels of this family."""
    return tuple(_KERNELS)


def ranks():
    """The compensation ranks of the kernels of this family: the default rank alone."""
    return (DEFAULT_RANK,)


def workspace_size(b, s, h, device, pv, rank):
    """Bytes of the workspace of one call of this shape with the kernel of PV format pv (one of pv_formats()) on
    `device`; rank is one of ranks(). deco_attention.api checks both."""
    _guard()
    return workspace_bytes(_KERNELS[pv].layout(b, s, h, num_sms(device)))


def prepare(q, k, v, workspace, pv, rank, rows):
    """The buffers of the call, views into the workspace (name -> tensor, the layout of the kernel package), with the
    operands filled from raw BF16 q, k, v [B, S, H, 128] on the current stream. rows: the int32 indices of the
    sampled tokens that estimate the compensation basis of Q' and K' (inputs.sample_rows)."""
    _guard()
    b, s, h, _ = q.shape
    kernel = _KERNELS[pv]
    buffers = carve(workspace, kernel.layout(b, s, h, num_sms(q.device)))
    kernel.prepare(q, k, v, rows, buffers)
    return buffers


def attention(buffers, o, pv, rank):
    """BF16 o [B, S, H, 128] from the buffers of prepare, on the current stream."""
    _guard()
    _KERNELS[pv].attention(buffers, o)
