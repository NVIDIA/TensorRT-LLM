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
"""DeCo-Attention for sm_120a (RTX PRO 6000): the kernel family built on the warp-level MMAs (mma.sync).

    workspace_size(b, s, h, device, pv, rank)             # bytes of the caller's workspace for one call
    prepare(q, k, v, workspace, pv, rank, rows)           # BF16 q, k, v [B, S, H, 128] -> the buffers of the call
    attention(buffers, o, pv, rank)                       # the buffers -> BF16 o [B, S, H, 128]

Kernels, one per PV format; a kernel package holds its attention, its V pass and the chain of its preparation:
    * "nvfp4" (nvfp4pv/): NVFP4 QK and NVFP4 PV; the default.
    * "fp8" (fp8pv/): NVFP4 QK and FP8 PV with V smoothing.
Shared by the kernels:
    * ptx/: the inline PTX that the family alone issues; one module for what both kernels or the preparation use, one
      per kernel.
    * prep/: the preparation passes of Q and K, the basis launch that the V passes join and the host side that the
      preparation chains share. Q' and K' have 192 channels in NVFP4 (the 128 Hadamard-rotated channels and the 2 x 32
      compensation channels of the shared rank-32 basis), with their UE4M3 block scales and the score scale.
Calls:
    * prepare and attention also take valid_seq_lens: int32 [B] on the device, read in place; None: all elements full.
    * The buffers of a call (the operands and the scratch of its launches) are views into the caller's workspace, laid
      out by the kernel package. A call writes every byte it reads: the workspace may hold anything.
    * Both calls enqueue their launches on the current stream, read no device data back and allocate nothing: they
      capture into a CUDA graph. deco_attention.api checks the arguments of the callers, not this package.
"""

import torch

from ..constants import DEFAULT_RANK
from ..runtime import carve, num_sms, workspace_bytes
from . import fp8pv, nvfp4pv

__all__ = ["DEFAULT_PV", "attention", "prepare", "pv_formats", "ranks", "workspace_size"]

# PV format -> the package of its kernel
_KERNELS = {"fp8": fp8pv, "nvfp4": nvfp4pv}
DEFAULT_PV = "nvfp4"


def pv_formats() -> tuple[str, ...]:
    """Return the PV formats of the kernels of this family."""
    return tuple(_KERNELS)


def ranks() -> tuple[int, ...]:
    """Return the compensation ranks of the kernels of this family: the default rank alone."""
    return (DEFAULT_RANK,)


def workspace_size(b: int, s: int, h: int, device: torch.device | str, pv: str, rank: int) -> int:
    """Return the bytes of the workspace of one call of this shape with the kernel of PV format pv on `device`."""
    return workspace_bytes(_KERNELS[pv].layout(b, s, h, num_sms(device)))


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    workspace: torch.Tensor,
    pv: str,
    rank: int,
    rows: torch.Tensor,
    valid_seq_lens: torch.Tensor | None = None,
    map_rows: bool = False,
) -> dict[str, torch.Tensor]:
    """Return the buffers of the call (name -> tensor, views into the workspace), filled from raw BF16 q, k, v.

    Args:
        rows: Int32 indices of the sampled tokens that estimate the compensation basis (inputs.sample_rows).
        valid_seq_lens: Batch element b attends its first valid_seq_lens[b] keys; its later query rows are zero in o.
        map_rows: With lengths. True: rows is the default draw over the padded sequence, mapped by the kernel into each
            valid prefix (row r stands for token floor(r n / S) of n valid tokens). False: the caller's rows, [n] or
            [B, n], taken as given (a row at or past a length stands for the last valid token).
    """
    b, s, h, _ = q.shape
    kernel = _KERNELS[pv]
    buffers = carve(workspace, kernel.layout(b, s, h, num_sms(q.device)))
    kernel.prepare(q, k, v, rows, buffers, valid_seq_lens, map_rows)
    return buffers


def attention(
    buffers: dict[str, torch.Tensor],
    o: torch.Tensor,
    pv: str,
    rank: int,
    valid_seq_lens: torch.Tensor | None = None,
) -> None:
    """Write BF16 o [B, S, H, 128] from the buffers of prepare, with the lengths prepare took, on the current stream."""
    _KERNELS[pv].attention(buffers, o, valid_seq_lens)
