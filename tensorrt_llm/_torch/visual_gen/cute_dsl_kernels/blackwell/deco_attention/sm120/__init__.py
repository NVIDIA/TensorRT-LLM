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

    workspace_size(b, s, h, device, pv, rank, bases)      # bytes of the caller's workspace for one call
    prepare(q, k, v, workspace, pv, valid_seq_lens, comp) # BF16 q, k, v [B, S, H, 128] -> the buffers of the call
    attention(buffers, o, pv, rank, ...)                  # the buffers -> BF16 o [B, S, H, 128]
    basis(buffers, pv, bases)                             # the compensation basis of the call

Kernels, one per PV format; a kernel package holds its attention, its V pass and the chain of its preparation:
    * "nvfp4" (nvfp4pv/): NVFP4 QK and NVFP4 PV; the default.
    * "fp8" (fp8pv/): NVFP4 QK and FP8 PV with V smoothing.
Shared by the kernels:
    * ptx/: the inline PTX that the family alone issues; one module for what both kernels or the preparation use, one
      per kernel.
    * prep/: the preparation passes of Q and K, the basis launch that the V passes join and the host side that the
      preparation chains share. Q' and K' have 192 channels in NVFP4 (the 128 Hadamard-rotated channels, then the
      r_q channels of the Q-residual term, the r_k of the K-residual term and zero blocks up to 64), with their
      UE4M3 block scales and the score scale.
Compensation (rank = (r_q, r_k) and the bases kind; prep/chain.py):
    * The eigensolver delivers up to 64 directions per basis. The attention kernels do not depend on the rank or
      on the bases kind.
    * The basis of a call (runtime.Basis) is FP32 [B, H, directions, 128], the leading direction last: one tensor
      of max(r_q, r_k) directions for bases "shared", the two of r_q and r_k directions for "separate".
Calls:
    * prepare and attention also take valid_seq_lens: int32 [B] on the device, read in place; None: all elements full.
    * The buffers of a call (the operands and the scratch of its launches) are views into the caller's workspace, laid
      out by the kernel package. A call writes every byte it reads: the workspace may hold anything.
    * Both calls enqueue their launches on the current stream. The rule of a CUDA graph capture is that of
      deco_attention.api (module docstring), which also checks the arguments of the callers. What a kernel of
      this family compiles for stands beside its key: prep/chain.py (_compile) for a preparation chain, the
      launch.py of a kernel package for its attention.
"""

import functools

import torch

from ..constants import COMP_CHANNELS, KEY_TILE
from ..inputs import Compensation
from ..runtime import Basis, Layout, carve, check_caller_basis, num_sms, workspace_bytes
from . import fp8pv, nvfp4pv
from .prep import chain

__all__ = [
    "DEFAULT_PV",
    "attention",
    "basis",
    "max_comp_channels",
    "min_seq_len",
    "prepare",
    "pv_formats",
    "workspace_size",
]

# PV format -> the package of its kernel
_KERNELS = {"fp8": fp8pv, "nvfp4": nvfp4pv}
DEFAULT_PV = "nvfp4"


def pv_formats() -> tuple[str, ...]:
    """Return the PV formats of the kernels of this family."""
    return tuple(_KERNELS)


def min_seq_len(pv: str) -> int:
    """Return the shortest sequence of the kernel of PV format pv: one key tile for "fp8", one token for "nvfp4"."""
    return {"fp8": KEY_TILE, "nvfp4": 1}[pv]


def max_comp_channels() -> int:
    """Return the most compensation channels r_q + r_k of this family: those of its one QK build."""
    return COMP_CHANNELS


@functools.cache
def _layout(
    pv: str, b: int, s: int, h: int, nsm: int, rank: tuple[int, int], bases: str
) -> tuple[Layout, int]:
    """Return the buffers of one call in workspace order (the layout of the kernel package of PV format pv) and
    their bytes, built once per PV format, shape, SM count, rank and bases kind."""
    layout = _KERNELS[pv].layout(b, s, h, nsm, rank, bases)
    return layout, workspace_bytes(layout)


def workspace_size(
    b: int, s: int, h: int, device: torch.device | str, pv: str, rank: tuple[int, int], bases: str
) -> int:
    """Return the bytes of the workspace of one call of this shape, rank and bases kind with the kernel of PV format
    pv on `device`."""
    return _layout(pv, b, s, h, num_sms(device), rank, bases)[1]


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    workspace: torch.Tensor,
    pv: str,
    valid_seq_lens: torch.Tensor | None,
    comp: Compensation,
) -> dict[str, torch.Tensor]:
    """Return the buffers of the call (name -> tensor, views into the workspace), filled from raw BF16 q, k, v.

    * valid_seq_lens: batch element b attends its first valid_seq_lens[b] keys; its later query rows are zero in o.
    * comp: the compensation of the call (prep.chain.run).
    * A caller's basis has the directions of the basis of these buffers, and one that overlaps the bytes of the
      call in the workspace is that basis (runtime.check_caller_basis); the kernels read it in place.
    """
    b, s, h, _ = q.shape
    kernel = _KERNELS[pv]
    layout, _ = _layout(pv, b, s, h, num_sms(q.device), comp.rank, comp.bases)
    buffers = carve(workspace, layout)
    if comp.basis is not None:
        check_caller_basis(comp.basis, chain.basis(buffers, comp.bases), workspace, layout)
    kernel.prepare(q, k, v, buffers, valid_seq_lens, comp)
    return buffers


def attention(
    buffers: dict[str, torch.Tensor],
    o: torch.Tensor,
    pv: str,
    rank: tuple[int, int],
    valid_seq_lens: torch.Tensor | None,
    rescale_threshold: float,
) -> None:
    """Write BF16 o [B, S, H, 128] from the buffers of prepare, with the lengths prepare took, on the current stream.
    rescale_threshold: one of constants.RESCALE_THRESHOLDS[pv]."""
    _KERNELS[pv].attention(buffers, o, valid_seq_lens, rescale_threshold)


def basis(buffers: dict[str, torch.Tensor], pv: str, bases: str) -> Basis:
    """Return the compensation basis that the eigensolver of prepare left in the buffers of a call: FP32
    [B, H, directions, 128] in the rotated frame, the leading direction last. Bases "shared": one tensor of
    max(r_q, r_k) directions; "separate": the two of the r_q directions of the Q-residual term and the r_k of the
    K-residual term."""
    return chain.basis(buffers, bases)
