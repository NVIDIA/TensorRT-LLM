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
"""DeCo-Attention on the tcgen05 targets (B200 sm_100a, GB300 sm_103a, Rubin sm_107a).

    workspace_size(b, s, h, device, pv, rank, bases)      # bytes of the caller's workspace for one call
    prepare(q, k, v, workspace, pv, valid_seq_lens, comp) # BF16 q, k, v [B, S, H, 128] -> the buffers of the call
    attention(buffers, o, pv, rank, ...)                  # the buffers -> BF16 o [B, S, H, 128]
    basis(buffers, pv, bases)                             # the compensation basis of the call

Kernels:
    * One kernel, of PV format "fp8" (fp8pv/): NVFP4 QK and FP8 PV with V smoothing.
    * rank is the pair (r_q, r_k) of the compensation channels of the two terms, r_q + r_k at most
      max_comp_channels(): 64, and 128 on the targets with the K = 128 block-scaled MMA.
Files (what does not depend on the PV format is at this level):
    ptx.py   the inline PTX that the family alone issues
    tmem.py  tensor-memory loads and stores of register values
    qk.py    the NVFP4 QK and its block-scaled layouts
    prep/    the preparation passes of Q and K: rotation, NVFP4 codes and scales, compensation channels and basis
    fp8pv/   the kernel package: its attention, its V pass and the chain of its preparation
Calls:
    * The buffers of a call (operands and scratch of its launches) are views into the caller's workspace, laid out
      by the kernel package (its layout); a call writes every byte it reads, so the workspace may hold anything.
    * Every scale stays on the device, nothing reads device data back to the host and a call allocates nothing:
      prepare and attention capture into one CUDA graph.
    * The callers' arguments are checked by deco_attention.api, not here.
"""

import functools

import torch

from .. import target
from ..constants import COMP_CHANNELS
from ..inputs import Compensation
from ..runtime import Basis, carve, check_caller_basis, num_sms, workspace_bytes
from . import fp8pv

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
_KERNELS = {"fp8": fp8pv}
DEFAULT_PV = "fp8"


def pv_formats() -> tuple[str, ...]:
    """Return the PV formats of the kernels of this family."""
    return tuple(_KERNELS)


def min_seq_len(pv: str) -> int:
    """Return the shortest sequence of the kernel of PV format pv: one token."""
    return 1


@functools.cache
def max_comp_channels() -> int:
    """Return the most compensation channels r_q + r_k of the compile target: those of the 256-channel QK build with
    the K = 128 instruction, else those of the 192-channel build."""
    return 2 * COMP_CHANNELS if target.has_k128_fp4_mma() else COMP_CHANNELS


def workspace_size(
    b: int, s: int, h: int, device: torch.device | str, pv: str, rank: tuple[int, int], bases: str
) -> int:
    """Return the bytes of the workspace of one call of this shape, PV format, rank and bases on `device`."""
    return workspace_bytes(_KERNELS[pv].layout(b, s, h, rank, bases, num_sms(device)))


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    workspace: torch.Tensor,
    pv: str,
    valid_seq_lens: torch.Tensor | None,
    comp: Compensation,
) -> dict[str, torch.Tensor]:
    """Carve the buffers of a call from the workspace and fill the operands from raw q, k, v, on the current stream.

    * valid_seq_lens, comp: the valid prefix lengths and the compensation of the call (fp8pv.prep.run).
    * A caller's basis has the directions of the basis of these buffers, and one that overlaps the bytes of the
      call in the workspace is that basis (runtime.check_caller_basis). Returns the buffers, name -> tensor.
    """
    b, s, h, _ = q.shape
    kernel = _KERNELS[pv]
    layout = kernel.layout(b, s, h, comp.rank, comp.bases, num_sms(q.device))
    buffers = carve(workspace, layout)
    if comp.basis is not None:
        check_caller_basis(comp.basis, kernel.basis(buffers, comp.bases), workspace, layout)
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
    """Run the attention on the buffers of prepare into BF16 o [B, S, H, 128], on the current stream. valid_seq_lens:
    the lengths that prepare took; rescale_threshold: one of constants.RESCALE_THRESHOLDS[pv]."""
    _KERNELS[pv].attention(buffers, o, rank, valid_seq_lens, rescale_threshold)


def basis(buffers: dict[str, torch.Tensor], pv: str, bases: str) -> Basis:
    """Return the compensation basis that the eigensolver of prepare left in the buffers of a call: FP32
    [B, H, eig_directions(rank), 128] in the rotated frame, the leading direction last; one tensor, or with bases
    "separate" the two (basis of the Q-residual term, basis of the K-residual term)."""
    return _KERNELS[pv].basis(buffers, bases)
