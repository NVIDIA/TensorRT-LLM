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

    workspace_size(b, s, h, device, pv, rank)             # bytes of the caller's workspace for one call
    prepare(q, k, v, workspace, pv, rank, rows)           # BF16 q, k, v [B, S, H, 128] -> the buffers of the call
    attention(buffers, o, pv, rank)                       # the buffers -> BF16 o [B, S, H, 128]

Kernels:
    * One kernel, of PV format "fp8" (fp8pv/): NVFP4 QK and FP8 PV with V smoothing.
    * Compensation rank 32 on every target; targets with the K = 128 block-scaled MMA also offer rank 64 (qk.ranks).
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

import torch

from ..runtime import carve, num_sms, workspace_bytes
from . import fp8pv
from .qk import ranks

__all__ = [
    "DEFAULT_PV",
    "attention",
    "min_seq_len",
    "prepare",
    "pv_formats",
    "ranks",
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


def workspace_size(b: int, s: int, h: int, device: torch.device | str, pv: str, rank: int) -> int:
    """Return the bytes of the workspace of one call of this shape, PV format and compensation rank on `device`."""
    return workspace_bytes(_KERNELS[pv].layout(b, s, h, rank, num_sms(device)))


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
    """Carve the buffers of a call from the workspace and fill the operands from raw q, k, v, on the current stream.

    rows, valid_seq_lens, map_rows: see fp8pv.prep.run (rows, lens, map_rows). Returns the buffers, name -> tensor.
    """
    b, s, h, _ = q.shape
    kernel = _KERNELS[pv]
    buffers = carve(workspace, kernel.layout(b, s, h, rank, num_sms(q.device)))
    kernel.prepare(q, k, v, rows, buffers, rank, valid_seq_lens, map_rows)
    return buffers


def attention(
    buffers: dict[str, torch.Tensor],
    o: torch.Tensor,
    pv: str,
    rank: int,
    valid_seq_lens: torch.Tensor | None = None,
) -> None:
    """Run the attention on the buffers of prepare into BF16 o [B, S, H, 128], on the current stream. valid_seq_lens:
    the lengths that prepare took."""
    _KERNELS[pv].attention(buffers, o, rank, valid_seq_lens)
