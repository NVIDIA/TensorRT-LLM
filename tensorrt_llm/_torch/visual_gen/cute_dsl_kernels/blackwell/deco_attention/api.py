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
"""Host API of the DeCo-Attention kernels: stateless functions on the current stream.

    workspace = torch.empty(workspace_size(batch, seq_len, heads), dtype=torch.uint8, device="cuda")
    out = torch.empty_like(q)
    attention(prepare(q, k, v, workspace), out)     # BF16 [B, S, H, 128] in and out

Workspace:
    * One flat buffer owned by the caller; a call keeps the operands of the attention and its scratch in it.
    * A call depends on nothing the buffer held before: one buffer serves calls of any shape that fits.
    * prepare returns the operands as views into it; they hold until the workspace is written again.

Kernels:
    * Family of the current GPU (target.kernel_family()): "tcgen05" on B200, GB300 and Rubin, "sm120" on RTX PRO 6000.
    * pv, one of pv_formats(): "fp8" (NVFP4 QK, FP8 PV) or "nvfp4" (NVFP4 QK, NVFP4 PV; on sm120, where it is the
      default); None is the family's default.
    * rank, one of ranks(): the rank of the QK compensation; 32 (the default) on every target, 64 where listed.
    * A family package deco_attention/<family> exports pv_formats, ranks, workspace_size, prepare and attention; the
      functions here check the caller's arguments (ValueError) and forward to it.

CUDA graphs:
    * prepare and attention can be captured: every scale stays on the device, nothing is read back or allocated.
    * The first call of a shape compiles its kernels and the first call on a device builds the rotation and the
      default rows: make it outside a capture.
"""

import functools
import importlib
from typing import NamedTuple

import torch

from . import target
from .constants import DEFAULT_RANK, HEAD_DIM
from .inputs import sample_rows


class Operands(NamedTuple):
    """What prepare returns and attention takes: the kernel (PV format, rank), the shape (batch, seq_len, heads)
    and the buffers of the call, name -> view into the workspace (the operands and the scratch of the attention)."""

    pv: str
    rank: int
    shape: tuple[int, int, int]
    buffers: dict[str, torch.Tensor]
    # the caller's valid prefix lengths (int32 [B], read in place by the attention), or None
    valid_seq_lens: torch.Tensor | None = None


@functools.cache
def _family():
    return importlib.import_module(f".{target.kernel_family()}", __spec__.parent)


def pv_formats():
    """Return the PV formats of the kernels of the current target, among "fp8" and "nvfp4" (module docstring)."""
    return _family().pv_formats()


def ranks():
    """Return the compensation ranks of the current target: among constants.RANKS, always with the default rank."""
    return _family().ranks()


def _kernel(pv, rank):
    """Check pv and rank; return the target's family and the PV format of its kernel (pv, or the family's default)."""
    family = _family()
    if pv is None:
        pv = family.DEFAULT_PV
    elif pv not in family.pv_formats():
        raise ValueError(
            f"pv {pv!r} is not offered on {target.arch()}: the PV formats of this target are "
            f"{family.pv_formats()}"
        )
    if rank not in family.ranks():
        raise ValueError(
            f"rank {rank!r} is not offered on {target.arch()}: the ranks of this target are "
            f"{family.ranks()}"
        )
    return family, pv


# The kernels move q, k, v and out through TMA descriptors and vector stores built on 256-byte aligned base addresses,
# the alignment of every allocation and of the workspace.
TENSOR_ALIGNMENT = 256


def token_rows(x):
    """Whether the tokens of x [B, S, H, HEAD_DIM] are rows of HEAD_DIM H contiguous elements at one pitch, a multiple
    of the alignment, over the batch: a contiguous tensor, or a view of the fused projection [B, S, Hq + 2 Hkv, D]."""
    return (
        x.stride(3) == 1
        and x.stride(2) == HEAD_DIM
        and x.stride(1) >= x.shape[2] * HEAD_DIM
        and x.stride(1) % (TENSOR_ALIGNMENT // x.element_size()) == 0
        and x.stride(0) == x.shape[1] * x.stride(1)
    )


def _check(shape, device, *, token_views=False, **tensors):
    """Check that the caller's tensors (name=tensor) are BF16 tensors of one shape [B, S, H, HEAD_DIM] (shape, or that
    of the first tensor for None) without an empty dimension, contiguous (token_views: or a view whose tokens are
    rows at one pitch, token_rows), on one device (device, or that of the first tensor), at 256-byte aligned
    addresses; return both."""
    for name, x in tensors.items():
        if (
            not torch.is_tensor(x)
            or x.dtype != torch.bfloat16
            or x.dim() != 4
            or x.shape[3] != HEAD_DIM
            or not (x.is_contiguous() or (token_views and token_rows(x)))
            or (shape is not None and tuple(x.shape) != shape)
        ):
            raise ValueError(
                f"{name} must be a contiguous BF16 tensor of shape "
                f"{list(shape) if shape else '[B, S, H, %d]' % HEAD_DIM}"
                + (" or a view of a fused projection with that shape" if token_views else "")
            )
        if x.numel() == 0:
            raise ValueError(
                f"{name} has an empty dimension; a sample without tokens is given by valid_seq_lens"
            )
        if x.data_ptr() % TENSOR_ALIGNMENT:
            raise ValueError(
                f"{name} must start at a {TENSOR_ALIGNMENT}-byte aligned address, as every allocation does: a view "
                f"into one at an offset that is not a multiple of {TENSOR_ALIGNMENT // 2} elements is not taken"
            )
        if device is not None and x.device != device:
            raise ValueError(f"{name} is on {x.device}, not on {device}")
        shape, device = tuple(x.shape), x.device
    return shape, device


def workspace_size(batch, seq_len, heads, device="cuda", pv=None, rank=DEFAULT_RANK):
    """Return the bytes of workspace that one call of this shape needs on `device`, whose SM count sets the launch
    plans: the operands, the scratch of the preparation and the scratch of the attention."""
    family, pv = _kernel(pv, rank)
    return family.workspace_size(batch, seq_len, heads, device, pv, rank)


def prepare(q, k, v, workspace, pv=None, rank=DEFAULT_RANK, rows=None, valid_seq_lens=None):
    """Run the preparation: write the operands of the attention into the workspace and return them (views into it).

    Args:
        q, k, v: BF16 [B, S, H, 128] on one device, at 256-byte aligned addresses (as every allocation and every view
            at an offset of a multiple of 128 elements is): contiguous, or the slices of a fused projection
            [B, S, Hq + 2 Hkv, 128], whose tokens the kernels read in place at the projection's pitch.
        workspace: uint8 [>= workspace_size(B, S, H, device, pv, rank)] on that device, 256-byte aligned, any content.
        rows: int32 [m] on that device, not empty: the tokens the compensation basis is estimated on.
            * None is the default draw, sample_rows(seq_len, device, rank=rank); a packed multimodal sequence passes
              sample_rows(seq_len, device, token_tags, rank), drawn on the device.
            * A caller's draw should hold at least 2 * rank tokens where the sequence has that many, the floor of the
              default draw; the kernels clamp a row outside [0, S) to the sequence.
        valid_seq_lens: None (every batch element is full), or int32 [B] on that device.
            * Batch element b has valid tokens on its first n = clamp(valid_seq_lens[b], 0, S) positions: it attends
              only its valid keys, its output rows from n on are zero, and q, k and v from n on have no influence.
            * The kernels read the tensor in place, also in the attention (it must hold until then); nothing reads it
              on the host.
            * The default draw is mapped into each valid prefix (row r stands for token floor(r n / S)); rows of the
              caller, [m] or [B, m] per batch element, are taken as given, and a row from n on stands for token n - 1.
    """
    family, pv = _kernel(pv, rank)
    shape, device = _check(None, None, token_views=True, q=q, k=k, v=v)
    if (
        not torch.is_tensor(workspace)
        or workspace.dtype != torch.uint8
        or workspace.dim() != 1
        or workspace.device != device
        or not workspace.is_contiguous()
    ):
        raise ValueError(f"workspace must be a contiguous one-dimensional uint8 tensor on {device}")
    lens = valid_seq_lens
    if lens is not None:
        if (
            not torch.is_tensor(lens)
            or lens.dtype != torch.int32
            or tuple(lens.shape) != shape[:1]
            or lens.device != device
            or not lens.is_contiguous()
            or lens.data_ptr() % 16
        ):
            raise ValueError(
                f"valid_seq_lens must be a contiguous int32 tensor [{shape[0]}] on {device} at a 16-byte aligned "
                "address"
            )
    map_rows = rows is None and lens is not None
    if rows is None:
        rows = sample_rows(shape[1], device, rank=rank)
    elif (
        not torch.is_tensor(rows)
        or rows.dtype != torch.int32
        or rows.device != device
        or not rows.is_contiguous()
        or rows.data_ptr() % 16
        or not (
            rows.dim() == 1 or (lens is not None and rows.dim() == 2 and rows.shape[0] == shape[0])
        )
    ):
        raise ValueError(
            f"rows must be a contiguous int32 tensor on {device} at a 16-byte aligned address, one-dimensional "
            f"or, with valid_seq_lens, [{shape[0]}, m]"
        )
    elif rows.numel() == 0:
        raise ValueError("rows must hold at least one token: the basis is estimated on them")
    buffers = family.prepare(q, k, v, workspace, pv, rank, rows, lens, map_rows)
    return Operands(pv, rank, shape[:3], buffers, lens)


def attention(operands, out):
    """Run the attention on operands of prepare: fill and return out, contiguous BF16 [B, S, H, 128] on their device at
    a 256-byte aligned address."""
    if not isinstance(operands, Operands):
        raise ValueError("operands is not a result of prepare")
    device = next(iter(operands.buffers.values())).device
    _check((*operands.shape, HEAD_DIM), device, out=out)
    _family().attention(operands.buffers, out, operands.pv, operands.rank, operands.valid_seq_lens)
    return out
