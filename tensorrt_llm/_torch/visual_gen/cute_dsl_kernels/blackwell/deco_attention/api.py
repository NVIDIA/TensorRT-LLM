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
    * Sequences of at least min_seq_len(pv) tokens: one token, or one key tile of 128 for "fp8" on sm120.
    * A family package deco_attention/<family> exports DEFAULT_PV, pv_formats, max_comp_channels, min_seq_len,
      workspace_size, prepare, attention and basis; the functions here check the caller's arguments (ValueError),
      resolve the defaults and forward resolved values to it: the compensation of a call as inputs.Compensation, a
      basis as the tuple of its tensors (runtime.Basis), the scale rule as its name.

Compensation (the switches of prepare; workspace_size takes rank and bases):
    * rank: the channels (r_q, r_k) of the two compensation terms, multiples of 16 with r_q + r_k at most
      max_comp_channels(): 64, or 128 on a target with the 256-channel QK. An int r is (r, r); None is half of
      max_comp_channels() each. The Q-residual term compensates the rounding residual of Q on r_q directions, the
      K-residual term that of K on r_k directions; 0 drops a term.
    * bases: "shared", both terms on the leading eigenvectors of w Gq / tr(Gq) + (1 - w) Gk / tr(Gk) (Gq, Gk: the
      Gram matrices of the sampled Q and K rows; w: gram_weight, 0.5 by default); "separate", the Q-residual term
      on the leading eigenvectors of Gk and the K-residual term on those of Gq, without a weight. With one term
      alone "separate" is the one basis of the other operand: the call of "shared" with weight 0 (r_k = 0) or 1
      (r_q = 0).
    * rows: the tokens the Gram matrices are estimated on, one tensor for both or a pair (rows of Q, rows of K).
    * basis: Operands.basis of an earlier call of the same rank and bases, in place of the eigensolver.
    * attention takes rescale_threshold, one of rescale_thresholds(pv).
    * A pair (rank, rows, basis) is a tuple or a list of two.

Quantizer of Q' and K' (a switch of prepare):
    * scale_rule: how the scale of an NVFP4 block of 16 values of the head channels is chosen, one of
      constants.SCALE_RULES; None is constants.DEFAULT_SCALE_RULE, "band". With m the largest magnitude of the block:
        - "max": m / 6, which puts the largest magnitude on the largest code.
        - "band": m / 4 where a value of the block has 0.76 m <= |x| < 0.85 m, else m / 6.
        - "search": the block is quantized with m / 6 and with m / 4 and keeps the scale with the smaller squared
          rounding error, m / 6 when they are equal.
    * The compensation terms take the residual of the kept scale; a block of the compensation channels takes m / 6
      under every rule.
    * It changes neither the operand layout nor the workspace nor the attention kernels.

CUDA graphs:
    * prepare and attention can be captured: every scale stays on the device, nothing is read back or allocated.
    * A capture follows one eager call with the same shape and arguments. That call compiles the kernels of the
      call, each logged with its key, and builds the rotation and the default rows; a first use of any of them under
      capture raises RuntimeError. A family lists what a kernel compiles for beside its key.
"""

from types import ModuleType
from typing import NamedTuple

import torch

from . import target
from .constants import BASES, DEFAULT_SCALE_RULE, HEAD_DIM, RESCALE_THRESHOLDS, SCALE_RULES
from .inputs import Compensation, rank_pair, sample_rows
from .runtime import Basis


class Operands(NamedTuple):
    """What prepare returns and attention takes: the kernel (PV format, rank), the shape (batch, seq_len, heads)
    and the buffers of the call, name -> view into the workspace (the operands and the scratch of the attention)."""

    pv: str
    rank: tuple[int, int]
    shape: tuple[int, int, int]
    buffers: dict[str, torch.Tensor]
    # the caller's valid prefix lengths (int32 [B], read in place by the attention), or None
    valid_seq_lens: torch.Tensor | None = None
    # the compensation basis of the call: FP32 [B, H, directions, 128] in the rotated frame, the leading direction
    # last; a pair (basis of the Q-residual term, basis of the K-residual term) with two terms on bases "separate";
    # None without compensation channels. The basis the caller passed, or views into the workspace (prepare: basis)
    basis: torch.Tensor | Basis | None = None


def pv_formats() -> tuple[str, ...]:
    """Return the PV formats of the kernels of the current target, among "fp8" and "nvfp4" (module docstring)."""
    return target.family().pv_formats()


def max_comp_channels() -> int:
    """Return the most compensation channels r_q + r_k of the current target: 64, or 128 with the 256-channel QK."""
    return target.family().max_comp_channels()


def rescale_thresholds(pv: str | None = None) -> tuple[float, ...]:
    """Return the rescale thresholds that the attention of the kernel of PV format pv (None: the family's default)
    offers, the default first."""
    return RESCALE_THRESHOLDS[_family(pv)[1]]


def _family(pv: str | None) -> tuple[ModuleType, str]:
    """Check pv; return the target's family and the PV format of its kernel: pv, or the family's default."""
    family = target.family()
    if pv is None:
        return family, family.DEFAULT_PV
    if pv not in family.pv_formats():
        raise ValueError(
            f"pv {pv!r} is not offered on {target.arch()}: the PV formats of this target are "
            f"{family.pv_formats()}"
        )
    return family, pv


def _terms(
    rank: int | tuple[int, int] | list[int] | None, bases: str, gram_weight: float | None
) -> tuple[tuple[int, int], str, float]:
    """Check rank, bases and gram_weight; return what a family runs: the rank as the pair (r_q, r_k), the bases kind
    and the Gram weight.

    * Bases "shared": gram_weight None is 0.5.
    * Bases "separate" takes no gram_weight; with one term alone the family runs "shared" with weight 0 (the
      Q-residual term, on Gk) or 1 (the K-residual term, on Gq).
    """
    if bases not in BASES:
        raise ValueError(f"bases {bases!r} is not one of {BASES}")
    rank = rank_pair(rank)
    if bases == "separate":
        if gram_weight is not None:
            raise ValueError(f"bases 'separate' takes no gram_weight, got {gram_weight!r}")
        if rank[0] and rank[1]:
            return rank, bases, 0.5
        return rank, "shared", 0.0 if rank[1] == 0 else 1.0
    if gram_weight is None:
        return rank, bases, 0.5
    if type(gram_weight) not in (float, int) or not 0.0 <= gram_weight <= 1.0:
        raise ValueError(f"gram_weight must be a float in [0, 1], got {gram_weight!r}")
    return rank, bases, float(gram_weight)


def _scale_rule(scale_rule: str | None) -> str:
    """Check scale_rule; return the rule for the scale of a head block that a family runs: one of
    constants.SCALE_RULES, the default for None."""
    if scale_rule is None:
        return DEFAULT_SCALE_RULE
    if type(scale_rule) is not str or scale_rule not in SCALE_RULES:
        raise ValueError(f"scale_rule {scale_rule!r} is not one of {SCALE_RULES}")
    return scale_rule


def min_seq_len(pv: str | None = None) -> int:
    """Return the shortest sequence that the kernel of PV format pv (None: the family's default) takes."""
    family, pv = _family(pv)
    return family.min_seq_len(pv)


def _check_seq_len(family: ModuleType, pv: str, seq_len: int) -> None:
    if seq_len < family.min_seq_len(pv):
        raise ValueError(
            f"pv {pv!r} on {target.arch()} takes sequences of at least {family.min_seq_len(pv)} tokens, "
            f"got {seq_len}"
        )


# The kernels move q, k, v and out through TMA descriptors and vector stores built on 256-byte aligned base addresses,
# the alignment of every allocation and of the workspace.
TENSOR_ALIGNMENT = 256


def token_rows(x: torch.Tensor) -> bool:
    """Whether the tokens of x [B, S, H, HEAD_DIM] are rows of HEAD_DIM H contiguous elements at one pitch, a multiple
    of the alignment, over the batch: a contiguous tensor, or a view of the fused projection [B, S, Hq + 2 Hkv, D]."""
    return (
        x.stride(3) == 1
        and x.stride(2) == HEAD_DIM
        and x.stride(1) >= x.shape[2] * HEAD_DIM
        and x.stride(1) % (TENSOR_ALIGNMENT // x.element_size()) == 0
        and x.stride(0) == x.shape[1] * x.stride(1)
    )


def _check(
    shape: tuple[int, ...] | None,
    device: torch.device | None,
    *,
    token_views: bool = False,
    **tensors: torch.Tensor,
) -> tuple[tuple[int, ...], torch.device]:
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


def _read_in_place(x: object, dtype: torch.dtype, device: torch.device) -> bool:
    """Whether x is what the kernels read in place besides q, k and v: a contiguous tensor of a dtype on the device
    at a 16-byte aligned address."""
    return (
        torch.is_tensor(x)
        and x.dtype == dtype
        and x.device == device
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
    )


def _check_rows(
    name: str, rows: torch.Tensor, shape: tuple[int, ...], device: torch.device, per_sample: bool
) -> None:
    """Check a caller's rows: an int32 tensor that the kernels read in place (_read_in_place), not empty, [m] or,
    where per_sample allows it (a call with valid_seq_lens), [B, m]."""
    if not _read_in_place(rows, torch.int32, device) or not (
        rows.dim() == 1 or (per_sample and rows.dim() == 2 and rows.shape[0] == shape[0])
    ):
        raise ValueError(
            f"{name} must be a contiguous int32 tensor on {device} at a 16-byte aligned address, one-dimensional "
            f"or, with valid_seq_lens, [{shape[0]}, m]"
        )
    if rows.numel() == 0:
        raise ValueError(f"{name} must hold at least one token: the basis is estimated on them")


def _basis_tensors(
    basis: torch.Tensor | Basis | list[torch.Tensor],
    bases: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> Basis:
    """Check a caller's basis and return the tuple of its tensors: for bases "shared" one FP32 tensor
    [B, H, directions, HEAD_DIM] that the kernels read in place (_read_in_place), for "separate" a pair of them (the
    family checks the directions and the place of a basis that overlaps the workspace)."""
    parts = (basis,) if bases == "shared" else basis
    if (
        not isinstance(parts, (tuple, list))
        or len(parts) != (1 if bases == "shared" else 2)
        or not all(
            _read_in_place(u, torch.float32, device)
            and u.dim() == 4
            and (u.shape[0], u.shape[1], u.shape[3]) == (shape[0], shape[2], HEAD_DIM)
            for u in parts
        )
    ):
        raise ValueError(
            f"basis must be Operands.basis of a call of this batch and heads with bases {bases!r}: "
            + ("one tensor" if bases == "shared" else "a pair of tensors")
            + f", contiguous FP32 [{shape[0]}, {shape[2]}, directions, {HEAD_DIM}] on {device} at a 16-byte "
            "aligned address"
        )
    return tuple(parts)


def workspace_size(
    batch: int,
    seq_len: int,
    heads: int,
    device: torch.device | str = "cuda",
    pv: str | None = None,
    rank: int | tuple[int, int] | None = None,
    *,
    bases: str = "shared",
) -> int:
    """Return the bytes of workspace that one call of this shape, rank and bases needs on `device`, whose SM count
    sets the launch plans: the operands, the scratch of the preparation and the scratch of the attention."""
    family, pv = _family(pv)
    rank, bases, _ = _terms(rank, bases, None)
    _check_seq_len(family, pv, seq_len)
    return family.workspace_size(batch, seq_len, heads, device, pv, rank, bases)


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    workspace: torch.Tensor,
    pv: str | None = None,
    rank: int | tuple[int, int] | list[int] | None = None,
    rows: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    valid_seq_lens: torch.Tensor | None = None,
    *,
    bases: str = "shared",
    gram_weight: float | None = None,
    basis: torch.Tensor | Basis | None = None,
    scale_rule: str | None = None,
) -> Operands:
    """Run the preparation: write the operands of the attention into the workspace and return them (views into it).

    Args:
        q, k, v: BF16 [B, S, H, 128] on one device, at 256-byte aligned addresses (as every allocation and every view
            at an offset of a multiple of 128 elements is): contiguous, or the slices of a fused projection
            [B, S, Hq + 2 Hkv, 128], whose tokens the kernels read in place at the projection's pitch.
        workspace: uint8 [>= workspace_size(B, S, H, device, pv, rank, bases=bases)] on that device, 256-byte aligned,
            any content.
        rank, bases: the compensation channels and their bases (module docstring).
        rows: int32 [m] on that device at a 16-byte aligned address, not empty: the tokens the Gram matrices of Q and
            of K are estimated on; a pair (rows of Q, rows of K) gives each operand its own.
            * None is the default draw, sample_rows(seq_len, device, rank=rank); a packed multimodal sequence passes
              sample_rows(seq_len, device, token_tags, rank), drawn on the device.
            * A caller's draw should hold at least twice the channels of the larger term where the sequence has that
              many tokens, as the default draw does; the kernels clamp a row outside [0, S) to the sequence.
        gram_weight: w in [0, 1] of bases "shared", the weight of the Gram matrix of Q against that of K: None is
            0.5, which weighs them equally; 1 takes Q alone and 0 takes K alone. Bases "separate" takes None only.
        basis: None, or Operands.basis of an earlier call of this batch, heads, rank and bases: the preparation then
            runs no eigensolver and reads the tensors in place, and rows is not used.
            * Operands.basis of a call that computed its basis is a view into the workspace. A later call on that
              workspace keeps it only if it has the same shape, rank and bases and passes a basis itself: a call
              that computes its basis writes it to the same place, and any other call lays the workspace out anew.
              Nothing detects an overwritten view, so a basis that is to outlive other calls is cloned first.
            * A basis that overlaps the bytes of this call in the workspace must be the view of such a call
              (ValueError otherwise); a clone lies anywhere else.
        valid_seq_lens: None (every batch element is full), or int32 [B] on that device at a 16-byte aligned address.
            * Batch element b has valid tokens on its first n = clamp(valid_seq_lens[b], 0, S) positions: it attends
              only its valid keys, its output rows from n on are zero, and q, k and v from n on have no influence.
            * The kernels read the tensor in place, also in the attention (it must hold until then); nothing reads it
              on the host.
            * The default draw is mapped into each valid prefix (row r stands for token floor(r n / S)); rows of the
              caller, [m] or [B, m] per batch element, are taken as given, and a row from n on stands for token n - 1.
              A caller's draw is therefore made within the valid tokens:
              sample_rows(seq_len, device, rank=rank, tokens=mask, rate=..., min_rows=...) with the bool mask
              [B, seq_len] of the valid prefixes, arange(seq_len) < valid_seq_lens[:, None], built once per set of
              lengths: sample_rows keeps a draw for the mask tensor it was made from.
        scale_rule: the rule for the scale of an NVFP4 block of the head channels of Q' and K' (module docstring):
            "max", "band" or "search", None for the default.
    """
    family, pv = _family(pv)
    rank, bases, gram_weight = _terms(rank, bases, gram_weight)
    scale_rule = _scale_rule(scale_rule)
    shape, device = _check(None, None, token_views=True, q=q, k=k, v=v)
    _check_seq_len(family, pv, shape[1])
    if (
        not torch.is_tensor(workspace)
        or workspace.dtype != torch.uint8
        or workspace.dim() != 1
        or workspace.device != device
        or not workspace.is_contiguous()
    ):
        raise ValueError(f"workspace must be a contiguous one-dimensional uint8 tensor on {device}")
    lens = valid_seq_lens
    if lens is not None and not (
        _read_in_place(lens, torch.int32, device) and tuple(lens.shape) == shape[:1]
    ):
        raise ValueError(
            f"valid_seq_lens must be a contiguous int32 tensor [{shape[0]}] on {device} at a 16-byte aligned "
            "address"
        )
    rows_q = rows_k = None
    compensated = rank != (0, 0)
    if basis is not None:
        if not compensated:
            raise ValueError(f"rank {rank} has no compensation channels and takes no basis")
        basis = _basis_tensors(basis, bases, shape, device)
    elif compensated:
        if rows is None:
            rows_q = rows_k = sample_rows(shape[1], device, rank=rank)
        elif isinstance(rows, (tuple, list)):
            if len(rows) != 2:
                raise ValueError("rows must be one tensor or a pair (rows of Q, rows of K)")
            rows_q, rows_k = rows
            _check_rows("rows of Q", rows_q, shape, device, lens is not None)
            _check_rows("rows of K", rows_k, shape, device, lens is not None)
            if rows_q.dim() != rows_k.dim():
                raise ValueError(
                    "the rows of Q and of K must both be one-dimensional or both [B, m]"
                )
        else:
            _check_rows("rows", rows, shape, device, lens is not None)
            rows_q = rows_k = rows
    map_rows = rows is None and lens is not None
    comp = Compensation(rank, bases, gram_weight, rows_q, rows_k, map_rows, basis)
    buffers = family.prepare(q, k, v, workspace, pv, lens, comp, scale_rule)
    if compensated and basis is None:
        basis = family.basis(buffers, pv, bases)
    if basis is not None and len(basis) == 1:
        basis = basis[0]
    return Operands(pv, rank, shape[:3], buffers, lens, basis)


def attention(
    operands: Operands, out: torch.Tensor, *, rescale_threshold: float | None = None
) -> torch.Tensor:
    """Run the attention on operands of prepare: fill and return out, contiguous BF16 [B, S, H, 128] on their device at
    a 256-byte aligned address. rescale_threshold: T of the online softmax, one of rescale_thresholds(pv) (None: the
    first): the anchor of a row moves, and its output is rescaled, when a key tile exceeds it by more than T binades."""
    if not isinstance(operands, Operands):
        raise ValueError("operands is not a result of prepare")
    offered = RESCALE_THRESHOLDS[operands.pv]
    if rescale_threshold is None:
        rescale_threshold = offered[0]
    elif isinstance(rescale_threshold, bool) or rescale_threshold not in offered:
        raise ValueError(
            f"rescale_threshold {rescale_threshold!r} is not offered for pv {operands.pv!r}: its thresholds are "
            f"{offered}"
        )
    device = next(iter(operands.buffers.values())).device
    _check((*operands.shape, HEAD_DIM), device, out=out)
    target.family().attention(
        operands.buffers,
        out,
        operands.pv,
        operands.rank,
        operands.valid_seq_lens,
        float(rescale_threshold),
    )
    return out
