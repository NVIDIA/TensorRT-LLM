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
"""Kimi K3's collective sandwiches in CuTe DSL, for M <= 8 decode tokens: a row-parallel projection, the TP
all-reduce of its output and the residual update (attention-residual selection + RMSNorm) in one kernel.
``trtllm::k3_sandwich_oproj`` is the post-attention step (the attention output projection);
``trtllm::k3_sandwich_tail`` the pre-attention step (the MoE tail, then the next layer's input norm);
``trtllm::k3_sandwich_plain`` a row-parallel projection with a plain residual add + RMSNorm (the drafter layers).

The all-reduce runs over a dedicated multicast buffer per TP group, a :class:`K3SandwichWorkspace` that the caller
creates (collectively, before CUDA-graph capture) and passes to every call; it is not the model's MNNVL all-reduce
workspace, so it keeps its own call parity. The kernel compiles on the first call for each (world, publish, input
source, PDL), which must happen outside CUDA-graph capture; the number of snapshots, the prefix and the slab buffer are
runtime arguments.

Publishing (``x_slab``, ``slab_buf``): with a slab (``slab_tensor``) the normed rows are also written into buffer
``slab_buf`` (0-2) of it, sentinel-armed Lamport words the next kernel polls, and buffer ``(slab_buf + 1) % 3`` is
re-armed. ``slab_buf`` is the ordinal of the call among this op's calls in the forward, mod 3.

The latent all-reduce folded into the tail (``lat_uc``, ``lat_flags`` of a :class:`K3SandwichLatentExchange`): k3_moe
pushes its routed latent partial into every rank's exchange buffer and exits; ``k3_sandwich_tail`` sums the ranks'
partials itself (bit-identical to the one-shot all-reduce) instead of reading a reduced ``latent``. Every push-only
k3_moe call must be followed by exactly one such tail call on the same exchange.

Polling the input (``src_slab``, ``src_buf``): when the producer of phase 1's input (the attention output core for
``k3_sandwich_oproj``, the reduced latent for ``k3_sandwich_tail``) publishes it as such a slab (int32 [3][8][cols /
2], sentinel 0xFFFFFFFF) and launches its dependents only after its own grid wait, the kernel polls buffer
``src_buf`` of it instead of waiting for the producer's grid; ``core`` / ``latent`` then give only the shape.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

MAX_TOKENS = 8
EMPTY_WORD = -(2**31)
SLAB_BUFS = 3
SLAB_SENTINEL = -1

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _arg(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    # detach(): DLPack refuses tensors that require grad (weights are parameters).
    return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.int32)


def _tap_words(tap: torch.Tensor) -> torch.Tensor:
    """A strided bf16 [M, 7168] view as the flat int32 words from its first row to the end of its last (the gaps
    between rows are never touched), without a copy."""
    words = tap.view(torch.int32)
    extent = (tap.shape[0] - 1) * words.stride(0) + words.shape[1]
    return words.as_strided((extent,), (1,))


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def slab_tensor(device) -> torch.Tensor:
    """A publication slab for one producer edge: int32 [3, 8, 3584] (three buffers of 8 bf16 rows of 7168), every
    word the sentinel."""
    from . import k3_sandwich_kernel as kernel

    return torch.full(
        (SLAB_BUFS, MAX_TOKENS, kernel.WORDS_PER_ROW),
        SLAB_SENTINEL,
        dtype=torch.int32,
        device=device,
    )


def _src_args(src_slab: Optional[torch.Tensor], src_buf: int, cols: int, fallback: torch.Tensor):
    """(slab words, buffer, x_src) of phase 1's input; ``fallback`` and x_src 0 without a slab."""
    if src_slab is None:
        return fallback, 0, 0
    words = SLAB_BUFS * MAX_TOKENS * cols // 2
    if src_slab.dtype != torch.int32 or src_slab.numel() != words or not src_slab.is_contiguous():
        raise ValueError(
            f"k3_sandwich: src_slab must be a contiguous int32 [3, 8, {cols // 2}] slab, got {tuple(src_slab.shape)} "
            f"{src_slab.dtype}"
        )
    if not 0 <= src_buf < SLAB_BUFS:
        raise ValueError(f"k3_sandwich: src_buf must be 0, 1 or 2, got {src_buf}")
    return src_slab.reshape(-1), int(src_buf), 1


def _slab_args(x_slab: Optional[torch.Tensor], slab_buf: int, fallback: torch.Tensor):
    """(slab words, buffer, publish) for the kernel; a dummy view and publish 0 without a slab."""
    from . import k3_sandwich_kernel as kernel

    if x_slab is None:
        return _words(fallback), 0, 0
    if (
        x_slab.dtype != torch.int32
        or x_slab.numel() != SLAB_BUFS * kernel.SLAB_WORDS
        or not x_slab.is_contiguous()
    ):
        raise ValueError(
            f"k3_sandwich: x_slab must be a contiguous int32 [3, 8, 3584] slab, "
            f"got {tuple(x_slab.shape)} {x_slab.dtype}"
        )
    if not 0 <= slab_buf < SLAB_BUFS:
        raise ValueError(f"k3_sandwich: slab_buf must be 0, 1 or 2, got {slab_buf}")
    return x_slab.reshape(-1), int(slab_buf), 1


def _create_buffer(cls, mapping, words: int, flag_words: int, fabric_handle: Optional[bool],
                   arm_flags: Optional[Callable[[torch.Tensor], None]] = None):  # fmt: skip
    """A ``cls`` over a new multicast buffer of ``words`` int32 per rank of ``mapping``'s TP group, every word empty,
    and ``flag_words`` int32 flags, zero (then ``arm_flags``). Collective and eager: every rank of the group calls it at
    the same point, outside CUDA-graph capture; it returns on every rank or raises on every rank."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{cls.__name__}.create is collective and allocates: call it outside CUDA-graph capture"
        )
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    use_fabric_handle = mapping.is_multi_node() if fabric_handle is None else bool(fabric_handle)
    comm = _get_mnnvl_workspace_comm(mapping)
    error: Optional[Exception] = None
    state = None
    try:
        handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, use_fabric_handle)
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        uc.fill_(EMPTY_WORD)
        flags = torch.zeros(flag_words, dtype=torch.int32, device=uc.device)
        if arm_flags is not None:
            arm_flags(flags)
        torch.cuda.synchronize()
        state = cls(
            uc=uc,
            mc=mc,
            flags=flags,
            rank=mapping.tp_rank,
            world_size=mapping.tp_size,
            handle=handle,
            comm=comm,
        )
    except Exception as exc:  # noqa: BLE001 -- reported to every rank below, then re-raised
        error = exc
    # Also the barrier that keeps any rank from pushing into a peer's buffer before the peer has emptied it.
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError(f"{cls.__name__}: allocation failed on at least one rank") from error
    return state


@dataclass(eq=False)
class K3SandwichWorkspace:
    """One TP group's sandwich all-reduce buffer, shared by ``k3_sandwich_oproj``, ``k3_sandwich_tail`` and
    ``k3_sandwich_plain``: two alternating halves of [8 tokens][world][7168] bf16 per rank behind one multicast mapping,
    and one call counter per CTA whose parity selects the half. Every sandwich call on it advances every counter, so all
    of a group's ranks make the same calls on it in the same order. Pass ``uc``, ``mc``, ``flags`` and ``rank`` as the
    ops' ``ws_uc``, ``ws_mc``, ``ws_flags`` and ``rank``."""

    uc: torch.Tensor
    """int32 [2 * 8 * world * 3584]: this rank's words (0x80000000 = empty)."""
    mc: torch.Tensor
    """The same words through the multicast mapping (where the peers push)."""
    flags: torch.Tensor
    """int32 [64]: the call count of each of the kernel's 56 CTAs."""
    rank: int
    world_size: int
    handle: Any
    """The ``McastGPUBuffer`` that owns the memory; the workspace is valid while this object lives."""
    comm: Any
    """The TP-group communicator the handles were exchanged over."""

    @classmethod
    def create(cls, mapping, fabric_handle: Optional[bool] = None) -> "K3SandwichWorkspace":
        """Allocate and arm a workspace for ``mapping``'s TP group. Collective: every rank of the group calls it at the
        same point, eagerly (not under CUDA-graph capture); it returns on every rank or raises on every rank.
        ``fabric_handle``: share the memory by fabric handle (required across nodes) rather than POSIX file
        descriptor; default ``mapping.is_multi_node()``."""
        from . import k3_sandwich_kernel as kernel

        return _create_buffer(
            cls, mapping, kernel.buffer_words(mapping.tp_size), kernel.FLAG_WORDS, fabric_handle
        )


@dataclass(eq=False)
class K3SandwichLatentExchange:
    """One TP group's latent exchange for ``k3_sandwich_tail`` with the latent all-reduce folded in: the push-only
    k3_moe stores every rank's routed partial into two alternating halves of [8 tokens][world][3584] bf16 per rank
    behind one multicast mapping, and the tail sums them. ``flags``: [0] the tail's call count mod 6 (its parity selects
    the half), then the tail's latent scale slab. Pass ``uc`` and ``flags`` as the tail's ``lat_uc`` and ``lat_flags``.
    Separate from :class:`K3SandwichWorkspace`."""

    uc: torch.Tensor
    """int32 [2 * 8 * world * 1792]: this rank's words (0x80000000 = empty)."""
    mc: torch.Tensor
    """The same words through the multicast mapping (where the producers push)."""
    flags: torch.Tensor
    """int32 [64]: [0] the call count mod 6, [32 + 8 b + t] buffer b of the latent scales (sentinel 0xFFFFFFFF)."""
    rank: int
    world_size: int
    handle: Any
    """The ``McastGPUBuffer`` that owns the memory; the exchange is valid while this object lives."""
    comm: Any
    """The TP-group communicator the handles were exchanged over."""

    @classmethod
    def create(cls, mapping, fabric_handle: Optional[bool] = None) -> "K3SandwichLatentExchange":
        """Allocate and arm an exchange for ``mapping``'s TP group; collective and eager, as
        :meth:`K3SandwichWorkspace.create`."""
        from . import k3_sandwich_kernel as kernel

        def arm(flags: torch.Tensor) -> None:
            scales = slice(
                kernel.LAT_SCALES, kernel.LAT_SCALES + kernel.LAT_SCALE_BUFS * MAX_TOKENS
            )
            flags[scales] = kernel.SCALE_SENTINEL

        return _create_buffer(
            cls,
            mapping,
            kernel.lat_buffer_words(mapping.tp_size),
            kernel.LAT_FLAG_WORDS,
            fabric_handle,
            arm,
        )


def supports(core: torch.Tensor, o_weight: torch.Tensor, num_snapshots: int) -> bool:
    """Whether the kernel runs this call: bf16, M <= 8, the TP16 per-rank o_proj shape [7168, 768]."""
    from . import k3_sandwich_kernel as kernel

    return (
        core.is_cuda
        and core.dtype == o_weight.dtype == torch.bfloat16
        and core.dim() == 2
        and 0 < core.shape[0] <= MAX_TOKENS
        and core.shape[1] == kernel.K_IN
        and tuple(o_weight.shape) == (kernel.H, kernel.K_IN)
        and core.is_contiguous()
        and o_weight.is_contiguous()
        and 0 <= num_snapshots < kernel.MAX_CANDIDATES
    )


def _compile_and_run(entry, key, args, runtime, consts, stream, name):
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"trtllm::{name} must run once per configuration outside CUDA-graph capture first "
                "(it compiles its kernel on the first call)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(entry, *args, *runtime, *consts, stream)
    # The compiled function takes the runtime arguments only.
    fn(*args, *runtime, stream)


@torch.library.custom_op(
    "trtllm::k3_sandwich_oproj", mutates_args=("ws_uc", "ws_mc", "ws_flags", "x_slab")
)
def k3_sandwich_oproj(
    core: torch.Tensor,
    o_weight: torch.Tensor,
    prefix: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    x_slab: Optional[torch.Tensor] = None,
    slab_buf: int = 0,
    src_slab: Optional[torch.Tensor] = None,
    src_buf: int = 0,
) -> List[torch.Tensor]:
    """``(normed, updated)`` of ``o_proj`` followed by ``allreduce_attn_res_rmsnorm``.

    ``core`` bf16 [M, 768] is this rank's attention output, ``o_weight`` [7168, 768] its o_proj slice;
    ``prefix`` [M, 7168] (or None), ``block_residual`` [S, M, 7168] the S valid snapshots, the weights [7168];
    ``ws_*`` from a :class:`K3SandwichWorkspace`. ``x_slab`` (from :func:`slab_tensor`) also receives normed in buffer
    ``slab_buf``; ``src_slab`` (int32 [3, 8, 384]) supplies core in buffer ``src_buf``."""
    import cuda.bindings.driver as cuda_driver

    from . import k3_sandwich_kernel as kernel

    num_tokens = core.shape[0]
    num_snapshots = block_residual.shape[0]
    if not supports(core, o_weight, num_snapshots):
        raise ValueError(
            f"k3_sandwich_oproj: unsupported call core {tuple(core.shape)} {core.dtype}, o_weight "
            f"{tuple(o_weight.shape)}, snapshots {num_snapshots}"
        )
    world = ws_uc.numel() // kernel.buffer_words(1)
    normed = torch.empty(num_tokens, kernel.H, dtype=torch.bfloat16, device=core.device)
    updated = torch.empty_like(normed)
    snaps = block_residual if num_snapshots > 0 else core.new_zeros(1, num_tokens, kernel.H)
    add_prefix = prefix is not None
    slab_words, buf, publish = _slab_args(x_slab, slab_buf, ws_flags)
    src_words, sbuf, x_src = _src_args(src_slab, src_buf, kernel.K_IN, ws_flags)
    args = (
        _arg(o_weight),
        _arg(core),
        _arg(ws_uc),
        _arg(ws_mc),
        _arg(ws_flags),
        _arg(_words(prefix if add_prefix else snaps)),
        _arg(_words(snaps)),
        _arg(_words(res_weight)),
        _arg(_words(rms_weight)),
        _arg(_words(output_rms_weight)),
        _arg(_words(updated)),
        _arg(_words(normed)),
        _arg(slab_words),
        _arg(src_words),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(core.device).cuda_stream)
    use_pdl = _use_pdl()
    key = (world, publish, x_src, use_pdl)
    runtime = (
        num_tokens,
        rank,
        num_snapshots + 1,
        int(add_prefix),
        float(rms_eps),
        float(output_rms_eps),
        buf,
        sbuf,
    )
    consts = (world, publish, x_src, use_pdl)
    _compile_and_run(
        kernel.k3_sandwich_oproj, key, args, runtime, consts, stream, "k3_sandwich_oproj"
    )
    return [normed, updated]


@k3_sandwich_oproj.register_fake
def _(core, o_weight, prefix, block_residual, res_weight, rms_weight, output_rms_weight, rms_eps, output_rms_eps,
      ws_uc, ws_mc, ws_flags, rank, x_slab=None, slab_buf=0, src_slab=None, src_buf=0):  # fmt: skip
    normed = core.new_empty((core.shape[0], o_weight.shape[0]), dtype=torch.bfloat16)
    return [normed, torch.empty_like(normed)]


def supports_tail(
    latent: torch.Tensor, act: torch.Tensor, tail_weight: torch.Tensor, num_snapshots: int
) -> bool:
    """Whether the pre-attention kernel runs this call: bf16, M <= 8, the TP16 shapes (latent [M, 3584], act
    [M, 384], weight [7168, 256 + 384])."""
    from . import k3_sandwich_kernel as kernel

    return (
        latent.is_cuda
        and latent.dtype == act.dtype == tail_weight.dtype == torch.bfloat16
        and latent.dim() == 2
        and act.dim() == 2
        and 0 < latent.shape[0] <= MAX_TOKENS
        and act.shape[0] == latent.shape[0]
        and latent.shape[1] == kernel.LATENT
        and act.shape[1] == kernel.TAIL_ACT
        and tuple(tail_weight.shape) == (kernel.H, kernel.TAIL_LAT + kernel.TAIL_ACT)
        and latent.is_contiguous()
        and act.is_contiguous()
        and tail_weight.is_contiguous()
        and 0 <= num_snapshots < kernel.MAX_CANDIDATES
    )


@torch.library.custom_op(
    "trtllm::k3_sandwich_tail",
    mutates_args=(
        "ws_uc",
        "ws_mc",
        "ws_flags",
        "x_slab",
        "lat_uc",
        "lat_flags",
        "tap",
        "updated_out",
    ),
)
def k3_sandwich_tail(
    latent: torch.Tensor,
    act: torch.Tensor,
    tail_weight: torch.Tensor,
    lo: int,
    lat_eps: float,
    prefix: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    x_slab: Optional[torch.Tensor] = None,
    slab_buf: int = 0,
    src_slab: Optional[torch.Tensor] = None,
    src_buf: int = 0,
    lat_uc: Optional[torch.Tensor] = None,
    lat_flags: Optional[torch.Tensor] = None,
    tap: Optional[torch.Tensor] = None,
    tap_updated: bool = False,
    updated_out: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """``(normed, updated)`` of the row-parallel MoE tail (``[rmsnorm(latent)[:, lo:lo+224] | act] @ tail_weight^T``,
    the RMS on the fp32 latent accumulator) followed by ``allreduce_attn_res_rmsnorm`` of that partial. ``latent`` is
    the whole reduced latent row (16-byte aligned: its rows are bulk-copied), ``act`` the shared-expert activation,
    ``tail_weight`` [7168, 256 + 384] the latent up columns of the slice zero-padded to 256 and the shared down
    projection; ``src_slab`` (int32 [3, 8, 1792]) supplies the latent in buffer ``src_buf``; with ``lat_uc`` /
    ``lat_flags`` (a :class:`K3SandwichLatentExchange`) the kernel sums the ranks' pushed partials itself and
    ``latent`` gives only the shape; with ``tap`` (bf16 [M, 7168], unit column stride, rows a multiple of 8 elements
    apart, 16-byte aligned: e.g. a column slice of a capture buffer) it also stores there the pre-norm attn_res mixture
    rows (a DSpark capture layer's tap) or, with ``tap_updated``, ``updated``; with ``updated_out`` (bf16 [M, 7168],
    contiguous, 16-byte aligned, e.g. the next row of the attention-residual snapshot bank, which this call does not
    read) it stores ``updated`` there instead of a new tensor and returns an empty [0, 7168] in its place; the rest as
    ``k3_sandwich_oproj``."""
    import cuda.bindings.driver as cuda_driver

    from . import k3_sandwich_kernel as kernel

    num_tokens = latent.shape[0]
    num_snapshots = block_residual.shape[0]
    if not supports_tail(latent, act, tail_weight, num_snapshots):
        raise ValueError(
            f"k3_sandwich_tail: unsupported call latent {tuple(latent.shape)}, act {tuple(act.shape)}, weight "
            f"{tuple(tail_weight.shape)}, snapshots {num_snapshots}"
        )
    world = ws_uc.numel() // kernel.buffer_words(1)
    fold = lat_uc is not None
    if fold:
        if src_slab is not None:
            raise ValueError(
                "k3_sandwich_tail: the latent comes from src_slab or from the exchange, not both"
            )
        if (
            lat_flags is None
            or lat_flags.dtype != torch.int32
            or lat_flags.numel() != kernel.LAT_FLAG_WORDS
        ):
            raise ValueError(f"k3_sandwich_tail: lat_flags must be int32 [{kernel.LAT_FLAG_WORDS}]")
        if (lat_uc.dtype != torch.int32 or lat_uc.numel() != kernel.lat_buffer_words(world)
                or not lat_uc.is_contiguous()):  # fmt: skip
            raise ValueError(
                f"k3_sandwich_tail: lat_uc must be the int32 [2, 8, {world}, 1792] exchange buffer"
            )
        if world % 2 != 0 or not (world <= 8 or world % 8 == 0):
            raise ValueError(f"k3_sandwich_tail: the latent exchange takes an even TP of at most 8 or a multiple of 8, "
                             f"not {world}")  # fmt: skip
    elif src_slab is None and latent.data_ptr() % 16 != 0:
        raise ValueError(
            "k3_sandwich_tail: the latent rows must be 16-byte aligned (they are bulk-copied)"
        )
    if tap is not None and (tap.dtype != torch.bfloat16 or tuple(tap.shape) != (num_tokens, kernel.H)
                            or tap.stride(1) != 1 or tap.stride(0) % 8 != 0 or tap.data_ptr() % 16 != 0):  # fmt: skip
        raise ValueError(f"k3_sandwich_tail: tap must be a 16-byte aligned bf16 [{num_tokens}, {kernel.H}] view with "
                         f"unit column stride and a row stride that is a multiple of 8, got {tuple(tap.shape)} "
                         f"{tap.dtype} strides {tuple(tap.stride())}")  # fmt: skip
    if updated_out is not None and (updated_out.dtype != torch.bfloat16
                                    or tuple(updated_out.shape) != (num_tokens, kernel.H)
                                    or not updated_out.is_contiguous()
                                    or updated_out.data_ptr() % 16 != 0):  # fmt: skip
        raise ValueError(f"k3_sandwich_tail: updated_out must be a contiguous, 16-byte aligned bf16 [{num_tokens}, "
                         f"{kernel.H}] tensor, got {tuple(updated_out.shape)} {updated_out.dtype}")  # fmt: skip
    normed = torch.empty(num_tokens, kernel.H, dtype=torch.bfloat16, device=latent.device)
    updated = updated_out if updated_out is not None else torch.empty_like(normed)
    snaps = block_residual if num_snapshots > 0 else latent.new_zeros(1, num_tokens, kernel.H)
    add_prefix = prefix is not None
    slab_words, buf, publish = _slab_args(x_slab, slab_buf, ws_flags)
    if fold:
        src_words, sbuf, x_src = lat_uc.reshape(-1), 0, 2
    else:
        src_words, sbuf, x_src = _src_args(src_slab, src_buf, kernel.LATENT, _words(latent))
    args = (
        _arg(tail_weight),
        _arg(latent),
        _arg(src_words),
        _arg(act),
        _arg(ws_uc),
        _arg(ws_mc),
        _arg(ws_flags),
        _arg(_words(prefix if add_prefix else snaps)),
        _arg(_words(snaps)),
        _arg(_words(res_weight)),
        _arg(_words(rms_weight)),
        _arg(_words(output_rms_weight)),
        _arg(_words(updated)),
        _arg(_words(normed)),
        _arg(slab_words),
        _arg(lat_flags if fold else ws_flags),
        _arg(_tap_words(tap) if tap is not None else ws_flags),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(latent.device).cuda_stream)
    use_pdl = _use_pdl()
    tap_out = 0 if tap is None else 2 if tap_updated else 1
    tap_stride = tap.stride(0) // 2 if tap is not None else kernel.WORDS_PER_ROW
    key = ("tail", world, publish, x_src, tap_out, use_pdl)
    runtime = (
        num_tokens, rank, num_snapshots + 1, int(add_prefix), float(rms_eps), float(output_rms_eps), lo,
        float(lat_eps), buf, sbuf, tap_stride,
    )  # fmt: skip
    consts = (world, publish, x_src, tap_out, use_pdl)
    _compile_and_run(
        kernel.k3_sandwich_tail, key, args, runtime, consts, stream, "k3_sandwich_tail"
    )
    if updated_out is not None:
        return [normed, normed.new_empty((0, kernel.H))]
    return [normed, updated]


@k3_sandwich_tail.register_fake
def _(latent, act, tail_weight, lo, lat_eps, prefix, block_residual, res_weight, rms_weight, output_rms_weight,
      rms_eps, output_rms_eps, ws_uc, ws_mc, ws_flags, rank, x_slab=None, slab_buf=0, src_slab=None,
      src_buf=0, lat_uc=None, lat_flags=None, tap=None, tap_updated=False, updated_out=None):  # fmt: skip
    normed = latent.new_empty((latent.shape[0], tail_weight.shape[0]), dtype=torch.bfloat16)
    if updated_out is not None:
        return [normed, normed.new_empty((0, tail_weight.shape[0]))]
    return [normed, torch.empty_like(normed)]


def supports_plain(x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor,
                   swiglu: bool = False) -> bool:  # fmt: skip
    """Whether the plain sandwich runs this call: bf16, M <= 8, a row-parallel slice [7168, K] with K a multiple of
    128 up to 896 (the drafter's TP16 o_proj: 384; its down projection: 896), residual [M, 7168]; x [M, K], or with
    ``swiglu`` (K 896: one k-tile per cluster CTA) a gate_up output [M, 2 K]."""
    from . import k3_sandwich_kernel as kernel

    k_in = weight.shape[1] if weight.dim() == 2 else -1
    return (
        x.is_cuda
        and x.dtype == weight.dtype == residual.dtype == norm_weight.dtype == torch.bfloat16
        and x.dim() == 2
        and 0 < x.shape[0] <= MAX_TOKENS
        and k_in % kernel.CTA_K == 0
        and 0 < k_in <= kernel.PLAIN_MAX_K
        and (not swiglu or k_in == kernel.CLUSTER * kernel.CTA_K)
        and x.shape[1] == (2 * k_in if swiglu else k_in)
        and tuple(weight.shape) == (kernel.H, k_in)
        and tuple(residual.shape) == (x.shape[0], kernel.H)
        and tuple(norm_weight.shape) == (kernel.H,)
        and x.is_contiguous()
        and weight.is_contiguous()
        and residual.is_contiguous()
    )


@torch.library.custom_op("trtllm::k3_sandwich_plain", mutates_args=("ws_uc", "ws_mc", "ws_flags"))
def k3_sandwich_plain(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    ipc_order: bool = False,
    swiglu: bool = False,
) -> List[torch.Tensor]:
    """``(normed, updated)`` of the row-parallel projection ``x @ weight^T`` followed by the TP all-reduce with the
    residual add and RMSNorm (``AllReduceFusionOp.RESIDUAL_RMS_NORM``): updated = residual + the sum,
    normed = RMSNorm(updated) * norm_weight. ``x`` bf16 [M, K], ``weight`` [7168, K] (K a multiple of 128 up to 896),
    ``residual`` [M, 7168]; ``ws_*`` from a :class:`K3SandwichWorkspace`. With ``swiglu``, ``x`` is a gate_up output
    [M, 2 K] (gate columns first) and the projection is ``silu_and_mul(x) @ weight^T`` with ``k3_ctm_gemv_swiglu``
    split 2's arithmetic (the drafter MLP's down projection). The arithmetic and summation order are those of the
    all-reduce kernel the call replaces: the MNNVL one-shot's, or with ``ipc_order`` the IPC one-shot's
    (``allreduce_fusion_kernel_oneshot_lamport`` with fp32 accumulation, TP <= 8 within one node)."""
    import cuda.bindings.driver as cuda_driver

    from . import k3_sandwich_kernel as kernel

    if not supports_plain(x, weight, residual, norm_weight, swiglu):
        raise ValueError(
            f"k3_sandwich_plain: unsupported call x {tuple(x.shape)} {x.dtype}, weight {tuple(weight.shape)}, "
            f"residual {tuple(residual.shape)}, norm_weight {tuple(norm_weight.shape)}, swiglu {swiglu}"
        )
    num_tokens, k_in = x.shape[0], weight.shape[1]
    world = ws_uc.numel() // kernel.buffer_words(1)
    if ipc_order and world > 8:
        raise ValueError(
            f"k3_sandwich_plain: the IPC one-shot's order is defined for TP <= 8, not {world}"
        )
    order = 2 if ipc_order else 1
    normed = torch.empty(num_tokens, kernel.H, dtype=torch.bfloat16, device=x.device)
    updated = torch.empty_like(normed)
    args = (
        _arg(weight),
        _arg(x),
        _arg(ws_uc),
        _arg(ws_mc),
        _arg(ws_flags),
        _arg(_words(residual)),
        _arg(_words(norm_weight)),
        _arg(_words(updated)),
        _arg(_words(normed)),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    use_pdl = _use_pdl()
    key = ("plain", world, k_in, order, int(swiglu), use_pdl)
    runtime = (num_tokens, rank, float(eps))
    consts = (world, k_in, order, int(swiglu), use_pdl)
    _compile_and_run(
        kernel.k3_sandwich_plain, key, args, runtime, consts, stream, "k3_sandwich_plain"
    )
    return [normed, updated]


@k3_sandwich_plain.register_fake
def _(
    x,
    weight,
    residual,
    norm_weight,
    eps,
    ws_uc,
    ws_mc,
    ws_flags,
    rank,
    ipc_order=False,
    swiglu=False,
):
    normed = x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)
    return [normed, torch.empty_like(normed)]
