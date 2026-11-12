# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLA q_a and kv_a RMSNorms in one CuTe DSL launch, kv normalised in place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common.cuda_stream import raw_stream as _raw_stream
from ...kernels.norm import self_attn_qkv_a_rmsnorm as _kernel

#: GLM-5.2's fused `q_a | kv_a | k_pe` projection: 2048 + 512 + 64 columns.
_Q, _KV, _ROPE = 2048, 512, 64
_ROW = _Q + _KV + _ROPE


#: The row count from which the warp-per-row kernel runs. The kernel module's own
#: threshold is 65; on the target's strided views the one-CTA-per-row kernel is faster
#: below 256 rows and slower above (B300, L2 flushed: 7.2 vs 9.2 us at 65 rows, a tie at
#: 256, 54.3 vs 17.4 us at 7895).
WARP_ROW_MIN_ROWS = 256


@dataclass(frozen=True)
class CompiledQkvARmsnorm:
    """The two kernels `compile` built for one `(device, eps)`: the one-CTA-per-row
    kernel, which serves every call, and the warp-per-row kernel, which the op takes
    from `WARP_ROW_MIN_ROWS` rows when every operand meets its 16-byte contract."""

    cta: Callable
    warp_row: Callable


def compile(device_id: int, eps: float) -> CompiledQkvARmsnorm:
    """Compile both kernels for `device_id` with `eps` baked in. Call once, at load.

    Returns the handle the op takes as `compiled`. `eps` is a compile-time constant
    of the kernels; the kernel's own default (1e-6) is not GLM-5.2's (1e-5), so the
    target passes the checkpoint's value. Both are compiled for strided
    destinations, which is what lets the kv half be written back into the
    projection row. Cached by the kernel module per `(device_id, eps)`, so a second
    call is free; the first costs two CuTe DSL compiles.
    """
    return CompiledQkvARmsnorm(
        cta=_kernel.compile_qkv_a_rmsnorm(device_id, float(eps), inplace_kv_view=True),
        warp_row=_kernel.compile_qkv_a_rmsnorm_warp_row(
            device_id, float(eps), inplace_kv_view=True
        ),
    )


def _served_layout(q_raw, q_weight, ckv_raw, kv_weight) -> bool:
    """Whether the call is the target's in-place shape, which is exactly what the kernel
    module's checks admit for `inplace_kv_view`: bf16 CUDA `[M, 2048]` / `[M, 512]` views
    at row pitch 2624 of one storage, kv starting 2048 elements after q, the parent's rows
    inside that storage, 16-byte aligned rows, and contiguous bf16 weights of 2048 / 512
    on the same device."""
    rows = q_raw.shape[0]
    if (
        q_raw.dtype is not torch.bfloat16
        or ckv_raw.dtype is not torch.bfloat16
        or not q_raw.is_cuda
        or q_raw.shape[1] != _Q
        or tuple(ckv_raw.shape) != (rows, _KV)
        or q_raw.stride() != (_ROW, 1)
        or ckv_raw.stride() != (_ROW, 1)
        or rows <= 0
        or q_raw.data_ptr() % 16
        or q_weight.dtype is not torch.bfloat16
        or kv_weight.dtype is not torch.bfloat16
        or q_weight.shape != (_Q,)
        or kv_weight.shape != (_KV,)
        or not q_weight.is_contiguous()
        or not kv_weight.is_contiguous()
        or q_weight.device != q_raw.device
        or kv_weight.device != q_raw.device
        or ckv_raw.device != q_raw.device
    ):
        return False
    storage = q_raw.untyped_storage()
    if ckv_raw.untyped_storage().data_ptr() != storage.data_ptr():
        return False
    offset = q_raw.storage_offset()
    return ckv_raw.storage_offset() == offset + _Q and offset + rows * _ROW <= storage.nbytes() // 2


def takes_direct(q_raw, q_weight, ckv_raw, kv_weight) -> bool:
    """Whether the op launches the compiled kernel directly rather than through the kernel
    module's `qkv_a_rmsnorm_run`: an eager call (not inside CUDA-graph capture) on the
    target's in-place layout. The module's per-call contract checks (storage identities,
    alias rules, a dozen validators) cost about 20 us of host against 3 us for the launch,
    on every one of a context step's 79 launches; captured decode steps replay without
    host work, so they keep the checked path."""
    return not torch.cuda.is_current_stream_capturing() and _served_layout(
        q_raw, q_weight, ckv_raw, kv_weight
    )


_STREAMS: dict[int, object] = {}


def _stream(device):
    """The `CUstream` of `device`'s current torch stream, built once per stream."""
    handle = _raw_stream(device)
    stream = _STREAMS.get(handle)
    if stream is None:
        stream = _STREAMS[handle] = _kernel.cuda.CUstream(handle)
    return stream


def takes_warp_row(q_raw, q_weight, ckv_raw, kv_weight) -> bool:
    """The warp-per-row kernel asserts 16-byte alignment of every row it touches: data
    pointers and row pitches. The target's views always meet it; anything else takes
    the one-CTA-per-row kernel, which proves only 2-byte alignment."""
    return int(q_raw.shape[0]) >= WARP_ROW_MIN_ROWS and _kernel._use_warp_row_path(
        q_raw, ckv_raw, q_weight, kv_weight
    )


class CuteDslQkvARmsnorm(OpWrapper):
    """`(rmsnorm(q_raw) * q_weight, rmsnorm(ckv_raw) * kv_weight)`, one launch.

    `q_raw` `[M, 2048]` and `ckv_raw` `[M, 512]` are the first two column
    slices of one row-major `[M, 2624]` projection output (`q_a | kv_a |
    k_pe`). The q result is a fresh compact `[M, 2048]` tensor. The kv result
    is written into `ckv_raw` itself and `ckv_raw` is returned, so the
    projection row afterwards holds `[q_a_raw | kv_a_normed | k_pe]` and its
    tail `[kv_a_normed | k_pe]` is the latent the attention reads, with no copy.
    `q_raw` and the `k_pe` columns are not written.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Every GLM-5.2 layer (the MTP layer included) calls this right after the
    # fused projection, at the step's full row count: decode 1..64 rows
    # (32 sequences x MTP 2), prefill chunks up to max_num_tokens 8192. Both
    # kernels are compiled once and the row count is a runtime value, so there
    # is no fallback: every row count runs one of them.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one token", spec=dict(rows=1, eps=1e-5)),
        Cell(why="the full decode batch, 32 sequences x MTP 2", spec=dict(rows=64, eps=1e-5)),
        Cell(
            why="the last row count on the one-CTA-per-row kernel",
            spec=dict(rows=WARP_ROW_MIN_ROWS - 1, eps=1e-5),
        ),
        Cell(
            why="the first row count on the warp-per-row kernel",
            spec=dict(rows=WARP_ROW_MIN_ROWS, eps=1e-5),
        ),
        Cell(
            why="a ragged mixed-batch row count, past the decode sizes",
            spec=dict(rows=777, eps=1e-5),
        ),
        Cell(why="max_num_tokens, the largest prefill chunk", spec=dict(rows=8192, eps=1e-5)),
    )

    note = """
    The target compiles one handle at load with the checkpoint's eps (1e-5) and
    runs it at every row count; there is no stock-op fallback at the call site.

    The handle holds both of the kernel module's variants. Below
    `WARP_ROW_MIN_ROWS` rows, or when an operand's pointer or row pitch is not
    16-byte aligned, the op runs the one-CTA-per-row kernel; otherwise the
    warp-per-row kernel. The two reduce in the same order and return
    bit-identical results. The one-CTA-per-row kernel can prove only 2-byte
    alignment, so it moves bf16 one element at a time: at 7895 rows it issues 8x
    the load requests of the warp-per-row kernel for the same DRAM bytes.

    The kernel validates the in-place layout itself and raises on anything
    other than the exact kv view of a `[M, 2624]` projection with row pitch
    2624. A compact (out-of-place) kv destination is not certified.
    """

    def __call__(
        self,
        q_raw: torch.Tensor,
        q_weight: torch.Tensor,
        ckv_raw: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        compiled: CompiledQkvARmsnorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fn = (
            compiled.warp_row
            if takes_warp_row(q_raw, q_weight, ckv_raw, kv_weight)
            else compiled.cta
        )
        if takes_direct(q_raw, q_weight, ckv_raw, kv_weight):
            # What `qkv_a_rmsnorm_run(..., None, ckv_raw, inplace_kv_view=True)` launches
            # once its checks pass: a fresh compact q output, kv normalised in place, the
            # row pitches, the tensors' stream. Bit-identical to that path.
            q_out = torch.empty((q_raw.shape[0], _Q), dtype=torch.bfloat16, device=q_raw.device)
            fn(
                q_raw,
                q_weight,
                ckv_raw,
                kv_weight,
                q_out,
                ckv_raw,
                _ROW,
                _ROW,
                _Q,
                _ROW,
                _stream(q_raw.device),
            )
            return q_out, ckv_raw
        return _kernel.qkv_a_rmsnorm_run(
            q_raw,
            q_weight,
            ckv_raw,
            kv_weight,
            eps,
            fn,
            None,
            ckv_raw,
            inplace_kv_view=True,
        )

    def reference(
        self,
        q_raw: torch.Tensor,
        q_weight: torch.Tensor,
        ckv_raw: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        compiled: CompiledQkvARmsnorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(q_a, ckv)` as new tensors, computed in fp64 and rounded once.

        Does not write `ckv_raw`: call it before the op, which does. `compiled`
        is ignored.
        """

        def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            xd = x.double()
            inv = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
            return (xd * inv * w.double()).to(x.dtype)

        return rms(q_raw, q_weight), rms(ckv_raw, kv_weight)

    def is_valid(
        self,
        q_raw: torch.Tensor,
        q_weight: torch.Tensor,
        ckv_raw: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        compiled: CompiledQkvARmsnorm,
    ) -> None:
        # The kernel's run function ignores `eps` whenever a handle is passed:
        # eps is baked into the handle. A handle compiled for another eps, for
        # compact destinations, or for another device returns plausible norms
        # (or, for compact destinations, writes kv at the wrong row pitch)
        # without raising. The kernel module caches every handle under its
        # compile key, which is what this reads back, for both variants.
        assert isinstance(compiled, CompiledQkvARmsnorm), (
            "compiled must be the handle this entry's compile() returns"
        )
        for cache, fn, name in (
            (_kernel._COMPILE_CACHE, compiled.cta, "cta"),
            (_kernel._WARP_ROW_COMPILE_CACHE, compiled.warp_row, "warp_row"),
        ):
            keys = [k for k, f in cache.items() if f is fn]
            assert keys, f"compiled.{name} must be a handle from this entry's compile()"
            device_id, compiled_eps, _use_pdl, inplace = keys[0]
            assert inplace, (
                f"compiled.{name} was built for compact outputs; use this entry's compile()"
            )
            assert compiled_eps == float(eps), (
                f"compiled.{name} has eps={compiled_eps} baked in but the call passes "
                f"eps={eps}; the kernel would silently use {compiled_eps}"
            )
            assert device_id == q_raw.device.index, (
                f"compiled.{name} is for device {device_id}, the tensors are on {q_raw.device}"
            )


cute_dsl_qkv_a_rmsnorm = CuteDslQkvARmsnorm()
