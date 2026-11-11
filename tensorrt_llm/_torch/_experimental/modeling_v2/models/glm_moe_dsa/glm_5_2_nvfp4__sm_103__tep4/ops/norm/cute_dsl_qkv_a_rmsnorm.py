# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLA q_a and kv_a RMSNorms in one CuTe DSL launch, kv normalised in place."""

from __future__ import annotations

from typing import Callable

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.norm import self_attn_qkv_a_rmsnorm as _kernel

#: GLM-5.2's fused `q_a | kv_a | k_pe` projection: 2048 + 512 + 64 columns.
_Q, _KV, _ROPE = 2048, 512, 64
_ROW = _Q + _KV + _ROPE


def compile(device_id: int, eps: float) -> Callable:
    """Compile the kernel for `device_id` with `eps` baked in. Call once, at load.

    Returns the compiled callable the op takes as `compiled`. `eps` is a
    compile-time constant of the kernel; the kernel's own default (1e-6) is not
    GLM-5.2's (1e-5), so the target passes the checkpoint's value. Compiled for
    strided destinations, which is what lets the kv half be written back into
    the projection row. Cached by the kernel module per `(device_id, eps)`, so a
    second call is free; the first costs one CuTe DSL compile.
    """
    return _kernel.compile_qkv_a_rmsnorm(device_id, float(eps), inplace_kv_view=True)


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
    # (32 sequences x MTP 2), prefill chunks up to max_num_tokens 8192. The
    # handle is compiled once and the row count is a runtime value, so there is
    # no fallback: every row count runs this kernel.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one token", spec=dict(rows=1, eps=1e-5)),
        Cell(why="the full decode batch, 32 sequences x MTP 2", spec=dict(rows=64, eps=1e-5)),
        Cell(
            why="a ragged mixed-batch row count, past the decode sizes",
            spec=dict(rows=777, eps=1e-5),
        ),
        Cell(why="max_num_tokens, the largest prefill chunk", spec=dict(rows=8192, eps=1e-5)),
    )

    note = """
    The target compiles one handle at load with the checkpoint's eps (1e-5) and
    runs it at every row count; there is no stock-op fallback at the call site.

    The handle is the kernel's one-CTA-per-row variant at every row count. The
    kernel module also has a warp-per-row variant for large row counts, which
    its own wrapper selects only when no handle is passed; this entry does not
    use or certify it.

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
        compiled: Callable,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _kernel.qkv_a_rmsnorm_run(
            q_raw,
            q_weight,
            ckv_raw,
            kv_weight,
            eps,
            compiled,
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
        compiled: Callable,
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
        compiled: Callable,
    ) -> None:
        # The kernel's run function ignores `eps` whenever a handle is passed:
        # eps is baked into the handle. A handle compiled for another eps, for
        # compact destinations, or for another device returns plausible norms
        # (or, for compact destinations, writes kv at the wrong row pitch)
        # without raising. The kernel module caches every handle under its
        # compile key, which is what this reads back.
        keys = [k for k, fn in _kernel._COMPILE_CACHE.items() if fn is compiled]
        assert keys, "compiled must be a handle from this entry's compile()"
        device_id, compiled_eps, _use_pdl, inplace = keys[0]
        assert inplace, "compiled was built for compact outputs; use this entry's compile()"
        assert compiled_eps == float(eps), (
            f"compiled has eps={compiled_eps} baked in but the call passes eps={eps}; "
            f"the kernel would silently use {compiled_eps}"
        )
        assert device_id == q_raw.device.index, (
            f"compiled is for device {device_id}, the tensors are on {q_raw.device}"
        )


cute_dsl_qkv_a_rmsnorm = CuteDslQkvARmsnorm()
