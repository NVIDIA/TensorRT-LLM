# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One rank's quarter of the MTP front's `concat(enorm(e), hnorm(h))`, in one CuTe DSL launch."""

from __future__ import annotations

from typing import Callable

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.norm import mtp_norm_shard_h6144_k3072_bf16 as _kernel


def compile(shard_offset: int) -> Callable:
    """Compile the kernel for one shard offset (0 or 3072). Call once, at load.

    Returns the compiled callable the op takes as `compiled`. The offset is a
    compile-time constant, so a rank compiles only its own. Costs one CuTe DSL
    compile; the kernel module caches it per offset.
    """
    return _kernel.compile(shard_offset)


class CuteDslMtpNormShard(OpWrapper):
    """`out = rmsnorm(source, eps)[:, shard_offset : shard_offset + 3072] * w[same cols]`.

    The norm statistic is over the full 6144-wide row of `source`; only 3072
    columns are scaled and written. `source` is the contiguous `[M, 6144]` bf16
    row block, `norm_weight` its `[6144]` weight, `out` a caller-owned
    contiguous `[M, 3072]` bf16 buffer, which the op writes and returns.

    The MTP front concatenates `enorm(e)` and `hnorm(h)` to 12288 columns and
    TP4 splits that into 3072-column quarters: ranks 0 and 1 own the halves of
    `enorm(e)`, ranks 2 and 3 the halves of `hnorm(h)`. So a rank passes
    `e`/enorm or `h`/hnorm as `source`/`norm_weight` and `(rank % 2) * 3072`
    as `shard_offset`, and gets its quarter without forming the concat.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target calls this only at the row counts the MTP eh_proj kernel is
    # compiled for: {1, 2, 4, 8, 16, 24, 32, 48, 64, 128, ..., 8192}. Both
    # offsets run in production, one per pair of ranks.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, one draft row, lower half (ranks 0 and 2)",
            spec=dict(rows=1, shard_offset=0, eps=1e-5),
        ),
        Cell(
            why="decode, one draft row, upper half (ranks 1 and 3)",
            spec=dict(rows=1, shard_offset=3072, eps=1e-5),
        ),
        Cell(
            why="the largest decode bucket, 64 rows, upper half",
            spec=dict(rows=64, shard_offset=3072, eps=1e-5),
        ),
        Cell(
            why="the largest prefill bucket, 8192 rows, lower half",
            spec=dict(rows=8192, shard_offset=0, eps=1e-5),
        ),
    )

    note = """
    The target compiles the handle for its rank's offset at load. It calls the
    op only when the row count is one the eh_proj kernel is compiled for and
    both MTP inputs are contiguous; every other step takes the stock path at
    the call site (two flashinfer RMSNorms, a concat, a cuBLAS GEMM over the
    full 12288 columns).

    eps is fixed at 1e-5 inside the kernel; `is_valid` refuses any other value.
    """

    def __call__(
        self,
        source: torch.Tensor,
        norm_weight: torch.Tensor,
        out: torch.Tensor,
        shard_offset: int,
        eps: float,
        compiled: Callable,
    ) -> torch.Tensor:
        _kernel._validate_tensor_contract(source, norm_weight, out)
        compiled(source, norm_weight, out)
        return out

    def reference(
        self,
        source: torch.Tensor,
        norm_weight: torch.Tensor,
        out: torch.Tensor,
        shard_offset: int,
        eps: float,
        compiled: Callable,
    ) -> torch.Tensor:
        """The shard as a new tensor, computed in fp64 and rounded once.

        `out` is named to mirror `__call__` and not written. `compiled` is
        ignored.
        """
        xd = source.double()
        inv = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
        cols = slice(shard_offset, shard_offset + _kernel.LOCAL_SIZE)
        return (xd[:, cols] * inv * norm_weight[cols].double()).to(source.dtype)

    def is_valid(
        self,
        source: torch.Tensor,
        norm_weight: torch.Tensor,
        out: torch.Tensor,
        shard_offset: int,
        eps: float,
        compiled: Callable,
    ) -> None:
        # eps is a constant of the kernel, not an input: any other value is
        # silently computed with 1e-5.
        assert eps == _kernel.EPSILON, (
            f"the kernel normalizes with eps={_kernel.EPSILON}; eps={eps} would be ignored"
        )
        # The offset is baked into the handle. A handle for the other half writes
        # the wrong 3072 columns -- valid-looking norms of the neighbouring
        # rank's quarter -- without raising. The kernel module caches handles
        # per offset, so the matching one is a cache lookup, not a compile.
        assert compiled is _kernel.compile(shard_offset), (
            f"compiled is not this entry's handle for shard_offset={shard_offset}; "
            "a handle for the other offset writes the other half's columns"
        )


cute_dsl_mtp_norm_shard = CuteDslMtpNormShard()
