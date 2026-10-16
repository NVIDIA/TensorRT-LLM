# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The absorbed-MLA K-B and V-B batched GEMMs on their CuTe DSL kernels: the decode kernels
at their row-count groups, the prefill kernels at every other row count."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ...kernels.gemm import self_attn_k_b_bmm_h16_m64_n512_k192_bf16 as _k_b
from ...kernels.gemm import self_attn_prefill_kv_b_bmm_bf16 as _prefill
from ...kernels.gemm import self_attn_v_b_bmm_h16_m64_n256_k512_bf16 as _v_b

_MODULES = {"k_b": _k_b, "v_b": _v_b}

#: The two decode row-count groups; one compiled object serves each.
GROUPS = ("m_le_8", "m_ge_16")
#: The prefill kernels serve every row count 1..PREFILL_MAX_ROWS.
PREFILL_MAX_ROWS = _prefill.MAX_ROWS


@dataclass(frozen=True)
class _Layout:
    """`a` is `[heads, M, k]` sliced from token-major `[M, heads, a_row]`
    storage, `out` is `[heads, M, n]` sliced from `[M, heads, out_row]`."""

    heads: int
    n: int
    k: int
    a_row: int
    out_row: int


#: K-B reads the 192-wide nope half of q `[M, 16, 256]` and writes the
#: 512-wide latent half of the fused query `[M, 16, 576]`. V-B reads the
#: attention output `[M, 16, 512]` and writes v `[M, 16, 256]`.
LAYOUTS = {
    "k_b": _Layout(heads=16, n=512, k=192, a_row=256, out_row=576),
    "v_b": _Layout(heads=16, n=256, k=512, a_row=512, out_row=256),
}


@dataclass(frozen=True)
class AbsorbedBmmRunner:
    """One compiled object of one kernel; serves every row count in `rows`.
    `group` is one of `GROUPS`, or "prefill"."""

    kernel: str
    group: str
    rows: Sequence[int]
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]


def _views(kernel: str, m: int, dev: torch.device):
    """Zero `(a, weight, out)` for `m` rows, in the strided layouts the target passes."""
    lay, bf16 = LAYOUTS[kernel], torch.bfloat16
    a = torch.zeros([m, lay.heads, lay.a_row], dtype=bf16, device=dev)[..., : lay.k]
    weight = torch.zeros([lay.heads, lay.n, lay.k], dtype=bf16, device=dev)
    out = torch.empty([m, lay.heads, lay.out_row], dtype=bf16, device=dev)[..., : lay.n]
    return a.transpose(0, 1), weight, out.transpose(0, 1)


def build(kernel: str, group: str, device_id: int) -> AbsorbedBmmRunner:
    """Compile `kernel` ("k_b" or "v_b") for row-count group `group` on
    `device_id`. Call at load.

    Compiles against the strided views the target passes (see `LAYOUTS`) and
    launches once on zeros, so nothing compiles in the forward. Costs one CuTe
    DSL compile per (kernel, group), cached by the kernel module. Raises on an
    unknown kernel or group.
    """
    module = _MODULES[kernel]
    if group not in GROUPS:
        raise ValueError(f"{kernel}: group must be one of {GROUPS}, got {group!r}")
    rows = tuple(module.rows_of_group(group))
    m = rows[-1]
    with torch.cuda.device(device_id):
        launch = module.build(active_m=m)
        launch(*_views(kernel, m, torch.device("cuda", device_id)))
    return AbsorbedBmmRunner(kernel, group, rows, launch)


def build_prefill(kernel: str, device_id: int) -> AbsorbedBmmRunner:
    """The prefill kernel for `kernel` ("k_b" or "v_b") on `device_id`. Call at load.

    One launcher serves every row count 1..PREFILL_MAX_ROWS through two schedules,
    split at `SCHEDULE_SPLIT_ROWS` (2048) rows. Each compiles on its first launch,
    so both are driven here once, at the largest row count each serves, on zero
    views in the target's layout. Costs two CuTe DSL compiles and a transient
    ~0.2 GB of zeros. Raises on an unknown kernel.
    """
    dev = torch.device("cuda", device_id)
    with torch.cuda.device(device_id):
        launch = _prefill.build(kernel)
        for m in (_prefill.SCHEDULE_SPLIT_ROWS, PREFILL_MAX_ROWS):
            launch(*_views(kernel, m, dev))
        torch.cuda.synchronize(device_id)
    return AbsorbedBmmRunner(kernel, "prefill", range(1, PREFILL_MAX_ROWS + 1), launch)


class CuteDslAbsorbedBmm(OpWrapper):
    """`out[h] = a[h] @ weight[h].T` for every head, in place. Returns `out`.

    `a` is `[16, M, K]` and `out` `[16, M, N]`, both the `transpose(0, 1)`
    views of token-major storage that `bmm_out` takes (see `LAYOUTS`).
    `weight` is `[16, N, K]` contiguous: the transpose of the `[16, K, N]`
    operand `bmm_out` takes, which for K-B is a copy the target makes at load.
    Only the first N columns of each `out` row are written.

        k_b   [16, M, 192] x [16, 512, 192] -> [16, M, 512]
        v_b   [16, M, 512] x [16, 256, 512] -> [16, M, 256]
    """

    ARCHS = frozenset({Arch.SM_103})

    # Two compiled objects per kernel serve rows (2, 4, 8) and (16, 32, 48, 64).
    # Cells take each group's ends, plus 48: the one row count that is not a
    # multiple of the large group's 32-row tile. The prefill cells take the row
    # counts the decode groups leave out on both sides of 64 (a mixed step's
    # generation rows are 1..63), and both sides of the prefill split at 2048.
    CELLS: tuple[Cell, ...] = (
        Cell(why="K-B at MTP width 2, the small group's smallest", spec=dict(kernel="k_b", rows=2)),
        Cell(why="K-B at the small group's largest", spec=dict(kernel="k_b", rows=8)),
        Cell(why="K-B at the large group's smallest", spec=dict(kernel="k_b", rows=16)),
        Cell(why="K-B at 48 rows, a partial last row tile", spec=dict(kernel="k_b", rows=48)),
        Cell(why="K-B at the full decode batch, 32 x MTP 2", spec=dict(kernel="k_b", rows=64)),
        Cell(why="V-B at MTP width 2", spec=dict(kernel="v_b", rows=2)),
        Cell(why="V-B at the small group's largest", spec=dict(kernel="v_b", rows=8)),
        Cell(why="V-B at 48 rows, a partial last row tile", spec=dict(kernel="v_b", rows=48)),
        Cell(why="V-B at the full decode batch", spec=dict(kernel="v_b", rows=64)),
        Cell(
            why="prefill K-B at one row: a decode step's lone row and a mixed step's smallest",
            spec=dict(kernel="k_b", rows=1, prefill=True),
        ),
        Cell(
            why="prefill V-B at 63 rows: a mixed step's generation rows, one under a tile",
            spec=dict(kernel="v_b", rows=63, prefill=True),
        ),
        Cell(
            why="prefill K-B at 65 rows, the first context row count past decode",
            spec=dict(kernel="k_b", rows=65, prefill=True),
        ),
        Cell(
            why="prefill V-B at 2048 rows, the small-T schedule's last row count",
            spec=dict(kernel="v_b", rows=2048, prefill=True),
        ),
        Cell(
            why="prefill K-B at 2049 rows, the large-T schedule's first, a ragged tile",
            spec=dict(kernel="k_b", rows=2049, prefill=True),
        ),
        Cell(
            why="prefill V-B at max_num_tokens, 8192 rows",
            spec=dict(kernel="v_b", rows=8192, prefill=True),
        ),
    )

    note = """
    The target builds both decode groups of both kernels at load (row counts 2,
    4, 8, 16, 32, 48 and 64) and both prefill kernels (every row count
    1..8192). A row count above 8192, or every row count outside the decode
    groups if a prefill build failed, goes to `bmm_out` at the target's call
    site, with the `[16, K, N]` weight that op takes.

    Both kernel kinds check every call: row count in the runner's range, and
    every shape and stride of `a` and `out` exactly as compiled. Anything else
    raises. The prefill kernel checks a weight only the first time it meets
    that weight's address, which is what `is_valid` covers.
    """

    def __call__(
        self,
        a: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: AbsorbedBmmRunner,
    ) -> torch.Tensor:
        runner.launch(a, weight, out)
        return out

    def reference(
        self,
        a: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: AbsorbedBmmRunner,
    ) -> torch.Tensor:
        """The product the op writes into `out`, fp32-accumulated (TF32 off),
        rounded once. `out` is not written; `runner` is ignored."""
        with true_fp32_matmul():
            return (a.float() @ weight.float().transpose(1, 2)).to(out.dtype)

    def is_valid(
        self,
        a: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: AbsorbedBmmRunner,
    ) -> None:
        # The prefill kernel caches its weight operand by address and validates a
        # weight only on first sight, so a later tensor at the same address with
        # another shape or layout (the `[16, K, N]` view `bmm_out` takes) runs on
        # the cached `[16, N, K]` operand.
        lay = LAYOUTS[runner.kernel]
        assert (
            tuple(weight.shape) == (lay.heads, lay.n, lay.k)
            and weight.is_contiguous()
            and weight.dtype == torch.bfloat16
        ), (
            f"weight must be the contiguous bf16 [{lay.heads}, {lay.n}, {lay.k}] operand, "
            f"got {tuple(weight.shape)} with strides {weight.stride()}"
        )


cute_dsl_absorbed_bmm = CuteDslAbsorbedBmm()
