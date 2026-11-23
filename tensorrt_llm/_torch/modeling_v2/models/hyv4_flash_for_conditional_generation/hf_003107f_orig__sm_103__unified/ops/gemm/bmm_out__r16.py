# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched matmul writing into a caller-owned output, via the trtllm bmm_out op,
at this checkpoint's latent-to-value projection.

Local variant of ``catalog/gemm/bmm_out.py`` (``__r16``) owned by the Hy4-Flash
unified target: same op call, same reference and same guards, with ``CELLS``
extended by the shapes this target calls it with. The shared entry and its
cells are unchanged; this file is what the target imports.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul


class BmmOut(OpWrapper):
    """`out[i] = a[i] @ b[i]` for every batch index, in place, returns None.

    The point of the op over `torch.bmm` is the caller-owned `out`: both
    shipped call sites pass a transposed view of a buffer the rest of the
    forward already holds, so the product lands where it is needed with no
    copy. That is also the hazard -- see `is_valid`.
    """

    #: Driven on the run's sm_103 device only. The shared entry also carries
    #: SM_100 from its own certification; that arch is not measured here and
    #: this target's routing admits sm_103 alone.
    ARCHS = frozenset({Arch.SM_103})

    # The first four cells are the shared entry's own (deepseek-r1's MLA, batch
    # 128). The rest are this checkpoint's latent-to-value projection: latent
    # 512 -> v_head 256 per head, batch the head count -- 32 sparse heads, 64
    # SWA heads, and the 8/16 a head set splits into per rank -- at one decode
    # row, the 6-row verify width, a 512-row block and the engine's 5120-row
    # prefill chunk, which the 64-head cell also runs at, the envelope's
    # largest workload.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, q_nope -> latent: [H, 1, nope] @ [H, nope, kv_lora]",
            spec=dict(batch=128, m=1, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="decode, latent -> v: [H, 1, kv_lora] @ [H, kv_lora, v_dim]",
            spec=dict(batch=128, m=1, k=512, n=128, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="prefill token count, the same absorbing orientation",
            spec=dict(batch=128, m=2048, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="contiguous out, to separate the op's own correctness from the "
            "transposed-view path the targets happen to use",
            spec=dict(batch=128, m=64, k=128, n=512, dtype=torch.bfloat16, transposed_out=False),
        ),
        Cell(
            why="hy4 decode, latent -> v: 32 sparse heads",
            spec=dict(batch=32, m=1, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 6-row verify step, latent -> v: 32 sparse heads",
            spec=dict(batch=32, m=6, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 512-row block, latent -> v: 32 sparse heads",
            spec=dict(batch=32, m=512, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 5120-token prefill chunk, latent -> v: 32 sparse heads",
            spec=dict(batch=32, m=5120, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 decode, latent -> v: 64 SWA heads",
            spec=dict(batch=64, m=1, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 decode at 8 heads per rank: sparse 32/4 at tp4 or SWA 64/8 at tp8",
            spec=dict(batch=8, m=1, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 512-row block at 8 heads per rank: sparse at tp4 or SWA at tp8",
            spec=dict(batch=8, m=512, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 decode at 16 heads per rank: SWA 64/4 at tp4",
            spec=dict(batch=16, m=1, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="hy4 5120-token prefill chunk through the SWA layers: 64 heads, "
            "the envelope's largest workload",
            spec=dict(batch=64, m=5120, k=512, n=256, dtype=torch.bfloat16, transposed_out=True),
        ),
    )

    note = """
    Local variant (`__r16`) of `catalog/gemm/bmm_out.py`, owned by the Hy4-Flash
    unified target. `out` is written, not accumulated into: its prior contents
    are ignored. The shared entry's four cells stay first -- deepseek-r1's MLA
    geometry -- and the nine below them are this checkpoint's latent-to-value
    projection, which the shared cells do not reach.

    `torch.ops.trtllm.bmm_out` is registered from Python by
    `torch.library.custom_op("trtllm::bmm_out", mutates_args=("out",))` in
    `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`, implemented as
    `torch.bmm(a, b, out=out)`. The kernel that runs is ATen's compiled-in
    cuBLAS bmm: no JIT compilation and nothing compiled for the arch at run
    time.

    Measured on sm_103 against `reference` over the cells below: worst element
    error 1.969 ulp of the row's largest element and 0.0268 ulp of relative
    RMS; the test certifies at 4 ulp element / 2 ulp RMS, and a product over
    another weight measures 505 ulp, 126 times the element band.

    Not certified: batch above 64, rows above 5120 (the engine's largest
    prefill chunk), k or n outside the shared cells' and 512/256, dtypes other
    than bf16, and a non-contiguous `a` or `b` (the op reads them dense).
    """

    def __call__(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        torch.ops.trtllm.bmm_out(a, b, out)
        return out

    def reference(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The product the op writes into `out`, fp32-accumulated.

        `out` is named to mirror `__call__`: the op takes a destination,
        the reference returns a value, and hiding the argument would hide that
        the op mutates one of its inputs.

        TF32 off: torch's default fp32 matmul on this hardware is a TF32
        product, ~1e-3 relative, which would make the reference the inaccurate
        side of the comparison.
        """
        with true_fp32_matmul():
            return (a.float() @ b.float()).to(a.dtype)

    def is_valid(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        # A wrong-shaped `out` is silently resized and re-allocated by the op,
        # detaching it from whatever buffer the caller aliased it with -- the
        # product then lands somewhere nobody reads, and the forward continues
        # with stale data rather than failing.
        assert a.dim() == 3 and b.dim() == 3 and out.dim() == 3, "a, b, out must be 3D"
        assert out.shape == (a.shape[0], a.shape[1], b.shape[2]), (
            "out must be [B, M, N] matching a [B, M, K] and b [B, K, N]"
        )
        # Mixed input dtypes are type-promoted instead of rejected, changing the
        # required out dtype; the catalog exposes only the single-dtype form.
        assert a.dtype == b.dtype == out.dtype, "a, b, out must share one dtype"


bmm_out = BmmOut()
