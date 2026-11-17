# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 MLA Q-B projection and the DSA indexer's WQ-B projection in one launch, at the
MTP3 / MTP5 wide decode row counts.

On an indexer layer the decode step runs two GEMMs over the same normalized ``q_a [M, 2048]``:
q_b (``[4096, 2048]``, this rank's 16 heads x 256) and the indexer's wq_b (``[4096, 2048]``,
32 heads x 128), each on its MTP decode cell (``self_attn_q_b_proj_...`` /
``self_attn_indexer_wq_b_...``: persistent 2-CTA, PDL wait / signal role with the weight
loaded ahead of the wait). Above 64 rows each is latency-bound (B300, cold weights, 192 rows:
about 5.8 us each, 11.4 for the pair in a chain), so this module runs them as ONE persistent
GEMM with a batch of two: A is the two weights stacked ``[2, 4096, 2048]``, C the two outputs
``[2, M, 4096]`` (``C[0]`` = q, ``C[1]`` = the indexer q, each contiguous, as the cells write
them), and B the one ``q_a``, read by both batches (``glm_b_broadcast_l``).

Every output element is one fp32 accumulation over K = 2048 in the cells' K order (2-CTA MMA,
no split-K): the N tile does not change it, so the outputs are the two cells' bit for bit. The
N tile is the whole decode batch here (128 / 192 rows: 64 clusters, one wave), where the cells
use 64 / 96-row tiles. Measured the same way: about 7.7 / 8.4 us at 128 / 192 rows for both.

ABI: ``compile(rows)`` returns a :class:`CompiledQbIxQb` pinned to ``rows`` whose
``launch_rows(q_a, w2, out2)`` takes ``1..rows`` rows at runtime (each row bitwise the pinned
count's: dynamic layouts, the tiles past the last row load zeros and store nothing):
``q_a [M, 2048]`` bf16 (unit stride along K), ``w2 [2, 4096, 2048]`` and ``out2 [2, M, 4096]``
bf16 contiguous.
"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import testing
from cutlass.cute.runtime import make_fake_stream
from cutlass.utils import create_cute_tensor_for_fp8

from . import _swapped_dense_gemm_pdl_chain as _chain
from . import dense_gemm_persistent as _dense

#: q_b's and wq_b's output features, their shared input features.
N = 4_096
K = 2_048
#: The wide decode row counts with a cell (32 sequences x MTP3 / MTP5 widths 4 / 6).
MTP_DECODE_M_VALUES = (128, 192)
#: Per cell: 2-CTA, MMA tile M 128 (two CTAs of 64 weight rows), N = the whole batch, raster
#: along N; the PDL role of the q_b / wq_b MTP cells (weight ahead of the wait, signal).
SCHEDULE_BY_M = {
    128: _chain.GemmSchedule(True, (128, 128), (2, 1), raster_along="n"),
    192: _chain.GemmSchedule(True, (128, 192), (2, 1), raster_along="n"),
}


def _arguments(q_a: torch.Tensor, w2: torch.Tensor, out2: torch.Tensor):
    """Dynamic-layout views in the swapped orientation, batch first: A ``(2, N, K)``, B
    ``(1, K, M)`` (one batch, shared), C ``(2, N, M)``."""
    a = create_cute_tensor_for_fp8(w2.detach(), cutlass.BFloat16, 2)
    b = create_cute_tensor_for_fp8(q_a.unsqueeze(0).transpose(1, 2), cutlass.BFloat16, 1)
    c = create_cute_tensor_for_fp8(out2.transpose(1, 2), cutlass.BFloat16, 1)
    return a, b, c


@dataclass(frozen=True)
class CompiledQbIxQb:
    """The fused GEMM compiled at ``rows`` rows."""

    rows: int
    compiled: object

    def launch_rows(self, q_a: torch.Tensor, w2: torch.Tensor, out2: torch.Tensor) -> None:
        """Launch on ``q_a``'s ``1..rows`` rows on the current stream.

        :raises ValueError: on another shape, or more rows than compiled.
        """
        m = int(q_a.shape[0])
        if (
            not 0 < m <= self.rows
            or tuple(q_a.shape[1:]) != (K,)
            or q_a.stride(-1) != 1
            or tuple(w2.shape) != (2, N, K)
            or tuple(out2.shape) != (2, m, N)
            or not (w2.is_contiguous() and out2.is_contiguous())
        ):
            raise ValueError(
                f"q_b | wq_b fold at {self.rows} rows: got q_a {tuple(q_a.shape)} "
                f"{tuple(q_a.stride())}, w2 {tuple(w2.shape)}, out2 {tuple(out2.shape)}"
            )
        a, b, c = _arguments(q_a, w2, out2)
        self.compiled(a, b, c, cuda.CUstream(torch.cuda.current_stream().cuda_stream))


def compile(rows: int, device: torch.device) -> CompiledQbIxQb:  # noqa: A001
    """Compile the fused GEMM for ``rows`` (one of ``MTP_DECODE_M_VALUES``) on zero tensors.

    :raises ValueError: for a row count without a schedule; ``CantImplementError`` if the
        schedule does not apply.
    """
    rows = int(rows)
    if rows not in SCHEDULE_BY_M:
        raise ValueError(f"q_b | wq_b fold: no schedule at {rows} rows ({sorted(SCHEDULE_BY_M)})")
    schedule = SCHEDULE_BY_M[rows]
    q_a = torch.zeros([rows, K], dtype=torch.bfloat16, device=device)
    w2 = torch.zeros([2, N, K], dtype=torch.bfloat16, device=device)
    out2 = torch.empty([2, rows, N], dtype=torch.bfloat16, device=device)
    a, b, c = _arguments(q_a, w2, out2)
    cluster_size = schedule.cluster_shape_mn[0] * schedule.cluster_shape_mn[1]
    max_active_clusters = _dense.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    gemm = _chain.make_pdl_chain_kernel(schedule, _chain.WAIT_SIGNAL_PRELOAD_WEIGHT_ROLE)
    gemm.glm_b_broadcast_l = True
    if not gemm.can_implement(
        (N, rows, K, 2), a.element_type, b.element_type, c.element_type, "k", "k", "m"
    ):
        raise testing.CantImplementError(f"q_b | wq_b fold cannot implement {rows} rows")
    # No kernel name prefix (the prefix is the kernel class's, shared by later compiles); in a
    # timeline the fold is the persistent pdl_chain grid of 64 clusters at 128 / 192 rows.
    compiled = cute.compile(
        _dense.bmm, gemm, a, b, c, max_active_clusters, make_fake_stream(), lambda value: value
    )
    return CompiledQbIxQb(rows, compiled)
