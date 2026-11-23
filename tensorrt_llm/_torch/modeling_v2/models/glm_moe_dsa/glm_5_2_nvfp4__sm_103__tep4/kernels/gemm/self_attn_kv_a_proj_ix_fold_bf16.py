# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 MLA KV-A projection with the DSA indexer's key / head-weight rows folded in,
at the MTP3 / MTP5 wide decode row counts.

On an indexer layer the decode step runs two GEMMs over the same normalized hidden state
``x [M, 6144]``: the fused ``q_a | kv_a`` projection (``[2624, 6144]``, bf16 out) and the
indexer's key / head-weight projection (``[160, 6144]`` = 128 key rows then 32 head-weight
rows; keys bf16, head weights fp32: ``self_attn_indexer_pre_indexer_proj_...``'s bf16 split
output). Above 64 rows each is latency-bound (B300, cold weights, 192 rows: about 10.8 and
8.8 us in a chain), so this module runs both as ONE persistent GEMM over the stacked weight
``[qkva (2624) ; zero rows (64) ; indexer (160)]`` = ``[2848, 6144]``: the qkva cell's own
schedule (2-CTA ``(128, 64)``, cluster ``(2, 1)``, raster along N), its C is the qkva output
``[M, 2624]`` exactly as the qkva cell writes it (same tiles, same K order: bitwise), and the
MMA tiles from row 2688 (an MMA tile edge; the zero rows pad up to it) are the indexer tiles
of the GEMM's indexer side output (``dense_gemm_persistent.IxSideOutput``).

The indexer tiles accumulate as the split-output projection does: 8 chains (its 8 split-K
ranks), chain ``r`` over the 64-element k-tiles ``r, r + 8, ...``, summed in fp32 in chain
order (its owner's reduction order); keys rounded to bf16 into a ``[rows_max, 128]`` buffer,
head weights as fp32 into ``[rows_max, 32]``. The chain constants are the split kernel's own
(``SPLITK_K_PARTS``, ``K_TILE_BF16``), so keys and head weights are the split's bit for bit.
Measured the same way: about 13.0 / 12.7 us at 192 / 128 rows for both GEMMs.

ABI: ``compile(rows, kb, wq)`` returns a :class:`CompiledPersistentGemm` pinned to ``rows``
(``launch_rows`` takes ``1..rows`` at runtime, each row bitwise the pinned count's), launched
as ``(x [M, 6144], w_ext [2848, 6144], out [M, 2624])`` (bf16, contiguous). ``kb`` / ``wq`` are
the caller's persistent side buffers (``[>= rows, 128]`` bf16, ``[>= rows, 32]`` fp32,
16-byte aligned), whose addresses are compiled in: they must outlive the handle and never move.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import testing
from cutlass.cute.runtime import make_fake_stream

from . import _swapped_dense_gemm_persistent as _sw
from . import dense_gemm_persistent as _dense
from . import self_attn_indexer_pre_indexer_proj_m64_n160_k6144_tf32 as _split

#: The fused q_a | kv_a projection's output columns, the indexer's key and head-weight rows.
QKVA_N = 2_624
IX_KEY = _split.INDEXER_K_DIM
IX_HEADS = _split.INDEXER_WEIGHT_DIM
K = _split.K
#: The first indexer row of the stacked weight: the MMA tile edge (128 rows) at or past
#: QKVA_N, so the qkva tiles stay the qkva cell's and the indexer tiles hold indexer rows only.
IX_ROW0 = 2_688
N_TOTAL = IX_ROW0 + IX_KEY + IX_HEADS
#: The indexer tiles' chains: the split-output projection's split-K ranks and k-tile.
K_PARTS = _split.SPLITK_K_PARTS
K_TILE = _split.K_TILE_BF16
K_CHUNK = K // (K_TILE * K_PARTS)
#: The wide decode row counts with a cell (32 sequences x MTP3 / MTP5 widths 4 / 6).
MTP_DECODE_M_VALUES = (128, 192)
#: The qkva MTP cells' schedule (``self_attn_kv_a_proj_with_mqa_..._l2prefetch``): no PDL role.
SCHEDULE = _sw.GemmSchedule(True, (128, 64), (2, 1), raster_along="n")


def compile(rows: int, kb: torch.Tensor, wq: torch.Tensor) -> _sw.CompiledPersistentGemm:  # noqa: A001
    """Compile the folded GEMM at ``rows`` rows writing its indexer columns to ``kb`` / ``wq``.

    Compiles on zero tensors of the ABI shapes (dynamic layouts: the handle takes any
    ``1..rows`` rows via ``launch_rows``). Raises ``ValueError`` on a side buffer of the wrong
    dtype, shape or alignment, and ``CantImplementError`` if the schedule does not apply.
    """
    rows = int(rows)
    if kb.dtype != torch.bfloat16 or tuple(kb.shape[1:]) != (IX_KEY,) or kb.shape[0] < rows:
        raise ValueError(f"kb must be bf16 [>= {rows}, {IX_KEY}], got {kb.dtype} {tuple(kb.shape)}")
    if wq.dtype != torch.float32 or tuple(wq.shape[1:]) != (IX_HEADS,) or wq.shape[0] < rows:
        raise ValueError(
            f"wq must be fp32 [>= {rows}, {IX_HEADS}], got {wq.dtype} {tuple(wq.shape)}"
        )
    if not (kb.is_contiguous() and wq.is_contiguous()):
        raise ValueError("kb and wq must be contiguous")
    rows_max = min(int(kb.shape[0]), int(wq.shape[0]))
    side = _dense.IxSideOutput(
        main_m=QKVA_N,
        side_m0=IX_ROW0,
        k_cols=IX_KEY,
        w_cols=IX_HEADS,
        k_parts=K_PARTS,
        k_chunk=K_CHUNK,
        kb_addr=kb.data_ptr(),
        wq_addr=wq.data_ptr(),
        rows_max=rows_max,
    )
    dev = kb.device
    x = torch.zeros([rows, K], dtype=torch.bfloat16, device=dev)
    w_ext = torch.zeros([N_TOTAL, K], dtype=torch.bfloat16, device=dev)
    out = torch.empty([rows, QKVA_N], dtype=torch.bfloat16, device=dev)
    a, b, c = _sw._make_runtime_arguments(x, w_ext, out, "swapped")
    mnkl = (N_TOTAL, rows, K, 1)
    cluster_size = SCHEDULE.cluster_shape_mn[0] * SCHEDULE.cluster_shape_mn[1]
    max_active_clusters = _dense.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    gemm = _sw._GlmPersistentDenseGemmKernel(
        cutlass.Float32,
        SCHEDULE.use_2cta_instrs,
        SCHEDULE.mma_tiler_mn,
        SCHEDULE.cluster_shape_mn,
        SCHEDULE.use_tma_store,
        SCHEDULE.swizzle_size,
        SCHEDULE.raster_along,
    )
    gemm.glm_ix_side = side
    # The indexer tiles' chains follow the GEMM's k-tiles: they must be the split's.
    if SCHEDULE.mma_tiler_mn[0] * 21 != IX_ROW0 or K_PARTS * K_CHUNK * K_TILE != K:
        raise testing.CantImplementError("qkva indexer fold: chain / tile constants disagree")
    if not gemm.can_implement(mnkl, a.element_type, b.element_type, c.element_type, "k", "k", "m"):
        raise testing.CantImplementError(f"qkva indexer fold cannot implement mnkl={mnkl}")
    # No kernel name prefix: the prefix is the kernel class's, shared by every persistent GEMM
    # compiled after it, and the qkva cells it replaces carry none either. In a timeline the
    # fold is the qkva-named persistent grid with 46 / 69 clusters at 128 / 192 rows (the qkva cell: 42 / 63).
    compiled = cute.compile(
        _dense.bmm,
        gemm,
        a,
        b,
        c,
        max_active_clusters,
        make_fake_stream(),
        lambda value: value,
    )
    return _sw.CompiledPersistentGemm(x.shape, w_ext.shape, out.shape, "swapped", compiled)
