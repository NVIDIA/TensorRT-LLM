# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cells of the catalog entries the gpt-oss-120b default target calls.

Each entry's own cells are the op's: tile edges, row counts, the dtypes its
kernel takes. These are the shapes this target calls it with. The entry's
test drives both through `certified_cells`; nothing in the forward reads
them, so a change here reruns the entry tests and nothing of the target's.
"""

from __future__ import annotations

import torch

from ....catalog._op import Cell, OpWrapper
from ....catalog.attention.fused_qk_norm_rope import FusedQkNormRope
from ....catalog.attention.thop_attention import STANDARD, ThopAttention
from ....catalog.gemm.cublas_mm import CublasMm
from ....catalog.moe.mxe4m3_mxe2m1_block_scale_moe_runner import Mxe4m3Mxe2m1BlockScaleMoeRunner
from ....catalog.norm.flashinfer_fused_add_rmsnorm import FlashinferFusedAddRmsnorm
from ....catalog.norm.flashinfer_rmsnorm import FlashinferRmsnorm
from ....catalog.quantization.mxfp8_quantize import Mxfp8Quantize

CELLS: dict[type[OpWrapper], tuple[Cell, ...]] = {
    FusedQkNormRope: (
        Cell(
            why="default target at attention TP8: 8q/1k/1v heads per rank",
            spec=dict(
                tokens=64,
                **{
                    **FusedQkNormRope.GPT_OSS,
                    "num_heads_q": 8,
                    "num_heads_k": 1,
                    "num_heads_v": 1,
                },
            ),
        ),
    ),
    ThopAttention: (
        Cell(
            why="gpt-oss sliding layer: GQA 64q/8kv d64, sinks, window 128",
            spec=dict(
                input_type=STANDARD,
                num_heads=64,
                num_kv_heads=8,
                head_size=64,
                v_head_dim=64,
                sinks=True,
                window=128,
                quant_mode=0,
            ),
        ),
        Cell(
            why="gpt-oss full layer: the same heads with no window",
            spec=dict(
                input_type=STANDARD,
                num_heads=64,
                num_kv_heads=8,
                head_size=64,
                v_head_dim=64,
                sinks=True,
                window=None,
                quant_mode=0,
            ),
        ),
        # The per-rank head counts the gpt-oss default target runs under
        # attention TP: the same code path as the two cells above, at the
        # head counts a rank holds after the split.
        Cell(
            why="gpt-oss default at attention TP8: 8q/1kv per rank, 128-token window",
            spec=dict(
                input_type=STANDARD,
                num_heads=8,
                num_kv_heads=1,
                head_size=64,
                v_head_dim=64,
                sinks=True,
                window=128,
                quant_mode=0,
            ),
        ),
    ),
    CublasMm: (
        Cell(
            why="gpt-oss qkv with fused bias: hidden 2880 -> (64 + 8 + 8) * 64",
            spec=dict(m=16, k=2880, n=5120, dtype=torch.bfloat16, bias=True, out_dtype=None),
        ),
    ),
    Mxe4m3Mxe2m1BlockScaleMoeRunner: (
        Cell(
            why="the gpt-oss target's own call: router entry point, all 128 experts local",
            spec=dict(
                tokens=64,
                routed_by_kernel=True,
                local_expert_offset=0,
                local_num_experts=128,
                **Mxe4m3Mxe2m1BlockScaleMoeRunner.GPT_OSS,
            ),
        ),
        Cell(
            why="default ep4: the kernel routes all 128 ids and runs this rank's window of 32",
            spec=dict(
                tokens=64,
                routed_by_kernel=True,
                local_expert_offset=64,
                local_num_experts=32,
                **Mxe4m3Mxe2m1BlockScaleMoeRunner.GPT_OSS,
            ),
        ),
        Cell(
            why="default ep8: the kernel routes all 128 ids and runs this rank's window of 16",
            spec=dict(
                tokens=64,
                routed_by_kernel=True,
                local_expert_offset=112,
                local_num_experts=16,
                **Mxe4m3Mxe2m1BlockScaleMoeRunner.GPT_OSS,
            ),
        ),
        Cell(
            why="default moe_tp2: 1440 intermediate columns padded to 1536",
            spec=dict(
                tokens=64,
                routed_by_kernel=True,
                local_expert_offset=0,
                local_num_experts=128,
                **{
                    **Mxe4m3Mxe2m1BlockScaleMoeRunner.GPT_OSS,
                    "intermediate": 1440,
                    "intermediate_pad": 1536,
                },
            ),
        ),
        Cell(
            why="default moe_tp4, ranks 0-2: 768 intermediate columns, no padding",
            spec=dict(
                tokens=64,
                routed_by_kernel=True,
                local_expert_offset=0,
                local_num_experts=128,
                **{
                    **Mxe4m3Mxe2m1BlockScaleMoeRunner.GPT_OSS,
                    "intermediate": 768,
                    "intermediate_pad": 768,
                },
            ),
        ),
        Cell(
            why="default moe_tp4, rank 3: 576 real columns inside a 768-wide operand",
            spec=dict(
                tokens=64,
                routed_by_kernel=True,
                local_expert_offset=0,
                local_num_experts=128,
                **{
                    **Mxe4m3Mxe2m1BlockScaleMoeRunner.GPT_OSS,
                    "intermediate": 576,
                    "intermediate_pad": 768,
                },
            ),
        ),
        Cell(
            why="default moe_tp8, ranks 0-6: 384 intermediate columns, no padding",
            spec=dict(
                tokens=64,
                routed_by_kernel=True,
                local_expert_offset=0,
                local_num_experts=128,
                **{
                    **Mxe4m3Mxe2m1BlockScaleMoeRunner.GPT_OSS,
                    "intermediate": 384,
                    "intermediate_pad": 384,
                },
            ),
        ),
        Cell(
            why="default moe_tp8, rank 7: 192 real columns inside a 384-wide operand",
            spec=dict(
                tokens=64,
                routed_by_kernel=True,
                local_expert_offset=0,
                local_num_experts=128,
                **{
                    **Mxe4m3Mxe2m1BlockScaleMoeRunner.GPT_OSS,
                    "intermediate": 192,
                    "intermediate_pad": 384,
                },
            ),
        ),
    ),
    FlashinferFusedAddRmsnorm: (
        Cell(
            why="decode-shaped, gpt-oss-120b hidden",
            spec=dict(shape=(1, 2880), dtype=torch.bfloat16, eps=1e-5),
        ),
    ),
    FlashinferRmsnorm: (
        Cell(
            why="decode-shaped, gpt-oss-120b hidden 2880",
            spec=dict(shape=(1, 2880), dtype=torch.bfloat16, eps=1e-5),
        ),
    ),
    Mxfp8Quantize: (
        Cell(
            why="gpt-oss hidden 2880, linear scales, aligned to 512 -> padded 3072",
            spec=dict(shape=(16, 2880), dtype=torch.bfloat16, swizzled=False, alignment=512),
        ),
    ),
}
