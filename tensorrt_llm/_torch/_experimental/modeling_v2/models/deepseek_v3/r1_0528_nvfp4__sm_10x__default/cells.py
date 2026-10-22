# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cells of the catalog entries the DeepSeek-R1-0528 NVFP4 default target calls.

Each entry's own cells are the op's: tile edges, row counts, the dtypes its
kernel takes. These are the shapes this target calls it with. The entry's
test drives both through `certified_cells`; nothing in the forward reads
them, so a change here reruns the entry tests and nothing of the target's.
"""

from __future__ import annotations

import torch

from ....catalog._op import Cell, OpWrapper
from ....catalog.activation.flashinfer_silu_and_mul import FlashinferSiluAndMul
from ....catalog.attention.mla_rope_append_paged_kv_assign_q import MlaRopeAppendPagedKvAssignQ
from ....catalog.attention.mla_rope_generation import Q_SCALING_R1, MlaRopeGeneration
from ....catalog.attention.thop_attention import (
    MLA_CONTEXT,
    MLA_GENERATION,
    QM_FP8_KV_CACHE,
    ThopAttention,
)
from ....catalog.gemm.bmm_out import BmmOut
from ....catalog.gemm.cublas_mm import CublasMm
from ....catalog.gemm.nvfp4_gemm import Nvfp4Gemm
from ....catalog.moe.fp4_block_scale_moe_runner import Fp4BlockScaleMoeRunner
from ....catalog.moe.noaux_tc_op import NoauxTcOp
from ....catalog.norm.flashinfer_fused_add_rmsnorm import FlashinferFusedAddRmsnorm
from ....catalog.norm.flashinfer_rmsnorm import FlashinferRmsnorm
from ....catalog.quantization.fp4_quantize import Fp4Quantize

CELLS: dict[type[OpWrapper], tuple[Cell, ...]] = {
    FlashinferSiluAndMul: (
        Cell(
            why="deepseek default at attention TP8: the shared expert's gate+up, 2 x 2048/8",
            spec=dict(shape=(16, 512), dtype=torch.bfloat16),
        ),
    ),
    MlaRopeAppendPagedKvAssignQ: (
        Cell(
            why="deepseek default at attention TP8: 16 heads per rank, production context",
            spec=dict(
                cached=[0, 32, 100],
                new=[37, 64, 300],
                tokens_per_block=32,
                **dict(MlaRopeAppendPagedKvAssignQ._R1, head_num=16),
            ),
        ),
    ),
    MlaRopeGeneration: (
        Cell(
            why="deepseek default at attention TP8: 16 heads per rank, production decode",
            spec=dict(
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                q_scaling=Q_SCALING_R1,
                write_scale=None,
                read_scale=None,
                **dict(MlaRopeGeneration._R1, num_heads=16),
            ),
        ),
        Cell(
            why="deepseek default MTP3 at attention TP8: P=4 (draft length 3 + 1) at 16 heads",
            spec=dict(
                predicted_tokens_per_seq=4,
                tokens_per_block=32,
                q_scaling=Q_SCALING_R1,
                write_scale=None,
                read_scale=None,
                **dict(MlaRopeGeneration._R1, num_heads=16),
            ),
        ),
    ),
    ThopAttention: (
        Cell(
            why="deepseek MLA context: explicit k/v, qk 576 -> v 128, fp8 pool",
            spec=dict(
                input_type=MLA_CONTEXT,
                num_heads=128,
                num_kv_heads=128,
                head_size=576,
                v_head_dim=128,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        Cell(
            why="deepseek MLA generation: one kv head, latent 576 -> 512, fp8 pool",
            spec=dict(
                input_type=MLA_GENERATION,
                num_heads=128,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        # The per-rank head counts the deepseek default target runs under
        # attention TP: the same two MLA code paths as the cells above, at the
        # 128 heads split 2, 4 or 8 ways.
        Cell(
            why="deepseek default at attention TP8: MLA context, 16 heads per rank",
            spec=dict(
                input_type=MLA_CONTEXT,
                num_heads=16,
                num_kv_heads=16,
                head_size=576,
                v_head_dim=128,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        Cell(
            why="deepseek default at attention TP2: MLA generation, 64 heads per rank",
            spec=dict(
                input_type=MLA_GENERATION,
                num_heads=64,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        Cell(
            why="deepseek default at attention TP8: MLA generation, 16 heads per rank",
            spec=dict(
                input_type=MLA_GENERATION,
                num_heads=16,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        # MTP3 under attention TP: the verify step's taller query block (draft
        # length 3 + 1 rows per sequence) at a split head count. The P > 1 mask
        # itself is pinned at 128 heads by the MTP cases in the test.
        Cell(
            why="deepseek default MTP3 at attention TP8: MLA generation, 16 heads, P=4",
            spec=dict(
                input_type=MLA_GENERATION,
                num_heads=16,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
                predicted_tokens_per_seq=4,
            ),
        ),
    ),
    BmmOut: (
        Cell(
            why="deepseek default at attention TP8, decode absorb: 16 heads per rank",
            spec=dict(batch=16, m=1, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="deepseek default at attention TP4, decode expand: 32 heads per rank",
            spec=dict(batch=32, m=1, k=512, n=128, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="deepseek default at attention TP2, prefill-height absorb: 64 heads per rank",
            spec=dict(batch=64, m=2048, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
    ),
    CublasMm: (
        Cell(
            why="deepseek q_a, no bias: hidden 7168 -> q_lora 1536",
            spec=dict(m=16, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek router: bf16 in, fp32 logits out, 256 experts",
            spec=dict(
                m=16, k=7168, n=256, dtype=torch.bfloat16, bias=False, out_dtype=torch.float32
            ),
        ),
        Cell(
            why="deepseek default q_b at attention TP4: q_lora 1536 -> 32 heads x 192",
            spec=dict(m=16, k=1536, n=6144, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek default kv_b at attention TP8: kv_lora 512 -> 16 heads x 256",
            spec=dict(m=64, k=512, n=4096, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek default o_proj at attention TP8: 16 heads x 128 -> hidden 7168",
            spec=dict(m=16, k=2048, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        # The MTP layer's own GEMMs. Its eh_proj is replicated on every rank; its
        # shared expert is bf16 (the NVFP4 export skips layer 61), so it runs
        # through cublas_mm rather than nvfp4_gemm, split over attention TP like
        # the trunk's.
        Cell(
            why="deepseek MTP eh_proj: [enorm(e) | hnorm(h)] 2 x 7168 -> hidden 7168",
            spec=dict(m=16, k=14336, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek MTP shared expert gate_up, whole under attention DP: 7168 -> 2 x 2048",
            spec=dict(m=16, k=7168, n=4096, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek MTP shared expert gate_up at attention TP8: 7168 -> 2 x 256",
            spec=dict(m=16, k=7168, n=512, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek MTP shared expert down at attention TP8: 256 -> hidden 7168",
            spec=dict(m=16, k=256, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
    ),
    Nvfp4Gemm: (
        Cell(
            why="deepseek default at attention TP8, shared expert gate+up: 2 x 2048/8 rows",
            spec=dict(m=16, k=7168, n=512, bias=False),
        ),
        Cell(
            why="deepseek default at attention TP8, shared expert down: K = 2048/8",
            spec=dict(m=16, k=256, n=7168, bias=False),
        ),
    ),
    Fp4BlockScaleMoeRunner: (
        Cell(
            why="deepseek default at tp2: all 256 experts, intermediate split 2 ways (1024)",
            spec=dict(
                tokens=64,
                local_expert_offset=0,
                local_num_experts=256,
                **dict(Fp4BlockScaleMoeRunner._R1, intermediate=1024),
            ),
        ),
        Cell(
            why="deepseek default at tp4: all 256 experts, intermediate split 4 ways (512)",
            spec=dict(
                tokens=64,
                local_expert_offset=0,
                local_num_experts=256,
                **dict(Fp4BlockScaleMoeRunner._R1, intermediate=512),
            ),
        ),
        Cell(
            why="deepseek default at tp8: all 256 experts, intermediate split 8 ways (256)",
            spec=dict(
                tokens=64,
                local_expert_offset=0,
                local_num_experts=256,
                **dict(Fp4BlockScaleMoeRunner._R1, intermediate=256),
            ),
        ),
        Cell(
            why="deepseek default at ep8 (tp8 + ep8 and dep8): the last 32-wide window",
            spec=dict(
                tokens=64,
                local_expert_offset=224,
                local_num_experts=32,
                **Fp4BlockScaleMoeRunner._R1,
            ),
        ),
        Cell(
            why="deepseek default at dep16: a 16-wide window",
            spec=dict(
                tokens=64,
                local_expert_offset=112,
                local_num_experts=16,
                **Fp4BlockScaleMoeRunner._R1,
            ),
        ),
    ),
    NoauxTcOp: (
        Cell(
            why="the default target's expert chunk, which its attention-DP gather fills",
            spec=dict(tokens=8192, experts=256, n_group=8, topk_group=4, topk=8, scale=2.5),
        ),
        Cell(
            why="deepseek MTP router: fp32 logits, so the weights are the fp32 "
            "token_final_scales fused_moe needs",
            spec=dict(
                tokens=256,
                experts=256,
                n_group=8,
                topk_group=4,
                topk=8,
                scale=2.5,
                logits_dtype=torch.float32,
            ),
        ),
    ),
    FlashinferFusedAddRmsnorm: (
        Cell(
            why="decode-shaped, deepseek-r1 hidden",
            spec=dict(shape=(1, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
    ),
    FlashinferRmsnorm: (
        Cell(
            why="decode-shaped, deepseek-r1 hidden 7168",
            spec=dict(shape=(1, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="deepseek q_lora rank 1536, the q_norm this op is called on",
            spec=dict(shape=(16, 1536), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="deepseek kv_lora rank 512, the narrowest row any target asks for",
            spec=dict(shape=(16, 512), dtype=torch.bfloat16, eps=1e-6),
        ),
    ),
    Fp4Quantize: (
        Cell(
            why="deepseek default at attention TP8: the shared expert's down input, 2048/8 wide",
            spec=dict(shape=(16, 256), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
    ),
}
