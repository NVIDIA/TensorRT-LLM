# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused mixture-of-experts layer: expert permutation + grouped FC1/FC2 GEMMs
with a gated activation between them + routing-weighted combine, in one call.

Everything a target author needs is below and all of it is checked: what the
op computes (`reference`), what it refuses (`is_valid`), how close the two
must be and why (the test's `_compare`), and the range CI drives (`CELLS`).
"""

from __future__ import annotations

from typing import List, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: NVFP4 block size: one e4m3 scale per 16 elements along K.
SF_VEC_SIZE = 16
#: Largest e2m1 magnitude and largest finite e4m3 magnitude.
E2M1_MAX = 6.0
E4M3_MAX = 448.0
#: The 16 e2m1 code points in code order: sign bit 3, exponent bits 2:1,
#: mantissa bit 0.
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _r1(experts_local: int, inter: int, ep_size: int, tp_size: int, **rest) -> dict:
    """One rank's share of the R1-0528 MTP layer's MoE (hidden 7168, top-8 over 256
    experts): `experts_local` of them at `inter` of the 2048-wide intermediate. Spelled
    out in every cell so a cell prints as a complete configuration rather than a delta.
    """
    return dict(
        hidden=7168,
        topk=8,
        experts_local=experts_local,
        inter=inter,
        ep_size=ep_size,
        tp_size=tp_size,
        **rest,
    )


def _step5(
    quant: str,
    experts_local: int,
    inter: int,
    ep_size: int,
    tp_size: int,
    tp_rank: int,
    ep_rank: int,
    num_tokens: int,
) -> dict:
    """One rank's share of a Step-5-Preview routed-expert layer: hidden 4096, top-8
    over 352 experts of intermediate 1536, the post-SiLU clamp at 7.0 on every
    expert. `quant` is `nvfp4` (layers 3-87, W4A4 over a pre-quantized input with
    swizzled block scales) or `bf16` (layers 88-90).
    """
    return dict(
        quant=quant,
        hidden=4096,
        topk=8,
        experts_local=experts_local,
        inter=inter,
        ep_size=ep_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_rank=ep_rank,
        num_tokens=num_tokens,
        swiglu_limit=7.0,
        clamp_after_silu=True,
    )


def _step5_cells() -> tuple[Cell, ...]:
    """The Step-5 sites: each rank's share at tp4 (all 352 experts, I 384, moe tp4),
    tep4/dep4 (88 experts at 1536, ep4) and the disagg dep2 member (176 at 1536,
    ep2), at a decode row count, a gathered decode row count and the 8192-row
    prefill chunk, in both expert precisions; then `_step5_unfused_cells`.
    """
    shares = (
        # (quant, label, experts_local, inter, ep_size, tp_size, tp_rank, ep_rank, token counts)
        ("nvfp4", "tp4 352xI384", 352, 384, 1, 4, 1, 0, (1, 128, 8192)),
        ("nvfp4", "ep4 r0 88xI1536", 88, 1536, 4, 1, 0, 0, (1, 128, 8192)),
        ("nvfp4", "ep4 r3 88xI1536", 88, 1536, 4, 1, 0, 3, (4, 512, 37)),
        ("nvfp4", "ep2 r1 176xI1536", 176, 1536, 2, 1, 0, 1, (2, 256, 8192)),
        ("bf16", "tp4 352xI384", 352, 384, 1, 4, 1, 0, (1, 128, 8192)),
        ("bf16", "ep4 r2 88xI1536", 88, 1536, 4, 1, 0, 2, (1, 512, 8192)),
        ("bf16", "ep2 r1 176xI1536", 176, 1536, 2, 1, 0, 1, (2, 256, 8192)),
    )
    purpose = {
        1: "a single decode row",
        2: "dep2's gathered decode rows (2 ranks x 1)",
        4: "dep4's gathered decode rows (4 ranks x 1)",
        37: "a row count that is a multiple of nothing",
        128: "the largest CUDA-graph decode batch",
        256: "dep2's gathered decode rows (2 ranks x 128)",
        512: "dep4's gathered decode rows (4 ranks x 128)",
        8192: "the prefill chunk the target splits at",
    }
    cells = []
    for quant, label, e_local, inter, ep, tp, tp_rank, ep_rank, counts in shares:
        for t in counts:
            cells.append(
                Cell(
                    why=f"step5 {quant} {label} T={t}: {purpose[t]}",
                    spec=_step5(quant, e_local, inter, ep, tp, tp_rank, ep_rank, t),
                )
            )
    return tuple(cells) + _step5_unfused_cells()


def _step5_unfused_cells() -> tuple[Cell, ...]:
    """The Step-5 sites again with `use_fused_finalize=False`, the call the target
    makes under FORCE_DETERMINISTIC=1: the disagg dep2 share (176 at 1536, ep2),
    both dep4/tep4 expert windows (88 at 1536, ep4) and the tp4 share (352 at 384,
    moe tp4), at decode, gathered-decode and prefill-chunk row counts.
    """
    shares = (
        # (quant, label, experts_local, inter, ep_size, tp_size, tp_rank, ep_rank, token counts)
        ("nvfp4", "ep2 r1 176xI1536", 176, 1536, 2, 1, 0, 1, (2, 256, 8192)),
        ("bf16", "ep2 r1 176xI1536", 176, 1536, 2, 1, 0, 1, (2, 8192)),
        ("nvfp4", "ep4 r3 88xI1536", 88, 1536, 4, 1, 0, 3, (4, 512)),
        ("nvfp4", "ep4 r0 88xI1536", 88, 1536, 4, 1, 0, 0, (8192,)),
        ("nvfp4", "tp4 352xI384", 352, 384, 1, 4, 1, 0, (128,)),
        ("bf16", "ep4 r2 88xI1536", 88, 1536, 4, 1, 0, 2, (37,)),
        ("bf16", "tp4 352xI384", 352, 384, 1, 4, 1, 0, (1,)),
    )
    purpose = {
        1: "a single decode row",
        2: "dep2's gathered decode rows",
        4: "dep4's gathered decode rows",
        37: "a row count that is a multiple of nothing",
        128: "the largest CUDA-graph decode batch",
        256: "dep2's gathered decode rows at batch 128",
        512: "dep4's gathered decode rows at batch 128",
        8192: "the prefill chunk the target splits at",
    }
    cells = []
    for quant, label, e_local, inter, ep, tp, tp_rank, ep_rank, counts in shares:
        for t in counts:
            cells.append(
                Cell(
                    why=f"step5 unfused {quant} {label} T={t}: {purpose[t]}, "
                    "the finalize kernel summing in fixed order",
                    spec=dict(
                        _step5(quant, e_local, inter, ep, tp, tp_rank, ep_rank, t),
                        use_fused_finalize=False,
                    ),
                )
            )
    return tuple(cells)


def _e2m1_rne(a: torch.Tensor) -> torch.Tensor:
    """Round to nearest even on the e2m1 grid, saturating at +-6.

    The grid step is 0.5 below 2, 1 below 4 and 2 above -- the binade step of
    e2m1's three exponents, the subnormal binade sharing the first normal one's.
    `torch.round` is banker's rounding, which is the format's ties-to-even-code
    rule: 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4,
    5.0 -> 4.
    """
    step = torch.where(a.abs() < 2.0, 0.5, torch.where(a.abs() < 4.0, 1.0, 2.0))
    return torch.sign(a) * torch.clamp(torch.round(a.abs() / step) * step, max=E2M1_MAX)


def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """`[..., K/2]` bytes (any dtype viewed as uint8) -> `[..., K]` fp32 e2m1 values,
    element 2i in the low nibble.
    """
    b = packed.contiguous().view(torch.uint8)
    codes = torch.stack([b & 0x0F, b >> 4], dim=-1).reshape(*b.shape[:-1], 2 * b.shape[-1])
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    return table[codes.long()]


def _unswizzle_sf(swizzled: torch.Tensor, groups: int, rows: int, cols: int) -> torch.Tensor:
    """Undo the 128x4 block-scale swizzle: `[groups, rows, cols]` fp32 scales.

    Each group (one expert, or the whole activation) is padded to a multiple of
    128 rows and 4 columns and stored contiguously. Within it the flat offset
    of scale `(m, c)` is `(m//128)*512*(C//4) + (c//4)*512 + (m%32)*16 +
    ((m%128)//32)*4 + (c%4)`, which is the dim order (m//128, c//4, m%32,
    (m%128)//32, c%4) read back here.
    """
    rows_p = (rows + 127) // 128 * 128
    cols_p = (cols + 3) // 4 * 4
    b = swizzled.contiguous().view(torch.uint8).reshape(-1)[: groups * rows_p * cols_p]
    v = b.reshape(groups, rows_p // 128, cols_p // 4, 32, 4, 4).permute(0, 1, 4, 3, 2, 5)
    sf = v.reshape(groups, rows_p, cols_p)[:, :rows, :cols].contiguous()
    return sf.view(torch.float8_e4m3fn).float()


def _requantize_nvfp4(a: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """The FC1 epilogue's NVFP4 quantize, returned as the value FC2's MMA reads.

    Per 16 consecutive columns of the bf16-rounded activation: `vecmax =
    max|a|`, `sf = e4m3(g * (vecmax * (1/6)))`, the scale `1 / (sf * (1/g))`,
    and the data rounded to e2m1 -- returned as `data * sf`, which is `g`
    times the reconstruction of `a`. The scale is clamped to 448 first because
    torch's fp32 -> e4m3 cast emits NaN past it where the kernel saturates.
    """
    rows, cols = a.shape
    blocks = a.reshape(rows, cols // SF_VEC_SIZE, SF_VEC_SIZE)
    vecmax = blocks.abs().amax(dim=-1, keepdim=True)
    sf = (
        (global_scale * (vecmax * torch.tensor(1.0 / E2M1_MAX, device=a.device)))
        .clamp(max=E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    out_scale = torch.where(sf == 0, torch.zeros_like(sf), 1.0 / (sf * (1.0 / global_scale)))
    return (_e2m1_rne(blocks * out_scale) * sf).reshape(rows, cols)


class FusedMoe(OpWrapper):
    """One MoE layer over pre-routed tokens; returns a fresh `[T, hidden]`.

    Fusion boundary: permutation, both GEMMs, the gated activation between
    them, and the routing-weighted combine. Routing itself happens before the
    call (`noaux_tc_op`), and the op\'s own `n_group`/`topk_group` stay inert.

    Of this op\'s ~50 parameters the deepseek-r1 call sets fourteen; the
    Step-5-Preview call adds `input_sf`, `swizzled_input_sf`, `swiglu_limit`,
    `swiglu_clamp_after_silu` and, under FORCE_DETERMINISTIC=1,
    `use_fused_finalize=False`. The rest -- biases, the other quantized
    paths, LoRA, alltoall, min-latency, `swiglu_alpha` / `swiglu_beta` -- are
    left at their defaults and are not certified here.

    Two operand families are certified. The unquantized one: bf16 input and
    `[E, 2I, H]` / `[E, H, I]` bf16 weights with `quant_scales=[]`. The NVFP4
    one (W4A4): the input pre-quantized by `fp4_quantize` -- `[T, H/2]` uint8
    e2m1 pairs with its e4m3 block scales in the swizzled layout as
    `input_sf` -- and int64-packed e2m1 weights `[E, 2I, H/16]` /
    `[E, H, I/16]`, with `quant_scales` the six tensors
    `[fc1_act_global (0-d), fc1_weight_block (int32 [E, 2I, H/64], the e4m3
    scales 128x4-swizzled per expert), fc1_global ([E] = 1 / (input global
    scale * fc1 weight global scale)), fc2_act_global (0-d, the FC2-input
    global scale), fc2_weight_block (int32 [E, H, I/64]), fc2_global ([E] =
    1 / (fc2_act_global * fc2 weight global scale))]`.

    The activation is SwiGLU over `fc1_expert_weights[e] = [up | gate]`. With
    `swiglu_limit` (fp32 `[E]`) the op switches to its clamped SwiGLU, and
    `swiglu_clamp_after_silu=True` selects `min(silu(gate), limit) *
    clamp(up, -limit, limit)` -- Step-5-Preview\'s order; False clamps the
    gate before SiLU instead. `activation_type=5` is Swiglu in *this* op\'s
    enum, which is not the trtllm-gen runner\'s enum of the same name.

    `tp_size` / `tp_rank` change nothing on the certified path: the op reads
    the intermediate width off the FC2 weight, and the TP rank only decides
    which rank adds the FC2 bias, which is absent. A moe_tp split is therefore
    a narrower intermediate in the weights, which is what the I=384, 512 and
    1024 cells certify.
    """

    #: What CI drives: first the R1-0528 MTP layer's MoE at each share the
    #: deepseek default target admits MTP at -- (experts on the rank,
    #: intermediate on the rank) -- and, at the dep4 share, how many tokens
    #: arrive in a chunk and which expert window this rank owns; then
    #: Step-5-Preview's routed-expert sites, with the FC2+finalize fusion on
    #: and then off. The Step-5 cells were driven on sm_103 only; see `note`.
    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: a single token, the shape that dominates serving",
            spec=_r1(64, 2048, ep_size=4, tp_size=1, num_tokens=1, ep_rank=0, tp_rank=0),
        ),
        Cell(
            why="a mid-sized batch between the decode and chunk-bound extremes",
            spec=_r1(64, 2048, ep_size=4, tp_size=1, num_tokens=256, ep_rank=0, tp_rank=0),
        ),
        Cell(
            why="the chunk bound itself: the target splits the gathered token "
            "set at 8192 because that is where this op stops being certified",
            spec=_r1(64, 2048, ep_size=4, tp_size=1, num_tokens=8192, ep_rank=0, tp_rank=0),
        ),
        Cell(
            why="a non-zero expert window -- rank 3 owns global ids [192, 256), "
            "so an off-by-one in the window lands on the wrong experts",
            spec=_r1(64, 2048, ep_size=4, tp_size=1, num_tokens=256, ep_rank=3, tp_rank=0),
        ),
        Cell(
            why="token count not a multiple of anything: the tail path",
            spec=_r1(64, 2048, ep_size=4, tp_size=1, num_tokens=37, ep_rank=1, tp_rank=0),
        ),
        Cell(
            why="moe_tp4 (tp4, dep4 with moe tp4): every expert, a quarter of the "
            "intermediate each, I=512",
            spec=_r1(256, 512, ep_size=1, tp_size=4, num_tokens=256, ep_rank=0, tp_rank=1),
        ),
        Cell(
            why="moe tp2 x ep4 (tp8): 64 experts at half the intermediate, I=1024",
            spec=_r1(64, 1024, ep_size=4, tp_size=2, num_tokens=256, ep_rank=2, tp_rank=1),
        ),
        Cell(
            why="ep8 (dep8): 32 experts per rank, the narrowest window, rank 5 "
            "owns global ids [160, 192)",
            spec=_r1(32, 2048, ep_size=8, tp_size=1, num_tokens=256, ep_rank=5, tp_rank=0),
        ),
    ) + _step5_cells()

    note = """
    Tokens whose expert id falls outside this rank\'s window contribute
    nothing; a token routed entirely outside it comes back as zeros. That is
    the mechanism expert parallelism rests on, not an error, and the
    reference reproduces it.

    Repeated ids in one token\'s top-k are summed, not deduplicated. A
    noaux_tc top-k never repeats an id.

    The autotuner\'s hot path is exercised by test_r1_mtp_autotuned_tactics
    rather than by a cell: what a tuning pass leaves in its cache is the
    tuner\'s behaviour, and `misc/test_autotuner.py` owns it.

    ARCHS: the R1 cells were driven on sm_100 and sm_103. The Step-5 cells
    (`why` starting `step5`) were driven on sm_103 only and are uncertified on
    sm_100 until they run there.

    Step-5-Preview cells. `swiglu_limit` is an fp32 `[E_local]` tensor (7.0
    everywhere) and `swiglu_clamp_after_silu=True`; any of `swiglu_alpha`,
    `swiglu_beta`, `swiglu_limit` switches the op from its plain SwiGLU to
    the clamped one, and without one of them the clamp flag is a loud error.
    At limit 7 the two clamp orders differ by at most 7 * (1 - sigmoid(7)) =
    0.0064 per element, so the cells alone barely tell them apart (0.67 vs
    0.74 RMS ulp on bf16, 0.67 vs 1.72 on NVFP4); the test pins the order at
    limit 0.5, where they are 87-131 RMS ulp apart. Measured with the op\'s
    whole gemm2 tactic space (gemm1\'s tactics are bitwise identical to each
    other), so the warm path after an autotune pass is covered, not only the
    default tactics. Two identical calls were measured bitwise identical on
    the default tactics; the GEMM2 tactics that fuse the finalize are not
    (next paragraph).

    Unfused finalize, the cells whose `why` starts `step5 unfused`:
    `use_fused_finalize=False`, the Step-5 target's call under
    FORCE_DETERMINISTIC=1 (the engine's `moe_disable_finalize_fusion`
    convention). The flag decides the runner's GEMM2 tactic population. The
    fused runner adds tactics whose epilogue adds each expert row into the
    output and skips the finalize kernel -- 168 GEMM2 tactics against 98 on
    NVFP4, 309 against 209 on bf16 -- and nearly all of the added tactics
    were measured run-to-run variable: two identical calls differed on 69-70
    of the 70 added NVFP4 tactics and 97-100 of the 100 bf16 ones at each
    swept cell, T=1 included. The unfused runner has none of them: GEMM2 stores each expanded row in bf16
    and `finalizeMoeRoutingKernel` sums a token's top-k rows in fp32 in slot
    order and rounds once. Certified for: the fused twins' band against the
    same reference (worst 2.0 / 0.692 ulp over the eleven cells); two
    identical calls bitwise identical at every unfused cell; and, at five of
    them, every GEMM1 and every GEMM2 tactic of the unfused runner returning
    the same bits twice -- the answer depends neither on the run nor on the
    autotuner's pick. At the default (untuned) tactic both runners launch the
    finalize kernel and return the same bits; the fusion only ever comes from
    a tuned tactic.

    The flag is fixed per process and dtype triple. The op caches its C++
    runner in `MoERunner.runner_dict` (`custom_ops/torch_custom_ops.py`) under
    a key that leaves `use_fused_finalize` out, and the runner takes the flag
    when it is built, so a call whose flag differs from the first call's on
    that key runs on the first call's runner: an unfused call after a fused
    one can be tuned onto a fusing tactic (measured; strict-xfail detector
    `test_the_flag_reaches_a_cached_runner`). A caller passes one value for
    the whole process; the test gives the unfused calls a runner cache of
    their own.

    NVFP4 preconditions a caller establishes, none checked by the op:
    `input_sf` is the swizzled layout `fp4_quantize(..., is_sf_swizzled_layout
    =True)` writes, with `swizzled_input_sf=True`; the weight block scales are
    `block_scale_interleave`d per expert (the test pins the cells\' layout to
    that op); `fc1_expert_weights[e]` is `[up | gate]` with 2I a multiple of
    128 (384 and 1536 need no padding); `fc1_global` and `fc2_global` fold the
    per-expert weight global scale in. The activation global scales are
    certified as 0-d tensors only; the op also takes them per expert (`[E]`),
    which no cell drives. `fc1_act_global` was measured bitwise inert on a
    pre-quantized input.

    Hard limits (computed, not probed -- the 8192-row envelope is far from
    them): the permuted FC1 output holds T * top_k * 2I elements, past 2^31
    at T = 87382 for I 1536 and T = 349526 for I 384; the permuted FC2 output
    holds T * top_k * hidden, past 2^31 at T = 65536 for hidden 4096.

    Not driven here: CUDA-graph capture of the call (the target captures its
    decode step), `out_tensor`, `enable_alltoall`, and any row count between
    the cells\' (the autotuner keys its picks by power-of-two token buckets).
    """

    def __call__(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        fc1_expert_weights: torch.Tensor,
        fc1_expert_biases: Optional[torch.Tensor],
        fc2_expert_weights: torch.Tensor,
        fc2_expert_biases: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor] = None,
        swizzled_input_sf: bool = True,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        enable_alltoall: bool = False,
        use_deepseek_fp8_block_scale: bool = False,
        use_w4_group_scaling: bool = False,
        use_int8_woq_per_channel: bool = False,
        use_mxfp8_act_scaling: bool = False,
        min_latency_mode: bool = False,
        use_fused_finalize: bool = True,
        tune_max_num_tokens: int = 8192,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        activation_type: int = 5,  # ActivationType.Swiglu
        unpadded_hidden_size: Optional[int] = None,
        out_tensor: Optional[torch.Tensor] = None,
        use_dynamic_fc2_scale: bool = False,
        use_mxfp8_weight_scaling: bool = False,
        fc1_lora_ranks: Optional[torch.Tensor] = None,
        fc1_lora_weight_ptrs: Optional[torch.Tensor] = None,
        fc2_lora_ranks: Optional[torch.Tensor] = None,
        fc2_lora_weight_ptrs: Optional[torch.Tensor] = None,
        gated_lora_ranks: Optional[torch.Tensor] = None,
        gated_lora_weight_ptrs: Optional[torch.Tensor] = None,
        host_request_types: Optional[torch.Tensor] = None,
        host_context_lengths: Optional[torch.Tensor] = None,
        lora_max_low_rank: int = 0,
        fc1_slot_lora_ranks: Optional[torch.Tensor] = None,
        fc1_slot_lora_weight_ptrs: Optional[torch.Tensor] = None,
        fc2_slot_lora_ranks: Optional[torch.Tensor] = None,
        fc2_slot_lora_weight_ptrs: Optional[torch.Tensor] = None,
        gated_slot_lora_ranks: Optional[torch.Tensor] = None,
        gated_slot_lora_weight_ptrs: Optional[torch.Tensor] = None,
        token_to_slot: Optional[torch.Tensor] = None,
        swiglu_clamp_after_silu: bool = False,
    ) -> List[torch.Tensor]:
        """Run one MoE layer over pre-routed tokens.

        Returns `[out]` with `out` a fresh `[num_tokens, hidden_size]` tensor in
        `output_dtype`, or `[]` when `out_tensor` is given (written in place).
        """
        return torch.ops.trtllm.fused_moe(
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc1_expert_biases,
            fc2_expert_weights,
            fc2_expert_biases,
            output_dtype,
            quant_scales,
            input_sf=input_sf,
            swizzled_input_sf=swizzled_input_sf,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            cluster_size=cluster_size,
            cluster_rank=cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4_group_scaling=use_w4_group_scaling,
            use_int8_woq_per_channel=use_int8_woq_per_channel,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            min_latency_mode=min_latency_mode,
            use_fused_finalize=use_fused_finalize,
            tune_max_num_tokens=tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            activation_type=activation_type,
            unpadded_hidden_size=unpadded_hidden_size,
            out_tensor=out_tensor,
            use_dynamic_fc2_scale=use_dynamic_fc2_scale,
            use_mxfp8_weight_scaling=use_mxfp8_weight_scaling,
            fc1_lora_ranks=fc1_lora_ranks,
            fc1_lora_weight_ptrs=fc1_lora_weight_ptrs,
            fc2_lora_ranks=fc2_lora_ranks,
            fc2_lora_weight_ptrs=fc2_lora_weight_ptrs,
            gated_lora_ranks=gated_lora_ranks,
            gated_lora_weight_ptrs=gated_lora_weight_ptrs,
            host_request_types=host_request_types,
            host_context_lengths=host_context_lengths,
            lora_max_low_rank=lora_max_low_rank,
            fc1_slot_lora_ranks=fc1_slot_lora_ranks,
            fc1_slot_lora_weight_ptrs=fc1_slot_lora_weight_ptrs,
            fc2_slot_lora_ranks=fc2_slot_lora_ranks,
            fc2_slot_lora_weight_ptrs=fc2_slot_lora_weight_ptrs,
            gated_slot_lora_ranks=gated_slot_lora_ranks,
            gated_slot_lora_weight_ptrs=gated_slot_lora_weight_ptrs,
            token_to_slot=token_to_slot,
            swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        )

    def is_valid(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        fc1_expert_weights: torch.Tensor,
        fc1_expert_biases: Optional[torch.Tensor],
        fc2_expert_weights: torch.Tensor,
        fc2_expert_biases: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        *unused_args: object,
        **unused_kwargs: object,
    ) -> None:
        """The one input the op takes and answers wrongly.

        On the unquantized high-precision path the store is done in the
        activation dtype while the output buffer is allocated with
        output_dtype, so a mismatch reinterprets the written bits. Observed to
        return plausible-looking wrong values, never to raise -- which is why
        this is a guard and not a comment.
        """
        unquantized = input.dtype in (torch.bfloat16, torch.float16)
        if unquantized and fc1_expert_weights.dtype == input.dtype:
            assert output_dtype == input.dtype, (
                f"output_dtype ({output_dtype}) must equal input.dtype ({input.dtype}) "
                "on the unquantized path; a mismatch is written as raw activation-dtype "
                "bits into an output_dtype buffer and silently gives wrong values"
            )

    def reference(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        fc1_expert_weights: torch.Tensor,
        fc1_expert_biases: Optional[torch.Tensor] = None,
        fc2_expert_weights: torch.Tensor = None,
        fc2_expert_biases: Optional[torch.Tensor] = None,
        output_dtype: torch.dtype = torch.bfloat16,
        quant_scales: Optional[List[torch.Tensor]] = None,
        input_sf: Optional[torch.Tensor] = None,
        swizzled_input_sf: bool = True,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        *unused_args: object,
        ep_size: int = 1,
        ep_rank: int = 0,
        swiglu_clamp_after_silu: bool = False,
        **unused_kwargs: object,
    ) -> torch.Tensor:
        """What the op computes, in plain torch, fp32 through both GEMMs.

        Takes the arguments that change the result rather than mirroring
        `__call__`. The other ~40 are inert on the certified path -- a
        reference that accepted them would have to ignore them, which reads
        as coverage it does not have. `swiglu_alpha` / `swiglu_beta` and the
        biases are named only to be refused: no cell supplies them.

        `fc1_expert_weights[e]` is `[up | gate]` stacked on dim 0. Ids are
        global; this rank owns `[ep_rank * E_local, (ep_rank + 1) * E_local)`
        and everything outside contributes nothing, which is how expert
        parallelism tiles the routing space.

        The op stores the FC1 GEMM output and the activation in its 16-bit
        type (the input dtype, or `output_dtype` on the NVFP4 path), and so
        does this -- measured: leaving the FC1 rounding out moves the bf16
        cells from 0.67 to 1.05 RMS ulp and the NVFP4 ones to 2.8-7.4.

        The NVFP4 path is read from the packed operands themselves: the e2m1
        codes, the swizzled e4m3 block scales and the global scales in
        `quant_scales`, so the layout is part of what is checked. The
        activation is then requantized to NVFP4 under `fc2_act_global` exactly
        as the op\'s activation epilogue does; that requantization is the one
        lossy stage between the GEMMs. `fc1_act_global` is not read: it scales
        a bf16 input the op quantizes itself, and the certified input arrives
        quantized (measured bitwise inert there).
        """
        assert swiglu_alpha is None and swiglu_beta is None, "no cell drives alpha / beta"
        assert fc1_expert_biases is None and fc2_expert_biases is None, "no cell drives biases"
        nvfp4 = fc1_expert_weights.dtype == torch.int64
        num_tokens = input.shape[0]
        num_local = fc1_expert_weights.shape[0]
        hidden = fc2_expert_weights.shape[1]
        dev = input.device
        if nvfp4:
            assert input.dtype == torch.uint8 and input_sf is not None and swizzled_input_sf
            two_inter = fc1_expert_weights.shape[1]
            inter = two_inter // 2
            x_sf = _unswizzle_sf(input_sf, 1, num_tokens, hidden // SF_VEC_SIZE)[0]
            x = _unpack_e2m1(input) * x_sf.repeat_interleave(SF_VEC_SIZE, dim=1)
            fc1_sf = _unswizzle_sf(quant_scales[1], num_local, two_inter, hidden // SF_VEC_SIZE)
            fc2_sf = _unswizzle_sf(quant_scales[4], num_local, hidden, inter // SF_VEC_SIZE)
            fc1_global, fc2_act_global, fc2_global = (
                quant_scales[2].float(),
                quant_scales[3].float(),
                quant_scales[5].float(),
            )
            stage_dtype = output_dtype
        else:
            x = input.float()
            stage_dtype = input.dtype
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=dev)
        with true_fp32_matmul():
            for local_e in range(num_local):
                mask = token_selected_experts == ep_rank * num_local + local_e
                tok, slot = mask.nonzero(as_tuple=True)
                if tok.numel() == 0:
                    continue
                if nvfp4:
                    w1 = _unpack_e2m1(fc1_expert_weights[local_e]) * fc1_sf[
                        local_e
                    ].repeat_interleave(SF_VEC_SIZE, dim=1)
                    h = (x[tok] @ w1.t()) * fc1_global[local_e]
                else:
                    h = x[tok] @ fc1_expert_weights[local_e].float().t()
                # the op stores the FC1 GEMM output in its 16-bit type
                h = h.to(stage_dtype).float()
                h_up, h_gate = h.chunk(2, dim=1)
                if swiglu_limit is None:
                    act = h_gate * torch.sigmoid(h_gate) * h_up
                else:
                    limit = swiglu_limit[local_e].float()
                    up = torch.minimum(torch.maximum(h_up, -limit), limit)
                    if swiglu_clamp_after_silu:
                        gate = torch.minimum(h_gate * torch.sigmoid(h_gate), limit)
                    else:
                        g = torch.minimum(h_gate, limit)
                        gate = g * torch.sigmoid(g)
                    act = gate * up
                # the op materializes the FC1 activation in its 16-bit type
                act = act.to(stage_dtype).float()
                if nvfp4:
                    g2 = fc2_act_global if fc2_act_global.dim() == 0 else fc2_act_global[local_e]
                    w2 = _unpack_e2m1(fc2_expert_weights[local_e]) * fc2_sf[
                        local_e
                    ].repeat_interleave(SF_VEC_SIZE, dim=1)
                    y = (_requantize_nvfp4(act, g2) @ w2.t()) * fc2_global[local_e]
                else:
                    y = act @ fc2_expert_weights[local_e].float().t()
                weight = (
                    torch.ones(tok.numel(), device=dev)
                    if token_final_scales is None
                    else token_final_scales[tok, slot].float()
                )
                out.index_add_(0, tok, y * weight.unsqueeze(1))
        return out.to(output_dtype)


fused_moe = FusedMoe()
