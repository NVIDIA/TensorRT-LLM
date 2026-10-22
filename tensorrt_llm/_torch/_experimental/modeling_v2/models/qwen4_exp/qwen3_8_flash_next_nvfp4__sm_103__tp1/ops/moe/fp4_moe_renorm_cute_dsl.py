# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 MoE layer entered with router logits, on TRT-LLM's CuTe DSL Blackwell
MoE kernels: Renormalize routing (`renorm_moe_routing_op`), the token sort,
a gathered grouped FC1 GEMM with SwiGLU and NVFP4 requantization in its
epilogue, and a grouped FC2 GEMM that scatter-adds the routing-weighted rows
into the output -- the layer `fp4_block_scale_moe_renorm_routed` computes, for
prefill-sized row counts, from a second copy of the experts in this kernel
family's layout."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops as _cdo
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.utils import ActivationType

from .fp4_block_scale_moe_renorm_routed import CERTIFIED_PAIRS, _ckpt
from .fp4_block_scale_moe_renorm_routed import fp4_block_scale_moe_renorm_routed as _routed

#: Token tile of the grouped GEMMs (the kernels' M tile).
TILE_TOKENS = 128
#: The GEMM tactics, pinned: `((mma M, mma N), cluster (M, N), raster along M)`. FC1's is what
#: the engine's autotune warmup picks at every prefill bucket; FC2's likewise, on the finalize
#: kernel run with a K-tile of 128 (`_FC2_MMA_INST_TILE_K`).
FC1_TACTIC = ((128, 256), (1, 1), False)
FC2_TACTIC = ((128, 256), (1, 2), False)
#: MMA instructions per FC2 A/B pipeline stage (K 128): five stages beside the C staging
#: buffer instead of two at the kernel's default K 256, so FC2's short K (640) pipelines.
_FC2_MMA_INST_TILE_K = 2
#: Launches of at least this many rows run FC2 with L2 cache hints (operands EVICT_FIRST,
#: scatter-add EVICT_LAST): a gain at ~16K rows, a small loss at 8K (GB300 micro-bench).
L2_HINT_MIN_ROWS = 12288  # capacity: fixed: dispatch threshold
_FP4 = torch.float4_e2m1fn_x2
_SWIGLU_LIMIT = _cdo._canonicalize_swiglu_limit_scalar(_cdo.SWIGLU_LIMIT_SCALAR_DISABLED)
_FinalizeKernel = _cdo.Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel


class _Fc2Kernel(_FinalizeKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, mma_inst_tile_k=_FC2_MMA_INST_TILE_K, row_owned_epilogue=True, **kwargs
        )


class _Fc2HintKernel(_FinalizeKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            mma_inst_tile_k=_FC2_MMA_INST_TILE_K,
            row_owned_epilogue=True,
            l2_hints=True,
            **kwargs,
        )


class _Fc2Runner(_cdo.Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner):
    kernel_class = _Fc2Kernel
    kernel_cache = dict()
    tuning_config_cache = dict()


class _Fc2HintRunner(_cdo.Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner):
    kernel_class = _Fc2HintKernel
    kernel_cache = dict()
    tuning_config_cache = dict()


_RUNNERS: dict = {}


def _runners(num_experts: int, top_k: int) -> tuple:
    """`(fc1, fc2, fc2 with L2 hints)` for a geometry, built once (they hold compiled kernels)."""
    key = (num_experts, top_k)
    if key not in _RUNNERS:
        _RUNNERS[key] = (
            _cdo.Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner(
                num_experts,
                top_k,
                num_experts,
                0,
                TILE_TOKENS,
                16,
                activation_type=ActivationType.Swiglu,
                swiglu_limit_scalar=_SWIGLU_LIMIT,
            ),
            _Fc2Runner(num_experts, top_k, num_experts, 0, TILE_TOKENS, torch.bfloat16, 16),
            _Fc2HintRunner(num_experts, top_k, num_experts, 0, TILE_TOKENS, torch.bfloat16, 16),
        )
    return _RUNNERS[key]


class Fp4MoeRenormCuteDsl(OpWrapper):
    """One NVFP4 MoE layer whose routing happens inside the call, on the CuTe
    DSL MoE kernels. Same function as `fp4_block_scale_moe_renorm_routed`:
    per token the `top_k` largest logits (ties to the LOWER expert index),
    weighted by an fp32 softmax over just those `top_k` rounded to bf16; the
    experts' SwiGLU FC1, the NVFP4 requantization of its output, FC2, and the
    weighted sum. Returns the combined `[T, H]` bf16 rows.

    Operands, in this kernel family's layout (see `weights.py`): `fc1_w`
    `[E, 2I, H/2]` uint8, `[up ; gate]` rows interleaved in 64-row groups;
    `fc1_s` `[E, 2I, H/16]` e4m3 bytes in that row order, 128x4-swizzled;
    `fc2_w` `[E, H, I/2]` as stored; `fc2_s` `[E, H, I/16]` swizzled.
    `hidden_states` / `hidden_states_scale` are the NVFP4 activation with
    **linear** block scales, quantized with the layer's FC1 activation global
    scale. Scalars: `fc1_alpha` `[E]` dequantizes the FC1 accumulator (the
    routed entry's `output1_scale_gate_scalar`); `fc2_input_scale` `[1]` is the
    FC2 activation's quantization multiplier (the reciprocal of its dequant
    scale), one number for every expert; `fc2_alpha` `[E]` dequantizes the FC2
    accumulator (the routed entry's `output2_scale_scalar`).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="4096 rows: where the target switches to this entry; the sort's cluster kernel",
            spec=_ckpt(4096),
        ),
        Cell(
            why="16377 rows: a con=128 mixed step (16256 context + 121 generation)",
            spec=_ckpt(16377),
        ),
        Cell(
            why="16384 rows: max_num_tokens of the tuned config; the sort's cooperative kernel",
            spec=_ckpt(16384),
        ),
    )

    note = """
    Semantics against `fp4_block_scale_moe_renorm_routed` (trtllm-gen) on the
    same experts: the same routing (both are the `renorm_moe_routing_op`
    function, weights rounded to bf16 once) and the same NVFP4 arithmetic per
    expert (e2m1 x e4m3-block x fp32-alpha products accumulated in fp32, SwiGLU
    on the FC1 accumulator, the FC2 input requantized to NVFP4 with 16-element
    blocks), but not bitwise: the GEMMs' accumulation order differs, and the
    weighted sum over a token's experts is a scatter-add with fp32 atomics, so
    its order -- and the output's last bit -- varies between launches. Both sit
    inside the routed entry's band against its fp32 reference.

    One FC2 activation scale for all experts: the routed entry takes it per
    expert (folded into `output1_scale_scalar`); a checkpoint whose experts
    disagree on `down_proj.input_scale` cannot use this entry as is.

    Kernels (all TRT-LLM's): `trtllm::moe_topk_sort` (routing + tile metadata),
    `trtllm::moe_output_memset_inplace` (zeroes the rows the scatter-add
    touches; on `memset_stream` beside FC1 when given), the gathered FC1 runner
    and the finalize-fused FC2 runner, both with pinned tactics (no AutoTuner
    lookup); FC2's kernel runs with a K-tile of 128, a row-owned epilogue and,
    from `L2_HINT_MIN_ROWS` rows, L2 cache hints. Not CUDA-graph certified
    (prefill steps run eagerly or under the breakable prefill graph).

    Hard limits
      T            certified 4096..16384 (the target calls it from
                   `_CUTE_DSL_MOE_MIN_ROWS` = 4096 rows)
      num_experts  512, top_k 10 (the routed entry's CERTIFIED_PAIRS)
    """

    def __call__(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc2_input_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        memset_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        hidden = hidden_states.shape[1] * 2
        fc1, fc2, fc2_hint = _runners(num_experts, top_k)
        if tokens >= L2_HINT_MIN_ROWS:
            fc2 = fc2_hint
        # Renormalize routing and the tile sort in one launch: the same (expert, weight) pairs
        # (weights rounded to bf16 once) and slot order as `renorm_moe_routing_op` + `moe_sort`.
        t2e, t2lim, e2p, p2e, _, live, weights = torch.ops.trtllm.moe_topk_sort(
            routing_logits, None, num_experts, top_k, None, None, 0, num_experts, None,
            TILE_TOKENS, 1,
        )  # fmt: skip
        scales = weights.float()
        out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=hidden_states.device)

        def memset() -> None:
            torch.ops.trtllm.moe_output_memset_inplace(
                input=out,
                tile_idx_to_mn_limit=t2lim,
                expanded_idx_to_permuted_idx=e2p,
                permuted_idx_to_expanded_idx=p2e,
                num_non_exiting_tiles=live,
                tile_tokens_dim=TILE_TOKENS,
                top_k=top_k,
                ep_size=1,
                enable_alltoall=False,
            )

        # The memset of the rows FC2 scatter-adds into runs beside FC1 on `memset_stream` (a
        # stream nothing else is queued on: the join before FC2 waits for all of it).
        main = torch.cuda.current_stream()
        if memset_stream is not None:
            memset_stream.wait_stream(main)
            with torch.cuda.stream(memset_stream):
                memset()
            out.record_stream(memset_stream)
        h, h_sf = fc1.forward(
            [
                hidden_states.view(_FP4),
                fc1_w.view(_FP4),
                hidden_states_scale.view(torch.uint8).view(tokens, hidden // 16),
                fc1_s.view(torch.uint8),
                fc1_alpha,
                t2e,
                t2lim,
                p2e,
                live,
                fc2_input_scale,
            ],
            tactic=FC1_TACTIC,
        )
        if memset_stream is not None:
            main.wait_stream(memset_stream)
        else:
            memset()
        fc2.forward(
            [
                h.view(_FP4),
                fc2_w.view(_FP4),
                h_sf.view(torch.uint8),
                fc2_s.view(torch.uint8),
                fc2_alpha,
                out,
                t2e,
                t2lim,
                p2e,
                live,
                scales,
            ],
            tactic=FC2_TACTIC,
        )
        return out

    def reference(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        top_k: int,
        up: tuple,
        gate: tuple,
        down: tuple,
        fc2_global_scale: float,
    ) -> torch.Tensor:
        """The routed entry's fp32 reference (routing, then the layer over the
        dequantized `(codes, scales)` per-expert operands); fp32 `[T, H]`."""
        return _routed.reference(
            routing_logits, hidden_states, top_k, up, gate, down, fc2_global_scale
        )

    def is_valid(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc2_input_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        memset_stream: torch.cuda.Stream | None = None,
    ) -> None:
        for name, tensor in (
            ("routing_logits", routing_logits),
            ("hidden_states", hidden_states),
            ("hidden_states_scale", hidden_states_scale),
            ("fc1_w", fc1_w),
            ("fc1_s", fc1_s),
            ("fc2_w", fc2_w),
            ("fc2_s", fc2_s),
        ):
            assert tensor.is_contiguous(), f"{name} must be contiguous"
        assert (num_experts, top_k) in CERTIFIED_PAIRS, (
            f"(num_experts, top_k) = ({num_experts}, {top_k}) is not certified"
        )
        tokens, half_hidden = hidden_states.shape
        assert hidden_states_scale.numel() == tokens * half_hidden * 2 // 16, (
            "hidden_states_scale must hold one linear e4m3 scale a 16-element block"
        )
        assert fc1_alpha.shape == (num_experts,) and fc2_alpha.shape == (num_experts,), (
            "per-expert alphas"
        )
        assert fc2_input_scale.numel() == 1, "one FC2 activation scale for every expert"


fp4_moe_renorm_cute_dsl = Fp4MoeRenormCuteDsl()
