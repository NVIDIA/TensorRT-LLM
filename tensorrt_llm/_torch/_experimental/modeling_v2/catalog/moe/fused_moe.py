# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused mixture-of-experts layer: expert permutation + grouped FC1/FC2 GEMMs
with a gated activation between them + routing-weighted combine, in one call.

Everything a target author needs is below and all of it is checked: what the
op computes (`reference`), what it refuses (`is_valid`), how close the two
must be and why (`compare`), and the range CI drives (`CELLS`).
"""

from __future__ import annotations

from typing import List, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: The one geometry this entry is certified at: the R1-0528 MTP layer's MoE,
#: which is the only thing that calls this op. Repeated into every cell so a
#: cell prints as a complete configuration rather than a delta.
_R1 = dict(hidden=7168, inter=2048, experts_local=64, ep_size=4, topk=8)


class _FusedMoe(OpWrapper):
    """One MoE layer over pre-routed tokens; returns a fresh `[T, hidden]`.

    Fusion boundary: permutation, both GEMMs, the gated activation between
    them, and the routing-weighted combine. Routing itself happens before the
    call (`noaux_tc_op`), and the op\'s own `n_group`/`topk_group` stay inert.

    Of this op\'s ~50 parameters the shipped call sets twelve. The rest --
    biases, the quantized paths, LoRA, alltoall, min-latency, the swiglu
    alpha/beta/limit triple -- are left at their defaults and are not
    certified here. `activation_type=5` is Swiglu in *this* op\'s enum, which
    is not the trtllm-gen runner\'s enum of the same name.
    """

    #: What CI drives. The geometry is fixed -- one op, one caller, one shape
    #: family -- so the cells vary what the caller actually varies: how many
    #: tokens arrive in a chunk, and which expert window this rank owns.
    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: a single token, the shape that dominates serving",
            spec=dict(**_R1, num_tokens=1, ep_rank=0),
        ),
        Cell(
            why="a mid-sized batch between the decode and chunk-bound extremes",
            spec=dict(**_R1, num_tokens=256, ep_rank=0),
        ),
        Cell(
            why="the chunk bound itself: the target splits the gathered token "
            "set at 8192 because that is where this op stops being certified",
            spec=dict(**_R1, num_tokens=8192, ep_rank=0),
        ),
        Cell(
            why="a non-zero expert window -- rank 3 owns global ids [192, 256), "
            "so an off-by-one in the window lands on the wrong experts",
            spec=dict(**_R1, num_tokens=256, ep_rank=3),
        ),
        Cell(
            why="token count not a multiple of anything: the tail path",
            spec=dict(**_R1, num_tokens=37, ep_rank=1),
        ),
    )

    note = """
    Tokens whose expert id falls outside this rank\'s window contribute
    nothing; a token routed entirely outside it comes back as zeros. That is
    the mechanism expert parallelism rests on, not an error, and the
    reference reproduces it.

    Repeated ids in one token\'s top-k are summed, not deduplicated.

    The autotuner\'s hot path is exercised by test_r1_mtp_autotuned_tactics
    rather than by a cell: what a tuning pass leaves in its cache is the
    tuner\'s behaviour, and `misc/test_autotuner.py` owns it.
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
        *unused_args: object,
        ep_size: int = 1,
        ep_rank: int = 0,
        **unused_kwargs: object,
    ) -> torch.Tensor:
        """What the op computes, in plain torch, fp32 through both GEMMs.

        Takes the arguments that change the result rather than mirroring
        `__call__`. The other ~40 are inert on the certified path -- a
        reference that accepted them would have to ignore them, which reads
        as coverage it does not have.

        `fc1_expert_weights[e]` is `[up | gate]` stacked on dim 0. Ids are
        global; this rank owns `[ep_rank * E_local, (ep_rank + 1) * E_local)`
        and everything outside contributes nothing, which is how expert
        parallelism tiles the routing space.
        """
        num_tokens, hidden = input.shape
        num_local = fc1_expert_weights.shape[0]
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=input.device)
        xf = input.float()
        for local_e in range(num_local):
            mask = token_selected_experts == ep_rank * num_local + local_e
            tok, slot = mask.nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            w_up, w_gate = fc1_expert_weights[local_e].float().chunk(2, dim=0)
            h_up = xf[tok] @ w_up.t()
            h_gate = xf[tok] @ w_gate.t()
            gated = h_gate * torch.sigmoid(h_gate)
            # the kernel materializes the FC1 activation in the input dtype
            inter = (gated * h_up).to(input.dtype).float()
            y = inter @ fc2_expert_weights[local_e].float().t()
            weight = (
                torch.ones(tok.numel(), device=input.device)
                if token_final_scales is None
                else token_final_scales[tok, slot].float()
            )
            out.index_add_(0, tok, y * weight.unsqueeze(1))
        return out.to(input.dtype)

    #: Relative distance between neighbouring representable values
    #: (1 + 2^-m for m stored mantissa bits): bf16 has 7, fp16 has 10.
    _ULP = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Wider than the base default, with the derivation rather than a number.

        Kernel and reference consume bit-identical operands and differ only in
        GEMM accumulation order and in which side of a rounding boundary each
        intermediate lands -- one flipped intermediate moves an output element
        by about one ulp of that row\'s scale. torch\'s default cannot express
        that: its bf16 `atol=1e-5` sits three orders of magnitude below one
        output ulp of a two-GEMM chain, and per-element `rtol` is meaningless
        where cancellation drives `|ref|` to zero.

        So: 8 ulp of the row\'s largest magnitude element-wise, 4 ulp of
        relative RMS in aggregate. Measured worst case across everything this
        entry drives is 3.98 / 1.43 ulp once the whole tactic space is walked
        -- 2.0x and 2.8x margin. Both gates were sized against deliberately
        wrong computations, which land two orders of magnitude out.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        ulp = self._ULP[ref.dtype]
        row_scale = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
        torch.testing.assert_close(
            out.float() / row_scale, ref.float() / row_scale, rtol=0.0, atol=8 * ulp
        )
        rel_rms = (
            (out.float() - ref.float()).pow(2).mean().sqrt()
            / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)
        ).item()
        assert rel_rms <= 4 * ulp, f"relative RMS {rel_rms:.3e} > {4 * ulp:.3e}"


fused_moe = _FusedMoe()
