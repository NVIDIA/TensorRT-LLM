# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused mixture-of-experts layer: expert permutation + grouped FC1/FC2 GEMMs
with an activation between them (gated SwiGLU, or element-wise Relu2) +
routing-weighted combine, in one call.

Everything a target author needs is below and all of it is checked: what the
op computes (`reference`), what it refuses (`is_valid`), how close the two
must be and why (the test's `_compare`), and the range CI drives (`CELLS`).
"""

from __future__ import annotations

from typing import List, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: `activation_type` for SwiGLU in *this* op's enum (`ActivationType` in
#: `tensorrt_llm/_torch/utils.py`, mirroring the CUTLASS enum), which is not the
#: trtllm-gen runner's enum of the same name. FC1 is the `[up; gate]` pair,
#: 2 * I rows per expert.
ACTIVATION_SWIGLU = 5

#: `activation_type` for Relu2, `relu(x) ** 2`. Element-wise, so FC1 is the up
#: projection alone: I rows per expert.
ACTIVATION_RELU2 = 8

#: Two geometries are certified, one per caller, each repeated into every cell
#: so a cell prints as a complete configuration rather than a delta.
#:
#: The R1-0528 MTP layer's MoE (deepseek): bf16, SwiGLU, 64 of 256 experts per
#: rank over 4 ranks, top-8.
_R1 = dict(
    hidden=7168,
    inter=2048,
    experts_local=64,
    ep_size=4,
    topk=8,
    activation_type=ACTIVATION_SWIGLU,
)

#: Nemotron-3 Ultra's routed experts (bf16 checkpoint), in the 2048-wide latent
#: space, as its tep8 target issues them: every rank routes the same tokens over
#: all 512 experts and answers for its own 64-wide window, so the windows'
#: outputs sum across the 8 ranks. FC1 is the up projection alone (5120 rows:
#: Relu2 has no gate half). The slot weights are the router's sigmoid scores of
#: the top 22, renormalized and scaled to sum to 5.0 (`routing="sigmoid_x5"`).
_ULTRA = dict(
    hidden=2048,
    inter=5120,
    experts_local=64,
    ep_size=8,
    topk=22,
    activation_type=ACTIVATION_RELU2,
    routing="sigmoid_x5",
)

#: The same experts as the tp8ep2 target issues them (moe_ep 2 x moe_tp 4): a rank runs one of
#: two 256-expert windows, each expert on its quarter of the intermediate width (FC1 rows and FC2
#: columns [1280 t, 1280 t + 1280) of 5120). The op is handed the rank's slices and its tp
#: coordinates; its output is the partial sum over the window and the slice.
_ULTRA_TP8EP2 = dict(
    hidden=2048,
    inter=1280,
    experts_local=256,
    ep_size=2,
    tp_size=4,
    topk=22,
    activation_type=ACTIVATION_RELU2,
    routing="sigmoid_x5",
)


class FusedMoe(OpWrapper):
    """One MoE layer over pre-routed tokens; returns a fresh `[T, hidden]`.

    Fusion boundary: permutation, both GEMMs, the gated activation between
    them, and the routing-weighted combine. Routing itself happens before the
    call (`noaux_tc_op`), and the op\'s own `n_group`/`topk_group` stay inert.

    Of this op\'s ~50 parameters the shipped calls set a dozen. The rest --
    biases, the quantized paths, LoRA, alltoall, min-latency, the swiglu
    alpha/beta/limit triple -- are left at their defaults and are not
    certified here. `activation_type=5` is Swiglu in *this* op\'s enum, which
    is not the trtllm-gen runner\'s enum of the same name; `activation_type=8`
    is Relu2, whose FC1 is `[E, I, H]` (the up projection alone) where
    Swiglu\'s is `[E, 2I, H]`.
    """

    #: What CI drives. The geometry is fixed -- one op, one caller, one shape
    #: family -- so the cells vary what the caller actually varies: how many
    #: tokens arrive in a chunk, and which expert window this rank owns.
    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

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
        Cell(
            why="Ultra tep8 decode, 1 row, rank 0: relu2 FC1 of 5120 rows, top-22 of 512, "
            "a 64-wide window",
            spec=dict(**_ULTRA, num_tokens=1, ep_rank=0),
        ),
        Cell(
            why="Ultra decode, 1 row, rank 7: the last window, ids 448-511 -- the top of "
            "the expert range",
            spec=dict(**_ULTRA, num_tokens=1, ep_rank=7),
        ),
        Cell(
            why="Ultra at 8 rows, rank 3: a mid window at a small row count",
            spec=dict(**_ULTRA, num_tokens=8, ep_rank=3),
        ),
        Cell(
            why="Ultra decode ceiling, 32 rows, rank 0",
            spec=dict(**_ULTRA, num_tokens=32, ep_rank=0),
        ),
        Cell(
            why="Ultra decode ceiling, 32 rows, rank 7",
            spec=dict(**_ULTRA, num_tokens=32, ep_rank=7),
        ),
        Cell(
            why="Ultra tep8 decode at the serving envelope's ceiling, 256 rows, rank 0",
            spec=dict(**_ULTRA, num_tokens=256, ep_rank=0),
        ),
        Cell(
            why="Ultra tep8 decode at the serving envelope's ceiling, 256 rows, rank 7",
            spec=dict(**_ULTRA, num_tokens=256, ep_rank=7),
        ),
        Cell(
            why="Ultra prefill chunk, 300 rows, rank 0: a mid-width FC1 token tile",
            spec=dict(**_ULTRA, num_tokens=300, ep_rank=0),
        ),
        Cell(
            why="Ultra prefill chunk, 300 rows, rank 7",
            spec=dict(**_ULTRA, num_tokens=300, ep_rank=7),
        ),
        Cell(
            why="Ultra prefill at 8192 rows, rank 0: the engine's default chunk and the "
            "top of the certified column",
            spec=dict(**_ULTRA, num_tokens=8192, ep_rank=0),
        ),
        Cell(
            why="Ultra prefill at 8192 rows, rank 7",
            spec=dict(**_ULTRA, num_tokens=8192, ep_rank=7),
        ),
        Cell(
            why="Ultra tp8ep2 ep0/tp0, 1 row: a 256-expert window, each expert's 1280 of 5120",
            spec=dict(**_ULTRA_TP8EP2, num_tokens=1, ep_rank=0, tp_rank=0),
        ),
        Cell(
            why="Ultra tp8ep2 ep1/tp3, 1 row: the upper window (ids 256-511), the last slice",
            spec=dict(**_ULTRA_TP8EP2, num_tokens=1, ep_rank=1, tp_rank=3),
        ),
        Cell(
            why="Ultra tp8ep2 ep1/tp1, decode batch 32",
            spec=dict(**_ULTRA_TP8EP2, num_tokens=32, ep_rank=1, tp_rank=1),
        ),
        Cell(
            why="Ultra tp8ep2 ep0/tp2, 256 rows: the serving envelope's decode ceiling",
            spec=dict(**_ULTRA_TP8EP2, num_tokens=256, ep_rank=0, tp_rank=2),
        ),
        Cell(
            why="Ultra tp8ep2 ep1/tp0, prefill chunk of 300 rows",
            spec=dict(**_ULTRA_TP8EP2, num_tokens=300, ep_rank=1, tp_rank=0),
        ),
        Cell(
            why="Ultra tp8ep2 ep1/tp3, prefill at 8192 rows: the engine's default chunk",
            spec=dict(**_ULTRA_TP8EP2, num_tokens=8192, ep_rank=1, tp_rank=3),
        ),
    )

    note = """
    Tokens whose expert id falls outside this rank\'s window contribute
    nothing; a token routed entirely outside it comes back as zeros. That is
    the mechanism expert parallelism rests on, not an error, and the
    reference reproduces it.

    Repeated ids in one token\'s top-k are summed, not deduplicated.

    Tensor parallelism inside the experts (`tp_size` / `tp_rank`, the
    tp8ep2 cells) is the caller\'s slicing: it hands the op each expert\'s
    slice of the intermediate width (Relu2 FC1 rows, FC2 columns) and gets
    back the partial sum over that slice, which the ranks\' all-reduce
    completes. The coordinates do not change the result on the certified
    path, so the reference takes neither.

    The autotuner\'s hot path is exercised by test_autotuned_tactics, and the
    whole tactic space by test_tactic_space, each at both geometries, rather
    than by a cell: what a tuning pass leaves in its cache is the tuner\'s
    behaviour, and `misc/test_autotuner.py` owns it.
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
        activation_type: int = ACTIVATION_SWIGLU,
        **unused_kwargs: object,
    ) -> torch.Tensor:
        """What the op computes, in plain torch, fp32 through both GEMMs.

        Takes the arguments that change the result rather than mirroring
        `__call__`. The other ~40 are inert on the certified path -- a
        reference that accepted them would have to ignore them, which reads
        as coverage it does not have.

        For SwiGLU `fc1_expert_weights[e]` is `[up | gate]` stacked on dim 0; for
        Relu2 it is the up projection alone. Ids are global; this rank owns
        `[ep_rank * E_local, (ep_rank + 1) * E_local)` and everything outside
        contributes nothing, which is how expert parallelism tiles the routing
        space.
        """
        if activation_type not in (ACTIVATION_SWIGLU, ACTIVATION_RELU2):
            raise NotImplementedError(f"activation_type {activation_type} is not certified")
        num_tokens, hidden = input.shape
        num_local = fc1_expert_weights.shape[0]
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=input.device)
        xf = input.float()
        for local_e in range(num_local):
            mask = token_selected_experts == ep_rank * num_local + local_e
            tok, slot = mask.nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            if activation_type == ACTIVATION_RELU2:
                relu = torch.relu(xf[tok] @ fc1_expert_weights[local_e].float().t())
                act = relu * relu
            else:
                w_up, w_gate = fc1_expert_weights[local_e].float().chunk(2, dim=0)
                h_up = xf[tok] @ w_up.t()
                h_gate = xf[tok] @ w_gate.t()
                act = h_gate * torch.sigmoid(h_gate) * h_up
            # the kernel materializes the FC1 activation in the input dtype
            inter = act.to(input.dtype).float()
            y = inter @ fc2_expert_weights[local_e].float().t()
            weight = (
                torch.ones(tok.numel(), device=input.device)
                if token_final_scales is None
                else token_final_scales[tok, slot].float()
            )
            out.index_add_(0, tok, y * weight.unsqueeze(1))
        return out.to(input.dtype)


fused_moe = FusedMoe()
