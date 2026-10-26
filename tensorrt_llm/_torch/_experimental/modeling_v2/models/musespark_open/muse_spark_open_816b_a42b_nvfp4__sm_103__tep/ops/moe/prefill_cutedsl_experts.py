# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill routed NVFP4 experts on the Rubin CuTe DSL grouped GEMMs, over TRTLLM-gen's tensors.

The unfinalized expert call of a large (prefill or mixed) step: the same
``(gemm2_output, expanded_idx_to_permuted_idx)`` contract as TRTLLM-gen's
``fp4_block_scale_moe_runner`` with ``do_finalize=False``, computed by

* ``torch.ops.trtllm.moe_sort`` (tile ``TILE`` rows),
* the Rubin gather + grouped GEMM + SwiGLU + NVFP4 quantization kernel (FC1),
  in its ``trtllm_gen_gated_layout`` mode,
* the Rubin grouped GEMM (FC2) in its ``store_permuted`` +
  ``trtllm_gen_n_shuffle`` + ``mma_k64`` mode: one BF16 row per permuted row,
  no finalize, accumulated with the same 64-deep MMA as TRTLLM-gen's FC2.

Both kernels read the served expert tensors as they are (TRTLLM-gen's shuffled
rows and 128x4 swizzled block scales), so no second weight copy exists. The
FC1 epilogue takes the served leaf's two FC1 scales as TRTLLM-gen's gated
GEMM does (gate scale ``fc31_alpha``, output scale ``fc31_scale_c``) and uses
TRTLLM-gen's SwiGLU and NVFP4 quantization arithmetic, so the intermediate
codes and block scales equal TRTLLM-gen's; the model's per-expert post-norm
then runs on the rows exactly as on TRTLLM-gen's.

Graph capture: every kernel is compiled by ``prepare`` (outside any capture);
the tactics are fixed per row bucket (``TACTICS``), so no autotuning happens in
the served process.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

#: Routing tile (``moe_sort`` tile_tokens_dim, the kernels' M tile).
TILE = 256

#: Static tactics: (FC1, FC2) per smallest row count they serve, largest first.
#: FC1 = (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m, a_path);
#: FC2 = (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m).
TACTICS: Tuple[Tuple[int, tuple, tuple], ...] = (
    (
        0,
        ((256, 256, 256), (256, 256, 128), (2, 1), False, "cpasync"),
        ((256, 256, 256), (256, 256, 128), (2, 1), False),
    ),
)


def tactics_for(rows: int, table=None) -> Tuple[tuple, tuple]:
    """The (FC1, FC2) tactics of a call of ``rows`` routed tokens."""
    for min_rows, fc1, fc2 in table or TACTICS:
        if rows >= min_rows:
            return fc1, fc2
    raise ValueError(f"no tactic for {rows} rows")


class PrefillCuteDslExperts:
    """One layer's prefill expert call over a TRTLLM-gen NVFP4 leaf's tensors.

    ``backend`` is the leaf (or any object with its attributes: ``w3_w1_weight``,
    ``w3_w1_weight_scale``, ``w2_weight``, ``w2_weight_scale``, ``fc31_alpha``,
    ``fc31_scale_c``, ``fc2_alpha``, ``fc2_input_scale``). The tensors are read
    on every call, as views, so a later re-binding of the leaf's parameters
    (e.g. as views of a larger expert stack) is followed and nothing is copied.
    The FC1 alpha pair ``[fc31_alpha | fc31_scale_c]`` is built once per set
    of scale tensors.
    """

    def __init__(self, backend, num_experts: int, top_k: int, local_expert_offset: int):
        from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as _ops

        self.backend = backend
        self.local_num_experts = int(backend.w3_w1_weight.shape[0])
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.local_expert_offset = int(local_expert_offset)
        self.hidden = int(backend.w2_weight.shape[1])
        e = self.local_num_experts
        self._fc1 = _ops.Sm107BlockScaledContiguousGatherGroupedGemmActFusionRunner(
            self.num_experts,
            self.top_k,
            e,
            self.local_expert_offset,
            TILE,
            trtllm_gen_gated_layout=True,
        )
        self._fc2 = _ops.Sm107BlockScaledContiguousGroupedGemmFinalizeFusionRunner(
            self.num_experts,
            self.top_k,
            e,
            self.local_expert_offset,
            TILE,
            torch.bfloat16,
            store_permuted=True,
            trtllm_gen_n_shuffle=True,
            mma_k64=True,
        )
        self._scales_key = None
        self._scales = None
        self._dummy_scales: Dict[torch.device, torch.Tensor] = {}

    def operands(self):
        """``(w1, s1, w2, s2, gate_up_alpha, fc2_alpha, global_sf)`` in the kernels' dtypes (views)."""
        be = self.backend
        e = self.local_num_experts
        w31 = be.w3_w1_weight
        n1 = int(w31.shape[1])
        w1 = w31.view(torch.uint8).view(torch.float4_e2m1fn_x2)
        s1 = be.w3_w1_weight_scale.view(torch.uint8).reshape(e, n1, -1)
        w2 = be.w2_weight.view(torch.uint8).view(torch.float4_e2m1fn_x2)
        s2 = be.w2_weight_scale.view(torch.uint8).reshape(e, self.hidden, -1)
        key = tuple(
            (t.data_ptr(), t._version if not t.is_inference() else None)
            for t in (be.fc31_alpha, be.fc31_scale_c, be.fc2_alpha, be.fc2_input_scale)
        )
        if key != self._scales_key:
            # The FC1 kernel takes the output scale from the alpha pair; global_sf
            # is a required operand it does not read in this mode.
            g2 = be.fc2_input_scale.reshape(1).float()
            gate_up_alpha = torch.cat(
                (be.fc31_alpha.reshape(e).float(), be.fc31_scale_c.reshape(e).float())
            ).contiguous()
            self._scales = (
                gate_up_alpha,
                be.fc2_alpha.reshape(e).float().contiguous(),
                g2.contiguous(),
            )
            self._scales_key = key
        return (w1, s1, w2, s2) + self._scales

    def __call__(
        self,
        x_fp4: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tactics: Optional[Tuple[tuple, tuple]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(gemm2_output [permuted rows, hidden] bf16, expanded_idx_to_permuted_idx)``.

        ``x_fp4`` is ``[T, hidden / 2]`` packed e2m1, ``x_sf`` the linear e4m3
        block scales (``T * hidden / 16`` bytes), ``topk_ids`` global int32 ids
        ``[T, top_k]``, ``topk_weights`` their combine weights (only sorted with
        the ids; the rows carry no weight).
        """
        rows = int(x_fp4.shape[0])
        fc1_tactic, fc2_tactic = tactics if tactics is not None else tactics_for(rows)
        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            _total_num_padded_tokens,
            num_non_exiting_tiles,
        ) = torch.ops.trtllm.moe_sort(
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights.float(),
            num_experts=self.num_experts,
            top_k=self.top_k,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.local_num_experts,
            tile_tokens_dim=TILE,
        )
        w1, s1, w2, s2, gate_up_alpha, fc2_alpha, global_sf = self.operands()
        x = x_fp4.view(torch.uint8).view(torch.float4_e2m1fn_x2)
        xs = x_sf.view(torch.uint8).reshape(rows, -1)
        h, h_sf = self._fc1(
            [
                x,
                w1,
                xs,
                s1,
                gate_up_alpha,
                tile_idx_to_expert_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
                global_sf,
                None,
                None,
            ],
            tactic=fc1_tactic,
        )
        m = int(permuted_idx_to_expanded_idx.shape[0])
        out = torch.empty(m, self.hidden, dtype=torch.bfloat16, device=x_fp4.device)
        scales = self._dummy_scales.get(x_fp4.device)
        if scales is None:
            # Unread in store_permuted mode; any valid pointer of the right dtype.
            scales = torch.zeros(1, self.top_k, dtype=torch.float32, device=x_fp4.device)
            self._dummy_scales[x_fp4.device] = scales
        self._fc2(
            [
                h,
                w2,
                h_sf,
                s2,
                fc2_alpha,
                out,
                tile_idx_to_expert_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
                scales,
            ],
            tactic=fc2_tactic,
        )
        return out, expanded_idx_to_permuted_idx

    def prepare(self, rows: int, table=None) -> None:
        """Compile both kernels for ``rows`` (outside any graph capture) with an all-local routing."""
        dev = self.backend.w3_w1_weight.device
        ids = (
            torch.arange(rows * self.top_k, device=dev, dtype=torch.int32) % self.local_num_experts
            + self.local_expert_offset
        ).reshape(rows, self.top_k)
        x = torch.zeros(rows, self.hidden // 2, dtype=torch.uint8, device=dev)
        xs = torch.zeros(rows * self.hidden // 16, dtype=torch.uint8, device=dev)
        w = torch.full((rows, self.top_k), 1.0 / self.top_k, device=dev)
        self(x, xs, ids, w, tactics=tactics_for(rows, table))
