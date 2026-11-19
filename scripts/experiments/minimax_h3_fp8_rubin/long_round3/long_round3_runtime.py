# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-local gate-up/SG, down/residual and QK/RoPE controls after construction."""

import sys

import torch


class Runtime:
    """Toggle the audited round-F control and the qualified round-G combination."""

    def __init__(self, pipeline):
        import half_rope_qk
        import residual_runtime
        import strict_swiglu
        from breuse_down import DownRuntime
        from breuse_gateup import GateupRuntime

        from tensorrt_llm._torch.visual_gen.models.minimax_h3 import scenarios
        from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import (
            config,
        )
        from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.kernels import (
            fused_fp8_qk,
        )

        model_config = pipeline.pipeline_config
        if scenarios._ulysses_size(model_config) != 1 or model_config.mapping.tp_size != 1:
            raise ValueError("Round-G qualification covers single-GPU U1/TP1 only")
        if fused_fp8_qk.SCENARIO is not config.SCENARIO:
            raise RuntimeError("The QK kernel captured an earlier configuration object")
        blocks = pipeline.transformer.transformer_blocks
        if len(blocks) != 50:
            raise ValueError("Expected the qualified 50-block H3 transformer")
        mixed = sys.modules.get("h3_mixed_layout")
        if mixed is not None and mixed.active():
            blocks = mixed.require_layout(pipeline).fp8_blocks
        for block in blocks:
            for linear in (
                block.attn.qkv_proj,
                block.attn.to_out[0],
                block.ff.gate_up_proj,
                block.ff.down_proj,
            ):
                if not (
                    linear.has_fp8_qdq
                    and linear.weight.dtype == torch.float8_e4m3fn
                    and linear.input_scale is not None
                    and linear.input_scale.numel() == 1
                ):
                    raise ValueError("Expected all 200 original-static-FP8 linears")
        self.qk = fused_fp8_qk
        self.sg = strict_swiglu
        self.half_rope = half_rope_qk.half_rope_tensor_cast
        self.residual = residual_runtime
        self.original_kernel = self.qk._tensor_cast_g_kernel
        self.original_config = (
            config.SCENARIO.fp8_fusion_block_h,
            config.SCENARIO.fp8_fusion_warps,
            self.qk._V2,
            self.qk._VAMAX_FOLD,
        )
        strict_swiglu.load()
        self.gateup = GateupRuntime(blocks)
        self.down = DownRuntime(blocks)
        self.configure(False)

    def configure(self, enabled):
        """Select a mode outside request timing; keep native attention and its threshold."""
        if enabled:
            values = (8, 2, True, False)
        else:
            values = self.original_config
        launch = (self.qk.SCENARIO.fp8_fusion_block_h, self.qk.SCENARIO.fp8_fusion_warps)
        if launch != values[:2]:
            raise ValueError("Select the final QK launch configuration before model construction")
        self.qk._tensor_cast_g_kernel = self.half_rope if enabled else self.original_kernel
        self.qk._V2, self.qk._VAMAX_FOLD = values[2:]
        self.residual.STATE.update(
            kf_enabled=bool(enabled), kf_function=self.sg.run, validate_kf=False
        )
        self.gateup.configure(bool(enabled))
        self.down.configure(bool(enabled))
