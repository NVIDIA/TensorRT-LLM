# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU static-FP8 H3 pipeline with the shared component scheduling path."""

from pathlib import Path

from torch.utils.cpp_extension import get_default_build_root

from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent

from ..minimax_h3_nvfp4__sm_107__u1_u4.pipeline import MiniMaxH3Nvfp4Sm107U1U4Pipeline
from .modeling import Fp8Transformer

_HYBRID_CUDNN_INITIAL_STEPS = 11
_HYBRID_DIRECT_FP8_OUT_START_STEP = 20
_HYBRID_SKIP_CORR_T1_START_STEP = 25


class MiniMaxH3Fp8Sm107U1Pipeline(MiniMaxH3Nvfp4Sm107U1U4Pipeline):
    """Static-FP8 DiT, BF16 residuals and FP32 media heads on SM107.

    The base contributes the FP8 Video VAE, component graphs and preloop
    scheduling. All DiT fusion and conditioning state belong to this instance.
    """

    def _init_transformer(self) -> None:
        self._hybrid_step_state: dict[str, int] | None = None
        self.transformer = Fp8Transformer(
            self.pipeline_config.model_configs[PipelineComponent.TRANSFORMER]
        ).eval()

    def _init_hybrid_attention(self) -> None:
        cfg = getattr(self, "pipeline_config", None)
        compile_cfg = getattr(cfg, "torch_compile", None) or getattr(cfg, "torch_compile_config", None)
        if getattr(compile_cfg, "enable_fullgraph", False):
            return
        blocks = getattr(self.transformer, "transformer_blocks", None)
        if not blocks:
            return
        from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention

        first_attn = getattr(blocks[0], "attn", None)
        stock = getattr(first_attn, "attn", None)
        if not isinstance(stock, CuDNNAttention) or stock.quant_dtype != "fp8":
            return
        cudnn_backend = CuDNNAttention(
            num_heads=stock.num_heads,
            num_kv_heads=stock.num_kv_heads,
            head_dim=stock.head_dim,
            dtype=stock.dtype,
            quant_attention_config=stock.quant_attention_config,
            fp8_dense_fmha="cudnn",
        )
        import torch

        direct_fp8_ok = False
        try:
            direct_fp8_ok = bool(
                torch.ops.trtllm.fmha_dense_fp8_is_supported(128, 56, torch.float8_e4m3fn, True)
            )
        except (RuntimeError, AttributeError):
            direct_fp8_ok = False

        step_state = {"step": 0}
        for block in blocks:
            block.attn._hybrid_cudnn_backend = cudnn_backend
            block.attn._hybrid_step_state = step_state
            block.attn._hybrid_cudnn_steps = _HYBRID_CUDNN_INITIAL_STEPS
            block.attn._hybrid_direct_fp8_steps = _HYBRID_DIRECT_FP8_OUT_START_STEP
            block.attn._hybrid_t1_steps = _HYBRID_SKIP_CORR_T1_START_STEP
            block.attn._hybrid_direct_fp8_scale = (
                block.attn.to_out[0].input_scale
                if direct_fp8_ok and getattr(block, "fp8_fused", False)
                else None
            )
            block.attn._hybrid_attention_forward = torch.compiler.disable(
                block.attn._hybrid_attention_forward
            )
        self.transformer.register_forward_hook(
            lambda _m, _i, _o: step_state.__setitem__("step", step_state["step"] + 1)
        )
        self._hybrid_step_state = step_state

    def post_load_weights(self) -> None:
        super().post_load_weights()
        kernel_cache = Path(get_default_build_root()) / "minimax_h3_fp8"
        self.transformer.initialize_fp8(kernel_cache)
        self._init_hybrid_attention()
        audio_vae = getattr(self, "audio_vae", None)
        if audio_vae is not None:
            from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_103_107__u4.vae_opt import (
                install_audio_vae_cuda_graph,
            )

            install_audio_vae_cuda_graph(audio_vae, self.device)
        if self.vae is not None:
            self.vae.enable_tiling(
                tile_sample_min_height=288,
                tile_sample_min_width=272,
                tile_sample_min_overlap_height=32,
                tile_sample_min_overlap_width=32,
            )

    def forward(self, *args, **kwargs):
        step_state = getattr(self, "_hybrid_step_state", None)
        if step_state is not None:
            step_state["step"] = 0
        try:
            return super().forward(*args, **kwargs)
        finally:
            if step_state is not None:
                step_state["step"] = 0
