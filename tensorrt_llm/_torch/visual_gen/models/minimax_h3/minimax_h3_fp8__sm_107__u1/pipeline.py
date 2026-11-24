# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU static-FP8 H3 pipeline with the shared component scheduling path."""

from pathlib import Path

from torch.utils.cpp_extension import get_default_build_root

from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent

from ..minimax_h3_nvfp4__sm_107__u1_u4.pipeline import MiniMaxH3Nvfp4Sm107U1U4Pipeline
from .modeling import Fp8Transformer


class MiniMaxH3Fp8Sm107U1Pipeline(MiniMaxH3Nvfp4Sm107U1U4Pipeline):
    """Static-FP8 DiT, BF16 residuals and FP32 media heads on SM107.

    The base contributes the FP8 Video VAE, component graphs and preloop
    scheduling. All DiT fusion and conditioning state belong to this instance.
    """

    def _init_transformer(self) -> None:
        self.transformer = Fp8Transformer(
            self.pipeline_config.model_configs[PipelineComponent.TRANSFORMER]
        ).eval()

    def post_load_weights(self) -> None:
        super().post_load_weights()
        kernel_cache = Path(get_default_build_root()) / "minimax_h3_fp8"
        self.transformer.initialize_fp8(kernel_cache)
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
