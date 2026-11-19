# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorrt_llm._torch.visual_gen.config import (
    DiffusionPipelineConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm.visual_gen.args import AttentionConfig
from tensorrt_llm.visual_gen.sparse_attention import VideoSparseAttentionConfig

from ..checkpoint import load_fasth3_manifest
from ..components.attention import MiniMaxH3VSAAttention
from ..components.transformer import MiniMaxH3Transformer3DModel
from ..pipeline import H3PipelineBase
from .schedule import set_dmd_timesteps


def configure_attention(config: DiffusionPipelineConfig, manifest: dict) -> None:
    attention = config.attention
    backend = attention.backend
    if backend == "VANILLA" and "backend" not in attention.model_fields_set:
        backend = "TRTLLM"
    if backend not in ("TRTLLM", "CUTEDSL"):
        raise ValueError("FastH3 requires VSA with the TRTLLM or CUTEDSL attention backend.")
    sparse = attention.sparse_attention_config
    if sparse is None:
        sparse = VideoSparseAttentionConfig(vsa_sparsity=manifest.get("vsa_sparsity", 0.8))
    elif sparse.algorithm != "vsa":
        raise ValueError("FastH3 requires the vsa sparse attention algorithm.")
    config.attention = AttentionConfig.model_validate(
        {**attention.model_dump(), "backend": backend, "sparse_attention_config": sparse}
    )
    # Resolve defaults before constructing the transformer and loading its trained gate.
    if backend == "TRTLLM" and config.attention_metadata_state is None:
        config.attention_metadata_state = create_attention_metadata_state()
    for model_config in config.model_configs.values():
        model_config.attention = config.attention.model_copy(deep=True)
        if backend == "TRTLLM" and model_config.attention_metadata_state is None:
            model_config.attention_metadata_state = create_attention_metadata_state()


class FastH3Pipeline(H3PipelineBase):
    """FastH3 single-GPU VSA pipeline with checkpoint-trained denoising schedules."""

    def _init_transformer(self) -> None:
        self.transformer = MiniMaxH3Transformer3DModel(
            self.pipeline_config.model_configs[PipelineComponent.TRANSFORMER],
            attention_cls=MiniMaxH3VSAAttention,
        )

    def load_standard_components(self, checkpoint_dir, device, skip_components=None, **kwargs):
        pretrained = getattr(self.pipeline_config, "primary_pretrained_config", None)
        manifest = getattr(pretrained, "_fasth3_manifest", None)
        self._fastvideo_inference = (
            manifest if manifest is not None else load_fasth3_manifest(checkpoint_dir)
        )
        return super().load_standard_components(checkpoint_dir, device, skip_components, **kwargs)

    @property
    def default_generation_params(self):
        params = super().default_generation_params
        manifest = getattr(self, "_fastvideo_inference", {})
        steps = manifest.get("dmd_denoising_steps")
        default = len(steps) + 1 if steps is not None else params["num_inference_steps"]
        params["num_inference_steps"] = int(manifest.get("num_inference_steps", default))
        return params

    @property
    def default_warmup_steps(self) -> int:
        steps = getattr(self, "_fastvideo_inference", {}).get("dmd_denoising_steps")
        if steps is not None:
            return len(steps) + 1
        return super().default_warmup_steps

    def _set_timesteps(self, num_inference_steps: int) -> None:
        manifest = getattr(self, "_fastvideo_inference", {})
        if manifest.get("dmd_denoising_steps") is None:
            return super()._set_timesteps(num_inference_steps)
        set_dmd_timesteps(
            self.scheduler, self.audio_scheduler, manifest, num_inference_steps, self.device
        )
