# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Qwen-Image-Layered transformer variants."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from ..qwen_image.transformer_qwen_image import (
    QwenEmbedRope,
    QwenImageTransformer2DModel,
    QwenTimestepProjEmbeddings,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner


class QwenEmbedLayer3DRope(QwenEmbedRope):
    """Layer-aware 3D RoPE used by Qwen-Image-Layered.

    The layered checkpoint represents generated RGBA layers followed by
    one conditioning image in ``img_shapes``. Generated layers use their
    layer index as the frame-axis RoPE offset; the conditioning image uses
    the negative frame index from diffusers' reference implementation.
    """

    def forward(
        self,
        video_fhw,
        max_txt_seq_len: int | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        condition_index = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx == condition_index:
                video_freq = self._compute_condition_freqs(frame, height, width, device)
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, condition_index)
        max_txt_seq_len_int = int(max_txt_seq_len)
        txt_freqs = self._pos_freqs_for_device(device)[
            max_vid_index : max_vid_index + max_txt_seq_len_int, ...
        ]
        vid_freqs = torch.cat(vid_freqs, dim=0)
        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=128)
    def _compute_condition_freqs(
        self,
        frame: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        seq_lens = frame * height * width
        pos_freqs = self._pos_freqs_for_device(device)
        neg_freqs = self.neg_freqs.to(device) if device is not None else self.neg_freqs

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = (
                freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            )
            freqs_width = (
                freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
            )

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenImageLayeredTransformer2DModel(QwenImageTransformer2DModel):
    """Qwen-Image transformer variant for RGBA layer decomposition."""

    def __init__(
        self,
        model_config: Optional["DiffusionModelConfig"] = None,
        *,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
        attn_backend: str = "sdpa",
    ):
        super().__init__(
            model_config=model_config,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            axes_dims_rope=axes_dims_rope,
            attn_backend=attn_backend,
        )
        if use_layer3d_rope:
            self.pos_embed = QwenEmbedLayer3DRope(
                theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True
            )
        if use_additional_t_cond:
            self.time_text_embed = QwenTimestepProjEmbeddings(
                embedding_dim=self.inner_dim,
                use_additional_t_cond=True,
            )

    @staticmethod
    def _normalize_img_shapes_for_cuda_graph(*args, **kwargs) -> Optional[Tuple]:
        img_shapes = kwargs.get("img_shapes")
        if img_shapes is None and len(args) > 4:
            img_shapes = args[4]
        if img_shapes is None:
            return None

        def normalize(value):
            if isinstance(value, (list, tuple)):
                return tuple(normalize(item) for item in value)
            return int(value)

        return normalize(img_shapes)

    def register_cuda_graph_extra_key_fns(self, runner: "CUDAGraphRunner") -> None:
        super().register_cuda_graph_extra_key_fns(runner)
        runner.register_extra_key_fn("img_shapes", self._normalize_img_shapes_for_cuda_graph)

    @classmethod
    def from_config_dict(
        cls, cfg: Dict[str, Any], **kwargs
    ) -> "QwenImageLayeredTransformer2DModel":
        """Build from a transformer/config.json dict."""
        return cls(
            patch_size=cfg.get("patch_size", 2),
            in_channels=cfg.get("in_channels", 64),
            out_channels=cfg.get("out_channels", 16),
            num_layers=cfg.get("num_layers", 60),
            attention_head_dim=cfg.get("attention_head_dim", 128),
            num_attention_heads=cfg.get("num_attention_heads", 24),
            joint_attention_dim=cfg.get("joint_attention_dim", 3584),
            axes_dims_rope=tuple(cfg.get("axes_dims_rope", [16, 56, 56])),
            use_additional_t_cond=cfg.get("use_additional_t_cond", False),
            use_layer3d_rope=cfg.get("use_layer3d_rope", False),
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        img_shapes: Optional[list] = None,
        txt_seq_lens: Optional[list] = None,
        additional_t_cond: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Forward pass with optional Qwen-Image-Layered timestep condition."""
        del kwargs, txt_seq_lens  # Only kept for diffusers API compat.
        missing = []
        if timestep is None:
            missing.append("timestep")
        if img_shapes is None:
            missing.append("img_shapes")
        if missing:
            raise ValueError(f"Missing required argument(s): {', '.join(missing)}")

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        text_seq_len = encoder_hidden_states.shape[1]
        temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)
        image_rotary_emb = self.pos_embed(
            img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device
        )

        block_attention_mask = None
        if encoder_hidden_states_mask is not None:
            if encoder_hidden_states_mask.dtype != torch.bool:
                encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones(
                (batch_size, image_seq_len),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            block_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=block_attention_mask,
                timestep=timestep,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=output)
        return (output,)
