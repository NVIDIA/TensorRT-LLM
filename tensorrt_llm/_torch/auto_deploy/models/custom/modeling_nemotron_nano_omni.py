# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prefill-only NemotronH Nano Omni model for AutoDeploy.

Source checkpoint:
  - nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning

This is a multimodal model (vision + audio + text) whose LLM backbone is
NemotronH, a hybrid Mamba/Attention/MoE architecture. The outer wrapper
remains eager so it can run Nemotron's vision merge path, while only the inner
text model is exported by AutoDeploy. Audio remains intentionally unsupported
in the current AD path and is dropped at load time.

The NemotronH backbone is translated fresh from the HuggingFace source into a
lean prefill-only implementation using AutoDeploy canonical ops for SSM,
causal conv1d, attention, and MoE.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from torch import nn
from torch.fx import GraphModule
from transformers import AutoProcessor, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.models.hf import (
    AutoModelForCausalLMFactory,
    AutoModelForImageTextToTextFactory,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_radio import RADIOVisionModel
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.multimodal import MultimodalInput, apply_mm_hashes, hexdigest_to_int32
from tensorrt_llm.inputs.registry import (
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
)
from tensorrt_llm.inputs.utils import VideoData, load_image, load_video

VIDEO_MAX_NUM_TILES = 1
IMAGE_PLACEHOLDER = "<image>"
VIDEO_PLACEHOLDER = "<video>"
AUDIO_PLACEHOLDER = "<so_embedding>"


def _media_to_raw_chw(media: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
    """Convert a single PIL Image or [0, 1] tensor to a CHW float tensor in [0, 255]."""
    if isinstance(media, Image.Image):
        rgb = media.convert("RGB") if media.mode != "RGB" else media
        return torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float()
    if isinstance(media, torch.Tensor):
        return media * 255.0 if media.ndim == 3 else media
    raise TypeError(f"Unsupported media type: {type(media)}")


def _resize_and_normalize(
    tensor: torch.Tensor,
    target_h: int,
    target_w: int,
    norm_mean: Optional[torch.Tensor] = None,
    norm_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bicubic resize + rescale to [0, 1] + optional mean/std normalization."""
    needs_unsqueeze = tensor.ndim == 3
    if needs_unsqueeze:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-2] != target_h or tensor.shape[-1] != target_w:
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    tensor = tensor.clamp(0, 255).div_(255.0)
    if norm_mean is not None and norm_std is not None:
        tensor = tensor.sub_(norm_mean).div_(norm_std)
    if needs_unsqueeze:
        tensor = tensor.squeeze(0)
    return tensor


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.pow(F.relu(x), 2)


# =============================================================================
# NemotronH Text Backbone — self-contained prefill-only implementation
# =============================================================================


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size, group_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, hidden_states, gate=None):
        return torch.ops.auto_deploy.torch_rmsnorm_gated(
            x=hidden_states,
            weight=self.weight,
            gate=gate,
            eps=self.variance_epsilon,
            group_size=self.group_size,
            norm_before_gate=False,
        )


class NemotronHRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class NemotronHMamba2Mixer(nn.Module):
    """Mamba-2 SSM mixer using AD canonical ops for causal conv1d and SSM."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.num_heads = config.mamba_num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.mamba_hidden_act
        self.act = ACT2FN[config.mamba_hidden_act]
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.n_groups = config.n_groups
        self.head_dim = config.mamba_head_dim
        self.chunk_size = config.chunk_size
        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)

        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=self.layer_norm_epsilon,
            group_size=self.intermediate_size // self.n_groups,
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def forward(self, input_states):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        projected_states = self.in_proj(input_states)
        gate, hidden_states_B_C, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        # Causal conv1d via canonical op
        hidden_states_B_C = self.act(
            torch.ops.auto_deploy.torch_causal_conv1d(
                hidden_states_B_C,
                self.conv1d.weight,
                self.conv1d.bias,
                self.conv1d.stride[0],
                self.conv1d.padding[0],
                self.conv1d.dilation[0],
                self.conv1d.groups,
                self.conv1d.padding_mode,
            )
        )

        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )

        # SSM via canonical op
        A = -torch.exp(self.A_log.float())
        y = torch.ops.auto_deploy.torch_ssm(
            hidden_states=hidden_states.view(batch_size, seq_len, -1, self.head_dim),
            A=A,
            B=B.view(batch_size, seq_len, -1, self.ssm_state_size),
            C=C.view(batch_size, seq_len, -1, self.ssm_state_size),
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            time_step_limit=list(self.time_step_limit),
            chunk_size=self.chunk_size,
        )
        y = y.reshape(batch_size, seq_len, -1)

        scan_output = self.norm(y, gate)
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states


class NemotronHMLP(nn.Module):
    def __init__(self, config, layer_idx: int, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class NemotronHTopkRouter(nn.Module):
    """DeepSeek-V3 style noaux_tc router in vanilla PyTorch."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32)
        )
        nn.init.normal_(self.weight, std=0.01)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )

    def _get_topk_indices(self, scores):
        scores_for_choice = scores.view(
            -1, self.n_routed_experts
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        topk_indices = self._get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class NemotronHMOE(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                NemotronHMLP(
                    config,
                    layer_idx=layer_idx,
                    intermediate_size=config.moe_intermediate_size,
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = NemotronHTopkRouter(config)
        self.shared_experts = NemotronHMLP(
            config=config,
            intermediate_size=config.moe_shared_expert_intermediate_size,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        x_flat = hidden_states.view(-1, hidden_states.shape[-1])

        # Shared expert first (dispatch order matters for AD stream forking)
        shared_out = self.shared_experts(residuals)

        # Routed experts via canonical MoE op
        out_flat = torch.ops.auto_deploy.torch_moe(
            x_flat,
            topk_indices,
            topk_weights,
            w1_weight=[e.up_proj.weight for e in self.experts],
            w2_weight=[e.down_proj.weight for e in self.experts],
            w3_weight=[],
            act_fn=ActivationType.Relu2,
            is_gated_mlp=False,
        )

        routed_out = out_flat.view(*orig_shape)
        return shared_out + routed_out


class NemotronHAttention(nn.Module):
    """Multi-headed attention using AD canonical attention op (handles GQA natively)."""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_heads, self.hidden_size, bias=config.attention_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # AD canonical attention op handles GQA natively — no repeat_kv needed
        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            layout="bsnd",
        )
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class NemotronHBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = NemotronHAttention(config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config, layer_idx=layer_idx)
        elif self.block_type == "moe":
            self.mixer = NemotronHMOE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid block type '{self.block_type}' at layer {layer_idx}")

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states)
        return residual + hidden_states


@dataclass
class NemotronHOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class NemotronHCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class NemotronHPreTrainedModel(PreTrainedModel):
    # config_class intentionally omitted — config loaded via trust_remote_code
    base_model_prefix = "backbone"
    _no_split_modules = ["NemotronHBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        if isinstance(module, NemotronHMamba2Mixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True
            dt = torch.exp(
                torch.rand(self.config.mamba_num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


class NemotronHModel(NemotronHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NemotronHBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm_f = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in list(state_dict.keys()):
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHOutput]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify at least one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds
        for mixer_block in self.layers:
            hidden_states = mixer_block(hidden_states)

        hidden_states = self.norm_f(hidden_states)
        return NemotronHOutput(last_hidden_state=hidden_states)


class NemotronHForCausalLM(NemotronHPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = NemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Keep stable Python-side refs for the eager multimodal wrapper. Some AD export /
        # compile stages can flatten or replace intermediate submodule structure (for
        # example, ``backbone``), but the wrapper still needs access to the token embedding
        # module to build ``inputs_embeds`` before calling the exported text model.
        self.__dict__["_input_embeddings_ref"] = self.backbone.get_input_embeddings()
        self.__dict__["_output_embeddings_ref"] = self.lm_head
        self.post_init()

    def get_input_embeddings(self):
        backbone = getattr(self, "backbone", None)
        if hasattr(backbone, "get_input_embeddings"):
            return backbone.get_input_embeddings()
        return self.__dict__["_input_embeddings_ref"]

    def set_input_embeddings(self, new_embeddings):
        backbone = getattr(self, "backbone", None)
        if hasattr(backbone, "set_input_embeddings"):
            backbone.set_input_embeddings(new_embeddings)
        self.__dict__["_input_embeddings_ref"] = new_embeddings

    def get_output_embeddings(self):
        if hasattr(self, "lm_head"):
            return self.lm_head
        return self.__dict__["_output_embeddings_ref"]

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.__dict__["_output_embeddings_ref"] = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHCausalLMOutput]:
        nemotron_h_outputs = self.backbone(input_ids, inputs_embeds=inputs_embeds)
        hidden_states = nemotron_h_outputs[0]
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return NemotronHCausalLMOutput(logits=logits)


# =============================================================================
# Multimodal wrapper and AD input processor
# =============================================================================


def _is_nemotron_video_frame(value: Any) -> bool:
    return isinstance(value, (Image.Image, torch.Tensor))


def _normalize_nemotron_image_items(images: Any) -> list[Any]:
    if images is None:
        return []
    if isinstance(images, list):
        return images
    return [images]


def _normalize_nemotron_video_items(videos: Any) -> list[Any]:
    if videos is None:
        return []
    if isinstance(videos, VideoData):
        return [videos]
    if isinstance(videos, list):
        if not videos:
            return []
        if all(_is_nemotron_video_frame(frame) for frame in videos):
            return [videos]
        return videos
    return [videos]


_NANO_VL_PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={
        "image": IMAGE_PLACEHOLDER,
        "video": VIDEO_PLACEHOLDER,
        "audio": AUDIO_PLACEHOLDER,
    },
    placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    placeholders_separator="\n",
    content_format=ContentFormat.STRING,
)


class NemotronNanoOmniPreTrainedModel(PreTrainedModel):
    """Base class for the multimodal wrapper. No config_class — loaded via trust_remote_code."""

    _no_split_modules = ["NemotronHBlock"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        pass


class NemotronNanoOmniForConditionalGeneration(NemotronNanoOmniPreTrainedModel, GenerationMixin):
    """Eager multimodal wrapper that exports only the inner text model."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.language_model = NemotronHForCausalLM(config.llm_config)
        self.__dict__["_input_embeddings_ref"] = self.language_model.get_input_embeddings()
        self.__dict__["_output_embeddings_ref"] = self.language_model.get_output_embeddings()
        self.__dict__["_language_model_hidden_size"] = config.llm_config.hidden_size

        if not hasattr(config, "text_config"):
            config.text_config = config.llm_config

        self.img_context_token_id = getattr(config, "img_context_token_id", None)
        self.video_temporal_patch_size = getattr(
            getattr(config, "vision_config", None), "video_temporal_patch_size", 1
        )
        self._vision_enabled = self._can_enable_vision(config)

        if self._vision_enabled:
            self.image_size = config.force_image_size
            self.patch_size = config.patch_size
            self.downsample_ratio = config.downsample_ratio
            self.ps_version = config.ps_version
            self.num_image_token = int(
                (self.image_size // self.patch_size) ** 2 * (self.downsample_ratio**2)
            )
            self.vit_hidden_size = config.vit_hidden_size
            self.vision_projection_hidden_size = config.projector_hidden_size
            self.llm_hidden_size = config.llm_config.hidden_size

            eps = getattr(config.llm_config, "rms_norm_eps", None)
            if eps is None:
                eps = config.llm_config.layer_norm_epsilon

            self.mlp1 = nn.Sequential(
                nn.RMSNorm(
                    self.vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                    eps=eps,
                    dtype=config.torch_dtype,
                ),
                nn.Linear(
                    self.vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                    self.vision_projection_hidden_size,
                    bias=False,
                    dtype=config.torch_dtype,
                ),
                SquaredReLU(),
                nn.Linear(
                    self.vision_projection_hidden_size,
                    self.llm_hidden_size,
                    bias=False,
                    dtype=config.torch_dtype,
                ),
            )

            vision_model_config = ModelConfig(pretrained_config=copy.deepcopy(config))
            vision_model_config.pretrained_config = (
                vision_model_config.pretrained_config.vision_config
            )
            self.vision_model = RADIOVisionModel(vision_model_config, disable_quantization=True)
        else:
            self.vision_model = nn.Module()
            self.mlp1 = nn.Module()

        self.sound_encoder = nn.Module()
        self.sound_projection = nn.Module()
        self._pending_video_embedder_weight_key: Optional[str] = None
        self._register_load_state_dict_pre_hook(self._drop_multimodal_weights)
        self.register_load_state_dict_post_hook(self._restore_video_embedder_loaded_flag)

    @staticmethod
    def _can_enable_vision(config) -> bool:
        required_attrs = (
            "force_image_size",
            "patch_size",
            "downsample_ratio",
            "ps_version",
            "vit_hidden_size",
            "projector_hidden_size",
            "vision_config",
        )
        return (
            all(hasattr(config, attr) for attr in required_attrs)
            and getattr(config, "vision_config", None) is not None
        )

    def _drop_multimodal_weights(self, state_dict, prefix, *args, **kwargs):
        prefixes_to_drop = ["sound_encoder.", "sound_projection."]
        self._pending_video_embedder_weight_key = None
        if not self._vision_enabled:
            prefixes_to_drop.extend(["vision_model.", "mlp1."])

        if self._vision_enabled:
            self._remember_video_embedder_weight_key(state_dict, prefix)
            self._remap_vision_weight_keys(state_dict, prefix)

        for key in list(state_dict.keys()):
            if any(key.startswith(prefix + mm_prefix) for mm_prefix in prefixes_to_drop):
                state_dict.pop(key)

    def _remember_video_embedder_weight_key(
        self, state_dict: Dict[str, torch.Tensor], prefix: str
    ) -> None:
        patch_generator = getattr(
            getattr(getattr(self.vision_model, "radio_model", None), "model", None),
            "patch_generator",
            None,
        )
        if getattr(patch_generator, "video_embedder", None) is None:
            return

        video_key = prefix + "vision_model.radio_model.model.patch_generator.video_embedder.weight"
        if video_key in state_dict:
            self._pending_video_embedder_weight_key = video_key

    def _restore_video_embedder_loaded_flag(self, module, incompatible_keys) -> None:
        del module
        pending_key = self._pending_video_embedder_weight_key
        self._pending_video_embedder_weight_key = None
        if pending_key is None:
            return

        if (
            pending_key in incompatible_keys.missing_keys
            or pending_key in incompatible_keys.unexpected_keys
        ):
            return

        patch_generator = getattr(
            getattr(getattr(self.vision_model, "radio_model", None), "model", None),
            "patch_generator",
            None,
        )
        if getattr(patch_generator, "video_embedder", None) is not None:
            patch_generator._video_embedder_loaded = True

    @staticmethod
    def _remap_vision_weight_keys(state_dict: Dict[str, torch.Tensor], prefix: str) -> None:
        vision_prefix = prefix + "vision_model."
        replacements = (
            (".attn.qkv.", ".attn.qkv_proj."),
            (".attn.proj.", ".attn.o_proj."),
            (".mlp.fc1.", ".mlp.up_proj."),
            (".mlp.fc2.", ".mlp.down_proj."),
        )

        for key in list(state_dict.keys()):
            if not key.startswith(vision_prefix):
                continue
            if key.startswith(vision_prefix + "radio_model.input_conditioner."):
                state_dict.pop(key)
                continue

            new_key = key
            for old, new in replacements:
                new_key = new_key.replace(old, new)

            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)

    def get_input_embeddings(self):
        try:
            return self.language_model.get_input_embeddings()
        except (AttributeError, KeyError):
            return self.__dict__["_input_embeddings_ref"]

    def set_input_embeddings(self, new_embeddings):
        self.language_model.set_input_embeddings(new_embeddings)
        self.__dict__["_input_embeddings_ref"] = new_embeddings

    def get_output_embeddings(self):
        try:
            return self.language_model.get_output_embeddings()
        except (AttributeError, KeyError):
            return self.__dict__["_output_embeddings_ref"]

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)
        self.__dict__["_output_embeddings_ref"] = new_embeddings

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
        )
        if self.ps_version == "v2":
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(
        self, pixel_values: torch.Tensor, num_frames: Optional[int] = None
    ) -> torch.Tensor:
        if not self._vision_enabled:
            raise ValueError("Nemotron vision tower is not initialized")

        vision_param = next(self.vision_model.parameters(), None)
        if vision_param is not None:
            pixel_values = pixel_values.to(device=vision_param.device, dtype=vision_param.dtype)

        temporal_patch_size = self.video_temporal_patch_size if num_frames is not None else 1
        micro_batch_size = 128 - (128 % temporal_patch_size) if temporal_patch_size > 1 else 128
        num_items = pixel_values.shape[0]
        height_patches = pixel_values.shape[2] // self.patch_size
        width_patches = pixel_values.shape[3] // self.patch_size

        vit_embeds_list = []
        for i in range(0, num_items, micro_batch_size):
            micro_batch_pixel_values = pixel_values[i : i + micro_batch_size]
            if num_frames is not None and temporal_patch_size > 1:
                vit_embeds = self.vision_model(
                    micro_batch_pixel_values, num_frames=micro_batch_pixel_values.shape[0]
                )
            else:
                vit_embeds = self.vision_model(micro_batch_pixel_values)

            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], height_patches, width_patches, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            vit_embeds = self.mlp1(vit_embeds)
            vit_embeds_list.append(vit_embeds)

        return torch.cat(vit_embeds_list, dim=0)

    def get_image_features(
        self, pixel_values: torch.Tensor, image_num_patches: torch.Tensor
    ) -> List[torch.Tensor]:
        image_embeds = self.extract_feature(pixel_values)
        split_sizes = [
            int(num_patches) * self.num_image_token for num_patches in image_num_patches.tolist()
        ]
        flattened = image_embeds.reshape(-1, image_embeds.shape[-1])
        return [embed.reshape(-1, embed.shape[-1]) for embed in torch.split(flattened, split_sizes)]

    def get_video_features(
        self, pixel_values_videos: torch.Tensor, video_size: torch.Tensor
    ) -> List[torch.Tensor]:
        video_span_embeds: List[torch.Tensor] = []
        frame_offset = 0
        for video_dims in video_size.tolist():
            num_frames, num_tiles_per_frame, _, _ = [int(v) for v in video_dims]
            total_tiles = num_frames * num_tiles_per_frame
            video_frames = pixel_values_videos[frame_offset : frame_offset + total_tiles]
            frame_offset += total_tiles

            video_embeds = self.extract_feature(video_frames, num_frames=num_frames)
            for span_embeds in video_embeds:
                video_span_embeds.append(span_embeds.reshape(-1, span_embeds.shape[-1]))
        return video_span_embeds

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor
    ) -> torch.Tensor:
        if self.img_context_token_id is None:
            raise ValueError("img_context_token_id is required for multimodal embedding merge")
        return (input_ids == self.img_context_token_id).unsqueeze(-1).expand_as(inputs_embeds)

    @staticmethod
    def _filter_graph_kwargs(submodule: GraphModule, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        expected_names = {node.name for node in submodule.graph.nodes if node.op == "placeholder"}
        return {k: v for k, v in kwargs.items() if k in expected_names}

    def _call_language_model(self, **kwargs) -> NemotronHCausalLMOutput:
        if isinstance(self.language_model, GraphModule):
            outputs = self.language_model(**self._filter_graph_kwargs(self.language_model, kwargs))
            if isinstance(outputs, NemotronHCausalLMOutput):
                return outputs
            if isinstance(outputs, dict):
                return NemotronHCausalLMOutput(logits=outputs["logits"])
            if isinstance(outputs, tuple):
                return NemotronHCausalLMOutput(logits=outputs[0])
            raise TypeError(
                f"Unsupported GraphModule output type for Nemotron language model: {type(outputs)}"
            )
        return self.language_model(**kwargs)

    def _select_request_chunk_multimodal_embeds(
        self,
        req_input_pos: int,
        req_seq_len: int,
        req_mm_item_types: Sequence[int],
        req_mm_positions: Sequence[int],
        req_mm_lengths: Sequence[int],
        req_special_offsets: Sequence[int],
        image_embeds_list: Optional[Sequence[torch.Tensor]],
        video_embeds_list: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        chunk_end = req_input_pos + req_seq_len
        mm_cumulative_offset = 0
        img_idx = 0
        vid_idx = 0
        chunks: List[torch.Tensor] = []
        special_offsets_set = {int(x) for x in req_special_offsets}

        for item_type, mm_start, mm_len in zip(req_mm_item_types, req_mm_positions, req_mm_lengths):
            item_mm_offset = mm_cumulative_offset
            item_mm_len = int(mm_len)
            item_abs_start = int(mm_start)
            item_abs_end = item_abs_start + item_mm_len
            overlap_start = max(req_input_pos, item_abs_start)
            overlap_end = min(chunk_end, item_abs_end)

            if item_type == 0:
                if image_embeds_list is None:
                    raise ValueError("Missing image embeddings for image multimodal item")
                item_embeds = image_embeds_list[img_idx]
                img_idx += 1
            elif item_type == 1:
                if video_embeds_list is None:
                    raise ValueError("Missing video embeddings for video multimodal item")
                item_embeds = video_embeds_list[vid_idx]
                vid_idx += 1
            else:
                raise ValueError(f"Unsupported Nemotron multimodal item type: {item_type}")

            local_to_feature_idx: List[Optional[int]] = []
            feature_idx = 0
            for rel_idx in range(item_mm_len):
                if item_mm_offset + rel_idx in special_offsets_set:
                    local_to_feature_idx.append(None)
                else:
                    local_to_feature_idx.append(feature_idx)
                    feature_idx += 1

            if feature_idx != item_embeds.shape[0]:
                raise ValueError(
                    "Nemotron multimodal embedding length mismatch: "
                    f"type={item_type}, expected={feature_idx}, actual={item_embeds.shape[0]}, "
                    f"mm_len={item_mm_len}, item_start={item_abs_start}"
                )

            if overlap_start < overlap_end:
                selected_indices = [
                    local_to_feature_idx[rel_idx]
                    for rel_idx in range(
                        overlap_start - item_abs_start, overlap_end - item_abs_start
                    )
                    if local_to_feature_idx[rel_idx] is not None
                ]
                if selected_indices:
                    chunks.append(item_embeds[selected_indices])

            mm_cumulative_offset += item_mm_len

        if chunks:
            return torch.cat(chunks, dim=0)

        sample_list = image_embeds_list if image_embeds_list else video_embeds_list
        if not sample_list:
            raise ValueError("Cannot build empty Nemotron multimodal chunk without embeddings")
        return torch.empty(
            0,
            self.__dict__["_language_model_hidden_size"],
            device=sample_list[0].device,
            dtype=sample_list[0].dtype,
        )

    def _build_chunked_multimodal_embeds(
        self,
        batch_info: torch.Tensor,
        input_pos: torch.Tensor,
        seq_len: torch.Tensor,
        image_embeds_list: Optional[Sequence[torch.Tensor]],
        video_embeds_list: Optional[Sequence[torch.Tensor]],
        mm_item_cu_seqlen: torch.Tensor,
        mm_item_types: torch.Tensor,
        mm_token_positions: torch.Tensor,
        mm_token_lengths: torch.Tensor,
        mm_special_offsets_cu_seqlen: Optional[torch.Tensor],
        mm_special_offsets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_prefill_seqs = int(batch_info[0].item())
        img_idx = 0
        vid_idx = 0
        chunks: List[torch.Tensor] = []

        for i in range(num_prefill_seqs):
            item_start = int(mm_item_cu_seqlen[i].item())
            item_end = int(mm_item_cu_seqlen[i + 1].item())
            req_mm_item_types = mm_item_types[item_start:item_end].tolist()
            req_mm_positions = mm_token_positions[item_start:item_end].tolist()
            req_mm_lengths = mm_token_lengths[item_start:item_end].tolist()

            req_special_offsets: List[int] = []
            if mm_special_offsets_cu_seqlen is not None and mm_special_offsets is not None:
                special_start = int(mm_special_offsets_cu_seqlen[i].item())
                special_end = int(mm_special_offsets_cu_seqlen[i + 1].item())
                req_special_offsets = mm_special_offsets[special_start:special_end].tolist()

            num_images = sum(item_type == 0 for item_type in req_mm_item_types)
            num_videos = sum(item_type == 1 for item_type in req_mm_item_types)
            req_image_embeds = (
                image_embeds_list[img_idx : img_idx + num_images]
                if image_embeds_list is not None
                else None
            )
            req_video_embeds = (
                video_embeds_list[vid_idx : vid_idx + num_videos]
                if video_embeds_list is not None
                else None
            )
            img_idx += num_images
            vid_idx += num_videos

            chunks.append(
                self._select_request_chunk_multimodal_embeds(
                    req_input_pos=int(input_pos[i].item()),
                    req_seq_len=int(seq_len[i].item()),
                    req_mm_item_types=req_mm_item_types,
                    req_mm_positions=req_mm_positions,
                    req_mm_lengths=req_mm_lengths,
                    req_special_offsets=req_special_offsets,
                    image_embeds_list=req_image_embeds,
                    video_embeds_list=req_video_embeds,
                )
            )

        if chunks:
            return torch.cat(chunks, dim=0)

        sample_list = image_embeds_list if image_embeds_list else video_embeds_list
        if not sample_list:
            raise ValueError("Cannot build empty Nemotron multimodal embeddings without features")
        return torch.empty(
            0,
            self.__dict__["_language_model_hidden_size"],
            device=sample_list[0].device,
            dtype=sample_list[0].dtype,
        )

    def _build_prefill_multimodal_embeds(
        self,
        seq_len: torch.Tensor,
        image_embeds_list: Optional[Sequence[torch.Tensor]],
        video_embeds_list: Optional[Sequence[torch.Tensor]],
        mm_item_cu_seqlen: torch.Tensor,
        mm_item_types: torch.Tensor,
        mm_token_positions: torch.Tensor,
        mm_token_lengths: torch.Tensor,
        mm_special_offsets_cu_seqlen: Optional[torch.Tensor],
        mm_special_offsets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_info = torch.tensor([seq_len.numel()], device=seq_len.device, dtype=torch.int32)
        input_pos = torch.zeros_like(seq_len)
        return self._build_chunked_multimodal_embeds(
            batch_info=batch_info,
            input_pos=input_pos,
            seq_len=seq_len,
            image_embeds_list=image_embeds_list,
            video_embeds_list=video_embeds_list,
            mm_item_cu_seqlen=mm_item_cu_seqlen,
            mm_item_types=mm_item_types,
            mm_token_positions=mm_token_positions,
            mm_token_lengths=mm_token_lengths,
            mm_special_offsets_cu_seqlen=mm_special_offsets_cu_seqlen,
            mm_special_offsets=mm_special_offsets,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_num_patches: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_size: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, NemotronHCausalLMOutput]:
        assert position_ids is not None

        has_images = pixel_values is not None and image_num_patches is not None
        has_videos = pixel_values_videos is not None and video_size is not None
        if (has_images or has_videos) and input_ids is None:
            raise ValueError("Nemotron multimodal forward requires input_ids for placeholder merge")

        if not has_images and not has_videos:
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            return self._call_language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        if not self._vision_enabled:
            raise ValueError(
                "Nemotron multimodal inputs were provided but vision is not initialized"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds_list = None
        if has_images:
            image_embeds_list = [
                embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
                for embeds in self.get_image_features(pixel_values, image_num_patches)
            ]

        video_embeds_list = None
        if has_videos:
            video_embeds_list = [
                embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
                for embeds in self.get_video_features(pixel_values_videos, video_size)
            ]

        placeholder_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
        num_multimodal_tokens = int((input_ids == self.img_context_token_id).sum().item())

        batch_info = kwargs.get("batch_info_host", kwargs.get("batch_info"))
        cu_seqlen = kwargs.get("cu_seqlen", kwargs.get("cu_seqlen_host"))
        seq_len = kwargs.get("seq_len")
        if seq_len is None and cu_seqlen is not None:
            seq_len = cu_seqlen[1:] - cu_seqlen[:-1]
        input_pos = kwargs.get("input_pos")
        if input_pos is None:
            seq_len_with_cache = kwargs.get(
                "seq_len_with_cache", kwargs.get("seq_len_with_cache_host")
            )
            if seq_len_with_cache is not None and seq_len is not None:
                input_pos = seq_len_with_cache.to(seq_len.device) - seq_len

        mm_item_cu_seqlen = kwargs.get("mm_item_cu_seqlen")
        mm_item_types = kwargs.get("mm_item_types")
        mm_token_positions = kwargs.get("mm_token_positions")
        mm_token_lengths = kwargs.get("mm_token_lengths")
        mm_special_offsets_cu_seqlen = kwargs.get("mm_special_offsets_cu_seqlen")
        mm_special_offsets = kwargs.get("mm_special_offsets")

        has_mm_layout = (
            mm_item_cu_seqlen is not None
            and mm_item_types is not None
            and mm_token_positions is not None
            and mm_token_lengths is not None
            and mm_item_cu_seqlen.numel() > 0
            and int(mm_item_cu_seqlen[-1].item()) > 0
        )

        if (
            batch_info is not None
            and input_pos is not None
            and seq_len is not None
            and has_mm_layout
        ):
            multimodal_embeds = self._build_chunked_multimodal_embeds(
                batch_info=batch_info,
                input_pos=input_pos,
                seq_len=seq_len,
                image_embeds_list=image_embeds_list,
                video_embeds_list=video_embeds_list,
                mm_item_cu_seqlen=mm_item_cu_seqlen,
                mm_item_types=mm_item_types,
                mm_token_positions=mm_token_positions,
                mm_token_lengths=mm_token_lengths,
                mm_special_offsets_cu_seqlen=mm_special_offsets_cu_seqlen,
                mm_special_offsets=mm_special_offsets,
            )
        elif seq_len is not None and has_mm_layout:
            multimodal_embeds = self._build_prefill_multimodal_embeds(
                seq_len=seq_len,
                image_embeds_list=image_embeds_list,
                video_embeds_list=video_embeds_list,
                mm_item_cu_seqlen=mm_item_cu_seqlen,
                mm_item_types=mm_item_types,
                mm_token_positions=mm_token_positions,
                mm_token_lengths=mm_token_lengths,
                mm_special_offsets_cu_seqlen=mm_special_offsets_cu_seqlen,
                mm_special_offsets=mm_special_offsets,
            )
        else:
            if image_embeds_list is not None and video_embeds_list is not None:
                raise ValueError(
                    "Nemotron mixed image/video multimodal merge requires layout metadata"
                )
            ordered_chunks: List[torch.Tensor] = []
            if image_embeds_list is not None:
                ordered_chunks.extend(image_embeds_list)
            if video_embeds_list is not None:
                ordered_chunks.extend(video_embeds_list)
            multimodal_embeds = torch.cat(ordered_chunks, dim=0)

        if multimodal_embeds.shape[0] != num_multimodal_tokens:
            raise ValueError(
                "Nemotron placeholder/token count mismatch: "
                f"placeholders={num_multimodal_tokens}, multimodal_embeds={multimodal_embeds.shape[0]}"
            )

        inputs_embeds = inputs_embeds.masked_scatter(placeholder_mask, multimodal_embeds)
        return self._call_language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


class NemotronNanoOmniADInputProcessor:
    """Nemotron-specific AD processor aligned with the HF image/video preprocessing contract."""

    def __init__(self, base_processor, processor, config):
        self.base_processor = base_processor
        self.processor = processor
        self.image_processor = getattr(processor, "image_processor", processor)
        self.tokenizer = getattr(base_processor.tokenizer, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("Nemotron AD input processor requires a HuggingFace tokenizer")

        self.multimodal_hashing_supported = False
        self.config = config
        self.dtype = getattr(config, "torch_dtype", torch.bfloat16)
        self.image_size = config.force_image_size
        self.patch_size = config.patch_size
        self.downsample_ratio = config.downsample_ratio
        self.num_image_token = int(
            (self.image_size // self.patch_size) ** 2 * (self.downsample_ratio**2)
        )
        self.img_context_token = config.img_context_token
        self.img_context_token_id = config.img_context_token_id
        self.video_context_token = config.video_context_token
        self.video_context_token_id = config.video_context_token_id
        self.img_start_token = config.img_start_token
        self.img_end_token = config.img_end_token
        self._img_start_token_ids = self.tokenizer.encode(
            self.img_start_token, add_special_tokens=False
        )
        self._img_end_token_ids = self.tokenizer.encode(
            self.img_end_token, add_special_tokens=False
        )

        vision_config = getattr(config, "vision_config", config)
        self.video_temporal_patch_size = getattr(vision_config, "video_temporal_patch_size", 1)
        self._add_video_prefix = True
        self.video_num_frames = getattr(vision_config, "num_frames", 8)

    def __getattr__(self, name: str):
        return getattr(self.base_processor, name)

    def get_vocab_size(self) -> int:
        return int(self.config.llm_config.vocab_size)

    def get_mm_token_ids(self) -> torch.Tensor:
        return torch.tensor([self.img_context_token_id], dtype=torch.int32)

    def get_mm_special_token_ids(self) -> torch.Tensor:
        return torch.tensor(
            sorted(set(self._img_start_token_ids + self._img_end_token_ids)), dtype=torch.int32
        )

    def _encode_chunks(self, chunks: Sequence[str]) -> List[int]:
        return self.tokenizer.encode("".join(chunks), add_special_tokens=False)

    @staticmethod
    def _extract_media_source(part: Dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            value = part.get(key)
            if value is None:
                continue
            if isinstance(value, dict):
                url = value.get("url")
                if url is not None:
                    return url
            else:
                return value
        return None

    def _load_message_image(self, part: Dict[str, Any]) -> Image.Image | torch.Tensor:
        image = self._extract_media_source(part, ("image", "image_url", "url"))
        if image is None:
            raise ValueError(f"Nemotron image content item is missing an image source: {part}")
        if isinstance(image, (Image.Image, torch.Tensor)):
            return image
        return load_image(image, format="pil")

    def _load_message_video(self, part: Dict[str, Any]) -> VideoData | List[Image.Image]:
        video = self._extract_media_source(part, ("video", "video_url", "url"))
        if video is None:
            raise ValueError(f"Nemotron video content item is missing a video source: {part}")
        if isinstance(video, (VideoData, list)):
            return video
        return load_video(video, num_frames=self.video_num_frames, format="pil")

    def _extract_message_multimodal_data(self, content: Any) -> Optional[Dict[str, List[Any]]]:
        if isinstance(content, str):
            return None
        if not isinstance(content, list):
            raise TypeError(f"Nemotron message content must be str or list, got {type(content)}")

        multimodal_data: Dict[str, List[Any]] = {}
        for part in content:
            if not isinstance(part, dict):
                raise TypeError(f"Nemotron message parts must be dicts, got {type(part)}")

            part_type = part.get("type")
            if part_type in ("text", "input_text"):
                continue
            if part_type in ("image", "image_url", "input_image"):
                multimodal_data.setdefault("image", []).append(self._load_message_image(part))
                continue
            if part_type in ("video", "video_url", "input_video"):
                multimodal_data.setdefault("video", []).append(self._load_message_video(part))
                continue
            if part_type in ("audio", "audio_url", "input_audio"):
                raise NotImplementedError("Nemotron AutoDeploy currently drops audio inputs")
            raise ValueError(f"Unsupported Nemotron message content type: {part_type}")

        return multimodal_data or None

    def _convert_messages_to_inputs(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = inputs.get("messages")
        if messages is None:
            return None

        multimodal_data: Dict[str, List[Any]] = {}
        has_multimodal = False

        for message in messages:
            message_mm_data = self._extract_message_multimodal_data(message.get("content", ""))
            if message_mm_data is None:
                continue
            has_multimodal = True
            for modality, modality_items in message_mm_data.items():
                multimodal_data.setdefault(modality, []).extend(modality_items)

        if not has_multimodal:
            return None

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        converted_inputs = {k: v for k, v in inputs.items() if k != "messages"}
        converted_inputs["prompt"] = prompt
        converted_inputs["multi_modal_data"] = multimodal_data
        return converted_inputs

    def _process_images(
        self, images: List[Image.Image | torch.Tensor], text_prompt: str
    ) -> Tuple[List[int], Dict[str, torch.Tensor]]:
        processed_images = self.image_processor(images=images, return_tensors="pt")
        parts = text_prompt.split(self.img_context_token)
        if len(parts) - 1 != len(processed_images["num_patches"]):
            raise ValueError(
                f"Number of {self.img_context_token} tokens ({len(parts) - 1}) doesn't match "
                f"the number of images ({len(processed_images['num_patches'])})"
            )

        processed_query = [parts[0]]
        for num_patches, part in zip(processed_images["num_patches"].tolist(), parts[1:]):
            feature_size = int(num_patches) * self.num_image_token
            processed_query.extend(
                [
                    self.img_start_token,
                    self.img_context_token * feature_size,
                    self.img_end_token,
                    part,
                ]
            )

        return self._encode_chunks(processed_query), {
            "pixel_values": processed_images["pixel_values"],
            "image_num_patches": processed_images["num_patches"].to(torch.int32),
        }

    def _process_videos_frames(
        self, videos: List[List[Image.Image | torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        pixel_values_list = []
        video_size_list = []
        original_max_num_tiles = getattr(self.image_processor, "max_num_tiles", None)
        if hasattr(self.image_processor, "max_num_tiles"):
            self.image_processor.max_num_tiles = VIDEO_MAX_NUM_TILES
        try:
            for video in videos:
                num_frames = len(video)
                processed_images = self.image_processor(images=video, return_tensors="pt")
                total_tiles, _, height, width = processed_images["pixel_values"].shape
                pixel_values_list.append(processed_images["pixel_values"])
                video_size_list.append([num_frames, total_tiles // num_frames, height, width])
        finally:
            if hasattr(self.image_processor, "max_num_tiles"):
                self.image_processor.max_num_tiles = original_max_num_tiles

        return {
            "pixel_values_videos": torch.cat(pixel_values_list, dim=0),
            "video_size": torch.tensor(video_size_list, dtype=torch.int32),
        }

    @staticmethod
    def _build_tubelet_separators(
        timestamps: List[float], frames_indices: List[int], temporal_patch_size: int
    ) -> List[str]:
        separators = []
        for group_idx, frame_start in enumerate(range(0, len(timestamps), temporal_patch_size)):
            group_frames = []
            for frame_offset in range(temporal_patch_size):
                frame_idx = frame_start + frame_offset
                if frame_idx >= len(timestamps):
                    continue
                prefix = "Frame" if frame_offset == 0 else "frame"
                group_frames.append(
                    f"{prefix} {frame_idx + 1} sampled at {timestamps[frame_idx]:.2f} seconds"
                )
            if not group_frames:
                continue
            separator = " and ".join(group_frames) + ": "
            if group_idx > 0:
                separator = "\n" + separator
            separators.append(separator)
        return separators

    def _get_frame_separators(
        self, video_size_list: List[List[int]], video_metadatas: List[Optional[Dict[str, Any]]]
    ) -> List[List[str]]:
        frame_separators_list = []
        temporal_patch_size = self.video_temporal_patch_size
        for metadata, video_size in zip(video_metadatas, video_size_list):
            num_frames = int(video_size[0])
            if metadata is not None:
                fps = metadata["fps"]
                frame_duration_ms = int(1000.0 / fps)
                frame_indices = metadata["frames_indices"]
                timestamps = [
                    int(frame_idx) * frame_duration_ms / 1000.0 for frame_idx in frame_indices
                ]
                if temporal_patch_size > 1:
                    frame_separators = self._build_tubelet_separators(
                        timestamps, frame_indices, temporal_patch_size
                    )
                else:
                    frame_separators = [
                        f"Frame {i + 1} sampled at {timestamp:.2f} seconds: "
                        for i, timestamp in enumerate(timestamps)
                    ]
            elif temporal_patch_size > 1:
                num_tubelets = math.ceil(num_frames / temporal_patch_size)
                frame_separators = [
                    ("\n" if tubelet_idx > 0 else "") + f"Frame {tubelet_idx + 1}: "
                    for tubelet_idx in range(num_tubelets)
                ]
            else:
                frame_separators = [f"Frame {frame_idx + 1}: " for frame_idx in range(num_frames)]
            frame_separators_list.append(frame_separators)
        return frame_separators_list

    def _compute_token_numbers_per_video(self, video_size_list: List[List[int]]) -> List[List[int]]:
        temporal_patch_size = self.video_temporal_patch_size
        num_tokens_per_video = []
        for video_size in video_size_list:
            num_frames = int(video_size[0])
            num_patches_per_frame = int(video_size[1])
            img_height = int(video_size[2])
            img_width = int(video_size[3])
            num_tubelets = (
                math.ceil(num_frames / temporal_patch_size)
                if temporal_patch_size > 1
                else num_frames
            )
            tokens_per_unit = int(
                (img_height * img_width // self.patch_size**2) * (self.downsample_ratio**2)
            )
            num_tokens_per_video.append([num_patches_per_frame * tokens_per_unit] * num_tubelets)
        return num_tokens_per_video

    def _process_video_prompts(
        self,
        split_text_prompt: List[str],
        num_tokens_per_video: List[List[int]],
        frame_separators_list: List[List[str]],
    ) -> List[int]:
        processed_query = []
        for video_index, (tokens_per_frame, frame_separators) in enumerate(
            zip(num_tokens_per_video, frame_separators_list)
        ):
            processed_query.append(split_text_prompt[video_index])
            if self._add_video_prefix:
                processed_query.append("This is a video:\n")
            for frame_separator, num_tokens in zip(frame_separators, tokens_per_frame):
                processed_query.extend(
                    [
                        frame_separator,
                        self.img_start_token,
                        self.img_context_token * int(num_tokens),
                        self.img_end_token,
                    ]
                )
        processed_query.append(split_text_prompt[-1])
        return self._encode_chunks(processed_query)

    def _process_videos(
        self, videos: List[Any], text_prompt: str
    ) -> Tuple[List[int], Dict[str, torch.Tensor], List[int]]:
        video_frames = []
        video_metadatas = []
        for video in videos:
            if isinstance(video, VideoData):
                video_frames.append(video.frames)
                video_metadatas.append(video.metadata)
            else:
                video_frames.append(video)
                video_metadatas.append(None)

        processed_videos = self._process_videos_frames(video_frames)
        video_size_list = processed_videos["video_size"].tolist()
        num_tokens_per_video = self._compute_token_numbers_per_video(video_size_list)
        frame_separators_list = self._get_frame_separators(video_size_list, video_metadatas)

        split_text_prompt = text_prompt.split(self.video_context_token)
        if len(split_text_prompt) - 1 != len(video_frames):
            raise ValueError(
                f"Number of {self.video_context_token} tokens ({len(split_text_prompt) - 1}) "
                f"doesn't match the number of videos ({len(video_frames)})"
            )

        return (
            self._process_video_prompts(
                split_text_prompt, num_tokens_per_video, frame_separators_list
            ),
            processed_videos,
            [len(tokens) for tokens in num_tokens_per_video],
        )

    def _find_placeholder_occurrences(self, text_prompt: str) -> List[Tuple[int, int]]:
        occurrences: List[Tuple[int, int]] = []
        cursor = 0
        while cursor < len(text_prompt):
            image_idx = text_prompt.find(self.img_context_token, cursor)
            video_idx = text_prompt.find(self.video_context_token, cursor)
            candidates = [
                (idx, item_type) for idx, item_type in ((image_idx, 0), (video_idx, 1)) if idx != -1
            ]
            if not candidates:
                break

            start_idx, item_type = min(candidates, key=lambda item: item[0])
            occurrences.append((item_type, start_idx))
            token = self.img_context_token if item_type == 0 else self.video_context_token
            cursor = start_idx + len(token)
        return occurrences

    def _process_mixed_multimodal_prompt(
        self,
        text_prompt: str,
        images: Optional[List[Image.Image | torch.Tensor]],
        videos: Optional[List[Any]],
    ) -> Tuple[List[int], Dict[str, torch.Tensor], List[Tuple[int, int]]]:
        processed_images = None
        if images:
            processed_images = self.image_processor(images=images, return_tensors="pt")

        processed_videos = None
        num_tokens_per_video: List[List[int]] = []
        frame_separators_list: List[List[str]] = []
        video_frames = []
        video_metadatas = []
        if videos:
            for video in videos:
                if isinstance(video, VideoData):
                    video_frames.append(video.frames)
                    video_metadatas.append(video.metadata)
                else:
                    video_frames.append(video)
                    video_metadatas.append(None)
            processed_videos = self._process_videos_frames(video_frames)
            video_size_list = processed_videos["video_size"].tolist()
            num_tokens_per_video = self._compute_token_numbers_per_video(video_size_list)
            frame_separators_list = self._get_frame_separators(video_size_list, video_metadatas)

        occurrences = self._find_placeholder_occurrences(text_prompt)
        if sum(item_type == 0 for item_type, _ in occurrences) != len(images or []):
            raise ValueError(
                "Number of Nemotron image placeholders does not match image inputs: "
                f"placeholders={sum(item_type == 0 for item_type, _ in occurrences)}, "
                f"images={len(images or [])}"
            )
        if sum(item_type == 1 for item_type, _ in occurrences) != len(videos or []):
            raise ValueError(
                "Number of Nemotron video placeholders does not match video inputs: "
                f"placeholders={sum(item_type == 1 for item_type, _ in occurrences)}, "
                f"videos={len(videos or [])}"
            )

        multimodal_data: Dict[str, torch.Tensor] = {}
        if processed_images is not None:
            multimodal_data["pixel_values"] = processed_images["pixel_values"]
            multimodal_data["image_num_patches"] = processed_images["num_patches"].to(torch.int32)
        if processed_videos is not None:
            multimodal_data.update(processed_videos)

        processed_query: List[str] = []
        ordered_items: List[Tuple[int, int]] = []
        cursor = 0
        image_idx = 0
        video_idx = 0
        for item_type, start_idx in occurrences:
            token = self.img_context_token if item_type == 0 else self.video_context_token
            processed_query.append(text_prompt[cursor:start_idx])
            cursor = start_idx + len(token)

            if item_type == 0:
                if processed_images is None:
                    raise ValueError("Nemotron image placeholder found without processed images")
                num_patches = int(processed_images["num_patches"][image_idx].item())
                feature_size = num_patches * self.num_image_token
                processed_query.extend(
                    [
                        self.img_start_token,
                        self.img_context_token * feature_size,
                        self.img_end_token,
                    ]
                )
                ordered_items.append((0, image_idx))
                image_idx += 1
                continue

            if processed_videos is None:
                raise ValueError("Nemotron video placeholder found without processed videos")
            if self._add_video_prefix:
                processed_query.append("This is a video:\n")
            for frame_separator, num_tokens in zip(
                frame_separators_list[video_idx], num_tokens_per_video[video_idx]
            ):
                processed_query.extend(
                    [
                        frame_separator,
                        self.img_start_token,
                        self.img_context_token * int(num_tokens),
                        self.img_end_token,
                    ]
                )
                ordered_items.append((1, video_idx))
            video_idx += 1

        processed_query.append(text_prompt[cursor:])
        return self._encode_chunks(processed_query), multimodal_data, ordered_items

    def _find_mm_spans(self, token_ids: Sequence[int]) -> List[Tuple[int, int, List[int]]]:
        spans = []
        start_ids = self._img_start_token_ids
        end_ids = self._img_end_token_ids
        i = 0
        while i < len(token_ids):
            if token_ids[i : i + len(start_ids)] != start_ids:
                i += 1
                continue

            j = i + len(start_ids)
            while j < len(token_ids) and token_ids[j] == self.img_context_token_id:
                j += 1

            if j == i + len(start_ids):
                i += 1
                continue
            if token_ids[j : j + len(end_ids)] != end_ids:
                i += 1
                continue

            span_len = j + len(end_ids) - i
            special_offsets = list(range(len(start_ids)))
            special_offsets.extend(range(span_len - len(end_ids), span_len))
            spans.append((i, span_len, special_offsets))
            i = j + len(end_ids)
        return spans

    def _build_multimodal_input(
        self,
        spans: List[Tuple[int, int, List[int]]],
        inputs: Dict[str, Any],
        ordered_items: Sequence[Tuple[int, int]],
    ) -> Tuple[MultimodalInput, List[int], List[int]]:
        mm_data = inputs.get("multi_modal_data", {})
        image_items = _normalize_nemotron_image_items(mm_data.get("image"))
        video_items = _normalize_nemotron_video_items(mm_data.get("video"))
        mm_uuids = inputs.get("multi_modal_uuids", None)

        expected_span_count = len(ordered_items)
        if len(spans) != expected_span_count:
            raise ValueError(
                "Mismatch between Nemotron multimodal prompt spans and multimodal inputs: "
                f"spans={len(spans)}, expected={expected_span_count}"
            )

        mm_hash_inputs = {}
        if image_items:
            mm_hash_inputs["image"] = image_items
        if video_items:
            mm_hash_inputs["video"] = video_items
        mm_hashes, _ = apply_mm_hashes(mm_hash_inputs, mm_uuids)

        image_hashes = [hexdigest_to_int32(h) for h in mm_hashes.get("image", [])]
        video_hashes = [hexdigest_to_int32(h) for h in mm_hashes.get("video", [])]
        image_uuids = list((mm_uuids or {}).get("image", [None] * len(image_items)))
        video_uuids = list((mm_uuids or {}).get("video", [None] * len(video_items)))

        starts: List[int] = []
        lengths: List[int] = []
        special_offsets: List[int] = []
        item_types: List[int] = []
        mm_hashes_flat: List[List[int]] = []
        mm_uuid_list: List[Optional[str]] = []
        mm_union_offset = 0
        for (span_start, span_len, span_special_offsets), (item_type, source_idx) in zip(
            spans, ordered_items
        ):
            starts.append(span_start)
            lengths.append(span_len)
            item_types.append(item_type)
            if item_type == 0:
                mm_hashes_flat.append(image_hashes[source_idx])
                mm_uuid_list.append(image_uuids[source_idx])
            elif item_type == 1:
                mm_hashes_flat.append(video_hashes[source_idx])
                mm_uuid_list.append(video_uuids[source_idx])
            else:
                raise ValueError(f"Unsupported Nemotron multimodal item type: {item_type}")
            special_offsets.extend(mm_union_offset + rel for rel in span_special_offsets)
            mm_union_offset += span_len

        return (
            MultimodalInput.from_components(
                mm_hashes_flat,
                starts,
                lengths,
                mm_uuid_list if mm_uuids is not None else None,
            ),
            special_offsets,
            item_types,
        )

    def __call__(self, inputs, sampling_params):
        converted_inputs = self._convert_messages_to_inputs(inputs)
        if converted_inputs is not None:
            inputs = converted_inputs

        if "multi_modal_data" not in inputs:
            return self.base_processor(inputs, sampling_params)

        text_prompt = inputs.get("prompt")
        if text_prompt is None:
            raise ValueError("Nemotron multimodal requests require a prompt")

        mm_data = inputs.get("multi_modal_data", {})
        images = mm_data.get("image")
        videos = mm_data.get("video")
        audios = mm_data.get("audio")

        if audios is not None:
            raise NotImplementedError("Nemotron AutoDeploy currently drops audio inputs")

        if images is None and videos is None:
            return self.base_processor(inputs, sampling_params)

        normalized_images = _normalize_nemotron_image_items(images) if images is not None else None
        normalized_videos = _normalize_nemotron_video_items(videos) if videos is not None else None
        token_ids, multimodal_data, ordered_items = self._process_mixed_multimodal_prompt(
            text_prompt=text_prompt,
            images=normalized_images,
            videos=normalized_videos,
        )

        spans = self._find_mm_spans(token_ids)
        multimodal_input, special_offsets, item_types = self._build_multimodal_input(
            spans, inputs, ordered_items
        )

        multimodal_data["layout_metadata"] = {
            "special_token_offsets": torch.tensor(special_offsets, dtype=torch.int32),
            "item_types": torch.tensor(item_types, dtype=torch.int32),
        }
        return token_ids, {
            "multimodal_input": multimodal_input,
            "multimodal_data": multimodal_data,
        }


@ModelFactoryRegistry.register("NemotronNanoOmniForConditionalGeneration")
class NemotronNanoOmniFactory(AutoModelForImageTextToTextFactory):
    """Factory for Nemotron Nano Omni that keeps only the inner text model exported."""

    def init_tokenizer(self) -> Optional[Any]:
        if self.tokenizer is None:
            return None
        tokenizer_kwargs = dict(self.tokenizer_kwargs)
        tokenizer_kwargs.setdefault("fix_mistral_regex", True)
        return AutoTokenizer.from_pretrained(self.tokenizer, **tokenizer_kwargs)

    def init_processor(self) -> Optional[Any]:
        if self.tokenizer is None:
            return None
        processor_kwargs = {"trust_remote_code": True, "fix_mistral_regex": True}
        if "use_fast" in self.tokenizer_kwargs:
            processor_kwargs["use_fast"] = self.tokenizer_kwargs["use_fast"]
        return AutoProcessor.from_pretrained(self.tokenizer, **processor_kwargs)

    def init_input_processor(self, base):
        model_config, _ = self._get_model_config()
        return NemotronNanoOmniADInputProcessor(base, self.init_processor(), model_config)


# =============================================================================
# Registration
# =============================================================================

AutoModelForCausalLMFactory.register_custom_model_cls(
    "NemotronH_Nano_Omni_Reasoning_V3_Config",
    NemotronNanoOmniForConditionalGeneration,
)
NemotronNanoOmniFactory.register_custom_model_cls(
    "NemotronH_Nano_Omni_Reasoning_V3_Config",
    NemotronNanoOmniForConditionalGeneration,
)
MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
    "NemotronH_Nano_Omni_Reasoning_V3",
    _NANO_VL_PLACEHOLDER_METADATA,
)
MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
    "NemotronH_Nano_VL_V2",
    _NANO_VL_PLACEHOLDER_METADATA,
)
