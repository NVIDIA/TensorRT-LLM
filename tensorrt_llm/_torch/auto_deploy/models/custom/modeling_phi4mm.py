# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prefill-only Phi-4 multimodal implementation for AutoDeploy export."""

import enum
import importlib.util
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.phi4_multimodal.configuration_phi4_multimodal import Phi4MultimodalConfig
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

_IMAGE_SPECIAL_TOKEN_ID = 200010
_AUDIO_SPECIAL_TOKEN_ID = 200011


class InputMode(enum.IntEnum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3


Phi4MMConfig = Phi4MultimodalConfig


def _load_hf_aux_module(model_name_or_path: str, filename: str, module_name: str):
    root = pathlib.Path(model_name_or_path)
    module_path = root / filename
    if not module_path.exists():
        snapshot_path = snapshot_download(model_name_or_path, allow_patterns=[filename])
        module_path = pathlib.Path(snapshot_path) / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phi4MMLoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        vision_lora: Optional[Dict],
        speech_lora: Optional[Dict],
        bias: bool = False,
    ):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.scaling: Dict[str, float] = {}
        self.active_adapter: Optional[str] = None
        self.adapters_enabled = False

        if vision_lora is not None:
            rank = vision_lora["r"]
            self.lora_A["vision"] = nn.Linear(in_features, rank, bias=False)
            self.lora_B["vision"] = nn.Linear(rank, out_features, bias=False)
            self.scaling["vision"] = vision_lora["lora_alpha"] / rank
        if speech_lora is not None:
            rank = speech_lora["r"]
            self.lora_A["speech"] = nn.Linear(in_features, rank, bias=False)
            self.lora_B["speech"] = nn.Linear(rank, out_features, bias=False)
            self.scaling["speech"] = speech_lora["lora_alpha"] / rank

    def set_adapter(self, adapter_name: Optional[str]):
        self.active_adapter = adapter_name
        self.adapters_enabled = adapter_name in self.lora_A

    def disable_adapter(self):
        self.active_adapter = None
        self.adapters_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        if self.adapters_enabled and self.active_adapter is not None:
            adapter = self.active_adapter
            out = out + self.lora_B[adapter](self.lora_A[adapter](x)) * self.scaling[adapter]
        return out


class Phi4MMRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Phi4MMRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, config: Phi4MMConfig):
        super().__init__()
        self.dim = dim
        self.base = config.rope_theta
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_scaling = config.rope_scaling
        self._set_cos_sin_cache()

    def _compute_inv_freq(self, scale_factors: Optional[list[float]] = None) -> torch.Tensor:
        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim
        if scale_factors is None:
            return 1.0 / (self.base**inv_freq_shape)
        ext_factors = torch.tensor(scale_factors, dtype=torch.float32)
        return 1.0 / (ext_factors * (self.base**inv_freq_shape))

    def _build_cache(
        self, inv_freq: torch.Tensor, scaling_factor: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timesteps = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(timesteps, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * scaling_factor, emb.sin() * scaling_factor

    def _set_cos_sin_cache(self):
        scaling_factor = 1.0
        if self.rope_scaling is not None:
            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale > 1.0:
                scaling_factor = math.sqrt(
                    1 + math.log(scale) / math.log(self.original_max_position_embeddings)
                )

        default_cos, default_sin = self._build_cache(self._compute_inv_freq(), scaling_factor)
        self.register_buffer("_ad_cos_cached", default_cos, persistent=False)
        self.register_buffer("_ad_sin_cached", default_sin, persistent=False)

        if self.rope_scaling is None:
            return

        short_cos, short_sin = self._build_cache(
            self._compute_inv_freq(self.rope_scaling["short_factor"]), scaling_factor
        )
        long_cos, long_sin = self._build_cache(
            self._compute_inv_freq(self.rope_scaling["long_factor"]), scaling_factor
        )
        self.register_buffer("_ad_short_cos_cached", short_cos, persistent=False)
        self.register_buffer("_ad_short_sin_cached", short_sin, persistent=False)
        self.register_buffer("_ad_long_cos_cached", long_cos, persistent=False)
        self.register_buffer("_ad_long_sin_cached", long_sin, persistent=False)

    def forward(self, x: torch.Tensor):
        if self.rope_scaling is None:
            return (
                self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
                self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
            )
        return (
            self._ad_short_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_short_sin_cached.to(dtype=x.dtype, device=x.device),
            self._ad_long_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_long_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class Phi4MMMLP(nn.Module):
    def __init__(self, config: Phi4MMConfig):
        super().__init__()
        self.gate_up_proj = Phi4MMLoRALinear(
            config.hidden_size,
            2 * config.intermediate_size,
            vision_lora=getattr(config, "vision_lora", None),
            speech_lora=getattr(config, "speech_lora", None),
            bias=False,
        )
        self.down_proj = Phi4MMLoRALinear(
            config.intermediate_size,
            config.hidden_size,
            vision_lora=getattr(config, "vision_lora", None),
            speech_lora=getattr(config, "speech_lora", None),
            bias=False,
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def set_adapter(self, adapter_name: Optional[str]):
        self.gate_up_proj.set_adapter(adapter_name)
        self.down_proj.set_adapter(adapter_name)

    def disable_adapter(self):
        self.gate_up_proj.disable_adapter()
        self.down_proj.disable_adapter()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)


class Phi4MMAttention(nn.Module):
    def __init__(self, config: Phi4MMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rotary_ndims = int(self.head_dim * config.partial_rotary_factor)
        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.qkv_proj = Phi4MMLoRALinear(
            config.hidden_size,
            op_size,
            vision_lora=getattr(config, "vision_lora", None),
            speech_lora=getattr(config, "speech_lora", None),
            bias=False,
        )
        self.o_proj = Phi4MMLoRALinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            vision_lora=getattr(config, "vision_lora", None),
            speech_lora=getattr(config, "speech_lora", None),
            bias=False,
        )
        self.rotary_emb = Phi4MMRotaryEmbedding(self.rotary_ndims, config)
        self.scaling = self.head_dim ** (-0.5)

    def set_adapter(self, adapter_name: Optional[str]):
        self.qkv_proj.set_adapter(adapter_name)
        self.o_proj.set_adapter(adapter_name)

    def disable_adapter(self):
        self.qkv_proj.disable_adapter()
        self.o_proj.disable_adapter()

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        kv_pos = self.num_key_value_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + kv_pos]
        value_states = qkv[..., query_pos + kv_pos :]

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        position_embeddings = self.rotary_emb(value_states)
        if self.rotary_emb.rope_scaling is None:
            cos, sin = position_embeddings
        else:
            short_cos, short_sin, long_cos, long_sin = position_embeddings
            if position_ids.is_meta:
                cos, sin = short_cos, short_sin
            else:
                seq_len = int(position_ids.max().item()) + 1
                if seq_len <= self.rotary_emb.original_max_position_embeddings:
                    cos, sin = short_cos, short_sin
                else:
                    cos, sin = long_cos, long_sin
        cos = cos[position_ids]
        sin = sin[position_ids]
        query_rot = query_states[..., : self.rotary_ndims]
        query_pass = query_states[..., self.rotary_ndims :]
        key_rot = key_states[..., : self.rotary_ndims]
        key_pass = key_states[..., self.rotary_ndims :]
        query_rot, key_rot = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            query_rot,
            key_rot,
            cos,
            sin,
            2,
        )
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)
        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            scale=self.scaling,
            layout="bsnd",
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Phi4MMDecoderLayer(nn.Module):
    def __init__(self, config: Phi4MMConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Phi4MMAttention(config, layer_idx)
        self.mlp = Phi4MMMLP(config)
        self.input_layernorm = Phi4MMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Phi4MMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def set_adapter(self, adapter_name: Optional[str]):
        self.self_attn.set_adapter(adapter_name)
        self.mlp.set_adapter(adapter_name)

    def disable_adapter(self):
        self.self_attn.disable_adapter()
        self.mlp.disable_adapter()

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Phi4MMImageEmbedding(nn.Module):
    def __init__(self, config: Phi4MMConfig, **kwargs):
        super().__init__()
        aux = _load_hf_aux_module(
            config._name_or_path, "vision_siglip_navit.py", "phi4mm_vision_aux"
        )
        self.img_processor = aux.get_siglip_vision_model()
        hidden_size = config.hidden_size
        pe_weight = self.img_processor.embeddings.position_embedding.weight
        length, image_dim_out = pe_weight.size()
        size = int(math.sqrt(length))
        self.image_dim_out = image_dim_out
        self.base_feat_height_target = size
        self.base_feat_height_reduction = 1
        self.use_hd_transform = kwargs.get("use_hd_transform", False)
        self.hd_transform_order = kwargs.get("hd_transform_order", "sub_glb")
        self.crop_size = kwargs.get("crop_size", 448)
        if kwargs.get("image_token_compression_cls") == "avg_pool_2d":
            self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
            self.base_feat_height_target //= 2
            self.base_feat_height_reduction = 2
        else:
            self.image_token_compression = None
        reduced_dim = image_dim_out * self.base_feat_height_reduction**2
        self.glb_GN = nn.Parameter(torch.zeros([1, 1, reduced_dim]))
        self.sub_GN = nn.Parameter(torch.zeros([1, 1, 1, reduced_dim]))
        self.img_projection = nn.Sequential(
            nn.Linear(reduced_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_idx = -2

    def get_img_features(
        self, img_embeds: torch.FloatTensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.FloatTensor:
        outputs = self.img_processor(
            img_embeds,
            output_hidden_states=True,
            patch_attention_mask=attention_mask,
        )
        patch_feature = outputs.hidden_states[self.layer_idx]
        if self.image_token_compression is not None:
            width = int(math.sqrt(patch_feature.size(1)))
            patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
            patch_feature = patch_feature.permute(0, 3, 1, 2)
            patch_feature = self.image_token_compression(patch_feature)
            patch_feature = patch_feature.permute(0, 2, 3, 1)
            patch_feature = patch_feature.view(
                -1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1)
            )
        return patch_feature

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeds: torch.FloatTensor,
        image_sizes=None,
        image_attention_mask=None,
        wte=None,
    ) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        positions_tuple = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=True)
        if len(positions_tuple[0]) == 0:
            return wte(input_ids)
        target_device = self.img_projection[0].weight.device
        target_dtype = self.img_projection[0].weight.dtype
        img_embeds = input_embeds.to(device=target_device, dtype=target_dtype)
        attn_mask = None
        if image_attention_mask is not None and image_attention_mask.numel() > 0:
            attn_mask = image_attention_mask.flatten(0, 1).to(
                dtype=torch.bool, device=target_device
            )
        img_features = self.get_img_features(img_embeds.flatten(0, 1), attn_mask)
        batch_size = input_embeds.shape[0]
        base_feat_size = int(math.sqrt(img_features.shape[1]))
        img_features = img_features.view(
            batch_size, -1, base_feat_size * base_feat_size, self.image_dim_out
        )
        image_sizes = image_sizes.view(-1, 2)
        output_imgs = []
        for idx in range(batch_size):
            height, width = image_sizes[idx]
            height_ratio = max(int(height.item() // self.crop_size), 1)
            width_ratio = max(int(width.item() // self.crop_size), 1)
            global_img = img_features[idx, :1].reshape(
                1, base_feat_size, base_feat_size, self.image_dim_out
            )
            global_img = global_img.reshape(
                1,
                base_feat_size // self.base_feat_height_reduction,
                self.base_feat_height_reduction,
                base_feat_size // self.base_feat_height_reduction,
                self.base_feat_height_reduction,
                self.image_dim_out,
            )
            global_img = global_img.permute(0, 1, 3, 2, 4, 5).reshape(
                1,
                base_feat_size // self.base_feat_height_reduction,
                base_feat_size // self.base_feat_height_reduction,
                -1,
            )
            global_img = torch.cat(
                [
                    global_img,
                    self.sub_GN.repeat(1, base_feat_size // self.base_feat_height_reduction, 1, 1),
                ],
                dim=2,
            ).reshape(1, -1, global_img.shape[-1])
            area_ratio = height_ratio * width_ratio
            sub_img = img_features[idx, 1 : area_ratio + 1]
            sub_img = sub_img.reshape(
                area_ratio, base_feat_size, base_feat_size, self.image_dim_out
            )
            sub_img = sub_img.reshape(
                area_ratio,
                base_feat_size // self.base_feat_height_reduction,
                self.base_feat_height_reduction,
                base_feat_size // self.base_feat_height_reduction,
                self.base_feat_height_reduction,
                self.image_dim_out,
            )
            sub_img = sub_img.permute(0, 1, 3, 2, 4, 5).reshape(
                area_ratio, -1, global_img.shape[-1]
            )
            sub_img = (
                sub_img.reshape(
                    1,
                    height_ratio,
                    width_ratio,
                    base_feat_size // self.base_feat_height_reduction,
                    base_feat_size // self.base_feat_height_reduction,
                    global_img.shape[-1],
                )
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(
                    1,
                    height_ratio * base_feat_size // self.base_feat_height_reduction,
                    width_ratio * base_feat_size // self.base_feat_height_reduction,
                    global_img.shape[-1],
                )
            )
            sub_img = torch.cat(
                [
                    sub_img,
                    self.sub_GN.repeat(
                        1,
                        height_ratio * base_feat_size // self.base_feat_height_reduction,
                        1,
                        1,
                    ),
                ],
                dim=2,
            ).reshape(1, -1, sub_img.shape[-1])
            if self.hd_transform_order == "sub_glb":
                merged = torch.cat([sub_img, self.glb_GN, global_img], dim=1)
            else:
                merged = torch.cat([global_img, self.glb_GN, sub_img], dim=1)
            output_imgs.append(self.img_projection(merged))
        merged_img_set_tensor = (
            torch.cat(output_imgs, dim=1)
            .squeeze(0)
            .to(device=wte.weight.device, dtype=wte.weight.dtype)
        )
        hidden_states = wte(input_ids)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            hidden_states = hidden_states.index_put(
                positions_tuple, merged_img_set_tensor, accumulate=False
            )
        return hidden_states


class Phi4MMAudioEmbedding(nn.Module):
    def __init__(self, config: Phi4MMConfig, **kwargs):
        super().__init__()
        aux = _load_hf_aux_module(
            config._name_or_path, "speech_conformer_encoder.py", "phi4mm_audio_aux"
        )
        encoder_config = getattr(config, "audio_processor", None)["config"]
        self.encoder = aux.ConformerEncoder(**encoder_config)
        hidden_size = config.hidden_size
        downsample_rate = kwargs.get("downsample_rate", 1)
        audio_dim_out = encoder_config["attention_dim"]
        self.audio_dim_in = encoder_config["input_size"]

        def _proj():
            return nn.Sequential(
                nn.Linear(audio_dim_out * downsample_rate, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )

        self.audio_projection = nn.ModuleDict({"speech": _proj(), "vision": _proj()})

    def post_init(self):
        self.encoder.post_init({})

    def get_audio_features(
        self,
        input_embeds: torch.FloatTensor,
        audio_attention_mask: Optional[torch.Tensor],
        audio_projection_mode: str,
    ) -> torch.FloatTensor:
        audio_features, _ = self.encoder(input_embeds, audio_attention_mask)
        return self.audio_projection[audio_projection_mode](audio_features)

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeds: torch.FloatTensor,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode: str = "speech",
        wte=None,
    ) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        positions_tuple = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=True)
        hidden_states = wte(input_ids)
        if len(positions_tuple[0]) == 0:
            return hidden_states
        target_device = self.audio_projection[audio_projection_mode][0].weight.device
        target_dtype = self.audio_projection[audio_projection_mode][0].weight.dtype
        input_embeds = input_embeds.to(device=target_device, dtype=target_dtype)
        audio_set_tensor = self.get_audio_features(
            input_embeds, audio_attention_mask, audio_projection_mode
        )
        merged_audio_set_tensor = torch.cat(
            [audio_set_tensor[i, : audio_embed_sizes[i], :] for i in range(len(audio_embed_sizes))],
            dim=0,
        ).to(device=hidden_states.device, dtype=hidden_states.dtype)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            hidden_states = hidden_states.index_put(
                positions_tuple, merged_audio_set_tensor, accumulate=False
            )
        return hidden_states


class Phi4MMImageAudioEmbedding(nn.Module):
    def __init__(self, config: Phi4MMConfig, **kwargs):
        super().__init__()
        self.image_embed = Phi4MMImageEmbedding(config, **kwargs["image_embd_layer"])
        self.audio_embed = Phi4MMAudioEmbedding(config, **kwargs["audio_embd_layer"])

    def post_init(self):
        self.audio_embed.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes=None,
        image_attention_mask=None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode: str = "speech",
        wte=None,
    ) -> torch.FloatTensor:
        image_hidden_states = None
        audio_hidden_states = None
        image_position_mask = (input_ids == _IMAGE_SPECIAL_TOKEN_ID).unsqueeze(-1)
        non_image_position_mask = ~image_position_mask
        if input_image_embeds is not None and input_image_embeds.numel() > 0:
            image_hidden_states = self.image_embed(
                input_ids=input_ids,
                input_embeds=input_image_embeds,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask,
                wte=wte,
            )
        if input_audio_embeds is not None and input_audio_embeds.numel() > 0:
            audio_hidden_states = self.audio_embed(
                input_ids=input_ids,
                input_embeds=input_audio_embeds,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                audio_projection_mode=audio_projection_mode,
                wte=wte,
            )
        if image_hidden_states is not None and audio_hidden_states is not None:
            dtype = image_hidden_states.dtype
            return image_hidden_states * image_position_mask.to(
                dtype
            ) + audio_hidden_states * non_image_position_mask.to(dtype)
        if image_hidden_states is not None:
            return image_hidden_states
        if audio_hidden_states is not None:
            return audio_hidden_states
        return wte(input_ids)


@dataclass
class Phi4MMModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Phi4MMCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class Phi4MMPreTrainedModel(PreTrainedModel):
    config_class = Phi4MMConfig
    base_model_prefix = "model"
    _no_split_modules = ["Phi4MMDecoderLayer"]
    _tied_weights_keys = ["lm_head.weight"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Phi4MMModel(Phi4MMPreTrainedModel):
    def __init__(self, config: Phi4MMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_tokens_extend = None
        embd_layer = getattr(config, "embd_layer", None)
        if isinstance(embd_layer, dict):
            self.embed_tokens_extend = Phi4MMImageAudioEmbedding(config, **embd_layer)
        self.layers = nn.ModuleList(
            [Phi4MMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Phi4MMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()
        if self.embed_tokens_extend is not None:
            self.embed_tokens_extend.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def set_lora_adapter(self, adapter_name: Optional[str]):
        for layer in self.layers:
            layer.set_adapter(adapter_name)

    def unset_lora_adapter(self):
        for layer in self.layers:
            layer.disable_adapter()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode: str = "speech",
        **kwargs,
    ) -> Phi4MMModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            if self.embed_tokens_extend is not None and (
                (input_image_embeds is not None and input_image_embeds.numel() > 0)
                or (input_audio_embeds is not None and input_audio_embeds.numel() > 0)
            ):
                inputs_embeds = self.embed_tokens_extend(
                    input_ids=input_ids,
                    input_image_embeds=input_image_embeds,
                    image_sizes=image_sizes,
                    image_attention_mask=image_attention_mask,
                    input_audio_embeds=input_audio_embeds,
                    audio_embed_sizes=audio_embed_sizes,
                    audio_attention_mask=audio_attention_mask,
                    audio_projection_mode=audio_projection_mode,
                    wte=self.embed_tokens,
                )
            else:
                inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        return Phi4MMModelOutput(last_hidden_state=hidden_states)


class Phi4MMForCausalLM(Phi4MMPreTrainedModel, GenerationMixin):
    def __init__(self, config: Phi4MMConfig):
        super().__init__(config)
        self.model = Phi4MMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _resolve_input_mode(
        self,
        input_mode,
        input_image_embeds: Optional[torch.Tensor],
        input_audio_embeds: Optional[torch.Tensor],
    ) -> InputMode:
        if isinstance(input_mode, torch.Tensor):
            input_mode = int(input_mode.flatten()[0].item())
        if input_mode is None:
            has_image = input_image_embeds is not None and input_image_embeds.numel() > 0
            has_audio = input_audio_embeds is not None and input_audio_embeds.numel() > 0
            if has_image and has_audio:
                input_mode = InputMode.VISION_SPEECH
            elif has_image:
                input_mode = InputMode.VISION
            elif has_audio:
                input_mode = InputMode.SPEECH
            else:
                input_mode = InputMode.LANGUAGE
        return InputMode(int(input_mode))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        **kwargs,
    ) -> Phi4MMCausalLMOutput:
        mode = self._resolve_input_mode(input_mode, input_image_embeds, input_audio_embeds)
        if mode in [InputMode.VISION, InputMode.VISION_SPEECH]:
            self.model.set_lora_adapter("vision")
            audio_projection_mode = "vision"
        elif mode == InputMode.SPEECH:
            self.model.set_lora_adapter("speech")
            audio_projection_mode = "speech"
        else:
            self.model.unset_lora_adapter()
            audio_projection_mode = "speech"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            input_image_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            audio_projection_mode=audio_projection_mode,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return Phi4MMCausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("Phi4MultimodalConfig", Phi4MMForCausalLM)
