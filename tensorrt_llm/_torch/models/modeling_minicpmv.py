# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from itertools import chain
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)
from transformers.activations import ACT2FN

from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.inputs.multimodal_data import VideoData

from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ContentFormat,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
)
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .checkpoints.hf.minicpmv_weight_mapper import MiniCPMVHfWeightMapper
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (
    _is_mm_disagg,
    find_input_mm_embeds,
    fuse_input_embeds,
    get_attached_multimodal_embeddings,
    get_multimodal_embeddings,
)
from .modeling_utils import register_auto_model, register_vision_encoder

_MINICPMV_MEDIA_PLACEHOLDER = "(<image>./</image>)"
_DEFAULT_VIDEO_MAX_NUM_FRAMES = 180
_DEFAULT_VIDEO_MAX_NUM_PACKING = 3
_DEFAULT_VIDEO_TIME_SCALE = 0.1
_MINICPMV_TOKEN_ATTRS = {
    "im_start": "<image>",
    "im_end": "</image>",
    "slice_start": "<slice>",
    "slice_end": "</slice>",
    "im_id_start": "<image_id>",
    "im_id_end": "</image_id>",
}


class _MiniCPMVTokenizerAdapter:
    """Expose MiniCPM-V tokenizer attributes expected by the remote processor."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        for attr, token in _MINICPMV_TOKEN_ATTRS.items():
            setattr(self, attr, getattr(tokenizer, attr, token))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.tokenizer)

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.decode(*args, **kwargs)

    def convert_tokens_to_ids(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.convert_tokens_to_ids(*args, **kwargs)

    @property
    def bos_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def unk_id(self) -> int:
        return self.tokenizer.unk_token_id

    @property
    def im_start_id(self) -> int:
        return self.convert_tokens_to_ids(self.im_start)

    @property
    def im_end_id(self) -> int:
        return self.convert_tokens_to_ids(self.im_end)

    @property
    def slice_start_id(self) -> int:
        return self.convert_tokens_to_ids(self.slice_start)

    @property
    def slice_end_id(self) -> int:
        return self.convert_tokens_to_ids(self.slice_end)

    @property
    def im_id_start_id(self) -> int:
        return self.convert_tokens_to_ids(self.im_id_start)

    @property
    def im_id_end_id(self) -> int:
        return self.convert_tokens_to_ids(self.im_id_end)

    @property
    def newline_id(self) -> int:
        return self.convert_tokens_to_ids("\n")

    @staticmethod
    def escape(text: str) -> str:
        return text

    @staticmethod
    def unescape(text: str) -> str:
        return text


def _filter_weights(weights: dict[str, torch.Tensor],
                    prefix: str) -> dict[str, torch.Tensor]:
    return {
        name[len(prefix):]: weight
        for name, weight in weights.items() if name.startswith(prefix)
    }


def _resolve_torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return str_dtype_to_torch(dtype)
    return torch.bfloat16


def _get_1d_sincos_pos_embed(embed_dim: int,
                             pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("...,d->...d", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=-1)


def _get_2d_sincos_pos_embed(embed_dim: int,
                             image_size: tuple[int, int]) -> np.ndarray:
    grid_h = np.arange(image_size[0], dtype=np.float32)
    grid_w = np.arange(image_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=-1)


def _get_temporal_sincos_pos_embed(embed_dim: int,
                                   max_size: int) -> np.ndarray:
    positions = np.arange(max_size, dtype=np.float32)
    return _get_1d_sincos_pos_embed(embed_dim, positions)


class MiniCPMVInputProcessor(BaseMultimodalDummyInputsBuilder,
                             BaseMultimodalInputProcessor):
    """Input processor for MiniCPM-V 4.5."""

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: AutoTokenizer,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast,
        )
        self._processor.tokenizer = self._load_processor_tokenizer(
            model_path, trust_remote_code)
        self._dtype = getattr(config, "torch_dtype", torch.bfloat16)
        vocab_size = self.get_vocab_size()
        if vocab_size is None:
            raise ValueError(
                "MiniCPMVInputProcessor requires a resolvable vocabulary size."
            )
        self.tllm_multimodal_token_id = vocab_size + 1

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _has_minicpmv_tokenizer_attrs(self, tokenizer: Any) -> bool:
        return all(
            hasattr(tokenizer, attr)
            for attr in ("im_start_id", "im_end_id", "slice_start_id",
                         "slice_end_id"))

    def _load_processor_tokenizer(self, model_path: str,
                                  trust_remote_code: bool) -> Any:
        processor_tokenizer = getattr(self._processor, "tokenizer", None)
        if self._has_minicpmv_tokenizer_attrs(processor_tokenizer):
            return processor_tokenizer

        try:
            processor_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast,
            )
        except Exception:
            processor_tokenizer = getattr(self._tokenizer, "tokenizer",
                                          self._tokenizer)

        if self._has_minicpmv_tokenizer_attrs(processor_tokenizer):
            return processor_tokenizer
        return _MiniCPMVTokenizerAdapter(processor_tokenizer)

    def get_mm_token_ids(self) -> torch.Tensor:
        return torch.tensor([self.tllm_multimodal_token_id], dtype=torch.int64)

    def _get_single_video_data(self, video: Any) -> Optional[VideoData]:
        if isinstance(video, VideoData):
            return video
        if isinstance(video, list) and len(video) == 1 and isinstance(
                video[0], VideoData):
            return video[0]
        return None

    def _to_nested_int_list(self, value: Any) -> list[list[int]]:
        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()
        result = []
        for item in value:
            if torch.is_tensor(item):
                item = item.detach().cpu().tolist()
            if isinstance(item, (int, np.integer)):
                result.append([int(item)])
            else:
                result.append([int(v) for v in item])
        return result

    def _normalize_video_data(self, video: Any) -> list[Any]:
        video_data = self._get_single_video_data(video)
        if video_data is not None:
            return list(video_data.frames)
        if isinstance(video, list):
            if any(isinstance(item, VideoData) for item in video):
                raise ValueError(
                    "MiniCPMVInputProcessor supports one video per request.")
            if len(video) == 1 and isinstance(video[0], list):
                return list(video[0])
        return list(video)

    def _get_media_inputs(self,
                          mm_data: dict[str, Any]) -> tuple[str, Any, Any]:
        if "image" in mm_data and "video" in mm_data:
            raise ValueError(
                "MiniCPMVInputProcessor supports either image or video per request, not both."
            )
        if "video" in mm_data:
            video = mm_data["video"]
            return "video", self._normalize_video_data(video), video
        image = mm_data.get("image")
        return "image", image, image

    def _build_video_temporal_ids(
        self,
        video: Any,
        frames: list[Any],
        mm_processor_kwargs: dict[str, Any],
    ) -> Optional[list[list[int]]]:
        enable_temporal_ids = mm_processor_kwargs.pop(
            "enable_video_temporal_ids",
            mm_processor_kwargs.pop("use_video_temporal_ids", False),
        )
        if not enable_temporal_ids:
            return None

        max_num_frames = int(
            mm_processor_kwargs.pop("video_max_num_frames",
                                    _DEFAULT_VIDEO_MAX_NUM_FRAMES))
        max_num_packing = int(
            mm_processor_kwargs.pop("video_max_num_packing",
                                    _DEFAULT_VIDEO_MAX_NUM_PACKING))
        force_packing = mm_processor_kwargs.pop("video_force_packing", None)
        time_scale = float(
            mm_processor_kwargs.pop("video_time_scale",
                                    _DEFAULT_VIDEO_TIME_SCALE))

        if max_num_frames <= 0:
            raise ValueError("video_max_num_frames must be positive.")
        if max_num_packing <= 0:
            raise ValueError("video_max_num_packing must be positive.")
        if time_scale <= 0:
            raise ValueError("video_time_scale must be positive.")

        packing_nums = min(
            max(1, int(np.ceil(len(frames) / max_num_frames))),
            max_num_packing,
        )
        if force_packing is not None:
            packing_nums = min(max(1, int(force_packing)), max_num_packing)

        video_data = self._get_single_video_data(video)
        metadata = video_data.metadata if video_data is not None else {}
        frame_indices = metadata.get("frames_indices")
        fps = metadata.get("fps")
        if frame_indices is not None and fps is not None and fps > 0:
            if len(frame_indices) != len(frames):
                frame_indices = np.arange(len(frames), dtype=np.float32)
            frame_ts = np.asarray(frame_indices, dtype=np.float32) / float(fps)
            temporal_ids = np.rint(frame_ts / time_scale).astype(np.int32)
        else:
            temporal_ids = np.arange(len(frames), dtype=np.int32)

        return [
            temporal_ids[start:start + packing_nums].tolist()
            for start in range(0, len(temporal_ids), packing_nums)
        ]

    def _expand_video_placeholders(self, text_prompt: str,
                                   frames: list[Any]) -> str:
        num_frames = len(frames)
        placeholder_count = text_prompt.count(_MINICPMV_MEDIA_PLACEHOLDER)
        if placeholder_count == num_frames:
            return text_prompt

        frame_placeholders = _MINICPMV_MEDIA_PLACEHOLDER * num_frames
        if placeholder_count == 0:
            return frame_placeholders + text_prompt
        if placeholder_count == 1:
            return text_prompt.replace(_MINICPMV_MEDIA_PLACEHOLDER,
                                       frame_placeholders, 1)
        raise ValueError(
            "MiniCPMV video prompt must contain either one video placeholder "
            f"or one placeholder per frame, got {placeholder_count} placeholders "
            f"for {num_frames} frames.")

    def _preprocess(
        self,
        text_prompt: str,
        mm_data: dict[str, Any],
        mm_processor_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        modality, media, raw_media = self._get_media_inputs(mm_data)
        temporal_ids = mm_processor_kwargs.pop("temporal_ids", None)
        if modality == "video":
            if temporal_ids is None:
                temporal_ids = self._build_video_temporal_ids(
                    raw_media, media, mm_processor_kwargs)
            text_prompt = self._expand_video_placeholders(text_prompt, media)
        return self.processor(
            text_prompt,
            media,
            temporal_ids=temporal_ids,
            return_tensors="pt",
            **mm_processor_kwargs,
        )

    def _postprocess(
        self,
        input_ids: torch.Tensor,
        image_bound: list[torch.Tensor],
    ) -> torch.Tensor:
        processed_ids = input_ids.clone()
        if processed_ids.dim() == 1:
            processed_ids = processed_ids.unsqueeze(0)
        for batch_idx, bounds in enumerate(image_bound):
            for start, end in bounds:
                processed_ids[batch_idx, int(start):int(
                    end)] = self.tllm_multimodal_token_id
        return processed_ids

    def get_num_tokens_per_image(
        self,
        *,
        image: Image.Image | torch.Tensor,
        **kwargs: Any,
    ) -> int:
        image_processor = self.processor.image_processor
        if isinstance(image, torch.Tensor):
            image = image_processor.to_pil_image(image).convert("RGB")
        max_slice_nums = kwargs.get("max_slice_nums", None)
        num_slices = len(
            image_processor.get_sliced_images(image,
                                              max_slice_nums=max_slice_nums))
        return int(image_processor.image_feature_size * num_slices)

    def get_num_tokens_per_video(
        self,
        *,
        video: list[Image.Image | torch.Tensor],
        temporal_ids: Optional[Any] = None,
        **kwargs: Any,
    ) -> int:
        if temporal_ids is None:
            return sum(
                self.get_num_tokens_per_image(image=frame, **kwargs)
                for frame in video)

        total_tokens = 0
        frame_offset = 0
        for temporal_group in temporal_ids:
            if len(temporal_group) == 0:
                continue
            total_tokens += self.get_num_tokens_per_image(
                image=video[frame_offset], **kwargs)
            frame_offset += len(temporal_group)
        return total_tokens

    def call_with_text_prompt(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> tuple[list[int], Optional[ExtraProcessedInputs]]:
        text_prompt = inputs.get("prompt")
        if text_prompt is None:
            raise ValueError("MiniCPMVInputProcessor requires a text prompt.")

        mm_data = inputs.get("multi_modal_data") or {}
        mm_processor_kwargs = dict(inputs.get("mm_processor_kwargs", {}) or {})
        if not mm_data:
            input_ids = self.tokenizer.encode(text_prompt)
            return torch.tensor(input_ids, dtype=torch.int32).tolist(), None

        modality, _, _ = self._get_media_inputs(mm_data)
        processed = self._preprocess(text_prompt, mm_data,
                                     mm_processor_kwargs)
        input_ids = self._postprocess(processed["input_ids"],
                                      processed["image_bound"])
        image_bound = processed["image_bound"][0]
        media_data = {
            "pixel_values": processed["pixel_values"][0],
            "image_sizes": processed["image_sizes"][0],
            "image_bound": image_bound,
            "tgt_sizes": processed["tgt_sizes"][0],
        }
        if "temporal_ids" in processed:
            media_data["temporal_ids"] = self._to_nested_int_list(
                processed["temporal_ids"][0])

        multimodal_data = {modality: media_data}
        return input_ids[0].to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data
        }


class MiniCPMVSiglipVisionEmbeddings(nn.Module):
    """NaViT SigLIP patch embedding with target-size position buckets."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=getattr(config, "num_channels", 3),
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.position_embedding = nn.Embedding(
            self.num_patches_per_side**2, self.embed_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = pixel_values.size(0)
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h = max_im_h // self.patch_size
        max_nb_patches_w = max_im_w // self.patch_size
        device = patch_attention_mask.device
        boundaries = torch.arange(
            1 / self.num_patches_per_side,
            1.0,
            1 / self.num_patches_per_side,
            device=device,
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w),
            fill_value=0,
            dtype=torch.long,
            device=device,
        )

        for batch_idx, patch_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = int(tgt_sizes[batch_idx][0])
                nb_patches_w = int(tgt_sizes[batch_idx][1])
            else:
                nb_patches_h = int(patch_mask[:, 0].sum())
                nb_patches_w = int(patch_mask[0].sum())

            fractional_coords_h = torch.arange(0,
                                               1 - 1e-6,
                                               1 / nb_patches_h,
                                               device=device)
            fractional_coords_w = torch.arange(0,
                                               1 - 1e-6,
                                               1 / nb_patches_w,
                                               device=device)
            bucket_coords_h = torch.bucketize(fractional_coords_h,
                                              boundaries,
                                              right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w,
                                              boundaries,
                                              right=True)
            pos_ids = (
                bucket_coords_h[:, None] * self.num_patches_per_side +
                bucket_coords_w).flatten()
            position_ids[batch_idx][patch_mask.view(-1)] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)
        return embeddings + self.position_embedding(position_ids)


class MiniCPMVSiglipAttention(nn.Module):
    """Eager SigLIP attention used by the correctness-first MiniCPM-V port."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states,
                                    key_states.transpose(2,
                                                         3)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1,
                                 dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights,
                                 p=self.dropout,
                                 training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return self.out_proj(attn_output.reshape(batch_size, q_len,
                                                 self.embed_dim))


class MiniCPMVSiglipMLP(nn.Module):

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class MiniCPMVSiglipEncoderLayer(nn.Module):

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.self_attn = MiniCPMVSiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.mlp = MiniCPMVSiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states,
                                                  attention_mask)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        return residual + self.mlp(hidden_states)


class MiniCPMVSiglipEncoder(nn.Module):

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MiniCPMVSiglipEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class MiniCPMVNavitSiglipVisionModel(nn.Module):
    """NaViT SigLIP vision tower for MiniCPM-V 4.5."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config.vision_config
        self.dtype = _resolve_torch_dtype(
            getattr(model_config.pretrained_config, "torch_dtype",
                    torch.bfloat16))
        self.embeddings = MiniCPMVSiglipVisionEmbeddings(self.config)
        self.encoder = MiniCPMVSiglipEncoder(self.config)
        if getattr(model_config.pretrained_config, "drop_vision_last_layer",
                   False):
            self.encoder.layers = self.encoder.layers[:-1]
        self.post_layernorm = nn.LayerNorm(self.config.hidden_size,
                                           eps=self.config.layer_norm_eps)
        self.to(dtype=self.dtype)

    def prepare_attn_metadata(
        self,
        patch_attention_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        del tgt_sizes
        flat_mask = patch_attention_mask.view(patch_attention_mask.shape[0],
                                              -1)
        mask = ~flat_mask[:, None, None, :]
        min_value = torch.finfo(self.dtype).min
        return mask.to(dtype=self.dtype) * min_value

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            tgt_sizes=tgt_sizes,
        )
        attention_mask = self.prepare_attn_metadata(patch_attention_mask,
                                                    tgt_sizes)
        hidden_states = self.encoder(hidden_states, attention_mask)
        return self.post_layernorm(hidden_states)

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(_filter_weights(weights, "vpm."), strict=True)
        self.to(dtype=self.dtype)


class MiniCPMVResampler(nn.Module):
    """Perceiver-style resampler for MiniCPM-V 4.5."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.dtype = _resolve_torch_dtype(
            getattr(self.config, "torch_dtype", torch.bfloat16))
        vision_config = self.config.vision_config
        self.num_queries = self.config.query_num
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.embed_dim // 128
        self.max_size = (70, 70)
        self.max_temporal_size = 72000
        self.batch_infer = getattr(self.config, "batch_3d_resampler", True)
        self.query = nn.Parameter(torch.zeros(self.num_queries,
                                             self.embed_dim))
        self.kv_proj = nn.Linear(vision_config.hidden_size,
                                 self.embed_dim,
                                 bias=False)
        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        self.ln_q = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ln_post = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.proj = nn.Parameter((self.embed_dim**-0.5) *
                                 torch.randn(self.embed_dim, self.embed_dim))
        self._set_2d_pos_cache(self.max_size)
        self._set_temporal_pos_cache(self.max_temporal_size)
        self.to(dtype=self.dtype)

    def _set_2d_pos_cache(self,
                          max_size: tuple[int, int],
                          device: torch.device | str = "cpu") -> None:
        pos_embed = torch.from_numpy(
            _get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(
                device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes: torch.Tensor,
                          device: torch.device) -> None:
        max_h = int(torch.max(tgt_sizes[:, 0]))
        max_w = int(torch.max(tgt_sizes[:, 1]))
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = (max(max_h, self.max_size[0]),
                             max(max_w, self.max_size[1]))
            self._set_2d_pos_cache(self.max_size, device)

    def _set_temporal_pos_cache(self,
                                max_temporal_size: int,
                                device: torch.device | str = "cpu") -> None:
        pos_embed = torch.from_numpy(
            _get_temporal_sincos_pos_embed(self.embed_dim,
                                           max_temporal_size)).float().to(
                                               device)
        self.register_buffer("temporal_pos_embed", pos_embed, persistent=False)

    def _adjust_temporal_pos_cache(self, max_temporal_size: int,
                                   device: torch.device) -> None:
        if max_temporal_size > self.max_temporal_size:
            self.max_temporal_size = max_temporal_size
            self._set_temporal_pos_cache(max_temporal_size, device)

    def _repeat(self, query: torch.Tensor, batch_size: int) -> torch.Tensor:
        return query.unsqueeze(1).repeat(1, batch_size, 1)

    def _batch_attn_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_embed_temporal: list[torch.Tensor],
        temporal_ids: Optional[list[list[int]]],
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = k.shape[1]
        if pos_embed_temporal:
            k = k + torch.stack(pos_embed_temporal, dim=0)
            assert temporal_ids is not None
            batch_size = len(temporal_ids)
            merge_k = []
            merge_v = []
            merge_key_padding_mask = []
            start = 0
            for temporal_group in temporal_ids:
                end = start + len(temporal_group)
                merge_k.append(k[:, start:end, :].permute(1, 0, 2).reshape(
                    -1, self.embed_dim))
                merge_v.append(v[:, start:end, :].permute(1, 0, 2).reshape(
                    -1, self.embed_dim))
                merge_key_padding_mask.append(
                    key_padding_mask[start:end, :].reshape(-1, 1))
                start = end
            k = torch.nn.utils.rnn.pad_sequence(
                merge_k, batch_first=True,
                padding_value=0.0).permute(1, 0, 2)
            v = torch.nn.utils.rnn.pad_sequence(
                merge_v, batch_first=True,
                padding_value=0.0).permute(1, 0, 2)
            key_padding_mask = torch.nn.utils.rnn.pad_sequence(
                merge_key_padding_mask, batch_first=True,
                padding_value=True).squeeze(-1)
        return self.attn(self._repeat(q, batch_size),
                         k,
                         v,
                         key_padding_mask=key_padding_mask)[0]

    def _foreach_attn_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_embed_temporal: list[torch.Tensor],
        temporal_ids: Optional[list[list[int]]],
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = k.shape[1]
        if pos_embed_temporal:
            k = k + torch.stack(pos_embed_temporal, dim=0)
            assert temporal_ids is not None
            out_list = []
            start = 0
            for temporal_group in temporal_ids:
                end = start + len(temporal_group)
                curr_k = k[:, start:end, :].reshape(-1, self.embed_dim)
                curr_v = v[:, start:end, :].reshape(-1, self.embed_dim)
                curr_mask = key_padding_mask[start:end, :].reshape(-1)
                out_list.append(
                    self.attn(q, curr_k, curr_v,
                              key_padding_mask=curr_mask)[0])
                start = end
            return torch.stack(out_list, dim=1)
        return self.attn(self._repeat(q, batch_size),
                         k,
                         v,
                         key_padding_mask=key_padding_mask)[0]

    def forward(
        self,
        vision_hidden_states: torch.Tensor,
        tgt_sizes: torch.Tensor,
        temporal_ids: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        device = vision_hidden_states.device
        dtype = vision_hidden_states.dtype
        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        self._adjust_pos_cache(tgt_sizes, device=device)

        temporal_ids_flatten = None
        temporal_pos_emb = False
        if temporal_ids is not None:
            temporal_ids_flatten = list(chain.from_iterable(temporal_ids))
            max_temporal_size = max(temporal_ids_flatten) + 1
            if max_temporal_size > -1:
                temporal_pos_emb = True
            if max_temporal_size > self.max_temporal_size:
                self._adjust_temporal_pos_cache(max_temporal_size, device)

        max_patch_len = vision_hidden_states.shape[1]
        key_padding_mask = torch.arange(
            max_patch_len,
            device=device,
        )[None, :] >= patch_len[:, None]
        pos_embed_2d = []
        pos_embed_temporal = []
        for i in range(vision_hidden_states.shape[0]):
            tgt_h, tgt_w = (int(tgt_sizes[i][0]), int(tgt_sizes[i][1]))
            pos_embed_2d.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape(
                tgt_h * tgt_w, -1).to(device=device, dtype=dtype))
            if temporal_pos_emb:
                assert temporal_ids_flatten is not None
                temporal_id = temporal_ids_flatten[i]
                if temporal_id == -1:
                    pos_embed_temporal.append(
                        torch.zeros(self.embed_dim, dtype=dtype,
                                    device=device))
                else:
                    pos_embed_temporal.append(
                        self.temporal_pos_embed[temporal_id].to(device=device,
                                                                dtype=dtype))

        pos_embed_2d = torch.nn.utils.rnn.pad_sequence(
            pos_embed_2d, batch_first=True,
            padding_value=0.0).permute(1, 0, 2)
        x = self.ln_kv(self.kv_proj(vision_hidden_states)).permute(1, 0, 2)
        q = self.ln_q(self.query)
        v = x
        k = x + pos_embed_2d
        if self.batch_infer:
            out = self._batch_attn_forward(q, k, v, pos_embed_temporal,
                                           temporal_ids, key_padding_mask)
        else:
            out = self._foreach_attn_forward(q, k, v, pos_embed_temporal,
                                             temporal_ids, key_padding_mask)
        x = out.permute(1, 0, 2)
        return self.ln_post(x) @ self.proj

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(_filter_weights(weights, "resampler."),
                             strict=True)
        self.to(dtype=self.dtype)


class MiniCPMVVisionModel(nn.Module):
    """MiniCPM-V vision encoder plus resampler."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.vision_tower = MiniCPMVNavitSiglipVisionModel(model_config)
        self.resampler = MiniCPMVResampler(model_config)

    def _parse_and_batch_multimodal_data(
        self,
        multimodal_params: list[MultimodalParams],
    ) -> dict[str, Any]:
        all_pixel_values = []
        all_tgt_sizes = []
        request_slice_counts = []
        all_temporal_ids = []
        has_temporal_ids = False

        for multimodal_param in multimodal_params:
            data = multimodal_param.multimodal_data
            media_data = data.get("image") or data.get("video") or {}
            pixel_values = media_data.get("pixel_values") or []
            tgt_sizes = media_data.get("tgt_sizes")
            temporal_ids = media_data.get("temporal_ids")
            request_slice_counts.append(len(pixel_values))
            all_pixel_values.extend([
                pixel_value.flatten(end_dim=1).permute(1, 0)
                for pixel_value in pixel_values
            ])
            if tgt_sizes is not None and len(pixel_values) > 0:
                all_tgt_sizes.append(torch.as_tensor(tgt_sizes))
            if temporal_ids is not None:
                has_temporal_ids = True
                all_temporal_ids.extend(temporal_ids)

        if not all_pixel_values:
            return {
                "pixel_values": None,
                "tgt_sizes": None,
                "patch_attention_mask": None,
                "temporal_ids": None,
                "request_slice_counts": request_slice_counts,
            }

        tgt_sizes = torch.vstack(all_tgt_sizes).to(
            device=all_pixel_values[0].device, dtype=torch.int32)
        padded_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values, batch_first=True, padding_value=0.0)
        batch_size, length, _ = padded_pixel_values.shape
        padded_pixel_values = padded_pixel_values.permute(0, 2, 1).reshape(
            batch_size, 3, -1, length)
        max_patches = length // self.config.vision_config.patch_size
        patch_lens = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        patch_attention_mask = torch.arange(
            max_patches,
            device=padded_pixel_values.device,
        )[None, None, :] < patch_lens[:, None, None]

        return {
            "pixel_values": padded_pixel_values,
            "tgt_sizes": tgt_sizes,
            "patch_attention_mask": patch_attention_mask,
            "temporal_ids": all_temporal_ids if has_temporal_ids else None,
            "request_slice_counts": request_slice_counts,
        }

    @torch.inference_mode()
    def forward(
        self, multimodal_params: list[MultimodalParams]
    ) -> list[torch.Tensor]:
        batched = self._parse_and_batch_multimodal_data(multimodal_params)
        pixel_values = batched["pixel_values"]
        if pixel_values is None:
            return []

        pixel_values = pixel_values.to(self.vision_tower.dtype)
        tgt_sizes = batched["tgt_sizes"]
        patch_attention_mask = batched["patch_attention_mask"]
        vision_batch_size = getattr(self.config, "vision_batch_size", 16)
        if pixel_values.shape[0] > vision_batch_size:
            hidden_states = []
            for start in range(0, pixel_values.shape[0], vision_batch_size):
                end = start + vision_batch_size
                hidden_states.append(
                    self.vision_tower(
                        pixel_values[start:end],
                        patch_attention_mask[start:end],
                        tgt_sizes[start:end],
                    ))
            vision_hidden_states = torch.cat(hidden_states, dim=0)
        else:
            vision_hidden_states = self.vision_tower(pixel_values,
                                                     patch_attention_mask,
                                                     tgt_sizes)

        vision_embedding = self.resampler(
            vision_hidden_states,
            tgt_sizes,
            batched["temporal_ids"],
        )
        return [vision_embedding.reshape(-1, vision_embedding.shape[-1])]

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.vision_tower.load_weights(weights)
        self.resampler.load_weights(weights)


@register_vision_encoder(MiniCPMVVisionModel,
                         vlm_base_model=MiniCPMVNavitSiglipVisionModel)
@register_auto_model("MiniCPMV")
@register_input_processor(
    MiniCPMVInputProcessor,
    model_type="minicpmv",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": _MINICPMV_MEDIA_PLACEHOLDER,
            "video": _MINICPMV_MEDIA_PLACEHOLDER,
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        content_format=ContentFormat.STRING,
    ),
)
class MiniCPMVForConditionalGeneration(PreTrainedModel):
    """TensorRT-LLM model wrapper for MiniCPM-V 4.5."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        config = model_config.pretrained_config
        super().__init__(config)
        if hasattr(self, "llm"):
            return

        self.model_config = model_config

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config.architectures = ["Qwen3ForCausalLM"]
        llm_model_config.extra_attrs = model_config.extra_attrs
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.mm_encoder = None
        if not _is_mm_disagg():
            self.mm_encoder = MiniCPMVVisionModel(copy.deepcopy(model_config)).eval()

        self.post_config()

    def post_config(self) -> None:
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        return [
            "image.pixel_values",
            "image.tgt_sizes",
            "image.temporal_ids",
            "video.pixel_values",
            "video.tgt_sizes",
            "video.temporal_ids",
            "multimodal_embedding",
        ]

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        num_context_requests = attn_metadata.num_contexts
        multimodal_params = kwargs.get("multimodal_params", [])
        mm_multimodal_params = self._get_requests_with_mm_data(
            multimodal_params[:num_context_requests]
        )
        mm_embeds = []
        if len(mm_multimodal_params) > 0:
            if self.mm_encoder is not None:
                mm_embeds = get_multimodal_embeddings(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=mm_multimodal_params,
                )
            else:
                mm_embeds = get_attached_multimodal_embeddings(mm_multimodal_params)
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_multimodal_params)

        if inputs_embeds is None:
            input_ids, inputs_embeds = fuse_input_embeds(
                self.llm.model.embed_tokens,
                input_ids,
                mm_embeds,
                **kwargs,
            )
        else:
            input_ids = None
        return self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
        )

    def _get_requests_with_mm_data(
        self,
        multimodal_params: list[MultimodalParams],
    ) -> list[MultimodalParams]:
        mm_multimodal_params = []
        for multimodal_param in multimodal_params:
            data = multimodal_param.multimodal_data
            if (
                data.get("image", {}).get("pixel_values") is not None
                or data.get("video", {}).get("pixel_values") is not None
                or data.get("multimodal_embedding") is not None
            ):
                mm_multimodal_params.append(multimodal_param)
        return mm_multimodal_params

    def load_weights(
        self,
        weights: dict[str, torch.Tensor],
        weight_mapper: Optional[MiniCPMVHfWeightMapper] = None,
    ) -> None:
        if not isinstance(weight_mapper, MiniCPMVHfWeightMapper):
            weight_mapper = MiniCPMVHfWeightMapper()

        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights)
            if hasattr(weights, "mark_consumed"):
                weights.mark_consumed("vpm")
                weights.mark_consumed("resampler")

        weight_mapper.init_model_and_config(self.llm, self.model_config)
        mapped_weights = weight_mapper.preprocess_weights(weights)
        llm_weights = {
            name: weight
            for name, weight in mapped_weights.items()
            if not name.startswith(("vpm.", "resampler."))
        }
        self.llm.load_weights(llm_weights, weight_mapper)
        if hasattr(weights, "mark_consumed"):
            weights.mark_consumed("llm")
