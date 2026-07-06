# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniCPM-V 4.6 (openbmb/MiniCPM-V-4.6) for the TRT-LLM PyTorch backend.

Composition:
  * A SigLIP2-style variable-resolution ViT (NaViT-packed) with an intermediate
    window-attention merger and a final 2x2 downsample-MLP merger, ported to
    TRT-LLM modules (``Attention`` / ``Linear`` / ``LayerNorm`` / ``MLP``).
  * A Qwen3.5 dense hybrid (linear + full attention) text tower, resolved from
    ``text_config`` through TRT-LLM's ``AutoModelForCausalLM`` (Qwen3_5ForCausalLM).

Both image and video modalities are wired up. Video reuses the exact same
NaViT-packed vision path: the HF processor packs every frame/slice into
``pixel_values_videos`` / ``target_sizes_videos`` with the same layout as image
inputs, and HF's ``get_video_features`` is (for batch/beam == 1) an identity
repack of ``get_image_features`` -- so the ported ``_get_image_features`` is
correct for video too.

The vision tower uses learned (interpolated) absolute position embeddings and no
rotary embedding, so the text tower's standard 1-D RoPE path is untouched (this
model does *not* use mRoPE).
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from packaging.version import Version
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)

from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.mapping import Mapping

from ..._utils import nvtx_range
from ...inputs import (BaseMultimodalDummyInputsBuilder,
                       BaseMultimodalInputProcessor, ContentFormat,
                       ExtraProcessedInputs, MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PredefinedAttentionMask
from ..attention_backend.utils import get_attention_backend
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear
from ..modules.mlp import MLP
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (_is_mm_disagg, find_input_mm_embeds,
                                        fuse_input_embeds,
                                        get_multimodal_embeddings)
from .modeling_qwen2vl import _prepare_qwen_vl_vision_attn_metadata
from .modeling_utils import (QuantConfig, _load_weights_impl,
                             register_auto_model)


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """``gelu_pytorch_tanh`` activation used by the ViT encoder / window merger."""
    return F.gelu(x, approximate="tanh")


# MiniCPM-V 4.6 was upstreamed into transformers as a native model
# (``minicpmv4_6``); its processor/config classes only ship in >=5.7.0, and the
# checkpoint carries no remote code (``auto_map``) to fall back on. The TRT-LLM
# model/config code is import-safe on older releases, so only the input
# processor (which loads the native ``MiniCPMV4_6Processor``) hard-requires it.
_MINICPMV4_6_MIN_TRANSFORMERS = "5.7.0"


def _ensure_transformers_supports_minicpmv4_6() -> None:
    """Raise a clear error if transformers is too old for the HF processor."""
    installed = transformers.__version__
    if Version(installed) < Version(_MINICPMV4_6_MIN_TRANSFORMERS):
        raise RuntimeError(
            f"MiniCPM-V 4.6 requires transformers>="
            f"{_MINICPMV4_6_MIN_TRANSFORMERS} for its native processor/config "
            f"(installed: {installed}). This model was upstreamed into "
            f"transformers and ships no remote code to fall back on. Please "
            f"upgrade, e.g. `pip install 'transformers>="
            f"{_MINICPMV4_6_MIN_TRANSFORMERS}'`.")


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------
class MiniCPMV4_6VisionEmbeddings(nn.Module):
    """Conv2d patch embedding + interpolated absolute position embeddings.

    Adapted from SigLIP's NaViT variant: images keep their native aspect ratio,
    so position ids are bucketized per (h, w) grid rather than looked up from a
    fixed square grid.
    """

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = config.image_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            dtype=dtype,
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_positions = self.num_patches_per_side**2
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim,
                                               dtype=dtype)

    def forward(self, pixel_values: torch.Tensor,
                target_sizes: torch.Tensor) -> torch.Tensor:
        # pixel_values: NaViT-packed [1, C, patch_size, total_patch_axis]
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)  # [1, T, D]

        nps = self.num_patches_per_side
        boundaries = torch.arange(1 / nps, 1.0, 1 / nps)
        position_embeddings = []
        # target_sizes lives on CPU (used for integer arithmetic here).
        for target_size in target_sizes:
            h, w = int(target_size[0]), int(target_size[1])
            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / w)
            bucket_coords_h = torch.bucketize(fractional_coords_h,
                                              boundaries,
                                              right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w,
                                              boundaries,
                                              right=True)
            pos_ids = (bucket_coords_h[:, None] * nps +
                       bucket_coords_w).flatten().to(
                           self.position_embedding.weight.device)
            position_embeddings.append(self.position_embedding(pos_ids))

        position_embeddings = torch.concat(position_embeddings,
                                           dim=0).unsqueeze(0)
        return embeddings + position_embeddings


class MiniCPMV4_6VisionAttention(Attention):
    """Variable-length full self-attention (no RoPE) for the ViT tower.

    Mirrors ``Qwen2_5_VLVisionAttention``: fused ``qkv_proj`` / ``o_proj``, a
    per-segment ``attn_metadata`` (cu_seqlens) built by the caller, and a
    custom forward that skips the generic RoPE path. Runs replicated (tp=1).
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int):
        config = model_config.pretrained_config.vision_config
        text_config = model_config.pretrained_config.text_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            max_position_embeddings=text_config.max_position_embeddings,
            bias=True,
            pos_embd_params=None,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            reduce_output=False,
            head_dim=config.hidden_size // config.num_attention_heads,
        )
        # Vision attention runs eagerly from the VL wrapper (outside the
        # compiled LM region); unregister so `forward_impl` uses the eager path
        # with the vision-local `attn_metadata` instead of the LM's.
        if self.register_to_config:
            model_config.extra_attrs.get("attn_layers",
                                         {}).pop(self.layer_idx_str, None)
            self.register_to_config = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv, None, None)
        q, k, v = self.convert_qkv(q, k, v)
        output = self.forward_impl(
            q=q,
            k=k,
            v=v,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
            attention_window_size=None,
            attention_mask_data=None,
            mrope_config=None,
            attention_sinks=None,
        )
        return self.o_proj(output, layer_idx=self.layer_idx)

    def forward_window(self, hidden_states: torch.Tensor,
                       window_len: int) -> torch.Tensor:
        """Fixed-size window self-attention via batched SDPA.

        The ViT window merger reorders tokens into consecutive, equal-length
        windows (``window_h * window_w`` tokens each) that never attend across
        window boundaries. Running this as one variable-length fused-attention
        call produces one segment per window; with many frames/slices the
        segment count is large enough to overflow the fused-attention kernel's
        launch limits. Because every window is the *same* small length, it maps
        exactly onto a regular batched attention instead -- which is both
        cheaper and free of that segment-count ceiling.
        """
        num_tokens = hidden_states.shape[0]
        num_windows = num_tokens // window_len
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv)

        def to_windows(x: torch.Tensor) -> torch.Tensor:
            # [num_windows * window_len, H*D] -> [num_windows, H, window_len, D]
            return x.view(num_windows, window_len, self.num_heads,
                          self.head_dim).transpose(1, 2)

        q, k, v = to_windows(q), to_windows(k), to_windows(v)
        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).reshape(
            num_tokens, self.num_heads * self.head_dim)
        return self.o_proj(output, layer_idx=self.layer_idx)


class MiniCPMV4_6VisionMLP(MLP):
    """ViT feed-forward (``fc1`` -> gelu_tanh -> ``fc2``) on TRT-LLM ``MLP``."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int):
        config = model_config.pretrained_config.vision_config
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=True,
            activation=_gelu_tanh,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )


class MiniCPMV4_6VisionEncoderLayer(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config.vision_config
        dtype = config.torch_dtype
        self.layer_norm1 = LayerNorm(hidden_size=config.hidden_size,
                                     eps=config.layer_norm_eps,
                                     dtype=dtype)
        self.self_attn = MiniCPMV4_6VisionAttention(model_config, layer_idx)
        self.layer_norm2 = LayerNorm(hidden_size=config.hidden_size,
                                     eps=config.layer_norm_eps,
                                     dtype=dtype)
        self.mlp = MiniCPMV4_6VisionMLP(model_config, layer_idx)

    def forward(self, hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attn_metadata)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiniCPMV4_6VisionEncoder(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        config = model_config.pretrained_config.vision_config
        self.layers = nn.ModuleList([
            MiniCPMV4_6VisionEncoderLayer(model_config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])


class MiniCPMV4_6ViTWindowAttentionMerger(nn.Module):
    """Intermediate window self-attention + 2x2 window merge (inserted mid-ViT).

    Reorders tokens into 2x2 windows, runs window-local self-attention, then
    merges every 2x2 window into one token (with a mean-pool residual) through a
    block-structured MLP (``linear_1`` -> gelu_tanh -> ``linear_2``). Halves the
    per-axis patch grid.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        config = model_config.pretrained_config.vision_config
        dtype = config.torch_dtype
        self.window_kernel_size = tuple(config.window_kernel_size)
        self.embed_dim = config.hidden_size
        mapping = model_config.mapping

        self.self_attn = MiniCPMV4_6VisionAttention(
            model_config, layer_idx=config.num_hidden_layers)
        self.layer_norm1 = LayerNorm(hidden_size=self.embed_dim,
                                     eps=config.layer_norm_eps,
                                     dtype=dtype)
        self.pre_norm = LayerNorm(hidden_size=config.window_hidden_size,
                                  eps=config.layer_norm_eps,
                                  dtype=dtype)
        self.linear_1 = Linear(config.window_hidden_size,
                               config.window_intermediate_size,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=None)
        self.linear_2 = Linear(config.window_intermediate_size,
                               self.embed_dim,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=None)

    def get_window_index(
        self, target_sizes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        window_h, window_w = self.window_kernel_size
        max_seqlens = window_h * window_w

        window_index_list = []
        cu_seqlens = [0]
        token_offset = 0
        for height, width in target_sizes:
            height, width = int(height), int(width)
            if height % window_h != 0 or width % window_w != 0:
                raise ValueError(
                    f"height={height}, width={width} must be divisible by "
                    f"window size {self.window_kernel_size}")
            index = torch.arange(height * width).reshape(height, width)
            num_windows_h = height // window_h
            num_windows_w = width // window_w
            num_windows = num_windows_h * num_windows_w
            index = index.reshape(num_windows_h, window_h, num_windows_w,
                                  window_w)
            index = index.permute(0, 2, 1, 3).reshape(num_windows,
                                                       window_h * window_w)
            window_index_list.append(index.reshape(-1) + token_offset)
            cu_this = torch.arange(1, num_windows + 1) * (
                window_h * window_w) + cu_seqlens[-1]
            cu_seqlens.extend(cu_this.tolist())
            token_offset += height * width

        window_index = torch.cat(window_index_list)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        return window_index, cu_seqlens, max_seqlens

    def forward(self, hidden_states: torch.Tensor, target_sizes: torch.Tensor,
                cu_seqlens: List[int]) -> torch.Tensor:
        # hidden_states: [T, D]
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        device = hidden_states.device

        window_index, _, window_len = self.get_window_index(target_sizes)
        window_index = window_index.to(device)

        # Every window is exactly `window_len` (= window_h*window_w) tokens, so
        # the window self-attention is a regular batched attention rather than a
        # variable-length one (see MiniCPMV4_6VisionAttention.forward_window).
        hidden_states = hidden_states[window_index, :]
        hidden_states = self.self_attn.forward_window(hidden_states, window_len)
        hidden_states = hidden_states[torch.argsort(window_index), :]
        hidden_states = residual + hidden_states

        window_h, window_w = self.window_kernel_size
        embed_dim = hidden_states.shape[-1]
        merged = []
        for i in range(len(target_sizes)):
            height, width = int(target_sizes[i][0]), int(target_sizes[i][1])
            patch = hidden_states[cu_seqlens[i]:cu_seqlens[i + 1], :]
            merged_h, merged_w = height // window_h, width // window_w
            patch_5d = patch.view(merged_h, window_h, merged_w, window_w,
                                  embed_dim).permute(0, 2, 1, 3, 4)
            hidden_state = patch_5d.reshape(merged_h * merged_w,
                                            window_h * window_w * embed_dim)
            mean_residual = patch_5d.reshape(merged_h * merged_w,
                                             window_h * window_w,
                                             embed_dim).mean(dim=1)
            hidden_state = self.pre_norm(hidden_state)
            hidden_state = self.linear_1(hidden_state)
            hidden_state = _gelu_tanh(hidden_state)
            hidden_state = self.linear_2(hidden_state)
            merged.append(hidden_state + mean_residual)

        return torch.concat(merged, dim=0)


class MiniCPMV4_6DownsampleMLP(nn.Module):
    """2x2 spatial-merge projection (``pre_norm`` -> ``linear_1`` -> GELU -> ``linear_2``)."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 hidden_size: int, out_dim: int):
        super().__init__()
        dtype = model_config.pretrained_config.vision_config.torch_dtype
        mapping = model_config.mapping
        merged_hidden_size = hidden_size * 4
        self.pre_norm = LayerNorm(hidden_size=merged_hidden_size,
                                  eps=1e-6,
                                  dtype=dtype)
        self.linear_1 = Linear(merged_hidden_size,
                               merged_hidden_size,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=None)
        self.linear_2 = Linear(merged_hidden_size,
                               out_dim,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states).view(
            -1, self.linear_1.in_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class MiniCPMV4_6Merger(nn.Module):
    """Final per-image/frame 2x2 merge into the LLM embedding space."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        self.merger_times = config.merger_times
        hidden_size = config.vision_config.hidden_size
        llm_embed_dim = config.text_config.hidden_size
        mlps = [
            MiniCPMV4_6DownsampleMLP(model_config, hidden_size, hidden_size)
            for _ in range(self.merger_times - 1)
        ]
        mlps.append(
            MiniCPMV4_6DownsampleMLP(model_config, hidden_size, llm_embed_dim))
        self.mlp = nn.ModuleList(mlps)

    def forward(self, hidden_states: torch.Tensor,
                target_sizes: torch.Tensor) -> List[torch.Tensor]:
        merge_h, merge_w = self.merge_kernel_size
        start = 0
        processed_features = []
        for i in range(len(target_sizes)):
            height, width = int(target_sizes[i][0]), int(target_sizes[i][1])
            num_patches = height * width
            embed_dim = hidden_states.shape[-1]
            merged_h, merged_w = height // merge_h, width // merge_w
            hidden_state = (hidden_states[start:start + num_patches, :].view(
                merged_h, merge_h, merged_w, merge_w,
                embed_dim).permute(0, 2, 1, 3,
                                   4).reshape(merged_h * merged_w,
                                              merge_h * merge_w * embed_dim))
            hidden_state = self.mlp[0](hidden_state)
            for j in range(1, self.merger_times):
                height = height // merge_h
                width = width // merge_w
                inner_dim = hidden_state.shape[-1]
                merged_h, merged_w = height // merge_h, width // merge_w
                hidden_state = (hidden_state.view(
                    merged_h, merge_h, merged_w, merge_w,
                    inner_dim).permute(0, 2, 1, 3,
                                       4).reshape(merged_h * merged_w,
                                                  merge_h * merge_w * inner_dim))
                hidden_state = self.mlp[j](hidden_state)
            start += num_patches
            processed_features.append(hidden_state)
        return processed_features


class MiniCPMV4_6VisionModel(nn.Module):
    """Full vision tower: embeddings -> encoder (+window merger) -> merger.

    Consumes a batch of :class:`MultimodalParams`, concatenates every request's
    NaViT-packed ``pixel_values`` / ``target_sizes`` into one packed sequence
    (Contract 4), and returns a single ``[total_mm_tokens, llm_hidden]`` tensor.
    Runs replicated on every rank (tp=1); the vision encoder is excluded from
    quantization.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.vision_config = config.vision_config
        # Authoritative model dtype (set by ModelConfig.from_pretrained on the
        # composite; the raw config.json omits torch_dtype, so fall back to
        # bf16 which matches the released checkpoint). Mirror it onto the
        # vision sub-config so ported modules pick up a concrete dtype.
        self.dtype = config.torch_dtype or torch.bfloat16
        self.vision_config.torch_dtype = self.dtype
        self.downsample_mode = config.downsample_mode
        self.insert_layer_id = self.vision_config.insert_layer_id

        # Replicated, unquantized ModelConfig for the vision sub-modules.
        # dataclasses.replace gives a fresh extra_attrs (init=False), so the
        # vision attention's transient layer registration can't collide with
        # the LLM's.
        vision_model_config = dataclasses.replace(model_config,
                                                  quant_config=QuantConfig(),
                                                  mapping=Mapping())
        self.model_config = vision_model_config
        self.config = self.vision_config

        self.embeddings = MiniCPMV4_6VisionEmbeddings(self.vision_config,
                                                      self.dtype)
        self.encoder = MiniCPMV4_6VisionEncoder(vision_model_config)
        self.post_layernorm = LayerNorm(hidden_size=self.vision_config.hidden_size,
                                        eps=self.vision_config.layer_norm_eps,
                                        dtype=self.dtype)
        self.vit_merger = MiniCPMV4_6ViTWindowAttentionMerger(
            vision_model_config)
        self.merger = MiniCPMV4_6Merger(vision_model_config)

        self.metadata_cls = get_attention_backend(
            model_config.attn_backend).Metadata

    def _make_attn_metadata(self, seq_lens: List[int]) -> AttentionMetadata:
        num_segments = len(seq_lens)
        attn_metadata = self.metadata_cls(
            max_num_requests=num_segments + 1,
            max_num_tokens=sum(seq_lens) + 1,
            kv_cache_manager=None,
        )
        return _prepare_qwen_vl_vision_attn_metadata(seq_lens, attn_metadata)

    @staticmethod
    def _grid_seq_lens(target_sizes: torch.Tensor) -> List[int]:
        return (target_sizes[:, 0] * target_sizes[:, 1]).tolist()

    @staticmethod
    def _grid_cu_seqlens(target_sizes: torch.Tensor) -> List[int]:
        cu = [0]
        for i in range(len(target_sizes)):
            cu.append(cu[-1] +
                      int(target_sizes[i][0]) * int(target_sizes[i][1]))
        return cu

    def _get_image_features(self, pixel_values: torch.Tensor,
                            target_sizes: torch.Tensor) -> List[torch.Tensor]:
        pixel_values = pixel_values.to(self.dtype)
        hidden_states = self.embeddings(pixel_values, target_sizes).squeeze(0)

        use_vit_merger = self.downsample_mode != "4x"
        if use_vit_merger and self.insert_layer_id >= 0:
            attn_metadata = self._make_attn_metadata(
                self._grid_seq_lens(target_sizes))
            for layer_index, encoder_layer in enumerate(self.encoder.layers):
                hidden_states = encoder_layer(hidden_states, attn_metadata)
                if layer_index == self.insert_layer_id:
                    cu_seqlens = self._grid_cu_seqlens(target_sizes)
                    hidden_states = self.vit_merger(hidden_states, target_sizes,
                                                    cu_seqlens)
                    target_sizes = target_sizes // 2
                    attn_metadata = self._make_attn_metadata(
                        self._grid_seq_lens(target_sizes))
        else:
            attn_metadata = self._make_attn_metadata(
                self._grid_seq_lens(target_sizes))
            for encoder_layer in self.encoder.layers:
                hidden_states = encoder_layer(hidden_states, attn_metadata)

        hidden_states = self.post_layernorm(hidden_states)
        return self.merger(hidden_states, target_sizes)

    @torch.inference_mode()
    @nvtx_range("MiniCPMV4_6 vision encoder")
    def forward(self,
                multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        pixel_values_list = []
        target_sizes_list = []
        for param in multimodal_params:
            # Image and video share the NaViT-packed vision path; a request may
            # carry either (or both). Concatenate in image-then-video order to
            # match the placeholder order the input processor emits.
            for modality in ("image", "video"):
                modality_data = param.multimodal_data.get(modality)
                if modality_data is None:
                    continue
                pixel_values_list.append(modality_data["pixel_values"])
                target_sizes_list.append(modality_data["target_sizes"])

        if not pixel_values_list:
            return []

        # NaViT packing: concat along the packed patch axis; target grids stack.
        pixel_values = (torch.cat(pixel_values_list, dim=-1)
                        if len(pixel_values_list) > 1 else pixel_values_list[0])
        target_sizes = (torch.cat(target_sizes_list, dim=0)
                        if len(target_sizes_list) > 1 else target_sizes_list[0])
        target_sizes = target_sizes.to("cpu")

        features = self._get_image_features(pixel_values, target_sizes)
        return [torch.cat(features, dim=0)]

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        converted_weights = {}
        for key, value in weights.items():
            if key.startswith("model.vision_tower."):
                converted_weights[key[len("model.vision_tower."):]] = value
            elif key.startswith("model.merger."):
                converted_weights["merger." +
                                  key[len("model.merger."):]] = value

        # q/k/v -> qkv_proj fusion is handled by _load_weights_impl's params_map;
        # only the projection / MLP renames need explicit regex substitutions.
        pattern_mapping = {
            r"(.*?)self_attn\.out_proj\.(.*)": r"\1self_attn.o_proj.\2",
            r"(.*?)mlp\.fc1\.(.*)": r"\1mlp.up_proj.\2",
            r"(.*?)mlp\.fc2\.(.*)": r"\1mlp.down_proj.\2",
        }
        _load_weights_impl(self, converted_weights, params_map=pattern_mapping)


# ---------------------------------------------------------------------------
# Input processor
# ---------------------------------------------------------------------------
class MiniCPMV4_6InputProcessor(BaseMultimodalInputProcessor,
                                BaseMultimodalDummyInputsBuilder):
    """Runs the HF MiniCPM-V processor and rewrites image tokens to the OOV
    sentinel expected by ``fuse_input_embeds``."""

    def __init__(self,
                 model_path: str,
                 config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True,
                 **kwargs):
        _ensure_transformers_supports_minicpmv4_6()
        super().__init__(model_path=model_path,
                         config=config,
                         tokenizer=tokenizer,
                         trust_remote_code=trust_remote_code,
                         **kwargs)
        self._config = config
        self._model_path = model_path
        self._tokenizer = tokenizer if tokenizer is not None else \
            AutoTokenizer.from_pretrained(model_path,
                                          trust_remote_code=trust_remote_code)
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=self.use_fast,
            trust_remote_code=trust_remote_code)
        self._dtype = (config.torch_dtype or config.text_config.torch_dtype
                       or torch.bfloat16)
        self.tllm_multimodal_token_id = self.get_vocab_size() + 1
        self.image_token_str = getattr(self._processor, "image_token",
                                       "<|image_pad|>")
        self.video_token_str = getattr(self._processor, "video_token",
                                       "<|video_pad|>")

    def get_vocab_size(self) -> int:
        return self.config.text_config.vocab_size

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _mm_token_ids(self) -> List[int]:
        return [
            tid for attr in ("image_token_id", "video_token_id")
            if (tid := getattr(self.config, attr, None)) is not None
        ]

    def _postprocess(self, input_ids: torch.IntTensor) -> torch.IntTensor:
        token_ids = self._mm_token_ids()
        if token_ids:
            ids_tensor = torch.tensor(token_ids,
                                      device=input_ids.device,
                                      dtype=input_ids.dtype)
            input_ids[torch.isin(input_ids,
                                 ids_tensor)] = self.tllm_multimodal_token_id
        return input_ids

    def _preprocess(self, text_prompt: str, images, videos, video_metadata,
                    mm_processor_kwargs: Dict):
        do_rescale = True
        if images and isinstance(images[0], torch.Tensor):
            do_rescale = False
        # load_video hands us CHW float frames already scaled to [0, 1], so
        # the HF rescale (x / 255) must be skipped for video.
        if videos and isinstance(videos[0][0], torch.Tensor):
            do_rescale = False
        call_kwargs = dict(text=[text_prompt],
                           images=images,
                           do_rescale=do_rescale,
                           return_tensors="pt",
                           **mm_processor_kwargs)
        if videos is not None:
            # Frames are already sampled by TRT-LLM's load_video, so disable the
            # HF processor's own frame sampling (which would otherwise require
            # full metadata and re-index against the original frame count).
            call_kwargs["videos"] = videos
            call_kwargs["video_metadata"] = video_metadata
            call_kwargs["do_sample_frames"] = False
        return self.processor(**call_kwargs)

    def get_num_tokens_per_image(self, *, image, **kwargs) -> int:
        do_rescale = not isinstance(image, torch.Tensor)
        processed = self.processor(text=[self.image_token_str],
                                   images=[image],
                                   do_rescale=do_rescale,
                                   return_tensors="pt")
        input_ids = processed["input_ids"][0]
        return int((input_ids == self.config.image_token_id).sum().item())

    def get_num_tokens_per_video(self,
                                 *,
                                 video,
                                 video_metadata=None,
                                 **kwargs) -> int:
        # `video` is the list of pre-sampled frames (see find_mm_token_lengths).
        frames = video
        do_rescale = not (frames and isinstance(frames[0], torch.Tensor))
        processed = self.processor(
            text=[self.video_token_str],
            videos=[frames],
            video_metadata=[video_metadata]
            if video_metadata is not None else None,
            do_sample_frames=False,
            do_rescale=do_rescale,
            return_tensors="pt")
        input_ids = processed["input_ids"][0]
        return int((input_ids == self.config.video_token_id).sum().item())

    @nvtx_range("MiniCPMV4_6InputProcessor call_with_text_prompt")
    @torch.inference_mode()
    def call_with_text_prompt(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data", {})
        mm_processor_kwargs = inputs.get("mm_processor_kwargs", {}) or {}

        if not mm_data:
            input_ids = self.tokenizer(text_prompt,
                                       return_tensors="pt").input_ids
            return input_ids[0].to(torch.int32).tolist(), None

        images = mm_data.get("image")
        video_datas = mm_data.get("video")
        videos = video_metadata = None
        if video_datas is not None:
            # Each item is a VideoData (frames + decode metadata) from
            # load_video, mirroring the Qwen-VL input processor.
            videos = [vd.frames for vd in video_datas]
            video_metadata = [
                getattr(vd, "metadata", None) for vd in video_datas
            ]
        processed_inputs = self._preprocess(text_prompt, images, videos,
                                            video_metadata, mm_processor_kwargs)

        multimodal_data = {}
        pixel_values = processed_inputs.get("pixel_values")
        if pixel_values is not None:
            multimodal_data["image"] = {
                # target_sizes stays on CPU: the vision encoder uses it for
                # integer window/grid arithmetic (not listed in
                # multimodal_data_device_paths).
                "pixel_values": pixel_values.to(self.dtype),
                "target_sizes": processed_inputs.get("target_sizes"),
            }
        # Video is packed identically to image (pixel_values_videos is a NaViT
        # [1, C, patch, seq] tensor); store it under the "video" key so the
        # vision tower runs the shared encoder path on it.
        pixel_values_videos = processed_inputs.get("pixel_values_videos")
        if pixel_values_videos is not None:
            multimodal_data["video"] = {
                "pixel_values": pixel_values_videos.to(self.dtype),
                "target_sizes": processed_inputs.get("target_sizes_videos"),
            }

        fused_input_ids = self._postprocess(processed_inputs["input_ids"][0])
        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


# ---------------------------------------------------------------------------
# Top-level VLM wrapper
# ---------------------------------------------------------------------------
@register_auto_model("MiniCPMV4_6ForConditionalGeneration")
@register_input_processor(
    MiniCPMV4_6InputProcessor,
    model_type="minicpmv4_6",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|image_pad|>",
            "video": "<|video_pad|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        placeholders_separator="\n",
        content_format=ContentFormat.STRING,
    ),
)
class MiniCPMV4_6Model(PreTrainedModel):

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs):
        config = model_config.pretrained_config
        super().__init__(config)
        if hasattr(self, "llm"):
            return

        if _is_mm_disagg():
            raise NotImplementedError(
                "MiniCPMV4_6 does not support disaggregated multimodal serving "
                "yet. Unset TLLM_MULTIMODAL_DISAGGREGATED or set it to '0'.")

        self.model_config = model_config
        self.mm_encoder = MiniCPMV4_6VisionModel(model_config).eval()

        # Inner LLM resolved from text_config (Qwen3_5ForCausalLM, hybrid).
        llm_model_config = dataclasses.replace(
            model_config, pretrained_config=config.text_config)
        # Share the wrapper's extra_attrs so the LM's attention layers register
        # into the same dict the engine binds via with_model_extra_attrs (needed
        # for the compiled / piecewise-CUDA-graph LM path). dataclasses.replace
        # would otherwise give a fresh (init=False) extra_attrs.
        llm_model_config.extra_attrs = model_config.extra_attrs
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.image_token_id = config.image_token_id
        self.post_config()

    def post_config(self):
        # Downstream (KV cache manager, engine) expects the LLM-shaped config.
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        num_context_requests = attn_metadata.num_contexts
        multimodal_params = kwargs.get("multimodal_params", [])

        mm_embeds = []
        if len(multimodal_params) > 0 and not _is_mm_disagg():
            mm_embeds = get_multimodal_embeddings(
                encoder_forward_fn=self.mm_encoder.forward,
                multimodal_params=multimodal_params[:num_context_requests],
            )
            mm_embeds = find_input_mm_embeds(
                mm_embeds, multimodal_params[:num_context_requests])

        input_ids, inputs_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens, input_ids, mm_embeds, **kwargs)
        return self.llm.forward(attn_metadata=attn_metadata,
                                input_ids=input_ids,
                                position_ids=position_ids,
                                inputs_embeds=inputs_embeds,
                                return_context_logits=return_context_logits)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values", "video.pixel_values", "multimodal_embedding"
        ]

    def load_weights(self, weights: Dict[str, torch.Tensor],
                     weight_mapper: BaseWeightMapper):
        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights)
            if hasattr(weights, "mark_consumed"):
                weights.mark_consumed("model.vision_tower")
                weights.mark_consumed("model.merger")

        llm_weight_mapper = Qwen3_5MoeHfWeightMapper()
        llm_weight_mapper.init_model_and_config(self.llm, self.model_config)
        llm_weights = {
            k: v
            for k, v in weights.items()
            if k.startswith("model.language_model.")
        }
        self.llm.load_weights(llm_weights, llm_weight_mapper)
        if hasattr(weights, "mark_consumed"):
            weights.mark_consumed("model.language_model")
