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
"""Kimi K2.5 multimodal model for TensorRT-LLM PyTorch backend.

Implements the full K2.5 vision-language model:
- MoonViT3d vision encoder (natively implemented)
- PatchMergerMLP projector (natively implemented)
- KimiK25InputProcessor for image/video preprocessing
- KimiK25ForConditionalGeneration that wires vision encoding with DeepSeek-V3

MoonViT3d Architecture:
    Patch embedding (Conv2d 14x14) + 2D learnable position embedding
    -> 27-layer encoder (2D RoPE, TRT-LLM spatial-temporal attention)
    -> tpool_patch_merger (temporal avg pool + 2x2 spatial merge)
    -> PatchMergerMLP (LayerNorm -> Linear -> GELU -> Linear)
    -> [N_tokens, text_hidden_size=7168]
"""

import copy
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.mapping import Mapping

from ..._utils import prefer_pinned
from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
)
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PredefinedAttentionMask
from ..attention_backend.utils import get_attention_backend
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mlp import MLP
from .checkpoints.base_weight_loader import ConsumableWeightsDict
from .modeling_deepseekv3 import DeepseekV3ForCausalLM
from .modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from .modeling_utils import (
    MetaInitException,
    QuantConfig,
    _load_weights_impl,
    filter_weights,
    register_auto_model,
    register_vision_encoder,
)

logger = logging.getLogger(__name__)

DISAGG = os.getenv("TLLM_MULTIMODAL_DISAGGREGATED", "0") == "1"


def _has_meta_tensors(module: nn.Module) -> bool:
    return any(getattr(param, "is_meta", False) for param in module.parameters()) or any(
        getattr(buffer, "is_meta", False) for buffer in module.buffers()
    )


def _format_video_timestamp(timestamp: float, timestamp_mode: str = "hh:mm:ss.fff") -> str:
    """Format video timestamps to match the HF Kimi K2.5 processor."""
    if timestamp_mode == "hh:mm:ss.fff":
        return (
            datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%H:%M:%S")
            + f".{int((timestamp % 1) * 1000):03d}"
        )
    if timestamp_mode == "mm:ss.fff":
        return (
            datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%M:%S")
            + f".{int((timestamp % 1) * 1000):03d}"
        )
    if timestamp_mode == "mm:ss":
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%M:%S")
    raise ValueError(f"Invalid timestamp mode: {timestamp_mode}")


def _decode_video_to_chunks(
    video_src: Union[str, bytes],
    temporal_merge_kernel_size: int = 4,
    sample_fps: float = 2.0,
    timestamp_mode: str = "hh:mm:ss.fff",
) -> Tuple[List[Dict[str, Any]], str]:
    """Decode a video file into temporal chunks of PIL frames.

    The HF KimiK25Processor uses ``mecord.VideoReader`` for video decoding.
    This function provides equivalent functionality using ``decord`` or
    ``cv2`` to avoid the ``mecord`` dependency.

    The output format matches what ``KimiK25Processor.preprocess_medias()``
    produces: a list of ``video_chunk`` dicts with PIL frames, plus a
    combined prompt string with timestamps.

    Args:
        video_src: Path to a video file, or raw bytes.
        temporal_merge_kernel_size: Number of frames per temporal chunk.
        sample_fps: Target sampling frame rate.
        timestamp_mode: Format for timestamp strings.

    Returns:
        Tuple of (chunks, video_prompt) where chunks is a list of dicts
        ``{"type": "video_chunk", "video_chunk": [PIL.Image, ...],
        "prompt": "<timestamp><|media_begin|>video<|media_content|><|media_pad|><|media_end|>"}``
        and video_prompt is the concatenated prompt for all chunks.
    """
    # --- Read video metadata and frames ---
    frames_pil: List[Image.Image] = []
    sampled_frame_inds: List[int] = []
    avg_fps: float = 30.0
    total_frames: int = 0

    # Try decord first, fall back to cv2.
    try:
        from decord import VideoReader as _VR

        vr = _VR(video_src, num_threads=1)
        total_frames = len(vr)
        avg_fps = float(vr.get_avg_fps())

        effective_fps = min(sample_fps, avg_fps)
        sampled_n = max(round(total_frames * effective_fps / avg_fps), 1)

        frame_inds = np.linspace(0, total_frames - 1, sampled_n).round().astype(int).tolist()
        raw_frames = vr.get_batch(frame_inds).asnumpy()
        frames_pil = [Image.fromarray(f) for f in raw_frames]
        sampled_frame_inds = frame_inds
    except Exception:
        import cv2

        _tmp_video_path = None
        if isinstance(video_src, bytes):
            fd, tmp = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            try:
                with open(tmp, "wb") as f:
                    f.write(video_src)
            except OSError:
                os.unlink(tmp)
                raise
            video_src = tmp
            _tmp_video_path = tmp

        cap = cv2.VideoCapture(video_src)
        avg_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        effective_fps = min(sample_fps, avg_fps)
        sampled_n = max(round(total_frames * effective_fps / avg_fps), 1)

        frame_inds = np.linspace(0, total_frames - 1, sampled_n).round().astype(int).tolist()

        target_frame_inds = set(frame_inds)
        max_frame_ind = max(frame_inds)
        all_frames: Dict[int, Any] = {}
        frame_ind = 0
        while frame_ind <= max_frame_ind:
            grabbed = cap.grab()
            if not grabbed:
                break
            if frame_ind in target_frame_inds:
                ret, frame = cap.retrieve()
                if ret:
                    all_frames[frame_ind] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ind += 1
        cap.release()

        # Clean up temp file created from bytes input
        if _tmp_video_path is not None:
            try:
                os.unlink(_tmp_video_path)
            except OSError:
                pass

        sampled_frame_inds = [i for i in frame_inds if i in all_frames]
        frames_pil = [Image.fromarray(all_frames[i]) for i in sampled_frame_inds]

    if not frames_pil:
        raise ValueError(f"Failed to decode any frames from video: {video_src}")

    # --- Split into temporal chunks (matching HF split_video_chunks) ---
    chunk_prompt_fmt = "{timestamp}<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
    chunks: List[Dict[str, Any]] = []
    prompts: List[str] = []

    for i in range(0, len(frames_pil), temporal_merge_kernel_size):
        chunk_frames = frames_pil[i : i + temporal_merge_kernel_size]
        # Compute timestamp from original frame index
        if i < len(sampled_frame_inds):
            start_time = sampled_frame_inds[i] / avg_fps
        else:
            start_time = 0.0
        ts_text = _format_video_timestamp(start_time, timestamp_mode)
        prompt = chunk_prompt_fmt.format(timestamp=ts_text)
        chunks.append(
            {
                "type": "video_chunk",
                "video_chunk": chunk_frames,
                "prompt": prompt,
            }
        )
        prompts.append(prompt)

    return chunks, "".join(prompts)


def _frames_to_chunks(
    frames: List,
    temporal_merge_kernel_size: int = 4,
    timestamp_mode: str = "hh:mm:ss.fff",
    fps: float = 2.0,
    frame_indices: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Split pre-decoded PIL frames into temporal chunks.

    Same output format as ``_decode_video_to_chunks`` but skips the
    decoding step (frames are already PIL Images).
    """
    if fps <= 0:
        raise ValueError(f"Video fps must be positive, got {fps}")
    if frame_indices is not None and len(frame_indices) != len(frames):
        raise ValueError(
            f"Video frame_indices length ({len(frame_indices)}) must match "
            f"frames length ({len(frames)})"
        )

    chunk_prompt_fmt = "{timestamp}<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
    chunks: List[Dict[str, Any]] = []
    prompts: List[str] = []

    for i in range(0, len(frames), temporal_merge_kernel_size):
        chunk_frames = frames[i : i + temporal_merge_kernel_size]
        start_frame = frame_indices[i] if frame_indices is not None else i
        start_time = start_frame / fps
        ts_text = _format_video_timestamp(start_time, timestamp_mode)
        prompt = chunk_prompt_fmt.format(timestamp=ts_text)
        chunks.append(
            {
                "type": "video_chunk",
                "video_chunk": chunk_frames,
                "prompt": prompt,
            }
        )
        prompts.append(prompt)

    return chunks, "".join(prompts)


# Default media placeholder token ID for K2.5
_MEDIA_PLACEHOLDER_TOKEN_ID = 163605

# Default vocabulary size for K2.5
_VOCAB_SIZE = 163840


# ---------------------------------------------------------------------------
# Native MoonViT3d Vision Encoder Components
# ---------------------------------------------------------------------------
# Ported from the HuggingFace Kimi-K2.5 model repository to avoid
# runtime dependency on dynamic custom code loading via trust_remote_code.


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """1D sinusoidal positional embedding from a grid of positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_1d_sincos_pos_embed(embed_dim: int, t_size: int) -> np.ndarray:
    """1D sinusoidal positional embedding for temporal dimension."""
    grid_t = np.arange(t_size, dtype=np.float32)
    return _get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)


def _get_rope_shape_impl(org: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Interpolate 2D position embeddings to target spatial shape."""
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode="bicubic",
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def _apply_rope_2d(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D rotary position embedding to query and key tensors.

    Args:
        xq: query ``(..., num_heads, head_dim)``
        xk: key ``(..., num_heads, head_dim)``
        freqs_cis: complex frequencies ``(..., head_dim/2)``
    """
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _get_vision_tp_mapping(model_config: ModelConfig) -> Mapping:
    if not model_config.mapping.enable_attention_dp:
        return model_config.mapping

    return Mapping(
        world_size=model_config.mapping.pp_size * model_config.mapping.tp_size,
        rank=model_config.mapping.rank,
        gpus_per_node=model_config.mapping.gpus_per_node,
        tp_size=1,
        pp_size=model_config.mapping.pp_size * model_config.mapping.tp_size,
    )


def _tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: Tuple[int, int] = (2, 2),
) -> List[torch.Tensor]:
    """Temporal average pooling + 2x2 spatial merge.

    Args:
        x: Flat vision features ``[total_tokens, hidden_dim]``.
        grid_thws: ``[N_media, 3]`` with ``(T, H, W)`` per media.
        merge_kernel_size: Spatial merge kernel ``(kh, kw)``.

    Returns:
        List of per-media tensors ``[N_merged, kh*kw, hidden_dim]``.
    """
    d_model = x.size(-1)
    kernel_height, kernel_width = merge_kernel_size
    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped = seq.view(t, new_height, kernel_height, new_width, kernel_width, d_model)
        # permute to (T, new_h, new_w, kh, kw, C) then temporal mean
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        # reshape to (new_h * new_w, kh * kw, C)
        outputs.append(reshaped.view(new_height * new_width, kernel_height * kernel_width, -1))
        pre_sum += t * h * w
    return outputs


class Learnable2DPosEmb(nn.Module):
    """Learnable 2D position embedding with bicubic interpolation + temporal sincos."""

    def __init__(self, height: int, width: int, num_frames: int, dim: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(_get_1d_sincos_pos_embed(dim, num_frames)).float().unsqueeze(1),
            persistent=False,
        )
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            if (h, w) == (self.height, self.width):
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = _get_rope_shape_impl(self.weight, (h, w))
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[:t]
            pos_embs.append(pos_emb_3d.reshape(-1, self.dim))
        return x + torch.cat(pos_embs)


class PatchEmbed3d(nn.Module):
    """Conv2d patch embedding + learnable 2D position embedding."""

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int,
        pos_emb_height: int,
        pos_emb_width: int,
        pos_emb_time: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = Learnable2DPosEmb(pos_emb_height, pos_emb_width, pos_emb_time, hidden_dim)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        return self.pos_emb(x, grid_thws)


class Rope2D(nn.Module):
    """2D rotary position embedding with height/width interleaved frequencies."""

    def __init__(
        self, dim: int, max_height: int = 512, max_width: int = 512, theta_base: float = 10000.0
    ) -> None:
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute(self, device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(N, device=device).float()
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4, device=device)[: self.dim // 4].float()
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_cis = torch.polar(torch.ones_like(torch.outer(x_pos, freqs)), torch.outer(x_pos, freqs))
        y_cis = torch.polar(torch.ones_like(torch.outer(y_pos, freqs)), torch.outer(y_pos, freqs))
        freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
        return freqs_cis.reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(self, grid_thws: torch.Tensor, device: torch.device) -> torch.Tensor:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer("freqs_cis", self._precompute(device), persistent=False)
        return torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in grid_thws.tolist()
            ],
            dim=0,
        )


class VisionMLP(MLP):
    """TRT-LLM MLP wrapper for each MoonViT3d encoder layer."""

    def __init__(
        self, model_config: ModelConfig, layer_idx: int, hidden_dim: int, mlp_dim: int
    ) -> None:
        super().__init__(
            hidden_size=hidden_dim,
            intermediate_size=mlp_dim,
            bias=True,
            activation=_gelu_tanh,
            dtype=model_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            overridden_tp_size=1 if model_config.mapping.enable_attention_dp else None,
        )


class KimiK25VisionAttention(Attention):
    """TRT-LLM attention wrapper that applies Kimi K2.5's 2D RoPE."""

    def __init__(
        self,
        model_config: ModelConfig,
        hidden_dim: int,
        num_heads: int,
        layer_idx: int,
        attn_bias: bool = True,
    ) -> None:
        super().__init__(
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=None,
            bias=attn_bias,
            pos_embd_params=None,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=model_config.torch_dtype,
            dense_bias=attn_bias,
            config=model_config,
            reduce_output=(
                not model_config.mapping.enable_attention_dp and model_config.mapping.tp_size > 1
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv)
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)

        q, k = _apply_rope_2d(q, k, freqs_cis)
        q, k, v = q.reshape(seq_len, -1), k.reshape(seq_len, -1), v.reshape(seq_len, -1)
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


class EncoderLayer(nn.Module):
    """Single MoonViT3d encoder layer: LayerNorm + QKV attention + 2D RoPE + MLP."""

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        attn_bias: bool = True,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm0 = LayerNorm(
            hidden_size=hidden_dim,
            eps=ln_eps,
            dtype=model_config.torch_dtype,
        )
        self.norm1 = LayerNorm(
            hidden_size=hidden_dim,
            eps=ln_eps,
            dtype=model_config.torch_dtype,
        )
        self.attn = KimiK25VisionAttention(
            model_config,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layer_idx=layer_idx,
            attn_bias=attn_bias,
        )
        self.mlp = VisionMLP(model_config, layer_idx, hidden_dim, mlp_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention with 2D RoPE
        residual = x
        x = self.norm0(x)
        x = self.attn(x, attn_metadata, freqs_cis)
        x = residual + x
        # MLP
        residual = x
        x = self.norm1(x)
        x = residual + self.mlp(x)
        return x


class MoonViT3dEncoder(nn.Module):
    """Stack of MoonViT3d encoder layers + 2D RoPE + final LayerNorm."""

    def __init__(
        self,
        model_config: ModelConfig,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        attn_bias: bool = True,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.rope_2d = Rope2D(hidden_dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                EncoderLayer(
                    model_config,
                    layer_idx=layer_idx,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    attn_bias=attn_bias,
                    ln_eps=ln_eps,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_layernorm = LayerNorm(
            hidden_size=hidden_dim,
            eps=ln_eps,
            dtype=model_config.torch_dtype,
        )

        self.metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        self.attn_metadata = self.metadata_cls(
            max_num_requests=8192,
            max_num_tokens=8192,
            kv_cache_manager=None,
        )

    def prepare_attn_metadata(
        self,
        seq_lens: List[int],
        attn_metadata: AttentionMetadata,
    ) -> AttentionMetadata:
        batch_size = len(seq_lens)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, pin_memory=prefer_pinned())
        attn_metadata.num_contexts = batch_size
        attn_metadata.request_ids = list(range(1, batch_size + 1))
        attn_metadata.prompt_lens = seq_lens
        attn_metadata.seq_lens = seq_lens_tensor
        attn_metadata.max_seq_len = seq_lens_tensor.max().item()
        attn_metadata.prepare()
        return attn_metadata

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        freqs_cis = self.rope_2d.get_freqs_cis(grid_thws, x.device)
        seq_lens = (grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]).tolist()
        attn_metadata = self.prepare_attn_metadata(seq_lens, self.attn_metadata)
        for block in self.blocks:
            x = block(x, attn_metadata, freqs_cis)
        return self.final_layernorm(x)


class PatchMergerMLP(nn.Module):
    """PatchMergerMLP projector: LayerNorm + spatial-merge reshape + MLP.

    Architecture:
        LayerNorm(mm_hidden_size) -> view(N, kh*kw*C) ->
        Linear(merged_dim, merged_dim) -> GELU ->
        Linear(merged_dim, text_hidden_size)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        mm_hidden_size: int,
        text_hidden_size: int,
        merge_kernel_size: Tuple[int, int] = (2, 2),
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        kh, kw = merge_kernel_size
        self.merged_dim = mm_hidden_size * kh * kw
        self.pre_norm = LayerNorm(
            hidden_size=mm_hidden_size,
            eps=ln_eps,
            dtype=model_config.torch_dtype,
        )
        mapping = _get_vision_tp_mapping(model_config)
        self.proj = nn.Sequential(
            Linear(
                self.merged_dim,
                self.merged_dim,
                bias=True,
                dtype=model_config.torch_dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                allreduce_strategy=model_config.allreduce_strategy,
            ),
            nn.GELU(),
            Linear(
                self.merged_dim,
                text_hidden_size,
                bias=True,
                dtype=model_config.torch_dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.ROW,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                allreduce_strategy=model_config.allreduce_strategy,
            ),
        )

    def forward(
        self,
        x: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if isinstance(x, (list, tuple)):
            return [self.proj(self.pre_norm(item).view(item.shape[0], -1)) for item in x]
        B = x.shape[0]
        return self.proj(self.pre_norm(x).view(B, -1, self.merged_dim))


# ---------------------------------------------------------------------------
# MoonViT3d Vision Encoder (top-level wrapper)
# ---------------------------------------------------------------------------


class KimiK25VisionModel(nn.Module):
    """Native MoonViT3d vision encoder + PatchMergerMLP projector for K2.5.

    Pipeline:
        pixel_values -> PatchEmbed3d (Conv2d + 2D pos emb)
        -> MoonViT3dEncoder (27 layers, 2D RoPE, TRT-LLM attention)
        -> tpool_patch_merger (temporal avg pool + 2x2 spatial merge)
        -> PatchMergerMLP (LayerNorm + Linear + GELU + Linear)
        -> [total_tokens, text_hidden_size]

    Args:
        model_config: TRT-LLM ModelConfig wrapping HF PretrainedConfig.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__()
        self.model_config = copy.copy(model_config)
        self.model_config.extra_attrs = copy.copy(model_config.extra_attrs)
        self.model_config._frozen = False
        self.model_config.quant_config = QuantConfig(
            kv_cache_quant_algo=model_config.quant_config.kv_cache_quant_algo
        )
        self.model_config.pretrained_config = copy.copy(model_config.pretrained_config)
        pretrained_config = self.model_config.pretrained_config
        model_dtype = (
            getattr(pretrained_config, "torch_dtype", None)
            or getattr(pretrained_config, "dtype", None)
            or torch.bfloat16
        )
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype, torch.bfloat16)
        pretrained_config.torch_dtype = model_dtype

        # Extract vision config dict
        vision_cfg = getattr(pretrained_config, "vision_config", {})
        if vision_cfg is None:
            vision_cfg = {}
        if not isinstance(vision_cfg, dict):
            vision_cfg = (
                vision_cfg.to_dict() if hasattr(vision_cfg, "to_dict") else vars(vision_cfg)
            )

        # Read HF-prefixed names with unprefixed fallback
        hidden_dim = vision_cfg.get("vt_hidden_size", vision_cfg.get("hidden_size", 1152))
        num_layers = vision_cfg.get("vt_num_hidden_layers", vision_cfg.get("num_hidden_layers", 27))
        num_heads = vision_cfg.get(
            "vt_num_attention_heads", vision_cfg.get("num_attention_heads", 16)
        )
        self.model_config.pretrained_config.head_dim = hidden_dim // num_heads
        self.model_config._frozen = True
        mlp_dim = vision_cfg.get("vt_intermediate_size", vision_cfg.get("intermediate_size", 4304))
        layer_norm_eps = vision_cfg.get("layer_norm_eps", 1e-5)
        mm_hidden_size = vision_cfg.get("mm_hidden_size", hidden_dim)
        text_hidden_size = vision_cfg.get("text_hidden_size", 7168)
        patch_size = vision_cfg.get("patch_size", 14)
        ln_eps = vision_cfg.get("projector_ln_eps", 1e-5)
        pos_h = vision_cfg.get("init_pos_emb_height", 64)
        pos_w = vision_cfg.get("init_pos_emb_width", 64)
        pos_t = vision_cfg.get("init_pos_emb_time", 4)

        merge_ks = vision_cfg.get("merge_kernel_size", [2, 2])
        if isinstance(merge_ks, int):
            self.merge_kernel_size = (merge_ks, merge_ks)
        elif isinstance(merge_ks, (list, tuple)):
            self.merge_kernel_size = tuple(merge_ks)
        else:
            self.merge_kernel_size = (2, 2)

        self.merge_type = vision_cfg.get("merge_type", "sd2_tpool")

        # Resolve dtype
        text_config = getattr(pretrained_config, "text_config", pretrained_config)
        self.model_dtype = model_dtype
        self.text_hidden_size = (
            text_config.get("hidden_size", text_hidden_size)
            if isinstance(text_config, dict)
            else getattr(text_config, "hidden_size", text_hidden_size)
        )
        self.config = PretrainedConfig(
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            tie_word_embeddings=False,
        )

        # --- Native sub-modules ---
        self.patch_embed = PatchEmbed3d(hidden_dim, patch_size, pos_h, pos_w, pos_t)
        self.encoder = MoonViT3dEncoder(
            self.model_config,
            hidden_dim,
            num_layers,
            num_heads,
            mlp_dim,
            ln_eps=layer_norm_eps,
        )
        self.mm_projector = PatchMergerMLP(
            self.model_config,
            mm_hidden_size,
            self.text_hidden_size,
            self.merge_kernel_size,
            ln_eps,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load vision encoder and projector weights from a flat dict.

        The HF checkpoint uses these prefixes:
        - ``vision_tower.patch_embed.*`` -> ``self.patch_embed.*``
        - ``vision_tower.encoder.*`` -> ``self.encoder.*``
        - ``mm_projector.*`` -> ``self.mm_projector.*``

        After stripping the top-level prefix (``vision_tower.``), the HF
        weight names map directly to our native module structure.
        """
        # Build a unified weight dict with our module paths
        mapped: Dict[str, torch.Tensor] = {}

        # Vision backbone: strip "vision_tower." prefix
        vis_prefix = _detect_prefix(weights, ["vision_tower.", "visual.", "vision_model."])
        if vis_prefix:
            for k, v in weights.items():
                if k.startswith(vis_prefix):
                    mapped[k[len(vis_prefix) :]] = v

        # Projector: strip "mm_projector." prefix and re-add our path
        proj_prefix = _detect_prefix(weights, ["mm_projector.", "mlp1."])
        if proj_prefix:
            for k, v in weights.items():
                if k.startswith(proj_prefix):
                    mapped[f"mm_projector.{k[len(proj_prefix) :]}"] = v

        if not mapped:
            logger.warning("No vision or projector weights found in checkpoint.")
            return

        converted: Dict[str, torch.Tensor] = {}
        for name, weight in mapped.items():
            if ".wqkv." in name:
                prefix, suffix = name.split(".wqkv.", 1)
                q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
                converted[f"{prefix}.attn.q_proj.{suffix}"] = q_weight
                converted[f"{prefix}.attn.k_proj.{suffix}"] = k_weight
                converted[f"{prefix}.attn.v_proj.{suffix}"] = v_weight
            elif ".wo." in name:
                converted[name.replace(".wo.", ".attn.o_proj.")] = weight
            else:
                converted[name] = weight

        pattern_mapping = {
            r"(.*?)mlp\.fc0\.(.*)": r"\1mlp.up_proj.\2",
            r"(.*?)mlp\.fc1\.(.*)": r"\1mlp.down_proj.\2",
        }
        _load_weights_impl(self, converted, params_map=pattern_mapping)
        logger.info(
            "Loaded %d vision+projector weights (vision=%r, proj=%r)",
            len(converted),
            vis_prefix,
            proj_prefix,
        )

        # Move to GPU and cast to model dtype to match pixel_values.
        device = torch.cuda.current_device()
        self.to(device=device, dtype=self.model_dtype)

    def _extract_features(
        self,
        pixel_values: torch.Tensor,
        grid_thws: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run MoonViT3d + PatchMergerMLP to produce LLM-space embeddings."""
        pixel_values = pixel_values.to(dtype=self.model_dtype)
        hidden = self.patch_embed(pixel_values, grid_thws)
        hidden = self.encoder(hidden, grid_thws)
        if self.merge_type == "sd2_tpool":
            merged = _tpool_patch_merger(hidden, grid_thws, self.merge_kernel_size)
        else:
            raise NotImplementedError(f"Unsupported merge_type: {self.merge_type}")
        projected = self.mm_projector(merged)
        return torch.cat(projected, dim=0)

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        """Encode images/video from multimodal params into LLM-space embeddings."""
        all_pixel_values = []
        all_grid_thws = []

        for param in multimodal_params:
            mm_data = param.multimodal_data
            if mm_data is None:
                continue
            for modality in ("image", "video"):
                if modality not in mm_data:
                    continue
                mod_data = mm_data[modality]
                pv_key = (
                    "pixel_values_videos"
                    if modality == "video" and "pixel_values_videos" in mod_data
                    else "pixel_values"
                )
                if pv_key not in mod_data:
                    continue
                all_pixel_values.append(mod_data[pv_key])
                thw_key = (
                    "video_grid_thw"
                    if modality == "video" and "video_grid_thw" in mod_data
                    else "image_grid_thw"
                )
                if thw_key in mod_data:
                    all_grid_thws.append(mod_data[thw_key])

        if not all_pixel_values:
            return []

        pixel_values = torch.cat(all_pixel_values, dim=0)
        grid_thws = torch.cat(all_grid_thws, dim=0) if all_grid_thws else None
        embeds = self._extract_features(pixel_values, grid_thws)
        if embeds.dim() == 3:
            embeds = embeds.reshape(-1, embeds.shape[-1])
        return [embeds]


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------


def _detect_prefix(
    weights: Dict[str, Any],
    candidates: List[str],
) -> Optional[str]:
    """Find which prefix is used in the weight dict.

    Args:
        weights: Weight name -> tensor mapping.
        candidates: List of prefix strings to try, in priority order.

    Returns:
        The first matching prefix, or None if no match.
    """
    for prefix in candidates:
        if any(k.startswith(prefix) for k in weights):
            return prefix
    return None


# ---------------------------------------------------------------------------
# Input Processor
# ---------------------------------------------------------------------------


class KimiK25InputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    """Input processor for Kimi K2.5 multimodal model.

    Handles image and video preprocessing using the HuggingFace AutoProcessor,
    including Moon-Patch dynamic tiling and token expansion.

    Args:
        model_path: Path to the HF model directory.
        config: HuggingFace pretrained config.
        tokenizer: Tokenizer instance.
        trust_remote_code: Whether to allow custom code from the model repo.
    """

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizerBase,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._config = config
        self._tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast,
            )
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast,
        )
        self._model_path = model_path

        # Resolve dtype from text_config
        text_config = getattr(config, "text_config", config)
        self._dtype = getattr(text_config, "torch_dtype", torch.bfloat16)
        if isinstance(self._dtype, str):
            self._dtype = getattr(torch, self._dtype, torch.bfloat16)

        # Media placeholder token ID
        self._media_placeholder_token_id = getattr(
            config, "media_placeholder_token_id", _MEDIA_PLACEHOLDER_TOKEN_ID
        )

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
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

    def get_vocab_size(self) -> int:
        """Return the K2.5 vocabulary size (163840)."""
        text_config = getattr(self._config, "text_config", self._config)
        return getattr(text_config, "vocab_size", _VOCAB_SIZE)

    def get_mm_token_ids(self) -> Optional[torch.Tensor]:
        """Return the media placeholder token ID used for multimodal tokens.

        The input processor expands each single ``<|media_pad|>`` into N
        copies of the same ``media_placeholder_token_id`` (163605).  This
        ID is within the vocabulary, so the model engine and
        ``fuse_input_embeds`` can locate multimodal tokens via
        ``torch.isin(input_ids, mm_token_ids)``.
        """
        return torch.tensor([self._media_placeholder_token_id], dtype=torch.int32)

    def get_num_tokens_per_image(self, *, image: Union[Image.Image, torch.Tensor], **kwargs) -> int:
        """Calculate the number of tokens for a single image.

        Delegates to the HF KimiK25VisionProcessor's
        ``media_tokens_calculator`` which accounts for Moon-Patch dynamic
        tiling, spatial merge kernel, and patch size.

        Args:
            image: PIL image or Torch tensor to compute the token count for.

        Returns:
            Number of visual tokens this image produces after
            MoonViT3d + PatchMergerMLP.
        """
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "media_tokens_calculator"):
            media_input = {"type": "image", "image": image}
            return image_processor.media_tokens_calculator(media_input)

        # Fallback: compute from config params using the same formula
        # as navit_resize_image in the HF reference code.
        vision_cfg = getattr(self._config, "vision_config", self._config)
        patch_size = getattr(vision_cfg, "patch_size", 14)
        merge_kernel_size = getattr(vision_cfg, "merge_kernel_size", [2, 2])
        if isinstance(merge_kernel_size, (list, tuple)):
            merge_k = merge_kernel_size[0]
        else:
            merge_k = int(merge_kernel_size)

        w, h = image.size
        factor = merge_k * patch_size
        in_patch_limit = 16384
        patch_limit_on_one_side = 512
        s1 = math.sqrt(in_patch_limit / (max(1.0, w // patch_size) * max(1.0, h // patch_size)))
        s2 = patch_limit_on_one_side * patch_size / w
        s3 = patch_limit_on_one_side * patch_size / h
        scale = min(1.0, s1, s2, s3)
        new_w = min(max(1, int(w * scale)), patch_limit_on_one_side * patch_size)
        new_h = min(max(1, int(h * scale)), patch_limit_on_one_side * patch_size)
        pad_w = (factor - new_w % factor) % factor
        pad_h = (factor - new_h % factor) % factor
        token_w = (new_w + pad_w) // factor
        token_h = (new_h + pad_h) // factor
        return token_h * token_w

    def get_num_tokens_per_video(self, *, video: List, **kwargs) -> int:
        """Calculate total tokens for a video (list of PIL frames).

        Each temporal chunk of ``temporal_merge_kernel_size`` frames produces
        tokens based on the spatial dimensions of the first frame.  The total
        is the sum across all chunks.

        Args:
            video: List of PIL Image frames.

        Returns:
            Total visual tokens for the entire video.
        """
        image_processor = getattr(self._processor, "image_processor", None)
        temporal_k = 4
        if image_processor is not None:
            media_proc_cfg = getattr(image_processor, "media_proc_cfg", {})
            temporal_k = media_proc_cfg.get("temporal_merge_kernel_size", 4)

        total_tokens = 0
        for i in range(0, len(video), temporal_k):
            chunk = video[i : i + temporal_k]
            # Each chunk is treated as a video_chunk; token count depends
            # on the spatial dims of the first frame.
            if image_processor is not None and hasattr(image_processor, "media_tokens_calculator"):
                media_input = {
                    "type": "video_chunk",
                    "video_chunk": chunk,
                }
                total_tokens += image_processor.media_tokens_calculator(media_input)
            else:
                # Fallback: same as image token count for the first frame
                total_tokens += self.get_num_tokens_per_image(image=chunk[0])
        return total_tokens

    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """Process multimodal inputs through the HF processor.

        Handles text-only, image, and video inputs. For multimodal inputs,
        runs the HF AutoProcessor to:
        1. Apply Moon-Patch dynamic tiling to images
        2. Generate pixel_values for all tiles
        3. Tokenize text with media placeholder tokens

        Args:
            inputs: Text prompt with optional ``multi_modal_data``.
            sampling_params: Sampling configuration.

        Returns:
            Tuple of (token_ids, extra_processed_inputs) where
            extra_processed_inputs contains ``multimodal_data`` with
            ``pixel_values``.
        """
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data", {})

        images = mm_data.get("image", [])
        videos = mm_data.get("video", [])

        # Text-only path
        if not images and not videos:
            token_ids = self._tokenizer(text_prompt, return_tensors="pt").input_ids[0]
            return token_ids.to(torch.int32).tolist(), {}

        # Build the ``medias`` list expected by KimiK25Processor.
        # The HF processor accepts either ``messages`` (chat format) or
        # both ``medias`` and ``text``.  Since we already have the
        # formatted prompt, we use the second form.
        #
        # For video inputs, we pre-decode them into ``video_chunk`` items
        # using decord/cv2 instead of letting the HF processor call
        # ``split_video_chunks()`` which requires ``mecord``.
        medias: List[Dict[str, Any]] = []
        for img in images:
            # The serving layer may pass images as torch.Tensor (default
            # format="pt"). Convert to PIL for the HF processor.
            if isinstance(img, torch.Tensor):
                from torchvision.transforms.functional import to_pil_image

                img = to_pil_image(img)
            medias.append({"type": "image", "image": img})

        video_prompt_parts: List[str] = []
        if videos:
            # Read video processing config from the HF image processor
            image_processor = getattr(self._processor, "image_processor", None)
            media_proc_cfg = (
                getattr(image_processor, "media_proc_cfg", {}) if image_processor else {}
            )
            tmks = media_proc_cfg.get("temporal_merge_kernel_size", 4)
            sfps = media_proc_cfg.get("sample_fps", 2.0)
            ts_mode = media_proc_cfg.get("timestamp_mode", "hh:mm:ss.fff")

            for vid in videos:
                # The serving layer returns VideoData objects with
                # pre-sampled frames. Extract frames and convert to PIL.
                from tensorrt_llm.inputs.utils import VideoData

                fps_for_chunks = sfps
                frame_indices = None
                if isinstance(vid, VideoData):
                    from torchvision.transforms.functional import to_pil_image

                    metadata_fps = vid.metadata.get("fps")
                    if metadata_fps is not None:
                        fps_for_chunks = float(metadata_fps)
                    frame_indices = vid.metadata.get("frames_indices")
                    frames = [
                        to_pil_image(f) if isinstance(f, torch.Tensor) else f for f in vid.frames
                    ]
                    vid = frames

                if isinstance(vid, list):
                    # Pre-decoded frames (from framework or VideoData).
                    # Split into temporal chunks directly.
                    chunks, vid_prompt = _frames_to_chunks(
                        vid,
                        temporal_merge_kernel_size=tmks,
                        timestamp_mode=ts_mode,
                        fps=fps_for_chunks,
                        frame_indices=frame_indices,
                    )
                else:
                    # File path or bytes: decode with decord/cv2.
                    chunks, vid_prompt = _decode_video_to_chunks(
                        vid,
                        temporal_merge_kernel_size=tmks,
                        sample_fps=sfps,
                        timestamp_mode=ts_mode,
                    )
                medias.extend(chunks)
                video_prompt_parts.append(vid_prompt)

        # Replace the video placeholder(s) in the prompt with actual
        # chunk-level timestamp prompts (matching HF update_raw_text).
        #
        # Two cases:
        # 1. Prompt contains <|kimi_k25_video_placeholder|> (explicit video
        #    placeholder from chat template or manual prompt).
        # 2. Prompt contains auto-inserted <|media_begin|><|media_pad|><|media_end|>
        #    from the framework's placeholder_map (high-level LLM.generate() API).
        #    In this case images and videos share the same placeholder string.
        #    Images come first in the prompt, videos after, so we skip the
        #    first N_images occurrences.
        #
        # In both cases, each video's single placeholder must be replaced
        # with the concatenated per-chunk timestamp prompts (each chunk has
        # its own <|media_pad|> for the vision encoder).
        if video_prompt_parts:
            video_placeholder = getattr(
                self._config, "video_placeholder", "<|kimi_k25_video_placeholder|>"
            )

            if video_placeholder in text_prompt:
                for vid_prompt in video_prompt_parts:
                    text_prompt = text_prompt.replace(video_placeholder, vid_prompt, 1)
            else:
                # Auto-inserted placeholder from framework's placeholder_map.
                # Both images and videos use the same placeholder string, so
                # skip the first N_images occurrences (image placeholders),
                # then replace the next ones (video placeholders).
                auto_ph = "<|media_begin|><|media_pad|><|media_end|>"
                n_img = len(images)
                for vid_prompt in video_prompt_parts:
                    # Find the (n_img+1)-th occurrence and replace it.
                    idx = 0
                    for _ in range(n_img + 1):
                        idx = text_prompt.find(auto_ph, idx)
                        if idx == -1:
                            break
                    if idx != -1:
                        text_prompt = (
                            text_prompt[:idx] + vid_prompt + text_prompt[idx + len(auto_ph) :]
                        )
                    # After replacement, the image placeholders still count
                    # the same but the replaced video placeholder is now
                    # multiple chunk prompts, so n_img stays the same.

        if videos:
            # When we have pre-decoded video_chunk items, we CANNOT pass
            # them through self._processor() because its preprocess_medias()
            # only handles type='image' and type='video' (not 'video_chunk').
            # Instead, call the sub-components directly:
            #   1. image_processor.preprocess() — handles both 'image' and
            #      'video_chunk' types natively.
            #   2. tokenizer — tokenize the text with expanded placeholders.
            image_processor = getattr(
                self._processor, "image_processor", self._processor.media_processor
            )
            preprocessed = image_processor.preprocess(medias, return_tensors="pt")
            text_inputs = self._tokenizer(text_prompt, return_tensors="pt")
            # Merge into a single dict matching the processor's output format.
            # Both BatchFeature and BatchEncoding support dict-like iteration.
            processed = dict(text_inputs)
            processed.update(dict(preprocessed))
        else:
            processed = self._processor(medias=medias, text=text_prompt, return_tensors="pt")

        input_ids = processed["input_ids"][0]

        # Expand each single <|media_pad|> placeholder into N copies so that
        # fuse_input_embeds sees the correct number of MM tokens.
        # N equals the number of visual tokens the vision encoder will
        # produce for that image/video_chunk.
        #
        # We reuse the *same* media_placeholder_token_id (an in-vocab ID)
        # for the expanded slots.  This keeps the token within the
        # vocabulary so it survives any downstream token-range checks and
        # is detectable by the model engine via ``mm_token_ids``.
        #
        # Token count after MoonViT3d + tpool_patch_merger:
        #   num_tokens = (h // merge_kh) * (w // merge_kw)
        # where (t, h, w) comes from grid_thws and merge_kernel_size is
        # typically (2, 2) from the vision config.
        placeholder_id = self._media_placeholder_token_id

        grid_thws = processed.get("grid_thws", processed.get("image_grid_thw"))
        if grid_thws is not None:
            vision_cfg = getattr(self._config, "vision_config", self._config)
            if hasattr(vision_cfg, "merge_kernel_size"):
                merge_kh, merge_kw = vision_cfg.merge_kernel_size
            else:
                merge_kh, merge_kw = 2, 2

            tokens_per_media = [(h // merge_kh) * (w // merge_kw) for _, h, w in grid_thws.tolist()]

            new_ids: List[int] = []
            media_idx = 0
            for tok in input_ids.tolist():
                if tok == self._media_placeholder_token_id:
                    if media_idx >= len(tokens_per_media):
                        raise ValueError(
                            f"More media placeholder tokens "
                            f"({media_idx + 1}) in input_ids than "
                            f"media items in grid_thws "
                            f"({len(tokens_per_media)}). Check that "
                            f"the prompt has the correct number of "
                            f"<|media_pad|> tokens."
                        )
                    new_ids.extend([placeholder_id] * tokens_per_media[media_idx])
                    media_idx += 1
                else:
                    new_ids.append(tok)
            input_ids = torch.tensor(new_ids, dtype=torch.int32)

        # Build multimodal_data dict with pixel values and grid_thws.
        # MoonViT3d requires image_grid_thw for 3D patch embedding.
        #
        # When videos are pre-decoded into video_chunks, the HF processor
        # merges ALL media (images + video_chunks) into unified
        # ``pixel_values`` and ``grid_thws`` tensors.  We put everything
        # under the "image" key because:
        # 1. MoonViT3d processes images and video chunks identically
        #    (temporal dim is encoded in grid_thws).
        # 2. The framework's multimodal_data_device_paths moves
        #    "image.pixel_values" and "image.image_grid_thw" to GPU.
        # 3. The VLM forward() extracts data from the "image" modality.
        #
        # Do NOT duplicate under "video" — that would cause forward()
        # to process the same tensors twice and trigger device mismatches.
        multimodal_data: Dict[str, Any] = {}
        pv = processed.get("pixel_values")
        gt = processed.get("grid_thws", processed.get("image_grid_thw"))
        if pv is not None:
            mm_entry: Dict[str, Any] = {"pixel_values": pv}
            if gt is not None:
                mm_entry["image_grid_thw"] = gt
            multimodal_data["image"] = mm_entry

        return input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }

    def get_prompt_token_ids(
        self,
        inputs: TextPrompt,
        mm_handles: List[Dict[str, Any]],
    ) -> Tuple[List[int], List[int], List[int]]:
        """Build token IDs with multimodal placeholders expanded for disaggregated serving.

        Args:
            inputs: Text prompt input container.
            mm_handles: List of multimodal embedding handles from the
                context phase, each containing ``tensor_size``.

        Returns:
            Tuple of (expanded_ids, mm_token_lengths, mm_token_offsets).
        """
        text_prompt = inputs.get("prompt")
        if not text_prompt:
            raise ValueError("Text prompt is required but not provided")

        text_config = getattr(self._config, "text_config", self._config)
        expected_hidden_size = text_config.hidden_size
        for i, mm_handle in enumerate(mm_handles):
            hidden_size = mm_handle["tensor_size"][1]
            if hidden_size != expected_hidden_size:
                raise RuntimeError(
                    f"Multimodal embedding {i} hidden size {hidden_size} "
                    f"must match model hidden size {expected_hidden_size}"
                )

        input_ids = self._tokenizer(text_prompt, return_tensors="pt").input_ids[0]

        placeholder_id = self._media_placeholder_token_id
        image_mask = input_ids == self._media_placeholder_token_id
        image_positions = torch.where(image_mask)[0]
        num_images = len(image_positions)
        assert num_images == len(mm_handles), (
            f"Number of placeholders ({num_images}) must match "
            f"number of mm_handles ({len(mm_handles)})"
        )

        total_mm_tokens = sum(h["tensor_size"][0] for h in mm_handles)
        final_length = len(input_ids) - num_images + total_mm_tokens

        expanded_ids = torch.empty(final_length, dtype=input_ids.dtype)
        write_pos = 0
        image_cnt = 0
        mm_token_length = []
        mm_token_offsets = []

        for read_pos in range(len(input_ids)):
            if input_ids[read_pos] == self._media_placeholder_token_id:
                mm_token_num = mm_handles[image_cnt]["tensor_size"][0]
                expanded_ids[write_pos : write_pos + mm_token_num] = placeholder_id
                mm_token_offsets.append(write_pos)
                mm_token_length.append(mm_token_num)
                write_pos += mm_token_num
                image_cnt += 1
            else:
                expanded_ids[write_pos] = input_ids[read_pos]
                write_pos += 1

        return (expanded_ids.to(torch.int32).tolist(), mm_token_length, mm_token_offsets)


# ---------------------------------------------------------------------------
# Full VLM Model
# ---------------------------------------------------------------------------


# TODO(kimi-k25): Add multimodal disaggregated serving support.
@register_vision_encoder(KimiK25VisionModel)
@register_auto_model("KimiK25ForConditionalGeneration")
@register_input_processor(
    KimiK25InputProcessor,
    model_type="kimi_k25",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|media_begin|><|media_pad|><|media_end|>",
            "video": "<|media_begin|><|media_pad|><|media_end|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ),
)
class KimiK25ForConditionalGeneration(PreTrainedModel):
    """Kimi K2.5 vision-language model: MoonViT3d + DeepSeek-V3 LLM.

    Single entry point for K2.5. Uses DeepseekV3ForCausalLM directly as
    the text backbone (no wrapper class needed).

    Architecture (matching HF ``KimiK25ForConditionalGeneration``):
        mm_encoder: KimiK25VisionModel (native MoonViT3d + PatchMergerMLP)
        llm: DeepseekV3ForCausalLM (MLA + MoE, 61 layers)
    """

    _LANG_PREFIX = "language_model."

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        config = model_config.pretrained_config
        self._supports_sdpa = True
        super().__init__(config)

        if hasattr(self, "llm"):
            return

        self.model_config = model_config
        # Save the original VLM config before it gets overwritten with
        # text_config at the end of __init__. Needed to create the vision
        # encoder in load_weights() when meta init skips it here.
        self._vlm_pretrained_config = config

        # --- Vision encoder ---
        # Defer the vision encoder when it was constructed under MetaInitMode.
        # TRT-LLM modules may be meta-init compatible, so construction can
        # succeed while parameters/buffers are still meta tensors. Those must
        # not be kept for runtime forward.
        self.mm_encoder = None
        if not DISAGG:
            try:
                mm_encoder = KimiK25VisionModel(model_config)
                if _has_meta_tensors(mm_encoder):
                    logger.info("Vision encoder deferred to load_weights() (MetaInitMode active)")
                else:
                    self.mm_encoder = mm_encoder
            except MetaInitException:
                logger.info("Vision encoder deferred to load_weights() (MetaInitMode active)")

        text_model_config = copy.copy(model_config)
        assert hasattr(config, "text_config"), "Kimi K2.5 config must have text_config"
        text_model_config._frozen = False
        text_model_config.pretrained_config = config.text_config

        # Remap quant exclude_modules: language_model.X -> model.X
        if text_model_config.quant_config.exclude_modules:
            text_model_config.quant_config = copy.copy(text_model_config.quant_config)
            p = self._LANG_PREFIX
            mapped = []
            for m in text_model_config.quant_config.exclude_modules:
                if m.startswith(p):
                    rest = m[len(p) :]
                    if rest.startswith("layers."):
                        rest = "model." + rest
                    mapped.append(rest)
                else:
                    mapped.append(m)
            text_model_config.quant_config.exclude_modules = mapped

        if not text_model_config.skip_create_weights_in_init:
            text_model_config.skip_create_weights_in_init = True
        text_model_config._frozen = True

        self.llm = DeepseekV3ForCausalLM(text_model_config)

        self._media_placeholder_token_id = getattr(
            config, "media_placeholder_token_id", _MEDIA_PLACEHOLDER_TOKEN_ID
        )

        # Align model config with the LLM backbone's text_config.
        # The executor reads eos_token_id and other generation params from
        # model_config.pretrained_config, so it must point to text_config
        # (not the composite VLM config).
        self.config = self.llm.config
        model_config._frozen = False
        model_config.pretrained_config = self.llm.config
        model_config._frozen = True

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "image.image_grid_thw",
            "video.pixel_values_videos",
            "video.video_grid_thw",
            "multimodal_embedding",
        ]

    def load_weights(self, weights) -> None:
        """Load vision + projector + LLM weights from checkpoint."""
        # Create vision encoder if it was skipped during meta init.
        # Use the saved VLM config since model_config.pretrained_config
        # has been overwritten with text_config by __init__.
        if self.mm_encoder is not None and _has_meta_tensors(self.mm_encoder):
            logger.info("Recreating deferred vision encoder after MetaInitMode")
            self.mm_encoder = None

        if self.mm_encoder is None and not DISAGG:
            vision_model_config = copy.copy(self.model_config)
            vision_model_config._frozen = False
            vision_model_config.pretrained_config = self._vlm_pretrained_config
            vision_model_config._frozen = True
            self.mm_encoder = KimiK25VisionModel(vision_model_config)
        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights)

        if any(k.startswith(self._LANG_PREFIX) for k in weights):
            lm_weights = filter_weights("language_model", weights)
            lm_weights = ConsumableWeightsDict(lm_weights)
        else:
            lm_weights = weights
        self.llm.load_weights(lm_weights)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        num_context_requests = attn_metadata.num_contexts
        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds: List[torch.Tensor] = []

        if len(multimodal_params) > 0:
            if not DISAGG:
                mm_embeds = get_multimodal_embeddings(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=multimodal_params[:num_context_requests],
                )
            else:
                raise NotImplementedError("Disaggregated inference not yet supported for K2.5.")
            mm_embeds = find_input_mm_embeds(
                mm_embeds,
                multimodal_params[:num_context_requests],
            )

        fuse_kwargs = kwargs
        mm_token_ids = None
        if len(mm_embeds) > 0:
            placeholder_id = self._media_placeholder_token_id
            num_mm_in_ids = int((input_ids == placeholder_id).sum().item())
            if num_mm_in_ids == 0:
                logger.warning(
                    "Vision embeddings computed but no placeholder tokens "
                    "found in input_ids — embeddings discarded."
                )
                mm_embeds = []
            else:
                # Exclude keys not accepted by fuse_input_embeds
                fuse_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("mm_token_indices", "text_token_indices")
                }
                mm_token_ids = torch.tensor(
                    [placeholder_id],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )

        input_ids, inputs_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            mm_token_ids=mm_token_ids,
            **fuse_kwargs,
        )

        return self.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
        )
