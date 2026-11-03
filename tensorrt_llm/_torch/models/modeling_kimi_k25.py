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
import functools
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from itertools import accumulate
from typing import Any, ClassVar, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union

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

from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalItemPartition,
    MultimodalParams,
    select_multimodal_fields,
)
from tensorrt_llm.inputs.multimodal_data import MEDIA_MODALITIES
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
from ..attention.attention import Attention
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import PredefinedAttentionMask
from ..attention.backends.utils import get_attention_backend
from ..model_config import ModelConfig
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mlp import MLP
from .checkpoints.base_weight_loader import ConsumableWeightsDict
from .modeling_deepseekv3 import DeepseekV3ForCausalLM
from .modeling_multimodal_encoder import MultimodalEncoderMixin
from .modeling_multimodal_mixin import MultimodalEncoderContractError, MultimodalModelMixin
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


def _as_pil_image(image: Union[Image.Image, torch.Tensor]) -> Image.Image:
    """Return the PIL image the HF processor accepts; the serving media loader yields tensors."""
    if isinstance(image, torch.Tensor):
        from torchvision.transforms.functional import to_pil_image

        return to_pil_image(image)
    return image


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


def _vision_requires_replication(model_config: ModelConfig, num_heads: int) -> bool:
    """Whether the MoonViT vision tower must run replicated (tp=1) rather than
    tensor-parallel sharded across the attention-TP ranks.

    Replication is required when either:

    * attention data-parallelism is enabled (attention weights are already
      replicated per rank), or
    * ``num_heads`` — the tower's attention head count, resolved by the caller
      with its per-model default (16 for K2.5, 12 for K3) — is not divisible
      by the attention-TP degree (e.g. Kimi K3's 12 heads under TP16). Such a
      tower cannot be TP-sharded at all, so it must be replicated instead of
      tripping the ``num_heads % tp_size`` assertion in :class:`Attention`.
    """
    mapping = model_config.mapping
    if mapping.enable_attention_dp:
        return True
    # Helix carries its parallelism in cp with tp_size=1, so a tp-only test
    # never trips; the tower has no context-parallel form, so any cp > 1
    # must replicate.
    if mapping.cp_size > 1:
        return True
    return (num_heads % mapping.tp_size) != 0


def _get_vision_tp_mapping(model_config: ModelConfig, num_heads: int) -> Mapping:
    if not _vision_requires_replication(model_config, num_heads):
        return model_config.mapping

    # Fold every parallel dimension (incl. helix cp) into pp so each rank
    # runs the tower replicated; without cp the world size collapses below
    # the rank range under helix.
    attn_ranks = (
        model_config.mapping.pp_size * model_config.mapping.tp_size * model_config.mapping.cp_size
    )
    return Mapping(
        world_size=attn_ranks,
        rank=model_config.mapping.rank,
        gpus_per_node=model_config.mapping.gpus_per_node,
        tp_size=1,
        pp_size=attn_ranks,
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


def _moonvit_payload_keys(modality: str, mod_data: Dict[str, Any]) -> Tuple[str, str]:
    """Return the ``(pixel_values, grid_thw)`` keys MoonViT reads from one modality payload."""
    if modality == "video":
        return (
            "pixel_values_videos" if "pixel_values_videos" in mod_data else "pixel_values",
            "video_grid_thw" if "video_grid_thw" in mod_data else "image_grid_thw",
        )
    return "pixel_values", "image_grid_thw"


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


class MoonViT3dEncoder(nn.Module, MultimodalEncoderMixin):
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

        # Context-only metadata (kv_cache_manager=None) built by the engine via
        # ``MultimodalEncoderMixin.setup_attn_metadata`` at the encoder budget;
        # filled per forward by ``prepare_attn_metadata``.
        self.metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        self.attn_metadata: Optional[AttentionMetadata] = None

    # Per-sequence attention capacity fixed by ``set_attn_max_seq_len``; 0 until the engine
    # sets it, in which case each batch uses its own longest sequence.
    _attn_max_seq_len: int = 0

    def set_attn_max_seq_len(self, max_seq_len: int) -> None:
        """Fix the attention ``max_seq_len`` to the longest sequence the engine can schedule.

        The native attention op is cached per layer by ``max_seq_len`` (which is also its
        attention window). A per-batch value builds, and keeps, a new op with its own device
        buffers in every layer for every new image size.
        """
        if max_seq_len <= 0:
            raise ValueError(f"MoonViT attention max_seq_len must be positive, got {max_seq_len}")
        self._attn_max_seq_len = max_seq_len

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
        attn_metadata.max_seq_len = max(self._attn_max_seq_len, max(seq_lens))
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
        num_heads: int,
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
        mapping = _get_vision_tp_mapping(model_config, num_heads)
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

        # Extract vision config dict (num_heads is resolved here, with this
        # model's default of 16 heads, because the TP-replication decision
        # below must use the same head count the tower is built with).
        vision_cfg = getattr(self.model_config.pretrained_config, "vision_config", {})
        if vision_cfg is None:
            vision_cfg = {}
        if not isinstance(vision_cfg, dict):
            vision_cfg = (
                vision_cfg.to_dict() if hasattr(vision_cfg, "to_dict") else vars(vision_cfg)
            )
        num_heads = vision_cfg.get(
            "vt_num_attention_heads", vision_cfg.get("num_attention_heads", 16)
        )

        # The MoonViT tower cannot be tensor-parallel sharded when its attention
        # head count is not divisible by the attention-TP degree (e.g. Kimi K3's
        # 12 heads under TP16); run the whole tower replicated (tp=1) in that
        # case so module construction and weight loading agree. Attention-DP
        # already replicates the vision tower via its own path, so leave it be.
        if not model_config.mapping.enable_attention_dp:
            self.model_config.mapping = _get_vision_tp_mapping(model_config, num_heads)
        pretrained_config = self.model_config.pretrained_config
        model_dtype = (
            getattr(pretrained_config, "torch_dtype", None)
            or getattr(pretrained_config, "dtype", None)
            or torch.bfloat16
        )
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype, torch.bfloat16)
        pretrained_config.torch_dtype = model_dtype

        # Read HF-prefixed names with unprefixed fallback
        hidden_dim = vision_cfg.get("vt_hidden_size", vision_cfg.get("hidden_size", 1152))
        num_layers = vision_cfg.get("vt_num_hidden_layers", vision_cfg.get("num_hidden_layers", 27))
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
            num_heads,
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
            # The runtime HF loader streams weights as lazy safetensors
            # ``PySafeSlice`` objects (``safe_open(...).get_slice(name)``), which
            # support slicing but none of the torch.Tensor attributes/methods the
            # downstream consume path touches (``.chunk`` here, and ``.device`` /
            # ``.dtype`` / ``.shape`` inside the child ``Linear.load_weights`` ->
            # ``load_weight_shard``). Materialize EVERY value once, up front, so
            # no raw PySafeSlice can leak into any child module. The vision tower
            # + projector is small (168 tensors, non-quantized, tp=1 replicated),
            # so full materialization is cheap; ``[:]`` on an already-real tensor
            # is a harmless view, so the module-parity path is unaffected.
            weight = weight[:]
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

    def validate_multimodal_data(self, multimodal_data: Dict[str, Any]) -> None:
        """Reject processed media that this tower cannot encode.

        Every grid row ``(t, h, w)`` must fit the temporal position table, the 2D RoPE table, and
        the patch merger, and the pixel payload must hold exactly the patches of its grids.

        Raises:
            ValueError: If a grid or a pixel payload is outside these limits.
        """
        kernel_h, kernel_w = self.merge_kernel_size
        max_frames = self.patch_embed.pos_emb.num_frames
        rope = self.encoder.rope_2d
        proj = self.patch_embed.proj
        patch_shape = (proj.in_channels, *proj.kernel_size)
        for modality in ("image", "video"):
            mod_data = multimodal_data.get(modality)
            if not isinstance(mod_data, dict):
                continue
            pv_key, thw_key = _moonvit_payload_keys(modality, mod_data)
            if mod_data.get(thw_key) is None:
                continue
            grid_thws = torch.as_tensor(mod_data[thw_key])
            if grid_thws.ndim != 2 or grid_thws.shape[-1] != 3:
                raise ValueError(
                    f"{thw_key} must have shape [num_grids, 3], got {tuple(grid_thws.shape)}"
                )
            num_patches = 0
            for t, h, w in grid_thws.tolist():
                if (
                    not (
                        1 <= t <= max_frames
                        and 1 <= h <= rope.max_height
                        and 1 <= w <= rope.max_width
                    )
                    or h % kernel_h
                    or w % kernel_w
                ):
                    raise ValueError(
                        f"MoonViT cannot encode the {modality} grid (t={t}, h={h}, w={w}); it "
                        f"supports t <= {max_frames}, h <= {rope.max_height}, "
                        f"w <= {rope.max_width}, and h and w divisible by the merge kernel "
                        f"{self.merge_kernel_size}"
                    )
                num_patches += t * h * w
            pixel_values = mod_data.get(pv_key)
            if isinstance(pixel_values, torch.Tensor) and tuple(pixel_values.shape) != (
                num_patches,
                *patch_shape,
            ):
                raise ValueError(
                    f"{pv_key} must have shape {(num_patches, *patch_shape)} for its grids, "
                    f"got {tuple(pixel_values.shape)}"
                )

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
                pv_key, thw_key = _moonvit_payload_keys(modality, mod_data)
                if pv_key not in mod_data:
                    continue
                all_pixel_values.append(mod_data[pv_key])
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
# Processed media layout and processor geometry
# ---------------------------------------------------------------------------

# Key in ``multimodal_data["image"]`` that lists the chunk count of every video, in order.
_VIDEO_CHUNK_COUNTS_KEY = "video_chunk_counts"

# Processor limits used when the HF image processor has no ``media_proc_cfg``.
_DEFAULT_IN_PATCH_LIMIT = 16384
_DEFAULT_PATCH_LIMIT_ON_ONE_SIDE = 512
_DEFAULT_TEMPORAL_MERGE_KERNEL_SIZE = 4


def _moonvit_item_rows(image_data: Dict[str, Any]) -> Dict[str, List[range]]:
    """Map every processed image and video to its rows of ``image_grid_thw``.

    The input processor packs all media of a request into ``multimodal_data["image"]``: one
    grid row per image, followed by the chunk rows of each video in order.

    Returns:
        ``{"image": [...]}``, plus ``"video"`` when the payload holds videos. Entry ``i`` of a
        modality is the row range of that modality's item ``i``.
    """
    num_rows = len(image_data["image_grid_thw"])
    chunk_counts = [int(count) for count in image_data.get(_VIDEO_CHUNK_COUNTS_KEY, ())]
    num_images = num_rows - sum(chunk_counts)
    if num_images < 0 or any(count <= 0 for count in chunk_counts):
        raise ValueError(
            f"A Kimi media payload with {num_rows} grid rows cannot hold video chunk counts "
            f"{chunk_counts}"
        )
    item_rows = {"image": [range(row, row + 1) for row in range(num_images)]}
    if chunk_counts:
        starts = accumulate(chunk_counts, initial=num_images)
        item_rows["video"] = [
            range(start, start + count) for start, count in zip(starts, chunk_counts)
        ]
    return item_rows


@functools.lru_cache(maxsize=16)
def _navit_max_patches(
    patch_size: int, merge_size: int, in_patch_limit: int, patch_limit_on_one_side: int
) -> int:
    """Return the most patches the HF ``navit_resize_image`` produces for any image size.

    The processor scales ``(w, h)`` by ``min(1, sqrt(in_patch_limit / (w // p * h // p)),
    side / w, side / h)``, where ``p`` is the patch size and ``side`` is
    ``patch_limit_on_one_side * p`` pixels, then pads each side up to a multiple of
    ``merge_size * p``. Flooring ``w // p`` and the padding let a grid exceed
    ``in_patch_limit``. All images with ``w // p == a`` and ``h // p == b`` (``a = 1`` also
    covers ``w < p``) resize to at most the cell's largest width and height under the largest
    scale any of them gets, so the largest cell bound bounds every image. The cells cover
    sides up to ``4 * patch_limit_on_one_side`` patches; the side limit scales a larger image
    down by more than 4x on that side.
    """
    factor = merge_size * patch_size
    side_px = patch_limit_on_one_side * patch_size
    cells = np.arange(1, 4 * patch_limit_on_one_side + 1, dtype=np.int64)
    largest_px = patch_size * cells + patch_size - 1
    smallest_px = np.where(cells > 1, patch_size * cells, 1)

    def padded_side(px: np.ndarray, scale: np.ndarray) -> np.ndarray:
        resized = np.clip(np.floor(px * scale).astype(np.int64), 1, side_px)
        return merge_size * -(-resized // factor)

    best = 0
    # The bound is symmetric in width and height, so cells with b >= a suffice.
    for index, a in enumerate(cells.tolist()):
        scale = np.minimum(
            np.minimum(1.0, np.sqrt(in_patch_limit / (a * cells[index:]))),
            np.minimum(side_px / smallest_px[index], side_px / smallest_px[index:]),
        )
        patches = padded_side(largest_px[index], scale) * padded_side(largest_px[index:], scale)
        best = max(best, int(patches.max()))
    return best


class _MoonViTGeometry(NamedTuple):
    """HF processor limits that bound MoonViT grids; all but ``patch_size`` are in patches."""

    # Pixels per patch side.
    patch_size: int
    # Padding unit of ``navit_resize_image``.
    resize_merge_size: int
    # Spatial kernel ``(kh, kw)`` of the encoder's patch merger.
    merge_kernel_size: Tuple[int, int]
    in_patch_limit: int
    # Per-frame limit for video chunks.
    frame_patch_limit: int
    patch_limit_on_one_side: int
    # Frames per video chunk; the merger pools them into one frame of output embeddings.
    temporal_merge_kernel_size: int


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

    supports_video: ClassVar[bool] = True
    """Whether the HF processor encodes video chunks."""

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
        image = _as_pil_image(image)
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "media_tokens_calculator"):
            media_input = {"type": "image", "image": image}
            return image_processor.media_tokens_calculator(media_input)

        # Fallback: compute from the processor geometry using the same
        # formula as navit_resize_image in the HF reference code.
        geometry = self._moonvit_geometry()
        patch_size = geometry.patch_size
        merge_k = geometry.resize_merge_size

        w, h = image.size
        factor = merge_k * patch_size
        in_patch_limit = geometry.in_patch_limit
        patch_limit_on_one_side = geometry.patch_limit_on_one_side
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
            video: List of PIL Image or Torch tensor frames.

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
            chunk = [_as_pil_image(frame) for frame in video[i : i + temporal_k]]
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

    # ------------------------------------------------------------------
    # Encoder items and profiling.
    #
    # Encoder costs are pre-merge patches (the unit of
    # ``encoder_max_num_tokens``). An item is one image or one whole video.
    # Each grid row is one MoonViT attention sequence and yields
    # ``(h // kh) * (w // kw)`` embeddings; the merger pools a video chunk's
    # frames in time.
    # ------------------------------------------------------------------
    def _merge_kernel_size(self) -> Tuple[int, int]:
        """Return the encoder patch-merger kernel ``(kh, kw)`` from the vision config."""
        vision_cfg = getattr(self._config, "vision_config", self._config)
        if isinstance(vision_cfg, dict):
            merge_kernel_size = vision_cfg.get("merge_kernel_size")
        else:
            merge_kernel_size = getattr(vision_cfg, "merge_kernel_size", None)
        if merge_kernel_size is None:
            return 2, 2
        if isinstance(merge_kernel_size, int):
            return merge_kernel_size, merge_kernel_size
        kernel_h, kernel_w = merge_kernel_size
        return int(kernel_h), int(kernel_w)

    def _moonvit_geometry(self) -> _MoonViTGeometry:
        """Read the grid limits from the loaded HF image processor's ``media_proc_cfg``."""
        image_processor = getattr(self._processor, "image_processor", None)
        cfg = getattr(image_processor, "media_proc_cfg", None) or {}
        merge_kernel_size = self._merge_kernel_size()
        vision_cfg = getattr(self._config, "vision_config", self._config)
        vision_patch_size = (
            vision_cfg.get("patch_size")
            if isinstance(vision_cfg, dict)
            else getattr(vision_cfg, "patch_size", None)
        )
        in_patch_limit = int(cfg.get("in_patch_limit") or _DEFAULT_IN_PATCH_LIMIT)
        return _MoonViTGeometry(
            patch_size=int(cfg.get("patch_size") or vision_patch_size or 14),
            resize_merge_size=int(cfg.get("merge_kernel_size") or merge_kernel_size[0]),
            merge_kernel_size=merge_kernel_size,
            in_patch_limit=in_patch_limit,
            frame_patch_limit=int(cfg.get("in_patch_limit_each_frame") or in_patch_limit),
            patch_limit_on_one_side=int(
                cfg.get("patch_limit_on_one_side") or _DEFAULT_PATCH_LIMIT_ON_ONE_SIDE
            ),
            temporal_merge_kernel_size=int(
                cfg.get("temporal_merge_kernel_size") or _DEFAULT_TEMPORAL_MERGE_KERNEL_SIZE
            ),
        )

    def _max_patches_per_frame(self, modality: str) -> int:
        """Return the most patches one image, or one frame of a video chunk, can have."""
        geometry = self._moonvit_geometry()
        return _navit_max_patches(
            geometry.patch_size,
            geometry.resize_merge_size,
            geometry.in_patch_limit if modality == "image" else geometry.frame_patch_limit,
            geometry.patch_limit_on_one_side,
        )

    def _supported_modalities(self) -> Tuple[str, ...]:
        return ("image", "video") if self.supports_video else ("image",)

    def _largest_grid(self, max_patches: int) -> Optional[Tuple[int, int]]:
        """Return the legal ``(h, w)`` with the most patches, at most ``max_patches``.

        Sides are multiples of the merge kernel and at most ``patch_limit_on_one_side``. Among
        the grids with the most patches, the most square one is returned.
        """
        geometry = self._moonvit_geometry()
        kernel_h, kernel_w = geometry.merge_kernel_size
        side_limit = geometry.patch_limit_on_one_side
        best: Optional[Tuple[int, int]] = None
        for height in range(kernel_h, side_limit + 1, kernel_h):
            width = min(side_limit, max_patches // height) // kernel_w * kernel_w
            if width < kernel_w:
                break
            if best is None or (height * width, -abs(height - width)) > (
                best[0] * best[1],
                -abs(best[0] - best[1]),
            ):
                best = (height, width)
        return best

    def _item_grids(
        self, modality: str, max_num_encoder_tokens: Optional[int]
    ) -> Optional[List[Tuple[int, int, int]]]:
        """Return the grid rows of the largest legal item of one modality.

        ``None`` for ``max_num_encoder_tokens`` asks for the startup maximum: the largest image,
        or one full chunk of the largest video frames. A budget asks for the largest item within
        it: an image, or as many of the largest video frames as fit. ``None`` is returned when
        no item of the modality fits.
        """
        frame_limit = self._max_patches_per_frame(modality)
        if modality == "image":
            budget = (
                frame_limit
                if max_num_encoder_tokens is None
                else min(frame_limit, max_num_encoder_tokens)
            )
            grid = self._largest_grid(budget)
            return None if grid is None else [(1, *grid)]
        if modality != "video" or not self.supports_video:
            raise ValueError(f"Kimi does not support the {modality} modality")
        frame_grid = self._largest_grid(frame_limit)
        frames_per_chunk = self._moonvit_geometry().temporal_merge_kernel_size
        if max_num_encoder_tokens is None:
            num_frames = frames_per_chunk
        else:
            num_frames = max_num_encoder_tokens // (frame_grid[0] * frame_grid[1])
            if num_frames == 0:
                frame_grid = self._largest_grid(max_num_encoder_tokens)
                if frame_grid is None:
                    return None
                num_frames = 1
        return [
            (min(frames_per_chunk, num_frames - start), *frame_grid)
            for start in range(0, num_frames, frames_per_chunk)
        ]

    def _media_item_lengths(
        self, image_data: Dict[str, Any]
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Return the encoder patches and output embeddings of every packed image and video.

        Raises:
            ValueError: If a grid is outside the startup processor geometry or the pixel
                payload does not hold exactly the patches of its grids.
        """
        geometry = self._moonvit_geometry()
        kernel_h, kernel_w = geometry.merge_kernel_size
        side_limit = geometry.patch_limit_on_one_side
        grid_rows = [
            tuple(int(value) for value in row)
            for row in torch.as_tensor(image_data["image_grid_thw"]).tolist()
        ]
        encoder_lengths: Dict[str, List[int]] = {}
        output_lengths: Dict[str, List[int]] = {}
        for modality, item_rows in _moonvit_item_rows(image_data).items():
            max_frames = 1 if modality == "image" else geometry.temporal_merge_kernel_size
            max_frame_patches = self._max_patches_per_frame(modality)
            encoder_lengths[modality] = []
            output_lengths[modality] = []
            for rows in item_rows:
                num_patches = num_embeddings = 0
                for t, h, w in (grid_rows[row] for row in rows):
                    if (
                        not (1 <= t <= max_frames and 1 <= h <= side_limit and 1 <= w <= side_limit)
                        or h % kernel_h
                        or w % kernel_w
                        or h * w > max_frame_patches
                    ):
                        raise ValueError(
                            f"Processed Kimi {modality} grid (t={t}, h={h}, w={w}) is outside "
                            f"the startup processor geometry: t <= {max_frames}, sides <= "
                            f"{side_limit} and divisible by {geometry.merge_kernel_size}, and at "
                            f"most {max_frame_patches} patches per frame"
                        )
                    num_patches += t * h * w
                    num_embeddings += (h // kernel_h) * (w // kernel_w)
                encoder_lengths[modality].append(num_patches)
                output_lengths[modality].append(num_embeddings)
        pixel_values = image_data.get("pixel_values")
        num_patches = sum(t * h * w for t, h, w in grid_rows)
        if isinstance(pixel_values, torch.Tensor) and pixel_values.shape[0] != num_patches:
            raise ValueError(
                f"pixel_values holds {pixel_values.shape[0]} patches, but its grids have "
                f"{num_patches}"
            )
        return encoder_lengths, output_lengths

    def get_mm_item_metadata(
        self,
        prompt_token_ids: List[int],
        multimodal_data: Dict[str, Any],
        mm_item_order: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[MultimodalItemMetadata]:
        """Return one encoder item per image and per video, in prompt order.

        An item costs its pre-merge patches and yields one embedding per merged cell of each of
        its grid rows. Videos follow the images in the processed payload, matching the order in
        which ``call_with_text_prompt`` expands their placeholders.
        """
        prepared = super().get_mm_item_metadata(prompt_token_ids, multimodal_data, mm_item_order)
        image_data = multimodal_data.get("image")
        if (
            multimodal_data.get("multimodal_embedding") is not None
            or not isinstance(image_data, dict)
            or image_data.get("image_grid_thw") is None
        ):
            return prepared
        encoder_lengths, output_lengths = self._media_item_lengths(image_data)
        if mm_item_order:
            item_refs = [(entry["modality"], entry["index"]) for entry in mm_item_order]
        elif prepared is not None:
            item_refs = list(prepared.item_refs)
        else:
            item_refs = [
                (modality, index)
                for modality, lengths in encoder_lengths.items()
                for index in range(len(lengths))
            ]
        num_items = {modality: len(lengths) for modality, lengths in encoder_lengths.items()}
        if len(item_refs) != sum(num_items.values()) or any(
            not 0 <= index < num_items.get(modality, 0) for modality, index in item_refs
        ):
            raise ValueError(
                f"Prompt multimodal items {item_refs} do not match the processed Kimi media "
                f"{num_items}"
            )
        return MultimodalItemMetadata(
            item_refs,
            encoder_token_lengths=[
                encoder_lengths[modality][index] for modality, index in item_refs
            ],
            output_embedding_lengths=[
                output_lengths[modality][index] for modality, index in item_refs
            ],
        )

    def get_mm_max_tokens_per_item(
        self,
        max_num_encoder_tokens: Optional[int] = None,
    ) -> Dict[str, int]:
        """Return the largest legal image (and video) encoder item, in pre-merge patches.

        The largest image follows the HF processor's ``in_patch_limit``,
        ``patch_limit_on_one_side``, patch size, and merge kernel; padding lets it exceed
        ``in_patch_limit``. At startup a video counts as one full chunk of the largest frames.
        """
        max_tokens_per_item = {}
        for modality in self._supported_modalities():
            grids = self._item_grids(modality, max_num_encoder_tokens)
            if grids is not None:
                max_tokens_per_item[modality] = sum(t * h * w for t, h, w in grids)
        return max_tokens_per_item

    def get_max_mm_encoder_output_embeddings(self, max_num_encoder_tokens: int) -> int:
        """Bound the embeddings of one encoder iteration: at most one per merge kernel."""
        kernel_h, kernel_w = self._merge_kernel_size()
        return max_num_encoder_tokens // (kernel_h * kernel_w)

    def get_mm_encoder_attention_metadata_capacity(
        self, max_num_tokens: int
    ) -> Optional[Dict[str, int]]:
        """Bound MoonViT attention sequences: one per grid row of at least one merge kernel."""
        kernel_h, kernel_w = self._merge_kernel_size()
        return {"attention": max(1, max_num_tokens // (kernel_h * kernel_w))}

    def get_dummy_mm_data(
        self,
        *,
        max_num_encoder_tokens: int,
        mm_counts: Dict[str, int],
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Build the packed MoonViT payload of the profiler-selected items.

        Every item is the largest legal one within ``max_num_encoder_tokens``. The payload has
        the layout ``call_with_text_prompt`` produces: images, then videos, all under
        ``"image"``.
        """
        if max_num_encoder_tokens <= 0:
            raise ValueError("max_num_encoder_tokens must be positive")
        unsupported = set(mm_counts) - set(self._supported_modalities())
        if unsupported:
            raise ValueError(f"Kimi cannot build dummy data for modalities {sorted(unsupported)}")
        item_grids: Dict[str, List[List[Tuple[int, int, int]]]] = {}
        total_patches = 0
        for modality in self._supported_modalities():
            num_items = mm_counts.get(modality, 0)
            if num_items < 0:
                raise ValueError(
                    f"Multimodal item counts must be nonnegative; got {num_items} for {modality}"
                )
            if num_items == 0:
                continue
            grids = self._item_grids(modality, max_num_encoder_tokens)
            if grids is None:
                raise ValueError(
                    f"Kimi {modality} dummy does not fit within "
                    f"max_num_encoder_tokens={max_num_encoder_tokens}"
                )
            total_patches += num_items * sum(t * h * w for t, h, w in grids)
            if total_patches > max_num_encoder_tokens:
                raise ValueError(
                    "Requested multimodal dummy items exceed "
                    f"max_num_encoder_tokens={max_num_encoder_tokens}"
                )
            item_grids[modality] = [grids] * num_items
        if not item_grids:
            return {}

        patch_size = self._moonvit_geometry().patch_size
        image_data: Dict[str, Any] = {
            "pixel_values": torch.zeros(
                (total_patches, 3, patch_size, patch_size), dtype=dtype or self.dtype
            ),
            "image_grid_thw": torch.tensor(
                [row for items in item_grids.values() for grids in items for row in grids],
                dtype=torch.int64,
            ),
        }
        if "video" in item_grids:
            image_data[_VIDEO_CHUNK_COUNTS_KEY] = [len(grids) for grids in item_grids["video"]]
        mm_data: Dict[str, Any] = {"image": image_data}
        mm_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = self.get_mm_item_metadata([], mm_data)
        return mm_data

    @torch.inference_mode()
    def call_with_text_prompt(
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
            medias.append({"type": "image", "image": _as_pil_image(img)})

        video_prompt_parts: List[str] = []
        video_chunk_counts: List[int] = []
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
                    metadata_fps = vid.metadata.get("fps")
                    if metadata_fps is not None:
                        fps_for_chunks = float(metadata_fps)
                    frame_indices = vid.metadata.get("frames_indices")
                    vid = [_as_pil_image(frame) for frame in vid.frames]

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
                video_chunk_counts.append(len(chunks))
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
            merge_kh, merge_kw = self._merge_kernel_size()

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
        # ``video_chunk_counts`` records which trailing grid rows belong to
        # which video, so each video can be encoded as one item.
        multimodal_data: Dict[str, Any] = {}
        pv = processed.get("pixel_values")
        gt = processed.get("grid_thws", processed.get("image_grid_thw"))
        if pv is not None:
            mm_entry: Dict[str, Any] = {"pixel_values": pv}
            if gt is not None:
                mm_entry["image_grid_thw"] = gt
            if video_chunk_counts:
                mm_entry[_VIDEO_CHUNK_COUNTS_KEY] = video_chunk_counts
            multimodal_data["image"] = mm_entry

        return input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }

    def build_disagg_prefill_multimodal_inputs(
        self,
        inputs: TextPrompt,
        mm_handles: List[Dict[str, Any]],
        *,
        mm_hashes: list[list[int]] | None = None,
    ) -> tuple[list[int], MultimodalParams]:
        """Build disaggregated prefill inputs from multimodal embedding handles.

        Args:
            inputs: Text prompt input container.
            mm_handles: List of multimodal embedding handles, each
                containing ``tensor_size``.
            mm_hashes: Optional prompt-ordered hashes for KV-cache reuse.

        Returns:
            Expanded token IDs and common multimodal params containing
            prompt layout, item metadata, and embedding handles.
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

        return expanded_ids.to(torch.int32).tolist(), MultimodalParams.from_embeddings(
            mm_handles,
            MultimodalItemMetadata(
                [("image", i) for i in range(len(mm_handles))],
                output_embedding_lengths=[handle["tensor_size"][0] for handle in mm_handles],
            ),
            multimodal_hashes=mm_hashes,
            multimodal_lengths=mm_token_length,
            multimodal_positions=mm_token_offsets,
            multimodal_item_run_cu_offsets=list(range(len(mm_token_length) + 1)),
            multimodal_run_positions=mm_token_offsets,
            multimodal_run_lengths=mm_token_length,
        )


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
class KimiK25ForConditionalGeneration(MultimodalModelMixin, PreTrainedModel):
    """Kimi K2.5 vision-language model: MoonViT3d + DeepSeek-V3 LLM.

    Single entry point for K2.5. Uses DeepseekV3ForCausalLM directly as
    the text backbone (no wrapper class needed).

    Architecture (matching HF ``KimiK25ForConditionalGeneration``):
        mm_encoder: KimiK25VisionModel (native MoonViT3d + PatchMergerMLP)
        llm: DeepseekV3ForCausalLM (MLA + MoE, 61 layers)
    """

    # A video is one encoder item (its hash and prompt runs cover all of its
    # chunks), and only the prompt length bounds its frame count, so no
    # startup encoder budget covers every video. K2.5 keeps inline encoding.
    supports_mm_encoder_item_scheduling = False

    _LANG_PREFIX = "language_model."
    # Vision tower class. Subclasses (Kimi K3) override this so the shared
    # __init__/load_weights MetaInitMode deferral-and-recreation logic
    # constructs their tower without re-implementing either method.
    _VISION_MODEL_CLS = KimiK25VisionModel

    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["V2"]:
        """Prefer KV cache manager V2 for Kimi K2.5."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["PYTHON"]:
        """Kimi-K2.5 defaults to the Python (v2) KV-cache transceiver.

        The DeepSeek-V3 MLA backbone transfers a large latent KV, which the
        Python transceiver handles better in disaggregated serving. This is
        only adopted when the user leaves
        ``cache_transceiver_config.transceiver_runtime`` at 'auto' and the
        effective backend is NIXL; otherwise the C++ transceiver is used.
        """
        return "PYTHON"

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
        # Text-only serving (disable_mm_encoder) and E/P disaggregation both
        # run without a local vision tower.
        self._skip_mm_encoder = DISAGG or model_config.disable_mm_encoder
        if not self._skip_mm_encoder:
            try:
                mm_encoder = self._VISION_MODEL_CLS(model_config)
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
        # Backing storage for the ``mm_token_ids`` property below — built once
        # here so the executor doesn't re-allocate per step.
        self._mm_token_ids = torch.tensor([self._media_placeholder_token_id], dtype=torch.int32)

        # Align model config with the LLM backbone's text_config.
        # The executor reads eos_token_id and other generation params from
        # model_config.pretrained_config, so it must point to text_config
        # (not the composite VLM config).
        self.config = self.llm.config
        model_config._frozen = False
        model_config.pretrained_config = self.llm.config
        model_config._frozen = True

    # Forward spec-dec / weight-loading attrs to self.llm: this wrapper is not
    # itself a spec-dec model but is the outer model that those paths see.
    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    @property
    def model(self):
        return self.llm.model

    @property
    def lm_head(self):
        return self.llm.lm_head

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def set_guided_decoder(self, *args, **kwargs):
        return self.llm.set_guided_decoder(*args, **kwargs)

    def load_draft_weights(self, *args, **kwargs):
        return self.llm.load_draft_weights(*args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "image.image_grid_thw",
            "video.pixel_values_videos",
            "video.video_grid_thw",
            "multimodal_embedding",
        ]

    @property
    def mm_token_ids(self) -> torch.Tensor:
        """Surface the in-vocab media placeholder to the model engine so
        ``runners.prepare_multimodal_indices`` selects the ``torch.isin`` predicate
        instead of the OOV (``>= vocab_size``) fallback (which would miss
        Kimi's placeholder and force ``fuse_input_embeds`` through the
        ``torch.where`` host-sync path on GPU input_ids).
        """
        return self._mm_token_ids

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        return self._mm_token_ids

    @property
    def language_model(self) -> nn.Module:
        return self.llm

    @property
    def text_embedding_layer(self) -> nn.Module:
        return self.llm.model.embed_tokens

    @property
    def embedding_dim(self) -> int:
        return self.llm.config.hidden_size

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.text_embedding_layer.weight.dtype

    def encode_multimodal_inputs(
        self,
        multimodal_params: Sequence[MultimodalParams],
    ) -> torch.Tensor:
        """Run MoonViT; rows follow param order, then grid-row order within each param."""
        if self.mm_encoder is None:
            raise ValueError("Raw multimodal inputs require a local multimodal encoder.")
        embeddings = self.mm_encoder.forward(list(multimodal_params))
        if not embeddings:
            raise MultimodalEncoderContractError("The multimodal params carry no MoonViT media")
        return embeddings[0]

    def build_multimodal_encoder_input(
        self,
        param: MultimodalParams,
        local_item_indices: Sequence[int],
        modality: Optional[str] = None,
    ) -> MultimodalParams:
        """Select images or videos from the packed MoonViT payload.

        Images and video chunks share ``multimodal_data["image"]`` (see
        ``_moonvit_item_rows``), so the items of either modality are grid-row ranges of that
        payload. The view keeps the selected rows in the requested item order.
        """
        image_data = param.multimodal_data.get("image")
        if not isinstance(image_data, dict) or image_data.get("image_grid_thw") is None:
            raise MultimodalEncoderContractError(
                "Kimi media items must be packed under multimodal_data['image']"
            )
        item_rows = _moonvit_item_rows(image_data)
        if modality is None:
            present = [name for name, rows in item_rows.items() if rows]
            if len(present) != 1:
                raise NotImplementedError(
                    "Selecting Kimi media items from a mixed image/video payload needs the modality"
                )
            modality = present[0]
        if modality not in item_rows:
            raise MultimodalEncoderContractError(
                f"The Kimi media payload holds no {modality} items"
            )
        rows = [row for index in local_item_indices for row in item_rows[modality][index]]
        row_patches = tuple(
            t * h * w for t, h, w in torch.as_tensor(image_data["image_grid_thw"]).tolist()
        )
        selected = select_multimodal_fields(
            image_data,
            {
                "pixel_values": MultimodalItemPartition(item_lengths=row_patches),
                "image_grid_thw": MultimodalItemPartition(),
            },
            rows,
        )
        selected.pop(_VIDEO_CHUNK_COUNTS_KEY, None)
        if modality == "video":
            selected[_VIDEO_CHUNK_COUNTS_KEY] = [
                len(item_rows["video"][index]) for index in local_item_indices
            ]
        residual_data = {
            key: value
            for key, value in param.multimodal_data.items()
            if key not in MEDIA_MODALITIES
        }
        residual_data["image"] = selected
        return MultimodalParams(
            multimodal_data=residual_data,
            multimodal_input=(
                copy.copy(param.multimodal_input) if param.multimodal_input is not None else None
            ),
        )

    def validate_multimodal_request_data(self, mm_data: dict) -> None:
        """Reject media that MoonViT cannot encode before the request is scheduled."""
        if self.mm_encoder is not None:
            self.mm_encoder.validate_multimodal_data(mm_data)

    def load_weights(self, weights) -> None:
        """Load vision + projector + LLM weights from checkpoint."""
        # Create vision encoder if it was skipped during meta init.
        # Use the saved VLM config since model_config.pretrained_config
        # has been overwritten with text_config by __init__.
        if self.mm_encoder is not None and _has_meta_tensors(self.mm_encoder):
            logger.info("Recreating deferred vision encoder after MetaInitMode")
            self.mm_encoder = None

        if self.mm_encoder is None and not self._skip_mm_encoder:
            vision_model_config = copy.copy(self.model_config)
            vision_model_config._frozen = False
            vision_model_config.pretrained_config = self._vlm_pretrained_config
            vision_model_config._frozen = True
            self.mm_encoder = self._VISION_MODEL_CLS(vision_model_config)
        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights)

        if any(k.startswith(self._LANG_PREFIX) for k in weights):
            lm_weights = filter_weights("language_model", weights)
            lm_weights = ConsumableWeightsDict(lm_weights)
            checkpoint_dir = getattr(weights, "checkpoint_dir", None)
            if checkpoint_dir is not None:
                lm_weights.checkpoint_dir = checkpoint_dir
            lm_weights.checkpoint_prefix = self._LANG_PREFIX
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
        multimodal_params: Optional[List[MultimodalParams]] = None,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_params = multimodal_params or []
        mm_embeds: List[torch.Tensor] = []
        mm_token_ids = None

        if len(multimodal_params) > 0:
            if self._skip_mm_encoder:
                raise NotImplementedError(
                    "Multimodal input is not supported: the vision encoder is disabled on this server "
                    "(disable_mm_encoder or TLLM_MULTIMODAL_DISAGGREGATED)."
                )
            mm_ctx_params = multimodal_params[: attn_metadata.num_contexts]
            mm_embeds = get_multimodal_embeddings(
                encoder_forward_fn=self.mm_encoder.forward,
                multimodal_params=mm_ctx_params,
            )
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_ctx_params)

            # ``runners.prepare_multimodal_indices`` now sees Kimi's
            # in-vocab placeholder via ``self.mm_token_ids`` and emits indices
            # that match ``find_input_mm_embeds``'s active-chunk slice. The
            # previous ``(input_ids == placeholder).sum().item()`` guard was a
            # host sync used to detect chunked-prefill / KV-reuse mismatches;
            # that mismatch surfaces as a row-count error inside
            # ``fuse_input_embeds`` itself, so the explicit check is no longer
            # needed.
            if len(mm_embeds) > 0:
                mm_token_ids = self.mm_token_ids.to(input_ids.device, dtype=input_ids.dtype)
        raw_input_ids = input_ids
        input_ids, inputs_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            mm_token_ids=mm_token_ids,
            mm_token_indices=kwargs.get("mm_token_indices"),
            text_token_indices=kwargs.get("text_token_indices"),
        )

        # With fused image embeddings input_ids is None; the speculative
        # drafter reads the prompt token IDs from orig_input_ids instead.
        return self.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            orig_input_ids=raw_input_ids,
            **kwargs,
        )
