# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project
"""Native TRTLLM Qwen-Image 2.1 transformer.

This module ports the public Diffusers Qwen-Image 2.1 transformer into
TRTLLM-owned PyTorch modules with matching checkpoint key names.  It does not
import the upstream Diffusers pipeline, transformer, attention processor, UNet
or DiT runtime components.  The implementation keeps the Qwen-Image 2.1
single-stream block-causal attention semantics and prefix KV cache required by
``QwenImage21Pipeline``.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel

_IMG_TOKENS_PER_SLOT = 4


class QwenImage21KVLayerCache:
    """Per-layer prefix KV cache used by Qwen-Image 2.1 attention."""

    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = k
        self.v = v

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None or self.v is None:
            raise RuntimeError("Qwen-Image 2.1 KV cache layer has not been populated.")
        return self.k, self.v


class QwenImage21KVCache:
    """Container for all transformer blocks' prefix KV caches."""

    def __init__(self, num_layers: int) -> None:
        self.layer_caches = [QwenImage21KVLayerCache() for _ in range(int(num_layers))]

    def get_layer(self, layer_idx: int) -> QwenImage21KVLayerCache:
        return self.layer_caches[layer_idx]


def apply_rotary_emb_qwen(x: torch.Tensor, freqs_cis: torch.Tensor, use_real: bool = False) -> torch.Tensor:
    """Apply Qwen complex rotary embeddings to ``[B, S, H, D]`` tensors."""

    if use_real:
        raise NotImplementedError("Qwen-Image 2.1 uses complex rotary frequencies (use_real=False).")
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenImage21TemporalTimesteps(nn.Module):
    """Sinusoidal timestep embedding with cos channels followed by sin channels."""

    def __init__(self, timestep_dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> None:
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor
        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self.time_factor * timestep.float()
        args = timestep[:, None] * self.freqs[None].to(timestep.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(timestep.dtype)


class _BiaslessTimestepEmbedding(nn.Module):
    """Diffusers-compatible ``TimestepEmbedding(..., sample_proj_bias=False)``."""

    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=False)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class QwenImage21TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        self.timestep_embedder = _BiaslessTimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    """RMSNorm whose checkpoint stores the effective scale as ``weight + 1``."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + self.eps)
        return (hidden_states * rrms * (self.weight.float() + 1)).to(input_dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.out_layer(hidden_states)


class QwenImage21SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out(self.activation_fn(self.gate_layer(hidden_states)) * self.proj(hidden_states))


def _select_modulation_rows(params: torch.Tensor, target_token_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Broadcast per-sample modulation rows over token positions."""

    if target_token_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)


class QwenImage21AdaLayerNormContinuous(nn.Module):
    """Final adaptive LayerNorm used by Qwen-Image 2.1 (scale only)."""

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        target_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        scale = _select_modulation_rows(scale, target_token_mask)
        return self.norm(hidden_states) * (1 + scale)


def _sdpa_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run SDPA on ``[B, S, H, D]`` tensors and return the same layout."""

    out = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return out.transpose(1, 2)


def _qwenimage21_prefix_segments(image_ids: torch.Tensor, prefix_len: int) -> list[tuple[int, int, bool]]:
    """Split the prefix into runs of equal image ids: ``(start, end, is_text)``."""

    prefix_ids = image_ids[:prefix_len].tolist()
    segments: list[tuple[int, int, bool]] = []
    if prefix_len <= 0:
        return segments
    start = 0
    for index in range(1, prefix_len + 1):
        if index == prefix_len or prefix_ids[index] != prefix_ids[start]:
            segments.append((start, index, prefix_ids[start] < 0))
            start = index
    return segments


def _qwenimage21_prepare_qkv(
    attn: "QwenImage21Attention",
    hidden_states: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    layer_cache: Optional[QwenImage21KVLayerCache],
    kv_cache_mode: Optional[str],
    cache_write_slice: Optional[slice],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))

    query = attn.norm_q(query).to(value.dtype)
    key = attn.norm_k(key).to(value.dtype)

    if rotary_emb is not None:
        query = apply_rotary_emb_qwen(query, rotary_emb, use_real=False)
        key = apply_rotary_emb_qwen(key, rotary_emb, use_real=False)

    if layer_cache is not None:
        if kv_cache_mode == "extract" and cache_write_slice is not None:
            layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
        elif kv_cache_mode == "cached":
            cached_k, cached_v = layer_cache.get()
            key = torch.cat([cached_k, key], dim=1)
            value = torch.cat([cached_v, value], dim=1)

    return query, key, value, query.shape[1]


class QwenImage21AttnProcessor:
    """Exact SDPA processor for Qwen-Image 2.1 block-causal attention."""

    def __call__(
        self,
        attn: "QwenImage21Attention",
        hidden_states: torch.Tensor,
        attention_mask: Any | None = None,
        rotary_emb: torch.Tensor | None = None,
        layer_cache: QwenImage21KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )

        if segments is None:
            mask = None if key_valid is None else key_valid[:, None, None, :]
            hidden_states = _sdpa_bshd(query, key, value, attn_mask=mask)
        else:
            outputs = []
            prefix_len = segments[-1][1] if segments else 0
            for start, end, is_text in segments:
                seg_mask = None
                if is_text:
                    seg_len = end - start
                    seg_mask = torch.cat(
                        [
                            torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                            torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
                        ],
                        dim=1,
                    )[None, None]
                if key_valid is not None:
                    seg_key_valid = key_valid[:, None, None, :end]
                    seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
                outputs.append(_sdpa_bshd(query[:, start:end], key[:, :end], value[:, :end], attn_mask=seg_mask))

            target_mask = None if key_valid is None else key_valid[:, None, None, :]
            outputs.append(_sdpa_bshd(query[:, prefix_len:], key, value, attn_mask=target_mask))
            hidden_states = torch.cat(outputs, dim=1)

        hidden_states = hidden_states[:, :seq_len_q].flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class QwenImage21Attention(nn.Module):
    """Checkpoint-compatible attention module for Qwen-Image 2.1."""

    _default_processor_cls = QwenImage21AttnProcessor

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6, processor: Any | None = None) -> None:
        super().__init__()
        self.heads = heads
        self.inner_dim = heads * dim_head
        self.use_bias = False
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = QwenImage21ZeroCenterRMSNorm(dim_head, eps=eps)
        self.norm_k = QwenImage21ZeroCenterRMSNorm(dim_head, eps=eps)
        self.set_processor(processor if processor is not None else self._default_processor_cls())

    def set_processor(self, processor: Any) -> None:
        self.processor = processor

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.processor(self, hidden_states, **kwargs)


class QwenImage21TransformerBlock(nn.Module):
    """One Qwen-Image 2.1 single-stream transformer block."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImage21Attention(dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenImage21SwiGLUFeedForward(hidden_size=dim, mlp_hidden_size=dim * mlp_ratio)

    def _modulate(
        self,
        hidden_states: torch.Tensor,
        mod_params: torch.Tensor,
        target_token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, gate = mod_params.chunk(2, dim=-1)
        scale = _select_modulation_rows(scale, target_token_mask)
        gate = _select_modulation_rows(gate, target_token_mask)
        return hidden_states * (1 + scale), gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        attention_mask: Any | None = None,
        target_token_mask: torch.Tensor | None = None,
        layer_cache: QwenImage21KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
        attn_output = self.attn(
            hidden_states=img_modulated,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            layer_cache=layer_cache,
            kv_cache_mode=kv_cache_mode,
            cache_write_slice=cache_write_slice,
            segments=segments,
            key_valid=key_valid,
        )
        hidden_states = hidden_states + img_gate1.tanh() * attn_output

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
        hidden_states = hidden_states + img_gate2.tanh() * self.img_mlp(img_modulated2)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class QwenImage21Rope(nn.Module):
    """3-axis rotary embedding over Qwen-Image 2.1 joint text/image sequence."""

    def __init__(self, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(8192)
        neg_index = torch.arange(1024).flip(0) * -1 - 1
        self.freqs = [
            torch.cat([self.rope_params(pos_index, dim, theta), self.rope_params(neg_index, dim, theta)], dim=0)
            for dim in axes_dim
        ]

    @staticmethod
    def rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, img_shapes: list[tuple[int, int, int]], image_pad_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        self.freqs = [freq.to(device) for freq in self.freqs]
        frame_index: list[int] = []
        image_height_index: list[int] = []
        image_width_index: list[int] = []
        cursor, position = 0, 0
        total_len = image_pad_mask.shape[-1]
        is_image_token = image_pad_mask.tolist()

        for _, height, width in img_shapes:
            block_start = is_image_token.index(True, cursor)
            text_len = block_start - cursor
            frame_index.extend(range(position, position + text_len))
            position += text_len

            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)

            image_height_index.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
            image_width_index.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])

        if cursor < total_len:
            frame_index.extend(range(position, position + total_len - cursor))

        frame_index_t = torch.tensor(frame_index, dtype=torch.long, device=device)
        height_index = frame_index_t.clone()
        width_index = frame_index_t.clone()
        height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long, device=device)
        width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long, device=device)

        return torch.cat([self.freqs[0][frame_index_t], self.freqs[1][height_index], self.freqs[2][width_index]], dim=-1)


class QwenImage21Transformer2DModel(BaseDiffusionModel):
    """TRTLLM-owned Qwen-Image 2.1 single-stream transformer."""

    config_name = "QwenImage21Transformer2DModel"
    _no_split_modules = ["QwenImage21TransformerBlock"]

    def __init__(
        self,
        model_config: Any = None,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if model_config is None:
            from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

            model_config = DiffusionModelConfig(component_name="transformer")
        super().__init__(model_config)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.num_layers = int(num_layers)
        self.attention_head_dim = int(attention_head_dim)
        self.num_attention_heads = int(num_attention_heads)
        self.context_in_dim = int(context_in_dim)
        self.mlp_ratio = int(mlp_ratio)
        self.axes_dims_rope = tuple(axes_dims_rope)
        self.eps = float(eps)
        self.causal_condition = bool(causal_condition)
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.config = SimpleNamespace(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            attention_head_dim=self.attention_head_dim,
            num_attention_heads=self.num_attention_heads,
            context_in_dim=self.context_in_dim,
            mlp_ratio=self.mlp_ratio,
            axes_dims_rope=self.axes_dims_rope,
            eps=self.eps,
            causal_condition=self.causal_condition,
        )

        self.pos_embed = QwenImage21Rope(theta=10000, axes_dim=list(self.axes_dims_rope))
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(embedding_dim=self.inner_dim)
        self.txt_in = QwenImage21TextProjection(self.context_in_dim, self.inner_dim, eps=self.eps)
        self.img_in = nn.Linear(self.in_channels * self.patch_size * self.patch_size, self.inner_dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    mlp_ratio=self.mlp_ratio,
                    eps=self.eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm_out = QwenImage21AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=self.eps)
        self.proj_out = nn.Linear(self.inner_dim, self.patch_size * self.patch_size * self.out_channels, bias=False)
        self.gradient_checkpointing = False

    @property
    def device(self) -> torch.device:
        return self.proj_out.weight.device

    @staticmethod
    def build_token_metadata(image_pad_mask: torch.Tensor, img_shapes: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        block_lengths = [math.prod(shape) for shape in img_shapes]
        if sum(block_lengths) != image_positions.numel():
            raise ValueError(
                f"img_shapes accounts for {sum(block_lengths)} image tokens but image_pad_mask marks {image_positions.numel()}."
            )
        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        block_ids = torch.repeat_interleave(
            torch.arange(len(block_lengths), device=image_pad_mask.device),
            torch.tensor(block_lengths, device=image_pad_mask.device),
        )
        image_ids[image_positions] = block_ids
        target_token_mask = torch.zeros_like(image_pad_mask)
        target_token_mask[image_positions[-block_lengths[-1] :]] = True
        return image_ids, target_token_mask

    @classmethod
    def from_config_dict(cls, config: dict[str, Any], model_config: Any = None) -> "QwenImage21Transformer2DModel":
        kwargs = dict(config)
        kwargs.pop("_class_name", None)
        kwargs.pop("_diffusers_version", None)
        if model_config is not None:
            kwargs["model_config"] = model_config
        return cls(**kwargs)

    @contextmanager
    def cache_context(self, branch: str):
        old = getattr(self, "_cache_branch", None)
        self._cache_branch = branch
        try:
            yield
        finally:
            self._cache_branch = old

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(weights, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Qwen-Image 2.1 transformer checkpoint mismatch: "
                f"missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
            )

    def to_inference_dtype(self) -> "QwenImage21Transformer2DModel":
        return self.to(dtype=self.model_config.torch_dtype)

    def post_load_weights(self) -> None:
        return None

    def register_cuda_graph_extra_key_fns(self, runner: Any) -> None:
        super().register_cuda_graph_extra_key_fns(runner)

        def img_shapes_key(*args: Any, **kwargs: Any) -> tuple | None:
            img_shapes = kwargs.get("img_shapes")
            if img_shapes is None:
                return None
            return tuple(tuple(tuple(shape) for shape in item) for item in img_shapes)

        runner.register_extra_key_fn("img_shapes", img_shapes_key)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: QwenImage21KVCache | None = None,
        kv_cache_mode: str | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        del attention_kwargs, kwargs
        batch_size = hidden_states.shape[0]
        if kv_cache is not None and not self.config.causal_condition:
            raise ValueError("kv_cache requires causal_condition=True.")
        if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
            raise ValueError(f"kv_cache_mode must be 'extract' or 'cached' when kv_cache is provided, got {kv_cache_mode!r}.")
        if kv_cache is None and kv_cache_mode is not None:
            raise ValueError(f"kv_cache_mode is {kv_cache_mode!r} but no kv_cache was passed.")

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        target_tokens = math.prod(img_shapes[0][-1])
        joint_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(batch_size, target_tokens // _IMG_TOKENS_PER_SLOT, encoder_hidden_states.shape[2]),
            ],
            dim=1,
        )
        joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
        joint_hidden_states[:, image_pad_mask] = hidden_states

        rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, device=hidden_states.device)
        image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])

        timestep = timestep.to(hidden_states.dtype)
        if self.config.causal_condition:
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask: Optional[torch.Tensor] = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)

        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(batch_size, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device)
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]

        prefix_len = int((~target_token_mask).sum())
        if kv_cache_mode == "cached":
            joint_hidden_states = joint_hidden_states[:, prefix_len:]
            rotary_emb = rotary_emb[prefix_len:]
            if modulation_mask is not None:
                modulation_mask = modulation_mask[prefix_len:]
            cache_write_slice = None
            block_segments = None
            block_key_valid = None if joint_key_valid is None else joint_key_valid
        else:
            cache_write_slice = slice(0, prefix_len) if kv_cache_mode == "extract" else None
            block_segments = _qwenimage21_prefix_segments(image_ids, prefix_len)
            block_key_valid = joint_key_valid

        for index_block, block in enumerate(self.transformer_blocks):
            layer_cache = kv_cache.get_layer(index_block) if kv_cache is not None else None
            joint_hidden_states = block(
                hidden_states=joint_hidden_states,
                modulation=modulation,
                rotary_emb=rotary_emb,
                attention_mask=None,
                target_token_mask=modulation_mask,
                layer_cache=layer_cache,
                kv_cache_mode=kv_cache_mode,
                cache_write_slice=cache_write_slice,
                segments=block_segments,
                key_valid=block_key_valid,
            )

        joint_hidden_states = self.norm_out(joint_hidden_states, temb, modulation_mask)
        output = self.proj_out(joint_hidden_states)
        if not return_dict:
            return (output,)
        return SimpleNamespace(sample=output)


# Compatibility aliases used by older parity tests/imports.
QwenImageTransformerBlock = QwenImage21TransformerBlock
QwenJointAttention = QwenImage21Attention

__all__ = [
    "QwenImage21Attention",
    "QwenImage21AttnProcessor",
    "QwenImage21KVCache",
    "QwenImage21KVLayerCache",
    "QwenImage21Transformer2DModel",
    "QwenImage21TransformerBlock",
    "QwenImageTransformerBlock",
    "QwenJointAttention",
    "apply_rotary_emb_qwen",
]
