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
"""MiniMax-H3 joint video + audio diffusion transformer.

Port of the Diffusers ``MiniMaxH3Transformer3DModel`` onto the TRT-LLM
VisualGen stack: the checkpoint module tree (and therefore the checkpoint key
layout) is preserved 1:1 so weights copy by name, while attention runs through
the TRT-LLM VisualGen attention backend and linear layers are TRT-LLM
``Linear`` modules (TP/quantization-ready).

MiniMax-H3 runs a single block stack over one packed 1-D sequence holding the
text condition, the conditioning video/audio rows and the target video/audio
rows. Attention is full self-attention over that sequence; there is no
cross-attention and no per-modality block weights. Modality-specific behaviour
comes only from the two input patch projections, the per-row AdaLN modality
tag and the two output heads.

The checkpoint is mixed-precision: ``proj_in`` / ``audio_proj_in`` /
``time_embedder`` / ``proj_out`` / ``audio_proj_out`` are float32 while
everything else is bfloat16.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

# MiniMax-H3 tags every row of the packed sequence with the modality it
# belongs to and keeps one set of AdaLN modulation parameters per
# (timestep, modality) pair: 0 = video, 1 = text, 2 = audio.
MINIMAX_H3_MODALITY_NUM = 3

# Modules kept in float32 by the mixed-precision checkpoint. Entries are
# matched as substrings of the parameter name.
_KEEP_IN_FP32 = ["proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"]


def _apply_rotary_emb(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    r"""
    Rotate the leading `rotary_dim` channels of every head and pass the
    remaining channels through unchanged. `hidden_states` is
    `(batch_size, seq_len, num_heads, head_dim)` and `cos`/`sin` are
    `(seq_len, rotary_dim)`.
    """
    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]

    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()


class MiniMaxH3RotaryPosEmbed(nn.Module):
    r"""
    3-axis rotary embedding over the `(t, h, w)` coordinates of the packed
    sequence.

    A single `inv_freq` buffer of `rope_freq_dim` frequencies is shared by the
    three axes. Each axis contributes `rope_freq_dim` angles, the three blocks
    are concatenated to `3 * rope_freq_dim` and then concatenated with
    themselves so that the `rotate_half` convention rotates
    `2 * 3 * rope_freq_dim` of the `head_dim` channels.
    """

    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
        super().__init__()
        self.rope_freq_dim = rope_freq_dim
        self.rope_theta = rope_theta
        self.register_buffer("inv_freq", self._build_inv_freq("cpu"), persistent=False)

    def _build_inv_freq(self, device) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, 2 * self.rope_freq_dim, 2, dtype=torch.float32, device=device)
                / (2 * self.rope_freq_dim)
            )
        )

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (seq_len, 3) -> cos/sin: (seq_len, 2 * 3 * rope_freq_dim)
        # The buffer is non-persistent, so it is absent from the checkpoint and
        # is neither materialized by meta-init nor moved by a state-dict load;
        # rebuild it whenever it does not match the input's device.
        if self.inv_freq.is_meta or self.inv_freq.device != position_ids.device:
            self.inv_freq = self._build_inv_freq(position_ids.device)
        position_ids = position_ids.to(torch.float32)
        freqs = position_ids.unsqueeze(-1) * self.inv_freq.view(
            1, 1, -1
        )  # (seq_len, 3, rope_freq_dim)
        freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
        freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3RMSNorm(nn.Module):
    r"""
    RMSNorm with the same math and checkpoint key (``weight``) as
    `torch.nn.RMSNorm`, built so it survives VisualGen's meta-init: the
    parameter comes from a factory call rather than ``nn.RMSNorm``'s
    ``reset_parameters``, whose in-place ``fill_`` the meta-init interceptor
    rejects.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.normalized_shape = (hidden_size,)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(hidden_states, self.normalized_shape, self.weight, self.eps)


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    r"""
    Projects the shared timestep embedding into the six per-(timestep,
    modality) modulation parameters of one transformer block.

    `(num_timesteps, time_embed_dim)` -> six tensors of shape
    `(num_timesteps * MINIMAX_H3_MODALITY_NUM, hidden_size)`, in the Diffusers
    `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp` order.
    The row layout of the returned tensors is
    `[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]`, which is what
    `timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags` addresses.

    A single projection is shared by `norm1` and `norm2` and by the three
    modalities, so it is a block-level module of its own, named after the
    checkpoint's `adaln_proj`.
    """

    def __init__(self, time_embed_dim: int, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(
            time_embed_dim, 6 * hidden_size * MINIMAX_H3_MODALITY_NUM, bias=True, dtype=dtype
        )

    def forward(self, temb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # The activation runs at `temb`'s own precision -- float32, since
        # `time_embedder` is a float32 module in this mixed-precision
        # checkpoint -- and only its result is cast down to the bfloat16
        # projection.
        temb = self.linear(nn.functional.silu(temb).to(self.linear.weight.dtype))
        temb = temb.view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    r"""
    Final norm of the packed sequence, shift/scale modulated per row.

    Same module layout and checkpoint keys as
    `AdaLayerNormContinuous` (`norm` plus a `linear` projecting the
    conditioning embedding to `2 * hidden_size`), with two MiniMax-H3
    specifics: the modulation table holds one row per *timestep* and is
    addressed per row of the packed sequence rather than per batch item, and
    the two halves of the projection are `shift` then `scale`.
    """

    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.norm = MiniMaxH3RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True, dtype=dtype)

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor
    ) -> torch.Tensor:
        # As in `MiniMaxH3AdaLayerNormModulation`: activate at `temb`'s
        # precision, cast to the projection's dtype after.
        shift, scale = self.linear(nn.functional.silu(temb).to(self.linear.weight.dtype)).chunk(
            2, dim=-1
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class MiniMaxH3Attention(Attention):
    r"""
    Full self-attention over one packed sequence, with per-head QK-RMSNorm
    and MiniMax-H3's 3-axis partial rotary embedding. There is no
    cross-attention anywhere in MiniMax-H3.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float,
        model_config: DiffusionModelConfig,
        layer_idx: int,
        module_name: Optional[str] = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qkv_mode=QKVMode.SEPARATE_QKV,
            qk_norm=True,
            qk_norm_mode="per_head",
            eps=qk_norm_eps,
            bias=False,
            config=model_config,
            layer_idx=layer_idx,
            module_name=module_name,
        )

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head RMSNorm on 4D tensors [B, S, H, D]."""
        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]

        q, k, v = self.get_qkv(hidden_states)
        q = q.view(batch_size, seq_len, self.local_num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.local_num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.local_num_key_value_heads, self.head_dim)

        q, k = self.apply_qk_norm(q, k)
        if rotary_emb is not None:
            q = _apply_rotary_emb(q, *rotary_emb)
            k = _apply_rotary_emb(k, *rotary_emb)

        out = self._attn_impl(
            q,
            k,
            v,
            attention_mask=PredefinedAttentionMask.FULL,
            timestep=timestep,
        )
        return self.to_out[0](out)


class MiniMaxH3SwiGLU(nn.Module):
    r"""
    Gated linear unit with SiLU, matching the Diffusers `SwiGLU` module the
    checkpoint keys are named after: one fused projection
    `Linear(dim, dim_out * 2)` whose two halves are `up` (first) and `gate`
    (second). Note the halves are the opposite of the usual gated-MLP
    convention.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool, model_config: DiffusionModelConfig):
        super().__init__()
        self.proj = Linear(
            dim_in,
            dim_out * 2,
            bias=bias,
            dtype=model_config.torch_dtype,
            mapping=model_config.mapping,
            quant_config=model_config.quant_config,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.silu(gate)


class MiniMaxH3FeedForward(nn.Module):
    r"""
    Feed-forward block with the Diffusers `FeedForward` module layout
    (`net.0` SwiGLU, `net.1` dropout, `net.2` down projection) so the
    checkpoint keys `ff.net.0.proj.*` and `ff.net.2.*` map by name.
    """

    def __init__(
        self, hidden_size: int, ffn_dim: int, bias: bool, model_config: DiffusionModelConfig
    ):
        super().__init__()
        self.net = nn.ModuleList(
            [
                MiniMaxH3SwiGLU(hidden_size, ffn_dim, bias, model_config),
                nn.Dropout(0.0),
                Linear(
                    ffn_dim,
                    hidden_size,
                    bias=bias,
                    dtype=model_config.torch_dtype,
                    mapping=model_config.mapping,
                    quant_config=model_config.quant_config,
                    skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                    force_dynamic_quantization=model_config.force_dynamic_quantization,
                ),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net[0](hidden_states)
        hidden_states = self.net[1](hidden_states)
        hidden_states = self.net[2](hidden_states)
        return hidden_states


class MiniMaxH3TokenRefinerBlock(nn.Module):
    r"""
    Plain pre-norm transformer block used to refine the projected text
    stream. No AdaLN and no rotary embedding.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        model_config: DiffusionModelConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.norm1 = MiniMaxH3RMSNorm(hidden_size, eps=norm_eps, dtype=model_config.torch_dtype)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
            model_config=model_config,
            layer_idx=layer_idx,
            module_name=f"token_refiner.refiner_blocks.{layer_idx}.attn",
        )
        self.norm2 = MiniMaxH3RMSNorm(hidden_size, eps=norm_eps, dtype=model_config.torch_dtype)
        self.ff = MiniMaxH3FeedForward(hidden_size, ffn_dim, bias=False, model_config=model_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), None)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
        model_config: DiffusionModelConfig,
    ):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    model_config=model_config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_norm = MiniMaxH3RMSNorm(
            hidden_size, eps=final_norm_eps, dtype=model_config.torch_dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    r"""
    MiniMax-H3 block: pre-norm self-attention and feed-forward, each
    modulated by AdaLN parameters selected per row of the packed sequence
    from the `(timestep, modality)` table.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        model_config: DiffusionModelConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.norm1 = MiniMaxH3RMSNorm(hidden_size, eps=norm_eps, dtype=model_config.torch_dtype)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
            model_config=model_config,
            layer_idx=layer_idx,
            module_name=f"transformer_blocks.{layer_idx}.attn",
        )
        self.norm2 = MiniMaxH3RMSNorm(hidden_size, eps=norm_eps, dtype=model_config.torch_dtype)
        self.ff = MiniMaxH3FeedForward(hidden_size, ffn_dim, bias=False, model_config=model_config)
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim=time_embed_dim, hidden_size=hidden_size, dtype=model_config.torch_dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        residual = hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_msa.index_select(0, adaln_indices)
        ) + shift_msa.index_select(0, adaln_indices)
        attn_output = self.attn(norm_hidden_states, rotary_emb, timestep)
        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attn_output

        residual = hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_mlp.index_select(0, adaln_indices)
        ) + shift_mlp.index_select(0, adaln_indices)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = residual + gate_mlp.index_select(0, adaln_indices) * ff_output

        return hidden_states


class MiniMaxH3Transformer3DModel(BaseDiffusionModel):
    r"""
    TRT-LLM port of the MiniMax-H3 joint video + audio transformer.

    The caller is responsible for building the packed layout: patchifying the
    video latents, ordering the rows, and producing the `(t, h, w)` position
    grid, the per-row modality tags and the per-row timestep indices. The
    sequence carries no padding -- attention runs unmasked over one document.
    """

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config

        num_attention_heads = cfg.num_attention_heads
        attention_head_dim = cfg.attention_head_dim
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_layers
        num_refiner_layers = cfg.num_refiner_layers
        ffn_dim = cfg.ffn_dim
        in_channels = cfg.in_channels
        audio_in_channels = cfg.audio_in_channels
        patch_size = tuple(cfg.patch_size)
        text_dim = cfg.text_dim
        freq_dim = cfg.freq_dim
        time_embed_hidden_dim = cfg.time_embed_hidden_dim
        time_embed_dim = cfg.time_embed_dim
        rope_freq_dim = cfg.rope_freq_dim
        rope_theta = cfg.rope_theta
        norm_eps = cfg.norm_eps
        qk_norm_eps = cfg.qk_norm_eps
        final_norm_eps = cfg.final_norm_eps

        self.patch_size = patch_size
        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]

        fp32 = torch.float32
        bf16 = model_config.torch_dtype

        def _linear(in_dim, out_dim, bias, dtype):
            return Linear(
                in_dim,
                out_dim,
                bias=bias,
                dtype=dtype,
                mapping=model_config.mapping,
                quant_config=model_config.quant_config,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                force_dynamic_quantization=model_config.force_dynamic_quantization,
            )

        # 1. Per-modality input projections (float32 in the checkpoint)
        self.proj_in = _linear(video_patch_dim, hidden_size, True, fp32)
        self.audio_proj_in = _linear(audio_in_channels, hidden_size, True, fp32)
        self.context_embedder = _linear(text_dim, hidden_size, True, bf16)

        # 2. Timestep embedding, shared by every AdaLN projection (float32)
        self.time_proj = Timesteps(
            num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=freq_dim, time_embed_dim=time_embed_hidden_dim, out_dim=time_embed_dim
        )

        # 3. Rotary embedding over the packed (t, h, w) grid
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)

        # 4. Text stream refiner
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
            model_config=model_config,
        )

        # 5. The block stack
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    model_config=model_config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(num_layers)
            ]
        )

        # 6. Shared output norm and the two per-modality output heads
        # (float32 in the checkpoint)
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size=hidden_size, time_embed_dim=time_embed_dim, eps=final_norm_eps, dtype=bf16
        )
        self.proj_out = _linear(hidden_size, video_patch_dim, True, fp32)
        self.audio_proj_out = _linear(hidden_size, audio_in_channels, True, fp32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            hidden_states: `(batch_size, num_video_tokens, in_channels * prod(patch_size))`
                Patchified video latent rows -- conditioning rows and target
                rows -- ordered to match `video_indices`.
            audio_hidden_states: `(batch_size, num_audio_tokens, audio_in_channels)`
                Audio latent rows, ordered to match `audio_indices`.
            encoder_hidden_states: `(batch_size, num_text_tokens, text_dim)`
                Text conditioning, ordered to match `text_indices`.
            timestep: `(num_timesteps,)` distinct timestep values in `[0, 1]`,
                unscaled. One forward serves rows at different noise levels.
            timestep_indices: `(seq_len,)` index of each row's timestep in
                `timestep`.
            token_tags: `(seq_len,)` modality of every row: 0 video, 1 text,
                2 audio.
            position_ids: `(seq_len, 3)` `(t, h, w)` rotary coordinates.
            video_indices / audio_indices / text_indices: positions of the
                rows of each modality in the packed sequence.

        Returns:
            The video velocity `(batch_size, num_video_tokens, video_patch_dim)`
            and the audio velocity `(batch_size, num_audio_tokens, audio_in_channels)`.
        """
        sequence_length = position_ids.shape[0]

        rotary_emb = self.rope(position_ids)

        # 1. Project each modality and scatter the rows into the packed
        # sequence buffer. The checkpoint is mixed-precision, so every input
        # is aligned with its projection's activation dtype. ``Linear.dtype``
        # (not ``weight.dtype``) is used: a quantized Linear keeps its
        # high-precision ``dtype`` while ``weight`` holds packed FP4.
        video_embeds = self.proj_in(hidden_states.to(self.proj_in.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.dtype))
        text_embeds = self.token_refiner(text_embeds)

        hidden_states = text_embeds.new_zeros(
            (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
        )
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(
            1, video_indices, video_embeds.to(text_embeds.dtype)
        )
        hidden_states = hidden_states.index_copy(
            1, audio_indices, audio_embeds.to(text_embeds.dtype)
        )

        # 2. One timestep embedding per distinct noise level. `temb` stays at
        # the time embedder's float32 precision; each AdaLN module applies its
        # own activation and casts to its projection's dtype afterwards.
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))

        # 3. Row -> AdaLN table row.
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, timestep=timestep)

        # 4. Both heads run over every row, then the rows of each modality
        # are selected. The heads stay float32 while the block stack runs in
        # bfloat16; align the activation with their parameter dtype.
        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(
            self.proj_out.weight.dtype
        )
        video_output = self.proj_out(hidden_states).index_select(1, video_indices)
        audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)

        return video_output, audio_output

    def load_weights(self, weights: dict) -> None:
        """Copy checkpoint weights into the module tree by (matching) name.

        The module tree mirrors the Diffusers checkpoint key layout 1:1.
        ``skip_create_weights_in_init`` defers ``Linear`` allocation to this
        loader, and when the model config requests dynamic quantization (e.g.
        ``VisualGenArgs(quant_config={"quant_algo": "NVFP4"})``) the
        :class:`DynamicLinearWeightLoader` quantizes each layer's weights on
        the fly while copying them.

        The mixed-precision head/input projections stay float32 regardless of
        ``quant_config``: they are structurally part of the checkpoint's
        precision mix, they are tiny next to the block stack, and keeping them
        full-precision avoids error accumulation at the latent interface.
        """
        # ``skip_create_weights_in_init`` defers Linear weight allocation to
        # the loader, and the loader has already moved the rest of the module
        # tree to its device by this point, so allocate on that same device.
        device = next(
            (p.device for p in self.parameters() if not p.is_meta),
            torch.device("cpu"),
        )

        # The fp32 set is a structural property of this checkpoint, not a user
        # preference: strip the quant config before allocation so
        # ``create_weights()`` builds plain float32 buffers, and so the
        # loader sees ``quant_algo=None`` for them.
        no_quant_config = QuantConfig()
        for name, module in self.named_modules():
            if isinstance(module, Linear) and any(frag in name for frag in _KEEP_IN_FP32):
                module.quant_config = no_quant_config

        for module in self.modules():
            if isinstance(module, Linear):
                module.create_weights()
                module.to(device)

        loader = DynamicLinearWeightLoader(self.model_config)
        loaded = set()
        for name, module in self.named_modules():
            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)
                if weight_dicts:
                    loader.load_linear_weights(module, name, weight_dicts)
                    loaded.update(f"{name}.{k}" for wd in weight_dicts for k in wd)
                continue
            for param_name, param in module._parameters.items():
                if param is None:
                    continue
                key = f"{name}.{param_name}"
                if key in weights:
                    param.data.copy_(weights[key].to(param.dtype))
                    loaded.add(key)
        missing = sorted(set(weights.keys()) - loaded)
        if missing:
            logger.warning(
                f"MiniMaxH3Transformer3DModel: {len(missing)} checkpoint keys not "
                f"consumed: {missing[:10]}..."
            )

    def post_load_weights(self) -> None:
        """Post-load processing: keep the mixed-precision module set in fp32."""
        for name, module in self.named_modules():
            if any(frag in name for frag in _KEEP_IN_FP32):
                module.to(torch.float32)
            if isinstance(module, Linear):
                module.post_load_weights()
