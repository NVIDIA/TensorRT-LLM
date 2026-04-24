# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image MMDiT transformer.

Phase 0 landed the stub. Phase 1 is being filled in incrementally,
one module at a time, with a per-milestone parity test vs the
``diffusers.models.transformers.transformer_qwenimage`` reference.

Milestone state:
- M2 (this file, done): ``QwenTimestepProjEmbeddings`` ported with
  parity test in ``tests/unittest/_torch/visual_gen/test_qwen_image_parity.py``.
- M3..M6 (pending): QwenEmbedRope, joint attention, MMDiT block,
  full transformer stack.
- M7 (pending): pipeline forward (denoise + VAE decode + text enc).

Reference source:
    /usr/local/lib/python3.12/dist-packages/diffusers/models/transformers/transformer_qwenimage.py
Full design doc: /home/scratch.asteiner/trtllm-qwen-image/PHASE1_PLAN.md
"""

import math
from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch import nn

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


_NOT_IMPLEMENTED_MSG = (
    "Qwen-Image transformer is not yet fully implemented in TensorRT-LLM. "
    "See PHASE1_PLAN.md in the qwen_image handoff directory for the full "
    "milestone plan. Individual modules (e.g. QwenTimestepProjEmbeddings) "
    "are available and parity-tested; the full transformer stack is WIP."
)


# ---------------------------------------------------------------------------
# Timestep embedding utilities (M2 -- done, parity-tested).
# ---------------------------------------------------------------------------


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal positional embedding, bit-exact port of diffusers.

    Source:
        ``diffusers.models.embeddings.get_timestep_embedding``
        (at commit matching diffusers 0.37.1).
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Sinusoidal timestep feature extractor.

    Stateless. Mirrors ``diffusers.models.embeddings.Timesteps`` with the
    Qwen-Image-specific defaults (``num_channels=256``,
    ``flip_sin_to_cos=True``, ``downscale_freq_shift=0``, ``scale=1000``).
    """

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """Two-linear timestep-to-conditioning projector.

    Mirrors ``diffusers.models.embeddings.TimestepEmbedding`` in the
    simple (bias-on, no post-act, SiLU-between) configuration used by
    Qwen-Image. Phase 2 should substitute
    ``tensorrt_llm._torch.modules.linear.Linear`` here to pick up
    FP8/NVFP4 quantization support; bare ``nn.Linear`` is used in Phase 1
    to guarantee parity while the rest of the transformer is brought up.
    """

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class QwenTimestepProjEmbeddings(nn.Module):
    """Qwen-Image timestep-conditioning stack.

    Mirrors ``diffusers.models.transformers.transformer_qwenimage.QwenTimestepProjEmbeddings``.
    The HF state_dict keys map directly:

    - ``time_text_embed.timestep_embedder.linear_1.{weight,bias}``
    - ``time_text_embed.timestep_embedder.linear_2.{weight,bias}``

    ``use_additional_t_cond`` is False for base Qwen-Image; we keep the
    argument for forward-compatibility with zero-cond variants.
    """

    def __init__(self, embedding_dim: int, use_additional_t_cond: bool = False):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        addition_t_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_states.dtype)
        )
        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError(
                    "When use_additional_t_cond=True, addition_t_cond "
                    "must be provided."
                )
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb
        return conditioning


# ---------------------------------------------------------------------------
# Top-level transformer (stub -- Phase 1 WIP, see PHASE1_PLAN.md).
# ---------------------------------------------------------------------------


class QwenImageTransformer2DModel(nn.Module):
    """MMDiT backbone for Qwen-Image (Phase 1 WIP).

    M2 submodules (``QwenTimestepProjEmbeddings``) are implemented above
    and parity-tested. Top-level ``forward`` and ``load_weights`` still
    raise ``NotImplementedError`` until M3..M7 land.
    """

    def __init__(
        self,
        model_config: "DiffusionModelConfig",
        *,
        attn_backend: str = "trtllm",
    ):
        super().__init__()
        self.model_config = model_config
        self.attn_backend = attn_backend

        self._placeholder = nn.Parameter(
            torch.zeros(
                1,
                dtype=getattr(model_config, "torch_dtype", torch.bfloat16),
            ),
            requires_grad=False,
        )

    @property
    def device(self) -> torch.device:
        return self._placeholder.device

    def load_weights(
        self, weights: Dict[str, torch.Tensor]  # noqa: ARG002
    ) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def post_load_weights(self) -> None:  # pragma: no cover - stub
        return None

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,  # noqa: ARG002
        encoder_hidden_states: Optional[torch.Tensor] = None,  # noqa: ARG002
        timestep: Optional[torch.Tensor] = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
