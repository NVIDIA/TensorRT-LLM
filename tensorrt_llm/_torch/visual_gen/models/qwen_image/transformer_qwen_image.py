# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image MMDiT transformer (Phase 0 stub).

Phase 0 only registers the class and error paths. The actual transformer
blocks (MMDiT double/single blocks, joint text-image attention, MSRoPE,
adaLN modulation, final norm+proj) will be filled in by the Phase 1 PR
against NVIDIA/TensorRT-LLM:main.

Reference: diffusers.models.transformers.transformer_qwenimage
"""

from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch import nn

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


_NOT_IMPLEMENTED_MSG = (
    "Qwen-Image transformer is not yet implemented in TensorRT-LLM "
    "(Phase 0 scaffolding only). See the visual_gen models/qwen_image/ "
    "module docstring for the Phase 1 port plan."
)


class QwenImageTransformer2DModel(nn.Module):
    """Stub MMDiT backbone for Qwen-Image.

    Matches the ``nn.Module`` shape of :class:`FluxTransformer2DModel` so
    that Phase 1 can fill in the parameter set and forward pass without
    changing callers in :class:`QwenImagePipeline`.
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

        # A single trainable parameter so that ``.device`` / ``.dtype``
        # queries on the transformer do not fail when the pipeline is
        # introspected before Phase 1 lands.
        self._placeholder = nn.Parameter(
            torch.zeros(
                1,
                dtype=getattr(model_config, "torch_dtype", torch.bfloat16),
            ),
            requires_grad=False,
        )

    # ------------------------------------------------------------------
    # Public interface (all Phase 0 stubs).
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._placeholder.device

    def load_weights(
        self, weights: Dict[str, torch.Tensor]  # noqa: ARG002
    ) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def post_load_weights(self) -> None:  # pragma: no cover - stub
        # No-op: nothing to post-process until Phase 1.
        return None

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,  # noqa: ARG002
        encoder_hidden_states: Optional[torch.Tensor] = None,  # noqa: ARG002
        timestep: Optional[torch.Tensor] = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
