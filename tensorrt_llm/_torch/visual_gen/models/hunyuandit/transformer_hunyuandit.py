# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HunyuanDiT 2D transformer wrapper.

Wraps the diffusers ``HunyuanDiT2DModel`` so it fits the TensorRT-LLM
VisualGen ``BasePipeline`` weight-loading contract:

  * ``_init_transformer`` (called in ``__init__``) builds the model from
    config alone (no weights) using :class:`HunyuanDiT2DModelWrapper`.
  * ``load_weights`` is called later with a flat ``{name: tensor}`` dict
    from the safetensors checkpoint; this method delegates to ``load_state_dict``.
  * ``forward`` delegates to the underlying diffusers transformer with the
    same kwargs the denoising loop passes (``hidden_states``, ``timestep``,
    ``encoder_hidden_states``, ``text_embedding_mask``,
    ``encoder_hidden_states_t5``, ``text_embedding_mask_t5``).

All non-transformer components (VAE, text encoders, scheduler) are loaded
in ``HunyuanDiTPipeline.load_standard_components``.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from tensorrt_llm.logger import logger


class HunyuanDiT2DModelWrapper(nn.Module):
    """Thin TRT-LLM wrapper around diffusers ``HunyuanDiT2DModel``.

    Args:
        model_config: ``DiffusionModelConfig`` instance from the pipeline.
        **transformer_kwargs: Config kwargs forwarded to ``HunyuanDiT2DModel``
            (e.g. ``num_attention_heads``, ``num_layers``, …).  All keyword
            args have defaults matching the published HunyuanDiT-v1.2 model so
            the wrapper works even when ``pretrained_config`` is sparse.
    """

    # Published HunyuanDiT-v1.2 config defaults
    _DEFAULTS: Dict[str, Any] = {
        "num_attention_heads": 16,
        "attention_head_dim": 88,
        "in_channels": 4,
        "patch_size": 2,
        "activation_fn": "gelu-approximate",
        "num_layers": 40,
        "use_linear_projection": False,
        "cross_attention_dim": 1024,
        "cross_attention_dim_t5": 2048,
        "pooled_projection_dim": 1024,
        "text_len": 77,
        "text_len_t5": 256,
        "norm_type": "ada_norm_continous",
        "sample_size": 128,
    }

    def __init__(self, model_config, **transformer_kwargs):
        super().__init__()
        self.model_config = model_config

        # Merge defaults → pretrained_config → caller overrides
        cfg: Dict[str, Any] = dict(self._DEFAULTS)
        pretrained = getattr(model_config, "pretrained_config", None)
        if pretrained is not None:
            src = pretrained if isinstance(pretrained, dict) else vars(pretrained)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]
        cfg.update(transformer_kwargs)

        try:
            from diffusers.models import HunyuanDiT2DModel
        except ImportError as exc:
            raise ImportError(
                "HunyuanDiT requires diffusers >= 0.26 "
                "(`pip install -U diffusers`)."
            ) from exc

        logger.info(
            "Building HunyuanDiT2DModel: %d layers, %d heads, head_dim=%d",
            cfg["num_layers"],
            cfg["num_attention_heads"],
            cfg["attention_head_dim"],
        )
        self.transformer = HunyuanDiT2DModel(**cfg)

        # Remember latent channel count so the pipeline can read it.
        self.in_channels = cfg["in_channels"]

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Populate transformer parameters from a flat state-dict.

        ``weights`` is provided by the TRT-LLM ``WeightLoader`` and contains
        the raw tensors from the checkpoint's ``transformer/`` safetensors
        shards.  We use ``strict=False`` so missing or extra keys (e.g. from
        a newer / older checkpoint version) do not abort loading.
        """
        result = self.transformer.load_state_dict(weights, strict=False)
        if result.missing_keys:
            logger.warning(
                "HunyuanDiT: %d missing keys in state dict "
                "(first 10: %s)",
                len(result.missing_keys),
                result.missing_keys[:10],
            )
        if result.unexpected_keys:
            logger.warning(
                "HunyuanDiT: %d unexpected keys in state dict "
                "(first 10: %s)",
                len(result.unexpected_keys),
                result.unexpected_keys[:10],
            )

    def to_inference_dtype(self):
        dtype = getattr(self.model_config, "torch_dtype", torch.bfloat16)
        self.transformer.to(dtype)
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        text_embedding_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_t5: Optional[torch.Tensor] = None,
        text_embedding_mask_t5: Optional[torch.Tensor] = None,
        image_meta_size: Optional[torch.Tensor] = None,
        style: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            text_embedding_mask=text_embedding_mask,
            encoder_hidden_states_t5=encoder_hidden_states_t5,
            text_embedding_mask_t5=text_embedding_mask_t5,
            image_meta_size=image_meta_size,
            style=style,
            image_rotary_emb=image_rotary_emb,
            return_dict=return_dict,
        )
