# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``KimiK3ForConditionalGeneration`` of the Kimi K3 Rubin package: the VL wrapper over the
package's text model.

A copy of main's ``models/modeling_kimi_k3_vl.py`` ``KimiK3ForConditionalGeneration`` together
with the ``KimiK25ForConditionalGeneration`` members it inherits there
(``models/modeling_kimi_k25.py``), so that the wrapper builds the package's
``text_model.KimiLinearForCausalLM`` instead of the built-in one. The Rubin agentic branch did
not change the wrapper. The vision tower, its input processor and the K2.5 helpers stay main's.

Unlike main's class it carries no ``@register_vision_encoder`` / ``@register_auto_model`` /
``@register_input_processor`` decorators: the modeling_v2 targets subclass it, and
``modeling_v2=off`` keeps the built-in, registered class.
"""

import copy
from typing import Any, List, Literal, Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.modeling_kimi_k3_vl import KimiK3VisionModel
from tensorrt_llm._torch.models.modeling_kimi_k25 import (
    _MEDIA_PLACEHOLDER_TOKEN_ID,
    DISAGG,
    _has_meta_tensors,
)
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from tensorrt_llm._torch.models.modeling_utils import MetaInitException, filter_weights
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger

from .text_model import KimiLinearForCausalLM


class KimiK3ForConditionalGeneration(PreTrainedModel):
    """Kimi K3 vision-language model: MoonViT3d + the package's KimiLinear text backbone.

    Same members as main's ``KimiK3ForConditionalGeneration``: its ``__init__`` and the K2.5
    wrapper's spec-dec / weight-loading property forwarding, :meth:`forward` and
    :meth:`load_weights` (which builds the tower through ``_VISION_MODEL_CLS``).
    """

    _LANG_PREFIX = "language_model."
    _VISION_MODEL_CLS = KimiK3VisionModel
    mamba_metadata_cls = KimiLinearForCausalLM.mamba_metadata_cls

    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["V2"]:
        """Prefer KV cache manager V2 (as Kimi K2.5)."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["PYTHON"]:
        """Default to the Python (v2) KV-cache transceiver (as Kimi K2.5).

        Only adopted when the user leaves
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
        # PreTrainedModel.__init__ resolves config._attn_implementation and
        # rejects the default "sdpa" unless the class declares support. This
        # wrapper never runs HF attention itself (TRT-LLM attention is wired
        # inside the sub-modules), so declare support to keep HF init happy.
        self._supports_sdpa = True
        PreTrainedModel.__init__(self, config)

        # Re-entry guard: the tail of this __init__ repoints
        # model_config.pretrained_config at the text_config and remaps the quant
        # exclude list, so running the body twice on the same instance would
        # fail the text_config assert below. If the text backbone already
        # exists this is a re-init on a built instance: no-op.
        if hasattr(self, "llm"):
            return

        self.model_config = model_config
        self._vlm_pretrained_config = config

        # --- Vision encoder (deferred under MetaInitMode, recreated in
        # load_weights) ---
        self.mm_encoder = None
        if not DISAGG:
            try:
                mm_encoder = self._VISION_MODEL_CLS(model_config)
                if _has_meta_tensors(mm_encoder):
                    logger.info("Vision encoder deferred to load_weights() (MetaInitMode active)")
                else:
                    self.mm_encoder = mm_encoder
            except MetaInitException:
                logger.info("Vision encoder deferred to load_weights() (MetaInitMode active)")

        text_model_config = copy.copy(model_config)
        assert hasattr(config, "text_config"), "Kimi K3 config must have text_config"
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

        self.llm = KimiLinearForCausalLM(text_model_config)

        self._media_placeholder_token_id = getattr(
            config, "media_placeholder_token_id", _MEDIA_PLACEHOLDER_TOKEN_ID
        )
        # Backing storage for the ``mm_token_ids`` property below — built once
        # here so the executor doesn't re-allocate per step.
        self._mm_token_ids = torch.tensor([self._media_placeholder_token_id], dtype=torch.int32)

        # Point model_config at the text_config so the executor reads generation
        # params (eos_token_id, ...) from the text backbone, matching K2.5.
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
            if DISAGG:
                raise NotImplementedError("Disaggregated inference not yet supported for K2.5.")
            mm_ctx_params = multimodal_params[: attn_metadata.num_contexts]
            mm_embeds = get_multimodal_embeddings(
                encoder_forward_fn=self.mm_encoder.forward,
                multimodal_params=mm_ctx_params,
            )
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_ctx_params)

            # ``runners.prepare_multimodal_indices`` sees Kimi's in-vocab
            # placeholder via ``self.mm_token_ids`` and emits indices that match
            # ``find_input_mm_embeds``'s active-chunk slice; a chunked-prefill /
            # KV-reuse mismatch surfaces as a row-count error inside
            # ``fuse_input_embeds`` itself.
            if len(mm_embeds) > 0:
                mm_token_ids = self.mm_token_ids.to(input_ids.device, dtype=input_ids.dtype)
        input_ids, inputs_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            mm_token_ids=mm_token_ids,
            mm_token_indices=kwargs.get("mm_token_indices"),
            text_token_indices=kwargs.get("text_token_indices"),
        )

        return self.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            **kwargs,
        )


__all__ = ["KimiK3ForConditionalGeneration"]
