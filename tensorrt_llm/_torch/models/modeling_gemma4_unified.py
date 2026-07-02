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
"""Gemma 4 12B *Unified* (encoder-free multimodal) — PyTorch backend.

Gemma 4 12B ships as the family's "unified" / encoder-free member: HF
`architectures = ["Gemma4UnifiedForConditionalGeneration"]`, `model_type =
"gemma4_unified"`. Unlike the standard Gemma 4 VLMs (26B/31B) it has **no**
vision/audio encoder *towers* — raw pixel patches and audio waveform frames are
projected directly into the LM embedding space through lightweight linear
pipelines:

* **vision** (`model.vision_embedder.*` + `model.embed_vision.*`):
  `LayerNorm -> Dense(6912->3840) -> LayerNorm` + factorized 2D positional
  embedding `(mm_posemb_size, 2, mm_embed_dim)` `-> LayerNorm` `->`
  `RMSNorm(no-scale) -> Linear` into the LM hidden space.
* **audio** (`model.embed_audio.*`): `RMSNorm(no-scale) -> Linear(640->3840)`.

The text backbone is a plain dense Gemma 4 text model nested under
`config.text_config` (already fully supported by :class:`Gemma4ForCausalLM`:
per-layer head_dim 256/512, interleaved VSWA, `k_eq_v` MQA on global layers,
`layer_scalar`, final-logit softcap, tied embeddings). 12B has PLE and
KV-sharing **off** (`hidden_size_per_layer_input=0`, `num_kv_shared_layers=0`).

This module reuses the existing Gemma 4 multimodal wrapper
(:class:`Gemma4ForConditionalGeneration`) for all engine plumbing
(`post_config` / `get_sub_model_config` / `infer_max_seq_len` /
`vocab_size_padded` / `get_model_defaults`), and reuses
:class:`Gemma4MultimodalEmbedder` (audio) and :class:`Gemma4InputProcessor`
(HF `AutoProcessor` resolves to `Gemma4UnifiedProcessor`; the output dict
keys match). It overrides `__init__` / `forward` / `_get_image_features` /
`_get_audio_features` / `load_weights` to drop the encoder towers and use the
encoder-free projections instead.

TRT-LLM provides its own `gemma4_unified` config classes
(`_torch/configs/gemma4_unified.py`) and multimodal preprocessing (the vendored
section at the end of this file), used whenever the installed transformers does
not ship them natively — the full model (text + image + audio + video) runs on
the repo's pinned transformers.
"""

import copy
import json
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torchvision.transforms.v2 import functional as tvF
from transformers import PreTrainedModel
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput, PILImageResampling
from transformers.processing_utils import ImagesKwargs, Unpack, VideosKwargs
from transformers.utils import TensorType, is_torch_available
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoInput

from ...inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
)
from ...inputs.multimodal import MultimodalParams
from ..attention_backend import AttentionMetadata
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear
from .modeling_gemma4 import Gemma4ForCausalLM
from .modeling_gemma4mm import (
    Gemma4ForConditionalGeneration,
    Gemma4InputProcessor,
    Gemma4MultimodalEmbedder,
)
from .modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from .modeling_utils import ModelConfig, filter_weights, register_auto_model

# Raw HF nn.LayerNorm default eps (the unified vision LayerNorms use this, NOT
# the text rms_norm_eps of 1e-6).
_VISION_LN_EPS = 1e-5
# HF unified checkpoint nests the text backbone under this prefix.
_LANG_PREFIX = "model.language_model."


class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Encoder-free vision projection (no ViT tower).

    Mirrors HF `Gemma4UnifiedVisionEmbedder`: each merged pixel patch (raw
    `48*48*3 = 6912` channels) is normalized, densely projected to the model
    embedding dim, gets a factorized 2D positional embedding added, is
    re-normalized, then projected into the LM hidden space by the shared
    `Gemma4MultimodalEmbedder` (RMSNorm-no-scale -> Linear).
    """

    def __init__(self, vision_config, text_config, dtype=torch.bfloat16, mapping=None):
        super().__init__()
        # Dims come from vision_config; only the "* 3" (RGB channels) is a literal.
        merged_patch_size = (
            vision_config.patch_size * vision_config.pooling_kernel_size
        )  # merged pixel-patch side length
        patch_dim = (
            merged_patch_size * merged_patch_size * 3
        )  # flattened raw-pixel patch: side^2 * 3 RGB channels
        mm_embed_dim = vision_config.mm_embed_dim  # multimodal embedding width
        projector_input_dim = (
            vision_config.output_proj_dims
        )  # projector input width (HF: multimodal_hidden_size)

        self.patch_ln1 = LayerNorm(hidden_size=patch_dim, eps=_VISION_LN_EPS, dtype=dtype)
        self.patch_dense = Linear(
            in_features=patch_dim,
            out_features=mm_embed_dim,
            bias=True,
            dtype=dtype,
            mapping=mapping,
        )
        self.patch_ln2 = LayerNorm(hidden_size=mm_embed_dim, eps=_VISION_LN_EPS, dtype=dtype)
        self.pos_embedding = nn.Parameter(
            torch.zeros(vision_config.mm_posemb_size, 2, mm_embed_dim, dtype=dtype)
        )
        # Cache the [0, 1] axis index instead of rebuilding it every forward.
        self.register_buffer("axis_index", torch.arange(2), persistent=False)
        self.pos_norm = LayerNorm(hidden_size=mm_embed_dim, eps=_VISION_LN_EPS, dtype=dtype)
        # Final projection into LM hidden space (RMSNorm-no-scale -> Linear).
        self.multimodal_embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=projector_input_dim,
            text_hidden_size=text_config.hidden_size,
            eps=vision_config.rms_norm_eps,
            dtype=dtype,
            mapping=mapping,
        )

    def forward(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = pixel_values.to(self.patch_dense.weight.dtype)  # (B, P, patch_dim)
        hidden_states = self.patch_ln1(hidden_states)
        hidden_states = self.patch_dense(hidden_states)  # (B, P, mm_embed_dim)
        hidden_states = self.patch_ln2(hidden_states)
        # Factorized 2D positional embedding: pos_embedding[idx, axis] summed
        # over the two axes; padding positions (-1) contribute zero.
        clamped_position_ids = image_position_ids.clamp(min=0).long()  # (B, P, 2)
        valid = (
            (image_position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)
        )  # (B, P, 2, 1)
        position_embeds = (self.pos_embedding[clamped_position_ids, self.axis_index] * valid).sum(
            dim=-2
        )  # (B, P, mm_embed_dim)
        hidden_states = hidden_states + position_embeds
        hidden_states = self.pos_norm(hidden_states)
        hidden_states = self.multimodal_embedder(hidden_states)  # (B, P, H_text)
        return hidden_states

    def load_weights(self, embedder_weights: Dict, projector_weights: Dict):
        # A valid checkpoint has all of these, so a missing key raises (fail loud)
        # rather than being silently skipped.
        self.patch_ln1.weight.data.copy_(embedder_weights["patch_ln1.weight"])
        self.patch_ln1.bias.data.copy_(embedder_weights["patch_ln1.bias"])
        self.patch_dense.weight.data.copy_(embedder_weights["patch_dense.weight"])
        self.patch_dense.bias.data.copy_(embedder_weights["patch_dense.bias"])
        self.patch_ln2.weight.data.copy_(embedder_weights["patch_ln2.weight"])
        self.patch_ln2.bias.data.copy_(embedder_weights["patch_ln2.bias"])
        self.pos_embedding.data.copy_(embedder_weights["pos_embedding"])
        self.pos_norm.weight.data.copy_(embedder_weights["pos_norm.weight"])
        self.pos_norm.bias.data.copy_(embedder_weights["pos_norm.bias"])
        # CRITICAL: the final vision projector lives at the TOP-LEVEL checkpoint
        # key `model.embed_vision.embedding_projection.weight` (not nested under
        # `model.vision_embedder.*`). Route it explicitly.
        self.multimodal_embedder.load_weights(projector_weights)


class Gemma4UnifiedInputProcessor(Gemma4InputProcessor):
    """Input processor for Gemma 4 12B Unified.

    Identical to :class:`Gemma4InputProcessor`: HF `AutoProcessor` resolves to
    `Gemma4UnifiedProcessor` (encoder-free image patches + raw audio frames),
    whose output dict keys (`pixel_values` / `image_position_ids` /
    `input_features` / `input_features_mask`) match what the base processor
    already produces. `mm_bidirectional_blocks` auto-derives to True from
    `text_config.use_bidirectional_attention == "vision"`.

    When `AutoProcessor` cannot resolve `Gemma4UnifiedProcessor` (the base class
    then leaves `self._processor` as None), fall back to the vendored equivalent
    in `gemma4_unified_processing.py` so all modalities keep working.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # AutoProcessor may fail outright (None) or fall back to a bare tokenizer
        # when it cannot resolve Gemma4UnifiedProcessor; either way it lacks the
        # per-modality components, so install the vendored processor instead.
        if self._processor is None or not hasattr(self._processor, "image_processor"):
            self._processor = load_gemma4_unified_processor(self._model_path, self._tokenizer)
            self._image_processor = self._processor.image_processor


@register_auto_model("Gemma4UnifiedForConditionalGeneration")
@register_input_processor(
    Gemma4UnifiedInputProcessor,
    model_type="gemma4_unified",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|image|>",
            "audio": "<|audio|>",
            "video": "<|video|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        content_format=ContentFormat.OPENAI,
        interleave_placeholders=True,
    ),
)
class Gemma4UnifiedForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Gemma 4 12B Unified (encoder-free). Reuses the Gemma 4 MM wrapper for
    engine plumbing + the text core (:class:`Gemma4ForCausalLM`); replaces the
    vision/audio towers with encoder-free linear embedders."""

    def __init__(self, model_config: ModelConfig):
        config = model_config.pretrained_config
        # Skip Gemma4ForConditionalGeneration.__init__ (it builds vision/audio
        # *towers* the unified architecture does not have) and init the HF base.
        PreTrainedModel.__init__(self, config)

        # ModelConfig always has `mapping`, and Mapping always has `local_rank`.
        local_rank = model_config.mapping.local_rank
        self._device = f"cuda:{local_rank}"
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        self._top_config = config
        # HF always defines these token ids (each may be None if that modality is absent).
        self.image_token_ids = (
            torch.tensor([config.image_token_id], dtype=torch.int32, device=self._device)
            if config.image_token_id is not None
            else None
        )
        self.audio_token_ids = (
            torch.tensor([config.audio_token_id], dtype=torch.int32, device=self._device)
            if config.audio_token_id is not None
            else None
        )
        self.video_token_ids = (
            torch.tensor([config.video_token_id], dtype=torch.int32, device=self._device)
            if config.video_token_id is not None
            else None
        )

        # --- Text backbone (reused verbatim) ---
        model_config_cp = copy.deepcopy(model_config)
        self.model_config = model_config_cp
        # Mirror the parent's quant exclude_modules remap so quantized unified
        # checkpoints exclude the right text modules.
        quant_config = model_config_cp.quant_config
        if quant_config.exclude_modules:
            remapped = []
            for pattern in quant_config.exclude_modules:
                remapped.append(pattern)
                if pattern.startswith(_LANG_PREFIX):
                    # Strip only the leading prefix, not any later occurrence.
                    remapped.append("model." + pattern[len(_LANG_PREFIX) :])
            quant_config.exclude_modules = remapped
        llm_model_config = self.get_sub_model_config(model_config_cp, "text_config")
        self.llm = Gemma4ForCausalLM(llm_model_config)

        # --- Encoder-free multimodal front-end (no towers) ---
        self.vision_tower = (
            None  # kept so the weight mapper's hasattr(self, "vision_tower") returns True
        )
        self.audio_tower = None
        self.embed_vision = None
        self.embed_audio = None
        # vision_config / audio_config are optional sub-configs (HF sets them to
        # None when a modality is absent), so build each embedder only if present.
        if config.vision_config is not None:
            self.embed_vision = (
                Gemma4UnifiedVisionEmbedder(
                    config.vision_config,
                    config.text_config,
                    dtype=self.model_dtype,
                    mapping=model_config.mapping,
                )
                .eval()
                .to(self._device)
            )
        if config.audio_config is not None:
            # HF's shared multimodal embedder reads output_proj_dims for both
            # modalities; for audio it aliases audio_embed_dim (the raw frame width).
            audio_input_dim = config.audio_config.output_proj_dims
            self.embed_audio = (
                Gemma4MultimodalEmbedder(
                    mm_hidden_size=audio_input_dim,
                    text_hidden_size=config.text_config.hidden_size,
                    eps=config.audio_config.rms_norm_eps,
                    dtype=self.model_dtype,
                    mapping=model_config.mapping,
                )
                .eval()
                .to(self._device)
            )

        # Surface the text (LLM) config so KV-cache sizing + is_gemma4_hybrid
        # (global head dim differs from the per-layer head dim) read the text geometry.
        self.post_config()
        self.is_loaded = True

    # --- Encoder-free feature extractors (override the tower-based parents) ---

    def _get_image_features(self, pixel_values, image_position_ids=None, image_seq_lens=None):
        with torch.autocast(device_type="cuda", dtype=self.model_dtype):
            features = self.embed_vision(pixel_values, image_position_ids)  # (B, P, H)
        if image_position_ids is not None:
            # -1 marks a padding patch (pads a variable-size image); drop those rows.
            padding_mask = (image_position_ids == -1).all(dim=-1)  # (B, P); True = padding patch
            features = features[~padding_mask]  # keep only real patches -> (N_valid, H)
        else:
            features = features.reshape(-1, features.shape[-1])
        return features.contiguous()

    def _get_audio_features(self, audio_features, audio_features_mask=None):
        target_dtype = self.embed_audio.embedding_projection.weight.dtype
        with torch.autocast(device_type="cuda", dtype=self.model_dtype):
            features = self.embed_audio(audio_features.to(target_dtype))  # (B, T, H) or (T, H)
        if audio_features_mask is not None:
            features = features[audio_features_mask.bool()]
        else:
            features = features.reshape(-1, features.shape[-1])
        return features.contiguous()

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "image.image_position_ids",
            "video.pixel_values",
            "video.image_position_ids",
            "audio.audio_features",
            "audio.audio_features_mask",
        ]

    @staticmethod
    def _has_active_multimodal_tokens(multimodal_param: MultimodalParams) -> bool:
        """Whether a context parameter needs embeddings in this forward.

        Mirrors `Gemma4ForConditionalGeneration._has_active_multimodal_tokens`
        from #15848 (this class overrides `forward` entirely, so the parent's
        version would not apply here; deduplicate once #15848 lands on main).
        """
        runtime = multimodal_param.multimodal_runtime
        if runtime is not None and runtime.num_mm_tokens_in_chunk == 0:
            return False

        multimodal_data = multimodal_param.multimodal_data
        if multimodal_data.get("multimodal_embedding") is not None:
            return True

        payload_fields = (
            ("image", "pixel_values"),
            ("video", "pixel_values"),
            ("audio", "audio_features"),
        )
        return any(
            multimodal_data.get(modality, {}).get(field) is not None
            for modality, field in payload_fields
        )

    def _forward_multimodal_encoder(
        self, multimodal_params: List[MultimodalParams]
    ) -> torch.Tensor:
        """Run the encoder-free projectors for all uncached multimodal payloads.

        Called by `get_multimodal_embeddings`, which caches the result in
        `multimodal_data["multimodal_embedding"]` so later prefill chunks reuse
        the embedding without re-running the projectors (mirrors the pattern in
        `Gemma4ForConditionalGeneration._forward_multimodal_encoder` from #15848).
        """
        pixel_values_list, image_position_ids_list = [], []
        audio_features_list, audio_mask_list = [], []
        video_pixel_values_list, video_position_ids_list = [], []
        for multimodal_param in multimodal_params:
            image_data = multimodal_param.multimodal_data.get("image", {})
            if image_data.get("pixel_values") is not None:
                pixel_values_list.append(image_data["pixel_values"])
                if image_data.get("image_position_ids") is not None:
                    image_position_ids_list.append(image_data["image_position_ids"])
            audio_data = multimodal_param.multimodal_data.get("audio", {})
            if audio_data.get("audio_features") is not None:
                audio_features_list.append(audio_data["audio_features"])
                audio_mask_list.append(audio_data.get("audio_features_mask"))
            video_data = multimodal_param.multimodal_data.get("video", {})
            if video_data.get("pixel_values") is not None:
                video_pixel_values_list.append(video_data["pixel_values"])
                if video_data.get("image_position_ids") is not None:
                    video_position_ids_list.append(video_data["image_position_ids"])

        embeddings = []
        if pixel_values_list and self.embed_vision is not None:
            pixel_values = torch.cat(pixel_values_list)
            image_position_ids = (
                torch.cat(image_position_ids_list)
                if len(image_position_ids_list) == len(pixel_values_list)
                else None
            )
            embeddings.append(self._get_image_features(pixel_values, image_position_ids))

        if video_pixel_values_list and self.embed_vision is not None:
            video_pixel_values = torch.cat(video_pixel_values_list)
            video_position_ids = (
                torch.cat(video_position_ids_list)
                if len(video_position_ids_list) == len(video_pixel_values_list)
                else None
            )
            embeddings.append(self._get_image_features(video_pixel_values, video_position_ids))

        if audio_features_list and self.embed_audio is not None:
            per_clip = []
            for clip_index, audio_feature in enumerate(audio_features_list):
                per_clip.append(
                    self._get_audio_features(audio_feature, audio_mask_list[clip_index])
                )
            embeddings.append(torch.cat(per_clip, dim=0))

        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_params = kwargs.get("multimodal_params", [])

        # Filter to params that have active multimodal tokens in this chunk.
        # get_multimodal_embeddings caches the result after the first run, so
        # later prefill chunks reuse the embedding without re-running the projectors.
        active_multimodal_params = [
            mp for mp in multimodal_params if self._has_active_multimodal_tokens(mp)
        ]

        mm_embeds: List[torch.Tensor] = []
        all_mm_token_ids: List[torch.Tensor] = []
        mm_token_type_ids = None

        if active_multimodal_params:
            mm_embeds = get_multimodal_embeddings(
                encoder_forward_fn=self._forward_multimodal_encoder,
                multimodal_params=active_multimodal_params,
            )
            mm_embeds = find_input_mm_embeds(mm_embeds, active_multimodal_params)

            # Collect every defined multimodal token id. On cache-hit chunks the
            # raw payloads (pixel_values / audio_features) may be absent while the
            # cached embedding is used, so the ids cannot be derived from payload
            # presence; extra ids are harmless (they simply match no position).
            for token_ids in (self.image_token_ids, self.video_token_ids, self.audio_token_ids):
                if token_ids is not None:
                    all_mm_token_ids.append(token_ids)

        # Integer mm_token_type_ids (0=text,1=image,2=video,3=audio) drive the
        # inherited bidirectional-vision attention mask in Gemma4ForCausalLM.
        if mm_embeds and input_ids is not None:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
            if self.image_token_ids is not None:
                mm_token_type_ids[torch.isin(input_ids, self.image_token_ids)] = 1
            if self.video_token_ids is not None:
                mm_token_type_ids[torch.isin(input_ids, self.video_token_ids)] = 2
            if self.audio_token_ids is not None:
                mm_token_type_ids[torch.isin(input_ids, self.audio_token_ids)] = 3

        fuse_token_ids = torch.cat(all_mm_token_ids) if all_mm_token_ids else self.image_token_ids

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embeds,
            mm_token_ids=fuse_token_ids,
            **kwargs,
        )
        # 12B has PLE off (hidden_size_per_layer_input=0) -> no ple_input_ids.
        return self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            mm_token_type_ids=mm_token_type_ids,
            lora_params=kwargs.get("lora_params", None),
        )

    def load_weights(self, weights: Dict, weight_mapper):
        # Text backbone: "model.language_model.X" -> "model.X" (same as the
        # parent), then load via the reused Gemma4 text core.
        llm_weights = {
            "model." + key[len(_LANG_PREFIX) :]: value
            for key, value in weights.items()
            if key.startswith(_LANG_PREFIX)
        }
        self.llm.load_weights(llm_weights, weight_mapper)

        # Encoder-free MM front-end: strip the outer "model." from the non-text
        # keys and route to the embedders.
        stripped_weights = {
            key[len("model.") :]: value
            for key, value in weights.items()
            if key.startswith("model.") and not key.startswith(_LANG_PREFIX)
        }
        if self.embed_vision is not None:
            vision_embedder_weights = filter_weights(
                "vision_embedder", stripped_weights
            )  # patch_ln1/dense/ln2/pos_embedding/pos_norm
            projector_weights = filter_weights(
                "embed_vision", stripped_weights
            )  # embedding_projection.weight (top-level)
            self.embed_vision.load_weights(vision_embedder_weights, projector_weights)
        if self.embed_audio is not None:
            self.embed_audio.load_weights(filter_weights("embed_audio", stripped_weights))


# =============================================================================
# Vendored multimodal preprocessing.
#
# From HuggingFace transformers (Apache-2.0), models/gemma4_unified at commit
# 181beb3ba4c47098ed8cbc97ee250d1d45ae0107:
#   feature_extraction_gemma4_unified.py (Gemma4UnifiedAudioFeatureExtractor),
#   image_processing_gemma4_unified.py (Gemma4UnifiedImageProcessor + helpers),
#   video_processing_gemma4_unified.py (Gemma4UnifiedVideoProcessor),
#   processing_gemma4_unified.py (placeholder-expansion formulas).
# Used by Gemma4UnifiedInputProcessor when AutoProcessor cannot resolve the
# HF Gemma4UnifiedProcessor. The per-modality components are verbatim (relative
# imports rewritten, duplicated helpers merged, doc-only decorators dropped);
# the top-level processor reproduces the generic ProcessorMixin.__call__ flow
# explicitly, keeping the replacement formulas byte-for-byte.
# =============================================================================

# ---------------------------------------------------------------------------
# Audio: vendored from feature_extraction_gemma4_unified.py
# ---------------------------------------------------------------------------


class Gemma4UnifiedAudioFeatureExtractor(SequenceFeatureExtractor):
    """Encoder-free audio feature extractor that chunks raw waveform into frames.

    Unlike the standard Gemma4 audio feature extractor which computes mel spectrograms,
    this unified version simply chunks raw 16 kHz audio into fixed-length frames
    of `audio_samples_per_token` samples each. Each frame becomes a single audio
    soft token with the raw waveform samples as its features.

    Args:
        feature_size (`int`, *optional*, defaults to 640):
            The feature dimension of the extracted features (samples per token).
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio.
        audio_samples_per_token (`int`, *optional*, defaults to 640):
            Number of raw audio samples per output token. At 16 kHz, 640 samples = 40ms.
    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 640,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        audio_samples_per_token: int = 640,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.audio_samples_per_token = audio_samples_per_token

    def _extract_waveform_features(
        self,
        waveform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Chunk a raw waveform into fixed-length frames.

        Each frame of `audio_samples_per_token` samples becomes one audio soft token.
        The waveform is zero-padded to be evenly divisible by the frame size.

        Args:
            waveform: 1-D array of raw audio samples.

        Returns:
            features: (num_tokens, audio_samples_per_token) array of waveform frames.
            mask: (num_tokens,) boolean array, True for all valid tokens.
        """
        # Pad waveform to be evenly divisible by samples_per_token
        pad_len = (-len(waveform)) % self.audio_samples_per_token
        if pad_len:
            waveform = np.pad(waveform, (0, pad_len))

        num_tokens = len(waveform) // self.audio_samples_per_token
        features = waveform.reshape(num_tokens, self.audio_samples_per_token).astype(np.float32)

        # All tokens are valid (padding is within the last frame, not creating extra frames)
        mask = np.ones(num_tokens, dtype=bool)
        return features, mask

    def __call__(
        self,
        raw_speech,
        padding: bool | str = "longest",
        max_length: int | None = None,
        truncation: bool = True,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Chunk raw audio waveforms into fixed-length frames for the unified model.

        Args:
            raw_speech:
                The raw audio waveform(s) to process.
            padding (`str`, *optional*, defaults to `"longest"`):
                Padding strategy for batches with different lengths.
            max_length (`int`, *optional*):
                Maximum number of tokens to produce per audio.
            truncation (`bool`, *optional*, defaults to `True`):
                Whether to truncate audio above `max_length` tokens.
            return_tensors (`str`, *optional*):
                The type of tensors to return.
        """
        # Normalize input to list of 1-D arrays
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            raw_speech = [raw_speech]
        elif not isinstance(raw_speech, (list, tuple)):
            raw_speech = [np.asarray(raw_speech)]
        else:
            raw_speech = [np.asarray(s) for s in raw_speech]

        # Extract features for each waveform
        all_features = [
            {"input_features": self._extract_waveform_features(waveform)[0]}
            for waveform in raw_speech
        ]

        # Delegate padding and truncation to the parent class
        padded_inputs = self.pad(
            all_features,
            padding=padding,
            max_length=max_length,
            truncation=truncation and max_length is not None,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

        # Rename attention_mask -> input_features_mask.
        # pad() produces int32 (0/1); downstream code expects a boolean mask for indexing.
        mask = padded_inputs.pop("attention_mask")
        if is_torch_available() and isinstance(mask, torch.Tensor):
            mask = mask.bool()
        else:
            mask = np.asarray(mask, dtype=bool)
        padded_inputs["input_features_mask"] = mask

        return padded_inputs


# ---------------------------------------------------------------------------
# Vision helpers: vendored from image_processing_gemma4_unified.py (the video
# file duplicates get_aspect_ratio_preserving_size / patches_merge; one copy here)
# ---------------------------------------------------------------------------


class Gemma4UnifiedImageProcessorKwargs(ImagesKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Size of each teacher image patch in pixels (before merging).
    max_soft_tokens (`int`, *optional*):
        Maximum number of soft (vision) tokens per image after patch merging.
        Must be one of {70, 140, 280, 560, 1120}.
    pooling_kernel_size (`int`, *optional*):
        Kernel size for merging teacher patches into model patches.
    """

    patch_size: int
    max_soft_tokens: int
    pooling_kernel_size: int


class Gemma4UnifiedVideoProcessorKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Size of each image patch in pixels.
    max_soft_tokens (`int`, *optional*):
        Maximum number of soft (vision) tokens per video frame.
        Must be one of {70, 140, 280, 560, 1120}.
    pooling_kernel_size (`int`, *optional*):
        Spatial pooling kernel size applied after patchification.
    """

    patch_size: int
    max_soft_tokens: int
    pooling_kernel_size: int


_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    """
    Image is resized to preserve aspect ratio so it fits within the patch budget.
    Target dimensions are the largest that:
    1) Produce at most `max_patches` patches when patchified with `patch_size`
    2) Have height and width divisible by `pooling_kernel_size * patch_size`
    """
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    # Round down to nearest multiple of side_mult
    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    # Handle edge cases where one or both dimensions round to 0
    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. Resized height should be divisible by "
            f"`pooling_kernel_size * patch_size`={pooling_kernel_size * patch_size}."
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(
            int(math.floor(width / height)) * side_mult,
            max_side_length,
        )
    elif target_width == 0:
        target_width = side_mult
        target_height = min(
            int(math.floor(height / width)) * side_mult,
            max_side_length,
        )

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] "
            f"but this exceeds {max_patches} patches with patch_size {patch_size}"
        )

    return target_height, target_width


def convert_image_to_patches(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    """
    Convert 3D tensor image of shape (num_channels, image_height, image_width) into 2D tensor of patches of shape
    (num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(
        num_channels, num_patches_height, patch_size, num_patches_width, patch_size
    )
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


# Adopted from Siglip2 (mask -> position ids)
def pad_along_first_dim(
    image: "torch.Tensor", positions: "torch.Tensor", target_length: int
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Pad the tensor along the first dimension.
    """
    current_length = image.shape[0]
    padding_length = target_length - current_length
    if padding_length > 0:
        padding = [0, 0] * (image.ndim - 1) + [0, padding_length]
        pos_padding = (0, 0, 0, padding_length)
        image = torch.nn.functional.pad(image, padding, mode="constant", value=0)
        positions = torch.nn.functional.pad(positions, pos_padding, mode="constant", value=-1)
    return image, positions


def patches_merge(
    patches: "torch.Tensor",
    positions_xy: "torch.Tensor",
    length: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Merge kxk groups of small patches into larger patches.

    Given `L` input patches of dimension `D = patch_size^2 x 3`, merge groups of
    `kxk` spatially adjacent patches into `length` output patches of dimension
    `(k x patch_size)^2 x 3`. The spatial grouping is determined by integer-dividing
    the XY positions by `k`.

    Args:
        patches: (*, L, D) -- input patches.
        positions_xy: (*, L, 2) -- integer XY positions for each patch (-1 for padding).
        length: target number of output patches. Must satisfy L = length x k^2.

    Returns:
        merged_patches: (*, length, k^2 x D) -- merged patch features.
        merged_positions: (*, length, 2) -- new XY positions for merged patches.
    """
    patch_size = math.isqrt(patches.shape[-1] // 3)
    if patches.shape[-1] != patch_size * patch_size * 3:
        raise ValueError(
            f"Patch dimension {patches.shape[-1]} is not a valid `patch_size * patch_size * 3`"
        )

    k = math.isqrt(patches.shape[-2] // length)
    if k * k * length != patches.shape[-2]:
        raise ValueError(f"Cannot merge {patches.shape} to {length}")

    # Compute target ordering for reordering patches into kernel-grouped order.
    # This ensures patches within each kxk kernel are contiguous.
    max_x = positions_xy[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(positions_xy, k, rounding_mode="floor")
    num_patches_from_top_left = k * k * kernel_idxs[..., 0] + k * max_x * kernel_idxs[..., 1]

    position_within_kernel = torch.remainder(positions_xy, k)
    num_patches_from_top_left_of_kernel = (
        position_within_kernel[..., 0] + position_within_kernel[..., 1] * k
    )
    target_ordering = num_patches_from_top_left_of_kernel + num_patches_from_top_left

    # Reorder patches by computing the inverse permutation via argsort,
    # then gathering patches into kernel-grouped order.
    perm = target_ordering.long().argsort(dim=-1)  # inverse permutation
    # Expand perm indices to match patch feature dimension for gathering
    perm_expanded = perm.unsqueeze(-1).expand_as(patches)
    kernel_ordered_patches = patches.gather(-2, perm_expanded)

    batch_shape = patches.shape[:-2]

    # Reshape: (*, length*k*k, patch_size*patch_size*3) -> (*, length, (k*patch_size)*(k*patch_size)*3)
    kernel_ordered_patches = kernel_ordered_patches.reshape(
        *batch_shape, length, k * k, patch_size, patch_size, 3
    )
    # Rearrange (l, a*b, p, q, c) -> (l, a*p, b*q, c)
    kernel_ordered_patches = kernel_ordered_patches.reshape(
        *batch_shape, length, k, k, patch_size, patch_size, 3
    )
    kernel_ordered_patches = kernel_ordered_patches.permute(
        *range(len(batch_shape)), -6, -5, -3, -4, -2, -1
    )  # (..., l, k, p, k, q, c)
    merged_patches = kernel_ordered_patches.reshape(
        *batch_shape, length, k * patch_size * k * patch_size * 3
    )

    # Compute new positions for merged patches
    perm_pos = perm.unsqueeze(-1).expand_as(positions_xy)
    kernel_ordered_positions = positions_xy.float().gather(-2, perm_pos.long())

    # Handle padding: preserve -1 positions
    padding = (positions_xy == -1).all(dim=-1, keepdim=True)  # (..., L, 1)
    kernel_ordered_positions = (
        kernel_ordered_positions * (~padding).float() + positions_xy.float() * padding.float()
    )

    # Reshape positions and take min within each kernel to get the merged position
    kernel_ordered_positions = kernel_ordered_positions.reshape(*batch_shape, length, k * k, 2)
    new_positions = torch.div(kernel_ordered_positions, k, rounding_mode="floor")
    # For each merged patch, take the minimum position across the kernel
    new_positions = new_positions.min(dim=-2)[0].to(torch.long)

    return merged_patches, new_positions


class Gemma4UnifiedImageProcessor(TorchvisionBackend):
    """Constructs a Gemma4 unified image processor."""

    resample = PILImageResampling.BICUBIC
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]
    size = None
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = False
    patch_size = 16
    max_soft_tokens = 280
    pooling_kernel_size = 3
    valid_kwargs = Gemma4UnifiedImageProcessorKwargs
    model_input_names = ["pixel_values", "image_position_ids", "num_soft_tokens_per_image"]

    def __init__(self, **kwargs: Unpack[Gemma4UnifiedImageProcessorKwargs]):
        super().__init__(**kwargs)

        if self.max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {self.max_soft_tokens}."
            )

    def _validate_preprocess_kwargs(self, **kwargs):
        # Gemma4Unified uses aspect_ratio_preserving_resize driven by patch_size,
        # max_soft_tokens, and pooling_kernel_size -- not the standard `size`
        # parameter. Temporarily disable do_resize so the base validation
        # doesn't require `size` to be set.
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(**kwargs)

    def aspect_ratio_preserving_resize(
        self,
        image: torch.Tensor,
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
        resample: tvF.InterpolationMode,
    ) -> torch.Tensor:
        height, width = image.shape[-2], image.shape[-1]
        target_height, target_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size,
            max_patches=max_patches,
            pooling_kernel_size=pooling_kernel_size,
        )

        if target_height == height and target_width == width:
            return image

        return tvF.resize(
            image,
            size=[target_height, target_width],
            interpolation=resample,
            antialias=True,
        )

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Gemma4UnifiedImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int | None = None,
        max_soft_tokens: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}."
            )

        # Compute max_patches from max_soft_tokens and pooling_kernel_size
        max_patches = max_soft_tokens * pooling_kernel_size**2

        # Process each image individually: resize, rescale/normalize, patchify, pad.
        # Images have different aspect ratios and thus different resized dimensions,
        # so patchification and padding must happen per-image before stacking.
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []

        for image in images:
            # Step 1: Aspect-ratio-preserving resize
            if do_resize:
                image = self.aspect_ratio_preserving_resize(
                    image=image,
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                    resample=resample,
                )

            # Step 2: Rescale pixel values (typically to [0, 1]) and optionally identity normalize
            image = self.rescale_and_normalize(
                image, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            # Step 3: Patchify into teacher-size patches (16px)
            # (num_channels, height, width) -> (num_teacher_patches, patch_size^2*3)
            teacher_patches = convert_image_to_patches(image, patch_size)
            patch_height = image.shape[-2] // patch_size
            patch_width = image.shape[-1] // patch_size

            # Step 4: Compute teacher-level position IDs
            device = image.device
            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=device),
                torch.arange(patch_height, device=device),
                indexing="xy",
            )
            teacher_positions = torch.stack(patch_grid, dim=-1).reshape(teacher_patches.shape[0], 2)

            # Step 5: Merge kxk teacher patches into model patches via patches_merge
            # (num_teacher_patches, 768) -> (num_model_patches, 6912)
            num_model_patches = teacher_patches.shape[0] // (pooling_kernel_size**2)
            merged_patches, merged_positions = patches_merge(
                teacher_patches.unsqueeze(0),  # add batch dim for patches_merge
                teacher_positions.unsqueeze(0),
                num_model_patches,
            )
            merged_patches = merged_patches.squeeze(0)  # remove batch dim
            merged_positions = merged_positions.squeeze(0)
            num_soft_tokens_per_image.append(merged_patches.shape[0])

            # Step 6: Pad merged patches and positions to max_soft_tokens
            merged_patches, merged_positions = pad_along_first_dim(
                merged_patches, merged_positions, max_soft_tokens
            )
            pixel_values.append(merged_patches)
            position_ids.append(merged_positions)

        # Stack into batch tensors
        pixel_values = torch.stack(
            pixel_values, dim=0
        )  # (batch, max_soft_tokens, model_patch_size^2*3)
        position_ids = torch.stack(position_ids, dim=0)  # (batch, max_soft_tokens, 2)

        data = {
            "pixel_values": pixel_values,
            "image_position_ids": position_ids,
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


# ---------------------------------------------------------------------------
# Video: vendored from video_processing_gemma4_unified.py
# ---------------------------------------------------------------------------


def convert_video_to_patches(video: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    """
    Convert 4D tensor video of shape (num_frames, num_channels, height, width) into 3D tensor of patches of shape
    (num_frames, num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    num_frames, num_channels, height, width = video.shape
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    patched_video = video.reshape(
        num_frames, num_channels, num_patches_height, patch_size, num_patches_width, patch_size
    )
    patched_video = patched_video.permute(0, 2, 4, 3, 5, 1)
    patched_video = patched_video.reshape(num_frames, num_patches_height * num_patches_width, -1)
    return patched_video


def pad_to_max_patches(
    video: "torch.Tensor", positions: "torch.Tensor", target_length: int
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Pad the video along to max number of patches
    """
    current_length = video.shape[1]
    padding_length = target_length - current_length
    if padding_length > 0:
        padding = [0, 0, 0, padding_length, 0, 0]
        pos_padding = (0, 0, 0, padding_length, 0, 0)
        video = torch.nn.functional.pad(video, padding, mode="constant", value=0)
        positions = torch.nn.functional.pad(positions, pos_padding, mode="constant", value=-1)
    return video, positions


class Gemma4UnifiedVideoProcessor(BaseVideoProcessor):
    """Constructs a Gemma4Unified video processor that samples frames from videos
    for use with the Gemma4Unified model."""

    resample = PILImageResampling.BICUBIC
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]
    size = None
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    num_frames = 32
    do_sample_frames = True
    patch_size = 16
    max_soft_tokens = 70
    pooling_kernel_size = 3
    valid_kwargs = Gemma4UnifiedVideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_position_ids"]

    def __init__(self, **kwargs: Unpack[Gemma4UnifiedVideoProcessorKwargs]):
        super().__init__(**kwargs)

        if self.max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {self.max_soft_tokens}."
            )

    def _validate_preprocess_kwargs(self, **kwargs):
        # Gemma4Unified uses aspect_ratio_preserving_resize driven by patch_size,
        # max_soft_tokens, and pooling_kernel_size -- not the standard `size`
        # parameter. Temporarily disable do_resize so the base validation
        # doesn't require `size` to be set.
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(**kwargs)

    def aspect_ratio_preserving_resize(
        self,
        video: torch.Tensor,
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
        resample: tvF.InterpolationMode,
    ) -> torch.Tensor:
        height, width = video.shape[-2], video.shape[-1]
        target_height, target_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size,
            max_patches=max_patches,
            pooling_kernel_size=pooling_kernel_size,
        )

        if target_height == height and target_width == width:
            return video

        return tvF.resize(
            video,
            size=[target_height, target_width],
            interpolation=resample,
            antialias=True,
        )

    def preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[Gemma4UnifiedVideoProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(videos, **kwargs)

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_resize: bool,
        resample: "tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int | None = None,
        max_soft_tokens: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}."
            )

        # Compute max_patches from max_soft_tokens and pooling_kernel_size
        max_patches = max_soft_tokens * pooling_kernel_size**2

        # Process each image individually: resize, rescale/normalize, patchify, pad.
        # Images have different aspect ratios and thus different resized dimensions,
        # so patchification and padding must happen per-image before stacking.
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_video = []

        for video in videos:
            if do_resize:
                # Step 1: Aspect-ratio-preserving resize
                video = self.aspect_ratio_preserving_resize(
                    video=video,
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                    resample=resample,
                )

            # Step 2: Rescale pixel values (typically to [0, 1]) and optionally identity normalize
            video = self.rescale_and_normalize(
                video, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            # Step 3: Patchify into teacher-size patches (16px)
            num_frames = video.shape[0]
            patches = convert_video_to_patches(video, patch_size)
            patch_height = video.shape[-2] // patch_size
            patch_width = video.shape[-1] // patch_size

            # Step 4: Compute teacher-level position IDs
            device = video.device
            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=device),
                torch.arange(patch_height, device=device),
                indexing="xy",
            )
            stacked_grid = torch.stack(patch_grid, dim=-1)
            teacher_positions = stacked_grid.reshape(patches.shape[1], 2)
            teacher_positions = teacher_positions[None, ...].repeat(num_frames, 1, 1)

            # Step 5: Merge kxk teacher patches into model patches via patches_merge
            # (num_frames, num_teacher_patches, 768) -> (num_frames, num_model_patches, 6912)
            num_model_patches = patches.shape[1] // (pooling_kernel_size**2)
            merged_patches, merged_positions = patches_merge(
                patches, teacher_positions, num_model_patches
            )
            num_soft_tokens_per_video.append(merged_patches.shape[1])

            # Step 6: Pad merged patches and positions to max_soft_tokens
            merged_patches, merged_positions = pad_to_max_patches(
                merged_patches, merged_positions, max_soft_tokens
            )
            pixel_values.append(merged_patches)
            position_ids.append(merged_positions)

        # Stack into batch tensors
        pixel_values = torch.stack(pixel_values, dim=0)
        position_ids = torch.stack(position_ids, dim=0)

        data = {
            "pixel_values_videos": pixel_values,
            "video_position_ids": position_ids,
            "num_soft_tokens_per_video": num_soft_tokens_per_video,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


# ---------------------------------------------------------------------------
# Top-level processor shim: a plain class reproducing the generic
# ProcessorMixin.__call__ flow (prepare_inputs_layout / _process_images /
# _process_audio / get_text_with_replacements) explicitly, with the upstream
# per-modality replacement formulas (replace_image_token / replace_audio_token)
# kept byte-for-byte.
# ---------------------------------------------------------------------------


class Gemma4UnifiedProcessorShim:
    """Drop-in stand-in for HF `Gemma4UnifiedProcessor`.

    Supports the call pattern TRT-LLM's Gemma4 input processor uses:
    `processor(text=..., images=..., audio=..., return_tensors="pt")` producing
    `input_ids` / `pixel_values` / `image_position_ids` / `input_features` /
    `input_features_mask`, plus a `video_processor` attribute for the separate
    video path.
    """

    def __init__(  # nosec B107 - multimodal delimiter tokens, not passwords
        self,
        feature_extractor,
        image_processor,
        tokenizer,
        video_processor,
        image_seq_length: int = 280,
        audio_seq_length: int = 750,
        audio_ms_per_token: int = 40,
        image_token: str = "<|image|>",
        boi_token: str = "<|image>",
        eoi_token: str = "<image|>",
        audio_token: str = "<|audio|>",
        boa_token: str = "<|audio>",
        eoa_token: str = "<audio|>",
        **kwargs,
    ):
        self.feature_extractor = feature_extractor
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.video_processor = video_processor

        self.image_seq_length = image_seq_length
        self.audio_seq_length = audio_seq_length
        self.audio_ms_per_token = audio_ms_per_token

        self.image_token = image_token
        self.boi_token = boi_token
        self.eoi_token = eoi_token
        self.audio_token = audio_token
        self.boa_token = boa_token
        self.eoa_token = eoa_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

        # Upstream Gemma4UnifiedProcessor.__init__ registers the video token the
        # same way (the checkpoint tokenizer does not declare it as an attribute).
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|video|>"]})
        self.video_token = "<|video|>"  # nosec B105 - multimodal delimiter token, not a password
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

    # Upstream replace_image_token, byte-for-byte.
    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        num_soft_tokens = image_inputs["num_soft_tokens_per_image"][image_idx]
        return f"{self.boi_token}{self.image_token * num_soft_tokens}{self.eoi_token}"

    # Upstream replace_audio_token, byte-for-byte.
    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        """Replace the audio placeholder with the correct number of audio tokens.

        Unlike standard Gemma4 which has a conformer audio encoder with two stride-2
        convolution blocks (reducing token count ~4x), the unified model projects raw
        waveform frames directly through RMSNorm -> Linear with **no downsampling**.
        So the number of output soft tokens equals the number of valid input frames.
        """
        mask = audio_inputs["input_features_mask"][audio_idx]
        return f"{self.boa_token}{self.audio_token * int(mask.sum())}{self.eoa_token}"

    def __call__(
        self,
        text=None,
        images=None,
        audio=None,
        videos=None,
        return_tensors: str | TensorType | None = "pt",
        do_rescale: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        if videos is not None:
            # TRT-LLM expands video placeholders itself and calls
            # `self.video_processor` directly; mirror upstream by not accepting
            # videos through the main entry point here.
            raise ValueError(
                "Gemma4UnifiedProcessorShim does not accept `videos`; call "
                "`.video_processor(videos=...)` directly."
            )
        if isinstance(text, str):
            text = [text]

        # Mirrors ProcessorMixin.__call__: run each modality's component, then
        # replace each placeholder occurrence in-order with its expanded string.
        data = {}
        images_replacements = []
        audio_replacements = []

        if images is not None:
            images_kwargs = {}
            if do_rescale is not None:
                images_kwargs["do_rescale"] = do_rescale
            processed_images = self.image_processor(
                images, return_tensors=return_tensors, **images_kwargs
            )
            for idx in range(len(processed_images["num_soft_tokens_per_image"])):
                images_replacements.append(
                    self.replace_image_token(processed_images, image_idx=idx)
                )
            data["pixel_values"] = processed_images["pixel_values"]
            data["image_position_ids"] = processed_images["image_position_ids"]

        if audio is not None:
            processed_audio = self.feature_extractor(audio, return_tensors=return_tensors)
            for idx in range(len(processed_audio["input_features_mask"])):
                audio_replacements.append(self.replace_audio_token(processed_audio, audio_idx=idx))
            data["input_features"] = processed_audio["input_features"]
            data["input_features_mask"] = processed_audio["input_features_mask"]

        # In-order placeholder replacement (mirrors get_text_with_replacements):
        # the i-th occurrence of a modality's placeholder consumes the i-th
        # replacement string for that modality.
        token_groups = []
        if images_replacements:
            token_groups.append(f"(?P<image>{re.escape(self.image_token)})")
        if audio_replacements:
            token_groups.append(f"(?P<audio>{re.escape(self.audio_token)})")
        if token_groups and text is not None:
            regex_special_mm_tokens = "|".join(token_groups)
            replacements_iters = {
                "image": iter(images_replacements),
                "audio": iter(audio_replacements),
            }
            text = [
                re.sub(
                    regex_special_mm_tokens,
                    lambda m: next(replacements_iters[m.lastgroup]),
                    sample,
                )
                for sample in text
            ]

        if text is not None:
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, padding=True)
            data["input_ids"] = text_inputs["input_ids"]

        return BatchFeature(data=data)


def load_gemma4_unified_processor(model_path: str, tokenizer) -> Gemma4UnifiedProcessorShim:
    """Build the vendored Gemma4 unified processor from a checkpoint directory.

    Reads the component parameters from `processor_config.json` and the
    multimodal special-token strings from `tokenizer_config.json` -- the same
    files `AutoProcessor.from_pretrained` consumes.
    """
    with open(os.path.join(model_path, "processor_config.json"), encoding="utf-8") as f:
        processor_config = json.load(f)

    def _component_kwargs(section: str) -> dict:
        section_config = dict(processor_config.get(section, {}))
        # Drop the HF class-name tags; the classes are fixed here.
        for type_key in (
            "image_processor_type",
            "feature_extractor_type",
            "video_processor_type",
            "processor_class",
        ):
            section_config.pop(type_key, None)
        return section_config

    image_processor = Gemma4UnifiedImageProcessor(**_component_kwargs("image_processor"))
    feature_extractor = Gemma4UnifiedAudioFeatureExtractor(**_component_kwargs("feature_extractor"))
    video_processor = Gemma4UnifiedVideoProcessor(**_component_kwargs("video_processor"))

    # TRT-LLM hands over its TransformersTokenizer wrapper; the HF methods the
    # shim calls (add_special_tokens, convert_tokens_to_ids) live on the
    # underlying tokenizer.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    with open(os.path.join(model_path, "tokenizer_config.json"), encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    token_kwargs = {
        key: tokenizer_config[key]
        for key in (
            "image_token",
            "boi_token",
            "eoi_token",
            "audio_token",
            "boa_token",
            "eoa_token",
        )
        if tokenizer_config.get(key) is not None
    }

    return Gemma4UnifiedProcessorShim(
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        image_seq_length=processor_config.get("image_seq_length", 280),
        audio_seq_length=processor_config.get("audio_seq_length", 750),
        audio_ms_per_token=processor_config.get("audio_ms_per_token", 40),
        **token_kwargs,
    )
