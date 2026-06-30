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
``architectures = ["Gemma4UnifiedForConditionalGeneration"]``, ``model_type =
"gemma4_unified"``. Unlike the standard Gemma 4 VLMs (26B/31B) it has **no**
vision/audio encoder *towers* — raw pixel patches and audio waveform frames are
projected directly into the LM embedding space through lightweight linear
pipelines:

* **vision** (``model.vision_embedder.*`` + ``model.embed_vision.*``):
  ``LayerNorm -> Dense(6912->3840) -> LayerNorm`` + factorized 2D positional
  embedding ``(mm_posemb_size, 2, mm_embed_dim)`` ``-> LayerNorm`` ``->``
  ``RMSNorm(no-scale) -> Linear`` into the LM hidden space.
* **audio** (``model.embed_audio.*``): ``RMSNorm(no-scale) -> Linear(640->3840)``.

The text backbone is a plain dense Gemma 4 text model nested under
``config.text_config`` (already fully supported by :class:`Gemma4ForCausalLM`:
per-layer head_dim 256/512, interleaved VSWA, ``k_eq_v`` MQA on global layers,
``layer_scalar``, final-logit softcap, tied embeddings). 12B has PLE and
KV-sharing **off** (``hidden_size_per_layer_input=0``, ``num_kv_shared_layers=0``).

This module reuses the existing Gemma 4 multimodal wrapper
(:class:`Gemma4ForConditionalGeneration`) for all engine plumbing
(``post_config`` / ``get_sub_model_config`` / ``infer_max_seq_len`` /
``vocab_size_padded`` / ``get_model_defaults``), and reuses
:class:`Gemma4MultimodalEmbedder` (audio) and :class:`Gemma4InputProcessor`
(HF ``AutoProcessor`` resolves to ``Gemma4UnifiedProcessor``; the output dict
keys match). It overrides ``__init__`` / ``forward`` / ``_get_image_features`` /
``_get_audio_features`` / ``load_weights`` to drop the encoder towers and use the
encoder-free projections instead.

Note: this file does NOT pin or bump transformers. The new ``gemma4_unified``
config class requires a recent transformers (>=5.10) — that is a user/runtime
install, never a TRT-LLM repo change.
"""

import copy
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from ...inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
)
from ..attention_backend import AttentionMetadata
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear
from .modeling_gemma4 import Gemma4ForCausalLM
from .modeling_gemma4mm import (
    Gemma4ForConditionalGeneration,
    Gemma4InputProcessor,
    Gemma4MultimodalEmbedder,
)
from .modeling_multimodal_utils import find_input_mm_embeds, fuse_input_embeds
from .modeling_utils import ModelConfig, filter_weights, register_auto_model

# Raw HF nn.LayerNorm default eps (the unified vision LayerNorms use this, NOT
# the text rms_norm_eps of 1e-6).
_VISION_LN_EPS = 1e-5
# HF unified checkpoint nests the text backbone under this prefix.
_LANG_PREFIX = "model.language_model."


class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Encoder-free vision projection (no ViT tower).

    Mirrors HF ``Gemma4UnifiedVisionEmbedder``: each merged pixel patch (raw
    ``48*48*3 = 6912`` channels) is normalized, densely projected to the model
    embedding dim, gets a factorized 2D positional embedding added, is
    re-normalized, then projected into the LM hidden space by the shared
    ``Gemma4MultimodalEmbedder`` (RMSNorm-no-scale -> Linear).
    """

    def __init__(self, vision_config, text_config, dtype=torch.bfloat16, mapping=None):
        super().__init__()
        model_patch = vision_config.patch_size * vision_config.pooling_kernel_size  # 16*3 = 48
        patch_dim = model_patch * model_patch * 3  # 48^2 * 3 = 6912
        mm_embed_dim = vision_config.mm_embed_dim  # 3840
        proj_in = getattr(vision_config, "output_proj_dims", mm_embed_dim)  # 3840

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
        self.pos_norm = LayerNorm(hidden_size=mm_embed_dim, eps=_VISION_LN_EPS, dtype=dtype)
        # Final projection into LM hidden space (RMSNorm-no-scale -> Linear).
        self.multimodal_embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=proj_in,
            text_hidden_size=text_config.hidden_size,
            eps=vision_config.rms_norm_eps,
            dtype=dtype,
            mapping=mapping,
        )

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> torch.Tensor:
        h = pixel_values.to(self.patch_dense.weight.dtype)  # (B, P, 6912)
        h = self.patch_ln1(h)
        h = self.patch_dense(h)  # (B, P, 3840)
        h = self.patch_ln2(h)
        # Factorized 2D positional embedding: pos_embedding[idx, axis] summed
        # over the two axes; padding positions (-1) contribute zero.
        clamped = image_position_ids.clamp(min=0).long()  # (B, P, 2)
        valid = (image_position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)  # (B,P,2,1)
        axes = torch.arange(2, device=clamped.device)
        pos = (self.pos_embedding[clamped, axes] * valid).sum(dim=-2)  # (B, P, 3840)
        h = h + pos
        h = self.pos_norm(h)
        h = self.multimodal_embedder(h)  # (B, P, H_text)
        return h

    def load_weights(self, embedder_weights: Dict, proj_weights: Dict):
        def _cp(param, key):
            w = embedder_weights.get(key)
            if w is not None:
                param.data.copy_(w)

        _cp(self.patch_ln1.weight, "patch_ln1.weight")
        _cp(self.patch_ln1.bias, "patch_ln1.bias")
        _cp(self.patch_dense.weight, "patch_dense.weight")
        _cp(self.patch_dense.bias, "patch_dense.bias")
        _cp(self.patch_ln2.weight, "patch_ln2.weight")
        _cp(self.patch_ln2.bias, "patch_ln2.bias")
        _cp(self.pos_embedding, "pos_embedding")
        _cp(self.pos_norm.weight, "pos_norm.weight")
        _cp(self.pos_norm.bias, "pos_norm.bias")
        # CRITICAL: the final vision projector lives at the TOP-LEVEL checkpoint
        # key ``model.embed_vision.embedding_projection.weight`` (not nested under
        # ``model.vision_embedder.*``). Route it explicitly.
        self.multimodal_embedder.load_weights(proj_weights)


class Gemma4UnifiedInputProcessor(Gemma4InputProcessor):
    """Input processor for Gemma 4 12B Unified.

    Identical to :class:`Gemma4InputProcessor`: HF ``AutoProcessor`` resolves to
    ``Gemma4UnifiedProcessor`` (encoder-free image patches + raw audio frames),
    whose output dict keys (``pixel_values`` / ``image_position_ids`` /
    ``input_features`` / ``input_features_mask``) match what the base processor
    already produces. ``mm_bidirectional_blocks`` auto-derives to True from
    ``text_config.use_bidirectional_attention == "vision"``.
    """


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

        _local_rank = getattr(getattr(model_config, "mapping", None), "local_rank", 0) or 0
        self._device = f"cuda:{_local_rank}"
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        self._top_config = config
        self.image_token_ids = (
            torch.tensor([config.image_token_id], dtype=torch.int32, device=self._device)
            if getattr(config, "image_token_id", None) is not None
            else None
        )
        self.audio_token_ids = (
            torch.tensor([config.audio_token_id], dtype=torch.int32, device=self._device)
            if getattr(config, "audio_token_id", None) is not None
            else None
        )
        self.video_token_ids = (
            torch.tensor([config.video_token_id], dtype=torch.int32, device=self._device)
            if getattr(config, "video_token_id", None) is not None
            else None
        )

        # --- Text backbone (reused verbatim) ---
        model_config_cp = copy.deepcopy(model_config)
        self.model_config = model_config_cp
        # Mirror the parent's quant exclude_modules HF->TRT remap so quantized
        # unified checkpoints (e.g. *-qat-w4a16) exclude the right text modules.
        qc = getattr(model_config_cp, "quant_config", None)
        if qc and getattr(qc, "exclude_modules", None):
            remapped = []
            for pat in qc.exclude_modules:
                remapped.append(pat)
                if pat.startswith(_LANG_PREFIX):
                    remapped.append(pat.replace(_LANG_PREFIX, "model."))
            qc.exclude_modules = remapped
        llm_model_config = self.get_sub_model_config(model_config_cp, "text_config")
        self.llm = Gemma4ForCausalLM(llm_model_config)

        # --- Encoder-free multimodal front-end (no towers) ---
        self.vision_tower = None  # kept so weight-mapper's hasattr(...) is stable
        self.audio_tower = None
        self.embed_vision = None
        self.embed_audio = None
        if getattr(config, "vision_config", None) is not None:
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
        if getattr(config, "audio_config", None) is not None:
            audio_in = (
                getattr(config.audio_config, "output_proj_dims", None)
                or getattr(config.audio_config, "audio_embed_dim", None)
                or config.audio_config.hidden_size
            )
            self.embed_audio = (
                Gemma4MultimodalEmbedder(
                    mm_hidden_size=audio_in,
                    text_hidden_size=config.text_config.hidden_size,
                    eps=config.audio_config.rms_norm_eps,
                    dtype=self.model_dtype,
                    mapping=model_config.mapping,
                )
                .eval()
                .to(self._device)
            )

        # Surface the text (LLM) config so KV-cache sizing + is_gemma4_hybrid
        # (global_head_dim 512 != head_dim 256) read the text geometry.
        self.post_config()
        self.is_loaded = True

    # --- Encoder-free feature extractors (override the tower-based parents) ---

    @torch.inference_mode()
    def _get_image_features(self, pixel_values, image_position_ids=None, image_seq_lens=None):
        with torch.autocast(device_type="cuda", dtype=self.model_dtype):
            feats = self.embed_vision(pixel_values, image_position_ids)  # (B, P, H)
        if image_position_ids is not None:
            pad_mask = (image_position_ids == -1).all(dim=-1)  # (B, P)
            feats = feats[~pad_mask]  # (N_valid, H)
        else:
            feats = feats.reshape(-1, feats.shape[-1])
        return feats.contiguous()

    @torch.inference_mode()
    def _get_audio_features(self, audio_features, audio_features_mask=None):
        target_dtype = self.embed_audio.embedding_projection.weight.dtype
        with torch.autocast(device_type="cuda", dtype=self.model_dtype):
            feats = self.embed_audio(audio_features.to(target_dtype))  # (B, T, H) or (T, H)
        if audio_features_mask is not None:
            feats = feats[audio_features_mask.bool()]
        else:
            feats = feats.reshape(-1, feats.shape[-1])
        return feats.contiguous()

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

    @torch.inference_mode()
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

        pixel_values_list, image_pos_list = [], []
        audio_features_list, audio_mask_list = [], []
        video_pixel_list, video_pos_list = [], []
        for mp in multimodal_params:
            img = mp.multimodal_data.get("image", {})
            if img.get("pixel_values") is not None:
                pixel_values_list.append(img["pixel_values"])
                if img.get("image_position_ids") is not None:
                    image_pos_list.append(img["image_position_ids"])
            aud = mp.multimodal_data.get("audio", {})
            if aud.get("audio_features") is not None:
                audio_features_list.append(aud["audio_features"])
                if aud.get("audio_features_mask") is not None:
                    audio_mask_list.append(aud["audio_features_mask"])
            vid = mp.multimodal_data.get("video", {})
            if vid.get("pixel_values") is not None:
                video_pixel_list.append(vid["pixel_values"])
                if vid.get("image_position_ids") is not None:
                    video_pos_list.append(vid["image_position_ids"])

        mm_embeds: List[torch.Tensor] = []
        all_mm_token_ids: List[torch.Tensor] = []
        mm_token_type_ids = None

        if len(pixel_values_list) > 0 and self.embed_vision is not None:
            pv = torch.cat(pixel_values_list)
            pid = (
                torch.cat(image_pos_list) if len(image_pos_list) == len(pixel_values_list) else None
            )
            mm_embeds.append(self._get_image_features(pv, pid))
            all_mm_token_ids.append(self.image_token_ids)

        if len(video_pixel_list) > 0 and self.embed_vision is not None:
            vpv = torch.cat(video_pixel_list)
            vpid = (
                torch.cat(video_pos_list) if len(video_pos_list) == len(video_pixel_list) else None
            )
            mm_embeds.append(self._get_image_features(vpv, vpid))
            all_mm_token_ids.append(
                self.video_token_ids if self.video_token_ids is not None else self.image_token_ids
            )

        if len(audio_features_list) > 0 and self.embed_audio is not None:
            per_audio = []
            for i, af in enumerate(audio_features_list):
                afm = audio_mask_list[i] if i < len(audio_mask_list) else None
                per_audio.append(self._get_audio_features(af, afm))
            mm_embeds.append(torch.cat(per_audio, dim=0))
            all_mm_token_ids.append(self.audio_token_ids)

        # Integer mm_token_type_ids (0=text,1=image,2=video,3=audio) drive the
        # inherited bidirectional-vision attention mask in Gemma4ForCausalLM.
        if len(mm_embeds) > 0 and input_ids is not None:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
            if self.image_token_ids is not None:
                mm_token_type_ids[torch.isin(input_ids, self.image_token_ids)] = 1
            if self.video_token_ids is not None:
                mm_token_type_ids[torch.isin(input_ids, self.video_token_ids)] = 2
            if self.audio_token_ids is not None:
                mm_token_type_ids[torch.isin(input_ids, self.audio_token_ids)] = 3

        fuse_token_ids = torch.cat(all_mm_token_ids) if all_mm_token_ids else self.image_token_ids
        mm_embeds = find_input_mm_embeds(mm_embeds, multimodal_params)

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
            "model." + k[len(_LANG_PREFIX) :]: v
            for k, v in weights.items()
            if k.startswith(_LANG_PREFIX)
        }
        self.llm.load_weights(llm_weights, weight_mapper)

        # Encoder-free MM front-end: strip the outer "model." from the non-text
        # keys and route to the embedders.
        stripped = {
            k[len("model.") :]: v
            for k, v in weights.items()
            if k.startswith("model.") and not k.startswith(_LANG_PREFIX)
        }
        if self.embed_vision is not None:
            ve = filter_weights(
                "vision_embedder", stripped
            )  # patch_ln1/dense/ln2/pos_embedding/pos_norm
            proj = filter_weights(
                "embed_vision", stripped
            )  # embedding_projection.weight (top-level)
            self.embed_vision.load_weights(ve, proj)
        if self.embed_audio is not None:
            self.embed_audio.load_weights(filter_weights("embed_audio", stripped))
