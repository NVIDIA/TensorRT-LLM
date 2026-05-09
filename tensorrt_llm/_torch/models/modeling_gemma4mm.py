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
"""Gemma4 multimodal model: multimodal embedder, input processor,
and the Gemma4ForConditionalGeneration wrapper.

Vision and audio towers use native transformers models via
AutoModel.from_config() (requires transformers>=5.5.0).
"""

import copy
import dataclasses
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from packaging.version import Version
from torch import nn

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper

from ..._utils import nvtx_range
from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ContentFormat,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..modules.linear import Linear
from .modeling_gemma4 import Gemma4ForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import ModelConfig, filter_weights, register_auto_model

_MIN_TRANSFORMERS_FOR_GEMMA4 = "5.5.0"
if Version(transformers.__version__) < Version(_MIN_TRANSFORMERS_FOR_GEMMA4):
    raise ImportError(
        f"Gemma4 requires transformers>={_MIN_TRANSFORMERS_FOR_GEMMA4}, "
        f"but found transformers=={transformers.__version__}. "
        f"Please upgrade: pip install 'transformers>={_MIN_TRANSFORMERS_FOR_GEMMA4}'"
    )

from transformers import (  # noqa: E402
    AutoModel,
    AutoTokenizer,
    Gemma4Config,
    PretrainedConfig,
    PreTrainedModel,
)

_MULTIMODAL_ENV_NAME = "TLLM_MULTIMODAL_DISAGGREGATED"


def _is_disagg() -> bool:
    return os.getenv(_MULTIMODAL_ENV_NAME, "0") == "1"


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale (for multimodal embedder pre-projection)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return normed.type_as(x)


# ---------------------------------------------------------------------------
# Multimodal embedder
# ---------------------------------------------------------------------------


class Gemma4MultimodalEmbedder(nn.Module):
    """Projects tower outputs into LM embedding space.

    Architecture (matching HF Gemma4MultimodalEmbedder):
        embedding_pre_projection_norm (RMSNorm, no learnable scale)
        -> embedding_projection (Linear, no bias)

    Only ``embedding_projection.weight`` exists in the checkpoint
    because the norm has no learnable parameters (with_scale=False).
    """

    def __init__(
        self,
        mm_hidden_size: int,
        text_hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
        mapping=None,
    ):
        super().__init__()
        self.embedding_pre_projection_norm = RMSNormNoScale(mm_hidden_size, eps=eps)
        self.embedding_projection = Linear(
            in_features=mm_hidden_size,
            out_features=text_hidden_size,
            bias=False,
            dtype=dtype,
            mapping=mapping,
        )

    def load_weights(self, weights: Dict):
        proj_weight = weights.get("embedding_projection.weight")
        if proj_weight is not None:
            self.embedding_projection.weight.data.copy_(proj_weight)

    @torch.inference_mode()
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)


def _normalize_audio_inputs(audios, target_sr: int = 16000):
    """Normalize multimodal audio payloads for the Gemma4 feature extractor.

    The TRT-LLM multimodal input loader stores audio as ``(waveform, sr)``
    tuples (from ``soundfile.read``); callers may also pass bare numpy or
    torch arrays.  The Gemma4 feature extractor wants a list of 1-D numpy
    float32 waveforms at its configured ``sampling_rate`` (16 kHz).

    Steps applied per entry:
    1. Strip the sampling-rate component of ``(array, sr)`` tuples.
    2. Convert torch tensors to numpy.
    3. Downmix multi-channel audio to mono by averaging channels.
    4. Resample to ``target_sr`` if the source rate differs and is known.
    """
    import numpy as np

    normalized = []
    for a in audios:
        src_sr = None
        if isinstance(a, tuple) and len(a) == 2:
            a, src_sr = a
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        arr = np.asarray(a)
        # Channels can appear as last or first axis depending on the loader.
        if arr.ndim == 2:
            # soundfile returns (samples, channels); librosa/torchaudio
            # sometimes return (channels, samples).
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T
            arr = arr.mean(axis=1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1)
        arr = arr.astype(np.float32, copy=False)
        if src_sr is not None and src_sr != target_sr:
            try:
                import scipy.signal as sps

                # Rational resample for simple ratios; fallback to polyphase
                # resample_poly for arbitrary ratios.
                gcd = np.gcd(int(src_sr), int(target_sr))
                up = int(target_sr // gcd)
                down = int(src_sr // gcd)
                arr = sps.resample_poly(arr, up, down).astype(np.float32, copy=False)
            except ImportError:
                # Fallback: naive linear interp (accuracy is lower but the
                # audio path is still functional).
                target_len = int(arr.size * target_sr / src_sr)
                arr = np.interp(
                    np.linspace(0, arr.size - 1, target_len), np.arange(arr.size), arr
                ).astype(np.float32)
        normalized.append(arr)
    return normalized


# ---------------------------------------------------------------------------
# Input processor
# ---------------------------------------------------------------------------


class Gemma4InputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    """Preprocesses image inputs for Gemma4.

    Tries to use the HF ``Gemma4Processor`` if available in the installed
    transformers; otherwise falls back to manual tokenization + image
    processing using the image processor saved in the model directory.
    """

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: AutoTokenizer,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._config = config
        self._tokenizer = tokenizer
        self._model_path = model_path
        self._dtype = getattr(config, "torch_dtype", torch.bfloat16)

        self._processor = None
        try:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast,
            )
        except Exception:
            logger.warning(
                "Could not load AutoProcessor for Gemma4. "
                "Image preprocessing will use manual fallback."
            )

        self._image_processor = None
        if self._processor is not None and hasattr(self._processor, "image_processor"):
            self._image_processor = self._processor.image_processor
        elif self._processor is None:
            try:
                from transformers import AutoImageProcessor

                self._image_processor = AutoImageProcessor.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code
                )
            except Exception:
                logger.warning("Could not load image processor for Gemma4.")

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self):
        return self._processor

    def get_vocab_size(self) -> Optional[int]:
        # Gemma4Config exposes vocab_size only on text_config; the base class
        # default also calls tokenizer.vocab_size which raises
        # NotImplementedError on transformers>=5.5 base PreTrainedTokenizerBase.
        text_config = getattr(self._config, "text_config", self._config)
        vocab_size = getattr(text_config, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)
        return super().get_vocab_size()

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @nvtx_range("[MM] preprocess")
    def _preprocess(self, inputs):
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data", {})
        allowed_keys = {"image", "audio", "video"}
        unknown = set(mm_data) - allowed_keys if mm_data else set()
        if unknown:
            raise KeyError(
                f"Gemma4 multi_modal_data only supports {sorted(allowed_keys)}, "
                f"got unknown keys {sorted(unknown)}."
            )

        images = mm_data.get("image")
        audios = mm_data.get("audio")
        videos = mm_data.get("video")
        pixel_values = None
        image_position_ids = None
        pixel_values_videos = None
        video_position_ids = None
        input_features = None
        input_features_mask = None

        # ---------------------------------------------------------------
        # Video pre-processing (manual path).
        #
        # Calling the unified ``Gemma4Processor`` with ``videos=`` currently
        # fails strict typed-dict validation in transformers 5.5.x
        # ("merged_typed_dict.__init__() got an unexpected keyword argument
        # 'video_sizes'"). Bypass it: call the underlying ``video_processor``
        # directly, then expand the ``<|video|>`` placeholder in the text
        # the way ``Gemma4Processor`` would have, before the regular text /
        # image / audio path runs.
        # ---------------------------------------------------------------
        if (
            videos is not None
            and self._processor is not None
            and hasattr(self._processor, "video_processor")
        ):
            import re

            norm_videos = []
            for v in videos:
                frames = getattr(v, "frames", None)
                if frames is None:
                    frames = v
                if len(frames) == 0:
                    raise ValueError("Got an empty frame list for a Gemma4 video input.")
                if isinstance(frames[0], torch.Tensor):
                    from torchvision.transforms.functional import to_pil_image

                    frames = [to_pil_image(f.cpu()) for f in frames]
                norm_videos.append(list(frames))

            video_out = self._processor.video_processor(videos=norm_videos, return_metadata=True)
            pixel_values_videos = video_out.get("pixel_values_videos")
            video_position_ids = video_out.get("video_position_ids")
            num_soft_tokens = video_out.get("num_soft_tokens_per_video", [])
            video_metadata = video_out.get("video_metadata", [])

            video_token = getattr(self._processor, "video_token", "<|video|>")
            boi_token = getattr(self._processor, "boi_token", "<start_of_image>")
            eoi_token = getattr(self._processor, "eoi_token", "<end_of_image>")
            replacements = []
            for metadata, n_tokens in zip(video_metadata, num_soft_tokens):
                fps = metadata.fps if metadata.fps is not None else 24
                metadata.fps = fps
                ts_strs = [f"{int(s // 60):02d}:{int(s % 60):02d}" for s in metadata.timestamps]
                replacements.append(
                    " ".join(f"{t} {boi_token}{video_token * n_tokens}{eoi_token}" for t in ts_strs)
                )
            replacements_iter = iter(replacements)
            text_prompt = re.sub(
                re.escape(video_token),
                lambda _: next(replacements_iter),
                text_prompt,
            )

        # If video is the only modality, skip self._processor (which fails
        # validation on the installed transformers when its video_processor
        # has been touched in the same process) and tokenize directly via
        # ``self._tokenizer``. The ``<|video|>`` placeholder has already
        # been expanded into the proper soft-token sequence above.
        if videos is not None and images is None and audios is None:
            ids = self._tokenizer(text_prompt, return_tensors="pt")["input_ids"]
            input_ids = ids
        elif self._processor is not None and (images is not None or audios is not None):
            do_rescale = self._image_processor.do_rescale
            if images is not None and isinstance(images[0], torch.Tensor):
                do_rescale = False
            proc_kwargs = dict(text=text_prompt, return_tensors="pt")
            if images is not None:
                proc_kwargs["images"] = images
                proc_kwargs["do_rescale"] = do_rescale
            if audios is not None:
                # ``load_audio`` returns ``(array, sampling_rate)`` tuples; the
                # Gemma4 processor's feature extractor only wants the raw 1-D
                # waveform array, so strip the sampling rate and normalize the
                # data to ``np.float32``.
                norm_audios = _normalize_audio_inputs(
                    audios,
                    target_sr=getattr(self._processor.feature_extractor, "sampling_rate", 16000),
                )
                proc_kwargs["audio"] = norm_audios
            proc_out = self._processor(**proc_kwargs).to(dtype=self.dtype)
            input_ids = proc_out["input_ids"]
            pixel_values = proc_out.get("pixel_values")
            image_position_ids = proc_out.get("image_position_ids")
            input_features = proc_out.get("input_features")
            input_features_mask = proc_out.get("input_features_mask")
        else:
            input_ids = self._tokenizer(
                text_prompt,
                return_tensors="pt",
            )["input_ids"]
            if images is not None and self._image_processor is not None:
                img_out = self._image_processor(images=images, return_tensors="pt")
                pixel_values = img_out.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(dtype=self.dtype)
                image_position_ids = img_out.get("image_position_ids")
            if (
                audios is not None
                and self._processor is not None
                and hasattr(self._processor, "feature_extractor")
            ):
                norm_audios = _normalize_audio_inputs(
                    audios,
                    target_sr=getattr(self._processor.feature_extractor, "sampling_rate", 16000),
                )
                af = self._processor.feature_extractor(norm_audios, return_tensors="pt")
                input_features = af.get("input_features")
                input_features_mask = af.get("input_features_mask")
                if input_features is not None:
                    input_features = input_features.to(dtype=self.dtype)

        # Cast video features to model dtype.
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(dtype=self.dtype)
        return (
            input_ids,
            pixel_values,
            image_position_ids,
            input_features,
            input_features_mask,
            pixel_values_videos,
            video_position_ids,
        )

    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        (
            input_ids,
            pixel_values,
            image_position_ids,
            input_features,
            input_features_mask,
            pixel_values_videos,
            video_position_ids,
        ) = self._preprocess(inputs)
        mm_inner: Dict = {}
        if pixel_values is not None:
            image_data: Dict = {"pixel_values": pixel_values}
            if image_position_ids is not None:
                image_data["image_position_ids"] = image_position_ids
            mm_inner["image"] = image_data
        if input_features is not None:
            audio_data: Dict = {"audio_features": input_features}
            if input_features_mask is not None:
                audio_data["audio_features_mask"] = input_features_mask
            mm_inner["audio"] = audio_data
        if pixel_values_videos is not None:
            # Reshape (B, num_frames, num_patches, C) -> per-frame
            # (num_frames, num_patches, C) so each video frame is fed to the
            # vision tower as if it were a separate image.  ``video_position_ids``
            # carries the (y, x) patch coordinates per frame, identical layout
            # to image_position_ids per image.
            v = pixel_values_videos
            vp = video_position_ids
            if v.dim() == 4:  # (B, F, P, C) — flatten to (B*F, P, C)
                v = v.reshape(-1, v.shape[-2], v.shape[-1])
            if vp is not None and vp.dim() == 4:
                vp = vp.reshape(-1, vp.shape[-2], vp.shape[-1])
            video_data: Dict = {"pixel_values": v}
            if vp is not None:
                video_data["image_position_ids"] = vp
            mm_inner["video"] = video_data
        multimodal_data = {"multimodal_data": mm_inner} if mm_inner else None
        return input_ids[0].to(torch.int32).tolist(), multimodal_data


# ---------------------------------------------------------------------------
# Gemma4ForConditionalGeneration (multimodal wrapper)
# ---------------------------------------------------------------------------


@register_auto_model("Gemma4ForConditionalGeneration")
@register_input_processor(
    Gemma4InputProcessor,
    model_type="gemma4",
    # Gemma4's HF chat template is ContentFormat.OPENAI: it iterates the
    # content list-of-dicts in order and emits ``<|image|>`` / ``<|audio|>``
    # tokens at exactly the position where each media item appears.  That
    # makes the *order* of items in the OpenAI content list load-bearing.
    #
    # ``interleave_placeholders=True`` opts this model into the
    # position-preserving build path in ``MultimodalLmEvalWrapper`` and
    # ``serve/chat_utils.py`` — when the user prompt embeds placeholders
    # inside text (e.g. MMMU Pro: "Consider <image 1>. What does <image 2>
    # show?"), the corresponding ``content_parts`` entry sits at the same
    # position, preserving question grounding.  ``placeholder_placement``
    # is retained as a fallback for callers that still bulk-insert (e.g.
    # ``add_multimodal_placeholders`` against a pre-stripped string).
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
class Gemma4ForConditionalGeneration(PreTrainedModel):
    """Gemma4 multimodal model: LLM + vision tower + multimodal embedder.

    Follows the Gemma3VLM pattern but adapted for Gemma4's architecture:
    - Custom vision tower with 2D RoPE and spatial pooling
    - Multimodal embedder with pre-projection RMSNorm (no scale)
    - Support for image_position_ids (2D patch coordinates)
    - mm_token_type_ids-based bidirectional masking
    """

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        """Gemma4-specific defaults — see Gemma4ForCausalLM.get_model_defaults."""
        return {
            "attn_backend": "FLASHINFER",
        }

    def __init__(self, model_config: ModelConfig[Gemma4Config]):
        if _is_disagg():
            raise NotImplementedError(
                "Gemma4ForConditionalGeneration does not support "
                "disaggregated inference yet. Please unset the "
                f"{_MULTIMODAL_ENV_NAME} environment variable, "
                "or set it to '0'."
            )

        config = model_config.pretrained_config
        super().__init__(config)

        # Pin multimodal tensors to the local rank so each rank of a multi-GPU
        # run operates on its own device (avoids silent cross-rank copies or
        # multi-node crashes).  model_config.mapping is always populated; fall
        # back to cuda:0 only for the unit-test path that never goes through
        # Mapping.
        _local_rank = getattr(getattr(model_config, "mapping", None), "local_rank", 0) or 0
        self._device = f"cuda:{_local_rank}"
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        self._top_config = config  # Preserve before post_config replaces it

        self.image_token_ids = torch.tensor(
            [config.image_token_id], dtype=torch.int32, device=self._device
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

        model_config_cp = copy.deepcopy(model_config)
        self.model_config = model_config_cp

        # --- Language model ---
        # Remap quantization exclude_modules patterns from HF naming to
        # TRT-LLM naming so that excluded layers (e.g., attention for
        # NVFP4) are correctly identified by apply_quant_config_exclude_modules
        # inside the LLM sub-model.
        # HF: "model.language_model.layers.X.self_attn*"
        # TRT-LLM CausalLM sub-model: "model.layers.X.self_attn*"
        qc = getattr(model_config_cp, "quant_config", None)
        if qc and getattr(qc, "exclude_modules", None):
            remapped = []
            for pat in qc.exclude_modules:
                remapped.append(pat)
                if pat.startswith("model.language_model."):
                    remapped.append(pat.replace("model.language_model.", "model."))
            qc.exclude_modules = remapped
        llm_model_config = self.get_sub_model_config(model_config_cp, "text_config")
        self.llm = Gemma4ForCausalLM(llm_model_config)

        # --- Vision tower (native transformers, eager mode) ---
        if config.vision_config is not None:
            self.vision_tower = AutoModel.from_config(config.vision_config).eval().to(self._device)
            vision_hidden = config.vision_config.hidden_size
            text_hidden = config.text_config.hidden_size
            vision_eps = config.vision_config.rms_norm_eps
            self.embed_vision = (
                Gemma4MultimodalEmbedder(
                    mm_hidden_size=vision_hidden,
                    text_hidden_size=text_hidden,
                    eps=vision_eps,
                    dtype=self.model_dtype,
                    mapping=model_config.mapping,
                )
                .eval()
                .to(self._device)
            )
        else:
            self.vision_tower = None
            self.embed_vision = None

        # --- Audio tower (native transformers, eager mode) ---
        if config.audio_config is not None:
            self.audio_tower = AutoModel.from_config(config.audio_config).eval().to(self._device)
            audio_hidden = getattr(
                config.audio_config, "output_proj_dims", config.audio_config.hidden_size
            )
            text_hidden = config.text_config.hidden_size
            audio_eps = config.audio_config.rms_norm_eps
            self.embed_audio = (
                Gemma4MultimodalEmbedder(
                    mm_hidden_size=audio_hidden,
                    text_hidden_size=text_hidden,
                    eps=audio_eps,
                    dtype=self.model_dtype,
                    mapping=model_config.mapping,
                )
                .eval()
                .to(self._device)
            )
        else:
            self.audio_tower = None
            self.embed_audio = None

        self.post_config()
        self.is_loaded = True

    @staticmethod
    def get_sub_model_config(
        model_config: ModelConfig[Gemma4Config],
        name: str,
    ) -> ModelConfig:
        assert name in ["text_config", "vision_config", "audio_config"], (
            f"Expected subconfig name to be 'text_config', 'vision_config', "
            f"or 'audio_config'. Got {name} instead."
        )
        pretrained_config = getattr(model_config.pretrained_config, name)
        quant_config = model_config.quant_config if name == "text_config" else None
        preferred_backend = "FLASHINFER" if name == "text_config" else "TRTLLM"
        sub_config: ModelConfig = dataclasses.replace(
            model_config,
            pretrained_config=pretrained_config,
            attn_backend=preferred_backend,
            quant_config=quant_config,
        )
        if (
            hasattr(sub_config.pretrained_config, "torch_dtype")
            and sub_config.pretrained_config.torch_dtype is None
        ):
            sub_config.pretrained_config.torch_dtype = model_config.pretrained_config.torch_dtype
        return sub_config

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        # Gemma4 checkpoint keys: model.language_model.X -> need model.X for LLM
        # Remap: "model.language_model.layers.0..." -> "model.layers.0..."
        _LANG = "model.language_model."
        llm_weights = {}
        for k, v in weights.items():
            if k.startswith(_LANG):
                llm_weights["model." + k[len(_LANG) :]] = v
        self.llm.load_weights(llm_weights, weight_mapper)

        # Strip outer "model." for non-LLM components
        stripped = {
            (k[len("model.") :] if k.startswith("model.") else k): v for k, v in weights.items()
        }

        if self.vision_tower is not None:
            vit_weights = filter_weights("vision_tower", stripped)
            # Native transformers models use load_state_dict, not load_weights
            self.vision_tower.load_state_dict(vit_weights, strict=False)

        if self.embed_vision is not None:
            embed_v_weights = filter_weights("embed_vision", stripped)
            self.embed_vision.load_weights(embed_v_weights)

        if self.audio_tower is not None:
            audio_weights = filter_weights("audio_tower", stripped)
            self.audio_tower.load_state_dict(audio_weights, strict=False)

        if self.embed_audio is not None:
            embed_a_weights = filter_weights("embed_audio", stripped)
            self.embed_audio.load_weights(embed_a_weights)

    def post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @nvtx_range("[Vision] process")
    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooling_k2 = self._top_config.vision_config.pooling_kernel_size**2
        target_dtype = self.embed_vision.embedding_projection.weight.dtype

        per_image_features = []
        for i in range(pixel_values.shape[0]):
            pv = pixel_values[i].unsqueeze(0)
            pp = image_position_ids[i].unsqueeze(0) if image_position_ids is not None else None

            max_patches = pv.shape[1]

            if pp is None:
                side = int(math.sqrt(max_patches))
                pp = torch.stack(
                    torch.meshgrid(
                        torch.arange(side, device=pv.device),
                        torch.arange(side, device=pv.device),
                        indexing="ij",
                    ),
                    dim=-1,
                ).reshape(1, -1, 2)

            output_length = max_patches // pooling_k2

            with torch.autocast(device_type="cuda", dtype=self.model_dtype):
                output = self.vision_tower(pv, pp, output_length=output_length)
                hidden = output.last_hidden_state
                projected = self.embed_vision(hidden.unsqueeze(0).to(target_dtype)).squeeze(0)
            per_image_features.append(projected)

        return torch.cat(per_image_features, dim=0).contiguous()

    @nvtx_range("[Audio] process")
    def _get_audio_features(
        self,
        audio_features: torch.Tensor,
        audio_features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process audio features through audio tower and embedder.

        Returns a 2-D tensor of shape (total_audio_tokens, text_hidden) so
        that it can be concatenated with image features before being fused
        into the LLM input embeddings.
        """
        target_dtype = self.embed_audio.embedding_projection.weight.dtype
        with torch.autocast(device_type="cuda", dtype=self.model_dtype):
            if audio_features_mask is not None:
                output = self.audio_tower(
                    audio_features, attention_mask=audio_features_mask.to(audio_features.device)
                )
            else:
                output = self.audio_tower(audio_features)
            hidden = output.last_hidden_state  # (B, T, H_audio)
            projected = self.embed_audio(hidden.to(target_dtype))  # (B, T, H_text)
        # Optionally drop padding frames reported by the audio tower.
        tower_mask = getattr(output, "attention_mask", None)
        if tower_mask is not None and tower_mask.dtype == torch.bool:
            projected = projected[tower_mask]
        else:
            projected = projected.reshape(-1, projected.shape[-1])
        return projected.contiguous()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_params = kwargs.get("multimodal_params", [])

        # --- Extract image data ---
        pixel_values_list = []
        image_position_ids_list = []
        # --- Extract audio data ---
        audio_features_list = []
        audio_features_mask_list = []
        # --- Extract video data (treated as image frames at the tower) ---
        video_pixel_values_list = []
        video_position_ids_list = []
        for mp in multimodal_params:
            img_data = mp.multimodal_data.get("image", {})
            pv = img_data.get("pixel_values")
            if pv is not None:
                pixel_values_list.append(pv)
                pid = img_data.get("image_position_ids")
                if pid is not None:
                    image_position_ids_list.append(pid)

            aud_data = mp.multimodal_data.get("audio", {})
            af = aud_data.get("audio_features")
            if af is not None:
                audio_features_list.append(af)
                afm = aud_data.get("audio_features_mask")
                if afm is not None:
                    audio_features_mask_list.append(afm)

            vid_data = mp.multimodal_data.get("video", {})
            vpv = vid_data.get("pixel_values")
            if vpv is not None:
                video_pixel_values_list.append(vpv)
                vpid = vid_data.get("image_position_ids")
                if vpid is not None:
                    video_position_ids_list.append(vpid)

        mm_embeds = []
        all_mm_token_ids = []
        mm_token_type_ids = None

        # --- Process image features ---
        if len(pixel_values_list) > 0:
            pixel_values = torch.cat(pixel_values_list)
            image_position_ids = (
                torch.cat(image_position_ids_list)
                if len(image_position_ids_list) == len(pixel_values_list)
                else None
            )
            image_features = self._get_image_features(
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
            )
            mm_embeds.append(image_features)
            all_mm_token_ids.append(self.image_token_ids)

        # --- Process video frames (each frame goes through the vision tower
        # exactly like an image).  Gemma4 expands ``<|video|>`` into
        # ``<|image><|video|>*N<image|>`` (i.e., the soft-token slots use
        # the dedicated ``video_token_id``, not the image token id), so we
        # add ``video_token_ids`` separately to ``all_mm_token_ids``.
        if len(video_pixel_values_list) > 0:
            video_pixel_values = torch.cat(video_pixel_values_list)
            video_pos_ids = (
                torch.cat(video_position_ids_list)
                if len(video_position_ids_list) == len(video_pixel_values_list)
                else None
            )
            video_features = self._get_image_features(
                pixel_values=video_pixel_values,
                image_position_ids=video_pos_ids,
            )
            mm_embeds.append(video_features)
            if self.video_token_ids is not None:
                all_mm_token_ids.append(self.video_token_ids)
            else:
                all_mm_token_ids.append(self.image_token_ids)

        # --- Process audio features ---
        if len(audio_features_list) > 0 and self.audio_tower is not None:
            # Different requests in the batch can have different audio
            # lengths (and ``input_features`` shape ``(1, frames, 128)``
            # therefore differs in dim 1).  Process each audio independently
            # and concatenate the resulting valid soft-token embeddings so
            # that ``mm_embeds`` is a single ``(N_total_audio_tokens, H)``
            # tensor matching the audio token positions in the batched
            # ``input_ids``.
            per_audio_embeds = []
            for i, af in enumerate(audio_features_list):
                afm = audio_features_mask_list[i] if i < len(audio_features_mask_list) else None
                per_audio_embeds.append(self._get_audio_features(af, afm))
            audio_embeds = torch.cat(per_audio_embeds, dim=0)
            mm_embeds.append(audio_embeds)
            if self.audio_token_ids is not None:
                all_mm_token_ids.append(self.audio_token_ids)

        # Build integer mm_token_type_ids: 0=text, 1=image, 2=video, 3=audio
        # (matches HF ``Gemma4Processor.create_mm_token_type_ids`` convention).
        # _get_token_type_mask expects integer type IDs, not a boolean mask.
        if len(mm_embeds) > 0:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
            mm_token_type_ids[torch.isin(input_ids, self.image_token_ids)] = 1
            if self.video_token_ids is not None:
                mm_token_type_ids[torch.isin(input_ids, self.video_token_ids)] = 2
            if self.audio_token_ids is not None:
                mm_token_type_ids[torch.isin(input_ids, self.audio_token_ids)] = 3

        fuse_token_ids = torch.cat(all_mm_token_ids) if all_mm_token_ids else self.image_token_ids

        # Build a PLE-safe view of the original input_ids where every
        # multimodal token is replaced by the text pad_token_id.  The
        # Gemma4 Per-Layer Embedding lookup uses this view so the PLE table
        # is not consulted at audio/image/video positions (matches HF's
        # Gemma4Model behaviour).  Without this, multimodal requests on
        # E2B/E4B (which use PLE) produce garbage output because
        # ``fuse_input_embeds`` returns ``input_ids=None`` and PLE is then
        # silently skipped inside ``Gemma4TextModel.forward``.
        ple_input_ids = None
        if input_ids is not None and len(mm_embeds) > 0:
            text_config = getattr(self.config, "text_config", self.config)
            pad_id = getattr(text_config, "pad_token_id", None)
            if pad_id is not None:
                mm_token_ids_for_mask = fuse_token_ids.to(input_ids.device)
                mm_mask = torch.isin(input_ids, mm_token_ids_for_mask)
                ple_input_ids = torch.where(mm_mask, torch.full_like(input_ids, pad_id), input_ids)

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embeds,
            mm_token_ids=fuse_token_ids,
            **kwargs,
        )
        logits = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            mm_token_type_ids=mm_token_type_ids,
            ple_input_ids=ple_input_ids,
            lora_params=kwargs.get("lora_params", None),
        )
        return logits

    @property
    def mm_token_ids(self):
        ids = [self.image_token_ids]
        if self.audio_token_ids is not None:
            ids.append(self.audio_token_ids)
        return torch.cat(ids) if len(ids) > 1 else self.image_token_ids
