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
"""Inkling vision + audio preprocessing, towers, and multimodal input processor.

Scope of this module (image + audio media paths; the video path reuses the image
tower as multi-frame images and lives with the vision plumbing here):
  * :class:`InklingImagePreprocessor` -- turn raw images into the Inkling hMLP
    ``vision_patches_bthwc`` tensor, numerically matching the SGLang reference
    ``sglang.srt.multimodal.inkling.image_processing.InklingImageProcessor``
    (the requested NVFP4 serving comparand). Long-edge rescale, an asymmetric
    ``W // patch + 1`` patch grid, CLIP-style mean/std normalization with a
    ``-1/255`` pad value, and a temporal duplication ``T=2``. Implemented in
    vectorized numpy (NO numba, NO ``sglang`` import) so the production path has
    no third-party kernel-cache dependency.
  * :class:`InklingInputProcessor` -- a
    :class:`~tensorrt_llm.inputs.registry.BaseMultimodalInputProcessor` that
    registers the Inkling image placeholder path with TRT-LLM's placeholder
    registry. It expands one ``<image>`` placeholder token into one token per
    vision patch (hMLP emits one text-hidden row per patch, so
    ``num_tokens == num_patches``), preserves text-only passthrough, and FAILS
    LOUDLY when the placeholder count and the media/feature-row count disagree.

The vision *tower* (``InklingVisionModel`` + projector) and the
``inputs_embeds`` fusion are the following goals (1.3 / 1.4); this module only
produces the preprocessed patch features and the validated, expanded token
stream that those goals consume.

References (read-only, on disk):
  SGLang image proc : python/sglang/srt/multimodal/inkling/image_processing.py
  SGLang mm proc    : python/sglang/srt/multimodal/processors/inkling.py
  SGLang tokenizer  : python/sglang/srt/parser/inkling_tokenizer.py
                      (internal IMAGE_TOKEN_ID = -101, AUDIO_TOKEN_ID = -102;
                       TRT uses the in-vocab chat-template image token 200054 --
                       see DEFAULT_IMAGE_TOKEN_ID)
  HF image proc     : transformers/.../models/inkling/image_processing_inkling.py
"""

from __future__ import annotations

import io
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...inputs.registry import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    DefaultInputProcessor,
)

# ---------------------------------------------------------------------------
# Preprocessing constants -- byte-exact copies of the SGLang comparand
# (image_processing.py:15-18). SGLang's IMAGE_STD differs from transformers'
# OPENAI_CLIP_STD only in the 7th+ significant digit; we track the SGLang values
# because SGLang is the requested serving comparand on the NVFP4 checkpoint.
# ---------------------------------------------------------------------------
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.2613026, 0.2757771], dtype=np.float32)
PAD_RAW_VALUE = np.float32(-1.0 / 255.0)
# Normalized pad value; equals normalize(PAD_RAW_VALUE) so a canvas pre-filled
# with PAD_RAW_VALUE and normalized wholesale reproduces SGLang's per-patch pad.
PAD_NORM = (np.full((3,), PAD_RAW_VALUE, dtype=np.float32) - IMAGE_MEAN) / IMAGE_STD

# Default vision geometry (checkpoint ``vision_config`` / SGLang processor).
DEFAULT_PATCH_SIZE = 40
DEFAULT_TEMPORAL_PATCH_SIZE = 2  # a static image is temporally duplicated (T=2)
DEFAULT_RESCALE_IMAGE_FRAC = 2.0
DEFAULT_RESCALE_MAX_UPSCALED_LONG_EDGE = 2048

# Inkling image placeholder id. One sentinel appears per image in the
# pre-rendered token stream; the input processor expands it into one token per
# patch and the vision fusion overwrites those positions with tower embeddings
# (so its own embedding is never used). The Inkling chat template renders an
# image content part as ``<|content_image|>(200005) <|unused_200054|>(200054)
# <|end_message|>(200010)``, so ``<|unused_200054|>`` (id 200054, IN-VOCAB) IS
# the trained image placeholder token. It MUST be in-vocab: TensorRT-LLM's
# executor validates request token ids against the vocab and rejects an
# out-of-range id (a negative SGLang-style -101 raises ``RequestError: Token ID
# out of range`` at ``llm.generate``). SGLang's serving uses -101 only as an
# INTERNAL sentinel (parser/inkling_tokenizer.py); its config.json omits
# ``image_token_id`` so its processor falls back to -101. The two ids are
# interchangeable for parity: both are replaced by the identical vision
# embeddings, so the prefilled stream and the logits are identical regardless of
# which placeholder id the token carried before replacement. config.json may
# override via ``image_token_id``.
DEFAULT_IMAGE_TOKEN_ID = 200054  # <|unused_200054|> (in-vocab; see note above)
# Inkling audio placeholder id. Like the image sentinel, one appears per audio
# clip in the pre-rendered token stream; the input processor expands it into one
# token per dMel frame and the audio fusion overwrites those positions with tower
# rows. The Inkling chat template renders an audio content part as
# ``<|content_audio_input|>(200020) <|unused_200053|>(200053) <|audio_end|>
# <|end_message|>``, so ``<|unused_200053|>`` (id 200053, IN-VOCAB) IS the trained
# audio placeholder token (the direct analogue of the image ``<|unused_200054|>``).
# SGLang uses -102 only as an INTERNAL sentinel; both ids are interchangeable for
# parity because the audio tower overwrites those positions identically.
# config.json / the top-level config may override via ``audio_token_id``.
DEFAULT_AUDIO_TOKEN_ID = 200053  # <|unused_200053|> (in-vocab; see note above)

# Default Inkling dMel audio-preprocessing geometry (checkpoint ``audio_config`` +
# ``processor_config.json`` ``feature_extractor``; SGLang
# ``multimodal/inkling/feature_extraction.py`` ``InklingAudioEncoderParams``).
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_WINDOW_SIZE_MULTIPLIER = 2.0
DEFAULT_AUDIO_N_MELS = 80
DEFAULT_AUDIO_NUM_DMEL_BINS = 16  # == audio_config.mel_vocab_size
DEFAULT_AUDIO_DMEL_MIN_VALUE = -7.0
DEFAULT_AUDIO_DMEL_MAX_VALUE = 2.0
DEFAULT_AUDIO_TOKEN_DURATION_S = 0.05  # 1 dMel frame == 1 audio token (hop/sr)


# ===========================================================================
# Image geometry (verbatim logic from SGLang image_processing.py)
# ===========================================================================
def scaled_image_dimensions(
    width: int,
    height: int,
    rescale_image_frac: Optional[float] = DEFAULT_RESCALE_IMAGE_FRAC,
    rescale_image_max_upscaled_long_edge: Optional[int] = DEFAULT_RESCALE_MAX_UPSCALED_LONG_EDGE,
) -> Tuple[int, int]:
    """Long-edge scale ``(width, height)`` before patching.

    Ports ``_scaled_image_dimensions`` (image_processing.py:46-75): scale the
    long edge by ``rescale_image_frac`` (aspect preserved), optionally capping
    only upscaling (the cap never shrinks an image already above it), with
    half-away-from-zero rounding ``floor(v * ratio + 0.5)``.
    """
    if rescale_image_frac is None:
        return width, height
    long_edge = max(width, height)
    if long_edge == 0:
        return width, height
    target_long_edge = float(long_edge) * rescale_image_frac
    if rescale_image_max_upscaled_long_edge is not None:
        effective_cap = max(rescale_image_max_upscaled_long_edge, long_edge)
        target_long_edge = min(target_long_edge, float(effective_cap))
    ratio = target_long_edge / float(long_edge)
    if ratio == 1.0:
        return width, height

    def scale(value: int) -> int:
        return max(1, math.floor(float(value) * ratio + 0.5))

    return scale(width), scale(height)


def patch_grid(
    height: int, width: int, patch_size: int = DEFAULT_PATCH_SIZE
) -> Tuple[int, int, int]:
    """Return ``(num_patches, nph, npw)``.

    ``nph = ceil(H / P)``, ``npw = W // P + 1`` (the asymmetric ``+1`` width
    padding; image_processing.py:117-118,167-168). One text-hidden token is
    emitted per patch, so ``placeholder_count == num_patches == nph * npw``.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")
    nph = (height + patch_size - 1) // patch_size
    npw = width // patch_size + 1
    return nph * npw, nph, npw


def _to_pil_rgb(image: Any):
    """Coerce one image input into a PIL ``RGB`` image.

    Accepts raw PNG/JPEG ``bytes``, a local path / ``file://`` URI, a PIL image,
    a numpy ``uint8`` HWC array, or a ``torch.Tensor``. HTTP/data URIs must be
    resolved to bytes upstream (matching SGLang's ``_load_image_bytes``), so a
    URL raises rather than silently reaching for the network here.
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            raise ValueError(
                "InklingImagePreprocessor received a URL/data: image; resolve "
                "it to bytes upstream before preprocessing."
            )
        path = image[len("file://") :] if image.startswith("file://") else image
        with open(path, "rb") as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
        arr = arr.astype("uint8") if arr.dtype != np.uint8 else arr
        return Image.fromarray(arr).convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image.astype("uint8") if image.dtype != np.uint8 else image
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


class InklingImagePreprocessor:
    """Raw images -> ``vision_patches_bthwc`` for the Inkling hMLP tower.

    Numerically matches SGLang ``InklingImageProcessor`` (defaults
    ``patch_size=40``, ``rescale_image_frac=2.0``,
    ``rescale_image_max_upscaled_long_edge=2048``, ``temporal_patch_size=2``),
    but is pure vectorized numpy + torch so the production path carries no numba
    kernel-cache dependency.
    """

    def __init__(
        self,
        patch_size: int = DEFAULT_PATCH_SIZE,
        temporal_patch_size: int = DEFAULT_TEMPORAL_PATCH_SIZE,
        rescale_image_frac: Optional[float] = DEFAULT_RESCALE_IMAGE_FRAC,
        rescale_image_max_upscaled_long_edge: Optional[
            int
        ] = DEFAULT_RESCALE_MAX_UPSCALED_LONG_EDGE,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be greater than zero")
        if temporal_patch_size <= 0:
            raise ValueError("temporal_patch_size must be greater than zero")
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.rescale_image_frac = rescale_image_frac
        self.rescale_image_max_upscaled_long_edge = rescale_image_max_upscaled_long_edge
        self.dtype = dtype

    def _patchify(self, arr: np.ndarray) -> np.ndarray:
        """``(H, W, 3)`` uint8 -> ``(num_patches, P, P, 3)`` float32.

        Vectorized equivalent of SGLang's ``_fill_patches_numba``: build a
        PAD_RAW_VALUE canvas of ``(nph*P, npw*P, 3)``, drop the image into its
        top-left, normalize the whole canvas in float32
        (``(uint8/255 - mean) / std``; the padded border becomes ``PAD_NORM``),
        then fold ``(nph, P, npw, P, 3)`` into row-major patch order ``i*npw+j``.
        """
        h, w = int(arr.shape[0]), int(arr.shape[1])
        num_patches, nph, npw = patch_grid(h, w, self.patch_size)
        p = self.patch_size
        inv255 = np.float32(1.0) / np.float32(255.0)

        canvas = np.empty((nph * p, npw * p, 3), dtype=np.float32)
        canvas[:] = PAD_RAW_VALUE
        canvas[:h, :w, :] = arr[:h, :w, :].astype(np.float32) * inv255
        canvas = (canvas - IMAGE_MEAN) / IMAGE_STD  # float32; border -> PAD_NORM

        # (nph*P, npw*P, 3) -> (nph, P, npw, P, 3) -> (nph, npw, P, P, 3)
        # -> (num_patches, P, P, 3) with patch index i*npw + j (row-major),
        # exactly matching SGLang's ``k -> (i=k//npw, j=k%npw)`` traversal.
        patches = (
            canvas.reshape(nph, p, npw, p, 3).transpose(0, 2, 1, 3, 4).reshape(num_patches, p, p, 3)
        )
        return np.ascontiguousarray(patches)

    def encode_one(self, image: Any) -> torch.Tensor:
        """One image -> ``(num_patches, T, P, P, 3)`` tensor in ``self.dtype``."""
        pil = _to_pil_rgb(image)
        sw, sh = scaled_image_dimensions(
            pil.width,
            pil.height,
            self.rescale_image_frac,
            self.rescale_image_max_upscaled_long_edge,
        )
        if (sw, sh) != pil.size:
            from PIL import Image

            pil = pil.resize((sw, sh), Image.Resampling.LANCZOS)
        arr = np.array(pil, dtype=np.uint8, copy=True)
        patches = self._patchify(arr)  # (num_patches, P, P, 3) float32
        n = int(patches.shape[0])
        t = torch.from_numpy(patches).to(self.dtype)
        # temporal duplication: (n, 1, P, P, 3) -> (n, T, P, P, 3)
        return (
            t.view(n, 1, self.patch_size, self.patch_size, 3)
            .expand(n, self.temporal_patch_size, self.patch_size, self.patch_size, 3)
            .contiguous()
        )

    def preprocess(self, images: Union[Any, Sequence[Any]]) -> Dict[str, Any]:
        """Raw images -> ``{vision_patches_bthwc, num_patches, num_tokens}``.

        ``vision_patches_bthwc`` concatenates all images' patches along dim 0
        (one contiguous slice per image, in encounter order); ``num_patches`` /
        ``num_tokens`` are per-image and equal (one token per patch).
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        per_image: List[torch.Tensor] = []
        num_patches: List[int] = []
        for img in images:
            vp = self.encode_one(img)
            per_image.append(vp)
            num_patches.append(int(vp.shape[0]))
        if len(per_image) == 1:
            vision_patches_bthwc = per_image[0]
        elif per_image:
            vision_patches_bthwc = torch.cat(per_image, dim=0)
        else:
            vision_patches_bthwc = torch.empty(0, dtype=self.dtype)
        return {
            "vision_patches_bthwc": vision_patches_bthwc,
            "num_patches": num_patches,
            "num_tokens": list(num_patches),  # hMLP: one token per patch
        }


def _resolve_vision_geometry(config: Any) -> Tuple[int, int]:
    """Return ``(patch_size, temporal_patch_size)`` from a top-level or vision
    config, defaulting to the checkpoint's ``40`` / ``2``."""
    vision_config = getattr(config, "vision_config", None)
    patch_size = getattr(vision_config, "patch_size", None)
    temporal = getattr(vision_config, "temporal_patch_size", None)
    return (
        int(patch_size) if patch_size else DEFAULT_PATCH_SIZE,
        int(temporal) if temporal else DEFAULT_TEMPORAL_PATCH_SIZE,
    )


# ===========================================================================
# Video utilities (Stage-7 / Goal 7.1) -- video is multi-frame images
# ===========================================================================
# Inkling has NO separate video tower: a video is decoded to frames, a subset of
# frames is sampled, and each sampled frame is fed as an ordinary image through
# the SAME hMLP vision tower (one ``<image>`` placeholder span per frame). So the
# only genuinely video-specific logic is choosing WHICH frames to keep; the
# per-frame preprocessing, tower forward, and fusion all reuse the accepted image
# path (:class:`InklingImagePreprocessor` + :meth:`InklingInputProcessor.assemble`
# already handle a list of images, i.e. a list of frames, in encounter order).
#
# ``sample_video_frames`` is a verbatim port of the SGLang reference
# ``sglang.srt.utils.common.sample_video_frames`` (the requested serving
# comparand; covered by SGLang ``test/registered/vlm/test_video_utils.py``), so
# frame sampling matches SGLang frame-for-frame on the same clip.


def sample_video_frames(video: Any, *, desired_fps: int, max_frames: int) -> List[int]:
    """Frame indices to keep from ``video``, matching SGLang frame-for-frame.

    ``video`` is any object exposing ``__len__`` (total decoded frames) and
    ``avg_fps`` (the clip's average FPS) -- exactly the interface SGLang's
    ``sample_video_frames`` and its ``test_video_utils.py`` ``DummyVideo`` use.
    The sampled count is bounded by the desired FPS, ``max_frames``, and the
    total frame count, with at least one frame always kept; the returned indices
    are strictly increasing (temporal order preserved). Verbatim port of SGLang
    ``sglang.srt.utils.common.sample_video_frames``.
    """
    total_frames = len(video)
    assert total_frames > 0, "Video must have at least one frame"

    avg_fps = video.avg_fps
    duration = total_frames / avg_fps if avg_fps > 0 else 0
    fps = min(desired_fps, avg_fps)

    num_frames = math.floor(duration * fps)
    num_frames = min(max_frames, num_frames, total_frames)
    num_frames = max(1, num_frames)  # At least one frame
    if num_frames == total_frames:
        return list(range(total_frames))
    else:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()


class DecodedVideo:
    """A minimal decoded-video view over an ordered list of frame images.

    Wraps already-decoded frames (PIL images, numpy arrays, or raw image bytes --
    whatever :class:`InklingImagePreprocessor` accepts) plus the clip's average
    FPS so :func:`sample_video_frames` can select a subset. This deliberately does
    NOT own codec/decoding: real serving decodes the container upstream (SGLang's
    ``encode_video``); this bring-up path takes the decoded frames directly, which
    keeps the video utility free of a third-party video-decoder dependency.
    """

    def __init__(self, frames: Sequence[Any], avg_fps: float) -> None:
        self.frames = list(frames)
        if not self.frames:
            raise ValueError("DecodedVideo requires at least one frame")
        self.avg_fps = float(avg_fps)

    def __len__(self) -> int:
        return len(self.frames)


def sample_video_as_images(
    video: "DecodedVideo", *, desired_fps: int, max_frames: int
) -> List[Any]:
    """Video -> the sampled frames as an ordered list of images.

    Selects frame indices with :func:`sample_video_frames` (SGLang-parity) and
    returns those frames in temporal order. The caller renders one ``<image>``
    content part per returned frame; the accepted multi-image
    :meth:`InklingInputProcessor.assemble` then expands each ``<image>``
    placeholder into that frame's own patch span and the shared hMLP tower + fusion
    handle the rest -- no separate video tower. Frame count and ordering are
    preserved (this is what the Stage-7 utility test asserts).
    """
    idxs = sample_video_frames(video, desired_fps=desired_fps, max_frames=max_frames)
    return [video.frames[i] for i in idxs]


def _resolve_image_token_id(config: Any) -> int:
    """The single ``<image>`` placeholder id the token stream carries per image.

    Prefer an explicit ``image_token_id`` on the (top-level or vision) config;
    fall back to the SGLang sentinel ``-101`` when the checkpoint config.json
    omits it (as the in-scope ``Inkling-NVFP4-full`` checkpoint does)."""
    for obj in (config, getattr(config, "vision_config", None)):
        tok = getattr(obj, "image_token_id", None)
        if tok is not None:
            return int(tok)
    return DEFAULT_IMAGE_TOKEN_ID


class InklingInputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    """Inkling image multimodal input processor (Stage-1 / Goal 1.2).

    Registered on ``InklingForConditionalGeneration`` via
    ``@register_input_processor`` with placeholder ``{"image": "<image>"}``.

    Inherits :class:`BaseMultimodalDummyInputsBuilder` alongside
    :class:`BaseMultimodalInputProcessor` (the SAME two-base pattern the
    in-tree VLM processors use -- Qwen2/3-VL, Mistral, Gemma3-VL, Vila,
    Phi4-MM, Llava-Next). ``OpenAIServer.__init__`` seeds media-IO defaults by
    calling ``ip.get_preferred_media_io_kwargs()`` on ANY
    ``BaseMultimodalInputProcessor`` (``serve/openai_server.py``), and the
    KV-cache encoder profiler calls ``get_mm_max_tokens_per_item()`` /
    ``get_dummy_mm_data_for_tokens()`` (``_torch/pyexecutor/_util.py``); those
    live on the dummy-inputs builder, not the base processor, so a processor
    that inherits only the base crashes ``trtllm-serve`` at startup with
    ``AttributeError: ... has no attribute 'get_preferred_media_io_kwargs'``
    (jobs 5597129/5597134). The builder's defaults are exactly right for
    image-only Inkling: ``get_preferred_media_io_kwargs`` -> ``{}`` (no
    special media-IO decode format like Qwen's video ``np`` frames) and
    ``get_mm_max_tokens_per_item`` -> ``{}`` (empty demand -> the profiler's
    ``total_demand<=0`` guard returns early and never reaches
    ``get_dummy_mm_data_for_tokens``, so image fusion keeps working exactly as
    it does under the in-process ``LLM`` API used by the MMMU runner).

    Contract (a faithful port of SGLang ``InklingMultimodalProcessor.assemble``):

      * text-only requests pass straight through (tokenize -> ids, no MM data);
      * each ``<image>`` placeholder token (the resolved ``image_token_id``
        sentinel) is expanded into ``num_patches`` tokens -- one per vision
        patch, since the hMLP emits one text-hidden row per patch;
      * a placeholder/media count mismatch, or an expanded-token vs feature-row
        count mismatch, FAILS LOUDLY (never drops or pads media silently).

    The vision tower + ``inputs_embeds`` fusion are Goals 1.3 / 1.4; this
    processor only emits the preprocessed patch features (``multimodal_data``)
    and the validated, expanded token stream those goals consume.
    """

    # Goal 1.4 runtime entry: accept a pre-tokenized ``prompt_token_ids`` stream
    # that already carries one ``image_token_id`` (200054) placeholder per image
    # plus ``multi_modal_data``, WITHOUT detokenizing (the checkpoint tokenizer
    # has no ``<image>`` -> placeholder mapping, so detokenize-and-retokenize
    # would drop it). This is the drift-free parity path used by the end-to-end
    # source_logit_replay / generation_parity tests: the SAME input_ids are fed
    # to both TRT (here, 200054) and the SGLang reference server (mapped to its
    # internal -101). See ``call_with_token_ids`` below.
    supports_token_id_mm_expansion = True

    def __init__(
        self, model_path, config, tokenizer, trust_remote_code: bool = True, **kwargs
    ) -> None:
        super().__init__(model_path, config, tokenizer, trust_remote_code, **kwargs)
        patch_size, temporal = _resolve_vision_geometry(config)
        self.image_token_id = _resolve_image_token_id(config)
        self.audio_token_id = _resolve_audio_token_id(config)
        self._dtype = torch.bfloat16
        self._preprocessor = InklingImagePreprocessor(
            patch_size=patch_size,
            temporal_patch_size=temporal,
            dtype=self._dtype,
        )
        # dMel audio preprocessor (Stage-6 / Goal 6.1). Built unconditionally so
        # an audio request works even for a checkpoint whose config omits an
        # ``audio_config`` blob (the geometry falls back to the checkpoint
        # defaults); it is only exercised when the token stream actually carries
        # an audio placeholder, so it is inert for text/image-only requests.
        self._audio_preprocessor = InklingAudioPreprocessor(**_resolve_audio_geometry(config))
        # Text-only requests must tokenize EXACTLY as the accepted text tower
        # did before this processor was registered (tiktoken special-token
        # handling, add_special_tokens, truncation, query). Delegate to the
        # stock DefaultInputProcessor so there is zero drift / no text-gate
        # regression; the image path reuses the same tokenization.
        self._text_processor = DefaultInputProcessor(
            model_path, config, tokenizer, trust_remote_code
        )

    # ---- required BaseMultimodalInputProcessor properties -----------------
    @property
    def processor(self):
        # Inkling ships no HF AutoProcessor; expose the local image
        # preprocessor so base helpers that probe ``self.processor`` work.
        return self._preprocessor

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self):
        return self._config

    @property
    def model_path(self):
        # Concrete impl of the ``BaseMultimodalDummyInputsBuilder`` abstract
        # ``model_path`` property (the base processor only stores
        # ``self._model_path`` without exposing it), so the two-base
        # ``InklingInputProcessor`` is instantiable.
        return self._model_path

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_mm_token_ids(self) -> Optional[torch.Tensor]:
        """The placeholder id(s) that mark media rows in the token stream.

        The image path expands each ``<image>`` sentinel into ``num_patches``
        copies of ``image_token_id`` and the audio path expands each ``<audio>``
        sentinel into ``num_frames`` copies of ``audio_token_id``; the model
        engine locates those rows to overwrite with tower embeddings (Goals 1.4 /
        6.1). Returned as int32 so the engine's
        ``torch.isin(input_ids, mm_token_ids)`` lookup matches both."""
        return torch.tensor([self.image_token_id, self.audio_token_id], dtype=torch.int32)

    # ---- core (pure, testable) expansion + validation ---------------------
    def assemble(
        self,
        input_ids: Sequence[int],
        image_data: Optional[Sequence[Any]] = None,
        audio_data: Optional[Sequence[Any]] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Expand ``<image>`` / ``<audio>`` placeholders and attach media features.

        Returns ``(expanded_ids, multimodal_data)`` where ``multimodal_data`` is
        ``{}`` for a text-only request, or carries an ``"image"`` and/or
        ``"audio"`` entry. Each image placeholder expands to one token per vision
        patch (hMLP emits one row per patch) and each audio placeholder to one
        token per dMel frame (the audio tower emits one row per frame). Raises
        ``ValueError`` on any placeholder/media/feature-row count mismatch
        (fail-loud contract; media is never dropped or padded silently).
        """
        input_ids = list(input_ids)
        image_data = list(image_data) if image_data else []
        audio_data = list(audio_data) if audio_data else []

        n_img_ph = sum(1 for t in input_ids if t == self.image_token_id)
        if n_img_ph != len(image_data):
            raise ValueError(
                f"InklingInputProcessor: {n_img_ph} image placeholder token(s) "
                f"(id={self.image_token_id}) in input_ids but "
                f"{len(image_data)} image(s) provided; counts must match "
                f"(media is never dropped or padded silently)."
            )
        n_aud_ph = sum(1 for t in input_ids if t == self.audio_token_id)
        if n_aud_ph != len(audio_data):
            raise ValueError(
                f"InklingInputProcessor: {n_aud_ph} audio placeholder token(s) "
                f"(id={self.audio_token_id}) in input_ids but "
                f"{len(audio_data)} audio clip(s) provided; counts must match "
                f"(media is never dropped or padded silently)."
            )

        if not image_data and not audio_data:
            return input_ids, {}  # text-only passthrough

        img_feat = self._preprocessor.preprocess(image_data) if image_data else None
        aud_feat = self._audio_preprocessor.preprocess(audio_data) if audio_data else None

        out_ids: List[int] = []
        img_offsets: List[Tuple[int, int]] = []  # (start, end) inclusive per image
        aud_offsets: List[Tuple[int, int]] = []  # (start, end) inclusive per clip
        i_img = 0
        i_aud = 0
        for tok in input_ids:
            if image_data and tok == self.image_token_id:
                n_tok = int(img_feat["num_tokens"][i_img])
                n_pat = int(img_feat["num_patches"][i_img])
                if n_tok != n_pat:
                    raise ValueError(
                        f"InklingInputProcessor: image {i_img} expands to "
                        f"{n_tok} token(s) but has {n_pat} patch/feature "
                        f"row(s); the hMLP emits exactly one token per patch."
                    )
                start = len(out_ids)
                out_ids.extend([self.image_token_id] * n_tok)
                img_offsets.append((start, start + n_tok - 1))
                i_img += 1
            elif audio_data and tok == self.audio_token_id:
                n_tok = int(aud_feat["num_tokens"][i_aud])
                n_frm = int(aud_feat["num_frames"][i_aud])
                if n_tok != n_frm:
                    raise ValueError(
                        f"InklingInputProcessor: audio {i_aud} expands to "
                        f"{n_tok} token(s) but has {n_frm} dMel frame(s); the "
                        f"audio tower emits exactly one token per frame."
                    )
                start = len(out_ids)
                out_ids.extend([self.audio_token_id] * n_tok)
                aud_offsets.append((start, start + n_tok - 1))
                i_aud += 1
            else:
                out_ids.append(int(tok))

        # Central fail-loud invariant, per modality: the number of expanded
        # placeholder tokens must equal the number of feature rows the tower emits.
        multimodal_data: Dict[str, Any] = {}
        if image_data:
            vision_patches = img_feat["vision_patches_bthwc"]
            total_rows = int(sum(img_feat["num_patches"]))
            n_mm_out = sum(1 for t in out_ids if t == self.image_token_id)
            feat_rows = int(vision_patches.shape[0]) if vision_patches.ndim else 0
            if not (n_mm_out == total_rows == feat_rows):
                raise ValueError(
                    f"InklingInputProcessor: image placeholder-token count "
                    f"({n_mm_out}) must equal feature-row count "
                    f"(sum(num_patches)={total_rows}, vision_patches rows={feat_rows})."
                )
            multimodal_data["image"] = {
                "vision_patches_bthwc": vision_patches,
                "num_patches": img_feat["num_patches"],
                "offsets": img_offsets,
            }
        if audio_data:
            dmel_bins = aud_feat["dmel_bins"]
            total_rows = int(sum(aud_feat["num_frames"]))
            n_mm_out = sum(1 for t in out_ids if t == self.audio_token_id)
            feat_rows = int(dmel_bins.shape[0]) if dmel_bins.ndim else 0
            if not (n_mm_out == total_rows == feat_rows):
                raise ValueError(
                    f"InklingInputProcessor: audio placeholder-token count "
                    f"({n_mm_out}) must equal feature-row count "
                    f"(sum(num_frames)={total_rows}, dmel_bins rows={feat_rows})."
                )
            multimodal_data["audio"] = {
                "dmel_bins": dmel_bins,
                "num_frames": aud_feat["num_frames"],
                "offsets": aud_offsets,
            }
        return out_ids, multimodal_data

    # ---- SamplingParams/TextPrompt entrypoint -----------------------------
    def call_with_text_prompt(self, inputs, sampling_params):
        """Turn a text prompt (optionally with images) into ``(ids, extra)``.

        Text-only requests are delegated verbatim to the stock
        ``DefaultInputProcessor`` -- byte-identical to the accepted text path
        (so registering this processor does not regress the text GSM8K/MMLU
        gate). Image requests tokenize the prompt the same way -- the stream
        must already carry one ``image_token_id`` placeholder per image -- then
        delegate to :meth:`assemble`. The Inkling chat renderer that injects
        those placeholders is wired at M1a/M1b (Goal 1.5); until then an image
        request with no placeholder in the stream fails loudly via the count
        check in :meth:`assemble`.
        """
        mm_data = inputs.get("multi_modal_data") or {}
        images = mm_data.get("image") or []
        if images and not isinstance(images, list):
            images = [images]
        audios = mm_data.get("audio") or []
        if audios and not isinstance(audios, list):
            audios = [audios]

        if not images and not audios:
            # Exactly the pre-registration text path (tiktoken-safe, honors
            # add_special_tokens / truncation / query). Returns (ids, None).
            return self._text_processor(inputs, sampling_params)

        token_ids, _extra = self._text_processor(inputs, sampling_params)
        expanded_ids, multimodal_data = self.assemble(token_ids, images, audios)
        # ``multimodal_data`` carries an ``"image"`` and/or ``"audio"`` entry here.
        return expanded_ids, {"multimodal_data": multimodal_data}

    # ---- pre-tokenized (-101) + image fast path ---------------------------
    def call_with_token_ids(self, inputs, sampling_params):
        """Pre-tokenized multimodal entry (``supports_token_id_mm_expansion``).

        The LLM API dispatches here (``registry.InputProcessor.__call__``) when a
        request carries ``prompt_token_ids`` + ``multi_modal_data`` and no
        ``prompt`` string. ``prompt_token_ids`` must already contain exactly one
        ``image_token_id`` (200054) placeholder per image; :meth:`assemble`
        expands each into ``num_patches`` copies and attaches the preprocessed
        ``vision_patches_bthwc`` (fail-loud on any count mismatch). This is the
        drift-free path the end-to-end parity tests use -- the identical token
        stream is fed to both this stack and the SGLang reference server (mapped
        to its internal -101), so no tokenizer re-render can desynchronize them.

        Overrides the base ``call_with_token_ids`` (which drives an HF processor
        over a synthetic dummy prompt): Inkling ships no HF AutoProcessor, and
        the sentinel is already in the stream, so the base dummy-prompt machinery
        is neither needed nor applicable.
        """
        ids = list(inputs.get("prompt_token_ids") or [])
        mm_data = inputs.get("multi_modal_data") or {}
        images = mm_data.get("image") or []
        if images and not isinstance(images, list):
            images = [images]
        audios = mm_data.get("audio") or []
        if audios and not isinstance(audios, list):
            audios = [audios]
        if not images and not audios:
            # No media -> plain token passthrough (no multimodal payload).
            return ids, None
        expanded_ids, multimodal_data = self.assemble(ids, images, audios)
        return expanded_ids, {"multimodal_data": multimodal_data}


# ===========================================================================
# HMLP vision tower (Stage-1 / Goal 1.3) -- InklingVisionModel
# ===========================================================================
# The Inkling vision encoder is a hierarchical MLP (hMLP): it repeatedly folds a
# temporal/spatial neighborhood of the ``vision_patches_bthwc`` tensor into the
# channel depth, projects with a bias-free Linear, and (for all but the last
# layer) applies RMSNorm + exact GELU. The last layer projects to the text hidden
# width (``decoder_dmodel``); a final RMSNorm then yields exactly one text-hidden
# row per input patch. This is a verbatim reimplementation of the HF
# ``InklingVisionModel`` / SGLang ``HMLPPatchEncoder`` math (identical on both
# sides); it is pure torch (Linear + RMSNorm + GELU + reshape) so no custom
# kernel is needed (parity-first, Python-first rule).
#
# For the in-scope checkpoint (temporal_patch_size=2, patch_size=40, n_layers=4,
# n_channels=3) ``plan_out_scales`` resolves to the scale progression
#   [(1,1,1,3), (1,5,5,128), (1,10,10,320), (1,40,40,4800), (2,40,40,9600)]
# giving four Linear layers 75->128, 512->320, 5120->4800, 9600->6144 plus norms
# norm_0..2 and final_norm -- matching the checkpoint ``model.visual.*`` weights
# exactly (linear_0..3, norm_0..2, final_norm).


def _prime_factors(n: int) -> List[int]:
    """Prime factors of ``n`` in ascending order (SGLang hmlp.py ``_prime_factors``)."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    factors: List[int] = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2
    if n > 1:
        factors.append(n)
    return factors


def plan_out_scales(
    temporal_patch_size: int,
    patch_size: int,
    n_layers: int,
    n_channels: int = 3,
) -> List[Tuple[int, int, int, int]]:
    """Plan the ``(time, height, width, channels)`` scale at each hMLP layer.

    Verbatim port of SGLang ``hmlp.plan_out_scales`` (identical to HF
    ``plan_out_scales``): build the candidate scale progression (spatial folds
    first, then temporal, channels rounded up to a multiple of 64), then assign
    ``n_layers + 1`` scales to the ideal log-spaced size reductions -- ``argmin``
    when ``n_layers >= len(scales)`` else a global ``linear_sum_assignment`` --
    pinning the first scale to the raw patch and the last to the full patch.
    """
    if patch_size <= 1:
        raise ValueError("patch_size must be greater than 1")

    def _round_up(x: int) -> int:
        return int(np.ceil(x / 64)) * 64

    last_h_scale = 1
    scales: List[Tuple[int, int, int, int]] = [(1, 1, 1, n_channels)]
    for pscale in _prime_factors(patch_size)[::-1]:
        last_h_scale *= pscale
        scales.append((1, last_h_scale, last_h_scale, _round_up((last_h_scale**2) * n_channels)))
    last_t_scale = 1
    for tscale in _prime_factors(temporal_patch_size)[::-1]:
        last_t_scale *= tscale
        scales.append(
            (
                last_t_scale,
                last_h_scale,
                last_h_scale,
                _round_up((last_h_scale**2) * n_channels * last_t_scale),
            )
        )

    size_reduction = np.prod(np.array(scales)[:, :-1], 1)
    log_ideal_scales = np.linspace(
        0, np.log(patch_size * patch_size * temporal_patch_size * n_channels), n_layers + 1
    )
    cost_matrix = np.abs(log_ideal_scales[:, None] - np.log(size_reduction)[None])

    if n_layers >= len(scales):
        idxs = np.argmin(cost_matrix, axis=1)
    else:
        from scipy.optimize import linear_sum_assignment

        idxs = linear_sum_assignment(cost_matrix)[1]

    assert len(idxs) >= 2
    idxs[0] = 0
    idxs[-1] = len(scales) - 1
    return [scales[i] for i in idxs]


def fold_timespace_to_depth(
    vision_patches_bthwc: torch.Tensor, t_fold: int, hw_fold: int
) -> torch.Tensor:
    """Fold a ``t_fold x hw_fold x hw_fold`` neighborhood into channel depth.

    ``(B, T, H, W, C) -> (B, T//t, H//hw, W//hw, C * t * hw**2)``. Verbatim port
    of HF/SGLang ``fold_timespace_to_depth`` (same reshape/permute order, which
    determines the exact channel interleave -- getting this wrong silently
    changes the projection input, so it is a direct copy)."""
    B, T, H, W, C = vision_patches_bthwc.shape
    assert T % t_fold == 0, f"T {T} not divisible by {t_fold}"
    assert H % hw_fold == 0, f"H {H} not divisible by {hw_fold}"
    assert W % hw_fold == 0, f"W {W} not divisible by {hw_fold}"
    t_new, h_new, w_new = T // t_fold, H // hw_fold, W // hw_fold
    x = vision_patches_bthwc.reshape(B, t_new, t_fold, h_new, hw_fold, w_new, hw_fold, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
    x = x.reshape(B, t_new, h_new, w_new, t_fold * hw_fold * hw_fold * C)
    return x


class InklingVisionRMSNorm(nn.Module):
    """RMSNorm over the last dim, matching HF ``InklingRMSNorm`` / SGLang
    ``RMSNorm`` (fp32 variance, ``eps=1e-6``). Uses ``F.rms_norm`` -- the exact
    path SGLang's ``RMSNorm`` falls back to when ``sgl_kernel`` is absent, so the
    tower is bit-comparable to the reference."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        return F.rms_norm(x, (self.hidden_size,), self.weight, self.variance_epsilon)


# ===========================================================================
# Priority-0 image-use probe (Stage-3 S3-C6 / human feedback #3 P0-B).
#
# ``INKLING_VISION_PROBE=1`` turns on env-gated, default-OFF stdout markers that
# PROVE -- from INSIDE the module, not by inspecting the request dict -- that the
# vision tower forward actually executed for a request and that its per-patch
# rows were scattered into every image placeholder position. Human feedback #3
# is explicit that "an assertion that merely inspects the input dict is NOT
# acceptable, it must prove the tower ran". The accepted text/production path
# never sets the env, so this is a zero-cost no-op there (the guard short-circuits
# before any device sync). The MMMU deterministic bs=1 runner sets it so the
# per-forward marker sequence lines up one-to-one with the scored items, and
# ``p0_verify.py`` parses the tee'd ARM log to gate the P0-B image-use criterion.
_VISION_FWD_CALLS = [0]  # monotonic tower-forward counter (per worker process)
_VISION_SCATTER_CALLS = [0]  # monotonic fusion-scatter counter (per worker process)


def vision_probe_enabled() -> bool:
    return os.environ.get("INKLING_VISION_PROBE", "0") == "1"


def _probe_rank() -> int:
    for k in ("SLURM_PROCID", "RANK", "LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None and str(v).lstrip("-").isdigit():
            return int(v)
    return 0


def emit_vision_tower_probe(in_patches: int, out: "torch.Tensor") -> None:
    """Marker emitted from INSIDE ``InklingVisionModel.forward`` proving the tower
    ran, with the input patch count, output row/dim shape, dtype/device, and a
    finiteness check. ``out_rows`` must equal the item's ``num_patches`` (one
    hMLP row per patch) -- the count evidence P0-B requires."""
    n = _VISION_FWD_CALLS[0] = _VISION_FWD_CALLS[0] + 1
    try:
        finite = bool(torch.isfinite(out).all().item())
        rows, dim = int(out.shape[0]), int(out.shape[-1])
    except Exception as e:  # noqa: BLE001
        print(
            f"INKLING_VISION_FWD_TOWER rank={_probe_rank()} call={n} "
            f"in_patches={int(in_patches)} ERROR={e!r}",
            flush=True,
        )
        return
    print(
        f"INKLING_VISION_FWD_TOWER rank={_probe_rank()} call={n} "
        f"in_patches={int(in_patches)} out_rows={rows} out_dim={dim} "
        f"dtype={out.dtype} dev={out.device} finite={finite}",
        flush=True,
    )


def emit_vision_scatter_probe(n_mm_idx: int, n_vision_rows: int) -> None:
    """Marker emitted from INSIDE the multimodal fusion after
    ``fuse_input_embeds`` proving every image placeholder position was filled by a
    vision row: ``n_mm_idx`` (placeholder positions) must equal ``n_vision_rows``
    (tower output rows). A mismatch is exactly the off-by-one / dropped-row scatter
    bug feedback #3 V4 asks us to rule out."""
    n = _VISION_SCATTER_CALLS[0] = _VISION_SCATTER_CALLS[0] + 1
    match = int(n_mm_idx) == int(n_vision_rows)
    print(
        f"INKLING_VISION_FWD_SCATTER rank={_probe_rank()} call={n} "
        f"n_mm_idx={int(n_mm_idx)} n_vision_rows={int(n_vision_rows)} "
        f"match={match}",
        flush=True,
    )


# ===========================================================================
# Human feedback #21.1a -- LIVE production vision-tower INTERNALS dump (B1-B4).
#
# ``INKLING_VISION_DUMP=<base>`` turns on an env-gated, default-OFF tensor dump of
# the tower interior (fold / linear_i / norm_i / final_norm / visual_out) from
# INSIDE ``InklingVisionModel.forward`` during a LIVE TP=4 production run, so the
# fb21 analyzer can compare the tower INTERNALS -- not just the tower output row
# (C4) -- against the SGLang ``visual.vision_encoder.*`` forward-hook capture.
# Sections A/B of the fb15 walk proved the two ports byte-identical only as
# CPU_SOURCE_REPLAY (feedback #16); this makes the internals a LIVE_RUNTIME
# comparison. The capture keys mirror the SGLang ``make_visual_hook`` capture
# points EXACTLY (linear_0..3 = layers.linear_i out, norm_0..2 = layers.norm_i out,
# final_norm = final_norm out, visual_out = tower output rows) so the analyzer
# compares like with like. Each TP worker writes its own rank file; the tower is
# replicated across ranks, so rank 0 carries the full interior. Default-off: the
# accepted text/production path never sets the env, so this is a zero-cost no-op
# there (the guard short-circuits before any clone/host-copy).
_VISION_DUMP_CALLS = [0]  # monotonic tower-forward dump counter (per worker process)


def vision_dump_base() -> "Optional[str]":
    """Base path for the env-gated tower-internal tensor dump, or ``None`` when off."""
    return os.environ.get("INKLING_VISION_DUMP") or None


def emit_vision_tower_dump(base: str, caps: Dict[str, "torch.Tensor"], num_patches: int) -> None:
    """Save the captured tower internals to ``<base>.call<n>.rank<r>.pt`` (feedback #21.1a).
    Wrapped so a dump failure prints a marker but never breaks the production forward."""
    try:
        n = _VISION_DUMP_CALLS[0] = _VISION_DUMP_CALLS[0] + 1
        r = _probe_rank()
        path = f"{base}.call{n}.rank{r}.pt"
        meta = {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "max_abs": float(v.abs().max()) if v.numel() else 0.0,
            }
            for k, v in caps.items()
        }
        torch.save(
            {"num_patches": int(num_patches), "call": n, "rank": r, "internals": caps, "meta": meta},
            path,
        )
        print(
            f"INKLING_VISION_DUMP_SAVED rank={r} call={n} num_patches={int(num_patches)} "
            f"keys={sorted(caps)} -> {path}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"INKLING_VISION_DUMP_ERROR rank={_probe_rank()} {e!r}", flush=True)


class InklingVisionModel(nn.Module):
    """Inkling hMLP vision tower: ``vision_patches_bthwc`` -> one text-hidden row
    per patch (Goal 1.3). Reimplements HF ``InklingVisionModel`` / SGLang
    ``HMLPPatchEncoder`` (identical math) in pure torch.

    Module tree mirrors the checkpoint ``model.visual.*`` layout exactly:
    ``layers.linear_{i}`` (bias-free Linear), ``layers.norm_{i}`` (RMSNorm, all
    but the last layer), and ``final_norm`` (present when ``use_vision_norm``).
    """

    def __init__(self, vision_config: Any) -> None:
        super().__init__()
        self.decoder_dmodel = int(_require_vc(vision_config, "decoder_dmodel"))
        self.patch_size = int(getattr(vision_config, "patch_size", DEFAULT_PATCH_SIZE))
        self.temporal_patch_size = int(
            getattr(vision_config, "temporal_patch_size", DEFAULT_TEMPORAL_PATCH_SIZE)
        )
        self.n_channels = int(getattr(vision_config, "n_channels", 3))
        self.n_layers = int(getattr(vision_config, "n_layers", 4))
        self.use_vision_norm = bool(getattr(vision_config, "use_vision_norm", True))

        self.scales = plan_out_scales(
            self.temporal_patch_size, self.patch_size, self.n_layers, self.n_channels
        )
        self.layers = nn.ModuleDict()
        for i, (start, end) in enumerate(zip(self.scales[:-1], self.scales[1:])):
            shuffle_mult = (end[0] // start[0]) * (end[1] // start[1]) * (end[2] // start[2])
            in_dim = start[3] * shuffle_mult
            if i == self.n_layers - 1:
                self.layers[f"linear_{i}"] = nn.Linear(in_dim, self.decoder_dmodel, bias=False)
            else:
                self.layers[f"linear_{i}"] = nn.Linear(in_dim, end[3], bias=False)
                self.layers[f"norm_{i}"] = InklingVisionRMSNorm(end[3])
        self.final_norm = (
            InklingVisionRMSNorm(self.decoder_dmodel) if self.use_vision_norm else None
        )

    def forward(self, vision_patches_bthwc: torch.Tensor) -> torch.Tensor:
        """``(num_patches, T, H, W, C)`` -> ``(num_patches, decoder_dmodel)``."""
        num_patches = vision_patches_bthwc.shape[0]
        x = vision_patches_bthwc
        # feedback #21.1a: env-gated, default-OFF LIVE tower-internals capture. ``caps`` stays None
        # (zero cost) unless INKLING_VISION_DUMP is set; the capture keys mirror the SGLang
        # ``make_visual_hook`` points exactly so fb21 compares like with like.
        _dump = vision_dump_base()
        caps: Optional[Dict[str, torch.Tensor]] = {} if _dump else None
        for i, (start, end) in enumerate(zip(self.scales[:-1], self.scales[1:])):
            t_fold = end[0] // start[0]
            hw_fold = end[1] // start[1]
            if hw_fold > 1 or t_fold > 1:
                x = fold_timespace_to_depth(x, t_fold, hw_fold)
                if caps is not None:
                    caps[f"fold_{i}"] = x.detach().float().cpu().clone()
            x = self.layers[f"linear_{i}"](x)
            if caps is not None:
                caps[f"linear_{i}"] = x.detach().float().cpu().clone()
            if i < self.n_layers - 1:
                x = self.layers[f"norm_{i}"](x)
                if caps is not None:
                    caps[f"norm_{i}"] = x.detach().float().cpu().clone()
                x = F.gelu(x)
                if caps is not None:
                    caps[f"gelu_{i}"] = x.detach().float().cpu().clone()
        if self.final_norm is not None:
            x = self.final_norm(x)
            if caps is not None:
                caps["final_norm"] = x.detach().float().cpu().clone()
        out = x.reshape(num_patches, -1)
        if caps is not None:
            caps["visual_out"] = out.detach().float().cpu().clone()
            emit_vision_tower_dump(_dump, caps, num_patches)
        if vision_probe_enabled():
            emit_vision_tower_probe(num_patches, out)
        return out

    @torch.no_grad()
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load ``model.visual.*`` checkpoint tensors into this tower.

        Accepts either full ``model.visual.layers.linear_0.weight`` keys or
        already-stripped ``layers.linear_0.weight`` keys. Loading is strict: the
        module tree mirrors the checkpoint naming exactly, so a missing or extra
        key fails loudly rather than silently leaving a layer at init."""
        prefix = "model.visual."
        mapped = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in weights.items()}
        if any(p.is_meta for p in self.parameters()):
            # Built under a meta-device init context (the full-model load path):
            # copy_ into a meta tensor is illegal, so assign the checkpoint
            # tensors directly (they carry the checkpoint's bf16 dtype).
            self.load_state_dict(mapped, strict=True, assign=True)
        else:
            target_dtype = self.layers["linear_0"].weight.dtype
            mapped = {
                k: (v.to(target_dtype) if v.dtype != target_dtype else v) for k, v in mapped.items()
            }
            self.load_state_dict(mapped, strict=True)


def _require_vc(vision_config: Any, name: str) -> Any:
    val = getattr(vision_config, name, None) if vision_config is not None else None
    if val is None:
        raise ValueError(
            f"InklingVisionModel: required vision_config field {name!r} is "
            f"missing; it must be set so the hMLP geometry matches the model."
        )
    return val


# ===========================================================================
# Audio dMel preprocessing (Stage-6 / Goal 6.1) -- InklingAudioPreprocessor
# ===========================================================================
# A faithful port of the SGLang reference
# ``sglang.srt.multimodal.inkling.feature_extraction`` (the requested NVFP4
# serving comparand): raw waveform -> log-mel spectrogram (Slaney mel scale,
# area-normalized) -> per-bin nearest-center quantization into ``num_dmel_bins``
# discrete levels ("dMel"). One STFT frame == one audio token
# (``hop_length == audio_token_duration_s * sample_rate``), matching the audio
# tower which emits one text-hidden row per dMel frame. Pure numpy + torch (the
# mel basis in numpy float64, the STFT/quantization in torch); ``soundfile`` /
# ``torchaudio`` are imported lazily and only when decoding raw file bytes, so
# passing an already-decoded waveform array needs neither dependency.


def _audio_hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    """Slaney mel scale (librosa/torchaudio convention), verbatim from the SGLang
    ``feature_extraction._hz_to_mel``."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = frequencies / f_sp
    log = min_log_mel + np.log(np.maximum(frequencies, min_log_hz) / min_log_hz) / logstep
    return np.where(frequencies >= min_log_hz, log, linear)


def _audio_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    """Inverse Slaney mel scale, verbatim from SGLang ``_mel_to_hz``."""
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = mels * f_sp
    log = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return np.where(mels >= min_log_mel, log, linear)


def audio_mel_basis(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Slaney area-normalized mel filterbank ``(n_mels, n_fft//2+1)`` (float32).

    Verbatim port of SGLang ``feature_extraction._mel_basis`` so the dMel bins
    match the reference on the same clip."""
    fft_bins = n_fft // 2 + 1
    fft_freqs = np.arange(fft_bins, dtype=np.float64) * sample_rate / n_fft
    mel_edges = _audio_mel_to_hz(
        np.linspace(
            _audio_hz_to_mel(np.array([0.0]))[0],
            _audio_hz_to_mel(np.array([sample_rate / 2.0]))[0],
            n_mels + 2,
            dtype=np.float64,
        )
    )
    mel_widths = np.diff(mel_edges)
    lower = (fft_freqs[None, :] - mel_edges[:-2, None]) / mel_widths[:-1, None]
    upper = (mel_edges[2:, None] - fft_freqs[None, :]) / mel_widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    # Slaney area normalization.
    weights *= (2.0 / (mel_edges[2:] - mel_edges[:-2]))[:, None]
    return np.ascontiguousarray(weights.astype(np.float32, copy=False))


def _audio_to_exact_int(value: float, name: str, tolerance: float = 1e-6) -> int:
    rounded = round(value)
    if abs(value - rounded) > tolerance:
        raise ValueError(f"{name} must resolve to an integer sample count, got {value}")
    return int(rounded)


class InklingAudioPreprocessor:
    """Raw audio -> Inkling dMel bin tensor for the audio tower (Goal 6.1).

    Numerically matches SGLang ``InklingAudioFeatureExtractor`` (defaults
    ``sample_rate=16000``, ``window_size_multiplier=2.0``, ``n_mels=80``,
    ``num_dmel_bins=16``, ``dmel_min_value=-7.0``, ``dmel_max_value=2.0``,
    ``audio_token_duration_s=0.05``). ``encode_one`` accepts an already-decoded
    mono waveform (``torch.Tensor`` / ``np.ndarray``) OR raw file bytes / a local
    path (decoded lazily via ``soundfile`` + ``torchaudio`` resample).
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
        window_size_multiplier: float = DEFAULT_AUDIO_WINDOW_SIZE_MULTIPLIER,
        n_fft: Optional[int] = None,
        n_mels: int = DEFAULT_AUDIO_N_MELS,
        num_dmel_bins: int = DEFAULT_AUDIO_NUM_DMEL_BINS,
        dmel_min_value: float = DEFAULT_AUDIO_DMEL_MIN_VALUE,
        dmel_max_value: float = DEFAULT_AUDIO_DMEL_MAX_VALUE,
        audio_token_duration_s: float = DEFAULT_AUDIO_TOKEN_DURATION_S,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.window_size_multiplier = float(window_size_multiplier)
        self.n_fft = int(n_fft) if n_fft else None
        self.n_mels = int(n_mels)
        self.num_dmel_bins = int(num_dmel_bins)
        self.dmel_min_value = float(dmel_min_value)
        self.dmel_max_value = float(dmel_max_value)
        self.audio_token_duration_s = float(audio_token_duration_s)
        self._mel_basis_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def _mel_basis(self, n_fft: int) -> torch.Tensor:
        key = (self.sample_rate, n_fft, self.n_mels)
        cached = self._mel_basis_cache.get(key)
        if cached is None:
            cached = torch.from_numpy(audio_mel_basis(self.sample_rate, n_fft, self.n_mels))
            self._mel_basis_cache[key] = cached
        return cached

    def _to_waveform(self, audio: Any) -> torch.Tensor:
        """Coerce one audio input into a 1-D mono float32 waveform at ``sample_rate``."""
        if isinstance(audio, torch.Tensor):
            wav = audio.detach().to(torch.float32)
            return wav.mean(dim=-1) if wav.ndim > 1 else wav.reshape(-1)
        if isinstance(audio, np.ndarray):
            wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
            return wav.mean(dim=-1) if wav.ndim > 1 else wav.reshape(-1)
        # Raw file bytes / path: decode lazily (matches SGLang ``_decode_audio``).
        import io as _io

        import soundfile as sf  # lazy: only real-file decode needs it

        if isinstance(audio, (bytes, bytearray, memoryview)):
            raw = bytes(audio)
        elif hasattr(audio, "read"):
            raw = audio.read()
        elif isinstance(audio, str):
            path = audio[len("file://") :] if audio.startswith("file://") else audio
            with open(path, "rb") as f:
                raw = f.read()
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio)!r}")
        samples, src_sr = sf.read(_io.BytesIO(raw), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        wav = torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))
        if int(src_sr) != self.sample_rate:
            import torchaudio.functional as AF  # lazy: only resampling needs it

            wav = AF.resample(wav, orig_freq=int(src_sr), new_freq=self.sample_rate)
        return wav

    def _dmel_bins(self, audio: torch.Tensor) -> torch.Tensor:
        """1-D float32 waveform -> ``(n_frames, n_mels)`` int32 dMel bins.

        Verbatim port of SGLang ``feature_extraction._dmel_bins``."""
        hop_length = _audio_to_exact_int(
            self.audio_token_duration_s * self.sample_rate,
            "audio_token_duration_s * sample_rate",
        )
        window_size = _audio_to_exact_int(
            self.audio_token_duration_s * self.window_size_multiplier * self.sample_rate,
            "audio_token_duration_s * window_size_multiplier * sample_rate",
        )
        n_fft = self.n_fft or window_size
        if hop_length <= 0 or window_size <= 0 or n_fft <= 0:
            raise ValueError("audio hop length, window size, and n_fft must be positive")
        if audio.numel() == 0:
            return torch.empty((0, self.n_mels), dtype=torch.int32)

        right_pad = math.ceil(audio.numel() / hop_length) * hop_length - audio.numel()
        left_pad = max(n_fft - hop_length, 0)
        audio = F.pad(audio, (left_pad, right_pad))

        window = torch.hann_window(window_size, periodic=True, dtype=torch.float32)
        spec = torch.stft(
            audio.unsqueeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_size,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec_ri = torch.view_as_real(spec)
        magnitude = (
            (spec_ri[..., 0].square() + spec_ri[..., 1].square()).clamp_min(1e-10).sqrt().squeeze(0)
        )
        mel = self._mel_basis(n_fft).matmul(magnitude).clamp_min(1e-10).log10()
        mel = mel.to(torch.float64).clamp(min=self.dmel_min_value, max=self.dmel_max_value)
        bin_centers = torch.linspace(
            self.dmel_min_value, self.dmel_max_value, self.num_dmel_bins, dtype=torch.float64
        )
        dmel_bins = (mel.unsqueeze(-1) - bin_centers).abs().argmin(dim=-1)
        return dmel_bins.to(torch.int32).T.contiguous()

    def encode_one(self, audio: Any) -> torch.Tensor:
        """One audio clip -> ``(n_frames, n_mels)`` int32 dMel bins."""
        return self._dmel_bins(self._to_waveform(audio))

    def preprocess(self, audios: Union[Any, Sequence[Any]]) -> Dict[str, Any]:
        """Raw audio -> ``{dmel_bins, num_frames, num_tokens}``.

        ``dmel_bins`` concatenates all clips' frames along dim 0 (one contiguous
        slice per clip, in encounter order); ``num_frames`` / ``num_tokens`` are
        per-clip and equal (one audio token per dMel frame)."""
        if not isinstance(audios, (list, tuple)):
            audios = [audios]
        per_clip: List[torch.Tensor] = []
        num_frames: List[int] = []
        for a in audios:
            bins = self.encode_one(a)
            per_clip.append(bins)
            num_frames.append(int(bins.shape[0]))
        if len(per_clip) == 1:
            dmel_bins = per_clip[0]
        elif per_clip:
            dmel_bins = torch.cat(per_clip, dim=0)
        else:
            dmel_bins = torch.empty((0, self.n_mels), dtype=torch.int32)
        return {
            "dmel_bins": dmel_bins,
            "num_frames": num_frames,
            "num_tokens": list(num_frames),  # one audio token per dMel frame
        }


def _resolve_audio_geometry(config: Any) -> Dict[str, Any]:
    """Return the dMel preprocessing kwargs from a top-level or audio config,
    defaulting to the checkpoint/``processor_config.json`` values."""
    ac = getattr(config, "audio_config", None)

    def _get(name, default):
        val = getattr(ac, name, None)
        return default if val is None else val

    return {
        "n_mels": int(_get("n_mel_bins", DEFAULT_AUDIO_N_MELS)),
        "num_dmel_bins": int(_get("mel_vocab_size", DEFAULT_AUDIO_NUM_DMEL_BINS)),
        "dmel_min_value": float(_get("dmel_min_value", DEFAULT_AUDIO_DMEL_MIN_VALUE)),
        "dmel_max_value": float(_get("dmel_max_value", DEFAULT_AUDIO_DMEL_MAX_VALUE)),
    }


def _resolve_audio_token_id(config: Any) -> int:
    """The single ``<audio>`` placeholder id the token stream carries per clip.

    Prefer an explicit ``audio_token_id`` on the (top-level or audio) config; fall
    back to the in-vocab chat-template token ``<|unused_200053|>`` (200053)."""
    for obj in (config, getattr(config, "audio_config", None)):
        tok = getattr(obj, "audio_token_id", None)
        if tok is not None:
            return int(tok)
    return DEFAULT_AUDIO_TOKEN_ID


def _require_ac(audio_config: Any, name: str) -> Any:
    val = getattr(audio_config, name, None) if audio_config is not None else None
    if val is None:
        raise ValueError(
            f"InklingAudioModel: required audio_config field {name!r} is missing; "
            f"it must be set so the dMel tower geometry matches the model."
        )
    return val


class InklingAudioModel(nn.Module):
    """Inkling dMel audio tower: dMel bins -> one text-hidden row per frame (Goal 6.1).

    Reimplements the SGLang ``InklingAudio`` / HF ``InklingAudioModel`` math
    (identical) in pure torch. ``audio_mode='dmel'``: each of the ``n_mel_bins``
    per-frame bin indices selects a row from a shared
    ``nn.Embedding(n_mel_bins * mel_vocab_size, decoder_dmodel)`` codebook (bin
    ``m`` occupies rows ``[m*mel_vocab_size, (m+1)*mel_vocab_size)``); the
    per-bin embeddings are summed per frame, then an optional ``final_norm``
    (RMSNorm, ``use_audio_norm``) yields one ``decoder_dmodel`` row per frame.

    Module tree mirrors the checkpoint ``model.audio.*`` layout exactly:
    ``encoder`` (the codebook embedding) and ``final_norm`` (RMSNorm, present
    when ``use_audio_norm``).
    """

    def __init__(self, audio_config: Any) -> None:
        super().__init__()
        audio_mode = getattr(audio_config, "audio_mode", "dmel")
        if audio_mode != "dmel":
            raise ValueError(
                f"InklingAudioModel supports audio_mode='dmel' only, got {audio_mode!r}."
            )
        self.decoder_dmodel = int(_require_ac(audio_config, "decoder_dmodel"))
        self.n_mel_bins = int(getattr(audio_config, "n_mel_bins", DEFAULT_AUDIO_N_MELS))
        self.mel_vocab_size = int(
            getattr(audio_config, "mel_vocab_size", DEFAULT_AUDIO_NUM_DMEL_BINS)
        )
        self.use_audio_norm = bool(getattr(audio_config, "use_audio_norm", True))
        self.encoder = nn.Embedding(self.n_mel_bins * self.mel_vocab_size, self.decoder_dmodel)
        self.final_norm = (
            InklingVisionRMSNorm(self.decoder_dmodel) if self.use_audio_norm else None
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """``(n_frames, n_mel_bins)`` dMel bin indices -> ``(n_frames, decoder_dmodel)``."""
        dev = self.encoder.weight.device
        if audio_features.numel() == 0:
            return torch.zeros((0, self.decoder_dmodel), dtype=self.encoder.weight.dtype, device=dev)
        if audio_features.shape[-1] != self.n_mel_bins:
            raise ValueError(
                f"InklingAudioModel: audio_features last dim {audio_features.shape[-1]} "
                f"!= n_mel_bins {self.n_mel_bins}."
            )
        af = audio_features.to(device=dev)
        n_frames = int(af.shape[0])
        # bin ``m`` -> codebook rows [m*mel_vocab_size, (m+1)*mel_vocab_size)
        bin_offsets = torch.arange(self.n_mel_bins, device=dev) * self.mel_vocab_size
        idx = bin_offsets.unsqueeze(0) + af.to(torch.long)  # (n_frames, n_mel_bins)
        hidden = (
            self.encoder(idx.reshape(-1)).reshape(n_frames, self.n_mel_bins, -1).sum(dim=1)
        )  # (n_frames, decoder_dmodel)
        if self.final_norm is not None:
            hidden = self.final_norm(hidden)
        return hidden

    @torch.no_grad()
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load ``model.audio.*`` checkpoint tensors into this tower.

        Accepts either full ``model.audio.encoder.weight`` keys or already-stripped
        ``encoder.weight`` keys. Strict: the module tree mirrors the checkpoint
        naming exactly (``encoder.weight``, ``final_norm.weight``), so a missing or
        extra key fails loudly rather than silently leaving a layer at init."""
        prefix = "model.audio."
        mapped = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in weights.items()}
        if any(p.is_meta for p in self.parameters()):
            self.load_state_dict(mapped, strict=True, assign=True)
        else:
            target_dtype = self.encoder.weight.dtype
            mapped = {
                k: (v.to(target_dtype) if v.dtype != target_dtype else v) for k, v in mapped.items()
            }
            self.load_state_dict(mapped, strict=True)
