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
"""Inkling vision preprocessing + multimodal input processor (Stage-1 / Goal 1.2).

Scope of this module (image path only):
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
# Audio placeholder (Stage 2). SGLang uses -102 internally; the audio content
# token is resolved from config/chat-template when the audio tower lands.
DEFAULT_AUDIO_TOKEN_ID = -102


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
        self._dtype = torch.bfloat16
        self._preprocessor = InklingImagePreprocessor(
            patch_size=patch_size,
            temporal_patch_size=temporal,
            dtype=self._dtype,
        )
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
        """The placeholder id(s) that mark vision rows in the token stream.

        The image path expands each ``<image>`` sentinel into ``num_patches``
        copies of ``image_token_id``; the model engine locates those rows to
        overwrite with vision embeddings (Goal 1.4). Returned as int32 so the
        engine's ``torch.isin(input_ids, mm_token_ids)`` lookup matches.
        """
        return torch.tensor([self.image_token_id], dtype=torch.int32)

    # ---- core (pure, testable) expansion + validation ---------------------
    def assemble(
        self,
        input_ids: Sequence[int],
        image_data: Optional[Sequence[Any]] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Expand ``<image>`` placeholders in ``input_ids`` and attach features.

        Returns ``(expanded_ids, multimodal_data)`` where ``multimodal_data`` is
        ``{}`` for a text-only request or ``{"image": {...}}`` otherwise. Raises
        ``ValueError`` on any count mismatch (fail-loud contract).
        """
        input_ids = list(input_ids)
        image_data = list(image_data) if image_data else []

        n_img_ph = sum(1 for t in input_ids if t == self.image_token_id)
        if n_img_ph != len(image_data):
            raise ValueError(
                f"InklingInputProcessor: {n_img_ph} image placeholder token(s) "
                f"(id={self.image_token_id}) in input_ids but "
                f"{len(image_data)} image(s) provided; counts must match "
                f"(media is never dropped or padded silently)."
            )

        if not image_data:
            return input_ids, {}  # text-only passthrough

        feat = self._preprocessor.preprocess(image_data)
        num_patches: List[int] = feat["num_patches"]
        num_tokens: List[int] = feat["num_tokens"]
        vision_patches = feat["vision_patches_bthwc"]

        out_ids: List[int] = []
        offsets: List[Tuple[int, int]] = []  # (start, end) inclusive per image
        i_img = 0
        for tok in input_ids:
            if tok == self.image_token_id:
                n_tok = int(num_tokens[i_img])
                n_pat = int(num_patches[i_img])
                if n_tok != n_pat:
                    raise ValueError(
                        f"InklingInputProcessor: image {i_img} expands to "
                        f"{n_tok} token(s) but has {n_pat} patch/feature "
                        f"row(s); the hMLP emits exactly one token per patch."
                    )
                start = len(out_ids)
                out_ids.extend([self.image_token_id] * n_tok)
                offsets.append((start, start + n_tok - 1))
                i_img += 1
            else:
                out_ids.append(int(tok))

        # Central fail-loud invariant: the number of expanded placeholder tokens
        # must equal the number of vision feature rows the tower will emit.
        total_rows = int(sum(num_patches))
        n_mm_out = sum(1 for t in out_ids if t == self.image_token_id)
        feat_rows = int(vision_patches.shape[0]) if vision_patches.ndim else 0
        if not (n_mm_out == total_rows == feat_rows):
            raise ValueError(
                f"InklingInputProcessor: placeholder-token count ({n_mm_out}) "
                f"must equal feature-row count "
                f"(sum(num_patches)={total_rows}, "
                f"vision_patches rows={feat_rows})."
            )

        multimodal_data = {
            "image": {
                "vision_patches_bthwc": vision_patches,
                "num_patches": num_patches,
                "offsets": offsets,
            }
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

        if not images:
            # Exactly the pre-registration text path (tiktoken-safe, honors
            # add_special_tokens / truncation / query). Returns (ids, None).
            return self._text_processor(inputs, sampling_params)

        token_ids, _extra = self._text_processor(inputs, sampling_params)
        expanded_ids, multimodal_data = self.assemble(token_ids, images)
        # ``multimodal_data`` is ``{"image": {...}}`` here (images are present).
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
        if not images:
            # No media -> plain token passthrough (no multimodal payload).
            return ids, None
        expanded_ids, multimodal_data = self.assemble(ids, images)
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
        for i, (start, end) in enumerate(zip(self.scales[:-1], self.scales[1:])):
            t_fold = end[0] // start[0]
            hw_fold = end[1] // start[1]
            if hw_fold > 1 or t_fold > 1:
                x = fold_timespace_to_depth(x, t_fold, hw_fold)
            x = self.layers[f"linear_{i}"](x)
            if i < self.n_layers - 1:
                x = self.layers[f"norm_{i}"](x)
                x = F.gelu(x)
        if self.final_norm is not None:
            x = self.final_norm(x)
        out = x.reshape(num_patches, -1)
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
