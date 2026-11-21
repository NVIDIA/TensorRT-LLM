# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1's input processor: turns an image prompt into token ids and the packed MoonViT inputs.

``routing.input_processor_class`` returns :class:`K31InputProcessor` and registers the prompt placeholders of
:data:`KIMI_K31_PLACEHOLDER_METADATA` for model_type ``kimi_k31``. The driver process builds the processor without
building the model, so this module imports no target. :class:`K31InputProcessor` runs the checkpoint's HF
``AutoProcessor`` (Moon-Patch tiling of each image), expands each media placeholder to one token per output embedding,
and states the encoder items (one per image: its patches and output embeddings). Kimi K3.1 takes images only; a
target that runs no vision tower refuses them in its own admission check, before the request is scheduled. The
processed images (``multimodal_data["image"]``: ``pixel_values`` and ``image_grid_thw``) are what the targets' vision
tower (``K31VisionTower`` in their ``modeling.py``) reads, and :func:`moonvit_max_image_grid` is the largest image
the processor produces.

A prompt without images is tokenized by :class:`KimiPromptEncoder`: a Kimi checkpoint ships a tiktoken tokenizer
(``TikTokenTokenizer`` in its ``tokenization_kimi.py``), and Hugging Face's generic slow path splits the added tokens
out of the text character by character in Python and maps every BPE token through string lookups. The encoder splits
the text on the added tokens with one regex and encodes the text between them with the tokenizer's native tiktoken
``encode``, which gives the same ids several times faster. A tokenizer or an option whose split a regex cannot
reproduce exactly takes Hugging Face's path.
"""

from __future__ import annotations

import functools
import math
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from tensorrt_llm.inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
)
from tensorrt_llm.inputs.multimodal import MultimodalItemMetadata
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.sampling_params import SamplingParams

#: The media placeholder token (``<|media_pad|>``) when the config names none; an in-vocabulary id.
MEDIA_PLACEHOLDER_TOKEN_ID = 163605
#: The text vocabulary size when the text config names none.
_VOCAB_SIZE = 163840


# Processor limits used when the HF image processor has no ``media_proc_cfg``.
_DEFAULT_IN_PATCH_LIMIT = 16384
_DEFAULT_PATCH_LIMIT_ON_ONE_SIDE = 512


def _as_pil_image(image: Union[Image.Image, torch.Tensor]) -> Image.Image:
    """Return the PIL image the HF processor accepts; the serving media loader yields tensors."""
    if isinstance(image, torch.Tensor):
        from torchvision.transforms.functional import to_pil_image

        return to_pil_image(image)
    return image


# ---------------------------------------------------------------------------
# Processor geometry
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _navit_max_patches(
    patch_size: int, merge_size: int, in_patch_limit: int, patch_limit_on_one_side: int
) -> int:
    """Return the most patches the HF ``navit_resize_image`` produces for any image size.

    The processor scales ``(w, h)`` by ``min(1, sqrt(in_patch_limit / (w // p * h // p)),
    side / w, side / h)``, where ``p`` is the patch size and ``side`` is
    ``patch_limit_on_one_side * p`` pixels, then pads each side up to a multiple of
    ``merge_size * p``. Flooring ``w // p`` and the padding let a grid exceed
    ``in_patch_limit``. All images with ``w // p == a`` and ``h // p == b`` (``a = 1`` also
    covers ``w < p``) resize to at most the cell's largest width and height under the largest
    scale any of them gets, so the largest cell bound bounds every image. The cells cover
    sides up to ``4 * patch_limit_on_one_side`` patches; the side limit scales a larger image
    down by more than 4x on that side.
    """
    factor = merge_size * patch_size
    side_px = patch_limit_on_one_side * patch_size
    cells = np.arange(1, 4 * patch_limit_on_one_side + 1, dtype=np.int64)
    largest_px = patch_size * cells + patch_size - 1
    smallest_px = np.where(cells > 1, patch_size * cells, 1)

    def padded_side(px: np.ndarray, scale: np.ndarray) -> np.ndarray:
        resized = np.clip(np.floor(px * scale).astype(np.int64), 1, side_px)
        return merge_size * -(-resized // factor)

    best = 0
    # The bound is symmetric in width and height, so cells with b >= a suffice.
    for index, a in enumerate(cells.tolist()):
        scale = np.minimum(
            np.minimum(1.0, np.sqrt(in_patch_limit / (a * cells[index:]))),
            np.minimum(side_px / smallest_px[index], side_px / smallest_px[index:]),
        )
        patches = padded_side(largest_px[index], scale) * padded_side(largest_px[index:], scale)
        best = max(best, int(patches.max()))
    return best


class _MoonViTGeometry(NamedTuple):
    """HF processor limits that bound MoonViT grids; all but ``patch_size`` are in patches."""

    # Pixels per patch side.
    patch_size: int
    # Padding unit of ``navit_resize_image``.
    resize_merge_size: int
    # Spatial kernel ``(kh, kw)`` of the encoder's patch merger.
    merge_kernel_size: Tuple[int, int]
    in_patch_limit: int
    patch_limit_on_one_side: int


def _moonvit_geometry(
    media_proc_cfg: Dict[str, Any],
    merge_kernel_size: Tuple[int, int],
    vision_patch_size: Optional[int],
) -> _MoonViTGeometry:
    """The grid limits of an HF image processor ``media_proc_cfg`` and the encoder's merge kernel."""
    cfg = media_proc_cfg
    return _MoonViTGeometry(
        patch_size=int(cfg.get("patch_size") or vision_patch_size or 14),
        resize_merge_size=int(cfg.get("merge_kernel_size") or merge_kernel_size[0]),
        merge_kernel_size=merge_kernel_size,
        in_patch_limit=int(cfg.get("in_patch_limit") or _DEFAULT_IN_PATCH_LIMIT),
        patch_limit_on_one_side=int(
            cfg.get("patch_limit_on_one_side") or _DEFAULT_PATCH_LIMIT_ON_ONE_SIDE
        ),
    )


def _max_image_patches(geometry: _MoonViTGeometry) -> int:
    """The most patches one processed image can have."""
    return _navit_max_patches(
        geometry.patch_size,
        geometry.resize_merge_size,
        geometry.in_patch_limit,
        geometry.patch_limit_on_one_side,
    )


def _largest_grid(geometry: _MoonViTGeometry, max_patches: int) -> Optional[Tuple[int, int]]:
    """Return the legal ``(h, w)`` with the most patches, at most ``max_patches``.

    Sides are multiples of the merge kernel and at most ``patch_limit_on_one_side``. Among
    the grids with the most patches, the most square one is returned.
    """
    kernel_h, kernel_w = geometry.merge_kernel_size
    side_limit = geometry.patch_limit_on_one_side
    best: Optional[Tuple[int, int]] = None
    for height in range(kernel_h, side_limit + 1, kernel_h):
        width = min(side_limit, max_patches // height) // kernel_w * kernel_w
        if width < kernel_w:
            break
        if best is None or (height * width, -abs(height - width)) > (
            best[0] * best[1],
            -abs(best[0] - best[1]),
        ):
            best = (height, width)
    return best


def moonvit_max_image_grid(
    media_proc_cfg: Dict[str, Any],
    merge_kernel_size: Tuple[int, int],
    vision_patch_size: Optional[int],
) -> Optional[Tuple[int, int, int]]:
    """The ``(t, h, w)`` grid of the largest image the HF processor of ``media_proc_cfg`` produces: the image
    largest legal grid within the processor's patch bound (sides multiples of the merge kernel, at most
    ``patch_limit_on_one_side``). None when no grid fits."""
    geometry = _moonvit_geometry(media_proc_cfg, merge_kernel_size, vision_patch_size)
    grid = _largest_grid(geometry, _max_image_patches(geometry))
    return None if grid is None else (1, *grid)


# ---------------------------------------------------------------------------
# Input processor
# ---------------------------------------------------------------------------


class K31InputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    """The Kimi K3 / K3.1 image input processor.

    The checkpoint's ``AutoProcessor`` (``KimiK3Processor``, loaded via ``trust_remote_code``) tiles each image and
    returns ``grid_thws`` / ``pixel_values``. The framework injects the ``<|kimi_image_placeholder|>`` marker (see
    :data:`KIMI_K31_PLACEHOLDER_METADATA`), which ``KimiK3Processor.update_raw_text`` expands into a ``<|media_pad|>``
    (id 163605); ``call_with_text_prompt`` repeats it to ``(h // merge_kh) * (w // merge_kw)`` tokens per image.

    Args:
        model_path: Path to the HF model directory.
        config: HuggingFace pretrained config.
        tokenizer: Tokenizer instance.
        trust_remote_code: Whether to allow custom code from the model repo.
    """

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizerBase,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._config = config
        self._tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast,
            )
        )
        self._encode_prompt = KimiPromptEncoder(self._tokenizer)
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast,
        )
        self._model_path = model_path

        text_config = getattr(config, "text_config", config)
        self._dtype = getattr(text_config, "torch_dtype", torch.bfloat16)
        if isinstance(self._dtype, str):
            self._dtype = getattr(torch, self._dtype, torch.bfloat16)

        self._media_placeholder_token_id = getattr(
            config, "media_placeholder_token_id", MEDIA_PLACEHOLDER_TOKEN_ID
        )

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_vocab_size(self) -> int:
        """Return the text vocabulary size."""
        text_config = getattr(self._config, "text_config", self._config)
        return getattr(text_config, "vocab_size", _VOCAB_SIZE)

    def get_mm_token_ids(self) -> Optional[torch.Tensor]:
        """Return the media placeholder token id.

        ``call_with_text_prompt`` expands each ``<|media_pad|>`` into one copy of this in-vocabulary id per output
        embedding, so the executor locates the image positions with ``torch.isin(input_ids, mm_token_ids)``.
        """
        return torch.tensor([self._media_placeholder_token_id], dtype=torch.int32)

    def get_num_tokens_per_image(self, *, image: Union[Image.Image, torch.Tensor], **kwargs) -> int:
        """Calculate the number of tokens for a single image.

        Delegates to the HF image processor's ``media_tokens_calculator``, which accounts for
        Moon-Patch dynamic tiling, the spatial merge kernel and the patch size.

        Args:
            image: PIL image or Torch tensor to compute the token count for.

        Returns:
            Number of visual tokens this image produces after
            MoonViT3d + PatchMergerMLP.
        """
        image = _as_pil_image(image)
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "media_tokens_calculator"):
            media_input = {"type": "image", "image": image}
            return image_processor.media_tokens_calculator(media_input)

        # Fallback: compute from the processor geometry using the same
        # formula as navit_resize_image in the HF reference code.
        geometry = self._moonvit_geometry()
        patch_size = geometry.patch_size
        merge_k = geometry.resize_merge_size

        w, h = image.size
        factor = merge_k * patch_size
        in_patch_limit = geometry.in_patch_limit
        patch_limit_on_one_side = geometry.patch_limit_on_one_side
        s1 = math.sqrt(in_patch_limit / (max(1.0, w // patch_size) * max(1.0, h // patch_size)))
        s2 = patch_limit_on_one_side * patch_size / w
        s3 = patch_limit_on_one_side * patch_size / h
        scale = min(1.0, s1, s2, s3)
        new_w = min(max(1, int(w * scale)), patch_limit_on_one_side * patch_size)
        new_h = min(max(1, int(h * scale)), patch_limit_on_one_side * patch_size)
        pad_w = (factor - new_w % factor) % factor
        pad_h = (factor - new_h % factor) % factor
        token_w = (new_w + pad_w) // factor
        token_h = (new_h + pad_h) // factor
        return token_h * token_w

    def _merge_kernel_size(self) -> Tuple[int, int]:
        """Return the encoder patch-merger kernel ``(kh, kw)`` from the vision config."""
        vision_cfg = getattr(self._config, "vision_config", self._config)
        if isinstance(vision_cfg, dict):
            merge_kernel_size = vision_cfg.get("merge_kernel_size")
        else:
            merge_kernel_size = getattr(vision_cfg, "merge_kernel_size", None)
        if merge_kernel_size is None:
            return 2, 2
        if isinstance(merge_kernel_size, int):
            return merge_kernel_size, merge_kernel_size
        kernel_h, kernel_w = merge_kernel_size
        return int(kernel_h), int(kernel_w)

    def _moonvit_geometry(self) -> _MoonViTGeometry:
        """Read the grid limits from the loaded HF image processor's ``media_proc_cfg``."""
        image_processor = getattr(self._processor, "image_processor", None)
        vision_cfg = getattr(self._config, "vision_config", self._config)
        vision_patch_size = (
            vision_cfg.get("patch_size")
            if isinstance(vision_cfg, dict)
            else getattr(vision_cfg, "patch_size", None)
        )
        return _moonvit_geometry(
            getattr(image_processor, "media_proc_cfg", None) or {},
            self._merge_kernel_size(),
            vision_patch_size,
        )

    def _media_item_lengths(self, image_data: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        """Return the encoder patches and output embeddings of every packed image.

        Raises:
            ValueError: If a grid is outside the processor geometry or the pixel payload does not hold exactly the
                patches of its grids.
        """
        geometry = self._moonvit_geometry()
        kernel_h, kernel_w = geometry.merge_kernel_size
        side_limit = geometry.patch_limit_on_one_side
        max_patches = _max_image_patches(geometry)
        grid_rows = [
            tuple(int(value) for value in row)
            for row in torch.as_tensor(image_data["image_grid_thw"]).tolist()
        ]
        encoder_lengths: List[int] = []
        output_lengths: List[int] = []
        for t, h, w in grid_rows:
            if (
                t != 1
                or not (1 <= h <= side_limit and 1 <= w <= side_limit)
                or h % kernel_h
                or w % kernel_w
                or h * w > max_patches
            ):
                raise ValueError(
                    f"Processed Kimi image grid (t={t}, h={h}, w={w}) is outside the processor geometry: "
                    f"t = 1, sides <= {side_limit} and divisible by {geometry.merge_kernel_size}, and at "
                    f"most {max_patches} patches"
                )
            encoder_lengths.append(h * w)
            output_lengths.append((h // kernel_h) * (w // kernel_w))
        pixel_values = image_data.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor) and pixel_values.shape[0] != sum(encoder_lengths):
            raise ValueError(
                f"pixel_values holds {pixel_values.shape[0]} patches, but its grids have "
                f"{sum(encoder_lengths)}"
            )
        return encoder_lengths, output_lengths

    def get_mm_item_metadata(
        self,
        prompt_token_ids: List[int],
        multimodal_data: Dict[str, Any],
        mm_item_order: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[MultimodalItemMetadata]:
        """Return one encoder item per image, in prompt order.

        An item costs its pre-merge patches and yields one embedding per merged cell of its grid.
        """
        prepared = super().get_mm_item_metadata(prompt_token_ids, multimodal_data, mm_item_order)
        image_data = multimodal_data.get("image")
        if (
            multimodal_data.get("multimodal_embedding") is not None
            or not isinstance(image_data, dict)
            or image_data.get("image_grid_thw") is None
        ):
            return prepared
        encoder_lengths, output_lengths = self._media_item_lengths(image_data)
        if mm_item_order:
            item_refs = [(entry["modality"], entry["index"]) for entry in mm_item_order]
        elif prepared is not None:
            item_refs = list(prepared.item_refs)
        else:
            item_refs = [("image", index) for index in range(len(encoder_lengths))]
        if len(item_refs) != len(encoder_lengths) or any(
            modality != "image" or not 0 <= index < len(encoder_lengths)
            for modality, index in item_refs
        ):
            raise ValueError(
                f"Prompt multimodal items {item_refs} do not match the {len(encoder_lengths)} processed Kimi images"
            )
        return MultimodalItemMetadata(
            item_refs,
            encoder_token_lengths=[encoder_lengths[index] for _, index in item_refs],
            output_embedding_lengths=[output_lengths[index] for _, index in item_refs],
        )

    @torch.inference_mode()
    def call_with_text_prompt(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """Tokenize the prompt and run the HF processor over its images.

        The HF ``AutoProcessor`` applies Moon-Patch dynamic tiling to each image, builds its ``pixel_values`` and
        tokenizes the text with one media placeholder per image, which is then repeated to the image's output
        embeddings. A prompt without images is tokenized by :class:`KimiPromptEncoder`.

        Args:
            inputs: Text prompt with optional ``multi_modal_data`` (images only).
            sampling_params: Sampling configuration.

        Returns:
            ``(token_ids, extra_processed_inputs)``; with images, ``extra_processed_inputs["multimodal_data"]["image"]``
            holds ``pixel_values`` and ``image_grid_thw``.

        Raises:
            ValueError: For a modality other than images.
        """
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data", {}) or {}
        unsupported = sorted(
            modality for modality, items in mm_data.items() if modality != "image" and items
        )
        if unsupported:
            raise ValueError(f"Kimi K3.1 takes image inputs only, got {unsupported}")
        images = mm_data.get("image", [])

        if not images:
            return self._encode_prompt(text_prompt), {}

        # The HF processor takes either chat ``messages`` or ``medias`` and the formatted ``text``; the prompt is
        # already formatted. The serving layer may pass images as tensors (format "pt"), the HF processor takes PIL.
        medias: List[Dict[str, Any]] = [
            {"type": "image", "image": _as_pil_image(img)} for img in images
        ]
        processed = self._processor(medias=medias, text=text_prompt, return_tensors="pt")
        input_ids = processed["input_ids"][0]

        # Repeat each <|media_pad|> to the number of rows the vision tower produces for its image,
        # (h // merge_kh) * (w // merge_kw) for its grid (t, h, w). The repeated id stays the in-vocabulary
        # placeholder, so the executor finds the image positions with the model's ``multimodal_token_ids``.
        placeholder_id = self._media_placeholder_token_id
        grid_thws = processed.get("grid_thws", processed.get("image_grid_thw"))
        if grid_thws is not None:
            merge_kh, merge_kw = self._merge_kernel_size()
            tokens_per_media = [(h // merge_kh) * (w // merge_kw) for _, h, w in grid_thws.tolist()]
            new_ids: List[int] = []
            media_idx = 0
            for tok in input_ids.tolist():
                if tok == placeholder_id:
                    if media_idx >= len(tokens_per_media):
                        raise ValueError(
                            f"More media placeholder tokens ({media_idx + 1}) in input_ids than images in "
                            f"grid_thws ({len(tokens_per_media)}). Check that the prompt has the correct number "
                            f"of <|media_pad|> tokens."
                        )
                    new_ids.extend([placeholder_id] * tokens_per_media[media_idx])
                    media_idx += 1
                else:
                    new_ids.append(tok)
            input_ids = torch.tensor(new_ids, dtype=torch.int32)

        multimodal_data: Dict[str, Any] = {}
        pixel_values = processed.get("pixel_values")
        if pixel_values is not None:
            image_entry: Dict[str, Any] = {"pixel_values": pixel_values}
            if grid_thws is not None:
                image_entry["image_grid_thw"] = grid_thws
            multimodal_data["image"] = image_entry

        return input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


# Prompt placeholders of the Kimi K3 / K3.1 vision input (registered for ``kimi_k31`` by
# ``routing.input_processor_class``).
KIMI_K31_PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={
        "image": "<|kimi_image_placeholder|>",
    },
    placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    # K3's reference renderer concatenates content parts with no separator.
    placeholders_separator="",
)


# Prompt tokenization


def is_kimi_tiktoken_tokenizer(tokenizer: Any) -> bool:
    """Whether ``tokenizer`` is the ``TikTokenTokenizer`` of a Kimi checkpoint's ``tokenization_kimi.py``."""
    tokenizer_type = type(tokenizer)
    return tokenizer_type.__name__ == "TikTokenTokenizer" and tokenizer_type.__module__.endswith(
        ".tokenization_kimi"
    )


def build_added_token_pattern(tokenizer: Any, added: Dict[str, int]) -> Optional[re.Pattern]:
    """The regex that splits text on added tokens the way Hugging Face does.

    Args:
        tokenizer: The Hugging Face tokenizer.
        added: Its added vocabulary, ``tokenizer.get_added_vocab()``.

    Returns:
        A pattern whose ``split`` keeps the added tokens as separate pieces, or None when a regex split could differ
        from Hugging Face's: when there are no added tokens, when one added token occurs inside another (Hugging
        Face's added-token trie then drops or merges matches), or when a special token is not an added token (Hugging
        Face keeps such a piece whole).
    """
    if not added or any(a != b and a in b for a in added for b in added):
        return None
    if not set(tokenizer.all_special_tokens).issubset(added):
        return None
    # Longest first, so an alternation tried left to right matches like the leftmost-longest trie.
    return re.compile(
        "(" + "|".join(re.escape(s) for s in sorted(added, key=len, reverse=True)) + ")"
    )


class KimiPromptEncoder:
    """The token ids of a formatted prompt: those of ``tokenizer(text, return_tensors="pt").input_ids[0]``.

    For a Kimi tiktoken tokenizer the text between added tokens is encoded with the tokenizer's native ``encode`` and
    the ids then go through ``prepare_for_model(add_special_tokens=True)``, the step Hugging Face's call ends with, so
    special tokens are added as that call adds them. Any other tokenizer takes Hugging Face's call.

    Args:
        tokenizer: The input processor's tokenizer, a :class:`TransformersTokenizer` or a Hugging Face tokenizer.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._hf_tokenizer = (
            tokenizer.tokenizer if isinstance(tokenizer, TransformersTokenizer) else tokenizer
        )
        # The added vocabulary and its split pattern from the last _added_token_pattern call (None until then).
        self._pattern_cache: Optional[Tuple[Dict[str, int], Optional[re.Pattern]]] = None

    def __call__(self, text: str) -> List[int]:
        """Return the token ids of ``text``."""
        tok = self._hf_tokenizer
        split = self._fast_split(text)
        if split is None:
            return self._tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        pattern, added = split
        # Keep Hugging Face's added-token boundaries: a native encode of the entire string can give a marker string
        # other ids.
        ids: List[int] = []
        for piece in pattern.split(text):
            if piece in added:
                ids.append(added[piece])
            elif piece:
                ids.extend(tok.encode(piece))
        return tok.prepare_for_model(ids, add_special_tokens=True)["input_ids"]

    def _fast_split(self, text: Any) -> Optional[Tuple[re.Pattern, Dict[str, int]]]:
        """The added-token pattern and added vocabulary to split ``text`` with, or None for Hugging Face's path.

        The guards leave lower-casing, split special tokens and added tokens that strip or match single words to
        Hugging Face.
        """
        tok = self._hf_tokenizer
        if not isinstance(text, str) or not is_kimi_tiktoken_tokenizer(tok):
            return None
        if getattr(tok, "do_lower_case", False) or getattr(tok, "split_special_tokens", False):
            return None
        if any(v.lstrip or v.rstrip or v.single_word for v in tok.added_tokens_decoder.values()):
            return None
        added = tok.get_added_vocab()
        pattern = self._added_token_pattern(added)
        return None if pattern is None else (pattern, added)

    def _added_token_pattern(self, added: Dict[str, int]) -> Optional[re.Pattern]:
        cached = self._pattern_cache
        if cached is None or cached[0] != added:
            # Copy: Hugging Face returns its live added-token dict, which add_tokens mutates in place.
            cached = (dict(added), build_added_token_pattern(self._hf_tokenizer, added))
            self._pattern_cache = cached
        return cached[1]
