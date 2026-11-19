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

:class:`K31ServingExtension` is K3.1's serving extension: Kimi K3's chat-request semantics plus a grammar for
``tool_choice="required"``. K3.1 writes K3's XTML tool calls and is served with the ``kimi_k3`` tool parser and the
``kimi_k31`` reasoning parser. For ``tool_choice="required"`` the extension builds a mandatory tool-call grammar (an
xgrammar structural tag) when the request can carry one; otherwise the chat template asks for the call.
``tensorrt_llm.serve.extensions``
imports this module, so every process that consults the serving-extension registry (the server, the disaggregated
router, ``trtllm-render``) resolves ``kimi_k31`` to this extension.
"""

from __future__ import annotations

import functools
import json
import math
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import xgrammar
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase
from xgrammar import structural_tag as st

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
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.extensions.kimi_k3 import KimiK3ServingExtension
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.serving_extensions import RequiredToolChoiceMode, register_serving_extension
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.kimi_k3_tool_parser import KimiK3ToolParser

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


# Serving extension


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


_OPEN = "<|open|>"
_CLOSE = "<|close|>"
_SEP = "<|sep|>"
_END_OF_MSG = "<|end_of_msg|>"
_ARGUMENT_CLOSE = f"{_CLOSE}argument{_SEP}"

# JSON Schema type -> XTML ``type`` attribute (encoding_k3._xtml_type folds
# integer into number).
_XTML_TYPES = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "object": "object",
    "array": "array",
}

# One raw string-value character: anything but the "<|" that starts a marker.
_RAW_STRING_CHAR = r"(?:[^<]|<[^|])"


# The K3 message-level tags whose close a channel's text must not contain.
_MESSAGE_TAGS = ("think", "response", "tools", "message")


def _free_text(channel: str) -> st.AnyTextFormat:
    """Text of one channel: no tag opens, no message end, no close of another message-level tag.

    The enclosing tag's end string (``<|close|>CHANNEL<|sep|>``) closes it; xgrammar cannot
    also exclude a prefix of that end string, so the other closes are excluded by name. Without
    them a response that closes as ``<|close|>think<|sep|>`` (or ``<|close|>message<|sep|>``)
    stays inside the free text, where the reopening ``<|open|>`` and the stop ``<|end_of_msg|>``
    are masked, and generation repeats ``<|close|>message<|sep|>`` until the token limit.
    """
    others = [f"{_CLOSE}{tag}" for tag in _MESSAGE_TAGS if tag != channel]
    return st.AnyTextFormat(excludes=[_OPEN, _END_OF_MSG, *others])


def _any_argument_tag() -> st.TagFormat:
    """An ``argument`` tag with any header and a raw value."""
    return st.TagFormat(
        begin=f"{_OPEN}argument ",
        content=st.SequenceFormat(
            elements=[
                st.RegexFormat(pattern=r"[^<]*<\|sep\|>"),
                st.AnyTextFormat(),
            ]
        ),
        end=_ARGUMENT_CLOSE,
    )


_DEFS_KEYS = ("$defs", "definitions")
# The ``$defs`` entry that carries the whole parameters document into a
# property's standalone schema, for pointers such as ``#`` that address it.
_PARAMETERS_DEF = "__kimi_k3_parameters"
# Keywords that only describe a schema; a property with nothing else accepts any value.
_ANNOTATION_KEYWORDS = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
    }
)


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "object" if isinstance(value, dict) else "array"


def _is_local_pointer(ref: str) -> bool:
    return ref == "#" or ref.startswith("#/")


def _resolve_pointer(root: Dict[str, Any], ref: str) -> Any:
    """The node a local JSON pointer (``#`` or ``#/a/b``) addresses in ``root``, or None."""
    node: Any = root
    for part in ref[2:].split("/") if ref != "#" else []:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _refs(node: Any):
    if isinstance(node, dict):
        if isinstance(node.get("$ref"), str):
            yield node["$ref"]
        for value in node.values():
            yield from _refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _refs(value)


def _value_types(schema: Any, root: Dict[str, Any], seen: tuple = ()) -> Optional[set]:
    """The JSON types a value valid against ``schema`` can have; None when any type can."""
    if not isinstance(schema, dict):
        return None
    json_type = schema.get("type")
    if isinstance(json_type, str):
        return {json_type}
    if isinstance(json_type, list) and json_type and all(isinstance(t, str) for t in json_type):
        return set(json_type)
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return {_json_type(v) for v in schema["enum"]}
    if "const" in schema:
        return {_json_type(schema["const"])}
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen or not _is_local_pointer(ref):
            return None
        return _value_types(_resolve_pointer(root, ref), root, (*seen, ref))
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and branches:
            types: set = set()
            for branch in branches:
                branch_types = _value_types(branch, root, seen)
                if branch_types is None:
                    return None
                types |= branch_types
            return types
    narrowed: Optional[set] = None
    for branch in schema.get("allOf") or []:
        branch_types = _value_types(branch, root, seen)
        if branch_types is not None:
            narrowed = branch_types if narrowed is None else narrowed & branch_types
    return narrowed or None


def _string_schema(schema: Any, root: Dict[str, Any]) -> Dict[str, Any]:
    """The string schema whose enum or length bounds a raw string value must meet, or {}."""
    seen = set()
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        if ref in seen or not _is_local_pointer(ref):
            return {}
        seen.add(ref)
        schema = _resolve_pointer(root, ref)
    if not isinstance(schema, dict):
        return {}
    if schema.get("type") == "string":
        return schema
    branches = [
        b
        for key in ("anyOf", "oneOf")
        for b in schema.get(key) or []
        if _value_types(b, root) == {"string"}
    ]
    return _string_schema(branches[0], root) if len(branches) == 1 else {}


def _standalone_schema(schema: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
    """``schema``, cut out of the parameters document ``root``, as a standalone document.

    The document's definition tables move along to the new root, so
    ``#/$defs/...`` and ``#/definitions/...`` pointers keep resolving. Any other
    local pointer (``#`` itself, ``#/properties/...``) addresses the parameters
    document, which is carried as one more ``$defs`` entry.
    """
    uses_root = False

    def rewrite(node: Any) -> Any:
        nonlocal uses_root
        if isinstance(node, list):
            return [rewrite(value) for value in node]
        if not isinstance(node, dict):
            return node
        out = {key: rewrite(value) for key, value in node.items()}
        ref = node.get("$ref")
        if isinstance(ref, str) and _is_local_pointer(ref):
            if not ref.startswith(tuple(f"#/{key}/" for key in _DEFS_KEYS)):
                uses_root = True
                out["$ref"] = f"#/$defs/{_PARAMETERS_DEF}{ref[1:]}"
        return out

    document = rewrite(schema)
    tables = {key: rewrite(root[key]) for key in _DEFS_KEYS if isinstance(root.get(key), dict)}
    needs_parameters = uses_root
    parameters = rewrite({key: value for key, value in root.items() if key not in _DEFS_KEYS})
    for key, table in tables.items():
        own = document.get(key)
        document[key] = {**own, **table} if isinstance(own, dict) else table
    if needs_parameters:
        document["$defs"] = {**document.get("$defs", {}), _PARAMETERS_DEF: parameters}
    return document


# JSON-schema keywords whose values are subschemas: a map of them, one, or a list.
_SUBSCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SUBSCHEMA_ONE = (
    "items",
    "additionalProperties",
    "additionalItems",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
)
_SUBSCHEMA_LISTS = ("items", "prefixItems", "anyOf", "oneOf", "allOf")


def _without_negative_lower_bounds(schema: Any) -> Any:
    """``schema`` without a negative ``minimum`` / ``exclusiveMinimum`` in it or its subschemas.

    xgrammar's range regex for a negative lower bound admits a bare "-" and rejects in-range negative numbers;
    without the bound the grammar still enforces the type. Only schema positions are rewritten: the values of other
    keywords (``enum``, ``const``, ``default``, ``examples``, ...) are kept as they are.
    """
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if (
            key in ("minimum", "exclusiveMinimum")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value < 0
        ):
            continue
        if key in _SUBSCHEMA_MAPS and isinstance(value, dict):
            value = {name: _without_negative_lower_bounds(sub) for name, sub in value.items()}
        elif key in _SUBSCHEMA_LISTS and isinstance(value, list):
            value = [_without_negative_lower_bounds(sub) for sub in value]
        elif key in _SUBSCHEMA_ONE and isinstance(value, dict):
            value = _without_negative_lower_bounds(value)
        out[key] = value
    return out


@functools.lru_cache(maxsize=1024)
def _json_schema_compiles(document: str) -> bool:
    try:
        xgrammar.Grammar.from_json_schema(document)
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning(f"kimi_k3 required-tool grammar: keeping a parameter unconstrained, {e}")
        return False
    return True


def _raw_string_content(schema: Dict[str, Any]) -> Any:
    """Raw string values: only an enum or a length bound can constrain them."""
    values = schema.get("enum")
    if values is None and isinstance(schema.get("const"), str):
        values = [schema["const"]]
    max_len = schema.get("maxLength")
    if (
        isinstance(values, list)
        and values
        and all(isinstance(v, str) and "<|" not in v for v in values)
    ):
        options = [st.ConstStringFormat(value=v) for v in values]
        return options[0] if len(options) == 1 else st.OrFormat(elements=options)
    if isinstance(max_len, int) and 0 <= max_len <= 4096:
        min_len = schema.get("minLength", 0)
        if not isinstance(min_len, int) or not 0 <= min_len <= max_len:
            min_len = 0
        return st.RegexFormat(pattern=f"{_RAW_STRING_CHAR}{{{min_len},{max_len}}}")
    return st.AnyTextFormat()


def _narrowed_schema(
    schema: Dict[str, Any], root: Dict[str, Any], xtml_type: str
) -> Dict[str, Any]:
    """``schema`` cut down to the values the K3 renderer writes with ``xtml_type``.

    The tag's ``type`` attribute then matches its value. xgrammar 0.1.32 ignores
    ``allOf`` and a sibling ``type`` next to ``anyOf``, so the union's branches
    of other types are dropped instead.
    """
    json_types = {t for t, x in _XTML_TYPES.items() if x == xtml_type}
    seen = set()
    while (
        set(schema) - _ANNOTATION_KEYWORDS == {"$ref"}
        and isinstance(schema["$ref"], str)
        and schema["$ref"] not in seen
    ):
        seen.add(schema["$ref"])
        target = _resolve_pointer(root, schema["$ref"])
        if not isinstance(target, dict):
            break
        schema = target
    out = dict(schema)
    if isinstance(out.get("type"), list):
        out["type"] = [t for t in out["type"] if t in json_types]
    for key in ("anyOf", "oneOf"):
        if isinstance(out.get(key), list):
            out[key] = [
                branch
                for branch in out[key]
                if (types := _value_types(branch, root)) is None or types & json_types
            ]
    return out


def _argument_tags(key: str, schema: Any, root: Dict[str, Any]) -> List[st.TagFormat]:
    """The ``argument`` tags one parameter can be written with, one per XTML type.

    The type follows the value as the K3 renderer writes it (``type="null"`` for
    null, ``type="object"`` for an object, ...), so a parameter whose schema
    admits several types (``$ref``, ``anyOf``, a type list) gets one tag per
    type. Strings are written raw; every other type is written as JSON that must
    match the parameter's schema restricted to that type. Only a parameter that
    constrains nothing, or whose schema xgrammar cannot compile, keeps a
    permissive tag.
    """
    if (
        not isinstance(schema, dict)
        or set(schema) <= _ANNOTATION_KEYWORDS
        or not all(_is_local_pointer(ref) for ref in _refs(schema))
    ):
        return [_any_argument_tag()]
    json_types = _value_types(schema, root)
    if json_types is not None and not json_types <= _XTML_TYPES.keys():
        return [_any_argument_tag()]
    tags = []
    for xtml_type in sorted({_XTML_TYPES[t] for t in json_types or _XTML_TYPES}):
        if xtml_type == "string":
            content: Any = _raw_string_content(_string_schema(schema, root))
        else:
            narrowed = _narrowed_schema(schema, root, xtml_type) if json_types else schema
            document = json.dumps(
                _without_negative_lower_bounds(_standalone_schema(narrowed, root))
            )
            if not _json_schema_compiles(document):
                return [_any_argument_tag()]
            content = st.JSONSchemaFormat(json_schema=json.loads(document))
        tags.append(
            st.TagFormat(
                begin=f'{_OPEN}argument key="{_escape_attr(key)}" type="{xtml_type}"{_SEP}',
                content=content,
                end=_ARGUMENT_CLOSE,
            )
        )
    return tags


def _call_tag(tool: Tool) -> st.TagFormat:
    """``<|open|>call tool="NAME" index="N"<|sep|>`` and the call's arguments."""
    parameters = tool.function.parameters
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    if isinstance(properties, dict) and properties:
        argument_tags = [
            tag for key, prop in properties.items() for tag in _argument_tags(key, prop, parameters)
        ]
        required = parameters.get("required")
        at_least_one = isinstance(required, list) and bool(required)
    else:
        argument_tags = [_any_argument_tag()]
        at_least_one = False
    return st.TagFormat(
        begin=f'{_OPEN}call tool="{_escape_attr(tool.function.name)}" index="',
        content=st.SequenceFormat(
            elements=[
                st.RegexFormat(pattern="[0-9]+"),
                st.ConstStringFormat(value=f'"{_SEP}'),
                st.TagsWithSeparatorFormat(
                    tags=argument_tags, separator="", at_least_one=at_least_one
                ),
            ]
        ),
        end=f"{_CLOSE}call{_SEP}",
    )


@register_serving_extension(model_types=("kimi_k31",), reasoning_parsers=("kimi_k31",))
class K31ServingExtension(KimiK3ServingExtension):
    """Kimi K3's serving semantics for kimi_k31 chat requests, with a grammar for required tool calls."""

    def required_tool_choice_mode(self) -> RequiredToolChoiceMode:
        """Enforce the call with the kimi_k3 tool grammar when the request allows it."""
        return "prefer_grammar"

    def build_required_structural_tag(
        self, parser: BaseToolParser, tools: List[Tool], *, thinking: bool
    ) -> Optional[st.StructuralTag]:
        """Mandatory-call grammar for ``tool_choice="required"``.

        The message is ``[think body <|close|>think<|sep|> <|open|>response<|sep|>]``
        (the prompt already opened the first channel), the response body,
        ``<|close|>response<|sep|>``, a tools section of one or more calls of
        the declared tools, and ``<|close|>message<|sep|>``; the request's stop
        token follows. Call bodies use the per-argument XTML form the chat
        template renders. Returns None for tool names the K3 wire format cannot
        carry. A server configured with another tool parser gets that parser's
        grammar.
        """
        if not isinstance(parser, KimiK3ToolParser):
            return super().build_required_structural_tag(parser, tools, thinking=thinking)
        if not tools or any("<" in tool.function.name for tool in tools):
            return None

        elements: List[Any] = []
        if thinking:
            elements += [
                st.TagFormat(begin="", content=_free_text("think"), end=f"{_CLOSE}think{_SEP}"),
                st.ConstStringFormat(value=f"{_OPEN}response{_SEP}"),
            ]
        elements += [
            st.TagFormat(begin="", content=_free_text("response"), end=f"{_CLOSE}response{_SEP}"),
            st.TagFormat(
                begin=parser.bot_token,
                content=st.TagsWithSeparatorFormat(
                    tags=[_call_tag(tool) for tool in tools], separator="", at_least_one=True
                ),
                end=parser.eot_token,
            ),
            st.ConstStringFormat(value=f"{_CLOSE}message{_SEP}"),
        ]
        return st.StructuralTag(format=st.SequenceFormat(elements=elements))
