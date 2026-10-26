# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image input for Step5ForConditionalGeneration: pixel tiles and the prompt span of each image.

Step-5-Preview ships no processor (no transformers class, no ``processor_config.json``),
and StepFun's reference consumes preprocessed tensors only: ``inference/README.md``
"Vision inputs" leaves resizing, normalization, crop selection and prompt construction
to the processor of the matching checkpoint family, whose geometry this checkpoint's
``config.json`` repeats (``image_token_len`` 169, ``patch_token_len`` 81,
``image_token_id`` 128001, the ``<im_start>``/``<im_end>`` framing). This module is
that policy, written against the checkpoint's own config and tokenizer.

**An assumption, not a published fact.** The policy below is ported from Step-3.7's
processor (``processing_step3.Step3VLProcessor``: the same perception encoder, the same
``vision_config``, the same structural tokens): its tiling, its normalization and its
prompt span. What StepFun's reference README fixes is only the geometry -- a 728 px
overview encoding to 169 rows, 504 px crops encoding to 81 rows each, and an image's
crops before its overview. If StepFun publishes a Step-5 processor, this module is
what has to be checked against it.

**One image** becomes an overview plus zero or more crops:

* The overview is the whole image squeezed to ``vision_config.image_size`` square
  (728). It encodes to 169 rows: 52 x 52 patches of 14, then the reference encoder's
  two stride-2 convolutions (``vit_downsampler1``/``2``), 26 x 26 and 13 x 13.
* Crops are square windows of 504 (81 rows: 36 -> 18 -> 9) cut from the image after
  it is stretched to whole windows (:func:`tile_plan`), row-major. A small image
  (longer side at most 728) is cropped only when elongated, with a window of its
  shorter side.

Pixels are ``ToTensor`` scaled to [0, 1], normalized with the CLIP mean and std, then
resized (bilinear, antialiased) to the tile side, in that order.

**The prompt** carries one ``<im_patch>`` per image, as ``chat_template.jinja``
renders an image content item. Each is replaced, in token ids, by the image's span:

    per crop:  <patch_start> <im_patch> x 81 <patch_end> [<patch_newline> after a row]
    then:      <im_start> <im_patch> x 169 <im_end>

which is the order StepFun's ``get_multimodal_embeddings`` emits feature rows in:
an image's crops first, then its overview. ``<im_patch>`` keeps its vocabulary id, so
the model finds the rows the vision features replace by that id (the reference's
``input_ids == image_token_id``); the framing tokens are ordinary text embeddings.

**multimodal_data["image"]** (what the target's vision tower reads):

* ``pixel_values``: ``[images, 3, 728, 728]`` in the model dtype, one overview per
  image in prompt order.
* ``patch_pixel_values``: ``[crops, 3, 504, 504]`` in the model dtype, every
  image's crops in prompt order; absent when no image has a crop.
* ``num_patches``: ``list[int]``, the crop count of each image.

**Text** is tokenized by the engine's tokenizer, the one ``tokenizer_factory`` loads
(``TransformersTokenizer.from_pretrained``, which reloads this checkpoint's ByteLevel
``tokenizer.json`` verbatim). Built without one, the processor loads it the same way,
never through a bare ``AutoTokenizer``, which drops this checkpoint's spaces.

**Memory profiling.** :meth:`Step5PreviewInputProcessor.get_dummy_mm_data` builds the
largest tile set that fits the engine's encoder token budget, so the engine runs the
vision tower over it before it sizes the KV cache.

The engine finds this processor through
``tensorrt_llm.inputs.registry.MODELING_V2_INPUT_PROCESSOR_MODULES``: no built-in model
class serves this architecture, so ``get_model_architecture`` has none to key on.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PretrainedConfig

from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.data import TextPrompt
from tensorrt_llm.inputs.registry import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    register_architecture_input_processor,
)
from tensorrt_llm.inputs.tokenization import tokenize_prompt
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.tokenizer import tokenizer_factory

ARCHITECTURE = "Step5ForConditionalGeneration"
MODEL_TYPE = "step5_vl"

#: Per-channel normalization of the perception encoder (the CLIP constants).
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)

#: Side of a crop tile.
CROP_SIDE = 504
#: The longer side is scaled down to this before anything else.
_MAX_SIDE = 3024
#: An image thinner than this whose aspect exceeds `_PAD_ASPECT` is padded square first.
_PAD_BELOW = 32
_PAD_ASPECT = 4
#: A small image (longer side at most the overview side) is cropped only when its
#: aspect exceeds this.
_SMALL_CROP_ASPECT = 1.5
#: A large image whose aspect exceeds this is cropped with a window of its shorter
#: side (when that is under `CROP_SIDE`).
_LARGE_THIN_ASPECT = 4
#: A side spanning a fraction of a further window above this grows by that window.
_ROUND_UP_FRACTION = 0.2

#: Structural tokens of an image span, by name in the tokenizer.
_PATCH_START = "<patch_start>"
_PATCH_END = "<patch_end>"
_PATCH_NEWLINE = "<patch_newline>"

ImageInput = Union[Image.Image, torch.Tensor, np.ndarray]


class TilePlan(NamedTuple):
    """Where one image's tiles come from, as a function of its size only.

    ``pad_side``: the side of the black square the image is pasted into (top left)
    first, or None. ``size``: the image size after padding and the `_MAX_SIDE` cap;
    the overview is taken from the image at this size. ``canvas``: the size the
    image is stretched to before cropping. ``boxes``: the crops, ``(left, top,
    right, bottom)`` on the canvas, row-major; a box may extend past the canvas, and
    that part is black. ``row_ends``: per crop, whether a ``<patch_newline>`` follows
    it (a crop that closes a row other than the last).
    """

    pad_side: Optional[int]
    size: Tuple[int, int]
    canvas: Tuple[int, int]
    boxes: Tuple[Tuple[int, int, int, int], ...]
    row_ends: Tuple[bool, ...]


def _window(long_side: int, short_side: int, overview_side: int) -> int:
    """The crop window of an image, 0 for none."""
    aspect = long_side / short_side
    if long_side <= overview_side:
        return short_side if aspect > _SMALL_CROP_ASPECT else 0
    if aspect > _LARGE_THIN_ASPECT:
        return min(short_side, CROP_SIDE)
    return CROP_SIDE


def _stretched(side: int, window: int) -> int:
    """`side` stretched to whole windows; a side under one window is kept."""
    ratio = side / window
    if ratio < 1:
        return side
    windows = int(ratio) + 1 if ratio - side // window > _ROUND_UP_FRACTION else int(ratio)
    return window * windows


def _starts(extent: int, window: int) -> List[int]:
    """Window origins along one axis: step one window, the last one flush with the end."""
    count = 1 if extent <= window else math.ceil((extent - window) / window + 1)
    starts = [window * i for i in range(count)]
    if count > 1 and starts[-1] + window > extent:
        starts[-1] = extent - window
    return starts


def tile_plan(width: int, height: int, overview_side: int) -> TilePlan:
    """The tiles of a ``width`` x ``height`` image."""
    pad_side = None
    aspect = width / height
    if min(width, height) < _PAD_BELOW and (aspect > _PAD_ASPECT or aspect < 1 / _PAD_ASPECT):
        pad_side = max(width, height)
        width = height = pad_side
    if max(width, height) > _MAX_SIDE:
        scale = _MAX_SIDE / max(width, height)
        width, height = int(width * scale), int(height * scale)
    window = _window(max(width, height), min(width, height), overview_side)
    if window == 0:
        return TilePlan(pad_side, (width, height), (width, height), (), ())
    canvas = (_stretched(width, window), _stretched(height, window))
    xs, ys = _starts(canvas[0], window), _starts(canvas[1], window)
    boxes = tuple((x, y, x + window, y + window) for y in ys for x in xs)
    last = len(boxes) - 1
    row_ends = tuple((i + 1) % len(xs) == 0 and i != last for i in range(len(boxes)))
    return TilePlan(pad_side, (width, height), canvas, boxes, row_ends)


def _rows_after_downsampling(side: int, patch_size: int) -> int:
    """Feature rows of a square tile: patches of `patch_size`, then the encoder's two
    3x3 stride-2 padding-1 convolutions, each taking a grid of g to ceil(g / 2)."""
    grid = side // patch_size
    for _ in range(2):
        grid = (grid + 1) // 2
    return grid * grid


def _sub(config: Any, name: str) -> Any:
    """A field of a (possibly nested) config that may be an object or a dict."""
    return config.get(name) if isinstance(config, dict) else getattr(config, name, None)


def _as_rgb_image(image: ImageInput) -> Image.Image:
    """Any loader format as an RGB PIL image: PIL as is, a ``[3, H, W]`` float tensor
    in [0, 1] (``load_image(format="pt")``), or an ``[H, W, 3]`` uint8 array."""
    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    if isinstance(image, torch.Tensor):
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"an image tensor must be [3, H, W]; got {tuple(image.shape)}")
        if image.dtype == torch.uint8:
            hwc = image.permute(1, 2, 0)
        else:
            # ToTensor's u / 255 times 255 rounds back to u exactly.
            hwc = (image.float() * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        return Image.fromarray(hwc.cpu().contiguous().numpy(), mode="RGB")
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"an image array must be [H, W, 3] uint8; got {image.shape} {image.dtype}"
            )
        return Image.fromarray(image, mode="RGB")
    raise TypeError(f"unsupported image type {type(image).__name__}")


def _image_size(image: ImageInput) -> Tuple[int, int]:
    """``(width, height)`` of any loader format, without decoding pixels."""
    if isinstance(image, Image.Image):
        return image.width, image.height
    if isinstance(image, torch.Tensor):
        return int(image.shape[-1]), int(image.shape[-2])
    if isinstance(image, np.ndarray):
        return int(image.shape[1]), int(image.shape[0])
    raise TypeError(f"unsupported image type {type(image).__name__}")


def _patches(side: int, patch_size: int) -> int:
    """Encoder rows (patches) of a square tile: what the tower attends over."""
    return (side // patch_size) ** 2


class Step5PreviewInputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    """Tokenizes a prompt and turns its images into Step-5-Preview's tiles and spans."""

    supports_tokenization_cache = True

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: Any,
        trust_remote_code: bool = True,
        enable_tokenization_cache: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if tokenizer is None:
            tokenizer = tokenizer_factory(model_path, trust_remote_code=trust_remote_code)
        self._tokenizer = tokenizer
        self._hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self._init_tokenization_cache(enable_tokenization_cache, tokenizer)

        text = _sub(config, "text_config") or config
        vision = _sub(config, "vision_config")
        dtype = _sub(text, "torch_dtype") or _sub(text, "dtype") or torch.bfloat16
        self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self._vocab_size = int(_sub(text, "vocab_size"))

        self._overview_side = int(_sub(vision, "image_size"))
        patch_size = int(_sub(vision, "patch_size"))
        self._overview_rows = _rows_after_downsampling(self._overview_side, patch_size)
        self._crop_rows = _rows_after_downsampling(CROP_SIDE, patch_size)
        self._overview_patches = _patches(self._overview_side, patch_size)
        self._crop_patches = _patches(CROP_SIDE, patch_size)
        for field, rows in (
            ("image_token_len", self._overview_rows),
            ("patch_token_len", self._crop_rows),
        ):
            stated = _sub(config, field)
            if stated is not None and int(stated) != rows:
                raise ValueError(
                    f"config.json states {field} = {stated}, but its tiles encode to {rows} "
                    f"rows ({self._overview_side} px overview, {CROP_SIDE} px crops, patch "
                    f"{patch_size})"
                )

        self._image_token_id = int(_sub(config, "image_token_id"))
        names = {
            "im_start": _sub(config, "im_start_token") or "<im_start>",
            "im_end": _sub(config, "im_end_token") or "<im_end>",
            "patch_start": _PATCH_START,
            "patch_end": _PATCH_END,
            "patch_newline": _PATCH_NEWLINE,
        }
        self._ids = {key: self._token_id(name) for key, name in names.items()}
        image_token = _sub(config, "im_patch_token") or "<im_patch>"
        if self._token_id(image_token) != self._image_token_id:
            raise ValueError(
                f"the tokenizer maps {image_token!r} to {self._token_id(image_token)}, "
                f"config.json's image_token_id is {self._image_token_id}"
            )
        self._mean = torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(_STD, dtype=torch.float32).view(3, 1, 1)

    def _token_id(self, token: str) -> int:
        token_id = self._hf_tokenizer.convert_tokens_to_ids(token)
        unk = getattr(self._hf_tokenizer, "unk_token_id", None)
        if token_id is None or (unk is not None and token_id == unk):
            raise ValueError(f"the tokenizer has no token {token!r}")
        return int(token_id)

    # ------- BaseMultimodalInputProcessor -----------------------------------

    @property
    def processor(self) -> None:
        """No HuggingFace processor exists for this checkpoint; this class is the policy."""
        return None

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_mm_token_ids(self) -> torch.Tensor:
        """Every id inside an image span: ``<im_patch>`` and the framing tokens, so one
        image is one contiguous span for multimodal hashing."""
        return torch.tensor([self._image_token_id, *self._ids.values()], dtype=torch.int32)

    def get_mm_special_token_ids(self) -> torch.Tensor:
        """The framing tokens: inside an image span, but text embeddings."""
        return torch.tensor(list(self._ids.values()), dtype=torch.int32)

    def get_num_tokens_per_image(self, *, image: ImageInput, **kwargs) -> int:
        """The prompt span of one image, framing included."""
        width, height = _image_size(image)
        return len(self._image_span(tile_plan(width, height, self._overview_side)))

    def get_preferred_media_io_kwargs(self) -> Dict[str, Dict[str, Any]]:
        """Per-modality media-IO decode defaults, which ``OpenAIServer`` asks every
        multimodal input processor for at start-up.

        None: images are this checkpoint's only modality, and :meth:`tiles` takes every
        format the loader decodes to (PIL, ``[3, H, W]`` tensor, ``[H, W, 3]`` uint8
        array), so the server's ``--media_io_kwargs`` and per-request overrides apply
        unchanged.
        """
        return {}

    # ------- BaseMultimodalDummyInputsBuilder: encoder memory profiling -------

    def _image_patches(self, crops: int) -> int:
        """Encoder rows of one image with `crops` crops: its overview's and its crops'."""
        return self._overview_patches + crops * self._crop_patches

    def _crops_within(self, budget: int) -> Optional[int]:
        """The most crops one image may carry and still fit `budget` encoder rows, or
        None when not even its overview fits."""
        if budget < self._overview_patches:
            return None
        return (budget - self._overview_patches) // self._crop_patches

    def get_mm_max_tokens_per_item(
        self, max_num_encoder_tokens: Optional[int] = None
    ) -> Dict[str, int]:
        """Under ``max_num_encoder_tokens``, the encoder rows (patches, before the
        downsamplers) of the largest image that fits it, overview plus crops (``{}`` when
        not even an overview fits).

        Without a budget, ``{}``: the tiling bounds no image's size. A thin image is cut
        into windows of its shorter side, so its crop count grows as that side shrinks
        (:func:`tile_plan` cuts a 4000 x 32 image into 126 crops, a 100000 x 40 one
        into 3024); only the prompt length the engine admits bounds it, and that is not
        the processor's to know.
        """
        if max_num_encoder_tokens is None:
            return {}
        crops = self._crops_within(max_num_encoder_tokens)
        return {} if crops is None else {"image": self._image_patches(crops)}

    def get_dummy_mm_data(
        self,
        *,
        max_num_encoder_tokens: int,
        mm_counts: Mapping[str, int],
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """``mm_counts["image"]`` images, each the largest tile set their share of
        ``max_num_encoder_tokens`` holds (an overview and as many crops as fit), as the
        ``multimodal_data`` the vision tower reads. The pixels are zeros: the tower's
        activations do not depend on their values."""
        unsupported = sorted(set(mm_counts) - {"image"})
        if unsupported:
            raise ValueError(f"Step-5-Preview takes image input only; asked for {unsupported}")
        images = int(mm_counts.get("image", 0))
        if images < 0:
            raise ValueError(f"the image count must be nonnegative; got {images}")
        if images == 0:
            return {}
        crops = self._crops_within(max_num_encoder_tokens // images)
        if crops is None:
            raise ValueError(
                f"{images} images do not fit {max_num_encoder_tokens} encoder rows: an "
                f"overview alone is {self._overview_patches}"
            )
        dtype = dtype or self._dtype
        side = self._overview_side
        data: Dict[str, Any] = {
            "pixel_values": torch.zeros(images, 3, side, side, dtype=dtype),
            "num_patches": [crops] * images,
        }
        if crops:
            data["patch_pixel_values"] = torch.zeros(
                images * crops, 3, CROP_SIDE, CROP_SIDE, dtype=dtype
            )
        return {"image": data}

    # ------- the policy -----------------------------------------------------

    def _image_span(self, plan: TilePlan) -> List[int]:
        ids = self._ids
        span: List[int] = []
        for row_end in plan.row_ends:
            span.append(ids["patch_start"])
            span.extend([self._image_token_id] * self._crop_rows)
            span.append(ids["patch_end"])
            if row_end:
                span.append(ids["patch_newline"])
        span.append(ids["im_start"])
        span.extend([self._image_token_id] * self._overview_rows)
        span.append(ids["im_end"])
        return span

    def _normalized(self, image: Image.Image, side: int) -> torch.Tensor:
        """``[3, side, side]`` float32: scale to [0, 1], normalize, then resize."""
        hwc = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
        pixels = hwc.permute(2, 0, 1).contiguous().to(torch.float32).div(255)
        pixels = (pixels - self._mean) / self._std
        if tuple(pixels.shape[-2:]) != (side, side):
            pixels = F.interpolate(
                pixels[None],
                size=(side, side),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
        return pixels

    def tiles(self, image: ImageInput) -> Tuple[torch.Tensor, torch.Tensor, TilePlan]:
        """One image's float32 overview ``[3, S, S]``, crops ``[n, 3, 504, 504]`` and plan."""
        image = _as_rgb_image(image)
        plan = tile_plan(image.width, image.height, self._overview_side)
        if plan.pad_side is not None:
            square = Image.new(image.mode, (plan.pad_side, plan.pad_side), 0)
            square.paste(image, (0, 0))
            image = square
        if image.size != plan.size:
            image = image.resize(plan.size, Image.Resampling.BILINEAR)
        overview = self._normalized(image, self._overview_side)
        crops = torch.empty(0, 3, CROP_SIDE, CROP_SIDE, dtype=torch.float32)
        if plan.boxes:
            canvas = image
            if canvas.size != plan.canvas:
                canvas = canvas.resize(plan.canvas, Image.Resampling.BILINEAR)
            crops = torch.stack(
                [self._normalized(canvas.crop(box), CROP_SIDE) for box in plan.boxes]
            )
        return overview, crops, plan

    def _process_images(
        self, images: Sequence[ImageInput]
    ) -> Tuple[Dict[str, Any], List[TilePlan]]:
        overviews, crops, plans = [], [], []
        for image in images:
            overview, image_crops, plan = self.tiles(image)
            overviews.append(overview)
            crops.append(image_crops)
            plans.append(plan)
        data: Dict[str, Any] = {
            "pixel_values": torch.stack(overviews).to(self._dtype),
            "num_patches": [len(plan.boxes) for plan in plans],
        }
        all_crops = torch.cat(crops)
        if all_crops.shape[0]:
            data["patch_pixel_values"] = all_crops.to(self._dtype)
        return data, plans

    def expand_image_placeholders(
        self, token_ids: Sequence[int], plans: Sequence[TilePlan]
    ) -> List[int]:
        """Replace the i-th ``<im_patch>`` of ``token_ids`` with image i's span."""
        positions = [i for i, token in enumerate(token_ids) if token == self._image_token_id]
        if len(positions) != len(plans):
            raise ValueError(
                f"the prompt has {len(positions)} image placeholders for {len(plans)} images; "
                "render it with the checkpoint's chat template, which emits one <im_patch> "
                "per image content item"
            )
        expanded: List[int] = []
        previous = 0
        for position, plan in zip(positions, plans):
            expanded.extend(token_ids[previous:position])
            expanded.extend(self._image_span(plan))
            previous = position + 1
        expanded.extend(token_ids[previous:])
        return expanded

    @torch.inference_mode()
    def call_with_text_prompt(
        self, inputs: TextPrompt, sampling_params: Optional[SamplingParams]
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        add_special_tokens = True if sampling_params is None else sampling_params.add_special_tokens
        truncate = None if sampling_params is None else sampling_params.truncate_prompt_tokens
        mm_data = inputs.get("multi_modal_data") or {}
        unsupported = sorted(set(mm_data) - {"image"})
        if unsupported:
            raise ValueError(f"Step-5-Preview takes image input only; got {unsupported}")
        images = mm_data.get("image")
        if images is not None and not isinstance(images, (list, tuple)):
            images = [images]
        if not images:
            token_ids = tokenize_prompt(
                self._tokenizer,
                inputs["prompt"],
                add_special_tokens=add_special_tokens,
                truncate_prompt_tokens=truncate,
                prefix_token_cache=self._prefix_token_cache,
            )
            return token_ids, None
        if truncate is not None:
            raise ValueError(
                "truncate_prompt_tokens would cut image spans; it is not supported with images"
            )
        token_ids = tokenize_prompt(
            self._tokenizer, inputs["prompt"], add_special_tokens=add_special_tokens
        )
        image_data, plans = self._process_images(images)
        token_ids = self.expand_image_placeholders(token_ids, plans)
        return token_ids, {"multimodal_data": {"image": image_data}}


register_architecture_input_processor(
    ARCHITECTURE,
    Step5PreviewInputProcessor,
    model_type=MODEL_TYPE,
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<im_patch>"},
        content_format=ContentFormat.OPENAI,
    ),
)
