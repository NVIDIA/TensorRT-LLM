# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 Klein pipelines for text-to-image and reference-image generation.

Sampling follows the Diffusers FLUX.3 modular pipelines: Klein Base uses classifier-free guidance,
Klein is a step-distilled flow map sampled without it.
"""

import dataclasses
import json
import math
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.utils.torch_utils import randn_tensor
from PIL import ImageOps
from torch import Tensor

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, RefSlotSpec, RoleSpec
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import make_noise_generator
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from .packed import pack
from .text_encoder import TEXT_MAX_TOKENS, TEXT_PAD_MULTIPLE, Flux3TextEncoder
from .transformer_flux3 import IMAGE, IMAGE_COND, IMAGE_ROPE_SCALE, TEXT, Flux3Transformer2DModel
from .variants import check_variant, load_variant_weights, resolve_variant_quantization_config

# Constant timestep shift of the FLUX.3 sampling recipe.
SCHEDULER_SHIFT = 6.93
DEFAULT_SIZE = 1024
# One image token covers 16 x 16 pixels: 8x VAE downsampling, then 2 x 2 latent packing.
PATCH_SIZE = 16
# Reference rules of the Diffusers Flux3ImageProcessor and BFL's reference preparation.
MAX_REFERENCES = 10
SINGLE_REFERENCE_MAX_PIXELS = 4 * 1024 * 1024
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024
MIN_REFERENCE_SIDE, MAX_REFERENCE_SIDE = 64, 4096
# Every padded text length is a multiple of this, so the sequence-parallel size must divide it.
TEXT_TOKEN_MULTIPLE = math.gcd(TEXT_PAD_MULTIPLE, TEXT_MAX_TOKENS)
# Text windows that warmup compiles and captures, for t2i and the edit at each resolution.
WARMUP_TEXT_WINDOWS = (TEXT_PAD_MULTIPLE, 2 * TEXT_PAD_MULTIPLE)
_PIPELINE_CLASS_NAMES = ("Flux3ModularPipeline", "Flux3DistilledModularPipeline")

_TRANSFORMER_PATTERN = "transformer/*"
_DOWNLOAD_PATTERNS = [
    "modular_model_index.json",
    _TRANSFORMER_PATTERN,
    "vae/*",
    "text_encoder/*",
    "LICENSE.md",
    "NOTICE.md",
    "README.md",
]
_COND, _UNCOND = "conditional", "unconditional"

ReferenceInput = Union[PIL.Image.Image, bytes]


class Flux3RequestError(ValueError):
    """A request FLUX.3 rejects because of its own content; served as a client error."""


class UnsupportedNumImagesError(Flux3RequestError):
    """A request for other than one image per prompt."""


@dataclasses.dataclass(frozen=True, eq=False)
class Flux3References:
    """Preprocessed reference images, each [1, 3, H, W] in [-1, 1], and the output crop box.

    ``crop`` (left, top, right, bottom) crops the decoded image back to a single padded reference
    when the output size was taken from it; otherwise it is None.
    """

    images: Tuple[Tensor, ...]
    crop: Optional[Tuple[int, int, int, int]] = None

    @property
    def shapes(self) -> Tuple[Tuple[int, int], ...]:
        return tuple((int(image.shape[-2]), int(image.shape[-1])) for image in self.images)

    @property
    def num_tokens(self) -> int:
        return sum((h // PATCH_SIZE) * (w // PATCH_SIZE) for h, w in self.shapes)


# =============================================================================
# Reference images
# =============================================================================


def load_reference_images(
    images: Union[ReferenceInput, Sequence[ReferenceInput]],
) -> List[PIL.Image.Image]:
    """Decode references as Diffusers' load_image does: EXIF orientation applied, then RGB."""
    items = list(images) if isinstance(images, (list, tuple)) else [images]
    decoded = []
    for index, item in enumerate(items):
        if not isinstance(item, (bytes, PIL.Image.Image)):
            raise Flux3RequestError(
                "FLUX.3 reference images must be PIL images or encoded bytes; "
                f"item {index} is {type(item).__name__}."
            )
        try:
            image = PIL.Image.open(BytesIO(item)) if isinstance(item, bytes) else item
            decoded.append(ImageOps.exif_transpose(image).convert("RGB"))
        except (OSError, PIL.Image.DecompressionBombError) as exc:
            raise Flux3RequestError(
                f"Unable to decode FLUX.3 reference image {index}: {exc}"
            ) from exc
    return decoded


def _single_reference_size(width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """(fitted, canvas) sizes as (width, height); the canvas is the fitted size on the token grid."""
    scale = min(
        1.0,
        math.sqrt(SINGLE_REFERENCE_MAX_PIXELS / (width * height)),
        MAX_REFERENCE_SIDE / max(width, height),
    )
    while True:
        fitted = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(
            min(MAX_REFERENCE_SIDE, max(MIN_REFERENCE_SIDE, -(-side // PATCH_SIZE) * PATCH_SIZE))
            for side in fitted
        )
        if canvas[0] * canvas[1] <= SINGLE_REFERENCE_MAX_PIXELS:
            return fitted, canvas
        # Rounding up to the grid can push the canvas over the cap; back off until it fits.
        scale *= 0.995


def _pad_single_reference(
    image: PIL.Image.Image,
) -> Tuple[PIL.Image.Image, Optional[Tuple[int, int, int, int]]]:
    """Fit a single reference and reflect-pad it, centered, to its canvas; returns the image box."""
    fitted, (canvas_width, canvas_height) = _single_reference_size(*image.size)
    if fitted != image.size:
        image = image.resize(fitted, PIL.Image.Resampling.LANCZOS)
    if (canvas_width, canvas_height) == image.size:
        return image, None
    left, top = (canvas_width - image.width) // 2, (canvas_height - image.height) // 2
    padding = (
        (top, canvas_height - image.height - top),
        (left, canvas_width - image.width - left),
        (0, 0),
    )
    # Reflection excludes the edge pixel, so references smaller than a token are mirrored.
    mode = "reflect" if min(image.size) > PATCH_SIZE else "symmetric"
    padded = PIL.Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))
    return padded, (left, top, left + image.width, top + image.height)


def _crop_multi_reference(image: PIL.Image.Image) -> PIL.Image.Image:
    """Fit one reference of several under 1 MP (sides floored) and center-crop it to the grid."""
    width, height = image.size
    if width * height > MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
    crop_width = width // PATCH_SIZE * PATCH_SIZE
    crop_height = height // PATCH_SIZE * PATCH_SIZE
    if not crop_width or not crop_height:
        raise Flux3RequestError(
            f"Each reference of a FLUX.3 multi-reference edit needs sides of at least "
            f"{PATCH_SIZE} pixels, got {width} x {height}."
        )
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def _image_to_tensor(image: PIL.Image.Image) -> Tensor:
    """RGB image to [1, 3, H, W] in [-1, 1], with the Diffusers VaeImageProcessor arithmetic."""
    array = np.array(image).astype(np.float32) / 255.0
    return (2.0 * torch.from_numpy(array).permute(2, 0, 1)[None] - 1.0).contiguous()


def preprocess_references(images: Sequence[PIL.Image.Image]) -> Flux3References:
    """Prepare decoded references with the Diffusers and BFL rules.

    One reference is fitted so that its 16-pixel canvas is at most 4 MP (sides 64-4096) and
    reflect-padded to it; several are each fitted under 1 MP and center-cropped to the grid.
    """
    if not images:
        raise Flux3RequestError("FLUX.3 image-to-image needs at least one reference image.")
    if len(images) > MAX_REFERENCES:
        raise Flux3RequestError(
            f"FLUX.3 accepts at most {MAX_REFERENCES} reference images, got {len(images)}."
        )
    if len(images) == 1:
        image, crop = _pad_single_reference(images[0])
        return Flux3References((_image_to_tensor(image),), crop)
    return Flux3References(tuple(_image_to_tensor(_crop_multi_reference(i)) for i in images))


# =============================================================================
# Scheduler, positions and latent layout
# =============================================================================


def flux3_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """Flow-match Euler scheduler with the FLUX.3 constant shift."""
    return FlowMatchEulerDiscreteScheduler(shift=SCHEDULER_SHIFT)


def set_flux3_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler, num_inference_steps: int, device: torch.device
) -> Tensor:
    """Set shifted linear sigmas from 1 down to 1/N; returns the timesteps (sigmas x 1000)."""
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    scheduler.set_begin_index(0)
    return scheduler.timesteps


def _image_ids(height: int, width: int, device: torch.device, l_coord: int = 0) -> Tensor:
    """(t, h, w, l) ids [height * width, 4] of a token grid, h and w scaled for RoPE."""
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(height), torch.arange(width), torch.tensor([l_coord])
    ).float()
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids.to(device)


def _text_ids(length: int, device: torch.device) -> Tensor:
    """(t, h, w, l) ids [length, 4] of text tokens, which advance along l."""
    ids = torch.zeros(length, 4, device=device)
    ids[:, 3] = torch.arange(length, device=device)
    return ids


def _patchify(latents: Tensor) -> Tensor:
    """VAE latents [B, C, H, W] to packed latents [B, 4C, H / 2, W / 2]."""
    batch, channels, height, width = latents.shape
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    return latents.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)


def _unpatchify(latents: Tensor) -> Tensor:
    """Packed latents [B, 4C, H, W] to VAE latents [B, C, 2H, 2W]."""
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    return latents.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, height * 2, width * 2)


def _transformer_inputs(inputs: Dict[str, Any], prepared: Dict[str, Tensor]) -> Dict[str, Any]:
    """The packed inputs and prepared tensors the transformer forward reads.

    A prepared stream replaces its modality's raw tokens, timesteps and RoPE tables.
    """
    call = {"layout": inputs["layout"], **prepared}
    for name, lengths in inputs["layout"]:
        if not any(lengths):
            continue
        if f"_flux3_{name}_tokens" in prepared:
            del call[f"_flux3_{name}_cos"], call[f"_flux3_{name}_sin"]
        else:
            call[f"x_{name}"] = inputs[f"x_{name}"]
            call[f"x_{name}_timesteps"] = inputs[f"x_{name}_timesteps"]
    return call


def _read_manifest(checkpoint_dir: str) -> Dict[str, Any]:
    """The checkpoint's model_index.json or modular_model_index.json, else an empty dict."""
    for name in ("model_index.json", "modular_model_index.json"):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return {}


def _manifest_subfolder(checkpoint_dir: str, component: str, default: str) -> str:
    """Subfolder that the manifest names for ``component``, else ``default``."""
    entry = _read_manifest(checkpoint_dir).get(component)
    spec = entry[2] if isinstance(entry, list) and len(entry) > 2 else None
    return (spec.get("subfolder") if isinstance(spec, dict) else None) or default


def _format_choices(values: Sequence[int]) -> str:
    """``1, 2, 4 or 8`` for (1, 2, 4, 8)."""
    words = [str(value) for value in values]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} or {words[-1]}"


# =============================================================================
# Pipelines
# =============================================================================


@register_pipeline(
    "Flux3ModularPipeline",
    hf_ids=["black-forest-labs/flux-3-image-klein-base"],
    defaults={"variant": None},
    download_patterns=_DOWNLOAD_PATTERNS,
    doc="Black Forest Labs FLUX.3 Klein Base (text-to-image and reference-image generation).",
)
class Flux3Pipeline(BasePipeline):
    """FLUX.3 Klein Base: classifier-free guidance over packed per-sample prompts and references."""

    supports_image_edit = True
    derive_output_size_from_reference = True
    flow_map = False
    default_num_inference_steps = 50
    default_guidance_scale: Optional[float] = 3.5

    def __init__(self, pipeline_config):
        self._check_pipeline_config(pipeline_config)
        super().__init__(pipeline_config)
        self._warmup_reference_keys: Set[tuple] = set()

    @classmethod
    def resolve_variant(cls, config: Any) -> Type["Flux3Pipeline"]:
        """Keep the class of an exact registered manifest name; otherwise choose it by flow_map."""
        pretrained = config.primary_pretrained_config
        manifest = _read_manifest(getattr(pretrained, "_name_or_path", None) or "")
        if manifest.get("_class_name") in _PIPELINE_CLASS_NAMES:
            return cls
        return Flux3DistilledPipeline if getattr(pretrained, "flow_map", False) else Flux3Pipeline

    @classmethod
    def hf_download_patterns(
        cls, patterns: List[str], pipeline_config: Dict[str, Any]
    ) -> List[str]:
        """With a variant, fetch its weights in place of the BF16 transformer weights."""
        variant = check_variant(pipeline_config.get("variant"))
        if variant is None:
            return patterns
        kept = [pattern for pattern in patterns if pattern != _TRANSFORMER_PATTERN]
        return [*kept, "transformer/config.json", f"variants/{variant}/*"]

    @classmethod
    def resolve_component_configs(
        cls,
        checkpoint_dir: str,
        component_configs: Dict[str, Dict[str, Any]],
        pipeline_config: Dict[str, Any],
        quant_config: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Give the transformer a selected variant's quantization_config."""
        variant = check_variant(pipeline_config.get("variant"))
        if variant is None:
            return component_configs
        user_algo = (
            quant_config.get("quant_algo")
            if isinstance(quant_config, dict)
            else getattr(quant_config, "quant_algo", None)
        )
        if user_algo is not None:
            raise ValueError(
                f"FLUX.3 variant {variant!r} is pre-quantized; remove quant_config.quant_algo or "
                "pipeline_config.variant."
            )
        name = PipelineComponent.TRANSFORMER.value
        transformer = dict(component_configs[name])
        transformer["quantization_config"] = resolve_variant_quantization_config(
            checkpoint_dir, variant, transformer
        )
        return {**component_configs, name: transformer}

    @classmethod
    def _check_pipeline_config(cls, config: Any) -> None:
        """Reject checkpoints and features this pipeline does not support, before weights load."""
        flow_map = bool(getattr(config.primary_pretrained_config, "flow_map", False))
        if flow_map != cls.flow_map:
            expected = "Flux3DistilledModularPipeline" if flow_map else "Flux3ModularPipeline"
            raise ValueError(
                f"{cls.__name__} expects transformer/config.json flow_map={cls.flow_map}, "
                f"got {flow_map}; set the manifest _class_name to {expected}."
            )
        if config.cpu_offload_config.enable:
            raise ValueError(
                "FLUX.3 does not support CPU offloading; set cpu_offload_config.enable to false."
            )
        if config.parallel.parallel_vae_size > 1:
            raise ValueError(
                "FLUX.3 does not support parallel VAE; set parallel_config.parallel_vae_size to 1."
            )
        if config.cache is not None:
            raise ValueError(
                f"FLUX.3 does not support {config.cache.cache_backend} yet; remove cache_config."
            )
        if config.attention.sparse_attention_config is not None:
            raise ValueError(
                "FLUX.3 does not support sparse attention yet; "
                "remove attention_config.sparse_attention_config."
            )
        quant_algo = config.quant_config.quant_algo
        if check_variant(config.extra_attrs.get("variant")) is None and not (
            quant_algo is None
            or (quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN and config.dynamic_weight_quant)
        ):
            mode = "" if config.dynamic_weight_quant else "static "
            raise ValueError(
                "FLUX.3 quantizes BF16 weights on load only with quant_config.quant_algo "
                f"FP8_PER_CHANNEL_PER_TOKEN, got {mode}{quant_algo}; for pre-quantized NVFP4 or FP8 "
                "rowwise weights, set pipeline_config.variant to 'nvfp4' or 'fp8r'."
            )
        if config.attention.quant_attention_config is not None:
            raise ValueError(
                "FLUX.3 does not support quantized attention yet; "
                "remove attention_config.quant_attention_config."
            )
        if config.runtime_lora is not None:
            raise ValueError(
                "FLUX.3 does not support runtime LoRA yet; remove runtime_lora_config."
            )
        vgm = config.visual_gen_mapping
        if flow_map and vgm is not None and vgm.cfg_size > 1:
            raise ValueError(
                "FLUX.3 Klein samples without classifier-free guidance, so CFG parallelism "
                "does not apply; set parallel_config.cfg_size to 1."
            )
        if vgm is not None and vgm.cp_size > 1:
            raise ValueError(
                "FLUX.3 does not support Ring Attention or Attention2D; set "
                "parallel_config.ring_size to 1 and attn2d_size to [1, 1]."
            )
        seq_size = vgm.seq_size if vgm is not None else 1
        num_heads = config.primary_pretrained_config.num_attention_heads
        # Ulysses splits attention heads, and every padded prompt window, across its ranks.
        max_size = math.gcd(TEXT_TOKEN_MULTIPLE, num_heads)
        if max_size % seq_size:
            sizes = [size for size in range(1, max_size + 1) if max_size % size == 0]
            raise ValueError(
                f"FLUX.3 Ulysses splits {num_heads} attention heads and prompts padded to "
                f"multiples of {TEXT_PAD_MULTIPLE} tokens (at most {TEXT_MAX_TOKENS}) across "
                f"ranks, so ulysses_size must be {_format_choices(sizes)}; got {seq_size}."
            )

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def _init_transformer(self) -> None:
        self.transformer = Flux3Transformer2DModel(
            self.pipeline_config.model_configs[PipelineComponent.TRANSFORMER.value]
        )

    def load_transformer_weights(self, checkpoint_dir: str) -> Dict[str, Any]:
        """The Diffusers transformer weights, or those of the selected pre-quantized variant."""
        variant = self.pipeline_config.extra_attrs.get("variant")
        if variant is None:
            return super().load_transformer_weights(checkpoint_dir)
        return load_variant_weights(
            checkpoint_dir, variant, self.transformer.checkpoint_quantization(), self.mapping
        )

    def load_weights(self, weights: Dict[str, Any]) -> None:
        self.transformer.load_weights(weights.get(PipelineComponent.TRANSFORMER.value, weights))
        self.transformer.eval()

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        """Load the Qwen3-VL text encoder, the FP32 VAE and the scheduler."""
        skip_components = skip_components or []
        if PipelineComponent.TEXT_ENCODER not in skip_components:
            logger.info("Loading FLUX.3 text encoder...")
            text_encoder = PipelineComponent.TEXT_ENCODER.value
            processor_dir = _manifest_subfolder(
                checkpoint_dir, PipelineComponent.TOKENIZER.value, text_encoder
            )
            self.text_encoder = Flux3TextEncoder.from_pretrained(
                os.path.join(checkpoint_dir, text_encoder),
                os.path.join(checkpoint_dir, processor_dir),
                dtype=self.dtype,
                device=device,
            )
        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading FLUX.3 VAE...")
            self.vae = (
                AutoencoderKLFlux2.from_pretrained(
                    os.path.join(checkpoint_dir, PipelineComponent.VAE.value),
                    torch_dtype=torch.float32,
                )
                .to(device)
                .eval()
            )
        self.scheduler = flux3_scheduler()

    def torch_compile_options(self) -> Optional[Dict[str, Any]]:
        # Keep the BF16 rounding of intermediates that eager execution has.
        return {"emulate_precision_casts": True}

    # -------------------------------------------------------------------------
    # Request contract
    # -------------------------------------------------------------------------

    @property
    def default_generation_params(self) -> dict:
        params = {
            "height": DEFAULT_SIZE,
            "width": DEFAULT_SIZE,
            "num_inference_steps": self.default_num_inference_steps,
        }
        if self.default_guidance_scale is not None:
            params["guidance_scale"] = self.default_guidance_scale
        return params

    @property
    def ref_slot_specs(self) -> Dict[str, RefSlotSpec]:
        return {
            "image_reference": RefSlotSpec(
                modality="image",
                roles=[RoleSpec(role="reference", min=0, max=MAX_REFERENCES)],
            )
        }

    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        return [(DEFAULT_SIZE, DEFAULT_SIZE)]

    @property
    def default_warmup_num_frames(self) -> List[int]:
        return [1]

    @property
    def resolution_multiple_of(self) -> Tuple[int, int]:
        return (PATCH_SIZE, PATCH_SIZE)

    def validate_resolution(self, height: int, width: int, num_frames: int = 1) -> None:
        try:
            super().validate_resolution(height, width, num_frames)
        except ValueError as exc:
            raise Flux3RequestError(str(exc)) from exc
        self._check_sequence_parallel({"target": (height // PATCH_SIZE) * (width // PATCH_SIZE)})

    def warmup_cache_key(self, height: int, width: int, num_frames: int = 1) -> tuple:
        return (height, width)

    @staticmethod
    def _reference_cache_key(height: int, width: int, references: Flux3References) -> tuple:
        return (height, width, len(references.images), references.shapes)

    def request_warmup_cache_key(self, req: Any) -> tuple:
        references = req.prepared_inputs.get("references")
        if references is None:
            return super().request_warmup_cache_key(req)
        return self._reference_cache_key(req.params.height, req.params.width, references)

    def classify_request_failure(self, exc: BaseException) -> Optional[str]:
        return "client" if isinstance(exc, Flux3RequestError) else None

    @staticmethod
    def _check_num_images(num_images_per_prompt: int) -> None:
        if num_images_per_prompt != 1:
            raise UnsupportedNumImagesError(
                "FLUX.3 generates one image per request; num_images_per_prompt must be 1, "
                f"got {num_images_per_prompt}."
            )

    def prepare_request(self, req: Any) -> None:
        """Reject unsupported image counts, then preprocess references and resolve the size."""
        params = req.params
        self._check_num_images(params.num_images_per_prompt)
        if not params.image_reference:
            return
        references, params.height, params.width = self._prepare_references(
            [ref.content for ref in params.image_reference], params.height, params.width
        )
        req.prepared_inputs["references"] = references

    def infer(self, req: Any) -> PipelineOutput:
        params = req.params
        self._check_num_images(params.num_images_per_prompt)
        prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
        if len(prompts) != 1:
            raise Flux3RequestError(
                f"FLUX.3 generates from one prompt per request, got {len(prompts)}."
            )
        image = req.prepared_inputs.get("references")
        if image is None and params.image_reference:
            image = [ref.content for ref in params.image_reference]
        return self.forward(
            prompts[0],
            negative_prompt=params.negative_prompt,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            seed=params.seed,
            image=image,
        )

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------

    def warmup(self) -> None:
        self._warmup_reference_keys = set()
        super().warmup()
        self._warmed_up_shapes |= self._warmup_reference_keys

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        """Warm text-to-image and a same-size single-reference edit at each warmup text window.

        Klein Base pairs each window with the 80-token empty negative prompt: (80, 80), (160, 80).
        """
        reference = PIL.Image.new("RGB", (width, height), (128, 128, 128))
        references, _, _ = self._prepare_references([reference], height, width)
        self._warmup_reference_keys.add(self._reference_cache_key(height, width, references))
        prompts = [self._warmup_prompt(window) for window in WARMUP_TEXT_WINDOWS]
        for image in (None, references):
            for prompt in prompts:
                self.forward(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    seed=42,
                    image=image,
                )

    def _warmup_prompt(self, window: int) -> str:
        """The shortest prompt of repeated "warmup" words whose text window is ``window``."""
        words = ["warmup"]
        while self.text_encoder.tokenize(" ".join(words))[0].shape[1] < window:
            words.append("warmup")
        return " ".join(words)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def _prepare_references(
        self,
        image: Union[None, Flux3References, ReferenceInput, Sequence[ReferenceInput]],
        height: Optional[int],
        width: Optional[int],
    ) -> Tuple[Optional[Flux3References], int, int]:
        """References and output size; an omitted side takes the first reference's, else 1024.

        A given side drops the crop-back of a single padded reference, as in Diffusers.
        Flux3References from :meth:`prepare_request` already carry their final crop box.
        """
        if image is None:
            return None, height or DEFAULT_SIZE, width or DEFAULT_SIZE
        if isinstance(image, Flux3References):
            references = image
        else:
            references = preprocess_references(load_reference_images(image))
            if height is not None or width is not None:
                references = dataclasses.replace(references, crop=None)
        reference_height, reference_width = references.shapes[0]
        return references, height or reference_height, width or reference_width

    @property
    def _sequence_parallel_size(self) -> int:
        vgm = self.pipeline_config.visual_gen_mapping
        return vgm.seq_size if vgm is not None else 1

    def _check_sequence_parallel(self, token_counts: Dict[str, int]) -> None:
        """Reject token counts that the sequence-parallel size does not divide."""
        size = self._sequence_parallel_size
        for modality, count in token_counts.items():
            if count % size:
                raise Flux3RequestError(
                    f"FLUX.3 sequence parallelism splits each sample's {modality} tokens across "
                    f"{size} ranks, but this request has {count} {modality} tokens; an image "
                    f"token covers {PATCH_SIZE} x {PATCH_SIZE} pixels."
                )

    def _resolve_guidance(
        self, guidance_scale: Optional[float], negative_prompt: Optional[str]
    ) -> float:
        if self.flow_map:
            if guidance_scale not in (None, 1.0):
                raise Flux3RequestError(
                    "FLUX.3 Klein samples without classifier-free guidance; omit guidance_scale."
                )
            if negative_prompt:
                raise Flux3RequestError(
                    "FLUX.3 Klein samples without classifier-free guidance and takes no "
                    "negative prompt."
                )
            return 1.0
        if guidance_scale is None:
            guidance_scale = self.default_guidance_scale
        if negative_prompt and math.isclose(guidance_scale, 1.0):
            logger.warning("FLUX.3 ignores negative_prompt at guidance_scale 1 (no CFG).")
        return guidance_scale

    def _cfg_samples(self, guidance_scale: float) -> Tuple[Tuple[str, ...], bool]:
        """Samples this rank runs and whether the CFG ranks exchange their predictions."""
        if self.flow_map or math.isclose(guidance_scale, 1.0):
            return (_COND,), False
        vgm = self.pipeline_config.visual_gen_mapping
        if vgm is not None and vgm.cfg_size >= 2:
            return ((_COND,) if vgm.is_cfg_conditional else (_UNCOND,)), True
        return (_COND, _UNCOND), False

    @torch.inference_mode()
    def forward(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int,
        image: Union[None, Flux3References, ReferenceInput, Sequence[ReferenceInput]] = None,
    ) -> PipelineOutput:
        """Generate one image; with reference ``image`` inputs, an image-to-image edit.

        ``image`` takes up to 10 PIL images or encoded images, or references from prepare_request.
        Returns a PipelineOutput whose image is uint8 [1, H, W, 3] on the decoding rank.
        """
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        start = time.time()
        if not isinstance(prompt, str):
            raise Flux3RequestError("FLUX.3 takes one prompt string per request.")
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps
        if num_inference_steps < 1:
            raise Flux3RequestError(
                f"num_inference_steps must be positive, got {num_inference_steps}."
            )
        guidance_scale = self._resolve_guidance(guidance_scale, negative_prompt)

        # Size and token checks come before any encoder runs.
        references, height, width = self._prepare_references(image, height, width)
        self.validate_resolution(height, width, 1)
        if references is not None:
            self._check_sequence_parallel({"reference": references.num_tokens})
        samples, cfg_parallel = self._cfg_samples(guidance_scale)
        prompts = {_COND: prompt, _UNCOND: negative_prompt or ""}
        tokens = [self.text_encoder.tokenize(prompts[sample]) for sample in samples]
        self._check_sequence_parallel(
            {f"{sample} text": ids.shape[1] for sample, (ids, _) in zip(samples, tokens)}
        )

        contexts = [self.text_encoder.encode(ids, mask)[0].to(self.dtype) for ids, mask in tokens]
        conditioning = self._encode_references(references) if references is not None else None
        latents, latent_ids = self._prepare_latents(height, width, seed)

        timer.mark_denoise_start()
        latents = self._denoise(
            latents,
            latent_ids,
            contexts,
            conditioning,
            num_inference_steps,
            guidance_scale,
            cfg_parallel,
        )
        timer.mark_post_start()
        crop = references.crop if references is not None else None
        output = self.decode_latents(latents, lambda x: self._decode(x, height, width, crop))
        timer.mark_end()
        if self.rank == 0:
            logger.info(
                f"FLUX.3 generated {height}x{width} with {num_inference_steps} steps and "
                f"{len(references.images) if references is not None else 0} reference(s) "
                f"in {time.time() - start:.2f}s"
            )
        return timer.fill(PipelineOutput(image=output))

    def _prepare_latents(self, height: int, width: int, seed: int) -> Tuple[Tensor, Tensor]:
        """Initial noise tokens [H/16 * W/16, C] in the compute dtype and their ids."""
        latent_height, latent_width = height // PATCH_SIZE, width // PATCH_SIZE
        channels = self.transformer.content_channels[IMAGE]
        generator = make_noise_generator(seed, self.device)
        # Diffusers' shape and FP32 draw; a seed matches it on the same generator device
        # (TLLM_VISUALGEN_CPU_RNG=1 matches a CPU torch.Generator).
        noise = randn_tensor(
            (1, channels, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        latents = noise.to(self.dtype).flatten(2).transpose(1, 2)[0].contiguous()
        return latents, _image_ids(latent_height, latent_width, self.device)

    def _encode_references(self, references: Flux3References) -> Tuple[Tensor, Tensor]:
        """Normalized packed reference tokens [N, C] and ids; references differ on the l axis."""
        vae = self.vae
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(self.device, vae.dtype)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            self.device, vae.dtype
        )
        tokens, ids = [], []
        for index, image in enumerate(references.images):
            latents = vae.encode(image.to(self.device, vae.dtype)).latent_dist.mode()
            latents = (_patchify(latents) - mean) / std
            latent_height, latent_width = latents.shape[-2:]
            tokens.append(latents.flatten(2).transpose(1, 2)[0])
            ids.append(_image_ids(latent_height, latent_width, self.device, l_coord=-(1 + index)))
        return torch.cat(tokens).to(self.dtype), torch.cat(ids)

    def _denoise(
        self,
        latents: Tensor,
        latent_ids: Tensor,
        contexts: List[Tensor],
        conditioning: Optional[Tuple[Tensor, Tensor]],
        num_inference_steps: int,
        guidance_scale: float,
        cfg_parallel: bool,
    ) -> Tensor:
        """Euler steps over the image tokens of one packed sample per text context."""
        samples = []
        for context in contexts:
            sample = {
                IMAGE: (latents, latent_ids),
                TEXT: (context, _text_ids(context.shape[0], self.device)),
            }
            if conditioning is not None:
                sample[IMAGE_COND] = conditioning
            samples.append(sample)
        inputs = pack(samples, self.transformer.packed_channels).kwargs()
        call = _transformer_inputs(inputs, self.transformer.prepare_static_inputs(**inputs))

        num_samples = len(samples)
        timesteps = set_flux3_timesteps(self.scheduler, num_inference_steps, self.device)
        targets = None
        if self.flow_map:
            # A flow-map step jumps to the next timestep, and the last one to zero.
            targets = torch.cat([timesteps[1:], timesteps.new_zeros(1)]) / 1000
        for step, timestep in self._profile_denoise_steps(timesteps):
            call["x_image"] = latents.repeat(num_samples, 1)
            call["x_image_timesteps"] = (timestep / 1000).repeat(num_samples)
            if targets is not None:
                call["target_timesteps"] = targets[step].repeat(num_samples)
            prediction = self.transformer(**call)[f"x_{IMAGE}"]
            velocity = self._guide(prediction, num_samples, guidance_scale, cfg_parallel)
            latents, _, _ = self._scheduler_step(
                latents, {}, velocity, {}, timestep, self.scheduler, {}
            )
        return latents

    def _guide(
        self, prediction: Tensor, num_samples: int, guidance_scale: float, cfg_parallel: bool
    ) -> Tensor:
        """Classifier-free guidance, uncond + scale * (cond - uncond), as in Diffusers."""
        if cfg_parallel:
            vgm = self.pipeline_config.visual_gen_mapping
            gathered = [torch.empty_like(prediction) for _ in range(vgm.cfg_size)]
            dist.all_gather(gathered, prediction.contiguous(), group=vgm.cfg_group)
            conditional, unconditional = gathered[0], gathered[1]
        elif num_samples == 2:
            conditional, unconditional = prediction.chunk(2)
        else:
            return prediction
        return unconditional + guidance_scale * (conditional - unconditional)

    def _decode(
        self,
        latents: Tensor,
        height: int,
        width: int,
        crop: Optional[Tuple[int, int, int, int]],
    ) -> Tensor:
        """Image tokens to uint8 [1, H, W, 3], cropped back to a single padded reference."""
        vae = self.vae
        latents = latents.reshape(1, height // PATCH_SIZE, width // PATCH_SIZE, -1)
        latents = latents.permute(0, 3, 1, 2)
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = _unpatchify(latents * std + mean)
        image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
        if crop is not None:
            left, top, right, bottom = crop
            image = image[..., top:bottom, left:right]
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return (image.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)


@register_pipeline(
    "Flux3DistilledModularPipeline",
    hf_ids=["black-forest-labs/flux-3-image-klein"],
    defaults={"variant": None},
    download_patterns=_DOWNLOAD_PATTERNS,
    doc="Black Forest Labs FLUX.3 Klein, step-distilled (text-to-image and reference-image "
    "generation).",
)
class Flux3DistilledPipeline(Flux3Pipeline):
    """FLUX.3 Klein: a step-distilled flow map, sampled without classifier-free guidance."""

    flow_map = True
    default_num_inference_steps = 16
    default_guidance_scale = None
