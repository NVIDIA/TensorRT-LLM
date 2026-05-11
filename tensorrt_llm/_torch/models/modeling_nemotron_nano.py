# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import copy
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from einops import rearrange as einops_rearrange
from PIL import Image

from tensorrt_llm._torch.models.checkpoints import NemotronHHfWeightMapper
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ...inputs import (
    AudioData,
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    compute_retained_tokens_count,
    compute_retained_tokens_from_tubelet_budget,
    compute_retention_mask,
    register_input_processor,
)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from .modeling_parakeet import ParakeetExtractor, ProjectedParakeet
from .modeling_radio import RADIOVisionModel, calc_seq_lens
from .modeling_utils import register_auto_model

# Set max_num_tiles to 1 for video modality, to match the training behavior.
VIDEO_MAX_NUM_TILES = 1
IMAGE_PLACEHOLDER = "<image>"
VIDEO_PLACEHOLDER = "<video>"
AUDIO_PLACEHOLDER = "<so_embedding>"


def _compute_aspect_preserving_size(
    orig_w: int,
    orig_h: int,
    target_num_patches: int,
    patch_size: int,
    downsample_ratio: float,
) -> Tuple[int, int]:
    """Compute target pixel dimensions preserving aspect ratio.

    Mirrors Megatron-LM / vLLM video frame resizing: target area in patch-grid
    space is *target_num_patches*, distributed according to the source aspect
    ratio, then snapped to a multiple of the required divisor (2 for pixel-shuffle).

    Returns:
        (target_w, target_h) in pixels.
    """
    aspect_wh = orig_w / max(orig_h, 1)
    ph = round(math.sqrt(target_num_patches / aspect_wh))
    pw = round(math.sqrt(target_num_patches * aspect_wh))
    ph = max(ph, 1)
    pw = max(pw, 1)

    reduction_factor = int(round(1 / downsample_ratio))
    required_divisor = reduction_factor
    if required_divisor > 1:
        rem_h = ph % required_divisor
        rem_w = pw % required_divisor
        ph_up = ph + (required_divisor - rem_h if rem_h else 0)
        ph_down = ph - rem_h
        pw_up = pw + (required_divisor - rem_w if rem_w else 0)
        pw_down = pw - rem_w
        if ph_up * pw_up <= target_num_patches:
            ph, pw = ph_up, pw_up
        else:
            ph = max(required_divisor, ph_down)
            pw = max(required_divisor, pw_down)

    return pw * patch_size, ph * patch_size  # (width, height) in pixels


def get_video_target_size_and_feature_size(
    orig_w: int,
    orig_h: int,
    target_patches: int,
    maintain_aspect_ratio: bool,
    patch_size: int,
    downsample_ratio: float,
) -> Tuple[int, int, int]:
    """Compute target (width, height) and feature_size for video resize.

    Returns:
        (target_w, target_h, feature_size) where feature_size is the
        post-pixel-shuffle token count per frame.
    """
    if maintain_aspect_ratio:
        target_w, target_h = _compute_aspect_preserving_size(
            orig_w=orig_w,
            orig_h=orig_h,
            target_num_patches=target_patches,
            patch_size=patch_size,
            downsample_ratio=downsample_ratio,
        )
    else:
        reduction_factor = int(round(1 / downsample_ratio))
        side = int(math.sqrt(target_patches))
        side = max(reduction_factor, (side // reduction_factor) * reduction_factor)
        target_w = side * patch_size
        target_h = side * patch_size

    feature_size = int((target_h // patch_size) * downsample_ratio) * int(
        (target_w // patch_size) * downsample_ratio
    )
    return target_w, target_h, feature_size


def _media_to_raw_chw(media: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
    """Convert a single PIL Image or [0,1] tensor to a CHW float tensor in [0, 255]."""
    if isinstance(media, Image.Image):
        return _pil_to_chw_tensor(media)
    if isinstance(media, torch.Tensor):
        return media * 255.0
    raise TypeError(f"Unsupported media type: {type(media)}")


def _video_frames_to_raw_tensor(
    video_frames: List[Union[Image.Image, torch.Tensor]],
) -> torch.Tensor:
    """Convert video frames (PIL or tensor) to a stacked [N, 3, H, W] float tensor in [0, 255]."""
    return torch.stack([_media_to_raw_chw(f) for f in video_frames])


def video_to_pixel_values(
    video_frames: List[Union[Image.Image, torch.Tensor]],
    *,
    input_size: int,
    video_target_num_patches: Optional[int] = None,
    video_maintain_aspect_ratio: bool = False,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    norm_mean: Optional[torch.Tensor] = None,
    norm_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert video frames (PIL Images or tensors) to a normalized pixel-value tensor.

    Resizes via bicubic interpolation, clamps overshoot, rescales to [0, 1],
    then applies mean/std normalization.

    Returns:
        Tensor of shape [num_frames, 3, H, W].
    """
    video_tensor = _video_frames_to_raw_tensor(video_frames)

    if video_target_num_patches is not None:
        target_w, target_h, _ = get_video_target_size_and_feature_size(
            orig_w=video_tensor.shape[3],
            orig_h=video_tensor.shape[2],
            target_patches=video_target_num_patches,
            maintain_aspect_ratio=video_maintain_aspect_ratio,
            patch_size=patch_size,
            downsample_ratio=downsample_ratio,
        )
    else:
        target_h, target_w = input_size, input_size

    if norm_mean is not None and norm_std is not None:
        norm_mean = norm_mean.to(video_tensor.device)
        norm_std = norm_std.to(video_tensor.device)
    return _resize_and_normalize(video_tensor, target_h, target_w, norm_mean, norm_std)


def _get_media_dimensions(media: Union[Image.Image, torch.Tensor]) -> Tuple[int, int]:
    """Return (width, height) for a PIL Image or a CHW tensor."""
    if isinstance(media, torch.Tensor):
        return media.shape[2], media.shape[1]
    return media.width, media.height


def _pil_to_chw_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a CHW float tensor in [0, 255]."""
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    return torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float()


def _resize_and_normalize(
    tensor: torch.Tensor,
    target_h: int,
    target_w: int,
    norm_mean: Optional[torch.Tensor] = None,
    norm_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bicubic resize + rescale to [0, 1] + optional mean/std normalization."""
    needs_unsqueeze = tensor.ndim == 3
    if needs_unsqueeze:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-2] != target_h or tensor.shape[-1] != target_w:
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    # Clamp bicubic overshoot to valid pixel range and rescale to [0, 1].
    tensor = tensor.clamp(0, 255).div_(255.0)
    if norm_mean is not None and norm_std is not None:
        tensor = tensor.sub_(norm_mean).div_(norm_std)
    if needs_unsqueeze:
        tensor = tensor.squeeze(0)
    return tensor


@dataclass
class DynamicResolutionParams:
    media: Union[Image.Image, torch.Tensor]
    num_tiles: int
    num_embeddings: int
    patch_size: Tuple[int, int]  # (width_patches, height_patches)


class DynamicResolutionImageTiler:
    """Adaptive image sizing for dynamic resolution encoding.

    Instead of the InternVL-style fixed-tile approach, dynamic resolution
    scales each image to a target size based on a token budget, preserving
    aspect ratio and using pixel shuffle for downsampling.
    """

    def __init__(
        self,
        *,
        max_model_len: int,
        patch_size: int,
        min_num_patches: int,
        max_num_patches: int,
        downsample_ratio: float,
        norm_mean: Sequence[float],
        norm_std: Sequence[float],
        factor_max: float = 1.0,
    ) -> None:
        self._patch_size = patch_size
        self._max_model_len = max_model_len
        self._min_num_patches = min_num_patches
        self._max_num_patches = max_num_patches if max_num_patches > 0 else float("inf")
        self._factor_max = factor_max
        self.norm_mean = torch.tensor(norm_mean).reshape(3, 1, 1)
        self.norm_std = torch.tensor(norm_std).reshape(3, 1, 1)
        # For pixel_shuffle with downsample_ratio=0.5, each 2x2 patch grid -> 1 token
        if downsample_ratio >= 1:
            raise ValueError(f"downsample_ratio must be < 1, got {downsample_ratio}.")
        reduction_factor = 1 / downsample_ratio
        if reduction_factor != 2.0:
            raise ValueError(
                "Only a reduction factor of 2.0 is supported (downsample_ratio=0.5), "
                f"got {reduction_factor} ({downsample_ratio=})."
            )
        self._reduction_factor = int(reduction_factor)

    @property
    def patch_pixel_size(self) -> int:
        """Pixel size of one patch."""
        return self._patch_size

    def _get_num_embeddings(self, width: int, height: int) -> int:
        """Post pixel-shuffle token count."""
        num_patches = width * height
        return num_patches // (self._reduction_factor**2)

    def max_num_tokens_available(self, text_prompt_length: int) -> int:
        # The -4 is to account for BOS, EOS, and image start / end tokens.
        # TODO: investigate whether this should take the number of images into account.
        return self._max_model_len - text_prompt_length - 4

    def process_media(
        self, media: Union[Image.Image, torch.Tensor], num_tokens_available: int
    ) -> Tuple[DynamicResolutionParams, int]:
        """Process a single media item and return its parameters.

        Args:
            media: The media item to process (PIL Image or CHW tensor).
            num_tokens_available: Number of tokens available for this media.

        Returns:
            DynamicResolutionParams for the media, and the token count.
        """
        orig_width, orig_height = _get_media_dimensions(media)
        closest_patch_height = round(orig_height / self._patch_size + 0.5)
        closest_patch_width = round(orig_width / self._patch_size + 0.5)
        patches = closest_patch_height * closest_patch_width

        factor = min(math.sqrt(num_tokens_available / patches), self._factor_max)
        target_patch_height = math.floor(factor * closest_patch_height)
        target_patch_width = math.floor(factor * closest_patch_width)

        # Enforce min_num_patches.
        if (
            num_tokens_available > self._min_num_patches
            and target_patch_height * target_patch_width < self._min_num_patches
        ):
            up_factor = math.sqrt(
                self._min_num_patches / (target_patch_height * target_patch_width)
            )
            target_patch_height = math.ceil(up_factor * target_patch_height)
            target_patch_width = math.ceil(up_factor * target_patch_width)

        # Round patch grid to be divisible by 2 for pixel shuffle.
        required_divisor = 2
        rem_h = target_patch_height % required_divisor
        if rem_h != 0:
            inc_h = required_divisor - rem_h
            if (target_patch_height + inc_h) * target_patch_width <= num_tokens_available:
                target_patch_height += inc_h
            else:
                target_patch_height = max(required_divisor, target_patch_height - rem_h)

        rem_w = target_patch_width % required_divisor
        if rem_w != 0:
            inc_w = required_divisor - rem_w
            if target_patch_height * (target_patch_width + inc_w) <= num_tokens_available:
                target_patch_width += inc_w
            else:
                target_patch_width = max(required_divisor, target_patch_width - rem_w)

        num_embeddings = self._get_num_embeddings(target_patch_width, target_patch_height)
        token_count = target_patch_width * target_patch_height

        return DynamicResolutionParams(
            media=media,
            num_tiles=1,
            num_embeddings=num_embeddings,
            patch_size=(target_patch_width, target_patch_height),
        ), token_count

    def compute_params(
        self, media_list: List[Union[Image.Image, torch.Tensor]], num_tokens_available: int
    ) -> List[DynamicResolutionParams]:
        """Compute parameters for all images with iterative token budgeting."""
        # Scale up by pixel shuffle factor (2^2 = 4)
        num_tokens_available = num_tokens_available * (self._reduction_factor**2)
        num_tokens_available = max(num_tokens_available, self._min_num_patches * len(media_list))

        num_tokens_per_media = [
            max(min(num_tokens_available, self._max_num_patches), self._min_num_patches)
        ] * len(media_list)

        # This loop keeps scaling down the number of tokens for each element in `num_tokens_per_media`
        # by the same amount until the sum of the token counts across all elements fits within the
        # `num_tokens_available` budget. The cap at 10 is to ensure the loop terminates, since the
        # `process_media` method applies rounding in such a way that could lead to the token count
        # (slightly) exceeding the prior iteration's downscaling.
        for _ in range(10):
            params = []
            token_counts = []

            for media, tokens_for_media in zip(media_list, num_tokens_per_media):
                param, token_count = self.process_media(media, tokens_for_media)
                params.append(param)
                token_counts.append(token_count)

            total_tokens = sum(token_counts)
            if total_tokens <= num_tokens_available:
                return params

            # Over budget - scale down proportionally.
            scaling_factor = num_tokens_available / total_tokens
            scaled = [max(self._min_num_patches, int(tc * scaling_factor)) for tc in token_counts]
            if any(s < o for s, o in zip(scaled, num_tokens_per_media)):
                num_tokens_per_media = scaled
            else:
                num_tokens_per_media = [self._min_num_patches] * len(media_list)

        raise ValueError("Token budget iteration failed to converge")

    @staticmethod
    def stack(images: List[torch.Tensor], patch_size: int) -> torch.Tensor:
        """Rearrange images into patches and concatenate."""
        imgs = [_rearrange_img(img, patch_size) for img in images]
        return torch.cat(imgs, dim=0).unsqueeze(0)


# Make this a runtime lookup rather than a module-wide constant for easier unit testing.
def _is_disagg() -> bool:
    return os.getenv("TLLM_MULTIMODAL_DISAGGREGATED", "0") == "1"


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2)


# Source codes are from NemotronH_Nano_VL_V2 modeling.py.
class NanoV2VLVisionEncoder(transformers.PreTrainedModel):
    def __init__(self, model_config: ModelConfig[transformers.PretrainedConfig]):
        config = model_config.pretrained_config
        super().__init__(config)
        self.image_size = config.force_image_size
        self.patch_size = config.patch_size
        self.num_image_token = int(
            (self.image_size // self.patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.spatial_merge_size = int(self.patch_size / self.downsample_ratio)
        # Pixel shuffle version.
        self.ps_version = config.ps_version
        if self.ps_version not in (supported_versions := {"v1", "v2"}):
            raise NotImplementedError(
                f"Unsupported {config.ps_version=}. Supported versions: {supported_versions}."
            )
        # Use config value if explicitly set (EVS enabled), otherwise default to 0.0 (EVS disabled)
        self.video_pruning_rate = (
            model_config.video_pruning_rate if model_config.video_pruning_rate is not None else 0.0
        )
        if hasattr(config, "norm_mean") and config.norm_mean is not None:
            self.register_buffer(
                "norm_mean", torch.tensor(config.norm_mean).reshape(3, 1, 1), persistent=False
            )
            self.register_buffer(
                "norm_std", torch.tensor(config.norm_std).reshape(3, 1, 1), persistent=False
            )
        else:
            self.norm_mean = None
            self.norm_std = None

        # Temporal compression config (for video).
        vision_config = config.vision_config if hasattr(config, "vision_config") else config
        self.video_temporal_patch_size = getattr(vision_config, "video_temporal_patch_size", 1)

        # Construct the vision projection.
        self.vit_hidden_size = config.vit_hidden_size
        self.vision_projection_hidden_size = config.projector_hidden_size
        self.llm_hidden_size = config.llm_config.hidden_size

        # Different versions of the configuration code may have a different name for the same value.
        eps = getattr(config.llm_config, "rms_norm_eps", None)
        if eps is None:
            eps = config.llm_config.layer_norm_epsilon

        self.mlp1 = nn.Sequential(
            nn.RMSNorm(
                self.vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                eps=eps,
                dtype=config.torch_dtype,
            ),
            nn.Linear(
                self.vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                self.vision_projection_hidden_size,
                bias=False,
                dtype=config.torch_dtype,
            ),
            SquaredReLU(),
            nn.Linear(
                self.vision_projection_hidden_size,
                self.llm_hidden_size,
                bias=False,
                dtype=config.torch_dtype,
            ),
        )

        # Construct the vision encoder.
        vision_model_config = copy.deepcopy(model_config)
        vision_model_config.pretrained_config = vision_model_config.pretrained_config.vision_config
        self.vision_model = RADIOVisionModel(vision_model_config, disable_quantization=True)

    def load_weights(self, weights):
        # Load mlp1 weights.
        mlp1_weights = {
            k.replace("mlp1.", ""): v for k, v in weights.items() if k.startswith("mlp1.")
        }
        self.mlp1.load_state_dict(mlp1_weights, strict=True)

        # Load vision encoder weights.
        vision_encoder_weights = {
            k.replace("vision_model.", ""): v
            for k, v in weights.items()
            if k.startswith("vision_model.")
        }
        self.vision_model.load_weights(vision_encoder_weights)

    @torch.compile
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
        )
        if self.ps_version == "v1":
            logger.warning(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values, num_frames=None):
        """Extract vision features, optionally with temporal compression.

        Args:
            pixel_values: [N, 3, H, W] tensor of frames/images.
            num_frames: When provided and video_temporal_patch_size > 1,
                enables temporal compression (tubelet embedding).
        """
        # When temporal compression is active, align micro-batch boundaries
        # to multiples of T so chunk splits don't break tubelets.
        T = self.video_temporal_patch_size if num_frames is not None else 1
        micro_batch_size = 128 - (128 % T) if T > 1 else 128

        n = pixel_values.shape[0]
        H_patches = pixel_values.shape[2] // self.patch_size
        W_patches = pixel_values.shape[3] // self.patch_size

        vit_embeds_lst = []
        for i in range(0, n, micro_batch_size):
            micro_batch_pixel_values = pixel_values[i : i + micro_batch_size]
            if num_frames is not None and T > 1:
                vit_embeds = self.vision_model(
                    micro_batch_pixel_values, num_frames=micro_batch_pixel_values.shape[0]
                )
            else:
                vit_embeds = self.vision_model(micro_batch_pixel_values)
            # Down-sampling and projection.
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], H_patches, W_patches, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            vit_embeds = self.mlp1(vit_embeds)
            vit_embeds_lst.append(vit_embeds)
        vit_embeds = torch.cat(vit_embeds_lst, dim=0)
        return vit_embeds

    def pixel_shuffle_dynamic_res(
        self, x: torch.Tensor, image_sizes: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Pixel shuffle for variable-size images in a concatenated sequence."""
        scale_factor = self.downsample_ratio
        patch_dim = self.patch_size
        seq_lens = calc_seq_lens(image_sizes, patch_dim)
        splits = torch.split(x, seq_lens, dim=1)
        out = []
        for i, sv in enumerate(splits):
            h = image_sizes[i][0] // patch_dim
            w = image_sizes[i][1] // patch_dim
            sv = sv.reshape(sv.shape[0], h, w, -1)

            n, h_dim, w_dim, c = sv.size()
            sv = sv.view(n, h_dim, int(w_dim * scale_factor), int(c / scale_factor))
            sv = sv.permute(0, 2, 1, 3).contiguous()
            sv = sv.view(
                n,
                int(w_dim * scale_factor),
                int(h_dim * scale_factor),
                int(c / (scale_factor * scale_factor)),
            )

            # NOTE: the input processor explicitly checks that dynamic resolution is always used
            # with `ps_version="v2"`..
            if self.ps_version != "v2":
                raise RuntimeError("Dynamic resolution requires pixel shuffling version 'v2'.")
            sv = sv.permute(0, 2, 1, 3).contiguous()

            sv = sv.reshape(sv.shape[0], -1, sv.shape[-1])
            out.append(sv)

        return torch.cat(out, dim=1)

    def _preprocess_raw_images(
        self,
        raw_pixel_values: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Resize, normalize, and rearrange raw images into patch format."""
        target_dtype = self.mlp1[1].weight.dtype
        processed = []
        for raw_tensor, (tgt_h, tgt_w) in zip(raw_pixel_values, image_sizes, strict=True):
            processed.append(
                _resize_and_normalize(raw_tensor, tgt_h, tgt_w, self.norm_mean, self.norm_std)
            )

        return DynamicResolutionImageTiler.stack(processed, self.patch_size).to(target_dtype)

    def extract_feature_dynamic(
        self, pixel_values_flat: torch.Tensor, image_sizes: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Dynamic resolution feature extraction for variable-size images."""
        vit_embeds = self.vision_model(pixel_values_flat, image_sizes=image_sizes)
        vit_embeds = self.pixel_shuffle_dynamic_res(vit_embeds, image_sizes=image_sizes)
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def apply_evs_per_video(
        self, mm_embed: torch.Tensor, video_sizes: List[Tuple]
    ) -> Tuple[torch.Tensor, List[int]]:
        """Apply EVS to the multimodal embedding for a single video.

        mm_embed may be 2D [total_tokens, hidden] (mixed aspect ratios) or
        3D [total_tiles, spatial_tokens, hidden] (uniform aspect ratios).
        """
        is_2d = mm_embed.dim() == 2
        hidden_size = mm_embed.shape[-1]
        T = self.video_temporal_patch_size
        start_idx, mm_embed_list, num_tokens_per_video = 0, [], []
        for video_size in video_sizes:
            # Fetch mm_embed correctly for the flattened temporal/patches dimension.
            t, p, ih, iw = video_size
            num_tubelets, wh = self._video_tubelet_geometry(t, T, ih, iw)

            if is_2d:
                total_tokens = num_tubelets * p * wh
                partial_mm_embed = mm_embed[start_idx : start_idx + total_tokens]
                partial_mm_embed = partial_mm_embed.reshape(num_tubelets * p, wh, hidden_size)
                start_idx += total_tokens
            else:
                partial_mm_embed = mm_embed[start_idx : start_idx + num_tubelets * p]
                # -> [num_tubelets * num_patches_per_frame, h*w, hidden_size]
                start_idx += num_tubelets * p

            # Need to expose temporal dimension for EVS.
            reshaped_partial_mm_embed = partial_mm_embed.reshape(
                num_tubelets, p, wh, hidden_size
            ).reshape(num_tubelets, p * wh, hidden_size)
            # -> [num_tubelets, num_patches_per_frame*h*w, hidden_size]

            original_retention_mask = compute_retention_mask(
                video_embeds=reshaped_partial_mm_embed,
                video_size=(num_tubelets, p * ih, iw),
                spatial_merge_size=self.spatial_merge_size,
                pruning_ratio=self.video_pruning_rate,
                flatten_output=False,
            ).flatten(start_dim=1)
            # -> [num_tubelets, num_patches_per_frame*h*w]
            num_tokens_per_frame = original_retention_mask.sum(dim=1)
            retention_mask = original_retention_mask.reshape(num_tubelets, p, wh).reshape(
                num_tubelets * p, wh
            )
            # -> [num_tubelets * num_patches_per_frame, h*w]

            partial_mm_embed = partial_mm_embed[retention_mask]
            mm_embed_list.append(partial_mm_embed)
            num_tokens_per_video.append(num_tokens_per_frame)
        mm_embed = torch.cat(mm_embed_list, dim=0)
        num_tokens_in_video = torch.cat(num_tokens_per_video, dim=0)
        return mm_embed, num_tokens_in_video

    def apply_evs(
        self, mm_embedding: List[torch.Tensor], multimodal_data_lst: List[Dict[str, Any]]
    ) -> Tuple[List[torch.Tensor], Optional[List[List[int] | None]]]:
        """Apply EVS to the multimodal embedding."""
        # Skip EVS if pruning ratio is 0.
        if self.video_pruning_rate <= 0:
            return mm_embedding, None

        modality_types = [
            multimodal_data["modality_type"] for multimodal_data in multimodal_data_lst
        ]
        # Skip EVS if there is no video modality.
        if "video" not in modality_types:
            return mm_embedding, None

        video_size_list = [
            multimodal_data[modality_type].get("video_size") if modality_type == "video" else None
            for modality_type, multimodal_data in zip(modality_types, multimodal_data_lst)
        ]
        mm_embedding_evs = []
        num_tokens_in_videos = []
        # Iterate over batch.
        for modality_type, mm_embed, video_sizes in zip(
            modality_types, mm_embedding, video_size_list
        ):
            # Skip EVS if modality type is not video.
            if modality_type != "video":
                mm_embedding_evs.append(mm_embed)
                num_tokens_in_videos.append(None)
            else:
                # Iterate over each video in the batch.
                mm_embed, num_tokens_per_frames = self.apply_evs_per_video(mm_embed, video_sizes)
                mm_embedding_evs.append(mm_embed)
                num_tokens_in_videos.append(num_tokens_per_frames)
        return mm_embedding_evs, num_tokens_in_videos

    def _extract_video_embeddings_temporal(self, video_data: Dict[str, Any]) -> torch.Tensor:
        """Extract video embeddings with temporal compression.

        Each video is processed separately through extract_feature with num_frames, which enables
        the tubelet embedding path in RADIO.

        `pixel_values` may be a single concatenated tensor (all videos share the same frame size)
        or a list of per-video tensors (mixed aspect ratios).
        """
        pixel_values = video_data["pixel_values"]
        video_size_list = video_data.get("video_size", [])
        per_video = isinstance(pixel_values, list)

        all_embeds = []
        frame_offset = 0
        for idx, video_size in enumerate(video_size_list):
            num_frames = video_size[0]
            num_tiles_per_frame = video_size[1]
            total_tiles = num_frames * num_tiles_per_frame
            if per_video:
                video_frames = pixel_values[idx]
            else:
                video_frames = pixel_values[frame_offset : frame_offset + total_tiles]
                frame_offset += total_tiles

            vit_embeds = self.extract_feature(video_frames, num_frames=num_frames)
            # Flatten to 2D [tokens, hidden] so videos with different spatial
            # resolutions (different dim-1) can be concatenated.
            all_embeds.append(vit_embeds.reshape(-1, vit_embeds.shape[-1]))

        # Concatenate all videos' embeddings (2D).
        return torch.cat(all_embeds, dim=0)

    def forward(
        self, multimodal_params: List[MultimodalParams]
    ) -> Tuple[List[torch.Tensor], List[List[int] | None]]:
        mm_embedding = []
        multimodal_data_lst = [
            multimodal_param.multimodal_data for multimodal_param in multimodal_params
        ]
        modality_types = [
            multimodal_data["modality_type"] for multimodal_data in multimodal_data_lst
        ]

        for modality_type, multimodal_data in zip(modality_types, multimodal_data_lst):
            data = multimodal_data[modality_type]
            # Dynamic resolution path is indicated by the presence of "image_sizes". For now, it is
            # only meant to be applied to images.
            if modality_type == "image" and "image_sizes" in data:
                image_sizes = data["image_sizes"]
                if self.norm_mean is not None:
                    pixel_values_flat = self._preprocess_raw_images(
                        data["pixel_values"],
                        image_sizes,
                    )
                else:
                    pixel_values_flat = data["pixel_values"]
                embeds = self.extract_feature_dynamic(pixel_values_flat, image_sizes)
                # Keep 3D shape for apply_evs, will reshape to 2D after EVS
                mm_embedding.append(embeds)
            elif modality_type == "video" and (
                self.video_temporal_patch_size > 1 or isinstance(data["pixel_values"], list)
            ):
                # Process each video separately when temporal compression is
                # enabled or when videos cannot be concatenated due to mixed
                # resized shapes.
                embeds = self._extract_video_embeddings_temporal(data)
                mm_embedding.append(embeds)
            # This applies to images without dynamic resolution, or videos with T=1.
            else:
                # Fallback to fixed-tile extraction for this modality.
                pixel_values = data["pixel_values"]
                embeds = self.extract_feature(pixel_values)
                # Keep 3D shape [num_patches, h*w, hidden] for apply_evs
                mm_embedding.append(embeds)

        # Apply EVS if video_pruning_rate > 0
        mm_embedding, num_tokens_in_videos = self.apply_evs(mm_embedding, multimodal_data_lst)
        # Reshape to 2D after EVS: [num_patches*h*w, hidden_size]
        mm_embedding = [m.reshape(-1, self.llm_hidden_size) for m in mm_embedding]
        return mm_embedding, num_tokens_in_videos

        # Existing fixed-tile path (unreachable, kept for reference).
        pixel_values = [
            multimodal_data[modality_type]["pixel_values"]
            for modality_type, multimodal_data in zip(modality_types, multimodal_data_lst)
        ]
        batched_pixel_values = torch.cat(pixel_values, dim=0)
        # -> [num_patches, channel, height, width]
        patch_list = [
            multimodal_data[modality_type]["num_patches"]
            for modality_type, multimodal_data in zip(modality_types, multimodal_data_lst)
        ]
        batched_num_patches = torch.cat(patch_list, dim=0).tolist()
        # -> list of[num_patches1, num_patches2, ...]
        batched_image_embeds = self.extract_feature(batched_pixel_values)
        # -> [num_patches, num_image_token, hidden_size]
        mm_embedding = torch.split(batched_image_embeds, batched_num_patches, dim=0)

        mm_embedding, num_tokens_in_videos = self.apply_evs(mm_embedding, multimodal_data_lst)

        mm_embedding = [m.reshape(-1, self.llm_hidden_size) for m in mm_embedding]
        # -> list of [num_patches*num_image_token, hidden_size]
        return mm_embedding, num_tokens_in_videos

    def _video_tubelet_geometry(self, t: int, T: int, ih: int, iw: int) -> Tuple[int, int]:
        """Return `(num_tubelets, wh)` for one video.

        num_tubelets: Temporal token count after tubelet compression (`T` consecutive frames -> 1
            tubelet).
        wh: spatial token count per tile.
        """
        num_tubelets = math.ceil(t / T) if T > 1 else t
        wh = int(ih // self.patch_size * self.downsample_ratio) * int(
            iw // self.patch_size * self.downsample_ratio
        )
        return num_tubelets, wh


class NanoV2VLInputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    def __init__(
        self,
        model_path: str,
        config: transformers.PretrainedConfig,
        tokenizer: transformers.AutoTokenizer,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        # Extract video_pruning_rate before passing kwargs to parent
        video_pruning_rate = kwargs.pop("video_pruning_rate", None) or 0.0

        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if not trust_remote_code:
            raise ValueError("trust_remote_code must be True for Phi4MM")

        self._config = config
        self._tokenizer = (
            tokenizer
            if tokenizer is not None
            else transformers.AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, use_fast=self.use_fast
            )
        )
        self._processor = transformers.AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=self.use_fast
        )
        self._model_path = model_path
        self._dtype = self.config.torch_dtype
        self.device = "cpu"

        self.image_size = self.config.force_image_size
        self.patch_size = self.config.patch_size
        self.downsample_ratio = self.config.downsample_ratio
        self.spatial_merge_size = int(self.patch_size / self.downsample_ratio)
        self.img_context_token_id = self.config.img_context_token_id
        self.num_image_token = int(
            (self.image_size // self.patch_size) ** 2 * (self.downsample_ratio**2)
        )
        self.video_pruning_rate = video_pruning_rate
        self.img_context_token = self.config.img_context_token
        self.video_context_token = self.config.video_context_token
        self.video_context_token_id = self.config.video_context_token_id
        self.img_start_token = self.config.img_start_token
        self.img_end_token = self.config.img_end_token
        # Pre-tokenize special tokens for video EVS processing (following vLLM).
        # These may be multi-token under BPE, so we store the full ID list.
        self._img_start_token_ids = self.tokenizer.encode(
            self.img_start_token, add_special_tokens=False
        )
        self._img_end_token_ids = self.tokenizer.encode(
            self.img_end_token, add_special_tokens=False
        )
        self._img_context_token_ids = self.tokenizer.encode(
            self.img_context_token, add_special_tokens=False
        )
        # Pre-tokenize the user-facing "<video>" placeholder string. NOTE this may
        # be a multi-token BPE sequence (e.g. [1060, 24073, 1062]) rather than
        # `video_context_token_id`, because `<video>` is typically NOT registered
        # as an added special token in the HF tokenizer (unlike `<image>` /
        # `<so_embedding>`). `video_context_token_id` is instead a reserved
        # extended-vocab ID that the model uses internally as the per-tubelet EVS
        # placeholder; the tokenizer cannot produce it from user text. The fast
        # path (token IDs & MM data) searches for this subsequence in the
        # caller's `prompt_token_ids`.
        self._video_placeholder_token_ids = self.tokenizer.encode(
            self.video_context_token, add_special_tokens=False
        )
        # Keep single-ID aliases for backward compat.
        self.image_start_token_id = self._img_start_token_ids[0]
        self.image_end_token_id = self._img_end_token_ids[0]

        # Detect dynamic resolution from config.
        self.dynamic_tiler = None
        vision_args = getattr(getattr(config, "vision_config", None), "args", None)
        if isinstance(vision_args, dict) and "min_num_patches" in vision_args:
            pixel_shuffle_version = config.ps_version
            if pixel_shuffle_version != "v2":
                raise NotImplementedError(
                    "Dynamic resolution (enabled via `vision_config.min_num_patches`) only supports "
                    f"`config.ps_version='v2'. Got {pixel_shuffle_version=}."
                )
            self.dynamic_tiler = DynamicResolutionImageTiler(
                max_model_len=config.max_sequence_length,
                patch_size=self.patch_size,
                downsample_ratio=self.downsample_ratio,
                min_num_patches=vision_args["min_num_patches"],
                max_num_patches=vision_args["max_num_patches"],
                norm_mean=config.norm_mean,
                norm_std=config.norm_std,
            )
            logger.info("Dynamic resolution enabled for NanoV2VL input processor")

        # Video temporal compression and sizing config.
        vision_config = getattr(config, "vision_config", config)
        self.video_temporal_patch_size = getattr(vision_config, "video_temporal_patch_size", 1)
        self.video_maintain_aspect_ratio = getattr(
            vision_config, "video_maintain_aspect_ratio", False
        )
        # Resolve video target size: video_target_num_patches or video_target_img_size.
        target_num_patches = getattr(vision_config, "video_target_num_patches", None)
        target_img_size = getattr(vision_config, "video_target_img_size", None)
        if target_num_patches is not None and target_img_size is not None:
            raise ValueError(
                "Exactly one of video_target_num_patches or "
                "video_target_img_size must be set, got both"
            )
        if target_num_patches is not None:
            self.video_target_num_patches = target_num_patches
        elif target_img_size is not None:
            base_patches = math.ceil(target_img_size / self.patch_size)
            self.video_target_num_patches = base_patches * base_patches
        else:
            self.video_target_num_patches = None

        # The original model used a "This is a video:\n" prefix before frame separators.
        # Newer models may not. Unfortunately, both map to the same "NemotronH_Nano_VL_V2"
        # transformers architecture, so the only way to distinguish is via the presence / absence
        # of certain fields.
        self._add_video_prefix = self.video_target_num_patches is None

        # Normalization mean/std for video frames (used by video_to_pixel_values).
        self._video_norm_mean = torch.tensor(config.norm_mean).reshape(3, 1, 1)
        self._video_norm_std = torch.tensor(config.norm_std).reshape(3, 1, 1)

        self._audio_extractor = None
        self._sound_context_token = getattr(config, "sound_context_token", AUDIO_PLACEHOLDER)
        # At time of writing (03/09/2026), these were not included in the config.
        self._sound_start = "<so_start>"
        self._sound_end = "<so_end>"
        self._sound_context_token_id = getattr(config, "sound_context_token_id", None)
        self._sound_start_token_id = None
        self._sound_end_token_id = None
        sound_config = getattr(config, "sound_config", None)
        if sound_config is not None:
            try:
                import librosa  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Audio support requires the `librosa` package. "
                    "Install it with: pip install librosa"
                )
            self._audio_extractor = ParakeetExtractor(sound_config)
            if self._sound_context_token_id is None:
                self._sound_context_token_id = self.tokenizer.encode(
                    self._sound_context_token, add_special_tokens=False
                )[0]
            self._sound_start_token_id = self.tokenizer.encode(
                self._sound_start, add_special_tokens=False
            )[0]
            self._sound_end_token_id = self.tokenizer.encode(
                self._sound_end, add_special_tokens=False
            )[0]

    @property
    def config(self) -> transformers.PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> transformers.AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self) -> transformers.AutoProcessor:
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_vocab_size(self):
        return self.config.llm_config.vocab_size

    def get_mm_special_token_ids(self) -> torch.Tensor:
        "Return multimodal special token ids for NanoV2VL."
        ids = list(self._img_start_token_ids) + list(self._img_end_token_ids)
        if self._sound_start_token_id is not None:
            ids.extend([self._sound_start_token_id, self._sound_end_token_id])
        return torch.tensor(ids)

    def get_mm_token_ids(self):
        ids = [self.img_context_token_id]
        if self._sound_context_token_id is not None:
            ids.append(self._sound_context_token_id)
        return torch.tensor(ids, dtype=torch.int32)

    def get_text_with_mm_placeholders(self, mm_counts: Dict[str, int]) -> str:
        """Return minimal placeholder text for the given multimodal item counts,
        so that the HF processor can be called with (dummy_text, mm_data)
        without error. Used when processing tokenized prompt + MM data.

        Args:
            mm_counts (Dict[str, int]): A mapping of each multimodal modality
                name (`'image'`, `'video'`, `'audio'`) to the count of items
                for that modality that need corresponding placeholders in the
                dummy text.

        Returns:
            str: A minimal placeholder string containing the correct number
                and type of multimodal placeholders, suitable for passing
                along with mm_data to the Hugging Face processor.
        """
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)
        parts: List[str] = []
        parts.extend([self.img_context_token] * num_images)
        parts.extend([self.video_context_token] * num_videos)
        parts.extend([self._sound_context_token] * num_audios)
        return "".join(parts)

    def get_num_tokens_per_image(
        self,
        *,
        image: Union[Image.Image, torch.Tensor],
        **kwargs,
    ):
        # Dynamic resolution path — only when max_num_tiles is not
        # explicitly overridden (e.g. for video frames which use
        # VIDEO_MAX_NUM_TILES and rely on the InternVL tiling logic
        # in the HF processor, not the dynamic tiler).
        if self.dynamic_tiler is not None and "max_num_tiles" not in kwargs:
            budget = self.dynamic_tiler._max_num_patches
            params, _ = self.dynamic_tiler.process_media(image, budget)
            num_image_tokens = params.num_embeddings
            num_image_tokens += len(self._img_start_token_ids) + len(self._img_end_token_ids)
            return num_image_tokens

        # The logic is copied and modified from HuggingFace ImageProcessor.

        def _get_internvl_target_ratios(
            min_num: int,
            max_num: int,
        ) -> list[tuple[int, int]]:
            target_ratios = {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if min_num <= i * j <= max_num
            }
            return sorted(target_ratios, key=lambda x: x[0] * x[1])

        def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float("inf")
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    image_area = image_size * image_size
                    ratio_prod = ratio[0] * ratio[1]
                    if area > 0.5 * image_area * ratio_prod:
                        best_ratio = ratio
            return best_ratio

        def _calculate_targets(
            orig_width: int,
            orig_height: int,
            target_ratios: list[tuple[int, int]],
            image_size: int,
        ) -> int:
            aspect_ratio = orig_width / orig_height

            # find the closest aspect ratio to the target
            target_aspect_ratio = _find_closest_aspect_ratio(
                aspect_ratio,
                target_ratios,
                width=orig_width,
                height=orig_height,
                image_size=image_size,
            )
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            return blocks

        image_width, image_height = _get_media_dimensions(image)
        if "max_num_tiles" in kwargs:
            max_num_tiles = kwargs["max_num_tiles"]
        else:
            max_num_tiles = self.processor.max_num_tiles
        target_ratios = _get_internvl_target_ratios(1, max_num_tiles)
        blocks = _calculate_targets(image_width, image_height, target_ratios, self.image_size)
        if self.processor.use_thumbnail and blocks != 1:
            blocks += 1
        num_image_tokens = self.num_image_token * blocks
        num_image_tokens += len(self._img_start_token_ids) + len(self._img_end_token_ids)
        return num_image_tokens

    def _get_video_tokens_per_frame(
        self, orig_w: Optional[int] = None, orig_h: Optional[int] = None
    ) -> int:
        """Token count per video frame (or tubelet).

        This accounts for `video_target_num_patches`, which may change the frame size.

        When `video_maintain_aspect_ratio` is enabled, the result depends on the source frame
        dimensions, so callers should pass the real `orig_w` and `orig_h`.

        Falls back to `self.image_size` (square) when not provided - correct only for square inputs.
        """
        if self.video_target_num_patches is not None:
            if orig_w is None:
                orig_w = self.image_size
            if orig_h is None:
                orig_h = self.image_size
            _, _, feature_size = get_video_target_size_and_feature_size(
                orig_w=orig_w,
                orig_h=orig_h,
                target_patches=self.video_target_num_patches,
                maintain_aspect_ratio=self.video_maintain_aspect_ratio,
                patch_size=self.patch_size,
                downsample_ratio=self.downsample_ratio,
            )
            return feature_size
        return self.num_image_token

    # TODO(TRTLLM-12465): Accept a VideoData container here instead of passing
    # frames, metadata, and audio as separate arguments.
    def get_num_tokens_per_video(
        self,
        *,
        video: List[Union[Image.Image, torch.Tensor]],
        video_pruning_rate: Optional[float] = None,
        video_metadata: Optional[dict] = None,
        video_audio: Optional[AudioData] = None,
        **kwargs,
    ):
        # Use VIDEO_PRUNING_RATIO if not explicitly provided
        if video_pruning_rate is None:
            video_pruning_rate = self.video_pruning_rate

        T = self.video_temporal_patch_size
        num_frames = len(video)
        num_tubelets = math.ceil(num_frames / T) if T > 1 else num_frames
        # Use actual frame dimensions so aspect-ratio-preserving resize is computed correctly
        # (instead of assuming square frames).
        frame = video[0]
        frame_w, frame_h = _get_media_dimensions(frame)
        tokens_per_unit = self._get_video_tokens_per_frame(orig_w=frame_w, orig_h=frame_h)

        num_special_tokens_per_frame = 2  # <img> and </img>
        if video_pruning_rate > 0:
            # `tokens_per_unit` already reflects the actual per-tubelet
            # post-pixel-shuffle spatial grid the vision encoder sees — it
            # comes from `_get_video_tokens_per_frame`, which is
            # video_target_num_patches / aspect-preserving / image_size
            # fallback aware. That's the same grid
            # `_process_videos_frames` feeds to the encoder, so retention
            # computed from it matches the encoder's actual EVS output.
            # `compute_retained_tokens_from_tubelet_budget` is the shared
            # helper `compute_retained_tokens_count` delegates to, so the
            # two paths stay in sync by construction.
            evs_tokens = compute_retained_tokens_from_tubelet_budget(
                num_tubelets=num_tubelets,
                tokens_per_tubelet=tokens_per_unit,
                pruning_ratio=video_pruning_rate,
            )
            num_total_tokens = evs_tokens + num_tubelets * num_special_tokens_per_frame
        else:
            # No pruning: tokens_per_unit * num_tubelets + special tokens.
            num_total_tokens = num_tubelets * (tokens_per_unit + num_special_tokens_per_frame)

        # If audio was extracted from this video, the prompt carries
        # <so_start><so_embedding>*M<so_end> appended after the video frames
        # (see _expand_video_placeholders_in_token_ids and the slow-path
        # _extract_audio_from_video). Account for those tokens so
        # total_mm_tokens_in_request matches the actual mm-token count in the
        # tokenized prompt and len(mm_embed) at forward time.
        if video_audio is not None and self._audio_extractor is not None:
            num_total_tokens += self.get_num_tokens_per_audio(
                audio=(video_audio.samples, video_audio.sample_rate)
            )

        return num_total_tokens

    # ------------------------------------------------------------------
    # Tokenized+MM fast path: expand placeholder tokens in token IDs
    # ------------------------------------------------------------------

    def expand_prompt_token_ids_for_mm(
        self,
        prompt_token_ids: List[int],
        num_mm_tokens_per_placeholder: List[int],
        hf_processor_mm_kwargs: Optional[Dict[str, Any]] = None,
        mm_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[int], Optional[Dict[str, Dict[str, Any]]]]:
        """Expand MM placeholder tokens in `prompt_token_ids` so that each
        single placeholder is replaced by the corresponding number of
        multimodal feature tokens.

        This is used when processing a tokenized prompt plus multimodal data,
        without calling the full HuggingFace processor. Detects which modality
        is present by scanning for placeholder token IDs and dispatches to the
        matching per-modality expansion helper. NanoV2VL allows only one
        modality per request, so this raises `ValueError` if placeholders for
        more than one modality are found.

        Args:
            prompt_token_ids (List[int]): The input prompt token IDs with
                image / video / audio placeholder tokens.
            num_mm_tokens_per_placeholder (List[int]): For each MM placeholder
                in `prompt_token_ids`, specifies the total number of MM tokens
                (including modality-specific start/end special tokens) that
                the placeholder should expand to. Produced by
                `find_mm_token_lengths`.
            hf_processor_mm_kwargs (Optional[Dict[str, Any]]): Optional
                dictionary of HF processor kwargs. Not currently consulted by
                NanoV2VL's expansion (kept for interface compatibility with
                other implementations of `expand_prompt_token_ids_for_mm`).
            mm_data (Optional[Dict[str, Any]]): The original
                `multi_modal_data` dict (`{"image": [...], "video": [...],
                "audio": [...]}`). Required by the video path for frame
                separator text reconstruction from per-video metadata; the
                image and audio paths don't use it.

        Returns:
            Tuple[List[int], Optional[Dict[str, Dict[str, Any]]]]:
                `expanded_ids` (List[int]) — prompt token IDs where each MM
                placeholder has been replaced/expanded with the appropriate
                number of MM feature tokens (plus frame separator tokens for
                video).
                `mm_data_updates` (Optional[Dict]) — additional fields to
                merge into `extra_processed_inputs["multimodal_data"]`.
                Currently only non-None for EVS-enabled video, in which case
                it is `{"video": {"evs_ids": evs_ids_tensor}}`.
        """
        # Detect modality primarily from `mm_data`, which is authoritative and
        # independent of tokenizer quirks. The token-scanning fallback is only
        # used when `mm_data` is absent (shouldn't happen in the fast path, but
        # kept as a safety net). Token scanning is fragile for video because
        # `<video>` is typically not registered as an added special token and
        # thus BPE-decomposes differently depending on trailing context — see
        # the two-tier matching strategy in `_expand_video_placeholders_in_token_ids`.
        mm_data = mm_data or {}
        if mm_data:
            has_image = "image" in mm_data and bool(mm_data["image"])
            has_video = "video" in mm_data and bool(mm_data["video"])
            has_audio = "audio" in mm_data and bool(mm_data["audio"])
        else:
            has_image = self.img_context_token_id in prompt_token_ids
            has_video = self._contains_video_placeholder(prompt_token_ids)
            has_audio = (
                self._sound_context_token_id is not None
                and self._sound_context_token_id in prompt_token_ids
            )

        active_count = sum([has_image, has_video, has_audio])
        if active_count > 1:
            raise ValueError(
                "NanoV2VL does not support multiple modalities in the same prompt yet."
            )
        if active_count == 0:
            return prompt_token_ids, None

        if has_image:
            expanded = self._expand_image_placeholders_in_token_ids(
                prompt_token_ids, num_mm_tokens_per_placeholder
            )
            return expanded, None
        if has_audio:
            expanded = self._expand_audio_placeholders_in_token_ids(
                prompt_token_ids, num_mm_tokens_per_placeholder
            )
            return expanded, None
        # has_video -- reuse `mm_data` already extracted above for detection.
        expanded_ids, evs_ids_tensor = self._expand_video_placeholders_in_token_ids(
            prompt_token_ids, num_mm_tokens_per_placeholder, mm_data or None
        )
        mm_data_updates: Optional[Dict[str, Dict[str, Any]]] = None
        if evs_ids_tensor is not None:
            mm_data_updates = {"video": {"evs_ids": evs_ids_tensor}}
        return expanded_ids, mm_data_updates

    def _contains_video_placeholder(self, prompt_token_ids: List[int]) -> bool:
        """Return True if `_video_placeholder_token_ids` appears as a
        contiguous subsequence of `prompt_token_ids`."""
        pattern = self._video_placeholder_token_ids
        if not pattern:
            return False
        pattern_len = len(pattern)
        for i in range(len(prompt_token_ids) - pattern_len + 1):
            if prompt_token_ids[i : i + pattern_len] == pattern:
                return True
        return False

    def _expand_single_token_placeholders(
        self,
        prompt_token_ids: List[int],
        num_mm_tokens_per_placeholder: List[int],
        *,
        context_token_id: int,
        start_token_ids: List[int],
        end_token_ids: List[int],
        modality_name: str,
    ) -> List[int]:
        """Shared per-placeholder expansion for modalities whose in-prompt
        marker is a single token ID (i.e. the tokenizer maps the placeholder
        string to exactly one token).

        Replaces each occurrence of `context_token_id` in `prompt_token_ids`
        with `start_token_ids + [context_token_id] * (N - num_start - num_end) + end_token_ids`,
        where `N = num_mm_tokens_per_placeholder[i]` is the total expanded
        length (including start / end special tokens) reported by the
        corresponding `get_num_tokens_per_<modality>`.

        Args:
            prompt_token_ids: Input prompt token IDs with single-token
                placeholders of `context_token_id`.
            num_mm_tokens_per_placeholder: One entry per placeholder;
                total expanded length (including start / end).
            context_token_id: Single token ID that marks each placeholder
                in the prompt (e.g. `img_context_token_id == 18`).
            start_token_ids: Tokens to emit before the repeated
                `context_token_id` run (e.g. `_img_start_token_ids == [19]`).
            end_token_ids: Tokens to emit after the repeated run
                (e.g. `_img_end_token_ids == [20]`).
            modality_name: Human-readable name (`"image"` / `"audio"`) used
                in error messages.

        Returns:
            The prompt token IDs with each placeholder replaced by
            `start + context*K + end`, where `K = N - len(start) - len(end)`.
        """
        num_start = len(start_token_ids)
        num_end = len(end_token_ids)

        expanded: List[int] = []
        idx = 0
        for tok in prompt_token_ids:
            if tok == context_token_id:
                if idx >= len(num_mm_tokens_per_placeholder):
                    raise ValueError(
                        f"More {modality_name} placeholder tokens in prompt "
                        f"than num_mm_tokens_per_placeholder entries: found "
                        f"{idx + 1} placeholders, "
                        f"num_mm_tokens_per_placeholder has "
                        f"{len(num_mm_tokens_per_placeholder)} entries."
                    )
                n = num_mm_tokens_per_placeholder[idx]
                num_context = n - num_start - num_end
                expanded.extend(start_token_ids)
                expanded.extend([context_token_id] * num_context)
                expanded.extend(end_token_ids)
                idx += 1
            else:
                expanded.append(tok)

        if idx != len(num_mm_tokens_per_placeholder):
            raise ValueError(
                f"Expected {len(num_mm_tokens_per_placeholder)} "
                f"{modality_name} placeholders, found {idx}."
            )
        return expanded

    def _expand_image_placeholders_in_token_ids(
        self,
        prompt_token_ids: List[int],
        num_mm_tokens_per_placeholder: List[int],
    ) -> List[int]:
        """Image expansion — thin wrapper around
        `_expand_single_token_placeholders` with image-specific tokens."""
        return self._expand_single_token_placeholders(
            prompt_token_ids,
            num_mm_tokens_per_placeholder,
            context_token_id=self.img_context_token_id,
            start_token_ids=self._img_start_token_ids,
            end_token_ids=self._img_end_token_ids,
            modality_name="image",
        )

    def _expand_audio_placeholders_in_token_ids(
        self,
        prompt_token_ids: List[int],
        num_mm_tokens_per_placeholder: List[int],
    ) -> List[int]:
        """Audio expansion — thin wrapper around
        `_expand_single_token_placeholders` with audio-specific tokens."""
        return self._expand_single_token_placeholders(
            prompt_token_ids,
            num_mm_tokens_per_placeholder,
            context_token_id=self._sound_context_token_id,
            start_token_ids=[self._sound_start_token_id],
            end_token_ids=[self._sound_end_token_id],
            modality_name="audio",
        )

    def _compute_video_shape_descriptor(
        self,
        video_frames: List[Image.Image],
    ) -> List[int]:
        """Compute the `video_size` descriptor `[num_frames, num_tiles, h, w]`
        from frame dimensions only, without running the pixel-processing
        pipeline.

        This mirrors the shape returned by `_process_videos_frames` but skips
        the expensive tensor construction, so the token-ID expansion path can
        cheaply derive per-frame token counts and frame separator strings.

        With `video_target_num_patches` set, target dimensions are computed via
        `get_video_target_size_and_feature_size`. Otherwise (HF-processor
        fallback), frames are assumed to be resized to `self.image_size` with
        a single tile per frame (`VIDEO_MAX_NUM_TILES == 1`).

        Args:
            video_frames (List[Image.Image]): The list of frames for a single
                video. Only the first frame's dimensions are inspected when
                `video_target_num_patches` is set.

        Returns:
            List[int]: `[num_frames, num_tiles_per_frame, h, w]`, where
                `num_tiles_per_frame` is always 1 for NanoV2VL video paths.
        """
        num_frames = len(video_frames)
        if self.video_target_num_patches is not None:
            frame = video_frames[0]
            if isinstance(frame, Image.Image):
                orig_w, orig_h = frame.width, frame.height
            else:
                orig_h, orig_w = frame.shape[-2], frame.shape[-1]
            target_w, target_h, _ = get_video_target_size_and_feature_size(
                orig_w=orig_w,
                orig_h=orig_h,
                target_patches=self.video_target_num_patches,
                maintain_aspect_ratio=self.video_maintain_aspect_ratio,
                patch_size=self.patch_size,
                downsample_ratio=self.downsample_ratio,
            )
            return [num_frames, 1, target_h, target_w]
        return [num_frames, 1, self.image_size, self.image_size]

    def _expand_video_placeholders_in_token_ids(
        self,
        prompt_token_ids: List[int],
        num_mm_tokens_per_placeholder: List[int],
        mm_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[int], Optional[torch.Tensor]]:
        """Replace each `<video>` placeholder in `prompt_token_ids` with the
        full per-frame expansion (optional "This is a video:\\n" prefix,
        then for each frame/tubelet: frame separator text tokens, `<img>`,
        `<image>` repeated per-frame-N times, `</img>`).

        The `<video>` placeholder is located by a **two-tier** strategy that
        mirrors vLLM's `_apply_prompt_updates`:

          1. Token-level subsequence match against
             `self._video_placeholder_token_ids` (the result of tokenizing
             `"<video>"` in isolation at init time).

          2. If token-level matching fails to find all `len(videos)`
             occurrences (typically because BPE merged the trailing `>` of
             `<video>` with the following character into a different token
             ID), decode the prompt back to text, split on the literal
             `"<video>"` string, and re-encode each segment.

        When EVS is enabled (`self.video_pruning_rate > 0`), a **parallel
        `evs_ids` stream** is also produced — same surrounding context (text
        segments, frame separators, `<img>`/`</img>` wrappers), but with a
        single `video_context_token_id` per tubelet instead of
        `img_context_token_id*num_tokens`. This matches the evs_ids built by
        `_process_video_prompts` in the non-fast-path and is consumed by
        `merge_evs_mm_embeds` at LLM forward time.

        Args:
            prompt_token_ids (List[int]): The input prompt token IDs with
                `<video>` placeholders.
            num_mm_tokens_per_placeholder (List[int]): For each video
                placeholder, the total number of MM tokens produced by
                `get_num_tokens_per_video`. Currently used only for sanity /
                iteration; per-frame counts are recomputed from `mm_data`.
            mm_data (Optional[Dict[str, Any]]): The original
                `multi_modal_data` dict, threaded through by the fast-path
                pipeline as a dedicated argument on
                `expand_prompt_token_ids_for_mm`. Must contain a `"video"`
                entry with `VideoData`-like items (`.frames`, `.metadata`).
                Required; cannot be reconstructed from `prompt_token_ids`
                alone because frame separators depend on per-video metadata.

        Returns:
            Tuple[List[int], Optional[torch.Tensor]]:
                `expanded_ids` (List[int]) — prompt token IDs with each
                `<video>` placeholder replaced by its full per-frame token
                sequence.
                `evs_ids` (Optional[torch.Tensor]) — parallel stream with one
                `video_context_token_id` placeholder per tubelet when EVS is
                enabled; `None` otherwise.

        Raises:
            ValueError: If `mm_data` is `None`, or if the number of
                `<video>` placeholders does not match the number of videos
                in `mm_data`.
        """
        if mm_data is None:
            raise ValueError(
                "Video expansion requires multi_modal_data (passed as the "
                "`mm_data` argument of expand_prompt_token_ids_for_mm)."
            )

        videos = mm_data.get("video", [])
        if not isinstance(videos, list):
            videos = [videos]

        evs_enabled = self.video_pruning_rate > 0

        # Pre-compute per-video expansion token sequences for BOTH streams.
        # Surrounding context (prefix, frame separators, <img>/</img>) is
        # byte-for-byte identical; only the per-tubelet MM content differs.
        #
        # Per-tubelet img_context counts come from
        # `_compute_token_numbers_per_video`, which matches what the
        # vision encoder actually produces post-EVS (single-tile
        # per-tubelet spatial grid). `get_num_tokens_per_video` — used by
        # `find_mm_token_lengths` — must agree with this.
        video_expansions: List[List[int]] = []
        video_evs_expansions: Optional[List[List[int]]] = [] if evs_enabled else None
        for video_data in videos:
            # `video_data` comes in two shapes:
            #   - A `VideoData`-like object (with `.frames` / `.metadata` /
            #     `.audio` attributes) — the production case (openai_server
            #     wraps video inputs this way).
            #   - A plain list of frames (no metadata) — the defensive
            #     fallback when callers pass raw frames directly, matching
            #     the pattern in `find_mm_token_lengths` in multimodal.py.
            # `getattr` with a default handles both uniformly.
            frames = getattr(video_data, "frames", video_data)
            metadata = getattr(video_data, "metadata", None)
            audio = getattr(video_data, "audio", None)

            video_size = self._compute_video_shape_descriptor(frames)
            # When EVS is enabled, `_compute_token_numbers_per_video`
            # returns the dummy pattern [K_pre_evs, 0, 0, ...] — same shape
            # the str-replacement-path uses to size `input_ids` pre-EVS-merge;
            # `merge_evs_mm_embeds` rewrites it at forward time.
            tokens_per_frame = self._compute_token_numbers_per_video([video_size])[0]
            frame_seps = self._get_frame_separators([video_size], [metadata])[0]

            expansion: List[int] = []
            evs_expansion: Optional[List[int]] = [] if evs_enabled else None
            if self._add_video_prefix:
                prefix_ids = self.tokenizer.encode("This is a video:\n", add_special_tokens=False)
                expansion.extend(prefix_ids)
                if evs_expansion is not None:
                    evs_expansion.extend(prefix_ids)
            for frame_sep, num_tokens in zip(frame_seps, tokens_per_frame, strict=True):
                sep_ids = self.tokenizer.encode(frame_sep, add_special_tokens=False)
                expansion.extend(sep_ids)
                expansion.extend(self._img_start_token_ids)
                expansion.extend([self.img_context_token_id] * num_tokens)
                expansion.extend(self._img_end_token_ids)
                if evs_expansion is not None:
                    evs_expansion.extend(sep_ids)
                    evs_expansion.extend(self._img_start_token_ids)
                    evs_expansion.append(self.video_context_token_id)
                    evs_expansion.extend(self._img_end_token_ids)

            # If audio was extracted from this video, append <so_start><so_embedding>*M<so_end>
            # so the prompt has placeholder slots for audio embeddings produced by
            # _interleave_video_audio_embeddings at forward time. Reuse
            # get_num_tokens_per_audio (which returns M + 2) and derive M; this keeps
            # the count consistent with get_num_tokens_per_video's accounting and avoids
            # an unnecessary librosa.resample (we only need the resampled length, not
            # the resampled samples themselves).
            # Audio is NOT pruned by EVS, so both streams need identical audio slots.
            if audio is not None and self._audio_extractor is not None:
                num_audio_context = (
                    self.get_num_tokens_per_audio(audio=(audio.samples, audio.sample_rate)) - 2
                )
                audio_ids = (
                    [self._sound_start_token_id]
                    + [self._sound_context_token_id] * num_audio_context
                    + [self._sound_end_token_id]
                )
                expansion.extend(audio_ids)
                if evs_expansion is not None:
                    evs_expansion.extend(audio_ids)

            video_expansions.append(expansion)
            if video_evs_expansions is not None and evs_expansion is not None:
                video_evs_expansions.append(evs_expansion)

        # Dispatch by placeholder token length:
        #
        #   - len == 1: `<video>` is a single added special token, stable
        #     across BPE boundaries — token-level subsequence matching is
        #     always correct, no fallback needed.
        #
        #   - len > 1: `<video>` decomposes into multiple BPE tokens whose
        #     last token can merge with the following character (e.g. `>` +
        #     `\n` fuses into a different token ID), so token-level search
        #     is unreliable. Go straight to the text-level path: decode,
        #     split on the literal `<video>` string, re-encode each segment.
        if len(self._video_placeholder_token_ids) == 1:
            expanded_ids, evs_ids_list = self._token_level_video_replace(
                prompt_token_ids, video_expansions, video_evs_expansions
            )
        else:
            expanded_ids, evs_ids_list = self._text_level_video_replace(
                prompt_token_ids, video_expansions, video_evs_expansions
            )

        evs_ids_tensor = (
            torch.tensor(evs_ids_list, dtype=torch.long) if evs_ids_list is not None else None
        )
        return expanded_ids, evs_ids_tensor

    def _token_level_video_replace(
        self,
        prompt_token_ids: List[int],
        video_expansions: List[List[int]],
        video_evs_expansions: Optional[List[List[int]]] = None,
    ) -> Tuple[List[int], Optional[List[int]]]:
        """Scan `prompt_token_ids` for each `<video>` placeholder as a
        single-token match and splice in the per-video expansion(s).

        Only called when `len(self._video_placeholder_token_ids) == 1`, so
        the placeholder is a stable added special token and a plain ID
        scan is reliable — no BPE boundary merging to worry about.

        Raises `ValueError` if the number of matches doesn't equal the
        number of videos (indicates malformed input, not a fallback case).

        When `video_evs_expansions` is provided (EVS), both streams are
        produced in parallel: outside MM chunks each token is copied into
        both streams; at each match, `video_expansions[i]` goes into
        `expanded` and `video_evs_expansions[i]` goes into `evs`.
        """
        placeholder_id = self._video_placeholder_token_ids[0]
        expanded: List[int] = []
        evs: Optional[List[int]] = [] if video_evs_expansions is not None else None
        video_idx = 0
        for tok in prompt_token_ids:
            if tok == placeholder_id:
                if video_idx >= len(video_expansions):
                    raise ValueError(
                        f"Video expansion: prompt_token_ids contains more "
                        f"'<video>' placeholders than videos "
                        f"({len(video_expansions)})."
                    )
                expanded.extend(video_expansions[video_idx])
                if evs is not None:
                    evs.extend(video_evs_expansions[video_idx])
                video_idx += 1
            else:
                expanded.append(tok)
                if evs is not None:
                    evs.append(tok)

        if video_idx != len(video_expansions):
            raise ValueError(
                f"Video expansion: prompt_token_ids contains {video_idx} "
                f"'<video>' placeholder(s), but mm_data has "
                f"{len(video_expansions)} video(s)."
            )
        return expanded, evs

    def _text_level_video_replace(
        self,
        prompt_token_ids: List[int],
        video_expansions: List[List[int]],
        video_evs_expansions: Optional[List[List[int]]] = None,
    ) -> Tuple[List[int], Optional[List[int]]]:
        """Fallback used when token-level subsequence matching fails because
        BPE merged the `<video>` placeholder with its trailing context.

        Decode `prompt_token_ids` back to text, split on the literal
        `<video>` string, re-tokenize each segment, and interleave with the
        per-video expansions. Correctness-preserving but does incur a
        decode/encode round-trip on CPU.

        When `video_evs_expansions` is provided (EVS), both streams are
        produced: each non-MM segment is tokenized once and emitted into
        both streams; at each placeholder boundary, `video_expansions[i]` is
        appended to `expanded` and `video_evs_expansions[i]` to `evs`.
        """
        text = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
        placeholder = self.video_context_token  # "<video>"
        segments = text.split(placeholder)
        num_placeholders = len(segments) - 1
        if num_placeholders != len(video_expansions):
            raise ValueError(
                f"Video expansion: decoded prompt text contains "
                f"{num_placeholders} '{placeholder}' placeholders, but "
                f"mm_data has {len(video_expansions)} videos."
            )

        expanded: List[int] = []
        evs: Optional[List[int]] = [] if video_evs_expansions is not None else None
        for i, segment in enumerate(segments):
            if segment:
                seg_ids = self.tokenizer.encode(segment, add_special_tokens=False)
                expanded.extend(seg_ids)
                if evs is not None:
                    evs.extend(seg_ids)
            if i < num_placeholders:
                expanded.extend(video_expansions[i])
                if evs is not None and video_evs_expansions is not None:
                    evs.extend(video_evs_expansions[i])
        return expanded, evs

    def _process_images(
        self, images: List[Image.Image | torch.Tensor], text_prompt: str
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        # Multiple images can be processed in one call.
        processed_images = self.processor(images=images, return_tensors="pt").to(self.device)

        # Prepare text prompt for image modality.
        parts = text_prompt.split(self.img_context_token)
        if len(parts) - 1 != len(processed_images["num_patches"]):
            raise ValueError(
                f"Number of {self.img_context_token} tokens ({len(parts) - 1}) doesn't match "
                f"num_patches_list length ({len(processed_images['num_patches'])})"
            )
        processed_query = parts[0]
        for num_patches, part in zip(processed_images["num_patches"], parts[1:]):
            feature_size = num_patches * self.num_image_token
            image_repl = (
                self.img_start_token + self.img_context_token * feature_size + self.img_end_token
            )
            processed_query += image_repl + part

        input_ids = self.tokenizer.encode(
            processed_query, add_special_tokens=False, return_tensors="pt"
        )
        return processed_images, input_ids

    def _process_images_dynamic(
        self, images: List[Image.Image | torch.Tensor], text_prompt: str
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Process images using dynamic resolution tiling.

        Converts images to raw tensors and computes target sizes; resize,
        normalize, and patch rearrangement are deferred to the vision encoder.
        """
        tiler = self.dynamic_tiler

        images = [_media_to_raw_chw(img) for img in images]

        # Compute text-only length for token budgeting.
        sans_images = text_prompt.replace(self.img_context_token, "")
        text_ids = self.tokenizer.encode(sans_images, add_special_tokens=False)
        text_prompt_length = len(text_ids)

        budget = tiler.max_num_tokens_available(text_prompt_length)
        params_list = tiler.compute_params(images, budget)

        raw_tensors = []
        image_sizes = []
        num_tokens_per_image = []
        for params in params_list:
            target_w = params.patch_size[0] * tiler.patch_pixel_size
            target_h = params.patch_size[1] * tiler.patch_pixel_size
            raw_tensors.append(params.media)
            image_sizes.append((target_h, target_w))
            num_tokens_per_image.append(params.num_embeddings)

        # Build text prompt with per-image token counts.
        parts = text_prompt.split(self.img_context_token)
        if len(parts) - 1 != len(images):
            raise ValueError(
                f"Number of {self.img_context_token} tokens ({len(parts) - 1}) doesn't match "
                f"the number of images ({len(images)})"
            )
        processed_query = parts[0]
        for num_tokens, part in zip(num_tokens_per_image, parts[1:]):
            image_repl = (
                self.img_start_token + self.img_context_token * num_tokens + self.img_end_token
            )
            processed_query += image_repl + part

        input_ids = self.tokenizer.encode(
            processed_query, add_special_tokens=False, return_tensors="pt"
        )

        processed_data = {
            "pixel_values": raw_tensors,
            "num_patches": torch.tensor([len(images)]),
            "image_sizes": image_sizes,
            "num_tokens_per_image": num_tokens_per_image,
        }
        return processed_data, input_ids

    def _process_videos_frames(
        self, videos: List[List[Image.Image | torch.Tensor]]
    ) -> Dict[str, Any]:
        """Process video frames using custom video preprocessing.

        Uses video_to_pixel_values for proper resize (optionally
        aspect-ratio-preserving) and mean/std normalization, matching vLLM.
        Falls back to the HF image processor when video_target_num_patches
        is not configured.
        """
        num_patches_list = []
        pixel_values_list = []
        video_size_list = []
        for video in videos:
            num_frames = len(video)

            if self.video_target_num_patches is not None:
                pixel_values = video_to_pixel_values(
                    video,
                    input_size=self.image_size,
                    video_target_num_patches=self.video_target_num_patches,
                    video_maintain_aspect_ratio=self.video_maintain_aspect_ratio,
                    patch_size=self.patch_size,
                    downsample_ratio=self.downsample_ratio,
                    norm_mean=self._video_norm_mean,
                    norm_std=self._video_norm_std,
                )
                t, _, h, w = pixel_values.shape
                num_patches_list.append(torch.tensor([num_frames]))
                pixel_values_list.append(pixel_values)
                video_size_list.append([num_frames, t // num_frames, h, w])
            else:
                # Fallback: use HF image processor with VIDEO_MAX_NUM_TILES.
                orig_max_num_tiles = self.processor.max_num_tiles
                # Several video code paths all assume a single tile per frame.
                # Increasing this constant would require updating those paths
                # accordingly — multi-tile video expansion isn't currently supported and the
                # prompt-side MM counts would diverge from the vision encoder's output.
                # Guard against accidental change.
                assert VIDEO_MAX_NUM_TILES == 1, (
                    "NanoV2VL video paths assume a single tile per frame; see comment above."
                )
                self.processor.max_num_tiles = VIDEO_MAX_NUM_TILES
                try:
                    processed_images = self.processor(images=video, return_tensors="pt").to(
                        self.device
                    )
                finally:
                    self.processor.max_num_tiles = orig_max_num_tiles

                t, _, h, w = processed_images["pixel_values"].shape
                num_patches_list.append(processed_images["num_patches"])
                pixel_values_list.append(processed_images["pixel_values"])
                video_size_list.append([num_frames, t // num_frames, h, w])

        # When aspect-ratio-preserving resize is active, different videos may have different (H, W).
        # Keep per-video tensors to avoid a concat mismatch; fall back to a flat tensor when all
        # shapes agree.
        shapes = {pv.shape[1:] for pv in pixel_values_list}
        if len(shapes) == 1:
            all_pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            # Store as a list - `_extract_video_embeddings_temporal` handles both.
            all_pixel_values = pixel_values_list
        result = {
            "num_patches": torch.tensor(
                [sum(np) if isinstance(np, torch.Tensor) else np.item() for np in num_patches_list]
            ),
            "pixel_values": all_pixel_values,
            "video_size": video_size_list,
        }

        return result

    def _get_frame_separators(
        self, video_size_lst: List[Tuple], video_metadatas: List[Dict[str, Any] | None]
    ) -> List[List[str]]:
        """Build per-tubelet (or per-frame when T=1) separator strings.

        When video_temporal_patch_size > 1, groups T consecutive frames into a
        single separator matching vLLM's `get_video_repl` format.
        """
        T = self.video_temporal_patch_size
        frame_separators_lst = []
        for metadata, video_size in zip(video_metadatas, video_size_lst):
            num_frames = video_size[0]

            if metadata is not None:
                metadata_fps = metadata["fps"]
                frame_duration_ms = int(1000.0 / metadata_fps)
                frames_indices = metadata["frames_indices"]
                timestamps = [int(fi) * frame_duration_ms / 1000.0 for fi in frames_indices]

                if T > 1:
                    frame_separators = self._build_tubelet_separators(timestamps, frames_indices, T)
                else:
                    frame_separators = [
                        f"Frame {i + 1} sampled at {ts:.2f} seconds: "
                        for i, ts in enumerate(timestamps)
                    ]
            else:
                if T > 1:
                    num_tubelets = math.ceil(num_frames / T)
                    frame_separators = [
                        ("\n" if t > 0 else "") + f"Frame {t + 1}: " for t in range(num_tubelets)
                    ]
                else:
                    frame_separators = [f"Frame {i + 1}: " for i in range(num_frames)]
            frame_separators_lst.append(frame_separators)

        return frame_separators_lst

    @staticmethod
    def _build_tubelet_separators(
        timestamps: List[float],
        frames_indices: List[int],
        T: int,
    ) -> List[str]:
        """Build frame separator strings for tubelets of T frames.

        For T > 1:
        `"Frame 1 and frame 2 sampled at X.XX and Y.YY seconds: "`
        """
        num_frames = len(timestamps)
        separators = []
        for group_idx, i in enumerate(range(0, num_frames, T)):
            group_frames = []
            for j in range(T):
                frame_idx = i + j
                if frame_idx < num_frames:
                    ts = timestamps[frame_idx]
                    frame_str = "Frame" if j == 0 else "frame"
                    group_frames.append(f"{frame_str} {frame_idx + 1} sampled at {ts:.2f} seconds")
            if group_frames:
                sep = " and ".join(group_frames) + ": "
                if group_idx > 0:
                    sep = "\n" + sep
                separators.append(sep)
        return separators

    def _process_video_prompts(
        self,
        split_text_prompt: List[str],
        num_tokens_per_frame_lst: List[List[int] | None],
        frame_separators_lst: List[List[str]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Process videos one by one to get correct processed_query.
        processed_query = []
        evs_token_ids: List[int] = []
        for video_index, (num_tokens_per_frame, frame_separators) in enumerate(
            zip(num_tokens_per_frame_lst, frame_separators_lst)
        ):
            # Prepare video and EVS query.
            processed_query.append(split_text_prompt[video_index])
            if self._add_video_prefix:
                processed_query.append("This is a video:\n")
            for frame_sep, num_tokens in zip(frame_separators, num_tokens_per_frame):
                frame_prompts = [
                    frame_sep,
                    self.img_start_token,
                    self.img_context_token * num_tokens,
                    self.img_end_token,
                ]
                processed_query.extend(frame_prompts)
            # Build EVS token IDs at the token-ID level (following vLLM's
            # get_video_repl approach) to avoid BPE splitting special tokens.
            # Each tubelet gets a single video_context_token_id placeholder
            # that merge_evs_mm_embeds will replace with the actual EVS count.
            if self.video_pruning_rate > 0:
                evs_token_ids.extend(
                    self.tokenizer.encode(split_text_prompt[video_index], add_special_tokens=False)
                )
                if self._add_video_prefix:
                    evs_token_ids.extend(
                        self.tokenizer.encode("This is a video:\n", add_special_tokens=False)
                    )
                for frame_sep in frame_separators:
                    evs_token_ids.extend(self.tokenizer.encode(frame_sep, add_special_tokens=False))
                    evs_token_ids.extend(self._img_start_token_ids)
                    evs_token_ids.append(self.video_context_token_id)
                    evs_token_ids.extend(self._img_end_token_ids)
        # Append the last part of the text prompt.
        processed_query.append(split_text_prompt[-1])

        # Tokenize processed query.
        input_ids_lst = [
            self.tokenizer.encode(
                query,
                add_special_tokens=False,
                return_tensors="pt",
            )
            for query in processed_query
        ]
        input_ids = torch.cat(input_ids_lst, dim=1)

        if self.video_pruning_rate > 0:
            evs_token_ids.extend(
                self.tokenizer.encode(split_text_prompt[-1], add_special_tokens=False)
            )
            evs_ids = torch.tensor(evs_token_ids, dtype=torch.long)
        else:
            evs_ids = None

        return input_ids, evs_ids

    def _compute_token_numbers_per_video(self, video_size_lst: List[Tuple]) -> List[List[int]]:
        """Compute the number of embedding tokens per tubelet (or per frame when T=1).

        With `video_temporal_patch_size > 1`, T consecutive frames are grouped into tubelets, so the
        returned list has `ceil(num_frames / T)` entries instead of `num_frames`.
        """
        T = self.video_temporal_patch_size
        num_tokens_per_frame_lst = []
        for video_size in video_size_lst:
            num_frames = video_size[0]
            num_patches_per_frame = video_size[1]
            img_height = video_size[2]
            img_width = video_size[3]
            num_tubelets = math.ceil(num_frames / T) if T > 1 else num_frames

            # Compute per-frame (or per-tubelet) token count from actual frame dimensions, not the
            # default `self.num_image_token`, since video_target_num_patches may change the frame size.
            tokens_per_unit = int(
                (img_height * img_width // self.patch_size**2) * (self.downsample_ratio**2)
            )

            if self.video_pruning_rate > 0:
                desired_num_tokens = compute_retained_tokens_count(
                    video_size=(num_tubelets, num_patches_per_frame * img_height, img_width),
                    spatial_merge_size=self.spatial_merge_size,
                    pruning_ratio=self.video_pruning_rate,
                )
                # Dummy tokens; adjusted in VisionEncoder after EVS.
                num_tokens_per_frame = [desired_num_tokens] + [0] * (num_tubelets - 1)
            else:
                num_tokens_per_frame = [num_patches_per_frame * tokens_per_unit] * num_tubelets

            num_tokens_per_frame_lst.append(num_tokens_per_frame)
        return num_tokens_per_frame_lst

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data = inputs.get("prompt"), inputs.get("multi_modal_data", {})
        images = mm_data.get("image", None)
        videos = mm_data.get("video", None)
        audios = mm_data.get("audio", None)
        # TODO(TRTLLM-11390): Functionally support multiple modalities in the same request.
        if sum([images is not None, videos is not None, audios is not None]) > 1:
            raise ValueError(
                "NanoV2VL does not support different modalities in the same prompt yet."
            )

        if images is None and videos is None and audios is None:
            input_ids = self.tokenizer.encode(
                text_prompt, add_special_tokens=False, return_tensors="pt"
            )
            return input_ids[0].to(torch.int32).tolist(), {}

        modality_type = None
        modality_data = dict()
        input_ids = None
        if images is not None:
            modality_type = "image"
            if self.dynamic_tiler is not None:
                # Dynamic resolution path.
                processed_data, input_ids = self._process_images_dynamic(images, text_prompt)
                modality_data["pixel_values"] = processed_data["pixel_values"]
                modality_data["num_patches"] = processed_data["num_patches"]
                modality_data["image_sizes"] = processed_data["image_sizes"]
                modality_data["num_tokens_per_image"] = processed_data["num_tokens_per_image"]
            else:
                # Existing fixed-tile path.
                processed_images, input_ids = self._process_images(images, text_prompt)
                modality_data["pixel_values"] = processed_images["pixel_values"].to(self.dtype)
                modality_data["num_patches"] = processed_images["num_patches"].sum(
                    dim=0, keepdim=True
                )
            modality_data["video_size"] = None
            # During model inference, the image/video modality data can be mixed during inflight-batching.
            # Store input_ids for image modality here when EVS is enabled,
            # which will be used in merge_evs_mm_embeds later.
            modality_data["evs_ids"] = (
                input_ids[0].to(torch.int32) if self.video_pruning_rate > 0 else None
            )
        elif videos is not None:
            modality_type = "video"
            video_frames, video_metadatas, video_audios = (
                [video_data.frames for video_data in videos],
                [video_data.metadata for video_data in videos],
                [getattr(video_data, "audio", None) for video_data in videos],
            )
            num_videos = len(video_frames)
            processed_images = self._process_videos_frames(video_frames)

            # Num_tokens_per_frame_lst is a dummy one when EVS is enabled.
            num_tokens_per_frame_lst = self._compute_token_numbers_per_video(
                processed_images["video_size"]
            )
            # Special video tokens will be added to match training behaviors.
            frame_separators_lst = self._get_frame_separators(
                processed_images["video_size"], video_metadatas
            )

            text_prompt, audio_data = self._extract_audio_from_video(text_prompt, video_audios)

            split_text_prompt = text_prompt.split(self.video_context_token)
            if len(split_text_prompt) - 1 != num_videos:
                raise ValueError(
                    f"Number of {self.video_context_token} tokens ({len(split_text_prompt) - 1})"
                    f"doesn't match the number of videos ({num_videos})"
                )
            input_ids, evs_ids = self._process_video_prompts(
                split_text_prompt, num_tokens_per_frame_lst, frame_separators_lst
            )
            pv = processed_images["pixel_values"]
            modality_data["pixel_values"] = (
                [v.to(self.dtype) for v in pv] if isinstance(pv, list) else pv.to(self.dtype)
            )
            modality_data["num_patches"] = processed_images["num_patches"].sum(dim=0, keepdim=True)
            modality_data["video_size"] = processed_images["video_size"]
            modality_data["evs_ids"] = evs_ids
            if audio_data is not None:
                modality_data["audio"] = audio_data
        elif audios is not None:
            modality_type = "audio"
            input_ids, modality_data = self._process_audio(text_prompt, audios)

        # Will package inputs for language model forward in AGGREGATE mode.
        multimodal_data = {}
        multimodal_data["modality_type"] = modality_type
        multimodal_data[modality_type] = modality_data
        return input_ids[0].to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }

    def _prepare_audio_features(
        self,
        text: str,
        audios: List[Union[np.ndarray, Tuple[np.ndarray, int]]],
    ) -> Tuple[str, dict]:
        """Resample audios, expand placeholder tokens, and extract mel-spectrogram features.

        Returns the expanded text prompt and an audio data dict.
        """
        extractor = self._audio_extractor
        target_sr = extractor.sampling_rate
        audios = self._resample_audios(audios, target_sr)

        expanded_text = self._expand_audio_placeholders(text, audios, extractor)

        audio_inputs = extractor(
            audios,
            sampling_rate=extractor.sampling_rate,
            return_tensors="pt",
        )
        audio_data = {
            "input_audio_features": audio_inputs.input_features,
            "feature_attention_mask": audio_inputs.attention_mask,
        }
        # audio_num_clips records how many clips each audio stream was split
        # into. Needed to regroup per-clip embeddings back to per-video.
        audio_data["audio_num_clips"] = audio_inputs.audio_num_clips
        return expanded_text, audio_data

    def _process_audio(
        self,
        text: str,
        audios: List[Union[np.ndarray, Tuple[np.ndarray, int]]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._audio_extractor is None:
            raise ValueError(
                "Audio inputs were passed in, but no audio preprocessing was configured "
                "due to the absence of a `sound_config` in the model config."
            )

        expanded_text, audio_inputs = self._prepare_audio_features(text, audios)

        input_ids = self.tokenizer.encode(
            expanded_text, add_special_tokens=False, return_tensors="pt"
        )
        return input_ids, audio_inputs

    def _extract_audio_from_video(
        self,
        text_prompt: str,
        video_audios: List[Optional[AudioData]],
    ) -> Tuple[str, Optional[dict]]:
        """Extract structured video audio streams and prepare audio features.

        Injects audio placeholder tokens after each video that carries an audio stream, resamples
        audio to the extractor's target sample rate, expands placeholder tokens, and computes
        mel-spectrogram features.

        Returns the (possibly modified) text prompt and an audio data dict (or
        `None` when no audio is present).
        """
        has_audio = [audio is not None for audio in video_audios]
        audio_from_video = [
            (audio.samples, audio.sample_rate) for audio in video_audios if audio is not None
        ]

        if not audio_from_video or self._audio_extractor is None:
            return text_prompt, None

        # Inject <so_embedding> after each <video> that has audio.
        # Split on <video> and rebuild so each placeholder lands after the
        # correct video, not all after the first one.
        parts = text_prompt.split(self.video_context_token)
        if len(parts) - 1 != len(video_audios):
            raise ValueError(
                f"Number of {self.video_context_token} tokens ({len(parts) - 1}) "
                f"doesn't match the number of videos ({len(video_audios)})"
            )
        rebuilt = [parts[0]]
        for i, part in enumerate(parts[1:]):
            rebuilt.append(self.video_context_token)
            if has_audio[i]:
                rebuilt.append(self._sound_context_token)
            rebuilt.append(part)
        text_prompt = "".join(rebuilt)

        text_prompt, audio_data = self._prepare_audio_features(text_prompt, audio_from_video)
        audio_data["has_audio"] = has_audio
        return text_prompt, audio_data

    def _expand_audio_placeholders(
        self,
        text: str,
        audios: List[np.ndarray],
        extractor: "ParakeetExtractor",
    ) -> str:
        """Replace each audio placeholder token with the expanded start/context/end sequence."""
        parts = [x for x in re.split(f"({re.escape(self._sound_context_token)})", text) if x]
        token_count = parts.count(self._sound_context_token)
        if token_count != len(audios):
            raise ValueError(
                "Number of audio tokens in text does not match the number "
                f"of audios (tokens={token_count}, audios={len(audios)})."
            )
        audio_idx = 0
        for idx, part in enumerate(parts):
            if part == self._sound_context_token:
                audio = audios[audio_idx]
                num_tokens = extractor.audio_token_count(len(audio))
                expanded = (
                    f"{self._sound_start}{self._sound_context_token * num_tokens}{self._sound_end}"
                )
                parts[idx] = expanded
                audio_idx += 1
        return "".join(parts)

    def get_num_tokens_per_audio(
        self,
        *,
        audio: Union[np.ndarray, Tuple[np.ndarray, int]],
        **kwargs,
    ) -> int:
        """Return the total number of MM tokens for a single audio item.

        The count includes the `<so_start>` and `<so_end>` special tokens.
        The audio is resampled to the extractor's sampling rate when
        provided as a `(np.ndarray, int)` tuple, so the returned count matches
        what `extractor.audio_token_count` would produce in the full path
        (`_process_audio` -> `_resample_audios` -> `_expand_audio_placeholders`).

        Args:
            audio (Union[np.ndarray, Tuple[np.ndarray, int]]): Either raw audio
                samples (assumed to already be at the extractor's sampling
                rate), or a `(samples, sample_rate)` tuple. If the provided
                sample rate differs from the extractor's, the audio is
                resampled before counting tokens.
            **kwargs: Ignored; accepted for interface compatibility with
                other `get_num_tokens_per_*` methods.

        Returns:
            int: The total number of MM tokens (feature tokens + 2 for
                `<so_start>` and `<so_end>`) that a single audio item expands
                to in the prompt.
        """
        if self._audio_extractor is None:
            raise ValueError(
                "Audio inputs were passed in, but no audio preprocessing was configured "
                "due to the absence of a `sound_config` in the model config."
            )

        extractor = self._audio_extractor
        target_sr = extractor.sampling_rate

        # Unpack (audio_data, sample_rate) tuples and resample if needed.
        if isinstance(audio, tuple):
            audio_data, orig_sr = audio
        else:
            audio_data = audio
            orig_sr = target_sr

        audio_length = len(audio_data)
        if orig_sr != target_sr:
            audio_length = math.ceil(audio_length * (target_sr / orig_sr))

        num_context_tokens = extractor.audio_token_count(audio_length)
        # +2 for <so_start> and <so_end>
        return num_context_tokens + 2

    @staticmethod
    def _resample_audios(
        audios: List[Union[np.ndarray, Tuple[np.ndarray, int]]],
        target_sr: int,
    ):
        resampled_audios: List[np.ndarray] = []
        for item in audios:
            if isinstance(item, tuple):
                audio_data, orig_sr = item
            else:
                audio_data = item
                orig_sr = target_sr

            if orig_sr != target_sr:
                import librosa

                # We use `fix=True` (even though it's currently the default) to guarantee that the
                # output length matches what we calculate in `get_num_tokens_per_audio`.
                audio_data = librosa.resample(
                    audio_data, orig_sr=orig_sr, target_sr=target_sr, fix=True
                )
            resampled_audios.append(audio_data)

        return resampled_audios


_NANO_VL_PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={
        "image": IMAGE_PLACEHOLDER,
        "video": VIDEO_PLACEHOLDER,
        "audio": AUDIO_PLACEHOLDER,
    },
    placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    placeholders_separator="\n",
)


@register_auto_model("NemotronH_Nano_Omni_Reasoning_V3")
@register_auto_model("NemotronH_Nano_VL_V2")
@register_input_processor(
    NanoV2VLInputProcessor,
    model_type="NemotronH_Nano_VL_V2",
    placeholder_metadata=_NANO_VL_PLACEHOLDER_METADATA,
)
@register_input_processor(
    NanoV2VLInputProcessor,
    model_type="NemotronH_Nano_Omni_Reasoning_V3",
    placeholder_metadata=_NANO_VL_PLACEHOLDER_METADATA,
)
class NemotronH_Nano_VL_V2(transformers.PreTrainedModel):
    _supports_flash_attn_2 = True

    def __init__(self, model_config: ModelConfig):
        if _is_disagg():
            raise ValueError("NanoV2VL does not support disaggregated inference yet.")

        config = model_config.pretrained_config
        super().__init__(config)

        self.model_config = model_config
        llm_model_config = copy.deepcopy(model_config)
        if hasattr(self, "llm"):
            return

        # Vision and sound encoders are constructed lazily in load_weights(),
        # after MetaInitMode has exited, because their HuggingFace-based
        # submodules (RADIO's nn.LayerNorm, HFParakeetEncoder, ...) use
        # deterministic init ops (ones_, zeros_, fill_, .to(dtype=...),
        # .detach()) that raise MetaInitException on meta tensors and would
        # otherwise force the entire model onto the slow fallback path.
        # Vision+sound weights are only ~2GB combined, so allocating them
        # on CPU first is cheap, and the LLM's fast meta-init path is kept
        # intact.
        #
        # Snapshot the multimodal ModelConfig now — self.post_config() below
        # reassigns self.model_config.pretrained_config to the LLM-only
        # NemotronHConfig, which loses vision_config / sound_config / etc.
        self._mm_model_config = copy.deepcopy(model_config)
        self.vision_encoder: Optional[NanoV2VLVisionEncoder] = None
        self.sound_encoder: ProjectedParakeet | None = None

        llm_model_config.pretrained_config = llm_model_config.pretrained_config.llm_config
        self._update_config_for_quantization(llm_model_config)

        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.vocab_size = llm_model_config.pretrained_config.vocab_size
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        self.img_context_token_id = config.img_context_token_id
        self.video_context_token_id = config.video_context_token_id
        self.sound_context_token_id = getattr(config, "sound_context_token_id", None)
        self.post_config()
        self.is_loaded = True
        # Use config value if explicitly set (EVS enabled), otherwise default to 0.0 (EVS disabled)
        self.video_pruning_rate = (
            model_config.video_pruning_rate if model_config.video_pruning_rate is not None else 0.0
        )

    def load_weights(self, weights):
        # Construct vision/sound encoders here (outside MetaInitMode) so their
        # HF-based submodules allocate regular CPU tensors. Then move them to
        # CUDA since model.to("cuda") in model_loader already ran for the LLM.
        # Use the snapshot of the multimodal ModelConfig taken in __init__ —
        # self.model_config.pretrained_config was overwritten by post_config()
        # to be the LLM-only config and no longer has vision_config /
        # sound_config / force_image_size / etc.
        mm_pretrained = self._mm_model_config.pretrained_config
        if self.vision_encoder is None and not _is_disagg():
            self.vision_encoder = NanoV2VLVisionEncoder(self._mm_model_config).eval().to("cuda")
        sound_config = getattr(mm_pretrained, "sound_config", None)
        if self.sound_encoder is None and sound_config is not None:
            self.sound_encoder = (
                ProjectedParakeet(
                    sound_config,
                    llm_hidden_size=mm_pretrained.llm_config.hidden_size,
                    dtype=getattr(mm_pretrained, "torch_dtype", torch.bfloat16),
                )
                .eval()
                .to("cuda")
            )

        # Load vision encoder weights.
        if self.vision_encoder is not None:
            self.vision_encoder.load_weights(weights)

        # Free vision encoder weights from the dict so the backing mmap pages can be released.
        if hasattr(weights, "mark_consumed"):
            weights.mark_consumed("vision_model")
            weights.mark_consumed("mlp1")

        # Load sound encoder weights.
        if self.sound_encoder is not None:
            self.sound_encoder.load_weights(weights)

        # Free sound encoder weights (whether loaded or not) to allow shard 1 mmap to be released.
        if hasattr(weights, "mark_consumed"):
            weights.mark_consumed("sound_encoder")
            weights.mark_consumed("sound_projection")

        # Load language model weights.
        filtered_weights = {
            k.replace("language_model.", ""): v
            for k, v in weights.items()
            if k.startswith("language_model.")
        }
        weight_mapper = NemotronHHfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)
        self.llm.load_weights(filtered_weights, weight_mapper=weight_mapper)

        # Free LLM weights from the dict to release the backing mmap pages for shards 2-17.
        if hasattr(weights, "mark_consumed"):
            weights.mark_consumed("language_model")

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def merge_evs_mm_embeds(
        self,
        num_tokens_in_videos: List[int],
        multimodal_params: List[MultimodalParams],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Merge EVS and MM embeds into input_ids.

        Args:
            num_tokens_in_videos: List of number of tokens in videos.
            multimodal_params: List of multimodal parameters. The shape is [num_context_requests, ].
            input_ids: Original dummy input ids. Note: it contains the generation ids when enabling inflight-batching.
                The shape is [num_tokens_context + num_tokens_generation, ].

        Returns:
            Input ids with EVS and MM embeds merged.
        """
        multimodal_data_lst = [
            multimodal_param.multimodal_data for multimodal_param in multimodal_params
        ]
        modalities = [multimodal_data["modality_type"] for multimodal_data in multimodal_data_lst]
        # Skip EVS if there is no video modality.
        if "video" not in modalities:
            return input_ids

        evs_ids_lst = [
            multimodal_data[modality]["evs_ids"]
            for modality, multimodal_data in zip(modalities, multimodal_data_lst)
        ]
        # Iterate over batch, replacing video_context_token_id placeholders with
        # the actual per-tubelet img_context_token counts from EVS.
        context_parts = []
        for evs_ids, modality, num_tokens_in_video in zip(
            evs_ids_lst, modalities, num_tokens_in_videos
        ):
            # Image modality: keep input_ids unchanged during inflight-batching.
            if modality == "image":
                context_parts.append(evs_ids)
                continue

            # evs_ids is a flat 1-D tensor built at token-ID level.
            # Find placeholder positions and replace each with the EVS count.
            placeholder_mask = evs_ids == self.video_context_token_id
            placeholder_positions = placeholder_mask.nonzero(as_tuple=True)[0]
            image_idx = 0
            prev_end = 0
            for pos in placeholder_positions:
                pos = pos.item()
                # Append tokens before this placeholder.
                if pos > prev_end:
                    context_parts.append(evs_ids[prev_end:pos])
                # Replace placeholder with actual img_context_token count.
                context_parts.append(
                    torch.full(
                        (int(num_tokens_in_video[image_idx]),),
                        fill_value=self.img_context_token_id,
                        dtype=evs_ids.dtype,
                        device=evs_ids.device,
                    )
                )
                image_idx += 1
                prev_end = pos + 1
            # Append remaining tokens after the last placeholder.
            if prev_end < len(evs_ids):
                context_parts.append(evs_ids[prev_end:])

        context_ids = torch.cat(context_parts, dim=0)
        # -> [num_tokens, ]

        # Special handling for inflight-batching.
        # Assume input ids format is [context_ids, generation_ids].
        input_ids[: context_ids.shape[0]] = context_ids
        del context_ids

        return input_ids

    def _encode_audio_data(self, audio_data: dict) -> Tuple[torch.Tensor, List[int]]:
        """Encode audio feature dict into LLM-space embeddings.

        Unlike `_encode_audio` which reads from `param.multimodal_data["audio"]`, this helper
        accepts the audio feature dict directly so it can be reused for audio extracted from video
        metadata.

        Returns:
            A tuple of (flat_embeddings, per_clip_token_counts) where
            `flat_embeddings` has shape `[total_tokens, llm_hidden_size]` and
            `per_clip_token_counts` contains the number of output tokens for
            each audio clip.
        """
        input_features = audio_data["input_audio_features"]  # [num_clips, time, mel_bins]
        attention_mask = audio_data["feature_attention_mask"]  # [num_clips, time]

        target_device = next(self.sound_encoder.parameters()).device
        input_features = input_features.to(dtype=self.model_dtype, device=target_device)
        attention_mask = attention_mask.to(device=target_device)

        sound_embeds = self.sound_encoder(input_features, attention_mask)

        valid_input_lens = attention_mask.sum(dim=1)
        valid_output_lens = self.sound_encoder.encoder._get_subsampling_output_length(
            valid_input_lens
        )

        truncated = []
        per_clip_counts: List[int] = []
        for i in range(sound_embeds.shape[0]):
            valid_len = int(valid_output_lens[i].item())
            truncated.append(sound_embeds[i, :valid_len])
            per_clip_counts.append(valid_len)

        return torch.cat(truncated, dim=0), per_clip_counts  # [total_tokens, llm_hidden_size]

    def _encode_audio(self, param: MultimodalParams) -> torch.Tensor:
        """Encode audio features into LLM-space embeddings."""
        emb, _ = self._encode_audio_data(param.multimodal_data["audio"])
        return emb

    def _interleave_video_audio_embeddings(
        self,
        vision_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        per_clip_audio_counts: List[int],
        has_audio: List[bool],
        audio_num_clips: torch.Tensor,
        video_sizes: List[List[int]],
        evs_num_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Interleave per-video vision and audio embeddings to match input_ids order.

        The vision encoder concatenates all videos' embeddings `[v1_vis, v2_vis, ...]`
        and the audio encoder concatenates all clips `[v1_aud, v2_aud, ...]`.
        `input_ids` expects them interleaved per-video:
        `[v1_vis, v1_aud, v2_vis, v2_aud, ...]`.
        """
        assert len(has_audio) == len(video_sizes), (
            f"has_audio length ({len(has_audio)}) != video_sizes length ({len(video_sizes)})"
        )

        # Get per-video vision token counts.
        vision_enc = self.vision_encoder
        T = vision_enc.video_temporal_patch_size

        per_video_geom = [
            vision_enc._video_tubelet_geometry(vs[0], T, vs[2], vs[3]) for vs in video_sizes
        ]

        if evs_num_tokens is not None:
            # EVS active: evs_num_tokens is a 1-D tensor of per-tubelet retained
            # token counts across all videos. Split by tubelets-per-video and sum.
            tubelets_per_video = [nt for nt, _ in per_video_geom]
            vision_counts = [
                int(chunk.sum().item()) for chunk in torch.split(evs_num_tokens, tubelets_per_video)
            ]
        else:
            # No EVS: deterministic from video_size.
            vision_counts = [
                nt * vs[1] * wh for (nt, wh), vs in zip(per_video_geom, video_sizes, strict=True)
            ]

        # Group per-clip audio counts into per-video audio token counts, since each audio stream
        # may be split into multiple clips by the extractor.
        # `audio_num_clips` tells us how many clips belong to each stream.
        per_video_audio_counts: List[int] = []
        clip_offset = 0
        for num_clips in audio_num_clips.tolist():
            per_video_audio_counts.append(
                sum(per_clip_audio_counts[clip_offset : clip_offset + num_clips])
            )
            clip_offset += num_clips

        # Split and interleave.
        vision_parts = vision_emb.split(vision_counts)
        audio_parts = audio_emb.split(per_video_audio_counts)

        parts: List[torch.Tensor] = []
        audio_idx = 0
        for i, vision_part in enumerate(vision_parts):
            parts.append(vision_part)
            if has_audio[i]:
                parts.append(audio_parts[audio_idx])
                audio_idx += 1

        return torch.cat(parts, dim=0)

    def _encode_multimodal(self, multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        """Dispatch multimodal encoding to the appropriate encoder.

        Returns a single-element `List[torch.Tensor]` (all per-request
        embeddings concatenated) to conform to the contract expected by
        `get_multimodal_embeddings`, which enables chunked-prefill
        caching.  Per-request `num_tokens_in_video` (needed by EVS) is
        stashed in each param's `multimodal_data` dict as a
        side-channel.
        """
        mm_embeddings = []
        for param in multimodal_params:
            modality_type = param.multimodal_data["modality_type"]
            if modality_type in ("image", "video"):
                embs, num_tokens = self.vision_encoder([param])
                vision_emb = embs[0]

                # If audio was extracted from video, encode it and interleave
                # with per-video vision embeddings so that the combined tensor
                # matches the token order in input_ids
                # (v1_img_context, v1_sound_context, v2_img_context, ...).
                audio_data = param.multimodal_data[modality_type].get("audio")
                if audio_data is not None and self.sound_encoder is not None:
                    audio_emb, per_clip_audio_counts = self._encode_audio_data(audio_data)
                    video_sizes = param.multimodal_data[modality_type].get("video_size", [])
                    vision_emb = self._interleave_video_audio_embeddings(
                        vision_emb,
                        audio_emb,
                        per_clip_audio_counts,
                        has_audio=audio_data["has_audio"],
                        audio_num_clips=audio_data["audio_num_clips"],
                        video_sizes=video_sizes,
                        evs_num_tokens=(num_tokens[0] if num_tokens is not None else None),
                    )

                mm_embeddings.append(vision_emb)

                # Stash per-request token counts for later EVS adjustment.
                if num_tokens is not None:
                    param.multimodal_data["num_tokens_in_video"] = num_tokens[0]
            elif modality_type == "audio":
                mm_embeddings.append(self._encode_audio(param))
            else:
                raise ValueError(f"Unknown modality: {modality_type}")

        # Concatenate per-request embeddings into a single tensor.
        # `get_multimodal_embeddings` expects a single-element list containing one tensor (all
        # items' embeddings concatenated).
        if mm_embeddings:
            return [torch.cat(mm_embeddings, dim=0)]
        return []

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        VLM forward logic with inflight batching support.
        """
        num_context_requests, num_generation_requests = (
            attn_metadata.num_contexts,
            attn_metadata.num_generations,
        )
        logger.debug(
            f"num_context_requests: {num_context_requests}, num_generation_requests: {num_generation_requests}"
        )

        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embedding = []
        if len(multimodal_params) > 0:
            if not _is_disagg():
                mm_embedding = get_multimodal_embeddings(
                    encoder_forward_fn=self._encode_multimodal,
                    multimodal_params=multimodal_params[:num_context_requests],
                )
            else:
                raise NotImplementedError(
                    "Nano-V2-VLM does not support disaggregated inference yet. Please unset "
                    "the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )
            # Adjust input_ids in videos if EVS is applied.
            if self.video_pruning_rate > 0:
                # Retrieve per-video count stashed by `_encode_multimodal`.
                ctx_params = multimodal_params[:num_context_requests]
                num_tokens_in_videos = [
                    param.multimodal_data.get("num_tokens_in_video")
                    for param in ctx_params
                    if param.has_content()
                ]
                input_ids = self.merge_evs_mm_embeds(
                    num_tokens_in_videos,
                    multimodal_params=ctx_params,
                    input_ids=input_ids,
                )

            mm_embedding = find_input_mm_embeds(
                mm_embedding, multimodal_params[:num_context_requests]
            )

        mm_token_ids_list = [self.img_context_token_id]
        if self.sound_context_token_id is not None:
            mm_token_ids_list.append(self.sound_context_token_id)
        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embedding,
            mm_token_ids=torch.tensor(mm_token_ids_list, dtype=torch.int32),
        )

        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            lora_params=kwargs.get("lora_params", None),
        )

        logger.debug(f"output shape: {output_prob.shape}")
        return output_prob

    @staticmethod
    def _update_config_for_quantization(llm_model_config: ModelConfig) -> None:
        # Strip the VL wrapper prefix from exclude_modules and
        # quant_config_dict so patterns match the inner LLM's module names
        # (e.g. "language_model.backbone.layers.0.mixer.conv1d" becomes
        # "backbone.layers.0.mixer.conv1d").
        _LM_PREFIX = "language_model."
        if llm_model_config.quant_config.exclude_modules is not None:
            llm_model_config.quant_config.exclude_modules = [
                m[len(_LM_PREFIX) :] if m.startswith(_LM_PREFIX) else m
                for m in llm_model_config.quant_config.exclude_modules
            ]
        if llm_model_config.quant_config_dict is not None:
            # NOTE: without `_frozen` toggling, `ModelConfig` cannot have its attributes
            # modified.
            old_frozen = llm_model_config._frozen
            llm_model_config._frozen = False
            try:
                llm_model_config.quant_config_dict = {
                    k[len(_LM_PREFIX) :] if k.startswith(_LM_PREFIX) else k: v
                    for k, v in llm_model_config.quant_config_dict.items()
                }
            finally:
                llm_model_config._frozen = old_frozen


def _rearrange_img(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    py = x.shape[-2] // patch_size
    px = x.shape[-1] // patch_size
    return einops_rearrange(
        x,
        "c (py yy) (px xx) -> (py px) (c yy xx)",
        py=py,
        yy=patch_size,
        px=px,
        xx=patch_size,
    )
