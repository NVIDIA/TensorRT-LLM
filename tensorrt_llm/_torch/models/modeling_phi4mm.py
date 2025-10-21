# Plan for phi4-mm model support.
# (done) step 1: support legacy inference pipeline for phi4-mm model.
# (done) step 2: refactor the inference pipeline to use AGGREGATE mode (https://github.com/NVIDIA/TensorRT-LLM/pull/5522).
# (done) step 3: optimization phi4-mm image modality inference.
# (todo) step 4: misc tasks:
#   * optimize audio modality.
#   * use TRTLLM-attention to replace original pytorch attention in vision/audio encoders.
#   * use data parallel to accelerate inference.

import copy
import enum
import importlib
import math
import os
import sys
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
import transformers
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import get_image_size, pad, resize
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (ImageInput, is_pil_image,
                                      make_list_of_images, valid_images)
from transformers.utils import TensorType

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ...executor.request import LoRARequest
from ...inputs import (BaseMultimodalInputProcessor, ExtraProcessedInputs,
                       InputProcessor, MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...lora_helper import LoraConfig
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (find_input_mm_embeds, fuse_input_embeds,
                                        get_multimodal_embeddings)
from .modeling_utils import register_auto_model

# Special token ids from the original Phi-4-multimodal-instruct implementation
# Hardcoded in https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py#L44.
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>' from HF `modeling_phi4mm.py`
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>' from HF `modeling_phi4mm.py`
_PAD_TOKEN_ID = 199999  # '<|endoftext|>' from HF `special_tokens_map.json`
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999,
                                            -1]  # from HF `modeling_phi4mm.py`
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float('-inf'), -10000
                                            ]  # from HF `modeling_phi4mm.py`

# SigLip input config.
# Hardcoded in https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py#L195.
_BASE_RESOLUTION = 448
_MASK_RESOLUTION = _BASE_RESOLUTION // 14

# Below classes will be loaded from HuggingFace code, rather than using transformers version,
# since transformers version is not compatible with checkpoints and configs from `microsoft/Phi-4-multimodal-instruct`.
Phi4MMAudioEmbedding = None
Phi4MMImageEmbedding = None
Phi4MMConfig = None


# Make this a runtime lookup rather than a module-wide constant for easier unit testing.
def _is_torch_compile() -> bool:
    return os.getenv("TLLM_MULTIMODAL_ENCODER_TORCH_COMPILE", "0") == "1"


def _is_disagg() -> bool:
    return os.getenv("TLLM_MULTIMODAL_DISAGGREGATED", "0") == "1"


# Load the Phi4MM classes from HuggingFace Phi-4-multimodal-instruct repo.
# Remove this function by using the transformers version of Phi4Multimodal when weights/configs are converted to transformers format.
def _load_phi4mm_classes(local_path):
    """Load Phi4MM classes from the specified local path."""
    global Phi4MMAudioEmbedding, Phi4MMImageEmbedding, Phi4MMConfig
    if Phi4MMAudioEmbedding is not None and Phi4MMImageEmbedding is not None and Phi4MMConfig is not None:
        return

    # Add parent folder to sys.path to enable relative import.
    original_sys_path = sys.path.copy()
    package_folder = Path(local_path)
    package_name = package_folder.name
    parent_folder = str(package_folder.parent)
    if parent_folder not in sys.path:
        sys.path.insert(0, parent_folder)
    try:
        # Import Phi4MMConfig from configuration_phi4mm.py.
        config_path = os.path.join(local_path, 'configuration_phi4mm.py')
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"configuration_phi4mm.py not found at {local_path}.")
        spec = importlib.util.spec_from_file_location("hf_config", config_path)
        hf_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hf_config)
        Phi4MMConfig = hf_config.Phi4MMConfig

        # Import Phi4MMAudioEmbedding and Phi4MMImageEmbedding from modeling_phi4mm.py.
        modeling_phi4mm_path = os.path.join(local_path, 'modeling_phi4mm.py')
        if not os.path.exists(modeling_phi4mm_path):
            raise FileNotFoundError(
                f"modeling_phi4mm.py not found at {local_path}.")
        # `Phi-4-multimodal-instruct` as the package name to avoid relative import errors.
        # `hf_modeling_phi4mm` as the module name to avoid name conflicts.
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.hf_modeling_phi4mm", modeling_phi4mm_path)
        hf_modeling_phi4mm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hf_modeling_phi4mm)
        Phi4MMAudioEmbedding = hf_modeling_phi4mm.Phi4MMAudioEmbedding
        Phi4MMImageEmbedding = hf_modeling_phi4mm.Phi4MMImageEmbedding
    finally:
        sys.path = original_sys_path


# Below code is optimized for image modality inference, including input_processor and vision encoder forward.
def vision_encoder_forward(self,
                           input_ids: torch.LongTensor,
                           input_embeds: torch.FloatTensor,
                           image_sizes=None,
                           **kwargs) -> torch.FloatTensor:
    """Optimize vision encoder forward.

    * Remove many unnecessary if-else conditions and assertions.
    * Optimize positions/positions_tuple calculation.
    * Optimize the input parameters for get_img_features.

    Ref code: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/modeling_phi4mm.py#L268
    """
    if isinstance(image_sizes, torch.Tensor):
        image_sizes = image_sizes.view(-1, 2)
    image_attention_mask = kwargs['image_attention_mask']
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    positions = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID,
                              as_tuple=False)
    positions_tuple = torch.unbind(positions, dim=1)
    if isinstance(self.img_projection, torch.nn.Sequential):
        target_device = self.img_projection[0].bias.device
        target_dtype = self.img_projection[0].bias.dtype
    else:
        target_device = self.img_projection.bias.device
        target_dtype = self.img_projection.bias.dtype

    # Inference with SigLip Vision Encoder.
    batch_size = input_embeds.shape[0]
    input_embeds = input_embeds.flatten(0, 1)
    flatten_img_attn_mask = image_attention_mask.to(torch.bool).flatten(0, 1)
    img_features = self.get_img_features(input_embeds,
                                         attention_mask=flatten_img_attn_mask)

    # Reshape and combine global/sub image features.
    base_resolution = self.crop_size
    base_feat_height_reduction = self.base_feat_height_reduction
    base_feat_height = self.base_feat_height_target
    img_features = img_features.view(batch_size, -1,
                                     base_feat_height * base_feat_height,
                                     self.image_dim_out)
    C = self.image_dim_out
    H = base_feat_height
    rh = base_feat_height_reduction
    _h1 = H // rh
    _fused_dim = rh * rh * C
    output_imgs, output_len = [], []
    for batch_idx in range(batch_size):
        h, w = image_sizes[batch_idx]
        h = h // base_resolution
        w = w // base_resolution
        B_ = h * w

        # Global image features.
        global_image_feature = img_features[batch_idx, :1]
        global_image_feature = global_image_feature.reshape(1, H, H, C)
        global_image_feature = global_image_feature.reshape(
            1, _h1, rh, _h1, rh, C)
        global_image_feature = global_image_feature.permute(0, 1, 3, 2, 4, 5)
        global_image_feature = global_image_feature.reshape(
            1, _h1, _h1, rh * rh * C)
        global_GN = self.sub_GN.repeat(1, _h1, 1, 1)
        global_image_feature = torch.cat([global_image_feature, global_GN],
                                         dim=2).reshape(1, -1, _fused_dim)

        # Sub image features.
        sub_image_feature = img_features[batch_idx, 1:(1 + B_)]
        sub_image_feature = sub_image_feature.reshape(B_, H, H, C)
        sub_image_feature = sub_image_feature.reshape(B_, _h1, rh, _h1, rh, C)
        sub_image_feature = sub_image_feature.permute(0, 1, 3, 2, 4, 5)
        sub_image_feature = sub_image_feature.reshape(B_, -1, rh * rh * C)
        sub_image_feature = sub_image_feature.reshape(1, h, w, _h1, _h1, -1)
        sub_image_feature = sub_image_feature.permute(0, 1, 3, 2, 4, 5)
        sub_image_feature = sub_image_feature.reshape(1, h * _h1, w * _h1,
                                                      rh * rh * C)

        # Fetch useful content.
        downsample_attn_mask = image_attention_mask[batch_idx, 1:B_ + 1, 0::2,
                                                    0::2]
        downsample_attn_mask = downsample_attn_mask.reshape(1, h, w, _h1, _h1)
        downsample_attn_mask = downsample_attn_mask.permute(0, 1, 3, 2, 4)
        downsample_attn_mask = downsample_attn_mask.reshape(1, h * _h1, w * _h1)
        useful_height = int(
            downsample_attn_mask[0, :,
                                 0].sum())  # Not optimized for D2H memcpy.
        useful_width = int(
            downsample_attn_mask[0,
                                 0, :].sum())  # Not optimized for D2H memcpy.
        sub_image_feature = sub_image_feature[:, :useful_height, :useful_width]
        sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
        num_image_tokens = image_attention_mask[
            batch_idx, :B_ + 1, 0::2, 0::2].sum().to(
                torch.int32) + (useful_height + 1) + _h1  # Not optimized.
        sub_image_feature = torch.cat([sub_image_feature, sub_GN],
                                      dim=2).reshape(1, -1, _fused_dim)

        # Concat global/sub image features.
        if self.hd_transform_order == 'glb_sub':
            output_imgs.append(
                torch.cat(
                    [global_image_feature, self.glb_GN, sub_image_feature],
                    dim=1))
        elif self.hd_transform_order == 'sub_glb':
            output_imgs.append(
                torch.cat(
                    [sub_image_feature, self.glb_GN, global_image_feature],
                    dim=1))
        else:
            raise NotImplementedError(
                f'hd_transform_order = {self.hd_transform_order}, not implemented'
            )
        output_len.append(num_image_tokens)

    # Project image features.
    img_set_tensor = []
    for _output_img in output_imgs:
        img_feature_proj = self.img_projection(
            _output_img.to(target_device).to(target_dtype))
        img_set_tensor.append(img_feature_proj)

    # Combine image embeddings with text embeddings.
    hidden_states = kwargs['wte'](input_ids)
    merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        new_hidden_states = hidden_states.index_put(
            indices=positions_tuple,
            values=merged_img_set_tensor,
            accumulate=False,
        )  # Not optimized for D2D memcpy.
    hidden_states = new_hidden_states

    if self.drop is not None:
        hidden_states = self.drop(hidden_states)

    return hidden_states


def siglip_embedding_forward(
        self, pixel_values: torch.FloatTensor,
        patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
    """Optimize SigLip Embedding forward.

    * Optimize by moving the _batch_nb_patches_h_inv and _batch_nb_patches_w_inv outside the loop.
    * Optimize by removing explicit D2H transfers.

    Ref code: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/vision_siglip_navit.py#L571
    """
    device = pixel_values.device
    batch_size = pixel_values.size(0)

    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose(1, 2)

    max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
    max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
    boundaries = torch.arange(1 / self.num_patches_per_side,
                              1.0,
                              1 / self.num_patches_per_side,
                              device=device)
    position_ids = torch.full(
        size=(
            batch_size,
            max_nb_patches_h * max_nb_patches_w,
        ),
        fill_value=0,
        device=device,
    )
    _batch_nb_patches_h_inv = 1.0 / patch_attention_mask[:, :, 0].sum(dim=1)
    _batch_nb_patches_w_inv = 1.0 / patch_attention_mask[:, 0, :].sum(dim=1)
    for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
        fractional_coords_h = torch.arange(0,
                                           1 - 1e-6,
                                           _batch_nb_patches_h_inv[batch_idx],
                                           device=device)
        fractional_coords_w = torch.arange(0,
                                           1 - 1e-6,
                                           _batch_nb_patches_w_inv[batch_idx],
                                           device=device)

        bucket_coords_h = torch.bucketize(
            fractional_coords_h, boundaries,
            right=True)  # Not optimized for D2H memcpy.
        bucket_coords_w = torch.bucketize(
            fractional_coords_w, boundaries,
            right=True)  # Not optimized for D2H memcpy.

        pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side +
                   bucket_coords_w).flatten()
        position_ids[batch_idx][p_attn_mask.view(
            -1)] = pos_ids  # Not optimized for D2H memcpy.

    embeddings = embeddings + self.position_embedding(position_ids)
    return embeddings


def dynamic_preprocess(
        self,
        image: ImageInput,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 384,
        mask_size: int = 27,
        return_image: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimize dynamic preprocess for image modality.

    * Support both PIL.Image.Image and torch.Tensor.
    * Optimize by removing explicit D2H transfers.

    Ref code: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py#L201
    """
    # Get target_width, target_height and target_aspect_ratio.
    orig_width, orig_height = get_image_size(image)
    w_crop_num = math.ceil(orig_width / float(image_size))
    h_crop_num = math.ceil(orig_height / float(image_size))
    if w_crop_num * h_crop_num > max_num:
        aspect_ratio = orig_width / orig_height
        target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                            for i in range(1, n + 1) for j in range(1, n + 1)
                            if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
    else:
        target_width = image_size * w_crop_num
        target_height = image_size * h_crop_num
        target_aspect_ratio = (w_crop_num, h_crop_num)

    # Generate attention mask.
    ratio_width = target_width / orig_width
    ratio_height = target_height / orig_height
    if ratio_width < ratio_height:
        new_size = (target_width, int(orig_height * ratio_width))
        padding_width = 0
        padding_height = target_height - int(orig_height * ratio_width)
    else:
        new_size = (int(orig_width * ratio_height), target_height)
        padding_width = target_width - int(orig_width * ratio_height)
        padding_height = 0
    attention_mask = torch.ones((int(mask_size * target_aspect_ratio[1]),
                                 int(mask_size * target_aspect_ratio[0])))
    if padding_width >= 14:
        attention_mask[:, -math.floor(padding_width / 14):] = 0
    if padding_height >= 14:
        attention_mask[-math.floor(padding_height / 14):, :] = 0
    if min(new_size[1], target_height) < 10 or min(new_size[0],
                                                   target_width) < 10:
        raise ValueError(
            f'The aspect ratio is very extreme {new_size} and not supported.')

    if return_image:
        image = resize(image, [new_size[1], new_size[0]])
        fill_values = [255, 255, 255] if is_pil_image(image) else 1.0
        resized_img = pad(image, [0, 0, padding_width, padding_height],
                          fill=fill_values)
    else:
        resized_img = None
    return resized_img, attention_mask


def _reshape_attention_masks(
        image_attention_masks: List[torch.Tensor],
        mask_resolution: int) -> Tuple[List[torch.Tensor], List[int]]:
    """Reshape attention mask and also return the number of image tokens."""
    mask_shapes = [[mask.size(0), mask.size(1)]
                   for mask in image_attention_masks]
    attention_masks_reshape = [
        rearrange(mask,
                  '(h rh) (w rw) -> (h w) rh rw',
                  h=h // mask_resolution,
                  w=w // mask_resolution,
                  rh=mask_resolution,
                  rw=mask_resolution)
        for mask, (h, w) in zip(image_attention_masks, mask_shapes)
    ]
    downsample_attention_masks = []
    for mask, (h, w) in zip(attention_masks_reshape, mask_shapes):
        mask = mask[:, 0::2, 0::2]
        h_stride = h // mask_resolution
        w_stride = w // mask_resolution
        h1 = mask_resolution // 2 + mask_resolution % 2
        mask = mask.reshape(1, h_stride, w_stride, h1, h1)
        mask = rearrange(mask,
                         '1 hs ws h1 w1 -> (hs h1) (ws w1)',
                         h1=h1,
                         w1=h1,
                         hs=h_stride,
                         ws=w_stride)
        downsample_attention_masks.append(mask)
    # 256: global image tokens with 16x16 patches.
    # 1: special token to aggregate information from all image patches.
    # int(mask.sum().item()): valid image tokens with real contents in sub images.
    # int(mask[:, 0].sum().item()): number of valid patches in the first column of the mask, to handle the vertical dimension of the image patches.
    # 16: padding value to ensure there's enough token space during vision encoding.
    num_img_tokens = [
        256 + 1 + int(mask.sum().item()) + int(mask[:, 0].sum().item()) + 16
        for mask in downsample_attention_masks
    ]
    return attention_masks_reshape, num_img_tokens


def image_preprocess(
    self,
    images: ImageInput,
    return_tensors: Optional[Union[str, TensorType]] = None,
):
    """Optimize preprocess for image modality.

    * Support both PIL.Image.Image and torch.Tensor.
    * Carve out the num_img_tokens calculation.

    Ref code: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py#L161
    """
    images = make_list_of_images(images)
    if not valid_images(images):
        raise TypeError(
            "Invalid image type. Must be of type PIL.Image.Image, torch.Tensor."
        )

    img_processor = [
        torchvision.transforms.ToTensor()
        if is_pil_image(images[0]) else lambda x: x,
        torchvision.transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
    base_resolution = _BASE_RESOLUTION
    if is_pil_image(images[0]):
        images = [image.convert('RGB') for image in images]
    mask_resolution = _MASK_RESOLUTION
    elems, image_attention_masks = [], []
    for im in images:
        elem, attention_mask = self.dynamic_preprocess(
            im,
            max_num=self.dynamic_hd,
            image_size=base_resolution,
            mask_size=mask_resolution,
            return_image=True)
        elems.append(elem)
        image_attention_masks.append(attention_mask)

    img_processor = torchvision.transforms.Compose(img_processor)
    hd_images = [img_processor(im) for im in elems]
    global_image = [
        torch.nn.functional.interpolate(
            im.unsqueeze(0).float(),
            size=(base_resolution, base_resolution),
            mode='bicubic',
        ).to(im.dtype) for im in hd_images
    ]
    shapes = [[im.size(1), im.size(2)] for im in hd_images]
    global_attention_mask = [
        torch.ones((1, mask_resolution, mask_resolution)) for _ in hd_images
    ]

    attention_masks_reshape, num_img_tokens = _reshape_attention_masks(
        image_attention_masks, mask_resolution)
    hd_images_reshape = [
        rearrange(im,
                  'c (h rh) (w rw) -> (h w) c rh rw',
                  h=h // base_resolution,
                  w=w // base_resolution,
                  rh=base_resolution,
                  rw=base_resolution) for im, (h, w) in zip(hd_images, shapes)
    ]
    hd_images_reshape = [
        torch.cat([_global_image] + [_im], dim=0)
        for _global_image, _im in zip(global_image, hd_images_reshape)
    ]
    hd_masks_reshape = [
        torch.cat([_global_mask] + [_mask],
                  dim=0) for _global_mask, _mask in zip(
                      global_attention_mask, attention_masks_reshape)
    ]
    max_crops = max([img.size(0) for img in hd_images_reshape])
    image_transformed = [
        self.pad_to_max_num_crops(im, max_crops) for im in hd_images_reshape
    ]
    image_transformed = torch.stack(image_transformed, dim=0)
    mask_transformed = [
        self.pad_mask_to_max_num_crops(mask, max_crops)
        for mask in hd_masks_reshape
    ]
    mask_transformed = torch.stack(mask_transformed, dim=0)

    returned_input_image_embeds = image_transformed
    returned_image_sizes = torch.tensor(shapes, dtype=torch.long)
    returned_image_attention_mask = mask_transformed
    returned_num_img_tokens = num_img_tokens

    data = {
        "input_image_embeds": returned_input_image_embeds,
        "image_sizes": returned_image_sizes,
        "image_attention_mask": returned_image_attention_mask,
        "num_img_tokens": returned_num_img_tokens,
    }

    return BatchFeature(data=data, tensor_type=return_tensors)


# Create a NoOp module to replace head layers in vision encoder.
class NoOp(torch.nn.Module):

    def forward(self, *args, **kwargs):
        return None


class InputMode(enum.Enum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3


class HFPhi4MultimodalEncoder(transformers.PreTrainedModel):

    config_class = Phi4MMConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config: transformers.PretrainedConfig, **kwargs):
        # Config for torch.compile
        if _is_torch_compile():
            # There some .item() calls in the original code, so we need to capture the scalar outputs
            # otherwise the graph will be broken.
            torch._dynamo.config.capture_scalar_outputs = True
            # Enable cudnn benchmark to get faster kernels and get better performance.
            torch.backends.cudnn.benchmark = True

        super().__init__(config, **kwargs)
        self.padding_idx = config.pad_token_id

        self.embed_tokens = torch.nn.Embedding(config.vocab_size,
                                               config.hidden_size,
                                               self.padding_idx)

        self._attn_implementation = config._attn_implementation

        self.vocab_size = config.vocab_size

        embedding_config = {
            'embedding_cls': config.embd_layer['embedding_cls'],
            **config.embd_layer
        }
        # The default values are from HuggingFace Phi-4-multimodal-instruct code.
        self.image_input_id = embedding_config.get('image_input_id', -1)
        self.audio_input_id = embedding_config.get('audio_input_id', -10000)
        if self.image_input_id == self.audio_input_id:
            raise ValueError(
                'image_input_id and audio_input_id should be different')

        self.image_embd_layer_kwargs = embedding_config['image_embd_layer']
        self.image_embed = Phi4MMImageEmbedding(config,
                                                **self.image_embd_layer_kwargs)
        # Bind optimized vision encoder forward.
        self.image_embed.forward = MethodType(vision_encoder_forward,
                                              self.image_embed)
        # Bind optimized siglip embedding forward.
        self.image_embed.img_processor.embeddings.forward = MethodType(
            siglip_embedding_forward,
            self.image_embed.img_processor.embeddings,
        )
        # Skip head layer in vision encoder to save runtime.
        self.image_embed.img_processor.head = NoOp()

        self.audio_embd_layer_kwargs = embedding_config['audio_embd_layer']
        self.audio_embed = Phi4MMAudioEmbedding(config,
                                                **self.audio_embd_layer_kwargs)

    @nvtx_range("[Encoder][Image] batch_infer_image_embeds")
    def _batch_infer_image_embeds(
            self, batched_input_ids: torch.Tensor,
            multimodal_params: List[MultimodalParams]) -> torch.Tensor:
        # Batch image inputs and attention mask with padding along dim=1 (patch num).
        input_image_embeds_list, input_image_attn_mask_list, input_image_sizes_list = [], [], []
        for mm_param in multimodal_params:
            mm_data = mm_param.multimodal_data
            input_image_embeds = mm_data["input_image_embeds"]
            if input_image_embeds is not None and input_image_embeds.numel(
            ) > 0:
                input_image_embeds_list.append(input_image_embeds)
                input_image_attn_mask_list.append(
                    mm_data["image_attention_mask"])
                input_image_sizes_list.append(mm_data["image_sizes"])
        batched_image_hidden_states = None
        if len(input_image_embeds_list) > 0:
            # Padding image embeds/attn_masks along dim=1 (patch dimension).
            b_list = [x.shape[0] for x in input_image_embeds_list]
            p_list = [x.shape[1] for x in input_image_embeds_list]
            c_i, h_i, w_i = input_image_embeds_list[0].shape[2:5]
            h_i_attn, w_i_attn = input_image_attn_mask_list[0].shape[2:4]
            total_b = sum(b_list)
            max_p = max(p_list)
            batched_image_embeds = torch.zeros(
                (total_b, max_p, c_i, h_i, w_i),
                dtype=input_image_embeds_list[0].dtype,
                device=input_image_embeds_list[0].device)
            batched_image_attn_mask = torch.zeros(
                (total_b, max_p, h_i_attn, w_i_attn),
                dtype=input_image_embeds_list[0].dtype,
                device=input_image_embeds_list[0].device)
            b_offset = 0
            for i, tensor in enumerate(input_image_embeds_list):
                b, p = tensor.shape[:2]
                batched_image_embeds[b_offset:b_offset + b, :p] = tensor
                if input_image_attn_mask_list[i] is not None:
                    batched_image_attn_mask[
                        b_offset:b_offset +
                        b, :p] = input_image_attn_mask_list[i]
                else:
                    batched_image_attn_mask[b_offset:b_offset + b, :p] = 1
                b_offset += b
            batched_image_sizes = torch.cat(input_image_sizes_list, dim=0)
            # Forward image encoder with batched image embeds.
            batched_image_hidden_states = self.image_embed(
                input_ids=batched_input_ids,
                input_embeds=batched_image_embeds,
                image_sizes=batched_image_sizes,
                image_attention_mask=batched_image_attn_mask,
                wte=self.embed_tokens,
            )
        return batched_image_hidden_states

    @nvtx_range("[Encoder][Audio] batch_infer_audio_embeds")
    def _batch_infer_audio_embeds(
            self, batched_input_ids: torch.Tensor,
            multimodal_params: List[MultimodalParams]) -> torch.Tensor:
        # Batch audio inputs and attention mask with padding along dim=1 (patch num).
        input_audio_embeds_list, input_audio_attn_mask_list, input_audio_sizes_list = [], [], []
        for mm_param in multimodal_params:
            mm_data = mm_param.multimodal_data
            input_audio_embeds = mm_data["input_audio_embeds"]
            if input_audio_embeds is not None and input_audio_embeds.numel(
            ) > 0:
                input_audio_embeds_list.append(input_audio_embeds)
                input_audio_attn_mask_list.append(
                    mm_data["audio_attention_mask"])
                input_audio_sizes_list.append(mm_data["audio_embed_sizes"])
        batched_audio_hidden_states = None
        if len(input_audio_embeds_list) > 0:
            b_list = [x.shape[0] for x in input_audio_embeds_list]
            p_list = [x.shape[1] for x in input_audio_embeds_list]
            d_a = input_audio_embeds_list[0].shape[2]
            total_b = sum(b_list)
            max_p = max(p_list)
            batched_audio_embeds = torch.zeros(
                (total_b, max_p, d_a),
                dtype=input_audio_embeds_list[0].dtype,
                device=input_audio_embeds_list[0].device)
            batched_audio_attn_mask = torch.zeros(
                (total_b, max_p),
                dtype=input_audio_embeds_list[0].dtype,
                device=input_audio_embeds_list[0].device)
            b_offset = 0
            for i, tensor in enumerate(input_audio_embeds_list):
                b, p = tensor.shape[:2]
                batched_audio_embeds[b_offset:b_offset + b, :p] = tensor
                if input_audio_attn_mask_list[i] is not None:
                    batched_audio_attn_mask[
                        b_offset:b_offset +
                        b, :p] = input_audio_attn_mask_list[i]
                else:
                    batched_audio_attn_mask[b_offset:b_offset + b, :p] = 1
                b_offset += b
            batched_audio_sizes = torch.cat(input_audio_sizes_list, dim=0)
            # Forward audio encoder with batched audio embeds.
            batched_audio_hidden_states = self.audio_embed(
                input_ids=batched_input_ids,
                input_embeds=batched_audio_embeds,
                audio_embed_sizes=batched_audio_sizes,
                audio_attention_mask=batched_audio_attn_mask,
                wte=self.embed_tokens,
            )
        return batched_audio_hidden_states

    @nvtx_range("[HFPhi4MultimodalEncoder] encoding_batch_request")
    def _encoding_batch_request(
            self, multimodal_params: List[MultimodalParams],
            mm_token_ids: torch.Tensor) -> List[torch.FloatTensor]:
        # Batch input_ids.
        input_ids_list = [
            multimodal_params[i].multimodal_data["input_ids"]
            for i in range(len(multimodal_params))
        ]
        max_input_ids_len = max(
            [input_ids.shape[1] for input_ids in input_ids_list])
        batched_input_ids = torch.full(
            (len(multimodal_params), max_input_ids_len),
            _PAD_TOKEN_ID,
            device=input_ids_list[0].device)
        for i, input_ids in enumerate(input_ids_list):
            batched_input_ids[i, :input_ids.shape[1]] = input_ids
        batched_input_ids = batched_input_ids.view(-1, max_input_ids_len)
        image_position_mask = batched_input_ids == _IMAGE_SPECIAL_TOKEN_ID
        non_image_position_mask = ~image_position_mask

        # Batch inference for image and audio embeds.
        batched_image_hidden_states = self._batch_infer_image_embeds(
            batched_input_ids, multimodal_params)
        batched_audio_hidden_states = self._batch_infer_audio_embeds(
            batched_input_ids, multimodal_params)

        # Combine different modalities into one.
        if batched_image_hidden_states is not None and batched_audio_hidden_states is not None:
            batched_hidden_states = batched_image_hidden_states * image_position_mask.unsqueeze(
                -1
            ) + batched_audio_hidden_states * non_image_position_mask.unsqueeze(
                -1)
        elif batched_image_hidden_states is not None:
            batched_hidden_states = batched_image_hidden_states
        elif batched_audio_hidden_states is not None:
            batched_hidden_states = batched_audio_hidden_states
        else:
            batched_hidden_states = self.embed_tokens(batched_input_ids)

        # Postprocessing to get multimodal-only embeddings.
        mm_token_mask = torch.isin(batched_input_ids, mm_token_ids)
        batched_hidden_states = batched_hidden_states[mm_token_mask]
        batched_hidden_states = [batched_hidden_states]
        return batched_hidden_states

    @torch.compile(disable=not _is_torch_compile())
    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams],
                mm_token_ids: torch.Tensor) -> List[torch.FloatTensor]:
        return self._encoding_batch_request(multimodal_params, mm_token_ids)


class Phi4MMInputProcessor(BaseMultimodalInputProcessor, InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: transformers.PretrainedConfig,
                 tokenizer: transformers.AutoTokenizer,
                 trust_remote_code: bool = True):
        if not trust_remote_code:
            raise ValueError("trust_remote_code must be True for Phi4MM")

        self.model_config = model_config
        self.device = 'cpu'

        self.tokenizer = tokenizer
        self.use_fast = True
        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast)

        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast)
        # Bind the optimized methods to the image processor instance
        self.processor.image_processor.dynamic_preprocess = MethodType(
            dynamic_preprocess,
            self.processor.image_processor,
        )
        self.processor.image_processor.preprocess = MethodType(
            image_preprocess,
            self.processor.image_processor,
        )

        self.dtype = model_config.torch_dtype

    def get_mm_token_ids(self) -> Optional[torch.Tensor]:
        return torch.tensor([_IMAGE_SPECIAL_TOKEN_ID, _AUDIO_SPECIAL_TOKEN_ID],
                            dtype=torch.int32,
                            device=self.device)

    def get_num_tokens_per_image(
        self,
        *,
        image: Image.Image,
        **kwargs,
    ):
        images = [image]
        # Dynamic HD
        base_resolution = _BASE_RESOLUTION
        mask_resolution = _MASK_RESOLUTION
        image_attention_masks = []
        for im in images:
            _, attention_mask = self.processor.image_processor.dynamic_preprocess(
                image=im,
                max_num=self.processor.image_processor.dynamic_hd,
                image_size=base_resolution,
                mask_size=mask_resolution,
                return_image=False,
            )
            image_attention_masks.append(attention_mask)
        _, num_img_tokens = _reshape_attention_masks(image_attention_masks,
                                                     mask_resolution)
        return num_img_tokens[0]

    def _post_process(self, image_inputs: BatchFeature,
                      audio_inputs: BatchFeature,
                      text_prompt: str) -> Dict[str, torch.Tensor]:
        # Combine image/audio/text embeddings.
        inputs = self.processor._convert_images_audios_text_to_inputs(
            image_inputs,
            audio_inputs,
            [text_prompt],
        )

        # Set input_mode and audio_projection_mode according to the modality.
        # Ref: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/modeling_phi4mm.py#L2103
        if len(image_inputs) > 0 and len(audio_inputs) > 0:
            input_mode = InputMode.VISION_SPEECH
            audio_projection_mode = 'vision'
        elif len(image_inputs) > 0:
            input_mode = InputMode.VISION
            audio_projection_mode = 'vision'
        elif len(audio_inputs) > 0:
            input_mode = InputMode.SPEECH
            audio_projection_mode = 'speech'
        else:
            input_mode = InputMode.LANGUAGE
            audio_projection_mode = 'speech'
        inputs["input_mode"] = torch.tensor([input_mode.value],
                                            dtype=torch.long)
        inputs["audio_projection_mode"] = audio_projection_mode

        # Inplace-replacement for special token ids.
        input_ids = inputs['input_ids']
        if len(image_inputs) > 0:
            torch.where(
                (input_ids >= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[0])
                & (input_ids <= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[1]),
                torch.tensor(_IMAGE_SPECIAL_TOKEN_ID),
                input_ids,
                out=input_ids,
            )
        if len(audio_inputs) > 0:
            torch.where(
                (input_ids >= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[0])
                & (input_ids <= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[1]),
                torch.tensor(_AUDIO_SPECIAL_TOKEN_ID),
                input_ids,
                out=input_ids,
            )
        inputs['input_ids'] = input_ids
        return inputs

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
        images = mm_data.get("image", None)
        audios = mm_data.get("audio", None)

        # Return ahead of time if no multimodal data.
        if images is None and audios is None:
            input_ids = self.tokenizer.encode(text_prompt,
                                              add_special_tokens=False,
                                              return_tensors="pt")
            return input_ids[0].to(torch.int32).tolist(), {}

        # Processing for multimodal data.
        image_inputs = self.processor.image_processor(
            images, return_tensors='pt') if images is not None else {}
        audio_inputs = self.processor.audio_processor(
            audios, return_tensors='pt') if audios is not None else {}

        # Postprocessing for multimodal data.
        inputs = self._post_process(image_inputs, audio_inputs, text_prompt)

        # Package inputs for language model forward in AGGREGATE mode.
        multimodal_data = {}
        multimodal_data['input_ids'] = inputs['input_ids']
        multimodal_data['input_image_embeds'] = inputs['input_image_embeds'].to(
            self.dtype)
        multimodal_data['image_sizes'] = inputs['image_sizes']
        multimodal_data['image_attention_mask'] = inputs['image_attention_mask']
        multimodal_data['input_audio_embeds'] = inputs['input_audio_embeds']
        multimodal_data['audio_embed_sizes'] = inputs['audio_embed_sizes']
        multimodal_data['audio_attention_mask'] = inputs['audio_attention_mask']
        multimodal_data['audio_projection_mode'] = inputs[
            'audio_projection_mode']
        return inputs['input_ids'][0].to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


@register_auto_model("Phi4MMForCausalLM")
@register_input_processor(
    Phi4MMInputProcessor,
    model_type="phi4mm",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|image_{0}|>",
            "audio": "<|audio_{0}|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        placeholders_separator="",
    ))
class Phi4MMForCausalLM(transformers.PreTrainedModel):

    _supports_flash_attn_2 = True

    def __init__(self, model_config: ModelConfig):
        if _is_disagg():
            raise ValueError(
                "Phi4MM does not support disaggregated inference yet.")

        config = model_config.pretrained_config
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        if not _is_disagg():
            _load_phi4mm_classes(config._name_or_path)

            self.hf_phi4mm_model = HFPhi4MultimodalEncoder(config).eval()
            self.hf_phi4mm_model.to(config.torch_dtype)
            # Required by HFPhi4MultimodalEncoder.
            self.phi4mm_wte = self.hf_phi4mm_model.embed_tokens

        # We use Phi3ForCausalLM as the language model.
        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config.architectures = ["Phi3ForCausalLM"]
        # Only build the language model architecture without loading weights.
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.vocab_size = config.vocab_size
        self.model_dtype = getattr(config, "torch_dtype", torch.float16)
        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):
        # Load weights into HFPhi4MultimodalEncoder.
        if not _is_disagg():
            filtered_weights = {}
            for k, v in weights.items():
                # Skip image_embed head weights since we set it as NoOp.
                if k.startswith(
                        "model.embed_tokens_extend.image_embed.img_processor.head."
                ):
                    continue
                if k.startswith("model.embed_tokens."):
                    new_k = k.replace("model.embed_tokens.", "embed_tokens.")
                    filtered_weights[new_k] = v
                elif k.startswith("model.embed_tokens_extend."):
                    new_k = k.replace("model.embed_tokens_extend.", "")
                    filtered_weights[new_k] = v
            self.hf_phi4mm_model.load_state_dict(filtered_weights, strict=True)

        # Filter out non-language model weights.
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith('model.embed_tokens_extend')
        }
        # Filter out LoRA weights.
        # LoRA weights will be loaded by LoraManager.
        weights = {k: v for k, v in weights.items() if '.lora_' not in k}
        # Rename base layer weights.
        updated_weights = {}
        base_layer_weight_names = [
            'weight', 'input_scale', 'weight_scale', 'weight_scale_2'
        ]
        for k in weights.keys():
            new_k = k
            for weight_name in base_layer_weight_names:
                if f'base_layer.{weight_name}' in k:
                    new_k = k.replace(f'base_layer.{weight_name}', weight_name)
                    break
            updated_weights[new_k] = weights[k]
        weights = updated_weights
        self.llm.load_weights(weights)

        # Move mm_token_ids to the correct device.
        self.mm_token_ids = torch.tensor(
            [_IMAGE_SPECIAL_TOKEN_ID, _AUDIO_SPECIAL_TOKEN_ID],
            device=self.device)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        # The data with the following keys will be moved to CUDA device,
        # the rest of them will be kept on CPU.
        return [
            "input_ids", "input_image_embeds", "image_attention_mask",
            "input_audio_embeds", "audio_attention_mask"
        ]

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
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(
            f"num_context_requests: {num_context_requests}, num_generation_requests: {num_generation_requests}"
        )

        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embedding = []
        if len(multimodal_params) > 0:
            if not _is_disagg():
                encoder_kwargs = {
                    "mm_token_ids": self.mm_token_ids,
                }
                mm_embedding = get_multimodal_embeddings(
                    encoder_forward_fn=self.hf_phi4mm_model.forward,
                    multimodal_params=multimodal_params[:num_context_requests],
                    encoder_kwargs=encoder_kwargs,
                )
            else:
                raise NotImplementedError(
                    "Phi-4-multimodal does not support disaggregated inference yet. Please unset "
                    f"the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )
            mm_embedding = find_input_mm_embeds(
                mm_embedding, multimodal_params[:num_context_requests])

        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embedding,
            mm_token_ids=self.mm_token_ids,
            **kwargs,
        )

        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            lora_params=kwargs.get("lora_params", None),
        )

        logger.debug(f'output shape: {output_prob.shape}')
        return output_prob

    @staticmethod
    def lora_config(model_dir: str):
        _lora_config = LoraConfig(
            lora_target_modules=[
                "attn_qkv",
                "attn_dense",
                "mlp_gate_up",
                "mlp_4h_to_h",
            ],
            trtllm_modules_to_hf_modules={
                "attn_qkv": "qkv_proj",
                "attn_dense": "o_proj",
                "mlp_gate_up": "gate_up_proj",
                "mlp_4h_to_h": "down_proj",
            },
            max_lora_rank=320,  # Max rank for Phi4MM.
            swap_gate_up_proj_lora_b_weight=
            False,  # Disable swap gate_up_proj.lora_B.weight for Phi4MM.
        )
        return _lora_config

    @staticmethod
    def lora_request(num_requests: int, modality: str, base_model_dir: str):
        # Prepare LoRA requests for different modalities.
        # Ref: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/modeling_phi4mm.py#L2103
        lora_request = None
        if modality == "image" or modality == "image_audio":
            lora_request = [
                LoRARequest(
                    lora_name="vision-lora",
                    lora_int_id=0,
                    lora_path=f"{base_model_dir}/vision-lora",
                ) for i in range(num_requests)
            ]
        elif modality == "audio":
            lora_request = [
                LoRARequest(
                    lora_name="speech-lora",
                    lora_int_id=1,
                    lora_path=f"{base_model_dir}/speech-lora",
                ) for i in range(num_requests)
            ]

        return lora_request
