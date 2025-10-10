import json
import os
import sys
from io import BytesIO

import requests

# isort: off
import torch
import numpy as np
# isort: on
import math
from typing import Optional, Tuple

import torch.nn.functional as F

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from safetensors import safe_open
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          AutoTokenizer)

from .. import profiler
from .._utils import (mpi_rank, str_dtype_to_torch, str_dtype_to_trt,
                      supports_inflight_batching, torch_dtype_to_trt,
                      trt_dtype_to_torch)
from ..functional import RopeEmbeddingUtils, RotaryScalingType
from ..layers import MropeParams
from ..logger import logger
from .enc_dec_model_runner import EncDecModelRunner
from .model_runner import ModelRunner
from .session import Session, TensorInfo

try:
    import tensorrt_llm.bindings  # NOQA
    PYTHON_BINDINGS = True
except ImportError:
    PYTHON_BINDINGS = False

if PYTHON_BINDINGS:
    from .model_runner_cpp import ModelRunnerCpp


class LlavaNextUtils:
    # https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py

    @staticmethod
    def select_best_resolution(original_size, possible_resolutions):
        """
            Selects the best resolution from a list of possible resolutions based on the original size.

            Args:
                original_size (tuple): The original size of the image in the format (width, height).
                possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

            Returns:
                tuple: The best fit resolution in the format (width, height).
            """
        original_width, original_height = original_size
        best_fit = None
        max_effective_resolution = 0
        min_wasted_resolution = float('inf')

        for width, height in possible_resolutions:
            scale = min(width / original_width, height / original_height)
            downscaled_width, downscaled_height = int(
                original_width * scale), int(original_height * scale)
            effective_resolution = min(downscaled_width * downscaled_height,
                                       original_width * original_height)
            wasted_resolution = (width * height) - effective_resolution

            if effective_resolution > max_effective_resolution or (
                    effective_resolution == max_effective_resolution
                    and wasted_resolution < min_wasted_resolution):
                max_effective_resolution = effective_resolution
                min_wasted_resolution = wasted_resolution
                best_fit = (width, height)

        return best_fit

    @staticmethod
    def get_anyres_image_grid_shape(image_size,
                                    patch_size,
                                    image_grid_pinpoints=None):
        """
            Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

            Args:
                image_size (tuple): The size of the input image in the format (width, height).
                patch_size (int): The size of each image patch.

            Returns:
                tuple: The shape of the image patch grid in the format (width, height).
            """
        if image_grid_pinpoints is None:
            image_grid_pinpoints = [[336, 672], [672, 336], [672, 672],
                                    [1008, 336], [336, 1008]]
        width, height = LlavaNextUtils.select_best_resolution(
            image_size, image_grid_pinpoints)
        return width // patch_size, height // patch_size

    @staticmethod
    def unpad_image(tensor, original_size):
        """
            Unpads a PyTorch tensor of a padded and resized image.

            Args:
            tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
            original_size (tuple): The original size of the image (width, height).

            Returns:
            torch.Tensor: The unpadded image tensor.
            """
        original_width, original_height = original_size
        current_height, current_width = tensor.shape[1:]

        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = current_width / original_width
            new_height = int(original_height * scale_factor)
            padding = (current_height - new_height) // 2
            unpadded_tensor = tensor[:, padding:current_height - padding, :]
        else:
            scale_factor = current_height / original_height
            new_width = int(original_width * scale_factor)
            padding = (current_width - new_width) // 2
            unpadded_tensor = tensor[:, :, padding:current_width - padding]

        return unpadded_tensor

    @staticmethod
    def rearrange_image_features(image_feature, image_newline, image_size):
        """
            Combine PyTorch feature grids from image patches.

            Args:
            image_feature (torch.Tensor): The feature grids, assumed to be in NxCxHxW format.
            image_newline (torch.Tensor): The newline embedding.
            image_size (tuple): Size of the original image (width, height).
            """
        CLIP_IMAGE_SIZE = 336
        CLIP_PATCH_SIZE = 14
        NUM_PATCHES_PER_SIDE = CLIP_IMAGE_SIZE // CLIP_PATCH_SIZE
        if image_feature.shape[0] == 1:
            return torch.cat((image_feature, image_newline[None]), dim=0)

        base_image_feature = image_feature[0]
        image_feature = image_feature[1:]
        height = width = NUM_PATCHES_PER_SIDE
        assert height * width == base_image_feature.shape[0]

        num_patch_width, num_patch_height = LlavaNextUtils.get_anyres_image_grid_shape(
            image_size, CLIP_IMAGE_SIZE)
        image_feature = image_feature.view(num_patch_height, num_patch_width,
                                           height, width, -1)

        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = LlavaNextUtils.unpad_image(image_feature, image_size)
        image_feature = torch.cat(
            (image_feature, image_newline[:, None, None].expand(
                *image_feature.shape[:-1], 1)),
            dim=-1)
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
        return image_feature


class LlavaOnevisionUtils:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/modeling_llava_onevision.py

    @staticmethod
    def pack_image_features(image_features, image_sizes, image_newline):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`torch.Tensor` of shape `(num_images, num_patches, image_length, embed_dim)`)
                Image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
        """

        IMAGE_SIZE = 384
        PATCH_SIZE = 14
        MAX_NUM_PATCHES = 9

        new_image_features = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = IMAGE_SIZE // PATCH_SIZE
                if height * width != base_image_feature.shape[0]:
                    raise ValueError(
                        "The number of patches is not consistent with the image size."
                    )

                IMAGE_GRID_PINPOINTS = [[384, 384], [384, 768], [384, 1152],
                                        [384, 1536], [384, 1920], [384, 2304],
                                        [768, 384], [768, 768], [768, 1152],
                                        [768, 1536], [768, 1920], [768, 2304],
                                        [1152, 384], [1152, 768], [1152, 1152],
                                        [1152, 1536],
                                        [1152, 1920], [1152, 2304], [1536, 384],
                                        [1536, 768], [1536, 1152], [1536, 1536],
                                        [1536, 1920], [1536, 2304], [1920, 384],
                                        [1920, 768], [1920, 1152], [1920, 1536],
                                        [1920, 1920], [1920, 2304], [2304, 384],
                                        [2304, 768], [2304, 1152], [2304, 1536],
                                        [2304, 1920], [2304, 2304]]
                num_patch_width, num_patch_height = LlavaNextUtils.get_anyres_image_grid_shape(
                    image_sizes[image_idx][[1, 0]].tolist(), IMAGE_SIZE,
                    IMAGE_GRID_PINPOINTS)
                image_feature = image_feature.view(num_patch_height,
                                                   num_patch_width, height,
                                                   width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1,
                                                      3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = LlavaNextUtils.unpad_image(
                    image_feature, image_sizes[image_idx][[1, 0]])

                channels, curr_height, curr_width = image_feature.shape
                ratio = math.sqrt(curr_height * curr_width /
                                  (MAX_NUM_PATCHES * height**2))
                if ratio > 1.1:
                    image_feature = image_feature[None]
                    image_feature = nn.functional.interpolate(
                        image_feature,
                        [int(curr_height // ratio),
                         int(curr_width // ratio)],
                        mode="bilinear")[0]

                image_feature = torch.cat(
                    (
                        image_feature,
                        image_newline[:, None, None].expand(
                            *image_feature.shape[:-1], 1).to(
                                image_feature.device, image_feature.dtype),
                    ),
                    dim=-1,
                )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature),
                                          dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat(
                        (image_feature, image_newline[None].to(image_feature)),
                        dim=0)
            new_image_features.append(image_feature)
        image_features = torch.stack(new_image_features)
        return image_features

    @staticmethod
    def apply_pooling(image_features):
        IMAGE_SIZE = 384
        PATCH_SIZE = 14
        height = width = IMAGE_SIZE // PATCH_SIZE
        batch_frames, seq_len, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2).contiguous()

        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / 2), math.ceil(width / 2)]
        image_features = nn.functional.interpolate(image_features,
                                                   size=scaled_shape,
                                                   mode="bilinear")

        image_features = image_features.permute(0, 2, 3, 1)
        image_features = image_features.view(batch_frames, -1, dim)
        return image_features


class PhiMMUtils:

    @staticmethod
    def add_image_newline(image_features, image_newline):
        h, w, d = image_features.shape
        image_newline = image_newline.expand(h, 1, -1)
        image_features_newline = torch.cat([image_features, image_newline],
                                           dim=1).flatten(0, 1)
        return image_features_newline

    @staticmethod
    def reshape_hd_patches(image_features, h_crop=1, w_crop=1):
        n_crops, n_tokens, d = image_features.shape
        assert n_crops == h_crop * w_crop

        h = w = int(n_tokens**0.5)
        image_features = image_features.reshape(
            h_crop, w_crop, h, w,
            d).permute(0, 2, 1, 3, 4).reshape(h_crop * h, w_crop * w, d)
        return image_features

    @staticmethod
    def hd_feature_transform(image_features,
                             h_crop,
                             w_crop,
                             sub_GN,
                             glb_GN,
                             patch_mask=None):
        glb_image_features = PhiMMUtils.add_image_newline(
            PhiMMUtils.reshape_hd_patches(image_features[:1]), sub_GN)

        num_crops = h_crop * w_crop
        sub_image_features = PhiMMUtils.reshape_hd_patches(
            image_features[1:num_crops + 1], h_crop, w_crop)
        if patch_mask is not None:
            h, w = patch_mask.shape[1] // 2, patch_mask.shape[2] // 2
            sub_image_mask = (patch_mask[1:num_crops + 1, 0::2,
                                         0::2].bool().reshape(
                                             h_crop, w_crop, h,
                                             w).permute(0, 2, 1, 3).reshape(
                                                 h_crop * h, w_crop * w))
            hh = int(sub_image_mask[:, 0].sum().item())
            ww = int(sub_image_mask[0, :].sum().item())
            sub_image_features = sub_image_features[:hh, :ww]
        sub_image_features = PhiMMUtils.add_image_newline(
            sub_image_features, sub_GN)

        image_features = torch.cat(
            [sub_image_features,
             glb_GN.expand(1, -1), glb_image_features])
        return image_features

    @staticmethod
    def reshape_audio_chunks(audio_features, chunk_mask=None):
        audio_features = audio_features.flatten(0, 1)
        if chunk_mask is not None:
            # only the last chunk may include paddings
            n_tokens = math.ceil(chunk_mask.flatten().sum().item() / 8)
            audio_features = audio_features[:n_tokens]
        return audio_features


class MultimodalModelRunner:

    def __init__(self, args):
        self.args = args
        self.use_trtllm_vision_engine = False

        self.runtime_rank = mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)

        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        if self.args.mm_embedding_offloading is None:
            self.args.mm_embedding_offloading = self.args.enable_chunked_context
        elif self.args.mm_embedding_offloading and not self.args.enable_chunked_context:
            logger.warning(
                "mm_embedding_offloading requires enable_chunked_context to be True. Setting mm_embedding_offloading to None."
            )
            self.args.mm_embedding_offloading = None

        # parse model type from visual engine config
        with open(os.path.join(self.visual_engine_dir, "config.json"),
                  "r") as f:
            config = json.load(f)
        if 'pretrained_config' in config:
            if config['pretrained_config'][
                    'architecture'] == 'LlavaNextForConditionalGeneration':
                self.model_type = 'llava_next'
                self.vision_precision = config['pretrained_config']['dtype']
                self.use_trtllm_vision_engine = True
            else:
                logger.error(
                    "Currently only Llava-NeXT supports TRT-LLM vision engines."
                )
        else:
            self.model_type = config['builder_config']['model_type']
            self.vision_precision = config['builder_config']['precision']
        if self.model_type == 'pix2struct':
            self.vision_precision = 'float16'
        self.decoder_llm = not (
            't5' in self.model_type
            or self.model_type in ['nougat', 'pix2struct']
        )  # BLIP2-T5, pix2struct and Nougat are using encoder-decoder models as LLMs

        if self.model_type == 'video-neva':
            self.num_frames = config['builder_config'].get('num_frames', None)
        if self.model_type == "llava_next":
            self.llm_name = AutoConfig.from_pretrained(
                self.args.hf_model_dir).text_config._name_or_path
        if 'internlm' in self.model_type:
            self.args.lora_task_uids = ['0'] * args.batch_size
        if self.model_type == "qwen2_vl":
            hf_config = AutoConfig.from_pretrained(self.args.hf_model_dir)
            self.vision_start_token_id = hf_config.vision_start_token_id
            self.vision_end_token_id = hf_config.vision_end_token_id
            self.vision_token_id = hf_config.vision_token_id
            self.image_token_id = hf_config.image_token_id
            self.video_token_id = hf_config.video_token_id
            self.spatial_merge_size = hf_config.vision_config.spatial_merge_size
            self.max_position_embeddings = hf_config.max_position_embeddings
            self.hidden_size = hf_config.hidden_size
            self.num_attention_heads = hf_config.num_attention_heads
            self.rope_theta = hf_config.rope_theta
        if self.model_type == 'llava_onevision':
            self.num_frames = self.args.video_num_frames
            if self.num_frames is None:
                self.num_frames = 8
            assert self.args.video_path is None or self.args.image_path is None
        if self.model_type == "pixtral":
            hf_config = AutoConfig.from_pretrained(self.args.hf_model_dir)
            self.image_size = hf_config.vision_config.image_size
            self.patch_size = hf_config.vision_config.patch_size
            self.vocab_size = hf_config.text_config.vocab_size
            self.image_token_index = hf_config.image_token_index
            self.spatial_merge_size = hf_config.spatial_merge_size

        self.audio_input_names = self.audio_output_names = None
        if self.model_type == "mllama":
            self.vision_input_names = [
                "pixel_values",
                "aspect_ratio_ids",
                "aspect_ratio_mask",
            ]
            self.vision_output_names = [
                "encoder_output",
            ]
        elif self.model_type == "llava_next" and self.use_trtllm_vision_engine:
            self.vision_input_names = [
                "pixel_values",
            ]
            self.vision_output_names = [
                "image_features",
            ]
        elif self.model_type == "phi-4-multimodal":
            self.vision_input_names = ["input", "attention_mask"]
            self.audio_input_names = ["input", "attention_mask"]
            self.audio_output_names = ["encoder_output"]
            self.vision_output_names = ["encoder_output"]
        else:
            self.vision_input_names = ["input"]
            self.vision_output_names = ["encoder_output"]

        self.session = args.session
        if self.cpp_e2e:
            self.visual_output_shape = config['builder_config'].get(
                'output_shape', None)
        if self.decoder_llm:
            if not supports_inflight_batching(self.llm_engine_dir):
                logger.warning(
                    "The given engine does not support in-flight batching, both visual engine and LLM fallback to python session"
                )
                self.session = 'python'

            if not PYTHON_BINDINGS and 'cpp' in args.session:
                logger.warning(
                    "Python bindings of C++ session is unavailable, both visual engine and LLM fallback to Python session."
                )
                self.session = 'python'

            args.debug_mode = False
            if args.debug_mode and 'cpp' in args.session:
                logger.warning(
                    "Debug mode is not supported in C++ session for now, both visual engine and LLM fallback to Python session."
                )
                self.session = 'python'

            if self.model_type == 'qwen2_vl':
                if self.args.session != "cpp_llm_only":
                    logger.warning(
                        "Qwen2-vl only support C++ session for now, fallback to C++ session."
                    )
                    self.args.session = "cpp_llm_only"

            if (not (self.model_type in
                     ('llava', 'vila', 'blip2-opt', 'kosmos-2', 'fuyu',
                      'cogvlm', 'neva', "internvl") or 'internlm'
                     in self.model_type)) and args.session == 'cpp':
                logger.warning(
                    f'C++ end-to-end mode does not support {self.model_type}. Visual engine fallbacks to Python session. See support matrix in README.'
                )
                args.session = 'cpp_llm_only'
            self.session = args.session

        else:
            self.session = 'cpp_llm_only'

        self.init_tokenizer()
        self.init_processor()
        self.init_image_encoder()
        self.init_llm()

        if self.audio_input_names is not None:
            with open(os.path.join(self.audio_engine_dir, "config.json"),
                      "r") as f:
                config = json.load(f)
            self.audio_precision = config['builder_config']['precision']
            self.init_audio_encoder()
        else:
            self.audio_encoder_session = self.audio_precision = None

    @property
    def cpp_e2e(self):
        return self.session == 'cpp'

    @property
    def cpp_llm_only(self):
        return self.session == 'cpp_llm_only'

    @property
    def python_e2e(self):
        return self.session == 'python'

    @property
    def visual_engine_dir(self):
        return os.path.join(self.args.engine_dir, 'vision')

    @property
    def audio_engine_dir(self):
        return os.path.join(self.args.engine_dir, 'audio')

    @property
    def llm_engine_dir(self):
        return os.path.join(self.args.engine_dir, 'llm')

    def init_tokenizer(self):
        if self.model_type == 'nougat':
            from transformers import NougatTokenizerFast
            self.tokenizer = NougatTokenizerFast.from_pretrained(
                self.args.hf_model_dir)
        elif self.model_type == 'neva' or self.model_type == 'video-neva':
            from sentencepiece import SentencePieceProcessor

            sp = SentencePieceProcessor(
                os.path.join(self.args.hf_model_dir, 'tokenizer.model'))

            class return_obj:

                def __init__(self, input_ids):
                    self.input_ids = input_ids

                def __getitem__(self, name):
                    if name in "input_ids":
                        return self.input_ids
                    else:
                        raise AttributeError(
                            f"'return_obj' has no item '{name}'")

            # sentencepiece does not follow the same interface as HF
            class HFTokenizerInterface():

                def encode(self, x, return_tensors=None, **kwargs):
                    out = sp.encode(x)
                    if return_tensors == "pt":
                        out = torch.tensor(out)
                    return return_obj(out)

                def __call__(self, x, return_tensors=None, **kwargs):
                    return self.encode(x, return_tensors, **kwargs)

                def decode(self, x, **kwargs):
                    return sp.decode(x.tolist())

                def batch_decode(self, x, **kwargs):
                    return self.decode(x, **kwargs)

            self.tokenizer = HFTokenizerInterface()
            self.tokenizer.eos_token_id = sp.eos_id()
            self.tokenizer.bos_token_id = sp.bos_id()
            self.tokenizer.pad_token_id = sp.pad_id()
        elif self.model_type == 'vila':
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.hf_model_dir + "/llm",
                use_fast=False,
                use_legacy=False)
        else:
            use_fast = self.model_type in [
                "phi-3-vision", "phi-4-multimodal", "internvl"
            ]
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.hf_model_dir,
                use_fast=use_fast,
                use_legacy=False,
                trust_remote_code=True)

        self.tokenizer.padding_side = "right"

    def init_processor(self):
        from torchvision import transforms

        if 'blip2' in self.model_type:
            from transformers import Blip2Processor
            self.processor = Blip2Processor.from_pretrained(
                self.args.hf_model_dir)

        elif 'nougat' in self.model_type:
            from transformers import NougatProcessor
            self.processor = NougatProcessor.from_pretrained(
                self.args.hf_model_dir)

        elif 'cogvlm' in self.model_type:
            image_size = 490
            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
                transforms.ConvertImageDtype(torch.bfloat16),
            ])

        elif self.model_type in [
                'phi-3-vision', 'pix2struct', 'llava_next', 'llava', 'fuyu',
                'kosmos-2', 'mllama', 'llava_onevision', 'qwen2_vl',
                'phi-4-multimodal'
        ]:
            self.processor = AutoProcessor.from_pretrained(
                self.args.hf_model_dir, trust_remote_code=True, num_crops=16)

        elif 'pixtral' in self.model_type:
            self.processor = AutoProcessor.from_pretrained(
                self.args.hf_model_dir)

        elif 'internlm' in self.model_type:
            image_size = 490
            self.processor = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])

        elif 'internvl' in self.model_type:
            from transformers import CLIPImageProcessor
            self.processor = CLIPImageProcessor.from_pretrained(
                'OpenGVLab/InternViT-300M-448px'
            )  # You can change the InternViT model type according to your InternVL type

        elif self.model_type == "neva":
            image_size = 384
            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.ConvertImageDtype(torch.float32),
            ])

        elif self.model_type == "video-neva":
            pass

        elif self.model_type == "vila":
            sys.path.append(self.args.hf_model_dir + "/../VILA")
            from llava.mm_utils import process_images
            from llava.model import LlavaLlamaConfig  # noqa
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                self.args.hf_model_dir,
                device_map='auto',
                trust_remote_code=True,
            )
            vision_tower = model.get_vision_tower()
            vision_tower.image_processor

            def processor(raw_image):
                return process_images(raw_image, vision_tower.image_processor,
                                      model.config).to(model.device,
                                                       dtype=torch.float16)

            self.processor = processor

        if self.model_type == 'mllama':
            from .processor_wrapper import MllamaProcessorWrapper
            self.processor = MllamaProcessorWrapper(self.processor, logger)

    def init_image_encoder(self):

        # Phi-4-multimodal uses pytorch engine due to issues with creating TRT engine.
        if self.model_type == "phi-4-multimodal":
            model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_dir,
                                                         dtype=torch.float16,
                                                         trust_remote_code=True,
                                                         device_map='cpu')
            self.vision_model = model.model.embed_tokens_extend.image_embed.to(
                self.device).eval()
            self.image_newlines = {}
            self.image_newlines['sub_GN'] = self.vision_model.img_projection(
                self.vision_model.sub_GN).squeeze()
            self.image_newlines['glb_GN'] = self.vision_model.img_projection(
                self.vision_model.glb_GN).squeeze()
            return

        if self.model_type == "phi-3-vision":
            model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_dir,
                                                         dtype=torch.float16,
                                                         trust_remote_code=True,
                                                         device_map='cpu')
            self.vision_model = model.model.vision_embed_tokens.to(
                self.device).eval()

            # Test run vision_model.get_img_features to pre-allocate memory for flash attention
            image = self.processor(text="<|image_1|>",
                                   images=Image.new('RGB', [10, 10]),
                                   return_tensors="pt")['pixel_values']
            image = image.flatten(0, 1)
            image = torch.rand(image.shape,
                               dtype=str_dtype_to_torch(self.vision_precision),
                               device=self.device)
            self.vision_model.get_img_features(image)
            return

        if self.cpp_e2e:
            logger.info(
                "Using C++ runtime for both visual engine and LLM decoder, skip loading visual engine in Python runtime."
            )
        elif self.model_type == "llava_next" and self.use_trtllm_vision_engine:
            cudart.cudaSetDevice(self.runtime_rank % torch.cuda.device_count())

            vision_encoder_path = os.path.join(
                self.visual_engine_dir, f"rank{self.runtime_rank}.engine")
            logger.info(f'Loading engine from {vision_encoder_path}')
            with open(vision_encoder_path, "rb") as f:
                engine_buffer = f.read()
            logger.info(f'Creating session from engine {vision_encoder_path}')
            assert engine_buffer is not None

            self.visual_encoder_session = Session.from_serialized_engine(
                engine_buffer)
        else:
            vision_encoder_path = os.path.join(self.visual_engine_dir,
                                               self.args.visual_engine_name)
            logger.info(f'Loading engine from {vision_encoder_path}')
            with open(vision_encoder_path, 'rb') as f:
                engine_buffer = f.read()
            logger.info(f'Creating session from engine {vision_encoder_path}')
            self.visual_encoder_session = Session.from_serialized_engine(
                engine_buffer)
        if self.model_type in ["llava_next", "llava_onevision"]:
            self.image_newlines = {}
            image_newlines_path = os.path.join(self.visual_engine_dir,
                                               'image_newlines.safetensors')
            with safe_open(image_newlines_path,
                           framework="pt",
                           device=self.device) as f:
                for k in f.keys():
                    self.image_newlines[k] = f.get_tensor(k)

    def init_audio_encoder(self):
        assert self.model_type == "phi-4-multimodal"
        model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_dir,
                                                     dtype=torch.float16,
                                                     trust_remote_code=True,
                                                     device_map='cpu')
        self.audio_model = model.model.embed_tokens_extend.audio_embed.to(
            self.device).eval()

    def init_llm(self):
        if self.decoder_llm:
            cross_kv_cache_fraction = None
            if self.model_type == 'mllama':
                cross_kv_cache_fraction = self.args.cross_kv_cache_fraction
            if self.python_e2e:
                logger.info(f'Running LLM with Python runner')
                self.model = ModelRunner.from_dir(
                    self.llm_engine_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=False,
                    stream=self.stream,
                    enable_context_fmha_fp32_acc=self.args.
                    enable_context_fmha_fp32_acc,
                    multi_block_mode=self.args.multi_block_mode,
                )
                self.model_config = self.model.session._model_config
            elif self.cpp_e2e:
                logger.info(
                    f'Running both visual engine and LLM with Python runner')
                self.model = ModelRunnerCpp.from_dir(
                    self.args.engine_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=False,
                    is_enc_dec=True,  # TODO: add a separate model variant here?
                    enable_context_fmha_fp32_acc=self.args.
                    enable_context_fmha_fp32_acc)
                self.model_config = self.model.model_config
            else:
                logger.info(f'Running LLM with C++ runner')
                self.model = ModelRunnerCpp.from_dir(
                    self.llm_engine_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=False,
                    enable_chunked_context=self.args.enable_chunked_context,
                    enable_context_fmha_fp32_acc=self.args.
                    enable_context_fmha_fp32_acc,
                    kv_cache_free_gpu_memory_fraction=self.args.
                    kv_cache_free_gpu_memory_fraction,
                    cross_kv_cache_fraction=cross_kv_cache_fraction,
                    multi_block_mode=self.args.multi_block_mode,
                    mm_embedding_offloading=self.args.mm_embedding_offloading,
                )
                self.model_config = self.model.model_config
            self.runtime_mapping = self.model.mapping
        else:
            self.model = EncDecModelRunner.from_engine(
                os.path.basename(self.args.hf_model_dir),
                self.llm_engine_dir,
                skip_encoder=self.model_type in ['nougat', 'pix2struct'],
                debug_mode=False,
                stream=self.stream,
                enable_context_fmha_fp32_acc=self.args.
                enable_context_fmha_fp32_acc)
            if self.model_type in ['nougat', 'pix2struct']:
                self.model_config = self.model.decoder_model_config
                self.runtime_mapping = self.model.decoder_runtime_mapping
            else:
                self.model_config = self.model.encoder_model_config
                self.runtime_mapping = self.model.encoder_runtime_mapping

    def video_preprocess(self, video_path):
        from decord import VideoReader
        if isinstance(video_path, str):
            vr = VideoReader(video_path)
            num_frames = self.num_frames
            if num_frames == -1:
                frames = [
                    Image.fromarray(frame.asnumpy()[:, :, ::-1]).convert('RGB')
                    for frame in vr
                ]
            else:
                # equally sliced frames into self.num_frames frames
                # if self.num_frames is greater than the number of frames in the video, we will repeat the last frame
                num_frames = min(num_frames, len(vr))
                indices = np.linspace(0, len(vr) - 1, num=num_frames, dtype=int)
                frames = [
                    Image.fromarray(
                        vr[idx].asnumpy()[:, :, ::-1]).convert('RGB')
                    for idx in indices
                ]
                if len(frames) < num_frames:
                    frames += [frames[-1]] * (num_frames - len(frames))
        else:
            frames = self.video_path

        from transformers import CLIPImageProcessor
        processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", dtype=torch.bfloat16)
        frames = processor.preprocess(frames,
                                      return_tensors="pt")['pixel_values']
        # make dtype consistent with vision encoder
        media_tensors = frames.to(str_dtype_to_torch(
            self.vision_precision))  # [num_frames, 3, H, W]
        return media_tensors.unsqueeze(0)  #[1, num_frames, 3, H, W]

    def preprocess(self, pre_prompt, post_prompt, image, other_vision_inputs,
                   other_audio_inputs):
        audio = None
        # same prompt for single/multiple image(s)
        n_prompts_n_images = False
        if isinstance(post_prompt,
                      list) and len(post_prompt) > 1 and image is not None:
            if hasattr(image, "pixel_values"):
                if len(post_prompt) == image["pixel_values"].shape[0]:
                    n_prompts_n_images = True
                    # n prompts and n images
            else:
                if isinstance(
                        image,
                        torch.Tensor) and len(post_prompt) == image.shape[0]:
                    n_prompts_n_images = True
                    # n prompts and n images

        if self.model_type == 'kosmos-2':
            input_ids = image['input_ids'].clone()
            image_mask = image["image_embeds_position_mask"]
            image = image['pixel_values']
            input_ids += image_mask * (self.model_config.vocab_size - 4)
            input_ids = input_ids.expand(self.args.batch_size,
                                         *input_ids.shape[1:])
            length = input_ids.shape[1]
        elif self.model_type == 'phi-3-vision':
            input = image
            image = input['pixel_values']
            image = image.flatten(0, 1)
        elif self.model_type == 'phi-4-multimodal':
            input = image
            image = input['input_image_embeds'].flatten(0, 1)
            other_vision_inputs['attention_mask'] = input[
                'image_attention_mask'].flatten(0, 1).bool()

            audio = input['input_audio_embeds']
            l, d = audio.shape[1], audio.shape[2]
            pad = 4000 - l % 4000
            audio = torch.cat([audio, audio.new_zeros(1, pad, d)],
                              dim=1).reshape(-1, 4000, d)
            audio_mask = audio.new_ones(*audio.shape[:2])
            audio_mask[-1, -pad:] = 0
            other_audio_inputs['attention_mask'] = audio_mask.bool()
        elif self.model_type == 'pixtral':
            # Hold on to pixel_values and input_ids.
            dtype = str_dtype_to_torch(self.vision_precision)
            # Shape of pixel values from the processor varies with the raw image.
            # So we create a new tensor with a fixed shape as expected by the vision
            # encoder and create a corresponding attention mask.
            image_size = self.image_size
            patch_size = self.patch_size
            d_min = torch.finfo(dtype).min
            num_patches = (image_size // patch_size)
            padded_image = torch.full(
                (self.args.batch_size, 3, image_size, image_size),
                fill_value=0,
                dtype=dtype,
                device="cuda")
            padded_attention_mask = torch.full(
                (self.args.batch_size, num_patches, num_patches),
                fill_value=d_min,
                dtype=dtype,
                device="cuda")
            h, w, input_ids = [], [], []
            for img_idx in range(self.args.batch_size):
                pixel_values = image["pixel_values"][img_idx]
                img_h, img_w = pixel_values.shape[-2:]
                padded_image[img_idx, :, :img_h, :img_w] = pixel_values
                padded_attention_mask[img_idx, :img_h // patch_size, :img_w //
                                      patch_size] = 0
                input_ids.append(image["input_ids"][img_idx])
                h.append(img_h)
                w.append(img_w)

            image = padded_image
            other_vision_inputs = {
                "attention_mask": padded_attention_mask,
            }
        elif self.model_type == 'llava_next':
            input = image
            image = input['pixel_values']
            image = image[0]
            image_size = input['image_sizes'][0].cpu()
        elif self.model_type == "qwen2_vl":
            input = image
            image = input['image']
            input_ids = input['input_ids']
            other_vision_inputs['image_grid_thw'].shape[0]
            attention_mask = other_vision_inputs['attention_mask_llm']
            other_vision_inputs.pop('attention_mask_llm')
            image_grid_thw = other_vision_inputs['image_grid_thw']
            other_vision_inputs.pop('image_grid_thw')
        elif self.model_type == 'llava_onevision':
            input = image
            if self.args.video_path is None:
                image = input['pixel_values']
                image = image[0].repeat(self.args.batch_size, 1, 1, 1)
                image_size = input['image_sizes'][0]
                image_size = image_size.repeat(self.args.batch_size, 1).cpu()
            else:
                image = input['pixel_values_videos']
                _, _, c, h, w = image.shape
                image = image.repeat(self.args.batch_size, 1, 1, 1, 1)
                image = image.view(-1, c, h, w)
        elif self.model_type == "fuyu":
            while len(image["image_patches"]) < self.args.batch_size:
                image["image_patches"].append(image["image_patches"][0])

        profiler.start("Vision encoder")
        visual_features, visual_atts, model_runner_input = None, None, None
        if image is not None:
            model_runner_input = torch.stack(
                image['image_patches'],
                dim=0) if self.model_type == 'fuyu' else image

            if self.model_type == "phi-3-vision":
                visual_features = self.vision_model.get_img_features(
                    image).reshape(1, image.shape[0], -1,
                                   self.vision_model.image_dim_out)
                visual_atts = None
            elif self.model_type == "phi-4-multimodal":
                visual_features = self.vision_model.get_img_features(
                    model_runner_input.to(
                        str_dtype_to_torch(self.vision_precision)),
                    other_vision_inputs['attention_mask'])
                visual_features = self.vision_model.img_projection(
                    visual_features)
                visual_atts = torch.ones(visual_features.size()[:-1],
                                         dtype=torch.long).to(
                                             model_runner_input.device)
            else:
                if self.cpp_e2e:
                    # If using E2E C++ runtime, visual_features will not be computed here in Python runtime.
                    # Instead, it only contains a shape read from the engine config, and is used for generating
                    # decoder prompt later
                    logger.info(
                        'Skip running visual engine, get visual output shape from engine config.'
                    )
                    model_runner_input = model_runner_input.to(
                        str_dtype_to_torch(self.vision_precision))
                    batch_size = model_runner_input.shape[0]
                    output_shape = list(self.visual_output_shape)
                    output_shape[0] = batch_size
                    if self.model_type == 'fuyu':
                        output_shape[1] = model_runner_input.shape[
                            2]  # fuyu's output patch number is not fixed, same as input patch number
                    visual_features = TensorInfo(
                        'encoder_output',
                        str_dtype_to_trt(self.vision_precision),
                        tuple(output_shape))
                    atts_shape = visual_features.shape[:-1]
                    visual_atts = TensorInfo('image_atts', None,
                                             tuple(atts_shape))
                    model_runner_input = torch.vsplit(
                        model_runner_input, model_runner_input.shape[0])
                else:
                    visual_features, visual_atts = self.get_visual_features(
                        model_runner_input, other_vision_inputs)
                    model_runner_input = None
        profiler.stop("Vision encoder")

        profiler.start("Audio encoder")
        audio_features = None
        if audio is not None:
            audio_features = self.get_audio_features(audio, other_audio_inputs)
        profiler.stop("Audio encoder")

        if self.model_type == 'fuyu':
            input_ids = image['input_ids'].to(torch.int32)
            image_patches_indices = image['image_patches_indices'].to(
                torch.int32)

            input_ids = input_ids.expand(self.args.batch_size,
                                         *input_ids.shape[1:])
            image_patches_indices = image_patches_indices.expand(
                self.args.batch_size, *image_patches_indices.shape[1:])

            input_ids = self.ptuning_setup_fuyu(input_ids,
                                                image_patches_indices)
            input_ids = torch.stack(input_ids, dim=0).to('cpu')
            length = input_ids.shape[1]
            if not self.cpp_e2e:  # TODO: bs > 1 for C++ E2E Fuyu
                visual_features = visual_features.repeat(
                    self.args.batch_size, 1, 1)
        elif self.model_type == 'qwen2_vl':
            length = input_ids.shape[1]
            input_lengths = torch.IntTensor([length] * self.args.batch_size).to(
                torch.int32)
            input_ids, ptuning_args, mrope_args = self.setup_fake_prompts_qwen2vl(
                visual_features, input_ids, image_grid_thw, attention_mask,
                input_lengths)
            return input_ids, input_lengths, ptuning_args, visual_features, mrope_args

        elif self.model_type == 'kosmos-2':
            visual_features = visual_features.squeeze(
            ) if visual_features is not None else None
        elif self.model_type == 'vila':
            if n_prompts_n_images:
                input_ids = self.tokenizer_image_token(self.args.batch_size,
                                                       pre_prompt[0],
                                                       post_prompt,
                                                       self.tokenizer)
            else:
                input_ids = self.tokenizer_image_token(self.args.batch_size,
                                                       pre_prompt[0],
                                                       post_prompt[0],
                                                       self.tokenizer)
            batch_split_prompts = self.split_prompt_by_images(input_ids)
            if not n_prompts_n_images:
                first_batch_split_prompts = batch_split_prompts[0]
                # compute prompt length + visual length
                length = sum(
                    [ids.shape[1] for ids in first_batch_split_prompts])
                if self.args.batch_size == 1 and len(image) > 1:
                    # mode 1: multiple image as a whole, flatten visual dims
                    length += visual_atts.shape[0] * visual_atts.shape[1]
                else:
                    length += visual_atts.shape[1]
                input_lengths = torch.IntTensor(
                    [length] * self.args.batch_size).to(torch.int32)
                input_ids, ptuning_args = self.setup_fake_prompts_vila(
                    self.args.batch_size, visual_features,
                    first_batch_split_prompts, input_lengths)
            else:
                # mode 2: multiple different prompts corresponding to multiple images (1-1 correspondence)
                length = [
                    sum([ids.shape[1] for ids in batch_split_prompt])
                    for batch_split_prompt in batch_split_prompts
                ]
                length = [l + visual_atts.shape[1] for l in length]
                input_lengths = torch.IntTensor(length).to(torch.int32)
                input_ids, ptuning_args = self.setup_fake_prompts_vila(
                    self.args.batch_size, visual_features, batch_split_prompts,
                    input_lengths)
            return input_ids, input_lengths, ptuning_args, visual_features, model_runner_input
        elif self.model_type == 'phi-3-vision':
            image_sizes = input["image_sizes"]
            profiler.start("Feature transform")
            visual_features = self.vision_model.hd_feature_transform(
                visual_features, image_sizes)
            profiler.stop("Feature transform")
            input_ids = input["input_ids"].clone()
            input_ids = input_ids.expand(self.args.batch_size,
                                         *input_ids.shape[1:])
            num_img_tokens = [visual_features.shape[0]]
            input_ids = self.ptuning_setup_phi3(visual_features=visual_features,
                                                audio_features=None,
                                                input_ids=input_ids,
                                                num_img_tokens=num_img_tokens,
                                                num_aud_tokens=None)
            visual_features = visual_features.unsqueeze(0).repeat(
                self.args.batch_size, 1, 1)
            length = input_ids.shape[1]
        elif self.model_type == 'phi-4-multimodal':
            h, w = input["image_sizes"][0]
            image_attention_mask = input.get("image_attention_mask")
            if image_attention_mask is not None:
                image_attention_mask = image_attention_mask[0].bool()
            patch_size = 336 if self.model_type == 'phi-3-vision' else 448
            profiler.start("Feature transform")
            visual_features = PhiMMUtils.hd_feature_transform(
                visual_features,
                h // patch_size,
                w // patch_size,
                self.image_newlines["sub_GN"],
                self.image_newlines["glb_GN"],
                patch_mask=image_attention_mask)
            profiler.stop("Feature transform")
            input_ids = input["input_ids"].clone()
            input_ids = input_ids.expand(self.args.batch_size,
                                         *input_ids.shape[1:])
            num_img_tokens = [visual_features.shape[0]]
            if audio_features is not None:
                dim = audio_features.shape[-1] // 2
                audio_features = audio_features[..., -dim:]
                audio_features = PhiMMUtils.reshape_audio_chunks(
                    audio_features, other_audio_inputs["attention_mask"])
                num_aud_tokens = [audio_features.shape[0]]
            else:
                num_aud_tokens = None
            input_ids = self.ptuning_setup_phi3(visual_features=visual_features,
                                                audio_features=audio_features,
                                                input_ids=input_ids,
                                                num_img_tokens=num_img_tokens,
                                                num_aud_tokens=num_aud_tokens)
            visual_features = visual_features.unsqueeze(0).repeat(
                self.args.batch_size, 1, 1)
            if audio_features is not None:
                audio_features = audio_features.unsqueeze(0).repeat(
                    self.args.batch_size, 1, 1)
            length = input_ids.shape[1]

        elif self.model_type == 'pixtral':
            relevant_patch_size = self.patch_size * self.spatial_merge_size
            output_img_size = self.image_size // relevant_patch_size
            # Note: max_h * max_w shall serve as the `tokens_per_task` in ptuning prompt table.
            max_h = max(h) // relevant_patch_size
            max_w = max(w) // relevant_patch_size
            visual_embed_dim = visual_features.shape[-1]
            relevant_visual_features = torch.zeros(self.args.batch_size,
                                                   max_h * max_w,
                                                   visual_embed_dim)
            for img_idx in range(self.args.batch_size):
                complete_features = visual_features[img_idx]
                complete_features = complete_features.reshape(
                    output_img_size, output_img_size, visual_embed_dim)
                relevant_h = h[img_idx] // relevant_patch_size
                relevant_w = w[img_idx] // relevant_patch_size
                flattened_features = complete_features[:relevant_h, :
                                                       relevant_w, :].flatten(
                                                           0, 1)
                relevant_visual_features[img_idx, :relevant_h *
                                         relevant_w, :] = flattened_features
            visual_features = relevant_visual_features
            input_ids = self.ptuning_setup_pixtral(input_ids=input_ids)
            # Note: length is not used for pixtral model downstream. Setting it to a list
            # of length of input_ids causes errors downstream. So, supplying a placeholder.
            length = input_ids[0].shape[0]

        elif self.model_type == 'llava_next':
            visual_features = LlavaNextUtils.rearrange_image_features(
                visual_features, self.image_newlines["image_newline"],
                image_size)
            input_ids = self.ptuning_setup_llava_next(visual_features,
                                                      pre_prompt, post_prompt)
            length = input_ids.shape[1]
        elif self.model_type == 'mllama':
            pre_input_ids = self.tokenizer(pre_prompt,
                                           return_tensors="pt",
                                           padding=True).input_ids
            if n_prompts_n_images:
                length = [pre_input_ids.shape[1]] * self.args.batch_size
            else:
                length = pre_input_ids.shape[1]
            post_input_ids = None
        elif self.model_type == 'llava_onevision':
            if self.args.video_path is None:
                visual_features = torch.split(visual_features,
                                              visual_features.shape[0] //
                                              self.args.batch_size,
                                              dim=0)
                visual_features = LlavaOnevisionUtils.pack_image_features(
                    visual_features,
                    image_size,
                    image_newline=self.image_newlines["image_newline"],
                )
            else:
                visual_features = LlavaOnevisionUtils.apply_pooling(
                    visual_features)
                visual_features = visual_features.reshape(
                    self.args.batch_size,
                    self.num_frames * visual_features.shape[1], -1)
                image_newline = self.image_newlines["image_newline"][
                    None, None, :].repeat(self.args.batch_size, 1,
                                          1).to(visual_features.device)
                visual_features = torch.cat((visual_features, image_newline),
                                            dim=1)

            pre_input_ids = self.tokenizer(pre_prompt,
                                           return_tensors="pt",
                                           padding=True).input_ids
            post_input_ids = self.tokenizer(post_prompt,
                                            return_tensors="pt",
                                            padding=True).input_ids
            length = pre_input_ids.shape[1] + visual_features.shape[
                1] + post_input_ids.shape[1]
        else:
            pre_input_ids = self.tokenizer(pre_prompt,
                                           return_tensors="pt",
                                           padding=True).input_ids
            if post_prompt[0] is not None:
                post_input_encoded = self.tokenizer(post_prompt,
                                                    return_tensors="pt",
                                                    padding=True)
                post_input_ids = post_input_encoded.input_ids
                if n_prompts_n_images and 'neva' not in self.model_type:
                    post_input_attention_mask = post_input_encoded.attention_mask
                    post_input_ids = [
                        input_id[mask.bool()] for input_id, mask in zip(
                            post_input_ids, post_input_attention_mask)
                    ]

                if self.model_type == 'video-neva':
                    length = pre_input_ids.shape[1] + post_input_ids.shape[
                        1] + visual_atts.shape[2] * visual_atts.shape[1]
                elif self.model_type == 'internvl':
                    length = pre_input_ids.shape[1] + post_input_ids.shape[
                        1] + visual_atts.shape[0] * visual_atts.shape[1]
                else:
                    if n_prompts_n_images:
                        length = [
                            pre_input_ids.shape[1] + visual_atts.shape[1] +
                            post_input_id.shape[0]
                            for post_input_id in post_input_ids
                        ]
                    else:
                        length = pre_input_ids.shape[1] + post_input_ids.shape[
                            1] + visual_atts.shape[1]
            else:
                post_input_ids = None
                assert pre_input_ids.shape[0] == visual_atts.shape[0]
                if visual_atts.shape[0] == 1:
                    length = pre_input_ids.shape[1] + visual_atts.shape[1]
                else:
                    length = [
                        pre_input_ids.shape[1] + visual_atts.shape[1]
                        for _ in range(visual_atts.shape[0])
                    ]

        if n_prompts_n_images:
            if isinstance(length, int): length = [length]
            assert isinstance(length, list)
            input_lengths = torch.IntTensor(length).to(torch.int32)
        else:
            assert isinstance(length, int)
            input_lengths = torch.IntTensor([length] * self.args.batch_size).to(
                torch.int32)

        if self.model_type in [
                'fuyu', 'kosmos-2', 'phi-3-vision', 'llava_next', 'pixtral'
        ]:
            return input_ids, input_lengths, [
                visual_features
            ], visual_features, model_runner_input
        if self.model_type == 'phi-4-multimodal':
            multimodal_features = torch.cat([visual_features, audio_features],
                                            dim=1)
            return input_ids, input_lengths, [
                multimodal_features
            ], multimodal_features, model_runner_input

        input_ids, ptuning_args = self.setup_fake_prompts(
            visual_features, pre_input_ids, post_input_ids, input_lengths)

        return input_ids, input_lengths, ptuning_args, visual_features, model_runner_input

    @staticmethod
    def tokenizer_image_token(batch_size,
                              pre_prompt,
                              post_prompt,
                              tokenizer,
                              image_token_index=-200):
        if isinstance(post_prompt, list):
            prompts = [pre_prompt + item for item in post_prompt]
        else:
            prompts = [pre_prompt + post_prompt]

        def insert_separator(X, sep):
            return [
                ele for sublist in zip(X, [sep] * len(X)) for ele in sublist
            ][:-1]

        result = []
        for prompt in prompts:
            prompt_chunks = [
                tokenizer(chunk).input_ids for chunk in prompt.split("<image>")
            ]
            input_ids = []
            offset = 0
            if (len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0
                    and prompt_chunks[0][0] == tokenizer.bos_token_id):
                offset = 1
                input_ids.append(prompt_chunks[0][0])

            for x in insert_separator(prompt_chunks,
                                      [image_token_index] * (offset + 1)):
                input_ids.extend(x[offset:])

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_ids[input_ids == image_token_index] = 0
            result.append(input_ids)

        if not isinstance(post_prompt, list):
            result = result[0].unsqueeze(0).expand(batch_size, -1)
        return result

    def split_prompt_by_images(self, tensor):
        batch_splits = []
        for batch in tensor:
            # Find indices where value is zero (<image>)
            zero_indices = (batch == 0).nonzero(as_tuple=False).squeeze(0)
            # Add starting point for slicing
            start_idx = 0
            splits = []
            for idx in zero_indices:
                if start_idx != idx:  # Ensure not slicing zero-length tensors
                    splits.append(batch[start_idx:idx].unsqueeze(0))
                start_idx = idx + 1  # Move start index past the zero
            if start_idx < len(
                    batch):  # Handle last segment if it's not zero-ending
                splits.append(batch[start_idx:].unsqueeze(0))
            # Remove empty tensors resulting from consecutive zeros
            splits = [split for split in splits if split.numel() > 0]
            batch_splits.append(splits)

        return batch_splits

    def prepare_position_ids_for_cogvlm(self, input_ids):
        batch_size = len(input_ids)
        position_ids = torch.arange(input_ids.shape[1])
        position_ids[2:1227] = 2
        position_ids[1227:] = torch.arange(3, input_ids.shape[1] + 1 - 1225)

        position_ids = position_ids.to(torch.int32).to('cuda')
        input_position_ids = []
        for i in range(batch_size):
            input_position_ids.append(position_ids)

        return input_position_ids

    def generate(self,
                 pre_prompt,
                 post_prompt,
                 image,
                 decoder_input_ids,
                 max_new_tokens,
                 other_vision_inputs={},
                 other_audio_inputs={},
                 other_decoder_inputs={}):
        profiler.start("Generate")
        profiler.start("Preprocess")
        if 'qwen2_vl' in self.model_type:
            input_ids, input_lengths, ptuning_args, visual_features, mrope_args = self.preprocess(
                pre_prompt, post_prompt, image, other_vision_inputs,
                other_audio_inputs)
            mrope_params = MropeParams(
                mrope_rotary_cos_sin=mrope_args[0],
                mrope_position_deltas=mrope_args[1],
            )
        else:
            input_ids, input_lengths, ptuning_args, visual_features, model_runner_input = self.preprocess(
                pre_prompt, post_prompt, image, other_vision_inputs,
                other_audio_inputs)
        profiler.stop("Preprocess")

        # use prompt tuning to pass multimodal features
        # model.generate() expects the following params (see layers/embedding.py):
        # args[0]: prompt embedding table, [batch_size, multimodal_len, hidden_size], later flattened to [batch_size * multimodal_len, hidden_size]
        # args[1]: prompt task ids, [batch_size]. in multimodal case, arange(batch_size), i.e. in VILA batching mode 2, each image is treated separately in the batch instead of concated together (although the prompt embedding table has to be concated)
        # args[2]: prompt task vocab size, [1]. assuming all table has the same length, which in multimodal case equals to multimodal_len
        profiler.start("LLM")
        if self.decoder_llm and self.model_type != "mllama":
            end_id = self.tokenizer.eos_token_id
            if 'opt' in self.model_type and 'blip2' in self.model_type:
                # For BLIP2-OPT, model outputs a "\n" at the end.
                # we avoid it by using newline as the end token
                end_id = self.tokenizer.encode("\n",
                                               add_special_tokens=False)[0]

            if self.model_type == 'cogvlm':
                input_position_ids = self.prepare_position_ids_for_cogvlm(
                    input_ids)

            prompt_tasks = None
            prompt_table = None
            if not self.cpp_e2e:
                batch_size = len(input_ids)
                prompt_tasks = ",".join(
                    np.arange(batch_size, dtype=np.int32).astype(str))
                prompt_table = torch.stack([ptuning_args[0]])
                prompt_table = prompt_table.view(batch_size, -1,
                                                 prompt_table.shape[-1])

            output_ids = self.model.generate(
                input_ids,
                input_position_ids=input_position_ids
                if self.model_type == 'cogvlm' else None,
                mrope_params=mrope_params
                if self.model_type == 'qwen2_vl' else None,
                encoder_input_features=model_runner_input
                if self.cpp_e2e else None,
                sampling_config=None,
                prompt_table=prompt_table,
                prompt_tasks=prompt_tasks,
                max_new_tokens=max_new_tokens,
                end_id=end_id,
                pad_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None else
                self.tokenizer.all_special_ids[0],
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                temperature=self.args.temperature,
                repetition_penalty=self.args.repetition_penalty,
                num_beams=self.args.num_beams,
                lora_uids=self.args.lora_task_uids,
                output_sequence_lengths=False,
                return_dict=False,
                mm_embedding_offloading=self.args.mm_embedding_offloading)
        elif self.model_type == "mllama":
            # When image is passed:
            # the shape of visual_features is [bs, 1, 4, 1025, hidden_size]
            # the shape of cross_attention_mask is [bs, decode_input_len, 1, 4]
            # When image is None, create dummy visual_features and cross_attention_mask
            if visual_features is None:
                visual_features = torch.zeros([
                    self.args.batch_size, 1, 4, 1,
                    self.model_config.hidden_size * self.runtime_mapping.tp_size
                ],
                                              dtype=self.model.dtype,
                                              device=self.device)
                dummy_cross_attention_mask = torch.zeros(
                    [self.args.batch_size, input_ids.shape[1], 1, 4],
                    dtype=bool,
                    device=self.device)
                skip_cross_attn_blocks = torch.ones([1],
                                                    dtype=torch.bool,
                                                    device='cpu')
            else:
                skip_cross_attn_blocks = torch.zeros([1],
                                                     dtype=torch.bool,
                                                     device='cpu')

            visual_features = visual_features.to(self.model.dtype).chunk(
                self.args.batch_size, dim=0)
            encoder_input_features = []
            cross_attention_masks = []
            encoder_output_lengths = []
            for batch_idx in range(self.args.batch_size):
                visual_feature = visual_features[batch_idx]
                num_vision_tokens = visual_feature.shape[3]
                visual_feature = visual_feature.reshape(
                    [-1, visual_feature.shape[-1]])
                encoder_max_input_length = visual_feature.shape[0]
                encoder_input_lengths = torch.IntTensor(
                    [encoder_max_input_length]).to(visual_feature.device)

                # prepare cross_attention_mask of context phase
                if 'cross_attention_mask' in other_decoder_inputs:
                    cross_attention_mask = other_decoder_inputs[
                        'cross_attention_mask'][batch_idx]
                else:
                    cross_attention_mask = dummy_cross_attention_mask[batch_idx]
                text_total_length, *_ = cross_attention_mask.shape
                cross_attention_mask = cross_attention_mask.repeat_interleave(
                    num_vision_tokens, dim=2)
                cross_attention_mask = cross_attention_mask.view(
                    text_total_length, -1)
                cross_attention_mask = cross_attention_mask.unsqueeze(1)
                cross_attention_mask = cross_attention_mask.to(
                    visual_feature.device).to(torch.bool).reshape(
                        [-1, cross_attention_mask.shape[-1]])

                # prepare cross_attention_mask for generation phase and concat them
                tmp_mask = [cross_attention_mask] + [
                    cross_attention_mask[-1:, :] for _ in range(max_new_tokens)
                ]
                cross_attention_mask = torch.concat(tmp_mask)

                encoder_input_features.append(visual_feature)
                cross_attention_masks.append(cross_attention_mask)
                encoder_output_lengths.append(encoder_input_lengths)

            outputs = self.model.generate(
                batch_input_ids=input_ids,
                encoder_input_ids=None,
                encoder_input_features=encoder_input_features,
                encoder_output_lengths=encoder_output_lengths,
                cross_attention_masks=cross_attention_masks,
                max_new_tokens=max_new_tokens,
                # max_attention_window_size=args.max_attention_window_size,
                # sink_token_length=args.sink_token_length,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                # length_penalty=args.length_penalty,
                # early_stopping=args.early_stopping,
                # beam_width_array=args.beam_width_array,
                repetition_penalty=self.args.repetition_penalty,
                # presence_penalty=args.presence_penalty,
                # frequency_penalty=args.frequency_penalty,
                # stop_words_list=stop_words_list,
                # bad_words_list=bad_words_list,
                # output_cum_log_probs=(args.output_cum_log_probs_npy != None),
                # output_log_probs=(args.output_log_probs_npy != None),
                # random_seed=args.random_seed,
                # lora_uids=args.lora_task_uids,
                # prompt_table=args.prompt_table_path,
                # prompt_tasks=args.prompt_tasks,
                # streaming=args.streaming,
                output_sequence_lengths=True,
                # no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                return_dict=True,
                # medusa_choices=args.medusa_choices,
                # return_all_generated_tokens=args.return_all_generated_tokens,
                # input_token_extra_ids=input_token_extra_ids,
                encoder_max_input_length=encoder_max_input_length,
                skip_cross_attn_blocks=skip_cross_attn_blocks,
            )
            if mpi_rank() == 0:
                output_ids = outputs["output_ids"]
        else:
            if self.model_type in ['nougat', 'pix2struct']:
                # Trim encoder input_ids to match visual features shape
                ids_shape = (self.args.batch_size, visual_features.shape[1])
                if self.model_type == 'nougat':
                    input_ids = torch.zeros(ids_shape, dtype=torch.int32)
                elif self.model_type == 'pix2struct':
                    input_ids = torch.ones(ids_shape, dtype=torch.int32)

            output_ids = self.model.generate(
                input_ids,
                decoder_input_ids,
                max_new_tokens,
                num_beams=self.args.num_beams,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                debug_mode=False,
                prompt_embedding_table=ptuning_args[0],
                prompt_tasks=ptuning_args[1],
                prompt_vocab_size=ptuning_args[2])

            # Reset input_lengths to match decoder_input_ids
            input_lengths = torch.ones(input_lengths.shape,
                                       dtype=input_lengths.dtype)
        profiler.stop("LLM")

        if mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            profiler.start("Tokenizer decode")
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx]:],
                    skip_special_tokens=True) for batch_idx in range(
                        min(self.args.batch_size, input_lengths.shape[0]))
            ]

            stripped_text = [[
                output_beams_list[batch_idx][beam_idx].strip()
                for beam_idx in range(self.args.num_beams)
            ] for batch_idx in range(
                min(self.args.batch_size, input_lengths.shape[0]))]
            profiler.stop("Tokenizer decode")
            profiler.stop("Generate")
            return stripped_text
        else:
            profiler.stop("Generate")
            return None

    def get_visual_features(self, image, other_vision_inputs):
        visual_features = {
            self.vision_input_names[0]:
            image.to(str_dtype_to_torch(self.vision_precision)),
        }
        if self.model_type == "qwen2_vl":
            other_vision_inputs['attention_mask'] = other_vision_inputs[
                'attention_mask'].to(str_dtype_to_torch(self.vision_precision))
        for key, tensor in other_vision_inputs.items():
            visual_features.update({key: tensor})

        tensor_info = [
            TensorInfo(self.vision_input_names[0],
                       str_dtype_to_trt(self.vision_precision), image.shape),
        ]
        for key, tensor in other_vision_inputs.items():
            tensor_info.append(
                TensorInfo(key, torch_dtype_to_trt(tensor.dtype), tensor.shape))

        visual_output_info = self.visual_encoder_session.infer_shapes(
            tensor_info)
        self.visual_encoder_session.set_shapes(visual_features)
        visual_outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device=image.device)
            for t in visual_output_info
        }

        ok = self.visual_encoder_session.run(visual_features, visual_outputs,
                                             self.stream.cuda_stream)
        assert ok, "Runtime execution failed for vision encoder session"
        self.stream.synchronize()

        image_embeds = visual_outputs[self.vision_output_names[0]]

        if self.args.mm_embedding_offloading:
            # CUDA Stream Overlapping Requirements:
            # 1. Both memory copy stream and kernel execution stream must be non-default streams
            # 2. For host<->device transfers (H2D/D2H), host memory MUST be page-locked (pinned)
            pinned_embeds = torch.empty_like(image_embeds,
                                             device='cpu',
                                             pin_memory=True)
            pinned_embeds.copy_(image_embeds, non_blocking=True)
            image_embeds = pinned_embeds

        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        return image_embeds, image_atts

    def get_audio_features(self, audio, other_audio_inputs):
        tmp_features, _ = self.audio_model.encoder(
            audio.to(str_dtype_to_torch(self.audio_precision)),
            other_audio_inputs['attention_mask'])
        speech_out = self.audio_model.audio_projection['speech'](tmp_features)
        vision_out = self.audio_model.audio_projection['vision'](tmp_features)
        return torch.cat((speech_out, vision_out), dim=-1)

    def setup_fake_prompts_vila(self, batch_size, visual_features,
                                split_input_ids, input_lengths):
        # visual_features (num_images, feature_len, token_embed)
        # Assemble fake prompts which points to image embedding actually
        fake_prompt_counter = self.model_config.vocab_size
        if batch_size == 1:
            # only check for multi-image inference (mode 1)
            assert len(visual_features) <= len(
                split_input_ids
            ), "Unexpected number of visual features. Please check #<image> in prompt and the #image files."

        input_ids = []
        if batch_size == 1:
            # mode 1: multiple image as a whole, concat all prompts together, <pre><image1><inter><image2>...<post>
            input_ids = [split_input_ids[0]]
            for idx in range(
                    len(visual_features)
            ):  # TODO:alternatively make TensorInfo iterable if this breaks others
                fake_prompt_id = torch.arange(
                    fake_prompt_counter,
                    fake_prompt_counter + visual_features.shape[1])
                fake_prompt_counter += visual_features.shape[1]
                fake_prompt_id = fake_prompt_id.unsqueeze(0)
                input_ids.append(fake_prompt_id)
                # in case no inter or post prompt
                if len(split_input_ids) > idx + 1:
                    input_ids.append(split_input_ids[idx + 1])
            input_ids = torch.cat(input_ids, dim=1).contiguous().to(torch.int32)
            input_ids = input_ids.reshape(batch_size, -1)

        elif batch_size > 1:
            # mode 2: each image have individual prompt, <pre><image><post>
            for idx in range(len(visual_features)):
                input_ids.append(split_input_ids[idx][0])
                fake_prompt_id = torch.arange(
                    fake_prompt_counter,
                    fake_prompt_counter + visual_features.shape[1])
                fake_prompt_id = fake_prompt_id.unsqueeze(0)
                input_ids.append(fake_prompt_id)
                if len(split_input_ids[idx]) > 1:
                    input_ids.append(split_input_ids[idx][1])
            result = []
            for i in range(0, len(input_ids), 3):
                # Concatenate every 3 items (<pre>, <image>, <post>)
                concatenated = torch.cat(input_ids[i:i + 3],
                                         dim=1).to(torch.int32).squeeze(0)
                result.append(concatenated)
            input_ids = result

        if (self.decoder_llm
                or self.runtime_mapping.is_first_pp_rank()) and isinstance(
                    visual_features, torch.Tensor):
            ptuning_args = self.ptuning_setup(visual_features, input_ids,
                                              input_lengths)
        else:
            ptuning_args = [None, None, None]

        return input_ids, ptuning_args

    def setup_fake_prompts(self, visual_features, pre_input_ids, post_input_ids,
                           input_lengths):
        # Assemble fake prompts which points to image embedding actually
        if hasattr(self, 'num_frames') and (visual_features.shape[1]
                                            == self.num_frames):
            visual_features = visual_features.view(visual_features.shape[0], -1,
                                                   visual_features.shape[-1])

        if visual_features is not None:
            if self.python_e2e:
                # Non-IFB Mode(used in python session): All requests in a batch have their prompt_table concatenated in
                # a shape of (bs*vision_embedding_len, vision_hidden). So only one fake_prompt_id is needed for the
                # entire batch, with values from 0 to bs * vision_embedding_len-1.
                fake_prompt_id = torch.arange(
                    self.model_config.vocab_size, self.model_config.vocab_size +
                    visual_features.shape[0] * visual_features.shape[1])
                fake_prompt_id = fake_prompt_id.reshape(
                    visual_features.shape[0], visual_features.shape[1])
            else:
                # IFB Mode(used in c++ session): Each request's prompt_table is independent and requires a fake_prompt_id
                # for each request, with values ranging from 0 to vision_embedding_len-1.
                fake_prompt_id = torch.arange(
                    self.model_config.vocab_size,
                    self.model_config.vocab_size + visual_features.shape[1])
                fake_prompt_id = fake_prompt_id.repeat(visual_features.shape[0],
                                                       1)

        if 'internvl' in self.model_type:
            fake_prompt_id = fake_prompt_id.reshape(1, -1)

        if 'cogvlm' in self.model_type:
            input_ids = torch.cat(
                [pre_input_ids[:, 0:1], fake_prompt_id, pre_input_ids[:, 1:]],
                dim=1).contiguous().to(torch.int32)
        elif self.model_type == 'mllama':
            input_ids = pre_input_ids.contiguous().to(torch.int32)
        else:
            if post_input_ids is not None:
                if isinstance(post_input_ids, list):
                    pre_input_fake_prompt_ids = [
                        pre_input_ids[:len(fake_prompt_id)], fake_prompt_id
                    ]
                    pre_input_fake_prompt_ids = torch.cat(
                        pre_input_fake_prompt_ids,
                        dim=1).contiguous().to(torch.int32)
                    input_ids = [
                        torch.cat((pre_input_fake_prompt_id,
                                   post_input_id)).contiguous().to(torch.int32)
                        for pre_input_fake_prompt_id, post_input_id in zip(
                            pre_input_fake_prompt_ids, post_input_ids)
                    ]
                else:
                    input_ids = [pre_input_ids, fake_prompt_id, post_input_ids]
                    input_ids = torch.cat(input_ids,
                                          dim=1).contiguous().to(torch.int32)
            else:
                input_ids = [fake_prompt_id, pre_input_ids]
                input_ids = torch.cat(input_ids,
                                      dim=1).contiguous().to(torch.int32)

        if (self.decoder_llm or self.runtime_mapping.is_first_pp_rank()
            ) and self.model_type != "mllama" and isinstance(
                visual_features, torch.Tensor):
            ptuning_args = self.ptuning_setup(visual_features, input_ids,
                                              input_lengths)
        else:
            ptuning_args = [None, None, None]

        return input_ids, ptuning_args

    def get_rope_index(
        self,
        input_ids: torch.IntTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.IntTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(3,
                                      input_ids.shape[0],
                                      input_ids.shape[1],
                                      dtype=input_ids.dtype,
                                      device=input_ids.device)
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                        llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) +
                        st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                        -1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                        llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                        llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len +
                        st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                        llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) +
                        st_idx)

                llm_positions = torch.cat(llm_pos_ids_list,
                                          dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 -
                                             len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
                    input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[
                    -1]
            else:
                position_ids = (torch.arange(input_ids.shape[1],
                                             device=input_ids.device).view(
                                                 1, 1, -1).expand(
                                                     3, input_ids.shape[0], -1))
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def setup_fake_prompts_qwen2vl(self, visual_features, input_ids,
                                   vision_grid_thws, attention_mask,
                                   input_lengths):

        visual_features = torch.unsqueeze(visual_features, 0)

        # Get the rope index
        # From HF's preprocess code
        mrope_position_ids, mrope_position_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw=vision_grid_thws,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        # This is where we convert input_ids of image features into fake_prompt_ids mapping for TRT-LLM engine.
        masks = (input_ids == self.image_token_id) | (
            input_ids == self.vision_token_id) | (input_ids
                                                  == self.video_token_id)
        cumulative_counts = masks.cumsum(dim=1)
        values = (self.model_config.vocab_size - 1) + cumulative_counts
        input_ids[masks] = values[masks]

        if self.decoder_llm or self.runtime_mapping.is_first_pp_rank():
            ptuning_args = self.ptuning_setup(visual_features, input_ids,
                                              input_lengths)
        else:
            ptuning_args = [None, None, None]

        # This does not have dependency on input.
        # Switch to attributes to use across iterations.
        if not hasattr(self, 'rotary_cos_sin'):
            inv_freq, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                num_pos=self.max_position_embeddings,
                dim=int(self.hidden_size / self.num_attention_heads),
                theta=float(self.rope_theta),
                scale_type=RotaryScalingType.mrope)
            self.rotary_cos_sin = torch.from_numpy(rotary_cos_sin).to(
                visual_features.device)
            self.rotary_cos_sin = self.rotary_cos_sin.reshape(
                self.max_position_embeddings,
                int(self.hidden_size / self.num_attention_heads / 2), 2)
            self.cos_ori = self.rotary_cos_sin[:, :, 0]
            self.sin_ori = self.rotary_cos_sin[:, :, 1]

        mrope_position_ids = mrope_position_ids.transpose(1, 0)
        mrope_position_ids_padding = torch.zeros(
            mrope_position_ids.shape[:-1] + (self.max_position_embeddings, ),
            dtype=torch.int32,
            device=visual_features.device)
        mrope_position_ids_padding[:, :, :mrope_position_ids.
                                   shape[-1]] = mrope_position_ids
        cos = self.cos_ori[mrope_position_ids_padding]
        sin = self.sin_ori[mrope_position_ids_padding]

        mrope_section = [16, 24, 24]
        cos = torch.cat([
            m[:, i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))
        ],
                        dim=-1).unsqueeze(-1)
        sin = torch.cat([
            m[:, i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))
        ],
                        dim=-1).unsqueeze(-1)
        concat_cos_sin = torch.concatenate((cos, sin), axis=-1)
        concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0], -1)

        mrope_args = [concat_cos_sin, mrope_position_deltas]
        return input_ids, ptuning_args, mrope_args

    def ptuning_setup_fuyu(self, input_ids, image_patches_indices):
        res_input_ids = []
        for cur_input_ids, cur_image_patches_indices in zip(
                input_ids, image_patches_indices):
            # Truncate input_ids to the length of image_patches_indices
            cur_image_patches_indices = cur_image_patches_indices[:len(
                cur_input_ids)]
            # Get ids of the image_patches
            non_zero_mask = cur_image_patches_indices != -1
            # Replace input_ids with image_patches_indices values (where the patches are placed)
            cur_input_ids = cur_input_ids.masked_scatter(
                non_zero_mask,
                cur_image_patches_indices[non_zero_mask] +
                self.model_config.vocab_size,
            )
            res_input_ids.append(cur_input_ids)
        return res_input_ids

    def ptuning_setup_pixtral(self, input_ids):
        # input_ids obtained from processor has token_ids for text as well as image tokens
        # where each image token is represented by the same image_token_index.
        image_token_index = self.image_token_index
        vocab_size = self.vocab_size
        # Replace all image tokens with a unique token_id > text_vacab_size.
        # This shall be used to lookup the prompt table.
        for img_idx in range(self.args.batch_size):
            # Note: We reset replacer to text_vocab_size for each sample. This is as opposed to doing `replacer = vocab_size + img_idx * tokens_per_task`.
            # That part of the look-up manipulation is done by the `task_ids` input to PromptEmbedding forward.
            replacer = vocab_size
            for token_idx in range(len(input_ids[img_idx])):
                if input_ids[img_idx][token_idx] == image_token_index:
                    input_ids[img_idx][token_idx] = replacer
                    replacer += 1
        return input_ids

    def ptuning_setup_llava_next(self, visual_features, pre_prompt,
                                 post_prompt):
        input_ids = []
        fake_prompt_ids = list(
            range(self.model_config.vocab_size,
                  self.model_config.vocab_size + visual_features.shape[0]))
        input_ids = self.tokenizer.encode(
            pre_prompt[0]) + fake_prompt_ids + self.tokenizer.encode(
                post_prompt[0])[self.tokenizer.add_bos_token:]
        input_ids = [input_ids] * len(pre_prompt)
        input_ids = torch.tensor(input_ids)
        return input_ids

    def ptuning_setup_phi3(self, visual_features, audio_features, input_ids,
                           num_img_tokens, num_aud_tokens):
        fake_prompt_id = torch.arange(
            self.model_config.vocab_size,
            self.model_config.vocab_size + visual_features.shape[0])
        if self.model_type == "phi-3-vision":
            MAX_INPUT_ID = int(1e9)
            positions = torch.nonzero(
                (input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=False)
        elif self.model_type == "phi-4-multimodal":
            IMAGE_TOKEN_ID = 200010
            positions = torch.nonzero(input_ids == IMAGE_TOKEN_ID,
                                      as_tuple=False)
        idx = 0
        for _, cnt in enumerate(num_img_tokens):
            input_ids[positions[idx, 0], positions[idx, 1]:positions[idx, 1] +
                      cnt] = fake_prompt_id[idx:idx + cnt]
            idx += cnt

        if self.model_type == "phi-4-multimodal" and audio_features is not None:
            prompt_id_offset = self.model_config.vocab_size + visual_features.shape[
                0]
            fake_prompt_id = torch.arange(
                prompt_id_offset, prompt_id_offset + audio_features.shape[0])
            AUDIO_TOKEN_ID = 200011
            positions = torch.nonzero(input_ids == AUDIO_TOKEN_ID,
                                      as_tuple=False)
            idx = 0
            for _, cnt in enumerate(num_aud_tokens):
                input_ids[positions[idx, 0], positions[idx,
                                                       1]:positions[idx, 1] +
                          cnt] = fake_prompt_id[idx:idx + cnt]
                idx += cnt
        return input_ids

    def ptuning_setup(self, prompt_table, input_ids, input_lengths):
        hidden_size = self.model_config.hidden_size * self.runtime_mapping.tp_size
        if prompt_table is not None:
            task_vocab_size = torch.tensor(
                [prompt_table.shape[1]],
                dtype=torch.int32,
            ).cuda()
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1],
                 prompt_table.shape[2]))

            assert prompt_table.shape[
                1] == hidden_size, "Prompt table dimensions do not match hidden size"

            if hasattr(self.model_config, 'dtype'):
                prompt_table = prompt_table.cuda().to(
                    dtype=str_dtype_to_torch(self.model_config.dtype))
            else:
                if self.args.mm_embedding_offloading:
                    # CUDA Stream Overlapping Requirements:
                    # 1. Both memory copy stream and kernel execution stream must be non-default streams
                    # 2. For host<->device transfers (H2D/D2H), host memory MUST be page-locked (pinned)
                    prompt_table = prompt_table.pin_memory().to(
                        dtype=self.model.dtype)
                else:
                    prompt_table = prompt_table.cuda().to(
                        dtype=self.model.dtype)
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        remove_input_padding = self.model_config.remove_input_padding if hasattr(
            self.model_config,
            'remove_input_padding') else self.model_config.use_packed_input
        if remove_input_padding:
            tasks = torch.zeros([torch.sum(input_lengths)],
                                dtype=torch.int32).cuda()
            if self.decoder_llm: tasks = tasks.unsqueeze(0)
        else:
            if not isinstance(input_ids, list):
                tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()
            else:
                max_length = max(input_id.size(-1) for input_id in input_ids)
                tasks = torch.zeros((len(input_ids), max_length),
                                    dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]

    def load_test_data(self, image_path=None, video_path=None):

        def load_images(image_paths):
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            images = []
            for image_path in image_paths:
                if image_path.startswith("http") or image_path.startswith(
                        "https"):
                    logger.info(f"downloading image from url {image_path}")
                    try:
                        response = requests.get(image_path, timeout=5)
                        response.raise_for_status()
                        if 'image' not in response.headers.get(
                                'Content-Type', ''):
                            raise Exception(
                                f"URL does not point to an image: {image_path}."
                            )
                        image = Image.open(BytesIO(
                            response.content)).convert("RGB")
                    except (UnidentifiedImageError, IOError):
                        raise Exception(
                            f"Cannot identify image file at URL: {image_path}.")
                    except Exception as e:
                        raise Exception(
                            f"Failed to download image from url {image_path}: {e}"
                        )
                else:
                    image = Image.open(image_path).convert("RGB")
                images.append(image)
            return images if len(images) > 1 else images[0]

        if "vila" in self.model_type:
            if image_path is None:
                img_urls = [
                    'https://github.com/NVlabs/VILA/blob/6b941da19e31ddfdfaa60160908ccf0978d96615/demo_images/av.png?raw=true',
                    'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
                ] * 4

                img_urls = img_urls[:self.args.batch_size]
                self.args.image_path = ",".join(img_urls)
                images = load_images(img_urls)
            else:
                if isinstance(image_path, str):
                    image_path = image_path.split(self.args.path_sep)
                images = load_images(image_path)
        elif "pixtral" in self.model_type:
            if image_path is None:
                image_urls = [
                    "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png",
                    "https://www.ilankelman.org/stopsigns/australia.jpg",
                    "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png",
                    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                ]
                while len(image_urls) < self.args.batch_size:
                    image_urls *= 2
                image_urls = image_urls[:self.args.batch_size]
                self.args.image_path = ",".join(image_urls)
                images = load_images(image_urls)
            else:
                if isinstance(image_path, str):
                    image_path = image_path.split(self.args.path_sep)
                images = load_images(image_path)
            images = [images] if not isinstance(images, list) else images
        elif "nougat" in self.model_type:
            filepath = hf_hub_download(
                repo_id="hf-internal-testing/fixtures_docvqa",
                filename="nougat_paper.png",
                revision="ec57bf8c8b1653a209c13f6e9ee66b12df0fc2db",
                repo_type="dataset")
            images = Image.open(filepath)
        elif "fuyu" in self.model_type:
            filepath = hf_hub_download(repo_id="adept/fuyu-8b",
                                       filename="skateboard.png",
                                       repo_type='model')
            images = Image.open(filepath)
        elif "kosmos" in self.model_type:
            if image_path is None:
                image_path = 'https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png'
            images = load_images(image_path)
        elif "pix2struct" in self.model_type:
            if image_path is None:
                image_path = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_40963.png'
            images = load_images(image_path)
        elif "video-neva" in self.model_type:
            images = video_path
        elif "internlm" in self.model_type:
            img_url = "https://huggingface.co/internlm/internlm-xcomposer2-vl-7b/resolve/main/image1.webp"
            images = load_images(img_url)
        elif "internvl" in self.model_type:
            if image_path is None:
                img_url = 'https://huggingface.co/OpenGVLab/InternVL2-4B/resolve/main/examples/image1.jpg'
                images = Image.open(
                    requests.get(img_url, stream=True,
                                 timeout=5).raw).convert('RGB')
            else:
                images = Image.open(image_path).convert('RGB')
        elif "qwen2_vl" in self.model_type:
            images = []
            if self.args.image_path is None:
                img_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
                image = Image.open(
                    requests.get(img_url, stream=True,
                                 timeout=5).raw).convert('RGB')
                image = image.resize((504, 504))
                images.append(image)
            else:
                images = []
                for image_path in self.args.image_path:
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize((504, 504))
                    images.append(image)
        elif "llava_onevision" in self.model_type and self.args.video_path is not None:
            if self.args.video_path == 'llava-onevision-accuracy':
                self.args.video_path = hf_hub_download(
                    repo_id="raushan-testing-hf/videos-test",
                    filename="sample_demo_1.mp4",
                    repo_type="dataset")
            import av
            with av.open(self.args.video_path) as container:
                total_frames = container.streams.video[0].frames
                assert total_frames >= self.num_frames
                indices = np.arange(0, total_frames,
                                    total_frames / self.num_frames).astype(int)
                frames = []
                container.seek(0)
                start_index = indices[0]
                end_index = indices[-1]
                for i, frame in enumerate(container.decode(video=0)):
                    if i > end_index:
                        break
                    if i >= start_index and i in indices:
                        frames.append(frame)
                images = np.stack(
                    [x.to_ndarray(format="rgb24") for x in frames])
            images = torch.tensor(images)
        else:
            if self.model_type != 'mllama':
                if image_path is None:
                    if self.model_type == "llava":
                        image_path = [
                            'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
                        ] * 8
                        image_path = image_path[:self.args.batch_size]
                        self.args.image_path = ",".join(image_path)
                    else:
                        image_path = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
                else:
                    if isinstance(image_path, str):
                        image_path = image_path.split(self.args.path_sep)

            images = load_images(image_path) if image_path is not None else None
        return images

    def load_test_audio(self, audio_path):
        if self.model_type != "phi-4-multimodal":
            return None

        assert audio_path is not None
        import soundfile
        audio = soundfile.read(audio_path)
        return audio

    def setup_inputs(self, input_text, raw_image, raw_audio=None):
        from ..tools.multimodal_builder import compute_rotary_pos_emb
        other_vision_inputs = {}
        other_audio_inputs = {}
        other_decoder_inputs = {}
        if self.model_type not in ['qwen2_vl', 'vila', 'llava']:
            input_text = input_text[0] if isinstance(input_text,
                                                     list) else input_text

        if 'blip2' in self.model_type:
            image = self.processor(raw_image, input_text,
                                   return_tensors="pt")['pixel_values']
            if input_text is None:
                input_text = "Question: which city is this? Answer:"
            pre_prompt = input_text
            post_prompt = None
        elif 'internlm' in self.model_type:
            #Feed the raw image into vis_processor, to get processed image
            image = self.processor(raw_image).unsqueeze(0).cuda()
            if input_text is None:
                input_text = "Please describe this image in detail."
            pre_prompt = ''
            meta_instruction = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
            '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
            '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
            '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.',
            pre_prompt += f"""[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
            pre_prompt += f"""[UNUSED_TOKEN_146]user\n"""
            post_prompt = f"""{input_text}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
        elif 'qwen2_vl' in self.model_type:
            from qwen_vl_utils import process_vision_info
            from transformers.models.qwen2_vl.modeling_qwen2_vl import \
                VisionRotaryEmbedding
            hf_config = AutoConfig.from_pretrained(self.args.hf_model_dir)
            if input_text is None:
                input_text = ["Question: Describe this image. Answer:"
                              ] * self.args.batch_size
            messages = [[{
                "role":
                "user",
                "content": [
                    {
                        "type": "image",
                        "image": raw_image[idx],
                    },
                    {
                        "type": "text",
                        "text": input_text[idx],
                    },
                ],
            }] for idx in range(self.args.batch_size)]

            texts = [
                self.processor.apply_chat_template(msg,
                                                   tokenize=False,
                                                   add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            image = inputs['pixel_values']
            image_grid_thw = inputs['image_grid_thw']
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            cu_seqlens = torch.repeat_interleave(
                image_grid_thw[:, 1] * image_grid_thw[:, 2],
                image_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

            seq_length = image.shape[0]
            # Create block indices using bucketing
            block_indices = torch.bucketize(torch.arange(seq_length,
                                                         device=image.device),
                                            cu_seqlens,
                                            right=True) - 1

            # Generate block diagonal mask using matrix expansion
            attention_mask_vit = torch.where(
                block_indices.view(-1, 1) == block_indices.view(1, -1),
                torch.zeros((), device=image.device, dtype=image.dtype),
                torch.full((),
                           torch.finfo(torch.float16).min,
                           device=image.device,
                           dtype=image.dtype)).unsqueeze(0)

            decoder_input_ids = None
            post_prompt = None
            pre_prompt = None
            images_qwenvl = {
                "image": image,
                "input_ids": input_ids,
            }
            rotary_pos_emb = compute_rotary_pos_emb(
                image_grid_thw, hf_config, VisionRotaryEmbedding).to("cuda")
            other_vision_inputs['attention_mask_llm'] = attention_mask
            other_vision_inputs['image_grid_thw'] = image_grid_thw
            other_vision_inputs['attention_mask'] = attention_mask_vit
            other_vision_inputs['rotary_pos_emb'] = rotary_pos_emb
            return input_text, pre_prompt, post_prompt, images_qwenvl, decoder_input_ids, other_vision_inputs, other_audio_inputs, other_decoder_inputs
        elif 'nougat' in self.model_type:
            image = self.processor(raw_image,
                                   return_tensors="pt")['pixel_values']
            # Nougat doesn't need text prompt (mBART use single token to start generation), just leave a dummy one here
            if input_text is None:
                input_text = "Question: which city is this? Answer:"
            pre_prompt = input_text
            post_prompt = None

        elif 'cogvlm' in self.model_type:
            image = self.transform(raw_image).unsqueeze(0)
            if input_text is None:
                input_text = " [INST] which city is this? [/INST] "
            pre_prompt = input_text
            post_prompt = None

        elif 'phi-3-vision' in self.model_type:
            pre_prompt = "<|user|>\n<|image_1|>\n"
            if input_text is None:
                input_text = "Which city is this?"
            post_prompt = input_text + "<|end|>\n<|assistant|>\n"
            prompt = pre_prompt + post_prompt
            image = self.processor(text=prompt,
                                   images=raw_image,
                                   return_tensors="pt")
        elif 'phi-4-multimodal' in self.model_type:
            pre_prompt = "<|user|><|image_1|><|audio_1|>"
            post_prompt = "<|end|><|assistant|>"
            prompt = pre_prompt + post_prompt
            image = self.processor(text=prompt,
                                   images=[raw_image],
                                   audios=[raw_audio],
                                   return_tensors="pt")

        elif 'pixtral' in self.model_type:
            # Send image and text prompt to processor.
            pre_prompt = "<s>[INST][IMG]"
            if input_text is None:
                input_text = "What is in the image?"
            post_prompt = "[/INST]"
            prompt = pre_prompt + input_text + post_prompt
            dtype = str_dtype_to_torch(self.vision_precision)
            image = {'pixel_values': [], 'input_ids': []}
            for img_idx in range(self.args.batch_size):
                image_info = self.processor(text=prompt,
                                            images=[raw_image[img_idx]],
                                            return_tensors="pt").to(dtype)
                image['pixel_values'].append(image_info['pixel_values'].to(
                    self.device))
                image['input_ids'].append(image_info['input_ids'][0].to(
                    self.device))

        elif 'internvl' in self.model_type:
            pre_prompt = "<|system|>\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|end|><|user|>\n<image>\n"
            if input_text is None:
                input_text = "Please describe the image shortly."
            post_prompt = input_text + "<|end|><|assistant|>\n"
            prompt = pre_prompt + post_prompt
            image = self.processor(images=raw_image,
                                   return_tensors='pt').pixel_values

        elif self.model_type == "pix2struct":
            if input_text is None:
                input_text = ""
            inputs = self.processor(
                images=raw_image,
                text=input_text,
                return_tensors="pt",
            )
            image = inputs['flattened_patches']
            image = image.expand(self.args.batch_size, -1, -1).contiguous()
            pre_prompt = ""
            post_prompt = None

        elif self.model_type == "neva":
            image = self.transform(raw_image).unsqueeze(0)

            if input_text is None:
                input_text = "Hi! What is in this image?"

            pre_prompt = "<extra_id_0>System\n\n<extra_id_1>User\n"
            post_prompt = f"\n{input_text}\n<extra_id_1>Assistant\n"

        elif self.model_type == "video-neva":

            image = self.video_preprocess(
                raw_image)  # shape (1, num_frames, 3, H, W)

            if input_text is None:
                input_text = "Hi! What is in this video?"

            # SteerLM prompt template
            pre_prompt = """<extra_id_0>System\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n<extra_id_1>User"""
            post_prompt = f"\n{input_text}\n<extra_id_1>Assistant\n<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:4\n" ""

        elif self.model_type == "llava_next":
            if self.llm_name == "mistralai/Mistral-7B-Instruct-v0.2":
                pre_prompt = "[INST] "
                if input_text is None:
                    input_text = "Question: which city is this? Answer:"
                post_prompt = f"\n{input_text} [/INST]"
                prompt = pre_prompt + post_prompt

            elif self.llm_name == "NousResearch/Nous-Hermes-2-Yi-34B":
                pre_prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n"
                if input_text is None:
                    input_text = "Question: which city is this? Answer:"
                post_prompt = f"\n{input_text}<|im_end|><|im_start|>assistant\n"
                prompt = pre_prompt + post_prompt

            else:
                raise Exception(
                    f"Prompt template for {self.llm_name} for not included currently"
                )

            image = self.processor(text=prompt,
                                   images=raw_image,
                                   return_tensors="pt")

        elif self.model_type == 'vila':
            if input_text is None:
                input_text = "<image>\n Please elaborate what you see in the images?"
            if '8b' in self.args.hf_model_dir.lower():
                pre_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                post_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif '40b' in self.args.hf_model_dir.lower():
                pre_prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n"
                post_prompt = "<|im_end|><|im_start|>assistant\n"
            else:
                pre_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
                post_prompt = " ASSISTANT:"
            if isinstance(input_text, list):
                post_prompt = [input + post_prompt for input in input_text]
            else:
                post_prompt = input_text + post_prompt
            if not isinstance(raw_image, list):
                raw_image = [raw_image]
            image = self.processor(raw_image)

        elif self.model_type in ['llava', 'fuyu', 'kosmos-2']:
            if self.model_type == "llava":
                pre_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
                post_prompt = " ASSISTANT:"
                if input_text is None:
                    input_text = "\n Which city is this? Answer:"
                if isinstance(input_text, list):
                    post_prompt = [input + post_prompt for input in input_text]
                else:
                    post_prompt = input_text + post_prompt
            elif self.model_type == 'fuyu':
                pre_prompt = "Describe this image:"
                post_prompt = None
                if input_text is None:
                    input_text = "Answer the following VQAv2 question based on the image: How many people are in the image?\n"
            elif self.model_type == "kosmos-2":
                pre_prompt = ""
                post_prompt = None
                if input_text is None:
                    input_text = "<grounding>An image of"

            if self.model_type not in ['fuyu', 'kosmos-2']:
                post_prompt = " ASSISTANT:"
                if isinstance(input_text, list):
                    post_prompt = [input + post_prompt for input in input_text]
                else:
                    post_prompt = input_text + post_prompt
            else:
                post_prompt = None
            if self.model_type in ['fuyu', 'kosmos-2']:
                image = self.processor(text=input_text,
                                       images=raw_image,
                                       return_tensors='pt')
            else:
                if isinstance(input_text, list):
                    image = self.processor(text=input_text[0],
                                           images=raw_image,
                                           padding=True,
                                           return_tensors="pt")['pixel_values']
                else:
                    image = self.processor(text=input_text,
                                           images=raw_image,
                                           return_tensors="pt")['pixel_values']

        elif self.model_type in ['mllama']:
            if raw_image is not None:
                input_text = self.processor.apply_chat_template(
                    images=raw_image, text=input_text)
                inputs = self.processor(images=raw_image,
                                        text=input_text,
                                        return_tensors="pt")
                other_vision_inputs = {
                    "aspect_ratio_ids":
                    inputs["aspect_ratio_ids"].to(self.device).expand(
                        self.args.batch_size, -1).contiguous(),
                    "aspect_ratio_mask":
                    inputs["aspect_ratio_mask"].to(self.device).expand(
                        self.args.batch_size, -1, -1).contiguous(),
                }
                other_decoder_inputs = {
                    "cross_attention_mask":
                    inputs["cross_attention_mask"].to(self.device).expand(
                        self.args.batch_size, -1, -1, -1).contiguous(),
                }
                pre_prompt = input_text
                post_prompt = None
                image = inputs["pixel_values"]
            else:
                pre_prompt = input_text
                post_prompt = None
                image = None
                logger.warning(
                    "image_path is None. Will not pass image as input, skipping the vision encoder."
                )
                image = None
        elif self.model_type in ['llava_onevision']:
            pre_prompt = "<|im_start|>user " + "<video>" if self.args.video_path is not None else "<image>"
            if input_text is None:
                input_text = "Question: which city is this? Answer:" if self.args.video_path is None else "Why is this video funny?"
            post_prompt = f"\n{input_text}<|im_end|><|im_start|>assistant\n"
            prompt = pre_prompt + post_prompt

            if self.args.video_path is None:
                image = self.processor(images=raw_image,
                                       text=prompt,
                                       return_tensors="pt")
            else:
                image = self.processor(videos=list(raw_image),
                                       text=prompt,
                                       return_tensors="pt")

        # Repeat inputs to match batch size
        pre_prompt = [pre_prompt] * self.args.batch_size
        if not isinstance(input_text, list):
            post_prompt = [post_prompt] * self.args.batch_size
        if self.model_type not in [
                'fuyu', 'pix2struct', 'kosmos-2', 'vila', 'phi-3-vision',
                'phi-4-multimodal', 'llava_next', 'internvl', 'llava_onevision',
                'pixtral'
        ]:
            if image is not None:
                if image.dim() == 5:
                    image = image.expand(self.args.batch_size, -1, -1, -1,
                                         -1).contiguous()
                elif image.dim() == 6:
                    image = image.expand(self.args.batch_size, -1, -1, -1, -1,
                                         -1).contiguous()
                else:
                    if not isinstance(input_text, list):
                        image = image.expand(self.args.batch_size, -1, -1,
                                             -1).contiguous()
                    else:
                        image = image.expand(
                            min(self.args.batch_size, len(input_text)), -1, -1,
                            -1).contiguous()
        # Note: For pixtral model, image is a dict with each value being a list of tensors.
        # Moving to device is handled above. So, it's safe to skip this for pixtral.
        if image is not None and 'pixtral' not in self.model_type:
            image = image.to(self.device)
        # Generate decoder_input_ids for enc-dec models
        # Custom prompts can be added as:
        # decoder_input_ids = model.tokenizer(decoder_prompt).input_ids
        if self.decoder_llm:
            decoder_input_ids = None
        else:
            config = AutoConfig.from_pretrained(self.args.hf_model_dir)
            if "blip2" in self.model_type:
                decoder_start_id = config.text_config.decoder_start_token_id  # T5
            elif "nougat" in self.model_type:
                decoder_start_id = config.decoder.bos_token_id  # Nougat
            else:
                decoder_start_id = config.decoder_start_token_id

            decoder_input_ids = torch.IntTensor([[decoder_start_id]])
            decoder_input_ids = decoder_input_ids.repeat(
                (self.args.batch_size, 1))

        return input_text, pre_prompt, post_prompt, image, decoder_input_ids, other_vision_inputs, other_audio_inputs, other_decoder_inputs

    def run(self, input_text, input_image, input_audio, max_new_tokens):
        input_text, pre_prompt, post_prompt, processed_image, decoder_input_ids, other_vision_inputs, other_audio_inputs, other_decoder_inputs = self.setup_inputs(
            input_text, input_image, input_audio)
        output_text = self.generate(pre_prompt,
                                    post_prompt,
                                    processed_image,
                                    decoder_input_ids,
                                    max_new_tokens,
                                    other_vision_inputs=other_vision_inputs,
                                    other_audio_inputs=other_audio_inputs,
                                    other_decoder_inputs=other_decoder_inputs)
        return input_text, output_text
