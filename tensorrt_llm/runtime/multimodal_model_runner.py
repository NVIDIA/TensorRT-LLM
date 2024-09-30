import json
import os
import sys
from io import BytesIO

import requests

# isort: off
import torch
import numpy as np
# isort: on
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from .. import profiler
from .._utils import (mpi_rank, str_dtype_to_torch, str_dtype_to_trt,
                      supports_inflight_batching, trt_dtype_to_torch)
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
    def get_anyres_image_grid_shape(image_size, patch_size):
        """
            Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

            Args:
                image_size (tuple): The size of the input image in the format (width, height).
                patch_size (int): The size of each image patch.

            Returns:
                tuple: The shape of the image patch grid in the format (width, height).
            """
        IMAGE_GRID_PINPOINTS = [[336, 672], [672, 336], [672, 672], [1008, 336],
                                [336, 1008]]
        width, height = LlavaNextUtils.select_best_resolution(
            image_size, IMAGE_GRID_PINPOINTS)
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


class MultimodalModelRunner:

    def __init__(self, args):
        self.args = args

        self.runtime_rank = mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)

        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        # parse model type from visual engine config
        with open(os.path.join(self.args.visual_engine_dir, "config.json"),
                  "r") as f:
            config = json.load(f)
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

        if self.decoder_llm:
            if not supports_inflight_batching(self.args.llm_engine_dir):
                logger.warning(
                    "The given engine does not support in-flight batching, fallback to python session"
                )
                self.args.use_py_session = True

            if not PYTHON_BINDINGS and not self.args.use_py_session:
                logger.warning(
                    "Python bindings of C++ session is unavailable, fallback to Python session."
                )
                self.args.use_py_session = True

            args.debug_mode = False
            if args.debug_mode and not self.args.use_py_session:
                logger.warning(
                    "Debug mode is not supported in C++ session for now, fallback to Python session."
                )
                self.args.use_py_session = True

            if self.model_type != 'cogvlm' and not self.args.use_py_session:
                logger.warning(
                    "Only the cogvlm is supported in C++ session for now, fallback to Python session."
                )
                self.args.use_py_session = True

            self.use_py_session = self.args.use_py_session
        else:
            self.use_py_session = True

        self.init_image_encoder()
        self.init_tokenizer()
        self.init_llm()

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
            use_fast = False if self.model_type != "phi-3-vision" else True
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.hf_model_dir, use_fast=use_fast, use_legacy=False)

        self.tokenizer.padding_side = "right"

    def init_image_encoder(self):
        vision_encoder_path = os.path.join(self.args.visual_engine_dir,
                                           self.args.visual_engine_name)
        logger.info(f'Loading engine from {vision_encoder_path}')
        with open(vision_encoder_path, 'rb') as f:
            engine_buffer = f.read()
        logger.info(f'Creating session from engine {vision_encoder_path}')
        self.visual_encoder_session = Session.from_serialized_engine(
            engine_buffer)
        if self.model_type in ["phi-3-vision", "llava_next"]:
            self.image_newlines = {}
            image_newlines_path = os.path.join(self.args.visual_engine_dir,
                                               'image_newlines.safetensors')
            with safe_open(image_newlines_path,
                           framework="pt",
                           device=self.device) as f:
                for k in f.keys():
                    self.image_newlines[k] = f.get_tensor(k)

    def init_llm(self):
        if self.decoder_llm:
            if self.use_py_session:
                self.model = ModelRunner.from_dir(
                    self.args.llm_engine_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=False,
                    stream=self.stream,
                    enable_context_fmha_fp32_acc=self.args.
                    enable_context_fmha_fp32_acc)
                self.model_config = self.model.session._model_config
            else:
                self.model = ModelRunnerCpp.from_dir(
                    self.args.llm_engine_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=False,
                    enable_context_fmha_fp32_acc=self.args.
                    enable_context_fmha_fp32_acc)
                self.model_config = self.model.model_config
            self.runtime_mapping = self.model.mapping
        else:
            self.model = EncDecModelRunner.from_engine(
                os.path.basename(self.args.hf_model_dir),
                self.args.llm_engine_dir,
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
            "openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)
        frames = processor.preprocess(frames,
                                      return_tensors="pt")['pixel_values']
        # make dtype consistent with vision encoder
        media_tensors = frames.to(str_dtype_to_torch(
            self.vision_precision))  # [num_frames, 3, H, W]
        return media_tensors.unsqueeze(0)  #[1, num_frames, 3, H, W]

    def preprocess(self, warmup, pre_prompt, post_prompt, image):
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
            bs = image.shape[0]
            image = image.flatten(0, 1)
        elif self.model_type == 'llava_next':
            input = image
            image = input['pixel_values']
            bs = image.shape[0]
            image = image[0]
            image_size = input['image_sizes'][0].cpu()

        if not warmup:
            profiler.start("Vision")

        visual_features, visual_atts = self.get_visual_features(
            torch.stack(image['image_patches'], dim=0) if self.model_type ==
            'fuyu' else image)

        if not warmup:
            profiler.stop("Vision")

        if self.model_type == 'fuyu':
            visual_features = visual_features.squeeze()
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

        elif self.model_type == 'kosmos-2':
            visual_features = visual_features.squeeze()
        elif self.model_type == 'vila':
            input_ids = self.tokenizer_image_token(
                self.args.batch_size, pre_prompt[0] + post_prompt[0],
                self.tokenizer)
            batch_split_prompts = self.split_prompt_by_images(input_ids)
            first_batch_split_prompts = batch_split_prompts[0]
            # compute prompt length + visual length
            length = sum([ids.shape[1] for ids in first_batch_split_prompts])
            if self.args.batch_size == 1 and len(image) > 1:
                # mode 1: multiple image as a whole, flatten visual dims
                length += visual_atts.shape[0] * visual_atts.shape[1]
            else:
                # mode 2: multiple images individually (replicate prompt for each image)
                length += visual_atts.shape[1]

            input_lengths = torch.IntTensor([length] * self.args.batch_size).to(
                torch.int32)
            input_ids, ptuning_args = self.setup_fake_prompts_vila(
                self.args.batch_size, visual_features,
                first_batch_split_prompts, input_lengths)
            return input_ids, input_lengths, ptuning_args, visual_features
        elif self.model_type == 'phi-3-vision':
            input_ids = input["input_ids"].clone()
            glb_GN = torch.squeeze(self.image_newlines["glb_GN"].clone(), dim=0)
            sub_GN = self.image_newlines["sub_GN"].clone()

            H = visual_features.shape[1]
            C = visual_features.shape[-1]
            #bs*17*12*12*3072
            visual_features = visual_features.view(bs, -1, H, H, C)
            global_img_feature = visual_features[:, 0]  #bs*12*12*3072
            temp_glb_GN = sub_GN.repeat(bs, H, 1, 1)  #bs*12*1*3072
            global_img_feature = torch.cat([global_img_feature, temp_glb_GN],
                                           dim=2).reshape(bs, -1, C)

            crop_visual_features = visual_features[:, 1:]
            patch_sizes = [
                image_size // image.shape[-1]
                for image_size in input["image_sizes"]
            ]
            visual_features = []
            for global_img_feature, crop_visual_feature, patch_size in zip(
                    global_img_feature, crop_visual_features, patch_sizes):
                crop_visual_feature = \
                    crop_visual_feature[:patch_size[0]*patch_size[1]].view(patch_size[0], patch_size[1], H, H, C).permute(0, 2, 1, 3, 4).reshape(patch_size[0]*H, patch_size[1]*H, C)
                temp_sub_GN = torch.squeeze(sub_GN.repeat(
                    1, patch_size[0] * H, 1, 1),
                                            dim=0)
                crop_visual_feature = torch.cat(
                    [crop_visual_feature, temp_sub_GN], dim=1).reshape(-1, C)
                visual_features.append(
                    torch.cat([crop_visual_feature, glb_GN, global_img_feature],
                              dim=0))

            num_img_tokens = [elem.size(0) for elem in visual_features]

            visual_features = torch.cat(visual_features, dim=0)
            input_ids = input_ids.expand(self.args.batch_size,
                                         *input_ids.shape[1:])
            input_ids = self.ptuning_setup_phi3(visual_features, input_ids,
                                                num_img_tokens)
            length = input_ids.shape[1]
        elif self.model_type == 'llava_next':
            visual_features = LlavaNextUtils.rearrange_image_features(
                visual_features, self.image_newlines["image_newline"],
                image_size)
            input_ids = self.ptuning_setup_llava_next(visual_features,
                                                      pre_prompt, post_prompt)
            length = input_ids.shape[1]
        else:
            pre_input_ids = self.tokenizer(pre_prompt,
                                           return_tensors="pt",
                                           padding=True).input_ids
            if post_prompt[0] is not None:
                post_input_ids = self.tokenizer(post_prompt,
                                                return_tensors="pt",
                                                padding=True).input_ids
                if self.model_type == 'video-neva':
                    length = pre_input_ids.shape[1] + post_input_ids.shape[
                        1] + visual_atts.shape[2] * visual_atts.shape[1]
                else:
                    length = pre_input_ids.shape[1] + post_input_ids.shape[
                        1] + visual_atts.shape[1]
            else:
                post_input_ids = None
                length = pre_input_ids.shape[1] + visual_atts.shape[1]

        input_lengths = torch.IntTensor([length] * self.args.batch_size).to(
            torch.int32)

        if self.model_type in [
                'fuyu', 'kosmos-2', 'phi-3-vision', 'llava_next'
        ]:
            return input_ids, input_lengths, [visual_features], visual_features

        input_ids, ptuning_args = self.setup_fake_prompts(
            visual_features, pre_input_ids, post_input_ids, input_lengths)

        return input_ids, input_lengths, ptuning_args, visual_features

    @staticmethod
    def tokenizer_image_token(batch_size,
                              prompt,
                              tokenizer,
                              image_token_index=-200):
        prompt_chunks = [
            tokenizer(chunk).input_ids for chunk in prompt.split("<image>")
        ]

        def insert_separator(X, sep):
            return [
                ele for sublist in zip(X, [sep] * len(X)) for ele in sublist
            ][:-1]

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
        input_ids = input_ids.unsqueeze(0).expand(batch_size, -1)
        return input_ids

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
                 warmup=False):
        if not warmup:
            profiler.start("Generate")

        input_ids, input_lengths, ptuning_args, visual_features = self.preprocess(
            warmup, pre_prompt, post_prompt, image)
        if warmup: return None

        profiler.start("LLM")
        if self.decoder_llm:
            end_id = self.tokenizer.eos_token_id
            if 'opt' in self.model_type and 'blip2' in self.model_type:
                # For BLIP2-OPT, model outputs a "\n" at the end.
                # we avoid it by using newline as the end token
                end_id = self.tokenizer.encode("\n",
                                               add_special_tokens=False)[0]

            ptuning_args[0] = torch.stack([ptuning_args[0]])
            if self.model_type == 'cogvlm':
                input_position_ids = self.prepare_position_ids_for_cogvlm(
                    input_ids)
                batch_size = len(input_ids)
                if not self.use_py_session:
                    prompt_tasks = ",".join(np.arange(batch_size).astype(str))
                    ptuning_args[0] = ptuning_args[0].view(
                        batch_size, ptuning_args[2].cpu().item(), -1)

            output_ids = self.model.generate(
                input_ids,
                input_position_ids=input_position_ids
                if self.model_type == 'cogvlm' else None,
                sampling_config=None,
                prompt_table=ptuning_args[0],
                prompt_tasks=prompt_tasks if not self.use_py_session else None,
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
                output_sequence_lengths=False,
                return_dict=False)
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
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx]:],
                    skip_special_tokens=True)
                for batch_idx in range(self.args.batch_size)
            ]

            stripped_text = [[
                output_beams_list[batch_idx][beam_idx].strip()
                for beam_idx in range(self.args.num_beams)
            ] for batch_idx in range(self.args.batch_size)]
            profiler.stop("Generate")
            return stripped_text
        else:
            profiler.stop("Generate")
            return None

    def get_visual_features(self, image):
        visual_features = {
            'input': image.to(str_dtype_to_torch(self.vision_precision))
        }
        tensor_info = [
            TensorInfo('input', str_dtype_to_trt(self.vision_precision),
                       image.shape)
        ]

        visual_output_info = self.visual_encoder_session.infer_shapes(
            tensor_info)

        visual_outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device=image.device)
            for t in visual_output_info
        }

        ok = self.visual_encoder_session.run(visual_features, visual_outputs,
                                             self.stream.cuda_stream)
        assert ok, "Runtime execution failed for vision encoder session"
        self.stream.synchronize()

        image_embeds = visual_outputs['output']
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        return image_embeds, image_atts

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
            for idx, visual_feature in enumerate(visual_features):
                fake_prompt_id = torch.arange(
                    fake_prompt_counter,
                    fake_prompt_counter + visual_feature.shape[0])
                fake_prompt_counter += visual_feature.shape[0]
                fake_prompt_id = fake_prompt_id.unsqueeze(0)
                input_ids.append(fake_prompt_id)
                # in case no post prompt
                if len(split_input_ids) > idx + 1:
                    input_ids.append(split_input_ids[idx + 1])

        elif batch_size > 1:
            # mode 2: each image have individual prompt, <pre><image><post>
            for idx, visual_feature in enumerate(visual_features):
                input_ids.append(split_input_ids[0])
                fake_prompt_id = torch.arange(
                    fake_prompt_counter,
                    fake_prompt_counter + visual_feature.shape[0])
                fake_prompt_counter += visual_feature.shape[0]
                fake_prompt_id = fake_prompt_id.unsqueeze(0)
                input_ids.append(fake_prompt_id)
                if len(split_input_ids) > 1:
                    input_ids.append(split_input_ids[1])

        input_ids = torch.cat(input_ids, dim=1).contiguous().to(torch.int32)
        input_ids = input_ids.reshape(batch_size, -1)

        if self.decoder_llm or self.runtime_mapping.is_first_pp_rank():
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

        if self.use_py_session:
            # Non-IFB Mode(used in python session): All requests in a batch have their prompt_table concatenated in
            # a shape of (bs*vision_embedding_len, vision_hidden). So only one fake_prompt_id is needed for the
            # entire batch, with values from 0 to bs * vision_embedding_len-1.
            fake_prompt_id = torch.arange(
                self.model_config.vocab_size, self.model_config.vocab_size +
                visual_features.shape[0] * visual_features.shape[1])
            fake_prompt_id = fake_prompt_id.reshape(visual_features.shape[0],
                                                    visual_features.shape[1])
        else:
            # IFB Mode(used in c++ session): Each request's prompt_table is independent and requires a fake_prompt_id
            # for each request, with values ranging from 0 to vision_embedding_len-1.
            fake_prompt_id = torch.arange(
                self.model_config.vocab_size,
                self.model_config.vocab_size + visual_features.shape[1])
            fake_prompt_id = fake_prompt_id.repeat(visual_features.shape[0], 1)

        if 'cogvlm' in self.model_type:
            input_ids = torch.cat(
                [pre_input_ids[:, 0:1], fake_prompt_id, pre_input_ids[:, 1:]],
                dim=1).contiguous().to(torch.int32)
        else:
            if post_input_ids is not None:
                input_ids = [pre_input_ids, fake_prompt_id, post_input_ids]
            else:
                input_ids = [fake_prompt_id, pre_input_ids]
            input_ids = torch.cat(input_ids, dim=1).contiguous().to(torch.int32)

        if self.decoder_llm or self.runtime_mapping.is_first_pp_rank():
            ptuning_args = self.ptuning_setup(visual_features, input_ids,
                                              input_lengths)
        else:
            ptuning_args = [None, None, None]

        return input_ids, ptuning_args

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

    def ptuning_setup_phi3(self, visual_features, input_ids, num_img_tokens):
        fake_prompt_id = torch.arange(
            self.model_config.vocab_size,
            self.model_config.vocab_size + visual_features.shape[0])
        MAX_INPUT_ID = int(1e9)
        positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID),
                                  as_tuple=False)
        idx = 0
        for i, cnt in enumerate(num_img_tokens):
            input_ids[positions[idx, 0], positions[idx, 1]:positions[idx, 1] +
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
                prompt_table = prompt_table.cuda().to(dtype=self.model.dtype)
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
            tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]

    def load_test_image(self):
        if "vila" in self.model_type:
            if self.args.image_path is None:
                img_url = 'https://github.com/Efficient-Large-Model/VILA/raw/main/demo_images/av.png'
                self.args.image_path = img_url
                image = Image.open(
                    requests.get(img_url, stream=True,
                                 timeout=5).raw).convert('RGB')
                return [image] * self.args.batch_size
            else:

                def load_image(image_path):
                    if image_path.startswith("http") or image_path.startswith(
                            "https"):
                        logger.info(f"downloading image from url {image_path}")
                        response = requests.get(image_path, timeout=5)
                        image = Image.open(BytesIO(
                            response.content)).convert("RGB")
                    else:
                        image = Image.open(image_path).convert("RGB")
                    return image

                out = []
                image_paths = self.args.image_path.split(self.args.path_sep)
                for image_path in image_paths:
                    image = load_image(image_path)
                    out.append(image)
                return out
        elif "nougat" in self.model_type:
            filepath = hf_hub_download(
                repo_id="hf-internal-testing/fixtures_docvqa",
                filename="nougat_paper.png",
                repo_type="dataset")
            image = Image.open(filepath)
        elif "fuyu" in self.model_type:
            filepath = hf_hub_download(repo_id="adept/fuyu-8b",
                                       filename="skateboard.png",
                                       repo_type='model')
            image = Image.open(filepath)
        elif "kosmos" in self.model_type:
            img_url = 'https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png'
            image = Image.open(
                requests.get(img_url, stream=True,
                             timeout=5).raw).convert('RGB')
        elif "pix2struct" in self.model_type:
            img_url = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_40963.png'
            image = Image.open(
                requests.get(img_url, stream=True,
                             timeout=5).raw).convert('RGB')
        elif "video-neva" in self.model_type:
            image = self.args.video_path
        else:
            img_url = self.args.image_path
            if img_url is None:
                img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'

            if img_url.startswith("http") or img_url.startswith("https"):
                image = Image.open(
                    requests.get(img_url, stream=True,
                                 timeout=5).raw).convert('RGB')
            else:
                image = Image.open(img_url).convert("RGB")

        return image

    def setup_inputs(self, input_text, raw_image):
        from torchvision import transforms
        if 'blip2' in self.model_type:
            from transformers import Blip2Processor
            processor = Blip2Processor.from_pretrained(self.args.hf_model_dir)
            image = processor(raw_image, input_text,
                              return_tensors="pt")['pixel_values']

            if input_text is None:
                input_text = "Question: which city is this? Answer:"

            pre_prompt = input_text
            post_prompt = None
        elif 'nougat' in self.model_type:
            from transformers import NougatProcessor
            processor = NougatProcessor.from_pretrained(self.args.hf_model_dir)
            image = processor(raw_image, return_tensors="pt")['pixel_values']

            # Nougat doesn't need text prompt (mBART use single token to start generation), just leave a dummy one here
            if input_text is None:
                input_text = "Question: which city is this? Answer:"

            pre_prompt = input_text
            post_prompt = None
        elif 'cogvlm' in self.model_type:
            image_size = 490
            dtype = torch.bfloat16
            transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])
            image = transform(raw_image).to(dtype).unsqueeze(0)

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
            processor = AutoProcessor.from_pretrained(self.args.hf_model_dir,
                                                      trust_remote_code=True)
            image = processor(text=prompt,
                              images=raw_image,
                              return_tensors="pt")
        elif self.model_type == "pix2struct":
            image_processor = AutoProcessor.from_pretrained(
                self.args.hf_model_dir)
            if input_text is None:
                input_text = ""
            inputs = image_processor(
                images=raw_image,
                text=input_text,
                return_tensors="pt",
            )
            image = inputs['flattened_patches']
            image = image.expand(self.args.batch_size, -1, -1).contiguous()
            pre_prompt = ""
            post_prompt = None
        elif self.model_type == "neva":
            image_size = 384
            dtype = torch.float32
            transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            image = transform(raw_image).to(dtype).unsqueeze(0)

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

            processor = AutoProcessor.from_pretrained(self.args.hf_model_dir,
                                                      trust_remote_code=True)
            image = processor(text=prompt,
                              images=raw_image,
                              return_tensors="pt")

        elif self.model_type in ['llava', 'vila', 'fuyu', 'kosmos-2']:
            # LLaVA and VILA
            if self.model_type == "llava":
                pre_prompt = "USER:\n"
                if input_text is None:
                    input_text = "Question: which city is this? Answer:"
            elif self.model_type == "vila":
                pre_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
                if input_text is None:
                    input_text = "<image>\n Please elaborate what you see in the images?"
            elif self.model_type == 'fuyu':
                pre_prompt = "Describe this image:"
                if input_text is None:
                    input_text = "Answer the following VQAv2 question based on the image: How many people are in the image?\n"
            elif self.model_type == "kosmos-2":
                pre_prompt = ""
                if input_text is None:
                    input_text = "<grounding>An image of"

            if self.model_type not in ['fuyu', 'kosmos-2']:
                post_prompt = input_text + " ASSISTANT:"
            else:
                post_prompt = None

            if self.model_type == "vila":
                sys.path.append(self.args.hf_model_dir + "/../VILA")
                from llava.model import LlavaLlamaConfig  # noqa
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    self.args.hf_model_dir,
                    device_map='auto',
                    trust_remote_code=True,
                )
                vision_tower = model.get_vision_tower()
                image_processor = vision_tower.image_processor
                from llava.mm_utils import process_images
                image = process_images(raw_image, image_processor,
                                       model.config).to(model.device,
                                                        dtype=torch.float16)
            else:
                processor = AutoProcessor.from_pretrained(
                    self.args.hf_model_dir)
                if self.model_type in ['fuyu', 'kosmos-2']:
                    image = processor(text=input_text,
                                      images=raw_image,
                                      return_tensors='pt')
                else:
                    image = processor(text=input_text,
                                      images=raw_image,
                                      return_tensors="pt")['pixel_values']

        # Repeat inputs to match batch size
        pre_prompt = [pre_prompt] * self.args.batch_size
        post_prompt = [post_prompt] * self.args.batch_size
        if self.model_type not in [
                'fuyu', 'pix2struct', 'kosmos-2', 'vila', 'phi-3-vision',
                'llava_next'
        ]:
            if image.dim() == 5:
                image = image.expand(self.args.batch_size, -1, -1, -1,
                                     -1).contiguous()
            else:
                image = image.expand(self.args.batch_size, -1, -1,
                                     -1).contiguous()
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

        return input_text, pre_prompt, post_prompt, image, decoder_input_ids

    def run(self, input_text, input_image, max_new_tokens):
        input_text, pre_prompt, post_prompt, processed_image, decoder_input_ids = self.setup_inputs(
            input_text, input_image)

        output_text = self.generate(pre_prompt,
                                    post_prompt,
                                    processed_image,
                                    decoder_input_ids,
                                    max_new_tokens,
                                    warmup=False)

        return input_text, output_text
