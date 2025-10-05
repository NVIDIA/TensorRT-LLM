# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import copy
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from PIL import Image

from tensorrt_llm._torch.models.checkpoints import NemotronHHfWeightMapper
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ...inputs import (BaseMultimodalInputProcessor, ExtraProcessedInputs,
                       InputProcessor, MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       compute_retention_mask, register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (find_input_mm_embeds, fuse_input_embeds,
                                        get_multimodal_embeddings)
from .modeling_radio import RADIOVisionModel
from .modeling_utils import register_auto_model

VIDEO_PRUNING_RATIO = float(os.getenv("TLLM_VIDEO_PRUNING_RATIO", "0"))


# Make this a runtime lookup rather than a module-wide constant for easier unit testing.
def _is_disagg() -> bool:
    return os.getenv("TLLM_MULTIMODAL_DISAGGREGATED", "0") == "1"


class SquaredReLU(nn.Module):

    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2)


# Source codes are from NemotronH_Nano_VL_V2 modeling.py.
class NanoV2VLVisionEncoder(transformers.PreTrainedModel):

    def __init__(self,
                 model_config: ModelConfig[transformers.PretrainedConfig]):
        config = model_config.pretrained_config
        super().__init__(config)
        self.image_size = config.force_image_size
        self.patch_size = config.patch_size
        self.num_image_token = int((self.image_size // self.patch_size)**2 *
                                   (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.spatial_merge_size = int(self.patch_size / self.downsample_ratio)
        self.ps_version = config.ps_version  # Pixel shuffle version.

        # Construct the vision projection.
        self.vit_hidden_size = config.vit_hidden_size
        self.vision_projection_hidden_size = config.projector_hidden_size
        self.llm_hidden_size = config.llm_config.hidden_size
        self.mlp1 = nn.Sequential(
            nn.RMSNorm(self.vit_hidden_size * int(1 / self.downsample_ratio)**2,
                       eps=config.llm_config.rms_norm_eps,
                       dtype=config.torch_dtype),
            nn.Linear(self.vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      self.vision_projection_hidden_size,
                      bias=False,
                      dtype=config.torch_dtype), SquaredReLU(),
            nn.Linear(self.vision_projection_hidden_size,
                      self.llm_hidden_size,
                      bias=False,
                      dtype=config.torch_dtype))

        # Construct the vision encoder.
        vision_model_config = copy.deepcopy(model_config)
        vision_model_config.pretrained_config = vision_model_config.pretrained_config.vision_config
        self.vision_model = RADIOVisionModel(vision_model_config,
                                             disable_quantization=True)

    def load_weights(self, weights):
        # Load mlp1 weights.
        mlp1_weights = {
            k.replace('mlp1.', ''): v
            for k, v in weights.items() if k.startswith('mlp1.')
        }
        self.mlp1.load_state_dict(mlp1_weights, strict=True)

        # Load vision encoder weights.
        vision_encoder_weights = {
            k.replace('vision_model.', ''): v
            for k, v in weights.items() if k.startswith('vision_model.')
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
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            logger.warning(
                "In ps_version 'v1', the height and width have not been swapped back, "
                'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        vit_embeds = self.vision_model(pixel_values)
        # Down-sampling and projection.
        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def apply_evs(
            self, mm_embedding: List[torch.Tensor],
            multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        """Apply EVS to the multimodal embedding."""
        if VIDEO_PRUNING_RATIO <= 0:
            return mm_embedding

        modality_types = [
            multimodal_param.multimodal_data['modality_type']
            for multimodal_param in multimodal_params
        ]
        video_size_list = [
            multimodal_param.multimodal_data['video_size']
            for multimodal_param in multimodal_params
        ]
        mm_embedding_evs = []
        # Iterate over batch.
        for modality_type, mm_embed, video_sizes in zip(modality_types,
                                                        mm_embedding,
                                                        video_size_list):
            if modality_type == "video":
                # Iterate over each video in the batch.
                start_idx, mm_embed_list = 0, []
                for video_size in video_sizes:
                    partial_mm_embed = mm_embed[start_idx:start_idx +
                                                video_size[0]]
                    retention_mask = compute_retention_mask(
                        video_embeds=partial_mm_embed,
                        video_size=video_size,
                        spatial_merge_size=self.spatial_merge_size,
                        q=VIDEO_PRUNING_RATIO,
                        flatten_output=False,
                    ).flatten(start_dim=1)
                    start_idx += video_size[0]
                    partial_mm_embed = partial_mm_embed[retention_mask]
                    mm_embed_list.append(partial_mm_embed)
                mm_embedding_evs.append(torch.cat(mm_embed_list, dim=0))
            else:
                mm_embedding_evs.append(mm_embed)
        return mm_embedding_evs

    def forward(self, multimodal_params: List[MultimodalParams]):
        mm_embedding = []
        # Batch data.
        pixel_values = [
            multimodal_param.multimodal_data["pixel_values"]
            for multimodal_param in multimodal_params
        ]
        batched_pixel_values = torch.cat(pixel_values, dim=0)
        # -> [num_patches, channel, height, width]
        patch_list = [
            multimodal_param.multimodal_data["num_patches"]
            for multimodal_param in multimodal_params
        ]
        batched_num_patches = torch.cat(patch_list, dim=0).tolist()
        # -> list of[num_patches1, num_patches2, ...]
        batched_image_embeds = self.extract_feature(batched_pixel_values)
        # -> [num_patches, num_image_token, hidden_size]
        mm_embedding = torch.split(batched_image_embeds,
                                   batched_num_patches,
                                   dim=0)

        mm_embedding = self.apply_evs(mm_embedding, multimodal_params)

        mm_embedding = [
            m.reshape(-1, self.llm_hidden_size) for m in mm_embedding
        ]
        # -> list of [num_patches*num_image_token, hidden_size]
        return mm_embedding


class NanoV2VLInputProcessor(BaseMultimodalInputProcessor, InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: transformers.PretrainedConfig,
                 tokenizer: transformers.AutoTokenizer,
                 trust_remote_code: bool = True):
        if not trust_remote_code:
            raise ValueError("trust_remote_code must be True for NanoV2VL")

        self.model_config = model_config
        self.image_size = model_config.force_image_size
        self.patch_size = model_config.patch_size
        self.downsample_ratio = model_config.downsample_ratio
        self.img_context_token_id = model_config.img_context_token_id
        self.num_image_token = int((self.image_size // self.patch_size)**2 *
                                   (self.downsample_ratio**2))

        self.device = 'cpu'

        self.tokenizer = tokenizer
        self.use_fast = True
        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=self.use_fast)

        self.processor = transformers.AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=self.use_fast)

        self.img_context_token = model_config.img_context_token
        self.video_context_token = model_config.video_context_token
        self.img_start_token = model_config.img_start_token
        self.img_end_token = model_config.img_end_token
        self.dtype = model_config.torch_dtype

    def get_vocab_size(self):
        return self.model_config.llm_config.vocab_size

    def get_mm_special_token_ids(self) -> torch.Tensor:
        " Return multimodal special token ids for NanoV2VL. "
        # TODO: Hardcoded for now, need extract from model config later.
        return torch.tensor([131073, 131074])

    def get_mm_token_ids(self):
        return torch.tensor([self.img_context_token_id], dtype=torch.int32)

    def get_num_tokens_per_image(
        self,
        *,
        image: Image.Image,
        **kwargs,
    ):

        def _get_internvl_target_ratios(
            min_num: int,
            max_num: int,
        ) -> list[tuple[int, int]]:
            target_ratios = {(i, j)
                             for n in range(min_num, max_num + 1)
                             for i in range(1, n + 1)
                             for j in range(1, n + 1)
                             if min_num <= i * j <= max_num}
            return sorted(target_ratios, key=lambda x: x[0] * x[1])

        def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width,
                                       height, image_size):
            best_factor = float('-inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                factor_based_on_area_n_ratio = min(
                    (ratio[0] * ratio[1] * image_size * image_size) / area,
                    0.6) * min(target_aspect_ratio / aspect_ratio,
                               aspect_ratio / target_aspect_ratio)
                if factor_based_on_area_n_ratio > best_factor:
                    best_factor = factor_based_on_area_n_ratio
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

        image_height = image.height
        image_width = image.width
        target_ratios = _get_internvl_target_ratios(
            1, self.processor.max_num_tiles)
        blocks = _calculate_targets(image_width, image_height, target_ratios,
                                    self.image_size)
        if self.processor.use_thumbnail and blocks != 1:
            blocks += 1
        num_image_tokens = self.num_image_token * blocks
        return num_image_tokens

    def get_num_tokens_per_video(
        self,
        *,
        video: List[Image.Image],
        video_pruning_ratio: Optional[float] = None,
        **kwargs,
    ):
        # Use VIDEO_PRUNING_RATIO if not explicitly provided
        if video_pruning_ratio is None:
            video_pruning_ratio = VIDEO_PRUNING_RATIO

        num_frames = len(video)

        if video_pruning_ratio > 0:
            num_tokens_per_frame = self.get_num_tokens_per_image(image=video[0],
                                                                 **kwargs)
            num_tokens_per_frame_list = [num_tokens_per_frame] * num_frames

            # Total patches across all frames
            total_num_tokens_base = sum(num_tokens_per_frame_list)

            # Calculate total desired tokens after pruning
            desired_num_tokens = int(total_num_tokens_base *
                                     (1.0 - video_pruning_ratio))

            # Calculate tokens for each frame except the last
            existing_num_tokens = 0
            for i in range(num_frames - 1):
                feature_size = num_tokens_per_frame_list[i]
                feature_size = int(feature_size * (1.0 - video_pruning_ratio))
                existing_num_tokens += feature_size

            # Last frame gets the remaining tokens
            last_frame_tokens = desired_num_tokens - existing_num_tokens
            num_total_tokens = existing_num_tokens + last_frame_tokens

            # Add start and end tokens for each frame
            num_total_tokens += num_frames * 2
        else:
            # No pruning - sum tokens for all frames
            num_total_tokens = sum(
                self.get_num_tokens_per_image(
                    image=frame, video_pruning_ratio=None, **kwargs)
                for frame in video)
            # Add start and end tokens for each frame
            num_total_tokens += num_frames * 2

        return num_total_tokens

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
        images = mm_data.get("image", None)
        videos = mm_data.get("video", None)
        if images is not None and videos is not None:
            raise ValueError(
                "NanoV2VL does not support both images and videos in the same prompt yet."
            )

        if images is None and videos is None:
            input_ids = self.tokenizer.encode(text_prompt,
                                              add_special_tokens=False,
                                              return_tensors="pt")
            return input_ids[0].to(torch.int32).tolist(), {}

        modality_type = None
        if images is not None:
            modality_type = "image"
            # Processing for multimodal data.
            processed_images = self.processor(images=images,
                                              return_tensors='pt').to(
                                                  self.device)
            # Insert enough special tokens for image embedding.
            parts = text_prompt.split(self.img_context_token)
            if len(parts) - 1 != len(processed_images['num_patches']):
                raise ValueError(
                    f"Number of {self.img_context_token} tokens ({len(parts) - 1}) doesn't match num_patches_list length ({len(processed_images['num_patches'])})"
                )
            processed_query = parts[0]
            for num_patches, part in zip(processed_images['num_patches'],
                                         parts[1:]):
                feature_size = num_patches * self.num_image_token
                image_repl = self.img_start_token + self.img_context_token * feature_size + self.img_end_token
                processed_query += image_repl + part
        elif videos is not None:
            modality_type = "video"
            num_videos = len(videos)
            num_patches_list = []
            pixel_values_list = []
            video_size_list = []
            parts = text_prompt.split(self.video_context_token)
            if len(parts) - 1 != num_videos:
                raise ValueError(
                    f"Number of {self.video_context_token} tokens ({len(parts) - 1}) doesn't match number of videos ({num_videos})"
                )
            # Process videos one by one to get correct processed_query.
            processed_query = ""
            for video_index, video in enumerate(videos):
                # Processing for multimodal data.
                processed_images = self.processor(images=video,
                                                  return_tensors='pt').to(
                                                      self.device)
                t, _, h, w = processed_images['pixel_values'].shape
                num_patches_list.append(processed_images['num_patches'])
                pixel_values_list.append(processed_images['pixel_values'])
                video_size_list.append([t, h, w])

                # Processing the text prompt.
                processed_query += parts[video_index]
                total_num_patches = sum(processed_images['num_patches'])
                desired_num_tokens = int(self.num_image_token *
                                         total_num_patches *
                                         (1.0 - VIDEO_PRUNING_RATIO))
                existing_num_tokens = 0
                for num_patches in processed_images['num_patches'][:-1]:
                    feature_size = num_patches * self.num_image_token
                    feature_size = int(feature_size *
                                       (1.0 - VIDEO_PRUNING_RATIO))
                    existing_num_tokens += feature_size
                    image_repl = self.img_start_token + self.img_context_token * feature_size + self.img_end_token
                    processed_query += image_repl
                # Special handling for the last patch of image tokens.
                image_repl = self.img_start_token + self.img_context_token * (
                    desired_num_tokens -
                    existing_num_tokens) + self.img_end_token
                processed_query += image_repl
            processed_query += parts[num_videos]
            processed_images['num_patches'] = torch.tensor(
                [sum(num_patches) for num_patches in num_patches_list])
            processed_images['pixel_values'] = torch.cat(pixel_values_list,
                                                         dim=0)
            processed_images['video_size'] = video_size_list

        input_ids = self.tokenizer.encode(processed_query,
                                          add_special_tokens=False,
                                          return_tensors="pt")

        # Will package inputs for language model forward in AGGREGATE mode.
        multimodal_data = {}
        multimodal_data['pixel_values'] = processed_images['pixel_values'].to(
            self.dtype)
        multimodal_data['num_patches'] = processed_images['num_patches'].sum(
            dim=0, keepdim=True)
        multimodal_data['modality_type'] = modality_type
        multimodal_data['video_size'] = processed_images[
            'video_size'] if modality_type == "video" else None
        return input_ids[0].to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


@register_auto_model("NemotronH_Nano_VL_V2")
@register_input_processor(
    NanoV2VLInputProcessor,
    model_type="NemotronH_Nano_VL_V2",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<image>",
            "video": "<video>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        placeholders_separator="",
    ))
class NemotronH_Nano_VL_V2(transformers.PreTrainedModel):

    _supports_flash_attn_2 = True

    def __init__(self, model_config: ModelConfig):
        if _is_disagg():
            raise ValueError(
                "NanoV2VL does not support disaggregated inference yet.")

        config = model_config.pretrained_config
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        if not _is_disagg():
            self.vision_encoder = NanoV2VLVisionEncoder(model_config).eval()

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = llm_model_config.pretrained_config.llm_config
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.vocab_size = llm_model_config.pretrained_config.vocab_size
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        self.img_context_token_id = config.img_context_token_id
        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):
        # Load vision encoder weights.
        self.vision_encoder.load_weights(weights)

        # Load language model weights.
        filtered_weights = {
            k.replace('language_model.', ''): v
            for k, v in weights.items() if k.startswith('language_model.')
        }
        weight_mapper = NemotronHHfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)
        self.llm.load_weights(filtered_weights, weight_mapper=weight_mapper)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

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
                mm_embedding = get_multimodal_embeddings(
                    encoder_forward_fn=self.vision_encoder.forward,
                    multimodal_params=multimodal_params[:num_context_requests])
            else:
                raise NotImplementedError(
                    "Nano-V2-VLM does not support disaggregated inference yet. Please unset "
                    f"the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )
            mm_embedding = find_input_mm_embeds(
                mm_embedding, multimodal_params[:num_context_requests])
        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embedding,
            mm_token_ids=torch.tensor([self.img_context_token_id],
                                      dtype=torch.int32),
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
