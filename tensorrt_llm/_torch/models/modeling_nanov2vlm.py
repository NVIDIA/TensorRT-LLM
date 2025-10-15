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
                       compute_retained_tokens_count, compute_retention_mask,
                       register_input_processor)
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
IMAGE_START_TOKEN_ID = 131073
IMAGE_END_TOKEN_ID = 131074


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
        self.video_pruning_ratio = VIDEO_PRUNING_RATIO

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
        multimodal_params: List[MultimodalParams]
    ) -> Tuple[List[torch.Tensor], Optional[List[List[int]]]]:
        """Apply EVS to the multimodal embedding."""
        if self.video_pruning_ratio <= 0:
            return mm_embedding, None

        modality_types = [
            multimodal_param.multimodal_data['modality_type']
            for multimodal_param in multimodal_params
        ]
        video_size_list = [
            multimodal_param.multimodal_data['video_size']
            for multimodal_param in multimodal_params
        ]
        mm_embedding_evs = []
        num_tokens_in_videos = []
        # Iterate over batch.
        for modality_type, mm_embed, video_sizes in zip(modality_types,
                                                        mm_embedding,
                                                        video_size_list):
            if modality_type == "video":
                # Iterate over each video in the batch.
                start_idx, mm_embed_list, num_tokens_per_video = 0, [], []
                for video_size in video_sizes:
                    t, p, ih, iw = video_size
                    partial_mm_embed = mm_embed[start_idx:start_idx + t * p]
                    # -> [num_frames * num_patches_per_frame, h*w, hidden_size]

                    # Need to expose temporal dimension for EVS.
                    _, wh, hidden_size = partial_mm_embed.shape
                    reshaped_partial_mm_embed = partial_mm_embed.reshape(
                        t, p, wh, hidden_size).reshape(t, p * wh, hidden_size)

                    # -> [num_frames, num_patches_per_frame*h*w, hidden_size]
                    original_retention_mask = compute_retention_mask(
                        video_embeds=reshaped_partial_mm_embed,
                        video_size=(t, p * ih, iw),
                        spatial_merge_size=self.spatial_merge_size,
                        pruning_ratio=self.video_pruning_ratio,
                        flatten_output=False,
                    ).flatten(start_dim=1)
                    # -> [num_frames, num_patches_per_frame*h*w]
                    num_tokens_per_frame = original_retention_mask.sum(dim=1)
                    retention_mask = original_retention_mask.reshape(
                        t, p, wh).reshape(t * p, wh)
                    # -> [num_frames * num_patches_per_frame, h*w]
                    start_idx += t * p
                    partial_mm_embed = partial_mm_embed[retention_mask]
                    mm_embed_list.append(partial_mm_embed)
                    num_tokens_per_video.append(num_tokens_per_frame)
                mm_embedding_evs.append(torch.cat(mm_embed_list, dim=0))
                num_tokens_in_videos.append(
                    torch.cat(num_tokens_per_video, dim=0))
            else:
                mm_embedding_evs.append(mm_embed)
        return mm_embedding_evs, num_tokens_in_videos if len(
            num_tokens_in_videos) > 0 else None

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

        mm_embedding, num_tokens_in_videos = self.apply_evs(
            mm_embedding, multimodal_params)

        mm_embedding = [
            m.reshape(-1, self.llm_hidden_size) for m in mm_embedding
        ]
        # -> list of [num_patches*num_image_token, hidden_size]
        return mm_embedding, num_tokens_in_videos


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
        self.spatial_merge_size = int(self.patch_size / self.downsample_ratio)
        self.img_context_token_id = model_config.img_context_token_id
        self.num_image_token = int((self.image_size // self.patch_size)**2 *
                                   (self.downsample_ratio**2))
        self.video_pruning_ratio = VIDEO_PRUNING_RATIO

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
        return torch.tensor([IMAGE_START_TOKEN_ID, IMAGE_END_TOKEN_ID])

    def get_mm_token_ids(self):
        return torch.tensor([self.img_context_token_id], dtype=torch.int32)

    def get_num_tokens_per_image(
        self,
        *,
        image: Image.Image,
        **kwargs,
    ):
        # The logic is copied and modified from HuggingFace ImageProcessor.

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

        image_height = image.height
        image_width = image.width
        target_ratios = _get_internvl_target_ratios(
            1, self.processor.max_num_tiles)
        blocks = _calculate_targets(image_width, image_height, target_ratios,
                                    self.image_size)
        if self.processor.use_thumbnail and blocks != 1:
            blocks += 1
        num_image_tokens = self.num_image_token * blocks
        # Add special tokens.
        num_image_tokens += len(self.get_mm_special_token_ids())
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
            video_pruning_ratio = self.video_pruning_ratio

        num_frames = len(video)
        if video_pruning_ratio > 0:
            num_tokens_per_frame = self.get_num_tokens_per_image(image=video[0],
                                                                 **kwargs)
            num_image_tokens_per_frame = num_tokens_per_frame - len(
                self.get_mm_special_token_ids())
            blocks = num_image_tokens_per_frame // self.num_image_token
            video_size = (num_frames, blocks * self.image_size, self.image_size)
            num_total_tokens = compute_retained_tokens_count(
                video_size=video_size,
                spatial_merge_size=self.spatial_merge_size,
                pruning_ratio=video_pruning_ratio,
            )
            # Add special tokens for each frame.
            num_total_tokens += num_frames * len(
                self.get_mm_special_token_ids())
        else:
            # No pruning - sum tokens for all frames
            num_total_tokens = sum(
                self.get_num_tokens_per_image(
                    image=frame, video_pruning_ratio=None, **kwargs)
                for frame in video)
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
                num_frames = len(video)
                processed_images = self.processor(images=video,
                                                  return_tensors='pt').to(
                                                      self.device)
                t, _, h, w = processed_images['pixel_values'].shape
                num_patches_list.append(processed_images['num_patches'])
                pixel_values_list.append(processed_images['pixel_values'])
                video_size_list.append([num_frames, t // num_frames, h, w])

                # Processing the text prompt.
                processed_query += parts[video_index]
                if self.video_pruning_ratio > 0:
                    desired_num_tokens = compute_retained_tokens_count(
                        video_size=(num_frames, t // num_frames * h, w),
                        spatial_merge_size=self.spatial_merge_size,
                        pruning_ratio=self.video_pruning_ratio,
                    )
                    # It is dummy tokens and will be adjusted in VisionEncoder after applied EVS.
                    # We need to know the length of the full input ids ahead, to make Mamba-mixer work.
                    num_tokens_per_frame = [desired_num_tokens
                                            ] + [0] * (num_frames - 1)
                else:
                    num_tokens_per_frame = [
                        num_patches * self.num_image_token
                        for num_patches in processed_images['num_patches']
                    ]

                # Prepare video prompts.
                # TODO: add metadata like FPS and frame_index when such metadata is available.
                image_repl = "This is a video:\n"
                for idx, num_tokens in enumerate(num_tokens_per_frame):
                    image_repl += f"Frame {idx + 1}: "
                    image_repl += self.img_start_token + self.img_context_token * num_tokens + self.img_end_token
                    image_repl += "\n"
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
        self.video_context_token_id = config.video_context_token_id
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

    def update_input_ids(self, input_ids, num_tokens_in_videos):
        """Update input_ids in videos if EVS is applied.

        Note: it is not compatible with chunked_prefill for now.
        """
        image_start_token_id = IMAGE_START_TOKEN_ID
        image_end_token_id = IMAGE_END_TOKEN_ID

        # Find the positions of start and end image tokens in input_ids
        image_start_token_mask = (input_ids == image_start_token_id)
        image_start_token_indices = torch.nonzero(image_start_token_mask,
                                                  as_tuple=False).flatten()
        image_end_token_mask = (input_ids == image_end_token_id)
        image_end_token_indices = torch.nonzero(image_end_token_mask,
                                                as_tuple=False).flatten()

        num_tokens_in_videos = torch.cat(num_tokens_in_videos, dim=0)
        results_ids = input_ids.clone()
        src_idx = 0
        target_idx1, target_idx2 = 0, 0
        for start_idx, end_idx, num_tokens in zip(
                image_start_token_indices.tolist(),
                image_end_token_indices.tolist(),
                num_tokens_in_videos.tolist()):
            if num_tokens == 0:
                continue
            target_idx1 = target_idx2
            target_idx2 = target_idx1 + start_idx - src_idx
            results_ids[target_idx1:target_idx2] = input_ids[src_idx:start_idx]

            mm_tokens = torch.tensor([image_start_token_id] +
                                     [self.img_context_token_id] * num_tokens +
                                     [image_end_token_id],
                                     dtype=input_ids.dtype,
                                     device=input_ids.device)
            target_idx1 = target_idx2
            target_idx2 = target_idx2 + num_tokens + 2
            results_ids[target_idx1:target_idx2] = mm_tokens

            src_idx = end_idx + 1
        results_ids[src_idx:] = input_ids[src_idx:]
        return results_ids

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
                mm_embedding, num_tokens_in_videos = get_multimodal_embeddings(
                    encoder_forward_fn=self.vision_encoder.forward,
                    multimodal_params=multimodal_params[:num_context_requests])
            else:
                raise NotImplementedError(
                    "Nano-V2-VLM does not support disaggregated inference yet. Please unset "
                    f"the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )
            # Adjust input_ids in videos if EVS is applied.
            if num_tokens_in_videos is not None:
                input_ids = self.update_input_ids(input_ids,
                                                  num_tokens_in_videos)

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
