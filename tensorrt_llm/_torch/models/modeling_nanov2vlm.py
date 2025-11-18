# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import copy
import os
from typing import Any, Dict, List, Optional, Tuple

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
# Set max_num_tiles to 1 for video modality, to match the training behavior.
VIDEO_MAX_NUM_TILES = 1


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
        # Use micro-batch to avoid OOM, especially for long video inputs.
        micro_batch_size = 128
        n = pixel_values.shape[0]
        vit_embeds_lst = []
        for i in range(0, n, micro_batch_size):
            micro_batch_pixel_values = pixel_values[i:i + micro_batch_size]
            vit_embeds = self.vision_model(micro_batch_pixel_values)
            # Down-sampling and projection.
            h = w = int(vit_embeds.shape[1]**0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds,
                                            scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                            vit_embeds.shape[-1])
            vit_embeds = self.mlp1(vit_embeds)
            vit_embeds_lst.append(vit_embeds)
        vit_embeds = torch.cat(vit_embeds_lst, dim=0)
        return vit_embeds

    def apply_evs_per_video(
            self, mm_embed: torch.Tensor,
            video_sizes: List[Tuple]) -> Tuple[torch.Tensor, List[int]]:
        """Apply EVS to the multimodal embedding for a single video."""
        start_idx, mm_embed_list, num_tokens_per_video = 0, [], []
        for video_size in video_sizes:
            # Fetch mm_embed correctly for the flattened temporal/patches dimension.
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
            retention_mask = original_retention_mask.reshape(t, p, wh).reshape(
                t * p, wh)
            # -> [num_frames * num_patches_per_frame, h*w]

            # Skip by temporal/patch dimension.
            start_idx += t * p

            partial_mm_embed = partial_mm_embed[retention_mask]
            mm_embed_list.append(partial_mm_embed)
            num_tokens_per_video.append(num_tokens_per_frame)
        mm_embed = torch.cat(mm_embed_list, dim=0)
        num_tokens_in_video = torch.cat(num_tokens_per_video, dim=0)
        return mm_embed, num_tokens_in_video

    def apply_evs(
        self, mm_embedding: List[torch.Tensor],
        multimodal_data_lst: List[Dict[str, Any]]
    ) -> Tuple[List[torch.Tensor], Optional[List[List[int] | None]]]:
        """Apply EVS to the multimodal embedding."""
        # Skip EVS if pruning ratio is 0.
        if self.video_pruning_ratio <= 0:
            return mm_embedding, None

        modality_types = [
            multimodal_data['modality_type']
            for multimodal_data in multimodal_data_lst
        ]
        # Skip EVS if there is no video modality.
        if "video" not in modality_types:
            return mm_embedding, None

        video_size_list = [
            multimodal_data[modality_type]["video_size"] for modality_type,
            multimodal_data in zip(modality_types, multimodal_data_lst)
        ]
        mm_embedding_evs = []
        num_tokens_in_videos = []
        # Iterate over batch.
        for modality_type, mm_embed, video_sizes in zip(modality_types,
                                                        mm_embedding,
                                                        video_size_list):
            # Skip EVS if modality type is not video.
            if modality_type != "video":
                mm_embedding_evs.append(mm_embed)
                num_tokens_in_videos.append(None)
            else:
                # Iterate over each video in the batch.
                mm_embed, num_tokens_per_frames = self.apply_evs_per_video(
                    mm_embed, video_sizes)
                mm_embedding_evs.append(mm_embed)
                num_tokens_in_videos.append(num_tokens_per_frames)
        return mm_embedding_evs, num_tokens_in_videos

    def forward(
        self, multimodal_params: List[MultimodalParams]
    ) -> Tuple[List[torch.Tensor], List[List[int] | None]]:
        mm_embedding = []
        multimodal_data_lst = [
            multimodal_param.multimodal_data
            for multimodal_param in multimodal_params
        ]
        modality_types = [
            multimodal_data['modality_type']
            for multimodal_data in multimodal_data_lst
        ]
        # Batch data.
        pixel_values = [
            multimodal_data[modality_type]["pixel_values"] for modality_type,
            multimodal_data in zip(modality_types, multimodal_data_lst)
        ]
        batched_pixel_values = torch.cat(pixel_values, dim=0)
        # -> [num_patches, channel, height, width]
        patch_list = [
            multimodal_data[modality_type]["num_patches"] for modality_type,
            multimodal_data in zip(modality_types, multimodal_data_lst)
        ]
        batched_num_patches = torch.cat(patch_list, dim=0).tolist()
        # -> list of[num_patches1, num_patches2, ...]
        batched_image_embeds = self.extract_feature(batched_pixel_values)
        # -> [num_patches, num_image_token, hidden_size]
        mm_embedding = torch.split(batched_image_embeds,
                                   batched_num_patches,
                                   dim=0)

        mm_embedding, num_tokens_in_videos = self.apply_evs(
            mm_embedding, multimodal_data_lst)

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
        self.image_start_token_id = self.tokenizer.encode(
            self.img_start_token, add_special_tokens=False)[0]
        self.image_end_token_id = self.tokenizer.encode(
            self.img_end_token, add_special_tokens=False)[0]

    def get_vocab_size(self):
        return self.model_config.llm_config.vocab_size

    def get_mm_special_token_ids(self) -> torch.Tensor:
        " Return multimodal special token ids for NanoV2VL. "
        return torch.tensor(
            [self.image_start_token_id, self.image_end_token_id])

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
        if 'max_num_tiles' in kwargs:
            max_num_tiles = kwargs['max_num_tiles']
        else:
            max_num_tiles = self.processor.max_num_tiles
        target_ratios = _get_internvl_target_ratios(1, max_num_tiles)
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
            num_tokens_per_frame = self.get_num_tokens_per_image(
                image=video[0],
                max_num_tiles=VIDEO_MAX_NUM_TILES,
                **kwargs,
            )
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
                    image=frame,
                    video_pruning_ratio=None,
                    max_num_tiles=VIDEO_MAX_NUM_TILES,
                    **kwargs,
                ) for frame in video)
        return num_total_tokens

    def _process_images(
            self, images: List[Image.Image | torch.Tensor],
            text_prompt: str) -> Tuple[Dict[str, Any], torch.Tensor]:
        # Multiple images can be processed in one call.
        processed_images = self.processor(images=images,
                                          return_tensors='pt').to(self.device)

        # Prepare text prompt for image modality.
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

        input_ids = self.tokenizer.encode(processed_query,
                                          add_special_tokens=False,
                                          return_tensors="pt")
        return processed_images, input_ids

    def _process_videos_frames(
            self,
            videos: List[List[Image.Image | torch.Tensor]]) -> Dict[str, Any]:
        num_patches_list = []
        pixel_values_list = []
        video_size_list = []
        for video in videos:
            num_frames = len(video)

            # Use VIDEO_MAX_NUM_TILES for video modality to match the training behaviors.
            orig_max_num_tiles = self.processor.max_num_tiles
            self.processor.max_num_tiles = VIDEO_MAX_NUM_TILES
            processed_images = self.processor(
                images=video, return_tensors='pt').to(self.device)
            self.processor.max_num_tiles = orig_max_num_tiles

            t, _, h, w = processed_images['pixel_values'].shape
            num_patches_list.append(processed_images['num_patches'])
            pixel_values_list.append(processed_images['pixel_values'])
            video_size_list.append([num_frames, t // num_frames, h, w])

        processed_images['num_patches'] = torch.tensor(
            [sum(num_patches) for num_patches in num_patches_list])
        processed_images['pixel_values'] = torch.cat(pixel_values_list, dim=0)
        processed_images['video_size'] = video_size_list
        return processed_images

    def _get_frame_separators(
            self, video_size_lst: List[Tuple],
            video_metadatas: List[Dict[str, Any] | None]) -> List[List[str]]:

        # Process videos one by one to get correct processed_query.
        frame_separators_lst = []
        for metadata, video_size in zip(video_metadatas, video_size_lst):
            num_frames = video_size[0]

            # Get frame separators.
            if metadata is not None:
                metedata_fps = metadata["fps"]
                frame_duration_ms = int(1000.0 / metedata_fps)
                frames_indices = metadata["frames_indices"]
                timestamps = [
                    int(frame_index) * frame_duration_ms / 1000.0
                    for frame_index in frames_indices
                ]
                frame_separators = [
                    f"Frame {i + 1} sampled at {timestamp:.2f} seconds: "
                    for i, timestamp in enumerate(timestamps)
                ]
            else:
                frame_separators = [
                    f"Frame {i + 1}: " for i in range(num_frames)
                ]
            frame_separators_lst.append(frame_separators)

        return frame_separators_lst

    def _process_video_prompts(
        self, split_text_prompt: List[str],
        num_tokens_per_frame_lst: List[List[int] | None],
        frame_separators_lst: List[List[str]]
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # Process videos one by one to get correct processed_query.
        processed_query = []
        evs_query = []
        for video_index, (num_tokens_per_frame, frame_separators) in enumerate(
                zip(num_tokens_per_frame_lst, frame_separators_lst)):

            # Prepare video and EVS query.
            processed_query.append(split_text_prompt[video_index])
            processed_query.append("This is a video:\n")
            for frame_sep, num_tokens in zip(frame_separators,
                                             num_tokens_per_frame):
                frame_prompts = [
                    frame_sep, self.img_start_token,
                    self.img_context_token * num_tokens, self.img_end_token,
                    "\n"
                ]
                processed_query.extend(frame_prompts)
            # Video_context_token as placeholder, and it will be combined with the real image_tokens_per_frames during model forward.
            if self.video_pruning_ratio > 0:
                evs_query.append(split_text_prompt[video_index])
                evs_query.append("This is a video:\n")
                for frame_sep in frame_separators:
                    frame_prompts = [
                        frame_sep, self.img_start_token,
                        self.video_context_token, self.img_end_token, "\n"
                    ]
                    evs_query.extend(frame_prompts)
        # Append the last part of the text prompt.
        processed_query.append(split_text_prompt[-1])

        # Tokenize processed query.
        input_ids_lst = [
            self.tokenizer.encode(
                query,
                add_special_tokens=False,
                return_tensors="pt",
            ) for query in processed_query
        ]
        input_ids = torch.cat(input_ids_lst, dim=1)

        if self.video_pruning_ratio > 0:
            evs_query.append(split_text_prompt[-1])
            evs_ids = [
                self.tokenizer.encode(
                    query,
                    add_special_tokens=False,
                    return_tensors="pt",
                )[0] for query in evs_query
            ]
        else:
            evs_ids = None

        return input_ids, evs_ids

    def _compute_token_numbers_per_video(
            self, video_size_lst: List[Tuple]) -> List[List[int]]:
        num_tokens_per_frame_lst = []
        for video_size in video_size_lst:
            num_frames = video_size[0]
            num_patches_per_frame = video_size[1]
            img_height = video_size[2]
            img_width = video_size[3]

            if self.video_pruning_ratio > 0:
                desired_num_tokens = compute_retained_tokens_count(
                    video_size=(num_frames, num_patches_per_frame * img_height,
                                img_width),
                    spatial_merge_size=self.spatial_merge_size,
                    pruning_ratio=self.video_pruning_ratio,
                )
                # It is dummy tokens and will be adjusted in VisionEncoder after applied EVS.
                # We need to know the length of the full input ids ahead, to make Mamba-mixer and inflight-batching work.
                num_tokens_per_frame = [desired_num_tokens
                                        ] + [0] * (num_frames - 1)
            else:
                num_tokens_per_frame = [
                    num_patches_per_frame * self.num_image_token
                ] * num_frames

            num_tokens_per_frame_lst.append(num_tokens_per_frame)
        return num_tokens_per_frame_lst

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
        modality_data = dict()
        input_ids = None
        if images is not None:
            modality_type = "image"
            processed_images, input_ids = self._process_images(
                images, text_prompt)
            evs_ids = None
            modality_data["pixel_values"] = processed_images["pixel_values"].to(
                self.dtype)
            modality_data["num_patches"] = processed_images["num_patches"].sum(
                dim=0, keepdim=True)
            modality_data["video_size"] = None
            # During model inference, the image/video modality data can be mixed during inflight-batching.
            # Store input_ids for image modality here when EVS is enabled, which will be used in merge_evs_mm_embeds later.
            modality_data["evs_ids"] = input_ids[0].to(
                torch.int32) if self.video_pruning_ratio > 0 else None
        elif videos is not None:
            modality_type = "video"
            video_frames, video_metadatas = [
                video_data.frames for video_data in videos
            ], [video_data.metadata for video_data in videos]
            num_videos = len(video_frames)
            processed_images = self._process_videos_frames(video_frames)

            # Num_tokens_per_frame_lst is a dummy one when EVS is enabled.
            num_tokens_per_frame_lst = self._compute_token_numbers_per_video(
                processed_images['video_size'])
            # Special video tokens will be added to match training behaviors.
            frame_separators_lst = self._get_frame_separators(
                processed_images['video_size'], video_metadatas)

            split_text_prompt = text_prompt.split(self.video_context_token)
            if len(split_text_prompt) - 1 != num_videos:
                raise ValueError(
                    f"Number of {self.video_context_token} tokens ({len(split_text_prompt) - 1}) doesn't match number of videos ({num_videos})"
                )
            input_ids, evs_ids = self._process_video_prompts(
                split_text_prompt, num_tokens_per_frame_lst,
                frame_separators_lst)
            modality_data["pixel_values"] = processed_images["pixel_values"].to(
                self.dtype)
            modality_data["num_patches"] = processed_images["num_patches"].sum(
                dim=0, keepdim=True)
            modality_data["video_size"] = processed_images["video_size"]
            modality_data["evs_ids"] = evs_ids

        # Will package inputs for language model forward in AGGREGATE mode.
        multimodal_data = {}
        multimodal_data['modality_type'] = modality_type
        multimodal_data[modality_type] = modality_data
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
        self.video_pruning_ratio = VIDEO_PRUNING_RATIO

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

    def merge_evs_mm_embeds(self, num_tokens_in_videos: List[int],
                            multimodal_params: List[MultimodalParams],
                            input_ids: torch.Tensor) -> torch.Tensor:
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
            multimodal_param.multimodal_data
            for multimodal_param in multimodal_params
        ]
        modalities = [
            multimodal_data['modality_type']
            for multimodal_data in multimodal_data_lst
        ]
        # Skip EVS if there is no video modality.
        if "video" not in modalities:
            return input_ids

        evs_ids_lst = [
            multimodal_data[modality]['evs_ids'] for modality, multimodal_data
            in zip(modalities, multimodal_data_lst)
        ]
        # Iterate over batch.
        context_ids = []
        for evs_ids, modality, num_tokens_in_video in zip(
                evs_ids_lst, modalities, num_tokens_in_videos):
            # Special handling for image modality when mixing image and video modalities during inflight-batching.
            if modality == "image":
                context_ids.append(evs_ids)
                continue

            image_idx = 0
            for evs_id in evs_ids:
                if len(evs_id
                       ) == 1 and evs_id[0] == self.video_context_token_id:
                    image_mm = torch.full(
                        (num_tokens_in_video[image_idx], ),
                        fill_value=self.img_context_token_id,
                        dtype=evs_id.dtype,
                        device=evs_id.device,
                    )
                    context_ids.append(image_mm)
                    image_idx += 1
                else:
                    context_ids.append(evs_id)

        context_ids = torch.cat(context_ids, dim=0)
        # -> [num_tokens, ]

        # Special handling for inflight-batching.
        # Assume input ids format is [context_ids, generation_ids].
        input_ids[:context_ids.shape[0]] = context_ids
        del context_ids

        return input_ids

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
            if self.video_pruning_ratio > 0:
                input_ids = self.merge_evs_mm_embeds(
                    num_tokens_in_videos,
                    multimodal_params=multimodal_params[:num_context_requests],
                    input_ids=input_ids,
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
