import copy
import math
import os
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)
from transformers.models.auto import CONFIG_MAPPING

from tensorrt_llm.inputs.multimodal import MultimodalParams

from ...inputs import (BaseMultimodalInputProcessor, ExtraProcessedInputs,
                       InputProcessor, MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (find_input_mm_embeds, fuse_input_embeds,
                                        get_multimodal_embeddings)
from .modeling_siglip import SiglipVisionModel
from .modeling_utils import register_auto_model

DISAGG = os.getenv('TLLM_MULTIMODAL_DISAGGREGATED', '0') == '1'


def select_best_resolution(original_size: tuple,
                           possible_resolutions: list) -> tuple:
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height,
                                   original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution
                and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def unpad_image(tensor: torch.Tensor,
                original_size: Tuple[int, int]) -> torch.Tensor:
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


def get_anyres_image_grid_shape(
    image_size: Tuple[int, int],
    grid_pinpoints: Union[str, List[Tuple[int, int]]],
    patch_size: int,
) -> Tuple[int, int]:
    import ast
    possible_resolutions = grid_pinpoints if isinstance(
        grid_pinpoints, list) else ast.literal_eval(grid_pinpoints)

    original_width, original_height = image_size
    height, width = select_best_resolution((original_height, original_width),
                                           possible_resolutions)
    return width // patch_size, height // patch_size


def reshape_and_unpad_image_features(
    image_feature: torch.Tensor,
    height: int,
    width: int,
    image_size: Tuple[int, int],
    possible_resolutions: List[Tuple[int, int]],
    grid_size: int,
    unpad: bool,
    image_newline: torch.Tensor,
) -> torch.Tensor:
    base_image_feature = image_feature[0]
    image_feature = image_feature[1:]

    assert (
        height * width == base_image_feature.shape[0]
    ), f"height: {height}, width: {width}, base_image_feature.shape[0]: {base_image_feature.shape[0]}"

    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
        image_size, possible_resolutions, grid_size)
    image_feature = image_feature.view(num_patch_height, num_patch_width,
                                       height, width, -1)

    if unpad:
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_size)
        image_feature = torch.cat(
            (
                image_feature,
                image_newline[:, None, None].expand(*image_feature.shape[:-1],
                                                    1).to(image_feature.device),
            ),
            dim=-1,
        )
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    else:
        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
        image_feature = image_feature.flatten(0, 3)
    image_feature = torch.cat((base_image_feature, image_feature), dim=0)

    return image_feature


def anyres_postprocessing(
    image_forward_outs: torch.FloatTensor,
    split_sizes: List[int],
    image_sizes: List[List[int]],
    possible_resolutions: List[Tuple[int, int]],
    is_videos: List[bool],
    patch_size: int,
    grid_size: int,
    image_newline: torch.FloatTensor,
    num_queries_vis_abstractor: int = -1,
    unpad: bool = False,
) -> List[torch.FloatTensor]:
    height = width = grid_size // patch_size

    if num_queries_vis_abstractor > 0:
        assert (num_queries_vis_abstractor**0.5
                ).is_integer(), "n_queries must be square number"
        height = width = int(num_queries_vis_abstractor**0.5)

    image_features = torch.split(image_forward_outs, split_sizes, dim=0)

    new_image_features = []
    for image_idx, (image_feature,
                    is_video) in enumerate(zip(image_features, is_videos)):
        if image_feature.shape[0] > 1:
            if not is_video:
                image_feature = reshape_and_unpad_image_features(
                    image_feature=image_feature,
                    height=height,
                    width=width,
                    image_size=image_sizes[image_idx],
                    possible_resolutions=possible_resolutions,
                    grid_size=grid_size,
                    unpad=unpad,
                    image_newline=image_newline,
                )
            else:
                image_feature = image_feature.flatten(0, 1)
        else:
            image_feature = image_feature[0]
            if unpad and not is_video:
                image_feature = torch.cat(
                    (image_feature, image_newline[None].to(
                        image_feature.device)),
                    dim=0)
        new_image_features.append(image_feature)
    image_features = new_image_features
    return image_features


def adaptive_anyres_postprocessing(
    image_forward_outs: torch.FloatTensor,
    image_sizes: List[List[int]],
    possible_resolutions: List[Tuple[int, int]],
    is_videos: List[bool],
    group_ids: List[List[int]],
    num_queries_vis_abstractors: List[List[int]],
    grid_size: int,
    image_newline: torch.FloatTensor,
    unpad: bool = False,
) -> List[torch.FloatTensor]:
    new_image_features = []
    for image_idx, (image_feature,
                    is_video) in enumerate(zip(image_forward_outs, is_videos)):
        num_queries_vis_abstractor = num_queries_vis_abstractors[image_idx]
        assert (num_queries_vis_abstractor**0.5
                ).is_integer(), "n_queries must be square number"
        height = width = int(num_queries_vis_abstractor**0.5)

        if image_feature.shape[0] > 1:
            if not is_video:
                image_feature = reshape_and_unpad_image_features(
                    image_feature=image_feature,
                    height=height,
                    width=width,
                    image_size=image_sizes[image_idx],
                    possible_resolutions=possible_resolutions,
                    grid_size=grid_size,
                    unpad=unpad,
                    image_newline=image_newline,
                )
            else:
                image_feature = image_feature.flatten(0, 1)
        else:
            image_feature = image_feature[0]
            if unpad and not is_video:
                image_feature = torch.cat(
                    (image_feature, image_newline[None].to(
                        image_feature.device)),
                    dim=0)
        new_image_features.append(image_feature)

    image_features = [
        torch.cat([new_image_features[group_id] for group_id in group_ids_list],
                  dim=0) for group_ids_list in group_ids
    ]
    return image_features


def compute_adaptive_params(
    pixel_values: Optional[List[List[torch.FloatTensor]]] = None,
    num_queries_vis_abstractors: Optional[List[List[int]]] = None,
    num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
    image_sizes: Optional[List[List[List[int]]]] = None,
    is_videos: Optional[List[bool]] = None,
    first_last_frames_slows: Optional[List[bool]] = None,
) -> Tuple[List[int], List[int], List[List[int]], List[bool], List[List[int]]]:
    assert all(
        all(isinstance(value, int) and value >= 0 for value in sublist)
        for sublist in num_queries_vis_abstractors
    ), "All values in num_queries_vis_abstractors must be integers >= 0."

    assert all(
        all(isinstance(value, int) and value >= 0 for value in sublist)
        for sublist in num_queries_vis_abstractors_slow
    ), "All values in num_queries_vis_abstractors_slow must be integers >= 0."

    assert is_videos is not None

    is_first_images = []
    is_last_images = []
    for is_video in is_videos:
        for idx, is_video_item in enumerate(is_video):
            if idx == 0:
                is_first_images.append(True)
            else:
                is_first_images.append(False)
            if idx == len(is_video) - 1:
                is_last_images.append(True)
            else:
                is_last_images.append(False)

    num_queries_vis_abstractors = list(chain(*num_queries_vis_abstractors))
    num_queries_vis_abstractors_slow = list(
        chain(*num_queries_vis_abstractors_slow))
    image_sizes = list(chain(*image_sizes))
    is_videos = list(chain(*is_videos))
    first_last_frames_slows = list(chain(*first_last_frames_slows))

    use_slowfast = any(
        [num_query > 0 for num_query in num_queries_vis_abstractors_slow])
    num_grids = [pixel_value.shape[0] for pixel_value in chain(*pixel_values)]
    num_grids = [0] + num_grids
    group_ids = []

    if use_slowfast:
        new_num_grids = [num_grids[0]]
        new_num_queries = []
        new_image_sizes = []
        new_is_videos = []

        for (
                num_query,
                num_query_slow,
                num_grid,
                image_size,
                is_video,
                first_last_frames_slow,
                is_first_image,
                is_last_image,
        ) in zip(
                num_queries_vis_abstractors,
                num_queries_vis_abstractors_slow,
                num_grids[1:],
                image_sizes,
                is_videos,
                first_last_frames_slows,
                is_first_images,
                is_last_images,
        ):

            if not first_last_frames_slow and num_query_slow > 0:
                assert is_video

                this_group_ids = [group_ids[-1][-1] + 1 if group_ids else 0]

                new_num_grids.append(new_num_grids[-1] + 1)
                new_num_queries.append(num_query_slow)
                new_image_sizes.append(image_size)
                new_is_videos.append(is_video)

                if num_grid >= 2:
                    new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                    new_num_queries.append(num_query)
                    new_image_sizes.append(image_size)
                    new_is_videos.append(is_video)
                    this_group_ids.append(this_group_ids[-1] + 1)

                group_ids.append(this_group_ids)
            elif (first_last_frames_slow and num_query_slow > 0
                  and (is_first_image or is_last_image)):
                assert is_video

                this_group_ids = [group_ids[-1][-1] + 1 if group_ids else 0]

                if num_grid == 1:
                    new_num_grids.append(new_num_grids[-1] + 1)
                    new_num_queries.append(num_query_slow)
                    new_image_sizes.append(image_size)
                    new_is_videos.append(is_video)

                if num_grid >= 2:
                    if is_first_image:
                        new_num_grids.append(new_num_grids[-1] + 1)
                        new_num_queries.append(num_query_slow)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                        new_num_queries.append(num_query)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        this_group_ids.append(this_group_ids[-1] + 1)
                    elif is_last_image:
                        new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                        new_num_queries.append(num_query)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        new_num_grids.append(new_num_grids[-1] + 1)
                        new_num_queries.append(num_query_slow)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        this_group_ids.append(this_group_ids[-1] + 1)
                    else:
                        raise Exception("This case should not be reached.")
                group_ids.append(this_group_ids)
            else:
                new_num_grids.append(new_num_grids[-1] + num_grid)
                new_num_queries.append(num_query)
                new_image_sizes.append(image_size)
                new_is_videos.append(is_video)

                start_group_id = group_ids[-1][-1] + 1 if group_ids else 0
                group_ids.append([start_group_id])

        num_grids = new_num_grids
        num_queries_vis_abstractors = new_num_queries
        image_sizes = new_image_sizes
        is_videos = new_is_videos
    else:
        num_grids = [sum(num_grids[:i]) for i in range(1, len(num_grids) + 1)]
        group_ids = [[group_id] for group_id in range(len(is_videos))]

    return num_queries_vis_abstractors, num_grids, image_sizes, is_videos, group_ids


def determine_non_vision_query_lengths(input_ids: torch.LongTensor, pad_id: int,
                                       img_start_id: int) -> List[int]:
    non_vision_query_lengths = []
    batch_size, len_seq = input_ids.size(0), input_ids.size(1)

    for i in range(batch_size):
        temp_idx = (input_ids[i] == pad_id).nonzero()
        eos_idx = temp_idx[0, 0].item() if len(temp_idx) > 0 else len_seq
        num_imgs = (input_ids[i] == img_start_id).sum().item()
        non_vision_query_lengths.append(eos_idx - num_imgs)

    if all([pad_id in input_id for input_id in input_ids.tolist()]):
        non_vision_query_lengths = [
            non_vision_query_length + 1
            for non_vision_query_length in non_vision_query_lengths
        ]

    return non_vision_query_lengths


class HCXVisionCAbstractor(nn.Module):
    """
    This module is based on C-Abstractor, whose license is under apache-2.0.
    You can check the original code at https://github.com/khanrc/honeybee/blob/main/honeybee/projectors/projectors.py
    and we made necessary modifications.
    """

    def __init__(
        self,
        num_queries: int,
        num_input_tokens: int,
        encoder_hidden_size: int,
        hidden_size: int,
        output_hidden_size: int,
        pos_emb: bool = True,
        prenorm: bool = False,
    ):
        super().__init__()
        self.num_input_tokens = num_input_tokens
        self.output_hidden_size = output_hidden_size

        # Positional embedding
        if pos_emb:
            self.pos_emb = torch.nn.Parameter(
                torch.zeros(1, num_input_tokens, encoder_hidden_size))
            self.pos_emb.data.normal_(mean=0.0, std=0.02)
        else:
            self.pos_emb = None

        # (Optional) Pre-normalization layer
        from timm.layers import LayerNorm
        if prenorm:
            self.prenorm = LayerNorm(encoder_hidden_size)
        else:
            self.prenorm = None

        self.build_net(num_queries, encoder_hidden_size, hidden_size,
                       output_hidden_size)
        self.dtype = next(self.parameters()).dtype

    def forward(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: Optional[List[List[int]]] = None,
        num_grids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (e.g. CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x = x + self.pos_emb

        x = self._forward(
            x,
            num_queries_vis_abstractors=num_queries_vis_abstractors,
            num_grids=num_grids,
        )  # (B, L, output_hidden_size)

        return x

    def _forward(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: Optional[List[List[int]]] = None,
        num_grids: Optional[List[int]] = None,
    ) -> torch.Tensor:

        # x: [B, L, dim]
        B, L, dim = x.shape
        hw = int(L**0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)

        if num_queries_vis_abstractors is not None:
            assert num_grids is not None
            return self._forward_adaptive_num_query(
                x, num_queries_vis_abstractors, num_grids)

        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        return x

    def _forward_adaptive_num_query(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: Optional[List[List[int]]] = None,
        num_grids: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        # self.net is consisted by 3 layers (s1, sampler, s2)
        assert len(self.net) == 3

        x = self.net[0](x)  # s1
        new_x = []
        for i, num_queries in enumerate(num_queries_vis_abstractors):
            hw = int(num_queries**0.5)
            sampler = nn.AdaptiveAvgPool2d((hw, hw))
            out = sampler(x[num_grids[i]:num_grids[i + 1], :])
            out = self.net[2](out)  # s2

            out = rearrange(out, "b d h w -> b (h w) d")
            out = self.readout(out)

            new_x.append(out)
        return new_x

    def build_net(
        self,
        n_queries: int,
        encoder_hidden_size: int,
        hidden_size: int,
        output_hidden_size: int,
        depth: int = 3,
        mlp_depth: int = 2,
    ):
        assert (n_queries**0.5).is_integer(
        ), f"n_queries must be square number. n_queries: {n_queries}"
        hw = int(n_queries**0.5)
        from timm.layers import LayerNorm2d
        from timm.models.regnet import RegStage

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = self.build_mlp(mlp_depth, hidden_size,
                                      output_hidden_size)

    def build_mlp(
        self,
        depth: int,
        hidden_size: int,
        output_hidden_size: int,
    ):
        layers = [nn.Linear(hidden_size, output_hidden_size)]
        for _ in range(1, depth):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(output_hidden_size, output_hidden_size))
        return nn.Sequential(*layers)


class HCXVisionInputProcessor(BaseMultimodalInputProcessor, InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True):

        self.pretrained_config = model_config
        self.tokenizer = tokenizer
        self.use_fast = True
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast)
        self.tllm_multimodal_token_id = self.pretrained_config.language_config[
            "vocab_size"] + 1
        self.vision_query_lengths = None
        self._vision_query_generator = None

    def get_vocab_size(self):
        return self.pretrained_config.language_config["vocab_size"]

    def get_num_tokens_per_image(
        self,
        image: Image.Image,
        **kwargs,
    ):
        """
        Get the number of tokens per image.

        This method must be called after __call__ is executed.
        Uses a generator pattern to safely iterate through vision_query_lengths.

        Args:
            image: PIL Image to get token count for
            **kwargs: Additional arguments for processor (unused)

        Returns:
            int: Number of tokens for the image

        Raises:
            RuntimeError: If called before __call__ is executed
            IndexError: If called more times than there are images
        """
        if self.vision_query_lengths is None:
            raise RuntimeError(
                "get_num_tokens_per_image() must be called after __call__() is executed. "
                "vision_query_lengths is not available.")

        # Initialize generator if not already done
        if self._vision_query_generator is None:
            self._vision_query_generator = self._create_vision_query_generator()

        try:
            return next(self._vision_query_generator)
        except StopIteration:
            raise IndexError(
                "get_num_tokens_per_image() called more times than the number of images processed. "
                "No more vision query lengths available.")

    def get_mm_token_ids(self):
        return torch.tensor([self.tllm_multimodal_token_id])

    def _create_vision_query_generator(self):
        """Create a generator that yields vision query lengths for each image."""
        if self.vision_query_lengths is None:
            return

        # Flatten all vision query lengths from all batches
        for batch_vision_queries in self.vision_query_lengths:
            for query_length in batch_vision_queries:
                yield query_length

    def _reset_vision_query_generator(self):
        """Reset the vision query generator."""
        self._vision_query_generator = None

    def _post_process(self,
                      input_ids: torch.Tensor,
                      preprocessed_image: dict[str, any] = None):
        if not preprocessed_image:
            return input_ids[0]

        vision_query_lengths = preprocessed_image.get("vision_query_lengths",
                                                      None)
        non_vision_query_lengths = determine_non_vision_query_lengths(
            input_ids, self.tokenizer.pad_token_id,
            self.pretrained_config.img_start_id)
        batch_size = input_ids.size(0)

        len_inputs_embeds = max([
            sum(vision_query_length) + non_vision_query_length
            for non_vision_query_length, vision_query_length in zip(
                non_vision_query_lengths, vision_query_lengths)
        ])

        len_inputs_embeds = min(self.pretrained_config.decoder_max_length,
                                len_inputs_embeds)

        image_cnts = (input_ids == self.pretrained_config.img_start_id).sum(
            dim=1).tolist()

        fused_input_ids = torch.zeros([batch_size, len_inputs_embeds],
                                      dtype=input_ids.dtype)
        for batch_idx, sample in enumerate(input_ids):
            non_vision_query_length = non_vision_query_lengths[batch_idx]
            sample = sample[:non_vision_query_length + image_cnts[batch_idx]]

            mask = (sample == self.pretrained_config.img_start_id)
            img_start_ids = mask.nonzero()
            input_start, temp_start = 0, 0

            for multi_img_idx, img_start_idx in enumerate(img_start_ids):
                token_len = img_start_idx - temp_start

                fused_input_ids[batch_idx, input_start:input_start +
                                token_len] = input_ids[batch_idx,
                                                       temp_start:temp_start +
                                                       token_len]

                fused_input_ids[
                    batch_idx,
                    input_start + token_len:input_start + token_len +
                    vision_query_lengths[batch_idx][multi_img_idx],
                ] = self.tllm_multimodal_token_id

                input_start += token_len + vision_query_lengths[batch_idx][
                    multi_img_idx]
                temp_start += token_len + 1

            token_len = min(sample[temp_start:].size(0),
                            fused_input_ids.size(1) - input_start)
            fused_input_ids[batch_idx, input_start:input_start +
                            token_len] = input_ids[batch_idx,
                                                   temp_start:temp_start +
                                                   token_len]

        return fused_input_ids[0]

    def _preprocess(self, text_prompt: dict[str, any], images: List[Any],
                    mm_processor_kwargs: Dict[str, Any]):

        preprocessed_image = None
        if images is not None:
            is_video_list = [False] * len(images)
            preprocessed_image = self.processor(
                images=images,
                is_video_list=is_video_list,
                **mm_processor_kwargs,
            )
            self.vision_query_lengths = preprocessed_image.get(
                "vision_query_lengths", None)
            # Reset generator when new vision_query_lengths are available
            self._reset_vision_query_generator()

        input_ids = self.tokenizer.encode(text_prompt,
                                          add_special_tokens=False,
                                          return_tensors="pt")
        return input_ids, preprocessed_image

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = inputs.get("prompt"), \
                        inputs.get("multi_modal_data", {}), inputs.get("mm_processor_kwargs", {})

        images = mm_data.get("image", None)
        if images is not None:
            if isinstance(images[0], torch.Tensor):
                # NOTE: HyperCLOVA-SEED-Vision-Instruct-3B uses the image data in the format of (H, W, C) and not in the range of [0, 1], otherwise gives an error.
                images = [(image.permute(1, 2, 0) * 255).to(torch.uint8)
                          for image in images]

        input_ids, preprocessed_image = self._preprocess(
            text_prompt, images, mm_processor_kwargs)

        fused_input_ids = self._post_process(input_ids, preprocessed_image)

        if not preprocessed_image:
            return fused_input_ids.to(torch.int32).tolist(), {}

        multimodal_data = {}
        multimodal_data["image"] = {
            "pixel_values":
            torch.stack(preprocessed_image['pixel_values'][0],
                        dim=0).to(torch.bfloat16),
            "image_sizes":
            preprocessed_image.get('image_sizes', None),
            "is_videos":
            preprocessed_image.get('is_videos', None),
            "num_queries_vis_abstractors":
            preprocessed_image.get('num_queries_vis_abstractors', None),
            "num_queries_vis_abstractors_slow":
            preprocessed_image.get('num_queries_vis_abstractors_slow', None),
            "first_last_frames_slows":
            preprocessed_image.get('first_last_frames_slows', None),
        }
        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data
        }


@register_auto_model("HCXVisionModel")
class HCXVisionModel(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        self.model_config = model_config
        self.pretrained_config = model_config.pretrained_config
        siglip_model_config = copy.deepcopy(self.model_config)
        siglip_model_config.pretrained_config = self.model_config.pretrained_config.vision_config
        self.visual_token_idx = 0 if "siglip" in self.model_config.pretrained_config.vision_config.model_type else 1
        self.dtype = self.model_config.pretrained_config.vision_config.torch_dtype
        self.vision_model = SiglipVisionModel(siglip_model_config).to(
            self.dtype)
        self.mm_projector = HCXVisionCAbstractor(
            num_queries=self.pretrained_config.num_queries_vis_abstractor,
            num_input_tokens=(
                self.pretrained_config.vision_config.image_size //
                self.pretrained_config.vision_config.patch_size)**2,
            encoder_hidden_size=self.pretrained_config.vision_config.
            hidden_size,
            hidden_size=self.pretrained_config.vision_config.hidden_size,
            output_hidden_size=self.pretrained_config.hidden_size,
            pos_emb=self.pretrained_config.proj_pos_emb,
            prenorm=self.pretrained_config.proj_prenorm,
        ).to(self.dtype)
        self.image_newline = nn.Parameter(torch.empty(
            self.pretrained_config.hidden_size, ),
                                          requires_grad=False).to(self.dtype)

        self.unpad = self.pretrained_config.unpad
        self.use_nth_layer = self.pretrained_config.use_nth_layer
        self.anyres = self.pretrained_config.anyres
        self.possible_resolutions = self._init_possible_resolutions()
        self.post_config()

    def post_config(self):
        self.config = self.vision_model.config
        self.model_config.pretrained_config = self.vision_model.config

    def _init_possible_resolutions(self):
        possible_resolutions = []
        if self.pretrained_config.anyres:
            assert self.pretrained_config.max_num_grids > 0
            for i in range(1, self.pretrained_config.max_num_grids + 1):
                for j in range(1, self.pretrained_config.max_num_grids + 1):
                    if i == 1 and j == 1 and not self.pretrained_config.use_1x1_grid:
                        continue
                    if i * j <= self.pretrained_config.max_num_grids:
                        possible_resolutions.append([i, j])

            possible_resolutions = [[
                ys * self.pretrained_config.vision_config.image_size,
                xs * self.pretrained_config.vision_config.image_size
            ] for ys, xs in possible_resolutions]
        return possible_resolutions

    def load_weights(self, weights):
        vision_weights = _filter_weights(weights, "vision_model.")
        self.vision_model.load_weights(vision_weights)

        mm_projector_weights = _filter_weights(weights, "mm_projector.")
        self.mm_projector.load_state_dict(mm_projector_weights, strict=True)

        self.image_newline.data.copy_(weights["image_newline"])

    def _parse_and_batch_multimodal_data(
        self, multimodal_params: List[MultimodalParams]
    ) -> Tuple[List[torch.Tensor], Dict[str, List[Any]]]:
        """Parse and batch multimodal data from MultimodalParams objects."""
        pixel_values = [
            list(
                torch.unbind(
                    multimodal_param.multimodal_data["image"]["pixel_values"],
                    dim=0)) for multimodal_param in multimodal_params
        ]
        mm_extra_data = {
            key: [
                multimodal_param.multimodal_data["image"][key][0]
                for multimodal_param in multimodal_params
            ]
            for key in multimodal_params[0].multimodal_data["image"].keys()
        }
        return pixel_values, mm_extra_data

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]):

        pixel_values, mm_extra_data = self._parse_and_batch_multimodal_data(
            multimodal_params)
        image_sizes = mm_extra_data.get("image_sizes", None)
        is_videos = mm_extra_data.get("is_videos", None)
        num_queries_vis_abstractors = mm_extra_data.get(
            "num_queries_vis_abstractors", None)
        num_queries_vis_abstractors_slow = mm_extra_data.get(
            "num_queries_vis_abstractors_slow", None)
        first_last_frames_slows = mm_extra_data.get("first_last_frames_slows",
                                                    None)

        len_pixel_values = [len(pixel_value) for pixel_value in pixel_values]
        concat_pixel_values = torch.cat(list(chain(*pixel_values)),
                                        dim=0)  # list of list of 4D Tensor

        n_chunks = 1
        total_len = concat_pixel_values.size(0)
        chunk_size = math.ceil(total_len / n_chunks) if total_len > 0 else 1
        image_forward_outs_chunks = []

        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = concat_pixel_values[start:end]
            if chunk.size(0) < chunk_size:
                pad_size = chunk_size - chunk.size(0)
                dummy_shape = (pad_size, ) + tuple(
                    concat_pixel_values.shape[1:])
                dummy = torch.zeros(
                    dummy_shape,
                    dtype=concat_pixel_values.dtype,
                    device=concat_pixel_values.device,
                )
                chunk = torch.cat([chunk, dummy], dim=0)
            attn_metadata = self.vision_model.prepare_attn_metadata(
                chunk.shape[0])
            if self.use_nth_layer == -1:
                self.vision_model.vision_model.post_layernorm = nn.Identity()
                outs = self.vision_model(chunk, attn_metadata=attn_metadata)
                outs = outs[:, self.visual_token_idx:]
            else:
                outs = self.vision_model(chunk, attn_metadata=attn_metadata)
                outs = outs[self.use_nth_layer][:, self.visual_token_idx:]
            image_forward_outs_chunks.append(outs)

        image_forward_outs = torch.cat(image_forward_outs_chunks, dim=0).to(
            image_forward_outs_chunks[0].dtype)

        if num_queries_vis_abstractors is None:
            assert num_queries_vis_abstractors_slow is None
            image_sizes = list(chain(*image_sizes))
            if is_videos is not None:
                is_videos = list(chain(*is_videos))
            group_ids = None
            image_forward_outs = self.mm_projector(image_forward_outs)
        else:
            (
                num_queries_vis_abstractors,
                num_grids,
                image_sizes,
                is_videos,
                group_ids,
            ) = compute_adaptive_params(
                pixel_values,
                num_queries_vis_abstractors,
                num_queries_vis_abstractors_slow,
                image_sizes,
                is_videos,
                first_last_frames_slows,
            )
            image_forward_outs = self.mm_projector(
                image_forward_outs,
                num_queries_vis_abstractors=num_queries_vis_abstractors,
                num_grids=num_grids,
            )
        if self.anyres:
            split_sizes = [
                pixel_value.shape[0] for pixel_value in chain(*pixel_values)
            ]
            if num_queries_vis_abstractors is None:
                image_features = anyres_postprocessing(
                    image_forward_outs=image_forward_outs,
                    split_sizes=split_sizes,
                    image_sizes=image_sizes,
                    num_queries_vis_abstractor=self.num_queries_vis_abstractor,
                    unpad=self.unpad,
                    is_videos=is_videos,
                    patch_size=self.vision_model.config.patch_size,
                    grid_size=self.vision_model.config.image_size,
                    image_newline=self.image_newline,
                    possible_resolutions=self.possible_resolutions,
                )
            else:
                image_features = adaptive_anyres_postprocessing(
                    image_forward_outs=image_forward_outs,
                    image_sizes=image_sizes,
                    num_queries_vis_abstractors=num_queries_vis_abstractors,
                    unpad=self.unpad,
                    is_videos=is_videos,
                    grid_size=self.vision_model.config.image_size,
                    image_newline=self.image_newline,
                    possible_resolutions=self.possible_resolutions,
                    group_ids=group_ids,
                )
        else:
            if num_queries_vis_abstractors is None:
                image_features = [
                    image_forward_out
                    for image_forward_out in image_forward_outs
                ]
            else:
                image_features = [
                    image_forward_out.unsqueeze(0)
                    for image_forward_out in image_forward_outs
                ]
        image_features = [
            image_features[sum(len_pixel_values[:i]):sum(len_pixel_values[:i +
                                                                          1])]
            for i in range(len(len_pixel_values))
        ]
        mm_embeds = [
            torch.cat(list(chain(image_feature)), dim=0)
            for image_feature in image_features
        ]
        return mm_embeds


@register_auto_model("HCXVisionForCausalLM")
@register_input_processor(
    HCXVisionInputProcessor,
    model_type="hyperclovax_vlm",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image":
            '<im_end>\n<|im_start|>user (mime) \n'
            '{{"type": "image/jpeg", "filename": ""}}<|im_end|>\n'
            '<|im_start|>user (vector)\n<|dummy3|><|im_end|>\n'
            '<|im_start|>image/aux\n'
            '다음 중 ocr은 사진에서 검출된 글자이고, lens_keyword는 사진에서 추출된 '
            'keyword와 bbox 위치입니다.bbox는 0~1 사이로 정규화된 [x1, y1, x2, y2]의 '
            '형태입니다. 참고하여 답변하세요. '
            '{{"ocr": "", "lens_keywords": "", "lens_local_keywords": ""}}'
        },
        placeholder_placement=MultimodalPlaceholderPlacement.AFTER_TEXT,
    ))
class HCXVisionForCausalLM(PreTrainedModel):

    def __init__(self, model_config: ModelConfig):
        config = model_config.pretrained_config
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return
        if not DISAGG:
            vision_model_config = copy.deepcopy(model_config)
            vision_model_type = self.model_config.pretrained_config.vision_config[
                "model_type"]
            vision_config_obj = CONFIG_MAPPING[vision_model_type](
                **self.model_config.pretrained_config.vision_config)
            vision_model_config.pretrained_config.vision_config = vision_config_obj
            vision_model_config.pretrained_config.architectures = [
                "HCXVisionModel"
            ]
            self.mm_encoder = AutoModelForCausalLM.from_config(
                vision_model_config)

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = PretrainedConfig.from_dict(
            llm_model_config.pretrained_config.language_config)
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)
        self.model_dtype = getattr(config.language_config, "torch_dtype",
                                   torch.bfloat16)
        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):
        language_weights = _filter_weights(weights, "language_model.")
        self.llm.load_weights(language_weights)

        if not DISAGG:
            self.mm_encoder.load_weights(weights)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
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
        mm_embeds = []
        if len(multimodal_params) > 0:
            if not DISAGG:
                mm_embeds = get_multimodal_embeddings(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=multimodal_params[:num_context_requests])
            else:
                raise NotImplementedError(
                    "HCXVisionForCausalLM does not support disaggregated inference yet. Please unset "
                    f"the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )
            mm_embeds = find_input_mm_embeds(
                mm_embeds, multimodal_params[:num_context_requests])

        input_ids, input_embeds = fuse_input_embeds(self.llm.model.embed_tokens,
                                                    input_ids, mm_embeds,
                                                    **kwargs)
        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits)

        logger.debug(f'output shape: {output_prob.shape}')
        return output_prob


def _filter_weights(weights: Dict[str, torch.Tensor],
                    prefix: str) -> Dict[str, torch.Tensor]:
    return {
        name[len(prefix):]: weight
        for name, weight in weights.items() if name.startswith(prefix)
    }
