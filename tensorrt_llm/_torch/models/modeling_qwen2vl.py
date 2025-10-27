import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed, Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLMLP,
    Qwen2_5_VLVisionBlock, apply_rotary_pos_emb_vision)
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    Qwen2VisionTransformerPretrainedModel

from tensorrt_llm._torch.attention_backend.interface import \
    PredefinedAttentionMask
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ..._utils import nvtx_range
from ...inputs import (BaseDummyInputsBuilder, BaseMultimodalInputProcessor,
                       ExtraProcessedInputs, InputProcessor,
                       MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       default_multimodal_input_loader,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..attention_backend.utils import get_attention_backend
from ..modules.rotary_embedding import MRotaryEmbedding
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (find_input_mm_embeds, fuse_input_embeds,
                                        get_multimodal_embeddings)
from .modeling_utils import (ModelConfig, register_auto_model,
                             register_vision_encoder)

DISAGG = os.getenv('TLLM_MULTIMODAL_DISAGGREGATED', '0') == '1'
PAD_INDEX = -100  # NOTE: refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py#L269


def process_weights(weights: Dict,
                    prefix: str = "visual",
                    weight_name_mapping: Dict[str, str] = None) -> Dict:
    """
    Filter and transform weights in a single modular function.

    Args:
        weights: Dictionary of all model weights
        prefix: Prefix to filter weights by (default: "visual")
        weight_name_mapping: Optional mapping to transform weight names

    Returns:
        Dictionary of processed weights ready for loading
    """

    # Filter weights by prefix (handles both direct and "model." prefixed keys)
    filtered_weights = {}
    for key, weight in weights.items():
        if key.startswith(prefix):
            filtered_weights[key] = weight
        elif key.startswith("model." + prefix):
            filtered_weights[key[len("model."):]] = weight

    # Transform weight names if mapping provided
    if weight_name_mapping:
        transformed_weights = {}
        for key, weight in filtered_weights.items():
            new_key = key
            for old_suffix, new_suffix in weight_name_mapping.items():
                if key.endswith(old_suffix):
                    new_key = key.replace(old_suffix, new_suffix)
                    break
            transformed_weights[new_key] = weight
        return transformed_weights

    return filtered_weights


class Qwen2VLInputProcessorBase(BaseDummyInputsBuilder,
                                BaseMultimodalInputProcessor, InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True):
        self.model_config = model_config
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            model_path)
        self.use_fast = True
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=self.use_fast,
            trust_remote_code=trust_remote_code)

        self.tllm_multimodal_token_id = self.model_config.vocab_size + 1
        # temporal patch size for video frames
        self.temporal_patch_size = getattr(model_config.vision_config,
                                           'temporal_patch_size', 1)

    @classmethod
    def get_rope_index(
        cls,
        model_config: PretrainedConfig,
        input_ids: Optional[torch.IntTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        This is a generalized implementation that can be used by both Qwen2VL and Qwen2_5_VL models.
        The main difference between the two implementations is how temporal position IDs are calculated.

        Args:
            model_config: The model configuration
            input_ids: Indices of input sequence tokens in the vocabulary
            image_grid_thw: The temporal, height and width of feature shape of each image in LLM
            video_grid_thw: The temporal, height and width of feature shape of each video in LLM
            attention_mask: Mask to avoid performing attention on padding token indices
            second_per_grid_ts: The time interval (in seconds) for each grid along the temporal dimension

        Returns:
            position_ids: A tensor of shape (3, batch_size, sequence_length)
            mrope_position_deltas: A tensor of shape (batch_size)
        """
        spatial_merge_size = model_config.vision_config.spatial_merge_size
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        mrope_position_deltas = []

        # Handle case with no vision inputs
        if image_grid_thw is None and video_grid_thw is None:
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

        # Handle case with vision inputs
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids in enumerate(total_input_ids):
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
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
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
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # Calculate temporal position IDs based on model type
                if hasattr(model_config.vision_config, 'tokens_per_second'):
                    # Qwen2_5_VL style temporal position calculation
                    if isinstance(second_per_grid_t, torch.Tensor):
                        second_per_grid_t = second_per_grid_t.item()
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(
                        -1, llm_grid_h * llm_grid_w)
                    time_tensor = expanded_range * second_per_grid_t * model_config.vision_config.tokens_per_second
                    t_index = time_tensor.long().flatten()
                else:
                    # Qwen2VL style temporal position calculation
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
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 -
                                         len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def get_dummy_text(self, input_seq_len: int) -> str:
        ids = np.random.randint(
            low=0,
            high=int(
                self.model_config.vocab_size),  # high is exclusive in NumPy
            size=input_seq_len,
        ).tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def get_dummy_image(self, max_width: int, max_height: int):
        image = Image.new("RGB", (max_width, max_height), color=255)
        return image

    def get_dummy_prompt(self, input_seq_len: int):
        text = ""
        # we use the max resolution as starting point
        img_max_dim = 3584
        image = self.get_dummy_image(max_width=img_max_dim,
                                     max_height=img_max_dim)

        test_mm_prompt = default_multimodal_input_loader(
            tokenizer=self.tokenizer,
            model_dir=self.model_path,
            model_type=self.model_config.model_type,
            modality="image",
            prompts=[text],
            media=[[image]],
            image_data_format="pt")[0]

        prompt_token_ids_single_img, _ = self(test_mm_prompt, None)

        # if the max img resolution results in a number of tokens greater then
        # input_seq_len, we keep lowering the resolution such as to find the
        # max resolution such as it does not exceed the input_seq_len
        while len(prompt_token_ids_single_img) > input_seq_len:
            # reduce img resolution
            img_max_dim = img_max_dim >> 1

            image = self.get_dummy_image(max_width=img_max_dim,
                                         max_height=img_max_dim)

            test_mm_prompt = default_multimodal_input_loader(
                tokenizer=self.tokenizer,
                model_dir=self.model_path,
                model_type=self.model_config.model_type,
                modality="image",
                prompts=[text],
                media=[[image]],
                image_data_format="pt")[0]

            prompt_token_ids_single_img, _ = self(test_mm_prompt, None)

        len_prompt_tokens_ids = len(prompt_token_ids_single_img)
        # There are corner cases where if we strictly try to generate a text based
        # on how many tokens we need to complete the input_seq_len, the output of
        # default_multimodal_input_loader may give more tokens then the input_seq_len and this
        # can lead to errors.
        # That is why we try to clip the variable text_token_left to a lower threshold
        # but close enough to the actual input_seq_len
        text_generation_perc_threshold = 0.95
        text_token_left = int((input_seq_len - len_prompt_tokens_ids) *
                              text_generation_perc_threshold)

        if text_token_left > 0:
            text = self.get_dummy_text(text_token_left)

        return default_multimodal_input_loader(
            tokenizer=self.tokenizer,
            model_dir=self.model_path,
            model_type=self.model_config.model_type,
            modality="image",
            prompts=[text],
            media=[[image]],
            image_data_format="pt")[0]

    def _preprocess(self, text: dict[str, any], mm_data: dict[str, any],
                    mm_processor_kwargs: Dict[str, Any]):
        images = mm_data.get("image")
        video_datas = mm_data.get("video")
        if video_datas is not None:
            videos = [video_data.frames for video_data in video_datas]
        else:
            videos = None
        do_rescale = True
        if images and isinstance(images[0], torch.Tensor):
            do_rescale = False
        if videos and isinstance(videos[0][0], torch.Tensor):
            do_rescale = False
            # transformers=4.53.1 does not support GPU video tensors in Qwen2VL processor.
            videos = [[frame.to("cpu") for frame in video] for video in videos]
        return self.processor(text=[text],
                              images=images,
                              videos=videos,
                              padding=True,
                              do_rescale=do_rescale,
                              return_tensors='pt',
                              **mm_processor_kwargs)

    def _postprocess(self, input_ids: torch.IntTensor) -> torch.IntTensor:
        masks = (input_ids == self.model_config.image_token_id) | (
            input_ids == self.model_config.vision_token_id) | (
                input_ids == self.model_config.video_token_id)
        input_ids[masks] = self.tllm_multimodal_token_id
        return input_ids

    def get_mrope_config(
            self,
            input_ids: torch.IntTensor,
            image_grid_thw: torch.LongTensor,
            video_grid_thw: torch.LongTensor,
            attention_mask: torch.Tensor,
            second_per_grid_ts: torch.Tensor = None) -> dict[str, torch.Tensor]:
        mrope_position_ids, mrope_position_deltas = Qwen2VLInputProcessorBase.get_rope_index(
            self.model_config, input_ids, image_grid_thw, video_grid_thw,
            attention_mask, second_per_grid_ts)

        mrope_config = {}
        mrope_config['mrope_position_ids'] = mrope_position_ids.to(
            'cpu').clone()
        mrope_config['mrope_position_deltas'] = mrope_position_deltas.to(
            'cpu').to(torch.int32).clone()
        return mrope_config

    @nvtx_range("Qwen2VLInputProcessorBase forward()")
    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = inputs.get("prompt"), \
                        inputs.get("multi_modal_data", {}), inputs.get("mm_processor_kwargs", {})
        processed_inputs = self._preprocess(text_prompt, mm_data,
                                            mm_processor_kwargs)

        multimodal_data = {}
        pixel_values = processed_inputs.get('pixel_values', None)
        if pixel_values is not None:
            multimodal_data["image"] = {
                "pixel_values": pixel_values,
                "image_grid_thw": processed_inputs.get('image_grid_thw')
            }

        pixel_values_videos = processed_inputs.get('pixel_values_videos', None)
        if pixel_values_videos is not None:
            multimodal_data["video"] = {
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": processed_inputs.get('video_grid_thw')
            }

        # NOTE: Even on the text-only prompts, we still need 'mrope_position_ids'.
        mrope_config = self.get_mrope_config(
            processed_inputs['input_ids'],
            processed_inputs.get('image_grid_thw', None),
            processed_inputs.get('video_grid_thw', None),
            processed_inputs.get('attention_mask', None),
            processed_inputs.get('second_per_grid_ts', None))
        multimodal_data["mrope_config"] = mrope_config

        fused_input_ids = processed_inputs['input_ids'][0]
        if mm_data:
            fused_input_ids = self._postprocess(fused_input_ids)

        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


class Qwen2VisionModelBase(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 model_class: Union[type[PreTrainedModel],
                                    type[torch.nn.Module]]):
        super().__init__()
        config = model_config.pretrained_config.vision_config
        config.torch_dtype = model_config.pretrained_config.torch_dtype
        self.model_config = model_config
        self.model_dtype = config.torch_dtype

        if model_class in [
                Qwen2VisionTransformerPretrainedModel,
                Qwen2_5_VisionTransformerPretrainedModel
        ]:
            # NOTE: For Qwen2VL, we use flash_attention_2 for attention implementation to avoid OOM issue.
            config._attn_implementation = 'flash_attention_2'
            self.visual = model_class(config).to(self.model_dtype).eval()
        elif model_class == Qwen2_5_VisionModel:
            self.visual = model_class(self.model_config).to(
                self.model_dtype).eval()
        else:
            raise NotImplementedError(
                f"Model class {model_class} not implemented")

        self.post_config()

    def post_config(self):
        self.config = self.model_config.pretrained_config.vision_config

    def _parse_and_batch_multimodal_data(
        self, multimodal_params: List[MultimodalParams]
    ) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:

        pixel_values_list = []
        pixel_values_videos_list = []
        image_grid_thw_list = []
        video_grid_thw_list = []

        for multimodal_param in multimodal_params:
            # Process images if present
            if multimodal_param.multimodal_data.get("image") is not None:
                pixel_values_list.append(
                    multimodal_param.multimodal_data["image"]["pixel_values"])
                image_grid_thw_list.append(
                    multimodal_param.multimodal_data["image"]["image_grid_thw"])

            # Process videos if present
            if multimodal_param.multimodal_data.get("video") is not None:
                pixel_values_videos_list.append(
                    multimodal_param.multimodal_data["video"]
                    ["pixel_values_videos"])
                video_grid_thw_list.append(
                    multimodal_param.multimodal_data["video"]["video_grid_thw"])

        # Concatenate tensors
        mm_content_dict = {}
        if pixel_values_list:
            mm_content_dict["pixel_values"] = torch.cat(
                pixel_values_list,
                dim=0) if len(pixel_values_list) > 1 else pixel_values_list[0]
        if pixel_values_videos_list:
            mm_content_dict["pixel_values_videos"] = torch.cat(
                pixel_values_videos_list,
                dim=0) if len(pixel_values_videos_list
                              ) > 1 else pixel_values_videos_list[0]

        # Prepare extra data
        mm_extra_data = {}
        if image_grid_thw_list:
            mm_extra_data["image_grid_thw"] = torch.cat(
                image_grid_thw_list, dim=0) if len(
                    image_grid_thw_list) > 1 else image_grid_thw_list[0]
        if video_grid_thw_list:
            mm_extra_data["video_grid_thw"] = torch.cat(
                video_grid_thw_list, dim=0) if len(
                    video_grid_thw_list) > 1 else video_grid_thw_list[0]

        return mm_content_dict, mm_extra_data

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]):

        mm_content_data, mm_extra_data = self._parse_and_batch_multimodal_data(
            multimodal_params)
        pixel_values = mm_content_data.get("pixel_values", None)
        pixel_values_videos = mm_content_data.get("pixel_values_videos", None)

        image_grid_thw = mm_extra_data.get("image_grid_thw", None)
        video_grid_thw = mm_extra_data.get("video_grid_thw", None)

        embeds = []
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model_dtype)
            embed = self.visual(pixel_values, grid_thw=image_grid_thw)
            embeds.append(embed)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(self.model_dtype)
            embeds.append(
                self.visual(pixel_values_videos, grid_thw=video_grid_thw))
        return embeds


class Qwen2_5_VLVisionAttention(Attention):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int) -> None:

        config = model_config.pretrained_config.vision_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            max_position_embeddings=model_config.pretrained_config.
            max_position_embeddings,
            bias=True,
            pos_embd_params=None,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        **kwargs,
    ) -> torch.Tensor:
        # NOTE: Need separate Attention forward() for Qwen2.5-VL for multiple reasons
        # 1. We don't have the route for handing over position_embeddings to the Attention forward()
        # 2. Could not override the apply_rope() as we don't have the position_ids in the Vision Attention's rotary embedding.
        # (TODO: yechank-nvidia) Make OOTO path more modular and reusable for Attention's Rotary Embedding.

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv, None, None
        q, k, v = self.split_qkv(q, k, v)
        seq_length = hidden_states.shape[0]
        q, k, v = (qkv.reshape(seq_length, 3, self.num_heads,
                               -1).permute(1, 0, 2, 3).unbind(0))

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k, v = q.reshape(seq_length,
                            -1), k.reshape(seq_length,
                                           -1), v.reshape(seq_length, -1)
        q, k, v = self.convert_qkv(q, k, v)
        output = self.forward_impl(q=q,
                                   k=k,
                                   v=v,
                                   attn_metadata=attn_metadata,
                                   attention_mask=PredefinedAttentionMask.FULL,
                                   attention_window_size=None,
                                   attention_mask_data=None,
                                   mrope_config=None,
                                   attention_sinks=None)
        attn_output = self.o_proj(output, layer_idx=self.layer_idx)
        return attn_output


class Qwen2_5_VLVisionBlock(torch.nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: Optional[int]):
        super().__init__()
        config = model_config.pretrained_config.vision_config
        self.norm1 = RMSNorm(hidden_size=config.hidden_size,
                             eps=model_config.pretrained_config.rms_norm_eps,
                             dtype=model_config.pretrained_config.torch_dtype)
        self.norm2 = RMSNorm(hidden_size=config.hidden_size,
                             eps=model_config.pretrained_config.rms_norm_eps,
                             dtype=model_config.pretrained_config.torch_dtype)
        self.attn = Qwen2_5_VLVisionAttention(model_config, layer_idx)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen2_5_VLPatchMerger(torch.nn.Module):

    def __init__(self,
                 model_config: ModelConfig[PretrainedConfig],
                 spatial_merge_size: int = 2) -> None:
        super().__init__()
        config = model_config.pretrained_config.vision_config
        dim = config.out_hidden_size
        context_dim = config.hidden_size
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(hidden_size=context_dim,
                            eps=model_config.pretrained_config.rms_norm_eps,
                            dtype=model_config.pretrained_config.torch_dtype)
        self.mlp = torch.nn.Sequential(
            Linear(in_features=self.hidden_size,
                   out_features=self.hidden_size,
                   bias=True,
                   dtype=model_config.pretrained_config.torch_dtype,
                   mapping=model_config.mapping),
            torch.nn.GELU(),
            Linear(in_features=self.hidden_size,
                   out_features=dim,
                   bias=True,
                   dtype=model_config.pretrained_config.torch_dtype,
                   mapping=model_config.mapping),
        )

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)
        x = self.mlp(x)
        return x


class Qwen2_5_VisionModel(torch.nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        config = model_config.pretrained_config.vision_config
        super().__init__()

        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = torch.nn.ModuleList([
            Qwen2_5_VLVisionBlock(model_config, layer_idx=layer_idx)
            for layer_idx in range(config.depth)
        ])
        self.merger = Qwen2_5_VLPatchMerger(model_config, )
        self.metadata_cls = get_attention_backend(
            model_config.attn_backend).Metadata

        self.full_attn_metadata = self.metadata_cls(
            max_num_requests=8192,  # TODO: Make this dynamic
            max_num_tokens=8192,  # TODO: Make this dynamic
            kv_cache_manager=None,
        )
        self.window_attn_metadata = self.metadata_cls(
            max_num_requests=8192,  # TODO: Make this dynamic
            max_num_tokens=8192,  # TODO: Make this dynamic
            kv_cache_manager=None,
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        seq_lens = []
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant",
                                 PAD_INDEX)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != PAD_INDEX).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != PAD_INDEX]
            window_index.append(index_new + window_index_id)
            seqlens = seqlens * self.spatial_merge_unit
            seq_lens.extend(seqlens.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, seq_lens

    def prepare_attn_metadata(self, seq_lens, attn_metadata: AttentionMetadata):
        # NOTE: The single prompt is divided into multiple seq_lens, so pretending have many batch_sizes.
        batch_size = len(seq_lens)
        prompt_lens = seq_lens
        seq_lens = torch.tensor(seq_lens, dtype=torch.int, pin_memory=True)
        request_ids = list(range(1, batch_size + 1))

        attn_metadata.num_contexts = batch_size
        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lens
        attn_metadata.seq_lens = seq_lens
        attn_metadata.max_seq_len = seq_lens.max().item()
        attn_metadata.prepare()
        return attn_metadata

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor,
                **kwargs) -> torch.Tensor:
        window_index, window_seq_lens = self.get_window_index(grid_thw)
        seq_lens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                           grid_thw[:, 0]).tolist()
        reverse_indices = torch.argsort(window_index)

        # Getting positional embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        full_attn_metadata = self.prepare_attn_metadata(seq_lens,
                                                        self.full_attn_metadata)
        window_attn_metadata = self.prepare_attn_metadata(
            window_seq_lens, self.window_attn_metadata)

        # From this point, pure GPU operation
        hidden_states = self.patch_embed(hidden_states)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, block in enumerate(self.blocks):

            if layer_num in self.fullatt_block_indexes:
                attn_metadata = full_attn_metadata
            else:
                attn_metadata = window_attn_metadata

            hidden_states = block(
                hidden_states,
                attn_metadata=attn_metadata,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class Qwen2VLModelBase(PreTrainedModel):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        model_config.pretrained_config.rope_scaling['type'] = 'mrope'
        self.original_arch = model_config.pretrained_config.architectures[0]
        # NOTE: Setting disable_fuse_rope to True to do mrope fusion in the model engine by pre-computing rotary_cos_sin in the model engine
        disabble_fuse_rope = kwargs.get('disable_fuse_rope', False)
        model_config.pretrained_config.text_config.disable_fuse_rope = disabble_fuse_rope
        config = model_config.pretrained_config

        assert model_config.attn_backend == 'TRTLLM', "Qwen2/2.5-VL only supports TRTLLM backend now"
        super().__init__(config)
        if not disabble_fuse_rope:
            self.init_mrope_embedding(model_config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        if not DISAGG:
            self.mm_encoder = Qwen2VisionModelBase(
                model_config, kwargs.get('vision_model_class', None)).eval()

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = config.text_config
        llm_model_config.pretrained_config.architectures = ["Qwen2ForCausalLM"]

        self.llm = AutoModelForCausalLM.from_config(llm_model_config)
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        logger.info(f"{self.dtype=} {self.model_dtype=}")
        self.post_config()
        self.is_loaded = True

    def init_mrope_embedding(self, model_config: ModelConfig[PretrainedConfig]):
        config = model_config.pretrained_config
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.from_string(config.rope_scaling["type"]),
            rope=RopeParams.from_config(config),
            mrope_section=config.rope_scaling.get('mrope_section', None))
        self.rotary_emb = MRotaryEmbedding(
            pos_embd_params.rope,
            head_dim=config.hidden_size // config.num_attention_heads,
            is_neox=pos_embd_params.is_neox,
            mrope_section=pos_embd_params.mrope_section,
        ).to('cuda')
        self.mrope_position_ids_padding_cuda = torch.zeros((
            3,
            1,
            config.max_position_embeddings,
        ),
                                                           dtype=torch.int32,
                                                           device='cuda')

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        pass

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    @nvtx_range("Qwen2.5-VL prepare_mrope_config")
    def prepare_mrope_config(self, multimodal_params: List[MultimodalParams],
                             num_context_requests: int):
        mrope_config = {}
        mrope_rotary_cos_sin = []
        mrope_position_deltas = []
        for multimodal_param in multimodal_params[:num_context_requests]:
            if multimodal_param.multimodal_data.get('mrope_config') is not None:
                with nvtx_range("Qwen2.5-VL get_cos_sin"):
                    if multimodal_param.multimodal_data['mrope_config'].get(
                            'mrope_position_ids') is not None:
                        mrope_position_ids = multimodal_param.multimodal_data[
                            'mrope_config']['mrope_position_ids']

                        self.mrope_position_ids_padding_cuda[:, :, :
                                                             mrope_position_ids.
                                                             shape[
                                                                 -1]] = mrope_position_ids
                        self.mrope_position_ids_padding_cuda[:, :,
                                                             mrope_position_ids.
                                                             shape[-1]:] = 0

                        cos, sin = self.rotary_emb.get_cos_sin(
                            self.mrope_position_ids_padding_cuda)
                        concat_cos_sin = torch.stack((cos, sin), dim=-1)
                        concat_cos_sin = concat_cos_sin.reshape(
                            concat_cos_sin.shape[0], -1)
                        mrope_rotary_cos_sin.append(concat_cos_sin)

        for multimodal_param in multimodal_params[num_context_requests:]:
            if multimodal_param.multimodal_data.get('mrope_config') is not None:
                if multimodal_param.multimodal_data['mrope_config'].get(
                        'mrope_position_deltas') is not None:
                    mrope_position_deltas.append(
                        multimodal_param.multimodal_data['mrope_config']
                        ['mrope_position_deltas'])

        with nvtx_range("Qwen2.5-VL concat mrope_rotary_cos_sin"):
            if mrope_rotary_cos_sin:
                mrope_config['mrope_rotary_cos_sin'] = torch.cat(
                    mrope_rotary_cos_sin, dim=0)
        with nvtx_range("Qwen2.5-VL concat mrope_position_deltas"):
            if mrope_position_deltas:
                mrope_config['mrope_position_deltas'] = torch.cat(
                    mrope_position_deltas, dim=0)

        return mrope_config

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
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
        mrope_config = {}

        if len(multimodal_params) > 0:
            if not DISAGG:
                mm_embeds = get_multimodal_embeddings(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=multimodal_params[:num_context_requests])
            else:
                raise NotImplementedError(
                    "Qwen2VLModel does not support disaggregated inference yet. Please unset "
                    f"the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )

            mm_embeds = find_input_mm_embeds(
                mm_embeds, multimodal_params[:num_context_requests])

            if not self.model_config.pretrained_config.disable_fuse_rope:
                mrope_config = self.prepare_mrope_config(
                    multimodal_params, num_context_requests)

        input_ids, input_embeds = fuse_input_embeds(self.llm.model.embed_tokens,
                                                    input_ids, mm_embeds,
                                                    **kwargs)
        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            mrope_config=mrope_config)

        logger.debug(f'output shape: {output_prob.shape}')
        return output_prob


@register_vision_encoder(Qwen2VisionModelBase,
                         vlm_base_model=Qwen2VisionTransformerPretrainedModel)
@register_auto_model("Qwen2VLForConditionalGeneration")
@register_input_processor(
    Qwen2VLInputProcessorBase,
    model_type="qwen2_vl",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>"
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ))
class Qwen2VLModel(Qwen2VLModelBase):

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs):
        # NOTE: Since Qwen2-VL is outdated model, we leave it as HF implementation.
        kwargs['vision_model_class'] = Qwen2VisionTransformerPretrainedModel
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values", "image.image_grid_thw",
            "video.pixel_values_videos", "video.video_grid_thw",
            "multimodal_embedding"
        ]

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        if not DISAGG:
            vision_encoder_weights = process_weights(weights, "visual")
            self.mm_encoder.load_state_dict(vision_encoder_weights, strict=True)

        self.llm.load_weights(weights, weight_mapper)


def getSMVersion():
    prop = torch.cuda.get_device_properties(0)
    sm_version = prop.major * 10 + prop.minor
    return sm_version


get_sm_version = getSMVersion()
if get_sm_version >= 100:
    # NOTE: Qwen2.5-VL with SM 100 and above uses HF's implementation due to lacking of TRT-LLM's Attention kernel.
    QWEN2_5_VL_VISION_MODEL_CLASS = Qwen2_5_VisionTransformerPretrainedModel
else:
    QWEN2_5_VL_VISION_MODEL_CLASS = Qwen2_5_VisionModel


@register_vision_encoder(Qwen2VisionModelBase,
                         vlm_base_model=QWEN2_5_VL_VISION_MODEL_CLASS)
@register_auto_model("Qwen2_5_VLForConditionalGeneration")
@register_input_processor(
    Qwen2VLInputProcessorBase,
    model_type="qwen2_5_vl",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>"
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ))
class Qwen2_5_VLModel(Qwen2VLModelBase):

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs):
        kwargs['vision_model_class'] = QWEN2_5_VL_VISION_MODEL_CLASS
        kwargs[
            'disable_fuse_rope'] = False  # TODO: Make this ModelConfig's argument
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        if get_sm_version >= 100:
            return [
                "image.pixel_values", "video.pixel_values_videos",
                "image.image_grid_thw", "video.video_grid_thw",
                "multimodal_embedding"
            ]
        else:
            return [
                "image.pixel_values", "video.pixel_values_videos",
                "multimodal_embedding"
            ]

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        if not DISAGG:
            if get_sm_version >= 100:
                weight_name_mapping = None
            else:
                # Process vision encoder weights
                weight_name_mapping = {
                    "attn.proj.weight": "attn.o_proj.weight",
                    "attn.proj.bias": "attn.o_proj.bias",
                    "attn.qkv.weight": "attn.qkv_proj.weight",
                    "attn.qkv.bias": "attn.qkv_proj.bias"
                }
            vision_weights = process_weights(weights, "visual",
                                             weight_name_mapping)
            self.mm_encoder.load_state_dict(vision_weights, strict=True)

        self.llm.load_weights(weights, weight_mapper)
