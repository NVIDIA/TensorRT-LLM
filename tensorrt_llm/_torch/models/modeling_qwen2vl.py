import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
    Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    Qwen2VisionTransformerPretrainedModel

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ..._utils import nvtx_range_debug
from ...functional import RopeEmbeddingUtils, RotaryScalingType
from ...inputs import (BaseMultimodalInputProcessor, ExtraProcessedInputs,
                       InputProcessor, MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (find_input_mm_embeds, fuse_input_embeds,
                                        get_multimodal_embeddings)
from .modeling_utils import register_auto_model, register_vision_encoder

DISAGG = os.getenv('TLLM_MULTIMODAL_DISAGGREGATED', '0') == '1'


def filter_weights(prefix, weights: Dict):
    result = {}
    for k, v in weights.items():
        if k.startswith(prefix):
            result[k] = v
        elif k.startswith("model." + prefix):
            result[k[len("model."):]] = v
    return result


class Qwen2VLInputProcessorBase(BaseMultimodalInputProcessor, InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.use_fast = True
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

    def _preprocess(self, text: dict[str, any], mm_data: dict[str, any],
                    mm_processor_kwargs: Dict[str, Any]):
        images = mm_data.get("image")
        videos = mm_data.get("video")
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
        # NOTE: Qwen2-VL's input processor is doing all the work for fusing input_ids with mm_tokens.
        # So, we just replace mm_tokens with expanded out-of-vocab ids
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

    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = inputs.get("prompt"), \
                        inputs.get("multi_modal_data", {}), inputs.get("mm_processor_kwargs", {})
        with nvtx_range_debug("transformers input preprocess"):
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

        # print(f"fused_input_ids: {fused_input_ids}")
        # print(f"multimodal_data['image']['pixel_values'].shape: {multimodal_data['image']['pixel_values'].shape}")
        # print(f"multimodal_data['image']['image_grid_thw']: {multimodal_data['image']['image_grid_thw']}")
        # print(f"multimodal_data['mrope_config']['mrope_position_ids'].shape: {multimodal_data['mrope_config']['mrope_position_ids'].shape}")
        # print(f"multimodal_data['mrope_config']['mrope_position_deltas'].shape: {multimodal_data['mrope_config']['mrope_position_deltas'].shape}")
        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


class Qwen2VisionModelBase(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 model_class: type[PreTrainedModel]):
        super().__init__()
        config = model_config.pretrained_config.vision_config
        config.torch_dtype = model_config.pretrained_config.torch_dtype
        self.model_dtype = config.torch_dtype

        # TODO: Change the model class to TRT-LLM's Qwen2VisionModel
        # NOTE: Using attn_implementation='flash_attention_2' to avoid the issue of vision model's GPU OOM.
        config._attn_implementation = 'flash_attention_2'
        self.visual = model_class._from_config(config).eval()

        # self.patch_embed = Qwen2_5_VisionPatchEmbed(
        #     patch_size=config.patch_size,
        #     temporal_patch_size=config.temporal_patch_size,
        #     in_channels=config.in_channels,
        #     embed_dim=config.hidden_size,
        # )

        # head_dim = config.hidden_size // config.num_heads
        # self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        # self.blocks = torch.nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        # self.merger = Qwen2_5_VLPatchMerger(
        #     dim=config.out_hidden_size,
        #     context_dim=config.hidden_size,
        #     spatial_merge_size=config.spatial_merge_size,
        # )

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
            pixel_values = pixel_values.to(self.visual.dtype)
            embeds.append(self.visual(pixel_values, grid_thw=image_grid_thw))

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(self.visual.dtype)
            embeds.append(
                self.visual(pixel_values_videos, grid_thw=video_grid_thw))
        return embeds


class Qwen2VLModelBase(PreTrainedModel):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        model_config.pretrained_config.rope_scaling['type'] = 'mrope'

        # NOTE: Setting disable_fuse_rope to True to do mrope fusion in the model engine by pre-computing rotary_cos_sin in the model engine
        model_config.pretrained_config.text_config.disable_fuse_rope = kwargs.get(
            'disable_fuse_rope', False)  #True #False

        config = model_config.pretrained_config

        assert model_config.attn_backend == 'TRTLLM', "Qwen2/2.5-VL only supports TRTLLM backend now"
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        if not DISAGG:
            self.mm_encoder = Qwen2VisionModelBase(
                model_config, kwargs.get('vision_model_class', None)).eval()

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = config.text_config
        # NOTE: Assigning name for LLM
        llm_model_config.pretrained_config.architectures = ["Qwen2ForCausalLM"]

        self.llm = AutoModelForCausalLM.from_config(llm_model_config)
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        logger.info(f"{self.dtype=} {self.model_dtype=}")
        self.post_config()
        self.is_loaded = True

    def init_rotary_cos_sin_ori(self):
        _, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            num_pos=self.model_config.pretrained_config.max_position_embeddings,
            dim=int(self.model_config.pretrained_config.hidden_size /
                    self.model_config.pretrained_config.num_attention_heads),
            theta=float(self.model_config.pretrained_config.rope_theta),
            scale_type=RotaryScalingType.mrope)
        self.rotary_cos_sin = torch.from_numpy(rotary_cos_sin).to(self.device)
        self.rotary_cos_sin = self.rotary_cos_sin.reshape(
            self.model_config.pretrained_config.max_position_embeddings,
            int(self.model_config.pretrained_config.hidden_size /
                self.model_config.pretrained_config.num_attention_heads / 2), 2)

        self.cos_ori = self.rotary_cos_sin[:, :, 0]
        self.sin_ori = self.rotary_cos_sin[:, :, 1]

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        if not DISAGG:
            print(f"weights.keys(): {weights.keys()}")
            vision_encoder_weights = filter_weights("visual", weights)
            print(
                f"vision_encoder_weights.keys(): {vision_encoder_weights.keys()}"
            )
            self.mm_encoder.load_state_dict(vision_encoder_weights, strict=True)

        self.llm.load_weights(weights, weight_mapper)
        if not self.model_config.pretrained_config.disable_fuse_rope:
            self.init_rotary_cos_sin_ori()

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    # def _process_mrope_position_ids(self, multimodal_params: List[MultimodalParams],
    #                                 num_context_requests: int,
    #                                 position_ids: torch.Tensor) -> torch.Tensor:
    #     """
    #     Process mrope position IDs and deltas from multimodal parameters.

    #     Args:
    #         multimodal_params: List of multimodal parameters
    #         num_context_requests: Number of context requests
    #         position_ids: Original position IDs from LLM

    #     Returns:
    #         torch.Tensor: Processed position IDs with mrope adjustments
    #     """
    #     # Pre-extract mrope configs in single pass to avoid redundant dict lookups
    #     ctx_mrope_configs = []
    #     gen_mrope_configs = []

    #     # Single loop to extract all mrope configs
    #     for i, param in enumerate(multimodal_params):
    #         mrope_config = param.multimodal_data.get('mrope_config')
    #         if mrope_config:
    #             if i < num_context_requests:
    #                 ctx_mrope_configs.append(mrope_config)
    #                 print(f"ctx_mrope_configs.mrope_position_ids.shape: {ctx_mrope_configs[0]['mrope_position_ids'].shape}")
    #             else:
    #                 gen_mrope_configs.append(mrope_config)

    #     # Process context phase - batch extract position_ids
    #     ctx_position_ids = [
    #         config['mrope_position_ids']
    #         for config in ctx_mrope_configs
    #         if config.get('mrope_position_ids') is not None
    #     ]

    #     # Process generation phase - batch extract position_deltas
    #     gen_position_deltas = [
    #         config['mrope_position_deltas']
    #         for config in gen_mrope_configs
    #         if config.get('mrope_position_deltas') is not None
    #     ]

    #     # Batch concatenate context position_ids if any exist
    #     new_position_ids_ctx = None
    #     num_ctx_tokens = 0
    #     if ctx_position_ids:
    #         new_position_ids_ctx = torch.cat(ctx_position_ids, dim=-1)
    #         num_ctx_tokens = new_position_ids_ctx.shape[-1]

    #     # Batch process generation deltas if any exist
    #     new_position_ids_gen = None
    #     if gen_position_deltas:
    #         position_ids_deltas = torch.cat(gen_position_deltas, dim=1)
    #         ori_position_ids_gen = position_ids[:, num_ctx_tokens:]
    #         new_position_ids_gen = (ori_position_ids_gen + position_ids_deltas).expand(3, 1, -1)

    #     # Efficiently combine results - avoid unnecessary concatenations
    #     if new_position_ids_ctx is not None:
    #         if new_position_ids_gen is not None:
    #             return torch.cat((new_position_ids_ctx, new_position_ids_gen), dim=-1)
    #         else:
    #             return new_position_ids_ctx
    #     elif new_position_ids_gen is not None:
    #         return new_position_ids_gen
    #     else:
    #         return position_ids

    def prepare_mrope_config(self, multimodal_params: List[MultimodalParams],
                             num_context_requests: int):
        mrope_config = {}

        mrope_position_ids = []
        mrope_position_deltas = []
        for multimodal_param in multimodal_params[:num_context_requests]:
            if multimodal_param.multimodal_data.get('mrope_config') is not None:
                if multimodal_param.multimodal_data['mrope_config'].get(
                        'mrope_position_ids') is not None:
                    mrope_position_ids.append(
                        multimodal_param.multimodal_data['mrope_config']
                        ['mrope_position_ids'])

        for multimodal_param in multimodal_params[num_context_requests:]:
            if multimodal_param.multimodal_data.get('mrope_config') is not None:
                if multimodal_param.multimodal_data['mrope_config'].get(
                        'mrope_position_deltas') is not None:
                    mrope_position_deltas.append(
                        multimodal_param.multimodal_data['mrope_config']
                        ['mrope_position_deltas'])

        if mrope_position_ids:
            # mrope_position_ids = torch.cat(mrope_position_ids, dim=-1)
            # mrope_position_ids = mrope_position_ids.transpose(1, 0)

            # # # TODO: Test whether padding is needed
            # # mrope_position_ids_padding = torch.zeros(
            # #     mrope_position_ids.shape[:-1] +
            # #     (self.model_config.pretrained_config.
            # #     max_position_embeddings, ),
            # #     dtype=torch.int32,
            # #     device=mrope_position_ids.device)
            # # mrope_position_ids_padding[:, :, :mrope_position_ids.
            # #                         shape[-1]] = mrope_position_ids

            # # mrope_position_ids_padding = mrope_position_ids_padding.to(
            # #     self.cos_ori.device)
            # cos = self.cos_ori[mrope_position_ids]
            # sin = self.sin_ori[mrope_position_ids]

            # mrope_section = [16, 24, 24]
            # cos = torch.cat([
            #     m[:, i % 3]
            #     for i, m in enumerate(cos.split(mrope_section, dim=-1))
            # ],
            #                 dim=-1).unsqueeze(-1)
            # sin = torch.cat([
            #     m[:, i % 3]
            #     for i, m in enumerate(sin.split(mrope_section, dim=-1))
            # ],
            #                 dim=-1).unsqueeze(-1)
            # concat_cos_sin = torch.concatenate((cos, sin), axis=-1)
            # concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0],
            #                                         -1)
            # mrope_config['mrope_rotary_cos_sin'] = concat_cos_sin.to(
            #     self.device)

            # mrope_position_ids = torch.cat(mrope_position_ids, dim=-1)

            concat_cos_sin_list = []
            for mrope_position_id in mrope_position_ids:
                mrope_position_id = mrope_position_id.transpose(1, 0)

                # TODO: Test whether padding is needed
                mrope_position_ids_padding = torch.zeros(
                    mrope_position_id.shape[:-1] +
                    (self.model_config.pretrained_config.
                     max_position_embeddings, ),
                    dtype=torch.int32,
                    device=mrope_position_id.device)
                mrope_position_ids_padding[:, :, :mrope_position_id.
                                           shape[-1]] = mrope_position_id

                mrope_position_ids_padding = mrope_position_ids_padding.to(
                    self.cos_ori.device)
                cos = self.cos_ori[mrope_position_ids_padding]
                sin = self.sin_ori[mrope_position_ids_padding]

                mrope_section = [16, 24, 24]
                cos = torch.cat([
                    m[:, i % 3]
                    for i, m in enumerate(cos.split(mrope_section, dim=-1))
                ],
                                dim=-1).unsqueeze(-1)
                sin = torch.cat([
                    m[:, i % 3]
                    for i, m in enumerate(sin.split(mrope_section, dim=-1))
                ],
                                dim=-1).unsqueeze(-1)
                concat_cos_sin = torch.concatenate((cos, sin), axis=-1)
                concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0],
                                                        -1)
                concat_cos_sin_list.append(concat_cos_sin)
            mrope_config['mrope_rotary_cos_sin'] = torch.cat(
                concat_cos_sin_list, dim=0).to(self.device)

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
                         vlm_base_model=Qwen2VLForConditionalGeneration)
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
        kwargs['vision_model_class'] = Qwen2VisionTransformerPretrainedModel
        super().__init__(model_config, *args, **kwargs)
    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values", "image.image_grid_thw",
            "video.pixel_values_videos", "video.video_grid_thw",
            "multimodal_embedding", "mrope_config.mrope_position_ids"
        ]

@register_vision_encoder(Qwen2VisionModelBase,
                         vlm_base_model=Qwen2_5_VLForConditionalGeneration)
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
        kwargs['vision_model_class'] = Qwen2_5_VisionTransformerPretrainedModel
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values", "image.image_grid_thw",
            "video.pixel_values_videos", "video.video_grid_thw",
            "multimodal_embedding", "mrope_config.mrope_position_ids"
        ]
