import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel, Qwen2_5_VLForConditionalGeneration,
                          Qwen2VLForConditionalGeneration)

from ...functional import RopeEmbeddingUtils, RotaryScalingType
from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import register_auto_model


class Qwen2VLInputProcessorBase(InputProcessor):

    def __init__(self, model_path: str, model_config: PretrainedConfig,
                 tokenizer: AutoTokenizer):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(model_path,
                                                       use_fast=False)

        # NOTE: Using attn_implementation='flash_attention_2' to avoid the issue of vision model's GPU OOM.
        model = self.get_model_class().from_pretrained(
            model_path,
            torch_dtype=model_config.torch_dtype,
            attn_implementation='flash_attention_2')
        self.device = 'cuda'
        self.visual = model.visual.to(self.device)
        self._post_init_()

    @classmethod
    def get_model_class(cls) -> type[PreTrainedModel]:
        raise NotImplementedError()

    @classmethod
    def get_rope_index(
        cls,
        model_config: PretrainedConfig,
        input_ids: Optional[torch.LongTensor] = None,
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

    def _post_init_(self):
        _, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            num_pos=self.model_config.max_position_embeddings,
            dim=int(self.model_config.hidden_size /
                    self.model_config.num_attention_heads),
            theta=float(self.model_config.rope_theta),
            scale_type=RotaryScalingType.mrope)
        self.rotary_cos_sin = torch.from_numpy(rotary_cos_sin).to(self.device)
        self.rotary_cos_sin = self.rotary_cos_sin.reshape(
            self.model_config.max_position_embeddings,
            int(self.model_config.hidden_size /
                self.model_config.num_attention_heads / 2), 2)

        self.cos_ori = self.rotary_cos_sin[:, :, 0]
        self.sin_ori = self.rotary_cos_sin[:, :, 1]

    def _preprocess(self, text: dict[str, any], mm_data: dict[str, any],
                    mm_processor_kwargs: Dict[str, Any]):
        return self.processor(text=[text],
                              images=mm_data.get("image", None),
                              videos=mm_data.get("video", None),
                              padding=True,
                              return_tensors='pt',
                              **mm_processor_kwargs)

    def _process(self, pixel_values: torch.Tensor,
                 pixel_values_videos: torch.Tensor,
                 image_grid_thw: torch.Tensor,
                 video_grid_thw: torch.Tensor) -> torch.Tensor:
        embeds = []

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.visual.dtype)
            embeds.append(self.visual(pixel_values, grid_thw=image_grid_thw))

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(self.visual.dtype)
            embeds.append(
                self.visual(pixel_values_videos, grid_thw=video_grid_thw))

        if embeds:
            return torch.cat(embeds, dim=1)
        return None

    def _postprocess(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        # NOTE: Qwen2-VL's input processor is doing all the work for fusing input_ids with mm_tokens. So, we just replace mm_tokens with expanded out-of-vocab ids

        masks = (input_ids == self.model_config.image_token_id) | (
            input_ids == self.model_config.vision_token_id) | (
                input_ids == self.model_config.video_token_id)
        cumulative_counts = masks.cumsum(dim=-1)
        values = (self.model_config.vocab_size - 1) + cumulative_counts
        input_ids[masks] = values[masks]
        return input_ids

    def get_mrope_config(
            self,
            input_ids: torch.LongTensor,
            image_grid_thw: torch.LongTensor,
            video_grid_thw: torch.LongTensor,
            attention_mask: torch.Tensor,
            second_per_grid_ts: torch.Tensor = None) -> dict[str, torch.Tensor]:
        mrope_position_ids, mrope_position_deltas = self.__class__.get_rope_index(
            self.model_config, input_ids, image_grid_thw, video_grid_thw,
            attention_mask, second_per_grid_ts)

        mrope_position_ids = mrope_position_ids.transpose(1, 0)
        mrope_position_ids_padding = torch.zeros(
            mrope_position_ids.shape[:-1] +
            (self.model_config.max_position_embeddings, ),
            dtype=torch.int32,
            device=input_ids.device)
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
        mrope_config = {}
        mrope_config['mrope_rotary_cos_sin'] = concat_cos_sin
        mrope_config['mrope_position_deltas'] = mrope_position_deltas
        return mrope_config

    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = inputs.get("prompt"), \
                        inputs.get("multi_modal_data"), inputs.get("mm_processor_kwargs", {})

        # NOTE: Since we are passed in Tensor images, we don't need to rescale them.
        mm_processor_kwargs['do_rescale'] = False
        processed_inputs = self._preprocess(text_prompt, mm_data,
                                            mm_processor_kwargs).to(self.device)

        mm_features = self._process(
            processed_inputs.get('pixel_values', None),
            processed_inputs.get('pixel_values_videos', None),
            processed_inputs.get('image_grid_thw', None),
            processed_inputs.get('video_grid_thw', None))

        input_ids = processed_inputs['input_ids']

        mrope_config = self.get_mrope_config(
            input_ids, processed_inputs.get('image_grid_thw', None),
            processed_inputs.get('video_grid_thw', None),
            processed_inputs.get('attention_mask', None),
            processed_inputs.get('second_per_grid_ts', None))

        fused_input_ids = self._postprocess(input_ids[0])

        return fused_input_ids.to(torch.int32).tolist(), {
            "prompt_tuning_config": [mm_features, None, None],
            "mrope_config": mrope_config
        }


class Qwen2VLInputProcessor(Qwen2VLInputProcessorBase):

    @classmethod
    def get_model_class(cls):
        return Qwen2VLForConditionalGeneration


class Qwen2_5_VLInputProcessor(Qwen2VLInputProcessorBase):

    @classmethod
    def get_model_class(cls):
        return Qwen2_5_VLForConditionalGeneration


class Qwen2VLModelBase(PreTrainedModel):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        model_config.pretrained_config.rope_scaling['type'] = 'mrope'
        config = model_config.pretrained_config

        assert model_config.attn_backend == 'TRTLLM', "Qwen2VL only supports TRTLLM backend now"
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config.architectures = ["Qwen2ForCausalLM"]
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)
        self.vocab_size = config.vocab_size
        self.model_dtype = getattr(config, "torch_dtype", torch.float16)
        logger.info(f"{self.dtype=} {self.model_dtype=}")
        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):
        self.llm.load_weights(weights)

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

        mm_embed = kwargs.get("multi_modal_data", [])

        error_msg = "Number of multimodal features (if provided) should be equal to number of context requests"
        assert mm_embed == [] or len(
            mm_embed) == num_context_requests, error_msg

        input_ids, input_embeds = fuse_input_embeds(self, input_ids, mm_embed)

        mrope_config = kwargs.get("mrope_config", {})
        if mrope_config:
            if mrope_rotary_cos_sin := mrope_config.get('mrope_rotary_cos_sin'):
                mrope_config['mrope_rotary_cos_sin'] = torch.cat(
                    mrope_rotary_cos_sin, dim=0)

            if mrope_position_deltas := mrope_config.get(
                    'mrope_position_deltas'):
                mrope_config['mrope_position_deltas'] = torch.cat(
                    mrope_position_deltas, dim=0)

        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            mrope_config=mrope_config)
        logger.debug(f'output shape: {output_prob.shape}')
        return output_prob


@register_auto_model("Qwen2VLForConditionalGeneration")
@register_input_processor(Qwen2VLInputProcessor)
class Qwen2VLModel(Qwen2VLModelBase):
    pass


@register_auto_model("Qwen2_5_VLForConditionalGeneration")
@register_input_processor(Qwen2_5_VLInputProcessor)
class Qwen2_5_VLModel(Qwen2VLModelBase):
    pass
