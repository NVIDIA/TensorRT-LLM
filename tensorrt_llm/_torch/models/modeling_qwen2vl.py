import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (AutoProcessor, PretrainedConfig, PreTrainedModel,
                          Qwen2VLConfig, Qwen2VLForConditionalGeneration)

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


# Copied from https://github.com/QwenLM/Qwen2-VL/blob/main/qwen2_vl/models/qwen2_vl.py
def get_rope_index(
    config: Qwen2VLConfig,
    input_ids: torch.LongTensor,
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
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
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
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id
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
            # if attention_mask is not None:
            #     input_ids = input_ids[attention_mask[i] == 1]
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
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

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
                                             1, 1,
                                             -1).expand(3, input_ids.shape[0],
                                                        -1))
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


class Qwen2VLInputProcessor(InputProcessor):

    def __init__(self, model_path, model_config: Qwen2VLConfig, tokenizer):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(model_path)

        # NOTE
        # Using attn_implementation='flash_attention_2' to avoid the issue of
        # vision model's GPU OOM.
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=model_config.torch_dtype,
            attn_implementation='flash_attention_2')
        self.device = 'cuda'
        self.visual = model.visual.to(self.device)
        self._post_init_()

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
                 image_grid_thw: torch.Tensor, video_grid_thw: torch.Tensor):

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

    def _postprocess(self, input_ids):
        # Qwen2-VL's input processor is doing all the work for fusing input_ids with mm_tokens
        # So, we just replace mm_tokens with expanded out-of-vocab ids

        masks = (input_ids == self.model_config.image_token_id) | (
            input_ids == self.model_config.vision_token_id) | (
                input_ids == self.model_config.video_token_id)
        cumulative_counts = masks.cumsum(dim=-1)
        values = (self.model_config.vocab_size - 1) + cumulative_counts
        input_ids[masks] = values[masks]
        return input_ids

    def get_mrope_config(self, input_ids, image_grid_thw, video_grid_thw,
                         attention_mask):
        mrope_position_ids, mrope_position_deltas = get_rope_index(
            self.model_config, input_ids, image_grid_thw, video_grid_thw,
            attention_mask)

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

        # NOTE
        # Since we are passed in Tensor images, we don't need to rescale them.
        mm_processor_kwargs['do_rescale'] = False
        processed_inputs = self._preprocess(text_prompt, mm_data,
                                            mm_processor_kwargs).to(self.device)

        mm_features = self._process(
            processed_inputs.get('pixel_values', None),
            processed_inputs.get('pixel_values_videos', None),
            processed_inputs.get('image_grid_thw', None),
            processed_inputs.get('video_grid_thw', None))

        input_ids = processed_inputs['input_ids']

        # Qwen2-VL needs special mrope args
        mrope_config = self.get_mrope_config(
            input_ids, processed_inputs.get('image_grid_thw', None),
            processed_inputs.get('video_grid_thw', None),
            processed_inputs.get('attention_mask', None))

        fused_input_ids = self._postprocess(input_ids[0])

        return fused_input_ids.to(torch.int32).tolist(), {
            "prompt_tuning_config": [mm_features, None, None],
            "mrope_config": mrope_config
        }


@register_auto_model("Qwen2VLForConditionalGeneration")
@register_input_processor(Qwen2VLInputProcessor)
class Qwen2VLModel(PreTrainedModel):

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
        logger.debug(
            f"output_ids: {(output_prob if output_prob.dim() == 2 else output_prob.unsqueeze(0)).argmax(dim=1).tolist()}"
        )
        logger.info(f'output shape: {output_prob.shape}')
        return output_prob
