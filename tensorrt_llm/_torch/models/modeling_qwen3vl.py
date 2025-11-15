import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ..._utils import nvtx_range, nvtx_range_debug
from ...inputs import (
    BaseMultimodalInputProcessor,
    ExtraProcessedInputs,
    InputProcessor,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..modules.embedding import Embedding
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (
    _cache_multimodal_embeddings,
    _get_uncached_multimodal_params,
    filter_mm_token_from_input_ids,
    find_input_mm_embeds,
)
from .modeling_utils import ModelConfig, register_auto_model, register_vision_encoder

DISAGG = os.getenv("TLLM_MULTIMODAL_DISAGGREGATED", "0") == "1"


def process_weights(
    weights: Dict, prefix: str = "visual", weight_name_mapping: Dict[str, str] = None
) -> Dict:
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
            filtered_weights[key[len("model.") :]] = weight

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


class Qwen3VLInputProcessorBase(BaseMultimodalInputProcessor, InputProcessor):
    def __init__(
        self,
        model_path: str,
        model_config: PretrainedConfig,
        tokenizer: AutoTokenizer,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        self.model_config = model_config
        self._dtype = self.model_config.torch_dtype
        self._tokenizer = (
            tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_path)
        )
        self._model_path = model_path
        self._processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True, trust_remote_code=trust_remote_code
        )
        # print(self.model_config)
        self.tllm_multimodal_token_id = self.model_config.text_config.vocab_size + 1
        # temporal patch size for video frames
        self.temporal_patch_size = getattr(model_config.vision_config, "temporal_patch_size", 1)

    @property
    def config(self) -> PretrainedConfig:
        return self.model_config

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_vocab_size(self) -> int:
        """Return the vocab size of the model."""
        return self.model_config.text_config.vocab_size

    # def get_mm_token_ids(self) -> torch.Tensor:
    #     """Get the IDs of all multimodal tokens (placeholders and special tokens alike)."""
    #     return torch.tensor([
    #         # This is the `<|image_pad|>` token id inserted into the prompt that should be replaced with image
    #         # embeddings.
    #         self.processor.image_token_id,
    #         # This is the `<|video_pad|>` token id inserted into the prompt that should be replaced with video
    #         # embeddings.
    #         self.processor.video_token_id,
    #         # This is the `<|vision_start|>` token id to signify the start of vision part.
    #         self.processor.vision_start_token_id,
    #         # This is the `<|vision_end|>` token id to signify the end of vision part.
    #         self.processor.vision_end_token_id,
    #     ])

    @classmethod
    def get_rope_index(
        cls,
        model_config: PretrainedConfig,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start>
        # <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        # print(model_config)
        spatial_merge_size = model_config.vision_config.spatial_merge_size
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
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
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
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

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the
                    # temporal information for videos)
                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _preprocess(
        self, text: dict[str, any], mm_data: dict[str, any], mm_processor_kwargs: Dict[str, Any]
    ):
        images = mm_data.get("image")
        videos = mm_data.get("video")
        do_rescale = True
        if images and isinstance(images[0], torch.Tensor):
            do_rescale = False
        if videos and isinstance(videos[0][0], torch.Tensor):
            do_rescale = False
            videos = [[frame for frame in video] for video in videos]
        return self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            do_rescale=do_rescale,
            return_tensors="pt",
            **mm_processor_kwargs,
        )

    def _postprocess(self, input_ids: torch.IntTensor) -> torch.IntTensor:
        masks = (input_ids == self.model_config.image_token_id) | (
            input_ids == self.model_config.video_token_id
        )
        input_ids[masks] = self.tllm_multimodal_token_id
        return input_ids

    def get_mrope_config(
        self,
        input_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor,
        video_grid_thw: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mrope_position_ids, mrope_position_deltas = Qwen3VLInputProcessorBase.get_rope_index(
            self.model_config, input_ids, image_grid_thw, video_grid_thw, attention_mask
        )

        mrope_config = {}
        mrope_config["mrope_position_ids"] = mrope_position_ids.to("cpu").clone()
        mrope_config["mrope_position_deltas"] = (
            mrope_position_deltas.to("cpu").to(torch.int32).clone()
        )
        return mrope_config

    @nvtx_range("Qwen2VLInputProcessorBase forward()")
    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = (
            inputs.get("prompt"),
            inputs.get("multi_modal_data", {}),
            inputs.get("mm_processor_kwargs", {}),
        )
        with nvtx_range_debug("transformers input preprocess"):
            processed_inputs = self._preprocess(text_prompt, mm_data, mm_processor_kwargs)

        multimodal_data = {}
        pixel_values = processed_inputs.get("pixel_values", None)
        if pixel_values is not None:
            multimodal_data["image"] = {
                "pixel_values": pixel_values,
                "image_grid_thw": processed_inputs.get("image_grid_thw"),
            }

        pixel_values_videos = processed_inputs.get("pixel_values_videos", None)
        if pixel_values_videos is not None:
            multimodal_data["video"] = {
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": processed_inputs.get("video_grid_thw"),
            }

        # NOTE: Even on the text-only prompts, we still need 'mrope_position_ids'.
        mrope_config = self.get_mrope_config(
            processed_inputs["input_ids"],
            processed_inputs.get("image_grid_thw", None),
            processed_inputs.get("video_grid_thw", None),
            processed_inputs.get("attention_mask", None),
        )
        multimodal_data["mrope_config"] = mrope_config

        fused_input_ids = processed_inputs["input_ids"][0]
        if mm_data:
            fused_input_ids = self._postprocess(fused_input_ids)

        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


class Qwen3VisionModelBase(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        model_class: Union[type[PreTrainedModel], type[torch.nn.Module]],
    ):
        super().__init__()
        config = model_config.pretrained_config.vision_config
        config.torch_dtype = model_config.pretrained_config.torch_dtype
        self.model_config = model_config
        self.model_dtype = config.torch_dtype

        if model_class == Qwen3VLVisionModel:
            # NOTE: For hf impl, we use flash_attention_2 for attention implementation to avoid OOM issue.
            config._attn_implementation = "flash_attention_2"
            self.visual = model_class(config).to(self.model_dtype).eval()
        else:
            raise NotImplementedError(f"Model class {model_class} not implemented")

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
            multimodal_data = multimodal_param.multimodal_data
            # Process images if present
            if multimodal_data.get("image") is not None:
                pixel_values_list.append(multimodal_data["image"]["pixel_values"])
                image_grid_thw_list.append(multimodal_data["image"]["image_grid_thw"])

            # Process videos if present
            if multimodal_data.get("video") is not None:
                pixel_values_videos_list.append(multimodal_data["video"]["pixel_values_videos"])
                video_grid_thw_list.append(multimodal_data["video"]["video_grid_thw"])

        # Concatenate tensors
        mm_content_dict = {}
        if pixel_values_list:
            mm_content_dict["pixel_values"] = (
                torch.cat(pixel_values_list, dim=0)
                if len(pixel_values_list) > 1
                else pixel_values_list[0]
            )
        if pixel_values_videos_list:
            mm_content_dict["pixel_values_videos"] = (
                torch.cat(pixel_values_videos_list, dim=0)
                if len(pixel_values_videos_list) > 1
                else pixel_values_videos_list[0]
            )

        # Prepare extra data
        mm_extra_data = {}
        if image_grid_thw_list:
            mm_extra_data["image_grid_thw"] = (
                torch.cat(image_grid_thw_list, dim=0)
                if len(image_grid_thw_list) > 1
                else image_grid_thw_list[0]
            )
        if video_grid_thw_list:
            mm_extra_data["video_grid_thw"] = (
                torch.cat(video_grid_thw_list, dim=0)
                if len(video_grid_thw_list) > 1
                else video_grid_thw_list[0]
            )

        return mm_content_dict, mm_extra_data

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]):
        mm_content_data, mm_extra_data = self._parse_and_batch_multimodal_data(multimodal_params)
        pixel_values = mm_content_data.get("pixel_values", None)
        pixel_values_videos = mm_content_data.get("pixel_values_videos", None)

        image_grid_thw = mm_extra_data.get("image_grid_thw", None)
        video_grid_thw = mm_extra_data.get("video_grid_thw", None)

        embeds = []
        deepstack_embeds = []
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model_dtype)
            image_embeds, deepstack_image_embeds = self.visual(
                pixel_values, grid_thw=image_grid_thw
            )
            # print("shapes", image_embeds.shape, len(deepstack_image_embeds))
            embeds.append(image_embeds)
            deepstack_embeds.append(torch.stack(deepstack_image_embeds))

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(self.model_dtype)
            video_embeds, deepstack_video_embeds = self.visual(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            embeds.append(video_embeds)
            deepstack_embeds.append(torch.stack(deepstack_video_embeds))
        return embeds, deepstack_embeds


class Qwen3VLModelBase(PreTrainedModel):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        model_config.pretrained_config.text_config.rope_scaling["type"] = "mrope"
        self.original_arch = model_config.pretrained_config.architectures[0]
        config = model_config.pretrained_config

        assert model_config.attn_backend == "TRTLLM", "Qwen3-VL only supports TRTLLM backend now"
        super().__init__(config)
        self.init_mrope_embedding(model_config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        if not DISAGG:
            self.mm_encoder = Qwen3VisionModelBase(
                model_config, kwargs.get("vision_model_class", None)
            ).eval()

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = config.text_config
        llm_model_config.pretrained_config.architectures = ["Qwen3ForCausalLM"]

        self.llm = AutoModelForCausalLM.from_config(llm_model_config)
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        logger.info(f"{self.dtype=} {self.model_dtype=}")
        self.post_config()
        self.is_loaded = True

    def init_mrope_embedding(self, model_config: ModelConfig[PretrainedConfig]):
        config = model_config.pretrained_config.text_config
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.from_string(config.rope_scaling["type"]),
            rope=RopeParams.from_config(config),
            mrope_section=config.rope_scaling.get("mrope_section", None),
        )
        self.rotary_cos_sin = pos_embd_params.rope.create_rope_const_params(interleave=False)[
            1
        ].reshape(pos_embd_params.rope.max_positions, 2, -1)
        self.mrope_section = pos_embd_params.mrope_section
        self.mrope_position_ids_padding_cuda = torch.zeros(
            (
                3,
                1,
                config.max_position_embeddings,
            ),
            dtype=torch.int32,
            device="cuda",
        )

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        pass

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    @nvtx_range("Qwen3-VL prepare_mrope_config")
    def prepare_mrope_config(
        self, multimodal_params: List[MultimodalParams], num_context_requests: int
    ):
        mrope_config = {}
        mrope_rotary_cos_sin = []
        mrope_position_deltas = []
        for multimodal_param in multimodal_params[:num_context_requests]:
            if multimodal_param.multimodal_data.get("mrope_config") is not None:
                with nvtx_range("Qwen3-VL get_cos_sin"):
                    if (
                        multimodal_param.multimodal_data["mrope_config"].get("mrope_position_ids")
                        is not None
                    ):
                        mrope_position_ids = multimodal_param.multimodal_data["mrope_config"][
                            "mrope_position_ids"
                        ]

                        self.mrope_position_ids_padding_cuda[
                            :, :, : mrope_position_ids.shape[-1]
                        ] = mrope_position_ids
                        self.mrope_position_ids_padding_cuda[
                            :, :, mrope_position_ids.shape[-1] :
                        ] = 0
                        cos_sin = self.rotary_cos_sin[
                            self.mrope_position_ids_padding_cuda.view(3, -1)
                        ]
                        cos, sin = cos_sin[:, :, 0, :], cos_sin[:, :, 1, :]
                        cos = apply_interleaved_rope(cos, self.mrope_section)
                        sin = apply_interleaved_rope(sin, self.mrope_section)
                        concat_cos_sin = torch.stack((cos, sin), dim=-1)
                        concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0], -1)
                        mrope_rotary_cos_sin.append(concat_cos_sin)

        for multimodal_param in multimodal_params[num_context_requests:]:
            if multimodal_param.multimodal_data.get("mrope_config") is not None:
                if (
                    multimodal_param.multimodal_data["mrope_config"].get("mrope_position_deltas")
                    is not None
                ):
                    mrope_position_deltas.append(
                        multimodal_param.multimodal_data["mrope_config"]["mrope_position_deltas"]
                    )

        with nvtx_range("Qwen3-VL concat mrope_rotary_cos_sin"):
            if mrope_rotary_cos_sin:
                mrope_config["mrope_rotary_cos_sin"] = torch.cat(mrope_rotary_cos_sin, dim=0)
        with nvtx_range("Qwen3-VL concat mrope_position_deltas"):
            if mrope_position_deltas:
                mrope_config["mrope_position_deltas"] = torch.cat(mrope_position_deltas, dim=0)

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
        num_context_requests, num_generation_requests = (
            attn_metadata.num_contexts,
            attn_metadata.num_generations,
        )
        logger.debug(
            f"num_context_requests: {num_context_requests}, num_generation_requests: {num_generation_requests}"
        )

        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds = []
        mrope_config = {}
        deepstack_features = []

        if len(multimodal_params) > 0:
            if not DISAGG:
                mm_embeds, deepstack_features = get_multimodal_embeddings_qwen3(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=multimodal_params[:num_context_requests],
                )
            else:
                raise NotImplementedError(
                    "Qwen3VLModel does not support disaggregated inference yet. Please unset "
                    "the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )

            mm_embeds = find_input_mm_embeds(
                mm_embeds,
                multimodal_params[:num_context_requests],
            )
            mrope_config = self.prepare_mrope_config(multimodal_params, num_context_requests)
            position_ids = mrope_config.get("mrope_position_ids", position_ids)

        input_ids, input_embeds, deepstack_features = fuse_input_embeds_qwen3(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            deepstack_features=deepstack_features,
            **kwargs,
        )

        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            deepstack_visual_embeds=deepstack_features,
            mrope_config=mrope_config,
        )

        logger.debug(f"output shape: {output_prob.shape}")
        return output_prob


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT], preserving frequency continuity.
    Copied from vllm
    """
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


def fuse_input_embeds_qwen3(
    embedding_layer: Embedding,
    input_ids: torch.IntTensor,
    mm_embeds: List[torch.Tensor],
    mm_token_ids: Optional[torch.IntTensor] = None,
    text_token_indices: Optional[torch.IntTensor] = None,
    mm_token_indices: Optional[torch.IntTensor] = None,
    deepstack_features: Optional[List[torch.Tensor]] = None,
    **kwargs,
) -> Tuple[Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
    if len(mm_embeds) == 0:
        return input_ids, None, None

    mm_embed = torch.cat(mm_embeds, dim=0)

    # TODO: support the case where only one index tensor is provided,
    # the other is derived as the complement (try to avoid implicit host-device synchronization)
    if text_token_indices is None or mm_token_indices is None:
        # NOTE: This function involves host-device synchronization due to torch.where() used in
        # filter_mm_token_from_input_ids.
        text_token_indices, mm_token_indices = filter_mm_token_from_input_ids(
            input_ids, vocab_size=embedding_layer.num_embeddings, mm_token_ids=mm_token_ids
        )

    if mm_token_indices.shape[0] != mm_embed.shape[0]:
        raise ValueError(
            f"Multimodal token count mismatch: found {len(mm_token_indices)} image tokens in input_ids "
            f"but received {mm_embed.shape[0]} image embeddings. "
            "This is likely due to KV cache reuse, chunk prefill, or other optimizations that "
            "cause token count mismatches within the inference batch."
        )

    text_embed = embedding_layer(input_ids[text_token_indices])
    input_embeds = torch.empty(
        input_ids.shape[0], mm_embed.shape[-1], device=text_embed.device, dtype=text_embed.dtype
    )
    if deepstack_features is not None and len(deepstack_features) > 0:
        # only support single modality for deepstack features for now
        deepstack_features = deepstack_features[0]
        deepstack_embeds = torch.zeros(
            deepstack_features.shape[0],
            input_ids.shape[0],
            mm_embed.shape[-1],
            device=deepstack_features.device,
            dtype=deepstack_features.dtype,
        )

        deepstack_embeds[:, mm_token_indices, :] = deepstack_features
    else:
        deepstack_embeds = None

    input_embeds[text_token_indices, :] = text_embed
    input_embeds[mm_token_indices, :] = mm_embed.to(
        dtype=input_embeds.dtype, device=input_embeds.device
    )

    return None, input_embeds, deepstack_embeds


def get_multimodal_embeddings_qwen3(
    encoder_forward_fn,
    multimodal_params: List[MultimodalParams],
    encoder_kwargs: Optional[Dict[str, Any]] = None,
) -> List[torch.Tensor]:
    if not multimodal_params:
        return [], []

    # Step 1: Find uncached multimodal params that need encoder processing
    uncached_multimodal_params = _get_uncached_multimodal_params(multimodal_params)

    # Step 2: Run encoder forward only on uncached parameters
    def valid_mm_runtime(param: MultimodalParams) -> bool:
        return (
            hasattr(param, "multimodal_runtime")
            and param.multimodal_runtime is not None
            and param.multimodal_runtime.total_mm_tokens_in_request is not None
        )

    deepstack_features = []
    if uncached_multimodal_params:
        kwargs = encoder_kwargs or {}
        encoder_outputs, deepstack_features = encoder_forward_fn(
            uncached_multimodal_params, **kwargs
        )

        # TODO: support multiple multimodal modalities per request
        if len(encoder_outputs) > 1:
            return encoder_outputs, deepstack_features

        # Validate that multimodal_runtime has required attributes for caching
        for param in uncached_multimodal_params:
            if not valid_mm_runtime(param):
                logger.warning(
                    "Multimodal runtime data missing or incomplete - recomputed all embeddings"
                )
                return encoder_outputs, deepstack_features

        # Step 3: Cache the computed embeddings to multimodal_data["multimodal_embedding"]
        _cache_multimodal_embeddings(uncached_multimodal_params, encoder_outputs)

    # Step 4: Gather all embeddings for the batch
    for param in multimodal_params:
        # concatenate if embeds is a list of tensors
        embeds = param.multimodal_data.get("multimodal_embedding")
        if isinstance(embeds, list):
            param.multimodal_data["multimodal_embedding"] = torch.cat(embeds, dim=0)

    all_embeddings = torch.cat(
        [param.multimodal_data["multimodal_embedding"] for param in multimodal_params], dim=0
    )
    return [all_embeddings], deepstack_features


@register_vision_encoder(Qwen3VisionModelBase, vlm_base_model=Qwen3VLVisionModel)
@register_auto_model("Qwen3VLForConditionalGeneration")
@register_input_processor(
    Qwen3VLInputProcessorBase,
    model_type="qwen3_vl",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ),
)
class Qwen3VLModelTRT(Qwen3VLModelBase):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        # NOTE: HF implementation.
        kwargs["vision_model_class"] = Qwen3VLVisionModel
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "image.image_grid_thw",
            "video.pixel_values_videos",
            "video.video_grid_thw",
            "multimodal_embedding",
        ]

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        if not DISAGG:
            vision_encoder_weights = process_weights(weights, "visual")
            self.mm_encoder.load_state_dict(vision_encoder_weights, strict=True)
        # print(weights.keys())
        transformed_weights = {}
        language_model_prefix = "model.language_model."
        for key, value in weights.items():
            if key.startswith(language_model_prefix):
                new_key = "model." + key[len(language_model_prefix) :]
                transformed_weights[new_key] = value
            else:
                transformed_weights[key] = value
        print("mapper:", weight_mapper)

        self.llm.load_weights(transformed_weights, weight_mapper)
