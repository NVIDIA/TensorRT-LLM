import copy
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN as HF_ACT2FN
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionPatchEmbed as HFQwen3VLVisionPatchEmbed,
)

from tensorrt_llm._torch.models.modeling_multimodal_utils import _is_disagg
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping

from ..._utils import nvtx_range, nvtx_range_debug, prefer_pinned
from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ContentFormat,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
    support_multimodal_disaggregated,
)
from ...inputs.multimodal import MultimodalParams
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..attention_backend.utils import get_attention_backend
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mlp import MLP
from ..modules.rotary_embedding import MRotaryEmbedding, RotaryEmbedding
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.qwen3vl_weight_mapper import Qwen3VLHfWeightMapper
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from .modeling_qwen2vl import Qwen2_5_VLVisionAttention, install_qwen_vl_processor_defaults_fix
from .modeling_utils import (
    ModelConfig,
    QuantConfig,
    _load_weights_impl,
    filter_weights,
    register_auto_model,
    register_vision_encoder,
)


class Qwen3VLInputProcessorBase(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: AutoTokenizer,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._dtype = self.config.text_config.dtype
        self._tokenizer = (
            tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_path)
        )
        self._model_path = model_path
        self._processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True, trust_remote_code=trust_remote_code
        )
        install_qwen_vl_processor_defaults_fix(self._processor)
        self.tllm_multimodal_token_id = self.get_vocab_size() + 1
        # temporal patch size for video frames
        self.temporal_patch_size = getattr(self.config.vision_config, "temporal_patch_size", 1)

    @property
    def config(self) -> PretrainedConfig:
        return self._config

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
        return self.config.text_config.vocab_size

    @classmethod
    def get_rope_index(
        cls,
        model_config: PretrainedConfig,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2>
        # <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

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

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode
                    # the temporal information for videos)
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

    def get_num_tokens_per_video(
        self,
        *,
        video: List[Image.Image],
        video_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> int:
        merge = self.config.vision_config.spatial_merge_size
        if video_grid_thw is not None:
            t, h, w = (int(x) for x in video_grid_thw)
            return t * (h // merge) * (w // merge)

        # Must run the full processor: HF's Qwen3VLProcessor._get_num_multimodal_tokens
        # (what the base class default delegates to) raises on video-only calls
        # and returns a wrong-formula fallback that would break chunked prefill.
        do_rescale = not (video and isinstance(video[0], torch.Tensor))
        processed = self._processor(
            text=["<|vision_start|><|video_pad|><|vision_end|>"],
            videos=[video],
            padding=True,
            do_rescale=do_rescale,
            return_tensors="pt",
            **kwargs,
        )
        vgt = processed.get("video_grid_thw")
        if vgt is None or len(vgt) == 0:
            raise RuntimeError(
                "get_num_tokens_per_video: HF processor returned no "
                "video_grid_thw for the provided video."
            )
        t, h, w = (int(x) for x in vgt[0].tolist())
        return t * (h // merge) * (w // merge)

    def _preprocess(
        self, text: Dict[str, Any], mm_data: Dict[str, Any], mm_processor_kwargs: Dict[str, Any]
    ):
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
        # The TRT-LLM safe-processor subclass installed in __init__ prevents
        # the upstream class-level _defaults mutation that would otherwise
        # cause processor *output* keys to leak into per-modality kwargs and
        # trip ProcessorMixin._merge_kwargs validation. So we can call the
        # processor directly here.
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
        masks = (input_ids == self.config.image_token_id) | (
            input_ids == self.config.video_token_id
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
            self.config, input_ids, image_grid_thw, video_grid_thw, attention_mask
        )

        mrope_config = {}
        mrope_config["mrope_position_ids"] = mrope_position_ids.to("cpu").clone()
        mrope_config["mrope_position_deltas"] = (
            mrope_position_deltas.to("cpu").to(torch.int32).clone()
        )

        return mrope_config

    @nvtx_range("Qwen3VLInputProcessorBase forward()")
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
                "pixel_values": pixel_values.to(self.dtype),
                "image_grid_thw": processed_inputs.get("image_grid_thw"),
            }

        pixel_values_videos = processed_inputs.get("pixel_values_videos", None)
        if pixel_values_videos is not None:
            multimodal_data["video"] = {
                "pixel_values_videos": pixel_values_videos.to(self.dtype),
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

    def get_prompt_token_ids(
        self, inputs: TextPrompt, mm_handles: List[Dict[str, Any]]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Build input token ids with multimodal placeholders expanded to the number of MM tokens.

        Args:
            inputs: Text prompt input container. Must contain a non-empty prompt string.
            mm_handles: List of multimodal embedding handles.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                - expanded_ids: token ids with each image token expanded to a placeholder repeated per MM token
                - mm_token_length: per-image MM token lengths
                - mm_token_offsets: start offsets (positions) for each image's MM tokens within expanded_ids
        """
        # TODO: Move this function to the base input processor class when extending for more models
        text_prompt = inputs.get("prompt")
        if not text_prompt:
            raise ValueError("Text prompt is required but not provided")

        if not isinstance(mm_handles, list):
            raise TypeError("mm_handles must be a list")

        num_deepstack_levels = len(self.config.vision_config.deepstack_visual_indexes)
        # This is because, unlike previous Qwen VL models, the embeddings are concatenated with
        # feature maps from deepstack layers.
        expected_size = self.config.text_config.hidden_size * (1 + num_deepstack_levels)
        for i, mm_handle in enumerate(mm_handles):
            hidden_size = mm_handle["tensor_size"][1]
            if hidden_size != expected_size:
                raise RuntimeError(
                    f"Expected multimodal embedding {i} to have hidden size {expected_size}, got {hidden_size}."
                )

        input_ids = self.tokenizer(text_prompt, return_tensors="pt").input_ids[0]

        # TODO: what about `video_token_id`?
        image_token_index = self.config.image_token_id

        image_mask = input_ids == image_token_index
        image_positions = torch.where(image_mask)[0]
        num_images = len(image_positions)
        assert num_images == len(mm_handles), "Number of images must match number of mm_handles"
        total_mm_tokens = sum(mm_handle["tensor_size"][0] for mm_handle in mm_handles)
        final_length = len(input_ids) - num_images + total_mm_tokens
        # Create output tensor
        expanded_ids = torch.empty(final_length, dtype=input_ids.dtype)
        placeholder_id = self.tllm_multimodal_token_id

        # Fill the expanded sequence
        write_pos = 0
        image_cnt = 0
        mm_token_length = []
        mm_token_offsets = []
        for read_pos in range(len(input_ids)):
            if input_ids[read_pos] == image_token_index:
                # Replace with placeholder id
                mm_token_num = mm_handles[image_cnt]["tensor_size"][0]
                expanded_ids[write_pos : write_pos + mm_token_num] = placeholder_id
                mm_token_offsets.append(write_pos)
                mm_token_length.append(mm_token_num)
                write_pos += mm_token_num
                image_cnt += 1
            else:
                # Copy text token as-is
                expanded_ids[write_pos] = input_ids[read_pos]
                write_pos += 1

        assert write_pos == final_length, f"Write position mismatch: {write_pos} != {final_length}"
        assert mm_token_length[-1] + mm_token_offsets[-1] <= final_length, (
            f"mm_token_length[-1] + mm_token_offsets[-1] ({mm_token_length[-1] + mm_token_offsets[-1]}) should be less "
            f"than or equal to final_length ({final_length})"
        )
        return expanded_ids.to(torch.int32).tolist(), mm_token_length, mm_token_offsets


class Qwen3VLVisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, model_config, layer_idx):
        # Qwen3-VL keeps `torch_dtype` only on `text_config` under transformers 5.x
        # strict mode; mirror it onto `vision_config` so the parent picks it up.
        # `max_position_embeddings` is handled by the parent's text_config fallback.
        model_config.pretrained_config.vision_config.torch_dtype = (
            model_config.pretrained_config.text_config.dtype
        )
        super().__init__(
            model_config,
            layer_idx=layer_idx,
            reduce_output=(
                not model_config.mapping.enable_attention_dp and model_config.mapping.tp_size > 1
            ),
        )


class Qwen3VLVisionMLP(MLP):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int):
        config = model_config.pretrained_config.vision_config
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=True,
            activation=HF_ACT2FN[config.hidden_act],
            dtype=model_config.pretrained_config.text_config.dtype,
            config=model_config,
            layer_idx=layer_idx,
            overridden_tp_size=1 if model_config.mapping.enable_attention_dp else None,
        )


class Qwen3VLVisionBlock(torch.nn.Module):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config.vision_config

        self.norm1 = LayerNorm(
            hidden_size=config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.text_config.dtype,
        )
        self.norm2 = LayerNorm(
            hidden_size=config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.text_config.dtype,
        )
        self.attn = Qwen3VLVisionAttention(model_config, layer_idx)
        self.mlp = Qwen3VLVisionMLP(model_config, layer_idx)

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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


class Qwen3VLVisionPatchMerger(torch.nn.Module):
    def __init__(
        self, model_config: ModelConfig[PretrainedConfig], use_postshuffle_norm: bool = False
    ) -> None:
        super().__init__()
        config = model_config.pretrained_config.vision_config
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = LayerNorm(
            hidden_size=self.hidden_size if use_postshuffle_norm else config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.text_config.dtype,
        )

        self.mapping = model_config.mapping
        overridden_tp_size = 1 if model_config.mapping.enable_attention_dp else None
        if overridden_tp_size is not None:
            assert self.mapping.tp_size % overridden_tp_size == 0
            tp_size = overridden_tp_size
            # "Misuse" pp_size here to perform all-reduce within smaller groups
            pp_size = self.mapping.pp_size * self.mapping.tp_size // overridden_tp_size
            mapping = Mapping(
                world_size=tp_size * pp_size,
                rank=self.mapping.rank,
                gpus_per_node=self.mapping.gpus_per_node,
                tp_size=tp_size,
                pp_size=pp_size,
            )
        else:
            mapping = self.mapping

        self.linear_fc1 = Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=True,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = Linear(
            in_features=self.hidden_size,
            out_features=config.out_hidden_size,
            bias=True,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            allreduce_strategy=model_config.allreduce_strategy,
        )

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.hidden_size)

        hidden_states = self.norm(hidden_states).view(-1, self.hidden_size)
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_fc2(hidden_states)
        return hidden_states


# Referenced from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl.py#L668
def pos_embed_interpolate_native(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Eager PyTorch bilinear position-embedding interpolation.

    Returns a tensor of shape ``(t * h * w, hidden_dim)`` with the
    bilinearly-interpolated position embeddings in spatial-merge order.
    """
    assert h % m_size == 0 and w % m_size == 0, f"{h=} and {w=} must be divisible by {m_size=}"
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device

    h_idxs = torch.linspace(
        0,
        num_grid_per_side - 1,
        h,
        dtype=torch.float32,
        device=device,
    )
    w_idxs = torch.linspace(
        0,
        num_grid_per_side - 1,
        w,
        dtype=torch.float32,
        device=device,
    )

    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01

    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    h_grid_idx = h_grid * num_grid_per_side

    indices = (h_grid_idx + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
    weights = weights.to(dtype=dtype)

    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)

    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
    return repeated.to(dtype=dtype)


class Qwen3VisionModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        self.model_config = model_config
        self.config = self.model_config.pretrained_config.vision_config

        self.spatial_merge_size = self.config.spatial_merge_size
        self.patch_size = self.config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = HFQwen3VLVisionPatchEmbed(
            config=self.config,
        )

        self.pos_embed = nn.Embedding(self.config.num_position_embeddings, self.config.hidden_size)
        self.num_grid_per_side = int(self.config.num_position_embeddings**0.5)

        text_config = getattr(
            model_config.pretrained_config, "text_config", model_config.pretrained_config
        )
        self.config.max_position_embeddings = text_config.max_position_embeddings
        self.config.partial_rotary_factor = 0.5
        self.config.num_attention_heads = self.config.num_heads
        self.head_dim = self.config.hidden_size // self.config.num_heads
        self.pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams.from_config(self.config),
        )
        self.rotary_pos_emb = RotaryEmbedding(
            self.pos_embd_params.rope,
            head_dim=self.head_dim,
            is_neox=self.pos_embd_params.is_neox,
        )

        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(model_config, layer_idx=layer_idx)
                for layer_idx in range(self.config.depth)
            ]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            model_config=model_config,
            use_postshuffle_norm=False,
        )
        self.deepstack_visual_indexes = self.config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    model_config=model_config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )
        self.metadata_cls = get_attention_backend(self.model_config.attn_backend).Metadata

        self.attn_metadata = self.metadata_cls(
            max_num_requests=8192,  # TODO: Make this dynamic
            max_num_tokens=8192,  # TODO: Make this dynamic
            kv_cache_manager=None,
        )

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    def rot_pos_emb(self, grid_thw: list[list[int]]):
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        pos_ids = [
            self.rot_pos_ids(h, w, self.spatial_merge_size)
            if t == 1
            else self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
            for t, h, w in grid_thw
        ]
        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos_sin = self.rotary_pos_emb.rotary_cos_sin[:max_grid_size]
        cos, sin = cos_sin[:, 0, :], cos_sin[:, 1, :]
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)

        return (cos_combined, sin_combined)

    # Referenced from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl.py#L668
    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
        interpolate_fn = pos_embed_interpolate_native
        outputs = []
        for t, h, w in grid_thw:
            outputs.append(
                interpolate_fn(
                    self.pos_embed.weight,
                    t,
                    h,
                    w,
                    self.num_grid_per_side,
                    self.spatial_merge_size,
                    self.dtype,
                )
            )
        return torch.cat(outputs, dim=0)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def prepare_attn_metadata(
        self, batch_size: int, seq_lens: List[int], attn_metadata: AttentionMetadata
    ):
        batch_size = len(seq_lens)
        seq_lens_torch = torch.tensor(seq_lens, dtype=torch.int, pin_memory=prefer_pinned())
        request_ids = list(range(1, batch_size + 1))

        attn_metadata.num_contexts = len(seq_lens)
        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = seq_lens
        attn_metadata.seq_lens = seq_lens_torch
        attn_metadata.max_seq_len = max(seq_lens)
        attn_metadata.prepare()
        return attn_metadata

    @torch.inference_mode()
    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        seq_lens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()
        grid_rows = grid_thw.detach().cpu().tolist()
        attn_metadata = self.prepare_attn_metadata(len(grid_thw), seq_lens, self.attn_metadata)

        # Getting positional embedding (use the CPU-materialized grid to avoid
        # converting CUDA tensors to numpy inside `rot_pos_ids`).
        rotary_pos_emb = self.rot_pos_emb(grid_rows)

        pos_embeds = self.fast_pos_embed_interpolate(grid_rows)
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + pos_embeds
        seq_len, _ = hidden_states.size()
        rope_position_ids = torch.arange(seq_len, dtype=torch.int32, pin_memory=prefer_pinned())
        rope_position_ids = rope_position_ids.to(
            device=self.device, dtype=torch.int32, non_blocking=True
        )
        hidden_states = hidden_states.reshape(seq_len, -1)

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                position_ids=rope_position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                position_embeddings=rotary_pos_emb,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


class Qwen3VisionModelBase(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        model_class: Union[type[PreTrainedModel], type[torch.nn.Module]],
    ):
        super().__init__()
        self.model_config = model_config
        self.model_dtype = self.model_config.pretrained_config.text_config.dtype

        # NOTE: Re-setting QuantConfig to exclude vision encoder from quantization,
        # including KV cache quantization (vision encoder head dims may not be
        # supported by FP8 FMHA kernels).
        self.model_config.quant_config = QuantConfig()

        self.visual = model_class(self.model_config).to(self.model_dtype)

        self.post_config()

    def post_config(self):
        self.config = self.model_config.pretrained_config.vision_config

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        visual_weights = filter_weights("model.visual", weights)
        converted_weights = {}

        qkv_pattern = re.compile(r"(.*?)attn\.qkv\.(.*)")
        for name in visual_weights:
            # Handle with weights and bias for vision transformer's qkv projection.
            match = qkv_pattern.match(name)
            if match:
                prefix, suffix = match.groups()
                q_name = f"{prefix}attn.q_proj.{suffix}"
                k_name = f"{prefix}attn.k_proj.{suffix}"
                v_name = f"{prefix}attn.v_proj.{suffix}"
                dim_shape = visual_weights[name].shape[0] // 3
                converted_weights[q_name] = visual_weights[name][:dim_shape]
                converted_weights[k_name] = visual_weights[name][dim_shape : 2 * dim_shape]
                converted_weights[v_name] = visual_weights[name][2 * dim_shape :]
            else:
                converted_weights[name] = visual_weights[name]
        pattern_mapping = {
            r"(.*?)attn.proj.(.*)": r"\1attn.o_proj.\2",
            r"(.*?)mlp.linear_fc1.(.*)": r"\1mlp.up_proj.\2",
            r"(.*?)mlp.linear_fc2.(.*)": r"\1mlp.down_proj.\2",
        }
        self.visual.config.num_attention_heads = self.visual.config.num_heads
        _load_weights_impl(self.visual, converted_weights, params_map=pattern_mapping)

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
    def forward(self, multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        mm_content_data, mm_extra_data = self._parse_and_batch_multimodal_data(multimodal_params)
        pixel_values = mm_content_data.get("pixel_values", None)
        pixel_values_videos = mm_content_data.get("pixel_values_videos", None)

        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("Currently only support single modality per request")

        image_grid_thw = mm_extra_data.get("image_grid_thw", None)
        video_grid_thw = mm_extra_data.get("video_grid_thw", None)

        embeds = []
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model_dtype)
            image_embeds, deepstack_image_embeds = self.visual(
                pixel_values, grid_thw=image_grid_thw
            )
            # NOTE: We concatenate deepstack_embeds to mm_embeds
            # The shape will be [seq_len, hidden_dim * (num_deepstack_layers + 1)]
            mixed_image_embeds = torch.cat([image_embeds] + deepstack_image_embeds, dim=1)
            embeds.append(mixed_image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(self.model_dtype)
            video_embeds, deepstack_video_embeds = self.visual(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            # NOTE: We concatenate deepstack_embeds to mm_embeds
            # The shape will be [seq_len, hidden_dim * (num_deepstack_layers + 1)]
            mixed_video_embeds = torch.cat([video_embeds] + deepstack_video_embeds, dim=1)
            embeds.append(mixed_video_embeds)
        return embeds


class Qwen3VLModelBase(PreTrainedModel):
    def _check_and_adjust_experts_implementation(self, *args, **kwargs):
        """No-op override.

        Transformers 5.x's ``PreTrainedModel.__init__`` calls this method
        (with an ``experts_implementation`` argument) which fails for VL
        wrapper models that do not directly contain MoE layers.  TRT-LLM
        manages expert implementations independently, so skip the check.
        """
        return None

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        self.original_arch = model_config.pretrained_config.architectures[0]

        disable_fuse_rope = kwargs.get("disable_fuse_rope", False)
        model_config.pretrained_config.disable_fuse_rope = disable_fuse_rope
        model_config.pretrained_config.text_config.disable_fuse_rope = disable_fuse_rope
        # transformers>=5.5 flipped the ``Qwen3VL[Moe]TextConfig`` class-level
        # default for ``tie_word_embeddings`` from inherited ``False`` to
        # ``True``. Real checkpoints set it only on the top-level config, so
        # the nested ``text_config`` silently picks up the new default and
        # ties lm_head to embed_tokens, producing systematic logits drift.
        # Mirror the top-level value onto ``text_config`` to stay aligned.
        model_config.pretrained_config.text_config.tie_word_embeddings = (
            model_config.pretrained_config.tie_word_embeddings
        )
        # In transformers 5.x, rope_scaling may delegate to rope_parameters which
        # can be None.  Ensure the dict exists before setting the type key.
        if model_config.pretrained_config.text_config.rope_scaling is None:
            model_config.pretrained_config.text_config.rope_scaling = {}
        model_config.pretrained_config.text_config.rope_scaling["type"] = "mrope"
        config = model_config.pretrained_config

        self._supports_sdpa = True
        self._supports_flash_attn = True
        super().__init__(config)
        if not disable_fuse_rope:
            self.init_mrope_embedding(model_config)

        self.model_config = model_config

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = config.text_config
        if self.original_arch == "Qwen3VLForConditionalGeneration":
            llm_model_config.pretrained_config.architectures = ["Qwen3ForCausalLM"]
        elif self.original_arch == "Qwen3VLMoeForConditionalGeneration":
            llm_model_config.pretrained_config.architectures = ["Qwen3MoeForCausalLM"]
        else:
            raise ValueError(f"Unsupported architecture: {self.original_arch}")
        # Qwen3ForCausalLM.
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        if not _is_disagg():
            self.mm_encoder = Qwen3VisionModelBase(
                model_config, kwargs.get("vision_model_class", None)
            ).eval()

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        )

        self.post_config()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.model_config.pretrained_config = self.llm.config
        self.config = self.model_config.pretrained_config

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def init_mrope_embedding(self, model_config: ModelConfig[PretrainedConfig]):
        config = model_config.pretrained_config.text_config
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.from_string(config.rope_scaling["type"]),
            rope=RopeParams.from_config(config),
            mrope_section=config.rope_scaling.get("mrope_section", None),
            mrope_interleaved=config.rope_scaling.get("mrope_interleaved", False),
        )
        self.rotary_emb = MRotaryEmbedding(
            pos_embd_params.rope,
            head_dim=config.hidden_size // config.num_attention_heads,
            is_neox=pos_embd_params.is_neox,
            mrope_section=pos_embd_params.mrope_section,
            mrope_interleaved=pos_embd_params.mrope_interleaved,
        ).to("cuda")
        self.mrope_position_ids_padding_cuda = torch.zeros(
            (
                3,
                1,
                config.max_position_embeddings,
            ),
            dtype=torch.int32,
            device="cuda",
        )

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
                        cos, sin = self.rotary_emb.get_cos_sin(self.mrope_position_ids_padding_cuda)
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

    def split_mm_embeds(self, mm_embed, deepstack_num_level):
        num_elements = mm_embed.shape[1] // (deepstack_num_level + 1)
        mm_embed_chunks = torch.split(mm_embed, [num_elements] * (deepstack_num_level + 1), dim=1)
        return mm_embed_chunks[0], list(mm_embed_chunks[1:])

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
        deepstack_embeds = []

        # NOTE: Qwen*-VL series has mrope_config even on the text-only prompts,
        # so we need to separate the mm_multimodal_params from the text-only prompts.
        mm_multimodal_params = self._get_requests_with_mm_data(multimodal_params)
        if len(mm_multimodal_params) > 0:
            if not _is_disagg():
                mm_embeds = get_multimodal_embeddings(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=mm_multimodal_params,
                )
            elif not getattr(self, "support_mm_disagg", False):
                raise NotImplementedError(
                    f"{type(self)} does not support disaggregated inference yet. Please unset "
                    "the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_multimodal_params)

            if self.use_deepstack:
                for i, mm_embed in enumerate(mm_embeds):
                    mm_embed, deepstack_embed = self.split_mm_embeds(
                        mm_embed, self.deepstack_num_level
                    )
                    mm_embeds[i] = mm_embed
                    deepstack_embeds.extend(deepstack_embed)

        if not self.model_config.pretrained_config.disable_fuse_rope:
            mrope_config = self.prepare_mrope_config(multimodal_params, num_context_requests)

        result = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            extra_embeds=deepstack_embeds,
            **kwargs,
        )
        if len(deepstack_embeds) > 0:
            input_ids, input_embeds, deepstack_embeds = result
        else:
            input_ids, input_embeds = result

        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            deepstack_embeds=deepstack_embeds,
            mrope_config=mrope_config,
        )
        logger.debug(f"output shape: {output_prob.shape}")
        return output_prob

    def _get_requests_with_mm_data(self, multimodal_params):
        mm_multimodal_params = []
        for multimodal_param in multimodal_params:
            data = multimodal_param.multimodal_data
            if (
                # The first 2 conditions check whether there is input on which inference should be run.
                data.get("image", {}).get("pixel_values") is not None
                or data.get("video", {}).get("pixel_values_videos") is not None
                # This condition corresponds to when the embeddings are already populated, as is e.g.
                # the case in EPD disagg in the prefill worker.
                or data.get("multimodal_embedding") is not None
            ):
                mm_multimodal_params.append(multimodal_param)

        return mm_multimodal_params


@support_multimodal_disaggregated
@register_vision_encoder(Qwen3VisionModelBase, vlm_base_model=Qwen3VisionModel)
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
        placeholders_separator="",
        content_format=ContentFormat.STRING,
    ),
)
class Qwen3VLModel(Qwen3VLModelBase):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        # NOTE: HF implementation.
        kwargs["vision_model_class"] = Qwen3VisionModel
        kwargs["disable_fuse_rope"] = kwargs.get(
            "disable_fuse_rope", False
        )  # TODO: Make this ModelConfig's argument
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return ["image.pixel_values", "video.pixel_values_videos", "multimodal_embedding"]

    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
        if not _is_disagg():
            self.mm_encoder.load_weights(weights)

        weight_mapper = Qwen3VLHfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)
        filtered_weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
        params_map = {
            r"^model\.language_model\.(.*)$": r"model.\1",
        }
        self.llm.load_weights(filtered_weights, weight_mapper, params_map=params_map)
