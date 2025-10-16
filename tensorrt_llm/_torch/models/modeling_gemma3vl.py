import copy
import dataclasses
import os
from typing import List, Optional, Tuple

import torch
from transformers import AutoProcessor, Gemma3Config, PreTrainedModel

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper

from ..._utils import nvtx_range
from ...inputs import (ExtraProcessedInputs, InputProcessor,
                       MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.linear import Linear
from ..modules.rms_norm import RMSNorm
from .modeling_gemma3 import Gemma3ForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_siglip import SiglipVisionModel
from .modeling_utils import ModelConfig, filter_weights, register_auto_model

_MULTIMODAL_ENV_NAME = "TLLM_MULTIMODAL_DISAGGREGATED"


# Make this a runtime lookup rather than a module-wide constant for easier unit testing.
def _is_disagg() -> bool:
    return os.getenv(_MULTIMODAL_ENV_NAME, "0") == "1"


class Gemma3InputProcessor(InputProcessor):

    def __init__(self, model_path, model_config, tokenizer, trust_remote_code):

        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.model_config = model_config
        self.device = 'cuda'

    @nvtx_range("[Vision] preprocess")
    def _preprocess(self, inputs):
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
        if mm_data and "image" not in mm_data:
            raise KeyError("Expected image data in multimodal data for Gemma3.")

        images = mm_data.get("image")
        do_rescale = self.processor.image_processor.do_rescale
        if images is not None and isinstance(images[0], torch.Tensor):
            do_rescale = False
        processor_output = self.processor(
            text=text_prompt,
            images=images,
            do_rescale=do_rescale,
            return_tensors="pt",
            device=self.device).to(dtype=torch.bfloat16)

        input_ids = processor_output["input_ids"]
        pixel_values = processor_output.get("pixel_values")

        return input_ids, pixel_values

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        input_ids, pixel_values = self._preprocess(inputs)
        multimodal_data = None
        if pixel_values is not None:
            multimodal_data = {
                "multimodal_data": {
                    "image": {
                        "pixel_values": pixel_values
                    }
                },
            }
        return input_ids[0].to(torch.int32).tolist(), multimodal_data


# Original HF implementation:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py#L684
class Gemma3MultiModalProjector(torch.nn.Module):
    """Gemma3MultiModalProjector using TRTLLM's Linear and RMSNorm."""

    def __init__(self, model_config: ModelConfig[Gemma3Config]):
        super().__init__()
        config = model_config.pretrained_config
        self.dtype = config.torch_dtype
        self.mm_input_projection = Linear(
            in_features=config.vision_config.hidden_size,
            out_features=config.text_config.hidden_size,
            bias=False,
            dtype=self.dtype,
            mapping=model_config.mapping)
        self.mm_soft_emb_norm = RMSNorm(
            hidden_size=config.vision_config.hidden_size,
            eps=config.vision_config.layer_norm_eps,
            dtype=self.dtype)
        self.patches_per_image = int(config.vision_config.image_size //
                                     config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=self.kernel_size,
                                           stride=self.kernel_size)

    def load_weights(self, weights):
        # Original `mm_input_projection_weight` is a matmul while we use a linear op with no bias.
        self.mm_input_projection.weight.data.copy_(
            weights["mm_input_projection_weight"].transpose(0, 1))
        # Gemma3RmsNorm is a layernorm-1P and needs a +1.0.
        self.mm_soft_emb_norm.weight.data.copy_(
            weights["mm_soft_emb_norm.weight"] + 1.0)

    @torch.inference_mode()
    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape
        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image,
            self.patches_per_image)
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs).to(
            self.dtype)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)  # [B, T, P*P].
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2).reshape(
            -1, seq_length)
        # FlashInfer's rmsnorm needs input to be contiguous.
        normed_vision_outputs = self.mm_soft_emb_norm(
            pooled_vision_outputs.contiguous())
        projected_vision_outputs = self.mm_input_projection(
            normed_vision_outputs)
        return projected_vision_outputs.type_as(vision_outputs)


@register_auto_model("Gemma3ForConditionalGeneration")
@register_input_processor(
    Gemma3InputProcessor,
    model_type="gemma3",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<start_of_image>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ))
class Gemma3VLM(PreTrainedModel):

    def __init__(self, model_config: ModelConfig[Gemma3Config]):
        if _is_disagg():
            raise NotImplementedError(
                "Gemma3VLM does not support disaggregated inference yet. Please unset "
                f"the {_MULTIMODAL_ENV_NAME} environment variable, or set it to '0'."
            )

        config = model_config.pretrained_config
        super().__init__(config)

        self._device = "cuda"
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)

        self.image_token_ids = torch.tensor([config.image_token_index],
                                            dtype=torch.int32,
                                            device=self._device)

        model_config_cp = copy.deepcopy(model_config)
        self.model_config = model_config_cp

        llm_model_config = self.get_sub_model_config(model_config_cp,
                                                     "text_config")
        self.llm = Gemma3ForCausalLM(llm_model_config)

        vision_model_config = self.get_sub_model_config(model_config_cp,
                                                        "vision_config")
        self.siglip_tower = SiglipVisionModel(vision_model_config,
                                              use_post_layernorm=True)

        self.mm_projector = Gemma3MultiModalProjector(model_config).eval().to(
            self._device)

        self.post_config()
        self.is_loaded = True

    @staticmethod
    def get_sub_model_config(
        model_config: ModelConfig[Gemma3Config],
        name: str,
    ) -> ModelConfig:
        # Extract the subconfig from the `transformers` config and create a copy of the
        # `ModelConfig` class with the subconfig and preferred backend updated.
        assert name in [
            "text_config", "vision_config"
        ], f"Expected subconfig name to be either 'text_config' or 'vision_config'. Got {name} instead."
        pretrained_config = getattr(model_config.pretrained_config, name)
        # ModelOpt currently doesn't quantize the vision part. Without setting quant config to None,
        # weight loading fails for vision.
        quant_config = model_config.quant_config if name == "text_config" else None
        # FlashInfer backend supports custom mask which is needed for bidirectional mask in decoder.
        preferred_backend = "FLASHINFER" if name == "text_config" else "TRTLLM"
        sub_model_config: ModelConfig[Gemma3Config] = dataclasses.replace(
            model_config,
            pretrained_config=pretrained_config,
            attn_backend=preferred_backend,
            quant_config=quant_config)
        # Make sure some fields that are not explicitly included in the sub config, but present
        # in the top-level config, are replicated.
        if (hasattr(sub_model_config.pretrained_config, "torch_dtype")
                and sub_model_config.pretrained_config.torch_dtype is None):
            sub_model_config.pretrained_config.torch_dtype = model_config.pretrained_config.torch_dtype
        return sub_model_config

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        llm_weights = filter_weights("language_model", weights)
        self.llm.load_weights(llm_weights, weight_mapper)

        vit_weights = filter_weights("vision_tower", weights)
        self.siglip_tower.load_weights(vit_weights)

        mm_projector_weights = filter_weights("multi_modal_projector", weights)
        self.mm_projector.load_weights(mm_projector_weights)

    def post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(
            f"[Gemma3Model::forward]{num_context_requests=}, {num_generation_requests=}"
        )

        multimodal_params = kwargs.get("multimodal_params", [])
        pixel_values = [
            multimodal_param.multimodal_data["image"]["pixel_values"]
            for multimodal_param in multimodal_params
        ]

        mm_embeds = []
        mm_token_mask = None
        if len(pixel_values) > 0:
            image_features = self._get_image_features(
                pixel_values=torch.cat(pixel_values))
            mm_embeds = [image_features.contiguous()]

            # Get token type ids. 0 corresponds to text tokens, 1 corresponds to image tokens.
            mm_token_mask = torch.isin(input_ids, self.image_token_ids)

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embeds,
            mm_token_ids=self.image_token_ids,
            **kwargs,
        )
        logits = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            image_token_mask=mm_token_mask,
            lora_params=kwargs.get("lora_params", None),
        )
        return logits

    @nvtx_range("[Vision] process")
    def _get_image_features(self, pixel_values):
        attn_metadata = self.siglip_tower.prepare_attn_metadata(
            pixel_values.shape[0])
        with torch.autocast(device_type="cuda", dtype=self.model_dtype):
            image_features = self.siglip_tower(pixel_values,
                                               attn_metadata=attn_metadata)[-1]
            image_features = self.mm_projector(image_features)
        return image_features

    @property
    def mm_token_ids(self):
        return self.image_token_ids
