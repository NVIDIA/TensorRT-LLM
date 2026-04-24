import os
import dataclasses
import copy
from typing import Optional, List, Tuple, Literal

import torch
from torch import nn
from ..._utils import nvtx_range
from transformers import (AutoProcessor, AutoTokenizer, Cohere2VisionConfig, PreTrainedModel, PretrainedConfig)
from transformers.activations import ACT2FN

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
    TextPrompt,
    ExtraProcessedInputs,
)
from ...logger import logger
from ..attention_backend import AttentionMetadata
from ...sampling_params import SamplingParams

from .modeling_auto import AutoModelForCausalLM
from .modeling_utils import ModelConfig, filter_weights, register_auto_model
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_cohere2 import Cohere2ForCausalLM
from .modeling_siglip import SiglipVisionModel
from ..modules.linear import Linear

class Cohere2InputProcessor(BaseMultimodalInputProcessor,
                             BaseMultimodalDummyInputsBuilder):
    
    def __init__(self,
                 model_path: str,
                 config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True,
                 **kwargs):
        super().__init__(model_path=model_path,
                         config=config,
                         tokenizer=tokenizer,
                         trust_remote_code=trust_remote_code,
                         **kwargs)
        self._config = config
        self._tokenizer = tokenizer
        self._model_path = model_path
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast
        )
        self._dtype = self.config.torch_dtype
    
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
    
    @nvtx_range("[Vision] preprocess")
    def _preprocess(self, inputs):
        text_prompt, mm_data = inputs.get("prompt"), inputs.get("multi_modal_data", {})
        if mm_data and "image" not in mm_data:
            raise KeyError("Expected image data in multimodal data for Cohere2Vision.")

        images = mm_data.get("image")
        do_rescale = self._processor.image_processor.do_rescale
        if images is not None and isinstance(images[0], torch.Tensor):
            do_rescale = False
        processor_output = self._processor(
            text=text_prompt,
            images=images,
            do_rescale=do_rescale,
            return_tensors="pt",
        ).to(dtype=self.dtype)

        input_ids = processor_output["input_ids"]
        pixel_values = processor_output.get("pixel_values")

        return input_ids, pixel_values
    
    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        inputs_ids, pixel_values = self._preprocess(inputs)
        multimodal_data = None
        if pixel_values is not None:
            multimodal_data = {
                "multimodal_data": {
                    "image": {
                        "pixel_values": pixel_values
                    }
                },
            }
        return inputs_ids[0].to(torch.int32).tolist(), multimodal_data

# Original HF implementation:
# https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/cohere2_vision/modeling_cohere2_vision.py
class Cohere2VisionMultiModalProjector(nn.Module):
    """Cohere2MultiModalProjector using TRTLLM's Linear and RMSNorm."""

    def __init__(self, model_config: ModelConfig[Cohere2VisionConfig]):
        assert model_config.pretrained_config is not None
        super().__init__()
        self.config = model_config
        config: Cohere2VisionConfig = model_config.pretrained_config  # Extract Hugging Face's config

        self.downsample_factor = config.downsample_factor
        self.intermediate_size = config.alignment_intermediate_size

        # A SwiGLU implementation with the fused gate and up projections
        self.gate_and_up_proj = Linear(
            config.vision_config.hidden_size * (config.downsample_factor**2),
            self.intermediate_size,
            bias=True,
        )
        self.act = nn.SiLU()  # TODO: check the counterpart in TRT LLM
        self.down_proj = Linear(
            self.intermediate_size // 2,
            config.text_config.hidden_size,
            bias=True,
        )

    def pixel_shuffle(self, image_features: torch.Tensor): # B, S, D
        # Concatenate a number of horizontal and vertical patches to the channel dimension
        batch_size, seq_length, feature_dim = image_features.shape
        height = width = int(seq_length ** 0.5)
        image_features = image_features.reshape(image_features.shape[0], width, height, -1)
        channels = image_features.shape[-1]
        image_features = image_features.reshape(
            batch_size,
            width,
            int(height / self.downsample_factor),
            int(channels * self.downsample_factor),
        )
        # Exchange height and width dimensions so that reshape can be applied
        image_features = image_features.permute(0, 2, 1, 3)
        image_features = image_features.reshape(
            batch_size,
            int(height / self.downsample_factor),
            int(width / self.downsample_factor),
            int(channels * self.downsample_factor * self.downsample_factor),
        )
        # Redo exchanging the height and width dimensions
        image_features = image_features.permute(0, 2, 1, 3)
        return image_features


    def load_weights(self, weights: dict[str, torch.Tensor]):
        self.gate_and_up_proj.weight.data.copy_(weights["linear_1.weight"])
        self.gate_and_up_proj.bias.data.copy_(weights["linear_1.bias"])
        self.down_proj.weight.data.copy_(weights["linear_2.weight"])
        self.down_proj.bias.data.copy_(weights["linear_2.bias"])
    
    def forward(self, image_features):
        # TODO: implement pixel_shuffle
        image_features = self.pixel_shuffle(image_features)
        hidden_states = self.gate_and_up_proj(image_features)

        # Split along last dimension and apply SwiGLU
        x, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * x

        hidden_states = self.down_proj(hidden_states)
        return hidden_states


@register_auto_model("Cohere2VisionForConditionalGeneration")
@register_input_processor(
    Cohere2InputProcessor,
    model_type="cohere2_vision",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<|IMG_PATCH|>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        placeholders_separator="",
        content_format=ContentFormat.STRING,
    ),
)
class Cohere2VisionModel(PreTrainedModel):
    _MULTIMODAL_ENV_NAME = "TLLM_MULTIMODAL_DISAGGREGATED"

    @staticmethod
    def _is_disagg() -> bool:
        return os.getenv(Cohere2VisionModel._MULTIMODAL_ENV_NAME, "0") == "1"

    def __init__(self, model_config: ModelConfig[Cohere2VisionConfig]):
        if self._is_disagg():
            raise NotImplementedError(
                "Cohere2Vision does not support disaggregated inference yet."
            )
        
        config = model_config.pretrained_config
        super().__init__(config)
        
        # Models must have a self.model_config attribute
        model_config_cp = copy.deepcopy(model_config)
        self.model_config = model_config_cp

        # Set extra internal configurations 
        self._device = "cuda"
        # TODO: Convert the image token index (single integer) to torch.Tensor
        self.image_token_ids = torch.tensor([config.image_token_id],
                                            dtype=torch.int32,
                                            device=self._device)


        # Configure self.llm, self.siglip_tower, self.vision_projector
        llm_model_config = self._get_submodel_config(model_config_cp, "text_config")
        self.llm = Cohere2ForCausalLM(llm_model_config)

        vision_model_config = self._get_submodel_config(model_config_cp, "vision_config")
        
        # Use post layernorm to prevent overflow in the outputs of SigLIP
        self.siglip_tower = SiglipVisionModel(vision_model_config, use_post_layernorm=True)
        
        self.multimodal_projector = Cohere2VisionMultiModalProjector(
            model_config,
        ).to(self._device).eval()

        self._device = "cuda"
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)

        # NOTE: From now on, the configuration behaves as that of the text encoder
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config


    @staticmethod
    def _get_submodel_config(
        model_config: ModelConfig[Cohere2VisionConfig],
        name: Literal["text_config", "vision_config"],
    ) -> ModelConfig:
        # Extract the subconfig from the `transformers` config and create a copy of the
        # `ModelConfig` class with the subconfig and preferred backend updated.

        pretrained_config = getattr(model_config.pretrained_config, name)
        # ModelOpt currently doesn't quantize the vision part. Without setting quant config to None,
        # weight loading fails for vision.
        quant_config = model_config.quant_config if name == "text_config" else None
        submodel_config: ModelConfig[Cohere2VisionConfig] = dataclasses.replace(
           model_config,
           pretrained_config=pretrained_config,
           quant_config=quant_config 
        )
        # Make sure the top-level data type is replicated by default
        if (hasattr(submodel_config.pretrained_config, "torch_dtype")
                and submodel_config.pretrained_config.torch_dtype is None):
            submodel_config.pretrained_config.torch_dtype = model_config.pretrained_config.torch_dtype
        return submodel_config

    @property
    def mm_token_ids(self):
        return self.image_token_ids

    @staticmethod
    def _rename_llm_weights(key: str, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Rename LLM weights so that load_weights can read

        Example (key = "models.language_model"):
        models.language_model.embed_tokens -> models.embed_tokens
        """
        llm_weights = {}
        for k, v in weights.items():
            if k.startswith(key):
                new_k = "model" + k[len(key):]
                llm_weights[new_k] = v
            elif k.startswith("lm_head."):
                llm_weights[k] = v
        return llm_weights
    
    def load_weights(self, weights: dict[str, torch.Tensor]):

        llm_weights = self._rename_llm_weights("model.language_model", weights)
        self.llm.load_weights(llm_weights)

        vit_weights = filter_weights("model.vision_tower", weights)
        self.siglip_tower.load_weights(vit_weights)

        multimodal_projector_weights = filter_weights("model.multi_modal_projector", weights)
        self.multimodal_projector.load_weights(multimodal_projector_weights)

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
            f"[Cohere2Vision::forward]{num_context_requests=}, {num_generation_requests=}"
        )
        multimodal_params = kwargs.get("multimodal_params", [])
        pixel_values = [
            multimodal_param.multimodal_data["image"]["pixel_values"]
            for multimodal_param in multimodal_params
        ]

        multimodal_embeds = []
        multimodal_token_mask = None
        if len(pixel_values) > 0:
            image_features = self._get_image_features(
                pixel_values=torch.cat(pixel_values)
            )
            # Flatten leading batch/spatial dims into a single token sequence:
            # (B, H/d, W/d, hidden_size) -> (B*H/d*W/d, hidden_size)
            image_features = image_features.reshape(-1, image_features.shape[-1])
            multimodal_embeds = [image_features.contiguous()]

            # Get token type ids. 0 corresponds to text tokens, 1 corresponds to image tokens
            multimodal_token_mask = torch.isin(input_ids, self.image_token_ids)

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=multimodal_embeds,
            mm_token_ids=self.image_token_ids,
            **kwargs,
        )
        logits = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            image_token_mask=multimodal_token_mask,
            lora_params=kwargs.get("lora_params", None),
        )

        return logits
        
    @nvtx_range("[Vision] process")
    def _get_image_features(self, pixel_values):
        attn_metadata = self.siglip_tower.prepare_attn_metadata(
            pixel_values.shape[0]
        )
        with torch.autocast(device_type="cuda", dtype=self.model_dtype):
            image_features = self.siglip_tower(
                pixel_values,
                attn_metadata=attn_metadata
            )[-1]
            image_features = self.multimodal_projector(image_features)
        return image_features