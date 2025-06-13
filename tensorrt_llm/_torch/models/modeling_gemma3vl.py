import copy
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel, AutoProcessor, Gemma3Config,
                          PretrainedConfig, PreTrainedModel)
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

from ..._utils import nvtx_range
from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       register_input_processor)
from ...llmapi.utils import download_hf_model
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_gemma3 import Gemma3ForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import ModelConfig, filter_weights, register_auto_model


class Gemma3InputProcessor(InputProcessor):

    def __init__(self, model_path, model_config, tokenizer, trust_remote_code):

        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.model_config = model_config
        self.device = 'cuda'

        # Determine the actual local path for model files
        if os.path.isdir(model_path):
            local_model_path = model_path
        else:
            local_model_path = download_hf_model(model_path)

        # Partially load the model to reduce memory usage(Vision tower and multi-modal projector)
        hf_model_config = AutoConfig.from_pretrained(local_model_path)
        self.dtype = hf_model_config.text_config.torch_dtype
        module_dict = nn.ModuleDict({
            "vision_tower":
            AutoModel.from_config(hf_model_config.vision_config),
            "multi_modal_projector":
            Gemma3MultiModalProjector(hf_model_config)
        })
        missing_keys, _ = load_sharded_checkpoint(module_dict,
                                                  local_model_path,
                                                  strict=False)
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        hf_vision_tower = module_dict["vision_tower"].to(self.dtype).to(
            self.device)
        hf_mm_projector = module_dict["multi_modal_projector"].to(
            self.dtype).to(self.device)

        # Use HF vision tower. To be replaced with TRTLLM vision tower.
        self.vision_tower = hf_vision_tower

        # Use HF multi-modal projector
        self.mm_projector = hf_mm_projector

    @nvtx_range("[Vision] preprocess")
    def _preprocess(self, inputs):
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
        assert 'image' in mm_data
        processor_output = self.processor(text=text_prompt,
                                          images=mm_data["image"][0],
                                          return_dict=True,
                                          return_tensors="pt",
                                          device=self.device).to(
                                              'cuda', dtype=torch.bfloat16)
        result_dict = {}
        result_dict["prompt"] = inputs["prompt"]
        result_dict["multimodal_data"] = {
            "image": [processor_output["pixel_values"]]
        }
        result_dict["mm_processor_kwargs"] = {}
        for key in ["input_ids", "token_type_ids", "pixel_values"]:
            result_dict["mm_processor_kwargs"][key] = processor_output[key]

        return [result_dict]

    @nvtx_range("[Vision] process")
    def _process(self, pixel_values):
        image_features: Tuple[torch.Tensor] = self.vision_tower(
            pixel_values).last_hidden_state
        image_features = self.mm_projector(image_features)
        return image_features

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        preprocess_outputs = self._preprocess(inputs)
        pixel_values = preprocess_outputs[0]["mm_processor_kwargs"][
            "pixel_values"]
        input_ids = preprocess_outputs[0]["mm_processor_kwargs"]["input_ids"]
        mm_features = self._process(pixel_values)
        return input_ids[0].to(torch.int32).tolist(), {
            "mm_embedding": mm_features
        }


@register_auto_model("Gemma3ForConditionalGeneration")
@register_input_processor(Gemma3InputProcessor, model_type="gemma3")
class Gemma3Model(PreTrainedModel):
    config_class = Gemma3Config

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs) -> None:
        config = model_config.pretrained_config
        super().__init__(config)
        if hasattr(self, "llm"):
            return

        self.image_token_index = config.image_token_index

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = model_config.pretrained_config.text_config

        llm_model_config.pretrained_config.torch_dtype = torch.bfloat16
        self.llm = Gemma3ForCausalLM(llm_model_config)

        self.model_config = model_config
        self.vocab_size = config.text_config.vocab_size
        self.model_dtype = getattr(config.text_config, "torch_dtype",
                                   torch.float16)
        logger.info(f"[Gemma3Model::__init__]{self.dtype=} {self.model_dtype=}")

        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):

        weights = filter_weights("language_model", weights)
        self.llm.load_weights(weights)

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

        mm_embed = kwargs.get("multi_modal_data", [])
        assert mm_embed == [] or len(
            mm_embed
        ) == num_context_requests, "Number of multimodal features (if provided) should be equal to number of context requests"

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embed,
            mm_token_ids=torch.tensor([self.image_token_index
                                       ]).to(input_ids.device))
        logits = self.llm.forward(attn_metadata, input_ids, position_ids,
                                  inputs_embeds, return_context_logits)
        return logits


AutoModel.register(Gemma3Config, Gemma3Model)
