# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
import math
from typing import Any, List, Literal, Optional, Tuple

import torch
from transformers import (AutoProcessor, AutoTokenizer, Gemma3Config,
                          PretrainedConfig, PreTrainedModel)
from transformers.image_utils import get_image_size, is_pil_image
from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper

from ..._utils import nvtx_range
from ...inputs import (BaseMultimodalDummyInputsBuilder,
                       BaseMultimodalInputProcessor, ContentFormat,
                       ExtraProcessedInputs, MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...inputs.multimodal import MultimodalItemMetadata
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention.backends import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.linear import Linear
from ..modules.rms_norm import RMSNorm
from ..peft.lora.ownership import DecoderLoraOwner
from .modeling_gemma3 import Gemma3ForCausalLM
from .modeling_multimodal_mixin import EncoderGroup, encode_multimodal_by_groups
from .modeling_multimodal_utils import (_MULTIMODAL_ENV_NAME, _is_mm_disagg,
                                        fuse_input_embeds)
from .modeling_siglip import SiglipVisionModel
from .modeling_utils import ModelConfig, filter_weights, register_auto_model


class Gemma3InputProcessor(BaseMultimodalInputProcessor,
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
            use_fast=self.use_fast)
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
        ).to(dtype=self.dtype)

        input_ids = processor_output["input_ids"]
        pixel_values = processor_output.get("pixel_values")

        # Pan-and-scan crops belong to their original image item.
        output_lengths = []
        if pixel_values is not None:
            image_kwargs = self.processor._merge_kwargs(
                Gemma3ProcessorKwargs,
                tokenizer_init_kwargs=self.processor.tokenizer.init_kwargs,
                do_rescale=do_rescale,
                return_tensors="pt",
            )["images_kwargs"]
            for image in images:
                height, width = ((image.height, image.width) if
                                 is_pil_image(image) else get_image_size(image))
                num_crops = 0
                if image_kwargs["do_pan_and_scan"]:
                    short, long = min(height, width), max(height, width)
                    if long / short >= image_kwargs[
                            "pan_and_scan_min_ratio_to_activate"]:
                        num_crops = min(
                            image_kwargs["pan_and_scan_max_num_crops"],
                            max(
                                2,
                                min(
                                    math.floor(long / short + 0.5), long //
                                    image_kwargs["pan_and_scan_min_crop_size"]))
                        )
                        if min(short, math.ceil(long / num_crops)
                               ) < image_kwargs["pan_and_scan_min_crop_size"]:
                            num_crops = 0
                output_lengths.append(
                    (1 + num_crops) * self.config.mm_tokens_per_image)
        return input_ids, pixel_values, output_lengths

    @torch.inference_mode()
    def call_with_text_prompt(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        input_ids, pixel_values, output_lengths = self._preprocess(inputs)
        multimodal_data = None
        if pixel_values is not None:
            multimodal_data = {
                "multimodal_data": {
                    "image": {
                        "pixel_values": pixel_values,
                        "output_embedding_lengths": output_lengths,
                    }
                },
            }
        return input_ids[0].to(torch.int32).tolist(), multimodal_data

    def get_mm_item_metadata(
        self,
        prompt_token_ids: List[int],
        multimodal_data: dict,
        mm_item_order: Optional[List[dict]] = None,
    ) -> Optional[MultimodalItemMetadata]:
        metadata = super().get_mm_item_metadata(prompt_token_ids,
                                                multimodal_data, mm_item_order)
        if metadata is not None or "image" not in multimodal_data:
            return metadata
        image = multimodal_data["image"]
        rows = image.get("output_embedding_lengths")
        if rows is None:
            rows = [self.config.mm_tokens_per_image
                    ] * image["pixel_values"].shape[0]
        refs = ([(entry["modality"], entry["index"]) for entry in mm_item_order]
                if mm_item_order else [("image", index)
                                       for index in range(len(rows))])
        return MultimodalItemMetadata(
            refs, output_embedding_lengths=[rows[index] for _, index in refs])


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
        content_format=ContentFormat.STRING,
    ))
class Gemma3VLM(PreTrainedModel):

    lora_owner = DecoderLoraOwner(module_path="language_model")

    @classmethod
    def get_preferred_kv_cache_manager_version(cls,
                                               pretrained_config: Any = None
                                               ) -> Literal["V2"]:
        """Prefer KV cache manager V2 — same VSWA rationale as
        Gemma3ForCausalLM (the wrapped text model)."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Optional[Literal["CPP", "PYTHON"]]:
        """Prefer the Python transceiver so disaggregated serving over NIXL keeps V2."""
        return "PYTHON"

    def __init__(self, model_config: ModelConfig[Gemma3Config]):
        if _is_mm_disagg():
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
        self._mm_token_ids = torch.tensor([config.image_token_index],
                                          dtype=torch.int32)

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

    @property
    def language_model(self) -> torch.nn.Module:
        return self.llm

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

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
        mm_embeds = []
        mm_token_mask = None
        if multimodal_params:
            image_features = encode_multimodal_by_groups(
                self.mm_encoder_groups, multimodal_params)
            mm_embeds = [image_features.contiguous()]

            # Get token type ids. 0 corresponds to text tokens, 1 corresponds to image tokens.
            mm_token_mask = torch.isin(input_ids, self.image_token_ids)

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embeds,
            mm_token_ids=self.image_token_ids,
            mm_token_indices=kwargs.get("mm_token_indices"),
            text_token_indices=kwargs.get("text_token_indices"),
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

    @property
    def mm_encoder_groups(self) -> Tuple[EncoderGroup, ...]:
        return (EncoderGroup(
            modalities=("image", ),
            encoder_fn=self._get_image_features,
            input_fields={"pixel_values": {
                "image": "pixel_values"
            }},
        ), )

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
        return self._mm_token_ids
