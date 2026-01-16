import copy
import dataclasses
from typing import Any, Dict, List, Tuple

import torch
import torchvision
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
from PIL import Image
from torch import nn
from transformers import (AutoProcessor, AutoTokenizer, Mistral3Config,
                          MistralConfig, PretrainedConfig, PreTrainedModel)
from transformers.activations import ACT2FN

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_pixtral
from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import \
    MistralTokenizer
from tensorrt_llm._torch.models.checkpoints.mistral.weight_mapper import \
    MistralWeightMapper
from tensorrt_llm._torch.models.modeling_mistral_large3 import (
    Mistral3Gate, MistralLarge3ForCausalLM)
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    _MULTIMODAL_ENV_NAME, _is_disagg, find_input_mm_embeds, fuse_input_embeds,
    get_multimodal_embeddings)
from tensorrt_llm._torch.models.modeling_utils import (DecoderModel,
                                                       DecoderModelForCausalLM,
                                                       _load_weights_impl,
                                                       filter_weights,
                                                       register_auto_model)
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.inputs import (BaseMultimodalDummyInputsBuilder,
                                 BaseMultimodalInputProcessor,
                                 ExtraProcessedInputs,
                                 MultimodalPlaceholderMetadata,
                                 MultimodalPlaceholderPlacement, TextPrompt,
                                 register_input_processor)
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.inputs.utils import encode_base64_image
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.logger import logger


class MistralAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[MistralConfig],
        layer_idx: int | None = None,
    ):
        config = model_config.pretrained_config
        rope_params = RopeParams.from_config(config)
        rope_params_section = getattr(config, "rope_scaling", None) or getattr(
            config, "rope_parameters", None)
        rope_type = getattr(rope_params_section, "rope_type", None)
        if rope_type == "yarn":
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.yarn,
                rope=rope_params,
                is_neox=False)
        else:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=rope_params,
            )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


class MistralDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[MistralConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        self.self_attn = MistralAttention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor | None = None,
        spec_metadata: SpecMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)
        return hidden_states, residual


class MistralModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[MistralConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            MistralDecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        spec_metadata: SpecMetadata | None = None,
        lora_params: Any | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("MistralForCausalLM")
class MistralForCausalLM(DecoderModelForCausalLM[MistralModel, MistralConfig]):

    def __init__(
        self,
        model_config: ModelConfig[MistralConfig],
    ):
        super().__init__(
            MistralModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )


class MistralCommonImageProcessor:

    def __init__(self, tokenizer: MistralTokenizer, dtype) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dtype = dtype

    @property
    def image_processor(self) -> ImageEncoder:
        image_encoder = self.tokenizer.instruct.mm_encoder
        assert isinstance(image_encoder, ImageEncoder)
        return image_encoder

    @property
    def image_break_id(self) -> int:
        return self.image_processor.special_ids.img_break

    @property
    def image_break_token_id(self) -> int:
        return self.image_break_id

    @property
    def image_token_id(self) -> int:
        return self.image_processor.special_ids.img

    @property
    def image_end_id(self) -> int:
        return self.image_processor.special_ids.img_end

    @property
    def image_end_token_id(self):
        return self.image_end_id

    @property
    def image_size(self) -> int:
        return self.image_processor.mm_config.max_image_size

    @property
    def patch_size(self) -> int:
        return self.image_processor.mm_config.image_patch_size

    def _get_num_multimodal_tokens(self, image_sizes):
        return {
            "num_image_tokens":
            [self.get_num_tokens_per_image(size) for size in image_sizes]
        }

    def get_num_tokens_per_image(self, image_sizes):
        h, w = image_sizes
        ncols, nrows = self.image_processor._image_to_num_tokens(
            Image.new("RGB", (w, h)))
        return ncols * nrows + nrows

    def __call__(self, text, images, **kwargs):
        mm_items = []
        if images:
            mm_items = [{
                "type": "image",
                "base64": encode_base64_image(image)
            } for image in images]

        conversation = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": text
            }, *mm_items]
        }]

        encoded = self.tokenizer.transformers_tokenizer.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors='pt')

        processed = {
            "input_ids": encoded.input_ids,
        }

        # text-only mode for VLM
        if "pixel_values" in encoded:
            processed.update({
                "pixel_values":
                encoded.pixel_values.to(self.dtype),
                "attention_mask":
                encoded.attention_mask,
                "image_sizes":
                torch.tensor([encoded.pixel_values.shape[2:]])
            })
        return processed


class Mistral3InputProcessor(BaseMultimodalInputProcessor,
                             BaseMultimodalDummyInputsBuilder):

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: AutoTokenizer | None,
        trust_remote_code: bool = False,
        model_type: str = "mistral3",
        **kwargs,
    ):
        super().__init__(model_path=model_path,
                         config=config,
                         tokenizer=tokenizer,
                         trust_remote_code=trust_remote_code,
                         **kwargs)
        self._config = config
        self._dtype = self._config.torch_dtype
        self._tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            model_path,
            config=config,
            use_fast=self.use_fast,
            trust_remote_code=trust_remote_code)
        self._model_path = model_path
        if model_type == "mistral_large_3":
            # For mistral large 3, we add chat template in the model forward, and the
            # MistralCommonImageProcessor is used to process the input when both text and images are provided.
            # When the input only contains text, we use the text processor to process the input.
            self._processor = MistralCommonImageProcessor(
                tokenizer=self._tokenizer, dtype=self.dtype)
            self.text_processor = AutoProcessor.from_pretrained(
                model_path,
                use_fast=self.use_fast,
                trust_remote_code=trust_remote_code)
        else:
            # For other mistral models, we use the AutoProcessor to process the input.
            self._processor = AutoProcessor.from_pretrained(
                model_path,
                use_fast=self.use_fast,
                trust_remote_code=trust_remote_code)
            self.text_processor = self._processor

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

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], ExtraProcessedInputs | None]:
        images = inputs.get("multi_modal_data", {}).get("image")
        do_rescale = getattr(self.processor.image_processor, "do_rescale",
                             False)
        if images is not None and isinstance(images[0], torch.Tensor):
            # The default multimodal input loader will normalize images to [0, 1] when the requested
            # format is "pt" (pytorch tensors), but not for "pil" (PIL images).
            do_rescale = False

        if images is not None:
            processed = self.processor(
                text=inputs["prompt"],
                images=images,
                do_rescale=do_rescale,
            )
        else:
            processed = self.text_processor(
                text=inputs["prompt"],
                do_rescale=do_rescale,
            )
        input_ids = processed.pop("input_ids").tolist()[0]
        # Remaining in `processed`:
        # * "attention_mask": [B, num_input_tokens]
        # * "pixel_values": [B, C, H, W]
        # * "image_sizes": [B, 2]
        extra_processed_inputs = None
        pixel_values = processed.get("pixel_values")
        if pixel_values is not None:
            # We have no use for the `attention_mask`.
            processed.pop("attention_mask")
            # `image_sizes` is a `[B, 2]` tensor indicating the height and width of each image in the
            # request. If we keep it as a regular tensor, it would get converted to a CUDA tensor before
            # reaching the model forward. Since its values are used to infer the amount of padding
            # + slice the patch embeddings, this would incur a D2H copy. We therefore convert it to a
            # list here to avoid this.
            processed["image_sizes"] = processed["image_sizes"].tolist()
            # NOTE: `processed` is a dict-like object, but not actually a dict.
            extra_processed_inputs = {
                "multimodal_data": {
                    "image": {
                        **processed
                    }
                }
            }

        return input_ids, extra_processed_inputs

    def get_vocab_size(self) -> int:
        """Return the vocab size of the model."""
        # Unlike some other VLMs, mistral3's vocab size is stored in its `text_config`, not the top-level
        # config.
        return self.config.text_config.vocab_size

    def get_mm_token_ids(self) -> torch.Tensor:
        """Get the IDs of all multimodal tokens (placeholders and special tokens alike)."""
        return torch.tensor([
            # This is the `[IMG]` token id inserted into the prompt that should be replaced with image
            # embeddings.
            self.processor.image_token_id,
            # This is the `[IMG_BREAK]` token id at the end of every "row".
            self.processor.image_break_token_id,
            # This is the `[IMG_END]` token id to signify the end of an image.
            self.processor.image_end_token_id,
        ])

    def get_mm_special_token_ids(self) -> torch.Tensor:
        """Get the IDs of special multimodal tokens (placeholders not included)."""
        return torch.tensor([
            self.processor.image_break_token_id,
            self.processor.image_end_token_id,
        ])


class MistralCommonInputProcessor(Mistral3InputProcessor):

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: AutoTokenizer,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        tokenizer = self.load_tokenizer(model_path,
                                        config=config,
                                        tokenizer=tokenizer)
        super().__init__(model_path=model_path,
                         config=config,
                         tokenizer=tokenizer,
                         trust_remote_code=trust_remote_code,
                         model_type=getattr(config, "input_processor_type",
                                            "mistral3"),
                         **kwargs)

    @staticmethod
    def load_tokenizer(model_path: str,
                       config: PretrainedConfig,
                       tokenizer: AutoTokenizer | None = None):
        if getattr(config, "input_processor_type", None) == "mistral_large_3":
            try:
                return MistralTokenizer.from_pretrained(model_path)

            except ValueError:
                logger.info(
                    f"Could not load mistral-common tokenizer from {model_path}, falling back to HuggingFace"
                )

        tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            model_path, config=config, use_fast=True, trust_remote_code=True)
        return tokenizer


@register_auto_model("Mistral3ForConditionalGeneration")
@register_auto_model("PixtralForConditionalGeneration")
@register_input_processor(
    MistralCommonInputProcessor,
    model_type="mistral_large_3",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            # NOTE: mistral-common uses the tokenizer to set placeholders, this will be ignored
            "image": "[IMG]",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ))
@register_input_processor(
    MistralCommonInputProcessor,
    model_type="mistral3",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "[IMG]",
        },
        # NOTE: for mistral3 multimodal models, it does not strictly have to be before the text.
        # Ref: https://github.com/mistralai/mistral-common/blob/039465db2bdc0486df36365c9bdb428188482a18/
        #      src/mistral_common/tokens/tokenizers/base.py#L326
        # However, accuracy tests show that the model generates higher quality output when the image
        # precedes the text (the relative difference can be as much as ~30% for both vLLM and TRT-LLM).
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ))
class Mistral3VLM(PreTrainedModel):
    """Mistral3VLM implementation for TRTLLM.

    NOTE: for the time being, image tokens are only placed after the text (see
    `tensorrt_llm/inputs/utils.py`).
    """

    def __init__(
        self,
        model_config: ModelConfig[Mistral3Config],
    ):
        if _is_disagg():
            raise NotImplementedError(
                "Mistral3VLM does not support disaggregated inference yet. Please unset "
                f"the {_MULTIMODAL_ENV_NAME} environment variable, or set it to '0'."
            )

        config = model_config.pretrained_config
        self._supports_sdpa = True
        super().__init__(config)

        vision_feature_layer = getattr(config, "vision_feature_layer", -1)
        if vision_feature_layer != -1:
            raise ValueError(
                f"Using intermediate layers ({vision_feature_layer}) in the `PixtralVisionModel` "
                f"is not supported. Please use `vision_feature_layer=-1`.")

        self._device = "cuda"
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        image_token_index = getattr(
            config, "image_token_index", None) or getattr(
                config.vision_config, "image_token_id", None)
        self._image_token_ids = torch.tensor([image_token_index],
                                             dtype=torch.int32,
                                             device=self._device)

        model_config_cp = copy.deepcopy(model_config)

        llm_model_config = self._get_sub_model_config(model_config_cp,
                                                      "text_config")
        self.model_config = model_config_cp
        llm_class = MistralForCausalLM
        if llm_model_config.pretrained_config.architectures[
                0] == "MistralLarge3ForCausalLM":
            llm_class = MistralLarge3ForCausalLM

        llm_model_config.pretrained_config.gate_cls = Mistral3Gate
        self.llm = llm_class(llm_model_config)
        self.model_config.extra_attrs.update(llm_model_config.extra_attrs)

        # NOTE: current `modelopt` does not support quantizing the vision portion.
        # NOTE: attn_backend: Pixtral head size not always divisible by 128
        vision_model_config = self._get_sub_model_config(model_config_cp,
                                                         "vision_config",
                                                         attn_backend="VANILLA",
                                                         quant_config=None)

        self._vision_tower = modeling_pixtral.PixtralVisionModel(
            vision_model_config)
        self._multi_modal_projector = Mistral3MultiModalProjector(
            model_config).eval().to(self._device)
        self._post_config()

    # This is necessary because the executor looks at
    # `model.model_config.pretrained_config.vocab_size`.
    def _post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def load_weights(self, weights: Dict, weight_mapper=None, *args, **kwargs):
        vit_params_map = None
        if weight_mapper:
            if isinstance(weight_mapper, MistralWeightMapper):
                vit_params_map = weight_mapper.pixtral_mapping

        llm_weights = filter_weights(weights=weights, prefix="language_model")
        logger.debug(f"Loading weights for {type(self.llm)}")
        self.llm.load_weights(llm_weights)
        logger.debug(f"Successfully loaded weights for {type(self.llm)}")

        vit_weights = filter_weights(weights=weights, prefix="vision_tower")
        logger.debug(f"Loading weights for {type(self._vision_tower)}")

        if vit_params_map is not None:
            vit_weights = weight_mapper.rename_by_params_map(
                weights=vit_weights, params_map=vit_params_map)

        self._vision_tower.load_weights(vit_weights)
        logger.debug(
            f"Successfully loaded weights for {type(self._vision_tower)}")

        logger.debug(f"Loading weights for {type(self._multi_modal_projector)}")
        mm_projector_weights = filter_weights(weights=weights,
                                              prefix="multi_modal_projector")

        if vit_params_map is not None:
            mm_projector_weights = weight_mapper.rename_by_params_map(
                weights=mm_projector_weights, params_map=vit_params_map)
        self._multi_modal_projector.load_state_dict(mm_projector_weights)
        logger.debug(
            f"Successfully loaded weights for {type(self._multi_modal_projector)}"
        )

    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    @property
    def load_draft_weights(self):
        return self.llm.load_draft_weights

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata: SpecMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward method."""
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(f"{num_context_requests=}, {num_generation_requests=}")

        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds = []
        multimodal_params_len = len(multimodal_params)
        if multimodal_params_len > 0:
            mm_embeds = get_multimodal_embeddings(
                encoder_forward_fn=self._vision_forward,
                multimodal_params=multimodal_params[:num_context_requests],
            )
            mm_embeds = find_input_mm_embeds(
                mm_embeds, multimodal_params[:num_context_requests])

        with nvtx_range("[mistral] Fuse input embeds"):
            input_ids, inputs_embeds = fuse_input_embeds(
                embedding_layer=self.llm.model.embed_tokens,
                input_ids=input_ids,
                mm_embeds=mm_embeds,
                mm_token_ids=self._image_token_ids,
                **kwargs,
            )

        return self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            spec_metadata=spec_metadata,
        )

    @staticmethod
    def _get_sub_model_config(
        model_config: ModelConfig[MistralConfig],
        name: str,
        **changes,
    ) -> ModelConfig:
        # Extract the subconfig from the `transformers` config and shove it into our own
        # `ModelConfig` class.
        assert name in [
            "text_config", "vision_config"
        ], f"Expected subconfig name to be either 'text_config' or 'vision_config'. Got {name} instead."
        pretrained_config = getattr(model_config.pretrained_config, name)

        sub_model_config: ModelConfig[MistralConfig] = dataclasses.replace(
            model_config,
            pretrained_config=getattr(model_config.pretrained_config, name),
            **changes,
        )
        if name == "text_config":
            sub_model_config._frozen = False
            sub_model_config.skip_create_weights_in_init = True
            if not hasattr(
                    sub_model_config.pretrained_config, "architectures"
            ) or sub_model_config.pretrained_config.architectures is None:
                sub_model_config.pretrained_config.architectures = model_config.pretrained_config.architectures
            sub_model_config._frozen = True

        # Make sure some fields that are not explicitly included in the sub config, but present
        # in the top-level config, are replicated.
        if (hasattr(sub_model_config.pretrained_config, "torch_dtype")
                and sub_model_config.pretrained_config.torch_dtype is None):
            sub_model_config.pretrained_config.torch_dtype = model_config.pretrained_config.torch_dtype or torch.bfloat16

        if name == "vision_config":
            pretrained_config = sub_model_config.pretrained_config
            defaults = {
                "head_dim": pretrained_config.hidden_size //
                pretrained_config.num_attention_heads,
                "hidden_act": "silu",
            }
            for attr, default in defaults.items():
                if not hasattr(pretrained_config, attr):
                    setattr(pretrained_config, attr, default)

        return sub_model_config

    # NOTE: this is defined as a separate method with this specific signature in order to be compatible
    # with `get_multimodal_embeddings`.
    def _vision_forward(
            self,
            multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        multimodal_params_len = len(multimodal_params)
        pixel_values = [
            x.multimodal_data["image"]["pixel_values"]
            for x in multimodal_params
        ]
        image_sizes = [
            x.multimodal_data["image"]["image_sizes"] for x in multimodal_params
        ]
        if not (len(pixel_values) == len(image_sizes) == multimodal_params_len):
            raise ValueError(
                f"Expected as many `pixel_values` ({len(pixel_values)}) and "
                f"`image_sizes` ({len(image_sizes)}) as number of multimodal parameters "
                f"({multimodal_params_len}).")
        image_sizes = [torch.tensor(x) for x in image_sizes]
        batched_pixel_values, batched_image_sizes = self.batch_pixel_values(
            pixel_values=pixel_values, image_sizes=image_sizes)
        mm_embeds = [
            self._get_image_features(pixel_values=batched_pixel_values,
                                     image_sizes=batched_image_sizes)
        ]

        return mm_embeds

    # Original implementation:
    # https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/mistral3/
    # modeling_mistral3.py#L341
    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ):
        with nvtx_range("[mistral] ViT"):
            image_outputs = self._vision_tower(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )

        with nvtx_range("[mistral] MM projector"):
            image_features = self._multi_modal_projector(
                image_outputs.squeeze(0), image_sizes)
        return image_features

    # Original HF implementation:
    # https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/pixtral/
    # image_processing_pixtral.py#L276
    # We switch to using torchvision's padding functionality since it supports torch tensors
    # (the transformers one expected numpy arrays).
    @staticmethod
    @torch.inference_mode()
    @nvtx_range("[mistral] Batch images")
    def batch_pixel_values(
        pixel_values: List[torch.Tensor],
        image_sizes: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTES:
        # * `pixel_values` is a list of `[B_idx, C, H_idx, W_idx]` tensors, i.e. a batch of images as
        #   padded + batched by the input processor.
        #   The height (H_idx) and width (W_idx) of each element need not coincide.
        # * Similarly, each element in `image_sizes` describes the original image sizes prior to
        #   padding for the corresponding element in `pixel_values`.

        # The below creates a single `[sum(B_idx), 2]` tensor describing all image sizes, and then
        # calculates the maximum height / width across all of them.
        batched_image_sizes = torch.cat(image_sizes)
        max_shape = batched_image_sizes.max(dim=0).values

        # This next step then pads the pixel values potentially a second time by using the `max_shape`
        # computed above. Note that as far as this function is concerned, the original sizes for
        # batching purposes can be deduced from looking at the tensors in `pixel_values`, NOT in
        # `image_sizes`.
        pixel_values = [
            torchvision.transforms.v2.functional.pad(
                image,
                # Per torchvision docs, this should be in LTRB order if it's a sequence of 4 numbers.
                padding=[
                    0, 0, max_shape[1] - image.shape[-1],
                    max_shape[0] - image.shape[-2]
                ],
                # Values extracted from HF implementation.
                fill=0.0,
                padding_mode="constant",
            ) for image in pixel_values
        ]
        return torch.cat(pixel_values), batched_image_sizes

    @property
    def mm_token_ids(self):
        return self._image_token_ids

    def load_draft_weights(
            self,
            weights: Dict,
            weight_mapper: MistralWeightMapper | None = None) -> None:
        self.llm.load_draft_weights(weights, weight_mapper=weight_mapper)


# Original implementation:
# https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/mistral3/modeling_mistral3.py#L66
# NOTE: the main difference is the usage of TRTLLM's own `Linear` layer over pytorch's built-in layer.
class Mistral3PatchMerger(torch.nn.Module):

    def __init__(self, model_config: ModelConfig[Mistral3Config]):
        super().__init__()
        config = model_config.pretrained_config
        # Both the below are needed in order to use `_load_weights_impl`.
        self.model_config = model_config
        self.config = config

        hidden_size = config.vision_config.hidden_size
        self._spatial_merge_size = getattr(
            config, "spatial_merge_size", None) or getattr(
                config.vision_config, "spatial_merge_size")
        self._patch_size = config.vision_config.patch_size
        self.merging_layer = Linear(
            in_features=hidden_size * self._spatial_merge_size**2,
            out_features=hidden_size,
            bias=False,
            dtype=config.torch_dtype or model_config.torch_dtype,
            mapping=model_config.mapping,
        )

    @torch.inference_mode()
    def forward(self, image_features: torch.Tensor,
                image_sizes: torch.Tensor) -> torch.Tensor:
        image_sizes = [(image_size[0] // self._patch_size,
                        image_size[1] // self._patch_size)
                       for image_size in image_sizes]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(
                image_features.split(tokens_per_image)):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0,
                                                            1).unsqueeze(0)
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self._spatial_merge_size,
                stride=self._spatial_merge_size)
            grid = grid.view(d * self._spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features)
        return image_features

    def load_weights(self, weights):
        _load_weights_impl(self, weights)


# Original implementation:
# https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/mistral3/
# modeling_mistral3.py#L104C1-L127C29
class Mistral3MultiModalProjector(torch.nn.Module):

    def __init__(self, model_config: ModelConfig[Mistral3Config]):
        super().__init__()
        config = model_config.pretrained_config
        # Both the below are needed in order to use `_load_weights_impl`.
        self.model_config = model_config
        self.config = config

        dtype = config.torch_dtype or model_config.torch_dtype
        self.norm = RMSNorm(
            hidden_size=config.vision_config.hidden_size,
            # NOTE: the original implementation actually does not look at the config for this value.
            # We therefore hardcode the default value `1e-6` from `Mistral3RMSNorm`.
            eps=1e-6,
            dtype=dtype,
        )
        self.patch_merger = Mistral3PatchMerger(model_config)
        # We have hidden_size * the number of vision feature layers
        vision_feature_layer = getattr(config, "vision_feature_layer", -1)
        num_feature_layers = 1 if isinstance(vision_feature_layer,
                                             int) else len(vision_feature_layer)
        self.linear_1 = Linear(
            in_features=config.vision_config.hidden_size * num_feature_layers,
            out_features=config.text_config.hidden_size,
            bias=getattr(config, "multimodal_projector_bias", None),
            dtype=dtype,
            mapping=model_config.mapping,
        )
        self.act = ACT2FN[getattr(config, "projector_hidden_act", "gelu")]
        self.linear_2 = Linear(
            in_features=config.text_config.hidden_size,
            out_features=config.text_config.hidden_size,
            bias=getattr(config, "multimodal_projector_bias", None),
            dtype=dtype,
            mapping=model_config.mapping,
        )

    @torch.inference_mode()
    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor):
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def load_weights(self, weights):
        _load_weights_impl(self, weights)
