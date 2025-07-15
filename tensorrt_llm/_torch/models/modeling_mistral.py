import dataclasses
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import (AutoProcessor, AutoTokenizer, Mistral3Config,
                          MistralConfig, PretrainedConfig, PreTrainedModel)
from transformers.activations import ACT2FN

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_pixtral
from tensorrt_llm._torch.models.modeling_multimodal_utils import \
    fuse_input_embeds
from tensorrt_llm._torch.models.modeling_utils import (DecoderModel,
                                                       DecoderModelForCausalLM,
                                                       _load_weights_impl,
                                                       register_auto_model)
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.inputs import (ExtraProcessedInputs, InputProcessor,
                                 TextPrompt, register_input_processor)
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.logger import logger

_MULTIMODAL_ENV_NAME = "TLLM_MULTIMODAL_DISAGGREGATED"


# Make this a runtime lookup rather than a module-wide constant for easier unit testing.
def _is_disagg() -> bool:
    return os.getenv(_MULTIMODAL_ENV_NAME, "0") == "1"


class MistralAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[MistralConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
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
        residual: Optional[torch.Tensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
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
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[Any] = None,
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


class Mistral3InputProcessor(InputProcessor):

    def __init__(
        self,
        model_path: str,
        model_config: PretrainedConfig,
        tokenizer: Optional[AutoTokenizer],
        trust_remote_code: bool = False,
    ):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                      use_fast=False)

        # To abide by the `InputProcessor` interface.
        self.model_path = model_path
        self.model_config = model_config
        self.tokenizer = tokenizer

        self._device = "cuda"
        self._processor = AutoProcessor.from_pretrained(model_path,
                                                        use_fast=False)

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        images = inputs.get("multi_modal_data", {}).get("image")
        do_rescale = self._processor.image_processor.do_rescale
        if images is not None and isinstance(images[0], torch.Tensor):
            # The default multimodal input loader will normalize images to [0, 1] when the requested
            # format is "pt" (pytorch tensors), but not for "pil" (PIL images).
            do_rescale = False

        processed = self._processor(
            text=inputs["prompt"],
            images=images,
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
            processed = processed.to(self._device)
            # NOTE: `processed` is a dict-like object, but not actually a dict.
            extra_processed_inputs = {
                "multimodal_data": {
                    "image": {
                        **processed
                    }
                },
            }

        return input_ids, extra_processed_inputs


@register_auto_model("Mistral3ForConditionalGeneration")
# The below informs the registry which input registry to create for this in `tensorrt_llm/llmapi/llm.py`.
@register_input_processor(Mistral3InputProcessor, model_type="mistral3")
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
        super().__init__(config)

        self.model_config = model_config

        llm_model_config = self._get_sub_model_config(model_config,
                                                      "text_config")
        self.llm = MistralForCausalLM(llm_model_config)

        self._device = "cuda"
        vision_model_config = self._get_sub_model_config(
            model_config, "vision_config")
        self._vision_tower = modeling_pixtral.PixtralVisionModel(
            vision_model_config)
        self._multi_modal_projector = Mistral3MultiModalProjector(model_config)
        vision_feature_layer = config.vision_feature_layer
        if vision_feature_layer != -1:
            raise ValueError(
                f"Using intermediate layers ({vision_feature_layer}) in the `PixtralVisionModel` "
                f"is not supported. Please use `vision_feature_layer=-1`.")

        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)

        self._image_token_ids = torch.tensor([config.image_token_index],
                                             dtype=torch.int32,
                                             device=self._device)
        self._post_config()

    # This is necessary because the executor looks at
    # `model.model_config.pretrained_config.vocab_size`.
    def _post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def load_weights(self, weights: Dict, *args, **kwargs):
        llm_weights = _filter_weights(weights, "language_model.")
        self.llm.load_weights(llm_weights, *args, **kwargs)

        vit_weights = _filter_weights(weights, "vision_tower.")
        self._vision_tower.load_weights(vit_weights, *args, **kwargs)

        mm_projector_weights = _filter_weights(weights,
                                               "multi_modal_projector.")
        # `_load_weights_impl` assumes `config.hidden_size` exists, which is not the case for the
        # top-level `Mistral3Config`.
        self._multi_modal_projector.load_state_dict(mm_projector_weights)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Forward method."""
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(f"{num_context_requests=}, {num_generation_requests=}")

        multimodal_params = kwargs.get("multimodal_params", [])
        image_features = []
        multimodal_params_len = len(multimodal_params)
        if multimodal_params_len > 0:
            if multimodal_params_len != num_context_requests:
                raise RuntimeError(
                    f"Number of multimodal tensors ({multimodal_params_len}) should be equal to number of "
                    f"context requests ({num_context_requests}) in the batch.")
            # NOTES:
            # 1. the pixel values in `multimodal_data["image"]` might vary in (height, width) between
            #    images, making them unsafe to batch in general. The input processor also cannot produce
            #    them in a batch, since it is always called with a single input - otherwise, we would
            #    have been able to naturally leverage the padding / resizing capabilities of the underlying
            #    `PixtralProcessor`.
            # 2. After each `pixel_values` tensor has gone through the vision tower's `patch_conv` layer,
            #    they are divided into patches that are then concatenated in order to treat them as a
            #    single "sequence" in the vision tower's attention layers, so some form of batching still
            #    happens in the vision tower.
            image_features = [
                self._get_image_features(**x.multimodal_data["image"])
                for x in multimodal_params
            ]

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=image_features,
            mm_token_ids=self._image_token_ids,
        )

        return self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
        )

    @staticmethod
    def _get_sub_model_config(
        model_config: ModelConfig[MistralConfig],
        name: str,
    ) -> ModelConfig:
        # Extract the subconfig from the `transformers` config and shove it into our own
        # `ModelConfig` class.
        sub_model_config: ModelConfig[MistralConfig] = dataclasses.replace(
            model_config,
            pretrained_config=getattr(model_config.pretrained_config, name),
        )
        # Make sure some fields that are not explicitly included in the sub config, but present
        # in the top-level config, are replicated.
        if (hasattr(sub_model_config.pretrained_config, "torch_dtype")
                and sub_model_config.pretrained_config.torch_dtype is None):
            sub_model_config.pretrained_config.torch_dtype = model_config.pretrained_config.torch_dtype

        return sub_model_config

    # Original implementation:
    # https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/mistral3/
    # modeling_mistral3.py#L341
    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ):
        image_outputs = self._vision_tower(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

        image_features = self._multi_modal_projector(image_outputs.squeeze(0),
                                                     image_sizes)
        return image_features


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
        self._spatial_merge_size = config.spatial_merge_size
        self._patch_size = config.vision_config.patch_size
        self.merging_layer = Linear(
            in_features=hidden_size * self._spatial_merge_size**2,
            out_features=hidden_size,
            bias=False,
            dtype=config.torch_dtype,
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

        dtype = config.torch_dtype
        self.norm = RMSNorm(
            hidden_size=config.vision_config.hidden_size,
            # NOTE: the original implementation actually does not look at the config for this value.
            # We therefore hardcode the default value `1e-6` from `Mistral3RMSNorm`.
            eps=1e-6,
            dtype=dtype,
        )
        self.patch_merger = Mistral3PatchMerger(model_config)
        # We have hidden_size * the number of vision feature layers
        num_feature_layers = 1 if isinstance(config.vision_feature_layer,
                                             int) else len(
                                                 config.vision_feature_layer)
        self.linear_1 = Linear(
            in_features=config.vision_config.hidden_size * num_feature_layers,
            out_features=config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
            dtype=dtype,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = Linear(
            in_features=config.text_config.hidden_size,
            out_features=config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
            dtype=dtype,
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


def _filter_weights(weights: Dict[str, torch.Tensor],
                    prefix: str) -> Dict[str, torch.Tensor]:
    return {
        name[len(prefix):]: weight
        for name, weight in weights.items() if name.startswith(prefix)
    }
