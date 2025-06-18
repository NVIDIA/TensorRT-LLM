import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import (AutoProcessor, AutoTokenizer, Mistral3Config,
                          Mistral3ForConditionalGeneration, MistralConfig,
                          PretrainedConfig, PreTrainedModel)

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_utils import \
    fuse_input_embeds
from tensorrt_llm._torch.models.modeling_utils import (DecoderModel,
                                                       DecoderModelForCausalLM,
                                                       register_auto_model)
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.inputs import (ExtraProcessedInputs, InputProcessor,
                                 TextPrompt, register_input_processor)
from tensorrt_llm.llmapi import SamplingParams


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

        # Ref implementation:
        # https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/
        # mistral3/modeling_mistral3.py

        # Load the whole model (onto CPU) so that weights can be loaded auto-magically.
        # TODO: see if there is a way to just load the vision tower / multi modal projector.
        ref_model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path).eval()

        self._device = "cuda"
        self._vision_tower = ref_model.vision_tower.to(self._device)
        self._vision_feature_layer = model_config.vision_feature_layer
        self._multi_modal_projector = ref_model.multi_modal_projector.to(
            self._device)

        self._processor = AutoProcessor.from_pretrained(model_path,
                                                        use_fast=False)

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        images = inputs.get("multi_modal_data", {}).get("image")
        if images is not None:
            # Although the `PixtralProcessor` supports PIL images, numpy arrays, and torch tensors,
            # they are expected to be in the [0, 255] range, not [0, 1]. This is a heuristic to
            # convert them to the expected range.
            if isinstance(images[0], torch.Tensor) and images[0].mean() < 1.0:
                images = [image * 255 for image in images]

        processed = self._processor(
            text=inputs["prompt"],
            images=images,
        )
        input_ids = processed.pop("input_ids").tolist()[0]
        # Remaining in `processed`:
        # * "attention_mask": [B, num_input_tokens]
        # * "pixel_values": [B, C, H, W]
        # * "image_sizes": [B, 2]
        extra_processed_inputs = None
        pixel_values = processed.get("pixel_values")
        if pixel_values is not None:
            image_features = self._get_image_features(
                pixel_values=pixel_values.to(self._device),
                image_sizes=processed["image_sizes"].to(self._device),
            )
            extra_processed_inputs = {"mm_embedding": image_features}

        return input_ids, extra_processed_inputs

    # Original implementation:
    # https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/mistral3/
    # modeling_mistral3.py#L341
    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ):
        vision_feature_layer = self._vision_feature_layer
        output_hidden_states = (vision_feature_layer != -1)
        image_outputs = self._vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=output_hidden_states)
        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them.
        if isinstance(vision_feature_layer, int):
            if vision_feature_layer == -1:
                selected_image_feature = image_outputs.last_hidden_state
            else:
                selected_image_feature = image_outputs.hidden_states[
                    vision_feature_layer]
        else:
            hs_pool = [
                image_outputs.hidden_states[layer_idx]
                for layer_idx in vision_feature_layer
            ]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self._multi_modal_projector(
            selected_image_feature.squeeze(0), image_sizes)
        return image_features


@register_auto_model("Mistral3ForConditionalGeneration")
# The below informs the registry which input registry to create for this in `tensorrt_llm/llmapi/llm.py`.
@register_input_processor(Mistral3InputProcessor, model_type="mistral3")
class Mistral3VLM(PreTrainedModel):

    def __init__(
        self,
        model_config: ModelConfig[Mistral3Config],
    ):
        config = model_config.pretrained_config
        super().__init__(config)

        self.model_config = model_config

        # Extract the `text_config` from the `transformers` config and shove it into our own
        # `ModelConfig` class.
        llm_model_config: ModelConfig[MistralConfig] = dataclasses.replace(
            model_config,
            pretrained_config=model_config.pretrained_config.text_config,
        )
        # Make sure some fields that are not explicitly included in the `text_config`, but present
        # in the top-level config, are replicated.
        llm_model_config.pretrained_config.torch_dtype = model_config.pretrained_config.torch_dtype
        self.llm = MistralForCausalLM(llm_model_config)

        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)

        self._device = "cuda"
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
        filtered_weights = _filter_weights(weights, "language_model.")
        self.llm.load_weights(filtered_weights, *args, **kwargs)

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
        multi_modal_data = kwargs.get("multi_modal_data", [])
        if len(multi_modal_data) > 0:
            input_ids, inputs_embeds = fuse_input_embeds(
                embedding_layer=self.llm.model.embed_tokens,
                input_ids=input_ids,
                mm_embeds=multi_modal_data,
                mm_token_ids=self._image_token_ids,
            )

        return self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
        )


def _filter_weights(weights: Dict[str, torch.Tensor],
                    prefix: str) -> Dict[str, torch.Tensor]:
    return {
        name[len(prefix):]: weight
        for name, weight in weights.items() if name.startswith(prefix)
    }
