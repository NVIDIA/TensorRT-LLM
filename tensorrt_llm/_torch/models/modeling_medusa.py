from typing import Dict, Optional

import torch
from torch import nn
from transformers import LlamaConfig

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                              WeightsLoadingConfig)
from ..speculative import SpecMetadata
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .modeling_llama import LlamaModel
from .modeling_utils import DecoderModelForCausalLM, register_auto_model


class MedusaModel(nn.Module):
    """
    Medusa model that combines a base LLM with a single Medusa head.

    This model processes hidden states through the base model and then
    through a Medusa head for draft token generation.
    """

    def __init__(self, model_config: ModelConfig[LlamaConfig], head_idx: int):
        super().__init__()
        self.model_config = model_config
        self.head_idx = head_idx
        self.hidden_size = model_config.pretrained_config.hidden_size
        self.vocab_size = model_config.pretrained_config.vocab_size

        # Get the number of layers from eagle_config
        # (Medusa uses the same config structure for parallel draft heads)
        self.num_layers = model_config.pretrained_config.eagle_config.get(
            "parallel_draft_heads_num_layers", 0
        )
        assert self.num_layers > 0, "parallel_draft_heads_num_layers must be > 0"

        # Create the Medusa layers
        self.medusa_layers = nn.ModuleList(
            [Linear(self.hidden_size,
                         self.hidden_size,
                         bias=True,
                         dtype=model_config.torch_dtype) for _ in range(self.num_layers)]
        )
        self.act = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the Medusa head.

        Args:
            hidden_states: Hidden states from the base model

        Returns:
            Processed hidden states after Medusa head
        """
        for layer in self.medusa_layers:
            hidden_states_out = layer(hidden_states)
            hidden_states = self.act(hidden_states_out) + hidden_states
        return hidden_states


@register_auto_model("MedusaForCausalLM")
class MedusaForCausalLM(DecoderModelForCausalLM[LlamaModel, LlamaConfig]):
    """
    Medusa model for causal language modeling with speculative decoding.

    This represents a single Medusa head with its own logits processor.
    Multiple instances of this class should be created for parallel draft heads.
    """

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        start_layer_idx: int = 0,
        head_idx: int = 0,
    ):
        # Initialize with the base model (shared or new)

        super().__init__(MedusaModel(model_config, start_layer_idx + head_idx),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

        self.head_idx = head_idx

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the Medusa model.

        Args:
            attn_metadata: Attention metadata for the forward pass
            input_ids: Input token IDs
            position_ids: Position IDs for positional encoding
            inputs_embeds: Pre-computed input embeddings (optional)
            return_context_logits: Whether to return context logits
            spec_metadata: Speculative decoding metadata
            hidden_states: Pre-computed hidden states (optional, for reusing base model output)
            **kwargs: Additional keyword arguments

        Returns:
            Logits from the language model head after Medusa head processing
        """

        # Process through Medusa head
        output = self.model(hidden_states)

        # Process through logits processor (each head has its own)
        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        """
        Load weights into the model.

        Args:
            weights: Dictionary mapping weight names to tensors
            weight_mapper: Weight mapper for handling weight name conversions
        """
        # Prepend "model." to weight keys if not already present
        # (except for lm_head and medusa_head)
        new_weights = {}
        for k, v in weights.items():
            if "lm_head" not in k and "medusa_head" not in k:
                new_k = "model." + k
            else:
                new_k = k
            new_weights[new_k] = v

        super().load_weights(weights=new_weights, weight_mapper=weight_mapper)
