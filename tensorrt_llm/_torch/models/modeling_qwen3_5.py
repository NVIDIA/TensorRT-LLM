from transformers import Qwen3NextConfig

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper

from ..model_config import ModelConfig
from .modeling_qwen3_next import Qwen3NextModel
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import register_auto_model


@register_auto_model("Qwen3_5MoeForCausalLM")
class Qwen3_5MoeForCausalLM(SpecDecOneEngineForCausalLM[Qwen3NextModel, Qwen3NextConfig]):
    """Thin wrapper that registers the Qwen3.5 MoE text architecture.

    Qwen3.5 text reuses the same model internals as Qwen3Next
    (Qwen3NextModel) -- the transformer, linear-attention layers, MoE blocks,
    and hybrid cache logic are all shared.  This separate class exists because:

    1. HF architecture routing: the HF checkpoint advertises
       Qwen3_5MoeForCausalLM (or top-level Qwen3_5MoeForConditionalGeneration
       with nested text_config), so TRT-LLM needs a matching
       @register_auto_model entry to route to the right class.
    2. Weight mapper dispatch: registering a distinct architecture name lets
       the checkpoint loader pick Qwen3_5MoeHfWeightMapper (which handles
       Qwen3.5-specific HF weight layout differences like split linear-attention
       projections and fused MoE expert tensors) instead of the base
       Qwen3NextHfWeightMapper.
    3. post_load_weights: wires next_layer_layernorm references needed by
       Qwen3Next linear-attention layers; identical to Qwen3NextForCausalLM.

    See Qwen3NextForCausalLM in modeling_qwen3_next.py for the equivalent
    class that serves the vanilla Qwen3NextForCausalLM architecture.
    """

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig]):
        super().__init__(Qwen3NextModel(model_config), model_config)
        self.preload_weight_modules = self.model.preload_weight_modules

    def load_weights(
        self,
        weights: dict,
        weight_mapper: BaseWeightMapper | None = None,
        params_map: dict[str, str] | None = None,
        allow_partial_loading: bool = False,
    ):
        if weight_mapper is not None:
            weights = weight_mapper.preprocess_weights(weights)
        super().load_weights(
            weights,
            weight_mapper=weight_mapper,
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )

    def post_load_weights(self):
        assert self.config is not None
        for idx, layer in enumerate(self.model.layers[: self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[idx + 1].input_layernorm
