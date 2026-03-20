from .modeling_qwen3_next import Qwen3NextForCausalLM
from .modeling_utils import register_auto_model


@register_auto_model("Qwen3_5MoeForCausalLM")
class Qwen3_5MoeForCausalLM(Qwen3NextForCausalLM):
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

    See Qwen3NextForCausalLM in modeling_qwen3_next.py for the equivalent
    class that serves the vanilla Qwen3NextForCausalLM architecture.
    """

    pass
