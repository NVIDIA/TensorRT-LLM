from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "DeciLMForCausalLM")
class NemotronNASHfWeightMapper(HfWeightMapper):
    """Weight mapper for Nemotron-NAS (DeciLM) models.

    NemotronNASModel rewrites ``config.num_key_value_heads`` to a per-layer
    list (with ``0`` for NO-OP blocks). The base ``_duplicate_kv_weights``
    expects a scalar and fails on a list. This mapper tracks the current
    layer index (parsed from the module path, e.g.
    ``model.layers.3.self_attn.qkv_proj``) so ``_num_kv_heads`` can resolve
    the list to the per-layer scalar and let the base callback run unchanged.
    """

    def __init__(self):
        super().__init__()
        self._current_layer_idx = None

    def apply_callbacks(
        self, module: nn.Module, module_name: str, module_names_breakdown: list[str], weights: dict
    ) -> list[dict]:
        self._current_layer_idx = self._extract_layer_idx(module_names_breakdown)
        try:
            return super().apply_callbacks(module, module_name, module_names_breakdown, weights)
        finally:
            self._current_layer_idx = None

    @property
    def _num_kv_heads(self):
        num_kv_heads = super()._num_kv_heads
        if isinstance(num_kv_heads, (list, tuple)) and self._current_layer_idx is not None:
            return num_kv_heads[self._current_layer_idx]
        return num_kv_heads

    @staticmethod
    def _extract_layer_idx(module_names_breakdown: list[str]):
        try:
            idx = module_names_breakdown.index("layers")
            return int(module_names_breakdown[idx + 1])
        except (ValueError, IndexError):
            return None
