from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "DeciLMForCausalLM")
class NemotronNasHfWeightMapper(HfWeightMapper):
    """Weight mapper for Nemotron-NAS / DeciLM.

    Nemotron-NAS sets ``config.num_key_value_heads`` to a *per-layer* list
    (see ``NemotronNASModel.__init__``), because layers can have differing
    KV-head counts (and no-op attention layers have 0). The base
    ``HfWeightMapper._duplicate_kv_weights`` assumes a single int, so this
    override resolves the layer index from the weight path and uses the
    per-layer KV-head count for k_proj / v_proj duplication.
    """

    def apply_callbacks(
        self, module: nn.Module, module_name: str, module_names_breakdown: list[str], weights: dict
    ) -> list[dict]:
        num_kv_heads = self._num_kv_heads
        if not isinstance(num_kv_heads, (list, tuple)):
            return super().apply_callbacks(module, module_name, module_names_breakdown, weights)

        layer_idx = self._layer_idx_from_breakdown(module_names_breakdown)
        assert layer_idx is not None, (
            "Nemotron-NAS uses per-layer num_key_value_heads but layer index "
            f"could not be inferred from {module_names_breakdown}"
        )
        layer_num_kv_heads = num_kv_heads[layer_idx]

        module_weights = []
        for new_name in self._mapping[module_name]:
            fw = self.filter_weights(".".join(module_names_breakdown + [new_name]), weights)
            fw = self._duplicate_kv_weights_for_layer(module, new_name, fw, layer_num_kv_heads)
            module_weights.append(fw)
        return module_weights

    @staticmethod
    def _layer_idx_from_breakdown(module_names_breakdown: list[str]):
        for i in range(len(module_names_breakdown) - 1):
            if module_names_breakdown[i] == "layers" and module_names_breakdown[i + 1].isdigit():
                return int(module_names_breakdown[i + 1])
        return None

    def _duplicate_kv_weights_for_layer(
        self, module: nn.Module, new_name: str, weights: dict, num_kv_heads: int
    ):
        if new_name not in ["k_proj", "v_proj"]:
            return weights

        duplicated_keys = ["weight", "bias"]
        if module.quant_config.quant_mode.has_nvfp4():
            duplicated_keys.append("weight_scale")

        return {
            k: self._duplicate_kv(
                weight=v[:], num_kv_heads=num_kv_heads, tensor_parallel_size=self._tp_size
            )
            if k in duplicated_keys
            else v
            for k, v in weights.items()
        }
