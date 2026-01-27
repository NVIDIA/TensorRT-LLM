from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.fused_moe.interface import MoE


@register_mapper("HF", "ExaoneMoEForCausalLM")
class ExaoneMoeWeightMapper(HfWeightMapper):
    def __init__(self):
        super().__init__()

        # MoE expert weights: gate_proj->w1, up_proj->w3, down_proj->w2
        # e_score_correction_bias: move into gate module
        self.params_map = {
            r"(.*experts\.\d+\.)gate_proj(.*)": r"\1w1\2",
            r"(.*experts\.\d+\.)up_proj(.*)": r"\1w3\2",
            r"(.*experts\.\d+\.)down_proj(.*)": r"\1w2\2",
            r"(.*)mlp\.e_score_correction_bias(.*)": r"\1mlp.gate.e_score_correction_bias\2",
        }
        self.mtp_mapping = {
            "mtp.fc": "eh_proj",
            "mtp.norm": "shared_head.norm",
            "mtp.pre_fc_norm_embedding": "enorm",
            "mtp.pre_fc_norm_hidden": "hnorm",
        }

    def preprocess_weights(self, weights: dict):
        mtp_layer_offset = self.config.pretrained_config.num_hidden_layers

        for name in weights.keys():
            if name.startswith("mtp.layers."):
                # mtp.layers.{idx}.* -> model.layers.{offset + idx}.*
                _, _, mtp_layer_idx, module_name = name.split(".", 3)
                new_name = f"model.layers.{mtp_layer_offset + int(mtp_layer_idx)}.{module_name}"
                weights[new_name] = weights.pop(name)
            elif name.startswith("mtp."):
                # mtp.fc.* -> model.layers.{offset}.eh_proj.*
                # mtp.norm.* -> model.layers.{offset}.shared_head.norm.*
                # etc.
                for mtp_prefix, trtllm_name in self.mtp_mapping.items():
                    if name.startswith(mtp_prefix):
                        suffix = name[len(mtp_prefix) :]
                        new_name = f"model.layers.{mtp_layer_offset}.{trtllm_name}{suffix}"
                        weights[new_name] = weights.pop(name)
                        break

    def is_special_instance_module(self, module: nn.Module) -> bool:
        return isinstance(module, MoE)

    def handle_special_instance_module(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: dict,
        allow_partial_loading: bool = False,
    ) -> None:
        if isinstance(module, MoE):
            updated_module_weights = {}
            for weight_name, weight_value in module_weights.items():
                new_weight_name = weight_name.replace("weight_scale", "weight_scale_inv")
                if new_weight_name.endswith(".weight_scale_inv"):
                    weight_value = weight_value.squeeze()
                updated_module_weights[new_weight_name] = weight_value
            module.load_weights(
                weights=[updated_module_weights], allow_partial_loading=allow_partial_loading
            )
