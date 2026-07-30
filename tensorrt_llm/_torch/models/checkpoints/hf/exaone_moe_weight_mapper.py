from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.moe.fused_moe.weight_owner import is_moe_weight_owner


# K-EXAONE2 spells the architecture `ExaoneMoeForCausalLM`; earlier EXAONE-MoE
# checkpoints use `ExaoneMoEForCausalLM`. Both share this weight layout, and
# without both registrations the K-EXAONE2 lookup silently falls back to the
# generic HfWeightMapper, which has no `params_map` and dies during loading.
@register_mapper("HF", "ExaoneMoeForCausalLM")
@register_mapper("HF", "ExaoneMoEForCausalLM")
class ExaoneMoeWeightMapper(HfWeightMapper):
    def __init__(self):
        super().__init__()

        # MoE expert weights: gate_proj->w1, up_proj->w3, down_proj->w2
        # (used for per-expert checkpoint layouts; upstream HF transformers
        # >=5.3 stores fused 3D tensors instead, handled below.)
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

        def rename(old_name: str, new_name: str) -> None:
            # `weights` is a ConsumableWeightsDict, which deliberately exposes
            # only __getitem__/__setitem__/__delitem__ (each lock-guarded) and
            # has no `pop`. Reading does not consume, so get-then-delete is the
            # exact equivalent of pop here.
            weights[new_name] = weights[old_name]
            del weights[old_name]

        for name in list(weights.keys()):
            if name.startswith("mtp.layers."):
                # mtp.layers.{idx}.* -> model.layers.{offset + idx}.*
                _, _, mtp_layer_idx, module_name = name.split(".", 3)
                rename(
                    name,
                    f"model.layers.{mtp_layer_offset + int(mtp_layer_idx)}.{module_name}",
                )
            elif name.startswith("mtp."):
                # mtp.fc.* -> model.layers.{offset}.eh_proj.*
                # mtp.norm.* -> model.layers.{offset}.shared_head.norm.*
                # etc.
                for mtp_prefix, trtllm_name in self.mtp_mapping.items():
                    if name.startswith(mtp_prefix):
                        suffix = name[len(mtp_prefix) :]
                        rename(
                            name,
                            f"model.layers.{mtp_layer_offset}.{trtllm_name}{suffix}",
                        )
                        break

    def is_special_instance_module(self, module: nn.Module) -> bool:
        return is_moe_weight_owner(module)

    def _renames_scales_to_inv(self) -> bool:
        """Whether this checkpoint's MoE scales must be exposed as ``weight_scale_inv``.

        True for everything except NVFP4, whose loader
        (:meth:`NVFP4FusedMoEMethod.load_quant_scales`) indexes ``weight_scale``
        and ``weight_scale_2`` directly. Unknown/absent quant config keeps the
        historical behaviour so non-quantised checkpoints are unaffected.
        """
        from tensorrt_llm.quantization.mode import QuantAlgo

        quant_config = getattr(self.config, "quant_config", None)
        return getattr(quant_config, "quant_algo", None) != QuantAlgo.NVFP4

    def handle_special_instance_module(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: dict,
        allow_partial_loading: bool = False,
    ) -> None:
        if is_moe_weight_owner(module):
            config = self.config.pretrained_config
            updated_module_weights = {}
            for weight_name, weight_value in module_weights.items():
                # Upstream HF ExaoneMoeExperts (transformers >=5.3) stores fused
                # 3D tensors: gate_up_proj [E, 2*I, H], down_proj [E, H, I].
                # Expand into per-expert views (zero-copy) so VANILLA loading
                # can handle them without the peak GPU memory of a full transpose.
                if weight_name == "gate_up_proj" and weight_value.ndim == 3:
                    if weight_value.shape[-2] == 2 * config.moe_intermediate_size and (
                        weight_value.shape[-1] == config.hidden_size
                    ):
                        half = weight_value.shape[-2] // 2
                        for i in range(weight_value.shape[0]):
                            updated_module_weights[f"{i}.w1.weight"] = weight_value[i, :half, :]
                            updated_module_weights[f"{i}.w3.weight"] = weight_value[i, half:, :]
                        continue
                elif weight_name == "down_proj" and weight_value.ndim == 3:
                    if weight_value.shape[-2] == config.hidden_size and (
                        weight_value.shape[-1] == config.moe_intermediate_size
                    ):
                        for i in range(weight_value.shape[0]):
                            updated_module_weights[f"{i}.w2.weight"] = weight_value[i]
                        continue
                # Only the FP8-block-scale and W4A16-MXFP4 MoE loaders read
                # `weight_scale_inv`. The NVFP4 loader reads `weight_scale` (the
                # per-block scales) and `weight_scale_2` (the per-tensor global
                # scale) verbatim, so renaming here hides both from it.
                #
                # The old unconditional `replace("weight_scale", ...)` also
                # rewrote `weight_scale_2` into `weight_scale_inv_2`, which no
                # loader has ever looked for -- wrong under any quantisation.
                new_weight_name = weight_name
                if self._renames_scales_to_inv() and (
                    weight_name == "weight_scale" or weight_name.endswith(".weight_scale")
                ):
                    new_weight_name = weight_name[: -len("weight_scale")] + "weight_scale_inv"
                    weight_value = weight_value.squeeze()
                updated_module_weights[new_weight_name] = weight_value
            module.load_weights(
                weights=[updated_module_weights], allow_partial_loading=allow_partial_loading
            )
