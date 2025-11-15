from typing import Dict, List, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.fused_moe import MoE  # matches GPT-OSS MoE


@register_mapper("HF", "GptOssForCausalLM")
class GptOssHfWeightMapper(HfWeightMapper):

    def __init__(self):
        super().__init__()
        self._trt_to_hf_map: List[Tuple[str, str]] = [
            ("embedding", "embed_tokens"),
            ("attn.norm", "input_layernorm"),
            ("attn", "self_attn"),
            ("mlp.norm", "post_attention_layernorm"),
            ("block", "layers"),
            ("gate", "router"),
        ]

    def _trt_prefix_to_hf_prefix(self, prefix: str) -> str:
        mapped = prefix
        for k, v in self._trt_to_hf_map:
            mapped = mapped.replace(k, v)
        return mapped

    def filter_weights(self, prefix: str, weights: Dict) -> Dict:
        hf_prefix = self._trt_prefix_to_hf_prefix(prefix)
        return super().filter_weights(hf_prefix, weights)

    def is_special_instance_module(self, module: nn.Module) -> bool:
        return isinstance(module, MoE)

    def handle_special_instance_module(self, module: nn.Module,
                                       module_name: str,
                                       module_weights: dict) -> None:
        # MoE needs to be handled specially
        try:
            # BF16 layout
            gate_up_weight = module_weights['gate_up_proj']
            gate, up = gate_up_weight[:, :, ::2], gate_up_weight[:, :, 1::2]
            fused_gate_up = torch.cat([gate, up], dim=-1)

            gate_up_bias = module_weights['gate_up_proj_bias']
            gate_b, up_b = gate_up_bias[:, ::2], gate_up_bias[:, 1::2]
            fused_b = torch.cat([gate_b, up_b], dim=-1)

            down = module_weights['down_proj']
            down_b = module_weights['down_proj_bias']

            moe_weights = {
                'gate_up_proj':
                [fused_gate_up[i, :, :] for i in range(fused_gate_up.shape[0])],
                'down_proj': [down[i, :, :] for i in range(down.shape[0])],
                'gate_up_proj.bias':
                [fused_b[i, :] for i in range(fused_b.shape[0])],
                'down_proj.bias':
                [down_b[i, :] for i in range(down_b.shape[0])],
            }
        except Exception:
            # MXFP4 blocks + scales
            gate_up_weight = module_weights['gate_up_proj_blocks'].flatten(
                -2, -1)
            gate_w, up_w = gate_up_weight[:, ::2, :], gate_up_weight[:, 1::2, :]
            fused_gate_up = torch.cat([gate_w, up_w], dim=-2)

            gate_up_bias = module_weights['gate_up_proj_bias']
            gate_b, up_b = gate_up_bias[:, ::2], gate_up_bias[:, 1::2]
            fused_b = torch.cat([gate_b, up_b], dim=-1)

            down_blocks = module_weights['down_proj_blocks'].flatten(-2, -1)

            moe_weights = {
                'gate_up_proj': [
                    fused_gate_up[i, :, :].transpose(0, 1)
                    for i in range(fused_gate_up.shape[0])
                ],
                'down_proj': [
                    down_blocks[i, :, :].transpose(0, 1)
                    for i in range(down_blocks.shape[0])
                ],
                'gate_up_proj.bias':
                [fused_b[i, :] for i in range(fused_b.shape[0])],
                'down_proj.bias': [
                    module_weights['down_proj_bias'][i, :]
                    for i in range(module_weights['down_proj_bias'].shape[0])
                ],
            }
            if 'gate_up_proj_scales' in module_weights and 'down_proj_scales' in module_weights:
                gate_up_sc = module_weights['gate_up_proj_scales']
                gate_w_sc, up_w_sc = gate_up_sc[:, ::2, :], gate_up_sc[:,
                                                                       1::2, :]
                moe_weights['gate_up_proj_weight_scale'] = [
                    torch.cat([gate_w_sc[i, :, :], up_w_sc[i, :, :]],
                              dim=-2).transpose(0, 1)
                    for i in range(gate_w_sc.shape[0])
                ]
                moe_weights['down_proj_weight_scale'] = [
                    module_weights['down_proj_scales'][i, :, :].transpose(0, 1)
                    for i in range(module_weights['down_proj_scales'].shape[0])
                ]

        module.load_weights(weights=[moe_weights])
