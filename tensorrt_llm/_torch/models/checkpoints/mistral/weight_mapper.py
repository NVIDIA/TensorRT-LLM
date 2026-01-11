from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("mistral", "MistralForCausalLM")
@register_mapper("mistral", "PixtralForConditionalGeneration")
class MistralWeightMapper(HfWeightMapper):
    def __init__(self):
        super().__init__()

        self._callbacks.append(self._permute_qk)

        self.pixtral_mapping = {
            "wq": "q_proj",
            "wk": "k_proj",
            "wv": "v_proj",
            "wo": "o_proj",
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
            "w_in": "linear_1",
            "w_out": "linear_2",
        }

        self.mistral_llm_mapping = {
            "layers": "model.layers",
            "attention": "self_attn",
            "qscale_act": "input_scale",
            "qscale_weight": "weight_scale_inv",
            "kv_fake_quantizer.qscale_act": "kv_scale",
            "q_fake_quantizer.qscale_act": "attn.q_scale",
            "k_fake_quantizer.qscale_act": "k_scale",
            "v_fake_quantizer.qscale_act": "v_scale",
            "attention_norm": "input_layernorm",
            "feed_forward": "mlp",
            "ffn_norm": "post_attention_layernorm",
            "tok_embeddings": "model.embed_tokens",
            "output": "lm_head",
            "norm": "model.norm",
            # For Eagle3
            "language_model.eagle_linear": "model.fc",
            "language_model.layers": "layers",
            "language_model.norm": "norm",
        }
        self.mistral_llm_mapping.update(self.pixtral_mapping)

    # Adapted from:
    # https://github.com/vllm-project/vllm/blob/883b42896a9ed9791750d721fad26005b7569eba/vllm/model_executor/models/llama.py#L657
    def rename_by_params_map(self, params_map: dict[str, str], weights: dict) -> dict:
        renamed_weights = {}

        for key in list(weights.keys()):
            new_key = key
            modules = key.split(".")
            num_modules = len(modules)
            for i in range(num_modules):
                item = modules[i]
                next_item = modules[i + 1] if i < num_modules - 1 else None

                combined_item = f"{item}.{next_item}" if next_item is not None else None

                if combined_item in params_map:
                    new_key = new_key.replace(combined_item, params_map[combined_item])
                elif item in params_map:
                    new_key = new_key.replace(item, params_map[item])

            renamed_weights[new_key] = weights[key]

        return renamed_weights

    def _permute_qk(self, module: nn.Module, new_name: str, weights: dict):
        # Adapted from:
        # https://github.com/vllm-project/vllm/blob/883b42896a9ed9791750d721fad26005b7569eba/vllm/model_executor/models/llama.py#L657

        processed_weights = {}
        config = self.config.pretrained_config

        def permute(w, n_heads: int, attn_out: int):
            attn_in = config.head_dim * n_heads

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        # rotary embeds should be sliced
        # If using quantized model in mistral format,
        # quantization scales (qscale_weight) also need to be sliced

        if new_name in ["k_proj", "q_proj"]:
            n_heads = (
                config.num_key_value_heads if new_name == "k_proj" else config.num_attention_heads
            )

            processed_weights["weight"] = permute(weights["weight"], n_heads, config.hidden_size)

            if "qscale_weight" in weights and weights["qscale_weight"].numel() > 1:
                processed_weights["qscale_weight"] = permute(weights["qscale_weight"], n_heads, 1)

            return processed_weights

        return weights


@register_mapper("mistral_large_3")
@register_mapper("mistral_large_3", "PixtralForConditionalGeneration")
@register_mapper("mistral_large_3", "MistralLarge3ForCausalLM")
@register_mapper("mistral", "MistralLarge3ForCausalLM")
class MistralLarge3WeightMapper(MistralWeightMapper):
    def __init__(self):
        super().__init__()

        self.mistral_llm_mapping.update(
            {
                "wkv_a_with_mqa": "kv_a_proj_with_mqa",
                "wkv_b": "kv_b_proj",
                "wq_a": "q_a_proj",
                "q_a_norm": "q_a_layernorm",
                "wq_b": "q_b_proj",
                "kv_a_norm": "kv_a_layernorm",
                "k_fake_quantizer.qscale_act": "mla_attn.mla_attn.k_scale",
                "q_fake_quantizer.qscale_act": "mla_attn.mla_attn.q_scale",
                "v_fake_quantizer.qscale_act": "mla_attn.mla_attn.v_scale",
                "gate": "mlp.gate",
                "shared_experts": "mlp.shared_experts",
                "experts": "mlp.experts",
                "router_biases": "mlp.gate.e_score_correction_bias",
            }
        )
