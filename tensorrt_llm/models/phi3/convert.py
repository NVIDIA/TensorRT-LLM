import torch

from ..._utils import str_dtype_to_torch
from .split_weights import shuffle_qkv_weights, split_weights_tp


def load_weights_from_hf_model(hf_model, config):
    torch_dtype = str_dtype_to_torch(config.dtype)
    hf_state_dict = hf_model.state_dict()
    weights = {}

    config.quant_mode.is_weight_only()
    if config.quant_mode.is_int8_weight_only():
        torch.int8
    elif config.quant_mode.is_int4_weight_only():
        torch.quint4x2
    # replace key name
    for key, value in hf_state_dict.items():
        # Decoder Layers
        orig_key = key
        if "model.layers." in key:
            key = key.replace("model.layers.", "transformer.layers.")
            #Attention
            key = key.replace("self_attn.", "attention.")
            key = key.replace("query_key_value.", "qkv.")  # small
            key = key.replace("Wqkv.weight", "qkv.weight")
            key = key.replace("qkv_proj.", "qkv.")  #128k
            #MLP
            key = key.replace("mlp.fc1.", "mlp.fc.")
            key = key.replace("mlp.fc2.", "mlp.proj.")
            key = key.replace("mlp.gate_up_proj.", "mlp.fc.")
            key = key.replace("mlp.up_proj.", "mlp.fc." if config.architecture
                              == 'Phi3SmallForCausalLM' else "mlp.gate.")  #128k
            key = key.replace("mlp.down_proj.", "mlp.proj.")  #128k
            key = key.replace("mlp.gate_proj.", "mlp.fc.")  #128k
            key = key.replace("o_proj.", "dense.")  #128k
            # LoRA
            key = key.replace("base_layer.", "")

            #MoE
            key = key.replace("block_sparse_moe.gate", "mlp.router")
            key = key.replace("block_sparse_moe.experts.0.w3", "mlp.fc")
            key = key.replace("block_sparse_moe.experts.0.w2", "mlp.proj")

            #Layer norm
            key = key.replace("post_attention_layernorm.",
                              "post_layernorm.")  #128k

        # Embedding
        key = key.replace("model.embed_tokens.weight",
                          "transformer.vocab_embedding.weight")
        # Final Layer norm
        key = key.replace("model.final_layernorm.", "transformer.ln_f.")
        key = key.replace("model.norm.", "transformer.ln_f.")  #128k

        if "mlp.gate_up_proj." in orig_key:  #4k
            original_weights = value.contiguous().clone()
            half_split = original_weights.shape[0] // 2
            first_half, second_half = original_weights[:
                                                       half_split, :], original_weights[
                                                           half_split:, :]
            # Swap the halves
            value = torch.cat((second_half, first_half), dim=0)

        if config.architecture == "PhiMoEForCausalLM":
            num_experts = config.moe["num_experts"]
            mlp_hidden_size = config.intermediate_size
            num_hidden = config.hidden_size
            rank_experts = list(range(num_experts))
            if config.mapping.has_moe_ep():
                rank_experts = config.mapping.ep_experts(num_experts)

            def get_moe_weight(key, suffix):
                param = []
                for expert in rank_experts:
                    name = key.replace(f"0.{suffix}", f"{expert}.{suffix}")
                    fc_value = hf_state_dict[name]
                    param.append(fc_value)
                w = torch.stack(param)
                return w.reshape(-1, mlp_hidden_size, num_hidden)

            if ".0.w3" in orig_key:
                w3 = get_moe_weight(orig_key, 'w3')
                w1 = get_moe_weight(orig_key.replace("w3", "w1"), 'w1')
                value = torch.concat([w3, w1], dim=-2)
            elif ".0.w2" in orig_key:
                w2 = get_moe_weight(orig_key, 'w2')
                value = w2.reshape(-1, num_hidden, mlp_hidden_size)
            elif any([k in orig_key for k in ["w1", "w2", "w3"]]):
                continue

        if "q_proj" in key:  #128k
            q_param = value
            k_param = hf_state_dict[orig_key.replace("q_proj", "k_proj")]
            v_param = hf_state_dict[orig_key.replace("q_proj", "v_proj")]
            value = torch.cat([q_param, k_param, v_param], dim=0)
            key = key.replace("q_proj", "qkv")
        elif "k_proj" in key or "v_proj" in key:
            continue

        dtype = torch.float if "router" in key else torch_dtype
        weights[key] = value.to(dtype).cpu()

    #This is for InternVL-4B
    if config.architecture == 'Phi3ForCausalLM':
        keys_to_rename = [
            key for key in weights.keys() if 'language_model.' in key
        ]
        keys_to_delete = [
            key for key in weights.keys() if 'vision_model.' in key
        ]
        for key in keys_to_rename:
            keys_rename = key.replace('language_model.', '')
            weights[keys_rename] = weights[key]
            del weights[key]
        for key in keys_to_delete:
            del weights[key]

    if config.tie_word_embeddings or config.architecture == 'Phi3SmallForCausalLM':
        weights['lm_head.weight'] = weights[
            'transformer.vocab_embedding.weight'].clone()

        if config.architecture == 'Phi3SmallForCausalLM':
            # Transform QKV weights from custom Phi3Small format to TRT-LLM format
            for key, value in weights.items():
                if "qkv." in key:
                    weights[key] = shuffle_qkv_weights(weights[key], config)

    if config.architecture in [
            'Phi3SmallForCausalLM', "PhiMoEForCausalLM", "Phi3ForCausalLM"
    ] and config.mapping.has_tp():
        weights = split_weights_tp(config, weights, torch_dtype)

    return weights
