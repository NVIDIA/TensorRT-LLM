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

        if "q_proj" in key:  #128k
            q_param = value
            k_param = hf_state_dict[orig_key.replace("q_proj", "k_proj")]
            v_param = hf_state_dict[orig_key.replace("q_proj", "v_proj")]
            value = torch.cat([q_param, k_param, v_param], dim=0)
            key = key.replace("q_proj.weight", "qkv.weight")
        elif "k_proj" in key or "v_proj" in key:
            continue

        weights[key] = value.to(torch_dtype).cpu()

    if config.architecture == 'Phi3SmallForCausalLM':
        weights['lm_head.weight'] = weights[
            'transformer.vocab_embedding.weight'].clone()

        # Transform QKV weights from custom Phi3Small format to TRT-LLM format
        for key, value in weights.items():
            if "qkv." in key:
                weights[key] = shuffle_qkv_weights(weights[key], config)

        weights = split_weights_tp(config, weights, torch_dtype)

    return weights
