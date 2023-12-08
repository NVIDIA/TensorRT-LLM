import configparser
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

import tensorrt_llm
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_np, str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.functional import is_gated_activation
from tensorrt_llm.models import GPTLMHeadModel
from tensorrt_llm.models.quantized.quant import get_dummy_quant_scales
from tensorrt_llm.quantization import QuantMode


def get_scaling_factors(
    model_path: Union[str, Path],
    num_layers: int,
    quant_mode: Optional[QuantMode] = None,
) -> Optional[Dict[str, List[int]]]:
    """ Get the scaling factors for MPT model

    Returns a dictionary of scaling factors for the selected layers of the
    MPT model.

    Args:
        model_path (str): Path to the quantized MPT model
        layers (list): List of layers to get the scaling factors for. If None,
            all layers are selected.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the
        LLaMA model.

        example:

        {
            'qkv_act': qkv_act_scale,
            'qkv_weights': qkv_weights_scale,
            'qkv_output' : qkv_outputs_scale,
            'dense_act': dense_act_scale,
            'dense_weights': dense_weights_scale,
            'fc_act': fc_act_scale,
            'fc_weights': fc_weights_scale,
            'proj_act': proj_act_scale,
            'proj_weights': proj_weights_scale,
        }
    """

    if model_path is None:
        logger.warning(f"--quantized_fp8_model_path not specified. "
                       f"Initialize quantization scales automatically.")
        return get_dummy_quant_scales(num_layers)
    weight_dict = np.load(model_path)

    # yapf: disable
    scaling_factor = {
        'qkv_act': [],
        'qkv_weights': [],
        'qkv_output': [],
        'dense_act': [],
        'dense_weights': [],
        'fc_act': [],
        'fc_weights': [],
        'proj_act': [],
        'proj_weights': [],
    }

    for layer in range(num_layers):
        scaling_factor['qkv_act'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:activation_scaling_factor'].item()
            ))
        scaling_factor['qkv_weights'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:weights_scaling_factor'].item()
            ))
        if quant_mode is not None and quant_mode.has_fp8_kv_cache():
            # Not calibrarting KV cache.
            scaling_factor['qkv_output'].append(1.0)
        scaling_factor['dense_act'].append(weight_dict[f'_np:layers:{layer}:attention:dense:activation_scaling_factor'].item())
        scaling_factor['dense_weights'].append(weight_dict[f'_np:layers:{layer}:attention:dense:weights_scaling_factor'].item())
        scaling_factor['fc_act'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:activation_scaling_factor'].item())
        scaling_factor['fc_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:weights_scaling_factor'].item())
        scaling_factor['proj_act'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:activation_scaling_factor'].item())
        scaling_factor['proj_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:weights_scaling_factor'].item())
    # yapf: enable
    for k, v in scaling_factor.items():
        assert len(v) == num_layers, \
        f'Expect scaling factor {k} of length {num_layers}, got {len(v)}'

    return scaling_factor


def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx].copy())
    elif len(v.shape) >= 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx].copy())
    return None

def parse_hf_config(hf_config):
    n_embd = hf_config.d_model
    n_head = hf_config.n_heads
    n_layer = hf_config.n_layers
    n_positions = hf_config.max_seq_len
    vocab_size = hf_config.vocab_size
    do_layer_norm_before = True
    hidden_act = 'gelu'
    rotary_pct = 0.0
    bias = not hf_config.no_bias
    inter_size = int(hf_config.expansion_ratio * hf_config.d_model)
    if "kv_n_heads" in hf_config.attn_config:
        n_kv_head = hf_config.attn_config["kv_n_heads"]
    else:
        n_kv_head = n_head
    dtype = hf_config.torch_dtype
    prompt_num_tasks = 0
    prompt_max_vocab_size = 0
    pos_embedding_type = 'alibi'
    return n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before, hidden_act, rotary_pct, bias, inter_size, n_kv_head, dtype, prompt_num_tasks, prompt_max_vocab_size, pos_embedding_type

def load_from_hf(tensorrt_llm_gpt: GPTLMHeadModel,
                 model_dir,
                 tp_rank=0,
                 tp_size=1,
                 dtype='float32',
                 use_parallel_embedding=False,
                 sharding_dim=0,
                 share_embedding_table=False):
    tensorrt_llm.logger.info('Loading weights from HF...')

    torch_data_type = str_dtype_to_torch(dtype)
    hf_config = transformers.AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    
    # Uncomment below line to do conversion on gpu
    # hf_config.init_device = 'cuda'
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True,
        torch_dtype=torch_data_type, config=hf_config)

    n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before, hidden_act, rotary_pct, bias, inter_size, n_kv_head, *_ = parse_hf_config(hf_config)
    mha_mode = (n_kv_head == n_head)
    np_dtype = str_dtype_to_np(dtype)
    
    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_gpt, "quant_mode", QuantMode(0))
    
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
        
    # TO DO: SmoothQuant for MPT models; refer GPT example to write your own conversion
    # Do we use SmoothQuant?
    use_smooth_quant = quant_mode.has_act_and_weight_quant()
    # Do we use quantization per token?
    quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    # Do we use quantization per channel?
    quant_per_channel = quant_mode.has_per_channel_scaling()

    # Do we use INT4/INT8 weight-only?
    use_weight_only = quant_mode.is_weight_only()

    # TO DO: Int8 KV cache for MPT models; refer GPT example to write your own conversion
    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    # Debug
    suffix = gen_suffix(tp_rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    model_params = dict(model.named_parameters())
    
    tik = time.time()
    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_data_type).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_data_type).detach().cpu())
        if 'wpe.weight' in k:
            tensorrt_llm_gpt.embedding.position_embedding.weight.value = v
        if 'wte.weight' in k:
            if use_parallel_embedding:
                padded_v = v
                if sharding_dim == 0:
                    if vocab_size % tp_size != 0:
                        # padding
                        vocab_size_padded = pad_vocab_size(
                            tensorrt_llm_gpt.embedding.vocab_embedding.num_embeddings,
                            tp_size)
                        pad_width = vocab_size_padded - vocab_size
                        padded_v = np.pad(padded_v, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
                split_v = split(padded_v, tp_size, tp_rank, dim=sharding_dim)
            else:
                split_v = v
            tensorrt_llm_gpt.embedding.vocab_embedding.weight.value = np.ascontiguousarray(split_v)
            if vocab_size % tp_size != 0:
                # padding
                vocab_size_padded = tensorrt_llm_gpt.lm_head.out_features * tp_size
                pad_width = vocab_size_padded - vocab_size
                v = np.pad(v, ((0, pad_width), (0, 0)),
                                        'constant',
                                        constant_values=0)
            tensorrt_llm_gpt.lm_head.weight.value = np.ascontiguousarray(split(v, tp_size, tp_rank,
                          dim=0))
        elif 'norm_f.weight' in k:
            tensorrt_llm_gpt.ln_f.weight.value = np.ascontiguousarray(v)
            if not bias:
                dst = tensorrt_llm_gpt.ln_f.bias
                dst.value = np.ascontiguousarray(np.zeros(v.shape[-1], dtype=np_dtype))
        elif 'norm_f.bias' in k:
            tensorrt_llm_gpt.ln_f.bias.value = np.ascontiguousarray(v)
        elif ('lm_head.weight' in k) and (not share_embedding_table):
            if vocab_size % tp_size != 0:
                # padding
                vocab_size_padded = tensorrt_llm_gpt.lm_head.out_features * tp_size
                pad_width = vocab_size_padded - vocab_size
                v = np.pad(v, ((0, pad_width), (0, 0)),
                                        'constant',
                                        constant_values=0)
            tensorrt_llm_gpt.lm_head.weight.value = np.ascontiguousarray(split(v, tp_size, tp_rank))
                    
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_gpt._num_layers:
                continue
            if 'norm_1.weight' in k:
                tensorrt_llm_gpt.layers[idx].input_layernorm.weight.value = v
                if not bias:
                    dst = tensorrt_llm_gpt.layers[idx].input_layernorm.bias
                    dst.value = np.ascontiguousarray(np.zeros(v.shape[-1], dtype=np_dtype))
            elif 'norm_1.bias' in k:
                tensorrt_llm_gpt.layers[idx].input_layernorm.bias.value = v
            elif 'norm_2.weight' in k:
                dst = tensorrt_llm_gpt.layers[idx].post_layernorm.weight
                dst.value = v
                if not bias:
                    dst = tensorrt_llm_gpt.layers[idx].post_layernorm.bias
                    dst.value = np.ascontiguousarray(np.zeros(v.shape[-1], dtype=np_dtype))
            elif 'norm_2.bias' in k:
                tensorrt_llm_gpt.layers[idx].post_layernorm.bias.value = v
            elif 'attn.Wqkv.weight' in k:
                dst = tensorrt_llm_gpt.layers[idx].attention.qkv.weight
                if not mha_mode:
                    head_dim = n_embd // n_head
                    v = np.split(v, [n_embd, n_embd + (n_kv_head * head_dim)])
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(v[0], tp_size, tp_rank)
                    wk = split(v[1], tp_size, tp_rank)
                    wv = split(v[2], tp_size, tp_rank)
                    split_v = np.ascontiguousarray(np.concatenate((wq, wk, wv)))
                else:
                    q_emb = v.shape[0] // 3
                    model_emb = v.shape[1]
                    v = v.reshape(3, q_emb, model_emb)
                    split_v = split(v, tp_size, tp_rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // tp_size),
                                              model_emb)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_gpt.layers[
                        idx].attention.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'attn.out_proj.weight' in k:
                dst = tensorrt_llm_gpt.layers[idx].attention.dense.weight
                split_v = split(v, tp_size, tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_gpt.layers[
                        idx].attention.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'ffn.up_proj.weight' in k:
                dst = tensorrt_llm_gpt.layers[idx].mlp.fc.weight
                split_v = split(v, tp_size, tp_rank, dim=0)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(split_v), plugin_weight_only_quant_type)
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_gpt.layers[i].mlp.fc.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                dst.value = np.ascontiguousarray(split_v)
            elif 'ffn.down_proj.weight' in k:
                dst = tensorrt_llm_gpt.layers[idx].mlp.proj.weight
                split_v = split(v, tp_size, tp_rank, dim=1)
                if use_weight_only:
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(split_v), plugin_weight_only_quant_type)
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_gpt.layers[i].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                dst.value = np.ascontiguousarray(split_v)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return