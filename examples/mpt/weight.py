import configparser
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_np
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
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx].copy())
    return None


def parse_ft_config(ini_file):
    gpt_config = configparser.ConfigParser()
    gpt_config.read(ini_file)

    n_embd = gpt_config.getint('gpt', 'n_embd')
    n_head = gpt_config.getint('gpt', 'n_head')
    n_layer = gpt_config.getint('gpt', 'n_layer')
    n_positions = gpt_config.getint('gpt', 'n_positions')
    vocab_size = gpt_config.getint('gpt', 'vocab_size')
    do_layer_norm_before = gpt_config.getboolean('gpt',
                                                 'do_layer_norm_before',
                                                 fallback=True)
    rotary_pct = gpt_config.getfloat('gpt', 'rotary_pct', fallback=0.0)
    hidden_act = gpt_config.get('gpt', 'activation_function')
    bias = gpt_config.getboolean('gpt', 'bias', fallback=True)
    inter_size = gpt_config.getint('gpt', 'intermediate_size', fallback=None)
    dtype = gpt_config.get('gpt', 'storage_dtype', fallback='float32')

    if inter_size is None:
        inter_size = 4 * n_embd
    n_kv_head = gpt_config.getint('gpt', 'n_kv_head', fallback=None)

    multi_query_mode = gpt_config.getboolean('gpt',
                                             'multi_query_mode',
                                             fallback=False)
    assert not (multi_query_mode and n_kv_head and n_kv_head != 1), \
        "if multi_query_mode is enabled, n_kv_head must be 1 or unset"
    if multi_query_mode:
        n_kv_head = 1
    prompt_num_tasks = gpt_config.getint('gpt', 'prompt_num_tasks', fallback=0)
    prompt_max_vocab_size = gpt_config.getint('gpt',
                                              'prompt_max_vocab_size',
                                              fallback=0)
    pos_embedding_type = gpt_config.get('gpt',
                                        'position_embedding_type',
                                        fallback='alibi')
    return n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before, hidden_act, rotary_pct, bias, inter_size, n_kv_head, dtype, prompt_num_tasks, prompt_max_vocab_size, pos_embedding_type


def check_embedding_share(dir_path):
    share_embedding_table = False
    lm_file = dir_path + '/' + 'model.lm_head.weight.bin'
    if not Path(lm_file).exists():
        share_embedding_table = True
    return share_embedding_table


def load_from_ft(tensorrt_llm_gpt: GPTLMHeadModel,
                 dir_path,
                 rank=0,
                 tensor_parallel=1,
                 dtype='float32',
                 use_parallel_embedding=False,
                 sharding_dim=0,
                 share_embedding_table=False):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_gpt, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before, hidden_act, rotary_pct, bias, inter_size, n_kv_head, *_ = parse_ft_config(
        Path(dir_path) / 'config.ini')
    np_dtype = str_dtype_to_np(dtype)

    def fromfile(dir_path, name, shape=None, dtype=None):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def set_smoothquant_scale_factors(module,
                                      pre_scale_weight,
                                      dir_path,
                                      basename,
                                      shape,
                                      per_tok_dyn,
                                      per_channel,
                                      is_qkv=False,
                                      rank=None):
        suffix = "bin"
        if per_channel:
            if rank is not None:
                suffix = f"{rank}." + suffix
            suffix = "col." + suffix

        col_shape = shape if (per_channel or is_qkv) else [1, 1]
        if per_tok_dyn:
            if pre_scale_weight is not None:
                pre_scale_weight.value = np.array([1.0], dtype=np.float32)
            t = fromfile(dir_path, f"{basename}scale_w_quant_orig.{suffix}",
                         col_shape, np.float32)
            module.per_channel_scale.value = t
        else:
            t = fromfile(dir_path, f"{basename}scale_x_orig_quant.bin", [1],
                         np.float32)
            pre_scale_weight.value = t
            t = fromfile(dir_path, f"{basename}scale_y_accum_quant.{suffix}",
                         col_shape, np.float32)
            module.per_channel_scale.value = t
            t = fromfile(dir_path, f"{basename}scale_y_quant_orig.bin", [1, 1],
                         np.float32)
            module.act_scale.value = t

    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_gpt, "quant_mode", QuantMode(0))
    # Do we use SmoothQuant?
    use_smooth_quant = quant_mode.has_act_and_weight_quant()
    # Do we use quantization per token?
    quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    # Do we use quantization per channel?
    quant_per_channel = quant_mode.has_per_channel_scaling()

    # Do we use INT4/INT8 weight-only?
    use_weight_only = quant_mode.is_weight_only()

    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    # Debug
    suffix = gen_suffix(rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    pe = fromfile(dir_path, 'model.wpe.bin', [n_positions, n_embd])
    if pe is not None:
        tensorrt_llm_gpt.embedding.position_embedding.weight.value = (pe)

    vocab_embedding_weight = fromfile(dir_path, 'model.wte.bin',
                                      [vocab_size, n_embd])
    if not use_parallel_embedding:
        tensorrt_llm_gpt.embedding.vocab_embedding.weight.value = vocab_embedding_weight
    else:
        if sharding_dim == 0:
            if vocab_size % tensor_parallel != 0:
                # padding
                vocab_size_padded = pad_vocab_size(
                    tensorrt_llm_gpt.embedding.vocab_embedding.num_embeddings,
                    tensor_parallel)
                pad_width = vocab_size_padded - vocab_size
                vocab_embedding_weight = np.pad(vocab_embedding_weight,
                                                ((0, pad_width), (0, 0)),
                                                'constant',
                                                constant_values=0)
        tensorrt_llm_gpt.embedding.vocab_embedding.weight.value = np.ascontiguousarray(
            split(vocab_embedding_weight,
                  tensor_parallel,
                  rank,
                  dim=sharding_dim))

    if do_layer_norm_before:
        tensorrt_llm_gpt.ln_f.bias.value = (fromfile(
            dir_path, 'model.final_layernorm.bias.bin'))
        tensorrt_llm_gpt.ln_f.weight.value = (fromfile(
            dir_path, 'model.final_layernorm.weight.bin'))

    # share input embedding
    if not share_embedding_table:
        lm_head_weight = fromfile(dir_path, 'model.lm_head.weight.bin',
                                  [vocab_size, n_embd])
        if lm_head_weight is None:
            lm_head_weight = fromfile(dir_path, 'model.wte.bin',
                                      [vocab_size, n_embd])
        if vocab_size % tensor_parallel != 0:
            # padding
            vocab_size_padded = tensorrt_llm_gpt.lm_head.out_features * tensor_parallel
            pad_width = vocab_size_padded - vocab_size
            lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                    'constant',
                                    constant_values=0)
        tensorrt_llm_gpt.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, tensor_parallel, rank))
    for i in range(n_layer):
        head_dim = n_embd // n_head
        if n_kv_head == 1:
            # multi-query attention.
            c_attn_out_dim = (n_embd // tensor_parallel) + (head_dim * 2)
        elif n_kv_head:
            # grouped-query attention.
            c_attn_out_dim = (n_embd // tensor_parallel +
                              (head_dim * n_kv_head * 2) // tensor_parallel)
        else:
            # multi-head attention.
            c_attn_out_dim = 3 * n_embd // tensor_parallel
        tensorrt_llm_gpt.layers[i].input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        tensorrt_llm_gpt.layers[i].input_layernorm.bias.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.bias.bin'))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + suffix,
            [n_embd, c_attn_out_dim], w_type)
        if t is not None:
            dst = tensorrt_llm_gpt.layers[i].attention.qkv.weight
            if use_smooth_quant:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
                set_smoothquant_scale_factors(
                    tensorrt_llm_gpt.layers[i].attention.qkv,
                    tensorrt_llm_gpt.layers[i].input_layernorm.scale_to_int,
                    dir_path,
                    'model.layers.' + str(i) + '.attention.query_key_value.',
                    [1, c_attn_out_dim],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=rank,
                    is_qkv=True)
            elif use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(t), plugin_weight_only_quant_type)
                dst.value = processed_torch_weights.numpy()
                scales = tensorrt_llm_gpt.layers[
                    i].attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
        if bias:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.bias.' + str(rank) + '.bin')
            if t is not None:
                dst = tensorrt_llm_gpt.layers[i].attention.qkv.bias
                dst.value = np.ascontiguousarray(t)

        dst = tensorrt_llm_gpt.layers[i].attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [n_embd // tensor_parallel, n_embd], w_type)
        if use_smooth_quant:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
            dense_scale = getattr(tensorrt_llm_gpt.layers[i].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_gpt.layers[i].attention.dense, dense_scale,
                dir_path, 'model.layers.' + str(i) + '.attention.dense.',
                [1, n_embd], quant_per_token_dyn, quant_per_channel)
            # change it to the real smoother if dense layer is applied smooth quant
            tensorrt_llm_gpt.layers[i].attention.dense.smoother.value = np.ones(
                [1, n_embd // tensor_parallel], dtype=np.float32)
        elif use_weight_only:
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gpt.layers[
                i].attention.dense.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        if bias:
            dst = tensorrt_llm_gpt.layers[i].attention.dense.bias
            dst.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.attention.dense.bias.bin')

        dst = tensorrt_llm_gpt.layers[i].post_layernorm.weight
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.weight.bin')

        dst = tensorrt_llm_gpt.layers[i].post_layernorm.bias
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.bias.bin')
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.dense_h_to_4h.weight.' + suffix,
            [n_embd, inter_size // tensor_parallel], w_type)
        if use_smooth_quant:
            tensorrt_llm_gpt.layers[
                i].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            set_smoothquant_scale_factors(
                tensorrt_llm_gpt.layers[i].mlp.fc,
                tensorrt_llm_gpt.layers[i].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_h_to_4h.',
                [1, inter_size // tensor_parallel],
                quant_per_token_dyn,
                quant_per_channel,
                rank=rank)
        elif use_weight_only:
            dst = tensorrt_llm_gpt.layers[i].mlp.fc.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gpt.layers[i].mlp.fc.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_gpt.layers[
                i].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
        if bias:
            tensorrt_llm_gpt.layers[i].mlp.fc.bias.value = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.mlp.dense_h_to_4h.bias.' + str(rank) + '.bin')
        if is_gated_activation(hidden_act):
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.mlp.dense_h_to_4h.gate.weight.' + str(rank) + '.bin',
                [n_embd, inter_size // tensor_parallel])
            tensorrt_llm_gpt.layers[
                i].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.mlp.dense_4h_to_h.weight.' + suffix,
            [inter_size // tensor_parallel, n_embd], w_type)
        if use_smooth_quant:
            tensorrt_llm_gpt.layers[
                i].mlp.proj.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))
            proj_scale = getattr(tensorrt_llm_gpt.layers[i].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_gpt.layers[i].mlp.proj, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            # change it to the real smoother if proj layer is applied smooth quant
            tensorrt_llm_gpt.layers[i].mlp.proj.smoother.value = np.ones(
                [1, inter_size // tensor_parallel], dtype=np.float32)
        elif use_weight_only:
            dst = tensorrt_llm_gpt.layers[i].mlp.proj.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            dst.value = processed_torch_weights.numpy()
            scales = tensorrt_llm_gpt.layers[i].mlp.proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_gpt.layers[i].mlp.proj.weight.value = (
                np.ascontiguousarray(np.transpose(t, [1, 0])))
        if bias:
            tensorrt_llm_gpt.layers[i].mlp.proj.bias.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.bias.bin')

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            tensorrt_llm_gpt.layers[
                i].attention.kv_orig_quant_scale.value = 1.0 / t
            tensorrt_llm_gpt.layers[i].attention.kv_quant_orig_scale.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
