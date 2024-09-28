import fnmatch
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset

from ..quantization import QuantAlgo


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()


def split_qkv_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    v = v.reshape(3, n_hidden, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel), n_hidden)
    return split_v.contiguous()


def split_qkv_bias_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV bias according to tensor parallelism
    """
    v = v.reshape(3, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel))
    return split_v.contiguous()


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return split(v, tensor_parallel, rank, dim=dim)


def weight_only_quantize(weight: torch.Tensor,
                         quant_algo: str,
                         plugin: bool = True):
    assert quant_algo in [QuantAlgo.W4A16, QuantAlgo.W8A16
                          ], f'unsupported quant algo: {quant_algo}'
    if quant_algo == QuantAlgo.W4A16:
        assert plugin, 'W4A16 is only supported with plugin'
    if weight.dim() > 2:
        v = weight.transpose(-1, -2)
    else:
        v = weight.t()
    t = torch.quint4x2 if quant_algo == QuantAlgo.W4A16 else torch.int8
    processed_torch_weights, torch_weight_scales = \
        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
        v.contiguous(), t)
    if plugin:
        return processed_torch_weights, torch_weight_scales
    else:
        return v, torch_weight_scales


def get_weight(params: Dict[str, torch.Tensor], prefix: str,
               dtype: torch.dtype) -> torch.Tensor:
    if f'{prefix}.weight' not in params:
        return None
    return params[f'{prefix}.weight'].to(dtype).detach().cpu()


def get_bias(params: Dict[str, torch.Tensor], prefix: str,
             dtype: torch.dtype) -> torch.Tensor:
    if f'{prefix}.bias' not in params:
        return None
    return params[f'{prefix}.bias'].to(dtype).detach().cpu()


def get_weight_and_bias(params: Dict[str, torch.Tensor], prefix: str,
                        dtype: torch.dtype) -> Tuple[torch.Tensor]:
    return get_weight(params, prefix, dtype), get_bias(params, prefix, dtype)


def dup_kv_weight(v, num_head, tp_size):
    assert tp_size % num_head == 0
    reps = tp_size // num_head
    head_size = v.shape[0] // num_head
    v = v.reshape(num_head, head_size,
                  -1)[:, None, :, :].expand(num_head, reps, head_size,
                                            v.shape[1])
    return v.reshape(num_head * reps * head_size, -1).clone().detach()


def weight_only_quantize_dict(weights: Dict[str, torch.Tensor],
                              quant_algo: str,
                              quant_weights=[
                                  'qkv.weight', 'dense.weight', 'fc.weight',
                                  'proj.weight', 'gate.weight'
                              ],
                              exclude_modules=None,
                              plugin: bool = True):
    if quant_algo not in [QuantAlgo.W4A16, QuantAlgo.W8A16]:
        return weights
    if exclude_modules is None:
        exclude_modules = ['*shared_expert_gate.weight']
    for name in list(weights):
        is_excluded = False
        for exclude_module in exclude_modules:
            if fnmatch.fnmatchcase(name, exclude_module):
                is_excluded = True
                break
        if not is_excluded and any([_name in name for _name in quant_weights
                                    ]) and weights[name].dtype != torch.int8:
            quant_weight, quant_scale = weight_only_quantize(
                weight=weights[name], quant_algo=quant_algo, plugin=plugin)
            weights[name] = quant_weight
            weights[name.replace('.weight', '.per_channel_scale')] = quant_scale
    return weights


def load_state_dict(
    file_path: Union[str, Path],
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """ Load weights from model file.

    `safetensors` or `pytorch binary` is supported.
    Args:
        file_path: model file path, ends with .bin or .safetensors.
        dtype: torch.dtype, data type.
        device: torch device like, optional. If None, load to cpu.
    Returns:
        Weights as state dict.
    """
    file_path = Path(file_path)
    if dtype is not None:
        assert isinstance(dtype, torch.dtype)

    if device is None:
        device = 'cpu'

    model_params = {}
    if file_path.suffix == '.safetensors':
        # load from safetensors file
        from safetensors import safe_open
        with safe_open(file_path, framework='pt', device=device) as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if dtype is not None:
                    tensor = tensor.to(dtype)
                model_params[name] = tensor
    elif file_path.suffix == '.bin':
        # load from pytorch bin file
        state_dict = torch.load(file_path, map_location=device)
        for name in state_dict:
            tensor = state_dict[name]
            if dtype is not None:
                tensor = tensor.to(dtype)
            model_params[name] = tensor
    else:
        raise NotImplementedError(
            f'Support .safetensors or .bin files, but got {str(file_path)}')
    return model_params


def get_model_path(
    model_dir: Union[str, Path],
    name: Optional[str] = None,
) -> Optional[str]:
    """ Get model path from model directory.

    `safetensors` or `pytorch binary` is supported.
    Args:
        model_dir: model directory.
        name: model file name without suffix.
    Returns:
        Full model path.
    """
    model_dir = Path(model_dir)
    if name is not None:
        if (model_dir / f"{name}.safetensors").exists():
            return str(model_dir / f"{name}.safetensors")
        elif (model_dir / f"{name}.bin").exists():
            return str(model_dir / f"{name}.bin")
        else:
            return None
    else:
        model_files = list(model_dir.glob('*.safetensors'))
        if len(model_files) > 0:
            assert len(
                model_files
            ) == 1, f"find multiple safetensors files in {model_dir}, please specify one"
            return str(model_files[0])
        model_files = list(model_dir.glob('*.bin'))
        if len(model_files) > 0:
            assert len(
                model_files
            ) == 1, f"find multiple bin files in {model_dir}, please specify one"
            return str(model_files[0])
        return None


def retrieved_layer_index_from_name(name: str) -> Optional[int]:
    # This method is a hacky function to retrieve the layer index from
    # HF model. Most of HF models have similar naming convention but
    # please check carefully before applying if this method works well
    # on your target model.
    res = re.search(r'\d+', name)
    return int(res.group()) if res is not None else res


def iterate_shard_files(model_dir: Union[Path, str],
                        rank: int,
                        progress_bar: bool = True):
    model_dir = Path(model_dir)

    # '.bin' or '.safetensors'. In case that both exist, '.safetensor'
    # files will be loaded first.
    shard_files = list(model_dir.glob('*.safetensors'))
    if not shard_files:
        # The model checkpoint is stored in .bin file.
        shard_files = list(model_dir.glob('*.bin'))
    if not shard_files:
        raise RuntimeError(
            f"Could not find any .safetensors or .bin files in {model_dir}")

    try:
        import tqdm
        if progress_bar:
            # Show a progress bar per rank.
            desc = f'Rank [{rank}] Loading weights'
            shard_files = tqdm.tqdm(shard_files, desc=desc, position=rank)

    except ImportError:
        pass

    for shard_file in shard_files:
        yield shard_file


def has_safetensors(model_dir: str):
    return len(list(Path(model_dir).glob('*.safetensors'))) > 0


DEFAULT_HF_DATASET_META = {
    'ccdv/cnn_dailymail': ('3.0.0', 'train', 'article'),
    'cnn_dailymail': ('3.0.0', 'train', 'article'),
    'lambada': (None, 'validation', 'text'),
}


def load_calib_dataset(dataset_name_or_dir: str,
                       config_name: Optional[str] = None,
                       split: Optional[str] = None,
                       key: Optional[str] = None,
                       trust_remote_code=True,
                       **kwargs):
    if config_name is None:
        for name, meta in DEFAULT_HF_DATASET_META.items():
            if name in dataset_name_or_dir:
                if config_name is None:
                    config_name = meta[0]
                if split is None:
                    split = meta[1]
                if key is None:
                    key = meta[2]
                break

    dataset = load_dataset(dataset_name_or_dir,
                           name=config_name,
                           split=split,
                           **kwargs)
    return dataset[key]


@torch.no_grad()
def apply_smoothing(
        scales: torch.Tensor,
        gemm_weights: Union[torch.Tensor, List[torch.Tensor]],
        layernorm_weights: Optional[Union[torch.Tensor,
                                          List[torch.Tensor]]] = None,
        layernorm_bias: Optional[Union[torch.Tensor,
                                       List[torch.Tensor]]] = None,
        dtype: torch.dtype = torch.float32,
        layernorm_1p: bool = False):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]

    if layernorm_weights is not None:
        assert layernorm_weights.numel() == scales.numel()
        layernorm_weights.div_(scales).to(dtype)
    if layernorm_bias is not None:
        assert layernorm_bias.numel() == scales.numel()
        layernorm_bias.div_(scales).to(dtype)
    if layernorm_1p:
        layernorm_weights += (1 / scales) - 1

    for gemm in gemm_weights:
        gemm.mul_(scales.view(1, -1)).to(dtype)


@torch.no_grad()
def smooth_gemm(gemm_weights,
                act_scales,
                layernorm_weights=None,
                layernorm_bias=None,
                alpha: Optional[float] = 0.5,
                weight_scales=None):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]
    orig_dtype = gemm_weights[0].dtype

    for gemm in gemm_weights:
        # gemm_weights are expected to be transposed
        assert gemm.shape[1] == act_scales.numel()

    if weight_scales is None:
        weight_scales = torch.cat(
            [gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights],
            dim=0)
        weight_scales = weight_scales.max(dim=0)[0]
    weight_scales.to(float).clamp(min=1e-5)
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    apply_smoothing(scales, gemm_weights, layernorm_weights, layernorm_bias,
                    orig_dtype)

    return scales


@torch.no_grad()
def smooth_gemm_fc1_gate(fc1_weights,
                         gate_weights,
                         act_scales,
                         layernorm_weights=None,
                         layernorm_bias=None,
                         alpha=0.5,
                         weight_scales=None):
    gemm_weights = []
    if not isinstance(fc1_weights, list):
        fc1_weights = [fc1_weights]
    if not isinstance(gate_weights, list):
        gate_weights = [gate_weights]

    for i in range(len(fc1_weights)):
        gemm_weight = torch.cat([fc1_weights[i], gate_weights[i]], dim=0)
        gemm_weights.append(gemm_weight)

    orig_dtype = gemm_weights[0].dtype

    for gemm in gemm_weights:
        # gemm_weights are expected to be transposed
        assert gemm.shape[1] == act_scales.numel()

    if weight_scales is None:
        weight_scales = torch.cat(
            [gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights],
            dim=0)
        weight_scales = weight_scales.max(dim=0)[0]
    weight_scales.to(float).clamp(min=1e-5)
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    apply_smoothing(scales, fc1_weights + gate_weights, layernorm_weights,
                    layernorm_bias, orig_dtype)

    return scales


def generate_int8(
    weights: torch.Tensor,
    act_range: Dict[str, torch.Tensor],
    is_qkv: bool = False,
    multi_query_mode: bool = False,
):
    """
     This function has two purposes:
      - compute quantized weights, scaled either per-tensor or per-column
      - compute scaling factors

      Depending on the GEMM API (CUTLASS/CUBLAS) the required scaling factors differ.
      CUTLASS uses two sets of scaling factors. One for the activation X, one for the weight W.
      CUBLAS only has one (we can't do per-row scaling). So we must provide pre-multiplied scaling factor.

      Here is the list of what we need (T means per-tensor, C per-column):
        - scale_x_orig_quant puts fp activation into the quantized range (i.e. [-128, 127], for int8). Used before the GEMM. (T)
        - scale_y_quant_orig puts quantized activation into the fp range. Used if the GEMM outputs int8. (T)
        - scale_w_quant_orig puts weights from quant range to fp range (used with CUTLASS) (T, C)
        - scale_y_accum_quant puts the GEMM result (XW) from accumulation range (int32)
          to quant range (int8) (used for CUBLAS) (T, C)

      Note that we don't do anything special about row-parallel GEMM. Theoretically, we could have per-GPU scaling factors too,
      but then the model would change depending on the number of GPUs used.

      For QKV projection, the behavior is special. Even if we have a single matrix to perform QKV projection, we consider it
      as three different matrices: Q, K, and V. So per-tensor actually means one scaling factor for each Q, K and V.
      For our GEMM implementation to respect this behavior, we use per-column mode and replicate values along columns.
    """

    # compute weight scaling factors for fp->int8 and int8->fp
    if is_qkv and not multi_query_mode:
        scale_w_orig_quant_t = 127. / act_range["w"].reshape(3, -1).max(
            dim=-1, keepdims=True)[0].cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].reshape(3,
                                                             -1).cpu().numpy()
    elif is_qkv and multi_query_mode:
        hidden_dim = weights.shape[0]
        local_dim = act_range["w"].shape[0]
        kv_dim = (local_dim - hidden_dim) // 2
        scale_w_q = act_range["w"][0:hidden_dim]
        scale_w_k = act_range["w"][hidden_dim:hidden_dim + kv_dim]
        scale_w_v = act_range["w"][-kv_dim:]

        scale_w_qkv_t = torch.concat([
            scale_w_q.max(dim=0, keepdim=True)[0],
            scale_w_k.max(dim=0, keepdim=True)[0],
            scale_w_v.max(dim=0, keepdim=True)[0]
        ])

        scale_w_orig_quant_t = 127. / scale_w_qkv_t.cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy()
    else:
        scale_w_orig_quant_t = 127. / act_range["w"].max().cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy()
    scale_w_quant_orig_t = 1.0 / scale_w_orig_quant_t
    scale_w_quant_orig_c = 1.0 / scale_w_orig_quant_c

    # compute the rest of needed scaling factors
    scale_x_orig_quant_t = np.array(127. / act_range["x"].max().item())
    scale_y_orig_quant_t = np.array(127. / act_range["y"].max().item())
    scale_y_quant_orig_t = np.array(act_range["y"].max().item() / 127.)
    scale_y_accum_quant_t = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                    scale_w_orig_quant_t)
    scale_y_accum_quant_c = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                    scale_w_orig_quant_c)
    if is_qkv and not multi_query_mode:
        scale_y_accum_quant_t = np.broadcast_to(scale_y_accum_quant_t,
                                                scale_w_orig_quant_c.shape)
        scale_w_quant_orig_t = np.broadcast_to(scale_w_quant_orig_t,
                                               scale_w_orig_quant_c.shape)
    if is_qkv and multi_query_mode:
        scale_q_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[0],
                                            scale_w_q.shape)
        scale_k_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[1],
                                            scale_w_k.shape)
        scale_v_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[2],
                                            scale_w_v.shape)
        scale_y_accum_quant_t = np.concatenate(
            [scale_q_y_accum_t, scale_k_y_accum_t, scale_v_y_accum_t])
        scale_w_quant_orig_t = np.concatenate([
            np.broadcast_to(scale_w_quant_orig_t[0], scale_w_q.shape),
            np.broadcast_to(scale_w_quant_orig_t[1], scale_w_k.shape),
            np.broadcast_to(scale_w_quant_orig_t[2], scale_w_v.shape)
        ])

    to_i8 = lambda x: x.round().clip(-127, 127).astype(np.int8)
    np_weights = weights.cpu().detach().numpy()

    if is_qkv and multi_query_mode:
        scale_w_quant_orig_t_expand = np.ones([weights.shape[-1]])
        scale_w_quant_orig_t_expand[:hidden_dim] = scale_w_quant_orig_t[0]
        scale_w_quant_orig_t_expand[hidden_dim:hidden_dim +
                                    kv_dim] = scale_w_quant_orig_t[1]
        scale_w_quant_orig_t_expand[-kv_dim:] = scale_w_quant_orig_t[2]
        weight_int8 = to_i8(np_weights * scale_w_quant_orig_t_expand)
    else:
        weight_int8 = to_i8(np_weights * scale_w_orig_quant_t)
    return {
        "weight.int8": weight_int8,
        "weight.int8.col": to_i8(np_weights * scale_w_orig_quant_c),
        "scale_x_orig_quant": scale_x_orig_quant_t.astype(np.float32),
        "scale_w_quant_orig": scale_w_quant_orig_t.astype(np.float32),
        "scale_w_quant_orig.col": scale_w_quant_orig_c.astype(np.float32),
        "scale_y_accum_quant": scale_y_accum_quant_t.astype(np.float32),
        "scale_y_accum_quant.col": scale_y_accum_quant_c.astype(np.float32),
        "scale_y_quant_orig": scale_y_quant_orig_t.astype(np.float32),
    }


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8,
                           dtype='float32',
                           use_gemm_woq_plugin=False,
                           use_fp8_rowwise=False,
                           weight_scale=None,
                           clamp_value=[-1200.0, 1200],
                           tp_rank=0,
                           postfix='weight',
                           quant_scale_name=None):
    results = {}
    if use_weight_only:
        if weight_scale:
            logger.error(
                "Weight only doesn't support loading scales from the weights.")
        if weight.dim() > 2:
            v = weight.transpose(1, 2).contiguous()
        else:
            v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v.cpu(), plugin_weight_only_quant_type)
        if not use_gemm_woq_plugin:
            results[prefix + postfix] = v.to(dtype)
        else:
            results[prefix + postfix] = processed_torch_weights
        if quant_scale_name is not None:
            results[quant_scale_name] = torch_weight_scales
        else:
            results[prefix + 'per_channel_scale'] = torch_weight_scales
    elif use_fp8_rowwise:
        if weight_scale is not None:
            assert weight.dtype == torch.float8_e4m3fn, "weight data type must be torch.float8_e4m3fn"
            results[prefix + postfix] = weight
            torch_weight_scales = weight_scale.to(torch.float32)
        else:
            processed_torch_weights, torch_weight_scales = fp8_per_channel_quant_weight_gpu(
                weight, clamp_value)
            results[prefix + postfix] = processed_torch_weights
            torch_weight_scales = torch_weight_scales.to(torch.float32)

        if quant_scale_name is not None:
            results[quant_scale_name] = torch_weight_scales
        else:
            results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + postfix] = weight

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results
