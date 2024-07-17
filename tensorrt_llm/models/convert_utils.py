import re
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from datasets import load_dataset

from ..quantization import QuantAlgo


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].clone()


def split_qkv_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    v = v.reshape(3, n_hidden, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel), n_hidden)
    return split_v.clone()


def split_qkv_bias_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV bias according to tensor parallelism
    """
    v = v.reshape(3, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel))
    return split_v.clone()


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


def weight_only_quantize_dict(weights: Dict[str, torch.Tensor],
                              quant_algo: str,
                              quant_weights=[
                                  'qkv.weight', 'dense.weight', 'fc.weight',
                                  'proj.weight', 'gate.weight'
                              ],
                              exclude_weights=['shared_expert_gate.weight'],
                              plugin: bool = True):
    if quant_algo not in [QuantAlgo.W4A16, QuantAlgo.W8A16]:
        return weights
    for name in list(weights):
        if any([_name in name for _name in exclude_weights]):
            continue
        if any([_name in name for _name in quant_weights
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
