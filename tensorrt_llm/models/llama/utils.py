import re
from pathlib import Path
from typing import Dict, Optional, Union

import torch


def load_state_dict(
    file_path: Union[str, Path],
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """ Load weights from model file

    `safetensors` or `pytorch binary` is supported

    # Args.
        file_path: model file path, ends with .bin or .safetensors.
        dtype: torch.dtype, data type.
        device: torch device like, optional. If None, load to cpu.
    # Returns.
        Dict[str, torch.Tensor]
    """
    file_path = Path(file_path)
    assert isinstance(dtype, torch.dtype)

    if device is None:
        device = 'cpu'

    model_params = {}
    if file_path.suffix == '.safetensors':
        # load from safetensors file
        from safetensors import safe_open
        with safe_open(file_path, framework='pt', device=device) as f:
            for name in f.keys():
                model_params[name] = f.get_tensor(name).to(dtype)
    elif file_path.suffix == '.bin':
        # load from pytorch bin file
        state_dict = torch.load(file_path, map_location=device)
        for name in state_dict:
            model_params[name] = state_dict[name].to(dtype)
    else:
        raise NotImplementedError(
            f'Support .safetensors or .bin files, but got {str(file_path)}')
    return model_params


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
