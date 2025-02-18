import torch

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor


def make_weak_ref(x):

    if isinstance(x, torch.Tensor):
        return convert_to_torch_tensor(
            TensorWrapper(x.data_ptr(), x.dtype, x.shape)) if x.is_cuda else x
    elif isinstance(x, tuple):
        return tuple(make_weak_ref(i) for i in x)
    elif isinstance(x, list):
        return [make_weak_ref(i) for i in x]
    elif isinstance(x, dict):
        return {k: make_weak_ref(v) for k, v in x.items()}
    elif isinstance(x, (int, float, bool)):
        return x
    else:
        raise TypeError(f"Invalid type {type(x)} to make weak ref")
