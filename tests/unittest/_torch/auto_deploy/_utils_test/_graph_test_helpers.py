import copy
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from _torch_test_utils import all_close, reset_parameters
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export, torch_export_to_gm


def count_parameters(model: torch.nn.Module):
    for n, p in model.named_parameters():
        print(n, p.shape)
    return sum(np.prod(p.shape) for p in model.parameters())


def count_buffers(model: torch.nn.Module):
    for n, b in model.named_buffers():
        print(n, b.shape)
    return sum(np.prod(b.shape) for b in model.buffers())


def run_test(
    model: nn.Module,
    x: torch.Tensor,
    transform: Callable[
        [GraphModule, Optional[str], Optional[List[str]], Optional[bool]], GraphModule
    ],
    check_transformed_graph: Callable[[GraphModule], bool],
    _get_expected_num_params: Callable[[int], int],
    atol: float = 1e-3,
    rtol: float = 1e-3,
    test_load_hook: bool = True,
    strict_loading: bool = True,
    dynamic_shapes: Dict = None,
    *args,  # Additional arguments for transform
) -> GraphModule:
    # run model once
    y_model = model(x)

    # num params
    num_params_model = count_parameters(model)
    print(num_params_model)

    # export + check (we clone the state dict to have a bit more freedom in testing below)
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dynamic_shapes,), clone=True)
    print(gm)
    y_gm = gm(x)
    num_params_gm = count_parameters(gm)

    assert num_params_model == num_params_gm
    assert all_close(y_model, y_gm)

    # graph transformation + check
    gm_transformed = transform(gm, *args)
    print(gm_transformed)
    # in case buffers or other tensors were added during the transform
    gm_transformed = gm_transformed.to("cuda")
    y_transformed = gm_transformed(x)
    n_p_transformed = count_parameters(gm_transformed)

    n_p_t_expected = _get_expected_num_params(num_params_model)
    assert n_p_transformed == n_p_t_expected, (
        f"actual params {n_p_transformed} != expected params {n_p_t_expected}"
    )

    # check if the transformation worked
    assert check_transformed_graph(gm_transformed)

    if strict_loading:
        # check if output equals without loading state dict
        assert all_close(y_model, y_transformed, atol=atol, rtol=rtol)

    if test_load_hook:
        # check if loading hook works from original state dict
        reset_parameters(gm_transformed)
        y_random = gm_transformed(x)
        assert not all_close(y_model, y_random)

        gm_transformed.load_state_dict(model.state_dict(), strict=True if strict_loading else False)
        y_loaded_from_original = gm_transformed(x)
        assert all_close(y_model, y_loaded_from_original, atol=atol, rtol=rtol)

        # check if loading hook works from state_dict of a transformed model
        state_dict_sharded = copy.deepcopy(gm_transformed.state_dict())
        reset_parameters(gm_transformed)
        y_random2 = gm_transformed(x)
        assert not all_close(y_model, y_random2)

        gm_transformed.load_state_dict(state_dict_sharded, strict=True if strict_loading else False)
        y_loaded_from_transformed = gm_transformed(x)
        assert all_close(y_model, y_loaded_from_transformed, atol=atol, rtol=rtol)

    # check if we can still export the model as expected
    torch_export(gm_transformed, args=(x,))
    torch_export_to_gm(gm_transformed, args=(x,))

    # return graph module for further testing
    return gm_transformed
