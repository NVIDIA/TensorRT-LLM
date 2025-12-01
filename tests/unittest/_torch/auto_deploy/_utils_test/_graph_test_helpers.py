import copy
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from _torch_test_utils import all_close, reset_parameters
from torch.export import export
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.factory import (
    FullModelExportInfo,
    ModelFactory,
    SubModuleExportInfo,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import ShardingTransformInfo


class FakeFactory(ModelFactory):
    """Dummy factory to pass cache_config for testing."""

    def __init__(self, model=None, cache_config=None, quant_config=None):
        self._model = model
        self.cache_config = cache_config
        self.quant_config = quant_config

    def build_model(self, device: str):
        return self._model.to(device=device) if self._model else None

    def _build_model(self, device: str):
        return

    def _load_checkpoint(self, model, device):
        return

    def get_cache_config(self):
        return self.cache_config

    def get_quant_config(self):
        return self.quant_config

    def get_export_infos(self, model: nn.Module) -> List[SubModuleExportInfo]:
        return [FullModelExportInfo()]


class SequenceEmbeddingInfo(SequenceInfo):
    """A sequence info object for testing that replaces the input_ids with an embedding tensor.

    This is useful to run tests without the tokenizer in the loop.
    """

    def _add_hidden_dim(self, input_ids: Sequence[Sequence[Any]]) -> torch.Tensor:
        return torch.rand(
            *input_ids.shape,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )

    def __init__(self, *args, hidden_size: int, dtype: torch.dtype, **kwargs):
        self._initialized = False
        super().__init__(*args, **kwargs)

        # overwrite input_ids with an embedding tensor and run reset again
        self.hidden_size = hidden_size
        self.dtype = dtype
        self._args_device["input_ids"] = self._add_hidden_dim(self._args_device["input_ids"])
        self._args_host["input_ids"] = self._args_device["input_ids"].cpu()
        self._initialized = True
        self.reset()

    def nest_sequences(self, input_ids: Sequence[Sequence[Any]], *args, **kwargs) -> None:
        # convert input_ids to an embedding tensor if needed
        if not (isinstance(input_ids, torch.Tensor) and input_ids.ndim == 3) and self._initialized:
            # first convert to a list of tensors
            input_embeds = [
                torch.tensor(ids, device=self.device, dtype=self.dtype) for ids in input_ids
            ]
            # then add the hidden dimension to every tensor
            input_embeds = [self._add_hidden_dim(ids) for ids in input_embeds]
        else:
            input_embeds = input_ids

        super().nest_sequences(input_embeds, *args, **kwargs)


def count_parameters(model: torch.nn.Module):
    for n, p in model.named_parameters():
        print(n, p.shape)
    return sum(np.prod(p.shape) for p in model.parameters())


def count_buffers(model: torch.nn.Module):
    for n, b in model.named_buffers():
        print(n, b.shape)
    return sum(np.prod(b.shape) for b in model.buffers())


def run_test_transformed_gm(
    model: nn.Module,
    x: torch.Tensor,
    gm_transformed: GraphModule,
    check_transformed_graph: Callable[[GraphModule], bool],
    _get_expected_num_params: Callable[[int], int],
    atol: float = 1e-3,
    rtol: float = 1e-3,
    test_load_hook: bool = True,
    strict_loading: bool = True,
    dynamic_shapes: Dict = None,
    skip_output_assert: bool = False,
    *args,  # Additional arguments for transform
) -> GraphModule:
    # run model once
    y_model = model(x)

    # num params
    num_params_model = count_parameters(model)
    print(num_params_model)

    # export + check (we clone the state dict to have a bit more freedom in testing below)
    gm_ref = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dynamic_shapes,), clone=True)
    print(gm_ref)
    y_gm = gm_ref(x)
    num_params_gm = count_parameters(gm_ref)

    assert num_params_model == num_params_gm
    if not skip_output_assert:
        torch.testing.assert_close(y_model, y_gm, atol=atol, rtol=rtol)

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

    if strict_loading and not skip_output_assert:
        # check if output equals without loading state dict
        torch.testing.assert_close(y_model, y_transformed, atol=atol, rtol=rtol)

    if test_load_hook and not skip_output_assert:
        # check if loading hook works from original state dict
        reset_parameters(gm_transformed)
        y_random = gm_transformed(x)
        assert not all_close(y_model, y_random), f"{y_model=}, {y_random=}"

        gm_transformed.load_state_dict(model.state_dict(), strict=True if strict_loading else False)
        y_loaded_from_original = gm_transformed(x)
        torch.testing.assert_close(y_model, y_loaded_from_original, atol=atol, rtol=rtol)

        # check if loading hook works from state_dict of a transformed model
        state_dict_sharded = copy.deepcopy(gm_transformed.state_dict())
        reset_parameters(gm_transformed)
        y_random2 = gm_transformed(x)
        assert not all_close(y_model, y_random2), f"{y_model=}, {y_random2=}"

        gm_transformed.load_state_dict(state_dict_sharded, strict=True if strict_loading else False)
        y_loaded_from_transformed = gm_transformed(x)
        torch.testing.assert_close(y_model, y_loaded_from_transformed, atol=atol, rtol=rtol)

        # check if we can still export the model as expected
        export(gm_transformed, args=(x,))


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
    check_num_matches: int = None,  # Additional check of # patterns detected
    skip_output_assert: bool = False,
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
    if not skip_output_assert:
        torch.testing.assert_close(y_model, y_gm, atol=atol, rtol=rtol)

    # graph transformation + check
    if check_num_matches:
        num_matches = transform(gm, *args)
        assert check_num_matches == num_matches, (
            f"expect {check_num_matches} matches, but got {num_matches}"
        )
    else:
        transform(gm, *args)
    print(gm)
    # in case buffers or other tensors were added during the transform
    gm = gm.to("cuda")
    y_transformed = gm(x)
    n_p_transformed = count_parameters(gm)

    n_p_t_expected = _get_expected_num_params(num_params_model)
    assert n_p_transformed == n_p_t_expected, (
        f"actual params {n_p_transformed} != expected params {n_p_t_expected}"
    )

    # check if the transformation worked
    assert check_transformed_graph(gm)

    if strict_loading and not skip_output_assert:
        # check if output equals without loading state dict
        torch.testing.assert_close(y_model, y_transformed, atol=atol, rtol=rtol)

    if test_load_hook and not skip_output_assert:
        # check if loading hook works from original state dict
        reset_parameters(gm)
        y_random = gm(x)
        assert not all_close(y_model, y_random), f"{y_model=}, {y_random=}"

        gm.load_state_dict(model.state_dict(), strict=True if strict_loading else False)
        y_loaded_from_original = gm(x)
        torch.testing.assert_close(y_model, y_loaded_from_original, atol=atol, rtol=rtol)

        # check if loading hook works from state_dict of a transformed model
        state_dict_sharded = copy.deepcopy(gm.state_dict())
        reset_parameters(gm)
        y_random2 = gm(x)
        assert not all_close(y_model, y_random2), f"{y_model=}, {y_random2=}"

        gm.load_state_dict(state_dict_sharded, strict=True if strict_loading else False)
        y_loaded_from_transformed = gm(x)
        torch.testing.assert_close(y_model, y_loaded_from_transformed, atol=atol, rtol=rtol)

    # check if we can still export the model as expected
    export(gm, args=(x,))

    # return graph module for further testing
    return gm


def run_sharding_pattern_detection_test(
    detected_transformations: List[ShardingTransformInfo],
    expected_transformations: List[ShardingTransformInfo],
) -> None:
    """Compare two lists of transformations ignoring order.

    Args:
        detected_transformations: List of detected transformation configurations
        expected_transformations: List of expected transformation configurations
    """
    # Convert to sets for unordered comparison
    detected_set = set(detected_transformations)
    expected_set = set(expected_transformations)
    print("detected_set", detected_set)
    print("expected_set", expected_set)

    assert detected_set == expected_set, "Expected sharding pattern does not match detected pattern"
