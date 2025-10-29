"""Main export functionality with utilities for torch.export."""

from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.export as te
import torch.nn as nn
from torch import fx

from ..utils._graph import canonicalize_graph, lift_to_meta, load_buffers_and_params, tree_to
from ..utils.logger import ad_logger
from ..utils.node_utils import is_op
from .interface import apply_export_patches

try:
    from modelopt.torch.quantization.utils import export_torch_mode as torch_export_context
except ImportError:
    torch_export_context = nullcontext


def _clean_up_device_info(gm: fx.GraphModule) -> None:
    """Correct device information in the graph."""
    devices = {t.device for _, t in gm.named_parameters()}
    if len(devices) == 0:
        return
    elif len(devices) > 1:
        raise AssertionError("All parameters should be on the same device.")
    device = devices.pop()
    meta_device = torch.device("meta")

    for node in gm.graph.nodes:
        if any(a == meta_device for a in node.args):
            new_args = list(node.args)
            new_args = [a if a != meta_device else device for a in new_args]
            node.args = tuple(new_args)
        if any(a == meta_device for a in node.kwargs.values()):
            new_kwargs = dict(node.kwargs)
            new_kwargs = {k: v if v != meta_device else device for k, v in new_kwargs.items()}
            node.kwargs = new_kwargs

    canonicalize_graph(gm)


def _load_hook_for_deduplication(
    state_dict, prefix, *args, param_key_remaining: str, param_key_removed: str
):
    """Check for removed param key and and put it into the key that is remaining."""
    ad_logger.debug(f"Loading hook for deduplication: {param_key_remaining} <- {param_key_removed}")
    k_remaining = prefix + param_key_remaining
    k_removed = prefix + param_key_removed
    if k_removed in state_dict:
        state_dict[k_remaining] = state_dict.pop(k_removed)


def _deduplicate_params_and_buffers(gm: fx.GraphModule) -> None:
    """This will de-duplicate params and buffers that share the same tensor."""
    # get all get_attr nodes
    get_attr_nodes = [n for n in gm.graph.nodes if n.op == "get_attr"]

    # sort by id of target
    targets: Dict[int, List[fx.Node]] = defaultdict(list)
    for n in get_attr_nodes:
        submod, _, name = n.target.rpartition(".")
        t_target = getattr(gm.get_submodule(submod), name)
        targets[id(t_target)].append(n)
    # now replace all instances of the same tensor with the same get_attr node (idx 0 in the list)
    for nodes in targets.values():
        node_kept = nodes[0]
        for n in nodes[1:]:
            n.replace_all_uses_with(node_kept)
            gm.graph.erase_node(n)

            # remove the param/buffer from the submodule
            submod, _, name = n.target.rpartition(".")
            delattr(gm.get_submodule(submod), name)

            # add load hooks to also load the weights correctly
            gm._register_load_state_dict_pre_hook(
                partial(
                    _load_hook_for_deduplication,
                    param_key_remaining=str(node_kept.target),
                    param_key_removed=str(n.target),
                )
            )

            ad_logger.debug(f"Deduplicated: {n.target} --> {node_kept.target}")

    canonicalize_graph(gm)


def _add_missing_load_hooks(gm: fx.GraphModule, model: nn.Module) -> None:
    """Adds back the state dict load hooks stripped away during export."""
    hooks = {
        k: mod._load_state_dict_pre_hooks
        for k, mod in model.named_modules()
        if mod._load_state_dict_pre_hooks
    }

    for mod_name, mod in gm.named_modules():
        if mod_name in hooks:
            for hook in hooks.pop(mod_name).values():
                mod._register_load_state_dict_pre_hook(hook.hook, with_module=hook.with_module)
    assert not (bool(hooks)), f"""Mismatch in names of exported and source modules with hooks.
        The following module names were not found in exported module {list(hooks.keys())}"""


def _add_load_hook_for_aliased_params(gm: fx.GraphModule, model: nn.Module) -> None:
    """
    Add a load hook to handle aliased parameters in the model.

    When parameters are aliased (multiple parameter names point to the same tensor),
    we need to ensure all aliases get the same value during loading. This hook:
    1. Identifies groups of aliased parameters
    2. For each group, finds a valid parameter value from the state dict
    3. Applies that value to all aliases in the group

    Args:
        gm: The graph module to add the hook to
        model: The source model containing the original parameter aliases
    """

    def find_valid_param_value(
        state_dict: Dict[str, torch.Tensor], param_names: List[str]
    ) -> Optional[torch.Tensor]:
        """Find a valid parameter value from state dict for a group of aliased parameters.

        Args:
            state_dict: The state dict being loaded
            param_names: List of parameter names that are aliases of each other

        Returns:
            A valid tensor value if found, None otherwise
        """
        # First try to find a non-meta tensor value
        value = None
        for name in param_names:
            if name in state_dict:
                value = state_dict[name]
                if value.device.type != "meta":
                    return value

        return value

    def aliasing_load_pre_hook(state_dict: Dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        """Load hook that ensures aliased parameters get the same value."""
        for group in aliased_groups:
            # Find a valid value for this group of aliases
            value = find_valid_param_value(state_dict, group)

            if value is not None:
                # Apply the value to all aliases
                for name in group:
                    state_dict[name] = value

                ad_logger.debug(f"Applied value from {group[0]} to aliased parameters: {group}")

    # Find all parameter aliases in the source model
    param_to_names = defaultdict(list)
    for name, param in model.named_parameters(remove_duplicate=False):
        param_to_names[id(param)].append(name)

    # Filter to only groups with multiple aliases
    aliased_groups = [names for names in param_to_names.values() if len(names) > 1]

    if not aliased_groups:
        return

    # Register the hook
    gm._register_load_state_dict_pre_hook(aliasing_load_pre_hook)


def _clean_up_assertions(gm: fx.GraphModule):
    """This transformations removes shape checks and assertions from the graph."""
    check_ops = {
        torch.ops.aten._assert_scalar,
        torch.ops.aten.sym_constrain_range,
        torch.ops.aten.sym_constrain_range_for_size,
        torch.ops.aten._assert_tensor_metadata,
        # torch.ops.aten._functional_sym_constrain_range,
        # torch.ops.aten._functional_sym_constrain_range_for_size
    }
    graph: fx.Graph = gm.graph
    for node in reversed(graph.nodes):
        if len(node.users) > 0 or not is_op(node, check_ops):
            continue
        graph.erase_node(node)
    canonicalize_graph(gm)


def run_forward_for_capture(
    model: nn.Module,
    capture_fn: Optional[Callable[..., nn.Module]] = None,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    clone: bool = False,  # clone or don't clone the model state_dict
    *,
    patch_configs: Optional[Dict[str, Union[dict, Any]]] = None,
    patch_list: Optional[List[str]] = None,
) -> nn.Module:
    """A wrapper to run the provided closure over the model on the meta device with patches.

    This utility automates the following steps for running a closure (``capture_fn``):

        1. Provide patches for certain corner cases that might not be supported by the closure.
        2. Standardize the execution of the closure to properly run on the meta device.

    Args:
        model: The model to run the closure on
        capture_fn: The closure to run the model with. If not provided, run a forward pass.
        args: Arguments for the model
        kwargs: Keyword arguments for the model
        clone: Whether to clone the model state_dict when the closure returns a different module
        patch_configs: Optional patch configurations. If None, all registered patches
                      will be applied with default settings.
        patch_list: Optional list of patch names to apply with default settings.
    """
    # run capture with patches and lifted to meta
    with apply_export_patches(patch_configs, patch_list), lift_to_meta(model) as state_dict:
        # clean up args, kwargs and move to correct device
        args, kwargs = tree_to((args or (), kwargs or {}), device="meta")

        # NOTE (lucaslie): export is VERY sensitive to the location of the inference_mode
        # context manager. Do NOT move it unless absolutely necessary.
        with torch.inference_mode():
            if capture_fn is None:
                model(*args, **kwargs)
                mod_after_capture = model
            else:
                mod_after_capture = capture_fn(model, args, kwargs)

        # load state_dict into egm
        # NOTE: export might have removed unused params/buffers (hence we allow unexpected keys)
        if mod_after_capture is not model:
            load_buffers_and_params(
                mod_after_capture,
                state_dict,
                strict_missing=True,
                strict_unexpected=False,
                clone=clone,
            )

    return mod_after_capture


def torch_export_to_gm(
    model: nn.Module,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    clone: bool = False,  # clone or don't clone the model state_dict
    *,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    strict: bool = False,
    patch_configs: Optional[Dict[str, Union[dict, Any]]] = None,
    patch_list: Optional[List[str]] = None,
) -> fx.GraphModule:
    """torch's export with wrapping into GraphModule + useful additions to the resulting module.

    This utility improves over stock torch.export.export in the following aspects:

        1. Provide patches for certain corner cases that torch.export does not support.
        2. Standardize the export process to strictly run on the meta device.
        3. Automatically extract the GraphModule from the exported program.
        4. Retain load hooks for state_dict loading from the original module.
        5. Manage parameter aliasing in the model.
        6. Remove assertions from the graph.

    Args:
        model: The model to export
        args: Arguments for the model
        kwargs: Keyword arguments for the model
        clone: Whether to clone the model state_dict
        dynamic_shapes: Dynamic shapes for the export
        strict: Whether to use strict mode for export
        patch_configs: Optional patch configurations. If None, all registered patches
                      will be applied with default settings.
        patch_list: Optional list of patch names to apply with default settings.
                   Cannot be used together with patch_configs.
    """

    def _capture_fn(model, args, kwargs):
        ep = te.export(model, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict)
        egm = ep.module()
        assert isinstance(egm, fx.GraphModule)
        return egm

    # run capture with export
    egm = run_forward_for_capture(
        model, _capture_fn, args, kwargs, clone, patch_list=patch_list, patch_configs=patch_configs
    )

    # Export strips away all methods not traced during forward. The model could have
    # load hooks that contain logic for correct state_dict loading. We need to add those
    # hooks back to the exported graph module.
    _add_missing_load_hooks(egm, model)

    # Add load hook to correctly load parameters that are aliased in the source model.
    # deduplicate params and buffers
    # TODO (lucaslie, suyoggupta): seems there is some overlap here. I believe we should just have
    # the deduplicate function and extend it to handle reading from state dict for any name.
    _add_load_hook_for_aliased_params(egm, model)
    _deduplicate_params_and_buffers(egm)

    # clean up devices in the graph
    # This is a consequence of lifting to meta during export.
    _clean_up_device_info(egm)

    # clean up checks --> generally the sanity checks are overly conservative and we can remove them
    _clean_up_assertions(egm)

    # show exported graph
    ad_logger.debug("exported graph: " + str(egm))

    return egm
