import importlib.metadata
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.export as te
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch import fx

from ..utils.logger import ad_logger
from ._graph import canonicalize_graph, lift_to_meta, load_buffers_and_params, tree_to

try:
    from modelopt.torch.quantization.utils import export_torch_mode as torch_export_context
except ImportError:
    torch_export_context = nullcontext


def _clean_up_device_info(gm: fx.GraphModule):
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


def _deduplicate_params_and_buffers(gm: fx.GraphModule):
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
                    param_key_remaining=node_kept.target,
                    param_key_removed=n.target,
                )
            )

            ad_logger.debug(f"Deduplicated: {n.target} --> {node_kept.target}")

    canonicalize_graph(gm)


# TODO: remove once https://github.com/pytorch/pytorch/issues/140710 is resolved
def _torch_where_patch(condition: torch.Tensor, *args, **kwargs):
    if len(args) == 0 and len(kwargs) == 0:
        return torch.nonzero(condition, as_tuple=True)
    return _torch_where_patch.where_original(condition, *args, **kwargs)


_torch_where_patch.where_original = torch.where


def _torch_linear_patch(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_linear_simple(input, weight, bias)


# TODO: remove once https://github.com/pytorch/pytorch/issues/142439 is resolved
def _torch_modulelist_getitem_patch(self: nn.ModuleList, idx):
    if isinstance(idx, slice):
        # return a simple list.
        # NOTE: this obviously only works for any use case where we access the sliced module list
        # like a regular list like a for-loop. For most other things, this hack will not work.
        return list(self._modules.values())[idx]
    else:
        return _torch_modulelist_getitem_patch.getitem_original(self, idx)


_torch_modulelist_getitem_patch.getitem_original = nn.ModuleList.__getitem__


def _torch_tensor_patch(data, **kwargs):
    """Patch torch.tensor to handle 0.0 on meta device.

    ``torch.tensor(0.0, device="meta")`` does not work and hence we are patching it to use
    ``torch.zeros((), device="meta")`` instead, which is equivalent.
    """
    device = kwargs.get("device", None)
    if data == 0.0 and device is not None and torch.device(device) == torch.device("meta"):
        return torch.zeros((), **kwargs)
    return _torch_tensor_patch.tensor_original(data, **kwargs)


_torch_tensor_patch.tensor_original = torch.tensor


def _transformers_version() -> str:
    """Get the version of transformers."""
    return version.parse(importlib.metadata.version("transformers")).base_version


# TODO (@lucaslie): https://github.com/NVIDIA/TensorRT-LLM/issues/5728
# not great that this patch is here but it's the least invasisve change until we make headway on the
# above issue.
@contextmanager
def _transformers_sdpa_mask_patch():
    """Patch transformers.masking_utils.sdpa_mask to be export-compatible."""
    # this patch is only needed+compatible for transformers >= 4.53.0
    if version.parse(_transformers_version()) < version.parse("4.53.0"):
        yield  # Just yield without doing anything (like nullcontext)
        return

    # imports only after version check
    from transformers import masking_utils
    from transformers.integrations.executorch import sdpa_mask_without_vmap

    # recall original implementation
    sdpa_mask_original = masking_utils.sdpa_mask

    # patch function and mask attention interface
    masking_utils.sdpa_mask = sdpa_mask_without_vmap
    if "sdpa" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._local_mapping:
        sdpa_local_original = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._local_mapping["sdpa"]
    else:
        sdpa_local_original = None
    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask_without_vmap

    try:
        yield
    finally:
        # revert patches
        masking_utils.sdpa_mask = sdpa_mask_original
        if sdpa_local_original is None:
            del masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
        else:
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_local_original


def add_missing_load_hooks(gm: fx.GraphModule, model: nn.Module) -> fx.GraphModule:
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


def add_load_hook_for_aliased_params(gm: fx.GraphModule, model: nn.Module):
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


@torch.inference_mode()
def torch_export(model: nn.Module, *args, **kwargs) -> te.ExportedProgram:
    """Just like torch.export except we decorate it to be in inference_mode."""
    with torch_export_context():
        ep = te.export(model, *args, **kwargs)

    # return the result
    return ep


def torch_export_to_gm(
    model: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    clone: bool = False,  # clone or don't clone the model state_dict
    *,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    strict: bool = False,
) -> fx.GraphModule:
    """torch_export with wrapping into GraphModule + useful additions to the resulting module.

    This utility improves over stock torch.export.export in the following aspects:

        1. Provide patches for certain corner cases that torch.export does not support.
        2. Standardize the export process to strictly run on the meta device.
        3. Automatically extract the GraphModule from the exported program.
        4. Retain load hooks for state_dict loading from the oroginal module.
        5. Manage parameter aliasing in the model.
    """
    # we need to better control how F.scaled_dot_product_attention is represented in the graph
    # there is no guarantee how it is represented and we need to make sure it is easily identifiable
    # in the graph.
    sdpa_original = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = torch.ops.auto_deploy.torch_attention_sdpa

    # We overwrite the linear functional as well. This basically avoids exporting the view ops
    # that are used to flatten/unflatten multiple batch dimensions of the input tensor.
    linear_original = F.linear
    # patch linear → always supply bias
    F.linear = _torch_linear_patch

    # patch torch.where(condition) to torch.nonzero(condition, as_tuple=True)
    torch.where = _torch_where_patch

    # patch nn.ModuleList.__getitem__ to handle slicing
    nn.ModuleList.__getitem__ = _torch_modulelist_getitem_patch

    # overwrite autocast/sdpa contextmanagers to be no-ops
    autocast_original = torch.autocast
    sdpa_kernel_original = torch.nn.attention.sdpa_kernel
    torch.autocast = lambda *args, **kwargs: nullcontext()
    torch.nn.attention.sdpa_kernel = lambda *args, **kwargs: nullcontext()

    # patch torch.tensor to handle 0.0 on meta device
    torch.tensor = _torch_tensor_patch

    # run export with sdpa masking patch and lifted to meta
    with _transformers_sdpa_mask_patch():
        with lift_to_meta(model) as state_dict:
            # clean up args, kwargs and move to correct device
            args, kwargs = tree_to((args, kwargs or {}), device="meta")

            # NOTE: we always export in non-strict mode for now as it relaxes some
            # assumptions around tracing. Strict mode uses torchdynamo (symbolic bytecode analysis),
            # which can be brittle since it relies on the exact bytecode representation of the model
            # see here as well: https://pytorch.org/docs/stable/export.html#non-strict-export

            # run export and extract graph module
            egm: fx.GraphModule = torch_export(
                model, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict
            ).module()

            # load state_dict into egm
            # NOTE: export might have removed unused params/buffers (hence we allow unexpected keys)
            load_buffers_and_params(
                egm, state_dict, strict_missing=True, strict_unexpected=False, clone=clone
            )

    # revert sdpa back to original
    F.scaled_dot_product_attention = sdpa_original

    # revert linear back to original
    F.linear = linear_original

    # revert torch.where patch
    torch.where = _torch_where_patch.where_original

    # revert nn.ModuleList.__getitem__ patch
    nn.ModuleList.__getitem__ = _torch_modulelist_getitem_patch.getitem_original

    # revert autocast/sdpa back to original
    torch.autocast = autocast_original
    torch.nn.attention.sdpa_kernel = sdpa_kernel_original

    # revert torch.tensor patch
    torch.tensor = _torch_tensor_patch.tensor_original

    # Export strips away all methods not traced during forward. The model could have
    # load hooks that contain logic for correct state_dict loading. We need to add those
    # hooks back to the exported graph module.
    add_missing_load_hooks(egm, model)

    # Add load hook to correctly load parameters that are aliased in the source model.
    # deduplicate params and buffers
    # TODO (lucaslie, suyoggupta): seems there is some overlap here. I believe we should just have
    # the deduplicate function and extend it to handle reading from state dict for any name.
    add_load_hook_for_aliased_params(egm, model)
    _deduplicate_params_and_buffers(egm)

    # clean up devices in the graph
    # This is a consequence of lifting to meta during export.
    _clean_up_device_info(egm)

    # show exported graph
    ad_logger.debug("exported graph: " + str(egm))

    return egm
