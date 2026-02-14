"""Main export functionality with utilities for torch.export."""

import re
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.export as te
import torch.nn as nn
from torch import fx
from torch.utils._python_dispatch import TorchDispatchMode

from ..utils._graph import canonicalize_graph, lift_to_meta, load_buffers_and_params, tree_to
from ..utils.logger import ad_logger
from ..utils.node_utils import is_op
from .interface import apply_export_patches

if TYPE_CHECKING:
    from ..llm_args import LlmArgs

try:
    from modelopt.torch.quantization.utils import export_torch_mode as torch_export_context
except ImportError:
    torch_export_context = nullcontext


# =====================================================================
# MOE export optimization: reduce experts for faster tracing, then
# expand the graph back to include all experts after export.
# =====================================================================


def _infer_target_pattern(target_0: str, target_1: str) -> Tuple[str, str]:
    """Infer ``(prefix, suffix)`` from two consecutive expert-weight targets.

    Compares two ``get_attr`` targets that differ only in the expert index and
    returns ``(prefix, suffix)`` such that ``target == prefix + str(idx) + suffix``.

    Example::

        >>> _infer_target_pattern('experts.0.gate.weight', 'experts.1.gate.weight')
        ('experts.', '.gate.weight')
    """
    parts_0 = target_0.split(".")
    parts_1 = target_1.split(".")
    if len(parts_0) != len(parts_1):
        raise ValueError(f"Target structure mismatch: {target_0} vs {target_1}")

    diff_positions = [i for i, (a, b) in enumerate(zip(parts_0, parts_1)) if a != b]
    if len(diff_positions) != 1:
        raise ValueError(
            f"Expected exactly one differing part, found {len(diff_positions)}: "
            f"{target_0} vs {target_1}"
        )

    idx = diff_positions[0]
    prefix = ".".join(parts_0[:idx]) + "." if idx > 0 else ""
    suffix = "." + ".".join(parts_0[idx + 1 :]) if idx < len(parts_0) - 1 else ""
    return prefix, suffix


def _infer_single_target_pattern(target: str, expert_prefix: str) -> Tuple[str, str]:
    """Infer ``(prefix, suffix)`` when only one expert target is available.

    Uses the known *expert_prefix* to locate the expert index position.

    Example::

        >>> _infer_single_target_pattern('layer.0.experts.0.w.weight', 'layer.0.experts')
        ('layer.0.experts.', '.w.weight')
    """
    full_prefix = expert_prefix + "."
    if not target.startswith(full_prefix):
        raise ValueError(f"Target '{target}' does not start with '{full_prefix}'")
    remainder = target[len(full_prefix) :]  # e.g. '0.w.weight'
    _idx_str, _, after_idx = remainder.partition(".")
    suffix = "." + after_idx if after_idx else ""
    return full_prefix, suffix


def _register_nested_parameter(gm: fx.GraphModule, dotted_name: str, param: nn.Parameter) -> None:
    """Register a parameter at a nested dotted path, creating intermediate modules as needed."""
    parts = dotted_name.split(".")
    current: nn.Module = gm
    for part in parts[:-1]:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            new_mod = nn.Module()
            current.add_module(part, new_mod)
            current = new_mod
    current.register_parameter(parts[-1], param)


class _MoeExpertProbe(TorchDispatchMode):
    """Dispatch mode that records parameter tensor IDs flowing into ``torch_moe``-family ops.

    Used by :func:`_find_moe_module_lists` to discover which ``nn.ModuleList``
    instances provide expert weights without relying on attribute naming conventions.
    """

    # MOE custom ops whose list arguments represent per-expert weight tensors.
    _MOE_OP_NAMES = ("torch_moe", "torch_quant_fp8_moe", "torch_quant_nvfp4_moe")

    def __init__(self):
        super().__init__()
        self.captured_param_ids: set = set()
        self._moe_ops = self._collect_moe_ops()

    @classmethod
    def _collect_moe_ops(cls) -> set:
        ops: set = set()
        for name in cls._MOE_OP_NAMES:
            try:
                ops.add(getattr(torch.ops.auto_deploy, name).default)
            except AttributeError:
                pass
        return ops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in self._moe_ops:
            for arg in list(args) + list(kwargs.values()):
                if isinstance(arg, (list, tuple)):
                    for item in arg:
                        if isinstance(item, torch.Tensor):
                            self.captured_param_ids.add(id(item))
        return func(*args, **kwargs)


def _find_moe_module_lists(
    model: nn.Module,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[nn.Module, str, nn.ModuleList]]:
    """Identify ``nn.ModuleList`` instances whose parameters feed into ``torch_moe`` ops.

    Runs a lightweight forward pass with :class:`_MoeExpertProbe` active to
    discover which ``nn.ModuleList`` children of the model contribute
    per-expert weight tensors to ``torch_moe``-family custom ops.

    Returns:
        Mapping of *module_list_path* → ``(parent_module, attr_name, module_list)``.
    """
    # Build reverse map:  id(param) → (parent_module, attr_name, module_list, full_path)
    param_to_modlist: Dict[int, Tuple[nn.Module, str, nn.ModuleList, str]] = {}
    for name, module in model.named_modules():
        for attr_name, child in module.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > 0:
                ml_path = f"{name}.{attr_name}" if name else attr_name
                for param in child.parameters():
                    param_to_modlist[id(param)] = (module, attr_name, child, ml_path)

    # Run a quick forward pass to see which params flow into MOE ops.
    probe = _MoeExpertProbe()
    with torch.inference_mode(), probe:
        model(*(args or ()), **(kwargs or {}))

    # Cross-reference captured tensor IDs with ModuleList parameters.
    result: Dict[str, Tuple[nn.Module, str, nn.ModuleList]] = {}
    for pid in probe.captured_param_ids:
        if pid in param_to_modlist:
            parent, attr_name, mod_list, path = param_to_modlist[pid]
            if path not in result:
                result[path] = (parent, attr_name, mod_list)

    return result


def _reduce_moe_experts(
    model: nn.Module,
    min_num_experts: int,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Reduce MOE expert ``nn.ModuleList``s for faster export tracing.

    Uses a probe forward pass to identify which ``nn.ModuleList`` instances
    feed into ``torch_moe``-family custom ops (see :func:`_find_moe_module_lists`),
    then truncates each to *min_num_experts* entries.  The returned list of dicts
    carries the metadata needed by :func:`_restore_moe_experts` and
    :func:`_expand_moe_experts_in_graph`.
    """
    if min_num_experts < 1:
        raise ValueError(f"min_num_experts must be >= 1, got {min_num_experts}")

    moe_lists = _find_moe_module_lists(model, args, kwargs)

    reductions: List[Dict[str, Any]] = []
    for path, (parent, attr_name, mod_list) in moe_lists.items():
        orig_count = len(mod_list)
        if orig_count <= min_num_experts:
            continue

        reductions.append(
            {
                "module": parent,
                "attr_name": attr_name,
                "original_list": mod_list,
                "original_count": orig_count,
                "expert_prefix": path,
            }
        )
        setattr(parent, attr_name, nn.ModuleList(list(mod_list[:min_num_experts])))
        ad_logger.info(
            f"Reduced MOE experts in '{path}' from {orig_count} to "
            f"{min_num_experts} for faster export"
        )
    return reductions


def _restore_moe_experts(reductions: List[Dict[str, Any]]) -> None:
    """Restore MOE expert ``nn.ModuleList``s to their original state."""
    for info in reductions:
        setattr(info["module"], info["attr_name"], info["original_list"])


def _find_original_num_experts(target: str, reductions: List[Dict[str, Any]]) -> Optional[int]:
    """Return the original expert count for a ``get_attr`` *target*, or ``None``."""
    for info in reductions:
        if target.startswith(info["expert_prefix"] + "."):
            return info["original_count"]
    return None


def _find_expert_prefix(target: str, reductions: List[Dict[str, Any]]) -> Optional[str]:
    """Return the ``expert_prefix`` that matches *target*, or ``None``."""
    for info in reductions:
        if target.startswith(info["expert_prefix"] + "."):
            return info["expert_prefix"]
    return None


def _expand_moe_experts_in_graph(
    gm: fx.GraphModule,
    model: nn.Module,
    reductions: List[Dict[str, Any]],
) -> None:
    """Expand MOE expert weights in *gm* to match the full *model*.

    After exporting with a reduced number of experts this function:

    1. Finds every ``torch_moe``-family node whose weight-list arguments are
       shorter than the original expert count.
    2. Registers the missing expert parameters on *gm* (copied from the
       already-restored *model*).
    3. Creates the corresponding ``get_attr`` nodes and extends the weight
       lists in the call node so the graph is equivalent to a full export.
    """
    if not reductions:
        return

    # MOE ops whose arguments include per-expert weight lists (from index 3 onward)
    moe_ops = {
        torch.ops.auto_deploy.torch_moe,
        torch.ops.auto_deploy.torch_quant_fp8_moe,
        torch.ops.auto_deploy.torch_quant_nvfp4_moe,
    }

    graph = gm.graph
    num_expanded = 0

    for node in list(graph.nodes):
        if not is_op(node, moe_ops):
            continue

        # Collect indices of list-of-node arguments (expert weight/scale lists)
        list_arg_indices = [
            i
            for i in range(3, len(node.args))
            if isinstance(node.args[i], (list, tuple)) and len(node.args[i]) > 0
        ]
        if not list_arg_indices:
            continue

        first_list = node.args[list_arg_indices[0]]
        current_num = len(first_list)
        first_target = first_list[0].target
        original_num = _find_original_num_experts(first_target, reductions)

        if original_num is None or original_num <= current_num:
            continue

        ad_logger.debug(
            f"Expanding MOE node '{node.name}': {current_num} -> {original_num} experts"
        )

        # Insert new get_attr nodes at the very beginning of the graph
        first_graph_node = next(iter(graph.nodes))

        new_args = list(node.args)
        for li in list_arg_indices:
            weight_list = list(node.args[li])

            # Determine the naming pattern: prefix + <expert_idx> + suffix
            if len(weight_list) >= 2:
                prefix, suffix = _infer_target_pattern(weight_list[0].target, weight_list[1].target)
            else:
                ep = _find_expert_prefix(weight_list[0].target, reductions)
                assert ep is not None, (
                    f"Could not find expert prefix for target '{weight_list[0].target}'"
                )
                prefix, suffix = _infer_single_target_pattern(weight_list[0].target, ep)

            # Add the missing expert weights
            for expert_idx in range(current_num, original_num):
                new_target = f"{prefix}{expert_idx}{suffix}"

                # Copy the parameter from the restored model
                orig_param = model.get_parameter(new_target)
                _register_nested_parameter(gm, new_target, nn.Parameter(orig_param.data))

                # Create a get_attr node
                with graph.inserting_before(first_graph_node):
                    new_node = graph.get_attr(new_target)
                    new_node.meta["val"] = gm.get_parameter(new_target)

                weight_list.append(new_node)

            new_args[li] = weight_list

        node.args = tuple(new_args)
        num_expanded += 1

    if num_expanded:
        canonicalize_graph(gm)
        ad_logger.info(f"Expanded {num_expanded} MOE node(s) in the exported graph")


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
    pre_hooks = {
        k: mod._load_state_dict_pre_hooks
        for k, mod in model.named_modules()
        if mod._load_state_dict_pre_hooks
    }

    for mod_name, mod in gm.named_modules():
        if mod_name in pre_hooks:
            for hook in pre_hooks.pop(mod_name).values():
                mod._register_load_state_dict_pre_hook(hook.hook, with_module=hook.with_module)
    assert not (bool(pre_hooks)), f"""Mismatch in names of exported and source modules with hooks.
        The following module names were not found in exported module {list(pre_hooks.keys())}"""

    post_hooks = {
        k: mod._load_state_dict_post_hooks
        for k, mod in model.named_modules()
        if mod._load_state_dict_post_hooks
    }

    for mod_name, mod in gm.named_modules():
        if mod_name in post_hooks:
            for hook in post_hooks.pop(mod_name).values():
                mod.register_load_state_dict_post_hook(hook)
    assert not (bool(post_hooks)), f"""Mismatch in names of exported and source modules with hooks.
        The following module names were not found in exported module {list(post_hooks.keys())}"""


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


def _rename_nodes_with_module_hierarchy(gm: fx.GraphModule) -> None:
    """Rename call_function nodes to reflect their module hierarchy.

    Uses nn_module_stack metadata to build hierarchical names like:
    'layers_0_self_attn_linear' instead of 'linear_2'
    """
    graph = gm.graph

    for node in graph.nodes:
        if node.op != "call_function":
            continue

        meta = getattr(node, "meta", None)
        if not isinstance(meta, dict):
            continue

        nn_stack = meta.get("nn_module_stack")
        if not nn_stack or not isinstance(nn_stack, dict):
            continue

        # Get innermost module path from the stack
        # nn_module_stack is OrderedDict: {path: (qualified_name, module_class), ...}
        module_path = list(nn_stack.keys())[-1] if nn_stack else ""
        # Strip the "L__self__" prefix that torch.export adds (internal representation)
        module_path = re.sub(r"^L__self__[._]?", "", module_path)

        # Get op name from target
        target = node.target
        if hasattr(target, "__name__"):
            op_name = target.__name__
        elif hasattr(target, "_name"):
            op_name = target._name
        else:
            op_name = str(target).split(".")[-1]

        unique_name = graph._graph_namespace.create_name(op_name, node)
        # Build new name: module_path + op_name (dots -> underscores)
        if module_path:
            node.name = f"{module_path}_{unique_name}".replace(".", "_")

    gm.recompile()


def _clean_up_assertions_and_guards(gm: fx.GraphModule):
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
    removed = False
    for node in reversed(graph.nodes):
        if len(node.users) > 0 or not is_op(node, check_ops):
            continue
        graph.erase_node(node)
        removed = True
    for node in reversed(graph.nodes):
        if node.op == "call_module" and (
            str(node.target) == "_guards_fn" or str(node.target).startswith("_guards")
        ):
            # there's typically no users of the guards, but if there are, we route through the first arg
            if len(node.users) > 0 and len(node.args) >= 1:
                node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
            removed = True

    if removed and hasattr(gm, "_guards_fn"):
        delattr(gm, "_guards_fn")
    if removed:
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
    num_moe_experts_for_export: Optional[int] = None,
) -> fx.GraphModule:
    """torch's export with wrapping into GraphModule + useful additions to the resulting module.

    This utility improves over stock torch.export.export in the following aspects:

        1. Provide patches for certain corner cases that torch.export does not support.
        2. Standardize the export process to strictly run on the meta device.
        3. Automatically extract the GraphModule from the exported program.
        4. Retain load hooks for state_dict loading from the original module.
        5. Manage parameter aliasing in the model.
        6. Remove assertions from the graph.
        7. Optionally speed up export for MOE models by tracing with fewer experts
           and expanding the graph afterward.

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
        num_moe_experts_for_export: If set, only this many experts are traced during
            ``torch.export`` (the graph is expanded to include all experts afterward).
            This can dramatically speed up export for large MOE models.
            Recommended value: 2.
    """

    def _capture_fn(model, args, kwargs):
        ep = te.export(model, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict)
        egm = ep.module()
        assert isinstance(egm, fx.GraphModule)
        return egm

    # Optionally reduce MOE experts for faster export tracing
    moe_reductions: List[Dict[str, Any]] = []
    if num_moe_experts_for_export is not None:
        moe_reductions = _reduce_moe_experts(model, num_moe_experts_for_export, args, kwargs)

    # run capture with export
    egm = run_forward_for_capture(
        model, _capture_fn, args, kwargs, clone, patch_list=patch_list, patch_configs=patch_configs
    )

    # Restore full expert lists on the source model and expand the graph to include
    # all expert weights.  This must happen before the load-hook / deduplication
    # post-processing so that those steps see the complete set of parameters.
    if moe_reductions:
        _restore_moe_experts(moe_reductions)
        _expand_moe_experts_in_graph(egm, model, moe_reductions)

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
    _clean_up_assertions_and_guards(egm)

    # Rename nodes to reflect module hierarchy for better debuggability
    _rename_nodes_with_module_hierarchy(egm)

    # show exported graph
    ad_logger.debug("exported graph: " + str(egm))

    return egm


def export_onnx(ad_config: "LlmArgs") -> nn.Module:
    """Export model to ONNX using InferenceOptimizer directly.

    This is a lightweight export path that avoids initializing the full LLM executor,
    which requires KVCacheManager and other runtime components not needed for ONNX export.

    Args:
        ad_config: The AutoDeploy configuration for the model. Should use a mode like
            "export_edgellm_onnx" that includes the export_to_onnx transform.

    Returns:
        The transformed model after running through the inference optimizer pipeline.

    Example:
        >>> from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs
        >>> from tensorrt_llm._torch.auto_deploy.export import export_onnx
        >>>
        >>> ad_config = LlmArgs(
        ...     model="meta-llama/Llama-2-7b-hf",
        ...     mode="export_edgellm_onnx",
        ...     max_batch_size=13,
        ...     max_seq_len=4,
        ...     device="cpu",
        ... )
        >>> ad_config.transforms["export_to_onnx"]["output_dir"] = "/tmp/onnx_output"
        >>> model = export_onnx(ad_config)
    """
    # Import here to avoid circular imports
    from ..shim.interface import CachedSequenceInterface
    from ..transform.optimizer import InferenceOptimizer

    # 1. Create factory from config
    factory = ad_config.create_factory()

    # 2. Create CachedSequenceInterface (lightweight, no KVCacheManager initialization)
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=ad_config.max_seq_len,
        max_batch_size=ad_config.max_batch_size,
        device=ad_config.device,
        kv_cache_config=ad_config.kv_cache_config,
        max_num_tokens=ad_config.max_num_tokens,
        vocab_size_padded=factory.vocab_size_padded,
    )

    # 3. Create InferenceOptimizer with transform config
    inference_optimizer = InferenceOptimizer(
        factory=factory,
        config=ad_config.transforms,
    )

    # 4. Run the transform pipeline (includes export_to_onnx transform)
    return inference_optimizer(cache_seq_interface)
