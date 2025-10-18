"""Graph-related utilities for transformations."""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import fx
from torch._export.utils import _detect_fake_mode_from_gm
from torch._prims_common import DeviceLikeType
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._pytree import _LEAF_SPEC

from .logger import ad_logger
from .node_utils import is_op

_NoValType = type("_NoValType", (), {})
_NO_VAL = _NoValType()


def get_buffers_and_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Store all buffers and parameters in the model in a dictionary."""
    buffers = {k: v for k, v in model.named_buffers(remove_duplicate=False)}
    params = {k: v for k, v in model.named_parameters(remove_duplicate=False)}
    return {**buffers, **params}


def load_buffers_and_params(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict_missing: bool,
    strict_unexpected: bool,
    clone: bool = False,
):
    """Load all buffers and parameters from the state_dict into the model.

    We control separately whether we allow for missing/unexpected keys.

    missing = keys in the model that are not in the state_dict
    unexpected = keys in the state_dict that are not in the model
    """
    keys_expected = get_buffers_and_params(model).keys()
    keys_missing = keys_expected - state_dict.keys()
    keys_unexpected = state_dict.keys() - keys_expected

    error_msg = ""
    if strict_missing and keys_missing:
        error_msg += f"Missing keys: {keys_missing}"
    if strict_unexpected and keys_unexpected:
        error_msg += f"Unexpected keys: {keys_unexpected}"
    if error_msg:
        raise RuntimeError(error_msg)

    for k in keys_expected - keys_missing:
        sub_name, _, name = k.rpartition(".")
        submod = model.get_submodule(sub_name)
        v = state_dict[k]
        if clone:
            v_new = v.detach().clone()
            if isinstance(v, torch.nn.Parameter):
                v_new = nn.Parameter(v_new, requires_grad=False)
        else:
            v_new = state_dict[k]
        setattr(submod, name, v_new)


def tree_to(tree: pytree.PyTree, *args, **kwargs):
    """Try to recursively move the pytree to the specified args/kwargs."""
    return pytree.tree_map(
        lambda t: t.to(*args, **kwargs) if isinstance(t, torch.Tensor) else t, tree
    )


@contextmanager
def lift_to_meta(model: nn.Module, strict_missing: bool = True, strict_unexpected: bool = True):
    """Temporarily lift all parameters and buffers to the meta device."""
    # lift state_dict
    state_dict = get_buffers_and_params(model)
    model.to("meta")

    # yield the state_dict
    try:
        yield state_dict
    finally:
        # unlift state_dict
        load_buffers_and_params(
            model, state_dict, strict_missing=strict_missing, strict_unexpected=strict_unexpected
        )


def named_graphmodules(gm: fx.GraphModule) -> Iterator[Tuple[str, fx.GraphModule]]:
    """Yield (name, submodule) for every fx.GraphModule inside gm (including gm itself)."""
    for name, m in gm.named_modules():
        if isinstance(m, fx.GraphModule):
            yield name, m


def _move_single_gm_to_device(gm: GraphModule, device: torch.device) -> None:
    """Move one GraphModule and its nodes to the specified device in-place.
    Partially inspired by https://github.com/pytorch/pytorch/blob/05cb98f91d49df9eadfcb3fc29bbd1b621d88860/torch/export/passes/__init__.py#L11
    """
    # move state dict
    gm.to(device)
    recompile_graph = False

    for node in gm.graph.nodes:
        # move all the nodes kwargs with burnt-in device
        if "device" in node.kwargs:
            recompile_graph = True
            kwargs = node.kwargs.copy()
            kwargs["device"] = device
            node.kwargs = kwargs

        if is_op(node, torch.ops.aten.to.device):
            recompile_graph = True
            args = list(node.args)
            args[1] = device
            node.args = tuple(args)

        # move all the tensor metadata
        node.meta["val"] = pytree.tree_map(
            lambda v: v.to(device) if isinstance(v, torch.Tensor) else v,
            node.meta.get("val"),
        )
    if recompile_graph:
        # recompile graph to update self generated codes in subgraph
        gm.graph.lint()
        gm.recompile()


def move_to_device(mod: nn.Module, device: DeviceLikeType) -> None:
    """Move the entire graph module and all sub-GraphModules to the specified device."""
    # get device
    device = torch.device(device)

    # move the model to the device
    mod.to(device)

    for _, subgm in reversed(list(named_graphmodules(mod))):
        # recompile graph to update self generated codes in subgraph
        _move_single_gm_to_device(subgm, device)


def _is_impure_node(node: Node) -> bool:
    """We use the default is_impure function for the node but avoid RNG check."""
    # temporarily disable RNG check
    is_set_to_true = False
    if getattr(node.target, "_nondeterministic_seeded", False):
        is_set_to_true = True
        node.target._nondeterministic_seeded = False
    try:
        return node.is_impure()
    finally:
        # restore RNG check
        if is_set_to_true:
            node.target._nondeterministic_seeded = True


def _canonicalize_single_gm(gm: GraphModule) -> None:
    # clean up graph (needs to be done repeatedly until no more dead code)
    gm.graph.eliminate_dead_code(is_impure_node=_is_impure_node)

    # recompile to propagate all graph changes to the graph module
    gm.recompile()

    # clean up graph module
    gm.delete_all_unused_submodules()

    # lint the graph
    gm.graph.lint()


def canonicalize_graph(mod: nn.Module) -> None:
    """Canonicalize the graph of the given GraphModule.

    Args:
        mod: The model containing GraphModules to canonicalize.
    Returns:
        The canonicalized (cleaned-up) model.
    """
    ad_logger.debug(f"Before canonicalizing: {mod}")

    for _, subgm in reversed(list(named_graphmodules(mod))):
        _canonicalize_single_gm(subgm)

    ad_logger.debug(f"After canonicalizing: {mod}")


def _run_shape_prop_single_gm(
    gm: GraphModule,
    args_static: Optional[Tuple[Any, ...]] = None,
) -> None:
    fake_mode: Optional[FakeTensorMode] = _detect_fake_mode_from_gm(gm)

    # get fake tensors from placeholder nodes
    # NOTE: use _NO_VAL instead of None since None is a valid value for node.meta["val"]
    inps = [node.meta.get("val", _NO_VAL) for node in gm.graph.nodes if node.op == "placeholder"]

    # check if we need to use args to create fake tensors
    if any(inp is _NO_VAL for inp in inps):
        if args_static is not None and fake_mode is not None and len(args_static) == len(inps):
            inps = [
                inp if inp is not _NO_VAL else fake_mode.from_tensor(arg, static_shapes=True)
                for inp, arg in zip(inps, args_static)
            ]

    # run shape propagation if we have all the fake tensors
    if all(inp is not _NO_VAL for inp in inps):
        FakeTensorProp(gm, fake_mode).propagate(*inps)
    else:
        ad_logger.warning("No fake tensors and no args available for shape propagation")

    # lint the graph
    gm.graph.lint()


def run_shape_prop(
    mod: nn.Module,
    args_static: Optional[Tuple[Any, ...]] = None,
) -> None:
    """Run FakeTensor-based shape propagation on the given GraphModule and its submodules.

    This pass attempts to populate shape/type metadata for all nodes by propagating
    FakeTensor inputs through the graph. If a placeholder node already has a
    ``node.meta["val"]`` entry, that FakeTensor will be used. Otherwise, if
    ``args_static`` is provided and a FakeTensorMode is detected, new FakeTensors
    are synthesized from the static arguments.

    Args:
        mod: The top-level model containing GraphModules on which to run shape propagation. All
            nested GraphModules are processed in reverse topological order.
        args_static: Optional tuple of concrete tensors used to create FakeTensors
            when placeholder metadata is missing. Only applied to the top-level
            GraphModule; submodules reuse their existing placeholder metadata.

    """
    ad_logger.debug(f"Before running shape propagation: {mod}")

    for _, subgm in reversed(list(named_graphmodules(mod))):
        _run_shape_prop_single_gm(subgm, args_static=args_static if subgm is mod else None)

    ad_logger.debug(f"After running shape propagation: {mod}")


def add_graph_input(
    gm: GraphModule,
    name: str,
    add_kwargs: bool = True,
    val: Union[Optional[torch.Tensor], _NoValType] = _NO_VAL,
    dynamic_shape=None,
) -> Node:
    """Add a graph input to the given GraphModule and return the newly created node.

    NOTE: function does NOT do any graph canonicalization. This is left to the user!

    Args:
        gm (GraphModule): The GraphModule to add the input to.
        name (str): The name of the input.
        add_kwargs (bool): Whether to add an arg or kwarg to the graph inputs.
        val (torch.Tensor): An example tensor to use for the input.
        dynamic_shape: The dynamic shape of the input tensor [NOT SUPPORTED YET]
    """
    # check that no dynamic shape is provided...
    if dynamic_shape:
        raise NotImplementedError("Dynamic shape not supported for adding graph inputs")

    # extract graph and input spec
    graph: Graph = gm.graph

    orig_args = graph._codegen.pytree_info.orig_args

    in_spec = graph._codegen.pytree_info.in_spec
    in_spec_for_args = in_spec.children_specs[1 if add_kwargs else 0]
    if add_kwargs:
        assert in_spec_for_args.type is dict
    else:
        assert in_spec_for_args.type is tuple

    # insert input node after currently last input node
    node_last_input = graph.find_nodes(op="placeholder", sort=True)[-1]
    with graph.inserting_after(node_last_input):
        in_node = graph.placeholder(name)
        in_spec_for_args.children_specs.append(_LEAF_SPEC)
        orig_args.append(name)

        if add_kwargs:
            in_spec_for_args.context.append(name)

    # update pytree info recursively with __post_init__ starting at leaves
    def call_post_init(spec):
        for child_spec in spec.children_specs:
            call_post_init(child_spec)
        spec.__post_init__()

    call_post_init(in_spec)

    # set fake tensor information if all required information is available
    fake_mode: Optional[FakeTensorMode] = _detect_fake_mode_from_gm(gm)
    if fake_mode and val is not _NO_VAL:
        if val is None:
            in_node.meta["val"] = None
        else:
            fake_tensor: FakeTensor = fake_mode.from_tensor(val, static_shapes=True)
            in_node.meta["val"] = fake_tensor
            in_node.meta["tensor_meta"] = _extract_tensor_metadata(fake_tensor)

    # return new node...
    return in_node


def placeholders_on_meta(mod: nn.Module) -> bool:
    """
    Return True if any placeholder node in the graph is on the meta device.
    """

    def _is_meta_tensor(t) -> bool:
        if t is None:
            return False
        # First check device.type == 'meta'
        dev = getattr(t, "device", None)
        if getattr(dev, "type", None) == "meta":
            return True
        # Fallback for objects with .is_meta attribute
        return bool(getattr(t, "is_meta", False))

    for _, subgm in reversed(list(named_graphmodules(mod))):
        for n in subgm.graph.nodes:
            if n.op != "placeholder":
                continue
            val = n.meta.get("val", None)

            # If placeholder packs multiple values, find the first tensor-like leaf
            t = val
            if isinstance(val, (list, tuple)):
                t = next((x for x in val if hasattr(x, "device") or hasattr(x, "is_meta")), None)

            if _is_meta_tensor(t):
                return True

    return False
