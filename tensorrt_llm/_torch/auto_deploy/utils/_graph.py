"""Graph-related utilities for transformations."""

import itertools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._export.utils import _detect_fake_mode_from_gm
from torch._prims_common import DeviceLikeType
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._pytree import _LEAF_SPEC, TreeSpec

from .logger import ad_logger
from .node_utils import get_weight_tensor, is_op

_NoValType = type("_NoValType", (), {})


def _call_post_init_recursive(spec: TreeSpec) -> None:
    """Recursively call __post_init__ on TreeSpec starting from leaves.

    This is needed to update internal cached values (like num_leaves) after
    modifying children_specs.
    """
    if hasattr(spec, "children_specs"):
        for child_spec in spec.children_specs:
            _call_post_init_recursive(child_spec)
    spec.__post_init__()


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
        lint(gm)
        recompile(gm)


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


def delete_all_unused_submodules(gm: GraphModule) -> None:
    """Optimized version of delete_all_unused_submodules with O(n+m) complexity.

    The original implementation uses a list for tracking used modules, making membership
    checks O(n). This version uses a set for O(1) lookups.

    Original implementation is at GraphModule.delete_all_unused_submodules

    Args:
        gm: The GraphModule to clean up.
    """
    used: Set[str] = set()

    for node in itertools.chain(
        gm.graph.find_nodes(op="call_module", sort=False),
        gm.graph.find_nodes(op="get_attr", sort=False),
    ):
        # check if it's already used and it's not a call_module node
        # in this case we can skip. We cannot skip if it's a call_module node because we need to
        # mark all recursive submodules as used.
        if node.target in used and node.op != "call_module":
            continue

        # A list of strings representing the different parts
        # of the path. For example, `foo.bar.baz` gives us
        # ["foo", "bar", "baz"]
        fullpath = node.target.split(".")

        # Progressively collect all the names of intermediate
        # modules. For example, if we have the target
        # `foo.bar.baz`, we'll add `foo`, `foo.bar`, and
        # `foo.bar.baz` to the list.
        used.update(".".join(fullpath[:i]) for i in range(1, len(fullpath) + 1))

        # For call_module, also mark all recursive submodules as used
        if node.op == "call_module":
            try:
                submod = gm.get_submodule(node.target)
                for submod_name, _ in submod.named_modules():
                    if submod_name != "":
                        used.add(f"{node.target}.{submod_name}")
            except AttributeError:
                # Node referenced nonexistent submodule, don't need to
                # worry about GCing anything
                pass

    # also add the root module to the used set
    used.add("")

    # Go over all modules and delete if on the list. Since we use named_modules, parents will be
    # deleted first and children will be automatically skipped inside delete_submodule.
    to_delete = [name for name, _ in gm.named_modules() if name not in used]
    for name in to_delete:
        gm.delete_submodule(name)


def eliminate_dead_code(
    gm: GraphModule, is_impure_node: Optional[Callable[[Node], bool]] = None
) -> None:
    """Eliminate dead code from the graph of the given GraphModule."""
    gm.graph.eliminate_dead_code(is_impure_node=is_impure_node)


def recompile(gm: GraphModule) -> None:
    """Recompile the graph of the given GraphModule."""
    gm.recompile()


def lint(gm: GraphModule) -> None:
    """Lint the graph of the given GraphModule."""
    gm.graph.lint()


def _canonicalize_single_gm(gm: GraphModule) -> None:
    # clean up graph (needs to be done repeatedly until no more dead code)
    eliminate_dead_code(gm, is_impure_node=_is_impure_node)

    # recompile to propagate all graph changes to the graph module
    recompile(gm)

    # clean up graph module
    delete_all_unused_submodules(gm)

    # lint the graph
    lint(gm)


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
        with enable_python_dispatcher():
            FakeTensorProp(gm, fake_mode).propagate(*inps)
    else:
        ad_logger.warning("No fake tensors and no args available for shape propagation")

    # lint the graph
    lint(gm)


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
    _call_post_init_recursive(in_spec)

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


def get_input_embeddings(model: nn.Module) -> torch.Tensor:
    """Find the unique embedding node across all graph modules."""
    embedding_weights = []
    for _, gm in named_graphmodules(model):
        found_nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.embedding.default
        )
        for node in found_nodes:
            embedding_weights.append(get_weight_tensor(node))

    if hasattr(model, "get_input_embeddings"):
        embedding_weights.append(model.get_input_embeddings())

    for _, gm in named_graphmodules(model):
        if hasattr(gm, "get_input_embeddings"):
            embedding_weights.append(gm.get_input_embeddings())

    assert len(embedding_weights) > 0, "No embedding weights found"
    unique_embedding_weights = [embedding_weights[0]]
    for weight in embedding_weights:
        if weight is not unique_embedding_weights[0]:
            unique_embedding_weights.append(weight)

    assert len(unique_embedding_weights) == 1, (
        f"Expected exactly 1 unique embedding weight, but found {len(unique_embedding_weights)}."
    )

    return unique_embedding_weights[0]


def get_output_node(model: nn.Module) -> tuple[GraphModule, Node]:
    """Find the unique output node across all graph modules."""
    output_nodes = []
    for _, gm in named_graphmodules(model):
        output_nodes.extend([(gm, node) for node in gm.graph.find_nodes(op="output")])

    assert len(output_nodes) == 1, f"Expected exactly 1 output node, but found {len(output_nodes)}."
    return output_nodes[0]


def get_lm_head_node(gm: GraphModule, output_node: Optional[Node] = None) -> Node:
    if output_node is None:
        output_node = gm.graph.find_nodes(op="output")[0]

    lm_head_node = output_node.all_input_nodes[0]
    if is_op(lm_head_node, torch.ops.aten.to):
        lm_head_node = lm_head_node.all_input_nodes[0]

    return lm_head_node


def get_lm_head_weights(model: nn.Module) -> torch.Tensor:
    gm, output_node = get_output_node(model)
    lm_head_node = get_lm_head_node(gm, output_node)
    return get_weight_tensor(lm_head_node)


def get_attr_by_name(obj, name):
    """Get an attribute specified by a dot-separated path on an object.

    Args:
        obj: The root object from which to resolve the attribute path.
        name (str): Dot-separated attribute path (e.g., "a.b.c").

    Returns:
        The value of the resolved attribute.

    Raises:
        AttributeError: If any component in the path does not exist.
    """
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def set_attr_by_name(obj, name, value):
    """Set an attribute specified by a dot-separated path on an object.

    Args:
        obj: The root object on which to set the attribute.
        name (str): Dot-separated attribute path (e.g., "a.b.c").
        value: The value to assign to the target attribute.

    Raises:
        AttributeError: If any intermediate component in the path does not exist.
    """
    parts = name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def del_attr_by_name(obj, name):
    """Delete an attribute specified by a dot-separated path from an object.

    Args:
        obj: The root object from which to delete the attribute.
        name (str): Dot-separated attribute path (e.g., "a.b.c").

    Raises:
        AttributeError: If any intermediate component in the path does not exist
            or if the final attribute does not exist.
    """
    parts = name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    delattr(obj, parts[-1])


def add_graph_output(gm: GraphModule, output_node: Node, name: str) -> None:
    """Add a graph output to the given GraphModule.

    This function appends a new output to the graph's output node and updates
    the pytree_info metadata accordingly.

    NOTE: function does NOT do any graph canonicalization. This is left to the user!

    Args:
        gm (GraphModule): The GraphModule to add the output to.
        output_node (Node): The node to add as an output.
        name (str): The name of the new output in the output dict.

    Raises:
        RuntimeError: If the graph has no output node or multiple output nodes.

    Implementation Notes (Hacky Details):
        1. **Handling single-output graphs (out_spec == _LEAF_SPEC)**:
           When the graph originally has a single output, its out_spec is a
           _LEAF_SPEC (a leaf node in the pytree). To add a new output, we must
           first convert it to a container spec (tuple with children_specs).

        2. **Forcing output type to dict**:
           The original out_spec.type is typically a model-specific output class
           (e.g., CausalLMOutputWithPast from transformers). When we add new
           outputs with custom names (like "present_key_values_xx"), the original
           class constructor will fail because these names are not valid data
           members of that class.

           PyTorch uses the out_spec.type's constructor to wrap the output:
               result_obj = result_type(**output)

           To avoid constructor failures, we forcibly change out_spec.type to
           dict using object.__setattr__ (since TreeSpec is a frozen dataclass).
           This ensures the output is returned as a plain dictionary instead of
           the original model-specific class.
    """
    # extract graph and output spec
    graph: Graph = gm.graph

    # find the output node
    output_nodes = graph.find_nodes(op="output")
    if len(output_nodes) != 1:
        raise RuntimeError(f"Expected exactly one output node, found {len(output_nodes)}")

    graph_output_node = output_nodes[0]

    # get current outputs
    current_outputs = graph_output_node.args[0]
    if not isinstance(current_outputs, (tuple, list)):
        # single output, convert to tuple
        current_outputs = (current_outputs,)

    # append new output
    new_outputs = tuple(current_outputs) + (output_node,)
    graph_output_node.args = (new_outputs,)

    # update pytree info: append spec for the new output
    # The out_spec structure mirrors the output structure
    # For a tuple of outputs, out_spec should be a TreeSpec with children_specs
    # And graph._codegen.pytree_info is a NamedTuple, so we have to use _replace to replace the out_spec
    out_spec = graph._codegen.pytree_info.out_spec
    assert out_spec is not None, "Graph must have an out_spec"

    if out_spec == _LEAF_SPEC:
        new_out_spec = TreeSpec(type=tuple, children_specs=[_LEAF_SPEC], context=["output"])
        graph._codegen.pytree_info = graph._codegen.pytree_info._replace(out_spec=new_out_spec)
        out_spec = graph._codegen.pytree_info.out_spec
    if hasattr(out_spec, "children_specs"):
        # out_spec is already a container (tuple/list), append to it
        out_spec.children_specs.append(_LEAF_SPEC)
        out_spec.context.append(name)
    else:
        # This shouldn't happen in normal cases, but handle it just in case
        # If out_spec is a single leaf, we need to convert it to a tuple spec
        raise NotImplementedError(
            "Cannot add output to a graph with non-container out_spec. "
            "This case is not currently supported."
        )

    # update pytree info recursively with __post_init__ starting at leaves
    _call_post_init_recursive(out_spec)

    # Change the type of the output spec to dict,
    #
    # NOTE(yocox) This will affect how torch inteprete the output of the graph.
    # The original type is some LLM output result class,
    # for example, CausalLMOutputWithPast, depends on the model.
    # To add new fields to the output, we need to change the type to dict.
    # The type's constructor will be used to create an object containing
    # the result, for example,
    #
    #   result_obj = reuslt_type(**output)
    #
    # Because we added new output's with names like "present_key_values_xx" which
    # is not a data member of CausalLMOutputWithPast, so the constructor will fail.
    # So we need to change the type to dict.
    #
    # However the out_spec.type is frozen dataclass,
    # so we need to use object.__setattr__ to change it.
    object.__setattr__(out_spec, "type", dict)


def remove_graph_input(gm: GraphModule, input_node: Node) -> str:
    """Remove a graph input from the given GraphModule.

    This is the inverse operation of add_graph_input(). It removes a placeholder node
    and updates the pytree_info metadata accordingly. The function automatically detects
    whether the node belongs to args or kwargs and updates the correct spec.

    NOTE: function does NOT do any graph canonicalization. This is left to the user!

    Args:
        gm (GraphModule): The GraphModule to remove the input from.
        input_node (Node): The placeholder node to remove.

    Returns:
        str: The name of the removed node.

    Raises:
        ValueError: If the provided Node is not a placeholder or not in the graph.
        RuntimeError: If the input node is still being used by other nodes.
    """
    graph: Graph = gm.graph

    # validate that the node is a placeholder
    if input_node.op != "placeholder":
        raise ValueError(
            f"Node '{input_node.name}' is not a placeholder node (op='{input_node.op}'). "
            f"Only placeholder nodes can be removed as graph inputs."
        )

    # find all placeholder nodes and validate the node is in this graph
    placeholder_nodes = graph.find_nodes(op="placeholder", sort=True)
    if input_node not in placeholder_nodes:
        raise ValueError(
            f"Node '{input_node.name}' is not found in the graph's placeholder nodes. "
            f"It may belong to a different graph or have already been removed."
        )

    # check that the node is not being used
    if len(input_node.users) > 0:
        user_names = [user.name for user in input_node.users.keys()]
        raise RuntimeError(
            f"Cannot remove input '{input_node.name}' "
            f"because it is still being used by nodes: {user_names}. "
            f"Please remove or replace all usages first."
        )

    # get pytree info
    in_spec = graph._codegen.pytree_info.in_spec
    args_spec = in_spec.children_specs[0]  # tuple for args
    kwargs_spec = in_spec.children_specs[1]  # dict for kwargs
    orig_args = graph._codegen.pytree_info.orig_args

    # determine the global index of the node
    global_idx = placeholder_nodes.index(input_node)

    # determine number of args (kwargs come after args in placeholder order)
    num_args = len(args_spec.children_specs)

    # determine if this is an arg or kwarg, and compute relative index
    if global_idx < num_args:
        # it's an arg
        relative_idx = global_idx
        target_spec = args_spec
        is_kwarg = False
    else:
        # it's a kwarg
        relative_idx = global_idx - num_args
        target_spec = kwargs_spec
        is_kwarg = True

    # save the node name before removing
    removed_node_name = input_node.name

    # remove the node from the graph
    graph.erase_node(input_node)

    # update pytree info: remove the corresponding spec and arg name
    target_spec.children_specs.pop(relative_idx)
    if is_kwarg:
        target_spec.context.pop(relative_idx)
    orig_args.pop(global_idx)

    # update pytree info recursively with __post_init__ starting at leaves
    _call_post_init_recursive(in_spec)

    return removed_node_name
