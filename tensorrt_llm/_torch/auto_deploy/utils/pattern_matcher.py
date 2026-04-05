# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for registering and applying Torch-Inductor FX-based pattern matchers.

Work with `torch._inductor.pattern_matcher` in PyTorch ≥ 2.7.0.
"""

import contextlib
import inspect
import itertools
import operator
import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Iterable, NoReturn, Optional, Union

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._inductor import lowering
from torch._inductor.pattern_matcher import (
    CallFunction,
    ExclusiveKeywordArg,
    Ignored,
    KeywordArg,
    Match,
    MultiOutputPattern,
    PatternEntry,
    PatternExpr,
    PatternMatcherPass,
    ReplacementPatternEntry,
    T,
    _transfer_meta,
    check_and_add_duplicate_pattern,
    functorch_config,
    gen_pattern_and_search_gm,
    is_match,
    joint_fwd_bwd,
    log_trace_failure,
    stable_topological_sort,
)
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm


@contextlib.contextmanager
def _patch_unsupported_input_tensor():
    """Context manager that patches `unsupported_input_tensor` to bypass meta tensor checks.

    NOTE (lucaslie): Since https://github.com/pytorch/pytorch/pull/150137 the inductor pattern
    matcher will skip over patterns that contain meta tensors. This is a problem for us because we
    use meta tensors for our graph. This patch skips over the check for meta tensors.
    """
    original_fn = lowering.unsupported_input_tensor

    def patched_fn(t: torch.Tensor, *args, **kwargs):
        """Bypass meta tensor check."""
        if t.is_meta:
            return False
        return original_fn(
            t, *args, **kwargs
        )  # a generic pass-through of the arguments to accommodate torch side change

    lowering.unsupported_input_tensor = patched_fn
    try:
        yield
    finally:
        lowering.unsupported_input_tensor = original_fn


class ADPatternMatcherPass(PatternMatcherPass):
    """The off-the-shelf PatternMatcherPass from PyTorch Inductor with some hacks for AutoDeploy."""

    def apply(self, graph: Union[torch.fx.Graph, GraphModule]) -> int:
        """Apply pattern matcher with unsupported_input_tensor patch to bypass meta tensor check."""
        with _patch_unsupported_input_tensor():
            return super().apply(graph)


class _WrapperModule(torch.nn.Module):
    """
    Wraps a stand-alone function into a torch.nn.Module so it can be exported.
    """

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def _trace_to_gm(fn: Callable, args: Sequence[torch.Tensor]) -> GraphModule:
    """Exports a function or Module into a GraphModule via torch_export_to_gm."""
    import torch.utils._python_dispatch as pd
    from torch._guards import detect_fake_mode

    module = fn if isinstance(fn, torch.nn.Module) else _WrapperModule(fn)

    # Convert FakeTensors to meta tensors to avoid mode mismatch with torch.export
    if detect_fake_mode(args) is not None:
        with pd._disable_current_modes():

            def _to_meta(t):
                if not isinstance(t, torch.Tensor):
                    return t
                return torch.empty(tuple(int(s) for s in t.shape), dtype=t.dtype, device="meta")

            args = tuple(_to_meta(t) for t in args)

    return torch_export_to_gm(module, tuple(args))


def _not_implemented(*args: Any, **kwargs: Any) -> NoReturn:
    raise NotImplementedError


def _return_true(match: Match) -> bool:
    return True


class ADReplacementPatternEntry(ReplacementPatternEntry):
    """Pattern replacement entry with local topological repair for multi-output rewrites."""

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        del node
        assert match.replacement_graph is not None
        output_nodes = match.output_nodes()
        self.replace_with_graph(
            match,
            graph,
            match.replacement_graph,
            self.normalize_args(*match.args, **match.kwargs),
        )

        if len(output_nodes) > 1:
            # Torch's generic replacement path inserts the copied replacement graph relative to the
            # earliest matched output node. That is usually fine for single-output rewrites, but it
            # can leave multi-output replacements non-topological: some inputs to the later tuple
            # outputs may still appear later in the linked-list order, or the copied tuple producer
            # and its getitem users can end up reversed. A concrete example is
            # fuse_allreduce_residual_rmsnorm: the fused tuple op may be inserted relative to the
            # earlier residual/add output even though a later RMSNorm weight/cast node is still one
            # of its inputs, or the copied tuple producer may end up after the newly rewritten
            # operator.getitem users that unpack it. Both shapes are dataflow-correct but violate
            # FX's producer-before-consumer linked-list invariant and later fail lint/dead-code
            # elimination with "used before it has been defined". We repair the linked-list order
            # here, at the rewrite site, so only multi-output replacements pay for a full-graph
            # stable sort instead of doing a broader canonicalization-time repair for every graph.
            # TODO: Narrow this to a local reorder over the replacement-affected region instead of
            # sorting the full graph. A bounded repair should start from the inserted replacement
            # nodes plus the rewritten multi-output/getitem chain, then expand only as needed to
            # make that slice dependency-safe before reordering it.
            stable_topological_sort(graph)


def _register_replacement_with_safe_insertion(
    search_fn: Callable,
    replace_fn: Callable,
    example_inputs: Iterable[Any],
    trace_fn: Callable[[Callable, Sequence[torch.Tensor]], GraphModule],
    pass_dicts: Union[PatternMatcherPass, Sequence[PatternMatcherPass]],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: Union[dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
    search_fn_pattern: Union[PatternExpr, None] = None,
    skip_duplicates: bool = False,
) -> bool:
    """Register a replacement rule with topologically safe insertion."""
    argnames_static = [*inspect.signature(search_fn).parameters.keys()]

    def check_fn(match: Match) -> bool:
        argnames = list(argnames_static)
        for name in argnames:
            if name not in match.kwargs:
                raise RuntimeError(
                    "Not all inputs to pattern found in match.kwargs. Perhaps one "
                    f"of the inputs is unused? argnames={argnames}, match.kwargs={match.kwargs}"
                )

        args = list(
            torch.fx.map_arg([match.kwargs[name] for name in argnames], lambda n: n.meta["val"])
        )

        sym_args: list[torch.SymInt] = []
        fake_mode = torch._dynamo.utils.detect_fake_mode(args)
        assert fake_mode is not None
        with fake_mode:
            for idx, grad in enumerate(requires_grad):
                if isinstance(args[idx], torch.Tensor):
                    if grad and torch._prims_common.is_integer_dtype(args[idx].dtype):
                        return False

                    args[idx] = torch.empty_strided(
                        args[idx].size(),
                        args[idx].stride(),
                        dtype=args[idx].dtype,
                        device=args[idx].device,
                        requires_grad=grad,
                    )
                    for value in itertools.chain(args[idx].shape, args[idx].stride()):
                        if isinstance(value, torch.SymInt) and all(
                            torch.fx.experimental.symbolic_shapes.statically_known_true(
                                value != known_value
                            )
                            for known_value in sym_args
                        ):
                            sym_args.append(value)

            specific_pattern = search_fn_pattern

            if not specific_pattern:
                if sym_args:

                    def search_fn_new(*args_new: Any) -> Any:
                        return search_fn(*args_new[len(args_new) - len(args) :])

                    try:
                        specific_graph = trace_fn(search_fn_new, sym_args + args)
                    except RuntimeError as err:
                        log_trace_failure(search_fn, err)
                        return False

                    sym_arg_names = []
                    for idx, placeholder in zip(
                        range(len(sym_args) + len(args)), specific_graph.graph.nodes
                    ):
                        if idx < len(sym_args):
                            sym_arg_names.append(placeholder.target)
                            continue

                        with specific_graph.graph.inserting_after(placeholder):
                            new_node = specific_graph.graph.placeholder(
                                argnames[idx - len(sym_args)]
                            )
                            new_node.target = new_node.name
                            placeholder.replace_all_uses_with(new_node)
                            specific_graph.graph.erase_node(placeholder)

                    argnames = sym_arg_names + argnames
                else:
                    try:
                        specific_graph = trace_fn(search_fn, args)
                    except RuntimeError as err:
                        log_trace_failure(search_fn, err)
                        return False

                specific_pattern = _fx_to_ad_pattern_with_op_ignore(
                    specific_graph,
                    argnames=argnames,
                    scalar_workaround=scalar_workaround,
                    exclusive_arg_names=exclusive_arg_names,
                )

            node = match.output_nodes()[0]
            assert node is not None
            specific_pattern_match = specific_pattern.match(node)

            if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
                match.replacement_graph = trace_fn(replace_fn, args)
                if len(match.nodes) == 1:
                    for replacement_node in match.replacement_graph.graph.nodes:
                        _transfer_meta(
                            replacement_node.meta,
                            match.nodes[0],
                            "replacement",
                        )
                return True
            return False

    def normalize_args(**kwargs: Any) -> list[Any]:
        args = [kwargs.pop(name) for name in argnames_static]
        for idx in range(1, len(kwargs) + 1):
            tangent_name = f"tangents_{idx}"
            if tangent_name not in kwargs:
                break
            args.append(kwargs.pop(tangent_name))
        assert not kwargs, f"leftover kwargs: {kwargs!r}"
        return args

    if trace_fn is joint_fwd_bwd and torch.is_inference_mode_enabled():
        return False

    with functorch_config.patch(functionalize_rng_ops=False):
        requires_grad: list[bool] = [
            isinstance(example_input, torch.Tensor) and example_input.requires_grad
            for example_input in example_inputs
        ]
        if search_fn_pattern is None:
            pattern, gm = gen_pattern_and_search_gm(
                search_fn,
                example_inputs,
                trace_fn,
                scalar_workaround,
                exclusive_arg_names,
            )
        else:
            pattern = search_fn_pattern
            gm = None

        pattern_passes = pass_dicts if isinstance(pass_dicts, Sequence) else [pass_dicts]
        for pattern_matcher_pass in pattern_passes:
            if isinstance(pattern_matcher_pass, PatternMatcherPass):
                if check_and_add_duplicate_pattern(
                    pattern,
                    gm.graph if gm else None,
                    pattern_matcher_pass.seen_patterns,
                    skip_duplicates=skip_duplicates,
                ):
                    return False

        pattern_entry: PatternEntry = ADReplacementPatternEntry(
            pattern=pattern,
            extra_check=check_fn,
            normalize_args=normalize_args,
        )
        pattern_entry.register(pass_dicts)
        return pattern_entry.pattern  # type: ignore[return-value]


def register_ad_pattern(
    search_fn: Callable,
    replace_fn: Callable,
    patterns: PatternMatcherPass,
    dummy_args: Iterable[Any],
    trace_fn: Callable[[Callable, Sequence[torch.Tensor]], GraphModule] = _trace_to_gm,
    # optional args input to our variants of `fx_to_pattern`
    ignore_types: Sequence[type[Any]] = (),
    op_ignore_types: Optional[Mapping[Callable[..., Any], Sequence[type[Any]]]] = None,
    scalar_workaround: Optional[Any] = None,
    exclusive_arg_names: Sequence[str] = (),
    # optional args input to torch._inductor's register_replacement
    extra_check: Callable[[Match], bool] = _return_true,
    skip_duplicates: bool = False,
) -> None:
    """
    Tracing a Python-level pattern into a GraphModule and registering its replacement.

    Args:
        search_fn:          Function defining the “before” pattern (traced for matching).
        replace_fn:         Function or op defining the “after” replacement.
        patterns:           PatternMatcherPass instance (passed to register_replacement).
        dummy_args:         Inputs matching search_fn's signature used to generate
                            the FX pattern (must reflect any real dims materialized
                            as aten-op literals).
        trace_fn:           Callable to export Python functions/modules into FX (e.g.
                            torch_export_to_gm).
        ignore_types:       Literal value types to ignore when converting an FX graph to a pattern.
        op_ignore_types:    Per-operator mapping of argument types to ignore during pattern matching.
        scalar_workaround:  Optional dict or value to workaround FX scalar lifting bugs.

        exclusive_arg_names:Names of pattern inputs that must match exactly (not treated as ignored).
        extra_check:        Additional run on each `Match` to decide if it should be replaced.
        skip_duplicates:    If True, don't re-register a pattern that's already been seen.

    Note:
    1. Tensor arguments—both the inputs you pass into the pattern and any tensor args inside its ops
    are matched purely by their place in the graph; their shape, device, and dtype are not validated.
    2. Your `dummy_args` may use any shape, device, or dtype so long as they successfully trace
        the pattern and do not alter the resulting FX graph.
    3. However, `dummy_args` *can* accidentally bake in layout or type information.
    To guard against this:
       a. If a shape value ends up as a literal in calls like `aten.view`, `aten.slice`,
       or `aten.reshape`, add those ops to `op_ignore_types` so their int args are ignored.
       b. If your pattern does explicit `to(device=…)` or `to(dtype=…)` calls,
       consider skipping those args in `op_ignore_types` too
       c. If your pattern includes `aten.to.device` or `aten.to.dtype`,
       FX may automatically drop that node when the dummy's device/dtype already matches the target
       altering your graph topology. Consider registering a separate pattern to handle this case.
    4. Numeric (integer/float) args input to the pattern will be lifted as hard-coded literals
        in the FX graph, utilize `scalar_workaround` to detect those literals and
        replace with args in the pattern
        - The literal value you choose must exactly match the default kwarg in your search function.
        - FX will treat every literal with that same value as an occurrence of your kwarg,
        so ensure no other ops in the pattern use that same number,
        otherwise pick a different workaround value.
    5. register_replacement can auto-generate `search_fn_pattern` if you input `example_inputs`,
        but that approach will fail when symbolic shapes are involved. Here
        we explicitly trace & convert via `fx_to_pattern`.
    6. The PatternMatcherPass would check num_users of the nodes, meaning that the pattern is required
        to be functionally isolated, no intermediate nodes are shared with the rest of the graph.

    """
    dummy_args = tuple(dummy_args)
    argnames = list(inspect.signature(search_fn).parameters.keys())
    specific_gm = trace_fn(search_fn, dummy_args)
    pattern = _fx_to_ad_pattern_with_op_ignore(
        specific_gm,
        argnames=argnames,
        ignore_types=ignore_types,
        op_ignore_types=op_ignore_types,
        exclusive_arg_names=exclusive_arg_names,
        scalar_workaround=scalar_workaround,
    )

    _register_replacement_with_safe_insertion(
        search_fn=search_fn,
        replace_fn=replace_fn,
        example_inputs=dummy_args,
        trace_fn=trace_fn,
        pass_dicts=patterns,
        search_fn_pattern=pattern,
        extra_check=extra_check,
        skip_duplicates=skip_duplicates,
    )


#   Copied from torch._inductor.pattern_matcher.fx_to_pattern
#   with additional argument `op_ignore_types` that ignore certain types of arg for certain op type
#   When a shape dimension is materialized as a literal argument to an aten op (e.g., slice or view);
#   it's helpful to ignore matching args of these ops, so that the corresponding dimension of your dummy_args
#   doesn't need to match the real runtime value
def _fx_to_ad_pattern_with_op_ignore(
    gm: Union[torch.fx.GraphModule, torch.fx.Graph],
    ignore_types: Sequence[type[Any]] = (),
    op_ignore_types: Optional[Mapping[Callable[..., Any], Sequence[type[Any]]]] = None,
    argnames: Sequence[str] = (),
    scalar_workaround: Union[dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
) -> PatternExpr:
    """
    Convert an FX graph into a PatternExpr.  This is useful for simple
    patterns that can only match single functions and fixed-length lists.
    """
    # scalar_workaround is a hack to capture dropout_p
    # see https://github.com/pytorch/pytorch/issues/97894
    scalar_workaround = scalar_workaround or {}
    inv_scalar_workaround = {v: k for k, v in scalar_workaround.items()}
    assert len(inv_scalar_workaround) == len(scalar_workaround)

    def process_arg(
        x: T, ignore_types_override: Optional[Sequence[type[Any]]] = None
    ) -> Union[T, KeywordArg, Ignored]:
        current_ignore_types = (
            ignore_types_override if ignore_types_override is not None else ignore_types
        )
        if isinstance(x, (float, int)) and x in inv_scalar_workaround:
            return KeywordArg(inv_scalar_workaround[x])
        if type(x) in current_ignore_types:
            return Ignored()
        if isinstance(x, list) and all(isinstance(y, Ignored) for y in x) and x:
            return Ignored()
        return x

    argnum = itertools.count()

    class Converter(torch.fx.Interpreter):
        call_method = _not_implemented
        call_module = _not_implemented
        get_attr = _not_implemented

        def placeholder(
            self,
            target: str,
            args: Sequence[Any],
            kwargs: Mapping[str, Any],  # type: ignore[override]
        ) -> Union[ExclusiveKeywordArg, KeywordArg]:
            n = next(argnum)
            if n < len(argnames):
                name = argnames[n]
            elif argnames:
                assert target.startswith("tangent")
                name = target
            else:
                target = re.sub(r"_\d+$", "", target)  # de-mangle arg name
                name = target
            if name in exclusive_arg_names:
                return ExclusiveKeywordArg(name)
            else:
                return KeywordArg(name)

        def call_function(
            self, target: Any, args: Sequence[Any], kwargs: Mapping[str, Any]
        ) -> PatternExpr:
            # build a per-op ignore list if specified
            if op_ignore_types and target in op_ignore_types:
                combined_ignore = tuple(ignore_types) + tuple(op_ignore_types[target])
            else:
                combined_ignore = None

            # by default use the global or per-op ignore set
            def _make_arg_fn(ignore_override: Optional[Sequence[type[Any]]]):
                return lambda x: process_arg(x, ignore_override)

            # special-case getitem so ints still match as indices
            if target == operator.getitem:
                # drop all ignores except int
                ints_only = tuple(t for t in (combined_ignore or ignore_types) if t is not int)
                process_arg_fn = _make_arg_fn(ints_only)
            else:
                process_arg_fn = _make_arg_fn(combined_ignore)

            args, kwargs = pytree.tree_map(process_arg_fn, (args, kwargs))
            if list in ignore_types:
                # handle burned-in tensor size lists of Ignored()
                args = [process_arg_fn(a) for a in args]
                kwargs = {k: process_arg_fn(v) for k, v in kwargs.items()}

            return CallFunction(target, *args, **kwargs)

        def run_node(self, n: torch.fx.Node) -> Any:
            rv = super().run_node(n)
            if n.op == "output" and isinstance(rv, tuple):
                assert len(rv) == len(n.args[0])  # type: ignore[arg-type]
                for r, arg in zip(rv, n.args[0]):  # type: ignore[arg-type]
                    r.users = len(arg.users)
            else:
                rv.users = len(n.users)
            return rv

    pattern = Converter(gm).run()  # type: ignore[arg-type]
    if not isinstance(pattern, PatternExpr):
        return MultiOutputPattern(pytree.tree_leaves(pattern))
    return pattern
