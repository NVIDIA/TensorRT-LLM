"""
Utilities for registering and applying Torch-Inductor FX-based pattern matchers.

Work with `torch._inductor.pattern_matcher` in PyTorch ≥ 2.7.0.
"""

import inspect
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, fx_to_pattern, register_replacement
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export_to_gm


class WrapperModule(torch.nn.Module):
    """
    Wraps a stand-alone function into a torch.nn.Module so it can be exported.
    """

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def trace_to_gm(fn: Callable, args: Sequence[torch.Tensor]) -> GraphModule:
    """
    Exports a function or Module into a GraphModule via torch_export_to_gm.
    """
    module = fn if isinstance(fn, torch.nn.Module) else WrapperModule(fn)
    return torch_export_to_gm(module, tuple(args))


def register_pattern(
    search_fn: Callable,
    replace_fn: Callable,
    patterns: PatternMatcherPass,
    example_inputs: List[torch.Tensor],
    dummy_args: List[torch.Tensor],
    trace_fn: Callable[[Callable, Sequence[torch.Tensor]], GraphModule] = trace_to_gm,
    exclusive_arg_names: Sequence[str] = (),
    scalar_workaround: Optional[Any] = None,
) -> None:
    """
    Tracing a Python-level pattern into a GraphModule and registering its replacement.

    The inductor matcher treats tensor-shaped arguments and numeric (literal) arguments differently:

    1. Tensor inputs (both pattern inputs and aten-op arguments):
       - Their shapes do not need to match the example_inputs exactly, except when a shape
         dimension is materialized as a literal argument to an aten op (e.g., slicing or splitting);
         in that case the corresponding dimension of your dummy_args must match the real runtime value
         so the matcher recognizes it.
       - Symbolic dimensions may be arbitrary (use small fake sizes).

    2. Numeric (integer/float) args (e.g., `unsqueeze_dim=1`):
       - These become hard-coded literals in the FX graph (e.g. `torch.ops.aten.unsqueeze.default(cos, 1)`).
       - The matcher requires an exact match on these literals, so you must register one pattern
         per distinct value.  (Torch ≥ 2.7.0 is required to handle multiple patterns that differ
         only by numeric literals.)

    Note:
      register_replacement can auto-generate `search_fn_pattern` if you omit it,
      but that approach will fail when symbolic shapes are involved. Here
      we explicitly trace & convert via `fx_to_pattern`.

    Args:
        search_fn:          Function defining the “before” pattern (traced for matching).
        replace_fn:         Function or op defining the “after” replacement.
        patterns:           PatternMatcherPass instance (passed to register_replacement).
        example_inputs:     Example inputs for register_replacement's initial trace.
        dummy_args:         Inputs matching search_fn's signature used to generate
                            the FX pattern (must reflect any real dims materialized
                            as aten-op literals).
        trace_fn:           Callable to export Python functions/modules into FX (e.g.
                            torch_export_to_gm).
        exclusive_arg_names:
                            Parameter names to ignore when matching.
        scalar_workaround:  Optional dict or value to workaround FX scalar lifting bugs.
    """
    argnames = list(inspect.signature(search_fn).parameters.keys())
    specific_gm = trace_fn(search_fn, dummy_args)
    pattern = fx_to_pattern(
        specific_gm,
        argnames=argnames,
        exclusive_arg_names=exclusive_arg_names,
        scalar_workaround=scalar_workaround,
    )

    register_replacement(
        search_fn=search_fn,
        replace_fn=replace_fn,
        example_inputs=example_inputs,
        trace_fn=trace_fn,
        pass_dicts=patterns,
        search_fn_pattern=pattern,
    )
