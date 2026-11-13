"""Restore argument names and READY defaults for loaded MegaMoE functions."""

from inspect import signature


def make_megamoe_aot_callable(function, kernel):
    """Use TVM-FFI's native adapter without attaching scheduler metadata."""
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

    argument_names = list(signature(kernel.__call__).parameters)
    # Operands declared after the READY pair (the NCCL GIN resources, Blackwell's per-token scale
    # and token count) are optional in every entry: omitted means None, exactly as in a JIT call.
    # aot_compile never binds the gin_* resources, so a loaded function always defaults them to None;
    # only the JIT path can pass device communication handles.
    ready_index = argument_names.index("hot_expert_weight_ready_generation")
    assert all(i > ready_index for i, name in enumerate(argument_names) if name.startswith("gin_")), argument_names
    trailing = len(argument_names) - 1 - ready_index
    # Home-only entries can omit the unused READY pair, as with direct JIT
    # calls. Helper-enabled entries require both READY operands explicitly.
    ready = 2 if kernel.helper_expert_count == 0 else 0
    defaults = (None,) * (ready + trailing)
    return make_kwargs_wrapper(function, argument_names, arg_defaults=defaults)
