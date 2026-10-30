"""Restore argument names and READY defaults for loaded MegaMoE functions."""

from inspect import signature


def make_megamoe_aot_callable(function, kernel):
    """Use TVM-FFI's native adapter without attaching scheduler metadata."""
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

    argument_names = list(signature(kernel.__call__).parameters)
    # Home-only entries can omit the unused READY pair, as with direct JIT
    # calls. Helper-enabled entries require both READY operands explicitly.
    # Kernels that take the trailing optional fc1_activation_per_token_scale
    # may omit it too (defaults are right-aligned, hence its position last).
    trailing_optional = 1 if argument_names[-1] == "fc1_activation_per_token_scale" else 0
    ready_optional = 2 if kernel.helper_expert_count == 0 else 0
    defaults = (None,) * (ready_optional + trailing_optional)
    return make_kwargs_wrapper(function, argument_names, arg_defaults=defaults)
