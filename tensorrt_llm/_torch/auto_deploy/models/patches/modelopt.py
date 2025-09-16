from __future__ import annotations

from typing import Tuple

import torch


# TODO: rename and move to custom op folder
# Adapted from modelopt.torch.quantization.tensor_quant._tensor_quant
# This patch is needed for two reasons:
# 1. original tensor_quant_legacy() calls .item() and create new tensor on the fly. This breaks torch.export
# 2. inplace operate round_() does not work with the pattern matcher
@torch.library.custom_op("auto_deploy::tensor_quant_legacy", mutates_args=())
def tensor_quant_legacy(
    inputs: torch.Tensor,
    amax: torch.Tensor,
    num_bits: int = 8,
    unsigned: bool = False,
    narrow_range: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Export-friendly replacement for _tensor_quant:
      - No torch.tensor(..., device=...)
      - Uses *_like / tensor arithmetic instead
      - Keeps the pieces you actually exercise
    Returns:
      (outputs, scale)
    """
    if unsigned and inputs.min() < 0.0:
        raise TypeError("Negative values encountered in unsigned quantization.")

    input_dtype = inputs.dtype
    if inputs.dtype == torch.half:
        inputs = inputs.float()
    if amax.dtype == torch.half:
        amax = amax.float()

    power = (2.0 ** (num_bits - 1 + int(unsigned))) - 1.0
    max_bound = torch.ones_like(amax) * power
    if unsigned:
        min_bound = 0
    elif narrow_range:
        min_bound = -max_bound
    else:
        min_bound = -max_bound - 1

    scale = max_bound / amax

    outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

    if input_dtype == torch.half:
        outputs = outputs.half()

    return outputs, scale


@tensor_quant_legacy.register_fake
def tensor_quant_legacy_fake(
    inputs: torch.Tensor,
    amax: torch.Tensor,
    num_bits: int = 8,
    unsigned: bool = False,
    narrow_range: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(inputs)
    scl = torch.empty_like(amax)
    return out, scl


# TODO: move this to the apply patch system
def apply_patch() -> None:
    """
    Monkey-patch modelopt.torch.quantization.tensor_quant._tensor_quant so the
    legacy CPU path in FakeTensorQuantFunction.forward uses our export-safe custom op.
    """
    try:
        import modelopt.torch.quantization.tensor_quant as tq
    except Exception:
        # raise RuntimeError(
        #     "Failed to import modelopt.torch.quantization.tensor_quant; "
        #     "ensure ModelOpt is on PYTHONPATH."
        # ) from e
        pass

    orig = getattr(tq, "_tensor_quant", None)

    def _tensor_quant_export(
        inputs: torch.Tensor,
        amax: torch.Tensor,
        num_bits: int = 8,
        unsigned: bool = False,
        narrow_range: bool = True,
    ):
        return torch.ops.auto_deploy.tensor_quant_legacy.default(
            inputs, amax, int(num_bits), bool(unsigned), bool(narrow_range)
        )

    setattr(tq, "_tensor_quant", _tensor_quant_export)

    setattr(tq, "_tensor_quant__original", orig)


def remove_patch() -> None:
    """Optional helper to restore the original _tensor_quant if needed."""
    import modelopt.torch.quantization.tensor_quant as tq

    orig = getattr(tq, "_tensor_quant__original", None)
    if orig is not None:
        setattr(tq, "_tensor_quant", orig)


apply_patch()
