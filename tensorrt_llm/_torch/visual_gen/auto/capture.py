# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""torch.export wrapper for the auto path.

Thin wrapper around `torch.export.export(strict=False, dynamic_shapes=...)`.
The family adapter supplies the example-input and dynamic-shape spec; this
module just runs `torch.export` against them, with optional CFG-parallel
example slicing so the captured graph operates at the per-rank batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from ..config import DiffusionModelConfig
    from .adapter import VisGenFamilyAdapter


_COMPUTE_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


def _infer_module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Pick a (device, compute-dtype) pair for example-input construction.

    For quantized models the first parameter may be FP8/NVFP4/INT — types that
    `torch.randn` does not support. We pick the *compute* dtype (BF16/FP16/FP32)
    that the model's forward actually accepts as input, not the weight storage
    dtype. Walk parameters/buffers for the first floating-point compute dtype.
    """
    device: Optional[torch.device] = None
    for p in module.parameters():
        if device is None:
            device = p.device
        if p.dtype in _COMPUTE_DTYPES:
            return p.device, p.dtype
    for b in module.buffers():
        if device is None:
            device = b.device
        if b.dtype in _COMPUTE_DTYPES:
            return b.device, b.dtype
    return device or torch.device("cpu"), torch.get_default_dtype()


def capture_transformer(
    module: nn.Module,
    adapter: "VisGenFamilyAdapter",
    cfg: "DiffusionModelConfig",
    cfg_size: int = 1,
) -> torch.export.ExportedProgram:
    """Capture `module` to an `ExportedProgram` using `torch.export`.

    Tensors are materialized on `module`'s device with `module`'s parameter
    dtype. The adapter supplies kwargs and dynamic-shape spec.

    When ``cfg_size > 1`` (CFG-parallel), the example-input batch is sliced
    to ``B // cfg_size`` so the captured graph operates on the per-rank
    batch the wrapper's CFG-split feeds it at runtime. Static-B adapters
    (PixArt-Σ, Sana) end up with the captured graph specialized at
    ``B_example / cfg_size``; dynamic-B adapters (FLUX, FLUX.2, SD3, WAN,
    LTX, LTX-2) re-specialize via ``Dim.AUTO`` / dynamic ``Dim``.

    Returns the `ExportedProgram`. Caller is responsible for downstream
    `run_decompositions()` / FX rewrites.
    """
    device, dtype = _infer_module_device_dtype(module)
    args, kwargs = adapter.example_inputs(cfg, device, dtype)
    dynamic_shapes = adapter.dynamic_shapes(cfg)

    if cfg_size > 1:

        def _slice_b(t):
            if not isinstance(t, torch.Tensor) or t.ndim == 0:
                return t
            B = t.shape[0]
            if B % cfg_size != 0 or B // cfg_size < 1:
                return t
            return t.narrow(0, 0, B // cfg_size).contiguous()

        args = tuple(_slice_b(a) for a in args)
        kwargs = {k: _slice_b(v) for k, v in kwargs.items()}

        # After slicing, any tensor whose dim-0 is now 1 will trip
        # torch.export's `ConstraintViolationError: Constraints violated (B)
        # — you marked B as dynamic but your code specialized it to (1)`
        # because the model code's `B == 1` branches force specialization.
        # Strip the dim-0 entry from each kwarg's dynamic_shapes spec
        # whenever its tensor's dim-0 is now 1.
        def _maybe_drop_dim0(t, spec):
            if not isinstance(t, torch.Tensor) or t.ndim == 0:
                return spec
            if t.shape[0] != 1 or not isinstance(spec, dict):
                return spec
            return {k: v for k, v in spec.items() if k != 0}

        if isinstance(dynamic_shapes, dict):
            dynamic_shapes = {
                k: _maybe_drop_dim0(kwargs.get(k), v) for k, v in dynamic_shapes.items()
            }

    logger.info(
        f"VisGen-Auto capture: family={adapter.family}, "
        f"device={device}, dtype={dtype}, "
        f"kwarg shapes={
            {k: tuple(v.shape) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        }"
    )

    with torch.no_grad():
        ep = torch.export.export(
            module,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    return ep
