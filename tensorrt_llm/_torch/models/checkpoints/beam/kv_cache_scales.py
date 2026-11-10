# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 factor validation shared by Beam checkpoint loading and attention."""

import torch


def fp8_kv_scale_pairs(scales, what: str, device=None) -> tuple:
    """The FP32 tensors ``(s, 1/s)`` of the per-layer FP8 KV-cache scales ``scales``.

    They are the factors the op reads (``code * s``) and writes (``x * (1/s)``) with. Raises
    ValueError naming the first unusable layer when its FP32 ``s`` or ``1/s`` is zero,
    subnormal or not finite: a value such as 1e-300, 1e300 or 1e-40 is a finite positive
    Python float, yet rounds to such a factor and would erase or overflow every stored value.
    ``what`` names the scales in the message.
    """
    values = [float(s) for s in scales]
    s = torch.tensor(values, dtype=torch.float32, device=device)
    inv = torch.ones_like(s) / s
    tiny = torch.finfo(torch.float32).tiny
    usable = torch.isfinite(s) & (s >= tiny) & torch.isfinite(inv) & (inv >= tiny)
    for layer, ok in enumerate(usable.tolist()):
        if not ok:
            raise ValueError(
                f"{what}: layer {layer} scale {values[layer]!r} is unusable in FP32 (s = "
                f"{float(s[layer])!r}, 1/s = {float(inv[layer])!r}; both must be finite and normal)"
            )
    return s, inv
