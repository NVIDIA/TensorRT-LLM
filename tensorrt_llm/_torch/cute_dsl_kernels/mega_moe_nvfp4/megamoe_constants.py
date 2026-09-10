# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared constants for the fused fc1+fc2 MegaMoE path.

``Fp32Max`` is the only CuteDSL-typed constant here; it is resolved
lazily via module ``__getattr__`` (PEP 562) so importing this module --
and therefore the package ``__init__`` and the host-side tactic
enumeration in ``cute_dsl_megamoe_custom_op.py``, which only need the
plain-Python constants -- does not require a cutlass-dsl install. The
kernel modules that consume ``Fp32Max`` (``epilogue_refactor.py``)
import cutlass themselves, so the lazy resolution always succeeds
wherever the constant is actually used.
"""

Nvfp4BlockSize = 16
SfPaddingBlock = 128
TmaLeadingDimByteAlign = 16

Nvfp4E2M1Max = 6.0
Fp8E4M3FNMax = 448.0

Nvfp4E2M1RcpLimit = 1.0 / Nvfp4E2M1Max
Fp8E4M3RcpLimit = 1.0 / Fp8E4M3FNMax

SupportedMmaTileM = (128, 256)
SupportedMmaTileN = (64, 128, 256)


def __getattr__(name: str):
    if name == "Fp32Max":
        from cutlass.cutlass_dsl import Float32

        value = Float32(3.40282346638528859812e38)
        # Cache so later lookups (and ``from ... import Fp32Max``) bind the
        # SAME object instead of re-wrapping per access.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
