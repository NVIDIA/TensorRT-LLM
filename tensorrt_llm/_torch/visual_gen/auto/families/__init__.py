# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family adapters for VisGen-Auto.

Each adapter declares the example inputs, dynamic shapes, and rewrite policy
for one Diffusers transformer family. The base contract is
``VisGenFamilyAdapter`` in ``../adapter.py``.
"""

from .flux import FluxAdapter
from .flux2 import Flux2Adapter
from .ltx import LTXAdapter
from .ltx2 import LTX2Adapter
from .mmdit import MMDiTAdapter
from .pixart import PixArtAdapter
from .sana import SanaAdapter
from .sd3 import SD3Adapter
from .wan import WanAdapter

__all__ = [
    "FluxAdapter",
    "Flux2Adapter",
    "LTXAdapter",
    "LTX2Adapter",
    "MMDiTAdapter",
    "PixArtAdapter",
    "SanaAdapter",
    "SD3Adapter",
    "WanAdapter",
]
