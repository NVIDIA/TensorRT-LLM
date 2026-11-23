# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tp4c1.api: the Slot API. Import from here, not from submodules."""

from .contracts import (
    DENSE_STAGES,
    EAGLE_COMPONENTS,
    FAMILIES,
    M31D_STAGES,
    SPARSE_STAGES,
    STAGES,
    Contract,
    allowed,
    violations,
)
from .ctx import BuildCtx, CommBuf, Loaded
from .modules import Asset, CuteModule, NvccModule
from .params import NoParams, Params, ParamsError, params
from .pdl import ZERO, Footprint, PdlSpec, Spin
from .stage import Stage, StageFn
from .variant import Variant

API_VERSION = "2.3"

__all__ = [
    "API_VERSION",
    "Asset",
    "BuildCtx",
    "CommBuf",
    "Contract",
    "CuteModule",
    "DENSE_STAGES",
    "M31D_STAGES",
    "EAGLE_COMPONENTS",
    "FAMILIES",
    "Footprint",
    "Loaded",
    "NoParams",
    "NvccModule",
    "Params",
    "ParamsError",
    "PdlSpec",
    "SPARSE_STAGES",
    "STAGES",
    "Spin",
    "Stage",
    "StageFn",
    "Variant",
    "ZERO",
    "allowed",
    "params",
    "violations",
]
