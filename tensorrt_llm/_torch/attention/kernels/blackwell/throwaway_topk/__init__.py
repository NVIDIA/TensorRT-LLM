# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Exact sample/filter/recover MQA top-k for SM100."""
import importlib

try:
    memory = importlib.import_module("cutlass.memory")
except ModuleNotFoundError as error:
    if error.name != "cutlass.memory":
        raise
    # CuTe DSL 4.7 exposes these allocators under utils.
    from cutlass import utils as memory
try:
    tensor_utils = importlib.import_module("cutlass.tensor_utils")
except ModuleNotFoundError as error:
    if error.name != "cutlass.tensor_utils":
        raise
    from cutlass import utils as tensor_utils
