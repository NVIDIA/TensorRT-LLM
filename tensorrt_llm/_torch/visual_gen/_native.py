# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loader of the VisualGen model libraries (``cpp/cmake/modules/visual_gen.cmake``)."""

import functools
from pathlib import Path
from typing import Any

import torch

# tensorrt_llm/libs, where build_wheel.py installs libth_common.so and the model libraries.
_LIBS_DIR = Path(__file__).resolve().parents[2] / "libs"


@functools.cache
def load_model_ops(model: str) -> Any:
    """Open ``libth_visual_gen_<model>.so`` once and return its ops, ``torch.ops.vg_<model>``."""
    path = _LIBS_DIR / f"libth_visual_gen_{model}.so"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not installed; it is built with TensorRT-LLM's C++ extensions"
        )
    torch.ops.load_library(str(path))
    return getattr(torch.ops, f"vg_{model}")
