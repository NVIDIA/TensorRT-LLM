# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM MR8 ``df90f706c5b62178dcb26da1c2093f58cd3bc121``'s Kimi K3 specialized decode (vendored).

``specialized_decode/`` and ``ops/`` are MR8's ``vllm/models/kimi_k3/nvidia/{specialized_decode,ops}``,
only the files the K3.1 generation path reaches. A file NVIDIA changed for TensorRT-LLM says so, and
what changed, in a ``Modified by NVIDIA for TensorRT-LLM`` line under its header; the record of the
changes is ``../../SOURCE.json``. ``_vllm/`` stands in for the other ``vllm`` modules that code
imports, as a regular package.
"""

from __future__ import annotations

import importlib
import types

import torch

MR8_SHA = "df90f706c5b62178dcb26da1c2093f58cd3bc121"


def install() -> None:
    """The ``torch.accelerator`` names the vendored code calls, on torch builds without them
    (idempotent)."""
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        torch.accelerator = accelerator = types.SimpleNamespace()
    if not hasattr(accelerator, "device_index"):
        accelerator.device_index = torch.cuda.device
    if not hasattr(accelerator, "empty_cache"):
        accelerator.empty_cache = torch.cuda.empty_cache


install()


def module(name: str):
    """``specialized_decode.<name>`` (e.g. ``"cutedsl.moe.prepared"``)."""
    install()
    return importlib.import_module(f"{__name__}.specialized_decode.{name}")
