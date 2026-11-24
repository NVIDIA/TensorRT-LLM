# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM main ``e3ddbdbd60``'s K3.1 DCP sparse MLA decode, DcpValue and the warp split-K
GEMM (vendored): ``specialized_decode/`` holds the closure the generation path reaches, each file
with its vLLM SPDX header and origin line (``cutedsl/common/spin.py`` is a TensorRT-LLM module).
A file NVIDIA changed for TensorRT-LLM says so in a ``Modified by NVIDIA for TensorRT-LLM`` line
under its header (import statements, the hunks marked ``TensorRT-LLM local change``, or the origin
comment only); the others are unmodified apart from their origin comment. The record of the
changes is ``../../SOURCE.json``. Its ``vllm`` imports resolve to ``../vllm_mr8`` and its
stand-ins."""

from __future__ import annotations

import importlib

from .. import vllm_mr8 as mr8  # noqa: F401 (install: the torch.accelerator names)

PKG = __name__ + ".specialized_decode"


def module(name: str):
    """``specialized_decode.<name>`` of this tree (e.g. ``"cutedsl.dsa.attn_sparse"``)."""
    return importlib.import_module(f"{PKG}.{name}")
