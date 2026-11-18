# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vendored code of the Kimi K3 family.

``vllm_mr8/`` is vLLM MR8 (``df90f706c5``)'s Kimi K3 specialized decode, the files the ULL chain
reaches from ``vllm/models/kimi_k3/nvidia/{specialized_decode,ops}`` at their relative paths, plus
the NVFP4 TP16 Rubin experts (``vllm_mr8/nv_experts.py``, ``vllm_mr8/nv_experts_runtime.py``). The
import statements differ from the source (``../SOURCE.json``, ``vendor["vllm_mr8"]``): imports of
the vendored tree are relative, and the few other ``vllm`` modules it imports are stand-ins in
``vllm_mr8/_compat*.py``. Other local changes are marked ``TensorRT-LLM local change`` and listed as
``local_edits`` there.
"""
