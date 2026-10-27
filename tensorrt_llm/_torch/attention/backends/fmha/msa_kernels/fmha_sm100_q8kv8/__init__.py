# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MSA Q8KV4 / Q8KV8 index scorers (CuTe DSL), vendored from vllm-project/MSA@25c1447 python/fmha_sm100/cute
(MIT, Copyright (c) 2026 MiniMax; q8kv4_indexer_decode.py Apache-2.0, vLLM project contributors). The tree is managed by
scripts/vendor_sources.py (entry msa-q8kv8-indexer); its persistent patch carries the MiniMax-M3.1 changes (the select-
layout decode output, the prefill MMA_K mode in the kernel key) and makes the package's imports relative.

The MSA CuTe DSL modules need TensorRT-LLM's CUTLASS DSL compatibility shim before they import.
"""

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    _install_msa_cutlass_compatibility,
)

_install_msa_cutlass_compatibility()

from .src.sm100.q8kv8_indexer_prefill import MMA_K  # noqa: E402,F401
