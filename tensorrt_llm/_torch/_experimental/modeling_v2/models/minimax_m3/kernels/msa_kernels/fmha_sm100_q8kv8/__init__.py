# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MSA Q8KV4 / Q8KV8 index scorers (CuTe DSL) from vllm-project/MSA python/fmha_sm100/cute.

MIT, Copyright (c) 2026 MiniMax (see LICENSE and NOTICE); ``src/sm100/q8kv4_indexer_decode.py`` is
Apache-2.0, Copyright contributors to the vLLM project. The tree is managed by ``scripts/vendor/manage.py``
(entry ``msa-m31-q8kv8-indexer`` in ``3rdparty/vendor_sources.lock.yaml``). Its persistent patch carries
the MiniMax-M3.1 changes (the select-layout decode output, the prefill MMA K in the kernel key) and makes
the package's imports relative.
"""

from .src.sm100.q8kv8_indexer_prefill import MMA_K  # noqa: F401
