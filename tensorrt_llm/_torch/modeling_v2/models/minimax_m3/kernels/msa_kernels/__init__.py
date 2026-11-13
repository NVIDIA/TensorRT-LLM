# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax sparse attention (MSA) kernels for the MiniMax-M3.1-Flash targets, vendored from upstream.

* ``fmha_sm100``: ``python/fmha_sm100`` of https://gitlab.com/nvidia/tensorrt-llm/oss-components/msa, vendor lock
  entry ``msa-m31-fmha-sm100`` (its CUTLASS submodule, tests and example are not selected).
* ``fmha_sm100_q8kv8``: ``python/fmha_sm100/cute`` of https://github.com/vllm-project/MSA (the Q8KV4 and Q8KV8
  index scorers), vendor lock entry ``msa-m31-q8kv8-indexer``.
* ``prefill_bf16x``: ``python/fmha_sm100/prefill_q8kv4`` of https://github.com/vllm-project/MSA (the BF16 / E4M3
  paged sparse prefill), vendor lock entry ``msa-m31-prefill-bf16x``.

All are MIT, Copyright (c) 2026 MiniMax (see each tree's LICENSE and NOTICE), and all compile at run time (JIT).
``3rdparty/vendor-sources.md`` describes the entries and their persistent patches.
"""


def install_cute_compatibility() -> None:
    """Provide the legacy CuTe DSL names the vendored MSA sources still reference.

    Runs when this package is imported, before any vendored module, so every import path sees the aliases.
    Remove it once the sources use the top-level thread classes and ``cute.make_rmem_tensor`` directly.
    """
    try:
        import cutlass.cute as cute
    except ImportError:
        return
    for thread_class_name in ("ThrCopy", "ThrMma"):
        if not hasattr(cute.core, thread_class_name):
            setattr(cute.core, thread_class_name, getattr(cute, thread_class_name))
    if not hasattr(cute, "make_fragment"):
        setattr(cute, "make_fragment", cute.make_rmem_tensor)


install_cute_compatibility()
