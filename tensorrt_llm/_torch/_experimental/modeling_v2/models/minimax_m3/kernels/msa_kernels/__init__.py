# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax sparse attention (MSA) index scorers for the MiniMax-M3.1-Flash targets, vendored from upstream.

``fmha_sm100_q8kv8`` is ``python/fmha_sm100/cute`` of https://github.com/vllm-project/MSA (the Q8KV4 and
Q8KV8 index scorers), vendor lock entry ``msa-m31-q8kv8-indexer``. It is MIT, Copyright (c) 2026 MiniMax
(see its LICENSE and NOTICE), and compiles at run time (JIT). ``3rdparty/vendor-sources.md`` describes the
entry and its persistent patch.
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
