# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the MSA index scorers the M3.1 targets call themselves (``kernels/msa_kernels/fmha_sm100_q8kv8``).

The MSA attention backend, its indexer proxy pass and its prefill run the packaged ``fmha_sm100`` that the engine
resolves through ``attention/backends/sparse/minimax_m3/kernels/msa_utils.require_msa_module``; the targets prewarm
that same module.
"""


def install_msa_cutlass_compatibility() -> None:
    """Provide the legacy CuTe DSL names the vendored MSA sources still reference.

    The MSA CuTe kernels annotate with ``cute.core.ThrMma`` / ``cute.core.ThrCopy``, which newer CUTLASS DSL releases
    only expose as ``cute.ThrMma`` / ``cute.ThrCopy``, and call ``cute.make_fragment``, now ``cute.make_rmem_tensor``.
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
