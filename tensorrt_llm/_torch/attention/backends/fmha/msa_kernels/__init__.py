# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax sparse attention (MSA) kernels vendored from https://gitlab.com/nvidia/tensorrt-llm/oss-components/msa
(MIT, Copyright (c) 2026 MiniMax; fmha_sm100/LICENSE and fmha_sm100/NOTICE). The tree is managed by
scripts/vendor_sources.py (entry msa-fmha-sm100 in 3rdparty/vendor_sources.lock.yaml); its CUDA modules are built
ahead of time by cpp/tensorrt_llm/kernels/minimaxM3Msa and load from tensorrt_llm/libs/msa_fmha_sm100."""


def install_cute_compatibility() -> None:
    """Provide the legacy CuTe DSL names the vendored MSA sources and quack-kernels still reference.

    quack-kernels 0.5.0 (layout_utils) annotates with ``cute.core.ThrMma``, which nvidia-cutlass-dsl 4.8 only exposes
    as ``cute.ThrMma`` (and likewise ThrCopy); the MSA CuTe kernels call ``cute.make_fragment``, now
    ``cute.make_rmem_tensor``. Installed when this package is imported, before any vendored module, so every import
    path (not only require_msa_module) sees the aliases. Remove once the sources use the new names.
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
