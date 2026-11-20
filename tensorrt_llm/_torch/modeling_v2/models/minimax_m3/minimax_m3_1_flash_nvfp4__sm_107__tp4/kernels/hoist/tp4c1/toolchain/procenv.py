# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Toolchain selection for every build in this process, including cpp_extension builds inside legacy copies, through
child-process overrides instead of os.environ writes. Called once by the JIT before the first build."""

from __future__ import annotations

import os


def cccl_escape(cuda_home: str) -> list[str]:
    """`targets/<t>/include/cuda` symlinks that leave the toolkit. The CUDA debs ship it as an absolute compat link into
    the default toolkit's `include/cccl/cuda`; next to an image whose default toolkit is another release (13.4) it pulls
    that release's CCCL into every nvcc 13.5 compile (cuda_runtime.h includes it by a quoted relative path, so no -I
    can override it): thrust/version.h static_assert 3.5.0 vs 3.4.2. Fix in the toolkit: a relative `cccl/cuda` link."""
    out = []
    tdir = os.path.join(cuda_home, "targets")
    for t in sorted(os.listdir(tdir)) if os.path.isdir(tdir) else []:
        link = os.path.join(tdir, t, "include", "cuda")
        if os.path.islink(link):
            real, home = os.path.realpath(link), os.path.realpath(cuda_home)
            if not real.startswith(home + os.sep):
                out.append(f"{link} -> {os.readlink(link)}")
    return out


def select_cuda(cuda_home: str) -> dict:
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc):
        raise FileNotFoundError(f"select_cuda: {nvcc} missing")
    bad = cccl_escape(cuda_home)
    if bad:
        raise FileNotFoundError(
            f"select_cuda: {cuda_home} borrows another toolkit's CCCL headers ({bad}); make the "
            "link relative (ln -sfn cccl/cuda targets/<t>/include/cuda)"
        )
    from . import child_env

    child_env.child_env(cuda_home=cuda_home, path_prefix=os.path.join(cuda_home, "bin"))
    return dict(child_env.OVERRIDES)
