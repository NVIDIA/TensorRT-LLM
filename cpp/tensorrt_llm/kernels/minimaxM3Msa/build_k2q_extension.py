# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the vendored MSA q2k -> k2q CSR builder (build_k2q_csr.cu) ahead of time.

The vendored module compiled it on first import with torch.utils.cpp_extension.load; this runs
the same load, with the same flags, at build time and places the extension where the vendored
module now loads it from.
usage: build_k2q_extension.py <build_k2q_csr.cu> <output dir> <build dir>
"""

import shutil
import sys
from pathlib import Path

from torch.utils.cpp_extension import load

# The vendored module's _extra_cflags / _extra_cuda_cflags: keep in sync with
# fmha_sm100/cute/src/sm100/build_k2q_csr/__init__.py.
_EXTRA_CFLAGS = ["-O3"]
_EXTRA_CUDA_CFLAGS = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "-arch=sm_100",
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
    "-I/usr/local/cuda/include/cccl",
]


def main() -> None:
    source, output, build = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    load(
        name="sparse_build_k2q_csr_ext",
        sources=[str(source)],
        extra_cflags=_EXTRA_CFLAGS,
        extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
        build_directory=str(build),
        is_python_module=False,
        verbose=False,
    )
    shutil.copy2(build / "sparse_build_k2q_csr_ext.so", output / "sparse_build_k2q_csr_ext.so")


if __name__ == "__main__":
    main()
