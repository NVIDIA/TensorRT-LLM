# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the vendored MSA q2k -> k2q CSR builder (build_k2q_csr.cu) ahead of time.

The vendored module compiles it on first import with torch.utils.cpp_extension.load; this runs
the same build, with the same flags, at build time without loading the library
(torch_extension_aot.py) and places the extension where the vendored module looks for an
ahead-of-time build first.
usage: build_k2q_extension.py <build_k2q_csr.cu> <output dir> <build dir>
"""

import shutil
import sys
from pathlib import Path

from torch_extension_aot import build_extension

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
    library = build_extension(
        name="sparse_build_k2q_csr_m31_ext",
        sources=[str(source)],
        extra_cflags=_EXTRA_CFLAGS,
        extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
        build_directory=build,
    )
    shutil.copy2(library, output / library.name)


if __name__ == "__main__":
    main()
