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
"""Build a torch C++/CUDA extension ahead of time without loading it.

torch.utils.cpp_extension.load compiles and links the library, then dlopens it. The wheel build runs on hosts without
a GPU driver, where a library linked against the CUDA driver (-lcuda; the toolkit's libcuda stub at link time) cannot
be loaded: libcuda.so.1 is missing. This runs the same ninja build load runs for a fresh build directory and stops
before the load, so the library is identical and the build needs no driver. The library gets the real driver when it
is loaded at run time.
"""

from pathlib import Path
from typing import Optional, Sequence


def build_extension(
    name: str,
    sources: Sequence[str],
    build_directory: Path,
    extra_cflags: Optional[Sequence[str]] = None,
    extra_cuda_cflags: Optional[Sequence[str]] = None,
    extra_ldflags: Optional[Sequence[str]] = None,
    extra_include_paths: Optional[Sequence[str]] = None,
    with_cuda: Optional[bool] = None,
) -> Path:
    """Compile and link `name` like load(..., is_python_module=False) does; return the library path."""
    from torch.utils import cpp_extension

    # load -> _jit_compile -> _write_ninja_file_and_build_library: the version suffix is 0 in a fresh process, so
    # the module keeps its name; only the trailing _import_module_from_library (the dlopen) is left out.
    cpp_extension._write_ninja_file_and_build_library(
        name=name,
        sources=list(sources),
        extra_cflags=list(extra_cflags or []),
        extra_cuda_cflags=list(extra_cuda_cflags or []),
        extra_sycl_cflags=[],
        extra_ldflags=list(extra_ldflags or []),
        extra_include_paths=list(extra_include_paths or []),
        build_directory=str(build_directory),
        verbose=False,
        with_cuda=with_cuda,
        with_sycl=None,
    )
    library = Path(build_directory) / f"{name}.so"
    if not library.is_file():
        raise RuntimeError(f"building {name} produced no {library}")
    return library
