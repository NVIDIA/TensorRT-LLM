# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test setup for the sparse_mla_decode package.

When the installed TensorRT-LLM build does not contain the sparse_mla_decode kernel yet and
``GLM_MOE_DSA_SPARSE_MLA_DECODE_DEV_SRC`` points at a source checkout, the kernel and its torch op
are compiled from that checkout so that the tests can run against an existing
image. Builds that compile the sparse_mla_decode sources into ``th_common`` skip this step.
"""

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_dev_op() -> None:
    source_root = os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_DEV_SRC")
    if not source_root:
        return
    import torch

    import tensorrt_llm

    if hasattr(torch.ops.trtllm, "glm_moe_dsa_sparse_mla_decode"):
        return
    from torch.utils.cpp_extension import load

    timing = os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_TIMING") == "1"
    # Development variants: GLM_MOE_DSA_SPARSE_MLA_DECODE_DEV_DEFINES="NAME_A+NAME_B" adds -DNAME_A
    # -DNAME_B.
    extra = [
        d for d in os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_DEV_DEFINES", "").split("+") if d
    ]
    major, minor = torch.cuda.get_device_capability()
    # GLM_MOE_DSA_SPARSE_MLA_DECODE_DEV_ARCH selects an architecture-specific target such as "103a".
    arch_name = os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_DEV_ARCH", f"{major}{minor}")
    variant = "_".join([arch_name] + (["timing"] if timing else []) + [d.lower() for d in extra])
    cpp = Path(source_root) / "cpp"
    libs = Path(tensorrt_llm.__file__).resolve().parent / "libs"
    build_dir = os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_DEV_BUILD_DIR")
    if build_dir:
        build_dir = os.path.join(build_dir, variant)
        os.makedirs(build_dir, exist_ok=True)
    defines = ["-DENABLE_BF16", "-DENABLE_FP8"] + (
        ["-DGLM_MOE_DSA_SPARSE_MLA_DECODE_TIMING"] if timing else []
    )
    defines += [f"-D{d}" for d in extra]
    # torch's extension defaults disable the half/bfloat16 operators that the
    # TensorRT-LLM CUDA headers use; the TensorRT-LLM build does not set them.
    undefines = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    arch = f"-gencode=arch=compute_{arch_name},code=sm_{arch_name}"
    load(
        name=f"glm_moe_dsa_sparse_mla_decode_dev_{variant}",
        sources=[
            str(cpp / "tensorrt_llm/kernels/glmMoeDsa/sparseMlaDecode.cu"),
            str(cpp / "tensorrt_llm/thop/glmMoeDsa/sparseMlaDecodeOp.cpp"),
        ],
        extra_include_paths=[str(cpp / "include"), str(cpp)],
        extra_cflags=["-O3", "-std=c++17", *defines],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            arch,
            *defines,
            *undefines,
        ],
        extra_ldflags=[f"-L{libs}", "-ltensorrt_llm", f"-Wl,-rpath,{libs}"],
        build_directory=build_dir,
        is_python_module=False,
        verbose=True,
    )


try:
    _load_dev_op()
except RuntimeError:
    # Keep the native-only tests runnable; the sparse_mla_decode tests skip without the op.
    traceback.print_exc()
