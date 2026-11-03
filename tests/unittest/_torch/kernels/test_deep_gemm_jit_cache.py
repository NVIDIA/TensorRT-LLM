# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import hashlib
import shutil
from pathlib import Path

import pytest
import torch
from utils.util import getSMVersion

import tensorrt_llm.deep_gemm as deep_gemm
import tensorrt_llm.deep_gemm_cpp_tllm as extension


@pytest.mark.skipif(getSMVersion() not in (100, 103, 107), reason="Requires an SM100-family GPU")
def test_cutlass_header_invalidates_jit_cache(tmp_path, monkeypatch):
    original_package = Path(deep_gemm.__file__).parent
    package = tmp_path / "deep_gemm"
    shutil.copytree(original_package, package)
    header = package / "include/cutlass/arch/config.h"
    original_header = header.read_bytes()
    cache = tmp_path / "cache"
    original_num_sms = deep_gemm.get_num_sms()
    generator = torch.Generator(device="cuda").manual_seed(20261016)
    a = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16, generator=generator)
    b = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16, generator=generator)
    output = torch.empty((128, 256), device="cuda", dtype=torch.bfloat16)
    reference = a @ b.T

    def run():
        extension.init(str(package))
        deep_gemm.bf16_gemm_nt(a, b, output)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, reference, atol=0.125, rtol=0.01)
        return {p.relative_to(cache) for p in cache.rglob("kernel.cubin")}

    def source_hashes(entries):
        return {
            hashlib.sha256((cache / p).with_name("kernel.cu").read_bytes()).hexdigest()
            for p in entries
        }

    try:
        # A restricted SM count selects DeepGEMM's JIT instead of cuBLASLt.
        deep_gemm.set_num_sms(original_num_sms // 4 * 2)
        with monkeypatch.context() as env:
            env.setenv("DG_JIT_CACHE_DIR", str(cache))
            first = run()
            assert first, "The probe must exercise the JIT compiler"
            assert run() == first, "Unmodified headers must reuse the cache"

            header.write_bytes(original_header + b"\n// JIT cache invalidation probe.\n")
            changed = run()
            assert first < changed, "Changed CUTLASS headers must produce a new cache key"
            assert source_hashes(first) == source_hashes(changed - first)

            header.write_bytes(original_header)
            assert run() == changed, "Restored headers must reuse their original cache entry"
    finally:
        extension.init(str(original_package))
        deep_gemm.set_num_sms(original_num_sms)
