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

"""Select native build targets from the collected C++ integration tests."""

_MULTI_GPU_TARGETS = {
    "test_mpi_utils": "mpiUtilsTest",
    "test_kv_cache_manager_v2_numa": "kvCacheManagerV2HostMemNumaTest",
    "test_fused_gemm_allreduce": "gemmAllReduceTest",
    "test_cache_transceiver": "cacheTransceiverTest",
    "test_user_buffer": "userBufferTest",
    "test_nccl_utils": "ncclUtilsTest",
}


def selected_build_targets(items, cuda_arch):
    """Include every collected consumer for this fixture's architecture."""
    targets = set()
    for item in items:
        if "build_google_tests" not in item.fixturenames:
            continue
        params = item.callspec.params
        if str(params["build_google_tests"]) != str(cuda_arch):
            continue
        if item.originalname == "test_unit_tests":
            group = params["test_group"]
            directory = "executor-bounce" if group == "executor_bounce" else group
            targets.add(f"cpp-tests-{directory}")
        else:
            # Future fixture consumers retain full coverage until mapped.
            targets.add(_MULTI_GPU_TARGETS.get(item.originalname, "google-tests"))
    if not targets:
        raise ValueError(f"No collected C++ tests for CUDA architecture {cuda_arch}")
    return sorted(targets)
