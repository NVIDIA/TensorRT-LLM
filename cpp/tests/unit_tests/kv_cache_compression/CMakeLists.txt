# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

set(NVFP4_COLD_PAGE_KERNEL_TEST_SRC
    ${PROJECT_SOURCE_DIR}/tests/unit_tests/kernels/nvfp4ColdPageKernelsTest.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/kernels/nvfp4ColdPageKernels.cu
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/assert.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/envUtils.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/logger.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/stringUtils.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/tllmException.cpp)
add_gtest(nvfp4ColdPageKernelsTest "${NVFP4_COLD_PAGE_KERNEL_TEST_SRC}"
          NO_TLLM_LINKAGE)
target_include_directories(
  nvfp4ColdPageKernelsTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
target_link_libraries(nvfp4ColdPageKernelsTest PRIVATE CUDA::cudart
                                                       CUDA::cuda_driver)
