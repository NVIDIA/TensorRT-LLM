# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

set(NVFP4_COLD_PAGE_CODEC_TEST_SRC
    nvfp4ColdPageCodecTest.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager/kv_cache_manager_v2/coldPageCopy.cu
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager/kv_cache_manager_v2/coldPageCodec.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/kv_cache_compression/nvfp4ColdPageCodec.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/assert.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/cudaDriverWrapper.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/envUtils.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/logger.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/stringUtils.cpp
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/common/tllmException.cpp)

add_gtest(nvfp4ColdPageCodecTest "${NVFP4_COLD_PAGE_CODEC_TEST_SRC}"
          NO_TLLM_LINKAGE)
target_link_libraries(nvfp4ColdPageCodecTest PRIVATE CUDA::cuda_driver CUDA::cudart)
target_include_directories(
  nvfp4ColdPageCodecTest
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
