# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

add_library(kv_cache_compression_src OBJECT nativeColdPageCodec.cpp)
target_include_directories(
  kv_cache_compression_src
  PRIVATE ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
set_property(TARGET kv_cache_compression_src PROPERTY POSITION_INDEPENDENT_CODE
                                                      ON)
