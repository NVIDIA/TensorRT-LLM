#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# qwen3_5 native code. Each target keeps its own in <target>/csrc/; see the
# modeling_v2 README.
add_modeling_v2_family(qwen3_5)

# The NVFP4 W4A16 GEMV (plain and with the fused conv1d update), the cuda-core
# NVFP4 GEMM, the Marlin NVFP4 GEMM (marlin/), the NVFP4 gate_up GEMM with the
# SwiGLU + NVFP4 epilogue and the e4m3 head-256 context FMHA (fmha/) run on
# SM120/121 only.
set(qwen3_8_27b_kernels
    "${CMAKE_CURRENT_SOURCE_DIR}/../qwen3_8_27b_nvfp4__sm_120__tp1/csrc/nvfp4W4a16Gemv.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/../qwen3_8_27b_nvfp4__sm_120__tp1/csrc/nvfp4GateUpSwigluSm120.cu"
)
# The CUTLASS GEMM waits on its PDL predecessor (grid dependency control).
set_property(
  SOURCE ${qwen3_8_27b_kernels}
  APPEND
  PROPERTY COMPILE_OPTIONS "-DCUTLASS_ENABLE_GDC_FOR_SM100=1")
if(NOT WIN32)
  set_property(SOURCE ${qwen3_8_27b_kernels} APPEND
               PROPERTY COMPILE_OPTIONS "-Xcompiler=-Wno-psabi")
endif()

# The XQA decode kernel (xqa/, compiled ahead of time through xqaDecodeSm120.cu)
# is built with fast math, as TensorRT-LLM's XQA JIT builds it.
set_property(
  SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/../qwen3_8_27b_nvfp4__sm_120__tp1/csrc/xqaDecodeSm120.cu"
    APPEND
  PROPERTY COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")

# The SM120 e4m3 head-256 context FMHA (fmha/) is built from target-private
# copies of the fmha_v2 pieces it changes plus the stock fmha_v2 headers, with
# the defines the stock FMHA kernels use
# (cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/CMakeLists.txt). The
# test op includes the same headers for the kernel parameter struct.
set(qwen3_8_27b_fmha_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/../qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/contextFmhaSm120.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/../qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/contextFmhaSm120TestOp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/contextFmhaSm120Run.cpp"
)
set_property(
  SOURCE ${qwen3_8_27b_fmha_sources} APPEND
  PROPERTY COMPILE_OPTIONS "-I${PROJECT_SOURCE_DIR}/kernels/fmha_v2/src")
set_property(
  SOURCE ${qwen3_8_27b_fmha_sources} APPEND
  PROPERTY COMPILE_DEFINITIONS USE_DEMO_BERT_PARAMS=1 GENERATE_CUBIN=1)

add_modeling_v2_target(
  qwen3_5 qwen3_8_27b_nvfp4__sm_120__tp1
  ARCHS 120f
  SOURCES *.cu *.cpp marlin/*.cu marlin/*.cpp fmha/*.cu fmha/*.cpp)
