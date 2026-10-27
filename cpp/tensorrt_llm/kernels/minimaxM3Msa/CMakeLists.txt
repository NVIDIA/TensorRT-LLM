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

# Ahead-of-time build of the MSA fmha_sm100 kernels (MiniMax sparse attention,
# vendored at
# tensorrt_llm/_torch/attention/backends/fmha/msa_kernels/fmha_sm100). The
# vendored jit.py compiled each module on first use; these rules compile the
# modules the MiniMax-M3.1 targets call, from the same sources with the same
# nvcc command, into the layout jit.py loads from
# (msa_fmha_sm100/<variant>/<variant>.so, plan/, reduction/, sparse_topk/).
# build_wheel.py installs the tree under tensorrt_llm/libs. The variant list and
# the rendered variant sources come from generate_fmha_variants.py.

set(MSA_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
list(FILTER MSA_ARCHS INCLUDE REGEX "^10[0-9]")
if(NOT MSA_ARCHS)
  message(STATUS "MSA fmha_sm100: no SM10x architecture in this build, skipped")
  return()
endif()

set(MSA_PACKAGE_DIR
    ${PROJECT_SOURCE_DIR}/../tensorrt_llm/_torch/attention/backends/fmha/msa_kernels/fmha_sm100
)
set(MSA_CSRC ${MSA_PACKAGE_DIR}/csrc)
set(MSA_GENERATED ${CMAKE_CURRENT_SOURCE_DIR}/generated)
set(MSA_OUT ${CMAKE_CURRENT_BINARY_DIR}/msa_fmha_sm100)
include(${MSA_GENERATED}/variants.cmake)
file(GLOB_RECURSE MSA_HEADERS ${MSA_CSRC}/*.h ${MSA_CSRC}/*.hpp
     ${MSA_CSRC}/*.cuh)

find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)
execute_process(
  COMMAND
    ${Python3_EXECUTABLE} -c
    "import pathlib, tvm_ffi; print(pathlib.Path(tvm_ffi.__path__[0]) / 'include', end='')"
  RESULT_VARIABLE MSA_TVM_FFI_RESULT
  OUTPUT_VARIABLE MSA_TVM_FFI_INCLUDE)
if(NOT MSA_TVM_FFI_RESULT EQUAL 0 OR NOT EXISTS ${MSA_TVM_FFI_INCLUDE})
  message(
    FATAL_ERROR
      "MSA fmha_sm100 needs the apache-tvm-ffi headers (pip package apache-tvm-ffi)"
  )
endif()

# jit.py's _get_nvcc_flags on SM107 (no proxy-mode defines): keep in sync with
# it.
set(MSA_NVCC_FLAGS
    -O3
    -std=c++20
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -gencode=arch=compute_100a,code=sm_100a
    -gencode=arch=compute_103a,code=sm_103a
    -gencode=arch=compute_100f,code=sm_100f
    -static-global-template-stub=false
    -DFLASHINFER_ENABLE_BF16
    -DFLASHINFER_ENABLE_FP8_E4M3
    -DFLASHINFER_ENABLE_FP8_E5M2
    -DFLASHINFER_ENABLE_FP8_E8M0
    -DFLASHINFER_ENABLE_FP4_E2M1
    -DFLASHINFER_ENABLE_F16
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -Xcudafe
    --diag_suppress=2908
    -I${MSA_CSRC}/include
    -I${CMAKE_BINARY_DIR}/_deps/cutlass-src/include
    -I${CMAKE_BINARY_DIR}/_deps/cutlass-src/tools/util/include
    -I${MSA_TVM_FFI_INCLUDE}
    -I${MSA_CSRC}
    -use_fast_math
    -DNDEBUG
    -Xcompiler
    -fPIC)

# msa_module(<output .so> <ptxas level> <sources...>)
function(msa_module output ptxas)
  set(objects)
  get_filename_component(directory ${output} DIRECTORY)
  get_filename_component(module ${output} NAME_WE)
  foreach(source ${ARGN})
    get_filename_component(name ${source} NAME_WE)
    set(object ${CMAKE_CURRENT_BINARY_DIR}/objects/${module}.${name}.o)
    add_custom_command(
      OUTPUT ${object}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${directory}
              ${CMAKE_CURRENT_BINARY_DIR}/objects
      COMMAND ${CMAKE_CUDA_COMPILER} ${MSA_NVCC_FLAGS} -Xptxas ${ptxas} -c
              ${source} -o ${object}
      DEPENDS ${source} ${MSA_HEADERS}
      COMMENT "MSA fmha_sm100: compiling ${name}"
      VERBATIM)
    list(APPEND objects ${object})
  endforeach()
  add_custom_command(
    OUTPUT ${output}
    COMMAND ${CMAKE_CUDA_COMPILER} -shared ${objects} -o ${output} -lcuda
    DEPENDS ${objects}
    COMMENT "MSA fmha_sm100: linking ${output}"
    VERBATIM)
  set_property(GLOBAL APPEND PROPERTY MSA_FMHA_SM100_OUTPUTS ${output})
endfunction()

foreach(variant ${MSA_FMHA_SM100_VARIANTS})
  msa_module(
    ${MSA_OUT}/${variant}/${variant}.so -O1
    ${MSA_GENERATED}/fmha_sm100_inst_${variant}.cu
    ${MSA_GENERATED}/fmha_sm100_run_${variant}.cu)
endforeach()
msa_module(${MSA_OUT}/plan/fmha_sm100_plan.so -O3
           ${MSA_CSRC}/fmha_sm100_plan.cu)
msa_module(${MSA_OUT}/reduction/fmha_sm100_reduction.so -O3
           ${MSA_CSRC}/fmha_sm100_reduction.cu)
msa_module(${MSA_OUT}/sparse_topk/sparse_topk_select.so -O3
           ${MSA_CSRC}/sparse_topk_select.cu)

# The q2k -> k2q CSR builder is a torch extension; build_k2q_extension.py runs
# the vendored module's own cpp_extension.load at build time.
get_filename_component(MSA_CUDA_HOME ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(MSA_CUDA_HOME ${MSA_CUDA_HOME} DIRECTORY)
set(MSA_K2Q_SOURCE
    ${MSA_PACKAGE_DIR}/cute/src/sm100/build_k2q_csr/build_k2q_csr.cu)
add_custom_command(
  OUTPUT ${MSA_OUT}/k2q/sparse_build_k2q_csr_ext.so
  COMMAND
    ${CMAKE_COMMAND} -E env CUDA_HOME=${MSA_CUDA_HOME} ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/build_k2q_extension.py ${MSA_K2Q_SOURCE}
    ${MSA_OUT}/k2q ${CMAKE_CURRENT_BINARY_DIR}/k2q_build
  DEPENDS ${MSA_K2Q_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/build_k2q_extension.py
  COMMENT "MSA fmha_sm100: building the k2q CSR builder extension"
  VERBATIM)
set_property(GLOBAL APPEND PROPERTY MSA_FMHA_SM100_OUTPUTS
                                    ${MSA_OUT}/k2q/sparse_build_k2q_csr_ext.so)

# The BF16X / E4M3X sparse-prefill forward (msa_kernels/prefill_bf16x) the
# MiniMax-M3.1 context path serves needs CUTLASS 4.8's SM107 MMA atoms (the
# private cutlass_msa fetch; the rest of TensorRT-LLM keeps its CUTLASS) and
# CUDA 13.4. build_bf16x_extension.py runs the vendored package's own
# cpp_extension.load for the served spec. Without it the context path keeps the
# CuTe DSL forward.
set(MSA_BF16X_DIR
    ${PROJECT_SOURCE_DIR}/../tensorrt_llm/_torch/attention/backends/fmha/msa_kernels/prefill_bf16x
)
set(MSA_BF16X_MODULE _fmha_sm100_prefill_attention_bf16x_e4m3x_shift0_fp16_107a)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.4
   AND DEFINED cutlass_msa_SOURCE_DIR)
  file(GLOB_RECURSE MSA_BF16X_SOURCES ${MSA_BF16X_DIR}/csrc/*)
  add_custom_command(
    OUTPUT ${MSA_OUT}/prefill_bf16x/${MSA_BF16X_MODULE}.so
    COMMAND
      ${CMAKE_COMMAND} -E env CUDA_HOME=${MSA_CUDA_HOME} ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/build_bf16x_extension.py ${MSA_BF16X_DIR}
      ${cutlass_msa_SOURCE_DIR} ${MSA_OUT}/prefill_bf16x
      ${CMAKE_CURRENT_BINARY_DIR}/bf16x_build
    DEPENDS ${MSA_BF16X_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/build_bf16x_extension.py
    COMMENT "MSA: building the BF16X / E4M3X sparse-prefill forward"
    VERBATIM)
  set_property(
    GLOBAL APPEND PROPERTY MSA_FMHA_SM100_OUTPUTS
                           ${MSA_OUT}/prefill_bf16x/${MSA_BF16X_MODULE}.so)
else()
  message(
    STATUS "MSA prefill_bf16x: needs CUDA 13.4 and cutlass_msa; not built")
endif()

get_property(MSA_OUTPUTS GLOBAL PROPERTY MSA_FMHA_SM100_OUTPUTS)
add_custom_target(msa_fmha_sm100_aot ALL DEPENDS ${MSA_OUTPUTS})
