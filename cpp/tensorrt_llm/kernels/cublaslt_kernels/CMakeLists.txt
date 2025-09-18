# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

option(ENABLE_CUBLASLT_FP4_GEMM "Enable cuBLASLt FP4 GEMM support" ON)

if(ENABLE_CUBLASLT_FP4_GEMM)
    find_package(CUDAToolkit REQUIRED)
    
    # FP4 GEMM
    add_library(cublaslt_fp4_gemm OBJECT
        fp4_gemm/fp4_gemm_cublaslt.cpp
    )
    
    target_include_directories(cublaslt_fp4_gemm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/fp4_gemm
        ${CMAKE_CURRENT_SOURCE_DIR}/../..  # Add tensorrt_llm root directory
    )
    
    target_link_libraries(cublaslt_fp4_gemm PRIVATE
        CUDA::cublasLt
    )
    
    target_compile_definitions(cublaslt_fp4_gemm PRIVATE
        ENABLE_CUBLASLT_FP4_GEMM
    )
    
    # Set compilation options
    target_compile_options(cublaslt_fp4_gemm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++17 -fPIC>
    )
    
    # Set position independent code property
    set_property(TARGET cublaslt_fp4_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
    
    # Set include directories
    target_include_directories(cublaslt_fp4_gemm PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()
