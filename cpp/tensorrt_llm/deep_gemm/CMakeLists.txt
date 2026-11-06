# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

add_custom_target(deep_gemm)

if(WIN32)
  return()
endif()

# Prepare files
# =============

# Use DeepGEMM submodule
set(DEEP_GEMM_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/deepgemm-src)
get_filename_component(DEEP_GEMM_SOURCE_DIR ${DEEP_GEMM_SOURCE_DIR} ABSOLUTE)

# Check if submodules are initialized
if(NOT EXISTS ${DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include
   OR NOT EXISTS ${DEEP_GEMM_SOURCE_DIR}/third-party/deep_jit/include)
  message(
    FATAL_ERROR
      "DeepGEMM submodules (cutlass, deep_jit) not initialized. Something went wrong with FetchContent.\n"
  )
endif()

# Copy and update python files
set(DEEP_GEMM_PYTHON_DEST ${CMAKE_CURRENT_BINARY_DIR}/python/deep_gemm)
file(REMOVE_RECURSE ${DEEP_GEMM_PYTHON_DEST})
file(MAKE_DIRECTORY ${DEEP_GEMM_PYTHON_DEST})

# Copy all files from deep_gemm directory
file(GLOB_RECURSE DEEP_GEMM_ALL_FILES ${DEEP_GEMM_SOURCE_DIR}/deep_gemm/*)
configure_file(${DEEP_GEMM_SOURCE_DIR}/LICENSE ${DEEP_GEMM_PYTHON_DEST}/LICENSE
               COPYONLY)
foreach(SOURCE_FILE ${DEEP_GEMM_ALL_FILES})
  file(RELATIVE_PATH REL_PATH ${DEEP_GEMM_SOURCE_DIR}/deep_gemm ${SOURCE_FILE})
  get_filename_component(REL_DIR ${REL_PATH} DIRECTORY)
  file(MAKE_DIRECTORY ${DEEP_GEMM_PYTHON_DEST}/${REL_DIR})

  # Check if it's a Python file that needs import renaming
  get_filename_component(FILE_EXT ${SOURCE_FILE} EXT)
  if(FILE_EXT STREQUAL ".py")
    # Read file content and replace module imports for Python files
    file(READ ${SOURCE_FILE} _content)
    string(REPLACE "from . import _C" "import tensorrt_llm.deep_gemm_cpp_tllm"
                   _content "${_content}")
    string(REPLACE "from .. import _C" "import tensorrt_llm.deep_gemm_cpp_tllm"
                   _content "${_content}")
    string(REPLACE ".._C" "tensorrt_llm.deep_gemm_cpp_tllm" _content
                   "${_content}")
    string(REPLACE "._C" "tensorrt_llm.deep_gemm_cpp_tllm" _content
                   "${_content}")
    string(REPLACE "_C." "tensorrt_llm.deep_gemm_cpp_tllm." _content
                   "${_content}")

    if(REL_PATH STREQUAL "__init__.py")
      string(
        PREPEND
        _content
        "# Use the PyTorch-owned cuBLASLt handle by default. DeepGEMM's\n"
        "# standalone handle destructor can run after PyTorch/cuBLASLt teardown\n"
        "# in MPI worker shutdown and trigger a double free.\n"
        "import os\n"
        "os.environ.setdefault(\"DG_USE_PYTORCH_CUBLASLT_HANDLE\", \"1\")\n")
    endif()

    # Add adaptation header
    string(
      PREPEND
      _content
      "# Adapted from https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/${REL_PATH}\n"
    )

    # Write modified content
    set(_dst "${DEEP_GEMM_PYTHON_DEST}/${REL_PATH}")
    file(WRITE ${_dst} "${_content}")
  else()
    # Copy non-Python files as-is
    set(_dst "${DEEP_GEMM_PYTHON_DEST}/${REL_PATH}")
    file(COPY ${SOURCE_FILE} DESTINATION ${DEEP_GEMM_PYTHON_DEST}/${REL_DIR})
  endif()

  # Add dependency tracking
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${SOURCE_FILE})
endforeach()

# Copy third-party includes (cutlass) to the include directory. DeepJIT is
# header-only and host-side only, so it is not needed at kernel JIT time.
set(DEEP_GEMM_INCLUDE_DEST ${DEEP_GEMM_PYTHON_DEST}/include)
file(MAKE_DIRECTORY ${DEEP_GEMM_INCLUDE_DEST})
file(COPY ${DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include/cute
     DESTINATION ${DEEP_GEMM_INCLUDE_DEST})
file(COPY ${DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include/cutlass
     DESTINATION ${DEEP_GEMM_INCLUDE_DEST})

# Find torch_python
find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)

# Use the standard formatter when available. Older C++20 standard libraries
# build host-only copies against header-only fmt, without modifying the fetched
# sources or changing the CUDA kernels included in the wheel.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
  "#include <format>
   int main() { return std::format(\"{}\", 42) != \"42\"; }"
  DEEP_GEMM_HAVE_STD_FORMAT)
set(DEEP_GEMM_CSRC_DIR ${DEEP_GEMM_SOURCE_DIR}/csrc)
set(DEEP_JIT_INCLUDE_DIR ${DEEP_GEMM_SOURCE_DIR}/third-party/deep_jit/include)
if(NOT DEEP_GEMM_HAVE_STD_FORMAT)
  message(
    STATUS "std::format unavailable; DeepGEMM host code uses header-only fmt")
  FetchContent_MakeAvailable(fmt)
  include(${CMAKE_CURRENT_SOURCE_DIR}/formatCompatibility.cmake)
  set(DEEP_GEMM_COMPAT_DIR ${CMAKE_CURRENT_BINARY_DIR}/format_compat)
  trtllm_copy_with_fmt(${DEEP_GEMM_CSRC_DIR} ${DEEP_GEMM_COMPAT_DIR}/csrc)
  trtllm_copy_with_fmt(${DEEP_JIT_INCLUDE_DIR}
                       ${DEEP_GEMM_COMPAT_DIR}/deep_jit/include)
  set(DEEP_GEMM_CSRC_DIR ${DEEP_GEMM_COMPAT_DIR}/csrc)
  set(DEEP_JIT_INCLUDE_DIR ${DEEP_GEMM_COMPAT_DIR}/deep_jit/include)
endif()

# Build deep_gemm_cpp_tllm extension (matching deep_gemm's setup.py)
set(DEEP_GEMM_SOURCES ${DEEP_GEMM_CSRC_DIR}/python_api.cpp)

pybind11_add_module(deep_gemm_cpp_tllm ${DEEP_GEMM_SOURCES})
set_target_properties(
  deep_gemm_cpp_tllm
  PROPERTIES CXX_STANDARD_REQUIRED ON
             CXX_STANDARD 20
             CXX_SCAN_FOR_MODULES OFF
             CUDA_SEPARABLE_COMPILATION ON
             LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/deep_gemm_cpp_tllm.version
             INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)

target_compile_options(deep_gemm_cpp_tllm PRIVATE ${TORCH_CXX_FLAGS} -O3 -fPIC
                                                  -Wno-psabi)

# Extension name definition
target_compile_definitions(deep_gemm_cpp_tllm
                           PRIVATE TORCH_EXTENSION_NAME=deep_gemm_cpp_tllm)
if(NOT DEEP_GEMM_HAVE_STD_FORMAT)
  target_compile_definitions(deep_gemm_cpp_tllm PRIVATE FMT_HEADER_ONLY=1)
  target_include_directories(
    deep_gemm_cpp_tllm PRIVATE ${fmt_SOURCE_DIR}/include
                               ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Include directories matching deep_gemm setup.py
target_include_directories(
  deep_gemm_cpp_tllm
  PRIVATE ${CUDA_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include/cccl
          ${DEEP_GEMM_SOURCE_DIR}/deep_gemm/include ${DEEP_JIT_INCLUDE_DIR}
          ${DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include)

# DeepJIT resolves libdwfl through dlopen at runtime but includes the elfutils
# header for its types. Hosts without the elfutils development headers use the
# minimal declaration shim under deep_jit_shim/ instead.
include(CheckIncludeFileCXX)
check_include_file_cxx("elfutils/libdwfl.h" DEEP_GEMM_HAVE_LIBDWFL_HEADER)
if(NOT DEEP_GEMM_HAVE_LIBDWFL_HEADER)
  message(
    STATUS
      "elfutils/libdwfl.h not found; DeepJIT builds against the libdwfl declaration shim"
  )
  target_include_directories(deep_gemm_cpp_tllm
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/deep_jit_shim)
endif()

# Link libraries (matching deep_gemm setup.py: cuda, cudart + torch)
target_link_libraries(
  deep_gemm_cpp_tllm PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB}
                             CUDA::cuda_driver CUDA::cudart)

# Link directories
target_link_directories(
  deep_gemm_cpp_tllm PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

# Set targets
# ===========
add_dependencies(deep_gemm deep_gemm_cpp_tllm)
