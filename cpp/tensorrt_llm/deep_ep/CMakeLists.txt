set(DEEP_EP_COMMIT 205817d98b692ed32a1d104775251292091cfab7)
set(NVSHMEM_URL_HASH
    SHA256=eb2c8fb3b7084c2db86bd9fd905387909f1dfd483e7b45f7b3c3d5fcf5374b5a)

add_custom_target(deep_ep)

# CUDA architectures
# ==================

# Filter CUDA arch >= 9.0
set(DEEP_EP_CUDA_ARCHITECTURES "")
foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES_NORMALIZED)
  string(REGEX MATCHALL "^([1-9][0-9]*)[0-9]a?(-real|-virtual)?$" MATCHES
               ${CUDA_ARCH})
  if(NOT CMAKE_MATCH_0)
    message(FATAL_ERROR "Invalid CUDA arch format: \"${CUDA_ARCH}\"")
  endif()
  set(CUDA_ARCH_MAJOR ${CMAKE_MATCH_1})
  if(${CUDA_ARCH_MAJOR} GREATER_EQUAL 9)
    list(APPEND DEEP_EP_CUDA_ARCHITECTURES ${CMAKE_MATCH_0})
  endif()
endforeach()

# Skip build if there is no suitable CUDA arch
if(WIN32)
  set(DEEP_EP_CUDA_ARCHITECTURES "")
endif()
message(
  STATUS "deep_ep DEEP_EP_CUDA_ARCHITECTURES: ${DEEP_EP_CUDA_ARCHITECTURES}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cuda_architectures.txt
     "${DEEP_EP_CUDA_ARCHITECTURES}")
if(NOT DEEP_EP_CUDA_ARCHITECTURES)
  return()
endif()

# Prepare files
# =============

# Download DeepEP
include(FetchContent)
if(DEFINED $ENV{GITHUB_MIRROR})
  set(GITHUB_URL "$ENV{GITHUB_MIRROR}")
else()
  set(GITHUB_URL "https://github.com")
endif()
FetchContent_Declare(
  deep_ep_download
  URL ${GITHUB_URL}/deepseek-ai/DeepEP/archive/${DEEP_EP_COMMIT}.tar.gz)
FetchContent_MakeAvailable(deep_ep_download)
set(DEEP_EP_SOURCE_DIR ${deep_ep_download_SOURCE_DIR})

# Copy and update python files
set(DEEP_EP_PYTHON_DEST ${CMAKE_CURRENT_BINARY_DIR}/python/deep_ep)
file(REMOVE_RECURSE ${DEEP_EP_PYTHON_DEST})
file(MAKE_DIRECTORY ${DEEP_EP_PYTHON_DEST})
file(COPY ${DEEP_EP_SOURCE_DIR}/LICENSE DESTINATION ${DEEP_EP_PYTHON_DEST})
set(_files __init__.py buffer.py utils.py)
foreach(_f IN LISTS _files)
  set(_src "${DEEP_EP_SOURCE_DIR}/deep_ep/${_f}")
  set(_dst "${DEEP_EP_PYTHON_DEST}/${_f}")
  file(READ "${_src}" _content)
  string(REPLACE "deep_ep_cpp" "tensorrt_llm.deep_ep_cpp_tllm" _content
                 "${_content}")
  string(
    PREPEND
    _content
    "# Adapted from https://github.com/deepseek-ai/DeepEP/blob/${DEEP_EP_COMMIT}/deep_ep/${_f}\n"
  )
  file(WRITE "${_dst}" "${_content}")
endforeach()

# Delete stale nvshmem on patch update
set(NVSHMEM_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/nvshmem_stamp.txt)
file(COPY ${DEEP_EP_SOURCE_DIR}/third-party/nvshmem.patch
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(SHA256 ${CMAKE_CURRENT_BINARY_DIR}/nvshmem.patch NVSHMEM_PATCH_HASH)
set(NVSHMEM_STAMP_CONTENT "${NVSHMEM_URL_HASH} patch ${NVSHMEM_PATCH_HASH}")
set(OLD_NVSHMEM_STAMP_CONTENT "")
if(EXISTS ${NVSHMEM_STAMP_FILE})
  file(READ ${NVSHMEM_STAMP_FILE} OLD_NVSHMEM_STAMP_CONTENT)
endif()
if(NOT "${OLD_NVSHMEM_STAMP_CONTENT}" STREQUAL "${NVSHMEM_STAMP_CONTENT}")
  file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/nvshmem_project-prefix)
  file(WRITE ${NVSHMEM_STAMP_FILE} "${NVSHMEM_STAMP_CONTENT}")
endif()

# Add NVSHMEM
# ===========

# Find libmlx5.so.1
execute_process(
  COMMAND
    bash -c
    "dirname $(ldconfig -p | grep libmlx5.so.1 | head -n1 | awk '{print $NF}')"
  RESULT_VARIABLE _LIBMLX5_DIR_SUCCESS
  OUTPUT_VARIABLE LIBMLX5_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _LIBMLX5_DIR_SUCCESS EQUAL 0)
  message(FATAL_ERROR "Failed to locate libmlx5.so.1")
endif()

# Add nvshmem external project
include(ExternalProject)
ExternalProject_Add(
  nvshmem_project
  URL https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
  URL_HASH ${NVSHMEM_URL_HASH}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build
  PATCH_COMMAND patch -p1 -i ${CMAKE_CURRENT_BINARY_DIR}/nvshmem.patch
  COMMAND sed "s/TRANSPORT_VERSION_MAJOR 3/TRANSPORT_VERSION_MAJOR 103/" -i
          src/CMakeLists.txt
  CMAKE_ARGS -DNVSHMEM_IBGDA_SUPPORT=1
             -DNVSHMEM_IBRC_SUPPORT=0
             -DNVSHMEM_MPI_SUPPORT=0
             -DNVSHMEM_PMIX_SUPPORT=0
             -DNVSHMEM_SHMEM_SUPPORT=0
             -DNVSHMEM_TIMEOUT_DEVICE_POLLING=0
             -DNVSHMEM_UCX_SUPPORT=0
             -DNVSHMEM_USE_NCCL=0
             -DNVSHMEM_USE_GDRCOPY=0
             -DNVSHMEM_BUILD_HYDRA_LAUNCHER=0
             -DMLX5_lib=${LIBMLX5_DIR}/libmlx5.so.1
             -DNVSHMEM_BUILD_BITCODE_LIBRARY=0
             -DNVSHMEM_BUILD_EXAMPLES=0
             -DNVSHMEM_BUILD_TESTS=0
  CMAKE_CACHE_ARGS
    -DCMAKE_CUDA_ARCHITECTURES:STRING=${DEEP_EP_CUDA_ARCHITECTURES}
  INSTALL_COMMAND "")
add_library(nvshmem_project::nvshmem STATIC IMPORTED)
add_dependencies(nvshmem_project::nvshmem nvshmem_project)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build/src/include)
set_target_properties(
  nvshmem_project::nvshmem
  PROPERTIES IMPORTED_LOCATION
             ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build/src/lib/libnvshmem.a
             INTERFACE_INCLUDE_DIRECTORIES
             ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build/src/include)

# Add DeepEP cpp
# ==============

# Remove -gencode, use CUDA_ARCHITECTURES
string(REGEX REPLACE "-gencode arch=[^ ]+ " "" CMAKE_CUDA_FLAGS
                     "${CMAKE_CUDA_FLAGS}")

# Let CMake generate `fatbinData` for -rdc. Set to FALSE and TRUE are both OK,
# but it generates `code=lto_90a` rather than `code=sm_90a` for
# CUDA_ARCHITECTURES 90a-real if set to TRUE.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

# Find torch_python
find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)

# Add deep_ep_cpp_tllm
pybind11_add_module(
  deep_ep_cpp_tllm
  ${DEEP_EP_SOURCE_DIR}/csrc/deep_ep.cpp
  ${DEEP_EP_SOURCE_DIR}/csrc/kernels/internode.cu
  ${DEEP_EP_SOURCE_DIR}/csrc/kernels/internode_ll.cu
  ${DEEP_EP_SOURCE_DIR}/csrc/kernels/intranode.cu
  ${DEEP_EP_SOURCE_DIR}/csrc/kernels/layout.cu
  ${DEEP_EP_SOURCE_DIR}/csrc/kernels/runtime.cu)
set_target_properties(
  deep_ep_cpp_tllm
  PROPERTIES CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD_REQUIRED ON
             CXX_STANDARD 17
             CUDA_STANDARD 17
             CUDA_SEPARABLE_COMPILATION ON
             CUDA_ARCHITECTURES "${DEEP_EP_CUDA_ARCHITECTURES}"
             INSTALL_RPATH "$ORIGIN/libs/nvshmem"
             BUILD_WITH_INSTALL_RPATH TRUE)
target_compile_options(
  deep_ep_cpp_tllm
  PRIVATE ${TORCH_CXX_FLAGS} -O3
          $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=--register-usage-level=10>)
target_compile_definitions(deep_ep_cpp_tllm
                           PRIVATE TORCH_EXTENSION_NAME=deep_ep_cpp_tllm)
target_link_libraries(
  deep_ep_cpp_tllm PRIVATE nvshmem_project::nvshmem ${TORCH_LIBRARIES}
                           ${TORCH_PYTHON_LIB})
target_link_options(
  deep_ep_cpp_tllm PRIVATE
  -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/deep_ep_cpp_tllm.version
  -Wl,--no-undefined-version)

# Set targets
# ===========
add_dependencies(deep_ep deep_ep_cpp_tllm nvshmem_project)
