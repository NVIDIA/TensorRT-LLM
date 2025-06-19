set(DEEP_EP_COMMIT 205817d98b692ed32a1d104775251292091cfab7)

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

# Delete stale nvshmem
set(NVSHMEM_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/nvshmem_stamp.txt)
file(SHA256 "${DEEP_EP_SOURCE_DIR}/third-party/nvshmem.patch"
     NVSHMEM_PATCH_HASH)
set(NVSHMEM_STAMP_CONTENT "3.2.5-1 ${NVSHMEM_PATCH_HASH}")
set(OLD_NVSHMEM_STAMP_CONTENT "")
if(EXISTS ${NVSHMEM_STAMP_FILE})
  file(READ ${NVSHMEM_STAMP_FILE} OLD_NVSHMEM_STAMP_CONTENT)
endif()
if(NOT "${OLD_NVSHMEM_STAMP_CONTENT}" STREQUAL "${NVSHMEM_STAMP_CONTENT}")
  file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-src)
  file(WRITE ${NVSHMEM_STAMP_FILE} "${NVSHMEM_STAMP_CONTENT}")
endif()

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

# Add nvshmem
include(ExternalProject)
ExternalProject_Add(
  nvshmem_project
  URL https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build
  PATCH_COMMAND patch -p1 -i ${DEEP_EP_SOURCE_DIR}/third-party/nvshmem.patch
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
             -DCMAKE_CUDA_ARCHITECTURES=90-real
             -DMLX5_lib=${LIBMLX5_DIR}/libmlx5.so.1
             -DNVSHMEM_BUILD_BITCODE_LIBRARY=0
             -DNVSHMEM_BUILD_EXAMPLES=0
             -DNVSHMEM_BUILD_TESTS=0
  INSTALL_COMMAND ""
  EXCLUDE_FROM_ALL TRUE)
add_library(nvshmem_project::nvshmem STATIC IMPORTED)
add_dependencies(nvshmem_project::nvshmem nvshmem_project)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build/src/include)
set_target_properties(
  nvshmem_project::nvshmem
  PROPERTIES IMPORTED_LOCATION
             ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build/src/lib/libnvshmem.a
             INTERFACE_INCLUDE_DIRECTORIES
             ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build/src/include)

# Find torch_python
find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)

# Add deep_ep_cpp_tllm
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
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
  PROPERTIES CXX_STANDARD_REQUIRED ON CUDA_STANDARD_REQUIRED ON CXX_STANDARD 17
             CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
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
