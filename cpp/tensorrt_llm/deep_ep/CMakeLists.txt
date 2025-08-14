set(DEEP_EP_COMMIT edf3ea2b086a393d3163bf2773eab69d9191cc01)
set(NVSHMEM_URL_HASH
    SHA256=eb2c8fb3b7084c2db86bd9fd905387909f1dfd483e7b45f7b3c3d5fcf5374b5a)

add_custom_target(deep_ep)

# CUDA architectures
# ==================

# Filter CUDA arch >= 9.0
set(DEEP_EP_CUDA_ARCHITECTURES "")
foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX MATCHALL "^([1-9][0-9]*)([0-9])[af]?(-real|-virtual)?$" MATCHES
               ${CUDA_ARCH})
  if(NOT CMAKE_MATCH_0)
    message(FATAL_ERROR "Invalid CUDA arch format: \"${CUDA_ARCH}\"")
  endif()
  set(CUDA_ARCH_MAJOR ${CMAKE_MATCH_1})
  set(CUDA_ARCH_MINOR ${CMAKE_MATCH_2})
  set(CUDA_ARCH_POSTFIX ${CMAKE_MATCH_3})
  if(${CUDA_ARCH_MAJOR} GREATER_EQUAL 9)
    list(APPEND DEEP_EP_CUDA_ARCHITECTURES
         "${CUDA_ARCH_MAJOR}${CUDA_ARCH_MINOR}${CUDA_ARCH_POSTFIX}")
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

# Ensure that dependent libraries are installed
find_library(MLX5_lib NAMES mlx5 REQUIRED)
set(NVSHMEM_INSTALL_PREFIX "${TORCH_INSTALL_PREFIX}/../nvidia/shmem")
# message("Torch Install Prefix: ${TORCH_INSTALL_PREFIX}")

find_path(NVSHMEM_INCLUDE_DIR nvshmem.h HINTS ${NVSHMEM_INSTALL_PREFIX}/include)
find_library(NVSHMEM_HOST_LIBRARY nvshmem_host.so.3 HINTS ${NVSHMEM_INSTALL_PREFIX}/lib)
find_library(NVSHMEM_DEVICE_LIBRARY nvshmem_device HINTS ${NVSHMEM_INSTALL_PREFIX}/lib)


# Prepare files
# =============

# Download DeepEP
include(FetchContent)
if(DEFINED ENV{GITHUB_MIRROR} AND NOT "$ENV{GITHUB_MIRROR}" STREQUAL "")
  set(GITHUB_URL "$ENV{GITHUB_MIRROR}")
else()
  set(GITHUB_URL "https://github.com")
endif()
set(DEEP_EP_URL
    "${GITHUB_URL}/deepseek-ai/DeepEP/archive/${DEEP_EP_COMMIT}.tar.gz")
message(STATUS "deep_ep DEEP_EP_URL: ${DEEP_EP_URL}")
FetchContent_Declare(deep_ep_download URL ${DEEP_EP_URL})
FetchContent_MakeAvailable(deep_ep_download)
set(DEEP_EP_SOURCE_DIR ${deep_ep_download_SOURCE_DIR})

# Copy and update python files
set(DEEP_EP_PYTHON_DEST ${CMAKE_CURRENT_BINARY_DIR}/python/deep_ep)
file(REMOVE_RECURSE ${DEEP_EP_PYTHON_DEST})
file(MAKE_DIRECTORY ${DEEP_EP_PYTHON_DEST})
configure_file(${DEEP_EP_SOURCE_DIR}/LICENSE ${DEEP_EP_PYTHON_DEST}/LICENSE
               COPYONLY)
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
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${_src})
endforeach()


# Add DeepEP cpp
# ==============

# Let CMake generate `fatbinData` for CUDA separable compilation. Set to FALSE
# or TRUE are both OK, but it generates `code=lto_90a` rather than `code=sm_90a`
# for arch `90a-real` if set to TRUE.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

# Find torch_python
find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)

# Add deep_ep_cpp_tllm
file(GLOB_RECURSE SRC_CPP ${DEEP_EP_SOURCE_DIR}/csrc/*.cpp)
file(GLOB_RECURSE SRC_CU ${DEEP_EP_SOURCE_DIR}/csrc/*.cu)
pybind11_add_module(deep_ep_cpp_tllm ${SRC_CPP} ${SRC_CU})
set_target_properties(
  deep_ep_cpp_tllm
  PROPERTIES CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD_REQUIRED ON
             CXX_STANDARD 17
             CUDA_STANDARD 17
             CUDA_SEPARABLE_COMPILATION ON
             CUDA_ARCHITECTURES "${DEEP_EP_CUDA_ARCHITECTURES}"
             LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/deep_ep_cpp_tllm.version
             INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)
target_compile_options(
  deep_ep_cpp_tllm
  PRIVATE ${TORCH_CXX_FLAGS} -O3 $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
          $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=--register-usage-level=10>)
target_compile_definitions(
  deep_ep_cpp_tllm PRIVATE DISABLE_AGGRESSIVE_PTX_INSTRS
                           TORCH_EXTENSION_NAME=deep_ep_cpp_tllm)
target_include_directories(deep_ep_cpp_tllm PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(
  deep_ep_cpp_tllm PRIVATE ${NVSHMEM_DEVICE_LIBRARY} ${TORCH_LIBRARIES}
                           ${TORCH_PYTHON_LIB})
target_link_options(
  deep_ep_cpp_tllm PRIVATE
  -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/deep_ep_cpp_tllm.version
  -Wl,--no-undefined-version)

# Set targets
# ===========
add_dependencies(deep_ep deep_ep_cpp_tllm)
