set(DEEP_EP_COMMIT 5be51b228a7c82dbdb213ea58e77bffd12b38af8)
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
    # The FP4-related conversion instructions in DeepEP require SM100a, SM110a,
    # or SM120a.
    if(${CUDA_ARCH_MAJOR} EQUAL 10 AND ${CUDA_ARCH_MINOR} EQUAL 0)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
        list(APPEND DEEP_EP_CUDA_ARCHITECTURES "100f${CUDA_ARCH_POSTFIX}")
      else()
        list(APPEND DEEP_EP_CUDA_ARCHITECTURES "100a${CUDA_ARCH_POSTFIX}"
             "103a${CUDA_ARCH_POSTFIX}")
      endif()
    elseif(${CUDA_ARCH_MAJOR} GREATER_EQUAL 10 AND ${CUDA_ARCH_MINOR} EQUAL 0)
      list(APPEND DEEP_EP_CUDA_ARCHITECTURES
           "${CUDA_ARCH_MAJOR}${CUDA_ARCH_MINOR}a${CUDA_ARCH_POSTFIX}")
    else()
      list(APPEND DEEP_EP_CUDA_ARCHITECTURES
           "${CUDA_ARCH_MAJOR}${CUDA_ARCH_MINOR}${CUDA_ARCH_POSTFIX}")
    endif()
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

# Delete stale nvshmem on patch update
set(NVSHMEM_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/nvshmem_stamp.txt)
file(SHA256 ${DEEP_EP_SOURCE_DIR}/third-party/nvshmem.patch NVSHMEM_PATCH_HASH)
file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/nvshmem_fast_build.patch
     NVSHMEM_PATCH_2_HASH)
set(NVSHMEM_STAMP_CONTENT "${NVSHMEM_URL_HASH}")
string(APPEND NVSHMEM_STAMP_CONTENT " PATCH_COMMAND v1")
string(APPEND NVSHMEM_STAMP_CONTENT " ${NVSHMEM_PATCH_HASH}")
string(APPEND NVSHMEM_STAMP_CONTENT " 103")
string(APPEND NVSHMEM_STAMP_CONTENT " ${NVSHMEM_PATCH_2_HASH}")
set(OLD_NVSHMEM_STAMP_CONTENT "")
if(EXISTS ${NVSHMEM_STAMP_FILE})
  file(READ ${NVSHMEM_STAMP_FILE} OLD_NVSHMEM_STAMP_CONTENT)
endif()
if(NOT OLD_NVSHMEM_STAMP_CONTENT STREQUAL NVSHMEM_STAMP_CONTENT)
  file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/nvshmem_project-prefix)
  file(WRITE ${NVSHMEM_STAMP_FILE} "${NVSHMEM_STAMP_CONTENT}")
endif()
set_property(
  DIRECTORY APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS
           ${DEEP_EP_SOURCE_DIR}/third-party/nvshmem.patch
           ${CMAKE_CURRENT_SOURCE_DIR}/nvshmem_fast_build.patch)

# Add NVSHMEM
# ===========

# NVSHMEM only works with GCC. Building NVSHMEM with Clang results in
# compilation errors. Using NVSHMEM with Clang results in slow builds and device
# link issues.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  set(CMAKE_CUDA_HOST_COMPILER g++)
endif()

# Add nvshmem external project
include(ExternalProject)
ExternalProject_Add(
  nvshmem_project
  URL file://${CMAKE_CURRENT_SOURCE_DIR}/nvshmem_src_3.2.5-1.txz
  URL_HASH ${NVSHMEM_URL_HASH}
  PATCH_COMMAND patch -p1 --forward --batch -i
                ${DEEP_EP_SOURCE_DIR}/third-party/nvshmem.patch
  COMMAND sed "s/TRANSPORT_VERSION_MAJOR 3/TRANSPORT_VERSION_MAJOR 103/" -i
          src/CMakeLists.txt
  COMMAND sed "s/_STANDARD 11/_STANDARD 17/" -i src/device/CMakeLists.txt
  COMMAND sed "s/_STANDARD 11/_STANDARD 17/" -i src/CMakeLists.txt
  COMMAND patch -p1 --forward --batch -i
          ${CMAKE_CURRENT_SOURCE_DIR}/nvshmem_fast_build.patch
  CMAKE_CACHE_ARGS
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER:STRING=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CUDA_ARCHITECTURES:STRING=${DEEP_EP_CUDA_ARCHITECTURES}
    -DCMAKE_CUDA_HOST_COMPILER:STRING=${CMAKE_CUDA_HOST_COMPILER}
    -DCMAKE_CUDA_COMPILER_LAUNCHER:STRING=${CMAKE_CUDA_COMPILER_LAUNCHER}
    -DNVSHMEM_BUILD_EXAMPLES:BOOL=0
    -DNVSHMEM_BUILD_PACKAGES:BOOL=0
    -DNVSHMEM_BUILD_TESTS:BOOL=0
    -DNVSHMEM_IBGDA_SUPPORT:BOOL=1
    -DNVSHMEM_IBRC_SUPPORT:BOOL=0
    -DNVSHMEM_MPI_SUPPORT:BOOL=0
    -DNVSHMEM_PMIX_SUPPORT:BOOL=0
    -DNVSHMEM_SHMEM_SUPPORT:BOOL=0
    -DNVSHMEM_TIMEOUT_DEVICE_POLLING:BOOL=0
    -DNVSHMEM_UCX_SUPPORT:BOOL=0
    -DNVSHMEM_USE_GDRCOPY:BOOL=0
    -DNVSHMEM_USE_NCCL:BOOL=0
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build
  BUILD_BYPRODUCTS
    ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build/src/lib/libnvshmem.a)
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
             INSTALL_RPATH "$ORIGIN/libs/nvshmem;${TORCH_INSTALL_PREFIX}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)
target_compile_options(
  deep_ep_cpp_tllm
  PRIVATE ${TORCH_CXX_FLAGS} -O3 $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
          $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=--register-usage-level=10>)
target_compile_definitions(
  deep_ep_cpp_tllm PRIVATE DISABLE_AGGRESSIVE_PTX_INSTRS
                           TORCH_EXTENSION_NAME=deep_ep_cpp_tllm)
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
