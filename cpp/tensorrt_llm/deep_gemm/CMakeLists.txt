add_custom_target(deep_gemm)

# CUDA architectures
# ==================

# Filter CUDA arch >= 9.0 (deep_gemm requirements)
set(DEEP_GEMM_CUDA_ARCHITECTURES "")
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
    list(APPEND DEEP_GEMM_CUDA_ARCHITECTURES
         "${CUDA_ARCH_MAJOR}${CUDA_ARCH_MINOR}${CUDA_ARCH_POSTFIX}")
  endif()
endforeach()

# Skip build if there is no suitable CUDA arch
if(WIN32)
  set(DEEP_GEMM_CUDA_ARCHITECTURES "")
endif()
message(
  STATUS
    "deep_gemm DEEP_GEMM_CUDA_ARCHITECTURES: ${DEEP_GEMM_CUDA_ARCHITECTURES}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cuda_architectures.txt
     "${DEEP_GEMM_CUDA_ARCHITECTURES}")
if(NOT DEEP_GEMM_CUDA_ARCHITECTURES)
  return()
endif()

# Prepare files
# =============

# Use DeepGEMM submodule
set(DEEP_GEMM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/deepgemm)
get_filename_component(DEEP_GEMM_SOURCE_DIR ${DEEP_GEMM_SOURCE_DIR} ABSOLUTE)

if(NOT EXISTS ${DEEP_GEMM_SOURCE_DIR})
  message(
    FATAL_ERROR
      "DeepGEMM submodule not found at ${DEEP_GEMM_SOURCE_DIR}. Please run: git submodule update --init --recursive"
  )
endif()

# Check if submodules are initialized
if(NOT EXISTS ${DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include)
  message(
    FATAL_ERROR
      "DeepGEMM submodules not initialized. Please run: git submodule update --init --recursive"
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
    string(REPLACE "deep_gemm_cpp" "tensorrt_llm.deep_gemm_cpp_tllm" _content
                   "${_content}")

    # Add adaptation header
    string(
      PREPEND
      _content
      "# Adapted from https://github.com/zongfeijing/DeepGEMM/blob/war/deep_gemm/${REL_PATH}\n"
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

# Copy third-party includes (cutlass and fmt) to the include directory
set(DEEP_GEMM_INCLUDE_DEST ${DEEP_GEMM_PYTHON_DEST}/include)
file(MAKE_DIRECTORY ${DEEP_GEMM_INCLUDE_DEST})
file(COPY ${DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include/cute
     DESTINATION ${DEEP_GEMM_INCLUDE_DEST})
file(COPY ${DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include/cutlass
     DESTINATION ${DEEP_GEMM_INCLUDE_DEST})

# Generate envs.py file similar to deep_gemm's setup.py
set(ENVS_CONTENT
    "# Pre-installed environment variables\npersistent_envs = dict()\n")
file(WRITE ${DEEP_GEMM_PYTHON_DEST}/envs.py "${ENVS_CONTENT}")

# Let CMake generate `fatbinData` for CUDA separable compilation
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

# Find torch_python
find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)

# Build deep_gemm_cpp_tllm extension (matching deep_gemm's setup.py)
set(DEEP_GEMM_SOURCES ${DEEP_GEMM_SOURCE_DIR}/csrc/python_api.cpp)

pybind11_add_module(deep_gemm_cpp_tllm ${DEEP_GEMM_SOURCES})
set_target_properties(
  deep_gemm_cpp_tllm
  PROPERTIES CXX_STANDARD_REQUIRED ON
             CXX_STANDARD 20
             CUDA_SEPARABLE_COMPILATION OFF
             CUDA_ARCHITECTURES "${DEEP_GEMM_CUDA_ARCHITECTURES}"
             LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/deep_gemm_cpp_tllm.version
             INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)

target_compile_options(deep_gemm_cpp_tllm PRIVATE ${TORCH_CXX_FLAGS} -std=c++20
                                                  -O3 -fPIC -Wno-psabi)

# Extension name definition
target_compile_definitions(deep_gemm_cpp_tllm
                           PRIVATE TORCH_EXTENSION_NAME=deep_gemm_cpp_tllm)

# Include directories matching deep_gemm setup.py
target_include_directories(
  deep_gemm_cpp_tllm
  PRIVATE ${CUDA_INCLUDE_DIRS} ${DEEP_GEMM_SOURCE_DIR}/deep_gemm/include
          ${DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include
          ${DEEP_GEMM_SOURCE_DIR}/third-party/fmt/include)

# Link libraries (matching deep_gemm setup.py: cuda, cudart + torch)
target_link_libraries(
  deep_gemm_cpp_tllm PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB} cuda cudart)

# Link directories
target_link_directories(
  deep_gemm_cpp_tllm PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

# Set targets
# ===========
add_dependencies(deep_gemm deep_gemm_cpp_tllm)
