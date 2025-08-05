add_custom_target(deep_gemm)

if(WIN32)
  return()
endif()

# Prepare files
# =============

# Use DeepGEMM submodule
set(DEEP_GEMM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/DeepGEMM)
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

# Copy third-party includes (cutlass and fmt) to the include directory
set(DEEP_GEMM_INCLUDE_DEST ${DEEP_GEMM_PYTHON_DEST}/include)
file(MAKE_DIRECTORY ${DEEP_GEMM_INCLUDE_DEST})
file(COPY ${DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include/cute
     DESTINATION ${DEEP_GEMM_INCLUDE_DEST})
file(COPY ${DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include/cutlass
     DESTINATION ${DEEP_GEMM_INCLUDE_DEST})

# Find torch_python
find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)

# Build deep_gemm_cpp_tllm extension (matching deep_gemm's setup.py)
set(DEEP_GEMM_SOURCES ${DEEP_GEMM_SOURCE_DIR}/csrc/python_api.cpp)

pybind11_add_module(deep_gemm_cpp_tllm ${DEEP_GEMM_SOURCES})
set_target_properties(
  deep_gemm_cpp_tllm
  PROPERTIES CXX_STANDARD_REQUIRED ON
             CXX_STANDARD 17
             CXX_SCAN_FOR_MODULES OFF
             CUDA_SEPARABLE_COMPILATION ON
             LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/deep_gemm_cpp_tllm.version
             INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)

target_compile_options(deep_gemm_cpp_tllm PRIVATE ${TORCH_CXX_FLAGS} -std=c++17
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
  deep_gemm_cpp_tllm PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB}
                             CUDA::cuda_driver CUDA::cudart)

# Link directories
target_link_directories(
  deep_gemm_cpp_tllm PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

# Set targets
# ===========
add_dependencies(deep_gemm deep_gemm_cpp_tllm)
