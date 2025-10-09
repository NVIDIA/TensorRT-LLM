add_custom_target(flash_mla)

if(WIN32)
  return()
endif()

# Prepare files
# =============

# Use FlashMLA submodule
set(FLASH_MLA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/flash-mla)
get_filename_component(FLASH_MLA_SOURCE_DIR ${FLASH_MLA_SOURCE_DIR} ABSOLUTE)

if(NOT EXISTS ${FLASH_MLA_SOURCE_DIR})
  message(
    FATAL_ERROR
      "FlashMLA submodule not found at ${FLASH_MLA_SOURCE_DIR}. Please run: git submodule update --init --recursive"
  )
endif()

# Check if submodules are initialized
if(NOT EXISTS ${FLASH_MLA_SOURCE_DIR}/csrc/cutlass/include)
  message(
    FATAL_ERROR
      "FlashMLA submodules not initialized. Please run: cd ${FLASH_MLA_SOURCE_DIR} && git submodule update --init --recursive"
  )
endif()

# Check CUDA version and architecture support
# ============================================

# FlashMLA requires CUDA 12.3+ for SM90, 12.9+ for SM100
set(SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3)
  list(APPEND SUPPORT_ARCHS "90")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
  list(APPEND SUPPORT_ARCHS "100")
endif()

# Find intersection of supported and requested architectures
set(FLASH_MLA_ARCHS)
foreach(ARCH ${SUPPORT_ARCHS})
  if("${ARCH}" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
    list(APPEND FLASH_MLA_ARCHS ${ARCH})
  endif()
endforeach()

if(NOT FLASH_MLA_ARCHS)
  message(
    STATUS
      "FlashMLA requires SM90 (CUDA 12.3+) or SM100 (CUDA 12.9+), skipping. "
      "Current CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}, "
      "Current architectures: ${CMAKE_CUDA_ARCHITECTURES_ORIG}")
  return()
endif()

message(STATUS "Building FlashMLA for architectures: ${FLASH_MLA_ARCHS}")

# Copy and update python files
# =============================
set(FLASH_MLA_PYTHON_DEST ${CMAKE_CURRENT_BINARY_DIR}/python/flash_mla)
file(REMOVE_RECURSE ${FLASH_MLA_PYTHON_DEST})
file(MAKE_DIRECTORY ${FLASH_MLA_PYTHON_DEST})

# Copy all files from flash_mla directory
file(GLOB_RECURSE FLASH_MLA_ALL_FILES ${FLASH_MLA_SOURCE_DIR}/flash_mla/*)
configure_file(${FLASH_MLA_SOURCE_DIR}/LICENSE ${FLASH_MLA_PYTHON_DEST}/LICENSE
               COPYONLY)

foreach(SOURCE_FILE ${FLASH_MLA_ALL_FILES})
  file(RELATIVE_PATH REL_PATH ${FLASH_MLA_SOURCE_DIR}/flash_mla ${SOURCE_FILE})
  get_filename_component(REL_DIR ${REL_PATH} DIRECTORY)
  file(MAKE_DIRECTORY ${FLASH_MLA_PYTHON_DEST}/${REL_DIR})

  # Check if it's a Python file that needs import renaming
  get_filename_component(FILE_EXT ${SOURCE_FILE} EXT)
  if(FILE_EXT STREQUAL ".py")
    # Read file content and replace module imports for Python files
    file(READ ${SOURCE_FILE} _content)
    # Replace the C++ extension module import
    string(REPLACE "flash_mla.cuda" "tensorrt_llm.flash_mla_cpp_tllm" _content
                   "${_content}")
    # Replace absolute imports with relative imports for internal modules
    string(REPLACE "from flash_mla." "from ." _content
                   "${_content}")

    # Add adaptation header
    string(
      PREPEND
      _content
      "# Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/${REL_PATH}\n"
    )

    # Write modified content
    set(_dst "${FLASH_MLA_PYTHON_DEST}/${REL_PATH}")
    file(WRITE ${_dst} "${_content}")
  else()
    # Copy non-Python files as-is
    set(_dst "${FLASH_MLA_PYTHON_DEST}/${REL_PATH}")
    file(COPY ${SOURCE_FILE} DESTINATION ${FLASH_MLA_PYTHON_DEST}/${REL_DIR})
  endif()

  # Add dependency tracking
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${SOURCE_FILE})
endforeach()

# Build flash_mla_cpp_tllm extension
# ===================================

# Find torch_python
find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)

# Define source files matching FlashMLA's setup.py
# Note: pybind.cpp has runtime checks (arch.is_sm90()), but still references all kernel symbols
# We compile all sources to avoid undefined symbols at module load time
set(FLASH_MLA_SOURCES
    ${FLASH_MLA_SOURCE_DIR}/csrc/pybind.cpp
    ${FLASH_MLA_SOURCE_DIR}/csrc/smxx/get_mla_metadata.cu
    ${FLASH_MLA_SOURCE_DIR}/csrc/smxx/mla_combine.cu)

# Add SM90 sources (always include if CUDA >= 12.3)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3)
  list(APPEND FLASH_MLA_SOURCES
       ${FLASH_MLA_SOURCE_DIR}/csrc/sm90/decode/dense/splitkv_mla.cu
       ${FLASH_MLA_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/splitkv_mla.cu
       ${FLASH_MLA_SOURCE_DIR}/csrc/sm90/prefill/sparse/fwd.cu)
endif()

# Add SM100 sources (always include if CUDA >= 12.9)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
  list(APPEND FLASH_MLA_SOURCES
       ${FLASH_MLA_SOURCE_DIR}/csrc/sm100/decode/sparse_fp8/splitkv_mla.cu
       ${FLASH_MLA_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu
       ${FLASH_MLA_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu
       ${FLASH_MLA_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd.cu)
endif()

# Disable LTO before creating target (matching DeepEP's approach)
# Let CMake generate fatbinData for CUDA separable compilation
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

pybind11_add_module(flash_mla_cpp_tllm ${FLASH_MLA_SOURCES})

set_target_properties(
  flash_mla_cpp_tllm
  PROPERTIES CXX_STANDARD_REQUIRED ON
             CXX_STANDARD 17
             CXX_SCAN_FOR_MODULES OFF
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             CUDA_SEPARABLE_COMPILATION ON
             CUDA_RESOLVE_DEVICE_SYMBOLS ON
             LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/flash_mla_cpp_tllm.version
             INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)

# Set CUDA architectures
# Compile kernels for all CUDA-version-supported architectures since pybind.cpp references them
set(FLASH_MLA_BUILD_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3)
  list(APPEND FLASH_MLA_BUILD_ARCHS "90")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
  list(APPEND FLASH_MLA_BUILD_ARCHS "100f")
endif()
set_cuda_architectures(flash_mla_cpp_tllm ${FLASH_MLA_BUILD_ARCHS})

# Compiler options matching FlashMLA setup.py
target_compile_options(
  flash_mla_cpp_tllm
  PRIVATE ${TORCH_CXX_FLAGS}
          $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
          $<$<COMPILE_LANGUAGE:CXX>:-O3>
          $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
          $<$<COMPILE_LANGUAGE:CXX>:-DNDEBUG>
          $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
          $<$<COMPILE_LANGUAGE:CXX>:-fno-lto>
          $<$<COMPILE_LANGUAGE:CUDA>:-O3>
          $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
          $<$<COMPILE_LANGUAGE:CUDA>:-DNDEBUG>
          $<$<COMPILE_LANGUAGE:CUDA>:-D_USE_MATH_DEFINES>
          $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>
          $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
          $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
          $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF2_OPERATORS__>
          $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
          $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
          $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v,--register-usage-level=10>)

# Extension name definition
target_compile_definitions(flash_mla_cpp_tllm
                           PRIVATE TORCH_EXTENSION_NAME=flash_mla_cpp_tllm)


# Include directories matching FlashMLA setup.py
target_include_directories(
  flash_mla_cpp_tllm
  PRIVATE ${CUDA_INCLUDE_DIRS} ${FLASH_MLA_SOURCE_DIR}/csrc
          ${FLASH_MLA_SOURCE_DIR}/csrc/sm90
          ${FLASH_MLA_SOURCE_DIR}/csrc/cutlass/include
          ${FLASH_MLA_SOURCE_DIR}/csrc/cutlass/tools/util/include)

# Link libraries (matching FlashMLA setup.py: cuda, cudart + torch)
target_link_libraries(
  flash_mla_cpp_tllm PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB}
                             CUDA::cuda_driver CUDA::cudart CUDA::cudart_static)

# Link directories
target_link_directories(
  flash_mla_cpp_tllm PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

# Set targets
# ===========
add_dependencies(flash_mla flash_mla_cpp_tllm)

