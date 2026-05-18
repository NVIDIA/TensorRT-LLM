cmake_minimum_required(VERSION 3.27)

set(DOCA_RDMA_EXT_TARGET doca_rdma_ext)
add_custom_target(${DOCA_RDMA_EXT_TARGET})

set(DOCA_RDMA_EXT_CUDA_ARCHS
    "${CMAKE_CUDA_ARCHITECTURES}"
    CACHE STRING "CUDA architectures for the optional DOCA RDMA extension")

set(DOCA_DIR
    "/opt/mellanox/doca"
    CACHE PATH "DOCA SDK installation path")
set(DOCA_INCLUDE_DIR "${DOCA_DIR}/include")
set(DOCA_LIB_DIR "${DOCA_DIR}/lib/x86_64-linux-gnu")

if(NOT EXISTS "${DOCA_INCLUDE_DIR}")
  message(FATAL_ERROR "DOCA include directory not found: ${DOCA_INCLUDE_DIR}")
endif()

find_library(DOCA_GPUNETIO_LIB doca_gpunetio HINTS "${DOCA_LIB_DIR}" REQUIRED)
find_library(DOCA_COMMON_LIB doca_common HINTS "${DOCA_LIB_DIR}" REQUIRED)
find_library(IBVERBS_LIB ibverbs REQUIRED)
find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(PYBIND11_LTO_CXX_FLAGS "")
set(PYBIND11_LTO_LINKER_FLAGS "")

set(DOCA_RDMA_EXT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(DOCA_RDMA_EXT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(doca_rdma_kernels STATIC
            "${DOCA_RDMA_EXT_SRC_DIR}/doca_rdma_kernels.cu")

set_target_properties(
  doca_rdma_kernels
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES
                                           "${DOCA_RDMA_EXT_CUDA_ARCHS}"
             POSITION_INDEPENDENT_CODE ON)

target_compile_options(
  doca_rdma_kernels
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler
          -rdc=true -std=c++17 -Xcompiler=-fPIC>)
target_compile_definitions(doca_rdma_kernels
                           PRIVATE DOCA_ALLOW_EXPERIMENTAL_API)
target_include_directories(
  doca_rdma_kernels PRIVATE "${DOCA_RDMA_EXT_INCLUDE_DIR}"
                            "${DOCA_INCLUDE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(doca_rdma_kernels PRIVATE ${DOCA_GPUNETIO_LIB}
                                                CUDA::cuda_driver CUDA::cudart)

pybind11_add_module(
  doca_rdma "${DOCA_RDMA_EXT_SRC_DIR}/doca_rdma_endpoint.cpp"
  "${DOCA_RDMA_EXT_SRC_DIR}/doca_verbs.cpp"
  "${DOCA_RDMA_EXT_SRC_DIR}/pybind_module.cpp")

target_include_directories(
  doca_rdma PRIVATE "${DOCA_RDMA_EXT_INCLUDE_DIR}" "${DOCA_INCLUDE_DIR}"
                    "${CUDAToolkit_INCLUDE_DIRS}")
target_compile_definitions(doca_rdma PRIVATE DOCA_ALLOW_EXPERIMENTAL_API)
target_compile_options(doca_rdma PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(
  doca_rdma
  PRIVATE doca_rdma_kernels
          ${DOCA_GPUNETIO_LIB}
          ${DOCA_COMMON_LIB}
          CUDA::cuda_driver
          CUDA::cudart
          ${IBVERBS_LIB}
          pthread)
set_target_properties(doca_rdma PROPERTIES INSTALL_RPATH "${DOCA_LIB_DIR}")

add_library(doca_rdma_ops SHARED
            "${DOCA_RDMA_EXT_SRC_DIR}/doca_rdma_torch_ops.cu")

set_target_properties(
  doca_rdma_ops
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
             CUDA_ARCHITECTURES "${DOCA_RDMA_EXT_CUDA_ARCHS}"
             PREFIX ""
             INSTALL_RPATH "${DOCA_LIB_DIR};${TORCH_INSTALL_PREFIX}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)

target_compile_options(
  doca_rdma_ops
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler
          -rdc=true -std=c++17 -Xcompiler=-fPIC>)
target_include_directories(
  doca_rdma_ops
  PRIVATE "${DOCA_RDMA_EXT_INCLUDE_DIR}" "${DOCA_INCLUDE_DIR}"
          "${CUDAToolkit_INCLUDE_DIRS}" ${TORCH_INCLUDE_DIRS}
          ${Python3_INCLUDE_DIRS})
target_compile_definitions(doca_rdma_ops PRIVATE DOCA_ALLOW_EXPERIMENTAL_API)
target_compile_options(doca_rdma_ops PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(
  doca_rdma_ops
  PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB} ${DOCA_GPUNETIO_LIB}
          ${DOCA_COMMON_LIB} CUDA::cuda_driver CUDA::cudart)

add_dependencies(${DOCA_RDMA_EXT_TARGET} doca_rdma doca_rdma_ops)
