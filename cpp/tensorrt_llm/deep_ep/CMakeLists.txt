set(NVSHMEM_ROOT_DIR /opt/custom_nvshmem)
set(DEEP_EP_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/DeepEP)

find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT_DIR}/lib/cmake/nvshmem)
find_library(TORCH_PYTHON_LIB torch_python REQUIRED
             HINTS ${TORCH_INSTALL_PREFIX}/lib)

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
  deep_ep_cpp_tllm PRIVATE nvshmem::nvshmem ${TORCH_LIBRARIES}
                           ${TORCH_PYTHON_LIB})
target_link_options(
  deep_ep_cpp_tllm PRIVATE
  -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/deep_ep_cpp_tllm.version
  -Wl,--no-undefined-version)
