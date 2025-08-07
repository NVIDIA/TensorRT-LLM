set(TRTLLM_PYBIND_MODULE bindings)
set(TRTLLM_PYBIND_MODULE
    ${TRTLLM_PYBIND_MODULE}
    PARENT_SCOPE)

set(SRCS
    batch_manager/algorithms.cpp
    batch_manager/bindings.cpp
    batch_manager/cacheTransceiver.cpp
    batch_manager/kvCacheConnector.cpp
    batch_manager/kvCacheManager.cpp
    batch_manager/llmRequest.cpp
    executor/bindings.cpp
    executor/executor.cpp
    executor/executorConfig.cpp
    executor/request.cpp
    runtime/bindings.cpp
    testing/modelSpecBinding.cpp
    runtime/moeBindings.cpp
    userbuffers/bindings.cpp
    ../runtime/ipcNvlsMemory.cu
    bindings.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

pybind11_add_module(${TRTLLM_PYBIND_MODULE} ${SRCS})

set_property(TARGET ${TRTLLM_PYBIND_MODULE} PROPERTY POSITION_INDEPENDENT_CODE
                                                     ON)

target_link_directories(${TRTLLM_PYBIND_MODULE} PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")

if(ENABLE_NVSHMEM)
  target_link_libraries(${TRTLLM_PYBIND_MODULE} PUBLIC nvshmem::nvshmem_host
                                                       nvshmem::nvshmem_device)
endif()

target_link_libraries(
  ${TRTLLM_PYBIND_MODULE}
  PUBLIC ${SHARED_TARGET}
         ${UNDEFINED_FLAG}
         ${NO_AS_NEEDED_FLAG}
         ${Python3_LIBRARIES}
         ${TORCH_LIBRARIES}
         torch_python
         ${CUDA_NVML_LIB})
target_compile_definitions(
  ${TRTLLM_PYBIND_MODULE} PUBLIC TRTLLM_PYBIND_MODULE=${TRTLLM_PYBIND_MODULE}
                                 PYBIND11_DETAILED_ERROR_MESSAGES=1)

if(NOT WIN32)
  set_target_properties(
    ${TRTLLM_PYBIND_MODULE}
    PROPERTIES
      LINK_FLAGS
      "-Wl,-rpath,'$ORIGIN/libs' -Wl,-rpath,'$ORIGIN/../nvidia/nccl/lib' -Wl,-rpath,'${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/stubs' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}"
  )
endif()
