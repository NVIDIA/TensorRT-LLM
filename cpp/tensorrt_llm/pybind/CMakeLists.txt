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
    process_group/bindings.cpp
    runtime/bindings.cpp
    runtime/hostfunc.cpp
    common/tllmExceptions.cpp
    testing/modelSpecBinding.cpp
    runtime/moeBindings.cpp
    testing/modelSpecBinding.cpp
    userbuffers/bindings.cpp
    ../runtime/ipcNvlsMemory.cu
    thop/bindings.cpp
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
         ${Python3_LIBRARIES}
         ${TORCH_LIBRARIES}
         torch_python
         ${CUDA_DRV_LIB}
         ${CUDA_NVML_LIB}
         th_common
         pg_utils)
target_compile_definitions(
  ${TRTLLM_PYBIND_MODULE} PUBLIC TRTLLM_PYBIND_MODULE=${TRTLLM_PYBIND_MODULE}
                                 PYBIND11_DETAILED_ERROR_MESSAGES=1)

if(NOT WIN32)
  set(TRTLLM_PYBIND_MODULE_RPATH_LIST
      "$ORIGIN/libs" # TRTLLM libraries
      "$ORIGIN/../../.." # Shared libraries under $PREFIX/lib
      "$ORIGIN/../nvidia/nccl/lib" # NCCL libraries
  )
  set_target_properties(
    ${TRTLLM_PYBIND_MODULE} PROPERTIES BUILD_RPATH
                                       "${TRTLLM_PYBIND_MODULE_RPATH_LIST}")
  set_target_properties(
    ${TRTLLM_PYBIND_MODULE} PROPERTIES LINK_FLAGS
                                       "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
endif()
