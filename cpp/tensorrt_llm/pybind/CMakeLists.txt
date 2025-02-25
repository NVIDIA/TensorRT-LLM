set(TRTLLM_PYBIND_MODULE bindings)
set(TRTLLM_PYBIND_MODULE
    ${TRTLLM_PYBIND_MODULE}
    PARENT_SCOPE)

set(SRCS
    batch_manager/algorithms.cpp
    batch_manager/bindings.cpp
    batch_manager/buffers.cpp
    batch_manager/cacheTransceiver.cpp
    batch_manager/kvCacheManager.cpp
    batch_manager/llmRequest.cpp
    executor/bindings.cpp
    executor/executor.cpp
    runtime/bindings.cpp
    userbuffers/bindings.cpp
    ../runtime/ipcNvlsMemory.cpp
    bindings.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

pybind11_add_module(${TRTLLM_PYBIND_MODULE} ${SRCS})

set_property(TARGET ${TRTLLM_PYBIND_MODULE} PROPERTY POSITION_INDEPENDENT_CODE
                                                     ON)

target_link_directories(${TRTLLM_PYBIND_MODULE} PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(
  ${TRTLLM_PYBIND_MODULE}
  PUBLIC ${SHARED_TARGET} ${UNDEFINED_FLAG} ${NO_AS_NEEDED_FLAG}
         ${Python3_LIBRARIES} ${TORCH_LIBRARIES} torch_python)
target_compile_definitions(
  ${TRTLLM_PYBIND_MODULE} PUBLIC TRTLLM_PYBIND_MODULE=${TRTLLM_PYBIND_MODULE}
                                 PYBIND11_DETAILED_ERROR_MESSAGES=1)

if(NOT WIN32)
  set_target_properties(
    ${TRTLLM_PYBIND_MODULE}
    PROPERTIES
      LINK_FLAGS
      "-Wl,-rpath,'$ORIGIN/libs' -Wl,-rpath,'$ORIGIN/../nvidia/nccl/lib' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}"
  )
endif()
