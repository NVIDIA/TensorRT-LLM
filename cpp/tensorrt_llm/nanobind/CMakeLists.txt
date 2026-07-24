set(TRTLLM_NB_MODULE bindings)
set(TRTLLM_NB_MODULE
    ${TRTLLM_NB_MODULE}
    PARENT_SCOPE)

set(SRCS
    batch_manager/algorithms.cpp
    batch_manager/bindings.cpp
    batch_manager/buffers.cpp
    batch_manager/cacheTransceiver.cpp
    batch_manager/kvCacheConnector.cpp
    batch_manager/kvCacheManager.cpp
    batch_manager/kvCacheManagerV2Utils.cpp
    batch_manager/kvCacheManagerV2.cpp
    batch_manager/llmRequest.cpp
    common/tllmExceptions.cpp
    executor/bindings.cpp
    executor/executorConfig.cpp
    executor/request.cpp
    process_group/bindings.cpp
    runtime/bindings.cpp
    runtime/hostfunc.cpp
    runtime/moeBindings.cpp
    suffixAutomaton/bindings.cpp
    testing/kvCacheManagerTestUtilBinding.cpp
    userbuffers/bindings.cpp
    thop/bindings.cpp
    ../runtime/ipcNvlsMemory.cu
    bindings.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

# KV Cache Manager V2 include directory (global, needed during compilation of
# SRCS) NOTE: PROJECT_SOURCE_DIR = cpp/ (build_wheel.py passes cpp/ as -S), so
# batch_manager is one level down and the vendored SHA-256 is at
# ../3rdparty/sha256 (repo root).
set(KV_CACHE_MANAGER_V2_INCLUDE
    ${PROJECT_SOURCE_DIR}/tensorrt_llm/batch_manager)
include_directories(${KV_CACHE_MANAGER_V2_INCLUDE}
                    ${PROJECT_SOURCE_DIR}/../3rdparty/sha256)

# NOSTRIP: Disable stripping for BOLT compatibility (requires --emit-relocs)
if(ENABLE_BOLT_COMPATIBLE)
  nanobind_add_module(${TRTLLM_NB_MODULE} NOSTRIP ${SRCS})
else()
  nanobind_add_module(${TRTLLM_NB_MODULE} ${SRCS})
endif()

# SHA-256 symbols are provided by tensorrt_llm_batch_manager_static (via the
# shared lib). Only add the include directory so headers resolve during
# compilation of kvCacheManagerV2.cpp.
target_include_directories(${TRTLLM_NB_MODULE}
                           PRIVATE ${PROJECT_SOURCE_DIR}/../3rdparty/sha256)

set_property(TARGET ${TRTLLM_NB_MODULE} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_directories(${TRTLLM_NB_MODULE} PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")

if(ENABLE_NVSHMEM)
  target_link_libraries(${TRTLLM_NB_MODULE} PUBLIC nvshmem::nvshmem_host
                                                   nvshmem::nvshmem_device)
endif()

target_link_libraries(
  ${TRTLLM_NB_MODULE}
  PUBLIC ${SHARED_TARGET}
         ${Python3_LIBRARIES}
         ${TORCH_LIBRARIES}
         torch_python
         ${CUDA_DRV_LIB}
         th_common
         pg_utils)
target_compile_definitions(
  ${TRTLLM_NB_MODULE} PUBLIC TRTLLM_NB_MODULE=${TRTLLM_NB_MODULE}
                             PYBIND11_DETAILED_ERROR_MESSAGES=1)

if(NOT WIN32)
  set(TRTLLM_NB_MODULE_RPATH_LIST
      "$ORIGIN/libs" # TRTLLM libraries
      "$ORIGIN/../../.." # Shared libraries under $PREFIX/lib
      "$ORIGIN/../nvidia/nccl/lib" # NCCL libraries
  )
  set_target_properties(${TRTLLM_NB_MODULE}
                        PROPERTIES BUILD_RPATH "${TRTLLM_NB_MODULE_RPATH_LIST}")
  set_target_properties(
    ${TRTLLM_NB_MODULE} PROPERTIES LINK_FLAGS
                                   "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
endif()

# Build transfer_agent_binding when building bindings (if NIXL is enabled)
if(TARGET ${TRANSFER_AGENT_BINDING_TARGET})
  add_dependencies(${TRTLLM_NB_MODULE} ${TRANSFER_AGENT_BINDING_TARGET})
endif()
