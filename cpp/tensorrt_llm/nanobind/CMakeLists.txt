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
    batch_manager/llmRequest.cpp
    common/tllmExceptions.cpp
    executor/bindings.cpp
    executor/executor.cpp
    executor/executorConfig.cpp
    executor/request.cpp
    process_group/bindings.cpp
    runtime/bindings.cpp
    runtime/hostfunc.cpp
    runtime/moeBindings.cpp
    suffixAutomaton/bindings.cpp
    testing/kvCacheManagerTestUtilBinding.cpp
    testing/modelSpecBinding.cpp
    userbuffers/bindings.cpp
    thop/compactPseudoKvBindings.cpp
    thop/bindings.cpp
    ../runtime/ipcNvlsMemory.cu
    bindings.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

# NOSTRIP: Disable stripping for BOLT compatibility (requires --emit-relocs)
if(ENABLE_BOLT_COMPATIBLE)
  nanobind_add_module(${TRTLLM_NB_MODULE} NOSTRIP ${SRCS})
else()
  nanobind_add_module(${TRTLLM_NB_MODULE} ${SRCS})
endif()

nanobind_add_module(compact_pseudokv_thop thop/compactPseudoKvModule.cpp
                    thop/compactPseudoKvBindings.cpp)
nanobind_add_module(compact_pseudokv_internal_bindings
                    thop/compactPseudoKvInternalModule.cpp
                    thop/compactPseudoKvBindings.cpp)
nanobind_add_module(compact_pseudokv_bindings_shim
                    thop/compactPseudoKvBindingsShimModule.cpp
                    thop/compactPseudoKvBindings.cpp)

set_property(TARGET ${TRTLLM_NB_MODULE} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET compact_pseudokv_thop PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET compact_pseudokv_internal_bindings
             PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET compact_pseudokv_bindings_shim
             PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(
  compact_pseudokv_bindings_shim
  PROPERTIES OUTPUT_NAME "bindings"
             LIBRARY_OUTPUT_DIRECTORY
             "${CMAKE_BINARY_DIR}/compact_pseudokv_overlay/tensorrt_llm")

target_link_directories(${TRTLLM_NB_MODULE} PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")
target_link_directories(compact_pseudokv_thop PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")
target_link_directories(compact_pseudokv_internal_bindings PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")
target_link_directories(compact_pseudokv_bindings_shim PUBLIC
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
         th_compact_pseudokv
         pg_utils)

target_link_libraries(compact_pseudokv_thop PUBLIC ${Python3_LIBRARIES}
                                                   ${TORCH_LIBRARIES}
                                                   torch_python
                                                   ${CUDA_DRV_LIB}
                                                   th_compact_pseudokv_common
                                                   th_compact_pseudokv)
target_link_libraries(compact_pseudokv_internal_bindings
                      PUBLIC ${Python3_LIBRARIES}
                             ${TORCH_LIBRARIES}
                             torch_python
                             ${CUDA_DRV_LIB}
                             th_compact_pseudokv_common
                             th_compact_pseudokv)
target_link_libraries(compact_pseudokv_bindings_shim
                      PUBLIC ${Python3_LIBRARIES}
                             ${TORCH_LIBRARIES}
                             torch_python
                             ${CUDA_DRV_LIB}
                             th_compact_pseudokv_common
                             th_compact_pseudokv)
target_compile_definitions(
  ${TRTLLM_NB_MODULE} PUBLIC TRTLLM_NB_MODULE=${TRTLLM_NB_MODULE}
                             PYBIND11_DETAILED_ERROR_MESSAGES=1)
target_compile_definitions(compact_pseudokv_thop
                           PUBLIC PYBIND11_DETAILED_ERROR_MESSAGES=1)
target_compile_definitions(compact_pseudokv_internal_bindings
                           PUBLIC PYBIND11_DETAILED_ERROR_MESSAGES=1)
target_compile_definitions(compact_pseudokv_bindings_shim
                           PUBLIC PYBIND11_DETAILED_ERROR_MESSAGES=1)

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
  set_target_properties(compact_pseudokv_thop
                        PROPERTIES BUILD_RPATH
                                   "${TRTLLM_NB_MODULE_RPATH_LIST}")
  set_target_properties(compact_pseudokv_thop
                        PROPERTIES LINK_FLAGS
                                   "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
  set_target_properties(compact_pseudokv_internal_bindings
                        PROPERTIES BUILD_RPATH
                                   "${TRTLLM_NB_MODULE_RPATH_LIST}")
  set_target_properties(compact_pseudokv_internal_bindings
                        PROPERTIES LINK_FLAGS
                                   "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
  set_target_properties(compact_pseudokv_bindings_shim
                        PROPERTIES BUILD_RPATH
                                   "${TRTLLM_NB_MODULE_RPATH_LIST}")
  set_target_properties(compact_pseudokv_bindings_shim
                        PROPERTIES LINK_FLAGS
                                   "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
endif()

# Build transfer_agent_binding when building bindings (if NIXL is enabled)
if(TARGET ${TRANSFER_AGENT_BINDING_TARGET})
  add_dependencies(${TRTLLM_NB_MODULE} ${TRANSFER_AGENT_BINDING_TARGET})
endif()
