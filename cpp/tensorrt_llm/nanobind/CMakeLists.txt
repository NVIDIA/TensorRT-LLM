set(TRTLLM_NB_MODULE bindings)
set(TRTLLM_NB_MODULE
    ${TRTLLM_NB_MODULE}
    PARENT_SCOPE)

set(SRCS ../runtime/ipcNvlsMemory.cu bindings.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

nanobind_add_module(${TRTLLM_NB_MODULE} ${SRCS})

set_property(TARGET ${TRTLLM_NB_MODULE} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_directories(${TRTLLM_NB_MODULE} PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(
  ${TRTLLM_NB_MODULE}
  PUBLIC ${SHARED_TARGET} ${UNDEFINED_FLAG} ${NO_AS_NEEDED_FLAG}
         ${Python3_LIBRARIES} ${TORCH_LIBRARIES} torch_python)

target_compile_definitions(
  ${TRTLLM_NB_MODULE} PUBLIC TRTLLM_NB_MODULE=${TRTLLM_NB_MODULE}
                             NB_DETAILED_ERROR_MESSAGES=1)

if(NOT WIN32)
  set_target_properties(
    ${TRTLLM_NB_MODULE}
    PROPERTIES
      LINK_FLAGS
      "-Wl,-rpath,'$ORIGIN/libs' -Wl,-rpath,'$ORIGIN/../nvidia/nccl/lib' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}"
  )
endif()
