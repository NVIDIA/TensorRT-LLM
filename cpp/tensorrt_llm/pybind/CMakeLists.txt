set(TRTLLM_PYBIND_MODULE bindings)
set(TRTLLM_PYBIND_MODULE
    ${TRTLLM_PYBIND_MODULE}
    PARENT_SCOPE)

if(NOT BUILD_PYT)
  message(
    FATAL_ERROR
      "Python bindings for C++ runtime require PyTorch. Please enable BUILD_PYT"
  )
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} "-c"
          "import pybind11 as pb11; print(pb11.get_cmake_dir(),end='');"
  RESULT_VARIABLE PYBIND_CMAKE_DIR_RET
  OUTPUT_VARIABLE PYBIND_CMAKE_DIR)

if(PYBIND_CMAKE_DIR_RET MATCHES 0)
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND_CMAKE_DIR}")
else()
  message(ERROR "pybind11 CMake directory not found.")
endif()

find_package(pybind11 REQUIRED)

set(SRCS
    bindings.cpp
    batch_manager/gptManager.cpp
    batch_manager/llmRequest.cpp
    batch_manager/inferenceRequest.cpp
    batch_manager/namedTensor.cpp
    runtime/generationInput.cpp
    runtime/generationOutput.cpp)

pybind11_add_module(${TRTLLM_PYBIND_MODULE} ${SRCS})

set_property(TARGET ${TRTLLM_PYBIND_MODULE} PROPERTY POSITION_INDEPENDENT_CODE
                                                     ON)

target_link_directories(${TRTLLM_PYBIND_MODULE} PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(
  ${TRTLLM_PYBIND_MODULE}
  PUBLIC ${STATIC_TARGET} ${Python3_LIBRARIES} ${TORCH_LIBRARIES} torch_python
         ${UNDEFINED_FLAG})
target_compile_definitions(${TRTLLM_PYBIND_MODULE}
                           PUBLIC TRTLLM_PYBIND_MODULE=${TRTLLM_PYBIND_MODULE})
