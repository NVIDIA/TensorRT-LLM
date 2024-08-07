cmake_minimum_required(VERSION 3.1)

# Enable C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
set(TARGET_NAME [[ plugin_lib ]])
project(${TARGET_NAME})

set(CMAKE_VERBOSE_MAKEFILE 1)

# Compile options
set(CMAKE_C_FLAGS "-Wall -pthread ")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

set(CMAKE_BUILD_TYPE release)

find_package(CUDA REQUIRED)

message(STATUS "CUDA library status:")
message(STATUS "    config: ${CUDA_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

if(NOT DEFINED TRT_INCLUDE_DIR)
  set(TRT_INCLUDE_DIR "/usr/local/tensorrt/include")
  if(NOT EXISTS ${TRT_INCLUDE_DIR})
    # In case of TensorRT installed from a deb package.
    set(TRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
  endif()
endif()
message(STATUS "tensorrt include path: ${TRT_INCLUDE_DIR}")


if(NOT DEFINED TRT_LIB_DIR)
  set(TRT_LIB_DIR "/usr/local/tensorrt/lib")
  if(NOT EXISTS ${TRT_INCLUDE_DIR})
    # In case of TensorRT installed from a deb package.
    set(TRT_LIB_DIR "/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
  endif()
endif()
find_library(
  TRT_LIB_PATH nvinfer
  HINTS ${TRT_LIB_DIR}
  NO_DEFAULT_PATH)
find_library(TRT_LIB_PATH nvinfer REQUIRED)
message(STATUS "TRT_LIB_DIR: ${TRT_LIB_DIR}")
message(STATUS "Found nvinfer library: ${TRT_LIB_PATH}")


# Declare the executable target built from your sources
add_library(
  ${TARGET_NAME} SHARED
  ${CMAKE_SOURCE_DIR}/tritonPlugins.cpp
  ${CMAKE_SOURCE_DIR}/plugin_common.cpp
{% for plugin in plugin_names %}
  ${CMAKE_SOURCE_DIR}/[[ plugin ]]/_generate_trt_plugin/plugin.cpp
{% endfor %}
  [[ ' '.join(kernel_object_files) ]]
  )

#set_property(TARGET ${TARGET_NAME} PROPERTY IMPORTED_LOCATION ${TRT_LIB_PATH})
#set_property(TARGET ${TARGET_NAME} PROPERTY IMPORTED_LOCATION
                                            #${TRT_LLM_LIB_PATH})
#target_link_libraries(${TARGET_NAME} LINK_PRIVATE ${CUDA_LIBRARIES})
#target_link_libraries(${TARGET_NAME} LINK_PRIVATE nvinfer)
#target_link_libraries(${TARGET_NAME} LINK_PRIVATE nvinfer_plugin_tensorrt_llm)
#target_link_libraries(${TARGET_NAME} LINK_PRIVATE cuda)

target_link_libraries(
  ${TARGET_NAME} PUBLIC cuda ${CUDA_LIBRARIES} ${TRT_LIB_PATH})

if(NOT MSVC)
  set_property(TARGET ${TARGET_NAME} PROPERTY LINK_FLAGS "-Wl,--no-undefined")
endif()

include_directories("/usr/local/cuda/include")
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${TRT_INCLUDE_DIR})
include_directories(${TRT_LLM_INCLUDE_DIR})
