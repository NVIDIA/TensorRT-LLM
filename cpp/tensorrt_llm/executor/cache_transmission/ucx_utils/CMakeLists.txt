# Build UCX data transceiver as shared library for runtime symbol loading, to
# make UCX on-demand runtime dependency.
if(ENABLE_UCX)
  find_package(ucx REQUIRED)
  find_package(ucxx REQUIRED)
  find_package(Torch REQUIRED)

  include_directories(${3RDPARTY_DIR}/cppzmq)

  # Find and link ZMQ
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(ZMQ REQUIRED libzmq)
  # Add the NIXL wrapper target

  add_library(${UCX_WRAPPER_TARGET} SHARED connection.cpp
                                           ucxCacheCommunicator.cpp)
  set_target_properties(
    ${UCX_WRAPPER_TARGET}
    PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES"
               CXX_EXTENSIONS "NO" POSITION_INDEPENDENT_CODE ON)

  set_property(TARGET ${UCX_WRAPPER_TARGET} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                     ON)
  set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
  target_compile_definitions(${UCX_WRAPPER_TARGET}
                             PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

  target_include_directories(${UCX_WRAPPER_TARGET}
                             PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_include_directories(${UCX_WRAPPER_TARGET} PRIVATE ${ZMQ_INCLUDE_DIRS})

  target_link_libraries(${UCX_WRAPPER_TARGET}
                        PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,ucxx::ucxx>)
  target_link_libraries(${UCX_WRAPPER_TARGET} PUBLIC ucxx::ucxx ucx::ucs)
  target_link_libraries(${UCX_WRAPPER_TARGET} PUBLIC ${CUDA_RT_LIB})
  target_link_libraries(${UCX_WRAPPER_TARGET} PUBLIC ${TORCH_LIBRARIES})
  target_link_libraries(${UCX_WRAPPER_TARGET} PRIVATE ${ZMQ_LIBRARIES})
  target_link_libraries(${UCX_WRAPPER_TARGET} PRIVATE pg_utils)
endif()
