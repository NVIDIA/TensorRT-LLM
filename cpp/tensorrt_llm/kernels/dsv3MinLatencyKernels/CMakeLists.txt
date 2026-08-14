file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)

add_library(dsv3_min_latency_kernels OBJECT ${SRC_CPP} ${SRC_CU})
set_property(TARGET dsv3_min_latency_kernels PROPERTY POSITION_INDEPENDENT_CODE
                                                      ON)
set_property(TARGET dsv3_min_latency_kernels
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
