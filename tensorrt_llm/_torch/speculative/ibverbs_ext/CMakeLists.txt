cmake_minimum_required(VERSION 3.27)

set(IBVERBS_EXT_TARGET ibverbs_ext)
add_custom_target(${IBVERBS_EXT_TARGET})

find_library(IBVERBS_LIB ibverbs REQUIRED)

add_library(ibv_wrapper SHARED src/ibv_wrapper.c)
target_link_libraries(ibv_wrapper PRIVATE ${IBVERBS_LIB})
set_target_properties(ibv_wrapper PROPERTIES OUTPUT_NAME ibv_wrapper)

add_dependencies(${IBVERBS_EXT_TARGET} ibv_wrapper)
