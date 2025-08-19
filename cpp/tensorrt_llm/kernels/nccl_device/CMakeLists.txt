# CMakeLists.txt for nccl_device
# This directory contains CUDA kernels and host launcher code

# Enable CUDA
enable_language(CUDA)

# Create CUDA library
add_library(tensorrt_llm_nccl_device
    config.cu
)

# Set properties for the CUDA library
set_target_properties(tensorrt_llm_nccl_device PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(tensorrt_llm_nccl_device PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Link libraries
target_link_libraries(tensorrt_llm_nccl_device
    tensorrt_llm_common
)

# Install target
install(TARGETS tensorrt_llm_nccl_device
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
