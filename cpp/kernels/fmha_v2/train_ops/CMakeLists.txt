cmake_minimum_required(VERSION 3.8)

project(
  MHATrainOps
  VERSION ${PROJECT_VERSION}
  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS}  \
-O3 \
-g -lineinfo \
--expt-relaxed-constexpr \
--expt-extended-lambda \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_90a,code=sm_90a \
-Wno-deprecated-declarations \
--resource-usage \
-use_fast_math \
-Xptxas=-v \
")

# --keep --keep-dir ./tmp \
set(TORCH_ROOT "/usr/local/lib/python3.8/dist-packages")
# set(TORCH_ROOT '/opt/pytorch/pytorch')
find_package(Torch REQUIRED PATHS "${TORCH_ROOT}/torch/share/cmake/Torch")
find_library(TORCH_PYTHON_LIBRARY torch_python
             PATHS "${TORCH_INSTALL_PREFIX}/lib")
# see https://github.com/pytorch/pytorch/issues/38122
find_package(pybind11 PATHS "${TORCH_ROOT}/pybind11")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add_compile_definitions(FMHA_TRAIN_OPS_DEBUG_HOPPER_FPROP)
# add_compile_definitions(FMHA_TRAIN_OPS_DEBUG_HOPPER_DGRAD)

message(${TORCH_INCLUDE_DIRS})
message(${TORCH_LIBRARIES})

set(MHA_PATH "${CMAKE_SOURCE_DIR}/../")
set(MHA_TRAIN_PATH "${MHA_PATH}/train_ops/")

include_directories(${TORCH_INCLUDE_DIRS} ${MHA_PATH}/src ${MHA_PATH}/train_ops
                    /usr/local/cuda/targets/x86_64-linux/include/)

link_directories(/usr/local/cuda/targets/x86_64-linux/lib64/)

link_libraries(${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

set(KERNEL_SRC_FP16
    fmha_noloop_reduce.cu
    # kernels/fmha_fprop_v2_fp16_128_64_kernel.sm80.cu
    # kernels/fmha_fprop_v2_fp16_256_64_kernel.sm80.cu
    # kernels/fmha_fprop_v2_fp16_384_64_kernel.sm80.cu
    # kernels/fmha_fprop_v2_fp16_512_64_kernel.sm80.cu
    # kernels/fmha_dgrad_v2_fp16_128_64_kernel.sm80.cu
    # kernels/fmha_dgrad_v2_fp16_256_64_kernel.sm80.cu
    # kernels/fmha_dgrad_v2_fp16_384_64_kernel.sm80.cu
    # kernels/fmha_dgrad_v2_fp16_512_64_kernel.sm80.cu
    flash_attention_kernels/fmha_dgrad_v2_flash_attention_fp16_S_40_kernel.sm80.cu
    flash_attention_kernels/fmha_dgrad_v2_flash_attention_fp16_S_64_kernel.sm80.cu
    flash_attention_kernels/fmha_dgrad_v2_flash_attention_fp16_S_80_kernel.sm80.cu
    flash_attention_kernels/fmha_dgrad_v2_flash_attention_fp16_S_96_kernel.sm80.cu
    flash_attention_kernels/fmha_dgrad_v2_flash_attention_fp16_S_128_kernel.sm80.cu
    flash_attention_kernels/fmha_fprop_v2_flash_attention_fp16_S_40_kernel.sm80.cu
    flash_attention_kernels/fmha_fprop_v2_flash_attention_fp16_S_64_kernel.sm80.cu
    flash_attention_kernels/fmha_fprop_v2_flash_attention_fp16_S_80_kernel.sm80.cu
    flash_attention_kernels/fmha_fprop_v2_flash_attention_fp16_S_96_kernel.sm80.cu
    flash_attention_kernels/fmha_fprop_v2_flash_attention_fp16_S_128_kernel.sm80.cu
    flash_attention_kernels/fmha_dgrad_v2_flash_attention_bf16_S_40_kernel.sm80.cu
    flash_attention_kernels/fmha_dgrad_v2_flash_attention_bf16_S_64_kernel.sm80.cu
    flash_attention_kernels/fmha_dgrad_v2_flash_attention_bf16_S_80_kernel.sm80.cu
    flash_attention_kernels/fmha_dgrad_v2_flash_attention_bf16_S_96_kernel.sm80.cu
    flash_attention_kernels/fmha_dgrad_v2_flash_attention_bf16_S_128_kernel.sm80.cu
    flash_attention_kernels/fmha_fprop_v2_flash_attention_bf16_S_40_kernel.sm80.cu
    flash_attention_kernels/fmha_fprop_v2_flash_attention_bf16_S_64_kernel.sm80.cu
    flash_attention_kernels/fmha_fprop_v2_flash_attention_bf16_S_80_kernel.sm80.cu
    flash_attention_kernels/fmha_fprop_v2_flash_attention_bf16_S_96_kernel.sm80.cu
    flash_attention_kernels/fmha_fprop_v2_flash_attention_bf16_S_128_kernel.sm80.cu
)

add_library(fmha_fp16_obj OBJECT ${KERNEL_SRC_FP16})

set(KERNEL_SRC_FP8 hopper/fmha_dgrad_fp8_512_sm90.cu
                   hopper/fmha_fprop_fp8_512_sm90.cu)

add_library(fmha_fp8_obj OBJECT ${KERNEL_SRC_FP8})

pybind11_add_module(bert_mha_train MODULE bert_mha_train_api.cpp
                    $<TARGET_OBJECTS:fmha_fp16_obj>)

target_link_libraries(bert_mha_train PRIVATE ${TORCH_LIBRARIES}
                                             ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(bert_mha_train
                           PUBLIC TORCH_EXTENSION_NAME=bert_mha_train)

pybind11_add_module(fp8_mha_api MODULE fp8_mha_api.cpp
                    $<TARGET_OBJECTS:fmha_fp8_obj>)

target_link_libraries(fp8_mha_api PRIVATE ${TORCH_LIBRARIES}
                                          ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(fp8_mha_api PUBLIC TORCH_EXTENSION_NAME=fp8_mha_api)

add_executable(test test.cpp test_bmm.cpp ../src/convert.cu
                    $<TARGET_OBJECTS:fmha_fp8_obj>)

target_link_libraries(test PRIVATE ${TORCH_LIBRARIES} cublasLt python3.8)
