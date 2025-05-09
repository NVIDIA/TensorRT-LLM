/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <nvrtc.h>
#include <string>
#include <vector>

#include "xqa_sources.h"

using namespace ::tensorrt_llm::kernels;

void checkNvrtc_(nvrtcResult result, char const* const func, char const* const file, int const line)
{
    if (result != NVRTC_SUCCESS)
    {
        fprintf(stderr, "%s:%d: %s\n", file, line, nvrtcGetErrorString(result));
    }
}

#define checkNvrtc(val) checkNvrtc_((val), #val, __FILE__, __LINE__)

inline void cuErrCheck_(CUresult stat, char const* file, int line)
{
    if (stat != CUDA_SUCCESS)
    {
        char const* msg = nullptr;
        cuGetErrorName(stat, &msg);
        fprintf(stderr, "CUDA Error: %s %s %d\n", msg, file, line);
    }
}

#define cuErrCheck(stat)                                                                                               \
    {                                                                                                                  \
        cuErrCheck_((stat), __FILE__, __LINE__);                                                                       \
    }

int main()
{
    cuErrCheck(cuInit(0));
    CUdevice device;
    CUcontext context;
    cuErrCheck(cuDeviceGet(&device, 0));
    cuErrCheck(cuCtxCreate(&context, 0, device));

    int major = 0, minor = 0;
    cuErrCheck(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    cuErrCheck(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    printf("Compute Capability: %d.%d\n", major, minor);

    std::vector<char const*> headers_content = {
        cuda_hint_cuh_content,
        defines_h_content,
        ldgsts_cuh_content,
        mha_h_content,
        mha_utils_cuh_content,
        mma_cuh_content,
        platform_h_content,
        ref_checker_cuh_content,
        utils_cuh_content,
        utils_h_content,
        mha_stdheaders_h_content,
        gmma_cuh_content,
        gmma_impl_cuh_content,
        barriers_h_content,
        tma_h_content,
        cuda_bf16_h_content,
        cuda_bf16_hpp_content,
        cuda_fp16_h_content,
        cuda_fp16_hpp_content,
        cuda_fp8_h_content,
        cuda_fp8_hpp_content,
        vector_types_h_content,
        vector_functions_h_content,
    };
    std::vector<char const*> headers_name = {"cuda_hint.cuh", "defines.h", "ldgsts.cuh", "mha.h", "mhaUtils.cuh",
        "mma.cuh", "platform.h", "ref_checker.cuh", "utils.cuh", "utils.h", "mha_stdheaders.cuh", "gmma.cuh",
        "gmma_impl.cuh", "barriers.cuh", "tma.h", "cuda_bf16.h", "cuda_bf16.hpp", "cuda_fp16.h", "cuda_fp16.hpp",
        "cuda_fp8.h", "cuda_fp8.hpp", "vector_types.h", "vector_functions.h"};
    for (bool use_paged_kv_cache : {false, true})
    {
        for (int beam_width : {1, 2, 4})
        {
            std::string arch_flag = "-arch=sm_" + std::to_string(major) + std::to_string(minor);
            std::vector<std::string> options = {
                "-dw",
                "-std=c++17",
                "--use_fast_math",
                arch_flag,
                "-default-device",
                "-DGENERATE_CUBIN=1",
                "-DNDEBUG",
                "-DDTYPE=__nv_bfloat16",
                "-DINPUT_FP16=0",
                "-DHEAD_DIM=128",
                "-DUSE_PAGED_KV_CACHE=" + std::to_string(use_paged_kv_cache ? 1 : 0),
                "-DBEAM_WIDTH=" + std::to_string(beam_width),
                "-DCACHE_ELEM_ENUM=0",
                "-DTOKENS_PER_PAGE=" + std::to_string(use_paged_kv_cache ? 64 : 0),
                "-DHEAD_GRP_SIZE=8",
                "-DM_TILESIZE=8",
                "-DKERNEL_FUNC_NAME=xqa_kernel_dt_fp16_d_128_beam_1_kvt_int8_nqpkv_8_m_8_sm_80",
                "-DUSE_CUSTOM_BARRIER=1",
            };
            std::vector<char const*> options_cstr;
            for (auto const& option : options)
            {
                options_cstr.push_back(option.c_str());
            }

            nvrtcProgram program;
            std::string log;

            auto before_nvrtc = std::chrono::high_resolution_clock::now();

            checkNvrtc(nvrtcCreateProgram(&program, mha_cu_content, "program", headers_content.size(),
                headers_content.data(), headers_name.data()));
            auto status = nvrtcCompileProgram(program, options_cstr.size(), options_cstr.data());
            if (status != NVRTC_SUCCESS)
            {
                size_t log_size;
                checkNvrtc(nvrtcGetProgramLogSize(program, &log_size));
                log.resize(log_size);
                checkNvrtc(nvrtcGetProgramLog(program, const_cast<char*>(log.data())));
                fprintf(stderr, "%s\n", log.c_str());
                exit(1);
            }

            auto after_nvrtc = std::chrono::high_resolution_clock::now();
            std::cout << "NVRTC took "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(after_nvrtc - before_nvrtc).count()
                      << " ms" << std::endl;

            size_t cubinSize;
            checkNvrtc(nvrtcGetCUBINSize(program, &cubinSize));
            std::cout << "cubinSize=" << cubinSize << std::endl;
            std::string cubinContent(cubinSize, ' ');
            checkNvrtc(nvrtcGetCUBIN(program, const_cast<char*>(cubinContent.c_str())));

            std::fstream file;
            file.open("nvrtc_result.o", std::ios::out | std::ios::binary);
            file.write(cubinContent.c_str(), cubinContent.size());

            checkNvrtc(nvrtcDestroyProgram(&program));

            ///
            CUmodule module;
            cuErrCheck(cuModuleLoad(&module, "nvrtc_result.o"));
            CUfunction function;
            cuErrCheck(cuModuleGetFunction(&function, module, "kernel_mha"));
            assert(function != nullptr);
            CUdeviceptr shmem_dev_ptr;
            cuErrCheck(cuModuleGetGlobal(&shmem_dev_ptr, nullptr, module, "smemSize"));
            unsigned int shmem_bytes = 0;
            cuErrCheck(cuMemcpyDtoH(&shmem_bytes, shmem_dev_ptr, sizeof(unsigned int)));
            printf("shmem_bytes=%u\n", shmem_bytes);
        }
    }
}
