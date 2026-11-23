/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Exhaustive device test for the 8-bit KV cache word converters in utils.cuh
// (convertKCacheWordToF16 / convertVCacheWordToF16).
//
// For every (InputElem, CacheElem) pair that mha.cu instantiates with an 8-bit cache, every one of the 256 byte
// values is placed in every byte lane of a 32-bit cache word (the other lanes hold the other values), converted by
// both helpers on the device, and compared bit for bit with the scalar reference InputElem(CacheElem) in the
// element order of the generic code path:
//   K: output element j of the word        = byte j
//   V: output element (i, j) (= 2 * i + j) = byte (2 * j + i)   (word i = {byte i, byte i + 2})
// For bf16 + int8 this also checks the I2F-free specialisation against the generic conversion.

#include "../utils.cuh"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

namespace
{

constexpr uint32_t kNbWords = 256 * 4; // every byte value in every lane

// Word w: lane l holds byte (w + 64 * l) & 0xFF, so each lane sees all 256 values over w in [0, 256),
// and words 256..1023 repeat that with the lane assignment rotated.
__host__ __device__ inline uint32_t testWord(uint32_t w)
{
    uint32_t const rot = w / 256;
    uint32_t word = 0;
    for (uint32_t l = 0; l < 4; l++)
    {
        uint32_t const b = ((w % 256) + 64 * ((l + rot) % 4)) & 0xFFu;
        word |= b << (8 * l);
    }
    return word;
}

template <typename InputElem, typename CacheElem>
__global__ void convertKernel(uint32_t* kOut, uint32_t* vOut, uint16_t* refOut)
{
    uint32_t const w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= kNbWords)
    {
        return;
    }
    uint32_t const word = testWord(w);
    Vec<uint32_t, 2> const k = convertKCacheWordToF16<InputElem, CacheElem>(word);
    Vec<uint32_t, 2> const v = convertVCacheWordToF16<InputElem, CacheElem>(word);
    kOut[2 * w] = k[0];
    kOut[2 * w + 1] = k[1];
    vOut[2 * w] = v[0];
    vOut[2 * w + 1] = v[1];
    // scalar reference for each byte lane: InputElem(CacheElem)
    for (uint32_t l = 0; l < 4; l++)
    {
        uint8_t const b = static_cast<uint8_t>(word >> (8 * l));
        InputElem const r = InputElem(reinterpret_cast<CacheElem const&>(b));
        refOut[4 * w + l] = reinterpret_cast<uint16_t const&>(r);
    }
}

template <typename InputElem>
bool isNan16(uint16_t x)
{
    if constexpr (mha::is_same_v<InputElem, half>)
    {
        return (x & 0x7C00u) == 0x7C00u && (x & 0x03FFu) != 0;
    }
    else
    {
        return (x & 0x7F80u) == 0x7F80u && (x & 0x007Fu) != 0;
    }
}

// Host reference for int8 caches: every int8 value is exact in fp16 and bf16, so the expected bits are known exactly
// (bf16 = upper half of the fp32 bits; fp16 via the host conversion, exact for |x| <= 128).
template <typename InputElem>
uint16_t hostFromInt8(int8_t x)
{
    if constexpr (mha::is_same_v<InputElem, half>)
    {
        half const h = __float2half_rn(static_cast<float>(x));
        return reinterpret_cast<uint16_t const&>(h);
    }
    else
    {
        float const f = static_cast<float>(x);
        return static_cast<uint16_t>(reinterpret_cast<uint32_t const&>(f) >> 16);
    }
}

template <typename InputElem>
bool same16(uint16_t a, uint16_t b)
{
    return a == b || (isNan16<InputElem>(a) && isNan16<InputElem>(b)); // e4m3 NaN encodings may differ in payload
}

#define CUDA_CHECK(x)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t const err_ = (x);                                                                                  \
        ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_);                                                      \
    } while (0)

template <typename InputElem, typename CacheElem>
void runExhaustive(char const* name)
{
    uint32_t *dK = nullptr, *dV = nullptr;
    uint16_t* dRef = nullptr;
    CUDA_CHECK(cudaMalloc(&dK, sizeof(uint32_t) * 2 * kNbWords));
    CUDA_CHECK(cudaMalloc(&dV, sizeof(uint32_t) * 2 * kNbWords));
    CUDA_CHECK(cudaMalloc(&dRef, sizeof(uint16_t) * 4 * kNbWords));
    convertKernel<InputElem, CacheElem><<<kNbWords / 128, 128>>>(dK, dV, dRef);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> k(2 * kNbWords), v(2 * kNbWords);
    std::vector<uint16_t> ref(4 * kNbWords);
    CUDA_CHECK(cudaMemcpy(k.data(), dK, sizeof(uint32_t) * k.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v.data(), dV, sizeof(uint32_t) * v.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, sizeof(uint16_t) * ref.size(), cudaMemcpyDeviceToHost));
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dRef);

    uint32_t nbMismatchK = 0, nbMismatchV = 0, nbMismatchRef = 0;
    bool seen[4][256] = {};
    for (uint32_t w = 0; w < kNbWords; w++)
    {
        uint32_t const word = testWord(w);
        uint16_t const* r = &ref[4 * w];
        uint16_t const* kh = reinterpret_cast<uint16_t const*>(&k[2 * w]); // element j = byte j
        uint16_t const* vh = reinterpret_cast<uint16_t const*>(&v[2 * w]); // element 2i+j = byte 2j+i
        for (uint32_t l = 0; l < 4; l++)
        {
            seen[l][(word >> (8 * l)) & 0xFFu] = true;
        }
        if constexpr (mha::is_same_v<CacheElem, int8_t>)
        {
            for (uint32_t l = 0; l < 4; l++)
            {
                uint16_t const want = hostFromInt8<InputElem>(static_cast<int8_t>(word >> (8 * l)));
                if (r[l] != want && nbMismatchRef++ < 8)
                {
                    ADD_FAILURE() << name << " device scalar reference for byte 0x" << std::hex
                                  << ((word >> (8 * l)) & 0xFFu) << ": 0x" << r[l] << " want 0x" << want;
                }
            }
        }
        for (uint32_t j = 0; j < 4; j++)
        {
            if (!same16<InputElem>(kh[j], r[j]) && nbMismatchK++ < 8)
            {
                ADD_FAILURE() << name << " K word 0x" << std::hex << word << " elem " << j << ": got 0x" << kh[j]
                              << " want 0x" << r[j];
            }
        }
        for (uint32_t i = 0; i < 2; i++)
        {
            for (uint32_t j = 0; j < 2; j++)
            {
                uint16_t const got = vh[2 * i + j];
                uint16_t const want = r[2 * j + i];
                if (!same16<InputElem>(got, want) && nbMismatchV++ < 8)
                {
                    ADD_FAILURE() << name << " V word 0x" << std::hex << word << " elem (" << i << "," << j
                                  << "): got 0x" << got << " want 0x" << want;
                }
            }
        }
    }
    for (uint32_t l = 0; l < 4; l++)
    {
        for (uint32_t b = 0; b < 256; b++)
        {
            ASSERT_TRUE(seen[l][b]) << "test vector does not cover byte " << b << " in lane " << l;
        }
    }
    EXPECT_EQ(nbMismatchK, 0u) << name;
    EXPECT_EQ(nbMismatchV, 0u) << name;
    EXPECT_EQ(nbMismatchRef, 0u) << name;
    printf("%s: %u words x 4 lanes, K mismatches %u, V mismatches %u, scalar-vs-host mismatches %u\n", name, kNbWords,
        nbMismatchK, nbMismatchV, nbMismatchRef);
}

} // namespace

TEST(XqaConvertKVCacheWord, Bf16FromInt8)
{
    runExhaustive<__nv_bfloat16, int8_t>("bf16<-int8");
}

TEST(XqaConvertKVCacheWord, Fp16FromInt8)
{
    runExhaustive<half, int8_t>("fp16<-int8");
}

TEST(XqaConvertKVCacheWord, Bf16FromFp8E4m3)
{
    runExhaustive<__nv_bfloat16, __nv_fp8_e4m3>("bf16<-e4m3");
}

TEST(XqaConvertKVCacheWord, Fp16FromFp8E4m3)
{
    runExhaustive<half, __nv_fp8_e4m3>("fp16<-e4m3");
}
