/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tuple>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/fusedLayernormKernels/fp4_converter.cuh"
#include "tensorrt_llm/kernels/fusedLayernormKernels/layernorm_param.h"
#include "tensorrt_llm/kernels/fusedLayernormKernels/low_latency_layernorm.cuh"
#include "tensorrt_llm/kernels/fusedLayernormKernels/ws_layernorm.cuh"

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{
template <typename _Param, typename _InputType, typename _OutputType, typename _AccumulatorType, bool _RMS_NORM,
    int _M_BLOCK, int _N_BLOCK, int _STAGES = 3, bool _PERSISTENT_MODE = true, bool _LOW_LATENCY_MODE = false>
struct FP4AddBiasResidualPreLayerNormTraits
{

    using FusedOperator = FP4Converter<_InputType>;

    using Param = _Param;
    using InputType = _InputType;
    using OutputType = _InputType; // Use FP16 output to fused FP4 converter.
    using AccumulatorType = _AccumulatorType;
    static constexpr bool RMS_NORM = _RMS_NORM;
    static constexpr int M_BLOCK = _M_BLOCK;
    static constexpr int N_BLOCK = _N_BLOCK;
    static constexpr int STAGES = _STAGES;
    static constexpr int MATH_WARPGROUPS = _LOW_LATENCY_MODE ? (_N_BLOCK > 4096 ? 4 : 2) : (_PERSISTENT_MODE ? 2 : 1);
    static constexpr int PACKED_ELEMS_PER_COMPUTE
        = std::min(16 / sizeof(InputType), (size_t) N_BLOCK / (_LOW_LATENCY_MODE ? MATH_WARPGROUPS * 128 : 128));
    static constexpr bool GAMMA = true;
    static constexpr bool BETA = false;
    static constexpr bool RESIDUAL = true;
    static constexpr bool UNNORMED_OUTPUT = true;
    static constexpr SCALE_TYPE INPUT_SCALE = SCALE_TYPE::NONE;
    static constexpr SCALE_TYPE BIAS = SCALE_TYPE::NONE;
    static constexpr SCALE_TYPE OUTPUT_SCALE = SCALE_TYPE::NONE;
    static constexpr bool USE_BULK_STORE = false;
    static constexpr bool PERSISTENT_MODE = _PERSISTENT_MODE;
    static constexpr bool LOW_LATENCY_MODE = _LOW_LATENCY_MODE;
    static constexpr bool PREFETCH_TO_L2 = false;
    static constexpr bool HIGH_PRECISION_NORMED_OUTPUT = false;
};

template <typename T>
void invokeWSLayerNormImpl(
    WarpSpecializedParam<GeneralFP4AddBiasResidualPreLayerNormParam<T>> param, bool use_rms_norm, int ctas)
{

    auto _invoke = [&](auto traits)
    {
        using Traits = decltype(traits);
        using Operator = std::conditional_t<Traits::LOW_LATENCY_MODE, LowLatencyLayerNorm<Traits>,
            WarpSpecializedLayerNorm<Traits>>;
        constexpr auto N_THREADS = Traits::MATH_WARPGROUPS * 128 + (Traits::LOW_LATENCY_MODE ? 0 : 128);
        assert(param.n % Traits::PACKED_ELEMS_PER_COMPUTE == 0);
        static_assert(sizeof(typename Operator::Shared) <= 262144);
        static bool printed = false;
        if (!printed)
        {
            int waves = ((param.m + Traits::M_BLOCK - 1) / Traits::M_BLOCK + ctas - 1) / ctas;
            TLLM_LOG_DEBUG(
                "Selected TILE_M = %d, N = %d, STAGE = %d, PERSISTENT_MODE = %d, LOW_LATENCY_MODE = %d for param M = "
                "%d, N = %d, num_sms = %d. (waves = %d)\n",
                Traits::M_BLOCK, Traits::N_BLOCK, Traits::STAGES, Traits::PERSISTENT_MODE, Traits::LOW_LATENCY_MODE,
                param.m, param.n, ctas, waves);
            printed = true;
        }

        CUDA_CALL(cudaFuncSetAttribute(warpSpecializedInvoker<Operator, N_THREADS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(typename Operator::Shared)));

        if constexpr (!Traits::PERSISTENT_MODE)
        {
            ctas = (param.m + Traits::M_BLOCK - 1) / Traits::M_BLOCK;
        }

        cudaLaunchConfig_t config;
        cudaLaunchAttribute attrs[1];
        config.gridDim = ctas;
        config.blockDim = N_THREADS;
        config.dynamicSmemBytes = sizeof(typename Operator::Shared);
        config.stream = param.stream;
        config.attrs = attrs;
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        config.numAttrs = 1;

        CUDA_CALL(cudaLaunchKernelEx(&config, &warpSpecializedInvoker<Operator, N_THREADS>, param));
    };

    auto _invokeSelectRMSNorm = [&](auto m_block, auto n_block, auto stages, auto persistent, auto low_latency_mode)
    {
        constexpr auto M_BLOCK = decltype(m_block)::value;
        constexpr auto N_BLOCK = decltype(n_block)::value;
        constexpr auto STAGES = decltype(stages)::value;
        constexpr auto PERSISTENT = decltype(persistent)::value;
        constexpr auto LOW_LATENCY_MODE = decltype(low_latency_mode)::value;

        if (use_rms_norm)
        {
            _invoke(FP4AddBiasResidualPreLayerNormTraits<GeneralFP4AddBiasResidualPreLayerNormParam<T>, T, T, float,
                true, M_BLOCK, N_BLOCK, STAGES, PERSISTENT, LOW_LATENCY_MODE>{});
        }
        else
        {
            _invoke(FP4AddBiasResidualPreLayerNormTraits<GeneralFP4AddBiasResidualPreLayerNormParam<T>, T, T, float,
                false, M_BLOCK, N_BLOCK, STAGES, PERSISTENT, LOW_LATENCY_MODE>{});
        }
    };

    auto _invokeSelectPersistentMode = [&](auto m_block, auto n_block)
    {
        constexpr auto M_BLOCK = decltype(m_block)::value;
        constexpr auto N_BLOCK = decltype(n_block)::value;

        constexpr int STAGES = N_BLOCK >= 8192 ? 2 : 3;

        int waves = ((param.m + M_BLOCK - 1) / M_BLOCK + ctas - 1) / ctas;

        if (M_BLOCK == 1 && waves == 1)
        {
            _invokeSelectRMSNorm(ConstInt<1>{}, n_block, ConstInt<1>{}, ConstBool<false>{}, ConstBool<true>{});
        }
        else if (waves <= 1)
        {
            _invokeSelectRMSNorm(m_block, n_block, ConstInt<1>{}, ConstBool<false>{}, ConstBool<false>{});
        }
        else if (waves <= 2)
        {
            _invokeSelectRMSNorm(
                m_block, n_block, ConstInt<std::min(2, STAGES)>{}, ConstBool<true>{}, ConstBool<false>{});
        }
        else if (waves <= 3)
        {
            _invokeSelectRMSNorm(
                m_block, n_block, ConstInt<std::min(3, STAGES)>{}, ConstBool<true>{}, ConstBool<false>{});
        }
        else
        {
            _invokeSelectRMSNorm(m_block, n_block, ConstInt<STAGES>{}, ConstBool<true>{}, ConstBool<false>{});
        }
    };

    auto _invokeSelectTileSize = [&](auto n_block)
    {
        constexpr auto N_BLOCK = decltype(n_block)::value;
        static_assert(16384 % N_BLOCK == 0);
        constexpr int MAX_M_BLOCK = (8192 / N_BLOCK < 4 ? 16384 / N_BLOCK : 8192 / N_BLOCK);

        auto desired_m_block = (param.m + (ctas * 2) - 1) / (ctas * 2);

        assert(desired_m_block);

        if (desired_m_block >= MAX_M_BLOCK)
        {
            _invokeSelectPersistentMode(ConstInt<MAX_M_BLOCK>{}, n_block);
            return;
        }

        int m_block = 1 << (31 - __builtin_clz(desired_m_block));

        if (m_block == 1)
        {
            if constexpr (1 <= MAX_M_BLOCK)
            {
                _invokeSelectPersistentMode(ConstInt<1>{}, n_block);
            }
            else
            {
                assert(false);
            }
        }
        else if (m_block == 2)
        {
            if constexpr (2 <= MAX_M_BLOCK)
            {
                _invokeSelectPersistentMode(ConstInt<2>{}, n_block);
            }
            else
            {
                assert(false);
            }
        }
        else if (m_block == 4)
        {
            if constexpr (4 <= MAX_M_BLOCK)
            {
                _invokeSelectPersistentMode(ConstInt<4>{}, n_block);
            }
            else
            {
                assert(false);
            }
        }
        else if (m_block == 8)
        {
            if constexpr (8 <= MAX_M_BLOCK)
            {
                _invokeSelectPersistentMode(ConstInt<8>{}, n_block);
            }
            else
            {
                assert(false);
            }
        }
        else if (m_block == 16)
        {
            if constexpr (16 <= MAX_M_BLOCK)
            {
                _invokeSelectPersistentMode(ConstInt<16>{}, n_block);
            }
            else
            {
                assert(false);
            }
        }
        else if (m_block == 32)
        {
            if constexpr (32 <= MAX_M_BLOCK)
            {
                _invokeSelectPersistentMode(ConstInt<32>{}, n_block);
            }
            else
            {
                assert(false);
            }
        }
        else if (m_block == 64)
        {
            if constexpr (64 <= MAX_M_BLOCK)
            {
                _invokeSelectPersistentMode(ConstInt<64>{}, n_block);
            }
            else
            {
                assert(false);
            }
        }
        else if (m_block == 128)
        {
            if constexpr (128 <= MAX_M_BLOCK)
            {
                _invokeSelectPersistentMode(ConstInt<128>{}, n_block);
            }
            else
            {
                assert(false);
            }
        }
        else
        {
            assert(false);
        }
    };

    auto _invokeSelectNBlock = [&]()
    {
        // if (param.n <= 512) {
        //     _invokeSelectTileSize(ConstInt<512>{});
        // } else if (param.n <= 1024) {
        //     _invokeSelectTileSize(ConstInt<1024>{});
        // } else
        if (param.n <= 2048)
        {
            _invokeSelectTileSize(ConstInt<2048>{});
        }
        else if (param.n <= 4096)
        {
            _invokeSelectTileSize(ConstInt<4096>{});
        }
        else if (param.n <= 8192)
        {
            _invokeSelectTileSize(ConstInt<8192>{});
        }
        else if (param.n <= 16384)
        {
            _invokeSelectTileSize(ConstInt<16384>{});
        }
        else
        {
            assert(false);
        }
    };

    _invokeSelectNBlock();
}

template <>
void invokeWSLayerNorm<GeneralFP4AddBiasResidualPreLayerNormParam<half>>(
    WarpSpecializedParam<GeneralFP4AddBiasResidualPreLayerNormParam<half>> param, bool use_rms_norm, int ctas)
{
    invokeWSLayerNormImpl(param, use_rms_norm, ctas);
}

template <>
void invokeWSLayerNorm<GeneralFP4AddBiasResidualPreLayerNormParam<__nv_bfloat16>>(
    WarpSpecializedParam<GeneralFP4AddBiasResidualPreLayerNormParam<__nv_bfloat16>> param, bool use_rms_norm, int ctas)
{
    invokeWSLayerNormImpl(param, use_rms_norm, ctas);
}

} // namespace tensorrt_llm::kernels
