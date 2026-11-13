/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/thop/attentionOp.h"

#include <gtest/gtest.h>

namespace
{

class FmhaSparseParamsTest : public ::testing::TestWithParam<bool>
{
protected:
    void SetUp() override
    {
        mPrimaryPool = torch::zeros({16}, torch::kInt32);
        mAuxPool = torch::ones({16}, torch::kInt32);
        auto poolPointers = torch::zeros({2, 2}, torch::kInt64);
        poolPointers.index_put_({1, 0}, reinterpret_cast<int64_t>(mPrimaryPool.data_ptr()));
        mParams.meta.host_kv_cache_pool_pointers = GetParam() ? poolPointers.unsqueeze(-1) : poolPointers;
        mParams.meta.host_kv_cache_pool_mapping = torch::tensor({{0, 0}, {1, 0}}, torch::kInt32);
        mParams.meta.kv_cache_block_offsets = torch::zeros({2, 1, 2, 1}, torch::kInt32);
        mParams.local_layer_idx = 1;
        mParams.fwd.sparse_runtime_params.sparse_attn_indices = torch::tensor({{0, 1}}, torch::kInt32);
        mParams.meta.num_sparse_topk = 2;
    }

    tensorrt_llm::torch_ext::FmhaParams mParams;
    torch::Tensor mPrimaryPool;
    torch::Tensor mAuxPool;
};

TEST_P(FmhaSparseParamsTest, DsaUsesAuxiliaryPool)
{
    mParams.fwd.sparse_runtime_params.aux_kv_cache_pool_ptr = reinterpret_cast<int64_t>(mAuxPool.data_ptr());

    auto const sparse = mParams.getSparseParams();

    EXPECT_EQ(sparse.sparse_kv_cache_pool, mAuxPool.data_ptr());
    EXPECT_EQ(sparse.sliding_window_kv_cache_pool, nullptr);
    EXPECT_EQ(sparse.sparse_attn_indices, mParams.fwd.sparse_runtime_params.sparse_attn_indices->data_ptr<int32_t>());
    EXPECT_EQ(sparse.num_sparse_topk, 2);
}

TEST_P(FmhaSparseParamsTest, ResidentDsaUsesMappedPool)
{
    auto const sparse = mParams.getSparseParams();

    EXPECT_EQ(sparse.sparse_kv_cache_pool, mPrimaryPool.data_ptr());
    EXPECT_EQ(sparse.sliding_window_kv_cache_pool, nullptr);
}

TEST_P(FmhaSparseParamsTest, DynamicSparseKeepsSlidingWindowAndAuxiliaryPoolsSeparate)
{
    mParams.fwd.sparse_runtime_params.sparse_attn_kv_lens = torch::tensor({2}, torch::kInt32);
    mParams.fwd.sparse_runtime_params.aux_kv_cache_pool_ptr = reinterpret_cast<int64_t>(mAuxPool.data_ptr());

    auto const sparse = mParams.getSparseParams();

    EXPECT_EQ(sparse.sparse_kv_cache_pool, mAuxPool.data_ptr());
    EXPECT_EQ(sparse.sliding_window_kv_cache_pool, mPrimaryPool.data_ptr());
}

TEST_P(FmhaSparseParamsTest, DynamicSparseWithoutAuxiliaryPoolUsesOnlySlidingWindowPool)
{
    mParams.fwd.sparse_runtime_params.sparse_attn_kv_lens = torch::tensor({2}, torch::kInt32);

    auto const sparse = mParams.getSparseParams();

    EXPECT_EQ(sparse.sparse_kv_cache_pool, nullptr);
    EXPECT_EQ(sparse.sliding_window_kv_cache_pool, mPrimaryPool.data_ptr());
}

INSTANTIATE_TEST_SUITE_P(PoolPointerLayouts, FmhaSparseParamsTest, ::testing::Bool());

} // namespace
