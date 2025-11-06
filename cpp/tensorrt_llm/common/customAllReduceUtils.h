/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include <cstddef>
#include <unordered_map>

using tensorrt_llm::kernels::AllReduceFusionOp;
using tensorrt_llm::kernels::AllReduceStrategyType;

namespace tensorrt_llm::utils::customAllReduceUtils
{

constexpr size_t NUM_POINTERS_PER_RANK = 7;

// WARNING: MUST BE KEPT IN SYNC with tensorrt_llm/plugin/plugin.py
inline size_t getMaxRequiredWorkspaceSize(int worldSize) noexcept
{
    if (common::getEnvForceDeterministicAllReduce())
    {
        return common::getEnvAllReduceWorkspaceSize();
    }
    if (worldSize <= 2)
    {
        return 16 * 1000 * 1000;
    }
    return 8 * 1000 * 1000;
}

// (SM major_version, TP_size) -> (NCCL_num_token_threshold, TWO_SHOT_numel_threshold)
inline std::unordered_map<int, std::unordered_map<int, std::pair<size_t, size_t>>> HeuristicThresholdLP{
    {90,
        {
            {2, {4096, 4096 * 4096}},
            {4, {4096, 1024 * 1024}},
            {8, {2048, 512 * 512}},
        }},
    {100,
        {
            {2, {4096, 4096 * 4096}},
            {4, {4096, 1024 * 2048}},
            {8, {4096, 1024 * 1024}},
        }},
};

inline AllReduceStrategyType SelectStrategyLP(size_t seq_len, size_t hidden_size, int world_size, AllReduceFusionOp op)
{
    // The heuristic is based on the following assumptions:
    //  __________________________________
    // |              \ TWO-SHOT zone |
    // | ONE-SHOT zone    \           | NCCL zone
    // |_______________________\______|___
    // sm_major is 90 or 100

    auto const sm_major = std::min(100, std::max(90, tensorrt_llm::common::getSMVersion()));

    auto const [nccl_num_token_threshold, two_shot_numel_threshold] = HeuristicThresholdLP[sm_major][world_size];
    auto const message_size = seq_len * hidden_size;
    if (message_size >= two_shot_numel_threshold)
    {
        return AllReduceStrategyType::TWOSHOT;
    }
    else
    {
        return AllReduceStrategyType::ONESHOT;
    }
    return AllReduceStrategyType::NCCL;
}

// use 1D vector to store the best strategy instead of a map for each sm version
// store int value instead of enum class
// The following layout, flattened to 1D vector
// (SM, TP, fusionOp, hidden_size, num_tokens) -> strategy
// table size estimate:
// SM: (<=90, 100)
// TP: (2, 4, 8)
// hidden_size: (128, 256, 512, 1024, 2048, 4096, 8192)
// fusionOp: (NONE, RESIDUAL_RMS_NORM, RESIDUAL_RMS_NORM_QUANT_FP8, RESIDUAL_RMS_NORM_QUANT_NVFP4)
// num_tokens: (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
// total size: 2 * 3 * 4 * 7 * 14 = 2352

constexpr int kTpSizeChoice = 3;
constexpr int kFusionOpChoice = 4;
constexpr int kHiddenSizeChoice = 7;
constexpr int kNumTokensChoice = 14;

inline std::unordered_map<AllReduceFusionOp, int> mapFusionOpToIndex = {
    {AllReduceFusionOp::NONE, 0},
    {AllReduceFusionOp::RESIDUAL_RMS_NORM, 1},
    {AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8, 2},
    {AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4, 3},
    // norm out quant fusion ops share the same index with norm quant fusion ops
    {AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8, 2},
    {AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4, 3},
};

// AllReduce lookup: [tp][fusion][hidden][tokens] = strategy
// TP:[2, 4, 8]
// Fusion:['NONE', 'RESIDUAL_RMS_NORM']
// Hidden:[128, 256, 512, 1024, 2048, 4096, 8192]
// Tokens:[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

using AllReduceBestStrategyTableType = std::vector<std::vector<std::vector<std::vector<int>>>>;

// Forward declarations for strategy tables
extern const std::unordered_map<int, AllReduceBestStrategyTableType> AllReduceBestStrategyTable;

inline AllReduceStrategyType selectStrategyLookUpTable(
    size_t num_tokens, size_t hidden_size, AllReduceFusionOp fusionOp, int tp_size)
{
    auto sm_version = tensorrt_llm::common::getSMVersion();
    auto tp_index = static_cast<size_t>(std::log2(tp_size) - 1);
    auto fusion_op_index = static_cast<size_t>(mapFusionOpToIndex.find(fusionOp)->second);
    auto num_token_index = static_cast<size_t>(std::log2(num_tokens));
    auto hidden_size_index = static_cast<size_t>(std::log2(hidden_size) - 7);

    // Map all pre-Blackwell sm versions to 90 for now
    if (sm_version < 100)
    {
        sm_version = 90;
    }

    // Map all post-Blackwell sm versions to 100 for now
    if (sm_version >= 100)
    {
        sm_version = 100;
    }

    // Check if the entry is out of bounds, otherwise return NCCL as fallback
    if (AllReduceBestStrategyTable.find(sm_version) == AllReduceBestStrategyTable.end()
        || tp_index >= AllReduceBestStrategyTable.at(sm_version).size()
        || fusion_op_index >= AllReduceBestStrategyTable.at(sm_version).at(tp_index).size()
        || hidden_size_index >= AllReduceBestStrategyTable.at(sm_version).at(tp_index).at(fusion_op_index).size()
        || num_token_index
            >= AllReduceBestStrategyTable.at(sm_version).at(tp_index).at(fusion_op_index).at(hidden_size_index).size())
    {
        return AllReduceStrategyType::NCCL;
    }

    return static_cast<AllReduceStrategyType>(
        AllReduceBestStrategyTable.at(sm_version)[tp_index][fusion_op_index][hidden_size_index][num_token_index]);
}

// Strategy table definitions
inline AllReduceBestStrategyTableType AllReduceBestStrategyTableSM90
    = {{    // TP=2
           {// Fusion=NONE
               {4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 0, 0},
               {4, 4, 5, 4, 4, 5, 5, 5, 4, 5, 4, 4, 4, 0, 0}, {4, 4, 4, 4, 5, 4, 5, 4, 4, 4, 4, 4, 0, 0, 0},
               {4, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 5, 0, 0, 0}, {4, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0},
               {4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0}},
           {// Fusion=RESIDUAL_RMS_NORM
               {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0},
               {4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 5, 5, 4, 5, 4, 4, 4, 0, 0, 0},
               {4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0},
               {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0}},
           {// Fusion=RESIDUAL_RMS_NORM_QUANT_FP8
               {4, 4, 4, 4, 4, 5, 5, 5, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 4, 0, 0},
               {4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 0, 0}, {4, 4, 4, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 0, 0},
               {4, 4, 4, 5, 5, 4, 4, 4, 5, 4, 4, 4, 0, 0, 0}, {4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0},
               {4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0}},
           {// Fusion=RESIDUAL_RMS_NORM_QUANT_NVFP4
               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
        {    // TP=4
            {// Fusion=NONE
                {4, 4, 4, 4, 5, 4, 4, 5, 4, 5, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 0, 0, 0, 0},
                {4, 4, 4, 4, 5, 4, 5, 5, 5, 4, 4, 5, 0, 0, 0}, {4, 4, 4, 4, 4, 5, 5, 4, 5, 4, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 5, 0, 0, 0, 0}, {4, 4, 4, 5, 5, 5, 5, 4, 5, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 5, 4, 5, 5, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 5, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM_QUANT_FP8
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 5, 5, 4, 5, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 5, 4, 5, 5, 4, 4, 5, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM_QUANT_NVFP4
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
        {    // TP=8
            {// Fusion=NONE
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 0, 5, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM_QUANT_FP8
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 0, 5, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM_QUANT_NVFP4
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}}};

inline AllReduceBestStrategyTableType AllReduceBestStrategyTableSM100
    = {{    // TP=2
           {// Fusion=NONE
               {4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
               {4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0}, {4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0},
               {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0},
               {4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0}},
           {// Fusion=RESIDUAL_RMS_NORM
               {4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
               {4, 4, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4, 4, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0},
               {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0},
               {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0}},
           {// Fusion=RESIDUAL_RMS_NORM_QUANT_FP8
               {4, 4, 4, 4, 5, 4, 5, 5, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 0, 0, 0},
               {4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0}, {4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 0, 0},
               {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0},
               {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0}},
           {// Fusion=RESIDUAL_RMS_NORM_QUANT_NVFP4
               {4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
               {4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0},
               {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0},
               {4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0}}},
        {    // TP=4
            {// Fusion=NONE
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM_QUANT_FP8
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM_QUANT_NVFP4
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0}}},
        {    // TP=8
            {// Fusion=NONE
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM_QUANT_FP8
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0, 0}},
            {// Fusion=RESIDUAL_RMS_NORM_QUANT_NVFP4
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 0, 0}}}};

inline const std::unordered_map<int, AllReduceBestStrategyTableType> AllReduceBestStrategyTable = {
    {90, AllReduceBestStrategyTableSM90},
    {100, AllReduceBestStrategyTableSM100},
};
} // namespace tensorrt_llm::utils::customAllReduceUtils
