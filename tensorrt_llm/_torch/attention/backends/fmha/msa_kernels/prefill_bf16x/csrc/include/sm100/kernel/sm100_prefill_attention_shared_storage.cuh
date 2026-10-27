// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once

#include "sm100/common/prefill_attention_config.cuh"
#include "sm100/device/prefill_attention.hpp"

#include <cstdint>

#include "cute/arch/copy_sm100.hpp"
#include "cute/tensor.hpp"

namespace fmha_sm100::prefill_bf16x::detail {

struct WorkTileInfo {
  int head_kv;
  int row_start;
  int q_count;
  int batch;
  int logical_page;
  int q_batch_offset;
  int q_length;
  int kv_length;
  int valid_cols;
  int physical_page;
  // Host factors times the K / V global scales, read once per CTA.
  float softmax_scale_log2;
  float output_scale;
};

static_assert(sizeof(WorkTileInfo) == 12 * sizeof(int));

template <class QSmemLayout, class KSmemLayout, class VSmemLayout>
struct SharedStorage {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<QSmemLayout>> q;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<KSmemLayout>> k;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<VSmemLayout>> v;

  alignas(128) uint8_t k_raw[kPageSize][kHeadDim / 2];
  alignas(128) uint8_t v_raw[kPageSize][kHeadDim / 2];
  alignas(128) uint8_t k_scale[kPageSize][kHeadDim / 16];
  alignas(128) uint8_t v_scale[kPageSize][kHeadDim / 16];

  // Q: load warps produce full stages, the MMA warp releases empty stages.
  alignas(16) uint64_t q_full[kQStages];
  alignas(16) uint64_t q_empty[kQStages];

  // KV: TMA load completes raw tiles, then one softmax warpgroup dequantizes
  // K or V and publishes the corresponding FP8 (or BF16) tile to the MMA warp.
  alignas(16) uint64_t k_tma_full;
  alignas(16) uint64_t v_tma_full;
  alignas(16) uint64_t k_dequant_full;
  alignas(16) uint64_t v_dequant_full;

  // QK -> softmax -> PV -> epilogue pipelines, double buffered by score stage.
  alignas(16) uint64_t score_full[kScoreStages];
  alignas(16) uint64_t score_empty[kScoreStages];
  alignas(16) uint64_t probability_early_full[kScoreStages];
  alignas(16) uint64_t probability_full[kScoreStages];
  alignas(16) uint64_t probability_empty[kScoreStages];
  alignas(16) uint64_t probability_last_empty[kScoreStages];
  alignas(16) uint64_t output_full[kScoreStages];
  alignas(16) uint64_t output_empty[kScoreStages];
  alignas(16) uint64_t stats_full[kScoreStages];
  alignas(16) uint64_t stats_empty[kScoreStages];
  alignas(16) uint32_t tmem_base;

  WorkTileInfo work_tile;
  int qsplit_indices[kQMetadataStages][kQueriesPerGroup];
  float row_sum[kScoreStages][128];
  float row_max[kScoreStages][128];
};

template <class TmaQ, class QGmemShape>
struct KernelParams {
  PrefillArguments arguments;
  TmaQ tma_q;
  QGmemShape q_gmem_shape;
  cute::TmaDescriptor tma_packed_k;
  cute::TmaDescriptor tma_packed_v;
  cute::TmaDescriptor tma_k_scale;
  cute::TmaDescriptor tma_v_scale;
};

}  // namespace fmha_sm100::prefill_bf16x::detail
