/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
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

// Signal KernelConfigBase.h to skip its own enum/toString/helper definitions;
// trtllmGenExportCompat.h provides them as TRT-LLM type aliases instead.
// CMakeLists.txt already passes -DTLLM_FMHA_TRTLLM_COMPAT; guard to avoid redefinition.
#ifndef TLLM_FMHA_TRTLLM_COMPAT
#define TLLM_FMHA_TRTLLM_COMPAT
#endif

// Compatibility header for building trtllm-gen FMHA export headers inside TensorRT-LLM.
// Provides type aliases, toString specializations, check macros, and utility functions
// that bridge the gap between trtllm-gen's native types and TRT-LLM's kernel types.
//
// This header is included by export headers (e.g., KernelTraits.h) when
// TLLM_GEN_EXPORT_INTERFACE is defined, replacing trtllm-gen-internal headers
// like KernelConfigBase.h, GenCtx.h, TmemTile.h, etc.

#include "../fmhaRunnerParams.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "trtllm/gen/CommonUtils.h"
#include "trtllm/gen/CudaArchDecl.h"
#include "trtllm/gen/DtypeDecl.h"
#include <nlohmann/json.hpp>
#include <cassert>
#include <sstream>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
namespace tk = tensorrt_llm::kernels;
namespace tg = trtllm::gen;
} // namespace

// Type aliases: map fmha:: names to TRT-LLM kernel enum types.
using AttentionMaskType = tk::TrtllmGenAttentionMaskType;
using FmhaKernelType = tk::FmhaKernelType;
using MultiCtasKvMode = tk::MultiCtasKvMode;
using QkvLayout = tk::QkvLayout;
using SparseType = tk::SparseType;
using TileScheduler = tk::TileScheduler;

using trtllm::gen::ceilDiv;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Check macros: streaming-style (matching trtllm-gen calling convention).
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {
template <typename... Args>
inline std::string checkErrorStr(Args const&... args) {
  std::ostringstream oss;
  ((oss << args), ...);
  return oss.str();
}
} // namespace detail

#ifndef TLLM_CHECK_ERROR
#define TLLM_CHECK_ERROR(cond, ...)                                                                \
  do {                                                                                             \
    if (!static_cast<bool>(cond)) {                                                                \
      tensorrt_llm::common::throwRuntimeError(                                                     \
          __FILE__, __LINE__, ::fmha::detail::checkErrorStr(__VA_ARGS__).c_str());                  \
    }                                                                                              \
  } while (0)
#endif

#ifndef TLLM_CHECK_INFO
#define TLLM_CHECK_INFO(cond, ...)                                                                 \
  do {                                                                                             \
    if (!static_cast<bool>(cond)) {                                                                \
      TLLM_LOG_INFO("[FMHA] %s", ::fmha::detail::checkErrorStr(__VA_ARGS__).c_str());              \
    }                                                                                              \
  } while (0)
#endif

// Printf-style check macro (used by KernelTraits.h).
#ifndef TLLM_CHECK_ERROR_FMT
#define TLLM_CHECK_ERROR_FMT(cond, fmt, ...)                                                       \
  do {                                                                                             \
    if (!static_cast<bool>(cond)) {                                                                \
      tensorrt_llm::common::throwRuntimeError(                                                     \
          __FILE__, __LINE__,                                                                      \
          tensorrt_llm::common::fmtstr(fmt, __VA_ARGS__).c_str());                                 \
    }                                                                                              \
  } while (0)
#endif

// Helper functions (isPagedKv, isContextKernel, etc.) are NOT defined here
// because fmhaRunnerParams.h already provides them in namespace tensorrt_llm::kernels.
// The using-aliases above make fmha:: types the same as tk:: types, so no ambiguity.

////////////////////////////////////////////////////////////////////////////////////////////////////
// MmaOrder enum (not in fmhaRunnerParams.h, defined natively here).
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class MmaOrder {
  Pv0_Qk0_Pv1_Qk1 = 0,
  Qk0_Pv0_Qk1_Pv1,
  Qk0_Qk1_Pv0_Pv1,
};

inline std::string mmaOrderToString(MmaOrder mmaOrder) {
  switch (mmaOrder) {
  case MmaOrder::Pv0_Qk0_Pv1_Qk1:
    return "Pv0_Qk0_Pv1_Qk1";
  case MmaOrder::Qk0_Pv0_Qk1_Pv1:
    return "Qk0_Pv0_Qk1_Pv1";
  case MmaOrder::Qk0_Qk1_Pv0_Pv1:
    return "Qk0_Qk1_Pv0_Pv1";
  default:
    assert(false);
    return "Invalid MmaOrder";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Serialization helpers (toString specializations).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> inline std::string toString(T e) {
  return std::to_string(e);
}

template <> inline std::string toString(bool flag) {
  return (flag ? "true" : "false");
}

template <> inline std::string toString(AttentionMaskType e) {
  switch (e) {
  case AttentionMaskType::Dense:
    return "Dense";
  case AttentionMaskType::Causal:
    return "Causal";
  case AttentionMaskType::SlidingOrChunkedCausal:
    return "SlidingOrChunkedCausal";
  case AttentionMaskType::Custom:
    return "Custom";
  default:
    return "";
  }
}

template <> inline std::string toString(tg::CudaArch arch) {
  return tg::cudaArchToString(arch);
}

template <> inline std::string toString(FmhaKernelType e) {
  switch (e) {
  case FmhaKernelType::Context:
    return "Context";
  case FmhaKernelType::Generation:
    return "Gen";
  case FmhaKernelType::SwapsMmaAbForGeneration:
    return "SwapsAbForGen";
  case FmhaKernelType::KeepsMmaAbForGeneration:
    return "KeepsAbForGen";
  default:
    return "";
  }
}

template <> inline std::string toString(MultiCtasKvMode mode) {
  switch (mode) {
  case MultiCtasKvMode::Disabled:
    return "Disabled";
  case MultiCtasKvMode::GmemReduction:
    return "GmemReduction";
  case MultiCtasKvMode::GmemReductionWithSeparateKernel:
    return "GmemReductionWithSeparateKernel";
  case MultiCtasKvMode::CgaSmemReduction:
    return "CgaSmemReduction";
  default:
    return "";
  }
}

inline std::string multiCtasKvModeToString(MultiCtasKvMode mode) {
  return toString(mode);
}

template <> inline std::string toString(TileScheduler scheduler) {
  switch (scheduler) {
  case TileScheduler::Static:
    return "Static";
  case TileScheduler::Persistent:
    return "Persistent";
  default:
    return "";
  }
}

template <> inline std::string toString(QkvLayout qkvLayout) {
  switch (qkvLayout) {
  case QkvLayout::SeparateQkv:
    return "SeparateQkv";
  case QkvLayout::PackedQkv:
    return "PackedQkv";
  case QkvLayout::PagedKv:
    return "PagedKv";
  case QkvLayout::ContiguousKv:
    return "ContiguousKv";
  default:
    return "";
  }
}

template <> inline std::string toString(tg::Dtype e) {
  return tg::dtypeToString(e);
}

template <> inline std::string toString(MmaOrder e) {
  return mmaOrderToString(e);
}

template <> inline std::string toString(SparseType e) {
  switch (e) {
  case SparseType::None:
    return "None";
  case SparseType::StaticTokenSparse:
    return "StaticTokenSparse";
  case SparseType::DynamicTokenSparse:
    return "DynamicTokenSparse";
  default:
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// JSON serialization macro (matching KernelConfigBase.h).
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef TO_JSON
#define TO_JSON(field) j[#field] = toString(field);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// TmemTile constant (replaces tg::TmemTile::TmemRowScale from TmemTile.h).
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg_compat {
static constexpr int TmemRowScale = 1 << 16;
} // namespace tg_compat

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha

// Provide stubs for trtllm-gen types not available in TRT-LLM.
namespace trtllm {
namespace gen {
struct TmemTile {
    static int constexpr TmemRowScale = ::fmha::tg_compat::TmemRowScale;
};
class GenCfg;
} // namespace gen
} // namespace trtllm

// Include KernelConfigBase.h for the KernelConfigBase struct definition.
// The enum/toString sections are skipped via #ifndef TLLM_FMHA_TRTLLM_COMPAT
// (already provided above as TRT-LLM type aliases).
#include "KernelConfigBase.h"
