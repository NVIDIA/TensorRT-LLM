/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cassert>
#include <cstring>
#include <string>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Be careful when modifying this file as it is included by the generated kernels. For example, do
// not add TLLM_CHECK_* constructs in this file. Thanks!
//
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace trtllm {
namespace gen {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class CudaArch {
  // Hopper
  Sm90a = 0,
  // Blackwell
  Sm100a,
  // Blackwell-family
  Sm100f,
  // Blackwell Ultra
  Sm103a,
#ifdef TLLM_RUBIN_FEATURES
  // SM107
  Sm107a,
#endif // TLLM_RUBIN_FEATURES
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool isArchHopper(CudaArch cudaArch) {
  return cudaArch == CudaArch::Sm90a;
}

#ifdef TLLM_RUBIN_FEATURES
inline bool isArchRubin(CudaArch cudaArch) {
  // Note: when compiling with a Blackwell target compatible with Rubin such as 100f,
  // no Rubin-specific features shall be used.
  if (cudaArch == CudaArch::Sm107a) {
    return true;
  }
  return false;
}
#endif // TLLM_RUBIN_FEATURES

inline bool isArchBlackwell(CudaArch cudaArch) {
#ifdef TLLM_RUBIN_FEATURES
  if (cudaArch == CudaArch::Sm107a) {
    return true;
  }
#endif // TLLM_RUBIN_FEATURES
  return cudaArch == CudaArch::Sm100a || cudaArch == CudaArch::Sm100f ||
         cudaArch == CudaArch::Sm103a;
}

inline bool isArchBlackwellUltra(CudaArch cudaArch) {
  return cudaArch == CudaArch::Sm103a;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string cudaArchToString(CudaArch cudaArch, bool isFull = true) {
  switch (cudaArch) {
  case CudaArch::Sm90a:
    return isFull ? "90a" : "90";
  case CudaArch::Sm100a:
    return isFull ? "100a" : "100";
  case CudaArch::Sm100f:
    return isFull ? "100f" : "100";
  case CudaArch::Sm103a:
    return isFull ? "103a" : "103";
#ifdef TLLM_RUBIN_FEATURES
  case CudaArch::Sm107a:
    return isFull ? "107a" : "107";
#endif // TLLM_RUBIN_FEATURES
  default:
    assert(false);
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline CudaArch stringToCudaArch(std::string const& str) {
  if (str == "90a") {
    return CudaArch::Sm90a;
  } else if (str == "100a") {
    return CudaArch::Sm100a;
  } else if (str == "100f") {
    return CudaArch::Sm100f;
  } else if (str == "103a") {
    return CudaArch::Sm103a;
#ifdef TLLM_RUBIN_FEATURES
  } else if (str == "107a") {
    return CudaArch::Sm107a;
#endif // TLLM_RUBIN_FEATURES
  } else {
    assert(false);
    return CudaArch::Sm100a;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm