/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ATTENTION_MASK_CUH_
#define FLASHINFER_ATTENTION_MASK_CUH_

namespace flashinfer {

enum class MaskMode {
  kNone = 0U,    // No mask
  kCausal = 1U,  // Causal mask
  kCustom = 2U,  // Custom mask
  kMultiItemScoring = 3U,
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_MASK_CUH_
