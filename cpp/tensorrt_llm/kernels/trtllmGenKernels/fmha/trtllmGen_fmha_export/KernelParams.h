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
#include "KernelParamsDecl.h"
#include <trtllm/gen/DtypeDecl.h>

#ifdef TLLM_ENABLE_CUDA
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cute/tensor.hpp>
#else
// Fake Cutlass data types for compilation (on Mac).
namespace cutlass {
struct float_e4m3_t {};
struct half_t {};
struct bfloat16_t {};
} // namespace cutlass
#endif

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace KernelParamsSetup {
#ifdef TLLM_ENABLE_CUDA
static CUtensorMap buildNdTmaDescriptor(tg::Dtype dtypeElt,
                                        const std::vector<uint64_t>& shapes,
                                        const std::vector<uint64_t>& strides,
                                        const std::vector<uint32_t>& tileShapes,
                                        void* gmemAddr,
                                        bool swizzled = true,
                                        bool unpack4b = false) {
  CUtensorMap desc{};
  // The data type.
  CUtensorMapDataType tmaDataFormat{CU_TENSOR_MAP_DATA_TYPE_FLOAT32};
  if (dtypeElt == tg::Dtype::E2m1) {
    tmaDataFormat =
      unpack4b ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B : CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (dtypeElt == tg::Dtype::E4m3 || dtypeElt == tg::Dtype::Int8) {
    tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (dtypeElt == tg::Dtype::Fp16) {
    tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if (dtypeElt == tg::Dtype::Bfloat16) {
    tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else {
    std::cerr << "Unexpected dtype " << static_cast<int32_t>(dtypeElt) << std::endl;
    assert(false);
  }

  // The swizzle type.
  CUtensorMapSwizzle swizzleType{CU_TENSOR_MAP_SWIZZLE_NONE};
  int32_t numBytesInLeadingDim = tileShapes[0] * tg::dtypeGetNumBits(dtypeElt) / 8 /*bits*/;
  if (!swizzled) {
    swizzleType = CU_TENSOR_MAP_SWIZZLE_NONE;
  } else if (tmaDataFormat == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) {
    swizzleType = CU_TENSOR_MAP_SWIZZLE_128B;
  } else if ((numBytesInLeadingDim % 128) == 0) {
    swizzleType = CU_TENSOR_MAP_SWIZZLE_128B;
  } else if ((numBytesInLeadingDim % 64) == 0) {
    swizzleType = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if ((numBytesInLeadingDim % 32) == 0) {
    swizzleType = CU_TENSOR_MAP_SWIZZLE_32B;
  } else {
    std::cerr << "Unexpected numBytesInLeadingDim " << numBytesInLeadingDim << std::endl;
    assert(false);
  }

  // Check gmem address must be 16B-aligned
  assert((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0);

  // Check shape must be in range [1, 2^32]
  int32_t dim = shapes.size();
  // Max five dimension and min 2 dimension.
  assert((dim <= 5) && (dim >= 2));
  // Check shape range.
  for (int32_t ii = 0; ii < dim; ++ii) {
    assert(shapes[ii] >= (uint64_t(1)));       // Size must be min 1
    assert(shapes[ii] <= (uint64_t(1) << 32)); // Size must be max 2^32
  }

  // TMA descriptor does not store the zeroth stride and assumes it is 1.
  assert(static_cast<int32_t>(strides.size()) == dim);
  assert(strides[0] == 1);

  // Build strides in bytes.
  // cuTensorMapEncodeTiled ignores the stride of the first dimention (implicitly 1).
  std::vector<uint64_t> stridesInBytes(dim - 1);
  for (int32_t ii = 0; ii < dim - 1; ++ii) {
    stridesInBytes[ii] = strides[ii + 1] * std::max(tg::dtypeGetNumBits(dtypeElt), 8) / 8;
  }

  // Set tile strides to 0;
  std::vector<uint32_t> tileStrides(dim, 1);

  // Build the descriptor.
  CUresult result = cuTensorMapEncodeTiled(&desc,
                                           tmaDataFormat,
                                           /*tensorRank=*/dim,
                                           /*globalAddress=*/gmemAddr,
                                           /*globalDim=*/shapes.data(),
                                           /*globalStrides=*/stridesInBytes.data(),
                                           /*boxDim=*/tileShapes.data(),
                                           /*elementStrides=*/tileStrides.data(),
                                           /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE,
                                           /*swizzle=*/swizzleType,
                                           /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                           /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (result != CUDA_SUCCESS) {
    // Get the error string.
    const char* errStr = nullptr;
    cuGetErrorString(result, &errStr);
    std::cerr << "Error: Failed to initialize the TMA descriptor " << errStr << std::endl;
    std::cerr << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim
              << " gmem: " << gmemAddr << std::endl;
    std::cerr << "Shape: " << shapes[0] << " " << shapes[1] << " " << shapes[2] << " " << shapes[3]
              << std::endl;
    std::cerr << "Stride: " << stridesInBytes[0] << " " << stridesInBytes[1] << " "
              << stridesInBytes[2] << std::endl;
    std::cerr << "tileShapes: " << tileShapes[0] << " " << tileShapes[1] << " " << tileShapes[2]
              << " " << tileShapes[3] << std::endl;
    std::cerr << "tileStrides: " << tileStrides[0] << " " << tileStrides[1] << " " << tileStrides[2]
              << " " << tileStrides[3] << std::endl;
    std::cerr << "swizzleType: " << int(swizzleType) << std::endl;
    assert(false);
  }

  return desc;
}

// Create the TMA shape/stride for Q.
template <class FmhaOptions, class KernelTraits_>
static auto makeTmaShapeStrideQ(FmhaOptions const& options, KernelTraits_ const& kernelTraits) {

  //
  // The Q has shape of [numTokens * numHeadsQPerKv, numHeadsKv * 1, headDim]
  // when grouping headsQ, otherwise it would be [numTokens, numHeadsQPerKv * numHeadsKv,
  // headDim].

  // The headDim for Q and K.
  int headDimQk{options.mHeadDimQk};
  // The headDim for V.
  int headDimV{options.mHeadDimV};

  // The number of grouped heads for the A matrix of MMA.
  int32_t numGroupedHeads{1};
  if (options.mGroupsHeadsQ) {
    numGroupedHeads = std::min(options.mTileSizeQ, options.mNumHeadsQPerKv);
  }

  // The number of heads.
  int32_t numHeads{options.mNumHeadsQ};
  if (options.mGroupsHeadsQ) {
    numHeads /= numGroupedHeads;
  }
  // Make sure the math works.
  assert(numHeads * numGroupedHeads == options.mNumHeadsQ && "internal error");

  // The number of tokens.
  int32_t numTokens{options.mSumOfSeqLensQ};

  // This maps to flattened TMA shape for Q: (headDim, numTokens, numHeads).
  auto shape = std::vector<uint64_t>{static_cast<uint64_t>(headDimQk),
                                     static_cast<uint64_t>(numGroupedHeads),
                                     static_cast<uint64_t>(numHeads),
                                     static_cast<uint64_t>(numTokens)};

  // The hidden dimension when the tensor contains only Q (i.e. not QKV packed).
  int32_t const hiddenDimQ{options.mNumHeadsQ * headDimQk};

  // The hidden dimension when the Q, K and V tensors are packed.
  int32_t hiddenDimQkv{hiddenDimQ};
  if (isPackedQkv(options.mQkvLayout)) {
    assert(!options.mGroupsHeadsQ && "internal error");
    hiddenDimQkv += options.mNumHeadsKv * (headDimQk + headDimV);
  }

  // The stride between grouped heads.
  int32_t strideGroupedHeads{headDimQk};
  // The stride between heads.
  int32_t strideHeads{numGroupedHeads * strideGroupedHeads};
  // The stride between tokens.
  int32_t strideTokens{hiddenDimQkv};

  // Assemble the stride (1, strideTokens, strideHeads).
  // Swap the first two dimension as mentioned before.
  auto stride = std::vector<uint64_t>{1,
                                      static_cast<uint64_t>(strideGroupedHeads),
                                      static_cast<uint64_t>(strideHeads),
                                      static_cast<uint64_t>(strideTokens)};

  // The tile shape for TMA.
  auto tileShapes =
    std::vector<uint32_t>{static_cast<uint32_t>(kernelTraits.mNumEltsInClampedHeadDimQ),
                          1,
                          1,
                          static_cast<uint32_t>(options.mTileSizeQ)};
  // The tileSizePerCtaQ.
  int32_t tileSizePerCtaQ{options.mTileSizeQ * options.mNumInstsQ};
  // The number of tokensQ per CTA.
  int32_t numTokensPerCtaQ{tileSizePerCtaQ};
  // Re-compute the number of tokensQ per CTA if groupsHeadsQ is enabled.
  if (options.mGroupsHeadsQ) {
    if (options.mGroupsTokensHeadsQ) {
      // Currently, it requires each CTA to process complete headsQ (i.e. numGroupedHeads) at a
      // time, so it allows paddings in the end. Removing paddings needs re-organizing the Q
      // tensor to [numTokensQ, numGroupedHeads, numHeads, headDimQ] and we might want to revisit
      // this in the future.
      numTokensPerCtaQ = static_cast<int32_t>(numTokensPerCtaQ / numGroupedHeads);
    } else {
      numGroupedHeads = tileSizePerCtaQ;
      numTokensPerCtaQ = 1;
    }
    tileShapes =
      std::vector<uint32_t>{static_cast<uint32_t>(kernelTraits.mNumEltsInClampedHeadDimQ),
                            static_cast<uint32_t>(numGroupedHeads),
                            1,
                            static_cast<uint32_t>(numTokensPerCtaQ)};
  }

  return std::make_tuple(shape, stride, tileShapes, numTokensPerCtaQ);
}

// Create the TMA shape/stride for O.
template <class FmhaOptions> static auto makeTmaShapeStrideO(FmhaOptions const& options) {

  // The TmaO (i.e. mFuseEpilogueIntoCorr = true) should not be used when mGroupsHeadsQ is true.
  assert((!options.mGroupsHeadsQ || options.mFuseEpilogueIntoCorr) &&
         "GroupsHeadsQ is not supported");

  // The number of tokens.
  int32_t numTokens{options.mMaxSeqLenQ};
  if (options.mSupportsVarSeqLens) {
    numTokens = options.mSumOfSeqLensQ;
  }

  // The batch dimension.
  int32_t batchSize{options.mBatchSize};
  if (options.mSupportsVarSeqLens) {
    batchSize = 1;
  }

  // The shape for O is [batchSize/1, numTokens, numHeads, 1] (fastest-moving dim on the right).
  auto shape = std::vector<uint64_t>{static_cast<uint64_t>(options.mHeadDimV),
                                     static_cast<uint64_t>(options.mNumHeadsQ),
                                     static_cast<uint64_t>(numTokens),
                                     static_cast<uint64_t>(batchSize)};

  // The strides.
  std::vector<uint64_t> strides(4, uint64_t{0});
  // The inner-most stride must be 1.
  strides[0] = 1;
  // The stride between heads.
  strides[1] = strides[0] * options.mHeadDimV;
  // The stride between tokens.
  strides[2] = strides[1] * options.mNumHeadsQ;
  // The stride between sequences.
  strides[3] = options.mSupportsVarSeqLens ? uint64_t{0} : strides[2] * options.mMaxSeqLenQ;

  return std::make_tuple(shape, strides);
}

// Create the shape for K and V.
template <class FmhaOptions>
static auto makeShapeKv(FmhaOptions const& options, KernelParams const& params) {

  // The number of keys/vals. WARNING: The if/else-if are sorted by priority.
  int32_t numKeysVals{options.mMaxSeqLenKv};
  if (isPagedKv(options.mQkvLayout)) {
    numKeysVals = options.mNumTokensPerPage;
  } else if (isContiguousKv(options.mQkvLayout)) {
    numKeysVals = options.mMaxSeqLenKv;
  } else if (options.mSupportsVarSeqLens) {
    numKeysVals = options.mSumOfSeqLensKv;
  }

  // The number of heads per K/V head (packed in the sequence length for mGroupsHeadsQ).
  int32_t numHeadsKv{options.mNumHeadsKv};

  // The batch dimension. WARNING: The if/else-if are sorted by priority.
  int32_t batchSize{options.mBatchSize};
  if (isPagedKv(options.mQkvLayout)) {
    batchSize = params.mNumPagesInMemPool;
  } else if (isContiguousKv(options.mQkvLayout)) {
    batchSize = options.mBatchSize;
  } else if (options.mSupportsVarSeqLens) {
    batchSize = 1;
  }

  // Return the number of keys and batch.
  return std::make_tuple(numKeysVals, numHeadsKv, batchSize);
}

// Compute the strides for K and V.
template <class FmhaOptions> static auto makeStrideKv(FmhaOptions const& options, bool isK) {

  // The headDim for Q and K.
  int headDimQk{options.mHeadDimQk};
  // The headDim for V.
  int headDimV{options.mHeadDimV};
  // The maximumal head size in bytes for K and V.
  int32_t const headSizeBytesK = headDimQk * tg::dtypeGetNumBits(options.mDtypeK) / 8;
  int32_t const headSizeBytesV = headDimV * tg::dtypeGetNumBits(options.mDtypeV) / 8;
  int32_t const maxHeadSizeBytesKv = std::max(headSizeBytesK, headSizeBytesV);
  // The padded head size expressed as element count in dtypeK and dtypeV.
  int32_t const paddedHeadDimK = maxHeadSizeBytesKv * 8 / tg::dtypeGetNumBits(options.mDtypeK);
  int32_t const paddedHeadDimV = maxHeadSizeBytesKv * 8 / tg::dtypeGetNumBits(options.mDtypeV);
  int32_t const paddedHeadDimKv = isK ? paddedHeadDimK : paddedHeadDimV;
  // The hidden dimension for the K or V tensor currently described.
  int32_t const hiddenDimKv{options.mNumHeadsKv * paddedHeadDimKv};

  // The hidden dimension when Q, K and V are packed together.
  int32_t const hiddenDimQkv{(options.mNumHeadsQ + options.mNumHeadsKv) * headDimQk +
                             options.mNumHeadsKv * headDimV};

  int32_t strideKeysVals{};
  int32_t strideHeads{};
  int32_t strideBatch{};

  if (options.mIsTrtllmLayout) {
    // TRT-LLM KV cache layout: K/V token strides differ; context MLA uses non-contiguous V layout.
    int32_t const hiddenDimK{options.mNumHeadsKv * headDimQk};
    int32_t const hiddenDimV{options.mNumHeadsKv * headDimV};

    strideKeysVals = isK ? hiddenDimK : hiddenDimV;
    if (isPagedKv(options.mQkvLayout)) {
      strideKeysVals = paddedHeadDimKv;
    } else if (isPackedQkv(options.mQkvLayout)) {
      strideKeysVals = hiddenDimQkv;
    } else if (isContiguousKv(options.mQkvLayout)) {
      strideKeysVals = paddedHeadDimKv;
    } else if (isSeparateQkv(options.mQkvLayout) && !isK && options.mHeadDimQk == 192 &&
               options.mDtypeKv != tg::Dtype::E4m3) {
      // Non-FP8 context MLA: tensor V is not contiguous.
      strideKeysVals = options.mNumHeadsKv * (options.mHeadDimQk - 64 + options.mHeadDimV);
    }

    strideHeads = isK ? headDimQk : headDimV;
    if (isPagedKv(options.mQkvLayout)) {
      strideHeads = options.mNumTokensPerPage * paddedHeadDimKv;
    } else if (isContiguousKv(options.mQkvLayout)) {
      strideHeads = options.mMaxSeqLenKv * paddedHeadDimKv;
    }

    strideBatch = options.mMaxSeqLenKv * hiddenDimKv;
    if (isPagedKv(options.mQkvLayout)) {
      strideBatch = options.mNumTokensPerPage * hiddenDimKv;
    } else if (isContiguousKv(options.mQkvLayout)) {
      strideBatch = 2 * options.mMaxSeqLenKv * hiddenDimKv;
    } else if (options.mSupportsVarSeqLens) {
      strideBatch = 0;
    } else if (isPackedQkv(options.mQkvLayout)) {
      strideBatch = options.mMaxSeqLenKv * hiddenDimQkv;
    }
  } else {
    // Default (trtllm-gen internal test): HeadDim * dtype is padded to maxHeadSizeBytesKv.
    strideKeysVals = hiddenDimKv;
    if (isPagedKv(options.mQkvLayout)) {
      strideKeysVals = paddedHeadDimKv;
    } else if (isPackedQkv(options.mQkvLayout)) {
      strideKeysVals = hiddenDimQkv;
    } else if (isContiguousKv(options.mQkvLayout)) {
      strideKeysVals = paddedHeadDimKv;
    }

    strideHeads = isK ? headDimQk : headDimV;
    if (isPagedKv(options.mQkvLayout)) {
      strideHeads = options.mNumTokensPerPage * paddedHeadDimKv;
    } else if (isContiguousKv(options.mQkvLayout)) {
      strideHeads = options.mMaxSeqLenKv * paddedHeadDimKv;
    } else if (isSeparateQkv(options.mQkvLayout)) {
      strideHeads = paddedHeadDimKv;
    }

    strideBatch = options.mMaxSeqLenKv * hiddenDimKv;
    if (isPagedKv(options.mQkvLayout)) {
      strideBatch = options.mNumTokensPerPage * hiddenDimKv;
    } else if (isContiguousKv(options.mQkvLayout)) {
      strideBatch = 2 * options.mMaxSeqLenKv * hiddenDimKv;
    } else if (options.mSupportsVarSeqLens) {
      strideBatch = 0;
    } else if (isPackedQkv(options.mQkvLayout)) {
      strideBatch = options.mMaxSeqLenKv * hiddenDimQkv;
    }
  }

  return std::make_tuple(strideKeysVals, strideHeads, strideBatch);
}

// Create the TMA shape/stride for K.
template <class FmhaOptions>
static auto makeTmaShapeStrideKv(FmhaOptions const& options,
                                 KernelParams const& params,
                                 bool isK,
                                 bool storeTransformedKvInTmem,
                                 int reshapeFactor) {
  // The shape elements.
  auto [numKeys, numHeadsQPerKv, batchSize] = makeShapeKv(options, params);
  // The stride elements.
  auto [strideKeys, strideHeads, strideBatch] = makeStrideKv(options, isK);

  // The TMA layout is 4D:
  // Shape:  (headDim, numKeys, numHeadsKv, batchSize)
  // Stride: (1, strideKeys, strideHeads, strideBatch)
  //
  // We reshape by a factor r so the headDim is 128B+, reducing the number of TMA read requests:
  // Shape:  (headDim * r, numKeys / r, numHeadsKv, batchSize)
  // Stride: (1, strideKeys * r, strideHeads, strideBatch)

  // For K the headDim may include extra RoPE coefficients.
  auto headDim = isK ? options.mHeadDimQk : options.mHeadDimV;
  // Note that for FP4 KV input, elements are stored as uint8_t, each packs 2 FP4 elements.
  // The column index and strides needs to divide by 2.
  auto const colIdxDivisor = options.mDtypeKv == tg::Dtype::E2m1 ? 2 : 1;
  // When mStoreTransformedKvInTmem is true, the dimensions reflect FP4 element dimensions, thus
  // no need to divide.
  auto shape = std::vector<uint64_t>{
    static_cast<uint64_t>((storeTransformedKvInTmem ? headDim : headDim / colIdxDivisor) *
                          reshapeFactor),
    static_cast<uint64_t>(numKeys / reshapeFactor),
    static_cast<uint64_t>(options.mNumHeadsKv),
    static_cast<uint64_t>(batchSize)};
  auto stride =
    std::vector<uint64_t>{1,
                          static_cast<uint64_t>(strideKeys / colIdxDivisor * reshapeFactor),
                          static_cast<uint64_t>(strideHeads / colIdxDivisor),
                          static_cast<uint64_t>(strideBatch / colIdxDivisor)};

  return std::make_tuple(shape, stride);
}

// Create the TMA shape/stride for KV scaling factors.
template <class FmhaOptions, class KernelTraits_>
static auto makeTmaShapeStrideKvSf(FmhaOptions const& options,
                                   KernelTraits_ const& kernelTraits,
                                   KernelParams const& params,
                                   bool isK,
                                   int reshapeFactor) {
  // The shape elements.
  auto [numKeys, numHeadsQPerKv, batchSize] = makeShapeKv(options, params);
  // The stride elements.
  auto [strideKeys, strideHeads, strideBatch] = makeStrideKv(options, isK);

  // The number of elements per SF.
  int32_t NumEltsPerSf = kernelTraits.mNumEltsPerSf;

  // The KV shape is:    (headDim, numKeys, numHeadsKv, batchSize)
  // The KV SF shape is: (headDim / NumEltsPerSf, numKeys, numHeadsKv, batchSize).
  // Considering the TMA requires box width to be multiple of 16B, and ideally >= 128B, we reshape
  // with a factor into: (headDim / NumEltsPerSf * r, numKeys / r, numHeadsKv, batchSize)

  // Note that it only works for pagedKv layout.
  assert(isPagedKv(options.mQkvLayout));

  // For K the headDim may include extra RoPE coefficients.
  auto headDim = isK ? options.mHeadDimQk : options.mHeadDimV;

  auto shape = std::vector<uint64_t>{static_cast<uint64_t>(headDim / NumEltsPerSf * reshapeFactor),
                                     static_cast<uint64_t>(numKeys / reshapeFactor),
                                     static_cast<uint64_t>(options.mNumHeadsKv),
                                     static_cast<uint64_t>(batchSize)};
  auto stride = std::vector<uint64_t>{1,
                                      static_cast<uint64_t>(headDim / NumEltsPerSf * reshapeFactor),
                                      static_cast<uint64_t>(strideHeads / NumEltsPerSf),
                                      static_cast<uint64_t>(strideBatch / NumEltsPerSf)};

  return std::make_tuple(shape, stride);
}

// Update the kernel parameters.
template <class FmhaOptions_>
static KernelParams updateKernelParams(FmhaOptions_ const& options,
                                       KernelParams const& params,
                                       void const* qBasePtr,
                                       void const* kBasePtr,
                                       void const* vBasePtr,
                                       void* oBasePtr,
                                       void const* kSfBasePtr,
                                       void const* vSfBasePtr) {
  return setKernelParams(options,
                         params.logicalGridDimX,
                         params.logicalGridDimY,
                         params.logicalGridDimZ,
                         params.ptrCumSeqLensQ,
                         params.ptrCumSeqLensKv,
                         params.ptrSeqLensKv,
                         qBasePtr,
                         kBasePtr,
                         vBasePtr,
                         kSfBasePtr,
                         vSfBasePtr,
                         params.ptrPageIdxKv,
                         params.ptrOutputScale,
                         params.ptrScaleSoftmaxLog2,
                         params.ptrScaleSfKv,
                         params.ptrScaleSfO,
                         params.ptrCustomMask,
                         params.ptrCustomMaskOffsets,
                         params.ptrFirstSparseMaskOffsetsKv,
                         params.ptrSageAttnSfsQ,
                         params.ptrSageAttnSfsK,
                         params.ptrSageAttnSfsP,
                         params.ptrSageAttnSfsV,
                         params.ptrAttentionSinks,
                         oBasePtr,
                         params.ptrSfO,
                         params.ptrMultiCtasKvCounter,
                         params.ptrPartialO,
                         params.ptrPartialStats,
                         params.ptrSkipSoftmaxStats,
                         params.ptrSoftmaxStats,
                         params.ptrDebugO,
                         params.mScaleSoftmaxLog2,
                         params.mInflateMax,
                         params.mScaleSfKv,
                         params.mScaleSfO,
                         params.mStartTokenIdxSfO,
                         params.mUseBlockSparseAttention,
                         params.mUsesSharedPagedKvIdx);
}

// Setup the kernel parameters.
template <class FmhaOptions_>
static KernelParams setKernelParams(FmhaOptions_ const& options,
                                    int32_t logicalGridDimX,
                                    int32_t logicalGridDimY,
                                    int32_t logicalGridDimZ,
                                    int const* cumSeqLensQPtrD,
                                    int const* cumSeqLensKvPtrD,
                                    int const* seqLensKvPtrD,
                                    void const* qBasePtr,
                                    void const* kBasePtr,
                                    void const* vBasePtr,
                                    void const* kSfBasePtr,
                                    void const* vSfBasePtr,
                                    int const* kvPageIdxD,
                                    float const* outputScaleD,
                                    float const* scaleSoftmaxLog2D,
                                    float const* kvSfScaleD,
                                    float const* oSfScaleD,
                                    uint32_t const* customMaskPtrD,
                                    int64_t const* customMaskOffsetsPtrD,
                                    int32_t const* firstSparseMaskOffsetsKvPtrD,
                                    float const* ptrSageAttnSfsQ,
                                    float const* ptrSageAttnSfsK,
                                    float const* ptrSageAttnSfsP,
                                    float const* ptrSageAttnSfsV,
                                    float const* ptrAttentionSinks,
                                    void* oPtrD,
                                    void* oSfPtrD,
                                    int* multiCtasKvCounterPtrD,
                                    void* partialOPtrD,
                                    float2* partialStatsPtrD,
                                    int* skipSoftmaxStatsPtrD,
                                    float2* softmaxStatsD,
                                    void* oDebugPtrD,
                                    float softmaxScale,
                                    float inflateMax,
                                    float kvSfScale,
                                    float oSfScale,
                                    int32_t startTokenIdxSfO,
                                    bool useBlockSparseAttention,
                                    bool usesSharedPagedKvIdx) {

  // Create the return struct.
  KernelParams params;

  params.logicalGridDimX = logicalGridDimX;
  params.logicalGridDimY = logicalGridDimY;
  params.logicalGridDimZ = logicalGridDimZ;

  // Set the number of pages in the memory pool for paged K/V cache.
  if (isPagedKv(options.mQkvLayout)) {
    params.mNumPagesInMemPool = options.mNumPagesInMemPool == 0
                                  ? options.mMaxNumPagesPerSeqKv * 2 * options.mBatchSize
                                  : options.mNumPagesInMemPool;
  }

  // Get the kernel traits from options.
  auto kernelTraits = getKernelTraitsFromOptions(options);

  // Shape/stride for gmem tensor Q.
  auto [shapeQ, strideQ, tileShapeQ, numTokensPerCtaQ] = makeTmaShapeStrideQ(options, kernelTraits);
  // Build tma descriptor for Q.
  params.tmaQ_ =
    buildNdTmaDescriptor(options.mDtypeQ, shapeQ, strideQ, tileShapeQ, const_cast<void*>(qBasePtr));

  // Build tma descriptor for K.
  // Whether store transformed K/V in TMEM.
  bool const storeTransformedKvInTmem{kernelTraits.mStoreTransformedKvInTmem};
  // Note that for FP4 KV input, elements are stored as uint8_t, each packs 2 FP4 elements.
  auto const numEltsDivisor =
    options.mDtypeKv == tg::Dtype::E2m1 && !storeTransformedKvInTmem ? 2 : 1;

  // Shape/stride for gmem tensor Kv.
  auto [shapeK, strideK] = makeTmaShapeStrideKv(options,
                                                params,
                                                /*isK*/ true,
                                                storeTransformedKvInTmem,
                                                kernelTraits.mReshapeFactorKv);

  // The tileShapes for K/V.
  std::vector<uint32_t> tileShapeKv(shapeK.size(), 1);
  tileShapeKv[0] =
    kernelTraits.mNumEltsInClampedHeadDimKv / numEltsDivisor * kernelTraits.mReshapeFactorKv;
  tileShapeKv[1] = kernelTraits.mNumKeysPerTile / kernelTraits.mReshapeFactorKv;
  // K and V might use different tileShapes.
  std::vector<uint32_t> tileShapeK(tileShapeKv);
  std::vector<uint32_t> tileShapeV(tileShapeKv);
  if (!storeTransformedKvInTmem && !kernelTraits.mSeparateLoadKvTask &&
      options.mDtypeK != options.mDtypeV) {
    // tileShapeKv is in dtypeK elements. When dtypeV != dtypeK, we need to express tileShapeV in
    // terms of dtypeV elements so the V TMA descriptor transfers the same number of bytes as K to
    // match barrier expectations.
    tileShapeV[0] =
      tileShapeV[0] * tg::dtypeGetNumBits(options.mDtypeK) / tg::dtypeGetNumBits(options.mDtypeV);
  }
  // If there is only one page per tile, each CTA will only load half of NumKeysPerTile for K.
  if (options.mClusterDimX == 2 && kernelTraits.mNumKeysPerTile == kernelTraits.mTileSizeKv) {
    tileShapeK[1] /= 2;
  }

  // If sparse attention is enabled, the shape and stride for K need to be updated for 2D layout
  // (numTokensKvInPagedKv, headDimQk). Use INT_MAX to better align with other cache layouts.
  if (isTokenSparse(options.mSparseType)) {
    shapeK = std::vector<uint64_t>{static_cast<uint64_t>(options.mHeadDimQk),
                                   static_cast<uint64_t>(INT_MAX)};
    strideK = std::vector<uint64_t>{1, static_cast<uint64_t>(options.mHeadDimQk)};
    tileShapeK[1] = 1;
  }

  // Create the TMA descriptor.
  params.tmaK_ = buildNdTmaDescriptor(options.mDtypeK,
                                      shapeK,
                                      strideK,
                                      tileShapeK,
                                      const_cast<void*>(kBasePtr),
                                      /*swizzled=*/kernelTraits.mSwizzleKv,
                                      /*unpack4b=*/storeTransformedKvInTmem);

  // Shape/stride for gmem tensor V.
  auto [shapeV, strideV] = makeTmaShapeStrideKv(options,
                                                params,
                                                /*isK*/ false,
                                                storeTransformedKvInTmem,
                                                kernelTraits.mReshapeFactorKv);
  // For sparse MQA/GQA (not MLA), V also needs 2D flattened descriptor with separate base pointer.
  if (isTokenSparse(options.mSparseType) && !options.mIsMlaGen) {
    shapeV = std::vector<uint64_t>{static_cast<uint64_t>(options.mHeadDimV),
                                   static_cast<uint64_t>(INT_MAX)};
    strideV = std::vector<uint64_t>{1, static_cast<uint64_t>(options.mHeadDimV)};
    tileShapeV[1] = 1;
  }
  // Build the tma descriptor for V.
  params.tmaV_ = buildNdTmaDescriptor(options.mDtypeV,
                                      shapeV,
                                      strideV,
                                      tileShapeV,
                                      // MlaGen kernels reuse the same buffer for K and V.
                                      const_cast<void*>(options.mIsMlaGen ? kBasePtr : vBasePtr),
                                      /*swizzled=*/kernelTraits.mSwizzleKv,
                                      /*unpack4b=*/storeTransformedKvInTmem);

  // If the KV dtype is E2m1, additional scaling factors are needed for dequant.
  if (options.mDtypeKv == tg::Dtype::E2m1) {
    // The maximum headDim for K and V.
    int maxHeadDimKv{std::max(options.mHeadDimQk, options.mHeadDimV)};
    // The number of elements per SF.
    int32_t NumEltsPerSf = kernelTraits.mNumEltsPerSf;
    // Compute the shape and stride for SF tensor.
    // FIXME: assume K and V uses the same shape.
    auto [shapeKvSf, strideKvSf] = makeTmaShapeStrideKvSf(options,
                                                          kernelTraits,
                                                          params,
                                                          /*isK*/ true,
                                                          kernelTraits.mReshapeFactorKvSf);
    // The tileShapes for K/V.
    std::vector<uint32_t> tileShapeKvSf(shapeKvSf.size(), 1);
    tileShapeKvSf[0] = maxHeadDimKv / NumEltsPerSf * kernelTraits.mReshapeFactorKvSf;
    tileShapeKvSf[1] = kernelTraits.mNumKeysPerTile / kernelTraits.mReshapeFactorKvSf;

    // The tile box is reshaped from (headDim / NumEltsPerSf, tileSizeKv) into
    // (headDim / NumEltsPerSf * reshapeFactorKvSf, tileSizeKv / reshapeFactorKvSf).
    // Build tma descriptor for K SF.
    // If multiple headDimStages are used, we will load the full headDim's SFs to avoid reshaping
    // SFs layout.
    params.tmaKSf_ = buildNdTmaDescriptor(tg::Dtype::E4m3,
                                          shapeKvSf,
                                          strideKvSf,
                                          tileShapeKvSf,
                                          const_cast<void*>(kSfBasePtr),
                                          /*swizzled = */ false);
    // Build tma descriptor for V SF.
    params.tmaVSf_ = buildNdTmaDescriptor(tg::Dtype::E4m3,
                                          shapeKvSf,
                                          strideKvSf,
                                          tileShapeKvSf,
                                          const_cast<void*>(vSfBasePtr),
                                          /*swizzled = */ false);
  }

  // Shape/stride for gmem tensor O.
  auto [shapeO, strideO] = makeTmaShapeStrideO(options);

  // The tileShapes for O (in the inner-most dimension and in the token dimension).
  std::vector<uint32_t> tileShapeO(shapeO.size(), 1);
  tileShapeO[0] = kernelTraits.mNumEltsInClampedHeadDimQ;
  tileShapeO[2] = options.mTileSizeQ;

  // Build tma descriptor for O.
  params.tmaO_ =
    buildNdTmaDescriptor(options.mDtypeQ, shapeO, strideO, tileShapeO, const_cast<void*>(oPtrD));

  // Set the attention sinks pointer.
  params.ptrAttentionSinks = ptrAttentionSinks;

  // The cumulative sequence lengths for Q.
  params.ptrCumSeqLensQ = cumSeqLensQPtrD;
  // The cumulative sequence lengths for K/V.
  params.ptrCumSeqLensKv = cumSeqLensKvPtrD;

  // The packed custom mask.
  params.ptrCustomMask = customMaskPtrD;
  // The packed custom mask's offsets of each sequence.
  params.ptrCustomMaskOffsets = customMaskOffsetsPtrD;
  // The first sparseMask offsets in the Kv sequence dimension.
  params.ptrFirstSparseMaskOffsetsKv = firstSparseMaskOffsetsKvPtrD;

  // The debug buffers.
  params.ptrDebugO = static_cast<float*>(oDebugPtrD);

  // The skip softmax stats buffer.
  params.ptrSkipSoftmaxStats = skipSoftmaxStatsPtrD;

  // The softmax stats buffer.
  params.ptrSoftmaxStats = options.mStoresSoftmaxStats ? softmaxStatsD : nullptr;

  // The output buffer.
  params.ptrO = oPtrD;
  params.ptrSfO = oSfPtrD;

  // TRT-LLM restrictions: the quantization scales must be on the device. It will only be loaded
  // when -loadsScalesFromGmem true -dtypeElt e4m3 are specified.
  params.ptrOutputScale = outputScaleD;

  // The partial buffers' pointers when the multiCtasKv mode is enabled.
  params.ptrMultiCtasKvCounter = multiCtasKvCounterPtrD;
  params.ptrPartialO = partialOPtrD;
  params.ptrPartialStats = partialStatsPtrD;
  params.ptrPageIdxKv = kvPageIdxD;
  params.ptrSageAttnSfsK = ptrSageAttnSfsK;
  params.ptrSageAttnSfsP = ptrSageAttnSfsP;
  params.ptrSageAttnSfsQ = ptrSageAttnSfsQ;
  params.ptrSageAttnSfsV = ptrSageAttnSfsV;
  params.ptrScaleSoftmaxLog2 = scaleSoftmaxLog2D;
  params.ptrScaleSfKv = kvSfScaleD;
  params.ptrScaleSfO = oSfScaleD;

  // The sequence lengths for K/V.
  params.ptrSeqLensKv = seqLensKvPtrD;


  params.mAttentionWindowSize = options.mAttentionWindowSize;
  if (options.mChunkedAttentionSize > 0) {
    // The chunked attention size is a power of 2 (verified in FmhaOptions.h).
    params.mChunkedAttentionSizeLog2 = std::log2(options.mChunkedAttentionSize);
  } else {
    params.mChunkedAttentionSizeLog2 = 0;
  }
  params.mBatchSize = options.mBatchSize;

  params.mInflateMax = inflateMax;

  // Compute the logs for the numbers of elements in the Sage Attention blocks.
  int32_t logNumEltsPerSageAttnBlkK{-1};
  if (options.mNumEltsPerSageAttnBlkK == -1) {
    logNumEltsPerSageAttnBlkK = 32;
  } else if (options.mNumEltsPerSageAttnBlkK != 0) {
    logNumEltsPerSageAttnBlkK = (int)log2f((float)options.mNumEltsPerSageAttnBlkK);
  }
  int32_t logNumEltsPerSageAttnBlkP{-1};
  if (options.mNumEltsPerSageAttnBlkP == -1) {
    logNumEltsPerSageAttnBlkP = 32;
  } else if (options.mNumEltsPerSageAttnBlkP != 0) {
    logNumEltsPerSageAttnBlkP = (int)log2f((float)options.mNumEltsPerSageAttnBlkP);
  }
  int32_t logNumEltsPerSageAttnBlkQ{-1};
  if (options.mNumEltsPerSageAttnBlkQ == -1) {
    logNumEltsPerSageAttnBlkQ = 32;
  } else if (options.mNumEltsPerSageAttnBlkQ != 0) {
    logNumEltsPerSageAttnBlkQ = (int)log2f((float)options.mNumEltsPerSageAttnBlkQ);
  }
  int32_t logNumEltsPerSageAttnBlkV{-1};
  if (options.mNumEltsPerSageAttnBlkV == -1) {
    logNumEltsPerSageAttnBlkV = 32;
  } else if (options.mNumEltsPerSageAttnBlkV != 0) {
    logNumEltsPerSageAttnBlkV = (int)log2f((float)options.mNumEltsPerSageAttnBlkV);
  }

  // Set the Sage Attention params.
  params.mLogNumEltsPerSageAttnBlkK = logNumEltsPerSageAttnBlkK;
  params.mLogNumEltsPerSageAttnBlkQ = logNumEltsPerSageAttnBlkQ;
  params.mLogNumEltsPerSageAttnBlkP = logNumEltsPerSageAttnBlkP;
  params.mLogNumEltsPerSageAttnBlkV = logNumEltsPerSageAttnBlkV;

  // Compute the log of numTokensPerPage
  int32_t numTokensPerPageLog2{-1};
  if (options.mDynamicNumTokensPerPage) {
    TLLM_CHECK_ERROR((options.mNumTokensPerPage & (options.mNumTokensPerPage - 1)) == 0,
                     "NumTokensPerPage must be power of 2");
    numTokensPerPageLog2 = (int)log2f((float)options.mNumTokensPerPage);
  }
  params.mNumTokensPerPageLog2 = numTokensPerPageLog2;

  params.mMaxSeqLenQ = options.mMaxSeqLenQ;
  params.mMaxSeqLenKv = options.mMaxSeqLenKv;
  params.mMaxNumCtasQ = options.mMaxNumCtasQ;
  params.mMaxNumCtasKv = options.mMaxNumCtasKv;
  params.mMaxNumPagesPerSeqKv = options.mMaxNumPagesPerSeqKv;
  params.mNumHeadsKv = options.mNumHeadsKv;
  params.mNumHeadsQ = options.mNumHeadsQ;
  params.mNumHeadsQPerKv = options.mNumHeadsQPerKv;
  params.mNumHeadsQPerKvDivisor = trtllm::dev::fast_mod_div(options.mNumHeadsQPerKv);
  params.mNumHiddenEltsO = options.mNumHeadsQ * options.mHeadDimV;
  params.mNumTokensPerCtaQ = numTokensPerCtaQ;
  params.mOutputScale = options.mOutputScale;
  params.mScaleSoftmaxLog2 = softmaxScale;
  params.mScaleSfKv = kvSfScale;
  params.mScaleSfO = oSfScale;
  params.mStartTokenIdxSfO = startTokenIdxSfO;
  params.mSumOfSeqLensQ = options.mSumOfSeqLensQ;
  params.mSumOfSeqLensKv = options.mSumOfSeqLensKv;
  params.mSkipSoftmaxThresholdScaleFactor = options.mSkipSoftmaxThresholdScaleFactor;
  params.mSparseAttnTopK = options.mSparseAttnTopK;

  // Set the block sparse attention flag.
  params.mUseBlockSparseAttention = useBlockSparseAttention;
  // Set the shared paged KV index flag.
  params.mUsesSharedPagedKvIdx = usesSharedPagedKvIdx;
  return params;
}
#else
// Update the kernel parameters.
template <class FmhaOptions_>
static KernelParams updateKernelParams(FmhaOptions_ const&,
                                       KernelParams const&,
                                       void const*,
                                       void const*,
                                       void const*,
                                       void*,
                                       void const*,
                                       void const*) {
  return KernelParams{};
}

// Setup the kernel parameters.
template <class FmhaOptions_>
static KernelParams setKernelParams(FmhaOptions_ const&,
                                    int32_t,
                                    int32_t,
                                    int32_t,
                                    int const*,
                                    int const*,
                                    int const*,
                                    void const*,
                                    void const*,
                                    void const*,
                                    int const*,
                                    float const*,
                                    uint32_t const*,
                                    int64_t const*,
                                    int32_t const*,
                                    float const*,
                                    float const*,
                                    float const*,
                                    float const*,
                                    void*,
                                    int*,
                                    void*,
                                    float2*,
                                    int*,
                                    float2*,
                                    void*,
                                    float,
                                    float,
                                    float,
                                    int32_t,
                                    bool,
                                    bool) {
  return KernelParams{};
}
#endif
} // namespace KernelParamsSetup

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
