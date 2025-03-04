#include "MixtureOfExpertsInterface.h"

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function will optionally reorder and shuffle the weights for a single expert.
//
// Both of these operations will use temporary buffers, so this function supports performing the
// transformations in-place.
//
// When running with transposed output, shuffling the weight (A) matrix improves the memory access
// pattern for storing data to shared memory in the epilogue.
//
//
// When using gated activation fusion, the weights for the activation and gates should be
// interleaved so that a CTA can compute the activation in the epilogue.
//
void prepareWeightsOnHost(void const* wIn, void const* wSfIn, void* wOut, void* wSfOut, tg::Dtype dtypeElt,
    const int32_t m, const int32_t k, const int32_t epilogueTileM, bool const useShuffleMatrix, bool const useFusedAct,
    bool const useBlockScaling, int const numEltsPerSf)
{
    void const* shuffleIn;
    void const* shuffleSfIn;
    if (useFusedAct)
    {
        if (dtypeElt == tg::Dtype::Fp16)
        {
            reorderRowsForGatedActGemm<cutlass::half_t>(wIn, wOut, m, k);
        }
        else if (dtypeElt == tg::Dtype::Bfloat16)
        {
            reorderRowsForGatedActGemm<cutlass::bfloat16_t>(wIn, wOut, m, k);
        }
        else if (dtypeElt == tg::Dtype::E4m3)
        {
            reorderRowsForGatedActGemm<cutlass::float_e4m3_t>(wIn, wOut, m, k);
        }
        else if (dtypeElt == tg::Dtype::E2m1)
        {
            reorderRowsForGatedActGemm<cutlass::float_e2m1_t>(wIn, wOut, m, k);
        }
        else
        {
            TLLM_LOG_ERROR(false, "Dtype not supported for matrix row reorder");
        }

        if (useBlockScaling)
        {
            // TODO Support other types
            reorderRowsForGatedActGemmSf<fmha::E2m1Utils::SfType>(wSfIn, wSfOut, m, k / numEltsPerSf);
        }
        shuffleIn = wOut;
        shuffleSfIn = wSfOut;
    }
    else
    {
        shuffleIn = wIn;
        shuffleSfIn = wSfIn;
    }

    if (useShuffleMatrix)
    {
        if (dtypeElt == tg::Dtype::Fp16)
        {
            shuffleMatrixA<cutlass::half_t>(shuffleIn, wOut, m, k, epilogueTileM);
        }
        else if (dtypeElt == tg::Dtype::Bfloat16)
        {
            shuffleMatrixA<cutlass::bfloat16_t>(shuffleIn, wOut, m, k, epilogueTileM);
        }
        else if (dtypeElt == tg::Dtype::E4m3)
        {
            shuffleMatrixA<cutlass::float_e4m3_t>(shuffleIn, wOut, m, k, epilogueTileM);
        }
        else if (dtypeElt == tg::Dtype::E2m1)
        {
            shuffleMatrixA<cutlass::float_e2m1_t>(shuffleIn, wOut, m, k, epilogueTileM);
        }
        else
        {
            TLLM_LOG_ERROR("Dtype not supported for matrix shuffle");
        }

        if (useBlockScaling)
        {
            shuffleMatrixSfA<fmha::E2m1Utils::SfType>(shuffleSfIn, wSfOut, m, k, epilogueTileM, numEltsPerSf);
        }
    }

    //
    // Copy the weights for consistent functionality if neither operation is done
    // //
    if (!useFusedAct && !useShuffleMatrix)
    {
        if (wIn != wOut)
        {
            int64_t const numBitsPerElt = tg::dtypeGetNumBits(dtypeElt);
            std::memcpy(wOut, wIn, m * k * numBitsPerElt / 8 /* bits */);
        }
        if (wSfIn != wSfOut)
        {

            int64_t const numSfBytes = useBlockScaling
                ? tg::dtypeGetNumBits(tg::dtypeBlockSfType(dtypeElt)) / 8 /* bits */
                : 0;

            std::memcpy(wOut, wIn, m * k / numEltsPerSf * numSfBytes);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// This function will copy and transform the weights for a set of experts.
//
// This takes in a vector of pointers to the individual weight matrices and will write to a single
// contiguous matrix.
//
// This assumes that the gate matrices are stored along side the activation weights for each expert
//
void prepareBatchWeightsOnHost(std::vector<void const*> wIns, std::vector<void const*> wSfIns, void* wOut, void* wSfOut,
    tg::Dtype dtypeElt, const int32_t m, const int32_t k, const int32_t epilogueTileM, const int32_t numBatches,
    bool const useShuffleMatrix, bool const useFusedAct, bool const useBlockScaling, int const numEltsPerSf)
{
    int64_t const numBitsPerElt = tg::dtypeGetNumBits(dtypeElt);
    int64_t const numSfBytes = useBlockScaling ? tg::dtypeGetNumBits(tg::dtypeBlockSfType(dtypeElt)) / 8 /* bits */
                                               : 0;

    for (int32_t ii = 0; ii < numBatches; ++ii)
    {
        const size_t weightsOffset = ii * m * k * numBitsPerElt / 8 /* bits */;
        const size_t weightsSfOffset = ii * m * k / numEltsPerSf * numSfBytes;
        prepareWeightsOnHost(wIns[ii], wSfIns[ii], (uint8_t*) wOut + weightsOffset, (uint8_t*) wSfOut + weightsSfOffset,
            dtypeElt, m, k, epilogueTileM, useShuffleMatrix, useFusedAct, useBlockScaling, numEltsPerSf);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function will copy and transform the weights for a set of experts.
//
// This function supports in-place operations
//
// This assumes that the gate matrices are stored along side the activation weights for each expert
//
void prepareBatchWeightsOnHost(void const* wIn, void const* wSfIn, void* wOut, void* wSfOut, tg::Dtype dtypeElt,
    const int32_t m, const int32_t k, const int32_t epilogueTileM, const int32_t numBatches,
    bool const useShuffleMatrix, bool const useFusedAct, bool const useBlockScaling, int const numEltsPerSf)
{
    std::vector<void const*> wIns;
    std::vector<void const*> wSfIns;
    int64_t const numBitsPerElt = tg::dtypeGetNumBits(dtypeElt);
    int64_t const numSfBytes = useBlockScaling ? tg::dtypeGetNumBits(tg::dtypeBlockSfType(dtypeElt)) / 8 /* bits */
                                               : 0;
    // Compute offsets into contiguous input buffer
    for (int32_t ii = 0; ii < numBatches; ++ii)
    {
        const size_t weightsOffset = ii * m * k * numBitsPerElt / 8 /* bits */;
        const size_t weightsSfOffset = ii * m * k / numEltsPerSf * numSfBytes;
        wIns.push_back((uint8_t*) wIn + weightsOffset);
        wSfIns.push_back((uint8_t*) wSfIn + weightsSfOffset);
    }

    prepareBatchWeightsOnHost(wIns, wSfIns, wOut, wSfOut, dtypeElt, m, k, epilogueTileM, numBatches, useShuffleMatrix,
        useFusedAct, useBlockScaling, numEltsPerSf);
}
