#include "Dtype.h"
#include "SfLayoutDecl.h"
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace fmha
{

struct E2m1Utils
{
    // The dtype for elts.
    using EltType = cutlass::float_e2m1_t;
    // The dtype for sf.
    using SfType = cutlass::float_ue4m3_t;

    // The number of elements per byte.
    static int32_t constexpr NumEltsPerByte = 2;
    // The max value E2m1 can represent.
    static float constexpr MaxE2m1Val = 6.0f;
    // The max value E4M3 can represent.
    // TODO(tizheng): double check if UE4M3_MAX or E4M3_MAX is used here.
    static float constexpr MaxE4m3Val = 448.0f;
    // Each SF corresponds to 16 elements.
    static int32_t constexpr NumEltsPerSf = 16;

    struct Layout128x4
    {
        // Scaling factors are stored as 512B blocks in GMEM. Each block contains 128x4 FP8 elements.
        // The rows of the SF block map to interleaved rows of the data block. The row ii in the SF
        // block corresponds to the row (ii % 4) * 32 + (ii / 4) in the data block. The col jj is the
        // scaling factor of the block jj in the data tensor.
        //
        // See https://nvbugspro.nvidia.com/bug/4165523 for details.

        // The number of rows of SF per block.
        static int32_t constexpr NumRowsPerSfBlock = 128;
        // The number of cols of SF per block.
        static int32_t constexpr NumColsPerSfBlock = 4;
        // The size of each SF block.
        static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;

        // Each block of scaling factors uses 512B.
        static_assert(NumBytesPerSfBlock == 512);

        // The number of rows of data per SF block.
        static int32_t constexpr NumDataRowsPerSfBlock = NumRowsPerSfBlock;
        // The number of cols of blocks of data per SF block.
        static int32_t constexpr NumDataBlkColsPerSfBlock = NumColsPerSfBlock;
        // The number of cols of data elements per SF block.
        static int32_t constexpr NumDataColsPerSfBlock = NumDataBlkColsPerSfBlock * NumEltsPerSf;

        // Make sure the size of the block of data that corresponds to a SF block has 128x64 elements.
        static_assert(NumDataRowsPerSfBlock == 128 && NumDataColsPerSfBlock == 64);
    };

    struct Layout8x4
    {
        // Scaling factors are stored as 32B blocks in GMEM. Each block contains 8x4 FP8 elements.

        // The number of rows of SF per block.
        static int32_t constexpr NumRowsPerSfBlock = 8;
        // The number of cols of SF per block.
        static int32_t constexpr NumColsPerSfBlock = 4;
        // The size of each SF block.
        static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;

        // Each block of scaling factors uses 32B.
        static_assert(NumBytesPerSfBlock == 32);

        // The number of rows of data per SF block.
        static int32_t constexpr NumDataRowsPerSfBlock = NumRowsPerSfBlock;
        // The number of cols of blocks of data per SF block.
        static int32_t constexpr NumDataBlkColsPerSfBlock = NumColsPerSfBlock;
        // The number of cols of data elements per SF block.
        static int32_t constexpr NumDataColsPerSfBlock = NumDataBlkColsPerSfBlock * NumEltsPerSf;

        // Make sure the size of the block of data that corresponds to a SF block has 8x64 elements.
        static_assert(NumDataRowsPerSfBlock == 8 && NumDataColsPerSfBlock == 64);
    };

    struct Layout8x16
    {
        // Scaling factors are stored as 128B blocks in GMEM. Each block contains 8x16 FP8 elements.

        // The number of rows of SF per block.
        static int32_t constexpr NumRowsPerSfBlock = 8;
        // The number of cols of SF per block.
        static int32_t constexpr NumColsPerSfBlock = 16;
        // The size of each SF block.
        static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;

        // Each block of scaling factors uses 128B.
        static_assert(NumBytesPerSfBlock == 128);

        // The number of rows of data per SF block.
        static int32_t constexpr NumDataRowsPerSfBlock = NumRowsPerSfBlock;
        // The number of cols of blocks of data per SF block.
        static int32_t constexpr NumDataBlkColsPerSfBlock = NumColsPerSfBlock;
        // The number of cols of data elements per SF block.
        static int32_t constexpr NumDataColsPerSfBlock = NumDataBlkColsPerSfBlock * NumEltsPerSf;

        // Make sure the size of the block of data that corresponds to a SF block has 8x256 elements.
        static_assert(NumDataRowsPerSfBlock == 8 && NumDataColsPerSfBlock == 256);
    };

    // Compute the offset that corresponds to (dataRowIdx, dataBlkColIdx) in the SF tensor where
    // dataRowIdx and dataBlkColIdx are the respective indices of the row and the block of 16 elts
    // from the K dim in the tensor of data.
    static int32_t getSfOffset(
        int32_t dataRowIdx, int32_t dataBlkColIdx, int32_t numDataBlksPerRow, trtllm::gen::SfLayout layout)
    {

        switch (layout)
        {
        case trtllm::gen::SfLayout::Linear:
            // Plain row-major layout.
            return dataRowIdx * numDataBlksPerRow + dataBlkColIdx;

        case trtllm::gen::SfLayout::R128c4:
        {
            // The row of the SF block in the SF tensor.
            int sfBlkRowIdx = dataRowIdx / Layout128x4::NumDataRowsPerSfBlock;
            // The col of the SF block in the SF tensor.
            int sfBlkColIdx = dataBlkColIdx / Layout128x4::NumDataBlkColsPerSfBlock;
            // The blocks are stored row-major in the tensor of scaling factors.
            int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / Layout128x4::NumDataBlkColsPerSfBlock + sfBlkColIdx;
            // Find the row in the SF block.
            int sfRowIdx = (dataRowIdx % 32) * 4 + (dataRowIdx % Layout128x4::NumRowsPerSfBlock) / 32;
            // Make sure the row index is between 0 and the number of rows in the block.
            TLLM_CHECK_ERROR(sfRowIdx >= 0 && sfRowIdx < Layout128x4::NumRowsPerSfBlock, "Invalid sfRowIdx ", sfRowIdx);
            // Find the col in the SF block.
            int sfColIdx = (dataBlkColIdx % 4);
            // Compute the offset in bytes.
            return sfBlkIdx * Layout128x4::NumBytesPerSfBlock + sfRowIdx * Layout128x4::NumColsPerSfBlock + sfColIdx;
        }

        case trtllm::gen::SfLayout::R8c4:
        {
            // The row of the SF block in the SF tensor.
            int sfBlkRowIdx = dataRowIdx / Layout8x4::NumDataRowsPerSfBlock;
            // The col of the SF block in the SF tensor.
            int sfBlkColIdx = dataBlkColIdx / Layout8x4::NumDataBlkColsPerSfBlock;
            // The blocks are stored row-major in the tensor of scaling factors.
            int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / Layout8x4::NumDataBlkColsPerSfBlock + sfBlkColIdx;
            // Find the row in the SF block.
            int sfRowIdx = dataRowIdx % Layout8x4::NumRowsPerSfBlock;
            // Find the col in the SF block.
            int sfColIdx = dataBlkColIdx % Layout8x4::NumDataBlkColsPerSfBlock;
            // Compute the offset in bytes.
            return sfBlkIdx * Layout8x4::NumBytesPerSfBlock + sfRowIdx * Layout8x4::NumColsPerSfBlock + sfColIdx;
        }

        case trtllm::gen::SfLayout::R8c16:
        {
            // The row of the SF block in the SF tensor.
            int sfBlkRowIdx = dataRowIdx / Layout8x16::NumDataRowsPerSfBlock;
            // The col of the SF block in the SF tensor.
            int sfBlkColIdx = dataBlkColIdx / Layout8x16::NumDataBlkColsPerSfBlock;
            // The blocks are stored row-major in the tensor of scaling factors.
            int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / Layout8x16::NumDataBlkColsPerSfBlock + sfBlkColIdx;
            // Find the row in the SF block.
            int sfRowIdx = dataRowIdx % Layout8x16::NumRowsPerSfBlock;
            // Find the col in the SF block.
            int sfColIdx = dataBlkColIdx % Layout8x16::NumDataBlkColsPerSfBlock;
            // Compute the offset in bytes.
            return sfBlkIdx * Layout8x16::NumBytesPerSfBlock + sfRowIdx * Layout8x16::NumColsPerSfBlock + sfColIdx;
        }

        default: TLLM_CHECK_ERROR(false, "Unsupported layout: ", sfLayoutToString(layout)); return 0;
        }
    }
};
} // namespace fmha

// clang-format off
inline constexpr std::array<int, 16> srcToDstBlk16RowMap =
      {
        0,  8,
        1,  9,
        2, 10,
        3, 11,
        4, 12,
        5, 13,
        6, 14,
        7, 15
      };
inline constexpr std::array<int, 32> srcToDstBlk32RowMap =
      {
        0,  8, 16, 24,
        1,  9, 17, 25,
        2, 10, 18, 26,
        3, 11, 19, 27,
        4, 12, 20, 28,
        5, 13, 21, 29,
        6, 14, 22, 30,
        7, 15, 23, 31
      };

// clang-format on

template <typename T>
void reorderRowsForGatedActGemm(void const* input, void* output, int M, int K)
{
    if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>)
    {
        K /= 2;
    }

    std::vector<T> outData(M * K);
    // Reorders rows
    // [r0, r1, r2, r3, ..., rN/2, r(N/2+1), .. r(N-1)]
    // to
    // [r0, rN/2, r1, rN/2+1, ..., r(N/2-1), r(N-1)]
    for (int mi = 0; mi < M / 2; ++mi)
    {
        for (int ki = 0; ki < K; ++ki)
        {
            int const srcIdx1 = mi * K + ki;
            int const srcIdx2 = (mi + M / 2) * K + ki;
            int const dstIdx1 = (2 * mi + 0) * K + ki;
            int const dstIdx2 = (2 * mi + 1) * K + ki;
            outData[dstIdx1] = reinterpret_cast<T const*>(input)[srcIdx1];
            outData[dstIdx2] = reinterpret_cast<T const*>(input)[srcIdx2];
        }
    }

    // Copy tmp data to the output pointer.
    std::memcpy(output, outData.data(), M * K * sizeof(T));
}

inline int32_t getShuffleBlockSize(int epilogueTileM)
{
    int shuffleBlockSize = 16;
    if (epilogueTileM % 128 == 0)
    {
        shuffleBlockSize = 32;
    }
    return shuffleBlockSize;
}

template <typename T>
void shuffleMatrixA(void const* input, void* output, int M, int K, int epilogueTileM)
{
    auto const shuffleBlockSize = getShuffleBlockSize(epilogueTileM);

    int numBytesPerRow{0};
    if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>)
    {
        numBytesPerRow = K / 2;
    }
    else
    {
        numBytesPerRow = K * sizeof(T);
    }

    std::vector<uint8_t> tmp(M * numBytesPerRow);
    for (int mi = 0; mi < M; ++mi)
    {
        int const dstRowBlockIdx = mi / shuffleBlockSize;
        int const srcRowInBlockIdx = mi % shuffleBlockSize;
        int const dstRowInBlockIdx
            = shuffleBlockSize == 16 ? srcToDstBlk16RowMap[srcRowInBlockIdx] : srcToDstBlk32RowMap[srcRowInBlockIdx];
        int const dstRowIdx = dstRowBlockIdx * shuffleBlockSize + dstRowInBlockIdx;
        std::memcpy(&tmp[dstRowIdx * numBytesPerRow], &reinterpret_cast<uint8_t const*>(input)[mi * numBytesPerRow],
            numBytesPerRow);
    }

    // Copy tmp data to the output pointer.
    std::memcpy(output, tmp.data(), M * numBytesPerRow);
}

template <typename T>
inline T divUp(T a, T b)
{
    return (a + b - 1) / b;
}

// Shuffle scaling factors using NvFp4's interleaved format (128x4 tiles)
template <typename T>
void shuffleMatrixSfA(void const* input, void* output, int M, int K, int epilogueTileM, int numEltsPerSf)
{
    auto const shuffleBlockSize = getShuffleBlockSize(epilogueTileM);

    int const sfK = divUp(K, numEltsPerSf);
    int const numSfTilesM = divUp(M, 128);
    int const numSfTilesK = divUp(sfK, 4);
    int const numSfElts = numSfTilesM * numSfTilesK * 512;

    std::vector<T> tmp(numSfElts);
    for (int mi = 0; mi < M; ++mi)
    {
        for (int ki = 0; ki < sfK; ++ki)
        {
            // Note: source and destination are in the same SF tile of 128 rows.
            int const tileOffset = ((mi / 128) * numSfTilesK + ki / 4) * 512;

            int const dstRowBlockIdx = mi / shuffleBlockSize;
            int const srcRowInBlockIdx = mi % shuffleBlockSize;
            int const dstRowInBlockIdx = shuffleBlockSize == 16 ? srcToDstBlk16RowMap[srcRowInBlockIdx]
                                                                : srcToDstBlk32RowMap[srcRowInBlockIdx];
            int const dstRowIdx = dstRowBlockIdx * shuffleBlockSize + dstRowInBlockIdx;
            int const dstIdx = tileOffset + (dstRowIdx % 32) * 16 + ((dstRowIdx % 128) / 32) * 4 + ki % 4;
            int const srcIdx = tileOffset + (mi % 32) * 16 + ((mi % 128) / 32) * 4 + ki % 4;
            tmp[dstIdx] = reinterpret_cast<T const*>(input)[srcIdx];
        }
    }

    // Copy tmp data to the output pointer.
    std::memcpy(output, tmp.data(), numSfElts * sizeof(T));
}

template <typename T>
void reorderRowsForGatedActGemmSf(void const* input, void* output, int M, int K)
{
    std::vector<T> outData(M * K);
    // Reorders rows
    // [r0, r1, r2, r3, ..., rN/2, r(N/2+1), .. r(N-1)]
    // to
    // [r0, rN/2, r1, rN/2+1, ..., r(N/2-1), r(N-1)]
    for (int mi = 0; mi < M / 2; ++mi)
    {
        for (int ki = 0; ki < K; ++ki)
        {
            int const srcIdx1 = fmha::E2m1Utils::getSfOffset(mi, ki, K, trtllm::gen::SfLayout::R128c4);
            int const srcIdx2 = fmha::E2m1Utils::getSfOffset((mi + M / 2), ki, K, trtllm::gen::SfLayout::R128c4);
            int const dstIdx1 = fmha::E2m1Utils::getSfOffset((2 * mi + 0), ki, K, trtllm::gen::SfLayout::R128c4);
            int const dstIdx2 = fmha::E2m1Utils::getSfOffset((2 * mi + 1), ki, K, trtllm::gen::SfLayout::R128c4);
            outData[dstIdx1] = reinterpret_cast<T const*>(input)[srcIdx1];
            outData[dstIdx2] = reinterpret_cast<T const*>(input)[srcIdx2];
        }
    }

    // Copy tmp data to the output pointer.
    std::memcpy(output, outData.data(), M * K * sizeof(T));
}

void prepareBatchWeightsOnHost(void const* wIn, void const* wSfIn, void* wOut, void* wSfOut,
    trtllm::gen::Dtype dtypeElt, const int32_t m, const int32_t k, const int32_t epilogueTileM,
    const int32_t numBatches, bool const useShuffleMatrix, bool const useFusedAct, bool const useBlockScaling,
    int const numEltsPerSf);
