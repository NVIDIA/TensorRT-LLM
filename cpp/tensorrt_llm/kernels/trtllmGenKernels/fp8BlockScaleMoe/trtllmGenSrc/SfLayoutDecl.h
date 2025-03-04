/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Be careful when modifying this file as it is included by the generated kernels. For example, do
// not add TLLM_CHECK_* constructs in this file. Thanks!
//
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace trtllm
{
namespace gen
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// This enumeration defines layouts for storing scale factors for FP4, FP6, and FP8 formats.
enum class SfLayout
{
    // Scale factors are stored in the same order as the associated matrix.
    // I.e., the SF buffer is a tensor [m, ⌈n/b⌉], where m, n, and b are respectively the number of
    // rows, columns and the block size.
    // The SF for the element (i, j) is stored at (i, j/b).
    Linear = 0,

    // A tile of 8x4 is stored contiguously. The order of elements inside the tile, and the order
    // of tiles, are both row-major.
    // I.e., the SF buffer is a tensor [⌈m/8⌉, ⌈n/b/4⌉, 8, 4].
    // The SF for the element (i, j) is stored at (i/8, j/b/4, i%8, (j/b)%4).
    R8c4,

    // A tile of 8x16 is stored contiguously. The order of elements inside the tile, and the order
    // of tiles, are both row-major.
    // I.e., the SF buffer is a tensor [⌈m/8⌉, ⌈n/b/16⌉, 8, 16].
    // The SF for the element (i, j) is stored at (i/8, j/b/16, i%8, (j/b)%16).
    //
    // NOTE: This is a niche format that is currently used for the weights of the
    // LowLatency FP4 kernels. It is not meant as an interchange format. In
    // addition to the above requirements it requires n to be a multiple of 256.
    R8c16,

    // A tile of 128x4 is stored contiguously. Rows 0-31, 32-63, 64-95 and 96-127 are interleaved
    // as illustrated below:
    // |  0,0 |  0,1 |  0,2 |  0,3 | 32,0 | 32,1 | 32,2 | 32,3 | ... |  96,3 |
    // |  1,0 |  1,1 |  1,2 |  1,3 | 33,0 | 33,1 | 33,2 | 33,3 | ... |  97,3 |
    // |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... | ... |   ... |
    // | 31,0 | 31,1 | 31,2 | 31,3 | 63,0 | 63,1 | 63,2 | 63,3 | ... | 127,3 |
    // See https://nvbugspro.nvidia.com/bug/4165523
    //
    // I.e., the SF buffer is a tensor [⌈m/128⌉, ⌈n/b/4⌉, 32, 4, 4]
    // The SF for the element (i, j) is stored at (i/128, j/b/4, i%32, (i%128)/32, (j/b)%4).
    R128c4,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm
