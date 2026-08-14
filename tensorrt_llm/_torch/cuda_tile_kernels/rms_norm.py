# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/NVIDIA/cutile-python/blob/main/test/kernels/rms_norm.py
from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE

if IS_CUDA_TILE_AVAILABLE:
    import cuda.tile as ct

    @ct.kernel
    def rms_norm_kernel(
        x,
        w,
        out,
        Rstd,
        N: ct.Constant[int],
        eps: ct.Constant[float],
        TILE_SIZE: ct.Constant[int],
        use_gemma: ct.Constant[bool],
    ):
        """Standard RMSNorm kernel for non-static persistent mode with tiled loads"""
        row = ct.bid(0)
        _rms = ct.full((1, TILE_SIZE), 0.0, dtype=ct.float32)
        num_tiles = ct.cdiv(x.shape[1], TILE_SIZE)

        for j in range(0, num_tiles):
            xj = ct.load(
                x,
                index=(row, j),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=1,
            )
            xj = ct.astype(xj, ct.float32)
            _rms += xj * xj

        # Calculate RMS Norm
        rms = ct.rsqrt(ct.sum(_rms, axis=1, keepdims=False) / N + eps)
        ct.store(Rstd, index=(row,), tile=rms)

        for j in range(0, num_tiles):
            wj = ct.load(
                w,
                index=(j,),
                shape=(TILE_SIZE,),
                allow_tma=False,
                latency=1,
            )
            wj = ct.astype(wj, ct.float32)
            # Apply Gemma-style bias if enabled
            if use_gemma:
                wj = wj + 1.0
            xj = ct.load(
                x,
                index=(row, j),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=1,
            )
            xj = ct.astype(xj, ct.float32)
            yj = xj * rms * wj
            yj = ct.astype(yj, x.dtype)
            ct.store(
                out,
                index=(row, j),
                tile=yj,
                allow_tma=False,
                latency=1,
            )

    @ct.kernel
    def rms_norm_kernel_gather(
        x,
        w,
        out,
        Rstd,
        N: ct.Constant[int],
        eps: ct.Constant[float],
        TILE_SIZE: ct.Constant[int],
        use_gemma: ct.Constant[bool],
    ):
        """Standard RMSNorm kernel for non-static persistent mode with ptr loads"""
        row = ct.bid(0)
        _rms = ct.full((TILE_SIZE,), 0.0, dtype=ct.float32)
        num_tiles = ct.cdiv(N, TILE_SIZE)
        offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

        for j in range(0, num_tiles):
            offs = j * TILE_SIZE + offsets
            xj = ct.gather(x, (row, offs), latency=1)
            xj = ct.astype(xj, ct.float32)
            _rms += xj * xj

        # Calculate RMS Norm
        rms = ct.rsqrt(ct.sum(_rms, axis=0, keepdims=False) / N + eps)
        ct.scatter(Rstd, row, rms)

        for j in range(0, num_tiles):
            offs = j * TILE_SIZE + offsets
            wj = ct.gather(w, offs, latency=1)
            wj = ct.astype(wj, ct.float32)
            # Apply Gemma-style bias if enabled
            if use_gemma:
                wj = wj + 1.0
            xj = ct.gather(x, (row, offs), latency=1)
            xj = ct.astype(xj, ct.float32)
            yj = xj * rms * wj
            yj = ct.astype(yj, x.dtype)
            ct.scatter(out, (row, offs), yj, latency=1)

    @ct.kernel
    def rms_norm_kernel_static_persistent(
        X,  # Input tensor
        Y,  # Output tensor
        W,  # Weight tensor
        TILE_SIZE_M: ct.Constant[int],  # 4 rows per block
        TILE_SIZE_N: ct.Constant[int],  # columns per block
        eps: ct.Constant[float],  # Epsilon value
        use_gemma: ct.Constant[bool],  # Gemma-style weight bias
    ):
        """
        CuTile static persistent RMSNorm kernel that processes multiple blocks per program.
        Each program processes multiple blocks in a loop for better efficiency.
        """
        # Get program ID
        pid = ct.bid(0)

        # Infer tensor dimensions from input shape
        M = X.shape[0]  # Number of rows
        N = X.shape[1]  # Number of columns

        # Calculate upper bound - number of row blocks to process
        upper_bound = (M + TILE_SIZE_M - 1) // TILE_SIZE_M

        # Load weight vector once (shared across all blocks processed by this program)
        w = ct.load(W, index=(0,), shape=(TILE_SIZE_N,))
        w = ct.astype(w, ct.float32)
        # Apply Gemma-style bias if enabled
        if use_gemma:
            w = w + 1.0

        # Static persistent loop: each program processes multiple blocks
        num_tiles_x = ct.num_blocks(0)
        for current_block in range(pid, upper_bound, num_tiles_x):
            # Load input tile
            x = ct.load(
                X,
                index=(current_block, 0),
                shape=(TILE_SIZE_M, TILE_SIZE_N),
                latency=10,  # +2% perf from this hint
            )
            x = ct.astype(x, ct.float32)

            # Step 1: Compute x^2
            x_squared = ct.mul(x, x)

            # Step 2: Reduce sum along axis=1 (columns)
            x2_sum = ct.sum(x_squared, axis=1, keepdims=True)  # Shape: [TILE_SIZE_M, 1]

            # Step 3: Compute variance (divide by N)
            N_f32 = ct.full((TILE_SIZE_M, 1), N * 1.0, dtype=ct.float32)
            variance = ct.truediv(x2_sum, N_f32)

            # Step 4: Add epsilon and compute rsqrt
            eps_tensor = ct.full((TILE_SIZE_M, 1), eps, dtype=ct.float32)
            variance_eps = ct.add(variance, eps_tensor)
            rsqrt_var = ct.rsqrt(variance_eps)

            # Step 5: Apply normalization
            x_normalized = ct.mul(x, rsqrt_var)

            # Step 6: Apply linear transformation
            # Broadcast weight to match input shape
            w_broadcasted = ct.reshape(w, (1, TILE_SIZE_N))
            b_broadcasted = ct.full((1, TILE_SIZE_N), 0.0, dtype=ct.float32)

            # Apply linear transformation: y = x_normalized * w + b
            y = ct.mul(x_normalized, w_broadcasted)
            y = ct.add(y, b_broadcasted)

            # Convert back to original dtype
            y = ct.astype(y, X.dtype)

            # Store result
            ct.store(
                Y,
                index=(current_block, 0),
                tile=y,
                allow_tma=False,  # +30% perf
                latency=3,  # +3% perf from this hint
            )
