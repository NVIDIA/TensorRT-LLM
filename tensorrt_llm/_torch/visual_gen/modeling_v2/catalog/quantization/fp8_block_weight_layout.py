# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._fp8_layout import get_col_major_tma_aligned_packed_tensor, resmooth_to_fp8_e8m0


class Fp8BlockWeightLayout(OpWrapper):
    """Update colocated CUDA weights and block scales in place and pack scales into MN-major words."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (Cell(why="An odd number of scale blocks", spec=dict(n=256, k=384)),)

    def __call__(self, weight, scales):
        if not weight.is_cuda or weight.device != scales.device:
            raise ValueError("FP8 weights and block scales must be on the same CUDA device")
        with torch.cuda.device(weight.device):
            resmooth_to_fp8_e8m0(weight, scales)
        rows = torch.arange(weight.shape[0], device=scales.device) // 128
        return weight, get_col_major_tma_aligned_packed_tensor(scales.index_select(0, rows))

    def reference(self, weight, scales):
        n, k = weight.shape
        values = weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
        tiles = torch.nn.functional.pad(values, (0, -k % 128, 0, -n % 128)).reshape(
            scales.size(0), 128, scales.size(1), 128
        )
        amax = tiles.abs().amax(dim=(1, 3)).clamp_min(1e-4)
        powers = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        result = (values / powers.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]).to(
            weight.dtype
        )
        weight.copy_(result)
        scales.copy_(powers)
        scale_bytes = (powers.contiguous().view(torch.int32) >> 23).to(torch.uint8)
        unpacked = scale_bytes.repeat_interleave(128, 0)[:n]
        padded = torch.nn.functional.pad(unpacked, (0, -unpacked.shape[1] % 4, 0, -n % 4))
        words = padded.contiguous().view(torch.int32)
        layout = torch.empty_strided(
            words.shape, (1, words.size(0)), dtype=torch.int32, device=weight.device
        )
        layout.copy_(words)
        return weight, layout[:n]

    def is_valid(self, weight, scales):
        n, k = weight.shape
        assert weight.is_cuda and scales.device == weight.device
        assert weight.dtype == torch.float8_e4m3fn
        assert scales.shape == ((n + 127) // 128, (k + 127) // 128)
        assert scales.dtype == torch.float32


fp8_block_weight_layout = Fp8BlockWeightLayout()
