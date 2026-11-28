# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._cudnn_sdpa import CudnnSdpaRunner
from ._key_padding import KeyPadding

__all__ = ["CudnnSdpaRunner", "cudnn_sdpa"]


class CudnnSdpa(OpWrapper):
    """Noncausal self-attention with optional per-tensor FP8 or block-scaled MXFP8 operands."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Batched BF16 self-attention", dict(quant=None, rows=32)),
        Cell("Per-tensor FP8 attention", dict(quant="fp8", rows=256)),
        Cell("Block-scaled MXFP8 attention", dict(quant="mxfp8", rows=200)),
    )

    def key_padding(self) -> KeyPadding:
        return KeyPadding.NONE

    def __call__(self, q, k, v, *, runner: CudnnSdpaRunner):
        return runner.run_dense(q, k, v)

    def reference(self, q, k, v, *, runner):
        inputs = [x.to(runner.dtype).transpose(1, 2).double() for x in (q, k, v)]
        out = torch.nn.functional.scaled_dot_product_attention(*inputs, scale=runner.scale)
        return out.transpose(1, 2).to(runner.dtype)

    def is_valid(self, q, k, v, *, runner):
        assert q.ndim == 4 and q.shape == k.shape == v.shape
        assert q.device == k.device == v.device
        assert q.shape[2:] == (runner.num_heads, runner.head_dim)


cudnn_sdpa = CudnnSdpa()
