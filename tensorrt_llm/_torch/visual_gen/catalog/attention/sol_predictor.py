# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import numbers

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

SOL_BLOCK_SIZE = 64


@torch.library.custom_op(
    "trtllm::visual_gen_v2_sol_predictor", mutates_args=(), device_types="cuda"
)
def _predict(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tau: float, sm_scale: float, thresh_type: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from ._sol_predictor_kernels import sol_predictor_kernel

    return sol_predictor_kernel(q, k, v, SOL_BLOCK_SIZE, tau, sm_scale, thresh_type)


@_predict.register_fake
def _fake(q, k, v, tau, sm_scale, thresh_type):
    batch, tokens, heads, dim = q.shape
    blocks = (tokens + SOL_BLOCK_SIZE - 1) // SOL_BLOCK_SIZE
    shape = (batch, blocks, heads, dim)
    return (
        q.new_empty((batch, heads, blocks, (blocks + 31) // 32), dtype=torch.uint32),
        k.new_empty(shape),
        v.new_empty(shape),
    )


class SolPredictor(OpWrapper):
    """Return packed exact-block routes and BF16 K/V means over 64-token BSHD blocks.

    Bit j % 32 of word j // 32 selects key block j. A query always selects its
    neighbors within one block. The threshold uses diagonal or full key covariance.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "compiled diagonal threshold across two mask words",
            dict(tokens=2081, thresh_type="diag", compiled=True),
        ),
        Cell(
            "full covariance with a partial final block",
            dict(tokens=321, thresh_type="exact", compiled=False),
        ),
    )

    def __call__(self, q, k, v, *, tau: float, sm_scale: float, thresh_type: str = "diag"):
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or min(q.shape) <= 0:
            raise ValueError("SOL predictor requires equal, nonempty BSHD Q/K/V tensors")
        if q.shape[-1] != 128 or any(x.dtype != torch.bfloat16 for x in (q, k, v)):
            raise ValueError("SOL predictor requires BF16 Q/K/V with head dimension 128")
        if not q.is_cuda or any(x.device != q.device or not x.is_contiguous() for x in (q, k, v)):
            raise ValueError("SOL predictor requires contiguous Q/K/V on one CUDA device")
        if thresh_type not in ("diag", "exact"):
            raise ValueError("SOL predictor threshold type must be 'diag' or 'exact'")
        for name, value in (("tau", tau), ("sm_scale", sm_scale)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a finite Python real")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if sm_scale <= 0:
            raise ValueError("sm_scale must be positive")
        return _predict(q, k, v, float(tau), float(sm_scale), thresh_type)

    def reference(self, q, k, v, *, tau: float, sm_scale: float, thresh_type: str = "diag"):
        batch, tokens, heads, dim = q.shape
        blocks = (tokens + SOL_BLOCK_SIZE - 1) // SOL_BLOCK_SIZE
        indices = torch.arange(blocks, device=q.device)
        lengths = (tokens - SOL_BLOCK_SIZE * indices).clamp(max=SOL_BLOCK_SIZE)

        def block_means(x):
            padded = F.pad(x, (0, 0, 0, 0, 0, blocks * SOL_BLOCK_SIZE - tokens))
            sums = padded.reshape(batch, blocks, SOL_BLOCK_SIZE, heads, dim).sum(
                2, dtype=torch.float64
            )
            return sums / lengths[None, :, None, None]

        centers = block_means(q).transpose(1, 2)
        keys = block_means(k).to(k.dtype)
        values = block_means(v).to(v.dtype)
        key_means = keys.double().transpose(1, 2)
        scale = sm_scale * math.log2(math.e)
        scores = (centers @ key_means.transpose(-1, -2)) * scale
        if thresh_type == "exact":
            variance = scores.var(dim=-1, correction=0)
        else:
            variance = centers.square() @ key_means.var(dim=-2, correction=0).unsqueeze(-1)
            variance = variance.squeeze(-1) * scale**2
        thresholds = scores.mean(-1) + tau * torch.sqrt(variance + 1.0e-6)
        select = (scores > thresholds[..., None]) | ((indices[:, None] - indices).abs() <= 1)
        words = (blocks + 31) // 32
        selected = F.pad(select, (0, words * 32 - blocks)).reshape(batch, heads, blocks, words, 32)
        bits = (selected.long() << torch.arange(32, device=q.device)).sum(-1)
        return bits.to(torch.uint32), keys, values


sol_predictor = SolPredictor()
