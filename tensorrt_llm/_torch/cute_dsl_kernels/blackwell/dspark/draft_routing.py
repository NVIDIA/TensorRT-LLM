# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 DSpark draft MoE routing: FP32 sqrt-softplus scores, then top-k or hashed experts."""

import functools

import cutlass
import cutlass.cute as cute
import torch

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

_TORCH_TO_CUTE = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


class DSparkDraftRoutingKernel:
    """One warp per token; each lane owns ``num_experts // 32`` consecutive experts."""

    warps_per_cta = 4

    def __init__(self, num_experts: int, top_k: int, routed_scaling_factor: float, hashed: bool):
        if num_experts % cute.arch.WARP_SIZE != 0 or top_k != 3:
            raise ValueError(
                f"expected top-3 over a multiple of 32 experts; got {top_k}, {num_experts}"
            )
        self.num_experts = num_experts
        self.experts_per_lane = num_experts // cute.arch.WARP_SIZE
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.hashed = hashed

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        input_ids: cute.Tensor,
        tid2eid: cute.Tensor,
        indices: cute.Tensor,
        weights: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(logits, bias, input_ids, tid2eid, indices, weights).launch(
            grid=[(logits.shape[0] + self.warps_per_cta - 1) // self.warps_per_cta, 1, 1],
            block=[cute.arch.WARP_SIZE * self.warps_per_cta, 1, 1],
            stream=stream,
        )

    @cute.jit
    def score(self, logit: cutlass.Float32) -> cutlass.Float32:
        # F.softplus (beta 1, threshold 20) followed by a correctly rounded sqrt.
        softplus = logit
        if logit <= 20.0:
            softplus = cute.math.log1p(cute.math.exp(logit))
        return cute.math.sqrt(softplus, rounding=cute.math.RoundingMode.NEAREST_EVEN)

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        input_ids: cute.Tensor,
        tid2eid: cute.Tensor,
        indices: cute.Tensor,
        weights: cute.Tensor,
    ):
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        lane = thread % cute.arch.WARP_SIZE
        row = cta * self.warps_per_cta + thread // cute.arch.WARP_SIZE
        if row < logits.shape[0]:
            scores = []
            for v in cutlass.range_constexpr(self.experts_per_lane):
                scores.append(
                    self.score(cutlass.Float32(logits[row, lane * self.experts_per_lane + v]))
                )
            picked = []
            if cutlass.const_expr(self.hashed):
                token = input_ids[row]
                for k in cutlass.range_constexpr(self.top_k):
                    picked.append(cutlass.Int32(tid2eid[token, k]))
            else:
                keys = []
                for v in cutlass.range_constexpr(self.experts_per_lane):
                    key = scores[v] + bias[lane * self.experts_per_lane + v]
                    # Like torch.topk, NaN ranks first, so every pick is a valid expert.
                    if key != key:
                        key = cutlass.Float32(float("inf"))
                    keys.append(key)
                for k in cutlass.range_constexpr(self.top_k):
                    # Highest key, lowest expert among ties: first within the lane,
                    # then across lanes.
                    best = keys[0]
                    best_expert = lane * self.experts_per_lane
                    for v in cutlass.range_constexpr(1, self.experts_per_lane):
                        if keys[v] > best:
                            best = keys[v]
                            best_expert = lane * self.experts_per_lane + v
                    for offset in (16, 8, 4, 2, 1):
                        other = cute.arch.shuffle_sync_bfly(best, offset=offset)
                        other_expert = cute.arch.shuffle_sync_bfly(best_expert, offset=offset)
                        if (other > best) | ((other == best) & (other_expert < best_expert)):
                            best = other
                            best_expert = other_expert
                    picked.append(best_expert)
                    for v in cutlass.range_constexpr(self.experts_per_lane):
                        if lane * self.experts_per_lane + v == best_expert:
                            keys[v] = cutlass.Float32(float("-inf"))
            # The mixture weights come from the unbiased scores of the picked experts.
            weights_k = []
            for k in cutlass.range_constexpr(self.top_k):
                local = scores[0]
                for v in cutlass.range_constexpr(1, self.experts_per_lane):
                    if picked[k] % self.experts_per_lane == v:
                        local = scores[v]
                weights_k.append(
                    cute.arch.shuffle_sync(local, offset=picked[k] // self.experts_per_lane)
                )
            if lane == 0:
                # torch.sum adds three values as (w0 + w2) + w1.
                total = weights_k[0] + weights_k[2] + weights_k[1] + 1e-20
                for k in cutlass.range_constexpr(self.top_k):
                    indices[row, k] = picked[k]
                    weights[row, k] = (
                        cute.math.div(
                            weights_k[k], total, rounding=cute.math.RoundingMode.NEAREST_EVEN
                        )
                        * self.routed_scaling_factor
                    )


@functools.cache
def _compile(
    num_experts: int,
    top_k: int,
    routed_scaling_factor: float,
    logits_dtype: torch.dtype,
    ids_dtype: torch.dtype | None,
    table_dtype: torch.dtype | None,
):
    hashed = ids_dtype is not None
    rows = cute.sym_int()
    logits = cute.runtime.make_fake_compact_tensor(
        _TORCH_TO_CUTE[logits_dtype], (rows, num_experts), stride_order=(1, 0)
    )
    bias = input_ids = tid2eid = None
    if hashed:
        input_ids = cute.runtime.make_fake_compact_tensor(
            _TORCH_TO_CUTE[ids_dtype], (rows,), stride_order=(0,)
        )
        tid2eid = cute.runtime.make_fake_compact_tensor(
            _TORCH_TO_CUTE[table_dtype], (cute.sym_int(), top_k), stride_order=(1, 0)
        )
    else:
        bias = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (num_experts,), stride_order=(0,)
        )
    indices = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (rows, top_k), stride_order=(1, 0)
    )
    weights = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (rows, top_k), stride_order=(1, 0)
    )
    kernel = DSparkDraftRoutingKernel(num_experts, top_k, routed_scaling_factor, hashed)
    return cute.compile(
        kernel,
        logits,
        bias,
        input_ids,
        tid2eid,
        indices,
        weights,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def dspark_draft_routing(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    tid2eid: torch.Tensor | None,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route ``[tokens, experts]`` logits; ``bias`` is None exactly when the gate is hashed."""
    num_tokens, num_experts = logits.shape
    indices = torch.empty(num_tokens, top_k, dtype=torch.int32, device=logits.device)
    weights = torch.empty(num_tokens, top_k, dtype=torch.float32, device=logits.device)
    if num_tokens == 0:
        return indices, weights
    hashed = bias is None
    compiled = _compile(
        num_experts,
        top_k,
        routed_scaling_factor,
        logits.dtype,
        input_ids.dtype if hashed else None,
        tid2eid.dtype if hashed else None,
    )
    compiled(logits, bias, input_ids, tid2eid, indices, weights)
    return indices, weights
