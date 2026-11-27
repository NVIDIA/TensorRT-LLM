# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 DSpark draft MoE routing: FP32 sqrt-softplus scores, then top-k or hashed experts;
optionally in the same launch the MXFP8 quantization of the MoE input."""

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import (
    _abs,
    _from_bits,
    _inline_asm,
    _mul_rn,
    _rcp_approx_ftz,
    _to_e4m3x4,
)

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

MXFP8_BLOCK = 32


@dsl_user_op
def _ue8m0_round_up(value: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Int32:
    """The UE8M0 code of ``value`` rounded toward +inf and saturated to finite."""
    return cutlass.Int32(
        _inline_asm(
            T.i32(),
            [cutlass.Float32(value)],
            "{\n.reg .b16 pair;\ncvt.rp.satfinite.ue8m0x2.f32 pair, $1, $1;\n"
            "cvt.u32.u16 $0, pair;\n}",
            "=r,f",
            loc,
            ip,
        )
    ) & cutlass.Int32(0xFF)


@cute.jit
def _e8m0_to_f32(code: cutlass.Int32) -> cutlass.Float32:
    """``static_cast<float>(__nv_fp8_e8m0)`` of a finite code: 2^(code - 127), code 0 a subnormal."""
    bits = code << cutlass.Int32(23)
    if code == cutlass.Int32(0):
        bits = cutlass.Int32(0x400000)
    return _from_bits(bits)


class DSparkDraftRoutingKernel:
    """One warp per token; each lane owns ``num_experts // 32`` consecutive experts. With
    ``hidden`` the launch's other CTAs also quantize the CTA's tokens to MXFP8 as
    ``mxfp8_quantize(x, False)`` does, a thread per 32-value block."""

    warps_per_cta = 4

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        hashed: bool,
        hidden: int = 0,
    ):
        if num_experts % cute.arch.WARP_SIZE != 0 or top_k != 3 or hidden % MXFP8_BLOCK:
            raise ValueError(
                f"expected top-3 over a multiple of 32 experts and hidden divisible by "
                f"{MXFP8_BLOCK}; got {top_k}, {num_experts}, {hidden}"
            )
        self.num_experts = num_experts
        self.experts_per_lane = num_experts // cute.arch.WARP_SIZE
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.hashed = hashed
        self.threads = cute.arch.WARP_SIZE * self.warps_per_cta
        self.blocks = hidden // MXFP8_BLOCK
        self.quant_ctas = (self.warps_per_cta * self.blocks + self.threads - 1) // self.threads

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        input_ids: cute.Tensor,
        tid2eid: cute.Tensor,
        indices: cute.Tensor,
        weights: cute.Tensor,
        x: Optional[cute.Tensor],
        x_fp8: Optional[cute.Tensor],
        x_sf: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        self.kernel(logits, bias, input_ids, tid2eid, indices, weights, x, x_fp8, x_sf).launch(
            grid=[
                (logits.shape[0] + self.warps_per_cta - 1) // self.warps_per_cta,
                1 + self.quant_ctas,
                1,
            ],
            block=[self.threads, 1, 1],
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
        x: Optional[cute.Tensor],
        x_fp8: Optional[cute.Tensor],
        x_sf: Optional[cute.Tensor],
    ):
        thread, _, _ = cute.arch.thread_idx()
        cta, part, _ = cute.arch.block_idx()
        if part == 0:
            self.route(logits, bias, input_ids, tid2eid, indices, weights, cta, thread)
        else:
            if cutlass.const_expr(self.quant_ctas > 0):
                item = (part - 1) * self.threads + thread
                row = cta * self.warps_per_cta + item // self.blocks
                if item < self.warps_per_cta * self.blocks:
                    if row < x.shape[0]:
                        self.quantize(x, x_fp8, x_sf, row, item % self.blocks)

    @cute.jit
    def quantize(self, x: cute.Tensor, x_fp8: cute.Tensor, x_sf: cute.Tensor, row, block):
        """One 32-value block as ``cvt_warp_fp16_to_mxfp8``: the UE8M0 scale of
        ``amax * rcp(448)`` rounded up, then the values times its reciprocal in E4M3."""
        values = cute.make_rmem_tensor((MXFP8_BLOCK,), x.element_type)
        cute.autovec_copy(cute.zipped_divide(x[row, None], (MXFP8_BLOCK,))[None, block], values)
        elements = [values[i].to(cutlass.Float32) for i in range(MXFP8_BLOCK)]
        amax = cutlass.Float32(0.0)
        for element in elements:
            amax = cute.arch.fmax(amax, _abs(element))
        code = _ue8m0_round_up(_mul_rn(amax, _rcp_approx_ftz(cutlass.Float32(448.0))))
        out_scale = cutlass.Float32(0.0)
        if amax != cutlass.Float32(0.0):
            out_scale = _rcp_approx_ftz(_e8m0_to_f32(code))
        scaled = [_mul_rn(element, out_scale) for element in elements]
        for word in cutlass.range_constexpr(MXFP8_BLOCK // 4):
            x_fp8[row, block * (MXFP8_BLOCK // 4) + word] = _to_e4m3x4(
                *scaled[4 * word : 4 * word + 4]
            )
        x_sf[row, block] = cutlass.Uint8(code)

    @cute.jit
    def route(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        input_ids: cute.Tensor,
        tid2eid: cute.Tensor,
        indices: cute.Tensor,
        weights: cute.Tensor,
        cta,
        thread,
    ):
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
                    weight = (
                        cute.math.div(
                            weights_k[k], total, rounding=cute.math.RoundingMode.NEAREST_EVEN
                        )
                        * self.routed_scaling_factor
                    )
                    weights[row, k] = weight.to(weights.element_type)


@functools.cache
def _compile(
    num_experts: int,
    top_k: int,
    routed_scaling_factor: float,
    logits_dtype: torch.dtype,
    ids_dtype: torch.dtype | None,
    table_dtype: torch.dtype | None,
    weights_dtype: torch.dtype = torch.float32,
    hidden: int = 0,
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
        _TORCH_TO_CUTE[weights_dtype], (rows, top_k), stride_order=(1, 0)
    )
    x = x_fp8 = x_sf = None
    if hidden:
        x = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (rows, hidden), stride_order=(1, 0), assumed_align=16
        )
        x_fp8 = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint32, (rows, hidden // 4), stride_order=(1, 0)
        )
        x_sf = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (rows, hidden // MXFP8_BLOCK), stride_order=(1, 0)
        )
    kernel = DSparkDraftRoutingKernel(num_experts, top_k, routed_scaling_factor, hashed, hidden)
    return cute.compile(
        kernel,
        logits,
        bias,
        input_ids,
        tid2eid,
        indices,
        weights,
        x,
        x_fp8,
        x_sf,
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
    compiled(logits, bias, input_ids, tid2eid, indices, weights, None, None, None)
    return indices, weights


def dspark_draft_route_mxfp8(
    logits: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``dspark_draft_routing`` (bias-selected) with the weights in ``weights_dtype``, and in the
    same launch ``mxfp8_quantize(x, False)`` of the BF16 MoE input: (ids, weights, E4M3 x,
    UE8M0 scales [tokens, hidden / 32])."""
    num_tokens, num_experts = logits.shape
    hidden = x.shape[1]
    device = logits.device
    indices = torch.empty(num_tokens, top_k, dtype=torch.int32, device=device)
    weights = torch.empty(num_tokens, top_k, dtype=weights_dtype, device=device)
    x_fp8 = torch.empty(num_tokens, hidden // 4, dtype=torch.int32, device=device)
    x_sf = torch.empty(num_tokens, hidden // MXFP8_BLOCK, dtype=torch.uint8, device=device)
    if num_tokens:
        _compile(
            num_experts,
            top_k,
            routed_scaling_factor,
            logits.dtype,
            None,
            None,
            weights_dtype,
            hidden,
        )(logits, bias, None, None, indices, weights, x, x_fp8, x_sf)
    return indices, weights, x_fp8.view(torch.float8_e4m3fn), x_sf
