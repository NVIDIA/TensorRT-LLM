# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renormalized MoE routing: top-k over the raw router logits, then a softmax
over just those k, in one trtllm call.

The cells, built by `_ml4` and on `_ML4_32K` and `_ML4_131K`, are the Mistral Large 4
router's: 128 experts, top-4, fp32 logits, bf16 weights -- `_ml4` at the FP8 target's
max_num_tokens 9216, `_ML4_32K` at the NVFP4 target's 32768 (and the NVFP4 dep8
target's chunks of the 8-rank gathered rows, with its decode graphs' gathered
8 x 1..16), `_ML4_131K` at the NVFP4 dep4 target's gathered 4 x 32768.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


def _ml4(**overrides) -> dict:
    """The Mistral Large 4 router: 128 experts, top-4, the fp32 logits of its
    replicated router GEMM, bf16 weights for the routed experts, at the FP8
    target's max_num_tokens 9216."""
    return dict(
        dict(
            tokens=9216,
            num_experts=128,
            topk=4,
            logits_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            dist="randn",
            magnitude=3.0,
        ),
        **overrides,
    )


# The Mistral Large 4 router at the NVFP4 target's envelope: max_num_tokens 32768
# with chunked prefill off, so one step routes up to 32768 rows. A literal, so
# the cells built on it read without running this module.
_ML4_32K = dict(
    tokens=32768,
    num_experts=128,
    topk=4,
    logits_dtype=torch.float32,
    output_dtype=torch.bfloat16,
    dist="randn",
    magnitude=3.0,
)

# The Mistral Large 4 router at the NVFP4 dep4 target: attention data parallel, so
# each of the 4 ranks zero-pads its rows to T_max = max(all_rank_num_tokens)
# and the router runs over the gathered 4 x T_max rows, up to 4 x 32768.
# A literal, like `_ML4_32K`.
_ML4_131K = dict(
    tokens=131072,
    num_experts=128,
    topk=4,
    logits_dtype=torch.float32,
    output_dtype=torch.bfloat16,
    dist="randn",
    magnitude=3.0,
)


class RenormMoeRoutingOp(OpWrapper):
    """`norm_topk_prob=True` routing. Returns `(topk_ids, topk_weights)` -- IDS FIRST.

    Per token: `ids` = the `topk` largest `router_logits`, ranked in the
    logits' own dtype, by decreasing value, ties to the LOWER expert index
    (a stable descending sort -- `torch.topk` breaks ties the other way, and at
    512 bf16 experts ties are common: it disagrees on ~30% of rows);
    `weights` = an fp32 softmax over just those `topk` logits, cast to
    `output_dtype` afterwards. Equal to HF's softmax-then-top-k-then-divide up
    to rounding (5.96e-08 at 512 / top-10; at 128 / top-4 on fp32 logits the
    bf16 weights are bitwise those of the softmax over all 128).

    `ids` is `[T, topk]` int32; `weights` `[T, topk]` in `output_dtype`, which
    defaults to fp32 -- not the logits' dtype. Sibling routing ops return
    `(weights, ids)`; both are `[T, topk]`, so a swapped unpack is no shape
    error.

    Fusion boundary: routing only. The router GEMM is the caller's, and so is
    expert parallelism -- every column of the row is ranked, so a TEP target
    passes the full 128-expert row and its expert runner keeps the ids in its
    own window. A DEP target gathers every rank's rows, zero padding
    included, before the router, and routes all of them on every rank.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why=(
                "Mistral Large 4 decode, CUDA graphs 1/2/4/8/16 (the captured, padded "
                "batches): each captured, replayed on new logits bitwise equal to eager"
            ),
            spec=_ml4(tokens=16, graph_tokens=(1, 2, 4, 8, 16)),
        ),
        Cell(
            why=(
                "Mistral Large 4 eager T 1..9216: every row count up to max_num_tokens, "
                "1 to 2304 blocks, each a call on the leading T rows of one set (~1 s)"
            ),
            spec=_ml4(tokens_range=(1, 9216)),
        ),
        Cell(
            why=(
                "Mistral Large 4 ties: integer fp32 logits 0..2 at 9216 tokens, every "
                "row ties inside its top-4 and most at the 4th/5th boundary"
            ),
            spec=_ml4(dist="coarse"),
        ),
        Cell(
            why=(
                "Mistral Large 4 wide logits (sigma 100) at 9216 tokens: the lower "
                "weights underflow through bf16 subnormals to 0"
            ),
            spec=_ml4(magnitude=100.0),
        ),
        Cell(
            why=(
                "Mistral Large 4 32768 eager T 1..32768: every row count of the NVFP4 "
                "target's max_num_tokens (a whole 32768-token prompt is one step), 1 "
                "to 8192 blocks -- the launcher's cap, reached at T 32765..32768 "
                "(8191 blocks at 32764) with one token per warp -- each a call on "
                "the leading T rows of one set (~2 s)"
            ),
            spec=dict(_ML4_32K, tokens_range=(1, 32768)),
        ),
        Cell(
            why=(
                "Mistral Large 4 32768 ties: integer fp32 logits 0..2 at 32768 tokens, "
                "every row ties inside its top-4 and at the 4th/5th boundary"
            ),
            spec=dict(_ML4_32K, dist="coarse"),
        ),
        Cell(
            why=(
                "Mistral Large 4 32768 wide logits (sigma 100) at 32768 tokens: the "
                "lower weights underflow through bf16 subnormals to 0"
            ),
            spec=dict(_ML4_32K, magnitude=100.0),
        ),
        Cell(
            why=(
                "Mistral Large 4 32768 past the block cap: 65537 rows on 8192 blocks, "
                "each warp looping over tokens 32768 apart (warp 0 three times); "
                "outside the NVFP4 envelope, for a step past 32768 rows"
            ),
            spec=dict(_ML4_32K, tokens=65537),
        ),
        Cell(
            why=(
                "Mistral Large 4 dep4 eager T 32769..131072: every row count of the "
                "dep4 target's gathered step (4 x T_max, T_max up to max_num_tokens "
                "32768) past the NVFP4 sweep, all on the 8192-block cap with each "
                "warp looping over up to four tokens 32768 apart (four at 131072) "
                "-- each a call on the leading T rows of one set (~5 s)"
            ),
            spec=dict(_ML4_131K, tokens_range=(32769, 131072)),
        ),
        Cell(
            why=(
                "Mistral Large 4 dep4 ties: integer fp32 logits 0..2 at 131072 gathered "
                "tokens, every row ties inside its top-4 and at the 4th/5th boundary"
            ),
            spec=dict(_ML4_131K, dist="coarse"),
        ),
        Cell(
            why=(
                "Mistral Large 4 dep4 wide logits (sigma 100) at 131072 gathered tokens: "
                "the lower weights underflow through bf16 subnormals to 0"
            ),
            spec=dict(_ML4_131K, magnitude=100.0),
        ),
        Cell(
            why=(
                "Mistral Large 4 dep4 padding: T_max 32768 with one rank at 32768 tokens "
                "and three at 1, so 98301 of the 131072 gathered rows are the all-+0.0 "
                "logits of a zero hidden row, each an exact 128-way tie routed to "
                "experts 0..3 at weight 0.25"
            ),
            spec=dict(_ML4_131K, rank_tokens=(32768, 1, 1, 1)),
        ),
        Cell(
            why=(
                "Mistral Large 4 dep4 decode, CUDA graphs 4/8/16/32/64: the captured, "
                "padded per-rank batches 1/2/4/8/16 gathered over 4 ranks, each "
                "captured, replayed on new logits bitwise equal to eager"
            ),
            spec=dict(_ML4_131K, tokens=64, graph_tokens=(4, 8, 16, 32, 64)),
        ),
        Cell(
            why=(
                "Mistral Large 4 dep8 decode, CUDA graphs 8..128 step 8: the per-rank "
                "batches 1..16 (the captured, padded 1/2/4/8/16 and every size of the "
                "default list) gathered over 8 ranks, 2 to 32 blocks, each captured, "
                "replayed on new logits bitwise equal to eager"
            ),
            spec=dict(
                _ML4_32K,
                tokens=128,
                graph_tokens=(8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128),
            ),
        ),
    )

    note = """
    Metadata consumed: none; stateless, deterministic, runs on the ambient
    stream. The launch (`invokeCustomMoeRouting`, in
    `kernels/moe/routing/customMoeRoutingKernels.cu`) picks its kernel
    instance by `nextPowerOfTwo(num_experts)` and `nextPowerOfTwo(topk)` --
    Mistral Large 4's 128 / top-4 is the (128, 4) instance with a 128-thread
    block, one warp per token, so ceil(T / 4) blocks: 1 to 2304 over the FP8
    target's T 1..9216 and 1 to 8192 over the NVFP4 target's T 1..32768.
    8192 is the launcher's block cap, and 32768 rows reach it exactly: the
    profiled grid is 8191 blocks at T 32764 and 8192 at T 32765 and 32768, so
    up to 32768 every warp routes one token. Past it the grid stays at 8192
    blocks and each warp loops over tokens 32768 apart (8192 blocks measured
    at T 32769, 65536, 65537 and 131072). The dep4 target's gathered steps,
    T 32769..131072, all run there, each warp routing up to four tokens
    (exactly four at 131072). That path is bit-exact at every T in
    32769..131072 and at 131073; its cells are 65537 and the dep4 sweep. PDL
    (`TRTLLM_ENABLE_PDL`) was measured bitwise inert at 128 / top-4 on 32768
    rows and on T 32769, 65537, 98305, 131071 and 131072 (0 and 1 both equal
    to unset); the Mistral Large 4 cells run with it unset, which
    `getEnvEnablePDL` reads as on for sm_90 and later.

    Preconditions, enforced loudly by the op: `router_logits` 2-D CUDA bf16 /
    fp16 / fp32; `output_dtype` fp32, bf16 or None (fp16 raises even for fp16
    logits); `topk <= 16`; `num_experts <= 512`. `topk == 0` and `num_tokens == 0`
    return empties. The outputs are allocated by the op, from a graph's pool
    under capture.

    Silent, refused by `is_valid`: a non-contiguous `router_logits` (read as a
    dense `[T, E]` buffer from `data_ptr()`) and `topk > num_experts` (the
    surplus slots come back as expert 0 at weight 0 -- duplicated ids). A
    contiguous row window (a nonzero storage offset) is read correctly.

    Non-finite logits poison the whole row to NaN (`+inf`, `NaN`, an all
    `-inf` row); a `-inf` logit among finite ones is a soft mask, at weight 0
    once `topk` exceeds the finite experts. At 128 experts every id stays in
    [0, 128), distinct within its row, whatever the row holds, and a row never
    reaches its neighbours: a padded batch's garbage rows cost only their own
    outputs.

    Zero padding rows, which the dep4 target appends to each rank's rows up
    to T_max before the gather: a zero hidden row through
    `torch.ops.trtllm.cublas_mm` (bf16, 6144 -> 128, fp32 out) gives +0.0 in
    every column (measured at m 4, 64, 32768, 32772, 65536 and 131072 with
    every other row zero), and an all-+0.0 row is an exact 128-way tie that
    the op and `reference` both route to experts 0..3 at weight exactly 0.25.
    With one rank at 32768 tokens and three at 1, 98301 of the 131072
    gathered rows are padding: in one draw, experts 0..3 took 99333 to 99350
    rows each and no other expert more than 1098, a skew the expert runner
    carries, not this op.

    The dep8 target gathers 8 x T_max rows and routes them in chunks of at
    most 32768, each a call on a fresh contiguous logits tensor inside the
    T 1..32768 sweep; with one rank at 32768 tokens and seven at 1, each of
    chunks 1..7 is one live row and 32767 padding rows. Its decode graphs
    route the 8 x b gathered rows of per-rank batch b in one call: 8, 16, 32,
    64 and 128 at the captured, padded 1/2/4/8/16, every multiple of 8 to 128
    under the default batch list, 2 to 32 blocks.

    Where the op and `reference` part, both from the kernel ranking by the
    float's bit pattern:
    - signed zeros: +0.0 ranks above -0.0, where `reference` (and torch)
      holds them equal and keeps the lower index first. A row whose top-k
      boundary falls inside a run of mixed-sign zeros gets different ids, and
      equal weights. A zero hidden row through `torch.ops.trtllm.cublas_mm`
      (bf16, 6144 -> 128, fp32 out) gives +0.0 throughout.
    - a NaN with its sign bit set ranks below -inf: the op skips it and
      answers finite weights over the rest of the row, where `reference` ranks
      it first and answers NaN. CUDA arithmetic produces the positive NaN
      (`0 * inf` and `-nan` both gave 0x7fffffff).

    `torch.ops.trtllm.default_moe_routing_op` has the identical signature and
    softmaxes before the top-k: same ids, un-renormalized weights (rows sum to
    0.09-0.27 at 512 / top-10). `RenormalizeMoeRoutingMethod`'s PyTorch
    fallback (used past 512 / 16) is not bit-compatible: it uses `torch.topk`
    and casts before the softmax.

    Bit-exact against `reference` over 2040 configurations (in dtype x out
    dtype x num_experts {1..512} x topk {0..16} x tokens {0..8192}); the
    test's sweep re-drives the boundaries of that space. At 128 / top-4
    on fp32 logits: bit-exact at 9216, 32768 and 131072 tokens for logits of
    sigma 1e-3 to 100 and integer ties, in both output dtypes, and at every T
    in 1..131072 with bf16 weights. At sigma 100, 14012 of the 131072 bf16
    weights on 32768 rows are 0 and 2490 subnormal; 55908 of the 524288 on
    131072 rows are 0 and 9986 subnormal. The cells cover only 128 / top-4 on
    fp32 logits with bf16 weights: bf16 and fp16 logits, fp32 weights, other
    expert counts and other topk are accepted by the op but not certified here.

    Cost: back to back, a call's time is linear in T past the block cap. The
    stateless op reads no KV pool, so the pool's size sets none of its limits.

    Hard limits (sm_103); the offsets are `customMoeRoutingKernel`'s, from a
    uint32 token index
      num_experts   <= 512 (raises past it)
      topk          <= 16 (raises past it); > num_experts silent, guarded
      logits offset token * num_experts in uint32; boundary 2^32 elements =
                    33554432 tokens at 128 experts (8388608 at 512); unprobed:
                    the dep4 target's largest step is 131072 x 128 = 16777216
                    = 2^24 elements (the NVFP4 tep4 target's and the dep8
                    target's 32768-row chunk's 32768 x 128 = 4194304, the FP8
                    target's 9216 x 128 = 1179648), and the
                    largest probed 131073 x 128 = 16777344, bit-exact
      out offset    token * topk in uint32; 2^30 tokens at top-4; unprobed
                    (the dep4 step's 131072 x 4 = 524288)
      num_tokens    passed to the kernel as int32; 2^31 tokens; unprobed
      grid          min(ceil(T / 4), 8192) blocks at 128 experts; the cap is
                    reached at T 32765 and is not a limit: past it warps loop
                    (certified at every T to 131072, probed at 131073,
                    bit-exact)
      no index passes through fp32; an expert id rides in a 16-bit field of
      the ranking key (`TopKRedType`, `kernels/moeTopKFuncs.cuh`), far above
      the 512-expert ceiling. The dep4 step's largest flat logits index is
      2^24 - 1, and the 131073-row probe crosses 2^24 bit-exact
      T             certified every T in 1..131072 at 128 / top-4, captured at
                    1/2/4/8/16, 4/8/16/32/64 and 8..128 step 8
    """

    def __call__(
        self,
        router_logits: torch.Tensor,
        topk: int,
        output_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.renorm_moe_routing_op(router_logits, topk, output_dtype)

    def reference(
        self,
        router_logits: torch.Tensor,
        topk: int,
        output_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stable descending sort in the logits' dtype, fp32 softmax over the
        selected k, then the cast."""
        order = torch.sort(router_logits, dim=-1, descending=True, stable=True).indices[:, :topk]
        selected = torch.gather(router_logits, 1, order).float()
        e = torch.exp(selected - selected.max(-1, keepdim=True).values)
        weights = e / e.sum(-1, keepdim=True)
        return order.to(torch.int32), weights.to(output_dtype or torch.float32)

    def is_valid(
        self,
        router_logits: torch.Tensor,
        topk: int,
        output_dtype: Optional[torch.dtype] = None,
    ) -> None:
        # The kernel addresses router_logits from data_ptr() as dense
        # [T, num_experts]: a column slice routes on the wider buffer's bytes.
        assert router_logits.is_contiguous(), (
            "router_logits must be contiguous; a strided view is read as a dense "
            "[num_tokens, num_experts] buffer and silently gives wrong routing"
        )
        # Surplus slots come back as duplicated expert 0 at weight 0.
        assert topk <= router_logits.shape[-1], (
            f"topk ({topk}) exceeds num_experts ({router_logits.shape[-1]}); the "
            "surplus slots come back as duplicates of expert 0 at weight 0"
        )


renorm_moe_routing_op = RenormMoeRoutingOp()
