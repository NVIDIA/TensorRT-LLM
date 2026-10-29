# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated, one cublas_mm call.

    `mat_b` is read column-major, which is what a stored weight's `.t()` already
    is -- the targets pass `weight.t()` and no transpose kernel runs.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Both targets use this for every dense projection, so the cells are the
    # three argument shapes that differ in kind rather than in size: fused bias
    # (gpt-oss carries one on qkv, o and the router), no bias (deepseek's
    # projections), and an out_dtype that is not the input's (deepseek's router
    # asks for fp32 logits).
    #
    # The hidden-6144 MLA + DSA target (tep4) calls it for its four unquantized
    # projections, each replicated on every rank, so k and n are the
    # checkpoint's own: kv_a_proj_with_mqa (576 on the RoPE layers, 512 on the
    # NoPE layers that own a latent), the indexer's wk and weights_proj stacked
    # into one 160-wide weight, and the router (128 experts, fp32 logits). Each
    # call site passes one row per token: decode is captured at the batches the
    # engine pads to (1, 2, 4, 8, 16), prefill is eager at any row count up to
    # the step's token budget -- 9216 for the FP8 checkpoint's target, 32768 for
    # the NVFP4 checkpoint's (max_num_tokens 32768 with chunked prefill off, so
    # one context step carries up to 32768 rows). cuBLASLt's heuristic picks the
    # kernel by m, at a grain finer than any set of edge cells could follow (see
    # `note`), so the eager cells drive every m in the range, each as its own
    # call. A 1..32768 sweep costs 4-5.5 s on sm_103, the 1..9216 one ~1.3 s.
    #
    # The NVFP4 checkpoint's attention-DP target (dep4) runs the router over
    # the token set gathered from its four ranks: each rank zero-pads its rows
    # to T_max = max(all_rank_num_tokens) <= 32768, so the router sees m =
    # 4 * T_max -- every multiple of 4 up to 131072 eager, and 4/8/16/32/64 in
    # the decode graphs (each rank padded to the captured batch 1/2/4/8/16).
    # Its cells drive every one of those m, with zero padding rows in the
    # operand (`zero_row_period`: every 3rd row), whose logits must be +0.0
    # bit for bit -- the exact tie the routing op is certified to break. The
    # 32772..131072 sweep costs ~6.5 s and 4.6 GiB on sm_103, the 4..32768 one
    # ~1.5 s.
    #
    # The dep8 target gathers from eight ranks: m = 8 * T_max, and it runs the
    # router once per chunk of at most 32768 gathered rows (a contiguous row
    # slice starting at a multiple of 32768, so m is a multiple of 8 up to
    # 32768 eager, which the 1..32768 and the gathered 4..32768 cells drive).
    # Its decode graphs gather m = 8 x b for the captured per-rank batch b:
    # 8/16/32/64/128 at the gates' 1/2/4/8/16, every 8 x 1..16 under the LLM
    # API's default list. Its graph cell captures each of those.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DSpark target capture projection, every prefill row 1..32768",
            spec=dict(
                m_range=(1, 32768),
                k=30720,
                n=6144,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=None,
                large=True,
            ),
        ),
        *(
            Cell(
                why=f"DSpark dense projection eager {k}->{n}, bias={bias}",
                spec=dict(
                    m_range=(1, 48), k=k, n=n, dtype=torch.bfloat16, bias=bias, out_dtype=None
                ),
            )
            for k, n, bias in (
                (6144, 8192, False),
                (6144, 2048, False),
                (6144, 1024, False),
                (8192, 6144, False),
                (2048, 6144, False),
                (1024, 6144, False),
                (6144, 24576, False),
                (6144, 6144, False),
                (6144, 3072, False),
                (24576, 6144, False),
                (3072, 6144, False),
                (7168, 1536, True),
                (512, 204800, False),
            )
        ),
        *(
            Cell(
                why=f"DSpark projection graph {k}->{n}, bias={bias}, fp32={fp32}",
                spec=dict(
                    graph_m=tuple(range(1, 65)),
                    k=k,
                    n=n,
                    dtype=torch.bfloat16,
                    bias=bias,
                    out_dtype=torch.float32 if fp32 else None,
                ),
            )
            for k, n, bias, fp32 in (
                (30720, 6144, False, False),
                (6144, 8192, False, False),
                (6144, 2048, False, False),
                (6144, 1024, False, False),
                (8192, 6144, False, False),
                (2048, 6144, False, False),
                (1024, 6144, False, False),
                (6144, 24576, False, False),
                (6144, 6144, False, False),
                (6144, 3072, False, False),
                (24576, 6144, False, False),
                (3072, 6144, False, False),
                (7168, 1536, True, False),
                (512, 204800, False, False),
                (6144, 576, False, False),
                (6144, 512, False, False),
                (6144, 160, False, False),
                (6144, 128, False, True),
            )
        ),
        Cell(
            why="DSpark DEP verification router graph, up to 8 ranks x 64 rows",
            spec=dict(
                graph_m=tuple(range(4, 513, 4)),
                k=6144,
                n=128,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=torch.float32,
                zero_row_period=3,
            ),
        ),
        Cell(
            why="gpt-oss qkv with fused bias: hidden 2880 -> (64 + 8 + 8) * 64",
            spec=dict(m=16, k=2880, n=5120, dtype=torch.bfloat16, bias=True, out_dtype=None),
        ),
        Cell(
            why="deepseek q_a, no bias: hidden 7168 -> q_lora 1536",
            spec=dict(m=16, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek router: bf16 in, fp32 logits out, 256 experts",
            spec=dict(
                m=16, k=7168, n=256, dtype=torch.bfloat16, bias=False, out_dtype=torch.float32
            ),
        ),
        Cell(
            why="decode-shaped: one token, the shape every projection sees per step",
            spec=dict(m=1, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="eager m 1..9216, kv_a on the RoPE layers: hidden 6144 -> [c_kv 512 | k_pe 64]; "
            "prefill at every row count up to the token budget, so every heuristic bucket",
            spec=dict(
                m_range=(1, 9216), k=6144, n=576, dtype=torch.bfloat16, bias=False, out_dtype=None
            ),
        ),
        Cell(
            why="eager m 1..9216, kv_a on the NoPE layers that own a latent: hidden 6144 -> c_kv "
            "512, no k_pe",
            spec=dict(
                m_range=(1, 9216), k=6144, n=512, dtype=torch.bfloat16, bias=False, out_dtype=None
            ),
        ),
        Cell(
            why="eager m 1..9216, indexer wk + weights_proj stacked: hidden 6144 -> [k 128 | head "
            "weights 32]",
            spec=dict(
                m_range=(1, 9216), k=6144, n=160, dtype=torch.bfloat16, bias=False, out_dtype=None
            ),
        ),
        Cell(
            why="eager m 1..9216, router: hidden 6144 -> 128 experts, bf16 in, fp32 logits out",
            spec=dict(
                m_range=(1, 9216),
                k=6144,
                n=128,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=torch.float32,
            ),
        ),
        Cell(
            why="eager m 1..32768, kv_a on the RoPE layers (576): the NVFP4 target's prefill at every "
            "row count up to max_num_tokens 32768, so every heuristic bucket up to the 32768-row step",
            spec=dict(
                m_range=(1, 32768), k=6144, n=576, dtype=torch.bfloat16, bias=False, out_dtype=None
            ),
        ),
        Cell(
            why="eager m 1..32768, kv_a on the NoPE owners (512): every row count up to "
            "max_num_tokens 32768, so every heuristic bucket up to the 32768-row step",
            spec=dict(
                m_range=(1, 32768), k=6144, n=512, dtype=torch.bfloat16, bias=False, out_dtype=None
            ),
        ),
        Cell(
            why="eager m 1..32768, indexer wk + weights_proj (160): every row count up to "
            "max_num_tokens 32768, so every heuristic bucket up to the 32768-row step",
            spec=dict(
                m_range=(1, 32768), k=6144, n=160, dtype=torch.bfloat16, bias=False, out_dtype=None
            ),
        ),
        Cell(
            why="eager m 1..32768, router (128, fp32 out): every row count up to max_num_tokens "
            "32768, so every heuristic bucket up to the 32768-row step",
            spec=dict(
                m_range=(1, 32768),
                k=6144,
                n=128,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=torch.float32,
            ),
        ),
        Cell(
            why="CUDA graphs 1/2/4/8/16, kv_a RoPE layers (576): the captured decode batches, "
            "replayed on new data bitwise equal to eager",
            spec=dict(
                graph_m=(1, 2, 4, 8, 16),
                k=6144,
                n=576,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=None,
            ),
        ),
        Cell(
            why="CUDA graphs 1/2/4/8/16, kv_a NoPE layers (512): the captured decode batches, "
            "replayed on new data bitwise equal to eager",
            spec=dict(
                graph_m=(1, 2, 4, 8, 16),
                k=6144,
                n=512,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=None,
            ),
        ),
        Cell(
            why="CUDA graphs 1/2/4/8/16, indexer wk + weights_proj (160): the captured decode "
            "batches, replayed on new data bitwise equal to eager",
            spec=dict(
                graph_m=(1, 2, 4, 8, 16),
                k=6144,
                n=160,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=None,
            ),
        ),
        Cell(
            why="CUDA graphs 1/2/4/8/16, router (128, fp32 out): the captured decode batches, "
            "replayed on new data bitwise equal to eager",
            spec=dict(
                graph_m=(1, 2, 4, 8, 16),
                k=6144,
                n=128,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=torch.float32,
            ),
        ),
        Cell(
            why="dep4 router, eager gathered m 32772..131072: every multiple of 4 above 32768 up "
            "to 4 x max_num_tokens, so every heuristic bucket a gathered step reaches past the "
            "32768 cells, with zero padding rows",
            spec=dict(
                m_range=(32769, 131072),
                m_step=4,
                k=6144,
                n=128,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=torch.float32,
                zero_row_period=3,
            ),
        ),
        Cell(
            why="dep4 router, eager gathered m 4..32768 with zero padding rows: the 1..32768 cell's "
            "kernels at every multiple of 4, its padding rows +0.0 bit for bit",
            spec=dict(
                m_range=(4, 32768),
                m_step=4,
                k=6144,
                n=128,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=torch.float32,
                zero_row_period=3,
            ),
        ),
        Cell(
            why="dep4 router, CUDA graphs 4/8/16/32/64: per-rank batches 1/2/4/8/16 gathered over "
            "4 ranks, replayed on new data bitwise equal to eager, with zero padding rows",
            spec=dict(
                graph_m=(4, 8, 16, 32, 64),
                k=6144,
                n=128,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=torch.float32,
                zero_row_period=3,
            ),
        ),
        Cell(
            why="dep8 router, CUDA graphs 8..128 step 8: per-rank batches 1..16 gathered over 8 "
            "ranks (8/16/32/64/128 at the captured 1/2/4/8/16), replayed on new data bitwise "
            "equal to eager, with zero padding rows",
            spec=dict(
                graph_m=(8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128),
                k=6144,
                n=128,
                dtype=torch.bfloat16,
                bias=False,
                out_dtype=torch.float32,
                zero_row_period=3,
            ),
        ),
    )

    note = """
    `output_buffer_kind` and `group` are the op's multi-stream and collective
    arguments. Every target that calls it leaves them at their defaults, so
    neither is a cell axis and neither is certified here. Metadata consumed:
    none.

    Kernel selection (sm_103). The thop caller first looks the call up in a
    fixed bf16 algorithm table (`bf16_algo_list` in `thop/cublasScaledMMLut.h`,
    consulted by `find_special_algo` in `thop/cublasScaledMM.cpp`, keyed on the
    next power of two of m, at least 8, with k and n). No cell's key is in it,
    so every cell runs on cuBLASLt's own heuristic choice. That choice moves
    with m at a fine grain: over m 1..9216 at k 6144 the sweep cells changed
    kernel 239 times (n 576), 274 (n 512), 75 (n 160) and 372 (n 128, fp32
    out) -- nvjet sm103 tiles with and without split-K (split-K adds a second
    launch, `cublasLt::splitKreduce_kernel`), and at m 2 for n 160 and n 128 an
    sm80 cutlass kernel. Every m in that range is driven, so whatever the
    heuristic picks there is covered; a change of cuBLASLt can move the
    buckets and the cells still cover them.

    Above 9216 rows (the 1..32768 cells; measured on a 152-SM GB300 from the
    profiler's launches at every m, each launch's kernel, grid and block):
    the launch configuration changes 955 times between m 9216 and 32768
    (n 576), 781 (n 512), 491 (n 160) and 861 (n 128). Every m above 9216 is
    one nvjet launch without split-K (the last split-K m is 544, 432, 1984 and
    1680). None of the configurations chosen above 9216 is chosen at any m up
    to 9216: 143, 196, 81 and 152 of them, over 29, 35, 14 and 41 tile
    kernels, of which the 1..9216 sweep never launches 16, 22, 14 and 41. So
    the 9216 cells say nothing about a longer step, and the 32768 cells drive
    every m again rather than sample. Measured but not relied on: above 9216
    the choice changes only on 16-row boundaries.

    Above 32768 rows (the dep4 router's gathered cells, n 128 fp32 out, every
    multiple of 4 in 32772..131072; the same profiler measurement): the launch
    configuration changes 4532 times, over 457 configurations of 43 tile
    kernels. 455 of those configurations are chosen at no m up to 32768 (which
    chooses 358 over every m 1..32768), and the 1..32768 sweep never launches
    25 of the 43 kernels. Every one is a single nvjet launch without split-K.
    So the 32768 cells say nothing about a gathered step past them, and the
    gathered cell drives every multiple of 4 again. Measured but not relied
    on: there too the choice changes only on 16-row boundaries.

    Not batch-invariant. The rows of an m-row call are not always bitwise
    equal to the same rows of a larger call, because the kernel and its
    split-K reduction order change with m. Measured at k 6144 against the
    9216-row call: equal for 8864 of the 9216 values of m (n 576), 8832
    (n 512), 7927 (n 160) and 7792 (n 128, fp32 out). Against the 32768-row
    call, every m in 9217..32768 was equal at all four shapes (no split-K
    there), and the counts up to 9216 were the same within one. Against the
    131072-row call at n 128 (fp32 out, zero padding rows in the operand),
    every multiple of 4 in 32772..131072 was equal, and 7836 of the 8192
    multiples of 4 up to 32768. Every m is
    within the test's band; a caller that needs a decode row to equal the
    prefill row for the same token bit for bit does not get that here.

    fp32 out_dtype is an fp32 result, not a bf16 one widened: bf16 products
    are exact in fp32 and the sum is accumulated in fp32. Measured worst case
    over m 1..9216 at k 6144 n 128: 122 fp32 ulp of the row's largest logit,
    where the same logits rounded through bf16 sit at 2.2e4 to 3.3e4; over
    m 1..32768, 121 (124 at a 32768-row call on other operands), against
    3.3e4 rounded through bf16; over the gathered multiples of 4 in
    4..131072, 133 (at a 131072-row call), against 3.25e4 rounded through
    bf16.

    Zero padding rows. An all-zero row of mat_a gives a row of +0.0 logits,
    bit for bit, never -0.0: measured at every multiple of 4 in 4..131072 with
    every 3rd row zero, in the replays at the gathered 4/8/16/32/64 and at
    every multiple of 8 in 8..128, at the dep4 skew of T_max 32768 with
    ranks at 32768 / 1 / 1 / 1 tokens (98301 zero rows of 131072), and at
    the 174764- and 349527-row probes below. The band cannot see a -0.0 (the
    reference row is +0.0, at zero distance), so the cells read the padding
    rows' bits.

    CUDA graphs. Captured at m 1, 2, 4, 8, 16 and replayed on new data, the
    output is bitwise equal to an eager call on that data, for each of the
    four hidden-6144 shapes; for the router also at the gathered 4, 8, 16,
    32, 64 (four ranks) and at every multiple of 8 in 8..128 (eight ranks at
    1..16 rows each), with zero padding rows in the replayed data. Every
    router call at m 8..128 is split-K (measured on sm_103: one nvjet sm103
    tile kernel, then `cublasLt::splitKreduce_kernel`), over four tile
    kernels, one each for m 8..32, 40..64, 72..96 and 104..128; the last two
    are run by none of m 8/16/32/64. The replays there were within 20.7 fp32
    ulp element-wise and 13.3 RMS of the reference. The thop caller keeps
    one cuBLASLt workspace per (device, stream), created on the first
    call on that stream (`getWorkspaceTensor` in `thop/cublasScaledMM.cpp`);
    the cells call once on the capture stream before capturing, as an
    engine's warm-up does. A capture with no call before it on that stream
    was not measured.

    DSpark cells add the 30720->6144 target-feature projection through every
    prefill row count to 32768, the BF16 draft's dense and RNN projections,
    and graph rows 1..64. The RNN joint projection has a fused bias at k 7168;
    its output projection reaches n 204800. The DEP router graphs cover
    gathered multiples of four through 512 rows, including padding zeros.
    At k 30720 the prefill operand reaches 1006632960 elements / 2013265920
    bytes: below the first 32-bit byte offset at 34953 rows, and the first
    32-bit element offset at 69906 rows. Neither boundary is reached by this
    32768-row envelope; these two k-30720 boundaries remain unprobed.

    Uncertified: a fused bias at any k but 2880 or 7168; m above 32768 at k 6144,
    except the router's (n 128, fp32 out) multiples of 4 up to 131072 -- an m
    there that is not a multiple of 4, and any m above 131072, is in no cell;
    any k, n or m outside the cells; a device with another SM count than the
    152-SM GB300 the cells ran on (cuBLASLt's choice was not measured there).

    Hard limits (sm_103), for the hidden-6144 cells. The serving envelope's
    largest call is the dep4 router's 131072 gathered rows (n 128): mat_a is
    805306368 elements, 1610612736 bytes, below 2^31 bytes. The dep8 router
    runs on chunks of at most 32768 of its up to 262144 gathered rows, so its
    largest call is 32768 rows (one call over a gathered set above 131072
    rows would be in no cell). The two nearest boundaries were probed at
    n 128, fp32 out, with every 3rd row zero, in a
    call past each: the 2048 rows up to the boundary, every row past it and
    the last 2048 rows were within the test's band against the reference, the
    padding rows +0.0 and no output NaN.
      m, n, k       int32 in `cublas_gemm_caller`        boundary m = 2^31           unprobed
      mat_a bytes   m * 12288, if counted in 32 bits     2^31 / 12288 = 174763 rows  probed at 174764: passed
      mat_a elems   m * 6144, if counted in 32 bits      2^31 / 6144 = 349526 rows   probed at 349527: passed
                    (and m * 12288 past 2^32 bytes from the same row)
      out bytes     m * n * 2 (bf16), * 4 (fp32 out)     2^31 / 1152 = 1864135 rows (n 576),
                                                         / 1024 = 2097152 (n 512), / 320 = 6710887
                                                         (n 160), / 512 = 4194304 (n 128 fp32) unprobed
      out elems     m * n, if counted in 32 bits         2^31 / 576 = 3728271 rows,
                                                         / 512 = 4194304, / 160 = 13421773,
                                                         / 128 = 16777216          unprobed
    The unprobed boundaries are 32x the router's envelope (131072 rows) and
    57x the other shapes' (32768 rows) or more. No argument is an index
    carried through fp32.
    """

    def __call__(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cublas_mm(mat_a, mat_b, bias, out_dtype, output_buffer_kind, group)

    def reference(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        """fp32-accumulated, TF32 off -- see `true_fp32_matmul`."""
        with true_fp32_matmul():
            ref = mat_a.float() @ mat_b.float()
        if bias is not None:
            ref = ref + bias.float()
        return ref.to(out_dtype if out_dtype is not None else mat_a.dtype)

    def is_valid(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> None:
        # The kernel reads mat_a as dense row-major and mat_b as dense
        # column-major; other layouts produce silently wrong results.
        assert mat_a.is_contiguous(), "mat_a must be dense row-major [M, K]"
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] (e.g. weight.t())"
        )
        if bias is not None:
            out_dt = out_dtype if out_dtype is not None else mat_a.dtype
            # A bias in the wrong dtype or shape is accepted by the op and
            # produces silently wrong results; with fp32 inputs the bias is
            # accepted but silently ignored.
            assert mat_a.dtype != torch.float32, "bias is silently ignored for fp32 inputs"
            assert bias.dtype == out_dt, "bias dtype must equal the output dtype"
            assert bias.shape == (mat_b.shape[1],) and bias.is_contiguous(), (
                "bias must be a contiguous [N] tensor"
            )


cublas_mm = CublasMm()
