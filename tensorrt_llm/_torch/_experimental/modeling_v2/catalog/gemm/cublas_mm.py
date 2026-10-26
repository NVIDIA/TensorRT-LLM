# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul


def _rows(why: str, ms: tuple[int, ...], **spec) -> tuple[Cell, ...]:
    """One cell per row count of a projection.

    The row count leads `why`, so each cell's test id (the first 44
    characters) stays distinct.
    """
    return tuple(Cell(why=f"m={m} {why}", spec=dict(m=m, **spec)) for m in ms)


# Row counts a Step-5 text projection sees per rank: one decode token, the
# 128-request decode batch (`max_batch_size`), and one 8192-token prefill chunk
# (`max_num_tokens`, the envelope's largest row count).
_TEXT_ROWS = (1, 128, 8192)
_BF16 = dict(dtype=torch.bfloat16, out_dtype=None)


class CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated, one cublas_mm call.

    `mat_b` is read column-major, which is what a stored weight's `.t()` already
    is -- the targets pass `weight.t()` and no transpose kernel runs.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # The first eleven cells are the argument shapes that differ in kind rather
    # than in size -- fused bias (gpt-oss carries one on qkv, o and the router),
    # no bias (deepseek's projections), and an out_dtype that is not the input's
    # (deepseek's router asks for fp32 logits). After them, Step-5-Preview's
    # projections at its attention-DP width (whole heads, "DP") and its TP4
    # shard ("TP4"), each at every row count it runs.
    CELLS: tuple[Cell, ...] = (
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
            why="deepseek default q_b at attention TP4: q_lora 1536 -> 32 heads x 192",
            spec=dict(m=16, k=1536, n=6144, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek default kv_b at attention TP8: kv_lora 512 -> 16 heads x 256",
            spec=dict(m=64, k=512, n=4096, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek default o_proj at attention TP8: 16 heads x 128 -> hidden 7168",
            spec=dict(m=16, k=2048, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        # The MTP layer's own GEMMs. Its eh_proj is replicated on every rank; its
        # shared expert is bf16 (the NVFP4 export skips layer 61), so it runs here
        # rather than in nvfp4_gemm, split over attention TP like the trunk's.
        Cell(
            why="deepseek MTP eh_proj: [enorm(e) | hnorm(h)] 2 x 7168 -> hidden 7168",
            spec=dict(m=16, k=14336, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek MTP shared expert gate_up, whole under attention DP: 7168 -> 2 x 2048",
            spec=dict(m=16, k=7168, n=4096, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek MTP shared expert gate_up at attention TP8: 7168 -> 2 x 256",
            spec=dict(m=16, k=7168, n=512, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek MTP shared expert down at attention TP8: 256 -> hidden 7168",
            spec=dict(m=16, k=256, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        # Attention input, fused at load: 64 q + 4 k + 4 v heads x 192, plus 64
        # head-wise gate columns (TP4: 16 + 1 + 1 heads, 16 gates).
        *_rows("step5 q|k|v|g DP: 4096 -> 13888", _TEXT_ROWS, k=4096, n=13888, bias=False, **_BF16),
        *_rows("step5 q|k|v|g TP4: 4096 -> 3472", _TEXT_ROWS, k=4096, n=3472, bias=False, **_BF16),
        # Sparse-layer indexer, fused at load: 16 q heads x 256 + k 256 + z 256 +
        # 16 w (TP4: 4 heads, 4 w = 1540, zero-padded to 1544 at load).
        *_rows(
            "step5 indexer q|k|z|w DP: 4096 -> 4624",
            _TEXT_ROWS,
            k=4096,
            n=4624,
            bias=False,
            **_BF16,
        ),
        *_rows(
            "step5 indexer q|k|z|w TP4: 4096 -> 1544",
            _TEXT_ROWS,
            k=4096,
            n=1544,
            bias=False,
            **_BF16,
        ),
        *_rows(
            "step5 o_proj DP: 64 x 192 -> 4096", _TEXT_ROWS, k=12288, n=4096, bias=False, **_BF16
        ),
        *_rows(
            "step5 o_proj TP4: 16 x 192 -> 4096", _TEXT_ROWS, k=3072, n=4096, bias=False, **_BF16
        ),
        # Dense MLP (layers 0-2 and 91), intermediate 13824.
        *_rows(
            "step5 dense gate_up DP: 4096 -> 27648",
            _TEXT_ROWS,
            k=4096,
            n=27648,
            bias=False,
            **_BF16,
        ),
        *_rows(
            "step5 dense down DP: 13824 -> 4096", _TEXT_ROWS, k=13824, n=4096, bias=False, **_BF16
        ),
        *_rows(
            "step5 dense gate_up TP4: 4096 -> 6912", _TEXT_ROWS, k=4096, n=6912, bias=False, **_BF16
        ),
        *_rows(
            "step5 dense down TP4: 3456 -> 4096", _TEXT_ROWS, k=3456, n=4096, bias=False, **_BF16
        ),
        # Shared expert (layers 3-90), intermediate 1536.
        *_rows(
            "step5 shared gate_up DP: 4096 -> 3072", _TEXT_ROWS, k=4096, n=3072, bias=False, **_BF16
        ),
        *_rows(
            "step5 shared down DP: 1536 -> 4096", _TEXT_ROWS, k=1536, n=4096, bias=False, **_BF16
        ),
        *_rows(
            "step5 shared gate_up TP4: 4096 -> 768", _TEXT_ROWS, k=4096, n=768, bias=False, **_BF16
        ),
        *_rows(
            "step5 shared down TP4: 384 -> 4096", _TEXT_ROWS, k=384, n=4096, bias=False, **_BF16
        ),
        # Router: fp32 logits over 352 experts; 512 rows is the decode batch
        # after the attention-DP all-gather (4 ranks x 128).
        *_rows(
            "step5 router, fp32 logits: 4096 -> 352",
            (1, 128, 512, 8192),
            k=4096,
            n=352,
            dtype=torch.bfloat16,
            bias=False,
            out_dtype=torch.float32,
        ),
        *_rows(
            "step5 lm_head replicated: 4096 -> 128896",
            _TEXT_ROWS,
            k=4096,
            n=128896,
            bias=False,
            **_BF16,
        ),
        *_rows(
            "step5 lm_head TP4 vocab split: 4096 -> 32224",
            _TEXT_ROWS,
            k=4096,
            n=32224,
            bias=False,
            **_BF16,
        ),
        # Vision tower (eager), bf16 bias fused: 2704 rows is one 728-px
        # overview tile (52 x 52 patches); 13072 is the overview plus eight
        # 504-px crops (36 x 36 each). The projector sees those rows after the
        # two stride-2 downsamplers: 169 per overview, 81 per crop -- 493 with
        # four crops, 817 with the eight the tower's 13072-row cell carries.
        *_rows(
            "step5 vision attn in_proj, bias: 1536 -> 4608",
            (2704, 13072),
            k=1536,
            n=4608,
            bias=True,
            **_BF16,
        ),
        *_rows(
            "step5 vision attn out_proj, bias: 1536 -> 1536",
            (2704, 13072),
            k=1536,
            n=1536,
            bias=True,
            **_BF16,
        ),
        *_rows(
            "step5 vision mlp c_fc, bias: 1536 -> 8960",
            (2704, 13072),
            k=1536,
            n=8960,
            bias=True,
            **_BF16,
        ),
        *_rows(
            "step5 vision mlp c_proj, bias: 8960 -> 1536",
            (2704, 13072),
            k=8960,
            n=1536,
            bias=True,
            **_BF16,
        ),
        *_rows(
            "step5 vision projector: 6144 -> 4096",
            (169, 493, 817),
            k=6144,
            n=4096,
            bias=False,
            **_BF16,
        ),
    )

    note = """
    `output_buffer_kind` and `group` are the op's multi-stream and collective
    arguments. Every shipped target leaves them at their defaults, so neither is
    a cell axis and neither is certified here.

    Algorithm selection: the op looks up `(next_pow2(max(m, 8)), k, n)` in the
    bf16 table of `cublasScaledMMLut.h` (`bf16_algo_list`, `find_special_algo`)
    and otherwise takes cuBLASLt's own heuristic. None of the Step-5 shapes is
    in that table, so every Step-5 cell runs the heuristic's choice; a table
    entry added for one of these shapes changes the kernel and is not covered
    by these cells until they run again.

    ARCHS: the first eleven cells (gpt-oss, deepseek) were driven on sm_100
    and sm_103. The Step-5 cells (every `why` naming step5) were driven on
    sm_103 only and are uncertified on sm_100 until they run there.

    Preconditions a caller establishes: `mat_a` dense row-major [M, K];
    `mat_b` [K, N] with stride (1, K) (a stored weight's `.t()`); a bias, when
    given, contiguous [N] in the output dtype and only on 16-bit inputs.

    Uncertified: the TP4 indexer's unpadded width n 1540 (the target pads the
    fused weight to 1544 at load, which is the certified width). A one-off
    probe on sm_103 found the op accepts n 1540 (an output row of 3080 bytes,
    not a 16-byte multiple) and lands within 1.12 bf16 ulp of the row scale of
    an unrounded fp32 product at m 1, 128 and 8192 -- the same as at 1544 --
    but no cell holds it. Also uncertified: row counts other than the cells'
    (the cuBLASLt heuristic is chosen per shape), `output_buffer_kind` and
    `group`.

    Test cost: the lm_head cell at m 8192, n 128896 peaks at 35.5 GiB of
    device memory (operands, the fp32 reference and the comparison's fp32
    copies); every other cell stays under 9 GiB.

    Hard limits: the op passes m, n, k to cuBLASLt as int32 and indexes the
    output with ldc = n. The largest output here is lm_head at m 8192,
    n 128896 = 1.06e9 elements, below 2^31; 2^31 output elements fall at
    m 16661 for n 128896 and m 77673 for n 27648, outside every cell and the
    8192-row envelope, and were not probed.
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
        """fp32-accumulated, TF32 off -- see `true_fp32_matmul`.

        `output_buffer_kind` and `group` choose where the op allocates its
        output, not what it computes; they are ignored here.
        """
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
