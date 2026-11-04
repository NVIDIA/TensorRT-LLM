# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 128x128 block-scale linear on operands that are already quantized: one DeepGEMM
``fp8_gemm_nt`` through ``torch.ops.trtllm.fp8_prequantized_swap_ab_gemm``.

Sourcing tier 1: a ``torch.library.custom_op`` in
``tensorrt_llm/_torch/custom_ops/torch_custom_ops.py``, the call
``FP8BlockScalesLinearMethod.apply`` (``_torch/modules/linear.py``) makes on the sm_100
family when its input arrives as an ``(e4m3, int32 scale)`` pair. Its body is a
single-tactic ``AutoTuner.choose_one`` that exists only to drive DeepGEMM's JIT warmup, the
output ``torch.empty``, and one ``deep_gemm.fp8_gemm_nt`` (plus one copy for an activation
scale in another layout than the quantizer's; see `note`).

This is the GEMM half of ``torch.ops.trtllm.fp8_swap_ab_gemm``, which quantizes its bf16
input itself first. The quantize is a separate op,
``torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, use_r128c4_layout=False)``, whose
``(codes, scale)`` this entry takes as they come. The engine's ``GatedMLP``
(``_torch/modules/gated_mlp.py``) feeds its down projection the same way from
``torch.ops.trtllm.silu_and_mul_fp8_quantize_1x128_packed_ue8m0``.

The Mistral Large 4 NVFP4 checkpoint's FP8 block linears are its ``FP8_PB_WO`` layers: wq, wo,
indexer wq_b, the shared experts' and layer 0's w1/w2/w3. Each stores its block scale as
``.weight_scale``, fp32 ``[N/128, 1, K/128, 1]`` (one per 128 x 128 block, every value a
power of two); viewed ``[N/128, K/128]`` it is the fp32 block scale the weight preparation
in `note` takes, so the operands this entry sees are those of any 128 x 128 block-FP8
checkpoint.

At dep4 (attention data parallel) every rank holds all 64 attention heads, so wq is the
whole ``[8192, 6144]`` and wo the whole ``[6144, 8192]``, and each call carries only the
rank's own tokens. The dense layer-0 MLP runs replicated on each rank's own tokens there, so
its gate_up is the whole ``[w1 ; w3]``, ``[49152, 6144]``, and its down the whole
``[6144, 24576]``.

At tep8 and tp8 (tensor parallel 8, attention data parallel off; tp8 differs from tep8 only
in the routed experts, which do not call this entry) every rank holds 8 of the 64 heads, so
wq is the rank's ``[1024, 6144]`` row slice and wo its ``[6144, 1024]`` K slice; the dense
layer-0 MLP and the shared experts shard eight ways, so the dense down is ``[6144, 3072]``,
the shared gate_up the stacked ``[w1 ; w3]`` slices, ``[1536, 6144]``, and the shared down
``[6144, 768]``, whose K of six 128-wide blocks leaves the last int32 of each scale row half
empty (see `note`). Every call carries every token of the step. The dense gate_up there is
``[6144, 6144]``, the tep4 dense down's shape. The projections named "at tep8" are both
targets' shards.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Every FP8 block-scale projection of the Mistral Large 4 targets, as the weight's [N, K]: per rank
#: at tep4 (FP8 and NVFP4 alike), then the NVFP4 target's attention projections at dep4, where
#: attention data parallel leaves the 64 heads unsplit, and its dense layer-0 MLP at dep4, which
#: runs replicated on each rank's own tokens; then the shards of the NVFP4 tep8 and tp8
#: targets, eight ways (their dense gate_up, [6144, 6144], is the tep4 dense down's shape above).
PROJECTIONS = (
    ("wq (q_proj, 16 heads x 128)", 2048, 6144),
    ("wo (o_proj, row-parallel K slice of 8192)", 6144, 2048),
    ("indexer wq_b (32 heads x 128, replicated)", 4096, 6144),
    ("dense layer-0 gate_up (TP4 of 2 x 24576)", 12288, 6144),
    ("dense layer-0 down (TP4 of 24576)", 6144, 6144),
    ("shared-expert gate_up (TP4 of 2 x 6144)", 3072, 6144),
    ("shared-expert down (TP4 of 6144)", 6144, 1536),
    ("wq at dep4 (q_proj, all 64 heads x 128)", 8192, 6144),
    ("wo at dep4 (o_proj, the whole K of 64 heads x 128)", 6144, 8192),
    ("dense layer-0 gate_up at dep4 (replicated, 2 x 24576)", 49152, 6144),
    ("dense layer-0 down at dep4 (replicated, the whole K of 24576)", 6144, 24576),
    ("wq at tep8 (q_proj, 8 heads x 128)", 1024, 6144),
    ("wo at tep8 (o_proj, row-parallel K slice of 8 heads x 128)", 6144, 1024),
    ("dense layer-0 down at tep8 (TP8 of 24576)", 6144, 3072),
    ("shared-expert gate_up at tep8 (TP8 of 2 x 6144)", 1536, 6144),
    ("shared-expert down at tep8 (TP8 of 6144)", 6144, 768),
)

#: The NVFP4 target's token budget (max_num_tokens; with chunked prefill off one context step
#: carries up to that many rows; per rank at dep4, where wq, wo and the replicated dense layer 0
#: see the rank's own tokens only; at tep8 and tp8 every rank's call carries all of the step's
#: tokens), the row count up to which the eager cells drive every m
#: (the FP8 target's max_num_tokens),
#: the window DeepGEMM's tile layout is constant on (see `note`), which the cells above
#: EVERY_M_TO drive at its edges and one interior row, and every decode batch a CUDA graph
#: can be captured at: max_batch_size 16 bounds both the padded 1/2/4/8/16 the gates capture
#: and the LLM API default's unpadded 1..16.
MAX_NUM_TOKENS = 32768
EVERY_M_TO = 9216
LAYOUT_WINDOW = 16
GRAPH_BATCHES = tuple(range(1, 17))

#: The [N, K] whose every-m sweep (LARGE_EVERY_M) or windowed sweep (LARGE_WINDOWED) runs past
#: the 10 s cell budget: warm bodies of 10.0 s (49152 x 6144, every m), 21.7 s (49152 x 6144,
#: windowed) and 9.8 s (6144 x 24576, windowed) on sm_103. Their cells carry `large`, which
#: pre-merge deselects; the pre-merge cells after the graph cells keep a share of each there.
LARGE_EVERY_M = ((49152, 6144),)
LARGE_WINDOWED = ((49152, 6144), (6144, 24576))

_AT_CALL_SITE = dict(output_dtype=torch.bfloat16, disable_ue8m0_cast=True)
_LARGE_WHY = "; large: past the 10 s cell budget at this N x K"


def _per_element_scale(scale: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Unpack an int32 [rows, ceil(k / 512)] UE8M0 scale into fp32 [rows, k].

    Byte ``j`` (little-endian) of int32 column ``c`` is the biased exponent ``e`` of the
    scale of K-block ``4 c + j``, worth ``2 ** (e - 127)``. Read by logical index, so any
    strides are fine. Exact for ``e`` in 1..254 (every scale a 1x128 quantizer with an amax
    floor writes); 0 would decode to 0.0 rather than 2^-127, 255 to inf rather than NaN.
    """
    shifts = torch.tensor((0, 8, 16, 24), dtype=torch.int32, device=scale.device)
    exponent = ((scale.unsqueeze(-1) >> shifts) & 0xFF).reshape(rows, -1)[:, : k // 128]
    return (exponent << 23).view(torch.float32).repeat_interleave(128, dim=1)


class Fp8PrequantizedSwapAbGemm(OpWrapper):
    """``(activation * a_scale) @ (weight * w_scale).T`` in fp32, rounded to `output_dtype`.

    `activation` is e4m3 ``[M, K]`` with one UE8M0 scale per row and 128-wide K block;
    `weight` is e4m3 ``[N, K]`` with one per 128 x 128 block, repeated over each block's 128
    rows. Both scales are int32, four exponent bytes per element, MN-major (see `note`).
    The GEMM and the scaling are inside the call; quantizing the activation and preparing
    the weight are not.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Sixteen projections call it, and N and K are compiled into DeepGEMM's kernel, so each is
    # a kernel set of its own. Each is driven three times, every m its own call (DeepGEMM
    # picks its tile layout by M): eagerly at every m in 1..9216; eagerly over 9217..32768
    # at three rows of each 16-row window, both of its edges among them, which reach every
    # layout picked there and both ends of each run of one layout; and captured in a CUDA
    # graph at every decode batch, 1..16. Three of those sweeps are `large` (LARGE_EVERY_M,
    # LARGE_WINDOWED); four cells after the graph cells drive a share of them in pre-merge.
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"eager {n}x{k} {site}: every m in 1..{EVERY_M_TO}, each its own "
                f"call, bitwise the leading rows of the {EVERY_M_TO}-row call, which "
                f"meets the reference{_LARGE_WHY if (n, k) in LARGE_EVERY_M else ''}",
                spec=dict(
                    n=n,
                    k=k,
                    m_range=(1, EVERY_M_TO),
                    **_AT_CALL_SITE,
                    **(dict(large=True) if (n, k) in LARGE_EVERY_M else {}),
                ),
            )
            for site, n, k in PROJECTIONS
        ),
        *(
            Cell(
                why=f"eager m {EVERY_M_TO + 1}..{MAX_NUM_TOKENS} {n}x{k} {site}: both edges "
                f"and one interior row of every {LAYOUT_WINDOW}-row window, each its own "
                f"call, bitwise the leading rows of the {MAX_NUM_TOKENS}-row call, which "
                f"meets the reference{_LARGE_WHY if (n, k) in LARGE_WINDOWED else ''}",
                spec=dict(
                    n=n,
                    k=k,
                    m_range=(EVERY_M_TO + 1, MAX_NUM_TOKENS),
                    m_window=LAYOUT_WINDOW,
                    **_AT_CALL_SITE,
                    **(dict(large=True) if (n, k) in LARGE_WINDOWED else {}),
                ),
            )
            for site, n, k in PROJECTIONS
        ),
        *(
            Cell(
                why=f"graphs {n}x{k} {site}: every decode batch 1..16, each captured, then "
                "replayed on new operands bitwise equal to eager",
                spec=dict(n=n, k=k, graph_m=GRAPH_BATCHES, **_AT_CALL_SITE),
            )
            for site, n, k in PROJECTIONS
        ),
        Cell(
            why="eager m 1..1024 49152x6144 dense layer-0 gate_up at dep4, pre-merge's share "
            "of its large every-m sweep: every m its own call, 17 of the 18 tile layouts that "
            "shape selects on 152 SMs (not the swap A/B block_m 256 one, first at m 2161), "
            "bitwise the leading rows of the 1024-row call, which meets the reference",
            spec=dict(n=49152, k=6144, m_range=(1, 1024), **_AT_CALL_SITE),
        ),
        Cell(
            why="eager m 21761..22016 49152x6144 dense layer-0 gate_up at dep4, pre-merge's "
            "share of its large windowed sweep: every m from 21846 on writes past byte 2^31 of "
            "the bf16 output; both edges and one interior row of every 16-row window, across "
            "a switch between the two tile layouts of 9217..32768, each its own call, bitwise "
            "the leading rows of the 22016-row call, which meets the reference",
            spec=dict(
                n=49152,
                k=6144,
                m_range=(21761, 22016),
                m_window=LAYOUT_WINDOW,
                **_AT_CALL_SITE,
            ),
        ),
        *(
            Cell(
                why=f"eager m {MAX_NUM_TOKENS - 255}..{MAX_NUM_TOKENS} {n}x{k} {site}, "
                "pre-merge's share of its large windowed sweep: the top of the envelope, both "
                f"edges and one interior row of every {LAYOUT_WINDOW}-row window, across a "
                "switch between the tile layouts of 9217..32768, each its own call, bitwise "
                f"the leading rows of the {MAX_NUM_TOKENS}-row call, which meets the reference",
                spec=dict(
                    n=n,
                    k=k,
                    m_range=(MAX_NUM_TOKENS - 255, MAX_NUM_TOKENS),
                    m_window=LAYOUT_WINDOW,
                    **_AT_CALL_SITE,
                ),
            )
            for site, n, k in PROJECTIONS
            if (n, k) in LARGE_WINDOWED
        ),
        Cell(
            why="default disable_ue8m0_cast=False on int32 scales, wq 2048x6144: bitwise "
            "equal to True, the call site's value",
            spec=dict(
                n=2048,
                k=6144,
                m_points=(1, 5, 16, 300, EVERY_M_TO, MAX_NUM_TOKENS),
                output_dtype=torch.bfloat16,
                disable_ue8m0_cast=False,
            ),
        ),
    )

    note = """
    Operands (sm_103, the layouts the cells drive).
      activation        e4m3 [M, K], unit stride along K. What
                        `fp8_quantize_1x128_packed_ue8m0(x, use_r128c4_layout=False)` returns.
      activation_scale  int32 [M, ceil(K/512)], strides (1, align(M, 4)): MN-major, each
                        int32 holding the exponent bytes of four consecutive 1x128 K blocks
                        (`_per_element_scale`). The storage the quantizer allocates ends at
                        row M of the last column; the gaps between columns are never read
                        for a valid row (the cells fill them with 0xFF bytes, UE8M0 NaN).
                        At K 768 (six blocks) bytes 2 and 3 of each row's second int32 are
                        empty: the quantizer writes 0 there, the cells 0xFF, and 0x00 or
                        0xFF there (in this scale or the weight's) gave bitwise one result
                        at ten m from 1 to 32768, while 0xFF in a live byte (block 5's) made
                        every output element NaN; a test pins both.
      weight            e4m3 [N, K] and weight_scale int32 [N, ceil(K/512)], strides (1, N),
                        prepared once at load the way `FP8BlockScalesLinearMethod.
                        transform_weights` does on the DeepGEMM path:
                        `resmooth_to_fp8_e8m0(weight, scale_fp32)` (re-derives each 128 x 128
                        block's power-of-two scale from its amax), then
                        `transform_sf_into_required_layout(scale, mn=N, k=K,
                        recipe=(1, 128, 128), is_sfa=False)` (repeats each block's scale per
                        row and packs it, 0 in K 768's empty bytes). `scale_fp32` is the NVFP4
                        checkpoint's `.weight_scale` [N/128, 1, K/128, 1] viewed
                        [N/128, K/128], or the FP8 checkpoint's `.qscale_weight`
                        [N/128, K/128]; the cells draw block scales 2^-15..2^-8, the range the
                        NVFP4 checkpoint's 404 FP8 block scales span. The tep4 and dep4
                        targets' loaders (`_block_sf` in their `weights.py`) assert the stored
                        scales are powers of two and pack them with the same
                        `transform_sf_into_required_layout` call, without the resmooth.
      output            a new [M, N] tensor; DeepGEMM's bf16 output is its fp32 result
                        rounded once (a float32 output, rounded to bf16, was bitwise equal
                        to it in a probe).
    Metadata consumed: none.

    Inside the call, besides the GEMM. An activation scale whose strides are not
    (1, align(M, 4)) is first copied into that layout by the runner, one extra copy kernel
    (a row-major int32 scale gave a bitwise-equal result). With int32 scales
    `disable_ue8m0_cast` selects nothing: DeepGEMM takes the same packed-UE8M0 path either
    way (a cell pins False against True). A row-strided activation (row stride K + 128,
    the TMA descriptor takes the stride) gave a bitwise-equal result too; a test pins it.

    Refused loudly (each pinned by a test):
      - the quantizer's default layout, `use_r128c4_layout=True` (a 1-D uint8 scale):
        `RuntimeError: dimensionality of sizes (1) must match dimensionality of strides (2)`,
        from the runner's re-stride;
      - an fp32 activation scale with `disable_ue8m0_cast=True`: DeepGEMM's
        `Unsupported architecture or scaling factor types`;
      - a weight scale that is not MN-major: DeepGEMM's `check_sf_layout` assertion
        `sf.stride(-2) == 1 or mn == 1`.
    An fp32 [M, K/128] activation scale with `disable_ue8m0_cast=False` is packed inside the
    call by DeepGEMM (its JIT kernel `pack_fp32_into_ue8m0`): powers of two gave the int32
    path's result bitwise; a scale that is not one faulted the device (`CUDA error:
    unspecified launch failure`), which leaves the context unusable -- the packer asserts a
    power of two on the device (`smxx_layout.cuh`). Probed only, at no cell.

    Batch invariance. Every eager cell asserts it: each m-row call's output is bitwise the
    leading m rows of the cell's top call on the same operands -- the 9216-row call at every
    m in 1..9216, the 32768-row call at both edges and one interior row of every 16-row
    window of 9217..32768 -- at every shape, and the pre-merge cells against their own top
    calls of 1024, 22016 and 32768 rows. A row's result does not depend on how many rows
    follow it.

    Tile layout by M. DeepGEMM picks one config (swap A/B or not, block M and N, cluster,
    stages) per call from M, N, K and the SM count (`get_layout_candidates` and `compare` in
    DeepGEMM's `csrc/jit_kernels/heuristics/sm100.hpp`). It reads M only as
    ceil(M / block_m) and, for the unswapped block_m, as M <= 32 and M <= 64; every candidate
    block_m is a multiple of 16, so the choice is constant on each window
    [16 j + 1, 16 j + 16]. Measured with `DG_PRINT_CONFIGS=1` at
    every M in 1..32768 and the sixteen shapes, on sm_103: constant on every window; 18 to
    25 configs per shape over 1..9216, and 2 to 8 over 9217..32768 (swap A/B with block_m 176
    to 256 and cluster 2 along N, or no swap with 128 x 224 or 128 x 256 tiles and cluster 2
    along M), each of which 1..9216 selects too. Over 9217..32768 they alternate in 82 to 188
    runs (every run 128 rows long at 49152 x 6144), and both edges of every run are rows the
    windowed cells drive. These are sm_103's configs; the heuristic takes the SM count, so another
    count may select others.

    DeepGEMM JIT. Each (N, K, config) is a kernel compiled with nvcc on its first use, about
    2.5 to 3 s each, and cached on disk (`$HOME/.dj`, or `DG_JIT_CACHE_DIR`): over M 1..32768
    the sixteen shapes select 323 kernels (19 at each of 8192 x 6144 and 6144 x 8192, which the
    warmup grid below compiled in 48.9 and 48.2 s, 18 at 49152 x 6144 and 19 at 6144 x 24576,
    in 47.7 and 49.1 s; at the tep8 shards 25 at 1024 x 6144, 19 at 6144 x 1024, 19 at
    6144 x 3072, 24 at 1536 x 6144 and 19 at 6144 x 768, in 72.4, 55.2, 56.5, 70.5 and
    55.3 s), so on a cold cache an every-m cell took 50 to 65 s, against 1.4 to
    3.1 s warm (4.7 s at 6144 x 24576, 10.0 s at 49152 x 6144; a windowed cell's body: 0.7 to
    5.7 s warm, 9.8 and 21.7 s at the two dense layer-0 dep4 shapes, peak 1.6 to 14.4 GiB
    above what the process held before). Calling the op at every M of the grid
    its single tactic tunes over (`deep_gemm_jit_warmup_buckets` in `_torch/utils.py`, 520
    values up to 8192; the grid is capped at 8192, so it is the same at max_num_tokens 9216
    and 32768) compiles all 323: after it, every M in 1..32768 at each shape compiled
    nothing. A first call outside a warm cache pays the compile; never time one.

    Uncertified: output_dtype float32 (DeepGEMM accepts it; probed at 32768 x 49152, whose
    6442450944 output bytes pass 2^32: 86.8 fp32 ulp of the row's scale from the fp32
    reference, 245.4 at 32768 x 6144 with K 24576, 20.1 to 80.6 at the tep8 shards); M above
    32768; the rows of
    9217..32768 the windowed cells skip, 13 of every 16 -- each runs its window's config
    (measured) and differs from the rows driven only in how full its last M tile is, which
    the driven rows vary across windows, so uniform behaviour there is inferred, not
    measured; any N x K not in `PROJECTIONS`; K not a multiple of 512 other than 768, where
    the last int32 of a scale row is partly padding too; exponent bytes 0
    and 255 in a live slot (no quantizer with an amax floor writes them); captured batches
    above 16; a
    CUDA-graph capture with no call before it on the capture stream (the cells warm each
    shape first, as the engine does).

    Hard limits (sm_103). The envelope reaches one boundary, byte 2^31 of the bf16 output at
    N 49152; the others are computed, not probed. The tep8 and tp8 shards reach none (their
    largest output, 32768 x 6144 bf16, is 402653184 bytes).
      M, N, K          int in DeepGEMM's host code           2^31                   unprobed
      activation elems M * K, largest 32768 * 24576          805306368 < 2^31       unprobed
      weight elems     N * K, largest 49152 * 6144           301989888 < 2^31       unprobed
      output elems     M * N, largest 32768 * 49152          1610612736 < 2^31      unprobed
      row offsets      m * K, m * N in 32 bits               2^31 / 24576 = 87381   unprobed
                                                             and 2^31 / 49152 =
                                                             43690 rows > 32768
      output bytes     M * N * 2 (bf16), largest 32768 *     at N 49152 every m     probed at 22016
                       49152 * 2 = 3221225472 > 2^31         from 21846 on          and 32768 rows:
                                                             writes past 2^31       passed, every row
                                                                                    in the band
    TMA descriptors carry 64-bit extents and strides. No argument is an index carried
    through fp32.
    """

    def __call__(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        disable_ue8m0_cast: bool = False,
    ) -> torch.Tensor:
        return torch.ops.trtllm.fp8_prequantized_swap_ab_gemm(
            activation,
            activation_scale,
            weight,
            weight_scale,
            output_dtype=output_dtype,
            disable_ue8m0_cast=disable_ue8m0_cast,
        )

    def reference(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        disable_ue8m0_cast: bool = False,
    ) -> torch.Tensor:
        """Dequantize both operands to fp32 (exact: a power-of-two scale times an e4m3
        code), multiply in fp32 with TF32 off (`true_fp32_matmul`), round once.

        Takes int32 packed scales only. `disable_ue8m0_cast` is ignored: on int32 scales it
        changes nothing.
        """
        m, k = activation.shape
        n = weight.shape[0]
        a = activation.float() * _per_element_scale(activation_scale, m, k)
        w = weight.float() * _per_element_scale(weight_scale, n, k)
        with true_fp32_matmul():
            out = a @ w.t()
        return out.to(output_dtype)


fp8_prequantized_swap_ab_gemm = Fp8PrequantizedSwapAbGemm()
