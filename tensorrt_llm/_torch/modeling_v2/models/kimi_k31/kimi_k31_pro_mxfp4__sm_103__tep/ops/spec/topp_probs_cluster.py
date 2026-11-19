# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target probabilities of the sampled DFlash2 verify under advanced sampling mode NO_TOPK
(temperature and per-row top-p): one CuTe DSL launch, one thread-block cluster per row."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (  # noqa: E501
    filtered_probs,
    sampling_spec,
)

from ...kernels.spec.topp_probs import supports, topp_probs

#: The K3.1 verify shapes at the model card's T = 1.0 / top-p 0.95 and the per-row variants a NO_TOPK
#: batch may carry. ``logits`` (read by the test) picks the logits: ``randn`` (the sampler cells'
#: N(0, 2^2)), ``peaked`` (three tokens 6 / 5 / 4 above the row's maximum) or ``flat`` (N(0, 0.5^2)).
CLUSTER_TOPP_CELLS: tuple[Cell, ...] = (
    Cell(
        why="TEP8 w5 B1 verify (5 rows) at the card's T=1.0, top-p 0.95",
        spec=sampling_spec(5, top_p=0.95),
    ),
    Cell(
        why="TEP32 w8 B1 verify (8 rows) at T=1.0, top-p 0.95",
        spec=sampling_spec(8, top_p=0.95),
    ),
    Cell(
        why="TEP8 w5 B2 verify (10 rows) at T=1.0, top-p 0.95",
        spec=sampling_spec(10, top_p=0.95),
    ),
    Cell(
        why="TEP8 w5 B3 verify (15 rows) at T=1.0, top-p 0.95",
        spec=sampling_spec(15, top_p=0.95),
    ),
    Cell(
        why="w5 verify of 8 requests (40 rows): more clusters than one wave",
        spec=sampling_spec(40, top_p=0.95),
    ),
    Cell(
        why="DEP16 per-rank verify (50 rows) at T=1.0, top-p 0.95",
        spec=sampling_spec(50, top_p=0.95),
    ),
    Cell(
        why="w8 verify of 9 requests (72 rows) at T=0.8, top-p 0.9",
        spec=sampling_spec(72, temperature=0.8, top_p=0.9),
    ),
    Cell(
        why="DEP16 per-rank verify (135 rows) at T=1.0, top-p 0.95",
        spec=sampling_spec(135, top_p=0.95),
    ),
    Cell(
        why="per-row top-p (1.0 rows take the plain softmax) and greedy-sentinel rows (T=1e-5)",
        spec=sampling_spec(40, top_p=0.9, greedy_rows=5, mixed=True),
    ),
    Cell(
        why="peaked rows: a top-p 0.95 set of a few tokens",
        spec=sampling_spec(16, top_p=0.95, logits="peaked"),
    ),
    Cell(
        why="flat rows: more cut candidates than a CTA may push, the cluster-wide fallback passes",
        spec=sampling_spec(4, top_p=0.95, logits="flat"),
    ),
)


def serves(logits: torch.Tensor, temperatures: torch.Tensor, top_p: torch.Tensor) -> bool:
    """Whether :data:`k31_topp_probs_cluster` takes these inputs: SM 10.3, float32 row-contiguous
    16-byte-aligned ``[rows >= 1, vocab]`` logits with ``vocab`` a multiple of the kernel's
    ``cluster * threads * 4`` whose slice fits shared memory, and float32 ``temperatures`` /
    ``top_p`` holding at least one contiguous entry per row on the same device."""
    if not logits.is_cuda or logits.dim() != 2 or logits.dtype != torch.float32:
        return False
    if torch.cuda.get_device_capability(logits.device) != (10, 3):
        return False
    rows, vocab = logits.shape
    if rows < 1 or not logits.is_contiguous() or logits.data_ptr() % 16 or not supports(vocab):
        return False
    for t in (temperatures, top_p):
        if (
            t.dtype != torch.float32
            or t.dim() != 1
            or t.shape[0] < rows
            or t.device != logits.device
        ):
            return False
        if t.stride(0) != 1:
            return False
    return True


class K31TopPProbsCluster(OpWrapper):
    """``[rows, vocab]`` float32: each row's ``softmax(logits / T)`` renormalized over its top-p
    nucleus (:func:`filtered_probs` with top-p alone), newly allocated, in one launch of
    ``kimi_k31_pro_mxfp4__sm_103__tep/kernels/spec/topp_probs``.

    ``logits`` float32 ``[rows, vocab]``; ``temperatures`` / ``top_p`` float32, one entry per row.
    A row with ``top_p >= 1`` gets the plain temperature softmax; a greedy-sentinel row (T=1e-5)
    comes out one-hot on its argmax.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = CLUSTER_TOPP_CELLS
    note = """
    Same distribution as `filtered_probs` and the engine's samplers, not bit-equal to either: the
    weights come from `ex2.approx` (as the engine's `__expf`), the top-p cut is exact on those
    weights, and a token sitting on the cut can land on either side, so the test compares kept sets
    by mass at the cut and bounds each row's L1 distance. Tie rule: every token whose weight equals
    the cut weight is kept (the engine's fused sampler's rule); the reference keeps tied tokens in
    id order until the mass is reached.

    Deterministic: integer histograms and fixed-order reductions make the output bit-identical run
    to run, under CUDA-graph replay, with or without PDL (TRTLLM_ENABLE_PDL, read per call) and on
    either of the kernel's candidate paths. It depends on the kernel's cluster size, not on the other
    rows of the call. No RNG, no workspace; the first call per configuration compiles, later calls
    are graph-capture safe.

    The Kimi K3.1 targets call it for the target distributions of DFlash2 block verification when
    advanced_sampling_mode is NO_TOPK (top-k resolved to None; min-p is rejected at admission
    outside FULL), in place of the flashinfer chain; `serves` says which inputs it takes.
    ARCHS lists only what was driven: sm_103 (GB300).
    """

    def __call__(
        self, logits: torch.Tensor, temperatures: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        out = torch.empty_like(logits)
        return topp_probs(logits, temperatures, top_p, out, get_env_enable_pdl())

    def reference(
        self, logits: torch.Tensor, temperatures: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """:func:`filtered_probs` in fp32 with top-p alone (no top-k, no min-p)."""
        return filtered_probs(logits, temperatures, None, top_p, None)

    def is_valid(
        self, logits: torch.Tensor, temperatures: torch.Tensor, top_p: torch.Tensor
    ) -> None:
        assert logits.dim() == 2 and logits.dtype == torch.float32, (
            f"float32 [rows, vocab] logits, got {logits.dtype} {tuple(logits.shape)}"
        )
        assert logits.is_contiguous() and logits.data_ptr() % 16 == 0, (
            "logits must be row-contiguous and 16-byte aligned (one bulk copy per CTA slice)"
        )
        assert supports(logits.shape[1]), f"vocab {logits.shape[1]} does not tile the clusters"
        for name, t in (("temperatures", temperatures), ("top_p", top_p)):
            assert t.dtype == torch.float32 and t.dim() == 1, f"{name} must be float32 [rows]"
            assert t.shape[0] >= logits.shape[0] and t.stride(0) == 1, (
                f"{name} needs one contiguous entry per row"
            )


k31_topp_probs_cluster = K31TopPProbsCluster()
