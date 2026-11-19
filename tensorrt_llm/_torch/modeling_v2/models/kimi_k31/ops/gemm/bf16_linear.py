# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 BF16 projections: the GEMM, and the parameter holders of the projections and the SiTU MLP.

Every attention, shared-expert and dense-MLP projection of Kimi K3.1 is BF16 (routing refuses quantized ones) and
bias-free. Its GEMM is ``trtllm::cublas_mm`` (the catalog ``cublas_mm`` entry) where the projection was built for it
(the unsharded fused q_a / kv_a and the DSA indexer's wq_b) and ``F.linear`` (cuBLASLt through torch) everywhere else;
a row-parallel projection then sums its partial output over the TP group with ``allreduce_module``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.activation.situ_and_mul import situ_and_mul
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (
    FUSION_NONE,
    allreduce_module,
    tp_allreduce,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import (
    k31_low_m_linear,
    low_m_choice,
)
from tensorrt_llm.mapping import Mapping

#: Kimi K3.1's hidden size, q / kv LoRA ranks, MLA head dims (the DFlash2 drafter's too), shared-expert, dense-MLP and
#: drafter-MLP widths, and the KDA head count and width.
_HIDDEN, _Q_LORA, _KV_LORA, _ROPE = 7168, 1536, 512, 64
_HEADS, _QK_HEAD, _V_HEAD, _NOPE = 96, 192, 128, 128
_SHARED, _DENSE, _DRAFT_INTER = 6144, 33792, 14336
_KDA_HEADS, _KDA_HEAD = 96, 128
#: (attention TP, layout) of the served deployments: TEP8 (the general and TEP targets, and Helix CP8, whose attention
#: runs on the CP ranks as a TP8 group), TEP16 (the TEP perf recipe) and DEP16 / TP1 (attention DP keeps every attention
#: projection, the shared expert and the drafter whole).
_LAYOUTS = ((8, "TEP8"), (16, "TEP16"), (1, "DEP16 / TP1"))
#: The DSA indexer's heads (``config.DSA_INDEXER``).
_INDEX_HEADS = 32


def _cell(
    why: str, m: int, n: int, k: int, cublas: bool = False, row_stride: Optional[int] = None
) -> Cell:
    """``row_stride``: x is the first ``k`` columns of rows that wide (a column slice of another GEMM's output)."""
    spec = dict(m=m, n=n, k=k, dtype=torch.bfloat16, cublas=cublas, row_stride=row_stride or k)
    return Cell(why=why, spec=spec)


def _cells() -> tuple[Cell, ...]:
    """Every BF16 projection shape the served deployments run, at the smallest and largest row count of its calls.

    The entry makes the engine ``Linear``'s own call, so M, N and K pick the same kernel on both sides; the branch
    (``cublas``) and the operand form (``f_b``'s strided input) are what a cell can tell apart. The cuBLAS
    projections keep one row count per ``trtllm::cublas_mm`` tactic-table key; every other projection keeps its
    decode and prefill row counts (1 and 2048, either side of the engine's low-M bound of 32), the drafter its block
    rows of 1 and 64 requests and the vision tower its smallest and largest image.
    """
    cells = []
    for m in (1, 40, 2048):
        cells.append(
            _cell(f"fused q_a / kv_a (cuBLAS), M={m}", m, _Q_LORA + _KV_LORA + _ROPE, _HIDDEN, True)
        )
    for m in (1, 16, 2048):
        cells.append(
            _cell(
                f"DSA indexer wq_b, {_INDEX_HEADS} heads x 128 (cuBLAS), M={m}",
                m,
                _INDEX_HEADS * 128,
                _Q_LORA,
                True,
            )
        )
    for m in (1, 2048):
        cells.append(
            _cell(
                f"DSA indexer [wk | weights_proj], {_INDEX_HEADS} heads, M={m}",
                m,
                128 + _INDEX_HEADS,
                _HIDDEN,
            )
        )
    for tp, layout in _LAYOUTS:
        heads = _HEADS // tp
        for m in (1, 2048):
            cells.append(_cell(f"{layout} MLA q_b, M={m}", m, heads * _QK_HEAD, _Q_LORA))
            cells.append(
                _cell(f"{layout} MLA output gate g_proj, M={m}", m, heads * _V_HEAD, _HIDDEN)
            )
            cells.append(
                _cell(f"{layout} MLA o_proj (partial), M={m}", m, _HIDDEN, heads * _V_HEAD)
            )
            cells.append(_cell(f"{layout} MLA kv_b, M={m}", m, heads * (_NOPE + _V_HEAD), _KV_LORA))
    for tp, layout in _LAYOUTS:
        inter = _SHARED // tp
        for m in (1, 2048):
            cells.append(_cell(f"{layout} shared-expert gate_up, M={m}", m, 2 * inter, _HIDDEN))
            cells.append(_cell(f"{layout} shared-expert down (partial), M={m}", m, _HIDDEN, inter))
    # The KDA layer (K31KdaLayer): the fused [q | k | v | g] and [f_a | b] (zero rows up to a multiple of 8)
    # projections, f_b (its input is the f_a columns of the [f_a | b] output, a strided view) and o_proj, at each
    # layout's KDA heads per rank.
    for tp, layout in _LAYOUTS:
        heads = _KDA_HEADS // tp
        width = heads * _KDA_HEAD
        bfa = -(-(_KDA_HEAD + heads) // 8) * 8
        for m in (1, 2048):
            cells.append(_cell(f"{layout} KDA [q | k | v | g], M={m}", m, 4 * width, _HIDDEN))
            cells.append(_cell(f"{layout} KDA [f_a | b], M={m}", m, bfa, _HIDDEN))
            cells.append(_cell(f"{layout} KDA f_b, M={m}", m, width, _KDA_HEAD, row_stride=bfa))
            cells.append(_cell(f"{layout} KDA o_proj (partial), M={m}", m, _HIDDEN, width))
    # The dense MLP runs over at most eight ranks (a TP subgroup on TEP16), whole on DEP16.
    for tp in (1, 8):
        inter = _DENSE // tp
        for m in (1, 2048):
            cells.append(
                _cell(f"dense MLP gate_up over {tp} rank(s), M={m}", m, 2 * inter, _HIDDEN)
            )
            cells.append(
                _cell(f"dense MLP down over {tp} rank(s) (partial), M={m}", m, _HIDDEN, inter)
            )
    # The DFlash2 drafter's block decode: replicated q_a and kv_a, q_b / o_proj over the attention TP group (its kv_b
    # is absorbed, no GEMM) and the SwiGLU MLP over the same group; M is the block rows of 1 and 64 requests.
    for m in (5, 320):
        cells.append(_cell(f"drafter q_a, M={m}", m, _Q_LORA, _HIDDEN))
        cells.append(_cell(f"drafter kv_a, M={m}", m, _KV_LORA + _ROPE, _HIDDEN))
    for tp, layout in _LAYOUTS:
        heads = _HEADS // tp
        inter = _DRAFT_INTER // tp
        for m in (5, 320):
            cells.append(_cell(f"{layout} drafter q_b, M={m}", m, heads * _QK_HEAD, _Q_LORA))
            cells.append(
                _cell(f"{layout} drafter o_proj (partial), M={m}", m, _HIDDEN, heads * _V_HEAD)
            )
            cells.append(_cell(f"{layout} drafter MLP gate_up, M={m}", m, 2 * inter, _HIDDEN))
            cells.append(_cell(f"{layout} drafter MLP down (partial), M={m}", m, _HIDDEN, inter))
    # The vision tower's projections, replicated; M is the call's patch (or merged-row) count: the smallest 2x2
    # image and the processor's largest image (67320 patches, 16830 merged rows).
    for name, k, n, rows in (
        ("wqkv", 1024, 3 * 1536, (4, 67320)),
        ("wo", 1536, 1024, (4, 67320)),
        ("mlp fc0", 1024, 4096, (4, 67320)),
        ("mlp fc1", 4096, 1024, (4, 67320)),
        ("projector fc0", 4096, 4096, (1, 16830)),
        ("projector fc1", 4096, 7168, (1, 16830)),
    ):
        for m in rows:
            cells.append(_cell(f"MoonViT {name}, M={m}", m, n, k))
    return tuple(cells)


class K31Bf16Linear(OpWrapper):
    """``x @ weight.T``: BF16 ``[M, K] x [N, K]^T -> [M, N]``, FP32 accumulation, one GEMM.

    ``cublas`` picks ``trtllm::cublas_mm`` (``weight.t()`` is read column-major, no transpose kernel, default output
    buffer); otherwise ``F.linear``. The two are not bitwise to each other: a projection keeps the one it was built
    with.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = _cells()

    note = """
    The GEMM of Kimi K3.1's projections at its settings: no low-M dispatcher (routing refuses
    ``TRTLLM_LOW_M_GEMM_BACKEND``), no CuTe DSL BF16 GEMM (routing refuses ``use_cute_dsl_bf16_gemm``), no bias, no
    LoRA. ``trtllm::cublas_mm`` gets ``bias=None, out_dtype=None`` and the default output buffer: every cuBLAS
    projection is unsharded, so its output never goes to an all-reduce window. The planned decode / verify GEMMs are
    ``k31_low_m_linear``, not this entry.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, cublas: bool) -> torch.Tensor:
        if cublas:
            return cublas_mm(x, weight.t(), None, None, 0, None)
        return F.linear(x, weight)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, cublas: bool) -> torch.Tensor:
        """The FP32 product (TF32 off), rounded once to ``x.dtype``. ``cublas`` does not change the value."""
        with true_fp32_matmul():
            return (x.float() @ weight.float().t()).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, cublas: bool) -> None:
        if cublas:
            # cublas_mm reads x as dense row-major and weight.t() as dense column-major; a strided view of either
            # gives a plausible wrong product.
            assert x.is_contiguous(), "the cuBLAS branch reads x as dense row-major"
            assert weight.is_contiguous(), (
                "the cuBLAS branch reads weight.t() as dense column-major"
            )


bf16_linear = K31Bf16Linear()


class K31Linear(nn.Module):
    """Parameter holder + op call; not a layer.

    Holds one BF16 projection's ``weight [out_local, in_local]`` with its tensor-parallel shard metadata:
    ``tp_mode`` ``"column"`` shards the output features, ``"row"`` the input features, None keeps the whole
    projection. ``forward`` calls :data:`bf16_linear` and, for a row-parallel projection built with ``reduce_output``,
    :data:`allreduce_module` on ``all_reduce``: the TP all-reduce handle ``tp_allreduce`` builds (an engine
    ``AllReduce`` module, kept as a child so the engine's MNNVL workspace prescale finds it). The holder reduces only
    through ``allreduce_module`` and picks no strategy.

    ``use_custom_cublas_mm`` selects the ``trtllm::cublas_mm`` branch, for unsharded projections only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype,
        mapping: Optional[Mapping] = None,
        tp_mode: Optional[str] = None,
        reduce_output: bool = True,
        allreduce_strategy=None,
        use_custom_cublas_mm: bool = False,
    ) -> None:
        super().__init__()
        if tp_mode not in (None, "column", "row"):
            raise ValueError(f"tp_mode {tp_mode!r}: 'column', 'row' or None")
        self.mapping = mapping or Mapping()
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.tp_mode = tp_mode
        self.dtype = dtype
        if use_custom_cublas_mm and self.tp_size > 1:
            raise ValueError("no Kimi K3.1 projection runs the cuBLAS branch on a TP group")
        self.use_custom_cublas_mm = use_custom_cublas_mm
        split = self.tp_size if tp_mode is not None else 1
        sharded = in_features if tp_mode == "row" else out_features
        if sharded % split:
            raise ValueError(f"{sharded} features do not split evenly over {split} ranks")
        self.in_features = in_features // split if tp_mode == "row" else in_features
        self.out_features = out_features // split if tp_mode == "column" else out_features
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=dtype), requires_grad=False
        )
        self.reduce_output = tp_mode == "row" and reduce_output
        self.all_reduce = (
            tp_allreduce(self.mapping, allreduce_strategy, dtype) if self.reduce_output else None
        )

    def forward(self, x: torch.Tensor, *, reduce: bool = True) -> torch.Tensor:
        """``x @ weight.T``, summed over the TP group for a reducing row-parallel projection unless ``reduce`` is
        False (then this rank's partial)."""
        out = bf16_linear(x, self.weight, self.use_custom_cublas_mm)
        if self.all_reduce is not None and reduce:
            out = allreduce_module(out, self.all_reduce, FUSION_NONE, None, None)
        return out

    def load_shard(self, weight: torch.Tensor, device: torch.device) -> torch.Tensor:
        """This rank's slice of the full 2-D checkpoint weight on ``device``: its rows of a column projection, its
        columns of a row projection (an even split), the whole weight of an unsharded one."""
        if self.tp_mode is None or self.tp_size == 1:
            return weight.to(device)
        dim = 1 if self.tp_mode == "row" else 0
        width, rest = divmod(weight.shape[dim], self.tp_size)
        if rest:
            raise ValueError(
                f"checkpoint weight {tuple(weight.shape)} does not split evenly over {self.tp_size} ranks"
            )
        return weight.narrow(dim, self.tp_rank * width, width).to(device)


class K31SituMLP(nn.Module):
    """Parameter holder + op calls; not a layer.

    Kimi K3.1's dense layer-0 MLP and shared experts: ``down_proj(situ_and_mul(gate_up_proj(x)))``. ``gate_up_proj``
    holds this rank's ``[gate | up]`` rows (column-parallel over ``mapping``'s TP group), ``down_proj`` the matching
    input columns (row-parallel; it sums over the group when ``reduce_output``). Both GEMMs are ``F.linear``; the
    activation is the ``situ_and_mul`` op with ``beta`` / ``linear_beta``.

    ``tp_size`` (when given) narrows the TP group to subgroups of that many consecutive ranks: the dense MLP's subgroup
    within a tray, 1 to replicate the MLP under attention DP.

    ``low_m_plan`` is the layout's low-M GEMM plan for this MLP (``k31_low_m_linear``), or None; ``low_m_forward``
    runs the MLP on it.
    """

    #: The layout's low-M plan of this MLP's GEMMs (None: none).
    _k31_low_m_plan = None

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: torch.dtype,
        mapping: Mapping,
        beta: float,
        linear_beta: float,
        tp_size: Optional[int] = None,
        reduce_output: bool = True,
        allreduce_strategy=None,
        low_m_plan=None,
    ) -> None:
        super().__init__()
        self._k31_low_m_plan = low_m_plan
        if tp_size is not None:
            assert mapping.tp_size % tp_size == 0, (mapping.tp_size, tp_size)
            # Subgroups of tp_size consecutive TP ranks: the outer factor goes to the pipeline axis.
            pp_size = mapping.pp_size * mapping.tp_size // tp_size
            mapping = Mapping(
                world_size=tp_size * pp_size,
                rank=mapping.rank,
                gpus_per_node=mapping.gpus_per_node,
                tp_size=tp_size,
                pp_size=pp_size,
                enable_attention_dp=mapping.enable_attention_dp,
            )
        self.beta = beta
        self.linear_beta = linear_beta
        self.gate_up_proj = K31Linear(
            hidden_size, 2 * intermediate_size, dtype=dtype, mapping=mapping, tp_mode="column"
        )
        self.down_proj = K31Linear(
            intermediate_size,
            hidden_size,
            dtype=dtype,
            mapping=mapping,
            tp_mode="row",
            reduce_output=reduce_output,
            allreduce_strategy=allreduce_strategy,
        )

    def activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        """``situ_and_mul`` over the rows of ``[gate | up]`` (leading dims kept)."""
        return situ_and_mul(
            gate_up.reshape(-1, gate_up.shape[-1]), self.beta, self.linear_beta
        ).reshape(*gate_up.shape[:-1], gate_up.shape[-1] // 2)

    def forward(self, x: torch.Tensor, *, reduce: bool = True) -> torch.Tensor:
        """The MLP output; with ``reduce`` False this rank's partial sum (no all-reduce)."""
        return self.down_proj(self.activation(self.gate_up_proj(x)), reduce=reduce)

    def _planned(self, x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
        """``x @ weight.T`` on the low-M plan where it names the shape and row count, else None."""
        plan = self._k31_low_m_plan
        if plan is None or low_m_choice(plan, x, weight) is None:
            return None
        return k31_low_m_linear(x, weight, plan, True)

    def low_m_forward(self, x: torch.Tensor, *, reduce: bool = True) -> torch.Tensor:
        """``forward`` with ``gate_up_proj`` on the low-M plan where it serves, and ``down_proj`` too where it does
        not reduce (attention DP); with ``reduce`` False this rank's partial sum."""
        h1 = self._planned(x, self.gate_up_proj.weight)
        if h1 is None:
            return self(x, reduce=reduce)
        h2 = self.activation(h1)
        down = self.down_proj
        if down.all_reduce is None:
            # No reduction in the down projection (attention DP): a plain GEMM, planned where it serves.
            out = self._planned(h2, down.weight)
            if out is not None:
                return out
        return down(h2, reduce=reduce)
