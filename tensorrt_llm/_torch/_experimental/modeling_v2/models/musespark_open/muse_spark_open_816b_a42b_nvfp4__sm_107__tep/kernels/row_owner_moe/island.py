# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``row_owner_moe``: the NVFP4 MoE layers of wide decode steps as a row-owner island.

Where it acts. On a wide generation server (TEP4, one tray) a decode step of ``R`` rows in the
lever's ``ROWS`` runs every decoder layer on all ``R`` rows of every rank: the o_proj partial
goes through the wide post-attention hop (``post_attn_hop_wide``: two-shot all-reduce, then the
gated residual join and the pre-FFN norm on every row), the MoE prologue (router GEMM, top-8
routing, pool GEMM, pre-expert RMSNorm + NVFP4 quantization) runs on every row, the experts run
EP over the rank's 64-expert window, the two-shot finalize all-reduces their output to every
row, and the unpool GEMM and the post-FFN join (with the next layer's pre-QKV norm) run on every
row again. Only the attention (TP over heads) and the experts (EP) need every row on every rank.

What it does instead. Rank ``r`` owns rows ``t = ranks * i + r``. For the decoder layers
``first..last`` (``1..L - 2``: the NVFP4 MoE layers between the BF16-expert edge layers, which
keep the served layer) the residual stream lives on the row's owner only (FP32, compact rows
``i``), and a layer runs:

1. the layer's attention on every row, unchanged (its input is every rank's normalized row);
2. the owner hop (``post_attn_hop_owner``): the wide hop's scatter shot and join on the owner's
   rows only (no gather shot);
3. the MoE prologue on the owner's rows (``ops/moe/owner_prologue``): the router GEMM at the
   full step's K-split count, the routing kernel launched as for ``R`` rows, the pool GEMM and
   the fused pre-expert RMSNorm + NVFP4 quantization;
4. the expert dispatch (``OwnerLLDispatch``): every owner's NVFP4 payload, scales, top-8 ids
   and weights to every rank, as dense ``[R, ...]`` tensors;
5. the layer's built-in expert call on those tensors (its routing served from the routing
   method's memo, which holds the dispatched ids and weights);
6. the finalize's scatter shot (``FinalizeTwoShotWorkspace(phase1_only=True)``): each token's
   combine on its owner, stored at the owner's compact row;
7. the unpool GEMM on the owner's rows;
8. the post-FFN join and the next layer's pre-QKV norm on the owner's rows, fused with the
   Lamport publication of every owner's normalized row to every rank (``OwnerRowPublish``) --
   and of the updated FP32 residual row on the DFlash capture layers and on ``last``.

Every value stored is the built-in kernels' value for that row: the owner hop and the finalize
scatter run the wide hop's and the two-shot finalize's arithmetic statement for statement; the
router GEMM, the routing kernel, the pool and unpool GEMMs and the quantization are per-row
functions once their launch choices are the full step's (the scans of ``owner_prologue``); the
row publication is the built-in Triton join, and its sentinels are NaN words no arithmetic
result takes. So the logits, the captures and every K/V byte equal the served path's.

The edge layers. Layer 0 runs as served and hands its full residual to the island (each rank
keeps its owner rows). ``last``'s publication carries the FP32 residual of every row, so layer
``L - 1`` runs as served on every row, and the final norm with it.

Contracts the callers keep (``owner_dispatch``'s notes): every exchange is called once per
island layer, ``last - first + 1`` (even) times per forward, so each forward and each captured
graph starts at an even host call count, which ``begin`` checks; the publication's views are
valid only until the next ``join_publish`` call, so the attention reading them and the capture
copying them run before it, in stream order (the next layer's attention is the first consumer
and the capture runs right after the publication); the dispatch's dense tensors are valid
until the next dispatch (the expert call and the finalize read them in between).

Refusals (``refusal``, at construction, agreed over the TP group; ``check_loaded`` once the
weights are in): anything but sm107 TEP4 on one tray with MoE EP over the TP group; a residual
join or post-attention join that is not the fused Triton kernel, or that norms the FP32 residual
(``MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL``); routing that is not the fused top-8 kernel at
every row count with the reference ordering (``MUSESPARK_OPEN_FULL_ROUTING_ALL_BATCHES`` and
``..._TORCH_ORDER``) or whose memo is off; an island layer whose experts are not the NVFP4
TRTLLM backend with the fused pre-expert quantization, or whose pool / unpool projection is not
the plain BF16 GEMM. Per forward (``DecodeTarget``): the island arms only on a step the served
chain it reproduces is armed for (the wide hop, the router GEMM and the two-shot finalize), with
a DFlash capture or none.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenMoE,
    MuseSparkOpenRoutingMethod,
    _version_or_none,
)
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger

from ...ops.allreduce import post_attn_hop_owner as _owner_hop
from ...ops.allreduce import post_attn_hop_wide as _wide
from ...ops.comm import owner_dispatch as _od
from ...ops.moe import finalize_two_shot as _two
from ...ops.moe import owner_prologue as _prologue
from ...ops.moe import router_gemm as _rg

HIDDEN = 8192
POOLED = 4096
TOP_K = 8
EXPERTS = 256
#: int32 words per row of the expert dispatch: the NVFP4 payload (two values per byte), its
#: E4M3 block scales (one byte per 16 values), the top-8 expert ids, the top-8 FP32 weights.
DISPATCH_SEGMENTS = (POOLED // 2 // 4, POOLED // 16 // 4, TOP_K, TOP_K)
#: The keywords ``DecodeTarget`` passes every layer call that the decoder layer itself consumes
#: (``projection_gathers``, ``capture_fold``); the attention never receives them.
LAYER_ONLY_KWARGS = frozenset({"fused_unpool_gather", "fused_pool_gather", "capture_fold"})


def _island_layers(model) -> range:
    """The decoder layers the island runs: every layer between the two edge layers."""
    return range(1, len(model.layers) - 1)


def refusal(model, rows: Sequence[Tuple[int, int]], row_cap: int) -> Optional[str]:
    """Why the island cannot be built on ``model`` for the ``rows`` ranges, or None.

    Reads the configuration only (no device memory, no weights); changes nothing.
    ``row_cap`` is the plan's row cap of ``router_gemm`` and ``finalize_two_shot``.
    """
    layers = list(model.layers)
    if len(layers) < 4:
        return f"it needs at least 4 decoder layers, got {len(layers)}"
    if len(_island_layers(model)) % 2:
        return (
            f"its {len(_island_layers(model))} island layers are odd: each exchange must make an "
            "even number of calls per forward"
        )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return "it is written for sm_107"
    mapping = layers[0].mapping
    if mapping.tp_size != 4 or mapping.is_multi_node():
        return f"it serves TP4 on one tray, got TP{mapping.tp_size}"
    if mapping.enable_attention_dp or mapping.moe_tp_size != 1:
        return "the MoE runs EP over the TP group only (no attention DP, no MoE TP)"
    if not rows:
        return "no ROWS"
    for lo, hi in rows:
        if lo > hi or lo < _owner_hop.MIN_ROWS or hi > row_cap:
            return (
                f"ROWS {lo}..{hi} must lie in {_owner_hop.MIN_ROWS}..{row_cap} (the row cap of "
                "router_gemm and finalize_two_shot, whose calls the island reproduces)"
            )
    if max(hi for _, hi in rows) > _two.GRID_STEP:
        return f"the finalize scatter serves at most {_two.GRID_STEP} rows"
    for index in _island_layers(model):
        layer = layers[index]
        why = _layer_refusal(layer, layers[index + 1])
        if why is not None:
            return f"layer {index}: {why}"
    return None


def _layer_refusal(layer, following) -> Optional[str]:
    """Why one island layer's configuration is not the served chain the island reproduces."""
    moe = layer.feed_forward
    if not isinstance(moe, MuseSparkOpenMoE):
        return f"its feed-forward is {type(moe).__name__}, not a MuseSparkOpenMoE"
    routing = getattr(moe.experts, "routing_method", None)
    if not isinstance(routing, MuseSparkOpenRoutingMethod):
        return f"its routing method is {type(routing).__name__}"
    if not routing._memo_enabled:
        return "the routing memo is off (MUSESPARK_OPEN_ROUTING_MEMO), which serves the expert call"
    if not (routing._full_routing_all_batches and routing._full_routing_torch_order):
        return (
            "its routing above 128 rows is not the fused top-8 kernel in the reference order "
            "(MUSESPARK_OPEN_FULL_ROUTING_ALL_BATCHES=1 and MUSESPARK_OPEN_FULL_ROUTING_TORCH_ORDER=1)"
        )
    if routing.top_k != TOP_K or routing.output_dtype != torch.float32:
        return "its routing is not top-8 with FP32 weights"
    if getattr(moe.experts, "comm", None) is not None:
        return "its experts run an external communication strategy"
    if not getattr(moe, "_pre_expert_quant_fused", True):
        return "its pre-expert quantization is not the fused RMSNorm + NVFP4 kernel"
    for name in ("pooler", "unpooler"):
        module = getattr(moe, name)
        if getattr(module, "bias", None) is not None:
            return f"its {name} has a bias"
        if getattr(module, "use_cute_dsl_bf16_gemm", False) or getattr(
            module, "use_custom_cublas_mm", False
        ):
            return f"its {name} does not run the plain BF16 GEMM"
    for name, join in (("post-attention", layer.add_attn), ("post-FFN", layer.add_ffn)):
        if not getattr(join, "_fused_enabled", False):
            return f"its {name} join is not the fused kernel (MUSESPARK_OPEN_FUSED_RESIDUAL)"
        if getattr(join, "_preqkv_norm_fp32", False):
            return (
                f"its {name} join norms the FP32 residual "
                "(MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL); the publication is the BF16 ordering"
            )
    for name, norm in (
        ("pre-FFN", layer.pre_ffn_norm),
        ("next pre-QKV", following.pre_qkv_proj_norm),
    ):
        if not getattr(norm, "use_gemma", False):
            return f"the {name} norm is not a Gemma norm"
    return None


def check_loaded(model, max_rows: int) -> None:
    """The weight-dependent refusals, once the weights are in.

    Raises:
        ValueError: if an island layer's experts are not the NVFP4 TRTLLM backend with the
            fused pre-expert quantization's single global input scale (on calls of up to
            ``max_rows`` rows), or its router weight is not FP32 ``[256, 8192]``.
    """
    from tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant import rmsnorm_fp4_quant_available
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend

    if not rmsnorm_fp4_quant_available():
        raise ValueError("[musespark_open] row_owner_moe needs flashinfer's rmsnorm_fp4quant")
    layers = list(model.layers)
    for index in _island_layers(model):
        moe = layers[index].feed_forward
        backend = getattr(moe.experts, "backend", moe.experts)
        weight = getattr(moe.router, "weight", None)
        why = None
        if not isinstance(getattr(backend, "op_backend", None), TRTLLMOpBackend):
            why = "its experts are not the TRTLLM backend"
        elif not getattr(backend, "has_nvfp4", False):
            why = "its experts are not NVFP4 (the island serves the NVFP4 layers)"
        elif getattr(backend, "fc31_act_scale", None) is not None:
            why = "its experts scale the activations per channel"
        elif (
            getattr(backend, "fc31_input_scale", None) is None
            or backend.fc31_input_scale.numel() != 1
        ):
            why = "its experts have no single global input scale"
        elif getattr(moe.experts, "moe_max_num_tokens", max_rows) < max_rows:
            why = f"its experts take fewer than {max_rows} tokens per call"
        elif (
            weight is None
            or weight.dtype != torch.float32
            or tuple(weight.shape) != (EXPERTS, HIDDEN)
            or not weight.is_contiguous()
        ):
            why = "its router weight is not a contiguous FP32 [256, 8192] tensor"
        if why is not None:
            raise ValueError(f"[musespark_open] row_owner_moe: layer {index}: {why}")


class RowOwnerIsland:
    """The island's transports and its per-layer forward; ``DecodeTarget`` arms it per step.

    Construction is collective over the TP group (four fabric allocations, the owner hop's
    and the finalize scatter's compilation) and succeeds or raises on every rank alike.
    """

    def __init__(self, model, rows: Sequence[Tuple[int, int]]) -> None:
        layers = list(model.layers)
        island = _island_layers(model)
        self.first = island.start
        self.last = island.stop - 1
        self.rows = tuple((int(lo), int(hi)) for lo, hi in rows)
        self.max_rows = max(hi for _, hi in self.rows)
        mapping = layers[0].mapping
        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        variants = sorted(
            {
                (
                    float(layers[i].post_attn_norm.variance_epsilon),
                    float(layers[i].pre_ffn_norm.variance_epsilon),
                    False,
                )
                for i in island
            }
        )
        self.hop = _owner_hop.OwnerHopWorkspace(
            mapping, variants=variants, max_rows=_wide.capacity_for(self.max_rows)
        )
        self.dispatch = _od.OwnerLLDispatch(mapping, DISPATCH_SEGMENTS, max_rows=self.max_rows)
        self.publish = _od.OwnerRowPublish(mapping, hidden=HIDDEN, max_rows=self.max_rows)
        self.finalize = _two.FinalizeTwoShotWorkspace(
            None, mapping, max_rows=self.max_rows, phase1_only=True
        )
        self.device = self.finalize.device
        self.sms = _rg._sms_of(self.device.index)
        self._reported: set = set()

    def materialize(self) -> None:
        """Put every buffer the transports allocated on ``meta`` on the island's device.

        The island is built with the model shell, under the loader's ``MetaInitMode``, which
        places every ``torch.empty`` on the meta device (the dispatch's dense receive rows,
        for one). Nothing writes such a buffer before its first forward (a write to a meta
        tensor raises in that mode), so a zeroed device buffer of its shape and dtype replaces
        it. Call it once the weights are in, outside that mode; it changes no rank's state that
        another rank reads.

        Raises:
            RuntimeError: if a meta tensor is left after the replacement.
        """

        def real(value):
            if isinstance(value, torch.Tensor) and value.is_meta:
                return torch.zeros(value.shape, dtype=value.dtype, device=self.device)
            if isinstance(value, list):
                return [real(v) for v in value]
            if isinstance(value, tuple):
                return tuple(real(v) for v in value)
            if isinstance(value, dict):
                return {k: real(v) for k, v in value.items()}
            return value

        def holds_meta(value) -> bool:
            if isinstance(value, torch.Tensor):
                return value.is_meta
            if isinstance(value, (list, tuple)):
                return any(holds_meta(v) for v in value)
            if isinstance(value, dict):
                return any(holds_meta(v) for v in value.values())
            return False

        parts = (
            ("hop", self.hop),
            ("dispatch", self.dispatch),
            ("publish", self.publish),
            ("finalize", self.finalize),
        )
        for _, part in parts:
            for attr, value in list(vars(part).items()):
                if holds_meta(value):
                    setattr(part, attr, real(value))
        left = [
            f"{name}.{attr}"
            for name, part in parts
            for attr, value in vars(part).items()
            if holds_meta(value)
        ]
        if left:
            raise RuntimeError(f"[musespark_open] row_owner_moe: meta buffers left: {left}")

    def takes(self, rows: int) -> bool:
        """Whether a decode step of ``rows`` rows is one of the island's (the same on every rank)."""
        return rows % self.ranks == 0 and any(lo <= rows <= hi for lo, hi in self.rows)

    def covers(self, index: int) -> bool:
        """Whether decoder layer ``index`` runs in the island."""
        return self.first <= index <= self.last

    def begin(self, rows: int, spec_metadata) -> bool:
        """Whether this armed step runs the island; checks the exchanges' call parity.

        A step whose speculative metadata captures through anything but the DFlash capture
        (whose rows the island publishes on the capture layers) keeps the served layers.

        Raises:
            RuntimeError: if an exchange's host call count is odd at a step boundary (a
                forward that stopped inside the island); every later call would reuse an epoch.
        """
        if spec_metadata is not None and (
            type(spec_metadata).maybe_capture_hidden_states
            is not DFlashSpecMetadata.maybe_capture_hidden_states
        ):
            self._report(rows, f"declined: {type(spec_metadata).__name__} has its own capture")
            return False
        if self.dispatch.calls % 2 or self.publish.calls % 2:
            raise RuntimeError(
                "[musespark_open] row_owner_moe: an exchange's call count is odd at a step boundary "
                f"(dispatch {self.dispatch.calls}, publication {self.publish.calls})"
            )
        self._report(rows, f"layers {self.first}..{self.last} run the row-owner island")
        return True

    def _report(self, rows: int, what: str) -> None:
        key = (rows, what)
        if key in self._reported:
            return
        self._reported.add(key)
        logger.info(f"[musespark_open] row_owner_moe: a {rows}-row decode step: {what}")

    @staticmethod
    def _captured(spec_metadata, layer_idx: int) -> bool:
        return (
            spec_metadata is not None
            and getattr(spec_metadata, "captured_hidden_states", None) is not None
            and layer_idx in getattr(spec_metadata, "_layer_to_idx", {})
        )

    def layer(
        self,
        index: int,
        layer,
        next_norm,
        *,
        position_ids,
        hidden_states: torch.Tensor,
        attn_metadata,
        precomputed_attn_input: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decoder layer ``index`` in the island; returns ``(residual, next attention input)``.

        ``hidden_states`` is the full FP32 residual on the island's first layer and the owner's
        compact rows after it; ``precomputed_attn_input`` holds every row. On ``last`` the
        residual returned holds every row again (the publication's FP32 segment), for the edge
        layer after it.
        """
        rows = precomputed_attn_input.shape[0]
        if index == self.first:
            residual = hidden_states[self.rank :: self.ranks].contiguous()
        else:
            residual = hidden_states
        attention_kwargs = {k: v for k, v in kwargs.items() if k not in LAYER_ONLY_KWARGS}
        partial = layer.self_attn(
            position_ids=position_ids,
            hidden_states=precomputed_attn_input,
            attn_metadata=attn_metadata,
            overlap_sp_output_projection=False,
            pad_sp_output_projection=False,
            **attention_kwargs,
        )
        if (
            partial.dtype != torch.bfloat16
            or partial.dim() != 2
            or tuple(partial.shape) != (rows, HIDDEN)
        ):
            raise RuntimeError(
                f"[musespark_open] row_owner_moe: layer {index}'s o_proj partial is "
                f"{tuple(partial.shape)} {partial.dtype}, not [{rows}, {HIDDEN}] BF16"
            )
        if not partial.is_contiguous():
            partial = partial.contiguous()
        alpha, beta = layer.add_attn.coefficients()
        updated, normed = self.hop.launch(
            partial,
            residual,
            alpha,
            beta,
            layer.pre_ffn_norm.weight,
            float(layer.post_attn_norm.variance_epsilon),
            float(layer.pre_ffn_norm.variance_epsilon),
            False,
        )
        unpooled = self._moe(layer.feed_forward, normed, rows, attn_metadata)
        spec_metadata = kwargs.get("spec_metadata")
        captured = self._captured(spec_metadata, layer.layer_idx)
        last = index == self.last
        alpha, beta = layer.add_ffn.coefficients()
        residual, views = self.publish.join_publish(
            updated,
            unpooled,
            layer.post_ffn_norm.gain(),
            alpha,
            beta,
            float(layer.post_ffn_norm.variance_epsilon),
            next_norm.weight,
            float(next_norm.variance_epsilon),
            rows,
            captured or last,
        )
        if captured:
            # The built-in capture of the layer's FP32 residual, from every rank's published rows,
            # before the next publication clears them.
            spec_metadata.maybe_capture_hidden_states(layer.layer_idx, views[1], None)
        if last:
            return views[1], views[0]
        return residual, views[0]

    def _moe(self, moe, normed: torch.Tensor, rows: int, attn_metadata) -> torch.Tensor:
        """The layer's MoE on the owner's normalized rows: the owner's unpooled rows."""
        own = rows // self.ranks
        routing = moe.experts.routing_method
        backend = getattr(moe.experts, "backend", moe.experts)
        gain = moe._pre_expert_quant_gain
        if gain is None:
            # As the built-in quantization makes it on its first call.
            gain = (moe.pre_expert_norm.weight.float() + 1).to(torch.bfloat16)
            moe._pre_expert_quant_gain = gain
        from flashinfer.norm import rmsnorm_fp4quant

        main = torch.cuda.current_stream()
        side = getattr(moe, "_prepare_stream", None)
        if side is not None:
            side.wait_stream(main)
        logits = _prologue.router_logits(
            normed, moe.router.weight, moe.metap_multiplier, _rg.splits(rows, self.sms)
        )
        selected, weights = _prologue.routing(
            logits, routing.e_score_correction_bias, rows, renorm_eps=routing.renorm_eps
        )
        with torch.cuda.stream(side) if side is not None else contextlib.nullcontext():
            pooled = F.linear(normed, moe.pooler.weight)
            fp4, scales = rmsnorm_fp4quant(
                pooled,
                gain,
                eps=moe.pre_expert_norm.variance_epsilon,
                global_scale=backend.fc31_input_scale.reshape(1),
                block_size=16,
                scale_format="e4m3",
                is_sf_swizzled_layout=False,
            )
        if side is not None:
            main.wait_stream(side)
            if not torch.cuda.is_current_stream_capturing():
                normed.record_stream(side)
                fp4.record_stream(main)
                scales.record_stream(main)
        dense = self.dispatch.dispatch(
            [
                fp4.view(torch.uint8).reshape(own, -1).view(torch.int32),
                scales.view(torch.uint8).reshape(own, -1).view(torch.int32),
                selected,
                weights.view(torch.int32),
            ],
            rows,
        )
        all_fp4, all_scales, all_selected, all_weights = dense
        all_weights = all_weights.view(torch.float32)
        # The quantization's own layout (flashinfer's shapes), over every row.
        all_fp4 = all_fp4.view(torch.uint8).reshape((rows * fp4.shape[0] // own, *fp4.shape[1:]))
        all_scales = all_scales.view(torch.uint8).reshape(
            (rows * scales.shape[0] // own, *scales.shape[1:])
        )
        # The expert call routes from its logits through the routing method, which serves a
        # logits tensor it has memoized: this key, holding the dispatched selection.
        key = torch.empty((rows, EXPERTS), dtype=torch.float32, device=normed.device)
        routing._routing_memo = (key, _version_or_none(key), all_selected, all_weights)
        try:
            outputs = moe.experts(
                Fp4QuantizedTensor(
                    all_fp4.view(torch.uint8), all_scales.view(torch.uint8), is_sf_swizzled=False
                ),
                key,
                do_finalize=False,
                output_dtype=torch.bfloat16,
                all_rank_num_tokens=(
                    attn_metadata.all_rank_num_tokens if attn_metadata is not None else None
                ),
                use_dp_padding=False,
            )
        finally:
            routing.clear_routing_memo()
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
            raise RuntimeError(
                "[musespark_open] row_owner_moe: the expert call did not return the unfinalized "
                f"(gemm2_output, expert_weights, expanded_idx_to_permuted_idx) triple, got "
                f"{type(outputs).__name__}"
            )
        gemm2_output, _, permuted = outputs
        combined = self.finalize.launch(
            gemm2_output,
            permuted,
            all_weights,
            moe.post_expert_norm_weight,
            moe.post_expert_norm_eps,
            rows,
        )
        return F.linear(combined, moe.unpooler.weight)


def build(model, rows: Sequence[Tuple[int, int]]) -> RowOwnerIsland:
    """The island for ``model`` (collective over the TP group; ``refusal`` agreed first)."""
    island = RowOwnerIsland(model, rows)
    logger.info(
        f"[musespark_open] row_owner_moe: layers {island.first}..{island.last} of decode steps of "
        f"{', '.join(f'{lo}..{hi}' for lo, hi in island.rows)} rows run as a row-owner island "
        f"(owner hop, owner MoE prologue, LL expert dispatch, finalize scatter, owner unpool, "
        f"published join)"
    )
    return island
