# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode kernels of the trtllm-gen NVFP4 MoE op (``FP4BlockScaleMoERunner``), where the model declares them
(``MoEDecodeOptions.tuning``). :func:`select` picks an un-finalized call's kernels (:class:`DecodeKernel`); each
produces the runner's gemm2 rows bit for bit and returns its ``do_finalize=False`` outputs:

* ``ENGINE``: a pre-routed SwiGLU call, FC1 -> SwiGLU -> NVFP4 -> FC2 in one persistent launch (``moe_nvfp4_engine``);
  given the router GEMM's per-tile candidates it also routes the call (:func:`engine_routes`).
* ``CUTE_FC2``: up to :data:`MAX_TOKENS` tokens whose FC2 cubin ``moe_lowm_fc2`` reproduces: the runner's FC1, then
  FC2 on ``moe_lowm_fc2`` (two smaller CTAs per SM).
* ``CUBIN``: the runner's own FC1 and FC2 (also where it runs its fine-grained-sync pair, which has no FC1-only
  entry: ``TLLM_USE_FINE_GRAINED_SYNC`` with such a cubin for the shape).

Lamport FC2 (:class:`LamportFc2Buffer`): ``ENGINE`` / ``CUTE_FC2`` store nothing before their PDL wait, release their
dependents only after it and write each word of a real row once (``cvt.rn.satfinite.bf16x2.f32``, never 0xFFFFFFFF),
so the MoE-output all-reduce can poll their rows instead of waiting for FC2.
"""

import enum
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.utils import ActType_TrtllmGen

from .decode_front import max_num_tiles
from .kernel_inputs import get_data_or_none, to_trtllm_gen_act_type

if TYPE_CHECKING:
    from ....custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoEInputs
    from .fp4_block_scale import TRTLLMGenFp4BlockScaleBase

#: Most tokens of a CuTe FC2 call (the kernel walks the runner's tile tables and has no token bound of its own).
MAX_TOKENS = 128
#: The runner's token tile whose FC2 cubins the CuTe FC2 reproduces.
TILE_N = 8
#: Most SMs ``ENGINE``'s persistent grid leaves free for work another stream runs beside it
#: (``TRTLLMGenFp4BlockScaleBase.sm_reserve_ceiling``): a shared expert's down-projection CTAs cannot share an SM with
#: an engine CTA. Measured on GB300 with Qwen3.5-397B-A17B NVFP4 (hidden 4096, top-10) MTP decode steps at the GEN
#: reference layouts; other GPUs and expert shapes are not measured.
ENGINE_SPARE_SMS = 8
#: Fewest rows of a call whose segments take MMA N 64 (fewer rows take N 32): the crossover measured on GB300 with the
#: same Qwen3.5-397B-A17B NVFP4 decode steps.
N64_MIN_ROWS = 225

#: The FC2 cubin options ``moe_lowm_fc2`` reproduces bit for bit (``get_gemm2_kernel_options`` of the C++ runner):
#: E2m1 x E2m1 -> FP32 -> BF16 in tiles of 128 x TILE_N with M128 N8 K64 MMAs over the whole K in order (one CTA per
#: tile, no split-K). The tile K, stage count, unrolling and tile scheduler only change when those MMAs issue.
_CUTE_FC2_OPTIONS = {
    "dtypeA": "E2m1",
    "dtypeB": "E2m1",
    "dtypeAcc": "Fp32",
    "dtypeC": "Bfloat16",
    "tileM": "128",
    "tileN": str(TILE_N),
    "mmaM": "128",
    "mmaN": str(TILE_N),
    "mmaK": "64",
    "clusterDimX": "1",
    "clusterDimY": "1",
    "clusterDimZ": "1",
    "numSlicesForSplitK": "1",
}


def cute_fc2_reproduces(gemm2_kernel_options: Mapping[str, str]) -> bool:
    """Whether ``moe_lowm_fc2`` reproduces the FC2 cubin of these options (the runner's
    ``get_gemm2_kernel_options(tile, config)``) bit for bit."""
    return all(gemm2_kernel_options.get(key) == value for key, value in _CUTE_FC2_OPTIONS.items())


class DecodeKernel(enum.Enum):
    CUBIN = 0
    CUTE_FC2 = 1
    ENGINE = 2


#: (act type, tile, config) -> whether ``moe_lowm_fc2`` reproduces that tactic's FC2 cubin. The C++ runners are
#: shared per act type (``FP4BlockScaleMoERunner.get_runner``), so the key names the cubin.
_FC2_REPRODUCED: Dict[Tuple[int, int, int], bool] = {}


def _cute_fc2_fits(
    kernel_runner: torch.ScriptObject,
    act_type: int,
    tactic: List[int],
    args: "FP4BlockScaleMoEInputs",
    top_k: int,
    intermediate_size: int,
) -> bool:
    if len(tactic) != 2 or tactic[0] != TILE_N or tactic[1] < 0:
        return False
    x, w2, w2_sf = args.hidden_states, args.gemm2_weights, args.gemm2_weights_scale
    if x.shape[0] > MAX_TOKENS or w2.dim() != 3 or w2_sf.dim() != 3:
        return False
    if w2.shape[2] * 2 != intermediate_size:
        return False
    key = (int(act_type), int(tactic[0]), int(tactic[1]))
    reproduced = _FC2_REPRODUCED.get(key)
    if reproduced is None:
        reproduced = _FC2_REPRODUCED[key] = cute_fc2_reproduces(
            kernel_runner.get_gemm2_kernel_options(key[1], key[2])
        )
    # The CuTe FC2 stands in for the tuned tactic's FC2; the runner runs its fine-grained-sync pair instead (which
    # has no FC1-only entry) only where TLLM_USE_FINE_GRAINED_SYNC is set and such a cubin exists for this shape.
    return reproduced and not kernel_runner.uses_fine_grained_sync(
        key[1], top_k, 2 * x.shape[1], intermediate_size, w2.shape[0], x.shape[0]
    )


def _engine_fits(args: "FP4BlockScaleMoEInputs", act_type: int, top_k: int) -> bool:
    """The engine's operands and shapes, with the K tile :func:`tile_k_for` picks for this call dividing both GEMMs'
    K and the call's (expert, row chunk) segments within the engine's segment table."""
    if act_type != ActType_TrtllmGen.SwiGlu or args.hidden_states_scale is None:
        return False
    if any(
        t is not None
        for t in (args.gemm1_bias, args.gemm1_alpha, args.gemm1_beta, args.gemm1_clamp_limit)
    ):
        return False
    if any(
        t is None
        for t in (
            args.output1_scale_scalar,
            args.output1_scale_gate_scalar,
            args.output2_scale_scalar,
        )
    ):
        return False
    from tensorrt_llm._torch.kernels.blackwell.moe_nvfp4_engine import MAX_LOCAL_EXPERTS, SEG_CAP

    w13, w2, x = args.gemm1_weights, args.gemm2_weights, args.hidden_states
    if w13.dim() != 3 or w2.dim() != 3 or w13.shape[0] > MAX_LOCAL_EXPERTS or x.dim() != 2:
        return False
    hidden, inter = w2.shape[1], w13.shape[1] // 2
    if w2.shape[2] * 2 != inter or w13.shape[2] * 2 != hidden or x.shape[1] * 2 != hidden:
        return False
    n_max = n_max_for(x.shape[0])
    if w13.shape[0] + -(-x.shape[0] * top_k // n_max) > SEG_CAP:
        return False
    tile_k = tile_k_for(n_max, hidden, inter)
    return hidden % tile_k == 0 and inter % tile_k == 0


def takes_engine(
    args: "FP4BlockScaleMoEInputs",
    act_type: int,
    *,
    top_k: int,
    do_finalize: bool,
    output: Optional[torch.Tensor],
    prerouted: bool,
    engine_state: Optional[torch.Tensor],
) -> bool:
    """Whether :func:`select` picks ``ENGINE`` for this call (arguments as there)."""
    return (
        not do_finalize
        and output is None
        and args.gemm2_bias is None
        and prerouted
        and engine_state is not None
        and _engine_fits(args, act_type, top_k)
    )


def engine_routes(
    backend: "TRTLLMGenFp4BlockScaleBase", num_rows: int, top_k: int, num_experts: int
) -> bool:
    """Whether ``backend``'s (``TRTLLMGenFp4BlockScaleBase``) un-finalized pre-routed call of ``num_rows`` NVFP4
    tokens over ``num_experts`` experts runs ``ENGINE`` with the routing in the engine (``moe_nvfp4_engine``'s
    ``route_top_k`` / ``route_lists``): :func:`takes_engine` on the operands its ``run_moe`` passes, within the
    routing's bounds: ``moe_nvfp4_engine.route_error`` on ``top_k`` keys of ``num_experts / 128`` lists, and the
    router GEMM's ``route_top_k`` epilogue that writes them."""
    from tensorrt_llm._torch.kernels.blackwell.moe_nvfp4_engine import ROUTE_MAX_ROWS, route_error
    from tensorrt_llm._torch.kernels.blackwell.moe_router_gemm import (
        QUANT_ROWS_PER_TILE,
        MoeRouterGemmKernel,
    )

    if not backend.decode_tuning or not 0 < num_rows <= ROUTE_MAX_ROWS:
        return False
    if num_experts % 128 or route_error(top_k, num_experts // 128) is not None:
        return False
    act_type = to_trtllm_gen_act_type(backend.activation_type)
    if act_type != ActType_TrtllmGen.SwiGlu:
        return False
    packed_hidden = backend.w3_w1_weight.shape[-1]
    if not MoeRouterGemmKernel.supports(
        num_rows=num_rows,
        k_extent=2 * packed_hidden,
        num_experts=num_experts,
        a_stride=2 * packed_hidden,
        token_tile=QUANT_ROWS_PER_TILE * (num_experts // 128),
        route_top_k=top_k,
    ):
        return False
    args = SimpleNamespace(
        hidden_states=torch.empty((num_rows, packed_hidden), device="meta"),
        hidden_states_scale=torch.empty((num_rows * packed_hidden // 8,), device="meta"),
        gemm1_weights=backend.w3_w1_weight,
        gemm1_bias=backend.w3_w1_bias if backend.bias else None,
        gemm1_alpha=backend.act_alpha,
        gemm1_beta=backend.act_beta,
        gemm1_clamp_limit=backend.act_clamp,
        gemm2_weights=backend.w2_weight,
        gemm2_bias=backend.w2_bias if backend.bias else None,
        output1_scale_scalar=get_data_or_none(backend, "fc31_scale_c"),
        output1_scale_gate_scalar=get_data_or_none(backend, "fc31_alpha"),
        output2_scale_scalar=get_data_or_none(backend, "fc2_alpha"),
    )
    return takes_engine(
        args,
        act_type,
        top_k=top_k,
        do_finalize=False,
        output=None,
        prerouted=True,
        engine_state=backend.decode_engine_state,
    )


def select(
    kernel_runner: torch.ScriptObject,
    act_type: int,
    tactic: List[int],
    args: "FP4BlockScaleMoEInputs",
    *,
    top_k: int,
    do_finalize: bool,
    output: Optional[torch.Tensor],
    intermediate_size: int,
    prerouted: bool,
    engine_state: Optional[torch.Tensor],
) -> DecodeKernel:
    """The kernels of one op call where the model declares its decode kernels (``FP4BlockScaleMoERunner``;
    host-side checks only). ``engine_state``: the calling backend's engine state
    (``moe_persistent_engine.new_state``), or None."""
    if takes_engine(
        args,
        act_type,
        top_k=top_k,
        do_finalize=do_finalize,
        output=output,
        prerouted=prerouted,
        engine_state=engine_state,
    ):
        return DecodeKernel.ENGINE
    if do_finalize or output is not None or args.gemm2_bias is not None:
        return DecodeKernel.CUBIN
    if _cute_fc2_fits(kernel_runner, act_type, tactic, args, top_k, intermediate_size):
        return DecodeKernel.CUTE_FC2
    return DecodeKernel.CUBIN


def n_max_for(num_rows: int) -> int:
    """MMA N = rows per segment. Each ring stage reserves N x 128 B of activation room, so a smaller N leaves more
    stages for the weight stream; an expert with more rows than N splits into segments whose weights are read again,
    mostly from L2."""
    from tensorrt_llm._torch.kernels.blackwell.moe_nvfp4_engine import N_MAX

    if num_rows <= N_MAX[0]:  # every expert's rows fit one segment
        return N_MAX[0]
    if num_rows < N64_MIN_ROWS:
        return N_MAX[1]
    return N_MAX[2]


def tile_k_for(n_max: int, hidden: int, inter: int) -> int:
    """K per ring stage: 512 halves the barrier round trips per byte while N <= 32 still leaves 4 stages and 512
    divides both GEMMs' K; else 256 (at N 64, 512 would leave 3 stages)."""
    return 512 if n_max <= 32 and hidden % 512 == 0 and inter % 512 == 0 else 256


def run_engine(
    args: "FP4BlockScaleMoEInputs",
    tables: List[torch.Tensor],
    tile: int,
    top_k: int,
    num_local: int,
    state: torch.Tensor,
    lamport_out: Optional[torch.Tensor],
    release_after_wait: bool,
    spare_sms: int = 0,
    candidates: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    local_start: int = 0,
) -> List[torch.Tensor]:
    """``ENGINE`` on the pre-routed ``tables`` (``prerouted_table_views`` for ``tile``); returns [gemm2 rows], as the
    runner's pre-routed un-finalized call. ``release_after_wait``: the engine's PDL release point (its dependency
    wait, else once its FC2 tickets are spent), the caller's choice for the kernel that follows.
    ``spare_sms``: SMs the persistent grid leaves free for work another stream runs beside it (the backend's
    ``reserve_sms_for_concurrent_kernels``; at most :data:`ENGINE_SPARE_SMS`).
    ``candidates`` (``PreroutedRouting.candidates``): the engine routes the call for the local experts from
    ``local_start`` and writes ``tables``, ``topk_ids`` and ``topk_weights`` itself (:func:`engine_routes`)."""
    from tensorrt_llm._torch.kernels.blackwell.moe_nvfp4_engine import moe_nvfp4_engine

    x = args.hidden_states
    m = x.shape[0]
    e2p, p2t, batch, limit, nct, npad = tables
    cap = max_num_tiles(m, top_k, num_local, tile) * tile
    w13, w2 = args.gemm1_weights, args.gemm2_weights
    hidden, inter = w2.shape[1], w13.shape[1] // 2
    dev = x.device
    h = torch.empty((cap, inter // 2), dtype=torch.uint8, device=dev)
    hsf = torch.empty((cap * inter // 16,), dtype=torch.uint8, device=dev)
    if lamport_out is not None:
        out = _lamport_out_view(lamport_out, cap, hidden)
    else:
        out = torch.empty((cap, hidden), dtype=torch.bfloat16, device=dev)
    n_max = n_max_for(m)
    moe_nvfp4_engine(
        x.view(torch.uint8),
        args.hidden_states_scale.view(torch.uint8).reshape(m, -1),
        p2t,
        batch,
        limit,
        nct,
        w13.view(torch.uint8),
        args.gemm1_weights_scale.view(torch.uint8),
        w2.view(torch.uint8),
        args.gemm2_weights_scale.view(torch.uint8),
        args.output1_scale_scalar,
        args.output1_scale_gate_scalar,
        args.output2_scale_scalar,
        h,
        hsf,
        out,
        state,
        tile=tile,
        n_max=n_max,
        tile_k=tile_k_for(n_max, hidden, inter),
        release_after_wait=release_after_wait,
        spare_sms=spare_sms,
        candidates=candidates,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        e2p=e2p,
        num_padded=npad,
        local_start=local_start,
    )
    return [out]


#: Positions of ``gemm2_bias``, ``do_finalize`` and ``output`` in the runner's ``run_moe`` / ``run_moe_prerouted``
#: arguments (``FP4BlockScaleMoeRunner::run`` / ``runPrerouted`` in fp4BlockScaleMoe.cpp), which its FC1-only entries
#: (``runFc1`` / ``runPreroutedFc1``) do not take.
GEMM2_BIAS_ARG, DO_FINALIZE_ARG, OUTPUT_ARG = 12, 25, 29


def fc1_only_args(run_args: Sequence[object]) -> Tuple[object, ...]:
    """The runner's ``run_moe`` / ``run_moe_prerouted`` arguments as its FC1-only entries take them: without
    ``gemm2_bias``, ``do_finalize`` and ``output``, which must be None, False and None."""
    run_args = tuple(run_args)
    if (
        run_args[GEMM2_BIAS_ARG] is not None
        or run_args[DO_FINALIZE_ARG]
        or run_args[OUTPUT_ARG] is not None
    ):
        raise ValueError("an FC1-only call is un-finalized, without FC2 bias or output tensor")
    return (
        run_args[:GEMM2_BIAS_ARG]
        + run_args[GEMM2_BIAS_ARG + 1 : DO_FINALIZE_ARG]
        + run_args[DO_FINALIZE_ARG + 1 : OUTPUT_ARG]
        + run_args[OUTPUT_ARG + 1 :]
    )


def run_cute_fc2(
    fc1_outputs: List[torch.Tensor],
    args: "FP4BlockScaleMoEInputs",
    lamport_out: Optional[torch.Tensor],
) -> List[torch.Tensor]:
    """``CUTE_FC2`` on the FC1-only runner's outputs; returns the runner's ``do_finalize=False`` triple."""
    from tensorrt_llm._torch.kernels.blackwell.moe_lowm_fc2 import moe_lowm_fc2

    (
        gemm1_output,
        gemm1_output_scale,
        gemm2_output,
        expert_weights,
        expanded_idx_to_permuted_idx,
        cta_batch,
        cta_limit,
        num_ctas,
    ) = fc1_outputs
    out = gemm2_output
    if lamport_out is not None:
        out = _lamport_out_view(lamport_out, gemm2_output.shape[0], gemm2_output.shape[1])
    moe_lowm_fc2(
        gemm1_output.view(torch.uint8),
        gemm1_output_scale.view(torch.uint8),
        args.gemm2_weights.view(torch.uint8),
        args.gemm2_weights_scale.view(torch.uint8),
        args.output2_scale_scalar,
        cta_batch,
        cta_limit,
        num_ctas.view(-1),
        out,
    )
    return [out, expert_weights, expanded_idx_to_permuted_idx]


def rows_in_lamport(rows: torch.Tensor, lamport_out: Optional[torch.Tensor]) -> bool:
    """Whether an un-finalized call's FC2 ``rows`` are the Lamport buffer ``lamport_out`` (``ENGINE`` / ``CUTE_FC2``
    wrote them there); a ``CUBIN`` call returns rows of its own and leaves the buffer untouched."""
    return lamport_out is not None and rows.data_ptr() == lamport_out.data_ptr()


def _lamport_out_view(lamport_out: torch.Tensor, rows: int, hidden: int) -> torch.Tensor:
    """The first ``rows`` rows of the Lamport FC2 buffer, the FC2 output of a call of that many rows."""
    if lamport_out.shape[0] < rows or lamport_out.shape[1] != hidden:
        raise RuntimeError(
            f"Lamport FC2 buffer {tuple(lamport_out.shape)} cannot hold {rows} x {hidden}"
        )
    return lamport_out[:rows]


def lamport_buffer_rows(
    runner: torch.ScriptObject,
    max_rows: int,
    hidden: int,
    inter: int,
    top_k: int,
    local_experts: int,
) -> int:
    """Rows of the largest Lamport call of up to ``max_rows`` tokens: at the largest tile the FP4 MoE ``runner``
    (``torch.classes.trtllm.FP4BlockScaleMoERunner``) offers for ``max_rows`` tokens (the tile-padded row count grows
    with tokens and tile, and so does the tile offered), or the CuTe FC2's MAX_TOKENS tokens at TILE_N."""
    tile = max(
        int(t[0]) for t in runner.get_valid_configs(top_k, hidden, inter, local_experts, max_rows)
    )
    return max(
        MAX_TOKENS * top_k + local_experts * (TILE_N - 1),
        max_num_tiles(max_rows, top_k, local_experts, tile) * tile,
    )


class LamportFc2Buffer:
    """The FC2 output buffer of the Lamport calls (up to ``max_rows`` tokens) of a model's MoE layers, which run one
    at a time and share it.

    It holds 0xFFFFFFFF wherever FC2 has not written: the MoE-output all-reduce's fold polls each word of the rows it
    reads (``MoEAllReduceParams.fc2_lamport``) instead of waiting for FC2, then writes the word back. Allocated at the
    first call, before CUDA-graph capture, and never reallocated, so graphs captured later keep a valid address."""

    max_rows = 0
    _buf: Optional[torch.Tensor] = None

    def __init__(self, max_rows: int) -> None:
        self.max_rows = max_rows
        self._buf: Optional[torch.Tensor] = None

    def get(
        self, hidden: int, inter: int, top_k: int, local_experts: int, device: torch.device
    ) -> torch.Tensor:
        """The buffer [>= lamport_buffer_rows(...), hidden] bf16 for the FP4 MoE layers of these shapes."""
        buf = self._buf
        if buf is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("allocate the Lamport FC2 buffer before CUDA graph capture")

            runner = torch.classes.trtllm.FP4BlockScaleMoERunner(int(ActType_TrtllmGen.SwiGlu))
            rows = lamport_buffer_rows(runner, self.max_rows, hidden, inter, top_k, local_experts)
            buf = torch.full((rows, hidden), -1, dtype=torch.int16, device=device).view(
                torch.bfloat16
            )
            self._buf = buf
        if buf.shape[1] != hidden:
            raise RuntimeError(
                f"Lamport FC2 buffer {tuple(buf.shape)} cannot hold {hidden} columns"
            )
        return buf
