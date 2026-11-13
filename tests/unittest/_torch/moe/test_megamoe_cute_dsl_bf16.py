# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unquantized BF16 MegaMoE CuteDSL: ``trtllm.cutedsl.mega_moe.none``.

Four groups, selected with ``-k``:

* ``cpu``: the BF16 weight method's FC1 rows (the gated ``[gate16, up16]``
  packing the NVFP4 leaf feeds the kernel; ReLU2's rows as loaded), the BF16
  leaf's host-side contract, the custom op's BF16 buffer, operand, tactic and
  activation rules and its problem keys, and the ``kind::f16`` instruction
  descriptor. No GPU.
* ``ep1``: one GPU. The kernels the op would launch, constructed but not
  compiled (extents, FC1 epilogue geometry, the fc1_done count, the Nemotron-3
  Ultra workspace plans); the backend against a float32 torch reference for
  every activation the leaf admits, ReLU2 up to top-k 32 and at Ultra's expert
  shape, SwiGLU at top-k 22, 0 to 3000 tokens; a quantized combine at top-k
  22; the custom op called directly, with the routing weights applied at
  FC1 or in the top-k reduce; and layers built through ``create_moe`` whose
  ``moe_max_num_tokens`` splits a forward into two to four launches.
* ``ep4``: four EP ranks on one node: DEP with uneven, single-token, zero-token
  and prefill-sized ranks, then TEP, for the gated activations and for ReLU2 at
  Ultra's 512 experts and top-22; then Nemotron-H's MoE layer built through
  ``create_moe`` with its DeepSeek-V3 router, DEP and TEP, in one launch and
  split into three by ``moe_max_num_tokens``. Runs as one process
  that spawns four MPI workers (the ``test_moe_module.py`` harness), or as four
  ranks started by ``mpirun -n 4`` / ``srun -n 4``, where every rank runs the
  body.
* ``cutlass``: backend-level parity with the CUTLASS BF16 backend on the same
  weights, routing and input.

``TRTLLM_TEST_MOE_CI=0`` (``IS_CI_MODE`` False) adds a long tier to ``ep1``,
``ep4`` and ``cutlass``: top-k 17, I=1216, repeated launches of the partial-tile
extents under the default and the mixed cluster-4 tactic, routing weights at
FC1 at Ultra's shape and at top-k 22 / 32, the fp4 combine, the full
candidate-tactic sweep, CUDA-graph replay, EP4 at Ultra's intermediate size,
ReLU2 under dynamic EPLB, and CUTLASS parity at Ultra's shape.

The reference is the kernel's arithmetic in float32 with its three bf16
stores: x (bf16) -> FC1 -> activation (of the gate and up halves for a gated
kind) -> (x the routing weight, when it is applied at FC1) -> bf16 (the
intermediate FC2 reads) -> FC2 -> bf16 (each route's row of the combine plane)
-> top-k sum, weighted unless FC1 applied the weights -> bf16 (the output).
What is left is accumulation order and the rare value that rounds to the other
side of a bf16 step, so the bounds sit near that floor (``_OVERALL_REL_L2`` /
``_PER_TOKEN_REL_L2``). A token whose routing weights are all zero must come
back exactly zero.

Selection is not under test here (``test_moe_impl.py`` covers it), so the
backends are built directly through ``create_moe_backend``; only the EP4
module-level runs go through ``create_moe``, as Nemotron-H does.
"""

import copy
import dataclasses
import math
import pickle
import sys
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cloudpickle
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from _torch.moe.moe_test_utils import IS_CI_MODE, MoeBackendType, get_backend_class
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op
from tensorrt_llm._torch.moe.fused_moe import (
    DeepSeekV3MoeRoutingMethod,
    RenormalizeMoeRoutingMethod,
    create_moe,
)
from tensorrt_llm._torch.moe.fused_moe.activation import (
    SimpleActivation,
    SiTuActivation,
    SwigluActivation,
    SwigluBiasActivation,
    activation_constant_names,
)
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoECommPlan,
    MoEDeployment,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import collect_moe_environment
from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind, MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.mega_moe import (
    TrtllmCutedslMegaMoeBf16Impl,
    TrtllmCutedslMegaMoeNvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4MegaMoECuteDslMethod,
    UnquantizedMegaMoECuteDslMethod,
)
from tensorrt_llm._torch.utils import ActivationType, Fp4QuantizedTensor, is_gated_activation
from tensorrt_llm._utils import get_sm_version, mpi_comm, mpi_rank, mpi_world_size
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig
from tensorrt_llm.mapping import Mapping

# pytest imports this file under its bare name, which spawned MPI workers
# cannot import; ship its functions by value, as test_moe_module.py does.
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# MegaMoE against ``_reference_moe``, which models every bf16 store of the
# kernel. Measured on GB200 over every case here: at most 6.3e-4 overall and
# 3.1e-3 for the worst token; the bounds keep about 3x of headroom. Without the
# two output roundings in the reference the floor would be about 2.3e-3, set by
# the bf16 output format alone, and a numerics regression of that size in one
# of the kernel's conversions would hide under it.
_OVERALL_REL_L2 = 2e-3
_PER_TOKEN_REL_L2 = 1e-2
# Nemotron-3 Ultra's expert shape (top-22, I 5120) sums 22 routes of a
# 5120-deep FC2 per token, so more of its bf16 stores land on the other side
# of a step: measured about 1e-3 overall (up to 1.2e-3) and up to 2.45e-3 for
# the worst token on GB200, gated and ReLU2 alike, about twice the floor of the
# smaller shapes. Its own overall bound keeps about 2.5x of headroom. 2e-3
# would keep 1.7x, and the overall error of a one-token case is that token's,
# which at this shape reaches 2.45e-3. The per-token bound is unchanged.
_ULTRA_OVERALL_REL_L2 = 3e-3
# Whatever involves the CUTLASS backend, which rounds at other points: the
# design bounds for a bf16 intermediate.
_CUTLASS_OVERALL_REL_L2 = 1e-2
_CUTLASS_PER_TOKEN_REL_L2 = 2e-2

# Long-tier-only tests; CI generates none of their cases.
_LONG_TIER = pytest.mark.skipif(IS_CI_MODE, reason="long tier: set TRTLLM_TEST_MOE_CI=0")


# =====================================================================
# Problem description and the float32 reference
# =====================================================================


@dataclasses.dataclass(frozen=True)
class _Shape:
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int

    def __str__(self) -> str:
        return f"e{self.num_experts}_k{self.top_k}_h{self.hidden_size}_i{self.intermediate_size}"


@dataclasses.dataclass(frozen=True)
class _ActivationCase:
    """One expert activation: the layer's carrier and its float32 math."""

    name: str
    build: Callable[[], object]
    # Gated: (gate, up) float32 FC1 halves -> float32 activation. Non-gated:
    # the float32 FC1 output -> float32 activation.
    reference: Callable[..., torch.Tensor]
    # Scales the input; > 1 pushes the FC1 outputs across a clamp.
    input_scale: float = 1.0

    @property
    def gated(self) -> bool:
        """Whether FC1 is a [gate, up] pair (w1 and w3) or one projection (w1).

        Read off the carrier, as the layer reads its expand ratio, so a case
        cannot claim a layout its layer does not build.
        """
        return is_gated_activation(ActivationType(self.build().kind))

    def __str__(self) -> str:
        return self.name


def _swiglu(clamp: Optional[float]) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if clamp is not None:
            gate = gate.clamp(max=clamp)
            up = up.clamp(min=-clamp, max=clamp)
        return F.silu(gate) * up

    return reference


def _minimax_swiglu_bias(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate = gate.clamp(max=7.0)
    up = up.clamp(min=-7.0, max=7.0)
    return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_softcap, linear_softcap = 4.0, 8.0
    capped_gate = gate_softcap * torch.tanh(gate / gate_softcap) * torch.sigmoid(gate)
    return capped_gate * (linear_softcap * torch.tanh(up / linear_softcap))


def _relu2(fc1: torch.Tensor) -> torch.Tensor:
    # The kernel's fmax(x, 0) also maps NaN to 0; the inputs here are finite.
    return F.relu(fc1).square()


_SWIGLU = _ActivationCase("swiglu", SwigluActivation, _swiglu(None))
_SWIGLU_CLAMP = _ActivationCase(
    "swiglu_clamp", lambda: SwigluActivation(clamp=7.0), _swiglu(7.0), input_scale=4.0
)
# MiniMax-M3: the SwigluBias package without expert bias.
_MINIMAX = _ActivationCase(
    "minimax_swiglu_bias",
    lambda: SwigluBiasActivation(gate_sigmoid_scale=1.702, linear_offset=1.0, clamp=7.0),
    _minimax_swiglu_bias,
    input_scale=4.0,
)
# Kimi K3. A soft cap below the FC1 output range so both caps saturate.
_SITU = _ActivationCase(
    "situ",
    lambda: SiTuActivation(gate_softcap=4.0, linear_softcap=8.0),
    _situ,
    input_scale=4.0,
)
# Nemotron-H: non-gated, no constants. Half the FC1 outputs land below zero.
_RELU2 = _ActivationCase("relu2", lambda: SimpleActivation(kind=ActivationType.Relu2), _relu2)

_SMALL = _Shape(num_experts=8, top_k=2, hidden_size=1024, intermediate_size=1024)
# Nemotron-3 Ultra's routed experts: top-22 of 512 on the 2048-wide latent, I
# 5120, with the 32 experts one DEP16 rank holds.
_ULTRA = _Shape(num_experts=32, top_k=22, hidden_size=2048, intermediate_size=5120)


def _overall_rel_l2(shape: _Shape) -> float:
    """``_OVERALL_REL_L2``, or Ultra's at its expert shape (any expert count)."""
    ultra = (shape.top_k, shape.hidden_size, shape.intermediate_size) == (
        _ULTRA.top_k,
        _ULTRA.hidden_size,
        _ULTRA.intermediate_size,
    )
    return _ULTRA_OVERALL_REL_L2 if ultra else _OVERALL_REL_L2


@dataclasses.dataclass
class _Weights:
    """Checkpoint-style per-expert weights and the stacked copies the reference reads."""

    checkpoint: Dict[str, torch.Tensor]
    w1: torch.Tensor  # (experts, intermediate, hidden): the gate, or a non-gated kind's only FC1
    w3: torch.Tensor  # (experts, intermediate, hidden) up; (experts, 0, hidden) when non-gated
    w2: torch.Tensor  # (experts, hidden, intermediate)


# Layers with at least this many weights per projection draw them on the GPU's
# own generator: the CPU draw (about 1.1 G values for the EP4 ReLU2 layer, on
# every rank) would dominate the test. Philox is deterministic for one GPU
# model, so the EP ranks of one node still hold the same experts.
_DEVICE_DRAW_MIN_WEIGHTS = 1 << 28


def _make_weights(
    shape: _Shape, *, seed: int, device: torch.device, gated: bool = True
) -> _Weights:
    """Weights scaled by fan-in, so FC1 outputs are O(1) and FC2 outputs too.

    Drawn from a fixed seed, so every EP rank holds the same experts. Below
    ``_DEVICE_DRAW_MIN_WEIGHTS`` the draws run on the CPU, which they name
    explicitly: the tests call this under a CUDA ``torch.device`` context,
    which would otherwise move them to the GPU and reject the CPU generator.

    ``gated`` False builds a non-gated layer: w1 is its only FC1 projection
    and w3 is empty, in the checkpoint too (``w3 = w1[e][:0]``), the form the
    Nemotron-H weight mapper produces from ``up_proj``.
    """
    experts, hidden, inter = shape.num_experts, shape.hidden_size, shape.intermediate_size
    draw_device = (
        torch.device(device)
        if experts * inter * hidden >= _DEVICE_DRAW_MIN_WEIGHTS
        else torch.device("cpu")
    )
    generator = torch.Generator(device=draw_device).manual_seed(seed)

    def draw(*size: int, fan_in: int) -> torch.Tensor:
        weight = torch.randn(*size, generator=generator, device=draw_device) * fan_in**-0.5
        return weight.to(dtype=torch.bfloat16, device=device)

    w1 = draw(experts, inter, hidden, fan_in=hidden)
    w3 = draw(experts, inter, hidden, fan_in=hidden) if gated else w1[:, :0]
    w2 = draw(experts, hidden, inter, fan_in=inter)
    checkpoint = {}
    for expert in range(experts):
        checkpoint[f"{expert}.w1.weight"] = w1[expert]
        checkpoint[f"{expert}.w3.weight"] = w3[expert]
        checkpoint[f"{expert}.w2.weight"] = w2[expert]
    return _Weights(checkpoint, w1, w3, w2)


def _make_tokens(
    num_tokens: int,
    shape: _Shape,
    case: _ActivationCase,
    *,
    seed: int,
    device: torch.device,
    zero_weight_token: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Input rows, top-k distinct expert ids per token, and softmax routing weights.

    Drawn on the CPU (see ``_make_weights``), then moved to ``device``.
    """
    generator = torch.Generator().manual_seed(seed)
    cpu = torch.device("cpu")
    x = torch.randn(num_tokens, shape.hidden_size, generator=generator, device=cpu)
    x = x * case.input_scale
    expert_ids = torch.rand(num_tokens, shape.num_experts, generator=generator, device=cpu)
    expert_ids = expert_ids.argsort(dim=1)[:, : shape.top_k]
    routing_weights = torch.softmax(
        torch.randn(num_tokens, shape.top_k, generator=generator, device=cpu), 1
    )
    if zero_weight_token is not None and zero_weight_token < num_tokens:
        routing_weights[zero_weight_token] = 0.0
    return (
        x.to(dtype=torch.bfloat16, device=device),
        expert_ids.to(dtype=torch.int32, device=device),
        routing_weights.to(dtype=torch.float32, device=device),
    )


def _bf16(t: torch.Tensor) -> torch.Tensor:
    """Round float32 to bf16 and back, as a bf16 store and reload does."""
    return t.to(torch.bfloat16).float()


def _reference_moe(
    x: torch.Tensor,
    weights: _Weights,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    case: _ActivationCase,
    *,
    apply_topk_in_fc1: bool = False,
) -> torch.Tensor:
    """float32 MoE over global expert ids, rounded where the kernel stores bf16.

    The kernel hands FC2 a bf16 intermediate, writes each route's FC2 output
    to the bf16 combine plane, and reduces the top-k routes in float32
    (TopkReduce) into a bf16 output. The routing weight scales each route in
    that reduce, or, with ``apply_topk_in_fc1`` (DeepSeek-V4's placement), the
    float32 activation before its bf16 store, and the reduce sums unweighted.
    """
    x32 = x.float()
    out = torch.zeros(x.shape[0], weights.w2.shape[1], dtype=torch.float32, device=x.device)
    for expert in expert_ids.unique().tolist():
        token, slot = (expert_ids == expert).nonzero(as_tuple=True)
        rows = x32[token]
        fc1 = rows @ weights.w1[expert].float().t()
        if case.gated:
            activated = case.reference(fc1, rows @ weights.w3[expert].float().t())
        else:
            activated = case.reference(fc1)
        scales = routing_weights[token, slot].unsqueeze(1)
        if apply_topk_in_fc1:
            activated = activated * scales
        expert_out = _bf16(_bf16(activated) @ weights.w2[expert].float().t())
        out.index_add_(0, token, expert_out if apply_topk_in_fc1 else expert_out * scales)
    return _bf16(out)


def _compare(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    zero_tokens: Sequence[int] = (),
    overall_bound: float = _OVERALL_REL_L2,
    per_token_bound: float = _PER_TOKEN_REL_L2,
) -> List[str]:
    """Failure messages for ``output`` against ``reference``; empty on a pass."""
    if output.shape != reference.shape:
        return [f"output shape {tuple(output.shape)} != reference {tuple(reference.shape)}"]
    failures = []
    output = output.float()
    if not torch.isfinite(output).all():
        failures.append("output has non-finite values")
    keep = torch.ones(output.shape[0], dtype=torch.bool, device=output.device)
    for token in zero_tokens:
        keep[token] = False
        if not torch.all(output[token] == 0):
            failures.append(f"token {token} has zero routing weight but a non-zero output")
    if keep.any():
        diff = output[keep] - reference[keep]
        overall = (diff.norm() / reference[keep].norm().clamp_min(1e-30)).item()
        per_token = diff.norm(dim=1) / reference[keep].norm(dim=1).clamp_min(1e-30)
        worst = per_token.max().item()
        if overall > overall_bound:
            failures.append(f"relative L2 {overall:.3e} > {overall_bound}")
        if worst > per_token_bound:
            failures.append(
                f"per-token relative L2 {worst:.3e} > {per_token_bound} "
                f"at kept token {per_token.argmax().item()}"
            )
    return failures


# =====================================================================
# Backends
# =====================================================================


def _model_config(
    shape: _Shape,
    mapping: Mapping,
    *,
    moe_backend: str,
    max_num_tokens: int,
    moe_load_balancer: Optional[MoeLoadBalancerConfig] = None,
    moe_max_num_tokens: Optional[int] = None,
) -> ModelConfig:
    """The model config of test_moe_backend.py's ``create_test_backend``, for an unquantized layer.

    ``moe_max_num_tokens`` caps the tokens per rank of one MoE launch; ``None``
    keeps ModelConfig's default (``max_num_tokens`` times the attention-DP size).
    """
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = shape.num_experts
    pretrained_config.hidden_size = shape.hidden_size
    pretrained_config.intermediate_size = shape.intermediate_size
    pretrained_config.torch_dtype = torch.bfloat16
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        moe_backend=moe_backend,
        max_num_tokens=max_num_tokens,
        moe_load_balancer=moe_load_balancer,
        moe_max_num_tokens=moe_max_num_tokens,
    )


def _make_backend(
    moe_cls: type,
    shape: _Shape,
    case: _ActivationCase,
    mapping: Mapping,
    *,
    moe_backend: str,
    max_num_tokens: int,
):
    """The ``create_test_backend`` recipe of test_moe_backend.py, for an unquantized layer."""
    return create_moe_backend(
        moe_cls=moe_cls,
        routing_method=RenormalizeMoeRoutingMethod(top_k=shape.top_k),
        num_experts=shape.num_experts,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        dtype=torch.bfloat16,
        model_config=_model_config(
            shape, mapping, moe_backend=moe_backend, max_num_tokens=max_num_tokens
        ),
        activation=case.build(),
    )


def _load(backend, weights: _Weights):
    backend.load_weights([weights.checkpoint])
    backend.post_load_weights()
    backend.cuda()
    return backend


def _moe_layer(shape: _Shape, case: _ActivationCase, routing_method, model_config: ModelConfig):
    """A routed-experts layer as a model builds it through ``create_moe``: no expert bias, the
    all-reduce left to the caller (``reduce_results=False``), no degradation, so the layer is
    the BF16 leaf's or nothing."""
    return create_moe(
        routing_method=routing_method,
        num_experts=shape.num_experts,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        dtype=torch.bfloat16,
        reduce_results=False,
        model_config=model_config,
        activation=case.build(),
        allow_backend_degradation=False,
    )


def _record_launches(backend) -> List[int]:
    """The token count of every ``run_moe`` call ``backend`` gets from here on; the calls still
    run the kernel."""
    launches = []
    run_moe = backend.run_moe

    def recording_run_moe(ctx: MoERunContext, **kwargs) -> torch.Tensor:
        launches.append(int(ctx.x.shape[0]))
        return run_moe(ctx, **kwargs)

    backend.run_moe = recording_run_moe
    return launches


def _run(
    backend, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor
) -> torch.Tensor:
    """``quantize_input`` then ``run_moe``, with what each scheduler kind hands the backend.

    Mirrors ``run_backend_moe`` in test_moe_backend.py: a fused-comm backend
    takes int64 ids and no comm plan; an external-comm one gets the single-GPU
    no-comm plan.
    """
    x_in, x_sf = backend.quantize_input(x, post_quant_comm=False)
    fused = backend.scheduler_kind == MoESchedulerKind.FUSED_COMM
    context = MoERunContext(
        token_selected_experts=expert_ids.to(torch.int64 if fused else torch.int32),
        token_final_scales=routing_weights,
        x=x_in,
        x_sf=x_sf,
        output_dtype=torch.bfloat16,
        comm_plan=(
            None
            if fused
            else MoECommPlan(
                input_sf_swizzled=True,
                enable_alltoall=False,
                moe_output=None,
                payload_in_workspace=False,
            )
        ),
    )
    return backend.run_moe(context)


# A machine without the leaf's kernel or one of its dependencies. Any other
# ``can_implement`` rejection is the leaf turning down a layer it is meant to
# serve, and a skip would hide that regression.
_MACHINE_REJECTIONS = frozenset({MoERejectReason.SM_UNSUPPORTED, MoERejectReason.DEP_MISSING})


def _skip_unless_bf16_megamoe_runs(shape: _Shape, case: _ActivationCase, ep_size: int) -> None:
    """Skip where this machine cannot run the BF16 leaf; fail where the leaf declines the layer.

    Skips only for no CUDA, too few GPUs, an SM the leaf has no kernel for, or
    a missing dependency.
    """
    if not torch.cuda.is_available():
        pytest.skip("MegaMoE CuteDSL BF16 needs CUDA")
    if torch.cuda.device_count() < ep_size:
        pytest.skip(f"needs {ep_size} GPUs")
    activation = case.build()
    kind = ActivationType(activation.kind)
    verdict = TrtllmCutedslMegaMoeBf16Impl.can_implement(
        MoEProblem(
            quant=None,
            dtype_act=torch.bfloat16,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            swiglu_gptoss_style=kind is ActivationType.SwigluBias,
            bias=False,
            activation=kind.name,
            activation_constants=activation_constant_names(activation),
        ),
        MoEDeployment(
            ep_size=ep_size,
            tp_size=1,
            parallel_size=ep_size,
            use_dp=ep_size > 1,
            num_slots=shape.num_experts,
            env=collect_moe_environment(),
        ),
    )
    if verdict.eligible:
        return
    if verdict.reject_reason in _MACHINE_REJECTIONS:
        pytest.skip(verdict.detail)
    pytest.fail(
        f"the BF16 leaf declines {shape}-{case} at ep_size={ep_size}: "
        f"{verdict.reject_reason.value}: {verdict.detail}"
    )


def _skip_under_multi_rank_launch() -> None:
    # Under ``mpirun -n 4`` every rank runs every selected test, and a
    # single-rank MegaMoE cannot be built once the EP ranks share a WORLD group.
    if mpi_world_size() > 1:
        pytest.skip("single-GPU test; run it without a multi-rank launcher")


def _kernel_fc1_weight(weights: _Weights, case: _ActivationCase) -> torch.Tensor:
    """FC1 ``(experts, expand, hidden)`` in the row order the BF16 weight method loads it.

    Goes through the method itself: the gated ``[gate16, up16]`` interleave,
    or a non-gated FC1's rows as loaded.
    """
    experts, inter, hidden = weights.w1.shape
    fc1_weight = torch.empty(
        experts,
        (2 if case.gated else 1) * inter,
        hidden,
        dtype=torch.bfloat16,
        device=weights.w1.device,
    )
    method = UnquantizedMegaMoECuteDslMethod()
    layer = SimpleNamespace(tp_size=1, tp_rank=0, is_gated_activation=case.gated)
    for expert in range(experts):
        method.load_expert_w3_w1_weight(
            layer, weights.w1[expert], weights.w3[expert], fc1_weight[expert]
        )
    return fc1_weight


def _staged_op_inputs(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    max_tokens: int,
) -> Dict[str, torch.Tensor]:
    """One rank's launch inputs padded to ``max_tokens`` the way the backend stages them.

    The padded tail routes nowhere (``topk_idx`` -1) with weight 0.
    """
    num_tokens, top_k = expert_ids.shape
    activation = torch.zeros(max_tokens, x.shape[1], dtype=torch.bfloat16, device=x.device)
    activation[:num_tokens] = x
    topk_idx = torch.full((max_tokens, top_k), -1, dtype=torch.int64, device=x.device)
    topk_idx[:num_tokens] = expert_ids
    topk_weights = torch.zeros(max_tokens, top_k, dtype=torch.float32, device=x.device)
    topk_weights[:num_tokens] = routing_weights
    return dict(activation=activation, topk_idx=topk_idx, topk_weights=topk_weights)


# =====================================================================
# CPU: layout and host-side contract
# =====================================================================


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_fc1_rows_match_the_nvfp4_mega_packing():
    """The BF16 method feeds the kernel the FC1 row order the NVFP4 leaf feeds it.

    The gated epilogue reads ``[gate16, up16]`` row blocks whatever the operand
    format, so the bf16 rows must be permuted exactly like the NVFP4 bytes in
    ``NVFP4MegaMoECuteDslMethod._build_mega_format_buffers``. Each row carries
    its own index, exact in bf16 up to 256.
    """
    # hidden and intermediate must be multiples of the NVFP4 scale block (16) for the mega packing.
    intermediate, hidden = 128, 64
    rows = torch.arange(2 * intermediate, dtype=torch.float32)
    # Checkpoint order: w3 is up, w1 is gate; TRT-LLM stacks them [w3 | w1].
    w3 = rows[:intermediate, None].expand(intermediate, hidden).to(torch.bfloat16)
    w1 = rows[intermediate:, None].expand(intermediate, hidden).to(torch.bfloat16)
    dst = torch.empty(2 * intermediate, hidden, dtype=torch.bfloat16)
    UnquantizedMegaMoECuteDslMethod().load_expert_w3_w1_weight(
        SimpleNamespace(tp_size=1, tp_rank=0, is_gated_activation=True), w1, w3, dst
    )
    bf16_order = dst[:, 0].to(torch.int64)

    nvfp4_method = NVFP4MegaMoECuteDslMethod.__new__(NVFP4MegaMoECuteDslMethod)
    mega_fc1, *_ = nvfp4_method._build_mega_format_buffers(
        raw_w3_w1=rows.to(torch.uint8)[None, :, None].expand(1, 2 * intermediate, hidden // 2),
        raw_w3_w1_sf=torch.zeros(1, 2 * intermediate, hidden // 16, dtype=torch.uint8),
        raw_w2=torch.zeros(1, hidden, intermediate // 2, dtype=torch.uint8),
        raw_w2_sf=torch.zeros(1, hidden, intermediate // 16, dtype=torch.uint8),
        num_slots=1,
        intermediate=intermediate,
        hidden=hidden,
        expand_intermediate=2 * intermediate,
    )
    nvfp4_order = mega_fc1[0, :, 0].to(torch.int64)

    assert torch.equal(bf16_order, nvfp4_order)
    # Gate rows (from w1) lead every 32-row block.
    assert bf16_order[:16].tolist() == list(range(intermediate, intermediate + 16))
    assert bf16_order[16:32].tolist() == list(range(16))


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_relu2_fc1_rows_load_as_is():
    """A non-gated FC1 reaches the kernel exactly as loaded.

    Relu2 has one FC1 row per feature: w1 (``up_proj``) with an empty w3, as
    the Nemotron-H weight mapper hands it over. The gated ``[gate16, up16]``
    interleave applied to these rows would permute the features silently.
    """
    intermediate, hidden = 128, 8
    w1 = torch.randn(intermediate, hidden, generator=torch.Generator().manual_seed(0))
    w1 = w1.to(torch.bfloat16)
    dst = torch.empty(intermediate, hidden, dtype=torch.bfloat16)
    UnquantizedMegaMoECuteDslMethod().load_expert_w3_w1_weight(
        SimpleNamespace(tp_size=1, tp_rank=0, is_gated_activation=False), w1, w1[:0], dst
    )
    assert torch.equal(dst, w1)


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_weight_method_shapes_and_refusals():
    method = UnquantizedMegaMoECuteDslMethod()
    # A gated activation's FC1 interleave needs both halves of an expert in
    # one load, so its partial loads (an RL weight sync's buckets) are refused
    # before anything is copied; Relu2 takes them (see the EP1 hot-reload test).
    with pytest.raises(NotImplementedError, match="gated activation's experts whole"):
        method.load_weights(
            SimpleNamespace(is_gated_activation=True),
            {},
            MoEWeightLoadingMode.VANILLA,
            allow_partial_loading=True,
        )

    def module(**overrides) -> torch.nn.Module:
        layer = torch.nn.Module()
        fields = dict(
            expert_size_per_partition=2,
            hidden_size=64,
            intermediate_size_per_partition=64,
            expand_intermediate_size_per_partition=128,
            dtype=torch.bfloat16,
            bias=False,
        )
        fields.update(overrides)
        for name, value in fields.items():
            setattr(layer, name, value)
        return layer

    with pytest.raises(ValueError, match="bfloat16"):
        method.create_weights(module(dtype=torch.float16))

    # FC1 holds expand rows: a gated activation's [gate, up] pairs (2 x 64),
    # Relu2's one row per feature (64). No dtype means bf16, as the backend
    # reads it for its output.
    for expand in (128, 64):
        layer = module(dtype=None, expand_intermediate_size_per_partition=expand)
        UnquantizedMegaMoECuteDslMethod().create_weights(layer)
        assert layer.w3_w1_weight.shape == (2, expand, 64)
        assert layer.w2_weight.shape == (2, 64, 64)
        assert layer.w3_w1_weight.dtype == layer.w2_weight.dtype == torch.bfloat16


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_hot_reload_refusals():
    """Which leaves take an RL weight sync in place, and the reason the others give.

    Relu2 BF16 reloads into the tensors the kernel reads; a gated activation
    regroups FC1 per expert as it loads, and NVFP4 frees its source weights after
    packing, so both refuse in ``pre_reload_weights``, before any weight changes.
    """
    bf16 = TrtllmCutedslMegaMoeBf16Impl._hot_reload_refusal
    assert bf16(SimpleNamespace(is_gated_activation=False)) is None
    assert "gate/up interleave" in bf16(SimpleNamespace(is_gated_activation=True))
    nvfp4 = TrtllmCutedslMegaMoeNvfp4Impl._hot_reload_refusal
    assert "releases its source weights" in nvfp4(SimpleNamespace())

    refusing = SimpleNamespace(_hot_reload_refusal=lambda: "for a test leaf")
    with pytest.raises(NotImplementedError, match="hot weight reloading for a test leaf"):
        TrtllmCutedslMegaMoeBf16Impl.pre_reload_weights(refusing)


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_leaf_host_contract():
    """Pass-through input, no scale-factor operands, the kernel reads w3_w1 / w2."""
    # ``__new__``: the constructor would want a GPU for its kernel probes.
    impl = TrtllmCutedslMegaMoeBf16Impl.__new__(TrtllmCutedslMegaMoeBf16Impl)
    torch.nn.Module.__init__(impl)

    x = torch.randn(5, 64, dtype=torch.float16)
    x_in, x_sf = impl.quantize_input(x)
    assert x_sf is None
    assert x_in.dtype == torch.bfloat16 and x_in.is_contiguous()
    assert torch.equal(x_in, x.to(torch.bfloat16))
    empty, empty_sf = impl.quantize_input(torch.empty(0, 64, dtype=torch.bfloat16))
    assert empty.shape == (0, 64) and empty_sf is None
    with pytest.raises(NotImplementedError, match="Fp4QuantizedTensor"):
        impl.quantize_input(
            Fp4QuantizedTensor(torch.empty(1, 32, dtype=torch.uint8), torch.empty(1, 4))
        )
    with pytest.raises(ValueError, match="no activation scale factors"):
        impl.run_moe(
            MoERunContext(
                token_selected_experts=torch.zeros(5, 2, dtype=torch.int64),
                token_final_scales=torch.ones(5, 2),
                x=x_in,
                x_sf=torch.empty(5, 4, dtype=torch.uint8),
            )
        )

    impl.w3_w1_weight = torch.nn.Parameter(
        torch.empty(2, 128, 64, dtype=torch.bfloat16), requires_grad=False
    )
    impl.w2_weight = torch.nn.Parameter(
        torch.empty(2, 64, 64, dtype=torch.bfloat16), requires_grad=False
    )
    view = impl._build_weight_view()
    assert view.fc1_weight is impl.w3_w1_weight
    assert view.fc2_weight is impl.w2_weight
    scale_fields = ("fc1_weight_sf", "fc2_weight_sf", "fc31_alpha", "fc2_alpha", "fc1_norm_const")
    assert all(getattr(view, field) is None for field in scale_fields)

    # Staged rows are the bf16 wire bytes; the kernel gets them back as bf16.
    staged = torch.zeros(4, megamoe_op.megamoe_activation_bytes_per_row(64, "bf16"))
    staged = staged.to(torch.uint8)
    operands = impl._kernel_operands(staged, None, view)
    assert operands.activation.dtype == torch.bfloat16
    assert operands.activation.shape == (4, 64)
    assert operands.activation.data_ptr() == staged.data_ptr()
    assert operands.fc1_weight is view.fc1_weight
    assert operands.fc2_weight is view.fc2_weight
    assert operands.activation_sf is None
    assert operands.fc1_weight_sf is None and operands.fc2_weight_sf is None
    assert operands.fc1_alpha is None and operands.fc2_alpha is None
    assert operands.fc1_norm_const is None


# The fields of ``_megamoe_problem_key`` for a small gated layer.
_PROBLEM_KEY_FIELDS = dict(
    sm_version=100,
    world_size=1,
    num_topk=2,
    num_experts_per_rank=8,
    hidden_size=1024,
    intermediate_size_per_partition=1024,
    expand_intermediate_size_per_partition=2048,
    max_tokens_per_rank=256,
    output_dtype=torch.bfloat16,
    apply_topk_in_fc1=False,
    gate_up_clamp=None,
    situ_beta=None,
    situ_linear_beta=None,
    in_kernel_fc2_reduce=False,
    combine_format="bf16",
)


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_buffer_layout_and_tactics():
    assert megamoe_op.megamoe_activation_bytes_per_row(2048, "bf16") == 4096
    assert megamoe_op.megamoe_activation_bytes_per_row(2048, "nvfp4") == 1024
    assert megamoe_op.megamoe_quant_kind_is_block_scaled("nvfp4")
    assert not megamoe_op.megamoe_quant_kind_is_block_scaled("bf16")
    # The kernel defines more kinds than the op launches.
    with pytest.raises(ValueError, match="quant_kind"):
        megamoe_op.megamoe_activation_bytes_per_row(2048, "mxfp8_e4m3")

    problem = dict(
        apply_topk_in_fc1=False,
        in_kernel_fc2_reduce=False,
    )
    for sm in (100, 103):
        for num_tokens in (1, 64, 1024, 4096, 16384):
            nvfp4 = megamoe_op.enumerate_megamoe_candidate_tactics(num_tokens, sm_version=sm)
            bf16 = megamoe_op.enumerate_megamoe_candidate_tactics(
                num_tokens, sm_version=sm, quant_kind="bf16"
            )
            # The same space, at the BF16 tile K.
            assert len(bf16) == len(nvfp4)
            assert {tuple(tactic[0]) for tactic in bf16} <= {(256, 256, 64), (256, 128, 64)}
            # The load-balanced space adds only token_tile READY, which is NVFP4-only.
            assert (
                megamoe_op.enumerate_megamoe_candidate_tactics(
                    num_tokens, sm_version=sm, quant_kind="bf16", load_balance=True
                )
                == bf16
            )
            for combine_format in ("bf16", "32e4m3xe8m0"):
                defaults = {
                    quant_kind: megamoe_op._default_megamoe_tactic_for_problem(
                        sm_version=sm,
                        max_tokens_per_rank=max(num_tokens, 256),
                        num_tokens=num_tokens,
                        combine_format=combine_format,
                        quant_kind=quant_kind,
                        **problem,
                    )
                    for quant_kind in ("nvfp4", "bf16")
                }
                assert defaults["nvfp4"][0][2] == 256
                assert defaults["bf16"][0][:2] == defaults["nvfp4"][0][:2]
                assert defaults["bf16"][0][2] == 64
                assert defaults["bf16"][1:] == defaults["nvfp4"][1:]
                megamoe_op.validate_megamoe_tactic(
                    defaults["bf16"], sm_version=sm, quant_kind="bf16"
                )
                # A rejected heuristic must fall back at the BF16 tile K too.
                fallbacks = megamoe_op._megamoe_default_tactic_candidates(defaults["bf16"])
                assert all(tactic[0][2] == 64 for tactic in fallbacks)

    # The EP tests cover both default tiles: buckets up to 1024 tokens run
    # N=128, the 2048 / 4096 buckets N=256.
    for bucket, tile in ((256, 128), (1024, 128), (2048, 256), (4096, 256)):
        default = megamoe_op._default_megamoe_tactic_for_problem(
            sm_version=100,
            max_tokens_per_rank=bucket,
            num_tokens=bucket,
            combine_format="bf16",
            quant_kind="bf16",
            **problem,
        )
        assert list(default[0]) == [256, tile, 64], bucket

    # token_tile READY is NVFP4-only: a tactic that meets every other condition.
    token_tile = (
        [256, 256, 256],
        [2, 1, 1],
        None,
        ("grouped", 512),
        "atomic_counter",
        "reuse_dispatch_warps",
        True,
        2,
        4,
        (2, 4),
        "token_tile",
    )
    megamoe_op.validate_megamoe_tactic(token_tile)
    with pytest.raises(ValueError, match="quant_kind='bf16'"):
        megamoe_op.validate_megamoe_tactic(
            megamoe_op._megamoe_tactic_with_mma_tile_k(token_tile, 64), quant_kind="bf16"
        )

    # No Rubin BF16 mainloop.
    with pytest.raises(ValueError, match="no SM107 kernel"):
        megamoe_op.enumerate_megamoe_candidate_tactics(64, sm_version=107, quant_kind="bf16")
    with pytest.raises(ValueError, match="no SM107 kernel"):
        megamoe_op.validate_megamoe_tactic(
            megamoe_op.default_megamoe_tactic(64), sm_version=107, quant_kind="bf16"
        )

    # The kind keys the compile, workspace and tuning caches.
    key = dict(_PROBLEM_KEY_FIELDS)
    # NVFP4 keeps the key persisted AutoTuner caches were recorded under; any
    # other kind appends itself, so the two never share an entry.
    nvfp4_key = megamoe_op._megamoe_problem_key(**key)
    bf16_key = megamoe_op._megamoe_problem_key(**key, quant_kind="bf16")
    assert len(nvfp4_key) == 18 and "nvfp4" not in nvfp4_key
    assert megamoe_op._megamoe_problem_key(**key, quant_kind="nvfp4") == nvfp4_key
    assert bf16_key == (*nvfp4_key, "bf16")
    assert megamoe_op._megamoe_problem_key_quant_kind(nvfp4_key) == "nvfp4"
    assert megamoe_op._megamoe_problem_key_quant_kind(bf16_key) == "bf16"
    # NVFP4's shape-tuned SM107 table is keyed by that historical key.
    dsv4_pro = dict(
        key,
        sm_version=107,
        world_size=4,
        num_topk=6,
        num_experts_per_rank=96,
        hidden_size=7168,
        intermediate_size_per_partition=3072,
        expand_intermediate_size_per_partition=6144,
        apply_topk_in_fc1=True,
        gate_up_clamp=10.0,
    )
    assert megamoe_op._megamoe_problem_key(**dsv4_pro) in megamoe_op._SM107_DSV4_PRO_DEFAULT_TACTICS


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_problem_key_carries_only_a_non_gated_activation():
    """Only a non-gated kind adds a token, so every gated key stays the one it was.

    Persisted AutoTuner caches and ``_SM107_DSV4_PRO_DEFAULT_TACTICS`` are
    keyed by the 18-field NVFP4 key (ending in the helper count) and the BF16
    runs by its 19-field form; Relu2 appends its kind after the quant kind,
    which it then always carries, so position 18 stays the quant kind the
    default tactic is read from.
    """
    key = dict(_PROBLEM_KEY_FIELDS)
    nvfp4_key = megamoe_op._megamoe_problem_key(**key)
    for gated in ("swiglu", "situ"):
        assert megamoe_op._megamoe_problem_key(**key, expert_activation=gated) == nvfp4_key
        assert megamoe_op._megamoe_problem_key(
            **key, quant_kind="bf16", expert_activation=gated
        ) == (*nvfp4_key, "bf16")

    # Relu2: one FC1 row per feature, so expand == intermediate.
    relu2_fields = dict(key, expand_intermediate_size_per_partition=1024)
    relu2_key = megamoe_op._megamoe_problem_key(
        **relu2_fields, quant_kind="bf16", expert_activation="relu2"
    )
    assert relu2_key == (*megamoe_op._megamoe_problem_key(**relu2_fields), "bf16", "relu2")
    assert all(type(field) is str for field in relu2_key[18:])
    assert megamoe_op._megamoe_problem_key_quant_kind(relu2_key) == "bf16"
    # The token forces the quant kind into position 18 even for the default kind.
    assert megamoe_op._megamoe_problem_key(**relu2_fields, expert_activation="relu2")[18:] == (
        "nvfp4",
        "relu2",
    )
    with pytest.raises(ValueError, match="expert_activation"):
        megamoe_op._megamoe_problem_key(**key, quant_kind="bf16", expert_activation="gelu")

    # Relu2 takes the BF16 defaults: the token sits past every field they read.
    for num_tokens in (256, 4096, 16384):
        default = megamoe_op._default_megamoe_tactic_for_key(relu2_key, num_tokens=num_tokens)
        assert default == megamoe_op._default_megamoe_tactic_for_problem(
            sm_version=100,
            max_tokens_per_rank=256,
            num_tokens=num_tokens,
            apply_topk_in_fc1=False,
            in_kernel_fc2_reduce=False,
            combine_format="bf16",
            quant_kind="bf16",
        )
        assert default[0][2] == 64


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_op_activation_table():
    """The op's restated activation table agrees with TRT-LLM and with what each leaf admits.

    The gated flag decides the FC1 extent the op checks (``expand == 2 * I``)
    and must be the one the layer derived its expand ratio from; every kind a
    leaf admits has to resolve, or the layer would fail at its first launch.
    """
    table = megamoe_op._MEGAMOE_EXPERT_ACTIVATIONS
    for kind, entry in table.items():
        assert entry.gated == is_gated_activation(kind), kind
        assert type(entry.kernel_value) is str, kind
    assert table[ActivationType.Relu2] == ("relu2", False)
    bf16_kinds = TrtllmCutedslMegaMoeBf16Impl.activation_support.kinds
    nvfp4_kinds = TrtllmCutedslMegaMoeNvfp4Impl.activation_support.kinds
    assert ActivationType.Relu2 in bf16_kinds
    assert bf16_kinds <= set(table) and nvfp4_kinds <= set(table)
    # The block-scaled FC1 epilogue re-quantizes gated outputs only.
    assert all(table[kind].gated for kind in nvfp4_kinds)


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_op_activation_table_matches_the_kernel_enum():
    """``_MEGAMOE_EXPERT_ACTIVATIONS`` restates the kernel's ``ExpertActivation`` without the DSL."""
    activation_def = pytest.importorskip(
        "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.activation_def"
    )
    expert_activation = activation_def.ExpertActivation
    table = megamoe_op._MEGAMOE_EXPERT_ACTIVATIONS
    assert {entry.kernel_value for entry in table.values()} == {
        kind.value for kind in expert_activation
    }
    for kind, entry in table.items():
        assert expert_activation(entry.kernel_value).is_gated == entry.gated, kind
    # FC2's K from FC1's rows, which the op passes as intermediate_gateup_size.
    assert expert_activation.relu2.intermediate_downproj(5120) == 5120
    assert expert_activation.swiglu.intermediate_downproj(10240) == 5120
    # The value keys the kernel name, so it must print as itself.
    assert str(expert_activation.relu2) == f"{expert_activation.relu2}" == "relu2"
    # A descriptor without the field keeps the kinds that predate it.
    assert expert_activation.from_problem_desc({}) is expert_activation.swiglu
    assert expert_activation.from_problem_desc({"situ_beta": 2.0}) is expert_activation.situ
    declared = {"expert_activation": "relu2", "situ_beta": None}
    assert expert_activation.from_problem_desc(declared) is expert_activation.relu2


def _resolve_expert_activation(activation_type, **overrides):
    """``_resolve_megamoe_expert_activation`` for a 64-wide layer; gated bf16 extents by default."""
    arguments = dict(
        quant_kind="bf16",
        intermediate_size_per_partition=64,
        expand_intermediate_size_per_partition=128,
        gate_up_clamp=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        situ_beta=None,
        situ_linear_beta=None,
    )
    arguments.update(overrides)
    return megamoe_op._resolve_megamoe_expert_activation(activation_type, **arguments)


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_op_resolves_the_expert_activation():
    situ_betas = dict(situ_beta=2.0, situ_linear_beta=3.0)
    # None keeps the op's contract from before it took activation_type: the
    # SiTU betas select SiTU, anything else runs SwiGLU.
    assert _resolve_expert_activation(None).kernel_value == "swiglu"
    assert _resolve_expert_activation(None, quant_kind="nvfp4").kernel_value == "swiglu"
    assert _resolve_expert_activation(None, gate_up_clamp=7.0).kernel_value == "swiglu"
    assert _resolve_expert_activation(None, **situ_betas).kernel_value == "situ"
    # A given kind is authoritative. Swiglu and SwigluBias both run the kernel's swiglu.
    for kind, constants, kernel_value in (
        (ActivationType.Swiglu, dict(gate_up_clamp=7.0), "swiglu"),
        (
            ActivationType.SwigluBias,
            dict(gate_up_clamp=7.0, swiglu_alpha=1.702, swiglu_beta=1.0),
            "swiglu",
        ),
        (ActivationType.SiTu, situ_betas, "situ"),
    ):
        resolved = _resolve_expert_activation(int(kind), **constants)
        assert resolved == (kernel_value, True), kind
    relu2 = _resolve_expert_activation(
        int(ActivationType.Relu2), expand_intermediate_size_per_partition=64
    )
    assert relu2 == ("relu2", False)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "activation_type, overrides, match",
    [
        # The FC1 extent must be the kind's: a gated kernel over a non-gated
        # FC1, or the reverse, would read the wrong rows instead of failing.
        pytest.param(
            ActivationType.Relu2, {}, "expand_intermediate_size_per_partition", id="relu2_2x_rows"
        ),
        pytest.param(
            ActivationType.Swiglu,
            dict(expand_intermediate_size_per_partition=64),
            "expand_intermediate_size_per_partition",
            id="swiglu_1x_rows",
        ),
        pytest.param(
            ActivationType.Relu2,
            dict(expand_intermediate_size_per_partition=64, gate_up_clamp=7.0),
            "gate_up_clamp",
            id="relu2_clamp",
        ),
        pytest.param(
            ActivationType.Relu2,
            dict(expand_intermediate_size_per_partition=64, swiglu_alpha=1.702, swiglu_beta=1.0),
            "swiglu_alpha",
            id="relu2_swiglu_constants",
        ),
        pytest.param(
            ActivationType.Relu2,
            dict(expand_intermediate_size_per_partition=64, situ_beta=2.0, situ_linear_beta=3.0),
            "situ_beta",
            id="relu2_situ_constants",
        ),
        # The non-gated FC1 epilogue stores a bf16 intermediate only.
        pytest.param(
            ActivationType.Relu2,
            dict(expand_intermediate_size_per_partition=64, quant_kind="nvfp4"),
            "nvfp4",
            id="relu2_nvfp4",
        ),
        pytest.param(
            ActivationType.SiTu, dict(situ_beta=2.0), "situ_linear_beta", id="situ_one_beta"
        ),
        pytest.param(
            ActivationType.Swiglu,
            dict(situ_beta=2.0, situ_linear_beta=3.0),
            "situ_beta",
            id="swiglu_situ_betas",
        ),
        pytest.param(
            ActivationType.Gelu,
            dict(expand_intermediate_size_per_partition=64),
            "Gelu",
            id="gelu",
        ),
    ],
)
def test_megamoe_bf16_cpu_op_refuses_an_activation_it_cannot_run(activation_type, overrides, match):
    """A ValueError before any kernel or tactic is built; a tactic fallback never sees it."""
    with pytest.raises(ValueError, match=match):
        _resolve_expert_activation(int(activation_type), **overrides)


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_op_refuses_a_kernel_that_ignores_the_activation(
    monkeypatch: pytest.MonkeyPatch,
):
    """A kernel that does not declare ``expert_activation`` never reads it.

    That is a re-import of the kernel package without the non-gated FC1
    epilogue: it would run the gated epilogue over ReLU2's FC1 rows, so the op
    refuses to build it instead.
    """
    package = pytest.importorskip("tensorrt_llm._torch.moe.kernels.megamoe_kernels")
    kernel_cls = package.BlackwellInferenceMegaMoE
    declared = kernel_cls.problem_desc_require()
    assert "expert_activation" in declared
    stale = {name: value for name, value in declared.items() if name != "expert_activation"}
    monkeypatch.setattr(kernel_cls, "problem_desc_require", classmethod(lambda cls: stale))
    common = dict(static_expert_shape=(8, 1024, 1024), quant_kind="bf16", expert_activation="relu2")
    with pytest.raises(ValueError, match="does not take expert_activation"):
        megamoe_op._construct_megamoe_kernel(sm_version=100, common=common, combine_format="bf16")


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_symmetric_regions(monkeypatch: pytest.MonkeyPatch):
    """bf16 rows are 2 * hidden bytes and there is no SF region; NVFP4 is unchanged."""
    if not megamoe_op.IS_MEGAMOE_OP_AVAILABLE:
        pytest.skip(str(megamoe_op.MEGAMOE_OP_UNAVAILABLE_REASON))
    # The provider only needs a buffer and peer pointers from symmetric memory.
    monkeypatch.setattr(megamoe_op.dist, "is_available", lambda: True)
    monkeypatch.setattr(megamoe_op.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        megamoe_op,
        "torch_symm_mem",
        SimpleNamespace(
            enable_symm_mem_for_group=lambda group_name: None,
            empty=lambda size, device, dtype: torch.empty(size, dtype=dtype),
            rendezvous=lambda buf, group_name: SimpleNamespace(buffer_ptrs=[buf.data_ptr()] * 2),
        ),
    )
    hidden, max_tokens = 2048, 256
    regions = {
        quant_kind: megamoe_op.MegaMoeSymmMemProvider(
            process_group=SimpleNamespace(group_name="fake"),
            world_size=2,
            rank=0,
            hidden_size=hidden,
            max_tokens_per_rank=max_tokens,
            num_topk=4,
            output_dtype=torch.bfloat16,
            shared_workspace_bytes=4096,
            quant_kind=quant_kind,
        ).get_regions()
        for quant_kind in ("nvfp4", "bf16")
    }
    nvfp4, bf16 = regions["nvfp4"], regions["bf16"]
    assert nvfp4.activation.shape == (max_tokens, hidden // 2)
    assert nvfp4.activation_sf.shape == (
        max_tokens,
        megamoe_op.megamoe_activation_sf_bytes_per_row(hidden),
    )
    assert bf16.activation.shape == (max_tokens, 2 * hidden)
    assert bf16.activation.view(torch.bfloat16).shape == (max_tokens, hidden)
    assert bf16.activation_sf is None
    # The regions after the activation do not overlap it.
    activation_end = bf16.activation.data_ptr() + bf16.activation.numel()
    assert bf16.topk_weights.data_ptr() >= activation_end
    assert bf16.base_buf.numel() > nvfp4.base_buf.numel()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu, id="swiglu"),
        pytest.param(ActivationType.Relu2, id="relu2"),
    ],
)
def test_megamoe_bf16_cpu_single_rank_staging(
    monkeypatch: pytest.MonkeyPatch, activation_type: ActivationType
):
    """Single-rank staging: bf16 byte rows, no SF buffer, the bf16 kernel's workspace.

    The workspace probe sizes the kernel of the layer's activation: without
    ``activation_type`` it would infer SwiGLU from the constants, and a Relu2
    layer's workspace would be sized for a kernel it never runs.
    """
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl

    if not megamoe_op.IS_MEGAMOE_OP_AVAILABLE:
        pytest.skip(str(megamoe_op.MEGAMOE_OP_UNAVAILABLE_REASON))
    probed = []

    def query_megamoe_kernel_properties(**kwargs):
        probed.append(kwargs)
        return 4096, False

    monkeypatch.setattr(
        megamoe_op, "query_megamoe_kernel_properties", query_megamoe_kernel_properties
    )
    monkeypatch.setattr(mega_moe_cute_dsl, "get_sm_version", lambda: 100)

    gated = is_gated_activation(activation_type)
    impl = TrtllmCutedslMegaMoeBf16Impl.__new__(TrtllmCutedslMegaMoeBf16Impl)
    torch.nn.Module.__init__(impl)
    for name, value in dict(
        ep_size=1,
        expert_size_per_partition=8,
        intermediate_size_per_partition=64,
        intermediate_size_expand_ratio=2 if gated else 1,
        apply_topk_in_fc1=False,
        act_clamp=None,
        act_alpha=None,
        act_beta=None,
        activation_type=activation_type,
        combine_format="bf16",
        layer_load_balancer=None,
        _topk_idx_dtype=torch.int64,
        _local_staging_cache={},
        _launch_buffers={},
        _last_staged_T={},
        _genphase_direct_input_buckets=set(),
    ).items():
        setattr(impl, name, value)

    hidden, max_tokens = 64, 16
    buffers = impl._acquire_buffers(
        top_k=2, hidden=hidden, device="cpu", output_dtype=torch.bfloat16, launch_max_T=max_tokens
    )
    assert [kwargs["quant_kind"] for kwargs in probed] == ["bf16"]
    assert probed[0]["activation_type"] == int(activation_type)
    assert probed[0]["expand_intermediate_size_per_partition"] == (128 if gated else 64)
    assert buffers.activation.shape == (max_tokens, 2 * hidden)
    assert buffers.activation.dtype == torch.uint8
    assert buffers.activation_sf is None
    assert buffers.shared_workspace.numel() == 4096

    x = torch.randn(5, hidden).to(torch.bfloat16)
    impl._stage_inputs(
        bufs=buffers,
        x=x,
        x_sf=None,
        topk_idx=torch.zeros(5, 2, dtype=torch.int64),
        topk_weights=torch.ones(5, 2),
        num_tokens=5,
        top_k=2,
        stage_activation=True,
    )
    view = SimpleNamespace(fc1_weight=None, fc2_weight=None)
    staged = impl._kernel_operands(buffers.activation, buffers.activation_sf, view).activation
    assert torch.equal(staged[:5], x)
    # The padded tail stays masked out of the dispatch.
    assert torch.all(buffers.topk_idx_local[5:] == -1)
    assert torch.all(buffers.topk_weights[5:] == 0)


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_op_formats_match_quant_kind():
    """The op's restated operand formats agree with the kernel's ``QuantKind``."""
    quant_def = pytest.importorskip("tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.quant_def")
    for kind, operand_format in megamoe_op._MEGAMOE_OPERAND_FORMATS.items():
        quant_kind = quant_def.QuantKind(kind)
        assert int(quant_kind.activation_dtype.width) == operand_format.activation_bits, kind
        assert quant_kind.is_block_scaled == operand_format.block_scaled, kind
        assert quant_kind.uses_global_scale == operand_format.global_scale, kind


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_op_operand_sets():
    """Each kind takes exactly its optional operands; NVFP4 never runs unscaled."""
    given = torch.empty(1)
    scale_factors = dict(activation_sf=given, fc1_weight_sf=given, fc2_weight_sf=given)
    global_scales = dict(fc1_alpha=given, fc2_alpha=given, fc1_norm_const=given)
    nvfp4 = {**scale_factors, **global_scales}
    bf16 = {name: None for name in nvfp4}
    megamoe_op._check_megamoe_operand_set("nvfp4", **nvfp4)
    megamoe_op._check_megamoe_operand_set("bf16", **bf16)
    for name in nvfp4:
        # The kernel would accept a missing global scale and skip the rescale.
        with pytest.raises(ValueError, match=f"nvfp4 requires .*{name}"):
            megamoe_op._check_megamoe_operand_set("nvfp4", **{**nvfp4, name: None})
        with pytest.raises(ValueError, match=f"{name}.* must be None"):
            megamoe_op._check_megamoe_operand_set("bf16", **{**bf16, name: given})


@pytest.mark.cpu_only
def test_megamoe_bf16_cpu_dense_idesc_packs_umma_instr_descriptor():
    """The ``kind::f16`` instruction descriptor, against CUTLASS's bit layout.

    A wrong format code, shift or reserved bit still compiles and runs, and
    only shows up as wrong numbers on a GPU. The expected value restates
    ``UMMA::InstrDescriptor`` (cute/arch/mma_sm100_desc.hpp) as (value, width)
    fields from bit 0 up, independently of the builder's shift constants.
    """
    quant_def = pytest.importorskip("tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.quant_def")
    mainloop = pytest.importorskip(
        "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.blackwell"
        ".inference.mega.dynamic_mainloop"
    )
    kind = quant_def.QuantKind.bf16

    def instr_descriptor(m: int, n: int) -> int:
        fields = (
            (0, 2),  # sparse_id2_
            (0, 1),  # sparse_flag_: dense
            (0, 1),  # saturate_
            (1, 2),  # c_format_: F32
            (0, 1),  # reserved
            (1, 3),  # a_format_: F16F32Format BF16
            (1, 3),  # b_format_: F16F32Format BF16
            (0, 1),  # a_negate_
            (0, 1),  # b_negate_
            (0, 1),  # a_major_: K-major
            (0, 1),  # b_major_: K-major
            (n >> 3, 6),  # n_dim_
            (0, 1),  # reserved
            (m >> 4, 5),  # m_dim_
            (0, 1),  # reserved
            (0, 2),  # max_shift_
        )
        desc = bit = 0
        for value, width in fields:
            assert 0 <= value < 1 << width
            desc |= value << bit
            bit += width
        assert bit == 32
        return desc

    for m, base in ((128, 0x08000490), (256, 0x10000490)):
        static = mainloop.build_static_idesc_base_f16(
            umma_m=m, a_format=kind.weight_format_code, b_format=kind.activation_format_code
        )
        # The value the builder's docstring and the compiled PTX carry.
        assert static == base, hex(static)
        # The kernel ORs in N = valid tokens rounded up to 16 at run time.
        for n in (16, 48, 128, 240, 256):
            assert static | ((n >> 3) << mainloop._BIT_N_DIM) == instr_descriptor(m, n), (m, n)


# =====================================================================
# EP1 host plans: the kernels the op launches, constructed, not compiled
# =====================================================================

# Nemotron-3 Ultra's per-rank expert count under DEP16, top-k and latent hidden.
_PLAN_LOCAL_EXPERTS = 32
_PLAN_TOP_K = 22
_PLAN_HIDDEN = 2048
# (activation, FC2's K). ReLU2 at the extents where an FC1 CTA tile can lie
# entirely past the last feature -- FC1 rows mod (128 * cluster M) in (0,
# 128 * (cluster M - 1)]: 1088 and 1152 at cluster M 2, 1216 at 4 -- and at
# Ultra's 5120; gated controls at 1152 and 5120.
_PLAN_CASES = (
    (_RELU2, 1088),
    (_RELU2, 1152),
    (_RELU2, 1216),
    (_RELU2, 5120),
    (_SWIGLU, 1152),
    (_SWIGLU, 5120),
)
# max_tokens_per_rank of the builds that take the bucket's default tactic: the
# N=128 tile up to 1024 tokens, N=256 up to 8192, the prefill tactic above.
_PLAN_BUCKETS = (256, 1024, 4096, 16384)
# Token counts whose candidate tactics are added when their cluster or fallback
# shape is not the defaults' (2, 1): the mixed (4, 1) / (2, 1) phase-interleave
# candidate autotuning reaches at >= 8192 tokens.
_PLAN_CANDIDATE_TOKENS = (256, 16384)
# The long tier builds every candidate of these token counts.
_PLAN_SWEEP_TOKENS = (256, 1024, 4096, 16384)


def _plan_runner(
    case: _ActivationCase,
    intermediate: int,
    *,
    max_tokens: int,
    world_size: int = 1,
    local_experts: int = _PLAN_LOCAL_EXPERTS,
    top_k: int = _PLAN_TOP_K,
    hidden: int = _PLAN_HIDDEN,
):
    """The op's runner for one bf16 layer of ``case`` (no activation constants) with FC2's K =
    ``intermediate``. Its ``_build_kernel_once`` constructs the kernel the op would launch
    without compiling it or allocating anything."""
    return megamoe_op.Sm100MegaMoENvfp4Runner(
        world_size=world_size,
        local_rank=0,
        num_topk=top_k,
        num_experts_per_rank=local_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        expand_intermediate_size_per_partition=(2 if case.gated else 1) * intermediate,
        max_tokens_per_rank=max_tokens,
        output_dtype=torch.bfloat16,
        apply_topk_in_fc1=False,
        quant_kind="bf16",
        activation_type=int(ActivationType(case.build().kind)),
    )


def _current_cuda_context() -> None:
    """Make the device's primary context current.

    The runner sizes a launch with an occupancy query, which needs one; a
    test that is the first in its process to touch the GPU has none yet.
    """
    torch.zeros(1, device=torch.device("cuda", torch.cuda.current_device()))


def _bucket_default_tactic(runner) -> Tuple:
    """The tactic ``forward`` resolves for a launch padded to the runner's max_tokens_per_rank."""
    return megamoe_op._default_megamoe_tactic_for_key(
        runner.unique_id(), num_tokens=runner.max_tokens_per_rank
    )


def _fc1_done_tiles(kernel) -> Tuple[List[int], List[int], int]:
    """For one (expert, token tile) slot: the FC1 CTA tiles that add to its fc1_done counter,
    the tiles that hold FC1 rows, and the count FC2 waits for.

    Every tile_m_idx the scheduler can hand out -- cluster block b and CTA rank
    r give b * cluster_m + r (fc12_mapping ``_decode_inside_expert`` /
    ``_decode_phase_work_id``) -- goes through the predicate
    ``signal_fc1_done`` evaluates on device. A tile without FC1 rows that
    signals completes the counter before the last features land: an
    intermittent wrong result no launch is guaranteed to catch.
    """
    # The device predicate and its compile-time fold, as signal_fc1_done calls them.
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.blackwell.inference.mega.block_scaled_swap_ab_fc12_epilogue import (  # noqa: E501
        fc1_signal_guard_needed,
        fc1_tile_signals,
    )

    epilogue = kernel.epilogue
    cta_tile_m = kernel._mainloop.cta_tile_m
    cluster_m = epilogue.cluster_shape_mn[0]
    fc1_rows = kernel.intermediate_gateup_size
    extent, width = epilogue.intermediate_downproj, epilogue.fc1_output_tile_size
    tiles = range(math.ceil(fc1_rows / (cta_tile_m * cluster_m)) * cluster_m)
    guard = fc1_signal_guard_needed(extent, width, cluster_m)
    signalling = [m for m in tiles if not guard or fc1_tile_signals(m, extent, width)]
    with_rows = [m for m in tiles if m * cta_tile_m < fc1_rows]
    # The kernel's fc2_spin_threshold: 128-row FC1 CTA tiles per slot.
    spin_threshold = math.ceil(fc1_rows / cta_tile_m)
    return signalling, with_rows, spin_threshold


def _kernel_plan_failures(kernel, case: _ActivationCase, intermediate: int, tag: str) -> List[str]:
    """What the kernel's components must agree on before a launch, as failure messages."""
    epilogue, mainloop, scheduler = kernel.epilogue, kernel._mainloop, kernel.scheduler
    failures = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(f"{tag}: {message}")

    kind = megamoe_op._MEGAMOE_EXPERT_ACTIVATIONS[ActivationType(case.build().kind)].kernel_value
    kinds = tuple(str(part.expert_activation) for part in (kernel, mainloop, epilogue))
    check(kinds == (kind,) * 3, f"K / ML / EP expert_activation {kinds} != {kind}")
    # FC2's K is the FC1 output width; any copy left at FC1_M // 2 drops half of it.
    downproj = tuple(part.intermediate_downproj for part in (kernel, mainloop, epilogue))
    check(downproj == (intermediate,) * 3, f"K / ML / EP intermediate_downproj {downproj}")
    region = kernel._device_workspace.region(kernel.fc1_output_region)
    check(region.shape[1] == intermediate, f"fc1_output region {region.shape}")
    expected = (64, 16, 8192) if case.gated else (128, 32, 16384)
    geometry = (
        epilogue.fc1_output_tile_size,
        epilogue.fc1_output_features_per_warp,
        epilogue.fc1_staging_stage_bytes,
    )
    check(
        geometry == expected,
        f"FC1 output tile / features per warp / stage bytes {geometry} != {expected}",
    )
    cluster_m = epilogue.cluster_shape_mn[0]
    check(
        epilogue.cluster_tile_intermediate_downproj == epilogue.fc1_output_tile_size * cluster_m,
        f"cluster tile {epilogue.cluster_tile_intermediate_downproj} at cluster M {cluster_m}",
    )
    # The enumeration in _fc1_done_tiles restates the scheduler's swap-AB
    # mapping: its N axis walks 128-row FC1 tiles, in clusters of cluster M.
    check(
        scheduler.mapping_cta_tile_shape_mnk[1] == mainloop.cta_tile_m
        and scheduler.mapping_cluster_shape_mn[1] == cluster_m,
        f"scheduler tile {scheduler.mapping_cta_tile_shape_mnk} cluster "
        f"{scheduler.mapping_cluster_shape_mn}",
    )
    signalling, with_rows, spin_threshold = _fc1_done_tiles(kernel)
    check(
        signalling == with_rows and len(signalling) == spin_threshold,
        f"cluster M {cluster_m}: tiles {signalling} signal fc1_done, tiles {with_rows} hold "
        f"FC1 rows, FC2 waits for {spin_threshold}",
    )
    check(
        not epilogue.overlapping_accum and not mainloop.overlapping_accum,
        "overlapping accumulator",
    )
    name = kernel.name()
    check(
        name.endswith("_actrelu2") == (not case.gated) and ("_act" in name) == (not case.gated),
        f"name ...{name[-40:]}",
    )
    return failures


@dataclasses.dataclass
class _PlanBuilds:
    """What a matrix of kernel builds found."""

    failures: List[str] = dataclasses.field(default_factory=list)
    rejected: List[str] = dataclasses.field(default_factory=list)
    # (max_tokens, tactic) -> {shared-workspace bytes}: one value per key.
    shared_bytes: Dict[Tuple[int, str], set] = dataclasses.field(default_factory=dict)

    def build(self, case: _ActivationCase, intermediate: int, max_tokens: int, tactic, *, required):
        """Build one kernel of the matrix; ``tactic`` None takes the bucket default.

        A tactic may turn a shape down with ValueError, which makes the op
        fall back; ``required`` builds (the shipped defaults) may not.
        NotImplementedError is never a tactic's answer.
        """
        runner = _plan_runner(case, intermediate, max_tokens=max_tokens)
        tactic = _bucket_default_tactic(runner) if tactic is None else tactic
        tag = f"{case} I={intermediate} max_T={max_tokens} tactic={tactic}"
        try:
            kernel = runner._build_kernel_once(tactic)
        except ValueError as error:
            (self.failures if required else self.rejected).append(f"{tag}: ValueError: {error}")
            return
        except NotImplementedError as error:
            self.failures.append(f"{tag}: NotImplementedError: {error}")
            return
        self.failures += _kernel_plan_failures(kernel, case, intermediate, tag)
        _, shared = kernel.get_workspace_sizes()
        self.shared_bytes.setdefault((max_tokens, repr(tactic)), set()).add(shared)

    def shared_bytes_failures(self) -> List[str]:
        # The shared regions belong to the token exchange alone, so gated and
        # ReLU2 kernels of one (hidden, experts, top-k, max_T, world) size them alike.
        return [
            f"max_T={max_tokens} tactic={tactic}: shared-workspace bytes {sorted(sizes)}"
            for (max_tokens, tactic), sizes in self.shared_bytes.items()
            if len(sizes) != 1
        ]


def _is_mixed_cluster(tactic: Tuple) -> bool:
    """Whether a candidate's cluster or fallback shape differs from the defaults' (2, 1)."""
    _, cluster, fallback, *_ = megamoe_op._unpack_tactic(tactic)
    return list(cluster[:2]) != [2, 1] or (fallback is not None and list(fallback[:2]) != [2, 1])


def test_megamoe_bf16_ep1_host_kernel_extents_and_fc1_done():
    """Per kernel the op would build: FC2's K, the FC1 output geometry and the fc1_done count.

    Covers ReLU2 and gated controls under the default tactic of every bucket
    and the mixed-cluster candidate. FC2's K and the FC1 output width must be
    I in the kernel, mainloop and epilogue, and every FC1 CTA tile that holds
    rows -- and only those -- must signal fc1_done, as many as FC2 waits for.
    """
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(_ULTRA, _RELU2, ep_size=1)
    _current_cuda_context()
    sm_version = get_sm_version()
    candidates = [
        (num_tokens, tactic)
        for num_tokens in _PLAN_CANDIDATE_TOKENS
        for tactic in megamoe_op.enumerate_megamoe_candidate_tactics(
            num_tokens, sm_version=sm_version, quant_kind="bf16"
        )
        if _is_mixed_cluster(tactic)
    ]
    assert candidates, "no candidate tactic with a cluster other than (2, 1)"
    builds = _PlanBuilds()
    for case, intermediate in _PLAN_CASES:
        for bucket in _PLAN_BUCKETS:
            builds.build(case, intermediate, bucket, None, required=True)
        for num_tokens, tactic in candidates:
            builds.build(case, intermediate, num_tokens, tactic, required=False)
    for rejection in builds.rejected:
        print(f"[rejected] {rejection}")
    failures = builds.failures + builds.shared_bytes_failures()
    assert not failures, "\n".join(failures)


@_LONG_TIER
def test_megamoe_bf16_ep1_host_full_candidate_sweep():
    """``test_megamoe_bf16_ep1_host_kernel_extents_and_fc1_done`` over every candidate tactic."""
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(_ULTRA, _RELU2, ep_size=1)
    _current_cuda_context()
    sm_version = get_sm_version()
    builds = _PlanBuilds()
    for case, intermediate in _PLAN_CASES:
        for num_tokens in _PLAN_SWEEP_TOKENS:
            for tactic in megamoe_op.enumerate_megamoe_candidate_tactics(
                num_tokens, sm_version=sm_version, quant_kind="bf16"
            ):
                builds.build(case, intermediate, num_tokens, tactic, required=False)
    for rejection in builds.rejected:
        print(f"[rejected] {rejection}")
    failures = builds.failures + builds.shared_bytes_failures()
    assert not failures, "\n".join(failures)


# (label, world size, local experts, hidden) of Nemotron-3 Ultra's 512 experts:
# DEP16, TEP8 (attention TP8 + MoE EP8, so 64 local experts), and that TEP8
# split at a 7168 hidden, where ReLU2's larger FC1 staging leaves the fewest
# mainloop stages.
_ULTRA_PLANS = (
    ("dep16", 16, 32, 2048),
    ("tep8", 8, 64, 2048),
    ("tep8_h7168", 8, 64, 7168),
)
# The engine's default ladder: Blackwell's 256 / 1024 / 4096 rungs plus the
# default max_num_tokens, 8192.
_ULTRA_PLAN_BUCKETS = (256, 1024, 4096, 8192)


def test_megamoe_bf16_ep1_host_ultra_plans(record_property):
    """Every bucket's default tactic builds as is for Ultra's ReLU2 layer, with no fallback.

    Records the local and the symmetric (shared) workspace bytes per bucket,
    which together with the weights decide the moe_max_num_tokens an Ultra
    deployment can afford (the default ladder does not fit next to BF16
    weights under DEP16).
    """
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(_ULTRA, _RELU2, ep_size=1)
    _current_cuda_context()
    intermediate = _ULTRA.intermediate_size
    failures = []
    for label, world_size, local_experts, hidden in _ULTRA_PLANS:
        for bucket in _ULTRA_PLAN_BUCKETS:
            runner = _plan_runner(
                _RELU2,
                intermediate,
                max_tokens=bucket,
                world_size=world_size,
                local_experts=local_experts,
                hidden=hidden,
            )
            tactic = _bucket_default_tactic(runner)
            tag = f"{label} max_T={bucket} tactic={tactic}"
            try:
                kernel = runner._build_kernel_once(tactic)
            except (ValueError, NotImplementedError) as error:
                failures.append(f"{tag}: {type(error).__name__}: {error}")
                continue
            failures += _kernel_plan_failures(kernel, _RELU2, intermediate, tag)
            local_bytes, shared_bytes = kernel.get_workspace_sizes()
            record_property(f"{label}_maxt{bucket}_local_bytes", local_bytes)
            record_property(f"{label}_maxt{bucket}_shared_bytes", shared_bytes)
            print(
                f"[plan] {label} max_T={bucket}: local {local_bytes / 1e9:.2f} GB, "
                f"shared {shared_bytes / 1e9:.3f} GB"
            )
    assert not failures, "\n".join(failures)


def test_megamoe_bf16_ep1_host_topk_reduce_loads_scores_per_route_above_16():
    """Top-k above 16 takes TopkReduce's per-route score load, the path Ultra's top-22 runs."""
    quant_def = pytest.importorskip("tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.quant_def")
    topk_reduce = pytest.importorskip(
        "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.topk_reduce"
    )
    for combine_format in ("bf16", "32e4m3xe8m0", "16e2m1xbf16"):
        parsed = quant_def.CombineFormat.parse(combine_format)
        assert topk_reduce.TopkReduce(_PLAN_HIDDEN, 16, parsed).prefetch, combine_format
        for top_k in (17, 22, 32):
            reduce = topk_reduce.TopkReduce(_PLAN_HIDDEN, top_k, parsed)
            assert not reduce.prefetch, (combine_format, top_k)


# =====================================================================
# EP1: one GPU against the float32 reference
# =====================================================================

# 0 returns without a launch; 1 and 7 are below one 16-token MMA N step; 7 and
# 1000 are not multiples of 16; 1000 lands in the second adaptive bucket.
_EP1_TOKEN_COUNTS = (0, 1, 7, 64, 256, 1000)
# Lands in the 4096 bucket, whose default tile is (256, 256, 64) where the
# smaller buckets run (256, 128, 64). Two cases take it, to bound the compiles.
_EP1_PREFILL_TOKENS = 2500
# Buckets 256, 1024 and 4096 (see ``_resolve_maxt_buckets``).
_EP1_MAX_NUM_TOKENS = 4096

_EP1_CASES = [
    pytest.param(shape, case, token_counts, id=f"{shape}-{case}")
    for shape, case, token_counts in (
        (_SMALL, _SWIGLU, _EP1_TOKEN_COUNTS),
        (_SMALL, _SWIGLU_CLAMP, _EP1_TOKEN_COUNTS),
        (_SMALL, _MINIMAX, _EP1_TOKEN_COUNTS),
        (_SMALL, _SITU, _EP1_TOKEN_COUNTS + (_EP1_PREFILL_TOKENS,)),
        (
            _Shape(num_experts=4, top_k=4, hidden_size=2048, intermediate_size=1024),
            _SWIGLU,
            _EP1_TOKEN_COUNTS,
        ),
        (
            _Shape(num_experts=8, top_k=8, hidden_size=2048, intermediate_size=1536),
            _SWIGLU,
            _EP1_TOKEN_COUNTS + (_EP1_PREFILL_TOKENS,),
        ),
        # ReLU2, the non-gated epilogue: 33 is a partial 64-token subtile.
        (_SMALL, _RELU2, (0, 1, 7, 33, 64, 256, 1000)),
        # FC1 CTA tile 9 of 10 holds no rows at cluster M 2 (1152 = 9 x 128):
        # the fc1_done guard; 3000 takes the 4096 bucket's N=256 tile.
        (
            _Shape(num_experts=8, top_k=4, hidden_size=1024, intermediate_size=1152),
            _RELU2,
            (1, 64, 256, 1000, 3000),
        ),
        # 1088 = 8.5 x 128: tile 8's second 64-feature box lies past the end.
        (
            _Shape(num_experts=8, top_k=4, hidden_size=1024, intermediate_size=1088),
            _RELU2,
            (64, 256),
        ),
        # Top-k above 16: TopkReduce's per-route score load. 32 is the leaf's cap.
        (
            _Shape(num_experts=64, top_k=22, hidden_size=1024, intermediate_size=1024),
            _RELU2,
            (1, 64, 256),
        ),
        (
            _Shape(num_experts=64, top_k=32, hidden_size=1024, intermediate_size=1024),
            _RELU2,
            (64,),
        ),
        # The cap belongs to the leaf, so gated layers get it too.
        (
            _Shape(num_experts=64, top_k=22, hidden_size=1024, intermediate_size=1024),
            _SWIGLU,
            (64,),
        ),
        (_ULTRA, _RELU2, (1, 7, 64, 256, 1000, 3000)),
        # Long tier: the first top-k past TopkReduce's prefetch limit (with an
        # even max_T, so its index loads stay inside topk_idx), and 1216 under
        # the default tactics; its FC1 CTA tiles without rows appear only at
        # cluster M 4, which test_megamoe_bf16_ep1_relu2_partial_tiles_repeated runs.
        *(
            ()
            if IS_CI_MODE
            else (
                (
                    _Shape(num_experts=64, top_k=17, hidden_size=1024, intermediate_size=1024),
                    _RELU2,
                    (1, 64, 256),
                ),
                (
                    _Shape(num_experts=8, top_k=4, hidden_size=1024, intermediate_size=1216),
                    _RELU2,
                    (1, 64, 256, 1000, 3000),
                ),
            )
        ),
    )
]


@pytest.mark.parametrize("shape, case, token_counts", _EP1_CASES)
def test_megamoe_bf16_ep1_matches_fp32_reference(
    shape: _Shape, case: _ActivationCase, token_counts: Tuple[int, ...]
):
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(shape, case, ep_size=1)
    device = torch.device("cuda", torch.cuda.current_device())
    failures = []
    with torch.device(device):
        weights = _make_weights(shape, seed=0, device=device, gated=case.gated)
        backend = _load(
            _make_backend(
                TrtllmCutedslMegaMoeBf16Impl,
                shape,
                case,
                Mapping(),
                moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
                max_num_tokens=_EP1_MAX_NUM_TOKENS,
            ),
            weights,
        )
        for num_tokens in token_counts:
            zero_tokens = (0,) if num_tokens > 1 else ()
            x, expert_ids, routing_weights = _make_tokens(
                num_tokens,
                shape,
                case,
                seed=num_tokens,
                device=device,
                zero_weight_token=zero_tokens[0] if zero_tokens else None,
            )
            # The lockstep chunk maximum the fused-comm scheduler hands over.
            backend.set_adaptive_launch_tokens(num_tokens)
            with torch.inference_mode():
                output = _run(backend, x, expert_ids, routing_weights)
                reference = _reference_moe(x, weights, expert_ids, routing_weights, case)
            failures += [
                f"T={num_tokens}: {message}"
                for message in _compare(
                    output,
                    reference,
                    zero_tokens=zero_tokens,
                    overall_bound=_overall_rel_l2(shape),
                )
            ]
    assert not failures, "\n".join(failures)


# An RL weight sync on Relu2 (Nemotron-H's experts), the llmapi/rlhf_utils.py
# sequence: pre_reload_weights, partial loads, process_weights_after_loading,
# post_load_weights. The shape has FC1 CTA tiles without rows (1152 = 9 x 128).
_RELOAD_SHAPE = _Shape(num_experts=8, top_k=4, hidden_size=1024, intermediate_size=1152)
_RELOAD_TOKENS = (7, 256, 1000)


def test_megamoe_bf16_ep1_relu2_hot_reload_matches_reference(monkeypatch: pytest.MonkeyPatch):
    """A loaded Relu2 layer takes new weights in place.

    A full sync in two buckets (every FC1 with the even experts' FC2, then the
    odd experts' FC2) leaves the layer on the new weights' reference, and a graph
    captured before the sync replays on them. A sync that carries only some
    experts' FC2 leaves the others' FC2 as they were.
    """
    shape, case = _RELOAD_SHAPE, _RELU2
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(shape, case, ep_size=1)
    # Scope the capture state to this test (see the CUDA-graph replay test).
    monkeypatch.setattr(megamoe_op, "_MEGAMOE_CAPTURED_SHARED_PTRS", set())
    monkeypatch.setattr(megamoe_op, "_MEGAMOE_GRAPH_CAPTURE_SEEN", False)
    device = torch.device("cuda", torch.cuda.current_device())
    failures = []
    with torch.device(device):
        old = _make_weights(shape, seed=11, device=device, gated=False)
        new = _make_weights(shape, seed=12, device=device, gated=False)
        backend = _load(
            _make_backend(
                TrtllmCutedslMegaMoeBf16Impl,
                shape,
                case,
                Mapping(),
                moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
                max_num_tokens=_EP1_MAX_NUM_TOKENS,
            ),
            old,
        )
        even = list(range(0, shape.num_experts, 2))
        odd = list(range(1, shape.num_experts, 2))
        every = list(range(shape.num_experts))

        def bucket(source: _Weights, experts: List[int], projections: Tuple[str, ...]) -> Dict:
            return {
                f"{e}.{p}.weight": source.checkpoint[f"{e}.{p}.weight"]
                for e in experts
                for p in projections
            }

        def sync(*buckets: Dict) -> None:
            # In place: captured graphs and the cached CuTe weight views keep
            # these addresses, and there is no saved metadata to rebuild from.
            addresses = (backend.w3_w1_weight.data_ptr(), backend.w2_weight.data_ptr())
            backend.pre_reload_weights()
            for weights in buckets:
                backend.load_weights([weights], allow_partial_loading=True)
            backend.process_weights_after_loading()
            backend.post_load_weights()
            assert (backend.w3_w1_weight.data_ptr(), backend.w2_weight.data_ptr()) == addresses
            assert backend.rebuild_tensor_metadata == {}

        def check(tag: str, expected: _Weights) -> None:
            for num_tokens in _RELOAD_TOKENS:
                x, expert_ids, routing_weights = _make_tokens(
                    num_tokens, shape, case, seed=100 + num_tokens, device=device
                )
                backend.set_adaptive_launch_tokens(num_tokens)
                with torch.inference_mode():
                    output = _run(backend, x, expert_ids, routing_weights)
                    reference = _reference_moe(x, expected, expert_ids, routing_weights, case)
                failures.extend(
                    f"{tag} T={num_tokens}: {message}"
                    for message in _compare(output, reference, overall_bound=_overall_rel_l2(shape))
                )

        check("as loaded", old)
        # Serving captures its graphs once, before any sync: an eager forward
        # (as the warmup does), then the capture.
        x, expert_ids, routing_weights = _make_tokens(7, shape, case, seed=1, device=device)

        def forward() -> torch.Tensor:
            backend.set_adaptive_launch_tokens(7)  # each launch consumes its hint
            return _run(backend, x, expert_ids, routing_weights)

        with torch.inference_mode():
            forward()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                captured = forward()
            torch.cuda.current_stream().wait_stream(stream)
        sync(
            bucket(new, every, ("w1", "w3")) | bucket(new, even, ("w2",)), bucket(new, odd, ("w2",))
        )
        with torch.inference_mode():
            graph.replay()
            torch.cuda.synchronize()
            reference = _reference_moe(x, new, expert_ids, routing_weights, case)
        failures.extend(
            f"graph replay after a full sync: {message}"
            for message in _compare(captured, reference, overall_bound=_overall_rel_l2(shape))
        )
        check("after a full sync", new)
        # Back to the old FC1 everywhere, but the old FC2 only on the even experts.
        sync(bucket(old, every, ("w1", "w3")) | bucket(old, even, ("w2",)))
        mixed_w2 = new.w2.clone()
        mixed_w2[even] = old.w2[even]
        check("after a partial sync", _Weights({}, old.w1, old.w3, mixed_w2))
    assert not failures, "\n".join(failures)


# Ultra's top-22 at a smaller expert. A quantized combine sends each route's
# FC2 row back in the format's precision, so the error is the format's, not the
# bf16 GEMMs': measured against the bf16 combine on GB200, mxfp8 2.7e-2 overall
# (2.9e-2 for the worst token), fp4 9.4e-2 (1.0e-1), at top-k 2 and 22 alike.
_QUANTIZED_COMBINE_SHAPE = _Shape(
    num_experts=64, top_k=22, hidden_size=1024, intermediate_size=1024
)
_QUANTIZED_COMBINE_TOKENS = (64, 256)
_QUANTIZED_COMBINE_REL_L2 = {"32e4m3xe8m0": 5e-2, "16e2m1xbf16": 1.5e-1}
_QUANTIZED_COMBINE_FORMATS = ["32e4m3xe8m0"] + ([] if IS_CI_MODE else ["16e2m1xbf16"])


@pytest.mark.parametrize("combine_format", _QUANTIZED_COMBINE_FORMATS)
def test_megamoe_bf16_ep1_quantized_combine_matches_bf16_combine(
    combine_format: str, monkeypatch: pytest.MonkeyPatch
):
    """``MEGAMOE_COMBINE_FORMAT`` at top-k 22: the quantized reduce over TopkReduce's per-route path.

    The same weights and tokens through the leaf with a bf16 combine (itself
    held to the reference) and with ``combine_format``, compared at the
    format's precision. A token with zero routing weights stays exactly zero.
    """
    shape, case = _QUANTIZED_COMBINE_SHAPE, _RELU2
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(shape, case, ep_size=1)
    device = torch.device("cuda", torch.cuda.current_device())
    bound = _QUANTIZED_COMBINE_REL_L2[combine_format]
    failures = []
    with torch.device(device):
        weights = _make_weights(shape, seed=5, device=device, gated=case.gated)
        backends = {}
        for backend_format in ("bf16", combine_format):
            # Read once, at construction.
            monkeypatch.setenv("MEGAMOE_COMBINE_FORMAT", backend_format)
            backends[backend_format] = _load(
                _make_backend(
                    TrtllmCutedslMegaMoeBf16Impl,
                    shape,
                    case,
                    Mapping(),
                    moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
                    max_num_tokens=max(_QUANTIZED_COMBINE_TOKENS),
                ),
                weights,
            )
        for backend_format, backend in backends.items():
            assert backend.combine_format == backend_format
        for num_tokens in _QUANTIZED_COMBINE_TOKENS:
            x, expert_ids, routing_weights = _make_tokens(
                num_tokens, shape, case, seed=num_tokens, device=device, zero_weight_token=0
            )
            outputs = {}
            with torch.inference_mode():
                for backend_format, backend in backends.items():
                    backend.set_adaptive_launch_tokens(num_tokens)
                    outputs[backend_format] = _run(backend, x, expert_ids, routing_weights)
                reference = _reference_moe(x, weights, expert_ids, routing_weights, case)
            failures += [
                f"T={num_tokens} bf16 combine vs reference: {message}"
                for message in _compare(outputs["bf16"], reference, zero_tokens=(0,))
            ]
            failures += [
                f"T={num_tokens} {combine_format} vs bf16 combine: {message}"
                for message in _compare(
                    outputs[combine_format],
                    outputs["bf16"].float(),
                    zero_tokens=(0,),
                    overall_bound=bound,
                    per_token_bound=bound,
                )
            ]
    assert not failures, "\n".join(failures)


_OP_MAX_TOKENS = 256
_OP_TOKEN_COUNTS = (1, 64, 256)
_OP_CASES = [
    pytest.param(
        shape,
        case,
        apply_topk_in_fc1,
        id=f"{shape}-{case}-{'topk_at_fc1' if apply_topk_in_fc1 else 'topk_at_reduce'}",
    )
    for shape, case, apply_topk_in_fc1 in (
        (_SMALL, _RELU2, False),
        (_SMALL, _RELU2, True),
        (_SMALL, _SWIGLU, False),
        (_SMALL, _SWIGLU, True),
        # Long tier: the routing weights at FC1 at Ultra's shape and at top-k 22 / 32.
        *(
            ()
            if IS_CI_MODE
            else (
                (_ULTRA, _RELU2, True),
                (
                    _Shape(num_experts=64, top_k=22, hidden_size=1024, intermediate_size=1024),
                    _RELU2,
                    True,
                ),
                (
                    _Shape(num_experts=64, top_k=32, hidden_size=1024, intermediate_size=1024),
                    _RELU2,
                    True,
                ),
            )
        ),
    )
]


@pytest.mark.parametrize("shape, case, apply_topk_in_fc1", _OP_CASES)
def test_megamoe_bf16_ep1_op_matches_fp32_reference(
    shape: _Shape, case: _ActivationCase, apply_topk_in_fc1: bool
):
    """The custom op called directly on one rank, with the routing weights at either end.

    The backends apply the weights at FC1 only for DeepSeek-V4, so this is
    where that placement runs for the BF16 kernels: the epilogue scales the
    float32 activation before its single bf16 store and TopkReduce sums the
    routes unweighted. Driven as the backend drives a single rank: a
    zero-filled local shared workspace sized by
    ``query_megamoe_kernel_properties`` for the same activation, weight
    placement and quant kind, inputs padded to max_tokens (``topk_idx`` -1,
    weights 0), ``peer_offsets`` [0], and the FC1 rows of the weight method.
    """
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(shape, case, ep_size=1)
    device = torch.device("cuda", torch.cuda.current_device())
    problem = dict(
        world_size=1,
        num_topk=shape.top_k,
        num_experts_per_rank=shape.num_experts,
        hidden_size=shape.hidden_size,
        intermediate_size_per_partition=shape.intermediate_size,
        expand_intermediate_size_per_partition=(2 if case.gated else 1) * shape.intermediate_size,
        max_tokens_per_rank=_OP_MAX_TOKENS,
        apply_topk_in_fc1=apply_topk_in_fc1,
        combine_format="bf16",
        quant_kind="bf16",
        activation_type=int(ActivationType(case.build().kind)),
    )
    failures = []
    with torch.device(device):
        weights = _make_weights(shape, seed=11, device=device, gated=case.gated)
        fc1_weight = _kernel_fc1_weight(weights, case)
        shared_bytes, _ = megamoe_op.query_megamoe_kernel_properties(**problem)
        shared_workspace = torch.zeros(shared_bytes, dtype=torch.uint8, device=device)
        for num_tokens in _OP_TOKEN_COUNTS:
            zero_tokens = (0,) if num_tokens > 1 else ()
            x, expert_ids, routing_weights = _make_tokens(
                num_tokens,
                shape,
                case,
                seed=100 + num_tokens,
                device=device,
                zero_weight_token=zero_tokens[0] if zero_tokens else None,
            )
            combine_output = torch.empty(
                _OP_MAX_TOKENS, 1, shape.hidden_size, dtype=torch.bfloat16, device=device
            )
            with torch.inference_mode():
                torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell(
                    **_staged_op_inputs(x, expert_ids, routing_weights, _OP_MAX_TOKENS),
                    activation_sf=None,
                    fc1_weight=fc1_weight,
                    fc1_weight_sf=None,
                    fc2_weight=weights.w2,
                    fc2_weight_sf=None,
                    fc1_alpha=None,
                    fc2_alpha=None,
                    fc1_norm_const=None,
                    combine_output=combine_output,
                    shared_workspace=shared_workspace,
                    local_rank=0,
                    peer_offsets=[0],
                    num_tokens=num_tokens,
                    **problem,
                )
                reference = _reference_moe(
                    x,
                    weights,
                    expert_ids,
                    routing_weights,
                    case,
                    apply_topk_in_fc1=apply_topk_in_fc1,
                )
            failures += [
                f"T={num_tokens}: {message}"
                for message in _compare(
                    combine_output[:num_tokens, 0],
                    reference,
                    zero_tokens=zero_tokens,
                    overall_bound=_overall_rel_l2(shape),
                )
            ]
    assert not failures, "\n".join(failures)


# Long tier: ReLU2 extents with FC1 CTA tiles past the last feature (see
# _PLAN_CASES), launched repeatedly under the default tactic (cluster M 2) and
# the mixed (4, 1) cluster candidate. 1088 and 1152 have such tiles under
# both, 1216 only at cluster M 4.
_REPEAT_TOKENS = 3000
_REPEAT_MAX_TOKENS = 4096
_REPEAT_LAUNCHES = 10
_REPEAT_CASES = [
    pytest.param(intermediate, tactic_name, id=f"i{intermediate}-{tactic_name}")
    for intermediate, tactic_name in (
        ()
        if IS_CI_MODE
        else (
            (1088, "default"),
            (1088, "mixed_cluster"),
            (1152, "default"),
            (1152, "mixed_cluster"),
            (1216, "mixed_cluster"),
        )
    )
]


@_LONG_TIER
@pytest.mark.parametrize("intermediate, tactic_name", _REPEAT_CASES)
def test_megamoe_bf16_ep1_relu2_partial_tiles_repeated(intermediate: int, tactic_name: str):
    """Ten launches through the runner with an explicit tactic, each against the reference and
    bit-identical to the first.

    An FC1 tile without rows that signals fc1_done lets FC2 read features
    whose stores have not landed yet, which only shows up now and then.
    """
    shape, case = (
        _Shape(num_experts=8, top_k=4, hidden_size=1024, intermediate_size=intermediate),
        _RELU2,
    )
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(shape, case, ep_size=1)
    device = torch.device("cuda", torch.cuda.current_device())
    runner = _plan_runner(
        case,
        intermediate,
        max_tokens=_REPEAT_MAX_TOKENS,
        local_experts=shape.num_experts,
        top_k=shape.top_k,
        hidden=shape.hidden_size,
    )
    if tactic_name == "default":
        tactic = None
    else:
        tactic = next(
            candidate
            for candidate in megamoe_op.enumerate_megamoe_candidate_tactics(
                16384, sm_version=runner.sm_version, quant_kind="bf16"
            )
            if _is_mixed_cluster(candidate)
        )
    failures = []
    with torch.device(device):
        weights = _make_weights(shape, seed=13, device=device, gated=case.gated)
        shared_bytes, _ = megamoe_op.query_megamoe_kernel_properties(
            world_size=1,
            num_topk=shape.top_k,
            num_experts_per_rank=shape.num_experts,
            hidden_size=shape.hidden_size,
            intermediate_size_per_partition=intermediate,
            expand_intermediate_size_per_partition=intermediate,
            max_tokens_per_rank=_REPEAT_MAX_TOKENS,
            tactic=tactic,
            apply_topk_in_fc1=False,
            quant_kind="bf16",
            activation_type=int(ActivationType.Relu2),
        )
        shared_workspace = torch.zeros(shared_bytes, dtype=torch.uint8, device=device)
        x, expert_ids, routing_weights = _make_tokens(
            _REPEAT_TOKENS, shape, case, seed=17, device=device, zero_weight_token=0
        )
        staged = _staged_op_inputs(x, expert_ids, routing_weights, _REPEAT_MAX_TOKENS)
        combine_output = torch.empty(
            _REPEAT_TOKENS, 1, shape.hidden_size, dtype=torch.bfloat16, device=device
        )
        inputs = [
            staged["activation"],
            None,
            staged["topk_idx"],
            staged["topk_weights"],
            weights.w1,
            None,
            weights.w2,
            None,
            None,
            None,
            None,
            combine_output,
        ]
        with torch.inference_mode():
            reference = _reference_moe(x, weights, expert_ids, routing_weights, case)
            first = None
            for launch in range(_REPEAT_LAUNCHES):
                runner(
                    inputs,
                    tactic=-1 if tactic is None else tactic,
                    peer_offsets=[0],
                    shared_workspace=shared_workspace,
                )
                output = combine_output[:, 0].clone()
                failures += [
                    f"launch {launch}: {message}"
                    for message in _compare(output, reference, zero_tokens=(0,))
                ]
                if first is None:
                    first = output
                elif not torch.equal(output, first):
                    failures.append(f"launch {launch} differs from launch 0")
    assert not failures, "\n".join(failures)


@_LONG_TIER
def test_megamoe_bf16_ep1_relu2_cuda_graph_replay(monkeypatch: pytest.MonkeyPatch):
    """Serving captures the decode forward: one eager forward (which fences the workspaces, as
    the warmup does), a capture of the same forward, then replays on new inputs copied into the
    captured tensors, each against the reference."""
    shape, case, num_tokens = _ULTRA, _RELU2, 7
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(shape, case, ep_size=1)
    # The op records, by address and for the life of the process, every shared
    # workspace a captured launch used, and refuses to fence one again. In
    # serving the graphs live as long as their layers; here the graph and the
    # backend's workspaces die with the test, and the caching allocator hands
    # their addresses to the next layer of the same size, whose first launch
    # must fence. Scope the capture state to this test, as test_moe_backend.py
    # scopes the CUTLASS runner cache around its capture.
    monkeypatch.setattr(megamoe_op, "_MEGAMOE_CAPTURED_SHARED_PTRS", set())
    monkeypatch.setattr(megamoe_op, "_MEGAMOE_GRAPH_CAPTURE_SEEN", False)
    device = torch.device("cuda", torch.cuda.current_device())
    failures = []
    with torch.device(device):
        weights = _make_weights(shape, seed=19, device=device, gated=case.gated)
        backend = _load(
            _make_backend(
                TrtllmCutedslMegaMoeBf16Impl,
                shape,
                case,
                Mapping(),
                moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
                max_num_tokens=256,
            ),
            weights,
        )
        x, expert_ids, routing_weights = _make_tokens(
            num_tokens, shape, case, seed=1, device=device
        )

        def forward():
            backend.set_adaptive_launch_tokens(num_tokens)
            return _run(backend, x, expert_ids, routing_weights)

        with torch.inference_mode():
            eager = forward()
            reference = _reference_moe(x, weights, expert_ids, routing_weights, case)
            failures += [
                f"eager: {m}"
                for m in _compare(eager, reference, overall_bound=_overall_rel_l2(shape))
            ]
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                captured = forward()
            torch.cuda.current_stream().wait_stream(stream)
            for seed in (2, 3, 4):
                new_x, new_ids, new_weights = _make_tokens(
                    num_tokens, shape, case, seed=seed, device=device, zero_weight_token=seed
                )
                x.copy_(new_x)
                expert_ids.copy_(new_ids)
                routing_weights.copy_(new_weights)
                graph.replay()
                torch.cuda.synchronize()
                reference = _reference_moe(x, weights, expert_ids, routing_weights, case)
                failures += [
                    f"replay {seed}: {m}"
                    for m in _compare(
                        captured,
                        reference,
                        zero_tokens=(seed,),
                        overall_bound=_overall_rel_l2(shape),
                    )
                ]
    assert not failures, "\n".join(failures)


# A forward with more tokens per rank than ``moe_max_num_tokens`` runs as
# ceil(T / moe_max_num_tokens) launches of near-equal chunks that the scheduler
# concatenates after the last one, while every launch of the leaf returns rows of
# the combine buffer that the next launch in its bucket rewrites. (shape, case,
# tokens, moe_max_num_tokens, max_num_tokens, chunk sizes): the capped layer
# launches its top bucket, the cap, and the uncapped one max_num_tokens; the
# _EP1_CASES above build both buckets for the same shapes.
_EP1_CHUNKED_CASES = [
    pytest.param(*params, id=f"{params[0]}-{params[1]}-{len(params[5])}_chunks")
    for params in (
        (_SMALL, _RELU2, 1001, 256, 1024, (251, 250, 250, 250)),
        (_SMALL, _SWIGLU, 301, 256, 1024, (151, 150)),
        # Ultra's expert shape under the cap that serving at DEP16 needs for memory.
        (_ULTRA, _RELU2, 2500, 1024, 4096, (834, 833, 833)),
    )
]


@pytest.mark.parametrize(
    "shape, case, num_tokens, moe_max_num_tokens, max_num_tokens, chunk_sizes", _EP1_CHUNKED_CASES
)
def test_megamoe_bf16_ep1_chunked_forward_matches_one_chunk(
    shape: _Shape,
    case: _ActivationCase,
    num_tokens: int,
    moe_max_num_tokens: int,
    max_num_tokens: int,
    chunk_sizes: Tuple[int, ...],
):
    """``create_moe`` + ``forward`` over more tokens than ``moe_max_num_tokens``.

    The capped layer against the float32 reference and against an uncapped
    layer, which runs the same weights and tokens in one launch, both within
    the file's bounds. The launches each layer makes are recorded, so a layer
    that did not split the forward as intended fails too.
    """
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(shape, case, ep_size=1)
    device = torch.device("cuda", torch.cuda.current_device())
    layers = {"chunked": (moe_max_num_tokens, list(chunk_sizes)), "one chunk": (None, [num_tokens])}
    failures = []
    outputs = {}
    with torch.device(device):
        weights = _make_weights(shape, seed=23, device=device, gated=case.gated)
        generator = torch.Generator().manual_seed(29)
        x = torch.randn(num_tokens, shape.hidden_size, generator=generator, device="cpu")
        x = (x * case.input_scale).to(dtype=torch.bfloat16, device=device)
        router_logits = torch.randn(
            num_tokens, shape.num_experts, generator=generator, device="cpu"
        ).to(device)
        routing_method = RenormalizeMoeRoutingMethod(
            top_k=shape.top_k, force_enable_pytorch_op=True
        )
        for name, (cap, expected_launches) in layers.items():
            model_config = _model_config(
                shape,
                Mapping(),
                moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
                max_num_tokens=max_num_tokens,
                moe_max_num_tokens=cap,
            )
            with _moe_layer(shape, case, routing_method, model_config) as moe:
                if type(moe.backend) is not TrtllmCutedslMegaMoeBf16Impl:
                    failures.append(f"{name}: built {type(moe.backend).__name__}")
                    continue
                moe.load_weights([weights.checkpoint])
                moe.post_load_weights()
                moe.cuda(device)
                launches = _record_launches(moe.backend)
                with torch.inference_mode():
                    # A one-launch forward returns a view of the layer's combine buffer.
                    outputs[name] = moe.forward(x, router_logits).clone()
                if launches != expected_launches:
                    failures.append(f"{name}: launched {launches} tokens, not {expected_launches}")
        with torch.inference_mode():
            expert_ids, routing_weights = routing_method.apply(router_logits)
            reference = _reference_moe(x, weights, expert_ids, routing_weights, case)
    overall_bound = _overall_rel_l2(shape)
    for name, output in outputs.items():
        failures += [
            f"{name} vs reference: {m}"
            for m in _compare(output, reference, overall_bound=overall_bound)
        ]
    if len(outputs) == len(layers):
        failures += [
            f"chunked vs one chunk: {m}"
            for m in _compare(
                outputs["chunked"], outputs["one chunk"].float(), overall_bound=overall_bound
            )
        ]
    assert not failures, "\n".join(failures)


# =====================================================================
# EP4: four EP ranks on one node
# =====================================================================

_EP4_WORLD_SIZE = 4
_EP4_SHAPE = _Shape(num_experts=16, top_k=4, hidden_size=2048, intermediate_size=1536)
_EP4_CASES = (_SWIGLU, _MINIMAX)
# ReLU2 at Ultra's 512 experts and top-22 (128 local experts per rank, four
# scheduler scan batches, 32 router push CTAs); Ultra's I runs on one GPU.
_EP4_RELU2_SHAPE = _Shape(num_experts=512, top_k=22, hidden_size=2048, intermediate_size=512)
# Long tier: Ultra's expert shape across the four ranks.
_EP4_ULTRA_SHAPE = _Shape(num_experts=64, top_k=22, hidden_size=2048, intermediate_size=5120)
# Per-rank token counts of each DEP round: uneven, a single token, counts that
# are not multiples of 16, and a rank with no tokens, which still launches so
# its peers can cross the dispatch barrier. The first two rounds run the 256
# bucket's (256, 128, 64) tile; the last one's 1500 selects the 2048 bucket,
# whose default tile is (256, 256, 64).
_EP4_DEP_ROUNDS = ((1, 7, 64, 200), (0, 256, 33, 16), (1500, 0, 33, 1024))
# TEP replicates the same tokens on every rank.
_EP4_TEP_TOKENS = 48
# Buckets 256, 1024 and 2048.
_EP4_MAX_NUM_TOKENS = 2048
# The module-level runs: per-rank tokens under DEP, replicated ones under TEP.
_EP4_MODULE_DEP_TOKENS = (5, 64, 1, 33)
_EP4_MODULE_TEP_TOKENS = 48
_EP4_MODULE_MAX_NUM_TOKENS = 256
# The module again under a cap that splits the largest rank into three chunks.
# Every rank launches those three chunks whatever its own count, so under DEP
# the ranks run different chunk sizes in the same launches, one rank none at all;
# TEP replicates the same tokens on every rank.
_EP4_CHUNKED_MOE_MAX_NUM_TOKENS = 256
_EP4_CHUNKED_DEP_TOKENS = (600, 2, 257, 0)
_EP4_CHUNKED_DEP_LAUNCHES = ([200, 200, 200], [1, 1, 0], [86, 86, 85], [0, 0, 0])
_EP4_CHUNKED_TEP_TOKENS = 600
_EP4_CHUNKED_TEP_LAUNCHES = [200, 200, 200]
_EP4_CHUNKED_MAX_NUM_TOKENS = 1024
# Long tier, dynamic EPLB: test_moe_module.py's EPLB layer (8 experts in 16
# slots, top-2) at the BF16 leaf's alignment, run until the balancer migrates.
_EPLB_SHAPE = _Shape(num_experts=8, top_k=2, hidden_size=1024, intermediate_size=1024)
_EPLB_NUM_SLOTS = 16
_EPLB_TOKENS = 64
_EPLB_STEPS = 8


def _init_ep_process_group(rank: int, world_size: int) -> None:
    """NCCL WORLD over a TCPStore whose port rank 0 broadcasts over MPI.

    Same bootstrap as ``_ensure_dist_for_megamoe`` in test_moe_module.py;
    single node, so the loopback address is enough.
    """
    if dist.is_initialized():
        return
    master_addr = "127.0.0.1"
    store = None
    if rank == 0:
        # Bind before publishing the OS-selected port; wait_for_workers must
        # be False because clients cannot connect until the port is broadcast.
        store = dist.TCPStore(
            host_name=master_addr,
            port=0,
            world_size=world_size,
            is_master=True,
            wait_for_workers=False,
        )
    port = mpi_comm().bcast(store.port if store is not None else None, root=0)
    if store is None:
        store = dist.TCPStore(
            host_name=master_addr,
            port=port,
            world_size=world_size,
            is_master=False,
            wait_for_workers=False,
        )
    dist.init_process_group(backend="nccl", store=store, rank=rank, world_size=world_size)


def _ep4_mapping(rank: int, world_size: int, parallel_mode: str) -> Mapping:
    """DEP (attention DP) or TEP (attention TP); MoE EP over all ranks either way."""
    return Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=parallel_mode == "DEP",
    )


def _ep4_backend_failures(
    rank: int, world_size: int, shape: _Shape, case: _ActivationCase, weights: _Weights
) -> List[str]:
    """The DEP rounds and the TEP round of one layer through the BF16 leaf's backend."""
    device = torch.device("cuda", torch.cuda.current_device())
    failures = []
    for parallel_mode in ("DEP", "TEP"):
        backend = _load(
            _make_backend(
                TrtllmCutedslMegaMoeBf16Impl,
                shape,
                case,
                _ep4_mapping(rank, world_size, parallel_mode),
                moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
                max_num_tokens=_EP4_MAX_NUM_TOKENS,
            ),
            weights,
        )
        rounds = _EP4_DEP_ROUNDS if parallel_mode == "DEP" else ((_EP4_TEP_TOKENS,) * world_size,)
        for round_idx, token_counts in enumerate(rounds):
            num_tokens = token_counts[rank]
            zero_tokens = (num_tokens - 1,) if num_tokens > 1 else ()
            x, expert_ids, routing_weights = _make_tokens(
                num_tokens,
                shape,
                case,
                # DEP ranks hold different tokens; TEP ranks the same.
                seed=100 * round_idx + (rank if parallel_mode == "DEP" else 0),
                device=device,
                zero_weight_token=zero_tokens[0] if zero_tokens else None,
            )
            backend.set_adaptive_launch_tokens(max(token_counts))
            with torch.inference_mode():
                output = _run(backend, x, expert_ids, routing_weights)
                reference = _reference_moe(x, weights, expert_ids, routing_weights, case)
            failures += [
                f"rank {rank} {parallel_mode} {shape}-{case} round {round_idx} "
                f"T={num_tokens}: {message}"
                for message in _compare(
                    output,
                    reference,
                    zero_tokens=zero_tokens,
                    overall_bound=_overall_rel_l2(shape),
                )
            ]
    return failures


def _nemotron_h_routing(shape: _Shape, device: torch.device) -> DeepSeekV3MoeRoutingMethod:
    """Nemotron-3 Ultra's router: sigmoid scores with a score-correction bias, one group,
    normalized top-k weights scaled by 5, on the fused noaux_tc kernel.

    The bias is a model weight, so it is drawn alike on every rank.
    """
    bias = torch.randn(shape.num_experts, generator=torch.Generator().manual_seed(4), device="cpu")
    bias = (bias * 0.01).to(device)
    return DeepSeekV3MoeRoutingMethod(
        top_k=shape.top_k,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=5.0,
        callable_e_score_correction_bias=lambda: bias,
    )


def _nemotron_h_moe(
    shape: _Shape,
    routing_method: DeepSeekV3MoeRoutingMethod,
    model_config: ModelConfig,
):
    """The routed experts of Nemotron-H's MoE layer as ``NemotronHMOE`` builds them: ReLU2, no
    expert bias, the shared expert's all-reduce left to the caller (``reduce_results=False``).
    No degradation, so the layer is the BF16 leaf's or nothing."""
    return _moe_layer(shape, _RELU2, routing_method, model_config)


def _ep4_module_failures(rank: int, world_size: int, shape: _Shape, weights: _Weights) -> List[str]:
    """Nemotron-H's routed experts through ``create_moe`` + ``forward``, DEP then TEP.

    The reference takes the router's own (ids, weights). Under TEP the fused
    kernel returns the globally combined output on every rank; with
    ``reduce_results=False`` the layer keeps it on TP rank 0 and returns zeros
    elsewhere, so the caller's all-reduce restores it exactly.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    failures = []
    for parallel_mode in ("DEP", "TEP"):
        if parallel_mode == "DEP":
            all_rank_num_tokens = list(_EP4_MODULE_DEP_TOKENS)
            num_tokens, seed = all_rank_num_tokens[rank], 500 + rank
        else:
            all_rank_num_tokens, num_tokens, seed = None, _EP4_MODULE_TEP_TOKENS, 600
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(num_tokens, shape.hidden_size, generator=generator, device="cpu")
        x = x.to(dtype=torch.bfloat16, device=device)
        router_logits = torch.randn(
            num_tokens, shape.num_experts, generator=generator, device="cpu"
        )
        router_logits = router_logits.to(device)
        routing_method = _nemotron_h_routing(shape, device)
        model_config = _model_config(
            shape,
            _ep4_mapping(rank, world_size, parallel_mode),
            moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
            max_num_tokens=_EP4_MODULE_MAX_NUM_TOKENS,
        )
        tag = f"rank {rank} module {parallel_mode} {shape}-relu2 T={num_tokens}"
        with _nemotron_h_moe(shape, routing_method, model_config) as moe:
            # Resolution is rank-identical, so every rank skips the launch alike.
            if type(moe.backend) is not TrtllmCutedslMegaMoeBf16Impl:
                failures.append(f"{tag}: built {type(moe.backend).__name__}")
                continue
            moe.load_weights([weights.checkpoint])
            moe.post_load_weights()
            moe.cuda(device)
            with torch.inference_mode():
                output = moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
                expert_ids, routing_weights = routing_method.apply(router_logits)
                reference = _reference_moe(x, weights, expert_ids, routing_weights, _RELU2)
                if parallel_mode == "DEP" or rank == 0:
                    failures += [f"{tag}: {m}" for m in _compare(output, reference)]
                elif not torch.all(output == 0):
                    failures.append(
                        f"{tag}: a TP rank other than 0 returned non-zero routed output"
                    )
                if parallel_mode == "TEP":
                    reduced = output.clone()
                    dist.all_reduce(reduced)
                    rank0_output = output.clone()
                    dist.broadcast(rank0_output, src=0)
                    if not torch.equal(reduced, rank0_output):
                        failures.append(f"{tag}: all-reduced output != TP rank 0's output")
    return failures


def _ep4_chunked_module_failures(
    rank: int, world_size: int, shape: _Shape, weights: _Weights
) -> List[str]:
    """``_ep4_module_failures`` over more tokens per rank than ``moe_max_num_tokens``.

    Each layer must launch the expected chunks on every rank, and the forward
    must match the reference: per rank under DEP (a rank without tokens returns
    an empty output), on TP rank 0 under TEP with zeros elsewhere.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    failures = []
    for parallel_mode in ("DEP", "TEP"):
        if parallel_mode == "DEP":
            all_rank_num_tokens = list(_EP4_CHUNKED_DEP_TOKENS)
            num_tokens, seed = all_rank_num_tokens[rank], 800 + rank
            expected_launches = _EP4_CHUNKED_DEP_LAUNCHES[rank]
        else:
            all_rank_num_tokens, num_tokens, seed = None, _EP4_CHUNKED_TEP_TOKENS, 900
            expected_launches = _EP4_CHUNKED_TEP_LAUNCHES
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(num_tokens, shape.hidden_size, generator=generator, device="cpu")
        x = x.to(dtype=torch.bfloat16, device=device)
        router_logits = torch.randn(
            num_tokens, shape.num_experts, generator=generator, device="cpu"
        ).to(device)
        routing_method = _nemotron_h_routing(shape, device)
        model_config = _model_config(
            shape,
            _ep4_mapping(rank, world_size, parallel_mode),
            moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
            max_num_tokens=_EP4_CHUNKED_MAX_NUM_TOKENS,
            moe_max_num_tokens=_EP4_CHUNKED_MOE_MAX_NUM_TOKENS,
        )
        tag = f"rank {rank} chunked module {parallel_mode} {shape}-relu2 T={num_tokens}"
        with _nemotron_h_moe(shape, routing_method, model_config) as moe:
            # Resolution is rank-identical, so every rank skips the launches alike.
            if type(moe.backend) is not TrtllmCutedslMegaMoeBf16Impl:
                failures.append(f"{tag}: built {type(moe.backend).__name__}")
                continue
            moe.load_weights([weights.checkpoint])
            moe.post_load_weights()
            moe.cuda(device)
            launches = _record_launches(moe.backend)
            with torch.inference_mode():
                output = moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
                if launches != expected_launches:
                    failures.append(f"{tag}: launched {launches} tokens, not {expected_launches}")
                if num_tokens == 0:
                    if output.shape != (0, shape.hidden_size):
                        failures.append(f"{tag}: output shape {tuple(output.shape)}")
                elif parallel_mode == "DEP" or rank == 0:
                    expert_ids, routing_weights = routing_method.apply(router_logits)
                    reference = _reference_moe(x, weights, expert_ids, routing_weights, _RELU2)
                    failures += [f"{tag}: {m}" for m in _compare(output, reference)]
                elif not torch.all(output == 0):
                    failures.append(
                        f"{tag}: a TP rank other than 0 returned non-zero routed output"
                    )
                if parallel_mode == "TEP":
                    reduced = output.clone()
                    dist.all_reduce(reduced)
                    rank0_output = output.clone()
                    dist.broadcast(rank0_output, src=0)
                    if not torch.equal(reduced, rank0_output):
                        failures.append(f"{tag}: all-reduced output != TP rank 0's output")
    return failures


def _ep4_relu2_eplb_failures(rank: int, world_size: int) -> List[str]:
    """ReLU2 under dynamic EPLB: every token routes alike, so the balancer moves experts
    between slots within a few steps. Each step must match the reference, and the expert
    placement must have changed by the end.

    ReLU2's FC1 rows migrate as loaded; the gated interleave is the weight
    method's alone, applied to the EPLB host staging too.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    shape = _EPLB_SHAPE
    weights = _make_weights(shape, seed=2468, device=device, gated=False)
    mapping = _ep4_mapping(rank, world_size, "DEP")
    model_config = _model_config(
        shape,
        mapping,
        moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
        max_num_tokens=_EP4_MODULE_MAX_NUM_TOKENS,
        moe_load_balancer=MoeLoadBalancerConfig(
            num_slots=_EPLB_NUM_SLOTS, layer_updates_per_iter=1
        ),
    )
    model_config.moe_load_balancer.setup(ep_rank=mapping.moe_ep_rank, ep_size=mapping.moe_ep_size)
    balancer = MoeLoadBalancer(
        ep_rank=mapping.moe_ep_rank, ep_size=mapping.moe_ep_size, layer_updates_per_iter=1
    )
    routing_method = _nemotron_h_routing(shape, device)
    # One row of logits for every token on every rank: the load piles onto the
    # same experts, which forces the balancer to replicate them.
    logits_row = torch.randn(
        1, shape.num_experts, generator=torch.Generator().manual_seed(700), device="cpu"
    )
    router_logits = logits_row.repeat(_EPLB_TOKENS, 1).to(device)
    x = torch.randn(
        _EPLB_TOKENS,
        shape.hidden_size,
        generator=torch.Generator().manual_seed(701 + rank),
        device="cpu",
    ).to(dtype=torch.bfloat16, device=device)
    # The EPLB host staging is built from host weights, as in test_moe_module.py.
    checkpoint = {name: tensor.cpu() for name, tensor in weights.checkpoint.items()}
    failures = []
    with balancer, _nemotron_h_moe(shape, routing_method, model_config) as moe:
        if type(moe.backend) is not TrtllmCutedslMegaMoeBf16Impl:
            return [f"rank {rank} EPLB: built {type(moe.backend).__name__}"]
        moe.load_weights([checkpoint])
        moe.post_load_weights()
        moe.cuda(device)
        balancer.register_weight_slots_after_to_cuda()
        balancer.finalize_model()
        balancer.set_iter_info(enable_statistic=True, enable_update_weights=True)
        layer_balancer = balancer.single_layer_load_balancers[0]
        initial_placement = copy.deepcopy(layer_balancer.get_old_rank_expert_ids())
        with torch.inference_mode():
            expert_ids, routing_weights = routing_method.apply(router_logits)
            reference = _reference_moe(x, weights, expert_ids, routing_weights, _RELU2)
        for step in range(_EPLB_STEPS):
            with torch.inference_mode(), MoeLoadBalancerIterContext(balancer):
                output = moe.forward(
                    x, router_logits, all_rank_num_tokens=[_EPLB_TOKENS] * world_size
                )
            torch.cuda.synchronize()
            failures += [f"rank {rank} EPLB step {step}: {m}" for m in _compare(output, reference)]
        if layer_balancer.get_old_rank_expert_ids() == initial_placement:
            failures.append(f"rank {rank} EPLB: no expert moved in {_EPLB_STEPS} steps")
    return failures


def _ep4_rank_body() -> None:
    """Every rank runs the same launches in the same order; failures are
    collected and raised only after the last launch, so one rank's mismatch
    cannot strand its peers inside a dispatch barrier.

    The gated layer's DEP and TEP rounds; then ReLU2 at Ultra's 512 experts
    and top-22, through the backend and through ``create_moe``, there in one
    launch and in three chunks; then, in the long tier, Ultra's expert shape
    and ReLU2 under dynamic EPLB.
    """
    rank, world_size = mpi_rank(), mpi_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    _init_ep_process_group(rank, world_size)
    device = torch.device("cuda", torch.cuda.current_device())
    failures = []
    with torch.device(device):
        weights = _make_weights(_EP4_SHAPE, seed=1234, device=device)
        for case in _EP4_CASES:
            failures += _ep4_backend_failures(rank, world_size, _EP4_SHAPE, case, weights)
        weights = _make_weights(_EP4_RELU2_SHAPE, seed=4321, device=device, gated=False)
        failures += _ep4_backend_failures(rank, world_size, _EP4_RELU2_SHAPE, _RELU2, weights)
        failures += _ep4_module_failures(rank, world_size, _EP4_RELU2_SHAPE, weights)
        failures += _ep4_chunked_module_failures(rank, world_size, _EP4_RELU2_SHAPE, weights)
        del weights
        if not IS_CI_MODE:
            weights = _make_weights(_EP4_ULTRA_SHAPE, seed=5432, device=device, gated=False)
            failures += _ep4_backend_failures(rank, world_size, _EP4_ULTRA_SHAPE, _RELU2, weights)
            del weights
            failures += _ep4_relu2_eplb_failures(rank, world_size)
    torch.cuda.synchronize()
    assert not failures, "\n".join(failures)


def _ep4_init_worker(paths: List[str]) -> None:
    # Spawned workers re-import this module by name; give them the parent's paths.
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)


def _ep4_worker_entry(_: int) -> None:
    _ep4_rank_body()


@pytest.mark.gpu4
@pytest.mark.threadleak(enabled=False)  # MPIPoolExecutor's manager thread outlives the test
def test_megamoe_bf16_ep4_matches_fp32_reference():
    world_size = mpi_world_size()
    if world_size not in (1, _EP4_WORLD_SIZE):
        pytest.skip(f"needs one process or {_EP4_WORLD_SIZE} ranks, got {world_size}")
    layers = [(_EP4_SHAPE, case) for case in _EP4_CASES] + [(_EP4_RELU2_SHAPE, _RELU2)]
    if not IS_CI_MODE:
        layers += [(_EP4_ULTRA_SHAPE, _RELU2), (_EPLB_SHAPE, _RELU2)]
    for shape, case in layers:
        _skip_unless_bf16_megamoe_runs(shape, case, ep_size=_EP4_WORLD_SIZE)
    if world_size == _EP4_WORLD_SIZE:
        # Started as one process per rank: this process is one of the ranks.
        _ep4_rank_body()
        return
    with MPIPoolExecutor(
        initializer=_ep4_init_worker,
        initargs=(sys.path,),
        max_workers=_EP4_WORLD_SIZE,
    ) as executor:
        results = list(executor.map(_ep4_worker_entry, range(_EP4_WORLD_SIZE)))
    assert results == [None] * _EP4_WORLD_SIZE


# =====================================================================
# Backend-level parity with CUTLASS BF16
# =====================================================================

_CUTLASS_TOKENS = 96
_CUTLASS_MAX_NUM_TOKENS = 256
_CUTLASS_CASES = [
    pytest.param(shape, case, id=f"{shape}-{case}")
    for shape, case in (
        (_SMALL, _SWIGLU),
        (_SMALL, _MINIMAX),
        (_Shape(num_experts=8, top_k=4, hidden_size=2048, intermediate_size=1536), _SWIGLU),
        # CUTLASS stores Relu2's pre-activation as bf16 and squares it in place:
        # two roundings to MegaMoE's one, inside the CUTLASS bounds.
        (_Shape(num_experts=8, top_k=4, hidden_size=2048, intermediate_size=1536), _RELU2),
        # Long tier: Ultra's expert shape.
        *(() if IS_CI_MODE else ((_ULTRA, _RELU2),)),
    )
]


@pytest.mark.parametrize("shape, case", _CUTLASS_CASES)
def test_megamoe_bf16_matches_cutlass(shape: _Shape, case: _ActivationCase):
    """Same weights, routing and input through both backends' ``quantize_input`` / ``run_moe``."""
    _skip_under_multi_rank_launch()
    _skip_unless_bf16_megamoe_runs(shape, case, ep_size=1)
    if not case.gated and not collect_moe_environment().oss_cutlass_moe:
        pytest.skip("CUTLASS runs Relu2 only in a build with USING_OSS_CUTLASS_MOE_GEMM=ON")
    device = torch.device("cuda", torch.cuda.current_device())
    outputs = {}
    with torch.device(device):
        weights = _make_weights(shape, seed=7, device=device, gated=case.gated)
        x, expert_ids, routing_weights = _make_tokens(
            _CUTLASS_TOKENS, shape, case, seed=7, device=device
        )
        for backend_type in (MoeBackendType.CUTLASS, MoeBackendType.MEGAMOE_CUTEDSL):
            moe_cls = get_backend_class(backend_type, None)
            backend = _load(
                _make_backend(
                    moe_cls,
                    shape,
                    case,
                    Mapping(),
                    moe_backend=backend_type.value,
                    max_num_tokens=_CUTLASS_MAX_NUM_TOKENS,
                ),
                weights,
            )
            if backend.scheduler_kind == MoESchedulerKind.FUSED_COMM:
                backend.set_adaptive_launch_tokens(_CUTLASS_TOKENS)
            with torch.inference_mode():
                outputs[backend_type] = _run(backend, x, expert_ids, routing_weights)
        with torch.inference_mode():
            reference = _reference_moe(x, weights, expert_ids, routing_weights, case)

    megamoe = outputs[MoeBackendType.MEGAMOE_CUTEDSL]
    cutlass = outputs[MoeBackendType.CUTLASS]
    loose = dict(overall_bound=_CUTLASS_OVERALL_REL_L2, per_token_bound=_CUTLASS_PER_TOKEN_REL_L2)
    failures = [
        f"MegaMoE vs reference: {m}"
        for m in _compare(megamoe, reference, overall_bound=_overall_rel_l2(shape))
    ]
    # CUTLASS is the control: if it misses the reference too, the reference is wrong.
    failures += [f"CUTLASS vs reference: {m}" for m in _compare(cutlass, reference, **loose)]
    failures += [f"MegaMoE vs CUTLASS: {m}" for m in _compare(megamoe, cutlass.float(), **loose)]
    assert not failures, "\n".join(failures)
