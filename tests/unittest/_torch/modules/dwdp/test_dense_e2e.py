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
"""Single-process end-to-end test of dense DWDP on one GPU.

The whole dense path runs as it does in a context worker, with an ``N``-rank DWDP group folded
into this process:

* A Kimi-K3-like toy decoder: a replicated layer 0, then eight covered layers mixing KDA-like and
  MLA-like attention with an MoE block.  Every covered layer moves FP8 buffers with row-view
  aliases (the fused ``qkvg`` projection), a ``_bfa_proj_weight`` attribute with ``nn.Parameter``
  row views and pad rows, a tuple attribute, an excluded scratch name that aliases a moved
  storage, MLA absorb parameters, a ``v_b_proj_scale`` parameter that views a buffer, and an fp32
  router.  The moved storages are above the default 1 MiB threshold, so real blobs move, while the
  scales, norms and decode constants below it stay.  The layers carry ``dense_dwdp_roots`` and the
  two model tables and call H1-H3 where ``modeling_kimi_linear`` does; the decoder model calls H0
  first.
* GPU stalls (``torch.cuda._sleep``) turn a missing stream wait into wrong bytes instead of a
  lucky race.  Each forward runs ``slow-compute`` (compute stalls before the attention reads and
  before the MoE-dense reads that follow the experts, so a copy that does not wait for its
  slot's reads overwrites them) or ``slow-copies`` (the copy stream stalls before every dense and
  expert copy, so a read that does not wait for its copy sees old bytes).
* The real ``DwdpManager``: ``setup`` (with ``setup_dwdp`` replaced by a fake expert weight
  manager, then the real ``setup_dense_dwdp`` with the loopback row store),
  ``prefetch_first_layers`` (the executor loop, on the default stream), ``ensure_primed`` (H0, on
  the execution stream), ``dense`` and ``cleanup``.  The fake expert weight manager owns a real
  CUDA copy stream, which the dense gathers share: it copies each MoE layer's routed-expert
  weights into a two-slot page pool, binds them at ``wait_and_bind`` and records consume events,
  so expert and dense copies interleave on one FIFO with real data dependencies.
* ``LoopbackGroup``: an ``N``-rank group whose ``allgather`` returns this rank's report ``N``
  times (``N`` = 4 and 16).  The default copy launcher runs in each transfer mode: the real
  ``trtllm::dwdp_fabric_gather``, the batched copy engine and per-slice copies.

The references are the model's outputs right before dense setup, inside ``DwdpManager.setup``
after the expert setup, so the same model with its replicated dense weights computes them.

Checked:

* Setup: five collectives; the commit log line; the caching allocator blocks setup frees are
  exactly the moved originals and the ones it adds are the slots, the store and the launcher
  arguments, sized by the layout rules; every alias is a slot view with its own
  size, stride and relative offset, Parameter objects and buffer registrations are kept.
* Forwards with varying token counts are bit-identical to the references: primed by the executor
  loop or only by H0, one at a time or back to back without a host synchronization (the priming
  then races the previous forward's last layers), under both stall patterns; the copy FIFO is
  ``L(1) E(1) L(2) E(2) ...``.  A forward around H0 takes two fenced demand fetches and is exact.
* Faults after H1 of a covered layer (in the attention, or in the experts after
  ``wait_and_bind``) are recovered by the next forward through the WAR barrier and re-priming;
  a fault in replicated layer 0 keeps the priming.  After such a fault, a forward captured into
  a CUDA graph is refused at H0 before it records or copies anything, and the next eager forward
  still recovers.  ``STRICT`` runs make no demand fetch and no fence fallback; ``DEBUG=1`` fills
  the slots with NaN at commit.  A second ``DwdpManager.setup`` keeps the committed runtime.
* ``DEBUG=2``: the slot watchdog checks every slot each H1 waits for against the layer's P1
  checksum (the device checksum equals the host one bit for bit) and finds nothing on every path
  above; it names the layer and segment of a store row changed after setup (``STRICT``: the
  last layer's H3 raises) and catches H1 reads that do not wait for their copies.
* ``cleanup`` releases dense DWDP before the expert teardown, leaves zero-size module tensors and
  returns the allocator to the pre-setup blocks minus the moved originals; it also frees
  everything, without an error or a warning, when the model was collected first (the executor
  shutdown order).
* Failures: ``REQUIRED=0`` with a Phase V mismatch, or with a P5 self-check failure (after the
  repoint, so the modules are rolled back), keeps the replicated weights and frees everything;
  ``REQUIRED=1`` fails ``DwdpManager.setup``.  A hidden holder of an original is a reported leak
  (``STRICT``: the setup fails after releasing the rank's dense state).
* ``TRTLLM_DWDP_DENSE`` unset makes one collective and changes nothing; ``MIN_BYTES=1`` moves
  every storage under the roots; ``SEGMENTS=attn``, the mode of the attention-only cluster
  canary (S9), moves only the attention weights while the MoE-dense block keeps its own.
"""

from __future__ import annotations

import contextlib
import gc
import math
import re
import weakref
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.modules.dwdp import fabric_gather
from tensorrt_llm._torch.modules.dwdp.dense import config as dense_config
from tensorrt_llm._torch.modules.dwdp.dense import plan as dense_plan
from tensorrt_llm._torch.modules.dwdp.dense import repoint as dense_repoint
from tensorrt_llm._torch.modules.dwdp.dense import runtime as dense_runtime
from tensorrt_llm._torch.modules.dwdp.dense import setup as dense_setup
from tensorrt_llm._torch.modules.dwdp.dense import store as dense_store
from tensorrt_llm._torch.modules.dwdp.dense import types as dense_types
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.layout import (
    checksum_bits,
    collect_dense_layout,
    device_checksum,
    tensor_checksum,
)
from tensorrt_llm._torch.modules.dwdp.dense.plan import DenseCopyLauncher
from tensorrt_llm._torch.modules.dwdp.dense.repoint import DenseBindings
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.setup import PhaseReport
from tensorrt_llm._torch.modules.dwdp.dense.store import LoopbackShardStore
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    RING_DEPTH,
    ROW_ALIGN,
    STORE_ALIGN,
    CopyMode,
    DenseLayout,
    DenseProtocolError,
    DenseRepointError,
    DenseSetupError,
    SegmentKind,
    align_up,
    ceil_div,
)
from tensorrt_llm._torch.modules.dwdp.rendezvous import DWDP_MASTER_ADDR_ENV, DWDP_RANK_ENV
from tensorrt_llm._torch.pyexecutor import dwdp as dwdp_facade
from tensorrt_llm._torch.pyexecutor.dwdp import (
    DwdpManager,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the dense DWDP end-to-end test needs a CUDA GPU"
)

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
FP32 = torch.float32

# ---------------------------------------------------------------------------
# Toy Kimi-K3 decoder
# ---------------------------------------------------------------------------

#: The model tables (``modeling_kimi_linear``), verbatim.
ATTR_ALLOW = frozenset(
    {
        "_bfa_proj_weight",
        "_qkvg_proj_weight",
        "_gate_b_t",
        "_packed_conv_weight",
        "_mtp_conv_weights",
        "_w_q_t",
        "_w_k_t",
        "_w_v_t",
        "_A_log_f32",
        "_dt_bias_f32",
        "_onorm_w_f32",
    }
)
EXCLUDE = frozenset(
    {
        "_o_dense",
        "_fused_q_placeholder",
        "softmax_stats_tensor",
        "temp_softmax_stats_tensor",
        "e_score_correction_bias",
    }
)
#: Excluded names in the toy that view no moved storage; they stay at any threshold.
UNALIASED_EXCLUDED = frozenset(
    {"e_score_correction_bias", "_fused_q_placeholder", "softmax_stats_tensor"}
)
#: The MoE-dense roots, in walk order.
MOE_ROOTS = (
    "gate",
    "shared_experts",
    "routed_expert_down_proj",
    "routed_expert_up_proj",
    "routed_expert_norm",
)

HIDDEN = 1024
VOCAB = 97
#: Rows of each of q, k, v and g in the fused FP8 ``qkvg`` projection.
KDA_HEAD = 320
#: ``_bfa_proj_weight`` rows: ``f_a``, ``b`` and pad rows no view covers.
F_ROWS, B_ROWS, PAD_ROWS = 600, 16, 8
MTP_ROWS = 520
MLA_LORA = 1088
ABSORB_HEADS, ABSORB_DIM = 8, 128
NUM_EXPERTS = 288
SHARED_INTER = 1024
LATENT = 1152
EXPERT_ROWS = 8
#: Layer 0 stays replicated (dense MLP, no roots); MLA layers sit among the KDA layers.
LAYER_KINDS = ("dense", "kda", "kda", "mla", "kda", "kda", "kda", "mla", "kda")
COVERED = tuple(range(1, len(LAYER_KINDS)))
MIDDLE_LAYER = 4
SEED = 1234
TOKEN_COUNTS = (17, 1, 128, 3)
#: GPU cycles of one stall: about 1 ms, long against a toy layer's compute and copies.
STALL_CYCLES = 2_000_000
SLOW_COMPUTE = "slow-compute"
SLOW_COPIES = "slow-copies"
#: ``(tokens, executor_primed, stalls)`` of the forwards each run compares with the references.
FORWARDS = (
    (17, True, SLOW_COMPUTE),
    (1, False, SLOW_COPIES),
    (128, True, SLOW_COPIES),
    (3, False, SLOW_COMPUTE),
    (17, False, SLOW_COPIES),
    (1, True, SLOW_COMPUTE),
)

#: Paths, relative to a covered layer, that dense DWDP moves: every alias of a moved storage,
#: including excluded names, parameter views and tuple items.
KDA_MOVED = (
    "linear_attn._bfa_proj_weight",
    "linear_attn._mtp_conv_weights[0]",
    "linear_attn._o_dense",
    "linear_attn.qkvg_proj.weight",
    "linear_attn.q_proj.weight",
    "linear_attn.k_proj.weight",
    "linear_attn.v_proj.weight",
    "linear_attn.g_proj.weight",
    "linear_attn.o_proj.weight",
    "linear_attn.f_a_proj.weight",
    "linear_attn.b_proj.weight",
    "linear_attn.f_b_proj.weight",
)
MLA_MOVED = (
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.kv_b_proj.weight_scale",
    "self_attn.v_b_proj_scale",
    "self_attn.o_proj.weight",
    "self_attn.k_b_proj_trans",
    "self_attn.k_b_proj_trans_dequant",
)
MOE_MOVED = (
    "block_sparse_moe.gate.weight",
    "block_sparse_moe.shared_experts.gate_up_proj.weight",
    "block_sparse_moe.shared_experts.down_proj.weight",
    "block_sparse_moe.routed_expert_down_proj.weight",
    "block_sparse_moe.routed_expert_up_proj.weight",
)


class InjectedFault(RuntimeError):
    """The fault a test injects into a forward."""


@dataclass
class StallPlan:
    """GPU cycles the toy model and the copies spin on the current stream; 0 means no stall."""

    compute: int = 0
    """Before the attention reads and before the MoE-dense reads after the experts."""

    copies: int = 0
    """On the copy stream, before every dense gather and every expert copy."""


#: The stalls of the forward running now; ``Harness.stalls`` sets them.
STALLS = StallPlan()


def _stall(cycles: int) -> None:
    if cycles:
        torch.cuda._sleep(cycles)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def _fail_if_armed(module: nn.Module) -> None:
    if module.fail_next:
        module.fail_next = False
        raise InjectedFault(f"fault injected into {type(module).__name__}")


class _Weights:
    """Seeded random weights made directly on the GPU."""

    def __init__(self, seed: int, device: torch.device) -> None:
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def normal(self, shape, scale: float, dtype: torch.dtype) -> torch.Tensor:
        values = torch.randn(shape, generator=self.generator, device=self.device) * scale
        return values.to(dtype)

    def fp8(self, shape) -> torch.Tensor:
        return self.normal(shape, 0.5, FP8)

    def bf16(self, shape, scale: float = 0.03) -> torch.Tensor:
        return self.normal(shape, scale, BF16)

    def scales(self, shape) -> torch.Tensor:
        return torch.randint(
            96, 160, shape, generator=self.generator, device=self.device, dtype=torch.uint8
        )


def _parameter(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor, requires_grad=False)


class ToyFp8Linear(nn.Module):
    """Stands in for ``_Fp8BlockScaleWeightReadLinear``: FP8 ``weight``, uint8 ``weight_scale``."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
        super().__init__()
        self._weights_transformed = True
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(FP32)
        scale = self.weight_scale.to(FP32).mean(dim=1) / (128.0 * math.sqrt(weight.shape[1]))
        return (x @ weight.t()) * scale


class ToyLinear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = _parameter(weight)


class ToyNorm(nn.Module):
    def __init__(self, weights: _Weights, size: int) -> None:
        super().__init__()
        self.weight = _parameter((1.0 + weights.normal((size,), 0.05, FP32)).to(BF16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms(x) * self.weight.to(FP32)


class ToyKDA(nn.Module):
    """KDA mixer after load.

    The fused FP8 ``qkvg_proj`` buffer with ``q/k/v/g_proj`` row views, ``o_proj``, the fused
    BF16 ``_bfa_proj_weight`` attribute whose rows ``f_a_proj`` and ``b_proj`` view (plus pad
    rows), ``f_b_proj``, the ``_mtp_conv_weights`` tuple (one element above the threshold), the
    excluded ``_o_dense`` aliasing ``o_proj.weight``, and small decode constants that stay
    replicated (``_A_log_f32`` shares ``A_log``'s storage).
    """

    def __init__(self, weights: _Weights) -> None:
        super().__init__()
        self.fail_next = False
        self.A_log = _parameter(weights.normal((64,), 0.5, FP32))
        fused = weights.fp8((4 * KDA_HEAD, HIDDEN))
        self.qkvg_proj = ToyFp8Linear(fused, weights.scales((4 * KDA_HEAD, 32)))
        for index, name in enumerate(("q_proj", "k_proj", "v_proj", "g_proj")):
            rows = fused[index * KDA_HEAD : (index + 1) * KDA_HEAD]
            setattr(self, name, ToyFp8Linear(rows, weights.scales((KDA_HEAD, 32))))
        self.o_proj = ToyFp8Linear(
            weights.fp8((HIDDEN, 4 * KDA_HEAD)), weights.scales((HIDDEN, 40))
        )
        bfa = weights.bf16((F_ROWS + B_ROWS + PAD_ROWS, HIDDEN))
        bfa[F_ROWS + B_ROWS :] = 0
        self.f_a_proj = ToyLinear(bfa[:F_ROWS])
        self.b_proj = ToyLinear(bfa[F_ROWS : F_ROWS + B_ROWS])
        self.f_b_proj = ToyLinear(weights.bf16((HIDDEN, F_ROWS)))
        self.o_norm = ToyNorm(weights, HIDDEN)
        self._bfa_proj_weight = bfa
        self._mtp_conv_weights = (
            weights.bf16((MTP_ROWS, HIDDEN)),
            weights.bf16((4, HIDDEN)),
            weights.bf16((4, HIDDEN)),
        )
        self._packed_conv_weight = weights.bf16((3, HIDDEN), 0.5)
        self._A_log_f32 = self.A_log.detach().float()
        self._o_dense = self.o_proj.weight[8:24]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        _stall(STALLS.compute)
        qkvg = self.qkvg_proj(h)
        _fail_if_armed(self)
        g_extra = self.g_proj(h)
        hb = h.to(BF16)
        fab = (hb @ self._bfa_proj_weight.t()).float()
        f_a = hb @ self.f_a_proj.weight.t()
        b = (hb @ self.b_proj.weight.t()).float()
        f_b = (f_a @ self.f_b_proj.weight.t()).float()
        mtp = sum((hb @ w.t()).float().mean(-1, keepdim=True) for w in self._mtp_conv_weights)
        conv = (hb * self._packed_conv_weight[0]).float().mean(-1, keepdim=True)
        decay = torch.exp(-torch.exp(self._A_log_f32)).mean()
        q, k, v, g = qkvg.split(KDA_HEAD, dim=-1)
        core = torch.tanh(q) * torch.sigmoid(g + g_extra) + torch.tanh(k) * torch.tanh(v)
        mixed = torch.cat([core, torch.tanh(qkvg[:, KDA_HEAD:])], dim=-1)
        out = self.o_norm(self.o_proj(mixed))
        gate = torch.sigmoid(b.mean(-1, keepdim=True) + fab.mean(-1, keepdim=True))
        return out * gate + 0.1 * _rms(f_b) + 0.1 * torch.tanh(mtp + conv) * decay


class ToyAttentionBackend:
    """Not an ``nn.Module``, so the walk never enters it (like MLA's ``mha`` / ``mqa``)."""

    def __init__(self, weights: _Weights) -> None:
        self.workspace = weights.bf16((2048,))


class ToyMLA(nn.Module):
    """MLA attention after load: FP8 projections, the BF16 absorb copy and excluded scratch.

    ``kv_b_proj.weight_scale`` is above the threshold and ``v_b_proj_scale`` is a parameter
    viewing part of it; ``_fused_q_placeholder`` is above the threshold but excluded, so it stays.
    """

    def __init__(self, weights: _Weights) -> None:
        super().__init__()
        self.fail_next = False
        self.kv_a_proj_with_mqa = ToyFp8Linear(
            weights.fp8((MLA_LORA, HIDDEN)), weights.scales((MLA_LORA, 32))
        )
        self.q_b_proj = ToyFp8Linear(weights.fp8((HIDDEN, MLA_LORA)), weights.scales((HIDDEN, 34)))
        self.kv_b_proj = ToyFp8Linear(
            weights.fp8((MLA_LORA, HIDDEN)), weights.scales((MLA_LORA, HIDDEN))
        )
        self.o_proj = ToyFp8Linear(weights.fp8((HIDDEN, MLA_LORA)), weights.scales((HIDDEN, 34)))
        self.k_b_proj_trans = _parameter(weights.fp8((ABSORB_HEADS, ABSORB_DIM, HIDDEN)))
        self.k_b_proj_trans_dequant = _parameter(weights.bf16((ABSORB_HEADS, ABSORB_DIM, HIDDEN)))
        self.v_b_proj_scale = _parameter(self.kv_b_proj.weight_scale[512:640])
        self._fused_q_placeholder = torch.zeros(F_ROWS, HIDDEN, dtype=BF16, device=weights.device)
        self.softmax_stats_tensor = torch.zeros(1024, dtype=FP32, device=weights.device)
        self.mha = ToyAttentionBackend(weights)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        _stall(STALLS.compute)
        qa = self.kv_a_proj_with_mqa(h)
        _fail_if_armed(self)
        q = self.q_b_proj(torch.tanh(qa))
        kv = self.kv_b_proj(q)
        v_scale = self.v_b_proj_scale.to(FP32).mean() / 128.0
        k_b = self.k_b_proj_trans.to(FP32).reshape(-1, HIDDEN)
        absorb = (q @ k_b.t()) / math.sqrt(HIDDEN)
        dequant = self.k_b_proj_trans_dequant.reshape(-1, HIDDEN)
        absorb_bf16 = (q.to(BF16) @ dequant.t()).float()
        out = self.o_proj(torch.tanh(kv))
        return out + 0.1 * torch.tanh(absorb) * v_scale + 0.1 * _rms(absorb_bf16)


class ToyGate(nn.Module):
    """The fp32 router of the attention-DP deployment, with its small excluded bias."""

    def __init__(self, weights: _Weights) -> None:
        super().__init__()
        self.weight = _parameter(weights.normal((NUM_EXPERTS, HIDDEN), 0.05, FP32))
        self.e_score_correction_bias = _parameter(weights.normal((NUM_EXPERTS,), 0.1, FP32))


class ToySharedMLP(nn.Module):
    def __init__(self, weights: _Weights) -> None:
        super().__init__()
        self.gate_up_proj = ToyFp8Linear(
            weights.fp8((2 * SHARED_INTER, HIDDEN)), weights.scales((2 * SHARED_INTER, 32))
        )
        self.down_proj = ToyFp8Linear(
            weights.fp8((HIDDEN, SHARED_INTER)), weights.scales((HIDDEN, 32))
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(h).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class ToyRoutedExperts(nn.Module):
    """Stands in for ``ConfigurableMoE``: registers with the DWDP manager and calls its protocol.

    Like ``CMOE:203-207``, ``CMOE:689-690`` and ``CMOE:708-709``: ``add_layer`` at construction,
    ``wait_and_bind`` before the expert compute, ``record_compute_and_prefetch_next`` right after
    it.  Expert DWDP owns this module; it is never a dense root.
    """

    def __init__(self, layer_idx: int, weights: _Weights) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.fail_next = False
        self.weight = _parameter(weights.bf16((EXPERT_ROWS, LATENT), 0.5))
        self.dwdp_manager = get_global_dwdp_manager()
        self.enable_dwdp = self.dwdp_manager is not None
        if self.enable_dwdp:
            self.dwdp_manager.add_layer(layer_idx)

    def forward(self, latent: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        if self.enable_dwdp:
            self.dwdp_manager.wait_and_bind(self, self.layer_idx)
        mix = probs[:, :EXPERT_ROWS] * NUM_EXPERTS
        out = torch.tanh(latent) * (mix @ self.weight.to(FP32))
        _fail_if_armed(self)
        if self.enable_dwdp:
            self.dwdp_manager.record_compute_and_prefetch_next(self.layer_idx)
        return out


class ToyMoE(nn.Module):
    """``KimiK3MoERuntime`` read order: router and W-down, experts, W-up, shared."""

    def __init__(self, layer_idx: int, weights: _Weights) -> None:
        super().__init__()
        self.gate = ToyGate(weights)
        self.shared_experts = ToySharedMLP(weights)
        self.routed_expert_down_proj = ToyFp8Linear(
            weights.fp8((LATENT, HIDDEN)), weights.scales((LATENT, 32))
        )
        self.routed_expert_up_proj = ToyFp8Linear(
            weights.fp8((HIDDEN, LATENT)), weights.scales((HIDDEN, 36))
        )
        self.routed_expert_norm = ToyNorm(weights, LATENT)
        self.routed_experts = ToyRoutedExperts(layer_idx, weights)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = h @ self.gate.weight.t()
        probs = torch.softmax(logits + self.gate.e_score_correction_bias, dim=-1)
        latent = self.routed_expert_norm(self.routed_expert_down_proj(h))
        experts = self.routed_experts(latent, probs)
        # The expert consume record above lets the copy stream run on; W-up and shared still read
        # this layer's MoE-dense slot.
        _stall(STALLS.compute)
        return self.routed_expert_up_proj(experts) + self.shared_experts(h)


class ToyDenseMLP(nn.Module):
    """Layer 0's replicated BF16 MLP."""

    def __init__(self, weights: _Weights) -> None:
        super().__init__()
        self.gate_up_proj = ToyLinear(weights.bf16((2 * SHARED_INTER, HIDDEN)))
        self.down_proj = ToyLinear(weights.bf16((HIDDEN, SHARED_INTER)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        gate, up = (h.to(BF16) @ self.gate_up_proj.weight.t()).float().chunk(2, dim=-1)
        return ((F.silu(gate) * up).to(BF16) @ self.down_proj.weight.t()).float()


class ToyDecoderLayer(nn.Module):
    """A Kimi-K3 decoder layer: the dense DWDP model seam and hooks H1-H3."""

    dense_dwdp_attr_allow = ATTR_ALLOW
    dense_dwdp_exclude = EXCLUDE

    def __init__(self, layer_idx: int, kind: str, weights: _Weights) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = kind != "dense"
        self.is_kda = kind != "mla"
        self.input_layernorm = ToyNorm(weights, HIDDEN)
        self.post_attention_layernorm = ToyNorm(weights, HIDDEN)
        if self.is_kda:
            self.linear_attn = ToyKDA(weights)
        else:
            self.self_attn = ToyMLA(weights)
        if self.is_moe:
            self.block_sparse_moe = ToyMoE(layer_idx, weights)
        else:
            self.mlp = ToyDenseMLP(weights)
        # M0: the DWDP manager exists before the model is built.
        self._dwdp = get_global_dwdp_manager()

    def dense_dwdp_roots(self, segment: str) -> list[tuple[str, nn.Module]]:
        if segment not in ("attn", "moe"):
            raise ValueError(f"unknown dense DWDP segment {segment!r}")
        if not self.is_moe:
            return []
        if segment == "attn":
            if self.is_kda:
                return [("linear_attn", self.linear_attn)]
            return [("self_attn", self.self_attn)]
        moe = self.block_sparse_moe
        return [(f"block_sparse_moe.{name}", getattr(moe, name)) for name in MOE_ROOTS]

    @property
    def attention(self) -> nn.Module:
        return self.linear_attn if self.is_kda else self.self_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dwdp = getattr(self, "_dwdp", None)
        dense = dwdp.dense if dwdp is not None else None
        if dense is not None and not dense.covers(self.layer_idx):
            dense = None
        if dense is not None:
            dense.wait_layer(self.layer_idx)  # H1
        h = self.input_layernorm(x)
        attention = self.attention(h)
        if dense is not None:
            dense.attention_done(self.layer_idx)  # H2
        x = x + attention
        h = self.post_attention_layernorm(x)
        mixed = self.block_sparse_moe(h) if self.is_moe else self.mlp(h)
        if dense is not None:
            dense.layer_done(self.layer_idx)  # H3
        return x + mixed


class ToyDecoderModel(nn.Module):
    """``KimiLinearModel``: the decoder with ``layers``; H0 at the top of ``forward``."""

    def __init__(self, weights: _Weights) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            VOCAB, HIDDEN, _weight=weights.bf16((VOCAB, HIDDEN), 1.0), _freeze=True
        )
        self.layers = nn.ModuleList(
            ToyDecoderLayer(layer_idx, kind, weights) for layer_idx, kind in enumerate(LAYER_KINDS)
        )
        self._dwdp = get_global_dwdp_manager()

    def forward(self, input_ids: torch.Tensor, *, skip_h0: bool = False) -> torch.Tensor:
        """``skip_h0`` stands for a path around this forward (demand fetches)."""
        dwdp = getattr(self, "_dwdp", None)
        if dwdp is not None and dwdp.dense is not None and not skip_h0:
            dwdp.ensure_primed()  # H0
        x = self.embed_tokens(input_ids).float()
        for layer in self.layers:
            x = layer(x)
        return x


class ToyCausalLM(nn.Module):
    """Like the deployed VL wrapper: ``model.model`` is the decoder that holds ``layers``."""

    def __init__(self, seed: int, device: torch.device) -> None:
        super().__init__()
        weights = _Weights(seed, device)
        self.model = ToyDecoderModel(weights)
        self.norm = ToyNorm(weights, HIDDEN)
        self.lm_head = ToyLinear(weights.bf16((VOCAB, HIDDEN)))

    def forward(self, input_ids: torch.Tensor, *, skip_h0: bool = False) -> torch.Tensor:
        hidden = self.norm(self.model(input_ids, skip_h0=skip_h0))
        return hidden @ self.lm_head.weight.to(FP32).t()


def expected_moved(kinds: tuple[SegmentKind, ...]) -> set[str]:
    moved = set()
    for layer, kind in enumerate(LAYER_KINDS):
        if kind == "dense":
            continue
        paths = KDA_MOVED if kind == "kda" else MLA_MOVED
        if MOE in kinds:
            paths = paths + MOE_MOVED
        moved.update(f"layers.{layer}.{path}" for path in paths)
    return moved


def segment_of(path: str) -> tuple[int, SegmentKind | None]:
    """``(layer, segment)`` of a module tensor path; None outside the dense roots."""
    _, layer, rest = path.split(".", 2)
    if rest.startswith(("linear_attn.", "self_attn.")):
        return int(layer), ATTN
    if rest.startswith(tuple(f"block_sparse_moe.{name}." for name in MOE_ROOTS)):
        return int(layer), MOE
    return int(layer), None


def is_expert_path(path: str) -> bool:
    """Routed-expert weights, which the fake expert weight manager binds to its page pool."""
    return ".routed_experts." in path


# ---------------------------------------------------------------------------
# Module tensors and allocator blocks
# ---------------------------------------------------------------------------


def module_tensors(decoder: nn.Module) -> dict[str, torch.Tensor]:
    """Every tensor the decoder layers hold, by ``layers.<l>.<path>``."""
    found = {}
    for index, layer in enumerate(decoder.layers):
        for path, module in layer.named_modules(prefix=f"layers.{index}"):
            for name, value in [*module._parameters.items(), *module._buffers.items()]:
                if value is not None:
                    found[f"{path}.{name}"] = value
            for name, value in vars(module).items():
                if isinstance(value, torch.Tensor):
                    found[f"{path}.{name}"] = value
                elif (
                    isinstance(value, (tuple, list))
                    and value
                    and all(isinstance(item, torch.Tensor) for item in value)
                ):
                    found.update({f"{path}.{name}[{i}]": item for i, item in enumerate(value)})
    return found


def placement(tensor: torch.Tensor) -> tuple:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


class TensorRecords:
    """The decoder's tensors before dense setup, holding none of their storages alive."""

    def __init__(self, decoder: nn.Module) -> None:
        self.placements: dict[str, tuple] = {}
        self.storage_refs: dict[str, weakref.ref] = {}
        self.storage_bytes: dict[str, int] = {}
        self.parameters: dict[str, nn.Parameter] = {}
        self.buffers: dict[str, tuple[nn.Module, str]] = {}
        for path, tensor in module_tensors(decoder).items():
            storage = tensor.untyped_storage()
            self.placements[path] = placement(tensor)
            self.storage_refs[path] = weakref.ref(storage)
            self.storage_bytes[path] = storage.nbytes()
            if isinstance(tensor, nn.Parameter):
                self.parameters[path] = tensor
        for index, layer in enumerate(decoder.layers):
            for path, module in layer.named_modules(prefix=f"layers.{index}"):
                for name in module._buffers:
                    self.buffers[f"{path}.{name}"] = (module, name)


def active_blocks(device: torch.device) -> dict[int, tuple[int, int]]:
    """``address -> (size, requested size)`` of every block the caching allocator hands out."""
    blocks = {}
    for segment in torch.cuda.memory_snapshot():
        if segment["device"] != device.index:
            continue
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                blocks[block["address"]] = (block["size"], block["requested_size"])
    return blocks


def _gb(nbytes: int) -> str:
    """How the dense setup log prints byte counts."""
    return f"{nbytes / 1e9:.2f}"


@dataclass(frozen=True)
class LayoutBytes:
    """Store handle sizes and slot allocation sizes of a dense layout."""

    handles: tuple[int, ...]
    slot_alloc: dict[SegmentKind, int]

    @property
    def store_bytes(self) -> int:
        return sum(self.handles)

    @classmethod
    def from_blobs(
        cls,
        blob_bytes: dict[tuple[int, SegmentKind], int],
        kinds: tuple[SegmentKind, ...],
        num_ranks: int,
        chunk_mb: int,
    ) -> LayoutBytes:
        """Row widths, store offsets and slot sizes from the blob sizes alone."""
        rows = {
            (layer, kind): align_up(
                ceil_div(blob_bytes.get((layer, kind), 0), num_ranks), ROW_ALIGN
            )
            for layer in COVERED
            for kind in kinds
        }
        chunks: list[int] = []
        used = 0
        for layer in COVERED:
            size = sum(rows[(layer, kind)] for kind in kinds)
            if not chunks or (chunk_mb > 0 and used > 0 and used + size > chunk_mb << 20):
                chunks.append(0)
                used = 0
            chunks[-1] += size
            used += size
        return cls(
            handles=tuple(align_up(chunk, STORE_ALIGN) for chunk in chunks),
            slot_alloc={
                kind: num_ranks * max(rows[(layer, kind)] for layer in COVERED) + ROW_ALIGN
                for kind in kinds
            },
        )

    @classmethod
    def from_layout(cls, layout: DenseLayout) -> LayoutBytes:
        return cls(
            handles=tuple(chunk.handle_bytes for chunk in layout.chunks),
            slot_alloc={kind: spec.alloc_bytes for kind, spec in layout.slot_specs.items()},
        )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class CaptureLogger:
    """Stands in for the TensorRT-LLM logger of the dense modules and keeps every message."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _add(self, level: str, message: object) -> None:
        self.records.append((level, str(message)))

    def debug(self, message: object) -> None:
        self._add("debug", message)

    def info(self, message: object) -> None:
        self._add("info", message)

    def warning(self, message: object) -> None:
        self._add("warning", message)

    def error(self, message: object) -> None:
        self._add("error", message)

    def messages(self, *levels: str) -> list[str]:
        return [message for level, message in self.records if level in levels]


class LoopbackGroup:
    """An ``N``-rank DWDP group whose ranks all live in this process.

    Every rank of a real group holds the same replicated weights, so every rank reports what this
    one does: ``allgather`` returns this rank's object ``N`` times.  The loopback store holds all
    ``N`` ranks' rows.
    """

    def __init__(self, rank: int, size: int, log: list[tuple]) -> None:
        self.rank = rank
        self.size = size
        self.log = log
        self.gathered: list[object] = []

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def allgather(self, obj: object) -> list[object]:
        self.gathered.append(obj)
        return [obj] * self.size

    def Barrier(self) -> None:
        pass

    def Free(self) -> None:
        self.log.append(("group.Free",))


class FakeExpertWeightManager:
    """The ``DWDPWeightManager`` surface ``DwdpManager`` and dense setup use, on real CUDA streams.

    Each MoE layer's routed-expert weights stay where the model loaded them; ``prefetch_layer(l)``
    copies them on the copy stream into page-pool slot ``position(l) % 2`` once the slot's consume
    event allows it (after the ``STALLS.copies`` stall), ``wait_and_bind`` makes the current
    stream wait for that copy and binds the slot as the module's weight, and ``record_compute``
    records the slot's consume event on the current stream.  ``wait_and_bind`` of a layer whose
    experts were not prefetched raises, which turns a missing expert prime into a test failure.
    Every call is appended to the shared log.
    """

    def __init__(
        self,
        experts: dict[int, nn.Module],
        device: torch.device,
        *,
        use_gather: bool,
        use_batched_prefetch: bool,
        log: list[tuple],
    ) -> None:
        self._layers = tuple(sorted(experts))
        self._positions = {layer: position for position, layer in enumerate(self._layers)}
        self._sources = {layer: module.weight.data for layer, module in experts.items()}
        template = self._sources[self._layers[0]]
        self._pool = tuple(torch.empty_like(template) for _ in range(2))
        self._copy_stream = torch.cuda.Stream(device)
        self._ready = tuple(torch.cuda.Event() for _ in range(2))
        self._consume = tuple(torch.cuda.Event() for _ in range(2))
        current = torch.cuda.current_stream(device)
        for event in self._consume:
            event.record(current)
        self._tags: list[int | None] = [None, None]
        self._use_gather = use_gather
        self._use_batched_prefetch = use_batched_prefetch
        self.log = log
        self.on_release = None

    @property
    def copy_stream(self) -> torch.cuda.Stream:
        return self._copy_stream

    @property
    def use_gather(self) -> bool:
        return self._use_gather

    @property
    def use_batched_prefetch(self) -> bool:
        return self._use_batched_prefetch

    def first_moe_layer(self) -> int:
        return self._layers[0]

    def next_moe_layer(self, layer: int) -> int | None:
        return next((later for later in self._layers if later > layer), None)

    def _slot(self, layer: int) -> int:
        return self._positions[layer] % 2

    def slot_index(self, layer: int) -> int:
        """The slot of ``layer`` (``DwdpManager`` asks when the cross-forward prefetch is on)."""
        return self._slot(layer)

    def prefetch_layer(self, layer: int) -> None:
        self.log.append(("E", layer))
        slot = self._slot(layer)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._consume[slot])
            _stall(STALLS.copies)
            self._pool[slot].copy_(self._sources[layer])
            self._ready[slot].record(self._copy_stream)
        self._tags[slot] = layer

    def wait_and_bind(self, backend: nn.Module, layer: int) -> None:
        self.log.append(("wait_and_bind", layer))
        slot = self._slot(layer)
        if self._tags[slot] != layer:
            raise AssertionError(
                f"the experts of layer {layer} were not prefetched: their slot {slot} holds "
                f"layer {self._tags[slot]}"
            )
        torch.cuda.current_stream().wait_event(self._ready[slot])
        backend.weight.data = self._pool[slot]

    def record_compute(self, layer: int) -> None:
        self.log.append(("record_compute", layer))
        self._consume[self._slot(layer)].record(torch.cuda.current_stream())

    def release(self) -> None:
        self.log.append(("wm.release", None if self.on_release is None else self.on_release()))


class CorruptingLoopbackStore(LoopbackShardStore):
    """A loopback store whose copy of one peer's rows has one flipped byte.

    The byte is the first one of row ``victim`` of the first ``(layer, segment)``, which lies
    inside an anchor; Phase V compares the gathered anchors with the originals and must catch it.
    """

    def __init__(self, layout, rank, device, *, victim: int) -> None:
        super().__init__(layout, rank, device)
        self._victim = victim
        first = next(iter(layout.segments.values()))
        self._target = (first.chunk, first.chunk_offset)

    def _write_rows(self, row, chunks) -> None:
        super()._write_rows(row, chunks)
        if row == self._victim:
            chunk, offset = self._target
            chunks[chunk][offset : offset + 1].bitwise_not_()


def require_copy_mode(mode: CopyMode, device: torch.device) -> None:
    """Skip unless this build and GPU can run ``mode``."""
    if mode is CopyMode.BATCHED:
        pytest.importorskip("cuda.bindings.runtime")
        return
    if mode is not CopyMode.GATHER:
        return
    op = fabric_gather.gather_op()
    if op is None:
        pytest.skip("trtllm::dwdp_fabric_gather is not registered by this TensorRT-LLM build")
    src = torch.arange(fabric_gather.LINE_BYTES, dtype=torch.uint8, device=device)
    dst = torch.zeros_like(src)

    def int64(values):
        return torch.tensor(values, dtype=torch.int64, device=device)

    try:
        op(int64([dst.data_ptr()]), int64([src.data_ptr()]), int64([1]), int64([0, 1]), 1, 32)
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        if "no kernel image" in str(exc):
            pytest.skip(f"this build has no gather kernel for this GPU: {exc}")
        raise
    assert torch.equal(dst, src), "a one-line probe gather did not copy its 16 bytes"


def assert_same(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    if torch.equal(actual, expected):
        return
    nans = int(torch.isnan(actual).sum())
    finite = torch.isfinite(actual)
    worst = (actual - expected).abs()[finite].max().item() if bool(finite.any()) else math.nan
    raise AssertionError(f"{what}: differs from its reference ({nans} NaN, max |diff| {worst})")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_DENSE_ENVS = (
    dense_config.DENSE_ENV,
    dense_config.DENSE_SEGMENTS_ENV,
    dense_config.DENSE_REQUIRED_ENV,
    dense_config.DENSE_MIN_BYTES_ENV,
    dense_config.DENSE_STORE_CHUNK_MB_ENV,
    dense_config.DENSE_STRICT_ENV,
    dense_config.DENSE_DEBUG_ENV,
)
#: The dense modules that log; ``layout`` reports through ``types.warn_or_raise``.
_DENSE_MODULES = (
    dense_types,
    dense_config,
    dense_repoint,
    dense_runtime,
    dense_plan,
    dense_store,
    dense_setup,
)
COMMIT_LINE = re.compile(
    r"^\[DWDP Dense\] committed: moved=(?P<moved>-?[\d.]+)GB store=(?P<store>-?[\d.]+)GB "
    r"slots=(?P<slots>-?[\d.]+)GB freed\(mem_get_info\)=(?P<freed_device>-?[\d.]+)GB "
    r"freed\(memory_allocated\)=(?P<freed_torch>-?[\d.]+)GB leaks=(?P<leaks>\d+) "
    r"layers=(?P<layers>\d+) segments=(?P<segments>[a-z,]+) setup=(?P<setup>[\d.]+)s$"
)
#: Largest launcher argument tensor (one gather plan's int64 descriptor column).
_LAUNCHER_ARGUMENT_BYTES = 4096


@dataclass(frozen=True)
class Scenario:
    """One run: transfer, group and ``TRTLLM_DWDP_DENSE_*`` switches."""

    mode: CopyMode
    enabled: bool = True
    num_ranks: int = 16
    rank: int = 5
    segments: str = "attn,moe"
    chunk_mb: int = 0
    strict: bool = False
    debug: int = 0
    required: bool | None = None
    min_bytes: int | None = None
    corrupt_row: int | None = None
    hidden_holder: str | None = None
    """A moved module tensor that the test also holds, outside the model, from before setup."""

    @property
    def kinds(self) -> tuple[SegmentKind, ...]:
        return (ATTN, MOE) if "moe" in self.segments.split(",") else (ATTN,)


class Harness:
    """One toy model, one real ``DwdpManager`` and the fakes around them, plus the checks."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, scenario: Scenario) -> None:
        self.scenario = scenario
        self.device = torch.device("cuda", torch.cuda.current_device())
        require_copy_mode(scenario.mode, self.device)
        self.log: list[tuple] = []
        self.capture = CaptureLogger()
        self.stores: list[LoopbackShardStore] = []
        self.references: dict[int, torch.Tensor] = {}
        self.moved: set[str] = set()
        self.hidden: torch.Tensor | None = None
        self.weight_manager: FakeExpertWeightManager | None = None
        self._configure_environment(monkeypatch)
        self._patch_seams(monkeypatch)

        set_global_dwdp_manager(None)
        self.group = LoopbackGroup(scenario.rank, scenario.num_ranks, self.log)
        world = MagicMock()
        world.Create_group.return_value = self.group
        world.Get_size.return_value = 1 << 12
        monkeypatch.setattr(dwdp_facade, "COMM_WORLD", world)
        monkeypatch.setattr(dwdp_facade, "dwdp_global_rank", lambda: scenario.rank)
        self.manager = DwdpManager(
            config=DwdpConfig(
                dwdp_size=scenario.num_ranks,
                num_groups=1,
                num_experts_per_worker=4,
                num_prefetch_experts=4,
            ),
            dist=MagicMock(spec=MPIDist),
            mapping=Mapping(
                world_size=1,
                rank=0,
                tp_size=1,
                dwdp_size=scenario.num_ranks,
                dwdp_rank=scenario.rank,
            ),
        )
        set_global_dwdp_manager(self.manager)
        self.model = ToyCausalLM(SEED, self.device)
        generator = torch.Generator(device=self.device).manual_seed(SEED + 1)
        self.inputs = {
            tokens: torch.randint(0, VOCAB, (tokens,), generator=generator, device=self.device)
            for tokens in TOKEN_COUNTS
        }
        self.execution_stream = torch.cuda.Stream(self.device)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _configure_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scenario = self.scenario
        for name in (*_DENSE_ENVS, DWDP_RANK_ENV, DWDP_MASTER_ADDR_ENV):
            monkeypatch.delenv(name, raising=False)
        if scenario.enabled:
            monkeypatch.setenv(dense_config.DENSE_ENV, "1")
        monkeypatch.setenv(dense_config.DENSE_SEGMENTS_ENV, scenario.segments)
        if scenario.chunk_mb:
            monkeypatch.setenv(dense_config.DENSE_STORE_CHUNK_MB_ENV, str(scenario.chunk_mb))
        if scenario.strict:
            monkeypatch.setenv(dense_config.DENSE_STRICT_ENV, "1")
        if scenario.debug:
            monkeypatch.setenv(dense_config.DENSE_DEBUG_ENV, str(scenario.debug))
        if scenario.required is not None:
            monkeypatch.setenv(dense_config.DENSE_REQUIRED_ENV, str(int(scenario.required)))
        if scenario.min_bytes is not None:
            monkeypatch.setenv(dense_config.DENSE_MIN_BYTES_ENV, str(scenario.min_bytes))

    def _patch_seams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for module in _DENSE_MODULES:
            monkeypatch.setattr(module, "logger", self.capture)

        launch = DenseCopyLauncher.__call__

        def recording_launch(launcher: DenseCopyLauncher, layer: int) -> None:
            # The runtime calls the launcher with the copy stream current.
            self.log.append(("L", layer))
            _stall(STALLS.copies)
            launch(launcher, layer)

        monkeypatch.setattr(DenseCopyLauncher, "__call__", recording_launch)
        monkeypatch.setattr(dwdp_facade, "setup_dwdp", self._setup_expert_dwdp)
        real_setup_dense = dwdp_facade.setup_dense_dwdp

        def setup_dense_after_references(model, weight_manager, comm, **kwargs):
            if model is not self.model or weight_manager is not self.weight_manager:
                raise AssertionError("DwdpManager.setup passed another model or weight manager")
            if comm is not self.group:
                raise AssertionError("DwdpManager.setup passed another DWDP group")
            self._record_references()
            return real_setup_dense(
                model, weight_manager, comm, store_factory=self._make_store, **kwargs
            )

        monkeypatch.setattr(dwdp_facade, "setup_dense_dwdp", setup_dense_after_references)

    def _setup_expert_dwdp(self, *, model, mapping, device_id, comm, layer_indices, **_kwargs):
        if comm is not self.group or device_id != self.device.index:
            raise AssertionError(f"setup_dwdp got group {comm!r} and device {device_id}")
        if list(layer_indices) != list(COVERED):
            raise AssertionError(f"setup_dwdp got MoE layers {layer_indices}")
        decoder = model.model
        experts = {
            layer: decoder.layers[layer].block_sparse_moe.routed_experts for layer in COVERED
        }
        self.weight_manager = FakeExpertWeightManager(
            experts,
            self.device,
            use_gather=self.scenario.mode is CopyMode.GATHER,
            use_batched_prefetch=self.scenario.mode is CopyMode.BATCHED,
            log=self.log,
        )
        self.weight_manager.on_release = self._state_at_expert_release
        return self.weight_manager

    def _make_store(self, layout, rank, device) -> LoopbackShardStore:
        victim = self.scenario.corrupt_row
        if victim is None:
            store = LoopbackShardStore(layout, rank, device)
        else:
            store = CorruptingLoopbackStore(layout, rank, device, victim=victim)
        self.stores.append(store)
        return store

    def flip_store_byte(self, layer: int, kind: SegmentKind) -> None:
        """Flip blob byte 0 of ``(layer, kind)`` in rank 0's row of the committed loopback store.

        Byte 0 is the first byte of the segment's first anchor; every later gather of the layer
        brings the flipped byte into its slot.  Flipping it again restores the row.
        """
        store = self.stores[-1]
        segment = store._layout.segment(layer, kind)
        row = store._chunks[0][segment.chunk]
        row[segment.chunk_offset : segment.chunk_offset + 1].bitwise_not_()
        torch.cuda.synchronize(self.device)

    def _state_at_expert_release(self) -> tuple[bool, set[int]]:
        """Whether dense DWDP is gone, and the sizes of the moved module tensors, at that time."""
        tensors = module_tensors(self.model.model)
        return self.manager.dense is None, {tensors[path].numel() for path in self.moved}

    def close(self) -> None:
        try:
            if self.manager.dwdp_group is not None:
                self.manager.cleanup()
        finally:
            set_global_dwdp_manager(None)

    # ------------------------------------------------------------------
    # Forwards
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def stalls(self, regime: str | None) -> Iterator[None]:
        """Run the block with the stalls of ``regime`` (``SLOW_COMPUTE`` or ``SLOW_COPIES``)."""
        STALLS.compute = STALL_CYCLES if regime == SLOW_COMPUTE else 0
        STALLS.copies = STALL_CYCLES if regime == SLOW_COPIES else 0
        try:
            yield
        finally:
            STALLS.compute = STALLS.copies = 0

    def forward(
        self, tokens: int, *, synchronize: bool = True, skip_h0: bool = False
    ) -> torch.Tensor:
        """One forward on the execution stream, like ``PyExecutor._forward_step``."""
        with torch.no_grad(), torch.cuda.stream(self.execution_stream):
            output = self.model(self.inputs[tokens], skip_h0=skip_h0)
        if synchronize:
            self.execution_stream.synchronize()
        return output

    def copies(self) -> list[tuple[str, int]]:
        """The dense (``L``) and expert (``E``) copies enqueued since the log was cleared."""
        return [entry for entry in self.log if entry[0] in ("L", "E")]

    @staticmethod
    def edf(layers=COVERED) -> list[tuple[str, int]]:
        return [(kind, layer) for layer in layers for kind in ("L", "E")]

    def _record_references(self) -> None:
        """Inside ``DwdpManager.setup``, after the expert setup: references and the pre-setup state."""
        if self.manager.dense is not None:
            raise AssertionError("dense DWDP is active before its setup")
        for tokens in TOKEN_COUNTS:
            self.manager.prefetch_first_layers()
            output = self.forward(tokens)
            assert torch.isfinite(output).all(), f"the {tokens}-token reference is not finite"
            self.references[tokens] = output
        self.log.clear()
        torch.cuda.synchronize(self.device)
        gc.collect()
        self.collected = LayoutBytes.from_layout(
            collect_dense_layout(
                list(enumerate(self.model.model.layers)),
                DenseConfig.from_env(),
                num_ranks=self.scenario.num_ranks,
                device=self.device,
                attr_allow=ATTR_ALLOW,
                exclude=EXCLUDE,
            )
        )
        gc.collect()
        self.before = TensorRecords(self.model.model)
        if self.scenario.hidden_holder is not None:
            self.hidden = module_tensors(self.model.model)[self.scenario.hidden_holder].detach()
        self.blocks_before = active_blocks(self.device)
        self.allocated_before = torch.cuda.memory_allocated(self.device)
        assert sum(size for size, _ in self.blocks_before.values()) == self.allocated_before

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        result = self.manager.setup(self.model)
        assert result is self.weight_manager
        self.after_setup()

    def after_setup(self) -> None:
        torch.cuda.synchronize(self.device)
        gc.collect()
        self.blocks_after_setup = active_blocks(self.device)
        self.allocated_after_setup = torch.cuda.memory_allocated(self.device)
        self.setup_copies = self.copies()
        self.log.clear()

    def expected_moved(self) -> set[str]:
        """Module tensor paths that must end up as slot views."""
        kinds = self.scenario.kinds
        if self.scenario.min_bytes is None:
            return expected_moved(kinds)
        # Every storage under the enabled roots of the covered layers moves, except the excluded
        # names that alias nothing that moves.
        moved = set()
        for path in self.before.placements:
            layer, kind = segment_of(path)
            if layer in COVERED and kind in kinds:
                if path.rsplit(".", 1)[1] not in UNALIASED_EXCLUDED:
                    moved.add(path)
        return moved

    def hidden_storages(self) -> set[int]:
        """The original storage the test holds outside the model, if any."""
        if self.scenario.hidden_holder is None:
            return set()
        return {self.before.placements[self.scenario.hidden_holder][0]}

    def originals(self) -> dict[int, int]:
        """Block address -> size of every moved storage that setup frees."""
        originals = {}
        hidden = self.hidden_storages()
        for path in self.moved:
            address = self.before.placements[path][0]
            if address not in hidden:
                originals[address] = self.blocks_before[address][0]
        return originals

    def blob_bytes(self) -> LayoutBytes:
        """The expected sizes from the moved storages, which must be whole pages."""
        blob_bytes: dict[tuple[int, SegmentKind], int] = {}
        seen = set()
        for path in sorted(self.moved):
            address = self.before.placements[path][0]
            if address in seen:
                continue
            seen.add(address)
            nbytes = self.before.storage_bytes[path]
            assert nbytes % ROW_ALIGN == 0, f"{path}: the oracle assumes page-sized storages"
            key = segment_of(path)
            blob_bytes[key] = blob_bytes.get(key, 0) + nbytes
        return LayoutBytes.from_blobs(
            blob_bytes, self.scenario.kinds, self.scenario.num_ranks, self.scenario.chunk_mb
        )

    def check_committed(self) -> DenseDWDP:
        """Everything setup phase P6 promises, right after ``DwdpManager.setup`` returned."""
        scenario = self.scenario
        dense = self.manager.dense
        assert isinstance(dense, DenseDWDP)
        assert dense.layers == COVERED and dense.segments == scenario.kinds
        assert not dense.covers(0)
        assert (dense.primed, dense.forward_open) == (False, False)
        assert self.counters(dense) == (0, 0)
        assert set(dense.slots) == set(scenario.kinds)
        if MOE not in scenario.kinds:
            with pytest.raises(DenseProtocolError, match="not enabled"):
                dense.slot_index(MOE, COVERED[0])
        assert self.model._dense_dwdp_active and self.model.model._dense_dwdp_active

        # Five collectives, all ok; Phase V and the P5 self-check each gathered every layer once.
        assert len(self.group.gathered) == 5
        assert all(isinstance(report, PhaseReport) and report.ok for report in self.group.gathered)
        assert self.setup_copies == [("L", layer) for layer in COVERED] * 2

        # Every alias of every moved storage is a slot view with its own size, stride and offset.
        slots = [slot for ring in dense.slots.values() for slot in ring]
        slot_storages = {slot.storage.untyped_storage().data_ptr() for slot in slots}
        tensors = module_tensors(self.model.model)
        self.moved = {
            path
            for path, tensor in tensors.items()
            if tensor.untyped_storage().data_ptr() in slot_storages
        }
        assert self.moved == self.expected_moved()
        for path, tensor in tensors.items():
            before = self.before.placements[path]
            if path in self.moved:
                assert placement(tensor)[2:] == before[2:], path
            elif not is_expert_path(path):
                assert placement(tensor) == before, f"{path} moved but is not a dense binding"
        by_storage: dict[int, list[str]] = {}
        for path in self.moved:
            by_storage.setdefault(self.before.placements[path][0], []).append(path)
        for paths in by_storage.values():
            anchor = paths[0]
            for path in paths[1:]:
                old = self._byte_offset(self.before.placements[path])
                old -= self._byte_offset(self.before.placements[anchor])
                new = self._byte_offset(placement(tensors[path]))
                new -= self._byte_offset(placement(tensors[anchor]))
                assert new == old, f"{path} lost its offset relative to {anchor}"
                same = tensors[path].untyped_storage().data_ptr()
                assert same == tensors[anchor].untyped_storage().data_ptr(), path
        for path, parameter in self.before.parameters.items():
            assert tensors[path] is parameter, f"{path} is a new Parameter object"
        for path, (module, name) in self.before.buffers.items():
            assert name in module._buffers, f"{path} is no longer a registered buffer"
        del tensors
        hidden = self.hidden_storages()
        alive = sorted(path for path in self.moved if self.before.storage_refs[path]() is not None)
        held = sorted(path for path in self.moved if self.before.placements[path][0] in hidden)
        assert alive == held, f"original storages that survived the commit: {alive[:4]}"

        # Slots are filled at commit: zeros, or NaN bytes under DEBUG.
        fill = 0xFF if scenario.debug else 0
        for slot in slots:
            values = slot.storage
            assert int(values.min()) == fill and int(values.max()) == fill
        if scenario.debug:
            assert torch.isnan(slots[0].storage[:ROW_ALIGN].view(BF16)).all()

        # Memory: setup freed exactly the moved originals and added the slots, the store and the
        # launcher arguments.
        if scenario.min_bytes is None:
            expected = self.blob_bytes()
            assert expected == self.collected, "the collector and the layout rules disagree"
        else:
            expected = self.collected
        originals = self.originals()
        freed = set(self.blocks_before) - set(self.blocks_after_setup)
        assert freed == set(originals)
        added = {
            address: requested
            for address, (_, requested) in self.blocks_after_setup.items()
            if address not in self.blocks_before
        }
        assert {slot.storage.data_ptr() for slot in slots} <= set(added)
        planned = Counter()
        for kind in scenario.kinds:
            planned[expected.slot_alloc[kind]] += RING_DEPTH[kind]
        for handle in expected.handles:
            planned[handle + ROW_ALIGN] += scenario.num_ranks
        allocated = Counter(added.values())
        assert planned - allocated == Counter(), "a slot or store allocation is missing"
        leftover = allocated - planned
        if scenario.mode is CopyMode.GATHER:
            assert all(size <= _LAUNCHER_ARGUMENT_BYTES for size in leftover.elements())
        else:
            assert leftover == Counter(), f"unexpected allocations {leftover}"
        for slot in slots:
            kind = slot.segment
            assert slot.storage.numel() == expected.slot_alloc[kind]

        # The commit line.
        lines = [
            message
            for message in self.capture.messages("info")
            if message.startswith("[DWDP Dense] committed:")
        ]
        assert len(lines) == 1, lines
        match = COMMIT_LINE.match(lines[0])
        assert match, lines[0]
        moved_bytes = sum(self.before.storage_bytes[paths[0]] for paths in by_storage.values())
        slot_bytes = sum(slot.storage.numel() for slot in slots)
        assert match["moved"] == _gb(moved_bytes)
        assert match["store"] == _gb(expected.store_bytes)
        assert match["slots"] == _gb(slot_bytes)
        assert match["freed_torch"] == _gb(self.allocated_before - self.allocated_after_setup)
        assert self.allocated_before - self.allocated_after_setup == sum(originals.values()) - sum(
            self.blocks_after_setup[address][0] for address in added
        )
        assert match["leaks"] == str(len(hidden))
        assert match["layers"] == str(len(COVERED))
        assert match["segments"] == ",".join(kind.value for kind in scenario.kinds)
        agreed = [m for m in self.capture.messages("info") if "layout agreed by" in m]
        assert len(agreed) == 1 and f"agreed by {scenario.num_ranks} ranks" in agreed[0]
        if hidden:
            leaks = [m for m in self.capture.messages("warning") if "alive after commit" in m]
            assert len(leaks) == 1 and f"storage of {scenario.hidden_holder} (" in leaks[0]
            self.capture.records.remove(("warning", leaks[0]))
        self.check_quiet()
        return dense

    @staticmethod
    def _byte_offset(entry: tuple) -> int:
        return entry[1] * entry[4].itemsize

    def check_quiet(self) -> None:
        """No dense module warned or reported an error."""
        noisy = self.capture.messages("warning", "error")
        assert noisy == [], noisy

    # ------------------------------------------------------------------
    # Forward checks
    # ------------------------------------------------------------------

    def check_forward(self, tokens: int, *, executor_primed: bool, stalls: str) -> None:
        """One forward, primed by the executor loop or only by H0, against its reference."""
        dense = self.manager.dense
        counters = self.counters(dense)
        self.log.clear()
        with self.stalls(stalls):
            if executor_primed:
                self.manager.prefetch_first_layers()
                assert dense.primed and not dense.forward_open
                assert self.copies() == self.edf(COVERED[:2])
                self.manager.prefetch_first_layers()
                assert self.copies() == self.edf(COVERED[:2]), "a primed runtime was primed again"
            output = self.forward(tokens)
        what = f"{tokens}-token {stalls} forward ({'executor' if executor_primed else 'H0'} prime)"
        assert_same(output, self.references[tokens], what)
        assert self.copies() == self.edf(), "the copy FIFO is not in deadline order"
        assert (dense.primed, dense.forward_open) == (False, False)
        assert self.counters(dense) == counters
        self.check_quiet()

    def check_back_to_back(self, tokens: tuple[int, ...], *, executor_primed: bool) -> None:
        """Forwards with no host synchronization between them, compute stalled.

        Each priming is enqueued while the previous forward's last layers still read their slots on
        the GPU: ``L(first)`` and ``L(second)`` refill the MoE-dense slots of the last two layers,
        which only those layers' H3 consume records protect.
        """
        dense = self.manager.dense
        counters = self.counters(dense)
        self.log.clear()
        outputs = []
        with self.stalls(SLOW_COMPUTE):
            for count in tokens:
                if executor_primed:
                    self.manager.prefetch_first_layers()
                outputs.append(self.forward(count, synchronize=False))
            self.execution_stream.synchronize()
        for index, (count, output) in enumerate(zip(tokens, outputs)):
            assert_same(output, self.references[count], f"back-to-back forward #{index}")
        assert self.copies() == self.edf() * len(tokens)
        assert (dense.primed, dense.forward_open) == (False, False)
        assert self.counters(dense) == counters
        self.check_quiet()

    def check_demand_fetch(self, tokens: int, *, stalls: str) -> None:
        """A forward around H0 (no dense priming): the first two layers fetch on demand, fenced.

        The experts are primed by hand, as expert DWDP alone leaves them.  At H1 of the first two
        covered layers the slots hold the previous forward's layers, so each runs one fenced
        demand fetch; the output stays exact (the demand fetch backstop).
        """
        dense = self.manager.dense
        fetches = dense.demand_fetches
        first, second = COVERED[:2]
        self.log.clear()
        with self.stalls(stalls):
            self.weight_manager.prefetch_layer(first)
            self.weight_manager.prefetch_layer(second)
            output = self.forward(tokens, skip_h0=True)
        assert_same(output, self.references[tokens], f"{stalls} forward around H0")
        assert dense.demand_fetches == fetches + 2 and dense.war_fences == 0
        demanded = [("E", first), ("E", second), ("L", first), ("L", 3), ("E", 3), ("L", second)]
        assert self.copies() == demanded + self.edf(COVERED[3:])
        assert (dense.primed, dense.forward_open) == (False, False)
        for message in self.capture.messages("warning"):
            assert "running a fenced demand fetch" in message, message
            self.capture.records.remove(("warning", message))
        self.check_quiet()

    @staticmethod
    def counters(dense: DenseDWDP) -> tuple[int, int]:
        """``(demand_fetches, war_fences)``; the normal path adds to neither."""
        return dense.demand_fetches, dense.war_fences

    def _arm(self, layer: int, fault: str) -> None:
        module = self.model.model.layers[layer]
        target = (
            module.attention if fault == "attention" else module.block_sparse_moe.routed_experts
        )
        target.fail_next = True

    def check_abort_recovery(
        self, layer: int, fault: str, *, stalls: str = SLOW_COMPUTE, tokens: int = 17
    ) -> None:
        """A fault after H1 of ``layer``; the next forward recovers in H0.

        ``fault`` is ``"attention"`` (both windows of the layer open) or ``"experts"`` (after
        ``wait_and_bind``, the expert slot and the MoE-dense window open).  The aborted forward
        stalls its compute, so the copies it enqueued run ahead of its reads.
        """
        dense = self.manager.dense
        counters = self.counters(dense)
        with self.stalls(SLOW_COMPUTE):
            self.manager.prefetch_first_layers()
            self._arm(layer, fault)
            with pytest.raises(InjectedFault):
                self.forward(tokens)
        gc.collect()
        assert dense.forward_open and not dense.primed
        self.log.clear()
        self.manager.prefetch_first_layers()
        assert self.log == [], "the executor loop must leave an aborted forward to the model"

        with self.stalls(stalls):
            output = self.forward(tokens)
        what = f"{stalls} forward after a fault in the {fault} of layer {layer}"
        assert_same(output, self.references[tokens], what)
        first, second = COVERED[:2]
        assert self.log[:2] == [("record_compute", first), ("record_compute", second)]
        assert self.copies() == self.edf()
        assert (dense.primed, dense.forward_open) == (False, False)
        assert self.counters(dense) == counters
        barriers = [m for m in self.capture.messages("info") if "WAR barrier" in m]
        assert len(barriers) == 1
        self.capture.records.remove(("info", barriers[0]))
        self.check_quiet()

    def check_abort_before_first_layer(self, tokens: int = 3) -> None:
        """A fault in replicated layer 0, after the priming: the slots stay primed and are used."""
        dense = self.manager.dense
        counters = self.counters(dense)
        self.log.clear()
        with self.stalls(SLOW_COPIES):
            self.manager.prefetch_first_layers()
            attention = self.model.model.layers[0].attention
            attention.fail_next = True
            with pytest.raises(InjectedFault):
                self.forward(tokens)
        gc.collect()
        assert dense.primed and not dense.forward_open
        self.log.clear()
        self.manager.prefetch_first_layers()
        assert self.log == []
        with self.stalls(SLOW_COPIES):
            output = self.forward(tokens)
        assert_same(output, self.references[tokens], "forward after a fault in layer 0")
        assert self.copies() == self.edf(COVERED[2:])
        assert self.counters(dense) == counters
        self.check_quiet()

    # ------------------------------------------------------------------
    # Teardown checks
    # ------------------------------------------------------------------

    def check_cleanup(self, *, exact_blocks: bool = True) -> None:
        """``DwdpManager.cleanup``: dense release before the expert teardown, then nothing left.

        Args:
            exact_blocks: The caching allocator must hand out exactly the blocks it did before
                setup, minus the moved originals.  False for a test that itself changed blocks
                outside dense DWDP (a CUDA graph capture makes PyTorch allocate its cuBLAS
                workspaces anew); then no block that dense setup added and no moved original may
                be left.
        """
        dense = self.manager.dense
        slot_refs = [
            weakref.ref(slot.storage.untyped_storage())
            for ring in dense.slots.values()
            for slot in ring
        ]
        assert slot_refs
        store = self.stores[-1]
        self.log.clear()
        self.manager.cleanup()
        assert self.log == [("wm.release", (True, {0})), ("group.Free",)]
        assert self.manager.dense is None
        assert dict(dense.slots) == {}
        del dense
        gc.collect()

        tensors = module_tensors(self.model.model)
        for path in self.moved:
            tensor = tensors[path]
            dtype = self.before.placements[path][4]
            assert tensor.numel() == 0 and tensor.dtype == dtype and tensor.device == self.device
        for path, tensor in tensors.items():
            if path not in self.moved and not is_expert_path(path):
                assert placement(tensor) == self.before.placements[path], path
        for path, parameter in self.before.parameters.items():
            assert tensors[path] is parameter
        del tensors
        assert [ref for ref in slot_refs if ref() is not None] == []
        with pytest.raises(RuntimeError, match="after release"):
            store.chunk_bases()
        assert self.model._dense_dwdp_active, "reloads must stay refused after the release"

        torch.cuda.synchronize(self.device)
        originals = self.originals()
        if not exact_blocks:
            added = set(self.blocks_after_setup) - set(self.blocks_before)
            assert (added | set(originals)) & set(active_blocks(self.device)) == set()
            self.check_quiet()
            return
        expected = {
            address: block
            for address, block in self.blocks_before.items()
            if address not in originals
        }
        assert active_blocks(self.device) == expected
        assert torch.cuda.memory_allocated(self.device) == self.allocated_before - sum(
            originals.values()
        )
        self.check_quiet()

    def check_cleanup_without_the_model(self) -> None:
        """``PyExecutor.shutdown`` drops the model before it exits the manager.

        The bindings went with the modules and count as detached: ``cleanup`` must neither raise
        nor log a warning or an error (the manager's included), and must still free the slots,
        the store and the launcher arguments.
        """
        dense = self.manager.dense
        slot_refs = [
            weakref.ref(slot.storage.untyped_storage())
            for ring in dense.slots.values()
            for slot in ring
        ]
        added = set(self.blocks_after_setup) - set(self.blocks_before)
        store = self.stores[-1]
        model_ref = weakref.ref(self.model)
        self.weight_manager.on_release = None
        self.before = None  # its Parameter objects view the slots
        del self.model
        gc.collect()
        assert model_ref() is None
        self.log.clear()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(dwdp_facade, "logger", self.capture)
            self.manager.cleanup()
        assert self.log == [("wm.release", None), ("group.Free",)]
        assert self.manager.dense is None and dict(dense.slots) == {}
        del dense
        gc.collect()
        assert [ref for ref in slot_refs if ref() is not None] == []
        with pytest.raises(RuntimeError, match="after release"):
            store.chunk_bases()
        torch.cuda.synchronize(self.device)
        assert added & set(active_blocks(self.device)) == set()
        self.check_quiet()

    def check_untouched(self) -> None:
        """Setup gave up: every module tensor is the original and every dense block is freed."""
        assert self.manager.dense is None
        assert not getattr(self.model, "_dense_dwdp_active", False)
        assert not getattr(self.model.model, "_dense_dwdp_active", False)
        tensors = module_tensors(self.model.model)
        for path, tensor in tensors.items():
            if not is_expert_path(path):
                assert placement(tensor) == self.before.placements[path], path
        for path, parameter in self.before.parameters.items():
            assert tensors[path] is parameter
        del tensors
        for store in self.stores:
            with pytest.raises(RuntimeError, match="after release"):
                store.chunk_bases()
        torch.cuda.synchronize(self.device)
        gc.collect()
        assert active_blocks(self.device) == self.blocks_before
        assert torch.cuda.memory_allocated(self.device) == self.allocated_before

    def check_replicated_forward(self, tokens: int, *, stalls: str = SLOW_COPIES) -> None:
        """Expert DWDP alone: the executor primes the experts, no hook runs, the output is exact."""
        self.log.clear()
        with self.stalls(stalls):
            self.manager.prefetch_first_layers()
            output = self.forward(tokens)
        assert_same(output, self.references[tokens], f"{tokens}-token forward without dense DWDP")
        assert self.copies() == [("E", layer) for layer in COVERED]


@pytest.fixture
def make_harness(monkeypatch):
    harnesses: list[Harness] = []

    def make(scenario: Scenario) -> Harness:
        harness = Harness(monkeypatch, scenario)
        harnesses.append(harness)
        return harness

    yield make
    STALLS.compute = STALLS.copies = 0
    for harness in harnesses:
        harness.close()
    harnesses.clear()
    set_global_dwdp_manager(None)
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SEGMENTS = [pytest.param("attn,moe", id="attn+moe"), pytest.param("attn", id="attn-canary")]
MODES = [pytest.param(mode, id=mode.value) for mode in CopyMode]
#: ``(N, rank, STORE_CHUNK_MB)``: the first and the last rank (the strict and debug runs use rank 5
#: of 16); one store chunk per layer, and a single chunk.
GROUPS = [
    pytest.param((4, 0, 1), id="N4-rank0-chunk1MB"),
    pytest.param((16, 15, 0), id="N16-rank15-onechunk"),
]


@pytest.mark.parametrize("segments", SEGMENTS)
@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("mode", MODES)
def test_dense_dwdp_end_to_end(make_harness, mode, group, segments):
    """Setup, exact forwards, a fault and its recovery, demand fetches and the teardown.

    Each transfer mode with the default launcher, a 4-rank group with one store chunk per layer
    and a 16-rank group with a single chunk, and both segment sets.  The forwards are primed by
    the executor loop or by H0 only, run one at a time or back to back, and stall compute or the
    copies.
    """
    num_ranks, rank, chunk_mb = group
    harness = make_harness(
        Scenario(mode=mode, num_ranks=num_ranks, rank=rank, segments=segments, chunk_mb=chunk_mb)
    )
    harness.setup()
    harness.check_committed()
    for tokens, executor_primed, stalls in FORWARDS:
        harness.check_forward(tokens, executor_primed=executor_primed, stalls=stalls)
    harness.check_back_to_back((3, 128, 1), executor_primed=True)
    harness.check_back_to_back((17, 3), executor_primed=False)
    harness.check_abort_recovery(MIDDLE_LAYER, "attention", stalls=SLOW_COPIES)
    harness.check_demand_fetch(17, stalls=SLOW_COPIES)
    harness.check_demand_fetch(3, stalls=SLOW_COMPUTE)
    harness.check_forward(TOKEN_COUNTS[2], executor_primed=True, stalls=SLOW_COMPUTE)
    harness.check_cleanup()


@pytest.mark.parametrize("segments", SEGMENTS)
def test_strict_runs_never_demand_fetch_or_fence(make_harness, segments):
    """``TRTLLM_DWDP_DENSE_STRICT=1`` turns every demand fetch and fence fallback into an error.

    The normal path, the recovery after faults in an MLA attention, in the experts of a middle
    layer and of the last layer, and a fault before the first covered layer all pass.
    """
    harness = make_harness(Scenario(mode=CopyMode.GATHER, segments=segments, strict=True))
    harness.setup()
    harness.check_committed()
    for tokens, executor_primed, stalls in FORWARDS:
        harness.check_forward(tokens, executor_primed=executor_primed, stalls=stalls)
    harness.check_back_to_back((128, 17), executor_primed=True)
    harness.check_back_to_back((1, 3), executor_primed=False)
    harness.check_abort_recovery(COVERED[0], "attention", stalls=SLOW_COMPUTE)
    harness.check_abort_recovery(3, "attention", stalls=SLOW_COPIES)
    harness.check_abort_recovery(MIDDLE_LAYER, "experts", stalls=SLOW_COMPUTE)
    harness.check_abort_recovery(COVERED[-1], "experts", stalls=SLOW_COPIES)
    harness.check_abort_before_first_layer()
    harness.check_forward(TOKEN_COUNTS[1], executor_primed=False, stalls=SLOW_COMPUTE)
    assert harness.counters(harness.manager.dense) == (0, 0)
    harness.check_cleanup()


@pytest.mark.parametrize("segments", SEGMENTS)
def test_debug_fills_the_slots_with_nan_and_outputs_stay_exact(make_harness, segments):
    """``TRTLLM_DWDP_DENSE_DEBUG=1``: NaN slots at commit; no read happens outside a window."""
    harness = make_harness(Scenario(mode=CopyMode.GATHER, segments=segments, debug=1))
    harness.setup()
    harness.check_committed()
    for tokens, executor_primed, stalls in FORWARDS:
        harness.check_forward(tokens, executor_primed=executor_primed, stalls=stalls)
    harness.check_back_to_back((17, 1), executor_primed=True)
    harness.check_abort_recovery(MIDDLE_LAYER, "experts", stalls=SLOW_COPIES)
    harness.check_forward(TOKEN_COUNTS[0], executor_primed=False, stalls=SLOW_COMPUTE)
    assert harness.counters(harness.manager.dense) == (0, 0)
    assert harness.manager.dense.watchdog_checks == 0, "DEBUG=1 runs no slot watchdog"
    harness.check_cleanup()


def test_h0_refuses_a_cuda_graph_capture_before_it_records_or_copies(make_harness):
    """A forward captured into a CUDA graph is refused at H0, before any event record or copy.

    The previous forward aborted after H1 of a middle layer, so H0 owes the WAR barrier and a new
    priming.  Under the capture H0 raises ``DenseProtocolError`` before it records the barrier
    or enqueues a dense or expert copy, so none of them goes into the graph (where it would not
    run, while the host state would count the slots as primed): the capture holds no dense
    work, the runtime still owes the barrier, and the next eager forward recovers through it and
    is exact.
    """
    harness = make_harness(Scenario(mode=CopyMode.GATHER))
    harness.setup()
    dense = harness.check_committed()
    tokens = TOKEN_COUNTS[0]
    harness.check_forward(tokens, executor_primed=True, stalls=SLOW_COPIES)
    with harness.stalls(SLOW_COMPUTE):
        harness.manager.prefetch_first_layers()
        harness._arm(MIDDLE_LAYER, "attention")
        with pytest.raises(InjectedFault):
            harness.forward(tokens)
    gc.collect()
    assert dense.forward_open and not dense.primed

    harness.log.clear()
    graph = torch.cuda.CUDAGraph()
    refused = r"ensure_primed \(H0\) while the current stream is capturing a CUDA graph"
    with pytest.raises(DenseProtocolError, match=refused):
        with torch.no_grad(), torch.cuda.graph(graph, stream=harness.execution_stream):
            harness.model(harness.inputs[tokens])
    del graph
    assert harness.log == [], "the refused H0 recorded or enqueued expert or dense work"
    assert dense.forward_open and not dense.primed
    assert harness.counters(dense) == (0, 0)

    with harness.stalls(SLOW_COPIES):
        output = harness.forward(tokens)
    assert_same(output, harness.references[tokens], "eager forward after a refused capture")
    first, second = COVERED[:2]
    assert harness.log[:2] == [("record_compute", first), ("record_compute", second)]
    assert harness.copies() == harness.edf()
    assert (dense.primed, dense.forward_open) == (False, False)
    assert harness.counters(dense) == (0, 0)
    barriers = [m for m in harness.capture.messages("info") if "WAR barrier" in m]
    assert len(barriers) == 1
    harness.capture.records.remove(("info", barriers[0]))
    harness.check_quiet()
    harness.check_forward(TOKEN_COUNTS[1], executor_primed=True, stalls=SLOW_COMPUTE)
    harness.check_cleanup(exact_blocks=False)


def test_a_second_setup_keeps_the_committed_runtime(make_harness):
    """``DwdpManager.setup`` again on the committed model returns at once and keeps the runtime.

    With ``REQUIRED=0``, a dense setup over the repointed modules would find nothing to move and
    give up, and the manager would drop the committed runtime, whose release would leave the
    modules zero-size.  The second call makes no expert or dense setup and no collective, and
    the forwards stay exact.
    """
    harness = make_harness(Scenario(mode=CopyMode.GATHER, required=False))
    harness.setup()
    dense = harness.check_committed()
    weight_manager = harness.weight_manager
    harness.log.clear()

    assert harness.manager.setup(harness.model) is weight_manager
    assert harness.manager.dense is dense and harness.weight_manager is weight_manager
    assert len(harness.group.gathered) == 5 and len(harness.stores) == 1
    assert harness.log == []
    for tokens, executor_primed, stalls in FORWARDS[:4]:
        harness.check_forward(tokens, executor_primed=executor_primed, stalls=stalls)
    harness.check_cleanup()


def test_device_checksum_matches_the_host_checksum():
    """The watchdog's device checksum is the P1 checksum, bit for bit.

    ``device_checksum`` keeps both sums on the device as the int64 values with the bits of
    ``tensor_checksum``'s unsigned sums: empty and partial words, misaligned storage offsets, two
    4M-word chunks, word origins far past ``2**32 / M``, sums that wrap, and other dtypes.  One
    flipped byte changes it.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(SEED)
    chunk_bytes = (1 << 22) * 4
    size = chunk_bytes + ROW_ALIGN + 7
    data = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=generator, device=device)
    floats = torch.randn(333, 7, generator=generator, device=device)
    # Every word is -1: both unsigned sums are at least 2**63, so their int64 bits are negative.
    ones = torch.full((ROW_ALIGN,), 0xFF, dtype=torch.uint8, device=device)
    assert max(checksum_bits(tensor_checksum(ones))) < 0
    cases = [
        (ones, 9),
        (data[:0], 0),
        (data[:1], 0),
        (data[:3], 5),
        (data[:4], 0),
        (data[: ROW_ALIGN + 1], 1023),
        (data[1 : ROW_ALIGN + 2], 0),
        (data, 0),
        (data[3 : chunk_bytes + 3], (1 << 30) + 3),
        (floats, 0),
        (floats.to(BF16), 17),
        (floats.to(torch.int32), 0),
    ]
    for tensor, word_offset in cases:
        expected = checksum_bits(tensor_checksum(tensor, word_offset=word_offset))
        actual = device_checksum(tensor, word_offset=word_offset)
        assert actual.device == device and actual.dtype == torch.int64
        what = f"{tensor.dtype} {tuple(tensor.shape)} at word {word_offset}"
        assert tuple(actual.tolist()) == expected, what
    flipped = data[: ROW_ALIGN + 1].clone()
    flipped[ROW_ALIGN // 2].bitwise_not_()
    assert not torch.equal(device_checksum(flipped), device_checksum(data[: ROW_ALIGN + 1]))
    with pytest.raises(ValueError, match="contiguous"):
        device_checksum(floats.t())


@pytest.mark.parametrize("segments", SEGMENTS)
def test_debug_2_watchdog_checks_every_slot_and_finds_nothing(make_harness, segments):
    """``TRTLLM_DWDP_DENSE_DEBUG=2``: every H1 checks its slots against the layer's P1 checksum.

    NaN slots at commit as with ``DEBUG=1``.  Every forward then checks each covered layer's slots
    once, whether the executor loop or H0 primed it, back to back, around H0 (demand fetches) or
    aborted after H1 of a middle layer, and no check finds other bytes.  The release logs the
    totals.
    """
    harness = make_harness(Scenario(mode=CopyMode.GATHER, segments=segments, debug=2))
    harness.setup()
    dense = harness.check_committed()
    kinds = harness.scenario.kinds
    per_forward = len(COVERED) * len(kinds)
    assert (dense.watchdog_checks, dense.watchdog_mismatches) == (0, 0)
    armed = [m for m in harness.capture.messages("info") if "slot watchdog on" in m]
    assert len(armed) == 1, armed

    forwards = 0
    for tokens, executor_primed, stalls in FORWARDS:
        harness.check_forward(tokens, executor_primed=executor_primed, stalls=stalls)
        forwards += 1
        assert dense.watchdog_checks == forwards * per_forward
    harness.check_back_to_back((17, 1, 128), executor_primed=True)
    harness.check_back_to_back((3, 17), executor_primed=False)
    harness.check_demand_fetch(3, stalls=SLOW_COPIES)
    harness.check_demand_fetch(17, stalls=SLOW_COMPUTE)
    forwards += 7
    assert dense.watchdog_checks == forwards * per_forward
    # The aborted forward checked the layers up to the faulty one, the recovery all of them.
    harness.check_abort_recovery(MIDDLE_LAYER, "attention", stalls=SLOW_COPIES)
    harness.check_abort_recovery(MIDDLE_LAYER, "experts", stalls=SLOW_COMPUTE)
    aborted = (COVERED.index(MIDDLE_LAYER) + 1) * len(kinds)
    checks = (forwards + 2) * per_forward + 2 * aborted
    assert (dense.watchdog_checks, dense.watchdog_mismatches) == (checks, 0)
    assert harness.counters(dense) == (4, 0)  # the demand fetches of the two forwards around H0

    harness.check_cleanup()
    summary = [m for m in harness.capture.messages("info") if "slot watchdog:" in m]
    assert summary == [f"[DWDP Dense] slot watchdog: {checks} slot checks, 0 mismatches"]
    assert (dense.watchdog_checks, dense.watchdog_mismatches) == (checks, 0)


@pytest.mark.parametrize("strict", [False, True], ids=["lenient", "strict"])
def test_debug_2_watchdog_names_a_row_changed_after_setup(make_harness, strict):
    """A store row changed after setup: every gather of its layer brings the other byte.

    The watchdog's check at that layer's H1 then fails each forward.  Lenient: H3 of the last
    layer logs one error per forward naming the layer and segment.  ``STRICT``: that H3 raises
    ``DenseProtocolError`` once the forward is closed, and the next forward primes and runs as
    usual.  With the row restored the forwards are exact and the watchdog is quiet.
    """
    harness = make_harness(Scenario(mode=CopyMode.GATHER, debug=2, strict=strict))
    harness.setup()
    dense = harness.check_committed()
    harness.check_forward(TOKEN_COUNTS[0], executor_primed=True, stalls=SLOW_COPIES)
    victims = ((MIDDLE_LAYER, MOE), (COVERED[-1], ATTN))
    for victim in victims:
        harness.flip_store_byte(*victim)
    pattern = (
        rf"slot watchdog: 2 slot checks at H1 found other bytes than their layer's checksum "
        rf"\(layers\.{MIDDLE_LAYER} moe x1, layers\.{COVERED[-1]} attn x1; (\d+) so far\)"
    )
    for forward in range(1, 3):
        harness.log.clear()
        harness.manager.prefetch_first_layers()
        if strict:
            with pytest.raises(DenseProtocolError, match=pattern):
                harness.forward(TOKEN_COUNTS[1])
            assert harness.capture.messages("error") == []
        else:
            harness.forward(TOKEN_COUNTS[1])
            errors = harness.capture.messages("error")
            assert len(errors) == 1 and re.search(pattern, errors[0]), errors
            assert re.search(pattern, errors[0])[1] == str(2 * forward)
            harness.capture.records.remove(("error", errors[0]))
        assert dense.watchdog_mismatches == 2 * forward
        assert harness.copies() == harness.edf(), "the forward did not run to its last layer"
        assert (dense.primed, dense.forward_open) == (False, False)
        harness.check_quiet()
    for victim in victims:
        harness.flip_store_byte(*victim)
    for tokens, executor_primed, stalls in FORWARDS[:4]:
        harness.check_forward(tokens, executor_primed=executor_primed, stalls=stalls)
    assert dense.watchdog_mismatches == 4
    assert harness.counters(dense) == (0, 0)
    harness.check_cleanup()


class _NoWaitEvent:
    """A ready event that streams cannot wait on: ``Stream.wait_event`` calls ``event.wait``."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self.event = event

    def record(self, stream: torch.cuda.Stream | None = None) -> None:
        self.event.record(stream)

    def wait(self, stream: torch.cuda.Stream | None = None) -> None:
        pass


def test_debug_2_watchdog_catches_reads_that_do_not_wait_for_their_copies(make_harness):
    """Without the RAW waits of H1 the watchdog sees slots whose copies have not landed.

    Compute's waits on the ready events do nothing, and the dense copies of one forward are held
    back until its compute has finished: the launcher only notes each layer, and once the forward
    is enqueued the copy stream waits for the execution stream and then runs the noted copies.
    Every H1 check of that forward therefore sees its slot as the previous forward left it,
    however the host and the GPU are timed: only the ``(layer, segment)`` pairs the previous
    forward left in their own slots (its last layers) match, and the error names the others in
    layer order.  With the waits and the copies restored the next forwards are exact and the
    watchdog is quiet again.
    """
    harness = make_harness(Scenario(mode=CopyMode.GATHER, debug=2))
    harness.setup()
    dense = harness.check_committed()
    kinds = harness.scenario.kinds
    harness.check_forward(TOKEN_COUNTS[0], executor_primed=True, stalls=SLOW_COPIES)
    # After a whole forward each slot holds the last covered layer that maps to it.
    holders = {(kind, dense.slot_index(kind, layer)): layer for layer in COVERED for kind in kinds}
    stale = [
        (layer, kind)
        for layer in COVERED
        for kind in kinds
        if holders[(kind, dense.slot_index(kind, layer))] != layer
    ]
    assert len(stale) > 8 and (COVERED[0], ATTN) in stale and (COVERED[-1], ATTN) not in stale

    ready, launcher = dense._ready, dense._launcher
    held: list[int] = []
    dense._ready = {
        kind: tuple(_NoWaitEvent(event) for event in events) for kind, events in ready.items()
    }
    dense._launcher = held.append
    try:
        harness.manager.prefetch_first_layers()
        harness.forward(TOKEN_COUNTS[2])
    finally:
        dense._ready, dense._launcher = ready, launcher
        finished = torch.cuda.Event()
        finished.record(harness.execution_stream)
        copy_stream = harness.weight_manager.copy_stream
        copy_stream.wait_event(finished)
        with torch.cuda.stream(copy_stream):
            for layer in held:
                launcher(layer)
        torch.cuda.synchronize(harness.device)
    del ready, launcher  # the launcher's argument tensors go with the runtime's release
    assert held == list(COVERED)
    assert dense.watchdog_mismatches == len(stale)
    errors = harness.capture.messages("error")
    assert len(errors) == 1, errors
    listed = ", ".join(f"layers.{layer} {kind.value} x1" for layer, kind in stale[:8])
    expected = (
        f"slot watchdog: {len(stale)} slot checks at H1 found other bytes than their layer's "
        f"checksum ({listed} and {len(stale) - 8} more; {len(stale)} so far)"
    )
    assert expected in errors[0], errors[0]
    harness.capture.records.remove(("error", errors[0]))
    for tokens, executor_primed, stalls in FORWARDS[:4]:
        harness.check_forward(tokens, executor_primed=executor_primed, stalls=stalls)
    assert dense.watchdog_mismatches == len(stale)
    harness.check_cleanup()


def test_phase_v_mismatch_with_required_off_keeps_the_replicated_weights(make_harness):
    """``TRTLLM_DWDP_DENSE_REQUIRED=0``: a P4 mismatch abandons dense DWDP on every rank.

    ``DwdpManager.setup`` returns the expert weight manager, ``dense`` stays None, everything dense
    setup allocated is freed and the model computes exactly with its original weights.
    """
    harness = make_harness(
        Scenario(mode=CopyMode.PER_SLICE, num_ranks=4, rank=1, required=False, corrupt_row=2)
    )
    harness.setup()
    assert len(harness.group.gathered) == 4
    assert not harness.group.gathered[-1].ok
    errors = harness.capture.messages("error")
    assert len(errors) == 1, errors
    assert "abandoned at collective #4 (P4 Phase V)" in errors[0]
    assert "TRTLLM_DWDP_DENSE_REQUIRED=0" in errors[0]
    assert "layers.1.linear_attn." in errors[0]
    harness.check_untouched()
    for tokens in TOKEN_COUNTS:
        harness.check_replicated_forward(tokens)
    harness.log.clear()
    harness.manager.cleanup()
    assert harness.log == [("wm.release", (True, set())), ("group.Free",)]
    torch.cuda.synchronize(harness.device)
    assert active_blocks(harness.device) == harness.blocks_before


def test_phase_v_mismatch_with_required_on_fails_the_setup(make_harness):
    """``TRTLLM_DWDP_DENSE_REQUIRED`` unset (on): the same mismatch fails ``DwdpManager.setup``.

    The model keeps its original weights, so the cleanup that follows (the executor creation
    rollback) tears expert DWDP down as usual.
    """
    harness = make_harness(Scenario(mode=CopyMode.PER_SLICE, num_ranks=4, rank=1, corrupt_row=3))
    with pytest.raises(DenseSetupError, match=r"collective #4 \(P4 Phase V\)"):
        harness.manager.setup(harness.model)
    harness.after_setup()
    assert len(harness.group.gathered) == 4
    harness.check_untouched()
    harness.check_replicated_forward(TOKEN_COUNTS[0])
    harness.log.clear()
    harness.manager.cleanup()
    assert harness.log == [("wm.release", (True, set())), ("group.Free",)]


def test_self_check_failure_rolls_the_modules_back(make_harness, monkeypatch):
    """A P5 self-check failure (``REQUIRED=0``): every rank rolls the repointed modules back.

    The modules expose their original tensors and Parameter objects again, the slots and the store
    are freed, and the model computes exactly with its replicated weights.
    """
    harness = make_harness(Scenario(mode=CopyMode.PER_SLICE, required=False))
    verify_layer = DenseBindings.verify_layer

    def failing_verify_layer(bindings: DenseBindings, layer: int) -> list[str]:
        found = verify_layer(bindings, layer)
        return found + ([f"layers.{layer}.injected"] if layer == MIDDLE_LAYER else [])

    monkeypatch.setattr(DenseBindings, "verify_layer", failing_verify_layer)
    harness.setup()
    assert len(harness.group.gathered) == 5
    assert not harness.group.gathered[-1].ok
    errors = harness.capture.messages("error")
    assert len(errors) == 1, errors
    assert "abandoned at collective #5 (P5 repoint self-check)" in errors[0]
    assert f"layers.{MIDDLE_LAYER}.injected" in errors[0]
    harness.check_untouched()
    for tokens in TOKEN_COUNTS:
        harness.check_replicated_forward(tokens)
    harness.log.clear()
    harness.manager.cleanup()
    assert harness.log == [("wm.release", (True, set())), ("group.Free",)]


def test_a_hidden_holder_is_reported_and_keeps_its_old_copy(make_harness):
    """A holder outside the bindings keeps one original alive.

    The commit line counts the leak and a warning names the anchor; that storage stays allocated
    (memory, not numerics), the model reads the slots and its outputs stay exact.
    """
    harness = make_harness(
        Scenario(
            mode=CopyMode.BATCHED,
            num_ranks=4,
            rank=3,
            hidden_holder=f"layers.{MIDDLE_LAYER}.linear_attn.f_b_proj.weight",
        )
    )
    harness.setup()
    harness.check_committed()
    for tokens, executor_primed, stalls in FORWARDS[:3]:
        harness.check_forward(tokens, executor_primed=executor_primed, stalls=stalls)
    harness.check_cleanup()
    assert harness.hidden.untyped_storage().data_ptr() in active_blocks(harness.device)


def test_a_hidden_holder_under_strict_fails_the_setup_after_releasing_it(make_harness):
    """``STRICT``: the leak raises past the commit point, on this rank only.

    The bindings are committed, so this rank's modules are left with zero-size tensors and it
    cannot serve; its slots, store and launcher are freed first, and only the held original stays.
    """
    hidden = f"layers.{MIDDLE_LAYER}.linear_attn.f_b_proj.weight"
    harness = make_harness(
        Scenario(mode=CopyMode.PER_SLICE, num_ranks=4, rank=3, strict=True, hidden_holder=hidden)
    )
    with pytest.raises(DenseRepointError, match="still alive after commit"):
        harness.manager.setup(harness.model)
    harness.after_setup()
    assert harness.manager.dense is None
    assert len(harness.group.gathered) == 5
    assert harness.model._dense_dwdp_active and harness.model.model._dense_dwdp_active
    harness.moved = harness.expected_moved()
    tensors = module_tensors(harness.model.model)
    assert {tensors[path].numel() for path in harness.moved} == {0}
    del tensors
    originals = harness.originals()
    assert harness.hidden.untyped_storage().data_ptr() in harness.blocks_before
    expected = {
        address: block
        for address, block in harness.blocks_before.items()
        if address not in originals
    }
    assert active_blocks(harness.device) == expected
    with pytest.raises(RuntimeError, match="after release"):
        harness.stores[-1].chunk_bases()
    harness.log.clear()
    harness.manager.cleanup()
    assert harness.log == [("wm.release", (True, {0})), ("group.Free",)]


def test_cleanup_after_the_model_was_collected(make_harness):
    """The executor shutdown order: the model is gone before ``DwdpManager.cleanup`` runs."""
    harness = make_harness(Scenario(mode=CopyMode.PER_SLICE, num_ranks=4, rank=2))
    harness.setup()
    harness.check_committed()
    harness.check_forward(TOKEN_COUNTS[0], executor_primed=True, stalls=SLOW_COMPUTE)
    harness.check_cleanup_without_the_model()


@pytest.mark.parametrize("segments", SEGMENTS)
def test_every_storage_under_the_roots_moves_at_the_lowest_threshold(make_harness, segments):
    """``TRTLLM_DWDP_DENSE_MIN_BYTES=1``: scales, norms, decode constants and small tuple items move.

    Blobs of storages that are not whole pages, and every read of the toy layers under the window
    contract; the expected sizes come from the collector.
    """
    harness = make_harness(Scenario(mode=CopyMode.GATHER, segments=segments, min_bytes=1))
    harness.setup()
    harness.check_committed()
    for tokens, executor_primed, stalls in FORWARDS:
        harness.check_forward(tokens, executor_primed=executor_primed, stalls=stalls)
    harness.check_back_to_back((3, 17), executor_primed=True)
    harness.check_abort_recovery(MIDDLE_LAYER, "attention", stalls=SLOW_COPIES)
    harness.check_cleanup()


def test_dense_off_makes_one_collective_and_changes_nothing(make_harness):
    """``TRTLLM_DWDP_DENSE`` unset on a model with roots: collective #1 only, then expert DWDP alone.

    ``DwdpManager.setup`` still runs the dense setup (so ranks that disagree would all fail at
    collective #1); with dense DWDP off everywhere nothing is collected, allocated or repointed.
    """
    harness = make_harness(Scenario(mode=CopyMode.GATHER, enabled=False))
    harness.setup()
    assert len(harness.group.gathered) == 1
    assert harness.stores == [] and harness.setup_copies == []
    harness.check_untouched()
    for tokens in TOKEN_COUNTS:
        harness.check_replicated_forward(tokens)
    harness.check_quiet()
    harness.log.clear()
    harness.manager.cleanup()
    assert harness.log == [("wm.release", (True, set())), ("group.Free",)]
