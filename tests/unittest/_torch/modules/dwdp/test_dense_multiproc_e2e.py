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
"""Multi-process GPU tests of dense DWDP end to end.

Every DWDP rank is its own process with one GPU visible, like an independently launched context
worker.  The test re-runs this file as the worker (``--spec``, ``--rank``, ``--out``); each worker
writes a JSON report that the parent turns into a pass, a failure listing every finding, or a skip
(all ranks skip together without FABRIC handles or ``trtllm::dwdp_fabric_gather``).  Rank 0 hosts
two ``torch.distributed.TCPStore`` instances on 127.0.0.1: the DWDP group, which the real
``DwdpManager`` builds with ``StoreDwdpGroup.from_env`` from the ``TRTLLM_DWDP_*`` environment,
and a test-owned ``StoreDwdpGroup`` for the test's own barriers, so the test adds no collective to
the DWDP group.  Every collective waits at most ``_COLLECTIVE_TIMEOUT_S``; the parent kills a group
that outlives ``_GROUP_DEADLINE_S`` after dumping the workers' Python stacks into their logs.

The model.  Every rank builds the same seeded toy decoder (``ToyCausalLM``): layer 0 stays
replicated and layers 1-7 are covered.  An attention root holds a fused projection buffer with
row-view buffer aliases, an allowed plain attribute with ``nn.Parameter`` row views and pad rows,
an allowed tuple attribute, an excluded alias of an anchor, an excluded scratch tensor, an
unlisted large attribute and a small same-storage alias; the MoE-dense roots hold an fp32 router,
a shared MLP and the latent projections.  Blobs are about 40 MB (attention) and 18 MB (MoE-dense)
per layer, and a forward carries at most 256 tokens.  The forward places the hooks as the Kimi-K3
model does: H0 ``DwdpManager.ensure_primed`` at the decoder's entry, H1 ``wait_layer`` before a
covered layer, H2 ``attention_done`` after its attention, H3 ``layer_done`` after its MoE block,
whose experts go through ``DwdpManager.wait_and_bind`` and
``DwdpManager.record_compute_and_prefetch_next`` as in ``ConfigurableMoE``.

Setup.  ``DwdpManager.setup(model)`` runs the real ``setup_dense_dwdp`` with nothing injected:
``DenseConfig`` from ``TRTLLM_DWDP_DENSE*`` (``DEBUG=1``: the slots hold NaN after the commit),
``FabricShardStore``, ``TorchCudaOps`` and ``DenseCopyLauncher`` in gather mode with the
production TMA kernel.  Only the expert side is a stand-in: the toy has no routed-expert weights to
shard, so a patched ``setup_dwdp`` returns ``_ToyExpertManager``, which implements the
``DWDPWeightManager`` calls with the expert protocol (two slots by MoE position, prefetch and
consume events, both consume events recorded at construction) on its own copy stream, the stream
the dense gathers share.  The test observes setup without changing it: it times each collective
of the DWDP group, keeps the layout, slots and copy plans setup builds, and logs every launch on
the copy stream.  After the commit every moved reference must view its slot (``nn.Parameter``
objects and buffer registrations kept), everything else must stay where it was, and the original
storages must be gone.

Protocol stress (``test_protocol_stress_end_to_end``, ``test_attention_only_end_to_end``).  Each
rank computes the replicated reference of 50 forwards before setup, with the hooks and the expert
stand-in out of the way.  After the commit the same forwards run with random token counts and no
host synchronization between them, so the host runs ahead of the GPU and every wait binds to
host-order records.  Copies are slow: the dense gather runs two blocks (one per peer row with four
ranks) and each expert prefetch spins 0.3-1.5 ms before its copy; compute is light except for spins
of up to 2.5 ms at random inside the attention and MoE-dense read windows, just before their last
weight reads, so a copy that overwrote a slot inside its read window would change the output.
Forwards are primed by ``prefetch_first_layers`` on the default stream (the executor loop) or only
by H0 on the execution stream; the injected anomalies are an exception after H1 of a random layer
(after the first attention read, after H2, or after the expert wait), an exception in replicated
layer 0 after priming, and a forward that skips H0 while only the experts are primed (a path that
bypasses the model's forward).  The reads an aborted forward leaves behind run after a 3 ms spin,
so the next forward's priming is enqueued before they run.  For every forward the copy-stream launches
must be exactly the deadline order the protocol prescribes (``L(1) E(1) L(2) E(2) L(3) E(3) ...``,
the recovery, or the two demand fetches) and the demand-fetch, WAR-fence and WAR-barrier counts
must change exactly as the anomaly implies (no change for a clean forward); every completed output
must equal the reference bit for bit, and so must the probes, sums over each layer's first
attention read and its MoE-block reads that every forward writes, an aborted one included.

Setup faults (``test_setup_fault_reaches_one_verdict``).  One rank's ``FabricShardStore.allocate``
fails in P2, or one rank's Phase V sees a corrupted row (the gather of one layer in P4 is followed
by flipping bytes of a peer's row in that rank's slot).  Every rank must reach the same verdict,
with the same reason, at the same collective without hanging: ``TRTLLM_DWDP_DENSE_REQUIRED=0``
returns None everywhere and the model then computes its references exactly, ``=1`` raises
``DenseSetupError`` everywhere.

Teardown.  ``DwdpManager.cleanup`` releases dense DWDP on every rank; then every moved reference
holds a zero-size tensor, a second release is a no-op, and, after a barrier (a rank's store is
freed only once every peer released its import of it), the memory must be back at its pre-setup
level counting what the commit dropped: the device memory in use within ``_MEMORY_TOLERANCE`` of
its pre-setup level less the allocator segments that held only original weights, and the caching
allocator exactly at its pre-setup bytes less the originals' blocks.  Device memory in use is what
``mem_get_info`` reports less what NVML charges to other processes, so other processes on a shared
GPU do not move it while memory charged to nobody, such as a store chunk a peer still imports,
counts.  The commit itself must free exactly the originals' blocks less the slots and gather
arguments.  To make these
numbers exact the workers run the default caching allocator (``PYTORCH_CUDA_ALLOC_CONF`` unset),
allocate their long-lived buffers before the model, and allocate the tensors that stay replicated
before the ones that move, sealing the segments in between.

The parent prints per-rank measurements: setup time per phase and per collective, the bytes moved,
stored and staged, the gather bytes per forward, the memory freed at commit, and the loop's
counters, copies and timing.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import faulthandler
import gc
import json
import math
import os
import random
import re
import signal
import site
import socket
import subprocess
import sys
import time
import traceback
import weakref
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import tensorrt_llm
from tensorrt_llm._torch.modules.dwdp import fabric_gather, transport, vmm
from tensorrt_llm._torch.modules.dwdp.dense import setup as dense_setup
from tensorrt_llm._torch.modules.dwdp.dense.config import (
    DENSE_DEBUG_ENV,
    DENSE_ENV,
    DENSE_REQUIRED_ENV,
    DENSE_SEGMENTS_ENV,
)
from tensorrt_llm._torch.modules.dwdp.dense.layout import tensor_checksum
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseSetupError, SegmentKind
from tensorrt_llm._torch.modules.dwdp.rendezvous import (
    DWDP_MASTER_ADDR_ENV,
    DWDP_MASTER_PORT_ENV,
    DWDP_RANK_ENV,
    DWDP_TIMEOUT_ENV,
    StoreDwdpGroup,
)

_MASTER_ADDR = "127.0.0.1"
_COLLECTIVE_TIMEOUT_S = 120  # TRTLLM_DWDP_RENDEZVOUS_TIMEOUT_S and every test-group wait
_GROUP_DEADLINE_S = 600.0  # importing TensorRT-LLM plus the whole scenario, per group
_MEMORY_TOLERANCE = 4 << 20
_ALLOCATED_TOLERANCE = 64 << 10
_MEMORY_POLLS = 40
_MEMORY_POLL_S = 0.25

FABRIC = vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
BF16 = torch.bfloat16
F32 = torch.float32
U8 = torch.uint8
ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE

# ---------------------------------------------------------------------------------------------
# Toy model geometry (identical on every rank)
# ---------------------------------------------------------------------------------------------

HIDDEN = 2048
#: Rows of the ``_bfa_proj_weight`` attribute: ``f_a_proj`` rows, ``b_proj`` rows, pad rows that
#: no view covers.
F_A_ROWS = 320
B_ROWS = 64
PAD_ROWS = 16
CONV_ROWS = 384
O_DENSE_ROWS = 8
SCRATCH_ROWS = 320
ROUTER_EXPERTS = 160
SHARED_INTER = 1024
LATENT = 512
MLP0_INTER = 512
#: The routed experts of one layer, as the expert stand-in copies them: ``[EXPERT_ROWS, LATENT]``
#: bf16 windows of one table, ``EXPERT_STRIDE`` bytes apart from layer to layer.
EXPERT_ROWS = 2048
EXPERT_PROBE_ROWS = 64
EXPERT_BYTES = EXPERT_ROWS * LATENT * 2
EXPERT_STRIDE = 64 << 10
NUM_LAYERS = 8
COVERED = tuple(range(1, NUM_LAYERS))
MODEL_SEED = 20261016
MAX_TOKENS = 256
STRESS_ITERATIONS = 50
FAULT_ITERATIONS = 6
#: Spin-kernel cycles per microsecond (an SM clock estimate); only the order of magnitude matters.
CYCLES_PER_US = 2400
#: ``TRTLLM_DWDP_GATHER_BLOCKS``: two blocks for the single gather row of two ranks (one cluster
#: of two, as production), one block per row with four ranks; the dense gathers are slow.
GATHER_BLOCKS = 2
#: Spin before the reads an injected abort leaves behind (the aborted layer's first attention
#: read, or its MoE block's reads), so they run after the next forward's priming is enqueued.
ABORT_SPIN_US = 3000
EXPERT_SPIN_US = (300, 1500)
WINDOW_SPIN_US = (50, 2500)
WINDOW_SPIN_PROBABILITY = 0.4

#: The model tables, carried by the toy decoder layers.
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
MOE_ROOTS = (
    "gate",
    "shared_experts",
    "routed_expert_down_proj",
    "routed_expert_up_proj",
    "routed_expert_norm",
)

#: Paths, relative to a covered decoder layer, of every reference that must move, by segment.
MOVED = {
    ATTN: (
        "attn.qkv_proj.weight",
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "attn._bfa_proj_weight",
        "attn.f_a_proj.weight",
        "attn.b_proj.weight",
        "attn.f_b_proj.weight",
        "attn.o_proj.weight",
        "attn._o_dense",
        "attn._mtp_conv_weights[0]",
        "attn._mtp_conv_weights[1]",
    ),
    MOE: (
        "moe.gate.weight",
        "moe.shared_experts.gate_up_proj",
        "moe.shared_experts.down_proj",
        "moe.routed_expert_down_proj.weight",
        "moe.routed_expert_up_proj.weight",
    ),
}
#: Paths, relative to a covered decoder layer, of references that always stay in place.
STAYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "attn.A_log",
    "attn._A_log_f32",
    "attn._fused_q_placeholder",
    "attn._unlisted_cache",
    "moe.gate.e_score_correction_bias",
    "moe.routed_expert_norm.weight",
)
#: Moved storages per covered layer and segment (anchors).
ANCHORS = {ATTN: 6, MOE: 5}

# What a worker must not inherit: launcher identities (importing TensorRT-LLM initializes MPI and
# each worker is a standalone MPI singleton, like a plain trtllm-serve process), the DWDP knobs,
# which the test sets for the workers itself, and allocator settings, which would change how the
# caching allocator returns memory.
_DROPPED_ENV_PREFIXES = (
    "SLURM_",
    "PMIX_",
    "PMI_",
    "OMPI_",
    "PRTE_",
    "I_MPI_",
    "HYDRA_",
    "MPI_",
    "TRTLLM_DWDP_",
)
_DROPPED_ENV_VARS = ("TLLM_DISABLE_MPI", "TLLM_LOG_LEVEL_BY_MODULE", "PYTORCH_CUDA_ALLOC_CONF")
_GATHER_ENABLE_ENV = "TRTLLM_DWDP_PREFETCH_GATHER"
_GATHER_KERNEL_ENV = "TRTLLM_DWDP_GATHER_KERNEL"
_GATHER_BLOCKS_ENV = "TRTLLM_DWDP_GATHER_BLOCKS"
_GATHER_LIB_ENV = "TRTLLM_DWDP_GATHER_LIB"

# ---------------------------------------------------------------------------------------------
# Toy decoder
# ---------------------------------------------------------------------------------------------


class _InjectedAbort(Exception):
    """The exception a forward raises where the iteration plan injects an abort."""


@dataclass
class _ForwardPlan:
    """Spin kernels, the abort point and the probe row of one forward.

    Attributes:
        attn_spin: Layer -> spin cycles before the attention's last weight read.
        moe_spin: Layer -> spin cycles before the MoE block's last dense weight read.
        abort: ``(layer, point)`` where the forward raises, if any.
        probes: ``[NUM_LAYERS, 2]`` fp32 row this forward writes: per covered layer, a sum over
            the attention's first read (0) and over the MoE block's router, latent and expert
            reads (1).  An aborted forward writes the entries of the reads it enqueued.
    """

    attn_spin: dict[int, int] = field(default_factory=dict)
    moe_spin: dict[int, int] = field(default_factory=dict)
    abort: tuple[int, str] | None = None
    probes: torch.Tensor | None = None

    def check(self, layer: int, point: str) -> None:
        if self.abort == (layer, point):
            raise _InjectedAbort(f"injected abort at layer {layer} ({point})")

    def before_reads(self, layer: int, point: str) -> None:
        """Delay the reads an abort at ``(layer, point)`` leaves behind, as a slow kernel would."""
        if self.abort == (layer, point):
            torch.cuda._sleep(ABORT_SPIN_US * CYCLES_PER_US)

    def record(self, layer: int, index: int, *tensors: torch.Tensor) -> None:
        if self.probes is not None:
            self.probes[layer, index] = sum(tensor.float().sum() for tensor in tensors)

    @staticmethod
    def spin(cycles: int | None) -> None:
        if cycles:
            torch.cuda._sleep(cycles)


#: ``(key, shape, dtype, init, segment)`` of a covered decoder layer's base tensors; aliases are
#: views the modules make of them.  ``segment`` None: the tensor stays replicated.
_COVERED_TENSORS = (
    ("input_norm", (HIDDEN,), BF16, "norm", None),
    ("post_norm", (HIDDEN,), BF16, "norm", None),
    ("attn.qkv", (3 * HIDDEN, HIDDEN), BF16, "weight", ATTN),
    ("attn.bfa", (F_A_ROWS + B_ROWS + PAD_ROWS, HIDDEN), BF16, "weight", ATTN),
    ("attn.f_b", (HIDDEN, F_A_ROWS), BF16, "weight", ATTN),
    ("attn.o_proj", (HIDDEN, HIDDEN), BF16, "weight", ATTN),
    ("attn.mtp0", (CONV_ROWS, HIDDEN), BF16, "weight", ATTN),
    ("attn.mtp1", (CONV_ROWS, HIDDEN), BF16, "weight", ATTN),
    ("attn.A_log", (HIDDEN,), F32, "small", None),
    ("attn.scratch", (SCRATCH_ROWS, HIDDEN), BF16, "zeros", None),
    ("attn.unlisted", (SCRATCH_ROWS, HIDDEN), BF16, "zeros", None),
    ("moe.gate", (ROUTER_EXPERTS, HIDDEN), F32, "weight", MOE),
    ("moe.bias", (ROUTER_EXPERTS,), F32, "tiny", None),
    ("moe.gate_up", (2 * SHARED_INTER, HIDDEN), BF16, "weight", MOE),
    ("moe.down", (HIDDEN, SHARED_INTER), BF16, "weight", MOE),
    ("moe.latent_down", (LATENT, HIDDEN), BF16, "weight", MOE),
    ("moe.latent_up", (HIDDEN, LATENT), BF16, "weight", MOE),
    ("moe.latent_norm", (LATENT,), BF16, "norm", None),
)
#: The same for replicated decoder layer 0.
_DENSE_TENSORS = (
    ("input_norm", (HIDDEN,), BF16, "norm", None),
    ("mlp.gate_up", (2 * MLP0_INTER, HIDDEN), BF16, "weight", None),
    ("mlp.down", (HIDDEN, MLP0_INTER), BF16, "weight", None),
)


def _init_tensor(shape, dtype, init: str, seed: int, device: torch.device) -> torch.Tensor:
    """One seeded tensor, allocated once: no temporaries share its allocator segment."""
    if init == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    tensor = torch.randn(shape, generator=gen, device=device, dtype=dtype)
    if init == "weight":
        return tensor.mul_(1.0 / math.sqrt(shape[-1]))
    if init == "norm":
        return tensor.mul_(0.1).add_(1.0)
    return tensor.mul_(0.1 if init == "small" else 0.01)


def _seal_allocator(device: torch.device, streams: list[torch.cuda.Stream]) -> list[torch.Tensor]:
    """Fill every free large-pool block, so the next allocations start segments of their own.

    Returns:
        The filler tensors; they must live as long as the tensors allocated after them.
    """
    by_id = {stream.cuda_stream: stream for stream in streams}
    fillers = []
    for segment in torch.cuda.memory_snapshot():
        stream = by_id.get(segment["stream"])
        if segment["device"] != device.index or segment["segment_type"] != "large" or not stream:
            continue
        for block in segment["blocks"]:
            if block["state"] == "inactive" and block["size"] > (1 << 20):
                with torch.cuda.stream(stream):
                    fillers.append(torch.empty(block["size"], dtype=U8, device=device))
    return fillers


def _toy_weights(
    device: torch.device, segments: tuple[SegmentKind, ...], streams: list[torch.cuda.Stream]
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Every base tensor of the toy decoder, identical on every rank, plus allocator fillers.

    Values depend only on each tensor's key.  The tensors that stay replicated are allocated
    first and their segments sealed, then the tensors that move: so the segments the moved
    tensors occupy hold nothing else, and the commit frees them whole.
    """
    entries = [
        (f"layers.{index}.{key}", shape, dtype, init, segment)
        for index in range(NUM_LAYERS)
        for key, shape, dtype, init, segment in (_COVERED_TENSORS if index else _DENSE_TENSORS)
    ]
    entries.append(("norm", (HIDDEN,), BF16, "norm", None))
    weights: dict[str, torch.Tensor] = {}
    fillers: list[torch.Tensor] = []
    for moving in (False, True):
        for seed, (key, shape, dtype, init, segment) in enumerate(entries):
            if (segment in segments) == moving:
                weights[key] = _init_tensor(shape, dtype, init, MODEL_SEED + seed, device)
        if not moving:
            torch.cuda.empty_cache()
            fillers += _seal_allocator(device, streams)
    return weights, fillers


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (y * weight.float()).to(x.dtype)


#: Looks up a decoder layer's base tensor by its key in ``_COVERED_TENSORS``.
_Lookup = Callable[[str], torch.Tensor]


class _Weight(nn.Module):
    """One tensor named ``weight``: an ``nn.Parameter`` or a registered buffer."""

    def __init__(self, tensor: torch.Tensor, *, buffer: bool = False) -> None:
        super().__init__()
        if buffer:
            self.register_buffer("weight", tensor)
        else:
            self.weight = nn.Parameter(tensor, requires_grad=False)


class ToyAttention(nn.Module):
    """Attention root with the reference kinds of a Kimi-K3 KDA mixer."""

    def __init__(self, w: _Lookup) -> None:
        super().__init__()
        fused = w("attn.qkv")
        self.qkv_proj = _Weight(fused, buffer=True)
        self.q_proj = _Weight(fused[:HIDDEN], buffer=True)
        self.k_proj = _Weight(fused[HIDDEN : 2 * HIDDEN], buffer=True)
        self.v_proj = _Weight(fused[2 * HIDDEN :], buffer=True)
        bfa = w("attn.bfa")
        self._bfa_proj_weight = bfa
        self.f_a_proj = _Weight(bfa[:F_A_ROWS])
        self.b_proj = _Weight(bfa[F_A_ROWS : F_A_ROWS + B_ROWS])
        self.f_b_proj = _Weight(w("attn.f_b"))
        self.o_proj = _Weight(w("attn.o_proj"))
        self._o_dense = self.o_proj.weight.data[:O_DENSE_ROWS]
        self._mtp_conv_weights = (w("attn.mtp0"), w("attn.mtp1"))
        self.A_log = nn.Parameter(w("attn.A_log"), requires_grad=False)
        self._A_log_f32 = self.A_log.data
        self._fused_q_placeholder = w("attn.scratch")
        self._unlisted_cache = w("attn.unlisted")

    def forward(self, x: torch.Tensor, layer: int, plan: _ForwardPlan) -> torch.Tensor:
        plan.before_reads(layer, "after_h1")
        q = F.linear(x, self.q_proj.weight)
        plan.record(layer, 0, q)
        plan.check(layer, "after_h1")
        k = F.linear(x, self.k_proj.weight)
        v = F.linear(x, self.v_proj.weight)
        fused = F.linear(x, self.qkv_proj.weight)
        f_a = F.linear(x, self.f_a_proj.weight)
        beta = torch.sigmoid(F.linear(x, self.b_proj.weight)).mean(-1, keepdim=True)
        gate = torch.sigmoid(F.linear(f_a, self.f_b_proj.weight))
        conv = sum(F.linear(x, w).mean(-1, keepdim=True) for w in self._mtp_conv_weights)
        decay = torch.exp(-torch.exp(self._A_log_f32)).to(x.dtype)
        mixed = (q * torch.sigmoid(k) + v) * gate + fused[:, :HIDDEN] * decay
        mixed = mixed + beta * fused[:, HIDDEN : 2 * HIDDEN] + conv * fused[:, 2 * HIDDEN :]
        mixed = mixed + F.linear(x, self._o_dense).mean(-1, keepdim=True)
        plan.spin(plan.attn_spin.get(layer))
        return F.linear(mixed, self.o_proj.weight)


class ToyRouter(nn.Module):
    """The fp32 router with the routing bias the model excludes."""

    def __init__(self, w: _Lookup) -> None:
        super().__init__()
        self.weight = nn.Parameter(w("moe.gate"), requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(w("moe.bias"), requires_grad=False)


class ToySharedMLP(nn.Module):
    """Shared expert: a buffer ``gate_up_proj`` and a Parameter ``down_proj``."""

    def __init__(self, w: _Lookup) -> None:
        super().__init__()
        self.register_buffer("gate_up_proj", w("moe.gate_up"))
        self.down_proj = nn.Parameter(w("moe.down"), requires_grad=False)


class ToyRoutedExperts(nn.Module):
    """The routed-expert backend expert DWDP binds; never a dense root.

    Attributes:
        replicated: The layer's experts as a replicated model holds them (the reference).
        bound: What ``wait_and_bind`` bound for the current forward.
    """

    def __init__(self, replicated: torch.Tensor) -> None:
        super().__init__()
        self.replicated = replicated
        self.bound: torch.Tensor | None = None


class ToyMoE(nn.Module):
    """MoE block: dense roots (router, shared expert, latent projections) plus routed experts."""

    def __init__(
        self, layer_idx: int, w: _Lookup, manager: Any, replicated_experts: torch.Tensor
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = ToyRouter(w)
        self.shared_experts = ToySharedMLP(w)
        self.routed_expert_down_proj = _Weight(w("moe.latent_down"))
        self.routed_expert_up_proj = _Weight(w("moe.latent_up"))
        self.routed_expert_norm = _Weight(w("moe.latent_norm"))
        self.routed_experts = ToyRoutedExperts(replicated_experts)
        # Registered with the DWDP manager at construction, as ConfigurableMoE registers.
        self._dwdp = manager
        if manager is not None:
            manager.add_layer(layer_idx)

    def forward(self, x: torch.Tensor, plan: _ForwardPlan) -> torch.Tensor:
        layer = self.layer_idx
        plan.before_reads(layer, "in_moe")
        logits = F.linear(x.float(), self.gate.weight) + self.gate.e_score_correction_bias
        top = torch.softmax(logits, dim=-1).amax(dim=-1, keepdim=True).to(x.dtype)
        latent = _rms_norm(
            F.linear(x, self.routed_expert_down_proj.weight), self.routed_expert_norm.weight
        )
        dwdp = self._dwdp
        if dwdp is not None:
            dwdp.wait_and_bind(self.routed_experts, layer)
            experts = self.routed_experts.bound
        else:
            experts = self.routed_experts.replicated
        head = F.linear(latent, experts[:EXPERT_PROBE_ROWS])
        tail = F.linear(latent, experts[-EXPERT_PROBE_ROWS:])
        gain = torch.sigmoid((head * tail).mean(-1, keepdim=True))
        plan.record(layer, 1, top, latent, gain)
        plan.check(layer, "in_moe")
        if dwdp is not None:
            dwdp.record_compute_and_prefetch_next(layer)
        up = F.linear(latent * gain, self.routed_expert_up_proj.weight)
        plan.spin(plan.moe_spin.get(layer))
        gate_up = F.linear(x, self.shared_experts.gate_up_proj)
        g, u = gate_up.chunk(2, dim=-1)
        shared = F.linear(F.silu(g) * u, self.shared_experts.down_proj)
        return up * top + shared


class ToyDenseMLP(nn.Module):
    """The replicated MLP of decoder layer 0."""

    def __init__(self, w: _Lookup) -> None:
        super().__init__()
        self.gate_up_proj = _Weight(w("mlp.gate_up"))
        self.down_proj = _Weight(w("mlp.down"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = F.linear(x, self.gate_up_proj.weight).chunk(2, dim=-1)
        return F.linear(F.silu(g) * u, self.down_proj.weight)


class ToyDecoderLayer(nn.Module):
    """A decoder layer with the dense DWDP seam and hooks placed as in the Kimi-K3 layer."""

    dense_dwdp_attr_allow = ATTR_ALLOW
    dense_dwdp_exclude = EXCLUDE

    def __init__(
        self, layer_idx: int, w: _Lookup, manager: Any, replicated_experts: torch.Tensor | None
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = replicated_experts is not None
        self.input_layernorm = _Weight(w("input_norm"))
        if self.is_moe:
            self.post_attention_layernorm = _Weight(w("post_norm"))
            self.attn = ToyAttention(w)
            self.moe = ToyMoE(layer_idx, w, manager, replicated_experts)
        else:
            self.mlp = ToyDenseMLP(w)
        self._dwdp = manager

    def dense_dwdp_roots(self, segment: str) -> list[tuple[str, nn.Module]]:
        if segment not in ("attn", "moe"):
            raise ValueError(f"unknown dense DWDP segment {segment!r}")
        if not self.is_moe:
            return []
        if segment == "attn":
            return [("attn", self.attn)]
        return [(f"moe.{name}", getattr(self.moe, name)) for name in MOE_ROOTS]

    def forward(self, hidden: torch.Tensor, plan: _ForwardPlan) -> torch.Tensor:
        dwdp = self._dwdp
        dense = dwdp.dense if dwdp is not None else None
        if dense is not None and not dense.covers(self.layer_idx):
            dense = None
        if dense is not None:
            dense.wait_layer(self.layer_idx)  # H1
        x = _rms_norm(hidden, self.input_layernorm.weight)
        if not self.is_moe:
            out = self.mlp(x)
            plan.check(self.layer_idx, "layer0")
            return hidden + out
        attended = self.attn(x, self.layer_idx, plan)
        if dense is not None:
            dense.attention_done(self.layer_idx)  # H2
        plan.check(self.layer_idx, "after_h2")
        hidden = hidden + attended
        mixed = self.moe(_rms_norm(hidden, self.post_attention_layernorm.weight), plan)
        if dense is not None:
            dense.layer_done(self.layer_idx)  # H3
        return hidden + mixed


class ToyDecoder(nn.Module):
    """The decoder (``.layers``); its forward runs H0 before any layer."""

    def __init__(self, weights: dict[str, torch.Tensor], manager: Any, experts: Callable) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ToyDecoderLayer(
                index,
                lambda key, index=index: weights[f"layers.{index}.{key}"],
                manager,
                experts(index) if index else None,
            )
            for index in range(NUM_LAYERS)
        )
        self.norm = _Weight(weights["norm"])
        self._dwdp = manager

    def forward(self, hidden: torch.Tensor, plan: _ForwardPlan) -> torch.Tensor:
        dwdp = self._dwdp
        if dwdp is not None and dwdp.dense is not None:
            dwdp.ensure_primed()  # H0
        return self.forward_layers(hidden, plan)

    def forward_layers(self, hidden: torch.Tensor, plan: _ForwardPlan) -> torch.Tensor:
        """The layers without H0: what a forward path that bypasses the decoder's entry runs."""
        for layer in self.layers:
            hidden = layer(hidden, plan)
        return _rms_norm(hidden, self.norm.weight)


class ToyCausalLM(nn.Module):
    """The model ``DwdpManager.setup`` gets; the decoder is ``.model``.

    Args:
        weights: ``_toy_weights``; the modules take their tensors from it and the caller drops
            it, so the modules hold the only references.
        manager: The DWDP manager the hooks reach, captured at construction as in Kimi-K3.
        experts: Covered layer -> its replicated routed experts.
    """

    def __init__(self, weights: dict[str, torch.Tensor], manager: Any, experts: Callable) -> None:
        super().__init__()
        self.model = ToyDecoder(weights, manager, experts)

    def forward(self, hidden: torch.Tensor, plan: _ForwardPlan) -> torch.Tensor:
        return self.model(hidden, plan)


def _set_dwdp(model: nn.Module, manager: Any) -> None:
    """Point the model's hooks at ``manager`` (None: a replicated model without DWDP)."""
    for module in model.modules():
        if "_dwdp" in vars(module):
            module._dwdp = manager


# ---------------------------------------------------------------------------------------------
# Iteration plans and what each one does
# ---------------------------------------------------------------------------------------------

_ABORT_POINTS = ("after_h1", "after_h2", "in_moe")


@dataclass
class _Iteration:
    """One forward of the stress loop.

    Attributes:
        index: Position in the loop.
        kind: ``clean``; ``abort`` (raises at ``abort_point`` of ``abort_layer``, after its H1);
            ``abort0`` (raises in replicated layer 0, after H0 primed); ``unprimed`` (skips H0,
            only the experts are primed, by the test).
        executor: ``DwdpManager.prefetch_first_layers`` runs before the forward, outside the
            execution stream, as the executor loop calls it; else only H0 primes.
        tokens: Rows of the forward's input.
        abort_layer: Covered layer of an ``abort``.
        abort_point: ``after_h1``, ``after_h2`` or ``in_moe``.
        attn_spin_us: Covered layer -> spin before the attention's last weight read.
        moe_spin_us: Covered layer -> spin before the MoE block's last dense weight read.
    """

    index: int
    kind: str
    executor: bool
    tokens: int
    abort_layer: int | None = None
    abort_point: str | None = None
    attn_spin_us: dict[int, int] = field(default_factory=dict)
    moe_spin_us: dict[int, int] = field(default_factory=dict)

    @property
    def completes(self) -> bool:
        return self.kind in ("clean", "unprimed")

    def forward_plan(self, probes: torch.Tensor | None) -> _ForwardPlan:
        abort = None
        if self.kind == "abort":
            abort = (self.abort_layer, self.abort_point)
        elif self.kind == "abort0":
            abort = (0, "layer0")
        return _ForwardPlan(
            attn_spin={layer: us * CYCLES_PER_US for layer, us in self.attn_spin_us.items()},
            moe_spin={layer: us * CYCLES_PER_US for layer, us in self.moe_spin_us.items()},
            abort=abort,
            probes=probes,
        )

    def probed(self, layer: int, index: int) -> bool:
        """Whether this forward enqueued the reads of probe ``index`` of covered ``layer``."""
        if self.completes:
            return True
        if self.kind == "abort0" or layer > self.abort_layer:
            return False
        return layer < self.abort_layer or index == 0 or self.abort_point == "in_moe"

    def describe(self) -> str:
        where = f"@L{self.abort_layer}/{self.abort_point}" if self.kind == "abort" else ""
        return (
            f"#{self.index} {self.kind}{where} {'exec' if self.executor else 'h0'} M={self.tokens}"
        )


def _next_state(it: _Iteration, state: str) -> str:
    """Dense state after ``it``: ``clean``, ``open`` (forward_open) or ``primed``."""
    if it.kind == "abort":
        return "open"
    if it.kind == "abort0":
        return "primed"
    return "clean"


def _expected_counters(it: _Iteration, state: str) -> tuple[int, int, int]:
    """``(WAR barriers, demand fetches, WAR fences)`` that ``it`` adds after ``state``.

    A forward after an aborted one runs the WAR barrier in H0 and re-primes, so neither a demand
    fetch nor a fence follows.  A forward that bypasses H0 after a completed one
    finds the first two layers' slots holding the last two layers, so H1 of each takes one fenced
    demand fetch; from there the H2 prefetches keep ahead.
    """
    barriers = int(state == "open" and it.kind != "unprimed")
    demand = 2 if it.kind == "unprimed" else 0
    return barriers, demand, 0


def _expected_copies(
    it: _Iteration, state: str, layers: tuple[int, ...], *, dense: bool
) -> list[tuple[str, int]]:
    """Launches on the copy stream, in host order, for ``it`` after ``state``.

    ``("L", n)`` is a dense gather, ``("E", n)`` an expert prefetch.  Priming enqueues
    ``L(1) E(1) L(2) E(2)`` (only ``E`` without dense DWDP); layer ``p`` enqueues ``L(p+2)`` at H2
    and ``E(p+2)`` in its MoE block; an unprimed forward demand-fetches ``L(1)`` and ``L(2)`` at H1.
    """
    seq: list[tuple[str, int]] = []
    if it.kind == "unprimed":
        seq += [("E", layers[0]), ("E", layers[1])]
    elif state != "primed":
        for layer in layers[:2]:
            seq += [("L", layer), ("E", layer)] if dense else [("E", layer)]
    if it.kind == "abort0":
        return seq
    for position, layer in enumerate(layers):
        if it.kind == "unprimed" and position < 2:
            seq.append(("L", layer))
        stop = it.kind == "abort" and layer == it.abort_layer
        if stop and it.abort_point == "after_h1":
            return seq
        ahead = layers[position + 2] if position + 2 < len(layers) else None
        if dense and ahead is not None:
            seq.append(("L", ahead))
        if stop:
            return seq
        if ahead is not None:
            seq.append(("E", ahead))
    return seq


def _plan_quota_met(plan: list[_Iteration]) -> bool:
    kinds = Counter(it.kind for it in plan)
    points = {it.abort_point for it in plan if it.kind == "abort"}
    clean = [it for it in plan if it.kind == "clean"]
    recoveries = {
        nxt.executor for prev, nxt in zip(plan, plan[1:]) if prev.kind == "abort" and nxt.completes
    }
    return (
        kinds["abort"] >= 4
        and points == set(_ABORT_POINTS)
        and kinds["abort0"] >= 1
        and kinds["unprimed"] >= 2
        and sum(it.executor for it in clean) >= 5
        and sum(not it.executor for it in clean) >= 5
        and recoveries == {True, False}
    )


def _plan_iterations(rng: random.Random, count: int, *, anomalies: bool) -> list[_Iteration]:
    """A random loop of ``count`` forwards; without ``anomalies`` every forward is executor-primed.

    With anomalies, the first and last forwards are clean, an unprimed forward follows only a
    completed one (so it finds the previous forward's last layers in the slots), and the plan is
    redrawn until it holds every anomaly, both priming paths and both recovery entries.
    """
    for _ in range(1000):
        plan: list[_Iteration] = []
        state = "clean"
        for index in range(count):
            kind = "clean"
            if anomalies and 0 < index < count - 1:
                draw = rng.random()
                if draw < 0.14:
                    kind = "abort"
                elif draw < 0.20:
                    kind = "abort0"
                elif draw < 0.28 and state == "clean":
                    kind = "unprimed"
            executor = kind != "unprimed" and (not anomalies or index == 0 or rng.random() < 0.5)
            tokens = rng.randint(1, 16) if rng.random() < 0.5 else rng.randint(17, MAX_TOKENS)
            it = _Iteration(index, kind, executor, tokens)
            if kind == "abort":
                it.abort_layer = rng.choice(COVERED)
                it.abort_point = rng.choice(_ABORT_POINTS)
            if anomalies:
                for layer in COVERED:
                    if rng.random() < WINDOW_SPIN_PROBABILITY:
                        it.attn_spin_us[layer] = rng.randint(*WINDOW_SPIN_US)
                    if rng.random() < WINDOW_SPIN_PROBABILITY:
                        it.moe_spin_us[layer] = rng.randint(*WINDOW_SPIN_US)
            plan.append(it)
            state = _next_state(it, state)
        if not anomalies or _plan_quota_met(plan):
            return plan
    raise RuntimeError("no iteration plan met the anomaly quota in 1000 draws")


# ---------------------------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scenario:
    """One group of worker processes.

    Attributes:
        name: Label in reports.
        num_ranks: DWDP group size, one process and GPU per rank.
        segments: ``TRTLLM_DWDP_DENSE_SEGMENTS``.
        required: ``TRTLLM_DWDP_DENSE_REQUIRED``.
        fault: None (protocol stress), ``p2_allocate`` or ``phase_v``.
        fault_rank: The rank that fails.
        iterations: Forwards after setup.
        seed: Seeds the ranks' inputs and iteration plans.
    """

    name: str
    num_ranks: int
    segments: str = "attn,moe"
    required: bool = True
    fault: str | None = None
    fault_rank: int = 0
    iterations: int = STRESS_ITERATIONS
    seed: int = 1


def _visible_gpus() -> list[str]:
    """The ``CUDA_VISIBLE_DEVICES`` entries of this process's GPUs, in device order."""
    count = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(index) for index in range(count)]
    return [entry.strip() for entry in visible.split(",") if entry.strip()][:count]


def _require_gpus(num_ranks: int) -> list[str]:
    """One GPU per rank and FABRIC peer handles, else skip; returns the ranks' GPUs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    gpus = _visible_gpus()
    if len(gpus) < num_ranks:
        pytest.skip(f"needs {num_ranks} GPUs, found {len(gpus)}")
    if vmm.peer_handle_type() != FABRIC:
        pytest.skip("dense DWDP needs FABRIC peer handles (aarch64 / NVL72)")
    return gpus[:num_ranks]


def _package_root() -> str | None:
    """The PYTHONPATH entry that makes workers import this process's ``tensorrt_llm``."""
    root = os.path.dirname(os.path.dirname(os.path.realpath(tensorrt_llm.__file__)))
    site_dirs = {os.path.realpath(p) for p in [*site.getsitepackages(), site.getusersitepackages()]}
    return None if root in site_dirs else root


def _worker_env(gpu: str, rank: int, scenario: _Scenario, dwdp_port: int) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_DROPPED_ENV_PREFIXES) and key not in _DROPPED_ENV_VARS
    }
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONUNBUFFERED": "1",
            "TLLM_LOG_LEVEL": "INFO",
            DWDP_RANK_ENV: str(rank),
            DWDP_MASTER_ADDR_ENV: _MASTER_ADDR,
            DWDP_MASTER_PORT_ENV: str(dwdp_port),
            DWDP_TIMEOUT_ENV: str(_COLLECTIVE_TIMEOUT_S),
            DENSE_ENV: "1",
            DENSE_SEGMENTS_ENV: scenario.segments,
            DENSE_REQUIRED_ENV: "1" if scenario.required else "0",
            DENSE_DEBUG_ENV: "1",
            _GATHER_ENABLE_ENV: "1",
            _GATHER_KERNEL_ENV: "tma",
            _GATHER_BLOCKS_ENV: str(GATHER_BLOCKS),
        }
    )
    if os.environ.get(_GATHER_LIB_ENV):
        env[_GATHER_LIB_ENV] = os.environ[_GATHER_LIB_ENV]
    root = _package_root()
    if root:
        env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH")) if p)
    return env


def _port_is_free(port: int) -> bool:
    """Whether a listener could bind ``port`` on every IPv4 and IPv6 address."""
    for family, host in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue
        with sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                sock.bind((host, port))
            except OSError:
                return False
    return True


def _pick_ports(count: int) -> list[int]:
    """Free store ports below the ephemeral range, away from the torch and DWDP defaults."""
    low = 32768
    with contextlib.suppress(OSError, ValueError):
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            low = int(f.read().split()[0])
    candidates = [p for p in range(20000, min(low, 32768)) if not 29400 <= p < 29700]
    random.SystemRandom().shuffle(candidates)  # independent of any seed another test sets
    ports = [port for port in candidates[:512] if _port_is_free(port)][:count]
    if len(ports) < count:
        raise RuntimeError(f"found only {len(ports)} free ports, need {count}")
    return ports


@dataclass
class _Outcome:
    """What the workers of one group did, as the parent saw it and as they reported."""

    workdir: Path
    reports: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    returncodes: dict[int, int | None] = field(default_factory=dict)
    hung: list[int] = field(default_factory=list)

    def log(self, rank: int) -> str:
        path = self.workdir / f"rank{rank}.log"
        return path.read_text(errors="replace") if path.exists() else ""

    def describe(self, ranks: list[int] | None = None) -> str:
        lines = [f"workdir {self.workdir}"]
        for rank, report in self.reports.items():
            lines.append(
                f"  rank {rank}: rc={self.returncodes.get(rank)} "
                f"report={'missing' if report is None else 'written'}"
            )
        for rank in (ranks if ranks is not None else list(self.reports))[:4]:
            lines.append(f"--- rank {rank} log tail ---")
            lines.append(self.log(rank)[-4000:])
        return "\n".join(lines)


def _run_group(workdir: Path, gpus: list[str], scenario: _Scenario) -> _Outcome:
    """Run one worker per GPU and wait until all exit or the deadline passes."""
    test_port, dwdp_port = _pick_ports(2)
    spec_path = workdir / "spec.json"
    spec_path.write_text(
        json.dumps({**asdict(scenario), "test_port": test_port, "parent_pid": os.getpid()})
    )
    outcome = _Outcome(workdir)
    procs: dict[int, subprocess.Popen] = {}
    try:
        for rank, gpu in enumerate(gpus):
            with open(workdir / f"rank{rank}.log", "wb") as log:
                procs[rank] = subprocess.Popen(
                    [sys.executable, "-u", "-X", "faulthandler", os.path.abspath(__file__)]
                    + ["--spec", str(spec_path), "--rank", str(rank)]
                    + ["--out", str(workdir / f"rank{rank}.json")],
                    cwd=workdir,
                    env=_worker_env(gpu, rank, scenario, dwdp_port),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        deadline = time.monotonic() + _GROUP_DEADLINE_S
        while any(proc.poll() is None for proc in procs.values()):
            if time.monotonic() > deadline:
                outcome.hung = [rank for rank, proc in procs.items() if proc.poll() is None]
                for rank in outcome.hung:  # dump the Python stacks into the log, then kill
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(procs[rank].pid, signal.SIGUSR1)
                time.sleep(1.0)
                break
            time.sleep(0.2)
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()
        for proc in procs.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)
    for rank in range(len(gpus)):
        outcome.returncodes[rank] = procs[rank].returncode if rank in procs else None
        path = workdir / f"rank{rank}.json"
        outcome.reports[rank] = json.loads(path.read_text()) if path.exists() else None
    return outcome


def _run(scenario: _Scenario, tmp_path: Path) -> dict[int, dict[str, Any]]:
    """Run ``scenario``; fail on a hang, a missing report or any worker finding; skip together."""
    gpus = _require_gpus(scenario.num_ranks)
    out = _run_group(tmp_path, gpus, scenario)
    assert not out.hung, f"ranks {out.hung} hung and were killed\n{out.describe(out.hung)}"
    missing = [rank for rank, report in out.reports.items() if report is None]
    assert not missing, f"ranks {missing} wrote no report\n{out.describe(missing)}"
    failures = [
        f"rank {rank}: {failure}"
        for rank, report in out.reports.items()
        for failure in report["failures"]
    ]
    if failures:
        shown = "\n".join(f"- {failure}" for failure in failures[:40])
        pytest.fail(f"{len(failures)} failure(s) in {scenario.name}\n{shown}\n{out.describe()}")
    skips = {rank: report["skip"] for rank, report in out.reports.items() if report["skip"]}
    if skips:
        assert len(skips) == scenario.num_ranks, f"only ranks {sorted(skips)} skipped: {skips}"
        pytest.skip(f"rank 0: {skips[0]}")
    bad_exits = {rank: code for rank, code in out.returncodes.items() if code != 0}
    assert not bad_exits, f"workers exited with {bad_exits}\n{out.describe(list(bad_exits))}"
    return {rank: report for rank, report in out.reports.items()}


def _ms(value: float) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


def _mb(nbytes: int) -> str:
    return f"{nbytes / 1e6:.1f}MB"


def _setup_line(info: dict[str, Any]) -> str:
    phases = info["setup_phase_ms"]
    local = " ".join(f"{key}={_ms(ms)}" for key, ms in phases.items() if key.startswith("P"))
    waits = " ".join(f"{key}={_ms(ms)}" for key, ms in phases.items() if key.startswith("#"))
    return f"setup {info['setup_s']:.2f}s, local ms {local}, collective ms {waits}"


def _stress_line(rank: int, info: dict[str, Any]) -> str:
    loop = info["loop"]
    memory = info["memory"]
    return (
        f"rank {rank}: {_setup_line(info)}; moved {_mb(info['moved_bytes'])} in "
        f"{info['anchors']} storages ({info['bindings']} references), store "
        f"{_mb(info['store_bytes'])} (handles {_mb(info['store_handle_bytes'])}), slots "
        f"{_mb(info['slot_bytes'])}, gather {_mb(info['gather_bytes_per_forward'])}/forward; "
        f"freed at commit {_mb(memory['freed_at_commit'])} ({info['memory_source']}) and "
        f"{_mb(memory['allocated_freed_at_commit'])} (allocator), original segments "
        f"{_mb(memory['released_original_segments'])}; after release {memory['end_delta']:+d} B "
        f"({info['memory_source']}) and {memory['allocated_end_delta']:+d} B (allocator) from "
        f"expected; "
        f"loop {loop['iterations']} forwards {loop['kinds']} "
        f"({loop['executor_primed']} executor-primed, {loop['probes']} probes), barriers "
        f"{loop['war_barriers']}, demand "
        f"fetches {loop['demand_fetches']}, fences {loop['war_fences']}, {loop['dense_copies']} "
        f"dense + {loop['expert_copies']} expert copies ({_mb(loop['dense_copy_bytes'])}), host "
        f"{_ms(loop['host_ms'])} ms, GPU {_ms(loop['gpu_ms'])} ms, the GPU trailing the host by "
        f"{_ms(loop['gpu_lag_ms_mean'])} ms on average and {_ms(loop['gpu_lag_ms_max'])} ms at most"
    )


def _assert_committed(reports: dict[int, dict[str, Any]], scenario: _Scenario, capfd) -> None:
    verdicts = {rank: report["info"].get("verdict") for rank, report in reports.items()}
    assert set(verdicts.values()) == {"committed"}, f"setup verdicts {verdicts}"
    digests = {report["info"]["digest"] for report in reports.values()}
    assert len(digests) == 1, f"the ranks committed different layouts: {digests}"
    for report in reports.values():
        loop = report["info"]["loop"]
        assert loop["iterations"] == scenario.iterations, loop
        assert loop["compared"] == loop["completed"] > 0, loop
        assert loop["probes"] > 0 and loop["wrong_probes"] == 0, loop
    lines = [_stress_line(rank, report["info"]) for rank, report in reports.items()]
    with capfd.disabled():
        print(f"\n[dense DWDP e2e] {scenario.name} ({scenario.segments}):\n  " + "\n  ".join(lines))


@pytest.mark.parametrize("num_ranks", [2, 4], ids=["2ranks", "4ranks"])
def test_protocol_stress_end_to_end(num_ranks: int, tmp_path: Path, capfd) -> None:
    """S8: 50 stressed forwards over the real setup equal the replicated reference bit for bit.

    See the module docstring: copy order, counters and outputs are checked per forward on every
    rank; the setup commits on every rank with one layout; teardown returns the memory.
    """
    scenario = _Scenario(name=f"stress-{num_ranks}ranks", num_ranks=num_ranks, seed=num_ranks)
    _assert_committed(_run(scenario, tmp_path), scenario, capfd)


def test_attention_only_end_to_end(tmp_path: Path, capfd) -> None:
    """S8 with ``TRTLLM_DWDP_DENSE_SEGMENTS=attn``, the mode of the attention-only canary (S9)."""
    scenario = _Scenario(name="stress-attn-2ranks", num_ranks=2, segments="attn", seed=7)
    _assert_committed(_run(scenario, tmp_path), scenario, capfd)


#: Fault -> (the collective every rank abandons at, text every rank's reason holds).
_FAULTS = {
    "p2_allocate": (2, "injected P2 failure"),
    "phase_v": (4, "Phase V"),
}


@pytest.mark.parametrize("required", [False, True], ids=["required0", "required1"])
@pytest.mark.parametrize("fault", list(_FAULTS))
def test_setup_fault_reaches_one_verdict(fault: str, required: bool, tmp_path: Path, capfd):
    """S5(b): one rank fails in setup; every rank reaches the same verdict without hanging.

    ``p2_allocate``: rank 1's FABRIC store allocation fails in P2 (its first ``cuMemSetAccess``),
    after it created and mapped a handle.  ``phase_v``: rank 0's Phase V sees one peer row of one
    layer corrupted in its slot.  ``REQUIRED=0``: every rank returns None and the model computes
    its references exactly; ``REQUIRED=1``: every rank raises ``DenseSetupError``.  Either way
    the model keeps its original tensors and teardown returns the memory to its pre-setup level.
    """
    collective, reason = _FAULTS[fault]
    fault_rank = 1 if fault == "p2_allocate" else 0
    scenario = _Scenario(
        name=f"{fault}-required{int(required)}",
        num_ranks=2,
        required=required,
        fault=fault,
        fault_rank=fault_rank,
        iterations=FAULT_ITERATIONS,
        seed=11,
    )
    reports = _run(scenario, tmp_path)
    want = "raised" if required else "none"
    reasons = {report["info"]["abandon_reason"] for report in reports.values()}
    assert len(reasons) == 1, f"the ranks abandoned for different reasons: {reasons}"
    lines = []
    for rank, report in reports.items():
        info = report["info"]
        assert info["verdict"] == want, f"rank {rank}: verdict {info['verdict']}, want {want}"
        assert info["abandoned_at"] == collective, (
            f"rank {rank} abandoned at collective #{info['abandoned_at']}, want #{collective}: "
            f"{info['abandon_reason']}"
        )
        assert info["collectives"] == collective, info
        assert f"rank {fault_rank}" in info["abandon_reason"], info["abandon_reason"]
        assert reason in info["abandon_reason"], info["abandon_reason"]
        lines.append(
            f"rank {rank}: {want} at #{collective} after {info['setup_s']:.2f}s "
            f"({_setup_line(info)}); memory after release {info['memory']['end_delta']:+d} B "
            f"({info['memory_source']}) and {info['memory']['allocated_end_delta']:+d} B "
            f"(allocator) from the pre-setup level; {info['loop']['compared']} forwards exact"
        )
    with capfd.disabled():
        print(f"\n[dense DWDP e2e] {scenario.name}:\n  " + "\n  ".join(lines))


# ---------------------------------------------------------------------------------------------
# Worker side (one DWDP rank)
# ---------------------------------------------------------------------------------------------


class _Skip(Exception):
    """Raised in a worker when this machine cannot run the test."""


class _Report:
    """Failures, notes and facts a worker hands back to the parent."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []
        self.info: dict[str, Any] = {}
        self.skip: str | None = None

    def check(self, ok: bool, message: str) -> bool:
        if not ok:
            self.failures.append(message)
        return ok

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Record an exception of the step as a failure, so the collectives stay in step."""
        try:
            yield
        except Exception:  # every rank must reach the next collective; the error is reported
            self.failures.append(f"{name} raised:\n{traceback.format_exc()}")

    def to_json(self) -> dict[str, Any]:
        return {
            "failures": self.failures,
            "notes": self.notes,
            "info": self.info,
            "skip": self.skip,
        }


def _die_with_parent(parent_pid: int) -> None:
    """SIGKILL this worker when the test process dies (PR_SET_PDEATHSIG)."""
    with contextlib.suppress(AttributeError, OSError):
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)
    if os.getppid() != parent_pid:
        os._exit(3)


def _capability_problem(device_id: int) -> str | None:
    """None if this process can export FABRIC handles and launch the gather, else what is missing."""
    if vmm.peer_handle_type() != FABRIC:
        return f"peer handles are {vmm.peer_handle_type().name}, not FABRIC"
    if not fabric_gather.is_enabled() or fabric_gather.gather_op() is None:
        return "trtllm::dwdp_fabric_gather is not registered by this build"
    granularity = vmm.get_allocation_granularity(device_id)
    try:
        handle = vmm.create_fabric_handle(granularity, device_id)
    except RuntimeError as exc:
        return f"cannot create a FABRIC handle: {exc}"
    try:
        transport._export_handle(handle, FABRIC)
    except RuntimeError as exc:
        return f"cannot export a FABRIC handle: {exc}"
    finally:
        vmm.release_handle(handle)
    return None


class _RecordingLogger:
    """Forwards to the TensorRT-LLM logger and keeps ``(level, message)`` of setup's lines."""

    def __init__(self, inner: Any, lines: list[tuple[str, str]]) -> None:
        self._inner = inner
        self._lines = lines

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _emit(self, level: str, message: Any, *args: Any, **kwargs: Any) -> None:
        self._lines.append((level, str(message)))
        getattr(self._inner, level)(message, *args, **kwargs)

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._emit("debug", message, *args, **kwargs)

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._emit("info", message, *args, **kwargs)

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._emit("warning", message, *args, **kwargs)

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._emit("error", message, *args, **kwargs)


class _ToyExpertManager:
    """The ``DWDPWeightManager`` calls of ``DwdpManager`` and dense setup, over local copies.

    The expert DWDP protocol on one copy stream: two expert slots by MoE position parity, a
    prefetch (RAW) and a consume (WAR) event per slot, both consume events recorded at
    construction.  A prefetch waits the slot's consume event, spins (standing in for the ~15 GB
    peer gather) and copies the layer's window of the replicated expert table into the slot;
    ``wait_and_bind`` makes the current stream wait the slot's prefetch event and binds the slot
    to the backend.  Every prefetch is appended to the shared copy log.
    """

    def __init__(
        self,
        layers: tuple[int, ...],
        device: torch.device,
        table: torch.Tensor,
        rng: random.Random,
        log: list[tuple[str, int]],
    ) -> None:
        self._layers = layers
        self._device = device
        self._table = table.view(U8)
        self._copy_stream = torch.cuda.Stream(device)
        self._use_gather = fabric_gather.is_enabled() and fabric_gather.gather_op() is not None
        self._slots = tuple(torch.zeros(EXPERT_BYTES, dtype=U8, device=device) for _ in range(2))
        self._views = tuple(slot.view(BF16).view(EXPERT_ROWS, LATENT) for slot in self._slots)
        self._prefetch_events = [torch.cuda.Event() for _ in range(2)]
        self._consume_events = [torch.cuda.Event() for _ in range(2)]
        current = torch.cuda.current_stream(device)
        for event in self._consume_events:
            event.record(current)
        self._rng = rng
        self._log = log
        self.releases = 0

    @property
    def copy_stream(self) -> torch.cuda.Stream:
        return self._copy_stream

    @property
    def use_gather(self) -> bool:
        return self._use_gather

    @property
    def use_batched_prefetch(self) -> bool:
        return False

    def first_moe_layer(self) -> int:
        return self._layers[0]

    def next_moe_layer(self, layer: int) -> int | None:
        return next((later for later in self._layers if later > layer), None)

    def replicated(self, layer: int) -> torch.Tensor:
        """The layer's experts in the table: the bytes a prefetch copies, as a model holds them."""
        start = self._layers.index(layer) * EXPERT_STRIDE
        return self._table[start : start + EXPERT_BYTES].view(BF16).view(EXPERT_ROWS, LATENT)

    def _slot(self, layer: int) -> int:
        return self._layers.index(layer) % 2

    def slot_index(self, layer: int) -> int:
        """The slot of ``layer`` (``DwdpManager`` asks when the cross-forward prefetch is on)."""
        return self._slot(layer)

    def prefetch_layer(self, layer: int) -> None:
        slot = self._slot(layer)
        self._log.append(("E", layer))
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._consume_events[slot])
            torch.cuda._sleep(self._rng.randint(*EXPERT_SPIN_US) * CYCLES_PER_US)
            self._views[slot].copy_(self.replicated(layer))
            self._prefetch_events[slot].record(self._copy_stream)

    def wait_and_bind(self, backend: ToyRoutedExperts, layer: int) -> None:
        slot = self._slot(layer)
        torch.cuda.current_stream(self._device).wait_event(self._prefetch_events[slot])
        backend.bound = self._views[slot]

    def record_compute(self, layer: int) -> None:
        stream = torch.cuda.current_stream(self._device)
        self._consume_events[self._slot(layer)].record(stream)

    def release(self) -> None:
        self.releases += 1
        torch.cuda.synchronize(self._device)


@dataclass
class _Corruption:
    """Flips ``nbytes`` bytes of one peer row in a slot right after the gather of ``layer``, once."""

    layer: int
    storage: torch.Tensor | None
    start: int
    nbytes: int
    path: str
    done: bool = False

    def after_launch(self, layer: int) -> None:
        if self.done or layer != self.layer or self.storage is None:
            return
        self.done = True
        self.storage[self.start : self.start + self.nbytes].bitwise_not_()


class _ObservedLauncher:
    """The dense copy launcher setup builds, logging each launch and checking its stream."""

    def __init__(self, inner: Callable, cap: _Captures, copy_stream: torch.cuda.Stream) -> None:
        self._inner = inner
        self._cap = cap
        self._copy_stream = copy_stream

    def __call__(self, layer: int) -> None:
        cap = self._cap
        stream = torch.cuda.current_stream()
        if stream.cuda_stream != self._copy_stream.cuda_stream and cap.stream_errors < 4:
            cap.stream_errors += 1
            cap.failures.append(f"L({layer}) launched on {stream}, not on the copy stream")
        cap.copies.append(("L", layer))
        self._inner(layer)
        if cap.corruption is not None:
            cap.corruption.after_launch(layer)

    def release(self) -> None:
        release = getattr(self._inner, "release", None)
        if callable(release):
            release()


@dataclass
class _Captures:
    """What the test observes of setup: layout, slots, plans, launches and log lines."""

    copies: list[tuple[str, int]]
    failures: list[str]
    layout: Any = None
    slots: dict[SegmentKind, tuple] = field(default_factory=dict)
    plans: dict[int, Any] = field(default_factory=dict)
    corruption: _Corruption | None = None
    corrupt_layer: int | None = None
    rank: int = 0
    lines: list[tuple[str, str]] = field(default_factory=list)
    stream_errors: int = 0

    def make_corruption(self) -> None:
        """Pick bytes of an anchor inside a peer's row of ``corrupt_layer``'s attention blob."""
        layout = self.layout
        segment = layout.segment(self.corrupt_layer, ATTN)
        slot = self.slots[ATTN][layout.slot_index(ATTN, self.corrupt_layer)]
        row_start, row_end = segment.row_range((self.rank + 1) % layout.num_ranks)
        for anchor in segment.anchors:
            start = max(anchor.blob_offset, row_start)
            if min(anchor.blob_end, row_end) - start >= 64:
                self.corruption = _Corruption(
                    self.corrupt_layer, slot.storage, slot.pad + start, 64, anchor.path
                )
                return
        raise RuntimeError(f"no anchor of layer {self.corrupt_layer} reaches the peer row")

    def drop_device_references(self) -> None:
        self.layout = None
        self.slots = {}
        self.plans = {}
        if self.corruption is not None:
            self.corruption.storage = None


@contextlib.contextmanager
def _observe_setup(cap: _Captures) -> Iterator[None]:
    """Keep what setup builds and log its launches; setup's own code runs unchanged."""
    collect, allocate = dense_setup.collect_dense_layout, dense_setup.allocate_slot_ring
    copy_launcher, logger = dense_setup._copy_launcher, dense_setup.logger

    def observed_collect(*args, **kwargs):
        cap.layout = collect(*args, **kwargs)
        return cap.layout

    def observed_allocate(spec, device):
        ring = allocate(spec, device)
        cap.slots[spec.segment] = ring
        return ring

    def observed_launcher(plans, weight_manager, device):
        cap.plans = dict(plans)
        if cap.corrupt_layer is not None:
            cap.make_corruption()
        inner = copy_launcher(plans, weight_manager, device)
        return _ObservedLauncher(inner, cap, weight_manager.copy_stream)

    dense_setup.collect_dense_layout = observed_collect
    dense_setup.allocate_slot_ring = observed_allocate
    dense_setup._copy_launcher = observed_launcher
    dense_setup.logger = _RecordingLogger(logger, cap.lines)
    try:
        yield
    finally:
        dense_setup.collect_dense_layout = collect
        dense_setup.allocate_slot_ring = allocate
        dense_setup._copy_launcher = copy_launcher
        dense_setup.logger = logger


@contextlib.contextmanager
def _failing_set_access(rank: int) -> Iterator[None]:
    """The first ``vmm.set_access`` raises: the FABRIC store's first chunk is created and mapped."""
    original = vmm.set_access
    calls = 0

    def set_access(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(f"injected P2 failure: cuMemSetAccess refused on rank {rank}")
        return original(*args, **kwargs)

    vmm.set_access = set_access
    try:
        yield
    finally:
        vmm.set_access = original


class _GroupTimeline:
    """Times every collective of the DWDP group (the wrapper is an instance attribute)."""

    def __init__(self, group: Any) -> None:
        self.calls: list[tuple[float, float]] = []
        inner = group.allgather

        def allgather(obj):
            enter = time.monotonic()
            result = inner(obj)
            self.calls.append((enter, time.monotonic()))
            return result

        group.allgather = allgather

    def phases(self, start: float, end: float) -> dict[str, float]:
        """Local work before each collective (``P<k>``), each collective (``#<k>``), the rest."""
        out: dict[str, float] = {}
        previous = start
        for number, (enter, leave) in enumerate(self.calls, 1):
            out[f"P{number}"] = (enter - previous) * 1e3
            out[f"#{number}"] = (leave - enter) * 1e3
            previous = leave
        out[f"P{len(self.calls) + 1}"] = (end - previous) * 1e3
        return {key: round(value, 3) for key, value in out.items()}


def _tensor_at(layer: nn.Module, path: str) -> tuple[nn.Module, str, int | None, torch.Tensor]:
    """``(owner, name, index, tensor)`` of a reference path like ``attn._mtp_conv_weights[1]``."""
    owner_path, _, name = path.rpartition(".")
    owner = layer.get_submodule(owner_path) if owner_path else layer
    index = None
    match = re.fullmatch(r"(\w+)\[(\d+)\]", name)
    if match:
        name, index = match.group(1), int(match.group(2))
    if name in owner._parameters:
        value = owner._parameters[name]
    elif name in owner._buffers:
        value = owner._buffers[name]
    else:
        value = vars(owner)[name]
    if index is not None:
        value = value[index]
    return owner, name, index, value


@dataclass
class _ModelFacts:
    """What the module tensors were before setup, without holding any of them."""

    moved: dict[str, tuple[torch.dtype, tuple[int, ...]]]
    parameters: dict[str, weakref.ReferenceType]
    buffers: set[str]
    stays: dict[str, int]
    anchors: dict[int, int]
    anchor_refs: list[weakref.ReferenceType]


def _model_facts(model: ToyCausalLM, segments: tuple[SegmentKind, ...]) -> _ModelFacts:
    moved, parameters, buffers, stays, anchors, refs = {}, {}, set(), {}, {}, []
    for index, layer in enumerate(model.model.layers):
        if not layer.is_moe:
            for name, tensor in layer.named_parameters():
                stays[f"layers.{index}.{name}"] = tensor.untyped_storage().data_ptr()
            continue
        stay_paths = list(STAYS) + [
            path for kind in (ATTN, MOE) if kind not in segments for path in MOVED[kind]
        ]
        for path in stay_paths:
            stays[f"layers.{index}.{path}"] = (
                _tensor_at(layer, path)[3].untyped_storage().data_ptr()
            )
        for kind in segments:
            for path in MOVED[kind]:
                owner, name, _, tensor = _tensor_at(layer, path)
                key = f"layers.{index}.{path}"
                moved[key] = (tensor.dtype, tuple(tensor.shape))
                if name in owner._parameters:
                    parameters[key] = weakref.ref(owner._parameters[name])
                if name in owner._buffers:
                    buffers.add(key)
                storage = tensor.untyped_storage()
                if storage.data_ptr() not in anchors:
                    anchors[storage.data_ptr()] = storage.nbytes()
                    refs.append(weakref.ref(storage))
    return _ModelFacts(moved, parameters, buffers, stays, anchors, refs)


class _DeviceMemory:
    """Device memory in use less what NVML attributes to other processes on the device.

    ``torch.cuda.mem_get_info`` is device wide; subtracting NVML's per-process counts of every
    other process makes the measure immune to other processes sharing the GPU.  Memory no process
    is charged for still counts, such as a FABRIC store chunk a peer keeps imported after its
    exporter released it.  Without NVML this is the device's used memory.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._nvml: Any = None
        self._handle: Any = None
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
            uuid = str(torch.cuda.get_device_properties(device).uuid)
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetUUID(handle)
                if (name.decode() if isinstance(name, bytes) else name).endswith(uuid):
                    self._nvml, self._handle = pynvml, handle
            processes = self._processes()
            if processes is not None and os.getpid() not in {pid for pid, _ in processes}:
                self._nvml = self._handle = None
        except pynvml.NVMLError:
            self._nvml = self._handle = None

    @property
    def source(self) -> str:
        return "device less others" if self._handle is not None else "device"

    def used(self) -> int:
        """Bytes in use after a collection and ``empty_cache``, less other processes' bytes."""
        gc.collect()
        torch.cuda.synchronize(self._device)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self._device)
        others = sum(used for pid, used in self._processes() or () if pid != os.getpid())
        return total - free - others

    def _processes(self) -> list[tuple[int, int]] | None:
        if self._handle is None:
            return None
        return [
            (process.pid, process.usedGpuMemory or 0)
            for process in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        ]


def _block_bytes(nbytes: int) -> int:
    """Bytes the caching allocator hands out for a small-pool request of ``nbytes``."""
    return max(512, -(-nbytes // 512) * 512)


def _live_blocks(device: torch.device) -> dict[int, int]:
    """Address -> size of every block the caching allocator has handed out on ``device``.

    A block can exceed its request: the allocator does not split off a large-pool remainder of
    1 MiB or less, so the sizes come from the allocator rather than from the tensors.
    """
    return {
        block["address"]: block["size"]
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == device.index
        for block in segment["blocks"]
        if block["state"] != "inactive"
    }


def _staging_blocks(device: torch.device, slots: dict[SegmentKind, tuple], plans: dict) -> int:
    """Allocator bytes setup holds after the commit: the slots and the gather launch arguments."""
    blocks = _live_blocks(device)
    slot_bytes = sum(blocks[slot.storage.data_ptr()] for ring in slots.values() for slot in ring)
    arguments = sum(
        3 * _block_bytes(8 * len(plan.descriptors)) + _block_bytes(8 * (len(plan.rows) + 1))
        for plan in plans.values()
        if not plan.is_empty
    )
    return slot_bytes + arguments


def _original_segment_bytes(device: torch.device, anchor_ptrs: set[int]) -> int:
    """Bytes of the allocator segments whose live blocks are all original anchor storages.

    Dropping the originals at commit frees exactly these segments, and the commit's
    ``empty_cache`` returns them to the device.
    """
    total = 0
    for segment in torch.cuda.memory_snapshot():
        if segment["device"] != device.index:
            continue
        live = [block for block in segment["blocks"] if block["state"] != "inactive"]
        if live and all(block["address"] in anchor_ptrs for block in live):
            total += segment["total_size"]
    return total


def _warm_up_blas(device: torch.device, streams: list[torch.cuda.Stream]) -> None:
    """Run the GEMM kinds of the forward on every stream that runs one, before any long-lived
    allocation, so the per-stream BLAS workspaces never share a segment with a weight."""
    for stream in streams:
        with torch.cuda.stream(stream):
            for dtype in (BF16, F32):
                for rows in (1, 64):
                    x = torch.ones(rows, HIDDEN, dtype=dtype, device=device)
                    F.linear(x, torch.ones(256, HIDDEN, dtype=dtype, device=device))
        stream.synchronize()


def _warm_up_kernels(device: torch.device) -> None:
    """Load the kernels setup, the loop and teardown use first, so no module load shows as a leak."""
    a = torch.zeros(1 << 20, dtype=U8, device=device)
    b = torch.ones_like(a)
    tensor_checksum(torch.randn(1 << 16, device=device))
    torch.equal(a, b)
    a.fill_(0xFF)
    b.copy_(a)
    a[:64].bitwise_not_()
    bool((a == 0xFF).all())
    torch.equal(a.view(torch.int16), b.view(torch.int16))
    bool((a.view(torch.int16) != b.view(torch.int16)).any(dim=-1).any())
    args = fabric_gather.build_plan([[(b.data_ptr(), a.data_ptr(), 1 << 16)]], device)
    fabric_gather.gather_op()(*args, fabric_gather.THREADS)
    torch.cuda._sleep(1000)
    torch.cuda.synchronize(device)


@dataclass
class _Rank:
    """One worker's run: its spec, devices, stores, model and observations."""

    report: _Report
    spec: dict[str, Any]
    rank: int
    device: torch.device
    test_group: StoreDwdpGroup
    execution: torch.cuda.Stream = None
    fac: Any = None
    manager: Any = None
    model: ToyCausalLM | None = None
    wm: _ToyExpertManager | None = None
    cap: _Captures | None = None
    iterations: list[_Iteration] = field(default_factory=list)
    inputs: torch.Tensor | None = None
    references: torch.Tensor | None = None
    outputs: torch.Tensor | None = None
    probes: torch.Tensor | None = None
    reference_probes: torch.Tensor | None = None
    segments: tuple[SegmentKind, ...] = ()
    fillers: list[torch.Tensor] = field(default_factory=list)
    meter: _DeviceMemory | None = None

    @property
    def stress(self) -> bool:
        return self.spec["fault"] is None


def _step(ctx: _Rank, it: _Iteration) -> bool:
    """One forward as the executor runs it; True if it completed.

    Executor priming runs on the default stream, outside the execution stream; the forward runs
    on the execution stream and the default stream waits it only when the forward returned (an
    exception skips the wait, as ``PyExecutor._forward_step`` does).
    """
    manager, wm = ctx.manager, ctx.wm
    if it.kind == "unprimed":
        first = wm.first_moe_layer()
        wm.prefetch_layer(first)
        wm.prefetch_layer(wm.next_moe_layer(first))
    elif it.executor:
        manager.prefetch_first_layers()
    hidden = ctx.inputs[it.index, : it.tokens]
    ctx.execution.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(ctx.execution):
            plan = it.forward_plan(ctx.probes[it.index])
            if it.kind == "unprimed":
                out = ctx.model.model.forward_layers(hidden, plan)
            else:
                out = ctx.model(hidden, plan)
            ctx.outputs[it.index, : it.tokens].copy_(out)
    except _InjectedAbort:
        return False
    torch.cuda.current_stream().wait_stream(ctx.execution)
    return True


def _compute_references(ctx: _Rank) -> None:
    """The replicated model's output for every planned input: hooks and DWDP out of the way."""
    _set_dwdp(ctx.model, None)
    try:
        ctx.execution.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(ctx.execution):
            for it in ctx.iterations:
                plan = _ForwardPlan(probes=ctx.reference_probes[it.index])
                out = ctx.model(ctx.inputs[it.index, : it.tokens], plan)
                ctx.references[it.index, : it.tokens].copy_(out)
        torch.cuda.synchronize(ctx.device)
    finally:
        _set_dwdp(ctx.model, ctx.manager)
    finite = all(
        bool(torch.isfinite(ctx.references[it.index, : it.tokens]).all()) for it in ctx.iterations
    )
    ctx.report.check(finite, "the replicated reference has non-finite outputs")


def _compare_outputs(ctx: _Rank, completed: list[_Iteration]) -> None:
    """Every completed forward's output must equal its replicated reference bit for bit."""
    mismatched = []
    for it in completed:
        got = ctx.outputs[it.index, : it.tokens].view(torch.int16)
        want = ctx.references[it.index, : it.tokens].view(torch.int16)
        if not torch.equal(got, want):
            wrong = (got != want).any(dim=-1).nonzero().flatten().tolist()
            mismatched.append(f"{it.describe()}: rows {wrong[:8]} of {it.tokens} differ")
    ctx.report.check(
        not mismatched,
        f"{len(mismatched)} of {len(completed)} forwards differ from the replicated reference: "
        + "; ".join(mismatched[:8]),
    )


def _compare_probes(ctx: _Rank) -> tuple[int, int]:
    """Probes of every read a forward enqueued must equal the reference's; returns the counts.

    A probe past an abort point must not have been written at all.

    Returns:
        ``(probes compared, probes that differ)``.
    """
    values = ctx.probes.cpu()
    probes = values.view(torch.int32)
    reference_probes = ctx.reference_probes.cpu().view(torch.int32)
    probed = wrong = unexpected = 0
    for it in ctx.iterations:
        for layer in COVERED:
            for index in range(2):
                if not it.probed(layer, index):
                    unexpected += math.isfinite(float(values[it.index, layer, index]))
                    continue
                probed += 1
                if probes[it.index, layer, index] != reference_probes[it.index, layer, index]:
                    wrong += 1
                    if wrong <= 8:
                        ctx.report.failures.append(
                            f"forward {it.describe()}: probe {index} of layer {layer} differs "
                            f"from the replicated reference"
                        )
    ctx.report.check(unexpected == 0, f"{unexpected} probes were written past an abort point")
    return probed, wrong


def _run_loop(ctx: _Rank, runtime: Any) -> None:
    """Run the planned forwards; check copies and counters per forward, outputs at the end."""
    report, cap = ctx.report, ctx.cap
    barriers = [0]
    if runtime is not None:
        war_barrier = runtime.war_barrier

        def counting_war_barrier() -> None:
            barriers[0] += 1
            war_barrier()

        runtime.war_barrier = counting_war_barrier

    def counters() -> tuple[int, int, int]:
        if runtime is None:
            return 0, 0, 0
        return barriers[0], runtime.demand_fetches, runtime.war_fences

    dense = runtime is not None
    state = "clean"
    completed: list[_Iteration] = []
    first_copy = len(cap.copies)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream())
    began = time.monotonic()
    marks: list[tuple[float, torch.cuda.Event]] = []
    for it in ctx.iterations:
        mark_event = torch.cuda.Event(enable_timing=True)
        mark_event.record(ctx.execution)
        marks.append((time.monotonic() - began, mark_event))
        before, mark = counters(), len(cap.copies)
        try:
            done = _step(ctx, it)
        except Exception:  # a forward that should complete raised: report it, keep the loop going
            report.failures.append(f"forward {it.describe()} raised:\n{traceback.format_exc()}")
            done = False
        after = counters()
        if done:
            completed.append(it)
        report.check(done == it.completes, f"forward {it.describe()} completed={done}")
        deltas = tuple(a - b for a, b in zip(after, before))
        want = _expected_counters(it, state) if dense else (0, 0, 0)
        report.check(
            deltas == want,
            f"forward {it.describe()} after a {state} forward changed (barriers, demand fetches, "
            f"fences) by {deltas}, expected {want}",
        )
        copies = cap.copies[mark:]
        want_copies = _expected_copies(it, state, COVERED, dense=dense)
        report.check(
            copies == want_copies,
            f"forward {it.describe()} after a {state} forward launched {copies} on the copy "
            f"stream, expected {want_copies}",
        )
        state = _next_state(it, state)
    host_ms = (time.monotonic() - began) * 1e3
    current = torch.cuda.current_stream()
    current.wait_stream(ctx.execution)
    current.wait_stream(ctx.wm.copy_stream)
    end.record(current)
    torch.cuda.synchronize(ctx.device)
    # How far the GPU trailed the host: when the execution stream reached each forward's start,
    # less when the host started enqueueing it.
    lags = [start.elapsed_time(event) - host_s * 1e3 for host_s, event in marks]
    _compare_outputs(ctx, completed)
    probed, wrong_probes = _compare_probes(ctx)
    plans = cap.plans if dense else {}
    copies = cap.copies[first_copy:]
    dense_copies = [layer for kind, layer in copies if kind == "L"]
    report.info["loop"] = {
        "iterations": len(ctx.iterations),
        "completed": len(completed),
        "compared": len(completed),
        "kinds": dict(Counter(it.kind for it in ctx.iterations)),
        "executor_primed": sum(it.executor for it in ctx.iterations),
        "war_barriers": barriers[0],
        "demand_fetches": runtime.demand_fetches if dense else 0,
        "war_fences": runtime.war_fences if dense else 0,
        "probes": probed,
        "wrong_probes": wrong_probes,
        "dense_copies": len(dense_copies),
        "expert_copies": sum(kind == "E" for kind, _ in copies),
        "dense_copy_bytes": sum(plans[layer].nbytes for layer in dense_copies if layer in plans),
        "host_ms": host_ms,
        "gpu_ms": start.elapsed_time(end),
        "gpu_lag_ms_max": max(lags),
        "gpu_lag_ms_mean": sum(lags) / len(lags),
    }


def _check_layout(ctx: _Rank, facts: _ModelFacts) -> None:
    """The committed layout: every expected storage moved, the warning for the unlisted name."""
    report, layout = ctx.report, ctx.cap.layout
    report.check(tuple(layout.layers) == COVERED, f"layout covers {layout.layers}")
    report.check(
        tuple(layout.segment_kinds) == ctx.segments, f"layout segments {layout.segment_kinds}"
    )
    report.check(
        layout.moved_bytes == sum(facts.anchors.values()),
        f"layout moves {layout.moved_bytes} bytes, the toy has {sum(facts.anchors.values())} "
        f"bytes in the storages that must move",
    )
    for (layer, kind), segment in layout.segments.items():
        report.check(
            len(segment.anchors) == ANCHORS[kind],
            f"layers.{layer} {kind.value}: {len(segment.anchors)} anchors, want {ANCHORS[kind]}",
        )
    unlisted = [text for text in layout.warnings if "_unlisted_cache" in text]
    report.check(
        len(unlisted) == len(COVERED),
        f"{len(unlisted)} warnings name the unlisted attribute, want {len(COVERED)}: "
        f"{layout.warnings[:4]}",
    )
    granularity = vmm.get_allocation_granularity(ctx.device.index)
    report.info.update(
        digest=layout.digest,
        moved_bytes=layout.moved_bytes,
        store_bytes=layout.store_bytes,
        store_handle_bytes=sum(
            max(vmm.align_up(chunk.handle_bytes, granularity), granularity)
            for chunk in layout.chunks
        ),
        slot_bytes=layout.slot_bytes,
        anchors=sum(1 for _ in layout.anchors()),
        bindings=sum(len(anchor.bindings) for anchor in layout.anchors()),
        gather_bytes_per_forward=sum(plan.nbytes for plan in ctx.cap.plans.values()),
        layout_warnings=len(layout.warnings),
    )


def _check_committed_model(ctx: _Rank, facts: _ModelFacts, runtime: Any) -> None:
    """After commit: moved references view their slots, the rest stays, the originals are gone."""
    report, model = ctx.report, ctx.model
    slot_ptrs = {
        kind: {slot.storage.untyped_storage().data_ptr() for slot in runtime.slots[kind]}
        for kind in ctx.segments
    }
    for kind in ctx.segments:
        for slot in runtime.slots[kind]:
            report.check(
                bool((slot.storage == 0xFF).all()),
                f"slot {kind.value}[{slot.index}] is not filled with 0xFF (DEBUG=1) at commit",
            )
    for index in COVERED:
        layer = model.model.layers[index]
        for kind in ctx.segments:
            for path in MOVED[kind]:
                key = f"layers.{index}.{path}"
                owner, name, _, tensor = _tensor_at(layer, path)
                report.check(
                    tensor.untyped_storage().data_ptr() in slot_ptrs[kind]
                    and (tensor.dtype, tuple(tensor.shape)) == facts.moved[key],
                    f"{key} is not a {kind.value} slot view of its dtype and shape after commit",
                )
                if key in facts.parameters:
                    report.check(
                        owner._parameters.get(name) is facts.parameters[key](),
                        f"{key}: the nn.Parameter object changed",
                    )
                if key in facts.buffers:
                    report.check(name in owner._buffers, f"{key} is no longer a buffer")
    _check_stays(ctx, facts, "after commit")
    alive = sum(ref() is not None for ref in facts.anchor_refs)
    report.check(alive == 0, f"{alive} original storages are still alive after commit")
    report.check(
        bool(getattr(model, "_dense_dwdp_active", False))
        and bool(getattr(model.model, "_dense_dwdp_active", False)),
        "the model or its decoder is not marked _dense_dwdp_active after commit",
    )
    leaks = [
        int(match.group(1))
        for _, line in ctx.cap.lines
        if "committed:" in line and (match := re.search(r"leaks=(\d+)", line))
    ]
    report.check(leaks == [0], f"setup's commit line reports leaks {leaks}")


def _check_stays(ctx: _Rank, facts: _ModelFacts, when: str) -> None:
    for key, pointer in facts.stays.items():
        index, path = key.split(".", 2)[1:]
        tensor = _tensor_at(ctx.model.model.layers[int(index)], path)[3]
        ctx.report.check(
            tensor.untyped_storage().data_ptr() == pointer, f"{key} moved {when}; it must stay"
        )


def _check_detached(ctx: _Rank, facts: _ModelFacts) -> None:
    for key, (dtype, _) in facts.moved.items():
        index, path = key.split(".", 2)[1:]
        tensor = _tensor_at(ctx.model.model.layers[int(index)], path)[3]
        ctx.report.check(
            tensor.numel() == 0 and tensor.dtype == dtype,
            f"{key} is {tuple(tensor.shape)} {tensor.dtype} after release; want a zero-size {dtype}",
        )
    _check_stays(ctx, facts, "after release")


def _teardown(ctx: _Rank, runtime: Any, memory: dict[str, int]) -> None:
    """Release on every rank, a second release, then the memory back at its pre-setup level.

    The pre-setup level counts what a commit dropped: with the runtime committed, the process
    must hold ``released_original_segments`` bytes of device memory less than before setup, and
    the caching allocator ``original_blocks`` bytes less.
    """
    report, manager = ctx.report, ctx.manager
    committed = runtime is not None
    ctx.test_group.Barrier()  # every rank has finished its forwards
    with report.step("DwdpManager.cleanup"):
        manager.cleanup()
    report.check(manager.dense is None, "DwdpManager.dense is set after cleanup")
    report.check(ctx.wm.releases == 1, f"the expert stand-in was released {ctx.wm.releases} times")
    if runtime is not None:
        with report.step("second DenseDWDP.release"):
            runtime.release()
        report.check(not runtime.slots, "the runtime still holds slots after release")
    with report.step("second DwdpManager.cleanup"):
        manager.cleanup()
    ctx.cap.drop_device_references()
    del runtime
    ctx.test_group.Barrier()  # every rank released: no peer maps this rank's store any more
    expected_used = memory["used_before_setup"]
    expected_allocated = memory["allocated_before_setup"]
    if committed:
        expected_used -= memory["released_original_segments"]
        expected_allocated -= memory["original_blocks"]
    used = ctx.meter.used()
    memory["allocated_end_delta"] = torch.cuda.memory_allocated(ctx.device) - expected_allocated
    report.check(
        abs(memory["allocated_end_delta"]) <= _ALLOCATED_TOLERANCE,
        f"after release the caching allocator holds {memory['allocated_end_delta']:+d} bytes "
        f"more than before setup, less the originals the commit dropped",
    )
    for _ in range(_MEMORY_POLLS):
        if abs(used - expected_used) <= _MEMORY_TOLERANCE:
            break
        time.sleep(_MEMORY_POLL_S)
        used = ctx.meter.used()
    memory["end_delta"] = used - expected_used
    report.check(
        abs(memory["end_delta"]) <= _MEMORY_TOLERANCE,
        f"after release {memory['end_delta']:+d} bytes more device memory ({ctx.meter.source}) "
        f"are in use than before setup, less the originals' segments the commit dropped "
        f"(tolerance {_MEMORY_TOLERANCE}): {memory}",
    )
    report.info["memory"] = memory


def _abandon_reason(verdict: str, error: str | None, lines: list[tuple[str, str]]) -> str:
    if verdict == "raised":
        return error or ""
    texts = [text for level, text in lines if level == "error" and "abandoned at" in text]
    return texts[-1] if texts else ""


def _setup(ctx: _Rank) -> tuple[str, Any, dict[str, int], _ModelFacts]:
    """Run ``DwdpManager.setup`` under observation; returns verdict, runtime, memory, facts."""
    report, spec, cap = ctx.report, ctx.spec, ctx.cap
    facts = _model_facts(ctx.model, ctx.segments)
    memory = {
        "used_before_setup": ctx.meter.used(),
        "allocated_before_setup": torch.cuda.memory_allocated(ctx.device),
        "released_original_segments": _original_segment_bytes(ctx.device, set(facts.anchors)),
        "original_blocks": sum(
            size for address, size in _live_blocks(ctx.device).items() if address in facts.anchors
        ),
    }
    report.info["memory_source"] = ctx.meter.source
    timeline = _GroupTimeline(ctx.manager.dwdp_group)
    stamps: dict[str, float] = {}
    ctx.fac.setup_dwdp = _toy_setup_dwdp(ctx, stamps)
    fault = spec["fault"] if spec["fault_rank"] == ctx.rank else None
    if fault == "phase_v":
        cap.corrupt_layer = COVERED[1]
    verdict, error = "error", None
    began = time.monotonic()
    with _observe_setup(cap):
        injection = _failing_set_access(ctx.rank) if fault == "p2_allocate" else None
        with injection or contextlib.nullcontext():
            try:
                ctx.manager.setup(ctx.model)
                verdict = "committed" if ctx.manager.dense is not None else "none"
            except DenseSetupError as exc:
                verdict, error = "raised", str(exc)
            except Exception:  # anything else is a finding; teardown still runs
                report.failures.append(f"DwdpManager.setup raised:\n{traceback.format_exc()}")
    ended = time.monotonic()
    runtime = ctx.manager.dense
    report.info.update(
        verdict=verdict,
        setup_s=ended - began,
        collectives=len(timeline.calls),
        setup_phase_ms=timeline.phases(stamps.get("dense_start", began), ended),
    )
    if verdict == "committed":
        memory["freed_at_commit"] = memory["used_before_setup"] - ctx.meter.used()
        freed = memory["allocated_before_setup"] - torch.cuda.memory_allocated(ctx.device)
        staging = _staging_blocks(ctx.device, cap.slots, cap.plans)
        memory.update(allocated_freed_at_commit=freed, staging_blocks=staging)
        report.check(
            abs(freed - (memory["original_blocks"] - staging)) <= _ALLOCATED_TOLERANCE,
            f"the commit freed {freed} allocated bytes; the originals' blocks less the slots' and "
            f"the gather arguments' are {memory['original_blocks'] - staging}",
        )
        _check_layout(ctx, facts)
        _check_committed_model(ctx, facts, runtime)
    else:
        reason = _abandon_reason(verdict, error, cap.lines)
        match = re.search(r"abandoned at collective #(\d+)", reason)
        report.info.update(
            abandon_reason=reason, abandoned_at=int(match.group(1)) if match else None
        )
        if fault == "phase_v":
            corruption = cap.corruption
            report.check(
                corruption is not None and corruption.done,
                "the Phase V corruption was not applied",
            )
            if corruption is not None:
                report.check(corruption.path in reason, f"{corruption.path} not in {reason!r}")
        _check_stays(ctx, facts, "after an abandoned setup")
        report.check(
            not getattr(ctx.model, "_dense_dwdp_active", False),
            "the model is marked _dense_dwdp_active after an abandoned setup",
        )
        for key, (dtype, shape) in facts.moved.items():
            index, path = key.split(".", 2)[1:]
            tensor = _tensor_at(ctx.model.model.layers[int(index)], path)[3]
            report.check(
                (tensor.dtype, tuple(tensor.shape)) == (dtype, shape),
                f"{key} changed after an abandoned setup",
            )
    return verdict, runtime, memory, facts


def _toy_setup_dwdp(ctx: _Rank, stamps: dict[str, float]) -> Callable:
    """Stands in for ``setup_dwdp``: checks what ``DwdpManager.setup`` passes, returns the stand-in."""

    def setup_dwdp(**kwargs):
        ctx.report.check(
            list(kwargs["layer_indices"]) == list(COVERED)
            and kwargs["comm"] is ctx.manager.dwdp_group
            and kwargs["model"] is ctx.model,
            f"DwdpManager.setup passed setup_dwdp other layers, group or model: layer_indices "
            f"{kwargs['layer_indices']}",
        )
        stamps["dense_start"] = time.monotonic()
        return ctx.wm

    return setup_dwdp


def _toy_teardown_dwdp(ctx: _Rank) -> Callable:
    def teardown_dwdp(model, expected_manager):
        ctx.report.check(
            model is ctx.model and expected_manager is ctx.wm,
            "DwdpManager.cleanup tore down another model or weight manager",
        )
        expected_manager.release()

    return teardown_dwdp


def _prepare(ctx: _Rank) -> None:
    """Long-lived buffers, the expert stand-in and the DWDP manager, before any model tensor."""
    from tensorrt_llm._torch.pyexecutor import dwdp as fac
    from tensorrt_llm.llmapi.llm_args import DwdpConfig
    from tensorrt_llm.mapping import Mapping

    spec, rank, device = ctx.spec, ctx.rank, ctx.device
    num_ranks = spec["num_ranks"]
    ctx.fac = fac
    ctx.meter = _DeviceMemory(device)
    ctx.segments = tuple(SegmentKind(name) for name in spec["segments"].split(","))
    ctx.execution = torch.cuda.Stream(device)
    _warm_up_blas(device, [torch.cuda.current_stream(device), ctx.execution])

    rng = random.Random(f"{spec['seed']}/{rank}")
    ctx.iterations = _plan_iterations(rng, spec["iterations"], anomalies=ctx.stress)
    gen = torch.Generator(device=device).manual_seed(1000 * spec["seed"] + rank)
    shape = (len(ctx.iterations), MAX_TOKENS, HIDDEN)
    ctx.inputs = torch.randn(shape, generator=gen, device=device, dtype=F32).to(BF16)
    ctx.references = torch.empty(shape, dtype=BF16, device=device)
    ctx.outputs = torch.empty(shape, dtype=BF16, device=device)
    ctx.outputs.view(torch.int16).fill_(-1)  # NaN: an output never written cannot match
    probe_shape = (len(ctx.iterations), NUM_LAYERS, 2)
    ctx.probes = torch.full(probe_shape, float("nan"), dtype=F32, device=device)
    ctx.reference_probes = torch.full(probe_shape, float("nan"), dtype=F32, device=device)
    table_gen = torch.Generator(device=device).manual_seed(MODEL_SEED + 1)
    table_elements = (EXPERT_BYTES + (len(COVERED) - 1) * EXPERT_STRIDE) // 2
    table = torch.randn(table_elements, generator=table_gen, device=device, dtype=F32)
    table = (table / math.sqrt(LATENT)).to(BF16)
    ctx.cap = _Captures(copies=[], failures=ctx.report.failures, rank=rank)
    ctx.wm = _ToyExpertManager(COVERED, device, table, random.Random(f"E/{rank}"), ctx.cap.copies)

    fac.teardown_dwdp = _toy_teardown_dwdp(ctx)
    config = DwdpConfig(
        dwdp_size=num_ranks, num_groups=1, num_experts_per_worker=4, num_prefetch_experts=4
    )
    mapping = Mapping(world_size=1, rank=0, tp_size=1, dwdp_size=num_ranks, dwdp_rank=rank)
    ctx.manager = fac.DwdpManager(config=config, dist=None, mapping=mapping)


def _build_model(ctx: _Rank) -> None:
    """The toy model, its references, and every kernel setup and the loop use, warmed up."""
    manager = ctx.fac.get_global_dwdp_manager()
    ctx.report.check(manager is ctx.manager, "the DWDP manager is not registered globally")
    streams = [torch.cuda.current_stream(ctx.device), ctx.execution, ctx.wm.copy_stream]
    torch.cuda.empty_cache()
    ctx.fillers += _seal_allocator(ctx.device, streams)
    weights, fillers = _toy_weights(ctx.device, ctx.segments, streams)
    ctx.fillers += fillers
    ctx.model = ToyCausalLM(weights, manager, ctx.wm.replicated)
    del weights
    ctx.report.check(
        sorted(manager._registered_layers) == list(COVERED),
        f"the MoE blocks registered {manager._registered_layers}",
    )
    _compute_references(ctx)
    _warm_up_kernels(ctx.device)


def _run_rank(report: _Report, spec: dict[str, Any], rank: int, device: torch.device) -> None:
    from torch.distributed import TCPStore

    tcp = TCPStore(
        host_name=_MASTER_ADDR,
        port=spec["test_port"],
        is_master=rank == 0,
        timeout=timedelta(seconds=_COLLECTIVE_TIMEOUT_S),
        wait_for_workers=False,
    )
    group = StoreDwdpGroup(
        global_rank=rank,
        dwdp_size=spec["num_ranks"],
        master_addr=_MASTER_ADDR,
        master_port=spec["test_port"],
        timeout_s=_COLLECTIVE_TIMEOUT_S,
        store=tcp,
    )
    try:
        problems = group.allgather(_capability_problem(device.index))
        problems = [f"rank {peer}: {problem}" for peer, problem in enumerate(problems) if problem]
        if problems:
            raise _Skip("; ".join(problems))
        ctx = _Rank(report, spec, rank, device, group)
        _prepare(ctx)
        with ctx.manager:
            _build_model(ctx)
            verdict, runtime, memory, facts = _setup(ctx)
            if verdict == "committed":
                with report.step("forward loop"):
                    _run_loop(ctx, runtime)
                _teardown(ctx, runtime, memory)
                _check_detached(ctx, facts)
            else:
                report.check(not ctx.stress, f"setup verdict {verdict} without an injected fault")
                if not ctx.stress:
                    with report.step("forward loop without dense DWDP"):
                        _run_loop(ctx, None)
                _teardown(ctx, None, memory)
                _check_stays(ctx, facts, "after teardown")
    finally:
        # Last man out: rank 0 hosts the test store, so it leaves only after every rank is done.
        try:
            tcp.set(f"test_done/{rank}", b"1")
            if rank == 0:
                keys = [f"test_done/{peer}" for peer in range(spec["num_ranks"])]
                tcp.wait(keys, timedelta(seconds=_COLLECTIVE_TIMEOUT_S))
        except RuntimeError as exc:  # the store's own errors; the test reports the rest
            report.notes.append(f"last man out failed: {exc!r}")


def _worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="dense DWDP end-to-end worker (one rank)")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    _die_with_parent(spec["parent_pid"])
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    report = _Report()
    try:
        if not torch.cuda.is_available():
            raise _Skip("CUDA is not available in the worker")
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        torch.zeros(1, device=device)
        _run_rank(report, spec, args.rank, device)
    except _Skip as exc:
        report.skip = str(exc)
    except Exception:  # the worker's top level: any error becomes a reported failure
        report.failures.append(f"the worker raised:\n{traceback.format_exc()}")
    finally:
        out = Path(args.out)
        partial = out.with_suffix(".partial")
        partial.write_text(json.dumps(report.to_json(), indent=2, default=str))
        os.replace(partial, out)
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
