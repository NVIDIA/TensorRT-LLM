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
"""Kimi K3 routed experts for decode: ``trtllm::k3_moe``, the persistent CuTe DSL kernel ``k3_moe``
(``k3_moe_kernel.py``): this rank's (expert, token) groups in its prologue, then FC1 + SiTU + FC2 with the
routing-weighted, deterministic combine, over the TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers read in place.

Its inputs are the routing and the MXFP8 latent of ``trtllm::k3_route_quant`` (the routing the TRTLLM-Gen path uses
under separated routing: sigmoid, top-16 of sigmoid + bias, unbiased scores renormalized times the routed scaling
factor, ties to the lower id; the CuTe DSL form of ``trtllm::kimi_k3_noaux_tc_mxfp8_quant`` with the same outputs bit
for bit) or of ``trtllm::k3_moe_front`` (head GEMV, head all-gather, routing, MXFP8 latent, shared gate_up + SiTU),
and it launches as their programmatic dependent. The result is this rank's routed partial ``[M, 3584]`` bf16, the
tensor the TRTLLM-Gen W4A8_MXFP4_MXFP8 op returns, so the routed-latent all-reduce and the latent-up tail are
unchanged.

Two builds: up to 8 tokens (:class:`K3MoeState`, optionally acquiring the front's outputs through the head
workspace's ready words) and up to 64 (:class:`K3MoeWideState`). Given a ``K3LatentExchange``, the 8-token build
pushes the partial into every rank's exchange for ``trtllm::k3_latent_reduce`` instead of returning it
(:meth:`K3MoeLayer.push`). The caller owns all state: the scratch a state's
layers share (the intermediate slab, left armed by every call, and the FC2 partial rows), each layer's counters
(:class:`K3MoeLayer`, left zero by every call), and the head all-gather buffers of the front
(:class:`K3MoeHeadWorkspace`, collective over the TP group). Build them before CUDA-graph capture; each build
compiles on its first call, which must also come before capture.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

HIDDEN_SIZE = 3584
NUM_EXPERTS = 896
TOP_K = 16
MAX_TOKENS = 8
_TOKEN_SLOTS = 8
_SF_VEC = 32
EMPTY_WORD = -(2**31)

_KERNEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_moe_kernel.py")
_modules: Dict[tuple, object] = {}


def _kernel_module(config: dict):
    """One kernel module per configuration: shapes are trace-time constants."""
    key = tuple(sorted(config.items()))
    mod = _modules.get(key)
    if mod is None:
        tag = "_".join(f"{k}{v}" for k, v in key)
        name = f"{__name__}_kernel_{tag}"
        spec = importlib.util.spec_from_file_location(name, _KERNEL_PATH)
        mod = importlib.util.module_from_spec(spec)
        mod.K3_CONFIG = dict(config)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _modules[key] = mod
    return mod


def _view(t: torch.Tensor, align: int, leading_dim: int, element_type=None):
    from cutlass.cute.runtime import from_dlpack

    # DLPack refuses tensors that require grad.
    v = from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=leading_dim)
    if element_type is not None:
        v.element_type = element_type
    return v


def is_supported(w3_w1_weight: torch.Tensor, w3_w1_weight_scale: torch.Tensor, w2_weight: torch.Tensor,
                 w2_weight_scale: torch.Tensor, local_num_experts: int) -> Tuple[bool, str]:  # fmt: skip
    """Whether these TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers fit the fused kernels.

    Only metadata is read, so the buffers may still be on the meta device."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        return False, "needs sm_100"
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False, "CuTe DSL (nvidia-cutlass-dsl) is not installed"
    e, two_i, k_half = w3_w1_weight.shape
    i_tp = two_i // 2
    expected = {
        "w3_w1_weight": (w3_w1_weight, (local_num_experts, two_i, HIDDEN_SIZE // 2)),
        "w3_w1_weight_scale": (
            w3_w1_weight_scale,
            (local_num_experts, two_i, HIDDEN_SIZE // _SF_VEC),
        ),
        "w2_weight": (w2_weight, (local_num_experts, HIDDEN_SIZE, i_tp // 2)),
        "w2_weight_scale": (w2_weight_scale, (local_num_experts, HIDDEN_SIZE, i_tp // _SF_VEC)),
    }
    for name, (t, shape) in expected.items():
        if t.dtype != torch.uint8 or tuple(t.shape) != shape or not t.is_contiguous():
            return False, f"{name} is {t.dtype} {tuple(t.shape)}, expected contiguous uint8 {shape}"
    if i_tp % 128 != 0:
        return False, f"intermediate size {i_tp} is not a multiple of 128"
    if local_num_experts > NUM_EXPERTS:
        return False, f"{local_num_experts} local experts"
    return True, ""


def create_mcast_state(name: str, mapping, words: int, fabric_handle: Optional[bool], build):
    """``build(uc, mc, handle, comm)`` over a new multicast buffer of ``words`` int32 per rank of ``mapping``'s TP
    group, every word empty (``uc``: this rank's words; ``mc``: the same words through the multicast mapping).
    Collective over the TP group only, and eager: every rank of the group, and no other rank, calls it at the same
    point (its communicator is made from the group's ranks alone).

    Failure model (as ``MnnvlWorkspace.create``): before allocating, the ranks agree that each of them can (not
    capturing a CUDA graph, the buffer within its device's free memory); if one cannot, every rank raises
    ``RuntimeError``, none allocates, and under MPI each frees the communicator made for the call. A failure that
    returns from the allocation or from ``build`` is agreed and handled the same way. A rank that fails inside the
    allocation's handle exchange can leave its peers waiting in that exchange: that failure is not turned into an
    error on the other ranks."""
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_tp_group_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_device_index,
        _mnnvl_workspace_all_succeeded,
    )
    from tensorrt_llm._utils import mpi_disabled

    use_fabric_handle = mapping.is_multi_node() if fabric_handle is None else bool(fabric_handle)
    comm = _get_mnnvl_tp_group_comm(mapping)
    # Every condition one rank alone can fail is checked before the allocation, and the ranks agree on it: a rank
    # failing inside the allocation would leave its peers in the handle exchange.
    problem: Optional[str] = None
    if torch.cuda.is_current_stream_capturing():
        problem = "it is collective and allocates: call it outside CUDA-graph capture"
    else:
        free_bytes, _ = torch.cuda.mem_get_info(_mnnvl_device_index(mapping))
        if free_bytes < words * 4:
            problem = f"its {words * 4} bytes exceed the {free_bytes} free on this rank's device"
    if not _mnnvl_workspace_all_succeeded(comm, problem is None):
        # Every rank takes this path: free the MPI communicator made above (a ProcessGroup is c10d's).
        if not mpi_disabled():
            comm.Free()
        raise RuntimeError(
            f"{name}.create: not every rank can allocate ({problem or 'another rank cannot'})"
        )
    error: Optional[Exception] = None
    state = None
    try:
        handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, use_fabric_handle)
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        uc.fill_(EMPTY_WORD)
        state = build(uc, mc, handle, comm)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 -- reported to every rank below, then re-raised
        error = exc
    # Also the barrier that keeps any rank from pushing into a peer's buffer before the peer has emptied it.
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        # Every rank takes this path too. A handle only borrows the communicator and makes no MPI call when
        # destroyed, so it may outlive the communicator.
        if not mpi_disabled():
            comm.Free()
        raise RuntimeError(f"{name}: allocation failed on at least one rank") from error
    return state


@dataclass(eq=False)
class K3MoeHeadWorkspace:
    """One TP group's MoE head all-gather buffers, read and written by ``trtllm::k3_moe_front``: two alternating Lamport
    buffers of every rank's head slice per token behind one multicast mapping, then the front's router partials; the
    flag words that rotate them; and the per-token ready words a publishing front releases and the ``head_flags``
    build of ``trtllm::k3_moe`` after it acquires. Every front call on it takes the next buffer, so all of a group's
    ranks make the same front calls on it in the same order. Pass ``uc``, ``mc``, ``flags``, ``rank`` and
    ``world_size`` as the front's ``ag_uc``, ``ag_mc``, ``ag_flags``, ``ag_rank`` and ``ag_world``, and ``ready`` as
    its ``ag_ready``. Separate from the MNNVL all-reduce workspace."""

    uc: torch.Tensor
    """int32 [workspace_words(world)]: this rank's words (0x80000000 = empty)."""
    mc: torch.Tensor
    """The same words through the multicast mapping (where the peers push)."""
    flags: torch.Tensor
    """int32 [4]: [0] the buffer of the next call; [2] the ready words' epoch (advanced by k3_moe's head_flags build);
    [3] the front's sign-ins after its reads of [0], counted by the CTA that flips [0]; [1] unused."""
    ready: torch.Tensor
    """int32 [32]: [t] token t's ids and weights, [8 + t] its MXFP8 row, released as ``epoch + 1``."""
    rank: int
    world_size: int
    handle: Any
    """The ``McastGPUBuffer`` that owns the memory; the workspace is valid while this object lives."""
    comm: Any
    """The TP-group communicator the handles were exchanged over."""

    @classmethod
    def create(cls, mapping, fabric_handle: Optional[bool] = None) -> "K3MoeHeadWorkspace":
        """Allocate and arm a workspace for ``mapping``'s TP group. Collective: every rank of the group calls it at the
        same point, eagerly (not under CUDA-graph capture); the failure model is :func:`create_mcast_state`'s.
        ``fabric_handle``: share the memory by fabric handle (required across nodes) rather than POSIX file
        descriptor; default ``mapping.is_multi_node()``."""
        from . import k3_route_quant_ag as layout

        def build(uc, mc, handle, comm):
            return cls(
                uc=uc,
                mc=mc,
                flags=torch.zeros(4, dtype=torch.int32, device=uc.device),
                ready=torch.zeros(32, dtype=torch.int32, device=uc.device),
                rank=mapping.tp_rank,
                world_size=mapping.tp_size,
                handle=handle,
                comm=comm,
            )

        words = layout.workspace_words(mapping.tp_size)
        return create_mcast_state("K3MoeHeadWorkspace", mapping, words, fabric_handle, build)


WIDE_MAX_TOKENS = 64

# The kernel's tensor arguments after the 18 it always reads: the fused all-reduce's buffers (3), the fold's inputs
# (5), the head flags build's ready words and head flags (2), the latent slab (1). The builds this module compiles
# read only the ready words and head flags (head_flags) and the fused all-reduce's buffers in its push-only mode (the
# push build), so the others are given a stand-in they never touch.
_OPTIONAL_ARGS = 11
# int32 words of one slot of a K3LatentExchange: two halves of 8 tokens' bf16 [3584] rows.
_EXCHANGE_SLOT_WORDS = 2 * _TOKEN_SLOTS * (HIDDEN_SIZE // 2)
# (alignment, leading dim) of every tensor argument, in the kernel's order.
_ALIGNS = [16, 16, 16, 16, 16, 4, 16, 16, 16, 16, 16, 16, 16, 16, 16, 4, 4, 4]
_ALIGNS += [16] * _OPTIONAL_ARGS
_LEADING = [0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
_LEADING += [0] * _OPTIONAL_ARGS
_compiled: Dict[tuple, object] = {}


def _part_rows(mod) -> int:
    """Rows of the FC2 partial buffer: the M <= 8 build's slices fit in its groups' rows, the wide build's are
    PART_ROWS."""
    return mod.PART_ROWS if mod.WIDE else mod.G_CAP * _TOKEN_SLOTS


def _config(
    i_tp: int,
    num_ctas: int,
    num_local: int,
    m_max: int,
    use_pdl: bool,
    head_flags: bool,
    push_world: int = 0,
) -> dict:
    """The kernel options of one build (trace-time constants). ``push_world``: the push build for a latent exchange of
    that many slots (the fused all-reduce's push-only mode)."""
    config = {
        "i_tp": i_tp,
        "num_ctas": num_ctas,
        "num_local": num_local,
        "m_max": m_max,
        "pdl": int(use_pdl),
        "head_flags": int(head_flags),
        "lat_slab": 0,
    }
    if push_world:
        config.update(ar_world=push_world, ar_push_only=1)
    return config


class _K3MoeScratch:
    """The scratch of one ``k3_moe`` build on one device, shared by the layers that run on it: the FC1 -> FC2
    intermediate slab (armed between calls: FP8 -0.0 values, E8M0 NaN scale words) and the FC2 partial rows."""

    def __init__(
        self,
        device: torch.device,
        i_tp: int,
        num_local: int,
        m_max: int,
        use_pdl: bool,
        head_flags: bool,
        num_ctas: Optional[int],
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{type(self).__name__} allocates its scratch: build it outside CUDA-graph capture"
            )
        # One persistent CTA per SM (num_ctas caps it, e.g. for a grid-size A/B).
        if num_ctas is None:
            num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        self.config = _config(i_tp, num_ctas, num_local, m_max, use_pdl, head_flags)
        self.mod = mod = _kernel_module(self.config)
        self.device = device
        self.i_tp = i_tp
        self.num_local = num_local
        self.m_max = m_max
        self.num_ctas = num_ctas
        self.use_pdl = bool(use_pdl)
        self.head_flags = bool(head_flags)
        g_cap = mod.G_CAP
        kw = dict(device=device)
        # Lamport slab, armed: FP8 -0.0 values; E8M0 NaN in bytes 0..3 of each 16-byte scale group. Every call
        # leaves the groups it used armed again.
        self.c = torch.full((g_cap, _TOKEN_SLOTS, i_tp), -128, dtype=torch.int8, **kw)
        self.cs = torch.zeros(g_cap, _TOKEN_SLOTS, mod.SF_STRIDE0, dtype=torch.int8, **kw)
        self.cs.view(g_cap, _TOKEN_SLOTS, mod.K2_TILES, mod.SFB_GROUP_BYTES)[..., :4] = -1
        # FC2 partial rows. A call writes the rows of its tokens; the M <= 8 build's combine also loads the rows past
        # M (their sums are dropped), so the buffer starts zeroed and those loads never read unwritten memory.
        self.part = torch.zeros(_part_rows(mod), HIDDEN_SIZE, dtype=torch.float32, **kw)

    def layer(
        self,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ) -> "K3MoeLayer":
        """A layer's handle: its experts' TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers (read in place) and its counters."""
        return K3MoeLayer(self, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)

    @property
    def compiled(self) -> bool:
        """Whether this build has been compiled on this device (by any state's first call)."""
        return _compile_key(self.device, self.config) in _compiled


class K3MoeState(_K3MoeScratch):
    """``k3_moe`` for 1..8 decode tokens on one device: the build's scratch, shared by its layers. Build it eagerly
    before CUDA-graph capture and keep it with the model; every layer takes its own counters from :meth:`layer`. The
    layers of one state run in one stream order (they share the scratch). The kernel compiles on its first call for
    the build, which must therefore come before capture.

    ``head_flags``: the build in which k3_moe acquires the MoE front's routing and MXFP8 rows through the head
    workspace's ready words instead of waiting for the front's grid (``trtllm::k3_moe_front`` with ``ag_ready``, then
    ``trtllm::k3_moe`` with the workspace's ``ready`` and ``flags``). ``use_pdl``: launch k3_moe as a programmatic
    dependent of its producer (default ``TRTLLM_ENABLE_PDL``). ``num_ctas``: the persistent grid (default one CTA per
    SM)."""

    def __init__(
        self,
        device: torch.device,
        i_tp: int,
        num_local: int,
        head_flags: bool = False,
        use_pdl: Optional[bool] = None,
        num_ctas: Optional[int] = None,
    ):
        if use_pdl is None:
            use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
        super().__init__(device, i_tp, num_local, MAX_TOKENS, use_pdl, head_flags, num_ctas)


class K3MoeWideState(_K3MoeScratch):
    """``k3_moe`` for 1..64 tokens on one device (the m_max 64 build): its scratch, sized for 64 tokens and shared by
    its layers, as :class:`K3MoeState`. ``use_pdl``: launch ``k3_moe`` as a programmatic dependent; its producer must
    then be ``trtllm::k3_route_quant`` with ``early_trigger=True`` (or any kernel whose outputs ``k3_moe`` may read
    once that grid has completed)."""

    def __init__(self, device: torch.device, i_tp: int, num_local: int, use_pdl: bool = True):
        super().__init__(device, i_tp, num_local, WIDE_MAX_TOKENS, use_pdl, False, None)


class K3MoeLayer:
    """One MoE layer on a :class:`K3MoeState` or :class:`K3MoeWideState`: its experts' weight buffers and its
    counters (int32, zero between calls; every call leaves them zero)."""

    def __init__(
        self,
        state: _K3MoeScratch,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3MoeLayer allocates its counters: build it outside CUDA-graph capture"
            )
        ok, why = is_supported(
            w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, state.num_local
        )
        if not ok or w3_w1_weight.shape[1] != 2 * state.i_tp:
            raise ValueError(f"k3_moe layer: {why or 'intermediate size differs from the state'}")
        self.state = state
        self.weights = (w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)
        self.counters = torch.zeros(state.mod.NUM_STATE, dtype=torch.int32, device=state.device)

    def __call__(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        head_ready: Optional[torch.Tensor] = None,
        head_flags: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``trtllm::k3_moe`` on this layer's experts, counters and its state's scratch: see the op. ``head_ready`` and
        ``head_flags`` are given for, and only for, the layers of a ``head_flags`` state."""
        st = self.state
        if (head_ready is not None) != st.head_flags:
            raise ValueError(
                "K3MoeLayer: head_ready / head_flags are given for, and only for, a head_flags state's layers"
            )
        y = torch.ops.trtllm.k3_moe(
            x_fp8, x_sf, topk_ids, topk_weights, *self.weights, st.c, st.cs, st.part, self.counters,
            local_expert_offset, st.num_local, st.num_ctas, st.m_max, st.use_pdl, head_ready, head_flags, out,
        )  # fmt: skip
        return y if out is None else out[: topk_ids.shape[0]]

    def push(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        exchange,
        slot: Optional[int] = None,
        head_ready: Optional[torch.Tensor] = None,
        head_flags: Optional[torch.Tensor] = None,
    ) -> None:
        """``trtllm::k3_moe``'s push build on this layer: the routed partial goes into slot ``slot`` (default
        ``exchange.rank``) of every rank's ``exchange`` (a ``K3LatentExchange``) instead of a returned tensor, for
        ``trtllm::k3_latent_reduce``. See the op. ``head_ready`` and ``head_flags`` are given for, and only for, the
        layers of a ``head_flags`` state."""
        st = self.state
        if (head_ready is not None) != st.head_flags:
            raise ValueError(
                "K3MoeLayer: head_ready / head_flags are given for, and only for, a head_flags state's layers"
            )
        torch.ops.trtllm.k3_moe(
            x_fp8, x_sf, topk_ids, topk_weights, *self.weights, st.c, st.cs, st.part, self.counters,
            local_expert_offset, st.num_local, st.num_ctas, st.m_max, st.use_pdl, head_ready, head_flags, None,
            exchange.uc, exchange.mc, exchange.flags, exchange.rank if slot is None else slot,
        )  # fmt: skip


def _compile_key(device: torch.device, config: dict) -> tuple:
    index = device.index if device.index is not None else torch.cuda.current_device()
    return (index, tuple(sorted(config.items())))


def _compile(mod, args, scalars):
    """TVM-FFI build of ``mod``'s k3_moe for these torch arguments' types and layouts (any M up to the build's)."""
    import cutlass.cute as cute

    assert len(args) == len(_ALIGNS)
    signature = [_view(t, a, d) for t, a, d in zip(args, _ALIGNS, _LEADING)]
    return cute.compile(
        mod.k3_moe,
        *signature,
        *scalars,
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@torch.library.custom_op(
    "trtllm::k3_moe",
    mutates_args=(
        "c", "cs", "part", "counters", "head_ready", "head_flags", "out", "exchange_uc", "exchange_mc",
    ),
)  # fmt: skip
def k3_moe(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    c: torch.Tensor,
    cs: torch.Tensor,
    part: torch.Tensor,
    counters: torch.Tensor,
    local_expert_offset: int,
    num_local: int,
    num_ctas: int,
    m_max: int,
    use_pdl: bool,
    head_ready: Optional[torch.Tensor] = None,
    head_flags: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    exchange_uc: Optional[torch.Tensor] = None,
    exchange_mc: Optional[torch.Tensor] = None,
    exchange_flags: Optional[torch.Tensor] = None,
    exchange_slot: int = 0,
) -> torch.Tensor:
    """This rank's routed partial ``[M, 3584]`` bf16 from the persistent ``k3_moe`` kernel: FC1 + SiTU + FC2 with the
    routing-weighted, deterministic combine over this rank's experts.

    ``x_fp8`` float8_e4m3fn ``[M, 3584]``, ``x_sf`` its E8M0 scales (``M * 112`` bytes), ``topk_ids`` int32
    ``[M, 16]`` global expert ids and ``topk_weights`` bf16 ``[M, 16]``: the outputs of ``trtllm::k3_route_quant``
    (or ``trtllm::k3_moe_front``). The weights are the TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers of ``num_local`` experts,
    which hold global ids ``[local_expert_offset, local_expert_offset + num_local)``, read in place. ``c``, ``cs``,
    ``part``: the scratch of a :class:`K3MoeState` (``m_max`` 8) or :class:`K3MoeWideState` (``m_max`` 64) of this
    build (``num_ctas``, ``use_pdl``); every call leaves the slab armed. ``counters``: the layer's, left zero.
    ``head_ready`` / ``head_flags``: a ``head_flags`` build's ready words and head flags (a ``K3MoeHeadWorkspace``'s
    ``ready`` and ``flags``), whose epoch the call advances. ``out``: bf16, contiguous, at least ``[M, 3584]``; the
    call writes its first M rows and returns an empty ``[0, 3584]`` instead of a new tensor. 1 <= M <= ``m_max``.

    ``exchange_uc`` / ``exchange_mc`` / ``exchange_flags``: a ``K3LatentExchange``'s words (this rank's, and the same
    words through the multicast mapping) and flags. With them the call is the push build (``m_max`` 8, no ``out``):
    the partial goes into slot ``exchange_slot`` of half ``exchange_flags[0] & 1`` of every rank's exchange through
    the multicast mapping (bf16 pairs, -0.0 stored as +0.0, zero rows when nothing is routed here) and the call returns
    an empty ``[0, 3584]``. Nothing reduces the slots and the flags are not written: ``trtllm::k3_latent_reduce`` sums
    the slots, empties the half it read and advances the flags. Each push of M tokens is followed by one reduce of M
    tokens on that exchange before the next push."""
    num_tokens = topk_ids.shape[0]
    head = head_ready is not None
    if head != (head_flags is not None):
        raise ValueError("k3_moe: head_ready and head_flags go together")
    push = exchange_uc is not None
    if push != (exchange_mc is not None) or push != (exchange_flags is not None):
        raise ValueError("k3_moe: exchange_uc, exchange_mc and exchange_flags go together")
    push_world = 0
    if push:
        push_world = exchange_uc.numel() // _EXCHANGE_SLOT_WORDS
        if (
            exchange_uc.dtype != torch.int32
            or exchange_mc.dtype != torch.int32
            or push_world == 0
            or exchange_uc.numel() != push_world * _EXCHANGE_SLOT_WORDS
            or exchange_mc.numel() != exchange_uc.numel()
            or exchange_flags.dtype != torch.int32
            or exchange_flags.numel() < 1
        ):
            raise ValueError(
                "k3_moe: the exchange is int32 words [2][8][slots][1792] (this rank's and multicast) and int32 flags"
            )
        if not 0 <= exchange_slot < push_world:
            raise ValueError(
                f"k3_moe: slot {exchange_slot} outside the exchange's {push_world} slots"
            )
        if m_max != MAX_TOKENS or out is not None:
            raise ValueError("k3_moe: the push build is the m_max 8 build and takes no out")
    if m_max not in (MAX_TOKENS, WIDE_MAX_TOKENS) or (head and m_max != MAX_TOKENS):
        raise ValueError(
            f"k3_moe: m_max is {MAX_TOKENS} (head flags possible) or {WIDE_MAX_TOKENS}, got {m_max}"
        )
    if not 0 < num_tokens <= m_max:
        raise ValueError(f"k3_moe: M must be in [1, {m_max}], got {num_tokens}")
    if (
        topk_ids.dtype != torch.int32
        or tuple(topk_ids.shape) != (num_tokens, TOP_K)
        or topk_weights.dtype != torch.bfloat16
        or tuple(topk_weights.shape) != (num_tokens, TOP_K)
        or x_fp8.dtype != torch.float8_e4m3fn
        or tuple(x_fp8.shape) != (num_tokens, HIDDEN_SIZE)
        or x_sf.numel() != num_tokens * (HIDDEN_SIZE // _SF_VEC)
        or not (topk_ids.is_contiguous() and topk_weights.is_contiguous())
        or not (x_fp8.is_contiguous() and x_sf.is_contiguous())
    ):
        raise ValueError("k3_moe: expects trtllm::k3_route_quant's outputs for M tokens")
    ok, why = is_supported(w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, num_local)
    if not ok:
        raise ValueError(f"k3_moe: {why}")
    e, two_i, _ = w3_w1_weight.shape
    i_tp = two_i // 2
    config = _config(i_tp, num_ctas, num_local, m_max, use_pdl, head, push_world)
    mod = _kernel_module(config)
    g_cap = mod.G_CAP
    if (
        c.dtype != torch.int8
        or tuple(c.shape) != (g_cap, _TOKEN_SLOTS, i_tp)
        or cs.dtype != torch.int8
        or tuple(cs.shape) != (g_cap, _TOKEN_SLOTS, mod.SF_STRIDE0)
        or part.dtype != torch.float32
        or tuple(part.shape) != (_part_rows(mod), HIDDEN_SIZE)
        or counters.dtype != torch.int32
        or tuple(counters.shape) != (mod.NUM_STATE,)
        or not all(t.is_contiguous() for t in (c, cs, part, counters))
    ):
        raise ValueError(
            "k3_moe: c / cs / part / counters are not this build's scratch and layer counters"
        )
    if head and (
        head_ready.dtype != torch.int32
        or head_ready.numel() < 2 * MAX_TOKENS
        or head_flags.dtype != torch.int32
        or head_flags.numel() < 3
    ):
        raise ValueError(
            "k3_moe: head_ready / head_flags must be a K3MoeHeadWorkspace's ready and flags"
        )
    if out is None:
        y = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=x_fp8.device)
    else:
        if (
            out.dtype != torch.bfloat16
            or out.dim() != 2
            or out.shape[0] < num_tokens
            or out.shape[1] != HIDDEN_SIZE
            or not out.is_contiguous()
        ):
            raise ValueError("k3_moe: out must be contiguous bf16 [>= M, 3584]")
        y = out[:num_tokens]
    sfb2 = (
        cs.view(torch.uint8)
        .reshape(g_cap, _TOKEN_SLOTS, mod.K2_TILES, mod.SFB_GROUP_BYTES)
        .permute(3, 2, 1, 0)
    )
    # Stand-in for the options this build does not have (fused all-reduce without a push, fold, latent slab, and the
    # head flags without head_flags): the kernel never touches it. The push build writes no y.
    unused = counters
    flag_args = (head_ready.view(-1), head_flags.view(-1)) if head else (unused, unused)
    ar_args = (
        (exchange_uc.view(-1), exchange_mc.view(-1), exchange_flags.view(-1))
        if push
        else (unused,) * 3
    )
    args = (
        w3_w1_weight.view(torch.int8).permute(2, 1, 0),
        x_fp8.view(torch.uint8).permute(1, 0),
        w3_w1_weight_scale.view(e, two_i // 128, HIDDEN_SIZE // 128, 512).permute(3, 2, 1, 0),
        x_sf.view(torch.uint8).view(num_tokens, HIDDEN_SIZE // _SF_VEC),
        c, cs, c.view(-1).view(torch.int32), cs.view(-1).view(torch.int32),
        w2_weight.view(torch.int8).permute(2, 1, 0),
        c.view(torch.uint8).permute(2, 1, 0),
        w2_weight_scale.view(e, HIDDEN_SIZE // 128, i_tp // 128, 512).permute(3, 2, 1, 0),
        sfb2, y, y.view(torch.int32), part, topk_ids, topk_weights, counters,
        *ar_args,  # the fused all-reduce's buffers: the exchange's words and flags (push build)
        unused, unused, unused, unused, unused,  # the fold's inputs
        *flag_args,  # the ready words and the head flags
        unused,  # the latent slab
    )  # fmt: skip
    # (tokens, local offset, local experts, all-reduce rank (the push's slot), routed scaling factor (fold only), slab
    # buffer, re-arm 0)
    scalars = (num_tokens, local_expert_offset, num_local, exchange_slot if push else 0, 1.0, 0, 0)
    key = _compile_key(x_fp8.device, config)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_moe compiles on its first call for each build: call it once before CUDA-graph capture"
            )
        fn = _compiled[key] = _compile(mod, args, scalars)
    fn(*args, *scalars, torch.cuda.current_stream(x_fp8.device).cuda_stream)
    if out is not None or push:
        return y.new_empty((0, HIDDEN_SIZE))
    return y


@k3_moe.register_fake
def _(x_fp8, x_sf, topk_ids, topk_weights, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, c, cs,
      part, counters, local_expert_offset, num_local, num_ctas, m_max, use_pdl, head_ready=None, head_flags=None,
      out=None, exchange_uc=None, exchange_mc=None, exchange_flags=None, exchange_slot=0):  # fmt: skip
    rows = 0 if out is not None or exchange_uc is not None else topk_ids.shape[0]
    return x_fp8.new_empty((rows, HIDDEN_SIZE), dtype=torch.bfloat16)


# ---------------------------------------------------------------------------------------------------------------------
# trtllm::k3_moe_m1 / trtllm::k3_moe_m2: the routed experts of one or two decode tokens as weight-stream kernels
# (k3_moe_m1_kernel.py, k3_moe_m2_kernel.py), on a caller-owned workspace.
# ---------------------------------------------------------------------------------------------------------------------
_ENGINE_KERNEL_PATHS = {
    "m1": os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_moe_m1_kernel.py"),
    "m2": os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_moe_m2_kernel.py"),
}
_M1_MIN_CTAS = HIDDEN_SIZE // 32  # one 32-row block of the down projection per CTA
# (alignment, leading dim) per tensor argument of k3_moe_m1 / k3_moe_m2, in _engine_args' order.
_ENGINE_ALIGNS = [16, 16, 16, 16, 16, 4, 2, 4, 16, 16, 4, 4, 2, 16, 4]
_ENGINE_LEADING = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def _engine_module(kind: str, config: dict):
    """One ``k3_moe_m1`` / ``k3_moe_m2`` module per configuration (shapes are trace-time constants), kept with
    k3_moe's."""
    key = (kind,) + tuple(sorted(config.items()))
    mod = _modules.get(key)
    if mod is None:
        name = f"{__name__}_{kind}_kernel_" + "_".join(f"{k}{v}" for k, v in key[1:])
        spec = importlib.util.spec_from_file_location(name, _ENGINE_KERNEL_PATHS[kind])
        mod = importlib.util.module_from_spec(spec)
        mod.K3_CONFIG = dict(config)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _modules[key] = mod
    return mod


def m1_supported(
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    local_num_experts: int,
    intermediate_size: int,
) -> Tuple[bool, str]:
    """Whether these TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers fit ``k3_moe_m1``. ``intermediate_size`` is a rank's
    logical intermediate; the buffers may hold it zero-padded to a multiple of 128, as TRT-LLM's loader lays out an
    unaligned shard (e.g. 192 -> 256 at TP16). Only metadata is read, so the buffers may still be on the meta
    device."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        return False, "needs sm_100"
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False, "CuTe DSL (nvidia-cutlass-dsl) is not installed"
    e, two_i, _ = w3_w1_weight.shape
    i_pad = two_i // 2
    expected = {
        "w3_w1_weight": (w3_w1_weight, (local_num_experts, two_i, HIDDEN_SIZE // 2)),
        "w3_w1_weight_scale": (
            w3_w1_weight_scale,
            (local_num_experts, two_i, HIDDEN_SIZE // _SF_VEC),
        ),
        "w2_weight": (w2_weight, (local_num_experts, HIDDEN_SIZE, i_pad // 2)),
        "w2_weight_scale": (w2_weight_scale, (local_num_experts, HIDDEN_SIZE, i_pad // _SF_VEC)),
    }
    for name, (t, shape) in expected.items():
        if t.dtype != torch.uint8 or tuple(t.shape) != shape or not t.is_contiguous():
            return False, f"{name} is {t.dtype} {tuple(t.shape)}, expected contiguous uint8 {shape}"
    if i_pad % 128 != 0 or not 0 < intermediate_size <= i_pad or intermediate_size % 32 != 0:
        return False, f"intermediate {intermediate_size} in buffers of {i_pad}"
    if local_num_experts > NUM_EXPERTS:
        return False, f"{local_num_experts} local experts"
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if sms < _M1_MIN_CTAS:
        return False, f"{sms} SMs, the kernel needs {_M1_MIN_CTAS}"
    return True, ""


def m2_supported(
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    local_num_experts: int,
    intermediate_size: int,
) -> Tuple[bool, str]:
    """Whether these TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers fit ``k3_moe_m2``: ``k3_moe_m1``'s conditions, and an
    intermediate of at most 256 (the TP16 slice; its FC2 tiles of every expert slot stay in shared memory)."""
    ok, why = m1_supported(
        w3_w1_weight,
        w3_w1_weight_scale,
        w2_weight,
        w2_weight_scale,
        local_num_experts,
        intermediate_size,
    )
    if ok and w3_w1_weight.shape[1] // 2 > 256:
        return False, f"padded intermediate {w3_w1_weight.shape[1] // 2} > 256"
    return ok, why


def _engine_config(i_tp: int, i_pad: int, num_local: int, num_ctas: int, m_max: int, push_world: int = 0,
                   copies: int = 1) -> dict:  # fmt: skip
    """The kernel options of one ``k3_moe_m1`` / ``k3_moe_m2`` build (trace-time constants). ``push_world``: the push
    build for a latent exchange of that many slots, this rank filling ``copies`` of them."""
    config = {
        "i_tp": i_tp,
        "i_pad": i_pad,
        "num_local": num_local,
        "num_ctas": num_ctas,
        "m_max": m_max,
    }
    if push_world:
        config.update(push=1, push_world=push_world, push_copies=copies)
    return config


def _engine_args(
    weights, x_fp8, x_sf, topk_ids, topk_weights, hbuf, counts, epochs, y, lat_mc, lat_flags
) -> tuple:
    """The tensor arguments of ``k3_moe_m1`` / ``k3_moe_m2`` in their order: the weight / activation views, then the
    flat arrays (ids, weights, w2 scale words, intermediate rows and their words, counts, epochs, output, the latent
    exchange's multicast words and flags)."""
    w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale = weights
    e, two_i, _ = w3_w1_weight.shape
    m = topk_ids.shape[0]
    return (
        w3_w1_weight.view(torch.int8).permute(2, 1, 0), x_fp8.view(torch.uint8).permute(1, 0),
        w3_w1_weight_scale.view(e, two_i // 128, HIDDEN_SIZE // 128, 512).permute(3, 2, 1, 0),
        x_sf.view(torch.uint8).view(m, HIDDEN_SIZE // _SF_VEC), w2_weight.view(torch.int8).permute(2, 1, 0),
        topk_ids.view(-1), topk_weights.view(torch.int16).view(-1), w2_weight_scale.view(-1).view(torch.int32), hbuf,
        hbuf.view(torch.int32), counts, epochs, y.view(-1), lat_mc.view(-1), lat_flags.view(-1),
    )  # fmt: skip


def _engine_scalars(kind: str, local_expert_offset: int, num_tokens: int, lat_rank: int) -> tuple:
    if kind == "m1":
        return (local_expert_offset, lat_rank)
    return (local_expert_offset, num_tokens, lat_rank)


def _engine_compile(kind: str, mod, args, scalars):
    """TVM-FFI build of ``mod``'s kernel for these torch arguments' types and layouts."""
    import cutlass.cute as cute

    assert len(args) == len(_ENGINE_ALIGNS)
    signature = [_view(t, a, d) for t, a, d in zip(args, _ENGINE_ALIGNS, _ENGINE_LEADING)]
    return cute.compile(
        getattr(mod, f"k3_moe_{kind}"),
        *signature,
        *scalars,
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _engine_workspace_sizes(kind: str, mod, m: int) -> Tuple[int, int]:
    """(intermediate row bytes, count words) of one build's workspace for ``m`` tokens."""
    if kind == "m1":
        return mod.G_CAP * m * mod.H_ROW, 4
    return mod.G_CAP * mod.M_MAX * mod.H_ROW, 2 * mod.GROUPS2 * mod.CW


def _engine_call(kind, x_fp8, x_sf, topk_ids, topk_weights, weights, hbuf, counts, epochs, local_expert_offset,
                 intermediate_size, lat_mc, lat_flags, lat_rank, copies, out, compile_num_local=0):  # fmt: skip
    """The body of ``trtllm::k3_moe_m1`` / ``trtllm::k3_moe_m2``: checks, then the build's launch (compiled on its
    first call). ``compile_num_local``: compile the build for ``compile_num_local`` local experts and launch nothing
    (``weights`` may then be stand-ins of any expert count)."""
    name = f"k3_moe_{kind}"
    m = topk_ids.shape[0]
    if (m not in (1, 2)) if kind == "m1" else m != 2:
        raise ValueError(
            f"{name}: {'1 or 2 tokens' if kind == 'm1' else '2 tokens'} per call, got {m}"
        )
    if (
        topk_ids.dtype != torch.int32
        or tuple(topk_ids.shape) != (m, TOP_K)
        or topk_weights.dtype != torch.bfloat16
        or tuple(topk_weights.shape) != (m, TOP_K)
        or x_fp8.dtype != torch.float8_e4m3fn
        or tuple(x_fp8.shape) != (m, HIDDEN_SIZE)
        or x_sf.numel() != m * (HIDDEN_SIZE // _SF_VEC)
        or not (topk_ids.is_contiguous() and topk_weights.is_contiguous())
        or not (x_fp8.is_contiguous() and x_sf.is_contiguous())
    ):
        raise ValueError(f"{name}: expects {m} token(s)' routing and MXFP8 latents")
    num_local = compile_num_local or weights[0].shape[0]
    if not compile_num_local:
        supported = m1_supported if kind == "m1" else m2_supported
        ok, why = supported(*weights, num_local, intermediate_size)
        if not ok:
            raise ValueError(f"{name}: {why}")
    push = lat_mc is not None
    if push != (lat_flags is not None):
        raise ValueError(f"{name}: exchange_mc and exchange_flags go together")
    slots = 0
    if push:
        slots = lat_mc.numel() // _EXCHANGE_SLOT_WORDS
        if (
            lat_mc.dtype != torch.int32
            or slots == 0
            or lat_mc.numel() != slots * _EXCHANGE_SLOT_WORDS
            or lat_flags.dtype != torch.int32
            or lat_flags.numel() < 1
        ):
            raise ValueError(
                f"{name}: the exchange is int32 words [2][8][slots][1792] and int32 flags"
            )
        if copies < 1 or not 0 <= lat_rank * copies <= slots - copies:
            raise ValueError(
                f"{name}: slots {lat_rank * copies}..+{copies} outside the exchange's {slots} slots"
            )
        if out is not None:
            raise ValueError(f"{name}: the push build takes no out")
    num_ctas = epochs.numel()
    i_pad = weights[0].shape[1] // 2
    config = _engine_config(
        intermediate_size, i_pad, num_local, num_ctas, m if kind == "m1" else 2, slots, copies
    )
    mod = _engine_module(kind, config)
    hbuf_bytes, count_words = _engine_workspace_sizes(kind, mod, m)
    if (
        num_ctas < _M1_MIN_CTAS
        or epochs.dtype != torch.int32
        or hbuf.dtype != torch.int8
        or hbuf.numel() != hbuf_bytes
        or counts.dtype != torch.int32
        or counts.numel() != count_words
        or not all(t.is_contiguous() for t in (hbuf, counts, epochs))
    ):
        raise ValueError(f"{name}: hbuf / counts / epochs are not this build's workspace")
    if out is None:
        y = torch.empty(m, HIDDEN_SIZE, dtype=torch.bfloat16, device=x_fp8.device)
    else:
        if (
            out.dtype != torch.bfloat16
            or tuple(out.shape) != (m, HIDDEN_SIZE)
            or not out.is_contiguous()
        ):
            raise ValueError(f"{name}: out must be contiguous bf16 [{m}, 3584]")
        y = out
    # The plain build never touches the exchange arguments: the counts stand in for them.
    args = _engine_args(weights, x_fp8, x_sf, topk_ids, topk_weights, hbuf, counts, epochs, y,
                        lat_mc if push else counts, lat_flags if push else counts)  # fmt: skip
    scalars = _engine_scalars(kind, local_expert_offset, m, lat_rank)
    key = (kind,) + _compile_key(x_fp8.device, config)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"trtllm::{name} compiles on its first call for each build: call it before capture"
            )
        fn = _compiled[key] = _engine_compile(kind, mod, args, scalars)
    if compile_num_local:
        return None
    fn(*args, *scalars, torch.cuda.current_stream(x_fp8.device).cuda_stream)
    if out is not None or push:
        return y.new_empty((0, HIDDEN_SIZE))
    return y


@torch.library.custom_op(
    "trtllm::k3_moe_m1",
    mutates_args=("hbuf", "counts", "epochs", "exchange_mc", "out"),
)
def k3_moe_m1(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    hbuf: torch.Tensor,
    counts: torch.Tensor,
    epochs: torch.Tensor,
    local_expert_offset: int,
    intermediate_size: int,
    exchange_mc: Optional[torch.Tensor] = None,
    exchange_flags: Optional[torch.Tensor] = None,
    exchange_rank: int = 0,
    copies: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """This rank's routed partial ``[M, 3584]`` bf16 for M = 1 or 2 decode tokens from the weight-stream kernel
    ``k3_moe_m1``: FC1 + SiTU + FC2 with k3_moe's combine over this rank's experts.

    ``x_fp8`` float8_e4m3fn ``[M, 3584]``, ``x_sf`` its E8M0 scales (``M * 112`` bytes), ``topk_ids`` int32
    ``[M, 16]`` global expert ids and ``topk_weights`` bf16 ``[M, 16]``: the outputs of ``trtllm::k3_route_quant`` or
    ``trtllm::k3_moe_front``. The weights are the TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers of a rank's experts (global ids
    ``[local_expert_offset, local_expert_offset + E)``), read in place, holding an ``intermediate_size`` slice
    zero-padded to a multiple of 128. ``hbuf``, ``counts``, ``epochs``: a :class:`K3MoeM1State`'s workspace for M
    tokens; every call re-arms the count slot the next call uses and advances the epochs. ``out``: contiguous bf16
    ``[M, 3584]``; the call writes it and returns an empty ``[0, 3584]`` instead of a new tensor.

    ``exchange_mc`` / ``exchange_flags``: a ``K3LatentExchange``'s multicast words and flags. With them the call is
    the push build: the partial goes into slots ``exchange_rank * copies .. + copies - 1`` of half
    ``exchange_flags[0] & 1`` of every rank's exchange (bf16 pairs, -0.0 stored as +0.0) for
    ``trtllm::k3_latent_reduce``, and the call returns an empty ``[0, 3584]``. Each push of M tokens is followed by one
    reduce of M tokens on that exchange before the next push."""
    weights = (w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)
    return _engine_call("m1", x_fp8, x_sf, topk_ids, topk_weights, weights, hbuf, counts, epochs,
                        local_expert_offset, intermediate_size, exchange_mc, exchange_flags, exchange_rank, copies,
                        out)  # fmt: skip


@k3_moe_m1.register_fake
def _(x_fp8, x_sf, topk_ids, topk_weights, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, hbuf, counts,
      epochs, local_expert_offset, intermediate_size, exchange_mc=None, exchange_flags=None, exchange_rank=0, copies=1,
      out=None):  # fmt: skip
    rows = 0 if out is not None or exchange_mc is not None else topk_ids.shape[0]
    return x_fp8.new_empty((rows, HIDDEN_SIZE), dtype=torch.bfloat16)


@torch.library.custom_op(
    "trtllm::k3_moe_m2",
    mutates_args=("hbuf", "counts", "epochs", "exchange_mc", "out"),
)
def k3_moe_m2(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    hbuf: torch.Tensor,
    counts: torch.Tensor,
    epochs: torch.Tensor,
    local_expert_offset: int,
    intermediate_size: int,
    exchange_mc: Optional[torch.Tensor] = None,
    exchange_flags: Optional[torch.Tensor] = None,
    exchange_rank: int = 0,
    copies: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``trtllm::k3_moe_m1``'s contract for exactly two tokens, from the weight-stream kernel ``k3_moe_m2`` (an
    intermediate slice of at most 256; the FC2 tiles of every expert slot stay in shared memory) on a
    :class:`K3MoeM2State`'s workspace."""
    weights = (w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)
    return _engine_call("m2", x_fp8, x_sf, topk_ids, topk_weights, weights, hbuf, counts, epochs,
                        local_expert_offset, intermediate_size, exchange_mc, exchange_flags, exchange_rank, copies,
                        out)  # fmt: skip


@k3_moe_m2.register_fake
def _(x_fp8, x_sf, topk_ids, topk_weights, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, hbuf, counts,
      epochs, local_expert_offset, intermediate_size, exchange_mc=None, exchange_flags=None, exchange_rank=0, copies=1,
      out=None):  # fmt: skip
    rows = 0 if out is not None or exchange_mc is not None else topk_ids.shape[0]
    return x_fp8.new_empty((rows, HIDDEN_SIZE), dtype=torch.bfloat16)


class _K3MoeEngineState:
    """The workspace of ``trtllm::k3_moe_m1`` / ``trtllm::k3_moe_m2`` on one device, shared by the layers that run on
    it: the intermediate rows (``hbuf``), the FC1 -> FC2 counts (``counts``, two sets by epoch parity) and the CTAs'
    epochs (``epochs``). Every call re-arms the count set the next call uses and advances the epochs, so the
    workspace never needs a reset; its layers run in one stream order. Build it with ``create`` before CUDA-graph
    capture and keep it with the model: ``create`` compiles the plain build and the push builds it is given. A build
    it did not compile compiles on its first call, which must come before capture too."""

    kind = ""

    def __init__(
        self, device: torch.device, i_tp: int, i_pad: int, num_local: int, num_tokens: int
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{type(self).__name__} allocates its workspace: build it outside CUDA-graph capture"
            )
        num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        self.config = _engine_config(i_tp, i_pad, num_local, num_ctas, num_tokens)
        self.mod = mod = _engine_module(self.kind, self.config)
        self.device = device
        self.i_tp = i_tp
        self.i_pad = i_pad
        self.num_local = num_local
        self.num_tokens = num_tokens
        hbuf_bytes, count_words = _engine_workspace_sizes(self.kind, mod, num_tokens)
        kw = dict(device=device)
        self.hbuf = torch.zeros(hbuf_bytes, dtype=torch.int8, **kw)
        self.counts = torch.zeros(count_words, dtype=torch.int32, **kw)
        self.epochs = torch.zeros(num_ctas, dtype=torch.int32, **kw)

    def _create(self, push: Tuple[Tuple[int, int], ...]) -> "_K3MoeEngineState":
        """Compile the plain build and a push build per (exchange slots, copies) in ``push``, launching nothing: two
        experts' zero buffers and zero tokens stand in for a layer's arguments (the builds take any shapes of these
        types and layouts)."""
        kw = dict(device=self.device)
        m, i_pad = self.num_tokens, self.i_pad
        weights = (
            torch.zeros(2, 2 * i_pad, HIDDEN_SIZE // 2, dtype=torch.uint8, **kw),
            torch.zeros(2, 2 * i_pad, HIDDEN_SIZE // _SF_VEC, dtype=torch.uint8, **kw),
            torch.zeros(2, HIDDEN_SIZE, i_pad // 2, dtype=torch.uint8, **kw),
            torch.zeros(2, HIDDEN_SIZE, i_pad // _SF_VEC, dtype=torch.uint8, **kw),
        )
        tokens = (
            torch.zeros(m, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, **kw),
            torch.zeros(m, HIDDEN_SIZE // _SF_VEC, dtype=torch.uint8, **kw),
            torch.zeros(m, TOP_K, dtype=torch.int32, **kw),
            torch.zeros(m, TOP_K, dtype=torch.bfloat16, **kw),
        )
        builds = [(None, None, 1)]
        for slots, copies in push:
            lat_mc = torch.zeros(slots * _EXCHANGE_SLOT_WORDS, dtype=torch.int32, **kw)
            builds.append((lat_mc, torch.zeros(4, dtype=torch.int32, **kw), copies))
        for lat_mc, lat_flags, copies in builds:
            _engine_call(self.kind, *tokens, weights, self.hbuf, self.counts, self.epochs, 0, self.i_tp, lat_mc,
                         lat_flags, 0, copies, None, compile_num_local=self.num_local)  # fmt: skip
        return self

    @property
    def compiled(self) -> bool:
        """Whether the plain build has been compiled on this device (by any state's ``create`` or first call)."""
        return (self.kind,) + _compile_key(self.device, self.config) in _compiled

    def push_compiled(self, slots: int, copies: int = 1) -> bool:
        """Whether the push build for an exchange of ``slots`` slots, this rank filling ``copies``, is compiled."""
        config = _engine_config(self.i_tp, self.i_pad, self.num_local, self.epochs.numel(), self.config["m_max"],
                                slots, copies)  # fmt: skip
        return (self.kind,) + _compile_key(self.device, config) in _compiled


class K3MoeM1State(_K3MoeEngineState):
    """``trtllm::k3_moe_m1``'s workspace on one device for ``num_tokens`` (1 or 2) tokens per call: see
    :class:`_K3MoeEngineState`. The count set is two words (one per epoch parity)."""

    kind = "m1"

    @classmethod
    def create(
        cls,
        device: torch.device,
        i_tp: int,
        i_pad: int,
        num_local: int,
        num_tokens: int = 1,
        push: Tuple[Tuple[int, int], ...] = (),
    ) -> "K3MoeM1State":
        """The state on ``device`` with its builds compiled: the plain build, and the push build for each (exchange
        slots, copies) in ``push`` (``((tp_size, 1),)`` for a TP group's ``K3LatentExchange``). Eager: it allocates and
        compiles, so it refuses to run under CUDA-graph capture. Not collective."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3MoeM1State.create allocates and compiles: call it before CUDA-graph capture"
            )
        return cls(torch.device(device), i_tp, i_pad, num_local, num_tokens)._create(push)

    def __init__(
        self, device: torch.device, i_tp: int, i_pad: int, num_local: int, num_tokens: int = 1
    ):
        if num_tokens not in (1, 2):
            raise ValueError(f"k3_moe_m1 takes 1 or 2 tokens per call, not {num_tokens}")
        super().__init__(device, i_tp, i_pad, num_local, num_tokens)

    def layer(
        self,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ) -> "K3MoeM1Layer":
        """A layer's handle: its experts' TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers, read in place."""
        return K3MoeM1Layer(self, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)


class K3MoeM2State(_K3MoeEngineState):
    """``trtllm::k3_moe_m2``'s workspace on one device (two tokens per call): see :class:`_K3MoeEngineState`. The
    count sets are per FC1 -> FC2 group."""

    kind = "m2"
    num_tokens = 2

    @classmethod
    def create(
        cls,
        device: torch.device,
        i_tp: int,
        i_pad: int,
        num_local: int,
        push: Tuple[Tuple[int, int], ...] = (),
    ) -> "K3MoeM2State":
        """The state on ``device`` with its builds compiled: as :meth:`K3MoeM1State.create`."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3MoeM2State.create allocates and compiles: call it before CUDA-graph capture"
            )
        return cls(torch.device(device), i_tp, i_pad, num_local)._create(push)

    def __init__(self, device: torch.device, i_tp: int, i_pad: int, num_local: int):
        super().__init__(device, i_tp, i_pad, num_local, 2)

    def layer(
        self,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ) -> "K3MoeM2Layer":
        """A layer's handle: its experts' TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers, read in place."""
        return K3MoeM2Layer(self, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)


class _K3MoeEngineLayer:
    """One MoE layer on a ``k3_moe_m1`` / ``k3_moe_m2`` state: its experts' weight buffers, read in place."""

    def __init__(
        self,
        state: _K3MoeEngineState,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ):
        supported = m1_supported if state.kind == "m1" else m2_supported
        ok, why = supported(
            w3_w1_weight,
            w3_w1_weight_scale,
            w2_weight,
            w2_weight_scale,
            state.num_local,
            state.i_tp,
        )
        if not ok or w3_w1_weight.shape[1] != 2 * state.i_pad:
            raise ValueError(
                f"k3_moe_{state.kind} layer: {why or 'padded intermediate differs from the state'}"
            )
        self.state = state
        self.weights = (w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)

    def _op(self):
        return torch.ops.trtllm.k3_moe_m1 if self.state.kind == "m1" else torch.ops.trtllm.k3_moe_m2

    def __call__(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This rank's routed partial ``[M, 3584]`` bf16 for the state's M tokens (``out``, written, when given): the
        op on this layer's experts and its state's workspace. See ``trtllm::k3_moe_m1``."""
        st = self.state
        y = self._op()(x_fp8, x_sf, topk_ids, topk_weights, *self.weights, st.hbuf, st.counts, st.epochs,
                       local_expert_offset, st.i_tp, out=out)  # fmt: skip
        return y if out is None else out

    def push(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        lat_mc: torch.Tensor,
        lat_flags: torch.Tensor,
        lat_rank: int,
        copies: int = 1,
    ) -> None:
        """The push build: the same routed partial, stored into slots ``lat_rank * copies .. + copies - 1`` of every
        rank's latent exchange (``lat_mc``: the multicast int32 words of a ``K3LatentExchange``, ``lat_flags`` its
        flags) for ``trtllm::k3_latent_reduce``, instead of returned. See ``trtllm::k3_moe_m1``."""
        st = self.state
        self._op()(x_fp8, x_sf, topk_ids, topk_weights, *self.weights, st.hbuf, st.counts, st.epochs,
                   local_expert_offset, st.i_tp, lat_mc, lat_flags, lat_rank, copies)  # fmt: skip


class K3MoeM1Layer(_K3MoeEngineLayer):
    """One MoE layer on a :class:`K3MoeM1State`."""


class K3MoeM2Layer(_K3MoeEngineLayer):
    """One MoE layer on a :class:`K3MoeM2State`."""
