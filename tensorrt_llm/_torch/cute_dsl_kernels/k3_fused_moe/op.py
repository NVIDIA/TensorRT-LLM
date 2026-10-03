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
"""Kimi K3 routed experts for decode: the persistent CuTe DSL kernel ``k3_moe`` (``k3_moe_kernel.py``).

For M <= 8 tokens (:class:`K3MoeState`, :class:`K3MoeLayer`), two kernels on the current stream, no host
synchronization:

1. the routing and the MXFP8 input quantization: ``trtllm::k3_route_quant`` from the router logits and the latent
   (the routing the TRTLLM-Gen path uses under separated routing: sigmoid, top-16 of sigmoid + bias, unbiased scores
   renormalized times the routed scaling factor, ties to the lower id; the CuTe DSL form of
   ``trtllm::kimi_k3_noaux_tc_mxfp8_quant`` with the same outputs bit for bit), or ``trtllm::k3_moe_front`` from the
   MoE input (:meth:`K3MoeLayer.front`: head GEMV, head all-gather, routing, MXFP8 latent, shared gate_up + SiTU);
2. ``k3_moe``, launched as a programmatic dependent of the first: this rank's (expert, token) groups in its prologue,
   then FC1 + SiTU + FC2 with the routing-weighted, deterministic combine.

The result is this rank's routed partial ``[M, 3584]`` bf16, the tensor the TRTLLM-Gen W4A8_MXFP4_MXFP8 op returns,
so the routed-latent all-reduce and the latent-up tail are unchanged. Weights are the TRTLLM-Gen buffers, read in
place.

Steps of up to 64 tokens use :class:`K3MoeWideState` (the m_max 64 build of ``k3_moe``, launched after
``trtllm::k3_route_quant``).

The caller owns all state: the scratch a state's layers share (the intermediate slab, left armed by every call, and
the FC2 partial rows), each layer's counters (left zero by every call), and the head all-gather buffers of the front
(:class:`K3MoeHeadWorkspace`, collective over the TP group). Build them before CUDA-graph capture; each kernel compiles
on its first call, which must also come before capture.
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

    v = from_dlpack(t, assumed_align=align).mark_layout_dynamic(leading_dim=leading_dim)
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


@dataclass(eq=False)
class K3MoeHeadWorkspace:
    """One TP group's MoE head all-gather buffers, read and written by ``trtllm::k3_moe_front`` (alone, or as the
    producer of :meth:`K3MoeLayer.front`): two alternating Lamport buffers of every rank's head slice per token behind
    one multicast mapping, then the front's router partials; the flag words that rotate them; and the per-token ready
    words a ``head_flags`` build of k3_moe acquires. Every front call on it takes the next buffer, so all of a group's
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
        same point, eagerly (not under CUDA-graph capture); it returns on every rank or raises on every rank.
        ``fabric_handle``: share the memory by fabric handle (required across nodes) rather than POSIX file
        descriptor; default ``mapping.is_multi_node()``."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3MoeHeadWorkspace.create is collective and allocates: call it outside CUDA-graph capture"
            )
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        from . import k3_route_quant_ag as layout

        words = layout.workspace_words(mapping.tp_size)
        use_fabric_handle = (
            mapping.is_multi_node() if fabric_handle is None else bool(fabric_handle)
        )
        comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[Exception] = None
        workspace = None
        try:
            handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, use_fabric_handle)
            uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
            mc = handle.get_mc_buffer((words,), torch.int32, 0)
            uc.fill_(EMPTY_WORD)
            flags = torch.zeros(4, dtype=torch.int32, device=uc.device)
            ready = torch.zeros(32, dtype=torch.int32, device=uc.device)
            torch.cuda.synchronize()
            workspace = cls(
                uc=uc,
                mc=mc,
                flags=flags,
                ready=ready,
                rank=mapping.tp_rank,
                world_size=mapping.tp_size,
                handle=handle,
                comm=comm,
            )
        except Exception as exc:  # noqa: BLE001 -- reported to every rank below, then re-raised
            error = exc
        # Also the barrier that keeps any rank from pushing into a peer's buffer before the peer has emptied it.
        if not _mnnvl_workspace_all_succeeded(comm, error is None):
            raise RuntimeError(
                "K3MoeHeadWorkspace: allocation failed on at least one rank"
            ) from error
        return workspace


class K3MoeState:
    """``k3_moe`` for 1..8 decode tokens on one device: its build and the scratch its layers share, i.e. the FC1 ->
    FC2 intermediate slab (armed between calls: FP8 -0.0 values, E8M0 NaN scale words) and the FC2 partial rows. Build
    it eagerly before CUDA-graph capture and keep it with the model; every layer takes its own counters from
    :meth:`layer`. The layers of one state run in one stream order (they share the scratch). The kernel compiles on the
    first call, which must therefore come before capture.

    ``head_flags``: the build in which k3_moe acquires the front's routing and MXFP8 rows through the head workspace's
    ready words instead of waiting for the front's grid (:meth:`K3MoeLayer.front` only). ``config`` overrides kernel
    options (tests and A/B runs: ``pdl``, ``num_ctas``, ...); anything it leaves out takes the kernel's default."""

    def __init__(
        self,
        device: torch.device,
        i_tp: int,
        num_local: int,
        head_flags: bool = False,
        config: Optional[dict] = None,
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3MoeState allocates its scratch: build it outside CUDA-graph capture"
            )
        # One persistent CTA per SM (config "num_ctas" caps it, e.g. for a grid-size A/B).
        num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        cfg = {
            "i_tp": i_tp,
            "num_ctas": num_ctas,
            "num_local": num_local,
            "head_flags": int(head_flags),
            "lat_slab": 0,
        }
        cfg.update(config or {})
        self.mod = mod = _kernel_module(cfg)
        if mod.FUSED_AR or mod.FOLD or mod.LAT_SLAB or mod.WIDE:
            raise ValueError(
                "K3MoeState is the M <= 8 build without the fused all-reduce, the fold or the slab"
            )
        self.device = device
        self.i_tp = i_tp
        self.num_local = num_local
        self.head_flags = mod.HEAD_FLAGS
        g_cap = mod.G_CAP
        kw = dict(device=device)
        # Lamport slab, armed: FP8 -0.0 values; E8M0 NaN in bytes 0..3 of each 16-byte scale
        # group. Every call leaves the groups it used armed again.
        self.c = torch.full((g_cap, _TOKEN_SLOTS, i_tp), -128, dtype=torch.int8, **kw)
        self.cs = torch.zeros(g_cap, _TOKEN_SLOTS, mod.SF_STRIDE0, dtype=torch.int8, **kw)
        self.cs.view(g_cap, _TOKEN_SLOTS, mod.K2_TILES, mod.SFB_GROUP_BYTES)[..., :4] = -1
        # FC2 partial rows. A call writes the rows of its M tokens; the combine also loads the rows past M (their sums
        # are dropped), so the buffer starts zeroed and those loads never read unwritten memory.
        self.part = torch.zeros(g_cap * _TOKEN_SLOTS, HIDDEN_SIZE, dtype=torch.float32, **kw)
        self.c_t = _view(self.c, 16, 2)
        self.cs_t = _view(self.cs, 4, 2)
        self.c_words_t = _view(self.c.view(-1).view(torch.int32), 16, 0)
        self.cs_words_t = _view(self.cs.view(-1).view(torch.int32), 16, 0)
        self.b2_t = _view(self.c.permute(2, 1, 0), 16, 0, mod.b_dtype)
        sfb2 = (
            self.cs.view(torch.uint8)
            .reshape(g_cap, _TOKEN_SLOTS, mod.K2_TILES, mod.SFB_GROUP_BYTES)
            .permute(3, 2, 1, 0)
        )
        self.sfb2_t = _view(sfb2, 16, 0, mod.sf_dtype)
        self.part_t = _view(self.part, 16, 1)
        # Stand-in for the buffers of the options this build does not have (fused all-reduce, fold inputs, latent
        # slab, and the ready words without head_flags).
        self.unused = torch.zeros(4, dtype=torch.int32, **kw)
        self.unused_t = _view(self.unused, 16, 0)
        # The route+quant kernel triggers k3_moe's launch right after its own grid dependency:
        # k3_moe waits for the whole route+quant grid before reading its outputs.
        self.route_kwargs = {"early_trigger": True} if mod.USE_PDL else {}
        from ..k3_route_quant import (
            op as _k3_route_quant_op,  # noqa: F401  (registers trtllm::k3_route_quant)
        )

        self.compiled = None

    def layer(
        self,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ) -> "K3MoeLayer":
        """A layer's handle: its experts' TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers (read in place) and its counters."""
        return K3MoeLayer(self, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)


class K3MoeLayer:
    """One MoE layer on a :class:`K3MoeState`: its weights as the kernel reads them and its counters (int32, zero
    between calls; every call leaves them zero)."""

    def __init__(
        self,
        state: K3MoeState,
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
        mod = state.mod
        e, two_i, _ = w3_w1_weight.shape
        i_tp = two_i // 2
        sfa1 = w3_w1_weight_scale.view(e, two_i // 128, HIDDEN_SIZE // 128, 512).permute(3, 2, 1, 0)
        sfa2 = w2_weight_scale.view(e, HIDDEN_SIZE // 128, i_tp // 128, 512).permute(3, 2, 1, 0)
        self.state = state
        self.counters = torch.zeros(mod.NUM_STATE, dtype=torch.int32, device=state.device)
        self.weights = (
            _view(w3_w1_weight.view(torch.int8).permute(2, 1, 0), 16, 0),
            _view(sfa1, 16, 0, mod.sf_dtype),
            _view(w2_weight.view(torch.int8).permute(2, 1, 0), 16, 0),
            _view(sfa2, 16, 0, mod.sf_dtype),
        )
        self.counters_t = _view(self.counters, 4, 0)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        local_expert_offset: int,
        routed_scaling_factor: float,
    ) -> torch.Tensor:
        """This rank's routed partial ``[M, 3584]`` bf16 for M <= 8 decode tokens: ``trtllm::k3_route_quant`` (the
        routing and the MXFP8 latent), then ``k3_moe`` launched as its programmatic dependent.

        ``hidden_states``: bf16 ``[M, 3584]`` latent; ``router_logits``: fp32 ``[M, 896]``;
        ``e_score_correction_bias``: fp32 ``[896]``; the layer's experts hold global ids
        ``[local_expert_offset, local_expert_offset + num_local)``."""
        st = self.state
        if st.head_flags:
            raise ValueError(
                "a head_flags build takes the front's ready words: call K3MoeLayer.front"
            )
        _check_tokens(hidden_states)
        ids, weights, x_fp8, x_sf = torch.ops.trtllm.k3_route_quant(
            router_logits.contiguous(), e_score_correction_bias, hidden_states.contiguous(),
            float(routed_scaling_factor), **st.route_kwargs,
        )  # fmt: skip
        return self._launch(ids, weights, x_fp8, x_sf, local_expert_offset, routed_scaling_factor)

    def front(
        self,
        x: torch.Tensor,
        w_front: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        local_expert_offset: int,
        routed_scaling_factor: float,
        shared_cols: int,
        gate_cap: float,
        linear_cap: float,
        head: K3MoeHeadWorkspace,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``trtllm::k3_moe_front`` (head GEMV, head all-gather over ``head``, routing, MXFP8 latent, shared gate_up +
        SiTU) then ``k3_moe`` for the MoE input ``x`` bf16 ``[M, 7168]`` (the same on every rank). Returns ``(y [M,
        3584] bf16, shared activation [M, shared_cols] bf16)``. A ``head_flags`` build acquires the front's routing and
        MXFP8 rows through ``head.ready`` instead of waiting for the front's grid, whose shared tiles may still be
        running."""
        from . import front_op  # noqa: F401  (registers trtllm::k3_moe_front)

        st = self.state
        _check_tokens(x)
        ready = head.ready if st.head_flags else None
        ids, weights, x_fp8, x_sf, shared = torch.ops.trtllm.k3_moe_front(
            x.contiguous(), w_front, e_score_correction_bias, float(routed_scaling_factor), shared_cols, gate_cap,
            linear_cap, head.uc, head.mc, head.flags, head.rank, head.world_size, ag_ready=ready,
        )  # fmt: skip
        flag_in = None
        if st.head_flags:
            flag_in = (_view(head.ready.view(-1), 16, 0), _view(head.flags.view(-1), 16, 0))
        y = self._launch(
            ids, weights, x_fp8, x_sf, local_expert_offset, routed_scaling_factor, flag_in
        )
        return y, shared

    def _launch(self, ids, weights, x_fp8, x_sf, local_offset, scale, flag_in=None) -> torch.Tensor:
        import cuda.bindings.driver as cuda_driver
        import cutlass.cute as cute

        st = self.state
        mod = st.mod
        num_tokens = ids.shape[0]
        a1, sfa1, a2, sfa2 = self.weights
        y = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=x_fp8.device)
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        b1 = _view(x_fp8.view(torch.uint8).permute(1, 0), 16, 0, mod.b_dtype)
        sfb1 = _view(
            x_sf.view(torch.uint8).view(num_tokens, HIDDEN_SIZE // _SF_VEC), 16, 1, mod.sf_dtype
        )
        u = st.unused_t
        args = [
            a1, b1, sfa1, sfb1, st.c_t, st.cs_t, st.c_words_t, st.cs_words_t, a2, st.b2_t, sfa2, st.sfb2_t,
            _view(y, 16, 1), _view(y.view(torch.int32), 16, 1), st.part_t, _view(ids, 4, 1), _view(weights, 4, 1),
            self.counters_t,
            u, u, u,  # the fused all-reduce's buffers
            u, u, u, u, u,  # the fold's inputs
            *(flag_in or (u, u)),  # the ready words and the head flags (head_flags)
            u,  # the latent slab
        ]  # fmt: skip
        scalars = (num_tokens, local_offset, st.num_local, 0, float(scale), 0, 0)
        if st.compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "k3_moe compiles on its first call: call it once before CUDA-graph capture"
                )
            st.compiled = cute.compile(mod.k3_moe, *args, *scalars, stream)
        st.compiled(*args, *scalars, stream)
        return y


def _check_tokens(x: torch.Tensor) -> None:
    if not 0 < x.shape[0] <= MAX_TOKENS:
        raise ValueError(f"k3_moe handles 1 to {MAX_TOKENS} tokens, got {x.shape[0]}")


# ---------------------------------------------------------------------------
# Steps of up to 64 tokens (e.g. R x 8 speculative verify tokens): the m_max 64 build of
# ``k3_moe``, one launch per call, after ``trtllm::k3_route_quant``. Its state belongs to the
# caller: nothing here is cached per process beyond the kernel module of the configuration.

WIDE_MAX_TOKENS = 64


class K3MoeWideState:
    """``k3_moe`` for 1..64 tokens on one device: the compiled kernel and the scratch its layers
    share, i.e. the FC1 -> FC2 intermediate slab (armed between calls: FP8 -0.0 values, E8M0 NaN
    scale words) and the FC2 slice partials, all sized for 64 tokens and kept at fixed addresses.
    Build it eagerly before CUDA-graph capture and keep it with the model; every layer takes its
    own counters from :meth:`layer`. The layers of one state run in one stream order (they share
    the scratch). The kernel is compiled (TVM-FFI, explicit stream) by the first call, which must
    therefore come before capture.

    ``use_pdl``: launch ``k3_moe`` as a programmatic dependent; its producer must then be
    ``trtllm::k3_route_quant`` with ``early_trigger=True`` (or any kernel whose outputs ``k3_moe``
    may read once that grid has completed)."""

    def __init__(self, device: torch.device, i_tp: int, num_local: int, use_pdl: bool = True):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3MoeWideState allocates its scratch: build it outside CUDA-graph capture"
            )
        num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        config = {
            "i_tp": i_tp,
            "num_ctas": num_ctas,
            "num_local": num_local,
            "m_max": WIDE_MAX_TOKENS,
            "pdl": int(use_pdl),
        }
        self.mod = mod = _kernel_module(config)
        self.device = device
        self.i_tp = i_tp
        self.num_local = num_local
        g_cap = mod.G_CAP
        kw = dict(device=device)
        self.c = torch.full((g_cap, _TOKEN_SLOTS, i_tp), -128, dtype=torch.int8, **kw)
        self.cs = torch.zeros(g_cap, _TOKEN_SLOTS, mod.SF_STRIDE0, dtype=torch.int8, **kw)
        self.cs.view(g_cap, _TOKEN_SLOTS, mod.K2_TILES, mod.SFB_GROUP_BYTES)[..., :4] = -1
        self.part = torch.empty(mod.PART_ROWS, HIDDEN_SIZE, dtype=torch.float32, **kw)
        # Stand-in for the buffers of the options this build does not have (fused all-reduce,
        # fold, head flags, latent slab).
        self.unused = torch.zeros(4, dtype=torch.int32, **kw)
        # The kernel's views of the scratch, in its argument order (FP8 / E8M0 data as bytes).
        sfb2 = (
            self.cs.view(torch.uint8)
            .reshape(g_cap, _TOKEN_SLOTS, mod.K2_TILES, mod.SFB_GROUP_BYTES)
            .permute(3, 2, 1, 0)
        )
        self.scratch = (
            self.c,
            self.cs,
            self.c.view(-1).view(torch.int32),
            self.cs.view(-1).view(torch.int32),
            self.c.view(torch.uint8).permute(2, 1, 0),
            sfb2,
            self.part,
        )
        self.compiled = None

    def layer(
        self,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ) -> "K3MoeWideLayer":
        """A layer's handle: its experts' TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers (read in place) and
        its counters."""
        return K3MoeWideLayer(self, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)

    def _compile(self, args, scalars):
        """TVM-FFI build for these torch arguments' types and layouts (any M up to 64)."""
        import cutlass.cute as cute

        # As the M <= 8 op's views: (alignment, leading dim) per tensor argument; the 11 stand-ins last.
        aligns = [16, 16, 16, 16, 16, 4, 16, 16, 16, 16, 16, 16, 16, 16, 16, 4, 4, 4] + [16] * 11
        leading = [0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0] + [0] * 11
        assert len(args) == len(aligns)
        signature = [_view(t, a, d) for t, a, d in zip(args, aligns, leading)]
        return cute.compile(
            self.mod.k3_moe,
            *signature,
            *scalars,
            cute.runtime.make_fake_stream(),
            options="--enable-tvm-ffi",
        )


class K3MoeWideLayer:
    """One MoE layer on a :class:`K3MoeWideState`: its weights as the kernel reads them and its
    counters (int32, zero between calls; every call leaves them zero)."""

    def __init__(
        self,
        state: K3MoeWideState,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3MoeWideLayer allocates its counters: build it outside CUDA-graph capture"
            )
        ok, why = is_supported(
            w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, state.num_local
        )
        if not ok or w3_w1_weight.shape[1] != 2 * state.i_tp:
            raise ValueError(
                f"k3_moe wide layer: {why or 'intermediate size differs from the state'}"
            )
        e, two_i, _ = w3_w1_weight.shape
        i_tp = two_i // 2
        self.state = state
        self.counters = torch.zeros(state.mod.NUM_STATE, dtype=torch.int32, device=state.device)
        self.weights = (
            w3_w1_weight.view(torch.int8).permute(2, 1, 0),
            w3_w1_weight_scale.view(e, two_i // 128, HIDDEN_SIZE // 128, 512).permute(3, 2, 1, 0),
            w2_weight.view(torch.int8).permute(2, 1, 0),
            w2_weight_scale.view(e, HIDDEN_SIZE // 128, i_tp // 128, 512).permute(3, 2, 1, 0),
        )

    def __call__(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This rank's routed partial ``[M, 3584]`` bf16 for the outputs of
        ``trtllm::k3_route_quant``: ``x_fp8`` float8_e4m3fn ``[M, 3584]``, ``x_sf`` its E8M0
        scales (``M * 112`` bytes), ``topk_ids`` int32 ``[M, 16]`` global expert ids,
        ``topk_weights`` bf16 ``[M, 16]``; 1 <= M <= 64. The layer's experts hold global ids
        ``[local_expert_offset, local_expert_offset + num_local)``. ``out``: bf16, contiguous, at
        least ``[M, 3584]``; its first M rows are the result (a fresh tensor without it). Writes
        the state's slab (left armed) and partials and this layer's counters (left zero)."""
        st = self.state
        num_tokens = topk_ids.shape[0]
        if not 0 < num_tokens <= WIDE_MAX_TOKENS:
            raise ValueError(f"k3_moe wide: M must be in [1, {WIDE_MAX_TOKENS}], got {num_tokens}")
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
            raise ValueError("k3_moe wide: expects trtllm::k3_route_quant's outputs for M tokens")
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
                raise ValueError("k3_moe wide: out must be contiguous bf16 [>= M, 3584]")
            y = out[:num_tokens]
        a1, sfa1, a2, sfa2 = self.weights
        c, cs, c_words, cs_words, b2, sfb2, part = st.scratch
        u = st.unused
        args = (
            a1, x_fp8.view(torch.uint8).permute(1, 0), sfa1,
            x_sf.view(torch.uint8).view(num_tokens, HIDDEN_SIZE // _SF_VEC), c, cs, c_words, cs_words, a2, b2,
            sfa2, sfb2, y, y.view(torch.int32), part, topk_ids, topk_weights, self.counters,
            u, u, u, u, u, u, u, u, u, u, u,
        )  # fmt: skip
        scalars = (num_tokens, local_expert_offset, st.num_local, 0, 1.0, 0, 0)
        if st.compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "k3_moe wide compiles on its first call: call it once before CUDA-graph capture"
                )
            st.compiled = st._compile(args, scalars)
        st.compiled(*args, *scalars, torch.cuda.current_stream().cuda_stream)
        return y
