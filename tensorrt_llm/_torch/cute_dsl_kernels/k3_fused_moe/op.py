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
"""``trtllm::k3_fused_moe``: Kimi K3 routed experts for decode (M <= 8 tokens).

Two kernels on the current stream, no host synchronization:

1. ``trtllm::k3_route_quant`` -- the routing and MXFP8 input quantization the TRTLLM-Gen
   path uses under separated routing (sigmoid, top-16 of sigmoid + bias, unbiased scores
   renormalized times the routed scaling factor; ties to the lower id), the CuTe DSL form of
   ``trtllm::kimi_k3_noaux_tc_mxfp8_quant`` with the same outputs bit for bit.
2. ``k3_moe`` -- one persistent CuTe DSL kernel, launched as a programmatic dependent of
   the first: this rank's (expert, token) groups in its prologue, then FC1 + SiTU + FC2
   with the routing-weighted, deterministic combine.

With the ``fold`` option there is one kernel: ``k3_moe`` computes the routing and the quantization in
its prologue (``trtllm::k3_route_quant``'s device code, so the same ids, weights and MXFP8 bits)
from the router logits and the latent, and its PDL predecessor is whatever produced them.

The result is this rank's routed partial ``[M, 3584]`` bf16, the tensor the TRTLLM-Gen
W4A8_MXFP4_MXFP8 op returns, so the routed-latent all-reduce and the latent-up tail are
unchanged. Weights are the TRTLLM-Gen buffers, read in place.

``trtllm::k3_fused_moe_ar`` also performs the all-reduce of that partial over the TP group
inside ``k3_moe`` (buffers from ``ar_workspace``) and returns the reduced latent.

The CuTe DSL kernel is compiled on the first call for each intermediate size (for the
model: the warmup that precedes CUDA-graph capture) and the scratch buffers are allocated
then too, so captured calls only launch. Each layer (keyed by its weight buffer) owns a
few counters that the kernel returns to zero; the intermediate slab, shared by all layers,
is left armed by every call.

Steps of up to 64 tokens use :class:`K3MoeWideState` (the m_max 64 build of ``k3_moe``,
launched after ``trtllm::k3_route_quant``), whose scratch and per-layer counters the caller
owns.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from typing import Dict, Optional, Tuple

import torch

HIDDEN_SIZE = 3584
NUM_EXPERTS = 896
TOP_K = 16
MAX_TOKENS = 8
_TOKEN_SLOTS = 8
_SF_VEC = 32

_KERNEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_moe_kernel.py")
_lock = threading.Lock()
_modules: Dict[tuple, object] = {}
_states: Dict[Tuple[int, int, int], "_K3FusedMoE"] = {}


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


class _K3FusedMoE:
    """Compiled kernel and scratch for one (device, intermediate size, local experts).

    ``config`` overrides kernel options (tests and A/B runs: ``pdl``,
    ``num_ctas``, ...); anything it leaves out takes the kernel's default."""

    def __init__(
        self, device: torch.device, i_tp: int, num_local: int, config: Optional[dict] = None
    ):
        # One persistent CTA per SM (config "num_ctas" caps it, e.g. for a grid-size A/B).
        num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        cfg = {"i_tp": i_tp, "num_ctas": num_ctas, "num_local": num_local}
        cfg.update(config or {})
        self.mod = mod = _kernel_module(cfg)
        self.device = device
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
        # Stand-ins for the all-reduce buffers of builds without the fused all-reduce, and for the
        # inputs a build does not read (fold: the top-k; otherwise the logits and the latent).
        self.no_ar = torch.zeros(4, dtype=torch.int32, **kw)
        self.no_ar_t = _view(self.no_ar, 16, 0)
        self.no_w = torch.zeros(8, dtype=torch.bfloat16, **kw)
        self.no_w_t = _view(self.no_w, 16, 0)
        self.ar_views: Dict[Tuple[int, int, int], tuple] = {}
        self.fold = mod.FOLD
        if self.fold:
            # Each CTA's MXFP8 latent rows [8 * cta, 8 * cta + M) and their linear scales.
            rows = mod.NUM_CTAS * mod.M_MAX
            self.xq = torch.empty(rows, HIDDEN_SIZE, dtype=torch.uint8, **kw)
            self.xsf = torch.empty(rows, HIDDEN_SIZE // _SF_VEC, dtype=torch.uint8, **kw)
            self.b1_t = _view(self.xq.permute(1, 0), 16, 0, mod.b_dtype)
            self.sfb1_t = _view(self.xsf, 16, 1, mod.sf_dtype)
            self.xq_words_t = _view(self.xq.view(-1).view(torch.int32), 16, 0)
            self.xsf_t = _view(self.xsf.view(-1), 16, 0)
        # The route+quant kernel triggers k3_moe's launch right after its own grid dependency:
        # k3_moe waits for the whole route+quant grid before reading its outputs.
        self.route_kwargs = {"early_trigger": True} if mod.USE_PDL else {}
        from ..k3_route_quant import op as _k3_route_quant_op  # noqa: F401

        self.route_quant = torch.ops.trtllm.k3_route_quant
        # Per layer (keyed by its w3_w1 buffer): weight views and the kernel's counters.
        self.layers: Dict[Tuple[int, int, int, int], tuple] = {}
        self.moe = None

    def _layer(self, w31, w31s, w2, w2s):
        key = (w31.data_ptr(), w31s.data_ptr(), w2.data_ptr(), w2s.data_ptr())
        layer = self.layers.get(key)
        if layer is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "trtllm::k3_fused_moe must run once per layer outside CUDA-graph capture "
                    "first (it allocates the layer's counters on the first call)."
                )
            mod = self.mod
            e, two_i, _ = w31.shape
            i_tp = two_i // 2
            sfa1 = w31s.view(e, two_i // 128, HIDDEN_SIZE // 128, 512).permute(3, 2, 1, 0)
            sfa2 = w2s.view(e, HIDDEN_SIZE // 128, i_tp // 128, 512).permute(3, 2, 1, 0)
            state = torch.zeros(mod.NUM_STATE, dtype=torch.int32, device=self.device)
            layer = (
                _view(w31.view(torch.int8).permute(2, 1, 0), 16, 0),
                _view(sfa1, 16, 0, mod.sf_dtype),
                _view(w2.view(torch.int8).permute(2, 1, 0), 16, 0),
                _view(sfa2, 16, 0, mod.sf_dtype),
                state,
                _view(state, 4, 0),
            )
            self.layers[key] = layer
        return layer

    def _flat_view(self, t: torch.Tensor):
        key = ("flat", t.data_ptr(), t.numel())
        v = self.ar_views.get(key)
        if v is None:
            v = self.ar_views[key] = _view(t.view(-1), 16, 0)
        return v

    def _ar_views(self, ar):
        if ar is None:
            return self.no_ar_t, self.no_ar_t, self.no_ar_t, 0
        uc, mc, flags, rank = ar
        key = (uc.data_ptr(), mc.data_ptr(), flags.data_ptr())
        views = self.ar_views.get(key)
        if views is None:
            views = self.ar_views[key] = (_view(uc, 16, 0), _view(mc, 16, 0), _view(flags, 4, 0))
        return (*views, rank)

    def __call__(
        self,
        x,
        router_logits,
        bias,
        w31,
        w31s,
        w2,
        w2s,
        local_offset: int,
        num_local: int,
        scale: float,
        ar: Optional[tuple] = None,
        head_ag: Optional[tuple] = None,
        head_ready: Optional[torch.Tensor] = None,
        front: Optional[tuple] = None,
        lat_slab: Optional[tuple] = None,
    ):
        """``ar``: (uc words, mc words, flags, rank) of the fused all-reduce, for a build
        with ``ar_world`` set; the result is then the all-reduced sum. With ``ar_push_only`` the
        rows only go to every rank's buffer (half ``flags[0] & 1``) and the result has no rows.

        ``head_ag``: (uc words, mc words, flags, rank) of the head all-gather's buffers; ``x``
        is then this rank's slice of the sharded MoE head (fp32 ``[M, (H + E) / world]``),
        ``router_logits`` is None, and ``trtllm::k3_route_quant_ag`` gathers, routes and
        quantizes before ``k3_moe``. ``head_ready``: route_quant_ag's ready words, for a build with
        ``head_flags`` (k3_moe then acquires them instead of waiting for that grid).

        ``front``: (front weight, shared columns, gate cap, linear cap, world) of
        ``trtllm::k3_moe_front``, with ``head_ag``; ``x`` is then the MoE input (bf16 ``[M, 7168]``,
        the same on every rank), the front stands in for the head GEMV and route_quant_ag, and
        the call returns ``(y, shared activation)``.

        ``lat_slab``: (slab, buffer, re-arm buffer 0) for a build with ``lat_slab`` (and the fused
        all-reduce): the reduced latent rows also go into ``slab`` (int32 ``[3, 8, 1792]``, all-ones
        empty) in ``buffer`` (the call's ordinal mod 3); the call re-arms the next buffer, and buffer
        0 too when the flag is set (the step's last call, whose own buffer must not be 0).
        """
        import cuda.bindings.driver as cuda_driver
        import cutlass.cute as cute

        mod = self.mod
        if (ar is not None) != mod.FUSED_AR:
            raise ValueError("the all-reduce buffers go with an ar_world build, and only with one")
        if head_ag is not None and self.fold:
            raise ValueError("the head all-gather feeds the unfolded route + quant")
        if (head_ready is not None) != mod.HEAD_FLAGS or (mod.HEAD_FLAGS and head_ag is None):
            raise ValueError(
                "the ready words go with a head_flags build and the head all-gather, and only there"
            )
        if front is not None and head_ag is None:
            raise ValueError("the front pushes its head into the head all-gather's buffers")
        if (lat_slab is not None) != mod.LAT_SLAB:
            raise ValueError("the latent slab goes with a lat_slab build, and only with one")
        lat_buf, lat_rearm0 = 0, 0
        if lat_slab is not None:
            slab, lat_buf, rearm0 = lat_slab
            if (
                slab.dtype != torch.int32
                or not slab.is_contiguous()
                or slab.numel() != mod.LAT_SLAB_BUFS * MAX_TOKENS * HIDDEN_SIZE // 2
                or not 0 <= lat_buf < mod.LAT_SLAB_BUFS
                or (rearm0 and lat_buf == 0)
            ):
                raise ValueError(
                    f"k3_moe latent slab: int32 [3, 8, 1792] contiguous, buffer in [0, 3), and not buffer 0 "
                    f"with the buffer-0 re-arm; got {tuple(slab.shape)} {slab.dtype}, buffer {lat_buf}, "
                    f"re-arm 0 {rearm0}"
                )
            lat_rearm0 = int(bool(rearm0))
        num_tokens = x.shape[0]
        shared = None
        a1, sfa1, a2, sfa2, _, state_t = self._layer(w31, w31s, w2, w2s)
        ar_uc_t, ar_mc_t, ar_flags_t, ar_rank = self._ar_views(ar)
        y = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=x.device)
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        if self.fold:
            ids_t, weights_t, b1, sfb1 = self.no_ar_t, self.no_w_t, self.b1_t, self.sfb1_t
            fold_in = [
                _view(router_logits.contiguous().view(-1), 16, 0),
                _view(bias.detach().contiguous().view(-1), 16, 0),
                _view(x.contiguous().view(-1).view(torch.int32), 16, 0),
                self.xq_words_t,
                self.xsf_t,
            ]
        else:
            if front is not None:
                # Registers trtllm::k3_moe_front.
                from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op  # noqa: F401

                w_front, shared_cols, gate_cap, linear_cap, world = front
                ids, weights, x_fp8, x_sf, shared = torch.ops.trtllm.k3_moe_front(
                    x, w_front, bias, scale, shared_cols, gate_cap, linear_cap, *head_ag, world,
                    ag_ready=head_ready,
                )  # fmt: skip
            elif head_ag is not None:
                ids, weights, x_fp8, x_sf = torch.ops.trtllm.k3_route_quant_ag(
                    x, bias, scale, *head_ag, early_trigger=mod.USE_PDL, ag_ready=head_ready
                )
            else:
                ids, weights, x_fp8, x_sf = self.route_quant(
                    router_logits, bias, x, scale, **self.route_kwargs
                )
            ids_t, weights_t = _view(ids, 4, 1), _view(weights, 4, 1)
            b1 = _view(x_fp8.view(torch.uint8).permute(1, 0), 16, 0, mod.b_dtype)
            sfb1 = _view(
                x_sf.view(torch.uint8).view(num_tokens, HIDDEN_SIZE // _SF_VEC), 16, 1, mod.sf_dtype
            )
            fold_in = [self.no_ar_t] * 5
        if mod.HEAD_FLAGS:
            flag_in = [self._flat_view(head_ready), self._flat_view(head_ag[2])]
        else:
            flag_in = [self.no_ar_t, self.no_ar_t]
        args = [
            a1, b1, sfa1, sfb1, self.c_t, self.cs_t, self.c_words_t, self.cs_words_t, a2,
            self.b2_t, sfa2, self.sfb2_t, _view(y, 16, 1), _view(y.view(torch.int32), 16, 1),
            self.part_t, ids_t, weights_t, state_t, ar_uc_t, ar_mc_t, ar_flags_t,
            *fold_in, *flag_in,
            self._flat_view(lat_slab[0]) if lat_slab is not None else self.no_ar_t,
        ]  # fmt: skip
        scalars = (num_tokens, local_offset, num_local, ar_rank, float(scale), lat_buf, lat_rearm0)
        if self.moe is None:
            self.moe = cute.compile(mod.k3_moe, *args, *scalars, stream)
        self.moe(*args, *scalars, stream)
        if mod.AR_PUSH_ONLY:
            y = y[:0]  # the rows went to every rank's buffer; the consumer reduces them
        return y if shared is None else (y, shared)


def _state(
    device: torch.device,
    i_tp: int,
    num_local: int,
    ar_world: int = 0,
    head_flags: bool = False,
    lat_slab: bool = False,
    ar_push_only: bool = False,
) -> _K3FusedMoE:
    key = (
        device.index if device.index is not None else torch.cuda.current_device(),
        i_tp,
        num_local,
        ar_world,
        head_flags,
        lat_slab,
        ar_push_only,
    )
    st = _states.get(key)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_fused_moe must run once outside CUDA-graph capture first "
                "(it compiles its kernels and allocates its scratch on the first call)."
            )
        with _lock:
            st = _states.get(key)
            if st is None:
                # head_flags always explicit: its environment fallback must not reach the plain op.
                config = {"head_flags": int(head_flags), "lat_slab": int(lat_slab)}
                if ar_world:
                    config["ar_world"] = ar_world
                    config["ar_push_only"] = int(ar_push_only)
                st = _states[key] = _K3FusedMoE(device, i_tp, num_local, config)
    return st


# ---------------------------------------------------------------------------
# Fused all-reduce: this TP group's Lamport buffers, 2 x [MAX_TOKENS][group][3584] bf16 per
# rank behind one multicast mapping (the MNNVL all-reduce's allocator), and a local flag
# holding the buffer the next call uses. The kernel leaves both buffers empty (every word
# 0x80000000) after each call. Separate from the model's all-reduce workspace, which the
# shared expert's all-reduce uses concurrently on its auxiliary stream.
# ---------------------------------------------------------------------------
AR_BUFFERS = 2
AR_EMPTY_WORD = -(2**31)
_ar_workspaces: Dict[object, dict] = {}
_head_workspaces: Dict[object, dict] = {}


def ar_buffer_words(world: int) -> int:
    return AR_BUFFERS * MAX_TOKENS * world * HIDDEN_SIZE // 2


def ar_workspace(mapping) -> dict:
    """The fused all-reduce buffers of ``mapping``'s TP group; collective on first use, so
    every rank of the group must make the first call at the same point, outside CUDA-graph
    capture."""
    return _lamport_workspace(mapping, ar_buffer_words(mapping.tp_size), _ar_workspaces)


def _rqag_module():
    """k3_route_quant_ag.py next to this file (also when op.py is loaded outside the package)."""
    mod = _modules.get("rqag")
    if mod is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_route_quant_ag.py")
        spec = importlib.util.spec_from_file_location(f"{__name__}_k3_route_quant_ag", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _modules["rqag"] = mod
    return mod


def head_workspace(mapping) -> dict:
    """The head all-gather's buffers of ``mapping``'s TP group (``trtllm::k3_route_quant_ag``), then
    ``trtllm::k3_moe_front``'s router partials; collective on first use like ``ar_workspace``.
    ``ready``: route_quant_ag's per-token ready words for k3_moe's flag handoff; flags[2] is their
    epoch."""
    ws = _lamport_workspace(
        mapping, _rqag_module().workspace_words(mapping.tp_size), _head_workspaces
    )
    if "ready" not in ws:
        ws["ready"] = torch.zeros(32, dtype=torch.int32, device=ws["uc"].device)
    return ws


def _lamport_workspace(mapping, words: int, cache: Dict[object, dict]) -> dict:
    ws = cache.get(mapping)
    if ws is not None:
        return ws
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "k3_fused_moe: the head all-gather and fused all-reduce buffers are collective on first "
            "use and must be allocated outside CUDA-graph capture"
        )
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    world = mapping.tp_size
    comm = _get_mnnvl_workspace_comm(mapping)
    use_fabric_handle = (
        os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
    )
    error: Optional[Exception] = None
    ws = None
    try:
        handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, use_fabric_handle)
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        with torch.inference_mode():
            uc.fill_(AR_EMPTY_WORD)
            flags = torch.zeros(4, dtype=torch.int32, device=uc.device)
        torch.cuda.synchronize()
        ws = dict(
            handle=handle, comm=comm, uc=uc, mc=mc, flags=flags, rank=mapping.tp_rank, world=world
        )
    except Exception as exc:  # noqa: BLE001 -- reported to every rank below, then re-raised
        error = exc
    # Also the barrier that keeps any rank from pushing into a peer's buffer before the
    # peer has emptied it.
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError("k3_fused_moe Lamport buffers failed on at least one rank") from error
    cache[mapping] = ws
    return ws


# ---------------------------------------------------------------------------
# The head all-gather + route + quant (k3_route_quant_ag.py): one kernel instead of the MNNVL
# all-gather of the sharded head followed by trtllm::k3_route_quant.
# ---------------------------------------------------------------------------
_rqag_compiled: Dict[Tuple[int, bool, bool, bool], object] = {}


@torch.library.custom_op("trtllm::k3_route_quant_ag", mutates_args=())
def k3_route_quant_ag(
    head: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    ag_uc: torch.Tensor,
    ag_mc: torch.Tensor,
    ag_flags: torch.Tensor,
    ag_rank: int,
    early_trigger: bool = False,
    ag_ready: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gathers every rank's slice of the sharded MoE head (``head``: this rank's fp32
    ``[M, (3584 + 896) / world]``, latent columns then router logits) through ``head_workspace``'s
    buffers and returns ``trtllm::k3_route_quant``'s outputs for the gathered logits and latent:
    ``(topk_ids, topk_weights, quantized, scales)``. With ``ag_ready`` it also releases the
    per-token ready words for a k3_moe built with head_flags."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    rqag = _rqag_module()
    num_tokens, width = head.shape
    world = (HIDDEN_SIZE + NUM_EXPERTS) // width
    if (
        head.dtype != torch.float32
        or width * world != HIDDEN_SIZE + NUM_EXPERTS
        or not 0 < num_tokens <= MAX_TOKENS
    ):
        raise ValueError(
            f"k3_route_quant_ag: head must be fp32 [M <= {MAX_TOKENS}, (3584 + 896) / world]"
        )
    device = head.device
    topk_ids = torch.empty(num_tokens, TOP_K, dtype=torch.int32, device=device)
    topk_weights = torch.empty(num_tokens, TOP_K, dtype=torch.bfloat16, device=device)
    quantized = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(num_tokens, HIDDEN_SIZE // _SF_VEC, dtype=torch.uint8, device=device)

    def arg(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=0)

    args = (
        arg(head.contiguous().view(-1).view(torch.int32)),
        arg(bias.contiguous().view(-1)),
        arg(ag_uc.view(-1)),
        arg(ag_mc.view(-1)),
        arg(ag_flags.view(-1)),
        arg(topk_ids.view(-1)),
        arg(topk_weights.view(-1).view(torch.int16)),
        arg(quantized.view(-1).view(torch.int32)),
        arg(scales.view(-1)),
        arg(ag_ready.view(-1) if ag_ready is not None else ag_flags.view(-1)),
    )
    publish = ag_ready is not None
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    key = (world, bool(early_trigger), publish, use_pdl)
    fn = _rqag_compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_route_quant_ag must run once outside CUDA-graph capture first"
            )
        with _lock:
            fn = _rqag_compiled.get(key)
            if fn is None:
                fn = _rqag_compiled[key] = cute.compile(
                    rqag.k3_route_quant_ag, *args, num_tokens, ag_rank, float(routed_scaling_factor), world,
                    bool(early_trigger), publish, use_pdl, stream,
                )  # fmt: skip
    fn(*args, num_tokens, ag_rank, float(routed_scaling_factor), stream)
    return topk_ids, topk_weights, quantized, scales


@k3_route_quant_ag.register_fake
def _(
    head,
    bias,
    routed_scaling_factor,
    ag_uc,
    ag_mc,
    ag_flags,
    ag_rank,
    early_trigger=False,
    ag_ready=None,
):
    num_tokens = head.shape[0]
    return (
        head.new_empty((num_tokens, TOP_K), dtype=torch.int32),
        head.new_empty((num_tokens, TOP_K), dtype=torch.bfloat16),
        head.new_empty((num_tokens, HIDDEN_SIZE), dtype=torch.float8_e4m3fn),
        head.new_empty((num_tokens, HIDDEN_SIZE // _SF_VEC), dtype=torch.uint8),
    )


@torch.library.custom_op("trtllm::k3_fused_moe_head", mutates_args=())
def k3_fused_moe_head(
    head: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
    ag_uc: torch.Tensor,
    ag_mc: torch.Tensor,
    ag_flags: torch.Tensor,
    ag_rank: int,
    ar_uc: Optional[torch.Tensor] = None,
    ar_mc: Optional[torch.Tensor] = None,
    ar_flags: Optional[torch.Tensor] = None,
    ar_rank: int = -1,
    ag_ready: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``trtllm::k3_fused_moe`` (or ``_ar`` with the ``ar_*`` buffers) from this rank's slice of
    the sharded MoE head: ``trtllm::k3_route_quant_ag`` gathers it, routes and quantizes, then
    ``k3_moe``. Returns ``[M, 3584]`` bf16 (the partial, or the reduced latent). With ``ag_ready``
    (``head_workspace``'s ready words) k3_moe acquires route_quant_ag's outputs through them
    instead of waiting for its grid."""
    if head.shape[0] > MAX_TOKENS:
        raise ValueError(f"k3_fused_moe handles at most {MAX_TOKENS} tokens, got {head.shape[0]}")
    ar = None
    world = 0
    if ar_uc is not None:
        ar = (ar_uc, ar_mc, ar_flags, ar_rank)
        world = ar_uc.numel() // ar_buffer_words(1)
    st = _state(
        head.device, w3_w1_weight.shape[1] // 2, local_num_experts, world, ag_ready is not None
    )
    return st(
        head, None, e_score_correction_bias, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale,
        local_expert_offset, local_num_experts, routed_scaling_factor, ar=ar,
        head_ag=(ag_uc, ag_mc, ag_flags, ag_rank), head_ready=ag_ready,
    )  # fmt: skip


@k3_fused_moe_head.register_fake
def _(head, e_score_correction_bias, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale,
      local_expert_offset, local_num_experts, routed_scaling_factor, ag_uc, ag_mc, ag_flags, ag_rank,
      ar_uc=None, ar_mc=None, ar_flags=None, ar_rank=-1, ag_ready=None):  # fmt: skip
    return head.new_empty((head.shape[0], HIDDEN_SIZE), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::k3_fused_moe_front", mutates_args=())
def k3_fused_moe_front(
    x: torch.Tensor,
    w_front: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
    shared_cols: int,
    gate_cap: float,
    linear_cap: float,
    ag_uc: torch.Tensor,
    ag_mc: torch.Tensor,
    ag_flags: torch.Tensor,
    ag_rank: int,
    ag_world: int,
    ar_uc: Optional[torch.Tensor] = None,
    ar_mc: Optional[torch.Tensor] = None,
    ar_flags: Optional[torch.Tensor] = None,
    ar_rank: int = -1,
    ag_ready: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``trtllm::k3_moe_front`` (head GEMV, head all-gather, routing, MXFP8 latent, shared gate_up + SiTU) then
    ``k3_moe`` (or its fused all-reduce build with the ``ar_*`` buffers) for the MoE input ``x`` bf16 ``[M, 7168]``.
    Returns ``(y [M, 3584] bf16, shared activation [M, shared_cols] bf16)``. With ``ag_ready``
    (``head_workspace``'s ready words) k3_moe acquires the front's routing and MXFP8 rows through them instead of
    waiting for the front's grid, whose shared tiles may still be running."""
    if x.shape[0] > MAX_TOKENS:
        raise ValueError(f"k3_fused_moe handles at most {MAX_TOKENS} tokens, got {x.shape[0]}")
    ar = None
    world = 0
    if ar_uc is not None:
        ar = (ar_uc, ar_mc, ar_flags, ar_rank)
        world = ar_uc.numel() // ar_buffer_words(1)
    st = _state(
        x.device, w3_w1_weight.shape[1] // 2, local_num_experts, world, ag_ready is not None
    )
    return st(
        x, None, e_score_correction_bias, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale,
        local_expert_offset, local_num_experts, routed_scaling_factor, ar=ar,
        head_ag=(ag_uc, ag_mc, ag_flags, ag_rank), head_ready=ag_ready,
        front=(w_front, shared_cols, gate_cap, linear_cap, ag_world),
    )  # fmt: skip


@k3_fused_moe_front.register_fake
def _(x, w_front, e_score_correction_bias, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale,
      local_expert_offset, local_num_experts, routed_scaling_factor, shared_cols, gate_cap, linear_cap, ag_uc, ag_mc,
      ag_flags, ag_rank, ag_world, ar_uc=None, ar_mc=None, ar_flags=None, ar_rank=-1, ag_ready=None):  # fmt: skip
    return (
        x.new_empty((x.shape[0], HIDDEN_SIZE), dtype=torch.bfloat16),
        x.new_empty((x.shape[0], shared_cols), dtype=torch.bfloat16),
    )


@torch.library.custom_op("trtllm::k3_fused_moe", mutates_args=())
def k3_fused_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Kimi K3 routed experts for M <= 8 decode tokens; returns ``[M, 3584]`` bf16.

    ``hidden_states``: bf16 ``[M, 3584]`` latent; ``router_logits``: fp32 ``[M, 896]``;
    ``e_score_correction_bias``: fp32 ``[896]``; weights: the TRTLLM-Gen
    W4A8_MXFP4_MXFP8 buffers of this rank's ``local_num_experts`` experts, which hold
    global ids ``[local_expert_offset, local_expert_offset + local_num_experts)``.
    """
    if hidden_states.shape[0] > MAX_TOKENS:
        raise ValueError(
            f"k3_fused_moe handles at most {MAX_TOKENS} tokens, got {hidden_states.shape[0]}"
        )
    st = _state(hidden_states.device, w3_w1_weight.shape[1] // 2, local_num_experts)
    return st(
        hidden_states, router_logits, e_score_correction_bias, w3_w1_weight, w3_w1_weight_scale,
        w2_weight, w2_weight_scale, local_expert_offset, local_num_experts, routed_scaling_factor,
    )  # fmt: skip


@k3_fused_moe.register_fake
def _(hidden_states, router_logits, e_score_correction_bias, w3_w1_weight, w3_w1_weight_scale, w2_weight,
      w2_weight_scale, local_expert_offset, local_num_experts, routed_scaling_factor):  # fmt: skip
    return hidden_states.new_empty((hidden_states.shape[0], HIDDEN_SIZE), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::k3_fused_moe_ar", mutates_args=("lat_slab",))
def k3_fused_moe_ar(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
    ar_uc: torch.Tensor,
    ar_mc: torch.Tensor,
    ar_flags: torch.Tensor,
    ar_rank: int,
    lat_slab: Optional[torch.Tensor] = None,
    lat_buf: int = 0,
    lat_rearm0: bool = False,
) -> torch.Tensor:
    """``trtllm::k3_fused_moe`` followed by the all-reduce of the routed partial over the
    group of ``ar_workspace``'s buffers (``ar_uc``, ``ar_mc``, ``ar_flags``), in one kernel.
    Returns the reduced ``[M, 3584]`` bf16 latent, identical on every rank. With ``lat_slab``
    (int32 ``[3, 8, 1792]``, all-ones empty) the reduced rows are also published into buffer
    ``lat_buf`` for a consumer that polls them, and the call re-arms buffer ``(lat_buf + 1) % 3``
    (and buffer 0 with ``lat_rearm0``, the step's last call)."""
    if hidden_states.shape[0] > MAX_TOKENS:
        raise ValueError(
            f"k3_fused_moe handles at most {MAX_TOKENS} tokens, got {hidden_states.shape[0]}"
        )
    world = ar_uc.numel() // ar_buffer_words(1)
    st = _state(
        hidden_states.device, w3_w1_weight.shape[1] // 2, local_num_experts, world,
        lat_slab=lat_slab is not None,
    )  # fmt: skip
    return st(
        hidden_states, router_logits, e_score_correction_bias, w3_w1_weight, w3_w1_weight_scale,
        w2_weight, w2_weight_scale, local_expert_offset, local_num_experts, routed_scaling_factor,
        ar=(ar_uc, ar_mc, ar_flags, ar_rank),
        lat_slab=(lat_slab, lat_buf, lat_rearm0) if lat_slab is not None else None,
    )  # fmt: skip


@k3_fused_moe_ar.register_fake
def _(hidden_states, router_logits, e_score_correction_bias, w3_w1_weight, w3_w1_weight_scale, w2_weight,
      w2_weight_scale, local_expert_offset, local_num_experts, routed_scaling_factor, ar_uc, ar_mc, ar_flags,
      ar_rank, lat_slab=None, lat_buf=0, lat_rearm0=False):  # fmt: skip
    return hidden_states.new_empty((hidden_states.shape[0], HIDDEN_SIZE), dtype=torch.bfloat16)


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
