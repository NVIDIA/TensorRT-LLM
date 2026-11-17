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
"""Cross-layer speculative routed-expert L2 prefetch (``cross_layer_expert_prefetch``).

At a decode call of MoE layer ``L`` (1 to 8 rows), the post-attention hop has
produced this layer's normalized hidden ``h_L`` and router logits. On a
private auxiliary stream, forked from the caller's stream at that point:

1. ``h_L`` times the next MoE layer's gate gives approximate router logits of
   layer ``L + 1``; every live row's top-8 of ``sigmoid(logit) + bias`` is
   taken, and the experts are ranked by how many rows select them (then by
   their best score). The first ``KMAX`` expert IDs are kept for layer
   ``L + 1``.
2. Layer ``L``'s own routing (from its router logits) is compared with the
   ranking predicted for it at layer ``L - 1``: per-layer device counters of
   calls, distinct experts and hits among the first K predicted experts.
3. After layer ``L``'s routed MoE (``start="moe"``),
   or right after the prediction (``hop``), or after layer ``L``'s post-MoE
   hop (``post``), a kernel issues ``cp.async.bulk.prefetch.L2`` over the
   local FC1 / FC2 weights and block scales of the first ``K`` predicted
   experts of layer ``L + 1`` (``k``, 0 = predict and count only).

With ``issue="xc"`` no prefetch kernel is launched: the routed_moe
expert-cluster kernel of layer ``L + 1`` issues the prefetch itself, before
its dependency wait, i.e. while the pre-MoE hop of layer ``L + 1`` still runs
(``routed_moe/kernel_xc.py``, ``spec_k``). With ``pred="self"`` there
is no auxiliary work at all: the expert-cluster kernel of layer ``L`` writes
its own experts, ranked by row count, and its next call (the next decode
step) prefetches them.

A prefetch has no architectural effect, and the auxiliary work writes only
this package's buffers, so model outputs are unchanged. The caller's stream
never waits for the auxiliary stream inside a layer: the hidden state and
router logits the auxiliary work reads are recorded on the auxiliary stream
(``Tensor.record_stream``), so the caching allocator does not reuse them
before that work is done (under CUDA-graph capture, not before the capture
ends). The caller's stream waits for all auxiliary work once, at the end of
the target model's forward (``join``), so a CUDA-graph capture always sees the
auxiliary stream rejoin.

Configuration (``ExpertPrefetchConfig``; this target runs ``SERVED_CONFIG``):

* ``k``: experts prefetched per layer, 0..32 (8).
* ``start``: ``moe``, ``hop`` or ``post`` (``moe``).
* ``pred``: where the prediction (and counters) is forked:
  ``hop`` (right after the pre-MoE hop, concurrent with layer L's
  routed MoE) or ``moe`` (right after layer L's routed MoE, together with a
  ``start=moe`` prefetch); ``moe`` excludes ``start=hop``. ``self``: no
  cross-layer prediction (see above; needs ``issue=xc``, no counters) (``self``).
* ``issue``: ``aux`` (a prefetch kernel on the auxiliary stream, placed by
  ``start`` and ``delay_ns``) or ``xc`` (the routed MoE kernel of the
  predicted layer) (``xc``).
* ``xc_ctas``: with ``issue=xc``, the routed-kernel CTAs that issue the
  prefetch (0 = all) (32).
* ``fused``: predict with one fused Triton launch (``kernels.gemm_rank``: gate
  GEMM, scores and ranking) when the next gate is a BF16 [256, 6144] weight;
  otherwise the next gate module plus the ranking kernel (True).
* ``delay_ns``: nanoseconds the auxiliary stream waits (one sleeping warp)
  right before each prefetch, to move the prefetch traffic later into layer
  L+1 (0).
* ``chunk``: bytes per prefetch instruction (16384).
* ``ctas``: CTAs of the prefetch kernel (16).

``start``, ``fused``, ``delay_ns``, ``chunk`` and ``ctas`` act only with the
``hop`` / ``moe`` predictors and the ``aux`` prefetch kernel; under ``self``
they appear only in the capability key.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ... import dispatch
from . import kernels

HIDDEN = 6144
STARTS = ("moe", "hop", "post")
PREDS = ("hop", "moe", "self")
ISSUES = ("aux", "xc")


@dataclass(frozen=True)
class ExpertPrefetchConfig:
    k: int
    start: str
    pred: str
    fused: bool
    delay_ns: int
    chunk: int
    ctas: int
    issue: str = "xc"
    xc_ctas: int = 32


#: The expert-prefetch configuration of this target: the routed MoE kernel of each layer writes
#: its own experts (``self``) and its next call prefetches the first 8 on 32 of its CTAs.
SERVED_CONFIG = ExpertPrefetchConfig(
    k=8,
    start="moe",
    pred="self",
    fused=True,
    delay_ns=0,
    chunk=16384,
    ctas=16,
    issue="xc",
    xc_ctas=32,
)


def config() -> ExpertPrefetchConfig:
    """The expert-prefetch configuration of this target, ``SERVED_CONFIG``."""
    return SERVED_CONFIG


class _DeviceState:
    """Per-device stream, events and buffers shared by every linked layer.

    Created on the first eager call of a device, never replaced (captured
    graphs keep the buffer addresses).
    """

    def __init__(self, device: torch.device, num_layers: int):
        self.device = device
        self.aux = torch.cuda.Stream(device=device)
        self.ev_fork = torch.cuda.Event()
        self.ev_tail = torch.cuda.Event()
        self.counters = torch.zeros(
            num_layers, kernels.NUM_COUNTERS, dtype=torch.int64, device=device
        )
        self.pred_ids = torch.zeros(num_layers + 1, kernels.KMAX, dtype=torch.int32, device=device)
        self.dummy_f32 = torch.zeros(
            kernels.MAX_ROWS, kernels.NUM_EXPERTS, dtype=torch.float32, device=device
        )
        self.dummy_bias = torch.zeros(kernels.NUM_EXPERTS, dtype=torch.float32, device=device)
        # scratch of the fused prediction (kernels.gemm_rank)
        self.scores = torch.zeros(
            kernels.MAX_ROWS, kernels.NUM_EXPERTS, dtype=torch.float32, device=device
        )
        self.ticket = torch.zeros(1, dtype=torch.int32, device=device)
        self.forked = False
        # prefetch launches issued from Python (eager calls and captures, not
        # graph replays)
        self.prefetch_calls = 0


_STATES: Dict[int, _DeviceState] = {}
_NUM_LAYERS = {"n": 0}


def _state(device: torch.device) -> _DeviceState:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _STATES.get(idx)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "[glm_moe_dsa] cross_layer_expert_prefetch device state must be created by an "
                "eager call before CUDA-graph capture"
            )
        try:
            from torch.utils._python_dispatch import _disable_current_modes
        except ImportError:
            import contextlib

            _disable_current_modes = contextlib.nullcontext
        with _disable_current_modes():
            st = _DeviceState(torch.device("cuda", idx), _NUM_LAYERS["n"])
        _STATES[idx] = st
    return st


def _expert_tensors(experts) -> Optional[List[torch.Tensor]]:
    """The four per-expert weight tensors of a native NVFP4 MoE leaf, or None."""
    from ..routed_moe.routed_moe import resolve_expert_owner

    owner, _ = resolve_expert_owner(experts)
    if owner is None:
        return None
    names = ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale")
    ts = [owner._parameters.get(n) for n in names]
    if any(t is None for t in ts):
        return None
    return ts


def _prefetch_reason(ts: Optional[List[torch.Tensor]], chunk: int) -> str:
    if ts is None:
        return "next layer has no native NVFP4 expert leaf"
    for t in ts:
        if not t.is_cuda or not t.is_contiguous():
            return "expert tensor is not contiguous on CUDA"
        if t.shape[0] != kernels.NUM_EXPERTS:
            return f"expert tensor {tuple(t.shape)} is not [256, ...]"
        if t.data_ptr() % 16:
            return "expert tensor is not 16-byte aligned"
        if (t[0].numel() * t.element_size()) % chunk:
            return f"expert slice of {tuple(t.shape)} is not a multiple of {chunk} bytes"
    return ""


class ExpertPrefetchLink:
    """cross_layer_expert_prefetch state of one MoE decoder layer ``L`` (bound by ``link_layers``)."""

    def __init__(self, layer_idx: int, cur_gate, next_gate, next_experts, has_prev: bool):
        self.layer_idx = layer_idx
        self.cur_gate = cur_gate
        self.next_gate = next_gate
        self.next_experts = next_experts
        self.has_prev = has_prev
        self._armed = False
        self._pending_prefetch = False
        self._pending_predict: Optional[tuple] = None
        self._next_tensors: Optional[List[torch.Tensor]] = None
        self._prefetch_ok: Optional[bool] = None
        self._fused_ok: Optional[bool] = None

    @staticmethod
    def _eligible(hidden, router_logits) -> str:
        if not isinstance(hidden, torch.Tensor) or not hidden.is_cuda:
            return "hidden is not a CUDA tensor"
        if hidden.dtype != torch.bfloat16 or hidden.dim() != 2 or hidden.shape[1] != HIDDEN:
            return f"hidden {tuple(hidden.shape)} {hidden.dtype}"
        rows = hidden.shape[0]
        if not 1 <= rows <= kernels.MAX_ROWS:
            return f"{rows} rows outside [1, {kernels.MAX_ROWS}]"
        if not hidden.is_contiguous():
            return "hidden is not contiguous"
        if router_logits is not None and (
            router_logits.dtype != torch.float32
            or tuple(router_logits.shape) != (rows, kernels.NUM_EXPERTS)
            or not router_logits.is_contiguous()
        ):
            return f"router logits {tuple(router_logits.shape)} {router_logits.dtype}"
        return ""

    def _prefetch_ready(self, cfg: ExpertPrefetchConfig) -> bool:
        if self._prefetch_ok is None:
            ts = _expert_tensors(self.next_experts) if self.next_experts is not None else None
            reason = _prefetch_reason(ts, cfg.chunk)
            self._prefetch_ok = not reason
            self._next_tensors = ts if not reason else None
            dispatch.log_capability(
                "cross_layer_expert_prefetch", "expert_prefetch", self._prefetch_ok, reason
            )
        return self._prefetch_ok

    def _launch_prefetch(self, st: _DeviceState, cfg: ExpertPrefetchConfig) -> None:
        if cfg.delay_ns:
            kernels.delay(cfg.delay_ns)
        w13, w13_sf, w2, w2_sf = self._next_tensors
        kernels.prefetch_experts(
            st.pred_ids[self.layer_idx + 1], w13, w13_sf, w2, w2_sf, cfg.k, cfg.chunk, cfg.ctas
        )
        st.prefetch_calls += 1

    def _fused_ready(self, cfg: ExpertPrefetchConfig) -> bool:
        if self._fused_ok is None:
            reason = ""
            if not cfg.fused:
                reason = "fused=0"
            else:
                w = getattr(self.next_gate, "weight", None)
                if (
                    not isinstance(w, torch.Tensor)
                    or w.dtype != torch.bfloat16
                    or tuple(w.shape) != (kernels.NUM_EXPERTS, HIDDEN)
                    or not w.is_contiguous()
                ):
                    reason = "next gate weight is not a contiguous BF16 [256, 6144]"
            self._fused_ok = not reason
            dispatch.log_capability(
                "cross_layer_expert_prefetch", "fused_predict", self._fused_ok, reason
            )
        return self._fused_ok

    def _predict(
        self,
        st: _DeviceState,
        cfg: ExpertPrefetchConfig,
        hidden,
        router_logits,
        has_pred: bool,
        has_stats: bool,
    ) -> None:
        """Prediction for layer L+1 and counters of layer L (current stream)."""
        rows = hidden.shape[0]
        out_ids = st.pred_ids[self.layer_idx + 1]
        fused = has_pred and self._fused_ready(cfg)
        if fused:
            kernels.gemm_rank(
                hidden,
                self.next_gate.weight,
                self.next_gate.e_score_correction_bias,
                st.scores,
                st.ticket,
                out_ids,
                rows,
            )
        if fused and not has_stats:
            return
        has_pred = has_pred and not fused
        next_logits = self.next_gate(hidden) if has_pred else st.dummy_f32
        next_bias = self.next_gate.e_score_correction_bias if has_pred else st.dummy_bias
        kernels.rank_and_stats(
            next_logits,
            next_bias,
            out_ids,
            router_logits if has_stats else st.dummy_f32,
            self.cur_gate.e_score_correction_bias if has_stats else st.dummy_bias,
            st.pred_ids[self.layer_idx],
            st.counters[self.layer_idx],
            rows,
            has_pred,
            has_stats,
        )

    def _fork(
        self, st: _DeviceState, cfg: ExpertPrefetchConfig, predict: Optional[tuple], prefetch: bool
    ) -> None:
        """Fork auxiliary work at this point of the caller's stream: the
        prediction (``predict`` = arguments of ``_predict``) and / or the
        prefetch of layer L+1, in that order."""
        caller = torch.cuda.current_stream()
        st.ev_fork.record(caller)
        st.aux.wait_event(st.ev_fork)
        with torch.cuda.stream(st.aux):
            if predict is not None:
                self._predict(st, cfg, *predict)
            if prefetch:
                self._launch_prefetch(st, cfg)
            st.ev_tail.record(st.aux)
        st.forked = True

    def on_hop(self, hidden, router_logits) -> None:
        """After layer L's pre-MoE hop: fork the prediction (``pred=hop``) or
        keep its inputs for ``after_moe`` (``pred=moe``)."""
        self._armed = False
        self._pending_prefetch = False
        self._pending_predict = None
        reason = self._eligible(hidden, router_logits)
        if reason:
            # calls outside the decode shape (e.g. warmup with many rows)
            dispatch.log_capability(
                "cross_layer_expert_prefetch", "spec_experts_declined", False, reason
            )
            return
        dispatch.log_capability("cross_layer_expert_prefetch", "spec_experts", True)
        cfg = config()
        # created eagerly here: the routed kernel's prefetch hook reads it
        st = _state(hidden.device)
        if cfg.pred == "self":
            return
        has_pred = self.next_gate is not None
        # The hit counters are not reported in this target.
        has_stats = False
        if not has_pred and not has_stats:
            return
        hidden.record_stream(st.aux)
        if has_stats:
            router_logits.record_stream(st.aux)
        predict = (hidden, router_logits, has_pred, has_stats)
        if cfg.pred == "hop":
            prefetch = (
                has_pred
                and cfg.k > 0
                and cfg.issue == "aux"
                and cfg.start == "hop"
                and self._prefetch_ready(cfg)
            )
            self._fork(st, cfg, predict, prefetch)
        else:
            self._pending_predict = predict
        self._armed = True
        self._pending_prefetch = (
            has_pred and cfg.k > 0 and cfg.issue == "aux" and cfg.start != "hop"
        )

    def after_moe(self) -> None:
        """After layer L's routed MoE: fork the pending prediction
        (``pred=moe``) and / or the prefetch of layer L+1 (``start=moe``)."""
        if not self._armed:
            return
        cfg = config()
        prefetch = self._pending_prefetch and cfg.start == "moe" and self._prefetch_ready(cfg)
        if prefetch:
            self._pending_prefetch = False
        predict, self._pending_predict = self._pending_predict, None
        if predict is not None or prefetch:
            st = _state(torch.device("cuda", torch.cuda.current_device()))
            self._fork(st, cfg, predict, prefetch)

    def after_post_moe(self) -> None:
        """After layer L's post-MoE hop (``start=post``)."""
        if not self._armed:
            return
        self._armed = False
        cfg = config()
        if self._pending_prefetch and cfg.start == "post" and self._prefetch_ready(cfg):
            self._pending_prefetch = False
            st = _state(torch.device("cuda", torch.cuda.current_device()))
            self._fork(st, cfg, None, True)


def _is_moe_with_gate(layer) -> bool:
    mlp = getattr(layer, "mlp", None)
    gate = getattr(mlp, "gate", None)
    return (
        mlp is not None
        and getattr(mlp, "experts", None) is not None
        and gate is not None
        and hasattr(gate, "e_score_correction_bias")
        and hasattr(gate, "weight")
    )


def _xc_hook(row: int, cfg: ExpertPrefetchConfig):
    """Prefetch hook of a routed-MoE leaf (read by ``routed_moe.routed_moe``): the
    ``pred_ids`` row its expert-cluster kernel prefetches, or ``None`` before
    the device state exists."""
    self_store = cfg.pred == "self"

    def hook():
        st = _STATES.get(torch.cuda.current_device())
        if st is None:
            return None
        return (st.pred_ids[row], cfg.k, cfg.xc_ctas, self_store)

    return hook


def _xc_issue_reason(cfg: ExpertPrefetchConfig) -> str:
    """Why the routed kernel cannot issue the prefetch ("" when it can)."""
    if not dispatch.is_enabled("routed_moe"):
        return "issue=xc needs routed_moe"
    return ""


def _bind_xc(experts, row: int, cfg: ExpertPrefetchConfig) -> bool:
    from ..routed_moe.routed_moe import resolve_expert_owner

    owner, _ = resolve_expert_owner(experts)
    if owner is None:
        return False
    owner._expert_prefetch_spec = _xc_hook(row, cfg)
    return True


def link_layers(layers, num_hidden_layers: int) -> int:
    """Bind a ``ExpertPrefetchLink`` to every target MoE layer (structural only)."""
    cfg = config()
    _NUM_LAYERS["n"] = max(_NUM_LAYERS["n"], num_hidden_layers)
    linked = 0
    xc_bound = 0
    xc_on = cfg.issue == "xc" and cfg.k > 0
    if xc_on:
        reason = _xc_issue_reason(cfg)
        xc_on = dispatch.log_capability(
            "cross_layer_expert_prefetch", "xc_prefetch", not reason, reason
        )
    for idx in range(num_hidden_layers):
        layer = layers[idx]
        if not _is_moe_with_gate(layer):
            continue
        nxt = layers[idx + 1] if idx + 1 < num_hidden_layers else None
        nxt_moe = nxt is not None and _is_moe_with_gate(nxt)
        prev_moe = idx > 0 and _is_moe_with_gate(layers[idx - 1])
        layer._expert_prefetch_link = ExpertPrefetchLink(
            idx,
            layer.mlp.gate,
            nxt.mlp.gate if nxt_moe else None,
            nxt.mlp.experts if nxt_moe else None,
            prev_moe,
        )
        linked += 1
        if xc_on:
            if cfg.pred == "self":
                xc_bound += _bind_xc(layer.mlp.experts, idx, cfg)
            elif nxt_moe:
                xc_bound += _bind_xc(nxt.mlp.experts, idx + 1, cfg)
    dispatch.log_capability(
        "cross_layer_expert_prefetch",
        f"link[k={cfg.k},start={cfg.start},pred={cfg.pred},fused={int(cfg.fused)},"
        f"delay_ns={cfg.delay_ns},chunk={cfg.chunk},"
        f"ctas={cfg.ctas},issue={cfg.issue},xc_ctas={cfg.xc_ctas}]",
        linked > 0,
        f"{linked} MoE layers, {xc_bound} routed-kernel prefetch hooks",
    )
    return linked


def join() -> None:
    """Make the current stream wait for all forked auxiliary work."""
    for st in _STATES.values():
        if st.forked and st.device.index == torch.cuda.current_device():
            torch.cuda.current_stream().wait_event(st.ev_tail)
            st.forked = False
