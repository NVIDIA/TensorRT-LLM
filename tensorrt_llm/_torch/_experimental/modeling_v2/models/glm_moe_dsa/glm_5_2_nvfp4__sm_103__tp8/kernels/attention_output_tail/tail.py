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
"""Per-layer attention_output_tail adapter between the MLA generation tail and ``o_proj``.

The native GEN tail is split across two call sites of one ``MLA.forward``:
``forward_absorption_generation`` applies W_UV inside the attention custom op
and ``_project_output`` applies ``o_proj``. ``AttnTail`` joins them in three
steps:

1. ``arm`` runs in ``MLA.forward`` before the attention op. It is the only
   place that decides whether this call may take the attention_output_tail, so the W_UV
   step inside the op and the projection step after it always see the same
   decision. It declines, without touching any state, while Dynamo traces the
   module, under TensorRT-LLM's torch.compile policy (``is_torch_compiling``)
   and inside a breakable CUDA graph: in those modes the attention op runs
   separately from the traced or captured projection site, so only the
   native path is consistent.
2. ``defer_w_uv`` runs at the W_UV step inside the op. It acts only on an
   armed call and checks the latent's shape before any GPU work. When it
   accepts, it keeps the native latent tensor in a packet tagged with the
   output buffer the native W_UV would have written, and the caller skips the
   native BMM. When it declines, nothing changes.
3. ``project`` runs before the native projection of an armed call. It
   consumes the packet, launches the tail and returns ``o_proj``'s rank-local
   partial before any communication (the attention_output_tail contract). The caller reduces it as
   the native ROW-parallel ``Linear.forward`` reduces this ``o_proj``'s GEMM
   output. With no packet it returns ``None`` and the caller runs the native
   projection.

With ``attention_output_tail_prefetch`` (``TailConfig.prefetch_l2``) there is one more step between
1 and 2: ``before_attention`` runs in the op's generation attention
(``forward_absorption_generation``, right before the attention backend call)
of an armed call and forks the off-path L2 prefetch of this layer's tail
weights (``prefetch.py``). The prefetch stream is joined in ``project`` (at
the configured point), which an armed call always reaches, so a CUDA-graph
capture always sees the fork rejoined within the layer. A prefetch changes
no tensor; outputs are identical with and without it. With the ``native``
engine (``attention_output_tail_prefetch`` experiments) the native W_UV runs in the op and
``project`` runs the native o_proj GEMM itself, then joins.

No path of this module communicates: the all-reduce of the partial, when
there is one, belongs to the caller.

The packet belongs to one module instance and one forward call. A packet that
is left over, or that does not match the projection's input buffer, raises
instead of producing a wrong result.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._torch.utils import is_torch_compiling

from ... import dispatch
from .kernels import (
    DEFAULT_CONFIG,
    INTO_HOP_SCHEDULE,
    INTO_HOP_SPARSE_DECODE_SCHEDULE,
    ROW_TILE,
    TAIL_PREFETCH_SCHEDULE,
    TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE,
    TAIL_PREFETCH_SPARSE_DECODE_SCHEDULE,
    TailConfig,
    attn_tail,
    check_tail_shapes,
    check_weights,
)
from .prefetch import PrefetchConfig, WeightPrefetch, check_prefetch_tensors

FEATURE_ID = "attention_output_tail"
TAIL_PREFETCH_PATH = "attention_output_tail_prefetch"
INTO_HOP_PATH = "attention_output_into_hop"
_JOIN_POINTS = ("before_wuv", "before_oproj", "after_oproj")
_PREFETCH_LAYERS = ("all", "shared", "full")
MAX_ENGAGED_ROWS = 8
"""Largest row count attention_output_tail takes: the kernels support 16, but above 8 rows the
tail needs a second n8 MMA tile and more ``v`` traffic, and the native tail runs."""
SUPPORTED_TP_SIZE = 8
"""o_proj tensor-parallel width attention_output_tail is verified for (the GEN role's TP8)."""
ENGAGED_DOMAIN = (
    f"GEN-only calls of 1-{MAX_ENGAGED_ROWS} token rows, "
    f"TP{SUPPORTED_TP_SIZE}, no attention DP; any draft length "
    "and request mix within that row count"
)
"""Per-call domain logged with an engaged capability decision. The tail is
row-wise, so the draft length and the request mix of a GEN batch do not change
its arithmetic; only the row count, GEN-only and the static checks gate it."""
MAX_INTO_HOP_ROWS = 8
"""Row capacity of the post_attention_hop transport (``attention_output_into_hop``): output rows, padding
included."""


@dataclass
class _Packet:
    latent: torch.Tensor
    output_ptr: int
    num_rows: int


def _static_reason(mla) -> Optional[str]:
    """Return why ``mla`` can never take the attention_output_tail, or ``None``."""
    from tensorrt_llm._torch.attention.backends.sparse.hooks import MLASparseHooks
    from tensorrt_llm._torch.attention.mla import MLA
    from tensorrt_llm._torch.modules.linear import TensorParallelMode, UnquantizedLinearMethod
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() not in (100, 103):
        return f"SM{get_sm_version()} is not SM100/SM103"
    if mla.mapping.enable_attention_dp:
        return "attention data parallelism"
    o_tp_size = getattr(mla, "mapping_o", mla.mapping).tp_size
    if o_tp_size != SUPPORTED_TP_SIZE:
        return f"o_proj tensor-parallel size {o_tp_size} is not {SUPPORTED_TP_SIZE}"
    if mla.mapping.has_cp_helix() or mla.num_heads_tp != mla.num_heads_tp_cp:
        return "helix context parallelism"
    if type(mla)._apply_output_gate is not MLA._apply_output_gate:
        return "module overrides the output gate"
    hooks = mla.sparse_attn_hooks
    if hooks is not None and (type(hooks).project_output is not MLASparseHooks.project_output):
        return "sparse hooks own the output projection"
    o_proj = getattr(mla, "o_proj", None)
    w_uv = getattr(mla, "v_b_proj", None)
    if o_proj is None or w_uv is None:
        return "module has no default o_proj or v_b_proj"
    # The FP8 linear methods subclass UnquantizedLinearMethod.
    if type(o_proj.quant_method) is not UnquantizedLinearMethod:
        return "o_proj is quantized"
    if o_proj.tp_mode != TensorParallelMode.ROW:
        return "o_proj is not ROW parallel"
    if o_proj.bias is not None or o_proj.lora is not None:
        return "o_proj has bias or LoRA"
    if o_proj.use_fused_gemm_allreduce:
        return "o_proj uses the fused GEMM+all-reduce"
    # The native GEMM writes its output into an NCCL symmetric-memory window
    # only through the custom cuBLAS path (UnquantizedLinearMethod.apply); the
    # default F.linear output is an ordinary buffer, which attention_output_tail matches.
    if (
        getattr(o_proj, "use_custom_cublas_mm", False)
        and o_proj.all_reduce is not None
        and o_proj.all_reduce.uses_nccl_symmetric_memory_window()
    ):
        return "o_proj writes into an NCCL window buffer"
    return check_weights(w_uv, o_proj.weight)


class AttnTail:
    """attention_output_tail state of one MLA module (one layer, one rank).

    Args:
        layer_key: name used in capability logs, e.g. ``"layer3"``.
        config: kernel launch configuration (benchmarks select ablations).
    """

    def __init__(self, layer_key: str, config: TailConfig = DEFAULT_CONFIG):
        self.layer_key = layer_key
        self.config = config
        self.engaged_calls = 0
        self.declined_calls = 0
        self._static_ok: Optional[bool] = None
        self.static_reason: Optional[str] = None
        """Why this layer never takes attention_output_tail (``None`` if it can); set on the
        first ``arm``."""
        self._armed = False
        self._pending: Optional[_Packet] = None
        # attention_output_tail_prefetch: off-path weight prefetch, created on first use.
        self._prefetch: Optional[WeightPrefetch] = None
        self.prefetch_forks = 0
        # True while a prefetch forked ahead of this layer's call (by the
        # previous layer's hook, prefetch_fork="prev") awaits the call.
        self._ahead = False
        # Weights of the op that follows the tail (successor_prefetch).
        self._successor_weights = []
        # attention_output_into_hop: (transport, hop rows) requested by the decoder layer for
        # the next call, the request this call may honour, and whether it
        # published.
        self._into_hop_request = None
        self._into_hop = None
        self._into_hop_published = False
        self.into_hop_published_calls = 0
        # Requests that arm accepted but project did not publish (output
        # rows differ from the hop's rows or exceed the transport).
        self.into_hop_declined_calls = 0
        # Counters of the (non-default) fused engine, private to this layer.
        # Allocated on the first eager call so a CUDA graph never captures
        # their zeroing.
        self._state: Optional[torch.Tensor] = None

    def _check_static(self, mla) -> bool:
        if self._static_ok is None:
            reason = _static_reason(mla)
            self.static_reason = reason
            self._static_ok = dispatch.log_capability(
                FEATURE_ID, self.layer_key, reason is None, reason or ENGAGED_DOMAIN
            )
        return self._static_ok

    def arm(self, mla, attn_metadata) -> bool:
        """Decide, before the attention op runs, whether this call may use attention_output_tail.

        Called from ``MLA.forward``. Returns False, and touches no state, while
        Dynamo traces the module or TensorRT-LLM's torch.compile policy is on;
        both call sites then take the native path. Otherwise records the
        decision for ``defer_w_uv`` and returns it.

        Args:
            mla: the calling ``MLA`` module.
            attn_metadata: the call's attention metadata.
        """
        if torch.compiler.is_compiling() or is_torch_compiling():
            return False
        from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph

        if self._pending is not None or (
            self._prefetch is not None and self._prefetch.forked and not self._ahead
        ):
            raise RuntimeError(
                f"[glm_moe_dsa] attention_output_tail/{self.layer_key}: previous tail packet or "
                "prefetch was not consumed by the output projection"
            )
        num_rows = attn_metadata.num_tokens
        armed = (
            self._check_static(mla)
            and not is_in_breakable_cuda_graph()
            and attn_metadata.num_contexts == 0
            and 1 <= num_rows <= min(ROW_TILE, MAX_ENGAGED_ROWS)
        )
        if armed and self.config.oproj_engine == "fused" and self._state is None:
            if torch.cuda.is_current_stream_capturing():
                armed = False
            else:
                from .cute_fused import new_state

                self._state = new_state(torch.device("cuda"))
        self._armed = armed
        # attention_output_into_hop: a call that takes the attention_output_tail kernels can also
        # publish its partial into the post_attention_hop transport; the request is consumed either
        # way.
        request, self._into_hop_request = self._into_hop_request, None
        self._into_hop = (
            request
            if armed and request is not None and self.config.oproj_engine == "cute_stream"
            else None
        )
        self._into_hop_published = False
        if self._ahead and not armed:
            # A prefetch forked ahead of a call that takes the native path:
            # join it right away so the fork never outlives the layer.
            self._join_prefetch()
        if not armed:
            self.declined_calls += 1
        return armed

    def request_into_hop(self, transport, rows: int) -> None:
        """``attention_output_into_hop``: ask the next call to publish its partial into
        ``transport`` (the ``IntoHopTransport`` of the layer's post_attention_hop).

        Called by the decoder layer before the attention, from rank-uniform
        facts. ``rows`` is the row count the hop will poll (the rows of its
        residual). The call honours the request only if ``arm`` accepts the
        call, the attention_output_tail kernels run and the projection's output has exactly
        ``rows`` rows (live rows plus padding, at most ``MAX_INTO_HOP_ROWS``);
        otherwise nothing is published and the layer runs the post_attention_hop.
        ``take_into_hop_published`` tells the layer which one happened.
        """
        self._into_hop_request = (transport, int(rows))

    def take_into_hop_published(self) -> bool:
        """``attention_output_into_hop``: whether the last call published its partial into the
        requested transport; if so the layer must run the into-hop entry next.

        Also drops a request the call did not consume (``arm`` returns early
        under torch.compile before it reads the request), so a request never
        carries over to a later call.
        """
        published, self._into_hop_published = self._into_hop_published, False
        self._into_hop_request = None
        return published

    def set_successor_weights(self, weights) -> None:
        """``attention_output_tail_prefetch``: weights the op after the tail reads first (staged into L2
        with ``TailConfig.successor_prefetch``). Called once by the model
        adapter; the tensors are read at fork time, so in-place weight
        loading after construction is fine."""
        self._successor_weights = list(weights)

    def _prefetch_tensors(self, mla):
        """The tensors this call stages, or an empty list (static facts)."""
        cfg = self.config
        tensors = []
        if self._prefetch_wanted(mla):
            tensors += [mla.v_b_proj, mla.o_proj.weight]
        if cfg.successor_prefetch and self._successor_weights:
            succ = [w.data if hasattr(w, "data") else w for w in self._successor_weights]
            if check_prefetch_tensors(succ, cfg.prefetch_chunk) == "":
                tensors += succ
        return tensors[:3]

    def _prefetch_wanted(self, mla) -> bool:
        cfg = self.config
        if not cfg.prefetch_l2:
            return False
        if (
            cfg.prefetch_layers not in _PREFETCH_LAYERS
            or cfg.prefetch_join not in _JOIN_POINTS
            or cfg.prefetch_fork not in ("start", "attn", "prev")
        ):
            raise ValueError(
                f"[glm_moe_dsa] attention_output_tail_prefetch/{self.layer_key}: bad prefetch "
                f"config {cfg}"
            )
        if cfg.prefetch_layers != "all":
            has_indexer = getattr(getattr(mla, "mqa", None), "indexer", None) is not None
            if has_indexer != (cfg.prefetch_layers == "full"):
                return False
        weights = [mla.v_b_proj, mla.o_proj.weight]
        if check_prefetch_tensors(weights, cfg.prefetch_chunk):
            return False
        if cfg.oproj_engine == "native" and cfg.native_oproj == "ring":
            from .cute_oproj_ring import check_shapes as ring_shapes

            n_dim, k_dim = mla.o_proj.weight.shape
            return ring_shapes(n_dim, k_dim, cfg.ring_grid, cfg.ring_slots) is None
        return True

    def before_attention(self, mla, point: str = "start") -> None:
        """``attention_output_tail_prefetch``: fork the off-path L2 prefetch of this layer's tail weights.

        Called at the start of the generation attention of the op
        (``point="start"``). Acts only on a call that ``arm`` accepted, only
        when the config selects the prefetch and only at the configured fork
        point; the decision uses static facts and the armed flag, before any
        device work. The fork is joined in ``project``.
        """
        if not self._armed or point != self.config.prefetch_fork:
            return
        tensors = self._prefetch_tensors(mla)
        if not tensors:
            return
        cfg = self.config
        if self._prefetch is None:
            self._prefetch = WeightPrefetch(
                PrefetchConfig(
                    chunk=cfg.prefetch_chunk,
                    ctas=cfg.prefetch_ctas,
                    threads=cfg.prefetch_threads,
                    gap_ns=cfg.prefetch_gap_ns,
                )
            )
        self._prefetch.fork(tensors)
        self.prefetch_forks += 1

    def fork_ahead(self, mla) -> None:
        """``attention_output_tail_prefetch`` with ``prefetch_fork="prev"``: fork this layer's weight
        prefetch now, ahead of its call.

        Called by the previous layer right before its post-attention or
        post-MoE hop (an NVLink-latency-bound window), from static facts only
        (config, capability, layer selection); no-op otherwise. The fork is
        joined in this layer's ``project``, or in ``arm`` if the call takes
        the native path.
        """
        if self.config.prefetch_fork != "prev" or self._ahead or not self._check_static(mla):
            return
        tensors = self._prefetch_tensors(mla)
        if not tensors:
            return
        if self._prefetch is None:
            cfg = self.config
            self._prefetch = WeightPrefetch(
                PrefetchConfig(
                    chunk=cfg.prefetch_chunk,
                    ctas=cfg.prefetch_ctas,
                    threads=cfg.prefetch_threads,
                    gap_ns=cfg.prefetch_gap_ns,
                )
            )
        self._prefetch.fork(tensors)
        self._ahead = True
        self.prefetch_forks += 1

    def _join_prefetch(self) -> None:
        self._ahead = False
        if self._prefetch is not None:
            self._prefetch.join()

    def defer_w_uv(self, mla, attn_metadata, latent: torch.Tensor, output: torch.Tensor) -> bool:
        """Take over this call's W_UV + o_proj if ``arm`` accepted the call.

        Args:
            mla: the calling ``MLA`` module.
            attn_metadata: the call's attention metadata.
            latent: ``[num_gen_tokens, H, C]`` native latent attention output.
            output: the generation rows of ``attn_output[0]``, which the native
                W_UV BMM would write.

        Returns:
            True if the caller must skip the native W_UV; the packet is then
            consumed by ``project``.
        """
        armed, self._armed = self._armed, False
        if not armed:
            return False
        if self._pending is not None:
            raise RuntimeError(
                f"[glm_moe_dsa] attention_output_tail/{self.layer_key}: previous tail packet was "
                "not consumed by the output projection"
            )
        if self.config.oproj_engine == "native":
            # Experiments only: keep the native W_UV/o_proj (the prefetch
            # fork, if any, is still joined in project).
            return False
        num_rows = latent.shape[0]
        eligible = (
            attn_metadata.num_contexts == 0
            and 1 <= num_rows <= min(ROW_TILE, MAX_ENGAGED_ROWS)
            and output.shape[0] == num_rows
            and check_tail_shapes(latent, mla.v_b_proj, mla.o_proj.weight, num_rows, num_rows)
            is None
        )
        if not eligible:
            self.declined_calls += 1
            return False
        self._pending = _Packet(latent=latent, output_ptr=output.data_ptr(), num_rows=num_rows)
        self.engaged_calls += 1
        return True

    def project(
        self, mla, attn_output: torch.Tensor, all_reduce_params=None
    ) -> Optional[torch.Tensor]:
        """Return this call's rank-local o_proj output, or ``None`` for the
        native path.

        The result is the partial before any communication: the caller
        reduces it exactly as the ROW-parallel ``Linear.forward`` reduces the
        output of this ``o_proj`` (no bias, LoRA or fused GEMM+all-reduce;
        see ``_static_reason``), with the same ``AllReduceParams``.

        Args:
            mla: the calling ``MLA`` module.
            attn_output: ``attn_output[0]``, the native o_proj input buffer
                ``[out_rows, H * V]`` (its rows were not written by W_UV when
                a packet is pending).
            all_reduce_params: unused; the tail does not communicate.
        """
        del all_reduce_params
        self._armed = False
        packet = self._pending
        if packet is None:
            if (
                self.config.oproj_engine == "native"
                and self._prefetch is not None
                and self._prefetch.forked
            ):
                # attention_output_tail_prefetch with the native kernels: the native W_UV already
                # ran in the op; run the native o_proj GEMM here so that the join can
                # follow it (as in MLA._project_output without helix CP,
                # which the static check guarantees).
                if self.config.prefetch_join != "after_oproj":
                    self._join_prefetch()
                if self.config.native_oproj == "ring":
                    out = self._ring_oproj(mla, attn_output)
                else:
                    # The GEMM of Linear.forward for an o_proj without bias
                    # or LoRA, without its all-reduce.
                    out = mla.o_proj.apply_linear(attn_output, None, layer_idx=mla.layer_idx)
                self._join_prefetch()
                self.engaged_calls += 1
                return out
            # Declined at the W_UV step: the native projection follows.
            self._join_prefetch()
            return None
        self._pending = None
        if attn_output.data_ptr() != packet.output_ptr or attn_output.shape[0] < packet.num_rows:
            self._join_prefetch()
            raise RuntimeError(
                f"[glm_moe_dsa] attention_output_tail/{self.layer_key}: tail packet does not match "
                "the output projection input"
            )
        join = self.config.prefetch_join
        if join == "before_wuv":
            self._join_prefetch()
        hop_transport = None
        request, self._into_hop = self._into_hop, None
        if request is not None:
            transport, hop_rows = request
            # The hop polls hop_rows rows and later resets exactly those rows
            # of the buffer, so the producer must publish exactly them.
            if attn_output.shape[0] == hop_rows <= MAX_INTO_HOP_ROWS:
                hop_transport = transport
            else:
                self.into_hop_declined_calls += 1
        partial = attn_tail(
            packet.latent,
            mla.v_b_proj,
            mla.o_proj.weight,
            packet.num_rows,
            out_rows=attn_output.shape[0],
            config=self.config,
            state=self._state,
            between=(self._join_prefetch if join == "before_oproj" else None),
            hop_transport=hop_transport,
        )
        if hop_transport is not None:
            self._into_hop_published = True
            self.into_hop_published_calls += 1
        # Always joined before the layer's forward returns.
        self._join_prefetch()
        return partial

    def _ring_oproj(self, mla, attn_output: torch.Tensor) -> torch.Tensor:
        """o_proj GEMM of the ``native`` engine with ``native_oproj="ring"``."""
        from .cute_oproj_ring import ring_oproj
        from .kernels import pdl_enabled

        cfg = self.config
        w_o = mla.o_proj.weight
        out = torch.empty((attn_output.shape[0], w_o.shape[0]), dtype=w_o.dtype, device=w_o.device)
        ring_oproj(
            attn_output,
            w_o,
            out,
            use_pdl=pdl_enabled(),
            grid=cfg.ring_grid,
            slots=cfg.ring_slots,
            early=cfg.ring_early,
            trigger=pdl_enabled(),
        )
        return out


def _selected(feature_id: str) -> bool:
    """Whether ``feature_id`` is selected; False on a build whose selector
    does not know it."""
    if feature_id not in dispatch.SELECTOR_PATHS:
        return False
    return dispatch.is_enabled(feature_id)


def tail_prefetch_selected() -> bool:
    """Whether ``attention_output_tail_prefetch`` (attention_output_tail with off-path weight prefetch) is selected."""
    return _selected(TAIL_PREFETCH_PATH)


def tail_prefetch_config() -> Optional[TailConfig]:
    """``attention_output_tail_prefetch``'s tail schedule for the neighbours selected in this process.

    The schedule depends on the attention kernel before the tail (``sparse_mla_decode``
    releases its PDL dependents well before it ends, the native FMHA only
    shortly before) and on the op after it (the ``post_attention_hop`` needs whole
    SMs and gains nothing from router-weight staging; the native router
    GEMM does). The selector is identical on every rank, so is the choice.
    Returns None where the native tail is kept
    (``post_attention_hop`` without ``sparse_mla_decode``).
    """
    post_hop, sparse_decode = _selected("post_attention_hop"), _selected("sparse_mla_decode")
    if post_hop and not sparse_decode:
        return None
    if post_hop:
        return TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE
    if sparse_decode:
        return TAIL_PREFETCH_SPARSE_DECODE_SCHEDULE
    return TAIL_PREFETCH_SCHEDULE


def make_attn_tail(mla) -> Optional[AttnTail]:
    """Create the attention_output_tail for a target-model MLA module, or ``None``.

    Draft (MTP) layers, whose index is at or beyond the target layer count,
    keep the native tail. With ``attention_output_into_hop`` and ``post_attention_hop`` selected the
    tail uses ``INTO_HOP_SPARSE_DECODE_SCHEDULE`` behind ``sparse_mla_decode`` and
    ``INTO_HOP_SCHEDULE`` behind the native FMHA; ``attention_output_into_hop``
    without ``post_attention_hop`` is decided as for
    ``attention_output_tail_prefetch``/``attention_output_tail``. With
    ``attention_output_tail_prefetch`` selected the tail uses ``tail_prefetch_config()``; with only
    ``attention_output_tail`` it uses the attention_output_tail default.
    """
    layer_idx = getattr(mla, "layer_idx", None)
    num_layers = getattr(mla, "num_hidden_layers", None)
    if layer_idx is None or (num_layers is not None and layer_idx >= num_layers):
        dispatch.log_capability(FEATURE_ID, f"layer{layer_idx}", False, "draft or unindexed layer")
        return None
    key = f"layer{layer_idx}"
    if _selected(INTO_HOP_PATH):
        if _selected("post_attention_hop"):
            return AttnTail(
                key,
                INTO_HOP_SPARSE_DECODE_SCHEDULE
                if _selected("sparse_mla_decode")
                else INTO_HOP_SCHEDULE,
            )
        if not (tail_prefetch_selected() or _selected(FEATURE_ID)):
            dispatch.log_capability(INTO_HOP_PATH, key, False, "needs the post_attention_hop hop")
            return None
    if tail_prefetch_selected():
        config = tail_prefetch_config()
        if config is None:
            if _selected(FEATURE_ID):
                # This selection runs the default tail schedule instead.
                dispatch.log_capability(
                    TAIL_PREFETCH_PATH,
                    "schedule",
                    False,
                    "post_attention_hop without sparse_mla_decode",
                )
                return AttnTail(f"layer{layer_idx}")
            dispatch.log_capability(
                TAIL_PREFETCH_PATH,
                f"layer{layer_idx}",
                False,
                "post_attention_hop without sparse_mla_decode: native tail is faster",
            )
            return None
        return AttnTail(f"layer{layer_idx}", config)
    return AttnTail(f"layer{layer_idx}")
