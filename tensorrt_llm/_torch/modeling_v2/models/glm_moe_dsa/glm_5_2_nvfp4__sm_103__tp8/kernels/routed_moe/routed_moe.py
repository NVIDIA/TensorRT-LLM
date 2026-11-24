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
"""Host side of the routed_moe routed NVFP4 MoE (the routed_moe contract).

``RoutedMoe`` wraps one stock TRT-LLM-Gen NVFP4 MoE leaf (one rank's TP
shard) and runs the single-launch CuTe DSL kernel on its weight buffers in
place: no packed copies, no layout conversion. The call site may pass the
served ``ConfigurableMoE`` wrapper or the bare leaf; the leaf that owns the
expert parameters is ``wrapper.backend``. The schedule is the
expert-cluster kernel (``kernel_xc.py``, raw tuple, plus a native-order
finalize kernel for finalized output); a call whose rows need more
co-resident clusters than the grid holds runs the grid-wide kernel
(``kernel.py``, ``RoutedMoe.schedule``). ``eligibility`` is decided from
host-side state (types, shapes, dtypes, routing configuration and the device)
before any device work, so an ineligible call site keeps the native path with
no side effect.

Outputs follow the routed_moe contract: the finalized routed output ``[R, 6144]`` BF16, or
the native unfinalized tuple ``(fc2_output [R*8, 6144] BF16, expert scales
[R, 8] BF16, expanded_idx_to_permuted_idx [R, 8] int32)`` whose permutation is
the identity ``t*8 + k`` (every row valid; a stock finalizer sums
``scale[t, k] * fc2_output[idx[t, k]]``).

Physical layout of the raw tuple. ``fc2_output`` is token-major: row
``t * 8 + k`` holds the FC2 output of token ``t``'s ``k``-th expert, with a row
pitch of 6144 elements, and the permutation maps ``(t, k)`` to that row. The
native runner groups its rows by expert and pads them; both layouts are read
through the permutation, so the native unfinalized consumers (the single-node
``MoEAllReduce`` finalize + all-reduce and the post_moe_hop) take either
one. The expert scales carry the native dtype, values and top-k order.

Programmatic dependent launch (PDL). Both kernels are launched as PDL
secondaries of whatever kernel precedes them on the stream (the router GEMM,
router_gemm's router or a pre-MoE hop):

* before ``griddepcontrol.wait`` they only set up shared memory, barriers and
  tensor memory and read the router bias, which no kernel writes during
  serving (it is a model parameter); every input produced on the stream
  (activation, router logits) and every output store comes after the wait.
  With package cross_layer_expert_prefetch's speculative prefetch, the expert-cluster kernel also
  reads the predicted expert IDs there and issues L2 prefetches of those
  experts' weights; the IDs are written on another stream (or by the
  previous call of the same layer) and only steer the prefetch, so any value
  read is harmless;
* the expert-cluster kernel issues ``griddepcontrol.launch_dependents`` after
  each CTA's last FC2 tile; that only lets the successor's CTAs be scheduled
  earlier, and the successor (the finalize kernel or the post-MoE hop) reads
  the routed output only after its own ``griddepcontrol.wait``, which returns
  when this grid has completed and its memory operations are visible;
* the finalize kernel reads the raw rows after its ``griddepcontrol.wait``
  and triggers its dependents after its stores;
* outputs written into a consumer's buffer (routed_moe_into_hop) are not reused by this
  package; the consumer owns their lifetime and clears the FP32 accumulator
  after reading it.

Work on other streams (the shared expert on the aux stream) joins through
events as in the native model, not through PDL.

Numerics. The expert-cluster schedule computes every token's FC1, SwiGLU,
intermediate quantization and FC2 independently of the other rows of the
call (no split-K), and the finalize kernel sums each token's top-k in order,
so the raw rows and the finalized output of a live row do not depend on the
content of padded rows (bit for bit). The FP32 accumulator output (routed_moe_into_hop
"fp32") adds the eight weighted rows of a token with atomic FP32 adds whose
order depends on the schedule, so it is run-to-run nondeterministic within
the FP32 reorder bound. The grid-wide ``v6`` schedule splits FC1 over the
distinct experts of the whole call and is not padding-invariant.
"""

import contextlib
import threading
from typing import Dict, List, Optional, Tuple

import torch

HIDDEN = 6144
INTER = 256
NUM_EXPERTS = 256
TOP_K = 8
MAX_ROWS = 8
ROUTED_SCALING = 2.5

_NUM_ADDRS = 19
# Expert-cluster schedule defaults for the served neighbours (attention
# weight stream, pre-MoE hop, the BF16 shared expert on the aux stream, post_moe_hop /
# MNNVL all-reduce hop). The shared expert runs concurrently with the routed
# kernel and needs SMs and DRAM latency headroom; how much slack it has
# depends on the output mode:
# * finalized: the finalize kernel, the add and the all-reduce follow on the
#   main stream, so at most 32 four-CTA clusters work;
# * raw tuple (into the post_moe_hop, which joins the shared branch right after the
#   routed kernel) with the router logits supplied by the pre-MoE hop (post_attention_hop or
#   router_gemm selected: the MoE's router GEMM is skipped, so the shared expert's
#   first GEMM starts together with the routed kernel instead of during the
#   router GEMM): 26 clusters, raised to cover rows x top-8 in two rounds for
#   R > 6;
# * the same with routed_moe_into_hop's FP32 accumulator output: the routed kernel writes R
#   rows instead of R x 8 and finishes earlier, so the shared expert is the
#   critical path at 26 clusters; 24 clusters (raised the same way for
#   R > 6);
# * raw tuple after the MoE's own router GEMM: 32 clusters.
# For D <= clusters distinct experts only clamp(D, 24, clusters) clusters
# work; the others exit after routing and leave their SMs to the shared
# expert.
XC_MAX_CLUSTERS = 32
XC_RAW_CLUSTERS = 26
XC_RED_CLUSTERS = 24
XC_MIN_CLUSTERS = 24
# selector IDs of the pre-MoE hops that emit the router logits (the router contract)
_LOGIT_HOPS = ("post_attention_hop", "router_gemm")


#: Stress-test delay injection of the routed kernels: ``(site bits, unit in ns)``, the sites being
#: ``kernel.DELAY_*`` / ``kernel_xc.DELAY_*``. Serving delays no site; tests set it to hold CTAs at
#: those sites. It is part of every runner's compile key.
DELAY_INJECTION = (0, 2000)


def xc_kernel_args() -> dict:
    """Launch parameters of the expert-cluster kernel (``kernel_xc``) and of its forks.

    Ring depth 8, TMEM accumulator stages 4, PDL trigger ``late``, at least
    ``XC_MIN_CLUSTERS`` working clusters, no expert-cluster L2 prefetch
    (``l2pf`` 0), every cluster working in a two-round call
    (``two_round_clusters`` 0) and ``DELAY_INJECTION``.
    """
    delay, delay_ns = DELAY_INJECTION
    return dict(
        num_stages=8,
        delay=delay,
        delay_ns=delay_ns,
        trigger="late",
        min_clusters=XC_MIN_CLUSTERS,
        l2pf=0,
        two_round_clusters=0,
        acc_stages=4,
    )


def _logits_from_hop() -> bool:
    """Whether a pre-MoE hop supplies the router logits (rank-identical)."""
    from ... import dispatch

    return any(dispatch.is_enabled(f) for f in _LOGIT_HOPS)


_NUM_XC_ADDRS = 18

_LOCK = threading.Lock()
_COMPILED: Dict[tuple, object] = {}
_WORKSPACES: Dict[int, "_Workspace"] = {}


def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


_MAX_CLUSTERS: Dict[tuple, int] = {}


def _max_active_clusters(device: torch.device, cluster: int) -> int:
    """Clusters of ``cluster`` CTAs that are co-resident at one CTA per SM
    (a cluster must fit in one GPC, so this can be below sm_count / cluster)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (idx, cluster)
    if key not in _MAX_CLUSTERS:
        import cutlass.utils

        _MAX_CLUSTERS[key] = int(
            cutlass.utils.HardwareInfo(device_id=idx).get_max_active_clusters(cluster)
        )
    return _MAX_CLUSTERS[key]


class _Workspace:
    """Graph-stable per-device scratch shared by every routed_moe call of the process.

    The scratch is private to routed_moe. Calls that use it must be ordered on the
    device: one workspace serves all layers, both output modes and every
    captured CUDA graph. ``ws`` holds one launch counter per CTA; flags and LL
    words carry the epoch of the launch that wrote them, so nothing needs
    clearing between calls or graph replays.

    Ordering is enforced for eager calls: a call issued from another stream
    than the previous eager call first makes its stream wait for that
    stream's submitted work (``order_on``). Captured calls are ordered by the
    stream the graph is replayed on, which must be the stream of the eager
    calls (or synchronized with it), as in the model executor.

    It is created once per device, outside any dispatch mode (a model built
    under the loader's meta-device mode still gets device memory), and never
    replaced, because captured graphs keep its addresses.
    """

    def __init__(self, device: torch.device, num_ctas: int):
        from .kernel import FC1_MT, HBUF_BYTES, LL_PART_BYTES, MAX_SLOTS, WS_WORDS

        self.num_ctas = num_ctas
        self.ws = torch.zeros(WS_WORDS, dtype=torch.int32, device=device)
        # FC1 and FC2 split partials, one LL slot per CTA each
        self.partials = torch.zeros(2 * num_ctas * LL_PART_BYTES, dtype=torch.uint8, device=device)
        # per-expert FC2 operand hand-off buffers and per-tile epoch flags
        self.himg = torch.zeros(MAX_SLOTS * HBUF_BYTES, dtype=torch.uint8, device=device)
        self.hflag = torch.zeros(MAX_SLOTS * FC1_MT, dtype=torch.int32, device=device)
        self.dummy = torch.zeros(64, dtype=torch.int32, device=device)
        # expert-cluster schedule, finalized mode: raw rows and routing
        # weights consumed by the finalize kernel
        self.rows = torch.empty(MAX_ROWS * TOP_K, HIDDEN, dtype=torch.bfloat16, device=device)
        self.scales = torch.empty(MAX_ROWS, TOP_K, dtype=torch.bfloat16, device=device)
        # per-output-tile arrival counters of the in-kernel finalize; the
        # closing CTA resets its tile's counter
        self.tile_count = torch.zeros(64, dtype=torch.int32, device=device)
        # stream of the last eager call (see order_on)
        self.stream: Optional[torch.cuda.Stream] = None

    def tensors(self) -> Dict[str, torch.Tensor]:
        return {name: t for name, t in vars(self).items() if isinstance(t, torch.Tensor)}

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors().values())

    def order_on(self, stream: torch.cuda.Stream) -> None:
        """Order an eager call on ``stream`` after the previous eager call.

        A no-op while ``stream`` is being captured: graph nodes run when the
        graph is replayed, in the order of the replay stream.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        prev = self.stream
        if prev is not None and prev != stream:
            stream.wait_stream(prev)
        self.stream = stream


def _workspace(device: torch.device, num_ctas: int) -> _Workspace:
    """The device's workspace, sized for every SM; created on first use."""
    key = device.index if device.index is not None else torch.cuda.current_device()
    ws = _WORKSPACES.get(key)
    if ws is None:
        try:
            from torch.utils._python_dispatch import _disable_current_modes
        except ImportError:  # the buffers are still checked below
            _disable_current_modes = contextlib.nullcontext
        dev = torch.device("cuda", key)
        with _disable_current_modes():
            ws = _Workspace(dev, max(num_ctas, _sm_count(dev)))
        bad = [n for n, t in ws.tensors().items() if t.is_meta or t.device != dev]
        if bad:
            raise RuntimeError(f"routed_moe workspace buffers {bad} are not device memory on {dev}")
        _WORKSPACES[key] = ws
    elif ws.num_ctas < num_ctas:
        raise RuntimeError(
            f"routed_moe workspace of cuda:{key} holds {ws.num_ctas} CTA slots, a launch needs "
            f"{num_ctas}; it cannot be replaced once CUDA graphs may reference it"
        )
    return ws


def resolve_expert_owner(module) -> Tuple[Optional[torch.nn.Module], str]:
    """The leaf that owns the NVFP4 expert parameters of ``module``.

    The served model passes a ``ConfigurableMoE`` wrapper, whose expert
    tensors live on ``wrapper.backend`` (the wrapper only has delegating
    properties); a bare TRT-LLM-Gen leaf owns them itself. Ownership is
    decided by registered parameters, not by attribute probes.
    """
    backend = getattr(module, "backend", None)
    owner = backend if isinstance(backend, torch.nn.Module) else module
    if not isinstance(owner, torch.nn.Module):
        return None, f"{type(module).__name__} is not a module"
    if "w3_w1_weight" not in owner._parameters or "w2_weight" not in owner._parameters:
        return None, f"{type(owner).__name__} owns no NVFP4 expert parameters"
    return owner, ""


def _param(owner, name):
    return owner._parameters.get(name) if isinstance(owner, torch.nn.Module) else None


def _check_vector(t, numel: int) -> bool:
    return t is not None and t.dtype == torch.float32 and t.numel() == numel and t.is_contiguous()


def _wrapper_reason(wrapper) -> str:
    """Wrapper-level behaviour the routed_moe call would skip; empty when none."""
    if getattr(wrapper, "comm", None) is not None:
        return "MoE communication strategy"
    if getattr(wrapper, "layer_load_balancer", None) is not None:
        return "EPLB load balancer"
    if getattr(wrapper, "enable_dwdp", False):
        return "DWDP"
    if getattr(wrapper, "unfinalized_combine_fn", None) is not None:
        return "unfinalized_combine_fn"
    if getattr(wrapper, "apply_router_weight_on_input", False):
        return "router weight on input"
    if getattr(wrapper, "_enable_perfect_router", False):
        return "perfect router"
    if getattr(wrapper, "reduce_results", False) and getattr(wrapper, "parallel_size", 1) > 1:
        return "in-op all-reduce"
    return ""


def _routing_reason(rm) -> str:
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod

    if not isinstance(rm, DeepSeekV3MoeRoutingMethod):
        return f"routing method {type(rm).__name__}"
    impl = rm.routing_impl
    groups = (impl.top_k, impl.n_group, impl.topk_group)
    if groups != (TOP_K, 1, 1) or float(impl.routed_scaling_factor) != ROUTED_SCALING:
        return f"routing {groups} scale {impl.routed_scaling_factor}"
    return ""


def _owner_reason(owner, device: torch.device) -> str:
    """Leaf-level eligibility: stock TRT-LLM-Gen NVFP4 TP8/EP1 shard."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.fp4_block_scale import TRTLLMGenNvfp4Base
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.identity import PROVIDER_TRTLLM
    from tensorrt_llm._torch.utils import ActivationType

    if not isinstance(owner, TRTLLMGenNvfp4Base):
        return f"{type(owner).__name__} is not a TRT-LLM-Gen NVFP4 leaf"
    if getattr(type(owner), "provider", None) != PROVIDER_TRTLLM:
        return f"provider {getattr(type(owner), 'provider', None)}"
    if getattr(owner, "use_dp", False):
        return "attention DP"
    routes_outside = getattr(owner, "_routes_outside_the_kernel", None)
    if routes_outside is None or routes_outside():
        return "separated routing"
    if getattr(owner, "activation_type", None) != ActivationType.Swiglu:
        return f"activation {getattr(owner, 'activation_type', None)}"
    if getattr(owner, "bias", False):
        return "expert bias"
    for name in ("act_alpha", "act_beta", "act_clamp"):
        if getattr(owner, name, None) is not None:
            return f"{name} set"
    if (
        getattr(owner, "num_slots", None) != NUM_EXPERTS
        or getattr(owner, "slot_start", None) != 0
        or getattr(owner, "expert_size_per_partition", None) != NUM_EXPERTS
    ):
        return "not all 256 experts local (EP or EPLB slots)"
    if getattr(owner, "hidden_size", None) != HIDDEN:
        return f"hidden {getattr(owner, 'hidden_size', None)}"
    quant_method = getattr(owner, "quant_method", None)
    lean = getattr(quant_method, "intermediate_size_per_partition_lean", None)
    if lean not in (None, INTER):
        return f"lean intermediate {lean}"
    w13 = _param(owner, "w3_w1_weight")
    w2 = _param(owner, "w2_weight")
    w13_sf = _param(owner, "w3_w1_weight_scale")
    w2_sf = _param(owner, "w2_weight_scale")
    if tuple(w13.shape) != (NUM_EXPERTS, 2 * INTER, HIDDEN // 2) or w13.dtype != torch.uint8:
        return f"w3_w1 {tuple(w13.shape)} {w13.dtype}"
    if tuple(w2.shape) != (NUM_EXPERTS, HIDDEN, INTER // 2) or w2.dtype != torch.uint8:
        return f"w2 {tuple(w2.shape)} {w2.dtype}"
    if (
        w13_sf is None
        or w2_sf is None
        or tuple(w13_sf.shape) != (NUM_EXPERTS, 2 * INTER, HIDDEN // 16)
        or tuple(w2_sf.shape) != (NUM_EXPERTS, HIDDEN, INTER // 16)
    ):
        return "block-scale shapes"
    for t in (w13, w2, w13_sf, w2_sf):
        if not t.is_contiguous() or t.device != device:
            return "expert buffers not contiguous on the activation's device"
    for name in ("fc31_alpha", "fc31_scale_c", "fc2_alpha"):
        t = _param(owner, name)
        if not _check_vector(t, NUM_EXPERTS) or t.device != device:
            return f"{name} is not FP32 [256] on the device"
    t = _param(owner, "fc31_input_scale")
    if not _check_vector(t, 1) or t.device != device:
        return "fc31_input_scale is not one FP32 scalar on the device"
    return _routing_reason(getattr(owner, "routing_method", None))


# Both schedules copy each activation row into shared memory with
# ``cp.async.bulk``, whose global source must be 16-byte aligned.
_BULK_ALIGN = 16
_REALIGNED_LOGGED = []


def _bulk_copy_ready(x: torch.Tensor) -> torch.Tensor:
    """``x`` itself, or an aligned copy when its base is not 16-byte aligned.

    The base address comes from this rank's allocator, so it never decides
    between routed_moe and the native path (with routed_moe_into_hop the choice also selects the
    post_moe_hop hop variant): a misaligned activation costs one device copy instead.
    """
    if x.data_ptr() % _BULK_ALIGN == 0:
        return x
    if not _REALIGNED_LOGGED:
        _REALIGNED_LOGGED.append(True)
        from tensorrt_llm.logger import logger

        logger.warning(
            "[glm_moe_dsa] routed_moe: activation base is not 16-byte aligned "
            f"(data_ptr % 16 = {x.data_ptr() % _BULK_ALIGN}); copying it before every such call"
        )
    return x.clone(memory_format=torch.contiguous_format)


def eligibility(module, x: torch.Tensor, router_logits: torch.Tensor) -> Tuple[bool, str]:
    """Decide routed_moe support for this call from host-side state only.

    ``module`` is the served ``ConfigurableMoE`` wrapper or a bare
    TRT-LLM-Gen NVFP4 leaf.
    """
    if not torch.cuda.is_available() or x.device.type != "cuda":
        return False, "not on CUDA"
    major, _ = torch.cuda.get_device_capability(x.device)
    if major != 10:
        return False, f"SM{major}x is not the SM100 family"
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != HIDDEN:
        return False, f"activation {tuple(x.shape)} {x.dtype}"
    rows = x.shape[0]
    if rows < 1 or rows > MAX_ROWS:
        return False, f"{rows} rows outside [1, {MAX_ROWS}]"
    if not x.is_contiguous() or not router_logits.is_contiguous():
        return False, "non-contiguous input"
    if (
        router_logits.dtype != torch.float32
        or tuple(router_logits.shape) != (rows, NUM_EXPERTS)
        or router_logits.device != x.device
    ):
        return False, f"logits {tuple(router_logits.shape)} {router_logits.dtype}"
    if torch.compiler.is_compiling():
        return False, "torch.compile tracing"
    owner, reason = resolve_expert_owner(module)
    if owner is None:
        return False, reason
    if owner is not module:
        reason = _wrapper_reason(module) or _routing_reason(getattr(module, "routing_method", None))
        if reason:
            return False, reason
    reason = _owner_reason(owner, x.device)
    if reason:
        return False, reason
    return True, ""


class RoutedMoe:
    """Callable bound to one native NVFP4 MoE leaf and its router bias.

    ``module`` may be the ``ConfigurableMoE`` wrapper or the leaf itself; the
    kernel always reads the leaf's parameters.
    """

    def __init__(self, module, routing_bias: torch.Tensor):
        owner, reason = resolve_expert_owner(module)
        if owner is None:
            raise ValueError(f"routed_moe needs an NVFP4 expert leaf: {reason}")
        self.module = owner
        self.bias = routing_bias
        self.device = owner.w3_w1_weight.device
        self.num_ctas = _sm_count(self.device)

    @classmethod
    def from_native(cls, module, routing_bias: torch.Tensor) -> "RoutedMoe":
        return cls(module, routing_bias)

    def _spec(self):
        """Speculative prefetch of this layer's experts (cross_layer_expert_prefetch):
        ``(ids, k, ctas, self_store)`` or ``None``."""
        hook = getattr(self.module, "_expert_prefetch_spec", None)
        return hook() if hook is not None else None

    def _key(self, finalize: bool, debug: int, rows: int = MAX_ROWS, red: bool = False):
        """``(num_ctas, finalize, debug, schedule, grid, red, prefetch spec, delay injection)`` of a
        launch."""
        sched = self.schedule(rows, finalize, red)
        grid = self._xc_ctas(rows, finalize, red) if sched == "xc" else self.num_ctas
        spec = self._spec() if sched == "xc" else None
        spec_key = spec[1:] if spec is not None else None
        return (self.num_ctas, finalize, debug, sched, grid, red, spec_key, DELAY_INJECTION)

    def schedule(self, rows: int = MAX_ROWS, finalize: bool = True, red: bool = False) -> str:
        """``xc`` (expert clusters) or ``v6`` (grid-wide hand-off).

        ``xc`` needs enough co-resident clusters for two rounds to cover the
        largest distinct-expert count of a ``rows``-row call (8 per row);
        otherwise the grid-wide schedule runs.
        """
        from .kernel_xc import CLUSTER, MAX_ROUNDS

        if (self._xc_ctas(rows, finalize, red) // CLUSTER) * MAX_ROUNDS < rows * TOP_K:
            return "v6"
        return "xc"

    @staticmethod
    def inkernel_finalize() -> bool:
        """Whether ``xc`` finalizes inside the routed kernel: no, a second, PDL-launched kernel
        finalizes."""
        return False

    def _xc_ctas(self, rows: int = MAX_ROWS, finalize: bool = True, red: bool = False) -> int:
        """Grid of the expert-cluster schedule for a ``rows``-row call.

        Every cluster must be co-resident, and in the served graph the BF16
        shared expert runs concurrently on the aux stream, so the grid leaves
        SMs free: ``XC_MAX_CLUSTERS`` clusters (32 on sm_103 = 128 of 152 SMs),
        or, when a pre-MoE hop supplies the router logits, ``XC_RAW_CLUSTERS``
        (26 = 104 SMs) for the raw tuple and ``XC_RED_CLUSTERS`` (24 = 96 SMs)
        for the FP32 accumulator output (``red``), raised to
        ``ceil(rows * 8 / 2)`` so that two rounds cover every distinct expert
        of the call.
        """
        from .kernel_xc import CLUSTER, MAX_ROUNDS

        co_resident = min(XC_MAX_CLUSTERS, _max_active_clusters(self.device, CLUSTER))
        clusters = co_resident
        if not finalize and _logits_from_hop():
            need = -(-rows * TOP_K // MAX_ROUNDS)
            base = XC_RED_CLUSTERS if red else XC_RAW_CLUSTERS
            clusters = min(co_resident, max(base, need))
        n = CLUSTER * clusters
        n = min(n, self.num_ctas)
        return n - n % CLUSTER

    def ready_for_capture(
        self, finalize: bool, debug: int = 0, rows: int = MAX_ROWS, red: bool = False
    ) -> bool:
        """True when a launch needs no compilation and no workspace
        allocation, i.e. it can be recorded into a CUDA graph."""
        key = self._key(finalize, debug, rows, red)
        dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
        ws = _WORKSPACES.get(dev)
        return key in _COMPILED and ws is not None and ws.num_ctas >= self.num_ctas

    def _compiled(self, finalize: bool, debug: int, rows: int = MAX_ROWS, red: bool = False):
        """Compiled launcher(s): ``(routed, finalize_or_None)``.

        ``red``: the expert-cluster kernel adds the weighted FC2 outputs into
        an FP32 accumulator instead of writing the raw rows (routed_moe_into_hop "fp32").
        """
        key = self._key(finalize, debug, rows, red)
        with _LOCK:
            fns = _COMPILED.get(key)
            if fns is not None:
                return fns
            import cutlass
            import cutlass.cute as cute
            from cuda.bindings import driver as cuda
            from cutlass.cute.runtime import make_ptr

            ptrs = self._pointers(make_ptr, cute, cutlass)
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            if key[3] == "xc":
                from .kernel_xc import RoutedMoeExpertClusterKernel, RoutedMoeFinalizeKernel

                inkernel = finalize and self.inkernel_finalize()
                spec = key[6] or (0, 0, False)
                kern = RoutedMoeExpertClusterKernel(
                    self._xc_ctas(rows, finalize, red),
                    debug=debug,
                    finalize=inkernel,
                    red_out=red,
                    **xc_kernel_args(),
                    spec_k=spec[0],
                    spec_ctas=spec[1],
                    spec_self=spec[2],
                )
                ints = [cutlass.Int64(0)] * _NUM_XC_ADDRS
                routed = cute.compile(kern.run, *ptrs, *ints, cutlass.Int32(1), stream)
                if spec[0]:
                    from ... import dispatch

                    dispatch.log_capability(
                        "routed_moe",
                        f"expert_prefetch[k={spec[0]},ctas={spec[1]},self={int(spec[2])}]",
                        True,
                    )
                fin = None
                if finalize and not inkernel and not red:
                    from .kernel_xc import DELAY_FINALIZE

                    delay, delay_ns = DELAY_INJECTION
                    fin = cute.compile(
                        RoutedMoeFinalizeKernel(
                            delay_ns=delay_ns if delay & DELAY_FINALIZE else 0
                        ).run,
                        cutlass.Int64(0),
                        cutlass.Int64(0),
                        cutlass.Int64(0),
                        cutlass.Int32(1),
                        stream,
                    )
                fns = (routed, fin)
            else:
                from .kernel import RoutedMoeKernel

                delay, delay_ns = DELAY_INJECTION
                kern = RoutedMoeKernel(
                    self.num_ctas, finalize=finalize, debug=debug, delay=delay, delay_ns=delay_ns
                )
                ints = [cutlass.Int64(0)] * _NUM_ADDRS
                fns = (cute.compile(kern.run, *ptrs, *ints, cutlass.Int32(1), stream), None)
            _COMPILED[key] = fns
            return fns

    def _pointers(self, make_ptr, cute, cutlass):
        m = self.module
        return [
            make_ptr(
                cutlass.Float4E2M1FN,
                m.w3_w1_weight.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                m.w3_w1_weight_scale.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float4E2M1FN,
                m.w2_weight.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                m.w2_weight_scale.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
        ]

    def __call__(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        do_finalize: bool = True,
        debug: int = 0,
        debug_buffer: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        raw_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        red_out: Optional[torch.Tensor] = None,
    ):
        """Run the routed MoE.

        ``out``: finalized output buffer. ``raw_out``: ``(rows [R*8, H] bf16,
        scales [R, 8] bf16)`` buffers for the raw tuple (identity layout; the
        permutation is not written). ``red_out``: FP32 ``[R, H]`` accumulator
        that receives ``sum_k w[t, k] * fc2[t, k]`` (zero on entry; expert
        cluster schedule only).
        """
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda
        from cutlass.cute.runtime import make_ptr

        m = self.module
        rows = x.shape[0]
        x = _bulk_copy_ready(x)
        red = red_out is not None
        if red:
            do_finalize = False
            if self.schedule(rows, False, True) != "xc":
                raise ValueError("FP32 accumulator output needs the expert-cluster schedule")
        routed_fn, fin_fn = self._compiled(do_finalize, debug, rows, red)
        ws = _workspace(self.device, self.num_ctas)
        dbg = debug_buffer if debug_buffer is not None else ws.dummy
        current = torch.cuda.current_stream()
        ws.order_on(current)
        stream = cuda.CUstream(current.cuda_stream)
        ptrs = self._pointers(make_ptr, cute, cutlass)
        params = [
            x.data_ptr(),
            router_logits.data_ptr(),
            self.bias.data_ptr(),
            m.fc31_alpha.data_ptr(),
            m.fc31_scale_c.data_ptr(),
            m.fc2_alpha.data_ptr(),
            m.fc31_input_scale.data_ptr(),
        ]
        if self.schedule(rows, do_finalize, red) == "xc":
            if red:
                out = red_out
                raw_rows, raw_scale, raw_idx = ws.rows, ws.scales, ws.dummy
            elif do_finalize:
                raw_rows, raw_scale, raw_idx = ws.rows, ws.scales, ws.dummy
                if out is None:
                    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=x.device)
            elif raw_out is not None:
                out = ws.dummy
                raw_rows, raw_scale = raw_out
                raw_idx = ws.dummy
            else:
                out = ws.dummy
                raw_rows = torch.empty(rows * TOP_K, HIDDEN, dtype=torch.bfloat16, device=x.device)
                raw_scale = torch.empty(rows, TOP_K, dtype=torch.bfloat16, device=x.device)
                raw_idx = torch.empty(rows, TOP_K, dtype=torch.int32, device=x.device)
            addrs = params + [
                raw_rows.data_ptr(),
                raw_scale.data_ptr(),
                raw_idx.data_ptr(),
                out.data_ptr(),
                ws.tile_count.data_ptr(),
                dbg.data_ptr(),
                m.w3_w1_weight.data_ptr(),
                m.w3_w1_weight_scale.data_ptr(),
                m.w2_weight.data_ptr(),
                m.w2_weight_scale.data_ptr(),
            ]
            spec = self._spec()
            addrs.append(spec[0].data_ptr() if spec is not None else ws.dummy.data_ptr())
            assert len(addrs) == _NUM_XC_ADDRS
            routed_fn(*ptrs, *[cutlass.Int64(a) for a in addrs], cutlass.Int32(rows), stream)
            if red:
                return [red_out]
            if not do_finalize:
                return [raw_rows, raw_scale, raw_idx]
            if fin_fn is None:
                return out
            fin_fn(
                cutlass.Int64(raw_rows.data_ptr()),
                cutlass.Int64(raw_scale.data_ptr()),
                cutlass.Int64(out.data_ptr()),
                cutlass.Int32(rows),
                stream,
            )
            return out
        if do_finalize:
            if out is None:
                out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=x.device)
            raw_scale = ws.dummy
            raw_idx = ws.dummy
        else:
            out = torch.empty(rows * TOP_K, HIDDEN, dtype=torch.bfloat16, device=x.device)
            raw_scale = torch.empty(rows, TOP_K, dtype=torch.bfloat16, device=x.device)
            raw_idx = torch.empty(rows, TOP_K, dtype=torch.int32, device=x.device)
        addrs = params + [
            out.data_ptr(),
            raw_scale.data_ptr(),
            raw_idx.data_ptr(),
            ws.ws.data_ptr(),
            ws.partials.data_ptr(),
            ws.himg.data_ptr(),
            ws.hflag.data_ptr(),
            dbg.data_ptr(),
            m.w3_w1_weight.data_ptr(),
            m.w3_w1_weight_scale.data_ptr(),
            m.w2_weight.data_ptr(),
            m.w2_weight_scale.data_ptr(),
        ]
        assert len(addrs) == _NUM_ADDRS
        routed_fn(*ptrs, *[cutlass.Int64(a) for a in addrs], cutlass.Int32(rows), stream)
        if do_finalize:
            return out
        return [out, raw_scale, raw_idx]


def _engaged_runner(
    module,
    routing_bias: torch.Tensor,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    do_finalize: bool,
    feature: str = "routed_moe",
    red: bool = False,
) -> Optional["RoutedMoe"]:
    """The runner for this call, or ``None`` (decided before any device work)."""
    from ... import dispatch

    ok, reason = eligibility(module, x, router_logits)
    owner = resolve_expert_owner(module)[0] if ok else None
    if ok:
        rm = getattr(owner, "routing_method", None)
        native_bias = getattr(rm, "e_score_correction_bias", None)
        if (
            routing_bias is None
            or routing_bias.dtype != torch.bfloat16
            or tuple(routing_bias.shape) != (NUM_EXPERTS,)
            or not routing_bias.is_contiguous()
            or routing_bias.device != x.device
        ):
            ok, reason = False, "router bias is not a contiguous BF16 [256] on the device"
        elif native_bias is not None and native_bias.data_ptr() != routing_bias.data_ptr():
            ok, reason = False, "router bias is not the routing method's bias"
    runner = getattr(owner, "_routed_moe_runner", None) if ok else None
    if ok and runner is not None and runner.bias.data_ptr() != routing_bias.data_ptr():
        runner = None
    if ok and torch.cuda.is_current_stream_capturing():
        # Compiling or allocating the workspace inside a capture would record
        # the workspace initialization into the graph; the model's eager
        # warmup forwards make both ready before capture.
        if runner is None or not runner.ready_for_capture(do_finalize, rows=x.shape[0], red=red):
            ok, reason = False, "not warmed up before CUDA graph capture"
    dispatch.log_capability(feature, f"rows={x.shape[0]} {reason or 'engaged'}", ok, reason)
    if not ok:
        return None
    if runner is None:
        runner = RoutedMoe(owner, routing_bias)
        owner._routed_moe_runner = runner
    return runner


def routed_output(
    module,
    routing_bias: torch.Tensor,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    do_finalize: bool,
) -> Optional[List[torch.Tensor]]:
    """Entry point for the routed-MoE call site; ``None`` means "use the native path".

    The decision is made before any device work, so returning ``None`` leaves
    no side effect behind.
    """
    runner = _engaged_runner(module, routing_bias, x, router_logits, do_finalize)
    if runner is None:
        return None
    return runner(x, router_logits, do_finalize=do_finalize)


def routed_output_into_hop(
    module,
    routing_bias: torch.Tensor,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    hop,
) -> Optional[List[torch.Tensor]]:
    """routed_moe_into_hop: routed_moe writes its per-rank routed result into the post_moe_hop
    post-MoE hop's routed-input buffer; post_moe_hop then adds the shared output,
    reduces across ranks and applies residual and RMSNorm (``hop.hop_routed``).

    The hop's routed-input mode (``hop.routed_mode``) decides what is written:

    * "row": the finalized routed output ``[R, H]`` bf16 (finalize kernel);
    * "identity": the raw FC2 rows ``[R * 8, H]`` in the identity layout plus
      the routing weights in the hop's scale buffer; post_moe_hop finalizes without the
      permutation round trip (bit-identical to the raw tuple);
    * "fp32": an FP32 ``[R, H]`` accumulator of ``w[t, k] * fc2[t, k]``, added
      by the routed kernel itself (no finalize kernel, no raw rows); the sum
      order differs from the native finalize (FP32 reorder class).

    Returns ``[buffer]`` (a view of the hop's buffer) or ``None`` when routed_moe or
    the hop's routed input is not eligible; the decision is taken before any
    device work and depends only on rank-identical state.
    """
    if hop is None or not getattr(hop, "_routed_enabled", False):
        return None
    rows = x.shape[0]
    if not hop.can_run(rows):
        return None
    mode = getattr(hop, "routed_mode", "row")
    if mode == "row":
        runner = _engaged_runner(
            module, routing_bias, x, router_logits, True, feature="routed_moe_into_hop"
        )
        if runner is None:
            return None
        target = hop.routed_input_buffer(rows)
        return [runner(x, router_logits, do_finalize=True, out=target)]
    red = mode == "fp32"
    runner = _engaged_runner(
        module, routing_bias, x, router_logits, False, "routed_moe_into_hop", red
    )
    if runner is None or runner.schedule(rows, False, red) != "xc":
        return None
    if red:
        return runner(x, router_logits, red_out=hop.routed_acc_buffer(rows))
    rows_buf, scales_buf = hop.routed_raw_buffers(rows)
    runner(x, router_logits, do_finalize=False, raw_out=(rows_buf, scales_buf))
    return [rows_buf]
