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
"""Host side of package draft_routed_moe: the routed experts of the MTP draft layer.

The MTP layer of the GLM-5.2 NVFP4 checkpoint is unquantized, so its routed
experts run FlashInfer's TRT-LLM-Gen BF16 MoE (``FlashinferTrtllmGenBf16Impl``,
``trtllm_bf16_moe``). ``routed_output`` replaces that call with
``native_ops.draft_routed_moe`` on the leaf's own weight buffers (no
copy, no re-layout) and returns the same finalized ``[R, 6144]`` BF16 tensor,
bit for bit:

* routing: FlashInfer's ``routingCustom`` with ``SigmoidBias`` preprocess and
  ``ScaledSumNormalize`` postprocess (DeepSeek-V3 with ``n_group <= 1``),
  including its fast-math instructions, tie order and FP32 sum order;
* FC1/FC2: ``mma.sync`` over K in natural order with one FP32 accumulator per
  output element (the batched-GEMM cubins accumulate K sequentially and have
  no split-K), the cubin's SwiGLU epilogue and BF16 rounding;
* finalize: ``finalizeKernel``'s FP32 ``fma`` chain in top-k slot order.

The eligibility decision uses host-side state only (module types, routing
configuration, shapes, dtypes, devices, the op's availability and whether the
scratch buffers exist before a CUDA graph capture) and is taken before any
device work; ``None`` leaves the native path with no side effect.

Programmatic dependent launch (memory order and progress):

* both kernels are PDL secondaries; neither reads anything before its
  ``griddepcontrol.wait`` (the FC1 kernel's first reads, the router logits
  and the activation, are the predecessor's outputs);
* the FC1 grid writes the routing record and the BF16 intermediate into the
  per-device scratch after its wait; the FC2 grid reads them after its own
  wait, which returns once the FC1 grid has completed and its writes are
  visible; the FC1 grid triggers its dependents after its tiles (or, with
  flag bit 4 ``FLAG_EARLY_TRIGGER``, after the routing), which only lets the
  FC2 CTAs be scheduled earlier;
* the scratch of call ``i`` is rewritten by the FC1 grid of call ``i + 1``
  only after that grid's wait, i.e. after its predecessor has completed; every
  kernel between the two calls on the stream either is launched without
  programmatic overlap or waits for its own predecessor before completing, so
  that predecessor's completion implies the completion of call ``i``'s FC2
  grid;
* the optional L2 prefetch of FC2 weights reads immutable model parameters;
* no CTA waits for another CTA of its grid (no spin, no grid barrier), so
  progress does not depend on co-residency with the shared expert's kernels.
"""

import os
from typing import Dict, Optional, Tuple

import torch

from ... import native_ops

HIDDEN = 6144
INTER = 256
NUM_EXPERTS = 256
TOP_K = 8
MAX_ROWS = 8
K_CHUNK = 64
SCRATCH_H_ELEMS = 64 * MAX_ROWS * INTER
ROUTE_WORDS = 1 + 64 + 64 + 64

FLAG_PDL = 1
FLAG_PREFETCH_FC2 = 2
FLAG_EARLY_TRIGGER = 4
FLAG_FC1_PAIRS = 8
FLAG_FC2_PAIRS = 16
FLAG_DEEP_RING = 32
FLAG_TMA = 64
FLAG_L2_PREFETCH = 128
FLAG_PREFETCH_FC2_EARLY = 256
FLAG_FC1_TRIPLES = 4096
FLAG_EARLY_TRIGGER_TRIPLES = 8192
FLAG_FC1_EVICT_FIRST = 32768
FLAG_STREAM_X = 65536
FLAG_OPTIONS = (
    FLAG_PREFETCH_FC2
    | FLAG_EARLY_TRIGGER
    | FLAG_FC1_PAIRS
    | FLAG_FC2_PAIRS
    | FLAG_DEEP_RING
    | FLAG_TMA
    | FLAG_L2_PREFETCH
    | FLAG_PREFETCH_FC2_EARLY
    | FLAG_FC1_TRIPLES
    | FLAG_EARLY_TRIGGER_TRIPLES
    | FLAG_FC1_EVICT_FIRST
    | FLAG_STREAM_X
)
# Shared memory of the FC1 kernel: the activation rows plus a weight ring per warp (2 KB chunks); the deep ring
# applies to a call only when it fits this budget (the static routing state takes about 3 KB of the 227 KB).
FC1_RING_CHUNKS_DEEP = 20
FC1_DYNAMIC_SMEM_BUDGET = 220 * 1024


# Launch configuration (``launch_config``): (ctas, warps) of each kernel and extra flag bits (2: FC2 weights prefetched
# into L2 by the FC1 warps, 4: early dependent trigger, 8: FC1 tile pairs, 16: FC2 expert pairs, 32: deeper weight
# rings, dropped for a call whose FC1 shared memory would not fit, 64: TMA-fed kernels, where "ctas" is the persistent
# grid size and "warps" the number of MMA warps (FC2 also 3), each owning a 16-row chain (two with bit 8), next to one
# producer warp; with them bit 2 is issued by the FC1 producer after its last weight copy, 16 gives every FC2 warp two
# experts per step, 128 adds L2 prefetches of the FC1 weights ahead of the ring, 256 moves the FC2 prefetch to right
# after routing, and bit 32 does not apply; 4096 (4 FC1 MMA warps): a call whose FC1 tiles exceed the grid runs in one
# round of three 64-row sub-tiles per CTA when they fit, chosen on the device from the routing; 8192: the early
# trigger for calls on that path only; 32768: FC1 weight loads with an L2 evict-first policy; 65536 (4 FC1 MMA warps
# with bit 8): the activation streamed through the FC1 ring with the weights instead of kept in shared memory, so the
# ring is deeper).
# Each value is a pair (calls with one row, calls with more rows). None of them changes a bit of the output.
# (one row, more rows): TMA-fed kernels. One row: 64 FC1 CTAs (64-row tiles, one per CTA), which leaves SMs to the
# shared expert on the auxiliary stream, and the FC2 weights prefetched into L2 behind the last FC1 copy. More rows:
# 128-row FC1 tiles (two chains per MMA warp) on up to 152 CTAs, one round for up to 38 distinct experts; above that
# the triples path (three 64-row sub-tiles per CTA, one round up to 57 distinct experts) with the early trigger, so
# the FC2 grid takes the SMs FC1 leaves and is resident when FC1 ends (calls on the regular path trigger after their
# tiles, which leaves those SMs to the shared expert); FC1 weights (read once per call) are loaded L2 evict-first.
DEFAULT_FC1 = ((64, 4), (152, 4))
DEFAULT_FC2 = ((128, 3), (128, 3))
DEFAULT_FLAGS = (
    FLAG_TMA | FLAG_PREFETCH_FC2,
    FLAG_TMA
    | FLAG_FC1_PAIRS
    | FLAG_FC1_TRIPLES
    | FLAG_EARLY_TRIGGER_TRIPLES
    | FLAG_FC1_EVICT_FIRST,
)


def _pdl_enabled() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0"


def launch_config(rows: int = 1) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """(FC1 ctas, warps), (FC2 ctas, warps) and the flag bits for calls with ``rows`` rows
    (``DEFAULT_FC1``, ``DEFAULT_FC2``, ``DEFAULT_FLAGS``)."""
    pick = 0 if rows == 1 else 1
    flags = DEFAULT_FLAGS[pick] & FLAG_OPTIONS
    if _pdl_enabled():
        flags |= FLAG_PDL
    fc1 = DEFAULT_FC1[pick]
    # (The TMA-fed kernels choose their ring depth in the launcher.)
    if (
        flags & FLAG_DEEP_RING
        and not flags & FLAG_TMA
        and rows * HIDDEN * 2 + fc1[1] * FC1_RING_CHUNKS_DEEP * 2048 > FC1_DYNAMIC_SMEM_BUDGET
    ):
        flags &= ~FLAG_DEEP_RING
    fc2 = DEFAULT_FC2[pick]
    if fc1[1] == 3 or (fc2[1] == 3 and not flags & FLAG_TMA):
        raise ValueError("3 warps per CTA exist only for the TMA-fed FC2 kernel")
    return fc1, fc2, flags


class _Scratch:
    """Per-device scratch of the two kernels (routing record and intermediate)."""

    def __init__(self, device: torch.device):
        self.h = torch.empty(SCRATCH_H_ELEMS, dtype=torch.bfloat16, device=device)
        self.route = torch.empty(ROUTE_WORDS, dtype=torch.int32, device=device)


# One scratch per device, shared by every draft_routed_moe call on it: correct only while those
# calls are ordered on one stream (the MTP layer's draft steps run in sequence on the model's main
# stream, eagerly and in every captured graph, and graphs replay one at a time). Calls that may
# overlap on other streams would each need their own scratch.
_SCRATCH: Dict[int, _Scratch] = {}


def _scratch(device: torch.device, create: bool) -> Optional[_Scratch]:
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _SCRATCH and create:
        _SCRATCH[key] = _Scratch(torch.device("cuda", key))
    return _SCRATCH.get(key)


def resolve_owner(module) -> Tuple[Optional[torch.nn.Module], str]:
    """The leaf owning the BF16 expert parameters (``ConfigurableMoE.backend`` or the leaf)."""
    backend = getattr(module, "backend", None)
    owner = backend if isinstance(backend, torch.nn.Module) else module
    if not isinstance(owner, torch.nn.Module):
        return None, f"{type(module).__name__} is not a module"
    if "w3_w1_weight" not in owner._parameters or "w2_weight" not in owner._parameters:
        return None, f"{type(owner).__name__} owns no expert parameters"
    return owner, ""


def _wrapper_reason(wrapper) -> str:
    """Wrapper-level behaviour the draft_routed_moe call would skip; empty when none."""
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
    if groups != (TOP_K, 1, 1):
        return f"routing {groups}"
    scale = impl.routed_scaling_factor
    if (
        scale is None
        or not (float(scale) == float(scale))
        or float(scale) in (float("inf"), float("-inf"))
    ):
        return f"routed scaling {scale}"
    return ""


def _owner_reason(owner, device: torch.device) -> str:
    """Leaf-level eligibility: FlashInfer TRT-LLM-Gen BF16 leaf, TP shard with every expert local."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import BF16TRTLLMGenFusedMoEMethod
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.flashinfer_bf16 import (
        FlashinferTrtllmGenBf16Impl,
    )
    from tensorrt_llm._torch.utils import ActivationType

    if not isinstance(owner, FlashinferTrtllmGenBf16Impl):
        return f"{type(owner).__name__} is not the FlashInfer TRT-LLM-Gen BF16 leaf"
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
    if getattr(owner, "intermediate_size_per_partition", None) != INTER:
        return f"intermediate {getattr(owner, 'intermediate_size_per_partition', None)}"
    qm = getattr(owner, "quant_method", None)
    if not isinstance(qm, BF16TRTLLMGenFusedMoEMethod):
        return f"quant method {type(qm).__name__}"
    if not getattr(qm, "use_shuffled_weight", False) or getattr(qm, "block_k", None) != K_CHUNK:
        return "weights not shuffled into 64-element BlockMajorK blocks"
    from tensorrt_llm._torch.moe.fused_moe.quantization import (
        TRTLLM_GEN_WEIGHT_LAYOUT_BLOCK_MAJOR_K,
    )

    if getattr(qm, "weight_layout", None) != TRTLLM_GEN_WEIGHT_LAYOUT_BLOCK_MAJOR_K:
        return f"weight layout {getattr(qm, 'weight_layout', None)}"
    if getattr(owner, "_trtllm_gen_layout_transform_pending", False):
        return "weight layout transform pending"
    w13 = owner._parameters["w3_w1_weight"]
    w2 = owner._parameters["w2_weight"]
    if (
        tuple(w13.shape) != (NUM_EXPERTS, HIDDEN // K_CHUNK, 2 * INTER, K_CHUNK)
        or w13.dtype != torch.bfloat16
    ):
        return f"w3_w1 {tuple(w13.shape)} {w13.dtype}"
    if (
        tuple(w2.shape) != (NUM_EXPERTS, INTER // K_CHUNK, HIDDEN, K_CHUNK)
        or w2.dtype != torch.bfloat16
    ):
        return f"w2 {tuple(w2.shape)} {w2.dtype}"
    for t in (w13, w2):
        if not t.is_contiguous() or t.device != device:
            return "expert buffers not contiguous on the activation's device"
    return _routing_reason(getattr(owner, "routing_method", None))


def eligibility(
    module, routing_bias, x: torch.Tensor, router_logits: torch.Tensor, do_finalize: bool
) -> Tuple[bool, str]:
    """Decide draft_routed_moe support for this call from host-side state only."""
    if not do_finalize:
        return False, "unfinalized output requested"
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
    if not x.is_contiguous() or router_logits is None or not router_logits.is_contiguous():
        return False, "non-contiguous input"
    if (
        router_logits.dtype != torch.float32
        or tuple(router_logits.shape) != (rows, NUM_EXPERTS)
        or router_logits.device != x.device
    ):
        return False, f"logits {tuple(router_logits.shape)} {router_logits.dtype}"
    if torch.compiler.is_compiling():
        return False, "torch.compile tracing"
    owner, reason = resolve_owner(module)
    if owner is None:
        return False, reason
    if owner is not module:
        reason = _wrapper_reason(module)
        if reason:
            return False, reason
    reason = _owner_reason(owner, x.device)
    if reason:
        return False, reason
    native_bias = getattr(getattr(owner, "routing_method", None), "e_score_correction_bias", None)
    if (
        routing_bias is None
        or routing_bias.dtype != torch.bfloat16
        or tuple(routing_bias.shape) != (NUM_EXPERTS,)
        or not routing_bias.is_contiguous()
        or routing_bias.device != x.device
    ):
        return False, "router bias is not a contiguous BF16 [256] on the device"
    if native_bias is None or native_bias.data_ptr() != routing_bias.data_ptr():
        return False, "router bias is not the routing method's bias"
    if torch.cuda.is_current_stream_capturing() and _scratch(x.device, create=False) is None:
        # Allocating inside a capture would record the allocation into the graph; the model's eager warmup
        # forwards create the scratch before capture.
        return False, "not warmed up before CUDA graph capture"
    return True, ""


def run(owner, routing_bias, x, router_logits, debug_fc2=None, timeline=None) -> torch.Tensor:
    """Launch the two kernels (no eligibility check; ``owner`` is the BF16 leaf)."""
    (c1, w1), (c2, w2), flags = launch_config(x.shape[0])
    scratch = _scratch(x.device, create=True)
    # The kernels read the activation, the logits and the bias with 16-byte loads.
    if x.data_ptr() % 16 != 0:
        x = x.clone(memory_format=torch.contiguous_format)
    if router_logits.data_ptr() % 16 != 0:
        router_logits = router_logits.clone(memory_format=torch.contiguous_format)
    if routing_bias.data_ptr() % 16 != 0:
        routing_bias = routing_bias.clone(memory_format=torch.contiguous_format)
    rm = owner.routing_method
    return native_ops.draft_routed_moe(
        x,
        router_logits,
        routing_bias,
        owner._parameters["w3_w1_weight"],
        owner._parameters["w2_weight"],
        scratch.h,
        scratch.route,
        float(rm.routing_impl.routed_scaling_factor),
        c1,
        w1,
        c2,
        w2,
        flags,
        debug_fc2,
        timeline,
    )


def routed_output(
    module,
    routing_bias: torch.Tensor,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    do_finalize: bool,
) -> Optional[torch.Tensor]:
    """Entry point of the MTP routed-MoE call site; ``None`` means "use the native path".

    ``module`` is the served ``ConfigurableMoE`` wrapper (or the bare BF16 leaf), ``routing_bias`` the gate's
    ``e_score_correction_bias``. The decision is made before any device work.
    """
    from ... import dispatch

    ok, reason = eligibility(module, routing_bias, x, router_logits, do_finalize)
    dispatch.log_capability(
        "draft_routed_moe", f"rows={x.shape[0]} {reason or 'engaged'}", ok, reason
    )
    if not ok:
        return None
    owner, _ = resolve_owner(module)
    return run(owner, routing_bias, x, router_logits)
