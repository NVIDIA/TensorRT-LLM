# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), dep16: the target class and its model.

Serves generation with attention DP over 16 ranks (tp 16), the routed experts over EP16 on
MEGAMOE_CUTEDSL. ``CONSTANTS`` records the optimizations this code is built for (and is what
``K3_CONSTANTS`` declares).

The checkpoint is the vision-language model's; this target is text only. It builds no vision
tower (its checkpoint tensors are never read, ``weights.py``) and refuses multimodal input. The
text model ``KimiK3DefaultForCausalLM``:

* 93 decoder layers (``KimiK3DefaultDecoderLayer``): 69 KDA (Kimi Delta Attention, linear
  attention) layers and 24 MLA (absorbed MQA, NoPE) layers, per the 1-indexed
  ``linear_attn_config.kda_layers`` / ``full_attn_layers`` schedule.
* Layer 0 has a dense SiTU MLP; layers 1..92 the latent MoE (``K3MoERuntime``): 896 routed
  experts at top-16 (sigmoid + ``e_score_correction_bias`` routing, FP32 router GEMM), 2 shared
  experts, and the latent down / norm / up projections around the routed experts. The MoE layer
  hands its residual update to the next layer's pre-attention mix up to 64 tokens.
* Attention residuals (``_apply_attn_res``): before ``input_layernorm``, before
  ``post_attention_layernorm`` and at the model output each token mixes a bank of earlier
  sublayer outputs (the K3 PDL kernels up to 64 tokens); a snapshot is appended whenever
  ``layer_idx % attn_res_block_size == 0``.

Caching: per request, a KDA layer keeps a short-convolution window
``[3 * num_heads * head_dim, W]`` (bf16, sections ``[q | k | v]``) and a delta-rule recurrent
state ``[num_heads, head_dim, head_dim]`` (fp32, ``[H, V, K]``) in ``KimiK3HybridCacheManager``
(KV cache manager V2); an MLA layer uses the paged KV cache with one KV head of
``kv_lora_rank + qk_rope_head_dim`` (576).

Parallelism: attention is data-parallel (each rank owns its requests' tokens), the dense MLP and
the shared experts are replicated, the routed experts are the MegaMoE module's (EP over the 16
ranks, its own dispatch / combine). ``lm_head`` is vocab-parallel with a gather, so every rank
has the full logits. There is no Helix / context parallelism and no pipeline parallelism.

Speculative decoding: one-engine DSpark with the family's standalone MLA drafter and worker
(``dspark/``) on the fused spec tail (``spec_tail_fusion``); the decode head is not fused. KDA
and MLA verify several tokens per step and keep the accepted state. Chunked prefill continues a
request's KDA state and attends over its cached MLA prefix. KV cache block reuse is opt-in
(``kv_cache_config.enable_block_reuse``); periodic KDA state snapshots come from
``mamba_state_config.periodic_snapshot_interval``.

Engine settings the recipe carries, not this class: the model loader applies model defaults of the
checkpoint's built-in class before a Modeling V2 target is routed, so a target cannot set
``llm_args``. The dep16 generation recipe turns
``attention_dp_config.enable_kv_balanced_admission`` on, which runs this target's planner
(``K3Dep16KvBalancedAdmission``).

Checkpoint loading is in ``weights.py``.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import gc
import math
import weakref
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch import kimi_k3_cache_policy as _kimi_k3_cache_policy
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.attention.backends import AttentionMetadata, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
    FP4_MLA_K_RESIDUAL_DIM,
    _fp4_mla_attention_backend,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.layout import (
    _get_fp4_mla_global_scale,
    _get_fp4_mla_kv_cache_tensors,
    _validate_fp4_mla_kv_storage_shape,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.metadata import (
    _materialize_fp4_mla_device_page_table_for_forward,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops import BufferKind
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE, sm_reserve
from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
    megamoe_activation_sf_bytes_per_row,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, RuntimeHooks
from tensorrt_llm._torch.utils import (
    ActivationType,
    AuxStreamType,
    Fp4QuantizedTensor,
    get_model_extra_attrs,
    maybe_compiled_cat,
    maybe_compiled_copy_,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..common.kda_metadata import K3MambaMetadata, KDASnapshotPlan
from ..common.multi_stream import do_multi_stream
from ..common.parallel import TensorParallelMode
from ..common.weight_io import check_text_only
from ..dspark.drafter import K3DSparkPath
from ..dspark.shell import KimiK3SpecForCausalLM
from ..kernels.k3r_kda._kda_kernels import (
    PAD_SLOT_ID,
    K3KDAKernelDispatch,
    extract_transpose_prefill_slice,
    reset_recurrent_state_rows,
)
from ..kernels.k3r_mla_ctx import fp8_static_quant
from ..kernels.k3r_mla_ctx.mla_kv_fp8 import mla_kv_to_fp8
from ..ops.activation.sm107_mlp import GatedMlp, gated_activation, subgroup_mapping
from ..ops.attention import _fp4_mla_decode as fp4_mla_decode
from ..ops.attention import k3_fp4_mla_decode as _decode_op
from ..ops.attention import k3_mla_attention as mla_attn
from ..ops.attention import k3_mla_cache as mla_cache
from ..ops.attention.k3_context_fmha import k3_context_fmha
from ..ops.attention.k3_mla_state import K3MlaAttnState
from ..ops.attention.static_quantize_e4m3 import static_quantize_e4m3
from ..ops.cache.k3_sm107_cache_manager import KimiK3Sm107CacheManager
from ..ops.common.sm107_launch import log_constants
from ..ops.common.sm107_streams import fork_join
from ..ops.embedding.sm107_embedding import EmbeddingTable
from ..ops.gemm.k3_fp8w_lowm_gemm import k3_fp8w_lowm_gemm as k3_fp8w_lowm_gemm_op
from ..ops.gemm.k3_mxfp8_gemm import k3_mxfp8_gemm as k3_mxfp8_gemm_op
from ..ops.gemm.k3_mxfp8_gemm_fp8_out import k3_mxfp8_gemm_fp8_out
from ..ops.gemm.sm107_linear import (
    Bf16Linear,
    Fp8BlockScaleLinear,
    fp8_block_scale_gemm,
    prepare_fp8_block_weight,
)
from ..ops.moe.k3_moe_residual_add import k3_moe_residual_add as k3_moe_residual_add_op
from ..ops.moe.k3r_experts import K3RoutedExperts, Route, SortedRouting
from ..ops.norm.attn_res_fwd import attn_res_fwd
from ..ops.norm.k3_attn_res_norm import k3_attn_res_norm
from ..ops.norm.rms_norm_gated_token_major import rms_norm_gated_token_major
from ..ops.norm.sm107_norm import RMSNormWeight, rms_norm
from ..ops.ssm.causal_conv1d_fwd import causal_conv1d_fwd
from ..ops.ssm.sm107_short_conv import ShortConvWeight


@dataclasses.dataclass(frozen=True)
class K3Sm107Constants:
    """What one Kimi K3 SM107 target runs. A field is a fixed property of the target."""

    # The speculative decode step's head in one kernel (``RuntimeHooks.decode_prologue``) and the
    # token embedding as a vectorized row copy.
    spec_head_fusion: bool
    # MoE (``K3MoERuntime``, ``_Fp8BlockScaleWeightReadLinear``).
    fuse_moe_residual: bool
    fp8w_lowm_gemm: bool
    ld_fused: bool
    shared_sm: int
    # FC12 die-major expert layout and schedule (the routed experts' kernel set).
    fc12_die_major: bool
    # The routed experts of a MEGAMOE_CUTEDSL layer: the target's MegaMoE module class, which
    # ``K3MoERuntime`` builds with keyword arguments (``num_experts``, ``top_k``, ``hidden_size``,
    # ``intermediate_size``, ``mapping``, ``layer_idx``, the two SiTU soft-caps,
    # ``max_num_tokens``, ``dtype``, ``skip_create_weights_in_init``); ``None`` where the target
    # runs no MegaMoE.
    megamoe_experts: Optional[Callable[..., Any]]
    router_exact_kda: bool
    # The MoE gate GEMM on BF16 operands with FP32 accumulation and output (else the FP32 GEMM of
    # the upcast operands; the two can pick different top-k experts at a tie).
    router_bf16_gate: bool
    # The BF16 shared and latent MoE MLP weights are read as block-scaled FP8 (quantized once after
    # loading).
    fp8_weight_read_moe_mlp: bool
    latent_ar_own_ws: bool
    # Attention residual (``_apply_attn_res`` and its fused forms, kernels/k3r_attn_res).
    attn_res_pdl: bool
    attn_res_pdl_max_m: int
    attn_res_mxfp8: bool
    attn_res_early_trigger: bool
    # KDA (``K3KDALinearAttention``).
    kda_prepad_prefill: bool
    kda_fused_conv_prefill: bool
    kda_l2_prefetch: bool
    # MLA and its FP4 generation decode (``K3MLAAttention``, ops/attention/_fp4_mla_decode.py).
    mla_q_norm_pdl: bool
    fp4_mla_page_plan: bool
    fp4_mla_reduce_pdl: bool
    # DSpark (dspark/, kernels/k3r_spec/spec_tail_kernels.py).
    spec_tail_fusion: bool
    dspark_tail_v2: bool
    # The target is a disaggregated context server: its drafter only writes the context KV the
    # generation servers' drafter reads (``context_kv_only`` of ``dspark/drafter.py``).
    dspark_context_kv_only: bool
    # Device memory, in bytes, the engine leaves outside the PyTorch memory cap
    # (``RuntimeHooks.torch_mem_safety_margin``).
    torch_mem_safety_margin: int


#: Values per NVFP4 block scale.
SF_VEC_SIZE = 16
#: Wire format of the combine (the returned rows travel as BF16).
COMBINE_FORMAT = "bf16"
#: Largest top-k the kernel is run at.
MAX_TOP_K = 16
#: FC1 rows are stored as alternating groups of this many gate rows and up rows.
FC1_GATE_UP_INTERLEAVE = 16
#: Rungs of the per-rank token bucket ladder, by architecture (each rung is one compiled kernel).
BUCKET_RUNGS_RUBIN = (32, 64, 128, 192, 256, 384, 512, 1024, 4096)
BUCKET_RUNGS_BLACKWELL = (256, 1024, 4096)

# Ladders whose buckets have been primed in this process, by layout.
_PRIMED_LADDERS: set = set()


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def fc1_sf_elements(hidden_size: int, intermediate_size: int) -> int:
    """E4M3 block scales of one expert's FC1 in the kernel's blocked layout (rows padded to 128,
    scale columns to 4)."""
    return _align_up(2 * intermediate_size, 128) * _align_up(hidden_size // SF_VEC_SIZE, 4)


def fc2_sf_elements(hidden_size: int, intermediate_size: int) -> int:
    """E4M3 block scales of one expert's FC2 in the kernel's blocked layout."""
    return _align_up(hidden_size, 128) * _align_up(intermediate_size // SF_VEC_SIZE, 4)


def resolve_buckets(max_num_tokens: int) -> List[int]:
    """The per-rank token bucket ladder: the architecture's rungs below ``max_num_tokens``, then
    ``max_num_tokens`` itself. A pure function of rank-identical inputs, so every EP rank has the
    same ladder."""
    full = int(max_num_tokens)
    if full <= 0:
        full = 4096
    rungs = BUCKET_RUNGS_RUBIN if get_sm_version() == 107 else BUCKET_RUNGS_BLACKWELL
    buckets = {b for b in rungs if b < full}
    buckets.add(full)
    return sorted(buckets)


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """One expert's FC1 rows in the kernel's order: ``[gate[0:16], up[0:16], gate[16:32], ...]``
    (``gate`` = w1, ``up`` = w3, ``[I, C]`` each; weights and their block scales alike)."""
    rows, cols = gate.shape
    pairs = rows // FC1_GATE_UP_INTERLEAVE
    return torch.stack(
        [
            gate.reshape(pairs, FC1_GATE_UP_INTERLEAVE, cols),
            up.reshape(pairs, FC1_GATE_UP_INTERLEAVE, cols),
        ],
        dim=1,
    ).reshape(2 * rows, cols)


def blocked_block_scales(sf: torch.Tensor, flat_size: int) -> torch.Tensor:
    """One expert's linear E4M3 block scales (``uint8 [rows, cols]``) in the kernel's blocked
    layout, flattened: zero-padded to 128 rows and 4 columns, then each 128x4 block in 32x4x4 atom
    order, zero-padded at the end to ``flat_size``."""
    rows, cols = sf.shape
    padded = F.pad(sf, (0, -cols % 4, 0, -rows % 128))
    blocked = (
        padded.view((rows + 127) // 128, 4, 32, (cols + 3) // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(-1)
    )
    if blocked.numel() == flat_size:
        return blocked
    out = torch.zeros(flat_size, dtype=blocked.dtype, device=blocked.device)
    out[: blocked.numel()] = blocked
    return out


def _as_nvfp4(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float4_e2m1fn_x2 else t.view(torch.float4_e2m1fn_x2)


def _as_fp8_sf(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float8_e4m3fn else t.view(torch.float8_e4m3fn)


def _leading_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    return t if t.shape[0] == rows else t[:rows]


def _resolve_ep_process_group(mapping: Mapping, ep_size: int):
    """The EP process group: the mapping's (Ray / DeviceMesh), else WORLD where it is the EP
    group."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Kimi K3 MegaMoE needs torch.distributed initialized before the layer is built"
        )
    try:
        return mapping.moe_ep_group_pg
    except (NotImplementedError, AttributeError):
        pass
    if dist.get_world_size() == ep_size:
        return dist.group.WORLD
    raise RuntimeError(
        f"Kimi K3 MegaMoE: the mapping has no EP process group and EP ({ep_size}) is not the "
        f"whole world ({dist.get_world_size()})"
    )


class K3MegaMoeNvfp4Experts(nn.Module):
    """One MoE layer's NVFP4 routed experts on this rank, in the MegaMoE kernel's layout.

    Holds experts ``[slot_start, slot_end)`` (``num_experts / ep_size`` contiguous experts;
    ``E`` below), ``H`` = hidden size, ``I`` = intermediate size:

    * ``mega_fc1_weight`` ``uint8 [E, 2 I, H / 2]``: packed E2M1 FC1 rows, gate (w1) and up (w3)
      interleaved in groups of 16 (:func:`interleave_gate_up`); ``mega_fc1_weight_sf``
      ``uint8 [E, align128(2 I) * align4(H / 16)]``: their E4M3 block scales, same row order,
      blocked (:func:`blocked_block_scales`).
    * ``mega_fc2_weight`` ``uint8 [E, H, I / 2]`` (the checkpoint bytes) and ``mega_fc2_weight_sf``
      ``uint8 [E, align128(H) * align4(I / 16)]`` (blocked).
    * ``fc31_input_scale``: the reciprocal of the largest checkpoint ``w1.input_scale`` of the
      rank's experts, the global scale ``x`` is quantized with; ``fc2_input_scale`` likewise for
      ``w2`` (not read by the kernel).
    * ``fc31_alpha`` ``[E]``: ``w1.weight_scale_2 / fc31_input_scale``; ``fc2_alpha`` ``[E]``:
      ``w2.weight_scale_2 * w2.input_scale``; ``fc1_norm_const`` ``[E]``: ``1 / w2.input_scale``,
      the global scale of the kernel's FP4 requantization of each expert's FC1 output.

    The checkpoint streams in expert by expert (:meth:`load_expert`, from any thread once
    :meth:`prepare_streaming_load` ran); :meth:`finalize` runs once the rank's last expert is in.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        ep_size: int,
        ep_rank: int,
    ) -> None:
        super().__init__()
        if num_experts % ep_size != 0:
            raise ValueError(f"{num_experts} experts do not split evenly over {ep_size} EP ranks")
        if hidden_size <= 0 or hidden_size % 64 != 0:
            raise ValueError(f"hidden_size {hidden_size} must be a positive multiple of 64")
        if intermediate_size <= 0 or intermediate_size % 64 != 0:
            raise ValueError(
                f"intermediate_size {intermediate_size} must be a positive multiple of 64"
            )
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size
        self.expand_intermediate_size_per_partition = 2 * intermediate_size
        self.ep_size, self.ep_rank = ep_size, ep_rank
        self.scaling_vector_size = SF_VEC_SIZE
        per_rank = num_experts // ep_size
        self.num_slots = num_experts
        self.expert_size_per_partition = per_rank
        self.slot_start = ep_rank * per_rank
        self.slot_end = self.slot_start + per_rank
        self.initial_local_expert_ids = list(range(self.slot_start, self.slot_end))
        self.initial_global_assignments = list(range(num_experts))
        self._staged: Optional[Dict] = None
        self._streamed_expert_slots: set = set()
        self._weights_created = False

    def create_weights(self) -> None:
        """Create the parameters (once; on the current default device)."""
        if self._weights_created:
            return
        e = self.expert_size_per_partition
        h, i = self.hidden_size, self.intermediate_size_per_partition

        def param(tensor: torch.Tensor) -> nn.Parameter:
            return nn.Parameter(tensor, requires_grad=False)

        self.mega_fc1_weight = param(torch.empty((e, 2 * i, h // 2), dtype=torch.uint8))
        self.mega_fc1_weight_sf = param(torch.empty((e, fc1_sf_elements(h, i)), dtype=torch.uint8))
        self.mega_fc2_weight = param(torch.empty((e, h, i // 2), dtype=torch.uint8))
        self.mega_fc2_weight_sf = param(torch.empty((e, fc2_sf_elements(h, i)), dtype=torch.uint8))
        self.fc31_input_scale = param(torch.tensor(1.0, dtype=torch.float32))
        self.fc2_input_scale = param(torch.tensor(1.0, dtype=torch.float32))
        self.fc31_alpha = param(torch.ones(e, dtype=torch.float32))
        self.fc2_alpha = param(torch.ones(e, dtype=torch.float32))
        self.fc1_norm_const = param(torch.ones(e, dtype=torch.float32))
        self._weights_created = True

    # ------------------------------------------------------------------
    # Loading: expert by expert, then one finalize per layer.
    # ------------------------------------------------------------------
    def prepare_streaming_load(self) -> None:
        """Create the per-layer staging the loader threads fill (before any of them runs)."""
        self._staged = {"input_scales": {}, "weight_scale_2": {}}
        self._streamed_expert_slots = set()

    def load_expert(
        self,
        *,
        global_expert_id: int,
        local_slot_id: int,
        w1_weight: torch.Tensor,
        w1_weight_scale: torch.Tensor,
        w1_weight_scale_2: torch.Tensor,
        w1_input_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_weight_scale_2: torch.Tensor,
        w2_input_scale: torch.Tensor,
        w3_weight: torch.Tensor,
        w3_weight_scale: torch.Tensor,
        w3_weight_scale_2: torch.Tensor,
        w3_input_scale: torch.Tensor,
    ) -> None:
        """Load one checkpoint expert (packed E2M1 ``uint8`` weights, E4M3 block scales, FP32
        global scales) into ``local_slot_id``, directly in the kernel's layout; the global scales
        are kept for :meth:`finalize`."""
        if not 0 <= local_slot_id < self.expert_size_per_partition:
            raise IndexError(
                f"local_slot_id={local_slot_id} is outside [0, {self.expert_size_per_partition})"
            )
        if global_expert_id != self.initial_local_expert_ids[local_slot_id]:
            raise ValueError(
                f"local slot {local_slot_id} holds expert "
                f"{self.initial_local_expert_ids[local_slot_id]}, got {global_expert_id}"
            )
        if self._staged is None:
            raise RuntimeError("prepare_streaming_load() must run before load_expert()")
        if local_slot_id in self._streamed_expert_slots:
            raise ValueError(f"NVFP4 local slot {local_slot_id} was loaded twice")
        h, i = self.hidden_size, self.intermediate_size_per_partition
        device = self.mega_fc1_weight.device

        def take(t: torch.Tensor, name: str, shape: Tuple[int, int], weight: bool) -> torch.Tensor:
            if weight and t.dtype != torch.uint8:
                raise TypeError(f"{name} must hold packed NVFP4 uint8, got {t.dtype}")
            if not weight and t.element_size() != 1:
                raise TypeError(f"{name} must hold one-byte E4M3 block scales, got {t.dtype}")
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} has shape {tuple(t.shape)}, expected {shape}")
            return t.to(device).contiguous().view(torch.uint8)

        w1 = take(w1_weight, "w1_weight", (i, h // 2), True)
        w3 = take(w3_weight, "w3_weight", (i, h // 2), True)
        w2 = take(w2_weight, "w2_weight", (h, i // 2), True)
        w1_sf = take(w1_weight_scale, "w1_weight_scale", (i, h // SF_VEC_SIZE), False)
        w3_sf = take(w3_weight_scale, "w3_weight_scale", (i, h // SF_VEC_SIZE), False)
        w2_sf = take(w2_weight_scale, "w2_weight_scale", (h, i // SF_VEC_SIZE), False)

        self.mega_fc1_weight.data[local_slot_id].copy_(interleave_gate_up(w1, w3))
        self.mega_fc2_weight.data[local_slot_id].copy_(w2)
        self.mega_fc1_weight_sf.data[local_slot_id].copy_(
            blocked_block_scales(interleave_gate_up(w1_sf, w3_sf), self.mega_fc1_weight_sf.shape[1])
        )
        self.mega_fc2_weight_sf.data[local_slot_id].copy_(
            blocked_block_scales(w2_sf, self.mega_fc2_weight_sf.shape[1])
        )

        self._staged["weight_scale_2"][local_slot_id] = {
            "w1": w1_weight_scale_2,
            "w3": w3_weight_scale_2,
            "w2": w2_weight_scale_2,
        }
        self._staged["input_scales"][global_expert_id] = {
            "w1": w1_input_scale[...].reshape([]),
            "w3": w3_input_scale[...].reshape([]),
            "w2": w2_input_scale[...].reshape([]),
        }
        self._streamed_expert_slots.add(local_slot_id)

    def loaded_slots(self) -> set:
        return self._streamed_expert_slots

    def finalize(self) -> None:
        """After the last expert of the layer: the global input scales, the per-expert alphas and
        ``fc1_norm_const``."""
        staged, self._staged = self._staged, None
        if staged is None:
            return
        missing = sorted(set(range(self.expert_size_per_partition)) - self._streamed_expert_slots)
        if missing:
            raise RuntimeError(
                f"Kimi K3 MegaMoE finalize with local slots {missing[:10]} never loaded"
            )
        input_scales = staged["input_scales"]

        norm_const = self.fc1_norm_const.data
        norm_const.copy_(
            torch.stack(
                [
                    input_scales[e]["w2"]
                    .to(device=norm_const.device, dtype=torch.float32)
                    .reciprocal()
                    for e in self.initial_local_expert_ids
                ]
            )
        )

        fc31_values, fc2_values = [], []
        for expert_id, entry in input_scales.items():
            assert torch.allclose(entry["w1"], entry["w3"]), (
                f"Expert {expert_id}: w1_input_scale != w3_input_scale"
            )
            fc31_values.append(entry["w1"])
            fc2_values.append(entry["w2"])
        self.fc31_input_scale.data.copy_(torch.stack(fc31_values).max().reciprocal())
        self.fc2_input_scale.data.copy_(torch.stack(fc2_values).max().reciprocal())

        fc31_input_scale = self.fc31_input_scale.data
        for slot, scales in staged["weight_scale_2"].items():
            w1_ws2, w3_ws2 = scales["w1"], scales["w3"]
            if not torch.allclose(w1_ws2, w3_ws2):
                logger.warning(
                    f"w1_weight_scale_2 != w3_weight_scale_2 ({w1_ws2} != {w3_ws2}), "
                    "selecting the larger value. Accuracy may be affected."
                )
                w1_ws2 = torch.max(w1_ws2, w3_ws2)
            gate_ws2 = 1.0 / w1_ws2[...].reshape([])
            self.fc31_alpha.data[slot].copy_(1.0 / (fc31_input_scale * gate_ws2))
            expert_id = self.initial_local_expert_ids[slot]
            fc2_input_scale = (
                input_scales[expert_id]["w2"][...].reshape([]).to(dtype=torch.float32).reciprocal()
            )
            w2_ws2 = 1.0 / scales["w2"][...].reshape([])
            self.fc2_alpha.data[slot].copy_(1.0 / (fc2_input_scale * w2_ws2))


class _LaunchBuffers(NamedTuple):
    """One bucket's kernel buffers: the symmetric regions (peer-mapped through ``peer_offsets``)
    and this module's local ``topk_idx`` staging."""

    activation: torch.Tensor  # uint8 [max_T, H / 2]
    activation_sf: torch.Tensor  # uint8 [max_T, sf bytes per row]
    topk_weights: torch.Tensor  # fp32 [max_T, top_k]
    combine_output: torch.Tensor  # bf16 [max_T, 1, H]
    shared_workspace: torch.Tensor  # uint8
    peer_offsets: Tuple[int, ...]
    topk_idx: torch.Tensor  # [max_T, top_k], int32 on Rubin, int64 on Blackwell


class K3MegaMoE(nn.Module):
    """One MoE layer's routed experts on this rank as the MegaMoE kernel (see the module
    docstring).

    ``mapping``: the routed experts' mapping (attention DP over the TP group, ``moe_ep_size`` =
    ``tp_size`` > 1, ``moe_tp_size`` 1). ``max_num_tokens``: tokens per rank per forward, the top
    of the bucket ladder (``min(moe_max_num_tokens, max_num_tokens)`` of the engine).
    ``skip_create_weights_in_init``: leave :meth:`create_weights` (the symmetric-memory
    rendezvous of every bucket, collective over the EP group, then the parameters) to the caller.

    ``torch.distributed`` is initialized here under MPI where it is not yet (a NCCL WORLD group
    over the MPI world, which must be the EP group).
    """

    #: The output is the complete routed output of this rank's tokens (no reduction left).
    returns_global_routed_output = True

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        mapping: Mapping,
        layer_idx: int,
        situ_beta: float,
        situ_linear_beta: float,
        max_num_tokens: int,
        dtype: torch.dtype = torch.bfloat16,
        skip_create_weights_in_init: bool = False,
    ) -> None:
        super().__init__()
        sm = get_sm_version()
        if sm not in (100, 103, 107):
            raise RuntimeError(f"the MegaMoE kernel runs on SM100, SM103 or SM107, not SM{sm}")
        if mapping.moe_tp_size != 1 or mapping.moe_cluster_size != 1:
            raise NotImplementedError("Kimi K3 MegaMoE splits the experts by EP only")
        ep_size = mapping.moe_ep_size
        if ep_size <= 1:
            raise NotImplementedError("Kimi K3 MegaMoE runs over an EP group of several ranks")
        if not mapping.enable_attention_dp or mapping.tp_size != ep_size:
            raise NotImplementedError(
                "Kimi K3 MegaMoE takes the tokens of attention-DP ranks, one rank per EP rank"
            )
        if not 0 < top_k <= MAX_TOP_K:
            raise ValueError(f"top_k {top_k} is outside [1, {MAX_TOP_K}]")
        if situ_beta <= 0 or situ_linear_beta <= 0:
            raise ValueError("the SiTU soft-caps must be positive")
        if max_num_tokens <= 0:
            raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}")
        if dtype != torch.bfloat16:
            raise ValueError(f"Kimi K3 MegaMoE takes BF16 activations, not {dtype}")
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ep_size, self.ep_rank = ep_size, mapping.moe_ep_rank
        self.situ_beta, self.situ_linear_beta = float(situ_beta), float(situ_linear_beta)
        self.dtype = dtype
        self.max_num_tokens = int(max_num_tokens)
        self.all_reduce = None
        self.backend = K3MegaMoeNvfp4Experts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ep_size=ep_size,
            ep_rank=self.ep_rank,
        )
        self._buckets = resolve_buckets(self.max_num_tokens)
        self._topk_idx_dtype = torch.int32 if sm == 107 else torch.int64
        self._sf_bytes_per_row = megamoe_activation_sf_bytes_per_row(hidden_size)
        # bucket -> MegaMoeSymmMemProvider (filled by create_weights).
        self._symm_providers: Dict[int, object] = {}
        # Buckets whose kernel reads x, x_sf and the top-k in place at an exact token count.
        self._direct_input_buckets: set = set()
        # (bucket, inference mode) -> buffers; views made under inference mode cannot be updated
        # in place outside it.
        self._launch_buffers: Dict[Tuple[int, bool], _LaunchBuffers] = {}
        # bucket -> topk_idx staging, and the live rows its last staging wrote.
        self._topk_idx_staging: Dict[int, torch.Tensor] = {}
        self._last_staged_rows: Dict[int, int] = {}
        # Cute views of this module's weights and topk_idx staging, reused across launches.
        self._cute_weight_cache: Dict = {}
        self._weights_created = False

        from tensorrt_llm._torch.distributed.initialization import ensure_cutedsl_torch_distributed

        ensure_cutedsl_torch_distributed(ep_size)
        self._ep_pg = _resolve_ep_process_group(mapping, ep_size)
        # The kernel's barrier-reset fence runs on WORLD.
        if dist.get_world_size() != ep_size:
            raise ValueError(
                f"Kimi K3 MegaMoE needs torch.distributed WORLD == EP; got "
                f"WORLD={dist.get_world_size()}, EP={ep_size}"
            )
        if not skip_create_weights_in_init:
            self.create_weights()

    @property
    def buckets(self) -> List[int]:
        """The per-rank token bucket ladder."""
        return list(self._buckets)

    # ------------------------------------------------------------------
    # Construction-time collectives and parameters.
    # ------------------------------------------------------------------
    def create_weights(self) -> None:
        """Rendezvous the symmetric memory of every bucket (collective over the EP group: every
        rank calls this in the same order), then create the parameters. Once per module."""
        if self._weights_created:
            return
        self._alloc_symm_providers()
        self.backend.create_weights()
        self._weights_created = True

    def _alloc_symm_providers(self) -> None:
        from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op

        if not op.IS_MEGAMOE_OP_AVAILABLE:
            raise RuntimeError(
                f"the MegaMoE CuTe DSL op is unavailable: {op.MEGAMOE_OP_UNAVAILABLE_REASON}"
            )
        b = self.backend
        for max_t in self._buckets:
            shared_workspace_bytes, direct_inputs = op.query_megamoe_kernel_properties(
                world_size=self.ep_size,
                num_topk=self.top_k,
                num_experts_per_rank=b.expert_size_per_partition,
                hidden_size=self.hidden_size,
                intermediate_size_per_partition=self.intermediate_size,
                expand_intermediate_size_per_partition=2 * self.intermediate_size,
                max_tokens_per_rank=max_t,
                apply_topk_in_fc1=False,
                gate_up_clamp=None,
                swiglu_alpha=None,
                swiglu_beta=None,
                situ_beta=self.situ_beta,
                situ_linear_beta=self.situ_linear_beta,
                in_kernel_fc2_reduce=False,
                combine_format=COMBINE_FORMAT,
            )
            if direct_inputs:
                self._direct_input_buckets.add(max_t)
            self._symm_providers[max_t] = op.get_megamoe_symm_provider(
                process_group=self._ep_pg,
                world_size=self.ep_size,
                rank=self.ep_rank,
                hidden_size=self.hidden_size,
                max_tokens_per_rank=max_t,
                num_topk=self.top_k,
                output_dtype=self.dtype,
                combine_format=COMBINE_FORMAT,
                shared_workspace_bytes=shared_workspace_bytes,
            )
        if len(self._buckets) > 1 and self.ep_rank == 0:
            logger.info(
                "[K3MegaMoE] max_tokens_per_rank buckets=%s (one symmetric provider + kernel per "
                "bucket; per-launch bucket = smallest >= max(all_rank_num_tokens))",
                self._buckets,
            )

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        *,
        route: Optional[Route] = None,
        routing: Optional[SortedRouting] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        input_ready_event: Optional[torch.cuda.Event] = None,
    ) -> torch.Tensor:
        """The routed output ``[n, hidden]`` (``x.dtype``) of this rank's ``n`` tokens of ``x``
        (BF16 ``[M, hidden]``, ``M >= n``; rows past ``n`` are attention-DP padding).

        ``all_rank_num_tokens``: every EP rank's token count; ``n`` is this rank's entry and the
        largest entry picks the bucket. Without it (or with one entry) ``n`` is ``M`` (that entry)
        and the launch takes the top bucket.

        The top-k (``int32 [n, top_k]`` global expert ids, ``fp32`` weights): ``routing``'s
        leading ``n`` rows, else ``route(router_logits[:n], slot_start, local experts)``.

        ``input_ready_event``: ``x`` is written on another stream; joined first, before the
        routing.

        The result is a view of the bucket's shared combine region (see the module docstring).
        """
        if input_ready_event is not None:
            input_ready_event.wait()
        if isinstance(x, Fp4QuantizedTensor):
            raise NotImplementedError("Kimi K3 MegaMoE quantizes a BF16 input itself")
        if not self._weights_created:
            raise RuntimeError("create_weights() must run before the forward")
        output_dtype = x.dtype

        if all_rank_num_tokens is not None and len(all_rank_num_tokens) > 1:
            counts = [int(v) for v in all_rank_num_tokens]
            if len(counts) != self.ep_size:
                raise ValueError(
                    f"all_rank_num_tokens has {len(counts)} entries for EP size {self.ep_size}"
                )
            num_tokens = counts[self.ep_rank]
            bucket_hint: Optional[int] = max(counts)
        else:
            num_tokens = int(all_rank_num_tokens[0]) if all_rank_num_tokens else int(x.shape[0])
            bucket_hint = None
        if num_tokens > x.shape[0]:
            raise ValueError(f"{num_tokens} tokens on this rank but x has {x.shape[0]} rows")
        x = _leading_rows(x, num_tokens)

        if num_tokens > 0:
            if routing is None:
                if route is None or router_logits is None:
                    raise ValueError("the forward needs a routing, or a route and router logits")
                routing = route(
                    _leading_rows(router_logits, num_tokens),
                    self.backend.slot_start,
                    self.backend.expert_size_per_partition,
                )
            ids = _leading_rows(routing.token_selected_experts, num_tokens)
            scales = _leading_rows(routing.token_final_scales, num_tokens)
            if ids.dtype != torch.int32:
                ids = ids.to(torch.int32)
            if scales.dtype != torch.float32 or not scales.is_contiguous():
                scales = scales.to(torch.float32).contiguous()
        else:
            ids = torch.empty((0, self.top_k), dtype=torch.int32, device=x.device)
            scales = torch.empty((0, self.top_k), dtype=torch.float32, device=x.device)

        x_fp4, x_sf = self.quantize_input(x)
        return self._run(x_fp4, x_sf, ids, scales, num_tokens, bucket_hint, output_dtype)

    def quantize_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(x, x_sf)``: packed E2M1 ``uint8 [n, H / 2]`` and linear E4M3 block scales
        ``[n, H / 16]`` of ``x`` against ``fc31_input_scale``."""
        hidden = x.shape[1]
        x_bf16 = x.to(torch.bfloat16).contiguous()
        rows = x_bf16.shape[0]
        if rows == 0:
            return (
                torch.empty((0, hidden // 2), dtype=torch.uint8, device=x.device),
                torch.empty((0, self._sf_bytes_per_row), dtype=torch.uint8, device=x.device),
            )
        x_fp4, x_sf = fp4_quantize(x_bf16, self.backend.fc31_input_scale, SF_VEC_SIZE, False, False)
        return x_fp4, x_sf.view(rows, self._sf_bytes_per_row)

    def _select_bucket(self, hint: Optional[int]) -> int:
        """The smallest bucket that holds ``hint`` tokens (rank-identical), the top one without."""
        if hint is None or hint <= 0:
            return self._buckets[-1]
        for bucket in self._buckets:
            if hint <= bucket:
                return bucket
        raise RuntimeError(
            f"Kimi K3 MegaMoE: {hint} tokens per rank exceed the top bucket {self._buckets[-1]}"
        )

    def _run(
        self,
        x_fp4: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        bucket_hint: Optional[int],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Pick the bucket, prime the ladder in tuning mode, stage (or hand over) the inputs and
        launch."""
        max_t = self._select_bucket(bucket_hint)
        if num_tokens > max_t:
            raise RuntimeError(
                f"Kimi K3 MegaMoE got {num_tokens} tokens but the bucket holds {max_t} "
                f"(buckets={self._buckets})"
            )
        device = x_fp4.device
        if AutoTuner.get().is_tuning_mode:
            self._prime_ladder(max_t, device, output_dtype)

        bufs = self._acquire_buffers(max_t, device)
        direct_inputs = (
            num_tokens == max_t
            and max_t in self._direct_input_buckets
            and x_fp4.shape == bufs.activation.shape
            and x_sf.shape == bufs.activation_sf.shape
            and x_fp4.is_contiguous()
            and x_sf.is_contiguous()
        )
        direct_routing = (
            direct_inputs
            and topk_idx.shape == (max_t, self.top_k)
            and topk_weights.shape == (max_t, self.top_k)
            and topk_idx.dtype == self._topk_idx_dtype
            and topk_weights.dtype == torch.float32
            and topk_idx.device == device
            and topk_weights.device == device
            and topk_idx.is_contiguous()
            and topk_weights.is_contiguous()
            and topk_idx.data_ptr() % 16 == 0
            and topk_weights.data_ptr() % 16 == 0
        )
        if not direct_routing:
            self._stage_inputs(
                bufs, x_fp4, x_sf, topk_idx, topk_weights, num_tokens, not direct_inputs
            )
        return self._launch(
            bufs,
            activation=x_fp4 if direct_inputs else bufs.activation,
            activation_sf=x_sf if direct_inputs else bufs.activation_sf,
            topk_idx=topk_idx if direct_routing else bufs.topk_idx,
            topk_weights=topk_weights if direct_routing else bufs.topk_weights[:, : self.top_k],
            num_tokens=num_tokens,
            max_t=max_t,
            output_dtype=output_dtype,
            staged_inputs=not direct_inputs,
        )

    def _acquire_buffers(self, max_t: int, device: torch.device) -> _LaunchBuffers:
        key = (max_t, torch.is_inference_mode_enabled())
        bufs = self._launch_buffers.get(key)
        if bufs is None:
            topk_idx = self._topk_idx_staging.get(max_t)
            if topk_idx is None:
                # -1: the kernel's dispatch skips the rows past the live tokens.
                topk_idx = torch.full(
                    (max_t, self.top_k), -1, dtype=self._topk_idx_dtype, device=device
                )
                self._topk_idx_staging[max_t] = topk_idx
            regions = self._symm_providers[max_t].get_regions()
            bufs = _LaunchBuffers(
                activation=regions.activation,
                activation_sf=regions.activation_sf,
                topk_weights=regions.topk_weights,
                combine_output=regions.combine_output,
                shared_workspace=regions.shared_workspace,
                peer_offsets=regions.peer_offsets,
                topk_idx=topk_idx,
            )
            self._launch_buffers[key] = bufs
        return bufs

    def _stage_inputs(
        self,
        bufs: _LaunchBuffers,
        x_fp4: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor],
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        num_tokens: int,
        stage_activation: bool,
    ) -> None:
        """Copy the live rows into the bucket's buffers; ``topk_idx`` is ``-1`` and
        ``topk_weights`` 0 on every other row (the combine reads a weight for every
        ``(token, k)``)."""
        max_t = bufs.topk_idx.shape[0]
        last = self._last_staged_rows.get(max_t)
        if last is not None and last > num_tokens:
            bufs.topk_idx[num_tokens:last].fill_(-1)
        if num_tokens > 0:
            if stage_activation:
                bufs.activation[:num_tokens].copy_(x_fp4.view(torch.uint8), non_blocking=True)
                bufs.activation_sf[:num_tokens].copy_(x_sf.view(torch.uint8), non_blocking=True)
            bufs.topk_weights[:num_tokens, : self.top_k].copy_(topk_weights, non_blocking=True)
        if num_tokens < max_t:
            bufs.topk_weights[num_tokens:max_t, : self.top_k].zero_()
        if num_tokens > 0:
            bufs.topk_idx[:num_tokens].copy_(topk_idx, non_blocking=True)
        self._last_staged_rows[max_t] = num_tokens

    def _launch(
        self,
        bufs: _LaunchBuffers,
        *,
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        max_t: int,
        output_dtype: torch.dtype,
        staged_inputs: bool,
    ) -> torch.Tensor:
        """The kernel on ``max_t``'s buffers; ``staged_inputs``: the inputs are those buffers,
        whose cute views the bucket's provider keeps."""
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            run_cute_dsl_megamoe_nvfp4_blackwell,
            set_active_megamoe_cute_buffer_cache,
            set_active_megamoe_cute_weight_cache,
        )

        b = self.backend
        set_active_megamoe_cute_weight_cache(self._cute_weight_cache)
        provider = self._symm_providers.get(max_t) if staged_inputs else None
        set_active_megamoe_cute_buffer_cache(provider.cute_views if provider is not None else None)
        try:
            run_cute_dsl_megamoe_nvfp4_blackwell(
                activation=_as_nvfp4(activation),
                activation_sf=_as_fp8_sf(activation_sf),
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                fc1_weight=_as_nvfp4(b.mega_fc1_weight),
                fc1_weight_sf=_as_fp8_sf(b.mega_fc1_weight_sf),
                fc2_weight=_as_nvfp4(b.mega_fc2_weight),
                fc2_weight_sf=_as_fp8_sf(b.mega_fc2_weight_sf),
                fc1_alpha=b.fc31_alpha,
                fc2_alpha=b.fc2_alpha,
                fc1_norm_const=b.fc1_norm_const,
                combine_output=bufs.combine_output,
                shared_workspace=bufs.shared_workspace,
                world_size=self.ep_size,
                local_rank=self.ep_rank,
                num_topk=self.top_k,
                num_experts_per_rank=b.expert_size_per_partition,
                hidden_size=self.hidden_size,
                intermediate_size_per_partition=self.intermediate_size,
                expand_intermediate_size_per_partition=2 * self.intermediate_size,
                max_tokens_per_rank=int(max_t),
                peer_offsets=bufs.peer_offsets,
                apply_topk_in_fc1=False,
                gate_up_clamp=None,
                swiglu_alpha=None,
                swiglu_beta=None,
                act_alpha=self.situ_beta,
                act_beta=self.situ_linear_beta,
                in_kernel_fc2_reduce=False,
                combine_format=COMBINE_FORMAT,
                tactic_autotune=False,
                num_tokens=num_tokens,
                helper_expert_count=0,
                hot_expert_weight_ready_flags=None,
                hot_expert_weight_ready_generation=0,
            )
        finally:
            set_active_megamoe_cute_weight_cache(None)
            set_active_megamoe_cute_buffer_cache(None)
        if num_tokens == 0:
            return torch.empty(
                (0, self.hidden_size), dtype=output_dtype, device=bufs.combine_output.device
            )
        return bufs.combine_output[:num_tokens].squeeze(1).to(output_dtype)

    def _prime_ladder(self, live_t: int, device: torch.device, output_dtype: torch.dtype) -> None:
        """Launch every other bucket once with zero tokens (once per process and layout), so no
        bucket compiles or fences inside a serving iteration or a CUDA graph capture."""
        key = (
            tuple(self._buckets),
            self.ep_size,
            self.hidden_size,
            self.intermediate_size,
            self.backend.expert_size_per_partition,
            self.top_k,
            COMBINE_FORMAT,
            None,
            self.situ_beta,
            self.situ_linear_beta,
            False,
            str(device),
            output_dtype,
        )
        if key in _PRIMED_LADDERS:
            return
        primed = []
        for max_t in self._buckets:
            if max_t == live_t:
                continue
            bufs = self._acquire_buffers(max_t, device)
            self._stage_inputs(bufs, None, None, None, None, 0, False)
            self._launch(
                bufs,
                activation=bufs.activation,
                activation_sf=bufs.activation_sf,
                topk_idx=bufs.topk_idx,
                topk_weights=bufs.topk_weights[:, : self.top_k],
                num_tokens=0,
                max_t=max_t,
                output_dtype=output_dtype,
                staged_inputs=False,
            )
            primed.append(max_t)
        _PRIMED_LADDERS.add(key)
        if self.ep_rank == 0:
            logger.debug("[K3MegaMoE] bucket priming complete: count=%s", len(primed))


__all__ = [
    "BUCKET_RUNGS_BLACKWELL",
    "BUCKET_RUNGS_RUBIN",
    "K3MegaMoE",
    "K3MegaMoeNvfp4Experts",
    "blocked_block_scales",
    "fc1_sf_elements",
    "fc2_sf_elements",
    "interleave_gate_up",
    "resolve_buckets",
]


CONSTANTS = K3Sm107Constants(
    spec_head_fusion=False,
    fuse_moe_residual=True,
    fp8w_lowm_gemm=True,
    ld_fused=False,
    shared_sm=0,
    fc12_die_major=False,
    megamoe_experts=K3MegaMoE,
    router_exact_kda=False,
    router_bf16_gate=False,
    fp8_weight_read_moe_mlp=True,
    latent_ar_own_ws=False,
    attn_res_pdl=True,
    attn_res_pdl_max_m=64,
    attn_res_mxfp8=False,
    attn_res_early_trigger=False,
    kda_prepad_prefill=False,
    kda_fused_conv_prefill=False,
    kda_l2_prefetch=False,
    mla_q_norm_pdl=False,
    fp4_mla_page_plan=False,
    fp4_mla_reduce_pdl=False,
    spec_tail_fusion=True,
    dspark_tail_v2=False,
    dspark_context_kv_only=False,
    torch_mem_safety_margin=4 * 1024**3,
)


def _resolve_kimi_situ_betas(cfg: Any) -> tuple[float, float]:
    """Return the finite SiTu betas required by the routed-expert kernels."""
    config_situ_beta = getattr(cfg, "activation_situ_beta", None)
    situ_beta = 1.0 if config_situ_beta is None else config_situ_beta
    situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
    if situ_linear_beta is None:
        raise ValueError(
            "Kimi K3 routed SiTu experts require activation_situ_linear_beta; "
            "None means an identity linear branch that the fused kernels cannot represent."
        )
    if situ_beta <= 0 or situ_linear_beta <= 0:
        raise ValueError(
            f"Kimi K3 SiTu betas must be positive; got {situ_beta} and {situ_linear_beta}."
        )
    return float(situ_beta), float(situ_linear_beta)


# ---------------------------------------------------------------------------
# Config helpers.
# ---------------------------------------------------------------------------


def _get_text_config(pretrained_config: "PretrainedConfig"):
    """Return the Kimi text config, unwrapping a composite kimi_k3 config."""
    if getattr(pretrained_config, "model_type", None) == "kimi_k3" or (
        not hasattr(pretrained_config, "linear_attn_config")
        and hasattr(pretrained_config, "text_config")
    ):
        return pretrained_config.text_config
    return pretrained_config


def _is_kda_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["kda_layers"]


def _is_mla_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["full_attn_layers"]


# ---------------------------------------------------------------------------
# attn_res: per-token snapshot mixing (HF `_apply_attn_res`).
# ---------------------------------------------------------------------------

# Fused attention-residual kernels (every Kimi K3 recipe runs the "split" topology):
# * up to _FUSED_ATTN_RES_MAX_TOKENS tokens the per-token C++ kernels (one CTA for N <= 4, one 8-CTA
#   cluster per token for N >= 5): trtllm::attn_res_rmsnorm_fwd / attn_res_add_rmsnorm_fwd;
# * above it the persistent kernel (one CTA per SM looping over tokens),
#   trtllm::attn_res_add_rmsnorm_persistent_fwd, where it implements the shape (H == 7168 and
#   2 <= N <= 9);
# * elsewhere trtllm::attn_res_fwd and the RMSNorm, or the FP32 reference past that op's contract.
# The K3 PDL kernels take precedence up to ``_PDL_MAX_M`` tokens.
_FUSED_ATTN_RES_MAX_TOKENS = 32


def _use_persistent_attn_res(M: int, H: int, N: int) -> bool:
    """Whether the persistent kernel serves this mix: H == 7168, 2 <= N <= 9 and more than
    ``_FUSED_ATTN_RES_MAX_TOKENS`` tokens (its grid is sized by the SM count, not the token
    count)."""
    return H == 7168 and 2 <= N <= 9 and M > _FUSED_ATTN_RES_MAX_TOKENS


# Token limit of trtllm::attn_res_fwd (cpp/tensorrt_llm/thop/attnResOp.cpp).
_ATTN_RES_OP_MAX_TOKENS = 16384


def _apply_attn_res_fused(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: RMSNormWeight
) -> Optional[torch.Tensor]:
    """Fused attn_res via the in-tree ``trtllm::attn_res_fwd`` op.

    Returns ``None`` when the call falls outside the fused kernel's
    contract (dtype/shape/arch) so the caller can use the exact fp32 reference
    instead. ``block_residual`` is kept in the kernel-native ``[K, M, H]``
    layout. Candidate order matches the reference: snapshots first, the
    running prefix sum last.
    """
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    if K + 1 > 12 or not (4096 <= H <= 8192 and H % 1024 == 0):
        return None
    attn_res_op = attn_res_fwd
    proj_w = proj.weight.reshape(-1).to(torch.bfloat16).contiguous()
    norm_w = norm.weight.to(torch.bfloat16).contiguous()

    def run(prefix, block):
        t = prefix.shape[0]
        output, _rsigma, _probs, _logits = attn_res_op(
            prefix.reshape(t, 1, H).contiguous(),
            block.reshape(K, t, 1, H).contiguous(),
            proj_w,
            norm_w,
            float(norm.eps),
        )
        return output.reshape(t, H)

    if M <= _ATTN_RES_OP_MAX_TOKENS:
        return run(prefix_sum, block_residual)
    # The op takes at most _ATTN_RES_OP_MAX_TOKENS tokens per call and is
    # per-token, so longer prefill chunks run in slices; the fp32 reference
    # below would otherwise cost seconds per iteration at 24k+ tokens.
    out = torch.empty_like(prefix_sum)
    for lo in range(0, M, _ATTN_RES_OP_MAX_TOKENS):
        hi = min(M, lo + _ATTN_RES_OP_MAX_TOKENS)
        out[lo:hi] = run(prefix_sum[lo:hi], block_residual[:, lo:hi])
    return out


def _note_attn_res_fusion(site: str, fused: bool, M: int, H: int, N: int) -> None:
    """Report whether a fused kernel served the mix, once per shape (a shape gate that rejects
    the call is otherwise silent). Emitted at debug level: it is once per distinct shape, not once
    per process, so it is a diagnostic rather than a summary.
    """
    logger.debug_once(
        f"Kimi K3 attn-res fusion [{site}]: "
        f"{'FUSED' if fused else 'fallback'} (M={M}, H={H}, N={N})",
        key=f"kimi_k3_attn_res_fusion_{site}_{fused}_{M}_{H}_{N}",
    )


# The K3 PDL kernels: up to ``_PDL_MAX_M`` tokens, the fused attn_res(+add)
# +RMSNorm runs as the k3_attn_res_norm entry (kernels/k3r_attn_res/k3_attn_res.py):
# a CTA cluster per token that loads the weights and snapshot rows and reduces their statistics
# before griddepcontrol.wait, i.e. under the kernel in front of it. It takes precedence over the
# persistent topology at those token counts. The updated prefix sum is bit-identical; the
# normalized output can differ from the C++ per-token kernels by one BF16 rounding step in a small
# fraction of the elements. The dependent grid is not released early (``early_trigger`` off).
# It also leaves the MoE residual update residual + bf16(routed + shared) of the BF16-add path
# (kimi_k3_moe_residual) to the next layer's pre-attention mix, which computes it bit-identically
# (k3_attn_res_moe_add_rmsnorm) behind the latent-up GEMM.
_PDL_MAX_M = 64


class _DeferredMoEResidual(NamedTuple):
    """A MoE layer's ``residual + bf16(routed + shared)`` that the next layer fuses or materializes."""

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor

    def materialize(self) -> torch.Tensor:
        return k3_moe_residual_add_op(self.routed, self.shared, self.residual)


def _defer_moe_residual(num_tokens: int) -> bool:
    """Whether a MoE layer leaves its BF16 residual update to the next layer."""
    return num_tokens <= _PDL_MAX_M


def _materialize_prefix_sum(hidden_states):
    if isinstance(hidden_states, _DeferredMoEResidual):
        return hidden_states.materialize()
    return hidden_states


def _k3_attn_res_pdl(
    prefix_sum: torch.Tensor,
    addend: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
):
    """``k3_attn_res_norm`` (with ``addend`` as the one addend) when applicable, else ``None``."""
    if prefix_sum.shape[0] > _PDL_MAX_M:
        return None
    from ..kernels.k3r_attn_res import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    addends = () if addend is None else (addend,)
    if not k3_attn_res.can_use_k3_attn_res(prefix_sum, addends, block_residual, *weights):
        return None
    eps = float(norm.eps)
    out_eps = float(output_norm.eps)
    return k3_attn_res_norm(
        prefix_sum,
        addends,
        block_residual,
        *weights,
        eps,
        out_eps,
        early_trigger=False,
    )


def _k3_attn_res_pdl_moe_add(
    deferred: _DeferredMoEResidual,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(prefix_sum, normed mix)`` of a deferred MoE residual update, or ``None``."""
    residual, routed, shared = deferred
    if residual.shape[0] > _PDL_MAX_M:
        return None
    from ..kernels.k3r_attn_res import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    if not k3_attn_res.can_use_k3_attn_res(residual, (routed, shared), block_residual, *weights):
        return None
    M, H = residual.shape
    _note_attn_res_fusion("moe_add+attn_res+norm/pdl", True, M, H, int(block_residual.shape[0]) + 1)
    return k3_attn_res_norm(
        residual,
        (routed, shared),
        block_residual,
        *weights,
        float(norm.eps),
        float(output_norm.eps),
        early_trigger=False,
    )


def _apply_attn_res_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Optional[torch.Tensor]:
    """Fuse attention-residual mixing with its immediately following norm."""
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # The fused path is taken for M <= _FUSED_ATTN_RES_MAX_TOKENS, H == 7168 and
    # N <= 12, which is the measured window; larger token counts have not been
    # measured and fall back to the unfused add + attn_res_fwd + RMSNorm path.
    pdl_out = _k3_attn_res_pdl(prefix_sum, None, block_residual, proj, norm, output_norm)
    if pdl_out is not None:
        _note_attn_res_fusion("attn_res+norm/pdl", True, M, H, N)
        return pdl_out
    if _use_persistent_attn_res(M, H, N):
        _, output = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
            prefix_sum.reshape(M, 1, H).contiguous(),
            None,
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            float(output_norm.eps),
            sm_reserve(),
        )
        _note_attn_res_fusion("attn_res+norm/persistent", True, M, H, N)
        return output.reshape(M, H)

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("attn_res+norm", False, M, H, N)
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    output = torch.ops.trtllm.attn_res_rmsnorm_fwd(
        layer_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        float(output_norm.eps),
    )
    _note_attn_res_fusion("attn_res+norm", True, M, H, N)
    return output.reshape(M, H)


def _apply_attn_res_add_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fuse ``prefix_sum + addend``, attention-residual, and trailing norm.

    The production residual add produces a BF16 tensor that remains live across
    the following MLP. The kernel therefore returns that materialized,
    BF16-rounded prefix sum alongside the normalized attention-residual output,
    while avoiding a separate add launch and a re-read of the intermediate by
    attention-residual selection.
    """
    if (
        prefix_sum.dtype is not torch.bfloat16
        or addend.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not addend.is_cuda
        or not block_residual.is_cuda
        or prefix_sum.shape != addend.shape
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # Same measured window as _apply_attn_res_rmsnorm_fused above.
    pdl_out = _k3_attn_res_pdl(prefix_sum, addend, block_residual, proj, norm, output_norm)
    if pdl_out is not None:
        _note_attn_res_fusion("add+attn_res+norm/pdl", True, M, H, N)
        return pdl_out
    if _use_persistent_attn_res(M, H, N):
        updated_prefix_sum, output = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
            prefix_sum.reshape(M, 1, H).contiguous(),
            addend.reshape(M, 1, H).contiguous(),
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            float(output_norm.eps),
            sm_reserve(),
        )
        _note_attn_res_fusion("add+attn_res+norm/persistent", True, M, H, N)
        return updated_prefix_sum.reshape(M, H), output.reshape(M, H)

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("add+attn_res+norm", False, M, H, N)
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    addend_kernel = addend.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    updated_prefix_sum, output = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        layer_kernel,
        addend_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        float(output_norm.eps),
    )
    _note_attn_res_fusion("add+attn_res+norm", True, M, H, N)
    return updated_prefix_sum.reshape(M, H), output.reshape(M, H)


def _apply_attn_res(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: RMSNormWeight
) -> torch.Tensor:
    """Exact port of HF ``modeling_kimi._apply_attn_res`` (fp32 math).

    prefix_sum:     ``[num_tokens, hidden_size]``
    block_residual: ``[num_snapshots, num_tokens, hidden_size]``

    Inputs fitting the fused kernel's contract dispatch directly to the in-tree
    ``trtllm::attn_res_fwd`` op. Only the fallback boundary restores the HF
    ``[M, K, H]`` layout.
    """
    fused = _apply_attn_res_fused(prefix_sum, block_residual, proj, norm)
    if fused is not None:
        return fused
    # The fp32 reference below is orders of magnitude slower at prefill
    # sizes; say so once per shape rather than fall back silently.
    _note_attn_res_fusion(
        "attn_res",
        False,
        prefix_sum.shape[0],
        prefix_sum.shape[1],
        int(block_residual.shape[0]) + 1,
    )
    block_residual_hf = block_residual.transpose(0, 1)
    v = torch.cat((block_residual_hf, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.eps)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


def _apply_attn_res_and_rmsnorm(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> torch.Tensor:
    """Apply attention-residual selection and the next RMSNorm."""
    fused = _apply_attn_res_rmsnorm_fused(prefix_sum, block_residual, proj, norm, output_norm)
    if fused is not None:
        return fused
    mixed = _apply_attn_res(prefix_sum, block_residual, proj, norm)
    return rms_norm(mixed, output_norm.weight, output_norm.eps)


def _apply_attn_res_add_and_rmsnorm(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm."""
    fused = _apply_attn_res_add_rmsnorm_fused(
        prefix_sum, addend, block_residual, proj, norm, output_norm
    )
    if fused is not None:
        return fused
    updated_prefix_sum = prefix_sum + addend
    return updated_prefix_sum, _apply_attn_res_and_rmsnorm(
        updated_prefix_sum, block_residual, proj, norm, output_norm
    )


# Prefill chunk of the kda_prefill entry.
_PREFILL_CHUNK = 64
# Single-token decode: [f_a | b] -> f_b runs on the projection aux stream next to the QKVG GEMM up
# to this many rows.
_KDA_BFA_MULTISTREAM_MAX_ROWS = 128

_FP8_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
_QKVG_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "g_proj")


def _o_proj_takes_mxfp8(o_proj: nn.Module, num_rows: int) -> bool:
    """Whether ``o_proj`` takes ``num_rows`` rows as the Rubin MXFP8 operand of the verify
    kernel's epilogue (``forward_prequantized``) rather than BF16 (the K3 FP8 weight-read
    projection decides per row count, see ``takes_prequantized_input``)."""
    takes = getattr(o_proj, "takes_prequantized_input", None)
    if takes is not None:
        return bool(takes(num_rows))
    return bool(getattr(o_proj, "supports_prequantized_input", False))


class _KdaPrequantizedCore(NamedTuple):
    """MXFP8 KDA core emitted by the fused verify epilogue.

    ``activation`` is the ``[tokens, H * V]`` E4M3 post-o_norm core and
    ``scale`` its UE8M0 1x128 scales in the flat uint8 R128c4 layout that
    ``o_proj.forward_prequantized`` consumes.
    """

    activation: torch.Tensor
    scale: torch.Tensor


@dataclass
class _KDAPrefillState:
    """The fp32 state rows the ``kda_prefill`` entry updates in place: the pool itself
    (``kernel_pool`` / ``kernel_indices``), or ``staged_state``, a dense fp32 copy of the
    addressed rows (plus the snapshot rows of a plan) that is written back afterwards."""

    kernel_pool: torch.Tensor
    kernel_indices: torch.Tensor
    staged_state: torch.Tensor | None
    # With a mid-chunk snapshot plan: the snapshot rows of ``kernel_pool`` (the pool's own slots,
    # or the rows after the sequences' in the staged copy).
    snapshot_rows: torch.Tensor | None = None


def _stage_state_rows(ssm_pool: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
    """Dense fp32 copy of the addressed recurrent-state rows (``_writeback_state_rows`` rounds it
    back into the pool)."""
    return ssm_pool.index_select(0, slot_indices.long()).float()


def _writeback_state_rows(
    ssm_pool: torch.Tensor, slot_indices: torch.Tensor, rows: torch.Tensor
) -> None:
    """Scatter dense state rows back into the pool, rounded to the pool dtype."""
    ssm_pool.index_copy_(0, slot_indices.long(), rows.to(ssm_pool.dtype))


def _stage_state_rows_with_snapshot_rows(
    ssm_pool: torch.Tensor, slot_indices: torch.Tensor, num_snapshots: int
) -> torch.Tensor:
    """``_stage_state_rows`` plus ``num_snapshots`` uninitialized rows after them.

    The prefill kernel writes mid-chunk snapshots into the extra rows.
    """
    num_rows = slot_indices.shape[0]
    staged = torch.empty(
        (num_rows + num_snapshots, *ssm_pool.shape[1:]),
        dtype=torch.float32,
        device=ssm_pool.device,
    )
    staged[:num_rows].copy_(ssm_pool.index_select(0, slot_indices.long()))
    return staged


def _snapshot_conv_windows(
    conv_input: torch.Tensor,
    conv_pool: torch.Tensor,
    snapshot: "KDASnapshotPlan",
) -> torch.Tensor:
    """The raw convolution window ending before each snapshot point, ``[n, 3D, W - 1]`` in
    ``conv_pool``'s dtype.

    ``conv_input`` is the pre-convolution projection, ``[3D, T]``; ``conv_pool`` is
    ``[slots, 3D, W - 1]``. A snapshot point is at least one KDA chunk into its sequence's step,
    so the window never reaches the sequence's previous convolution state.
    """
    window = conv_pool.shape[-1]
    offsets = torch.arange(-window, 0, device=conv_input.device)
    token_indices = (snapshot.token_indices[:, None] + offsets).reshape(-1)
    columns = conv_input.index_select(1, token_indices)
    windows = columns.view(-1, snapshot.num_snapshots, window).transpose(0, 1)
    return windows.to(conv_pool.dtype)


def _swap_inforward_conv_windows(
    conv_pool: torch.Tensor,
    stage_conv: torch.Tensor,
    snapshot: "KDASnapshotPlan",
    windows: torch.Tensor,
) -> None:
    """Move each snapshotted sequence's chunk-end convolution window from its live slot
    (``snapshot.slots``) to its staging row and put the window ending at the snapshot point
    (``windows``, :func:`_snapshot_conv_windows`) in the live slot."""
    stage_conv.index_copy_(0, snapshot.stage_rows, conv_pool.index_select(0, snapshot.slots))
    conv_pool.index_copy_(0, snapshot.slots, windows)


def _merge_row_weights(modules: Tuple[nn.Module, ...], pad_rows_to: int = 1) -> torch.Tensor:
    """Concatenate the ``weight`` rows of ``modules`` (zero rows up to a multiple of
    ``pad_rows_to``) and repoint each module's ``weight`` to its row view of the result."""
    weights = [module.weight.data for module in modules]
    padding = (-sum(weight.shape[0] for weight in weights)) % pad_rows_to
    if padding:
        weights.append(weights[0].new_zeros((padding, weights[0].shape[1])))
    fused = torch.cat(weights, dim=0).contiguous()
    offset = 0
    for module in modules:
        rows = module.weight.shape[0]
        module.weight.data = fused[offset : offset + rows]
        offset += rows
    return fused


def _extract_kda_extra_attrs(layer_idx: str):
    """Resolve the live attention metadata and KDA module for ``layer_idx``."""
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata")
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, AttentionMetadata), "Metadata must be AttentionMetadata"

    kda_layers = extra_attrs.get("kda_layers")
    assert kda_layers is not None, "KDA layer registry is not set"
    kda_layer_ref = kda_layers.get(layer_idx)
    assert kda_layer_ref is not None, f"Cannot find KDA layer for layer {layer_idx}"
    kda_layer = kda_layer_ref()
    assert isinstance(kda_layer, K3KDALinearAttention), "KDA layer must be K3KDALinearAttention"
    return metadata, kda_layer


def kda_core_inplace(hidden_states: torch.Tensor, layer_idx: str, output: torch.Tensor) -> None:
    """Run the metadata-dependent KDA core and write it into ``output``."""
    metadata, kda_layer = _extract_kda_extra_attrs(layer_idx)
    kda_layer._forward_impl(hidden_states, metadata, output=output)


maybe_bcg_kda_core_inplace = eager_on_graph(kda_core_inplace)


def _check_kda_shape(lin: dict) -> None:
    """The KDA kernels take Kimi K3's shape only."""
    gate_lower_bound = lin.get("gate_lower_bound", None)
    problems = []
    if lin["head_dim"] != 128:
        problems.append(f"head_dim {lin['head_dim']} (the kernels take 128)")
    if lin["short_conv_kernel_size"] != 4:
        problems.append(f"short_conv_kernel_size {lin['short_conv_kernel_size']} (4)")
    if not lin.get("use_full_rank_gate", True):
        problems.append("a low-rank output gate (the full-rank g_proj)")
    if gate_lower_bound is None or not (math.isfinite(gate_lower_bound) and gate_lower_bound < 0):
        problems.append(f"gate_lower_bound {gate_lower_bound} (a finite negative bound)")
    if problems:
        raise ValueError("Kimi K3 KDA: unsupported linear_attn_config: " + "; ".join(problems))


class K3KDALinearAttention(nn.Module):
    """One Kimi K3 KDA layer of the dep16 target (all heads on every rank) on K3-owned weights."""

    def __init__(
        self,
        cfg,
        layer_idx: int,
        mapping=None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__()
        lin = cfg.linear_attn_config
        _check_kda_shape(lin)
        self.hidden_size = cfg.hidden_size
        self.layer_idx = layer_idx
        self.head_dim = lin["head_dim"]
        self.head_k_dim = self.head_dim
        self.conv_size = lin["short_conv_kernel_size"]
        self.gate_lower_bound = lin["gate_lower_bound"]
        self.rms_norm_eps = cfg.rms_norm_eps

        # Attention DP (or one rank): the layer runs every head on its rank's own requests, so
        # there is no head split and no o_proj all-reduce. The weight loader reads the shard
        # size and rank.
        if mapping is not None and mapping.tp_size > 1 and not mapping.enable_attention_dp:
            raise ValueError(
                "Kimi K3 dep16 KDA runs under attention DP (or on one rank); head-split TP "
                f"over {mapping.tp_size} ranks is not part of this target"
            )
        self._kda_tp_size = 1
        self._kda_tp_rank = 0

        self.num_heads = lin["num_heads"]
        projection_size = self.num_heads * self.head_dim
        self.proj_size = projection_size

        projection_quant_configs = (model_config.quant_config_dict or {}) if model_config else {}
        default_quant_config = (
            model_config.quant_config if model_config else None
        ) or QuantConfig()
        declared = {
            name: projection_quant_configs.get(name, default_quant_config).quant_algo
            for name in _FP8_PROJECTIONS
        }
        if any(algo != QuantAlgo.FP8_BLOCK_SCALES for algo in declared.values()):
            raise ValueError(
                "Kimi K3 KDA takes FP8 128 x 128 block-scale q/k/v/g/o projections (the "
                f"checkpoint's FP8_PB_WO); declared {declared}"
            )
        if self.hidden_size % 128 or projection_size % 128:
            raise ValueError("KDA FP8 projection shards must be aligned to 128x128 blocks")

        bf16 = torch.bfloat16
        # This rank's shard (the loader slices the checkpoint by head), so the containers hold
        # rank-local dimensions.
        self.q_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.k_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.v_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.q_conv1d = ShortConvWeight(projection_size, self.conv_size, dtype=bf16)
        self.k_conv1d = ShortConvWeight(projection_size, self.conv_size, dtype=bf16)
        self.v_conv1d = ShortConvWeight(projection_size, self.conv_size, dtype=bf16)
        # Initialized (checkpoint loading overwrites them) with single random-init ops, which
        # MetaInitMode allows: uninitialized NaN bit patterns would poison randomly built layers.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=bf16).uniform_(1, 16))
        )
        self.f_a_proj = Bf16Linear(self.hidden_size, self.head_dim, dtype=bf16)
        self.f_b_proj = Bf16Linear(self.head_dim, projection_size, dtype=bf16)
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=bf16).uniform_(math.log(1e-3), math.log(1e-1))
        )
        self.b_proj = Bf16Linear(self.hidden_size, self.num_heads, dtype=bf16)
        self.g_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.o_norm = RMSNormWeight(self.head_dim, self.rms_norm_eps, dtype=bf16)
        self.o_proj = Fp8BlockScaleLinear(projection_size, self.hidden_size)
        # Built from q/k/v/g by finalize_decode_weights: the fused [q | k | v | g] projection and
        # its output sections.
        self.qkvg_proj: Optional[nn.Module] = None
        self.qkvg_split_sizes: Optional[List[int]] = None

        self._dispatch = K3KDAKernelDispatch()

        # Built after checkpoint load (finalize_decode_weights): the fused BF16 [f_a | b] weight
        # (rows padded to 8) and the kernel-layout constants.
        self._bfa_proj_weight: Optional[torch.Tensor] = None
        self._w_q_t = self._w_k_t = self._w_v_t = None
        self._A_log_f32 = self._dt_bias_f32 = self._onorm_w_f32 = None
        # Fork/join state for overlapping the small [f_a | b] -> f_b chain
        # with the wide QKVG projection during CUDA-graph execution.
        self._projection_aux_stream = aux_stream
        self._projection_fork_event = torch.cuda.Event()
        self._projection_join_event = torch.cuda.Event()
        # Output buffer of the in-place decode kernel outside breakable CUDA graphs.
        self._o_dense: Optional[torch.Tensor] = None
        # [3D, W] BF16 conv weight of causal_conv1d_fwd and the fp32 [D, W] weights of the verify
        # kernels.
        self._packed_conv_weight: Optional[torch.Tensor] = None
        self._mtp_conv_weights: Optional[Tuple[torch.Tensor, ...]] = None

        self.register_to_config = False
        self.layer_idx_str = str(layer_idx)
        if model_config is not None:
            kda_layers = model_config.extra_attrs.setdefault("kda_layers", {})
            suffix = 0
            while self.layer_idx_str in kda_layers:
                self.layer_idx_str = f"{layer_idx}_{suffix}"
                suffix += 1
            kda_layers[self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

    # ------------------------------------------------------------------
    # Post-load weight preparation.
    # ------------------------------------------------------------------

    def finalize_decode_weights(self) -> None:
        """Build the fused projection weights and kernel constants after weight load.

        1. ``qkvg_proj``: one FP8 ``[q | k | v | g]`` projection from the checkpoint codes and
           scales (the swap to the FP8 weight-read GEMM transforms its scales afterwards); the
           four sources become row views of it. ``_bfa_proj_weight``: BF16 ``[f_a | b]`` with
           rows padded to 8; ``f_a_proj`` / ``b_proj`` become row views of it.
        2. Kernel-layout constants: transposed BF16 conv weights ``[W, D]``, fp32 ``A_log`` /
           ``dt_bias`` / ``o_norm.weight`` and the packed / fp32 conv weights of the prefill and
           verify kernels.
        """
        if self.q_proj.weight.device.type != "cuda":
            return
        with torch.no_grad():
            sources = tuple(getattr(self, name) for name in _QKVG_PROJECTIONS)
            if all(isinstance(module, Fp8BlockScaleLinear) for module in sources):
                self.qkvg_proj = self._fuse_checkpoint_projections(sources)
                self.qkvg_split_sizes = [module.out_features for module in sources]
                # Without it the four sources keep their own storage next to the fused copy.
                self._alias_projections_into_fused(sources, self.qkvg_proj)
            self._bfa_proj_weight = _merge_row_weights((self.f_a_proj, self.b_proj), pad_rows_to=8)
            self._build_decode_kernel_constants()

    @staticmethod
    def _fuse_checkpoint_projections(
        modules: Tuple[Fp8BlockScaleLinear, ...],
    ) -> Fp8BlockScaleLinear:
        """One FP8 projection over the rows of ``modules`` (block-aligned): their checkpoint codes
        and FP32 scale grids, copied before any scale preparation."""
        if any(module.out_features % 128 for module in modules[:-1]):
            raise ValueError("KDA FP8 fusion boundaries must be aligned to 128 rows")
        first = modules[0]
        fused = Fp8BlockScaleLinear(
            first.in_features,
            sum(module.out_features for module in modules),
            device=first.weight.device,
        )
        offset = 0
        scale_offset = 0
        for module in modules:
            rows = module.weight.shape[0]
            fused.weight.data[offset : offset + rows].copy_(module.weight.data)
            offset += rows
            scale_rows = module.weight_scale.shape[0]
            fused.weight_scale.data[scale_offset : scale_offset + scale_rows].copy_(
                module.weight_scale.data
            )
            scale_offset += scale_rows
        fused.input_scale.data = first.input_scale.data
        fused.inv_input_scale.data = first.inv_input_scale.data
        return fused

    @staticmethod
    def _alias_projections_into_fused(modules: Tuple[nn.Module, ...], fused: nn.Module) -> None:
        """Repoint the FP8 source projections at row views of the fused codes and scale grid."""
        offset = 0
        scale_offset = 0
        for module in modules:
            rows = module.weight.shape[0]
            module.weight.data = fused.weight.data[offset : offset + rows]
            offset += rows
            scale_rows = module.weight_scale.shape[0]
            module.weight_scale.data = fused.weight_scale.data[
                scale_offset : scale_offset + scale_rows
            ]
            scale_offset += scale_rows

    def _build_decode_kernel_constants(self) -> None:
        """Kernel-layout constants of the decode / verify / prefill kernels."""
        self._w_q_t = (
            self.q_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._w_k_t = (
            self.k_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._w_v_t = (
            self.v_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._A_log_f32 = self.A_log.detach().float().contiguous()
        self._dt_bias_f32 = self.dt_bias.detach().float().contiguous()
        self._onorm_w_f32 = self.o_norm.weight.detach().float().contiguous()
        # Built eagerly too, so the first verify call never allocates (a capture-unsafe lazy
        # allocation).
        self._build_mtp_conv_weights()

    def _build_mtp_conv_weights(self) -> None:
        """Prebuild the packed-prefill and fused-verify convolution weights.

        Building them lazily at first use would allocate at runtime; under
        CUDA graph capture that would bake capture-pool pointers into the
        cached tensors.
        """
        conv_weights = tuple(
            conv.weight.detach().squeeze(1)
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        )
        self._packed_conv_weight = torch.cat(conv_weights, dim=0).to(torch.bfloat16).contiguous()
        self._mtp_conv_weights = tuple(weight.float().contiguous() for weight in conv_weights)

    def _require_finalized(self) -> None:
        if self.qkvg_proj is None or self._bfa_proj_weight is None or self._onorm_w_f32 is None:
            raise RuntimeError(
                "Kimi K3 KDA runs on the fused projections and kernel constants that "
                "finalize_decode_weights() builds after weight load"
            )

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """Select the eager path or breakable-CUDA-graph path; returns the o_proj output."""
        if self.register_to_config and is_in_breakable_cuda_graph():
            core = hidden_states.new_empty(
                (hidden_states.shape[0], self.num_heads, self.head_dim),
                dtype=torch.bfloat16,
            )
            maybe_bcg_kda_core_inplace(hidden_states, self.layer_idx_str, core)
            return self._project_output(core)

        core = self._forward_impl(hidden_states, attn_metadata)
        return self._project_output(core)

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, _KdaPrequantizedCore]:
        """Split the batch into its prefill, decode and verify rows and run their chains.

        ``output`` is the BCG post-o_norm, pre-o_proj core buffer. When it
        is supplied, subpaths fill and return its corresponding slices;
        otherwise this method returns an allocated core tensor. A
        verify-only batch without ``output`` may instead return the fused
        verify epilogue's MXFP8 core (see ``forward_verify_fused``).
        """
        mamba_metadata = attn_metadata.mamba_metadata
        num_prefills = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens
        batch_size = attn_metadata.seq_lens.shape[0]
        state_indices = mamba_metadata.state_indices[:batch_size]
        cu_seqlens = mamba_metadata.query_start_loc_long[: num_prefills + 1]
        num_decodes = batch_size - num_prefills

        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        conv_pool = layer_cache.conv  # [slots, 3D, W - 1] bf16
        ssm_pool = layer_cache.temporal  # [slots, H, V, K] fp32 or bf16
        generation_state_indices = getattr(mamba_metadata, "generation_state_indices", None)
        if generation_state_indices is None:
            generation_state_indices = state_indices[num_prefills:]

        cores: List[torch.Tensor] = []
        if num_prefills > 0:
            snapshot = getattr(mamba_metadata, "kda_snapshot", None)
            inforward_stage = None
            if snapshot is not None and snapshot.stage_rows is not None:
                inforward_stage = attn_metadata.kv_cache_manager.inforward_snapshot_stage_views(
                    self.layer_idx
                )
            prefill_core = self.forward_prefill(
                hidden_states[:num_ctx_tokens],
                cu_seqlens,
                mamba_metadata,
                num_prefills,
                conv_pool,
                ssm_pool,
                state_indices[:num_prefills],
                layer_cache,
                output=output[:num_ctx_tokens] if output is not None else None,
                inforward_stage=inforward_stage,
            )
            if output is None:
                cores.append(prefill_core)
        if num_decodes > 0:
            decode_rows = num_tokens - num_ctx_tokens
            if decode_rows == num_decodes:
                decode_core = self.forward_decode(
                    hidden_states[num_ctx_tokens:num_tokens],
                    conv_pool,
                    ssm_pool,
                    generation_state_indices,
                    layer_cache,
                    output=output[num_ctx_tokens:num_tokens] if output is not None else None,
                )
                if output is None:
                    cores.append(decode_core)
            else:
                # Speculative verification: each generation request carries
                # 1 + draft_len tokens (drafts are padded to the static max,
                # so T is uniform).
                assert decode_rows % num_decodes == 0, (
                    f"ragged generation batch: {decode_rows} tokens for {num_decodes} requests"
                )
                verify_core = self.forward_verify(
                    hidden_states[num_ctx_tokens:num_tokens],
                    decode_rows // num_decodes,
                    layer_cache,
                    ssm_pool,
                    generation_state_indices,
                    output=(output[num_ctx_tokens:num_tokens] if output is not None else None),
                    # The MXFP8 core cannot be concatenated with a prefill
                    # core or stored into the BCG buffer.
                    allow_prequantized_output=output is None and num_prefills == 0,
                    context_only=getattr(attn_metadata.kv_cache_manager, "kda_context_only", False)
                    is True,
                )
                if output is None:
                    cores.append(verify_core)
        if output is not None:
            return output
        return cores[0] if len(cores) == 1 else torch.cat(cores, dim=0)

    def _project_output(self, core: Union[torch.Tensor, _KdaPrequantizedCore]) -> torch.Tensor:
        """o_proj on the post-o_norm core (the BF16 rows, or the verify epilogue's MXFP8 form)."""
        if isinstance(core, _KdaPrequantizedCore):
            return self.o_proj.forward_prequantized(core.activation, core.scale)
        return self.o_proj(core.reshape(-1, self.proj_size))

    def _sync_kda_replay_conv_window(self, layer_cache, slot_indices, conv_pool) -> None:
        """Seed the fused verify kernel's replay conv caches from the live committed conv pool.

        Their committed window (columns ``[0, W-1)``) must hold the live pool's ``W - 1`` raw
        inputs whenever prefill or plain decode advances it. No-op without replay caches (no
        speculative decoding, or a context-only server).
        """
        if not bool(getattr(layer_cache, "has_kda_replay_caches", False)):
            return
        layer_cache.commit_conv_window(slot_indices, conv_pool)

    def _project_gate_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(beta, forget_gate)``: ``F.linear`` on the fused ``[f_a | b]`` weight, then
        ``F.linear`` of its ``f_a`` columns on ``f_b``. ``beta`` and the forget gate are the raw
        logits (the kernels apply the sigmoid / gate)."""
        hd, H = self.head_dim, self.num_heads
        bfa = F.linear(x, self._bfa_proj_weight)
        return bfa[..., hd : hd + H], F.linear(bfa[..., :hd], self.f_b_proj.weight)

    def _project_packed_conv_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(conv_input, onorm_g)`` of the prefill rows ``x`` ``[1, T, hidden]``: the QKVG GEMM,
        its ``[q | k | v]`` columns repacked channel-major ``[3D, T]``
        (``extract_transpose_prefill_slice``, for ``causal_conv1d_fwd``), and its output-gate
        columns."""
        d = self.proj_size
        qkvg = self.qkvg_proj(x)
        onorm_g = qkvg[..., 3 * d : 4 * d]
        qkvg_2d = qkvg.squeeze(0)
        return extract_transpose_prefill_slice(qkvg_2d, qkvg_2d.shape[0], 0, 3 * d), onorm_g

    def forward_prefill(
        self,
        x2d,
        cu_seqlens,
        mamba_metadata,
        num_prefills,
        conv_pool,
        ssm_pool,
        slot_indices,
        layer_cache=None,
        output: Optional[torch.Tensor] = None,
        inforward_stage: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Prefill ``num_prefills`` packed sequences and update their conv / recurrent rows.

        The convolution is ``causal_conv1d_fwd`` + ``fused_kda_post_conv``; Q/K leave it
        L2-normalized for ``kda_prefill``. A single unaligned sequence is chunk-padded by the
        op. A BF16 state pool (and every in-forward snapshot plan) is staged through an fp32 copy;
        a ``kda_snapshot`` plan also stores mid-chunk states and convolution windows.
        """
        if x2d.shape[0] == 0:
            reset_recurrent_state_rows(
                ssm_pool, slot_indices, mamba_metadata.has_initial_states[:num_prefills], conv_pool
            )
            self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)
            return self._store_core(x2d.new_empty((0, self.num_heads, self.head_dim)), output)
        self._require_finalized()
        x = x2d.unsqueeze(0)  # [1, T, hidden]
        # Metadata owns stable int32 indices; never convert per layer.
        slot_indices = mamba_metadata.state_indices[:num_prefills]

        # Mid-chunk snapshots: the recurrent state and convolution window of a sequence after its
        # first tokens of this step. The prefill op writes the state (snapshot_rows /
        # snapshot_chunks). With ``inforward_stage`` (the in-forward snapshots of the cache
        # manager, ``snapshot.stage_rows`` set) the live slot keeps the state at the snapshot
        # point and the staging row receives the chunk-end state; otherwise the snapshot goes to
        # a pool slot of its own.
        snapshot = getattr(mamba_metadata, "kda_snapshot", None)
        if (snapshot is not None and snapshot.stage_rows is not None) != (
            inforward_stage is not None
        ):
            raise RuntimeError(
                "Kimi K3 KDA in-forward snapshots need both the plan's staging rows and the "
                "cache manager's staging views"
            )

        packed_conv, onorm_g = self._project_packed_conv_input(x)
        snapshot_windows = None
        if snapshot is not None:
            # The window is read before the convolution below, which may overwrite its input in
            # place. A live slot's window still feeds that convolution; it is swapped after it.
            snapshot_windows = _snapshot_conv_windows(packed_conv, conv_pool, snapshot)
            if inforward_stage is None:
                conv_pool.index_copy_(0, snapshot.slots, snapshot_windows)

        # Initial states: present for continuation chunks (chunked prefill)
        # and for prefix-cache hits (block reuse), where the previous
        # conv/recurrent state was onboarded into this request's slot.
        has_init = mamba_metadata.has_initial_states[:num_prefills]
        prepared_state = self._prepare_prefill_state(
            cu_seqlens,
            mamba_metadata,
            num_prefills,
            ssm_pool,
            slot_indices,
            snapshot=snapshot,
        )
        # The convolution input is a contiguous channel-major [3D, T] copy: the op writes its
        # output over it.
        causal_conv1d_fwd(
            packed_conv,
            self._packed_conv_weight,
            None,
            conv_pool,
            mamba_metadata.query_start_loc[: num_prefills + 1],
            mamba_metadata.state_indices[:num_prefills],
            has_init,
            True,
            PAD_SLOT_ID,
        )

        beta, g = self._project_gate_inputs(x)
        g = g.unflatten(-1, (self.num_heads, self.head_dim))

        # Imported here: the entry imports this package's kernels, and the package imports this
        # module. The prefill op chunk-pads a single unaligned sequence itself.
        from ..ops.ssm.fused_kda_post_conv import fused_kda_post_conv

        beta = beta.float()
        q, k, v = fused_kda_post_conv(
            packed_conv,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        o = self._prefill_recurrence(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            mamba_metadata,
            num_prefills,
            ssm_pool,
            slot_indices,
            prepared_state,
            snapshot=snapshot,
            inforward_stage_ssm=None if inforward_stage is None else inforward_stage[0],
        )
        # Fused-verify replay caches: seed the committed conv window so the
        # first verify round convolves the correct history (pending drafts
        # are zero for a fresh request, so the tail columns are unused).
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)
        if inforward_stage is not None:
            _swap_inforward_conv_windows(conv_pool, inforward_stage[1], snapshot, snapshot_windows)

        core = self._output_gate(o, onorm_g)
        return self._store_core(core, output)

    def _prepare_prefill_state(
        self,
        cu_seqlens: torch.Tensor,
        mamba_metadata,
        num_prefills: int,
        ssm_pool: torch.Tensor,
        slot_indices: torch.Tensor,
        snapshot: Optional["KDASnapshotPlan"] = None,
    ) -> _KDAPrefillState:
        """The state rows the prefill op updates in place, zeroed where a sequence starts fresh.

        The op takes fp32 V-first pool rows. A BF16 pool is staged through a dense fp32 copy of
        the addressed rows (plus the snapshot rows of a ``snapshot`` plan), as is an fp32 pool
        under an in-forward snapshot plan (whose snapshot rows are the live slots, which the
        kernel's chunk-end write would overwrite); :meth:`_prefill_recurrence` writes the copy
        back.
        """
        has_init = mamba_metadata.has_initial_states[:num_prefills]
        staged_state = None
        kernel_pool, kernel_indices = ssm_pool, slot_indices
        snapshot_rows = None if snapshot is None else snapshot.pool_rows
        inforward = snapshot is not None and snapshot.stage_rows is not None
        if ssm_pool.dtype != torch.float32 or inforward:
            if snapshot is None:
                staged_state = _stage_state_rows(ssm_pool, slot_indices)
            else:
                staged_state = _stage_state_rows_with_snapshot_rows(
                    ssm_pool, slot_indices, snapshot.num_snapshots
                )
                snapshot_rows = snapshot.staged_rows
            kernel_pool = staged_state
            kernel_indices = mamba_metadata._arange_buffer[:num_prefills]
        if not self._dispatch.can_use_indexed_prefill(
            state_pool=kernel_pool,
            state_indices=kernel_indices,
            has_initial_states=has_init,
            cu_seqlens=cu_seqlens,
            num_sequences=num_prefills,
        ):
            raise RuntimeError(
                "Kimi K3 KDA prefill: the kda_prefill entry does not take this state pool / "
                "metadata layout (fp32 [slots, H, V, K] rows, 16-byte aligned int32 / int64 "
                "indices, bool has_initial_states)"
            )
        # The convolution clears fresh convolution rows itself.
        reset_recurrent_state_rows(kernel_pool, kernel_indices, has_init)
        return _KDAPrefillState(
            kernel_pool, kernel_indices, staged_state, snapshot_rows=snapshot_rows
        )

    def _prefill_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        mamba_metadata,
        num_prefills: int,
        ssm_pool: torch.Tensor,
        slot_indices: torch.Tensor,
        state: _KDAPrefillState,
        *,
        snapshot: Optional["KDASnapshotPlan"] = None,
        inforward_stage_ssm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``kda_prefill`` on the token-major L2-normalized Q/K, V, the raw gate and
        beta logits, updating ``state``'s fp32 V-first ``[slots, H, D, D]`` rows in place, then
        the write-back of a staged copy (and the snapshot moves)."""
        o = self._dispatch.prefill_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self._A_log_f32,
            dt_bias=self._dt_bias_f32,
            scale=self.head_k_dim**-0.5,
            lower_bound=self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=getattr(mamba_metadata, "kda_chunk_indices", None),
            chunk_size=_PREFILL_CHUNK,
            state_pool=state.kernel_pool,
            state_indices=state.kernel_indices,
            varlen_is_aligned=getattr(mamba_metadata, "kda_varlen_is_aligned", None),
            single_sequence_length=getattr(mamba_metadata, "kda_single_sequence_length", None),
            snapshot_rows=state.snapshot_rows if snapshot is not None else None,
            snapshot_chunks=None if snapshot is None else snapshot.chunks,
        )

        # The convolution persisted the live convolution pool in place.
        staged_state = state.staged_state
        if staged_state is not None:
            if snapshot is None:
                _writeback_state_rows(ssm_pool, slot_indices, staged_state)
            else:
                _writeback_state_rows(ssm_pool, slot_indices, staged_state[:num_prefills])
                if snapshot.stage_rows is not None:
                    # In-forward: the chunk-end states move to the staging rows; the live
                    # slots (``snapshot.slots``) take the states at the snapshot points.
                    inforward_stage_ssm.index_copy_(
                        0, snapshot.stage_rows, ssm_pool.index_select(0, snapshot.slots)
                    )
                _writeback_state_rows(ssm_pool, snapshot.slots, staged_state[num_prefills:])
        elif snapshot is not None and snapshot.stage_rows is not None:
            raise RuntimeError("Kimi K3 KDA in-forward snapshots need a staged state copy")
        return o

    def forward_decode(
        self,
        x2d,
        conv_pool,
        ssm_pool,
        slot_indices,
        layer_cache=None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One token per request: the QKVG GEMM (next to ``[f_a | b]`` -> ``f_b`` on the
        projection aux stream up to 128 rows in CUDA graphs), then the fused decode kernel, which
        convolves, gates, runs the recurrence and the gated RMSNorm, and updates the conv and
        fp32 state pools by slot. A BF16 state pool is staged through dense copies of the
        addressed rows (fp32 state, conv window as is) and written back (state rounded).
        """
        self._require_finalized()
        d = self.proj_size
        hd = self.head_dim
        H = self.num_heads
        B = x2d.shape[0]

        staged_state = staged_conv = None
        slots_long = None
        kernel_state, kernel_state_indices, conv_src = ssm_pool, slot_indices, conv_pool
        if ssm_pool.dtype != torch.float32:
            slots_long = slot_indices.long()
            staged_state = _stage_state_rows(ssm_pool, slots_long)
            staged_conv = conv_pool.index_select(0, slots_long)
            kernel_state, kernel_state_indices, conv_src = staged_state, None, staged_conv

        # The decode kernel writes its core in place: the BCG core buffer, or outside breakable
        # CUDA graphs a persistent buffer whose pointer stays stable across graph captures.
        if output is not None:
            kda_out = output.view(B, 1, H, hd)
        else:
            if self._o_dense is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Kimi K3 KDA decode: its output buffer is allocated by the eager forward "
                        "that precedes CUDA graph capture"
                    )
                self._o_dense = torch.empty(
                    max(conv_pool.shape[0], B),
                    1,
                    H,
                    hd,
                    dtype=torch.bfloat16,
                    device=x2d.device,
                )
            else:
                assert self._o_dense.shape[0] >= B, (
                    f"KDA decode output buffer holds {self._o_dense.shape[0]} rows "
                    f"but the decode batch is {B}; reallocating would corrupt "
                    f"previously captured CUDA graphs"
                )
            kda_out = self._o_dense[:B]

        projection_aux_stream = (
            self._projection_aux_stream if B <= _KDA_BFA_MULTISTREAM_MAX_ROWS else None
        )
        qkvg, (beta, g) = fork_join(
            lambda: self.qkvg_proj(x2d),
            lambda: self._project_gate_inputs(x2d),
            self._projection_fork_event,
            self._projection_join_event,
            projection_aux_stream,
        )
        x_qkvg = qkvg[:, : 3 * d]
        onorm_g = qkvg[:, 3 * d : 4 * d]

        # Section views retain the live pool's slot stride, including V2
        # manager padding. The kernel uses ssm_state_indices for both pools.
        o = self._dispatch.decode_kda(
            x_q=x_qkvg[:, :d].unflatten(-1, (H, hd)).unsqueeze(0),
            x_k=x_qkvg[:, d : 2 * d].unflatten(-1, (H, hd)).unsqueeze(0),
            x_v=x_qkvg[:, 2 * d : 3 * d].unflatten(-1, (H, hd)).unsqueeze(0),
            w_q_t=self._w_q_t,
            w_k_t=self._w_k_t,
            w_v_t=self._w_v_t,
            bias_q=None,
            bias_k=None,
            bias_v=None,
            cs_q=conv_src[:, :d],
            cs_k=conv_src[:, d : 2 * d],
            cs_v=conv_src[:, 2 * d :],
            A_log=self._A_log_f32,
            g=g.unflatten(-1, (H, hd)).unsqueeze(0),
            dt_bias=self._dt_bias_f32,
            beta=beta.unsqueeze(0),
            state=kernel_state,
            onorm_g=onorm_g.unflatten(-1, (H, hd)).unsqueeze(0),
            onorm_weight=self._onorm_w_f32,
            out=kda_out,
            ssm_state_indices=kernel_state_indices,
            scale=hd**-0.5,
            onorm_eps=self.o_norm.eps,
            lower_bound=self.gate_lower_bound,
            use_beta_sigmoid_in_kernel=True,
            verbose=False,
            update_conv_cache=True,
        )
        if staged_state is not None:
            conv_pool.index_copy_(0, slots_long, staged_conv)
            _writeback_state_rows(ssm_pool, slots_long, staged_state)
        # Fused-verify replay caches (spec decoding only): keep the
        # committed conv window in sync with the plain-decode advance.
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)

        # The kernel applied the output gate / RMSNorm and returns the supplied buffer.
        return o.view(B, H, hd)

    def forward_verify(
        self,
        x2d,
        num_steps,
        layer_cache,
        ssm_pool,
        slot_indices,
        output: Optional[torch.Tensor] = None,
        allow_prequantized_output: bool = False,
        context_only: bool = False,
    ) -> Union[torch.Tensor, _KdaPrequantizedCore]:
        """Speculative verification: advance each request ``num_steps`` tokens (1 golden +
        ``num_steps - 1`` padded drafts) with the fused verify kernel on the cache manager's
        replay caches (:meth:`forward_verify_fused`).

        On a context-only server the rows are dummies and take
        ``forward_verify_context_only`` instead (the manager marks such a server with
        ``kda_context_only``, passed in as ``context_only``).
        """
        if bool(getattr(layer_cache, "has_kda_replay_caches", False)):
            return self.forward_verify_fused(
                x2d,
                num_steps,
                layer_cache,
                ssm_pool,
                slot_indices,
                output=output,
                allow_prequantized_output=allow_prequantized_output,
            )
        if context_only or getattr(layer_cache, "context_only", False):
            return self.forward_verify_context_only(x2d, output)
        raise RuntimeError(
            "Kimi K3 KDA verification runs the fused verify kernel on the cache manager's KDA "
            "replay caches, which this layer cache does not have"
        )

    def forward_verify_context_only(
        self, x2d: torch.Tensor, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Verification rows on a context-only server.

        Such a server (disaggregated context server) runs no real generation
        request; its generation rows are dummies (warmup, attention-DP
        padding) whose outputs are discarded, and its cache manager keeps no
        replay caches for them. The recurrence is skipped and the live pools
        are left untouched; the output projection (``_project_output``) still
        runs on a zero attention core, so the layer's projection and collectives
        match those of a real verification step.
        """
        core = x2d.new_zeros(x2d.shape[0], self.num_heads, self.head_dim)
        return self._store_core(core, output)

    def _project_verify_inputs(self, x: torch.Tensor, num_rows: int) -> tuple:
        """The verify projections: the QKVG GEMM on the calling stream, ``F.linear``
        ``[f_a | b]`` -> ``f_b`` on the projection aux stream.

        Returns ``(q, k, v, forget_gate, beta, onorm_g)``, the q / k / v / output-gate sections as
        strided views of the QKVG output (the verify kernel reads them in place) and beta as one
        of the ``[f_a | b]`` output.
        """
        # The unfused gate chain overlaps the (all-SM) QKVG GEMM better than it runs after it.
        projection_aux_stream = self._projection_aux_stream if num_rows > 0 else None
        qkvg, (beta, gate) = fork_join(
            lambda: self.qkvg_proj(x),
            lambda: self._project_gate_inputs(x),
            self._projection_fork_event,
            self._projection_join_event,
            projection_aux_stream,
        )

        d = self.proj_size
        q_proj, k_proj, v_proj = qkvg[..., : 3 * d].split(d, dim=-1)
        onorm_g = qkvg[..., 3 * d : 4 * d]
        return q_proj, k_proj, v_proj, gate, beta, onorm_g

    def forward_verify_fused(
        self,
        x2d,
        num_steps,
        layer_cache,
        ssm_pool,
        slot_indices,
        output: Optional[torch.Tensor] = None,
        allow_prequantized_output: bool = False,
    ) -> Union[torch.Tensor, _KdaPrequantizedCore]:
        """Fused multi-token verify (``mtp_ops.py``: the v1 / v2 kernels).

        The runtime packs ``num_steps`` new rows contiguously per request, so the kernel derives
        the row offsets from the request index and reads each request's pending accepted count
        from the slot-indexed replay pool (``packed_token_layout``). One launch replays the
        previous round's accepted drafts from the per-slot replay caches, then processes the new
        tokens, committing the recurrent state and conv windows in place after the golden token
        and caching the new drafts, and applies the gated RMSNorm in its epilogue:

        * ``quantize_output``: where ``allow_prequantized_output`` and ``o_proj`` takes a
          prequantized activation for this row count (SM107), the core quantized to MXFP8 for
          ``o_proj.forward_prequantized``;
        * otherwise the BF16 core, a programmatic dependent of the QKVG GEMM.
        """
        self._require_finalized()
        num_generations = x2d.shape[0] // num_steps
        num_spec = num_steps - 1
        H = self.num_heads
        K = self.head_k_dim
        x = x2d.view(num_generations, num_steps, -1)  # [B, T, hidden]
        T_total = num_generations * num_steps

        # The MXFP8 core cannot be concatenated with a prefill core or stored into the BCG buffer
        # (``allow_prequantized_output``).
        mxfp8_output = (
            allow_prequantized_output
            and output is None
            and _o_proj_takes_mxfp8(self.o_proj, T_total)
            and callable(getattr(self.o_proj, "forward_prequantized", None))
        )
        q_proj, k_proj, v_proj, forget_gate, beta_proj, onorm_g = self._project_verify_inputs(
            x, T_total
        )
        x_q = q_proj.view(1, T_total, H, K)
        x_k = k_proj.view(1, T_total, H, K)
        x_v = v_proj.view(1, T_total, H, self.head_dim)
        # Raw gate / beta: the kernel applies dt_bias, A_log, the
        # lower-bound sigmoid gate, and the beta sigmoid itself.
        beta = beta_proj.view(1, T_total, H)
        w_q, w_k, w_v = self._mtp_conv_weights
        onorm_g_view = onorm_g.view(1, T_total, H, self.head_dim)

        out = self._dispatch.mtp_verify(
            mxfp8_output=mxfp8_output,
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            cs_q=layer_cache.kda_conv_q,
            cs_k=layer_cache.kda_conv_k,
            cs_v=layer_cache.kda_conv_v,
            g=forget_gate.view(1, T_total, H, K),
            beta=beta,
            # .detach(): the CuTe DSL DLPack bridge rejects grad-tracking tensors.
            A_log=self.A_log.detach(),
            dt_bias=self.dt_bias.detach(),
            recurrent_state=ssm_pool,
            k_cache=layer_cache.kda_k_cache,
            g_cache=layer_cache.kda_g_cache,
            v_cache=layer_cache.kda_v_cache,
            beta_cache=layer_cache.kda_beta_cache,
            ssm_state_indices=slot_indices,
            cu_seqlens=None,
            num_spec=num_spec,
            num_accepted_tokens=layer_cache.prev_num_accepted_tokens,
            packed_token_layout=True,
            lower_bound=self.gate_lower_bound,
            scale=self.head_k_dim**-0.5,
            onorm_g=onorm_g_view,
            onorm_weight=self._onorm_w_f32,
            onorm_eps=self.o_norm.eps,
            fuse_output_norm=True,
            # A BF16-output verify launches as a programmatic dependent of the QKVG projection:
            # its state, replay and weight loads run before griddepcontrol.wait.
            use_pdl=not mxfp8_output,
        )
        if mxfp8_output:
            if not isinstance(out, tuple):
                raise RuntimeError("MXFP8-output MTP KDA must return activation and scale")
            return _KdaPrequantizedCore(*out)
        return self._store_core(out, output)

    def _output_gate(self, o: torch.Tensor, onorm_g: torch.Tensor) -> torch.Tensor:
        """The gated output RMSNorm ``rmsnorm(o) * sigmoid(g)`` (``rms_norm_gated_token_major``)
        of the prefill core."""
        return rms_norm_gated_token_major(
            o.reshape(-1, self.head_dim),
            onorm_g.reshape(-1, self.num_heads, self.head_dim),
            self.o_norm.weight,
            self.o_norm.eps,
            gate_activation="sigmoid",
        )

    def _store_core(self, core: torch.Tensor, output: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the normalized core ``[tokens, H, D]``, copied into the BCG buffer ``output``
        where one is given."""
        core = core.reshape(-1, self.num_heads, self.head_dim)
        if output is None:
            return core
        output.copy_(core)
        return output


# Head dimensions (K-nope, K-pe, V) the split-K context FMHA is built for.
_SPLIT_K_HEAD_DIMS = (128, 64, 128)


def split_k_context_unavailable_reason(attn) -> Optional[str]:
    """Why ``attn`` cannot run the Rubin FP8 split-K context FMHA, or None if it can."""
    sm_version = get_sm_version()
    if sm_version != 107:
        return f"requires SM107, got SM{sm_version}"
    if attn.num_heads <= 0 or attn.num_heads != attn.num_kv_heads:
        return (
            "requires matching positive Q/KV head counts per rank, "
            f"got {attn.num_heads}/{attn.num_kv_heads}"
        )
    head_dims = (attn.qk_nope_head_dim, attn.qk_rope_head_dim, attn.v_head_dim)
    if head_dims != _SPLIT_K_HEAD_DIMS:
        return f"requires head dimensions (K-nope, K-pe, V)={_SPLIT_K_HEAD_DIMS}, got {head_dims}"
    from ..kernels.k3r_mla_ctx import k3_context_fmha_op

    if k3_context_fmha_op.IS_K3_CONTEXT_FMHA_AVAILABLE:
        return None
    if k3_context_fmha_op.K3_CONTEXT_FMHA_IMPORT_ERROR is not None:
        return f"CuTe DSL kernel import failed: {k3_context_fmha_op.K3_CONTEXT_FMHA_IMPORT_ERROR!r}"
    return "CuTe DSL kernel lacks split-K epilogue merge support"


def supports_k3_fp8_split_k_context(attn) -> bool:
    """Whether the attention backend ``attn`` can run the Rubin FP8 split-K context FMHA."""
    reason = split_k_context_unavailable_reason(attn)
    if reason is None:
        return True
    logger.info_once(
        f"Kimi K3 MLA: Rubin FP8 split-K context FMHA disabled: {reason}; NVFP4 KV-cache layers "
        "cannot attend a cached prefix (chunked prefill raises).",
        key=("k3_mla_ctx_split_k_unavailable", reason),
    )
    return False


def run_k3_split_k_context_partition(
    attn,
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    softmax_stats: torch.Tensor,
    merge_op: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    *,
    max_q_len: int,
    max_kv_len: int,
    is_causal: bool,
) -> None:
    """Attend FP8 Q to one FP8 KV partition and merge it into ``output`` / ``softmax_stats``.

    ``q`` ``[T_q, H * 192]``, ``k_nope`` / ``v`` ``[T_kv, H * 128]`` with unit inner stride,
    ``k_pe`` ``[T_kv, 64]``, ``output`` ``[T_q, H * 128]`` BF16, ``softmax_stats``
    ``[T_q, H, 2]`` FP32; the softmax scale is the attention backend's.
    """

    sm_scale = 1.0 / (attn.q_scaling * (attn.qk_nope_head_dim + attn.qk_rope_head_dim) ** 0.5)
    num_heads = attn.num_heads
    logger.info_once(
        "Kimi K3 MLA context FMHA: k3_context_fmha (Rubin FP8 split-K, epilogue "
        f"merge) serves {'causal current-chunk' if is_causal else 'cached-prefix'} partitions, "
        f"{num_heads} heads per rank.",
        key=("k3_mla_ctx_split_k_fmha", is_causal, num_heads),
    )
    k3_context_fmha(
        q.view(q.shape[0], num_heads, 192),
        k_nope.view(k_nope.shape[0], num_heads, 128),
        k_pe.view(k_pe.shape[0], 1, 64),
        v.view(v.shape[0], num_heads, 128),
        output.view(q.shape[0], num_heads, 128),
        cu_q_seqlens,
        cu_kv_seqlens,
        softmax_stats,
        merge_op,
        max_q_len,
        max_kv_len,
        sm_scale,
        is_causal,
    )


def load_fp4_mla_chunked_kv_cache_mxfp8(
    metadata,
    layer_idx: int,
    *,
    num_ctx_cached_tokens: int,
    cu_chunked_seq_len: torch.Tensor,
    chunked_global_offset: torch.Tensor,
    chunked_max_seq_len: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Gather one cached-prefix partition of dense FP4 MLA V2 storage as MXFP8 kv_b_proj input.

    The partition and metadata are those of ``load_fp4_mla_chunked_kv_cache``. Returns
    ``((E4M3 [M, 512], UE8M0 [pad_up(M, 128) * 16]), E4M3 [M, 64])``: the BF16-rounded latent
    quantized 1x128 with scales in the K32 R128c4 layout ``cute_dsl_mxfp8_gemm_rubin`` reads, and
    the BF16-rounded RoPE part plus its residual in unit-scale E4M3.
    """
    if kv_lora_rank != 512 or qk_rope_head_dim != FP4_MLA_K_RESIDUAL_DIM:
        raise ValueError(
            "MXFP8 gather requires kv_lora_rank=512 and "
            f"qk_rope_head_dim={FP4_MLA_K_RESIDUAL_DIM}, got {kv_lora_rank} and {qk_rope_head_dim}."
        )
    if num_ctx_cached_tokens < 0:
        raise ValueError("FP4 MLA chunk gather token count must be non-negative.")
    if chunked_max_seq_len < 0:
        raise ValueError("FP4 MLA chunk gather max sequence length must be non-negative.")
    if num_ctx_cached_tokens > 0 and chunked_max_seq_len == 0:
        raise ValueError("A non-empty FP4 MLA chunk gather requires a positive max length.")
    num_contexts = int(metadata.num_contexts)
    if num_ctx_cached_tokens > 0 and num_contexts <= 0:
        raise ValueError("A non-empty FP4 MLA chunk gather requires context requests.")
    tensors = (cu_chunked_seq_len, chunked_global_offset)
    if any(not isinstance(tensor, torch.Tensor) or not tensor.is_cuda for tensor in tensors):
        raise ValueError("FP4 MLA chunk gather metadata must use CUDA tensors.")
    if (
        cu_chunked_seq_len.dtype != torch.int64
        or cu_chunked_seq_len.ndim != 1
        or cu_chunked_seq_len.numel() < num_contexts + 1
        or not cu_chunked_seq_len.is_contiguous()
    ):
        raise ValueError(
            "FP4 MLA chunk gather requires a contiguous int64 cumulative-length tensor."
        )
    if (
        chunked_global_offset.dtype != torch.int64
        or chunked_global_offset.ndim != 1
        or chunked_global_offset.numel() < num_contexts
        or not chunked_global_offset.is_contiguous()
    ):
        raise ValueError("FP4 MLA chunk gather requires contiguous int64 global offsets.")
    if cu_chunked_seq_len.device != chunked_global_offset.device:
        raise ValueError("FP4 MLA chunk gather metadata tensors must share one device.")

    device = cu_chunked_seq_len.device
    compressed_kv = torch.empty(
        (num_ctx_cached_tokens, kv_lora_rank), dtype=torch.float8_e4m3fn, device=device
    )
    compressed_sf = torch.empty(
        (triton.cdiv(num_ctx_cached_tokens, 128) * 128 * (kv_lora_rank // 32),),
        dtype=torch.uint8,
        device=device,
    )
    k_pe = torch.empty(
        (num_ctx_cached_tokens, qk_rope_head_dim), dtype=torch.float8_e4m3fn, device=device
    )
    if num_ctx_cached_tokens == 0:
        return (compressed_kv, compressed_sf), k_pe

    state = metadata.fp4_mla_state
    if not bool(getattr(state, "device_page_table", False)):
        raise RuntimeError("FP4 MLA chunk gather requires fixed-stride device page metadata.")
    _materialize_fp4_mla_device_page_table_for_forward(metadata)
    page_table_stride = int(state.page_table_stride)
    page_ids = state._paged_kv_indices
    if (
        page_table_stride <= 0
        or not isinstance(page_ids, torch.Tensor)
        or page_ids.dtype != torch.int32
        or not page_ids.is_cuda
        or page_ids.device != device
        or page_ids.numel() < num_contexts * page_table_stride
    ):
        raise RuntimeError("FP4 MLA chunk gather received invalid device page metadata.")

    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    head_dim = kv_lora_rank + qk_rope_head_dim
    storage_head_dim = _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=_fp4_mla_attention_backend(),
    )
    if storage_head_dim != head_dim + FP4_MLA_K_RESIDUAL_DIM:
        raise RuntimeError(
            "FP4 MLA chunk gather requires K residual storage for BF16 reconstruction."
        )
    from ..ops.attention.fp4_mla_gather_mxfp8 import fp4_mla_gather_mxfp8

    logger.info_once(
        "Kimi K3 MLA context gather: cached-prefix chunks are read from the FP4 cache as MXFP8 "
        "latent + FP8 K-pe (fp4_mla_gather_mxfp8).",
        key="k3_mla_ctx_gather_mxfp8",
    )
    fp4_mla_gather_mxfp8(
        compressed_kv,
        compressed_sf,
        k_pe,
        kv_cache,
        sf_cache.view(torch.float8_e4m3fn),
        page_ids,
        cu_chunked_seq_len,
        chunked_global_offset,
        _get_fp4_mla_global_scale(metadata, kv_cache.device),
        chunked_max_seq_len,
        metadata.page_size,
        page_table_stride=page_table_stride,
        num_contexts=num_contexts,
    )
    return (compressed_kv, compressed_sf), k_pe


def uses_k3_split_k_context(layer, q: torch.Tensor, output: torch.Tensor) -> bool:
    """Whether the FP4-cache chunked prefill of ``layer`` runs on the split-K context FMHA: BF16 Q
    and output, an NVFP4 latent cache and :func:`supports_k3_fp8_split_k_context`."""
    mha = layer.mha
    return (
        q.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and layer.out_scale is None
        and isinstance(mha, K3MlaAttnState)
        and mha.uses_fp4_mla_attention
        and supports_k3_fp8_split_k_context(mha)
    )


def kv_b_takes_mxfp8(layer) -> bool:
    """Whether kv_b runs as the SM107 MXFP8 GEMM on an MXFP8 cached-prefix gather.

    Needs SM107 with the CuTe DSL Rubin kernels, the 512 / 64 latent / RoPE dimensions of the
    gather, and a bias-free FP8 block-scale kv_b that takes a prequantized CuTe-layout input, with
    its UE8M0 weight scale in the CuTe MXFP8 layout.
    """
    kv_b_proj = layer.kv_b_proj
    weight = getattr(kv_b_proj, "weight", None)
    weight_scale = getattr(kv_b_proj, "weight_scale", None)
    return (
        get_sm_version() == 107
        and IS_CUTLASS_DSL_RUBIN_AVAILABLE
        and layer.kv_lora_rank == 512
        and layer.qk_rope_head_dim == 64
        and getattr(kv_b_proj, "bias", None) is None
        and getattr(kv_b_proj, "supports_prequantized_input", False)
        and weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale is not None
        and weight_scale.dtype == torch.uint8
    )


def forward_context_split_k_fp4(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    latent_cache: torch.Tensor,
    attn_metadata: TrtllmAttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """FP4 MLA chunked prefill of ``layer`` with every partition on the split-K context FMHA."""
    mha = layer.mha
    if latent_cache is None:
        raise RuntimeError("FP4 MLA chunked prefill requires latent_cache.")
    kv_b_mxfp8 = kv_b_takes_mxfp8(layer)
    logger.info_once(
        "Kimi K3 MLA cached-prefix context: NVFP4 KV-cache layers run the Rubin FP8 split-K "
        "context FMHA (k3_context_fmha); cached prefix: "
        + (
            "MXFP8 gather + FP8-output MXFP8 kv_b GEMM"
            if kv_b_mxfp8
            else "BF16 gather + kv_b_proj + FP8 quantization"
        )
        + "; Q FP8 quantization: static_quantize_e4m3 (Triton).",
        key=("k3_mla_ctx_split_k", kv_b_mxfp8),
    )
    lora, rope = layer.kv_lora_rank, layer.qk_rope_head_dim
    heads = layer.num_heads_tp

    # Rotate Q and the current K in place and append the chunk to the FP4 cache; the current
    # chunk's rotated K-pe is then read from latent_cache.
    mla_cache.append_fp4_mla_context_cache(mha, q, latent_cache, attn_metadata)
    current_k_pe = latent_cache.reshape(-1, lora + rope)[:, lora:]
    fp8_unit_scale = mha.kv_cache_scaling_factor
    q_fp8 = static_quantize_e4m3(q.contiguous(), fp8_unit_scale)
    context_cu_q_seqlens = attn_metadata.mla_prepare_ctx_cu_seqlens()
    if context_cu_q_seqlens is None:
        raise RuntimeError("K3 split-K context FMHA requires context requests.")

    # The first partition of every request writes the output and its stats (merge op 2).
    softmax_stats = torch.empty(
        (attn_metadata.num_ctx_tokens, heads, 2), dtype=torch.float, device=q.device
    )
    kv_split = [heads * layer.qk_nope_head_dim, heads * layer.v_head_dim]

    def expand_kv(
        partition_compressed_kv: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        partition_k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if kv_b_mxfp8:
            if isinstance(partition_compressed_kv, tuple):
                activation, activation_scale = partition_compressed_kv
            else:
                activation, activation_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
                    partition_compressed_kv.contiguous()
                )
            # The FP8 output keeps the GEMM's BF16 rounding (unit output scale).
            partition_kv = k3_mxfp8_gemm_fp8_out(
                activation,
                layer.kv_b_proj.weight,
                activation_scale,
                layer.kv_b_proj.weight_scale,
            )
        else:
            partition_kv, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                layer.kv_b_proj(partition_compressed_kv), fp8_unit_scale
            )
        if partition_k_pe.dtype == torch.float8_e4m3fn:
            k_pe_source = "FP8 from the gather"
        elif fp8_static_quant.supports_rows(partition_k_pe):
            k_pe_source = "static_quantize_e4m3_rows (Triton)"
            partition_k_pe = static_quantize_e4m3(partition_k_pe, fp8_unit_scale, rows=True)
        else:
            k_pe_source = "static_quantize_e4m3_per_tensor"
            partition_k_pe, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                partition_k_pe.contiguous(), fp8_unit_scale
            )
        logger.info_once(
            "Kimi K3 MLA context expand: per-chunk kv_b_proj runs as "
            + (
                "k3_mxfp8_gemm_fp8_out (MXFP8 in, FP8 out)"
                if kv_b_mxfp8
                else "kv_b_proj (BF16) + static_quantize_e4m3_per_tensor"
            )
            + f"; K-pe: {k_pe_source}; K-nope / K-pe / V go to the split-K FMHA unconcatenated.",
            key=("k3_mla_ctx_split_k_expand", kv_b_mxfp8, k_pe_source),
        )
        partition_k_nope, partition_v = partition_kv.split(kv_split, -1)
        return partition_k_nope, partition_k_pe, partition_v

    def run_partition(
        partition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cu_kv_seqlens: torch.Tensor,
        max_kv_len: int,
        merge_op: torch.Tensor,
        is_causal: bool,
    ) -> None:
        partition_k_nope, partition_k_pe, partition_v = partition
        run_k3_split_k_context_partition(
            mha,
            q_fp8,
            partition_k_nope,
            partition_k_pe,
            partition_v,
            output,
            softmax_stats,
            merge_op,
            context_cu_q_seqlens,
            cu_kv_seqlens,
            max_q_len=attn_metadata.max_ctx_seq_len,
            max_kv_len=max_kv_len,
            is_causal=is_causal,
        )

    for loop_idx in range(attn_metadata.chunked_loop_num):
        cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
        total_chunked_tokens = int(
            attn_metadata.host_cu_chunked_seq_len[loop_idx, attn_metadata.num_contexts]
        )
        chunked_max_seq_len = int(attn_metadata.max_chunk_len_per_loop[loop_idx])
        if kv_b_mxfp8:
            chunked_compressed_kv, chunked_k_pe = load_fp4_mla_chunked_kv_cache_mxfp8(
                attn_metadata,
                mha.layer_idx,
                num_ctx_cached_tokens=total_chunked_tokens,
                cu_chunked_seq_len=cu_chunked_seq_len,
                chunked_global_offset=attn_metadata.chunked_global_offset[loop_idx],
                chunked_max_seq_len=chunked_max_seq_len,
                kv_lora_rank=lora,
                qk_rope_head_dim=rope,
            )
        else:
            chunked_compressed_kv, chunked_k_pe = mla_cache.load_chunked_kv_cache_for_mla(
                mha,
                attn_metadata,
                num_ctx_cached_tokens=total_chunked_tokens,
                cu_chunked_seq_len=cu_chunked_seq_len,
                chunked_global_offset=attn_metadata.chunked_global_offset[loop_idx],
                chunked_max_seq_len=chunked_max_seq_len,
                out_dtype=q.dtype,
            )
        partition = expand_kv(chunked_compressed_kv, chunked_k_pe)
        chunked_compressed_kv = None
        chunked_k_pe = None
        run_partition(
            partition,
            cu_chunked_seq_len,
            chunked_max_seq_len,
            attn_metadata.merge_op_tensor[loop_idx],
            is_causal=False,
        )

    run_partition(
        expand_kv(compressed_kv, current_k_pe),
        context_cu_q_seqlens,
        attn_metadata.max_ctx_seq_len,
        attn_metadata.merge_op_tensor[attn_metadata.chunked_loop_num],
        is_causal=True,
    )
    return output


__all__ = [
    "forward_context_split_k_fp4",
    "kv_b_takes_mxfp8",
    "load_fp4_mla_chunked_kv_cache_mxfp8",
    "run_k3_split_k_context_partition",
    "split_k_context_unavailable_reason",
    "supports_k3_fp8_split_k_context",
    "uses_k3_split_k_context",
    "context_cached_kv",
    "context_chunked_prefill",
    "context_default",
    "forward_context",
    "project_context_kv",
    "forward_absorption_generation",
    "gen_attention",
    "K3MLAAttention",
]


def project_context_kv(layer, compressed_kv: torch.Tensor) -> torch.Tensor:
    """kv_b over the context heads: ``[T, H * qk_nope | H * v_head_dim]``."""
    return layer.kv_b_proj(compressed_kv)


def _expand_kv_bf16(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor):
    """kv_b, then K ``[T, H * 192]`` (K-nope and the shared K-pe, concatenated) and V."""
    heads = layer.num_heads_tp
    kv = project_context_kv(layer, compressed_kv)
    k_nope, v = kv.split([heads * layer.qk_nope_head_dim, heads * layer.v_head_dim], -1)
    k_nope = k_nope.view(-1, heads, layer.qk_nope_head_dim)
    k_pe = k_pe.view(-1, 1, layer.qk_rope_head_dim)
    k = maybe_compiled_cat((k_nope, k_pe.expand(-1, heads, -1)), dim=-1)
    return k.view(-1, heads * layer.qk_head_dim), v


def _expand_kv_fp8(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor):
    """kv_b, then FP8 K ``[T, H * 192]`` and V ``[T, H * 128]`` in one pass: the bytes the
    attention op's context copy writes for the BF16 K and V."""
    return mla_kv_to_fp8(
        project_context_kv(layer, compressed_kv),
        k_pe.view(-1, layer.qk_rope_head_dim),
        layer.num_heads_tp,
        layer.qk_nope_head_dim,
        layer.v_head_dim,
    )


def _fp8_q_args(layer, q_fp8):
    """Attention-op arguments that hand it ``q_fp8`` as the already-quantized FP8 Q."""
    if q_fp8 is None:
        return {}
    return {"quant_q_buffer": q_fp8, "quant_scale_qkv": layer.mha.kv_scale_orig_quant}


def _quantize_q_fp8(layer, q: torch.Tensor) -> torch.Tensor:
    """Q in FP8 as the attention op's context copy quantizes it (unit scale)."""
    q_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        q, layer.mha.kv_scale_orig_quant
    )
    return q_fp8


@functools.cache
def _warmup_expand_kv_fp8(
    num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, device
):
    """Compile ``mla_kv_to_fp8`` before the first cached-KV context step (k_pe contiguous for
    loaded chunks, a ``latent_cache`` view for the current tokens)."""
    num_tokens = 2
    kv = torch.zeros(
        num_tokens, num_heads * (qk_nope_head_dim + v_head_dim), dtype=torch.bfloat16, device=device
    )
    latent = torch.zeros(
        num_tokens, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16, device=device
    )
    for k_pe in (latent[:, kv_lora_rank:].contiguous(), latent[:, kv_lora_rank:]):
        mla_kv_to_fp8(kv, k_pe, num_heads, qk_nope_head_dim, v_head_dim)


@functools.cache
def _warmup_expand_kv_bf16(
    num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, dtype, device
):
    """Compile ``maybe_compiled_cat`` for both K layouts (a loaded chunk's contiguous K-pe and the
    current tokens' ``latent_cache`` view) with the token dim dynamic; one token count still
    specializes, so it is warmed separately."""

    def warmup(num_tokens):
        k_nope = torch.empty(
            num_tokens, num_heads * (qk_nope_head_dim + v_head_dim), dtype=dtype, device=device
        )[:, : num_heads * qk_nope_head_dim].view(num_tokens, num_heads, qk_nope_head_dim)
        chunked_k_pe = torch.empty(
            num_tokens, 1, qk_rope_head_dim, dtype=dtype, device=device
        ).expand(-1, num_heads, -1)
        k_pe = torch.empty(
            num_tokens, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
        )[:, :, -qk_rope_head_dim:].expand(-1, num_heads, -1)
        torch._dynamo.maybe_mark_dynamic(k_nope, 0)
        torch._dynamo.maybe_mark_dynamic(chunked_k_pe, 0)
        torch._dynamo.maybe_mark_dynamic(k_pe, 0)
        maybe_compiled_cat((k_nope, chunked_k_pe), dim=-1)
        maybe_compiled_cat((k_nope, k_pe), dim=-1)

    warmup(2)
    warmup(1)


def _kv_b_strided_out(layer) -> bool:
    """Whether the no-prefix expansion writes K-nope / V straight into the attention inputs with
    per-head BMMs: a bias-free BF16 kv_b on SM100f."""
    w = layer.kv_b_proj.weight
    return (
        is_sm_100f()
        and w.dtype == torch.bfloat16
        and tuple(w.shape)
        == (layer.num_heads_tp * (layer.qk_nope_head_dim + layer.v_head_dim), layer.kv_lora_rank)
        and getattr(layer.kv_b_proj, "bias", None) is None
    )


def context_default(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: torch.Tensor,
) -> torch.Tensor:
    """Context attention without a cached prefix (see the module docstring)."""
    heads = layer.num_heads_tp
    nope, vdim = layer.qk_nope_head_dim, layer.v_head_dim
    k = torch.empty_like(q).view(-1, heads, layer.qk_head_dim)
    if _kv_b_strided_out(layer):
        num_tokens = compressed_kv.shape[0]
        w = layer.kv_b_proj.weight
        nope_rows = layer.num_heads_tp * nope
        w_k = w[:nope_rows].view(layer.num_heads_tp, nope, layer.kv_lora_rank)
        w_v = w[nope_rows:].view(layer.num_heads_tp, vdim, layer.kv_lora_rank)
        ckv = compressed_kv.unsqueeze(0).expand(heads, num_tokens, layer.kv_lora_rank)
        # [H, T, nope] view of K with strides (192, H * 192, 1).
        k_nope_out = k[..., :nope].transpose(0, 1)
        # The context FP8 quantize kernel reads V with the [K-nope | V] row stride of the kv_b
        # output: V is the second half of such a buffer (its first half is never written).
        kv_buf = torch.empty(num_tokens, heads * (nope + vdim), dtype=q.dtype, device=q.device)
        v = kv_buf[:, heads * nope :]
        v_out = v.unflatten(-1, (heads, vdim)).transpose(0, 1)
        bmm_out(ckv, w_k.transpose(1, 2), k_nope_out)
        bmm_out(ckv, w_v.transpose(1, 2), v_out)
    else:
        kv = project_context_kv(layer, compressed_kv)
        k_nope, v = kv.split([heads * nope, heads * vdim], -1)
        maybe_compiled_copy_(k[..., :nope], k_nope.view(-1, heads, nope))
    # With the RoPE fused in the op the K-pe slot stays unwritten; the op reads latent_cache.
    k = k.view(-1, heads * layer.qk_head_dim)
    if layer.mha.uses_fp4_mla_attention:
        # The FP8 context FMHA over the step's FP8 scratch cache beside the FP4 scatter.
        return mla_cache.fp4_mla_context_attention(layer.mha, q, k, v, md, output, latent_cache)
    return mla_attn.mla_context_attention(layer.mha, q, k, v, md, output, latent_cache=latent_cache)


def context_cached_kv(
    layer,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Context attention over the whole paged cache (cached prefix, no chunked prefill)."""
    mha = layer.mha
    mla_cache.mla_rope_append_paged_kv_assign_q(mha, q, latent_cache, md)
    full_compressed_kv, full_k_pe = mla_cache.load_paged_kv_cache_for_mla(mha, md, q.dtype)
    assert full_compressed_kv.shape == (
        md.num_ctx_cached_tokens + md.num_ctx_tokens,
        layer.kv_lora_rank,
    )
    assert full_k_pe.shape == (md.num_ctx_cached_tokens + md.num_ctx_tokens, layer.qk_rope_head_dim)
    assert full_compressed_kv.is_contiguous() and full_k_pe.is_contiguous()
    q_fp8 = None
    if layer.fp8_cached_kv_context:
        q_fp8 = _quantize_q_fp8(layer, q)
        full_k, full_v = _expand_kv_fp8(layer, full_compressed_kv, full_k_pe)
    else:
        full_k, full_v = _expand_kv_bf16(layer, full_compressed_kv, full_k_pe)
    full_compressed_kv = full_k_pe = None
    return mla_attn.mla_context_attention(
        mha, q, full_k, full_v, md, output, **_fp8_q_args(layer, q_fp8)
    )


def context_chunked_prefill(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    latent_cache: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Chunked-prefill context attention over an FP8 / BF16 cache (see the module docstring)."""
    mha = layer.mha
    heads = layer.num_heads_tp
    mla_cache.mla_rope_append_paged_kv_assign_q(mha, q, latent_cache, md)
    fp8 = layer.fp8_cached_kv_context
    expand = _expand_kv_fp8 if fp8 else _expand_kv_bf16
    # With FP8 attention inputs, Q is quantized once for every partition.
    q_fp8 = _quantize_q_fp8(layer, q) if fp8 else None
    q_args = _fp8_q_args(layer, q_fp8)
    softmax_stats = torch.empty((md.num_ctx_tokens, heads, 2), dtype=torch.float, device="cuda")
    temp_softmax_stats = torch.empty(
        (md.num_ctx_tokens, heads, 2), dtype=torch.float, device="cuda"
    )
    temp_output = q.new_empty((q.size(0), heads * layer.v_head_dim), dtype=q.dtype)
    buffer_batch_size = md.runtime_features.chunked_prefill_buffer_batch_size

    # Each partition attends with its own KV lengths; the originals are restored at the end.
    origin_kv_lens_cuda_runtime = md.kv_lens_cuda_runtime
    origin_kv_lens_runtime = md.kv_lens_runtime
    origin_ctx_total_kv_len = md.host_total_kv_lens[0].item()

    for loop_idx in range(md.chunked_loop_num):
        total_chunked_tokens = md.host_cu_chunked_seq_len[loop_idx, md.num_contexts]
        chunked_compressed_kv, chunked_k_pe = mla_cache.load_chunked_kv_cache_for_mla(
            mha,
            md,
            num_ctx_cached_tokens=total_chunked_tokens,
            cu_chunked_seq_len=md.cu_chunked_seq_len[loop_idx],
            chunked_global_offset=md.chunked_global_offset[loop_idx],
            chunked_max_seq_len=md.max_chunk_len_per_loop[loop_idx],
            out_dtype=q.dtype,
        )
        chunked_k, chunked_v = expand(layer, chunked_compressed_kv, chunked_k_pe)
        chunked_compressed_kv = chunked_k_pe = None
        md.kv_lens_runtime = md.host_chunked_seq_len[loop_idx]
        md.kv_lens_cuda_runtime = md.chunked_seq_len[loop_idx]
        md.host_total_kv_lens[0] = total_chunked_tokens
        temp_output = mla_attn.mla_context_attention(
            mha,
            q,
            chunked_k,
            chunked_v,
            md,
            temp_output,
            full_mask=True,
            softmax_stats=temp_softmax_stats,
            chunked_prefill_buffer_batch_size=buffer_batch_size,
            **q_args,
        )
        mla_cache.merge_attention_for_mla(
            mha,
            output,
            temp_output,
            softmax_stats,
            temp_softmax_stats,
            md.merge_op_tensor[loop_idx],
            md,
        )

    # The current chunk, causal; its rotated K-pe is in latent_cache.
    _, k_pe = latent_cache.view([-1, layer.kv_lora_rank + layer.qk_rope_head_dim]).split(
        [layer.kv_lora_rank, layer.qk_rope_head_dim], -1
    )
    k, v = expand(layer, compressed_kv, k_pe)
    md.kv_lens_runtime = md.prompt_lens_cpu_runtime
    md.kv_lens_cuda_runtime = md.prompt_lens_cuda_runtime
    md.host_total_kv_lens[0] = md.prompt_lens_cpu_runtime[: md.num_contexts].sum().item()
    temp_output = mla_attn.mla_context_attention(
        mha,
        q,
        k,
        v,
        md,
        temp_output,
        softmax_stats=temp_softmax_stats,
        chunked_prefill_buffer_batch_size=buffer_batch_size,
        **q_args,
    )
    mla_cache.merge_attention_for_mla(
        mha,
        output,
        temp_output,
        softmax_stats,
        temp_softmax_stats,
        md.merge_op_tensor[md.chunked_loop_num],
        md,
    )
    md.kv_lens_runtime = origin_kv_lens_runtime
    md.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime
    md.host_total_kv_lens[0] = origin_ctx_total_kv_len
    return output


def forward_context(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: torch.Tensor,
) -> torch.Tensor:
    """Context attention of the context rows into ``output``: picks the chain by cache state."""
    mha = layer.mha
    if mha.has_cached_kv_for_mla_context_warmup(md):
        dims = (
            layer.num_heads_tp,
            layer.qk_nope_head_dim,
            layer.qk_rope_head_dim,
            layer.kv_lora_rank,
            layer.v_head_dim,
        )
        if layer.fp8_cached_kv_context:
            _warmup_expand_kv_fp8(*dims, q.device)
        else:
            _warmup_expand_kv_bf16(*dims, q.dtype, q.device)
    chunked = mha.is_chunked_prefill_for_mla_context(md)
    if chunked and get_sm_version() >= 100:
        if mha.uses_fp4_mla_attention:
            if not uses_k3_split_k_context(layer, q, output):
                raise NotImplementedError(
                    "Kimi K3 MLA: an FP4 MLA cache attends a cached prefix only through the Rubin "
                    "FP8 split-K context FMHA (SM107, BF16 Q and output, see ctx_split_k)"
                )
            return forward_context_split_k_fp4(layer, q, compressed_kv, latent_cache, md, output)
        return context_chunked_prefill(layer, q, compressed_kv, latent_cache, md, output)
    if chunked or mha.has_cached_kv_for_mla_context(md):
        if mha.uses_fp4_mla_attention:
            raise RuntimeError(
                "FP4 MLA attends a cached prefix only through chunked-prefill partitions on "
                "SM100+; enable chunked prefill."
            )
        return context_cached_kv(layer, q, latent_cache, md, output)
    return context_default(layer, q, compressed_kv, md, output, latent_cache)


def gen_attention(layer, q: torch.Tensor, md: AttentionMetadata, **kwargs) -> torch.Tensor:
    """The absorbed decode of ``q`` (``[T, H * 576]``) into a new ``[T, H * kv_lora_rank]``: K3's
    packed FP4 decode on an FP4 cache with the CuTe DSL FP4 MLA backend (no page plan, no PDL
    reduction); on an FP8 or BF16 cache the CuTe DSL MLA decode where it serves the batch, else
    the attention op's MLA generation (``ops/attention/k3_mla_attention``). ``kwargs``: the buffers of
    the generation cache update (``mla_rope_generation``) and ``softmax_stats_tensor``."""
    if fp4_mla_decode.applies(layer.mqa):
        assert kwargs.get("out_scale") is None
        return _decode_op.k3_fp4_mla_decode(
            layer.mqa,
            q,
            md,
            softmax_stats_tensor=kwargs.get("softmax_stats_tensor"),
            page_plan=False,
            reduce_pdl=False,
        )
    mqa = layer.mqa
    if mqa.uses_fp4_mla_attention:
        raise NotImplementedError(
            "Kimi K3 MLA: an FP4 MLA cache decodes on the CuTe DSL FP4 MLA backend (SM107)"
        )
    assert kwargs.pop("out_scale", None) is None
    softmax_stats = kwargs.pop("softmax_stats_tensor", None)
    output = q.new_empty((q.shape[0], mqa.num_heads * mqa.kv_lora_rank))
    return mla_attn.mla_decode(mqa, q, md, output, softmax_stats=softmax_stats, **kwargs)


def forward_absorption_generation(
    layer,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Absorbed generation of the ``q`` rows (``[T, H * 192]``, q_b's output) into ``output``
    (``[T, H * v_head_dim]``); ``latent_cache`` ``[T, 576]`` is the normalized latent and
    K-pe the cache update appends."""
    num_tokens = q.shape[0]
    heads, lora, rope = layer.num_heads_tp, layer.kv_lora_rank, layer.qk_rope_head_dim
    device = q.device
    q_nope, q_pe = q.view([-1, heads, layer.qk_head_dim]).split(
        [layer.qk_nope_head_dim, rope], dim=-1
    )
    num_seqs = md.num_seqs
    # The RoPE kernel writes seqQOffset[batch + 1] and scans total / per-sequence lengths over the
    # presented rows: one per token under the ragged (token-major) verify presentation.
    cu_rows = (
        max(num_seqs, num_tokens) + 1 if getattr(md, "is_ragged_verify", False) else num_seqs + 1
    )
    cu_q_seqlens = torch.empty(cu_rows, dtype=torch.int32, device=device)
    cu_kv_seqlens = torch.empty(cu_rows, dtype=torch.int32, device=device)
    fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)
    mla_bmm1_scale = mla_bmm2_scale = quant_q_buffer = None
    if layer.quantized_kv:
        mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
        mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
        quant_q_buffer = torch.empty(
            num_tokens, heads, lora + rope, dtype=torch.uint8, device=device
        )
    # [T, H, kv_lora_rank | rope]: the W_UK output and the rotated q_pe the RoPE kernel writes.
    fused_q = torch.empty([num_tokens, heads, lora + rope], dtype=q.dtype, device=device)
    rope_kwargs = {"fuse_fp4_q_quant": True} if layer.fp4_kv else {}

    def rope_and_cache_update():
        mla_cache.mla_rope_generation(
            layer.mqa,
            fused_q,
            q_pe,
            latent_cache,
            md,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            **rope_kwargs,
        )

    # [H, T, qk_nope] x [H, qk_nope, kv_lora_rank] -> [H, T, kv_lora_rank], written into fused_q.
    q_nope_out = fused_q[..., :lora].transpose(0, 1)
    if layer.k_b_proj_trans.dtype == torch.bfloat16:

        def w_uk():
            bmm_out(q_nope.transpose(0, 1), layer.k_b_proj_trans.transpose(1, 2), q_nope_out)

    else:
        w_uk_bf16 = layer.k_b_proj_trans_dequant
        if w_uk_bf16 is None:
            raise NotImplementedError(
                "Kimi K3 MLA: the FP8 absorption BMM runs on the dequantized weight (SM100f)"
            )

        def w_uk():
            torch.bmm(q_nope.transpose(0, 1), w_uk_bf16.transpose(1, 2), out=q_nope_out)

    # The quantized-cache RoPE kernel writes the quantized query too: same stream as the decode.
    rope_stream = None if layer.quantized_kv else layer.aux_stream
    fork_join(w_uk, rope_and_cache_update, layer.ln_events[0], layer.ln_events[1], rope_stream)
    fused_q = fused_q.view([num_tokens, heads * (lora + rope)])

    attn_out_latent = gen_attention(
        layer,
        fused_q,
        md,
        out_scale=None,
        latent_cache=latent_cache,
        q_pe=q_pe,
        cu_q_seqlens=cu_q_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        fmha_scheduler_counter=fmha_scheduler_counter,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        quant_q_buffer=quant_q_buffer,
    )
    fused_q = None
    heads_out = layer.num_heads_tp
    assert attn_out_latent.shape == (num_tokens, heads_out * lora)
    latent = attn_out_latent.view([-1, heads_out, lora])
    attn_output = output.view([num_tokens, heads_out, layer.v_head_dim])
    # [H, T, kv_lora_rank] x [H, kv_lora_rank, v_head_dim] -> [H, T, v_head_dim].
    if layer.v_b_proj.dtype == torch.bfloat16:
        bmm_out(latent.transpose(0, 1), layer.v_b_proj.transpose(1, 2), attn_output.transpose(0, 1))
    else:
        torch.bmm(
            latent.transpose(0, 1),
            layer.v_b_proj_dequant.transpose(1, 2),
            out=attn_output.transpose(0, 1),
        )
    return output


# ---------------------------------------------------------------------------------------------
# The attention part under breakable CUDA graphs: the layer is found by its index string, so the
# break's frozen arguments are tensors and a string.
# ---------------------------------------------------------------------------------------------


def _registered_layer(layer_idx: str):
    """The step's attention metadata and the MLA layer registered under ``layer_idx``."""
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"
    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, TrtllmAttentionMetadata), "the MLA metadata must be TRTLLM's"
    mla_layers = extra_attrs.get("mla_layers", None)
    assert mla_layers is not None, "Attention layer is not registered"
    layer_ref = mla_layers.get(layer_idx, None)
    assert layer_ref is not None, f"Cannot find attention layer for layer {layer_idx}"
    layer = layer_ref()
    assert isinstance(layer, K3MLAAttention), "the registered MLA layer must be K3's"
    return metadata, layer


def create_mla_outputs(hidden_states: torch.Tensor, layer_idx: str) -> torch.Tensor:
    """The attention output buffer of the registered layer ``layer_idx`` for these rows."""
    _, layer = _registered_layer(layer_idx)
    return layer.create_output(hidden_states)


def mla_custom_op_inplace(
    hidden_states: torch.Tensor, layer_idx: str, output: torch.Tensor
) -> None:
    """The attention part of the registered layer ``layer_idx`` into ``output``, on the step's
    metadata registered with it."""
    metadata, layer = _registered_layer(layer_idx)
    layer.run_attention(hidden_states, metadata, output)


_bcg_mla_custom_op_inplace = eager_on_graph(mla_custom_op_inplace)


# ---------------------------------------------------------------------------------------------
# K3MLAAttention.
# ---------------------------------------------------------------------------------------------


class K3MLAAttention(nn.Module):
    """Kimi K3 MLA (see the module docstring)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        max_position_embeddings: int = 8192,
        model_config: ModelConfig,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        super().__init__()
        config = model_config
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.max_position_embeddings = max_position_embeddings
        self.use_output_gate = use_output_gate
        # The checkpoint's separate q_a / kv_a rows load into one fused projection.
        self.fuse_qkv_a_proj = True
        self.out_scale = None
        if rms_norm_eps is None:
            rms_norm_eps = getattr(config.pretrained_config, "rms_norm_eps", None)
        if rms_norm_eps is None:
            rms_norm_eps = 1e-6
        self.kv_cache_dtype = config.extra_attrs.get("kv_cache_dtype", "auto")

        # Each projection's checkpoint recipe: FP8 128 x 128 block scales or BF16.
        projection_configs = config.quant_config_dict or {}

        def is_fp8(name: str) -> bool:
            algo = getattr(projection_configs.get(name, config.quant_config), "quant_algo", None)
            if algo not in (None, QuantAlgo.FP8_BLOCK_SCALES):
                raise ValueError(f"Kimi K3 MLA: {name} has unsupported quantization {algo}")
            return algo == QuantAlgo.FP8_BLOCK_SCALES

        fp8 = {
            name: is_fp8(name)
            for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj")
            + (("g_proj",) if use_output_gate else ())
        }
        self._reject_unsupported(config, fp8)

        # Registered for the custom ops; a later stack (one-model speculative decoding) with the
        # same local layer index registers as "<layer_idx>_<n>".
        self.register_to_config = True
        registry = config.extra_attrs.setdefault("mla_layers", {})
        suffix = 0
        while self.layer_idx_str in registry:
            self.layer_idx_str = f"{layer_idx}_{suffix}"
            suffix += 1
        registry[self.layer_idx_str] = weakref.ref(self)

        # Head sharding: q_b / kv_b / o_proj / g_proj over TP (attention DP: none; context
        # parallelism is rejected above, cp_size is 1).
        self.mapping = config.mapping
        tp_size, pp_size, cp_size = self.mapping.tp_size, self.mapping.pp_size, self.mapping.cp_size
        dp_size = 1
        if self.mapping.enable_attention_dp:
            dp_size, tp_size = tp_size, 1
        world_size = pp_size * dp_size * tp_size * cp_size
        mapping = Mapping(
            world_size=world_size,
            tp_size=tp_size,
            pp_size=pp_size * dp_size,
            cp_size=cp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        self.mapping_o = Mapping(
            world_size=world_size,
            tp_size=tp_size * cp_size,
            pp_size=pp_size * dp_size,
            cp_size=1,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        assert num_heads % (tp_size * cp_size) == 0
        self.num_heads_tp = num_heads // tp_size

        self.aux_stream = (aux_stream_dict or {}).get(AuxStreamType.Attention)
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        def projection(fp8: bool, in_features: int, out_features: int, mode, mp: Optional[Mapping]):
            shard = {} if mp is None else {"tp_size": mp.tp_size, "tp_rank": mp.tp_rank}
            if fp8:
                return Fp8BlockScaleLinear(in_features, out_features, tp_mode=mode, **shard)
            return Bf16Linear(in_features, out_features, dtype=dtype, tp_mode=mode, **shard)

        column, row = TensorParallelMode.COLUMN, TensorParallelMode.ROW
        self.kv_a_proj_with_mqa = projection(
            fp8["kv_a_proj_with_mqa"],
            hidden_size,
            q_lora_rank + kv_lora_rank + qk_rope_head_dim,
            None,
            None,
        )
        if fp8["kv_a_proj_with_mqa"]:
            # q_lora + kv_lora + rope rows (2112 for K3) leave a partial last 128-row tile,
            # which the FP8 low-M GEMM serves only on request.
            self.kv_a_proj_with_mqa.fp8w_lowm_partial_n_tile = True
        self.q_a_layernorm = RMSNormWeight(q_lora_rank, rms_norm_eps, dtype=dtype)
        self.q_b_proj = projection(
            fp8["q_b_proj"], q_lora_rank, num_heads * self.qk_head_dim, column, mapping
        )
        self.kv_a_layernorm = RMSNormWeight(kv_lora_rank, rms_norm_eps, dtype=dtype)
        self.kv_b_proj = projection(
            fp8["kv_b_proj"],
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            column,
            mapping,
        )
        self._kv_b_fp8 = isinstance(self.kv_b_proj, Fp8BlockScaleLinear)
        # The V absorption of the tp-local heads (bound by the loader).
        self.v_b_proj = nn.Parameter(
            torch.empty((self.num_heads_tp, v_head_dim, kv_lora_rank), dtype=dtype),
            requires_grad=False,
        )
        self.o_proj = projection(
            fp8["o_proj"], num_heads * v_head_dim, hidden_size, row, self.mapping_o
        )
        self.attention_output_hidden_size = self.o_proj.in_features
        if use_output_gate:
            # o_proj's input sharding.
            self.g_proj = projection(
                fp8["g_proj"], hidden_size, num_heads * v_head_dim, column, self.mapping_o
            )

        quant_config = config.get_quant_config()
        self.quant_config = quant_config
        # The two attention roles over the engine's KV cache (``ops/attention/k3_mla_state``): the
        # absorbed generation (``mqa``: one 576-dim latent head per query head) and the context
        # (``mha``: 192 / 128-dim heads), each with the identity RoPE table (K3 is NoPE).
        role_dims = dict(
            layer_idx=layer_idx,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            skip_correction_threshold=config.skip_correction_threshold,
        )
        self.mqa = K3MlaAttnState.generation(
            num_heads=self.num_heads_tp,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            **role_dims,
        )
        self.mha = K3MlaAttnState.context(
            num_context_heads=self.num_heads_tp,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            **role_dims,
        )

        # Set in create_weights() from the backends' KV cache quantization.
        self.quantized_kv = False
        self.fp4_kv = False
        self.fp8_cached_kv_context = False
        self._weights_transformed = False

        if not config.skip_create_weights_in_init:
            self.create_weights()

    @staticmethod
    def _reject_unsupported(config: ModelConfig, fp8: dict[str, bool]) -> None:
        """Configurations whose kernel chain K3 does not carry."""
        mapping = config.mapping
        unsupported = []
        if config.attn_backend.upper() != "TRTLLM":
            unsupported.append(f"attention backend {config.attn_backend} (TRTLLM only)")
        if config.sparse_attention_config is not None:
            unsupported.append("sparse attention")
        if fp8["q_a_proj"] != fp8["kv_a_proj_with_mqa"]:
            unsupported.append("q_a / kv_a with different recipes (no fused q_a / kv_a GEMM)")
        if config.extra_attrs.get("kv_cache_dtype", "auto") == "fp8_ds_mla":
            unsupported.append("the fp8_ds_mla KV cache")
        if hasattr(config.pretrained_config, "llama_4_scaling"):
            unsupported.append("Llama-4 attention scaling")
        # The CuTe DSL GEMM / BMM opt-ins of main's MLA, where they would select another kernel:
        # BF16 projections, BF16 absorption BMMs (BF16 kv_b), FP8 absorption BMMs.
        if config.use_cute_dsl_bf16_gemm and not all(fp8.values()):
            unsupported.append("use_cute_dsl_bf16_gemm with BF16 projections")
        if config.use_cute_dsl_bf16_bmm and not fp8["kv_b_proj"]:
            unsupported.append("use_cute_dsl_bf16_bmm")
        if config.use_cute_dsl_blockscaling_bmm and fp8["kv_b_proj"]:
            unsupported.append("use_cute_dsl_blockscaling_bmm")
        if mapping.cp_size > 1:
            unsupported.append(f"context parallelism (cp_size {mapping.cp_size})")
        if unsupported:
            raise NotImplementedError("Kimi K3 MLA does not support: " + "; ".join(unsupported))

    def create_weights(self) -> None:
        """Backend KV cache quantization and the absorption weights (their dtype follows kv_b)."""
        self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)
        self.quantized_kv = bool(self.mqa.has_fp8_kv_cache or self.mqa.has_fp4_kv_cache)
        self.fp4_kv = bool(
            self.mqa.support_fp4_kv_cache()
            and self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_fp4_kv_cache()
        )
        if self.fp4_kv and not self.mqa.rope_duplicate_data:
            raise ValueError("FP4 MLA requires RopeParams.duplicate_data=True.")
        # Cached-prefix context on the FP8 context FMHA takes FP8 Q / K / V (plain conversions at
        # the op's unit KV scale).
        mha = self.mha
        scale = mha.kv_scale_orig_quant
        self.fp8_cached_kv_context = bool(
            mha.has_fp8_context_mla
            and (self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim) == (128, 64, 128)
            and isinstance(scale, torch.Tensor)
            and bool(torch.all(scale == 1.0).item())
        )
        heads = self.num_heads_tp
        lora, nope, vdim = self.kv_lora_rank, self.qk_nope_head_dim, self.v_head_dim
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (heads, lora, nope), dtype=torch.float8_e4m3fn if self._kv_b_fp8 else self.dtype
            ),
            requires_grad=False,
        )
        self.k_b_proj_trans_dequant = None
        self.v_b_proj_dequant = None
        if self._kv_b_fp8:
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty((heads, lora // 128, nope // 128), dtype=torch.float32),
                requires_grad=False,
            )
            self.v_b_proj_scale = nn.Parameter(
                torch.empty((heads, vdim // 128, lora // 128), dtype=torch.float32),
                requires_grad=False,
            )
            if is_sm_100f():
                # The absorption BMMs run in BF16 on dequantized copies.
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty((heads, lora, nope), dtype=self.dtype), requires_grad=False
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty((heads, vdim, lora), dtype=self.dtype), requires_grad=False
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None
        self._weights_transformed = False

    # -----------------------------------------------------------------------------------------
    # Post-load.
    # -----------------------------------------------------------------------------------------

    def transform_weights(self) -> None:
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.cache_derived_state()

    # -----------------------------------------------------------------------------------------
    # Front.
    # -----------------------------------------------------------------------------------------

    def project_qkv_a(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The fused q_a / kv_a projection ``[T, q_lora | kv_lora | rope]``: the GEMM on the BF16
        rows (a BF16 weight: ``cublas_mm``)."""
        proj = self.kv_a_proj_with_mqa
        if isinstance(proj, Bf16Linear):
            return cublas_mm(
                hidden_states,
                proj.weight.t(),
                None,
                out_dtype=None,
                output_buffer_kind=int(BufferKind.DEFAULT),
                group=None,
            )
        return proj(hidden_states)

    def q_a_norm(self, q: torch.Tensor) -> torch.Tensor:
        """The q_a RMSNorm (FlashInfer)."""
        norm = self.q_a_layernorm
        return rms_norm(q, norm.weight, norm.variance_epsilon)

    def _project_q_by_phase(
        self, q: torch.Tensor, num_ctx_tokens: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """q_b of the context rows and of the generation rows."""
        projected = self.q_b_proj(q)
        return projected[:num_ctx_tokens], projected[num_ctx_tokens:]

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        md: AttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """Front, context and generation attention of the batch's rows into ``output``."""
        num_ctx_tokens = md.num_ctx_tokens
        num_tokens = md.num_tokens
        hidden_states = hidden_states[:num_tokens, ...]
        q, compressed_kv, k_pe = self.project_qkv_a(hidden_states).split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        kv_norm = self.kv_a_layernorm
        q, compressed_kv = fork_join(
            lambda: self.q_a_norm(q),
            lambda: rms_norm(compressed_kv, kv_norm.weight, kv_norm.variance_epsilon),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        (q_ctx, q_gen), latent_cache = fork_join(
            lambda: self._project_q_by_phase(q, num_ctx_tokens),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        if md.num_contexts > 0:
            forward_context(
                self,
                q_ctx,
                compressed_kv[:num_ctx_tokens, ...],
                md,
                output[:num_ctx_tokens, :],
                latent_cache[:num_ctx_tokens, ...],
            )
        if md.num_generations > 0:
            assert q_gen is not None
            forward_absorption_generation(
                self,
                q_gen,
                latent_cache[num_ctx_tokens:, ...],
                md,
                output[num_ctx_tokens:num_tokens, :],
            )

    def run_attention(
        self,
        hidden_states: torch.Tensor,
        md: AttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """The attention part of a forward (everything before the output gate) into ``output``,
        on the mixed-precision KV staging copy where the metadata has one."""
        with md.mixed_precision_kv_staging(self.layer_idx):
            self._forward_attention(hidden_states, md, output)

    def create_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The attention output ``[rows, H * v_head_dim]`` for the layer input's rows."""
        return hidden_states.new_empty(
            [hidden_states.shape[0], self.attention_output_hidden_size], dtype=hidden_states.dtype
        )

    # -----------------------------------------------------------------------------------------
    # Forward.
    # -----------------------------------------------------------------------------------------

    def _gate_and_o_proj(self, hidden_states: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        if self.use_output_gate:
            attn_out = attn_out * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attn_out)

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """o_proj's output of the layer for ``hidden_states`` (K3 is NoPE:
        ``position_ids`` is None)."""
        assert position_ids is None, "Kimi K3 MLA is NoPE"
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"Kimi K3 MLA takes BF16 rows, got {type(hidden_states).__name__}")
        if self.register_to_config:
            output = create_mla_outputs(hidden_states, self.layer_idx_str)
            _bcg_mla_custom_op_inplace(hidden_states, self.layer_idx_str, output)
        else:
            output = self.create_output(hidden_states)
            self.run_attention(hidden_states, attn_metadata, output)
        return self._gate_and_o_proj(hidden_states, output)


# Identity-RoPE table positions for the MLA backends: max_position_embeddings. K3 is NoPE (the
# table holds cos=1/sin=0), but the chunked-context path indexes the table by absolute position,
# so it must cover max_position_embeddings (~512MB per backend for the 1M-position checkpoint); a
# smaller table is read out of bounds.


def _kimi_k3_fp8_kv_layer_ids(cfg, kv_quant_algo: Optional[QuantAlgo]) -> frozenset:
    """MLA layers whose KV cache falls back to FP8 under NVFP4 KV: the engine's cache policy
    decides (``kimi_k3_cache_policy.get_kimi_k3_fp8_kv_layer_ids``, which the FP4 MLA cache
    manager reads too, so the layer and its cache agree; under NVFP4 the last two MLA layers)."""
    return _kimi_k3_cache_policy.get_kimi_k3_fp8_kv_layer_ids(
        cfg, nvfp4_enabled=kv_quant_algo == QuantAlgo.NVFP4
    )


def resolve_attention_quant_config(
    config: ModelConfig | None, layer_idx: int, projection: str
) -> QuantConfig:
    """Resolve a checkpoint projection, including mixed-precision exclusions."""
    if config is None:
        return QuantConfig()
    global_config = config.quant_config or QuantConfig()
    names = [
        f"{prefix}layers.{layer_idx}.self_attn.{projection}"
        for prefix in ("language_model.model.", "model.", "")
    ]
    if any(global_config.is_module_excluded_from_quantization(name) for name in names):
        return QuantConfig(kv_cache_quant_algo=global_config.kv_cache_quant_algo)
    declarations = config.quant_config_dict or {}
    matches = [declarations[name] for name in names if name in declarations]
    if matches:
        selected = matches[0]
        if any(match.quant_algo != selected.quant_algo for match in matches[1:]):
            raise ValueError(f"Conflicting Kimi K3 quantization aliases for {names[0]}")
    elif global_config.quant_algo == QuantAlgo.MIXED_PRECISION:
        selected = QuantConfig()
    else:
        selected = global_config
    if selected.quant_algo not in (None, QuantAlgo.FP8_BLOCK_SCALES):
        raise ValueError(
            f"Kimi K3 attention projection {names[0]} has unsupported checkpoint "
            f"quantization {selected.quant_algo}"
        )
    if selected.quant_algo == QuantAlgo.FP8_BLOCK_SCALES and selected.group_size not in (None, 128):
        raise ValueError(f"Kimi K3 attention requires 128x128 FP8 blocks for {names[0]}")
    result = copy.copy(selected)
    result.kv_cache_quant_algo = global_config.kv_cache_quant_algo
    return result


# ---------------------------------------------------------------------------
# MLA runtime.
# ---------------------------------------------------------------------------


class K3MLARuntime(nn.Module):
    """Wraps K3 MLA. Attention runs data-parallel (dep16: attention DP; the single-rank harness:
    tp 1), so o_proj's output is the layer's full output: there is no TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        super().__init__()

        max_positions = int(cfg.max_position_embeddings)
        self.layer_idx = layer_idx
        attention_config = copy.copy(model_config)
        attention_config._frozen = False
        attention_config.quant_config_dict = {
            name: resolve_attention_quant_config(model_config, layer_idx, name)
            for name in (
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "q_b_proj",
                "kv_b_proj",
                "g_proj",
                "o_proj",
            )
        }
        kv_quant_algo = (
            model_config.quant_config.kv_cache_quant_algo
            if model_config.quant_config is not None
            else None
        )
        fp8_kv_layer_ids = _kimi_k3_fp8_kv_layer_ids(cfg, kv_quant_algo)
        if layer_idx in fp8_kv_layer_ids:
            if kv_quant_algo != QuantAlgo.NVFP4:
                raise ValueError(
                    "The Kimi K3 FP8 KV-cache fallback layers require the model-wide KV cache "
                    "dtype to be NVFP4."
                )
            kv_quant_algo = QuantAlgo.FP8
            logger.info_once(
                f"Kimi K3 MLA layers using FP8 KV-cache fallback: {sorted(fp8_kv_layer_ids)}.",
                key="kimi_k3_fp8_kv_fallback_layers",
            )
            for projection_config in attention_config.quant_config_dict.values():
                projection_config.kv_cache_quant_algo = kv_quant_algo
        attention_config.quant_config = QuantConfig(kv_cache_quant_algo=kv_quant_algo)
        attention_config._frozen = model_config._frozen
        self.mixer = K3MLAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            use_output_gate=cfg.mla_use_output_gate,
            max_position_embeddings=max_positions,
            model_config=attention_config,
            aux_stream_dict=aux_stream_dict,
        )

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        # MLA.forward takes position_ids first; K3 is NoPE, so pass None.
        return self.mixer(None, hidden_states, attn_metadata)


# ---------------------------------------------------------------------------
# FP8 block-scale weight read for attention and replicated MoE-layer MLP projections.
# ---------------------------------------------------------------------------


def _rubin_mxfp8_is_available() -> bool:
    """Whether this rank can run the SM107 native block-scaled MXFP8 GEMM.

    Recomputed rather than cached: the SM version cannot change within a run,
    so weight preparation and forward always agree.
    """
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _is_fp8_block_projection(module: nn.Module) -> bool:
    """A checkpoint FP8 128 x 128 block-scale projection: the loader reads its codes and scales
    together and post-load swaps it for ``_Fp8BlockScaleWeightReadLinear``."""
    return isinstance(module, Fp8BlockScaleLinear)


def _k3_lowm_gemm_can_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    allow_partial_n_tile: bool = False,
) -> bool:
    """Whether the K3 low-M GEMM (``ops.gemm.k3_fp8w_lowm_gemm``) serves this projection."""
    from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm.can_run(
        x, weight, weight_scale, allow_partial_n_tile=allow_partial_n_tile
    )


class _Fp8BlockScaleWeightReadLinear(nn.Module):
    """Bias-free FP8 block-scale GEMM for checkpoint attention and converted MLPs.

    Attention loads checkpoint E4M3 codes and 128x128 scales directly. The optional
    MLP path quantizes BF16 weights once at load. On SM100/SM103 the UE8M0
    scales are packed for DeepGEMM, which quantizes BF16 activations inside the
    GEMM. On Rubin (SM107) the scales are packed into the CuTe DSL K32 R128c4
    layout and the activation is quantized by the specialized CuTe quantizer
    before the native MXFP8 GEMM, which runs with K3's fine-grained low-M tuning
    buckets. That dispatch is ``layers.fp8_block_scale_gemm``.

    The family's K3 kernels sit in front of that dispatch: the K3 low-M GEMM
    (``kernels/k3r_gemm/k3_fp8w_lowm_gemm.py``) owns the low-M rows (per-shape, per-architecture
    limit, see ``k3_fp8w_lowm_gemm.max_m_for``), so the generic low-M GEMM
    (``trtllm::fp8_block_scale_low_m_gemm``) is not prepared. On SM107 the KDA MTP verify's
    o_proj may take the core's MXFP8 output (:meth:`forward_prequantized`).
    """

    def __init__(
        self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor, out_features: int
    ) -> None:
        super().__init__()
        self.in_features = weight_fp8.shape[1]
        self.out_features = out_features
        self._weights_transformed = True
        self._fused_projection = None
        # Buffers (not parameters): these are the module's weights post-load;
        # there is nothing further to load into them and they must not be
        # touched by any later autocast/dtype move.
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        # Copied from the source ``Linear`` (the MLA fused q_a/kv_a projection sets it): the
        # low-M GEMM may serve this weight although N is not a multiple of 128.
        self.fp8w_lowm_partial_n_tile = False
        # SM107: the weight scale in the DeepGEMM-packed UE8M0 layout of the K3 low-M GEMM
        # (``weight_scale`` holds the CuTe MXFP8 layout there).
        self.lowm_weight_scale: Optional[torch.Tensor] = None

    @property
    def has_fp8_qdq(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def has_w4a8_nvfp4_fp8(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def supports_prequantized_input(self) -> bool:
        """Whether a producer may hand this GEMM an FP8 activation in CuTe layout."""
        return _rubin_mxfp8_is_available()

    def takes_prequantized_input(self, num_rows: int) -> bool:
        """Whether a producer that can emit this GEMM's scale layout should hand ``num_rows``
        rows to :meth:`forward_prequantized` rather than BF16 rows to :meth:`forward`.

        False where :meth:`forward` serves BF16 rows with the low-M GEMM (activation quantized
        inside the kernel) faster than the CuTe MXFP8 GEMM runs on the prequantized operand.
        """
        if not self.supports_prequantized_input:
            return False
        if self.lowm_weight_scale is None:
            return True
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        return num_rows > k3_fp8w_lowm_gemm.bf16_over_prequantized_max_m(
            self.out_features, self.in_features
        )

    @staticmethod
    def quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 ``[out, in]`` weight -> (FP8 weight, backend-ready scale).

        Both dims must be multiples of 128. Because the 128x128 block scale is
        computed per block, concatenating several such weights along ``out``
        and quantizing the result is per-block identical to quantizing each
        separately (no block crosses a 128-aligned boundary), so a fused
        weight's row slices equal the individually quantized weights.
        """
        # Lazy imports: only pulled in on the FP8 path.
        from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8

        weight_fp8, weight_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        # Both GEMMs consume UE8M0 scales rather than the FP32 grid.
        return prepare_fp8_block_weight(weight_fp8, weight_scale)

    @staticmethod
    def _prepare_scale_layout(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
        from tensorrt_llm.quantization.utils.fp8_utils import (
            transform_k128_scales_to_cutedsl_mxfp8_layout,
            transform_sf_into_required_layout,
        )

        if _rubin_mxfp8_is_available():
            # cute_dsl_mxfp8_gemm_rubin reads K32 R128c4 UE8M0 scales.
            return transform_k128_scales_to_cutedsl_mxfp8_layout(weight_scale, mn=n, k=k)
        # fp8_swap_ab_gemm with disable_ue8m0_cast=True consumes packed,
        # TMA-aligned UE8M0 scales.
        return transform_sf_into_required_layout(
            weight_scale,
            mn=n,
            k=k,
            recipe=(1, 128, 128),
            is_sfa=False,
        )

    @classmethod
    def from_linear(cls, linear: nn.Module) -> "_Fp8BlockScaleWeightReadLinear":
        """The FP8 weight read of a loaded projection, sharing its codes and scales:

        * ``core.Fp8BlockScaleLinear`` (checkpoint E4M3 codes and FP32 128 x 128 scales, or their
          ``prepare``d form);
        * a bias-free BF16 projection (``nn.Linear`` or ``core.Bf16Linear``): quantized here.
        """
        assert linear.bias is None, "FP8 weight read expects a bias-free Linear"
        if isinstance(linear, Fp8BlockScaleLinear):
            result = cls(linear.weight.detach(), linear.weight_scale.detach(), linear.out_features)
            result.quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
            result.tp_size, result.tp_rank = linear.tp_size, linear.tp_rank
            result._weights_transformed = getattr(linear, "prepared", False)
            result.fp8w_lowm_partial_n_tile = getattr(linear, "fp8w_lowm_partial_n_tile", False)
            return result
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features)

    def transform_weights(self) -> None:
        if not self._weights_transformed:
            if self._fused_projection is not None:
                owner_ref, start = self._fused_projection
                owner = owner_ref()
                if owner is None:
                    raise RuntimeError("KDA fused FP8 projection was released before its views")
                if self.weight_scale.dtype != torch.float32 or self.weight_scale.ndim != 2:
                    raise ValueError("KDA fused FP8 views require a raw FP32 scale grid")
                # Codes and raw scale grids alias the fused projection. Only its
                # owner may resmooth them in place; each view packs the updated
                # scales without modifying the shared codes again.
                owner.transform_weights()
                self.weight = owner.weight[start : start + self.out_features]
                self.weight_scale = self._prepare_scale_layout(
                    self.weight_scale, self.out_features, self.in_features
                )
            else:
                self.weight, self.weight_scale = prepare_fp8_block_weight(
                    self.weight, self.weight_scale
                )
            self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.prepare_lowm_gemm()

    def prepare_lowm_gemm(self) -> bool:
        """Build ``lowm_weight_scale`` on SM107 for the K3 low-M GEMM.

        Derived from the transformed MXFP8 scale, so it follows weight reloads; rebuilt on
        every call (post-load), before CUDA graph capture. Costs N * K / 128 bytes (0.8% of
        the FP8 weight). SM100/SM103 run these GEMMs on ``weight_scale`` itself.
        """
        self.lowm_weight_scale = None
        if not _rubin_mxfp8_is_available():
            return False
        if self._fused_projection is not None:
            # A row view of the KDA fused q/k/v/g weight; the fused module serves forward.
            return False
        self.transform_weights()
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        n, k = self.weight.shape
        if n % k3_fp8w_lowm_gemm.N_ALIGN or k % 128:
            return False
        self.lowm_weight_scale = k3_fp8w_lowm_gemm.pack_block_scale_from_mxfp8_layout(
            self.weight_scale, n, k
        )
        return True

    def _lowm_gemm_applies(self, x_flat: torch.Tensor, weight_scale: torch.Tensor) -> bool:
        """Whether the ``k3_fp8w_lowm_gemm`` entry serves ``x_flat`` with ``weight_scale``."""
        return x_flat.dtype == torch.bfloat16 and _k3_lowm_gemm_can_run(
            x_flat,
            self.weight,
            weight_scale,
            allow_partial_n_tile=self.fp8w_lowm_partial_n_tile,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        out_shape = (*x.shape[:-1], self.out_features)
        x_flat = x.reshape(-1, x.shape[-1])

        # The K3 low-M GEMM up to its per-shape M gate, on SM107 with ``lowm_weight_scale``, on
        # SM100 / SM103 with ``weight_scale``; else the MXFP8 (SM107) / swap-AB (SM100 / SM103)
        # GEMM of ``layers.fp8_block_scale_gemm``.
        lowm_scale = self.lowm_weight_scale if _rubin_mxfp8_is_available() else self.weight_scale
        if lowm_scale is not None and self._lowm_gemm_applies(x_flat, lowm_scale):
            out = k3_fp8w_lowm_gemm_op(x_flat, self.weight, lowm_scale)
        else:
            out = fp8_block_scale_gemm(x_flat, self.weight, self.weight_scale)
        return out.reshape(out_shape)

    def forward_prequantized(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Project an FP8 activation a fused producer already quantized.

        Args:
            activation: ``[..., in_features]`` ``float8_e4m3fn`` activation.
            activation_scale: ``uint8`` UE8M0 scales in the CuTe K32 R128c4
                layout that ``fp8_quantize_1x128_cutedsl_ue8m0`` produces.
        """
        if activation.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized Kimi K3 activation must be float8_e4m3fn")
        if not self.supports_prequantized_input:
            raise RuntimeError("prequantized Kimi K3 activation requires the Rubin CuTe GEMM")
        if activation_scale.dtype is not torch.uint8:
            raise TypeError("prequantized Kimi K3 activation scale must be uint8 R128c4")
        out = k3_mxfp8_gemm_op(
            activation.reshape(-1, activation.shape[-1]),
            self.weight,
            activation_scale,
            self.weight_scale,
        )
        return out.reshape(activation.shape[:-1] + (self.out_features,))


def _swap_linear_to_fp8_weight_read(
    parent: nn.Module,
    attr: str,
    linear_types: Tuple[type, ...] = (nn.Linear,),
) -> int:
    """Replace ``parent.<attr>`` with an FP8 weight-read module if it is a
    plain linear of one of ``linear_types``; return the number of modules
    converted (0 or 1), so callers can accumulate a conversion count.

    Transfer checkpoint FP8 storage or release the original BF16 storage
    after conversion. The loader's transient parameter map must not retain
    an unused BF16 copy until loading completes.
    """
    child = getattr(parent, attr, None)
    if not isinstance(child, linear_types):
        return 0
    converted = _Fp8BlockScaleWeightReadLinear.from_linear(child)
    setattr(parent, attr, converted)
    child.weight.data = child.weight.data.new_empty(0)
    return 1


def _has_weights(module: nn.Module) -> bool:
    """False once ``modeling_utils.remove_weights()`` has stripped a module.

    Post-load finalization walks every decoder layer, so it must skip layers
    whose parameters were dropped — the layer-wise benchmarks keep only the
    profiled slice resident.
    """
    return not getattr(module, "_weights_removed", False)


def _convert_moe_mlps_to_fp8_weight_read(
    model: nn.Module, include_fused_gate_up: bool = True
) -> int:
    """Swap the replicated MoE-layer MLP projections to an FP8 weight read.

    Targets the shared-expert MLP (gate/up/down) and the latent up/down
    projection on every MoE layer — the bias-free BF16 projections that
    attention data-parallelism re-reads in full each decode step. Attention
    follows checkpoint quantization separately. Routed experts and the dense
    layer-0 MLP are not converted. Returns the number of projections converted.
    """
    count = 0

    for layer in model.layers:
        if not _has_weights(layer):
            continue
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        shared = getattr(moe, "shared_experts", None)
        if shared is not None:
            # GatedMlp fuses gate and up into gate_up_proj; keep the split
            # names too so either MLP layout converts. The fused gate_up read
            # only pays off when attention DP re-reads it per rank per step;
            # under TP the bf16 GEMM overlaps on the aux stream and the FP8
            # quantize+GEMM would serialize onto the critical path.
            shared_attrs = (
                ("gate_proj", "up_proj", "gate_up_proj", "down_proj")
                if include_fused_gate_up
                else ("gate_proj", "up_proj", "down_proj")
            )
            for attr in shared_attrs:
                child = getattr(shared, attr, None)
                if isinstance(child, Bf16Linear) and child.tp_size != 1:
                    continue
                count += _swap_linear_to_fp8_weight_read(
                    shared, attr, linear_types=(nn.Linear, Bf16Linear)
                )
        for attr in ("routed_expert_down_proj", "routed_expert_up_proj"):
            count += _swap_linear_to_fp8_weight_read(moe, attr)

    # Return the freed BF16 blocks to the driver so the raw (non-caching-
    # allocator) allocations made during executor creation succeed on the
    # tight DEP16 memory headroom.
    if count:
        gc.collect()
        torch.cuda.empty_cache()
    return count


# For up to _K3_ROUTER_FIRST_MAX_M tokens, the router GEMM is issued (captured) on the main stream
# before latent_down is issued on the latent stream. Both depend only on the MoE input (the latent
# fork event is recorded before the router), so the graph has the same edges as the reverse
# order; only the node creation order differs, which decides which of the two the
# graph launches first once the MoE input's producer releases its dependents (PDL). At decode
# batch sizes latent_down's low-M GEMM is 112 CTAs x ~225 KB of shared memory, and with up to 16
# tokens the router GEMM (dsv3 router MMA kernel) is 112 CTAs whose registers fill most of an SM,
# so no SM holds one of each: launched second, the router gets only the SMs latent_down leaves
# and runs in several waves, and the routing (which gates the routed FC1 together with
# latent_down) ends late. Launched first, the router takes its SMs and latent_down starts on the
# rest. nsys node tracing turns latent_down's cross-stream programmatic edge into a full edge (it
# then waits for the MoE input's producer to complete), so such traces show the router first in
# either order.
_K3_ROUTER_FIRST_MAX_M = 64

# The MoE backends the target builds: MEGAMOE_CUTEDSL (the recipe's), and CUTEDSL, whose CuTe DSL
# kernels take only their own rank's tokens (attention DP over one rank).
_K3_DEP16_MOE_BACKENDS = ("MEGAMOE_CUTEDSL", "CUTEDSL")


class K3MoEGate(nn.Module):
    """Kimi K3 gate: the router weight, the score-correction bias and the routing kernel.

    Routing is ungrouped sigmoid top-k with the bias added for the selection and the selected
    scores renormalized and scaled (``noaux_tc``, :meth:`route`). The gate GEMM runs on FP32
    operands (the upcast BF16 input and the FP32 weight) with FP32 output.
    """

    def __init__(
        self,
        config: Any,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size

        if self.moe_router_activation_func != "sigmoid" or not self.moe_renormalize:
            raise ValueError(
                "Kimi K3 routing takes sigmoid scores with top-k weight renormalization"
            )
        if self.num_expert_group != 1 or self.topk_group != 1:
            raise ValueError("Kimi K3 routing is ungrouped (num_expert_group = topk_group = 1)")
        if self.num_experts > 1024 or self.top_k > 32:
            raise ValueError(
                f"the fused routing kernel serves up to 1024 experts and top-32, got "
                f"{self.num_experts} and top-{self.top_k}"
            )

        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim), dtype=torch.float32, device=device)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32, device=device)
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute fp32 routing logits shaped ``[num_tokens, num_experts]``."""
        hidden_2d = hidden_states.reshape(-1, self.gating_dim)
        return torch.nn.functional.linear(
            hidden_2d.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )

    def route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(ids int32, weights fp32)``, each ``[num_tokens, top_k]``: ``noaux_tc_op``."""
        values, indices = noaux_tc_op(
            logits,
            self.e_score_correction_bias,
            self.num_expert_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
        )
        return indices.to(torch.int32), values.to(torch.float32)


# ---------------------------------------------------------------------------
# Routed-expert checkpoint keys.
# ---------------------------------------------------------------------------

# Routed-expert key spellings that ModelOpt emits for Kimi K3. The NVFP4
# checkpoint (``nvidia/Kimi-K3-NVFP4``) lists every prefix x module-name
# combination in ``quantized_layers``, so a lookup over this product finds it
# without needing the MiniMax-M3-style prefix normalization in ``ModelConfig``.
_K3_ROUTED_EXPERT_KEY_PREFIXES = ("language_model.model.", "model.", "")
_K3_ROUTED_EXPERT_KEY_SUFFIXES = ("block_sparse_moe.experts", "mlp.experts")

# The subset of the above that can be a real module path. ``exclude_modules``
# matches with wildcards and walks ancestor prefixes, so an empty prefix would
# widen what matches instead of just missing, as it does in the dict lookup.
_K3_ROUTED_EXPERT_MODULE_PREFIXES = ("language_model.model.", "model.")

# Routed-expert quantization used when the checkpoint declares nothing per
# layer. The original ``moonshotai/Kimi-K3`` ships a compressed-tensors
# ``mxfp4-pack-quantized`` config with no ModelOpt per-layer entries, and that
# checkpoint is what this default has always served.
_K3_DEFAULT_ROUTED_QUANT_ALGO = QuantAlgo.W4A8_MXFP4_MXFP8


# ---------------------------------------------------------------------------
# Routed-expert checkpoint layout.
#
# K3 streams routed experts one at a time (see ``load_weights``): the tensor names, the per-expert
# loader and the per-layer finalize it needs are decided by the checkpoint's routed-expert
# quantization. The expected-key plan, the loader and the file grouping read them off one spec.
# ---------------------------------------------------------------------------


def _load_nvfp4_expert(
    backend, base, expert_idx, local_slot_id, get_tensor, get_optional_tensor
) -> None:
    for w in ("w1", "w3"):
        if get_optional_tensor(f"{base}.{expert_idx}.{w}.pre_quant_scale") is not None:
            raise NotImplementedError(
                "Kimi K3 routed experts take plain NVFP4 checkpoints; this one carries a "
                "pre-quant activation scale (NVFP4_AWQ / NVFP4_ARC)"
            )
    tensors = {
        f"{w}_{kind}": get_tensor(f"{base}.{expert_idx}.{w}.{kind}")
        for w in ("w1", "w2", "w3")
        for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }
    backend.load_expert(global_expert_id=expert_idx, local_slot_id=local_slot_id, **tensors)


class _K3ExpertCkptSpec(NamedTuple):
    """How one routed-expert quantization is spelled and loaded."""

    # Per-``w{1,2,3}`` checkpoint tensor suffixes this layout stores. Every
    # combination is required: a missing one fails the key plan.
    kinds: Tuple[str, ...]
    loader: Callable[..., None]
    # Set of filled slots the loader maintains, checked after the load.
    loaded_slots_attr: str
    # The loader stages per-expert state that the layer's finalize consumes.
    needs_layer_finalize: bool
    # Whole per-expert key suffixes the loader reads WHEN PRESENT (to reject them).
    optional_suffixes: Tuple[str, ...] = ()


_K3_EXPERT_CKPT_SPECS = {
    QuantAlgo.NVFP4: _K3ExpertCkptSpec(
        kinds=("weight", "weight_scale", "weight_scale_2", "input_scale"),
        loader=_load_nvfp4_expert,
        loaded_slots_attr="_streamed_expert_slots",
        needs_layer_finalize=True,
        optional_suffixes=("w1.pre_quant_scale", "w3.pre_quant_scale"),
    ),
}


def _k3_expert_ckpt_spec(quant_config: QuantConfig) -> _K3ExpertCkptSpec:
    spec = _K3_EXPERT_CKPT_SPECS.get(quant_config.quant_algo)
    if spec is None:
        raise NotImplementedError(
            f"Kimi K3 routed experts are quantized as {quant_config.quant_algo}, for which "
            "no per-expert checkpoint layout is known. Supported: "
            f"{sorted(a.name for a in _K3_EXPERT_CKPT_SPECS)}."
        )
    group_size = getattr(quant_config, "group_size", None)
    if group_size not in (None, 16):
        raise NotImplementedError(
            f"Kimi K3 NVFP4 routed experts take 16-element scale blocks, got group_size={group_size}"
        )
    return spec


class _K3SharedExpert(GatedMlp):
    """The MoE layer's shared expert: :class:`GatedMlp` called through its projection modules,
    so a projection ``_convert_moe_mlps_to_fp8_weight_read`` swapped for an FP8 weight read runs
    its own kernel (a ``Bf16Linear`` without bias is ``F.linear`` either way)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gate_up_proj(x)
        a = gated_activation(h, self.activation, self.beta, self.linear_beta)
        return self.down_proj(a)


class K3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block under attention DP: the gate routes, the routed experts run the
    target's MegaMoE module (CUTEDSL: the CuTe DSL kernel chain of ``ops/moe/k3r_experts``), the
    replicated shared expert runs beside them."""

    # Under attention DP the routed experts return this rank's tokens complete and the shared
    # expert is replicated: neither output is all-reduced, so no all-reduce is deferred either.
    _reduce_routed_output = False
    _defer_shared_allreduce = False

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        """Build the routed experts and the shared expert for one MoE layer.

        ``cfg`` is the raw ``PretrainedConfig`` rather than anything derived:
        the SiTU soft-caps and the routed-expert geometry are Kimi K3 fields
        that ``ModelConfig`` does not carry.

        ``aux_stream_dict`` is shared across every layer of the model, so the
        streams reached through it are borrowed and must not be synchronized
        or reassigned here.
        """
        super().__init__()
        if not model_config.mapping.enable_attention_dp:
            raise NotImplementedError("the Kimi K3 dep16 MoE block runs under attention DP")
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.moe_hidden_size = cfg.routed_expert_hidden_size
        # ValueError (not assert): these guard unsupported checkpoint
        # configurations and must stay active under ``python -O``.
        if self.moe_hidden_size is None:
            raise ValueError("Kimi K3 runtime expects the latent MoE (routed_expert_hidden_size)")
        if not getattr(cfg, "latent_moe_use_norm", False):
            raise ValueError("Kimi K3 runtime expects latent_moe_use_norm=True")

        situ_beta, situ_linear_beta = _resolve_kimi_situ_betas(cfg)
        dtype = torch.bfloat16

        # Routing scores stay fp32: the gate GEMM of the upcast operands. The bf16-input min-latency
        # GEMM's different reduction order flips borderline top-16 picks (GSM8K 96.7 -> 96.1/96.4,
        # three runs), and its bs1-latency win is irrelevant at DEP batch sizes.
        self.gate = K3MoEGate(cfg)

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        routed_quant_config = self._resolve_routed_quant_config(model_config, layer_idx)
        # Resolved here so ``load_weights`` reads the checkpoint layout off the
        # module instead of re-deriving it at each of its three call sites.
        self.expert_ckpt_spec = _k3_expert_ckpt_spec(routed_quant_config)
        self._moe_backend = routed_moe_model_config.moe_backend
        routed_mapping = routed_moe_model_config.mapping
        if self._moe_backend == "CUTEDSL":
            if routed_moe_model_config.moe_disable_finalize_fusion:
                raise NotImplementedError(
                    "Kimi K3 routed experts run the fused-finalize kernels "
                    "(moe_disable_finalize_fusion / FORCE_DETERMINISTIC is not supported)"
                )
            from tensorrt_llm._torch.locality_domain.policy import LocalityDomainExecutionPlanner

            locality_plan = LocalityDomainExecutionPlanner(
                routed_moe_model_config.locality_domain_policy
            ).plan_moe(
                routed_quant_config,
                moe_backend=self._moe_backend,
                use_fused_finalize=True,
                dtype_activation=dtype,
                activation=ActivationType.SiTu.name,
            )
            if locality_plan.enabled:
                raise NotImplementedError(
                    "Kimi K3 routed experts do not run the locality-domain MoE kernels; "
                    "disable enable_locality_domains"
                )
            self.routed_experts = K3RoutedExperts(
                kernels=self._moe_backend,
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=self.moe_hidden_size,
                intermediate_size=cfg.moe_intermediate_size,
                mapping=routed_mapping,
                layer_idx=layer_idx,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                aux_stream=aux_stream_dict.get(AuxStreamType.MoeOutputMemset),
                max_chunk_tokens=routed_moe_model_config.moe_max_num_tokens,
            )
        else:
            per_rank_cap = int(
                routed_moe_model_config.moe_max_num_tokens
                or routed_moe_model_config.max_num_tokens
                or 4096
            )
            if routed_moe_model_config.max_num_tokens:
                per_rank_cap = min(per_rank_cap, int(routed_moe_model_config.max_num_tokens))
            self.routed_experts = K3MegaMoE(
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=self.moe_hidden_size,
                intermediate_size=cfg.moe_intermediate_size,
                mapping=routed_mapping,
                layer_idx=layer_idx,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                max_num_tokens=per_rank_cap,
                dtype=dtype,
                skip_create_weights_in_init=routed_moe_model_config.skip_create_weights_in_init,
            )
        local_expert_ids = list(self.routed_experts.backend.initial_local_expert_ids)
        if local_expert_ids != list(
            range(local_expert_ids[0], local_expert_ids[0] + len(local_expert_ids))
        ):
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming currently requires a "
                "contiguous static expert partition."
            )
        self.local_expert_ids = tuple(local_expert_ids)
        self.experts_per_rank = len(local_expert_ids)
        self.expert_lo = local_expert_ids[0]
        self.expert_hi = self.expert_lo + self.experts_per_rank

        shared_intermediate = cfg.moe_intermediate_size * cfg.num_shared_experts
        # Under attention DP each rank owns different tokens, so the shared
        # expert is replicated (TP size 1) and does not reduce across ranks.
        if model_config.use_cute_dsl_bf16_gemm:
            raise NotImplementedError(
                "Kimi K3's shared expert runs the cuBLAS BF16 GEMMs "
                "(use_cute_dsl_bf16_gemm is not supported)"
            )
        self.shared_experts = _K3SharedExpert(
            cfg.hidden_size,
            shared_intermediate,
            activation="situ",
            beta=situ_beta,
            linear_beta=situ_linear_beta,
            dtype=dtype,
            mapping=subgroup_mapping(model_config.mapping, 1),
        )
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        # CuteDSL already quantizes the MoE input on the MoeOutputMemset stream;
        # the latent down-projection runs there too, beside the routing chain.
        self.latent_down_stream = (
            aux_stream_dict.get(AuxStreamType.MoeOutputMemset)
            if self._moe_backend == "CUTEDSL"
            else None
        )
        has_latent_stream = self.latent_down_stream is not None
        self.moe_latent_fork_event = torch.cuda.Event() if has_latent_stream else None
        self.moe_latent_ready_event = torch.cuda.Event() if has_latent_stream else None
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        # FlashInfer RMSNorm on the routed latent.
        self.routed_expert_norm = RMSNormWeight(self.moe_hidden_size, cfg.rms_norm_eps, dtype=dtype)

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits."""
        return self.gate.compute_logits(hidden_states)

    def prepare_expert_load(self) -> None:
        """Before this layer's first routed expert streams in."""
        self.routed_experts.backend.prepare_streaming_load()

    def finalize_expert_load(self) -> None:
        """After this layer's last routed expert streamed in."""
        self.routed_experts.backend.finalize()

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if not isinstance(projection, nn.Linear):
            return projection(hidden_states)
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            hidden_states, projection.weight.t(), None, None
        )

    @staticmethod
    def _select_moe_tp_ep(mapping: Mapping) -> Tuple[int, int]:
        """Resolve the routed-expert ``(moe_tp, moe_ep)`` split.

        Precedence:

        1. Explicit ``moe_tensor_parallel_size`` / ``moe_expert_parallel_size``
           from the user config. Detected via
           ``mapping.moe_tp_ep_user_specified`` so the auto-resolved mapping
           default (``moe_tp=tp_size, moe_ep=1``) is NOT mistaken for a TP
           request.
        2. Default: EP-only (``moe_tp=1, moe_ep=tp_size``), the historical
           K3 layout.
        """
        tp_size = mapping.tp_size
        if getattr(mapping, "moe_tp_ep_user_specified", False):
            return mapping.moe_tp_size, mapping.moe_ep_size
        return 1, tp_size

    @staticmethod
    def _resolve_routed_quant_config(model_config: ModelConfig, layer_idx: int) -> QuantConfig:
        """Routed-expert quantization for ``layer_idx``, taken from the checkpoint.

        ``nvidia/Kimi-K3-NVFP4`` declares the routed experts per layer as
        ``NVFP4`` with ``group_size=16``; the original ``moonshotai/Kimi-K3``
        declares nothing per layer and keeps the historical
        ``W4A8_MXFP4_MXFP8`` default. Reading the checkpoint instead of
        hardcoding is what lets one code path serve both.

        An exclusion outranks the per-layer entry and the default below:
        ``create_weights`` treats an override as authoritative over anything
        ``__post_init__`` wrote, so this return value stands in for both
        quantization passes and exclusion is the one that runs second. It is
        matched as a pattern, so it is asked only about real module names.
        """
        quant_config = model_config.quant_config
        if quant_config is not None and any(
            quant_config.is_module_excluded_from_quantization(
                f"{prefix}layers.{layer_idx}.{suffix}"
            )
            for prefix in _K3_ROUTED_EXPERT_MODULE_PREFIXES
            for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES
        ):
            logger.debug(
                "Kimi K3 layer %d routed experts: excluded from quantization, "
                "keeping them unquantized",
                layer_idx,
            )
            return QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)

        per_layer = getattr(model_config, "quant_config_dict", None)
        if per_layer:
            for prefix in _K3_ROUTED_EXPERT_KEY_PREFIXES:
                for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES:
                    cfg = per_layer.get(f"{prefix}layers.{layer_idx}.{suffix}")
                    if cfg is not None and cfg.quant_algo is not None:
                        # Logged once per layer: the routed-expert format decides
                        # which MoE backends can serve this checkpoint at all.
                        logger.debug(
                            "Kimi K3 layer %d routed experts: %s (group_size=%s) "
                            "from the checkpoint",
                            layer_idx,
                            cfg.quant_algo,
                            cfg.group_size,
                        )
                        return cfg
        logger.debug(
            "Kimi K3 layer %d routed experts: no per-layer quant config in the "
            "checkpoint, defaulting to %s",
            layer_idx,
            _K3_DEFAULT_ROUTED_QUANT_ALGO,
        )
        return QuantConfig(quant_algo=_K3_DEFAULT_ROUTED_QUANT_ALGO)

    @staticmethod
    def _routed_moe_model_config(model_config: ModelConfig) -> ModelConfig:
        """Build a private routed-expert mapping without mutating the shared
        config. Default split is EP-only; see ``_select_moe_tp_ep``."""
        if model_config.moe_backend not in _K3_DEP16_MOE_BACKENDS:
            raise ValueError(
                f"Kimi K3 dep16 routed experts run on {_K3_DEP16_MOE_BACKENDS}; "
                f"got {model_config.moe_backend!r}."
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping

        moe_tp, moe_ep = K3MoERuntime._select_moe_tp_ep(mapping)
        if moe_tp < 1 or moe_ep < 1 or moe_tp * moe_ep != mapping.tp_size:
            raise ValueError(
                f"Kimi K3 routed MoE split moe_tp={moe_tp} x moe_ep={moe_ep} "
                f"must multiply to tp_size={mapping.tp_size}."
            )
        if moe_tp > 1 and mapping.enable_attention_dp:
            raise NotImplementedError(
                "Kimi K3 MoE tensor parallelism requires "
                "enable_attention_dp=false (the attention-DP dispatch/combine "
                "path is validated for EP-only splits)."
            )
        logger.info_once(
            f"Kimi K3 routed MoE parallelism: moe_tp={moe_tp}, "
            f"moe_ep={moe_ep} (tp_size={mapping.tp_size})",
            key="kimi_k3_moe_tp_ep_split",
        )

        mapping_dict = mapping.to_dict()
        mapping_dict["moe_cluster_size"] = 1
        mapping_dict["moe_tp_size"] = moe_tp
        mapping_dict["moe_ep_size"] = moe_ep
        routed_mapping = Mapping.from_dict(mapping_dict)

        routed_model_config = copy.copy(model_config)
        routed_model_config._frozen = False
        routed_model_config.extra_attrs = copy.copy(model_config.extra_attrs)
        routed_model_config.mapping = routed_mapping
        routed_model_config.moe_backend = model_config.moe_backend
        # MegaMoE uses this value as global DP SymmBuffer capacity, then
        # divides it by EP size for the per-rank allocation. Other backends
        # keep the user-configured value as their MoE chunking bound.
        # Preserve an explicitly larger capacity.
        if routed_model_config.moe_backend == "MEGAMOE_CUTEDSL":
            default_moe_max_num_tokens = routed_model_config.max_num_tokens * routed_mapping.dp_size
            configured_moe_max_num_tokens = int(routed_model_config.moe_max_num_tokens or 0)
            if configured_moe_max_num_tokens < default_moe_max_num_tokens:
                logger.info_once(
                    "Kimi K3 MegaMoE raises moe_max_num_tokens from "
                    f"{configured_moe_max_num_tokens} to {default_moe_max_num_tokens} "
                    "because the global DP SymmBuffer requires capacity for "
                    "max_num_tokens * dp_size.",
                    key=(
                        "kimi_k3_megamoe_capacity_override_"
                        f"{configured_moe_max_num_tokens}_{default_moe_max_num_tokens}"
                    ),
                )
            routed_model_config.moe_max_num_tokens = max(
                configured_moe_max_num_tokens,
                default_moe_max_num_tokens,
            )
        routed_model_config._frozen = True
        return routed_model_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        *,
        residual: Optional[torch.Tensor] = None,
        defer_residual: bool = False,
    ) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16.

        With ``residual``, returns ``residual + moe_output``; with ``defer_residual`` as well, a
        ``_DeferredMoEResidual`` instead (the next layer's attention-residual mix adds it).
        """
        identity = hidden_states
        early_routed_in = None
        router_issued = False
        if self.latent_down_stream is not None and do_multi_stream():
            # Forked ahead of the router GEMM: the latent chain needs only hidden_states.
            self.moe_latent_fork_event.record()
            if hidden_states.shape[0] <= _K3_ROUTER_FIRST_MAX_M:
                # Router before latent_down in creation order (see _K3_ROUTER_FIRST_MAX_M).
                router_logits = self._router_logits(hidden_states)
                router_issued = True
            with torch.cuda.stream(self.latent_down_stream):
                self.moe_latent_fork_event.wait()
                early_routed_in = self._routed_projection(
                    hidden_states, self.routed_expert_down_proj
                )
                self.moe_latent_ready_event.record()
            # No record_stream (under graph capture it holds every layer's
            # blocks until capture ends): both tensors are read on main only
            # after the join, and later work on this stream forks from main.
        if not router_issued:
            router_logits = self._router_logits(hidden_states)

        def _routed_output():
            # Latent down/up projections via the min-latency fused GEMM op:
            # at <=16 tokens (decode graphs) it runs a single pipelined
            # bf16 kernel per projection instead of cuBLAS's split-K GEMV +
            # splitKreduce pair (~17+3.6us -> ~8us for 7168->3584 at M=1);
            # for larger token counts the op falls back to cuBLAS internally.
            # When the FP8 weight-read conversion has
            # replaced the projection module, call it directly: its weight is
            # an e4m3 buffer the bf16 dsv3 op must not read, and its forward
            # is already a single fused GEMM (fp8_swap_ab_gemm).
            if early_routed_in is not None:
                routed_in = early_routed_in
                # The routed experts join the event before their first read of x.
                input_ready_event = self.moe_latent_ready_event
            else:
                routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
                input_ready_event = None
            # noaux_tc_op on this rank's tokens (the CuTe DSL experts then run moe_sort).
            gate = self.gate
            y = self.routed_experts(
                routed_in,
                router_logits,
                route=lambda lg, offset, num: SortedRouting(*gate.route(lg), None),
                input_ready_event=input_ready_event,
                all_rank_num_tokens=all_rank_num_tokens,
            )
            # The routed experts return this rank's complete routed result.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        def _shared_output():
            """The shared expert's output (replicated: no all-reduce)."""
            return self.shared_experts(identity)

        # Shared experts depend only on the block input, so overlap their GEMMs
        # with the routed dispatch/expert/combine chain. Multi-stream engages
        # only under CUDA graphs; otherwise both branches run in order on the
        # default stream.
        routed_out, shared_out = fork_join(
            _routed_output,
            _shared_output,
            self.moe_main_event,
            self.moe_shared_event,
            self.shared_expert_stream,
        )
        if residual is not None:
            if defer_residual:
                return _DeferredMoEResidual(residual, routed_out, shared_out)
            return k3_moe_residual_add_op(routed_out, shared_out, residual)
        return routed_out + shared_out


__all__ = ["K3Dep16KvBalancedAdmission", "plan_kv_balanced_admission"]


def plan_kv_balanced_admission(
    candidate_tokens: Sequence[int],
    candidate_forced: Sequence[bool],
    rank_tokens: Sequence[int],
    rank_free_slots: Sequence[int],
    max_new: int,
) -> List[Tuple[int, int]]:
    """Return ``(candidate_index, rank)`` pairs, in admission order.

    Args:
        candidate_tokens: KV tokens each waiting candidate brings (queue order).
        candidate_forced: Whether the candidate has waited too long and must be admitted before any other choice.
        rank_tokens: Current KV tokens per rank.
        rank_free_slots: Free request slots per rank.
        max_new: Upper bound on the number of admissions.
    """
    loads = list(rank_tokens)
    free = [max(0, f) for f in rank_free_slots]
    num_ranks = len(loads)
    remaining = list(range(len(candidate_tokens)))
    budget = min(max_new, sum(free), len(remaining))
    plan: List[Tuple[int, int]] = []
    if budget <= 0:
        return plan

    # Mean load once every slot this round fills holds an average-sized candidate.
    avg_candidate = sum(candidate_tokens) / len(candidate_tokens)
    target = (sum(loads) + avg_candidate * budget) / num_ranks

    for _ in range(budget):
        open_ranks = [r for r in range(num_ranks) if free[r] > 0]
        if not open_ranks or not remaining:
            break
        forced = [i for i in remaining if candidate_forced[i]]
        if forced:
            idx = forced[0]
            rank = min(open_ranks, key=lambda r: (loads[r], r))
        else:
            # Fill the rank furthest below target first; give it the candidate that lands it closest to the target
            # (ties: earlier in the queue).
            rank = min(open_ranks, key=lambda r: (loads[r] - target, r))
            want = target - loads[rank]
            idx = min(remaining, key=lambda i: (abs(candidate_tokens[i] - want), i))
        plan.append((idx, rank))
        remaining.remove(idx)
        loads[rank] += candidate_tokens[idx]
        free[rank] -= 1
    return plan


class K3Dep16KvBalancedAdmission:
    """``KvBalancedAdmissionPlanner`` of the dep16 target: a fixed head window of ``WINDOW`` requests and
    :func:`plan_kv_balanced_admission`. Admission groups are not used: every candidate is placed on its own."""

    WINDOW = 32

    def window(self, max_new_requests: int, free_slots: int) -> int:
        return self.WINDOW

    def plan(
        self,
        *,
        candidate_tokens: list[int],
        candidate_forced: list[bool],
        rank_tokens: list[int],
        rank_num_requests: list[int],
        max_requests_per_rank: int,
        max_new_requests: int,
        candidate_groups: list[int],
    ) -> list[tuple[int, int]]:
        return plan_kv_balanced_admission(
            candidate_tokens,
            candidate_forced,
            rank_tokens,
            [max_requests_per_rank - n for n in rank_num_requests],
            max_new_requests,
        )


# CUDA stream priority of the MoE input stream (latent_down chain, routed-output memset).
# Negative values rank above the main and shared-expert streams, so the latent chain that
# gates the routed GEMM1 gets free SMs first.
_K3_LATENT_STREAM_PRIORITY = 0

# CUDA stream priority of the stream shared by the attention-side aux work and the MoE shared
# expert. Negative values let the shared-expert chain take SMs ahead of the persistent routed
# GEMMs on the main stream, so its all-reduce does not land after them.
_K3_SHARED_STREAM_PRIORITY = 0

# The DFlash/DSpark hidden-state tap captures the pre-norm attn_res mixture (the value the next
# consumer reads), the convention of SGLang and of the RadixArk/Kimi-K3-DSpark drafter
# (``kimi_k3.py _dspark_capture_stream`` -> ``attn_residual.py aggregate_stream``). A drafter
# distilled against the raw running prefix sum (vLLM's ``VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=0``)
# would score lower here with nothing raised (K3 + RadixArk: AR 71.4% -> 66.9% the other way).


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class KimiK3DefaultDecoderLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        dtype = torch.bfloat16
        self.is_moe = (
            cfg.num_experts is not None
            and layer_idx >= cfg.first_k_dense_replace
            and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0
        )

        self.is_kda = _is_kda_layer(cfg, layer_idx)
        is_mla = _is_mla_layer(cfg, layer_idx)
        if self.is_kda == is_mla:
            raise ValueError(f"Kimi K3 layer {layer_idx} must be exactly one of KDA/MLA")

        if self.is_kda:
            projection_names = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
            attention_config = copy.copy(model_config)
            attention_config._frozen = False
            attention_config.quant_config_dict = {
                name: resolve_attention_quant_config(model_config, layer_idx, name)
                for name in projection_names
            }
            attention_config._frozen = model_config._frozen
            self.linear_attn = K3KDALinearAttention(
                cfg,
                layer_idx,
                mapping=model_config.mapping.mapping_for_full_rows(),
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                model_config=attention_config,
            )
        else:
            self.self_attn = K3MLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
            )

        if self.is_moe:
            self.block_sparse_moe = K3MoERuntime(model_config, cfg, layer_idx, aux_stream_dict)
        else:
            # The dense MLP (layer 0): BF16 gate_up GEMM, SiTU-and-mul, down GEMM. Attention DP
            # replicates it (ranks own different tokens): no all-reduce.
            mlp_mapping = model_config.mapping.mapping_for_full_rows()
            self.mlp_tp_size = 1
            self.mlp = GatedMlp(
                cfg.hidden_size,
                cfg.intermediate_size,
                activation="situ",
                beta=getattr(cfg, "activation_situ_beta", None) or 1.0,
                linear_beta=getattr(cfg, "activation_situ_linear_beta", None),
                dtype=dtype,
                mapping=subgroup_mapping(mlp_mapping, self.mlp_tp_size),
            )

        # FlashInfer RMSNorm on the K3 weights: the input norm after the pre-attention mix and the
        # post-attention norm after A_mid (fused into the attention-residual kernels where they
        # serve, ``trtllm::flashinfer_rmsnorm`` otherwise).
        self.input_layernorm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNormWeight(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype
        )

        # Attention residual scheme (always on for K3): the score norms and the [1, H] score
        # projections the attention-residual kernels read (``.weight`` / ``.eps``).
        self.attn_res_block_size = cfg.attn_res_block_size
        assert self.attn_res_block_size is not None, (
            "Kimi K3 runtime expects attn_res_block_size to be set"
        )
        self.self_attention_res_norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.mlp_res_norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.mlp_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)

    @staticmethod
    def _capture_hidden_states(capture: Tuple[Any, int], value: torch.Tensor) -> None:
        """The DSpark tap of ``capture[1]``."""
        capture[0].maybe_capture_hidden_states(capture[1], value, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        capture: Optional[Tuple[Any, int]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        ``block_residual`` is a preallocated snapshot bank in kernel-native
        ``[K_max, M, H]`` layout. Returns the running prefix sum and the
        number of valid bank rows. The prefix sum (input and output) can be a
        ``_DeferredMoEResidual`` (a MoE layer's residual update, up to 64 tokens).

        ``capture`` is ``(spec_metadata, layer_id)`` and taps the DSpark aux
        stream for the layer BEFORE this one: the aggregated stream for layer j
        is by definition what its next consumer sees, so the mixture computed
        below already is it. Reading it here beats recomputing it, and is only
        possible because K3 asserts pp_size == 1 -- layer j+1 is always local.
        PP support would need a recompute at the rank boundary.
        """
        all_rank_num_tokens = getattr(attn_metadata, "all_rank_num_tokens", None)
        valid_block_residual = block_residual[:num_snapshots]
        normed = None
        if isinstance(hidden_states, _DeferredMoEResidual):
            # The previous layer's MoE residual update, folded into this mix when possible.
            if capture is None and num_snapshots > 0:
                fused = _k3_attn_res_pdl_moe_add(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                )
                if fused is not None:
                    hidden_states, normed = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        prefix_sum = hidden_states

        if capture is not None:
            # The mixture tap needs the PRE-norm value (SGLang's
            # aggregate_stream is a "softmax mixture, no output norm"), which
            # the fused attn-res + RMSNorm kernel does not expose. Keep the two
            # steps split on captured layers only and fuse everywhere else.
            mixed = (
                _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
                if num_snapshots > 0
                else hidden_states
            )
            self._capture_hidden_states(capture, mixed)
            hidden_states = rms_norm(mixed, self.input_layernorm.weight, self.input_layernorm.eps)
        elif normed is not None:
            hidden_states = normed
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            hidden_states = rms_norm(
                hidden_states, self.input_layernorm.weight, self.input_layernorm.eps
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None

        if self.is_kda:
            hidden_states = self.linear_attn(hidden_states, attn_metadata)
        else:
            hidden_states = self.self_attn(hidden_states, attn_metadata)

        if prefix_sum is None:
            prefix_sum = hidden_states
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        else:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                hidden_states,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        if self.is_moe:
            # The MoE layer applies the residual update itself (or defers it to the next mix).
            prefix_sum = self.block_sparse_moe(
                hidden_states,
                all_rank_num_tokens,
                residual=prefix_sum,
                defer_residual=_defer_moe_residual(prefix_sum.shape[0]),
            )
            return prefix_sum, num_snapshots
        hidden_states = self.mlp(hidden_states)

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op stand-in for ``forward``, matching ``DecoderLayer.skip_forward``.

        ``modeling_utils.skip_forward()`` only drops a module's weights when it
        finds this attribute, so without it the layer-wise benchmarks would
        allocate all 93 layers instead of the profiled slice.
        """
        return hidden_states, block_residual


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class KimiK3DefaultModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16

        # Attention and MoE phases are sequential, so their branch-overlap
        # roles share one stream; MoE-internal overlap roles remain separate.
        aux_stream_list = [torch.cuda.Stream(priority=_K3_SHARED_STREAM_PRIORITY)]
        aux_stream_list += [torch.cuda.Stream() for _ in range(2)]
        aux_stream_list.append(torch.cuda.Stream(priority=_K3_LATENT_STREAM_PRIORITY))
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }

        # The whole vocabulary on every rank: a row gather (``F.embedding``).
        self.embed_tokens = EmbeddingTable(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                KimiK3DefaultDecoderLayer(model_config, cfg, layer_idx, self.aux_stream_dict)
                for layer_idx in range(cfg.num_hidden_layers)
            ]
        )
        # The final norm, fused into the output attention-residual mix where that kernel serves.
        self.norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.output_attn_res_norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.output_attn_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.num_attn_res_snapshots = (
            cfg.num_hidden_layers + cfg.attn_res_block_size - 1
        ) // cfg.attn_res_block_size

        # Which convention the drafter tap is on is not recoverable from the
        # served output -- a mismatch only lowers acceptance -- so state it once
        # at construction rather than leaving it to be inferred from an AL.
        logger.info_once(
            "Kimi K3 aux hidden capture: mode=attn_res_stream",
            key="kimi_k3_aux_capture_mode",
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = F.embedding(input_ids, self.embed_tokens.weight)
        hidden_states = inputs_embeds

        # The FP4 MLA decode plan of the previous step is stale.
        fp4_mla_decode.begin_step(attn_metadata)

        block_residual = hidden_states.new_empty(
            self.num_attn_res_snapshots,
            hidden_states.shape[0],
            hidden_states.shape[1],
        )
        num_snapshots = 0
        capture_set = (
            getattr(spec_metadata, "_capture_layer_set", None)
            if spec_metadata is not None
            else None
        )
        for i, layer in enumerate(self.layers):
            # DFlash/DSpark hidden-state capture. The drafter is distilled on
            # the aggregated stream value -- the pre-norm softmax mixture its
            # next consumer sees -- not on the raw prefix sum a layer returns,
            # which is SGLang's fallback for models without the
            # attention-residual scheme. Capturing the prefix sum costs 4.5pt
            # of draft acceptance on K3 + RadixArk DSpark (AR 66.9% -> 71.4%).
            # The tap fires inside layer i+1, which computes that tensor
            # anyway; see its forward docstring. Ground truth: SGLang
            # kimi_k3.py:2697 _dspark_capture_stream, attn_residual.py:313
            # aggregate_stream_torch.
            capture = None
            if (
                spec_metadata is not None
                and i > 0
                and (capture_set is None or self.layers[i - 1].layer_idx in capture_set)
            ):
                capture = (spec_metadata, self.layers[i - 1].layer_idx)
            hidden_states, num_snapshots = layer(
                hidden_states,
                block_residual,
                num_snapshots,
                attn_metadata,
                capture=capture,
            )
        hidden_states = _materialize_prefix_sum(hidden_states)

        # The last layer has no successor, so this one recompute is
        # unavoidable -- output-side score weights, matching SGLang's
        # layer_idx + 1 >= end_layer branch. Unreachable for K3's capture set
        # against 93 layers; kept so a set that does include the final layer
        # gets the right tensor rather than the raw prefix sum.
        if spec_metadata is not None and len(self.layers) > 0:
            last = self.layers[-1]
            if capture_set is None or last.layer_idx in capture_set:
                tail = (
                    _apply_attn_res(
                        hidden_states,
                        block_residual[:num_snapshots],
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    if num_snapshots > 0
                    else hidden_states
                )
                spec_metadata.maybe_capture_hidden_states(last.layer_idx, tail, None)

        output = _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
        )
        return output


# The weight mixins: weights.py imports the blocks above from this module.
from .weights import KimiK3DefaultTextWeights, KimiK3DefaultWeights  # noqa: E402


class KimiK3DefaultForCausalLM(
    KimiK3DefaultTextWeights, KimiK3SpecForCausalLM[KimiK3DefaultModelingV2, Any]
):
    """The Kimi K3 text model (KDA + MLA + MoE decoder layers) of ``ModelingV2KimiK3Nvfp4Sm107Dep16``.

    Registered under no architecture: the target class builds it as its ``llm``. Weight loading
    is the ``KimiK3DefaultTextWeights`` mixin's (``weights.py``).

    Speculative decoding: one-engine DSpark with a standalone MLA drafter, built by the family's
    ``KimiK3SpecForCausalLM`` (``dspark/``): the fused spec tail (``spec_tail_fusion``, not the
    v2 tail), the full drafter of a generation server (not context-KV-only), and no fused decode
    head (no ``RuntimeHooks.decode_prologue``)."""

    mamba_metadata_cls = K3MambaMetadata

    @property
    def dspark_path(self) -> K3DSparkPath:
        """The DSpark deployment path: the fused spec tail, not the v2 tail."""
        return K3DSparkPath(spec_tail_fusion=True, tail_v2=False)

    @property
    def dspark_context_kv_only(self) -> bool:
        """A generation server: the drafter is not the context-KV-only one."""
        return False

    @property
    def spec_decode_prologue(self):
        """No fused decode head."""
        return None

    def __init__(self, model_config: ModelConfig):
        cfg = _get_text_config(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3 does not support pipeline parallelism"
        if model_config.mapping.has_cp_helix():
            raise NotImplementedError("the Kimi K3 dep16 target runs no Helix context parallelism")
        # Speculative decoding is one-engine DSpark with a standalone drafter checkpoint;
        # KimiK3SpecForCausalLM rejects every other speculative config. The target side is the
        # hidden-state capture in KimiK3DefaultModelingV2.forward.
        super().__init__(
            KimiK3DefaultModelingV2(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )


_KV_BALANCED_ADMISSION = K3Dep16KvBalancedAdmission()


class ModelingV2KimiK3Nvfp4Sm107Dep16(KimiK3DefaultWeights, PreTrainedModel):
    """Kimi K3 NVFP4 on SM107, dep16.

    The text model ``KimiK3DefaultForCausalLM`` as ``llm`` behind the vision-language model's
    interface: text only, no vision tower (``mm_encoder`` is ``None``, as under multimodal
    disaggregation), and a step carrying multimodal input is refused. The speculative-decoding
    and weight-loading attributes the engine reads off the outer model (``draft_model``, ``lm_head``,
    ``runtime_hooks``, ...) are the text model's. Weight loading is the
    ``KimiK3DefaultWeights`` mixin's (``weights.py``).
    """

    # The optimizations this target's code is built for (a record: the code reads none of it).
    K3_CONSTANTS: ClassVar[K3Sm107Constants] = CONSTANTS

    _LANG_PREFIX = "language_model."
    mamba_metadata_cls = KimiK3DefaultForCausalLM.mamba_metadata_cls

    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["V2"]:
        """Prefer KV cache manager V2 (as Kimi K2.5)."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["PYTHON"]:
        """Kimi K3 disaggregated serving requires the Python transceiver.

        Only the Python NIXL transceiver (KvCacheTransceiverV2) can move
        the KDA recurrent state; the C++ transceiver has no KDA support.
        Adopted when the user leaves
        ``cache_transceiver_config.transceiver_runtime`` at 'auto' and the
        effective backend is NIXL. An explicit non-Python runtime is
        rejected by ``get_kv_cache_manager_cls`` rather than silently
        routed to a path that cannot transfer the recurrent state.
        """
        return "PYTHON"

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        config = model_config.pretrained_config
        # PreTrainedModel.__init__ resolves config._attn_implementation and
        # rejects the default "sdpa" unless the class declares support. This
        # wrapper never runs HF attention itself (TRT-LLM attention is wired
        # inside the sub-modules), so declare support to keep HF init happy.
        self._supports_sdpa = True
        PreTrainedModel.__init__(self, config)

        # Re-entry guard: the tail of this __init__ repoints
        # model_config.pretrained_config at the text_config and remaps the quant
        # exclude list, so running the body twice on the same instance would
        # fail the text_config assert below. If the text backbone already
        # exists this is a re-init on a built instance: no-op.
        if hasattr(self, "llm"):
            return

        self.model_config = model_config
        log_constants(type(self).__module__, type(self).K3_CONSTANTS)
        self._vlm_pretrained_config = config

        # Text only: no vision tower.
        self.mm_encoder = None

        text_model_config = copy.copy(model_config)
        assert hasattr(config, "text_config"), "Kimi K3 config must have text_config"
        text_model_config._frozen = False
        text_model_config.pretrained_config = config.text_config

        # Remap quant exclude_modules: language_model.X -> model.X
        if text_model_config.quant_config.exclude_modules:
            text_model_config.quant_config = copy.copy(text_model_config.quant_config)
            p = self._LANG_PREFIX
            mapped = []
            for m in text_model_config.quant_config.exclude_modules:
                if m.startswith(p):
                    rest = m[len(p) :]
                    if rest.startswith("layers."):
                        rest = "model." + rest
                    mapped.append(rest)
                else:
                    mapped.append(m)
            text_model_config.quant_config.exclude_modules = mapped

        if not text_model_config.skip_create_weights_in_init:
            text_model_config.skip_create_weights_in_init = True
        text_model_config._frozen = True

        self.llm = KimiK3DefaultForCausalLM(text_model_config)

        # Point model_config at the text_config so the executor reads generation
        # params (eos_token_id, ...) from the text backbone, matching K2.5.
        self.config = self.llm.config
        model_config._frozen = False
        model_config.pretrained_config = self.llm.config
        model_config._frozen = True

    # Forward spec-dec / weight-loading attrs to self.llm: this wrapper is not
    # itself a spec-dec model but is the outer model that those paths see.
    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    @property
    def model(self):
        return self.llm.model

    @property
    def lm_head(self):
        return self.llm.lm_head

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The text model's step-loop hooks (the engine reads them off the outer model), 4 GiB of
        device memory left outside the PyTorch memory cap, and the KV-balanced admission planner
        of the dep16 generation recipe (window 32, the KV load closest to the mean)."""
        return replace(
            self.llm.runtime_hooks or DEFAULT_RUNTIME_HOOKS,
            torch_mem_safety_margin=4 * 1024**3,
            kv_balanced_admission_planner=_KV_BALANCED_ADMISSION,
            kv_cache_manager_cls=KimiK3Sm107CacheManager,
        )

    def set_guided_decoder(self, *args, **kwargs):
        return self.llm.set_guided_decoder(*args, **kwargs)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        multimodal_params: Optional[List[MultimodalParams]] = None,
        **kwargs,
    ) -> torch.Tensor:
        check_text_only(multimodal_params)
        # The text model embeds ``input_ids`` itself; the wrapper hands it no embeddings.
        inputs_embeds = None
        return self.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            **kwargs,
        )


__all__ = ["KimiK3DefaultForCausalLM", "ModelingV2KimiK3Nvfp4Sm107Dep16"]
