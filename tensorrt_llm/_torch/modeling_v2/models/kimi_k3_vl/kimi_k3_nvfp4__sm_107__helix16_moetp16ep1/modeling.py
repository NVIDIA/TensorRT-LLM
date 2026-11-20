# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), Helix CP16 generation server, routed experts MoE TP16: the model.

``ModelingV2KimiK3Nvfp4Sm107Helix16Moetp16ep1`` serves the generation steps of a disaggregated
deployment: 16 ranks, every step generation-only, each request verifying ``1 + 7`` DSpark tokens,
1 to 4 requests (8, 16, 24 or 32 token rows). It loads the vision-language checkpoint's text
model ``KimiK3Helix16UllForCausalLM`` (no vision tower: the context server encodes media).

The step is the served Kimi K3 ULL generation step (``user/xguan/k3-rubin-ull-chain-mlamoe-gsm``
@ 6904cd9672, ``ull/decode.py`` + ``ull/stages.py`` + the ``chain`` / ``dense0`` / ``mlain``
ports) written out as plain calls; ``FLAT_BOUNDARY_SOURCE.json`` maps each function here to its
served source.

The text model has 93 decoder layers: 69 KDA (Kimi Delta Attention) layers, 24 MLA layers
(3, 7, ..., 87, 91, 92), layer 0 with a dense MLP, layers 1..92 with the latent MoE. Most of the
step runs on the MR8 kernel chain (``chain.K3UllChain``, its own weights from
``_chain_weights``): every KDA + MoE layer whole, the MoE of every MLA layer, and the
attention-residual mixes and norms that consume each MoE tail. This file runs the rest, in model
order (:meth:`KimiK3Helix16UllModelingV2.forward`):

* layer 0: the input norm, the KDA in-projection and core on the chain (``run_solo``), the KDA
  o_proj with its pushed TP16 all-reduce and the dense MLP (``layer0.py``);
* layer 1's pre-attention mix (it finishes the dense MLP's all-reduce and releases the chain's
  in-projection at kernel entry);
* every MLA layer's attention (``mla.py``; its input projection folded with the output gate at
  up to 16 rows, ``mlain.py``), its o_proj with the pushed all-reduce and the post-attention mix
  that finishes it;
* the DSpark hidden-state taps (``spec_metadata.maybe_capture_hidden_states``, the pre-norm mix
  each consumer stores).

The output rows are the chain's final consumer's (output attention residual + final norm of layer
92); the one-engine shell then runs the vocab-parallel LM head and the DSpark drafter
(``drafter.py`` / ``dflash.py``).

Caching: ``cache.K3UllCacheManager`` (KV cache manager V2): the MLA layers' NVFP4 latent pages
(FP8 for layers 91 and 92), the drafter's pages, and per KDA layer and request one checkpoint per
verify position; the chain reads and writes those checkpoints, and the drafter's accept tail
records the accepted count that picks the next step's starting checkpoint.

Weights: one copy each. The chain's own containers hold the KDA layers' and the MoE layers'
weights and every layer's attention-residual and norm vectors (the mixes this file runs read
them there); the modules here hold the embedding, LM head, MLA attention, and layer 0's KDA
o_proj and dense MLP. Checkpoint loading is in ``weights.py``.

Merged into this module, in this order (``MOVE.json`` in the family directory):

* ``text_config.py``: Kimi K3 NVFP4 on Rubin (SM107): reading the text config.
* ``layers.py``: Kernel-call primitives of the Kimi K3 Rubin model: K3-owned weights and direct
  kernel calls.
* ``attn_res.py``: Kimi K3 NVFP4 on Rubin (SM107): the attention-residual mixes this target runs off
  the chain.
* ``layer0.py``: Layer 0 (KDA + dense MLP) of the generation step, past its KDA core.
* ``mlain.py``: MLAIN: the FP4-KV MLA layer's input projection with the output gate's rows folded
  into it.
* ``mla.py`` (but its Helix tail workspace): Kimi K3 MLA on Rubin (SM107), Helix CP16: the layer's
  attention half of a generation step.
* ``fp8_weight_read.py``: Kimi K3 NVFP4 on Rubin (SM107): projections that read FP8 block-scaled
  weights.
* ``decode.py``: What the generation step needs around the model: its route check, the KDA slot
  metadata and the chain's process group.
* ``_chain_weights.py``: The kernel chain's weights, read from the Hugging Face checkpoint into
  private containers.
* ``_chain_nv.py``: The NVFP4 routed experts of the chain's MoE layers
  (``moe.prepared.PreparedMoE``).
* ``chain.py``: The KDA + MoE kernel chain of one rank (vLLM MR8's TP16 decode kernels,
  ``kernels/k3r_*/ull/``).
* ``dflash.py``: The target head and the DSpark worker of the Kimi K3 SM107 Helix16 ULL target.
* ``drafter.py``: The DSpark drafter of the Kimi K3 SM107 Helix16 ULL target
  (Inferact/Kimi-K3-DSpark, 5 MLA layers) and its installation on the target.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import math
import platform
import socket
import weakref
from collections import deque
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
import triton
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch import kimi_k3_cache_policy as _kimi_k3_cache_policy
from tensorrt_llm._torch.attention.backends import AttentionMetadata, TrtllmAttentionMetadata
from tensorrt_llm._torch.custom_ops import BufferKind
from tensorrt_llm._torch.distributed import HelixAllToAllNative
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.distributed.ops import (
    MNNVLAllReduce,
    allgather,
    get_allreduce_workspace,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2 import NoTarget, disable_v2_by_default
from tensorrt_llm._torch.modeling_v2.catalog.comm.allreduce import allreduce
from tensorrt_llm._torch.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAReplayState,
    KimiK3HybridCacheManager,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor import step_inputs
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph
from tensorrt_llm._torch.pyexecutor.config_utils import is_mla
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, RuntimeHooks
from tensorrt_llm._torch.speculative import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    SpecWorkerBase,
)
from tensorrt_llm._torch.speculative.spec_tail_kernels import decode_draft_block_table
from tensorrt_llm._torch.utils import AuxStreamType, get_model_extra_attrs
from tensorrt_llm._utils import (
    get_sm_version,
    is_device_integrated,
    is_sm_100f,
    mpi_comm,
    prefer_pinned,
)
from tensorrt_llm.functional import (
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
    RopeEmbeddingUtils,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..common.helix_kv_mask import _helix_zero_kv_mask
from ..common.multi_stream import do_multi_stream
from ..common.parallel import TensorParallelMode
from ..common.spec_positions import _slice_spec_position_ids
from ..kernels.k3r_attn_res import k3_attn_res_ull as k3_attn_res
from ..kernels.k3r_gemm.low_m_bf16_splitk import SplitKTactic, run_splitk_dense
from ..kernels.k3r_mla_decode.cute_dsl_mla import launch_mla_split_kv_plan
from ..kernels.k3r_moe import k3_route_tail_ull  # noqa: F401 - registers the dispatcher op
from ..kernels.k3r_spec.dspark_draft3 import draft_qhead, fold_rmsnorm_weight
from ..kernels.k3r_spec.dspark_draft_ar_ull import (
    ELTS,
    EMBED_THREADS,
    ar_norm_config,
    run_draft_ar_norm,
)
from ..kernels.k3r_spec.dspark_draft_gemm_push_ull import (
    ensure_cute_runtime,
    run_gemm_push,
    validate_push_tactic,
)
from ..kernels.k3r_spec.dspark_ull2 import draft_helix_tail
from ..kernels.k3r_spec.spec_decode_head import spec_decode_head
from ..kernels.k3r_spec.spec_tail_kernels_ull import (
    TAIL_V2_MAX_GENS,
    dflash_ctx_positions,
    dflash_gen_prepare,
    dflash_noise_embedding,
    fused_next_new_tokens,
    fused_next_new_tokens_rewind,
    fused_select_accept,
    fused_strict_accept,
    store_mla_context_latents,
)
from ..kernels.k3r_spec.vocab_parallel import (
    _LOCAL_BLOCK_V,
    _REDUCE_BLOCK_ROWS,
    _TARGET_PROGRAMS,
    _local_argmax_kernel,
    _local_sample_kernel,
    _PairLayout,
    _reduce_pairs_kernel,
)
from ..ops.attention import _fp4_mla_cache_update as fp4_mla_cache_update
from ..ops.attention import _fp4_mla_decode_ull as fp4_mla_decode
from ..ops.attention import k3_fp4_mla_decode_ull as _decode_op
from ..ops.attention import k3_mla_cache as mla_cache
from ..ops.attention import k3_mla_decode_fp8 as _fp8_decode
from ..ops.attention.k3_mla_state import K3MlaAttnState
from ..ops.attention.k3_ull_helix_tail_workspace import K3HelixTail, ull2_tail_enabled
from ..ops.cache.k3_ull_cache_manager import K3UllCacheManager, K3UllKdaCheckpointState
from ..ops.chain.k3_ull_chain import K3UllChain
from ..ops.chain.mr8_chain import mr8_chain_moe, mr8_chained_layer, mr8_solo_kda
from ..ops.comm.k3_attn_res_lamport_ull import k3_attn_res_lamport
from ..ops.common.k3_ull_features import PROFILE, K3Features
from ..ops.common.k3_ull_features import active as _features
from ..ops.common.k3_ull_runtime import make_runtime_hooks
from ..ops.common.sm107_launch import USE_PDL, log_constants
from ..ops.gemm.k3_fp8w_lowm_gemm_push import k3_fp8w_lowm_gemm_push as k3_fp8w_lowm_gemm_push_op
from ..ops.gemm.k3_fp8w_lowm_gemm_ull import k3_fp8w_lowm_gemm as k3_fp8w_lowm_gemm_op
from ..ops.gemm.k3_fp8w_lowm_mxfp8_gemm_ull import (
    k3_fp8w_lowm_mxfp8_gemm as k3_fp8w_lowm_mxfp8_gemm_op,
)
from ..ops.gemm.k3_fp8w_lowm_mxfp8_push_ull import (
    k3_fp8w_lowm_mxfp8_push as k3_fp8w_lowm_mxfp8_push_op,
)
from ..ops.gemm.k3_fp8w_mid_gemm import k3_fp8w_mid_gemm as k3_fp8w_mid_gemm_op
from ..ops.gemm.k3_fp8w_mid_gemm_push import k3_fp8w_mid_gemm_push as k3_fp8w_mid_gemm_push_op
from ..ops.gemm.k3_mxfp8_gemm_ull import k3_mxfp8_gemm as k3_mxfp8_gemm_op
from ..ops.norm.attn_res_fwd_ull import attn_res_fwd
from ..ops.norm.k3_attn_res_norm_ull import k3_attn_res_norm
from ..ops.spec.k3_draft_embed_ull import k3_draft_embed
from ..ops.spec.k3_dspark_mla_ctx_norm_rope import k3_dspark_mla_ctx_norm_rope


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


MNNVL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
#: Bytes (``tokens * hidden * ranks * element size``) up to which an MNNVL all-reduce is one-shot:
#: the deployment's ``TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES`` (served ``distributed/ops.py``
#: ``_MNNVL_ONE_SHOT_THRESHOLD_BYTES``).
MNNVL_ONE_SHOT_MAX_BYTES = 2097152
#: Fusions ``trtllm::mnnvl_fusion_allreduce`` runs (besides ``NONE``).
MNNVL_FUSION_OPS = frozenset(
    {
        AllReduceFusionOp.RMS_NORM,
        AllReduceFusionOp.RESIDUAL_RMS_NORM,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
    }
)


def mnnvl_serves(mapping: Mapping, dtype: torch.dtype, strategy: AllReduceStrategy) -> bool:
    """Whether the all-reduce of ``mapping`` runs on MNNVL for ``strategy`` (``AUTO`` / ``MNNVL``)."""
    if strategy not in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL):
        return False
    if dtype not in MNNVL_DTYPES or mapping.has_cp():
        return False
    supported = MnnvlMemory.supports_mnnvl() and "aarch64" in platform.machine().lower()
    return supported and (strategy == AllReduceStrategy.MNNVL or mapping.is_multi_node())


def mnnvl_workspace_bytes(num_tokens: int, hidden: int, world: int, dtype: torch.dtype) -> int:
    """Lamport buffer bytes one MNNVL call of ``[num_tokens, hidden]`` needs."""
    elem = torch.tensor([], dtype=dtype).element_size()
    if num_tokens * hidden * world * elem <= MNNVL_ONE_SHOT_MAX_BYTES:
        return num_tokens * hidden * world * elem
    return 2 * math.ceil(num_tokens / world) * world * hidden * elem


class TpAllReduce:
    """The all-reduce over ``mapping``'s TP group; ``self(x)`` returns the reduced tensor."""

    def __init__(
        self,
        mapping: Mapping,
        dtype: torch.dtype,
        strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.strategy = strategy
        self.symm_mem_allreduce = None
        self.transport = "identity"
        self.workspace = None
        if mapping.tp_size <= 1 or mapping.enable_attention_dp:
            return
        if mnnvl_serves(mapping, dtype, strategy):
            self.transport = "mnnvl"
            get_or_scale_allreduce_mnnvl_workspace(mapping, dtype)
            return
        if strategy in (
            AllReduceStrategy.MNNVL,
            AllReduceStrategy.SYMM_MEM,
            AllReduceStrategy.UB,
            AllReduceStrategy.LOWPRECISION,
        ):
            raise ValueError(f"K3 all-reduce: strategy {strategy.name} is not served here")
        self.transport = "op"
        if strategy not in (AllReduceStrategy.NCCL, AllReduceStrategy.NCCL_SYMMETRIC):
            self.workspace = get_allreduce_workspace(mapping)

    get_required_workspace_size = staticmethod(mnnvl_workspace_bytes)

    @property
    def mnnvl_allreduce(self) -> Optional["TpAllReduce"]:
        """Self on the MNNVL transport (what ``layers.py`` reads), else None."""
        return self if self.transport == "mnnvl" else None

    def __call__(
        self, x: torch.Tensor, *, all_reduce_params: Optional[AllReduceParams] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        params = all_reduce_params
        if self.transport == "identity" or (params is not None and not params.enable_allreduce):
            return x
        x = x.contiguous()
        params = params or AllReduceParams()
        if self.transport == "mnnvl":
            return self._mnnvl(x, params)
        return allreduce(
            x,
            params.residual,
            params.norm_weight,
            params.scale,
            params.bias,
            self.workspace,
            self.mapping.tp_group,
            self.strategy,
            params.fusion_op,
            params.eps,
            params.trigger_completion_at_end,
        )

    def _mnnvl(
        self, x: torch.Tensor, params: AllReduceParams
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        fusion_op = params.fusion_op
        is_fusion = fusion_op != AllReduceFusionOp.NONE
        if is_fusion and fusion_op not in MNNVL_FUSION_OPS:
            raise ValueError(f"the MNNVL all-reduce does not fuse {fusion_op!r}")
        hidden = x.shape[-1]
        num_tokens = x.numel() // hidden
        need = mnnvl_workspace_bytes(num_tokens, hidden, self.mapping.tp_size, self.dtype)
        if need >= 2**32 - 1:
            raise ValueError(
                f"MNNVL all-reduce of ({num_tokens}, {hidden}) over {self.mapping.tp_size} ranks "
                f"needs {need} workspace bytes, past the kernel's uint32 limit"
            )
        workspace = get_or_scale_allreduce_mnnvl_workspace(
            self.mapping, self.dtype, buffer_size_bytes=need
        )
        outputs = torch.ops.trtllm.mnnvl_fusion_allreduce(
            x,
            params.norm_weight,
            params.residual,
            params.eps,
            workspace["uc_buffer"].view(self.dtype).view(3, -1),
            workspace["buffer_flags"],
            is_fusion,
            params.scale,
            int(fusion_op),
            fp8_payload=False,
        )
        if is_fusion and len(outputs) > 1:
            return tuple(outputs)
        return outputs[0]


FP8_BLOCK = 128


def _rubin_mxfp8() -> bool:
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _split(width: int, parts: int, index: int) -> int:
    return (width // parts) * index + min(width % parts, index)


def shard_range(width: int, tp_size: int, tp_rank: int, alignment: int = 1) -> Tuple[int, int]:
    """``[start, end)`` of rank ``tp_rank`` over a split dim of ``width`` elements."""
    if alignment <= 1:
        return _split(width, tp_size, tp_rank), _split(width, tp_size, tp_rank + 1)
    if width % alignment:
        raise ValueError(f"split dim {width} is not a multiple of {alignment}")
    blocks = width // alignment
    return (
        _split(blocks, tp_size, tp_rank) * alignment,
        _split(blocks, tp_size, tp_rank + 1) * alignment,
    )


def load_shard(
    tensor: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    mode: Optional[TensorParallelMode],
    *,
    features: Optional[int] = None,
    alignment: int = 1,
    scale_span: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """This rank's slice of a checkpoint tensor (a weight, or with ``scale_span`` its block-scale
    grid) on ``device``. ``features``: the full split-dim width of the weight (default: the
    tensor's). A 1-D tensor of a ``ROW`` projection, or a split dim of width 1, is not split."""
    device = device if device is not None else tensor.device
    if mode is None or tp_size <= 1:
        return tensor.to(device)
    dim = TensorParallelMode.split_dim(mode)
    if (tensor.dim() == 1 and dim == 1) or tensor.shape[dim] == 1:
        return tensor.to(device)
    start, end = shard_range(features or tensor.shape[dim], tp_size, tp_rank, alignment)
    if scale_span:
        if start % scale_span or end % scale_span:
            raise ValueError(f"shard [{start}, {end}) does not align with {scale_span}-wide scales")
        start, end = start // scale_span, end // scale_span
    return tensor.narrow(dim, start, end - start).to(device)


def load_weight_shard(
    weight,
    tensor_parallel_size: int = 1,
    tensor_parallel_rank: int = 0,
    tensor_parallel_mode: Optional[TensorParallelMode] = None,
    device: torch.device = torch.device("cpu"),
    return_slice_indices: bool = False,
) -> torch.Tensor:
    """This rank's ceil-divided slice of ``weight`` (a tensor or a safetensors slice) on ``device``:
    ``ceil(width / tp)`` elements of the split dim per rank, the last rank the remainder. A 1-D
    tensor of a ``ROW`` projection, or a split dim of width 1, is not split. On an integrated GPU
    the tensor stays on its own device. ``return_slice_indices`` (tensors only): also the slices."""
    # Skip device transfers on integrated GPUs to conserve shared memory
    if weight.device.type != device.type and is_device_integrated():
        # For integrated GPU systems (e.g., DGX Spark), CPU and GPU share limited physical memory.
        # Avoiding device transfers reduces memory consumption and unnecessary data copies,
        # enabling support for larger models on memory-constrained systems.
        logger.warning_once(
            f"[load_weight_shard] Skipping device transfer from {weight.device} to {device} "
            "on integrated GPU to conserve shared memory.",
            key="load_weight_shard_skip_device_transfer_with_integrated_gpu",
        )
        device = weight.device
    if isinstance(weight, torch.Tensor):
        tensor_shape = weight.shape

        def maybe_convert_to_torch_tensor(tensor: torch.Tensor, indices: list[slice] | None = None):
            if indices is None:
                # Avoid unnecessary copy
                result = (tensor.to(device), [slice(d) for d in tensor.shape])
            else:
                result = (tensor[indices].to(device), indices)
            return result if return_slice_indices else result[0]

    # WAR to check whether it is a safetensor slice since safetensor didn't register the type to
    # the module; safetensors slice, supports lazy loading, type(weight) is `builtin.PySafeSlice`
    elif hasattr(weight, "get_shape"):
        tensor_shape = weight.get_shape()

        def maybe_convert_to_torch_tensor(
            tensor, indices: Union[slice, tuple[slice]] = slice(None)
        ):
            return tensor[indices].to(device)
    else:
        raise ValueError(f"unsupported weight type: {type(weight)}")
    if tensor_parallel_mode is None or tensor_parallel_size <= 1:
        return maybe_convert_to_torch_tensor(weight)

    split_dim = TensorParallelMode.split_dim(tensor_parallel_mode)

    if len(tensor_shape) == 1 and split_dim == 1:
        return maybe_convert_to_torch_tensor(weight)

    width = tensor_shape[split_dim]
    if width == 1:
        return maybe_convert_to_torch_tensor(weight)

    slice_width = math.ceil(width / tensor_parallel_size)
    slice_start = tensor_parallel_rank * slice_width
    slice_end = min((tensor_parallel_rank + 1) * slice_width, width)
    slice_obj = [slice(d) for d in tensor_shape]
    slice_obj[split_dim] = slice(slice_start, slice_end)
    return maybe_convert_to_torch_tensor(weight, tuple(slice_obj))


class _ShardedLinear(nn.Module):
    alignment = 1

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        tp_mode: Optional[TensorParallelMode],
    ) -> None:
        super().__init__()
        self.tp_size, self.tp_rank, self.tp_mode = tp_size, tp_rank, tp_mode
        self.full_in_features, self.full_out_features = in_features, out_features
        self.in_features, self.out_features = in_features, out_features
        if tp_size > 1 and tp_mode is not None:
            dim_width = in_features if tp_mode == TensorParallelMode.ROW else out_features
            start, end = shard_range(dim_width, tp_size, tp_rank, self.alignment)
            if tp_mode == TensorParallelMode.ROW:
                self.in_features = end - start
            else:
                self.out_features = end - start

    def split_features(self) -> int:
        return (
            self.full_in_features
            if self.tp_mode == TensorParallelMode.ROW
            else self.full_out_features
        )

    def shard(self, tensor: torch.Tensor, scale_span: Optional[int] = None) -> torch.Tensor:
        """This rank's slice of a full checkpoint tensor of the projection, on its device."""
        return load_shard(
            tensor,
            self.tp_size,
            self.tp_rank,
            self.tp_mode,
            features=self.split_features(),
            alignment=self.alignment,
            scale_span=scale_span,
            device=self.weight.device,
        )


def bf16_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``x @ weight.T (+ bias)``: cuBLASLt via ``F.linear``."""
    return F.linear(x, weight, bias)


class Bf16Linear(_ShardedLinear):
    """BF16 (or FP16 / FP32) ``weight [out, in]`` (and ``bias [out]``) of one rank."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_mode: Optional[TensorParallelMode] = None,
        all_reduce: Optional[Callable[..., torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(in_features, out_features, tp_size, tp_rank, tp_mode)
        if all_reduce is not None and tp_mode != TensorParallelMode.ROW:
            raise ValueError("only a row-parallel projection reduces its output")
        self.all_reduce = all_reduce
        self.reduce_output = all_reduce is not None
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=dtype, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, dtype=dtype, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def load(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        """Copy this rank's shard of the full checkpoint ``weight`` (and ``bias``)."""
        self.weight.data.copy_(self.shard(weight).to(self.weight.dtype))
        if bias is not None:
            self.bias.data.copy_(self.shard(bias).to(self.bias.dtype))

    def forward(
        self, x: torch.Tensor, *, all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        # Bias on rank 0 only for a row-parallel projection (its partial sums are all-reduced).
        bias = self.bias if self.tp_mode != TensorParallelMode.ROW or self.tp_rank == 0 else None
        out = bf16_linear(x, self.weight, bias)
        if self.all_reduce is not None:
            out = self.all_reduce(out, all_reduce_params=all_reduce_params)
        return out


def prepare_fp8_block_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scale)`` for :func:`fp8_block_scale_gemm` from checkpoint E4M3 codes and their FP32
    128 x 128 grid: scales rounded to UE8M0 with the codes rescaled to match (in place), then the
    scale laid out for this GPU's GEMM (K32 R128c4 UE8M0 bytes on SM107, DeepGEMM-packed UE8M0
    otherwise)."""
    from tensorrt_llm.quantization.utils.fp8_utils import (
        resmooth_to_fp8_e8m0,
        transform_k128_scales_to_cutedsl_mxfp8_layout,
        transform_sf_into_required_layout,
    )

    weight, weight_scale = resmooth_to_fp8_e8m0(
        weight.contiguous(), weight_scale.contiguous().float()
    )
    n, k = weight.shape
    if _rubin_mxfp8():
        return weight, transform_k128_scales_to_cutedsl_mxfp8_layout(weight_scale, mn=n, k=k)
    return weight, transform_sf_into_required_layout(
        weight_scale, mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
    )


def low_m_gemm_rows(n: int, k: int) -> int:
    """Largest M ``trtllm::fp8_block_scale_low_m_gemm`` serves for an ``[n, k]`` weight here (0:
    none)."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        return 0
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.max_m_for(n, k)


def low_m_gemm_scale(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The DeepGEMM-packed scale the low-M GEMM reads, from a :func:`prepare_fp8_block_weight`
    scale (itself on SM100 / SM103, repacked from K32 R128c4 on SM107)."""
    if not _rubin_mxfp8():
        return weight_scale
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.pack_block_scale_from_mxfp8_layout(weight_scale, n, k)


def fp8_block_scale_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    low_m_max: int = 0,
    low_m_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BF16 ``x [M, K] @ weight.T`` on :func:`prepare_fp8_block_weight` operands (see the module
    docstring for the chain per GPU)."""
    if (
        low_m_max
        and 1 <= x.shape[0] <= low_m_max
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    ):
        return torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, weight, low_m_scale)
    if _rubin_mxfp8():
        from ..ops.gemm.k3_mxfp8_gemm_ull import k3_mxfp8_gemm

        act, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x)
        return k3_mxfp8_gemm(act, weight, act_scale, weight_scale)
    return torch.ops.trtllm.fp8_swap_ab_gemm(
        x, weight, weight_scale, output_dtype=x.dtype, disable_ue8m0_cast=True
    )


class Fp8BlockScaleLinear(_ShardedLinear):
    """FP8 block-scale ``weight`` / ``weight_scale`` of one rank (plus the unit ``input_scale`` /
    ``inv_input_scale`` checkpoint FP8 projections also register), bias-free."""

    alignment = FP8_BLOCK

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_mode: Optional[TensorParallelMode] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(in_features, out_features, tp_size, tp_rank, tp_mode)
        n, k = self.out_features, self.in_features
        self.weight = nn.Parameter(
            torch.empty(n, k, dtype=torch.float8_e4m3fn, device=device), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(-(-n // FP8_BLOCK), -(-k // FP8_BLOCK), dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.input_scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device), requires_grad=False
        )
        self.inv_input_scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device), requires_grad=False
        )
        self.register_parameter("bias", None)
        self.prepared = False
        self.low_m_max = 0
        self.low_m_scale: Optional[torch.Tensor] = None

    def load(self, weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
        """Copy this rank's shard of the full checkpoint codes and their FP32 block-scale grid
        (a modelopt ``[N/128, 1, K/128, 1]`` grid is squeezed first)."""
        if weight_scale.dim() == 4:
            weight_scale = weight_scale.squeeze(1).squeeze(-1)
        self.weight.data.copy_(self.shard(weight))
        self.weight_scale.data.copy_(self.shard(weight_scale, scale_span=FP8_BLOCK).float())
        self.prepared = False

    def prepare(self, low_m: bool = True) -> None:
        """Once after loading: :func:`prepare_fp8_block_weight` in place of the parameters, and
        with ``low_m`` the rows the low-M GEMM serves and its scale."""
        if self.prepared:
            return
        weight, scale = prepare_fp8_block_weight(self.weight.data, self.weight_scale.data)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)
        n, k = self.out_features, self.in_features
        self.low_m_max = low_m_gemm_rows(n, k) if low_m else 0
        self.low_m_scale = low_m_gemm_scale(scale, n, k) if self.low_m_max else None
        self.prepared = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.prepared:
            raise RuntimeError("Fp8BlockScaleLinear.prepare() must run after loading")
        x_flat = x.reshape(-1, x.shape[-1])
        out = fp8_block_scale_gemm(
            x_flat,
            self.weight,
            self.weight_scale,
            low_m_max=self.low_m_max,
            low_m_scale=self.low_m_scale,
        )
        return out.reshape(*x.shape[:-1], self.out_features)


_FLASHINFER_DTYPES = (torch.bfloat16, torch.float16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``weight * x / sqrt(mean(x^2) + eps)`` over the last dim of a BF16 / FP16 ``x``."""
    if x.dtype not in _FLASHINFER_DTYPES:
        raise TypeError(f"rms_norm takes BF16 / FP16 activations, got {x.dtype}")
    return flashinfer_rmsnorm(x, weight, eps)


def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In place: ``residual += x``, then ``x = rms_norm(residual)``. Returns ``(x, residual)``."""
    if x.dtype not in _FLASHINFER_DTYPES:
        raise TypeError(f"fused_add_rms_norm takes BF16 / FP16 activations, got {x.dtype}")
    return flashinfer_fused_add_rmsnorm(x, residual, weight, eps)


class RMSNormWeight(nn.Module):
    """The ``[hidden]`` RMSNorm weight (``weight``, initialized to ones) and its epsilon."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    @property
    def eps(self) -> float:
        """``variance_epsilon``, under the name the gated (KDA ``o_norm``) and attention-residual
        norm readers use."""
        return self.variance_epsilon

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return rms_norm(x, self.weight, self.variance_epsilon)
        return fused_add_rms_norm(x, residual, self.weight, self.variance_epsilon)


def _record_on(obj: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            try:
                obj.record_stream(stream)
            except RuntimeError as error:
                # cudaMallocAsync asserts on memory it did not allocate (an external buffer wrapped
                # as a tensor), which it never frees: nothing to record.
                if (
                    torch.cuda.memory.get_allocator_backend() != "cudaMallocAsync"
                    or "ptr not found in ptr_info" not in str(error)
                ):
                    raise
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _record_on(item, stream)
    elif isinstance(obj, dict):
        for item in obj.values():
            _record_on(item, stream)


def fork(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    fork_event: torch.cuda.Event,
    join_event: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream],
) -> Tuple[Any, Any, Optional[torch.cuda.Event]]:
    """``fn0()`` and ``fn1()`` and the event the calling stream must wait on before it reads
    ``fn1()``'s outputs (None when both ran on the calling stream)."""
    if not (do_multi_stream() and aux_stream is not None):
        return fn0(), fn1(), None
    fork_event.record()
    result0 = fn0()
    with torch.cuda.stream(aux_stream):
        fork_event.wait()
        result1 = fn1()
        join_event.record()
    if not torch.cuda.is_current_stream_capturing():
        _record_on(result1, torch.cuda.current_stream())
    return result0, result1, join_event


def fork_join(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    fork_event: torch.cuda.Event,
    join_event: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream],
) -> Tuple[Any, Any]:
    """``(fn0(), fn1())``, ``fn1`` on ``aux_stream`` where :func:`fork` forks; joined."""
    result0, result1, join = fork(fn0, fn1, fork_event, join_event, aux_stream)
    if join is not None:
        join.wait()
    return result0, result1


def masked_embedding(ids: torch.Tensor, weight: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Rows ``ids - start`` of ``weight`` where ``start <= ids < end``, zeros elsewhere."""
    hit = (ids >= start) & (ids < end)
    out = F.embedding(hit * (ids - start * hit), weight)
    out.masked_fill_(~hit.unsqueeze(-1), 0)
    return out


def gather_embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """:func:`masked_embedding` of a whole table in one Triton launch (PDL where enabled)."""
    from ..ops.embedding.k3_embedding_gather import k3_embedding_gather

    return k3_embedding_gather(ids, weight)


class EmbeddingTable(nn.Module):
    """``weight`` of one rank and the lookup above."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        vocab_parallel: bool = False,
        reduce_output: bool = True,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        mapping = mapping or Mapping()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.vocab_parallel = (
            vocab_parallel and mapping.tp_size > 1 and not mapping.enable_attention_dp
        )
        if self.vocab_parallel:
            rows = math.ceil(num_embeddings / mapping.tp_size)
            self.vocab_start = mapping.tp_rank * rows
            self.vocab_end = min(self.vocab_start + rows, num_embeddings)
            self.all_reduce = (
                TpAllReduce(mapping, dtype or torch.bfloat16, allreduce_strategy)
                if reduce_output
                else None
            )
        else:
            rows = num_embeddings
            self.vocab_start, self.vocab_end = 0, num_embeddings
            self.all_reduce = None
        self.weight = nn.Parameter(
            torch.empty(rows, embedding_dim, dtype=dtype, device=device), requires_grad=False
        )

    def load(self, weight: torch.Tensor) -> None:
        """Copy this rank's rows of the full checkpoint table (zero padding rows past the vocab)."""
        rows = self.vocab_end - self.vocab_start
        self.weight.data[:rows].copy_(weight[self.vocab_start : self.vocab_end])
        self.weight.data[rows:].zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out = masked_embedding(
            ids,
            self.weight,
            self.vocab_start,
            self.vocab_end if self.vocab_parallel else self.weight.shape[0],
        )
        if self.all_reduce is not None:
            out = self.all_reduce(out)
        return out


def subgroup_mapping(mapping: Mapping, tp_size: int) -> Mapping:
    """``mapping`` regrouped into TP groups of ``tp_size`` consecutive ranks."""
    if mapping.tp_size % tp_size:
        raise ValueError(f"TP {mapping.tp_size} does not split into groups of {tp_size}")
    pp_size = mapping.pp_size * mapping.tp_size // tp_size
    return Mapping(
        world_size=tp_size * pp_size,
        rank=mapping.rank,
        gpus_per_node=mapping.gpus_per_node,
        tp_size=tp_size,
        pp_size=pp_size,
        enable_attention_dp=mapping.enable_attention_dp,
    )


def gated_activation(
    h: torch.Tensor, activation: str, beta: float = 1.0, linear_beta: Optional[float] = None
) -> torch.Tensor:
    """``act(gate) * up`` of a packed ``[..., 2 * I]`` = ``[gate | up]`` tensor."""
    if activation == "situ":
        out = torch.ops.trtllm.situ_and_mul(h.reshape(-1, h.shape[-1]), beta, linear_beta)
        return out.reshape(*h.shape[:-1], h.shape[-1] // 2)
    if activation == "silu":
        return torch.ops.trtllm.silu_and_mul(
            h, swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None
        )
    raise ValueError(f"unknown gated activation {activation!r}")


class GatedMlp(nn.Module):
    """``gate_up_proj`` / ``down_proj`` BF16 weights of one rank and the chain above."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: str,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
        dtype: torch.dtype = torch.bfloat16,
        mapping: Optional[Mapping] = None,
        reduce_output: bool = True,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        mapping = mapping or Mapping()
        self.mapping = mapping
        self.activation, self.beta, self.linear_beta = activation, beta, linear_beta
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        tp, rank = mapping.tp_size, mapping.tp_rank
        self.inter_start, self.inter_end = shard_range(intermediate_size, tp, rank)
        local = self.inter_end - self.inter_start
        self.gate_up_proj = Bf16Linear(hidden_size, 2 * local, dtype=dtype, device=device)
        self.down_proj = Bf16Linear(
            intermediate_size,
            hidden_size,
            dtype=dtype,
            tp_size=tp,
            tp_rank=rank,
            tp_mode=TensorParallelMode.ROW,
            all_reduce=(
                TpAllReduce(mapping, dtype, allreduce_strategy)
                if reduce_output and tp > 1
                else None
            ),
            device=device,
        )

    def load(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
        """Copy this rank's rows of the full checkpoint gate / up ``[I, H]`` and columns of down
        ``[H, I]``."""
        start, end = self.inter_start, self.inter_end
        local = end - start
        weight = self.gate_up_proj.weight
        weight.data[:local].copy_(gate[start:end].to(weight.device, weight.dtype))
        weight.data[local:].copy_(up[start:end].to(weight.device, weight.dtype))
        self.down_proj.load(down)

    def forward(
        self, x: torch.Tensor, final_all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        h = bf16_linear(x, self.gate_up_proj.weight)
        a = gated_activation(h, self.activation, self.beta, self.linear_beta)
        return self.down_proj(a, all_reduce_params=final_all_reduce_params)


# Pushed all-reduces of at most this many bytes (num_tokens * hidden_dim * group_size * 2) use the
# one-shot transport, larger ones the two-shot transport: the transport of the C++ kernel the
# call replaces. The result reproduces that C++ kernel.
_K3_AR_PUSH_ONE_SHOT_MAX_BYTES = MNNVL_ONE_SHOT_MAX_BYTES
# Pushed all-reduces whose consumer finishes them (``deferred``: the K3 attn_res kernel reduces
# the slots itself, after o_proj and after the row-parallel MoE tail) use the one-shot transport up
# to this many bytes, the two-shot transport above.
_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES = _K3_AR_PUSH_ONE_SHOT_MAX_BYTES


@dataclasses.dataclass(frozen=True)
class LamportWorkspaceView:
    """Device addresses of one MNNVL all-reduce workspace for pushed calls, as the routed experts'
    output push takes them.

    ``buffer``: this rank's unicast mapping of the workspace; ``mc_base``: its multicast mapping;
    ``peer_ptrs``: int64 CUDA tensor with every rank's unicast mapping (this rank's own at index
    ``rank``); ``flags``: the uint32 rotation state shared with the C++ kernels;
    ``bytes_per_buffer``: size of one of the three Lamport buffers.
    """

    world: int
    rank: int
    buffer: torch.Tensor
    mc_base: int
    peer_ptrs: torch.Tensor
    flags: torch.Tensor
    bytes_per_buffer: int

    @property
    def uc_local(self) -> int:
        """Device address of ``buffer``."""
        return self.buffer.data_ptr()


# Device views of the workspaces for pushed all-reduces, by the address of the workspace's buffer
# (a scaled workspace is a new buffer).
_LAMPORT_VIEWS: Dict[int, object] = {}
# The latest shared MNNVL workspace an eager call here got, by all-reduce mapping (workspaces only
# grow: a later one is at least as large).
_SEEN_WORKSPACES: Dict[Mapping, Dict[str, Any]] = {}


class LamportPushTarget:
    """One MNNVL all-reduce whose input the producer pushes.

    Made by :func:`lamport_push_target` for a single call. The producer stores its BF16
    ``[num_tokens, hidden]`` output into the Lamport slots of ``workspace`` (a
    ``k3_lamport_ar.LamportWorkspace``) with the ``one_shot`` or two-shot transport and sets
    ``pushed``; :func:`finish_lamport_push` then writes the all-reduce result into ``output``,
    reproducing the C++ one-shot kernel if ``cpp_one_shot``, else the two-shot kernel.
    """

    def __init__(self, workspace, one_shot: bool, cpp_one_shot: bool, output: torch.Tensor):
        self.workspace = workspace
        self.one_shot = one_shot
        self.cpp_one_shot = cpp_one_shot
        self.output = output
        self.pushed = False
        # Set by ``lamport_push_target(deferred=True)``: the one-shot transport, finished by the
        # consumer (``LamportPending``).
        self.deferred = False
        # The call carries all-gather rows (``k3_lamport_ar`` "ag", one-shot transport): its
        # result is ``bf16(y + bf16(sum of the slots))``.
        self.ag = False


class LamportPending:
    """A pushed all-reduce whose finish is left to its consumer
    (``finish_lamport_push(..., deferred=True)``).

    Every rank has stored its partial into the slots of the call (one-shot or two-shot
    transport), but no kernel has reduced them or advanced the workspace's rotation state: the
    consumer's kernel does both (``k3_attn_res_lamport``), or ``materialize`` runs
    the finish kernel into a tensor. Exactly one of the two must run, before the next all-reduce on
    the workspace.
    """

    def __init__(self, target: LamportPushTarget):
        self.target = target
        self.workspace = target.workspace
        self.shape = tuple(target.output.shape)

    @property
    def finish_args(self) -> Tuple[int, int, bool, int, int]:
        """``(finish_world, finish_chunk, finish_two_shot, mc_base, rank)`` of
        ``k3_attn_res_lamport``: the ranks per partial sum of the C++ kernel the result
        reproduces, the transport and the workspace addresses."""
        from ..kernels.k3r_comm import k3_lamport_ar

        ws = self.workspace
        return (
            ws.world,
            k3_lamport_ar.reduce_chunk(ws.world, self.target.cpp_one_shot),
            not self.target.one_shot,
            ws.mc_base,
            ws.rank,
        )

    @property
    def finish_ag(self) -> bool:
        """Whether the call carries all-gather rows (``finish_ag`` of ``k3_attn_res_lamport``)."""
        return getattr(self.target, "ag", False)

    def materialize(self) -> torch.Tensor:
        """Finish the all-reduce into a BF16 tensor (the finish kernel on the current stream)."""
        from ..ops.comm.k3_lamport_finish import k3_lamport_finish

        return k3_lamport_finish(
            self.workspace,
            self.target.output,
            self.target.one_shot,
            use_pdl=USE_PDL,
            cpp_one_shot=self.target.cpp_one_shot,
            ag=self.finish_ag,
        )


def _lamport_view(mnnvl: Any, workspace: dict):
    """The ``k3_lamport_ar.LamportWorkspace`` of an MNNVL workspace (built once per buffer)."""
    key = workspace["uc_buffer"].data_ptr()
    view = _LAMPORT_VIEWS.get(key)
    if view is None:
        from ..kernels.k3r_comm import k3_lamport_ar

        view = k3_lamport_ar.LamportWorkspace.from_mnnvl(
            workspace, mnnvl.mapping.tp_size, mnnvl.mapping.tp_rank
        )
        _LAMPORT_VIEWS[key] = view
        logger.info_once(
            f"[MNNVL AllReduce] pushed all-reduce: Lamport workspace view "
            f"for tp_size {mnnvl.mapping.tp_size} ({view.bytes_per_buffer} bytes per buffer)",
            key=("k3_ar_push_view", mnnvl.mapping.tp_size),
        )
    return view


def mnnvl_lamport_push_target(
    mnnvl: Any,
    num_tokens: int,
    hidden_dim: int,
    one_shot: Optional[bool] = None,
) -> Optional[LamportPushTarget]:
    """``MNNVLAllReduce.lamport_push_target`` of the branch: a target for a plain all-reduce of
    BF16 ``[num_tokens, hidden_dim]`` on the workspace of ``mnnvl`` (an all-reduce's MNNVL
    transport, ``mnnvl_allreduce``), or None. ``one_shot`` forces the transport instead of the
    size rule."""
    from ..kernels.k3r_comm import k3_lamport_ar

    world = mnnvl.mapping.tp_size
    nbytes = num_tokens * hidden_dim * world * 2
    cpp_one_shot = nbytes <= MNNVL_ONE_SHOT_MAX_BYTES
    if one_shot is None:
        one_shot = nbytes <= _K3_AR_PUSH_ONE_SHOT_MAX_BYTES
    need = max(
        mnnvl.get_required_workspace_size(num_tokens, hidden_dim, world, mnnvl.dtype),
        k3_lamport_ar.required_bytes(num_tokens, hidden_dim, world, one_shot),
    )
    if need >= 2**32 - 1:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        # Main raises when a workspace is created or grows during capture, and building the view
        # is a host-to-device copy: both must have happened in the eager warm-up of this shape.
        seen = _SEEN_WORKSPACES.get(mnnvl.mapping)
        if (
            seen is None
            or seen["buffer_size_bytes"] < need
            or seen["uc_buffer"].data_ptr() not in _LAMPORT_VIEWS
        ):
            return None
    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mnnvl.mapping, mnnvl.dtype, buffer_size_bytes=need
    )
    if capturing:
        if workspace["uc_buffer"].data_ptr() not in _LAMPORT_VIEWS:
            # Another all-reduce grew the workspace after the warm-up's push here.
            return None
    else:
        _SEEN_WORKSPACES[mnnvl.mapping] = workspace
    view = _lamport_view(mnnvl, workspace)
    output = torch.empty(
        num_tokens, hidden_dim, dtype=mnnvl.dtype, device=workspace["uc_buffer"].device
    )
    return LamportPushTarget(view, one_shot, cpp_one_shot, output)


def mnnvl_workspace_mpi_comm(workspace: Dict[str, object]):
    """The MPI communicator of a shared MNNVL all-reduce workspace
    (``get_or_scale_allreduce_mnnvl_workspace``), for the package's private Lamport workspaces:
    ``McastGPUBuffer`` takes its Fortran handle. The workspace keys it ``"comm"``; under a non-MPI
    orchestrator that entry is the TP ProcessGroup, which these workspaces do not support."""
    comm = workspace.get("comm")
    if comm is None or not hasattr(comm, "py2f"):
        raise RuntimeError(
            "Kimi K3 private MNNVL workspaces need the MPI communicator of the shared MNNVL "
            "workspace (MPI orchestrator, workspace attached)"
        )
    return comm


def _mnnvl_finish(
    target: LamportPushTarget, deferred: bool = False
) -> Union[torch.Tensor, LamportPending]:
    """``MNNVLAllReduce.finish_lamport_push`` of the branch."""
    from ..ops.comm.k3_lamport_finish import k3_lamport_finish

    if deferred:
        return LamportPending(target)
    return k3_lamport_finish(
        target.workspace,
        target.output,
        target.one_shot,
        use_pdl=USE_PDL,
        cpp_one_shot=target.cpp_one_shot,
    )


def lamport_push_target(
    allreduce: Any,
    num_tokens: int,
    hidden_dim: int,
    all_reduce_params: Optional[AllReduceParams] = None,
    deferred: bool = False,
) -> Optional[LamportPushTarget]:
    """A ``LamportPushTarget`` if ``allreduce`` would run the MNNVL all-reduce on a BF16
    ``[num_tokens, hidden_dim]`` input with these parameters (plain sum, no fusion); else None.

    A producer that stores its output into the target (and sets ``pushed``) replaces the input of
    this all-reduce; :func:`finish_lamport_push` then returns the same result the all-reduce would.
    Targets must be finished in the order they were pushed, by every rank. ``deferred`` sets
    ``target.deferred`` for ``finish_lamport_push(deferred=True)``, whose consumer finishes the
    call; it takes the one-shot transport where the call is at most
    ``_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES`` (``num_tokens * hidden_dim * world * 2``), the
    two-shot one above.
    """
    mnnvl = getattr(allreduce, "mnnvl_allreduce", None)
    if (
        mnnvl is None
        or allreduce.mapping.tp_size == 1
        or getattr(allreduce, "symm_mem_allreduce", None) is not None
    ):
        return None
    if all_reduce_params is not None and (
        not all_reduce_params.enable_allreduce
        or all_reduce_params.fusion_op != AllReduceFusionOp.NONE
    ):
        return None
    if mnnvl.dtype != torch.bfloat16 or num_tokens <= 0 or hidden_dim % 8:
        return None
    if not deferred:
        return mnnvl_lamport_push_target(mnnvl, num_tokens, hidden_dim)
    one_shot = (
        num_tokens * hidden_dim * allreduce.mapping.tp_size * 2
        <= _K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES
    )
    target = mnnvl_lamport_push_target(mnnvl, num_tokens, hidden_dim, one_shot=one_shot)
    if target is not None:
        target.deferred = True
    return target


def lamport_push_tensor(allreduce: Any, target: LamportPushTarget, x: torch.Tensor) -> None:
    """Push the contiguous BF16 ``[num_tokens, hidden_dim]`` tensor ``x`` into ``target`` (a copy
    kernel), for a producer without the push epilogue."""
    del allreduce  # the target carries its workspace; kept for the branch's call shape
    from ..ops.comm.k3_lamport_push import k3_lamport_push

    k3_lamport_push(target.workspace, x.contiguous(), target.one_shot, use_pdl=USE_PDL)
    target.pushed = True


def finish_lamport_push(
    allreduce: Any,
    target: LamportPushTarget,
    deferred: bool = False,
) -> Union[torch.Tensor, LamportPending]:
    """The all-reduce result of a target the producer has pushed: a tensor, or with ``deferred``
    (where ``target.deferred``) a ``LamportPending`` that the next kernel finishes."""
    del allreduce  # the target carries its workspace; kept for the branch's call shape
    if not target.pushed:
        raise RuntimeError("the producer did not push this all-reduce")
    return _mnnvl_finish(target, deferred and target.deferred)


__all__ = [
    "LamportPending",
    "LamportPushTarget",
    "finish_lamport_push",
    "lamport_push_target",
    "lamport_push_tensor",
    "mnnvl_lamport_push_target",
]


_ATTN_RES_F = _features()

#: The PDL kernels serve up to this many rows (``TRTLLM_K3_ATTN_RES_PDL_MAX_M``).
_K3_ATTN_RES_PDL_MAX_M = _ATTN_RES_F.attn_res_pdl_max_m
#: The post-attention mix finishes the o_proj all-reduce before its ``griddepcontrol.wait``
#: (``k3_attn_res`` ``early_poll``).
_K3_ULL_OPROJ_SANDWICH = _ATTN_RES_F.ull_oproj_sandwich
#: The pre-attention mix finishes the row-parallel tail's all-reduce the same way.
_K3_ULL_TAIL_SANDWICH = _ATTN_RES_F.ull_tail_sandwich


class _DeferredRpTail(NamedTuple):
    """A layer's ``bf16(residual + all-reduce of the ranks' row-parallel tail partials)``:
    ``result`` is the pushed all-reduce, left for the next layer's pre-attention mix to finish
    (``LamportPending``)."""

    residual: torch.Tensor
    result: Any

    def materialize(self) -> torch.Tensor:
        # A BF16 add rounds the FP32 sum to nearest, as the fused mix does.
        return self.residual + self.result.materialize().view(self.residual.shape)


def _weights(proj: nn.Linear, norm: RMSNormWeight, output_norm: RMSNormWeight):
    return proj.weight.reshape(-1), norm.weight, output_norm.weight


def _outside(site: str, prefix_sum: torch.Tensor, block_residual: torch.Tensor) -> RuntimeError:
    return RuntimeError(
        f"Kimi K3 ULL {site}: {tuple(prefix_sum.shape)} {prefix_sum.dtype} with "
        f"{int(block_residual.shape[0])} snapshots is outside the served attention-residual "
        "kernels (BF16, hidden 7168, 1..12 candidates, at most "
        f"{_K3_ATTN_RES_PDL_MAX_M} rows)"
    )


def _k3_attn_res_pdl(
    prefix_sum: torch.Tensor,
    addend: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
):
    """``trtllm::k3_attn_res_(add_)rmsnorm`` where it applies, else ``None``."""
    if prefix_sum.shape[0] > _K3_ATTN_RES_PDL_MAX_M:
        return None
    weights = _weights(proj, norm, output_norm)
    addends = () if addend is None else (addend,)
    if not k3_attn_res.can_use_k3_attn_res(prefix_sum, addends, block_residual, *weights):
        return None
    return k3_attn_res_norm(
        prefix_sum, addends, block_residual, *weights, float(norm.eps), float(output_norm.eps)
    )


def _k3_attn_res_lamport_applies(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> bool:
    """Whether the post-attention mix on ``prefix_sum`` can read its addend from a Lamport
    workspace: the PDL attn_res kernel serves the ADD_PREFIX mix."""
    if prefix_sum.shape[0] > _K3_ATTN_RES_PDL_MAX_M:
        return False
    weights = _weights(proj, norm, output_norm)
    return k3_attn_res.can_use_k3_attn_res(prefix_sum, (prefix_sum,), block_residual, *weights)


def _k3_attn_res_pdl_lamport(
    prefix_sum: torch.Tensor,
    addend: LamportPending,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
    entry_trigger: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(prefix_sum + all-reduce result, normed mix)``: the mix finishes the pushed all-reduce and
    reads the result from the Lamport workspace (``trtllm::k3_attn_res_lamport_add_rmsnorm``), or
    ``None`` if that does not apply."""
    if tuple(addend.shape) != tuple(prefix_sum.shape) or not _k3_attn_res_lamport_applies(
        prefix_sum, block_residual, proj, norm, output_norm
    ):
        return None
    ws = addend.workspace
    return k3_attn_res_lamport(
        prefix_sum,
        ws.uc_local,
        ws.flags.data_ptr(),
        block_residual,
        *_weights(proj, norm, output_norm),
        float(norm.eps),
        float(output_norm.eps),
        *addend.finish_args,
        False,
        _K3_ULL_OPROJ_SANDWICH,
        entry_trigger,
    )


def _k3_attn_res_pdl_rp_tail(
    deferred: _DeferredRpTail,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
    mxfp8: bool = False,
    entry_trigger: bool = False,
    prefix_out: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """``(prefix_sum, normed mix)`` of a deferred row-parallel tail: the mix finishes the pushed
    all-reduce, reads it from the Lamport workspace and adds it to the residual with a
    round-to-nearest BF16 add; ``None`` if the PDL kernel does not serve the shape. With
    ``mxfp8``, ``(prefix_sum, normed mix, xq, xs)``. ``entry_trigger``: the kernel releases its
    dependent at entry. ``prefix_out`` (a contiguous buffer of the residual's shape outside
    ``block_residual``): the kernel stores ``prefix_sum`` there (the same bits)."""
    residual, result = deferred
    if tuple(result.shape) != tuple(residual.shape):
        return None
    weights = _weights(proj, norm, output_norm)
    if not k3_attn_res.can_use_k3_attn_res(residual, (residual,), block_residual, *weights):
        return None
    ws = result.workspace
    finish_ag = result.finish_ag
    return k3_attn_res_lamport(
        residual,
        ws.uc_local,
        ws.flags.data_ptr(),
        block_residual,
        *weights,
        float(norm.eps),
        float(output_norm.eps),
        *result.finish_args,
        True,
        _K3_ULL_TAIL_SANDWICH or finish_ag,
        entry_trigger,
        finish_ag,
        mxfp8=mxfp8,
        updated=prefix_out,
    )


def _apply_attn_res(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: RMSNormWeight
) -> torch.Tensor:
    """The pre-norm mixture (HF ``modeling_kimi._apply_attn_res``): ``trtllm::attn_res_fwd`` on
    ``prefix_sum [M, H]`` (the last candidate) and ``block_residual [K, M, H]``."""
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
        or K + 1 > 12
        or not (4096 <= H <= 8192 and H % 1024 == 0)
    ):
        raise _outside("attn_res", prefix_sum, block_residual)
    output, _rsigma, _probs, _logits = attn_res_fwd(
        prefix_sum.reshape(M, 1, H).contiguous(),
        block_residual.reshape(K, M, 1, H).contiguous(),
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
    )
    return output.reshape(M, H)


def _apply_attn_res_add_and_rmsnorm(
    prefix_sum: torch.Tensor,
    addend: Any,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
    entry_trigger: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm:
    ``(prefix_sum + addend, normed mix)``.

    ``addend`` may be a ``LamportPending``: the attention block's pushed all-reduce, which the PDL
    kernel finishes. ``entry_trigger``: release the dependent at kernel entry."""
    if isinstance(addend, LamportPending):
        fused = _k3_attn_res_pdl_lamport(
            prefix_sum, addend, block_residual, proj, norm, output_norm, entry_trigger
        )
    else:
        fused = _k3_attn_res_pdl(prefix_sum, addend, block_residual, proj, norm, output_norm)
    if fused is None:
        raise _outside("post-attention mix", prefix_sum, block_residual)
    return fused


#: Kimi-K3 hidden size (``k3_dense0.HIDDEN``).
HIDDEN = 7168
#: Largest step the dense kernels serve.
LAYER0_MAX_ROWS = 32


class KdaOutputProjection(nn.Module):
    """Layer 0's KDA o_proj and its TP16 all-reduce (pushed by the GEMM where it can).

    ``o_proj``: this rank's ``[hidden_size, proj_size]`` FP8 shard (input columns
    ``proj_size * rank`` on) of the ``[hidden_size, proj_size * tp]`` checkpoint projection, swapped
    for the FP8 weight-read GEMM after loading. ``proj_size``: this rank's heads x head dim."""

    def __init__(self, hidden_size: int, proj_size: int, mapping, allreduce_strategy) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self._kda_tp_size = mapping.tp_size
        self._kda_tp_rank = mapping.tp_rank
        self._o_allreduce = TpAllReduce(mapping, torch.bfloat16, allreduce_strategy)
        self.o_proj = Fp8BlockScaleLinear(
            proj_size * mapping.tp_size,
            hidden_size,
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            tp_mode=TensorParallelMode.ROW,
        )

    def forward(self, core: torch.Tensor) -> torch.Tensor:
        """o_proj of the core rows ``[rows, proj_size]`` and the all-reduce of its partials."""
        rows = core.shape[0]
        ar_push = None
        if hasattr(self.o_proj, "next_ar_push"):
            ar_push = lamport_push_target(self._o_allreduce, rows, self.hidden_size)
            self.o_proj.next_ar_push = ar_push
        try:
            out = self.o_proj(core.reshape(rows, self.proj_size))
        finally:
            if ar_push is not None:
                self.o_proj.next_ar_push = None
        if ar_push is not None and ar_push.pushed:
            return finish_lamport_push(self._o_allreduce, ar_push).view(rows, self.hidden_size)
        return self._o_allreduce(out)


def dense_tp_size(intermediate_size: int, world: int, gpus_per_node: int) -> int:
    """The TP size of the dense MLP's shards whose sub-slices the ranks take (module docstring)."""
    tp = math.gcd(intermediate_size, world)
    if tp > gpus_per_node:
        tp = math.gcd(tp, gpus_per_node)
    return tp


def dense_slice_index(rank: int, world: int, tp: int) -> int:
    """Rank ``rank``'s slice of the ``world``-way split of the intermediate (module docstring)."""
    return (rank % tp) * (world // tp) + rank // tp


class Dense0(nn.Module):
    """Layer 0's post-attention mix and dense MLP on this rank's ``1 / world`` slice.

    Weights in the kernels' layout (``k3_dense0.slice_weights``): ``gate_up`` ``[2 I_s, H]`` in
    64-row tiles of 32 gate then 32 up rows, ``down`` ``[H, pad_k(I_s)]`` with zero padding
    columns; ``hidden`` the ``[MAX_ROWS, pad_k(I_s)]`` BF16 buffer of the SiTU output (padding
    columns stay zero); ``grid_out`` the ``[MAX_ROWS, 2 I_s]`` launch shape."""

    def __init__(
        self,
        intermediate_size: int,
        *,
        rank: int,
        world: int,
        gpus_per_node: int,
        beta: float,
        linear_beta: Optional[float],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        from ..kernels.k3r_gemm import k3_dense0

        if intermediate_size % world:
            raise ValueError(f"dense intermediate {intermediate_size} does not split {world} ways")
        self.intermediate_size = intermediate_size
        self.world = world
        self.tp = dense_tp_size(intermediate_size, world, gpus_per_node)
        self.index = dense_slice_index(rank, world, self.tp)
        self.inter = intermediate_size // world
        padded = k3_dense0.pad_k(self.inter)
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)
        self.gate_up = nn.Parameter(
            torch.empty(2 * self.inter, HIDDEN, dtype=dtype), requires_grad=False
        )
        self.down = nn.Parameter(torch.zeros(HIDDEN, padded, dtype=dtype), requires_grad=False)
        self.register_buffer(
            "hidden", torch.zeros(LAYER0_MAX_ROWS, padded, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "grid_out", torch.empty(LAYER0_MAX_ROWS, 2 * self.inter, dtype=dtype), persistent=False
        )

    def load(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
        """This rank's slice (:attr:`index`) of the checkpoint projections: ``gate`` / ``up``
        ``[I_s, H]`` rows and ``down`` ``[H, I_s]`` columns."""
        from ..kernels.k3r_gemm import k3_dense0

        inter = self.inter
        if gate.shape != (inter, HIDDEN) or up.shape != gate.shape or down.shape != (HIDDEN, inter):
            raise ValueError(
                f"Kimi K3 ULL dense slice: gate {tuple(gate.shape)} / up {tuple(up.shape)} / "
                f"down {tuple(down.shape)}, expected [{inter}, {HIDDEN}] / [{HIDDEN}, {inter}]"
            )
        device, dtype = self.gate_up.device, self.gate_up.dtype
        with torch.no_grad():
            self.gate_up.data.copy_(
                k3_dense0.fuse_gate_up(gate.to(device, dtype), up.to(device, dtype))
            )
            self.down.data.zero_()
            self.down.data[:, :inter].copy_(down.to(device, dtype))

    def forward(
        self,
        attn_out: torch.Tensor,
        valid: torch.Tensor,
        layer: nn.Module,
        allreduce: TpAllReduce,
    ) -> _DeferredRpTail:
        """The post-attention mix of layer 0 (the attention output starts the prefix: layer 0 is
        a block boundary) and the dense MLP pushed into a deferred all-reduce on ``allreduce``'s
        workspace. ``layer``: layer 0's ``mlp_res_proj`` / ``mlp_res_norm`` /
        ``post_attention_layernorm`` (``.weight`` / ``.eps``)."""
        from ..kernels.k3r_attn_res import k3_attn_res_ull as k3_attn_res
        from ..kernels.k3r_gemm import k3_dense0

        rows = attn_out.shape[0]
        if not 1 <= rows <= LAYER0_MAX_ROWS:
            raise RuntimeError(f"Kimi K3 ULL layer 0: {rows} rows (1..{LAYER0_MAX_ROWS})")
        gu_tactic, down_tactic = k3_dense0.default_tactics(rows)
        k3_dense0.validate_tactics(rows, self.inter, gu_tactic, down_tactic)
        target = lamport_push_target(allreduce, rows, HIDDEN, deferred=True)
        if target is None or target.workspace.world != self.world:
            raise RuntimeError("Kimi K3 ULL layer 0: no pushed TP16 all-reduce for the dense MLP")

        prefix = attn_out
        weights = (
            layer.mlp_res_proj.weight.reshape(-1),
            layer.mlp_res_norm.weight,
            layer.post_attention_layernorm.weight,
        )
        if not k3_attn_res.can_use_k3_attn_res(prefix, (), valid, *weights):
            raise RuntimeError(
                f"Kimi K3 ULL layer 0: the post-attention mix of {tuple(prefix.shape)} with "
                f"{int(valid.shape[0])} snapshots is outside the k3_attn_res kernel"
            )
        # The fused mix, released at kernel entry: the gate/up GEMM after it streams its weights
        # while the mix runs.
        x = torch.empty_like(prefix)
        k3_attn_res.run_k3_attn_res(
            prefix,
            (),
            valid,
            *weights,
            float(layer.mlp_res_norm.eps),
            float(layer.post_attention_layernorm.eps),
            x,
            None,
            entry_trigger=True,
        )
        hidden = self.hidden[:rows]
        k3_dense0.run_situ_gate_up(
            x,
            self.gate_up,
            hidden,
            self.grid_out[:rows],
            gu_tactic,
            self.beta,
            self.linear_beta,
        )
        k3_dense0.run_down_push(hidden, self.down, target, down_tactic)
        pending = finish_lamport_push(allreduce, target, deferred=True)
        return _DeferredRpTail(prefix, pending)

    def reference(
        self,
        attn_out: torch.Tensor,
        valid: torch.Tensor,
        layer: nn.Module,
        allreduce: TpAllReduce,
    ) -> _DeferredRpTail:
        """:meth:`forward` in plain PyTorch on the same slice (FP32 GEMM accumulation, the SiTU in
        FP32), pushed into the same deferred all-reduce: for tests, not bitwise the kernels."""
        prefix = attn_out
        x = k3_attn_res_norm.reference(
            prefix,
            (),
            valid,
            layer.mlp_res_proj.weight.reshape(-1),
            layer.mlp_res_norm.weight,
            layer.post_attention_layernorm.weight,
            float(layer.mlp_res_norm.eps),
            float(layer.post_attention_layernorm.eps),
        )
        rows = x.shape[0]
        gate_up = self.gate_up.view(-1, 2, 32, HIDDEN)
        gate = gate_up[:, 0].reshape(-1, HIDDEN)
        up = gate_up[:, 1].reshape(-1, HIDDEN)
        g = (x.float() @ gate.float().t()).to(x.dtype).float()
        u = (x.float() @ up.float().t()).to(x.dtype).float()
        beta = self.beta
        h = beta * torch.tanh(g / beta) * torch.sigmoid(g)
        if self.linear_beta is not None:
            u = self.linear_beta * torch.tanh(u / self.linear_beta)
        h = (h * u).to(x.dtype)
        partial = (h.float() @ self.down[:, : self.inter].float().t()).to(x.dtype)
        target = lamport_push_target(allreduce, rows, HIDDEN, deferred=True)
        if target is None:
            raise RuntimeError("Kimi K3 ULL layer 0: no pushed TP16 all-reduce for the dense MLP")
        lamport_push_tensor(allreduce, target, partial)
        return _DeferredRpTail(prefix, finish_lamport_push(allreduce, target, deferred=True))


__all__ = [
    "Dense0",
    "KdaOutputProjection",
    "dense_slice_index",
    "dense_tp_size",
]


#: Largest step (token rows) the fold serves: the fused query's limit.
MLAIN_MAX_ROWS = 16
#: Weight rows per tile of the low-M GEMM.
_TILE = 128
#: Weight-scale word of the zero padding rows (four UE8M0 exponents 127: scale 1).
_UNIT_SCALE_WORD = 0x7F7F7F7F


class FoldedInProj:
    """``[q_a | kv_a | 0 | g]``: the fused q_a / kv_a projection's FP8 weight, zero rows up to the
    next whole 128-row tile, then the output gate's FP8 weight, with their DeepGEMM-packed UE8M0
    scales (``packed_weight_scale``, int32 ``[N, K / 512]`` with stride ``(1, N)``). Built once
    per layer outside CUDA graph capture (``build``); ``sources`` are the tensors it was built from
    (rebuilt where the projections' weights were replaced, :meth:`is_current`)."""

    def __init__(self, kv_a: torch.nn.Module, g_proj: torch.nn.Module):
        w_a, w_g = kv_a.weight, g_proj.weight
        s_a, s_g = kv_a.packed_weight_scale, g_proj.packed_weight_scale
        self.sources = (w_a, w_g, s_a, s_g)
        if not foldable(w_a, w_g, s_a, s_g):
            raise ValueError("the q_a / kv_a and gate weights do not fold into one low-M GEMM")
        n_a, k = w_a.shape
        self.n_qkv = n_a
        self.pad_rows = -(-n_a // _TILE) * _TILE
        self.n_gate = w_g.shape[0]
        self.sigmoid_tile0 = self.pad_rows // _TILE
        n = self.pad_rows + self.n_gate
        with torch.no_grad():
            weight = torch.zeros((n, k), dtype=w_a.dtype, device=w_a.device)
            weight[:n_a].copy_(w_a)
            weight[self.pad_rows :].copy_(w_g)
            cols = torch.empty((s_a.shape[1], n), dtype=torch.int32, device=s_a.device)
            cols[:, :n_a].copy_(s_a.t())
            cols[:, n_a : self.pad_rows].fill_(_UNIT_SCALE_WORD)
            cols[:, self.pad_rows :].copy_(s_g.t())
        self.weight = weight
        self.scale = cols.t()

    @classmethod
    def build(cls, kv_a: torch.nn.Module, g_proj: Optional[torch.nn.Module]):
        """The fold of these projections, or ``None`` where it does not apply."""
        if g_proj is None or torch.cuda.is_current_stream_capturing():
            return None
        if not all(hasattr(p, "packed_weight_scale") for p in (kv_a, g_proj)):
            return None
        try:
            return cls(kv_a, g_proj)
        except ValueError:
            return None

    def is_current(self, kv_a: torch.nn.Module, g_proj: torch.nn.Module) -> bool:
        """Whether the projections still hold the tensors this fold was built from."""
        now = (kv_a.weight, g_proj.weight, kv_a.packed_weight_scale, g_proj.packed_weight_scale)
        return all(a is b for a, b in zip(now, self.sources))

    def tactic(self, rows: int, staged: bool):
        """The MXFP8 low-M tactic on ``rows`` rows (``staged``: the weight ring sized for staging,
        ``staged_mx_tactic``), or ``None`` where none fits."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8_ull as mx

        n, k = self.weight.shape
        try:
            return mx.staged_mx_tactic(rows, n, k) if staged else mx.default_mx_tactic(rows, n, k)
        except ValueError:
            return None

    def on_chip(self, rows: int) -> bool:
        """Whether the staged tactic keeps the CTA's whole weight slice on chip before its wait
        (the producer before the projection may then release it at kernel entry)."""
        t = self.tactic(rows, True)
        if t is None:
            return False
        k_tiles = -(-(self.weight.shape[1] // _TILE) // t.split_k)
        return t.a_stages + t.tmem_tiles >= k_tiles

    def run(self, xq: torch.Tensor, xs: torch.Tensor, tactic) -> torch.Tensor:
        """The folded projection on the MXFP8 rows ``(xq, xs)``: BF16 ``[M, N]`` (module docstring
        layout)."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8_ull as mx

        out = torch.empty(
            (xq.shape[0], self.weight.shape[0]), dtype=torch.bfloat16, device=xq.device
        )
        mx.run_k3_fp8w_lowm_gemm_mxfp8(
            xq, xs, self.weight, self.scale, out, tactic=tactic, sigmoid_tile0=self.sigmoid_tile0
        )
        return out


def foldable(w_a, w_g, s_a, s_g) -> bool:
    """Whether the q_a / kv_a weight ``w_a`` and the gate weight ``w_g`` with their packed scales
    fold into one low-M GEMM weight."""
    if not all(isinstance(t, torch.Tensor) for t in (w_a, w_g, s_a, s_g)):
        return False
    if w_a.dtype != torch.float8_e4m3fn or w_g.dtype != w_a.dtype or w_a.dim() != 2:
        return False
    if w_g.dim() != 2 or w_g.shape[1] != w_a.shape[1] or w_g.shape[0] % _TILE:
        return False
    if not (w_a.is_contiguous() and w_g.is_contiguous()):
        return False
    words = -(-(w_a.shape[1] // _TILE) // 4)
    for s, w in ((s_a, w_a), (s_g, w_g)):
        if s.dtype != torch.int32 or tuple(s.shape) != (w.shape[0], words) or s.stride(0) != 1:
            return False
    return True


__all__ = ["FoldedInProj", "MLAIN_MAX_ROWS", "foldable"]


# Head dimensions (K-nope, K-pe, V) the split-K context FMHA is built for.
_SPLIT_K_HEAD_DIMS = (128, 64, 128)


# Batches of at most this many tokens (helix n4 batches included) start the tail after the decode
# instead of at the decode's entry (the decode's completion releases it).
_LATE_TAIL_MAX_TOKENS = 32


def decode_releases_tail_early(num_tokens: int) -> bool:
    """Whether the FP4 decode before the fused tail may release it at the decode's entry
    (``DeferredPackedReduce.release_early``; it does with ``features.fp4_mla_reduce_pdl``, on).

    With ``features.ull2_mla_tail`` and at most :data:`_LATE_TAIL_MAX_TOKENS` tokens it does not:
    a tail grid that is resident during the whole decode folds and projects 2-4 us slower once
    the decode completes than one launched at its end, more than the early launch hides. Larger
    batches launch most of their tail CTAs at the decode's end anyway and gain from the others.
    """
    return not (ull2_tail_enabled() and num_tokens <= _LATE_TAIL_MAX_TOKENS)


def fold_inputs_of(deferred, rows: torch.Tensor, stats: torch.Tensor):
    """``FoldInputs`` of the reduction a ``DeferredPackedReduce`` recorded for this decode (its
    output must be ``rows`` and its statistics ``stats``), or None if it recorded none. A recorded
    call on other buffers is launched instead (the caller then reads ``rows`` and ``stats``)."""
    from ..kernels.k3r_helix.k3_helix_tail_ull import FoldInputs

    args = deferred.args
    if args is None:
        return None
    own_buffers = (
        args["output"].data_ptr() == rows.data_ptr()
        and args["output"].numel() == rows.numel()
        and args["stats"] is not None
        and args["stats"].data_ptr() == stats.data_ptr()
    )
    if not own_buffers:
        deferred.run()
        return None
    return FoldInputs(
        args["partial"],
        args["maxima"],
        args["sums"],
        args["plan"],
        args["q_scale"],
        args["kv_scale"],
        num_requests=args["num_requests"],
        valid_m=args["valid_m"],
        physical_m=args["physical_m"],
        sm_scale=args["sm_scale"],
        output_scale=args["output_scale"],
    )


_gen_F = _features()

# Largest generation batch (tokens) the fused query serves (the kernel takes up to 16).
_Q_FUSED_MAX_TOKENS = 16


class HelixGenHooks:
    """What the side-gate forward (``K3MLAAttention._forward_side_gate``) hands the Helix
    generation chain for one forward, and what the chain hands back.

    ``zero_kv_mask_fn`` returns the zero-local-KV mask computed ahead on the side stream (called
    where the exchange needs it); ``pre_exchange_fn`` forks the output gate; ``pre_tail_fn`` joins
    it before the fused tail and returns it (None if not forked yet); ``gate_place`` is the
    ``features.ull3_mla_gate_at`` placement. Back: ``tail_mx``, the MXFP8 o_proj operand
    ``(xq, xs)`` the fused tail stored, and ``gen_output``, the combined all-to-all output."""

    __slots__ = (
        "zero_kv_mask_fn",
        "pre_exchange_fn",
        "pre_tail_fn",
        "gate_place",
        "tail_mx",
        "gen_output",
    )

    def __init__(
        self,
        zero_kv_mask_fn: Callable[[], Optional[torch.Tensor]],
        pre_exchange_fn: Callable[[], None],
        pre_tail_fn: Callable[[], Optional[torch.Tensor]],
        gate_place: str,
    ) -> None:
        self.zero_kv_mask_fn = zero_kv_mask_fn
        self.pre_exchange_fn = pre_exchange_fn
        self.pre_tail_fn = pre_tail_fn
        self.gate_place = gate_place
        self.tail_mx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.gen_output: Optional[torch.Tensor] = None


def helix_gen_zero_kv_mask(md: AttentionMetadata, num_tokens: int) -> Optional[torch.Tensor]:
    """Per-token mask of the ``num_tokens`` generation tokens this CP rank owns no KV for.

    With valid speculative verify groups (the deployment's steps) KV ownership is per token
    (``helix_kv_bounds``); otherwise the per-sequence mask of the generation requests
    (``_helix_zero_kv_mask``). None without those buffers."""
    helix_kv_bounds = getattr(md, "helix_kv_bounds", None)
    if helix_kv_bounds is not None and getattr(md, "_helix_spec_tokens_valid", False):
        return helix_kv_bounds[:num_tokens] == 0
    return _helix_zero_kv_mask(
        md, num_tokens, seq_start=md.num_contexts, num_seqs=md.num_generations
    )


def helix_exchange(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    mapping,
    num_heads_tp_cp: int,
    value_dim: int,
    zero_kv_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """All-to-all of this rank's partials over the CP group (fifo v2 on the MNNVL workspace; the
    sender zeroes the ``zero_kv_mask`` rows) and the combine: ``[T, num_heads_tp_cp * value_dim]``
    BF16."""
    num_tokens = partial_o.shape[0]
    cp_size = mapping.cp_size
    partial_o_out, stats_out = HelixAllToAllNative.get(mapping).alltoall_native(
        partial_o.view(num_tokens, cp_size, num_heads_tp_cp * value_dim),
        softmax_stats.view(num_tokens, cp_size, num_heads_tp_cp * 2),
        None if zero_kv_mask is None else zero_kv_mask[:num_tokens],
    )
    return torch.ops.trtllm.helix_post_process_native(
        partial_o_out.view(num_tokens, cp_size, num_heads_tp_cp, value_dim),
        stats_out.view(num_tokens, cp_size, num_heads_tp_cp, 2),
        1.0,
        1,
    )


def project_helix_v_partials(layer, partial_o: torch.Tensor) -> torch.Tensor:
    """Every tp-local head's V up-projection of this CP rank's partials (it commutes with the
    per-head weighted combine): ``[T, H * kv_lora_rank]`` -> ``[T, H * v_head_dim]``."""
    num_tokens = partial_o.shape[0]
    heads = layer.num_heads_tp
    partial_v = partial_o.new_empty((num_tokens, heads, layer.v_head_dim))
    bmm_out(
        partial_o.view(num_tokens, heads, layer.kv_lora_rank).transpose(0, 1),
        layer.helix_v_b_proj_full.transpose(1, 2),
        partial_v.transpose(0, 1),
    )
    return partial_v.view(num_tokens, heads * layer.v_head_dim)


def tail_output_buffer(
    buf: Optional[torch.Tensor], num_tokens: int, cols: int
) -> Optional[torch.Tensor]:
    """``buf`` as the fused tail's BF16 ``[T, cols]`` output, or None where the tail cannot write
    it as is."""
    if (
        buf is None
        or buf.dtype != torch.bfloat16
        or not buf.is_contiguous()
        or buf.numel() != num_tokens * cols
    ):
        return None
    return buf.view(num_tokens, cols)


def helix_tail_applies(layer, num_tokens: int) -> bool:
    """Whether the Helix tail of this batch is one ``k3_helix_tail`` kernel (``helix_tail``)."""
    w_uv = layer.helix_v_b_proj_full
    return (
        isinstance(w_uv, torch.Tensor)
        and w_uv.dtype == torch.bfloat16
        and w_uv.is_contiguous()
        and K3HelixTail.applies(
            layer.mapping, num_tokens, layer.num_heads_tp, layer.kv_lora_rank, layer.v_head_dim
        )
    )


class QuantQuery(NamedTuple):
    """The quantized query the FP8-KV cache update wrote for the FP8 decode: ``buffer`` (uint8
    ``[T, H, 576]``, FP8 E4M3 codes) and the device BMM scales ``bmm1`` (``[2]``, element 0: the
    softmax scale) and ``bmm2`` (``[1]``, the output scale)."""

    buffer: torch.Tensor
    bmm1: torch.Tensor
    bmm2: torch.Tensor


def gen_attention(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    softmax_stats: torch.Tensor,
    quant_q: Optional[QuantQuery] = None,
) -> torch.Tensor:
    """The absorbed decode of ``q`` (``[T, H * 576]``; on the quantized caches only its shape is
    read): ``[T, H * 512]`` BF16 rows and their Helix ``softmax_stats``. K3's packed FP4 decode on an
    FP4 cache (served: ``TrtllmAttention.forward`` -> ``Fp4MlaFmha``), the CuTe DSL FP8 decode on
    an FP8 cache (served: ``TrtllmAttention.forward`` -> ``CuteDslMlaFmha``)."""
    if fp4_mla_decode.applies(layer.mqa):
        return _decode_op.k3_fp4_mla_decode(layer.mqa, q, md, softmax_stats_tensor=softmax_stats)
    if layer.mqa.has_fp8_kv_cache and quant_q is not None:
        return fp8_mla_decode(layer, q, md, softmax_stats, quant_q)
    raise NotImplementedError(
        f"Kimi K3 ULL MLA layer {layer.layer_idx}: the generation decode runs on an FP4 cache "
        "(CuTe DSL backend) or an FP8 cache only"
    )


def fp8_mla_decode(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    softmax_stats: torch.Tensor,
    quant_q: QuantQuery,
) -> torch.Tensor:
    """The CuTe DSL FP8 MLA decode of an FP8-KV layer (served ``CuteDslMlaFmha._run_mla_decode``
    and ``_split_kv_plan`` with ``PhasedFmha.forward``'s generation parameters): the per-request
    split plan once per step (``_mla_split_kv_plan_kernel``), then the decode (split +
    reduction) on the FP8 query ``quant_q`` over the layer's page pool. Returns ``[T, H * 512]``."""
    mqa = layer.mqa
    num_tokens = q.shape[0]
    num_heads = mqa.num_heads
    d_latent, d_rope = mqa.kv_lora_rank, mqa.qk_rope_head_dim
    batch_size = md.num_generations
    seq_len_q = num_tokens // batch_size
    page_size = md.tokens_per_block if md.tokens_per_block is not None else 64
    output = q.new_empty((num_tokens, num_heads * d_latent))

    # The workspace this kernel owns at the front of the attention workspace (sized here eagerly,
    # as the served FMHA's prepare_workspace).
    workspace = md.effective_workspace
    required = _fp8_decode.workspace_bytes(num_heads, seq_len_q, d_latent, md.max_num_requests)
    if workspace.numel() * workspace.element_size() < required:
        if md.is_cuda_graph and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Attention CUDA graph workspace is smaller than the required size for the CuTe DSL "
                "MLA decode."
            )
        workspace.resize_((math.ceil(required / workspace.element_size()),))
    # A non-reclaimable FMHA library ran on this metadata (the served dispatch's side effect).
    md.workspace_reclaimable = False
    workspace = workspace[: math.ceil(required / workspace.element_size())]

    q_view = quant_q.buffer.view(torch.float8_e4m3fn).view(
        batch_size, seq_len_q, num_heads, d_latent + d_rope
    )
    kv_pool = md.kv_cache_manager.get_buffers(mqa.layer_idx)
    # KVCacheManagerV1 exposes a per-layer view over one interleaved pool (dim 0 strides by all
    # local layers): address it as a packed combined-slot pool and fold this layer's slot into the
    # page table. V2 exposes each layer as a densely packed pool.
    packed_block = math.prod(kv_pool.shape[1:])
    block_stride = kv_pool.stride(0)
    layers_in_pool = block_stride // packed_block if packed_block else 1
    layer_in_pool = 0
    if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
        layer_in_pool = kv_pool.storage_offset() // packed_block
        kv_pool = kv_pool.as_strided(
            (kv_pool.shape[0] * layers_in_pool, *kv_pool.shape[1:]),
            (packed_block, *kv_pool.stride()[1:]),
            0,
        )
    else:
        layers_in_pool = 1
    kv_pool = kv_pool.view(torch.float8_e4m3fn)
    if kv_pool.dim() != 5 or kv_pool.shape[1] != 1 or kv_pool.shape[3] != 1:
        raise RuntimeError(
            "The CuTe DSL MLA decode expects KV cache layout [num_pages, 1, page_size, 1, "
            f"head_dim], got {tuple(kv_pool.shape)}."
        )
    if kv_pool.shape[2] != page_size or kv_pool.shape[-1] < d_latent + d_rope:
        raise RuntimeError(
            f"The CuTe DSL MLA decode got KV cache shape {tuple(kv_pool.shape)} for "
            f"page_size={page_size}, kv_lora_rank={d_latent}, qk_rope_head_dim={d_rope}."
        )
    pool_idx = int(md.host_kv_cache_pool_mapping[mqa.get_local_layer_idx(md), 0])
    page_table = (
        md.kv_cache_block_offsets[pool_idx, :, 0, :][md.num_contexts : md.num_contexts + batch_size]
        .transpose(0, 1)
        .to(torch.int32)
    )
    if layers_in_pool > 1:
        page_table = page_table + layer_in_pool
    cache_seqs = md.kv_lens_cuda_runtime[md.num_contexts :].to(torch.int32)
    kv_pages = kv_pool[:, 0, :, 0, : d_latent + d_rope]

    block_split_kvs = fp8_mla_split_plan(md, cache_seqs, batch_size, seq_len_q, num_heads)
    kv_bounds = None
    if md.helix_position_offsets is not None and md._helix_spec_tokens_valid:
        kv_bounds = md.helix_kv_bounds[:num_tokens]
    _fp8_decode.k3_mla_decode_fp8(
        q_view[..., :d_latent].permute(2, 3, 1, 0),
        q_view[..., d_latent:].permute(2, 3, 1, 0),
        kv_pages[..., :d_latent].permute(1, 2, 0),
        kv_pages[..., d_latent:].permute(1, 2, 0),
        page_table,
        cache_seqs,
        output.view(batch_size, seq_len_q, num_heads, d_latent).permute(2, 3, 1, 0),
        workspace,
        num_heads,
        seq_len_q,
        page_size,
        quant_q.bmm1,
        quant_q.bmm2,
        int(md.max_num_requests),
        softmax_stats,
        kv_bounds,
        block_split_kvs,
    )
    return output


# The split plan covers at most this many requests (served ``_SPLIT_PLAN_MAX_REQUESTS``).
_FP8_SPLIT_PLAN_MAX_REQUESTS = 1024


def fp8_mla_split_plan(
    md: AttentionMetadata,
    cache_seqs: torch.Tensor,
    batch_size: int,
    seq_len_q: int,
    num_heads: int,
) -> Optional[torch.Tensor]:
    """This step's per-request split-KV counts for the FP8 decode, planned once per step (served
    ``CuteDslMlaFmha._split_kv_plan``): the first FP8 layer of a step launches the plan, the others
    reuse it (the metadata resets the key when the KV lengths change). None (uniform split) where it
    cannot be planned here."""
    if not 0 < batch_size <= _FP8_SPLIT_PLAN_MAX_REQUESTS:
        return None
    plan = getattr(md, "_cute_dsl_mla_split_plan", None)
    capacity = max(int(getattr(md, "max_num_requests", 0) or 0), batch_size)
    if plan is None or plan.numel() < batch_size or plan.device != cache_seqs.device:
        if torch.cuda.is_current_stream_capturing():
            return None
        plan = torch.empty(capacity, dtype=torch.int32, device=cache_seqs.device)
        md._cute_dsl_mla_split_plan = plan
    rows_per_request, wave_clusters = _fp8_decode.split_plan_geometry(num_heads, seq_len_q)
    key = (batch_size, rows_per_request, wave_clusters, cache_seqs.data_ptr(), plan.data_ptr())
    if getattr(md, "_cute_dsl_mla_split_plan_key", None) != key:
        _fp8_decode.launch_split_plan(
            cache_seqs,
            plan,
            num_requests=batch_size,
            rows_per_request=rows_per_request,
            wave_clusters=wave_clusters,
        )
        md._cute_dsl_mla_split_plan_key = key
    return plan[:batch_size]


def _helix_gen(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    helix_v_up: bool,
    hooks: Optional[HelixGenHooks],
    quant_q: Optional[QuantQuery],
) -> torch.Tensor:
    """Decode with softmax statistics, (V up-projection,) exchange and combine (served
    ``_attn_forward_gen_helix_hooked``; a batch the fused tail does not serve)."""
    softmax_stats = torch.empty(
        (q.shape[0], layer.num_heads_tp, 2), device=q.device, dtype=torch.float32
    )
    partial_o = gen_attention(layer, q, md, softmax_stats, quant_q)
    assert partial_o.shape[-1] == layer.num_heads_tp * layer.kv_lora_rank
    if hooks is not None:
        zero_kv_mask = hooks.zero_kv_mask_fn()
    else:
        zero_kv_mask = helix_gen_zero_kv_mask(md, partial_o.shape[0])
    value_dim = layer.kv_lora_rank
    if helix_v_up:
        partial_o = project_helix_v_partials(layer, partial_o)
        value_dim = layer.v_head_dim
    if hooks is not None:
        hooks.pre_exchange_fn()
    return helix_exchange(
        partial_o, softmax_stats, layer.mapping, layer.num_heads_tp_cp, value_dim, zero_kv_mask
    )


def _helix_gen_fused_tail(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    output: Optional[torch.Tensor],
    hooks: Optional[HelixGenHooks],
    quant_q: Optional[QuantQuery],
) -> torch.Tensor:
    """The decode with its split reduction deferred, then one tail kernel (fold, V
    up-projection, exchange, combine) writing ``output`` where it can (served
    ``_attn_forward_gen_fused_tail``). With the side gate forked early (gate place other than
    "exchange") the tail also stores o_proj's MXFP8 operand ``bf16(out * gate)`` where o_proj's
    gated MXFP8 GEMM takes it. The FP8 decode records no reduction: the tail reads its rows."""
    from ..kernels.k3r_fp4_mla.fp4_mla_packed_reduce_ull import DeferredPackedReduce

    num_tokens = q.shape[0]
    softmax_stats = torch.empty(
        (num_tokens, layer.num_heads_tp, 2), device=q.device, dtype=torch.float32
    )
    gate_early = hooks is not None and hooks.gate_place != "exchange"
    if gate_early:
        hooks.pre_exchange_fn()
    with DeferredPackedReduce(release_early=decode_releases_tail_early(num_tokens)) as deferred:
        partial_o = gen_attention(layer, q, md, softmax_stats, quant_q)
    assert partial_o.shape[-1] == layer.num_heads_tp * layer.kv_lora_rank
    fold = fold_inputs_of(deferred, partial_o, softmax_stats)
    if hooks is not None:
        zero_kv_mask = hooks.zero_kv_mask_fn()
    else:
        zero_kv_mask = helix_gen_zero_kv_mask(md, num_tokens)
    gate = None
    if hooks is not None:
        if gate_early:
            gate = hooks.pre_tail_fn()
        else:
            hooks.pre_exchange_fn()
    cols = layer.num_heads_tp_cp * layer.v_head_dim
    out = tail_output_buffer(output, num_tokens, cols)
    mx_out = None
    mx_applies = getattr(layer.o_proj, "gated_mxfp8_applies", None)
    if gate is not None and mx_applies is not None:
        if out is None:
            out = partial_o.new_empty((num_tokens, cols))
        if mx_applies(out, gate):
            xq = torch.empty((num_tokens, cols), dtype=torch.float8_e4m3fn, device=q.device)
            xs = torch.empty((num_tokens, cols // 32), dtype=torch.uint8, device=q.device)
            mx_out = (gate, xq, xs)
            hooks.tail_mx = (xq, xs)
    return K3HelixTail.get(layer.mapping)(
        partial_o,
        softmax_stats,
        layer.helix_v_b_proj_full,
        zero_kv_mask,
        fold=fold,
        out=out,
        mx_out=mx_out,
    )


def attn_forward_gen(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    *,
    helix_v_up: bool,
    output: Optional[torch.Tensor] = None,
    hooks: Optional[HelixGenHooks] = None,
    quant_q: Optional[QuantQuery] = None,
) -> torch.Tensor:
    """The absorbed decode of ``q`` (``[T, H * 576]``) with the Helix exchange (served
    ``KimiK3MLAAttention._attn_forward_gen``).

    Returns ``[T, H / P * v_head_dim]`` with the V up-projection before the exchange
    (``helix_v_up``), ``[T, H / P * kv_lora_rank]`` otherwise. ``output``: the layer's buffer for
    the combined rows, which the fused tail writes."""
    num_tokens = q.shape[0]
    if not layer.mapping.has_cp_helix():
        raise NotImplementedError("Kimi K3 ULL MLA runs under Helix CP only")
    if helix_v_up and helix_tail_applies(layer, num_tokens):
        out = _helix_gen_fused_tail(layer, q, md, output, hooks, quant_q)
    else:
        out = _helix_gen(layer, q, md, helix_v_up, hooks, quant_q)
    if helix_v_up and hooks is not None:
        hooks.gen_output = out
    return out


def forward_absorption_generation(
    layer,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
    hooks: Optional[HelixGenHooks] = None,
) -> torch.Tensor:
    """Absorbed generation of the ``q`` rows (``[T, H * 192]``, q_b's output) into ``output``
    (``[T, H / P * v_head_dim]``); ``latent_cache`` ``[T, 576]`` is the normalized latent and
    K-pe the cache update appends (served base ``MLA.forward_absorption_generation``). W_UK, then
    on the same stream the cache update that also quantizes the query for the decode: on an FP4
    cache ``_fp4_mla_cache_update.rope_generation`` (served ``TrtllmAttention.mla_rope_generation``
    -> ``_fp4_mla_rope_generation``), on an FP8 cache ``k3_mla_cache.mla_rope_generation``
    (served ``TrtllmAttention.mla_rope_generation``: ``applyMLARopeAndAssignQKVKernelGeneration``,
    which also writes the BMM scales)."""
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

    def rope_and_cache_update():
        if layer.fp4_kv:
            fp4_mla_cache_update.rope_generation(layer.mqa, fused_q, q_pe, latent_cache, md)
        else:
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

    helix_v_up = layer.helix_v_up_before_exchange
    quant_q = None
    if layer.quantized_kv and not layer.fp4_kv:
        quant_q = QuantQuery(quant_q_buffer, mla_bmm1_scale, mla_bmm2_scale)
    attn_out_latent = attn_forward_gen(
        layer,
        fused_q,
        md,
        helix_v_up=helix_v_up,
        output=output if helix_v_up else None,
        hooks=hooks,
        quant_q=quant_q,
    )
    fused_q = None
    if helix_v_up:
        # The combine produced [T, H / P * v_head_dim]; the fused tail wrote it into output.
        if attn_out_latent.data_ptr() != output.data_ptr():
            output.view(attn_out_latent.shape).copy_(attn_out_latent)
        return output

    heads_out = layer.num_heads_tp_cp
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
# Fused query: generation-only Helix batches of FP4-KV layers.
# ---------------------------------------------------------------------------------------------


def _flashinfer_cute_norm(device: torch.device) -> bool:
    """Whether FlashInfer's ``rmsnorm`` runs its CuTe DSL ``RMSNormKernel`` on ``device`` (the
    arithmetic ``k3_mla_latent_row`` and ``k3_mla_q_fused`` reproduce), not its CUDA JIT kernel."""
    try:
        from flashinfer.norm import _use_cuda_norm
    except ImportError:
        return False
    return not _use_cuda_norm(device)


def q_fused_w_uk(layer) -> Optional[torch.Tensor]:
    """The BF16 absorption weight ``[H, kv_lora_rank, qk_nope_head_dim]`` the W_UK BMM reads (the
    weight, or its dequantized copy)."""
    w_kc = layer.k_b_proj_trans
    if w_kc is not None and w_kc.dtype != torch.bfloat16:
        w_kc = layer.k_b_proj_trans_dequant
    return w_kc


def q_fused_qb_exponents(layer) -> Optional[torch.Tensor]:
    """Block exponents of the q_b weight (``k3_mla_q_fused.block_exponents``), cached per weight
    scale; None if they would have to be built during CUDA graph capture."""
    scale = getattr(layer.q_b_proj, "weight_scale", None)
    if not isinstance(scale, torch.Tensor):
        return None
    cached = layer._q_fused_qb_exp
    if cached is not None and cached[0] is scale:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        return None
    from ..kernels.k3r_mla_q import k3_mla_q_fused_ull as k3_mla_q_fused

    weight = layer.q_b_proj.weight
    try:
        exponents = k3_mla_q_fused.block_exponents(scale, *weight.shape)
    except ValueError:
        return None
    layer._q_fused_qb_exp = (scale, exponents)
    return exponents


def q_fused_applies(layer, hidden_states: torch.Tensor, md: AttentionMetadata) -> bool:
    """Whether this forward takes :func:`forward_gen_q_fused`: a generation-only Helix batch of an
    FP4-KV layer (V up-projection before the exchange) whose query fits the fused kernel."""
    num_tokens = md.num_tokens
    if (
        hidden_states.dim() != 2
        or hidden_states.shape[0] != num_tokens
        or md.num_contexts != 0
        or md.num_generations <= 0
        or num_tokens % md.num_generations != 0
        or not layer.helix_v_up_before_exchange
        or not layer.mqa.uses_fp4_mla_attention
        or getattr(md, "helix_position_offsets", None) is None
        or getattr(md, "helix_is_inactive_rank", None) is None
        or layer.mqa.rotary_cos_sin is None
        or layer.mqa.rotary_cos_sin.dtype != torch.float32
        or not layer.mqa.rotary_cos_sin.is_contiguous()
    ):
        return False
    from ..kernels.k3r_mla_q import k3_mla_q_fused_ull as k3_mla_q_fused

    if not (
        layer.num_heads_tp == k3_mla_q_fused.NUM_HEADS
        and layer.q_lora_rank == k3_mla_q_fused.Q_LORA_RANK
        and layer.kv_lora_rank == k3_mla_q_fused.KV_LORA_RANK
        and layer.qk_nope_head_dim == k3_mla_q_fused.NOPE_DIM
        and layer.qk_rope_head_dim == k3_mla_q_fused.ROPE_DIM
        and num_tokens <= min(k3_mla_q_fused.MAX_TOKENS, _Q_FUSED_MAX_TOKENS)
    ):
        return False
    weight = layer.q_b_proj.weight
    w_uk = q_fused_w_uk(layer)
    if w_uk is None or not k3_mla_q_fused.weights_supported(weight, w_uk):
        return False
    return q_fused_qb_exponents(layer) is not None


def latent_row(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """``concat([kv_a RMSNorm(compressed_kv), k_pe], -1)``: FlashInfer's norm written straight into
    the row and ``k_pe`` copied beside it."""
    norm = layer.kv_a_layernorm
    if not (IS_FLASHINFER_AVAILABLE and compressed_kv.dtype == torch.bfloat16):
        return torch.concat(
            [rms_norm(compressed_kv, norm.weight, norm.variance_epsilon), k_pe], dim=-1
        )
    from flashinfer.norm import rmsnorm

    lora = compressed_kv.shape[-1]
    latent = torch.empty(
        (compressed_kv.shape[0], lora + k_pe.shape[-1]),
        dtype=compressed_kv.dtype,
        device=compressed_kv.device,
    )
    rmsnorm(
        compressed_kv,
        norm.weight,
        norm.variance_epsilon,
        out=latent[:, :lora],
        enable_pdl=USE_PDL,
    )
    latent[:, lora:].copy_(k_pe)
    return latent


def _q_fused_latent_row(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """:func:`latent_row` in one ``k3_mla_latent_row`` launch where FlashInfer's norm is its CuTe
    DSL kernel (whose arithmetic it repeats, bit for bit)."""
    from ..kernels.k3r_mla_q import k3_mla_q_fused_ull as k3_mla_q_fused
    from ..ops.attention.k3_mla_latent_row_ull import k3_mla_latent_row

    norm = layer.kv_a_layernorm
    if (
        IS_FLASHINFER_AVAILABLE
        and _flashinfer_cute_norm(compressed_kv.device)
        and k3_mla_q_fused.latent_row_supported(compressed_kv, k_pe, norm.weight)
    ):
        return k3_mla_latent_row(
            compressed_kv,
            k_pe,
            norm.weight,
            norm.variance_epsilon,
            use_pdl=USE_PDL,
        )
    return latent_row(layer, compressed_kv, k_pe)


def forward_gen_q_fused(
    layer,
    hidden_states: torch.Tensor,
    md: AttentionMetadata,
    hooks: Optional[HelixGenHooks] = None,
) -> torch.Tensor:
    """Absorbed Helix generation of an FP4-KV layer with the fused query; returns the combined
    all-to-all output ``[T, H / P * v_head_dim]``.

    After the fused q_a / kv_a projection the KV half (``k3_mla_latent_row``, then the FP4 cache
    update without its query rows) runs on the aux stream (with multi-stream on) and the main
    stream runs ``k3_mla_q_fused`` (on the raw q_a rows with the RMSNorm inside, or after the
    separate q_a RMSNorm), which writes the FP4 query rows into the Q buffers the cache update
    recorded on the metadata. The decode joins both; it reads only those buffers (the BF16
    ``fused_q`` it is handed provides shapes)."""
    from tensorrt_llm._torch.attention.backends.fp4_mla.layout import _get_fp4_mla_q_global_scale

    from ..kernels.k3r_mla_q import k3_mla_q_fused_ull as q_fused_kernel
    from ..ops.attention._k3_fp4_mla_cache import _prepare_fp4_mla_q_buffers
    from ..ops.attention.fp4_mla_kv_append_ull import fp4_mla_kv_append
    from ..ops.attention.k3_mla_q_fused_ull import k3_mla_q_fused

    num_tokens = md.num_tokens
    heads, lora, rope = layer.num_heads_tp, layer.kv_lora_rank, layer.qk_rope_head_dim
    device = hidden_states.device
    gen_len = num_tokens // md.num_generations
    # The Q buffers the KV half records on the metadata, created (zero-filled, once) on this
    # stream, which writes them.
    _prepare_fp4_mla_q_buffers(
        md,
        num_tokens,
        heads,
        device,
        rows_per_seq=fp4_mla_decode.packed_rows_per_seq(heads, gen_len),
    )
    q, compressed_kv, k_pe = layer.project_qkv_a(hidden_states).split(
        [layer.q_lora_rank, lora, rope], -1
    )
    kv_fork, kv_ready = layer._kv_events
    kv_stream = layer.aux_stream if do_multi_stream() and layer.aux_stream is not None else None
    if kv_stream is not None:
        kv_fork.record()
    with torch.cuda.stream(kv_stream) if kv_stream is not None else contextlib.nullcontext():
        if kv_stream is not None:
            kv_fork.wait()
        latent_cache = _q_fused_latent_row(layer, compressed_kv, k_pe)
        fp4_mla_kv_append(layer.mqa, latent_cache, md, heads)
        if kv_stream is not None:
            kv_ready.record()
    norm = layer.q_a_layernorm
    fuse_norm = (
        q_fused_kernel.NORM_FUSION_AVAILABLE
        and IS_FLASHINFER_AVAILABLE
        and _flashinfer_cute_norm(device)
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.is_contiguous()
        and q_fused_kernel.input_supported(q)
    )
    if not fuse_norm:
        q = layer.q_a_norm(q)
    assert q_fused_kernel.input_supported(q), "the fused query reads BF16 q_a rows"
    state = md.fp4_mla_state
    rows_per_seq = int(state.prequantized_q_rows_per_seq or 0)
    k3_mla_q_fused(
        q,
        layer.q_b_proj.weight,
        q_fused_qb_exponents(layer),
        q_fused_w_uk(layer),
        _get_fp4_mla_q_global_scale(md, device),
        layer.mqa.rotary_cos_sin,
        md.helix_position_offsets,
        state.prequantized_q,
        state.prequantized_q_sf.view(torch.uint8),
        gen_len=gen_len,
        rows_per_seq=rows_per_seq or gen_len * heads,
        use_pdl=USE_PDL,
        norm=(norm.weight, norm.variance_epsilon) if fuse_norm else None,
        # features.ull2_mla_decode_pdl: the decode (a programmatic dependent reading only the
        # query after its wait) starts its prologue as this kernel's CTAs finish.
        early_trigger=_gen_F.ull2_mla_decode_pdl,
    )
    fp4_mla_decode.mark_query_launch(md)
    if kv_stream is not None:
        kv_ready.wait()
    # The decode reads only the packed FP4 query rows; this BF16 tensor provides shapes. (Served
    # also allocates the unused cu_q / cu_kv / scheduler / BMM-scale / FP8-query buffers here: no
    # kernels.)
    fused_q = torch.empty([num_tokens, heads, lora + rope], dtype=q.dtype, device=device)
    attn_out = attn_forward_gen(
        layer,
        fused_q.view([num_tokens, heads * (lora + rope)]),
        md,
        helix_v_up=True,
        hooks=hooks,
    )
    assert attn_out.shape == (num_tokens, layer.num_heads_tp_cp * layer.v_head_dim)
    return attn_out


_MLA_F = _features()

# Generation-only Helix batches of at most this many rows (V up-projection before the
# all-to-all) take the side-gate forward: the zero-KV mask and the output gate
# sigmoid(g_proj(x)) run on a side stream, and o_proj gates its input in its GEMM.
_K3_MLA_GATE_SIDE_MAX_M = 128
# features.mla_q_norm_pdl: the q_a RMSNorm releases the q_b GEMM at launch instead of after its
# store (``q_norm_pdl``; same kernel code), so q_b streams its weights while q_a and the norm run.
_K3_MLA_Q_NORM_PDL = _MLA_F.mla_q_norm_pdl
# features.ull3_mla_gate_at (side-gate forward with the fused Helix tail): when the side-stream
# output gate GEMM is forked. "decode" (default): right before the decode, joined before the
# tail; "start": at the layer start with the zero-KV mask, beside q_a / kv_a; "exchange": right
# after the decode is launched and joined before o_proj, which then gates its input itself. A
# value "<place>:<T>,<place>" picks the first placement up to T tokens (default 16) and the
# second above, e.g. "exchange:16,start". Scheduling only.
_ULL3_GATE_PLACES = ("decode", "start", "exchange")


def _parse_ull3_gate_at(value: str) -> Tuple[str, int, str]:
    small, _, large = value.partition(",")
    small, _, limit = small.partition(":")
    large = large or small
    if small not in _ULL3_GATE_PLACES or large not in _ULL3_GATE_PLACES:
        raise ValueError(
            f"features.ull3_mla_gate_at={value!r}: expected <place>[:<tokens>][,<place>] with "
            f"places {_ULL3_GATE_PLACES}"
        )
    return small, int(limit) if limit else 16, large


_K3_ULL3_MLA_GATE_AT = _parse_ull3_gate_at(_MLA_F.ull3_mla_gate_at)


def ull3_gate_place(num_tokens: int) -> str:
    """The features.ull3_mla_gate_at placement of a batch of ``num_tokens`` tokens."""
    small, limit, large = _K3_ULL3_MLA_GATE_AT
    return small if num_tokens <= limit else large


# One side stream per device, shared by all MLA layers (their side work never overlaps).
_GATE_SIDE_STREAMS: dict[int, torch.cuda.Stream] = {}


def _gate_side_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _GATE_SIDE_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index, priority=0)
        _GATE_SIDE_STREAMS[index] = stream
    return stream


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
        mapping_with_cp: Optional[Mapping] = None,
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
        self._reject_unsupported(config, mapping_with_cp, fp8)

        # Registered for the custom ops; a later stack (one-model speculative decoding) with the
        # same local layer index registers as "<layer_idx>_<n>".
        self.register_to_config = True
        registry = config.extra_attrs.setdefault("mla_layers", {})
        suffix = 0
        while self.layer_idx_str in registry:
            self.layer_idx_str = f"{layer_idx}_{suffix}"
            suffix += 1
        registry[self.layer_idx_str] = weakref.ref(self)

        # Head sharding: q_b / kv_b over TP (attention DP: none), the post-exchange heads of
        # o_proj / g_proj over TP x CP.
        self.mapping = mapping_with_cp if mapping_with_cp is not None else config.mapping
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
        self.num_heads_tp_cp = self.num_heads_tp // cp_size
        # Helix context attends this CP rank's head range only.
        self.num_context_heads = (
            self.num_heads_tp_cp if self.mapping.has_cp_helix() else self.num_heads_tp
        )

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
        # This CP rank's post-exchange heads of the V absorption (bound by the loader).
        self.v_b_proj = nn.Parameter(
            torch.empty((self.num_heads_tp_cp, v_head_dim, kv_lora_rank), dtype=dtype),
            requires_grad=False,
        )
        # Helix only: the BF16 V weight of every tp-local head, applied to this rank's partials
        # before the all-to-all (``project_helix_v_partials``).
        self.register_buffer("helix_v_b_proj_full", None, persistent=False)
        self.helix_v_up_before_exchange = False
        self.o_proj = projection(
            fp8["o_proj"], num_heads * v_head_dim, hidden_size, row, self.mapping_o
        )
        self.attention_output_hidden_size = self.o_proj.in_features
        if use_output_gate:
            # o_proj's input sharding (under Helix the post-exchange 1/cp head chunk).
            self.g_proj = projection(
                fp8["g_proj"], hidden_size, num_heads * v_head_dim, column, self.mapping_o
            )

        quant_config = config.get_quant_config()
        self.quant_config = quant_config
        # The two attention roles over the engine's KV cache (``ops/attention/k3_mla_state``): the
        # absorbed generation (``mqa``: one 576-dim latent head per query head), which the decode,
        # the cache updates and the fused query read, and the context (``mha``: 192 / 128-dim
        # heads), of which this generation-only target reads the KV cache quantization only. Both
        # carry the identity RoPE table (K3 is NoPE).
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
            num_context_heads=self.num_context_heads,
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
        # features.ull4_mla_plan_early: event of the step metadata the model pre-launched.
        self.step_metadata_ready: Optional[torch.cuda.Event] = None
        # Side-gate forward: fork / join of the zero-KV mask and the gate on the side stream.
        self._gate_events = tuple(torch.cuda.Event() for _ in range(4))
        # Fused query: fork of the KV half after q_a, join before the decode.
        self._kv_events = (torch.cuda.Event(), torch.cuda.Event())
        # Set by the decoder layer for the next forward only: the MXFP8 form ``(xq, xs)`` of the
        # layer input, which the fused q_a / kv_a projection consumes (see q_a_takes_mxfp8).
        self.next_input_mxfp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # Fused query: (q_b weight scale, its block exponents), built on first use.
        self._q_fused_qb_exp: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # MLAIN (``mlain.py``): the folded ``[q_a | kv_a | 0 | g]`` projection (``mlain_fold``),
        # or None.
        self.in_fold: Optional[FoldedInProj] = None
        # Set by ``K3MLARuntime.decode_attention`` for the MLAIN call in flight (its ``released``:
        # the producer released the projection at kernel entry); None outside an MLAIN call.
        self.mlain_released: Optional[bool] = None
        # The folded gate of the call in flight: ``(layer input, sigmoid(g) column view)``.
        self._folded_gate: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if not config.skip_create_weights_in_init:
            self.create_weights()

    @staticmethod
    def _reject_unsupported(
        config: ModelConfig, mapping_with_cp: Optional[Mapping], fp8: dict[str, bool]
    ) -> None:
        """Configurations whose kernel chain K3 does not carry."""
        mapping = mapping_with_cp if mapping_with_cp is not None else config.mapping
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
        # BF16 projections, BF16 absorption BMMs (BF16 kv_b, or the Helix V up-projection), FP8
        # absorption BMMs.
        if config.use_cute_dsl_bf16_gemm and not all(fp8.values()):
            unsupported.append("use_cute_dsl_bf16_gemm with BF16 projections")
        if config.use_cute_dsl_bf16_bmm and (mapping.has_cp_helix() or not fp8["kv_b_proj"]):
            unsupported.append("use_cute_dsl_bf16_bmm")
        if config.use_cute_dsl_blockscaling_bmm and fp8["kv_b_proj"]:
            unsupported.append("use_cute_dsl_blockscaling_bmm")
        if mapping.has_cp_ulysses():
            unsupported.append("Ulysses CP")
        if mapping.cp_size > 1 and not mapping.has_cp_helix():
            unsupported.append(f"CP type {mapping.cp_config.get('cp_type')}")
        if mapping.has_cp_helix():
            if mapping.enable_attention_dp:
                unsupported.append("Helix CP with attention DP")
            if mapping.cp_config.get("use_nccl_for_alltoall", True):
                unsupported.append("the NCCL Helix all-to-all (use_nccl_for_alltoall: false)")
            elif mapping.cp_config.get("fifo_version", 2) != 2:
                unsupported.append("the fifo-v1 Helix all-to-all")
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
            and not self.mapping.has_cp_helix()
            and (self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim) == (128, 64, 128)
            and isinstance(scale, torch.Tensor)
            and bool(torch.all(scale == 1.0).item())
        )
        heads, heads_cp = self.num_heads_tp, self.num_heads_tp_cp
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
                torch.empty((heads_cp, vdim // 128, lora // 128), dtype=torch.float32),
                requires_grad=False,
            )
            if is_sm_100f():
                # The absorption BMMs run in BF16 on dequantized copies.
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty((heads, lora, nope), dtype=self.dtype), requires_grad=False
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty((heads_cp, vdim, lora), dtype=self.dtype), requires_grad=False
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None
        self._weights_transformed = False

    # -----------------------------------------------------------------------------------------
    # Post-load.
    # -----------------------------------------------------------------------------------------

    def _bind_helix_v_up_weight(self, weight: torch.Tensor) -> None:
        """Apply the V up-projection with ``weight`` (BF16 ``[num_heads_tp, v_head_dim,
        kv_lora_rank]``) before the Helix all-to-all."""
        assert self.mapping.has_cp_helix()
        assert weight.dtype == torch.bfloat16
        assert tuple(weight.shape) == (self.num_heads_tp, self.v_head_dim, self.kv_lora_rank)
        self.helix_v_b_proj_full = weight
        self.helix_v_up_before_exchange = True

    def _build_helix_v_up_weight(self) -> None:
        """Under Helix, every tp-local head's BF16 V weight from kv_b's loaded rows (BF16, or the
        raw FP8 codes with their FP32 128 x 128 block scales: this runs before kv_b's own
        transform, which the post-load walk visits after this module)."""
        if not self.mapping.has_cp_helix() or self.dtype != torch.bfloat16:
            return
        weight = self.kv_b_proj.weight
        nope_rows = self.num_heads_tp * self.qk_nope_head_dim
        num_rows = nope_rows + self.num_heads_tp * self.v_head_dim
        v_shape = (self.num_heads_tp, self.v_head_dim, self.kv_lora_rank)
        if weight.is_meta or tuple(weight.shape) != (num_rows, self.kv_lora_rank):
            return
        if weight.dtype == torch.bfloat16:
            v_weight = weight[nope_rows:].view(v_shape)
        elif weight.dtype == torch.float8_e4m3fn:
            scale = self.kv_b_proj.weight_scale
            if scale.dtype != torch.float32 or tuple(scale.shape) != (
                -(-num_rows // 128),
                -(-self.kv_lora_rank // 128),
            ):
                return
            row_blocks = torch.arange(nope_rows, num_rows, device=weight.device) // 128
            v_scale = scale[row_blocks].repeat_interleave(128, dim=1)[:, : self.kv_lora_rank]
            v_weight = (weight[nope_rows:].float() * v_scale).to(self.dtype).view(v_shape)
            if self.v_b_proj_dequant is not None:
                # This rank's dequantized V chunk holds the same values.
                start = self.mapping.cp_rank * self.num_heads_tp_cp
                self.v_b_proj_dequant = nn.Parameter(
                    v_weight[start : start + self.num_heads_tp_cp], requires_grad=False
                )
        else:
            return
        self._bind_helix_v_up_weight(v_weight)

    def transform_weights(self) -> None:
        if self._weights_transformed:
            return
        self._build_helix_v_up_weight()
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        """The scales of the q_b / kv_b output slices a Helix rank's context heads read."""
        if self.num_context_heads != self.num_heads_tp:
            first_head = self.mapping.cp_rank * self.num_context_heads
            last_head = first_head + self.num_context_heads
            value_start = self.num_heads_tp * self.qk_nope_head_dim
            for proj, start, end in (
                (self.q_b_proj, first_head * self.qk_head_dim, last_head * self.qk_head_dim),
                (
                    self.kv_b_proj,
                    first_head * self.qk_nope_head_dim,
                    last_head * self.qk_nope_head_dim,
                ),
                (
                    self.kv_b_proj,
                    value_start + first_head * self.v_head_dim,
                    value_start + last_head * self.v_head_dim,
                ),
            ):
                if not isinstance(proj, Bf16Linear):
                    proj.prepare_output_slice(start, end)
        self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.cache_derived_state()

    # -----------------------------------------------------------------------------------------
    # Front.
    # -----------------------------------------------------------------------------------------

    def q_a_mx_atom(self, num_rows: int) -> bool:
        """Whether the fused q_a / kv_a GEMM takes the MXFP8 rows with the scales in the mid-M
        GEMM's layout (``takes_mid_mxfp8_input``)."""
        takes_mid = getattr(self.kv_a_proj_with_mqa, "takes_mid_mxfp8_input", None)
        return takes_mid is not None and bool(takes_mid(num_rows))

    def q_a_takes_mxfp8(self, num_rows: int) -> bool:
        """Whether the fused q_a / kv_a GEMM of ``num_rows`` rows can take their MXFP8 form
        (``next_input_mxfp8``)."""
        takes = getattr(self.kv_a_proj_with_mqa, "takes_mxfp8_input", None)
        return takes is not None and (bool(takes(num_rows)) or self.q_a_mx_atom(num_rows))

    def mlain_fold(self) -> Optional[FoldedInProj]:
        """The layer's folded projection (``mlain.FoldedInProj``), built on first use outside CUDA
        graph capture and rebuilt where the projections' weights were replaced; None where it does
        not apply. Served ``ull/mlain.py`` ``MlaInAttention._fold``."""
        held = self.in_fold
        g_proj = getattr(self, "g_proj", None)
        if held is not None and held.is_current(self.kv_a_proj_with_mqa, g_proj):
            return held
        if torch.cuda.is_current_stream_capturing() or g_proj is None:
            return None
        self.in_fold = None
        self.in_fold = FoldedInProj.build(self.kv_a_proj_with_mqa, g_proj)
        return self.in_fold

    def prepare_in_fold(self) -> bool:
        """Build the fold of an FP4-KV layer with the output gate ahead of the first step (outside
        CUDA graph capture; :meth:`mlain_fold` builds it on first use otherwise); returns whether
        one is built."""
        if not (self.use_output_gate and self.fp4_kv):
            return False
        return self.mlain_fold() is not None

    def fold_call(self, num_rows: int, mx: Optional[Tuple[torch.Tensor, torch.Tensor]], released):
        """``(fold, tactic)`` where this call runs the folded projection on the MXFP8 rows ``mx``
        (``released``: the producer released the projection at kernel entry, so it takes the
        staged tactic), else None. Served ``MlaInAttention._fold_call``."""
        if mx is None:
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8_ull as mxk

        fold = self.mlain_fold()
        if fold is None or fold.n_gate != self.o_proj.in_features:
            return None
        tactic = fold.tactic(num_rows, released)
        if tactic is None or num_rows > tactic.mma_n:
            return None
        if not mxk.can_run_mxfp8(mx[0], mx[1], fold.weight, fold.scale):
            return None
        return fold, tactic

    def project_qkv_a(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The fused q_a / kv_a projection ``[T, q_lora | kv_lora | rope]``.

        In an MLAIN call (:attr:`mlain_released` set; served ``MlaInAttention.run``'s
        ``project``): the folded projection on the MXFP8 rows where :meth:`fold_call` serves it
        (its gate columns kept for :meth:`_gate_of`), else ``forward_mxfp8`` with the weights
        staged, else the projection of the BF16 rows. Otherwise (served
        ``KimiK3MLAAttention._project_qkv_a``): on the MXFP8 rows the caller handed over for these
        rows (``forward_mxfp8``), else the GEMM on the BF16 rows (a BF16 weight: ``cublas_mm``)."""
        mxfp8_input, self.next_input_mxfp8 = self.next_input_mxfp8, None
        released, self.mlain_released = self.mlain_released, None
        proj = self.kv_a_proj_with_mqa
        if released is not None:
            fold_call = self.fold_call(hidden_states.shape[0], mxfp8_input, released)
            if fold_call is not None:
                fold, tactic = fold_call
                out = fold.run(*mxfp8_input, tactic)
                self._folded_gate = (hidden_states, out[:, fold.pad_rows :])
                return out[:, : fold.n_qkv]
            if mxfp8_input is not None:
                return proj.forward_mxfp8(hidden_states, *mxfp8_input, stage_weights=True)
            return proj(hidden_states)
        if isinstance(proj, Bf16Linear):
            return cublas_mm(
                hidden_states,
                proj.weight.t(),
                None,
                out_dtype=None,
                output_buffer_kind=int(BufferKind.DEFAULT),
                group=None,
            )
        if mxfp8_input is not None and mxfp8_input[0].shape[0] == hidden_states.shape[0]:
            return proj.forward_mxfp8(hidden_states, *mxfp8_input)
        return proj(hidden_states)

    def q_a_norm(self, q: torch.Tensor) -> torch.Tensor:
        """The q_a RMSNorm (FlashInfer); with ``features.mla_q_norm_pdl`` and PDL the same kernel
        releasing its dependent (q_b) at launch."""
        norm = self.q_a_layernorm
        if _K3_MLA_Q_NORM_PDL and IS_FLASHINFER_AVAILABLE and USE_PDL:
            from ..kernels.k3r_mla_q.q_norm_pdl import can_use
            from ..ops.attention.rmsnorm_release_at_launch import rmsnorm_release_at_launch

            if can_use(q, norm.weight):
                return rmsnorm_release_at_launch(q, norm.weight, norm.variance_epsilon)
        return rms_norm(q, norm.weight, norm.variance_epsilon)

    @staticmethod
    def forward_slice(proj, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Output features ``[start, end)`` of a column-parallel projection, contiguous."""
        if isinstance(proj, Bf16Linear):
            return torch.nn.functional.linear(x, proj.weight[start:end])
        return proj.forward_slice(x, start, end)

    def _project_q_by_phase(
        self, q: torch.Tensor, num_ctx_tokens: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """q_b of the context rows (a Helix rank's context heads only) and of the generation rows."""
        if self.num_context_heads == self.num_heads_tp or num_ctx_tokens == 0:
            projected = self.q_b_proj(q)
            return projected[:num_ctx_tokens], projected[num_ctx_tokens:]
        first = self.mapping.cp_rank * self.num_context_heads * self.qk_head_dim
        last = first + self.num_context_heads * self.qk_head_dim
        ctx = self.forward_slice(self.q_b_proj, q[:num_ctx_tokens], first, last)
        gen = self.q_b_proj(q[num_ctx_tokens:]) if q.shape[0] > num_ctx_tokens else None
        return ctx, gen

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        md: AttentionMetadata,
        output: torch.Tensor,
        hooks: Optional[HelixGenHooks],
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
            raise RuntimeError(
                f"Kimi K3 ULL MLA layer {self.layer_idx}: context rows are not served "
                "(generation steps only)"
            )
        if md.num_generations > 0:
            assert q_gen is not None
            forward_absorption_generation(
                self,
                q_gen,
                latent_cache[num_ctx_tokens:, ...],
                md,
                output[num_ctx_tokens:num_tokens, :],
                hooks,
            )

    def run_attention(
        self,
        hidden_states: torch.Tensor,
        md: AttentionMetadata,
        output: torch.Tensor,
        hooks: Optional[HelixGenHooks] = None,
    ) -> None:
        """The attention part of a forward (everything before the output gate) into ``output``,
        on the mixed-precision KV staging copy where the metadata has one."""
        with md.mixed_precision_kv_staging(self.layer_idx):
            self._forward_attention(hidden_states, md, output, hooks)

    def create_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The attention output ``[rows, H / P * v_head_dim]`` for the layer input's rows."""
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

    def _side_gate_applies(self, hidden_states: torch.Tensor, md: AttentionMetadata) -> bool:
        """Whether this forward takes :meth:`_forward_side_gate`."""
        return (
            self.use_output_gate
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.shape[0] == md.num_tokens
            and md.num_contexts == 0
            and self.helix_v_up_before_exchange
            and hasattr(self.o_proj, "gated_applies")
            and hidden_states.shape[0] <= _K3_MLA_GATE_SIDE_MAX_M
        )

    def _gate_of(self, x: torch.Tensor) -> torch.Tensor:
        """``sigmoid(g_proj(x))``: the folded projection's gate columns where this call's input
        projection was the fold (no kernel), else ``g_proj.forward_sigmoid`` (one GEMM)."""
        folded = self._folded_gate
        if folded is not None and x is folded[0]:
            return folded[1]
        forward_sigmoid = getattr(self.g_proj, "forward_sigmoid", None)
        if forward_sigmoid is not None:
            return forward_sigmoid(x)
        return self.g_proj(x).sigmoid()

    def _side_gate_core(
        self, hidden_states: torch.Tensor, md: AttentionMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """:meth:`_forward_side_gate` up to the o_proj launch: ``(attn_out, gate, tail_mx)``, the
        operands of :meth:`gated_o_proj`.

        Generation forward with the zero-KV mask and the output gate on a side stream.

        The mask (one small kernel) is forked once the layer input is written; the gate (one GEMM
        with a sigmoid epilogue) is forked when the all-to-all is enqueued (or per
        ``features.ull3_mla_gate_at``), so it runs on the SMs the all-to-all leaves idle. The main
        stream waits for the mask where the exchange reads it and for the gate before o_proj; with
        the fused Helix tail and an early gate, before the tail kernel, so o_proj depends on the
        tail alone and gates its input in its GEMM (``forward_gated(_mxfp8)``)."""
        num_tokens = hidden_states.shape[0]
        mask_fork, mask_ready, gate_fork, gate_ready = self._gate_events
        multi_stream = do_multi_stream()
        side = _gate_side_stream(hidden_states.device) if multi_stream else None
        gates = []
        mask_joined = [not multi_stream]
        gate_joined = [not multi_stream]

        gate_of = self._gate_of
        if multi_stream:
            mask_fork.record()
            with torch.cuda.stream(side):
                mask_fork.wait()
                zero_kv_mask = helix_gen_zero_kv_mask(md, num_tokens)
                mask_ready.record()
        else:
            zero_kv_mask = helix_gen_zero_kv_mask(md, num_tokens)

        def zero_kv_mask_fn():
            if not mask_joined[0]:
                mask_ready.wait()
                mask_joined[0] = True
            return zero_kv_mask

        def pre_exchange():
            if gates:
                return  # already forked (gate place "start")
            if multi_stream:
                gate_fork.record()
                with torch.cuda.stream(side):
                    gate_fork.wait()
                    gates.append(gate_of(hidden_states))
                    gate_ready.record()
            else:
                gates.append(gate_of(hidden_states))

        def pre_tail():
            if gates and not gate_joined[0]:
                gate_ready.wait()
                gate_joined[0] = True
            return gates[0] if gates else None

        gate_place = ull3_gate_place(num_tokens) if multi_stream else "decode"
        if gate_place == "start":
            pre_exchange()
        hooks = HelixGenHooks(zero_kv_mask_fn, pre_exchange, pre_tail, gate_place)
        try:
            if q_fused_applies(self, hidden_states, md):
                attn_out = forward_gen_q_fused(self, hidden_states, md, hooks)
            else:
                output = self.create_output(hidden_states)
                self.run_attention(hidden_states, md, output, hooks)
                attn_out = hooks.gen_output if hooks.gen_output is not None else output
            if not mask_joined[0]:
                mask_ready.wait()
            if gates:
                gate = gates[0]
                if not gate_joined[0]:
                    gate_ready.wait()
            else:
                gate = gate_of(hidden_states)
        finally:
            self._folded_gate = None
        return attn_out, gate, hooks.tail_mx

    def gated_o_proj(
        self,
        attn_out: torch.Tensor,
        gate: torch.Tensor,
        tail_mx: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """o_proj of ``attn_out * gate``: on the MXFP8 operand the fused Helix tail stored
        (``tail_mx``), else the gated low-M GEMM, else the multiply and o_proj (the push target
        in ``o_proj.next_ar_push`` is the caller's)."""
        if tail_mx is not None:
            return self.o_proj.forward_gated_mxfp8(attn_out, gate, *tail_mx)
        if self.o_proj.gated_applies(attn_out, gate):
            return self.o_proj.forward_gated(attn_out, gate)
        return self.o_proj(attn_out * gate)

    def _forward_side_gate(
        self, hidden_states: torch.Tensor, md: AttentionMetadata
    ) -> torch.Tensor:
        """Generation forward with the zero-KV mask and the output gate on a side stream
        (:meth:`_side_gate_core`, then :meth:`gated_o_proj`)."""
        return self.gated_o_proj(*self._side_gate_core(hidden_states, md))

    def decode_core(
        self, hidden_states: torch.Tensor, md: AttentionMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """The generation step's attention up to the o_proj launch (:meth:`_side_gate_core`):
        ``(attn_out, gate, tail_mx)``. Joins the step metadata the model pre-launched first;
        raises where the side-gate forward does not serve the step (this target serves nothing
        else)."""
        try:
            if not self._side_gate_applies(hidden_states, md):
                raise RuntimeError(
                    f"Kimi K3 ULL MLA layer {self.layer_idx}: the generation step "
                    f"({hidden_states.shape[0]} rows, {md.num_contexts} context requests) is not "
                    "a side-gate Helix generation batch"
                )
            ready = self.step_metadata_ready
            if ready is not None:
                self.step_metadata_ready = None
                torch.cuda.current_stream().wait_event(ready)
            return self._side_gate_core(hidden_states, md)
        finally:
            self.next_input_mxfp8 = None
            self.mlain_released = None

    def prelaunch_step_metadata(self, attn_metadata: AttentionMetadata) -> bool:
        """``features.ull4_mla_plan_early``: launch, on the current stream, the step metadata the
        first FP4 MLA generation kernel would write (generation lengths fused with the packed
        split plan of this layer's query heads, then the device page table). Returns whether
        anything was launched; the caller sets :attr:`step_metadata_ready` to an event recorded
        after it."""
        if not fp4_mla_decode.applies(self.mqa) or attn_metadata.num_contexts != 0:
            return False
        use_helix = (
            getattr(attn_metadata, "helix_position_offsets", None) is not None
            or getattr(attn_metadata, "helix_is_inactive_rank", None) is not None
        )
        return fp4_mla_decode.prelaunch_generation_metadata(
            attn_metadata,
            num_heads=self.num_heads_tp,
            use_helix=use_helix,
            device=self.o_proj.weight.device,
        )

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """o_proj's partial output of the layer for ``hidden_states`` (K3 is NoPE:
        ``position_ids`` is None)."""
        assert position_ids is None, "Kimi K3 MLA is NoPE"
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"Kimi K3 MLA takes BF16 rows, got {type(hidden_states).__name__}")
        ready = self.step_metadata_ready
        if ready is not None:
            # features.ull4_mla_plan_early: the step metadata written on the model's side stream.
            self.step_metadata_ready = None
            torch.cuda.current_stream().wait_event(ready)
        try:
            if self._side_gate_applies(hidden_states, attn_metadata):
                return self._forward_side_gate(hidden_states, attn_metadata)
            if q_fused_applies(self, hidden_states, attn_metadata):
                attn_out = forward_gen_q_fused(self, hidden_states, attn_metadata)
                return self._gate_and_o_proj(hidden_states, attn_out)
            if self.register_to_config:
                output = create_mla_outputs(hidden_states, self.layer_idx_str)
                _bcg_mla_custom_op_inplace(hidden_states, self.layer_idx_str, output)
            else:
                output = self.create_output(hidden_states)
                self.run_attention(hidden_states, attn_metadata, output)
            return self._gate_and_o_proj(hidden_states, output)
        finally:
            # Consumed by the projection; never carried into another forward.
            self.next_input_mxfp8 = None
            self.mlain_released = None


# Identity-RoPE table positions of the MLA backends. K3 is NoPE (the table holds cos=1 / sin=0),
# but the cache-update kernels index the table by absolute position, so it covers every position
# of the deployment. Served: KIMI_K3_MLA_MAX_POSITIONS, 1048576 in 1007804.
MLA_MAX_POSITIONS = 1048576

# The MLA layers whose KV cache is FP8 under the model-wide NVFP4 KV cache: the last two MLA layers.
# Served: TLLM_KIMI_K3_FP8_KV_LAYERS=91,92 in 1007804. The KV cache manager allocates them from the
# engine's selection (``kimi_k3_cache_policy.get_kimi_k3_fp8_kv_layer_ids``), which must agree.
FP8_KV_LAYERS = frozenset({91, 92})


def _kimi_k3_fp8_kv_layer_ids(cfg, kv_quant_algo: Optional[QuantAlgo]) -> frozenset:
    """The MLA layers the KV cache manager allocates as FP8 (the engine's selection, which the model
    follows): :data:`FP8_KV_LAYERS` for the full checkpoint under an NVFP4 KV cache (checked; a
    truncated test model keeps whatever its manager selects), none otherwise."""
    engine = _kimi_k3_cache_policy.get_kimi_k3_fp8_kv_layer_ids(
        cfg, nvfp4_enabled=kv_quant_algo == QuantAlgo.NVFP4
    )
    full_model = int(getattr(cfg, "num_hidden_layers", 0)) > max(FP8_KV_LAYERS)
    if full_model and kv_quant_algo == QuantAlgo.NVFP4 and engine != FP8_KV_LAYERS:
        raise ValueError(
            f"Kimi K3 ULL: the KV cache manager allocates FP8 KV for MLA layers {sorted(engine)}, "
            f"this target serves {sorted(FP8_KV_LAYERS)} (TLLM_KIMI_K3_FP8_KV_LAYERS)"
        )
    return engine


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


class MlaDecodeCall(NamedTuple):
    """A generation step's MLA attention output before o_proj (``K3MLARuntime.decode_attention``):
    the o_proj operands and the push target of its all-reduce (``deferred``: finished by the
    post-attention mix)."""

    attn_out: torch.Tensor
    gate: torch.Tensor
    tail_mx: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ar_push: Any
    deferred: bool


class K3MLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()

        max_positions = MLA_MAX_POSITIONS
        self.layer_idx = layer_idx
        # K3MLAAttention owns MLA projection/head sharding. Keep only the
        # final output reduction in this wrapper so the output gate remains
        # between attention and the row-parallel o_proj.
        # Helix: mapping_with_cp (the CP original) activates the base MLA's
        # helix machinery; this wrapper's allreduce over the repurposed
        # mapping sums the base o_proj's tp*cp partials. reduce_output=False
        # leaves that reduction to the caller (helix_a2a reduce-scatter).
        mapping = model_config.mapping.mapping_for_full_rows()
        reduce_output = reduce_output and not mapping.enable_attention_dp and mapping.tp_size > 1
        self._o_allreduce = (
            TpAllReduce(mapping, torch.bfloat16, model_config.allreduce_strategy)
            if reduce_output
            else None
        )
        # Set by the decoder layer for one forward: a pushed o_proj all-reduce may be returned
        # unfinished, as a LamportPending that A_mid finishes.
        self.ar_finish_deferred = False
        # MLAIN (served ``ull/mlain.py`` ``MlaInAttention.released``): set by the caller for the
        # next decode_attention only, where the producer of the layer input released the
        # projection at kernel entry (:meth:`stages_weights`).
        self.released = False
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
                    "FP8 KV-cache MLA layers require the model-wide KV cache dtype to be NVFP4."
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
            mapping_with_cp=mapping_with_cp,
        )

    # ------------------------------------------------------------------ MLAIN
    def takes_q_fused(self, hidden_states, attn_metadata: AttentionMetadata) -> bool:
        """Whether the generation step runs the side-gate path with the fused query on
        ``hidden_states`` (a tensor of the attention input's shape and dtype) on at most
        ``mlain.MAX_ROWS`` rows. Served ``ull/mlain.py`` ``MlaInAttention.takes_q_fused`` (with
        ``MlaAttention._splits``)."""
        mixer = self.mixer
        if not isinstance(hidden_states, torch.Tensor):
            return False
        if not 1 <= hidden_states.shape[0] <= MLAIN_MAX_ROWS:
            return False
        if mixer.mapping.has_cp_helix() and mixer.mapping.enable_attention_dp:
            return False
        return mixer._side_gate_applies(hidden_states, attn_metadata) and q_fused_applies(
            mixer, hidden_states, attn_metadata
        )

    def stages_weights(self, rows: int, attn_metadata: AttentionMetadata, probe) -> bool:
        """Whether this layer's projection on ``rows`` rows keeps its weights on chip before its
        wait (the producer of the layer input then releases it at kernel entry). Served
        ``MlaInAttention.stages_weights``."""
        if not self.takes_q_fused(probe, attn_metadata):
            return False
        mixer = self.mixer
        fold = mixer.mlain_fold()
        if fold is not None and fold.tactic(rows, True) is not None:
            return fold.on_chip(rows)
        return bool(mixer.kv_a_proj_with_mqa.takes_staged_mxfp8_input(rows))

    # ------------------------------------------------------------------ decode
    def decode_attention(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> "MlaDecodeCall":
        """The generation step's attention up to its o_proj (``K3MLAAttention.decode_core``),
        with the o_proj all-reduce's push target made first (finished by :meth:`decode_output`;
        with :attr:`ar_finish_deferred` left to the post-attention mix).

        Where the step takes the fused query (:meth:`takes_q_fused`) the projection runs as
        served ``MlaInAttention.run``'s: :attr:`released` (consumed here) picks the staged tactic
        of the folded projection (``K3MLAAttention.fold_call``)."""
        mixer = self.mixer
        released, self.released = self.released, False
        o_proj = mixer.o_proj
        ar_push = None
        if self._o_allreduce is not None and hasattr(o_proj, "next_ar_push"):
            ar_push = lamport_push_target(
                self._o_allreduce,
                hidden_states.shape[0],
                o_proj.out_features,
                deferred=self.ar_finish_deferred,
            )
        if self.takes_q_fused(hidden_states, attn_metadata):
            mixer.mlain_released = released
        attn_out, gate, tail_mx = mixer.decode_core(hidden_states, attn_metadata)
        return MlaDecodeCall(attn_out, gate, tail_mx, ar_push, self.ar_finish_deferred)

    def decode_output(self, call: "MlaDecodeCall") -> Union[torch.Tensor, LamportPending]:
        """o_proj of a :meth:`decode_attention` call and its all-reduce: the reduced rows, or a
        ``LamportPending`` the post-attention mix finishes (``call.deferred``)."""
        o_proj = self.mixer.o_proj
        ar_push = call.ar_push
        o_proj.next_ar_push = ar_push
        try:
            out = self.mixer.gated_o_proj(call.attn_out, call.gate, call.tail_mx)
        finally:
            o_proj.next_ar_push = None
        if ar_push is not None and ar_push.pushed:
            out_shape = out.shape
            out = finish_lamport_push(self._o_allreduce, ar_push, deferred=call.deferred)
            if not isinstance(out, LamportPending):
                out = out.view(out_shape)
        elif ar_push is not None and ar_push.deferred:
            lamport_push_tensor(self._o_allreduce, ar_push, out.reshape(ar_push.output.shape))
            out = finish_lamport_push(self._o_allreduce, ar_push, deferred=True)
        elif self._o_allreduce is not None:
            out = self._o_allreduce(out)
        return out

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        o_proj = self.mixer.o_proj
        # o_proj stores its partial straight into the all-reduce's Lamport buffers where the
        # low-M GEMM serves it.
        ar_push = None
        if self._o_allreduce is not None and hasattr(o_proj, "next_ar_push"):
            ar_push = lamport_push_target(
                self._o_allreduce,
                hidden_states.shape[0],
                o_proj.out_features,
                deferred=self.ar_finish_deferred,
            )
            o_proj.next_ar_push = ar_push
        # MLA.forward takes position_ids first; K3 is NoPE, so pass None.
        out = self.mixer(None, hidden_states, attn_metadata)
        if ar_push is not None:
            o_proj.next_ar_push = None
        if ar_push is not None and ar_push.pushed:
            # A_mid may finish the all-reduce (a LamportPending).
            out_shape = out.shape
            out = finish_lamport_push(self._o_allreduce, ar_push, deferred=self.ar_finish_deferred)
            if not isinstance(out, LamportPending):
                out = out.view(out_shape)
        elif ar_push is not None and ar_push.deferred:
            # o_proj ran without the push epilogue; A_mid finishes.
            lamport_push_tensor(self._o_allreduce, ar_push, out.reshape(ar_push.output.shape))
            out = finish_lamport_push(self._o_allreduce, ar_push, deferred=True)
        elif self._o_allreduce is not None:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = self._o_allreduce(out)
        return out


_FP8_WEIGHT_READ_F = _features()

# FP8 weight-read projections with few rows (per-shape, per-architecture limit, see
# k3_fp8w_lowm_gemm.max_m_for) run the CuTe low-M GEMM: on SM100/SM103 instead of DeepGEMM
# swap-AB, on SM107 instead of the CuTe MXFP8 GEMM (with a DeepGEMM-packed copy of the weight
# scale, built at post-load, see _Fp8BlockScaleWeightReadLinear.prepare_lowm_gemm).
_K3_FP8W_LOWM_GEMM = _FP8_WEIGHT_READ_F.fp8w_lowm_gemm


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
    buckets. On both, the low-M GEMM (``trtllm::fp8_block_scale_low_m_gemm``) serves
    the rows ``layers.low_m_gemm_rows`` gives for the weight shape; it reads a
    DeepGEMM-packed scale, which is ``weight_scale`` on SM100/SM103 and a copy
    repacked from the MXFP8 scale on SM107. That dispatch is
    ``layers.fp8_block_scale_gemm``.

    The Rubin agentic kernels sit in front of that dispatch: the K3 low-M GEMM
    (``TRTLLM_K3_FP8W_LOWM_GEMM``, ``kernels/k3_fp8w_lowm_gemm.py``) with its gated,
    sigmoid, MXFP8-input and all-reduce-push variants. With ``TRTLLM_K3_FP8W_LOWM_GEMM``
    set the K3 low-M GEMM owns the low-M rows exactly as on that branch and the generic low-M GEMM is not
    prepared; with it unset the generic one serves them as on main.
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
        # Scales of the output-row slices :meth:`forward_slice` serves (an
        # ``Fp8BlockScaleWeightSliceCache``), prepared by :meth:`prepare_output_slice`.
        self._output_slice_cache: Optional[nn.Module] = None
        # Rows the low-M GEMM serves (0: none) and the DeepGEMM-packed scale it reads; set at
        # post-load.
        self._low_m_max = 0
        self._low_m_weight_scale: Optional[torch.Tensor] = None
        # Set by a caller for the next forward only: a contiguous BF16 ``[M, out_features]``
        # tensor that the low-M GEMM writes its result into. Other paths return their own
        # tensor, so the caller compares the result with the buffer it handed in.
        self.next_output: Optional[torch.Tensor] = None
        # Set by a caller for the next forward only (every forward consumes and clears it): the
        # ``LamportPushTarget`` (``layers.lamport_push_target``) of the all-reduce that
        # follows this projection. Where the low-M GEMM without split-K (or the mid-M GEMM with
        # its push epilogue) serves ``forward`` / ``forward_gated``, it stores its output into
        # that all-reduce's Lamport buffers instead, marks the target pushed and returns the
        # all-reduce's output tensor, which ``layers.finish_lamport_push`` fills.
        self.next_ar_push = None
        # Copied from the source ``Linear`` (the MLA fused q_a/kv_a projection sets it): the
        # low-M GEMM may serve this weight although N is not a multiple of 128.
        self.fp8w_lowm_partial_n_tile = False
        # SM107 with TRTLLM_K3_FP8W_LOWM_GEMM: the weight scale in the DeepGEMM-packed UE8M0
        # layout of the low-M GEMM (``weight_scale`` holds the CuTe MXFP8 layout there).
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

    @property
    def packed_weight_scale(self) -> Optional[torch.Tensor]:
        """The weight scale in DeepGEMM's packed UE8M0 layout, as the low-M GEMM reads it:
        ``weight_scale`` on SM100/SM103, ``lowm_weight_scale`` on SM107 (``None`` unless
        :meth:`prepare_lowm_gemm` built it)."""
        if _rubin_mxfp8_is_available():
            return self.lowm_weight_scale
        return self.weight_scale

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
            self._output_slice_cache = None
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
        self._prepare_low_m_gemm()

    def _prepare_low_m_gemm(self) -> None:
        """Select the low-M GEMM for this weight where it has a DeepGEMM-packed scale.

        On SM107 that scale is a copy (N * K / 128 bytes) derived from the MXFP8
        ``weight_scale``; row views of a fused KDA projection have none (the fused module
        serves decode). Not prepared under ``TRTLLM_K3_FP8W_LOWM_GEMM``: the K3 low-M GEMM
        owns those rows then (see the class docstring).
        """
        self._low_m_max, self._low_m_weight_scale = 0, None
        if _K3_FP8W_LOWM_GEMM:
            return
        n, k = self.out_features, self.in_features
        low_m_max = low_m_gemm_rows(n, k)
        if not _rubin_mxfp8_is_available():
            self._low_m_max, self._low_m_weight_scale = low_m_max, self.weight_scale
        elif low_m_max and self._fused_projection is None:
            self._low_m_max = low_m_max
            self._low_m_weight_scale = low_m_gemm_scale(self.weight_scale, n, k)

    def prepare_lowm_gemm(self) -> bool:
        """Build ``lowm_weight_scale`` on SM107 if TRTLLM_K3_FP8W_LOWM_GEMM is set.

        Derived from the transformed MXFP8 scale, so it follows weight reloads; rebuilt on
        every call (post-load), before CUDA graph capture. Costs N * K / 128 bytes (0.8% of
        the FP8 weight). SM100/SM103 run these GEMMs on ``weight_scale`` itself.
        """
        self.lowm_weight_scale = None
        if not (_K3_FP8W_LOWM_GEMM and _rubin_mxfp8_is_available()):
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

    def _run_lowm_gemm(
        self,
        x_flat: torch.Tensor,
        weight_scale: torch.Tensor,
        out_buffer: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """The low-M GEMM, into ``out_buffer`` when the caller handed in a usable one."""
        if (
            out_buffer is not None
            and out_buffer.dtype == torch.bfloat16
            and out_buffer.shape == (x_flat.shape[0], self.out_features)
            and out_buffer.is_contiguous()
        ):
            return k3_fp8w_lowm_gemm_op(x_flat, self.weight, weight_scale, out_buffer)
        return k3_fp8w_lowm_gemm_op(x_flat, self.weight, weight_scale)

    def _take_next_output(self, num_rows: int) -> Optional[torch.Tensor]:
        """Clear :attr:`next_output`; return it if it is a contiguous, 16-byte aligned BF16
        ``[num_rows, out_features]`` tensor."""
        out, self.next_output = self.next_output, None
        if (
            out is None
            or out.dtype != torch.bfloat16
            or tuple(out.shape) != (num_rows, self.out_features)
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            return None
        return out

    def _take_next_ar_push(self, x_flat: torch.Tensor, gated: bool = False):
        """Clear :attr:`next_ar_push`; return it if the low-M GEMM can push this forward's
        ``[M, out_features]`` output into it (low-M kernel without split-K; above the low-M
        range, on ungated rows, the mid-M GEMM's push epilogue)."""
        push, self.next_ar_push = self.next_ar_push, None
        if (
            push is None
            or x_flat.dtype != torch.bfloat16
            or tuple(push.output.shape) != (x_flat.shape[0], self.out_features)
        ):
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.ar_push_tactic(
            x_flat.shape[0], self.out_features, self.in_features, gated=gated
        )
        if tactic is None and not gated:
            tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(
                x_flat.shape[0], self.out_features, self.in_features
            )
        return push if tactic is not None else None

    def mid_push_pdl_ready(self, num_rows: int) -> bool:
        """``TRTLLM_K3_KDA_MX_PDL``: whether :meth:`forward_prequantized` on the producer's
        ``[num_rows, in_features]`` R128c4 operand will push with the mid-M GEMM (the conditions
        of :meth:`_mid_prequantized_push_tactic` that do not depend on the tensors), which reads
        nothing the producer writes before its ``griddepcontrol.wait``: the producer may then
        release it early."""
        push = self.next_ar_push
        if push is None or self.lowm_weight_scale is None:
            return False
        if num_rows > 128:
            return False
        if tuple(push.output.shape) != (num_rows, self.out_features):
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(num_rows, self.out_features, self.in_features)
        return tactic is not None and tactic.mma_n <= 128

    def _mid_prequantized_push_tactic(self, activation: torch.Tensor, scale: torch.Tensor, push):
        """The mid-M tactic :meth:`forward_prequantized` pushes the KDA verify's MXFP8 operand with,
        or None. Up to 128 rows (one scale-factor block), where the R128c4 scales of the operand are
        the mid-M GEMM's ``[K / 128][1][512 B]`` layout; the mid-M GEMM instead of the CuTe MXFP8
        GEMM, so the projection's FP32 accumulation order can differ there."""
        if push is None or self.lowm_weight_scale is None:
            return None
        if activation.dim() != 2 or not activation.is_contiguous():
            return None
        m, k = activation.shape
        if k != self.in_features or tuple(push.output.shape) != (m, self.out_features):
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(m, self.out_features, k)
        if tactic is None or tactic.mma_n > 128:
            return None
        if scale.numel() != (k // 128) * 512 or not scale.is_contiguous():
            return None
        return tactic

    def _run_ar_push(self, x_flat: torch.Tensor, push, gate=None) -> torch.Tensor:
        """The low-M GEMM with the all-reduce push epilogue; returns the all-reduce output."""
        k3_fp8w_lowm_gemm_push_op(
            x_flat, self.weight, self.packed_weight_scale, push.workspace, push.one_shot, gate=gate
        )
        push.pushed = True
        return push.output

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
        out_buffer, self.next_output = self.next_output, None
        ar_push = self._take_next_ar_push(x_flat)

        # The K3 low-M GEMM (``TRTLLM_K3_FP8W_LOWM_GEMM``) up to its per-shape M gate, on SM107
        # with ``lowm_weight_scale``, on SM100 / SM103 with ``weight_scale``; else the low-M /
        # MXFP8 (SM107) / swap-AB (SM100 / SM103) GEMM of ``layers.fp8_block_scale_gemm``.
        lowm_scale = self.lowm_weight_scale if _rubin_mxfp8_is_available() else self.weight_scale
        if (
            _K3_FP8W_LOWM_GEMM
            and lowm_scale is not None
            and self._lowm_gemm_applies(x_flat, lowm_scale)
        ):
            if ar_push is not None:
                out = self._run_ar_push(x_flat, ar_push)
            else:
                out = self._run_lowm_gemm(x_flat, lowm_scale, out_buffer)
        else:
            out = fp8_block_scale_gemm(
                x_flat,
                self.weight,
                self.weight_scale,
                low_m_max=self._low_m_max,
                low_m_scale=self._low_m_weight_scale,
            )
        return out.reshape(out_shape)

    def prepare_output_slice(self, start: int, end: int) -> None:
        """Prepare the scales of output features ``[start, end)`` for :meth:`forward_slice`.

        The shared MLA calls this at post-load for the context heads of a Helix rank. The slice
        keeps the 128-row scale grid: it covers the whole scale tiles around ``[start, end)``.
        """
        from tensorrt_llm.quantization.utils.fp8_utils import Fp8BlockScaleWeightSliceCache

        if not 0 <= start < end <= self.out_features:
            raise ValueError(f"Invalid output feature range [{start}, {end})")
        self.transform_weights()
        if self._output_slice_cache is None:
            self._output_slice_cache = Fp8BlockScaleWeightSliceCache()
        self._output_slice_cache.prepare(self.weight, self.weight_scale, start, end)

    def forward_slice(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Output features ``[start, end)`` of :meth:`forward`, contiguous.

        One GEMM over the weight rows of the slice's scale tiles (the general path of
        :meth:`forward`: the CuTe MXFP8 GEMM on SM107, ``fp8_swap_ab_gemm`` elsewhere), cropped
        to the slice.
        """
        self.prepare_output_slice(start, end)
        weight, scale, crop = self._output_slice_cache.get(self.weight, start, end)
        x_flat = x.reshape(-1, x.shape[-1])
        if _rubin_mxfp8_is_available():
            act_fp8, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x_flat)
            out = k3_mxfp8_gemm_op(act_fp8, weight, act_scale, scale)
        else:
            out = torch.ops.trtllm.fp8_swap_ab_gemm(
                x_flat, weight, scale, output_dtype=x.dtype, disable_ue8m0_cast=True
            )
        out = out.reshape(*x.shape[:-1], weight.shape[0])
        return out[..., crop : crop + end - start].contiguous()

    def _lowm_forward_scale(self, x_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """The weight scale :meth:`forward` runs the K3 low-M GEMM with on the BF16 rows
        ``x_flat`` (``lowm_weight_scale`` on SM107, ``weight_scale`` on SM100/SM103), or None
        where :meth:`forward` runs another GEMM."""
        if not _K3_FP8W_LOWM_GEMM or x_flat.dtype != torch.bfloat16:
            return None
        if self.lowm_weight_scale is not None:
            scale = self.lowm_weight_scale
        elif _rubin_mxfp8_is_available():
            return None
        else:
            scale = self.weight_scale
        return scale if self._lowm_gemm_applies(x_flat, scale) else None

    def _runs_low_m_kernel(self, num_rows: int) -> bool:
        """Whether :meth:`forward` on ``num_rows`` BF16 rows would run the low-M kernel (the
        input-independent part of its dispatch: the low-M GEMM on, not the Rubin CuTe MXFP8
        path, ``num_rows`` within the low-M kernel's range). On SM107 forward runs it on
        ``lowm_weight_scale`` up to the per-shape gate (``max_m_for``)."""
        if not _K3_FP8W_LOWM_GEMM:
            return False
        if _rubin_mxfp8_is_available() and self.lowm_weight_scale is None:
            return False
        if self.weight.shape[0] % 128 and not self.fp8w_lowm_partial_n_tile:
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        limit = k3_fp8w_lowm_gemm.low_m_max_for(*self.weight.shape)
        if self.lowm_weight_scale is not None:
            limit = min(limit, k3_fp8w_lowm_gemm.max_m_for(*self.weight.shape))
        return 1 <= num_rows <= limit

    def takes_mxfp8_input(self, num_rows: int) -> bool:
        """Whether a producer should hand :meth:`forward_mxfp8` the MXFP8 form of ``num_rows``
        rows: where :meth:`forward` would run the low-M kernel, whose in-kernel quantization the
        MXFP8 GEMM replaces (the attention's first projection also asks
        :meth:`takes_mid_mxfp8_input`)."""
        return self._runs_low_m_kernel(num_rows)

    def takes_mid_mxfp8_input(self, num_rows: int) -> bool:
        """Whether :meth:`forward` on ``num_rows`` BF16 rows runs the mid-M path of
        ``k3_fp8w_lowm_gemm`` (above the low-M cut-over, up to the per-shape gate) with one
        128-row scale block, so that a producer may hand :meth:`forward_mxfp8` the rows' MXFP8 form
        with the scales in that GEMM's layout (``k3_attn_res`` ``mx_atom``): the GEMM then runs
        without its quantization kernel, bitwise the BF16 path."""
        return self.mid_operand_tactic(num_rows) is not None

    def mid_operand_tactic(self, num_rows: int):
        """c240: the mid-M tactic :meth:`forward` runs on ``num_rows`` BF16 rows (SM107 low-M GEMM
        switch on, above the low-M cut-over, up to the per-shape gate) when it has one 128-row
        scale block, else None. A producer may then store the rows' MXFP8 form in that GEMM's
        layout (``mid_prequant_plan``) for :meth:`forward_mxfp8` / :meth:`forward_mid_mxfp8`."""
        if not _K3_FP8W_LOWM_GEMM:
            return None
        if self.lowm_weight_scale is None:
            return None
        n, k = self.weight.shape
        if n % 128 and not self.fp8w_lowm_partial_n_tile:
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        if (
            not k3_fp8w_lowm_gemm.low_m_max_for(n, k)
            < num_rows
            <= min(128, k3_fp8w_lowm_gemm.max_m_for(n, k))
        ):
            return None
        plan = k3_fp8w_lowm_gemm.mid_prequant_plan(num_rows, n, k)
        return plan[0] if plan is not None and plan[1] == 1 else None

    def forward_mid_mxfp8(
        self, x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor
    ) -> torch.Tensor:
        """``TRTLLM_K3_MID_OPROJ_PC`` (c240): :meth:`forward` on rows whose MXFP8 form a producer
        stored with the scales in the mid-M GEMM's layout (``xs`` 1-D, see
        :meth:`mid_operand_tactic`; ``x`` only gives the output shape): the mid-M GEMM without
        its quantization kernel, with the all-reduce push epilogue where :attr:`next_ar_push`
        takes it. Bitwise :meth:`forward` on the rows the producer quantized."""

        m = xq.shape[0]
        tactic = self.mid_operand_tactic(m)
        if tactic is None or xs.dim() != 1:
            raise RuntimeError("forward_mid_mxfp8 needs the mid-M operand layout")
        x_flat = x.reshape(-1, x.shape[-1])
        push = self._take_next_ar_push(x_flat)
        self.next_output = None
        if push is not None:
            k3_fp8w_mid_gemm_push_op(
                xq.reshape(m, -1),
                xs,
                self.weight,
                self.lowm_weight_scale,
                push.workspace,
                push.one_shot,
                tactic,
            )
            push.pushed = True
            return push.output
        out = torch.empty((m, self.out_features), dtype=torch.bfloat16, device=xq.device)
        k3_fp8w_mid_gemm_op(xq.reshape(m, -1), xs, self.weight, self.lowm_weight_scale, out, tactic)
        return out.reshape(x.shape[:-1] + (self.out_features,))

    def _staged_above_cutover(self, num_rows: int) -> bool:
        """Whether ``forward_mxfp8(..., stage_weights=True)`` serves ``num_rows`` rows above the
        low-M kernel's cut-over with the staged MXFP8 low-M GEMM: :meth:`forward` runs the mid-M
        path of ``k3_fp8w_lowm_gemm`` there, whose bits the staged kernel reproduces
        (``k3_fp8w_lowm_mxfp8.staged_matches_mid_m``)."""
        if not _K3_FP8W_LOWM_GEMM or self.lowm_weight_scale is None:
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8_ull as k3_fp8w_lowm_mxfp8

        n, k = self.weight.shape
        if num_rows > k3_fp8w_lowm_gemm.max_m_for(n, k):
            return False
        return k3_fp8w_lowm_mxfp8.staged_matches_mid_m(num_rows, n, k)

    def takes_staged_mxfp8_input(self, num_rows: int) -> bool:
        """Whether ``forward_mxfp8(..., stage_weights=True)`` runs the staged MXFP8 low-M GEMM on
        the MXFP8 form of ``num_rows`` rows (bitwise :meth:`forward`): where :meth:`forward` runs
        the low-M kernel, and above its cut-over where it runs the mid-M path with the same
        split-K (up to 64 rows)."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8_ull as k3_fp8w_lowm_mxfp8

        n, k = self.weight.shape
        if self._runs_low_m_kernel(num_rows):
            # Also where the shared-memory ring alone holds the CTA's whole weight slice.
            return k3_fp8w_lowm_mxfp8.weights_on_chip(num_rows, n, k)
        return self._staged_above_cutover(num_rows)

    def forward_mxfp8(
        self, x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor, stage_weights: bool = False
    ) -> torch.Tensor:
        """:meth:`forward` on the BF16 rows ``x``, bit for bit, given their MXFP8 form ``(xq,
        xs)`` (E4M3 ``[M, K]`` and one UE8M0 exponent per 1x32 group, e.g. from
        the ``k3_attn_res_norm`` MXFP8 variants): the low-M GEMM on the prequantized operand where
        :meth:`forward` would run the low-M kernel, else :meth:`forward` on ``x``.
        ``stage_weights``: the low-M GEMM keeps its whole weight slice on chip before its
        ``griddepcontrol.wait`` (``k3_fp8w_lowm_mxfp8.staged_mx_tactic``, the same bits), also
        above the low-M cut-over where :meth:`takes_staged_mxfp8_input` says so."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8_ull as k3_fp8w_lowm_mxfp8

        x_flat = x.reshape(-1, x.shape[-1])
        if xs.dim() == 1 and self.takes_mid_mxfp8_input(x_flat.shape[0]):
            # The producer stored the scales in the mid-M GEMM's layout.
            from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

            m = x_flat.shape[0]
            tactic, _ = k3_fp8w_lowm_gemm.mid_prequant_plan(m, self.out_features, self.in_features)
            self.next_ar_push = None
            out = self._take_next_output(m)
            if out is None:
                out = torch.empty((m, self.out_features), dtype=torch.bfloat16, device=x.device)
            k3_fp8w_mid_gemm_op(
                xq.reshape(m, -1), xs, self.weight, self.lowm_weight_scale, out, tactic
            )
            return out.reshape(x.shape[:-1] + (self.out_features,))
        allow_partial = self.fp8w_lowm_partial_n_tile
        # The scale forward's low-M branch uses (lowm_weight_scale on SM107).
        w_scale = self.packed_weight_scale
        above_cutover = (
            stage_weights
            and not self._runs_low_m_kernel(x_flat.shape[0])
            and self._staged_above_cutover(x_flat.shape[0])
        )
        if not (
            x.dtype == torch.bfloat16
            and (above_cutover or self._runs_low_m_kernel(x_flat.shape[0]))
            and xq.shape[0] == x_flat.shape[0]
            and _k3_lowm_gemm_can_run(
                x_flat, self.weight, w_scale, allow_partial_n_tile=allow_partial
            )
            and k3_fp8w_lowm_mxfp8.can_run_mxfp8(
                xq,
                xs,
                self.weight,
                w_scale,
                allow_partial_n_tile=allow_partial,
                above_cutover=above_cutover,
            )
        ):
            return self.forward(x)
        out = self._take_next_output(x_flat.shape[0])
        # The MXFP8 GEMM has no all-reduce push epilogue: the caller runs the all-reduce.
        self.next_ar_push = None
        if out is None:
            out = torch.empty(
                (x_flat.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device
            )
        stage_weights = stage_weights and k3_fp8w_lowm_mxfp8.weights_on_chip(
            x_flat.shape[0], self.out_features, self.in_features
        )
        k3_fp8w_lowm_mxfp8_gemm_op(xq, xs, self.weight, w_scale, out, stage_weights=stage_weights)
        return out.reshape(x.shape[:-1] + (self.out_features,))

    def gated_applies(self, x: torch.Tensor, gate: torch.Tensor) -> bool:
        """Whether :meth:`forward_gated` serves ``x`` and ``gate`` with one gated low-M GEMM.

        That is where :meth:`forward` would run the low-M GEMM on ``x * gate`` (same kernel
        family and split), so the gated launch is bitwise equal to the multiply plus
        :meth:`forward`.
        """
        if x.dim() != 2 or not x.is_contiguous():
            return False
        scale = self._lowm_forward_scale(x)
        if scale is None:
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        return k3_fp8w_lowm_gemm.can_run_gated(
            x,
            gate,
            self.weight,
            scale,
            allow_partial_n_tile=self.fp8w_lowm_partial_n_tile,
        )

    def forward_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """``forward(x).sigmoid()``, bit for bit; one low-M GEMM launch with a sigmoid
        epilogue where :meth:`forward` would run the low-M GEMM."""
        x_flat = x.reshape(-1, x.shape[-1])
        scale = self._lowm_forward_scale(x_flat)
        if scale is not None:
            from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

            if x_flat.shape[0] <= k3_fp8w_lowm_gemm.low_m_max_for(
                self.out_features, self.in_features
            ):
                self.next_output = None
                self.next_ar_push = None
                out = k3_fp8w_lowm_gemm_op(x_flat, self.weight, scale, sigmoid=True)
                return out.reshape(x.shape[:-1] + (self.out_features,))
        return self.forward(x).sigmoid()

    def forward_gated(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``forward(x * gate)`` in one gated low-M GEMM launch (where :meth:`gated_applies`).

        ``x`` and ``gate`` are BF16 ``[M, in_features]``; the gate may be a row-strided view.
        """
        ar_push = self._take_next_ar_push(x, gated=True)
        if ar_push is not None:
            self.next_output = None
            return self._run_ar_push(x, ar_push, gate=gate)
        return k3_fp8w_lowm_gemm_op(
            x,
            self.weight,
            self.packed_weight_scale,
            out=self._take_next_output(x.shape[0]),
            gate=gate,
        )

    def gated_mxfp8_applies(self, x: torch.Tensor, gate: torch.Tensor) -> bool:
        """Whether :meth:`forward_gated_mxfp8` runs the MXFP8 low-M GEMM on the producer's MXFP8
        form of ``bf16(x * gate)``: where :meth:`forward_gated` runs
        the gated low-M GEMM, whose tiling the MXFP8 kernel shares (same bits)."""
        if not self.gated_applies(x, gate):
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8_ull as k3_fp8w_lowm_mxfp8

        m, k = x.shape
        try:
            k3_fp8w_lowm_mxfp8.default_mx_tactic(m, self.out_features, k)
        except ValueError:
            return False
        return k % 32 == 0

    def forward_gated_mxfp8(
        self, x: torch.Tensor, gate: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor
    ) -> torch.Tensor:
        """:meth:`forward_gated` ``(x, gate)``, bit for bit, given the MXFP8 form ``(xq, xs)`` of
        ``bf16(x * gate)`` (E4M3 ``[M, K]`` and one UE8M0 exponent per 1x32 group, the gated
        low-M GEMM's own quantization): the MXFP8 low-M GEMM, with the all-reduce push epilogue
        where :attr:`next_ar_push` is set, so the activation path after its
        ``griddepcontrol.wait`` is two TMA loads instead of the BF16 loads, the gate multiply and
        the quantization."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8_ull as k3_fp8w_lowm_mxfp8

        w_scale = self.packed_weight_scale
        if w_scale is None or not k3_fp8w_lowm_mxfp8.can_run_mxfp8(xq, xs, self.weight, w_scale):
            return self.forward_gated(x, gate)
        ar_push = self._take_next_ar_push(x, gated=True)
        if ar_push is not None:
            self.next_output = None
            k3_fp8w_lowm_mxfp8_push_op(
                xq, xs, self.weight, w_scale, ar_push.workspace, ar_push.one_shot
            )
            ar_push.pushed = True
            return ar_push.output
        out = self._take_next_output(x.shape[0])
        if out is None:
            out = torch.empty(
                (x.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device
            )
        k3_fp8w_lowm_mxfp8_gemm_op(xq, xs, self.weight, w_scale, out)
        return out

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
        push, self.next_ar_push = self.next_ar_push, None
        if activation.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized Kimi K3 activation must be float8_e4m3fn")
        if not self.supports_prequantized_input:
            raise RuntimeError("prequantized Kimi K3 activation requires the Rubin CuTe GEMM")
        if activation_scale.dtype is not torch.uint8:
            raise TypeError("prequantized Kimi K3 activation scale must be uint8 R128c4")
        rows = activation.reshape(-1, activation.shape[-1])
        tactic = self._mid_prequantized_push_tactic(rows, activation_scale, push)
        if tactic is not None:
            # The mid-M GEMM pushes into the o_proj all-reduce.
            k3_fp8w_mid_gemm_push_op(
                rows,
                activation_scale.reshape(-1),
                self.weight,
                self.lowm_weight_scale,
                push.workspace,
                push.one_shot,
                tactic,
            )
            push.pushed = True
            return push.output
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


class Step(NamedTuple):
    """One generation step: ``rows`` token rows of ``requests`` requests, ``q`` each."""

    rows: int
    requests: int
    q: int


def plan_step(attn_metadata: AttentionMetadata, rows: int, chain=None) -> Step:
    """The step of a forward on ``rows`` token rows; raises where the target does not serve it
    (context requests, ragged verify lengths, or ``(rows, q)`` outside ``chain.supports``)."""
    if attn_metadata.num_contexts != 0:
        raise RuntimeError(
            f"Kimi K3 ULL serves generation-only steps; got {attn_metadata.num_contexts} context "
            "requests (this target runs on the generation server of a disaggregated deployment)"
        )
    requests = int(attn_metadata.num_seqs)
    if requests <= 0 or rows % requests:
        raise RuntimeError(
            f"Kimi K3 ULL: {rows} token rows over {requests} generation requests (each request "
            "verifies the same number of tokens)"
        )
    q = rows // requests
    if chain is not None and not chain.supports(rows, q):
        raise RuntimeError(
            f"Kimi K3 ULL: {requests} requests x {q} tokens is not a step the decode chain "
            f"serves (rows {tuple(chain.rows)}, q {chain.q})"
        )
    return Step(rows, requests, q)


class K3UllMambaMetadata(Mamba2Metadata):
    """``Mamba2Metadata`` with the generation requests' state slots as an int32 16-byte aligned
    tensor (``generation_state_indices``; a view of ``state_indices`` where that is aligned, else
    a copy into a fixed buffer, so CUDA graphs keep the address)."""

    def __init__(self, max_batch_size: int, chunk_size: int, max_num_tokens: int) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens)
        self._generation_state_indices = torch.empty(
            max_batch_size, dtype=torch.int32, device="cuda"
        )
        self.generation_state_indices: Optional[torch.Tensor] = None

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        super().prepare(attn_metadata)
        batch_size = attn_metadata.seq_lens.shape[0]
        lo = attn_metadata.num_contexts
        indices = self.state_indices[lo:batch_size]
        if indices.numel() > 0 and (
            indices.data_ptr() % 16 or indices.dtype != torch.int32 or indices.stride(-1) != 1
        ):
            aligned = self._generation_state_indices[: indices.shape[0]]
            if self._state_indices_aliased_ptr is None:
                # The indices came from the host (``state_indices_cpu``): upload the same values,
                # in the step's input pack when one is open (it lands at the graph replay).
                step_inputs.copy_(aligned, self.state_indices_cpu[lo:batch_size])
            else:
                aligned.copy_(indices)
            indices = aligned
        self.generation_state_indices = indices


def native_process_group(rank: int, world: int = 16):
    """The default NCCL process group of the ``world`` generation ranks (``rank`` this one's),
    created once: MPI rank 0 picks a TCP rendezvous and broadcasts it. Reused when the default
    group already exists with the same rank and size."""
    import torch.distributed as dist

    from tensorrt_llm._utils import mpi_comm

    if dist.is_initialized():
        if dist.get_world_size() != world or dist.get_rank() != rank:
            raise ValueError(
                f"Kimi K3 ULL: the default process group is rank {dist.get_rank()} of "
                f"{dist.get_world_size()}, not {rank} of {world}"
            )
        return dist.group.WORLD
    comm = mpi_comm()
    if comm.Get_size() != world or comm.Get_rank() != rank:
        raise ValueError(
            f"Kimi K3 ULL: the MPI world is rank {comm.Get_rank()} of {comm.Get_size()}, not "
            f"{rank} of {world}"
        )
    rendezvous = None
    if rank == 0:
        try:
            host = socket.gethostbyname(socket.gethostname())
            if host.startswith("127.") and world > 1:
                raise ValueError("the NCCL rendezvous must be reachable from the other nodes")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            rendezvous = (host, port, None)
        except (OSError, ValueError) as error:
            rendezvous = (None, None, str(error))
    host, port, error = comm.bcast(rendezvous, root=0)
    if error is not None:
        raise RuntimeError(f"Kimi K3 ULL: process group rendezvous failed: {error}")
    timeout = timedelta(minutes=5)
    store = dist.TCPStore(host, port, world, rank == 0, timeout=timeout)
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        store=store,
        rank=rank,
        world_size=world,
        timeout=timeout,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    return dist.group.WORLD


__all__ = ["K3UllMambaMetadata", "Step", "native_process_group", "plan_step"]

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

# The 1007804 values of the environment the served path reads (unset there): TRTLLM_ENABLE_PDL
# (default on; routing.py declines the target otherwise) and TRTLLM_FORCE_MNNVL_AR (default off).
_PDL = True
_FORCE_MNNVL_AR = False

# The drafter's FP8 context: E4M3 at a unit per-tensor scale, saturated (served
# ``dflash_attention.DRAFT_FP8_KV_SCALE`` / ``to_draft_fp8``).
DRAFT_FP8_KV_SCALE = 1.0
DRAFT_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def to_draft_fp8(x: torch.Tensor) -> torch.Tensor:
    """``x`` as saturated FP8 E4M3 at the drafter's unit per-tensor scale."""
    return x.clamp(-DRAFT_FP8_MAX, DRAFT_FP8_MAX).to(torch.float8_e4m3fn)


@dataclass(frozen=True)
class HelixDraftShard:
    """One rank's share of a drafter's paged context KV under helix.

    Global page ``g`` of a request belongs to CP rank ``g % cp_size``, which
    stores its pages back to back: local page ``g // cp_size``. Every helper
    maps global token positions or lengths into that rank-local layout.
    """

    cp_size: int
    cp_rank: int
    page_size: int

    def local_slots(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank-local slot of each global position, and whether this rank owns it.

        An unowned position maps to the slot this rank keeps for the token one
        or more pages away, never one of the ``page_size`` tokens around it.
        """
        page = torch.div(positions, self.page_size, rounding_mode="floor")
        slots = torch.div(page, self.cp_size, rounding_mode="floor") * self.page_size
        return slots + positions % self.page_size, page % self.cp_size == self.cp_rank

    def local_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Number of owned positions below each global length."""
        span = self.page_size * self.cp_size
        rem = lengths % span - self.cp_rank * self.page_size
        return torch.div(lengths, span, rounding_mode="floor") * self.page_size + rem.clamp(
            0, self.page_size
        )


_NUM_LAMPORT_BUFFERS = 3


class PrivateMNNVLAllReduce:
    """``MNNVLAllReduce`` over a private workspace (for a side stream).

    Construction is collective over the mapping's TP group.
    """

    def __init__(self, mapping: Mapping, dtype: torch.dtype, max_tokens: int, hidden: int):
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        self.mapping = mapping
        self.dtype = dtype
        elem = torch.tensor([], dtype=dtype).element_size()
        need = mnnvl_workspace_bytes(max_tokens, hidden, mapping.tp_size, dtype)
        self.buffer_size_bytes = math.ceil(max(need, 1) / (8 * 1024 * 1024)) * (8 * 1024 * 1024)
        self.max_tokens = max_tokens
        self.hidden = hidden
        self._elem = elem
        workspace_bytes = _NUM_LAMPORT_BUFFERS * self.buffer_size_bytes
        comm = mpi_comm().Split(
            int(mapping.pp_rank * mapping.cp_size + mapping.cp_rank), mapping.tp_rank
        )
        self._handle = McastGPUBuffer(
            workspace_bytes,
            mapping.tp_size,
            mapping.tp_rank,
            mapping.local_rank,
            _FORCE_MNNVL_AR or mapping.is_multi_node(),
            comm.py2f(),
        )
        buffer = self._handle.get_uc_buffer(
            mapping.tp_rank, (workspace_bytes // torch.float32.itemsize,), torch.float32, 0
        )
        buffer.fill_(-0.0)
        torch.cuda.synchronize()
        comm.Barrier()
        self._comm = comm
        self._buffer_base = buffer.view(dtype).view(_NUM_LAMPORT_BUFFERS, -1)
        # [cur idx, dirty idx, bytes per buffer, dirty num stages, numBytesToClear[4], access count]
        self._buffer_flags = torch.tensor(
            [0, 2, self.buffer_size_bytes, 0, 0, 0, 0, 0, 0],
            dtype=torch.uint32,
            device=torch.device("cuda", mapping.local_rank),
        )

    def fits(self, num_tokens: int, hidden: int) -> bool:
        need = mnnvl_workspace_bytes(num_tokens, hidden, self.mapping.tp_size, self.dtype)
        return need <= self.buffer_size_bytes

    def rms_norm(self, partial: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
        """``RMSNorm(all_reduce(partial))``, as ``MNNVLAllReduce`` with ``RMS_NORM``."""
        assert partial.dtype == self.dtype and self.fits(
            partial.numel() // partial.shape[-1], partial.shape[-1]
        )
        outputs = torch.ops.trtllm.mnnvl_fusion_allreduce(
            partial,
            gamma,
            None,
            eps,
            self._buffer_base,
            self._buffer_flags,
            True,
            None,
            int(AllReduceFusionOp.RMS_NORM),
        )
        return outputs[0]


class _PairExchange:
    """Exchanges ``(value, global index)`` float32 pairs per row across TP.

    Over MNNVL each rank owns ``splits`` consecutive pair slots of a zero-padded
    ``[rows, width]`` buffer that an all-reduce sums; over NCCL each rank
    publishes one pair per row and the pairs are all-gathered.
    """

    def __init__(self, mapping: Mapping):
        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self._mnnvl = None
        if self.tp_size > 1:
            if MNNVLAllReduce.is_mnnvl(mapping, torch.float32):
                try:
                    self._mnnvl = MNNVLAllReduce(mapping, torch.float32)
                except (RuntimeError, ValueError) as e:
                    logger.info(
                        f"Vocab-parallel pair exchange: MNNVL unavailable ({e}); using NCCL."
                    )
        self.backend = "mnnvl" if self._mnnvl is not None else "nccl"

    def markov_v3_buffers(self, device, entries: int):
        """``(local address, multicast address, state)`` of the CuTe DSL Markov
        chain's key exchange for ``entries`` (ranks x CTAs per rank): its
        rotating sets, zero-filled, and this rank's int32 launch state. Allocated
        collectively on first use, outside graph capture. A single rank uses one
        local buffer for both addresses."""
        cached = getattr(self, "_markov_v3", None)
        if cached is not None and cached[0] == entries:
            return cached[1:4]
        assert cached is None, "the Markov exchange is sized once per process"
        assert not torch.cuda.is_current_stream_capturing()
        from ..kernels.k3r_spec.dspark_markov_greedy import NUM_SETS, STATE_WORDS, set_bytes

        words = NUM_SETS * set_bytes(entries) // 4
        state = torch.zeros((STATE_WORDS,), dtype=torch.int32, device=device)
        if self.tp_size == 1:
            buf = torch.zeros((words,), dtype=torch.int32, device=device)
            self._markov_v3 = (entries, buf.data_ptr(), buf.data_ptr(), state, buf)
            return self._markov_v3[1:4]
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        assert self.backend == "mnnvl"
        comm = MNNVLAllReduce.allreduce_mnnvl_workspaces[self.mapping]["comm"]
        handle = McastGPUBuffer(
            words * 4,
            self.tp_size,
            self.tp_rank,
            self.mapping.local_rank,
            _FORCE_MNNVL_AR or self.mapping.is_multi_node(),
            comm.py2f(),
        )
        local = handle.get_uc_buffer(self.tp_rank, [words], torch.int32, 0)
        mcast = handle.get_mc_buffer([words], torch.int32, 0)
        local.zero_()
        torch.cuda.synchronize()
        comm.Barrier()
        self._markov_v3 = (entries, local.data_ptr(), mcast.data_ptr(), state, handle)
        return self._markov_v3[1:4]

    def layout(self, rows: int, n_cols: int, device) -> _PairLayout:
        # Split each row over enough programs to fill the GPU when rows are few;
        # the per-split pairs travel as extra slots of the same exchange.
        max_splits = triton.cdiv(n_cols, _LOCAL_BLOCK_V)
        splits = 1
        if self.backend == "mnnvl" and self.tp_size > 1:
            splits = max(1, min(max_splits, triton.cdiv(_TARGET_PROGRAMS, max(rows, 1))))
        chunk = triton.cdiv(triton.cdiv(n_cols, splits), _LOCAL_BLOCK_V) * _LOCAL_BLOCK_V
        splits = max(1, triton.cdiv(n_cols, chunk))
        if self.tp_size == 1 or self.backend == "nccl":
            buf = torch.empty((rows, 2 * splits), dtype=torch.float32, device=device)
            return _PairLayout(buf, 2 * splits, 0, 0, 0, 0, splits, chunk)
        width = triton.next_power_of_2(2 * self.tp_size * splits)
        buf = torch.empty((rows, width), dtype=torch.float32, device=device)
        lo = 2 * self.tp_rank * splits
        return _PairLayout(
            buf, width, self.tp_rank * splits, lo, lo + 2 * splits, width, splits, chunk
        )

    def exchange(self, layout: _PairLayout) -> tuple[torch.Tensor, int, int, int]:
        """Every rank's pairs: (tensor, row stride, slot stride, number of slots)."""
        buf = layout.buffer
        rows = buf.shape[0]
        if self.tp_size == 1:
            return buf, layout.stride_out, 2, layout.splits
        if self.backend == "mnnvl":
            out = self._mnnvl(buf, all_reduce_params=AllReduceParams())
            return out, layout.stride_out, 2, self.tp_size * layout.splits
        assert layout.splits == 1
        out = allgather(buf, self.mapping, dim=0)
        return out, 2, 2 * rows, self.tp_size


class VocabParallelHead:
    """Vocab-parallel token selection over one column-parallel LM head.

    Args:
        lm_head: an ``LMHead`` (or ``Linear``) in COLUMN tensor-parallel mode
            whose weight holds this rank's contiguous vocab shard.
        vocab_size: the full vocabulary size.
    """

    def __init__(self, lm_head, vocab_size: int):
        self.lm_head = lm_head
        self.mapping = lm_head.mapping
        self.vocab_size = int(vocab_size)
        self.shard_width = int(lm_head.out_features)
        self.col_base = self.mapping.tp_rank * self.shard_width
        self.exchange = _PairExchange(self.mapping)
        self._markov_w2_shard = None
        self._markov_w2_src_ptr = None
        logger.info(
            f"Vocab-parallel LM-head token selection: tp={self.mapping.tp_size}, "
            f"shard={self.shard_width}, vocab={self.vocab_size}, exchange={self.exchange.backend}"
        )

    @staticmethod
    def supported(lm_head, vocab_size: int) -> bool:
        """Whether ``lm_head`` is a plain vocab-sharded head this class handles."""
        from ..common.parallel import TensorParallelMode

        mapping = getattr(lm_head, "mapping", None)
        if mapping is None or mapping.tp_size <= 1 or mapping.enable_attention_dp:
            return False
        if getattr(lm_head, "tp_mode", None) != TensorParallelMode.COLUMN:
            return False
        if not getattr(lm_head, "gather_output", False):
            return False
        if getattr(lm_head, "padding_size", 0) != 0 or getattr(lm_head, "has_any_quant", False):
            return False
        # The sampler draws Philox noise for 4-column groups of the global vocab.
        shard = int(lm_head.out_features)
        return shard % 4 == 0 and shard * mapping.tp_size == int(vocab_size) and vocab_size < 2**24

    def local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's ``[rows, shard]`` logits in the head's output dtype."""
        return self.lm_head.apply_linear(hidden_states, self.lm_head.bias)

    def shard_logits(self, hidden_states: torch.Tensor) -> "VocabShardedLogits":
        """This rank's logits wrapped for a spec worker (see ``VocabShardedLogits``)."""
        return VocabShardedLogits(self, self.local_logits(hidden_states))

    def gather_full_logits(self, local: torch.Tensor) -> torch.Tensor:
        """Full float32 logits, identical to the gathered head + ``.float()``."""
        return allgather(local, self.mapping, sizes=self.lm_head.gather_output_sizes).float()

    def _launch_local(self, kernel, local: torch.Tensor, extra_args: tuple) -> _PairLayout:
        rows, n_cols = local.shape
        layout = self.exchange.layout(rows, n_cols, local.device)
        if rows > 0:
            kernel[(rows, layout.splits)](
                local,
                local.stride(0),
                n_cols,
                self.col_base,
                layout.chunk,
                *extra_args,
                layout.buffer,
                layout.stride_out,
                layout.slot0,
                layout.zero_lo,
                layout.zero_hi,
                ZERO_WIDTH=layout.zero_width,
                BLOCK_V=_LOCAL_BLOCK_V,
                num_warps=8,
            )
        return layout

    def _exchange_and_reduce(self, layout: _PairLayout, rows: int, out: torch.Tensor):
        pairs, stride_row, stride_slot, n_slots = self.exchange.exchange(layout)
        if rows > 0:
            _reduce_pairs_kernel[(triton.cdiv(rows, _REDUCE_BLOCK_ROWS),)](
                pairs,
                stride_row,
                stride_slot,
                rows,
                n_slots,
                out,
                BLOCK_ROWS=_REDUCE_BLOCK_ROWS,
                BLOCK_SLOTS=triton.next_power_of_2(n_slots),
                num_warps=4,
            )
        return out

    def selection_pairs(self, local: torch.Tensor, spec_metadata) -> tuple:
        """Every rank's ``(value, global index)`` pairs of the target selection,
        not yet reduced: ``(pairs, row stride, slot stride, slots)``.

        The same local kernel and exchange as :meth:`VocabShardedLogits.select_tokens`
        (argmax for an all-greedy batch, else the Gumbel-max draw), for a caller
        that reduces the pairs inside a fused consumer kernel.
        """
        assert local.dim() == 2 and local.stride(-1) == 1
        if spec_metadata.is_all_greedy_sample:
            layout = self._launch_local(_local_argmax_kernel, local, ())
        else:
            from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
                GREEDY_TEMPERATURE_THRESHOLD,
            )

            rows = local.shape[0]
            layout = self._launch_local(
                _local_sample_kernel,
                local,
                (
                    spec_metadata.temperatures[:rows],
                    spec_metadata.seeds[:rows],
                    spec_metadata.offsets[:rows],
                    float(GREEDY_TEMPERATURE_THRESHOLD),
                ),
            )
        return self.exchange.exchange(layout)

    def argmax(self, local: torch.Tensor, out_dtype=torch.int64) -> torch.Tensor:
        """Global argmax over the vocab of shard logits ``[rows, shard]``."""
        assert local.dim() == 2 and local.stride(-1) == 1
        layout = self._launch_local(_local_argmax_kernel, local, ())
        out = torch.empty((local.shape[0],), dtype=out_dtype, device=local.device)
        return self._exchange_and_reduce(layout, local.shape[0], out)

    def sample(
        self,
        local: torch.Tensor,
        temperatures: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        greedy_threshold: float,
        out_dtype=torch.int64,
    ) -> torch.Tensor:
        """Temperature-only sampling (Gumbel-max); rows at or below the greedy
        temperature threshold take the plain argmax."""
        assert local.dim() == 2 and local.stride(-1) == 1
        layout = self._launch_local(
            _local_sample_kernel, local, (temperatures, seeds, offsets, float(greedy_threshold))
        )
        out = torch.empty((local.shape[0],), dtype=out_dtype, device=local.device)
        return self._exchange_and_reduce(layout, local.shape[0], out)

    def markov_v3_supported(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: Optional[torch.Tensor],
        markov_w2: Optional[torch.Tensor],
    ) -> bool:
        """Whether :meth:`markov_tokens_v3` serves these operands: BF16 logits
        and Markov weights (rank a multiple of 16, at most 256), at most 16
        requests and 16 block positions, an SM100-family device with a CTA per
        256 shard rows resident at once, and the MNNVL (or single-rank)
        exchange."""
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        if not IS_CUTLASS_DSL_AVAILABLE or markov_w1 is None or markov_w2 is None:
            return False
        if self.exchange.tp_size > 1 and self.exchange.backend != "mnnvl":
            return False
        num_reqs, block_size, shard = local_block.shape
        rank_dim = markov_w1.shape[1]
        device = local_block.device
        if not (1 <= num_reqs <= 16 and 1 <= block_size <= 16) or shard != self.shard_width:
            return False
        if (
            local_block.dtype != torch.bfloat16
            or markov_w1.dtype != torch.bfloat16
            or markov_w2.dtype != torch.bfloat16
            or local_block.stride(-1) != 1
            or not markov_w1.is_contiguous()
            or not markov_w2.is_contiguous()
            or markov_w2.shape[1] != rank_dim
            or rank_dim % 16
            or rank_dim > 256
            or first_prev_tokens.dtype not in (torch.int32, torch.int64)
            or first_prev_tokens.stride(0) != 1
            or markov_w2.data_ptr() % 16
        ):
            return False
        from ..kernels.k3r_spec.dspark_markov_greedy import ctas_for, smem_bytes

        props = torch.cuda.get_device_properties(device)
        return (
            props.major == 10
            and ctas_for(shard) <= props.multi_processor_count
            and smem_bytes(rank_dim) <= props.shared_memory_per_block_optin
        )

    def markov_tokens_v3(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
    ) -> torch.Tensor:
        """:meth:`markov_tokens_chain` as one CuTe DSL launch without the
        intra-GPU reduction.

        Each CTA keeps its 256 ``markov_w2`` shard rows in shared memory, adds
        the BF16-rounded bias of every request to the step's shard logits in
        FP32 (the arithmetic of the other chains) and stores its own
        (value, global index) key into every rank's exchange; every CTA polls
        all keys of the step (``dspark_markov_greedy``). Same tokens as
        :meth:`markov_tokens_chain` wherever both compute the same BF16 bias.
        The caller checks :meth:`markov_v3_supported`.
        """
        from ..kernels.k3r_spec.dspark_markov_greedy import ctas_for, run_dspark_markov_greedy

        num_reqs, block_size, shard = local_block.shape
        device = local_block.device
        tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
        tp = max(self.exchange.tp_size, 1)
        box_uc, box_mc, state = self.exchange.markov_v3_buffers(device, tp * ctas_for(shard))
        run_dspark_markov_greedy(
            local_block,
            first_prev_tokens,
            markov_w1,
            self.markov_w2_shard(markov_w2),
            tokens,
            vocab=self.vocab_size,
            tp=tp,
            rank=self.mapping.tp_rank if tp > 1 else 0,
            state=state,
            box_uc=box_uc,
            box_mc=box_mc,
            multicast=tp > 1,
        )
        return tokens

    def markov_w2_shard(self, markov_w2: torch.Tensor) -> torch.Tensor:
        """This rank's rows of ``markov_w2`` (cached; a contiguous row slice)."""
        if self._markov_w2_src_ptr != markov_w2.data_ptr():
            self._markov_w2_shard = markov_w2[self.col_base : self.col_base + self.shard_width]
            self._markov_w2_src_ptr = markov_w2.data_ptr()
        return self._markov_w2_shard


class VocabShardedLogits:
    """This rank's vocab shard of the target logits, handed to a spec worker in
    place of the gathered float32 logits.

    The worker either selects tokens from it vocab-parallel
    (:meth:`select_tokens`) or, for a batch the vocab-parallel samplers do not
    cover, materializes the gathered logits with :meth:`gather`.
    """

    __slots__ = ("head", "local")

    def __init__(self, head: VocabParallelHead, local: torch.Tensor):
        self.head = head
        self.local = local

    @property
    def device(self) -> torch.device:
        return self.local.device

    def gather(self) -> torch.Tensor:
        """The gathered ``[rows, vocab]`` float32 logits."""
        return self.head.gather_full_logits(self.local)

    def select_pairs(self, spec_metadata) -> tuple:
        """The unreduced exchanged pairs of :meth:`select_tokens` (see
        :meth:`VocabParallelHead.selection_pairs`)."""
        return self.head.selection_pairs(self.local, spec_metadata)

    def select_tokens(self, spec_metadata) -> torch.Tensor:
        """One target token per row: the argmax for an all-greedy batch, else a
        temperature-only draw (the caller guarantees the sampling mode)."""
        if spec_metadata.is_all_greedy_sample:
            return self.head.argmax(self.local)
        from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

        rows = self.local.shape[0]
        return self.head.sample(
            self.local,
            spec_metadata.temperatures[:rows],
            spec_metadata.seeds[:rows],
            spec_metadata.offsets[:rows],
            GREEDY_TEMPERATURE_THRESHOLD,
        )


_PINS = {
    # [q_a | kv_a]: 2112 x 7168
    "q_kv_a": (
        (8, "splitk", (64, 16, 4, 8)),
        (16, "splitk", (64, 16, 4, 9)),
        (32, "splitk", (64, 16, 2, 11)),
    ),
    # q_b: 768 x 1536 per rank
    "q_b": ((16, "splitk", (64, 8, 4, 10)), (32, "splitk", (64, 16, 3, 10))),
    # gate_up: 1792 x 7168 per rank
    "gate_up": ((16, "splitk", (64, 16, 4, 8)), (32, "splitk", (64, 32, 4, 8))),
    # down: 7168 x 896 per rank
    "down": ((32, "cublas", None),),
    # context latents of all layers: 2880 x 7168
    "ctx_latent": (
        (8, "splitk", (128, 8, 4, 6)),
        (16, "splitk", (64, 16, 2, 10)),
        (32, "splitk", (64, 32, 2, 9)),
    ),
    # the draft block's LM head shard: 10240 x 7168
    "lm_head": ((16, "splitk", (128, 16, 1, 6)), (32, "splitk", (128, 32, 1, 5))),
}

_PINS_SM107 = {
    # [q_a | kv_a] (replicated; scripts/draftfuse/bench_gemm.py, job 2706844): 6.46 / 6.48 / 7.17
    # -> 3.95 / 4.07 / 4.95.
    (2112, 7168): (
        (8, "splitk", (64, 8, 4, 12)),
        (16, "splitk", (64, 16, 4, 10)),
        (32, "splitk", (64, 32, 4, 8)),
    ),
    # context latents of all layers (replicated; job 2706844): 6.79 / 7.45 / 7.15 -> 4.61 / 4.71 /
    # 5.38.
    (2880, 7168): (
        (8, "splitk", (64, 8, 2, 12)),
        (16, "splitk", (64, 16, 4, 10)),
        (32, "splitk", (64, 32, 4, 8)),
    ),
    # The MLP and LM head shards of the TP16 group (scripts/ull_shell/bench_draft_gemm.py
    # BG_TP=16, hecate job 834542, HBM locked): gate_up 6.02 / 6.11 / 5.72 -> 3.66 / 3.86 / 4.54;
    # down 4.22 / 3.31 / 3.37 -> 2.50 / 2.48 / 2.60; LM head 15.65 / 14.56 / 16.09 -> 14.12 /
    # 14.14 / 14.45.
    # gate_up: 2 x 896
    (1792, 7168): (
        (8, "splitk", (64, 8, 4, 12)),
        (16, "splitk", (64, 16, 4, 10)),
        (32, "splitk", (64, 32, 4, 8)),
    ),
    # down
    (7168, 896): (
        (8, "splitk", (64, 8, 1, 12)),
        (16, "splitk", (64, 16, 1, 11)),
        (32, "splitk", (64, 32, 1, 9)),
    ),
    # the draft block's LM head shard
    (10240, 7168): (
        (8, "splitk", (128, 8, 2, 6)),
        (16, "splitk", (64, 16, 1, 11)),
        (32, "splitk", (64, 32, 1, 9)),
    ),
    # q_b of a helix drafter (latent pages sharded over the group, every head on every rank):
    # 64 heads x 192. cuBLAS -> pinned at 8 / 16 / 32 rows: 4.75 / 4.79 / 6.65 -> 4.26 / 4.34 /
    # 4.49 us (scripts l0ull2drafter/bench_draft_gemm.py BG_TP=1, hecate job 838031, HBM locked).
    (12288, 1536): (
        (8, "splitk", (64, 8, 1, 12)),
        (16, "splitk", (64, 16, 1, 11)),
        (32, "splitk", (64, 32, 1, 9)),
    ),
}


def _pins(role: str, n: int, k: int, device: torch.device):
    if (n, k) in _PINS_SM107 and _sm107(device.index):
        return _PINS_SM107[(n, k)]
    return _PINS.get(role)


@lru_cache(maxsize=None)
def _sm107(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (10, 7)


def draft_gemm(
    x: torch.Tensor, weight: torch.Tensor, role: str, strided: bool = False
) -> Optional[torch.Tensor]:
    """``x @ weight.T`` with the kernel pinned for ``role`` at this row count.

    Returns None (the caller keeps its module path) when the switch is off, the
    row count is above the pinned range, or the operands do not fit the kernel.
    ``strided``: a 2-D ``x`` may be a row-major view with a 16-byte-aligned row
    stride (e.g. the q_a columns of the [q_a | kv_a] rows), read in place.
    """
    if weight.dim() != 2 or not x.is_cuda:
        return None
    k = x.shape[-1]
    m = x.numel() // k if k else 0
    pins = _pins(role, weight.shape[0], k, x.device)
    pin = next((p for p in pins or () if 0 < m <= p[0]), None)
    if (
        pin is None
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or weight.shape[1] != k
        or not weight.is_contiguous()
        or not x.is_cuda
    ):
        return None
    _, kind, tactic = pin
    if kind == "cublas":
        return F.linear(x, weight)
    # detach(): the kernel's DLPack export refuses tensors that require grad,
    # which module parameters do even under inference mode.
    strided = strided and x.dim() == 2 and x.stride(1) == 1 and x.stride(0) % 8 == 0
    x2d = x.detach() if strided else x.detach().reshape(m, k)
    weight = weight.detach()
    if (
        not (strided or x2d.is_contiguous())
        or k % 128
        or x2d.data_ptr() % 32
        or weight.data_ptr() % 32
    ):
        return None
    out = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device)
    ensure_cute_runtime()
    run_splitk_dense(
        x2d,
        weight.t(),
        None,
        out,
        _PDL,
        SplitKTactic(*tactic),
        strided_a=strided and not x2d.is_contiguous(),
    )
    return out.view(*x.shape[:-1], weight.shape[0])


_PUSH_TACTICS = {
    "o_proj": (
        (8, (64, 8, 1, 0)),
        (16, (64, 16, 1, 0)),
        (32, (64, 32, 1, 0)),
        (64, (64, 64, 1, 0)),
    ),
    "down": ((8, (64, 8, 1, 0)), (16, (64, 16, 1, 0)), (32, (64, 32, 1, 0)), (64, (64, 64, 1, 0))),
}


def push_tactic(role: str, m: int, n: int, k: int, table=None):
    """The ``SplitKTactic`` of the push GEMM for ``role`` at ``m`` rows, or None. A table entry
    with zero stages takes the deepest pipeline that fits."""
    entry = next((t for rows, t in (table or _PUSH_TACTICS).get(role, ()) if m <= rows), None)
    if entry is None:
        return None
    mma_m, mma_n, split_k, stages = entry
    for ab_stages in [stages] if stages else range(12, 0, -1):
        tactic = SplitKTactic(mma_m, mma_n, split_k, ab_stages)
        try:
            validate_push_tactic(tactic, m, n, k)
        except ValueError:
            continue
        return tactic
    return None


class DraftArSites:
    """The pushed all-reduce sites of one drafter, on its o_proj ``AllReduce`` (whose MNNVL
    workspace every drafter all-reduce shares)."""

    def __init__(self, all_reduce):
        self.all_reduce = all_reduce

    def linear_ar_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        role: str,
        residual: torch.Tensor,
        norm,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """``norm(all_reduce(x @ weight^T) + residual)`` and the new residual, or None if the site
        does not apply (the caller then runs its own path)."""
        if (
            x.dim() != 2
            or x.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or not x.is_contiguous()
            or not weight.is_contiguous()
            or not residual.is_contiguous()
            or residual.dtype != torch.bfloat16
            or getattr(norm, "weight", None) is None
            or x.data_ptr() % 32
            or weight.data_ptr() % 32
        ):
            return None
        m, k = x.shape
        n = weight.shape[0]
        if residual.shape != (m, n) or m == 0:
            return None
        tactic = push_tactic(role, m, n, k)
        if tactic is None or not ar_norm_config(m, n, x.device)[0]:
            return None
        target = lamport_push_target(self.all_reduce, m, n)
        if target is None:
            return None
        run_gemm_push(x.detach(), weight.detach(), target.output, target, tactic, pdl=_PDL)
        out = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        run_draft_ar_norm(
            target.workspace,
            target.one_shot,
            target.cpp_one_shot,
            residual,
            norm.weight.detach(),
            norm.variance_epsilon,
            out,
            residual_out,
            use_pdl=_PDL,
        )
        return out, residual_out


def draft_embed_site(embedding, ids: torch.Tensor) -> Optional[torch.Tensor]:
    """``embedding(ids)`` of a vocab-parallel ``layers.EmbeddingTable`` whose all-reduce
    runs over MNNVL, as one owner-multicast kernel; None where it does not apply."""
    weight = getattr(embedding, "weight", None)
    all_reduce = getattr(embedding, "all_reduce", None)
    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    if weight is None or mnnvl is None or not getattr(embedding, "vocab_parallel", False):
        return None
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or ids.dim() != 1
        or ids.dtype not in (torch.int32, torch.int64)
        or not ids.is_contiguous()
        or ids.shape[0] == 0
        or weight.shape[1] % (ELTS * EMBED_THREADS)
    ):
        return None
    # The table's shards: ceil(vocab / tp) rows per rank (this rank's padded to that), the last
    # one possibly shorter.
    vocab = embedding.num_embeddings
    slice_width = weight.shape[0]
    if slice_width != -(-vocab // mnnvl.mapping.tp_size) or embedding.vocab_start % slice_width:
        return None
    m, hidden = ids.shape[0], weight.shape[1]
    target = mnnvl_lamport_push_target(mnnvl, m, hidden, one_shot=True)
    if target is None:
        return None
    k3_draft_embed(
        target.workspace,
        ids,
        weight.detach(),
        slice_width,
        vocab,
        target.output,
        use_pdl=_PDL,
    )
    return target.output


def dflash_context_dtype(draft_model: nn.Module) -> torch.dtype:
    """The projection's activation dtype, which the context latents are computed in."""
    projection = getattr(draft_model, "fc", None)
    if projection is None:
        return torch.bfloat16
    return getattr(projection, "dtype", projection.weight.dtype)


def dflash_draft_slot_ids(
    num_gens: int,
    block_size: int,
    num_draft_tokens: int,
    shift_label: bool,
    device="cuda",
) -> torch.Tensor:
    """Flat indices into the [num_gens * block_size] drafter block outputs
    whose hidden states produce the K draft-token logits per request.

    Plain DFlash (K2.7 convention, shift_label off): the hidden state at
    mask slot j (j = 1..K) predicts draft token j; slot 0 (which holds the
    anchor/bonus token) is unused.

    DSpark shift_label convention (DeepSpec: labels for a block anchored at
    position p are input_ids[p+1 .. p+block_size], so the hidden state at
    block slot j predicts the token at position p+1+j): slots 0..K-1 are
    used, and slot 0 — the anchor token slot — predicts the first draft
    token.
    """
    request_bases = torch.arange(num_gens, dtype=torch.long, device=device) * block_size
    offsets = torch.arange(num_draft_tokens, dtype=torch.long, device=device)
    first_slot = 0 if shift_label else 1
    return (request_bases.unsqueeze(1) + first_slot + offsets.unsqueeze(0)).flatten()


def dflash_position_ceiling(max_ctx: int, block_size: int, max_draft_len: int) -> int:
    """Positions a drafter that indexes absolute positions must be able to encode.

    The forward reads two position sets past the accumulated context, both
    starting from a ctx_len clamped to ``max_ctx``:

      * block decode  ``ctx_len + num_accepted + j``, ``j in [0, block_size)``
      * context KV    ``ctx_len + [0, max_draft_len]``

    so the largest index is ``max_ctx + max(block_size, max_draft_len + 1) - 1``
    and a table needs one entry more than that. Derived from the runtime
    ``max_ctx`` rather than from model_config: py_executor_creator raises the
    engine's max_seq_len past the configured value and never writes it back, so
    any config-derived reconstruction drifts from what ctx_len is clamped to.
    """
    return int(max_ctx) + max(int(block_size), int(max_draft_len) + 1)


def dflash_allocated_ctx_limit(
    block_counts: torch.Tensor, page_size: int, block_size: int
) -> torch.Tensor:
    """Context length a request may advertise, given the pages it holds.

    Leaves ``block_size`` positions of room: the MLA path writes the block's own
    latents at ``ctx_len .. ctx_len + block_size``, so stopping at exactly the
    allocation puts the first of them on an unallocated page, whose table entry
    ``_refresh_ctx_block_tables`` clamped to 0 -- another request's block. A
    silent cross-request write, not a fault.

    Per-request only on the ``python`` V2 backend, which propagates
    ``BAD_PAGE_INDEX``. The default ``cpp`` one maps it to 0
    (kvCacheManagerV2Utils.cu), every row counts full, and this degenerates to
    the block-table-width bound upstream already applied. Do not read a
    per-request guarantee into it there.
    """
    return (block_counts * page_size - block_size).clamp(min=0)


@dataclass
class VocabParallelSelection:
    """The target LM head the worker selects tokens from vocab-parallel: a plain vocab-sharded,
    gathered TP head (:meth:`VocabParallelHead.supported`), no occurrence penalties and no
    guided decoder. The sampling decision is made per batch
    (:meth:`K3DSparkWorker._vocab_parallel_target_selection_applies`)."""

    head: VocabParallelHead


_HOST_BLOCK_COUNT_CHECKS = 8


class K3DSparkWorker(SpecWorkerBase):
    """The one-engine DSpark worker of the target (served DSparkWorker on DFlashWorker).

    Holds the drafter's per-request context state (slots, _ctx_len, the draft page table) and
    sequences one step (:meth:`_forward_impl`). The runtime contract is #21996's
    SpecWorkerBase / DFlashSpecMetadata.prepare() (_ctx_buf_inited, _req_to_slot,
    _req_ctx_pos, _free_slots, _write_ctx_len, _graph_dummy_id_floor,
    _assign_slot, _dummy_slot, _batch_to_slot_buffer).
    """

    def __init__(
        self,
        spec_config: "DFlashDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping
        self._resolved_mask_token_id = None
        self._resolved_block_size = None
        self._compute_block_size = None

        # Pre-allocated per-slot context K/V buffers, built lazily on the
        # first forward. Fixed-size so slot-indexed reads/writes are CUDA
        # graph compatible.
        self._ctx_buf_inited = False
        self._ctx_len = None
        # Host shadows of _ctx_len and of each request's prompt progress.
        self._ctx_len_host = None
        self._req_ctx_pos = {}
        # True when the drafter's context lives in the target manager's draft
        # layers (the engine unified them); see _lazy_init_ctx_buffers.
        self._ctx_unified_cache = False
        # The manager layer ids of the drafter's context, and the K-buffer
        # offset that manager's per-layer page tables carry.
        self._ctx_layer_ids = None
        self._ctx_block_offset = 0
        self._ctx_kv_factor = 2
        # Iterations left in which the host block counts are checked against
        # the device-decoded table (see _check_host_block_counts).
        self._draft_table_checks_left = 0
        # False once a check found those counts disagreeing with the table.
        self._ctx_unified_host_counts = True
        # Snapshot for rolling back in-place _ctx_len updates when a forward
        # fails (or after warmup). See _ensure_spec_dec_state_restored.
        self._saved_ctx_len = None
        self._saved_ctx_len_host = None
        self._saved_req_ctx_pos = None
        self._ctx_len_restore_pending = False
        # Deferred kv_lens_cuda rewind state (see _prepare_kv_for_draft_forward,
        # _apply_kv_rewind_after_draft, _ensure_spec_dec_state_restored).
        self._kv_rewind_pending = False
        self._kv_rewind_amount = None
        self._kv_rewind_nc = None
        self._kv_rewind_bs = None
        # The rewind rides in the next-new-tokens kernel (see _fused_select_accept,
        # _fused_kv_rewind_operands).
        self._fused_kv_rewind_armed = False
        self._batch_to_slot_buffer = None
        self._max_ctx = 0
        # Set by _lazy_init_ctx_buffers.
        self._ctx_paged = False
        # The KV cache manager's per-layer draft pool views [pages, 1, 1, page, 576].
        self._ctx_kv_buf = None
        # [L] int64 base addresses of the manager's per-layer pool views.
        self._ctx_kv_ptrs = None
        # The manager _ctx_kv_buf is borrowed from, so a swap can be detected.
        self._ctx_kv_manager = None
        self._ctx_block_tables = None  # [max_batch+1, max_blocks_per_seq] int32
        self._ctx_block_indptr = None
        # Pages the manager handed this request, per batch row. Saturates at
        # the table width on the default backend -- dflash_allocated_ctx_limit.
        self._ctx_block_counts = None
        # Host copy of this iteration's per-row counts, read from the manager's
        # caches (unified draft cache); None when only the device holds them.
        self._ctx_block_counts_host: list[int] | None = None
        self._ctx_pool_idx = 0
        self._ctx_block_divisor = 1  # encoded offset -> raw pool block index
        self._ctx_page_size = 32
        # True only when the drafter reads the draft KV cache manager's pool
        # AND that manager joins the target's reuse protocol, which is what
        # makes a reused prefix's drafter K/V still addressable. Set in
        # _lazy_init_ctx_buffers; read by _store_prefill_context.
        self._ctx_reuse_addressable = False
        self._dflash_attention_backend = spec_config.attention_backend
        # Set by bind_target_lm_head.
        self._vocab_parallel: Optional[VocabParallelSelection] = None
        # The private VocabParallelHead per LM head (keyed by id), built on
        # first use and shared by the target and draft selections of one head.
        self._vocab_parallel_heads = {}
        # The type of the logits processor the target's base class built (set by
        # install_k3_dspark); bind_target_lm_head takes over only that processor.
        self.plain_logits_processor_type: Optional[type] = None
        # Set by set_target_model and set_draft_model.
        self._target_model_ref: Optional[weakref.ref] = None
        self._draft_model_ref: Optional[weakref.ref] = None
        # The drafter's helix mapping when its context pages are CP-sharded.
        self._helix_mapping = None
        # The side-stream context latents' state (see _launch_ctx_side).
        self._ctx_side = None
        self._ctx_side_ar = None
        self._ctx_side_stream = None

        # Slot management (Python, updated in prepare() and eager mode)
        self._req_to_slot = {}  # request_id -> slot index
        self._free_slots = deque()  # available slot indices
        self._dummy_slot = None  # for cudagraph padding or warmup dummy requests
        self._graph_dummy_id_floor = 1 << 62

        logger.info(
            f"{type(self).__name__} initialized with "
            f"use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    @property
    def _draft_tokens_per_req(self) -> int:
        """Target-forward tokens per gen request: K drafts + 1 bonus.

        Previously this was 2K (K+1 accepted + K-1 mask fillers), but the
        fillers contribute nothing beyond the K+1 prefix the acceptance
        check and context store consume, so carrying them through every
        target layer is wasted work that dominates step time at large batch.
        """
        return self.max_draft_len + 1

    def _validate_draft_attention_backend(self, draft_model) -> None:
        draft_model_dflash_attention_backend = getattr(
            draft_model, "dflash_attention_backend", None
        )
        if draft_model_dflash_attention_backend is None:
            raise ValueError("DFlash draft model is missing dflash_attention_backend.")
        if self._dflash_attention_backend == "AUTO":
            # The draft model resolved AUTO against its own family and this
            # build; adopt that, because _lazy_init_ctx_buffers keys paging and
            # the trtllm-gen shape checks off the concrete value.
            self._dflash_attention_backend = draft_model_dflash_attention_backend
            return
        if draft_model_dflash_attention_backend != self._dflash_attention_backend:
            raise ValueError(
                "DFlash worker and draft model attention backends must match; "
                f"worker={self._dflash_attention_backend}, "
                f"model={draft_model_dflash_attention_backend}."
            )

    def set_draft_model(self, draft_model) -> None:
        super().set_draft_model(draft_model)
        self._validate_draft_attention_backend(draft_model)
        # Non-owning: the draft model is a sibling module of this worker.
        self._draft_model_ref = weakref.ref(draft_model)
        self._helix_mapping = getattr(draft_model, "_helix_mapping", None)
        # A d2t-remapped draft vocab is rejected at load, not per decode step.
        if self._d2t is not None and getattr(draft_model, "has_markov_head", False):
            raise NotImplementedError(
                "DSpark Markov head requires a shared draft/target vocab "
                "(d2t vocab mapping is not supported); drafter "
                f"{type(draft_model).__name__} declares one."
            )
        # Built by bind_draft_lm_head when the drafter's head is known.
        self._markov_tp_exchange = None

    def set_target_model(self, target_model: nn.Module) -> None:
        # Non-owning: the target model owns this worker.
        self._target_model_ref = weakref.ref(target_model)

    def bind_target_lm_head(self, lm_head: nn.Module, logits_processor: nn.Module) -> None:
        """Decide the :class:`VocabParallelSelection` over the target LM head.

        The shard logits are ``lm_head(hidden_states)``, which are the target's
        logits only when its logits processor is the plain processor the
        target's base class built (``plain_logits_processor_type``); a custom
        one (a final norm or head transform before the LM head) keeps the
        gathered logits. Binding the same head again keeps its selection.

        The head is :class:`VocabParallelHead` (MNNVL pair exchange), when there is no guided
        decoder, no occurrence penalties, the plain logits processor, and
        :meth:`VocabParallelHead.supported` holds.
        """
        if (
            self.guided_decoder is not None
            or self.spec_config.enable_penalty
            or self.plain_logits_processor_type is None
            or type(logits_processor) is not self.plain_logits_processor_type
        ):
            self._vocab_parallel = None
            return
        head = self._vocab_parallel_head(lm_head)
        if head is None:
            self._vocab_parallel = None
            return
        if self._vocab_parallel is None or self._vocab_parallel.head is not head:
            # Temperature-only sampling is decided per batch
            # (_vocab_parallel_target_selection_applies), not here.
            self._vocab_parallel = VocabParallelSelection(head=head)

    def bind_draft_lm_head(self, draft_model) -> None:
        """Post-load hook for the draft side of the selection.

        The greedy draft tokens are chosen per step by
        :meth:`_draft_vocab_parallel_head`, so nothing is bound here.
        """
        del draft_model

    def setup_aliases(self) -> None:
        """Post-load hook: decide the vocab-parallel selection, the target side
        and then the draft side.

        The model loader runs it after loading: the target model's logits
        processor is final (models install a custom one after the base
        constructor), and the drafter has taken over the target modules it
        shares (``load_weights_from_target_model``).
        """
        target_model = None if self._target_model_ref is None else self._target_model_ref()
        if target_model is not None:
            self.bind_target_lm_head(target_model.lm_head, target_model.logits_processor)
        if self._draft_model_ref is not None:
            self.bind_draft_lm_head(self._draft_model_ref())

    def post_load_weights(self) -> None:
        self.setup_aliases()

    def set_guided_decoder(self, guided_decoder) -> bool:
        # The guided decoder masks the full target logits.
        self._vocab_parallel = None
        return super().set_guided_decoder(guided_decoder)

    @property
    def target_vocab_parallel_head(self) -> Optional[VocabParallelHead]:
        return None if self._vocab_parallel is None else self._vocab_parallel.head

    def target_head_rows(self, hidden_states: torch.Tensor, attn_metadata, spec_metadata):
        """A generation-only batch gathers every token in order (gather ids 0..num_tokens-1), so
        the head reads ``hidden_states`` itself instead of an index copy of it."""
        gather_ids = spec_metadata.gather_ids
        if (
            attn_metadata.num_contexts == 0
            and gather_ids is not None
            and gather_ids.shape[0] == hidden_states.shape[0]
        ):
            return hidden_states
        return hidden_states[gather_ids]

    def target_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        logits_processor,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        """This rank's vocab shard (``VocabShardedLogits``) once the target
        head is bound for vocab-parallel selection, the gathered logits
        otherwise."""
        head = self.target_vocab_parallel_head
        if head is not None:
            return head.shard_logits(hidden_states)
        return super().target_logits(
            hidden_states, lm_head, logits_processor, attn_metadata, spec_metadata, draft_model
        )

    def _helix_shard(self) -> Optional[HelixDraftShard]:
        """This rank's share of the context pages when the drafter is helix-sharded."""
        helix = self._helix_mapping
        if helix is None:
            return None
        return HelixDraftShard(helix.cp_size, helix.cp_rank, self._ctx_page_size)

    def _vocab_parallel_head(self, lm_head) -> Optional[VocabParallelHead]:
        """The cached private ``VocabParallelHead`` of ``lm_head``, or None when
        the head is not a plain vocab shard."""
        if lm_head is None:
            return None
        key = id(lm_head)
        if key not in self._vocab_parallel_heads:
            vocab = int(getattr(lm_head, "num_embeddings", 0) or 0)
            supported = vocab > 0 and VocabParallelHead.supported(lm_head, vocab)
            self._vocab_parallel_heads[key] = (
                VocabParallelHead(lm_head, vocab) if supported else None
            )
        return self._vocab_parallel_heads[key]

    def _vocab_parallel_target_selection_applies(self, spec_metadata) -> bool:
        """Whether this batch's target tokens can come from shard logits.

        Greedy batches always; temperature-only batches only under the
        ``no_topk_no_topp`` sampling mode and without rejection sampling, which
        needs the full target distribution.
        """
        if spec_metadata.is_all_greedy_sample:
            return True
        if self._vocab_parallel is None:
            return False
        if spec_metadata.use_rejection_sampling:
            return False
        return spec_metadata.advanced_sampling_mode == AdvancedSamplingMode.NO_TOPK_NO_TOPP

    def _draft_vocab_parallel_head(self, draft_model, spec_metadata):
        """The head greedy draft tokens are selected vocab-parallel with, or None: no advanced
        draft sampling, no guided decoder, a shared vocab, and Markov weights the chain can
        index (contiguous ``[vocab, rank]`` tensors of one dtype)."""
        if (
            spec_metadata.wants_advanced_draft_sampling
            or self.guided_decoder is not None
            or self._d2t is not None
        ):
            return None
        head = self._vocab_parallel_head(getattr(draft_model, "lm_head", None))
        if head is None or not getattr(draft_model, "has_markov_head", False):
            return head
        w1, w2 = draft_model.markov_w1, draft_model.markov_w2
        if (
            w1.is_contiguous()
            and w2.is_contiguous()
            and w1.shape == w2.shape
            and w1.dtype == w2.dtype
            and w2.shape[0] == head.vocab_size
        ):
            return head
        return None

    def _vocab_parallel_markov_weights(self, draft_model):
        if not getattr(draft_model, "has_markov_head", False):
            return None, None
        return draft_model.markov_w1, draft_model.markov_w2

    def _k3_vocab_parallel_draft_tokens(
        self,
        draft_head: VocabParallelHead,
        draft_model,
        hidden_states_out: torch.Tensor,
        gen_gather_ids: torch.Tensor,
        num_gens: int,
        num_draft_tokens: int,
        inputs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy draft tokens straight from this rank's vocab shard of the
        block logits through the private head, bit-identical to refining the
        gathered logits and taking the per-position argmax.

        Returns ``([num_gens, K]`` int32 draft tokens, the ``[num_gens * K,
        hidden]`` block hidden states they were computed from).
        """
        K = num_draft_tokens
        markov_w1, markov_w2 = self._vocab_parallel_markov_weights(draft_model)
        # A contiguous row range of the block output (one request) is read in place instead of
        # gathered.
        rows = self._block_gather_range(draft_model, num_gens, K, hidden_states_out.shape[0])
        block_hidden = (
            hidden_states_out[rows[0] : rows[1]]
            if rows is not None
            else hidden_states_out[gen_gather_ids]
        )
        local_block = (
            draft_gemm(block_hidden, draft_head.lm_head.weight, "lm_head")
            if getattr(draft_head.lm_head, "bias", None) is None
            else None
        )
        if local_block is None:
            local_block = draft_head.local_logits(block_hidden)
        local_block = local_block.view(num_gens, K, -1)
        if not draft_head.markov_v3_supported(
            local_block, inputs["first_prev_tokens"], markov_w1, markov_w2
        ):
            raise RuntimeError(
                "K3 DSpark: the CuTe DSL Markov chain (markov_tokens_v3) does not serve "
                f"{num_gens} x {K} block rows of {local_block.shape[-1]} vocab columns"
            )
        gen_draft_tokens = draft_head.markov_tokens_v3(
            local_block, inputs["first_prev_tokens"], markov_w1, markov_w2
        )
        return gen_draft_tokens, block_hidden

    def _managed_ctx_pool(
        self,
        draft_kv_cache_manager,
        L: int,
        nkv: int,
        hd: int,
        dtype: torch.dtype,
        kv_factor: int = 2,
    ) -> list[torch.Tensor] | None:
        """Per-layer views of the draft KV cache manager's pool, or None.

        The drafter's layers are the manager's ``draft_layer_ids`` when it
        appends them to the target's (the unified draft cache), else its first
        ``L`` layers. ``get_buffers(..., "HND")`` hands back the per-layer shape
        the private arena uses; ``dtype`` is the pool's (FP8 E4M3, the drafter's
        ``context_kv_fp8``).

        A list, not one stacked tensor: only the two consumers that index it by
        layer need it, and the two managers disagree on what a stack would mean.
        V1 slices one page-major allocation, so the layers are strided views of
        a shared storage; V2 wraps each layer's pool base address in its own
        tensor. Everything downstream indexes ``[layer_idx]`` first anyway.

        Note the page count is not the drafter's to interpret: a V2 view spans
        the whole interleaved pool, not this layer's slice, so only shape[1:] is
        checked here; the index space is settled in _init_ctx_block_offsets.
        A unified cache cannot fall back to the private arena (the manager
        allocates, reuses and transfers the drafter's pages), so a mismatch
        there raises.
        """
        if draft_kv_cache_manager is None:
            # No separate draft KV cache. Attention DP alone does NOT disable
            # it: _util.py skips that bail for an external drafter, which is
            # what DSpark is, so this drafter does get a manager under DP.
            return None
        draft_layer_ids = tuple(getattr(draft_kv_cache_manager, "draft_layer_ids", ()))
        layer_ids = draft_layer_ids if draft_layer_ids else tuple(range(L))
        if len(layer_ids) != L:
            reason = (
                f"the KV cache manager exposes {len(layer_ids)} draft layers but the "
                f"drafter has {L}"
            )
            if self._ctx_unified_cache:
                raise ValueError(f"DFlash: {reason}.")
            logger.warning(f"DFlash: {reason}; using the private ctx arena.")
            return None
        self._ctx_layer_ids = layer_ids
        layers = [
            draft_kv_cache_manager.get_buffers(layer_id, kv_layout="HND") for layer_id in layer_ids
        ]
        base = layers[0]
        # kv_factor 1 is the MLA drafter's SELFKONLY pool: one latent, no V.
        expected = (kv_factor, nkv, base.size(-2), hd)
        for i, layer in enumerate(layers):
            if tuple(layer.shape[1:]) != expected or layer.dtype != dtype:
                reason = (
                    f"draft KV pool layer {i} is {tuple(layer.shape)} / {layer.dtype}, "
                    f"not {('pages',) + expected} / {dtype}"
                )
                if self._ctx_unified_cache:
                    raise ValueError(f"DFlash: the unified draft KV pool does not match: {reason}.")
                # Fall back rather than fail: the private arena is what every
                # DFlash model shipped with, so an unrecognized pool costs the
                # old memory footprint, not the run.
                logger.warning(f"DFlash: {reason}; using the private ctx arena.")
                return None
        logger.info(
            f"DFlash: binding the drafter ctx cache to the draft KV cache manager pool: "
            f"{L} x {tuple(base.shape)} {base.dtype}, page_size={base.size(-2)}, "
            f"layers={list(layer_ids)}, unified={self._ctx_unified_cache}"
        )
        return layers

    def _init_ctx_block_offsets(self, draft_kv_cache_manager, pool, L, nkv, hd, page_size) -> bool:
        """Validate the pool layout and resolve its offset decode constants.

        ``kv_cache_block_offsets`` entries are ``pool_block_index *
        num_pool_layers * kv_factor`` plus the K/V field index. What that has to
        be divided by to index a ``get_buffers`` view depends on the manager --
        see the stride check below; attention_backend/utils.py hardcodes the V1
        divisor and is not a general contract. A V2 manager with per-layer page
        tables also offsets every entry by the layer's K buffer within the
        physical slot (``_ctx_block_offset``).
        """
        mgr = draft_kv_cache_manager
        pool_mapping = mgr.kv_cache_pool_mapping
        first_layer_id = self._ctx_layer_ids[0] if self._ctx_layer_ids is not None else 0
        layer_offset = mgr.layer_offsets[first_layer_id]
        self._ctx_pool_idx = int(pool_mapping[layer_offset, 0])
        per_layer_tables = bool(getattr(mgr, "_use_per_layer_page_tables", False))
        num_pool_layers = int((pool_mapping[:, 0] == self._ctx_pool_idx).sum())
        self._ctx_block_offset = 0
        if isinstance(mgr, KVCacheManagerV2):
            converter = mgr.impl.get_page_index_converter(layer_offset, Role.KEY)
            layer_group_id = mgr.layer_to_pool_mapping_dict[layer_offset]
            unsupported = None
            if mgr.enable_swa_scratch_reuse:
                unsupported = "DFlash managed context does not support SWA scratch pages"
            for layer_id in self._ctx_layer_ids:
                local_id = mgr.layer_offsets[layer_id]
                if (
                    mgr.layer_to_pool_mapping_dict[local_id] != layer_group_id
                    or mgr.get_layer_page_index_scale(layer_id) != converter.scale
                ):
                    unsupported = (
                        "DFlash's shared context block table requires draft layers "
                        "with the same lifecycle and page-index scale"
                    )
                    break
            if unsupported is not None:
                # A unified manager was selected for this layout, so the drafter
                # cannot move to private storage without breaking allocation,
                # reuse and transfer. A separate manager keeps the private arena.
                if self._ctx_unified_cache:
                    raise NotImplementedError(unsupported)
                logger.warning(f"{unsupported}; using the private ctx arena.")
                return False
            if per_layer_tables:
                # Attention tables address the physical pool origin, whereas
                # get_buffers() starts at this layer's own K buffer.
                self._ctx_block_offset = int(converter.layer_offset)
        elif num_pool_layers != L:
            logger.warning(
                f"DFlash: draft KV pool {self._ctx_pool_idx} spans {num_pool_layers} layers but "
                f"the drafter has {L}; using the private ctx arena."
            )
            return False
        manager_kv_factor = (
            mgr.get_layer_kv_factor(first_layer_id)
            if hasattr(mgr, "get_layer_kv_factor")
            else mgr.kv_factor
        )
        if manager_kv_factor != self._ctx_kv_factor:
            if self._ctx_unified_cache:
                raise ValueError(
                    f"DFlash: the unified draft layers have KV factor {manager_kv_factor}, "
                    f"the drafter {self._ctx_kv_factor}."
                )
            logger.warning(
                "DFlash: draft KV layer factor does not match the drafter; "
                "using the private ctx arena."
            )
            return False
        # V1 slices a page-major pool, so stride(0) spans every layer; V2 wraps
        # the layer base in a dense view over interleaved pages, so stride(0) is
        # one layer slot. Guessing wrong is silent: acceptance quietly drops.
        layer_block_elems = manager_kv_factor * nkv * page_size * hd
        stride0 = pool[0].stride(0)
        if stride0 == layer_block_elems:
            self._ctx_block_divisor = manager_kv_factor
        elif stride0 == num_pool_layers * layer_block_elems:
            self._ctx_block_divisor = num_pool_layers * manager_kv_factor
        else:
            message = (
                f"DFlash: draft KV pool layer stride {stride0} matches neither V1's "
                f"page-major slice ({num_pool_layers * layer_block_elems}) nor V2's "
                f"interleaved view ({layer_block_elems})"
            )
            if self._ctx_unified_cache:
                raise ValueError(f"{message}.")
            logger.warning(
                f"{message}; using the private ctx arena rather than guessing block indices."
            )
            return False
        logger.info(
            f"DFlash: ctx block offsets pool_idx={self._ctx_pool_idx}, "
            f"divisor={self._ctx_block_divisor}, block_offset={self._ctx_block_offset}, "
            f"view_pages={pool[0].size(0)}"
        )
        return True

    def _init_ctx_block_tables(self, num_slots: int, max_blocks: int) -> None:
        """Size the per-iteration block table."""
        self._ctx_block_tables = torch.zeros(
            (num_slots, max_blocks), dtype=torch.int32, device="cuda"
        )
        self._ctx_block_indptr = torch.arange(
            0, (num_slots + 1) * max_blocks, max_blocks, dtype=torch.int32, device="cuda"
        )
        self._ctx_block_counts = torch.zeros(num_slots, dtype=torch.long, device="cuda")
        logger.info(f"DFlash: ctx block tables {tuple(self._ctx_block_tables.shape)}")

    @staticmethod
    def _check_block_within_allocation(
        num_ctx: torch.Tensor, allocated_ctx_limit: torch.Tensor, block_size: int
    ) -> None:
        """Raise when a request's drafter block would leave its allocated pages.

        The shared manager sizes the request, so a shortfall there would have the
        block's scratch write land on the last context position, or on another
        request's page. Asynchronous, so it adds no host sync and is safe to
        capture in a CUDA graph; on the device it traps instead of raising.
        """
        torch._assert_async(
            (num_ctx <= allocated_ctx_limit).all(),
            f"DFlash: the drafter's block of {block_size} positions falls outside the "
            "pages the shared KV cache manager allocated for the request",
        )

    def _refresh_ctx_block_tables(
        self, attn_metadata, num_seqs: int, request_ids: list[int] | None = None
    ) -> bool:
        """Decode this iteration's draft block table into the persistent buffer.

        In place and fixed-shape so a captured graph keeps reading the same
        tensor. Returns False when there is no table to fill, i.e. every path
        that stayed on the private arena.

        The unified draft cache's pages are rows of the target metadata's
        ``kv_cache_block_offsets`` (the draft layers' pool), a separate draft
        manager's are ``draft_kv_cache_block_offsets``. For the unified cache the
        per-request block counts are also read on the host from the manager's
        caches, so the prefill store need not wait for the device.
        """
        self._ctx_block_counts_host = None
        if self._ctx_block_tables is None or num_seqs <= 0:
            return False
        offsets_name = (
            "kv_cache_block_offsets" if self._ctx_unified_cache else "draft_kv_cache_block_offsets"
        )
        src = getattr(attn_metadata, offsets_name, None)
        if src is None:
            # The table is bound but unfillable. Leaving it at zeros would send
            # every request's every layer to pool block 0 -- silently.
            raise RuntimeError(
                "DFlash: bound to the draft KV pool but "
                f"{type(attn_metadata).__name__} carries no "
                f"{offsets_name} this iteration."
            )
        # Field 0 of the K/V pair: both name the same block for a kv_factor=2
        # pool, and the drafter addresses blocks, not planes. The block count
        # is taken before the clamp erases the placeholders; it is informative
        # only on the python V2 backend, the cpp one already mapped
        # BAD_PAGE_INDEX to 0. The kernels remove the layer offset in int64
        # and leave the metadata's table unchanged.
        # Served builds this table on the host and uploads it only when it changes (its managed
        # draft cache); #21996 hands the draft layers' pages over as rows of the target's
        # kv_cache_block_offsets, so one decode kernel per step reads them from there.
        decode_draft_block_table(
            src[self._ctx_pool_idx, :num_seqs, 0],
            self._ctx_block_divisor,
            self._ctx_block_tables,
            self._ctx_block_counts,
            block_offset=self._ctx_block_offset,
        )
        # Only the context rows' counts are read on the host (the prefill store).
        num_contexts = min(int(attn_metadata.num_contexts), num_seqs)
        cache_map = getattr(self._ctx_kv_manager, "kv_cache_map", None)
        if (
            self._ctx_unified_cache
            and self._ctx_unified_host_counts
            and cache_map is not None
            and request_ids is not None
            and num_contexts > 0
        ):
            counts = [
                int(cache_map[request_id].num_blocks) if request_id in cache_map else 0
                for request_id in request_ids[:num_contexts]
            ]
            if self._draft_table_checks_left > 0 and not torch.cuda.is_current_stream_capturing():
                self._draft_table_checks_left -= 1
                if not self._check_host_block_counts(counts):
                    return True
            self._ctx_block_counts_host = counts
        return True

    def _check_host_block_counts(self, counts: list[int]) -> bool:
        """Compare the manager's block counts of the context rows with the decoded table's.

        Run on the first eager iterations that carry context requests (it reads
        the device counts back). The decoded count saturates at the table width
        where the manager writes unallocated entries as page 0 and the layer
        offset is 0, which bounds nothing, so that is accepted for any host
        count within the width. On a disagreement the host counts are dropped
        for good and the prefill store reads the device counts instead.
        """
        width = self._ctx_block_tables.size(1)
        device_counts = self._ctx_block_counts[: len(counts)].tolist()
        if all(d == h or (d == width and h <= width) for d, h in zip(device_counts, counts)):
            return True
        logger.error(
            "DFlash: the unified draft cache's host block counts "
            f"{counts[:8]} disagree with the decoded table's {device_counts[:8]}; "
            "the prefill store reads the device counts from now on."
        )
        self._draft_table_checks_left = 0
        self._ctx_unified_host_counts = False
        return False

    def _check_unified_helix(self, draft_kv_cache_manager) -> None:
        """Require the drafter's helix sharding to be the target manager's.

        A unified cache's draft pages follow the target's helix ledger (global
        page g on CP rank g % cp_size, as its local page g // cp_size), which is
        what the drafter's own sharding (``_helix_shard``) assumes of its pages.
        """
        mapping = draft_kv_cache_manager.mapping
        target = (mapping.cp_size, mapping.cp_rank) if mapping.has_cp_helix() else None
        helix = self._helix_mapping
        drafter = None if helix is None else (helix.cp_size, helix.cp_rank)
        if target != drafter:
            raise ValueError(
                "DFlash: the unified draft KV pages are helix-sharded as "
                f"(cp_size, cp_rank)={target}, but the drafter reads them as {drafter}."
            )

    def _lazy_init_ctx_buffers(
        self, draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager=None
    ):
        self._validate_draft_attention_backend(draft_model)
        # The target manager also holding the drafter's layers (the unified draft
        # cache): their pages are rows of the target metadata's block offsets.
        self._ctx_unified_cache = (
            draft_kv_cache_manager is not None
            and draft_kv_cache_manager is getattr(attn_metadata, "kv_cache_manager", None)
        )
        offsets_name = (
            "kv_cache_block_offsets" if self._ctx_unified_cache else "draft_kv_cache_block_offsets"
        )
        # Only TrtllmAttentionMetadata builds the block offsets. Without them the
        # table stays all-zero and every request would read and write pool
        # block 0 -- silently, at acceptance 1.0.
        if draft_kv_cache_manager is not None and not hasattr(attn_metadata, offsets_name):
            if self._ctx_unified_cache:
                raise NotImplementedError(
                    f"DFlash unified KV requires {offsets_name} in attention metadata"
                )
            logger.warning(
                f"DFlash: {type(attn_metadata).__name__} carries no draft KV block offsets; "
                "using the private ctx arena."
            )
            draft_kv_cache_manager = None

        if self._ctx_buf_inited:
            if self._ctx_kv_manager is draft_kv_cache_manager:
                return
            # KV cache estimation swaps a probe manager for the real one.
            # Holding the old pool's tensors across that writes freed memory.
            logger.info("DFlash: draft KV cache manager replaced, rebinding the ctx cache.")
            self._ctx_buf_inited = False

        # Recorded even when the buffers stay private, so that the estimation
        # pass falling back to the arena still rebinds once the real pool lands.
        self._ctx_kv_manager = draft_kv_cache_manager
        self._ctx_block_tables = None
        self._ctx_block_counts = None
        self._ctx_block_counts_host = None
        self._ctx_layer_ids = None
        self._ctx_block_offset = 0
        self._ctx_reuse_addressable = False
        max_batch = spec_metadata.max_num_requests

        # ctx_len is 1:1 with the target's positions, so max_seq_len bounds it.
        # max_position_embeddings does not: K3's drafter advertises 1048576,
        # which sizes the arena at hundreds of GiB.
        config = getattr(draft_model, "config", None)
        max_pos = getattr(config, "max_position_embeddings", None) if config else None
        # Take the served length from the KV cache manager, which every backend
        # carries on the metadata base class. A `max_seq_len` attribute is only
        # declared by some metadata classes (TrtllmAttentionMetadata has one --
        # a wrapper returning this same manager value; the base class and
        # FlashInferAttentionMetadata do not), so reading the attribute alone
        # lets max_position_embeddings win the min() below and sizes the arena
        # for the advertised positions instead of the served length. The
        # attribute stays as the fallback for no-cache attention, where it is
        # set by hand and there is no manager to ask.
        cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        runtime_max = getattr(cache_manager, "max_seq_len", None)
        if runtime_max is None:
            runtime_max = getattr(attn_metadata, "max_seq_len", None)
        candidates = [c for c in (runtime_max, max_pos) if c is not None]
        self._max_ctx = min(candidates) if candidates else 8192
        # py_executor_creator inflates max_seq_len by at most this much for spec
        # dec (one term is overlap-only), so compare against what is served.
        from tensorrt_llm._torch.speculative.utils import get_num_extra_kv_tokens

        spec_slack = 2 * (self.spec_config.tokens_per_gen_step - 1) + get_num_extra_kv_tokens(
            self.spec_config
        )
        if runtime_max is not None and self._max_ctx < runtime_max - spec_slack:
            # Warn, not raise: the per-request skip in _store_prefill_context
            # keeps an oversized request from taking the executor down.
            logger.warning(
                f"DFlash drafter covers {self._max_ctx} positions but the engine serves "
                f"up to {runtime_max - spec_slack}. max_position_embeddings={max_pos} is "
                "the binding constraint, so requests past it will draft nothing."
            )

        # The context latents are computed in the projection's activation dtype (BF16) and held in
        # an FP8 pool: the draft layers the target's manager appends for a drafter that declares an
        # FP8 draft_kv_dtype (context_kv_fp8); every store converts.
        dtype = dflash_context_dtype(draft_model)
        if dtype != torch.bfloat16:
            raise ValueError(f"K3 DSpark: the drafter's activation dtype must be BF16, got {dtype}")
        if getattr(draft_model, "context_kv_fp8", False) is not True:
            raise ValueError("K3 DSpark: the drafter's context must be FP8 (context_kv_fp8)")
        pool_dtype = torch.float8_e4m3fn

        # Reserve slot index max_batch as a scratch slot for padding/unknown
        # dummies; real requests only draw slots 0..max_batch-1, so dummy
        # writes land here and can't corrupt a real request's context.
        self._dummy_slot = max_batch
        num_slots = max_batch + 1
        # CUDA-graph padding uses ids counting down from
        # CUDA_GRAPH_DUMMY_REQUEST_ID; a floor of one draft block separates them
        # from real ids. Imported lazily to avoid an import cycle.
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

        self._graph_dummy_id_floor = CUDA_GRAPH_DUMMY_REQUEST_ID - self.max_draft_len

        self._ctx_len = torch.zeros(num_slots, dtype=torch.long, device="cuda")
        self._ctx_len_host = [0] * num_slots
        self._batch_to_slot_buffer = DeviceCopyBuffer(
            torch.zeros(max_batch, dtype=torch.long, device="cuda"),
            pin_memory=prefer_pinned(),
        )
        self._free_slots = deque(range(max_batch))
        self._req_to_slot = {}
        self._req_ctx_pos = {}

        # checkpoint's trained block width
        self._resolved_block_size = getattr(draft_model, "block_size", None) or (
            self.max_draft_len + 1
        )
        # what the draft forward actually computes
        self._compute_block_size = self._draft_block_width(draft_model)
        if self._compute_block_size > self._resolved_block_size:
            slack = self._compute_block_size - self.max_draft_len
            raise ValueError(
                f"DFlash checkpoint was trained with block_size={self._resolved_block_size}."
                f"Lower max_draft_len to at most {self._resolved_block_size - slack}."
                f"Current max_draft_len={self.max_draft_len}."
            )
        if self._ctx_unified_cache:
            reserved = getattr(draft_kv_cache_manager, "_shared_draft_block_width", 0)
            if self._compute_block_size > reserved:
                raise ValueError(
                    f"Drafter block width {self._compute_block_size} exceeds the "
                    f"{reserved} positions reserved by the shared KV cache manager. "
                    "The manager must reserve the drafter's resolved block width."
                )

        # +block_size slack lets each backend append the per-iteration noise
        # K/V after the accumulated request context.
        assert hasattr(draft_model, "_build_fused_kv_buffers"), (
            "DFlash draft model must define _build_fused_kv_buffers."
        )
        draft_model._build_fused_kv_buffers()
        L = draft_model._num_attn_layers
        nkv = draft_model._num_kv_heads
        hd = draft_model._head_dim
        kv_factor = getattr(draft_model, "_kv_factor", 2)
        self._ctx_kv_factor = kv_factor
        # Published before any drafter table is built, so a drafter may read it
        # instead of its config cap. _max_ctx is attn_metadata.max_seq_len, which
        # py_executor_creator raises past model_config.max_seq_len and never
        # writes back. _compute_block_size: the block decode's j runs over it.
        draft_model._runtime_position_ceiling = dflash_position_ceiling(
            self._max_ctx, self._compute_block_size, self.max_draft_len
        )
        if not getattr(draft_model, "_paged_ctx_cache", False):
            raise ValueError("K3 DSpark: the drafter's context is paged (CUTEDSL decode)")
        self._ctx_paged = True
        pool = self._managed_ctx_pool(draft_kv_cache_manager, L, nkv, hd, pool_dtype, kv_factor)
        if pool is not None:
            # The manager's page size: its pool is already carved.
            page_size = pool[0].size(-2)
            self._ctx_page_size = page_size
            if not self._init_ctx_block_offsets(
                draft_kv_cache_manager, pool, L, nkv, hd, page_size
            ):
                pool = None
        if pool is None or not self._ctx_unified_cache:
            raise ValueError(
                "K3 DSpark: the drafter's context must be the FP8 draft layers the target's "
                "KV cache manager appends (unified draft cache); no private arena"
            )
        # Pages come from the manager's per-request block table one iteration at a time.
        self._check_unified_helix(draft_kv_cache_manager)
        self._draft_table_checks_left = _HOST_BLOCK_COUNT_CHECKS
        self._ctx_unified_host_counts = True
        self._ctx_kv_buf = pool
        self._ctx_kv_ptrs = torch.tensor(
            [layer.data_ptr() for layer in pool], dtype=torch.int64, device="cuda"
        )
        self._init_ctx_block_tables(num_slots, draft_kv_cache_manager.max_blocks_per_seq)
        # The unified manager publishes and matches its own blocks, so a reused prefix's
        # drafter latents stay addressable.
        self._ctx_reuse_addressable = True
        self._ctx_buf_inited = True

        logger.info(
            f"DFlash: allocated ctx buffers: max_batch={max_batch}, "
            f"max_ctx={self._max_ctx}, dtype={dtype}, pool_dtype={pool_dtype}, "
            f"dflash_attention_backend={self._dflash_attention_backend}, "
            f"reuse_addressable={self._ctx_reuse_addressable}, "
            f"unified={self._ctx_unified_cache}"
        )

    def _assign_slot(
        self,
        req_id: int,
        reset: bool = False,
        updates: Optional[dict] = None,
        bases: Optional[dict] = None,
    ):
        """Get (or refresh) this request's context slot; None when none is free.

        ``reset`` recycles an existing slot, which is how a reused request id is
        told apart from a continuation. Pass ``updates`` to defer the length
        resets into a caller's batch instead of writing _ctx_len per request;
        either way the host mirror stays in step with the device tensor.
        ``bases`` is the base-page batch of main's ``DFlashSpecMetadata.prepare()``;
        this worker's slots always read from position 0, so it is left untouched.
        """

        def clear(slot: int) -> None:
            if updates is None:
                self._write_ctx_len({slot: 0})
            else:
                updates[slot] = 0

        if reset and req_id in self._req_to_slot:
            old_slot = self._req_to_slot.pop(req_id)
            clear(old_slot)
            self._free_slots.append(old_slot)
        if req_id not in self._req_to_slot:
            if not self._free_slots:
                return None
            slot = self._free_slots.popleft()
            self._req_to_slot[req_id] = slot
            clear(slot)
        return self._req_to_slot[req_id]

    def _store_context_kv_paged(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        owned: Optional[torch.Tensor] = None,
    ) -> None:
        """Write K/V at rank-local ``positions``.

        ``owned`` (helix): False entries rewrite their slot's current value,
        which keeps a fixed-size, CUDA-graph-safe store.
        """
        # Single-latent (MLA) FP8 cache: one vector per token per layer, a plain scatter of the
        # saturated FP8 latent (to_draft_fp8, as the block decode quantizes its query) into the
        # page the block table names, through byte views (index_put support for FP8 varies).
        if v is not None:
            raise RuntimeError("K3 DSpark: the drafter's context is one MLA latent, no V")
        page_size = self._ctx_page_size
        pages = self._ctx_block_tables[rows.long(), positions.long() // page_size].long()
        offsets = positions.long() % page_size
        keep = None if owned is None else owned.view(-1, *([1] * (k.dim() - 2)))
        for layer_idx in range(k.size(1)):
            view = self._ctx_kv_buf[layer_idx].view(torch.uint8)
            new = to_draft_fp8(k[:, layer_idx]).view(torch.uint8)
            if keep is not None:
                new = torch.where(keep, new, view[pages, 0, :, offsets])
            view[pages, 0, :, offsets] = new

    def _prepare_attn_metadata_for_dflash(self, attn_metadata, spec_metadata):
        """Save attn_metadata fields that DFlash modifies during forward."""
        is_capturing = torch.cuda.is_current_stream_capturing()

        if spec_metadata.is_cuda_graph and not is_capturing:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")
        else:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")

    def _prepare_kv_for_draft_forward(
        self,
        attn_metadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        """Adjust kv_lens_cuda so the draft model sees correct RoPE positions."""
        if hasattr(attn_metadata, "kv_lens_cuda"):
            self._kv_rewind_amount = 1 - num_accepted_tokens[num_contexts:batch_size]
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size

            if batch_size > num_contexts:
                attn_metadata.kv_lens_cuda[num_contexts:batch_size] += 1
            self._kv_rewind_pending = True

            attn_metadata.update_for_spec_dec()

    def _apply_kv_rewind_after_draft(self, attn_metadata, spec_metadata):
        """Apply the deferred kv_lens rewind after the draft forward."""
        self._kv_rewind_pending = False
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if is_warmup:
            # kv_lens_cuda was saved by prepare_for_spec_dec in this mode and
            # is restored wholesale, so no rewind is needed.
            return

        if self._kv_rewind_amount is not None and hasattr(attn_metadata, "kv_lens_cuda"):
            nc = self._kv_rewind_nc
            bs = self._kv_rewind_bs
            attn_metadata.kv_lens_cuda[nc:bs] -= self._kv_rewind_amount
            attn_metadata.kv_lens_cuda[nc:bs].clamp_(min=0)

    def _advance_ctx_len(self, slots: torch.Tensor, num_accepted: torch.Tensor) -> None:
        """Add this step's accepted tokens to the context lengths of ``slots``, on the device.

        Padding, warmup and attention-DP dummy rows share the dummy slot, which no request owns
        or frees. Emptying it right after the add keeps it at 0 between steps: those rows attend
        no context, and their next context K / V starts at column 0, inside the padding
        request's own pages (past them, its page-table row maps page 0, another request's).
        The store is to a fixed slot, so a captured step replays it.
        """
        self._ctx_len[slots] += num_accepted
        self._ctx_len.clamp_(max=self._max_ctx)
        self._ctx_len[self._dummy_slot].zero_()

    def _rebuild_batch_to_slot(self, request_ids: list[int]) -> None:
        mapping = [self._req_to_slot.get(rid, self._dummy_slot) for rid in request_ids]
        self._batch_to_slot_buffer.update(torch.tensor(mapping, dtype=torch.long))

    def _seed_generation_slot(self, slot: int, position: int, updates: dict, bases: dict) -> None:
        """Record the context length of a slot newly assigned to a generation row whose first
        token sits at absolute ``position`` (main's ``DFlashSpecMetadata.prepare()`` calls this
        for a disaggregated generation request, or one that lost its slot).

        On the unified cache the draft layers travel with the target's prompt KV, so the slot
        holds drafter K/V for every position below ``position``. The private arena holds none
        for such a request, so its slot starts empty. Either way the slot reads from position 0,
        so ``bases`` is left untouched.
        """
        updates[slot] = min(position, self._max_ctx) if self._ctx_unified_cache else 0

    def _write_ctx_len(
        self, updates: dict[int, int], base_updates: Optional[dict[int, int]] = None
    ) -> None:
        """Apply a slot-to-length mapping to _ctx_len in one async scatter.

        ``base_updates`` comes from main's ``DFlashSpecMetadata.prepare()``; this worker has no
        base pages (its slots always read from position 0), so it is ignored.
        """
        if not updates:
            return
        for slot, value in updates.items():  # Update host side mirror
            self._ctx_len_host[slot] = value

        slots, values = zip(*updates.items())
        pinned = prefer_pinned()
        slots_pinned = torch.tensor(slots, dtype=torch.long, pin_memory=pinned)
        vals_pinned = torch.tensor(values, dtype=torch.long, pin_memory=pinned)
        self._ctx_len.index_copy_(
            0,
            slots_pinned.to("cuda", non_blocking=True),
            vals_pinned.to("cuda", non_blocking=True),
        )

    def _restore_ctx_len_host(self) -> None:
        """Roll the host shadows back to their pre-forward snapshot."""
        if self._saved_ctx_len_host is not None:
            self._ctx_len_host = list(self._saved_ctx_len_host)
        if self._saved_req_ctx_pos is not None:
            self._req_ctx_pos = dict(self._saved_req_ctx_pos)

    def _store_prefill_context(
        self,
        draft_model,
        spec_metadata: "DFlashSpecMetadata",
        attn_metadata,
        position_ids: torch.Tensor,
        total_target_tokens: int,
    ):
        """Capture prefill hidden states and store as initial accumulated context.

        During prefill (context requests), the target model processes all prompt
        tokens.  We project their captured hidden states through fc + hidden_norm
        and store them per-request so the draft model can use the full prompt
        as cross-attention context on subsequent gen steps.
        """
        if not hasattr(draft_model, "fc") or not hasattr(draft_model, "hidden_norm"):
            return

        num_ctx_tokens = attn_metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            return

        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        if captured_hs is None:
            return
        # Project context tokens through fc + hidden_norm
        ctx_proj = draft_model.project_target_hidden(captured_hs[:num_ctx_tokens])

        # Split by request and store/append accumulated context.
        # Context requests may arrive in chunks (chunked prefill), so we must
        # APPEND successive chunks for the same request rather than overwriting.
        # A chunk is only an append if it starts where the last one ended

        # Scalar (non-rope) token indices on the host. The device tensor holds
        # the same values, but reading it costs a D2H sync per request.
        host_pos = spec_metadata.host_position_ids
        # Slot -> value, uploaded once after the loop: assigning a Python scalar
        # through a CUDA index copies from pageable memory and waits for the
        # stream, i.e. for the whole target forward of this step. A failure
        # mid-loop needs no partial upload: the forward's snapshot restores both.
        ctx_len_updates = {}
        offset = 0
        num_contexts = attn_metadata.num_contexts
        shard = self._helix_shard()
        # A context write addresses pages through row i of the block table, so
        # it must fit that row's allocation as well as the arena's length. The
        # unified draft cache read this iteration's counts from the manager on
        # the host; reading them back from the device would wait for the whole
        # target forward of this step, so only a separate draft manager's table
        # needs one tolist() for the loop. Under helix each local page stands
        # for cp_size global ones.
        cp = 1 if shard is None else shard.cp_size
        page_span = cp * self._ctx_page_size
        host_counts = self._ctx_block_counts_host
        if self._ctx_block_tables is None:
            ctx_alloc = None
        elif host_counts is not None and len(host_counts) >= num_contexts:
            ctx_alloc = [count * page_span for count in host_counts[:num_contexts]]
        else:
            ctx_alloc = (self._ctx_block_counts[:num_contexts] * page_span).tolist()
        for i in range(num_contexts):
            req_id = spec_metadata.request_ids[i]
            slen = int(attn_metadata._seq_lens[i])
            chunk_proj = ctx_proj[offset : offset + slen].detach()
            chunk_pos = position_ids[offset : offset + slen].long().detach()

            if slen == 0:
                first_pos = 0
            elif host_pos is not None:
                first_pos = int(host_pos[offset])
            else:
                first_pos = chunk_pos[0].item()

            # A chunk continues what we already stored only if it starts exactly
            # where the previous one ended. Everything else -- a fresh request, a
            # request id reused after completion, a prefill restarted after
            # preemption -- has to start from a clean slot.
            reset = self._req_ctx_pos.get(req_id) != first_pos
            if self._assign_slot(req_id, reset=reset, updates=ctx_len_updates) is None:
                logger.warning("DFlash: no free slots, skipping context store")
                self._req_ctx_pos.pop(req_id, None)
                offset += slen
                continue

            slot = self._req_to_slot[req_id]
            cur = ctx_len_updates.get(slot, self._ctx_len_host[slot])
            cap = self._max_ctx if ctx_alloc is None else min(self._max_ctx, ctx_alloc[i])
            if cur == 0 and first_pos > 0 and self._ctx_reuse_addressable:
                # Reuse let the target skip [0, first_pos), so only the tail
                # reaches the capture buffer. The prefix K/V are still in the
                # paired pool: precompute_context_kv is per-token in (hidden, p).
                cur = first_pos
            if cur + slen > cap:
                # Request-level, like the no-free-slots path above: truncating
                # would silently draft from a stale prefix, but killing the
                # forward would take every other in-flight request with it.
                logger.warning(
                    f"DFlash: ctx overflow on slot {slot} "
                    f"({cur} + {slen} > {cap}); skipping its context "
                    "store, so this request drafts nothing."
                )
                self._req_to_slot.pop(req_id, None)
                self._req_ctx_pos.pop(req_id, None)
                ctx_len_updates[slot] = 0
                self._free_slots.append(slot)
                offset += slen
                continue
            end = cur + slen
            actual = end - cur
            # Anchor to the true position rather than accumulating, and advance
            # even when the slot is full so the next chunk still reads as a
            # continuation.
            self._req_ctx_pos[req_id] = first_pos + slen
            if actual > 0:
                # The latent projection runs in the drafter's weight dtype; the store converts its
                # outputs to the FP8 pool.
                chunk_proj_cast = chunk_proj[:actual].to(self._ctx_latent_dtype(draft_model))
                ctx_len_updates[slot] = end
                # Precompute post-norm/post-RoPE K,V for this prefill chunk
                # so decode iters can read without re-projecting.
                #
                # RoPE positions must be slot-local (cur..end), NOT the
                # target-forward positions in chunk_pos: gen-time appends and
                # draft queries number positions from _ctx_len, so the stored
                # keys must share that numbering. Without KV-block reuse the
                # two coincide, but on a prefix-cache hit chunk_pos starts at
                # the reused-token count, which would offset every stored key
                # from the numbering the queries use (RoPE attention depends
                # only on position differences, so the uniform slot-local
                # renumbering is exact).
                local_pos = torch.arange(cur, end, dtype=torch.long, device="cuda")
                chunk_k, chunk_v = draft_model.precompute_context_kv(chunk_proj_cast, local_pos)
                # chunk_k/v: [actual, L, nkv, hd] → [L, actual, nkv, hd]
                # The manager's block tables are keyed by batch position.
                positions, store_k = local_pos, chunk_k
                if shard is not None:
                    # Eager prefill: keep only this rank's pages.
                    positions, owned = shard.local_slots(positions)
                    positions, store_k = positions[owned], chunk_k[owned]
                self._store_context_kv_paged(
                    store_k,
                    None,
                    torch.full((positions.numel(),), i, dtype=torch.long, device="cuda"),
                    positions,
                )
            offset += slen

        self._write_ctx_len(ctx_len_updates)

    def _check_no_real_generation(
        self, spec_metadata, num_contexts: int, batch_size: int, kv_cache_manager
    ):
        """Reject a generation request on a context-KV-only drafter.

        Only dummies reach a context server's generation rows: warmup and
        attention-DP padding requests (flagged by the target KV cache manager)
        and CUDA-graph padding. A real one means the server is not context-only.
        """
        is_dummy = getattr(kv_cache_manager, "_request_id_to_is_dummy", {})
        real = [
            request_id
            for request_id in spec_metadata.request_ids[num_contexts:batch_size]
            if request_id != ATTENTION_DP_DUMMY_REQUEST_ID
            and request_id < self._graph_dummy_id_floor
            and not is_dummy.get(request_id, False)
        ]
        if real:
            raise RuntimeError(
                f"Generation requests {real[:4]} reached a DSpark drafter built with "
                "speculative_config.context_kv_only, which never drafts; set it only "
                "on disaggregated context servers."
            )

    def _ensure_spec_dec_state_restored(self, attn_metadata, spec_metadata):
        # Restore first (in warmup mode kv_lens_cuda was saved and comes back
        # wholesale), then apply any pending rewind for the other modes so a
        # failed draft forward does not leave kv_lens_cuda incremented.
        super()._ensure_spec_dec_state_restored(attn_metadata, spec_metadata)
        if (
            getattr(self, "_kv_rewind_pending", False)
            and attn_metadata is not None
            and spec_metadata is not None
        ):
            self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)
        if (
            getattr(self, "_ctx_len_restore_pending", False)
            and self._ctx_len is not None
            and self._saved_ctx_len is not None
        ):
            # A failed forward must not keep this iteration's in-place
            # _ctx_len updates: roll back to the pre-forward snapshot.
            self._ctx_len.copy_(self._saved_ctx_len)
            self._restore_ctx_len_host()
            self._ctx_len_restore_pending = False

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits
        logits_vocab_shard = False
        K = spec_metadata.runtime_draft_len
        if isinstance(logits, VocabShardedLogits):
            if K > 0 and self._vocab_parallel_target_selection_applies(spec_metadata):
                # The target tokens come from the shard: this step returns
                # the shard, which the engine gathers only for a logits
                # post-processor.
                raw_logits = logits.local
                logits_vocab_shard = True
            else:
                logits = raw_logits = logits.gather()

        if K == 0:
            return self.skip_drafting(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
            )

        # Lazy init buffers and attach worker reference for prepare()
        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)
        previous_ctx_len = self._ctx_len
        self._lazy_init_ctx_buffers(
            draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
        )
        spec_metadata._dflash_worker = self
        if (
            self._ctx_unified_cache
            and self._ctx_len is not previous_ctx_len
            and spec_metadata.request_ids
        ):
            # New buffers clear request slots; seed them before generation.
            spec_metadata.prepare()
        # Before any store: prefill and decode both address pages through it.
        # Returning False here means an empty batch -- the missing-offsets case
        # raises inside, with the metadata type in the message.
        self._refresh_ctx_block_tables(attn_metadata, batch_size, spec_metadata.request_ids)

        # Save context lengths so both warmup and a failed forward can roll
        # back the in-place _ctx_len updates made during drafting.
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if not torch.cuda.is_current_stream_capturing():
            # Never allocate the snapshot while capturing a CUDA graph: the
            # clone would live in the graph memory pool, and its replay-time
            # writes could alias blocks reused by later captures. Rollback is
            # only meaningful for eager/warmup forwards anyway; a failure
            # during capture aborts the graph itself, and captured ops do not
            # mutate _ctx_len until replay.
            self._saved_ctx_len = self._ctx_len.clone()
            self._saved_ctx_len_host = list(self._ctx_len_host)
            self._saved_req_ctx_pos = dict(self._req_ctx_pos)
            self._ctx_len_restore_pending = True

        self._ctx_side = self._launch_ctx_side(
            draft_model, spec_metadata, attn_metadata, input_ids.shape[0]
        )

        self._execute_guided_decoder_if_present(logits)

        fused_accept = self._fused_select_accept(logits, attn_metadata, spec_metadata)
        if fused_accept is not None:
            accepted_tokens, num_accepted_tokens, bookkeeping_done = fused_accept
        else:
            bookkeeping_done = False
            accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
                logits, attn_metadata, spec_metadata
            )

        # Update GDN/Mamba recurrent states to the accepted token's state.
        if (
            not bookkeeping_done
            and num_gens > 0
            and isinstance(attn_metadata.kv_cache_manager, MambaHybridCacheManager)
        ):
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
            )

        if not bookkeeping_done:
            self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
            self._prepare_kv_for_draft_forward(
                attn_metadata, num_accepted_tokens, num_contexts, batch_size
            )

        # Collapse mrope [3, 1, N] to 1D by taking the first (temporal) dimension.
        # The draft model uses standard 1D RoPE, so only scalar positions are needed.
        if position_ids.ndim == 3:
            position_ids = position_ids[0, 0]
        else:
            position_ids = position_ids.squeeze(0)

        # Get total tokens processed by target model (for hidden state extraction)
        total_target_tokens = input_ids.shape[0]

        # Capture prefill (context) hidden states for future gen steps.
        # This gives the draft model the full prompt context, not just gen tokens.
        if num_contexts > 0:
            self._store_prefill_context(
                draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
            )
            # Rebuild batch_to_slot after prefill assigns new slots
            if self._ctx_buf_inited and spec_metadata.request_ids:
                self._rebuild_batch_to_slot(spec_metadata.request_ids)

        # A context-KV-only drafter (disaggregated context server) stores the
        # prompt's drafter context above and never drafts. Main has no
        # speculative_config.context_kv_only yet, so a drafter reports False.
        drafts = not getattr(draft_model, "context_kv_only", False)
        if not drafts and num_gens > 0:
            self._check_no_real_generation(
                spec_metadata, num_contexts, batch_size, attn_metadata.kv_cache_manager
            )
        inputs = (
            self.prepare_1st_drafter_inputs(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
                total_target_tokens=total_target_tokens,
            )
            if drafts
            else None
        )

        vocab_size = None
        if num_gens > 0 and drafts:
            with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
                # The fused preparation's MLA block fixup operands place the block in the pages.
                fixup_kwargs = (
                    {"block_fixup": inputs["mla_block_fixup"]}
                    if inputs.get("mla_block_fixup") is not None
                    else {}
                )
                hidden_states_out = draft_model.dflash_forward(
                    noise_embedding=inputs["noise_embedding"],
                    query_positions=inputs["query_positions"],
                    num_ctx_per_req=inputs["num_ctx_per_req"],
                    ctx_k_cache=inputs["ctx_k_cache"],
                    ctx_v_cache=inputs["ctx_v_cache"],
                    ctx_cache_batch_idx=inputs["ctx_cache_batch_idx"],
                    ctx_kv_cache=inputs["ctx_kv_cache"],
                    ctx_page_table=inputs["ctx_page_table"],
                    block_placement=inputs["block_placement"],
                    **fixup_kwargs,
                )

                # Gather K logits per gen request from the block outputs.
                # hidden_states_out is flat: [num_gens * block_size, hidden_dim].
                # Which block slots carry them is a drafter-family convention,
                # resolved through _draft_slot_ids.
                block_size = self._compute_block_size
                gen_gather_ids = self._block_gather_ids(
                    draft_model, num_gens, block_size, K, hidden_states_out.shape[0]
                )

                # Draft-token selection through the vocab-parallel draft head.
                draft_head = self._draft_vocab_parallel_head(draft_model, spec_metadata)
                if draft_head is not None:
                    gen_draft_tokens, gen_hidden_states = self._k3_vocab_parallel_draft_tokens(
                        draft_head,
                        draft_model,
                        hidden_states_out,
                        gen_gather_ids,
                        num_gens,
                        K,
                        inputs,
                    )
                    vocab_size = draft_head.vocab_size
                else:
                    # Served K3 always selects the draft tokens through the vocab-parallel
                    # head (greedy drafts, shared vocab, Markov head).
                    raise RuntimeError(
                        "K3 DSpark: draft tokens need the vocab-parallel draft head (greedy "
                        "draft sampling, no guided decoder, shared vocab)"
                    )

        else:
            # Warmup and attention-DP padding rows of a context-KV-only drafter
            # get the same zero placeholder as context requests.
            gen_draft_tokens = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")

        # Context requests are not drafted by the block worker (zero placeholder
        # token); fill their draft-prob slot rows with a one-hot placeholder so
        # they are a legal distribution when they become gen requests next iter.
        self.write_context_onehot_draft_probs(spec_metadata, num_contexts, num_gens, K, vocab_size)

        if num_contexts > 0 and num_gens > 0:
            ctx_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        elif num_contexts > 0:
            next_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
        else:
            next_draft_tokens = gen_draft_tokens

        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        next_new_tokens = None
        fused_rewind = self._fused_kv_rewind_operands(attn_metadata, spec_metadata, batch_size)
        if (
            fused_rewind is not None
            and next_draft_tokens.shape[0] == batch_size
            and next_draft_tokens.is_contiguous()
            and accepted_tokens.stride(-1) == 1
        ):
            # The rewind rides in the next-new-tokens kernel.
            self._kv_rewind_pending = False
            next_new_tokens = fused_next_new_tokens_rewind(
                accepted_tokens[:batch_size],
                next_draft_tokens,
                num_accepted_tokens[:batch_size],
                *fused_rewind,
            )
        else:
            self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)

        self._rollback_guided_decoder_after_verify(num_accepted_tokens)

        if next_new_tokens is None:
            next_new_tokens = self._prepare_next_new_tokens(
                accepted_tokens,
                next_draft_tokens,
                spec_metadata.batch_indices_cuda,
                batch_size,
                num_accepted_tokens,
            )

        # Restore context lengths after warmup; real runs keep the updates.
        if is_warmup:
            self._ctx_len.copy_(self._saved_ctx_len)
            self._restore_ctx_len_host()
        self._ctx_len_restore_pending = False

        return {
            # The gathered float32 target logits, or this rank's vocab shard
            # when the target tokens were selected from it.
            "logits": raw_logits,
            "logits_vocab_shard": logits_vocab_shard,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def _launch_ctx_side(self, draft_model, spec_metadata, attn_metadata, total_target_tokens):
        """Start the context-latent chain on a side stream.

        The step's context latents (TP-sharded fc + all-reduce with hidden_norm,
        the all-layer latent GEMM, RMSNorm + RoPE at ``ctx_len[slot] + j``) only
        need the captured target hidden states, so they run concurrently with
        the target-token selection and acceptance. The all-reduce uses a private
        MNNVL workspace (the main stream's all-reduces run meanwhile). Same
        kernels and inputs as the main-stream path, so the same values.

        Returns ``(latents [L, N, D], positions-read event, done event)``, or None
        when the path does not apply (then the main stream computes them).
        """
        if draft_model is None or getattr(draft_model, "context_kv_only", False):
            return None
        num_gens = attn_metadata.num_seqs
        if attn_metadata.num_contexts != 0 or num_gens == 0 or num_gens > TAIL_V2_MAX_GENS:
            return None
        if not self._fused_gen_drafter_inputs_applies(draft_model, spec_metadata):
            return None
        # A TP-sharded drafter fc publishes its shard as (weight, begin, end,
        # all_reduce); without one the main stream computes the latents.
        shard = getattr(draft_model, "_fc_shard", None)
        layer_major = getattr(draft_model, "precompute_context_kv_layer_major", None)
        if not shard or layer_major is None:
            return None
        weight, begin, end, all_reduce = shard
        if all_reduce.mnnvl_allreduce is None or draft_model.hidden_norm.eps is None:
            return None
        width = self._draft_tokens_per_req
        tokens = num_gens * width
        hidden = weight.shape[0]
        ar = self._ctx_side_ar
        if ar is None or not ar.fits(tokens, hidden):
            if torch.cuda.is_current_stream_capturing():
                return None
            # Collective: every rank reaches this at the same warmup step.
            ar = PrivateMNNVLAllReduce(all_reduce.mapping, weight.dtype, max(tokens, 256), hidden)
            self._ctx_side_ar = ar
            self._ctx_side_stream = torch.cuda.Stream()
        main = torch.cuda.current_stream()
        side = self._ctx_side_stream
        fork = torch.cuda.Event()
        fork.record(main)
        side.wait_event(fork)
        positions_read = torch.cuda.Event()
        done = torch.cuda.Event()
        with torch.cuda.stream(side):
            positions = dflash_ctx_positions(
                self._ctx_len, self._batch_to_slot_buffer.tensor[:num_gens], width
            )
            positions_read.record(side)
            captured = spec_metadata.get_hidden_states(total_target_tokens)[:tokens]
            captured = captured.to(draft_model.fc.weight.dtype)
            partial = F.linear(captured[:, begin:end], weight)
            projected = ar.rms_norm(
                partial, draft_model.hidden_norm.weight, draft_model.hidden_norm.eps
            )
            projected = projected.to(self._ctx_latent_dtype(draft_model))
            latent = layer_major(projected, positions)
            done.record(side)
        if latent is None:
            main.wait_event(done)
            return None
        # Consumed on the main stream after it waits on ``done``.
        latent.record_stream(main)
        return latent, positions_read, done

    def _fused_select_accept(self, logits, attn_metadata, spec_metadata):
        """Target selection, strict acceptance and the drafter's KV-length prep in one kernel.

        Applies to vocab-parallel target selection through the private head
        with strict, uniform-width acceptance.
        It writes what ``sample_and_accept_draft_tokens`` and
        ``_prepare_kv_for_draft_forward`` write, after
        ``_prepare_attn_metadata_for_dflash`` (moved ahead: it only saves
        metadata tensors the acceptance does not touch). Returns ``(accepted,
        num_accepted, bookkeeping_done)``, or None to take the unfused path.

        With a mamba-hybrid target the kernel also writes what
        ``update_mamba_states`` writes when that is only the KDA replay record
        (:meth:`_kda_replay_operands`). Otherwise it fuses only the selection
        and the acceptance (``bookkeeping_done`` False), and
        ``update_mamba_states`` and the KV-length prep run as on the unfused
        path, in the same order.
        """
        self._fused_kv_rewind_armed = False
        if not isinstance(logits, VocabShardedLogits):
            return None
        if logits.head.exchange.backend != "mnnvl" and logits.head.exchange.tp_size > 1:
            return None
        # Main's ragged verification packs per-request windows; the kernel
        # reads a uniform K + 1 tokens per generation request.
        if spec_metadata.is_ragged_verify:
            return None
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        if num_gens > TAIL_V2_MAX_GENS:
            return None
        if num_gens > 0 and self._can_use_rejection_sampling(spec_metadata):
            return None
        draft_tokens = self._reshape_draft_tokens_for_accept(spec_metadata, num_gens, logits.device)
        pairs = logits.select_pairs(spec_metadata)
        force = self._force_accept_kernel_args(draft_tokens.shape[-1], spec_metadata, logits.device)
        kda_replay = None
        if num_gens > 0 and isinstance(attn_metadata.kv_cache_manager, MambaHybridCacheManager):
            fusable, kda_replay = self._kda_replay_operands(attn_metadata)
            if not fusable:
                accepted, num_accepted, _ = fused_select_accept(
                    pairs, draft_tokens, num_contexts, batch_size, force
                )
                return accepted, num_accepted, False
        self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
        kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
        accepted, num_accepted, rewind = fused_select_accept(
            pairs,
            draft_tokens,
            num_contexts,
            batch_size,
            force,
            kda_replay=kda_replay,
            kv_lens=kv_lens,
        )
        if kv_lens is not None:
            self._kv_rewind_amount = rewind
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size
            self._kv_rewind_pending = True
            attn_metadata.update_for_spec_dec()
        self._fused_kv_rewind_armed = kv_lens is not None
        return accepted, num_accepted, True

    @staticmethod
    def _kda_replay_operands(attn_metadata):
        """``(fusable, kda_replay)`` for :func:`fused_select_accept` with a
        mamba-hybrid target.

        Fusable when ``update_mamba_states(attn_metadata, num_accepted,
        state_indices=mamba_metadata.state_indices)`` is
        ``KimiK3HybridCacheManager``'s and writes nothing but
        ``KDAReplayState.record_acceptance``: the replay state is the
        runtime's ``KDAReplayState`` or the checkpoint state
        (``K3UllKdaCheckpointState``), whose ``prev_num_accepted_tokens`` the
        next step's ``checkpoint_indices`` reads back as the accepted
        checkpoint. ``kda_replay`` is then ``(prev_num_accepted_tokens,
        state_indices, dummy_mask)``, or None when that call records nothing (a
        context-only KDA manager, or no local KDA layers).
        """
        mgr = attn_metadata.kv_cache_manager
        if (
            not isinstance(mgr, KimiK3HybridCacheManager)
            or type(mgr).update_mamba_states is not KimiK3HybridCacheManager.update_mamba_states
        ):
            return False, None
        if mgr.kda_context_only or mgr.local_num_mamba_layers == 0:
            return True, None
        replay = mgr._speculative_state
        if (
            type(replay) not in (KDAReplayState, K3UllKdaCheckpointState)
            or replay.prev_num_accepted_tokens is None
            or mgr._dummy_request_mask is None
        ):
            return False, None
        state_indices = attn_metadata.mamba_metadata.state_indices
        if state_indices is None:
            state_indices = mgr.get_state_indices()
        dummy_mask = mgr._dummy_request_mask
        if state_indices.stride(-1) != 1 or dummy_mask.stride(-1) != 1:
            return False, None
        return True, (replay.prev_num_accepted_tokens, state_indices, dummy_mask)

    def _fused_kv_rewind_operands(self, attn_metadata, spec_metadata, batch_size: int):
        """``(kv_lens, amounts, num_contexts)`` when the rewind after the draft
        forward runs inside the next-new-tokens kernel, else None (the
        ``_apply_kv_rewind_after_draft`` path, which also covers warmup)."""
        if not self._fused_kv_rewind_armed:
            return None
        self._fused_kv_rewind_armed = False
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if (
            is_warmup
            or self._kv_rewind_amount is None
            or not hasattr(attn_metadata, "kv_lens_cuda")
            or self._kv_rewind_bs != batch_size
        ):
            return None
        return attn_metadata.kv_lens_cuda, self._kv_rewind_amount, self._kv_rewind_nc

    def _force_accept_kernel_args(
        self, runtime_draft_len: int, spec_metadata, device: torch.device
    ) -> Optional[tuple]:
        """The ``force`` argument of ``fused_strict_accept`` / ``fused_select_accept``
        that reproduces :meth:`_apply_force_accepted_tokens` (same warmup skip
        and draws)."""
        if self.force_num_accepted_tokens == 0.0:
            return None
        if (
            spec_metadata is not None
            and spec_metadata.is_cuda_graph
            and not torch.cuda.is_current_stream_capturing()
        ):
            return None
        int_part = int(self.force_num_accepted_tokens)
        frac_part = self.force_num_accepted_tokens - int_part
        max_total = runtime_draft_len + 1
        base_total = min(int_part + 1, max_total)
        if frac_part > 0.0 and base_total < max_total:
            self._ensure_force_accept_rng_state(device)
            return (
                base_total,
                frac_part,
                max_total,
                self._force_accept_rng_pool,
                self._force_accept_rng_counter,
                _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
                _FORCE_ACCEPT_RNG_SLOT_STRIDE,
            )
        return (base_total,)

    def _sample_and_accept_draft_tokens_base(
        self,
        logits,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        spec_metadata,
    ):
        """Strict acceptance and the force override in one launch (``fused_strict_accept``);
        main's implementation under ragged verification."""
        if spec_metadata is not None and spec_metadata.is_ragged_verify:
            return super()._sample_and_accept_draft_tokens_base(
                logits, draft_tokens, num_contexts, batch_size, spec_metadata
            )
        if isinstance(logits, torch.Tensor) and logits.dim() == 1:
            logits = logits.unsqueeze(0)
        target_tokens = self._sample_tokens_for_batch(
            logits, spec_metadata, num_contexts, batch_size
        )
        return fused_strict_accept(
            target_tokens,
            draft_tokens,
            num_contexts,
            batch_size,
            self._force_accept_kernel_args(
                draft_tokens.shape[-1], spec_metadata, target_tokens.device
            ),
        )

    def _sample_tokens_for_batch(
        self,
        logits,
        spec_metadata,
        num_contexts: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Main's sampler, which also takes the private head's shard logits.

        The worker only passes shard logits through when vocab-parallel
        selection covers this batch's sampling mode.
        """
        if isinstance(logits, VocabShardedLogits):
            return logits.select_tokens(spec_metadata)
        return super()._sample_tokens_for_batch(logits, spec_metadata, num_contexts, batch_size)

    def _block_gather_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int, num_rows: int
    ) -> torch.Tensor:
        """The block-output rows gathered for the K draft logits per request.

        Shape-only: built once per shape and reused instead of being recomputed every step.
        """
        key = (num_gens, block_size, num_draft_tokens, num_rows)
        cache = self.__dict__.setdefault("_block_gather_id_cache", {})
        if key in cache:
            return cache[key]
        ids = self._draft_slot_ids(draft_model, num_gens, block_size, num_draft_tokens)
        # Shields only the last request: at block_size == K with shift_label
        # off, slots run 1..K, so every request reads the next one's slot 0
        # and the last overruns. Degrades, never raises.
        ids = ids.clamp(max=num_rows - 1)
        if not torch.cuda.is_current_stream_capturing():
            cache[key] = ids
        return ids

    def _prepare_next_new_tokens(
        self,
        accepted_tokens,
        next_draft_tokens,
        batch_indices_cuda,
        batch_size,
        num_accepted_tokens,
    ):
        # batch_indices_cuda is the identity here (DFlashSpecMetadata.prepare).
        if (
            next_draft_tokens.shape[0] == batch_size
            and next_draft_tokens.is_contiguous()
            and accepted_tokens.stride(-1) == 1
        ):
            return fused_next_new_tokens(
                accepted_tokens[:batch_size], next_draft_tokens, num_accepted_tokens[:batch_size]
            )
        return super()._prepare_next_new_tokens(
            accepted_tokens, next_draft_tokens, batch_indices_cuda, batch_size, num_accepted_tokens
        )

    def _draft_first_slot(self, draft_model) -> int:
        return 0 if getattr(draft_model, "_dspark_shift_label", False) else 1

    def _block_gather_range(
        self, draft_model, num_gens: int, num_draft_tokens: int, num_rows: int
    ) -> Optional[tuple]:
        """``(start, stop)`` when the gathered block rows are one contiguous range
        (one request), else None."""
        if num_gens != 1:
            return None
        first = self._draft_first_slot(draft_model)
        stop = first + num_draft_tokens
        return (first, stop) if stop <= num_rows else None

    def _draft_block_width(self, draft_model) -> int:
        """Block width under the dspark ``shift_label`` convention.

        shift_label reads slots 0..K-1, so K draft tokens fit in K slots and
        the base class' K+1 over-demands by one -- enough to reject a block-7
        checkpoint at max_draft_len=7, which is how both published DSpark
        drafters are meant to run.
        """
        if getattr(draft_model, "_dspark_shift_label", False):
            return self.max_draft_len
        return self.max_draft_len + 1

    def _draft_slot_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int
    ) -> torch.Tensor:
        """Block-output slots under the dspark ``shift_label`` convention.

        The drafter checkpoint declares the convention, so it is read off the
        draft model rather than assumed: a DSpark drafter trained with the
        legacy DFlash slot layout keeps the base class' slots 1..K.
        """
        shift_label = getattr(draft_model, "_dspark_shift_label", False)
        return dflash_draft_slot_ids(
            num_gens, block_size, num_draft_tokens, shift_label, device="cuda"
        )

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DFlashSpecMetadata,
        draft_model: nn.Module,
        total_target_tokens: int = 0,
    ):
        """The drafter's inputs of this step: :meth:`_prepare_gen_drafter_inputs_fused` for the
        generation rows (served ``TLLM_SPEC_TAIL_FUSION`` path, the only one this target runs),
        empty inputs for a context-only batch."""
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts

        if self._resolved_mask_token_id is None:
            if getattr(self.spec_config, "mask_token_id", None) is not None:
                self._resolved_mask_token_id = self.spec_config.mask_token_id
            elif hasattr(draft_model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.mask_token_id
            else:
                raise ValueError(
                    "DSpark requires mask_token_id in DFlashDecodingConfig or the drafter config."
                )
        mask_token_id = self._resolved_mask_token_id
        # The drafter's own trained mask row, if it kept one for the mask id in use.
        trained_mask_embedding = (
            getattr(draft_model, "mask_token_embedding", None)
            if mask_token_id == getattr(draft_model, "mask_token_id", None)
            else None
        )

        if num_gens > 0:
            if not self._fused_gen_drafter_inputs_applies(draft_model, spec_metadata):
                raise RuntimeError(
                    "K3 DSpark: the fused drafter-input preparation does not cover this step "
                    "(it needs the paged single-latent draft pool and K + 1 verify rows)"
                )
            return self._prepare_gen_drafter_inputs_fused(
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                spec_metadata,
                draft_model,
                draft_model.draft_model_full.model.embed_tokens,
                total_target_tokens,
                mask_token_id,
                trained_mask_embedding,
            )

        hidden_dim = (
            spec_metadata.hidden_size if spec_metadata.hidden_size > 0 else hidden_states.shape[-1]
        )
        empty = torch.empty(0, dtype=torch.long, device="cuda")
        return {
            "noise_embedding": hidden_states.new_empty(0, 0, hidden_dim),
            "query_positions": torch.empty(0, 0, dtype=torch.long, device="cuda"),
            "num_ctx_per_req": empty,
            "ctx_k_cache": None,
            "ctx_v_cache": None,
            "ctx_cache_batch_idx": empty,
            "first_prev_tokens": empty,
            "ctx_kv_cache": self._ctx_kv_buf,
            "ctx_page_table": self._ctx_block_tables,
            "block_placement": None,
        }

    def _ctx_latent_dtype(self, draft_model) -> torch.dtype:
        """Dtype the context latents are computed in: the projection's activation dtype (BF16);
        the stores quantize them into the FP8 pool."""
        return dflash_context_dtype(draft_model)

    def _fused_gen_drafter_inputs_applies(self, draft_model, spec_metadata) -> bool:
        """Whether ``_prepare_gen_drafter_inputs_fused`` covers this step: a generation batch
        drafting from a single-latent (MLA) context that lives in the draft KV manager's paged
        pool."""
        return (
            self._ctx_paged
            and self._ctx_block_tables is not None
            and getattr(draft_model, "_kv_factor", 2) == 1
            and hasattr(draft_model, "fc")
            and hasattr(draft_model, "hidden_norm")
            # The fused latent store converts the projection's latents to FP8.
            and self._ctx_kv_buf[0].dtype == torch.float8_e4m3fn
            and spec_metadata.get_hidden_states(0) is not None
            and spec_metadata.runtime_draft_len + 1 == self._draft_tokens_per_req
            and self._compute_block_size is not None
        )

    def _ctx_kv_pool_ptrs(self) -> torch.Tensor:
        """Device array of the per-layer context-pool base addresses (cached)."""
        ptrs = tuple(buf.data_ptr() for buf in self._ctx_kv_buf)
        if getattr(self, "_ctx_kv_ptrs_key", None) != ptrs:
            self._ctx_kv_ptrs = torch.tensor(ptrs, dtype=torch.int64, device="cuda")
            self._ctx_kv_ptrs_key = ptrs
        return self._ctx_kv_ptrs

    def _fused_prepare_constants(
        self, num_contexts: int, num_gens: int, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The block fixup's upper-triangle mask and the gen rows' batch indices.

        Shape-only: built once per shape outside graph capture and reused, so no kernel
        recomputes them per step.
        """
        key = (num_contexts, num_gens, block_size)
        cache = self.__dict__.setdefault("_fused_prepare_constant_cache", {})
        if key in cache:
            return cache[key]
        idx = torch.arange(block_size, device="cuda")
        valid = (idx.view(1, block_size) > idx.view(block_size, 1)).view(
            1, block_size, 1, block_size
        )
        batch_idx = torch.arange(
            num_contexts, num_contexts + num_gens, dtype=torch.long, device="cuda"
        )
        if not torch.cuda.is_current_stream_capturing():
            cache[key] = (valid, batch_idx)
        return valid, batch_idx

    def _prepare_gen_drafter_inputs_fused(
        self,
        accepted_tokens,
        num_accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
        embed_tokens,
        total_target_tokens: int,
        mask_token_id: int,
        trained_mask_embedding: Optional[torch.Tensor] = None,
    ) -> dict:
        """The drafter inputs of the generation rows from the paged single-latent context: the
        per-request bookkeeping in one kernel (``dflash_gen_prepare``: context lengths advanced,
        ``seq_lens`` set, block placement), the owner-multicast draft embedding, the noise block in
        one copy kernel and the FP8 context-latent store on the drafter side stream. Also returns
        the block-decode fixup operands ``dflash_forward`` reads."""

        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        width = self._draft_tokens_per_req
        block_size = self._compute_block_size
        slots = self._batch_to_slot_buffer.tensor[num_contexts : num_contexts + num_gens]
        side = self._ctx_side
        self._ctx_side = None
        if side is not None:
            # The side stream has read _ctx_len before prep advances it.
            torch.cuda.current_stream().wait_event(side[1])
        shard = self._helix_shard()
        prep = dflash_gen_prepare(
            num_accepted_tokens,
            accepted_tokens,
            slots,
            self._ctx_len,
            self._ctx_block_counts,
            self._ctx_block_tables,
            attn_metadata._seq_lens_cuda,
            num_contexts,
            num_gens,
            self._dummy_slot,
            self._max_ctx,
            int(mask_token_id),
            self._ctx_page_size,
            block_size,
            width,
            position_offset=None,
            cp_size=1 if shard is None else shard.cp_size,
            cp_rank=0 if shard is None else shard.cp_rank,
        )
        attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = width

        # The vocab-sharded embedding's all-reduce as an owner-rank multicast of each token's row.
        combined_embed = draft_embed_site(embed_tokens, prep["embed_ids"])
        if combined_embed is None:
            raise RuntimeError(
                "K3 DSpark: the drafter embedding is not a vocab-sharded MNNVL EmbeddingTable "
                "(draft_embed_site)"
            )
        if trained_mask_embedding is not None:
            # The drafter's own trained mask row replaces the shared lookup's
            # (the last row).
            combined_embed = torch.cat(
                (
                    combined_embed[:num_gens],
                    trained_mask_embedding.to(combined_embed.dtype).unsqueeze(0),
                )
            )
        noise_embedding = dflash_noise_embedding(combined_embed, block_size)

        if side is not None:
            # Computed on the side stream from the same captured hidden states
            # and the same positions (ctx_len[slot] + j).
            torch.cuda.current_stream().wait_event(side[2])
            latent = side[0].transpose(0, 1)
        else:
            gen_start = attn_metadata.num_ctx_tokens
            captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
            gen_hs = captured_hs[gen_start : gen_start + num_gens * width]
            projected = draft_model.project_target_hidden(gen_hs.reshape(-1, gen_hs.shape[-1]))
            projected = projected.to(self._ctx_latent_dtype(draft_model))
            # Layer-major latents skip the per-layer staging copies when the
            # drafter offers them; the values are the same either way.
            layer_major = getattr(draft_model, "precompute_context_kv_layer_major", None)
            latent = None if layer_major is None else layer_major(projected, prep["ctx_positions"])
            if latent is not None:
                latent = latent.transpose(0, 1)
            else:
                latent = draft_model.precompute_context_kv(projected, prep["ctx_positions"])[0][
                    :, :, 0
                ]
        store_args = (
            latent,
            self._ctx_kv_buf,
            self._ctx_kv_pool_ptrs(),
            prep["store_pages"],
            prep["store_offsets"],
            prep["store_mask"],
        )
        # The store runs beside the drafter's first projections; dflash_forward joins it before
        # its first draft-pool write (the store's masked rows zero slots the block latents then
        # take).
        draft_model.draft3_side_launch(
            lambda: store_mla_context_latents(*store_args),
            keep=(latent, prep["store_pages"], prep["store_offsets"], prep["store_mask"]),
        )

        valid, batch_idx = self._fused_prepare_constants(num_contexts, num_gens, block_size)
        block_fixup = _MLABlockFixup(
            pages=prep["blk_pages"],
            offsets=prep["blk_offsets"],
            valid=valid,
            page_tables_i32=self._ctx_block_tables[num_contexts : num_contexts + num_gens],
            seq_lens_i32=prep["blk_seq_lens"],
            kv_bounds_i32=prep["blk_kv_bounds"],
            owned=None if shard is None else self._blk_owned_mask(prep["blk_owned"]),
        )
        return {
            "noise_embedding": noise_embedding,
            "query_positions": prep["query_positions"],
            "num_ctx_per_req": prep["num_ctx"],
            "ctx_k_cache": None,
            "ctx_v_cache": None,
            "ctx_cache_batch_idx": batch_idx,
            "first_prev_tokens": prep["bonus"],
            "ctx_kv_cache": self._ctx_kv_buf,
            "ctx_page_table": self._ctx_block_tables,
            "block_placement": None,
            "mla_block_fixup": block_fixup,
        }

    @staticmethod
    def _blk_owned_mask(blk_owned: torch.Tensor) -> torch.Tensor:
        """The ``[gens, block, 1]`` bool ownership mask of the prep kernel's 0 / 1 int8 rows,
        reinterpreted in place (same bytes as the conversion)."""
        if blk_owned.dtype in (torch.int8, torch.uint8):
            return blk_owned.view(torch.bool).unsqueeze(-1)
        return blk_owned.bool().unsqueeze(-1)


_DSPARK_HEAD_KEY_ALIASES = {
    # RadixArk/Kimi-K3-DSpark ships ``enable_confidence_head`` top-level.
    "use_confidence_head": ("use_confidence_head", "enable_confidence_head"),
}


def resolve_dspark_head_config(pretrained_config, key):
    """Resolve one DSpark head switch across every spelling in the wild.

    Looks in ``dspark_config``, then ``dflash_config``, then the top level as
    ``dspark_<key>`` and as ``<key>``, for each accepted alias of ``key``.
    Returns ``None`` when the drafter declares the switch nowhere.
    """
    dspark_cfg = getattr(pretrained_config, "dspark_config", None) or {}
    dflash_cfg = getattr(pretrained_config, "dflash_config", None) or {}
    for name in _DSPARK_HEAD_KEY_ALIASES.get(key, (key,)):
        for value in (
            dspark_cfg.get(name),
            dflash_cfg.get(name),
            getattr(pretrained_config, f"dspark_{name}", None),
            getattr(pretrained_config, name, None),
        ):
            if value is not None:
                return value
    return None


# ----------------------------------------------------------------------------
# YaRN RoPE table of the drafter's rope slice.
# ----------------------------------------------------------------------------


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def build_dspark_mla_yarn_rope(
    *,
    dim: int,
    base: float,
    scaling_factor: float,
    original_max_position_embeddings: int,
    max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """YaRN cos/sin tables for an MLA drafter's ``qk_rope_head_dim`` slice.

    Repacks ``RopeEmbeddingUtils.create_sinusoidal_positions_yarn``, the same HF
    DeepSeek-V2 transcription the drafters are distilled under. NOT reused:
    ``RopeParams.create_rope_const_params()`` pins the fused MLA kernel's GPT-J
    packing and hard-codes ``device='cuda'``.

    ``duplicate_data=True`` though only the first half is read as a complex
    pair: the util's two modes disagree by 1 ulp on 6 of 8224 entries, and the
    duplicated one is what every measured drafter run used.

    Returns ``(cos, sin)``, both ``[max_position_embeddings, dim]`` fp32, with
    the frequency half duplicated so ``rotate_half`` applies.
    """
    # The util returns a flat numpy [pos][slot](cos, sin); split the pair.
    _, packed = RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
        num_pos=max_position_embeddings,
        dim=dim,
        base=base,
        scaling_factor=scaling_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        mscale=mscale,
        mscale_all_dim=mscale_all_dim,
        duplicate_data=True,
    )
    table = torch.from_numpy(packed).reshape(max_position_embeddings, dim, 2)
    return (
        table[..., 0].contiguous().to(device),
        table[..., 1].contiguous().to(device),
    )


def build_dspark_mla_yarn_freqs_cis(*, device: str = "cuda", **kwargs) -> torch.Tensor:
    """The same YaRN table as :func:`build_dspark_mla_yarn_rope`, adjacent-pair.

    Returns ``[max_position_embeddings, dim // 2]`` complex64: the convention the
    fused ``cute_dsl_dspark_rmsnorm_rope`` kernel consumes, where the last dim of
    the rotated slice is read as (re, im) pairs.

    HF DeepSeek de-interleaves and rotates halves instead, writing the same
    values to lanes i and i + dim/2 rather than 2i and 2i+1. That permutation
    cancels inside ``q_rope . k_rope``, so either convention works -- but every
    producer of a rope slice must use the SAME one, and mixing them only shows
    up as lower AL. ``mscale`` rides in the modulus, not the phase.
    """
    cos, sin = build_dspark_mla_yarn_rope(device=device, **kwargs)
    half = cos.shape[-1] // 2
    return torch.complex(cos[..., :half], sin[..., :half]).contiguous()


def _runtime_position_cap(model_config, pretrained_config, slack: int = 0) -> int:
    """Fallback RoPE table length: the served length, not the advertised one.

    The drafter sizes a position table from the checkpoint's
    max_position_embeddings. K3 advertises 1,048,576: ~256 MiB per rank as
    complex64, ~768 MiB transient via full-size fp32 cos/sin, while the context
    cache is bounded by the runtime max_seq_len. Only the table length moves;
    YaRN's correction range comes from original_max_position_embeddings.

    Shrinking the table this way is also what made its bound matter: sized from
    max_position_embeddings it could not be indexed past, sized from
    model_config.max_seq_len it can. Hence _runtime_position_ceiling, which
    the drafter prefers over this value whenever a worker has published it.
    """
    runtime = getattr(model_config, "max_seq_len", None)
    return int(runtime or getattr(pretrained_config, "max_position_embeddings", 163840)) + slack


def _resolve_dspark_mla_rope_params(pretrained_config, model_config=None, slack: int = 0) -> Dict:
    """RoPE settings for an MLA drafter, from either HF spelling.

    Transformers v5 checkpoints (TorchSpec) carry ``rope_parameters``; older
    ones carry ``rope_scaling``. Only YaRN is accepted: both published MLA
    drafters use it, and silently treating an unknown scaling as plain RoPE
    would cost acceptance without an error.
    """
    scaling = (
        getattr(pretrained_config, "rope_parameters", None)
        or getattr(pretrained_config, "rope_scaling", None)
        or {}
    )
    rope_type = scaling.get("rope_type") or scaling.get("type") or "default"
    if rope_type != "yarn":
        raise ValueError(
            f"MLA DSpark drafter declares rope_type={rope_type!r}; only 'yarn' is "
            "supported (the block decode builds its own YaRN table)."
        )
    theta = scaling.get("rope_theta") or getattr(pretrained_config, "rope_theta", 10000.0)
    return {
        "theta": float(theta),
        "scaling_factor": float(scaling["factor"]),
        "original_max_positions": int(scaling["original_max_position_embeddings"]),
        # Table length, not YaRN math: the correction range below uses
        # original_max_positions, so capping this at the served length is free.
        # K3 advertises 1,048,576, which would build a ~256 MiB complex64 table
        # per rank -- through full-size fp32 cos/sin, so ~768 MiB transient --
        # for a drafter whose point is 5760 B/token instead of 20480.
        "max_positions": _runtime_position_cap(model_config, pretrained_config, slack),
        "beta_fast": float(scaling.get("beta_fast", 32.0)),
        "beta_slow": float(scaling.get("beta_slow", 1.0)),
        "mscale": float(scaling.get("mscale", 1.0)),
        "mscale_all_dim": float(scaling.get("mscale_all_dim", 0.0)),
    }


# ----------------------------------------------------------------------------
# MLA-shaped drafter backbone (Inferact/Kimi-K3-DSpark): weights only (layers.py
# containers). K3MLADSparkForCausalLM.dflash_forward runs the block decode over them.
# ----------------------------------------------------------------------------


def _draft_mappings(mapping: Mapping) -> tuple[Mapping, Optional[Mapping]]:
    """``(dense, helix)`` of the drafter's mapping.

    ``helix`` is the helix CP mapping attention keeps (the latent cache split by pages over the CP
    ranks), None off helix; ``dense`` is the mapping of everything else, the helix CP ranks
    repurposed as TP ranks.
    """
    if mapping.has_cp_helix():
        return mapping.repurpose_helix_cp_to_tp(), mapping
    if mapping.cp_size > 1:
        raise NotImplementedError(
            f"The MLA DSpark drafter does not support context parallelism "
            f"(cp_size={mapping.cp_size})."
        )
    return mapping, None


class K3MLADSparkAttention(nn.Module):
    """The MLA weights of one drafter layer.

    The low-rank ``*_a`` projections are replicated (every head reads the same latent);
    ``q_b_proj`` / ``kv_b_proj`` are column-sharded by head over the attention TP group and
    ``o_proj`` is row-sharded. Under attention DP the attention TP group is one rank, which is
    also why the drafter's latent cache is unsharded. Under helix the latent cache is split by
    pages over the CP ranks, so every rank holds every head (``q_b`` / ``kv_b`` replicated over
    the CP group) and ``o_proj`` is row-parallel over the repurposed (CP folded into TP) group.
    """

    def __init__(
        self,
        config,
        dense: Mapping,
        helix: Optional[Mapping],
        allreduce_strategy: AllReduceStrategy,
    ):
        super().__init__()
        dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # What the drafter's KV cache stores per token: one MLA latent, no V.
        self.kv_cache_head_dim = self.kv_lora_rank + self.qk_rope_head_dim

        mapping = dense if helix is None else helix
        tp_size, dp_size = mapping.tp_size, 1
        if mapping.enable_attention_dp:
            tp_size, dp_size = 1, mapping.tp_size
        elif helix is not None:
            # CP ranks hold identical head projections: fold them like DP.
            dp_size = mapping.cp_size
        if self.num_heads % tp_size != 0:
            raise ValueError(
                f"DSpark MLA drafter has {self.num_heads} heads, "
                f"not divisible by tp_size {tp_size}."
            )
        self.num_heads_tp = self.num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads_tp // mapping.cp_size
        self.mapping = Mapping(
            world_size=mapping.pp_size * dp_size * tp_size,
            tp_size=tp_size,
            pp_size=mapping.pp_size * dp_size,
            rank=mapping.rank,
            gpus_per_node=mapping.gpus_per_node,
            enable_attention_dp=mapping.enable_attention_dp,
        )
        o_mapping = self.mapping if helix is None else dense
        # Row-parallel o_proj reduces only when the heads are actually split.
        reduce_output = not dense.enable_attention_dp and dense.tp_size > 1
        eps = config.rms_norm_eps
        by_head = dict(
            dtype=dtype,
            tp_size=self.mapping.tp_size,
            tp_rank=self.mapping.tp_rank,
            tp_mode=TensorParallelMode.COLUMN,
        )

        self.q_a_proj = Bf16Linear(self.hidden_size, self.q_lora_rank, dtype=dtype)
        self.q_a_layernorm = RMSNormWeight(self.q_lora_rank, eps, dtype)
        self.q_b_proj = Bf16Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, **by_head)
        self.kv_a_proj_with_mqa = Bf16Linear(self.hidden_size, self.kv_cache_head_dim, dtype=dtype)
        self.kv_a_layernorm = RMSNormWeight(self.kv_lora_rank, eps, dtype)
        self.kv_b_proj = Bf16Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            **by_head,
        )
        self.o_proj = Bf16Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            dtype=dtype,
            tp_size=o_mapping.tp_size,
            tp_rank=o_mapping.tp_rank,
            tp_mode=TensorParallelMode.ROW,
            all_reduce=(
                TpAllReduce(o_mapping, dtype, allreduce_strategy) if reduce_output else None
            ),
        )


class K3MLADSparkDecoderLayer(nn.Module):
    """One drafter layer: the MLA weights, the dense SwiGLU MLP and the two RMSNorms."""

    def __init__(
        self,
        config,
        layer_idx: int,
        dense: Mapping,
        helix: Optional[Mapping],
        allreduce_strategy: AllReduceStrategy,
    ):
        super().__init__()
        dtype, hidden, eps = config.torch_dtype, config.hidden_size, config.rms_norm_eps
        self.layer_idx = layer_idx
        self.self_attn = K3MLADSparkAttention(config, dense, helix, allreduce_strategy)
        # Under attention DP every rank runs the whole MLP: TP groups of one rank.
        self.mlp = GatedMlp(
            hidden,
            config.intermediate_size,
            activation="silu",
            dtype=dtype,
            mapping=subgroup_mapping(dense, 1) if dense.enable_attention_dp else dense,
            allreduce_strategy=allreduce_strategy,
        )
        self.input_layernorm = RMSNormWeight(hidden, eps, dtype)
        self.post_attention_layernorm = RMSNormWeight(hidden, eps, dtype)


class K3MLADSparkModel(nn.Module):
    """The drafter's embedding, layers and final norm."""

    def __init__(self, model_config):
        super().__init__()
        config = model_config.pretrained_config
        dense, helix = _draft_mappings(model_config.mapping)
        strategy = model_config.allreduce_strategy
        # The drafter ships its own trained embedding rather than sharing the target's (verified
        # different weights on Inferact/Kimi-K3-DSpark), so it is loaded from the drafter
        # checkpoint. Vocab-parallel over the dense TP group, the lookup all-reduced at the
        # default strategy.
        self.embed_tokens = EmbeddingTable(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=dense,
            vocab_parallel=True,
        )
        self.layers = nn.ModuleList(
            [
                K3MLADSparkDecoderLayer(config, layer_idx, dense, helix, strategy)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNormWeight(config.hidden_size, config.rms_norm_eps, config.torch_dtype)


class K3MLADSparkBackbone(nn.Module):
    """The drafter backbone (``K3MLADSparkForCausalLM.draft_model_full``): weights only.

    ``K3MLADSparkForCausalLM.dflash_forward`` runs the block decode over ``model``. No ``lm_head``:
    the drafter shares the target's (``K3MLADSparkForCausalLM.load_weights_from_target_model``).
    """

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.model = K3MLADSparkModel(model_config)
        self.lm_head = None


# What the drafter config names once the drafter is built: the draft weight loader picks its
# checkpoint mapper by it (the checkpoint itself says "K3DSparkModel", registered nowhere).
_CHECKPOINT_ARCH = "K3DsparkForCausalLM"


# Both published drafters name the head tensors after the submodules that own them:
# markov_head is a VanillaMarkov, confidence_head an AcceptRatePredictor whose linear is
# ``proj``. The bare spellings are kept for drafters exported without that nesting.
_DSPARK_HEAD_WEIGHT_ALIASES = {
    "markov_w1.weight": ("markov_head.markov_w1.weight", "markov_w1.weight"),
    "markov_w2.weight": ("markov_head.markov_w2.weight", "markov_w2.weight"),
    "confidence_proj.weight": ("confidence_head.proj.weight", "confidence_proj.weight"),
    "confidence_proj.bias": ("confidence_head.proj.bias", "confidence_proj.bias"),
}


# Lower bound of the CuTe DSL MLA decode's workspace (served _MLA_DECODE_WORKSPACE_BYTES).
_MLA_DECODE_WORKSPACE_BYTES = 256 * 1024 * 1024
# Fixed cost of one split-KV piece of the FP8 block decode's per-request split plan, in KV tiles
# (served _MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8).
_MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8 = 60
# The block decode's only tactic options the served runner fixes (CuteDSLNVMlaDecodeBlackwellRunner
# _IS_VAR_SEQ / _IS_VAR_SPLIT_KV), for its can_implement probe.
_DECODE_IS_VAR_SEQ = True
_DECODE_IS_VAR_SPLIT_KV = False


class _MLABlockFixup(NamedTuple):
    """Layer-invariant operands of the MLA block decode.

    The block's own latents are written into the pool at ``ctx_len + j``. Those
    slots belong to this request -- the draft KV manager adds
    ``max(draft_len, max_total_draft_tokens)`` tokens per generation request each
    step (resource_manager.py:1060) -- and the accepted tokens overwrite them next
    step. The decode then reads context and block in one pass, every block token
    bounded at ``ctx_len + block_size`` (``kv_bounds_i32``). Every field depends on
    ctx_len, the page table and the block geometry alone, so it is built once per
    forward.
    """

    pages: torch.Tensor  # [B, block] page holding each block position
    offsets: torch.Tensor  # [B, block] offset of that position inside its page
    valid: torch.Tensor  # [1, block, 1, block] strict upper triangle (unread by the decode)
    page_tables_i32: torch.Tensor  # kernel operands, cast once
    seq_lens_i32: torch.Tensor  # ctx_len + block_size
    kv_bounds_i32: torch.Tensor  # [B * block] seq_lens repeated per query token
    # Helix: [B, block, 1] whether this rank's pages hold each block position;
    # pages/offsets and the lengths above are then rank-local.
    owned: Optional[torch.Tensor] = None


@lru_cache(maxsize=1)
def cute_dsl_mla_decode_unavailability_reason() -> Optional[str]:
    """Why the CuTe DSL FP8 MLA decode (``kernels/mla/mla_decode_fp8.py``, per-token KV bounds)
    cannot run here, or None."""
    if (sm := get_sm_version()) not in (100, 103, 107):
        return f"requires SM100, SM103 or SM107, got SM{sm}"
    try:
        import cutlass  # noqa: F401

        from ..kernels.k3r_mla_decode import mla_decode_fp8  # noqa: F401
    except ImportError as error:
        return f"the cute-dsl kernels are not importable: {error}"
    return None


class _PendingDown(NamedTuple):
    """A drafter MLP whose down projection runs at the all-reduce site that consumes it."""

    h2: torch.Tensor
    mlp: GatedMlp


def _draft_mlp(mlp: GatedMlp, x: torch.Tensor) -> _PendingDown:
    """The drafter MLP on ``x`` up to its down projection: the pinned split-K gate_up GEMM and
    ``silu_and_mul``; the down projection is left to the pushed all-reduce site."""
    h1 = draft_gemm(x, mlp.gate_up_proj.weight, "gate_up")
    if h1 is None:
        raise RuntimeError(f"K3 DSpark: no pinned gate_up GEMM for {x.shape[0]} rows")
    return _PendingDown(gated_activation(h1, "silu"), mlp)


class _DSparkHeadMixin:
    """The DSpark heads of the drafter: the vanilla Markov intra-block logit bias (``markov_w1``
    / ``markov_w2``, read by the worker's Markov chain), the ``shift_label`` slot convention and
    the confidence head (loaded, unused)."""

    def _init_dspark_heads(self, cfg) -> None:
        # Defaults on, unlike the DFlash base: the shift_label slot layout is
        # part of what DSpark *is*, and both published drafters set
        # block_size == max_draft_len, where the DFlash layout (slots 1..K)
        # runs one slot past the block and reads the next request's anchor.
        # An explicit false still selects the legacy layout.
        shift_label = resolve_dspark_head_config(cfg, "shift_label")
        self._dspark_shift_label = True if shift_label is None else bool(shift_label)
        self._dspark_markov_rank = int(resolve_dspark_head_config(cfg, "markov_rank") or 0)
        self._dspark_markov_head_type = str(
            resolve_dspark_head_config(cfg, "markov_head_type") or "vanilla"
        ).lower()
        self._dspark_use_confidence_head = bool(
            resolve_dspark_head_config(cfg, "use_confidence_head") or False
        )
        # Plain None placeholders rather than nn.Parameter/buffer: the shapes
        # ([vocab, rank]) are checkpoint-dependent, so nothing is pre-allocated
        # and nothing is constructed in the module's default dtype. Using the
        # checkpoint tensor as-is is also what keeps the head in the checkpoint's
        # dtype instead of an nn.Module default. load_weights() fills them in;
        # consumers treat None as "head absent".
        self.markov_w1 = None  # [vocab, rank] (nn.Embedding weight layout)
        self.markov_w2 = None  # [vocab, rank] (nn.Linear(rank->vocab) weight)
        self.confidence_proj_weight = None  # loaded, unused (follow-up MR)
        self.confidence_proj_bias = None

        if self._dspark_markov_rank > 0 and self._dspark_markov_head_type != "vanilla":
            raise ValueError(
                f"DSpark drafter declares markov_head_type="
                f"'{self._dspark_markov_head_type}'; only 'vanilla' is "
                "supported (gated/rnn heads need per-step hidden features)."
            )
        # The block decode only supports the non-causal DSpark convention.
        # Legacy DFlash drafter configs (e.g. Laguna) also carry a causal field
        # and handle it in the legacy decode path, which is why this check lives
        # here rather than in the DFlash base.
        if resolve_dspark_head_config(cfg, "causal"):
            raise ValueError(
                "DSpark drafter sets causal=true; the block decode only "
                "supports the non-causal DSpark convention."
            )
        if self._dspark_use_confidence_head:
            logger.warning(
                "DSpark drafter declares use_confidence_head; "
                "confidence-scheduled verification is not implemented yet "
                "(confidence_proj weights are loaded but unused, drafting "
                "always proposes the full K tokens)."
            )

    @property
    def has_markov_head(self) -> bool:
        return self._dspark_markov_rank > 0 and self.markov_w1 is not None

    def _take_dspark_head_weights(self, weights: Dict) -> tuple[Dict, Dict]:
        """Pull the head tensors out of ``weights`` and load them onto self.

        The head keys are pulled out before the backbone remap: left in, they
        would pick up a ``model.`` prefix and be dropped by partial loading.
        Returns the remaining weights and the extracted head weights.
        """
        dspark_weights = {}
        consumed = set()
        for canonical, aliases in _DSPARK_HEAD_WEIGHT_ALIASES.items():
            for name in aliases:
                if name in weights:
                    dspark_weights[canonical] = weights[name]
                    consumed.add(name)
                    break
        if consumed:
            weights = {k: v for k, v in weights.items() if k not in consumed}
        # The inverse of the missing-weights check below. Without it, a config
        # whose head switches this build cannot resolve loads the drafter with
        # the heads silently dropped -- correct output, lower acceptance.
        if self._dspark_markov_rank <= 0 and "markov_w1.weight" in dspark_weights:
            raise ValueError(
                "DSpark drafter ships markov_w1/markov_w2 but markov_rank resolved to 0. "
                "The checkpoint's head switches were not found in dspark_config, "
                "dflash_config, or at the top level; loading it would drop the Markov "
                "head silently."
            )
        if self._dspark_markov_rank > 0:
            vocab = self.config.vocab_size
            rank = self._dspark_markov_rank
            for k in ("markov_w1.weight", "markov_w2.weight"):
                if k not in dspark_weights:
                    raise ValueError(
                        f"DSpark drafter declares markov_rank="
                        f"{self._dspark_markov_rank} but the checkpoint is "
                        f"missing {k}."
                    )
                if tuple(dspark_weights[k].shape) != (vocab, rank):
                    raise ValueError(
                        f"DSpark {k} has shape "
                        f"{tuple(dspark_weights[k].shape)}, expected "
                        f"[vocab, markov_rank] = ({vocab}, {rank})."
                    )
            self.markov_w1 = dspark_weights["markov_w1.weight"].to("cuda")
            self.markov_w2 = dspark_weights["markov_w2.weight"].to("cuda")
        if "confidence_proj.weight" in dspark_weights:
            self.confidence_proj_weight = dspark_weights["confidence_proj.weight"].to("cuda")
        if "confidence_proj.bias" in dspark_weights:
            self.confidence_proj_bias = dspark_weights["confidence_proj.bias"].to("cuda")
        return weights, dspark_weights


class _K3DFlashDrafterMixin:
    """The capture projection of the K3 drafter: ``hidden_norm(fc(target hidden))``.

    ``fc`` and ``hidden_norm`` are built from the checkpoint by ``load_weights``, ``fc`` with its
    whole weight. Under plain TP (no attention DP) whose group divides the fc input width, every
    rank multiplies its slice of the captured features by its slice of the ``fc`` columns and one
    all-reduce sums the partials, with ``hidden_norm`` fused in over MNNVL.
    """

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project captured target hidden states into the draft hidden space."""
        hidden_states = hidden_states.to(self.fc.weight.dtype)
        if hidden_states.shape[0] > 0:
            projected = self._tp_sharded_project_target_hidden(hidden_states)
            if projected is not None:
                return projected
        norm = self.hidden_norm
        return F.rms_norm(
            bf16_linear(hidden_states, self.fc.weight),
            (norm.weight.shape[0],),
            norm.weight,
            norm.eps,
        )

    def _resolve_fc_shard(self):
        """``(weight slice, begin, end, all_reduce)`` of the TP-sharded fc, or None.

        Resolved once: the shard exists when the drafter runs plain TP (no attention DP) and the
        fc input width splits evenly over it. The TP group is o_proj's all-reduce group: under
        helix the drafter's ``model_config.mapping`` carries the CP split of its context KV
        instead.
        """
        if getattr(self, "_fc_shard", None) is None:
            self._fc_shard = False
            all_reduce = self.model.layers[0].self_attn.o_proj.all_reduce
            mapping = None if all_reduce is None else all_reduce.mapping
            in_features = self.fc.weight.shape[1]
            if (
                mapping is not None
                and mapping.tp_size > 1
                and not mapping.enable_attention_dp
                and in_features % mapping.tp_size == 0
            ):
                width = in_features // mapping.tp_size
                begin = mapping.tp_rank * width
                weight = self.fc.weight.data[:, begin : begin + width].contiguous()
                self._fc_shard = (weight, begin, begin + width, all_reduce)
        return self._fc_shard or None

    def _tp_sharded_project_target_hidden(self, hidden_states: torch.Tensor):
        """The TP-sharded fc (``_resolve_fc_shard``) and its all-reduce with ``hidden_norm``, or
        None when the fc is not sharded."""
        shard = self._resolve_fc_shard()
        if shard is None:
            return None
        weight, begin, end, all_reduce = shard
        partial = bf16_linear(hidden_states[:, begin:end], weight)
        norm = self.hidden_norm
        if all_reduce.mnnvl_allreduce is not None:
            return all_reduce(
                partial,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RMS_NORM, norm_weight=norm.weight, eps=norm.eps
                ),
            )
        return F.rms_norm(all_reduce(partial), (norm.weight.shape[0],), norm.weight, norm.eps)


def _latent_cache_helix_mapping(
    mapping: Mapping, helix_mapping: Optional[Mapping], inherit_cp_config: bool = False
) -> Optional[Mapping]:
    """The mapping that splits a one-latent-per-token cache by pages over a helix group.

    ``mapping`` is the drafter's: the target's helix group with its CP ranks repurposed
    as TP ranks. ``helix_mapping`` is the target's helix CP mapping, or None. The result
    keeps every head on every rank (tp = 1) and shards the sequence over cp =
    ``mapping.tp_size`` consecutive ranks, the target's CP groups. Returns None where
    that does not apply: no helix, attention DP, pipeline parallelism, a single rank, or
    a draft TP group that does not divide the target's CP (target tp > 1).

    ``inherit_cp_config=True`` gives the result ``helix_mapping``'s cp_config (its
    all-to-all transport, e.g. ``use_nccl_for_alltoall: false`` for the MNNVL exchange)
    with cp_type HELIX; otherwise its cp_config is ``{cp_type: HELIX}`` alone, the NCCL
    all-to-all.
    """
    if (
        helix_mapping is None
        or not helix_mapping.has_cp_helix()
        or mapping.enable_attention_dp
        or mapping.pp_size != 1
        or mapping.tp_size <= 1
        or helix_mapping.cp_size % mapping.tp_size != 0
    ):
        return None
    cp_config = {"cp_type": CpType.HELIX}
    if inherit_cp_config:
        cp_config = {**helix_mapping.cp_config, **cp_config}
    return Mapping(
        world_size=mapping.world_size,
        rank=mapping.rank,
        gpus_per_node=mapping.gpus_per_node,
        cp_size=mapping.tp_size,
        cp_config=cp_config,
        tp_size=1,
        pp_size=1,
        moe_tp_size=mapping.moe_tp_size,
        moe_ep_size=mapping.moe_ep_size,
        moe_cluster_size=mapping.moe_cluster_size,
        enable_lm_head_tp_in_adp=mapping.enable_lm_head_tp_in_adp,
    )


def _shard_context_kv_by_helix(draft_config, helix_mapping: Optional[Mapping]) -> Optional[Mapping]:
    """Split the latent cache by pages over the whole repurposed helix group.

    One latent per token cannot be split by head, so attention runs at tp = 1 with every head on
    every rank, and the CuTe DSL decode takes the rank-local bounds this needs. The sharding
    (:func:`_latent_cache_helix_mapping` of the target's helix CP mapping, with its cp_config:
    the MNNVL all-to-all of ``use_nccl_for_alltoall: false``) replaces ``draft_config.mapping``.
    Returns it, or None outside Helix.
    """
    mapping = _latent_cache_helix_mapping(
        draft_config.mapping, helix_mapping, inherit_cp_config=True
    )
    if mapping is None:
        return None
    frozen, draft_config._frozen = draft_config._frozen, False
    draft_config.mapping = mapping
    draft_config._frozen = frozen
    logger.info(f"MLA DSpark: latent cache pages sharded over helix cp={mapping.cp_size}")
    return mapping


class K3MLADSparkForCausalLM(_DSparkHeadMixin, _K3DFlashDrafterMixin, nn.Module):
    """The DSpark drafter on an MLA backbone (:class:`K3MLADSparkBackbone`).

    A plain ``nn.Module``: the engine and :class:`K3DSparkWorker` read its declarations by
    attribute (the cache contract ``_num_attn_layers`` / ``_num_kv_heads`` / ``_head_dim`` /
    ``_kv_factor`` / ``_helix_mapping``, ``reads_managed_ctx_pages``,
    ``reads_helix_sharded_ctx_pages``, ``draft_kv_dtype``, ``dflash_attention_backend``,
    ``shift_label``, ``block_size``, ``mask_token_id``) and call ``load_weights``,
    ``load_weights_from_target_model``, ``project_target_hidden``, ``precompute_context_kv`` and
    ``dflash_forward``.

    The context cache stores one ``kv_lora_rank + qk_rope_head_dim`` FP8 E4M3 latent per token per
    layer (unit scale, saturated) and no V, in the draft layers of the KV cache manager, its pages
    sharded over the Helix group. The block decode is one CuTe DSL FP8 pass over context and block
    per layer with per-token ``kv_bounds``.
    """

    # The block decode is this class's own: no worker attention backend applies.
    _uses_worker_attention_backend = False
    _default_attention_backend = "CUTEDSL"
    _supported_attention_backends = ("CUTEDSL",)
    # The context KV comes out of the manager's pool.
    _paged_ctx_cache = True
    # Positions the worker serves, published by K3DSparkWorker before the first forward; None
    # (direct construction) keeps the config-derived cap.
    _runtime_position_ceiling = None
    #: The dtype of the drafter's context KV pool, which the engine allocates the appended draft
    #: layers with: FP8 E4M3 latents at a unit per-tensor scale, as the context servers' drafter
    #: (the family ``dspark/drafter.py``) declares and writes them.
    draft_kv_dtype = torch.float8_e4m3fn

    @property
    def context_kv_fp8(self) -> bool:
        """Whether the context KV is held in FP8 E4M3 (the worker's name for ``draft_kv_dtype``)."""
        return self.draft_kv_dtype == torch.float8_e4m3fn

    def __init__(
        self,
        draft_config,
        *,
        dflash_attention_backend: str = "AUTO",
        helix_mapping: Optional[Mapping] = None,
    ):
        super().__init__()
        cfg = draft_config.pretrained_config
        if dflash_attention_backend == "AUTO":
            dflash_attention_backend = self._default_attention_backend
        if dflash_attention_backend not in self._supported_attention_backends:
            raise ValueError(
                f"{type(self).__name__} attention backend must be one of "
                f"{list(self._supported_attention_backends)}, got {dflash_attention_backend!r}."
            )
        reason = self._attention_backend_unavailability_reason(dflash_attention_backend)
        if reason is not None:
            raise ValueError(
                f"attention_backend={dflash_attention_backend!r} is unavailable for "
                f"{type(self).__name__}: {reason}."
            )
        self.dflash_attention_backend = dflash_attention_backend
        # Before the backbone is built: attention keeps the sharding's CP mapping.
        helix = _shard_context_kv_by_helix(draft_config, helix_mapping)
        cfg.architectures = [_CHECKPOINT_ARCH]

        # The drafter's own config: no speculative decoding, no gathered LM head output.
        model_config = replace(draft_config, spec_config=None, lm_head_gather_output=False)
        # ModelConfig.extra_attrs is init=False, so replace() does not keep it.
        model_config.extra_attrs = draft_config.extra_attrs
        self.draft_model_full = K3MLADSparkBackbone(model_config)
        self.model = self.draft_model_full.model
        # No head of its own: load_weights_from_target_model binds the target's.
        self.lm_head = None
        self.model_config = model_config
        self.config = model_config.pretrained_config
        # Set by install_k3_dspark (the target's logits processor) under parallel drafting.
        self.logits_processor = None

        dflash_config = getattr(cfg, "dflash_config", None) or {}
        self.mask_token_id = dflash_config.get(
            "mask_token_id", getattr(cfg, "mask_token_id", cfg.vocab_size)
        )
        self.target_layer_ids = dflash_config.get("target_layer_ids", None)
        self.block_size = dflash_config.get("block_size", getattr(cfg, "block_size", None))
        # The block width the engine sizes the shared draft KV with
        # (KvCacheCreator._unified_drafter_block_width): DFlash's slot convention. The
        # worker drafts in DSpark's (``_dspark_shift_label``, from the checkpoint).
        self.shift_label = False
        # Built from the checkpoint by load_weights: fc / hidden_norm, then the fused KV
        # projection (_build_fused_kv_buffers).
        self._fused_kv_weight = None
        self._fused_kv_bias = None
        # The TP-sharded fc, resolved on first use.
        self._fc_shard = None
        # The helix page sharding of the latent cache, or None.
        self._helix_mapping = helix
        self._init_dspark_heads(cfg)
        self._validate_mla_shape()

        attn0 = self.model.layers[0].self_attn
        self.qk_nope_head_dim = attn0.qk_nope_head_dim
        self.qk_rope_head_dim = attn0.qk_rope_head_dim
        self.kv_lora_rank = attn0.kv_lora_rank
        self.v_head_dim = attn0.v_head_dim
        # The cache contract the worker sizes its pool binding from: one MLA latent per token
        # per layer, no V half.
        self._num_attn_layers = len(self.model.layers)
        self._num_heads = attn0.num_heads_tp
        self._num_kv_heads = 1
        self._head_dim = attn0.kv_cache_head_dim
        self._kv_factor = 1

        rope = _resolve_dspark_mla_rope_params(cfg, draft_config)
        self._mla_rope_params = rope
        self._mla_workspace = None
        self._mla_freqs = None
        self._mla_freqs_cap = None
        self._cute_dsl_checked_shape = None
        # (softmax_scale, output_scale) device scalars of the FP8 decode.
        self._mla_fp8_scales = None
        # (fixup, plan) of the last _mla_split_kv_plan.
        self._mla_split_plan = None
        # HF DeepSeek folds YaRN's attention scaling into the softmax scale:
        # softmax_scale = mscale^2 / sqrt(qk_nope + qk_rope).
        mscale = _yarn_get_mscale(rope["scaling_factor"], rope["mscale_all_dim"])
        self.softmax_scale = (mscale * mscale) / math.sqrt(
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )

    # -- shape / buffers ---------------------------------------------------

    def _validate_mla_shape(self):
        """One uniform MLA shape over the layers: the latent projection is fused across them."""
        layers = getattr(self.model, "layers", None)
        if not layers:
            return
        attn0 = layers[0].self_attn
        keys = (
            "num_heads_tp",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
        )
        base = tuple(getattr(attn0, k) for k in keys)
        for idx, layer in enumerate(layers[1:], start=1):
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "kv_a_proj_with_mqa"):
                raise ValueError(
                    f"MLA DSpark block decode needs self_attn.kv_a_proj_with_mqa on every "
                    f"draft layer, but layer {idx} of {type(self.config).__name__} has none."
                )
            if tuple(getattr(attn, k) for k in keys) != base:
                raise ValueError(
                    "MLA DSpark fuses the latent projection across layers and needs one "
                    f"uniform MLA shape, but layer {idx} differs from layer 0 "
                    f"({dict(zip(keys, base))})."
                )

    @property
    def reads_helix_sharded_ctx_pages(self) -> bool:
        """The CuTe DSL decode reads rank-local pages and returns softmax stats;
        ``_shard_context_kv_by_helix`` shards the cache for it."""
        return self._helix_mapping is not None

    def reads_managed_ctx_pages(self, tokens_per_block: int) -> bool:
        """The context is paged (``_paged_ctx_cache``); the CuTe DSL decode takes a page size
        that divides 128, above 1."""
        return tokens_per_block > 1 and 128 % tokens_per_block == 0

    @classmethod
    def _attention_backend_unavailability_reason(cls, backend: str) -> Optional[str]:
        if backend == "CUTEDSL":
            return cute_dsl_mla_decode_unavailability_reason()
        return None

    def _mla_decode_workspace(self, device, min_bytes: int = 0):
        want = max(_MLA_DECODE_WORKSPACE_BYTES, int(min_bytes))
        if self._mla_workspace is None or self._mla_workspace.numel() < want:
            self._mla_workspace = torch.zeros(want, dtype=torch.uint8, device=device)
        return self._mla_workspace

    def _mla_fp8_decode_scales(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """The FP8 decode's device scales at unit query and context scales (served
        ``_mla_fp8_decode_scales``): ``bmm1 = softmax_scale * dq_q * dq_kv`` and ``bmm2 = dq_kv``,
        both dequant scales ``DRAFT_FP8_KV_SCALE``. Built on first use (the eager warmup, before
        any graph capture)."""
        if self._mla_fp8_scales is None or self._mla_fp8_scales[0].device != device:
            dq = DRAFT_FP8_KV_SCALE
            self._mla_fp8_scales = (
                torch.full((1,), self.softmax_scale * dq * dq, dtype=torch.float32, device=device),
                torch.full((1,), dq, dtype=torch.float32, device=device),
            )
        return self._mla_fp8_scales

    def _assert_cute_dsl_can_implement(self, batch: int, block_size: int, page_size: int) -> None:
        """Reject a shape the FP8 decode (FP8 in, BF16 out) cannot serve, naming it, instead of
        compiling an unvalidated tactic. Checked once per shape (host only)."""
        key = (batch, block_size, page_size)
        if self._cute_dsl_checked_shape == key:
            return
        import cutlass

        from ..kernels.k3r_mla_decode.mla_decode_fp8 import (
            BlackwellMultiHeadLatentAttentionForwardFP8,
        )

        tiler_qk, tiler_pv = _fp8_decode.MMA_QK_TILER_MN, _fp8_decode.MMA_PV_TILER_MN
        ok = BlackwellMultiHeadLatentAttentionForwardFP8.can_implement(
            batch,
            block_size,
            page_size,
            self._num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            cutlass.Float8E4M3FN,
            cutlass.BFloat16,
            cutlass.Float32,
            cutlass.Float32,
            tiler_qk,
            tiler_pv,
            1,
            False,
            _DECODE_IS_VAR_SEQ,
            _DECODE_IS_VAR_SPLIT_KV,
            page_size,
        )
        if not ok:
            raise ValueError(
                f"K3 DSpark: the CuTe DSL FP8 MLA decode cannot serve batch={batch}, "
                f"block_size={block_size}, page_size={page_size}, num_heads={self._num_heads}"
            )
        self._cute_dsl_checked_shape = key

    def _mla_split_kv_plan(
        self, fixup: _MLABlockFixup, block_size: int, stream: Optional[torch.cuda.Stream] = None
    ) -> torch.Tensor:
        """Per-request split-KV counts of the step's block decode (the decode's planner kernel
        at the FP8 piece cost, ``_MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8``), launched once per step
        on ``stream`` and reused by every layer; a new fixup (the next step, a graph capture)
        plans again. The output is allocated on the current stream; the caller joins ``stream``
        before a decode reads it."""
        cached = self._mla_split_plan
        if cached is not None and cached[0] is fixup:
            return cached[1]
        seq_lens = fixup.seq_lens_i32
        batch = seq_lens.shape[0]
        rows, wave_clusters = _fp8_decode.split_plan_geometry(self._num_heads, block_size)
        plan = torch.empty(batch, dtype=torch.int32, device=seq_lens.device)
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            launch_mla_split_kv_plan(
                seq_lens,
                plan,
                num_requests=batch,
                rows_per_request=rows,
                wave_clusters=wave_clusters,
                piece_cost=_MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8,
            )
        self._mla_split_plan = (fixup, plan)
        return plan

    def _cute_dsl_block_decode(
        self,
        kernel_query: torch.Tensor,
        layer_cache: torch.Tensor,
        fixup: _MLABlockFixup,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One CuTe DSL FP8 MLA decode over context AND block of one layer (kernel
        ``BlackwellMultiHeadLatentAttentionForwardFP8`` through ``k3_mla_decode_fp8``, the served
        runner's only tactic): every block token bounded by ``kv_bounds`` (the whole block,
        non-causal), each request split as the step's plan says. ``kernel_query`` is the FP8
        ``[B, block, heads, kv_lora_rank + rope]`` absorbed query (``draft_qhead``),
        ``layer_cache`` the layer's ``[pages, 1, 1, page, 576]`` FP8 pool. Returns the BF16
        latent output ``[B, block, heads, kv_lora_rank]`` of this rank's pages and its
        ``[B * block, heads, 2]`` (max, sum) softmax stats.
        """
        batch, _, num_heads, _ = kernel_query.shape
        device = kernel_query.device
        pool = layer_cache[:, 0, 0]  # [pages, page_size, head_dim]
        page_size = pool.shape[1]
        d_latent = self.kv_lora_rank
        self._assert_cute_dsl_can_implement(batch, block_size, page_size)
        # stride[1] == 1 is the kernel's only layout demand, kept by a 576-wide row sliced at 512.
        q_latent = kernel_query[..., :d_latent].permute(2, 3, 1, 0)
        q_rope = kernel_query[..., d_latent:].permute(2, 3, 1, 0)
        c_latent = pool.detach()[..., :d_latent].permute(1, 2, 0)
        c_rope = pool.detach()[..., d_latent:].permute(1, 2, 0)
        out = torch.empty(
            batch, block_size, num_heads, d_latent, dtype=torch.bfloat16, device=device
        )
        stats = torch.empty(batch * block_size, num_heads, 2, dtype=torch.float32, device=device)
        # Served passes the batch as the decode's max_batch_size (workspace layout and the largest
        # split a batch between two buckets keeps).
        workspace = self._mla_decode_workspace(
            device, _fp8_decode.workspace_bytes(num_heads, block_size, d_latent, batch)
        )
        softmax_scale, output_scale = self._mla_fp8_decode_scales(device)
        _fp8_decode.k3_mla_decode_fp8(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            # (max_blocks, B) with dim 0 stride-1: the transposed view, not a copy.
            fixup.page_tables_i32.t(),
            fixup.seq_lens_i32,
            out.permute(2, 3, 1, 0),
            workspace,
            num_heads,
            block_size,
            page_size,
            softmax_scale,
            output_scale,
            batch,
            stats,
            fixup.kv_bounds_i32,
            self._mla_split_kv_plan(fixup, block_size),
        )
        return out, stats

    def _mla_freqs_cis(self, device):
        """Adjacent-pair YaRN table, cached. See build_dspark_mla_yarn_freqs_cis.

        Sized from the ceiling the worker publishes once it knows the runtime one
        (_runtime_position_ceiling, set in K3DFlashWorker._lazy_init_ctx_buffers before
        any forward), because that is what ctx_len is clamped to and it exceeds
        model_config.max_seq_len whenever spec decoding is on. The config cap is the
        fallback for direct construction in tests.
        """
        rope = dict(self._mla_rope_params)
        runtime_cap = self._runtime_position_ceiling
        if runtime_cap is not None:
            # Outright, not max(): the published ceiling is already
            # min(runtime, max_position_embeddings) + lookahead, so it is
            # both sufficient and the only thing that keeps the table small
            # when max_seq_len is unset and the config cap is the
            # checkpoint's advertised 1,048,576.
            rope["max_positions"] = int(runtime_cap)
        # Keyed on the cap, not just built once: _lazy_init_ctx_buffers
        # re-publishes the ceiling when the KV-estimation probe manager is
        # swapped for the real one, and drafter forwards do run during
        # estimation. A table built short for a later, larger cap would index
        # out of range on freqs[positions].
        if self._mla_freqs is None or self._mla_freqs_cap != rope["max_positions"]:
            self._mla_freqs = build_dspark_mla_yarn_freqs_cis(
                dim=self.qk_rope_head_dim,
                base=rope["theta"],
                scaling_factor=rope["scaling_factor"],
                original_max_position_embeddings=rope["original_max_positions"],
                max_position_embeddings=rope["max_positions"],
                beta_fast=rope["beta_fast"],
                beta_slow=rope["beta_slow"],
                mscale=rope["mscale"],
                mscale_all_dim=rope["mscale_all_dim"],
                device=device,
            )
            self._mla_freqs_cap = rope["max_positions"]
        return self._mla_freqs

    def _mla_freqs_real(self, device) -> torch.Tensor:
        """``_mla_freqs_cis`` as a ``[positions, rope_dim // 2, 2]`` FP32 (cos, sin) view."""
        freqs = self._mla_freqs_cis(device)
        return torch.view_as_real(freqs).reshape(freqs.shape[0], freqs.shape[-1], 2)

    def _build_fused_kv_buffers(self) -> None:
        """Stack the per-layer latent projection and its RMSNorm weight.

        The MLA analogue of the base's fused K/V GEMM: one ``[L*576, hidden]``
        weight, plus the ``[L, kv_lora_rank]`` layernorm scales applied after.
        """
        if self._fused_kv_weight is not None:
            return
        layers_attn = [layer.self_attn for layer in self.model.layers]
        attn0 = layers_attn[0]
        self._fused_kv_weight = torch.cat(
            [a.kv_a_proj_with_mqa.weight for a in layers_attn], dim=0
        ).contiguous()
        self._fused_kv_bias = None
        self._kv_a_norm_stacked = torch.stack([a.kv_a_layernorm.weight.data for a in layers_attn])
        self._kv_a_norm_eps = attn0.kv_a_layernorm.variance_epsilon
        self._input_ln_eps = None
        self._has_qk_norm = False
        self._use_fused_qk_norm_rope = False

        # Split kv_b_proj into the absorption operands once. K absorbs into the
        # query, V un-absorbs the attention output. Block decode only.
        nh = attn0.num_heads_tp
        for a in layers_attn:
            w = a.kv_b_proj.weight.view(nh, self.qk_nope_head_dim + self.v_head_dim, -1)
            a._k_b_proj = w[:, : self.qk_nope_head_dim].contiguous()
            a._v_b_proj = w[:, self.qk_nope_head_dim :].contiguous()

        logger.debug(
            f"MLA DSpark: fused latent projection built for {self._num_attn_layers} layers "
            f"(weight={tuple(self._fused_kv_weight.shape)}, head_dim={self._head_dim})"
        )

    # -- context cache -----------------------------------------------------

    def precompute_context_kv_layer_major(self, projected_hidden, positions):
        """Post-norm / post-RoPE MLA latents of every drafter layer, ``[L, N, head_dim]``: the
        all-layer latent GEMM (pinned split-K, cuBLAS above the pinned rows) and one RMSNorm +
        RoPE launch of every layer, the RoPE table read by position in the kernel."""
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        n = projected_hidden.shape[0]
        weight_dtype = self._fused_kv_weight.dtype
        if projected_hidden.dtype != weight_dtype:
            projected_hidden = projected_hidden.to(weight_dtype)
        latent = draft_gemm(projected_hidden, self._fused_kv_weight, "ctx_latent")
        if latent is None:
            latent = F.linear(projected_hidden, self._fused_kv_weight)
        latent = latent.view(n, self._num_attn_layers, self._head_dim)
        return k3_dspark_mla_ctx_norm_rope(
            latent,
            self._kv_a_norm_stacked,
            positions.view(-1).long(),
            self._mla_freqs_real(latent.device),
            self._kv_a_norm_eps,
        )

    def precompute_context_kv(self, projected_hidden, positions):
        """``precompute_context_kv_layer_major`` as ``([N, L, 1, head_dim], None)`` (one latent,
        no V half), for the worker's context-request store. Served computes this layout with
        cuBLAS and one RMSNorm + RoPE launch per layer; the GEN step never stores a context
        request, so this target keeps the one path."""
        latent = self.precompute_context_kv_layer_major(projected_hidden, positions)
        return latent.transpose(0, 1).unsqueeze(2).contiguous(), None

    # -- block decode ------------------------------------------------------

    def dflash_forward(
        self,
        noise_embedding,
        query_positions,
        num_ctx_per_req,
        ctx_k_cache,
        ctx_v_cache,
        ctx_cache_batch_idx,
        ctx_kv_cache=None,
        ctx_page_table=None,
        block_placement=None,
        block_fixup: Optional[_MLABlockFixup] = None,
    ):
        """Absorbed-MLA block decode of every drafter layer over the FP8 paged latent cache.

        ``noise_embedding`` ``[B, block, hidden]``, ``query_positions`` ``[B, block]``,
        ``ctx_kv_cache`` each layer's ``[pages, 1, 1, page, 576]`` FP8 E4M3 pool and
        ``block_fixup`` the step's rank-local placement, lengths and ownership (the worker's fused
        prepare). The other arguments are the DFlash worker contract's and are not read. Returns
        the final normed hidden states ``[B * block, hidden]``.
        """
        del num_ctx_per_req, ctx_k_cache, ctx_v_cache, ctx_cache_batch_idx, ctx_page_table
        del block_placement
        helix = self._helix_mapping
        fixup = block_fixup
        if helix is None or fixup is None or fixup.owned is None or ctx_kv_cache is None:
            raise RuntimeError(
                "K3 DSpark: the block decode needs the Helix-sharded draft pool and the "
                "worker's rank-local block fixup"
            )
        if ctx_kv_cache[0].dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"K3 DSpark: the draft pool is {ctx_kv_cache[0].dtype}; this drafter's context "
                "is FP8 E4M3 (context_kv_fp8)"
            )
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()

        batch, block_size = noise_embedding.shape[0], noise_embedding.shape[1]
        device = noise_embedding.device
        num_heads = self._num_heads
        num_tokens = batch * block_size
        if block_size > ctx_kv_cache[0].shape[-2]:
            raise NotImplementedError("Helix-sharded MLA DSpark needs block_size <= page size")
        freqs_real = self._mla_freqs_real(device)
        flat_positions = query_positions.reshape(-1).long()
        # The tokens this rank holds no key of, layer-invariant.
        helix_zero_kv = fixup.kv_bounds_i32 == 0
        # The step's split-KV plan on the drafter's side stream, beside layer 0's projections and
        # behind the context store the worker put there (draft3_side_launch); joined before
        # layer 0's first pool write.
        plan_stream = self._draft3_side_stream()
        plan_stream.wait_stream(torch.cuda.current_stream())
        self._mla_split_kv_plan(fixup, block_size, stream=plan_stream)

        # o_proj and down_proj push their partials into the o_proj all-reduce's MNNVL workspace
        # and one site kernel finishes each reduction with the residual add and the following
        # RMSNorm (post_attention_layernorm, the next layer's input_layernorm, the final norm).
        sites = self._draft_ar_sites(self._draft_fused_all_reduce())
        if sites is None:
            raise RuntimeError("K3 DSpark: the drafter's o_proj all-reduce is not on MNNVL")
        hidden_states = noise_embedding
        residual = None
        partial = None
        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            if residual is None:
                # The noise embedding is this forward's own buffer, read only here; the sites
                # write new residual tensors.
                residual = hidden_states
                hs_normed = layer.input_layernorm(
                    hidden_states.reshape(-1, hidden_states.shape[-1])
                )
            else:
                res_flat = residual.reshape(-1, residual.shape[-1])
                hs_normed, res_flat = self._reduce_partial(
                    sites, partial, res_flat, layer.input_layernorm
                )
                residual = res_flat.reshape(batch, block_size, -1)

            # q_a and kv_a read the same normed input: one GEMM, both halves read as row-strided
            # views (the q_b GEMM and the query head kernel read strided rows).
            fused_a = self._draft_fused_a_proj(layer_idx)
            q_kv_a = None if fused_a is None else draft_gemm(hs_normed, fused_a, "q_kv_a")
            if q_kv_a is None:
                raise RuntimeError(
                    f"K3 DSpark: no pinned [q_a | kv_a] GEMM for {num_tokens} rows "
                    "(or the fused weight was first needed under graph capture)"
                )
            q_a = q_kv_a[:, : attn.q_lora_rank]
            blk_latent_rows = q_kv_a[:, attn.q_lora_rank :]
            # Folded query head: q_b of the un-normalized q_a with q_a_layernorm's weight folded
            # into W_qb, directly behind the [q_a | kv_a] GEMM; draft_qhead applies the inverse
            # RMS and does the rest.
            folded = self._draft3_folded_q_b(layer_idx)
            q = None if folded is None else draft_gemm(q_a, folded, "q_b", strided=True)
            if q is None:
                raise RuntimeError(
                    f"K3 DSpark: no pinned folded q_b GEMM for {num_tokens} rows "
                    "(or the folded weight was first needed under graph capture)"
                )
            if layer_idx == 0:
                self._draft3_join_side()
            # One launch: the block latents normalized, rotated and stored as FP8 into this rank's
            # pages (ownership mask), and the FP8 absorbed query of every head.
            kernel_query = draft_qhead(
                q_a,
                blk_latent_rows,
                q.view(num_tokens, num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim),
                attn.q_a_layernorm.variance_epsilon,
                attn._k_b_proj,
                attn.kv_a_layernorm.weight,
                attn.kv_a_layernorm.variance_epsilon,
                flat_positions,
                freqs_real,
                ctx_kv_cache[layer_idx][:, 0, 0],
                fixup.pages.reshape(-1),
                fixup.offsets.reshape(-1),
                rows_mask=fixup.owned,
            ).view(batch, block_size, num_heads, self.kv_lora_rank + self.qk_rope_head_dim)
            ctx_out, stats = self._cute_dsl_block_decode(
                kernel_query, ctx_kv_cache[layer_idx], fixup, block_size
            )
            # V up-projection, helix exchange and combine in one launch.
            attn_out = draft_helix_tail(helix, ctx_out, stats, attn._v_b_proj, helix_zero_kv)
            if attn_out is None:
                raise RuntimeError(
                    "K3 DSpark: the helix tail kernel does not serve this layer (receive buffer "
                    "missing before graph capture, or the shapes / transport do not fit)"
                )
            res_flat = residual.reshape(-1, residual.shape[-1])
            site = sites.linear_ar_norm(
                attn_out.reshape(num_tokens, -1),
                attn.o_proj.weight,
                "o_proj",
                res_flat,
                layer.post_attention_layernorm,
            )
            if site is None:
                raise RuntimeError(f"K3 DSpark: no pushed o_proj site for {num_tokens} rows")
            hidden_out, res_flat = site
            partial = _draft_mlp(layer.mlp, hidden_out)
            residual = res_flat.reshape(batch, block_size, -1)

        res_flat = residual.reshape(-1, residual.shape[-1])
        hidden_states_out, _ = self._reduce_partial(sites, partial, res_flat, self.model.norm)
        return hidden_states_out

    def _draft3_folded_q_b(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Layer ``layer_idx``'s q_b weight with q_a_layernorm's weight folded in
        (``fold_rmsnorm_weight``), built once outside graph capture; None until then."""
        cache = self.__dict__.setdefault("_draft3_qb_folded", {})
        folded = cache.get(layer_idx)
        if folded is None and not torch.cuda.is_current_stream_capturing():
            attn = self.model.layers[layer_idx].self_attn
            folded = cache[layer_idx] = fold_rmsnorm_weight(
                attn.q_b_proj.weight.detach(), attn.q_a_layernorm.weight.detach()
            )
        return folded

    def _draft3_side_stream(self) -> torch.cuda.Stream:
        stream = self.__dict__.get("_draft3_stream")
        if stream is None:
            stream = self.__dict__["_draft3_stream"] = torch.cuda.Stream()
        return stream

    @property
    def _draft3_side_keep(self) -> list:
        return self.__dict__.setdefault("_draft3_keep", [])

    def draft3_side_launch(self, launch: Callable[[], None], keep=()) -> None:
        """Run ``launch()`` on the drafter's side stream after the current stream's work
        (``draft3_glue``). The next ``dflash_forward`` joins the side stream before its first
        draft-pool write; ``keep`` (the tensors the launch reads) stays referenced until then, so
        the caching allocator does not hand their memory to the current stream while the side
        stream may still read it."""
        stream = self._draft3_side_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            launch()
        self._draft3_side_keep.append(tuple(keep))

    def _draft3_join_side(self) -> None:
        """The current stream waits for the side stream; the kept tensors are released."""
        torch.cuda.current_stream().wait_stream(self._draft3_side_stream())
        self._draft3_side_keep.clear()
        return None

    def _draft_fused_all_reduce(self) -> Optional[TpAllReduce]:
        """The all-reduce the block decode fuses with residual + RMSNorm, or None.

        For a TP-sharded drafter whose o_proj reduces over MNNVL (the fusion the
        MNNVL all-reduce implements).
        """
        all_reduce = self.model.layers[0].self_attn.o_proj.all_reduce
        if all_reduce is None or all_reduce.mnnvl_allreduce is None:
            return None
        return all_reduce

    def _draft_ar_sites(self, fused_ar) -> Optional[DraftArSites]:
        """The pushed all-reduce sites on the fused all-reduce, or None.

        The sites push into the Lamport buffers of the all-reduce's MNNVL
        workspace (``layers.lamport_push_target``); a call a site does not
        take runs the fused all-reduce.
        """
        if fused_ar is None:
            return None
        sites = self.__dict__.get("_ar_sites")
        if sites is None or sites.all_reduce is not fused_ar:
            sites = self.__dict__["_ar_sites"] = DraftArSites(fused_ar)
        return sites

    @staticmethod
    def _reduce_partial(sites: DraftArSites, partial: _PendingDown, residual, norm):
        """``norm(all_reduce(h2 @ down^T) + residual)`` and the new residual of a pending down
        projection, as one pushed split-K GEMM and one site kernel."""
        site = sites.linear_ar_norm(
            partial.h2, partial.mlp.down_proj.weight, "down", residual, norm
        )
        if site is None:
            raise RuntimeError(f"K3 DSpark: no pushed down site for {partial.h2.shape[0]} rows")
        return site

    def _draft_fused_a_proj(self, layer_idx: int) -> Optional[torch.Tensor]:
        """The ``[q_a_proj; kv_a_proj_with_mqa]`` weight of a layer, for one GEMM.

        Both replicated projections read the same normed input, so one GEMM
        replaces two. The separate projections' weights become row views of the
        fused weight, so no memory is added. Built once, outside graph capture
        (None if the first call is under capture).
        """
        cache = self.__dict__.setdefault("_fused_a_weights", {})
        if layer_idx in cache:
            return cache[layer_idx]
        attn = self.model.layers[layer_idx].self_attn
        q_a, kv_a = attn.q_a_proj, attn.kv_a_proj_with_mqa
        fused = None
        if q_a.weight.dtype == kv_a.weight.dtype and not torch.cuda.is_current_stream_capturing():
            fused = torch.cat([q_a.weight.data, kv_a.weight.data], dim=0).contiguous()
            q_a.weight.data = fused[: q_a.out_features]
            kv_a.weight.data = fused[q_a.out_features :]
        cache[layer_idx] = fused
        return fused

    # -- weights -----------------------------------------------------------

    #: Tensors the wrapper itself owns, built from the checkpoint in load_weights, and required.
    WRAPPER_OWNED_WEIGHTS = ("fc.weight", "hidden_norm.weight")
    #: Only the head is shared. This drafter ships its own embed_tokens whose
    #: values differ from the target's -- see load_weights_from_target_model --
    #: so a checkpoint missing it must fail rather than run on random weights.
    WEIGHTS_SHARED_WITH_TARGET = ("lm_head",)

    #: The checkpoint's TorchSpec spellings of the capture projection and the final norm
    #: (the SpecForge drafters name them ``fc`` / ``hidden_norm`` / ``norm``).
    _TORCHSPEC_RENAMES = {
        "context_proj.weight": "fc.weight",
        "context_norm.weight": "hidden_norm.weight",
        "final_norm.weight": "norm.weight",
    }

    def _backbone_tensors(self) -> Iterator[tuple[str, Callable[..., None], tuple[str, ...]]]:
        """``(target, load, checkpoint names)`` of every backbone tensor, ``load`` taking the
        named full checkpoint tensors in order."""

        def copy_into(param):
            return lambda tensor: param.data.copy_(tensor)

        backbone = self.model
        yield "embed_tokens", backbone.embed_tokens.load, ("embed_tokens.weight",)
        yield "norm", copy_into(backbone.norm.weight), ("norm.weight",)
        for idx, layer in enumerate(backbone.layers):
            prefix = f"layers.{idx}."
            attn = layer.self_attn
            for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                yield (
                    f"{prefix}self_attn.{name}",
                    getattr(attn, name).load,
                    (f"{prefix}self_attn.{name}.weight",),
                )
            for name in ("q_a_layernorm", "kv_a_layernorm"):
                yield (
                    f"{prefix}self_attn.{name}",
                    copy_into(getattr(attn, name).weight),
                    (f"{prefix}self_attn.{name}.weight",),
                )
            yield (
                f"{prefix}mlp",
                layer.mlp.load,
                tuple(f"{prefix}mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")),
            )
            for name in ("input_layernorm", "post_attention_layernorm"):
                yield (
                    f"{prefix}{name}",
                    copy_into(getattr(layer, name).weight),
                    (f"{prefix}{name}.weight",),
                )

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load the drafter checkpoint into the K3-owned weights, each tensor by its name.

        The DSpark heads go to :meth:`_take_dspark_head_weights`; ``fc`` / ``hidden_norm`` (or
        their TorchSpec names ``context_proj`` / ``context_norm``) build the capture projection;
        every backbone tensor (HF MLA names, with or without the ``model.`` prefix; the final
        norm also as ``final_norm``) is copied into this rank's shard of its weight. A checkpoint
        missing any of them raises. ``weight_mapper`` is not used.
        """
        del weight_mapper, kwargs
        weights, _ = self._take_dspark_head_weights(weights)
        weights = {self._TORCHSPEC_RENAMES.get(k, k): v for k, v in weights.items()}
        weights = {
            k[len("model.") :] if k.startswith("model.") else k: v for k, v in weights.items()
        }

        # Without fc the drafter has no capture projection and drafts from an empty
        # context forever.
        wrapper_missing = [k for k in self.WRAPPER_OWNED_WEIGHTS if k not in weights]
        if wrapper_missing:
            raise ValueError(
                f"{type(self).__name__}: checkpoint is missing {wrapper_missing}, "
                "which this wrapper owns and builds from the checkpoint."
            )
        tensors = list(self._backbone_tensors())
        missing = sorted(
            target for target, _, names in tensors if any(n not in weights for n in names)
        )
        if missing:
            raise ValueError(
                f"{type(self).__name__}: checkpoint provides no weights for "
                f"{missing[:8]}{' ...' if len(missing) > 8 else ''}. These are "
                f"not in WEIGHTS_SHARED_WITH_TARGET "
                f"({', '.join(self.WEIGHTS_SHARED_WITH_TARGET)}), so loading "
                "would leave them uninitialized."
            )

        fc = weights["fc.weight"]
        self.fc = Bf16Linear(fc.shape[1], fc.shape[0], dtype=fc.dtype, device="cuda")
        self.fc.load(fc)
        norm = weights["hidden_norm.weight"]
        self.hidden_norm = RMSNormWeight(
            norm.shape[0], getattr(self.config, "rms_norm_eps", 1e-6), norm.dtype, device="cuda"
        )
        self.hidden_norm.weight.data.copy_(norm)
        used = set(self.WRAPPER_OWNED_WEIGHTS)
        for _, load, names in tensors:
            load(*(weights[n] for n in names))
            used.update(names)
        unused = sorted(set(weights) - used)
        if unused:
            logger.warning(
                f"{type(self).__name__}: checkpoint tensors not loaded: "
                f"{unused[:8]}{' ...' if len(unused) > 8 else ''}"
            )

    def load_weights_from_target_model(self, target_model: torch.nn.Module) -> None:
        """Share only ``lm_head``; the drafter has its own trained embedding.

        Inferact/Kimi-K3-DSpark ships ``embed_tokens`` whose values differ from
        the target's, so adopting the target's would feed the block decode
        embeddings it was not distilled against -- silently, at some acceptance
        cost.
        """
        self.draft_model_full.lm_head = target_model.lm_head
        self.lm_head = target_model.lm_head


def build_k3_dspark_draft(model_config, draft_config, lm_head, model) -> nn.Module:
    """Build the Kimi-K3 standalone MLA DSpark drafter.

    The K3 shell routes only the standalone MLA drafter here; every other
    DSpark drafter (embedded DeepSeek-V4, standalone GQA) stays with main's
    registry builder.

    Args:
        model_config: the target engine's ``ModelConfig``.
        draft_config: the drafter's own ``ModelConfig``.
        lm_head: unused; the drafter shares the target's head at weight-load time.
        model: unused; kept for the draft-builder signature.

    Returns:
        The :class:`K3MLADSparkForCausalLM` drafter.

    Raises:
        ValueError: the draft is embedded in the target checkpoint, or its
            backbone is not MLA-shaped.
    """
    del lm_head, model
    spec_config = model_config.spec_config
    if getattr(spec_config, "draft_is_embedded_in_target", False):
        raise ValueError(
            "build_k3_dspark_draft builds the standalone MLA DSpark drafter only; this draft "
            "is embedded in the target checkpoint (main's DSpark builder serves it)."
        )
    if not is_mla(draft_config.pretrained_config):
        raise ValueError(
            "build_k3_dspark_draft builds the standalone MLA DSpark drafter only; this "
            "drafter's backbone is not MLA-shaped (main's DSpark builder serves it)."
        )
    return K3MLADSparkForCausalLM(
        draft_config,
        dflash_attention_backend=spec_config.attention_backend,
        # The target's helix CP original, set while its modules are built on the repurposed
        # mapping (KimiK3Helix16UllForCausalLM._setup_helix_mappings).
        helix_mapping=getattr(model_config, "_helix_mapping_with_cp", None),
    )


def uses_k3_dspark(spec_config) -> bool:
    """Whether ``spec_config`` is one-engine DSpark with a separately shipped drafter.

    Whether that drafter is MLA-shaped, and so served by the private classes, is only
    known from its loaded config (:func:`install_k3_dspark`).
    """
    return (
        spec_config is not None
        and spec_config.spec_dec_mode.use_one_engine()
        and spec_config.spec_dec_mode.is_dspark()
        and not bool(getattr(spec_config, "draft_is_embedded_in_target", False))
    )


def _replace_in_epilogue(target: nn.Module, old: Optional[nn.Module], new: nn.Module) -> None:
    """Put ``new`` where main's shell appended ``old`` (by identity), else append it."""
    for i, module in enumerate(target.epilogue):
        if module is old:
            target.epilogue[i] = new
            return
    target.epilogue.append(new)


def install_k3_dspark(target: nn.Module, model_config: ModelConfig) -> bool:
    """Replace main's DSpark drafter and worker on ``target`` with the private ones.

    Args:
        target: a ``SpecDecOneEngineForCausalLM`` whose ``__init__`` just ran (it reads
            ``draft_config``, ``draft_model``, ``spec_worker``, ``use_separate_draft_kv_cache``,
            ``epilogue``, ``lm_head``, ``logits_processor`` and ``model``).
        model_config: the target's ``ModelConfig``, with the mapping main's drafter was
            built under still in place.

    Returns:
        True when the private drafter and worker were installed; False when the
        configuration keeps main's (no DSpark, embedded draft, GQA drafter).
    """
    spec_config = getattr(model_config, "spec_config", None)
    if not uses_k3_dspark(spec_config):
        return False
    # Main's shell built the drafter's own ModelConfig for the external drafter
    # (external_drafter_config_kwargs: mapping_for_full_rows(), the draft KV quant
    # algo, the target's extra_attrs); the private drafter is built on the same one,
    # which the KV cache creator also reads.
    draft_config = target.draft_config
    if draft_config is None or not is_mla(draft_config.pretrained_config):
        logger.info("K3 DSpark: not an MLA drafter; main's DSpark drafter and worker serve it")
        return False

    # Imported here: only an MLA DSpark deployment pays for the private drafter and
    # worker modules (and the kernel modules they import) at import time.

    draft_model = build_k3_dspark_draft(model_config, draft_config, target.lm_head, target.model)
    if spec_config.spec_dec_mode.is_parallel_draft():
        draft_model.logits_processor = target.logits_processor
    # The drafter runs on every row: the mapping main hands get_spec_worker.
    spec_worker = K3DSparkWorker(
        spec_config,
        model_config.mapping.mapping_for_full_rows(),
        target.use_separate_draft_kv_cache,
    )
    # Only this processor makes the target logits ``lm_head(hidden_states)``, which the
    # worker's vocab-parallel selection computes per shard.
    spec_worker.plain_logits_processor_type = type(target.logits_processor)

    _replace_in_epilogue(target, target.draft_model, draft_model)
    _replace_in_epilogue(target, target.spec_worker, spec_worker)
    target.draft_model = draft_model
    target.spec_worker = spec_worker
    # As SpecDecOneEngineForCausalLM.__init__: cache the drafter's static vocab map
    # and bind the worker to both models (vocab-parallel selection binds the heads
    # in setup_aliases / post_load_weights, after loading).
    spec_worker.set_draft_model(draft_model)
    spec_worker.set_target_model(target)
    logger.info(
        f"K3 DSpark: private drafter {type(draft_model).__module__}.{type(draft_model).__name__}"
        f" and worker {type(spec_worker).__module__}.{type(spec_worker).__name__}"
    )
    return True


__all__ = [
    "K3MLADSparkBackbone",
    "K3MLADSparkForCausalLM",
    "K3MLADSparkModel",
    "build_dspark_mla_yarn_freqs_cis",
    "build_dspark_mla_yarn_rope",
    "build_k3_dspark_draft",
    "install_k3_dspark",
    "uses_k3_dspark",
]


if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

_F = _features()

#: Layer kinds (``chain.OURS`` / ``CHAINED`` / ``MOE``): layer 0 (off the chain but for its KDA
#: core), a KDA + MoE layer the chain runs whole, an MLA + MoE layer whose MoE the chain runs.
OURS, CHAINED, MOE = 0, 1, 2

#: Ranks of the deployment (attention head-TP x Helix CP, repurposed as TP).
WORLD = 16

#: The DSpark hidden-state tap captures the pre-norm attention-residual mixture its next consumer
#: reads (served ``KIMI_K3_AUX_ATTN_RES_STREAM`` unset), not the running prefix sum: the convention
#: the drafter was distilled against.
TAP_MIX = True

#: The chain entry's pre-attention mix releases the chain's in-projection at kernel entry (served
#: ``_EntryMix``: ``(ull_tail_sandwich or ull2_coll_cp_tail) and releases_at_entry``).
ENTRY_RELEASE = _F.ull_tail_sandwich or _F.ull2_coll_cp_tail

#: Rows up to which an MLA layer with the output gate and the FP4 KV takes its input as the MXFP8
#: rows the chain's run-exit consumer stored (served ``ull/mlain.py`` ``MAX_ROWS``).
MLAIN_MXFP8_MAX_ROWS = 16

#: The media placeholder token id of the Kimi K3 vision-language checkpoint.
MEDIA_PLACEHOLDER_TOKEN_ID = 163605


def layer_kinds(cfg) -> Tuple[int, ...]:
    """The kind of every layer (:data:`OURS` / :data:`CHAINED` / :data:`MOE`): layer 0 (KDA +
    dense MLP) is ours, the other KDA layers are chained, the MLA layers' MoE is on the chain."""
    kinds = []
    for i in range(cfg.num_hidden_layers):
        kda, mla = _is_kda_layer(cfg, i), _is_mla_layer(cfg, i)
        if kda == mla:
            raise ValueError(f"Kimi K3 layer {i} must be exactly one of KDA / MLA")
        dense = i < cfg.first_k_dense_replace
        if dense != (i == 0) or (i == 0 and not kda):
            raise ValueError(
                "Kimi K3 ULL: layer 0 must be the one KDA + dense layer "
                f"(layer {i}: KDA {kda}, dense {dense})"
            )
        kinds.append(OURS if i == 0 else CHAINED if kda else MOE)
    if len(kinds) < 2 or kinds[1] != CHAINED or kinds[-1] != MOE:
        raise ValueError("Kimi K3 ULL: layer 1 must be a KDA layer and the last an MLA layer")
    return tuple(kinds)


class _SideCopies:
    """Copies nothing reads before the end of the forward (the DSpark taps) on a side stream. Each
    is forked after its source; the sources stay referenced until :meth:`join` makes the current
    stream wait for every pending copy, so the caching allocator cannot hand their memory to a
    later kernel while a copy may still read it."""

    def __init__(self, stream: torch.cuda.Stream):
        self.stream = stream
        self._pending: List[Tuple[torch.Tensor, torch.cuda.Event]] = []

    @staticmethod
    def active() -> bool:
        """Side copies only where the model forks side streams (CUDA graph capture)."""
        return do_multi_stream()

    def run(self, fn: Callable[[], None], source: torch.Tensor) -> None:
        """Run ``fn`` (a copy reading ``source``) on the side stream."""
        fork = torch.cuda.Event()
        fork.record()
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(fork)
            fn()
            done.record()
        self._pending.append((source, done))

    def join(self) -> None:
        current = torch.cuda.current_stream()
        for _, done in self._pending:
            current.wait_event(done)
        self._pending.clear()


# ---------------------------------------------------------------------------
# Decoder layers: the modules of the parts this file runs.
# ---------------------------------------------------------------------------


class KimiK3Helix16UllDecoderLayer(nn.Module):
    """One decoder layer's modules outside the chain.

    * layer 0 (:data:`OURS`): ``linear_attn`` (the KDA o_proj and its all-reduce,
      ``layer0.KdaOutputProjection``) and ``mlp`` (``layer0.Dense0``);
    * an MLA layer (:data:`MOE`): ``self_attn`` (``mla.K3MLARuntime``);
    * chained KDA layers: nothing (the chain holds the whole layer).

    ``attn_res``: the layer's attention-residual and norm vectors (``_chain_weights``'
    ``AttnResWeights``: ``self_attention_res_{norm,proj}``, ``input_layernorm``,
    ``mlp_res_{norm,proj}``, ``post_attention_layernorm``; ``.weight`` / ``.eps``), the chain's
    copy, which the loader binds; the mixes this file runs (layer 0's, the chain entry's, the MLA
    layers' post-attention mix) read them there."""

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        kind: int,
        entry: bool,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.kind = kind
        self.entry = entry
        self.is_kda = kind != MOE
        self.attn_res = None
        # An MLA layer: whether it takes the chain's MXFP8 input rows (set by ``build_chain``).
        self.mlain = False
        dtype = torch.bfloat16
        h = cfg.hidden_size
        if kind == OURS:
            mapping = model_config.mapping.mapping_for_full_rows()
            linear = cfg.linear_attn_config
            heads, head_dim = linear["num_heads"], linear["head_dim"]
            if heads % mapping.tp_size:
                raise ValueError(f"{heads} KDA heads do not split over {mapping.tp_size} ranks")
            self.linear_attn = KdaOutputProjection(
                h,
                heads // mapping.tp_size * head_dim,
                mapping,
                model_config.allreduce_strategy,
            )
            self.mlp = Dense0(
                cfg.intermediate_size,
                rank=mapping.tp_rank,
                world=mapping.tp_size,
                gpus_per_node=mapping.gpus_per_node,
                beta=getattr(cfg, "activation_situ_beta", None) or 1.0,
                linear_beta=getattr(cfg, "activation_situ_linear_beta", None),
                dtype=dtype,
            )
        elif kind == MOE:
            self.self_attn = K3MLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
                # The CP original stashed by _setup_helix_mappings.
                mapping_with_cp=getattr(model_config, "_helix_mapping_with_cp", None),
            )


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class KimiK3Helix16UllModelingV2(DecoderModel):
    """The text backbone: embedding, the layers' modules, and the generation step
    (:meth:`forward`) over them and the chain."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16
        self.kinds = layer_kinds(cfg)
        self.attn_res_block_size = cfg.attn_res_block_size
        self.num_attn_res_snapshots = (
            cfg.num_hidden_layers + cfg.attn_res_block_size - 1
        ) // cfg.attn_res_block_size

        # The MLA layers' side work (KV append, gate, zero-KV mask) forks to this stream.
        self.aux_stream_dict = {AuxStreamType.Attention: torch.cuda.Stream()}
        # The whole vocabulary on every rank (``F.embedding``; ``embedding_gather``: the fused
        # head's 16-byte row gather).
        self.embed_tokens = EmbeddingTable(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                KimiK3Helix16UllDecoderLayer(
                    model_config,
                    cfg,
                    i,
                    kind,
                    entry=kind == CHAINED and self.kinds[i - 1] == OURS,
                    aux_stream_dict=self.aux_stream_dict,
                )
                for i, kind in enumerate(self.kinds)
            ]
        )
        self.mla_layers = [layer for layer in self.layers if layer.kind == MOE]
        # The DSpark taps' side stream; the step metadata of the FP4 MLA decode, prelaunched on
        # its own stream at the step start.
        self._side_copies = _SideCopies(torch.cuda.Stream())
        self._mla_plan_stream = torch.cuda.Stream()
        self._mla_plan_events = (torch.cuda.Event(), torch.cuda.Event())

        mapping = model_config.mapping.mapping_for_full_rows()
        if mapping.tp_size != WORLD or mapping.enable_attention_dp:
            raise ValueError(
                f"Kimi K3 ULL runs on {WORLD} head-TP ranks (Helix CP repurposed as TP); got TP "
                f"{mapping.tp_size}, attention DP {mapping.enable_attention_dp}"
            )
        self.rank = mapping.tp_rank
        # The process group of the chain's symmetric-memory mailboxes (all ranks construct).
        self.group = native_process_group(self.rank, WORLD)
        # Set by the loader (``_chain_weights.load_chain_weights``); the chain takes them over.
        self.chain_weights = None
        self.chain = None

    # ------------------------------------------------------------------ the chain
    def build_chain(self, kv_cache_manager, kda_state=None) -> None:
        """Build the chain (all ranks together, outside CUDA graph capture): its kernels for every
        served step shape, its collective mailboxes over :attr:`group` and the KDA checkpoint
        views of ``kv_cache_manager`` (``cache.K3UllKdaState``; or ``kda_state``, any
        ``chain.KdaCheckpointState``); and every MLA layer's folded input projection."""
        if self.chain is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Kimi K3 ULL: the decode chain is built on the first (eager) generation step, "
                "before CUDA graph capture"
            )
        if self.chain_weights is None:
            raise RuntimeError("Kimi K3 ULL: the chain's weights are not loaded")
        from ..ops.cache.k3_ull_cache_manager import K3UllKdaState

        rubin = torch.cuda.get_device_capability() == (10, 7)
        self.chain = K3UllChain(
            tuple(enumerate(self.kinds)),
            self.group,
            weights=self.chain_weights,
            kda_state=K3UllKdaState(kv_cache_manager) if kda_state is None else kda_state,
            # Elsewhere than Rubin the NVFP4 routed experts' Rubin engines run forced.
            nv_force=not rubin,
            solo=(0,),
            tap_mix=TAP_MIX,
        )
        self.chain_weights = None
        for layer in self.mla_layers:
            mixer = layer.self_attn.mixer
            # Served ``ull/mlain.py`` ``_serves``: an FP4-KV MLA layer with the output gate.
            layer.mlain = bool(
                mixer.use_output_gate and getattr(mixer.mqa, "uses_fp4_mla_attention", False)
            )
            mixer.prepare_in_fold()
        logger.info(f"Kimi K3 ULL chain: {self.chain.describe()}")

    def _prelaunch_mla_step_metadata(self, attn_metadata: AttentionMetadata) -> None:
        """Fork the step's FP4 MLA metadata kernels (plan, page table) to a side stream; the first
        FP4 MLA layer joins it (``step_metadata_ready``)."""
        if not do_multi_stream():
            return
        mixer = None
        for layer in self.mla_layers:
            candidate = layer.self_attn.mixer
            if getattr(candidate.mqa, "uses_fp4_mla_attention", False):
                mixer = candidate
                break
        if mixer is None:
            return
        fork, done = self._mla_plan_events
        fork.record()
        with torch.cuda.stream(self._mla_plan_stream):
            fork.wait()
            mixer.prelaunch_step_metadata(attn_metadata)
            done.record()
        mixer.step_metadata_ready = done

    def _tap(self, spec_metadata, capture_set, layer_idx: int) -> bool:
        return spec_metadata is not None and (capture_set is None or layer_idx in capture_set)

    def _capture(self, spec_metadata, layer_idx: int, value: torch.Tensor) -> None:
        """DSpark tap of layer ``layer_idx`` (on the side stream under multi-stream, joined at the
        end of the forward)."""
        side = self._side_copies
        if not side.active():
            spec_metadata.maybe_capture_hidden_states(layer_idx, value, None)
            return
        side.run(lambda: spec_metadata.maybe_capture_hidden_states(layer_idx, value, None), value)

    # ------------------------------------------------------------------ the step
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        """One generation step: the output rows (output attention residual + final norm) of the
        ``rows`` token rows."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            weight = self.embed_tokens.weight
            if _F.embedding_gather and input_ids.dim() == 1 and weight.is_contiguous():
                inputs_embeds = gather_embedding(input_ids, weight)
            else:
                inputs_embeds = F.embedding(input_ids, weight)
        embeds = inputs_embeds
        rows = embeds.shape[0]
        plan_step(attn_metadata, rows)
        self.build_chain(attn_metadata.kv_cache_manager)
        chain = self.chain
        plan_step(attn_metadata, rows, chain)
        md = attn_metadata
        capture_set = (
            getattr(spec_metadata, "_capture_layer_set", None)
            if spec_metadata is not None
            else None
        )

        # The FP4 MLA decode plan of the previous step is stale.
        fp4_mla_decode.begin_step(md)
        self._prelaunch_mla_step_metadata(md)
        # The attention-residual bank [K, M, H]; the chain reads it as [M, K, H].
        bank = embeds.new_empty(self.num_attn_res_snapshots, rows, embeds.shape[1])
        blocks = chain.blocks_view(bank)

        # Layer 0: KDA (core on the chain) + dense MLP. A block boundary: the embedding is the
        # first bank row and the attention output starts the prefix.
        # (served ``PreMix`` with no snapshot yet: the input norm, then the embedding as the
        # first bank row)
        l0 = self.layers[0]
        a0 = l0.attn_res
        attn_in = rms_norm(embeds, a0.input_layernorm.weight, a0.input_layernorm.eps)
        bank[0].copy_(embeds)
        num_snapshots = 1
        # (served ``ChainKda``)
        slots = md.mamba_metadata.generation_state_indices
        chain.bind_slots(md.mamba_metadata.state_indices if slots is None else slots, rows)
        core = mr8_solo_kda(chain, 0, attn_in)
        # (served ``Oproj``: the KDA mixer's ``_project_output(core, allow_ar_push=True)``)
        attn_out = l0.linear_attn(core)
        # (served ``ull/dense0.py`` ``Dense0._issue``)
        hidden = l0.mlp(attn_out, bank[:num_snapshots], a0, l0.linear_attn._o_allreduce)

        # Layer 1's pre-attention mix (served ``_EntryMix``): finishes the dense MLP's all-reduce,
        # mixes and norms, and releases the chain's in-projection at kernel entry.
        x, prefix, num_snapshots = self._entry_mix(
            self.layers[1],
            hidden,
            bank,
            num_snapshots,
            spec_metadata if self._tap(spec_metadata, capture_set, 0) else None,
        )

        for layer in self.layers[1:]:
            i = layer.layer_idx
            tap = self._tap(spec_metadata, capture_set, i)
            aux = torch.empty_like(x) if tap else None
            if layer.kind == CHAINED:
                # (served ``ChainLayer``)
                x = mr8_chained_layer(chain, i, x, prefix, blocks, aux, i in chain.last)
            else:
                moe_in, prefix = self._mla_layer(layer, x, prefix, bank[:num_snapshots], md)
                # (served ``ChainMoe``)
                nxt = chain.next_of[i]
                x = mr8_chain_moe(
                    chain, i, moe_in, prefix, blocks, aux, nxt is not None and nxt in chain.sites
                )
            if tap:
                self._capture(spec_metadata, i, aux)
            num_snapshots = chain.num_snapshots_after(i)
        # (served ``ChainMoe`` of the last layer, then ``FinalMix``: the chain's last consumer
        # stored the output rows; the DSpark taps land before the drafter reads them)
        chain.join()
        self._side_copies.join()
        return x

    def _entry_mix(
        self,
        layer: KimiK3Helix16UllDecoderLayer,
        hidden,
        bank: torch.Tensor,
        num_snapshots: int,
        spec_metadata,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """The chain entry layer's pre-attention mix of layer 0's ``_DeferredRpTail`` (served
        ``ull/stages.py`` ``PreMix.run`` as ``_EntryMix``): ``(normed input, prefix, bank rows)``.
        With ``spec_metadata`` (layer 0 tapped) unfused: the all-reduce finished, the mix, the
        tap, the norm."""
        a = layer.attn_res
        proj, norm, out_norm = (
            a.self_attention_res_proj,
            a.self_attention_res_norm,
            a.input_layernorm,
        )
        valid = bank[:num_snapshots]
        boundary = layer.layer_idx % self.attn_res_block_size == 0
        if spec_metadata is None:
            # The prefix goes straight into the bank on a block-boundary layer
            # (served ``_snapshot_slot``).
            slot = bank[num_snapshots] if boundary and num_snapshots < bank.shape[0] else None
            fused = _k3_attn_res_pdl_rp_tail(
                hidden,
                valid,
                proj,
                norm,
                out_norm,
                mxfp8=False,
                entry_trigger=ENTRY_RELEASE,
                prefix_out=slot,
            )
            if fused is None:
                raise RuntimeError(
                    f"Kimi K3 ULL layer {layer.layer_idx}: the chain entry's mix of "
                    f"{tuple(hidden.residual.shape)} is outside the k3_attn_res kernel"
                )
            prefix, normed = fused[0], fused[1]
        else:
            prefix = hidden.materialize()
            mixed = _apply_attn_res(prefix, valid, proj, norm) if TAP_MIX else prefix
            self._capture(spec_metadata, 0, mixed)
            normed = rms_norm(mixed, out_norm.weight, out_norm.eps)
        if boundary:
            snapshot = bank[num_snapshots]
            if prefix.data_ptr() != snapshot.data_ptr():
                snapshot.copy_(prefix)
            num_snapshots += 1
            # The chain's attention output starts the prefix (served ``ChainLayer`` entry).
            prefix = self.chain.prefix_buffer(normed.shape[0])
        return normed, prefix, num_snapshots

    def _mla_input(
        self,
        layer: KimiK3Helix16UllDecoderLayer,
        x: torch.Tensor,
        prefix: torch.Tensor,
        valid: torch.Tensor,
        md: AttentionMetadata,
    ) -> None:
        """An MLA layer's input from the chain (served ``ull/chain.py`` ``ChainMlaInput.run``):
        the o_proj all-reduce left to the post-attention mix where it finishes it, and on up to
        :data:`MLAIN_MAX_ROWS` rows of an MLAIN layer the MXFP8 rows the run-exit consumer stored
        handed to the attention (``released``: its projection stages its weights before its wait,
        the consumer released it at entry)."""
        attn, a = layer.self_attn, layer.attn_res
        attn.ar_finish_deferred = _k3_attn_res_lamport_applies(
            prefix, valid, a.mlp_res_proj, a.mlp_res_norm, a.post_attention_layernorm
        )
        rows = x.shape[0]
        if not layer.mlain or rows > MLAIN_MXFP8_MAX_ROWS:
            return
        mx = self.chain.mx_input(layer.layer_idx, rows)
        if mx is None or not attn.takes_q_fused(x, md):
            return
        attn.mixer.next_input_mxfp8 = mx
        attn.released = attn.stages_weights(rows, md, x)

    def _mla_layer(
        self,
        layer: KimiK3Helix16UllDecoderLayer,
        x: torch.Tensor,
        prefix: torch.Tensor,
        valid: torch.Tensor,
        md: AttentionMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """An MLA layer up to its MoE (served program ``ChainMlaInput``, ``MlaAttention``,
        ``ChainPrefetch("attention_moe")``, ``Oproj``, ``ChainPrefetch("oproj_down")``,
        ``ChainPostMix``): ``(normed MoE input, prefix)``. ``x``: the normed attention input the
        chain's consumer stored."""
        chain, i, rows = self.chain, layer.layer_idx, x.shape[0]
        attn, a = layer.self_attn, layer.attn_res
        self._mla_input(layer, x, prefix, valid, md)
        try:
            call = attn.decode_attention(x, md)
            chain.prefetch("attention_moe", i, rows)
            attn_out = attn.decode_output(call)
        finally:
            attn.ar_finish_deferred = False
        chain.prefetch("oproj_down", i, rows)
        prefix, moe_in = _apply_attn_res_add_and_rmsnorm(
            prefix,
            attn_out,
            valid,
            a.mlp_res_proj,
            a.mlp_res_norm,
            a.post_attention_layernorm,
            entry_trigger=False,
        )
        return moe_in, prefix


# The weight mixins: weights.py imports the blocks above from this module.
from .weights import KimiK3Helix16UllTextWeights, KimiK3Helix16UllWeights  # noqa: E402


class KimiK3Helix16UllForCausalLM(
    KimiK3Helix16UllTextWeights, SpecDecOneEngineForCausalLM[KimiK3Helix16UllModelingV2, Any]
):
    """The Kimi K3 text model (KDA + MLA + MoE decoder layers) of ``ModelingV2KimiK3Nvfp4Sm107Helix16Moetp16ep1``.

    Registered under no architecture: the target class builds it as its ``llm``. Weight loading
    is the ``KimiK3Helix16UllTextWeights`` mixin's (``weights.py``).

    Speculative decoding: ``SpecDecOneEngineForCausalLM`` builds the drafter and the worker;
    :meth:`_install_private_spec` replaces them with the target's DSpark drafter and worker
    (``drafter.py``, ``dflash.py``)."""

    mamba_metadata_cls = K3UllMambaMetadata

    @property
    def runtime_hooks(self) -> Optional[RuntimeHooks]:
        """The base hooks plus the ones the target's profile turns on: the speculative decode
        step's head in one kernel (``embedding_gather``, ``decode_prologue``) and the served
        decode-step shell (``pack_step_inputs``; ``runtime.make_runtime_hooks``: the input
        commit, the folded overlap gathers, the pinned executor thread and the GIL switch
        interval), and the KV cache manager with KDA checkpoints (``kv_cache_manager_cls``:
        ``cache.K3UllCacheManager``). Built once: the engine binds its buffers to the hooks'
        objects."""
        hooks = getattr(self, "_step_shell_hooks", None)
        if hooks is None:
            features = _features()
            hooks = replace(
                super().runtime_hooks or DEFAULT_RUNTIME_HOOKS,
                kv_cache_manager_cls=K3UllCacheManager,
            )
            if features.embedding_gather:
                hooks = replace(hooks, decode_prologue=spec_decode_head)
            if features.pack_step_inputs:
                hooks = make_runtime_hooks(hooks)
            self._step_shell_hooks = hooks
        return hooks

    def __init__(self, model_config: ModelConfig):
        cfg = _get_text_config(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3 does not support pipeline parallelism"
        spec_config = getattr(model_config, "spec_config", None)
        if spec_config is None or not spec_config.spec_dec_mode.is_dspark():
            raise ValueError(
                "Kimi K3 ULL serves one-engine DSpark generation steps (each request verifies "
                "1 + 7 tokens); speculative decoding must be DSpark"
            )

        # Helix: swap in the repurposed mapping; restored after super().__init__.
        self._setup_helix_mappings(model_config, cfg, spec_config)
        if self.mapping_with_cp is None:
            raise ValueError("Kimi K3 ULL runs on a Helix CP generation server")
        super().__init__(
            KimiK3Helix16UllModelingV2(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        # The drafter and worker of main's shell are built (under the repurposed helix mapping,
        # which is still active here); the package's own replace them at this one point.
        self._install_private_spec(model_config)

        # Restore the CP original: executor-side helix bookkeeping keys off
        # has_cp_helix() at runtime.
        if self.mapping_with_cp is not None:
            model_config._frozen = False
            model_config.mapping = self.mapping_with_cp
            model_config._frozen = True

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ) -> Any:
        """The one-engine speculative step (served ``SpecDecOneEngineForCausalLM.forward``): the
        text model, the target head on ``spec_worker.target_head_rows`` (a generation-only batch's
        rows in place, no index copy), then the DSpark worker. A configuration that keeps main's
        worker (``install_k3_dspark`` returned False) takes the same path with
        ``SpecWorkerBase.target_head_rows`` (``hidden_states[spec_metadata.gather_ids]``)."""
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states)
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]
        worker = self.spec_worker
        if worker is None:
            return self.logits_processor.forward(
                hidden_states, self.lm_head, attn_metadata, return_context_logits
            )
        logits = worker.target_logits(
            worker.target_head_rows(hidden_states, attn_metadata, spec_metadata),
            self.lm_head,
            self.logits_processor,
            attn_metadata,
            spec_metadata,
            self.draft_model,
        )
        spec_input_ids = input_ids if input_ids is not None else kwargs.get("orig_input_ids")
        spec_position_ids = position_ids
        if attn_metadata.padded_num_tokens is not None:
            if spec_input_ids is not None:
                spec_input_ids = spec_input_ids[: attn_metadata.num_tokens]
            spec_position_ids = _slice_spec_position_ids(position_ids, attn_metadata.num_tokens)
        return worker(
            input_ids=spec_input_ids,
            position_ids=spec_position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )

    def _install_private_spec(self, model_config: ModelConfig) -> None:
        """Swap in the target's DSpark drafter and worker (``drafter.py``, ``dflash.py``).

        Called once from ``__init__``, right after ``SpecDecOneEngineForCausalLM.__init__`` and
        before the helix CP mapping is restored, i.e. with the same mapping main's drafter was
        built under. At that point main's shell has set ``self.spec_config``,
        ``self.draft_config``, ``self.draft_model`` (from the draft-builder registry, or None for a
        separate draft engine), ``self.spec_worker`` (``mapping_for_full_rows()`` mapping,
        ``set_target_model(self)`` done), ``self.use_separate_draft_kv_cache`` and ``self.epilogue``.

        A private implementation must replace ``draft_model`` / ``spec_worker`` consistently:
        swap them in ``self.epilogue`` (main appends both there), call ``set_target_model(self)`` on the
        new worker, keep ``self.lm_head`` a stock ``LMHead`` and ``self.logits_processor`` exactly a
        ``LogitsProcessor`` (main's vocab-parallel selection checks the type), and keep the
        contract main's ``DFlashSpecMetadata.prepare()`` has with the worker (the attributes it
        reads and the slot methods it calls, listed in ``drafter.py``).

        One-engine DSpark with a standalone MLA drafter gets the package's drafter
        (``drafter.py``) and worker (``drafter.py`` on ``dflash.py``)
        through :func:`~.drafter.install_k3_dspark`, which keeps that contract; every other
        configuration keeps main's.
        """
        install_k3_dspark(self, model_config)

    def _setup_helix_mappings(
        self,
        model_config: ModelConfig,
        cfg: "PretrainedConfig",
        spec_config: Optional["DecodingBaseConfig"],
    ) -> None:
        """Validate helix preconditions and stage the dual-mapping swap.

        DeepseekV3 pattern: the MLA layers keep the CP original; everything
        else is built against the repurposed mapping (CP ranks become TP
        ranks). Sets ``mapping_with_cp`` (restored after construction) and
        ``_repurposed_tp_mapping`` (load_weights shard selection); both stay
        None outside helix.
        """
        self.mapping_with_cp = None
        self._repurposed_tp_mapping = None
        if not model_config.mapping.has_cp_helix():
            return
        # The K3 helix targets run head-TP attention, the residual stream and the MoE on every
        # row of the CP group; there is no token-sliced (attention-DP) variant.
        if model_config.mapping.enable_attention_dp:
            raise NotImplementedError(
                "Kimi K3 helix runs without attention DP: KDA and MLA are head-TP over the CP "
                "group, so there is no request-level DP to add."
            )
        if spec_config is not None:
            # Helix supports only the standalone DSpark drafter (verified on
            # the V2 superblock ledger); reject everything else loudly rather
            # than let an unsupported spec mode run silently wrong.
            if not spec_config.spec_dec_mode.is_dspark():
                raise ValueError(
                    "Kimi K3 helix supports speculative decoding only with "
                    f"DSpark (standalone drafter); got "
                    f"{spec_config.decoding_type!r}."
                )
            # The SpeculationGate acceptance-rate trip permanently disables
            # speculation mid-flight while enable_spec_decode stays True;
            # in-flight helix requests then fall into the plain generation
            # loop whose position math (total_input_len_cp +
            # py_decoding_iter - 1) is stale once any draft token was
            # accepted -> silently wrong RoPE positions and KV slots. Reject
            # the trip wires until that loop is helix-group aware.
            if (
                spec_config.acceptance_rate_window_size is not None
                or spec_config.acceptance_rate_threshold is not None
            ):
                raise ValueError(
                    "Kimi K3 helix does not support the speculation "
                    "acceptance-rate gate (acceptance_rate_window_size / "
                    "acceptance_rate_threshold): dynamically disabling "
                    "speculation mid-flight leaves helix requests on a "
                    "single-token position formula."
                )
            # max_concurrency is the same trip wire by another name: the
            # drafter re-evaluates should_use_spec_decode on every scheduling
            # iteration and flips enable_spec_decode off as soon as the active
            # batch exceeds the cap. In-flight helix requests then take the
            # plain generation loop, whose position formula counts ITERATIONS
            # (total_input_len_cp + py_decoding_iter - 1) rather than
            # committed tokens, so it is stale by however many draft tokens
            # were accepted -- a wrong RoPE position, and across a ledger page
            # boundary a KV write to the wrong CP rank. Mirror the drafter's
            # own "unset" test (Drafter.should_use_spec_decode returns True
            # when max_concurrency is None) so an unset value is not rejected.
            if spec_config.max_concurrency is not None:
                raise ValueError(
                    "Kimi K3 helix does not support the speculation "
                    "concurrency cutoff (max_concurrency): disabling "
                    "speculation above the cap leaves in-flight helix "
                    "requests on a position formula that assumes one "
                    "committed token per iteration, which accepted draft "
                    "tokens break."
                )
            # draft_len_schedule is the user-facing alternative to
            # max_concurrency (llm_args rejects setting both) and reaches the
            # same end by a route that does not go through
            # should_use_spec_decode at all: py_executor turns speculation off
            # directly once the schedule yields draft_len 0 for the active
            # batch size. Guarding only max_concurrency would leave this door
            # open. Skip the schedule that llm_args synthesized from
            # max_concurrency, so a config that set only that field raises the
            # message above naming the field the user actually wrote.
            if (
                spec_config.draft_len_schedule is not None
                and not spec_config._translated_from_max_concurrency
            ):
                raise ValueError(
                    "Kimi K3 helix does not support the dynamic draft-length "
                    "schedule (draft_len_schedule): a batch size past the "
                    "last entry drops the draft length to 0 and turns "
                    "speculation off mid-run, leaving in-flight helix "
                    "requests on a position formula that assumes one "
                    "committed token per iteration."
                )
        cp = model_config.mapping.cp_size
        repurposed_tp = model_config.mapping.tp_size * cp
        if cfg.num_attention_heads % repurposed_tp != 0:
            raise ValueError(
                f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) "
                f"to divide the MLA head count ({cfg.num_attention_heads})."
            )
        kda_heads = cfg.linear_attn_config["num_heads"]
        if kda_heads % repurposed_tp != 0:
            raise ValueError(
                f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) to "
                f"divide the KDA head count ({kda_heads})."
            )
        # MoE splits apply to the repurposed tp*cp group (helix
        # moe_world_size = tp*cp); default EP-only. The Mapping constructor
        # skips its product check when both sizes are 1, so validate here.
        moe_ep = repurposed_tp
        if model_config.mapping.moe_tp_ep_user_specified:
            moe_tp = model_config.mapping.moe_tp_size
            moe_ep = model_config.mapping.moe_ep_size
            if moe_tp * moe_ep != repurposed_tp:
                raise ValueError(
                    f"Kimi K3 helix: moe_tensor_parallel_size ({moe_tp}) x "
                    f"moe_expert_parallel_size ({moe_ep}) must equal "
                    f"tp_size*cp_size ({repurposed_tp}): MoE runs on the "
                    "repurposed tp*cp group."
                )
        if cfg.num_experts and cfg.num_experts % moe_ep != 0:
            raise ValueError(
                f"Kimi K3 helix requires the MoE EP size ({moe_ep}) to "
                f"divide the routed expert count ({cfg.num_experts}): each "
                "EP rank of the repurposed tp*cp group holds whole experts."
            )
        self.mapping_with_cp = copy.deepcopy(model_config.mapping)
        repurposed = model_config.mapping.repurpose_helix_cp_to_tp()
        # repurpose passes resolved moe sizes, which the Mapping constructor
        # mistakes for user-specified values; restore the flag.
        repurposed.moe_tp_ep_user_specified = self.mapping_with_cp.moe_tp_ep_user_specified
        # load_weights shard selection must use this tp_rank; the restored
        # CP original's tp_rank is 0 on every rank.
        self._repurposed_tp_mapping = repurposed
        model_config._frozen = False
        model_config.mapping = repurposed
        # Side-channel for the MLA layers (avoids threading a kwarg through
        # every intermediate signature).
        model_config._helix_mapping_with_cp = self.mapping_with_cp
        model_config._frozen = True

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # - Keep block reuse opt-in because its combination with SA
        #   speculative decoding is not validated. Both settings use V2
        #   with the default use_kv_cache_manager_v2=True. The "auto"
        #   setting follows the model preference and can resolve to V1.
        # - tokens_per_block=64: keep the validated K3 page size. Earlier
        #   releases needed it to avoid a legacy MLA fallback that could not
        #   serve attention-DP's 96 query heads. The current FlashInfer
        #   TRTLLM-Gen backend declines H=96 at either page size, allowing
        #   the updated legacy TRTLLM-Gen autotuner to select a Q16 kernel.
        return {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "tokens_per_block": 64,
            }
        }

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


@disable_v2_by_default
class ModelingV2KimiK3Nvfp4Sm107Helix16Moetp16ep1(KimiK3Helix16UllWeights, PreTrainedModel):
    """Kimi K3 NVFP4 on SM107, Helix CP16 generation server, routed experts MoE TP16.

    The vision-language checkpoint's generation server: the text model
    ``KimiK3Helix16UllForCausalLM`` as ``llm``. The vision tower is not built (generation steps
    carry no media; the context server encodes them). The speculative-decoding and weight-loading
    attributes the engine reads off the outer model (``draft_model``, ``lm_head``,
    ``runtime_hooks``, ...) are the text model's. Weight loading is the
    ``KimiK3Helix16UllWeights`` mixin's (``weights.py``).

    Built only where the deployment sets ``modeling_v2`` itself (``@disable_v2_by_default``).
    Constructs on SM107 (Rubin) and on SM103 (GB300: the same code, the chain's Rubin-only
    kernels forced or replaced, for tests); raises ``NoTarget`` elsewhere. The KV cache manager
    is ``cache.K3UllCacheManager``, declared through the text model's ``runtime_hooks``.
    """

    _LANG_PREFIX = "language_model."
    mamba_metadata_cls = KimiK3Helix16UllForCausalLM.mamba_metadata_cls
    #: The profile the target runs (``ops/common/k3_ull_features.py``), logged at construction as
    #: the other SM107 targets log their ``K3Sm107Constants``.
    K3_CONSTANTS: ClassVar[K3Features] = PROFILE
    #: The devices the target constructs on.
    SUPPORTED_SMS = ((10, 7), (10, 3))

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
        """Default to the Python (v2) KV-cache transceiver (as Kimi K2.5).

        Only adopted when the user leaves
        ``cache_transceiver_config.transceiver_runtime`` at 'auto' and the
        effective backend is NIXL; otherwise the C++ transceiver is used.
        """
        return "PYTHON"

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        capability = torch.cuda.get_device_capability()
        if capability not in self.SUPPORTED_SMS:
            raise NoTarget(
                f"Kimi K3 ULL runs on SM107 (or SM103 for tests); this device is SM"
                f"{capability[0]}{capability[1]}"
            )
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

        self.llm = KimiK3Helix16UllForCausalLM(text_model_config)

        self._media_placeholder_token_id = getattr(
            config, "media_placeholder_token_id", MEDIA_PLACEHOLDER_TOKEN_ID
        )
        # Backing storage for the ``mm_token_ids`` property below — built once
        # here so the executor doesn't re-allocate per step.
        self._mm_token_ids = torch.tensor([self._media_placeholder_token_id], dtype=torch.int32)

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
    def runtime_hooks(self) -> Optional[RuntimeHooks]:
        """The text model's step-loop hooks: the engine reads them off the outer model."""
        return self.llm.runtime_hooks

    def set_guided_decoder(self, *args, **kwargs):
        return self.llm.set_guided_decoder(*args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "image.image_grid_thw",
            "video.pixel_values_videos",
            "video.video_grid_thw",
            "multimodal_embedding",
        ]

    @property
    def mm_token_ids(self) -> torch.Tensor:
        """Surface the in-vocab media placeholder to the model engine so
        ``runners.prepare_multimodal_indices`` selects the ``torch.isin`` predicate
        instead of the OOV (``>= vocab_size``) fallback (which would miss
        Kimi's placeholder and force ``fuse_input_embeds`` through the
        ``torch.where`` host-sync path on GPU input_ids).
        """
        return self._mm_token_ids

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
        if multimodal_params:
            raise NotImplementedError(
                "Kimi K3 ULL serves generation steps; media are encoded on the context server"
            )
        return self.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            **kwargs,
        )


__all__ = ["ModelingV2KimiK3Nvfp4Sm107Helix16Moetp16ep1"]
