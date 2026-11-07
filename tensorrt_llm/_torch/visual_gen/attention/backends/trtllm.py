# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Diffusion TRTLLM Attention Backend

Wraps TrtllmAttention with simplified metadata for visual generation (diffusion) models.
Handles the specifics of no-KV-cache operation and fused QKV requirements.
"""

import functools
import math
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Optional, Union

import torch

from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

if TYPE_CHECKING:
    from ....attention.backends.prims_ts.vc_attention.quantization import (
        VCAttentionOperands,
        VCAttentionPreprocessor,
    )

from ....attention.backends.fmha.fallback import FallbackFmha
from ....attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from ....attention.backends.sparse.params import SparseBackendForwardArgs, SparseParams
from ....attention.backends.sparse.timestep_phase import graph_phase_for_timestep
from ....attention.backends.trtllm import TrtllmAttention as BaseTrtllmAttention
from ....attention.backends.trtllm import TrtllmAttentionMetadata as BaseTrtllmAttentionMetadata
from ...cache import ForwardPlan, PagedKVTables
from ...cuda_graph_runner import resolved_extra_key
from .interface import AttentionBackend, AttentionTensorLayout
from .paged_rollout import forward_over_paged_rollout


def vc_smooth_phase(denoise_step: Optional[tuple[int, int]]) -> Optional[int]:
    """Return the V-Smooth phase of a denoising step for the CUDA graph key.

    ``VCAttentionPreprocessor.prepare`` chooses its path from the step alone,
    so a graph captured at one step is only valid for steps on the same path.
    2 regroups the value tokens and demeans V, 1 keeps the last grouping and
    demeans V, 0 runs the plain per-channel E4M3 V after the window. ``None``
    means no denoising step is active; CUDA Graph runners omit their phase key
    part in that case. The schedule constants are the defaults the layer-owned
    preprocessor runs with.
    """
    if denoise_step is None:
        return None
    from ....attention.backends.prims_ts.vc_attention.quantization import (
        VC_PERM_REFRESH_EVERY,
        VC_SMOOTH_STEP_FRACTION,
    )

    step, num_steps = denoise_step
    if step >= math.ceil(VC_SMOOTH_STEP_FRACTION * num_steps):
        return 0
    return 2 if step == 0 or step % VC_PERM_REFRESH_EVERY == 0 else 1


def vc_denoise_step_for_phase(phase: int) -> tuple[int, int]:
    """Return a representative denoising step of the schedule in ``phase``.

    Under CUDA graph capture the real step must not change what the graph
    records, so the host resolved phase selects one step of that phase on a
    schedule whose window holds exactly one refresh period.
    """
    from ....attention.backends.prims_ts.vc_attention.quantization import (
        VC_PERM_REFRESH_EVERY,
        VC_SMOOTH_STEP_FRACTION,
    )

    num_steps = math.ceil(VC_PERM_REFRESH_EVERY / VC_SMOOTH_STEP_FRACTION)
    if phase == 2:
        return 0, num_steps
    if phase == 1:
        return 1, num_steps
    return num_steps - 1, num_steps


def prepare_vc_operands(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qk_fp8: bool,
    repair_budget: float,
    preprocessor: Optional["VCAttentionPreprocessor"],
    denoise_step: Optional[tuple[int, int]],
) -> "VCAttentionOperands":
    """Quantize ``[B, S, H, D]`` bf16 Q, K and V for one VC-Attention forward.

    With a repair budget the highest-residual tokens are appended as V repair
    rows. Otherwise the layer-owned ``preprocessor`` groups the value tokens
    and splits V into tile means and E4M3 residuals on the V-Smooth schedule
    given by ``denoise_step``. Under QK8 ``q`` is rotated and quantized too.
    """
    from ....attention.backends.prims_ts.vc_attention.quantization import vc_quantize_repair

    vc_q = q if qk_fp8 else None
    if repair_budget:
        return vc_quantize_repair(k, v, budget=repair_budget, q=vc_q)
    if preprocessor is None:
        raise ValueError("the tile-mean VC-Attention recipe needs a preprocessor")
    return preprocessor.prepare(k, v, q=vc_q, denoise_step=denoise_step)


@functools.lru_cache(maxsize=None)
def _paged_context_kernel_exists(
    head_dim: int, dtype: torch.dtype, tokens_per_page: int, device: int
) -> bool:
    """Whether this build has a fused paged context attention kernel for this head dim, K/V and
    output dtype, and page size on ``device``'s GPU. The probe asks for a causal configuration;
    the full-attention kernels come from the same kernel sets."""
    with torch.cuda.device(device):
        # The lookup loads kernels through the driver, which needs a current context.
        torch.empty(1, device="cuda")
        binding = torch_dtype_to_binding(dtype)
        return bool(
            thop.fused_context_fmha_kernel_exists(head_dim, binding, tokens_per_page, binding)
        )


class TrtllmAttentionMetadata:
    """
    Simplified metadata adapter for diffusion models using TRTLLM backend.

    Lazy initialization with auto-growing capacity:
    - Metadata created only when capacity needs increase
    - prepare() called only when seq_lens actually change
    - Automatically reallocates when batch_size or seq_len exceeds current capacity

    Args:
        device: Target device for tensors.
        attention_metadata_state: Mutable model-scoped state shared by all
            attention layers in one model instance.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        attention_metadata_state: Optional[dict] = None,
    ):
        self.device = device or torch.device("cuda")
        if attention_metadata_state is None:
            raise ValueError(
                "TRTLLM attention requires `attention_metadata_state` to be provided "
                "by visual-gen config for model-scoped metadata sharing."
            )
        self._metadata_state = attention_metadata_state

        # Lazily created BaseTrtllmAttentionMetadata objects. Diffusion blocks
        # can launch video and audio attention back-to-back with different
        # sequence lengths, so keep separate metadata buffers per shape instead
        # of mutating one shared object while kernels may still be in flight.
        self._metadata_cache = self._metadata_state.setdefault("metadata_cache", {})
        self._metadata: Optional[BaseTrtllmAttentionMetadata] = None

        # Track prepared state
        self._cached_seq_lens: Optional[torch.Tensor] = None
        self._prepared = False

    def _needs_prepare(self, batch_size: int, seq_lens: torch.Tensor) -> bool:
        """Check if we need to call prepare() (current request seq_lens or shared metadata object seq_lens changed).

        Assumes uniform sequence length per batch; if per-sample lengths vary,
        we may need to check seq_lens tensor instead.

        In addition, multiple visual gen attention modules share one metadata object.  A
        different module may have prepared it for another sequence length even
        when this wrapper's local cached seq_lens are unchanged.
        """
        if not self._prepared:
            return True
        if self._cached_seq_lens is None:
            return True
        if self._cached_seq_lens.shape[0] != batch_size:
            return True
        if not torch.equal(self._cached_seq_lens[:batch_size], seq_lens):
            return True

        metadata = self._metadata
        if metadata is None:
            return True
        if getattr(metadata, "num_contexts", None) != batch_size:
            return True

        max_seq_len = seq_lens.max().item()
        if getattr(metadata, "max_seq_len", None) != max_seq_len:
            return True

        metadata_seq_lens = getattr(metadata, "seq_lens", None)
        if metadata_seq_lens is None or metadata_seq_lens.shape[0] < batch_size:
            return True
        if not torch.equal(metadata_seq_lens[:batch_size].to(seq_lens.device), seq_lens):
            return True

        return False

    def _create_metadata(self, batch_size: int, max_seq_len: int) -> None:
        """Create new metadata with given capacity."""
        self._metadata = BaseTrtllmAttentionMetadata(
            max_num_requests=batch_size,
            max_num_tokens=batch_size * max_seq_len,
            max_num_sequences=batch_size,
            kv_cache_manager=None,  # No KV cache for diffusion
            mapping=Mapping(),
            runtime_features=AttentionRuntimeFeatures(),
        )
        self._prepared = False  # Reset prepare state on new metadata

    def _select_cached_metadata(self, cached) -> None:
        self._metadata = cached["metadata"]
        self._prepared = cached["prepared"]
        self._cached_seq_lens = cached["seq_lens"]

    def prepare(
        self,
        batch_size: int,
        seq_lens: Union[int, torch.Tensor],
    ) -> BaseTrtllmAttentionMetadata:
        """
        Prepare metadata for a forward pass.

        Lazy behavior:
        - Creates metadata only when capacity needs increase
        - Calls prepare() only when (batch_size, max_seq_len) actually change
        """
        if isinstance(seq_lens, int):
            seq_lens_tensor = torch.full((batch_size,), seq_lens, dtype=torch.int32)
        else:
            seq_lens_tensor = seq_lens.to(dtype=torch.int32)
        max_seq_len = seq_lens_tensor.max().item()
        # Keep CUDA graph-captured metadata buffers stable per batch/seq-lens shape.
        cache_key = (batch_size, tuple(int(x) for x in seq_lens_tensor.tolist()))

        cached = self._metadata_cache.get(cache_key)
        if cached is None:
            self._create_metadata(batch_size, max_seq_len)
            cached = {
                "metadata": self._metadata,
                "prepared": False,
                "seq_lens": None,
            }
            self._metadata_cache[cache_key] = cached

        self._select_cached_metadata(cached)

        if self._needs_prepare(batch_size, seq_lens_tensor):
            cached_seq_lens = seq_lens_tensor.clone()
            self._metadata.seq_lens = cached_seq_lens
            self._metadata.num_contexts = batch_size
            self._metadata.max_seq_len = max_seq_len
            self._metadata.request_ids = list(range(batch_size))
            self._metadata.prepare()

            # Cache per-shape state without sharing the tensor across entries.
            cached["prepared"] = True
            cached["seq_lens"] = cached_seq_lens

            self._select_cached_metadata(cached)

        return self._metadata


class TrtllmAttention(BaseTrtllmAttention, AttentionBackend):
    """
    TRTLLM Attention wrapper for diffusion models.

    Handles:
    - Fused QKV requirement for TRTLLM kernel (used when no quant_attention_config is provided)
    - Metadata creation and preparation
    - No KV cache operation
    - SageAttention per-block QKV quantization (when a quant_attention_config is provided. requires unfused QKV)
    - PrimTS per-tensor FP8 recipes (quant_attention_config with zero block sizes, requires unfused QKV)
    - VC-Attention recipes (quant_attention_config with vc_attention, requires unfused QKV)
    - Separate-QKV forwarding for generic block-sparse attention and backends that reject fused QKV
    - Attention over a ``PagedRolloutKVCache``: the TRTLLM attention op reads the cache, given
      the queries only
    """

    # Per device, the attention op's workspace for paged K/V attention, shared by every layer.
    # The op grows it to the size it needs on the first, eager forward.
    _paged_kv_workspaces: ClassVar[dict[torch.device, torch.Tensor]] = {}

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        dtype: Optional[torch.dtype] = None,
        max_batch_size: int = 16,
        max_seq_len: int = 4096,
        quant_attention_config: Optional[QuantAttentionConfig] = None,
        attention_metadata_state: Optional[dict] = None,
        sparse_params: Optional[SparseParams] = None,
    ):
        num_kv_heads = num_kv_heads or num_heads
        if attention_metadata_state is None:
            raise ValueError(
                "TRTLLM attention requires `attention_metadata_state` to be provided "
                "by visual-gen config for model-scoped metadata and plan sharing."
            )

        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            quant_config=quant_config,
            sparse_params=sparse_params,
            dtype=dtype,
        )

        # TRTLLM expects flat [B*S, H*D] format
        self._preferred_layout = AttentionTensorLayout.NHD

        self.metadata = TrtllmAttentionMetadata(
            attention_metadata_state=attention_metadata_state,
        )

        self.quant_attention_config = quant_attention_config
        # The tile-mean VC-Attention recipe keeps its value grouping between denoising
        # steps in a preprocessor owned by this layer.
        self._vc_preprocessor: Optional["VCAttentionPreprocessor"] = None
        # The K/V shape the eager preprocessor currently holds a grouping for.
        self._vc_shape: Optional[tuple[int, int, int]] = None
        if quant_attention_config is not None and quant_attention_config.vc_attention:
            if sparse_params is not None:
                raise ValueError("VC-Attention does not compose with sparse attention.")
            if not any(fmha.supports_vc_attention for fmha in self._fmha_manager.fmha_libs):
                raise RuntimeError(
                    "The VC-Attention recipe needs the PrimTS FMHA library "
                    "(TLLM_FMHA_LIBS=+prims_ts), which is not enabled or not available on this "
                    "platform."
                )
            if not quant_attention_config.vc_repair_budget:
                from ....attention.backends.prims_ts.vc_attention.quantization import (
                    VCAttentionPreprocessor,
                )

                self._vc_preprocessor = VCAttentionPreprocessor()
        elif (
            quant_attention_config is not None
            and quant_attention_config.q_block_size == 0
            and not any(fmha.supports_per_tensor_fp8 for fmha in self._fmha_manager.fmha_libs)
        ):
            raise RuntimeError(
                "The per-tensor FP8 attention recipe needs the PrimTS FMHA library "
                "(TLLM_FMHA_LIBS=+prims_ts), which is not enabled or not available on this "
                "platform."
            )
        # Compiled callers guard on this flag; testing layer_idx per call would specialize, and
        # recompile, the attention frames for every layer.
        self.is_dense_layer = self.layer_idx in self.dense_layers
        self._dtype = dtype
        self._paged_kv_host_cache: dict[tuple[int, int, int, int], dict[str, torch.Tensor]] = {}

    @property
    def timestep_cutoff(self) -> Optional[float]:
        """Normalized timestep below which the sparse algorithm is enabled, if any."""

        return getattr(self.sparse_params, "disabled_until_timestep", None)

    @property
    def dense_layers(self) -> frozenset[int]:
        """Layer indices that always run dense attention."""

        return getattr(self.sparse_params, "dense_layers", frozenset())

    @torch.compiler.disable
    def _sparse_attn_phase(self, timestep: object) -> Optional[int]:
        """Return the dense (0) or sparse (1) phase of this call, or ``None``.

        The phase belongs to the transformer call, not to the layer: every layer
        of one call sees the same timestep and the same cutoff. The CUDA graph
        runner resolves it once per call on the host and publishes it during
        warmup and capture, so capture never reads the timestep tensor. Eager
        calls outside the runner derive it from the timestep. Layers without a
        timestep cutoff have no phase.
        """

        if self.timestep_cutoff is None:
            return None
        phase = resolved_extra_key("sparse_attn_phase")
        if phase is None:
            phase = graph_phase_for_timestep(timestep, disabled_until_timestep=self.timestep_cutoff)
        return phase

    @torch.compiler.disable
    def _vc_operands(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        kv_seq_len: int,
        timestep: object,
    ) -> "VCAttentionOperands":
        """Prepare the VC-Attention operands of this call from compact Q, K and V.

        The denoising step comes from the pipeline through the shared attention
        metadata state, so the V-Smooth window is a fraction of the schedule's
        steps on any scheduler. Under the CUDA graph runner the host resolved
        phase stands in for the step, which capture must not record, and
        selects a representative step of that phase.
        """
        del timestep
        quant_cfg = self.quant_attention_config
        denoise_step = None
        if self._vc_preprocessor is not None:
            phase = resolved_extra_key("vc_smooth_phase")
            if phase is not None:
                denoise_step = vc_denoise_step_for_phase(phase)
            else:
                denoise_step = self.metadata._metadata_state.get("denoise_step")
                # Eager runs keep the grouping of the current K/V shape only. Under
                # the CUDA graph runner every captured shape keeps its grouping, as
                # the captured graphs read those buffers by address.
                shape = (batch_size, kv_seq_len, self.num_kv_heads)
                if self._vc_shape is not None and shape != self._vc_shape:
                    from ....attention.backends.prims_ts.vc_attention.quantization import (
                        VCAttentionPreprocessor,
                    )

                    self._vc_preprocessor = VCAttentionPreprocessor()
                self._vc_shape = shape
        return prepare_vc_operands(
            q.view(batch_size, seq_len, self.num_heads, self.head_dim),
            k.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim),
            v.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim),
            qk_fp8=quant_cfg.qk_dtype == "fp8",
            repair_budget=quant_cfg.vc_repair_budget,
            preprocessor=self._vc_preprocessor,
            denoise_step=denoise_step,
        )

    def should_use_sparse(self, sparse_attn_phase: Optional[int]) -> bool:
        """Return whether this layer runs its sparse path in ``sparse_attn_phase``.

        Dense layers never do. Otherwise only the dense prefix (phase 0) is
        dense; a call without a phase (no cutoff or no timestep) is sparse.
        """

        if self.is_dense_layer:
            return False
        return sparse_attn_phase != 0

    # Needed to work with torch compile cause of attention metadata
    # make attn metadata as input for it to work
    @torch.compiler.disable
    def _prepare_metadata(self, batch_size: int, seq_len: int):
        return self.metadata.prepare(batch_size, seq_len)

    @torch.compile
    def _concat_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        kv_seq_len: int,
    ):
        # Separate Q, K, V provided - fuse them
        q = q.view(batch_size * seq_len, -1)
        k = k.view(batch_size * kv_seq_len, -1)
        v = v.view(batch_size * kv_seq_len, -1)
        qkv = torch.cat([q, k, v], dim=-1)
        return qkv

    @torch.compile
    def _compact_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        kv_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Separate Q, K, V stay separate - compact each into a contiguous token-major matrix.
        # Slices of a fused QKV projection are strided; the compiled copy keeps them on a
        # vectorized kernel, while already contiguous inputs pass through without a copy.
        q = q.reshape(batch_size * seq_len, -1).contiguous()
        k = k.reshape(batch_size * kv_seq_len, -1).contiguous()
        v = v.reshape(batch_size * kv_seq_len, -1).contiguous()
        return q, k, v

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
        sparse_backend_args: Optional[SparseBackendForwardArgs] = None,
        timestep: Optional[torch.Tensor] = None,
        kv: Optional[ForwardPlan] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with automatic metadata handling.

        Dimensions are derived from tensor shapes (NHD layout: ``[B, S, H, D]``).

        For diffusion models, expects:
        - Fused QKV: q contains [Q, K, V] concatenated, k and v are None
            - does not support SageAttention or block-sparse routes
        - OR separate Q, K, V which:
            - for regular TRTLLM attention, will be fused internally
            - for SageAttention, block-sparse routes, and the sparse calls of
              backends that reject fused QKV, will be passed to the core as
              separate tensors; the dense-layer and dense-phase calls of those
              backends are fused internally like regular attention

        Args:
            q: Query tensor [B, S, H, D] or fused QKV [B, S, H_qkv, D]
            k: Key tensor [B, S_kv, H_kv, D] or None if fused
            v: Value tensor [B, S_kv, H_kv, D] or None if fused
            batch_size: Batch size
            seq_len: Sequence length for Q
            attention_mask: Attention mask type
            seq_len_kv: Sequence length for K/V (for cross-attention, defaults to seq_len)
            sparse_backend_args: Module-predicted sparse inputs handed to the core
                prediction hooks. A ``block_sparse_inputs`` payload selects the
                generic block-sparse FMHA.
            timestep: Denoising timestep consumed by the sparse prediction hooks.
            kv: A ``ForwardPlan`` from a ``PagedRolloutKVCache``. When given,
                ``k``/``v`` (or the K/V heads of a fused ``q``) are the new tokens only:
                they are written at ``staging_offset`` and attention runs over everything
                cached before them plus themselves, full within a causal block and causal
                across causal blocks. The first ``kv.num_tokens`` rows are real; rows past
                them are padding, neither written to the cache nor attended, and their
                output rows are zero. ``batch_size`` must be 1.
            **kwargs: Backend-specific keyword arguments forwarded by the attention
                module, such as ``key_padding_mask``; ignored here, as by the other
                backends. The TRTLLM kernels have no key-padding input, so callers
                that need padded keys must guard at the model level or select the
                ``VANILLA`` backend.

        Returns:
            Output tensor [B, S, H*D]
        """
        if kv is not None:
            if batch_size != 1:
                raise ValueError("K/V cache attention takes one video: batch_size must be 1.")
            if k is None and v is None:
                # Fused QKV ``[1, S, H + 2 * H_kv, D]``: take Q, K and V as views, no copy.
                kv_heads = (q.shape[2] - self.num_heads) // 2
                k = q[:, :, self.num_heads : self.num_heads + kv_heads]
                v = q[:, :, self.num_heads + kv_heads :]
                q = q[:, :, : self.num_heads]
            elif k is None or v is None:
                raise ValueError("K/V cache attention needs separate q, k, v, or one fused qkv.")
            output = forward_over_paged_rollout(self, self.layer_idx, q, k, v, kv, attention_mask)
            return output.view(batch_size, seq_len, -1)
        sparse_attn_phase = self._sparse_attn_phase(timestep)
        block_sparse_inputs = (
            sparse_backend_args.block_sparse_inputs if sparse_backend_args is not None else None
        )
        # A backend that predicts routes inside the core needs separate Q, K and
        # V only on the calls that run its sparse path. Its dense-layer and
        # dense-phase calls take the fused path: the TRTLLM kernel serves dense
        # self-attention from fused QKV only.
        use_separate_qkv = (
            block_sparse_inputs is not None
            or self.quant_attention_config is not None
            or (not self.support_fused_qkv() and self.should_use_sparse(sparse_attn_phase))
        )
        if use_separate_qkv and (k is None or v is None):
            raise ValueError("This TRTLLM attention call requires separate q, k, and v tensors.")

        kv_seq_len = seq_len_kv if seq_len_kv is not None else seq_len
        prepared_metadata = self._prepare_metadata(batch_size, seq_len)
        quantization_kwargs = {}
        if use_separate_qkv:
            q, k, v = self._compact_qkv(q, k, v, batch_size, seq_len, kv_seq_len)
            quant_cfg = self.quant_attention_config
            if quant_cfg is not None and quant_cfg.vc_attention:
                quantization_kwargs = {
                    "vc_attention": True,
                    "vc_operands": self._vc_operands(
                        q, k, v, batch_size, seq_len, kv_seq_len, timestep
                    ),
                }
            elif quant_cfg is not None:
                quantization_kwargs = {
                    "sage_attn_num_elts_per_blk_q": quant_cfg.q_block_size,
                    "sage_attn_num_elts_per_blk_k": quant_cfg.k_block_size,
                    "sage_attn_num_elts_per_blk_v": quant_cfg.v_block_size,
                    "sage_attn_qk_int8": quant_cfg.qk_dtype == "int8",
                    "sage_attn_smooth_k": quant_cfg.smooth_k,
                    "sage_attn_smooth_v": quant_cfg.smooth_v,
                    "qk_fp8_per_tensor": quant_cfg.q_block_size == 0
                    and quant_cfg.qk_dtype == "fp8",
                    "v_fp8_per_tensor": quant_cfg.v_block_size == 0 and quant_cfg.v_dtype == "fp8",
                }
        else:
            if k is None and v is None:
                q = q.reshape(batch_size * seq_len, -1)
            else:
                q = self._concat_qkv(q, k, v, batch_size, seq_len, kv_seq_len)
            k = None
            v = None
        output = super().forward(
            q=q,
            k=k,
            v=v,
            metadata=prepared_metadata,
            forward_args=AttentionForwardArgs(
                attention_mask=attention_mask,
                timestep=timestep,
                sparse_attn_phase=sparse_attn_phase,
                sparse_backend_args=sparse_backend_args,
                **quantization_kwargs,
            ),
        )
        return output.view(batch_size, seq_len, -1)

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    def support_fused_qkv(self) -> bool:
        """Standard path fuses QKV; SageAttention path does not."""
        return self.quant_attention_config is None

    def support_kv_cache(self) -> bool:
        """Whether ``forward`` accepts a ``kv`` plan.

        The cache path calls the TRTLLM attention op with the queries only, so the op attends to
        K/V already in the cache and does not write to it; the shared helper stages the new
        tokens. It does not combine with quantized or sparse attention.
        """
        return self.quant_attention_config is None and self.sparse_params is None

    def support_kv_cache_page_size(
        self, tokens_per_page: int, dtype: Optional[torch.dtype] = None
    ) -> bool:
        """Whether this build has a fused paged context attention kernel for this head dim,
        page size and bf16 or fp16 K/V on the current GPU. Without one, the op would fall back
        to an unfused path that cannot read the cache."""
        dtype = dtype or self._dtype
        return (
            self.support_kv_cache()
            and dtype in (torch.bfloat16, torch.float16)
            and _paged_context_kernel_exists(
                self.head_dim, dtype, tokens_per_page, torch.cuda.current_device()
            )
        )

    def _paged_kv_host_tensors(
        self, kv_pool: torch.Tensor, batch_size: int, seq_len: int, max_kv_len: int
    ) -> dict[str, torch.Tensor]:
        """Host-side inputs of the op for one layer's pool and one causal block geometry. They
        are fixed for that pair, so they are built once and reused, also under CUDA graphs."""
        key = (kv_pool.data_ptr(), batch_size, seq_len, max_kv_len)
        tensors = self._paged_kv_host_cache.get(key)
        if tensors is None:
            lengths = partial(torch.full, (batch_size,), dtype=torch.int32)
            tensors = {
                # The op addresses a layer's pages from its pool's base pointer. Each layer's
                # pool is presented as a one-layer pool of its own: pool 0, layer 0 within it.
                "host_kv_cache_pool_pointers": torch.tensor(
                    [[kv_pool.data_ptr(), 0]], dtype=torch.int64
                ),
                "host_kv_cache_pool_mapping": torch.zeros((1, 2), dtype=torch.int32),
                # Upper bounds: the actual key counts are on the device (seq_lens_kv).
                "host_past_key_value_lengths": lengths(max_kv_len),
                "host_context_lengths": lengths(seq_len),
                "host_total_kv_lens": torch.tensor([batch_size * max_kv_len, 0], dtype=torch.int32),
                "host_request_types": lengths(0),
            }
            self._paged_kv_host_cache[key] = tensors
        return tensors

    def forward_paged_kv(
        self, q: torch.Tensor, kv_pool: torch.Tensor, tables: PagedKVTables
    ) -> torch.Tensor:
        """See ``AttentionBackend.forward_paged_kv``."""
        if not self.support_kv_cache():
            raise NotImplementedError(
                "K/V cache attention does not combine with quantized or sparse attention."
            )
        batch, seq_len, num_heads, head_dim = q.shape
        page_size = kv_pool.shape[3]
        max_kv_len = tables.kv_block_offsets.shape[-1] * page_size
        num_tokens = batch * seq_len
        # Queries only: with no K/V in the input, the op reads K/V from the cache and leaves it
        # as it is.
        q = q.reshape(num_tokens, num_heads * head_dim).contiguous()
        out = torch.empty_like(q)
        host = self._paged_kv_host_tensors(kv_pool, batch, seq_len, max_kv_len)
        workspace = self._paged_kv_workspaces.setdefault(
            q.device, torch.empty(0, dtype=torch.int8, device=q.device)
        )
        if workspace.numel() == 0 and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Paged K/V attention workspace first needed during CUDA graph capture; run the "
                "forward eagerly once before capturing."
            )
        rope = self.rope_params
        FallbackFmha.attention(
            q=q,
            k=None,
            v=None,
            output=out,
            output_sf=None,
            workspace=workspace,
            # Per causal block: its key count (the cached keys it sees plus its own tokens) and
            # its query count.
            sequence_length=tables.seq_lens_kv,
            context_lengths=tables.seq_lens_q,
            # The op's view of the pool: K/V page ids per causal block, for pool 0.
            kv_cache_block_offsets=tables.kv_block_offsets.unsqueeze(0),
            **host,
            max_context_q_len_override=None,
            cache_indirection=None,
            kv_scale_orig_quant=None,
            kv_scale_quant_orig=None,
            out_scale=None,
            rotary_inv_freq=self.rotary_inv_freq,
            rotary_cos_sin=self.rotary_cos_sin,
            latent_cache=None,
            q_pe=None,
            block_ids_per_seq=None,
            attention_sinks=None,
            is_fused_qkv=False,
            update_kv_cache=False,
            predicted_tokens_per_seq=1,
            local_layer_idx=0,
            num_heads=num_heads,
            num_kv_heads=kv_pool.shape[2],
            head_size=head_dim,
            tokens_per_block=page_size,
            max_num_requests=batch,
            max_context_length=seq_len,
            max_seq_len=max_kv_len,
            # No cyclic window: every key a causal block lists is attended.
            attention_window_size=max_kv_len,
            beam_width=1,
            # Full attention within a causal block; the page lists make it causal across them.
            mask_type=int(AttentionMaskType.padding),
            quant_mode=0,
            q_scaling=self.q_scaling,
            position_embedding_type=int(self.position_embedding_type),
            rope_dim=rope.dim,
            rope_base=rope.theta,
            rope_scale_type=int(rope.scale_type),
            rope_scale=rope.scale,
            rope_short_m_scale=rope.short_m_scale,
            rope_long_m_scale=rope.long_m_scale,
            rope_max_positions=rope.max_positions,
            rope_original_max_positions=rope.original_max_positions,
            use_paged_context_fmha=True,
            attention_input_type=int(AttentionInputType.context_only),
            is_mla_enable=False,
            chunked_prefill_buffer_batch_size=None,
            q_lora_rank=None,
            kv_lora_rank=None,
            qk_nope_head_dim=None,
            qk_rope_head_dim=None,
            v_head_dim=None,
            rope_append=None,
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            helix_position_offsets=None,
            helix_is_inactive_rank=None,
            attention_chunk_size=None,
            softmax_stats_tensor=None,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            is_spec_dec_tree=False,
            spec_decoding_generation_lengths=None,
            spec_decoding_position_offsets=None,
            spec_decoding_packed_mask=None,
            spec_decoding_bl_tree_mask_offset=None,
            spec_decoding_bl_tree_mask=None,
            spec_bl_tree_first_sparse_mask_offset_kv=None,
            sparse_kv_indices=None,
            sparse_kv_offsets=None,
            sparse_attn_indices=None,
            sparse_attn_offsets=None,
            sparse_attn_indices_block_size=1,
            num_contexts=batch,
            num_ctx_tokens=num_tokens,
            max_num_sequences=batch,
        )
        return out.view(batch, seq_len, num_heads, head_dim)
