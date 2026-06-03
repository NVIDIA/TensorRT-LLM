"""RocketKV — sparse-attention executor implementation.

This is the executor-framework implementation of RocketKV. The attention
shims, metadata, and executor here plug into the sparse-attention framework
(``BaseKVCacheCompressionExecutor`` + ``KVCacheBehaviorCoordinator``,
inheriting ``BaseResourceManager``). The algorithm body lives in the
executor's HOOK callbacks rather than in a cache-manager subclass.

Paper: arxiv 2502.14837 — 2-stage hybrid:
- Stage I (prefill): SnapKV-style top-pB physical eviction + per-page KT
  summary build streaming through every attention layer. Cache shrinks
  to ``prompt_budget`` tokens by end of prefill.
- Stage II (decode): query-aware HSA mask over the shrunk cache, using
  KT summaries as the page-level lookup table.

Structure:

| Algorithm step                                         | Location                                     |
|--------------------------------------------------------|----------------------------------------------|
| Stage I-a SnapKV sparse-kv prediction + KT build       | ``RocketKV.on_context_attention`` (HOOK 2)   |
| Stage II query-aware HSA mask                          | ``RocketKV.on_generation_attention`` (HOOK 4)|
| Stage I-b physical-evict rewind                        | ``RocketKV.on_context_end`` (HOOK 3)         |
| per-request init                                       | ``RocketKV.on_request_init`` (HOOK 1)        |
| KT pool wiring                                          | ``RocketKV.__init__``                        |
| metadata buffer setup / per-iter prepare               | ``RocketKVTrtllmAttentionMetadata`` (KT-     |
|                                                        | related fields access the executor via       |
|                                                        | ``self.coordinator.get_executor("sparse")``) |

KT_CACHE storage: the KT pool is registered as a V2 sub-page pool (Path A),
sharing the K/V page allocation so V2 frees KT sub-pages with the block.

Two public algorithm values coexist via the algorithm discriminator: the
legacy ``rocket`` algorithm (``algorithm="rocket"``, which routes through a
cache-manager subclass in ``sparse/rocket.py``) and this ``rocketkv``
algorithm (``algorithm="rocketkv"``, the executor-framework method here).
"""

import math
from typing import TYPE_CHECKING, ClassVar, List, Optional

import torch
from triton import next_power_of_2

from tensorrt_llm._torch.attention_backend.sparse.kv_cache_compression_executor import (
    SparseAttentionExecutor,
    SparseAttentionIndices,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttention, VanillaAttentionMetadata

# KT cache pool is EXECUTOR-OWNED (per-layer pool + free-list BlockManager
# in RocketKV.__init__); V2 core has no KT pool.
from tensorrt_llm._utils import prefer_pinned

from .kernel import (
    triton_bmm,
    triton_flatten_to_batch,
    triton_rocket_batch_to_flatten,
    triton_rocket_paged_kt_cache_bmm,
    triton_rocket_qk_split,
    triton_rocket_reduce_scores,
    triton_rocket_update_kt_cache_ctx,
    triton_rocket_update_kt_cache_gen,
    triton_softmax,
    triton_topk,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2


# =========================================================================
# L0 attention shim — minimal subclasses for backend routing                #
#                                                                          #
# These are intentionally thin: the framework HOOK 2/4 callbacks fire from #
# the base ``TrtllmAttention.forward`` / ``VanillaAttention.forward`` via  #
# ``metadata.coordinator.on_*_attention(...)``. The shim's only job is to  #
# carry the RocketKV-specific Metadata class (which holds prompt_budget,   #
# kt_cache_block_offsets, etc.) so backend routing picks them up.          #
# =========================================================================


class RocketKVTrtllmAttention(TrtllmAttention):
    """RocketKV attention shim — TRT-LLM backend.

    - No ``sparse_kv_predict`` / ``sparse_attn_predict`` method overrides;
      those algorithm bodies live in :class:`RocketKV` executor's HOOK 2/4
      callbacks (fired by base ``TrtllmAttention.forward`` via
      ``metadata.coordinator``).
    - Holds :class:`RocketKVTrtllmAttentionMetadata` for backend routing.
    """

    Metadata: ClassVar[type] = None  # set below after metadata class defined


class RocketKVVanillaAttention(VanillaAttention):
    """RocketKV attention shim — vanilla backend (used in tests).

    Algorithm body (HOOK 2/4 in :class:`RocketKV`) is the single source
    of truth; vanilla path uses Python/torch kernels instead of triton.
    """

    Metadata: ClassVar[type] = None  # set below

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RocketKV does not support the VANILLA attention backend yet; "
            "use attn_backend='TRTLLM'."
        )


# =========================================================================
# L0 metadata                                                              #
#                                                                          #
# Access to KT cache pool / max_kt_blocks_per_seq is resolved via          #
# ``self._rocket_executor`` (resolved through metadata.coordinator)        #
# rather than through ``self.kv_cache_manager.<...>``.                     #
# =========================================================================


class RocketKVTrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Metadata for the RocketKV TRT-LLM attention path.

    KT-cache-related access routes through the coordinator-resolved
    executor instead of a cache-manager subclass.
    """

    @property
    def _rocket_executor(self) -> Optional["RocketKV"]:
        """Resolve the RocketKV executor instance through the coordinator.
        Returns ``None`` if no behavior-layer sparse method is configured
        (e.g., dummy attn_metadata init path); the metadata then skips
        KT-cache-related buffer setup."""
        if getattr(self, "coordinator", None) is None:
            return None
        return self.coordinator.get_executor("sparse")  # type: ignore[return-value]

    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attention_config is None:
            raise ValueError("Sparse attention config is not set")
        self.prompt_budget = self.sparse_attention_config.prompt_budget
        # ``RocketKVSparseAttentionConfig`` doesn't carry window_size /
        # topk / kernel_size — they're carried by the executor instance.
        # Resolve through coordinator if available; fall back to defaults.
        e = self._rocket_executor
        self.window_size = e.window_size if e else 32
        self.page_size = self.sparse_attention_config.page_size
        self.topk = e.topk if e else 256

        assert self.page_size == next_power_of_2(self.page_size), "Page size must be a power of 2"

        capture_graph = self.is_cuda_graph

        # ---- Cumulative valid sequence lengths for query and key
        self.q_cu_seqlens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="q_cu_seqlens_cuda",
            capture_graph=capture_graph,
        )
        self.q_cu_seqlens = torch.zeros_like(
            self.q_cu_seqlens_cuda, device="cpu", dtype=torch.int32
        )

        self.k_cu_seqlens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="k_cu_seqlens_cuda",
            capture_graph=capture_graph,
        )
        self.k_cu_seqlens = torch.zeros_like(
            self.k_cu_seqlens_cuda, device="cpu", dtype=torch.int32
        )

        # ---- Context length of RocketKV key for each valid sequence
        self.k_context_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences,),
            dtype=torch.int32,
            cache_name="k_context_lens_cuda",
            capture_graph=capture_graph,
        )
        self.k_context_lens = torch.zeros_like(
            self.k_context_lens_cuda, device="cpu", dtype=torch.int32
        )

        # ---- Start index of RocketKV key for each valid sequence
        self.k_context_start_cuda = self.get_empty(
            None,
            (self.max_num_sequences,),
            dtype=torch.int32,
            cache_name="k_context_start_cuda",
            capture_graph=capture_graph,
        )

        # ---- Cumulative context lengths
        self.context_cumsum_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="context_cumsum_cuda",
            capture_graph=capture_graph,
        )
        self.context_cumsum = torch.zeros_like(
            self.context_cumsum_cuda, device="cpu", dtype=torch.int32
        )

        # ---- Sparse kv indices offsets for context phase
        self.sparse_offsets_ctx_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="sparse_offsets_ctx_cuda",
            capture_graph=capture_graph,
        )
        self.sparse_offsets_ctx = torch.zeros_like(
            self.sparse_offsets_ctx_cuda, device="cpu", dtype=torch.int32
        )

        # ---- Valid sequence indices
        self.valid_seq_indices_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences,),
            dtype=torch.int32,
            cache_name="valid_seq_indices_cuda",
            capture_graph=capture_graph,
        )

        # ---- KT cache block offsets (accesses executor for max_kt_blocks_per_seq)
        max_kt_blocks = e.max_kt_blocks_per_seq if e else 0
        if max_kt_blocks > 0:
            self.kt_cache_block_offsets = self.get_empty(
                self.cuda_graph_buffers,
                [self.max_num_sequences, max_kt_blocks],
                dtype=torch.int32,
                cache_name="kt_cache_block_offsets",
                capture_graph=capture_graph,
            )
            self.host_kt_cache_block_offsets = torch.zeros_like(
                self.kt_cache_block_offsets,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
        else:
            # No executor wired (dummy/init path) — leave as None; algorithm
            # body will early-return on access.
            self.kt_cache_block_offsets = None
            self.host_kt_cache_block_offsets = None

        # ---- Number of KT tokens
        self.num_kt_tokens = torch.empty(
            self.max_num_sequences,
            device="cpu",
            dtype=torch.int32,
        )

        # ---- Cumulative KT lengths
        self.cum_kt_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="cum_kt_lens_cuda",
            capture_graph=capture_graph,
        )
        self.cum_kt_lens = torch.zeros_like(self.cum_kt_lens_cuda, device="cpu", dtype=torch.int32)

        # ---- Sparse attn indices offsets for generation phase
        self.sparse_offsets_gen_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int32,
            cache_name="sparse_offsets_gen_cuda",
            capture_graph=capture_graph,
        )
        self.sparse_offsets_gen = torch.zeros_like(
            self.sparse_offsets_gen_cuda, device="cpu", dtype=torch.int32
        )

        # ---- Maximum number of KT tokens
        self.max_kt_tokens = (self.max_seq_len + self.page_size - 1) // self.page_size

    @property
    def kt_tokens_per_block(self) -> Optional[int]:
        """Proxy to executor's kt_tokens_per_block. Used by triton
        kernels (passed as a kernel arg)."""
        e = self._rocket_executor
        return e.kt_tokens_per_block if e else None

    def prepare(self):
        """Per-iteration metadata setup.

        Rewind ``num_cached_tokens_per_seq`` for
        sequences whose prompt exceeded ``prompt_budget``, clamp prompt_lens
        to prompt_budget for generation requests, and build the various
        cu-seqlen / sparse-offset / valid-mask CUDA buffers used by the
        triton kernels in HOOK 2/4.
        """

        assert self.kv_cache_manager is not None, "RocketKV always runs with a KV cache manager"
        num_contexts = self.num_contexts
        num_generations = self.num_generations
        num_requests = num_contexts + num_generations

        # Rewind gen-request num_cached to match the Stage I-b
        # physical evict (cache shrunk to prompt_budget+1). Context requests
        # keep their real num_cached: under chunked prefill a continuation
        # chunk's K/V must land at the correct cache offset (num_cached>0); for
        # non-chunked / first-chunk prefill it is already 0, so this is a no-op.
        # No env toggle -- this runs in the executor worker process, which would
        # not see a runtime-set environment variable.
        for i in range(num_requests):
            if i >= num_contexts and self.prompt_lens[i] > self.prompt_budget:
                self.kv_cache_params.num_cached_tokens_per_seq[i] += (
                    self.prompt_budget - self.prompt_lens[i]
                )

        super().prepare()
        # Clamp gen-request prompt_lens to prompt_budget
        # (paired with the Stage I-b cache rewind to prompt_budget+1).
        _prompt_lens = self.prompt_lens.copy()
        for i in range(num_requests):
            if i >= num_contexts:
                _prompt_lens[i] = min(_prompt_lens[i], self.prompt_budget)
        _prompt_lens = torch.tensor(_prompt_lens, dtype=torch.int, device="cpu")
        self.prompt_lens_cpu[: self.num_seqs].copy_(_prompt_lens)
        self.prompt_lens_cuda[: self.num_seqs].copy_(
            self.prompt_lens_cpu[: self.num_seqs], non_blocking=True
        )
        self.prompt_lens_cuda_runtime = self.prompt_lens_cuda[: self.num_seqs]
        self.prompt_lens_cpu_runtime = self.prompt_lens_cpu[: self.num_seqs]

        # Copy KT block offsets from executor
        e = self._rocket_executor
        if e is not None and self.host_kt_cache_block_offsets is not None:
            _kt_counts = [
                math.ceil(int(self.kv_lens[i]) / self.page_size) for i in range(self.num_seqs)
            ]
            e.copy_kt_block_offsets(self.request_ids, self.host_kt_cache_block_offsets, _kt_counts)
            self.kt_cache_block_offsets[: self.num_seqs].copy_(
                self.host_kt_cache_block_offsets[: self.num_seqs], non_blocking=True
            )

        # ---- Context phase setup
        self.context_cumsum[1 : self.num_contexts + 1] = torch.cumsum(
            self.prompt_lens_cpu[: self.num_contexts], dim=0
        )
        self.context_cumsum_cuda[: self.num_contexts + 1].copy_(
            self.context_cumsum[: self.num_contexts + 1], non_blocking=True
        )

        # Filter sequences too short for sparse kv prediction
        valid_mask = self.prompt_lens_cpu[: self.num_contexts] >= self.prompt_budget
        valid_seq_indices = torch.where(valid_mask)[0]
        invalid_seq_indices = torch.where(~valid_mask)[0]
        valid_batch_size = len(valid_seq_indices)
        self.valid_seq_indices_cuda[:valid_batch_size].copy_(valid_seq_indices, non_blocking=True)

        # k_context_lens for valid sequences
        self.k_context_lens[:valid_batch_size] = (
            self.prompt_lens_cpu[valid_seq_indices] - self.window_size
        )
        self.k_context_lens_cuda[:valid_batch_size].copy_(
            self.k_context_lens[:valid_batch_size], non_blocking=True
        )

        sparse_counts_ctx = torch.zeros(self.num_contexts, dtype=torch.int32, device="cpu")
        sparse_counts_ctx[valid_seq_indices] = self.prompt_budget
        sparse_counts_ctx[invalid_seq_indices] = self.prompt_lens_cpu[invalid_seq_indices]

        self.sparse_offsets_ctx[1 : self.num_contexts + 1] = torch.cumsum(sparse_counts_ctx, dim=0)
        self.sparse_offsets_ctx_cuda[: self.num_contexts + 1].copy_(
            self.sparse_offsets_ctx[: self.num_contexts + 1], non_blocking=True
        )

        # q_cu_seqlens
        self.q_cu_seqlens[: valid_batch_size + 1] = (
            torch.arange(valid_batch_size + 1, device="cpu", dtype=torch.int32) * self.window_size
        )
        self.q_cu_seqlens_cuda[: valid_batch_size + 1].copy_(
            self.q_cu_seqlens[: valid_batch_size + 1], non_blocking=True
        )

        self.k_cu_seqlens[1 : valid_batch_size + 1] = torch.cumsum(
            self.k_context_lens[:valid_batch_size], dim=0
        )
        self.k_cu_seqlens_cuda[: valid_batch_size + 1].copy_(
            self.k_cu_seqlens[: valid_batch_size + 1], non_blocking=True
        )

        if valid_batch_size > 0:
            self.max_rocket_k_ctx_len = self.k_context_lens[:valid_batch_size].max().item()
            self.total_rocket_k_ctx_tokens = self.k_cu_seqlens[valid_batch_size].item()
        else:
            self.max_rocket_k_ctx_len = 0
            self.total_rocket_k_ctx_tokens = 0

        self.valid_batch_size = valid_batch_size
        self.total_sparse_ctx_indices = self.sparse_offsets_ctx[self.num_contexts].item()

        # ---- Generation phase setup
        self.num_kt_tokens[: self.num_generations] = (
            self.kv_lens[self.num_contexts : self.num_seqs] + self.page_size - 1
        ) // self.page_size

        self.cum_kt_lens[1 : self.num_generations + 1] = torch.cumsum(
            self.num_kt_tokens[: self.num_generations], dim=0
        )
        self.cum_kt_lens_cuda[: self.num_generations + 1].copy_(
            self.cum_kt_lens[: self.num_generations + 1], non_blocking=True
        )

        self.total_kt_tokens = self.num_generations * self.max_kt_tokens

        topk_tensor = torch.tensor(self.topk, dtype=torch.int32)

        sparse_counts_gen = torch.minimum(topk_tensor, self.num_kt_tokens[: self.num_generations])

        self.sparse_offsets_gen[1 : self.num_generations + 1] = torch.cumsum(
            sparse_counts_gen[: self.num_generations], dim=0
        )
        self.sparse_offsets_gen_cuda[: self.num_generations + 1].copy_(
            self.sparse_offsets_gen[: self.num_generations + 1], non_blocking=True
        )

        self.total_sparse_gen_indices = self.topk * self.num_generations


class RocketKVVanillaAttentionMetadata(VanillaAttentionMetadata):
    """Metadata for the RocketKV vanilla decode path.

    The vanilla decode path is not yet implemented; this class is the
    structural placeholder so backend dispatch resolves correctly.
    """

    @property
    def _rocket_executor(self) -> Optional["RocketKV"]:
        if getattr(self, "coordinator", None) is None:
            return None
        return self.coordinator.get_executor("sparse")  # type: ignore[return-value]

    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attention_config is None:
            raise ValueError("Sparse attention config is not set")
        self.prompt_budget = self.sparse_attention_config.prompt_budget
        e = self._rocket_executor
        max_kt_blocks = e.max_kt_blocks_per_seq if e else 0
        if max_kt_blocks > 0:
            self.kt_cache_block_offsets = torch.empty(
                [self.max_num_sequences, max_kt_blocks],
                dtype=torch.int32,
                device="cuda",
            )
            self.host_kt_cache_block_offsets = torch.zeros_like(
                self.kt_cache_block_offsets,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
        else:
            self.kt_cache_block_offsets = None
            self.host_kt_cache_block_offsets = None

    def prepare(self) -> None:
        raise NotImplementedError(
            "RocketKV does not support the VANILLA attention backend yet; "
            "use attn_backend='TRTLLM'."
        )


# Wire Metadata class refs (set after Metadata classes are defined).
RocketKVTrtllmAttention.Metadata = RocketKVTrtllmAttentionMetadata
RocketKVVanillaAttention.Metadata = RocketKVVanillaAttentionMetadata


# RocketKV uses ``KVCacheManagerV2`` directly (no subclass). Stage I-b
# physical evict goes through its public ``rewind_kv_cache`` (=
# ``self.impl.rewind_kv_cache``). The KT cache pool shares the K/V page
# allocation, so the manager core is untouched.
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2  # noqa: E402
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2._common import (  # noqa: E402
    BAD_PAGE_INDEX as _BAD_PAGE,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole as _DataRole  # noqa: E402

# =========================================================================
# L2 executor — algorithm body                                              #
#                                                                          #
# The KT cache pool is registered as a V2 sub-page pool (Path A), sharing  #
# the K/V page allocation; the executor reads it via the V2 manager.       #
# =========================================================================


class RocketKV(SparseAttentionExecutor):
    """RocketKV sparse-attention executor. See module docstring for the
    structure table.

    RocketKV is a 2-stage hybrid sparse attention method:
    - Stage I (prefill): SnapKV-style top-pB physical eviction (HOOK 3)
      + per-page KT summary build streaming through every attention layer
      (HOOK 2 side-effect).
    - Stage II (decode): query-aware HSA mask over the shrunk cache,
      using KT summaries as the page-level lookup table (HOOK 4).
    """

    axis: ClassVar[str] = "sparse"

    # RocketKV physically deletes tokens at prefill end (Stage I-b SnapKV
    # top-pB keep + ``compact_request_cache``).
    physically_evicts_kv: ClassVar[bool] = True

    # Stage I keep-set depends on this prompt's last-window attn scores +
    # KT_CACHE is request-specific → KV reuse between requests breaks.
    supports_kv_cache_reuse: ClassVar[bool] = False

    # RocketKV uses plain KVCacheManagerV2; Stage I-b evict goes through
    # its public rewind_kv_cache (= impl.rewind_kv_cache).
    kv_cache_manager_class: ClassVar[Optional[type]] = None

    # Access type for different dtype sizes.
    _access_type = {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
        8: torch.int64,
    }

    def __init__(
        self,
        kv_cache_manager: "KVCacheManagerV2",
        page_size: int = 16,
        prompt_budget: int = 2048,
        kt_cache_dtype: str = "bfloat16",
        kt_tokens_per_block: Optional[int] = None,
        # Hyperparameters not yet carried by the config (defaults applied
        # here); these may move into RocketKVSparseAttentionConfig later.
        window_size: int = 32,
        kernel_size: int = 5,
        topk: int = 256,
        topr: int = 32,
    ):
        super().__init__(kv_cache_manager)
        self.page_size = page_size
        self.prompt_budget = prompt_budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.topk = topk
        self.topr = topr

        # Chunked-prefill: per-request full prompt_len (HOOK1) + per-(rid,layer)
        # chunk accumulation so HOOK2 can score the full prefix at the true
        # last chunk. Activated automatically when chunking is in effect.
        self._cp_prompt_len: dict = {}  # rid -> full prompt len
        self._cp_k_accum: dict = {}  # (rid, layer) -> [qkv_chunk]
        self._cp_seen: dict = {}  # (rid, layer) -> ctx tokens seen
        self._cp_cc: list = []  # per-forward chunk-local ctx cumsum
        # (captured at layer 0, reused 1..N-1;
        # layer-0 override clobbers metadata)

        # Path A: KT pool is V2-MANAGED (Role.KT_CACHE BufferConfig). The
        # executor owns NO KT pool/BlockManager -- it reads the KT pool tensor
        # (get_kt_buffers) + KT block offsets (fill_kt_block_offsets) from the
        # V2 manager. KT lifecycle (alloc/evict/free) is automatic: KT shares
        # the K/V page allocation, so V2 frees KT sub-pages with the block.
        # `is True` short-circuits MagicMock fixtures
        # [[feedback_mock_truthy_gate_trap]].
        self._kt_v2 = (
            hasattr(kv_cache_manager, "_kt_v2_managed")
            and getattr(kv_cache_manager, "_kt_v2_managed", False) is True
        )
        if self._kt_v2:
            self.kt_tokens_per_block = kv_cache_manager._kt_tokens_per_block
            self.kt_cache_dtype = kv_cache_manager._kt_torch_dtype
            # per-seq KT block count == K/V block count (shared allocation);
            # sizes metadata.kt_cache_block_offsets.
            self.max_kt_blocks_per_seq = kv_cache_manager.max_blocks_per_seq
        else:
            self.kt_tokens_per_block = kt_tokens_per_block or next_power_of_2(
                math.ceil(page_size / page_size)
            )
            self.kt_cache_dtype = (
                torch.bfloat16 if kt_cache_dtype == "bfloat16" else torch.float8_e5m2
            )
            self.max_kt_blocks_per_seq = 0

    # ===================================================================== #
    # Pool access helpers — the KT pool is V2-managed; the executor reads   #
    # it via the V2 manager (get_kt_buffers / fill_kt).         #
    # ===================================================================== #

    def get_kt_buffers(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Path A: the KT pool is V2-managed; delegate to the V2 manager's
        get_kt_buffers (tensor view of the Role.KT_CACHE pool). Returns
        ``None`` for mocked tests / non-rocketkv."""
        if not self._kt_v2:
            return None
        return self.kv_cache_manager.get_kt_buffers(layer_idx)

    def copy_kt_block_offsets(
        self,
        request_ids: List[int],
        block_offsets: torch.Tensor,
        kt_token_counts: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Path A: KT block offsets come from the V2 manager (KT shares the
        K/V page allocation; offset = base page index * KT page-index scale).
        Delegates to fill_kt_block_offsets; V2 owns the KT lifecycle."""
        if not self._kt_v2:
            return block_offsets
        return self.kv_cache_manager.fill_kt_block_offsets(list(request_ids), block_offsets)

    # ===================================================================== #
    # HOOK 1 — on_request_init (per-request init).                           #
    # ===================================================================== #

    def on_request_init(self, request: "LlmRequest") -> None:
        """HOOK 1. Path A: KT is V2-managed, so there is no executor-side KT
        init. Chunked-prefill: record the full prompt_len per request so HOOK 2
        can detect the true last chunk (accumulated ctx tokens == prompt_len)
        and run full-prefix scoring there."""
        rid = request.py_request_id
        pl = request.prompt_len if hasattr(request, "prompt_len") else request.get_num_tokens(0)
        self._cp_prompt_len[rid] = int(pl)
        return

    # ===================================================================== #
    # HOOK 2 — on_context_attention (Stage I-a sparse-kv prediction)         #
    # ===================================================================== #

    def on_context_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Stage I-a: compute SnapKV sparse kv indices via:
          1. Split observation window Q from prefix K
          2. BMM(Q_window, K_prefix) → scores
          3. Softmax + reduce per-head → per-token importance scores
          4. Max-pool + topk → selected prefix indices
          5. Combine with window indices, flatten across batch
          6. Update KT cache pool (Stage I-a side effect)

        Returns ``(sparse_kv_indices, sparse_kv_offsets)`` for the kernel
        to consume as input-side sparse mask; ``None`` if no valid context
        sequences this iter.
        """
        if not isinstance(metadata, RocketKVTrtllmAttentionMetadata):
            return None
        if not self._kt_v2:
            return None
        num_ctx_tokens = metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            return None

        # Chunked-prefill: RocketKV metadata is chunk-local
        # (each chunk looks like a standalone prompt). Accumulate every context
        # request's qkv per (rid, layer). When >=1 request finishes its prefill
        # (accumulated tokens >= prompt_len) this forward, rebuild a full-prefix
        # scoring batch over ALL ctx requests (each at its accumulated length):
        # finished + long-enough requests are scored and compacted to budget;
        # the rest (mid-chunk / too-short) are kept whole -- the same
        # valid/invalid machinery the non-chunked path already uses. Supports
        # num_contexts > 1 (concurrent chunked requests batched in one forward).
        _saved_prompt_lens = None
        _nctx = metadata.num_contexts
        if _nctx >= 1 and len(getattr(metadata, "request_ids", []) or []) >= _nctx:
            _rids = [int(metadata.request_ids[i]) for i in range(_nctx)]
            _pls = [self._cp_prompt_len.get(r) for r in _rids]
            if all(p is not None for p in _pls):
                # chunk-local per-request offsets in this forward's ctx
                # qkv. Capture at layer 0 (pristine, set by prepare); reuse for
                # layers 1..N-1 -- layer 0's _chunked_override_metadata clobbers
                # metadata.context_cumsum_cuda in-place for the rest of this fwd.
                if layer_idx == 0 or len(self._cp_cc) != _nctx + 1:
                    self._cp_cc = metadata.context_cumsum_cuda[: _nctx + 1].tolist()
                _cc = self._cp_cc
                _qkv_comb = q[:num_ctx_tokens] if k is None else torch.cat([q, k], dim=1)
                _seen = []
                for _i in range(_nctx):
                    _key = (_rids[_i], layer_idx)
                    self._cp_k_accum.setdefault(_key, []).append(
                        _qkv_comb[_cc[_i] : _cc[_i + 1]].detach()
                    )
                    _s = self._cp_seen.get(_key, 0) + (_cc[_i + 1] - _cc[_i])
                    self._cp_seen[_key] = _s
                    _seen.append(_s)
                _finished = [_seen[_i] >= _pls[_i] for _i in range(_nctx)]
                if not any(_finished):
                    return None  # no request finished prefill this forward
                # rebuild full-prefix qkv over ALL ctx requests (accumulated)
                _full = [
                    torch.cat(self._cp_k_accum[(_rids[_i], layer_idx)], dim=0)
                    for _i in range(_nctx)
                ]
                full_qkv = torch.cat(_full, dim=0)
                _saved_prompt_lens = metadata.prompt_lens_cuda[: metadata.num_seqs].clone()
                self._chunked_override_metadata(metadata, _seen, _pls)
                q, k, num_ctx_tokens = full_qkv, None, int(sum(_seen))
                # drop finished requests' accumulation (prefill complete)
                for _i in range(_nctx):
                    if _finished[_i]:
                        self._cp_k_accum.pop((_rids[_i], layer_idx), None)
                        self._cp_seen.pop((_rids[_i], layer_idx), None)

        # Cache num_heads_per_kv on first call so non-metadata-scope
        # helpers (algorithm-body internals) can recover num_heads.
        self._cached_num_heads_per_kv = int(getattr(metadata, "num_heads_per_kv", 1) or 1)

        # Prepare qkv input
        if k is None:
            qkv_input = q[:num_ctx_tokens]
        else:
            qkv_input = torch.cat([q, k], dim=1)
        if metadata.valid_batch_size > 0:
            # Split observation window Q from prefix K
            q_window, k_context = triton_rocket_qk_split(
                qkv_input,
                metadata.prompt_lens_cuda,
                metadata.context_cumsum_cuda,
                metadata.valid_seq_indices_cuda,
                metadata.k_cu_seqlens_cuda,
                metadata.total_rocket_k_ctx_tokens,
                # num_heads / num_kv_heads / head_dim from base TrtllmAttention;
                # the attention class instance isn't directly available here, so
                # derive from kv_cache_manager.
                self._num_heads_from_kv_cache_manager(),
                self._num_kv_heads_from_kv_cache_manager(),
                self._head_dim_from_kv_cache_manager(),
                self.window_size,
                metadata.valid_batch_size,
            )

            # BMM scores
            scores = triton_bmm(
                q_window,
                k_context,
                metadata.q_cu_seqlens_cuda,
                metadata.k_cu_seqlens_cuda,
                metadata.valid_batch_size,
                causal=False,
            )

            # softmax
            scores = triton_softmax(scores, metadata.k_cu_seqlens_cuda, metadata.valid_batch_size)

            # reduce over (heads_per_kv, window)
            num_kv_heads = self._num_kv_heads_from_kv_cache_manager()
            num_heads = self._num_heads_from_kv_cache_manager()
            scores = scores.view(num_kv_heads, num_heads // num_kv_heads, self.window_size, -1).sum(
                dim=(1, 2)
            )

            # flatten variable-length batch
            scores = triton_flatten_to_batch(
                scores,
                metadata.k_cu_seqlens_cuda,
                metadata.valid_batch_size,
                metadata.max_rocket_k_ctx_len,
            )

            # max-pool smoothing
            scores = torch.nn.functional.max_pool1d(
                scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1
            )

            # indexer topk prefill
            total_tasks = metadata.valid_batch_size * num_kv_heads
            # Zero-init: torch.empty left duplicate-output garbage, so
            # force zeros.
            selected_prefix_indices = torch.zeros(
                (total_tasks, self.prompt_budget - self.window_size),
                device=qkv_input.device,
                dtype=torch.int32,
            )
            scores = scores.view(total_tasks, -1)

            row_starts = metadata.k_context_start_cuda[
                : metadata.valid_batch_size
            ].repeat_interleave(num_kv_heads)
            row_ends = metadata.k_context_lens_cuda[: metadata.valid_batch_size].repeat_interleave(
                num_kv_heads
            )
            torch.ops.trtllm.indexer_topk_prefill(
                scores,
                row_starts,
                row_ends,
                selected_prefix_indices,
                self.prompt_budget - self.window_size,
            )

            # sort selected indices
            selected_prefix_indices = torch.sort(selected_prefix_indices, dim=-1).values
        else:
            selected_prefix_indices = torch.empty(
                (0, self.prompt_budget - self.window_size),
                device=qkv_input.device,
                dtype=torch.int32,
            )

        # build sparse_kv_offsets + sparse_kv_indices
        sparse_kv_offsets = metadata.sparse_offsets_ctx_cuda[: metadata.num_contexts + 1]
        sparse_kv_indices = triton_rocket_batch_to_flatten(
            selected_prefix_indices,
            metadata.prompt_lens_cuda,
            metadata.valid_seq_indices_cuda,
            sparse_kv_offsets,
            metadata.num_contexts,
            metadata.total_sparse_ctx_indices,
            self.window_size,
            self.prompt_budget,
            self._num_kv_heads_from_kv_cache_manager(),
        )

        # Stage I-a side-effect — update KT cache pool
        kt_cache_tensor = self.get_kt_buffers(layer_idx)
        if kt_cache_tensor is not None:
            triton_rocket_update_kt_cache_ctx(
                qkv_input.contiguous(),
                kt_cache_tensor,
                metadata.kt_cache_block_offsets[: metadata.num_contexts],
                metadata.context_cumsum_cuda[: metadata.num_contexts + 1],
                sparse_kv_indices,
                sparse_kv_offsets,
                self._num_heads_from_kv_cache_manager(),
                self._num_kv_heads_from_kv_cache_manager(),
                self._head_dim_from_kv_cache_manager(),
                self.page_size,
                self.prompt_budget,
                metadata.kt_tokens_per_block,
                self.max_kt_blocks_per_seq,
            )

        # Chunked-prefill: restore the FMHA-shared prompt_lens clobbered for
        # scoring BEFORE any return, so the main attention sees the real
        # chunk-local context length.
        if _saved_prompt_lens is not None:
            metadata.prompt_lens_cuda[: metadata.num_seqs].copy_(_saved_prompt_lens)

        # reduce post-processing
        if metadata.valid_batch_size == 0:
            return None

        return sparse_kv_indices, sparse_kv_offsets

    # ===================================================================== #
    # HOOK 3 — on_context_end (Stage I-b physical-evict rewind)              #
    # ===================================================================== #

    def on_context_end(self, request: "LlmRequest", metadata: "AttentionMetadata") -> None:
        """Stage I-b SnapKV physical eviction:
            seq_len = request.get_num_tokens(0)
            rewind_len = max(seq_len - 1 - self.prompt_budget, 0)
            self.rewind_kv_cache(request, rewind_len)
        Fires once per request at prefill→decode state transition.

        Uses the manager's public ``rewind_kv_cache`` (=
        ``self.impl.rewind_kv_cache``) to shrink the cache to
        prompt_budget+1.
        """
        # Skip terminated mid-prefill
        try:
            from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

            if request.state == LlmRequestState.GENERATION_COMPLETE:
                return
        except Exception:
            pass

        seq_len = request.get_num_tokens(0) if hasattr(request, "get_num_tokens") else 0
        rewind_len = max(seq_len - 1 - self.prompt_budget, 0)
        if rewind_len <= 0:
            return

        # Now shrink the cache to prompt_budget+1 tokens via the public
        # rewind_kv_cache (= self.impl.rewind_kv_cache; frees tail
        # blocks, sets per-request sticky py_kt_target_history so V2
        # update_resources gen branch honors the truncation).
        rewind_fn = getattr(self.kv_cache_manager, "rewind_kv_cache", None)
        if rewind_fn is not None:
            rewind_fn(request, rewind_len)

        # KT rewind: not needed in Path A -- V2 frees the tail blocks
        # (incl. their KT sub-pages) when rewind_kv_cache shrinks the cache.

    def _preprocess_for_gen(self, q, k, metadata):
        """Split and reshape qkv for the generation phase."""
        num_heads = self._num_heads_from_kv_cache_manager()
        num_kv_heads = self._num_kv_heads_from_kv_cache_manager()
        head_dim = self._head_dim_from_kv_cache_manager()
        if k is None:
            qkv_input = q[metadata.num_ctx_tokens :]
            q_hidden_size = num_heads * head_dim
            k_hidden_size = num_kv_heads * head_dim
            q = qkv_input[:, :q_hidden_size]
            k = qkv_input[:, q_hidden_size : q_hidden_size + k_hidden_size]
        else:
            q = q[metadata.num_ctx_tokens :]
            k = k[metadata.num_ctx_tokens :]
        q = q.view(-1, num_kv_heads, num_heads // num_kv_heads, head_dim)
        return q, k

    @torch.compile(dynamic=True, disable=True)
    def _topr_filter(self, q):
        """Apply the top-r channel filter to the query."""
        i1 = torch.topk(q.abs().sum(dim=2, keepdim=True), self.topr, dim=-1).indices
        q_mask = torch.zeros_like(q)
        q_mask.scatter_(-1, i1.expand_as(q[..., : self.topr]), 1)
        return q * q_mask

    def on_generation_attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        attn_scores: Optional[torch.Tensor],
        metadata: "AttentionMetadata",
    ) -> Optional[SparseAttentionIndices]:
        """Stage II HSA: per-decode-step, per-attention-layer, build sparse
        attention mask over the (already-shrunk by Stage I-b) cache using
        KT page summaries as a coarse-grained lookup.
        """
        if not isinstance(metadata, RocketKVTrtllmAttentionMetadata):
            return None
        if not self._kt_v2:
            return None
        if metadata.num_generations == 0:
            return None
        self._cached_num_heads_per_kv = int(getattr(metadata, "num_heads_per_kv", 1) or 1)

        q, k = self._preprocess_for_gen(q, k, metadata)

        head_dim = self._head_dim_from_kv_cache_manager()
        if self.topr < head_dim:
            q = self._topr_filter(q)

        kt_cache_tensor = self.get_kt_buffers(layer_idx)
        if kt_cache_tensor is None:
            return None

        num_kv_heads = self._num_kv_heads_from_kv_cache_manager()
        num_heads = self._num_heads_from_kv_cache_manager()

        # update KT cache for new gen step
        triton_rocket_update_kt_cache_gen(
            k,
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[metadata.num_contexts :],
            metadata.kv_lens_cuda_runtime[metadata.num_contexts :],
            metadata.page_size,
            metadata.kt_tokens_per_block,
            self.max_kt_blocks_per_seq,
            num_kv_heads,
            head_dim,
        )

        # BMM Q · KT
        scores = triton_rocket_paged_kt_cache_bmm(
            q,
            kt_cache_tensor,
            metadata.kt_cache_block_offsets[metadata.num_contexts :],
            metadata.kv_lens_cuda_runtime[metadata.num_contexts :],
            metadata.cum_kt_lens_cuda,
            metadata.page_size,
            metadata.kt_tokens_per_block,
            self.max_kt_blocks_per_seq,
            metadata.total_kt_tokens,
        )

        scores = triton_softmax(scores, metadata.cum_kt_lens_cuda, metadata.num_generations)

        scores = triton_rocket_reduce_scores(
            scores,
            metadata.cum_kt_lens_cuda,
            metadata.num_generations,
            num_kv_heads,
            num_heads // num_kv_heads,
        )

        sparse_attn_offsets = metadata.sparse_offsets_gen_cuda[: metadata.num_generations + 1]
        selected_indices = triton_topk(
            scores,
            metadata.cum_kt_lens_cuda,
            sparse_attn_offsets,
            metadata.total_sparse_gen_indices,
            metadata.topk,
        )

        return selected_indices, sparse_attn_offsets

    # ===================================================================== #
    # HOOK 6 — on_request_finish (per-request cleanup)                       #
    # ===================================================================== #

    def on_request_finish(self, request: "LlmRequest") -> None:
        """Per-request cleanup at finish.

        Path A: V2 frees KT sub-pages with the block (shared allocation),
        so there is no executor-side KT free; this only drops the
        chunked-prefill accumulation state for the request.
        """
        # Path A: V2 frees KT sub-pages with the block (shared allocation).
        # Chunked-prefill: drop this request accumulation state (the last
        # chunk pops it; safety net for early-finished requests).
        rid = request.py_request_id
        for _d in (self._cp_k_accum, self._cp_seen):
            for _k in [k for k in _d if k[0] == rid]:
                _d.pop(_k, None)
        self._cp_prompt_len.pop(rid, None)
        return

    # ===================================================================== #
    # Helpers — proxy attention layer params via kv_cache_manager.            #
    # ===================================================================== #

    def _num_kv_heads_from_kv_cache_manager(self) -> int:
        # ``KVCacheManager`` exposes a scalar ``self.num_kv_heads``.
        # ``KVCacheManagerV2`` exposes a list ``self.num_kv_heads_per_layer``
        # (uniform across layers for Llama-class models). Fall back to the
        # per-layer list when the scalar is absent.
        v1 = getattr(self.kv_cache_manager, "num_kv_heads", None)
        if v1:
            return int(v1)
        per_layer = getattr(self.kv_cache_manager, "num_kv_heads_per_layer", None)
        if per_layer:
            return int(per_layer[0])
        return 0

    def _head_dim_from_kv_cache_manager(self) -> int:
        v1 = getattr(self.kv_cache_manager, "head_dim", None)
        if v1:
            return int(v1)
        per_layer = getattr(self.kv_cache_manager, "head_dim_per_layer", None)
        if per_layer:
            return int(per_layer[0])
        return 0

    # Backward-compat alias for sites that don't have metadata in scope.
    def _num_heads_from_kv_cache_manager(self) -> int:
        # ``KVCacheManager`` scalar shortcut; ``KVCacheManagerV2`` fallback
        # uses cached num_heads_per_kv set at the first HOOK 2/4 call.
        v1 = getattr(self.kv_cache_manager, "num_heads", None)
        if v1:
            return int(v1)
        return self._num_kv_heads_from_kv_cache_manager() * int(
            getattr(self, "_cached_num_heads_per_kv", 1)
        )

    def _num_heads_from_metadata(self, metadata) -> int:
        """num_heads = num_kv_heads * num_heads_per_kv (read from metadata)."""
        nhpkv = getattr(metadata, "num_heads_per_kv", 1) or 1
        return self._num_kv_heads_from_kv_cache_manager() * int(nhpkv)

    def _chunked_override_metadata(self, metadata, seen, pls) -> None:
        """Chunked-prefill scoring: override metadata ctx tensors in-place to a
        full-prefix batch built from each context request's accumulated length
        ``seen[i]`` (cached so far) vs full prompt ``pls[i]``. A request is
        'valid' (scored + compacted to budget) iff it finished prefill
        (seen >= pl) AND is long enough (pl >= budget); the rest are kept whole.
        Mirrors prepare()'s context-phase setup for a batch of ``len(seen)``
        requests. Reduces to the single-request case when len(seen) == 1."""
        w = self.window_size
        B = self.prompt_budget
        n = len(seen)
        dev = metadata.prompt_lens_cuda.device
        L = torch.tensor(seen, dtype=torch.int32)
        P = torch.tensor(pls, dtype=torch.int32)
        valid_mask = (L >= P) & (P >= B)
        valid_idx = torch.where(valid_mask)[0].to(torch.int32)
        invalid_idx = torch.where(~valid_mask)[0].to(torch.int32)
        vbs = int(valid_idx.numel())
        # context_cumsum over cached lengths; prompt_lens := cached lengths
        cum = torch.zeros(n + 1, dtype=torch.int32)
        cum[1:] = torch.cumsum(L, 0)
        metadata.context_cumsum_cuda[: n + 1].copy_(cum.to(dev), non_blocking=True)
        metadata.prompt_lens_cuda[:n].copy_(L.to(dev), non_blocking=True)
        # sparse counts: valid -> budget; invalid -> keep all cached
        sc = torch.zeros(n, dtype=torch.int32)
        sc[valid_idx.long()] = B
        sc[invalid_idx.long()] = L[invalid_idx.long()]
        soff = torch.zeros(n + 1, dtype=torch.int32)
        soff[1:] = torch.cumsum(sc, 0)
        metadata.sparse_offsets_ctx_cuda[: n + 1].copy_(soff.to(dev), non_blocking=True)
        metadata.total_sparse_ctx_indices = int(soff[n].item())
        metadata.valid_batch_size = vbs
        if vbs:
            metadata.valid_seq_indices_cuda[:vbs].copy_(valid_idx.to(dev), non_blocking=True)
            kcl = L[valid_idx.long()] - w
            metadata.k_context_lens_cuda[:vbs].copy_(kcl.to(dev), non_blocking=True)
            metadata.k_context_start_cuda[:vbs].copy_(
                torch.zeros(vbs, dtype=torch.int32).to(dev), non_blocking=True
            )
            qcu = torch.arange(vbs + 1, dtype=torch.int32) * w
            metadata.q_cu_seqlens_cuda[: vbs + 1].copy_(qcu.to(dev), non_blocking=True)
            kcu = torch.zeros(vbs + 1, dtype=torch.int32)
            kcu[1:] = torch.cumsum(kcl, 0)
            metadata.k_cu_seqlens_cuda[: vbs + 1].copy_(kcu.to(dev), non_blocking=True)
            metadata.max_rocket_k_ctx_len = int(kcl.max().item())
            metadata.total_rocket_k_ctx_tokens = int(kcu[vbs].item())
        else:
            metadata.max_rocket_k_ctx_len = 0
            metadata.total_rocket_k_ctx_tokens = 0


_KT_ROLE = _DataRole("kt_cache")


class RocketKVCacheManagerV2(KVCacheManagerV2):
    """RocketKV KT pool registered as a ``KVCacheManagerV2`` sub-page pool
    (Path A). Overrides ONLY the config-building methods to add the KT pool
    -- the manager core is untouched. KT shares the K/V page
    (block index) and is freed automatically with the block. ``tokens_per_block``
    and ``sparse_attn_config`` arrive as keywords from _create_kv_cache_manager."""

    def __init__(self, *args, **kwargs):
        sparse = kwargs.pop("sparse_attn_config")
        tpb = kwargs["tokens_per_block"]
        self._kt_tokens_per_block = next_power_of_2(math.ceil(tpb / sparse.page_size))
        assert tpb % self._kt_tokens_per_block == 0, (
            f"kt_tokens_per_block={self._kt_tokens_per_block} must divide tokens_per_block={tpb}"
        )
        self._kt_override = tpb // self._kt_tokens_per_block
        _dt = getattr(sparse, "kt_cache_dtype", "bfloat16")
        self._kt_torch_dtype = torch.bfloat16 if _dt == "bfloat16" else torch.float8_e5m2
        self._kt_dtype_bytes = 2 if self._kt_torch_dtype == torch.bfloat16 else 1
        self._kt_v2_managed = True
        super().__init__(*args, **kwargs)
        self.kt_index_scale = self.impl.get_page_index_scale(
            self.impl.layer_grouping[0][0], _KT_ROLE
        )

    def _build_cache_config(self, *args, **kwargs):
        cfg = super()._build_cache_config(*args, **kwargs)
        for layer in cfg.layers:
            nkv = self.num_kv_heads_per_layer[layer.layer_id]
            hd = self.head_dim_per_layer[layer.layer_id]
            layer.buffers.append(
                BufferConfig(
                    role=_KT_ROLE,
                    size=nkv * hd * 2 * self._kt_dtype_bytes,
                    tokens_per_block_override=self._kt_override,
                )
            )
        return cfg

    def get_cache_bytes_per_token(self) -> int:
        kt = 0
        for L in range(self.num_local_layers):
            bb = (
                self._kt_tokens_per_block
                * self.num_kv_heads_per_layer[L]
                * self.head_dim_per_layer[L]
                * 2
                * self._kt_dtype_bytes
            )
            kt += -(-bb // self.tokens_per_block)
        return super().get_cache_bytes_per_token() + kt

    def get_kt_buffers(self, layer_idx: int) -> Optional[torch.Tensor]:
        layer_offset = self.layer_offsets[layer_idx]
        addr = self.impl.get_mem_pool_base_address(layer_offset, _KT_ROLE)
        shape = [
            self.impl.get_page_index_upper_bound(layer_offset, _KT_ROLE),
            self._kt_tokens_per_block,
            self.num_kv_heads_per_layer[layer_offset],
            self.head_dim_per_layer[layer_offset] * 2,
        ]
        dt = self._kt_torch_dtype
        # TensorWrapper cannot infer fp8 dtypes (esp. e5m2): wrap as int8 then view.
        if dt in (torch.float8_e5m2, torch.float8_e4m3fn):
            return convert_to_torch_tensor(TensorWrapper(addr, torch.int8, shape)).view(dt)
        return convert_to_torch_tensor(TensorWrapper(addr, dt, shape))

    def fill_kt_block_offsets(self, request_ids: List[int], out: torch.Tensor) -> torch.Tensor:
        scale = int(self.kt_index_scale)
        for i, req_id in enumerate(request_ids):
            kvc = self.kv_cache_map.get(req_id)
            if kvc is None:
                continue
            idx = torch.as_tensor(kvc.get_base_page_indices(0))
            kt = torch.where(idx != _BAD_PAGE, idx * scale, torch.zeros_like(idx))
            n = min(int(kt.numel()), out.shape[1])
            out[i, :n] = kt[:n].to(out.dtype)
        return out
