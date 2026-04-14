"""Single-process test for KVCacheManager (V1/V2) + KvCacheTransceiverV2.

Uses threading + ThreadSafeDistributed to create KvCacheTransceiverV2 instances
in a single process. Validates KV cache transfer correctness across different
TP/PP/DP/MLA/sliding-window configurations for both V1 and V2 cache managers.
"""

import os
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
import tensorrt_llm.tensorrt_llm_transfer_agent_binding  # noqa: F401
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.utils import get_global_layer_ids
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, KVCacheManagerV2
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings import LayerType as LayerTypeCpp
from tensorrt_llm.bindings import ModelConfig as ModelConfigCpp
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType

# Reduce NIXL threads for unit test: default 8 threads per agent causes heavy
# contention when creating multiple agents on a single GPU in the same process.
os.environ.setdefault("TRTLLM_NIXL_NUM_THREADS", "0")
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 4
NUM_KV_HEADS = 4  # 1 for MLA
HEAD_DIM = 128
TOKENS_PER_BLOCK = 8
MAX_SEQ_LEN = 256
MAX_BATCH_SIZE = 4
VOCAB_SIZE = 32000
REQUEST_LENGTHS = [30, 60, 80]


# ---------------------------------------------------------------------------
# PP layer distribution helpers (mirrors C++ getLayerNumPPRank)
# ---------------------------------------------------------------------------
def _get_layers_per_pp(num_layers: int, pp_size: int) -> List[int]:
    """Return a list of layer counts per PP rank.

    When num_layers is not evenly divisible by pp_size, the first
    (num_layers % pp_size) ranks get one extra layer.
    Matches Mapping.pp_layers / torch.tensor_split behaviour.
    """
    base = num_layers // pp_size
    extra = num_layers % pp_size
    return [base + (1 if r < extra else 0) for r in range(pp_size)]


def _pp_rank_of_layer(layer_idx: int, num_layers: int, pp_size: int) -> int:
    """Return the PP rank that owns *layer_idx*."""
    cumulative = 0
    for pp_rank, count in enumerate(_get_layers_per_pp(num_layers, pp_size)):
        cumulative += count
        if layer_idx < cumulative:
            return pp_rank
    raise ValueError(f"layer_idx {layer_idx} out of range for num_layers={num_layers}")


def _pp_layer_start(pp_rank: int, num_layers: int, pp_size: int) -> int:
    """Return the global layer index of the first layer on *pp_rank*."""
    start = 0
    for r, count in enumerate(_get_layers_per_pp(num_layers, pp_size)):
        if r == pp_rank:
            return start
        start += count
    raise ValueError(f"pp_rank {pp_rank} out of range for pp_size={pp_size}")


@dataclass
class KvCacheConfigV2:
    """KvCacheConfig wrapper with max_util_for_resume for KVCacheManagerV2."""

    max_tokens: Optional[int] = None
    enable_block_reuse: bool = False
    max_attention_window: Optional[List[int]] = None
    sink_token_length: Optional[int] = None
    free_gpu_memory_fraction: Optional[float] = None
    host_cache_size: Optional[int] = None
    onboard_blocks: bool = True
    cross_kv_cache_fraction: Optional[float] = None
    secondary_offload_min_priority: Optional[int] = None
    event_buffer_max_size: int = 0
    max_gpu_total_bytes: Optional[int] = None
    enable_partial_reuse: bool = False
    copy_on_partial_reuse: bool = False
    dtype: str = "auto"
    max_util_for_resume: float = 0.95


# ---------------------------------------------------------------------------
# ThreadSafeDistributed: threading.Barrier-based Distributed mock
# ---------------------------------------------------------------------------
class ThreadSafeDistributed:
    """Distributed mock using threading.Barrier for single-process multi-rank testing."""

    def __init__(
        self,
        local_rank: int,
        world_size: int,
        tp_size: int,
        pp_size: int,
        tp_rank: int,
        pp_rank: int,
        shared: dict,
    ):
        self.rank = local_rank
        self._world_size = world_size
        self._tp_size = tp_size
        self._pp_size = pp_size
        self._tp_rank = tp_rank
        self._pp_rank = pp_rank
        self._s = shared
        self._bcast_idx = 0
        self._ag_idx = 0
        self._pp_ag_idx = 0
        self._tp_ag_idx = 0

    @property
    def tp_size(self):
        return self._tp_size

    @property
    def pp_size(self):
        return self._pp_size

    @property
    def world_size(self):
        return self._world_size

    def broadcast(self, obj, root=0):
        idx = self._bcast_idx
        self._bcast_idx += 1
        key = f"bcast_{idx}"
        if self.rank == root:
            self._s[key] = obj
        self._s["barrier"].wait()
        result = self._s[key]
        self._s["barrier"].wait()
        return result

    def allgather(self, obj):
        idx = self._ag_idx
        self._ag_idx += 1
        key = f"ag_{idx}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._world_size
            self._s[key][self.rank] = obj
        self._s["barrier"].wait()
        result = list(self._s[key])
        self._s["barrier"].wait()
        return result

    def pp_allgather(self, obj):
        idx = self._pp_ag_idx
        self._pp_ag_idx += 1
        key = f"pp_ag_{idx}_tp{self._tp_rank}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._pp_size
            self._s[key][self._pp_rank] = obj
        self._s["barrier"].wait()
        result = list(self._s[key])
        self._s["barrier"].wait()
        return result

    def tp_allgather(self, obj):
        idx = self._tp_ag_idx
        self._tp_ag_idx += 1
        key = f"tp_ag_{idx}_pp{self._pp_rank}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._tp_size
            self._s[key][self._tp_rank] = obj
        self._s["barrier"].wait()
        result = list(self._s[key])
        self._s["barrier"].wait()
        return result


# ---------------------------------------------------------------------------
# Threading helpers
# ---------------------------------------------------------------------------
def run_concurrent(items, fn):
    """Run fn(item) for each item concurrently in threads and propagate errors."""
    errors = [None] * len(items)
    results = [None] * len(items)

    def _worker(idx, item):
        try:
            results[idx] = fn(item)
        except Exception as e:
            errors[idx] = e

    threads = [threading.Thread(target=_worker, args=(i, item)) for i, item in enumerate(items)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, err in enumerate(errors):
        if err is not None:
            raise err
    return results


# ---------------------------------------------------------------------------
# Cache Manager Creation
# ---------------------------------------------------------------------------
def _get_ctx_seed(
    tp_rank: int, pp_rank: int, tp_size: int, num_kv_heads: int, seed_base: int = 1000
) -> int:
    """Compute random seed for ctx pool initialization.

    When num_kv_heads >= tp_size, heads are split across TP ranks -> unique seed per rank.
    When num_kv_heads < tp_size, heads are replicated -> same seed for all TP ranks in a PP group.
    """
    if num_kv_heads >= tp_size:
        return seed_base + pp_rank * tp_size + tp_rank
    else:
        return seed_base + pp_rank


def _create_cache_manager(
    mapping: Mapping,
    is_mla: bool,
    use_v2: bool,
    max_attention_window_vec: Optional[List[int]] = None,
    num_layers: int = NUM_LAYERS,
    max_batch_size: int = MAX_BATCH_SIZE,
) -> "KVCacheManager | KVCacheManagerV2":
    """Create a KVCacheManager (V1) or KVCacheManagerV2 for the given mapping."""
    num_kv_heads = 1 if is_mla else NUM_KV_HEADS
    cache_type = CacheTypeCpp.SELFKONLY if is_mla else CacheTypeCpp.SELF

    if max_attention_window_vec is None:
        max_attention_window_vec = [MAX_SEQ_LEN]

    if use_v2:
        return KVCacheManagerV2(
            KvCacheConfigV2(
                max_tokens=2048,
                enable_block_reuse=False,
                max_attention_window=max_attention_window_vec,
            ),
            cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=HEAD_DIM,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=DataType.FLOAT,
            vocab_size=VOCAB_SIZE,
        )
    else:
        is_vswa = max_attention_window_vec and len(set(max_attention_window_vec)) > 1
        model_config = None
        if is_vswa:
            model_config = ModelConfigCpp(
                vocab_size=VOCAB_SIZE,
                num_layers=num_layers,
                num_attention_layers=num_layers,
                num_rnn_layers=0,
                num_heads=num_kv_heads,
                hidden_size=num_kv_heads * HEAD_DIM,
                data_type=DataType.FLOAT,
            )
            model_config.layer_types = [LayerTypeCpp.ATTENTION] * num_layers
            model_config.set_num_kv_heads(num_kv_heads)
            model_config.size_per_head = HEAD_DIM
            model_config.tokens_per_block = TOKENS_PER_BLOCK

        if is_vswa:
            kv_cache_cfg = KvCacheConfig(
                max_tokens=2048,
                enable_block_reuse=False,
                max_attention_window=max_attention_window_vec,
            )
        else:
            kv_cache_cfg = trtllm.KvCacheConfig(
                max_tokens=2048,
                enable_block_reuse=False,
                max_attention_window=max_attention_window_vec,
            )
        return KVCacheManager(
            kv_cache_cfg,
            cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=HEAD_DIM,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=DataType.FLOAT,
            model_config=model_config,
        )


def _create_managers_for_instance(
    tp: int,
    pp: int,
    enable_dp: bool,
    is_mla: bool,
    use_v2: bool,
    max_attention_window_vec: Optional[List[int]] = None,
    num_layers: int = NUM_LAYERS,
    max_batch_size: int = MAX_BATCH_SIZE,
) -> List:
    """Create cache managers for all ranks in an instance."""
    managers = []
    for pp_rank in range(pp):
        for tp_rank in range(tp):
            rank = pp_rank * tp + tp_rank
            mapping = Mapping(
                world_size=tp * pp,
                rank=rank,
                tp_size=tp,
                pp_size=pp,
                enable_attention_dp=enable_dp,
            )
            managers.append(
                _create_cache_manager(
                    mapping, is_mla, use_v2, max_attention_window_vec, num_layers, max_batch_size
                )
            )
    return managers


# ---------------------------------------------------------------------------
# Pool Data Initialization
# ---------------------------------------------------------------------------
def _init_pool_data_v1(
    managers: List[KVCacheManager],
    tp: int,
    is_mla: bool,
    fill_random: bool = True,
    seed_base: int = 1000,
):
    """Initialize pool data for V1 managers."""
    num_kv_heads = 1 if is_mla else NUM_KV_HEADS
    for rank, mgr in enumerate(managers):
        pp_rank = rank // tp
        tp_rank = rank % tp
        pool_tensor = mgr.get_unique_primary_pool()
        if fill_random:
            seed = _get_ctx_seed(tp_rank, pp_rank, tp, num_kv_heads, seed_base)
            generator = torch.Generator(device=pool_tensor.device).manual_seed(seed)
            random_values = torch.rand(
                pool_tensor.shape,
                dtype=pool_tensor.dtype,
                device=pool_tensor.device,
                generator=generator,
            )
            pool_tensor.copy_(random_values)
        else:
            pool_tensor.zero_()


def _init_pool_data_v2(
    managers: List[KVCacheManagerV2],
    tp: int,
    is_mla: bool,
    fill_random: bool = True,
    seed_base: int = 1000,
):
    """Initialize pool data for V2 managers."""
    num_kv_heads = 1 if is_mla else NUM_KV_HEADS
    for rank, mgr in enumerate(managers):
        pp_rank = rank // tp
        tp_rank = rank % tp
        page_table = KVRegionExtractorV1(mgr).page_table

        unique_pools: Dict[int, int] = {}
        for pg in page_table.pool_groups:
            for pool_desc in pg.pools:
                key = pool_desc.base_address
                pool_bytes = pool_desc.slot_bytes * pool_desc.num_slots
                if key not in unique_pools or pool_bytes > unique_pools[key]:
                    unique_pools[key] = pool_bytes

        element_bytes = get_size_in_bytes(1, mgr.dtype)
        for pool_base_ptr, pool_size in unique_pools.items():
            pool_size_elements = pool_size // element_bytes
            pool_tensor = convert_to_torch_tensor(
                TensorWrapper(pool_base_ptr, mgr.dtype, [pool_size_elements])
            )
            if fill_random:
                seed = _get_ctx_seed(tp_rank, pp_rank, tp, num_kv_heads, seed_base)
                generator = torch.Generator(device=pool_tensor.device).manual_seed(seed)
                random_values = torch.rand(
                    pool_tensor.shape,
                    dtype=pool_tensor.dtype,
                    device=pool_tensor.device,
                    generator=generator,
                )
                pool_tensor.copy_(random_values)
            else:
                pool_tensor.zero_()


def _init_pool_data(managers, tp, is_mla, use_v2, fill_random=True, seed_base=1000):
    """Initialize pool data for managers (dispatches to V1/V2)."""
    if use_v2:
        _init_pool_data_v2(managers, tp, is_mla, fill_random, seed_base)
    else:
        _init_pool_data_v1(managers, tp, is_mla, fill_random, seed_base)


# ---------------------------------------------------------------------------
# Add sequence to manager
# ---------------------------------------------------------------------------
def _add_sequence(mgr, request_id: int, prompt_len: int, use_v2: bool):
    """Add a sequence to the cache manager. Returns kv_cache for V2 (needed for cleanup)."""
    if use_v2:
        kv_cache = mgr._create_kv_cache(request_id, None, None)
        success = kv_cache.resume(torch.cuda.current_stream().cuda_stream)
        assert success, f"Failed to resume kv_cache for request {request_id}"
        kv_cache.resize(prompt_len)
        return kv_cache
    else:
        # V1: create a dummy LlmRequest for add_sequence
        sampling_params = SamplingParams()
        dummy_request = LlmRequest(
            request_id=request_id,
            max_new_tokens=1,
            input_tokens=list(range(prompt_len)),
            sampling_config=tensorrt_llm.bindings.SamplingConfig(
                sampling_params._get_sampling_config()
            ),
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
        )
        mgr.impl.add_sequence(request_id, prompt_len, 1, dummy_request)
        return None


# ---------------------------------------------------------------------------
# Transceiver creation (threaded)
# ---------------------------------------------------------------------------
def _create_transceiver_in_thread(
    rank, mapping, cache_manager, dist_mock, attention_type, config, results, errors
):
    """Thread target: create one KvCacheTransceiverV2."""
    try:
        tc = KvCacheTransceiverV2(
            mapping=mapping,
            dist=dist_mock,
            kv_cache_manager=cache_manager,
            cache_transceiver_config=config,
        )
        results[rank] = tc
    except Exception as e:
        errors[rank] = e


def create_instance_transceivers(
    tp: int,
    pp: int,
    enable_dp: bool,
    cache_managers: List,
    config: CacheTransceiverConfig,
    is_mla: bool,
) -> List[KvCacheTransceiverV2]:
    """Create KvCacheTransceiverV2 for all ranks via threaded init."""
    world_size = tp * pp
    shared = {"barrier": threading.Barrier(world_size), "lock": threading.Lock()}
    results = [None] * world_size
    errors = [None] * world_size
    threads = []

    attention_type = AttentionTypeCpp.MLA if is_mla else AttentionTypeCpp.DEFAULT

    for rank in range(world_size):
        pp_rank = rank // tp
        tp_rank = rank % tp
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=tp,
            pp_size=pp,
            enable_attention_dp=enable_dp,
        )
        dist_mock = ThreadSafeDistributed(rank, world_size, tp, pp, tp_rank, pp_rank, shared)
        t = threading.Thread(
            target=_create_transceiver_in_thread,
            args=(
                rank,
                mapping,
                cache_managers[rank],
                dist_mock,
                attention_type,
                config,
                results,
                errors,
            ),
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for rank, err in enumerate(errors):
        if err is not None:
            raise err

    return results


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def _get_block_data_for_layer(mgr, request_id, layer_idx, use_v2, expected_valid=None):
    """Get block data for a specific layer and request in HND layout.

    Returns tensor of shape [num_blocks, kv_factor, num_kv_heads_per_rank, tokens_per_block, head_dim].
    The pool's physical layout is HND (heads vary slowest within each kv_factor),
    so we must use kv_layout='HND' to get a correct view.  Using the default NHD
    layout would reinterpret HND memory as NHD, which breaks cross-TP verification
    because the formatter concat/split operates on HND-ordered flat data.

    If *expected_valid* is given, keeps only the last *expected_valid* blocks
    (used for sliding-window: only the window-tail blocks were transferred).
    This works regardless of whether eviction has happened:
      - Pre-eviction: all blocks present, keep last N.
      - Post-eviction: stale blocks already removed (or -1), keep last N is no-op.
    """
    block_indices = mgr.get_batch_cache_indices([request_id], layer_idx)[0]
    valid_indices = [idx for idx in block_indices if idx >= 0]
    if not valid_indices:
        return None
    if expected_valid is not None and len(valid_indices) > expected_valid:
        valid_indices = valid_indices[-expected_valid:]
    layer_buffer = mgr.get_buffers(layer_idx, kv_layout="HND")
    return layer_buffer[valid_indices]


def _gather_full_layer_data(
    managers: List,
    tp: int,
    pp: int,
    request_id: int,
    layer_idx: int,
    use_v2: bool,
    num_kv_heads: int,
    expected_valid: Optional[int] = None,
    enable_dp: bool = False,
    req_idx: int = 0,
    num_layers: int = NUM_LAYERS,
):
    """Gather the full (unsharded) KV data for a layer by concatenating across TP ranks.

    Returns tensor of shape [num_blocks, kv_factor, num_kv_heads, tokens_per_block, head_dim]
    (HND layout) where num_kv_heads is the total (not per-rank) count.
    For replicated heads (MLA), returns data from a single TP rank (all are identical).
    """
    pp_rank = _pp_rank_of_layer(layer_idx, num_layers, pp)
    is_replicated = num_kv_heads < tp

    if enable_dp:
        # Only one TP rank handled this request
        tp_rank = req_idx % tp
        rank = pp_rank * tp + tp_rank
        return _get_block_data_for_layer(
            managers[rank], request_id, layer_idx, use_v2, expected_valid
        )

    if is_replicated:
        # All TP ranks have identical data; take from tp_rank=0
        rank = pp_rank * tp + 0
        return _get_block_data_for_layer(
            managers[rank], request_id, layer_idx, use_v2, expected_valid
        )

    # Gather from all TP ranks and concat along kv_heads dim.
    # In HND layout the shape is [blocks, kv_factor, heads, tpb, dim],
    # so heads are at dim=2.
    tp_data = []
    for tp_rank in range(tp):
        rank = pp_rank * tp + tp_rank
        data = _get_block_data_for_layer(
            managers[rank], request_id, layer_idx, use_v2, expected_valid
        )
        if data is not None:
            tp_data.append(data)

    if not tp_data:
        return None
    return torch.cat(tp_data, dim=2)


def _get_layer_to_window_size(
    managers: List,
    tp: int,
    pp: int,
    use_v2: bool,
    num_layers: int = NUM_LAYERS,
) -> Dict[int, Optional[int]]:
    """Build a mapping from global layer_id to its sliding_window_size.

    For V2 managers, extracts from page_table.layer_group_metas.
    For V1 managers, uses _get_window_size_to_layers().

    Aggregates from all PP ranks since each rank only handles a subset of layers.
    """
    layer_to_window: Dict[int, Optional[int]] = {}

    # Get one manager per PP rank (tp_rank=0 for each pp_rank)
    for pp_rank in range(pp):
        manager = managers[pp_rank * tp]

        if use_v2:
            page_table = KVRegionExtractorV1(manager).page_table
            for lg in page_table.layer_groups:
                for layer_id in get_global_layer_ids(lg):
                    layer_to_window[layer_id] = lg.sliding_window_size
        else:
            window_to_layers = manager._get_window_size_to_layers()
            # For V1, local layer ids need to be converted to global layer ids
            layer_start = _pp_layer_start(pp_rank, num_layers, pp)
            for window_size, local_layers in window_to_layers.items():
                for local_layer_id in local_layers:
                    global_layer_id = layer_start + local_layer_id
                    layer_to_window[global_layer_id] = window_size

    return layer_to_window


def verify_all_requests(
    request_lengths: List[int],
    ctx_managers: List,
    gen_managers: List,
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool,
    gen_enable_dp: bool,
    ctx_request_ids: List[int],
    gen_request_ids: List[int],
    is_mla: bool,
    use_v2: bool,
    max_attention_window_vec: Optional[List[int]] = None,
    num_layers: int = NUM_LAYERS,
):
    """Verify transferred cache data for all requests.

    For each request and each global layer, reconstruct the full (unsharded)
    KV data from both ctx and gen sides by gathering across TP ranks, then
    compare.  This correctly handles asymmetric TP/PP configurations.
    """
    num_kv_heads = 1 if is_mla else NUM_KV_HEADS

    # Build layer_to_window_size mapping from ctx_managers
    # Aggregate from all PP ranks since each rank only handles a subset of layers
    layer_to_window = _get_layer_to_window_size(ctx_managers, ctx_tp, ctx_pp, use_v2, num_layers)

    for req_idx, req_len in enumerate(request_lengths):
        ctx_rid = ctx_request_ids[req_idx]
        gen_rid = gen_request_ids[req_idx]

        for layer_idx in range(num_layers):
            # Compute expected_valid: the number of non-stale blocks that
            # were actually transferred, using the same eviction formula as
            # _create_kv_slice.  Only compare these blocks in verification.
            expected_valid = None
            win = layer_to_window.get(layer_idx)
            if win is not None and win < MAX_SEQ_LEN:
                total_blocks = (req_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
                stale_end = max(0, (req_len + 1 - win) // TOKENS_PER_BLOCK)
                expected_valid = total_blocks - stale_end

            ctx_full = _gather_full_layer_data(
                ctx_managers,
                ctx_tp,
                ctx_pp,
                ctx_rid,
                layer_idx,
                use_v2,
                num_kv_heads,
                expected_valid,
                ctx_enable_dp,
                req_idx,
                num_layers,
            )
            gen_full = _gather_full_layer_data(
                gen_managers,
                gen_tp,
                gen_pp,
                gen_rid,
                layer_idx,
                use_v2,
                num_kv_heads,
                expected_valid,
                gen_enable_dp,
                req_idx,
                num_layers,
            )

            if ctx_full is None or gen_full is None:
                continue

            assert ctx_full.shape == gen_full.shape, (
                f"Shape mismatch at req={req_idx} layer={layer_idx}: "
                f"ctx={ctx_full.shape} gen={gen_full.shape}"
            )
            torch.testing.assert_close(
                gen_full,
                ctx_full,
                rtol=0,
                atol=0,
                msg=lambda m: (f"Data mismatch at req={req_idx} layer={layer_idx}: {m}"),
            )


# ---------------------------------------------------------------------------
# Main test orchestrator
# ---------------------------------------------------------------------------
def run_transfer_test(
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool,
    gen_enable_dp: bool,
    is_mla: bool,
    use_v2: bool,
    max_attention_window_vec: Optional[List[int]] = None,
    num_layers: int = NUM_LAYERS,
    request_lengths: Optional[List[int]] = None,
):
    """Run a full KV transfer test using KvCacheTransceiverV2."""
    if request_lengths is None:
        request_lengths = REQUEST_LENGTHS
    max_batch_size = max(MAX_BATCH_SIZE, len(request_lengths))
    ctx_world = ctx_tp * ctx_pp
    gen_world = gen_tp * gen_pp

    # 1. Create cache managers
    ctx_managers = _create_managers_for_instance(
        ctx_tp,
        ctx_pp,
        ctx_enable_dp,
        is_mla,
        use_v2,
        max_attention_window_vec,
        num_layers,
        max_batch_size,
    )
    gen_managers = _create_managers_for_instance(
        gen_tp,
        gen_pp,
        gen_enable_dp,
        is_mla,
        use_v2,
        max_attention_window_vec,
        num_layers,
        max_batch_size,
    )

    # 2. Initialize data: random for ctx, zeros for gen
    _init_pool_data(ctx_managers, ctx_tp, is_mla, use_v2, fill_random=True, seed_base=1000)
    _init_pool_data(gen_managers, gen_tp, is_mla, use_v2, fill_random=False)

    # 3. Create KvCacheTransceiverV2 instances (threaded init)
    config = CacheTransceiverConfig(
        backend="NIXL",
        transceiver_runtime="PYTHON",
        max_tokens_in_buffer=512,
    )
    ctx_tcs = create_instance_transceivers(
        ctx_tp, ctx_pp, ctx_enable_dp, ctx_managers, config, is_mla
    )
    gen_tcs = create_instance_transceivers(
        gen_tp, gen_pp, gen_enable_dp, gen_managers, config, is_mla
    )

    ctx_info_endpoint = ctx_tcs[0]._context_info_endpoint

    # ===== 4. Create requests =====
    ctx_handle_map: Dict[int, List] = {r: [] for r in range(ctx_world)}
    gen_handle_map: Dict[int, List] = {r: [] for r in range(gen_world)}
    ctx_request_ids: List[int] = []
    gen_request_ids: List[int] = []
    ctx_kv_caches: Dict[int, List] = {r: [] for r in range(ctx_world)}
    gen_kv_caches: Dict[int, List] = {r: [] for r in range(gen_world)}

    sampling_params = SamplingParams()

    for req_idx, req_len in enumerate(request_lengths):
        unique_rid = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
        ctx_rid = req_idx * 2
        gen_rid = req_idx * 2 + 1
        ctx_request_ids.append(ctx_rid)
        gen_request_ids.append(gen_rid)

        ctx_dp_rank = req_idx % ctx_tp if ctx_enable_dp else 0

        ctx_request = LlmRequest(
            request_id=ctx_rid,
            max_new_tokens=1,
            input_tokens=list(range(req_len)),
            sampling_config=tensorrt_llm.bindings.SamplingConfig(
                sampling_params._get_sampling_config()
            ),
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
        )
        ctx_request.py_disaggregated_params = DisaggregatedParams(disagg_request_id=unique_rid)

        gen_request = LlmRequest(
            request_id=gen_rid,
            max_new_tokens=1,
            input_tokens=list(range(req_len)),
            sampling_config=tensorrt_llm.bindings.SamplingConfig(
                sampling_params._get_sampling_config()
            ),
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        )
        gen_request.py_disaggregated_params = DisaggregatedParams(
            ctx_request_id=ctx_rid,
            ctx_dp_rank=ctx_dp_rank,
            ctx_info_endpoint=ctx_info_endpoint,
            disagg_request_id=unique_rid,
        )

        for rank in range(ctx_world):
            tp_rank = rank % ctx_tp
            should_handle = (not ctx_enable_dp) or (req_idx % ctx_tp == tp_rank)
            if should_handle:
                ctx_handle_map[rank].append((req_idx, ctx_request))

        for rank in range(gen_world):
            tp_rank = rank % gen_tp
            should_handle = (not gen_enable_dp) or (req_idx % gen_tp == tp_rank)
            if should_handle:
                gen_handle_map[rank].append((req_idx, gen_request))

    # 5. Add sequences and gen receive first
    for rank in range(gen_world):
        for req_idx, req in gen_handle_map[rank]:
            kv = _add_sequence(gen_managers[rank], req.py_request_id, req.prompt_len, use_v2)
            if kv is not None:
                gen_kv_caches[rank].append(kv)
            gen_tcs[rank].request_and_receive_async(req)

    # 6. ctx send after
    for rank in range(ctx_world):
        for req_idx, req in ctx_handle_map[rank]:
            kv = _add_sequence(ctx_managers[rank], req.py_request_id, req.prompt_len, use_v2)
            if kv is not None:
                ctx_kv_caches[rank].append(kv)
            ctx_tcs[rank].respond_and_send_async(req)

    # 7. Wait for completion (threaded, dist calls inside)
    run_concurrent(ctx_tcs, lambda tc: tc.check_context_transfer_status(None, mark_complete=True))
    run_concurrent(gen_tcs, lambda tc: tc.check_gen_transfer_status(None))

    # 8. Verify
    verify_all_requests(
        request_lengths=request_lengths,
        ctx_managers=ctx_managers,
        gen_managers=gen_managers,
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        ctx_request_ids=ctx_request_ids,
        gen_request_ids=gen_request_ids,
        is_mla=is_mla,
        use_v2=use_v2,
        max_attention_window_vec=max_attention_window_vec,
        num_layers=num_layers,
    )

    # 9. Cleanup
    if use_v2:
        torch.cuda.current_stream().synchronize()
        for rank in range(ctx_world):
            for kv in ctx_kv_caches[rank]:
                kv.close()
        for rank in range(gen_world):
            for kv in gen_kv_caches[rank]:
                kv.close()

    for tc in ctx_tcs:
        if hasattr(tc, "transfer_worker") and tc.transfer_worker is not None:
            tc.transfer_worker.shutdown()
    for tc in gen_tcs:
        if hasattr(tc, "transfer_worker") and tc.transfer_worker is not None:
            tc.transfer_worker.shutdown()


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
TEST_CONFIGS = [
    # (ctx_tp, ctx_pp, gen_tp, gen_pp, ctx_dp, gen_dp, is_mla, test_id)
    # --- Basic symmetric ---
    (1, 1, 1, 1, False, False, False, "tp1_pp1"),
    (2, 1, 2, 1, False, False, False, "tp2_pp1"),
    (1, 2, 1, 2, False, False, False, "tp1_pp2"),
    (2, 2, 2, 2, False, False, False, "tp2_pp2"),
    (4, 1, 4, 1, False, False, False, "tp4_pp1"),
    (1, 4, 1, 4, False, False, False, "tp1_pp4"),
    (4, 2, 4, 2, False, False, False, "tp4_pp2"),
    (2, 4, 2, 4, False, False, False, "tp2_pp4"),
    (4, 4, 4, 4, False, False, False, "tp4_pp4"),
    # --- Asymmetric TP/PP between ctx and gen ---
    (1, 2, 1, 1, False, False, False, "pp2_to_pp1"),
    (2, 1, 1, 2, False, False, False, "tp2_to_pp2"),
    (4, 1, 2, 2, False, False, False, "tp4_to_tp2pp2"),
    (4, 1, 1, 4, False, False, False, "tp4_to_pp4"),
    (2, 2, 4, 1, False, False, False, "tp2pp2_to_tp4"),
    (1, 4, 4, 1, False, False, False, "pp4_to_tp4"),
    (4, 2, 2, 4, False, False, False, "tp4pp2_to_tp2pp4"),
    # --- DP (attention data parallelism) ---
    (2, 1, 2, 1, True, True, False, "tp2_dp_both"),
    (4, 1, 4, 1, True, True, False, "tp4_dp_both"),
    (2, 1, 1, 2, True, False, False, "ctx_dp_gen_pp2"),
    (4, 1, 2, 2, True, False, False, "tp4_dp_to_tp2pp2"),
    (2, 2, 2, 2, True, True, False, "tp2_pp2_dp_both"),
    (4, 2, 4, 2, True, True, False, "tp4_pp2_dp_both"),
    # --- MLA (Multi-head Latent Attention / SELFKONLY) ---
    (1, 1, 1, 1, False, False, True, "mla_tp1_pp1"),
    (2, 1, 2, 1, False, False, True, "mla_tp2"),
    (4, 1, 4, 1, False, False, True, "mla_tp4"),
    (2, 2, 2, 2, False, False, True, "mla_tp2_pp2"),
    (4, 2, 4, 2, False, False, True, "mla_tp4_pp2"),
    (2, 1, 2, 1, True, False, True, "mla_tp2_ctx_dp"),
    (4, 1, 4, 1, True, True, True, "mla_tp4_dp_both"),
    (2, 1, 1, 2, False, False, True, "mla_tp2_to_pp2"),
    # --- MLA asymmetric TP ---
    (4, 1, 8, 1, True, False, True, "mla_ctx_dp4_gen_tp8"),
    (4, 1, 8, 1, False, False, True, "mla_ctx_tp4_gen_tp8"),
]

WINDOW_CONFIGS = [
    # (max_attention_window_vec, test_id)
    (None, "no_window"),
    ([24], "uniform_window"),
    ([MAX_SEQ_LEN, 24], "vswa"),
]

# --- Uneven PP layer configs ---
# These test configurations use num_layers not evenly divisible by pp_size,
# so the first (num_layers % pp_size) PP ranks get one extra layer.
# Mirrors the C++ getLayerNumPPRank pattern in cacheTransceiverTest.cpp.
# Format: (ctx_tp, ctx_pp, gen_tp, gen_pp, ctx_dp, gen_dp, is_mla, num_layers, test_id)
UNEVEN_PP_CONFIGS = [
    # 5 layers, pp=2 → [3, 2]
    (1, 2, 1, 2, False, False, False, 5, "5L_tp1_pp2"),
    (2, 2, 2, 2, False, False, False, 5, "5L_tp2_pp2"),
    # 5 layers, pp=3 → [2, 2, 1]
    (1, 3, 1, 3, False, False, False, 5, "5L_tp1_pp3"),
    # 7 layers, pp=2 → [4, 3]
    (1, 2, 1, 2, False, False, False, 7, "7L_tp1_pp2"),
    (2, 2, 2, 2, False, False, False, 7, "7L_tp2_pp2"),
    # 7 layers, pp=3 → [3, 2, 2]
    (1, 3, 1, 3, False, False, False, 7, "7L_tp1_pp3"),
    # 7 layers, pp=4 → [2, 2, 2, 1]
    (1, 4, 1, 4, False, False, False, 7, "7L_tp1_pp4"),
    # Asymmetric TP/PP with uneven layers
    (2, 1, 1, 2, False, False, False, 5, "5L_tp2_to_pp2"),
    (1, 2, 2, 1, False, False, False, 5, "5L_pp2_to_tp2"),
    (4, 1, 1, 4, False, False, False, 7, "7L_tp4_to_pp4"),
    (1, 4, 4, 1, False, False, False, 7, "7L_pp4_to_tp4"),
    (2, 2, 1, 4, False, False, False, 7, "7L_tp2pp2_to_pp4"),
    # Uneven layers + MLA
    (1, 2, 1, 2, False, False, True, 5, "5L_mla_tp1_pp2"),
    (2, 2, 2, 2, False, False, True, 5, "5L_mla_tp2_pp2"),
    (1, 3, 1, 3, False, False, True, 7, "7L_mla_tp1_pp3"),
]


# ---------------------------------------------------------------------------
# pytest test functions
# ---------------------------------------------------------------------------
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,gen_tp,gen_pp,ctx_enable_dp,gen_enable_dp,is_mla",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6]) for c in TEST_CONFIGS],
    ids=[c[7] for c in TEST_CONFIGS],
)
@pytest.mark.parametrize(
    "max_attention_window_vec",
    [w[0] for w in WINDOW_CONFIGS],
    ids=[w[1] for w in WINDOW_CONFIGS],
)
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
def test_cache_transceiver(
    ctx_tp,
    ctx_pp,
    gen_tp,
    gen_pp,
    ctx_enable_dp,
    gen_enable_dp,
    is_mla,
    max_attention_window_vec,
    use_v2,
):
    """Test KvCacheTransceiverV2 with V1/V2 cache managers."""
    # VSWA (variable sliding window) only supported for V1 with ModelConfigCpp
    is_vswa = max_attention_window_vec is not None and len(set(max_attention_window_vec)) > 1
    if is_vswa and not use_v2:
        pytest.skip(
            "VSWA not supported: V2 lacks ModelConfigCpp, "
            "V1 transceiver requires getUniquePrimaryPool (single window)"
        )

    # V1 + MLA + sliding window: blocks_per_window key mismatch (known issue)
    if not use_v2 and is_mla and max_attention_window_vec is not None:
        pytest.skip(
            "V1 KVCacheManager + MLA + sliding window: "
            "blocks_per_window key mismatch for non-SELF cache types"
        )

    # # V2 + sliding window causes infinite loop in
    # # KVCacheManagerV2.get_num_available_tokens -> clamp_max_seq_len_for_mem
    # if use_v2 and max_attention_window_vec is not None:
    #     pytest.skip(
    #         "KVCacheManagerV2 + sliding window: infinite loop in "
    #         "clamp_max_seq_len_for_mem (known issue)"
    #     )

    print(
        f"\nRunning transfer test: "
        f"ctx_tp={ctx_tp} ctx_pp={ctx_pp} gen_tp={gen_tp} gen_pp={gen_pp} "
        f"ctx_dp={ctx_enable_dp} gen_dp={gen_enable_dp} "
        f"mla={is_mla} v2={use_v2} window={max_attention_window_vec}"
    )

    run_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        is_mla=is_mla,
        use_v2=use_v2,
        max_attention_window_vec=max_attention_window_vec,
    )

    print("PASSED")


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,gen_tp,gen_pp,ctx_enable_dp,gen_enable_dp,is_mla,num_layers",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]) for c in UNEVEN_PP_CONFIGS],
    ids=[c[8] for c in UNEVEN_PP_CONFIGS],
)
@pytest.mark.parametrize(
    "max_attention_window_vec",
    [w[0] for w in WINDOW_CONFIGS],
    ids=[w[1] for w in WINDOW_CONFIGS],
)
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
def test_cache_transceiver_uneven_pp(
    ctx_tp,
    ctx_pp,
    gen_tp,
    gen_pp,
    ctx_enable_dp,
    gen_enable_dp,
    is_mla,
    num_layers,
    max_attention_window_vec,
    use_v2,
):
    """Test KvCacheTransceiverV2 with uneven layers-per-PP-rank.

    When num_layers is not evenly divisible by pp_size, the first
    (num_layers % pp_size) PP ranks get one extra layer. This exercises
    the layer distribution logic that mirrors the C++ getLayerNumPPRank
    pattern in cacheTransceiverTest.cpp.
    """
    # VSWA (variable sliding window) only supported for V2 in transceiver tests
    is_vswa = max_attention_window_vec is not None and len(set(max_attention_window_vec)) > 1
    if is_vswa and not use_v2:
        pytest.skip(
            "VSWA not supported: V2 lacks ModelConfigCpp, "
            "V1 transceiver requires getUniquePrimaryPool (single window)"
        )

    # V1 + MLA + sliding window: blocks_per_window key mismatch (known issue)
    if not use_v2 and is_mla and max_attention_window_vec is not None:
        pytest.skip(
            "V1 KVCacheManager + MLA + sliding window: "
            "blocks_per_window key mismatch for non-SELF cache types"
        )

    print(
        f"\nRunning uneven PP transfer test: "
        f"ctx_tp={ctx_tp} ctx_pp={ctx_pp} gen_tp={gen_tp} gen_pp={gen_pp} "
        f"ctx_dp={ctx_enable_dp} gen_dp={gen_enable_dp} "
        f"mla={is_mla} v2={use_v2} num_layers={num_layers} "
        f"window={max_attention_window_vec} "
        f"layers_per_pp(ctx)={_get_layers_per_pp(num_layers, ctx_pp)} "
        f"layers_per_pp(gen)={_get_layers_per_pp(num_layers, gen_pp)}"
    )

    run_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        is_mla=is_mla,
        use_v2=use_v2,
        max_attention_window_vec=max_attention_window_vec,
        num_layers=num_layers,
    )

    print("PASSED")


# ---------------------------------------------------------------------------
# Boundary request-length tests
# ---------------------------------------------------------------------------
# Lengths chosen to exercise block-boundary and window-boundary edge cases:
#   TOKENS_PER_BLOCK=8, window_size=24
#   1         : single token (1 partial block)
#   7, 8, 9   : tpb-1, tpb, tpb+1 (block boundary)
#   23, 24, 25: win-1, win, win+1 (window boundary)
#   32, 33    : first stale_end=1 transition for win=24
BOUNDARY_REQUEST_LENGTHS = [1, 7, 8, 9, 23, 24, 25, 32, 33]

# Use a focused set of topologies to keep test count reasonable.
BOUNDARY_CONFIGS = [
    # (ctx_tp, ctx_pp, gen_tp, gen_pp, ctx_dp, gen_dp, is_mla, test_id)
    (1, 1, 1, 1, False, False, False, "tp1_pp1"),
    (2, 1, 2, 1, False, False, False, "tp2_pp1"),
    (1, 2, 1, 2, False, False, False, "tp1_pp2"),
    (2, 1, 1, 2, False, False, False, "tp2_to_pp2"),
    (1, 1, 1, 1, False, False, True, "mla_tp1_pp1"),
]


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,gen_tp,gen_pp,ctx_enable_dp,gen_enable_dp,is_mla",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6]) for c in BOUNDARY_CONFIGS],
    ids=[c[7] for c in BOUNDARY_CONFIGS],
)
@pytest.mark.parametrize(
    "max_attention_window_vec",
    [w[0] for w in WINDOW_CONFIGS],
    ids=[w[1] for w in WINDOW_CONFIGS],
)
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
def test_cache_transceiver_boundary_lengths(
    ctx_tp,
    ctx_pp,
    gen_tp,
    gen_pp,
    ctx_enable_dp,
    gen_enable_dp,
    is_mla,
    max_attention_window_vec,
    use_v2,
):
    """Test KvCacheTransceiverV2 with boundary request lengths.

    Exercises block-boundary and window-boundary edge cases
    (e.g. prompt_len == tokens_per_block, prompt_len == window_size ± 1).
    """
    is_vswa = max_attention_window_vec is not None and len(set(max_attention_window_vec)) > 1
    if is_vswa and not use_v2:
        pytest.skip(
            "VSWA not supported: V2 lacks ModelConfigCpp, "
            "V1 transceiver requires getUniquePrimaryPool (single window)"
        )

    # V1 + MLA + sliding window: blocks_per_window key mismatch (known issue)
    if not use_v2 and is_mla and max_attention_window_vec is not None:
        pytest.skip(
            "V1 KVCacheManager + MLA + sliding window: "
            "blocks_per_window key mismatch for non-SELF cache types"
        )

    print(
        f"\nRunning boundary transfer test: "
        f"ctx_tp={ctx_tp} ctx_pp={ctx_pp} gen_tp={gen_tp} gen_pp={gen_pp} "
        f"ctx_dp={ctx_enable_dp} gen_dp={gen_enable_dp} "
        f"mla={is_mla} v2={use_v2} window={max_attention_window_vec}"
    )

    run_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        is_mla=is_mla,
        use_v2=use_v2,
        max_attention_window_vec=max_attention_window_vec,
        request_lengths=BOUNDARY_REQUEST_LENGTHS,
    )

    print("PASSED")


if __name__ == "__main__":
    # Quick smoke test
    run_transfer_test(1, 1, 1, 1, False, False, False, False)
