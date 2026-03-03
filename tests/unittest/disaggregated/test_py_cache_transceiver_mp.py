"""Multi-process test for PyNativeCacheTransceiver (V2 backend).

This test uses torch.multiprocessing to spawn multiple processes simulating
ctx and gen instances with different TP/PP configurations.
"""

import os
import signal
import sys
import uuid

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType


def broadcast_string(s: str | None, src: int, group: dist.ProcessGroup | None = None) -> str:
    """Broadcast a string from src rank to all other ranks in the group."""
    if dist.get_rank(group) == src:
        encoded = s.encode("utf-8")
        length = torch.tensor([len(encoded)], dtype=torch.long)
    else:
        length = torch.tensor([0], dtype=torch.long)

    dist.broadcast(length, src=src, group=group)

    if dist.get_rank(group) == src:
        data = torch.tensor(list(encoded), dtype=torch.uint8)
    else:
        data = torch.empty(length.item(), dtype=torch.uint8)

    dist.broadcast(data, src=src, group=group)

    return bytes(data.tolist()).decode("utf-8")


def allgather_strings(s: str, group: dist.ProcessGroup | None = None) -> list[str]:
    """Allgather strings from all ranks in the group."""
    world_size = dist.get_world_size(group)

    encoded = s.encode("utf-8")
    local_length = torch.tensor([len(encoded)], dtype=torch.long)
    all_lengths = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_lengths, local_length, group=group)

    max_length = max(length.item() for length in all_lengths)
    local_data = torch.zeros(max_length, dtype=torch.uint8)
    local_data[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)

    all_data = [torch.zeros(max_length, dtype=torch.uint8) for _ in range(world_size)]
    dist.all_gather(all_data, local_data, group=group)

    results = []
    for data, length in zip(all_data, all_lengths):
        results.append(bytes(data[: length.item()].tolist()).decode("utf-8"))

    return results


def allgather_int(value: int, group: dist.ProcessGroup | None = None) -> list[int]:
    """Allgather integers from all ranks in the group."""
    world_size = dist.get_world_size(group)
    local_tensor = torch.tensor([value], dtype=torch.long)
    all_tensors = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_tensors, local_tensor, group=group)
    return [t.item() for t in all_tensors]


def broadcast_int(value: int | None, src: int, group: dist.ProcessGroup | None = None) -> int:
    """Broadcast an int64 from src rank to all other ranks in the group."""
    if dist.get_rank(group) == src:
        data = torch.tensor([value], dtype=torch.long)
    else:
        data = torch.zeros(1, dtype=torch.long)

    dist.broadcast(data, src=src, group=group)
    return data.item()


def find_free_port():
    """Find a free port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class TorchDistributedWrapper:
    """A wrapper that provides the Distributed interface using torch.distributed.

    This is used to create a compatible Distributed object for PyNativeCacheTransceiver
    in multi-process tests.
    """

    def __init__(
        self,
        mapping: Mapping,
        instance_group,
        instance_ranks: list[int],
        tp_group,
        tp_ranks: list[int],
        pp_group,
        pp_ranks: list[int],
    ):
        """Wrapper for torch.distributed to implement Distributed interface.

        Args:
            mapping: The Mapping object for this instance (ctx or gen).
            instance_group: Process group containing all ranks of this instance.
            instance_ranks: List of global ranks in instance_group (sorted).
            tp_group: Process group for TP communication.
            tp_ranks: List of global ranks in tp_group (sorted).
            pp_group: Process group for PP communication.
            pp_ranks: List of global ranks in pp_group (sorted).
        """
        self.mapping = mapping
        self._instance_group = instance_group
        self._instance_ranks = instance_ranks
        self._tp_group = tp_group
        self._tp_ranks = tp_ranks
        self._pp_group = pp_group
        self._pp_ranks = pp_ranks

    @property
    def rank(self):
        return self.mapping.rank

    @property
    def world_size(self):
        return self.mapping.world_size

    @property
    def tp_size(self):
        return self.mapping.tp_size

    @property
    def pp_size(self):
        return self.mapping.pp_size

    def broadcast(self, obj, root=0):
        ret = [obj]
        # root is local rank, convert to global rank
        global_src = self._instance_ranks[root]
        dist.broadcast_object_list(
            ret, src=global_src, group=self._instance_group, device=torch.device("cpu")
        )
        return ret[0]

    def allgather(self, obj):
        output_list = [None] * self.world_size
        dist.all_gather_object(output_list, obj, group=self._instance_group)
        return output_list

    def tp_allgather(self, obj):
        output_list = [None] * self.tp_size
        dist.all_gather_object(output_list, obj, group=self._tp_group)
        return output_list

    def tp_broadcast(self, obj, root=0, **kwargs):
        ret = [obj]
        # root is local rank within TP group, convert to global rank
        global_src = self._tp_ranks[root]
        dist.broadcast_object_list(
            ret, src=global_src, group=self._tp_group, device=torch.device("cpu")
        )
        return ret[0]

    def pp_allgather(self, obj):
        output_list = [None] * self.pp_size
        dist.all_gather_object(output_list, obj, group=self._pp_group)
        return output_list

    def pp_broadcast(self, obj, root=0):
        ret = [obj]
        # root is local rank within PP group, convert to global rank
        global_src = self._pp_ranks[root]
        dist.broadcast_object_list(
            ret, src=global_src, group=self._pp_group, device=torch.device("cpu")
        )
        return ret[0]


def worker_fn(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool = False,
    gen_enable_dp: bool = False,
    is_mla: bool = False,
):
    """Worker function for each process."""

    # Signal handler for graceful termination
    def signal_handler(signum, frame):
        print(f"[Rank {rank}] Received signal {signum}, exiting...", flush=True)
        # Try to cleanup distributed
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Set environment variables for distributed
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize distributed (use gloo for single GPU compatibility)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tensorrt_llm.logger.set_level("info")

    ctx_instance_num = ctx_tp * ctx_pp
    gen_instance_num = gen_tp * gen_pp

    # Determine if this process is ctx or gen
    is_ctx = rank < ctx_instance_num
    local_rank = rank if is_ctx else rank - ctx_instance_num

    # Set GPU device
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)

    # Create process groups
    ctx_ranks = list(range(ctx_instance_num))
    gen_ranks = list(range(ctx_instance_num, ctx_instance_num + gen_instance_num))

    ctx_group = dist.new_group(ranks=ctx_ranks)
    gen_group = dist.new_group(ranks=gen_ranks)
    # Group for broadcasting ctx_info_endpoint from ctx rank 0 to all gen ranks
    ctx_to_gen_group = dist.new_group(ranks=[0] + gen_ranks)

    # Common parameters
    num_layers = 4
    head_dim = 128
    # MLA uses num_kv_heads = 1
    num_kv_heads = 1 if is_mla else 4
    tokens_per_block = 8
    max_seq_len = 256
    max_batch_size = 4
    dtype = DataType.FLOAT

    # Cache type: SELFKONLY for MLA, SELF otherwise
    cache_type = (
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY
        if is_mla
        else tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
    )

    # Import PyNativeCacheTransceiver
    from tensorrt_llm._torch.disaggregation.native.py_cache_transceiver import (
        PyNativeCacheTransceiver,
    )

    # ===== Create all TP/PP groups in the same order for all ranks =====
    # dist.new_group is a collective operation - ALL ranks must call it in the same order!
    # First, create all ctx TP/PP groups (all ranks participate)
    ctx_tp_groups = {}  # pp_rank -> tp_group
    ctx_pp_groups = {}  # tp_rank -> pp_group
    for ctx_pp_rank in range(ctx_pp):
        for ctx_tp_rank in range(ctx_tp):
            # TP group for this pp_rank
            tp_ranks = [ctx_pp_rank * ctx_tp + i for i in range(ctx_tp)]
            if ctx_tp_rank == 0:  # Only create once per pp_rank
                ctx_tp_groups[ctx_pp_rank] = dist.new_group(ranks=tp_ranks)
            # PP group for this tp_rank
            pp_ranks = [i * ctx_tp + ctx_tp_rank for i in range(ctx_pp)]
            if ctx_pp_rank == 0:  # Only create once per tp_rank
                ctx_pp_groups[ctx_tp_rank] = dist.new_group(ranks=pp_ranks)

    # Then, create all gen TP/PP groups (all ranks participate)
    gen_tp_groups = {}  # pp_rank -> tp_group
    gen_pp_groups = {}  # tp_rank -> pp_group
    for gen_pp_rank in range(gen_pp):
        for gen_tp_rank in range(gen_tp):
            # TP group for this pp_rank (using global ranks)
            tp_ranks = [ctx_instance_num + gen_pp_rank * gen_tp + i for i in range(gen_tp)]
            if gen_tp_rank == 0:  # Only create once per pp_rank
                gen_tp_groups[gen_pp_rank] = dist.new_group(ranks=tp_ranks)
            # PP group for this tp_rank (using global ranks)
            pp_ranks = [ctx_instance_num + i * gen_tp + gen_tp_rank for i in range(gen_pp)]
            if gen_pp_rank == 0:  # Only create once per tp_rank
                gen_pp_groups[gen_tp_rank] = dist.new_group(ranks=pp_ranks)

    print(f"[Rank {rank}] All TP/PP groups created", flush=True)

    # ===== Now create instance-specific resources =====
    # Initialize variables that will be used in nested functions
    ctx_tp_group = ctx_pp_group = None
    ctx_tp_ranks_local = ctx_pp_ranks_local = []
    gen_tp_group = gen_pp_group = None
    gen_tp_ranks_local = gen_pp_ranks_local = []

    if is_ctx:
        # Create ctx mapping
        pp_rank = local_rank // ctx_tp
        tp_rank = local_rank % ctx_tp
        mapping = Mapping(
            world_size=ctx_instance_num,
            rank=local_rank,
            tp_size=ctx_tp,
            pp_size=ctx_pp,
            enable_attention_dp=ctx_enable_dp,
        )

        # Get the TP/PP groups for this rank
        ctx_tp_group = ctx_tp_groups[pp_rank]
        ctx_pp_group = ctx_pp_groups[tp_rank]
        ctx_tp_ranks_local = [pp_rank * ctx_tp + i for i in range(ctx_tp)]
        ctx_pp_ranks_local = [i * ctx_tp + tp_rank for i in range(ctx_pp)]

        # Create KVCacheManager
        kv_cache_manager = KVCacheManager(
            trtllm.KvCacheConfig(max_tokens=2048, enable_block_reuse=False),
            cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
        )

        # Initialize with random data
        # For MLA without DP, all TP ranks within ctx/gen must have the same data (use fixed seed)
        block_data_pool = kv_cache_manager.get_unique_primary_pool()
        if is_mla and not ctx_enable_dp:
            # Use seed=42 for ctx ranks to ensure they have identical data
            generator = torch.Generator(device=block_data_pool.device).manual_seed(42)
        else:
            generator = None
        random_values = torch.rand(
            block_data_pool.shape,
            dtype=torch.float32,
            device=block_data_pool.device,
            generator=generator,
        )
        block_data_pool.copy_(random_values)

        # Create Distributed wrapper (ctx_group contains all ctx ranks)
        dist_wrapper = TorchDistributedWrapper(
            mapping,
            ctx_group,
            ctx_ranks,
            ctx_tp_group,
            ctx_tp_ranks_local,
            ctx_pp_group,
            ctx_pp_ranks_local,
        )

        # Create cache transceiver config
        cache_transceiver_config = CacheTransceiverConfig(
            backend="NIXL", transceiver_runtime="PYTHON", max_tokens_in_buffer=512
        )

        # Create PyNativeCacheTransceiver
        attention_type = AttentionTypeCpp.MLA if is_mla else AttentionTypeCpp.DEFAULT
        print(f"[Rank {rank}] CTX: Creating transceiver...", flush=True)
        transceiver = PyNativeCacheTransceiver(
            mapping=mapping,
            dist=dist_wrapper,
            kv_cache_manager=kv_cache_manager,
            attention_type=attention_type,
            cache_transceiver_config=cache_transceiver_config,
        )
        print(f"[Rank {rank}] CTX: Transceiver created", flush=True)
        ctx_info_endpoint = transceiver.context_info_endpoint if local_rank == 0 else None

    else:  # gen process
        # Create gen mapping
        pp_rank = local_rank // gen_tp
        tp_rank = local_rank % gen_tp
        mapping = Mapping(
            world_size=gen_instance_num,
            rank=local_rank,
            tp_size=gen_tp,
            pp_size=gen_pp,
            enable_attention_dp=gen_enable_dp,
        )

        # Get the TP/PP groups for this rank
        gen_tp_group = gen_tp_groups[pp_rank]
        gen_pp_group = gen_pp_groups[tp_rank]
        gen_tp_ranks_local = [ctx_instance_num + pp_rank * gen_tp + i for i in range(gen_tp)]
        gen_pp_ranks_local = [ctx_instance_num + i * gen_tp + tp_rank for i in range(gen_pp)]

        # Create KVCacheManager
        kv_cache_manager = KVCacheManager(
            trtllm.KvCacheConfig(max_tokens=2048, enable_block_reuse=False),
            cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
        )

        # Initialize gen with random data (to verify transfer overwrites it)
        block_data_pool = kv_cache_manager.get_unique_primary_pool()
        random_values = torch.rand(
            block_data_pool.shape, dtype=torch.float32, device=block_data_pool.device
        )
        block_data_pool.copy_(random_values)

        # Create Distributed wrapper (gen_group contains all gen ranks)
        dist_wrapper = TorchDistributedWrapper(
            mapping,
            gen_group,
            gen_ranks,
            gen_tp_group,
            gen_tp_ranks_local,
            gen_pp_group,
            gen_pp_ranks_local,
        )

        # Create cache transceiver config
        cache_transceiver_config = CacheTransceiverConfig(
            backend="NIXL", transceiver_runtime="PYTHON", max_tokens_in_buffer=512
        )

        # Create PyNativeCacheTransceiver
        attention_type = AttentionTypeCpp.MLA if is_mla else AttentionTypeCpp.DEFAULT
        print(f"[Rank {rank}] GEN: Creating transceiver...", flush=True)
        transceiver = PyNativeCacheTransceiver(
            mapping=mapping,
            dist=dist_wrapper,
            kv_cache_manager=kv_cache_manager,
            attention_type=attention_type,
            cache_transceiver_config=cache_transceiver_config,
        )
        print(f"[Rank {rank}] GEN: Transceiver created", flush=True)
        ctx_info_endpoint = None

    # Broadcast ctx_info_endpoint from ctx rank 0 to gen ranks
    if rank in [0] + gen_ranks:
        ctx_info_endpoint = broadcast_string(ctx_info_endpoint, src=0, group=ctx_to_gen_group)

    # Synchronize all processes
    dist.barrier()

    # ===== Batch process multiple requests (like C++ cacheTransceiverTest) =====
    # Reference: C++ test uses lenList = {30, 10, 60, 80}
    request_lengths = [30, 10, 60, 80]

    # Filter out lengths that are too short
    request_lengths = [length for length in request_lengths if length > 0]

    # Helper: create request with given parameters
    def create_request(
        request_idx: int, ctx_request_id: int, gen_request_id: int, req_len: int, unique_rid: int
    ) -> LlmRequest:
        """Create LlmRequest for ctx or gen."""
        sampling_params = SamplingParams()

        # Determine ctx_dp_rank for this request (when DP is enabled)
        # Use request_idx (not ctx_request_id) to match DP round-robin assignment
        if ctx_enable_dp:
            actual_ctx_dp_rank = request_idx % ctx_tp
        else:
            actual_ctx_dp_rank = 0

        if is_ctx:
            request = LlmRequest(
                request_id=ctx_request_id,
                max_new_tokens=1,
                input_tokens=list(range(req_len)),
                sampling_config=tensorrt_llm.bindings.SamplingConfig(
                    sampling_params._get_sampling_config()
                ),
                is_streaming=False,
                llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
            )
            request.py_disaggregated_params = DisaggregatedParams(disagg_request_id=unique_rid)
        else:
            request = LlmRequest(
                request_id=gen_request_id,
                max_new_tokens=1,
                input_tokens=list(range(req_len)),
                sampling_config=tensorrt_llm.bindings.SamplingConfig(
                    sampling_params._get_sampling_config()
                ),
                is_streaming=False,
                llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
            )
            request.py_disaggregated_params = DisaggregatedParams(
                ctx_request_id=ctx_request_id,
                ctx_dp_rank=actual_ctx_dp_rank,
                ctx_info_endpoint=ctx_info_endpoint,
                disagg_request_id=unique_rid,
            )
        return request

    # Helper: merge block data from all ranks for verification
    def merge_block_data(
        block_datas: list[torch.Tensor],
        tp_size: int,
        pp_size: int,
        layer_num_per_pp: list[int],
        enable_dp: bool,
        request_idx: int,
    ) -> torch.Tensor:
        """Merge block data from all TP/PP ranks into a single tensor.

        When DP is enabled, only ranks with tp_rank == request_idx % tp_size have valid data.
        block_datas is ordered by local_rank: [pp0_tp0, pp0_tp1, ..., pp1_tp0, pp1_tp1, ...]
        i.e., local_rank = pp_rank * tp_size + tp_rank
        """
        # When DP is enabled or MLA, only one TP rank has valid data
        valid_tp = 1 if enable_dp else tp_size
        if is_mla:
            valid_tp = 1

        # Determine which tp_rank has valid data when DP is enabled
        valid_tp_rank = request_idx % tp_size if enable_dp else 0

        # MLA only has K cache (dim=1), non-MLA has K+V cache (dim=2)
        kv_dim = 1 if is_mla else 2
        merged = torch.zeros(
            size=(
                block_datas[0].shape[0],
                num_layers,
                kv_dim,
                block_datas[0].shape[3] * valid_tp,
            ),
            device="cuda",
        )
        for pp_idx in range(pp_size):
            for tp_offset in range(valid_tp):
                layer_start_idx = sum(layer_num_per_pp[:pp_idx])
                layer_end_idx = layer_start_idx + layer_num_per_pp[pp_idx]
                head_dim_per_rank = num_kv_heads // valid_tp * head_dim * tokens_per_block
                start_head_offset = tp_offset * head_dim_per_rank
                end_head_offset = start_head_offset + head_dim_per_rank

                # Calculate actual tp_rank to use
                actual_tp_rank = valid_tp_rank + tp_offset if enable_dp else tp_offset
                # local_rank = pp_rank * tp_size + tp_rank
                rank_idx = pp_idx * tp_size + actual_tp_rank

                merged[:, layer_start_idx:layer_end_idx, :, start_head_offset:end_head_offset] = (
                    block_datas[rank_idx]
                )
        return merged

    # Helper: gather and verify a single request's data
    def gather_and_verify_request(
        request: LlmRequest, ctx_request_id: int, request_idx: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Gather block data from all ranks to local_rank 0, then verify on world rank 0.

        All ranks have all requests' block data (via add_sequence), so gather is simple.
        In DP mode, merge_block_data knows which ranks have valid (transferred) data.
        """
        blocks = kv_cache_manager.get_batch_cache_indices([request.py_request_id])[0]
        block_data = kv_cache_manager.get_unique_primary_pool()[blocks]
        block_data_cpu = block_data.cpu()

        ctx_block_datas = None
        gen_block_datas = None

        # All ranks gather to local_rank 0 within their instance group
        if is_ctx:
            if local_rank == 0:
                ctx_block_datas_cpu = [
                    torch.zeros_like(block_data_cpu) for _ in range(ctx_instance_num)
                ]
            else:
                ctx_block_datas_cpu = None
            dist.gather(block_data_cpu, ctx_block_datas_cpu, dst=0, group=ctx_group)
            if local_rank == 0:
                ctx_block_datas = [d.cuda() for d in ctx_block_datas_cpu]
        else:
            gen_dst = gen_ranks[0]  # First global rank in gen_group
            if local_rank == 0:
                gen_block_datas_cpu = [
                    torch.zeros_like(block_data_cpu) for _ in range(gen_instance_num)
                ]
            else:
                gen_block_datas_cpu = None
            dist.gather(block_data_cpu, gen_block_datas_cpu, dst=gen_dst, group=gen_group)
            if local_rank == 0:
                gen_block_datas = [d.cuda() for d in gen_block_datas_cpu]

        dist.barrier()

        # Send gen_block_datas from gen local_rank 0 to world rank 0
        gen_rank_0 = ctx_instance_num
        if rank == gen_rank_0 and local_rank == 0:
            shape_tensor = torch.tensor(list(block_data.shape), dtype=torch.long)
            dist.send(shape_tensor, dst=0)
            for data in gen_block_datas:
                dist.send(data.cpu().contiguous(), dst=0)
        elif rank == 0:
            shape_tensor = torch.zeros(4, dtype=torch.long)
            dist.recv(shape_tensor, src=gen_rank_0)
            gen_shape = tuple(shape_tensor.tolist())
            gen_block_datas = []
            for _ in range(gen_instance_num):
                data_cpu = torch.zeros(gen_shape, dtype=block_data.dtype)
                dist.recv(data_cpu, src=gen_rank_0)
                gen_block_datas.append(data_cpu.cuda())

        dist.barrier()

        # Merge and compare on world rank 0
        if rank == 0:
            ctx_layer_num_per_pp = [num_layers // ctx_pp] * ctx_pp
            gen_layer_num_per_pp = [num_layers // gen_pp] * gen_pp

            ctx_merged = merge_block_data(
                ctx_block_datas, ctx_tp, ctx_pp, ctx_layer_num_per_pp, ctx_enable_dp, request_idx
            )
            gen_merged = merge_block_data(
                gen_block_datas, gen_tp, gen_pp, gen_layer_num_per_pp, gen_enable_dp, request_idx
            )
            return ctx_merged, gen_merged
        return None, None

    # ===== Phase 1: Create all requests and broadcast unique_rids =====
    # When DP is enabled, only specific tp_rank handles specific requests (round-robin)
    all_requests = []  # All requests for consistent indexing
    my_requests = []  # Only requests this rank should handle
    unique_rids = []

    for i, req_len in enumerate(request_lengths):
        unique_rid = broadcast_int(
            uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF if rank == 0 else None, src=0
        )
        unique_rids.append(unique_rid)

        ctx_request_id = i * 2
        gen_request_id = i * 2 + 1
        request = create_request(i, ctx_request_id, gen_request_id, req_len, unique_rid)
        all_requests.append(request)

        # Determine if this rank should handle this request (DP round-robin)
        should_handle = True
        if is_ctx and ctx_enable_dp:
            # Context DP: only handle if request_index % ctx_tp == tp_rank
            should_handle = i % ctx_tp == tp_rank
        elif not is_ctx and gen_enable_dp:
            # Generation DP: only handle if request_index % gen_tp == tp_rank
            should_handle = i % gen_tp == tp_rank

        # All ranks add_sequence so they have block data for verification
        # But only ranks that should_handle will submit to transceiver
        kv_cache_manager.impl.add_sequence(request.py_request_id, request.prompt_len, 1, request)

        if should_handle:
            my_requests.append((i, request))  # Store index and request for transfer

    print(
        f"[Rank {rank}] Created {len(all_requests)} requests, handling {len(my_requests)}, "
        f"{'CTX' if is_ctx else 'GEN'} mode, tp_rank={tp_rank}",
        flush=True,
    )

    # ===== Phase 2a: Warmup (only when DP is disabled) =====
    do_warmup = not ctx_enable_dp and not gen_enable_dp and len(my_requests) > 0
    if do_warmup:
        warmup_idx, warmup_request = my_requests[0]
        remaining_requests = my_requests[1:]

        if is_ctx:
            print(f"[Rank {rank}] CTX: Submitting warmup request {warmup_idx}...", flush=True)
            transceiver.respond_and_send_async(warmup_request)

        # Barrier to ensure ctx has submitted warmup before gen starts
        print(f"[Rank {rank}] Before warmup barrier", flush=True)
        dist.barrier()
        print(f"[Rank {rank}] After warmup barrier", flush=True)

        if not is_ctx:
            print(f"[Rank {rank}] GEN: Submitting warmup request {warmup_idx}...", flush=True)
            transceiver.request_and_receive_async(warmup_request)

        # Wait for warmup to complete
        print(f"[Rank {rank}] Waiting for warmup to complete...", flush=True)
        if is_ctx:
            transceiver.check_context_transfer_status(None)
            print(f"[Rank {rank}] CTX: Warmup completed", flush=True)
        else:
            transceiver.check_gen_transfer_status(None)
            print(f"[Rank {rank}] GEN: Warmup completed", flush=True)

        # Barrier after warmup
        print(f"[Rank {rank}] Before post-warmup barrier", flush=True)
        dist.barrier()
        print(f"[Rank {rank}] After post-warmup barrier", flush=True)
    else:
        remaining_requests = my_requests

    # ===== Phase 2b: Batch send/receive remaining requests =====
    if is_ctx:
        for req_idx, request in remaining_requests:
            print(f"[Rank {rank}] CTX: Submitting request {req_idx}...", flush=True)
            transceiver.respond_and_send_async(request)
        print(f"[Rank {rank}] CTX: Submitted {len(remaining_requests)} send requests", flush=True)

    # Barrier to ensure ctx has submitted all requests before gen starts receiving
    print(f"[Rank {rank}] Before phase2 barrier", flush=True)
    dist.barrier()
    print(f"[Rank {rank}] After phase2 barrier", flush=True)

    if not is_ctx:
        for req_idx, request in remaining_requests:
            print(f"[Rank {rank}] GEN: Submitting request {req_idx}...", flush=True)
            transceiver.request_and_receive_async(request)
        print(
            f"[Rank {rank}] GEN: Submitted {len(remaining_requests)} receive requests", flush=True
        )

    # ===== Phase 3: Wait for remaining transfers to complete =====
    num_remaining = len(remaining_requests)
    print(f"[Rank {rank}] Phase 3: Waiting for {num_remaining} transfers...", flush=True)
    if is_ctx and remaining_requests:
        transceiver.check_context_transfer_status(None)
        print(f"[Rank {rank}] CTX: All {num_remaining} transfers completed", flush=True)
    elif not is_ctx and remaining_requests:
        transceiver.check_gen_transfer_status(None)
        print(f"[Rank {rank}] GEN: All {num_remaining} transfers completed", flush=True)

    # Synchronize before verification
    print(f"[Rank {rank}] Before phase3 barrier", flush=True)
    dist.barrier()
    print(f"[Rank {rank}] After phase3 barrier", flush=True)

    # ===== Phase 4: Batch verify all requests =====
    # All ranks must participate in gather (collective op), so iterate all_requests.
    # Verification happens on rank 0.
    print(f"[Rank {rank}] Starting batch verification...", flush=True)

    all_passed = True
    verification_results = []

    for req_idx, request in enumerate(all_requests):
        ctx_request_id = req_idx * 2
        ctx_merged, gen_merged = gather_and_verify_request(request, ctx_request_id, req_idx)

        # Only rank 0 has the merged data for verification
        if rank == 0:
            req_len = request_lengths[req_idx]
            if ctx_merged.equal(gen_merged):
                verification_results.append((ctx_request_id, req_len, True, None))
            else:
                # Find first mismatch for debugging
                diff = (ctx_merged != gen_merged).nonzero()
                first_diff = diff[0].tolist() if len(diff) > 0 else None
                ctx_val = ctx_merged[tuple(diff[0])].item() if first_diff else None
                gen_val = gen_merged[tuple(diff[0])].item() if first_diff else None
                verification_results.append(
                    (ctx_request_id, req_len, False, (first_diff, ctx_val, gen_val))
                )

    # Print results and assert on rank 0
    if rank == 0:
        print(f"\n[Rank {rank}] ===== Verification Results =====", flush=True)
        for ctx_request_id, req_len, passed, debug_info in verification_results:
            if passed:
                print(f"  Request {ctx_request_id} (len={req_len}): PASSED", flush=True)
            else:
                print(f"  Request {ctx_request_id} (len={req_len}): FAILED!", flush=True)
                if debug_info:
                    first_diff, ctx_val, gen_val = debug_info
                    print(f"    First mismatch at {first_diff}", flush=True)
                    print(f"    CTX value: {ctx_val}", flush=True)
                    print(f"    GEN value: {gen_val}", flush=True)
                all_passed = False

        if all_passed:
            print(
                f"\n[Rank {rank}] All {len(all_requests)} requests verified successfully!",
                flush=True,
            )
        else:
            print(f"\n[Rank {rank}] Some requests FAILED verification!", flush=True)

    # Broadcast pass/fail to all ranks for assertion
    pass_tensor = torch.tensor([1 if all_passed else 0], dtype=torch.int)
    dist.broadcast(pass_tensor, src=0)
    assert pass_tensor.item() == 1, "Some requests failed verification!"

    dist.barrier()

    # ===== Phase 5: Cleanup requests =====
    # All ranks added all requests, so all need to remove them
    for request in all_requests:
        # remove_sequence(request_id, llm_request, release_blocks)
        kv_cache_manager.impl.remove_sequence(request.py_request_id, request, True)

    if rank == 0:
        print(f"[Rank {rank}] Cleanup completed")

    # Cleanup
    dist.destroy_process_group()


def run_v2_transceiver_mp(
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool = False,
    gen_enable_dp: bool = False,
    is_mla: bool = False,
):
    """Multi-process test for PyNativeCacheTransceiver using mp.spawn."""
    world_size = ctx_tp * ctx_pp + gen_tp * gen_pp

    master_addr = "127.0.0.1"
    master_port = find_free_port()

    dp_str = (
        f", ctx_dp={ctx_enable_dp}, gen_dp={gen_enable_dp}"
        if ctx_enable_dp or gen_enable_dp
        else ""
    )
    mla_str = ", MLA" if is_mla else ""
    print(
        f"Starting {world_size} processes for V2 transceiver test: "
        f"ctx_tp={ctx_tp}, ctx_pp={ctx_pp}, gen_tp={gen_tp}, gen_pp={gen_pp}{dp_str}{mla_str}"
    )

    mp.spawn(
        worker_fn,
        args=(
            world_size,
            master_addr,
            master_port,
            ctx_tp,
            ctx_pp,
            gen_tp,
            gen_pp,
            ctx_enable_dp,
            gen_enable_dp,
            is_mla,
        ),
        nprocs=world_size,
        join=True,
    )

    print(f"Test passed: ctx_tp={ctx_tp}, ctx_pp={ctx_pp}, gen_tp={gen_tp}, gen_pp={gen_pp}\n")


# Test configurations as pytest parameters
# Reference: cpp/tests/unit_tests/multi_gpu/cacheTransceiverTest.cpp
MP_TEST_CONFIGS = [
    # (ctx_tp, ctx_pp, gen_tp, gen_pp, ctx_enable_dp, gen_enable_dp, is_mla, test_id)
    # Basic configurations
    (1, 1, 1, 1, False, False, False, "v2_mp_tp1_pp1_to_tp1_pp1"),
    (1, 1, 1, 2, False, False, False, "v2_mp_tp1_pp1_to_tp1_pp2"),
    (1, 2, 1, 1, False, False, False, "v2_mp_tp1_pp2_to_tp1_pp1"),
    (1, 2, 1, 2, False, False, False, "v2_mp_tp1_pp2_to_tp1_pp2"),
    # TP variations
    (2, 1, 1, 2, False, False, False, "v2_mp_tp2_pp1_to_tp1_pp2"),
    (2, 1, 2, 1, False, False, False, "v2_mp_tp2_pp1_to_tp2_pp1"),
    # Mixed TP/PP
    (2, 2, 2, 2, False, False, False, "v2_mp_tp2_pp2_to_tp2_pp2"),
    # DP (Data Parallelism for Attention) configurations
    (2, 1, 2, 1, True, True, False, "v2_mp_tp2_pp1_dp_to_tp2_pp1_dp"),
    (2, 1, 1, 2, True, False, False, "v2_mp_tp2_pp1_dp_to_tp1_pp2"),
    (1, 4, 2, 2, False, True, False, "v2_mp_tp1_pp4_to_tp2_pp2_dp"),
    # MLA configurations
    (2, 1, 2, 1, False, True, True, "v2_mp_tp2_pp1_to_tp2_pp1_dp_mla"),
    (2, 1, 2, 1, True, False, True, "v2_mp_tp2_pp1_dp_to_tp2_pp1_mla"),
]


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,gen_tp,gen_pp,ctx_enable_dp,gen_enable_dp,is_mla",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6]) for c in MP_TEST_CONFIGS],
    ids=[c[7] for c in MP_TEST_CONFIGS],
)
def test_v2_transceiver_mp(ctx_tp, ctx_pp, gen_tp, gen_pp, ctx_enable_dp, gen_enable_dp, is_mla):
    """Test PyNativeCacheTransceiver with multi-process configurations."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    run_v2_transceiver_mp(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        is_mla=is_mla,
    )
