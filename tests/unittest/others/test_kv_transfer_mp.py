import os
import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.base.kv_transfer import KVSlice, SessionStatus
from tensorrt_llm._torch.disaggregation.native.aux_buffer import AuxBuffer
from tensorrt_llm._torch.disaggregation.native.kv_transfer import TransferWorker
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType


def broadcast_string(s: str | None, src: int, group: dist.ProcessGroup | None = None) -> str:
    """Broadcast a string from src rank to all other ranks in the group."""
    # Use CPU tensors for gloo backend compatibility
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

    # Use CPU tensors for gloo backend compatibility
    # First gather lengths
    encoded = s.encode("utf-8")
    local_length = torch.tensor([len(encoded)], dtype=torch.long)
    all_lengths = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_lengths, local_length, group=group)

    # Then gather data
    max_length = max(length.item() for length in all_lengths)
    local_data = torch.zeros(max_length, dtype=torch.uint8)
    local_data[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)

    all_data = [torch.zeros(max_length, dtype=torch.uint8) for _ in range(world_size)]
    dist.all_gather(all_data, local_data, group=group)

    # Decode
    results = []
    for data, length in zip(all_data, all_lengths):
        results.append(bytes(data[: length.item()].tolist()).decode("utf-8"))

    return results


def allgather_int(value: int, group: dist.ProcessGroup | None = None) -> list[int]:
    """Allgather integers from all ranks in the group."""
    # Use CPU tensors for gloo backend compatibility
    world_size = dist.get_world_size(group)
    local_tensor = torch.tensor([value], dtype=torch.long)
    all_tensors = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_tensors, local_tensor, group=group)
    return [t.item() for t in all_tensors]


def broadcast_int(value: int | None, src: int, group: dist.ProcessGroup | None = None) -> int:
    """Broadcast an int64 from src rank to all other ranks in the group."""
    # Use CPU tensors for gloo backend compatibility
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


def worker_fn(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
):
    """Worker function for each process."""
    # Set environment variables for distributed
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize distributed (use gloo for single GPU compatibility)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

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
    meta_max_batch_size = 32
    beam_width = 1
    max_draft_len = 4
    num_layers = 4
    head_dim = 128
    num_kv_heads = 4
    tokens_per_block = 8
    max_seq_len = 256
    max_batch_size = 4
    dtype = DataType.FLOAT
    request_len = 16

    ctx_instance_name = "ctx_instance"
    gen_instance_name = "gen_instance"

    if is_ctx:
        # Create ctx mapping
        _ = local_rank // ctx_tp
        tp_rank = local_rank % ctx_tp
        mapping = Mapping(
            world_size=ctx_instance_num,
            rank=local_rank,
            tp_size=ctx_tp,
            pp_size=ctx_pp,
        )

        # Create KVCacheManager
        kv_cache_manager = KVCacheManager(
            trtllm.KvCacheConfig(max_tokens=2048, enable_block_reuse=False),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
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
        block_data_pool = kv_cache_manager.get_unique_primary_pool()
        random_values = torch.rand(
            block_data_pool.shape, dtype=torch.float32, device=block_data_pool.device
        )
        block_data_pool.copy_(random_values)

        # Create TransferWorker
        aux_buffer = AuxBuffer(meta_max_batch_size, beam_width, max_draft_len)
        transfer_worker = TransferWorker(
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
            device_id=device_id,
            instance_name=ctx_instance_name,
            aux_buffer=aux_buffer,
        )

        # Get local endpoint
        local_endpoint = transfer_worker._sender.endpoint
        local_layer_num = len(kv_cache_manager.pp_layers)

        # Allgather endpoints within ctx group
        ctx_endpoints = allgather_strings(local_endpoint, group=ctx_group)

        # Allgather layer_num_per_pp (only from pp_rank's first tp_rank)
        layer_num_for_gather = local_layer_num if tp_rank == 0 else 0
        all_layer_nums = allgather_int(layer_num_for_gather, group=ctx_group)

        # Extract layer_num_per_pp (one per pp_rank)
        ctx_layer_num_per_pp = []
        for pp in range(ctx_pp):
            ctx_layer_num_per_pp.append(all_layer_nums[pp * ctx_tp])

        # Update instance info
        transfer_worker.populate_instance_and_rank_info(
            endpoints=ctx_endpoints, layer_num_per_pp=ctx_layer_num_per_pp
        )

        # Get ctx_info_endpoint (only rank 0 has the real value)
        if local_rank == 0:
            ctx_info_endpoint = transfer_worker._instance_info_server.endpoint
        else:
            ctx_info_endpoint = None

    else:  # gen process
        # Create gen mapping
        _ = local_rank // gen_tp  # pp_rank (unused)
        tp_rank = local_rank % gen_tp
        mapping = Mapping(
            world_size=gen_instance_num,
            rank=local_rank,
            tp_size=gen_tp,
            pp_size=gen_pp,
        )

        # Create KVCacheManager
        kv_cache_manager = KVCacheManager(
            trtllm.KvCacheConfig(max_tokens=2048, enable_block_reuse=False),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
        )

        # Create TransferWorker
        aux_buffer = AuxBuffer(meta_max_batch_size, beam_width, max_draft_len)
        transfer_worker = TransferWorker(
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
            device_id=device_id,
            instance_name=gen_instance_name,
            aux_buffer=aux_buffer,
        )

        # Get local endpoint
        local_endpoint = transfer_worker._sender.endpoint
        local_layer_num = len(kv_cache_manager.pp_layers)

        # Allgather endpoints within gen group
        gen_endpoints = allgather_strings(local_endpoint, group=gen_group)

        # Allgather layer_num_per_pp
        layer_num_for_gather = local_layer_num if tp_rank == 0 else 0
        all_layer_nums = allgather_int(layer_num_for_gather, group=gen_group)

        gen_layer_num_per_pp = []
        for pp in range(gen_pp):
            gen_layer_num_per_pp.append(all_layer_nums[pp * gen_tp])

        # Update instance info
        transfer_worker.populate_instance_and_rank_info(
            endpoints=gen_endpoints, layer_num_per_pp=gen_layer_num_per_pp
        )

        ctx_info_endpoint = None  # Will be received later

    # Broadcast ctx_info_endpoint from ctx rank 0 to gen ranks
    if rank in [0] + gen_ranks:
        ctx_info_endpoint = broadcast_string(ctx_info_endpoint, src=0, group=ctx_to_gen_group)

    # Broadcast gen_layer_num_per_pp from gen rank 0 to world rank 0
    # gen rank 0 is world rank ctx_instance_num
    # Use CPU tensors for gloo backend compatibility
    if rank in [0] + gen_ranks:
        if rank == ctx_instance_num:
            # gen rank 0 sends
            gen_layer_num_per_pp_tensor = torch.tensor(gen_layer_num_per_pp, dtype=torch.long)
        else:
            gen_layer_num_per_pp_tensor = torch.zeros(gen_pp, dtype=torch.long)
        dist.broadcast(gen_layer_num_per_pp_tensor, src=ctx_instance_num, group=ctx_to_gen_group)
        if rank == 0:
            gen_layer_num_per_pp = gen_layer_num_per_pp_tensor.tolist()

    # Synchronize all processes
    dist.barrier()

    # Helper function to process and verify a single request
    def process_and_verify_request(
        ctx_request_id: int, gen_request_id: int, req_len: int, unique_rid: int
    ):
        """Process a single request and verify the transfer."""
        sampling_params = SamplingParams()

        if is_ctx:
            # Create ctx request
            ctx_request = LlmRequest(
                request_id=ctx_request_id,
                max_new_tokens=1,
                input_tokens=list(range(req_len)),
                sampling_config=tensorrt_llm.bindings.SamplingConfig(
                    sampling_params._get_sampling_config()
                ),
                is_streaming=False,
                llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
            )
            ctx_request.py_disaggregated_params = DisaggregatedParams(disagg_request_id=unique_rid)

            # Add sequence to KVCacheManager
            kv_cache_manager.impl.add_sequence(
                ctx_request.py_request_id, ctx_request.prompt_len, 1, ctx_request
            )

            # Create sender session
            sender_session = transfer_worker.create_tx_session(ctx_request)

            # Get block ids and send
            block_ids = kv_cache_manager.get_batch_cache_indices([ctx_request.py_request_id])[0]
            send_kv_slice = KVSlice(is_last_slice=True, block_ids=block_ids)
            send_slice_task = sender_session._kv_tasks[sender_session.send(send_kv_slice)]

            # Wait for send to complete
            send_slice_task.future.result()
            assert sender_session.state.status == SessionStatus.TRANSFERRED

            # Get block data for verification
            block_data = kv_cache_manager.get_unique_primary_pool()[block_ids]

        else:  # gen process
            # Create gen request
            gen_request = LlmRequest(
                request_id=gen_request_id,
                max_new_tokens=1,
                input_tokens=list(range(req_len)),
                sampling_config=tensorrt_llm.bindings.SamplingConfig(
                    sampling_params._get_sampling_config()
                ),
                is_streaming=False,
                llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
            )
            gen_request.py_disaggregated_params = DisaggregatedParams(
                ctx_request_id=ctx_request_id,
                ctx_dp_rank=0,
                ctx_info_endpoint=ctx_info_endpoint,
                disagg_request_id=unique_rid,
            )

            # Add sequence to KVCacheManager
            kv_cache_manager.impl.add_sequence(
                gen_request.py_request_id, gen_request.prompt_len, 1, gen_request
            )

            # Create receiver session
            receiver_session = transfer_worker.create_rx_session(gen_request)

            # Get block ids and receive
            block_ids = kv_cache_manager.get_batch_cache_indices([gen_request.py_request_id])[0]
            recv_kv_slice = KVSlice(is_last_slice=True, block_ids=block_ids)
            recv_slice_task = receiver_session._kv_tasks[receiver_session.receive(recv_kv_slice)]

            # Wait for receive to complete
            recv_slice_task.future.result()
            assert receiver_session.state.status == SessionStatus.TRANSFERRED

            # Get block data for verification
            block_data = kv_cache_manager.get_unique_primary_pool()[block_ids]

        # Synchronize before verification
        dist.barrier()

        print(
            f"[Rank {rank}] Request {ctx_request_id}: "
            f"{'CTX' if is_ctx else 'GEN'} finished. Block data shape: {block_data.shape}"
        )

        # ===== Gather block_data for verification =====
        # Use CPU tensors for gloo backend compatibility
        block_data_cpu = block_data.cpu()

        # Gather ctx block_data to ctx local rank 0 (world rank 0)
        ctx_block_datas = None
        gen_block_datas = None

        if is_ctx:
            # Gather within ctx group to ctx local rank 0
            if local_rank == 0:
                ctx_block_datas_cpu = [
                    torch.zeros_like(block_data_cpu) for _ in range(ctx_instance_num)
                ]
            else:
                ctx_block_datas_cpu = None
            # Use group_dst for group-local rank
            dist.gather(block_data_cpu, ctx_block_datas_cpu, group_dst=0, group=ctx_group)
            if local_rank == 0:
                ctx_block_datas = [d.cuda() for d in ctx_block_datas_cpu]
        else:
            # Gather within gen group to gen local rank 0 (world rank ctx_instance_num)
            if local_rank == 0:
                gen_block_datas_cpu = [
                    torch.zeros_like(block_data_cpu) for _ in range(gen_instance_num)
                ]
            else:
                gen_block_datas_cpu = None
            # Use group_dst for group-local rank
            dist.gather(block_data_cpu, gen_block_datas_cpu, group_dst=0, group=gen_group)
            if local_rank == 0:
                gen_block_datas = [d.cuda() for d in gen_block_datas_cpu]

        dist.barrier()

        # Send gen_block_datas from gen rank 0 to world rank 0
        gen_rank_0 = ctx_instance_num
        if rank == gen_rank_0:
            # Send shape first
            shape_tensor = torch.tensor(list(block_data.shape), dtype=torch.long)
            dist.send(shape_tensor, dst=0)
            # Send each block_data (as CPU tensors)
            for data in gen_block_datas:
                dist.send(data.cpu().contiguous(), dst=0)
        elif rank == 0:
            # Receive shape
            shape_tensor = torch.zeros(4, dtype=torch.long)
            dist.recv(shape_tensor, src=gen_rank_0)
            gen_shape = tuple(shape_tensor.tolist())
            # Receive each block_data
            gen_block_datas = []
            for _ in range(gen_instance_num):
                data_cpu = torch.zeros(gen_shape, dtype=block_data.dtype)
                dist.recv(data_cpu, src=gen_rank_0)
                gen_block_datas.append(data_cpu.cuda())

        dist.barrier()

        # Verification on rank 0
        if rank == 0:
            print(f"\n[Rank {rank}] Request {ctx_request_id}: Starting verification...")

            # Merge ctx block data
            # shape [block_num, layer_num, 2, block_size]
            ctx_block_data_merge = torch.zeros(
                size=(
                    ctx_block_datas[0].shape[0],
                    num_layers,
                    2,
                    ctx_block_datas[0].shape[3] * ctx_tp,
                ),
                device="cuda",
            )
            for pp_idx in range(ctx_pp):
                for tp_idx in range(ctx_tp):
                    layer_start_idx = sum(ctx_layer_num_per_pp[:pp_idx])
                    layer_end_idx = layer_start_idx + ctx_layer_num_per_pp[pp_idx]
                    head_dim_per_rank = num_kv_heads // ctx_tp * head_dim * tokens_per_block
                    start_head_offset = tp_idx * head_dim_per_rank
                    end_head_offset = start_head_offset + head_dim_per_rank
                    block_idx = pp_idx * ctx_tp + tp_idx
                    ctx_block_data_merge[
                        :,
                        layer_start_idx:layer_end_idx,
                        :,
                        start_head_offset:end_head_offset,
                    ] = ctx_block_datas[block_idx]

            # Merge gen block data
            gen_block_data_merge = torch.zeros(
                size=(
                    gen_block_datas[0].shape[0],
                    num_layers,
                    2,
                    gen_block_datas[0].shape[3] * gen_tp,
                ),
                device="cuda",
            )
            for pp_idx in range(gen_pp):
                for tp_idx in range(gen_tp):
                    layer_start_idx = sum(gen_layer_num_per_pp[:pp_idx])
                    layer_end_idx = layer_start_idx + gen_layer_num_per_pp[pp_idx]
                    head_dim_per_rank = num_kv_heads // gen_tp * head_dim * tokens_per_block
                    start_head_offset = tp_idx * head_dim_per_rank
                    end_head_offset = start_head_offset + head_dim_per_rank
                    block_idx = pp_idx * gen_tp + tp_idx
                    gen_block_data_merge[
                        :,
                        layer_start_idx:layer_end_idx,
                        :,
                        start_head_offset:end_head_offset,
                    ] = gen_block_datas[block_idx]

            # Verify
            assert ctx_block_data_merge.equal(gen_block_data_merge), (
                f"Request {ctx_request_id}: Data mismatch!"
            )
            print(
                f"[Rank {rank}] Request {ctx_request_id}: Verification PASSED! "
                f"ctx shape: {ctx_block_data_merge.shape}, "
                f"gen shape: {gen_block_data_merge.shape}"
            )

        dist.barrier()

    # ===== Process multiple requests =====
    # Request 1
    unique_rid_1 = broadcast_int(random.getrandbits(63) if rank == 0 else None, src=0)
    process_and_verify_request(
        ctx_request_id=0, gen_request_id=1, req_len=request_len, unique_rid=unique_rid_1
    )

    # Request 2 (with different length)
    unique_rid_2 = broadcast_int(random.getrandbits(63) if rank == 0 else None, src=0)
    process_and_verify_request(
        ctx_request_id=2, gen_request_id=3, req_len=request_len * 2, unique_rid=unique_rid_2
    )

    if rank == 0:
        print(f"\n[Rank {rank}] All requests verified successfully!")

    # Cleanup
    dist.destroy_process_group()


def run_transfer_worker_mp(ctx_tp: int, ctx_pp: int, gen_tp: int, gen_pp: int):
    """Multi-process test for TransferWorker using mp.spawn."""
    world_size = ctx_tp * ctx_pp + gen_tp * gen_pp

    master_addr = "127.0.0.1"
    master_port = find_free_port()

    print(
        f"Starting {world_size} processes for test: "
        f"ctx_tp={ctx_tp}, ctx_pp={ctx_pp}, gen_tp={gen_tp}, gen_pp={gen_pp}"
    )

    # Use mp.spawn to start all processes
    mp.spawn(
        worker_fn,
        args=(world_size, master_addr, master_port, ctx_tp, ctx_pp, gen_tp, gen_pp),
        nprocs=world_size,
        join=True,
    )

    print(f"Test passed: ctx_tp={ctx_tp}, ctx_pp={ctx_pp}, gen_tp={gen_tp}, gen_pp={gen_pp}\n")


# Test configurations as pytest parameters
MP_TEST_CONFIGS = [
    # (ctx_tp, ctx_pp, gen_tp, gen_pp, test_id)
    (1, 1, 1, 1, "mp_tp1_pp1_to_tp1_pp1"),
    (1, 1, 1, 2, "mp_tp1_pp1_to_tp1_pp2"),
    (2, 1, 1, 2, "mp_tp2_pp1_to_tp1_pp2"),
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,gen_tp,gen_pp",
    [(c[0], c[1], c[2], c[3]) for c in MP_TEST_CONFIGS],
    ids=[c[4] for c in MP_TEST_CONFIGS],
)
def test_transfer_worker_mp(ctx_tp, ctx_pp, gen_tp, gen_pp):
    """Test transfer worker with multi-process configurations."""
    # Set multiprocessing start method
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    run_transfer_worker_mp(ctx_tp=ctx_tp, ctx_pp=ctx_pp, gen_tp=gen_tp, gen_pp=gen_pp)
