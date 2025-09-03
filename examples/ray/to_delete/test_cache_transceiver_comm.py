import os
import sys
import time

import torch
import torch.distributed as dist


def _run_distributed_worker():
    import importlib
    tllm_bindings = importlib.import_module("tensorrt_llm.bindings")

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
    )
    try:
        world_pg = torch.distributed.group.WORLD
        try:
            cacheComm = getattr(tllm_bindings, "CacheTransceiverComm")
        except AttributeError:
            # In current bindings, CacheTransceiverComm is registered under
            # tensorrt_llm.bindings.internal.batch_manager
            bm = importlib.import_module(
                "tensorrt_llm.bindings.internal.batch_manager")
            cacheComm = getattr(bm, "CacheTransceiverComm")
        comm = cacheComm(world_pg.boxed())

        # Split into 2 subgroups by parity of world rank
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        color = rank // 2
        key = rank % 2
        print(f"[rank {rank}: color: {color}, key: {key}]")
        sub = comm.split(color, key)

        # Validate subgroup size
        expected_group_size = 2
        assert sub.get_size() == expected_group_size

        # allgather scalar: gather world ranks in subgroup order (sorted by key)
        ok, gathered_ranks = sub.allgather(rank)
        assert ok is True
        expected_world_ranks = [
            r for r in range(world_size) if (r // 2) == color
        ]
        print(
            f"[rank {rank}: gathered_ranks: {gathered_ranks}, expected_world_ranks: {expected_world_ranks}]"
        )
        assert gathered_ranks == expected_world_ranks

        # allgatherv: variable-sized vectors per rank
        # Define local payload as [world_rank] * (world_rank + 1)
        local_len = rank + 1
        payload = [rank] * local_len

        # First collect sizes across subgroup
        ok_sizes, sizes64 = sub.allgather(local_len)
        assert ok_sizes is True
        sizes = [int(x) for x in sizes64]
        print(f"[rank {rank}: sizes: {sizes}]")

        ok_v, out = sub.allgatherv(payload, sizes)
        assert ok_v is True

        expected_concat = []
        for r in expected_world_ranks:
            expected_concat.extend([r] * (r + 1))
        print(f"[rank {rank}: out: {out}, expected_concat: {expected_concat}]")
        assert out == expected_concat

        # Test allgatherv with char: use ASCII characters (A=65, B=66, etc.)
        char_payload = [chr(65 + rank)
                        ] * local_len  # Convert rank to ASCII char
        ok_char, char_out = sub.allgatherv(char_payload, sizes)
        assert ok_char is True

        expected_char_concat = []
        for r in expected_world_ranks:
            expected_char_concat.extend([chr(65 + r)] * (r + 1))
        print(
            f"[rank {rank}: char_out: {char_out}, expected_char_concat: {expected_char_concat}]"
        )
        assert char_out == expected_char_concat

    finally:
        # Give time for async backend cleanup on lower end machines
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        time.sleep(0.1)


def _main():
    # Ensure torch.distributed and Gloo backend are available
    if not dist.is_available():
        print(
            "ERROR: torch.distributed is not available in this PyTorch build.",
            file=sys.stderr)
        sys.exit(1)

    backends = getattr(torch.distributed, "is_gloo_available", None)
    if callable(backends) and not torch.distributed.is_gloo_available():
        print("ERROR: Gloo backend is not available in this PyTorch build.",
              file=sys.stderr)
        sys.exit(1)

    # Ensure TRT-LLM Python bindings are importable before running workers
    import importlib
    try:
        importlib.import_module("tensorrt_llm.bindings")
    except Exception as exc:
        print(f"ERROR: TRT-LLM bindings not importable: {exc}", file=sys.stderr)
        sys.exit(1)

    # Require torchrun environment
    world_size_env = os.environ.get("WORLD_SIZE")
    rank_env = os.environ.get("RANK")
    if not world_size_env or not rank_env:
        print(
            "ERROR: This script must be launched with torchrun. Example:\n"
            "  torchrun --nproc_per_node=4 test_cache_transceiver_comm.py",
            file=sys.stderr,
        )
        sys.exit(2)

    world_size = int(world_size_env)
    if world_size < 2 or (world_size % 2) != 0:
        print("ERROR: WORLD_SIZE must be an even number >= 2 for this test.",
              file=sys.stderr)
        sys.exit(2)

    _run_distributed_worker()

    # Only rank 0 prints success message
    try:
        if dist.get_rank() == 0:
            print("CacheTransceiverComm split and collectives: OK")
    except Exception:
        pass


if __name__ == "__main__":
    _main()
