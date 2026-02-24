"""Common utilities for distributed inference."""

import atexit
import os
import sys
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._utils import get_free_port

from ..utils.logger import ad_logger

# TODO: check to what extend we can reuse _torch/distributed.py


class _DistGroup:
    """Global instance to set/get the default process group for distributed ops."""

    def __init__(self):
        self._group = None

    def set(self, group):
        self._group = group

    def get(self):
        return self._group


DistGroup = _DistGroup()

ReduceOp = dist.ReduceOp


def all_gather(tensor_list, tensor, group=None, async_op=False):
    """Torch's all_gather with our default process group."""
    if group is None:
        group = DistGroup.get()
    return dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    """Torch's all_reduce with our default process group."""
    if group is None:
        group = DistGroup.get()
    return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)


def broadcast(tensor, src, group=None, async_op=False):
    """Torch's broadcast with our default process group."""
    if group is None:
        group = DistGroup.get()
    return dist.broadcast(tensor, src=src, group=group, async_op=async_op)


def broadcast_object_list(object_list, src=0, group=None, device=None):
    """Torch's broadcast_object_list with our default process group."""
    if group is None:
        group = DistGroup.get()
    return dist.broadcast_object_list(object_list, src=src, group=group, device=device)


def all_gather_object(object_list, object, group=None):
    """Torch's all_gather_object with our default process group."""
    if group is None:
        group = DistGroup.get()
    return dist.all_gather_object(object_list, object, group=group)


def get_world_size() -> int:
    return dist.get_world_size()


def get_rank() -> int:
    return dist.get_rank()


def get_rank_world_size() -> Tuple[int, int]:
    return get_rank(), get_world_size()


def initialize_or_skip(
    rank: int = 0,
    world_size: int = 1,
    port: Optional[int] = None,
    shared_port: Optional["mp.Value"] = None,
    port_ready_barrier: Optional["mp.Barrier"] = None,
) -> Tuple[int, int]:
    if not dist.is_initialized():
        return initialize(
            rank=rank,
            world_size=world_size,
            port=port,
            shared_port=shared_port,
            port_ready_barrier=port_ready_barrier,
        )
    return get_rank(), get_world_size()


def is_ompi():
    """Check whether multi-processing was initialized with explicitly calling mpirun."""
    return "OMPI_COMM_WORLD_SIZE" in os.environ


def is_torchelastic():
    """Check whether multi-processing was initialized with torchelastic."""
    return "TORCHELASTIC_RUN_ID" in os.environ


def is_initialized():
    return dist.is_initialized()


def cleanup():
    """Destroy process group when the program exits."""
    if dist.is_initialized():
        ad_logger.info("Destroying process group")
        dist.destroy_process_group()


def _set_distributed_env_vars(local_rank: int, world_size: int, port: int) -> None:
    """Set environment variables required by NCCL's env:// init method."""
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_RANK"] = str(local_rank)


def _try_init_process_group(local_rank: int, world_size: int, port: int) -> bool:
    """Attempt to initialize process group. Returns True on success, False on EADDRINUSE."""
    _set_distributed_env_vars(local_rank, world_size, port)

    try:
        dist.init_process_group(
            "nccl",
            world_size=world_size,
            rank=local_rank,
            device_id=torch.device(local_rank),
        )
        return True
    except Exception as e:
        # Check if this is a port-in-use error (only rank 0 binds, so only rank 0 can get this)
        if "EADDRINUSE" in str(e) or "address already in use" in str(e).lower():
            ad_logger.warning(f"Port {port} already in use, will retry with new port")
            return False
        raise


def initialize(
    rank: int = 0,
    world_size: int = 1,
    port: Optional[int] = None,
    shared_port: Optional["mp.Value"] = None,
    port_ready_barrier: Optional["mp.Barrier"] = None,
    max_retries: int = 5,
) -> Tuple[int, int]:
    """Initialize distributed process group.

    Args:
        rank: Process rank (ignored for OMPI/torchelastic).
        world_size: Total number of processes (ignored for OMPI/torchelastic).
        port: Initial port to try. If None, a free port will be selected.
        shared_port: Optional mp.Value for rank 0 to share the final port with other ranks.
        port_ready_barrier: Optional mp.Barrier to synchronize port selection.
        max_retries: Maximum number of port retry attempts for rank 0.
    """
    if is_ompi():
        lib = "OMPI"
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    elif is_torchelastic():
        lib = "TORCHELASTIC"
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        port = int(os.environ["MASTER_PORT"])
    else:
        lib = "MP"
        local_rank = rank

    if port is None:
        assert world_size == 1, "Port is required for world_size > 1."
        port = get_free_port()

    ad_logger.set_rank(local_rank)

    # Necessary to assign a device to each rank.
    torch.cuda.set_device(local_rank)

    # If we have shared port synchronization (multiprocess spawn mode)
    if shared_port is not None and port_ready_barrier is not None:
        if local_rank == 0:
            # Rank 0: try ports until one works, then share with other ranks
            init_success = False
            init_error = None
            try:
                for attempt in range(max_retries):
                    ad_logger.info(
                        f"Initializing for: {lib=}, {local_rank=}, {world_size=}, {port=} (attempt {attempt + 1})"
                    )
                    if _try_init_process_group(local_rank, world_size, port):
                        # Success! Share the working port with other ranks
                        shared_port.value = port
                        init_success = True
                        break
                    else:
                        # Port was taken, try a new one
                        port = get_free_port()
                else:
                    # All retries exhausted
                    init_error = RuntimeError(
                        f"Failed to find available port after {max_retries} attempts"
                    )
            except Exception as e:
                # Catch any unexpected error so we can still signal other ranks
                init_error = e
            finally:
                # ALWAYS signal other ranks, even on error, to prevent deadlock
                if not init_success:
                    shared_port.value = -1
                port_ready_barrier.wait()

            if init_error is not None:
                raise init_error
        else:
            # Other ranks: wait for rank 0 to find a working port
            port_ready_barrier.wait()
            port = shared_port.value
            if port == -1:
                raise RuntimeError("Rank 0 failed to initialize, cannot proceed")
            ad_logger.info(f"Initializing for: {lib=}, {local_rank=}, {world_size=}, {port=}")
            _set_distributed_env_vars(local_rank, world_size, port)
            dist.init_process_group(
                "nccl",
                world_size=world_size,
                rank=local_rank,
                device_id=torch.device(local_rank),
            )
    else:
        # Original path: no retry mechanism (OMPI, torchelastic, or single process)
        ad_logger.info(f"Initializing for: {lib=}, {local_rank=}, {world_size=}, {port=}")
        _set_distributed_env_vars(local_rank, world_size, port)
        dist.init_process_group(
            "nccl",
            world_size=world_size,
            rank=local_rank,
            device_id=torch.device(local_rank),
        )

    # Register cleanup function to be called at exit
    atexit.register(cleanup)

    # set a manual seed for reproducibility
    torch.manual_seed(1111)

    return local_rank, world_size


def init_and_run_process(
    job, rank, size, port, shared_port=None, port_ready_barrier=None, **kwargs
):
    try:
        initialize_or_skip(
            rank, size, port, shared_port=shared_port, port_ready_barrier=port_ready_barrier
        )
        job(rank, size, **kwargs)
    except Exception as e:
        # Close the input and output queues to parent process can exit.
        for q in ["input_queue", "output_queue"]:
            if q in kwargs and kwargs[q] is not None:
                kwargs[q].put(None)
                kwargs[q].close()
        raise e
    finally:
        # Make sure to clean up even if an exception occurs
        cleanup()


def _start_multiprocess_job(
    job: Callable[[int, int], None],
    size: Optional[int] = None,
    input_queues=None,
    output_queue=None,
    **kwargs,
):
    if not kwargs:
        kwargs = {}

    # check if it was called as simple python or with openmpi
    # if called with openmpi, we don't need to spawn multiple processes...
    if is_ompi():
        assert size is None, "Cannot set size when running with openmpi"
        rank, world_size = initialize_or_skip()
        if input_queues:
            kwargs["input_queue"] = input_queues[rank]
        if output_queue:
            kwargs["output_queue"] = output_queue if rank == 0 else None
        job(rank, world_size, **kwargs)
        # We don't have to join it.
        return None
    # retrieve world_size if it exists to overwrite the default size
    elif size is not None:
        pass
    else:
        size = 1

    port = get_free_port()

    if size == 1 and not input_queues and not output_queue:
        # If we don't provide the IO queues, we run the single GPU model in place.
        ad_logger.info("Launching the job in-place.")
        init_and_run_process(job, 0, 1, port, **kwargs)
        return None

    # Use explicit spawn context to ensure synchronization primitives work correctly
    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []

    # Create shared state for port synchronization with retry mechanism:
    # - shared_port: rank 0 writes the final working port here
    # - port_ready_barrier: all ranks wait here until rank 0 has bound successfully
    shared_port = ctx.Value("i", port)  # 'i' = signed int
    port_ready_barrier = ctx.Barrier(size)

    for rank in range(size):
        if input_queues:
            kwargs["input_queue"] = input_queues[rank]
        if output_queue:
            kwargs["output_queue"] = output_queue if rank == 0 else None

        p = ctx.Process(
            target=init_and_run_process,
            args=(job, rank, size, port),
            kwargs={**kwargs, "shared_port": shared_port, "port_ready_barrier": port_ready_barrier},
            daemon=True,
        )
        p.start()
        processes.append(p)

    return processes


def _join_multiprocess_job(processes):
    if not processes:
        ad_logger.warning("No valid processes")
        return

    for p in processes:
        p.join()

        # Ensure that all processes have exited successfully
        if isinstance(p, mp.Process):
            assert not p.exitcode, f"Process {p.pid} exited with code {p.exitcode}"


def spawn_multiprocess_job(job: Callable[[int, int], None], size: Optional[int] = None):
    processes = _start_multiprocess_job(job, size)
    if processes:
        _join_multiprocess_job(processes)
    cleanup()


class MultiProcessExecutor:
    """A simple multi process executor using queues for IO."""

    def __init__(self, job: Callable, world_size: int = 1, **kwargs):
        self.world_size = torch.cuda.device_count() if world_size == -1 else world_size
        mp.set_start_method("spawn", force=True)
        self.input_queues = []
        for _ in range(self.world_size):
            self.input_queues.append(mp.Queue())
        self.output_queue = mp.Queue()
        self.processes = _start_multiprocess_job(
            job, self.world_size, self.input_queues, self.output_queue, **kwargs
        )

        sys.excepthook = self.stop
        atexit.register(self.stop)

    def __del__(self):
        self.stop()

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs) -> mp.Queue:
        """Put args/kwargs into queues for all ranks and return the output queue as reference."""
        assert self.processes, "Terminated processes"

        for i in range(self.world_size):
            self.input_queues[i].put((args, kwargs))
        del args, kwargs
        return self.output_queue

    def stop(self, *args, **kwargs):
        if not getattr(self, "processes", None):
            return

        for i in range(self.world_size):
            self.input_queues[i].put(None)
        _join_multiprocess_job(self.processes)
        self.processes = None
        for q in self.input_queues:
            q.close()
            q.join_thread()
        self.output_queue.close()
        self.output_queue.join_thread()

        # Make sure all process groups are cleaned up
        if dist.is_initialized():
            dist.destroy_process_group()
