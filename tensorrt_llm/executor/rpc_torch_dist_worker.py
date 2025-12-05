import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin
from tensorrt_llm.logger import logger


class RpcTorchDistWorker(RpcWorkerMixin, BaseWorker):
    def __init__(
        self, rank: int, world_size: int, device_id: int, rpc_addr: Optional[str] = None, **kwargs
    ):
        # Initialize BaseWorker
        super().__init__(**kwargs)

        self.rank = rank
        self.global_rank = rank
        self.world_size = world_size
        self.device_id = device_id

        # Create control group for worker orchestration
        # Use Gloo for control messages as it doesn't require GPU
        # and is robust.
        self.control_group = dist.new_group(backend="gloo")

        if self.rank == 0:
            if rpc_addr is None:
                raise ValueError("rpc_addr must be provided for rank 0")
            self.init_rpc_worker(self.rank, rpc_addr)
            self.start_rpc_server()

    def setup_engine(self):
        # Broadcast command if rank 0
        if self.rank == 0:
            self._broadcast_command("setup_engine")

        # Ensure we are synchronized before setting up engine if needed
        if dist.is_initialized():
            dist.barrier()

        super().setup_engine()

    def start(self):
        pass

    def shutdown(self):
        if self.doing_shutdown:
            return

        # Broadcast command if rank 0
        if self.rank == 0:
            try:
                self._broadcast_command("shutdown")
            except Exception as e:
                logger.warning(f"Failed to broadcast shutdown command: {e}")

        super().shutdown()

        if self.rank == 0 and hasattr(self, "rpc_server") and self.rpc_server:
            self.rpc_server.shutdown()

    def _broadcast_command(self, command: str, args: Any = None):
        if not dist.is_initialized():
            return
        cmd_list = [command, args]
        try:
            dist.broadcast_object_list(cmd_list, src=0, group=self.control_group)
        except Exception as e:
            logger.error(f"Broadcast error: {e}")

    @classmethod
    def worker_main(
        cls,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: str,
        rpc_addr: Optional[str],
        worker_kwargs: Dict,
    ):
        # Setup environment
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["TLLM_DISABLE_MPI"] = "1"

        # Setup device
        if torch.cuda.is_available():
            device_id = rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
        else:
            device_id = 0

        # Initialize process group
        # Use nccl for GPU, gloo for CPU
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        logger.info(f"Worker {rank}/{world_size} initialized with backend {backend}")

        try:
            worker = cls(
                rank=rank,
                world_size=world_size,
                device_id=device_id,
                rpc_addr=rpc_addr,
                **worker_kwargs,
            )

            if rank == 0:
                # Rank 0 waits for RPCs.
                # The RPC server runs in a background thread started by start_rpc_server.
                # We wait on the shutdown event which is set by shutdown() method (called via RPC).
                worker.shutdown_event.wait()
            else:
                # Rank > 0 command loop
                while True:
                    cmd_list = [None, None]
                    try:
                        dist.broadcast_object_list(cmd_list, src=0, group=worker.control_group)
                    except Exception as e:
                        # If broadcast fails (e.g. rank 0 died), we should exit
                        logger.error(f"Rank {rank} broadcast receive error: {e}")
                        break

                    cmd, args = cmd_list
                    # logger.debug(f"Rank {rank} received command: {cmd}")

                    if cmd == "setup_engine":
                        worker.setup_engine()
                    elif cmd == "shutdown":
                        worker.shutdown()
                        break
                    elif cmd == "report_device_id":
                        # Optional: handle other commands if needed
                        pass
                    else:
                        logger.warning(f"Rank {rank} received unknown command: {cmd}")

        except Exception as e:
            logger.error(f"Worker {rank} failed with error: {e}")
            raise e
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
