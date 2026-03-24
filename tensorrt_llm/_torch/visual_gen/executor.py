import os
import queue
import threading
import traceback
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.distributed as dist
import zmq

from tensorrt_llm._torch.visual_gen.config import VisualGenArgs
from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.logger import logger


@dataclass
class DiffusionRequest:
    """Request for diffusion inference with explicit model-specific parameters."""

    request_id: int
    prompt: List[str]
    negative_prompt: Optional[str] = None
    height: int = 720
    width: int = 1280
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    max_sequence_length: int = 512
    seed: int = 42

    # Video-specific parameters
    num_frames: int = 81
    frame_rate: float = 24.0

    # Image-specific parameters
    num_images_per_prompt: int = 1

    # Advanced parameters
    guidance_rescale: float = 0.0
    output_type: str = "pt"

    # LTX-2 multi-modal guidance (STG / modality guidance)
    stg_scale: float = 0.0
    stg_blocks: Optional[List[int]] = None
    modality_scale: float = 1.0
    rescale_scale: float = 0.0
    guidance_skip_step: int = 0
    enhance_prompt: bool = False

    # Image-to-video parameters
    image: Optional[Union[str, List[str]]] = None
    image_cond_strength: float = 1.0

    # Wan-specific parameters
    guidance_scale_2: Optional[float] = None
    boundary_ratio: Optional[float] = None
    last_image: Optional[Union[str, List[str]]] = None


@dataclass
class DiffusionResponse:
    """Response with model-specific output.

    Attributes:
        request_id: Unique identifier for the request
        output: Generated media as MediaOutput with model-specific fields populated
        error_msg: Error message if generation failed
    """

    request_id: int
    output: Optional[MediaOutput] = None
    error_msg: Optional[str] = None


class DiffusionExecutor:
    """Execution engine for diffusion models running in worker processes."""

    def __init__(
        self,
        request_queue_addr: str,
        response_queue_addr: str,
        device_id: int,
        diffusion_args: "VisualGenArgs",
    ):
        self.request_queue_addr = request_queue_addr
        self.response_queue_addr = response_queue_addr
        self.device_id = device_id
        self.diffusion_args = diffusion_args

        self.pipeline = None  # initialized in _load_pipeline
        self.requests_ipc = None
        self.rank = dist.get_rank()
        self.response_queue = queue.Queue()
        self.sender_thread = None

        # Only rank 0 handles IPC
        if self.rank == 0:
            logger.info(f"Worker {device_id}: Connecting to request queue")
            self.requests_ipc = ZeroMqQueue(
                (request_queue_addr, None),
                is_server=False,
                socket_type=zmq.PULL,
                use_hmac_encryption=False,
            )
            self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self.sender_thread.start()

        self._load_pipeline()

    def _sender_loop(self):
        """Background thread for sending responses."""
        logger.info(f"Worker {self.device_id}: Connecting to response queue")
        responses_ipc = ZeroMqQueue(
            (self.response_queue_addr, None),
            is_server=False,
            socket_type=zmq.PUSH,
            use_hmac_encryption=False,
        )

        while True:
            try:
                resp = self.response_queue.get()
                if resp is None:
                    break
                responses_ipc.put(resp)
            except Exception as e:
                logger.error(f"Worker {self.device_id}: Sender error: {e}")

        if responses_ipc.socket:
            responses_ipc.socket.setsockopt(zmq.LINGER, 0)
        responses_ipc.close()

    def _load_pipeline(self):
        """
        Load pipeline using proper flow:
        VisualGenArgs → PipelineLoader → DiffusionModelConfig → AutoPipeline → BasePipeline
        """
        logger.info(f"Worker {self.device_id}: Loading pipeline")

        try:
            args = self.diffusion_args.model_copy(update={"device": f"cuda:{self.device_id}"})

            loader = PipelineLoader(args)
            self.pipeline = loader.load(skip_warmup=args.skip_warmup)

        except Exception as e:
            logger.error(f"Worker {self.device_id}: Failed to load pipeline: {e}")
            raise

        logger.info(f"Worker {self.device_id}: Pipeline ready")

        # Sync all workers
        dist.barrier()

        # Send READY signal
        if self.rank == 0:
            logger.info(f"Worker {self.device_id}: Sending READY")
            self.response_queue.put(DiffusionResponse(request_id=-1, output="READY"))

    def serve_forever(self):
        """Main execution loop."""
        while True:
            req = None
            if self.rank == 0:
                req = self.requests_ipc.get()
                logger.info(f"Worker {self.device_id}: Request available")

            # Broadcast to all ranks
            obj_list = [req]
            dist.broadcast_object_list(obj_list, src=0)
            req = obj_list[0]

            if req is None:
                logger.info(f"Worker {self.device_id}: Shutdown signal received")
                if self.rank == 0 and self.sender_thread:
                    self.response_queue.put(None)
                    self.sender_thread.join()
                break

            logger.info(f"Worker {self.device_id}: Processing request {req.request_id}")
            self.process_request(req)

    def process_request(self, req: DiffusionRequest):
        """Process a single request."""
        if (
            self.pipeline._warmed_up_shapes
            and (req.height, req.width, req.num_frames) not in self.pipeline._warmed_up_shapes
        ):
            logger.warning(
                f"Requested shape (height={req.height}, width={req.width}, "
                f"num_frames={req.num_frames}) "
                f"was not warmed up. First request with this shape will be slower due to "
                f"torch.compile recompilation or CUDA graph capture."
                f"Warmed-up shapes: {self.pipeline._warmed_up_shapes}"
            )
        try:
            output = self.pipeline.infer(req)
            if self.rank == 0:
                self.response_queue.put(DiffusionResponse(request_id=req.request_id, output=output))
        except Exception as e:
            logger.error(f"Worker {self.device_id}: Error: {e}")
            logger.error(traceback.format_exc())
            if self.rank == 0:
                self.response_queue.put(
                    DiffusionResponse(request_id=req.request_id, error_msg=str(e))
                )


def run_diffusion_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    request_queue_addr: str,
    response_queue_addr: str,
    diffusion_args: "VisualGenArgs",
    log_level: str = "info",
):
    """Entry point for worker process."""
    try:
        # Set log level before any other work so loading logs are visible
        logger.set_level(log_level)

        # Setup distributed env — use PyTorch distributed, not MPI
        os.environ["TLLM_DISABLE_MPI"] = "1"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Runtime check: parallel config vs actual world size
        diffusion_args.parallel.validate_world_size(world_size)

        # Calculate device_id before init_process_group
        device_id = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else None,
        )

        executor = DiffusionExecutor(
            request_queue_addr=request_queue_addr,
            response_queue_addr=response_queue_addr,
            device_id=device_id,
            diffusion_args=diffusion_args,
        )
        executor.serve_forever()
        if executor.pipeline is not None:
            executor.pipeline.cleanup()
        dist.destroy_process_group()

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        traceback.print_exc()
