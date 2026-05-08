import os
import queue
import threading
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.distributed as dist
import zmq

from tensorrt_llm._torch.visual_gen.config import VisualGenArgs
from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.params import VisualGenParams


@dataclass
class DiffusionRequest:
    """Request for diffusion inference.

    Generation parameters live in the optional ``params`` object
    (a :class:`~tensorrt_llm.visual_gen.params.VisualGenParams` instance).
    When ``params`` is ``None`` (the default), the executor creates a
    ``VisualGenParams()`` and fills it with pipeline-specific defaults
    before calling ``pipeline.infer()``.
    """

    request_id: int
    prompt: List[str]
    params: Optional["VisualGenParams"] = None


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


# Python type name → accepted Python types for ExtraParamSchema validation.
_TYPE_MAP = {
    "float": (float, int),
    "int": (int,),
    "bool": (bool,),
    "str": (str,),
    "list": (list,),
}

# Generation config fields that pipelines declare defaults for.
# If a user sets one of these but the pipeline doesn't declare it in
# default_generation_params, the value will be silently ignored.
# Conditioning inputs (image, negative_prompt, mask, image_cond_strength)
# are excluded — they are validated at runtime by the pipeline's infer().
_GENERATION_CONFIG_FIELDS = frozenset(
    {
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "max_sequence_length",
        "num_frames",
        "frame_rate",
    }
)


class DiffusionExecutor:
    """Execution engine for diffusion models running in worker processes."""

    def __init__(
        self,
        request_queue_addr: str,
        response_queue_addr: str,
        device_id: int,
        diffusion_args: "VisualGenArgs",
        req_hmac_key: Optional[bytes] = None,
        resp_hmac_key: Optional[bytes] = None,
    ):
        self.request_queue_addr = request_queue_addr
        self.response_queue_addr = response_queue_addr
        self.device_id = device_id
        self.diffusion_args = diffusion_args
        self.resp_hmac_key = resp_hmac_key

        self.pipeline = None  # initialized in _load_pipeline
        self.requests_ipc = None
        self.rank = dist.get_rank()
        self.response_queue = queue.Queue()
        self.sender_thread = None

        # Only rank 0 handles IPC
        if self.rank == 0:
            logger.info(f"Worker {device_id}: Connecting to request queue")
            self.requests_ipc = ZeroMqQueue(
                (request_queue_addr, req_hmac_key),
                is_server=False,
                socket_type=zmq.PULL,
                use_hmac_encryption=True,
            )
            self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self.sender_thread.start()

        self._load_pipeline()

    def _sender_loop(self):
        """Background thread for sending responses."""
        logger.info(f"Worker {self.device_id}: Connecting to response queue")
        responses_ipc = ZeroMqQueue(
            (self.response_queue_addr, self.resp_hmac_key),
            is_server=False,
            socket_type=zmq.PUSH,
            use_hmac_encryption=True,
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

        # Send READY signal with pipeline metadata for the client.
        if self.rank == 0:
            logger.info(f"Worker {self.device_id}: Sending READY")
            self.response_queue.put(
                DiffusionResponse(
                    request_id=-1,
                    output={
                        "status": "READY",
                        "default_generation_params": self.pipeline.default_generation_params,
                        "extra_param_specs": self.pipeline.extra_param_specs,
                    },
                )
            )

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

    def _merge_defaults(self, req: DiffusionRequest):
        """Fill ``None`` fields in *req.params* with pipeline-specific defaults.

        Merges both universal defaults (from ``default_generation_params``)
        and extra_param defaults (from ``extra_param_specs``).
        """
        if req.params is None:
            from tensorrt_llm.visual_gen.params import VisualGenParams

            kwargs = dict(self.pipeline.default_generation_params)
            specs = self.pipeline.extra_param_specs
            if specs:
                kwargs["extra_params"] = {key: spec.default for key, spec in specs.items()}
            req.params = VisualGenParams(**kwargs)
            return

        params = req.params
        # Universal field defaults
        for field_name, default_value in self.pipeline.default_generation_params.items():
            if hasattr(params, field_name) and getattr(params, field_name) is None:
                setattr(params, field_name, default_value)

        # Extra param defaults — fill all declared keys so infer() can use direct access
        specs = self.pipeline.extra_param_specs
        if specs:
            if params.extra_params is None:
                params.extra_params = {}
            for key, spec in specs.items():
                if key not in params.extra_params:
                    params.extra_params[key] = spec.default

        self._validate_request(req)

    def _validate_request(self, req: DiffusionRequest):
        """Validate *req.params* against the loaded pipeline's declared parameters.

        Raises ``VisualGenParamsError`` on:
        - Unknown ``extra_params`` keys
        - Universal fields (e.g. ``num_frames``) set by the user but not
          declared in the pipeline's ``default_generation_params``
        - Type mismatches for ``extra_params`` values
        - Out-of-range ``extra_params`` values
        """
        # Lazy import to avoid circular dependency
        # (executor → visual_gen.visual_gen → _torch.visual_gen → executor)
        from tensorrt_llm.visual_gen.visual_gen import VisualGenParamsError

        params = req.params
        errors: list[str] = []
        pipeline_name = self.pipeline.__class__.__name__
        declared_defaults = self.pipeline.default_generation_params
        specs = self.pipeline.extra_param_specs

        # --- unknown extra_params keys ---
        if params.extra_params:
            unknown = set(params.extra_params.keys()) - set(specs.keys())
            if unknown:
                errors.append(
                    f"Unknown extra_params {sorted(unknown)} for {pipeline_name}. "
                    f"Supported: {sorted(specs.keys())}"
                )

        # --- unsupported universal fields ---
        # Check generation config fields the user explicitly set (not None)
        # that the pipeline never declared in default_generation_params.
        # Conditioning inputs (image, negative_prompt, mask) are excluded —
        # they are validated at runtime by the pipeline's infer().
        for field_name in _GENERATION_CONFIG_FIELDS:
            value = getattr(params, field_name, None)
            if value is not None and field_name not in declared_defaults:
                errors.append(
                    f"Parameter '{field_name}' is set but {pipeline_name} does "
                    f"not use it (not in default_generation_params). "
                    f"It will be silently ignored."
                )

        # --- extra_params type and range checks ---
        if params.extra_params:
            for key, value in params.extra_params.items():
                if key not in specs:
                    continue  # already reported as unknown above
                spec = specs[key]
                # Skip None values (param left at its None default)
                if value is None:
                    continue
                # Type check
                expected_types = _TYPE_MAP.get(spec.type)
                if expected_types and not isinstance(value, expected_types):
                    errors.append(
                        f"extra_params['{key}'] expected type '{spec.type}', "
                        f"got {type(value).__name__}: {value!r}"
                    )
                    continue  # skip range check if type is wrong
                # Range check (numeric only)
                if spec.range is not None and isinstance(value, (int, float)):
                    lo, hi = spec.range
                    if not (lo <= value <= hi):
                        errors.append(
                            f"extra_params['{key}'] value {value} is out of range [{lo}, {hi}]"
                        )

        if errors:
            msg = f"Parameter validation failed for {pipeline_name}:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise VisualGenParamsError(msg)

    def process_request(self, req: DiffusionRequest):
        """Process a single request."""
        try:
            self._merge_defaults(req)
            cache_key = self.pipeline.warmup_cache_key(
                req.params.height, req.params.width, num_frames=req.params.num_frames
            )
            if self.pipeline._warmed_up_shapes and cache_key not in self.pipeline._warmed_up_shapes:
                logger.warning(
                    f"Requested shape {cache_key} was not warmed up. "
                    f"First request with this shape will be slower due to "
                    f"torch.compile recompilation or CUDA graph capture. "
                    f"Warmed-up shapes: {self.pipeline._warmed_up_shapes}"
                )
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
    request_queue_addr: Optional[str],
    response_queue_addr: Optional[str],
    diffusion_args: "VisualGenArgs",
    log_level: str = "info",
    req_hmac_key: Optional[bytes] = None,
    resp_hmac_key: Optional[bytes] = None,
    local_rank: Optional[int] = None,
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

        # Determine local_rank: explicit arg > LOCAL_RANK env > global rank.
        # In multi-node runs (torchrun / srun --ntasks-per-node) SLURM/torchelastic
        # sets LOCAL_RANK; in single-node mp.Process mode it equals the global rank.
        _local_rank = (
            local_rank if local_rank is not None else int(os.environ.get("LOCAL_RANK", rank))
        )
        os.environ["LOCAL_RANK"] = str(_local_rank)

        # Use local_rank for device assignment so that each node's ranks map to
        # GPUs 0..gpus_per_node-1 rather than wrapping the global rank.
        device_id = _local_rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
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
            req_hmac_key=req_hmac_key,
            resp_hmac_key=resp_hmac_key,
        )
        executor.serve_forever()
        if executor.pipeline is not None:
            executor.pipeline.cleanup()
        dist.destroy_process_group()

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        traceback.print_exc()
