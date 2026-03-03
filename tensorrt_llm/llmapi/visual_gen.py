import asyncio
import queue
import socket
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch.multiprocessing as mp
import zmq

from tensorrt_llm._torch.visual_gen import DiffusionRequest, DiffusionResponse
from tensorrt_llm._torch.visual_gen.executor import run_diffusion_worker
from tensorrt_llm._torch.visual_gen.output import MediaOutput

__all__ = ["VisualGen", "VisualGenParams", "MediaOutput"]
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.inputs.data import VisualGenInputs
from tensorrt_llm.logger import logger

# Timeouts (seconds)
POLL_TIMEOUT = 0.01
AWAIT_TIMEOUT = 0.05
THREAD_TIMEOUT = 5.0
WORKER_TIMEOUT = 2.0
READY_TIMEOUT = 1200  # 20 minutes for large models (Wan 2.2 with transformer_2)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_ip_address() -> str:
    """Get local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class DiffusionRemoteClient:
    """Client proxy for remote DiffusionExecutor in worker processes."""

    def __init__(
        self,
        model_path: Union[str, Path],
        n_workers: int = 1,
        diffusion_config: Optional[dict] = None,
    ):
        self.model_path = str(model_path)
        self.n_workers = n_workers
        self.diffusion_config = diffusion_config

        # Setup distributed env
        self.master_addr = "127.0.0.1"
        self.master_port = find_free_port()

        # Setup IPC addresses
        self.host_ip = get_ip_address()
        req_port, resp_port = find_free_port(), find_free_port()

        self.request_queue_addr = f"tcp://0.0.0.0:{req_port}"
        self.response_queue_addr = f"tcp://0.0.0.0:{resp_port}"
        self.req_addr_connect = f"tcp://{self.host_ip}:{req_port}"
        self.resp_addr_connect = f"tcp://{self.host_ip}:{resp_port}"

        # IPC setup
        self.requests_ipc = None
        self.responses_ipc = None
        self.pending_requests = queue.Queue()
        self.completed_responses: Dict[int, DiffusionResponse] = {}

        # We'll create asyncio primitives in the background thread's event loop
        self._event_loop = None
        self.response_event = None
        self.lock = None
        self.shutdown_event = threading.Event()
        self.event_loop_ready = threading.Event()

        # Start background thread (it will create its own event loop)
        self.background_thread = threading.Thread(target=self._serve_forever_thread, daemon=True)
        self.background_thread.start()

        # Wait for the background thread to initialize the event loop
        self.event_loop_ready.wait()

        # Launch workers
        logger.info(f"DiffusionClient: Launching {n_workers} workers")
        ctx = mp.get_context("spawn")
        self.worker_processes = []
        for rank in range(n_workers):
            p = ctx.Process(
                target=run_diffusion_worker,
                kwargs={
                    "rank": rank,
                    "world_size": n_workers,
                    "master_addr": self.master_addr,
                    "master_port": self.master_port,
                    "model_path": self.model_path,
                    "request_queue_addr": self.req_addr_connect,
                    "response_queue_addr": self.resp_addr_connect,
                    "diffusion_config": self.diffusion_config,
                },
            )
            p.start()
            self.worker_processes.append(p)

        self._wait_ready()

    @staticmethod
    def _close_socket(ipc_queue):
        if ipc_queue and ipc_queue.socket:
            ipc_queue.socket.setsockopt(zmq.LINGER, 0)
            ipc_queue.close()

    def enqueue_requests(self, requests: List[DiffusionRequest]) -> List[int]:
        """Enqueue requests and return their IDs."""
        req_ids = []
        for req in requests:
            self.pending_requests.put(req)
            req_ids.append(req.request_id)
        return req_ids

    async def await_responses(
        self, request_ids: Union[int, List[int]], timeout: Optional[float] = None
    ) -> Union[DiffusionResponse, List[DiffusionResponse]]:
        """Wait for responses by request IDs.

        Args:
            request_ids: Single request ID or list of request IDs to wait for
            timeout: Maximum total wait time in seconds (None = wait indefinitely)

        Returns:
            Single response or list of responses (None if request timed out)
        """
        is_single = isinstance(request_ids, int)
        ids = [request_ids] if is_single else request_ids

        start_time = time.time()
        results = {}

        while len(results) < len(ids):
            async with self.lock:
                for req_id in ids:
                    if req_id in self.completed_responses:
                        results[req_id] = self.completed_responses.pop(req_id)

            # All responses collected
            if len(results) == len(ids):
                break

            # Check if overall timeout exceeded
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    break
                # Wait for remaining time or AWAIT_TIMEOUT, whichever is shorter
                wait_time = min(timeout - elapsed, AWAIT_TIMEOUT)
            else:
                wait_time = AWAIT_TIMEOUT

            try:
                await asyncio.wait_for(self.response_event.wait(), timeout=wait_time)
            except asyncio.TimeoutError:
                pass
            self.response_event.clear()

        out = [results.get(rid) for rid in ids]
        return out[0] if is_single else out

    def await_responses_sync(
        self, request_ids: Union[int, List[int]], timeout: Optional[float] = None
    ) -> Union[DiffusionResponse, List[DiffusionResponse]]:
        """Sync wrapper to await responses from the main thread."""
        future = asyncio.run_coroutine_threadsafe(
            self.await_responses(request_ids, timeout), self._event_loop
        )
        return future.result(timeout=timeout if timeout else None)

    def _init_ipc(self) -> bool:
        """Initialize IPC queues."""
        try:
            logger.info("DiffusionClient: Initializing IPC")
            self.requests_ipc = ZeroMqQueue(
                (self.request_queue_addr, None),
                is_server=True,
                socket_type=zmq.PUSH,
                use_hmac_encryption=False,
            )
            self.responses_ipc = ZeroMqQueue(
                (self.response_queue_addr, None),
                is_server=True,
                socket_type=zmq.PULL,
                use_hmac_encryption=False,
            )
            logger.info("DiffusionClient: IPC ready")
            return True
        except Exception as e:
            logger.error(f"DiffusionClient: IPC init failed: {e}")
            return False

    def _send_shutdown(self):
        """Send shutdown signal."""
        logger.info("DiffusionClient: Sending shutdown signal")
        if self.requests_ipc:
            self.requests_ipc.put(None)
            self._close_socket(self.requests_ipc)

    def _process_requests(self):
        """Process pending requests."""
        try:
            req = self.pending_requests.get(timeout=POLL_TIMEOUT)
            if req is None:
                self._send_shutdown()
                self.shutdown_event.set()
                return

            logger.info(f"DiffusionClient: Sending request {req.request_id}")
            self.requests_ipc.put(req)
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"DiffusionClient: Error sending request: {e}")
            logger.error(traceback.format_exc())

    def _process_responses(self):
        """Poll and process responses."""
        try:
            if self.responses_ipc.poll(timeout=POLL_TIMEOUT):
                response = self.responses_ipc.get()
                if isinstance(response, DiffusionResponse):
                    if response.request_id == -1:
                        logger.info("DiffusionClient: Received READY signal")

                    # Schedule the lock acquisition and event setting in the event loop
                    asyncio.run_coroutine_threadsafe(
                        self._store_response(response), self._event_loop
                    )
        except Exception as e:
            logger.error(f"DiffusionClient: Error processing response: {e}")

    async def _store_response(self, response: DiffusionResponse):
        """Store response in the completed_responses dict (async helper)."""
        async with self.lock:
            self.completed_responses[response.request_id] = response
        self.response_event.set()

    def _cleanup_ipc(self):
        """Cleanup IPC."""
        logger.info("DiffusionClient: Cleaning up IPC")
        self._close_socket(self.requests_ipc)
        self._close_socket(self.responses_ipc)

    def _serve_forever_thread(self):
        """Background thread wrapper that creates and runs an event loop."""
        logger.info("DiffusionClient: Background thread started")

        # Create a new event loop for this thread
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        # Create async primitives in this thread's event loop
        self.response_event = asyncio.Event()
        self.lock = asyncio.Lock()

        # Signal that the event loop is ready
        self.event_loop_ready.set()

        # Run the async serve_forever
        try:
            self._event_loop.run_until_complete(self._serve_forever())
        finally:
            self._event_loop.close()
            logger.info("DiffusionClient: Background thread stopped")

    async def _serve_forever(self):
        """Background thread main loop (async version)."""
        if not self._init_ipc():
            return

        while not self.shutdown_event.is_set():
            self._process_requests()
            self._process_responses()
            await asyncio.sleep(0.001)  # Yield control to allow other coroutines to run

        self._cleanup_ipc()

    def shutdown(self):
        """Shutdown client and workers."""
        logger.info("DiffusionClient: Shutting down")
        self.pending_requests.put(None)

        self.background_thread.join(timeout=THREAD_TIMEOUT)
        if self.background_thread.is_alive():
            logger.warning("DiffusionClient: Force stopping background thread")
            self.shutdown_event.set()
            self.background_thread.join(timeout=1.0)

        # Shutdown workers
        logger.info("DiffusionClient: Stopping workers")
        for p in self.worker_processes:
            p.join(timeout=WORKER_TIMEOUT)
            if p.is_alive():
                logger.warning(f"DiffusionClient: Terminating worker {p.pid} with SIGTERM")
                p.terminate()
                p.join(timeout=WORKER_TIMEOUT)
                if p.is_alive():
                    logger.warning(f"DiffusionClient: Force killing worker {p.pid} with SIGKILL")
                    p.kill()
                    p.join(timeout=WORKER_TIMEOUT)

    def _wait_ready(self, timeout: float = READY_TIMEOUT):
        """Wait for workers to be ready (sync wrapper for async operation)."""
        logger.info("DiffusionClient: Waiting for workers")

        # Run the async wait in the background thread's event loop
        future = asyncio.run_coroutine_threadsafe(self._wait_ready_async(timeout), self._event_loop)
        return future.result(timeout=timeout)

    async def _wait_ready_async(self, timeout: float = READY_TIMEOUT):
        """Wait for workers to be ready (async version)."""
        start_time = time.time()

        while True:
            async with self.lock:
                if -1 in self.completed_responses:
                    self.completed_responses.pop(-1)
                    logger.info("DiffusionClient: Workers ready")
                    return

            if time.time() - start_time > timeout:
                raise RuntimeError("DiffusionClient: Timeout waiting for workers")

            try:
                await asyncio.wait_for(self.response_event.wait(), timeout=AWAIT_TIMEOUT)
            except asyncio.TimeoutError:
                pass
            self.response_event.clear()


class DiffusionGenerationResult:
    """Future-like object for async generation."""

    def __init__(self, request_id: int, executor: DiffusionRemoteClient):
        self.request_id = request_id
        self.executor = executor
        self._result = None
        self._finished = False
        self._error = None

    async def result(self, timeout: Optional[float] = None) -> Any:
        """Wait for and return result (async version).

        Can be awaited from any async context (e.g., FastAPI background tasks).
        """
        if self._finished:
            if self._error:
                raise RuntimeError(self._error)
            return self._result

        # Use run_coroutine_threadsafe to execute in the background thread's event loop
        future = asyncio.run_coroutine_threadsafe(
            self.executor.await_responses(self.request_id, timeout=timeout),
            self.executor._event_loop,
        )

        # Await the future in the current event loop
        response = await asyncio.wrap_future(future)

        if response.error_msg:
            self._error = response.error_msg
            self._finished = True
            raise RuntimeError(f"Generation failed: {response.error_msg}")

        self._result = response.output
        self._finished = True
        return self._result

    def cancel(self):
        raise NotImplementedError("Cancel request (not yet implemented).")


@dataclass
class VisualGenParams:
    """Parameters for visual generation.

    Attributes:
        height: Output height in pixels
        width: Output width in pixels
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        max_sequence_length: Maximum sequence length for text encoding
        seed: Random seed for reproducibility

        # Video-specific parameters
        num_frames: Number of video frames to generate
        frame_rate: Frame rate for video output in fps

        # Image-specific parameters
        num_images_per_prompt: Number of images to generate per prompt (for image models)

        # Advanced parameters
        guidance_rescale: Guidance rescale factor (for some models)
        output_type: Output type ("pt" for PyTorch tensors, "pil" for PIL images)
    """

    height: int = 720
    width: int = 1280
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    max_sequence_length: int = 512
    seed: int = 42

    # Video-specific parameters
    num_frames: int = 81
    frame_rate: float = 24.0
    input_reference: Optional[str] = None

    # Image-specific parameters
    num_images_per_prompt: int = 1

    ## Image edit parameters
    image: Optional[List[str]] = None
    mask: Optional[str] = None

    # Advanced parameters
    guidance_rescale: float = 0.0
    output_type: str = "pt"

    # Wan-specific parameters
    guidance_scale_2: Optional[float] = None
    boundary_ratio: Optional[float] = None
    last_image: Optional[str] = None


class VisualGen:
    """High-level API for visual generation."""

    def __init__(
        self,
        model_path: Union[str, Path],
        n_workers: int = 1,
        diffusion_config: Optional[dict] = None,
    ):
        self.model_path = str(model_path)
        self.n_workers = n_workers
        self.diffusion_config = diffusion_config

        self.executor = DiffusionRemoteClient(
            model_path=self.model_path,
            n_workers=self.n_workers,
            diffusion_config=self.diffusion_config,
        )
        self.req_counter = 0

    def generate(
        self,
        inputs: VisualGenInputs,
        params: VisualGenParams,
    ) -> MediaOutput:
        """Synchronous generation. Blocks until complete.

        Args:
            params: Generation parameters.

        Returns:
            MediaOutput: Generated media with model-specific fields populated:
                - FLUX2: MediaOutput(image=torch.Tensor)
                - WAN: MediaOutput(video=torch.Tensor)
                - LTX2: MediaOutput(video=torch.Tensor, audio=torch.Tensor)
        """
        future = self.generate_async(
            inputs=inputs,
            params=params,
        )

        # Use the sync wrapper to get result
        response = self.executor.await_responses_sync(future.request_id, timeout=None)
        if response.error_msg:
            raise RuntimeError(f"Generation failed: {response.error_msg}")
        return response.output

    def generate_async(
        self,
        inputs: VisualGenInputs,
        params: VisualGenParams,
    ) -> DiffusionGenerationResult:
        """Async generation. Returns immediately with future-like object.

        Args:
            params: Generation parameters.

        Returns:
            DiffusionGenerationResult: Call result() to get output dict.
        """
        req_id = self.req_counter
        self.req_counter += 1

        if isinstance(inputs, dict):
            prompt = inputs.get("prompt")
            negative_prompt = inputs.get("negative_prompt", None)
        elif isinstance(inputs, str):
            prompt = inputs
            negative_prompt = None
        else:
            # TODO: Support batch generation
            raise ValueError(f"Invalid inputs type: {type(inputs)}")

        request = DiffusionRequest(
            request_id=req_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            max_sequence_length=params.max_sequence_length,
            seed=params.seed,
            num_frames=params.num_frames,
            frame_rate=params.frame_rate,
            num_images_per_prompt=params.num_images_per_prompt,
            guidance_rescale=params.guidance_rescale,
            output_type=params.output_type,
            image=params.input_reference,
            guidance_scale_2=params.guidance_scale_2,
            boundary_ratio=params.boundary_ratio,
            last_image=params.last_image,
        )

        self.executor.enqueue_requests([request])
        return DiffusionGenerationResult(req_id, self.executor)

    def shutdown(self):
        """Shutdown executor and cleanup."""
        logger.info("VisualGen: Shutting down")
        self.executor.shutdown()
