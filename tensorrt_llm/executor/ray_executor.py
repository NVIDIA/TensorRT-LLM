import os
import socket
from typing import Dict, List, Optional, Tuple

import ray
from ray.util.placement_group import (PlacementGroup,
                                      PlacementGroupSchedulingStrategy,
                                      get_current_placement_group,
                                      placement_group)

from tensorrt_llm.logger import logger

from .._utils import nvtx_range_debug
from .executor import GenerationExecutor
from .postproc_worker import PostprocWorkerConfig
from .ray_gpu_worker import RayGPUWorker, RayWorkerWrapper
from .request import GenerationRequest
from .result import (GenerationResult, IterationResult, RayAsyncQueue,
                     RaySyncQueue)

__all__ = [
    "RayExecutor",
]


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to port 0 to get a random free port
        s.listen(1)
        port = s.getsockname()[1]
    return port


class RayExecutor(GenerationExecutor):

    def __init__(self,
                 worker_kwargs: Dict,
                 model_world_size: int,
                 postproc_worker_config: PostprocWorkerConfig,
                 is_llm_executor: bool,
                 tp_size=1):
        os.environ['RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'] = '1'
        os.environ["RAY_DEDUP_LOGS"] = "0"  # for debug

        super().__init__(model_world_size, postproc_worker_config,
                         is_llm_executor)

        self.has_start_local_cluser = False
        runtime_env = {
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"
            }
        }

        ray_init_args = {
            "include_dashboard": False,
            "namespace": "trtllm",
            "ignore_reinit_error": True,
            "runtime_env": runtime_env
        }
        try:
            ray.init(address="auto", **ray_init_args)
            logger.info(f"Attached to an existing Ray cluster.")
        except ConnectionError:
            logger.info(f"Ray cluster not found, starting a new one.")

        if not ray.is_initialized():
            ray.init(**ray_init_args)
            self.has_start_local_cluser = True

        self.world_size = model_world_size
        self.tp_size = tp_size
        self.master_address = ray.util.get_node_ip_address()
        self.master_port = get_free_port()
        self.response_queue = RayAsyncQueue.options().remote()
        self.response_sync_queue = RaySyncQueue.options().remote()
        self.async_response_queue_weakref = self.create_actor_weak_ref(
            self.response_queue)
        self.sync_response_queue_weakref = self.create_actor_weak_ref(
            self.response_sync_queue)

        self.response_queue.warmup.remote()
        self.response_sync_queue.warmup.remote()

        self.create_workers(RayGPUWorker, worker_kwargs)

    @staticmethod
    def create_actor_weak_ref(actor_handle: ray.actor.ActorHandle):
        state, _, _ = actor_handle._serialization_helper()
        return ray.actor.ActorHandle._deserialization_helper(state,
                                                             weak_ref=True)

    def use_ray_queue(self) -> bool:
        return True

    def create_workers(self, worker_cls, worker_kwargs):
        # If this is set to be a fraction, it allows Ray to schedule
        # multiple actors on a single GPU for colocate use cases.
        num_gpus = float(os.getenv("TRTLLM_RAY_PER_WORKER_GPUS", "1.0"))
        logger.debug(f"{num_gpus=} for each worker.")

        runtime_env = ray.runtime_env.RuntimeEnv()
        runtime_env["env_vars"] = os.environ.copy()
        runtime_env["env_vars"].update({
            "TLLM_DISABLE_MPI": "1",
            "MASTER_ADDR": self.master_address,  # head-IP for NCCL/Gloo
            "MASTER_PORT": str(self.master_port)
        })

        self.placement_group, self.bundle_indices = self._get_placement_group(
            tp_size=self.tp_size)

        self.workers = [
            RayWorkerWrapper.options(
                num_gpus=num_gpus,
                runtime_env=runtime_env,  # per-actor env
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_group,
                    placement_group_bundle_index=self.bundle_indices[rank],
                )).remote(worker_cls, worker_kwargs, self.world_size, rank)
            for rank in range(self.world_size)
        ]

        ray.get([worker.__ray_ready__.remote() for worker in self.workers])

    def call_all_ray_workers(self, func: str, leader_only: bool,
                             async_call: bool, *args, **kwargs):
        workers = (self.workers[0], ) if leader_only else self.workers

        if async_call:
            return [(getattr(self.workers, func).remote(*args, **kwargs))
                    for worker in workers]
        else:
            return ray.get([
                getattr(worker, func).remote(*args, **kwargs)
                for worker in workers
            ])

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult
            which can be waited.
            Forwards the request to the workers through the request queue.
        """

        request.set_id(self._get_next_client_id())
        logprob_params = self._get_logprob_params(request)

        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params)

        with nvtx_range_debug("request_queue.put"):
            self.call_all_ray_workers("enqueue_request",
                                      leader_only=True,
                                      request=request,
                                      async_call=False,
                                      result_wait_queue=result.queue)

        return result

    def abort_request(self, request_id: int) -> None:
        self.call_all_ray_workers("abort_request",
                                  leader_only=True,
                                  async_call=False,
                                  request_id=request_id)

    def shutdown(self):
        # Release actors
        self.response_queue = None
        self.response_sync_queue = None
        self.async_response_queue_weakref = None
        self.sync_response_queue_weakref = None

        self.workers = None
        if hasattr(self,
                   "placement_group") and self.placement_group is not None:
            ray.util.remove_placement_group(self.placement_group)
            self.placement_group = None
        self.bundle_indices = None

        if self.has_start_local_cluser:
            logger.debug("Shutting down Ray cluster")
            ray.shutdown()

    @property
    def enable_postprocess_parallel(self) -> bool:
        raise NotImplementedError(
            "enable_postprocess_parallel is not implemented")

    def get_stats(self, timeout: float) -> List[dict]:
        raise NotImplementedError("get_stats is not implemented")

    def aget_stats(self, timeout: float) -> IterationResult:
        raise NotImplementedError("aget_stats is not implemented")

    def get_kv_events(self, timeout: float) -> List[dict]:
        raise NotImplementedError("get_kv_events is not implemented")

    def aget_kv_events(self, timeout=None) -> IterationResult:
        raise NotImplementedError("aget_kv_events is not implemented")

    def wait_first_completed(self, futures: List[GenerationResult]):
        raise NotImplementedError("wait_first_completed is not implemented")

    def _get_placement_group(self,
                             tp_size: int) -> Tuple[PlacementGroup, List[int]]:
        """
        Either use the existing placement group from driver script (e.g., in the case of RL FW integration),
        or create a default PACK placement group where each bundle has tp_size GPUs.
         - When tp_size ≤ GPUs per node, keep one TP group per node.
         - When tp_size >  GPUs per node, allow a TP group span nodes.
         - rank 0 must be put on the driver node
        """
        # check whether a placement group exists
        pg = get_current_placement_group()
        bundle_indices = os.getenv("TRTLLM_RAY_BUNDLE_INDICES", None)

        if pg is not None and bundle_indices:
            bundle_indices = list(map(int, bundle_indices.split(",")))
            assert len(bundle_indices) == self.world_size, (
                f"Need {self.world_size} bundle indices for world_size, got {bundle_indices=}"
            )
            assert len(set(bundle_indices)) == len(bundle_indices), \
                f"TRTLLM_RAY_BUNDLE_INDICES cannot have duplicate values, but got {bundle_indices=}."

            assert max(bundle_indices) < len(pg.bundle_specs), \
                f"{bundle_indices=} out of range for PG with {len(pg.bundle_specs)} bundles"

            logger.info(
                f"Found existing placement group {pg.bundle_specs=}. {bundle_indices=}"
            )

            # TODO: need to ping TP group onto the same node for RL FW integration case

            return pg, bundle_indices

        if bundle_indices is not None:
            logger.warning(
                f"Ignoring TRTLLM_RAY_BUNDLE_INDICES={bundle_indices} because no global placement group is found."
            )

        if self.world_size % tp_size:
            raise ValueError("world_size must be a multiple of tp_size")

        head_tag = f"node:{self.master_address}"
        nodes = ray.nodes()
        gpus_per_node = int(nodes[0]["Resources"].get(
            "GPU", 0))  # assume symmetric across nodes
        # Don't set bundle_cpu to be bundle_gpu * 2, because RayAsyncQueue
        # currently requires 2 CPUs.
        bundle_cpu = bundle_gpu = min(tp_size, gpus_per_node)

        bundles, bundle_indices = [], []
        current = 0
        for rank in range(self.world_size):
            if current == 0:
                bundle = {"GPU": bundle_gpu, "CPU": bundle_cpu}
                if len(bundles) == 0:
                    bundle[head_tag] = 0.01  # to force placement on head
                bundles.append(bundle)

            bundle_indices.append(len(bundles) - 1)
            current = (current + 1) % bundle_gpu

        strategy = "PACK"
        logger.debug(
            f"[Strategy={strategy}] Bundles: {bundles} for tp_size: {tp_size} and world_size: {self.world_size}"
        )
        pg = placement_group(bundles, strategy=strategy)
        ray.get(pg.ready())  # blocks until reserved
        logger.debug(f"Placement group ready.")

        return pg, bundle_indices
