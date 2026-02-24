import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import ray
except ModuleNotFoundError as e:
    e.msg = """Cannot import Ray. Please install 'ray' package to use ray orchestrator"""
    raise

from ray.util.placement_group import (PlacementGroupSchedulingStrategy,
                                      get_current_placement_group,
                                      placement_group)

from tensorrt_llm._ray_utils import unwrap_ray_errors
from tensorrt_llm._utils import nvtx_range_debug
from tensorrt_llm.logger import logger

from ..llmapi.utils import logger_debug
from .executor import GenerationExecutor
from .postproc_worker import PostprocWorkerConfig
from .ray_gpu_worker import RayGPUWorker, RayWorkerWrapper
from .request import GenerationRequest
from .result import GenerationResult
from .rpc_proxy_mixin import RpcExecutorMixin
from .utils import has_event_loop

__all__ = [
    "RayExecutor",
]


class RayExecutor(RpcExecutorMixin, GenerationExecutor):

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
            if os.environ.get("TLLM_RAY_FORCE_LOCAL_CLUSTER", "0") != "1":
                try:
                    ray.init(address="auto", **ray_init_args)
                    logger.info(f"Attached to an existing Ray cluster.")
                except ConnectionError:
                    logger.info(f"Ray cluster not found, starting a new one.")

                if not ray.is_initialized():
                    ray.init(**ray_init_args)
                    self.has_start_local_cluser = True
            else:
                ray.init(address="local", **ray_init_args)
                self.has_start_local_cluser = True

            self.world_size = model_world_size
            self.tp_size = tp_size
            self.master_address = ray.util.get_node_ip_address()

            self.worker_kwargs = dict(
                **worker_kwargs,
                postproc_worker_config=postproc_worker_config,
                is_llm_executor=is_llm_executor)

            self.init_rpc_executor()
            # Inject the generated HMAC key into worker_kwargs for workers
            self.worker_kwargs['hmac_key'] = self.hmac_key
            self.worker_kwargs['rpc_addr'] = self.rpc_addr

            placement_config = getattr(self.worker_kwargs['llm_args'],
                                       'ray_placement_config', None)
            defer_workers_init = placement_config.defer_workers_init if placement_config else False

            if defer_workers_init:
                self.workers = [
                ]  # Placeholder, will be initialized in setup_async
                self._mainloop_started = False  # DO NOT start mainloop until after setup_engine_remote_async is called
            else:
                if not has_event_loop():
                    self.init_workers_sync()
                self.setup_engine_remote()
                self.setup_mainloop(tasks=[self._fetch_responses_loop_async],
                                    thread_name="ray_executor_main_loop")

        except Exception as e:
            self.shutdown()
            logger.error(f"Failed to initialize RayExecutor: {e}")
            raise e

    def create_workers(self, worker_cls, worker_kwargs):
        llm_args = worker_kwargs.get("llm_args")
        placement_config = getattr(llm_args, 'ray_placement_config',
                                   None) if llm_args else None

        # When set to be a fraction, it allows Ray to schedule
        # multiple actors on a single GPU for colocate use cases.
        num_gpus = float(os.getenv("TRTLLM_RAY_PER_WORKER_GPUS", "1.0"))
        if placement_config and placement_config.per_worker_gpu_share is not None:
            num_gpus = placement_config.per_worker_gpu_share

        logger.debug(f"{num_gpus=} for each worker.")

        runtime_env = ray.runtime_env.RuntimeEnv()
        runtime_env["env_vars"] = os.environ.copy()
        runtime_env["env_vars"].update({
            "TLLM_DISABLE_MPI": "1",
            "MASTER_ADDR": self.master_address,  # head-IP for NCCL/Gloo
        })

        placement_groups, self.bundle_indices = self._get_placement_group(
            tp_size=self.tp_size, worker_kwargs=worker_kwargs)

        if isinstance(placement_groups, list):
            self.placement_group = None
        else:
            self.placement_group = placement_groups

        self.workers = []
        for rank in range(self.world_size):
            pg = placement_groups[rank] if isinstance(
                placement_groups, list) else placement_groups
            worker = RayWorkerWrapper.options(
                num_gpus=num_gpus,
                runtime_env=runtime_env,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=self.bundle_indices[rank],
                )).remote(worker_cls, worker_kwargs, self.world_size, rank)
            self.workers.append(worker)

    def init_workers_sync(self):
        self.create_workers(RayGPUWorker, self.worker_kwargs)
        try:
            ray.get(self._get_worker_ready_futures())
        except ray.exceptions.ActorDiedError as e:
            raise RuntimeError("RayGPUWorker died during initialization") from e
        port = self.call_all_ray_workers("setup_tcp_store",
                                         leader_only=True,
                                         async_call=False)[0]
        self.call_all_ray_workers("setup_distributed_env_and_worker",
                                  leader_only=False,
                                  async_call=False,
                                  port=port)

    async def init_workers_async(self):
        self.create_workers(RayGPUWorker, self.worker_kwargs)
        try:
            await asyncio.gather(*self._get_worker_ready_futures())
        except ray.exceptions.ActorDiedError as e:
            raise RuntimeError("RayGPUWorker died during initialization") from e
        port = (await asyncio.gather(*self.call_all_ray_workers(
            "setup_tcp_store", leader_only=True, async_call=True)))[0]
        await asyncio.gather(
            *self.call_all_ray_workers("setup_distributed_env_and_worker",
                                       leader_only=False,
                                       async_call=True,
                                       port=port))

    @unwrap_ray_errors()
    def call_all_ray_workers(self, func: str, leader_only: bool,
                             async_call: bool, *args, **kwargs):
        workers = (self.workers[0], ) if leader_only else self.workers
        if async_call:
            return [
                getattr(worker, func).remote(*args, **kwargs)
                for worker in workers
            ]
        else:
            return ray.get([
                getattr(worker, func).remote(*args, **kwargs)
                for worker in workers
            ])

    @unwrap_ray_errors()
    def collective_rpc(self,
                       method: str,
                       args: tuple = (),
                       kwargs: Optional[dict] = None,
                       non_block: bool = False,
                       unique_reply_rank: Optional[int] = None) -> list[Any]:
        workers = (self.workers[unique_reply_rank],
                   ) if unique_reply_rank is not None else self.workers
        kwargs = kwargs or {}

        refs = []
        for w in workers:
            try:
                refs.append(getattr(w, method).remote(*args, **kwargs))
            except AttributeError:
                # Here worker is the RayWorkerWrapper.
                # For extended worker methods, we need to use call_worker_method since
                # Ray actor doesn't work with __getattr__ delegation.
                refs.append(w.call_worker_method.remote(method, *args,
                                                        **kwargs))
        return refs if non_block else ray.get(refs)

    @unwrap_ray_errors()
    async def collective_rpc_async(
            self,
            method: str,
            args: tuple = (),
            kwargs: Optional[dict] = None,
            unique_reply_rank: Optional[int] = None) -> list[Any]:
        refs = self.collective_rpc(method,
                                   args,
                                   kwargs,
                                   non_block=True,
                                   unique_reply_rank=unique_reply_rank)
        return await asyncio.gather(*refs)

    def submit(self, request: "GenerationRequest") -> "GenerationResult":
        """
        Low-level API to the executor. Return a "future" GenerationResult
        which can be waited. Forwards the request to the workers through RPC.
        """
        if request.id is None:
            request.set_id(self._get_next_client_id())
        logprob_params = self._get_logprob_params(request)

        with nvtx_range_debug("rpc_submit"):
            self.rpc_client.submit(request).remote(need_response=False)

        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params)
        self._results[request.id] = result

        return result

    def start(self):
        pass

    def setup_engine_remote(self):
        return self.collective_rpc("setup_engine", non_block=False)

    async def setup_engine_remote_async(self):
        """Async version of setup_engine_remote for use after async worker initialization."""
        if not self.workers or len(self.workers) == 0:
            raise RuntimeError(
                "Workers must be initialized before calling setup_engine_remote_async"
            )

        # Setup engine on all workers
        result = await self.collective_rpc_async("setup_engine")
        logger.info("setup_engine_remote_async finished")

        # Now that engine is set up, start the mainloop for fetching responses
        if hasattr(self, '_mainloop_started') and not self._mainloop_started:
            logger.info("Starting mainloop after engine setup")
            self.setup_mainloop(tasks=[self._fetch_responses_loop_async],
                                thread_name="ray_executor_main_loop")
            self._mainloop_started = True

        return result

    def report_device_ids(self) -> list[str]:
        gpu_ids = self.call_all_ray_workers("report_device_id",
                                            leader_only=False,
                                            async_call=False)
        return sorted(gpu_ids)

    def abort_request(self, request_id: int) -> None:
        self.call_all_ray_workers("abort_request",
                                  leader_only=True,
                                  async_call=False,
                                  request_id=request_id)

    def shutdown(self):
        if hasattr(self, '_shutdown_event') and self._shutdown_event.is_set():
            return
        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()

        logger_debug(f"Shutting down RayExecutor", color="yellow")

        if hasattr(self, 'main_loop') and self.main_loop and hasattr(
                self, 'main_loop_task_obj') and self.main_loop_task_obj:
            logger_debug("Cancelling main loop task.", color="yellow")
            try:
                self.main_loop.call_soon_threadsafe(
                    self.main_loop_task_obj.cancel)
            except Exception as e:
                logger_debug(f"Error cancelling main loop task: {e}",
                             color="yellow")

            if hasattr(self, 'main_loop_thread'):
                self.main_loop_thread.join()

        # Then, shutdown the workers
        if hasattr(self, 'workers') and self.workers is not None:
            try:
                shutdown_refs = [
                    worker.shutdown.remote() for worker in self.workers
                ]
                # Add timeout to prevent indefinite hanging
                ray.get(shutdown_refs, timeout=30.0)
            except ray.exceptions.GetTimeoutError:
                logger.warning(
                    "Timeout waiting for workers to shutdown after 30 seconds")
            except Exception as e:
                logger.warning(f"Error shutting down: {e}")

        if hasattr(self, 'rpc_client') and self.rpc_client is not None:
            try:
                self.rpc_client.close()
            except Exception as e:
                logger_debug(f"Suppressed error during RPC client close: {e}")

        self.workers = None
        if hasattr(self,
                   "placement_group") and self.placement_group is not None:
            # Only remove placement group if Ray is still initialized
            # to avoid triggering auto_init_ray() during program exit
            if ray.is_initialized():
                ray.util.remove_placement_group(self.placement_group)
            self.placement_group = None
        self.bundle_indices = None

        if self.has_start_local_cluser and ray.is_initialized():
            logger.debug("Shutting down Ray cluster")
            ray.shutdown()

    def _get_worker_ready_futures(self):
        return [worker.__ray_ready__.remote() for worker in self.workers]

    def _get_placement_group(
            self,
            tp_size: int,
            worker_kwargs: Dict = None) -> Tuple[Any, List[int]]:
        """
        Obtain placement group(s) and bundle indices for workers.

        Priorities:
        1. `ray_placement_config` in `llm_args`.
        2. `TRTLLM_RAY_BUNDLE_INDICES` environment variable (uses current placement group).
        3. Default creation: A PACK placement group where each bundle has `tp_size` GPUs.
           - When `tp_size` <= GPUs per node, keep one TP group per node.
           - When `tp_size` > GPUs per node, allow a TP group to span nodes.
           - rank 0 is forced onto the driver node.

        Returns:
            Tuple[Union[PlacementGroup, List[PlacementGroup]], List[int]]:
            - placement_group(s): A single `PlacementGroup` (shared by all workers) or a list of `PlacementGroup` (one per worker).
            - bundle_indices: A list of bundle indices.
              If `placement_group(s)` is a single object, `bundle_indices[i]` maps worker `i` to that bundle in the group.
              If `placement_group(s)` is a list, `bundle_indices[i]` maps worker `i` to that bundle in `placement_groups[i]`.
        """
        llm_args = worker_kwargs.get("llm_args") if worker_kwargs else None

        placement_config = getattr(llm_args, 'ray_placement_config',
                                   None) if llm_args else None

        def _get_from_placement_config(placement_config):
            total_workers = sum(
                len(indices)
                for indices in placement_config.placement_bundle_indices)
            if total_workers != self.world_size:
                raise ValueError(
                    f"Total bundle indices ({total_workers}) must equal world_size ({self.world_size})"
                )

            logger.info(
                f"Creating {self.world_size} workers with external placement groups"
            )

            flat_pgs = []
            flat_indices = []
            for pg, indices in zip(placement_config.placement_groups,
                                   placement_config.placement_bundle_indices):
                for idx in indices:
                    flat_pgs.append(pg)
                    flat_indices.append(idx)

            return flat_pgs, flat_indices

        def _get_from_env(bundle_indices):
            pg = get_current_placement_group()
            if pg is not None:
                bundle_indices = list(map(int, bundle_indices.split(",")))
                assert len(bundle_indices) == self.world_size, (
                    f"Need {self.world_size} bundle indices for world_size, got {bundle_indices=}"
                )
                assert len(set(bundle_indices)) == len(bundle_indices), (
                    f"TRTLLM_RAY_BUNDLE_INDICES cannot have duplicate values, but got {bundle_indices=}."
                )
                assert max(bundle_indices) < len(pg.bundle_specs), (
                    f"{bundle_indices=} out of range for PG with {len(pg.bundle_specs)} bundles"
                )
                return pg, bundle_indices
            else:
                raise ValueError("No global placement group is found.")

        def _get_default(tp_size):
            head_tag = f"node:{self.master_address}"
            nodes = ray.nodes()
            gpus_per_node = int(nodes[0]["Resources"].get(
                "GPU", 0))  # assume symmetric across nodes

            bundle_cpu = bundle_gpu = min(tp_size, gpus_per_node)

            bundles, bundle_indices = [], []
            current = 0
            for rank in range(self.world_size):
                if current == 0:
                    bundle = {"GPU": bundle_gpu, "CPU": bundle_cpu}
                    if len(bundles) == 0:
                        bundle[
                            head_tag] = 0.01  # to force placement on head node
                    bundles.append(bundle)

                bundle_indices.append(len(bundles) - 1)
                current = (current + 1) % bundle_gpu

            strategy = "PACK"
            logger.debug(
                f"[Strategy={strategy}] Bundles: {bundles} for tp_size: {tp_size} and world_size: {self.world_size}"
            )
            pg = placement_group(bundles, strategy=strategy)

            return pg, bundle_indices

        if self.world_size % tp_size != 0:
            raise ValueError(
                f"world_size {self.world_size} must be a multiple of tp_size {tp_size}"
            )

        # path 0
        if placement_config and placement_config.placement_groups is not None:
            return _get_from_placement_config(placement_config)
        # path 1
        if bundle_indices := os.getenv("TRTLLM_RAY_BUNDLE_INDICES", None):
            return _get_from_env(bundle_indices)
        # path 2
        return _get_default(tp_size)

    @property
    def enable_postprocess_parallel(self) -> bool:
        ret = super().enable_postprocess_parallel
        assert ret == False, "Postprocess parallel is not supported in RayExecutor"
        return ret
