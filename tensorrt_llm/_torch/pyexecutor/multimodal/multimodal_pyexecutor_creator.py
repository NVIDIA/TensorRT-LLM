import copy

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...distributed import MPIDist

from .multimodal_executor import MMExecutor, MultimodalModelEngine
from tensorrt_llm.bindings.executor import ExecutorConfig

def create_multimodal_pyexecutor(executor_config: ExecutorConfig,
                       checkpoint_dir: str = None):

    if executor_config.mapping is None:
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          gpus_per_node=tensorrt_llm.default_gpus_per_node(),
                          rank=tensorrt_llm.mpi_rank())
    else:
        mapping = copy.deepcopy(executor_config.mapping)
        mapping.rank = tensorrt_llm.mpi_rank()

    dist = MPIDist(mapping=mapping)

    model_engine = MultimodalModelEngine(
        checkpoint_dir,
        #mapping=mapping,
        max_batch_size=executor_config.max_batch_size,
        dist=dist,
    )
    resources_manager = {}
    scheduler = None
    py_executor = MMExecutor(resources_manager,
                      scheduler,
                      model_engine=model_engine,
                      dist=dist,
                      enable_overlap_scheduler=False,
                      max_num_active_requests=executor_config.max_num_active_requests,
                      max_batch_size=executor_config.max_batch_size,
                      start_worker=False)

    py_executor.start_worker()
    return py_executor
