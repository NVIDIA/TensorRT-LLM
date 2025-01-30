import torch

from tensorrt_llm._torch.pyexecutor.backend_registries.backend_registry import \
    register_tllmptp_backend
from tensorrt_llm._torch.pyexecutor.decoder import *
from tensorrt_llm._torch.pyexecutor.distributed import *
from tensorrt_llm._torch.pyexecutor.model_engine import *
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import *
from tensorrt_llm._torch.pyexecutor.scheduler import *
from tensorrt_llm.bindings.executor import ExecutorConfig


@register_tllmptp_backend('trt_python', {'need_trt_engine': True})
def create_trt_python_executor(executor_config: ExecutorConfig,
                               checkpoint_dir: str = None,
                               engine_dir: str = None):
    max_num_requests = executor_config.max_batch_size
    model_engine = TRTModel(Path(engine_dir), meta_config={})
    torch.cuda.set_device(
        model_engine.mapping.get_local_rank(model_engine.mapping.rank))

    model_config = model_engine.model_config

    if model_config.quant_mode.has_fp8_kv_cache:
        kv_cache_dtype = DataType.FP8
    elif model_config.quant_mode.has_int8_kv_cache:
        kv_cache_dtype = DataType.INT8
    else:
        kv_cache_dtype = model_config.data_type

    kv_cache_manager = KVCacheManager.from_model_config(
        model_config,
        executor_config.kv_cache_config,
        mapping=model_engine.mapping,
        kv_cache_type=CacheTypeCpp.SELF,
        dtype=kv_cache_dtype)

    # update config
    model_config.max_seq_len = kv_cache_manager.max_seq_len
    executor_config.max_seq_len = kv_cache_manager.max_seq_len

    resource_manager = ResourceManager({'kv_cache_manager': kv_cache_manager})

    capacity_scheduler = BindCapacityScheduler(max_num_requests,
                                               kv_cache_manager.impl)
    #capacity_scheduler = GuaranteedNoEvictScheduler(max_num_requests,
    #                                                kv_cache_manager)
    mb_scheduler = BindMicroBatchScheduler(
        max_num_requests, model_engine.model_config.max_num_tokens)
    scheduler = SimpleScheduler(capacity_scheduler, mb_scheduler)
    decoder = TorchGreedySearchDecoder(max_seq_len=model_config.max_seq_len)
    dist = MPIDist(model_engine.mapping)
    py_executor = PyExecutor(resource_manager,
                             scheduler,
                             model_engine=model_engine,
                             decoder=decoder,
                             dist=dist)
    return py_executor
