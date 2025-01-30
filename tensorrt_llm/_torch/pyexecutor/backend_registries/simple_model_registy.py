from tensorrt_llm._torch.pyexecutor.backend_registries.backend_registry import \
    register_tllmptp_backend
from tensorrt_llm._torch.pyexecutor.decoder import TorchGreedySearchDecoder
from tensorrt_llm._torch.pyexecutor.distributed import *
from tensorrt_llm._torch.pyexecutor.model_engine import *
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import *
from tensorrt_llm._torch.pyexecutor.scheduler import *
from tensorrt_llm.bindings.executor import ExecutorConfig
from tensorrt_llm.mapping import Mapping


@register_tllmptp_backend('simple')
def create_simple_executor(executor_config: ExecutorConfig,
                           checkpoint_dir: str = None,
                           engine_dir: str = None):
    max_num_tokens = executor_config.max_num_tokens
    max_seq_len = executor_config.max_seq_len
    max_num_requests = executor_config.max_batch_size
    kv_cache_manager = DummyKvCacheManager(1024, max_num_tokens)
    capacitor_scheduler = GuaranteedNoEvictScheduler(max_num_requests,
                                                     kv_cache_manager)
    mb_scheduler = BindMicroBatchScheduler(max_num_requests, max_num_tokens)
    scheduler = SimpleScheduler(capacitor_scheduler, mb_scheduler)
    model_engine = TorchUnbatchedModelEngine(checkpoint_dir)
    decoder = TorchGreedySearchDecoder(max_seq_len=max_seq_len)
    # TODO: add tp/pp support
    dist = TorchDist(Mapping())
    py_executor = PyExecutor(kv_cache_manager,
                             scheduler,
                             model_engine=model_engine,
                             decoder=decoder,
                             dist=dist)
    return py_executor
