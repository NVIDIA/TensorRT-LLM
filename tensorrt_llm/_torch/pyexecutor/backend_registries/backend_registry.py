import copy
import os
from typing import Callable, Dict, NamedTuple

import torch

from tensorrt_llm.bindings.executor import Executor, ExecutorConfig, ModelType

backend_registry = {}

_backend_info_default_values = {
    'need_hf_model': False,
    'need_trt_engine': False,
}


class BackendEntry(NamedTuple):
    func: Callable
    backend_dict: Dict


def register_tllmptp_backend(name: str, backend_info: dict = None):

    def decorator(func):
        backend_dict = copy.deepcopy(_backend_info_default_values)
        if backend_info:
            for k, v in backend_info.items():
                if k not in backend_dict:
                    raise KeyError(
                        f"Unknown key {k} in backend registry, supported keys: {backend_dict.keys()}"
                    )
                backend_dict[k] = v
        backend_registry[name] = BackendEntry(func, backend_dict)
        return func

    return decorator


def has_backend(backend_name: str) -> bool:
    return backend_name in backend_registry


def get_backend_info(backend_name: str, info_key: str):
    if backend_name not in backend_registry:
        raise ValueError(
            f'backend name={backend_name} not registered, registered backends={backend_registry}'
        )
    return backend_registry[backend_name].backend_dict[info_key]


def create_py_executor_by_config(name: str,
                                 checkpoint_dir: str = None,
                                 engine_dir: str = None,
                                 executor_config: ExecutorConfig = None):
    if name not in backend_registry:
        raise ValueError(
            f'backend name={name} not registered, registered backends={backend_registry}'
        )
    py_executor = backend_registry[name].func(executor_config, checkpoint_dir,
                                              engine_dir)
    return py_executor


def create_py_executor(name: str,
                       checkpoint_dir: str = None,
                       engine_dir: str = None,
                       executor_kwargs=None,
                       additional_kwargs=None):
    if executor_kwargs is None:
        executor_kwargs = {}
    executor_config = ExecutorConfig(**executor_kwargs)
    for key, value in additional_kwargs.items():
        assert not hasattr(executor_config,
                           key), f'key={key} already exists in executor_config'
        setattr(executor_config, key, value)
    return create_py_executor_by_config(name, checkpoint_dir, engine_dir,
                                        executor_config)


def unique_create_executor(model_path: os.PathLike,
                           model_type: ModelType,
                           executor_config: ExecutorConfig,
                           device_id: int = torch.cuda.current_device()):
    if not hasattr(executor_config, "backend"):
        engine = Executor(model_path, model_type, executor_config)
    else:
        torch.cuda.set_device(device_id)
        engine = create_py_executor_by_config(executor_config.backend,
                                              executor_config.hf_model_dir,
                                              executor_config.trt_engine_dir,
                                              executor_config)
    return engine
