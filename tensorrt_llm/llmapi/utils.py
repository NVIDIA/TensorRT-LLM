import asyncio
import hashlib
import io
import os
import sys
import tempfile
import threading
import traceback
import weakref
from functools import cache, wraps
from pathlib import Path
from queue import Queue
from typing import Any, Callable, List, Optional, Tuple

import filelock
import huggingface_hub
import torch
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from tensorrt_llm.logger import Singleton, logger


def print_traceback_on_error(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise e

    return wrapper


def print_colored(message,
                  color: Optional[str] = None,
                  writer: io.TextIOWrapper = sys.stderr):
    colors = dict(
        grey="\x1b[38;20m",
        yellow="\x1b[33;20m",
        red="\x1b[31;20m",
        bold_red="\x1b[31;1m",
        bold_green="\033[1;32m",
        green="\033[0;32m",
    )
    reset = "\x1b[0m"

    if color:
        writer.write(colors[color] + message + reset)
    else:
        writer.write(message)


def file_with_glob_exists(directory, glob) -> bool:
    path = Path(directory)
    for file_path in path.glob(glob):
        if file_path.is_file():
            return True
    return False


def file_with_suffix_exists(directory, suffix) -> bool:
    return file_with_glob_exists(directory, f'*{suffix}')


def get_device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_total_gpu_memory(device: int) -> float:
    return torch.cuda.get_device_properties(device).total_memory


class GpuArch:

    @staticmethod
    def get_arch() -> int:
        return get_gpu_arch()

    @staticmethod
    def is_post_hopper() -> bool:
        return get_gpu_arch() >= 9

    @staticmethod
    def is_post_ampere() -> bool:
        return get_gpu_arch() >= 8

    @staticmethod
    def is_post_volta() -> bool:
        return get_gpu_arch() >= 7


def get_gpu_arch(device: int = 0) -> int:
    return torch.cuda.get_device_properties(device).major


class ContextManager:
    ''' A helper to create a context manager for a resource. '''

    def __init__(self, resource):
        self.resource = resource

    def __enter__(self):
        return self.resource.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.resource.__exit__(exc_type, exc_value, traceback)


def is_directory_empty(directory: Path) -> bool:
    return not any(directory.iterdir())


class ExceptionHandler(metaclass=Singleton):

    def __init__(self):
        self._sys_excepthook: Callable = sys.excepthook
        self._obj_refs_and_callbacks: List[Tuple[weakref.ReferenceType,
                                                 str]] = []

    def __call__(self, exc_type, exc_value, traceback):
        self._sys_excepthook(exc_type, exc_value, traceback)

        for obj_ref, callback_name in self._obj_refs_and_callbacks:
            if (obj := obj_ref()) is not None:
                callback = getattr(obj, callback_name)
                callback()

    def register(self, obj: Any, callback_name: str):
        assert callable(getattr(obj, callback_name, None))
        self._obj_refs_and_callbacks.append((weakref.ref(obj), callback_name))


exception_handler = ExceptionHandler()
sys.excepthook = exception_handler

# Use the system temporary directory to share the cache
temp_dir = tempfile.gettempdir()


def get_file_lock(model_name: str,
                  cache_dir: Optional[str] = None) -> filelock.FileLock:
    # Hash the model name to avoid invalid characters in the lock file path
    hashed_model_name = hashlib.sha256(model_name.encode()).hexdigest()

    cache_dir = cache_dir or temp_dir
    os.makedirs(cache_dir, exist_ok=True)

    lock_file_path = os.path.join(cache_dir, f"{hashed_model_name}.lock")

    return filelock.FileLock(lock_file_path)


class DisabledTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def download_hf_model(model: str, revision: Optional[str] = None) -> Path:
    with get_file_lock(model):
        hf_folder = snapshot_download(
            model,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            revision=revision,
            tqdm_class=DisabledTqdm)
    return Path(hf_folder)


def download_hf_pretrained_config(model: str,
                                  revision: Optional[str] = None) -> Path:
    with get_file_lock(model):
        hf_folder = snapshot_download(
            model,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            revision=revision,
            allow_patterns=["config.json"],
            tqdm_class=DisabledTqdm)
    return Path(hf_folder)


def append_docstring(docstring: str):
    ''' A decorator to append a docstring to a function. '''

    def decorator(fn):
        fn.__doc__ = (fn.__doc__ or '') + docstring
        return fn

    return decorator


def set_docstring(docstring: str):
    ''' A decorator to set a docstring to a function. '''

    def decorator(fn):
        fn.__doc__ = docstring
        return fn

    return decorator


def get_directory_size_in_gb(directory: Path) -> float:
    """ Get the size of the directory. """
    if not (directory.is_dir() and directory.exists()):
        raise ValueError(f"{directory} is not a directory.")
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024**3  # GB


class ManagedThread(threading.Thread):
    """ A thread that will put exceptions into an external queue if the task fails.

    There are two approaches to stop the thread:
        1. Set stop_event to stop the loop
        2. Let `task` return False

    Args:
        task (Callable[..., bool]): The task to run repeatedly in the thread, should return False if break the loop.
        error_queue (Queue): The queue to put exceptions into if the task fails.
        name (str): The name of the thread.
        **kwargs: The arguments to pass to the task
    """

    def __init__(self,
                 task: Callable[..., bool],
                 error_queue: Queue,
                 name: Optional[str] = None,
                 **kwargs):
        super().__init__(name=name)
        self.task = task
        self.error_queue = error_queue
        self.kwargs = kwargs
        self.daemon = True

        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                if not self.task(**self.kwargs):
                    break
            except Exception as e:
                logger.error(
                    f"Error in thread {self.name}: {e}\n{traceback.format_exc()}"
                )
                self.error_queue.put(e)

        logger.info(f"Thread {self.name} stopped.")

    def stop(self):
        self.stop_event.set()


@cache
def enable_llm_debug() -> bool:
    ''' Tell whether to enable the debug mode for LLM class.  '''
    return os.environ.get("TLLM_LLM_ENABLE_DEBUG", "0") == "1"


class AsyncQueue:
    '''
    AsyncQueue is container containing `async_q` for `async get` and `sync_q` for sync `get`.
    This is used to provide a compatible interface for janus.Queue.
    '''

    class EventLoopShutdownError(Exception):
        pass

    def __init__(self):
        self._q = Queue()
        self.async_q = _AsyncQueue(self._q)
        self.sync_q = _SyncQueue(self._q, self.async_q._event)


class _SyncQueue:
    '''
    A simplified Queue that provides a `put` method that is compatible with the asyncio event loop.
    '''

    def __init__(self,
                 queue: Queue,
                 event: asyncio.Event,
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        self._q = queue
        self._event = event
        self._loop = loop or asyncio.get_event_loop()

    def put(self, item) -> None:

        async def _set_event(event):
            event.set()

        self._q.put_nowait(item)

        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(_set_event(self._event),
                                             self._loop)
        else:
            raise AsyncQueue.EventLoopShutdownError

    def put_nowait(self, item) -> None:
        ''' Put item without notify the event. '''
        self._q.put_nowait(item)

    @staticmethod
    def notify_events(loop: asyncio.AbstractEventLoop,
                      events: List[asyncio.Event]) -> None:
        ''' Notify the events in the loop. '''

        async def _set_events(events):
            for event in events:
                event.set()

        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_set_events(events), loop)
        else:
            raise AsyncQueue.EventLoopShutdownError

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    @property
    def event(self) -> asyncio.Event:
        return self._event

    def full(self) -> bool:
        return self._q.full()

    def get(self, timeout=None):
        # Here is the WAR for jupyter scenario where trt-llm detects the event loop existence.
        # However, this event loop launched by jupyter rather than trt-llm. It led the GenerationResult initialized
        # w/ AsyncQueue and call the get() unintentionally.
        res = self._q.get()
        if self._q.empty():
            self._event.clear()
        return res


class _AsyncQueue:
    '''
    A simplified asyncio.Queue that provides a `get` method that is compatible with the standard library Queue.
    '''

    def __init__(self, queue: Queue):
        self._event = asyncio.Event()
        self._q = queue

    async def get(self, timeout=None):
        # This may raise asyncio.TimeoutError
        await asyncio.wait_for(self._event.wait(), timeout=timeout)

        res = self._q.get()
        if self._q.empty():
            self._event.clear()
        return res
