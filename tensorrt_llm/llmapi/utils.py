import asyncio
import collections
import datetime
import hashlib
import inspect
import io
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import warnings
import weakref
from functools import cache, wraps
from pathlib import Path
from queue import Queue
from typing import (Any, Callable, Iterable, List, Optional, Tuple, Type,
                    get_type_hints)

import filelock
import huggingface_hub
import psutil
import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from tqdm.auto import tqdm

from tensorrt_llm.logger import Singleton, logger


def print_traceback_on_error(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger_debug(f"Exception in {func.__name__}: {e}\n", "red")
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


def get_current_location(skip_frames: int = 2) -> str:
    """
    Get the current execution location in format 'module.class.function'.

    Args:
        skip_frames: Number of stack frames to skip (default 2 to skip this function and its caller)

    Returns:
        String in format 'module.class.function' or 'module.function' if not in a class
    """
    stack = inspect.stack()
    if len(stack) <= skip_frames:
        return "unknown"

    frame = stack[skip_frames]
    module_name = frame.frame.f_globals.get('__name__', 'unknown')
    function_name = frame.function

    # Try to determine if we're in a class method
    class_name = None
    if 'self' in frame.frame.f_locals:
        # This is likely an instance method
        obj = frame.frame.f_locals['self']
        class_name = obj.__class__.__name__
    elif 'cls' in frame.frame.f_locals:
        # This might be a class method
        cls = frame.frame.f_locals['cls']
        if inspect.isclass(cls):
            class_name = cls.__name__

    # Build the location string
    if class_name:
        return f"{module_name}.{class_name}.{function_name}"
    else:
        return f"{module_name}.{function_name}"


def logger_debug(message,
                 color: Optional[str] = None,
                 writer: io.TextIOWrapper = sys.stderr):
    """ Print the message if the llmapi debug mode is enabled. Fallback to logger.debug if not. """
    if enable_llmapi_debug():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location = get_current_location()
        cur_dualname = "..." + location[-47:] if len(
            location) > 50 else location
        print_colored(f"{timestamp} [{cur_dualname}]", "bold_green", writer)
        print_colored(f" {message}\n", color, writer)
    else:
        # Fallback to logger.debug
        logger.debug(message)


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
    ignore_patterns = ["original/**/*"]
    with get_file_lock(model):
        hf_folder = snapshot_download(
            model,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_patterns=ignore_patterns,
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
                 stop_event: Optional[threading.Event] = None,
                 **kwargs):
        super().__init__(name=name)
        self.task = task
        self.error_queue = error_queue
        self.kwargs = kwargs
        self.daemon = True
        self.stop_event = stop_event or threading.Event()

    def run(self):

        while not self.stop_event.is_set():
            task = self.task
            if isinstance(task, weakref.WeakMethod):
                task = task()
                if task is None:
                    # Normally, this should not happen.
                    logger.warning("WeakMethod is expired.")
                    break

            try:
                if not task(**self.kwargs):
                    break
            except Exception as e:
                logger.error(
                    f"Error in thread {self.name}: {e}\n{traceback.format_exc()}"
                )
                self.error_queue.put(e)

        logger.info(f"Thread {self.name} stopped.")

    def stop(self):
        self.stop_event.set()


_enable_llm_debug_ = None


def enable_llm_debug() -> bool:
    ''' Tell whether to enable the debug mode for LLM class.  '''
    global _enable_llm_debug_
    if _enable_llm_debug_ is None:
        _enable_llm_debug_ = os.environ.get("TLLM_LLM_ENABLE_DEBUG", "0") == "1"
    return _enable_llm_debug_


_enable_llmapi_debug_ = None


def enable_llmapi_debug() -> bool:
    global _enable_llmapi_debug_
    if _enable_llmapi_debug_ is None:
        _enable_llmapi_debug_ = os.environ.get("TLLM_LLMAPI_ENABLE_DEBUG",
                                               "0") == "1"
    return _enable_llmapi_debug_


@cache
def enable_worker_single_process_for_tp1() -> bool:
    ''' Tell whether to make worker use single process for TP1.
    This is helpful for return-logits performance and debugging. '''
    return os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS", "0") == "1"


class AsyncQueue:
    """
    A queue-style container that provides both sync and async interface.
    This is used to provide a compatible interface for janus.Queue.
    """

    class EventLoopShutdownError(Exception):
        pass

    class MixedSyncAsyncAPIError(Exception):
        pass

    def __init__(self):
        self._q = collections.deque()
        self._event = asyncio.Event()
        self._tainted = False
        self._sync_q = _SyncQueue(self)

    @property
    def sync_q(self):
        return self._sync_q

    def full(self) -> bool:
        return len(self._q) == self._q.maxlen

    def empty(self) -> bool:
        return not self._q

    def put(self, item) -> None:
        self._q.append(item)
        self._event.set()

    # Decoupled put and notify.
    # Deque is thread safe so we can put from outside the event loop.
    # However, we have to schedule notify in event loop because it's not thread safe.
    # In this case the notify part may get scheduled late, to the point that
    # corresponding data in deque may have already been consumed.
    # Avoid firing the event in this case.
    def put_nowait(self, item) -> None:
        self._q.append(item)

    def notify(self) -> None:
        if self._q:
            self._event.set()

    def unsafe_get(self):
        # Unsafe get taints the queue, renders it unusable by async methods.
        self._tainted = True
        # Use exception to detect empty. Pre-check is not thread safe.
        try:
            return self._q.popleft()
        except IndexError:
            raise asyncio.QueueEmpty() from None

    async def get(self, timeout=None):
        if self._tainted:
            raise AsyncQueue.MixedSyncAsyncAPIError()

        # Blocking path: timeout is None (wait indefinitely)
        if timeout is None:
            # Wait indefinitely until the queue is non-empty.
            # It is necessary to check if the queue is empty after waking.
            # Because multiple waiting coroutines may be awakened simultaneously when a new item entries empty queue.
            # These coroutines will all pop this item from queue, and then raise IndexError.
            while not self._q:
                await self._event.wait()
        # Blocking path: timeout > 0 (timed wait, retry with remaining time).
        elif timeout > 0:
            # Compute the deadline; if the queue is still empty after waking, continue waiting for the remaining time.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while not self._q:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                # This may raise asyncio.TimeoutError.
                await asyncio.wait_for(self._event.wait(), timeout=remaining)
        # Non-blocking path: timeout <= 0.
        elif not self._q:
            raise asyncio.QueueEmpty()

        res = self._q.popleft()
        if not self._q:
            self._event.clear()
        return res


class _SyncQueue:
    """
    A simplified Queue that provides a `put` method that is compatible with the asyncio event loop.
    """

    def __init__(self,
                 queue: "AsyncQueue",
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        self._aq = queue
        self._loop = loop or asyncio.get_event_loop()

    async def _notify(self):
        self._aq.notify()

    def put(self, item) -> None:
        self._aq.put_nowait(item)

        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._notify(), self._loop)
        else:
            raise AsyncQueue.EventLoopShutdownError()

    def put_nowait(self, item) -> None:
        """ Put item without notify the event. """
        self._aq.put_nowait(item)

    # Notify many queues in one coroutine, to cut down context switch overhead.
    @staticmethod
    async def _notify_many(queues: Iterable["_SyncQueue"]):
        for queue in queues:
            queue._aq.notify()

    @staticmethod
    def notify_many(loop: asyncio.AbstractEventLoop,
                    queues: List["_SyncQueue"]) -> None:
        """ Notify the events in the loop. """

        if loop.is_running():
            asyncio.run_coroutine_threadsafe(
                _SyncQueue._notify_many(frozenset(queues)), loop)
        else:
            raise AsyncQueue.EventLoopShutdownError()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def full(self) -> bool:
        return self._aq.full()

    def get(self, timeout=None):
        # Here is the WAR for jupyter scenario where trt-llm detects the event loop existence.
        # However, this event loop launched by jupyter rather than trt-llm. It led the GenerationResult initialized
        # w/ AsyncQueue and call the get() unintentionally.

        warnings.warn(
            "LLM API is running in async mode because you have a running event loop,"
            " but you are using sync API. This may lead to potential performance loss."
        )

        # We can't call asyncio.run_coroutine_threadsafe(self._aq.get(), self.loop) and wait the returned Future,
        # since we are in the same event loop, and we can't yield the thread while waiting result.
        deadline = None if timeout is None else time.time() + timeout
        while deadline is None or time.time() < deadline:
            try:
                return self._aq.unsafe_get()
            except asyncio.QueueEmpty:
                time.sleep(0.01)


def set_sched_setaffinity(required_cores: int):
    ''' Set the CPU affinity of the current process to the required number of
    cores.

    Known issue: This may race with other processes that also set the affinity.
    '''
    cpu_percentages = psutil.cpu_percent(percpu=True)
    # sort the cores by usage
    free_cores = sorted(range(len(cpu_percentages)),
                        key=lambda i: cpu_percentages[i])

    pid = os.getpid()
    os.sched_setaffinity(pid, set(free_cores[:required_cores]))


def clear_sched_affinity(pid: int):
    ''' Clear the CPU affinity of the current process. '''
    os.sched_setaffinity(pid, set(range(psutil.cpu_count())))


def generate_api_docs_as_docstring(model: Type[BaseModel],
                                   include_annotations=False,
                                   indent="") -> str:
    """
    Generates API documentation for a Pydantic BaseModel, formatted as a
    Python docstring.

    Args:
        model: The Pydantic BaseModel class.

    Returns:
        A string containing the API documentation formatted as a docstring.
    """
    docstring_lines = []

    if include_annotations:
        # Class docstring
        if model.__doc__:
            docstring_lines.append(model.__doc__.strip())
            docstring_lines.append(
                "")  # Add a blank line after the class docstring

        docstring_lines.append(f"{indent}Args:")

    schema = model.schema()
    type_hints = get_type_hints(model)
    type_alias = {
        'integer': 'int',
        'number': 'float',
        'boolean': 'bool',
        'string': 'str',
        'array': 'list',
    }

    for field_name, field_info in schema['properties'].items():
        if field_name.startswith("_"):  # skip private fields
            continue
        if field_info.get("status", None) == "deprecated":
            continue

        field_type = field_info.get('type', None)
        field_description = field_info.get('description', '')
        field_default = field_info.get('default', None)
        field_required = field_name in schema.get('required', [])

        # Get full type path from type hints if available
        if field_type:
            type_str = type_alias.get(field_type, field_type)
        elif field_name in type_hints:
            type_str = str(type_hints[field_name])
            type_str = type_str.replace("typing.", "")
            # Extract just the class name from full class path
            for regex in [r"<class '([^']+)'>", r"<enum '([^']+)'>"]:
                if (match := re.match(regex, type_str)) is not None:
                    type_str = match.group(1)
                    break
        else:
            type_str = field_type or 'Any'

        # Format the argument documentation with 12 spaces indent for args
        arg_line = f"{indent}    {field_name} ({type_str}): "
        if status := field_info.get("status", None):
            arg_line += f":tag:`{status}` "
        elif LABEL_STABLE_APIS:
            arg_line += f":tag:`stable` "

        if field_description:
            arg_line += field_description.split('\n')[0]  # First line with type

        docstring_lines.append(arg_line)

        # Add remaining description lines and default value with 16 spaces indent
        if field_description and '\n' in field_description:
            remaining_lines = field_description.split('\n')[1:]
            for line in remaining_lines:
                docstring_lines.append(f"{indent}        {line}")

        if not field_required or field_default is not None:
            default_str = str(
                field_default) if field_default is not None else "None"
            docstring_lines[-1] += f" Defaults to {default_str}."

    if include_annotations:
        docstring_lines.append("")  # Empty line before Returns
        return_annotation = "None"  # Default to None, adjust if needed
        docstring_lines.append(
            f"{indent}Returns:\n{indent}    {return_annotation}")

    return "\n".join(docstring_lines)


def get_type_repr(cls):
    """Handle built-in types gracefully. """
    module_name = cls.__module__
    if module_name == 'builtins':  # Special case for built-in types
        return cls.__qualname__
    return f"{module_name}.{cls.__qualname__}"


LABEL_STABLE_APIS: bool = True
""" Whether to label the stable APIs with `stable` tags. """


class ApiParamTagger:
    ''' A helper to tag the api doc according to the status of the fields.
    The status is set in the json_schema_extra of the field.
    '''

    def __call__(self, cls: Type[BaseModel]) -> None:
        """ The main entry point to tag the api doc. """
        if cls.__name__ in ["LlmArgs", "TorchLlmArgs"]:
            # TODO: apply this to other classes
            self._process_pydantic_model(cls)

    def _process_pydantic_model(self, cls: Type[BaseModel]) -> None:
        """Process the Pydantic model to add tags to the fields.
        """
        for field_name, field_info in cls.model_fields.items():
            if field_info.json_schema_extra and 'status' in field_info.json_schema_extra:
                status = field_info.json_schema_extra['status']
                self._amend_pydantic_field_description_with_tags(
                    cls, [field_name], status)
            else:
                self._amend_pydantic_field_description_with_tags(
                    cls, [field_name], "stable")

    def _amend_pydantic_field_description_with_tags(self, cls: Type[BaseModel],
                                                    field_names: list[str],
                                                    tag: str) -> None:
        """Amend the description of the fields with tags.
        e.g. :tag:`beta` or :tag:`prototype`
        Args:
            cls: The Pydantic BaseModel class.
            field_names: The names of the fields to amend.
            tag: The tag to add to the fields.
        """
        assert field_names
        for field_name in field_names:
            field = cls.model_fields[field_name]
            cls.model_fields[
                field_name].description = f":tag:`{tag}` {field.description}"
        cls.model_rebuild(force=True)


def tag_llm_params():
    from tensorrt_llm.llmapi.llm_args import LlmArgs
    ApiParamTagger()(LlmArgs)


class ApiStatusRegistry:
    ''' A registry to store the status of the api.

    usage:

    @ApiStatusRegistry.set_api_status("beta")
    def my_method(self, *args, **kwargs):
        pass

    class App:
        @ApiStatusRegistry.set_api_status("beta")
        def my_method(self, *args, **kwargs):
            pass
    '''
    method_to_status = {}

    @classmethod
    def set_api_status(cls, status: str):

        def decorator(func):
            # Use qualified name to support class methods
            if func.__qualname__ in cls.method_to_status:
                logger.debug(
                    f"Method {func.__qualname__} already has a status, skipping the decorator"
                )
                return func
            cls.method_to_status[func.__qualname__] = status
            func.__doc__ = cls.amend_api_doc_with_status_tags(func)
            return func

        return decorator

    @classmethod
    def get_api_status(cls, method: Callable) -> Optional[str]:
        return cls.method_to_status.get(method.__qualname__, None)

    @classmethod
    def amend_api_doc_with_status_tags(cls, method: Callable) -> str:
        status = cls.get_api_status(method)
        if status is None:
            return method.__doc__
        return f":tag:`{status}` {method.__doc__}"


set_api_status = ApiStatusRegistry().set_api_status
get_api_status = ApiStatusRegistry().get_api_status
