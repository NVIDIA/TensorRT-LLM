import functools
import os
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, Callable, List, Literal

from tensorrt_llm.quantization.mode import QuantAlgo

VALID_MODELS = Literal[
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
    "tiiuae/falcon-180B",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "EleutherAI/gpt-j-6b",
]
VALID_COMPUTE_DTYPES = Literal["auto", "float16", "bfloat16"]
VALID_CACHE_DTYPES = Literal["float16", "float8", "int8"]
VALID_QUANT_ALGOS = Literal[f"{QuantAlgo.W8A16}", f"{QuantAlgo.W4A16}",
                            f"{QuantAlgo.W4A16_AWQ}", f"{QuantAlgo.W4A8_AWQ}",
                            f"{QuantAlgo.W4A16_GPTQ}", f"{QuantAlgo.FP8}",
                            f"{QuantAlgo.INT8}", f"{QuantAlgo.NVFP4}"]
VALID_SCHEDULING_POLICIES = \
    Literal["max_utilization", "guaranteed_no_evict", "static"]


class _MethodFunctionAdapter:
    """An adapter class for running decorators on both methods and functions.

    Found here: https://stackoverflow.com/a/1288936 with help of ChatGPT. This
    works via the following logic.

    1. During function declaration, store the decorator and function in an
    instance of this class using `detect_methods`. Works for both functions
    and methods because a method will be a reference to `Class.method` at
    declaration time.
    2. The __call__ method makes this class callable. In the case of functions,
    the wrapper will simply call the decorator as a wrapper function simply
    passing arguments without accessing a class descriptor.
    3. The __get__ method is part of the descriptor protocol for Python classes.
    In the case of running a method, the call will access the property of a
    class instance which has been wrapped by this class. __get__ overrides are
    used to control how a class property is returned. In this case, we would
    like to return the method reference wrapped in the decorator.
    """

    def __init__(self, decorator, func):
        self.decorator = decorator
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.decorator(self.func)(*args, **kwargs)

    def __get__(self, instance, owner):
        return self.decorator(self.func.__get__(instance, owner))


def detect_methods(decorator):
    """Decorator for applying a wrapper to both methods and functions."""

    def apply_wrapper(func):
        return _MethodFunctionAdapter(decorator, func)

    return apply_wrapper


def command_logger(prefix: str = "") -> Callable:
    """Logs the command for functions that call subprocesses.

    NOTE: This helper assumes the command is in the first argument.

    Args:
        func (Callable): Function whose first argument is a list of arguments.
        prefix (str, optional): Prefix to prepend to command. Defaults to "".

    Returns:
        Callable: Function that includes command logging.
    """

    @detect_methods
    def inner_wrapper(func):

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # Append the prefix and join the command.
            print(f"{prefix}{' '.join(args[0])}")
            return func(*args, **kwargs)

        return wrapped

    return inner_wrapper


@detect_methods
def process_error_check(func: Callable) -> subprocess.CompletedProcess:
    """Logs standard error and raises an exception on failed processes.

    Args:
        func (Callable): Callable that returns a CompletedProcess.

    Returns:
        subprocess.CompletedProcess: Returns a completed process just as
        an unwrapped `subprocess.run` would.

    Raises:
        RuntimeError: If the wrapped function returns a non-zero error code.
    """

    @functools.wraps(func)
    def runtime_check(*args, **kwargs):
        finished_process = func(*args, **kwargs)

        if finished_process.returncode != 0:
            raise RuntimeError(
                "Process failed. Output below.\n"
                "================================================================\n"
                f"{finished_process.stderr}")

        return finished_process

    return runtime_check


def run_process(cmd: List[Any],
                run_dir: Path = None,
                use_environ: bool = False,
                stderr_on_stdout: bool = False) -> subprocess.CompletedProcess:
    """Utility function for launching processes.

    Args:
        cmd (List[Any]): A list of arguments that must be able to be cast to a string.
        run_dir (Path, optional): The directory to run the process from. Defaults to None.
        use_environ (bool, optional): Use the environment of the container to run the process. Necessary for any commands that start with `mpirun`, as mpi4py initializes its own MPI environment
        stderr_on_stdout (bool, optional): Pipe STDERR to STDOUT. Defaults to False.

    Returns:
        subprocess.CompletedProcess: _description_
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if stderr_on_stdout else subprocess.PIPE,
        env=os.environ if use_environ else None,
        text=True,
    )
    return result
