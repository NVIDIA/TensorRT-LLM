import os
import tempfile

from tensorrt_llm.logger import logger
from typing import Callable
from typing_extensions import ParamSpec

def set_prometheus_multiproc_dir():
    # Set prometheus multiprocess directory
    # sglang uses prometheus multiprocess mode
    # we need to set this before importing prometheus_client
    # https://prometheus.github.io/client_python/multiprocess/
    global prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        logger.info("User set PROMETHEUS_MULTIPROC_DIR detected.")
        prometheus_multiproc_dir = tempfile.TemporaryDirectory(
            dir=os.environ["PROMETHEUS_MULTIPROC_DIR"]
        )
    else:
        prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
    logger.info(f"PROMETHEUS_MULTIPROC_DIR: {os.environ['PROMETHEUS_MULTIPROC_DIR']}")

P = ParamSpec('P')
#From: https://stackoverflow.com/a/4104188/2749989
def run_once(f: Callable[P, None]) -> Callable[P, None]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        if not wrapper.has_run:  # type: ignore[attr-defined]
            wrapper.has_run = True  # type: ignore[attr-defined]
            return f(*args, **kwargs)
    wrapper.has_run = False  # type: ignore[attr-defined]
    return wrapper
