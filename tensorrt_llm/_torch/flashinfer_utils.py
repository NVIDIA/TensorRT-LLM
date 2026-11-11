import functools
import os
import platform
import traceback
from typing import Any, Callable

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def get_env_enable_pdl() -> bool:
    enabled = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    if enabled and not getattr(get_env_enable_pdl, "_printed", False):
        logger.info("PDL enabled")
        setattr(get_env_enable_pdl, "_printed", True)
    return enabled


if platform.system() != "Windows":
    try:
        import flashinfer
        logger.info(f"flashinfer is available: {flashinfer.__version__}")
        IS_FLASHINFER_AVAILABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )

if IS_FLASHINFER_AVAILABLE:
    from .flashinfer_cubin_pool import install_shared_cubin_pool

    install_shared_cubin_pool()


def queue_flashinfer_module_build(name: str, gen_spec: Callable[[],
                                                                Any]) -> bool:
    """Queue a build of the FlashInfer JIT module ``gen_spec()`` describes on a
    KernelBuilder worker thread; False when nothing was queued.

    The worker only builds, under the lock FlashInfer's own ``build_and_load``
    takes, so the first call that needs the module finds it built, or waits on
    that lock, and loads it on its own thread.
    """
    if not IS_FLASHINFER_AVAILABLE:
        return False
    from .kernel_builder import KernelBuilder

    return KernelBuilder.get().submit(
        "flashinfer", name, functools.partial(_build_flashinfer_module,
                                              gen_spec))


def _build_flashinfer_module(gen_spec: Callable[[], Any]) -> bool:
    if os.environ.get("FLASHINFER_DISABLE_JIT"):
        return False
    spec = gen_spec()
    if getattr(spec, "is_aot", False) or spec.is_compiled:
        return False
    spec.build(need_lock=True)
    return True
