"""Load the NVFP4 CUDA extension only when a kernel-backed operation runs."""

from __future__ import annotations

from types import ModuleType


def require_nvfp4() -> ModuleType:
    """Return the usable NVFP4 extension or explain why it is unavailable."""
    try:
        from ltx_kernels import nvfp4 as nvfp4_mod  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "ltx-kernels not built; NVFP4 quantization requires the nvfp4 extension. "
            "Build on a CUDA host: TORCH_CUDA_ARCH_LIST='10.0' "
            "uv sync --group kernels (or "
            "`uv pip install -e packages/ltx-kernels --no-build-isolation`)."
        ) from error
    if reason := nvfp4_mod.unavailable_reason():
        raise RuntimeError(reason)
    return nvfp4_mod
