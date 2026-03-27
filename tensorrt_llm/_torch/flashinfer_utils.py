import os
import platform
import traceback

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def _flashinfer_pdl_supported_on_current_device() -> bool:
    """Return whether the active GPU can run FlashInfer JIT with PDL enabled.

    PDL (programmatic dependent launch) lowers to PTX ``griddepcontrol`` and
    related modifiers that require Hopper (sm_90) or newer. Compiling those
    kernels for sm_80/sm_86 fails at ptxas time, so we gate PDL by arch.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major >= 9


def get_env_enable_pdl() -> bool:
    """Whether FlashInfer norm/activation ops should use PDL (``enable_pdl=``).

    ``TRTLLM_ENABLE_PDL``: unset or ``1`` means "want PDL when the GPU supports
    it"; ``0`` / ``false`` / ``off`` / ``no`` forces off. On GPUs below sm_90,
    PDL is always off regardless of env, so CUTLASS DSL JIT does not emit
    unsupported instructions.
    """
    raw = os.environ.get("TRTLLM_ENABLE_PDL", "1").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False

    if not _flashinfer_pdl_supported_on_current_device():
        # Distinguish explicit TRTLLM_ENABLE_PDL=1 on Ampere (warn once) vs default path (info once).
        env_explicit = "TRTLLM_ENABLE_PDL" in os.environ
        if env_explicit and raw in ("1", "true", "on", "yes"):
            if not getattr(get_env_enable_pdl, "_unsupported_warned", False):
                logger.warning(
                    "TRTLLM_ENABLE_PDL requests PDL but this GPU is below sm_90; "
                    "FlashInfer kernels will run with PDL disabled.")
                setattr(get_env_enable_pdl, "_unsupported_warned", True)
        elif not getattr(get_env_enable_pdl, "_disabled_info", False):
            logger.info(
                "FlashInfer PDL disabled: requires CUDA sm_90+ (Hopper or newer)."
            )
            setattr(get_env_enable_pdl, "_disabled_info", True)
        return False

    # Hopper+: honor env and log once that PDL is active.
    if not getattr(get_env_enable_pdl, "_printed", False):
        logger.info("PDL enabled")
        setattr(get_env_enable_pdl, "_printed", True)
    return True


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
