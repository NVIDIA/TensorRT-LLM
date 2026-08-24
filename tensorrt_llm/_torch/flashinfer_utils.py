import os
import platform
import traceback

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False

# flashinfer builds its CuTe-DSL kernels only when its own ``import
# cutlass.cute`` succeeds, so IS_FLASHINFER_AVAILABLE cannot answer for the
# symbols they publish: the package imports fine (mamba2_mixer hard-depends on
# selective_state_update) while these two are absent. Resolving them here, next
# to the flag they qualify, keeps callers from mistaking "package present" for
# "symbol present" -- whether as an import target or as a dispatch predicate.
FLASHINFER_SSD_COMBINED = None
FLASHINFER_CHUNK_GATED_DELTA_RULE = None


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

        import flashinfer.mamba
        FLASHINFER_SSD_COMBINED = getattr(flashinfer.mamba, "SSDCombined", None)
        FLASHINFER_CHUNK_GATED_DELTA_RULE = getattr(flashinfer,
                                                    "chunk_gated_delta_rule",
                                                    None)
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )
