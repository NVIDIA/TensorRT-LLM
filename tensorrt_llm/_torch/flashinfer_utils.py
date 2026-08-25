import os
import platform
import traceback

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False

# IS_FLASHINFER_AVAILABLE is a package-level probe and cannot answer for these
# two: flashinfer builds its CuTe-DSL kernels only when its own ``import
# cutlass.cute`` succeeds, so the package imports fine (mamba2_mixer hard-depends
# on selective_state_update) while both symbols are absent. Resolved here, next
# to the flag they qualify, so no consumer re-derives the contract -- whether as
# an import target or as a dispatch predicate.
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
