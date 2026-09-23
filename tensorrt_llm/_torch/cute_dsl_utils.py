import platform

from ..logger import logger

IS_CUTLASS_DSL_AVAILABLE = False

# Whether the public CuTe DSL package provides the SM107/Rubin helper module.
# Rubin kernels stay disabled when it is absent and callers retain their
# existing fallback paths.
# TODO: flips to True once a Rubin-capable CuTe DSL package ships in the image.
IS_CUTLASS_DSL_RUBIN_AVAILABLE = False
# The Rubin fused FC1+FC2 MoE kernel (cute_dsl_kernels/rubin/moe/
# rubin_contiguous_grouped_blockscaled_gemm_fused_fc12.py) needs a newer CuTe
# DSL build than the bare rubin_helpers probe guarantees: it reaches
# ``cutlass.memory`` / ``cutlass.tensor_utils`` as real submodules, and older
# builds (e.g. 0.3.0+20260518) that lack them also miscompile its MXFP8
# block-scaled path even when the attribute access is shimmed. Treat the
# presence of both submodules as the capability marker for that kernel.
IS_CUTLASS_DSL_FUSED_FC12_AVAILABLE = False
# The SM100/SM103 (data-center Blackwell) port of the fused FC1+FC2 MoE kernel
# (cute_dsl_kernels/blackwell/moe/blackwell_contiguous_grouped_blockscaled_gemm_fused_fc12.py)
# only needs the public Blackwell helpers plus the same ``cutlass.memory`` /
# ``cutlass.tensor_utils`` submodules; it does not need ``rubin_helpers``.
IS_CUTLASS_DSL_FUSED_FC12_BLACKWELL_AVAILABLE = False

if platform.system() != "Windows":
    try:
        import cutlass  # noqa
        import cutlass.cute as cute  # noqa
        logger.info(f"cutlass dsl is available")
        IS_CUTLASS_DSL_AVAILABLE = True

        try:
            import cutlass.utils.rubin_helpers  # noqa
        except ImportError:
            pass
        else:
            logger.info("cutlass dsl Rubin helpers are available")
            IS_CUTLASS_DSL_RUBIN_AVAILABLE = True

        if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            try:
                import cutlass.memory  # noqa
                import cutlass.tensor_utils  # noqa
                logger.info(
                    "cutlass dsl supports the Rubin fused FC12 MoE kernel")
                IS_CUTLASS_DSL_FUSED_FC12_AVAILABLE = True
            except ImportError:
                logger.info(
                    "cutlass dsl build is too old for the Rubin fused "
                    "FC12 MoE kernel (no cutlass.memory / cutlass.tensor_utils); "
                    "MXFP8 CuTe DSL MoE is disabled")
        try:
            import cutlass.memory  # noqa
            import cutlass.tensor_utils  # noqa
            import cutlass.utils.blackwell_helpers  # noqa
            import cutlass.utils.blockscaled_layout  # noqa
            logger.info(
                "cutlass dsl supports the Blackwell (SM100/SM103) fused FC12 MoE kernel"
            )
            IS_CUTLASS_DSL_FUSED_FC12_BLACKWELL_AVAILABLE = True
        except ImportError:
            logger.info(
                "cutlass dsl build is too old for the Blackwell fused FC12 MoE "
                "kernel (needs cutlass.memory / cutlass.tensor_utils / "
                "cutlass.utils.blackwell_helpers); MXFP8 CuTe DSL MoE on "
                "SM100/SM103 is disabled")
    except ImportError:
        pass


def install_cutlass_dsl_compatibility() -> None:
    """Restore CuTe aliases required by pinned third-party FA4 and QuACK."""
    if not IS_CUTLASS_DSL_AVAILABLE:
        return

    import cutlass.cute as cute

    for name in ("ThrCopy", "ThrMma"):
        if not hasattr(cute.core, name) and hasattr(cute, name):
            setattr(cute.core, name, getattr(cute, name))
    if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
        cute.make_fragment = cute.make_rmem_tensor
