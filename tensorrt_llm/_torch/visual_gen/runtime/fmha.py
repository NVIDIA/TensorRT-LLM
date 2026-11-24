# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace


def enabled_fmha_libraries() -> tuple[str, ...]:
    """Read the shared TRTLLM FMHA registry in enabled priority order."""
    from ...attention.backends.fmha.registry import FMHA_LIBS, get_enabled_fmha_lib_classes

    names = {cls: name for name, cls in FMHA_LIBS.items()}
    return tuple(names[cls] for cls in get_enabled_fmha_lib_classes())


def is_context_prims_available(name: str) -> bool:
    """Check a registered PrimTS provider for standard, non-MLA context attention."""
    from ...attention.backends.fmha.registry import FMHA_LIBS

    layer = SimpleNamespace(is_mla_enable=False, skip_correction_threshold=0.0)
    return FMHA_LIBS[name].is_available(layer)
