# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def enabled_fmha_libraries() -> tuple[str, ...]:
    """Read the shared TRTLLM FMHA registry in enabled priority order."""
    from ...attention.backends.fmha.registry import FMHA_LIBS, get_enabled_fmha_lib_classes

    names = {cls: name for name, cls in FMHA_LIBS.items()}
    return tuple(names[cls] for cls in get_enabled_fmha_lib_classes())
