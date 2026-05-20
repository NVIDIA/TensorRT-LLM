# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error types for ``compile_backend='grafia'``."""


class GrafiaCompileError(RuntimeError):
    """Base error for ``compile_backend='grafia'``."""


class GrafiaUnsupportedError(GrafiaCompileError):
    """Raised when Grafia cannot lower or execute a selected region."""
