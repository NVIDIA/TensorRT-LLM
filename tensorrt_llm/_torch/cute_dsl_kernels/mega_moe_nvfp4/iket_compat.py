# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for optional ``cutlass.cute.iket`` support."""

import logging

_logger = logging.getLogger(__name__)

# IKET (In-Kernel Event Tracing) markers are only available in cutlass-dsl
# wheels that ship the ``iket`` dialect. Functional tests do not need the
# dialect, so fall back to no-op markers when the import is unavailable.
try:
    from cutlass.cute.experimental import iket  # Latest tot DKG.
except (ImportError, NotImplementedError
        ):  # pragma: no cover -- fallback for wheels without cute.iket
    # ``cute.experimental`` raises NotImplementedError (NOT ImportError) on
    # CUDA toolkits < 13.1, so the public-release / CTK-12.9 CI wheels land
    # here; catch both so the no-op shim below actually takes over instead
    # of propagating up through the caller's ImportError-only guard.
    try:
        from cutlass.cute import iket  # type: ignore
    except (ImportError, NotImplementedError):
        _logger.debug("IKET dialect not available; using no-op IKET shim.")

        class _IketShim:
            """No-op IKET shim used when the dialect is not available."""

            @staticmethod
            def range_push(_name, *_args, **_kwargs):
                return None

            @staticmethod
            def range_pop(*_args, **_kwargs):
                return None

            @staticmethod
            def range_start(_name, *_args, **_kwargs):
                return None

            @staticmethod
            def range_end(_token=None, *_args, **_kwargs):
                return None

            @staticmethod
            def mark(_name, *_args, **_kwargs):
                return None

        iket = _IketShim()  # type: ignore
