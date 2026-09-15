# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors the OpenEngine adapter raises across its modules."""


class UnsupportedFeatureError(ValueError):
    """Raised when OpenEngine requests a feature this adapter cannot map."""


class AbortFailedError(RuntimeError):
    """Raised when an active request could not be aborted on the engine."""

    def __init__(self, request_id: str, cause: BaseException) -> None:
        super().__init__(f"failed to abort request '{request_id}': {cause}")
        self.request_id = request_id


__all__ = [
    "AbortFailedError",
    "UnsupportedFeatureError",
]
