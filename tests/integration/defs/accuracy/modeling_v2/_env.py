# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-scoped environment overrides for accuracy tests."""

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


@contextmanager
def scoped_env_overrides(overrides: Mapping[str, str]) -> Iterator[dict[str, str]]:
    """Apply rank overrides locally and restore their prior values on exit.

    Pass the yielded mapping to ``LLM(env_overrides=...)`` inside this scope,
    so rank workers receive the same environment and engine shutdown completes
    before the parent environment is restored. Only the overridden keys are
    restored; unrelated environment changes are preserved.
    """
    values = dict(overrides)
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield values
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
