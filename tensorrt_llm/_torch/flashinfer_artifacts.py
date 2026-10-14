# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Report the FlashInfer kernel artifacts that could not be fetched at runtime.

FlashInfer fetches kernel artifacts (checksum files, headers, cubins) the first
time it needs them, from its cubin directory or by downloading them. A fetch
that fails returns empty bytes instead of raising, so the failure surfaces later
and somewhere else: as an assertion while FlashInfer generates a module, as
"Failed to load cubin" from an attention kernel, as "Error occurred when running
GEMM!" from a MoE kernel, or not at all. None of these names the file.

``install_flashinfer_artifact_monitor`` records every fetch that fails and logs
it with the file name, without changing what FlashInfer does.
``raise_on_flashinfer_artifact_failure`` turns the fetches that failed inside a
block into one ``FlashInferArtifactUnavailableError``.
"""

import collections
import contextlib
import functools
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import ModuleType

from ..logger import logger

__all__ = [
    "FlashInferArtifactFailure",
    "FlashInferArtifactSource",
    "FlashInferArtifactUnavailableError",
    "install_flashinfer_artifact_monitor",
    "raise_on_flashinfer_artifact_failure",
]

# Bounds the memory of a process whose fetches keep failing. A block that sees
# more failures than this reports the most recent ones.
_RECENT_FAILURE_LIMIT = 64


@dataclass(frozen=True)
class FlashInferArtifactSource:
    """Where FlashInfer reads kernel artifacts, and where it downloads missing ones from."""

    flashinfer_version: str
    cubin_dir: str
    repository: str


@dataclass(frozen=True)
class FlashInferArtifactFailure:
    """A kernel artifact that FlashInfer could not fetch."""

    file_name: str
    """Path of the artifact relative to the cubin directory."""

    reason: str


class FlashInferArtifactUnavailableError(RuntimeError):
    """FlashInfer could not fetch kernel artifacts that it downloads at runtime.

    Attributes:
        failures: The distinct fetches that failed, oldest first.
        source: Where FlashInfer looked for the artifacts.
    """

    def __init__(
        self,
        failures: tuple[FlashInferArtifactFailure, ...],
        source: FlashInferArtifactSource,
    ) -> None:
        self.failures = failures
        self.source = source
        super().__init__(_describe_failures(failures, source))


def _describe_source(source: FlashInferArtifactSource) -> str:
    return (
        f"FlashInfer {source.flashinfer_version} reads these artifacts from {source.cubin_dir}"
        f" and downloads missing ones from {source.repository}. To run without runtime"
        f" downloads, install flashinfer-cubin=={source.flashinfer_version}, or point"
        " FLASHINFER_CUBIN_DIR at a populated cubin directory and set FLASHINFER_NO_DOWNLOAD=1."
    )


def _describe_failures(
    failures: tuple[FlashInferArtifactFailure, ...], source: FlashInferArtifactSource
) -> str:
    fetched = "; ".join(f"{failure.file_name} ({failure.reason})" for failure in failures)
    message = (
        "FlashInfer could not fetch kernel artifacts that it downloads at runtime:"
        f" {fetched}. {_describe_source(source)}"
    )
    if any(failure.file_name.endswith(".cubin") for failure in failures):
        message += (
            " Some FlashInfer kernels keep a module that failed to load cached, so they can"
            " stay unusable until the process restarts."
        )
    return message


class _ArtifactFetchMonitor:
    """Stands in for FlashInfer's ``get_artifact`` and keeps the fetches that failed.

    Calls the original with the same arguments, returns what it returns and
    re-raises what it raises, so FlashInfer behaves as before.
    """

    def __init__(
        self, get_artifact: Callable[..., bytes], source: FlashInferArtifactSource
    ) -> None:
        functools.update_wrapper(self, get_artifact)
        self.source = source
        self._get_artifact = get_artifact
        self._lock = threading.Lock()
        self._failure_count = 0
        self._recent_failures: collections.deque[tuple[int, FlashInferArtifactFailure]] = (
            collections.deque(maxlen=_RECENT_FAILURE_LIMIT)
        )

    def __call__(self, file_name: str, sha256: str, *args: object, **kwargs: object) -> bytes:
        try:
            artifact = self._get_artifact(file_name, sha256, *args, **kwargs)
        except Exception as error:
            # Whatever FlashInfer raises is recorded, then re-raised unchanged.
            self._record(file_name, f"{type(error).__name__}: {error}")
            raise
        if not artifact:
            self._record(file_name, "download failed or sha256 mismatch, see the FlashInfer log")
        return artifact

    def mark(self) -> int:
        """Return a mark that ``failures_since`` reports later failures against."""
        with self._lock:
            return self._failure_count

    def failures_since(self, mark: int) -> tuple[FlashInferArtifactFailure, ...]:
        """Return the distinct fetches that failed after ``mark`` was taken, oldest first."""
        with self._lock:
            failures = [failure for number, failure in self._recent_failures if number > mark]
        return tuple(dict.fromkeys(failures))

    def _record(self, file_name: str, reason: str) -> None:
        failure = FlashInferArtifactFailure(file_name=file_name, reason=reason)
        with self._lock:
            self._failure_count += 1
            self._recent_failures.append((self._failure_count, failure))
        logger.error(
            f"FlashInfer could not fetch {file_name}: {reason}. {_describe_source(self.source)}"
        )


_monitor: _ArtifactFetchMonitor | None = None


def install_flashinfer_artifact_monitor() -> None:
    """Record every FlashInfer artifact fetch that fails, and log it with the file name.

    Every runtime download in FlashInfer goes through its ``get_artifact``, so this
    wraps that one function for ``raise_on_flashinfer_artifact_failure``. Calling it
    again does nothing. If this FlashInfer version has no ``get_artifact``, logs a
    warning and leaves FlashInfer as is.
    """
    try:
        import flashinfer
        from flashinfer.jit import cubin_loader
    except ImportError as error:
        logger.warning(f"FlashInfer artifact download failures will not be reported: {error}")
        return
    _install_monitor(cubin_loader, flashinfer.__version__, sys.modules)


def _install_monitor(
    cubin_loader: ModuleType,
    flashinfer_version: str,
    loaded_modules: Mapping[str, ModuleType | None],
) -> None:
    """Replace ``cubin_loader.get_artifact``, and every module-level reference to it.

    Args:
        cubin_loader: FlashInfer's ``flashinfer.jit.cubin_loader`` module.
        flashinfer_version: Version reported in the error messages.
        loaded_modules: Modules searched for references to ``get_artifact``,
            normally ``sys.modules``. Only those in FlashInfer's package are changed.
    """
    global _monitor

    get_artifact = getattr(cubin_loader, "get_artifact", None)
    if isinstance(get_artifact, _ArtifactFetchMonitor):
        return
    if get_artifact is None:
        logger.warning(
            "FlashInfer artifact download failures will not be reported:"
            f" {cubin_loader.__name__}.get_artifact was not found (flashinfer {flashinfer_version})."
        )
        return
    monitor = _ArtifactFetchMonitor(
        get_artifact,
        FlashInferArtifactSource(
            flashinfer_version=flashinfer_version,
            cubin_dir=str(getattr(cubin_loader, "FLASHINFER_CUBIN_DIR", "<unknown>")),
            repository=str(getattr(cubin_loader, "FLASHINFER_CUBINS_REPOSITORY", "<unknown>")),
        ),
    )
    # The kernel loaders' cubin callback looks the name up in cubin_loader when it
    # runs, but several modules hold their own reference from importing it by name,
    # and cubin_loader keeps an alias. Modules imported after this see the monitor.
    package = cubin_loader.__name__.partition(".")[0]
    package_modules = [
        module
        for name, module in list(loaded_modules.items())
        if module is not None and name.partition(".")[0] == package
    ]
    for module in [cubin_loader, *package_modules]:
        for attribute, value in list(vars(module).items()):
            if value is get_artifact:
                setattr(module, attribute, monitor)
    _monitor = monitor


@contextlib.contextmanager
def raise_on_flashinfer_artifact_failure() -> Iterator[None]:
    """Raise if a FlashInfer artifact fetch fails while the block runs.

    Only fetches that fail inside the block are reported, so a later unrelated
    error is not blamed on an earlier failure. If the block raised, its error is
    the cause of the raised one; if no fetch failed, the block's own error
    propagates unchanged. Does nothing until the monitor is installed.

    Raises:
        FlashInferArtifactUnavailableError: A fetch failed inside the block.
    """
    monitor = _monitor
    if monitor is None:
        yield
        return
    mark = monitor.mark()
    try:
        yield
    except FlashInferArtifactUnavailableError:
        raise
    except Exception as error:
        # A failed fetch can surface as any exception type, from FlashInfer or a kernel.
        failures = monitor.failures_since(mark)
        if failures:
            raise FlashInferArtifactUnavailableError(failures, monitor.source) from error
        raise
    failures = monitor.failures_since(mark)
    if failures:
        raise FlashInferArtifactUnavailableError(failures, monitor.source)
