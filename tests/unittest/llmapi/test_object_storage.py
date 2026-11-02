# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `tensorrt_llm.llmapi.object_storage`, the metadata cache of object-storage models.

Model Streamer is replaced by a fake whose ``ObjectStorageModel`` keeps the
completion-marker behavior of the real one: it downloads only when the marker
is missing, and writes the marker after a download.
"""

import fcntl
import io
import os
import shutil
import sys
import threading
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from tensorrt_llm.llmapi import object_storage

pytestmark = pytest.mark.cpu_only

URI = "s3://bucket/models/kimi"
SENTINEL = ".runai_complete"


class _FakeObjectStorageModel:
    """Stands in for ``runai_model_streamer.ObjectStorageModel``."""

    pulls: list[tuple[str, list[str] | None]] = []

    def __init__(self, model_path: str, dst: str) -> None:
        self.model_path = model_path
        self.dir = dst
        self._skip = os.path.exists(os.path.join(dst, SENTINEL))
        if not self._skip:
            shutil.rmtree(dst, ignore_errors=True)
            os.makedirs(dst)

    def pull_files(self, allow_pattern=None, ignore_pattern=None) -> None:
        if self._skip:
            return
        type(self).pulls.append((self.model_path, ignore_pattern))
        Path(self.dir, "config.json").write_text("{}")

    def __enter__(self) -> "_FakeObjectStorageModel":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None and not self._skip:
            Path(self.dir, SENTINEL).touch()
        return False


@pytest.fixture(autouse=True)
def streamer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[_FakeObjectStorageModel]:
    _FakeObjectStorageModel.pulls = []
    module = ModuleType("runai_model_streamer")
    module.ObjectStorageModel = _FakeObjectStorageModel
    monkeypatch.setitem(sys.modules, "runai_model_streamer", module)
    monkeypatch.setenv("TRTLLM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv(object_storage.FORCE_PULL_ENV, raising=False)
    # The helper exports the token it creates; restore the variable afterwards.
    monkeypatch.setenv(object_storage.PULL_TOKEN_ENV, "")
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    return _FakeObjectStorageModel


@pytest.mark.parametrize(
    "path",
    [
        "s3://bucket/model",
        "gs://bucket/model",
        "az://container/model",
        "S3://Bucket/Model",
    ],
)
def test_object_storage_uris_are_detected(path: str) -> None:
    assert object_storage.is_object_storage_uri(path)


@pytest.mark.parametrize(
    "path",
    [
        "/data/model",
        "relative/model",
        "meta-llama/Llama-3.1-8B",
        "https://example.com/model",
        # A Path has already collapsed "s3://" to "s3:/".
        Path("s3://bucket/model"),
        None,
    ],
)
def test_other_paths_are_not_object_storage_uris(path) -> None:
    assert not object_storage.is_object_storage_uri(path)


def test_cache_dir_is_a_deterministic_function_of_the_uri(tmp_path: Path) -> None:
    first = object_storage.get_cache_dir(URI)

    assert object_storage.get_cache_dir(URI) == first
    assert object_storage.get_cache_dir("s3://bucket/models/other") != first
    assert first.parent == tmp_path / "cache" / "object_storage"


def test_cache_dir_defaults_to_the_home_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TRTLLM_CACHE_DIR")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cache_dir = object_storage.get_cache_dir(URI)

    assert cache_dir.parent == tmp_path / ".cache" / "tensorrt_llm" / "object_storage"


def test_pull_metadata_skips_the_weights(streamer: type[_FakeObjectStorageModel]) -> None:
    cache_dir = object_storage.pull_metadata(URI)

    assert cache_dir == object_storage.get_cache_dir(URI)
    assert (cache_dir / "config.json").is_file()
    [(uri, ignored)] = streamer.pulls
    assert uri == URI
    assert {"*.safetensors", "*.bin", "*.pt", "*.pth", "*.tensors"} <= set(ignored)
    # Other formats that hold a whole model are not copied either.
    assert {"*.gguf", "*.onnx", "*.onnx_data", "*.h5", "*.msgpack", "*.ckpt"} <= set(ignored)


def test_pull_metadata_does_not_download_a_complete_cache_again(
    streamer: type[_FakeObjectStorageModel],
) -> None:
    object_storage.pull_metadata(URI)
    object_storage.pull_metadata(URI)

    assert len(streamer.pulls) == 1


def test_pull_metadata_downloads_again_after_an_interrupted_download(
    streamer: type[_FakeObjectStorageModel],
) -> None:
    cache_dir = object_storage.pull_metadata(URI)
    # An interrupted download leaves no completion marker.
    (cache_dir / SENTINEL).unlink()

    object_storage.pull_metadata(URI)

    assert len(streamer.pulls) == 2


def test_pull_metadata_rejects_a_path_that_is_not_object_storage() -> None:
    with pytest.raises(ValueError, match="Not an object storage URI"):
        object_storage.pull_metadata("/data/model")


def test_pull_metadata_needs_a_streamer_with_object_storage_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "runai_model_streamer", ModuleType("runai_model_streamer"))

    with pytest.raises(ImportError, match="ObjectStorageModel"):
        object_storage.pull_metadata(URI)


def test_forced_pull_downloads_once_for_each_token(
    streamer: type[_FakeObjectStorageModel],
) -> None:
    cache_dir = object_storage.pull_metadata(URI)
    assert len(streamer.pulls) == 1

    # The cache is complete, but a forced launch downloads it again.
    object_storage.pull_metadata(URI, force_token="launch-1")
    assert len(streamer.pulls) == 2
    # Another process of the same launch finds the cache current.
    object_storage.pull_metadata(URI, force_token="launch-1")
    assert len(streamer.pulls) == 2
    assert (cache_dir / ".trtllm_pull_token").read_text() == "launch-1"
    # A later launch forces again.
    object_storage.pull_metadata(URI, force_token="launch-2")
    assert len(streamer.pulls) == 3
    # A launch that does not force keeps the cache.
    object_storage.pull_metadata(URI)
    assert len(streamer.pulls) == 3


def test_forced_pull_waits_for_a_process_that_uses_the_cache(
    streamer: type[_FakeObjectStorageModel],
) -> None:
    cache_dir = object_storage.pull_metadata(URI)
    # A Model Streamer process that works in the directory holds a shared lock.
    reader = open(f"{cache_dir}.lock", "a")
    fcntl.flock(reader, fcntl.LOCK_SH)
    forced = threading.Thread(target=object_storage.pull_metadata, args=(URI, "launch-1"))
    forced.start()

    forced.join(timeout=0.5)
    # The forced pull waits, and the directory it would delete is intact.
    assert forced.is_alive()
    assert (cache_dir / "config.json").is_file()
    assert len(streamer.pulls) == 1

    fcntl.flock(reader, fcntl.LOCK_UN)
    reader.close()
    forced.join(timeout=10)
    assert not forced.is_alive()
    assert len(streamer.pulls) == 2


def test_forced_pull_does_not_hold_the_cache_lock_while_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_storage.pull_metadata(URI)
    lock_was_free = []

    class _CheckingObjectStorageModel(_FakeObjectStorageModel):
        def __init__(self, model_path: str, dst: str) -> None:
            # The real class takes this lock itself. A lock that the caller still holds
            # would block it for good, so a non-blocking request must succeed here.
            with open(f"{dst}.lock", "a") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_was_free.append(True)
            super().__init__(model_path, dst)

    monkeypatch.setattr(
        sys.modules["runai_model_streamer"], "ObjectStorageModel", _CheckingObjectStorageModel
    )

    object_storage.pull_metadata(URI, force_token="launch-1")

    assert lock_was_free == [True]


def test_force_pull_token_is_none_unless_forced() -> None:
    assert object_storage.get_force_pull_token() is None


def test_force_pull_token_is_shared_by_the_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(object_storage.FORCE_PULL_ENV, "1")

    token = object_storage.get_force_pull_token()

    assert token
    # Child processes inherit it, and later calls return it.
    assert os.environ[object_storage.PULL_TOKEN_ENV] == token
    assert object_storage.get_force_pull_token() == token


def test_attached_frontend_never_forces_a_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(object_storage.FORCE_PULL_ENV, "1")
    monkeypatch.setenv("TLLM_EXECUTOR_ATTACH_INFO", "{}")

    assert object_storage.get_force_pull_token() is None


@pytest.mark.parametrize(
    ("fs_type", "warns"), [("nfs4", True), ("lustre", True), ("ext4", False), ("xfs", False)]
)
def test_network_filesystem_cache_dir_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fs_type: str, warns: bool
) -> None:
    mounts = f"rootfs / ext4 rw 0 0\nserver:/export {tmp_path.resolve()} {fs_type} rw 0 0\n"
    monkeypatch.setattr(object_storage, "open", lambda *_: io.StringIO(mounts), raising=False)
    logger = MagicMock()
    monkeypatch.setattr(object_storage, "logger", logger)

    object_storage._warn_if_network_filesystem(tmp_path / "cache")

    assert logger.warning.called is warns
