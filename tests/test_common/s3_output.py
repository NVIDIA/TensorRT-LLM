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
import argparse
import fcntl
import hashlib
import json
import logging
import os
import posixpath
import re
import socket
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

_CAPTURE_SECTION_PATTERN = re.compile(r"^Captured (stdout|stderr|log)(?: (setup|call|teardown))?$")
_STREAM_FILENAMES = {
    "stdout": "stdout",
    "stderr": "stderr",
    "log": "logging",
}
_SPOOL_ROOT_NAME = ".s3-spool"
_SPOOL_CONFIG_NAME = "upload-config.json"
_SPOOL_WRITE_CHARS = 1024 * 1024
_FAILED_OUTPUT_MAX_LINES = 200
_FAILED_OUTPUT_MAX_BYTES = 65536


def _spool_root(output_path: str) -> Path:
    output_root = Path(os.path.abspath(output_path))
    return output_root.parent / f"{_SPOOL_ROOT_NAME}-{output_root.name}"


class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if envvar and envvar in os.environ:
            default = os.environ[envvar]
        if required and default:
            required = False
        super().__init__(default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


@dataclass(frozen=True)
class PendingUpload:
    source_path: str
    object_key: str
    test_name: str
    filename: str


@dataclass(frozen=True)
class CapturedSection:
    content: str
    attempt: int


@dataclass
class CapturedStream:
    test_name: str
    filename: str
    source_path: str
    filesize: int = 0
    message: str = ""
    force_output: bool = False
    finalized: bool = False


def _create_s3_client(
    endpoint_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
):
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise RuntimeError("boto3 is required to upload test logs") from exc

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_empty_parents(path: Path, stop: Path) -> None:
    parent = path.parent
    while parent != stop.parent:
        try:
            parent.rmdir()
        except OSError:
            return
        if parent == stop:
            return
        parent = parent.parent


def drain_pending_uploads(output_path: str, secret_key: str | None = None) -> bool:
    """Upload files left by pytest processes that exited before session finish."""
    spool_root = _spool_root(output_path)
    if not spool_root.exists():
        return True

    config_paths = list(spool_root.rglob(_SPOOL_CONFIG_NAME))
    if not config_paths:
        return True

    secret_key = secret_key or os.environ.get("S3_SECRET_KEY")
    if not secret_key:
        logger.warning("Cannot drain S3 test logs without S3_SECRET_KEY")
        return False

    success = True
    current_host = socket.gethostname()
    for config_path in config_paths:
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                try:
                    fcntl.flock(config_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue

                config = json.load(config_file)
                owner_host = config.get("hostname")
                owner_pid = int(config.get("pid", 0))
                if owner_host != current_host or (owner_pid and _process_is_alive(owner_pid)):
                    continue

                spool_dir = config_path.parent
                source_paths = [
                    path for path in spool_dir.rglob("*") if path.is_file() and path != config_path
                ]
                config_success = True
                if source_paths:
                    client = _create_s3_client(
                        config["endpoint_url"],
                        config["aws_access_key_id"],
                        secret_key,
                    )
                    uploads = {}
                    worker_count = min(
                        max(1, int(config.get("upload_workers", 8))),
                        len(source_paths),
                    )
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        for source_path in source_paths:
                            relative_path = source_path.relative_to(spool_dir)
                            object_key = posixpath.join(config["upload_path"], *relative_path.parts)
                            future = executor.submit(
                                client.upload_file,
                                str(source_path),
                                config["bucket"],
                                object_key,
                                ExtraArgs={"ContentType": "text/plain"},
                            )
                            uploads[future] = (source_path, object_key)

                        for future in as_completed(uploads):
                            source_path, object_key = uploads[future]
                            try:
                                future.result()
                            except Exception as exc:
                                config_success = False
                                success = False
                                logger.warning(
                                    "Failed to drain S3 test log %s to %s: %s",
                                    source_path,
                                    object_key,
                                    exc,
                                )
                            else:
                                source_path.unlink(missing_ok=True)
                                _remove_empty_parents(source_path, spool_dir)

                if config_success:
                    config_path.unlink(missing_ok=True)
                    _remove_empty_parents(config_path, spool_dir)
                    try:
                        spool_dir.parent.rmdir()
                    except OSError:
                        pass
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            success = False
            logger.warning("Failed to read S3 spool config %s: %s", config_path, exc)

    return success


class UploadLogPlugin:
    def __init__(
        self,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str | None,
        bucket: str,
        upload_path: str,
        output_path: str,
        skip_upload: bool = False,
        upload_mode: str = "sync",
        upload_workers: int = 8,
        inline_output_max_bytes: int = 256,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.bucket = bucket
        self.upload_path = upload_path
        self.output_path = output_path
        self.skip_upload = skip_upload
        self.upload_mode = upload_mode
        self.upload_workers = max(1, upload_workers)
        self.inline_output_max_bytes = inline_output_max_bytes
        if self.inline_output_max_bytes < 0:
            raise ValueError("--s3-inline-output-max-bytes must be >= 0")
        if self.upload_mode not in ("sync", "deferred"):
            raise ValueError("--s3-upload-mode must be 'sync' or 'deferred'")

        self.s3 = None
        if not self.skip_upload:
            if not aws_secret_access_key:
                raise ValueError("S3 secret key is required to upload test logs")
            self.s3 = _create_s3_client(
                endpoint_url,
                aws_access_key_id,
                aws_secret_access_key,
            )

        suffix = f"{socket.gethostname()}-{os.getpid()}-{time.time_ns()}"
        self._spool_dir = str(_spool_root(output_path) / suffix)
        self._spool_config_path = os.path.join(self._spool_dir, _SPOOL_CONFIG_NAME)
        self._test_names: dict[str, str] = {}
        self._used_test_names: set[str] = set()
        self._captured_sections: dict[str, dict[tuple[str, str, int], CapturedSection]] = {}
        self._captured_streams: dict[str, dict[tuple[int, str], CapturedStream]] = {}
        self._pending_reruns: set[str] = set()
        self._upload_failed = False
        self._executor = None
        self._pending_uploads: dict[Future, PendingUpload] = {}
        self._max_pending_uploads = self.upload_workers * 2
        if not self.skip_upload:
            self._write_spool_config()
            if self.upload_mode == "deferred":
                self._executor = ThreadPoolExecutor(
                    max_workers=self.upload_workers,
                    thread_name_prefix="s3-test-log-upload",
                )

    def normalize_test_name(self, nodeid: str) -> str:
        test_name = re.sub(r"[^\w\-]", "_", nodeid)
        suffix = hashlib.md5(nodeid.encode(), usedforsecurity=False).hexdigest()[:8]
        timestamp = int(time.time())
        if len(test_name) > 200:
            test_name = test_name[:200]
        return f"{test_name}-{suffix}-{timestamp}"

    def _write_spool_config(self) -> None:
        os.makedirs(self._spool_dir, exist_ok=True)
        config = {
            "endpoint_url": self.endpoint_url,
            "aws_access_key_id": self.aws_access_key_id,
            "bucket": self.bucket,
            "upload_path": self.upload_path,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "upload_workers": self.upload_workers,
        }
        temporary_path = f"{self._spool_config_path}.{os.getpid()}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file)
        os.replace(temporary_path, self._spool_config_path)

    def _test_name(self, nodeid: str) -> str:
        test_name = self._test_names.get(nodeid)
        if test_name is None:
            test_name = self.normalize_test_name(nodeid)
            base_name = test_name
            collision_index = 1
            while test_name in self._used_test_names:
                test_name = f"{base_name}-{collision_index}"
                collision_index += 1
            self._used_test_names.add(test_name)
            self._test_names[nodeid] = test_name
        return test_name

    def _object_key(self, test_name: str, filename: str) -> str:
        return posixpath.join(self.upload_path, test_name, filename)

    def _file_url(self, object_key: str) -> str:
        return posixpath.join(
            self.endpoint_url,
            "v1/AUTH_" + self.aws_access_key_id,
            self.bucket,
            object_key,
        )

    def _write_spool_file(self, test_name: str, filename: str, content: str) -> str:
        if not self.skip_upload and not os.path.exists(self._spool_config_path):
            self._write_spool_config()
        test_dir = os.path.join(self._spool_dir, test_name)
        os.makedirs(test_dir, exist_ok=True)
        source_path = os.path.join(test_dir, filename)
        file_descriptor = os.open(
            source_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(file_descriptor, "w", encoding="utf-8", errors="replace") as output:
            for offset in range(0, len(content), _SPOOL_WRITE_CHARS):
                output.write(content[offset : offset + _SPOOL_WRITE_CHARS])
        return source_path

    def _append_spool_file(self, test_name: str, filename: str, content: str) -> str:
        if not self.skip_upload and not os.path.exists(self._spool_config_path):
            self._write_spool_config()
        test_dir = os.path.join(self._spool_dir, test_name)
        os.makedirs(test_dir, exist_ok=True)
        source_path = os.path.join(test_dir, filename)
        file_descriptor = os.open(
            source_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        with os.fdopen(file_descriptor, "a", encoding="utf-8", errors="replace") as output:
            for offset in range(0, len(content), _SPOOL_WRITE_CHARS):
                output.write(content[offset : offset + _SPOOL_WRITE_CHARS])
        return source_path

    def _remove_source(self, source_path: str) -> None:
        path = Path(source_path)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Failed to remove local S3 log file %s: %s", source_path, exc)
            return
        _remove_empty_parents(path, Path(self._spool_dir))
        try:
            Path(self._spool_dir).parent.rmdir()
        except OSError:
            pass

    def _upload_source(self, source_path: str, object_key: str) -> None:
        self.s3.upload_file(
            source_path,
            self.bucket,
            object_key,
            ExtraArgs={"ContentType": "text/plain"},
        )

    def _finish_uploads(self, futures: set[Future]) -> None:
        for future in futures:
            upload = self._pending_uploads.pop(future)
            try:
                future.result()
            except Exception as exc:
                self._upload_failed = True
                logger.warning(
                    "Deferred upload failed. test_name: %s, filename: %s, "
                    "object_key: %s, source: %s, error: %s",
                    upload.test_name,
                    upload.filename,
                    upload.object_key,
                    upload.source_path,
                    exc,
                )
            else:
                self._remove_source(upload.source_path)

    def _reap_uploads(self, block_when_full: bool = False) -> None:
        completed = {future for future in self._pending_uploads if future.done()}
        if completed:
            self._finish_uploads(completed)
        if block_when_full and len(self._pending_uploads) >= self._max_pending_uploads:
            completed, _ = wait(self._pending_uploads, return_when=FIRST_COMPLETED)
            self._finish_uploads(completed)

    def _schedule_upload(
        self,
        source_path: str,
        object_key: str,
        test_name: str,
        filename: str,
    ) -> None:
        self._reap_uploads(block_when_full=True)
        future = self._executor.submit(self._upload_source, source_path, object_key)
        self._pending_uploads[future] = PendingUpload(
            source_path=source_path,
            object_key=object_key,
            test_name=test_name,
            filename=filename,
        )

    def _tail(self, source_path: str) -> str:
        filesize = os.path.getsize(source_path)
        read_size = min(filesize, _FAILED_OUTPUT_MAX_BYTES)
        with open(source_path, "rb") as source:
            source.seek(filesize - read_size)
            tail = source.read(read_size).decode("utf-8", errors="replace")

        lines = tail.splitlines(keepends=True)
        truncated = filesize > read_size
        if len(lines) > _FAILED_OUTPUT_MAX_LINES:
            lines = lines[-_FAILED_OUTPUT_MAX_LINES:]
            truncated = True
        tail = "".join(lines)
        if truncated:
            tail = "... [truncated]\n" + tail
        return tail

    @staticmethod
    def _format_report_content(message: str, tail: str | None = None) -> str:
        if tail is None:
            return f"{message}\n"
        formatted = f"{message}\n\nLast {_FAILED_OUTPUT_MAX_LINES} lines:\n{tail}"
        if not formatted.endswith("\n"):
            formatted += "\n"
        return formatted

    @staticmethod
    def _stream_filename(stream_name: str, attempt: int) -> str:
        filename = _STREAM_FILENAMES[stream_name]
        if attempt > 1:
            filename += f"-attempt-{attempt}"
        return f"{filename}.log"

    def _stream_message(self, stream: CapturedStream) -> str:
        object_key = self._object_key(stream.test_name, stream.filename)
        file_url = self._file_url(object_key)
        if self.skip_upload:
            return f"{stream.filesize} bytes (upload skipped, would upload to {file_url})"
        return f"{stream.filesize} bytes scheduled for upload to {file_url}"

    def _append_capture(
        self,
        nodeid: str,
        attempt: int,
        stream_name: str,
        content: str,
    ) -> CapturedStream:
        streams = self._captured_streams.setdefault(nodeid, {})
        stream_key = (attempt, stream_name)
        stream = streams.get(stream_key)
        if stream is None:
            test_name = self._test_name(nodeid)
            filename = self._stream_filename(stream_name, attempt)
            source_path = self._append_spool_file(test_name, filename, content)
            stream = CapturedStream(
                test_name=test_name,
                filename=filename,
                source_path=source_path,
            )
            streams[stream_key] = stream
        else:
            if stream.finalized:
                raise RuntimeError(f"Captured output arrived after {stream.filename} was finalized")
            self._append_spool_file(stream.test_name, stream.filename, content)

        stream.filesize = os.path.getsize(stream.source_path)
        stream.message = self._stream_message(stream)
        return stream

    def _should_inline_stream(self, stream_name: str, stream: CapturedStream) -> bool:
        return (
            stream_name in ("stdout", "stderr")
            and self.inline_output_max_bytes > 0
            and stream.filesize < self.inline_output_max_bytes
        )

    def _finalize_stream(self, stream_name: str, stream: CapturedStream) -> None:
        if stream.finalized:
            return
        stream.finalized = True
        if self._should_inline_stream(stream_name, stream):
            self._remove_source(stream.source_path)
            return

        object_key = self._object_key(stream.test_name, stream.filename)
        file_url = self._file_url(object_key)
        if self.skip_upload:
            stream.message = self._stream_message(stream)
            return
        if self.upload_mode == "deferred":
            self._schedule_upload(
                stream.source_path,
                object_key,
                stream.test_name,
                stream.filename,
            )
            stream.message = f"{stream.filesize} bytes scheduled for upload to {file_url}"
            return

        try:
            self._upload_source(stream.source_path, object_key)
        except Exception as exc:
            self._upload_failed = True
            logger.warning(
                "Upload failed. test_name: %s, filename: %s, error: %s",
                stream.test_name,
                stream.filename,
                exc,
            )
            stream.message = f"upload failed: {exc}\nsize: {stream.filesize} bytes"
            stream.force_output = True
            return

        self._remove_source(stream.source_path)
        stream.message = f"{stream.filesize} bytes uploaded to {file_url}"

    def _finalize_attempt(self, nodeid: str, attempt: int) -> None:
        streams = self._captured_streams.get(nodeid, {})
        for (stream_attempt, stream_name), stream in streams.items():
            if stream_attempt == attempt:
                self._finalize_stream(stream_name, stream)

    def _finalize_node(self, nodeid: str) -> None:
        for (_, stream_name), stream in self._captured_streams.get(nodeid, {}).items():
            self._finalize_stream(stream_name, stream)

    @staticmethod
    def _attempt(report) -> int:
        rerun = getattr(report, "rerun", 0)
        return rerun + 1 if isinstance(rerun, int) and rerun >= 0 else 1

    def _process_report(self, report, default_phase: str) -> None:
        nodeid = getattr(report, "nodeid", "unknown")
        failed = getattr(report, "outcome", None) == "failed"
        if getattr(report, "outcome", None) == "rerun":
            self._pending_reruns.add(nodeid)
        attempt = self._attempt(report)
        captured_sections = self._captured_sections.setdefault(nodeid, {})
        duplicate_counts: dict[tuple[str, str], int] = {}
        section_entries = []
        represented_streams = []
        represented_stream_set = set()
        for section_name, content in report.sections:
            match = _CAPTURE_SECTION_PATTERN.fullmatch(section_name)
            if match is None:
                section_entries.append((section_name, content, None))
                continue
            stream_name = match.group(1)
            phase = match.group(2) or default_phase
            duplicate_key = (stream_name, phase)
            duplicate_index = duplicate_counts.get(duplicate_key, 0)
            duplicate_counts[duplicate_key] = duplicate_index + 1

            section_key = (stream_name, phase, duplicate_index)
            captured_section = captured_sections.get(section_key)
            if captured_section is None or captured_section.content != content:
                self._append_capture(
                    nodeid,
                    attempt,
                    stream_name,
                    content,
                )
                captured_section = CapturedSection(content, attempt)
                captured_sections[section_key] = captured_section

            stream_key = (captured_section.attempt, stream_name)
            section_entries.append((section_name, content, stream_key))
            if stream_key not in represented_stream_set:
                represented_stream_set.add(stream_key)
                represented_streams.append(stream_key)

        failure_tails = {}
        streams = self._captured_streams.get(nodeid, {})
        if failed:
            for stream_key in represented_streams:
                stream_attempt, stream_name = stream_key
                stream = streams[stream_key]
                if stream_attempt == attempt and not self._should_inline_stream(
                    stream_name, stream
                ):
                    failure_tails[stream_key] = self._tail(stream.source_path)

        if default_phase in ("teardown", "collect"):
            self._finalize_attempt(nodeid, attempt)

        transformed_sections = []
        reported_streams = set()
        for section_name, content, stream_key in section_entries:
            if stream_key is None:
                transformed_sections.append((section_name, content))
                continue
            _, stream_name = stream_key
            stream = streams[stream_key]
            if self._should_inline_stream(stream_name, stream):
                transformed_sections.append((section_name, content))
                continue
            if stream_key in reported_streams:
                continue
            reported_streams.add(stream_key)

            tail = failure_tails.get(stream_key)
            if tail is None and stream.force_output:
                tail = self._tail(stream.source_path)
            transformed_sections.append(
                (
                    f"Captured {stream_name}",
                    self._format_report_content(stream.message, tail),
                )
            )
        report.sections[:] = transformed_sections

    def pytest_runtest_logstart(self, nodeid: str, location) -> None:
        self._test_name(nodeid)

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_logreport(self, report):
        self._process_report(report, getattr(report, "when", "call"))
        return (yield)

    @pytest.hookimpl(tryfirst=True)
    def pytest_collectreport(self, report) -> None:
        self._process_report(report, "collect")
        nodeid = getattr(report, "nodeid", "unknown")
        self._test_names.pop(nodeid, None)
        self._captured_sections.pop(nodeid, None)
        self._captured_streams.pop(nodeid, None)
        self._pending_reruns.discard(nodeid)

    def pytest_runtest_logfinish(self, nodeid: str, location) -> None:
        self._finalize_node(nodeid)
        self._test_names.pop(nodeid, None)
        if nodeid in self._pending_reruns:
            self._pending_reruns.remove(nodeid)
            return
        self._captured_sections.pop(nodeid, None)
        self._captured_streams.pop(nodeid, None)

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus) -> None:
        for nodeid in tuple(self._captured_streams):
            self._finalize_node(nodeid)
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._finish_uploads(set(self._pending_uploads))
        if not self.skip_upload and not self._upload_failed:
            try:
                Path(self._spool_config_path).unlink(missing_ok=True)
                Path(self._spool_dir).rmdir()
                Path(self._spool_dir).parent.rmdir()
            except OSError:
                pass
        self._captured_sections.clear()
        self._captured_streams.clear()
        self._pending_reruns.clear()


def add_options(parser) -> None:
    """Register S3 options on a pytest parser."""
    parser.addoption(
        "--s3-endpoint",
        action=EnvDefault,
        envvar="S3_ENDPOINT",
        default="https://pbss.s8k.io",
        help="S3 endpoint",
    )
    parser.addoption(
        "--s3-username",
        action=EnvDefault,
        envvar="S3_USERNAME",
        default="svc_tensorrt",
        help="S3 username",
    )
    parser.addoption(
        "--s3-secret-key",
        action=EnvDefault,
        envvar="S3_SECRET_KEY",
        required=False,
        help="S3 secret key",
    )
    parser.addoption(
        "--s3-bucket",
        action=EnvDefault,
        envvar="S3_BUCKET",
        default="trtllm-ci-logs",
        help="S3 bucket name",
    )
    parser.addoption(
        "--s3-upload-path",
        action=EnvDefault,
        envvar="S3_UPLOAD_PATH",
        required=False,
        help="S3 upload path",
    )
    parser.addoption(
        "--s3-skip-upload",
        action="store_true",
        default=False,
        help="Transform captured output into S3 report links without uploading it.",
    )
    parser.addoption(
        "--s3-upload-mode",
        action="store",
        choices=("sync", "deferred"),
        default="sync",
        help="Upload report sections synchronously or with a bounded background pool.",
    )
    parser.addoption(
        "--s3-upload-workers",
        action="store",
        type=int,
        default=8,
        help="Maximum worker threads used by --s3-upload-mode=deferred.",
    )
    parser.addoption(
        "--s3-inline-output-max-bytes",
        action="store",
        type=int,
        default=256,
        help="Inline stdout/stderr smaller than this size instead of uploading it.",
    )


def register_plugin(config):
    """Register the report transformer when S3 output is configured."""
    existing_plugin = config.pluginmanager.getplugin("upload_log_plugin")
    if existing_plugin is not None:
        return existing_plugin

    s3_upload_path = config.getoption("--s3-upload-path", default=None)
    output_dir = config.getoption("--output-dir", default=None)
    if not (s3_upload_path and output_dir):
        return None
    capture_mode = config.getoption("capture", default="fd")
    if capture_mode != "fd":
        raise ValueError("--s3-upload-path requires pytest --capture=fd")

    skip_upload = config.getoption("--s3-skip-upload", default=False)
    s3_secret_key = config.getoption("--s3-secret-key", default=None)
    if not skip_upload and not s3_secret_key:
        raise ValueError(
            "--s3-secret-key (or S3_SECRET_KEY env var) is required when --s3-upload-path is set"
        )

    plugin = UploadLogPlugin(
        endpoint_url=config.getoption("--s3-endpoint"),
        aws_access_key_id=config.getoption("--s3-username"),
        aws_secret_access_key=s3_secret_key,
        bucket=config.getoption("--s3-bucket"),
        upload_path=s3_upload_path,
        output_path=os.path.abspath(output_dir),
        skip_upload=skip_upload,
        upload_mode=config.getoption("--s3-upload-mode", default="sync"),
        upload_workers=config.getoption("--s3-upload-workers", default=8),
        inline_output_max_bytes=config.getoption("--s3-inline-output-max-bytes", default=256),
    )
    config.pluginmanager.register(plugin, "upload_log_plugin")
    return plugin


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drain-spool", metavar="OUTPUT_PATH")
    args = parser.parse_args()
    if args.drain_spool:
        return 0 if drain_pending_uploads(args.drain_spool) else 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
