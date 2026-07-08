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
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pytest
from _pytest.logging import catching_logs

logger = logging.getLogger(__name__)


class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if envvar:
            if envvar in os.environ:
                default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class FDRedirector:
    def __init__(
        self,
        target_fd,
        log_file_path,
        echo_to_original=False,
        timestamp_format="%(asctime)s.%(msecs)03d",
        date_format="%Y-%m-%d %H:%M:%S",
    ):
        self.target_fd = target_fd
        self.log_file_path = log_file_path
        self.echo_to_original = echo_to_original
        self.timestamp_format = timestamp_format
        self.date_format = date_format

        self.saved_fd = None
        self._reader_thread = None

    def _flush_target_stream(self):
        streams = []
        if self.target_fd == 1:
            streams = [sys.stdout]
        elif self.target_fd == 2:
            streams = [sys.stderr]
        else:
            streams = [sys.stdout, sys.stderr]

        for stream in streams:
            try:
                stream.flush()
            except Exception:
                pass

    def __enter__(self):
        # Drain any pending buffered writes to the current target fd before we
        # redirect it. sys.stdout/sys.stderr are block-buffered when connected
        # to a pipe; without this, late flushes (e.g. of a previous test's
        # post-teardown print) would land in this test's file after we swap
        # the underlying fd.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass

        log_file = open(self.log_file_path, "w", encoding="utf-8", buffering=1)
        pipe_read, pipe_write = os.pipe()
        self.saved_fd = os.dup(self.target_fd)
        os.dup2(pipe_write, self.target_fd)
        os.close(pipe_write)

        pipe_stream = os.fdopen(pipe_read, "r", encoding="utf-8", errors="replace", buffering=1)

        # Child processes may inherit the redirected fd and keep the pipe open
        # after capture is restored. Do not let that block pytest shutdown.
        self._reader_thread = threading.Thread(
            target=self._reader_loop, args=(pipe_stream, log_file), daemon=True
        )
        self._reader_thread.start()

        return self

    def _reader_loop(self, pipe_stream, log_file):
        need_timestamp = True

        try:
            while True:
                chunk = pipe_stream.read(4096)
                if not chunk:
                    break

                # Build the timestamp prefix once per chunk and reuse it for every
                # line in that chunk. The previous implementation walked the chunk
                # char-by-char in Python, which was the bottleneck under high-volume
                # output (e.g. tqdm progress bars, repetitive kernel info logs).
                # Within a single 4 KiB chunk the timestamp would have been
                # identical anyway (the reader processes the chunk under the GIL),
                # so reusing one prefix is semantics-preserving.
                now = datetime.now()
                ts_str = self.timestamp_format % {
                    "asctime": now.strftime(self.date_format),
                    "msecs": now.microsecond // 1000,
                }
                prefix = f"[{ts_str}] "

                parts = []
                for line in chunk.splitlines(keepends=True):
                    if need_timestamp:
                        parts.append(prefix)
                    parts.append(line)
                    need_timestamp = line.endswith("\n")
                log_file.write("".join(parts))
                log_file.flush()

                if self.echo_to_original and self.saved_fd is not None:
                    ret = os.write(self.saved_fd, chunk.encode("utf-8"))
                    if ret != len(chunk):
                        logger.warning(
                            f"Partial write to original FD {self.target_fd}: {ret} != {len(chunk)}"
                        )

        except Exception as e:
            logger.error(f"Error reading from pipe: {e}")
        finally:
            log_file.close()
            pipe_stream.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.saved_fd is not None:
            self._flush_target_stream()
            # Restore the original fd binding. This also closes the previous
            # binding (our pipe's write end), so the reader thread will see
            # EOF and finish draining.
            os.dup2(self.saved_fd, self.target_fd)
            # NB: keep self.saved_fd valid until the reader thread joins;
            # the reader may still drain pending bytes and want to echo them
            # through saved_fd. Closing it here would race with that write.

        if self._reader_thread:
            self._reader_thread.join(timeout=5.0)
            if self._reader_thread.is_alive():
                logger.warning(f"Reader thread for FD {self.target_fd} did not exit in time")

        if self.saved_fd is not None:
            os.close(self.saved_fd)
            self.saved_fd = None

        return False


class DirectFDRedirector:
    def __init__(self, target_fd, log_file_path):
        self.target_fd = target_fd
        self.log_file_path = log_file_path

        self.saved_fd = None
        self._log_file = None

    def _flush_target_stream(self):
        streams = []
        if self.target_fd == 1:
            streams = [sys.stdout]
        elif self.target_fd == 2:
            streams = [sys.stderr]
        else:
            streams = [sys.stdout, sys.stderr]

        for stream in streams:
            try:
                stream.flush()
            except Exception:
                pass

    def __enter__(self):
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass

        self._log_file = open(self.log_file_path, "w", encoding="utf-8", buffering=1)
        self.saved_fd = os.dup(self.target_fd)
        os.dup2(self._log_file.fileno(), self.target_fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._flush_target_stream()

        if self.saved_fd is not None:
            os.dup2(self.saved_fd, self.target_fd)
            os.close(self.saved_fd)
            self.saved_fd = None

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

        return False


class UploadLogPlugin:
    def __init__(
        self,
        endpoint_url,
        aws_access_key_id,
        aws_secret_access_key,
        bucket,
        upload_path,
        output_path,
        echo_to_stdout=False,
        skip_upload=False,
        capture_mode="timestamped",
        upload_mode="sync",
        upload_workers=8,
        inline_output_max_bytes=256,
    ):
        self.upload_path = upload_path
        self.output_path = output_path
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.echo_to_stdout = echo_to_stdout
        self.skip_upload = skip_upload
        self.capture_mode = capture_mode
        self.upload_mode = upload_mode
        self.upload_workers = max(1, upload_workers)
        self.inline_output_max_bytes = inline_output_max_bytes
        if self.inline_output_max_bytes < 0:
            raise ValueError("--s3-inline-output-max-bytes must be >= 0")
        if self.capture_mode == "direct" and self.echo_to_stdout:
            raise ValueError("--s3-capture-mode=direct cannot be used with --s3-echo-stdout")
        self.s3 = None
        if not self.skip_upload:
            try:
                import boto3
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "boto3 is required when --s3-upload-path is set without --s3-skip-upload"
                ) from exc
            self.s3 = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        # nodeid -> dict of open capture context managers + handler.
        # Capture must span setup + call (so fixture output is recorded) but
        # not teardown — closing the capture before teardown lets us add
        # upload metadata inside pytest_runtest_logreport(teardown), where
        # `report.sections.append(...)` is still seen by the junitxml plugin
        # (pluggy LIFO order makes our handler fire before junitxml's).
        self._active_capture: dict = {}
        self._test_names: dict = {}
        self._deferred_uploads = []

    def normalize_test_name(self, nodeid):
        import hashlib

        test_name = re.sub(r"[^\w\-]", "_", nodeid)
        suffix = hashlib.md5(nodeid.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        # Linux limits a single path component to 255 bytes.
        if len(test_name) > 200:
            test_name = test_name[:200]
        return f"{test_name}-{suffix}-{timestamp}"

    def _open_capture(self, item):
        """Open FDRedirector + logging capture for ``item``.

        Returns a state dict on success, or ``None`` if setup failed (in which
        case the test runs uncaptured). Never propagates exceptions.
        """
        state = {}
        try:
            test_name = self.normalize_test_name(item.nodeid)
            state["test_name"] = test_name
            output_path = os.path.join(self.output_path, test_name)
            os.makedirs(output_path, exist_ok=True)

            stdout_file = os.path.join(output_path, "stdout.log")
            stderr_file = os.path.join(output_path, "stderr.log")
            log_file = os.path.join(output_path, "logging.log")

            log_date_format = item.config.getini("log_date_format")
            log_format = item.config.getini("log_format")

            timestamp_format = None
            if log_format:
                match = re.search(r"\[([^\]]*%\(asctime\)s[^\]]*)\]", log_format)
                if match:
                    timestamp_format = match.group(1)

            handler = logging.FileHandler(log_file)
            logging_plugin = item.config.pluginmanager.getplugin("logging-plugin")
            if logging_plugin is not None:
                handler.setFormatter(logging_plugin.formatter)
            elif log_format:
                formatter = logging.Formatter(log_format, datefmt=log_date_format)
                handler.setFormatter(formatter)
            state["handler"] = handler

            fd_kwargs = {}
            if log_date_format:
                fd_kwargs["date_format"] = log_date_format
            if timestamp_format:
                fd_kwargs["timestamp_format"] = timestamp_format

            if self.capture_mode == "direct":
                state["stdout_redir"] = DirectFDRedirector(1, stdout_file)
            else:
                state["stdout_redir"] = FDRedirector(
                    1, stdout_file, echo_to_original=self.echo_to_stdout, **fd_kwargs
                )
            state["stdout_redir"].__enter__()
            if self.capture_mode == "direct":
                state["stderr_redir"] = DirectFDRedirector(2, stderr_file)
            else:
                state["stderr_redir"] = FDRedirector(
                    2, stderr_file, echo_to_original=self.echo_to_stdout, **fd_kwargs
                )
            state["stderr_redir"].__enter__()
            state["log_cm"] = catching_logs(handler)
            state["log_cm"].__enter__()
            return state
        except Exception as e:
            logger.warning(
                "S3 capture setup failed for %r: %s; running without capture", item.nodeid, e
            )
            self._close_capture(state)
            return None

    def _close_capture(self, state):
        """Close capture state opened by ``_open_capture``. Never raises."""
        if not state:
            return
        # Close in reverse order so fd 1/2 are restored before logging stops.
        for key in ("log_cm", "stderr_redir", "stdout_redir"):
            cm = state.get(key)
            if cm is None:
                continue
            try:
                cm.__exit__(None, None, None)
            except Exception as e:
                logger.warning("Error closing %s capture: %s", key, e)
        handler = state.get("handler")
        if handler is not None:
            try:
                handler.close()
            except Exception:
                pass

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_setup(self, item):
        state = self._open_capture(item)
        if state is not None:
            self._active_capture[item.nodeid] = state
            self._test_names[item.nodeid] = state["test_name"]
        yield

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_teardown(self, item, nextitem):
        # Stop capturing BEFORE teardown runs, so the captured log files are
        # final by the time we upload in pytest_runtest_logreport(teardown).
        self._close_capture(self._active_capture.pop(item.nodeid, None))
        yield

    def get_file_size(self, path):
        try:
            return os.path.getsize(path)
        except FileNotFoundError:
            return None

    def _object_key(self, test_name, filename):
        return os.path.join(self.upload_path, test_name, filename)

    def _file_url(self, object_key):
        return os.path.join(
            self.endpoint_url,
            "v1/AUTH_" + self.aws_access_key_id,
            self.bucket,
            object_key,
        )

    def _upload_file(self, filepath, object_key):
        self.s3.upload_file(
            filepath,
            self.bucket,
            object_key,
            ExtraArgs={"ContentType": "text/plain"},
        )

    def _append_upload_failed(self, report, section_name, filepath, filesize, error):
        with open(filepath, "r", encoding="utf-8") as f:
            limit = 65536
            # Limit content to 64k (65536 bytes)
            trail_content = "... [truncated]"
            content = f.read(limit + 1)
            if len(content) > limit:
                content = content[: limit - len(trail_content)] + trail_content
        report.sections.append(
            (
                section_name,
                f"""upload failed: {error}\nsize: {filesize} bytes\ncontent: {content}""",
            )
        )

    def _should_inline_output(self, filename, filesize):
        return filename in ("stdout.log", "stderr.log") and filesize < self.inline_output_max_bytes

    def _append_inline_output(self, report, section_name, filepath):
        with open(filepath, "rb") as f:
            content = f.read().decode("utf-8", errors="replace")
        report.sections.append((section_name, content))

    def upload_and_report(self, report, test_name, filename, section_name):
        filepath = os.path.join(self.output_path, test_name, filename)
        if not os.path.exists(filepath):
            report.sections.append((section_name, "<not exist>"))
            return
        filesize = os.path.getsize(filepath)
        if filesize == 0:
            report.sections.append((section_name, "<empty>"))
            return
        if self._should_inline_output(filename, filesize):
            self._append_inline_output(report, section_name, filepath)
            return
        object_key = self._object_key(test_name, filename)
        fileurl = self._file_url(object_key)
        if self.skip_upload:
            # Experiment knob: skip the actual S3 upload to measure how much
            # of this plugin's per-test overhead comes from boto3 network IO.
            report.sections.append(
                (
                    section_name,
                    f"{filesize} bytes (upload skipped, would upload to {fileurl})",
                )
            )
            return
        if self.upload_mode == "deferred":
            self._deferred_uploads.append((filepath, object_key, test_name, filename))
            report.sections.append(
                (
                    section_name,
                    f"{filesize} bytes scheduled for upload to {fileurl}",
                )
            )
            return
        try:
            self._upload_file(filepath, object_key)
            report.sections.append(
                (
                    section_name,
                    f"{filesize} bytes uploaded to {fileurl}",
                )
            )
        except Exception as e:
            logger.warning(
                f"Upload failed. test_name: {test_name}, filename: {filename}, error: {e}"
            )
            self._append_upload_failed(report, section_name, filepath, filesize, e)

    def pytest_runtest_logreport(self, report):
        if report.when == "teardown":
            test_name = self._test_names.pop(report.nodeid, None)
            if test_name is None:
                test_name = self.normalize_test_name(report.nodeid)
            # Add S3 report sections here so they are visible to the junitxml
            # plugin's logreport handler. Pluggy hook order is LIFO; this
            # plugin is registered after junitxml, so our handler runs first.
            self.upload_and_report(report, test_name, "stdout.log", "Captured stdout")
            self.upload_and_report(report, test_name, "stderr.log", "Captured stderr")
            self.upload_and_report(report, test_name, "logging.log", "Captured log")

    def pytest_sessionfinish(self, session, exitstatus):
        if self.skip_upload or not self._deferred_uploads:
            return
        workers = min(self.upload_workers, len(self._deferred_uploads))
        logger.info(
            "Uploading %d deferred S3 test log files with %d workers",
            len(self._deferred_uploads),
            workers,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._upload_file, filepath, object_key): (
                    filepath,
                    object_key,
                    test_name,
                    filename,
                )
                for filepath, object_key, test_name, filename in self._deferred_uploads
            }
            for future in as_completed(futures):
                filepath, object_key, test_name, filename = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning(
                        "Deferred upload failed. test_name: %s, filename: %s, "
                        "object_key: %s, filepath: %s, error: %s",
                        test_name,
                        filename,
                        object_key,
                        filepath,
                        e,
                    )
        self._deferred_uploads.clear()


def add_options(parser):
    """Register S3 CLI options. Call from pytest_addoption in any conftest that needs S3 upload."""
    parser.addoption(
        "--s3-endpoint",
        action=EnvDefault,
        envvar="S3_ENDPOINT",
        default="https://pbss.s8k.io",
        help="s3 endpoint",
    )
    parser.addoption(
        "--s3-username",
        action=EnvDefault,
        envvar="S3_USERNAME",
        default="svc_tensorrt",
        help="s3 username",
    )
    parser.addoption(
        "--s3-secret-key",
        action=EnvDefault,
        envvar="S3_SECRET_KEY",
        required=False,
        help="s3 secret key",
    )
    parser.addoption(
        "--s3-bucket",
        action=EnvDefault,
        envvar="S3_BUCKET",
        default="trtllm-ci-logs",
        help="s3 bucket name",
    )
    parser.addoption(
        "--s3-upload-path",
        action=EnvDefault,
        envvar="S3_UPLOAD_PATH",
        required=False,
        help="s3 upload path",
    )
    parser.addoption(
        "--s3-echo-stdout",
        action="store_true",
        default=False,
        help="Besides capturing stdout/stderr to per-test log files, also echo "
        "them through to the original stdout/stderr (e.g. so progress stays "
        "visible in the CI console). Should be set on the outer pytest "
        "invocation; nested pytest invocations spawned by individual tests "
        "should NOT set this, to avoid duplicating their output back through "
        "the outer pipe.",
    )
    parser.addoption(
        "--s3-skip-upload",
        action="store_true",
        default=False,
        help="Experiment knob: still capture stdout/stderr/log per test and "
        "append URLs to report sections, but skip the actual s3.upload_file "
        "call. Used to measure how much of the plugin's per-test overhead "
        "comes from boto3/network IO vs. local capture machinery.",
    )
    parser.addoption(
        "--s3-capture-mode",
        action="store",
        choices=("timestamped", "direct"),
        default="timestamped",
        help="Capture stdout/stderr through the timestamped pipe reader, or "
        "redirect file descriptors directly to files. Direct mode is intended "
        "for nested pytest runs with many short cases and does not support "
        "--s3-echo-stdout.",
    )
    parser.addoption(
        "--s3-upload-mode",
        action="store",
        choices=("sync", "deferred"),
        default="sync",
        help="Upload each per-test log file synchronously in the test teardown, "
        "or defer uploads until pytest session finish. Deferred mode keeps "
        "per-test files and report URLs while reducing teardown latency for "
        "nested pytest runs with many cases.",
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
        help="Inline captured stdout/stderr into the test report and skip S3 "
        "upload when each file is smaller than this many bytes. Set to 0 to "
        "disable inlining for non-empty output.",
    )


def register_plugin(config):
    """Register UploadLogPlugin if --s3-upload-path and --output-dir are both set."""
    s3_upload_path = config.getoption("--s3-upload-path", default=None)
    output_dir = config.getoption("--output-dir", default=None)
    if not (s3_upload_path and output_dir):
        return
    capture_mode = config.getoption("capture", default="no")
    if capture_mode != "no":
        raise ValueError("capture mode must be 'no' when upload path is specified")
    s3_secret_key = config.getoption("--s3-secret-key")
    skip_upload = config.getoption("--s3-skip-upload", default=False)
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
        output_path=output_dir,
        echo_to_stdout=config.getoption("--s3-echo-stdout", default=False),
        skip_upload=skip_upload,
        capture_mode=config.getoption("--s3-capture-mode", default="timestamped"),
        upload_mode=config.getoption("--s3-upload-mode", default="sync"),
        upload_workers=config.getoption("--s3-upload-workers", default=8),
        inline_output_max_bytes=config.getoption("--s3-inline-output-max-bytes", default=256),
    )
    config.pluginmanager.register(plugin, "upload_log_plugin")
