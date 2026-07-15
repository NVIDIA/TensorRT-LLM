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

import io
import logging
import os
import subprocess
import sys

import pytest
from test_common.s3_output import (
    FDRedirector,
    FileSlice,
    FileSliceReader,
    SessionCapture,
    UploadLogPlugin,
)


class Report:
    def __init__(self, when=None, outcome=None):
        self.sections = []
        self.when = when
        self.outcome = outcome


class CaptureContext:
    def __init__(self):
        self.closed = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True


class RecordingS3Client:
    def __init__(self):
        self.uploads = []
        self.fileobj_uploads = []

    def upload_file(self, filepath, bucket, object_key, ExtraArgs=None):
        self.uploads.append((filepath, bucket, object_key, ExtraArgs))

    def upload_fileobj(self, fileobj, bucket, object_key, ExtraArgs=None):
        self.fileobj_uploads.append((fileobj.read(), bucket, object_key, ExtraArgs))


def make_plugin(tmp_path, inline_output_max_bytes, capture_mode="timestamped"):
    return UploadLogPlugin(
        endpoint_url="https://example.com",
        aws_access_key_id="user",
        aws_secret_access_key=None,
        bucket="bucket",
        upload_path="logs",
        output_path=str(tmp_path),
        skip_upload=True,
        capture_mode=capture_mode,
        inline_output_max_bytes=inline_output_max_bytes,
    )


def write_log(tmp_path, test_name, filename, content):
    test_dir = tmp_path / test_name
    test_dir.mkdir()
    (test_dir / filename).write_text(content, encoding="utf-8")


def complete_makereport_hook(plugin, nodeid, report):
    item = type("Item", (), {"nodeid": nodeid})()
    hook = plugin.pytest_runtest_makereport(item, call=None)
    next(hook)
    with pytest.raises(StopIteration) as stop:
        hook.send(report)
    assert stop.value.value is report


def test_capture_stays_open_after_successful_setup_report(tmp_path):
    nodeid = "test_module.py::test_case"
    plugin = make_plugin(tmp_path, inline_output_max_bytes=256)
    capture = CaptureContext()
    plugin._active_capture[nodeid] = {"stdout_redir": capture}

    complete_makereport_hook(plugin, nodeid, Report(when="setup", outcome="passed"))

    assert not capture.closed
    assert nodeid in plugin._active_capture
    plugin._close_capture(plugin._active_capture.pop(nodeid))


@pytest.mark.parametrize(
    ("when", "outcome"),
    [
        ("call", "passed"),
        ("call", "failed"),
        ("setup", "failed"),
        ("setup", "skipped"),
    ],
)
def test_capture_closes_before_terminal_report(tmp_path, when, outcome):
    nodeid = "test_module.py::test_case"
    plugin = make_plugin(tmp_path, inline_output_max_bytes=256)
    capture = CaptureContext()
    plugin._active_capture[nodeid] = {"stdout_redir": capture}

    complete_makereport_hook(plugin, nodeid, Report(when=when, outcome=outcome))

    assert capture.closed
    assert nodeid not in plugin._active_capture


def test_teardown_fallback_warns_when_capture_is_still_active(tmp_path, caplog):
    nodeid = "test_module.py::test_case"
    plugin = make_plugin(tmp_path, inline_output_max_bytes=256)
    capture = CaptureContext()
    plugin._active_capture[nodeid] = {"stdout_redir": capture}
    item = type("Item", (), {"nodeid": nodeid})()

    caplog.set_level(logging.WARNING, logger="test_common.s3_output")
    hook = plugin.pytest_runtest_teardown(item, nextitem=None)
    next(hook)

    assert capture.closed
    assert nodeid not in plugin._active_capture
    assert "remained active until teardown" in caplog.text

    with pytest.raises(StopIteration):
        next(hook)


def test_session_capture_closes_after_teardown(tmp_path):
    nodeid = "test_module.py::test_case"
    plugin = make_plugin(tmp_path, inline_output_max_bytes=256, capture_mode="session")
    capture = CaptureContext()
    plugin._active_capture[nodeid] = {"stdout_redir": capture}
    item = type("Item", (), {"nodeid": nodeid})()

    hook = plugin.pytest_runtest_teardown(item, nextitem=None)
    next(hook)
    assert not capture.closed

    with pytest.raises(StopIteration):
        next(hook)
    assert capture.closed
    assert nodeid not in plugin._active_capture


def test_small_stdout_is_inlined_without_upload(tmp_path):
    test_name = "test-small"
    write_log(tmp_path, test_name, "stdout.log", "ok\n")
    plugin = make_plugin(tmp_path, inline_output_max_bytes=4)
    report = Report()

    plugin.upload_and_report(report, test_name, "stdout.log", "Captured stdout")

    assert report.sections == [("Captured stdout", "ok\n")]
    assert plugin._deferred_uploads == []
    assert not (tmp_path / test_name).exists()


def test_empty_stdout_is_removed_after_report(tmp_path):
    test_name = "test-empty"
    write_log(tmp_path, test_name, "stdout.log", "")
    plugin = make_plugin(tmp_path, inline_output_max_bytes=4)
    report = Report()

    plugin.upload_and_report(report, test_name, "stdout.log", "Captured stdout")

    assert report.sections == [("Captured stdout", "<empty>")]
    assert plugin._deferred_uploads == []
    assert not (tmp_path / test_name).exists()


def test_large_stdout_keeps_existing_upload_report(tmp_path):
    test_name = "test-large"
    write_log(tmp_path, test_name, "stdout.log", "large output")
    plugin = make_plugin(tmp_path, inline_output_max_bytes=3)
    report = Report()

    plugin.upload_and_report(report, test_name, "stdout.log", "Captured stdout")

    assert len(report.sections) == 1
    section_name, section_content = report.sections[0]
    assert section_name == "Captured stdout"
    assert "upload skipped" in section_content
    assert "large output" not in section_content
    assert (tmp_path / test_name / "stdout.log").exists()


def test_uploaded_stdout_is_removed_after_sync_upload(tmp_path):
    test_name = "test-uploaded"
    write_log(tmp_path, test_name, "stdout.log", "large output")
    plugin = make_plugin(tmp_path, inline_output_max_bytes=3)
    plugin.skip_upload = False
    plugin.s3 = RecordingS3Client()
    report = Report()

    plugin.upload_and_report(report, test_name, "stdout.log", "Captured stdout")

    assert len(plugin.s3.uploads) == 1
    assert plugin.s3.uploads[0][1:] == (
        "bucket",
        "logs/test-uploaded/stdout.log",
        {"ContentType": "text/plain"},
    )
    assert len(report.sections) == 1
    assert "uploaded to" in report.sections[0][1]
    assert not (tmp_path / test_name).exists()


def test_deferred_stdout_is_removed_after_upload_finishes(tmp_path):
    test_name = "test-deferred"
    write_log(tmp_path, test_name, "stdout.log", "large output")
    plugin = make_plugin(tmp_path, inline_output_max_bytes=3)
    plugin.skip_upload = False
    plugin.upload_mode = "deferred"
    plugin.s3 = RecordingS3Client()
    report = Report()

    plugin.upload_and_report(report, test_name, "stdout.log", "Captured stdout")

    assert (tmp_path / test_name / "stdout.log").exists()
    assert len(plugin._deferred_uploads) == 1

    plugin.pytest_sessionfinish(session=None, exitstatus=0)

    assert len(plugin.s3.uploads) == 1
    assert plugin._deferred_uploads == []
    assert not (tmp_path / test_name).exists()


def test_file_slice_upload_reads_only_test_range(tmp_path):
    test_name = "test-slice"
    spool = tmp_path / "stdout-spool.log"
    prefix = b"previous test\n"
    content = b"parent\nchild\nparent again\n"
    spool.write_bytes(prefix + content + b"next test\n")
    plugin = make_plugin(tmp_path, inline_output_max_bytes=3, capture_mode="session")
    plugin.skip_upload = False
    plugin.s3 = RecordingS3Client()
    plugin._captured_slices[(test_name, "stdout.log")] = FileSlice(
        str(spool), len(prefix), len(content)
    )
    report = Report()

    plugin.upload_and_report(report, test_name, "stdout.log", "Captured stdout")

    assert plugin.s3.fileobj_uploads == [
        (
            content,
            "bucket",
            "logs/test-slice/stdout.log",
            {"ContentType": "text/plain"},
        )
    ]
    assert "uploaded to" in report.sections[0][1]


def test_file_slice_reader_supports_upload_size_probe(tmp_path):
    spool = tmp_path / "stdout-spool.log"
    spool.write_bytes(b"prefix\ntest output\nsuffix\n")
    file_slice = FileSlice(str(spool), len(b"prefix\n"), len(b"test output\n"))

    with io.BufferedReader(FileSliceReader(file_slice)) as source_file:
        assert source_file.tell() == 0
        assert source_file.seek(0, os.SEEK_END) == file_slice.size
        assert source_file.tell() == file_slice.size
        assert source_file.seek(0) == 0
        assert source_file.read() == b"test output\n"


def test_session_capture_preserves_parent_and_existing_child_stdout_order(tmp_path):
    capture = SessionCapture(str(tmp_path))
    capture.start()
    child = None
    try:
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os, sys; sys.stdin.buffer.read(1); "
                "os.write(1, b'child stdout\\n'); os.write(2, b'child stderr\\n')",
            ],
            stdin=subprocess.PIPE,
        )
        offsets = capture.snapshot()
        os.write(1, b"parent before\n")
        child.stdin.write(b"x")
        child.stdin.close()
        assert child.wait(timeout=10) == 0
        os.write(1, b"parent after\n")
        slices = capture.slices_since(offsets)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        capture.stop()

    with io.BufferedReader(FileSliceReader(slices["stdout.log"])) as stdout_file:
        assert stdout_file.read() == b"parent before\nchild stdout\nparent after\n"
    with io.BufferedReader(FileSliceReader(slices["stderr.log"])) as stderr_file:
        assert stderr_file.read() == b"child stderr\n"
    capture.remove_files()


def test_small_log_file_is_not_inlined(tmp_path):
    test_name = "test-log"
    write_log(tmp_path, test_name, "logging.log", "ok\n")
    plugin = make_plugin(tmp_path, inline_output_max_bytes=256)
    report = Report()

    plugin.upload_and_report(report, test_name, "logging.log", "Captured log")

    assert len(report.sections) == 1
    section_name, section_content = report.sections[0]
    assert section_name == "Captured log"
    assert "upload skipped" in section_content


def test_fd_redirector_reader_thread_is_daemon_when_pipe_writer_lingers(tmp_path):
    redir = FDRedirector(1, str(tmp_path / "stdout.log"))
    proc = None
    try:
        redir.__enter__()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        redir.__exit__(None, None, None)

        assert redir._reader_thread is not None
        assert redir._reader_thread.is_alive()
        assert redir._reader_thread.daemon
    finally:
        if proc is not None:
            proc.kill()
            proc.wait()
        if redir._reader_thread is not None:
            redir._reader_thread.join(timeout=2.0)
