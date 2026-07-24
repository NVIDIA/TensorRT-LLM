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

import subprocess
import sys

from test_common.s3_output import FDRedirector, UploadLogPlugin


class Report:
    def __init__(self):
        self.sections = []


class RecordingS3Client:
    def __init__(self):
        self.uploads = []

    def upload_file(self, filepath, bucket, object_key, ExtraArgs=None):
        self.uploads.append((filepath, bucket, object_key, ExtraArgs))


def make_plugin(tmp_path, inline_output_max_bytes):
    return UploadLogPlugin(
        endpoint_url="https://example.com",
        aws_access_key_id="user",
        aws_secret_access_key=None,
        bucket="bucket",
        upload_path="logs",
        output_path=str(tmp_path),
        skip_upload=True,
        inline_output_max_bytes=inline_output_max_bytes,
    )


def write_log(tmp_path, test_name, filename, content):
    test_dir = tmp_path / test_name
    test_dir.mkdir()
    (test_dir / filename).write_text(content, encoding="utf-8")


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
