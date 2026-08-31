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

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_common import s3_output, s3_output_hooks
from test_common.s3_output import UploadLogPlugin


class Report:
    def __init__(
        self,
        sections,
        nodeid="test_module.py::test_case",
        when="call",
        outcome="passed",
        rerun=None,
    ):
        self.sections = sections
        self.nodeid = nodeid
        self.when = when
        self.outcome = outcome
        if rerun is not None:
            self.rerun = rerun


class RecordingS3Client:
    def __init__(self):
        self.uploads = []

    def upload_file(self, filepath, bucket, object_key, ExtraArgs=None):
        content = Path(filepath).read_bytes()
        self.uploads.append((content, bucket, object_key, ExtraArgs))


class FailingS3Client:
    def upload_file(self, filepath, bucket, object_key, ExtraArgs=None):
        raise RuntimeError("upload error")


class BlockingS3Client(RecordingS3Client):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def upload_file(self, filepath, bucket, object_key, ExtraArgs=None):
        self.started.set()
        assert self.release.wait(timeout=10)
        super().upload_file(filepath, bucket, object_key, ExtraArgs)


class ConcurrentS3Client(RecordingS3Client):
    def __init__(self, workers):
        super().__init__()
        self.barrier = threading.Barrier(workers)

    def upload_file(self, filepath, bucket, object_key, ExtraArgs=None):
        self.barrier.wait(timeout=5)
        super().upload_file(filepath, bucket, object_key, ExtraArgs)


class PluginManager:
    def __init__(self):
        self.plugins = {}

    def getplugin(self, name):
        return self.plugins.get(name)

    def register(self, plugin, name):
        self.plugins[name] = plugin


class Config:
    def __init__(self, **options):
        self.options = options
        self.option = SimpleNamespace(numprocesses=options.get("numprocesses"))
        self.known_args_namespace = self.option
        self.pluginmanager = PluginManager()

    def getoption(self, name, default=None):
        return self.options.get(name, default)


def make_plugin(
    tmp_path,
    inline_output_max_bytes=256,
    skip_upload=True,
    upload_mode="sync",
    upload_workers=8,
):
    return UploadLogPlugin(
        endpoint_url="https://example.com",
        aws_access_key_id="user",
        aws_secret_access_key=None if skip_upload else "secret",
        bucket="bucket",
        upload_path="logs",
        output_path=str(tmp_path),
        skip_upload=skip_upload,
        upload_mode=upload_mode,
        upload_workers=upload_workers,
        inline_output_max_bytes=inline_output_max_bytes,
    )


def make_uploading_plugin(tmp_path, monkeypatch, client, **kwargs):
    monkeypatch.setattr(s3_output, "_create_s3_client", lambda *args: client)
    return make_plugin(tmp_path, skip_upload=False, **kwargs)


def process_report(plugin, report):
    hook = plugin.pytest_runtest_logreport(report)
    next(hook)
    with pytest.raises(StopIteration):
        next(hook)


def test_small_stdout_remains_inline(tmp_path):
    plugin = make_plugin(tmp_path, inline_output_max_bytes=4)
    report = Report([("Captured stdout call", "ok\n")])

    process_report(plugin, report)
    plugin.pytest_runtest_logfinish(report.nodeid, None)

    assert report.sections == [("Captured stdout call", "ok\n")]
    assert not s3_output._spool_root(str(tmp_path)).exists()


def test_stdout_at_threshold_is_replaced_with_url(tmp_path):
    plugin = make_plugin(tmp_path, inline_output_max_bytes=4)
    report = Report([("Captured stdout call", "four")])

    process_report(plugin, report)

    section_name, section_content = report.sections[0]
    assert section_name == "Captured stdout"
    assert "4 bytes (upload skipped" in section_content
    assert "/stdout.log" in section_content
    assert "four" not in section_content


def test_inline_threshold_applies_to_combined_stream(tmp_path):
    plugin = make_plugin(tmp_path, inline_output_max_bytes=8)
    setup_section = ("Captured stdout setup", "abc")
    call_section = ("Captured stdout call", "defgh")
    setup_report = Report([setup_section], when="setup")
    call_report = Report([setup_section, call_section])
    teardown_report = Report([setup_section, call_section], when="teardown")

    process_report(plugin, setup_report)
    process_report(plugin, call_report)
    process_report(plugin, teardown_report)

    assert setup_report.sections == [setup_section]
    assert len(call_report.sections) == 1
    assert "/stdout.log" in call_report.sections[0][1]
    assert len(teardown_report.sections) == 1
    stdout_file = next(s3_output._spool_root(str(tmp_path)).rglob("stdout.log"))
    assert stdout_file.read_text(encoding="utf-8") == "abcdefgh"


def test_logging_section_is_uploaded_even_when_small(tmp_path):
    plugin = make_plugin(tmp_path, inline_output_max_bytes=256)
    report = Report([("Captured log call", "log\n")])

    process_report(plugin, report)

    assert "upload skipped" in report.sections[0][1]
    assert "/logging.log" in report.sections[0][1]


def test_sync_upload_transforms_native_sections(tmp_path, monkeypatch):
    client = RecordingS3Client()
    plugin = make_uploading_plugin(
        tmp_path,
        monkeypatch,
        client,
        inline_output_max_bytes=0,
    )
    report = Report(
        [
            ("Captured stdout setup", "setup output\n"),
            ("custom", "keep me"),
            ("Captured stderr call", "call error\n"),
        ],
        when="teardown",
    )

    process_report(plugin, report)
    plugin.pytest_sessionfinish(None, 0)

    assert [upload[0] for upload in client.uploads] == [
        b"setup output\n",
        b"call error\n",
    ]
    assert client.uploads[0][2].endswith("/stdout.log")
    assert client.uploads[1][2].endswith("/stderr.log")
    assert report.sections[1] == ("custom", "keep me")
    assert all("uploaded to" in report.sections[index][1] for index in (0, 2))
    assert not s3_output._spool_root(str(tmp_path)).exists()


def test_duplicate_capture_sections_share_one_object(tmp_path):
    plugin = make_plugin(tmp_path, inline_output_max_bytes=0)
    report = Report(
        [
            ("Captured stdout call", "first"),
            ("Captured stdout call", "second"),
        ]
    )

    process_report(plugin, report)

    assert len(report.sections) == 1
    assert "/stdout.log" in report.sections[0][1]
    stdout_file = next(s3_output._spool_root(str(tmp_path)).rglob("stdout.log"))
    assert stdout_file.read_text(encoding="utf-8") == "firstsecond"


def test_cumulative_capture_sections_are_uploaded_once(tmp_path, monkeypatch):
    client = RecordingS3Client()
    plugin = make_uploading_plugin(
        tmp_path,
        monkeypatch,
        client,
        inline_output_max_bytes=0,
    )
    setup_section = ("Captured stdout setup", "setup output\n")
    call_section = ("Captured stdout call", "call output\n")
    teardown_section = ("Captured stderr teardown", "teardown error\n")

    setup_report = Report([setup_section], when="setup")
    call_report = Report(
        [setup_section, call_section],
        when="call",
        outcome="failed",
    )
    teardown_report = Report(
        [setup_section, call_section, teardown_section],
        when="teardown",
    )

    process_report(plugin, setup_report)
    process_report(plugin, call_report)
    assert client.uploads == []
    process_report(plugin, teardown_report)
    plugin.pytest_runtest_logfinish(setup_report.nodeid, None)
    plugin.pytest_sessionfinish(None, 0)

    assert [upload[0] for upload in client.uploads] == [
        b"setup output\ncall output\n",
        b"teardown error\n",
    ]
    assert [upload[2].rsplit("/", 1)[-1] for upload in client.uploads] == [
        "stdout.log",
        "stderr.log",
    ]
    assert "Last 200 lines:" in call_report.sections[0][1]
    assert "setup output" in call_report.sections[0][1]
    assert "call output" in call_report.sections[0][1]
    assert "Last 200 lines:" not in teardown_report.sections[0][1]
    assert sum("/stdout.log" in content for _, content in teardown_report.sections) == 1
    assert sum("/stderr.log" in content for _, content in teardown_report.sections) == 1
    assert len(teardown_report.sections) == 2
    assert all(content.endswith("\n") for _, content in teardown_report.sections)


def test_same_nodeid_rerun_gets_distinct_test_path(tmp_path, monkeypatch):
    monkeypatch.setattr(s3_output.time, "time", lambda: 1234)
    plugin = make_plugin(tmp_path, inline_output_max_bytes=0)
    nodeid = "test_module.py::test_case"

    plugin.pytest_runtest_logstart(nodeid, None)
    first_name = plugin._test_names[nodeid]
    plugin.pytest_runtest_logfinish(nodeid, None)
    plugin.pytest_runtest_logstart(nodeid, None)
    second_name = plugin._test_names[nodeid]

    assert second_name == f"{first_name}-1"


def test_failed_report_keeps_only_recent_bounded_output(tmp_path):
    plugin = make_plugin(tmp_path, inline_output_max_bytes=0)
    content = "".join(f"line-{index:03d}\n" for index in range(250))
    report = Report(
        [("Captured stdout call", content)],
        outcome="failed",
    )

    process_report(plugin, report)

    section_content = report.sections[0][1]
    assert "Last 200 lines:" in section_content
    assert "line-000" not in section_content
    assert "line-249" in section_content

    large_line = "x" * 70000
    report = Report(
        [("Captured stderr call", large_line)],
        nodeid="test_module.py::test_other",
        outcome="failed",
    )
    process_report(plugin, report)
    assert "... [truncated]" in report.sections[0][1]
    assert len(report.sections[0][1].encode()) < 66000


def test_deferred_upload_starts_before_session_finish(tmp_path, monkeypatch):
    client = BlockingS3Client()
    plugin = make_uploading_plugin(
        tmp_path,
        monkeypatch,
        client,
        inline_output_max_bytes=0,
        upload_mode="deferred",
        upload_workers=1,
    )
    report = Report([("Captured stdout call", "background output")])

    process_report(plugin, report)
    plugin.pytest_runtest_logfinish(report.nodeid, None)

    assert client.started.wait(timeout=5)
    assert "scheduled for upload" in report.sections[0][1]
    client.release.set()
    plugin.pytest_sessionfinish(None, 0)
    assert client.uploads[0][0] == b"background output"
    assert not s3_output._spool_root(str(tmp_path)).exists()


def test_deferred_upload_reuses_cumulative_section(tmp_path, monkeypatch):
    client = BlockingS3Client()
    plugin = make_uploading_plugin(
        tmp_path,
        monkeypatch,
        client,
        inline_output_max_bytes=0,
        upload_mode="deferred",
        upload_workers=1,
    )
    section = ("Captured stdout call", "background output")
    call_report = Report([section], outcome="failed")
    teardown_report = Report([section], when="teardown")

    process_report(plugin, call_report)
    process_report(plugin, teardown_report)

    assert client.started.wait(timeout=5)
    assert len(plugin._pending_uploads) == 1
    assert (
        call_report.sections[0][1].splitlines()[0]
        == (teardown_report.sections[0][1].splitlines()[0])
    )
    client.release.set()
    plugin.pytest_sessionfinish(None, 0)
    assert [upload[0] for upload in client.uploads] == [b"background output"]


def test_rerun_keeps_one_url_per_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(s3_output.time, "time", lambda: 1234)
    client = RecordingS3Client()
    plugin = make_uploading_plugin(
        tmp_path,
        monkeypatch,
        client,
        inline_output_max_bytes=0,
    )
    nodeid = "test_module.py::test_case"
    first_sections = [
        ("Captured stdout call", "first stdout\n"),
        ("Captured stderr call", "first stderr\n"),
    ]
    all_sections = first_sections + [
        ("Captured stdout call", "second stdout\n"),
        ("Captured stderr call", "second stderr\n"),
    ]

    plugin.pytest_runtest_logstart(nodeid, None)
    first_report = Report(
        first_sections,
        nodeid=nodeid,
        outcome="rerun",
        rerun=0,
    )
    process_report(plugin, first_report)
    plugin.pytest_runtest_logfinish(nodeid, None)

    plugin.pytest_runtest_logstart(nodeid, None)
    second_report = Report(
        list(all_sections),
        nodeid=nodeid,
        outcome="failed",
        rerun=1,
    )
    process_report(plugin, second_report)
    teardown_report = Report(
        list(all_sections),
        nodeid=nodeid,
        when="teardown",
        rerun=1,
    )
    process_report(plugin, teardown_report)
    plugin.pytest_runtest_logfinish(nodeid, None)
    plugin.pytest_sessionfinish(None, 0)

    assert [upload[0] for upload in client.uploads] == [
        b"first stdout\n",
        b"first stderr\n",
        b"second stdout\n",
        b"second stderr\n",
    ]
    object_keys = [upload[2] for upload in client.uploads]
    assert object_keys[0].endswith("/stdout.log")
    assert object_keys[1].endswith("/stderr.log")
    assert object_keys[2].endswith("/stdout-attempt-2.log")
    assert object_keys[3].endswith("/stderr-attempt-2.log")
    assert object_keys[0].rsplit("/", 1)[0] != object_keys[2].rsplit("/", 1)[0]
    assert all(content.endswith("\n") for _, content in teardown_report.sections)
    assert "Last 200 lines:" not in teardown_report.sections[0][1]


def test_parent_drain_retries_upload_left_by_failed_process(tmp_path, monkeypatch):
    plugin = make_uploading_plugin(
        tmp_path,
        monkeypatch,
        FailingS3Client(),
        inline_output_max_bytes=0,
    )
    report = Report([("Captured stdout call", "recover me")], when="teardown")
    process_report(plugin, report)
    assert "upload failed" in report.sections[0][1]

    config_path = Path(plugin._spool_config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pid"] = 99999999
    config_path.write_text(json.dumps(config), encoding="utf-8")

    client = RecordingS3Client()
    monkeypatch.setattr(s3_output, "_create_s3_client", lambda *args: client)
    assert s3_output.drain_pending_uploads(str(tmp_path), secret_key="secret")
    assert client.uploads[0][0] == b"recover me"
    assert client.uploads[0][2].endswith("/stdout.log")
    assert not s3_output._spool_root(str(tmp_path)).exists()


def test_parent_drain_uses_configured_upload_workers(tmp_path, monkeypatch):
    plugin = make_uploading_plugin(
        tmp_path,
        monkeypatch,
        RecordingS3Client(),
        upload_workers=2,
    )
    plugin._append_spool_file("test", "stdout-call.log", "stdout")
    plugin._append_spool_file("test", "stderr-call.log", "stderr")
    config_path = Path(plugin._spool_config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pid"] = 99999999
    config_path.write_text(json.dumps(config), encoding="utf-8")

    client = ConcurrentS3Client(workers=2)
    monkeypatch.setattr(s3_output, "_create_s3_client", lambda *args: client)

    assert s3_output.drain_pending_uploads(str(tmp_path), secret_key="secret")
    assert sorted(upload[0] for upload in client.uploads) == [b"stderr", b"stdout"]
    assert not s3_output._spool_root(str(tmp_path)).exists()


def test_parent_drain_skips_live_pytest_process(tmp_path, monkeypatch):
    client = RecordingS3Client()
    plugin = make_uploading_plugin(
        tmp_path,
        monkeypatch,
        client,
        inline_output_max_bytes=0,
        upload_mode="deferred",
    )
    plugin._append_spool_file("test", "stdout-call.log", "still active")

    assert s3_output.drain_pending_uploads(str(tmp_path), secret_key="secret")
    assert client.uploads == []
    assert Path(plugin._spool_config_path).exists()
    plugin.pytest_sessionfinish(None, 0)


def test_register_plugin_requires_native_fd_capture(tmp_path):
    config = Config(
        **{
            "--s3-upload-path": "logs",
            "--output-dir": str(tmp_path),
            "--s3-skip-upload": True,
            "capture": "no",
        }
    )

    with pytest.raises(ValueError, match="requires pytest --capture=fd"):
        s3_output.register_plugin(config)


def test_register_plugin_uses_report_transformer(tmp_path):
    config = Config(
        **{
            "--s3-upload-path": "logs",
            "--output-dir": str(tmp_path),
            "--s3-skip-upload": True,
            "--s3-endpoint": "https://example.com",
            "--s3-username": "user",
            "--s3-bucket": "bucket",
            "capture": "fd",
        }
    )

    plugin = s3_output.register_plugin(config)

    assert isinstance(plugin, UploadLogPlugin)
    assert config.pluginmanager.getplugin("upload_log_plugin") is plugin


def test_s3_hook_skips_xdist_controller(monkeypatch):
    registered = []
    monkeypatch.setattr(s3_output_hooks.s3_output, "register_plugin", registered.append)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

    controller = Config(numprocesses=2)
    s3_output_hooks.pytest_configure(controller)
    assert registered == []

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    worker = Config(numprocesses=2)
    s3_output_hooks.pytest_configure(worker)
    assert registered == [worker]
