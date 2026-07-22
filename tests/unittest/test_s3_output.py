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
import os
import subprocess
import sys
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
    ):
        self.sections = sections
        self.nodeid = nodeid
        self.when = when
        self.outcome = outcome


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

    assert report.sections == [("Captured stdout call", "ok\n")]
    assert not s3_output._spool_root(str(tmp_path)).exists()


def test_stdout_at_threshold_is_replaced_with_url(tmp_path):
    plugin = make_plugin(tmp_path, inline_output_max_bytes=4)
    report = Report([("Captured stdout call", "four")])

    process_report(plugin, report)

    section_name, section_content = report.sections[0]
    assert section_name == "Captured stdout call"
    assert "4 bytes (upload skipped" in section_content
    assert "/stdout-call.log" in section_content
    assert "four" not in section_content


def test_logging_section_is_uploaded_even_when_small(tmp_path):
    plugin = make_plugin(tmp_path, inline_output_max_bytes=256)
    report = Report([("Captured log call", "log\n")])

    process_report(plugin, report)

    assert "upload skipped" in report.sections[0][1]
    assert "/logging-call.log" in report.sections[0][1]


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
        ]
    )

    process_report(plugin, report)
    plugin.pytest_sessionfinish(None, 0)

    assert [upload[0] for upload in client.uploads] == [
        b"setup output\n",
        b"call error\n",
    ]
    assert client.uploads[0][2].endswith("/stdout-setup.log")
    assert client.uploads[1][2].endswith("/stderr-call.log")
    assert report.sections[1] == ("custom", "keep me")
    assert all("uploaded to" in report.sections[index][1] for index in (0, 2))
    assert not s3_output._spool_root(str(tmp_path)).exists()


def test_duplicate_capture_sections_get_distinct_objects(tmp_path):
    plugin = make_plugin(tmp_path, inline_output_max_bytes=0)
    report = Report(
        [
            ("Captured stdout call", "first"),
            ("Captured stdout call", "second"),
        ]
    )

    process_report(plugin, report)

    assert "/stdout-call.log" in report.sections[0][1]
    assert "/stdout-call-1.log" in report.sections[1][1]


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

    assert client.started.wait(timeout=5)
    assert "scheduled for upload" in report.sections[0][1]
    client.release.set()
    plugin.pytest_sessionfinish(None, 0)
    assert client.uploads[0][0] == b"background output"
    assert not s3_output._spool_root(str(tmp_path)).exists()


def test_parent_drain_retries_upload_left_by_failed_process(tmp_path, monkeypatch):
    plugin = make_uploading_plugin(
        tmp_path,
        monkeypatch,
        FailingS3Client(),
        inline_output_max_bytes=0,
    )
    report = Report([("Captured stdout call", "recover me")])
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
    assert client.uploads[0][2].endswith("/stdout-call.log")
    assert not s3_output._spool_root(str(tmp_path)).exists()


def test_parent_drain_uses_configured_upload_workers(tmp_path, monkeypatch):
    plugin = make_uploading_plugin(
        tmp_path,
        monkeypatch,
        RecordingS3Client(),
        upload_workers=2,
    )
    plugin._write_spool_file("test", "stdout-call.log", "stdout")
    plugin._write_spool_file("test", "stderr-call.log", "stderr")
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
    plugin._write_spool_file("test", "stdout-call.log", "still active")

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


def _write_nested_pytest_plugin(tmp_path, register_plugin=True):
    tests_root = Path(s3_output.__file__).parents[1]
    conftest = (
        "from test_common import s3_output\n"
        "def pytest_addoption(parser):\n"
        "    parser.addoption('--output-dir')\n"
        "    s3_output.add_options(parser)\n"
    )
    if register_plugin:
        conftest += "def pytest_configure(config):\n    s3_output.register_plugin(config)\n"
    (tmp_path / "conftest.py").write_text(conftest, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tests_root), env.get("PYTHONPATH", "")])
    return env


def test_native_capture_is_replaced_before_junit_consumes_report(tmp_path):
    env = _write_nested_pytest_plugin(tmp_path)
    (tmp_path / "test_native.py").write_text(
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "def test_output():\n"
        "    os.write(1, b'parent before\\n')\n"
        "    subprocess.run([sys.executable, '-c', "
        "\"import os; os.write(1, b'child capture ' + b'x' * 300 + b'\\\\n')\"], check=True)\n"
        "    os.write(1, b'parent after\\n')\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    xml_path = tmp_path / "results.xml"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--capture=fd",
            "--s3-upload-path=logs",
            f"--output-dir={output_dir}",
            "--s3-skip-upload",
            f"--junitxml={xml_path}",
            "-o",
            "junit_logging=all",
            str(tmp_path / "test_native.py"),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    xml = xml_path.read_text(encoding="utf-8")
    assert "upload skipped" in xml
    assert "child capture xxxxx" not in xml
    assert "stdout-call.log" in xml
    stdout_files = list(s3_output._spool_root(str(output_dir)).rglob("stdout-call.log"))
    assert len(stdout_files) == 1
    assert stdout_files[0].read_bytes() == (
        b"parent before\n" + b"child capture " + b"x" * 300 + b"\nparent after\n"
    )


def test_xdist_worker_transforms_report_before_controller_junit(tmp_path):
    env = _write_nested_pytest_plugin(tmp_path, register_plugin=False)
    (tmp_path / "test_xdist.py").write_text(
        "import os\ndef test_output():\n    os.write(1, b'xdist capture ' + b'x' * 300)\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    xml_path = tmp_path / "results.xml"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "2",
            "-p",
            "test_common.s3_output_hooks",
            "--capture=fd",
            "--s3-upload-path=logs",
            f"--output-dir={output_dir}",
            "--s3-skip-upload",
            f"--junitxml={xml_path}",
            "-o",
            "junit_logging=all",
            str(tmp_path / "test_xdist.py"),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    xml = xml_path.read_text(encoding="utf-8")
    assert "upload skipped" in xml
    assert "xdist capture xxxxx" not in xml
    assert "stdout-call.log" in xml


def test_native_capture_restores_timeout_output_to_console(tmp_path):
    env = _write_nested_pytest_plugin(tmp_path)
    (tmp_path / "test_timeout.py").write_text(
        "import os\n"
        "import time\n"
        "import pytest\n"
        "@pytest.mark.timeout(0.2, method='thread')\n"
        "def test_timeout():\n"
        "    os.write(1, b'timeout stdout\\n')\n"
        "    os.write(2, b'timeout stderr\\n')\n"
        "    time.sleep(30)\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--capture=fd",
            "--s3-upload-path=logs",
            f"--output-dir={tmp_path / 'output'}",
            "--s3-skip-upload",
            str(tmp_path / "test_timeout.py"),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    console = result.stdout + result.stderr
    assert result.returncode == 1
    assert "timeout stdout" in console
    assert "timeout stderr" in console
    assert "Timeout" in console
