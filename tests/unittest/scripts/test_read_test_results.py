#!/usr/bin/env python3
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
"""Tests for ``jenkins/scripts/read_test_results.sh``.

The script reads the test-result archives of one CI run with the read helper,
``jenkins/scripts/fetch_build_artifact.sh``: every final archive, and a stage attempt's checkpoint
only when the attempt's final archive could not be read. An archive that was read and is not a
valid archive counts as not read. The loopback servers of ``test_fetch_build_artifact`` stand in
for artifactory.nvidia.com, and the script runs the real helper through a recorder that notes each
call's arguments.
"""

from __future__ import annotations

import gzip
import io
import subprocess
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from .test_fetch_build_artifact import (
    AUTHORIZATION,
    BASIC_TOKEN,
    CREDENTIAL_ID,
    FETCH_BUILD_ARTIFACT,
    NETRC_PASSWORD,
    REPO_ROOT,
    RESULTS_DIR,
    FakeArtifactory,
    Helper,
    Reply,
    _in_turn,
)

pytestmark = pytest.mark.cpu_only

READ_TEST_RESULTS = REPO_ROOT / "jenkins" / "scripts" / "read_test_results.sh"
SYNOPSIS = (
    "read_test_results.sh <helper> <test-results dir> <dest dir> [<name ERE> [<exclude ERE>]]"
)
LEGACY_RESULTS_DIR = f"sw-tensorrt-generic/{RESULTS_DIR}"

# One stage attempt's final archive and its checkpoint.
FINAL = "results-A10-PyTorch-1.tar.gz"
CHECKPOINT = "results-A10-PyTorch-1-progress.tar.gz"
OTHER_FINAL = "results-H100-PyTorch-2.tar.gz"
OTHER_CHECKPOINT = "results-H100-PyTorch-2-progress.tar.gz"
B200_FINAL = "results-B200-PyTorch-3.tar.gz"
OTHER_B200_FINAL = "results-B200-PyTorch-4.tar.gz"
SBSA_FINAL = "results-GH200-PyTorch-1.tar.gz"
# What a run's test-results/ folder holds besides the stages' archives.
OTHER_FILES = ["doc-html-preview.tar.gz", "rerun_report.html"]

RESULT_ARCHIVES = r"^results-.*\.tar\.gz$"
SBSA_ARCHIVES = r"^results-(GH200|GB10|GB200|GB300|CPU-Generic-arm)-.*\.tar\.gz$"

STOP_MESSAGE = "[test-results] 3 reads in a row failed; not reading the remaining names"


@dataclass
class Outcome:
    returncode: int
    stdout: str
    stderr: str
    # The arguments of each call of the helper, in order.
    helper_calls: list[list[str]]
    # The files in <dest dir>, by name.
    files: dict[str, bytes]

    @property
    def log(self) -> list[str]:
        return self.stderr.splitlines()

    @property
    def summary(self) -> str:
        """The last line on stderr, where the script prints its summary."""
        return self.log[-1]


class Reader:
    """Runs the script against the fake servers, with the environment that ``Helper`` sets up."""

    def __init__(self, tmp_path: Path, helper: Helper) -> None:
        self.artifactory = helper.artifactory
        self.dest = tmp_path / "results"
        self.dest.mkdir()
        self._env = helper.env
        self._cwd = helper.cwd
        self._tmpdir = helper.tmpdir
        self._calls = tmp_path / "helper-calls.log"
        # Notes the arguments of a call, separated by tabs, and then runs the real helper. The
        # script runs its helper with bash, so the recorder needs no execute permission.
        self.helper = tmp_path / "recording_helper.sh"
        self.helper.write_text(
            f"(IFS=$'\\t'; printf '%s\\n' \"$*\") >> '{self._calls}'\n"
            f"exec bash '{FETCH_BUILD_ARTIFACT}' \"$@\"\n"
        )

    def run(
        self,
        *patterns: str,
        args: list[str] | None = None,
        cwd: Path | None = None,
        **env_overrides: str | None,
    ) -> Outcome:
        """Run the script on the run's test-results folder, reading into ``self.dest``.

        Args:
            *patterns: The ``<name ERE>`` and the ``<exclude ERE>``, when given.
            args: Replaces all of the script's arguments.
            cwd: Replaces the working directory, which is otherwise an empty directory.
            **env_overrides: Environment variables to set, or to remove when ``None``.
        """
        env = dict(self._env)
        for name, value in env_overrides.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        if args is None:
            args = [str(self.helper), LEGACY_RESULTS_DIR, str(self.dest), *patterns]
        self._calls.unlink(missing_ok=True)

        proc = subprocess.run(
            ["bash", str(READ_TEST_RESULTS), *args],
            env=env,
            cwd=cwd or self._cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        output = proc.stdout + proc.stderr
        assert NETRC_PASSWORD not in output
        assert BASIC_TOKEN not in output
        assert list(self._tmpdir.iterdir()) == [], "temporary files were left behind"
        assert list(self._cwd.iterdir()) == [], "files were written outside <dest dir>"
        calls = self._calls.read_text().splitlines() if self._calls.exists() else []
        files = {path.name: path.read_bytes() for path in self.dest.iterdir()}
        return Outcome(
            proc.returncode, proc.stdout, proc.stderr, [call.split("\t") for call in calls], files
        )

    def list_call(self) -> list[str]:
        """Return the helper's arguments for the listing."""
        return ["--list", LEGACY_RESULTS_DIR]

    def read_call(self, name: str) -> list[str]:
        """Return the helper's arguments for the read of ``name``."""
        return [f"{LEGACY_RESULTS_DIR}/{name}", str(self.dest / name)]


@pytest.fixture
def artifactory() -> Iterator[FakeArtifactory]:
    fake = FakeArtifactory()
    yield fake
    fake.close()


@pytest.fixture
def reader(tmp_path: Path, artifactory: FakeArtifactory) -> Reader:
    return Reader(tmp_path, Helper(tmp_path, artifactory))


def _archive(text: str) -> bytes:
    """Return a gzip-compressed tar archive whose one member holds ``text``.

    The same text gives the same bytes: neither the member nor the gzip header carries a time.
    """
    data = text.encode()
    member = tarfile.TarInfo("results/results.xml")
    member.size = len(data)
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w") as archive:
        archive.addfile(member, io.BytesIO(data))
    return gzip.compress(tar.getvalue(), mtime=0)


def _content(name: str) -> bytes:
    """Return what the folder holds under ``name``: an archive when the name ends in .tar.gz."""
    text = f"the content of {name}"
    return _archive(text) if name.endswith(".tar.gz") else text.encode()


# Files that the helper reads without complaint, since each download matches its stored file,
# and that are not archives of test results.
NOT_ARCHIVES = {
    # The archive without its last bytes, as an upload that was cut short leaves it.
    "cut-short": _archive("results of a stage")[:-12],
    "not-gzip": b"<html><body>502 Bad Gateway</body></html>\n",
    "empty": b"",
    "gzip-of-other-data": gzip.compress(bytes(range(256)) * 4, mtime=0),
    # A tar archive that ends before its first member, which tar lists without an error.
    "no-member": gzip.compress(bytes(1024), mtime=0),
    # One block of zeros, which tar may name in a message of its own and lists nothing from.
    "lone-zero-block": gzip.compress(bytes(512), mtime=0),
}


def _not_valid(name: str) -> str:
    """Return the line that the script logs for a file that is not a valid archive."""
    return f"[test-results] failed: {name} was read and is not a valid archive; removed it"


def _publish(artifactory: FakeArtifactory, *names: str) -> None:
    """Serve each of ``names`` in the run's test-results folder."""
    for name in names:
        artifactory.publish("aws", _content(name), f"{RESULTS_DIR}/{name}")


def _lookup(artifactory: FakeArtifactory, name: str) -> tuple[str, str]:
    """Return the route of the storage lookup that starts every read of ``name``."""
    return ("aws", artifactory.storage_path("aws", f"{RESULTS_DIR}/{name}"))


def _fail(artifactory: FakeArtifactory, name: str, status: int = 500) -> None:
    """Answer the lookup of ``name`` with ``status``."""
    artifactory.routes[_lookup(artifactory, name)] = Reply(status)


def _asked(artifactory: FakeArtifactory, name: str) -> list[str]:
    """Return the server of each request for ``name``, lookups and downloads, in order."""
    suffix = f"/{RESULTS_DIR}/{name}"
    return [request.server for request in artifactory.requests if request.path.endswith(suffix)]


def _summary(
    *,
    listed: int,
    read: int,
    skipped: int = 0,
    replaced: int = 0,
    absent: int = 0,
    failed: int = 0,
) -> str:
    """Return the summary line."""
    return (
        f"[test-results] listed {listed}, read {read}, checkpoints skipped {skipped}, "
        f"replaced {replaced}, absent {absent}, failed {failed}"
    )


# ---------------------------------------------------------------------------
# Listing and reading
# ---------------------------------------------------------------------------


def test_reads_each_listed_archive_once(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL, OTHER_FINAL, *OTHER_FILES, "logs/"])
    _publish(artifactory, FINAL, OTHER_FINAL, *OTHER_FILES)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.stdout == ""
    assert outcome.files == {FINAL: _content(FINAL), OTHER_FINAL: _content(OTHER_FINAL)}
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(OTHER_FINAL),
    ]
    assert outcome.summary == _summary(listed=2, read=2)
    # The listing, and a lookup and a download for each of the two archives.
    assert [request.authorization for request in artifactory.received("aws")] == [AUTHORIZATION] * 5


@pytest.mark.parametrize(
    "results_dir",
    [LEGACY_RESULTS_DIR, RESULTS_DIR, f"{LEGACY_RESULTS_DIR}/"],
    ids=["legacy", "layout", "trailing-slash"],
)
def test_reads_into_the_current_directory(reader: Reader, results_dir: str) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL])
    _publish(artifactory, FINAL)

    outcome = reader.run(
        args=[str(reader.helper), results_dir, ".", RESULT_ARCHIVES], cwd=reader.dest
    )

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {FINAL: _content(FINAL)}
    assert outcome.summary == _summary(listed=1, read=1)


def test_folder_that_does_not_exist_is_complete(reader: Reader) -> None:
    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {}
    assert outcome.helper_calls == [reader.list_call()]
    assert outcome.summary == _summary(listed=0, read=0)


@pytest.mark.parametrize(
    ("patterns", "kept"),
    [
        ((RESULT_ARCHIVES,), [FINAL, SBSA_FINAL]),
        ((SBSA_ARCHIVES,), [SBSA_FINAL]),
        ((RESULT_ARCHIVES, SBSA_ARCHIVES), [FINAL]),
        ((RESULT_ARCHIVES, ""), [FINAL, SBSA_FINAL]),
        (("", SBSA_ARCHIVES), [FINAL, *OTHER_FILES]),
        ((), [FINAL, SBSA_FINAL, *OTHER_FILES]),
    ],
    ids=["name", "sbsa-name", "name-and-exclude", "empty-exclude", "exclude-only", "no-pattern"],
)
def test_patterns_select_the_names_to_read(
    reader: Reader, patterns: tuple[str, ...], kept: list[str]
) -> None:
    artifactory = reader.artifactory
    names = [FINAL, SBSA_FINAL, *OTHER_FILES]
    artifactory.publish_folder("aws", [*names, "logs/"])
    _publish(artifactory, *names)

    outcome = reader.run(*patterns)

    assert outcome.returncode == 0, outcome.stderr
    assert sorted(outcome.files) == sorted(kept)
    for name in names:
        if name not in kept:
            assert _asked(artifactory, name) == []
    assert outcome.summary == _summary(listed=len(kept), read=len(kept))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [[CHECKPOINT, FINAL], [FINAL, CHECKPOINT]],
    ids=["checkpoint-listed-first", "final-listed-first"],
)
def test_skips_a_checkpoint_whose_final_archive_was_read(reader: Reader, names: list[str]) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", names)
    _publish(artifactory, *names)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {FINAL: _content(FINAL)}
    assert _asked(artifactory, CHECKPOINT) == []
    assert outcome.helper_calls == [reader.list_call(), reader.read_call(FINAL)]
    assert (
        f"[test-results] skipped checkpoint {CHECKPOINT}: its final archive {FINAL} was read"
        in outcome.log
    )
    assert outcome.summary == _summary(listed=2, read=1, skipped=1)


def test_reads_a_checkpoint_whose_final_archive_is_absent(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    _publish(artifactory, CHECKPOINT)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {CHECKPOINT: _content(CHECKPOINT)}
    # The final archive is asked for first, because it may have landed after the listing.
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
    ]
    assert _asked(artifactory, FINAL) == ["aws"]
    assert outcome.summary == _summary(listed=1, read=1)


def test_reads_an_unlisted_final_archive_in_place_of_its_checkpoint(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    _publish(artifactory, CHECKPOINT, FINAL)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {FINAL: _content(FINAL)}
    assert _asked(artifactory, CHECKPOINT) == []
    assert (
        f"[test-results] replaced checkpoint {CHECKPOINT}: read its final archive {FINAL} instead"
        in outcome.log
    )
    assert outcome.summary == _summary(listed=1, read=1, replaced=1)


@pytest.mark.parametrize(
    "final_at_the_second_try", [True, False], ids=["final-present", "final-absent"]
)
def test_checkpoint_gone_when_read_sends_the_read_to_its_final_archive(
    reader: Reader, final_at_the_second_try: bool
) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    if final_at_the_second_try:
        _publish(artifactory, FINAL)
        lookup = _lookup(artifactory, FINAL)
        found = artifactory.routes[lookup]
        assert isinstance(found, Reply)
        artifactory.routes[lookup] = _in_turn(Reply(404), found)

    outcome = reader.run(RESULT_ARCHIVES)

    # A stage deletes its checkpoint only after its final archive is uploaded.
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
        reader.read_call(FINAL),
    ]
    if final_at_the_second_try:
        assert outcome.returncode == 0, outcome.stderr
        assert outcome.files == {FINAL: _content(FINAL)}
        assert outcome.summary == _summary(listed=1, read=1, replaced=1)
    else:
        assert outcome.returncode == 1
        assert outcome.files == {}
        assert (
            f"[test-results] failed: checkpoint {CHECKPOINT} is gone, and its final archive "
            f"{FINAL} is absent"
        ) in outcome.log
        assert outcome.summary == _summary(listed=1, read=0, failed=1)


def test_failed_read_of_the_final_archive_of_a_checkpoint_that_is_gone_counts_once(
    reader: Reader,
) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    artifactory.routes[_lookup(artifactory, FINAL)] = _in_turn(Reply(404), Reply(500))

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    assert outcome.files == {}
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
        reader.read_call(FINAL),
    ]
    assert outcome.summary == _summary(listed=1, read=0, failed=1)


def test_checkpoint_deleted_between_its_lookup_and_its_download(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    _publish(artifactory, CHECKPOINT, FINAL)
    final_lookup = _lookup(artifactory, FINAL)
    found = artifactory.routes[final_lookup]
    assert isinstance(found, Reply)
    artifactory.routes[final_lookup] = _in_turn(Reply(404), found)
    # The checkpoint's lookup finds it; the delete lands before the download and the next lookup.
    checkpoint_lookup = _lookup(artifactory, CHECKPOINT)
    listed = artifactory.routes[checkpoint_lookup]
    assert isinstance(listed, Reply)
    artifactory.routes[checkpoint_lookup] = _in_turn(listed, Reply(404))
    del artifactory.routes[("aws", artifactory.file_path("aws", f"{RESULTS_DIR}/{CHECKPOINT}"))]

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {FINAL: _content(FINAL)}
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
        reader.read_call(FINAL),
    ]
    assert outcome.summary == _summary(listed=1, read=1, replaced=1)


def test_listed_final_archive_that_is_absent_makes_the_read_incomplete(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL, OTHER_FINAL])
    _publish(artifactory, OTHER_FINAL)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    assert outcome.files == {OTHER_FINAL: _content(OTHER_FINAL)}
    assert (
        f"[test-results] absent: {FINAL} was listed and is gone, and no checkpoint stands in for it"
    ) in outcome.log
    assert outcome.summary == _summary(listed=2, read=1, absent=1)


def test_checkpoint_stands_in_for_a_listed_final_archive_that_is_absent(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT, FINAL])
    _publish(artifactory, CHECKPOINT)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {CHECKPOINT: _content(CHECKPOINT)}
    assert outcome.summary == _summary(listed=2, read=1)


def test_reads_a_checkpoint_when_its_listed_final_archive_cannot_be_read(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT, FINAL])
    _publish(artifactory, CHECKPOINT)
    _fail(artifactory, FINAL)

    outcome = reader.run(RESULT_ARCHIVES)

    # The final archive is tried with the finals and again before its checkpoint; each failed
    # read counts, so the read is incomplete although the checkpoint stands in.
    assert outcome.returncode == 1
    assert outcome.files == {CHECKPOINT: _content(CHECKPOINT)}
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
    ]
    assert (
        f"[test-results] failed to read {FINAL} (fetch_build_artifact.sh exited 4)" in outcome.log
    )
    assert outcome.summary == _summary(listed=2, read=1, failed=2)


# ---------------------------------------------------------------------------
# Files that are not valid archives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content", NOT_ARCHIVES.values(), ids=NOT_ARCHIVES.keys())
def test_checkpoint_stands_in_for_a_final_archive_that_is_not_valid(
    reader: Reader, content: bytes
) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL, CHECKPOINT])
    artifactory.publish("aws", content, f"{RESULTS_DIR}/{FINAL}")
    _publish(artifactory, CHECKPOINT)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    assert outcome.stdout == ""
    assert outcome.files == {CHECKPOINT: _content(CHECKPOINT)}
    # The final archive is read by name once more before its checkpoint, as after any failed read.
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
    ]
    assert outcome.log.count(_not_valid(FINAL)) == 2
    assert outcome.summary == _summary(listed=2, read=1, failed=2)


@pytest.mark.parametrize("content", NOT_ARCHIVES.values(), ids=NOT_ARCHIVES.keys())
def test_final_archive_that_is_not_valid_is_removed_and_makes_the_read_incomplete(
    reader: Reader, content: bytes
) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL, OTHER_FINAL])
    artifactory.publish("aws", content, f"{RESULTS_DIR}/{FINAL}")
    _publish(artifactory, OTHER_FINAL)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    assert outcome.files == {OTHER_FINAL: _content(OTHER_FINAL)}
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(OTHER_FINAL),
    ]
    assert _not_valid(FINAL) in outcome.log
    assert outcome.summary == _summary(listed=2, read=1, failed=1)


def test_final_archive_with_many_members_is_valid(reader: Reader) -> None:
    # The list of its names is longer than a pipe holds at once.
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w") as archive:
        for index in range(4000):
            archive.addfile(tarfile.TarInfo(f"results/test-{index:04d}.xml"))
    content = gzip.compress(tar.getvalue(), mtime=0)
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL])
    artifactory.publish("aws", content, f"{RESULTS_DIR}/{FINAL}")

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {FINAL: content}
    assert outcome.summary == _summary(listed=1, read=1)


def test_checkpoint_that_is_not_valid_is_removed_and_makes_the_read_incomplete(
    reader: Reader,
) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    artifactory.publish("aws", NOT_ARCHIVES["cut-short"], f"{RESULTS_DIR}/{CHECKPOINT}")

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    assert outcome.files == {}
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
    ]
    assert _not_valid(CHECKPOINT) in outcome.log
    assert outcome.summary == _summary(listed=1, read=0, failed=1)


def test_only_a_name_that_ends_in_tar_gz_is_checked(reader: Reader) -> None:
    artifactory = reader.artifactory
    names = ["rerun_report.html", "results.tar.gz.sha1"]
    artifactory.publish_folder("aws", names)
    for name in names:
        artifactory.publish("aws", NOT_ARCHIVES["not-gzip"], f"{RESULTS_DIR}/{name}")

    outcome = reader.run()

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == dict.fromkeys(names, NOT_ARCHIVES["not-gzip"])
    assert outcome.summary == _summary(listed=2, read=2)


def test_files_that_are_not_valid_archives_do_not_stop_the_reading(reader: Reader) -> None:
    artifactory = reader.artifactory
    not_valid = [FINAL, OTHER_FINAL, B200_FINAL, OTHER_B200_FINAL]
    artifactory.publish_folder("aws", [*not_valid, SBSA_FINAL])
    for name in not_valid:
        artifactory.publish("aws", NOT_ARCHIVES["not-gzip"], f"{RESULTS_DIR}/{name}")
    _publish(artifactory, SBSA_FINAL)

    outcome = reader.run(RESULT_ARCHIVES)

    # The stop is for reads that were rejected or failed, and every file here was served.
    assert outcome.returncode == 1
    assert STOP_MESSAGE not in outcome.log
    assert outcome.files == {SBSA_FINAL: _content(SBSA_FINAL)}
    assert outcome.summary == _summary(listed=5, read=1, failed=4)


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 403], ids=["failed", "rejected"])
def test_listing_failure_makes_the_read_incomplete(reader: Reader, status: int) -> None:
    artifactory = reader.artifactory
    artifactory.routes[("aws", artifactory.storage_path("aws", RESULTS_DIR))] = Reply(status)
    _publish(artifactory, FINAL)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    helper_status = 4 if status == 500 else 3
    assert (
        f"[test-results] the listing failed (fetch_build_artifact.sh exited {helper_status}); "
        "some names may be missing"
    ) in outcome.log
    assert outcome.helper_calls == [reader.list_call()]
    assert outcome.files == {}
    assert outcome.summary == _summary(listed=0, read=0)


def test_names_that_a_failed_listing_printed_are_read(reader: Reader, tmp_path: Path) -> None:
    artifactory = reader.artifactory
    _publish(artifactory, FINAL)
    # A helper whose listing prints a name and then fails; its other calls go to the recorder.
    helper = tmp_path / "failing_listing_helper.sh"
    helper.write_text(
        'if [[ "$1" == --list ]]; then\n'
        f"    printf '%s\\n' '{FINAL}'\n"
        "    exit 4\n"
        "fi\n"
        f"exec bash '{reader.helper}' \"$@\"\n"
    )

    outcome = reader.run(args=[str(helper), LEGACY_RESULTS_DIR, str(reader.dest), RESULT_ARCHIVES])

    assert outcome.returncode == 1
    assert (
        "[test-results] the listing failed (fetch_build_artifact.sh exited 4); "
        "some names may be missing"
    ) in outcome.log
    assert outcome.helper_calls == [reader.read_call(FINAL)]
    assert outcome.files == {FINAL: _content(FINAL)}
    assert outcome.summary == _summary(listed=1, read=1)


def test_helper_configuration_error_makes_the_read_incomplete(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL])
    _publish(artifactory, FINAL)

    outcome = reader.run(RESULT_ARCHIVES, ARTIFACTORY_NETRC=None)

    assert outcome.returncode == 1
    assert "ARTIFACTORY_NETRC is not set" in outcome.stderr
    assert artifactory.requests == []
    assert outcome.summary == _summary(listed=0, read=0)


@pytest.mark.parametrize("status", [500, 403], ids=["failed", "rejected"])
def test_stops_after_three_failed_reads_in_a_row(reader: Reader, status: int) -> None:
    artifactory = reader.artifactory
    first, *failing, last = [f"results-A10-PyTorch-{index}.tar.gz" for index in range(1, 6)]
    # FINAL is the first name. Its checkpoint was left beside it, and OTHER_CHECKPOINT has no
    # final archive.
    assert first == FINAL
    names = [first, *failing, last, CHECKPOINT, OTHER_CHECKPOINT]
    artifactory.publish_folder("aws", names)
    _publish(artifactory, first, last, CHECKPOINT, OTHER_CHECKPOINT)
    for name in failing:
        _fail(artifactory, name, status)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    assert outcome.files == {first: _content(first)}
    assert outcome.helper_calls == [
        reader.list_call(),
        *(reader.read_call(name) for name in (first, *failing)),
    ]
    assert STOP_MESSAGE in outcome.log
    # Nothing is read after the stop; a checkpoint whose final archive was read is still counted.
    assert outcome.summary == _summary(listed=7, read=1, skipped=1, failed=3)


@pytest.mark.parametrize("status", [500, 403], ids=["failed", "rejected"])
def test_stops_at_a_final_archive_read_before_its_checkpoint(reader: Reader, status: int) -> None:
    artifactory = reader.artifactory
    failing = [f"results-A10-PyTorch-{index}.tar.gz" for index in (7, 8)]
    artifactory.publish_folder("aws", [*failing, CHECKPOINT])
    _publish(artifactory, CHECKPOINT)
    for name in (*failing, FINAL):
        _fail(artifactory, name, status)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    # The third failed read in a row is that of the checkpoint's final archive, so the checkpoint
    # is not read.
    assert outcome.files == {}
    assert outcome.helper_calls == [
        reader.list_call(),
        *(reader.read_call(name) for name in (*failing, FINAL)),
    ]
    assert STOP_MESSAGE in outcome.log
    assert outcome.summary == _summary(listed=3, read=0, failed=3)


@pytest.mark.parametrize("between", ["success", "absent", "refused", "not-valid"])
def test_stop_count_starts_again_after_any_other_result(reader: Reader, between: str) -> None:
    artifactory = reader.artifactory
    failing = [f"results-A10-PyTorch-{index}.tar.gz" for index in range(1, 5)]
    # The helper refuses a name that holds a blank as a path that it cannot map.
    middle = "results-B200 PyTorch-3.tar.gz" if between == "refused" else B200_FINAL
    # Two names whose reads fail, the name read in between, two more whose reads fail, and a last
    # one, so the reads come in this order.
    names = [*failing[:2], middle, *failing[2:], OTHER_B200_FINAL]
    artifactory.publish_folder("aws", names)
    for name in failing:
        _fail(artifactory, name)
    _publish(artifactory, OTHER_B200_FINAL)
    if between == "success":
        _publish(artifactory, B200_FINAL)
    elif between == "not-valid":
        artifactory.publish("aws", NOT_ARCHIVES["not-gzip"], f"{RESULTS_DIR}/{B200_FINAL}")

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    assert outcome.helper_calls == [reader.list_call(), *(reader.read_call(name) for name in names)]
    assert STOP_MESSAGE not in outcome.log
    assert outcome.files.get(OTHER_B200_FINAL) == _content(OTHER_B200_FINAL)
    assert outcome.summary == _summary(
        listed=6,
        read=2 if between == "success" else 1,
        absent=1 if between == "absent" else 0,
        failed=5 if between in ("refused", "not-valid") else 4,
    )


# ---------------------------------------------------------------------------
# Command line and credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help(reader: Reader, flag: str) -> None:
    outcome = reader.run(args=[flag])

    assert outcome.returncode == 0
    assert SYNOPSIS in outcome.stdout
    assert outcome.stderr == ""
    assert outcome.helper_calls == []


@pytest.mark.parametrize(
    "case",
    [
        "no-arguments",
        "no-dest-dir",
        "too-many-arguments",
        "missing-helper",
        "missing-dest-dir",
        "dest-dir-is-a-file",
        "empty-test-results-dir",
        "invalid-name-pattern",
        "invalid-exclude-pattern",
    ],
)
def test_usage_errors_exit_64(reader: Reader, tmp_path: Path, case: str) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL])
    _publish(artifactory, FINAL)
    helper, dest = str(reader.helper), str(reader.dest)
    a_file = tmp_path / "a-file"
    a_file.write_text("")
    args = {
        "no-arguments": [],
        "no-dest-dir": [helper, LEGACY_RESULTS_DIR],
        "too-many-arguments": [helper, LEGACY_RESULTS_DIR, dest, RESULT_ARCHIVES, "", "extra"],
        "missing-helper": [str(tmp_path / "no-such-helper.sh"), LEGACY_RESULTS_DIR, dest],
        "missing-dest-dir": [helper, LEGACY_RESULTS_DIR, str(tmp_path / "no-such-directory")],
        "dest-dir-is-a-file": [helper, LEGACY_RESULTS_DIR, str(a_file)],
        "empty-test-results-dir": [helper, "", dest],
        "invalid-name-pattern": [helper, LEGACY_RESULTS_DIR, dest, "results-("],
        "invalid-exclude-pattern": [helper, LEGACY_RESULTS_DIR, dest, RESULT_ARCHIVES, "results-("],
    }[case]

    outcome = reader.run(args=args)

    assert outcome.returncode == 64
    assert SYNOPSIS in outcome.stderr
    assert outcome.stdout == ""
    assert outcome.helper_calls == []
    assert artifactory.requests == []
    assert outcome.files == {}


@pytest.mark.parametrize("traced", [False, True], ids=["plain", "xtrace"])
def test_output_never_shows_the_netrc_password(reader: Reader, traced: bool) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL, OTHER_FINAL])
    _publish(artifactory, FINAL)
    _fail(artifactory, OTHER_FINAL, 401)
    # A caller running under `set -x` that exports SHELLOPTS passes xtrace on to the script.
    overrides = {"SHELLOPTS": "xtrace", "BASH_ENV": None} if traced else {}

    outcome = reader.run(RESULT_ARCHIVES, **overrides)

    # Reader.run checks every run's output for the netrc password and the token made from it.
    # The message for a rejected credential names the CI credential and not what it holds.
    assert outcome.returncode == 1
    assert outcome.files == {FINAL: _content(FINAL)}
    assert CREDENTIAL_ID in outcome.stderr
    assert [request.authorization for request in artifactory.received("aws")] == [AUTHORIZATION] * 4
