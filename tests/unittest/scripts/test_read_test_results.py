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
only when the attempt's final archive could not be read. The loopback servers of
``test_fetch_build_artifact`` stand in for the two Artifactory locations, and the script runs the
real helper through a recorder that notes each call's arguments.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from .test_fetch_build_artifact import (
    AUTHORIZATION,
    BASIC_TOKEN,
    CREDENTIAL_ID,
    FETCH_BUILD_ARTIFACT,
    HOSTS,
    NETRC_PASSWORD,
    REPO_ROOT,
    REPOS,
    RESULTS_DIR,
    URM_NETRC_PASSWORD,
    FakeArtifactory,
    Helper,
    Reply,
    _in_turn,
    _unused_port,
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
URM_ONLY = "results-B200-PyTorch-3.tar.gz"
OTHER_URM_ONLY = "results-B200-PyTorch-4.tar.gz"
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
        assert URM_NETRC_PASSWORD not in output
        assert BASIC_TOKEN not in output
        assert list(self._tmpdir.iterdir()) == [], "temporary files were left behind"
        assert list(self._cwd.iterdir()) == [], "files were written outside <dest dir>"
        calls = self._calls.read_text().splitlines() if self._calls.exists() else []
        files = {path.name: path.read_bytes() for path in self.dest.iterdir()}
        return Outcome(
            proc.returncode, proc.stdout, proc.stderr, [call.split("\t") for call in calls], files
        )

    def list_call(self, *, optional: bool = True) -> list[str]:
        """Return the helper's arguments for the listing."""
        flags = ["--secondary-optional"] if optional else []
        return [*flags, "--list", LEGACY_RESULTS_DIR]

    def read_call(self, name: str, *, optional: bool = True) -> list[str]:
        """Return the helper's arguments for the read of ``name``."""
        flags = ["--secondary-optional"] if optional else []
        return [*flags, f"{LEGACY_RESULTS_DIR}/{name}", str(self.dest / name)]


@pytest.fixture
def artifactory() -> Iterator[FakeArtifactory]:
    fake = FakeArtifactory()
    yield fake
    fake.close()


@pytest.fixture
def reader(tmp_path: Path, artifactory: FakeArtifactory) -> Reader:
    return Reader(tmp_path, Helper(tmp_path, artifactory))


def _content(location: str, name: str) -> bytes:
    return f"{name} as {location} holds it".encode()


def _publish(artifactory: FakeArtifactory, location: str, *names: str) -> None:
    """Serve each of ``names`` in the run's test-results folder on a location."""
    for name in names:
        artifactory.publish(location, _content(location, name), f"{RESULTS_DIR}/{name}")


def _lookup(artifactory: FakeArtifactory, location: str, name: str) -> tuple[str, str]:
    """Return the route of the storage lookup that starts every read of ``name``."""
    return (location, artifactory.storage_path(location, f"{RESULTS_DIR}/{name}"))


def _fail(artifactory: FakeArtifactory, location: str, name: str, status: int = 500) -> None:
    """Make a location answer the lookup of ``name`` with ``status``."""
    artifactory.routes[_lookup(artifactory, location, name)] = Reply(status)


def _closed_url(location: str) -> str:
    """Return a base URL for a location on a port that nothing listens on."""
    host = HOSTS[location]
    return f"http://{host}:{_unused_port(host)}/artifactory/{REPOS[location]}"


def _asked(artifactory: FakeArtifactory, name: str) -> list[str]:
    """Return the location of each request for ``name``, lookups and downloads, in order."""
    suffix = f"/{RESULTS_DIR}/{name}"
    return [request.server for request in artifactory.requests if request.path.endswith(suffix)]


def _summary(
    *,
    listed: int,
    aws: int | None,
    urm: int,
    read: int,
    skipped: int = 0,
    replaced: int = 0,
    absent: int = 0,
    unreachable: int = 0,
    failed: int = 0,
) -> str:
    """Return the summary line; ``aws`` is ``None`` when aws was not asked."""
    on_aws = "aws not asked" if aws is None else f"aws {aws}"
    return (
        f"[test-results] listed {listed} ({on_aws}, urm {urm}), read {read}, "
        f"checkpoints skipped {skipped}, replaced {replaced}, absent {absent}, "
        f"urm unreachable {unreachable}, failed {failed}"
    )


# ---------------------------------------------------------------------------
# Listing and reading
# ---------------------------------------------------------------------------


def test_reads_each_listed_archive_once(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL, OTHER_FINAL, *OTHER_FILES, "logs/"])
    artifactory.publish_folder("urm", [FINAL, OTHER_FINAL, URM_ONLY, *OTHER_FILES, "logs/"])
    _publish(artifactory, "aws", FINAL, OTHER_FINAL, *OTHER_FILES)
    _publish(artifactory, "urm", FINAL, OTHER_FINAL, URM_ONLY, *OTHER_FILES)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.stdout == ""
    # The primary's copy of a name on both locations, and urm's of a name that only urm holds.
    assert outcome.files == {
        FINAL: _content("aws", FINAL),
        OTHER_FINAL: _content("aws", OTHER_FINAL),
        URM_ONLY: _content("urm", URM_ONLY),
    }
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(OTHER_FINAL),
        reader.read_call(URM_ONLY),
    ]
    assert outcome.summary == _summary(listed=3, aws=2, urm=3, read=3)
    assert all(request.authorization == AUTHORIZATION for request in artifactory.received("aws"))
    assert all(request.authorization is None for request in artifactory.received("urm"))


@pytest.mark.parametrize(
    "results_dir",
    [LEGACY_RESULTS_DIR, RESULTS_DIR, f"{LEGACY_RESULTS_DIR}/"],
    ids=["legacy", "layout", "trailing-slash"],
)
def test_reads_into_the_current_directory(reader: Reader, results_dir: str) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL])
    _publish(artifactory, "aws", FINAL)

    outcome = reader.run(
        args=[str(reader.helper), results_dir, ".", RESULT_ARCHIVES], cwd=reader.dest
    )

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {FINAL: _content("aws", FINAL)}
    assert outcome.summary == _summary(listed=1, aws=1, urm=0, read=1)


def test_no_folder_on_either_location_is_complete(reader: Reader) -> None:
    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {}
    assert outcome.helper_calls == [reader.list_call()]
    assert outcome.summary == _summary(listed=0, aws=0, urm=0, read=0)


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
    _publish(artifactory, "aws", *names)

    outcome = reader.run(*patterns)

    assert outcome.returncode == 0, outcome.stderr
    assert sorted(outcome.files) == sorted(kept)
    for name in names:
        if name not in kept:
            assert _asked(artifactory, name) == []
    assert outcome.summary == _summary(listed=len(kept), aws=len(kept), urm=0, read=len(kept))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "listings",
    [
        {"aws": [CHECKPOINT, FINAL], "urm": [CHECKPOINT, FINAL]},
        {"aws": [FINAL, CHECKPOINT], "urm": [FINAL, CHECKPOINT]},
        {"aws": [FINAL], "urm": [CHECKPOINT, FINAL]},
    ],
    ids=["checkpoint-listed-first", "final-listed-first", "checkpoint-left-on-urm"],
)
def test_skips_a_checkpoint_whose_final_archive_was_read(
    reader: Reader, listings: dict[str, list[str]]
) -> None:
    artifactory = reader.artifactory
    for location, names in listings.items():
        artifactory.publish_folder(location, names)
        _publish(artifactory, location, *names)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {FINAL: _content("aws", FINAL)}
    assert _asked(artifactory, CHECKPOINT) == []
    assert outcome.helper_calls == [reader.list_call(), reader.read_call(FINAL)]
    assert (
        f"[test-results] skipped checkpoint {CHECKPOINT}: its final archive {FINAL} was read"
        in outcome.log
    )
    assert outcome.summary == _summary(listed=2, aws=len(listings["aws"]), urm=2, read=1, skipped=1)


def test_reads_a_checkpoint_whose_final_archive_is_absent(reader: Reader) -> None:
    artifactory = reader.artifactory
    for location in ("aws", "urm"):
        artifactory.publish_folder(location, [CHECKPOINT])
        _publish(artifactory, location, CHECKPOINT)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {CHECKPOINT: _content("aws", CHECKPOINT)}
    # The final archive is asked for first, because it may have landed after the listing.
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
    ]
    assert _asked(artifactory, FINAL) == ["aws", "urm"]
    assert outcome.summary == _summary(listed=1, aws=1, urm=1, read=1)


def test_reads_an_unlisted_final_archive_in_place_of_its_checkpoint(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    _publish(artifactory, "aws", CHECKPOINT, FINAL)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {FINAL: _content("aws", FINAL)}
    assert _asked(artifactory, CHECKPOINT) == []
    assert (
        f"[test-results] replaced checkpoint {CHECKPOINT}: read its final archive {FINAL} instead"
        in outcome.log
    )
    assert outcome.summary == _summary(listed=1, aws=1, urm=0, read=1, replaced=1)


@pytest.mark.parametrize(
    "final_at_the_second_try", [True, False], ids=["final-present", "final-absent"]
)
def test_checkpoint_gone_when_read_sends_the_read_to_its_final_archive(
    reader: Reader, final_at_the_second_try: bool
) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    if final_at_the_second_try:
        _publish(artifactory, "aws", FINAL)
        lookup = _lookup(artifactory, "aws", FINAL)
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
        assert outcome.files == {FINAL: _content("aws", FINAL)}
        assert outcome.summary == _summary(listed=1, aws=1, urm=0, read=1, replaced=1)
    else:
        assert outcome.returncode == 1
        assert outcome.files == {}
        assert (
            f"[test-results] failed: checkpoint {CHECKPOINT} is gone, and neither location has "
            f"its final archive {FINAL}"
        ) in outcome.log
        assert outcome.summary == _summary(listed=1, aws=1, urm=0, read=0, failed=1)


def test_listed_final_archive_that_is_absent_makes_the_read_incomplete(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL, OTHER_FINAL])
    _publish(artifactory, "aws", OTHER_FINAL)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    assert outcome.files == {OTHER_FINAL: _content("aws", OTHER_FINAL)}
    assert (
        f"[test-results] absent: neither location has {FINAL}, which was listed, and no "
        "checkpoint stands in for it"
    ) in outcome.log
    assert outcome.summary == _summary(listed=2, aws=2, urm=0, read=1, absent=1)


def test_checkpoint_stands_in_for_a_listed_final_archive_that_is_absent(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT, FINAL])
    _publish(artifactory, "aws", CHECKPOINT)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {CHECKPOINT: _content("aws", CHECKPOINT)}
    assert outcome.summary == _summary(listed=2, aws=2, urm=0, read=1)


def test_reads_a_checkpoint_when_its_listed_final_archive_cannot_be_read(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT, FINAL])
    _publish(artifactory, "aws", CHECKPOINT)
    _fail(artifactory, "aws", FINAL)

    outcome = reader.run(RESULT_ARCHIVES)

    # The final archive is tried with the finals and again before its checkpoint; each failed
    # read on the primary counts, so the read is incomplete although the checkpoint stands in.
    assert outcome.returncode == 1
    assert outcome.files == {CHECKPOINT: _content("aws", CHECKPOINT)}
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
    ]
    assert (
        f"[test-results] failed to read {FINAL} (fetch_build_artifact.sh exited 4)" in outcome.log
    )
    assert outcome.summary == _summary(listed=2, aws=2, urm=0, read=1, failed=2)


# ---------------------------------------------------------------------------
# Failures of urm while aws is the primary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failure", [500, 403, "unreachable"])
def test_urm_listing_failure_is_only_logged(reader: Reader, failure: int | str) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL, OTHER_FINAL])
    _publish(artifactory, "aws", FINAL, OTHER_FINAL)
    overrides = {}
    if failure == "unreachable":
        overrides["LLM_ARTIFACTS_URM_URL"] = _closed_url("urm")
    else:
        assert isinstance(failure, int)
        folder = artifactory.storage_path("urm", RESULTS_DIR)
        artifactory.routes[("urm", folder)] = Reply(failure)

    outcome = reader.run(RESULT_ARCHIVES, **overrides)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {
        FINAL: _content("aws", FINAL),
        OTHER_FINAL: _content("aws", OTHER_FINAL),
    }
    assert (
        "[test-results] urm could not be listed (fetch_build_artifact.sh exited 5); going on "
        "with the names that aws listed"
    ) in outcome.log
    assert outcome.summary == _summary(listed=2, aws=2, urm=0, read=2, unreachable=1)


def test_urm_read_failure_is_only_logged(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL])
    artifactory.publish_folder("urm", [FINAL, URM_ONLY, OTHER_URM_ONLY])
    _publish(artifactory, "aws", FINAL)
    _fail(artifactory, "urm", URM_ONLY, 500)
    _fail(artifactory, "urm", OTHER_URM_ONLY, 403)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {FINAL: _content("aws", FINAL)}
    assert _asked(artifactory, URM_ONLY) == ["aws", "urm", "urm", "urm"]
    assert _asked(artifactory, OTHER_URM_ONLY) == ["aws", "urm"]
    for name in (URM_ONLY, OTHER_URM_ONLY):
        assert (
            f"[test-results] urm unreachable: aws does not have {name}, and urm could not "
            "answer (fetch_build_artifact.sh exited 5); not read"
        ) in outcome.log
    assert outcome.summary == _summary(listed=3, aws=1, urm=3, read=1, unreachable=2)


def test_urm_failure_for_a_final_archive_leaves_its_checkpoint_to_read(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    artifactory.publish_folder("urm", [CHECKPOINT, FINAL])
    _publish(artifactory, "aws", CHECKPOINT)
    _fail(artifactory, "urm", FINAL)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {CHECKPOINT: _content("aws", CHECKPOINT)}
    # urm fails for the final archive with the finals and again before the checkpoint.
    assert outcome.summary == _summary(listed=2, aws=1, urm=2, read=1, unreachable=2)


@pytest.mark.parametrize(
    "final_at_the_second_try", [True, False], ids=["final-present", "final-absent"]
)
def test_checkpoint_gone_from_aws_sends_the_read_to_its_final_archive_while_urm_fails(
    reader: Reader, final_at_the_second_try: bool
) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    _fail(artifactory, "urm", FINAL)
    _fail(artifactory, "urm", CHECKPOINT)
    if final_at_the_second_try:
        _publish(artifactory, "aws", FINAL)
        lookup = _lookup(artifactory, "aws", FINAL)
        found = artifactory.routes[lookup]
        assert isinstance(found, Reply)
        artifactory.routes[lookup] = _in_turn(Reply(404), found)

    outcome = reader.run(RESULT_ARCHIVES)

    # aws listed the checkpoint and lacks it now, so its stage has uploaded the final archive
    # there; urm's failure does not hide that.
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
        reader.read_call(FINAL),
    ]
    if final_at_the_second_try:
        assert outcome.returncode == 0, outcome.stderr
        assert outcome.files == {FINAL: _content("aws", FINAL)}
        assert outcome.summary == _summary(
            listed=1, aws=1, urm=0, read=1, replaced=1, unreachable=2
        )
    else:
        assert outcome.returncode == 1
        assert outcome.files == {}
        assert (
            f"[test-results] failed: checkpoint {CHECKPOINT} is gone from aws, which does not "
            f"have its final archive {FINAL} either, and urm could not answer"
        ) in outcome.log
        assert outcome.summary == _summary(listed=1, aws=1, urm=0, read=0, unreachable=3, failed=1)


@pytest.mark.parametrize(
    "final_at_the_second_try", [True, False], ids=["final-present", "final-absent"]
)
def test_checkpoint_gone_from_aws_is_not_taken_from_the_copy_that_urm_still_has(
    reader: Reader, final_at_the_second_try: bool
) -> None:
    artifactory = reader.artifactory
    for location in ("aws", "urm"):
        artifactory.publish_folder(location, [CHECKPOINT])
    _publish(artifactory, "urm", CHECKPOINT)
    if final_at_the_second_try:
        _publish(artifactory, "aws", FINAL)
        lookup = _lookup(artifactory, "aws", FINAL)
        found = artifactory.routes[lookup]
        assert isinstance(found, Reply)
        artifactory.routes[lookup] = _in_turn(Reply(404), found)

    outcome = reader.run(RESULT_ARCHIVES)

    # aws listed the checkpoint and lacks it now, so its stage has uploaded the final archive
    # there and deleted the checkpoint; urm's copy is one that the stage has not deleted yet.
    assert outcome.returncode == 0, outcome.stderr
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
        reader.read_call(FINAL),
    ]
    assert _asked(artifactory, CHECKPOINT) == ["aws", "urm", "urm"]
    if final_at_the_second_try:
        assert outcome.files == {FINAL: _content("aws", FINAL)}
        assert (
            f"[test-results] replaced checkpoint {CHECKPOINT}: read its final archive {FINAL} "
            "instead"
        ) in outcome.log
        assert outcome.summary == _summary(listed=1, aws=1, urm=1, read=1, replaced=1)
    else:
        assert outcome.files == {CHECKPOINT: _content("urm", CHECKPOINT)}
        assert (
            f"[test-results] kept urm's copy of checkpoint {CHECKPOINT}: aws no longer has it, "
            f"and its final archive {FINAL} could not be read"
        ) in outcome.log
        assert outcome.summary == _summary(listed=1, aws=1, urm=1, read=1)


def test_checkpoint_that_only_urm_listed_is_read_from_urm(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [])
    artifactory.publish_folder("urm", [CHECKPOINT])
    _publish(artifactory, "urm", CHECKPOINT)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {CHECKPOINT: _content("urm", CHECKPOINT)}
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
    ]
    assert outcome.summary == _summary(listed=1, aws=0, urm=1, read=1)


def test_checkpoint_deleted_on_aws_between_its_lookup_and_its_download(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [CHECKPOINT])
    _publish(artifactory, "aws", CHECKPOINT, FINAL)
    final_lookup = _lookup(artifactory, "aws", FINAL)
    found = artifactory.routes[final_lookup]
    assert isinstance(found, Reply)
    artifactory.routes[final_lookup] = _in_turn(Reply(404), found)
    # The checkpoint's lookup finds it; the delete lands before the download and the next lookup.
    checkpoint_lookup = _lookup(artifactory, "aws", CHECKPOINT)
    listed = artifactory.routes[checkpoint_lookup]
    assert isinstance(listed, Reply)
    artifactory.routes[checkpoint_lookup] = _in_turn(listed, Reply(404))
    del artifactory.routes[("aws", artifactory.file_path("aws", f"{RESULTS_DIR}/{CHECKPOINT}"))]

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {FINAL: _content("aws", FINAL)}
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
        reader.read_call(FINAL),
    ]
    assert outcome.summary == _summary(listed=1, aws=1, urm=0, read=1, replaced=1)


def test_urm_failure_for_a_checkpoint_that_only_urm_listed_is_only_logged(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [])
    artifactory.publish_folder("urm", [CHECKPOINT])
    _fail(artifactory, "urm", FINAL)
    _fail(artifactory, "urm", CHECKPOINT)

    outcome = reader.run(RESULT_ARCHIVES)

    # aws never listed the attempt, so nothing shows that aws lost a file: the final archive is
    # not asked for again, and urm's failures mark nothing.
    assert outcome.returncode == 0, outcome.stderr
    assert outcome.helper_calls == [
        reader.list_call(),
        reader.read_call(FINAL),
        reader.read_call(CHECKPOINT),
    ]
    assert outcome.files == {}
    assert outcome.summary == _summary(listed=1, aws=0, urm=1, read=0, unreachable=2)


# ---------------------------------------------------------------------------
# Failures of the primary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 403], ids=["failed", "rejected"])
def test_aws_listing_failure_makes_the_read_incomplete(reader: Reader, status: int) -> None:
    artifactory = reader.artifactory
    artifactory.routes[("aws", artifactory.storage_path("aws", RESULTS_DIR))] = Reply(status)
    artifactory.publish_folder("urm", [FINAL, URM_ONLY])
    _publish(artifactory, "aws", FINAL)
    _publish(artifactory, "urm", FINAL, URM_ONLY)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    helper_status = 4 if status == 500 else 3
    assert (
        f"[test-results] the listing failed (fetch_build_artifact.sh exited {helper_status}); "
        "some names may be missing"
    ) in outcome.log
    if status == 500:
        # The helper still lists urm, and the names that urm listed are read by the read rule.
        assert outcome.files == {
            FINAL: _content("aws", FINAL),
            URM_ONLY: _content("urm", URM_ONLY),
        }
        assert outcome.summary == _summary(listed=2, aws=0, urm=2, read=2)
    else:
        # The helper does not ask urm after aws rejects the credential.
        assert outcome.files == {}
        assert outcome.summary == _summary(listed=0, aws=0, urm=0, read=0)


def test_helper_configuration_error_makes_the_read_incomplete(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL])
    _publish(artifactory, "aws", FINAL)

    outcome = reader.run(RESULT_ARCHIVES, ARTIFACTORY_NETRC=None)

    assert outcome.returncode == 1
    assert "ARTIFACTORY_NETRC is not set" in outcome.stderr
    assert artifactory.requests == []
    assert outcome.summary == _summary(listed=0, aws=0, urm=0, read=0)


@pytest.mark.parametrize("status", [500, 403], ids=["failed", "rejected"])
def test_stops_after_three_failed_reads_in_a_row(reader: Reader, status: int) -> None:
    artifactory = reader.artifactory
    first, *failing, last = [f"results-A10-PyTorch-{index}.tar.gz" for index in range(1, 6)]
    # FINAL is the first name. Its checkpoint was left beside it, and OTHER_CHECKPOINT has no
    # final archive.
    assert first == FINAL
    names = [first, *failing, last, CHECKPOINT, OTHER_CHECKPOINT]
    artifactory.publish_folder("aws", names)
    _publish(artifactory, "aws", first, last, CHECKPOINT, OTHER_CHECKPOINT)
    for name in failing:
        _fail(artifactory, "aws", name, status)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    assert outcome.files == {first: _content("aws", first)}
    assert outcome.helper_calls == [
        reader.list_call(),
        *(reader.read_call(name) for name in (first, *failing)),
    ]
    assert STOP_MESSAGE in outcome.log
    # Nothing is read after the stop; a checkpoint whose final archive was read is still counted.
    assert outcome.summary == _summary(listed=7, aws=7, urm=0, read=1, skipped=1, failed=3)


@pytest.mark.parametrize("between", ["success", "absent", "urm-failure"])
def test_stop_count_starts_again_after_any_other_result(reader: Reader, between: str) -> None:
    artifactory = reader.artifactory
    failing = [f"results-A10-PyTorch-{index}.tar.gz" for index in range(1, 5)]
    # aws lists two names that it fails to serve. urm lists the name read in between, two more
    # that aws fails to serve, and a last one, so the reads come in this order.
    artifactory.publish_folder("aws", failing[:2])
    artifactory.publish_folder("urm", [URM_ONLY, *failing[2:], OTHER_URM_ONLY])
    for name in failing:
        _fail(artifactory, "aws", name)
    _publish(artifactory, "urm", OTHER_URM_ONLY)
    if between == "success":
        _publish(artifactory, "urm", URM_ONLY)
    elif between == "urm-failure":
        _fail(artifactory, "urm", URM_ONLY)

    outcome = reader.run(RESULT_ARCHIVES)

    assert outcome.returncode == 1
    assert outcome.helper_calls == [
        reader.list_call(),
        *(reader.read_call(name) for name in (*failing[:2], URM_ONLY, *failing[2:])),
        reader.read_call(OTHER_URM_ONLY),
    ]
    assert STOP_MESSAGE not in outcome.log
    assert outcome.files.get(OTHER_URM_ONLY) == _content("urm", OTHER_URM_ONLY)
    assert outcome.summary == _summary(
        listed=6,
        aws=2,
        urm=4,
        read=2 if between == "success" else 1,
        absent=1 if between == "absent" else 0,
        unreachable=1 if between == "urm-failure" else 0,
        failed=4,
    )


# ---------------------------------------------------------------------------
# LLM_ARTIFACTS_PRIMARY=urm (rollback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aws", ["has-the-files", "unreachable"])
def test_urm_primary_asks_urm_alone(reader: Reader, aws: str) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("urm", [FINAL, OTHER_CHECKPOINT])
    _publish(artifactory, "urm", FINAL, OTHER_CHECKPOINT)
    overrides = {"LLM_ARTIFACTS_PRIMARY": "urm"}
    if aws == "has-the-files":
        # A final archive that only aws holds is missed, and its checkpoint is read.
        artifactory.publish_folder("aws", [FINAL, OTHER_FINAL, URM_ONLY])
        _publish(artifactory, "aws", FINAL, OTHER_FINAL, URM_ONLY)
    else:
        overrides["LLM_ARTIFACTS_AWS_URL"] = _closed_url("aws")

    outcome = reader.run(RESULT_ARCHIVES, **overrides)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.files == {
        FINAL: _content("urm", FINAL),
        OTHER_CHECKPOINT: _content("urm", OTHER_CHECKPOINT),
    }
    assert outcome.helper_calls == [
        reader.list_call(optional=False),
        reader.read_call(FINAL, optional=False),
        reader.read_call(OTHER_FINAL, optional=False),
        reader.read_call(OTHER_CHECKPOINT, optional=False),
    ]
    assert artifactory.paths("aws") == []
    assert outcome.summary == _summary(listed=2, aws=None, urm=2, read=2)


def test_urm_primary_failure_of_urm_makes_the_read_incomplete(reader: Reader) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("urm", [FINAL, OTHER_FINAL])
    _publish(artifactory, "urm", FINAL)
    _fail(artifactory, "urm", OTHER_FINAL)

    outcome = reader.run(RESULT_ARCHIVES, LLM_ARTIFACTS_PRIMARY="urm")

    assert outcome.returncode == 1
    assert outcome.files == {FINAL: _content("urm", FINAL)}
    assert artifactory.paths("aws") == []
    assert outcome.summary == _summary(listed=2, aws=None, urm=2, read=1, failed=1)


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
    _publish(artifactory, "aws", FINAL)
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


@pytest.mark.parametrize("primary", ["bogus", "AWS"])
def test_unknown_primary_is_a_usage_error(reader: Reader, primary: str) -> None:
    outcome = reader.run(RESULT_ARCHIVES, LLM_ARTIFACTS_PRIMARY=primary)

    assert outcome.returncode == 64
    assert "LLM_ARTIFACTS_PRIMARY" in outcome.stderr
    assert outcome.helper_calls == []


@pytest.mark.parametrize("primary", [None, ""], ids=["unset", "empty"])
def test_aws_is_the_default_primary(reader: Reader, primary: str | None) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL])
    _publish(artifactory, "aws", FINAL)

    outcome = reader.run(RESULT_ARCHIVES, LLM_ARTIFACTS_PRIMARY=primary)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.helper_calls == [reader.list_call(), reader.read_call(FINAL)]
    assert outcome.summary == _summary(listed=1, aws=1, urm=0, read=1)


@pytest.mark.parametrize("traced", [False, True], ids=["plain", "xtrace"])
def test_output_never_shows_the_netrc_password(reader: Reader, traced: bool) -> None:
    artifactory = reader.artifactory
    artifactory.publish_folder("aws", [FINAL, OTHER_FINAL])
    _publish(artifactory, "aws", FINAL)
    _fail(artifactory, "aws", OTHER_FINAL, 401)
    # A caller running under `set -x` that exports SHELLOPTS passes xtrace on to the script.
    overrides = {"SHELLOPTS": "xtrace", "BASH_ENV": None} if traced else {}

    outcome = reader.run(RESULT_ARCHIVES, **overrides)

    # Reader.run checks every run's output for the netrc passwords and the token made from them.
    # The message for a rejected credential names the CI credential and not what it holds.
    assert outcome.returncode == 1
    assert outcome.files == {FINAL: _content("aws", FINAL)}
    assert CREDENTIAL_ID in outcome.stderr
    assert [request.authorization for request in artifactory.received("aws")] == [AUTHORIZATION] * 4
