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
"""Tests for CBTS's shared repository-wide static import and binding-reference index."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.repository_reference import RepositoryReferenceIndex

pytestmark = pytest.mark.cpu_only


@pytest.fixture()
def reference_root(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=60)
    return tmp_path


def _stage_reference_files(reference_root: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=reference_root, check=True, timeout=60)


def test_repository_reference_index_follows_import_relationships(
    reference_root: Path,
) -> None:
    package = reference_root / "pkg"
    package.mkdir()
    (package / "owner.py").write_text("VALUE = 1\nOTHER = 2\nTHIRD = 3\n")
    (package / "direct.py").write_text("from pkg.owner import VALUE\n")
    (package / "module.py").write_text("import pkg.owner as owner\nprint(owner.OTHER)\n")
    (package / "relative.py").write_text("from . import owner\nprint(owner.THIRD)\n")
    (package / "unrelated.py").write_text("VALUE = 4\n")
    _stage_reference_files(reference_root)

    references = RepositoryReferenceIndex(reference_root).external_references(
        "pkg/owner.py", {"VALUE", "OTHER", "THIRD"}
    )

    assert references == {"VALUE", "OTHER", "THIRD"}


def test_repository_reference_index_treats_module_escape_as_unresolved(
    reference_root: Path,
) -> None:
    package = reference_root / "pkg"
    package.mkdir()
    (package / "owner.py").write_text("VALUE = 1\nOTHER = 2\n")
    (package / "consumer.py").write_text("import pkg.owner as owner\nregister(owner)\n")
    _stage_reference_files(reference_root)

    references = RepositoryReferenceIndex(reference_root).external_references(
        "pkg/owner.py", {"VALUE", "OTHER"}
    )

    assert references == {"VALUE", "OTHER"}


def test_repository_reference_index_tracks_simple_module_alias(reference_root: Path) -> None:
    package = reference_root / "pkg"
    package.mkdir()
    (package / "owner.py").write_text("VALUE = 1\nOTHER = 2\n")
    (package / "consumer.py").write_text(
        "import pkg.owner as owner\nalias = owner\nprint(alias.OTHER)\n"
    )
    _stage_reference_files(reference_root)

    references = RepositoryReferenceIndex(reference_root).external_references(
        "pkg/owner.py", {"VALUE", "OTHER"}
    )

    assert references == {"OTHER"}


def test_repository_reference_index_tracks_direct_import_module(
    reference_root: Path,
) -> None:
    package = reference_root / "pkg"
    package.mkdir()
    (package / "owner.py").write_text("VALUE = 1\nOTHER = 2\n")
    (package / "consumer.py").write_text(
        'from importlib import import_module\nowner = import_module("pkg.owner")\n'
    )
    _stage_reference_files(reference_root)

    references = RepositoryReferenceIndex(reference_root).external_references(
        "pkg/owner.py", {"VALUE", "OTHER"}
    )

    assert references == {"VALUE", "OTHER"}


def test_repository_reference_index_finds_untracked_module(reference_root: Path) -> None:
    package = reference_root / "pkg"
    package.mkdir()
    (package / "owner.py").write_text("VALUE = 1\n")
    _stage_reference_files(reference_root)
    (package / "consumer.py").write_text("from pkg.owner import VALUE\n")

    references = RepositoryReferenceIndex(reference_root).external_references(
        "pkg/owner.py", {"VALUE"}
    )

    assert references == {"VALUE"}


def test_repository_reference_index_reports_direct_importers(tmp_path: Path) -> None:
    package = tmp_path / "perf"
    package.mkdir()
    (package / "helper.py").write_text("VALUE = 1\n")
    (package / "test_relative.py").write_text("from .helper import VALUE\n")
    (package / "test_prefixed.py").write_text("from defs.perf.helper import VALUE\n")

    importers = RepositoryReferenceIndex(
        tmp_path,
        module_prefixes=("defs",),
    ).direct_importers("perf/helper.py")

    assert importers.complete
    assert importers.paths == ("perf/test_prefixed.py", "perf/test_relative.py")
    assert not importers.limitation


def test_repository_reference_index_ignores_non_code_leaf_mentions(tmp_path: Path) -> None:
    package = tmp_path / "perf"
    package.mkdir()
    (package / "helper.py").write_text("VALUE = 1\n")
    (package / "test_consumer.py").write_text("from .helper import VALUE\n")
    (package / "unrelated.py").write_text(
        'helper_fn = 1\nLABEL = "helper"\n# helper is not referenced\n'
    )

    importers = RepositoryReferenceIndex(tmp_path).direct_importers("perf/helper.py")

    assert importers.complete
    assert importers.paths == ("perf/test_consumer.py",)
    assert not importers.limitation


def test_repository_reference_index_reports_ambiguous_short_import(tmp_path: Path) -> None:
    package = tmp_path / "perf"
    package.mkdir()
    (package / "helper.py").write_text("VALUE = 1\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "consumer.py").write_text("import helper\n")

    importers = RepositoryReferenceIndex(tmp_path).direct_importers("perf/helper.py")

    assert not importers.complete
    assert not importers.paths
    assert importers.limitation == "ambiguous short import in other/consumer.py: helper"


def test_repository_reference_index_reports_unresolved_module_reference(
    tmp_path: Path,
) -> None:
    package = tmp_path / "perf"
    package.mkdir()
    (package / "helper.py").write_text("VALUE = 1\n")
    (tmp_path / "consumer.py").write_text("print(helper.VALUE)\n")

    importers = RepositoryReferenceIndex(tmp_path).direct_importers("perf/helper.py")

    assert not importers.complete
    assert not importers.paths
    assert importers.limitation == "unresolved module reference in consumer.py"


def test_repository_reference_index_marks_dynamic_importers_incomplete(tmp_path: Path) -> None:
    package = tmp_path / "perf"
    package.mkdir()
    (package / "helper.py").write_text("VALUE = 1\n")
    (package / "test_consumer.py").write_text(
        "import importlib\nmodule_name = '.helper'\nimportlib.import_module(module_name)\n"
    )

    importers = RepositoryReferenceIndex(tmp_path).direct_importers("perf/helper.py")

    assert not importers.complete
    assert not importers.paths
    assert importers.limitation == "dynamic import may target perf/helper.py"
