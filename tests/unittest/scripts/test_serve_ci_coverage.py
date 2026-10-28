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
"""Tests for scripts/check_serve_ci_coverage.py (the trtllm-serve CI-coverage lint).

Covers the parsers and the three checks on synthetic trees, plus the live
non-VisualGen checks on this repository. The live VisualGen check lives in
tests/unittest/scripts/visual_gen/ so that CBTS keeps it for VisualGen-only PRs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_serve_ci_coverage.py"
DATA_FILE = REPO_ROOT / "scripts" / "serve_ci_coverage.yml"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_serve_ci_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(repo: Path, rel: str, content: str = "x\n") -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _groovy(entries: list[str], raw_lines: list[str] = ()) -> str:
    lines = ["def getMultiGpuFileChanged(pipeline)", "{", "    def relatedFileList = ["]
    lines += [f'        "{entry}",' for entry in entries]
    lines += list(raw_lines)
    lines += ["    ]", "}"]
    return "\n".join(lines) + "\n"


def _mg_data(required=(), forbidden=(), managed_prefix="pkg/serve/"):
    return {
        "multi_gpu": {
            "file": "table.groovy",
            "managed_prefix": managed_prefix,
            "required": [{"path": p, "why": w} for p, w in required],
            "forbidden": [{"path": p, "why": w} for p, w in forbidden],
        }
    }


def _co_data(required=(), managed_prefixes=("/pkg/serve",)):
    return {
        "codeowners": {
            "file": "OWNERS",
            "managed_prefixes": list(managed_prefixes),
            "required": [{"path": p, "owners": list(o)} for p, o in required],
        }
    }


# --------------------------- live repository checks ---------------------------


def test_repository_passes_non_visual_gen_checks(mod):
    data = mod.load_data(DATA_FILE)
    failures = mod.check_multigpu(REPO_ROOT, data) + mod.check_codeowners(REPO_ROOT, data)
    assert not failures, "\n\n".join(failures)


def test_data_file_is_valid(mod):
    data = mod.load_data(DATA_FILE)
    assert mod.validate_data(data) == []
    rows = data["multi_gpu"]["required"] + data["multi_gpu"]["forbidden"]
    for path in (row["path"] for row in rows):
        # A directory prefix must end with "/": a bare directory name would also
        # cover unrelated sibling paths through the substring match.
        assert path.endswith("/") or "." in path.rsplit("/", 1)[-1], (
            f"multi_gpu path `{path}` has no file suffix, so it must end with `/`"
        )


# --------------------------- parse_related_file_list ---------------------------


def test_parse_related_file_list_entries_lines_and_comments(mod):
    text = _groovy(
        ["pkg/serve/a.py"],
        raw_lines=[
            "        // a comment line",
            "",
            '        "pkg/serve/b/", // trailing comment',
            '        "last/entry.py"',
        ],
    )
    assert mod.parse_related_file_list(text) == [
        (4, "pkg/serve/a.py"),
        (7, "pkg/serve/b/"),
        (8, "last/entry.py"),
    ]


def test_parse_related_file_list_rejects_unknown_line_shape(mod):
    text = _groovy(["pkg/serve/a.py"], raw_lines=['        "x.py" + variable,'])
    with pytest.raises(mod.ParseError, match="line 5"):
        mod.parse_related_file_list(text)


def test_parse_related_file_list_requires_exactly_one_list(mod):
    with pytest.raises(mod.ParseError, match="found 0"):
        mod.parse_related_file_list("def other = [\n]\n")
    with pytest.raises(mod.ParseError, match="found 2"):
        mod.parse_related_file_list(_groovy(["a.py"]) + _groovy(["b.py"]))
    with pytest.raises(mod.ParseError, match="never closed"):
        mod.parse_related_file_list('    def relatedFileList = [\n        "a.py",\n')


# --------------------------- parse_vg_prefixes ---------------------------


def test_parse_vg_prefixes_returns_lines_and_rejects_renames(mod):
    text = (
        "_VG_SRC_PREFIXES: tuple[str, ...] = (\n"
        '    "vg/a/",\n'
        '    "vg/b/",\n'
        ")\n"
        '_VG_OUTWARD_PREFIXES = ("vg/a/",)\n'
    )
    parsed = mod.parse_vg_prefixes(text)
    assert parsed["_VG_SRC_PREFIXES"] == [(2, "vg/a/"), (3, "vg/b/")]
    assert parsed["_VG_OUTWARD_PREFIXES"] == [(5, "vg/a/")]
    with pytest.raises(mod.ParseError, match="_VG_OUTWARD_PREFIXES"):
        mod.parse_vg_prefixes('_VG_SRC_PREFIXES = ("vg/a/",)\n_RENAMED = ("vg/a/",)\n')
    with pytest.raises(mod.ParseError, match="not a non-empty literal tuple"):
        mod.parse_vg_prefixes("_VG_SRC_PREFIXES = tuple(x)\n_VG_OUTWARD_PREFIXES = ()\n")


# --------------------------- check_multigpu (synthetic) ---------------------------


def test_multigpu_forward_fails_only_for_managed_dangling_entries(mod, tmp_path):
    _write(tmp_path, "pkg/serve/real.py")
    _write(
        tmp_path,
        "table.groovy",
        _groovy(["pkg/serve/real.py", "pkg/serve/missing.py", "elsewhere/also_gone.py"]),
    )
    failures = mod.check_multigpu(tmp_path, _mg_data())
    assert len(failures) == 1, "\n\n".join(failures)
    assert "`pkg/serve/missing.py`" in failures[0]
    assert "matches no file" in failures[0]
    assert "@NVIDIA/trt-llm-infra-devs" in failures[0]


def test_multigpu_dir_entry_needs_a_real_file_inside(mod, tmp_path):
    (tmp_path / "pkg/serve/empty/__pycache__").mkdir(parents=True)
    _write(tmp_path, "pkg/serve/empty/__pycache__/cached.pyc")
    _write(tmp_path, "table.groovy", _groovy(["pkg/serve/empty/"]))
    failures = mod.check_multigpu(tmp_path, _mg_data())
    assert len(failures) == 1 and "`pkg/serve/empty/`" in failures[0]


def test_multigpu_reverse_fails_for_uncovered_required_path(mod, tmp_path):
    _write(tmp_path, "pkg/serve/newpkg/mod.py")
    _write(tmp_path, "table.groovy", _groovy(["pkg/serve/other.py"]))
    _write(tmp_path, "pkg/serve/other.py")
    data = _mg_data(required=[("pkg/serve/newpkg/", "the reason"), ("pkg/serve/absent/", "x")])
    failures = mod.check_multigpu(tmp_path, data)
    assert len(failures) == 1, "\n\n".join(failures)  # absent/ does not exist yet: no failure
    assert "`pkg/serve/newpkg/` exists but no multi-GPU trigger entry covers it" in failures[0]
    assert "the reason" in failures[0]


def test_multigpu_reverse_passes_with_exact_or_ancestor_entry(mod, tmp_path):
    _write(tmp_path, "pkg/serve/newpkg/mod.py")
    data = _mg_data(required=[("pkg/serve/newpkg/", "x"), ("pkg/serve/newpkg/mod.py", "y")])
    _write(tmp_path, "table.groovy", _groovy(["pkg/serve/newpkg/"]))
    assert mod.check_multigpu(tmp_path, data) == []  # exact for the dir, ancestor for the file


def test_multigpu_forbidden_entry_fails_even_when_it_exists(mod, tmp_path):
    _write(tmp_path, "pkg/serve/sub/mod.py")
    _write(tmp_path, "table.groovy", _groovy(["pkg/serve/"]))
    data = _mg_data(forbidden=[("pkg/serve/", "it covers everything")])
    failures = mod.check_multigpu(tmp_path, data)
    assert len(failures) == 1
    assert "must not be in the multi-GPU trigger table: it covers everything" in failures[0]


# --------------------------- check_codeowners (synthetic) ---------------------------


def test_codeowners_forward_fails_only_for_managed_dangling_rules(mod, tmp_path):
    _write(tmp_path, "pkg/serve/mod.py")
    _write(
        tmp_path,
        "OWNERS",
        "/pkg/serve @org/team-a\n/pkg/serve/gone.py @org/team-a\n/unmanaged/gone.py @org/team-b\n",
    )
    failures = mod.check_codeowners(tmp_path, _co_data())
    assert len(failures) == 1, "\n\n".join(failures)
    assert "OWNERS:2: rule `/pkg/serve/gone.py` matches no file" in failures[0]


def test_codeowners_star_patterns_match_within_a_segment(mod, tmp_path):
    _write(tmp_path, "pkg/serve/my_server.py")
    _write(tmp_path, "pkg/serve/deep/nested/mod.py")
    _write(
        tmp_path,
        "OWNERS",
        "/pkg/serve/*_server.py @org/team-a\n"
        "/pkg/serve/**/mod.py @org/team-a\n"
        "/pkg/serve/*_gone.py @org/team-a\n",
    )
    failures = mod.check_codeowners(tmp_path, _co_data())
    assert len(failures) == 1, "\n\n".join(failures)
    assert "rule `/pkg/serve/*_gone.py` matches no file" in failures[0]


def test_codeowners_required_owner_fails_when_a_broader_rule_wins(mod, tmp_path):
    _write(tmp_path, "pkg/serve/special.py")
    # The specific rule is placed before the broader one, so the broader one wins.
    _write(tmp_path, "OWNERS", "/pkg/serve/special.py @org/team-b\n/pkg/serve @org/team-a\n")
    data = _co_data(
        required=[("pkg/serve", ("@org/team-a",)), ("pkg/serve/special.py", ("@org/team-b",))]
    )
    failures = mod.check_codeowners(tmp_path, data)
    assert len(failures) == 1, "\n\n".join(failures)
    assert "`pkg/serve/special.py` is owned by @org/team-a (decided by OWNERS:2)" in failures[0]
    assert "expected @org/team-b" in failures[0]


def test_codeowners_most_specific_required_path_wins(mod, tmp_path):
    _write(tmp_path, "pkg/serve/special.py")
    _write(tmp_path, "pkg/serve/plain.py")
    _write(tmp_path, "OWNERS", "/pkg/serve @ORG/Team-A\n/pkg/serve/special.py @org/team-b\n")
    data = _co_data(
        required=[("pkg/serve", ("@org/team-A",)), ("pkg/serve/special.py", ("@org/team-b",))]
    )
    assert mod.check_codeowners(tmp_path, data) == []  # owner comparison is case-insensitive


# --------------------------- CLI ---------------------------


def _cli_repo(tmp_path: Path, break_codeowners: bool = False) -> Path:
    _write(tmp_path, "pkg/serve/mod.py")
    _write(tmp_path, "vg/mod.py")
    _write(tmp_path, "table.groovy", _groovy(["pkg/serve/mod.py"]))
    _write(
        tmp_path,
        "rules.py",
        '_VG_SRC_PREFIXES: tuple[str, ...] = ("vg/",)\n_VG_OUTWARD_PREFIXES = ("vg/",)\n',
    )
    owners = "/pkg/serve @org/team-a\n"
    if break_codeowners:
        owners += "/pkg/serve/gone.py @org/team-a\n"
    _write(tmp_path, "OWNERS", owners)
    data = {
        "version": 1,
        "multi_gpu": {
            "file": "table.groovy",
            "managed_prefix": "pkg/serve/",
            "required": [{"path": "pkg/serve/mod.py", "why": "w"}],
            "forbidden": [{"path": "pkg/serve/", "why": "w"}],
        },
        "visual_gen": {"file": "rules.py", "tuples": ["_VG_SRC_PREFIXES", "_VG_OUTWARD_PREFIXES"]},
        "codeowners": {
            "file": "OWNERS",
            "managed_prefixes": ["/pkg/serve"],
            "required": [{"path": "pkg/serve", "owners": ["@org/team-a"]}],
        },
    }
    _write(tmp_path, "data.yml", yaml.safe_dump(data))
    return tmp_path


def test_cli_exit_codes_and_check_selection(mod, tmp_path, capsys):
    repo = _cli_repo(tmp_path / "clean")
    args = ["--repo-root", str(repo), "--data", str(repo / "data.yml")]
    assert mod.main(args) == 0
    assert "serve CI coverage: 3 checks passed" in capsys.readouterr().out

    broken = _cli_repo(tmp_path / "broken", break_codeowners=True)
    args = ["--repo-root", str(broken), "--data", str(broken / "data.yml")]
    assert mod.main(args) == 1
    assert "matches no file" in capsys.readouterr().out
    # --checks selects subsets: the broken check is not selected, so the run passes.
    assert mod.main(args + ["--checks", "multigpu,visual_gen"]) == 0
    assert "serve CI coverage: 2 checks passed" in capsys.readouterr().out
    assert mod.main(args + ["--checks", "codeowners"]) == 1
    capsys.readouterr()


def test_cli_rejects_unknown_check_and_invalid_data(mod, tmp_path, capsys):
    repo = _cli_repo(tmp_path)
    args = ["--repo-root", str(repo), "--data", str(repo / "data.yml")]
    with pytest.raises(SystemExit):
        mod.main(args + ["--checks", "nonsense"])
    capsys.readouterr()
    _write(repo, "bad.yml", "version: 2\n")
    assert mod.main(["--repo-root", str(repo), "--data", str(repo / "bad.yml")]) == 1
    out = capsys.readouterr().out
    assert "cannot parse" in out and "`version` must be 1" in out
