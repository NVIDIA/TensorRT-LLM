import json
from pathlib import Path

import pytest
from trtllm_sbom.core import (
    add_dependencies_from_json,
    add_dependency_from_args,
    generate_outputs,
    register_files_from_args,
    register_files_from_json,
)


def write(tmp: Path, rel: str, content: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_add_register_generate_flow(tmp_path: Path, stub_requests_head):
    data_dir = tmp_path / "scripts/attribution/data"
    lic = write(tmp_path, "LICENSE", "MIT\nCopyright 2024\n")
    f = write(tmp_path, "include/a.h", "// hdr\n")

    add_dependency_from_args(
        data_dir=data_dir,
        dependency="nvtx",
        version="1.0+abcdef0123456789abcdef0123456789abcdef01",
        license_path=lic,
        copyright_path=None,
        attribution_path=None,
        source="https://github.com/NVIDIA/nvtx/commit/abcdef0123456789abcdef0123456789abcdef01",
        overwrite=False,
    )
    register_files_from_args(data_dir, "nvtx", "1.0+abcdef0123456789abcdef0123456789abcdef01", [f])

    ok, missing, attr_text, sbom = generate_outputs(data_dir, [f], dry_run=False)
    assert ok
    assert not missing
    assert "# Software Attributions" in attr_text
    assert "## nvtx" in attr_text
    assert "MIT" in attr_text
    assert sbom.get("components")


def test_warning_on_missing_copyright_in_license(tmp_path: Path, stub_requests_head, capsys):
    data_dir = tmp_path / "scripts/attribution/data"
    lic = write(tmp_path, "LICENSE", "MIT License\n")
    f = write(tmp_path, "inc/a.h", "// hdr\n")

    add_dependency_from_args(
        data_dir=data_dir,
        dependency="lib",
        version="1.0+abcdef0123456789abcdef0123456789abcdef01",
        license_path=lic,
        copyright_path=None,
        attribution_path=None,
        source="https://github.com/NVIDIA/lib/commit/abcdef0123456789abcdef0123456789abcdef01",
        overwrite=False,
    )
    out = capsys.readouterr().out
    assert "Warning:" in out

    # Now registering without SPDX notices should fail and list file
    with pytest.raises(ValueError):
        register_files_from_args(
            data_dir, "lib", "1.0+abcdef0123456789abcdef0123456789abcdef01", [f]
        )


def test_transitive_attributions_are_appended(tmp_path: Path, stub_requests_head):
    data_dir = tmp_path / "scripts/attribution/data"
    lic = write(tmp_path, "LICENSE", "MIT\n")
    attrib = write(tmp_path, "ATTR.txt", "Extra attribution line\nSecond line\n")
    f = write(tmp_path, "inc/a.h", "// SPDX-FileCopyrightText: 2024 Example Co\n")

    add_dependency_from_args(
        data_dir=data_dir,
        dependency="nvtx",
        version="1.0+abcdef0123456789abcdef0123456789abcdef01",
        license_path=lic,
        copyright_path=None,
        attribution_path=attrib,
        source="https://github.com/NVIDIA/nvtx/commit/abcdef0123456789abcdef0123456789abcdef01",
        overwrite=False,
    )
    register_files_from_args(data_dir, "nvtx", "1.0+abcdef0123456789abcdef0123456789abcdef01", [f])
    ok, missing, attr_text, _ = generate_outputs(data_dir, [f], dry_run=False)
    assert ok and not missing
    assert "# Attributions of nvtx" in attr_text
    assert "Extra attribution line" in attr_text


def test_json_add_and_register(tmp_path: Path, stub_requests_head):
    data_dir = tmp_path / "scripts/attribution/data"
    add_json = tmp_path / "add.json"
    lic_path = write(tmp_path, "L.txt", "MIT\nCopyright 2024")
    add_json.write_text(
        json.dumps(
            [
                {
                    "dependency": "pkg",
                    "version": "1.2.3+abcdef0123456789abcdef0123456789abcdef01",
                    "license": str(lic_path),
                    "source": "https://github.com/NVIDIA/pkg/commit/abcdef0123456789abcdef0123456789abcdef01",
                }
            ]
        ),
        encoding="utf-8",
    )
    register_json = tmp_path / "reg.json"
    f = write(tmp_path, "a.h", "// hdr\n")
    register_json.write_text(
        json.dumps(
            [
                {
                    "dependency": "pkg",
                    "version": "1.2.3+abcdef0123456789abcdef0123456789abcdef01",
                    "files": [str(f)],
                }
            ]
        ),
        encoding="utf-8",
    )

    add_dependencies_from_json(data_dir, add_json, overwrite=False)
    register_files_from_json(data_dir, register_json)
    ok, missing, _, _ = generate_outputs(data_dir, [f], dry_run=True)
    assert ok
    assert not missing
