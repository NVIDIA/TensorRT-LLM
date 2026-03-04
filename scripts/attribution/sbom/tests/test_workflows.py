import json
from pathlib import Path

import yaml
from click.testing import CliRunner
from trtllm_sbom.cli import main


def _write_files(tmp: Path) -> list[Path]:
    paths = [
        tmp / "include" / "a.h",
        tmp / "include" / "b.h",
    ]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"// header {p.name}\n", encoding="utf-8")
    # add SPDX copyright line to one file
    (tmp / "include" / "a.h").write_text(
        "// SPDX-FileCopyrightText: 2024 Example Corp\n", encoding="utf-8"
    )
    return paths


def test_workflow_add_update(tmp_path: Path, stub_requests_head):
    runner = CliRunner()
    data_dir = tmp_path / "scripts/attribution/data"
    files = _write_files(tmp_path)

    # Step 2 (dry-run generate) before registering should list missing files (non-zero exit)
    missing_path = tmp_path / "missing.json"
    res = runner.invoke(
        main,
        [
            "generate",
            "--data-dir",
            str(data_dir),
            "--dry-run",
            "--missing-files",
            str(missing_path),
            *map(str, files),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code != 0
    # Missing file list should be populated
    assert missing_path.exists()
    missing_data = json.loads(missing_path.read_text(encoding="utf-8"))
    assert sorted(missing_data) == sorted([str(p) for p in files])

    # Step 4: add dependency
    add_json = tmp_path / "add.json"
    lic_path = tmp_path / "L.txt"
    lic_path.write_text("MIT\nCopyright (c) 2024 Example", encoding="utf-8")
    add_json.write_text(
        json.dumps(
            [
                {
                    "dependency": "tensorrt",
                    "version": "10.14.0.26+abcdef0123456789abcdef0123456789abcdef01",
                    "license": str(lic_path),
                    "source": "https://github.com/NVIDIA/TensorRT/commit/abcdef0123456789abcdef0123456789abcdef01",
                }
            ]
        ),
        encoding="utf-8",
    )
    res = runner.invoke(
        main, ["import", "--data-dir", str(data_dir), "-j", str(add_json)], catch_exceptions=False
    )
    assert res.exit_code == 0, res.output

    # Step 5: map files to dependency
    res = runner.invoke(
        main,
        [
            "map-files",
            "--data-dir",
            str(data_dir),
            "-d",
            "tensorrt",
            "-v",
            "10.14.0.26+abcdef0123456789abcdef0123456789abcdef01",
            *map(str, files),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    # Step 6: validate
    res = runner.invoke(main, ["validate", "--data-dir", str(data_dir)], catch_exceptions=False)
    assert res.exit_code == 0, res.output

    # Step 7: dry-run should succeed now and write empty missing list
    res = runner.invoke(
        main,
        [
            "generate",
            "--data-dir",
            str(data_dir),
            "--dry-run",
            "--missing-files",
            str(missing_path),
            *map(str, files),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    missing_data = json.loads(missing_path.read_text(encoding="utf-8"))
    assert missing_data == []

    # Update workflow: overwrite existing dependency
    add_json2 = tmp_path / "add2.json"
    lic_path2 = tmp_path / "L2.txt"
    lic_path2.write_text("BSD-3-Clause\nCopyright (c) 2024 Example", encoding="utf-8")
    add_json2.write_text(
        json.dumps(
            [
                {
                    "dependency": "tensorrt",
                    "version": "10.14.0.26+abcdef0123456789abcdef0123456789abcdef01",
                    "license": str(lic_path2),
                    "source": "https://github.com/NVIDIA/TensorRT/commit/abcdef0123456789abcdef0123456789abcdef01",
                }
            ]
        ),
        encoding="utf-8",
    )
    res = runner.invoke(
        main,
        ["import", "--data-dir", str(data_dir), "--overwrite", "-j", str(add_json2)],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    # Check metadata updated
    meta_path = data_dir / "dependency_metadata.yml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert "tensorrt/10.14.0.26+abcdef0123456789abcdef0123456789abcdef01" in meta
    assert meta["tensorrt/10.14.0.26+abcdef0123456789abcdef0123456789abcdef01"].get("license")


def test_workflow_generate_outputs(tmp_path: Path, stub_requests_head):
    runner = CliRunner()
    data_dir = tmp_path / "scripts/attribution/data"
    files = _write_files(tmp_path)

    # Add + register
    lic_path = tmp_path / "LICENSE"
    lic_path.write_text("MIT\nCopyright 2024", encoding="utf-8")
    res = runner.invoke(
        main,
        [
            "import",
            "--data-dir",
            str(data_dir),
            "-d",
            "tensorrt",
            "-v",
            "10.14.0.26+abcdef0123456789abcdef0123456789abcdef01",
            "-s",
            "https://github.com/NVIDIA/TensorRT/commit/abcdef0123456789abcdef0123456789abcdef01",
            "-l",
            str(lic_path),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    res = runner.invoke(
        main,
        [
            "map-files",
            "--data-dir",
            str(data_dir),
            "-d",
            "tensorrt",
            "-v",
            "10.14.0.26+abcdef0123456789abcdef0123456789abcdef01",
            *map(str, files),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    # Generate outputs
    attr_out = tmp_path / "attr.txt"
    sbom_out = tmp_path / "sbom.json"
    res = runner.invoke(
        main,
        [
            "generate",
            "--data-dir",
            str(data_dir),
            "-a",
            str(attr_out),
            "-s",
            str(sbom_out),
            *map(str, files),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    assert attr_out.exists()
    assert sbom_out.exists()
    text = attr_out.read_text(encoding="utf-8")
    assert "# Software Attributions" in text
    assert "## tensorrt" in text
    assert "### License Text" in text
    assert "MIT" in text
