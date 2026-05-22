from pathlib import Path
from textwrap import dedent

import pytest
from trtllm_sbom.models import DependencyMetadataEntry
from trtllm_sbom.storage import DependencyDatabase, _extract_file_copyrights, compute_cas_address


def test_compute_cas_address_stable_and_length():
    a = compute_cas_address(b"hello")
    b = compute_cas_address(b"hello")
    c = compute_cas_address(b"hello!")
    assert a == b
    assert a != c
    # 16-byte blake2b digest as hex → 32 chars
    assert len(a) == 32


def test_cas_put_get_roundtrip(tmp_path: Path):
    db = DependencyDatabase(tmp_path)
    expected = "Test Content\nMulti-line"
    addr = db.cas_put(expected)
    text = db.cas_get(addr)
    assert text == expected


def test_extract_spdx_lines():
    text = dedent("""
        // SPDX-FileCopyrightText: 2024 Example Co
        # SPDX-FileCopyrightText: 2023 Another Co
        /*
         * SPDX-FileCopyrightText: 2022 Third Co
         */
    """)
    lines = _extract_file_copyrights(text)
    expected = {"2024 Example Co", "2023 Another Co", "2022 Third Co"}
    assert set(lines) == expected


def write(tmp: Path, rel: str, content: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_register_requires_metadata(tmp_path: Path):
    db = DependencyDatabase(tmp_path)
    # No metadata yet
    f = write(tmp_path, "include/a.h", "// hdr\n")
    with pytest.raises(ValueError):
        db.register_dependency_files("foo/1.0", [f])


def test_register_and_map_files(tmp_path: Path, stub_requests_head):
    db = DependencyDatabase(tmp_path)
    # Add metadata (license with copyright text present)
    lic_addr = db.cas_put("Copyright 2024\nMIT\n")
    db.add_dependency_metadata(
        "nvtx/1.0+abcdef0123456789abcdef0123456789abcdef01",
        DependencyMetadataEntry(
            license=lic_addr,
            source="https://github.com/NVIDIA/nvtx/commit/abcdef0123456789abcdef0123456789abcdef01",
        ),
        overwrite=False,
    )
    # Create files and register
    f1 = write(tmp_path, "include/a.h", "// hdr\n")
    f2 = write(tmp_path, "include/b.h", "// hdr\n")
    db.register_dependency_files("nvtx/1.0+abcdef0123456789abcdef0123456789abcdef01", [f1, f2])
    # Validate and map
    res = db.validate_all()
    assert res.ok, res.problems
    unmapped, dep_to_meta, dep_to_notices = db.map_files_to_dependencies([f1, f2])
    assert not unmapped
    assert "nvtx/1.0+abcdef0123456789abcdef0123456789abcdef01" in dep_to_meta.root


def test_register_enforces_file_copyright_when_license_missing_notice(
    tmp_path: Path, stub_requests_head
):
    db = DependencyDatabase(tmp_path)
    # License without explicit copyright
    lic_addr = db.cas_put("MIT License\n")
    db.add_dependency_metadata(
        "lib/1.0+abcdef0123456789abcdef0123456789abcdef01",
        DependencyMetadataEntry(
            license=lic_addr,
            source="https://github.com/NVIDIA/lib/commit/abcdef0123456789abcdef0123456789abcdef01",
        ),
        overwrite=False,
    )
    # One file has SPDX notice, one does not
    f1 = write(tmp_path, "src/ok.h", "// SPDX-FileCopyrightText: 2024 OK Co\n")
    f2 = write(tmp_path, "src/missing.h", "// hdr\n")
    with pytest.raises(ValueError):
        db.register_dependency_files("lib/1.0+abcdef0123456789abcdef0123456789abcdef01", [f1, f2])
    # If both have notices, registration succeeds
    f2.write_text("// SPDX-FileCopyrightText: 2023 OK Co\n", encoding="utf-8")
    db.register_dependency_files("lib/1.0+abcdef0123456789abcdef0123456789abcdef01", [f1, f2])


def test_validate_all_detects_bad_cas(tmp_path: Path, stub_requests_head):
    db = DependencyDatabase(tmp_path)
    lic_addr = db.cas_put("MIT\n")
    db.add_dependency_metadata(
        "x/1.0+abcdef0123456789abcdef0123456789abcdef01",
        DependencyMetadataEntry(
            license=lic_addr,
            source="https://github.com/NVIDIA/x/commit/abcdef0123456789abcdef0123456789abcdef01",
        ),
        overwrite=False,
    )
    # Corrupt the CAS file
    shard = lic_addr[0:2]
    cas_file = tmp_path / "cas" / shard / lic_addr
    cas_file.write_text("CORRUPTED\n", encoding="utf-8")
    res = db.validate_all()
    assert not res.ok
    assert any("CAS mismatch" in p for p in res.problems)


def test_nv_only_notice_is_mapped(tmp_path: Path, stub_requests_head):
    """Test that files with only NVIDIA notices are still mapped to their dependency."""
    db = DependencyDatabase(tmp_path)
    lic_addr = db.cas_put("MIT\n")
    db.add_dependency_metadata(
        "nlib/1.0+abcdef0123456789abcdef0123456789abcdef01",
        DependencyMetadataEntry(
            license=lic_addr,
            source="https://github.com/NVIDIA/nlib/commit/abcdef0123456789abcdef0123456789abcdef01",
        ),
        overwrite=False,
    )
    f = write(
        tmp_path,
        "src/nv.h",
        "// (c) 2024 NVIDIA Corporation\n// SPDX-FileCopyrightText: 2024 NVIDIA Corporation\n",
    )
    db.register_dependency_files("nlib/1.0+abcdef0123456789abcdef0123456789abcdef01", [f])
    # The file checksum should be present in files_to_dependency mapping
    cs = compute_cas_address(f.read_bytes())
    assert cs in db.f2d_raw.get("nlib/1.0+abcdef0123456789abcdef0123456789abcdef01", [])
    # The file should be mapped to its dependency
    unmapped, dep_to_meta, dep_to_notices = db.map_files_to_dependencies([f])
    assert not unmapped
    assert "nlib/1.0+abcdef0123456789abcdef0123456789abcdef01" in dep_to_meta.root


def test_mixed_nv_notice_is_attributed(tmp_path: Path, stub_requests_head):
    db = DependencyDatabase(tmp_path)
    lic_addr = db.cas_put("MIT\n")
    db.add_dependency_metadata(
        "plib/1.0+abcdef0123456789abcdef0123456789abcdef01",
        DependencyMetadataEntry(
            license=lic_addr,
            source="https://github.com/NVIDIA/plib/commit/abcdef0123456789abcdef0123456789abcdef01",
        ),
        overwrite=False,
    )
    f = write(
        tmp_path,
        "src/ok.h",
        "// SPDX-FileCopyrightText: 2024 Example Co\n// SPDX-FileCopyrightText: 2024 NVIDIA Corporation\n",
    )
    db.register_dependency_files("plib/1.0+abcdef0123456789abcdef0123456789abcdef01", [f])
    cs = compute_cas_address(f.read_bytes())
    # The checksum should be registered
    assert cs in db.f2d_raw.get("plib/1.0+abcdef0123456789abcdef0123456789abcdef01", [])
    unmapped, dep_to_meta, dep_to_notices = db.map_files_to_dependencies([f])
    assert not unmapped
    assert "plib/1.0+abcdef0123456789abcdef0123456789abcdef01" in dep_to_meta.root
    assert dep_to_notices.get("plib/1.0+abcdef0123456789abcdef0123456789abcdef01")
