# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage of stage-KV cubin archive generation."""

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def stage_kv_generator() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/trtllm_gen_fmha_stage_kv/gen_stage_kv_cubins.py"
    )
    specification = importlib.util.spec_from_file_location("stage_kv_generator", script)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stem", ["StageKv", "StageKv" + "LongName" * 16], ids=["short", "pax"])
def test_archive_ownership_and_reproducibility(
    stage_kv_generator: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stem: str
) -> None:
    payload = b"neutral cubin payload"
    archives = []
    listings = []
    for directory_name in ("first", "second"):
        destination = tmp_path / directory_name
        destination.mkdir()
        monkeypatch.setattr(stage_kv_generator, "CUBIN_DIR", destination)
        stage_kv_generator.write_archive(stem, payload)
        archive = destination / f"{stem}.cubin.tar.zst"
        archives.append(archive.read_bytes())
        listings.append(
            subprocess.check_output(
                ["cmake", "-E", "tar", "tvf", str(archive)], text=True, timeout=30
            )
        )
        extracted = destination / "extracted"
        extracted.mkdir()
        subprocess.run(
            ["cmake", "-E", "tar", "xf", str(archive)], cwd=extracted, check=True, timeout=30
        )
        members = list(extracted.iterdir())
        assert len(members) == 1 and members[0].name == f"{stem}.cubin"
        assert members[0].read_bytes() == payload

    assert archives[0] == archives[1]
    for listing in listings:
        fields = listing.split()
        assert fields[0] == "-rw-r--r--"
        assert fields[2:4] == ["root", "nogroup"]
        assert int(fields[4]) == len(payload)
        assert fields[-1] == f"{stem}.cubin"
