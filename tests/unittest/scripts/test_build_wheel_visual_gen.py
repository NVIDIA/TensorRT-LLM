# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only


def test_install_visual_gen_libs(tmp_path):
    script = Path(__file__).resolve().parents[3] / "scripts/build_wheel.py"
    spec = importlib.util.spec_from_file_location("build_wheel", script)
    build_wheel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_wheel)

    build, libs = tmp_path / "build", tmp_path / "libs"
    build.mkdir()
    libs.mkdir()
    manifest = "visual_gen_native.json"
    catalog = "libth_visual_gen_catalog.so"
    family = "libth_visual_gen_example.so"
    entries = {
        "catalog/activation": {"library": catalog, "namespace": "vg_catalog"},
        "models/example": {"library": family, "namespace": "vg_example"},
        "models/example/fast": {"library": family, "namespace": "vg_example_fast"},
    }
    (build / manifest).write_text(json.dumps(entries))
    (build / catalog).write_bytes(b"catalog")
    (libs / "libth_common.so").write_bytes(b"stock")
    (libs / "libth_visual_gen_removed.so").symlink_to(tmp_path / "gone")
    install_file = Mock(wraps=shutil.copy2)

    with pytest.raises(FileNotFoundError, match=family):
        build_wheel.install_visual_gen_libs(build, libs, install_file)
    install_file.assert_not_called()

    (build / family).write_bytes(b"family")
    build_wheel.install_visual_gen_libs(build, libs, install_file)
    assert [args.args[0].name for args in install_file.call_args_list].count(family) == 1
    assert {p.name for p in libs.iterdir()} == {manifest, catalog, family, "libth_common.so"}
    assert json.loads((libs / manifest).read_text()) == entries
    assert (libs / catalog).read_bytes() == b"catalog"
    assert (libs / family).read_bytes() == b"family"

    (build / manifest).write_text("{}")
    build_wheel.install_visual_gen_libs(build, libs, install_file)
    assert {p.name for p in libs.iterdir()} == {manifest, "libth_common.so"}
    assert json.loads((libs / manifest).read_text()) == {}
