# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def _requirement(path: Path, name: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value.startswith(name):
            return value
    raise AssertionError(f"{name} is not declared in {path}")


def test_openengine_video_dependency_excludes_numpy_incompatible_opencv() -> None:
    root = Path(__file__).resolve().parents[3]
    numpy = _requirement(root / "requirements.txt", "numpy")
    opencv = _requirement(root / "requirements-openengine.txt", "opencv-python-headless")

    assert numpy == "numpy>=2.0.0,<2.4"
    assert opencv == "opencv-python-headless==4.11.0.86"
