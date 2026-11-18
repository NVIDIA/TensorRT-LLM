# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

import tensorrt_llm

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("family_launcher", [False, True])
def test_rank_job_does_not_shadow_installed_package(tmp_path, monkeypatch, family_launcher):
    source_root = tmp_path / "source"
    comm = source_root / "tests/unittest/_torch/modeling_v2/comm"
    comm.mkdir(parents=True)
    unbuilt_package = source_root / "tensorrt_llm"
    unbuilt_package.mkdir()
    (unbuilt_package / "__init__.py").touch()
    rank_job_path = comm / "_rank_job.py"
    rank_job_path.write_bytes(Path(__file__).with_name("_rank_job.py").read_bytes())
    installed_package = Path(tensorrt_llm.__file__).resolve()
    monkeypatch.setenv("PYTHONPATH", str(installed_package.parent.parent))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("TRT_LLM_NO_LIB_INIT", "1")
    monkeypatch.setenv("MPI4PY_RC_INITIALIZE", "0")
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    result = tmp_path / "package-path.txt"
    launcher_dir = comm.parent / "models/example_family" if family_launcher else comm
    launcher_dir.mkdir(parents=True, exist_ok=True)
    (launcher_dir / "_package_selection_op_matrix.py").write_text(
        "from pathlib import Path\n"
        "DEADLINE_S = 30\n"
        "if __name__ == '__main__':\n"
        "    import tensorrt_llm\n"
        "    import tensorrt_llm.bindings\n"
        f"    Path({str(result)!r}).write_text(tensorrt_llm.__file__)\n"
    )
    spec = importlib.util.spec_from_file_location("_rank_job_package_selection", rank_job_path)
    rank_job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rank_job)
    rank_job.run("package_selection", directory=launcher_dir if family_launcher else None)
    assert Path(result.read_text()).resolve() == installed_package
