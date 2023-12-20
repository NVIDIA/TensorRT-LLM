import logging as _log
import os as _os
import pathlib as _pl
import subprocess as _sp
import sys as _sys
import typing as _tp

import numpy as _np
import pytest


@pytest.fixture(scope="module")
def llm_root() -> _pl.Path:
    environ_root = _os.environ.get("LLM_ROOT", None)
    return _pl.Path(environ_root) if environ_root is not None else _pl.Path(
        __file__).parent.parent.parent


@pytest.fixture(scope="module")
def llm_model_root() -> _pl.Path | None:
    return _os.environ.get("LLM_MODEL_ROOT", None)


@pytest.fixture(scope="module")
def resource_path(llm_root: _pl.Path) -> _pl.Path:
    return llm_root / "cpp" / "tests" / "resources"


@pytest.fixture(scope="module")
def engine_path(resource_path: _pl.Path) -> _pl.Path:
    return resource_path / "models" / "rt_engine"


@pytest.fixture(scope="module")
def data_path(resource_path: _pl.Path) -> _pl.Path:
    return resource_path / "data"


def run_command(command: _tp.Sequence[str],
                cwd: _pl.Path,
                *,
                shell=False,
                env=None) -> None:
    _log.info("Running: cd %s && %s", str(cwd), " ".join(command))
    _sp.check_call(command, cwd=cwd, shell=shell, env=env)


def prepare_model_tests(
    llm_root: _pl.Path,
    resource_path: _pl.Path,
    model_name: str,
    model_cache_arg=[],
):
    scripts_dir = resource_path / "scripts"
    python_exe = _sys.executable
    model_env = {**_os.environ, "PYTHONPATH": f"examples/{model_name}"}
    build_engines = [
        python_exe,
        str(scripts_dir / f"build_{model_name}_engines.py")
    ] + model_cache_arg
    run_command(build_engines, cwd=llm_root, env=model_env)

    model_env["PYTHONPATH"] = "examples"
    generate_expected_output = [
        python_exe,
        str(scripts_dir / f"generate_expected_{model_name}_output.py")
    ]
    run_command(generate_expected_output, cwd=llm_root, env=model_env)


def sequence_lengths(sequences: _np.ndarray, pad_id: int) -> _np.ndarray:
    return _np.apply_along_axis(lambda x: _np.searchsorted(x, True), 1,
                                sequences == pad_id).astype("int32")
