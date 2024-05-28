import os as _os
import pathlib as _pl

import pytest


@pytest.fixture(scope="module")
def llm_root() -> _pl.Path:
    environ_root = _os.environ.get("LLM_ROOT", None)
    return _pl.Path(environ_root) if environ_root is not None else _pl.Path(
        __file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def resource_path(llm_root: _pl.Path) -> _pl.Path:
    return llm_root / "cpp" / "tests" / "resources"


@pytest.fixture(scope="module")
def data_path(resource_path: _pl.Path) -> _pl.Path:
    return resource_path / "data"


@pytest.fixture(scope="module")
def input_data_path(data_path):
    return data_path / "input_tokens.npy"


@pytest.fixture(scope="module")
def engine_path(resource_path: _pl.Path) -> _pl.Path:
    return resource_path / "models" / "rt_engine"


@pytest.fixture(scope="module")
def model_path(engine_path):
    return engine_path / "gpt2/fp16-plugin-packed-paged/tp1-pp1-gpu"


@pytest.fixture(scope="module")
def model_path_return_logits(engine_path):
    return engine_path / "gpt2/fp16-plugin-packed-paged-gather/tp1-pp1-gpu"


@pytest.fixture(scope="module")
def results_data_path(data_path: _pl.Path) -> _pl.Path:
    return data_path / "gpt2/sampling/output_tokens_fp16_plugin_packed_paged_tp1_pp1.npy"


@pytest.fixture(scope="module")
def results_data_path_beam_width_2(data_path: _pl.Path) -> _pl.Path:
    return data_path / "gpt2/beam_search_2/output_tokens_fp16_plugin_packed_paged_tp1_pp1.npy"
