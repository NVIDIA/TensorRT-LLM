import os as _os
import pathlib as _pl
import sys as _sys

import pytest

import tensorrt_llm.bindings as _tb
from tensorrt_llm.bindings.internal.testing import ModelSpec

_sys.path.append(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))


@pytest.fixture(scope="module")
def llm_root() -> _pl.Path:
    environ_root = _os.environ.get("LLM_ROOT", None)
    return _pl.Path(environ_root) if environ_root is not None else _pl.Path(
        __file__).resolve().parent.parent.parent.parent


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


def get_base_model_spec() -> ModelSpec:
    model_spec_obj = ModelSpec('input_tokens.npy', _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin().set_kv_cache_type(
        _tb.KVCacheType.PAGED).use_packed_input()
    return model_spec_obj


@pytest.fixture(scope="module")
def model_path(engine_path):
    return engine_path / f"gpt2/{get_base_model_spec().get_model_path()}/tp1-pp1-cp1-gpu"


@pytest.fixture(scope="module")
def model_path_return_logits(engine_path):
    return engine_path / f"gpt2/{get_base_model_spec().gather_logits().get_model_path()}/tp1-pp1-cp1-gpu"


@pytest.fixture
def model_path_lora(engine_path: _pl.Path) -> _pl.Path:
    return engine_path / f"gpt2/{get_base_model_spec().use_lora_plugin().get_model_path()}/tp1-pp1-cp1-gpu"


@pytest.fixture
def model_path_draft_tokens_external(engine_path: _pl.Path) -> _pl.Path:
    return engine_path / f"gpt2/{get_base_model_spec().use_draft_tokens_external_decoding().get_model_path()}/tp1-pp1-cp1-gpu"


@pytest.fixture
def lora_config_path(data_path: _pl.Path) -> _pl.Path:
    return data_path / "lora-test-weights-gpt2-tp1"


@pytest.fixture(scope="module")
def results_data_path(data_path: _pl.Path) -> _pl.Path:
    return data_path / f"gpt2/sampling/{get_base_model_spec().get_results_file()}"


@pytest.fixture(scope="module")
def results_data_path_beam_width_2(data_path: _pl.Path) -> _pl.Path:
    return data_path / f"gpt2/beam_search_2/{get_base_model_spec().get_results_file()}"


@pytest.fixture(scope="module")
def results_data_path_fmhafp32acc(data_path: _pl.Path) -> _pl.Path:
    return data_path / f"gpt2/sampling/{get_base_model_spec().enable_context_fmha_fp32_acc().get_results_file()}"
