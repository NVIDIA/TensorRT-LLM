"""fold_save_last_enabled: default on, env off-switch, forced on, graceful fallback."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.modules.mamba import fold_support
from tensorrt_llm._torch.pyexecutor.kv_cache import mamba_cache_manager as mcm


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("TLLM_MAMBA_FOLD_SAVE_LAST", raising=False)
    monkeypatch.setattr(fold_support, "_UNSUPPORTED", [])
    monkeypatch.setattr(mcm, "_fold_save_last_platform_reason", lambda: None)
    monkeypatch.setattr(mcm, "_fold_disabled_warned", set())


def _manager(spec_config=None):
    return SimpleNamespace(spec_config=spec_config)


def test_default_on_when_supported():
    assert mcm.fold_save_last_enabled(_manager()) is True
    assert mcm.fold_save_last_enabled() is True


def test_env_zero_turns_it_off(monkeypatch):
    monkeypatch.setenv("TLLM_MAMBA_FOLD_SAVE_LAST", "0")
    assert mcm.fold_save_last_enabled(_manager()) is False


def test_mamba2_layers_fall_back_by_default_and_refuse_when_forced(monkeypatch):
    fold_support.mark_fold_unsupported("the model has Mamba2 mixer layers")
    assert mcm.fold_save_last_enabled(_manager()) is False
    monkeypatch.setenv("TLLM_MAMBA_FOLD_SAVE_LAST", "1")
    with pytest.raises(RuntimeError, match="Mamba2"):
        mcm.fold_save_last_enabled(_manager())


def test_speculative_decoding_falls_back(monkeypatch):
    assert mcm.fold_save_last_enabled(_manager(spec_config=object())) is False
    monkeypatch.setenv("TLLM_MAMBA_FOLD_SAVE_LAST", "1")
    with pytest.raises(RuntimeError, match="speculative"):
        mcm.fold_save_last_enabled(_manager(spec_config=object()))


def test_platform_reason_falls_back(monkeypatch):
    monkeypatch.setattr(mcm, "_fold_save_last_platform_reason", lambda: "no FlashInfer GDN kernels")
    assert mcm.fold_save_last_enabled(_manager()) is False
    monkeypatch.setenv("TLLM_MAMBA_FOLD_SAVE_LAST", "1")
    with pytest.raises(RuntimeError, match="FlashInfer"):
        mcm.fold_save_last_enabled(_manager())
