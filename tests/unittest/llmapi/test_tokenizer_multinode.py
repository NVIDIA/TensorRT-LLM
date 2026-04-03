"""RCCA test for nvbugs/5823783 — multi-node hang with --trust_remote_code."""

import pickle  # nosec B403
import sys
import types
from unittest import mock

import pytest

from tensorrt_llm.tokenizer.tokenizer import TransformersTokenizer, load_hf_tokenizer


def test_trust_remote_code_tokenizer_pickle_roundtrip_multinode():
    """nvbugs/5823783 regression.

    Loading a tokenizer via load_hf_tokenizer with trust_remote_code=True must
    produce a tokenizer that survives pickle/unpickle after the dynamic
    transformers_modules are removed from sys.modules, simulating what happens
    on a worker node in a multi-node MPI setup.
    """
    # RCCA: nvbugs/5823783: multi-node hang when --trust_remote_code is used (e.g. Kimi-K2-Instruct)
    pytest.importorskip("cloudpickle")

    # Simulate the dynamic module that AutoTokenizer creates with trust_remote_code=True
    fake_tm = types.ModuleType("transformers_modules")

    class KimiK2Tokenizer:
        eos_token_id = 151643
        all_special_tokens = []

    # Put the class in a proper submodule so standard pickle can serialize it by
    # reference on "rank-0" (where the module exists), exactly as HF does when
    # trust_remote_code downloads custom tokenizer code.
    # Both __module__ and __qualname__ must be set: pickle checks __qualname__
    # first and refuses to serialize if it sees "<locals>" in the name.
    fake_sub = types.ModuleType("transformers_modules.kimi_k2.tokenization_kimi")
    KimiK2Tokenizer.__module__ = "transformers_modules.kimi_k2.tokenization_kimi"
    KimiK2Tokenizer.__qualname__ = "KimiK2Tokenizer"
    fake_sub.KimiK2Tokenizer = KimiK2Tokenizer
    fake_tm.KimiK2Tokenizer = KimiK2Tokenizer

    # Rank-0: both modules present — standard pickle would succeed here but
    # serialize the class by reference (module path only, no class definition).
    # Save prior values so the finally block can restore them instead of
    # unconditionally deleting, avoiding cross-test sys.modules contamination.
    _SENTINEL = object()
    prev_tm = sys.modules.get("transformers_modules", _SENTINEL)
    prev_sub = sys.modules.get("transformers_modules.kimi_k2.tokenization_kimi", _SENTINEL)
    sys.modules["transformers_modules"] = fake_tm
    sys.modules["transformers_modules.kimi_k2.tokenization_kimi"] = fake_sub
    try:
        with mock.patch(
            "tensorrt_llm.tokenizer.tokenizer.AutoTokenizer.from_pretrained",
            return_value=KimiK2Tokenizer(),
        ):
            tok = load_hf_tokenizer("Kimi-K2-Instruct", trust_remote_code=True)
        assert tok is not None
        data = pickle.dumps(tok)
    finally:
        if prev_tm is _SENTINEL:
            sys.modules.pop("transformers_modules", None)
        else:
            sys.modules["transformers_modules"] = prev_tm
        if prev_sub is _SENTINEL:
            sys.modules.pop("transformers_modules.kimi_k2.tokenization_kimi", None)
        else:
            sys.modules["transformers_modules.kimi_k2.tokenization_kimi"] = prev_sub

    # Worker node: module is gone, but cloudpickle embedded the class definition.
    restored = pickle.loads(data)  # nosec B301
    assert isinstance(restored, TransformersTokenizer)
    assert restored.eos_token_id == 151643
