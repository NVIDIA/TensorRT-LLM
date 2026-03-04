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

    KimiK2Tokenizer.__module__ = "transformers_modules.kimi_k2.tokenization_kimi"
    fake_tm.KimiK2Tokenizer = KimiK2Tokenizer

    # Rank-0: dynamic module present, load tokenizer via the user-facing API.
    sys.modules["transformers_modules"] = fake_tm
    try:
        with mock.patch(
            "tensorrt_llm.tokenizer.tokenizer.AutoTokenizer.from_pretrained",
            return_value=KimiK2Tokenizer(),
        ):
            tok = load_hf_tokenizer("Kimi-K2-Instruct", trust_remote_code=True)
        assert tok is not None
        data = pickle.dumps(tok)
    finally:
        sys.modules.pop("transformers_modules", None)

    # Worker node: module is gone, but cloudpickle embedded the class definition.
    restored = pickle.loads(data)  # nosec B301
    assert isinstance(restored, TransformersTokenizer)
    assert restored.eos_token_id == 151643
