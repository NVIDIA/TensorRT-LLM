import pickle

import numpy as np
import pytest
import torch

from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


def _ids(n=1000):
    return np.arange(n, dtype=np.int32) * 7 % 129000


@pytest.mark.parametrize("as_tensor", [False, True])
def test_flat_array_prompt_is_kept_as_int32_buffer(as_tensor):
    ids = _ids()
    prompt = torch.from_numpy(ids.copy()) if as_tensor else ids
    req = GenerationRequest(prompt_token_ids=prompt, sampling_params=SamplingParams(max_tokens=1))
    assert req.__dict__["_prompt_token_ids"] is None
    buf = req.__dict__["_prompt_token_ids_i32"]
    assert buf.dtype == np.int32 and buf.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(buf, ids)


def test_flat_array_prompt_round_trips_through_pickle_without_a_list():
    ids = _ids()
    req = GenerationRequest(prompt_token_ids=ids, sampling_params=SamplingParams(max_tokens=1))
    state = req.__getstate__()
    tag, raw = state["_prompt_token_ids"]
    assert tag == GenerationRequest._I32
    assert raw == ids.tobytes()
    clone = pickle.loads(pickle.dumps(req))
    assert clone.__dict__["_prompt_token_ids"] is None
    np.testing.assert_array_equal(clone.__dict__["_prompt_token_ids_i32"], ids)
    assert clone.prompt_token_ids == ids.tolist()


def test_array_prompt_is_snapshotted_at_construction():
    ids = _ids()
    req = GenerationRequest(prompt_token_ids=ids, sampling_params=SamplingParams(max_tokens=1))
    ids[:] = 0
    np.testing.assert_array_equal(req.__dict__["_prompt_token_ids_i32"], _ids())
    assert req.prompt_token_ids == _ids().tolist()


def test_list_and_2d_array_prompts_are_unchanged():
    req = GenerationRequest(
        prompt_token_ids=[3, 4, 5], sampling_params=SamplingParams(max_tokens=1)
    )
    assert req.prompt_token_ids == [3, 4, 5]
    two_d = np.array([[1, 2], [3, 4]], dtype=np.int32)
    req = GenerationRequest(prompt_token_ids=two_d, sampling_params=SamplingParams(max_tokens=1))
    assert req.prompt_token_ids == [[1, 2], [3, 4]]


def test_replacing_the_prompt_drops_the_int32_buffer():
    req = GenerationRequest(prompt_token_ids=_ids(), sampling_params=SamplingParams(max_tokens=1))
    req.prompt_token_ids = [6, 7, 8]
    assert req.__dict__["_prompt_token_ids_i32"] is None
    assert req.prompt_token_ids == [6, 7, 8]
    clone = pickle.loads(pickle.dumps(req))
    assert clone.prompt_token_ids == [6, 7, 8]
    np.testing.assert_array_equal(
        clone.__dict__["_prompt_token_ids_i32"], np.array([6, 7, 8], dtype=np.int32)
    )
