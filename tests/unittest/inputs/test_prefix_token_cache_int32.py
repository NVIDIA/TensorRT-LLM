# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PrefixTokenCache array splice (P1) and TokenIdList int32 reply (P2): exact versus the list path."""

import copy
import importlib
import pickle
import random
import re
from array import array

import numpy as np
import pytest

import tensorrt_llm.inputs.prefix_token_cache as ptc


class _Tok:
    """Word / space tokenizer with offsets; ids are a stable function of the piece."""

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        ids, offs = [], []
        for m in re.finditer(r"\S+|\s+", text):
            ids.append(sum(map(ord, m.group())) * 7919 % 200000)
            offs.append((m.start(), m.end()))
        return {"input_ids": ids, "offset_mapping": offs}


def _cache(mod):
    return mod.PrefixTokenCache(mod.PrefixTokenCacheConfig(min_chars=64, overlap=8, resync=4))


@pytest.fixture
def modes(monkeypatch):
    def load(splice, int32):
        monkeypatch.setenv("TRTLLM_M31_DISABLE_PTC_ARRAY_SPLICE", "0" if splice else "1")
        monkeypatch.setenv("TRTLLM_M31_DISABLE_PTC_INT32_REPLY", "0" if int32 else "1")
        return importlib.reload(ptc)

    yield load
    monkeypatch.delenv("TRTLLM_M31_DISABLE_PTC_ARRAY_SPLICE", raising=False)
    monkeypatch.delenv("TRTLLM_M31_DISABLE_PTC_INT32_REPLY", raising=False)
    importlib.reload(ptc)


def _run(mod, seed, turns=40):
    random.seed(seed)
    words = ["alpha", "be", "gam", "delta", "e", "zeta-x", "eta.", "th"]
    convs = [" ".join(random.choice(words) for _ in range(200)) for _ in range(4)]
    cache, tok, out = _cache(mod), _Tok(), []
    for _ in range(turns):
        i = random.randrange(4)
        convs[i] += " " + " ".join(random.choice(words) for _ in range(random.randrange(1, 60)))
        out.append(cache.encode(tok, convs[i]))
    state = sorted(
        (e.text, e.ids.tolist(), e.split_token, e.split_char) for e in cache._entries.values()
    )
    return out, state, cache


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_splice_and_int32_match_list_path(modes, seed):
    old_ids, old_state, _ = _run(modes(False, False), seed)
    new_ids, new_state, cache = _run(modes(True, True), seed)
    assert [list(x) for x in new_ids] == old_ids
    assert new_state == old_state
    assert cache.hits > 0
    tl = [x for x in new_ids if isinstance(x, ptc.TokenIdList)]
    assert len(tl) == cache.hits
    for ids in new_ids:
        assert ptc.int32_bytes(ids) == np.asarray(list(ids), dtype=np.int32).tobytes()


@pytest.mark.parametrize("splice,int32", [(True, False), (False, True)])
def test_kill_switches(modes, splice, int32):
    ids, _, cache = _run(modes(splice, int32), 0)
    assert cache.hits > 0
    assert not any(isinstance(x, ptc.TokenIdList) for x in ids)


def test_token_id_list_is_a_plain_list_across_pickle_and_copy(modes):
    mod = modes(True, True)
    ids = mod.TokenIdList([1, 2, 3, 200063])
    ids.int32 = array("i", ids)
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        back = pickle.loads(pickle.dumps(ids, protocol=proto))
        assert type(back) is list and back == [1, 2, 3, 200063]
    assert type(copy.copy(ids)) is list and type(copy.deepcopy(ids)) is list
    assert ids == [1, 2, 3, 200063] and len(ids) == 4


@pytest.mark.parametrize(
    "edit",
    [
        lambda ids: ids.__setitem__(500, 7),  # mid-sequence edit, length and both ends unchanged
        lambda ids: ids.__setitem__(slice(400, 402), [9, 9]),
        lambda ids: ids.__delitem__(500),
        lambda ids: ids.__iadd__([1]),
        lambda ids: ids.__imul__(2),
        lambda ids: ids.append(9),
        lambda ids: ids.extend([9, 9]),
        lambda ids: ids.insert(500, 9),
        lambda ids: ids.pop(500),
        lambda ids: ids.remove(500),
        lambda ids: ids.reverse(),
        lambda ids: ids.sort(reverse=True),
        lambda ids: ids.clear(),
    ],
)
def test_any_mutation_drops_the_int32_array(modes, edit):
    mod = modes(True, True)
    ids = mod.TokenIdList(list(range(1000)))
    ids.int32 = array("i", ids)
    assert mod.int32_bytes(ids) == ids.int32.tobytes()
    edit(ids)
    assert ids.int32 is None
    assert mod.int32_bytes(ids) == np.asarray(list(ids), dtype=np.int32).tobytes()
