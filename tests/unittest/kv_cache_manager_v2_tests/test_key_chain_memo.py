# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Block-key chain memo (TLLM_KV_CACHE_MANAGER_V2_KEY_CHAIN_MEMO) must not change any match.

The switch is read once per process, so one deterministic scenario runs in two child processes,
with the memo off and on, and the transcripts of every probe and cache creation must be equal.
The scenario commits conversations that share a system prefix, then queries prefixes, extensions,
mid-block divergences, other reuse scopes, partial last blocks and int32 inputs while the tree
keeps changing (new commits) between queries, so memo entries outlive the tree states they were
recorded in.
"""

import json
import os
import subprocess
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV = "TLLM_KV_CACHE_MANAGER_V2_KEY_CHAIN_MEMO"


def _scenario() -> list:
    sys.path.insert(0, _TEST_DIR)
    import random

    import numpy as np
    from cuda_test_utils import TemporaryCudaStream, init_cuda_once
    from test_kv_cache_manager_v2 import create_config

    from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager, ReuseScope, TokenId

    init_cuda_once()
    tpb = 4
    manager = KVCacheManager(
        create_config(
            tokens_per_block=tpb,
            gpu_quota=64 << 20,
            host_quota=0,
            disk_quota=0,
            num_layers=2,
            window_size=None,
            sink_tokens=0,
        )
    )
    rng = random.Random(7)
    transcript = []

    def tok(seq):
        return [TokenId(t) for t in seq]

    def commit(seq, scope):
        with TemporaryCudaStream([]) as holder:
            kv = manager.create_kv_cache(scope, tok(seq))
            assert kv.resume(holder.handle)
            assert kv.resize(len(seq))
            done = kv.num_committed_tokens
            if done < len(seq):
                kv.commit(tok(seq[done:]))
            kv.close()
        holder.take_finish_event().synchronize()

    def query(seq, scope, as_array):
        data = np.asarray(seq, dtype=np.int32) if as_array else tok(seq)
        reuse = manager.probe_reuse(scope, data)
        key = manager.probe_first_new_block_key(scope, data)
        kv = manager.create_kv_cache(scope, data)
        record = [
            len(seq),
            reuse,
            None if key is None else bytes(key).hex(),
            kv.num_committed_tokens,
            list(kv.cached_tokens_by_level),
        ]
        kv.close()
        transcript.append(record)

    scopes = [ReuseScope(), ReuseScope(salt=11), ReuseScope(lora_id=3)]
    system = [rng.randrange(1, 5000) for _ in range(37)]
    convs = []
    for i in range(6):
        conv = system + [rng.randrange(1, 5000) for _ in range(rng.randrange(5, 60))]
        convs.append(conv)
        commit(conv, scopes[i % 2])
    for step in range(120):
        conv = convs[rng.randrange(len(convs))]
        scope = scopes[rng.randrange(len(scopes))]
        kind = rng.randrange(6)
        if kind == 0:
            seq = conv[: rng.randrange(1, len(conv) + 1)]
        elif kind == 1:
            seq = conv + [rng.randrange(1, 5000) for _ in range(rng.randrange(1, 30))]
        elif kind == 2:
            cut = rng.randrange(1, len(conv))
            seq = conv[:cut] + [5001 + rng.randrange(100)] + conv[cut + 1 :]
        elif kind == 3:
            seq = system[: rng.randrange(1, len(system) + 1)]
        elif kind == 4:
            seq = [rng.randrange(1, 5000) for _ in range(rng.randrange(1, 50))]
        else:
            seq = conv
        query(seq, scope, as_array=bool(step % 2))
        if step % 9 == 0:
            new_conv = conv + [rng.randrange(1, 5000) for _ in range(rng.randrange(3, 40))]
            convs.append(new_conv)
            commit(new_conv, scope)
    manager.shutdown()
    return transcript


def _run(enabled: bool) -> list:
    env = dict(os.environ)
    if enabled:
        env[_ENV] = "1"
    else:
        env.pop(_ENV, None)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT ") :])


def test_memo_does_not_change_matches():
    off = _run(False)
    on = _run(True)
    assert len(off) == len(on) == 120
    # The scenario must exercise real reuse, not only misses.
    assert sum(1 for r in off if r[1] > 0) > 40
    for i, (a, b) in enumerate(zip(off, on)):
        assert a == b, (i, a, b)


if __name__ == "__main__" and "--worker" in sys.argv:
    print("RESULT " + json.dumps(_scenario()), flush=True)
