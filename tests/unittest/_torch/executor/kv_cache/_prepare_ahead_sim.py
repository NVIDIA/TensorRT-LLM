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
"""Simulator of ``TRTLLM_SCHED_PREPARE_AHEAD_PASSES`` for ``test_kv_cache_v2_scheduler_prepare_ahead.py``.

Runs in its own process: the real ``KVCacheV2Scheduler`` of a given ``scheduler_v2.py`` is copied
into a temporary tree of stub ``tensorrt_llm`` modules and driven against a fake KV cache manager
with a block radix tree (prefix match, claim, resume, resize, free), under a closed-loop
multi-turn workload with forked siblings (prefix skip), arrivals right before the predecessor's
cache closes, cancellations, a background load on the GPU pool that walks between ``SIM_BG_LO`` and
``SIM_BG_HI`` of it (default 0.30 to 0.60) and, on ``SIM_SPIKE_P`` of the passes, a spike inside
phase 1 that fills the pool to ``SIM_SPIKE_MARGIN`` pages short of full, counting only what the
scheduler would hold without prepare-ahead. Allocations fail instead of asserting.

    python3 _prepare_ahead_sim.py SCHEDULER_V2_PY [ENV=VALUE ...]   prints one JSON line
    python3 _prepare_ahead_sim.py SCHEDULER_V2_PY knobs             prints the knob table

Invariants counted on every pass (a violation does not stop the run):

* I1: every claim taken ahead starts with every pool group at or below the ceiling;
* I2: after a deferred pass took claims, the pool with all held claims is at or below the ceiling;
* I3: when context admission starts, claims are held only if the pool is at or below it;
* I4: no generation request is evicted or paused (the pool is sized so that it never needs to).

The trace digest covers every pass's scheduled context requests (id, position, chunk), generation
requests and paused requests; the claims digest every claim depth taken at admission. Two runs
admit the same requests in the same steps iff their digests match.
"""

import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import types

TPB = 16

STUBS = {
    "tensorrt_llm/__init__.py": "",
    "tensorrt_llm/llmapi/__init__.py": "",
    "tensorrt_llm/llmapi/llm_args.py": """
import enum
class CapacitySchedulerPolicy(enum.Enum):
    MAX_UTILIZATION = 0
    GUARANTEED_NO_EVICT = 1
class ContextChunkingPolicy(enum.Enum):
    FIRST_COME_FIRST_SERVED = 0
    EQUAL_PROGRESS = 1
    FORCE_CHUNK = 2
""",
    "tensorrt_llm/logger.py": """
class _Logger:
    def __init__(self):
        self.lines = []
    def _log(self, msg, *args):
        self.lines.append(msg % args if args else msg)
    info = warning = debug = error = _log
logger = _Logger()
""",
    "tensorrt_llm/_torch/__init__.py": "",
    "tensorrt_llm/_torch/pyexecutor/__init__.py": "",
    "tensorrt_llm/_torch/pyexecutor/llm_request.py": """
import enum
class LlmRequestState(enum.Enum):
    ENCODER_INIT = 1
    DISAGG_GENERATION_INIT = 8
    DISAGG_GENERATION_TRANS_IN_PROGRESS = 9
    CONTEXT_INIT = 10
    GENERATION_IN_PROGRESS = 13
    GENERATION_TO_COMPLETE = 14
    GENERATION_COMPLETE = 20
    DISAGG_CONTEXT_TRANS_IN_PROGRESS = 21
    DISAGG_CONTEXT_COMPLETE = 22
class LlmRequest:
    pass
def get_draft_token_length(request):
    if request.py_draft_tokens is not None:
        return len(request.py_draft_tokens)
    return 0
def rewind_context_after_cache_drop(request, tokens_per_block):
    request.set_prepopulated_prompt_len(0, tokens_per_block)
    request.context_current_position = 0
    request.context_chunk_size = request.prompt_len
    request.estimated_reusable_tokens = 0
    request.py_ctx_pre_resize_cap = None
""",
    "tensorrt_llm/_torch/pyexecutor/scheduler/__init__.py": "",
    "tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py": """
from collections import namedtuple
RequestList = list
class RequestScheduler:
    pass
class SchedulerOutput(namedtuple("SchedulerOutput", ["encoder_requests", "context_requests",
        "generation_requests", "paused_requests", "fitting_disagg_gen_init_requests",
        "num_fitting_requests", "recompute_paused_requests"])):
    pass
def _get_forced_context_chunk_size(req):
    raise AssertionError("FORCE_CHUNK not simulated")
def _is_forced_context_chunk_boundary(req, chunk):
    raise AssertionError("FORCE_CHUNK not simulated")
def _get_lora_task_id(req):
    return (0, 0)
def drop_decoder_context_requests_waiting_for_encoder_output(active_requests):
    return list(active_requests)
""",
    "tensorrt_llm/_torch/pyexecutor/kv_cache/__init__.py": "",
    "tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py": """
import enum
GPU_LEVEL = 0
class BlockReusePolicy(str, enum.Enum):
    ALL_REUSABLE = "all_reusable"
    PER_REQUEST = "per_request"
class _Introspection:
    @staticmethod
    def storage_utilization(impl, level):
        return impl.storage_utilization()
_introspection = _Introspection()
class KVCacheManagerV2:
    pass
def _rewind_context_cursor(req, reuse):
    if reuse < req.context_current_position:
        req.context_current_position = reuse
def _settle_context_cursor(req, reuse, tokens_per_block):
    reuse = max(reuse, req.py_connector_served_position)
    _rewind_context_cursor(req, reuse)
    req.set_prepopulated_prompt_len(reuse, tokens_per_block)
    req.context_chunk_size = req.context_remaining_length
""",
}


def build(sched_path):
    root = tempfile.mkdtemp(prefix="prepare-ahead-sim-", dir=os.environ.get("TMPDIR"))
    for rel, text in STUBS.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    shutil.copy(
        sched_path, os.path.join(root, "tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py")
    )
    sys.path.insert(0, root)
    return root


class Req:
    """The LlmRequest surface the V2 scheduler reads, with C++ cursor semantics."""

    def __init__(self, rid, tokens, group_keys, osl, conv):
        self.py_request_id = self.request_id = rid
        self.tokens = tokens
        self.keys = group_keys  # cumulative block keys of the prompt
        self.prompt_len = len(tokens)
        self.state = self.state_cls.CONTEXT_INIT
        self.context_current_position = 0
        self.prepopulated = 0
        self.context_chunk_size = self.prompt_len
        self.py_draft_tokens = None
        self.lora_task_id = None
        self.is_dummy = False
        self.py_beam_width = 1
        self.py_multimodal_data = None
        self.cache_salt = None
        self.py_connector_served_position = 0
        self.estimated_reusable_tokens = 0
        self.py_ctx_pre_resize_cap = None
        self.is_generation_only_request = False
        self.py_decoding_iter = 0
        self.osl = osl
        self.generated = []
        self.conv = conv

    state_cls = None

    @property
    def state_value(self):
        return self.state.value

    @property
    def is_first_context_chunk(self):
        return self.context_current_position == self.prepopulated

    @property
    def is_last_context_chunk(self):
        return self.context_current_position + self.context_chunk_size == self.prompt_len

    @property
    def context_remaining_length(self):
        return self.prompt_len - self.context_current_position

    @property
    def is_context_init_state(self):
        return self.state is self.state_cls.CONTEXT_INIT

    @property
    def is_generation_in_progress_state(self):
        return self.state in (
            self.state_cls.GENERATION_IN_PROGRESS,
            self.state_cls.GENERATION_TO_COMPLETE,
        )

    @property
    def is_generation_to_complete_state(self):
        return self.state is self.state_cls.GENERATION_TO_COMPLETE

    @property
    def is_disagg_generation_init_state(self):
        return False

    def get_beam_width_by_iter(self, for_next_iteration=False):
        return 1

    def set_prepopulated_prompt_len(self, n, tpb):
        assert n < self.prompt_len
        self.prepopulated = n
        if n > 0:
            chunk = self.context_chunk_size
            if n + chunk < self.prompt_len:
                chunk = (n + chunk) // tpb * tpb - n
            self.context_current_position = n
            self.context_chunk_size = min(chunk, self.prompt_len - n)


def block_keys(tokens, parent=b""):
    keys, key = [], parent
    for i in range(len(tokens) // TPB):
        key = hashlib.blake2b(
            key + bytes(str(tokens[i * TPB : (i + 1) * TPB]), "ascii"), digest_size=8
        ).digest()
        keys.append(key)
    return keys


class Cache:
    def __init__(self, claimed):
        self.claimed = claimed
        self.pages = -(-claimed // TPB)
        self.active = False


class FakeKvManager:
    bg = 0
    sched = None
    spike_pending = False
    spike_margin = 80
    gen_alloc_failures = 0
    can_evict = True
    has_cache_tier_below_gpu = True

    def held_ahead_pages(self):
        held = getattr(self.sched, "_prepared_ahead", {}) or {}
        return sum(
            self.kv_cache_map[r].pages
            for r in held
            if r in self.kv_cache_map and self.kv_cache_map[r].active
        )

    def util(self):
        return (self.used_pages() + self.bg) / self.total_pages

    def _maybe_spike(self):
        if self.spike_pending:
            self.spike_pending = False
            self.spikes += 1
            base_used = self.used_pages() - self.held_ahead_pages()
            self.bg = max(0, self.total_pages - self.spike_margin - base_used)

    def preempt_request(self, req):
        return False

    def has_pending_preemption(self):
        return False

    """Radix tree of committed block keys; claim depth = matched full blocks, capped like DFlash's
    recompute tail; claims and resizes hold pages; resume refuses above max_util_for_resume."""

    def __init__(self, total_pages, tail):
        self.tokens_per_block = TPB
        self.enable_block_reuse = True
        self.enable_joint_kv_cache_reuse = False
        self._has_cp_helix = False
        self.kv_connector_manager = None
        self.kv_cache_map = {}
        self.tree = set()
        self.total_pages = total_pages
        self.tail = tail
        self.claims = []  # (rid, depth) per created cache
        self.spikes = 0
        mgr = self

        class Impl:
            init_config = types.SimpleNamespace(max_util_for_resume=0.95)

            def storage_utilization(self):
                return [mgr.util()]

        self.impl = Impl()

    def used_pages(self):
        return sum(c.pages for c in self.kv_cache_map.values() if c.active)

    def _match(self, req, end):
        n = 0
        for key in req.keys[: max(end, 0) // TPB]:
            if key not in self.tree:
                break
            n += 1
        return n * TPB

    def probe_context_reuse(self, req):
        if req.py_request_id in self.kv_cache_map:
            return None
        return self._match(req, req.prompt_len - 1)

    def probe_first_new_block_key(self, req):
        m = self._match(req, req.prompt_len - 1) // TPB
        if (m + 1) * TPB > req.prompt_len - 1:
            return None
        return req.keys[m]

    def prepare_context(self, req):
        cache = self.kv_cache_map.get(req.py_request_id)
        if req.is_first_context_chunk:
            if cache is None:
                limit = max(0, (req.prompt_len - self.tail) // TPB * TPB)
                depth = self._match(req, min(req.prompt_len - 1, limit))
                cache = Cache(depth)
                self.kv_cache_map[req.py_request_id] = cache
                self.claims.append((req.py_request_id, depth))
            if not cache.active:
                if (
                    self.util() > 0.95
                    or self.used_pages() + self.bg + cache.pages > self.total_pages
                ):
                    return False
                cache.active = True
            from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
                _settle_context_cursor,
            )

            _settle_context_cursor(req, cache.claimed, TPB)
            return True
        cache.active = True
        return True

    def resize_context(self, req, num_tokens):
        cache = self.kv_cache_map[req.py_request_id]
        need = -(-(req.context_current_position + num_tokens) // TPB) - cache.pages
        if need > 0:
            if self.used_pages() + self.bg + need > self.total_pages:
                self.ctx_alloc_failures = getattr(self, "ctx_alloc_failures", 0) + 1
                return False
            cache.pages += need
        return True

    def try_allocate_generation(self, req):
        self._maybe_spike()
        cache = self.kv_cache_map[req.py_request_id]
        length = req.prompt_len + len(req.generated) + 8
        need = -(-length // TPB) - cache.pages
        extra = need if need > 0 else 0
        if not cache.active:
            extra += cache.pages
        if extra and self.used_pages() + self.bg + extra > self.total_pages:
            self.gen_alloc_failures += 1
            return False
        cache.active = True
        if need > 0:
            cache.pages += need
        return True

    def revert_allocate_generation(self, req):
        pass

    def suspend_request(self, req):
        cache = self.kv_cache_map.get(req.py_request_id)
        if cache is not None:
            cache.active = False

    def is_request_active(self, rid):
        cache = self.kv_cache_map.get(rid)
        return cache is not None and cache.active

    def has_active_cache(self, req):
        return self.is_request_active(req.py_request_id)

    def free_resources(self, req):
        self.kv_cache_map.pop(req.py_request_id, None)

    def commit(self, keys, upto):
        self.tree.update(keys[: upto // TPB])


def run(sched_path, env, seed=7, passes=6000):
    os.environ.update(env)
    build(sched_path)
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm_mod
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
    from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2
    from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, ContextChunkingPolicy
    from tensorrt_llm.logger import logger

    Req.state_cls = LlmRequestState

    class Manager(FakeKvManager, kvm_mod.KVCacheManagerV2):
        block_reuse_policy = kvm_mod.BlockReusePolicy.ALL_REUSABLE

    rng = random.Random(seed)
    kvm = Manager(total_pages=int(os.environ.get("SIM_TOTAL_PAGES", "5000")), tail=32)
    bg_rng = random.Random(seed * 7919 + 1)
    bg_lo = float(os.environ.get("SIM_BG_LO", "0.30"))
    bg_hi = float(os.environ.get("SIM_BG_HI", "0.60"))
    spike_p = float(os.environ.get("SIM_SPIKE_P", "0"))
    kvm.spike_margin = int(os.environ.get("SIM_SPIKE_MARGIN", "80"))
    sched = scheduler_v2.KVCacheV2Scheduler(
        max_batch_size=int(os.environ.get("SIM_MAX_BATCH", "24")),
        max_num_tokens=768,
        kv_cache_manager=kvm,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        ctx_chunk_config=(ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, TPB),
    )
    kvm.sched = sched
    reads = [0]
    impl_cls = type(kvm.impl)  # a class per fake manager instance
    orig_util = impl_cls.storage_utilization
    impl_cls.storage_utilization = lambda self: (
        reads.__setitem__(0, reads[0] + 1) or orig_util(self)
    )
    cls = type(sched)
    ceiling = getattr(sched, "_prepare_ahead_max_util", None) or 0.475
    inv = {
        "I1": 0,
        "I2": 0,
        "I3": 0,
        "I4": 0,
        "ahead_claims": 0,
        "max_held_ahead_pages": 0,
        "max_util_with_held": 0.0,
        "phase2_checks": 0,
    }
    in_ahead = [False]
    orig_pair = cls._prepare_context_pair
    orig_ahead = getattr(cls, "_prepare_context_ahead", None)
    orig_adm = cls._admissible_pending_ctx

    def pair(self, req):
        if in_ahead[0]:
            inv["ahead_claims"] += 1
            if kvm.util() > ceiling + 1e-12:
                inv["I1"] += 1
        return orig_pair(self, req)

    def ahead(self, *a, **k):
        before = set(self._prepared_ahead)
        in_ahead[0] = True
        try:
            orig_ahead(self, *a, **k)
        finally:
            in_ahead[0] = False
        if set(self._prepared_ahead) - before and kvm.util() > ceiling + 1e-12:
            inv["I2"] += 1

    def adm(self, *a, **k):
        inv["phase2_checks"] += 1
        if getattr(self, "_prepared_ahead", None):
            held = kvm.held_ahead_pages()
            inv["max_held_ahead_pages"] = max(inv["max_held_ahead_pages"], held)
            inv["max_util_with_held"] = max(inv["max_util_with_held"], kvm.util())
            if kvm.util() > ceiling + 1e-12:
                inv["I3"] += 1
        return orig_adm(self, *a, **k)

    cls._prepare_context_pair = pair
    if orig_ahead is not None:
        cls._prepare_context_ahead = ahead
    cls._admissible_pending_ctx = adm
    bg_target = [bg_lo]
    system = [rng.randrange(1000) for _ in range(160)]
    kvm.commit(block_keys(system), len(system))
    next_id = [1]
    arrivals = {}  # pass -> list of requests

    def new_req(tokens, conv):
        rid = next_id[0]
        next_id[0] += 1
        return Req(rid, tokens, block_keys(tokens), osl=rng.randrange(4, 40), conv=conv)

    convs = {}
    for c in range(30):
        convs[c] = {"turn": 0}
        tokens = system + [rng.randrange(1000) for _ in range(rng.randrange(40, 600))]
        arrivals.setdefault(rng.randrange(0, 40), []).append(new_req(tokens, c))
    next_conv = [30]
    active, closing, trace, admitted_claims = [], [], hashlib.sha256(), []
    stats = {
        "ctx_admitted": 0,
        "fresh_claims_at_admission": 0,
        "cancelled": 0,
        "forks": 0,
        "race_arrivals": 0,
        "max_util": 0.0,
    }

    for p in range(passes):
        for req in arrivals.pop(p, []):
            active.append(req)
        # cancel a pending request now and then (frees whatever it holds, as _terminate_request)
        if rng.random() < 0.01:
            pend = [
                r
                for r in active
                if r.state is LlmRequestState.CONTEXT_INIT and r.is_first_context_chunk
            ]
            if pend:
                victim = rng.choice(pend)
                kvm.free_resources(victim)
                active.remove(victim)
                stats["cancelled"] += 1
        if p % 37 == 0:
            bg_target[0] = bg_rng.uniform(bg_lo, bg_hi)
        kvm.bg = int(bg_target[0] * kvm.total_pages)
        kvm.spike_pending = bg_rng.random() < spike_p
        claims_before = len(kvm.claims)
        out = sched.schedule_request(active, set())
        kvm.spike_pending = False
        ctx = [
            (r.py_request_id, r.context_current_position, r.context_chunk_size)
            for r in out.context_requests
        ]
        gen = sorted(r.py_request_id for r in out.generation_requests)
        paused = sorted(
            r.py_request_id for r in out.paused_requests + out.recompute_paused_requests
        )
        inv["I4"] += len(paused)
        trace.update(repr((p, ctx, gen, paused)).encode())
        admitted_ids = {r.py_request_id for r in out.context_requests}
        for rid, depth in kvm.claims[claims_before:]:
            if rid in admitted_ids:
                stats["fresh_claims_at_admission"] += 1
        for r in out.context_requests:
            if r.context_current_position == r.prepopulated:
                admitted_claims.append((r.py_request_id, r.prepopulated))
        stats["ctx_admitted"] += len(out.context_requests)
        # execute: context chunks commit their blocks, generation decodes
        for r in out.context_requests:
            r.context_current_position += r.context_chunk_size
            kvm.commit(r.keys, r.context_current_position)
            if r.context_current_position == r.prompt_len:
                r.state = LlmRequestState.GENERATION_IN_PROGRESS
        finished = []
        for r in out.generation_requests:
            step = 1 + (hash((r.py_request_id, len(r.generated))) % 4)
            r.generated += [
                (r.py_request_id * 31 + len(r.generated) + i) % 1000 for i in range(step)
            ]
            if len(r.generated) >= r.osl:
                r.state = LlmRequestState.GENERATION_COMPLETE
                finished.append(r)
        # the overlap executor closes a finished request's cache one loop later; its next turn
        # can already be fetched in that loop (gap 0), before the close
        for r in closing:
            full = r.tokens + r.generated
            kvm.commit(block_keys(full), len(full))
            kvm.free_resources(r)
            active.remove(r)
        closing = finished
        for r in finished:
            full = r.tokens + r.generated
            conv = convs[r.conv]
            conv["turn"] += 1
            if conv["turn"] < 6:
                gap = rng.choice([0, 0, 1, 2, 3, 5, 8, 13])
                stats["race_arrivals"] += gap == 0
                tokens = full + [rng.randrange(1000) for _ in range(rng.randrange(8, 120))]
                arrivals.setdefault(p + 1 + gap, []).append(new_req(tokens, r.conv))
                if rng.random() < 0.15:  # forked siblings sharing an uncached segment
                    shared = [rng.randrange(1000) for _ in range(3 * TPB)]
                    for _ in range(rng.choice([2, 3])):
                        c = next_conv[0]
                        next_conv[0] += 1
                        convs[c] = {"turn": 3}
                        sib = (
                            full
                            + shared
                            + [rng.randrange(1000) for _ in range(rng.randrange(8, 60))]
                        )
                        arrivals.setdefault(p + 1 + gap, []).append(new_req(sib, c))
                        stats["forks"] += 1
            else:
                c = next_conv[0]
                next_conv[0] += 1
                convs[c] = {"turn": 0}
                tokens = system + [rng.randrange(1000) for _ in range(rng.randrange(40, 600))]
                arrivals.setdefault(p + 1 + rng.randrange(0, 10), []).append(new_req(tokens, c))
        stats["max_util"] = max(stats["max_util"], kvm.util())

    lines = [ln for ln in logger.lines if "prepare-ahead" in ln]
    ahead = [ln for ln in lines if "over" in ln]
    return {
        "digest": trace.hexdigest()[:16],
        "claims_digest": hashlib.sha256(repr(admitted_claims).encode()).hexdigest()[:16],
        **stats,
        **inv,
        "ceiling": ceiling,
        "spikes": kvm.spikes,
        "pool_reads": reads[0],
        "gen_alloc_failures": kvm.gen_alloc_failures,
        "ctx_alloc_failures": getattr(kvm, "ctx_alloc_failures", 0),
        "init_line": next((ln for ln in lines if "passes=" in ln), None),
        "last_stats_line": ahead[-1] if ahead else None,
    }


def knobs(sched_path):
    """The ceiling the scheduler reads from its knobs, and the values it refuses."""
    build(sched_path)
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm_mod
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
    from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2
    from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, ContextChunkingPolicy

    Req.state_cls = LlmRequestState

    class Manager(FakeKvManager, kvm_mod.KVCacheManagerV2):
        block_reuse_policy = kvm_mod.BlockReusePolicy.ALL_REUSABLE

    passes, max_util = "TRTLLM_SCHED_PREPARE_AHEAD_PASSES", "TRTLLM_SCHED_PREPARE_AHEAD_MAX_UTIL"

    def make(env):
        for name in (passes, max_util):
            os.environ.pop(name, None)
        os.environ.update({"TRTLLM_SCHED_PREFILL_DECODE_INTERVAL": "8", **env})
        manager = Manager(total_pages=1000, tail=32)
        return scheduler_v2.KVCacheV2Scheduler(
            max_batch_size=8,
            max_num_tokens=256,
            kv_cache_manager=manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=(ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, TPB),
        )

    cases = [
        ("off", {}),
        ("default", {passes: "8"}),
        ("empty", {passes: "8", max_util: ""}),
        ("0.7", {passes: "8", max_util: "0.7"}),
        ("0.95", {passes: "8", max_util: "0.95"}),
        ("ceiling without passes", {max_util: "0.7"}),
    ]
    out = {"ceiling": {name: make(env)._prepare_ahead_max_util for name, env in cases}}
    rejected = []
    for bad in ("0", "-0.1", "0.96", "1.5", "abc"):
        try:
            make({passes: "8", max_util: bad})
        except ValueError:
            rejected.append(bad)
    out["rejected"] = rejected
    return out


if __name__ == "__main__":
    if sys.argv[2:] == ["knobs"]:
        print(json.dumps(knobs(sys.argv[1])))
    else:
        env = dict(arg.split("=", 1) for arg in sys.argv[2:])
        seed, passes = (
            int(os.environ.get("SIM_SEED", "7")),
            int(os.environ.get("SIM_PASSES", "6000")),
        )
        print(json.dumps(run(sys.argv[1], env, seed=seed, passes=passes)))
