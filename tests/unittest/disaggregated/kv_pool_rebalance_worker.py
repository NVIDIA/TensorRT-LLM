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
"""Worker for test_kv_pool_rebalance.py: context-side KV pool rebalance under disagg.

ctx and gen are separate single-rank MPI worlds, as in the cache_transceiver_test
harness, whose transfer protocol (run_one_request) this reuses. Sequence:

  1. transfer + verify a few requests (pre-rebalance baseline);
  2. ctx forces a real pool resize with the same primitives the executor uses
     (begin_kv_pool_resize / adjust / end_kv_pool_resize);
  3. ctx parks filler sequences so later requests land in slots past the
     *old* registered extent of the grown pool group;
  4. transfer + verify more requests; report whether any of them used slots
     past the old extent (i.e. whether re-registration was actually exercised).

MODE=refresh (default) uses the resize hooks. MODE=none skips them (negative
control: adjust() under a stale registration). MODE=skiptail runs the hooks, but
with incremental registration the end hook registers nothing past each pool's
pre-resize length, so the grown tail alone stays unregistered (negative control
for the segment path; expect NIXL_ERR_NOT_FOUND).

Usage: python kv_pool_rebalance_worker.py --role ctx|gen   (env: ZMQ_PORT, MODE)
The ctx role ends with a "KV_POOL_REBALANCE_EXERCISED <n>/<m>" line: how many of
the post-resize requests read slots past the old registered extent.
"""

import argparse
import os
import sys

HARNESS = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        os.pardir,
        "examples",
        "disaggregated",
        "slurm",
        "cache_transceiver_test",
    )
)
sys.path.insert(0, HARNESS)

import run_cache_transceiver_test as h  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402
from mpi4py import MPI  # noqa: E402

import tensorrt_llm  # noqa: E402
from tensorrt_llm._torch.disaggregation.resource.utils import (  # noqa: E402
    get_unique_pool_memory_descs,
)
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection as intro  # noqa: E402

NUM_LAYERS = 5
TPB = 32
MAX_SEQ_LEN = 1024
# VSWA with unequal layer counts per window -> two pool groups with different
# slot sizes, which is what gives adjust() something to rebalance.
WINDOWS = [256, 256, 256, 256, MAX_SEQ_LEN]
REQ_LEN = 200
MODE = os.environ.get("MODE", "refresh")
# One deadline for every mode. It also covers the three warm-up transfers, and the
# first of those includes NIXL/UCX connection setup, which can take over 10 s on a
# busy or slow node. The negative control's ctx completion check waits out this
# deadline after its send fails, which only makes that case slower.
TRANSFER_TIMEOUT_MS = 60_000


def build_mgr(mapping):
    return h.KVCacheManagerV2(
        h.KvCacheConfigV2(
            max_tokens=MAX_SEQ_LEN * 6,
            enable_block_reuse=False,
            max_attention_window=WINDOWS,
        ),
        h.CacheTypeCpp.SELF,
        vocab_size=32000,
        num_layers=NUM_LAYERS,
        num_kv_heads=8,
        head_dim=128,
        tokens_per_block=TPB,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=64,
        mapping=mapping,
        dtype=h.DataType.HALF,
    )


def pool_slots(page_table):
    return [pg.pools[0].num_slots for pg in page_table.pool_groups]


def max_slot_per_pool_group(kvm, page_table, req):
    out = {}
    for lg_idx, slots in enumerate(kvm.get_page_indices_by_layer_group(req)):
        pg = page_table.layer_groups[lg_idx].pool_group_idx
        valid = [s for s in slots if s >= 0]
        if valid:
            out[pg] = max(out.get(pg, -1), max(valid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["ctx", "gen"])
    role = ap.parse_args().role
    is_ctx = role == "ctx"
    comm = MPI.COMM_WORLD
    torch.cuda.set_device(0)
    tensorrt_llm.logger.set_level("warning")

    mapping = h.Mapping(world_size=1, rank=0, tp_size=1, pp_size=1, gpus_per_node=1)
    dist = h.Distributed.get(mapping)
    kvm = build_mgr(mapping)
    xcvr = h.create_kv_cache_transceiver(
        mapping,
        dist,
        kvm,
        h.AttentionTypeCpp.DEFAULT,
        CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            max_tokens_in_buffer=MAX_SEQ_LEN,
            kv_transfer_timeout_ms=TRANSFER_TIMEOUT_MS,
        ),
    )
    zctx = zmq.Context.instance()
    sock = zctx.socket(zmq.REP if is_ctx else zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 120_000)
    port = int(os.environ.get("ZMQ_PORT", "29555"))
    if is_ctx:
        sock.bind(f"tcp://127.0.0.1:{port}")
    else:
        sock.connect(f"tcp://127.0.0.1:{port}")

    def transfer(rid):
        return h.run_one_request(
            role, comm, kvm, xcvr, "PYTHON", True, NUM_LAYERS, rid, REQ_LEN, 0, sock
        )

    tag = f"[{role}]"
    results = []
    for rid in range(1, 4):
        results.append(transfer(rid))

    fillers = []
    exercised = []
    if is_ctx:
        worker = xcvr._transfer_worker
        # gen registered with this rank on the same node: no cross-node veto.
        assert not worker.has_cross_node_peer, "same-node peer flagged as cross-node"
        dev = worker._rank_info.device_id
        pt_before = worker.page_table
        slots_before = pool_slots(pt_before)
        reg_before = get_unique_pool_memory_descs(pt_before, dev)
        assert not xcvr.kv_pool_layout_shared_with_peers(), "ctx must not have registered"
        assert not xcvr.has_inflight_transfers()
        assert xcvr.kv_pool_rebalance_unsupported_reason() is None
        desc_before = worker._rank_info.transfer_engine_info

        # Grow the pool group that limits admission (SKEW_GROUP), so fillers can
        # push later requests past its old extent. force_rebalance_precondition
        # always grows group 0.
        skew_group = int(os.environ.get("SKEW_GROUP", "1"))
        current = intro.current_gpu_ratio(kvm.impl)
        skewed = [r * (3.0 if i == skew_group else 1.0) for i, r in enumerate(current)]
        intro.set_num_sampled_kv_caches(kvm.impl, 2001)
        intro.set_last_adjustment_time(kvm.impl, 0.0)
        intro.set_target_ratio_list_gpu(kvm.impl, [r / sum(skewed) for r in skewed])
        assert kvm.impl.need_adjustment
        if MODE == "refresh":
            xcvr.begin_kv_pool_resize()
            try:
                kvm.impl.adjust()
            finally:
                xcvr.end_kv_pool_resize()
        elif MODE == "skiptail":
            tm = sys.modules[type(worker).__module__]
            assert worker._kv_segments is not None, "needs incremental registration (14)"
            orig_pools = tm._paged_kv_pools
            old_pools = orig_pools(pt_before)

            def pools_capped_at_old_length(page_table):
                pools = orig_pools(page_table)
                return {
                    base: (stride, pg, min(n, old_pools[base][2]) if base in old_pools else n)
                    for base, (stride, pg, n) in pools.items()
                }

            xcvr.begin_kv_pool_resize()
            try:
                kvm.impl.adjust()
            finally:
                tm._paged_kv_pools = pools_capped_at_old_length
                try:
                    xcvr.end_kv_pool_resize()
                finally:
                    tm._paged_kv_pools = orig_pools
            for base, segs in worker._kv_segments.items():
                stride, _, n = old_pools[base]
                print(
                    f"{tag} pool {base:#x}: registered up to {segs[-1][1] if segs else 0} "
                    f"bytes, old length {n * stride}",
                    flush=True,
                )
        else:
            kvm.impl.adjust()

        pt_after = worker.page_table
        actual = [int(d.num_slots) for d in kvm.impl.pool_group_descs]
        print(
            f"{tag} num_slots before={slots_before} manager after={actual} "
            f"page_table after={pool_slots(pt_after)}",
            flush=True,
        )
        print(f"{tag} registered before={[(hex(p), s) for p, s, _, _ in reg_before]}", flush=True)
        reg_after = get_unique_pool_memory_descs(pt_after, dev)
        print(f"{tag} registered after ={[(hex(p), s) for p, s, _, _ in reg_after]}", flush=True)
        assert actual != slots_before, "adjust() did not resize any pool group"
        if MODE in ("refresh", "skiptail"):
            assert pool_slots(pt_after) == actual
            assert xcvr._page_table is pt_after
            assert worker._peer_registrar.self_extractor.page_table is pt_after
            assert worker._rank_info.transfer_engine_info != desc_before
        grown = [i for i, (a, b) in enumerate(zip(slots_before, actual)) if b > a]
        print(f"{tag} grown pool groups: {grown}", flush=True)

        # Park fillers until the grown group has handed out slots past its old
        # extent; later test requests then compete for the tail.
        for i in range(48):
            req = h.make_request(True, 900_000 + i, REQ_LEN, "PYTHON")
            try:
                h.add_sequence(kvm, req, REQ_LEN, True)
            except RuntimeError:
                break
            fillers.append(req)
            top = max_slot_per_pool_group(kvm, pt_after, req)
            if all(top.get(g, -1) >= slots_before[g] for g in grown):
                break
        print(f"{tag} parked {len(fillers)} fillers", flush=True)

        orig_add = h.add_sequence

        def add_and_record(mgr, req, prompt_len, use_v2):
            r = orig_add(mgr, req, prompt_len, use_v2)
            top = max_slot_per_pool_group(mgr, pt_after, req)
            past = {g: top[g] for g in grown if top.get(g, -1) >= slots_before[g]}
            exercised.append((req.py_request_id, top, past))
            return r

        h.add_sequence = add_and_record

    for rid in range(4, 8):
        results.append(transfer(rid))

    if is_ctx:
        h.add_sequence = orig_add
        for rid, top, past in exercised:
            print(
                f"{tag} rid={rid} max slot per pool group={top} past old extent={past}", flush=True
            )
        n_past = sum(1 for _, _, past in exercised if past)
        print(
            f"{tag} requests reading past the old registered extent: {n_past}/{len(exercised)}",
            flush=True,
        )
        print(f"KV_POOL_REBALANCE_EXERCISED {n_past}/{len(exercised)}", flush=True)
        for req in fillers:
            h.free_sequence(kvm, req, None, True)
    else:
        print(f"{tag} verify results: {results}", flush=True)
        print(
            f"{tag} gen layout shared with peers: {xcvr.kv_pool_layout_shared_with_peers()}",
            flush=True,
        )
        assert xcvr.kv_pool_layout_shared_with_peers()
        print(f"{tag} RESULT {'PASS' if all(results) else 'FAIL'}", flush=True)
    xcvr.shutdown()


if __name__ == "__main__":
    main()
