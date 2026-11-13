# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker for test_kv_pool_rebalance.py::test_generation_side_rebalance_integrity.

Generation-side KV pool rebalance under copy-based rebalancing (CoReba). The context and generation
sides are separate single-rank MPI worlds on one GPU and reuse the cache-transceiver harness's
transfer protocol (run_one_request). Both managers keep their GPU KV cache in an arena.

Sequence:
  1. transfer and verify requests 1-3; the generation rank has now registered with the context peer,
     which holds its page table and registration keys;
  2. the generation side parks a live sequence filled with a known pattern;
  3. it forces a rebalance with the executor's primitives (begin_kv_pool_resize / adjust /
     end_kv_pool_resize) and checks that nothing a peer holds changed (registration, agent
     descriptor) and that the live sequence's pattern survived;
  4. it parks fillers until the grown pool group hands out slots outside its old window;
  5. transfer and verify requests 4-7 into the new layout, the context peer still using the page
     table and keys from step 1, and report how many landed outside the old window.

Usage: python kv_pool_rebalance_gen_worker.py --role ctx|gen   (env: ZMQ_PORT, SKEW_GROUP)
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
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvPoolRebalanceConfig  # noqa: E402
from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection as intro  # noqa: E402

NUM_LAYERS = 5
TPB = 32
MAX_SEQ_LEN = 1024
# VSWA with unequal layer counts per window -> two pool groups with different slot sizes (512 KiB and
# 128 KiB, the layout of kv_pool_rebalance_worker.py).
WINDOWS = [256, 256, 256, 256, MAX_SEQ_LEN]
REQ_LEN = 200
LIVE_RID = 800_000


def build_mgr(mapping):
    return h.KVCacheManagerV2(
        h.KvCacheConfigV2(
            max_tokens=MAX_SEQ_LEN * 6,
            enable_block_reuse=False,
            max_attention_window=WINDOWS,
            kv_pool_rebalance_config=KvPoolRebalanceConfig(copy_based_rebalancing=True),
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


def slots_per_pool_group(kvm, page_table, req):
    out = {}
    for lg_idx, slots in enumerate(kvm.get_page_indices_by_layer_group(req)):
        pg = page_table.layer_groups[lg_idx].pool_group_idx
        out.setdefault(pg, []).extend(int(s) for s in slots if s >= 0)
    return out


def outside(slots_by_pg, windows, groups):
    """Slots of the given pool groups that lie outside that group's window in `windows`."""
    return {
        g: [s for s in slots_by_pg.get(g, []) if not windows[g][0] <= s < windows[g][1]]
        for g in groups
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["ctx", "gen"])
    role = ap.parse_args().role
    is_ctx = role == "ctx"
    comm = MPI.COMM_WORLD
    torch.cuda.set_device(0)
    tensorrt_llm.logger.set_level("warning")
    tag = f"[{role}]"

    mapping = h.Mapping(world_size=1, rank=0, tp_size=1, pp_size=1, gpus_per_node=1)
    dist = h.Distributed.get(mapping)
    kvm = build_mgr(mapping)
    extent = kvm.impl.get_gpu_arena_extent()
    assert extent is not None, "expected a CoReba arena"
    print(
        f"{tag} arena base={extent[0]:#x} bytes={extent[1]} bound_pools={extent[2]} "
        f"windows={intro.gpu_slot_windows(kvm.impl)}",
        flush=True,
    )
    xcvr = h.create_kv_cache_transceiver(
        mapping,
        dist,
        kvm,
        h.AttentionTypeCpp.DEFAULT,
        CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            max_tokens_in_buffer=MAX_SEQ_LEN,
            kv_transfer_timeout_ms=60_000,
        ),
    )
    worker = xcvr._transfer_worker
    print(
        f"{tag} registered: {[(hex(d[0]), d[1], d[3]) for d in worker._kv_registered_mem.descs]}"
        if hasattr(worker._kv_registered_mem, "descs")
        else f"{tag} registered: {worker._kv_registered_mem}",
        flush=True,
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
        try:
            return h.run_one_request(
                role, comm, kvm, xcvr, "PYTHON", True, NUM_LAYERS, rid, REQ_LEN, 0, sock
            )
        except Exception as e:  # noqa: BLE001 - reported before the harness aborts
            print(f"{tag} rid={rid} transfer raised {e!r}", flush=True)
            raise

    results = []
    for rid in range(1, 4):
        results.append(transfer(rid))

    fillers = []
    exercised = []
    live_ok = None
    if not is_ctx:
        assert xcvr.kv_pool_rebalance_unsupported_reason() is None
        assert worker.has_registered_with_peers, "gen must have registered with the ctx peer"
        shared = xcvr.kv_pool_layout_shared_with_peers()
        veto = xcvr.kv_pool_rebalance_veto_reason()
        print(f"{tag} layout shared with peers: {shared}, veto: {veto}", flush=True)
        assert not shared and veto is None
        assert not xcvr.has_inflight_transfers()

        # A live sequence with a known pattern, local to gen; the rebalance may move its pages.
        live = h.make_request(True, LIVE_RID, REQ_LEN, "PYTHON")
        h.add_sequence(kvm, live, REQ_LEN, True)
        h.fill_request(kvm, LIVE_RID, 0, NUM_LAYERS)
        pt_before = worker.page_table
        live_slots_before = slots_per_pool_group(kvm, pt_before, live)

        windows_before = [tuple(w) for w in intro.gpu_slot_windows(kvm.impl)]
        reg_before = worker._kv_registered_mem
        desc_before = worker._rank_info.transfer_engine_info

        skew_group = int(os.environ.get("SKEW_GROUP", "1"))
        current = intro.current_gpu_ratio(kvm.impl)
        skewed = [r * (3.0 if i == skew_group else 1.0) for i, r in enumerate(current)]
        intro.set_num_sampled_kv_caches(kvm.impl, 2001)
        intro.set_last_adjustment_time(kvm.impl, 0.0)
        intro.set_target_ratio_list_gpu(kvm.impl, [r / sum(skewed) for r in skewed])
        assert kvm.impl.need_adjustment
        torch.cuda.current_stream().synchronize()
        xcvr.begin_kv_pool_resize()
        try:
            # As the executor does in suspend mode (_rebalance_kv_pools_now).
            kvm.suspend_request(live)
            kvm.impl.adjust()
            kvm.resume_request(live)
        finally:
            xcvr.end_kv_pool_resize()
        torch.cuda.current_stream().synchronize()

        windows_after = [tuple(w) for w in intro.gpu_slot_windows(kvm.impl)]
        pt_after = worker.page_table
        live_slots_after = slots_per_pool_group(kvm, pt_after, live)
        moved = sum(
            len(set(live_slots_before.get(g, [])) - set(live_slots_after.get(g, [])))
            for g in live_slots_before
        )
        live_ok = h.verify_request(kvm, LIVE_RID, 0, NUM_LAYERS)
        print(f"{tag} windows before={windows_before} after={windows_after}", flush=True)
        print(
            f"{tag} registration unchanged: {worker._kv_registered_mem is reg_before}, "
            f"agent desc unchanged: {worker._rank_info.transfer_engine_info == desc_before}",
            flush=True,
        )
        print(f"{tag} live sequence: {moved} slots moved, pattern intact: {live_ok}", flush=True)
        assert windows_after != windows_before, "adjust() did not move any window"
        assert worker._kv_registered_mem is reg_before
        assert worker._rank_info.transfer_engine_info == desc_before
        grown = [
            g
            for g, (a, b) in enumerate(zip(windows_before, windows_after))
            if b[1] - b[0] > a[1] - a[0]
        ]
        print(f"{tag} grown pool groups: {grown}", flush=True)

        # Park fillers until the grown group has handed out slots outside its old window.
        for i in range(48):
            req = h.make_request(True, 900_000 + i, REQ_LEN, "PYTHON")
            try:
                h.add_sequence(kvm, req, REQ_LEN, True)
            except RuntimeError:
                break
            fillers.append(req)
            if all(
                outside(slots_per_pool_group(kvm, pt_after, req), windows_before, grown)[g]
                for g in grown
            ):
                break
        print(f"{tag} parked {len(fillers)} fillers", flush=True)

        orig_add = h.add_sequence

        def add_and_record(mgr, req, prompt_len, use_v2):
            r = orig_add(mgr, req, prompt_len, use_v2)
            out = outside(slots_per_pool_group(mgr, pt_after, req), windows_before, grown)
            exercised.append((req.py_request_id, {g: len(v) for g, v in out.items()}))
            return r

        h.add_sequence = add_and_record

    for rid in range(4, 8):
        results.append(transfer(rid))

    if not is_ctx:
        h.add_sequence = orig_add
        for rid, out in exercised:
            print(f"{tag} rid={rid} slots outside the old window per grown group={out}", flush=True)
        n_out = sum(1 for _, out in exercised if any(out.values()))
        print(
            f"{tag} requests received into slots outside the old window: {n_out}/{len(exercised)}",
            flush=True,
        )
        for req in fillers:
            h.free_sequence(kvm, req, None, True)
        h.free_sequence(kvm, live, None, True)
        print(f"{tag} verify results: {results}", flush=True)
        ok = all(results) and live_ok and n_out > 0
        print(f"{tag} RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    xcvr.shutdown()


if __name__ == "__main__":
    main()
