#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused invariant test for the Inkling short-conv state pool sizing contract.

The runtime schedules at most ``max_batch_size`` real requests, so the pool is
built with ``max_batch_size + 1`` rows (the ``+1`` is the CUDA-graph pad row).
But the one-time KV-cache *estimation* forward presents a dummy batch sized to
saturate ``max_num_tokens`` (and replicated ``x tp_size`` under attention DP),
which can exceed that capacity and used to crash with ``IndexError: pop from
empty list`` at the very first forward of executor init.

``InklingConvStateCache`` now grows elastically when one forward presents more
fresh requests than it has free rows. This test pins that behavior on CPU (no
checkpoint / no GPU needed -- it is pure pool bookkeeping):

* growth only when a batch oversubscribes the pool,
* every concurrently-live request keeps a DISTINCT row,
* growth preserves in-flight requests' carried windows and slot ids,
* the ``state_indices`` CUDA buffer AND its pinned ``state_indices_cpu`` host
  staging buffer grow in lock-step (the eager ``write_state_indices`` writes the
  resolved slots through the staging buffer into ``state_indices``, so a
  size mismatch would index past the staging buffer's end),
* ``free`` returns rows so later batches reuse them without further growth.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_inkling import InklingConvStateCache


def _fake_model_config(num_layers=4, kv_heads=8, head_dim=16, hidden=32,
                       kernel=4, tp=1):
    """A minimal duck-typed ModelConfig for InklingConvStateCache.__init__."""
    text = SimpleNamespace(
        sconv_kernel_size=kernel,
        num_hidden_layers=num_layers,
        hidden_size=hidden,
        layer_num_kv_heads=lambda i: kv_heads,
        layer_head_dim=lambda i: head_dim,
    )
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(text_config=text),
        mapping=SimpleNamespace(tp_size=tp),
    )


def test_inkling_conv_pool_elastic_growth():
    device = torch.device("cpu")
    cfg = _fake_model_config()
    cap0 = 3  # what InklingConvStateManager passes as max_batch_size + 1
    cache = InklingConvStateCache(cfg, cap0, device, dtype=torch.float32)
    assert cache.max_batch_size == cap0
    assert len(cache._free) == cap0
    assert cache.state_indices.numel() == cap0

    # 1) Allocate within capacity -> distinct rows, no growth.
    slots = cache.slots_for([10, 11])
    assert len(set(slots)) == 2
    assert cache.max_batch_size == cap0

    # Stamp request 10's carried window so we can prove growth preserves it.
    s10 = cache._slot_of[10]
    cache.layer_state(0).k[s10].fill_(7.0)
    cache.layer_state(cfg.pretrained_config.text_config.num_hidden_layers -
                      1).mlp[s10].fill_(5.0)

    # 2) Oversubscribe in ONE forward (estimation-style): 6 live ids incl. 4
    #    fresh, but only 1 free row -> the pool must grow, not raise.
    big = [10, 11, 20, 21, 22, 23]
    slots2 = cache.slots_for(big)
    assert len(slots2) == len(big)
    # Every concurrently-live request must own a distinct row.
    assert len(set(slots2)) == len(big)
    assert cache.max_batch_size >= len(big)
    # state_indices AND its pinned host staging grew in lock-step (write_state_indices
    # copies resolved slots through state_indices_cpu into state_indices).
    assert cache.state_indices.numel() == cache.max_batch_size
    assert cache.state_indices_cpu.numel() == cache.max_batch_size

    # 3) Growth preserved request 10's slot id AND its carried windows.
    assert cache._slot_of[10] == s10
    assert torch.all(cache.layer_state(0).k[s10] == 7.0)
    last = cfg.pretrained_config.text_config.num_hidden_layers - 1
    assert torch.all(cache.layer_state(last).mlp[s10] == 5.0)

    # 4) A fresh id allocated during growth gets a zeroed row.
    s20 = cache._slot_of[20]
    assert torch.all(cache.layer_state(0).k[s20] == 0.0)
    assert torch.all(cache.layer_state(last).mlp[s20] == 0.0)

    # 5) free returns rows; a later batch reuses them without growing again.
    grown = cache.max_batch_size
    cache.free(big)
    assert len(cache._free) == grown
    assert cache._slot_of == {}
    reused = cache.slots_for([30, 31])
    assert len(set(reused)) == 2
    assert cache.max_batch_size == grown  # reused freed rows, no new growth


def test_inkling_conv_pool_growth_repeats():
    """Repeated oversubscription keeps rows distinct and windows intact."""
    device = torch.device("cpu")
    cache = InklingConvStateCache(_fake_model_config(),
                                  2,
                                  device,
                                  dtype=torch.float32)
    live = {}
    rid = 0
    for _ in range(4):
        batch = list(range(rid, rid + 5))  # 5 fresh ids each round
        rid += 5
        cache.slots_for(batch)
        for r in batch:
            live[r] = cache._slot_of[r]
        # All currently-live rows are distinct.
        assert len(set(live.values())) == len(live)
        # Stamp each row so a later growth reallocation must preserve it.
        for r, s in live.items():
            cache.layer_state(0).k[s].fill_(float(r) + 1.0)
    for r, s in live.items():
        assert cache._slot_of[r] == s
        assert torch.all(cache.layer_state(0).k[s] == float(r) + 1.0)


def test_write_state_indices_stable_pointer_refreshed_contents():
    """The eager per-forward slot write is the CUDA-graph-safety contract.

    A captured decode graph aliases ``state_indices`` (via the ``gen_indices``
    view), so ``write_state_indices`` must (1) keep a STABLE pointer across
    forwards once the pool no longer grows -- else replay reads a stranded
    buffer -- while (2) REFRESHING the contents to the current batch's rows every
    call -- else replay reuses stale capture-time slots and decodes the wrong
    per-request conv windows. It must also (3) refuse to grow under a graph
    forward, loudly, instead of silently stranding the captured pointer.
    """
    device = torch.device("cpu")
    cache = InklingConvStateCache(_fake_model_config(), 4, device,
                                  dtype=torch.float32)

    # (1)+(2): steady-state graph forwards -- stable pointer, fresh contents.
    slots_a = cache.write_state_indices([100, 101, 102, 103], is_graph=True)
    ptr = cache.state_indices.data_ptr()
    assert torch.equal(cache.state_indices[:4],
                       torch.tensor(slots_a, dtype=torch.int32))
    # A different batch (subset, reordered) refreshes state_indices in place...
    slots_b = cache.write_state_indices([103, 101], is_graph=True)
    assert slots_b == [cache._slot_of[103], cache._slot_of[101]]
    assert torch.equal(cache.state_indices[:2],
                       torch.tensor(slots_b, dtype=torch.int32))
    # ...without moving the buffer a captured graph aliases.
    assert cache.state_indices.data_ptr() == ptr

    # (3): a graph forward that would oversubscribe raises, not silently grows.
    with pytest.raises(RuntimeError, match="CUDA graph"):
        cache.write_state_indices([100, 101, 102, 103, 200], is_graph=True)
    # The eager estimation/warmup window (is_graph=False) may still grow.
    grown = cache.write_state_indices([100, 101, 102, 103, 200],
                                      is_graph=False)
    assert len(set(grown)) == 5
    assert cache.max_batch_size >= 5
    assert cache.state_indices_cpu.numel() == cache.state_indices.numel()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
