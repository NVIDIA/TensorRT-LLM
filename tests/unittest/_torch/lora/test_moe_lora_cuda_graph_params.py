# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the routed-expert MoE LoRA *CUDA-graph slot-table*
Python wiring.

The op-level tests (`test_moe_lora_op.py`, `test_moe_lora_grouped_gemm.py`)
exercise `torch.ops.trtllm.fused_moe` with hand-built slot kwargs, bypassing the
plumbing that actually produces those kwargs at decode time. These tests cover
that glue end-to-end on the Python side, without model weights or the built C++
op:

  * `CudaGraphLoraParams.get_moe_slot_inputs` packs the per-module
    (A, B, dora) slot-pointer table and aliases the shared per-slot rank table.
  * `CutlassFusedMoE._extract_moe_lora_tensors_cuda_graph` threads those tables
    plus `token_to_slot` into the slot-indexed kwargs the op consumes, applying
    the moe_h_to_4h->fc1 / moe_gate->gated / moe_4h_to_h->fc2 mapping and the
    global `max_rank`.

`CudaGraphLoraParams` allocates pinned host buffers and moves a small tensor to
CUDA at construction, so these require a GPU, but no weights and no
`trtllm::fused_moe` op.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_params import CudaGraphLoraParams
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CudaGraphLoraParams allocates pinned/CUDA buffers; requires a GPU.",
)

_HIDDEN = 128
_INTER = 256


class _ExtractStub:
    """Minimal stand-in for a CutlassFusedMoE so the unbound extraction method
    can run without constructing a full MoE layer. The method reads
    `self.layer_idx` and the shared slot-gathering helpers, which we borrow from
    CutlassFusedMoE."""

    _gather_moe_lora_slots = CutlassFusedMoE._gather_moe_lora_slots
    _empty_kernel_slot_dict = staticmethod(CutlassFusedMoE._empty_kernel_slot_dict)

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx


def _make_params(max_lora_size=2, max_rank=8, max_batch_size=2, layer_idx=0):
    """Build a CudaGraphLoraParams carrying one MoE layer with all three
    routed-expert modules (fc1=moe_h_to_4h, gated=moe_gate, fc2=moe_4h_to_h)."""
    module_ids = (
        int(LoraModuleType.MOE_H_TO_4H),
        int(LoraModuleType.MOE_GATE),
        int(LoraModuleType.MOE_4H_TO_H),
    )
    key = CudaGraphLoraParams.LoraLayerKey(layer_idx=layer_idx, module_ids=module_ids)
    layer_info = {
        key: CudaGraphLoraParams.LoraLayerInfo(
            module_num=3,
            # fc1/gated produce intermediate_size, fc2 produces hidden_size.
            output_sizes=[_INTER, _INTER, _HIDDEN],
        )
    }
    params = CudaGraphLoraParams(
        max_batch_size=max_batch_size,
        max_lora_size=max_lora_size,
        max_rank=max_rank,
        layer_info=layer_info,
    )
    return params, key, module_ids


def _fake_ptr(module_id, slot, side):
    """Distinct non-zero int64 pointer bits per (module, slot, A/B side)."""
    return (int(module_id) << 40) | (int(slot) << 8) | int(side)


def _populate_pointers(params, key, module_ids, rank):
    """Fill per-module A/B host pointer tables and the shared per-slot rank
    table, mimicking what the PEFT cache manager would do for occupied slots."""
    layer_param = params.layer_params[key]
    for local_id, module_id in enumerate(module_ids):
        for slot in range(params.max_lora_size):
            layer_param.h_b_ptrs[local_id, slot] = _fake_ptr(module_id, slot, 0)
            layer_param.h_b_prime_ptrs[local_id, slot] = _fake_ptr(module_id, slot, 1)
    for slot in range(params.max_lora_size):
        params.slot_ranks_host[slot] = rank


@requires_cuda
def test_get_moe_slot_inputs_packs_pointers_and_aliases_ranks():
    rank = 8
    params, key, module_ids = _make_params(max_lora_size=2, max_rank=rank)
    _populate_pointers(params, key, module_ids, rank)

    for module_id in module_ids:
        out = params.get_moe_slot_inputs(layer_idx=0, module_id=module_id)
        assert out is not None
        ranks_host, packed = out
        # Ranks are the shared per-slot table aliased (not a per-module copy).
        assert ranks_host is params.slot_ranks_host
        assert packed.shape == (params.max_lora_size, 3)
        for slot in range(params.max_lora_size):
            assert packed[slot, 0].item() == _fake_ptr(module_id, slot, 0)  # A
            assert packed[slot, 1].item() == _fake_ptr(module_id, slot, 1)  # B
            assert packed[slot, 2].item() == 0  # dora always zero

    # Unknown (layer, module) returns None rather than raising.
    assert (
        params.get_moe_slot_inputs(layer_idx=0, module_id=int(LoraModuleType.ATTENTION_QKV)) is None
    )
    assert (
        params.get_moe_slot_inputs(layer_idx=99, module_id=int(LoraModuleType.MOE_H_TO_4H)) is None
    )


@requires_cuda
def test_get_moe_slot_inputs_packed_buffer_is_address_stable():
    """The packed (A, B, dora) buffer must be cached per (layer, module) so its
    data_ptr() is stable across calls, which is required for CUDA-graph capture."""
    rank = 4
    params, key, module_ids = _make_params(max_lora_size=2, max_rank=rank)
    _populate_pointers(params, key, module_ids, rank)

    mid = int(LoraModuleType.MOE_H_TO_4H)
    _, packed_first = params.get_moe_slot_inputs(layer_idx=0, module_id=mid)
    first_ptr = packed_first.data_ptr()

    # Reassign a slot's pointers in place and re-extract; same backing buffer.
    params.layer_params[key].h_b_ptrs[0, 0] = _fake_ptr(mid, 0, 0) + 777
    _, packed_second = params.get_moe_slot_inputs(layer_idx=0, module_id=mid)
    assert packed_second.data_ptr() == first_ptr
    assert packed_second[0, 0].item() == _fake_ptr(mid, 0, 0) + 777


@requires_cuda
def test_extract_moe_lora_tensors_cuda_graph_wires_slot_tables():
    """End-to-end Python wiring: get_moe_slot_inputs -> the slot-indexed kwargs
    that torch.ops.trtllm.fused_moe consumes."""
    rank = 8
    num_seqs = 2
    params, key, module_ids = _make_params(max_lora_size=2, max_rank=rank, max_batch_size=num_seqs)
    _populate_pointers(params, key, module_ids, rank)
    # token 0 -> slot 1, token 1 -> slot 0.
    params.update_sorted_indices([1, 0])

    lora_params = {
        "use_cuda_graph_mode": True,
        "cuda_graph_params": params,
        "num_seqs": num_seqs,
    }
    kwargs = CutlassFusedMoE._extract_moe_lora_tensors_cuda_graph(_ExtractStub(0), lora_params)
    assert kwargs is not None

    # max_rank is the global cap, not a per-step value.
    assert kwargs["lora_max_low_rank"] == rank

    # All three modules present, mapped to their kernel slots.
    for kernel, module_id in (
        ("fc1", int(LoraModuleType.MOE_H_TO_4H)),
        ("gated", int(LoraModuleType.MOE_GATE)),
        ("fc2", int(LoraModuleType.MOE_4H_TO_H)),
    ):
        ranks = kwargs[f"{kernel}_slot_lora_ranks"]
        ptrs = kwargs[f"{kernel}_slot_lora_weight_ptrs"]
        assert ranks.shape == (params.max_lora_size,)
        assert ptrs.shape == (params.max_lora_size, 3)
        for slot in range(params.max_lora_size):
            assert ranks[slot].item() == rank
            assert ptrs[slot, 0].item() == _fake_ptr(module_id, slot, 0)
            assert ptrs[slot, 1].item() == _fake_ptr(module_id, slot, 1)

    # token_to_slot is sliced to the live token count and matches the routing.
    token_to_slot = kwargs["token_to_slot"]
    assert token_to_slot.shape == (num_seqs,)
    assert token_to_slot[0].item() == 1
    assert token_to_slot[1].item() == 0


@requires_cuda
def test_extract_returns_none_without_layer_or_required_modules():
    rank = 8
    params, _, _ = _make_params(max_rank=rank)

    # layer_idx None -> no MoE LoRA for this layer.
    none_layer = CutlassFusedMoE._extract_moe_lora_tensors_cuda_graph(
        _ExtractStub(None), {"cuda_graph_params": params, "num_seqs": 1}
    )
    assert none_layer is None

    # No cuda_graph_params -> None.
    assert (
        CutlassFusedMoE._extract_moe_lora_tensors_cuda_graph(_ExtractStub(0), {"num_seqs": 1})
        is None
    )

    # A layer that only carries fc1 (missing fc2) must bail out: the in-GEMM and
    # out-GEMM are both required for a routed-expert LoRA delta.
    fc1_only_key = CudaGraphLoraParams.LoraLayerKey(
        layer_idx=0, module_ids=(int(LoraModuleType.MOE_H_TO_4H),)
    )
    fc1_only = CudaGraphLoraParams(
        max_batch_size=1,
        max_lora_size=1,
        max_rank=rank,
        layer_info={
            fc1_only_key: CudaGraphLoraParams.LoraLayerInfo(module_num=1, output_sizes=[_INTER])
        },
    )
    assert (
        CutlassFusedMoE._extract_moe_lora_tensors_cuda_graph(
            _ExtractStub(0), {"cuda_graph_params": fc1_only, "num_seqs": 1}
        )
        is None
    )
