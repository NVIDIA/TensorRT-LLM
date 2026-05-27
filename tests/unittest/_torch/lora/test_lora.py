import torch

from tensorrt_llm._torch.peft.lora.adapter_slot_manager import AdapterSlotManager
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_params import CudaGraphLoraParams


def test_cuda_graph_lora_params_handle_missing_peft_table():
    layer_key = CudaGraphLoraParams.LoraLayerKey(layer_idx=0, module_ids=(1, 2))
    layer_info = {layer_key: CudaGraphLoraParams.LoraLayerInfo(module_num=2, output_sizes=[16, 32])}
    params = CudaGraphLoraParams(
        max_batch_size=2, max_lora_size=2, max_rank=8, layer_info=layer_info
    )
    layer_params = params.layer_params[layer_key]

    layer_params.h_b_ptrs[:, 0] = torch.tensor([11, 22], dtype=torch.int64)
    layer_params.h_b_prime_ptrs[:, 0] = torch.tensor([33, 44], dtype=torch.int64)
    layer_params.h_b_ptrs[:, 1] = torch.tensor([55, 66], dtype=torch.int64)
    layer_params.h_b_prime_ptrs[:, 1] = torch.tensor([77, 88], dtype=torch.int64)
    params.slot_ranks_host[:] = torch.tensor([4, 7], dtype=torch.int32)

    params.update_weight_pointers(None, (123, None))

    assert params.slot_ranks_host.tolist() == [4, 0]
    assert layer_params.h_b_ptrs[:, 0].tolist() == [11, 22]
    assert layer_params.h_b_prime_ptrs[:, 0].tolist() == [33, 44]
    assert layer_params.h_b_ptrs[:, 1].tolist() == [0, 0]
    assert layer_params.h_b_prime_ptrs[:, 1].tolist() == [0, 0]


def test_adapter_slot_manager_handles_missing_peft_cache_manager():
    manager = AdapterSlotManager(max_num_adapters=2)
    manager.slot2task[0] = 123
    manager.task2slot[123] = 0

    manager.remove_evicted_slots_in_cpp(None)

    assert manager.get_slot_to_task_mapping() == (123, None)
    assert manager.task2slot[123] == 0


def test_cuda_graph_lora_params_token_to_slot_population():
    """update_sorted_indices must populate token_to_slot_host so that
    token_to_slot_host[token_idx] == slot id of the sequence that token belongs to.
    This is the inverse map used by the routed-expert MoE LoRA path."""
    layer_key = CudaGraphLoraParams.LoraLayerKey(layer_idx=0, module_ids=(1, 2))
    layer_info = {layer_key: CudaGraphLoraParams.LoraLayerInfo(module_num=2, output_sizes=[16, 32])}
    params = CudaGraphLoraParams(
        max_batch_size=4, max_lora_size=4, max_rank=8, layer_info=layer_info,
        max_tokens_per_seq=1,
    )

    params.update_sorted_indices([0, 2, 1, 3], tokens_per_seq=1)
    assert params.token_to_slot_host[:4].tolist() == [0, 2, 1, 3]


def test_cuda_graph_lora_params_token_to_slot_population_spec_decode():
    """For tokens_per_seq>1 each sequence contributes tokens_per_seq tokens all
    carrying the same slot id."""
    layer_key = CudaGraphLoraParams.LoraLayerKey(layer_idx=0, module_ids=(1,))
    layer_info = {layer_key: CudaGraphLoraParams.LoraLayerInfo(module_num=1, output_sizes=[16])}
    params = CudaGraphLoraParams(
        max_batch_size=2, max_lora_size=4, max_rank=8, layer_info=layer_info,
        max_tokens_per_seq=3,
    )

    params.update_sorted_indices([1, 3], tokens_per_seq=3)
    assert params.token_to_slot_host[:6].tolist() == [1, 1, 1, 3, 3, 3]


def test_cuda_graph_lora_params_get_moe_slot_inputs_returns_packed_view():
    """get_moe_slot_inputs must pack [A_ptr, B_ptr, 0] into a [max_lora_size, 3]
    tensor whose address is stable across calls (so it can be safely captured
    in a CUDA graph)."""
    layer_key = CudaGraphLoraParams.LoraLayerKey(layer_idx=5, module_ids=(7, 8))
    layer_info = {layer_key: CudaGraphLoraParams.LoraLayerInfo(module_num=2, output_sizes=[16, 32])}
    params = CudaGraphLoraParams(
        max_batch_size=2, max_lora_size=3, max_rank=8, layer_info=layer_info,
    )

    layer_params = params.layer_params[layer_key]
    layer_params.h_b_ptrs[0, :] = torch.tensor([10, 20, 30], dtype=torch.int64)
    layer_params.h_b_prime_ptrs[0, :] = torch.tensor([100, 200, 300], dtype=torch.int64)
    layer_params.h_b_ptrs[1, :] = torch.tensor([40, 50, 60], dtype=torch.int64)
    layer_params.h_b_prime_ptrs[1, :] = torch.tensor([400, 500, 600], dtype=torch.int64)
    params.slot_ranks_host[:] = torch.tensor([4, 7, 0], dtype=torch.int32)

    out1 = params.get_moe_slot_inputs(layer_idx=5, module_id=7)
    assert out1 is not None
    ranks, ptrs = out1
    assert ranks.tolist() == [4, 7, 0]
    assert ptrs.shape == (3, 3)
    assert ptrs[:, 0].tolist() == [10, 20, 30]
    assert ptrs[:, 1].tolist() == [100, 200, 300]
    assert ptrs[:, 2].tolist() == [0, 0, 0]

    # Stability: a second call for the same (layer, module) must return the
    # same packed buffer object (same .data_ptr()), so it is graph-capture safe.
    out2 = params.get_moe_slot_inputs(layer_idx=5, module_id=7)
    assert out2 is not None
    assert out2[1].data_ptr() == ptrs.data_ptr()


def test_cuda_graph_lora_params_get_moe_slot_inputs_missing_module():
    """Querying an (layer, module) that is not in the layer map must return None."""
    layer_key = CudaGraphLoraParams.LoraLayerKey(layer_idx=0, module_ids=(1,))
    layer_info = {layer_key: CudaGraphLoraParams.LoraLayerInfo(module_num=1, output_sizes=[16])}
    params = CudaGraphLoraParams(
        max_batch_size=1, max_lora_size=1, max_rank=4, layer_info=layer_info,
    )
    assert params.get_moe_slot_inputs(layer_idx=0, module_id=999) is None
    assert params.get_moe_slot_inputs(layer_idx=42, module_id=1) is None
