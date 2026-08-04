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
