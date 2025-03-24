import torch

from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType


def create_gated_mlp_example():
    # Configuration
    hidden_size = 64
    intermediate_size = hidden_size * 4
    batch_size = 1
    seq_len = 16
    dtype = torch.float16
    device = torch.device('cuda')

    # Create GatedMLP module
    gated_mlp = GatedMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bias=True,
        dtype=dtype,
        layer_idx=0  # Important for LoRA
    ).to(device)

    # Create input tensor
    hidden_states = torch.randn(size=[batch_size, seq_len, hidden_size],
                                dtype=dtype,
                                device=device)

    # Create LoRA parameters
    lora_rank = 8

    # Create weights for gate projection
    gate_weight_in = torch.randn(hidden_size,
                                 lora_rank,
                                 device=device,
                                 dtype=dtype).T
    gate_weight_out = torch.randn(
        lora_rank,
        intermediate_size,  # Gate projection size
        device=device,
        dtype=dtype).T

    # Create weights for up projection
    up_weight_in = torch.randn(hidden_size,
                               lora_rank,
                               device=device,
                               dtype=dtype).T
    up_weight_out = torch.randn(
        lora_rank,
        intermediate_size,  # Up projection size
        device=device,
        dtype=dtype).T

    # Create weights for down projection
    down_weight_in = torch.randn(intermediate_size,
                                 lora_rank,
                                 device=device,
                                 dtype=dtype).T
    down_weight_out = torch.randn(lora_rank,
                                  hidden_size,
                                  device=device,
                                  dtype=dtype).T

    # Make weights contiguous
    gate_weight_in = gate_weight_in.contiguous()
    gate_weight_out = gate_weight_out.contiguous()
    up_weight_in = up_weight_in.contiguous()
    up_weight_out = up_weight_out.contiguous()
    down_weight_in = down_weight_in.contiguous()
    down_weight_out = down_weight_out.contiguous()

    # Create LoRA parameters dictionary
    lora_params = {
        'num_seqs': batch_size,
        'host_request_types': torch.zeros(batch_size, dtype=torch.int32),
        'prompt_lens_cpu': torch.tensor([seq_len] * batch_size),
        0: {  # layer_idx
            LoraModuleType.MLP_H_TO_4H: {  # Up projection
                'adapter_size':
                torch.tensor([lora_rank]),
                'weight_pointers':
                torch.tensor(
                    [[up_weight_out.data_ptr(),
                      up_weight_in.data_ptr()]]),
                'is_dora':
                False,
                'weight_tensors': [up_weight_out, up_weight_in]
            },
            LoraModuleType.MLP_GATE: {  # Gate projection
                'adapter_size':
                torch.tensor([lora_rank]),
                'weight_pointers':
                torch.tensor(
                    [[gate_weight_out.data_ptr(),
                      gate_weight_in.data_ptr()]]),
                'is_dora':
                False,
                'weight_tensors': [gate_weight_out, gate_weight_in]
            },
            LoraModuleType.MLP_4H_TO_H: {  # Down projection
                'adapter_size':
                torch.tensor([lora_rank]),
                'weight_pointers':
                torch.tensor(
                    [[down_weight_out.data_ptr(),
                      down_weight_in.data_ptr()]]),
                'is_dora':
                False,
                'weight_tensors': [down_weight_out, down_weight_in]
            }
        }
    }

    # Run forward pass
    output = gated_mlp(
        hidden_states.squeeze(
            0),  # Remove batch dimension as expected by the module
        lora_params=lora_params)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")

    return output


if __name__ == "__main__":
    output = create_gated_mlp_example()
