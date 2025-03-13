import pytest
import torch
from _custom_op_utils import torch_rope_reference

from tensorrt_llm._torch.auto_deploy.custom_ops.triton_attention import TritonWithFlattenedInputs


@pytest.mark.parametrize("d_head", [16, 96])
def test_rope(d_head):
    SEQ_LEN = 4
    N_ELEM = d_head

    input_position = torch.tensor([10], dtype=torch.int32, device="cuda")
    freqs_cis = TritonWithFlattenedInputs._precompute_freqs_cis(1024, N_ELEM)
    print(freqs_cis.shape)

    x = torch.randn((1, SEQ_LEN, 8, N_ELEM), dtype=torch.float16)
    y_ref = torch_rope_reference(x, freqs_cis, input_position)
    freqs_cis = freqs_cis.to("cuda")
    x_reshaped = x.unflatten(-1, (N_ELEM // 2, 2)).transpose(-1, -2).flatten(-2).contiguous()
    y = torch.ops.rope.apply_rope_with_input_pos(
        x_reshaped.to("cuda"), freqs_cis, input_position, "bsnd"
    )
    y_reshaped = y.unflatten(-1, (2, N_ELEM // 2)).transpose(-2, -1).flatten(-2).contiguous()
    assert torch.allclose(y_ref.cpu(), y_reshaped.cpu(), atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("d_head", [16, 96])
def test_rope_flattened(d_head):
    SEQ_LENS = [4, 16, 28]
    N_ELEM = d_head

    freqs_cis = TritonWithFlattenedInputs._precompute_freqs_cis(1024, N_ELEM).to("cuda")

    input_position = torch.tensor([0] * len(SEQ_LENS), device="cuda")
    x = []
    y_ref = []
    for i, s in enumerate(SEQ_LENS):
        tmp = torch.randn((1, s, 8, N_ELEM), dtype=torch.float16, device="cuda")
        y_ref.append(torch_rope_reference(tmp, freqs_cis, input_position[i]))
        x.append(tmp)
    x = torch.cat(x, 1).squeeze()  # [B*S,...]
    y_ref = torch.cat(y_ref, 1).squeeze()

    x_reshaped = x.unflatten(-1, (N_ELEM // 2, 2)).transpose(-1, -2).flatten(-2).contiguous()

    seq_lens = torch.tensor(SEQ_LENS, device="cuda")
    seq_start_indices = torch.zeros(len(SEQ_LENS), dtype=torch.int32, device="cuda")
    seq_start_indices[1:] = torch.cumsum(seq_lens[:-1], 0)

    y = torch.ops.rope.apply_rope_on_flattened_inputs(
        x_reshaped.to("cuda"), freqs_cis, input_position, seq_lens, seq_start_indices
    )
    y_reshaped = y.unflatten(-1, (2, N_ELEM // 2)).transpose(-2, -1).flatten(-2).contiguous()

    assert torch.allclose(y_ref.cpu(), y_reshaped.cpu(), atol=1e-2, rtol=1e-2)
