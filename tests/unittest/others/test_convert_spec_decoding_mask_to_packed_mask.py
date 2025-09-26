import pytest
import torch

import tensorrt_llm  # noqa


@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("max_draft_tokens", [20, 41, 101])
def test_convert_mask_to_packed_mask(batch_size: int, max_draft_tokens: int):
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(1234)
    device = 'cuda'
    num_packed_mask = (max_draft_tokens + 1 + 31) // 32
    spec_decoding_generation_lengths_tensor = torch.randint(1,
                                                            max_draft_tokens,
                                                            (batch_size, ),
                                                            device=device,
                                                            dtype=torch.int32)
    spec_decoding_mask_tensor = torch.rand(
        (batch_size, max_draft_tokens + 1, max_draft_tokens + 1),
        device=device) < 0.8
    spec_decoding_packed_mask_tensor = torch.zeros(
        (batch_size * (max_draft_tokens + 1), num_packed_mask),
        device=device,
        dtype=torch.int32)
    spec_decoding_packed_mask_tensor_ref = spec_decoding_packed_mask_tensor.detach(
    ).clone()

    torch.ops.tensorrt_llm.convert_spec_decoding_mask_to_packed_mask(
        spec_decoding_generation_lengths_tensor, spec_decoding_mask_tensor,
        max_draft_tokens, spec_decoding_packed_mask_tensor, None)
    torch.cuda.synchronize()

    def get_packed_mask(num_draft_tokens,
                        spec_decoding_mask,
                        max_draft_tokens=None):
        max_draft_tokens = max_draft_tokens or num_draft_tokens
        num_packed_masks = (max_draft_tokens + 1 + 32 - 1) // 32
        spec_decoding_packed_mask = torch.zeros(
            (num_draft_tokens + 1, num_packed_masks), dtype=torch.int32)
        for token_idx in range(num_draft_tokens + 1):
            if token_idx == 0:
                spec_decoding_packed_mask[0, 0] = 1
            else:
                mask_list = spec_decoding_mask[token_idx - 1, :].tolist()
                # insert 1 as there is one extra new token from the original lm head.
                mask_list.insert(0, True)
                # convert binary bits into 4 int32_t
                mask_str_list = [str(int(val)) for val in mask_list]
                mask_str_list.reverse()

                for mask_idx in range(num_packed_masks):
                    if mask_idx * 32 >= len(mask_str_list):
                        break
                    mask_32bits_str = ''.join(
                        mask_str_list[-(mask_idx + 1) * 32:(-mask_idx * 32 -
                                                            1)] +
                        [mask_str_list[(-mask_idx * 32 - 1)]])
                    valid_num_bits = len(mask_32bits_str)
                    first_bit1 = mask_32bits_str[0] == '1'
                    mask_31bits_str = mask_32bits_str[1:]
                    mask_31bits = 0 if mask_31bits_str == "" else int(
                        mask_31bits_str, 2)
                    if valid_num_bits == 32:
                        mask_32bits = mask_31bits - first_bit1 * (2**(
                            valid_num_bits - 1))
                    else:
                        mask_32bits = mask_31bits + first_bit1 * (2**(
                            valid_num_bits - 1))
                    spec_decoding_packed_mask[token_idx, mask_idx] = mask_32bits
        return spec_decoding_packed_mask

    def convert_spec_decoding_mask_to_packed_mask(
            batch_size, spec_decoding_generation_lengths, spec_decoding_mask,
            max_draft_tokens, spec_decoding_packed_mask):
        # for now just do with pytorch, we may need to write a custom kernel
        offset = 0
        max_gen_len = max(spec_decoding_generation_lengths)
        # spec_decoding_mask is populated inside engine as [bs, max_gen_len, max_gen_len]
        masks = spec_decoding_mask.view([-1])[:batch_size * max_gen_len *
                                              max_gen_len]
        masks = masks.view([batch_size, max_gen_len, max_gen_len])
        for i in range(batch_size):
            cur_tokens = spec_decoding_generation_lengths[i]
            m = masks[i]
            pm = get_packed_mask(cur_tokens - 1, m[1:, 1:], max_draft_tokens)
            assert spec_decoding_packed_mask.shape[-1] == pm.shape[-1], \
                f"{spec_decoding_packed_mask.shape[-1]} != {pm.shape[-1]} for packed mask length"
            spec_decoding_packed_mask[offset:offset + cur_tokens, :] = pm
            offset += cur_tokens
        return

    convert_spec_decoding_mask_to_packed_mask(
        batch_size, spec_decoding_generation_lengths_tensor,
        spec_decoding_mask_tensor, max_draft_tokens,
        spec_decoding_packed_mask_tensor_ref)
    torch.cuda.synchronize()
    assert torch.equal(spec_decoding_packed_mask_tensor,
                       spec_decoding_packed_mask_tensor_ref)
