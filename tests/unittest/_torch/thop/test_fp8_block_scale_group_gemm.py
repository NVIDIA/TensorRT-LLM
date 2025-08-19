import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import math
import random

import pytest
from _torch.helpers import per_block_cast_to_fp8

from tensorrt_llm._torch.custom_ops.torch_custom_ops import \
    cute_dsl_fp8_group_gemm_blackwell
from tensorrt_llm._utils import get_sm_version


def cute_dsl_fp8_group_blockwise_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    offset_array: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape[0], a.shape[1]
    l, n, k = b.shape[0], b.shape[1], b.shape[2]
    num_group, w_n, w_k = b_sf.shape[0], b_sf.shape[1], b_sf.shape[2]

    # Note: view(int8) will cause error.
    a_tmp = a.as_strided((m, k, 1), (k, 1, m * k))
    b_tmp = b.permute(1, 2, 0)

    # Note: we have different output scale shape for fp8_quantize_1x128, so we need to handle it differently for sm100 and other archs.
    if get_sm_version() == 100:
        input_scale_tmp = a_sf.permute(1, 0).as_strided((m, w_k, 1),
                                                        (1, m, m * w_k))
    else:
        m_padded = (m + 3) // 4 * 4
        input_scale_tmp = a_sf[0:m_padded * w_k]
        input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
        input_scale_tmp = input_scale_tmp[:w_k, :m].contiguous().permute(1, 0)
        input_scale_tmp = input_scale_tmp.as_strided((m, w_k, 1),
                                                     (1, m, m * w_k))

    weight_scale_tmp = b_sf.permute(1, 2, 0)

    def pad_and_multiply(scale, tensor):
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape
        IsGroupWise = False
        IsBlockWise = False
        if ck == math.ceil(k / 128):
            IsGroupWise = True
        if cm == math.ceil(m / 128):
            IsBlockWise = True
        if not IsBlockWise and not IsGroupWise:
            raise ValueError("Only support granularity = 128")

        k_idx = torch.arange(k, device=scale.device)
        if IsGroupWise:
            k_idx = k_idx // 128
        m_idx = torch.arange(m, device=scale.device)
        if IsBlockWise:
            m_idx = m_idx // 128
        expanded_scale = scale[m_idx[:, None], k_idx, :]

        result = expanded_scale * tensor

        return result

    updated_a = pad_and_multiply(input_scale_tmp, a_tmp.to(torch.float32))
    updated_b = pad_and_multiply(weight_scale_tmp, b_tmp.to(torch.float32))

    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)

    len_offset_array = offset_array.shape[0]
    for i in range(len_offset_array - 1):
        start = offset_array[i]
        end = offset_array[i + 1]
        ref[start:end, :] = torch.einsum("mk,nk->mn", updated_a[start:end, :,
                                                                0],
                                         updated_b[:, :, i])
    ref = ref.to(torch.bfloat16)
    return ref


@pytest.mark.parametrize("num_experts", [72])
@pytest.mark.parametrize("k", [1536])
@pytest.mark.parametrize("n", [2560])
@pytest.mark.parametrize("max_tokens_per_group",
                         [10, 50, 100, 128, 256, 512, 1000, 1024])
def test_cute_dsl_fp8_block_scale_group_gemm(num_experts, k, n,
                                             max_tokens_per_group):
    torch.manual_seed(0)

    group_m = []
    for i in range(num_experts):
        group_m.append(random.randint(0, max_tokens_per_group))
    group_m = torch.tensor(group_m, dtype=torch.int32, device="cuda")

    offset_group = torch.cumsum(group_m, dim=0)
    offset_group = torch.cat(
        [torch.tensor([0], dtype=torch.int32, device="cuda"), offset_group],
        dim=0)
    offset_group = offset_group.to(torch.int32)

    m = sum(group_m)
    a = torch.empty(m, k, dtype=torch.uint8).to(torch.bfloat16).cuda().normal_(
        0, 1) * 0.1
    b = torch.empty(num_experts, n, k, dtype=torch.uint8).to(
        torch.bfloat16).cuda().normal_(0, 1) * 0.1

    a_fp8, a_scale = torch.ops.trtllm.fp8_quantize_1x128(a)

    b_fp8 = torch.empty(num_experts,
                        n,
                        k,
                        dtype=torch.float8_e4m3fn,
                        device="cuda")
    b_scale = torch.empty(num_experts,
                          math.ceil(n / 128),
                          math.ceil(k / 128),
                          dtype=torch.float32,
                          device="cuda")
    for i in range(num_experts):
        cur_b, cur_b_scale = per_block_cast_to_fp8(b[i, :, :])
        b_fp8[i, :, :] = cur_b
        b_scale[i, :, :] = cur_b_scale

    c_actual_ref = cute_dsl_fp8_group_blockwise_gemm_ref(
        a=a_fp8, b=b_fp8, a_sf=a_scale, b_sf=b_scale, offset_array=offset_group)

    c_actual = cute_dsl_fp8_group_gemm_blackwell(input=a_fp8,
                                                 weight=b_fp8,
                                                 input_scale=a_scale,
                                                 weight_scale=b_scale,
                                                 group_offset=offset_group)

    c_ref = torch.zeros(m, n, dtype=torch.float32)

    def fp32_ref(num_experts, group_m, a, b, c_ref):
        start = 0
        for i in range(num_experts):
            end = start + group_m[i]
            c_ref[start:end, :] = torch.einsum("mk,nk->mn", a[start:end, :],
                                               b[i, :, :])
            start = end

    fp32_ref(num_experts, group_m, a.to(torch.float32), b.to(torch.float32),
             c_ref.to(torch.float32))
    c_ref = c_ref.to(torch.bfloat16)

    torch.set_printoptions(precision=4, sci_mode=False)
    print(
        f"torch.allclose(c_actual.cpu(), c_actual_ref.cpu(), atol=1e-1) = {torch.allclose(c_actual.cpu(), c_actual_ref.cpu(), atol=1e-1)}"
    )
    print("torch.allclose(c_actual_ref.cpu(), c_ref.cpu(), atol=1e-1) = ",
          torch.allclose(c_actual_ref.cpu(), c_ref.cpu(), atol=1e-1))
    print("torch.allclose(c_actual.cpu(), c_ref.cpu(), atol=1e-1) = ",
          torch.allclose(c_actual.cpu(), c_ref.cpu(), atol=1e-1))
    torch.testing.assert_close(c_actual.cpu(),
                               c_ref.cpu(),
                               atol=0.1,
                               rtol=1e-03)
    torch.testing.assert_close(c_actual.cpu(),
                               c_actual_ref.cpu(),
                               atol=0.1,
                               rtol=1e-03)
    print("PASS")


if __name__ == "__main__":
    test_cute_dsl_fp8_block_scale_group_gemm(72, 1536, 2560, 1024)
