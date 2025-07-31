import os
import sys

import nvtx
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import math
import random

import pytest
from _torch.helpers import per_block_cast_to_fp8

# from tensorrt_llm._torch.custom_ops.cute_dsl_ops import cute_dsl_fp8_group_blockwise_gemm_ref
from tensorrt_llm._torch.custom_ops.torch_custom_ops import \
    cute_dsl_fp8_group_gemm_blackwell
from tensorrt_llm._utils import get_sm_version


def load_tensor_from_file(dir, filename, dtype):
    import os

    import numpy as np

    # Read original tensor
    path = os.path.join(dir, filename)

    # Check if files exist
    if not os.path.exists(path):
        print(f"Files do not exist: {path}")
        return

    # Read and parse tensor info
    data = np.load(path)

    # 将numpy数组转换为torch tensor
    tensor = torch.from_numpy(data).to(dtype)

    return tensor


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

    # print(f"limin: a.shape = {a.shape}, a.stride = {a.stride()}")
    # print(f"limin: b.shape = {b.shape}, b.stride = {b.stride()}")
    # print(f"limin: a_sf.shape = {a_sf.shape}, a_sf.stride = {a_sf.stride()}")
    # print(f"limin: b_sf.shape = {b_sf.shape}, b_sf.stride = {b_sf.stride()}")
    # print(f"limin: offset_array.shape = {offset_array.shape}, offset_array.stride = {offset_array.stride()}")

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
        # # assert start <= end, f"Invalid group boundaries: start={start} > end={end}"
        # if start <= 406 and end >= 406:
        #     print(f"limin: start = {start}, end = {end}")
        #     print(f"limin: updated_a[406, :, 0] = {updated_a[406, :, 0]}")
        #     print(f"limin: updated_b[:, 588, i] = {updated_b[:, 588, i]}")
        #     o = torch.dot(updated_a[406, :, 0], updated_b[588, :, i])
        #     # limin: torch.dot = 0.0071868896484375, 0.0072021484375
        #     print(f"limin: torch.dot = {o}, {o.to(torch.bfloat16)}")
        #     o1 = torch.einsum("k,k->", updated_a[406, :, 0], updated_b[588, :,
        #                                                                i])
        #     print(f"limin: o1 = {o1}, {o1.to(torch.bfloat16)}")
        # if start <= 3 and end >= 3:
        #     print(f"\nlimin: start = {start}, end = {end}")
        #     print(f"limin: updated_a[3, :, 0] = {updated_a[3, :, 0]}")
        #     print(f"limin: updated_b[:, 1509, i] = {updated_b[:, 1509, i]}")
        #     o = torch.dot(updated_a[3, :, 0], updated_b[1509, :, i])
        #     print(f"limin: torch.dot = {o}, {o.to(torch.bfloat16)}")
        #     o1 = torch.einsum("k,k->", updated_a[3, :, 0], updated_b[1509, :,
        #                                                              i])
        #     print(f"limin: o1 = {o1}, {o1.to(torch.bfloat16)}")
        # ref[start:end, :] = torch.einsum("mk,nk->mn", updated_a[start:end, :,
        #                                                         0],
        #                                  updated_b[:, :, i])
        ref[start:end, :] = torch.matmul(updated_a[start:end, :, 0],
                                         updated_b[:, :, i].T)
        # for m in range(start, end):
        #     for nn in range(n):
        #         ref[m, nn] = torch.dot(updated_a[m, :, 0], updated_b[nn, :, i])
    # limin: ref[406][588] = -0.11326146125793457
    # print(f"limin: ref[406][588] = {ref[406][588]}")
    # print(f"limin: ref[3][1509] = {ref[3][1509]}")
    ref = ref.to(torch.bfloat16)
    return ref


@pytest.mark.parametrize("num_experts", [72])
@pytest.mark.parametrize("k", [1536])
@pytest.mark.parametrize("n", [2560])
@pytest.mark.parametrize("max_tokens_per_group",
                         [10, 50, 100, 128, 256, 512, 1000, 1024])
def test_cute_dsl_group_gemm(num_experts, k, n, max_tokens_per_group):
    torch.manual_seed(0)
    # for i in range(num_experts):
    #     group_m.append(m_aligned * random.randint(
    #         int(expect_m * 0.1) // m_aligned, int(expect_m * 5.5) // m_aligned
    #     ))
    # for max_tokens_per_group in range(1, 1500, 2):
    for i in range(1):
        group_m = []
        for i in range(num_experts):
            group_m.append(random.randint(0, max_tokens_per_group))
        print("limin: group_m", group_m)
        group_m = torch.tensor(group_m, dtype=torch.int32, device="cuda")

        offset_group = torch.cumsum(group_m, dim=0)
        offset_group = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device="cuda"), offset_group],
            dim=0)
        offset_group = offset_group.to(torch.int32)
        print("limin: offset_group", offset_group)

        m = sum(group_m)
        print("limin: m", m)
        # TODO: how to initialize a and b
        # a = torch.ones(m, k, dtype=torch.bfloat16, device="cuda") * 0.1
        # b = torch.ones(num_experts, n, k, dtype=torch.bfloat16, device="cuda") * 0.1
        a = torch.empty(m, k, dtype=torch.uint8).to(
            torch.bfloat16).cuda().normal_(0, 1) * 0.1
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

        # print("limin: a_scale.shape", a_scale.shape)
        # c_actual = torch.ops.trtllm.cute_dsl_fp8_group_blockwise_gemm(a_fp8, b_fp8, a_scale.reshape(-1), b_scale, group_m)
        # c_actual_ref = cute_dsl_fp8_group_blockwise_gemm_ref(a_fp8, b_fp8, a_scale.reshape(-1), b_scale, group_m)
        c_actual_ref = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=a_fp8,
            b=b_fp8,
            a_sf=a_scale,
            b_sf=b_scale,
            offset_array=offset_group)

        for i in range(1):
            with nvtx.annotate("cute_dsl_fp8_group_gemm_blackwell",
                               color="red"):
                c_actual = cute_dsl_fp8_group_gemm_blackwell(
                    input=a_fp8,
                    weight=b_fp8,
                    input_scale=a_scale,
                    weight_scale=b_scale,
                    group_offset=offset_group)

        c_ref = torch.zeros(m, n, dtype=torch.float32)

        def ref(num_experts, group_m, a, b, c_ref):
            start = 0
            for i in range(num_experts):
                end = start + group_m[i]
                c_ref[start:end, :] = torch.einsum("mk,nk->mn", a[start:end, :],
                                                   b[i, :, :])
                start = end

        ref(num_experts, group_m, a.to(torch.float32), b.to(torch.float32),
            c_ref.to(torch.float32))
        c_ref = c_ref.to(torch.bfloat16)

        torch.set_printoptions(precision=4, sci_mode=False)
        print(f"limin: c_ref = {c_ref}")
        print(f"limin: c_actual = {c_actual}")
        print(f"limin: c_actual_ref = {c_actual_ref}")
        print(
            f"limin: torch.allclose(c_actual.cpu(), c_actual_ref.cpu(), atol=1e-1) = {torch.allclose(c_actual.cpu(), c_actual_ref.cpu(), atol=1e-1)}"
        )
        print(
            "limin: torch.allclose(c_actual_ref.cpu(), c_ref.cpu(), atol=1e-1) = ",
            torch.allclose(c_actual_ref.cpu(), c_ref.cpu(), atol=1e-1))
        print(
            "limin: torch.allclose(c_actual.cpu(), c_ref.cpu(), atol=1e-1) = ",
            torch.allclose(c_actual.cpu(), c_ref.cpu(), atol=1e-1))
        # assert torch.allclose(c_actual.cpu(), c_actual_ref.cpu(), atol=1e-2), "c_actual != c_actual_ref"
        # assert torch.allclose(c_actual.cpu(), c_expected.cpu(), atol=1e-2), "c_actual != c_expected"
        # assert torch.allclose(c_actual_ref.cpu(), c_expected.cpu(), atol=1e-2), "c_actual_ref != c_expected"
        # torch.testing.assert_close(c_actual.cpu(),
        #                            c_actual_ref.cpu(),
        #                            atol=0.1,
        #                            rtol=1e-03)
        # torch.testing.assert_close(c_actual_ref.cpu(), c_ref.cpu(), atol=0.1, rtol=1e-03)
        torch.testing.assert_close(c_actual.cpu(),
                                   c_ref.cpu(),
                                   atol=0.1,
                                   rtol=1e-03)
        print("PASS")


def optimized_fill_pattern_1(x, num_row, num_cols):
    """使用向量化操作加速"""
    # 创建网格坐标
    i_coords, j_coords = torch.meshgrid(torch.arange(num_row),
                                        torch.arange(num_cols),
                                        indexing='ij')

    # 计算基础值
    base_values = (i_coords * 0.1 + j_coords * 0.2) % 2

    # 计算sign模式：每3个元素翻转一次
    sign_pattern = torch.ones_like(base_values)
    element_indices = torch.arange(num_row * num_cols).reshape(
        num_row, num_cols)
    sign_flip_mask = (element_indices % 3 == 0)
    # sign_flip_mask = (element_indices % 5 == 0)
    sign_pattern[sign_flip_mask] = -1

    # 应用sign模式
    x[:] = base_values * sign_pattern


def set_tensor_value_2(x, num_row, num_cols):
    # # Create 2x2 base pattern matrix
    # pattern = torch.tensor([[0.2, -0.5], [-0.3, 0.1]], device=x.device)

    # # Repeat pattern to cover entire matrix
    # repeated = pattern.repeat((num_row + 1) // 2,
    #                           (num_cols + 1) // 2)[:num_row, :num_cols]

    # x.copy_(repeated)
    sign = -1
    sum = 0
    for i in range(num_row):
        for j in range(num_cols):
            x[i, j] = (i * 0.1 + j * 0.2) % 2 * sign * 0.1
            sum += 1
            if sum % 3 == 0:
                sign = -sign


def set_tensor_value_3(x, num_row, num_cols):
    # Create 3x3 base pattern matrix
    pattern = torch.tensor(
        [[0.1, 0.21, 0.31], [0.3, 0.6, 0.1], [0.11, 0.51, 0.62]],
        device=x.device)

    # Repeat pattern to cover entire matrix
    repeated = pattern.repeat((num_row + 2) // 3,
                              (num_cols + 2) // 3)[:num_row, :num_cols]

    x.copy_(repeated)


def set_tensor_value_4(x, num_row, num_cols):
    # Create 4x4 base pattern matrix
    pattern = torch.tensor(
        [
            [0.1, 0.21, 0.31, 0.41],
            [0.3, 0.6, 0.1, 0.2],
            [0.11, 0.51, 0.61, 0.71],
            [0.11, 0.52, 0.62, 0.72],
        ],
        device=x.device,
    )

    # Repeat pattern to cover entire matrix
    repeated = pattern.repeat((num_row + 3) // 4,
                              (num_cols + 3) // 4)[:num_row, :num_cols]

    x.copy_(repeated)


def test_cute_dsl_group_gemm_input_from_file():
    x = load_tensor_from_file("debug", "act_input_fp8.npy",
                              torch.float32).cuda().to(torch.float8_e4m3fn)
    # print("limin: x.shape", x.shape)
    # print("limin: x", x)
    w = load_tensor_from_file("debug", "w3_w1_weight.npy",
                              torch.float32).cuda().to(torch.float8_e4m3fn)
    # print("limin: w.shape", w.shape)
    # print("limin: w", w)
    w_sf = load_tensor_from_file("debug", "quant_scales_0.npy",
                                 torch.float32).cuda()
    # print("limin: w_sf.shape", w_sf.shape)
    # print("limin: w_sf", w_sf)
    offset_array = load_tensor_from_file(
        "debug", "expert_first_token_offset_tensor.npy", torch.int32).cuda()
    # print("limin: offset_array.shape", offset_array.shape)
    # print("limin: offset_array", offset_array)
    x_sf = load_tensor_from_file("debug", "act_input_sf.npy",
                                 torch.float32).cuda()
    # print("limin: x_sf.shape", x_sf.shape)
    # print("limin: x_sf", x_sf)

    # optimized_fill_pattern_1(x, x.shape[0], x.shape[1])
    # for i in range(w.shape[0]):
    #     optimized_fill_pattern_1(w[i, :, :], w.shape[1], w.shape[2])
    x = (x.to(torch.float32) * 0.01).to(torch.float8_e4m3fn)
    w = (w.to(torch.float32) * 0.012).to(torch.float8_e4m3fn)
    # x = x.fill_(1.1)
    # w = w.fill_(1.2)
    optimized_fill_pattern_1(x_sf, x_sf.shape[0], x_sf.shape[1])
    for i in range(w_sf.shape[0]):
        optimized_fill_pattern_1(w_sf[i, :, :], w_sf.shape[1], w_sf.shape[2])

    # print("limin: x", x.shape)
    # print("limin: w", w.shape)
    # print("limin: x_sf", x_sf.shape)
    # print("limin: w_sf", w_sf.shape)

    print(
        f"limin: torch.cuda.memory_allocated() = {torch.cuda.memory_allocated()}"
    )

    h1 = cute_dsl_fp8_group_blockwise_gemm_ref(a=x,
                                               b=w,
                                               a_sf=x_sf,
                                               b_sf=w_sf,
                                               offset_array=offset_array)
    torch.cuda.empty_cache()
    for i in range(10):
        h2 = cute_dsl_fp8_group_gemm_blackwell(input=x,
                                               weight=w,
                                               input_scale=x_sf,
                                               weight_scale=w_sf,
                                               group_offset=offset_array)
    # a61 = x[61, :]
    # a_sf_61 = x_sf[:, 61].repeat_interleave(128)
    # print(f"limin: a61 = {a61.shape}")
    # print(f"limin: a_sf_61 = {a_sf_61.shape}")
    # # [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536]
    # b742 = w[6, 742, :]
    # b_sf_742 = w_sf[6, 5, :].repeat_interleave(128)
    # print(f"limin: b742 = {b742.shape}")
    # print(f"limin: b_sf_742 = {b_sf_742.shape}")
    # a_scale_out = torch.mul(a61.to(torch.float32), a_sf_61)
    # b_scale_out = torch.mul(b742.to(torch.float32), b_sf_742)
    # # c61742 = torch.dot(a_scale_out, b_scale_out)
    # o1 = torch.dot(a_scale_out[0:128], b_scale_out[0:128])
    # o2 = torch.dot(a_scale_out[128:256], b_scale_out[128:256])
    # o3 = torch.dot(a_scale_out[256:384], b_scale_out[256:384])
    # o4 = torch.dot(a_scale_out[384:512], b_scale_out[384:512])
    # o5 = torch.dot(a_scale_out[512:640], b_scale_out[512:640])
    # o6 = torch.dot(a_scale_out[640:768], b_scale_out[640:768])
    # o7 = torch.dot(a_scale_out[768:896], b_scale_out[768:896])
    # o8 = torch.dot(a_scale_out[896:1024], b_scale_out[896:1024])
    # o9 = torch.dot(a_scale_out[1024:1152], b_scale_out[1024:1152])
    # o10 = torch.dot(a_scale_out[1152:1280], b_scale_out[1152:1280])
    # o11 = torch.dot(a_scale_out[1280:1408], b_scale_out[1280:1408])
    # o12 = torch.dot(a_scale_out[1408:1536], b_scale_out[1408:1536])
    # o13 = torch.dot(a_scale_out[1536:1664], b_scale_out[1536:1664])
    # o14 = torch.dot(a_scale_out[1664:1792], b_scale_out[1664:1792])
    # o15 = torch.dot(a_scale_out[1792:1920], b_scale_out[1792:1920])
    # o16 = torch.dot(a_scale_out[1920:2048], b_scale_out[1920:2048])
    # o17 = torch.dot(a_scale_out[2048:2176], b_scale_out[2048:2176])
    # o18 = torch.dot(a_scale_out[2176:2304], b_scale_out[2176:2304])
    # o19 = torch.dot(a_scale_out[2304:2432], b_scale_out[2304:2432])
    # o20 = torch.dot(a_scale_out[2432:2560], b_scale_out[2432:2560])
    # c61742 = o1 + o2 + o3 + o4 + o5 + o6 + o7 + o8 + o9 + o10 + o11 + o12 + o13 + o14 + o15 + o16 + o17 + o18 + o19 + o20
    # print(f"limin: c61742 = {c61742.to(torch.bfloat16)}")
    # print(f"limin: h1[61][742] = {h1[61][742]}")
    # print(f"limin: h2[61][742] = {h2[61][742]}")
    # a406_588 = x[406, :]
    # a_sf_406_588 = x_sf[:, 406].repeat_interleave(128)
    # b_588 = w[36, 588, :]
    # b_sf_588 = w_sf[36, 4, :].repeat_interleave(128)
    # print(f"limin: a406_588 = {a406_588.shape}")
    # print(f"limin: b_sf_588 = {b_sf_588.shape}")
    # a_scale_out = torch.mul(a406_588.to(torch.float32), a_sf_406_588)
    # b_scale_out = torch.mul(b_588.to(torch.float32), b_sf_588)
    # o1 = torch.dot(a_scale_out, b_scale_out)
    # # print(f"limin: o_406_588 = {o1.to(torch.bfloat16)}")
    # # print(f"limin: h1[406][588] = {h1[406][588]}")
    # # print(f"limin: h2[406][588] = {h2[406][588]}")
    # # print(f"limin: h1[3][1509] = {h1[3][1509]}")
    # # print(f"limin: h2[3][1509] = {h2[3][1509]}")
    torch.set_printoptions(precision=4, sci_mode=False)
    print(f"limin: ref = {h1}")
    print(f"limin: actual = {h2}")
    print(
        f"limin: torch.allclose(h2.cpu(), h1.cpu(), atol=1e-1) = {torch.allclose(h2.cpu(), h1.cpu(), atol=1e-1)}"
    )
    torch.testing.assert_close(h1.cpu(), h2.cpu(), atol=0.1, rtol=1e-03)


if __name__ == "__main__":
    # test_pad_and_remove_padding()
    # test_filter_and_remap_sorted_experts()

    # for expect_m in [128, 256, 512]:
    #     for num_experts in [4, 8, 16, 32, 64, 72]:
    #         for k in [128, 256]:
    #             for n in [64, 128, 256, 512]:
    #                 print(f"limin: expect_m = {expect_m}, num_experts = {num_experts}, k = {k}, n = {n}")
    #                 test_cute_dsl_group_gemm(num_experts = num_experts, k = k, n = n, expect_m = expect_m)

    # Error:
    # test_cute_dsl_group_gemm(num_experts = 72, k = 512, n = 512, expect_m = 128)
    # test_cute_dsl_group_gemm(num_experts = 72, k = 7168, n = 4096, expect_m = 128)
    # test_cute_dsl_group_gemm_input_from_file()
    test_cute_dsl_group_gemm(72, 1536, 2560, 1024)
