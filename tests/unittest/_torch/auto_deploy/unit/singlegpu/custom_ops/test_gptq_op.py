import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401


def pack_gptq_qweight_from_u4(U4_nk: torch.Tensor) -> torch.Tensor:
    """
    GPTQ: pack along K, 8 nibbles per int32.
    U4_nk: [N,K] uint8 in [0..15]
    -> qweight: [K/8, N] int32
    """
    assert U4_nk.dtype == torch.uint8 and U4_nk.dim() == 2
    N, K = U4_nk.shape
    assert K % 8 == 0
    shifts = torch.arange(8, dtype=torch.int32).view(1, 8, 1) * 4  # [1,8,1]
    U4_kn = U4_nk.T.contiguous()  # [K, N] u8
    U4_blocks = U4_kn.view(K // 8, 8, N).to(torch.int32)  # [K/8,8,N]
    qweight = torch.sum(U4_blocks << shifts, dim=1)  # [K/8, N] i32
    return qweight


def pack_qzeros_all_8(G: int, N: int) -> torch.Tensor:
    """
    Build qzeros: [G, N/8] int32 such that each unpacked nibble == 8.
    Each int32 holds 8 nibbles; signed int32 value -0x77777778 has the same
    bit pattern as 0x88888888 (unsigned).
    """
    assert N % 8 == 0
    val = torch.tensor(-0x77777778, dtype=torch.int32)  # == -2004318072
    return val.repeat(G, N // 8)  # [G, N/8]


def pack_uint8_from_Qs_signed(Qs_nk: torch.Tensor) -> torch.Tensor:
    """
    ModelOpt: pack along N, 2 nibbles per byte from signed int4 Qs in [-8..7].
    Qs_nk: [N,K] int8
    -> packed: [N/2, K] uint8 (low nibble = even row, high nibble = odd row)
    """
    assert Qs_nk.dtype == torch.int8 and Qs_nk.dim() == 2
    N, K = Qs_nk.shape
    assert N % 2 == 0

    # map signed -> nibble (two's complement)
    def to_u4(x: torch.Tensor) -> torch.Tensor:
        x16 = x.to(torch.int16)
        u = torch.where(x16 >= 0, x16, x16 + 16).to(torch.uint8)  # [N,K] in 0..15
        return u

    even_u4 = to_u4(Qs_nk[0::2, :])  # [N/2, K] u8
    odd_u4 = to_u4(Qs_nk[1::2, :])  # [N/2, K] u8
    return (even_u4 | (odd_u4 << 4)).contiguous().to(torch.uint8)  # [N/2, K]


def gptq_unpack_unsigned_u4_KN(
    qweight: torch.Tensor, wf_unsqueeze_neg_one: torch.Tensor
) -> torch.Tensor:
    """
    Mirror the custom op's unpack (for the weight path): returns unsigned nibbles [K,N] u8.
    """
    pack_factor = 8
    w = torch.bitwise_right_shift(
        qweight.unsqueeze(1).expand(-1, pack_factor, -1),  # [K/8,8,N]
        wf_unsqueeze_neg_one.to(qweight.dtype),  # [1,8,1]
    ).to(torch.int16)
    w = (w & 15).to(torch.uint8).reshape(-1, qweight.shape[1])  # [K,N] u8
    return w


def modelopt_unpack_Qs_signed_NK(weight_packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack ModelOpt packed bytes back to signed int4 in [-8..7], [N,K] int8.
    """
    pw = weight_packed.T.contiguous()  # [K, N/2] u8
    low = (pw & 0x0F).to(torch.int16)  # [K, N/2]
    high = ((pw >> 4) & 0x0F).to(torch.int16)  # [K, N/2]
    low_s = torch.where(low >= 8, low - 16, low).to(torch.int8)  # [-8..7]
    high_s = torch.where(high >= 8, high - 16, high).to(torch.int8)  # [-8..7]
    rebuilt = torch.stack([low_s, high_s], dim=-1)  # [K, N/2, 2] i8
    int8_T = rebuilt.reshape(pw.shape[0], -1)  # [K, N] i8
    return int8_T.T.contiguous()  # [N, K] i8


def gptq_weight_and_out(input_x, qweight, qzeros, scales_gn, g_idx, wf_zero, wf_neg1):
    """
    Exact math that your custom op implements:
      weights = scales[g_idx] * (unpacked_u4 - unpacked_qzeros[g_idx]).to(input.dtype)   # [K,N]
      out = input @ weights
    """
    # unpack unsigned nibbles [K,N]
    u4_kn = gptq_unpack_unsigned_u4_KN(qweight, wf_neg1).to(torch.int16)  # [K,N]
    # unpack qzeros to nibbles [G,N] -> broadcast with g_idx
    zeros = (
        torch.bitwise_right_shift(
            qzeros.unsqueeze(2).expand(-1, -1, 8),  # [G, N/8, 8]
            wf_zero.to(qzeros.dtype),  # [1,1,8]
        ).to(torch.int16)
        & 15
    )
    zeros_gn = zeros.reshape(scales_gn.shape)  # [G,N]
    z_kn = zeros_gn[g_idx.long()]  # [K,N] int16

    scale_kn = scales_gn[g_idx.long()].to(torch.float32)  # [K,N]
    W_kn = scale_kn * (u4_kn - z_kn).to(torch.float32)  # [K,N]
    y = input_x.to(torch.float32) @ W_kn
    return W_kn, y


def modelopt_weight_and_out(input_x, weight_packed, weight_scale_ng):
    Qs_nk = modelopt_unpack_Qs_signed_NK(weight_packed).to(torch.float32)  # [N,K]
    S_nk = weight_scale_ng.repeat_interleave(128, dim=1).to(torch.float32)  # [N,K]
    W_kn = (Qs_nk * S_nk).T.contiguous()  # [K,N]
    y = input_x.to(torch.float32) @ W_kn
    return W_kn, y


@pytest.mark.parametrize("N,K,BLOCK_SIZE", [(896, 4864, 128)])
def test_gptq_vs_modelopt_qzeros_8_match(N, K, BLOCK_SIZE):
    torch.manual_seed(0)
    G = K // BLOCK_SIZE
    assert K % 8 == 0 and K % BLOCK_SIZE == 0 and N % 2 == 0

    # Ground-truth signed int4 weights Q_s in [-8..7]
    Qs_nk = torch.randint(-8, 8, (N, K), dtype=torch.int8)

    # Convert to codebooks for each path
    U4_gptq = (Qs_nk.to(torch.int16) + 8).to(torch.uint8)  # [N,K] 0..15
    weight_quantized = pack_uint8_from_Qs_signed(Qs_nk)  # [N/2, K] u8

    # Pack GPTQ qweight and qzeros (all nibbles = 8)
    qweight_gptq = pack_gptq_qweight_from_u4(U4_gptq)  # [K/8, N] i32
    qzeros = pack_qzeros_all_8(G, N)  # [G, N/8] i32

    # Scales: GPTQ stores [G,N], ModelOpt stores [N,G] (transpose)
    scales_gn = torch.rand(G, N, dtype=torch.float32) * 2.0  # [G,N]
    weight_scale_ng = scales_gn.T.contiguous()  # [N,G]

    # Index & shifts
    g_idx = torch.arange(K, dtype=torch.int32) // BLOCK_SIZE  # [K]
    wf = torch.arange(8, dtype=torch.int32) * 4
    wf_zero = wf.view(1, 1, 8)  # [1,1,8]
    wf_neg1 = wf.view(1, 8, 1)  # [1,8,1]

    x = torch.randn(3, K, dtype=torch.float32)

    Wg, yg = gptq_weight_and_out(x, qweight_gptq, qzeros, scales_gn, g_idx, wf_zero, wf_neg1)
    Wm, ym = modelopt_weight_and_out(x, weight_quantized, weight_scale_ng)

    torch.testing.assert_close(Wg, Wm, rtol=0, atol=0)
    torch.testing.assert_close(yg, ym, rtol=0, atol=0)

    bias = None
    pre_scale = torch.tensor(1.0, dtype=torch.float32)
    input_scale_list = [pre_scale]
    weight_scale_list = [weight_scale_ng]
    input_zp_list, weight_zp_list = [torch.tensor(0)], [torch.tensor(0)]

    y_gptq = torch.ops.auto_deploy.torch_fake_quant_int4_gptq_linear(
        x,
        qweight_gptq,
        None,  # bias
        [],  # input_scale
        [scales_gn],  # weight_scale
        [],  # input_zp
        [qzeros],  # weight_zp
    )
    y_mo = torch.ops.auto_deploy.torch_fake_quant_int4_linear(
        x,
        weight_quantized,
        bias,
        input_scale_list,
        weight_scale_list,
        input_zp_list,
        weight_zp_list,
    )

    # small mismatch ≈ 5/2048, likely from the GEMM calculation
    torch.testing.assert_close(y_gptq, y_mo, rtol=0, atol=3e-3)
